#include "attention.cuh"
#include "cuda_helpers.cuh"
#include "helpers.h"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__device__ void __attn_update_o(
        bf16* const __restrict__ o,
        const bf16* const __restrict__ p,
        const bf16* const __restrict__ v,
        const f32* const __restrict__ scalar_1,
        const f32* const __restrict__ scalar_2,
        const u32 head_dim)
{
    // every thread block needs to output a 16x512
    // every thread block has 512 threads
    // every *thread* will calculate 16 elements of the resulting matrix:
    // thread t owns column t of O_i (expects blockDim.x == head_dim)
    //
    // O_i = diag(scalar_1) * O_i + diag(scalar_2) * (P~_ij * V_j)

    // V_j[:, col] is reused by all FA_BR rows -> load it once into registers.
    // FA_BC is constexpr and the loops are unrolled, so this stays in registers.
    f32 v_col[FA_BC];
#pragma unroll
    for (u32 j = 0; j < FA_BC; ++j) {
        v_col[j] = __bfloat162float(_d_dn_rm_get(v, head_dim, j, threadIdx.x));
    }

#pragma unroll
    for (u32 r = 0; r < FA_BR; ++r) {
        f32 acc = 0.0f;
#pragma unroll
        for (u32 j = 0; j < FA_BC; ++j) {
            // p[r][j] is the same address for every thread in the warp -> broadcast
            acc += __bfloat162float(_d_dn_rm_get(p, FA_BC, r, j)) * v_col[j];
        }

        // each so[r][col] is owned by exactly one thread, so updating in place is safe
        const f32 o_old = __bfloat162float(_d_dn_rm_get(o, head_dim, r, threadIdx.x));
        _d_dn_rm_set(o, head_dim, r, threadIdx.x, __float2bfloat16(scalar_1[r] * o_old + scalar_2[r] * acc));
    }
}

__device__ void __attn_gemmt(
    const bf16* const __restrict__ a,  // expect rm, smem
    const bf16* const __restrict__ b,  // expect rm, smem
    bf16* const __restrict__ c,        // output rm, smem
    const u32 m,
    const u32 k,
    const u32 n,
    const u32 c_row, const u32 c_col)
{
    // every thread block needs to output a 16x16 = 256
    // every thread block has 512 threads

    // TODO: Add a check that n is a power of 2

    f32 acc = 0.0f;
    for (u32 i = 0; i < k; ++i) {
        acc += __bfloat162float(_d_dn_rm_get(a, k, c_row, i) * _d_dn_rm_get(b, k, c_col, i));
    }
    _d_dn_rm_set(c, n, c_row, c_col, acc);
}

__device__ bf16 __attn_rowmax(
    const bf16* const ss,
    const u32 br, const u32 bc,
    const u32 c_row, const u32 c_col)
{
    // one warp per two rows
    // I still need 16 * 16 threads

    // the goal is to:
    // - make lane_id ~ 0 of warp_id ~ 0 hold the rowmax of row 0
    // - make lane_id ~ 16 of warp_id ~ 0 hold the rowmax of row 1
    // - make lane_id ~ 0 of warp_id ~ 1 hold the rowmax of row 2
    // - make lane_id ~ 16 of warp_id ~ 1 hold the rowmax of row 2

    bf16 local_max = _d_dn_rm_get(ss, bc, c_row, c_col);
    for (u32 offset = _CONSTANTS_WARP_SIZE_HALF / 2; offset > 0; offset /= 2) {
        bf16 tmp = __shfl_xor_sync(0xffffffff, local_max, offset, _CONSTANTS_WARP_SIZE_HALF);
        local_max = MAX(local_max, tmp);
    }
    // NOTE: Remember that since in butterfly reduction, the value are exchanged
    // and due to the nature of the MAX operation, *all* lanes will be left with the row max.

    return local_max;
}

__device__ void __attn_correction_factor(
    const bf16* const ss,
    const bf16 rowmax,
    bf16* const       sp,
    const u32 br, const u32 bc,
    const u32 c_row, const u32 c_col)
{
    const f32 score = __bfloat162float(_d_dn_rm_get(ss, bc, c_row, c_col));
    const f32 p = expf(score - __bfloat162float(rowmax));
    _d_dn_rm_set(sp, bc, c_row, c_col, __float2bfloat16(p));
}

__device__ bf16 __attn_rowsum(
    const bf16* const sp,
    const u32 br, const u32 bc,
    const u32 c_row, const u32 c_col)
{
    f32 val = __bfloat162float(_d_dn_rm_get(sp, br, c_row, c_col));
    for (u32 offset = _CONSTANTS_WARP_SIZE_HALF / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset, _CONSTANTS_WARP_SIZE_HALF);
    }

    return __float2bfloat16(val);
}

__global__ void __attn_global(
    const bf16* const __restrict__ q,
    const bf16* const __restrict__ k,
    const bf16* const __restrict__ v,
    bf16* const __restrict__ o,
    const u32 n,
    const u32 tc, const u32 tr, const u32 q_heads, const u32 kv_heads, const u32 head_dim)
{
    // grid of size (Tr, q heads), block of head_dim threads.
    // Each block owns one Q tile (FA_BR rows of one head) and loops over all KV tiles.
    // O_i, l_i, m_i live in smem for the whole loop and only O_i is written back, once.
    extern __shared__ bf16 sk[];

    const u32 c_row = threadIdx.x / FA_BC;
    const u32 c_col = MOD_POW2(threadIdx.x, FA_BC);

    const u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);

    const u32 head_col0 = blockIdx.y * head_dim;  // blockIdx.y plays the role of a head

    bf16* const sv = sk + FA_BC * head_dim;
    bf16* const sq = sv + FA_BC * head_dim;
    bf16* const so = sq + FA_BR * head_dim;
    f32* const  sl = (f32*)(so + FA_BR * head_dim);
    f32* const  sm = sl + FA_BR;
    f32* const  s_scalar1 = sm + FA_BR;
    f32* const  s_scalar2 = s_scalar1 + FA_BR;
    bf16* const ss = (bf16*)(s_scalar2 + FA_BR);
    bf16* const sp = ss + FA_BR * FA_BC;

    // Load Q_i, init O_i = 0. Rows past n (last tile when n % FA_BR != 0) are zero-filled and never stored.
    for (u32 r = 0; r < FA_BR; ++r) {
        const u32 row = blockIdx.x * FA_BR + r;
        const bf16 val = row < n ? _d_dn_rm_get(q, q_heads * head_dim, row, head_col0 + threadIdx.x) : __float2bfloat16(0.0f);
        _d_dn_rm_set(sq, head_dim, r, threadIdx.x, val);
        _d_dn_rm_set(so, head_dim, r, threadIdx.x, __float2bfloat16(0.0f));
    }

    // l_i = 0, m_i = -inf
    if (threadIdx.x < FA_BR) {
        sl[threadIdx.x] = 0.0f;
        sm[threadIdx.x] = -INFINITY;
    }

    __syncthreads();

    for (u32 i_tile = 0; i_tile < tc; ++i_tile) {
        for (u32 tile_row = 0; tile_row < FA_BC; ++tile_row) {
            const u32 row = i_tile * FA_BC + tile_row;
            const bool in_range = row < n;

            bf16 val = in_range ? _d_dn_rm_get(k, kv_heads * head_dim, row, threadIdx.x) : __float2bfloat16(0.0f);
            _d_dn_rm_set(sk, head_dim, tile_row, threadIdx.x, val);

            val = in_range ? _d_dn_rm_get(v, kv_heads * head_dim, row, threadIdx.x) : __float2bfloat16(0.0f);
            _d_dn_rm_set(sv, head_dim, tile_row, threadIdx.x, val);
        }
        __syncthreads();

        // NOTE: Don't know if I can avoid this, i.e. deactivating half the threads of the block.
        if (threadIdx.x < FA_BR * FA_BC) { // the first br * bc
            __attn_gemmt(sq, sk, ss, FA_BR, head_dim, FA_BC, c_row, c_col); // calculates a br * bc block of the attention scores S, i.e. S_{ij}
            // Keys past n (last KV tile) must not take part in the softmax: S = -inf -> P~ = 0.
            if (i_tile * FA_BC + c_col >= n) {
                _d_dn_rm_set(ss, FA_BC, c_row, c_col, __float2bfloat16(-INFINITY));
            }
            __syncwarp();
            const bf16 rowmax = __attn_rowmax(ss, FA_BR, FA_BC, c_row, c_col);
            __syncwarp();
            __attn_correction_factor(ss, rowmax, sp, FA_BR, FA_BC, c_row, c_col);
            __syncwarp();
            const bf16 rowsum = __attn_rowsum(sp, FA_BR, FA_BC, c_row, c_col);
            __syncwarp();

            // every lane of the 16-lane row group holds rowmax/rowsum; one lane per row writes the stats
            if (lane_id == 0 || lane_id == _CONSTANTS_WARP_SIZE_HALF) {
                const f32 m_old = sm[c_row];
                const f32 l_old = sl[c_row];
                const f32 m_tile = __bfloat162float(rowmax);
                const f32 m_new = fmaxf(m_old, m_tile);

                // e^{m_{i} - m_{i}^{new}}, 0 on the first tile since m_old = -inf
                const f32 exp_1 = expf(m_old - m_new);
                // e^{\tilde{m}_{ij} - m_{i}^{new}}
                const f32 exp_2 = expf(m_tile - m_new);

                const f32 l_new = exp_1 * l_old + exp_2 * __bfloat162float(rowsum);

                // diag(l_i) * e^{m_{i} - m_{i}^{new}} * diag(l_i^{new})^{-1}
                s_scalar1[c_row] = l_old * exp_1 / l_new;
                // e^{\tilde{m}_{ij} - m_{i}^{new}} * diag(l_i^{new})^{-1}
                s_scalar2[c_row] = exp_2 / l_new;
                sm[c_row] = m_new;
                sl[c_row] = l_new;
            }
        }
        __syncthreads();
        __attn_update_o(so, sp, sv, s_scalar1, s_scalar2, head_dim);
        __syncthreads();  // next iteration overwrites sk/sv, and this one's update reads sv/sp
    }

    // O_i is already normalized (scalar_1/2 divide by l_new every step): write it back once.
    for (u32 r = 0; r < FA_BR; ++r) {
        const u32 row = blockIdx.x * FA_BR + r;
        if (row < n) {
            _d_dn_rm_set(o, q_heads * head_dim, row, head_col0 + threadIdx.x, _d_dn_rm_get(so, head_dim, r, threadIdx.x));
        }
    }
}
