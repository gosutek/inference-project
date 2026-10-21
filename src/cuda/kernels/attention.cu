#include "attention.cuh"
#include "cuda_helpers.cuh"
#include "helpers.h"
#include <__clang_cuda_builtin_vars.h>

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__device__ bf16 __attn_gemmt(
	const bf16* const __restrict__ a,  // expect rm, smem
	const bf16* const __restrict__ b,  // expect rm, smem
	bf16* const __restrict__ c,        // output rm, smem
	const u32 m,
	const u32 k,
	const u32 n,
	const u32 c_row, const u32 c_col)
{
	// every block needs to output a 16x16 = 256
	// every block has 512 threads

	// TODO: Add a check that n is a power of 2

	bf16 acc = 0.0;
	for (u32 i = 0; i < k; ++i) {
		acc += _d_dn_rm_get(a, k, c_row, i) * _d_dn_rm_get(b, k, c_col, i);
	}
	return acc;
}

__device__ bf16 __attn_rowmax(
	const bf16* const ss,
	const u32 br, const u32 bc,
	const u32 c_row, const u32 c_col,
	const u32 warp_id, const u32 lane_id)
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
		local_max = MAX(local_max, __shfl_xor_sync(0xffffffff, local_max, offset, _CONSTANTS_WARP_SIZE_HALF));
	}

	if (lane_id == 0 || lane_id == _CONSTANTS_WARP_SIZE_HALF) {
		return local_max;
	}
	return 0;  // NOTE: IS this correct? It's supposed to return a bf16
}

__device__ void __attn_correction_factor(
	const bf16* const ss,
	const bf16* const sm,
	bf16* const       sp,
	const u32 br, const u32 bc,
	const u32 c_row, const u32 c_col,
	const u32 warp_id, const u32 lane_id)
{
	// NOTE: Broadcast approach
	// if (lane_id == 0 || lane_id == _CONSTANTS_WARP_SIZE_HALF) {
	// 	rowmax = _d_dn_rm_get(sm, bc, 0, c_row);
	// }
	// rowmax = __shfl_sync(0xffffffff, rowmax, 0, _CONSTANTS_WARP_SIZE_HALF);  // broadcast to all threads in the two warp partitions

	// NOTE: Copy from the buffer approach
	const f32 rowmax = __bfloat162float(_d_dn_rm_get(sm, bc, 0, c_row));
	const f32 score = __bfloat162float(_d_dn_rm_get(ss, bc, c_row, c_col));
	const f32 p = expf(score - rowmax);
	_d_dn_rm_set(sp, bc, c_row, c_col, __float2bfloat16(p));
}

__device__ bf16 __attn_rowsum(
	const bf16* const sp,
	const u32 br, const u32 bc,
	const u32 c_row, const u32 c_col,
	const u32 warp_id, const u32 lane_id)
{
	f32 acc = 0.0f;
	f32 val = __bfloat162float(_d_dn_rm_get(sp, br, c_row, c_col));
	for (u32 offset = _CONSTANTS_WARP_SIZE_HALF / 2; offset > 0; offset /= 2) {
		acc += __shfl_xor_sync(0xffffffff, val, offset, _CONSTANTS_WARP_SIZE_HALF);
	}

	if (lane_id == 0 || lane_id == _CONSTANTS_WARP_SIZE_HALF) {
		return __float2bfloat16(acc);
	}
	return 0;  // NOTE: IS this correct? It's supposed to return a bf16
}

__global__ void __attn_global(
	const bf16* const __restrict__ q,
	const bf16* const __restrict__ k,
	const bf16* const __restrict__ v,
	bf16* const __restrict__ o,
	bf16* const __restrict__ l,
	bf16* const __restrict__ m,
	const u32 n,
	const u32 bc, const u32 br, const u32 tc, const u32 tr, const u32 q_heads, const u32 kv_heads, const u32 head_dim)
{
	// grid of size (Tr, q heads) = (50, 8)
	extern __shared__ bf16 sk[];
	// block [0] to block [ceil(N / br)] (49) will load K_0, V_0, [Q_0, Q_49], [O_0, O_49]
	// ...

	const u32 c_row = threadIdx.x / br;
	const u32 c_col = MOD_POW2(threadIdx.x, bc);

	const u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;
	const u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);

	bf16* const sv = sk + bc * head_dim;
	bf16* const sq = sv + bc * head_dim;
	bf16* const so = sq + br * head_dim;
	bf16* const sl = so + br * head_dim;
	bf16* const sm = sl + br;
	bf16* const ss = sm + br;
	bf16* const sp = ss + br * bc;

	for (u32 r = blockIdx.x * br; r < (blockIdx.x + 1) * br; ++r) {                              // iterating over the 16 rows of the 16x512 FA tile
		bf16 val = _d_dn_rm_get(q, q_heads * head_dim, r, blockIdx.y * head_dim + threadIdx.x);  // blockIdx.y here plays the role of a head
		_d_dn_rm_set(sq, head_dim, r - blockIdx.x * br, threadIdx.x, val);
		// the tile we loaded is local to the block and starts from 0 to 511
	}

	for (u32 r = blockIdx.x * br; r < (blockIdx.x + 1) * br; ++r) {
		bf16 val = _d_dn_rm_get(o, q_heads * head_dim, r, blockIdx.y * head_dim + threadIdx.x);
		_d_dn_rm_set(so, head_dim, r - blockIdx.x * br, threadIdx.x, val);
	}

	if (threadIdx.x < _CU_CONST_WARP_SIZE / 2) {
		sl[threadIdx.x] = l[blockIdx.x * br + threadIdx.x];
	} else if (threadIdx.x >= _CU_CONST_WARP_SIZE && threadIdx.x < _CU_CONST_WARP_SIZE + _CU_CONST_WARP_SIZE / 2) {
		sm[threadIdx.x % (_CU_CONST_WARP_SIZE / 2)] = m[(blockIdx.x * br) + (threadIdx.x % (_CU_CONST_WARP_SIZE / 2))];
	}

	__syncthreads();
	// so at *this* point, every block of the 50x8 we have has loaded their respective tiles into shared memory.
	// Now we need to loop of every KV tile

	for (u32 i_tile = 0; i_tile < tc; ++i_tile) {
		for (u32 tile_row = 0; tile_row < br; ++tile_row) {
			bf16 val = _d_dn_rm_get(k, kv_heads * head_dim, i_tile * br + tile_row, threadIdx.x);
			_d_dn_rm_set(sk, kv_heads * head_dim, tile_row, threadIdx.x, val);

			val = _d_dn_rm_get(v, kv_heads * head_dim, i_tile * br + tile_row, threadIdx.x);
			_d_dn_rm_set(sv, kv_heads * head_dim, tile_row, threadIdx.x, val);
		}

		// NOTE: Don't know if I can avoid this, i.e. deactivating half the threads of the block.
		if (threadIdx.x < br * bc) {
			__attn_gemmt(sq, sk, ss, br, head_dim, bc, c_row, c_col);
			__syncthreads();
			const bf16 rowmax = __attn_rowmax(ss, br, bc, c_row, c_col, warp_id, lane_id);
			__syncwarp();
			__attn_correction_factor(ss, sm, sp, br, bc, c_row, c_col, warp_id, lane_id);
			__syncwarp();
			const bf16 rowsum = __attn_rowsum(sp, br, bc, c_row, c_col, warp_id, lane_id);
			__syncwarp();

			if (lane_id == 0 || lane_id == _CONSTANTS_WARP_SIZE_HALF) {
				const bf16 hbm_m = _d_dn_rm_get(sm, n, 0, c_row);
				const bf16 new_m = MAX(hbm_m, rowmax);

				const bf16 hbm_l = _d_dn_rm_get(sl, n, 0, c_row);

				const f32 new_m_f32 = __bfloat162float(new_m);
				const f32 exp_1 = expf(__bfloat162float(hbm_m) - new_m_f32);
				const f32 exp_2 = expf(__bfloat162float(rowmax) - new_m_f32);

				const f32 term_1 = exp_1 * __bfloat162float(hbm_l);
				const f32 term_2 = exp_2 * __bfloat162float(rowsum);

				const bf16 new_l = __float2bfloat16(term_1 + term_2);
			}
		}
	}
}
