#include "attention.cuh"
#include "helpers.h"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__device__ bf16 __gemmt(
	const bf16* const __restrict__ a,  // expect rm, smem
	const bf16* const __restrict__ b,  // expect rm, smem
	bf16* const __restrict__ c,        // output rm, smem
	const u32 block_size,
	const u32 m,
	const u32 k,
	const u32 n)
{
	// every block needs to output a 16x16 = 256
	// every block has 512 threads
	// one warp per two rows (or columns doesn't matter) of Sij

	const u32 c_row = threadIdx.x / m;
	const u32 c_col = MOD_POW2(threadIdx.x, m);

	bf16 acc = 0.0;
}

__global__ void __attn_global(
	const bf16* const __restrict__ q,
	const bf16* const __restrict__ k,
	const bf16* const __restrict__ v,
	bf16* const __restrict__ o,
	bf16* const __restrict__ l,
	bf16* const __restrict__ m,
	const u32 n, const u32 dim,
	const u32 bc, const u32 br, const u32 tc, const u32 tr, const u32 q_heads, const u32 kv_heads, const u32 head_dim)
{
	// grid of size (Tr, q heads) = (50, 8)
	extern __shared__ bf16 sk[];
	// block [0] to block [ceil(N / br)] (49) will load K_0, V_0, [Q_0, Q_49], [O_0, O_49]
	// ...

	bf16* const sv = sk + bc * head_dim;
	bf16* const sq = sv + bc * head_dim;
	bf16* const so = sq + br * head_dim;
	bf16* const sl = so + br * head_dim;
	bf16* const sm = sl + br;

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
		__syncthreads();
	}
}
