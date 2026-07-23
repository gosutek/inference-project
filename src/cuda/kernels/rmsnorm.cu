#include "cuda_helpers.cuh"
#include "helpers.h"
#include "rmsnorm.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__global__ void k_rmsnorm(
	bf16* const __restrict__ input_embeddings,
	const u32 dim,
	const bf16* const __restrict__ norm_weights)
{
	extern __shared__ f32 smem[];
	// note the cast to the higher precision
	f32 acc = 0.0f;

	for (u32 i = threadIdx.x; i < dim; i += blockDim.x) {
		const f32 a = (f32)_d_dn_rm_get(input_embeddings, dim, blockIdx.x, i);
		acc += a * a;
	}
	__syncwarp();

	for (u32 i = _CU_CONST_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CU_CONST_WARP_SIZE);
	}

	const u32 lane_id = MOD_POW2(threadIdx.x, _CU_CONST_WARP_SIZE);
	const u32 warp_id = threadIdx.x / _CU_CONST_WARP_SIZE;

	if (lane_id == 0) {
		smem[warp_id] = acc;
	}
	__syncthreads();

	const u32 warp_cnt = blockDim.x / _CU_CONST_WARP_SIZE;

	if (warp_id == 0 && lane_id < warp_cnt) {
		f32 acc = smem[lane_id];

		const u32 mask = LOWER_BITS_MASK(warp_cnt);

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
		}
		smem[0] = sqrt(acc / dim);
	}
	__syncthreads();

	for (u32 i = threadIdx.x; i < dim; i += blockDim.x) {
		const f32 a = (f32)_d_dn_rm_get(input_embeddings, dim, blockIdx.x, i);
		const f32 g = (f32)_d_dn_rm_get(norm_weights, dim, blockIdx.x, i);

		const bf16 norm_a = (bf16)((a * g) / smem[0]);
		_d_dn_rm_set(input_embeddings, dim, blockIdx.x, i, norm_a);
	}
}
