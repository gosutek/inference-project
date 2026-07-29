#include "gemm.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

/**
 * c = a * b
**/
__global__ void k_gemm(
	const bf16* const __restrict__ a,  // expect rm
	const bf16* const __restrict__ b,  // expect rm
	bf16* const __restrict__ c,        // output rm
	const u32 m,
	const u32 k,
	const u32 n)
{
	const u32 row = threadIdx.y + blockIdx.y * blockDim.y;

	/**
   * consecutive threadIdx.x's are assigned to the same warp (up to 32)
   * since we treat that as the column we are essentially distributing
   * a warp across the columns of C
   * vvvvvvvvvvvvvvvvvvvvvvvvv
   **/
	const u32 col = threadIdx.x + blockIdx.x * blockDim.x;

	bf16 acc = 0;
	for (u32 i = 0; i < k; ++i) {
		acc += _d_dn_rm_get(a, k, row, i) * _d_dn_rm_get(b, n, i, col);
	}
	_d_dn_rm_set(c, n, row, col, acc);
}

/**
 * c = a * b^T
**/
__global__ void k_gemmt(
	const bf16* const __restrict__ a,  // expect rm
	const bf16* const __restrict__ b,  // expect rm
	bf16* const __restrict__ c,        // output rm
	const u32 block_size,
	const u32 m,
	const u32 k,
	const u32 n)
{
	// NOTE: Check the n_cols of tranposed b
	extern __shared__ bf16 smem_a[];
	bf16* const            smem_b = (bf16*)((u8*)smem_a + block_size * block_size * sizeof *smem_a);

	const u32 c_col = threadIdx.x + blockIdx.x * blockDim.x;  // (0 + 33 * 32) = 1056
	const u32 c_row = threadIdx.y + blockIdx.y * blockDim.y;  // (10)

	bf16 acc = 0.0;
	for (u32 offset = 0; offset < k; offset += block_size) {
		bf16 load_val = _d_dn_rm_get(a, k, c_row, threadIdx.x + offset);
		_d_dn_rm_set(smem_a, block_size, threadIdx.y, threadIdx.x, load_val);

		load_val = _d_dn_rm_get(b, k, c_col, threadIdx.y + offset);
		_d_dn_rm_set(smem_b, block_size, threadIdx.x, threadIdx.y, load_val);  // transpose
		__syncthreads();

		for (u32 i = 0; i < block_size; ++i) {
			acc += _d_dn_rm_get(smem_a, block_size, threadIdx.y, i) * _d_dn_rm_get(smem_b, block_size, i, threadIdx.x);
		}
		__syncthreads();
	}

	_d_dn_rm_set(c, n, c_row, c_col, acc);
}
