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
