#include "gemm.cuh"

__global__ void k_gemm(
	const bf16* const __restrict__ a,
	const bf16* const __restrict__ b,
	bf16* const __restrict__ c,
	const u32 m,
	const u32 k,
	const u32 n)
{
	for (u32 i = threadIdx.x; i < k; i += blockDim.x) {
		const bf16 val_a = _d_dn_rm_get(a, k, (u32)blockIdx.x, i);
		const bf16 val_b = _d_dn_cm_get(b, k, i, (u32)blockIdx.x);

		_d_dn_rm_set(c, n, (u32)blockIdx.x, i, val_a * val_b);
	}
}
