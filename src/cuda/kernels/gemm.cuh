#pragma once

#include "cuda_helpers.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__global__ void k_gemm(
	const bf16* const __restrict__ a,
	const bf16* const __restrict__ b,
	const bf16* const __restrict__ c,
	const u32 m,
	const u32 k,
	const u32 n);
