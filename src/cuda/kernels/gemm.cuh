#pragma once

#include "cuda_helpers.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__global__ void k_gemm(
	const bf16* const __restrict__ a,  // expect rm
	const bf16* const __restrict__ b,  // expect rm
	bf16* const __restrict__ c,        // output rm
	const u32 m,
	const u32 k,
	const u32 n);

__global__ void k_gemmt(
	const bf16* const __restrict__ a,  // expect rm
	const bf16* const __restrict__ b,  // expect rm
	bf16* const __restrict__ c,        // output rm
	const u32 m,
	const u32 k,
	const u32 n);
