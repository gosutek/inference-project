#pragma once

#include "cuda_helpers.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

/**
  * Flash Attention 2022
**/

__global__ void k_att(
	const bf16* const __restrict__ q,
	const bf16* const __restrict__ k,
	const bf16* const __restrict__ v,
	const bf16* const __restrict__ o,
	const bf16* const __restrict__ l,
	const bf16* const __restrict__ m,
	const u64 head_dim,
	const u64 input_seq_len);
