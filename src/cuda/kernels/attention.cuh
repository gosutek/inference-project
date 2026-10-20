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

__global__ void __attn_global(
	const bf16* const __restrict__ q,
	const bf16* const __restrict__ k,
	const bf16* const __restrict__ v,
	bf16* const __restrict__ o,
	bf16* const __restrict__ l,
	bf16* const __restrict__ m,
	const u32 n, const u32 dim,  // dim is unused
	const u32 bc, const u32 br, const u32 tc, const u32 tr, const u32 q_heads, const u32 kv_heads, const u32 head_dim);
