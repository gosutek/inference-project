#pragma once

#include "cuda_helpers.cuh"

constexpr u32 FA_BR = 16;
constexpr u32 FA_BC = 16;

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
	const u32 n,
	const u32 tc, const u32 tr, const u32 q_heads, const u32 kv_heads, const u32 head_dim);
