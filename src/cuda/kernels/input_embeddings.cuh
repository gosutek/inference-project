#pragma once

#include "cuda_helpers.cuh"

__global__ void k_fetch_input_embeddings(
	const u32* const __restrict__ input_tokens,
	const u64 input_tokens_len,
	const bf16* const __restrict__ embeddings_table,
	bf16* const __restrict__ input_embeddings);
