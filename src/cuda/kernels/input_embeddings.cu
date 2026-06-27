#include "cuda_helpers.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

/*
  * +------------------------------------------------------------------------------+
  * |                               VERSION 1                                      |
  * +------------------------------------------------------------------------------+
*/

__global__ void k_fetch_input_embeddings(
	const u32* const __restrict__ input_tokens,
	const u64 input_tokens_len,
	const u32 dim,
	const bf16* const __restrict__ embeddings_table,
	bf16* const __restrict__ input_embeddings,
	const u32 stride)
{
	_d_dn_rm_set(input_embeddings, dim, blockIdx.x, threadIdx.x * stride + 0, _d_dn_rm_get(embeddings_table, dim, input_tokens[blockIdx.x], threadIdx.x * stride + 0));
	_d_dn_rm_set(input_embeddings, dim, blockIdx.x, threadIdx.x * stride + 1, _d_dn_rm_get(embeddings_table, dim, input_tokens[blockIdx.x], threadIdx.x * stride + 1));
	_d_dn_rm_set(input_embeddings, dim, blockIdx.x, threadIdx.x * stride + 2, _d_dn_rm_get(embeddings_table, dim, input_tokens[blockIdx.x], threadIdx.x * stride + 2));
	_d_dn_rm_set(input_embeddings, dim, blockIdx.x, threadIdx.x * stride + 3, _d_dn_rm_get(embeddings_table, dim, input_tokens[blockIdx.x], threadIdx.x * stride + 3));
}
