#include "cuda_helpers.cuh"

#include "helpers.h"
#include "input_embeddings.cuh"

/*
 * +------------------------------------------------------------------------------+
 * |                                  KERNELS                                     |
 * +------------------------------------------------------------------------------+
 */
__global__ void __input_emb_v1(
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

__global__ void __input_emb(
	const u32* const __restrict__ input_tokens,
	const u64 input_tokens_len,
	const u32 dim,
	const bf16* const __restrict__ embeddings_table,
	bf16* const __restrict__ input_embeddings)
{
	for (u32 i = threadIdx.x; i < dim; i += blockDim.x) {
		const bf16 a = _d_dn_rm_get(embeddings_table, dim, input_tokens[blockIdx.x], i);
		_d_dn_rm_set(input_embeddings, dim, blockIdx.x, i, a);
	}
}

/*
 * +------------------------------------------------------------------------------+
 * |                                  WRAPPER                                     |
 * +------------------------------------------------------------------------------+
 */
Error_t k_input_emb(const Model* const model, u32* const _d_input_tokens, bf16* const _d_input_embeddings, const u32 input_seq_len)
{
	if (!_is_dev_ptr(_d_input_tokens) || !_is_dev_ptr(_d_input_embeddings)) {
		return ErrorInvalidDevPtr;
	}
	dim3 block_size = model->config.hidden_size / 4;  // 1 thread per 4 elements
	dim3 grid_size = input_seq_len;                   // one block per token

	// Consult Max Threads per Block : Because model.config.dim > Max Threads per block for 4070
	__input_emb<<<grid_size, block_size>>>(_d_input_tokens, input_seq_len, model->config.hidden_size, model->weights.token_embedding_table, _d_input_embeddings);

	return Success;
}
