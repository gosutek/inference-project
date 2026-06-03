#pragma once

#include "cuda_helpers.cuh"
#include "helpers.h"

typedef struct Config
{
	u32 dim;              // 1536 for gemma-4-E2B-it
	u32 ffn_dim;          // 6144 for gemma-4-E2B-it
	u32 global_head_dim;  // 512 for gemma-4-E2B-it
	u32 n_heads;          // 8 for gemma-4-E2B-it
	u32 vocab_size;       // 262,144 tokens for gemma-4-E2B-it
	u32 n_layers;         // 35 layers for gemma-4-E2B-it
} Config;

typedef struct Weights
{
	// Embeddings
	bf16* token_embedding_table;  // [vocab_size, dim]

	/**
   * Let's assume that, for example, the tokenizer split the input string into 1 token
   * We index `token_embedding_table` by its token id
   * Hence, we are left with a [1, 1536] matrix (a vector)
   * this needs to be projected to [1, 4096]
   * such that it can pass through the multi-head attention layer of the model which uses 8 heads of size 512 (8 * 512 = 4096)
   * the multiplication will also 'infuse' the weights onto the vector.
   * that is how we derive Q = X * W_Q^T, here X is of size [1, 1536], W_Q^T is of size [1536, 4096] and Q is [1, 4096]
  **/

	// f32* token_embedding_table_per_layer; // TODO: For later

	// RMS
	bf16* rms_input;
	// f32* rms_q;  // TODO: For later
	// f32* rms_k; // TODO: For later
	// f32* rms_post_attn; // TODO: For later
	// f32* rms_pre_ffn; // TODO: For later
	// f32* rms_post_ffn; // TODO: For later

	// Attention
	// 35 pointers, on for each layer, each pointing to a bf16 flat matrix in device memory of size [1536, 4096] (already transposed)
	bf16** wq;
	bf16** wk;
	bf16** wv;
	bf16** wo;

	// FFN
	// f32* mlp_gate; // TODO: For later
	// f32* mlp_up; // TODO: For later
	// f32* mlp_down; // TODO: For later

	// TODO: PLE Block
	// ...
	// ...
	// ...

} Weights;

typedef struct Model
{
	Config  config;
	Weights weights;
	i32     fd;
	f32*    data;  // NOTE: Should be bf16? mmap ptr
	u64     file_bsize;
} Model;
