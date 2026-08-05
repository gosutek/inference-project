#pragma once

#include "model.cuh"

Error_t k_input_emb(const Model* const model, u32* const _d_input_tokens, bf16* const _d_input_embeddings, const u32 input_seq_len);
