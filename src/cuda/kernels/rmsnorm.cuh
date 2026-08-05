#pragma once

#include "model.cuh"

Error_t k_rmsnorm(const Model* const model, bf16* const _d_input_embeddings, const u32 input_seq_len);
