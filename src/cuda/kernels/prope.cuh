#pragma once

#include <cuda_bf16.h>

#include "helpers.h"

typedef __nv_bfloat16 bf16;

Error_t k_prope(bf16* const x, const u32 input_sequence_len, const u32 model_dim);
