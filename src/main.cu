#include <assert.h>  // works only in debug
#include <cstdint>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "cJSON.h"

#include "allocator.h"
#include "cuda_allocator.cuh"
#include "cuda_helpers.cuh"
#include "cuda_mem_wrapper.cuh"
#include "helpers.h"
#include "kernels/gemm.cuh"
#include "kernels/input_embeddings.cuh"
#include "kernels/rmsnorm.cuh"
#include "model.cuh"
#include "tokenizer.h"

const u32 MAX_SEQ_LEN = 512;

static Error_t print_dev_buf_u32(ExecCtx* const e_ctx, u32* src, const u64 bsize)
{
	u32*      dst = NULL;
	const u64 size = bsize / sizeof *dst;
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, bsize, (void**)&dst));
	CHECK_ERROR(cu_memcpy_dth(dst, src, bsize));

	const u32 n_iter = size > 5 ? 5 : size;

	printf("|----------------------------------------------------|\n");
	for (u32 i = 0; i < n_iter; ++i) {
		printf("%u\n", dst[i]);
	}
	printf("|----------------------------------------------------|\n");

	arena_host_pop((HostArena*)e_ctx, bsize);
	return Success;
}

static Error_t print_dev_buf_bf16(ExecCtx* const e_ctx, bf16* src, const u64 bsize)
{
	bf16*     dst = NULL;
	const u64 size = bsize / sizeof *dst;
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, bsize, (void**)&dst));
	CHECK_ERROR(cu_memcpy_dth(dst, src, bsize));

	const u32 n_iter = size > 5 ? 5 : size;

	printf("|----------------------------------------------------|\n");
	for (u32 i = 0; i < n_iter; ++i) {
		printf("%.2f\n", (f32)dst[i]);
	}
	printf("|----------------------------------------------------|\n");

	arena_host_pop((HostArena*)e_ctx, bsize);
	return Success;
}

static void print_model_config(const Model* const model)
{
	printf(
		"dim: %u\n"
		"ffn_dim: %u\n"
		"global_head_dim: %u\n"
		"n_heads: %u\n"
		"vocab_size: %u\n"
		"n_layers: %u\n",
		model->config.dim, model->config.ffn_dim, model->config.global_head_dim, model->config.n_heads, model->config.vocab_size, model->config.n_layers);
}

static void print_dev_props()
{
	cudaDeviceProp dev_prop = {};
	CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, 0));

	printf(
		"- Name %s\n- SM count %d\n- Total global memory %lu MB\n- L2 cache size %lu MB\n- cc %d.%d\n"
		"- Shared memory per block %lu KB\n- Shared memory per SM %lu KB\n- Constant memory %lu KB\n- Warp size %d\n"
		"- Max threads per SM %d\n- Max threads per block %d\n- Max block dimensions [%d %d %d]\n- Max grid dimensions [%d %d %d]\n"
		"- Max regs per block %d\n- Max regs per SM %d\n",
		dev_prop.name, dev_prop.multiProcessorCount, (u64)(dev_prop.totalGlobalMem / 1e6), (u64)(dev_prop.l2CacheSize / 1e6),
		dev_prop.major, dev_prop.minor, (u64)(dev_prop.sharedMemPerBlock / 1e3), (u64)(dev_prop.sharedMemPerMultiprocessor / 1e3), (u64)(dev_prop.totalConstMem / 1e3),
		dev_prop.warpSize, dev_prop.maxThreadsPerMultiProcessor, dev_prop.maxThreadsPerBlock, dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2],
		dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2], dev_prop.regsPerBlock, dev_prop.regsPerMultiprocessor);
}

static Error_t correctness_weight_ptr_partition(ExecCtx* const e_ctx, const bf16* const d_ptr, const bf16* const h_ptr, i32 n)
{
	bf16* h_buf = NULL;
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, n * sizeof *d_ptr, (void**)&h_buf));
	cu_memcpy_dth((void*)h_buf, (void*)d_ptr, n * sizeof *d_ptr);

	for (i32 i = 0; i < n; ++i) {
		f32 d_val = __bfloat162float(h_buf[i]);
		f32 h_val = __bfloat162float(h_ptr[i]);
		if (d_val != h_val) {
			arena_host_pop((HostArena*)e_ctx, n * sizeof *d_ptr);
			fprintf(stderr, "correctness_weight_ptr_partition failed [%d] gpu: %.5f cpu: %.5f\n", i, d_val, h_val);
			return ErrorGeneric;
		}
	}
	arena_host_pop((HostArena*)e_ctx, n * sizeof *d_ptr);
	return Success;
}

static Error_t get_file_bsize(const char* filepath, u64* const bsize)
{
	struct stat st;
	if (stat(filepath, &st) == 0) {
		*bsize = (u64)st.st_size;
		return Success;
	}
	return ErrorGeneric;
}

static void model_parse_config(ExecCtx* const e_ctx, Model* const model, const char* model_config_filepath)
{
	FILE* file = fopen(model_config_filepath, "rb");
	if (!file) {
		fprintf(stderr, "couldn't read file %s\n", model_config_filepath);
		exit(EXIT_FAILURE);
	}

	u64 model_config_bsize = 0;
	CHECK_ERROR(get_file_bsize(model_config_filepath, &model_config_bsize));

	char* json_buf = NULL;
	// WARN: Am I popping this?
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, model_config_bsize + 1, (void**)&json_buf));

	if (fread(json_buf, sizeof *json_buf, model_config_bsize, file) != model_config_bsize) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	json_buf[model_config_bsize] = '\0';  // cJSON works on null-terminated strings

	cJSON* model_config_root = cJSON_Parse(json_buf);

	arena_host_pop((HostArena*)e_ctx, model_config_bsize + 1);
	fclose(file);

	cJSON* model_config = model_config_root->child;
	// TODO: Replace this with cJSON_GetObjectItemCaseSensitive
	while (strcmp(model_config->string, "text_config") != 0) {
		model_config = model_config->next;
	}
	model->config.dim = cJSON_GetObjectItem(model_config, "hidden_size")->valueint;
	model->config.ffn_dim = cJSON_GetObjectItem(model_config, "intermediate_size")->valueint;
	model->config.global_head_dim = cJSON_GetObjectItem(model_config, "global_head_dim")->valueint;
	model->config.n_heads = cJSON_GetObjectItem(model_config, "num_attention_heads")->valueint;
	model->config.vocab_size = cJSON_GetObjectItem(model_config, "vocab_size")->valueint;
	model->config.n_layers = cJSON_GetObjectItem(model_config, "num_hidden_layers")->valueint;

	cJSON_Delete(model_config_root);
	return;
}

static void print_model(const Model* const model)
{
	printf(
		"Model Configuration:\n\t\
        - dim: %d\n\t\
        - ffn_dim: %d\n\t\
        - global_head_dim: %d\n\t\
        - n_heads: %d\n\t\
        - vocab_size: %d\n\t\
        - n_layers: %d\n",
		model->config.dim, model->config.ffn_dim, model->config.global_head_dim, model->config.n_heads, model->config.vocab_size, model->config.n_layers);
	return;
}

static Error_t parse_model_header(ExecCtx* const e_ctx, Model* const model, FILE* const file, cJSON** header)
{
	if (!file || (*header)) {
		return ErrorInvalidValue;
	}
	char* json_buf = NULL;
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, model->header_bsize + 1, (void**)&json_buf));
	if (fread(json_buf, sizeof *json_buf, model->header_bsize, file) != model->header_bsize) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	json_buf[model->header_bsize] = '\0';

	*header = cJSON_Parse(json_buf);
	if (!(*header)) {
		return ErrorGeneric;
	}
	arena_host_pop((HostArena*)e_ctx, model->header_bsize + 1);  // free my own json_buf as cJSON allocates its own that we free later on

	return Success;
}

static void model_build(ExecCtx** e_ctx, Model* const model, const char* model_filepath, const char* model_config_filepath)
{
	CHECK_ERROR(get_file_bsize(model_filepath, &model->file_bsize));

	FILE* file = fopen(model_filepath, "rb");
	if (!file) {
		fprintf(stderr, "couldn't load %s\n", model_filepath);
		exit(EXIT_FAILURE);
	}

	if (fread(&model->header_bsize, sizeof model->header_bsize, 1, file) != 1) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	CHECK_ERROR(exec_ctx_create(e_ctx));

	model_parse_config(*e_ctx, model, model_config_filepath);
	const u64 rms_input_ptrs_bsize = model->config.n_layers * sizeof *model->weights.rms_input;  // this is n_layers * size of a bf16 pointer
	const u64 wq_ptrs_bsize = model->config.n_layers * sizeof *model->weights.wq;
	const u64 wk_ptrs_bsize = model->config.n_layers * sizeof *model->weights.wk;
	const u64 wv_ptrs_bsize = model->config.n_layers * sizeof *model->weights.wv;
	const u64 wo_ptrs_bsize = model->config.n_layers * sizeof *model->weights.wo;

	CHECK_ERROR(arena_host_push((HostArena*)(*e_ctx), rms_input_ptrs_bsize, (void**)&(model->weights.rms_input)));
	CHECK_ERROR(arena_host_push((HostArena*)(*e_ctx), wq_ptrs_bsize, (void**)&(model->weights.wq)));
	CHECK_ERROR(arena_host_push((HostArena*)(*e_ctx), wk_ptrs_bsize, (void**)&(model->weights.wk)));
	CHECK_ERROR(arena_host_push((HostArena*)(*e_ctx), wv_ptrs_bsize, (void**)&(model->weights.wv)));
	CHECK_ERROR(arena_host_push((HostArena*)(*e_ctx), wo_ptrs_bsize, (void**)&(model->weights.wo)));

	cJSON* header_root = NULL;
	CHECK_ERROR(parse_model_header(*e_ctx, model, file, &header_root));

	const char* TENSOR_FILTER = "model.language_model.";
	const u64   TENSOR_FILTER_LEN = strlen(TENSOR_FILTER);

	cJSON* first_lm_node = NULL;
	u64    lm_offset_start = UINT64_MAX;
	u64    lm_offset_end = 0;

	cJSON* header = header_root->child;
	for (cJSON* node = header; node != NULL; node = node->next) {
		if (strlen(node->string) < TENSOR_FILTER_LEN || strncmp(node->string, TENSOR_FILTER, TENSOR_FILTER_LEN) != 0) {
			continue;
		}
		if (first_lm_node == NULL) {
			first_lm_node = node;
		}

		cJSON* offsets = cJSON_GetObjectItem(node, "data_offsets");
		u64    start = (u64)cJSON_GetArrayItem(offsets, 0)->valuedouble;
		u64    end = (u64)cJSON_GetArrayItem(offsets, 1)->valuedouble;

		lm_offset_start = MIN(lm_offset_start, start);
		lm_offset_end = MAX(lm_offset_end, end);
	}

	model->model_bsize = lm_offset_end - lm_offset_start;
	const u64 padded_dev_alloc_bsize = model->model_bsize + PADDING_POW2(model->model_bsize, GIB(1));
	CHECK_ERROR(arena_dev_create(&(*e_ctx)->dev_arena, padded_dev_alloc_bsize));

	CHECK_ERROR(arena_dev_push(&(*e_ctx)->dev_arena, model->model_bsize, (void**)&model->data));

	model->fd = fileno(file);
	void* model_mmap = mmap(NULL, model->file_bsize, PROT_READ, MAP_PRIVATE, model->fd, 0);
	if (model_mmap == MAP_FAILED) {
		fprintf(stderr, "failed to mmap safetensor\n");
		exit(EXIT_FAILURE);
	}
	model_mmap = (void*)((u8*)model_mmap + lm_offset_start);
	printf("Tranferring\n");
	cu_memcpy_htd((void*)model->data, model_mmap, model->model_bsize);
	printf("Tranfer complete\n");

	// #ifndef NDEBUG
	// 	i32 dbg_counter = 0; // this fails for different model weights :)
	// #endif

#ifndef NDEBUG
	if (strcmp("model.language_model.embed_tokens.weight", first_lm_node->string) != 0) {
		fprintf(stderr, "unxpected first node of json\n");
		exit(EXIT_FAILURE);
	}
#endif

	model->weights.token_embedding_table = model->data;

	const u64 PREFIX_LEN = TENSOR_FILTER_LEN + strlen("layers.");
	for (cJSON* node = first_lm_node; cJSON_GetArrayItem(cJSON_GetObjectItem(node, "data_offsets"), 0)->valuedouble < lm_offset_end; node = node->next) {
		/**
      * node->string will be of this format
      * model.language_model.layers.[layer_number].[tensor_name]
      * with the following exceptions:
      * 1. model.language_model.embed_tokens.weight
      * 2. model.language_model.embed_tokens_per_layer.weight
      * 3. model.language_model.norm.weight
      * 4. model.language_model.per_layer_model_projection.weight
      * 5. model.language_model.per_layer_projection_norm.weight
      */

		// NOTE: This addition pushes p into uninitialised memory territory
		// for the above 5 string exceptions. Should be fine since they will never match in the 'strcmp'
		const char* p = node->string + PREFIX_LEN;
		u32         layer = atoi(p);
		while (*p && *p != '.') ++p;  // reach '.'
		++p;                          // skip '.'
		const u64 offset = (u64)(cJSON_GetArrayItem(cJSON_GetObjectItem(node, "data_offsets"), 0)->valuedouble - lm_offset_start);
		if (strcmp("input_layernorm.weight", p) == 0) {
			model->weights.rms_input[layer] = (bf16*)((u8*)model->data + offset);
#ifndef NDEBUG
			bf16* h_ptr = (bf16*)((u8*)model_mmap + offset);
			CHECK_ERROR(correctness_weight_ptr_partition(*e_ctx, model->weights.rms_input[layer], h_ptr, 5));
#endif
		} else if (strcmp("self_attn.q_proj.weight", p) == 0) {
			model->weights.wq[layer] = (bf16*)((u8*)model->data + offset);
#ifndef NDEBUG
			bf16* h_ptr = (bf16*)((u8*)model_mmap + offset);
			CHECK_ERROR(correctness_weight_ptr_partition(*e_ctx, model->weights.wq[layer], h_ptr, 5));
#endif
		} else if (strcmp("self_attn.k_proj.weight", p) == 0) {
			model->weights.wk[layer] = (bf16*)((u8*)model->data + offset);
#ifndef NDEBUG
			bf16* h_ptr = (bf16*)((u8*)model_mmap + offset);
			CHECK_ERROR(correctness_weight_ptr_partition(*e_ctx, model->weights.wk[layer], h_ptr, 5));
#endif
		} else if (strcmp("self_attn.v_proj.weight", p) == 0) {
			model->weights.wv[layer] = (bf16*)((u8*)model->data + offset);
#ifndef NDEBUG
			bf16* h_ptr = (bf16*)((u8*)model_mmap + offset);
			CHECK_ERROR(correctness_weight_ptr_partition(*e_ctx, model->weights.wv[layer], h_ptr, 5));
#endif
		} else if (strcmp("self_attn.o_proj.weight", p) == 0) {
			model->weights.wo[layer] = (bf16*)((u8*)model->data + offset);
#ifndef NDEBUG
			bf16* h_ptr = (bf16*)((u8*)model_mmap + offset);
			CHECK_ERROR(correctness_weight_ptr_partition(*e_ctx, model->weights.wo[layer], h_ptr, 5));
#endif
		}
	}

#if !defined(__REMOTE__)
	const char* tokenizer_json_filepath = "gemma-4-E2B-it/tokenizer.json";
#else
	const char* tokenizer_json_filepath = "gemma-4-12B-it/tokenizer.json";
#endif

	tokenizer_build(*e_ctx, &model->tokenizer, tokenizer_json_filepath);

	cJSON_Delete(header_root);
	munmap(model_mmap, model->file_bsize);
	fclose(file);
	return;
}

static void model_destroy(ExecCtx** e_ctx, Model* model)
{
	tokenizer_destroy(&model->tokenizer);
	CHECK_ERROR(exec_ctx_destroy(e_ctx));
}

int main(void)
{
	// print_dev_props();
	cudaDeviceProp dev_prop = {};
	CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, 0));

#if !defined(__REMOTE__)
	const char* model_filepath = "gemma-4-E2B-it/model.safetensors";
	const char* model_config_filepath = "gemma-4-E2B-it/config.json";
#else
	const char* model_filepath = "gemma-4-12B-it/model.safetensors";
	const char* model_config_filepath = "gemma-4-12B-it/config.json";
#endif

	ExecCtx* e_ctx = NULL;
	Model    model = { 0 };
	model_build(&e_ctx, &model, model_filepath, model_config_filepath);
	print_model_config(&model);

	u32*        _h_input_tokens = NULL;
	u32         input_tokens_len = 0;
	u64         pop_pos = 0;
	const char* token_prompt32 = "As far as I understand, the first thing I want to set right, is linear lighting along with the HDR tonemapper. In linear lighting I have enabled";
	tokenizer_encode(e_ctx, &model.tokenizer, token_prompt32, &_h_input_tokens, &input_tokens_len, &pop_pos);

	// Do this allocation first
	bf16*     _d_input_embeddings = NULL;
	const u64 input_embeddings_bsize = input_tokens_len * model.config.dim * sizeof *_d_input_embeddings;
	CHECK_ERROR(arena_dev_push(&e_ctx->dev_arena, input_embeddings_bsize, (void**)&_d_input_embeddings));

	// So that we can pop this allocation
	const u64 input_tokens_bsize = input_tokens_len * sizeof *_h_input_tokens;
	u32*      _d_input_tokens = NULL;
	CHECK_ERROR(arena_dev_push(&e_ctx->dev_arena, input_tokens_bsize, (void**)&_d_input_tokens));
	CHECK_ERROR(cu_memcpy_htd(_d_input_tokens, _h_input_tokens, input_tokens_bsize));
	arena_host_pop_at((HostArena*)e_ctx, pop_pos);

	dim3 block_size = model.config.dim / 4;  // 1 thread per 4 elements
	if (block_size.x > dev_prop.maxThreadsPerBlock) {
		fprintf(stderr, "threads launched for k_fetch_input_embeddings exceeds max number of threads per block for this device\n");
		exit(EXIT_FAILURE);
	}
	dim3 grid_size = input_tokens_len;  // one block per token
	// print_dev_buf_u32(e_ctx, _d_input_tokens, input_tokens_len * sizeof *_d_input_tokens);
	// Consult Max Threads per Block : Because model.config.dim > Max Threads per block for 4070
	k_fetch_input_embeddings<<<grid_size, block_size>>>(_d_input_tokens, input_tokens_len, model.config.dim, model.weights.token_embedding_table, _d_input_embeddings);
	CHECK_CUDA(cudaDeviceSynchronize());
	// print_dev_buf_bf16(e_ctx, _d_input_embeddings, input_embeddings_bsize);

	k_rmsnorm<<<grid_size, block_size, block_size.x / _CU_CONST_WARP_SIZE>>>(_d_input_embeddings, model.config.dim, model.weights.rms_input[0]);
	CHECK_CUDA(cudaDeviceSynchronize());
	// print_dev_buf_bf16(e_ctx, _d_input_embeddings, input_embeddings_bsize);
	arena_dev_pop(&e_ctx->dev_arena, input_tokens_bsize);

	/**
    * TODO: 
    * 1. Calculate the size for each projected matrix Q,K,V for layer 0
    * 2. Allocate that
    * 3. Perform 3 GEMMs
    */

	const u64 projected_bsize = input_tokens_len * model.config.global_head_dim * model.config.n_heads * sizeof **model.weights.wq;

	bf16* q = NULL;
	bf16* k = NULL;
	bf16* v = NULL;

	arena_dev_push(&e_ctx->dev_arena, projected_bsize, (void**)&q);
	arena_dev_push(&e_ctx->dev_arena, projected_bsize, (void**)&k);
	arena_dev_push(&e_ctx->dev_arena, projected_bsize, (void**)&v);

	// NOTE: What if safetensors doesn't provide matrices in row major?
	// TODO: Have some sort of failsafe in case we try to use more than the device has
	// TODO: This should go into a kernel wrapper
	const u32 tiling_block_size = 32;
	grid_size = dim3(CEIL_DIVI(model.config.dim, tiling_block_size), CEIL_DIVI(input_tokens_len, tiling_block_size));
	block_size = dim3(tiling_block_size, tiling_block_size);
	// I need enough smem for 2 32x32 BF16 matrices
	const u64 smem_dyn_size = 2 * tiling_block_size * tiling_block_size * sizeof *model.weights.wq[0];

	cudaStream_t stream_q, stream_k, stream_v;
	// NOTE: Creating these is not free!
	CHECK_CUDA(cudaStreamCreate(&stream_q));
	CHECK_CUDA(cudaStreamCreate(&stream_k));
	CHECK_CUDA(cudaStreamCreate(&stream_v));

	// BUG: I don't think this'll work for input sequence length that is *not* a multiple of 32
	k_gemmt<<<grid_size, block_size, smem_dyn_size, stream_q>>>(_d_input_embeddings, model.weights.wq[0], q, tiling_block_size, input_tokens_len, model.config.dim, model.config.global_head_dim * model.config.n_heads);
	k_gemmt<<<grid_size, block_size, smem_dyn_size, stream_k>>>(_d_input_embeddings, model.weights.wk[0], k, tiling_block_size, input_tokens_len, model.config.dim, model.config.global_head_dim * model.config.n_heads);
	k_gemmt<<<grid_size, block_size, smem_dyn_size, stream_v>>>(_d_input_embeddings, model.weights.wv[0], v, tiling_block_size, input_tokens_len, model.config.dim, model.config.global_head_dim * model.config.n_heads);
	CHECK_CUDA(cudaDeviceSynchronize());

	cudaStreamDestroy(stream_q);
	cudaStreamDestroy(stream_k);
	cudaStreamDestroy(stream_v);

	model_destroy(&e_ctx, &model);

#ifndef NDEBUG
	if (e_ctx) {
		printf("[WARNING] e_ctx is not null\n");
	}
#endif

	return 0;
}
