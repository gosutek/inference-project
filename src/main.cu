#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "cJSON.h"

#include "allocator.h"
#include "helpers.h"
#include "model.cuh"

static i32 get_file_bsize(const char* filepath, u64* const bsize)
{
	struct stat st;
	if (stat(filepath, &st) == 0) {
		*bsize = (u64)st.st_size;
		return 1;
	}
	return 0;
}

static i32 build_model(ExecCtx* const e_ctx, Model* const model, const char* model_config_filepath)
{
	FILE* file = fopen(model_config_filepath, "r");
	if (!file) {
		fprintf(stderr, "couldn't read file %s\n", model_config_filepath);
		exit(EXIT_FAILURE);
	}

	u64 model_config_bsize = 0;
	get_file_bsize(model_config_filepath, &model_config_bsize);

	char* json_buf = NULL;
	if (mem_arena_host_push((HostArena*)e_ctx, model_config_bsize + 1, (void**)&json_buf) != 1) {
		fprintf(stderr, "failed pool push\n");
		exit(EXIT_FAILURE);
	}
	if (fread(json_buf, sizeof *json_buf, model_config_bsize, file) != model_config_bsize) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	json_buf[model_config_bsize] = '\0';  // cJSON works on null-terminated strings

	cJSON* model_config = cJSON_Parse(json_buf);
	model_config = model_config->child;
	while (strcmp(model_config->string, "text_config") != 0) {
		model_config = model_config->next;
	}
	model->config.dim = cJSON_GetObjectItem(model_config, "hidden_size")->valueint;
	model->config.ffn_dim = cJSON_GetObjectItem(model_config, "intermediate_size")->valueint;
	model->config.global_head_dim = cJSON_GetObjectItem(model_config, "global_head_dim")->valueint;
	model->config.n_heads = cJSON_GetObjectItem(model_config, "num_attention_heads")->valueint;
	model->config.vocab_size = cJSON_GetObjectItem(model_config, "vocab_size")->valueint;
	model->config.n_layers = cJSON_GetObjectItem(model_config, "num_hidden_layers")->valueint;

	cJSON_Delete(model_config);
	return 1;
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

int main(void)
{
	const char* model_filepath = "gemma-4-E2B-it/model.safetensors";
	const char* model_config_filepath = "gemma-4-E2B-it/config.json";

	u64 safetensor_bsize = 0;  // total file size of the safetensor in bytes
	if (get_file_bsize(model_filepath, &safetensor_bsize) != 1) {
		fprintf(stderr, "couldn't read file size of %s\n", model_filepath);
		exit(EXIT_FAILURE);
	}

	FILE* file = fopen(model_filepath, "r");
	if (!file) {
		fprintf(stderr, "couldn't load %s\n", model_filepath);
		exit(EXIT_FAILURE);
	}

	u64 header_bsize = 0;  // header size of the safetensor in bytes
	if (fread(&header_bsize, sizeof header_bsize, 1, file) != 1) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	// This is the byte size of the actual model (weights), we need to allocate at least this much for DevArena
	const u64 model_bsize = safetensor_bsize - header_bsize - sizeof header_bsize;

	ExecCtx* e_ctx = NULL;
	if (exec_ctx_create(&e_ctx, model_bsize) != 1) {
		fprintf(stderr, "failed pool allocation\n");
		exit(EXIT_FAILURE);
	}
	Model model;
	build_model(e_ctx, &model, model_config_filepath);
	print_model(&model);

	char* json_buf = NULL;
	if (mem_arena_host_push((HostArena*)e_ctx, header_bsize + 1, (void**)&json_buf) != 1) {
		fprintf(stderr, "failed pool push\n");
		exit(EXIT_FAILURE);
	}

	if (fread(json_buf, sizeof *json_buf, header_bsize, file) != header_bsize) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	json_buf[header_bsize] = '\0';

	cJSON* header = cJSON_Parse(json_buf);

	int   fd = fileno(file);
	void* mapped = mmap(NULL, safetensor_bsize, PROT_READ, MAP_PRIVATE, fd, 0);
	if (mapped == MAP_FAILED) {
		fprintf(stderr, "failed to mmap safetensor\n");
		exit(EXIT_FAILURE);
	}
	fclose(file);

	cJSON*      node = header->child;
	const char* tensor_filter = "model.language_model";
	cJSON*      prev_end = NULL;
	while (node != NULL) {
		if (strlen(node->string) >= strlen(tensor_filter) && strncmp(node->string, tensor_filter, strlen(tensor_filter)) == 0) {  // filter out the vision and audio model
			cJSON* dtype = cJSON_GetObjectItem(node, "dtype");
			if (strcmp(dtype->valuestring, "BF16") != 0) {
				fprintf(stderr, "Unexpected dtype for tensor '%s': %s\n", node->string, dtype->valuestring);
				exit(EXIT_FAILURE);
			}
			cJSON* offsets = cJSON_GetObjectItem(node, "data_offsets");
			cJSON* start = cJSON_GetArrayItem(offsets, 0);
			cJSON* end = cJSON_GetArrayItem(offsets, 1);
			if (prev_end && prev_end->valuedouble != start->valuedouble) {  // contiguous memory check
				fprintf(stderr, "Unexpected ordering of tensors '%s'\n", node->string);
				exit(EXIT_FAILURE);
			}
			prev_end = end;
			// printf("%s: [%.0f, %.0f] %s\n", node->string, start->valuedouble, end->valuedouble, dtype->valuestring);
		}
		node = node->next;
	}

	cJSON_Delete(header);
	exec_ctx_destroy(&e_ctx);

#ifndef NDEBUG
	if (e_ctx) {
		printf("[WARNING] e_ctx is not null\n");
	}
#endif

	return 0;
}
