#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "cJSON.h"

#include <helpers.h>
#include <tokenizer.h>

// TODO: This is a duplicate with the one in main.cu
static i32 get_file_bsize(const char* filepath, u64* const bsize)
{
	struct stat st;
	if (stat(filepath, &st) == 0) {
		*bsize = (u64)st.st_size;
		return 1;
	}
	return 0;
}

static inline u32 utf8_byte_count(const char* const c)
{
	const unsigned char* const uc = (const unsigned char* const)c;

	if (*uc < 0x80) {
		return 1;
	} else if (*uc < 0xe0) {
		return 2;
	} else if (*uc < 0xf0) {
		return 3;
	} else {
		return 4;
	}
}

// char* not because we're expecting a string
// but because 'c' might be unicode of more than 1 byte
static inline u32 utf8_decode(const char* const c)
{
	const unsigned char* const uc = (const unsigned char* const)c;

	// https://en.wikipedia.org/wiki/UTF-8
	if (*uc < 0x80)  // up to and including b01111111 = 0x7F
	{
		return (u32)uc[0];
	} else if (*uc < 0xe0)  // up to and including b11011111 = 0xdf
	{
		/** 1. Capture the last 5 bits from the first byte (110xxxyy): 00011111 = 0x1f & *uc
      * 2. Capture the last 6 bits from the second byte (10yyzzzz): 00111111 = 0x3f & *(uc + 1)
      * 3. Move the first 5 bits 6 spaces to the left, making space for the 6 bits of the second byte.
      **/
		return ((u32)*uc & 0x1f) << 6 | ((u32)(*(uc + 1)) & 0x3f);  // Need to cast here to u32 due to 'integer promotion' -> Read more https://en.cppreference.com/c/language/conversion#Integer_promotions
	} else if (*uc < 0xf0)                                          // up to and including b11101111 = 0xef
	{
		/** 1. Capture the last 4 bits from the first byte (1110wwww): 00001111 = 0xf & *uc
      * 2. Capture the last 6 bits from the second byte (10xxxxyy): 00111111 = 0x3f & *(uc + 1)
      * 3. Capture the last 6 bits from the third byte (10yyzzzz): 00111111 = 0x3f & *(uc + 2)
      * 4. Move the first 4 bits 12 spaces to the left, the following 6 bits 6 space to the left.
      **/
		return ((u32)*uc & 0xf) << 12 | ((u32)(*(uc + 1)) & 0x3f) << 6 | ((u32)(*(uc + 2)) & 0x3f);
	} else {
		/** 1. Capture the last 3 bits from the first byte (11110uvv): 00000111 = 0x7 & *uc
      * 2. Capture the last 6 bits from the second byte (10vvwwww): 00111111 = 0x3f & *(uc + 1)
      * 3. Capture the last 6 bits from the third byte (10xxxxyy): 00111111 = 0x3f & *(uc + 2)
      * 4. Capture the last 6 bits from the fourth byte (10yyzzzz): 00111111 = 0x3f & *(uc + 3)
      * 5. Move the first 3 bits 18 spaces to the left, the following 6 bits 12 spaces to the left and the following 6 bits 6 spaces to the left.
      **/
		return ((u32)*uc & 0x7) << 18 | ((u32)(*(uc + 1)) & 0x3f) << 12 | ((u32)(*(uc + 2)) & 0x3f) << 6 | ((u32)(*(uc + 3)) & 0x3f);
	}
}

static void tokenizer_normalizer(ExecCtx* e_ctx, const char* input, char** output)
{
	const char* pattern = " ";  // TODO: Eventually have these be read from the tokenizer.json
	const char* content = "▁";

	const u64 content_byte_count = utf8_byte_count(content);

	u64 normalized_strlen = 0;
	for (const char* c = input; *c != '\0'; ++c) {
		if (*c == *pattern) {  // WARN: This only works if 'pattern' is one byte
			normalized_strlen += content_byte_count;
		} else {
			++normalized_strlen;
		}
	}

	if (mem_arena_host_push((HostArena*)e_ctx, normalized_strlen + 1, (void**)output) != 1) {  // +1 for '\0'
		fprintf(stderr, "failed to push to arena\n");
		exit(EXIT_FAILURE);
	}
	char* c = *output;
	while (*input != '\0') {
		if (*input == *pattern) {
			memcpy(c, content, content_byte_count);
			c += content_byte_count;
		} else {
			*c = *input;
			++c;
		}
		++input;
	}
	*c = '\0';
}

static void tokenizer_parse_config(ExecCtx* const e_ctx)
{
	FILE* file = fopen("gemma-4-E2B-it/tokenizer.json", "rb");
	if (!file) {
		fprintf(stderr, "failed to open 'gemma-4-E2B-it/tokenizer.json'\n");
		exit(EXIT_FAILURE);
	}
	u64 tokenizer_config_file_bsize = 0;
	get_file_bsize("gemma-4-E2B-it/tokenizer.json", &tokenizer_config_file_bsize);

	char* json_buf = NULL;
	if (mem_arena_host_push((HostArena*)e_ctx, tokenizer_config_file_bsize + 1, (void**)&json_buf) != 1) {
		fprintf(stderr, "failed to push to arena\n");
		exit(EXIT_FAILURE);
	}

	if (fread(json_buf, sizeof *json_buf, tokenizer_config_file_bsize, file) != tokenizer_config_file_bsize) {
		fprintf(stderr, "failed to fread tokenizer.json\n");
		exit(EXIT_FAILURE);
	}

	json_buf[tokenizer_config_file_bsize] = '\0';

	cJSON* tokenizer_config_json = cJSON_Parse(json_buf);
	if (tokenizer_config_json == NULL) {
		fprintf(stderr, "failed to cJSON_Parse 'tokenizer_confg.json\n");
		exit(EXIT_FAILURE);
	}

	mem_arena_host_pop((HostArena*)e_ctx, tokenizer_config_file_bsize + 1);
	fclose(file);

	cJSON* model_object = cJSON_GetObjectItemCaseSensitive(tokenizer_config_json, "model");
	if (model_object == NULL) {
		fprintf(stderr, "unexpected error: 'tokenizer_config.json' doesn't have a 'model' object\n");
		exit(EXIT_FAILURE);
	}
	cJSON* vocab_object = cJSON_GetObjectItemCaseSensitive(model_object, "vocab");
	if (vocab_object == NULL) {
		fprintf(stderr, "unexpected error: 'tokenizer_config.json' doesn't have a 'model/vocab' object\n");
		exit(EXIT_FAILURE);
	}

	VocabMap* vocab_map = NULL;
	cJSON*    vocab_item_json = vocab_object->child;
	u64       vocab_items_bsize = 0;
	u64       vocab_strings_bsize = 0;
	while (vocab_item_json != NULL) {
		VocabMap* vocab_item_map = NULL;
		if (mem_arena_host_push((HostArena*)e_ctx, sizeof *vocab_item_map, (void**)&vocab_item_map) != 1) {
			fprintf(stderr, "failed to push to arena\n");
			exit(EXIT_FAILURE);
		}
		vocab_items_bsize += sizeof *vocab_item_map;  // keep track of the bsize to pop at the end

		if (mem_arena_host_push((HostArena*)e_ctx, strlen(vocab_item_json->string), (void**)&vocab_item_map->token) != 1) {
			fprintf(stderr, "failed to push to arena\n");
			exit(EXIT_FAILURE);
		}
		strcpy(vocab_item_map->token, vocab_item_json->string);
		vocab_strings_bsize += strlen(vocab_item_json->string);  // keep track of the bsize to pop at the end

		vocab_item_map->id = vocab_item_json->valueint;
		HASH_ADD_KEYPTR(hh, vocab_map, vocab_item_map->token, strlen(vocab_item_map->token), vocab_item_map);
		vocab_item_json = vocab_item_json->next;
	}

	MergesMap* merges_map = NULL;
	cJSON*     merges_item_json = cJSON_GetObjectItemCaseSensitive(tokenizer_config_json, "merges");
	if (merges_item_json == NULL || !cJSON_IsArray(merges_item_json)) {
		fprintf(stderr, "unexpected error: 'tokenizer_config.json' doesn't have a 'merges' object\n");
		exit(EXIT_FAILURE);
	}
	u64    merges_items_bsize = 0;
	u64    merges_strings_bsize = 0;
	cJSON* merge_item = NULL;
	cJSON_ArrayForEach(merge_item, merges_item_json)
	{
		MergesMap* merges_item_map = NULL;
		if (mem_arena_host_push((HostArena*)e_ctx, sizeof *merges_item_map, (void**)&merges_item_map) != 1) {
			fprintf(stderr, "failed to push to arena\n");
			exit(EXIT_FAILURE);
		}
		merges_items_bsize += sizeof *merges_item_map;
		cJSON* left = cJSON_GetArrayItem(merge_item, 0);
		cJSON* right = cJSON_GetArrayItem(merge_item, 1);

		if (mem_arena_host_push((HostArena*)e_ctx, strlen(left->string), (void**)&merges_item_map->str_1) != 1) {
			fprintf(stderr, "failed to push to arena\n");
			exit(EXIT_FAILURE);
		}
		strcpy(merges_item_map->str_1, left->string);
		merges_strings_bsize += strlen(left->string);

		if (mem_arena_host_push((HostArena*)e_ctx, strlen(right->string), (void**)&merges_item_map->str_2) != 1) {
			fprintf(stderr, "failed to push to arena\n");
			exit(EXIT_FAILURE);
		}
		strcpy(merges_item_map->str_2, right->string);
		merges_strings_bsize += strlen(right->string);
	}

	VocabMap *e, *tmp;
	HASH_ITER(hh, vocab_map, e, tmp)
	{
		HASH_DEL(vocab_map, e);
	}
	mem_arena_host_pop((HostArena*)e_ctx, vocab_items_bsize + vocab_strings_bsize);

	cJSON_Delete(tokenizer_config_json);
}

void tokenizer_encode(ExecCtx* e_ctx, const char* input)
{
	char* normalized_input = NULL;
	tokenizer_normalizer(e_ctx, input, &normalized_input);
	printf("%s\n", normalized_input);
	mem_arena_host_pop((HostArena*)e_ctx, strlen(normalized_input));
	tokenizer_parse_config(e_ctx);
}
