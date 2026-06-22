#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "cJSON.h"

#include <helpers.h>
#include <tokenizer.h>
#include <unistd.h>

// TODO: This is a duplicate with the one in main.cu
static Error_t get_file_bsize(const char* filepath, u64* const bsize)
{
	struct stat st;
	if (stat(filepath, &st) == 0) {
		*bsize = (u64)st.st_size;
		return Success;
	}
	return ErrorGeneric;
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

void tokenizer_encode(ExecCtx* e_ctx, const Tokenizer* const tokenizer, const char* input, u32** input_tokens, u32* const input_tokens_len, u64* const pop_arena_pos)
{
	char**      buf = NULL;
	const char* uni_underscore = "▁";
	const u64   uni_underscore_bsize = strlen(uni_underscore);

	u32 input_tokens_idx = 0;
	*pop_arena_pos = e_ctx->host_arena.pos;
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, strlen(input) * sizeof **input_tokens, (void**)input_tokens));

	const char* p = input;
	u32         n_words = 0;
	while (*p != '\0') {
#if !defined(NDEBUG)
		printf("Outer Loop: %u\n", n_words + 1);
#endif
		const char* l = p;
		u64         word_len = 0;
		u64         allocation_bsize = 0;

		while (*p != ' ' && *p != '\0') {
			++word_len;
			++p;
		}  // p is now on the whitespace

		b32 is_first = (l == input);
		if (!is_first) {
			++word_len;  // account for the added "▁"
		}

		CHECK_ERROR(arena_host_push((HostArena*)e_ctx, word_len * sizeof(char*), (void**)&buf));
		allocation_bsize += word_len * sizeof(char*);

		u32 char_idx = 0;
		if (!is_first) {
			CHECK_ERROR(arena_host_push((HostArena*)e_ctx, uni_underscore_bsize + 1, (void**)&buf[char_idx]));
			allocation_bsize += uni_underscore_bsize + 1;
			strcpy(buf[char_idx], uni_underscore);
			++char_idx;
		}

		while (l < p)  // every char before the whitespace of p
		{
			const u32 char_len = utf8_byte_count(l);
			CHECK_ERROR(arena_host_push((HostArena*)e_ctx, char_len + 1, (void**)&buf[char_idx]));
			allocation_bsize += char_len + 1;
			memcpy(buf[char_idx], l, char_len);
			buf[char_idx][char_len] = '\0';

			++char_idx;
			l += char_len;
		}
		++n_words;
		if (*p == ' ') {
			++p;
		}

		// BPE LOOP
		char* pair_buf = NULL;
		CHECK_ERROR(arena_host_push((HostArena*)e_ctx, tokenizer->max_token_length, (void**)&pair_buf));
		allocation_bsize += tokenizer->max_token_length;

		while (1) {
			u64 min_rank = UINT64_MAX;
			for (u32 i = 0; i < char_idx - 1; ++i) {
				char* p1 = buf[i];
				char* p2 = buf[i + 1];

				sprintf(pair_buf, "%s %s", p1, p2);

				MergesMap* found_merge = NULL;
				HASH_FIND_STR(tokenizer->merges, pair_buf, found_merge);
				if (found_merge && min_rank >= found_merge->rank) {
					min_rank = found_merge->rank;
				}
			}

			if (min_rank == UINT64_MAX) {
				break;
			}

			u32 read_idx = 0;
			u32 write_idx = 0;

			while (read_idx < char_idx)  // I plan to read every character
			{
				if (read_idx < char_idx - 1)  // The code inside this 'if' is for pairs only. The outer while handles all characters
				{
					// So, if we got a pair
					// 1. is it of minimum rank?
					// 2. If YES then construct the concatenated string
					// 2.1 The buffer 'buf' at the index of the first member of the pair will point to that new string.
					// 2.2 We must now read the next character. However, the next character on i + 2 cause we merged i and i + 1.
					// 2.3 The next write must occur on i + 1 as the character on that potition got merged with the previous character and on the next loop we must consider the merged pair with the i + 2'th character.
					// 3. If NO then just move both the write and read ptr's to the next character.

					MergesMap* found_merge = NULL;
					sprintf(pair_buf, "%s %s", buf[read_idx], buf[read_idx + 1]);
					HASH_FIND_STR(tokenizer->merges, pair_buf, found_merge);
					if (found_merge == NULL || found_merge->rank > min_rank) {
						buf[write_idx++] = buf[read_idx++];
						continue;
					}
					char* merge = NULL;
					u64   merge_allocation_bsize = strlen(buf[read_idx]) + strlen(buf[read_idx + 1]) + 1;
					CHECK_ERROR(arena_host_push((HostArena*)e_ctx, merge_allocation_bsize, (void**)&merge));
					allocation_bsize += merge_allocation_bsize;
					strcpy(merge, buf[read_idx]);
					strcat(merge, buf[read_idx + 1]);
					buf[write_idx++] = merge;
					read_idx += 2;
					continue;
				}
				buf[write_idx++] = buf[read_idx++];
			}
			char_idx = write_idx;
		}

		// NOTE: Pickle:
		// You don't know the exact size to allocate until you reach this point
		// But then you need to allocate on top of all the non-persistent buffers used in the BPE loop
		// That means you can't pop them.
		// Solution:
		// 1. Allocate them on top.
		// 2. Record the size in loop_allocation_bsize
		// 3. Return it to the caller.
		// 4. Pop all of the buffers used in this function after you copy the input tokens to the device

		*input_tokens_len += char_idx;
		for (u32 i = 0; i < char_idx; ++i) {
			VocabMap* found_vocab = NULL;
			HASH_FIND_STR(tokenizer->vocab, buf[i], found_vocab);
			if (found_vocab != NULL) {
				(*input_tokens)[input_tokens_idx++] = found_vocab->id;
#if !defined(NDEBUG)
				printf("[%d] %s\n", found_vocab->id, buf[i]);
#endif
			}
		}
#if !defined(NDEBUG)
		printf("input_tokens_len: %u\n", *input_tokens_len);
#endif
		arena_host_pop((HostArena*)e_ctx, allocation_bsize);
	}
}

void tokenizer_build(ExecCtx* e_ctx, Tokenizer* tokenizer, const char* config_filepath)
{
	tokenizer->max_token_length = 0;
	FILE* file = fopen(config_filepath, "rb");
	if (!file) {
		fprintf(stderr, "failed to open %s\n", config_filepath);
		exit(EXIT_FAILURE);
	}
	u64 tokenizer_config_file_bsize = 0;
	CHECK_ERROR(get_file_bsize(config_filepath, &tokenizer_config_file_bsize));

	char* json_buf = NULL;
	CHECK_ERROR(arena_host_push((HostArena*)e_ctx, tokenizer_config_file_bsize + 1, (void**)&json_buf));

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

	arena_host_pop((HostArena*)e_ctx, tokenizer_config_file_bsize + 1);
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

	cJSON* vocab_item_json = vocab_object->child;
	while (vocab_item_json != NULL) {
		const u64 vocab_item_str_len = strlen(vocab_item_json->string);
		VocabMap* vocab_item_map = NULL;
		CHECK_ERROR(arena_host_push((HostArena*)e_ctx, sizeof *vocab_item_map, (void**)&vocab_item_map));

		tokenizer->max_token_length = MAX(vocab_item_str_len, tokenizer->max_token_length);
		CHECK_ERROR(arena_host_push((HostArena*)e_ctx, vocab_item_str_len, (void**)&vocab_item_map->token));
		strcpy(vocab_item_map->token, vocab_item_json->string);

		vocab_item_map->id = vocab_item_json->valueint;
		HASH_ADD_KEYPTR(hh, tokenizer->vocab, vocab_item_map->token, strlen(vocab_item_map->token), vocab_item_map);
		vocab_item_json = vocab_item_json->next;
	}

	cJSON* merges_object = cJSON_GetObjectItemCaseSensitive(model_object, "merges");
	if (merges_object == NULL || !cJSON_IsArray(merges_object)) {
		fprintf(stderr, "unexpected error: 'tokenizer_config.json' doesn't have a 'merges' object\n");
		exit(EXIT_FAILURE);
	}
	cJSON* merge_item_json = NULL;
	u64    priority_rank = 0;
	cJSON_ArrayForEach(merge_item_json, merges_object)
	{
		MergesMap* merges_item_map = NULL;
		CHECK_ERROR(arena_host_push((HostArena*)e_ctx, sizeof *merges_item_map, (void**)&merges_item_map));
		cJSON* left = cJSON_GetArrayItem(merge_item_json, 0);
		cJSON* right = cJSON_GetArrayItem(merge_item_json, 1);

		u64 pair_len = strlen(left->valuestring) + strlen(right->valuestring) + 1 + 1;  // ' ' + '\0'
		CHECK_ERROR(arena_host_push((HostArena*)e_ctx, pair_len, (void**)&merges_item_map->pair));
		snprintf(merges_item_map->pair, pair_len, "%s %s", left->valuestring, right->valuestring);
		merges_item_map->rank = priority_rank++;
		HASH_ADD_KEYPTR(hh, tokenizer->merges, merges_item_map->pair, strlen(merges_item_map->pair), merges_item_map);
	}

	cJSON_Delete(tokenizer_config_json);
}

void tokenizer_destroy(Tokenizer* tokenizer)
{
	VocabMap *vocab_item, *vocab_tmp;
	HASH_ITER(hh, tokenizer->vocab, vocab_item, vocab_tmp)
	{
		HASH_DEL(tokenizer->vocab, vocab_item);
	}

	MergesMap *merges_item, *merges_tmp;
	HASH_ITER(hh, tokenizer->merges, merges_item, merges_tmp)
	{
		HASH_DEL(tokenizer->merges, merges_item);
	}
}
