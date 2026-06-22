#if !defined(TOKENIZER_H)
#define TOKENIZER_H

#include "uthash.h"

#include "allocator.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef struct VocabMap
	{
		char*          token;
		u32            id;
		UT_hash_handle hh;
	} VocabMap;

	typedef struct MergesMap
	{
		char*          pair;
		u64            rank;
		UT_hash_handle hh;
	} MergesMap;

	typedef struct Tokenizer
	{
		VocabMap*  vocab;
		MergesMap* merges;

		u64 max_token_length;
	} Tokenizer;

	void tokenizer_build(ExecCtx* e_ctx, Tokenizer* tokenizer, const char* config_filepath);
	void tokenizer_destroy(Tokenizer* tokenizer);
	void tokenizer_encode(ExecCtx* e_ctx, const Tokenizer* const tokenizer, const char* input, u32** input_tokens, u32* const input_tokens_len, u64* const pop_arena_pos);
	void tokenizer_decode(ExecCtx* e_ctx, const char* input);

#if defined(__cplusplus)
}
#endif

#endif
