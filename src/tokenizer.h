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
		i32            id;
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
	} Tokenizer;

	void tokenizer_build(ExecCtx* e_ctx, Tokenizer* tokenizer, const char* config_filepath);
	void tokenizer_destroy(Tokenizer* tokenizer);
	void tokenizer_encode(ExecCtx* e_ctx, const Tokenizer* const tokenizer, const char* input);
	void tokenizer_decode(ExecCtx* e_ctx, const char* input);

#if defined(__cplusplus)
}
#endif

#endif
