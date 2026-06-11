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
		char*          str_1;
		char*          str_2;
		UT_hash_handle hh;
	} MergesMap;

	void tokenizer_encode(ExecCtx* e_ctx, const char* input);
	void tokenizer_decode(ExecCtx* e_ctx, const char* input);

#if defined(__cplusplus)
}
#endif

#endif
