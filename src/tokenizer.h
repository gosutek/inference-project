#if !defined(TOKENIZER_H)
#define TOKENIZER_H

#include <allocator.h>

#if defined(__cplusplus)
extern "C"
{
#endif

	void tokenizer_tokenize(ExecCtx* e_ctx, const char* input);

#if defined(__cplusplus)
}
#endif

#endif
