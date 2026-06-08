#include <string.h>

#include <helpers.h>
#include <tokenizer.h>

// char* not because we're expecting a string
// but because 'c' might be unicode of more than 1 byte
static u32 utf8_decoder(const char* const c)
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

static void tokenizer_normalize(char* c)
{
	const char* pattern = " ";
	const char* content = "▁";

	while (*c != '\0') {
		if (*c == *pattern) {
			*c = *content;
		}
		++c;
	}
}

void tokenizer_tokenize(ExecCtx* e_ctx, const char* input)
{
	char* input_buf = NULL;
	mem_arena_host_push((HostArena*)e_ctx, strlen(input) * sizeof *input, (void**)&input_buf);
	strcpy(input_buf, input);
	tokenizer_normalize(input_buf);
	printf("%s\n", input_buf);
	mem_arena_host_pop((HostArena*)e_ctx, strlen(input) * sizeof *input);
}
