// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
// implementation ported from llama2.c with some modifications to work with r/pi

#include "rpi.h"

# define NULL ((void *)0)

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// --- algo helpers ---

// called in decode
char *parse_raw_byte_token(Tokenizer* t, char *piece) {
    // original code had the following sscanf call:
    //   if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)...
    // we don't have sscanf/libc so we make do manually
    unsigned char byte_val = 0;
    char* p = piece + 3;  // Start parsing after '<0x'
    while (*p) {
        char c = *p;
        if (c >= '0' && c <= '9') {
            byte_val = (byte_val << 4) + (c - '0');
        } else if (c >= 'A' && c <= 'F') {
            byte_val = (byte_val << 4) + (c - 'A' + 10);
        } else if (c >= 'a' && c <= 'f') {
            byte_val = (byte_val << 4) + (c - 'a' + 10);
        } else {
            break;
        }
        p++;
    }
    if (*p == '>') {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

// called in str_lookup
// src: https://github.com/gcc-mirror/gcc/blob/master/libiberty/bsearch.c
void * bsearch (const void *key, const void *base0,
         size_t nmemb, size_t size,
         int (*compar)(const void *, const void *)) {
	const char *base = (const char *) base0;
	int lim, cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (void *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return NULL;
}

// basic sorting function implementation
// qsort import fails ("undefined reference to `__aeabi_uidiv'") so here is a suboptimal but working algo
// called in encode

#if 0
void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *)) {
    // personally avoiding kmalloc calls b/c can't free
    // can potentially lead to horrible memory corruption. unsure... we will see...
    char *arr = (char *)base;
    size_t i, j, min_idx;
    char temp;

    for (i = 0; i < nmemb - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < nmemb; j++) {
            if (compar(arr + j * size, arr + min_idx * size) < 0) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            // Swap elements at i and min_idx
            for (j = 0; j < size; j++) {
                temp = arr[i * size + j];
                arr[i * size + j] = arr[min_idx * size + j];
                arr[min_idx * size + j] = temp;
            }
        }
    }
}
#endif

// backup qsort with malloc
void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *)) {
    char *arr = (char *)base;
    char *temp = kmalloc(size);
    for (size_t i = 0; i < nmemb - 1; i++) {
        for (size_t j = i + 1; j < nmemb; j++) {
            if (compar(arr + i * size, arr + j * size) > 0) {
                memcpy(temp, arr + i * size, size);
                memcpy(arr + i * size, arr + j * size, size);
                memcpy(arr + j * size, temp, size);
            }
        }
    }
}

// --- tokenizer functions ---

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, void *tokenizer_data, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)kmalloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)kmalloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // first int is max_token len. read this value in
    t->max_token_length = *(int*)tokenizer_data;
    tokenizer_data += sizeof(int);
    // read in the file
    int len;
    for (int i = 0; i < vocab_size; i++) {
        memcpy(t->vocab_scores + i, tokenizer_data, sizeof(float));
        tokenizer_data += sizeof(float);
        memcpy(&len, tokenizer_data, sizeof(int));
        // sanity checking so we don't boom the heap
        if (len > 1024) { panic("token of size %u too long\n", len); }
        tokenizer_data += sizeof(int);
        t->vocab[i] = (char*)kmalloc(len + 1);
        memcpy(t->vocab[i], tokenizer_data, len);
        t->vocab[i][len] = '\0'; // add the string terminating token
        tokenizer_data += len;
    }
}

void safe_printk(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        // ascii in this range are non-printable control codes
        if (byte_val < 32 || (byte_val >= 127 && byte_val <= 159)) {
            return;
        }
    }
    printk("%s", piece);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        return parse_raw_byte_token(t, piece);
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { panic("cannot encode NULL text\n"); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = kmalloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = kmalloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            snprintk(str_buffer, t->max_token_length * 2, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;
}