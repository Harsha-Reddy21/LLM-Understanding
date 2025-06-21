from transformers import GPT2Tokenizer, BertTokenizer, AlbertTokenizer

sentence = "The cat sat on the mat because it was tired."


bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
wordpiece_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentencepiece_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


bpe_tokens = bpe_tokenizer.tokenize(sentence)
bpe_ids = bpe_tokenizer.convert_tokens_to_ids(bpe_tokens)

wp_tokens = wordpiece_tokenizer.tokenize(sentence)
wp_ids = wordpiece_tokenizer.convert_tokens_to_ids(wp_tokens)

sp_tokens = sentencepiece_tokenizer.tokenize(sentence)
sp_ids = sentencepiece_tokenizer.convert_tokens_to_ids(sp_tokens)

# Report
def report(name, tokens, ids):
    print(f"\n{name} Tokenizer")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Token IDs: {ids}")

report("BPE (GPT2)", bpe_tokens, bpe_ids)
report("WordPiece (BERT)", wp_tokens, wp_ids)
report("SentencePiece (ALBERT)", sp_tokens, sp_ids)
