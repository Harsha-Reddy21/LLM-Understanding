# Tokenization Algorithm Comparison

## BPE (GPT2)
- **Tokens (11)**: ['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', 'Ġbecause', 'Ġit', 'Ġwas', 'Ġtired', '.']
- **Token IDs**: [464, 3797, 3332, 319, 262, 2603, 780, 340, 373, 10032, 13]
- **Total Count**: 11 tokens

## WordPiece (BERT)
- **Tokens (11)**: ['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']
- **Token IDs**: [1996, 4937, 2938, 2006, 1996, 13523, 2138, 2009, 2001, 5458, 1012]
- **Total Count**: 11 tokens

## SentencePiece (ALBERT)
- **Tokens (11)**: ['▁the', '▁cat', '▁sat', '▁on', '▁the', '▁mat', '▁because', '▁it', '▁was', '▁tired', '.']
- **Token IDs**: [14, 2008, 847, 27, 14, 4277, 185, 32, 23, 4117, 9]
- **Total Count**: 11 tokens

## Why the splits differ

The three tokenization algorithms use different approaches to handle whitespace:

1. **BPE (GPT2)** uses a prefix 'Ġ' to mark tokens that begin with a space. The first token 'The' has no prefix since it starts the sentence.

2. **WordPiece (BERT)** normalizes all text to lowercase and doesn't explicitly mark spaces in its tokens, handling whitespace during preprocessing.

3. **SentencePiece (ALBERT)** uses '▁' (underscore) to represent spaces before tokens, similar to BPE but with a different symbol.

Despite these differences in representation, all three tokenizers split this simple sentence into the same semantic units, resulting in identical token counts. For more complex text with rare words, we would see more significant differences as each algorithm would apply its unique subword segmentation strategy.
