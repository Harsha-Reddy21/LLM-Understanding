# NLP Tokenization and Masked Language Modeling

This project demonstrates two key NLP techniques:
1. Comparison of different tokenization algorithms
2. Masked language modeling prediction

## Project Structure

- `tokenise.py`: Compares three tokenization algorithms (BPE, WordPiece, SentencePiece)
- `compare.md`: Analysis of tokenization differences between the algorithms
- `mask.py`: Implements masked language modeling using a pre-trained model
- `predictions.json`: Stores the results of masked token predictions

## Tokenization Comparison

The project compares three popular tokenization algorithms:
- **BPE (GPT2)**: Uses byte-pair encoding with 'Ġ' prefix for whitespace
- **WordPiece (BERT)**: Normalizes text to lowercase with implicit whitespace handling
- **SentencePiece (ALBERT)**: Uses '▁' prefix to represent spaces

Run the tokenization comparison:
```
python tokenise.py
```

## Masked Language Modeling

The `mask.py` script demonstrates how to:
1. Load a pre-trained language model
2. Create a sentence with masked tokens
3. Predict the most likely tokens for each mask
4. Evaluate the plausibility of predictions

The model predicts tokens for the sentence: "The student [MASK] the homework [MASK] the deadline."

Run the masked language model:
```
python mask.py
```

## Results

The predictions for the masked tokens are stored in `predictions.json`, showing:
- Top 3 predictions for each masked position
- Confidence scores for each prediction
- Plausibility assessment for each prediction

The current predictions show that for the first mask, the model predicted "sat", "slept", and "stayed", while for the second mask (after filling the first with "sat"), it predicted "floor", "bed", and "couch".

## Requirements

See `requirements.txt` for dependencies.
