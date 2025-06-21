from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import json
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")


model_name = "facebook/opt-2.7b"  # Using a smaller model that supports fill-mask
tokenizer = AutoTokenizer.from_pretrained(model_name)


mask_token = tokenizer.mask_token if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None else "<mask>"


fill = pipeline("fill-mask", model=model_name, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)


sentence = f"The student {mask_token} the homework {mask_token} the deadline."
print(f"Using sentence: {sentence}")


results = fill(sentence, top_k=3)


predictions = {}

print("First Mask:")
predictions["first_mask"] = []
for i, pred in enumerate(results[0]):  # First mask predictions
    token = pred['token_str']
    score = pred['score']
    
    # Add plausibility comment
    if token in ["completed", "finished", "submitted"]:
        plausibility = "Highly plausible - common action for homework"
    elif token in ["did", "wrote", "started"]:
        plausibility = "Plausible - general action for homework"
    else:
        plausibility = "Less plausible for this context"
    
    prediction_info = {
        "token": token,
        "score": float(score),  # Convert to float for JSON serialization
        "plausibility": plausibility
    }
    predictions["first_mask"].append(prediction_info)
    print(f"{i+1}. {token} (score: {score:.4f}) - {plausibility}")


filled_sentence = sentence.replace(mask_token, results[0][0]["token_str"], 1)
second_results = fill(filled_sentence, top_k=3)

print("\nSecond Mask (after filling first):")
predictions["second_mask"] = []
for i, pred in enumerate(second_results[1] if len(second_results) > 1 and isinstance(second_results, list) else second_results):
    token = pred['token_str']
    score = pred['score']
    
 
    if token in ["before", "by", "at"]:
        plausibility = "Highly plausible - common temporal preposition for deadlines"
    elif token in ["after", "despite", "without"]:
        plausibility = "Less plausible - contradicts typical deadline behavior"
    else:
        plausibility = "Moderately plausible depending on context"
    
    prediction_info = {
        "token": token,
        "score": float(score),  # Convert to float for JSON serialization
        "plausibility": plausibility
    }
    predictions["second_mask"].append(prediction_info)
    print(f"{i+1}. {token} (score: {score:.4f}) - {plausibility}")

with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("\nPredictions saved to predictions.json")
