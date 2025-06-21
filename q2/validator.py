import json
import os
import re

def load_knowledge_base(kb_file="kb.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, kb_file)
    
    with open(kb_path, 'r') as f:
        data = json.load(f)
    return data["knowledge_base"]

def normalize_text(text):

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def is_similar(text1, text2, threshold=0.8):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Simple exact match
    if text1 == text2:
        return True
    
    # Check if one is a subset of the other
    if text1 in text2 or text2 in text1:
        return True
    
    # Calculate word overlap
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return False
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard = len(intersection) / len(union)
    return jaccard >= threshold

def validate_answer(question, model_answer, kb=None):
    """
    Validate if the model's answer matches the knowledge base.
    
    Returns:
    - "VALID" if the answer is valid
    - "RETRY: answer differs from KB" if question is in KB but answer doesn't match
    - "RETRY: out-of-domain" if question is not in KB
    """
    if kb is None:
        kb = load_knowledge_base()
    
    for item in kb:
        if is_similar(question, item["question"]):
            if is_similar(model_answer, item["answer"]):
                return "VALID", item["answer"]
            else:
                return "RETRY: answer differs from KB", item["answer"]
    
    return "RETRY: out-of-domain", None

def log_validation(question, model_answer, validation_result, kb_answer, attempt, log_file="run.log"):

    script_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(script_dir, log_file)
    
    with open(log_path, 'a') as f:
        f.write(f"Question: {question}\n")
        f.write(f"Attempt {attempt} - Model Answer: {model_answer}\n")
        f.write(f"Validation: {validation_result}\n")
        if kb_answer:
            f.write(f"KB Answer: {kb_answer}\n")
        f.write("-" * 50 + "\n")

if __name__ == "__main__":

    kb = load_knowledge_base()
    question = "What is the capital of France?"
    model_answer = "Paris is the capital of France."
    
    result, kb_answer = validate_answer(question, model_answer, kb)
    print(f"Question: {question}")
    print(f"Model Answer: {model_answer}")
    print(f"Validation Result: {result}")
    if kb_answer:
        print(f"KB Answer: {kb_answer}") 