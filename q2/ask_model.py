import os
import json
import sys
import openai
from validator import load_knowledge_base, validate_answer, log_validation
from dotenv import load_dotenv

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path)


if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key with:")
    print("export OPENAI_API_KEY='your-api-key'")
    sys.exit(1)


openai.api_key = os.environ["OPENAI_API_KEY"]

def ask_model(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide concise, factual answers."},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return "Error: Could not get response from model"

def process_question(question, kb, max_attempts=2):

    attempt = 1
    while attempt <= max_attempts:
        model_answer = ask_model(question)
        
        result, kb_answer = validate_answer(question, model_answer, kb)
        
        log_validation(question, model_answer, result, kb_answer, attempt)
        
        if result == "VALID" or attempt >= max_attempts:
            return question, model_answer, result, kb_answer
        
        if "differs from KB" in result:
            question = f"{question} Please be precise and concise."
        else:  
            question = f"{question} If you're not sure, please say 'I don't know'."
        
        attempt += 1
    
    return question, model_answer, result, kb_answer

def main():
    kb = load_knowledge_base()
    
    kb_questions = [item["question"] for item in kb]
    
    additional_questions = [
        "What is the population of Tokyo?",
        "Who is the current CEO of Microsoft?",
        "What is the theory of relativity?",
        "How do quantum computers work?",
        "What are the health benefits of meditation?"
    ]
    
    all_questions = kb_questions + additional_questions
    results = []
    

    log_path = os.path.join(script_dir, "run.log")
    with open(log_path, 'w') as f:
        f.write("Starting new validation run\n")
        f.write("=" * 50 + "\n")
    

    for i, question in enumerate(all_questions):
        print(f"Processing question {i+1}/{len(all_questions)}: {question}")
        question, model_answer, result, kb_answer = process_question(question, kb)
        
        results.append({
            "question": question,
            "model_answer": model_answer,
            "validation": result,
            "kb_answer": kb_answer
        })
    

    valid_count = sum(1 for r in results if r["validation"] == "VALID")
    kb_mismatch_count = sum(1 for r in results if "differs from KB" in r["validation"])
    out_of_domain_count = sum(1 for r in results if "out-of-domain" in r["validation"])
    
    summary_path = os.path.join(script_dir, "summary.md")
    with open(summary_path, 'w') as f:
        f.write("# Hallucination Detection Summary\n\n")
        f.write(f"Total questions: {len(results)}\n")
        f.write(f"Valid answers: {valid_count}\n")
        f.write(f"KB mismatches: {kb_mismatch_count}\n")
        f.write(f"Out-of-domain: {out_of_domain_count}\n\n")
        
        f.write("## Detailed Results\n\n")
        for i, r in enumerate(results):
            f.write(f"### Question {i+1}: {r['question']}\n")
            f.write(f"- Model answer: {r['model_answer']}\n")
            f.write(f"- Validation: {r['validation']}\n")
            if r['kb_answer']:
                f.write(f"- KB answer: {r['kb_answer']}\n")
            f.write("\n")

if __name__ == "__main__":
    main() 