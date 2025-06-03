import os
from dotenv import load_dotenv
import openai

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def academic_write_and_explain(text):
    """
    Takes student text, improves academic style,
    explains changes, and points out where citations are needed.
    """
    prompt = f"""
You are Illuminy, an expert academic writing coach.

Your task is to take the student's writing below and:

1. Rewrite it to improve academic style, clarity, and grammar.
2. Explain the main improvements you made in simple terms.
3. Identify places where in-text citations should be added, especially after definitions, theories, or concepts, and explain why citations are important there.

Here is the student text:
\"\"\"
{text}
\"\"\"

Respond in three parts, clearly labeled:

---  
1. Improved Text:

2. Explanation of Improvements:

3. Citation Suggestions:
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and encouraging academic writing coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    print("Welcome to Illuminy Academic Writing Assistant!")
    print("Paste your paragraph below and Illuminy will help you improve it academically.")
    print("Type 'exit' to quit.")

    while True:
        student_text = input("\nYour text: ")
        if student_text.lower() in ["exit", "quit"]:
            print("Goodbye! Keep writing and learning.")
            break
        
        result = academic_write_and_explain(student_text)
        print("\n=== Illuminy's Academic Writing Help ===")
        print(result)