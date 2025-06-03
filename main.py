import openai
import os
from dotenv import load_dotenv

print("Starting Illuminy...")

load_dotenv()  # Load .env file

openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API key loaded: {bool(openai.api_key)}")

def ask_illuminy(prompt):
    print(f"Received prompt: {prompt}")
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful academic research assistant named Illuminy."},
                {"role": "user", "content": prompt}
            ]
        )
        print("Received response from OpenAI.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        return "Sorry, something went wrong."

if __name__ == "__main__":
    print("Entering main loop. Type 'exit' to quit.")
    while True:
        user_input = input("Ask Illuminy: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Illuminy. Goodbye!")
            break
        answer = ask_illuminy(user_input)
        print("\nIlluminy says:\n", answer, "\n")