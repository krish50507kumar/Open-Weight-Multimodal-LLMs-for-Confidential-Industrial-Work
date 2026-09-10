import os
from src.rag import answer_question

PDF_FOLDER = "data/raw_pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)
messages = []
n_results=3
model = input("enter your model:  ")
while True:
    question = input("Enter your message | exit: ")
    if(question.strip().lower() == "exit"):
        break
    messages.append({"role": "user", "content": question})
    answer = answer_question(
                        question,
                        messages,
                        n_results,
                        model
                    )
    print("model: ", answer)
    messages.append({"role": "assistant", "content": answer})
