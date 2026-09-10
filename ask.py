import sys
sys.path.append('src')

from rag import answer_question

while True:
    question = input("\nAsk (or 'exit'): ")
    if question.lower() == 'exit':
        break

    answer = answer_question(question)
    print("\nAnswer:", answer)
