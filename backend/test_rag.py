from rag.vector_store import search_documents
from rag.generator import generate_answer


def ask_rag(question: str):

    # Step 1: Retrieve relevant documents
    results = search_documents(question)

    documents = results["documents"][0]

    if not documents:
        return "No relevant documentation found."

    # Step 2: Combine retrieved documents
    context = "\n\n".join(documents)

    # Step 3: Generate answer using Qwen
    answer = generate_answer(question, context)

    return answer


if __name__ == "__main__":

    question = "Why is my GPU running out of memory?"

    print("\nQuestion:", question)

    answer = ask_rag(question)

    print("\nAI Generated Answer:\n")
    print(answer)
