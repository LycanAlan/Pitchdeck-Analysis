from rag_query_agent import setup_rag, answer_query


def main():
    qa_chain = setup_rag()

    while True:
        query = input(" Ask your question (or type 'exit' to quit'): ")
        if query.lower() in {"exit", "quit"}:
            break
        answer_query(query, qa_chain)


if __name__ == "__main__":
    main()


