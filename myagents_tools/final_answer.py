from smolagents import tool

@tool
def final_answer(answer: str) -> None:
    """
    Outputs the final answer

    Args:
        answer: The final answer text
    """
    print("Final Answer:", answer)

