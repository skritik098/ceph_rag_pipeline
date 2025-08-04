# Assume run_llm_query_with_ollama function from previous step (Ollama setup) is available
# from .llm_interaction import get_llm_response # If in a separate file

# --- 5. Output Analysis & Response Generation Module ---


def analyze_response(
    user_query: str,
    selected_command: str,
    command_output: str,
    llm_response_func,  # Your get_llm_response function
    command_description: str,  # Optional: the specific description text of the chosen command purpose
    model_name: str  # Model for Ollama LLM
) -> str:
    """
    Analyzes the command output based on the user's query and generates a
    concise, human-readable response using the LLM. The LLM is strictly
    constrained to the provided command output.

    Args:
        user_query (str): The original natural language query from the user.
        command_name (str): The name of the command that was executed.
        command_output (str): The raw stdout from the executed command.
        llm_response_func (callable): The function to call the LLM (e.g.,
            get_llm_response from your Ollama setup).
        selected_purpose_description (str): An optional detailed description of
            *why* this command was chosen. Helps LLM focus on the relevant part
            of output.
        model_name (str): The name of the Ollama LLM model (e.g., "llama3").

    Returns:
        str: A natural language answer extracted and summarized from the command
            output. Returns an informative message if extraction fails.
    """
    print("Agent Action: Analyzing command output and generating response...")

    # Craft the system prompt to strictly constrain the LLM
    system_prompt = (
        "You are an expert Ceph administrator assistant. Your ONLY task is to "
        "extract and summarize information from the provided 'COMMAND OUTPUT' "
        "to answer the 'USER QUERY'. DO NOT use any external knowledge. "
        "DO NOT invent information. If the information is not directly available "
        "or cannot be clearly inferred from the 'COMMAND OUTPUT', state that the "
        "information is not found in the provided output or that you cannot answer. "
        "Be concise, clear, and directly answer the question based *only* on the "
        "provided text."
    )

    # Craft the user-specific prompt
    user_prompt_parts = [
        f"User Query: {user_query}",
        f"Command Executed: {selected_command}"
    ]

    if command_description:
        user_prompt_parts.append(
            f"The executed command's relevance is: {command_description}"
        )

    user_prompt_parts.append(
        f"COMMAND OUTPUT:\n```\n{command_output}\n```\n\n"
        "Please extract the relevant information from the COMMAND OUTPUT to answer "
        "the USER QUERY. Provide a concise, human-readable answer."
    )

    prompt = "\n\n".join(user_prompt_parts)
    try:
        # Call your Ollama-backed LLM function
        # A slightly higher temperature might allow for more natural phrasing,
        # but keep it low for factual extraction
        llm_response = llm_response_func(
            prompt=f"{system_prompt}\n\n{prompt}",
            model=model_name,
            temperature=0.1,  # Keep temperature low for factual extraction

        )
        return llm_response.strip()

    except Exception as e:
        print(f"Error during LLM response generation: {e}")
        return "Sorry, I encountered an error while processing the command output."