import ollama
import openai


def run_llm_query_with_ollama(
    prompt: str,
    model,
    temperature: float = 0.2
):
    """
    Executes a prompt using the LLM model and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The LLM model to use.

    Returns:
        str: The LLM's response.
    """
    print(f"Using the model {model}\n")
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()


# Point OpenAI client to LM Studio's local server
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"


def run_llm_query_with_lmstudio(
    prompt: str,
    model="llama3",
    temperature: float = 0.2
):
    """
    Executes a prompt using the LLM model and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The LLM model to use.

    Returns:
        str: The LLM's response.
    """
    response = openai.ChatCompletion.create(
        model=model_name,  # Replace with the actual name of the model loaded in LM Studio
        messages=[
            #{"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def select_command_with_ollama(user_query, filtered_commands, model):
    """
    Uses an LLM to choose the most appropriate command from the top-k search results.

    Args:
        query (str): The user's natural language query.
        top_k_results (List[Tuple[str, float]]): A list of (command, score) tuples from FAISS.
        model: An instance of an LLM interface (e.g., OpenAI, Ollama) with a `.generate()` or `.chat()` method.

    Returns:
        str: The selected command from the provided options.
    """

    '''
    system_prompt = """You are a strict assistant. You MUST only select ONE command from the provided options based on the user query. 
    Do not generate any new command. Respond with the selected command string only."""

    choices_str = "\n".join([f"- {cmd['command']}: {cmd['description'][0]}" for cmd in filtered_commands])
    prompt = f"{system_prompt}\n\nAvailable Commands:\n{choices_str}\n\nUser Query: {user_query}\n\nWhich command best fits?"
    '''

    # Format the prompt with explicit instruction
    choices_str = "\n".join([f"Command Name: {cmd['command']}, Command Description: {cmd['description'][0]}" for cmd in filtered_commands])
    prompt = f"""
        You are an expert in Ceph command-line usage. Given the following user query:
        "{user_query}"
        Choose ONLY ONE most appropriate command from the list below. DO NOT generate any new command, and DO NOT modify any command.
        Commands:
        "{choices_str}"
        Reply with the "Command Name" corresponding to the best matching command ONLY and NOTHING ELSE example "ceph -s" or "ceph osd df".
        DO NOT include or add prefix in reponse like `The answer is: `
        """

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()


# Point OpenAI client to LM Studio's local server
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"


def select_command_with_lmstudio(user_query, filtered_commands, model_name):
    system_prompt = """You are a strict assistant. You MUST only select ONE command from the provided options based on the user query. 
    Do not generate any new command. Respond with the selected command string only."""

    choices_str = "\n".join([f"- {cmd['command']}: {cmd['description'][0]}" for cmd in filtered_commands])
    prompt = f"""
        You are an expert in Ceph command-line usage. Given the following user query:
        "{user_query}"
        Choose ONLY ONE most appropriate command from the list below. DO NOT generate any new command, and DO NOT modify any command.
        Commands:
        "{choices_str}"
        Reply with the number corresponding to the best matching command ONLY (e.g., 1 or 2 or 3).
        """

    response = openai.ChatCompletion.create(
        model=model_name,  # Replace with the actual name of the model loaded in LM Studio
        messages=[
            #{"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()
