import ollama
import openai


# Here declare a class of LLM to access any of it's method easily

class llmResponse:
    def __init__(self, model_name: str, temperature: float) -> None:
        if model_name:
            self.model = model_name
        else:
            self.model = "granite3.3:8b"
        if temperature:
            self.temperature = temperature
        else:
            self.temperature = float(0.2)

    def _run_llm_query_with_ollama(self, prompt: str):
        """
        Executes a prompt using the LLM model and returns the response.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str): The LLM model to use.

        Returns:
            str: The LLM's response.
        """
        print(f"Using the model {self.model}\n")
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()

    def _run_llm_query_with_lmstudio(self, prompt: str):
        # Point OpenAI client to LM Studio's local server
        openai.api_base = "http://localhost:1234/v1"
        openai.api_key = "not-needed"

        """
        Executes a prompt using the LLM model and returns the response.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str): The LLM model to use.

        Returns:
            str: The LLM's response.
        """
        response = openai.ChatCompletion.create(
            model=self.model_name,  # Replace with the actual name of the model loaded in LM Studio
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()