# Assume run_llm_query_with_ollama function from previous step (Ollama setup) is available
# from .llm_interaction import get_llm_response # If in a separate file

# --- 5. Output Analysis & Response Generation Module ---

# Here define the function to analyse the output of the command against 
# the user query.

from llm.llm_response import llmResponse


class analysePrompt(llmResponse):
    def __init__(
        self,
        query: str,
        selected_command: str,
        command_out: str,
        command_description: str,
        model_choice: str,
        model_name: str = "granite3.3:8b",
        temperature: float = float(0.2)
    ) -> None:
        if model_name and temperature:
            super().__init__(model_name, temperature)
        self.query = query
        self.selected_command = selected_command
        self.command_out = command_out
        self.command_description = command_description
        self.model_choice = model_choice

    def _generate_prompt(self) -> str:
        system_prompt = """
        You are an expert Ceph administrator assistant. Your ONLY task is to 
        extract and summarize information from the provided 'COMMAND OUTPUT' 
        to answer the 'USER QUERY'. DO NOT use any external knowledge. 
        DO NOT invent information. If the information is not directly available 
        or cannot be clearly inferred from the 'COMMAND OUTPUT', state that the
        information is not found in the provided output or that you cannot answer. 
        Be concise, clear, and directly answer the question based *only* on the 
        provided text.
        """
        # Craft the user-specific prompt
        user_prompt_parts = [
            f"User Query: {self.query}",
            f"Command Executed: {self.selected_command}"
        ]

        if self.command_description:
            user_prompt_parts.append(
                f"The executed command's relevance is: {self.command_description}"
            )

        user_prompt_parts.append(
            f"COMMAND OUTPUT:\n```\n{self.command_out}\n```\n\n"
            "Please extract the relevant information from the COMMAND OUTPUT to answer "
            "the USER QUERY. Provide a concise, human-readable answer."
        )

        prompt = "\n\n".join(user_prompt_parts)
        prompt = f"{system_prompt}\n\n{prompt}"
        return prompt

    def _analyze_response(self):
        print("Agent Action: Analyzing command output and generating response...")
        prompt = self._generate_prompt()
        try:
            # Call your Ollama-backed LLM function
            # A slightly higher temperature might allow for more natural phrasing,
            # but keep it low for factual extraction
            if self.model_choice == 'o':
                user_response = self._run_llm_query_with_ollama(
                    prompt,
                    )
            elif self.model_choice == 'l':
                user_response = self._run_llm_query_with_lmstudio(
                    prompt
                )
            return user_response.strip()

        except Exception as e:
            print(f"Error during LLM response generation: {e}")
            return "Sorry, I encountered an error while processing the command output."
            pass