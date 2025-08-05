# --------------------
# Phase 3: Semantic Search
# --------------------

from utils.file_ops import vectorBuilder
from llm.llm_response import llmResponse


# Here we will create an another class for Semantic Search of Ceph Commands
class semanticCephSearch(llmResponse):
    """
    This class encapsulate the function to do a command search using RAG & LLM
    """
    def __init__(
        self,
        vector_store: vectorBuilder,
        top_k: int,
        threshold: int,
        llm_model: str = "granite3.3:8b",
        temperature: float = float(0.2)
    ) -> None:

        super().__init__(llm_model, temperature)
        self.llm_model = llm_model
        self.top_k = top_k
        self.threshold = threshold
        self.vector_store = vector_store

    def _search_command(self, query: str):
        query_embedding = self.vector_store.model.encode(
            [query],
            convert_to_numpy=True
        )
        distances, indices = self.vector_store.index.search(
            query_embedding,
            self.top_k
        )

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if score <= self.threshold:  # Only include if similarity is good enough
                matched_data = self.vector_store.metadata[int(idx)]
                results.append({
                    "score": float(score),
                    "command": matched_data["command"],
                    "description": matched_data["description"],
                    "query_intent": matched_data["query_intent"]
                })
        return results
    
    def _get_relevance_judge_prompt(self, user_query, available_commands):
        judge_prompt = f"""
        You are a relevance judge. Your only task is to analyze a user's query and a list of commands, and determine if ANY of the commands are relevant to the query.
        
        Here is the user's query:
        User Query: "{user_query}"
        
        ---
        Available Commands:
        """
        for cmd_data in available_commands:
            command_name = cmd_data.get("command", "N/A")
            description = cmd_data.get("description", "N/A")
            judge_prompt += f"- Command Name: {command_name}\n"
            judge_prompt += f"Description: {description}\n"
            judge_prompt += "\n\n"

        judge_prompt += """
        Based ONLY on the information above, does this list contain at ANY one command that is relevant to the user's query?

        Respond ONLY with the word 'YES' or 'NO'. Do not provide any other text.
        Your Answer: 
        """
        return judge_prompt

    def _get_llm_selection_prompt(self, user_query, available_commands):
        llm_selection_prompt = f"""
        You are a command classifier. Your only task is to analyze a user's query and select a command name from a provided list.

        Your knowledge is strictly limited to the commands and their descriptions below. You must not use any external knowledge to answer.

        Here is the user's query:
        User Query: "{user_query}"

        ---
        Available Commands:
        """
        # --- The for loop from before remains the same ---
        for idx, cmd_data in enumerate(available_commands):
            command_name = cmd_data.get("command", "N/A")
            description = cmd_data.get("description", "N/A")
            query_intent = cmd_data.get("query_intent", "N/A")

            llm_selection_prompt += f"Command Name: {command_name}\n"
            llm_selection_prompt += f"Description: {description}\n"
            llm_selection_prompt += f"Query Intent: {query_intent}\n"
            llm_selection_prompt += "---\n"
        # --- End of loop ---

        llm_selection_prompt += """
        ---
        Based **ONLY** on the information provided above, your task is to choose the single best-matching `Command Name` for the user's query.

        **Decision Rules:**
        1. **The selected command MUST be one of the exact `Command Name` strings from the list above.**
        2. **If NO command in the list is a perfect or even a highly relevant match for the user's query, you MUST respond with the exact string 'NO_MATCH'.**

        Respond ONLY with the `Command Name` or 'NO_MATCH'. DO NOT generate ANY new commands or EXPLANATORY text.

        Your Answer: """
        return llm_selection_prompt

    def _validates_LLM_halicunation(self, selected_command, results):
        # Validates the halicunation of the LLM
        # --- After you get the LLM response from the model ---
        print(f"LLM selected a valid command: {selected_command}")
        # Get a list of all available command names from your retrieved results
        available_commands = [
            cmd_data.get("command", "N/A") for cmd_data in results
            ]

        # --- The failsafe check ---
        if selected_command == "NO_MATCH":
            print("LLM correctly determined no suitable command from the list.")
            return {}, []
        elif selected_command in available_commands:
            print(f"LLM selected a valid command: {selected_command}")
            return results, selected_command
        else:
            # This block catches the exact problem you encountered
            print(f"LLM hallucinated a command: '{selected_command}'. It was not in the provided list.")
            return {}, []

    def _search_select_with_llm(self, query: str, model_choice: str):
        results = self._search_command(query=query)
        if results:
            print(f"Vector search provided the following top-{self.top_k} commands:")
            for r in results:
                print(f"[Score: {r['score']:.4f}] ➜ {r['command']}")
                print(f"  → {r['description']}\n")
        else:
            print("No relevant command found in the vector DB search.")
            return None, None

        prompt = self._get_llm_selection_prompt(query, results)
        # Step 4: Use LLM to select best matching command
        if model_choice == 'o':
            selected_command_name = self._run_llm_query_with_ollama(
                prompt,
                )
        elif model_choice == 'l':
            selected_command_name = self._run_llm_query_with_lmstudio(
                prompt
            )
        else:
            print("Invalid choice. Use 'o' for Ollama or 'l' for LM Studio.")
            return
        return self._validates_LLM_halicunation(
            selected_command=selected_command_name,
            results=results
        )