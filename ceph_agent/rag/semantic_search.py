# --------------------
# Phase 3: Semantic Search
# --------------------

from utils.file_ops import vectorBuilder
from llm.llm_response import llmResponse


class semanticCephSearch(llmResponse):
    """
    This class encapsulate the function to do a command search using RAG & LLM
    """
    def __init__(
        self,
        vector_store: vectorBuilder,
        top_k: int,
        threshold: float, # UPDATED: Threshold should be a float for similarity scores
        llm_model: str = "granite3.3:8b",
        temperature: float = 0.2
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
            # Similarity scores are often 0-1, where lower is better. Adjust if using cosine similarity.
            if score <= self.threshold:
                matched_data = self.vector_store.metadata[int(idx)]
                results.append({
                    "score": float(score),
                    "command": matched_data["command"],
                    "description": matched_data["description"],
                    "query_intent": matched_data["query_intent"]
                })
        return results
    
    def _get_relevance_judge_prompt(self, user_query, available_commands):
        # This prompt is good, no changes needed, but we will now use it.
        judge_prompt = f"""
        You are a relevance judge. Your only task is to analyze a user's query and a list of potential commands and determine if ANY of the commands are relevant.

        <user_query>
        "{user_query}"
        </user_query>

        <available_commands>
        """
        for cmd_data in available_commands:
            judge_prompt += f"- Command: {cmd_data.get('command')}\n"
            judge_prompt += f"  Description: {cmd_data.get('description')}\n\n"
        judge_prompt += "</available_commands>\n\n"

        judge_prompt += """
        Does the <available_commands> list contain at least one command that is highly relevant to the <user_query>?
        Respond ONLY with the word 'YES' or 'NO'.
        """
        return judge_prompt

    # UPDATED: The selection prompt is heavily revised to focus on intent.
    def _get_llm_selection_prompt(self, user_query, available_commands):
        llm_selection_prompt = f"""
        You are an expert Ceph command selector. Your task is to analyze a user's intent and select the single best command from a provided list that fulfills that intent.

        **Step 1: Analyze the User's Intent**
        First, understand what the user is trying to accomplish, ignoring any specific commands they might have mentioned.

        <user_query>
        "{user_query}"
        </user_query>

        **Step 2: Select the Best Command**
        Review the following commands and choose the one whose description best matches the user's intent.

        <available_commands>
        """
        for cmd_data in available_commands:
            command_name = cmd_data.get("command", "N/A")
            description = cmd_data.get("description", "N/A")
            llm_selection_prompt += f"<command>\n"
            llm_selection_prompt += f"  <name>{command_name}</name>\n"
            llm_selection_prompt += f"  <description>{description}</description>\n"
            llm_selection_prompt += f"</command>\n"
        llm_selection_prompt += "</available_commands>\n\n"

        llm_selection_prompt += """
        **Decision Rules:**
        1. Your choice MUST be based on the command's `<description>`, not on whether its `<name>` appears in the user query.
        2. The selected command MUST be one of the exact `<name>` strings from the list.
        3. If NO command is a highly relevant match for the user's intent, you MUST respond with the exact string 'NO_MATCH'.

        Respond ONLY with the chosen `<name>` or 'NO_MATCH'. Do not provide any explanation.

        Your Answer: """
        return llm_selection_prompt
    
    # UPDATED: Renamed for clarity.
    def _validate_llm_selection(self, selected_command: str, results: list):
        # Get a list of all available command names from your retrieved results
        available_commands = [cmd_data.get("command") for cmd_data in results]

        # Failsafe check
        if selected_command == "NO_MATCH":
            print("INFO: LLM correctly determined no suitable command from the list.")
            return None, None
        elif selected_command in available_commands:
            print(f"INFO: LLM selected a valid command: {selected_command}")
            return results, selected_command
        else:
            print(f"WARNING: LLM hallucinated a command: '{selected_command}'. It was not in the provided list.")
            return None, None

    # NEW: Helper function to reduce code duplication.
    def _run_llm_query(self, prompt: str, model_choice: str) -> str:
        if model_choice == 'o':
            return self._run_llm_query_with_ollama(prompt)
        elif model_choice == 'l':
            return self._run_llm_query_with_lmstudio(prompt)
        else:
            print(f"Invalid model choice: {model_choice}")
            return ""

    # UPDATED: The main workflow now uses the two-stage chain.
    # Also made it a public method by removing the leading underscore.
    def search_and_select(self, query: str, model_choice: str):
        results = self._search_command(query=query)
        if not results:
            print("INFO: No relevant commands found in the vector DB search.")
            return None, None

        print(f"Vector search provided the following top-{self.top_k} commands:")
        for r in results:
            print(f"[Score: {r['score']:.4f}] ➜ {r['command']}")

        # --- STAGE 1: Relevance Judge ---
        judge_prompt = self._get_relevance_judge_prompt(query, results)
        relevance_response = self._run_llm_query(judge_prompt, model_choice).strip().upper()
        
        if "NO" in relevance_response:
            print("INFO: Relevance Judge determined no commands are suitable. Stopping.")
            return None, None
        
        print("INFO: Relevance Judge confirmed potential match. Proceeding to selection.")

        # --- STAGE 2: Command Selector ---
        selection_prompt = self._get_llm_selection_prompt(query, results)
        selected_command_name = self._run_llm_query(selection_prompt, model_choice).strip()

        return self._validate_llm_selection(
            selected_command=selected_command_name,
            results=results
        )