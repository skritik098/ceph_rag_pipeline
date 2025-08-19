# semantic_search.py

from utils.file_ops import vectorBuilder
from llm.llm_response import llmResponse
import json
import re


def extract_json(text):
    # Your existing extract_json function (no changes needed)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No valid JSON found in response")


class semanticCephSearch(llmResponse):
    """
    This class encapsulates the function to do a command search using RAG & LLM,
    and also BUILDS the final command if it is dynamic.
    """
    def __init__(self, vector_store: vectorBuilder, top_k: int, threshold: float, llm_model: str = "granite3.3:8b", temperature: float = 0.2) -> None:
        super().__init__(llm_model, temperature)
        self.llm_model = llm_model
        self.top_k = top_k
        self.threshold = threshold
        self.vector_store = vector_store

    def _search_command_templates(self, query: str):
        query_embedding = self.vector_store.model.encode(
            [query],
            convert_to_numpy=True
        )
        distances, indices = self.vector_store.index.search(query_embedding, self.top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            # FAISS IndexFlatIP returns dot product similarity. Higher is better.
            # A common threshold for normalized embeddings is > 0.7 or 0.8.
            # We will assume a threshold on the similarity score itself.
            if score >= self.threshold:
                matched_data = self.vector_store.metadata[int(idx)]
                # Add score for logging/debugging
                matched_data['score'] = float(score)
                results.append(matched_data)
        return results

    # --- RE-INTEGRATED: The Relevance Judge prompt, updated for templates ---
    def _get_relevance_judge_prompt(self, user_query, available_templates):
        judge_prompt = f"""
        You are a relevance judge. Your only task is to analyze a user's query and a list of potential command templates and determine if ANY of the templates are relevant.

        <user_query>
        {user_query}
        </user_query>

        <available_commands>
        """
        for template in available_templates:
            judge_prompt += f"- Command: {template.get('command_id')}\n"
            judge_prompt += f"  Description: {template.get('description')}\n\n"
        judge_prompt += "</available_commands>\n\n"

        judge_prompt += """
        Does the <available_commands> LIST contain at least one command that is highly relevant to the <user_query>?

        Respond ONLY with the word 'YES' or 'NO'. Do NOT provide any other text.
        """
        return judge_prompt

    def _get_llm_selection_prompt(self, user_query, available_templates):
        prompt = f"""
        You are an expert Ceph command selector. Your task is to analyze a user's intent and select the single best command_id from the provided list.

        <user_query>
        {user_query}
        </user_query>

        <available_commands>
        """
        for template in available_templates:
            prompt += f"""<command>
  <command_id>{template.get('command_id')}</command_id>
  <description>{template.get('description')}</description>
</command>
"""
        prompt += "</available_commands>\n\n"
        prompt += """
        **Decision Rules:**
        1. Your choice MUST be based on the command's `<description>`.
        2. If NO command is a good match, respond with 'NO_MATCH'.
        3. Respond ONLY with the chosen `<command_id>` from the above list of <available_commands> or 'NO_MATCH'. DO NOT generate ANY new commands or EXPLANATORY text.
        

        Your Answer: """
        return prompt

    def _get_command_builder_prompt(self, user_query, template):
        # This function remains the same
        return f"""
        You are a command-building assistant. Your task is to populate a command template using information from a user's query.

        **User's Goal:**
        <user_query>{user_query}</user_query>

        **Command Template:**
        <command_template>{json.dumps(template, indent=2)}</command_template>

        **Your Task:**
        Analyze the query to find values for all required parameters and construct the final command. If a required parameter is missing, identify it.

        **Response Format:**
        Respond in STRICT JSON.
        - If successful: {{"status": "success", "final_command": "<the constructed command>"}}
        - If missing parameter: {{"status": "missing_parameter", "missing_parameter_name": "<parameter name>"}}
        Your JSON Response:
        """
    '''
    def _build_final_command(self, user_query, template, model_choice):
        # This function remains the same
        print("➡️ Dynamic command detected. Engaging builder...")
        prompt = self._get_command_builder_prompt(user_query, template)
        response_str = self._run_llm_query(prompt, model_choice)

        # Sometime the LLM does not Respond back in proper JSON only response, 
        # So better to extract it.

        #response_str = extract_json(response_str)

        try:
            response_json = json.loads(response_str)
            if response_json.get("status") == "success":
                final_command = response_json.get("final_command")
                print(f"✅ Builder successful. Final command: '{final_command}'")
                return final_command
            else:
                missing_param = response_json.get("missing_parameter_name")
                print(f"🔴 Builder failed: Missing parameter '{missing_param}'.")
                return None
        except (json.JSONDecodeError, AttributeError):
            print(f"🔴 Builder failed: Could not parse LLM response for command building.")
            return None
    '''
    def _build_final_command(self, user_query, template, model_choice):
        """
        Interactively builds a final command from a template.
        If parameters are missing, it will prompt the user for them.
        """
        print("➡️ Dynamic command detected. Engaging builder...")
        
        # Start with the original user query
        cumulative_query_context = user_query
        
        # Loop until the command is built or the process is aborted
        while True:
            prompt = self._get_command_builder_prompt(cumulative_query_context, template)
            response_str = self._run_llm_query(prompt, model_choice)

            try:
                # Attempt to parse the LLM's JSON response
                response_json = json.loads(response_str)

                if response_json.get("status") == "success":
                    # --- SUCCESS CASE ---
                    final_command = response_json.get("final_command")
                    print(f"✅ Builder successful. Final command: '{final_command}'")
                    return final_command  # Exit the function with the result

                elif response_json.get("status") == "missing_parameter":
                    # --- MISSING PARAMETER CASE ---
                    missing_param_name = response_json.get("missing_parameter_name")
                    
                    # Find the description for the missing parameter for a better prompt
                    param_details = next((p for p in template.get("parameters", []) if p['name'] == missing_param_name), None)
                    param_description = param_details.get('description', f"the value for '{missing_param_name}'") if param_details else f"the value for '{missing_param_name}'"

                    print(f"⚠️ Builder needs more information.")
                    
                    # Ask the user for the missing information
                    user_input = input(f"Please provide {param_description} (or type 'cancel' to abort): ").strip()

                    if user_input.lower() == 'cancel':
                        print("🔴 Command building aborted by user.")
                        return None # Exit the function

                    # Add the new information to the context for the next loop iteration
                    cumulative_query_context += f"\n\nTo answer a follow-up question, the user provided this value for '{missing_param_name}': '{user_input}'"
                    print("🔄 Re-engaging builder with new information...")

                else:
                    # Handle other potential statuses from the LLM
                    print(f"🔴 Builder returned an unknown status: {response_json.get('status')}")
                    return None

            except (json.JSONDecodeError, AttributeError):
                print(f"🔴 Builder failed: Could not parse LLM response for command building.")
                return None # Exit on a fatal error


    def _run_llm_query(self, prompt, model_choice):
        # This should be implemented in your parent class as per your original file
        if model_choice == 'o':
            return self._run_llm_query_with_ollama(
                prompt,
                )
        elif model_choice == 'l':
            return self._run_llm_query_with_lmstudio(
                prompt
            )
        else:
            print("Invalid choice. Use 'o' for Ollama or 'l' for LM Studio.")
            return

    # --- UPDATED: The main workflow now includes the Relevance Judge ---
    def search_select_and_build(self, query: str, model_choice: str):
        # STAGE 1: SEARCH for templates
        templates = self._search_command_templates(query=query)
        if not templates:
            print("INFO: No relevant command templates found in vector search.")
            return {}, []
        
        print(f"Vector search found {len(templates)} potential templates:")
        for t in templates:
            print(f"[Score: {t.get('score', 0):.4f}] ➜ {t.get('command_id')}")

        # STAGE 2: JUDGE relevance
        judge_prompt = self._get_relevance_judge_prompt(query, templates)
        relevance_response = self._run_llm_query(judge_prompt, model_choice).strip().upper()

        if "NO" in relevance_response:
            print("INFO: Relevance Judge determined no commands are suitable. Stopping.")
            return {}, []
            
        print("INFO: Relevance Judge confirmed potential match. Proceeding to selection.")

        # STAGE 3: SELECT the best template
        selection_prompt = self._get_llm_selection_prompt(query, templates)
        selected_id = self._run_llm_query(selection_prompt, model_choice).strip()

        if selected_id == "NO_MATCH" or not selected_id:
            print("INFO: LLM Selector determined no suitable command template from the list.")
            return {}, []
            
        selected_template = next((t for t in templates if t.get('command_id') == selected_id), None)
        
        if not selected_template:
            print(f"WARNING: LLM selected an invalid command_id '{selected_id}'.")
            return {}, []
            
        print(f"INFO: LLM selected template: '{selected_id}'")

        # STAGE 4: BUILD the final command
        if "parameters" in selected_template and selected_template["parameters"]:
            return self._build_final_command(query, selected_template, model_choice), templates
        else:
            final_command = selected_template.get("base_command")
            print(f"✅ Simple command selected. Final command: '{final_command}'")
            return final_command, templates