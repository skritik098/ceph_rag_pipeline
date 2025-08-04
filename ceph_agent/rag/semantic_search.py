# --------------------
# Phase 3: Semantic Search
# --------------------

from llm.llm_response import run_llm_query_with_ollama


def get_relevance_judge_prompt(user_query, available_commands):
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
Your Answer: """
    return judge_prompt


def get_llm_selection_prompt(user_query, available_commands):
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

    llm_selection_prompt += f"""
---
Based **ONLY** on the information provided above, your task is to choose the single best-matching `Command Name` for the user's query.

**Decision Rules:**
1. **The selected command MUST be one of the exact `Command Name` strings from the list above.**
2. **If NO command in the list is a perfect or even a highly relevant match for the user's query, you MUST respond with the exact string 'NO_MATCH'.**

Respond ONLY with the `Command Name` or 'NO_MATCH'. DO NOT generate ANY new commands or EXPLANATORY text.

Your Answer: """
    return llm_selection_prompt


def search_and_select_command_with_llm(
    index,
    metadata,
    model,
    query,
    model_choice,
    llm_model,
    top_k=3,
    threshold=0.9
):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if score <= threshold:  # Only include if similarity is good enough
            matched_data = metadata[int(idx)]
            results.append({
                "score": float(score),
                "command": matched_data["command"],
                "description": matched_data["description"],
                "query_intent": matched_data["query_intent"]
            })
    
    if results:
        print(f"Vector search provided the following top-{top_k} commands:")
        for r in results:
            print(f"[Score: {r['score']:.4f}] ➜ {r['command']}")
            print(f"  → {r['description']}\n")
    else:
        print("No relevant command found with sufficient confidence.")
        return None, None

    llm_selection_prompt = get_llm_selection_prompt(query, results)
    #print(llm_selection_prompt)

    #judgment_prompt = get_relevance_judge_prompt(query, results)
    prompt = llm_selection_prompt
    print(prompt)

    # Step 4: Use LLM to select best matching command
    if model_choice == 'o':
        #selected_command = select_command_with_ollama(user_query, filtered_commands, model="llama3")
        selected_command_name = run_llm_query_with_ollama(prompt, model=llm_model)
    elif model_choice == 'l':
        #selected_command = select_command_with_lmstudio(user_query, filtered_commands)
        selected_command_name = run_llm_query_with_lmstudio(prompt)
    else:
        print("Invalid choice. Use 'o' for Ollama or 'l' for LM Studio.")
        return
    
    # Validates the halicunation of the LLM
    # --- After you get the LLM response from the model ---
    print(f"LLM selected a valid command: {selected_command_name}")
    # Get a list of all available command names from your retrieved results
    available_command_names = [cmd_data.get("command", "N/A") for cmd_data in results]

    # --- The failsafe check ---
    if selected_command_name == "NO_MATCH":
        print("LLM correctly determined no suitable command from the list.")
        return {}, []
    elif selected_command_name in available_command_names:
        print(f"LLM selected a valid command: {selected_command_name}")
        return results, selected_command_name
        # ... (proceed with your logic to find and return the command data) ...
    else:
        # This block catches the exact problem you encountered
        print(f"LLM hallucinated a command: '{selected_command_name}'. It was not in the provided list.")
        return {}, []


def search_command(index, metadata, model, query, top_k=3, threshold=0.9):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    '''
    for idx in indices[0]:
        entry = metadata[idx]
        results.append({
            "command": entry["command"],
            "description": entry["description"],
            "matched_on": entry["query_intent"]
        })
    return results
    '''
    for score, idx in zip(distances[0], indices[0]):
        if score <= threshold:  # Only include if similarity is good enough
            matched_data = metadata[int(idx)]
            results.append({
                "score": float(score),
                "command": matched_data["command"],
                "description": matched_data["description"],
                "query_intent": matched_data["query_intent"]
            })
    return results