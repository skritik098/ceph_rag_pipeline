from core.agent_logic import analysePrompt
from rag.semantic_search import semanticCephSearch
from utils.file_ops import vectorBuilder
from ceph.executor import execute_command
import os
import re
import json
import ollama


def userSystemPrompt() -> str:
    prompt = """
    You are an expert assistant for a Ceph AI agent.  
    Your task is to:  

    1. Classify the user query into one of two modes:  
    - **Planning Mode** → The query is about "how to" instructions, troubleshooting, workflows, or multi-step processes. In this case, break down the query into a sequence of steps, where each step will later map to commands via vector search.
    - **Direct Mode** → The query is a simple lookup or single command request. In this case, send the query directly to vector search without breaking it down.  

    2. Detect whether the query involves **destructive or high-risk actions** (delete, remove, purge, shutdown).  
    - If yes, mark it as `"unsafe"`. Do NOT suggest commands.  
    - If no, continue as normal.  

    3. When in **Planning Mode**, only return **logical task steps** 
    (e.g., "Check OSD status") and **NOT** raw commands. Commands will be retrieved later.  

    4. Always respond in **strict JSON only**. Do not include any extra text outside the JSON.  

    The response must match exactly this schema:  

    ```json: response:
    {
    "mode": "planning" | "direct",
    "safety": "safe" | "unsafe",
    "reasoning": "Short explanation of classification",
    "steps": [
        "If planning: high-level logical steps only",
        "If direct: leave empty"
    ],
    "warning": "Only if unsafe, else empty"
    }
    """
    return prompt


def extract_json(text):
    """Extract first JSON object from LLM output safely"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No valid JSON found in response")


def search_and_execute(cephSearch, user_query, model_choice, execution_context):
    print("Searching for command...")
    print("Results:")
    print("--------------------------------")

    # Step-4: Search & Select the best available command
    vect_results, selected_command = cephSearch._search_select_with_llm(
        query=user_query,
        model_choice=model_choice
    )

    if not selected_command:
        print(
            "Agent Response: I couldn't find a suitable Ceph command for your "
            "query in my knowledge base."
        )
        return execution_context
    # Step-5: Execute the selected command on the ceph cluster
    stdout, stderr, retcode = execute_command(selected_command.strip())

    print(stdout)

    # Store the output in the context for future steps
    current_step_key = f"step_{len(execution_context) + 1}"
    execution_context[current_step_key] = {
        "query": user_query,
        "command": selected_command.strip(),
        "stdout": stdout,
        "stderr": stderr,
        "returncode": retcode
    }

    # Step-6: Analyse the output as per the user_query
    # To get the description for 'docker_build'
    if retcode != 0:
        agent_response = (
            f"Agent Response: I executed '{selected_command}', but it "
            f"returned an error.\n"
            f"Error details: {stderr if stderr else 'No specific error message.'}\n"
            "Please check the command or your Ceph environment."
        )
    else:
        # Step 5: Output Analysis & Response Generation Module
        # The selected_purpose_description can be extracted from selected_command_data
        # if you have a specific description of *why* this command was chosen.
        # For simplicity, let's just use the main description text if available.
        description = next((item['description'] for item in vect_results if item['command'] == selected_command.strip()), 'Description not found.')
        agent = analysePrompt(
            query=user_query,
            selected_command=selected_command,
            command_out=stdout,
            command_description=description,
            model_choice=model_choice
        )

        agent_response = agent._analyze_response()

        if not agent_response:
            agent_response = (
                "Agent Response: I executed the command, but could not extract "
                "a clear answer from its output based on your query."
            )
        print(f"\nAgent Response: {agent_response}")
        print("\n------------------------------------")

        return execution_context


def main():
    print("Initializing Ceph Agent...")
    print("--------------------------------")
    # Set the environment variable to false to disable parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Step-1: Loading the Database
    vector_store = vectorBuilder(
        json_path="./database/basic_commands.json",
        model_name="all-MiniLM-L6-v2",
        index_path="./faiss_index_store/ceph_faiss.index",
        metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
    )
    # Either retrieve the below parameters as it is as it is already 
    # retrieved as part of initialisation
    #index, metadata, model = vector_store._load_index()

    cephSearch = semanticCephSearch(
        vector_store=vector_store,
        llm_model="granite3.3:8b",
        top_k=3,
        threshold=0.9
    )

    while True:
        # Step-2: Take the user query input
        user_query = input(
            "\nYour Ceph Query (e.g., 'check cluster health'): "
        ).strip()

        if user_query.lower() in ['exit', 'quit']:
            print("Exiting Ceph Agent. Goodbye!")
            break
        
        # Step 3: Pick which LLM to use
        model_choice = input("Use Ollama or LM Studio? (o/l): ").strip().lower()
        print("\n--- Processing Query ---")

        system_prompt = userSystemPrompt()
        if model_choice == 'o':
            decisionResponse = ollama.chat(
                            model="granite3.3:8b",
                            messages=[
                                {
                                    "role": "user",
                                    "content": user_query
                                },
                                {
                                    "role": "system",
                                    "content": system_prompt
                                },
                            ]
                        )["message"]["content"].strip()
        else:
            return

        print(decisionResponse)
        modeResponse = extract_json(decisionResponse)

        if modeResponse["safety"] == "unsafe":
            print("⚠️ Unsafe operation detected:", modeResponse["warning"])
            continue
        elif modeResponse["mode"] == "direct":
            print(f"User Query: {user_query}")
            search_and_execute(
                cephSearch=cephSearch,
                user_query=user_query,
                model_choice=model_choice,
                execution_context={}
            )
        else:
            print(f"User Query: {user_query}")
            # 1. Initialize the shared context for this plan
            execution_context = {}
            for i, step in enumerate(modeResponse["steps"]):
                print(f"--------- Step {i}: {step}--------")

                # 2. Create a rich query that includes historical context
                # This helps the LLM find a command relevant to the *current* state
                contextual_query = f"""
                Previous Steps and Outputs: {json.dumps(execution_context, indent=2)}
                Current Goal: "{step}"
                """

                # 3. Call search_and_execute and update the context
                execution_context = search_and_execute(
                    cephSearch=cephSearch,
                    user_query=contextual_query,  # Use the new contextual query
                    model_choice=model_choice,
                    execution_context=execution_context
                )

                # Optional: Add a check to stop if a step failed
                last_step_key = f"step_{len(execution_context)}"
                if execution_context.get(last_step_key, {}).get("returncode", 0) != 0:
                    print(f"⚠️ Step {i + 1} failed. Aborting plan execution.")
                    break
            
            print("\n--- Plan Execution Finished ---")
            print("Final Execution Context:", json.dumps(execution_context, indent=2))



if __name__ == "__main__":
    main()