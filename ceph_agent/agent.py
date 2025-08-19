from rag.semantic_search import semanticCephSearch
from utils.file_ops import vectorBuilder
from agent.agentsList import RetrieverAgent, ExecutorAgent, AnalyzerAgent
from utils.utilities import userSystemPrompt, extract_json
import os
import json
import ollama


# --- Main Controller ---
def main():
    """
    The Controller.
    Orchestrates the workflow by managing agents and state.
    """
    print("Initializing Ceph Agent Controller...")
    print("--------------------------------")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Step 1: Initialize Vector Store and Agents ---
    vector_store = vectorBuilder(
        json_path="./database/basic_commands.json",
        model_name="all-MiniLM-L6-v2",
        index_path="./faiss_index_store/ceph_faiss.index",
        metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
    )

    cephSearch = semanticCephSearch(
        vector_store=vector_store,
        llm_model="granite3.3:8b",
        top_k=3,
        threshold=0.9
    )

    # Instantiate our specialized agents
    retriever = RetrieverAgent(cephSearch)
    executor = ExecutorAgent()
    analyzer = AnalyzerAgent()

    while True:
        # --- Step 2: Get User Input ---
        user_query = input("\nYour Ceph Query (e.g., 'check cluster health'): ").strip()
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting Ceph Agent. Goodbye!")
            break

        model_choice = input("Use Ollama or LM Studio? (o/l): ").strip().lower()
        print("\n--- Processing Query ---")

        # --- Step 3: Classify the Task (Controller Logic) ---
        system_prompt = userSystemPrompt()
        try:
            modeResponse = extract_json(
                ollama.chat(
                    model="granite3.3:8b",
                    messages=[
                        {"role": "user", "content": user_query},
                        {"role": "system", "content": system_prompt},
                    ]
                )["message"]["content"].strip()
            )
        except (ValueError, json.JSONDecodeError) as e:
            print(f"🔴 Controller: Could not parse LLM response for classification. Error: {e}")
            continue

        if modeResponse.get("safety") == "unsafe":
            print(f"⚠️ Controller: Unsafe operation detected. {modeResponse.get('warning', '')}")
            continue

        # --- Step 4: Orchestrate Agent Workflow ---
        if modeResponse.get("mode") == "direct":
            print(f"🕹️ Controller: Direct Mode. Executing single task for '{user_query}'")
            command, vect_results = retriever.find_command(user_query, model_choice)
            if command:
                stdout, stderr, retcode = executor.run(command)
                if retcode == 0:
                    final_response = analyzer.analyze(user_query, command, stdout, vect_results, model_choice)
                    print(f"\n💡 Agent Response: {final_response}")
                else:
                    print(f"\n💡 Agent Response: I executed '{command}', but it failed. Error: {stderr}")

        elif modeResponse.get("mode") == "planning":
            print(f"🗺️ Controller: Planning Mode. Executing plan for '{user_query}'")
            execution_context = {}
            steps = modeResponse.get("steps", [])
            print(steps.join(".\n"))
            plan_successful = True  # Flag to track plan success
            
            for i, step_goal in enumerate(steps):
                print(f"\n--------- Executing Step {i + 1}: {step_goal} --------")
                
                contextual_query = f"""
                Original User Goal: "{user_query}"
                Previous Steps and Summarized Outputs: {json.dumps(execution_context, indent=2)}
                Current Goal: "{step_goal}"
                """
                
                command, vect_results = retriever.find_command(contextual_query, model_choice)
                if command:
                    stdout, stderr, retcode = executor.run(command)
                    
                    if retcode == 0:
                        # UPDATED: Analyze the output and store the SUMMARY in the context.
                        step_response = analyzer.analyze(step_goal, command, stdout, vect_results, model_choice)
                        print(f"✅ Step {i + 1} Summary: {step_response}")
                        execution_context[f"step_{i+1}"] = {
                            "goal": step_goal,
                            "command": command,
                            "summary": step_response # Store the concise summary
                        }
                    else:
                        # Handle step failure
                        print(f"🔴 Step {i + 1} failed. Aborting plan.")
                        execution_context[f"step_{i+1}"] = {"goal": step_goal, "command": command, "error": stderr}
                        plan_successful = False
                        break
                else:
                    print(f"🔴 Could not find a command for step '{step_goal}'. Aborting plan.")
                    plan_successful = False
                    break
            
            print("\n--- Plan Execution Finished ---")

            # UPDATED: Add the final synthesis step.
            if plan_successful:
                print("➡️ Synthesizing final answer from plan results...")
                synthesis_prompt = f"""
                The user's original query was: "{user_query}"
                A multi-step plan was executed. Here are the summaries of what was done in each step:
                {json.dumps(execution_context, indent=2)}
                
                Based on the results of these steps, provide a comprehensive final answer to the user's original query.
                """
                # We can reuse the analyzer's LLM call for this.
                # Here we pass the synthesis prompt as the "query" to the analyzer's underlying LLM.
                #final_answer = analyzer.agent.llm.invoke(synthesis_prompt) # You may need to expose the llm call from the analyzer agent.
                # A simpler way if you don't want to modify the analyzer:
                final_answer = ollama.chat(
                    model="granite3.3:8b",
                    messages=[{
                        'role': 'user',
                        'content': synthesis_prompt
                    }])['message']['content']
                print(f"\n✅ Final Answer: {final_answer}")
            else:
                print("Plan failed. Final context log:")
                print(json.dumps(execution_context, indent=2))


if __name__ == "__main__":
    main()