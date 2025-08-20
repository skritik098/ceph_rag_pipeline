from core.agent_logic import analysePrompt
from rag.semantic_search import semanticCephSearch
from utils.file_ops import vectorBuilder
from ceph.executor import execute_command
import os
import re
import json
import ollama

# --- Agent Definitions ---


class RetrieverAgent:
    """Finds the best command for a given query."""
    def __init__(self, ceph_search_instance):
        self.ceph_search = ceph_search_instance

    def find_command(self, query: str, model_choice: str) -> (str, list):
        print("➡️ RetrieverAgent: Searching for command...")
        selected_command, vect_results = self.ceph_search.search_select_and_build(
            query=query,
            model_choice=model_choice
        )
        if not selected_command:
            print("🔴 RetrieverAgent: Could not find a suitable command.")
            return None, []
        
        print(f"✅ RetrieverAgent: Found command: '{selected_command.strip()}'")
        return selected_command.strip(), vect_results


class ExecutorAgent:
    """Executes a command on the Ceph cluster."""
    def run(self, command: str) -> (str, str, int):
        print(f"➡️ ExecutorAgent: Running command: '{command}'")
        stdout, stderr, retcode = execute_command(command)
        if retcode != 0:
            print(f"🔴 ExecutorAgent: Command failed with return code {retcode}.")
        else:
            print("✅ ExecutorAgent: Command executed successfully.")
        return stdout, stderr, retcode


class AnalyzerAgent:
    """Analyzes command output to generate a final response."""
    def analyze(self, query: str, command: str, command_out: str, vect_results: list, model_choice: str) -> str:
        print("➡️ AnalyzerAgent: Analyzing command output...")

        #description = next((item['description'] for item in vect_results if item['command'] == command), 'Description not found.')
        
        # --- CORRECTED LOGIC ---
        # Find the template where the final 'command' string starts with the template's 'base_command'.
        # This correctly handles dynamic commands with parameters.
        description = next(
            (item['description'] for item in vect_results if command.startswith(item['base_command'])), 
            'Description not found.'
        )

        agent = analysePrompt(
            query=query,
            selected_command=command,
            command_out=command_out,
            command_description=description,
            model_choice=model_choice
        )

        agent_response = agent._analyze_response()

        if not agent_response:
            agent_response = "I executed the command, but could not extract a clear answer from its output."

        print("✅ AnalyzerAgent: Analysis complete.")
        return agent_response


# --- Utility Functions ---
'''
def userSystemPrompt() -> str:
    prompt = """
    You are an expert assistant for a Ceph Command AI agent. You are a high-level PLANNER. Your job is to analyze a user's goal and create a plan. Another agent will be responsible for finding the specific commands later.

    1.  **Classify the user query into one of two modes:**
        -   **Direct Mode:** The query can be answered with a **single command**. This includes most "check," "get," "list," or "show" requests.
            -   *Example Direct Queries:* "check cluster health", "what is the status of the OSDs?", "list all the pools".
        -   **Planning Mode:** The query requires **multiple, sequential commands** to achieve a final goal. This is for complex workflows, troubleshooting, or tasks with dependencies.
            -   *Example Planning Queries:* "Create a new RBD image and map it to a host", "Find all inactive PGs and attempt to repair them".

    2.  **Detect Destructive Actions:**
        -   If the query involves high-risk actions (delete, remove, purge, shutdown), mark it as `"safety": "unsafe"`.
        -   Otherwise, mark it as `"safety": "safe"`.

    3.  **CRITICAL RULE for Planning Mode Steps:**
        -   Steps **MUST** be high-level goals described in natural language.
        -   Under NO circumstances should you ever include a raw command (like "ceph osd tree" or "rbd create") in the "steps" array. Your role is to define WHAT to do, not HOW to do it.

    4.  **Good vs. Bad Step Examples:**
        -   **BAD Step (Vague/GUI-based):** "Navigate to the Ceph cluster management interface."
        -   **BAD Step (Contains a command):** "Run 'ceph health' to see the status."
        -   **GOOD Step (Clear CLI Goal):** "Check the overall health of the cluster."
        -   **GOOD Step (Clear CLI Goal):** "Identify all unhealthy OSDs."

    5.  **Respond in STRICT JSON only.** The response must match this schema exactly:
        ```json
        {
          "mode": "planning" | "direct",
          "safety": "safe" | "unsafe",
          "reasoning": "Short explanation of your classification and plan.",
          "steps": [
            "If planning: natural language goals only. NO COMMANDS.",
            "If direct: leave empty."
          ],
          "warning": "Only if unsafe, else empty."
        }
        ```
    """
    return prompt
'''


def userSystemPrompt() -> str:
    prompt = """
    You are an expert assistant for a Ceph AI agent, specialized in Ceph storage operations (e.g., OSDs, pools, CRUSH maps). Your task is to:

    1. Always generate a plan for the user query, using the **minimal number of high-level logical steps** needed to address it. 
    - For simple queries (e.g., status checks, listings), aim for 1-2 steps.
    - For complex queries (e.g., troubleshooting, recovery), include only essential steps, avoiding unnecessary checks.
    - Steps should be high-level (e.g., "Check OSD status") and **not** raw commands (e.g., do not include "ceph osd status").

    2. Detect whether the query involves **destructive or high-risk actions** (e.g., delete, remove, purge, shutdown, rm, out, down). 
    - If yes, mark as "unsafe" and return an empty plan with a warning.
    - If no, generate the minimal plan.

    3. Reason step-by-step: First, interpret the query’s intent in Ceph context. Second, identify the minimal steps needed. Third, check for destructive keywords to assess safety.

    4.  **CRITICAL RULE for Planning Mode Steps:**
        -   Steps **MUST** be high-level goals described in natural language.
        -   Under NO circumstances should you ever include a raw command (like "ceph osd tree" or "rbd create") in the "steps" array. Your role is to define WHAT to do, not HOW to do it.

    5.  **Good vs. Bad Step Examples:**
        -   **BAD Step (Vague/GUI-based):** "Navigate to the Ceph cluster management interface."
        -   **BAD Step (Contains a command):** "Run 'ceph health' to see the status."
        -   **GOOD Step (Clear CLI Goal):** "Check the overall health of the cluster."
        -   **GOOD Step (Clear CLI Goal):** "Identify all unhealthy OSDs."


    6. Always respond in **strict JSON only**, matching the schema below. Do not include text outside the JSON.

    Examples:
    - Query: "What is the status of OSD 5?"
    Response: {
        "mode": "planning",
        "safety": "safe",
        "reasoning": "Simple status check requiring one step.",
        "steps": ["Check OSD status"],
        "warning": ""
    }

    - Query: "How do I fix a down OSD?"
    Response: {
        "mode": "planning",
        "safety": "safe",
        "reasoning": "Troubleshooting a down OSD requires minimal diagnostic and repair steps.",
        "steps": ["Check OSD status", "Review OSD logs", "Attempt restart if safe"],
        "warning": ""
    }

    - Query: "Delete pool mypool"
    Response: {
        "mode": "planning",
        "safety": "unsafe",
        "reasoning": "Query involves destructive pool deletion.",
        "steps": [],
        "warning": "This action is destructive and could lead to data loss. Confirmation required."
    }

    - Query: "Troubleshoot slow requests in the cluster"
    Response: {
        "mode": "planning",
        "safety": "safe",
        "reasoning": "Slow requests require minimal diagnostic steps to identify bottlenecks.",
        "steps": ["Monitor cluster health", "Check PG states", "Analyze OSD performance"],
        "warning": ""
    }

    Response schema:
    {
    "mode": "planning",
    "safety": "safe" | "unsafe",
    "reasoning": "Short explanation of plan and safety",
    "steps": [
        "High-level logical steps, minimal and essential only"
    ],
    "warning": "Only if unsafe, else empty"
    }
    """

    return prompt

def extract_json(text):
    # Your existing extract_json function (no changes needed)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No valid JSON found in response")


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
        json_path="./database/complex_commands.json",
        model_name="all-MiniLM-L6-v2",
        index_path="./faiss_index_store/ceph_faiss.index",
        metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
    )

    cephSearch = semanticCephSearch(
        vector_store=vector_store,
        llm_model="granite3.3:8b",
        top_k=3,
        threshold=0.5
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
            plan_successful = True  # Flag to track plan success
            print(steps)

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