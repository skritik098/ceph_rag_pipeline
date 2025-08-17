from agent.agentTools import analyze_response_tool, final_answer_tool, search_command_tool
from core.agent_logic import analysePrompt
from rag.semantic_search import semanticCephSearch
from utils.file_ops import vectorBuilder
from ceph.executor import execute_command
import os
import re

# using Chat object from OpenAI
#from langchain_openai import ChatOpenAI

# Using the Chat object for Ollama
from langchain_ollama import ChatOllama

from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_xml_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents.output_parsers.xml import XMLAgentOutputParser



def create_manager_agent(llm):
    """Creates the Manager agent responsible for planning."""
    manager_prompt_text = """
    You are the Manager. Your role is to take a high-level user request and break it down into a simple, step-by-step, numbered list of tasks for your Worker.
    Do not find or execute commands yourself. Your only output should be the numbered plan.

    Example:
    User Request: What is wrong with the cluster?
    Your Plan:
    1. Check the overall cluster health.
    2. Identify any OSDs that are marked as down.
    3. Get the list of pools in the cluster.
    4. Report a summary of the findings.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", manager_prompt_text),
            ("human", "Create a plan for this request: {input}"),
        ]
    )
    chain = prompt | llm
    return chain

'''
def create_worker_agent(llm, tools):
    """Creates the Worker agent responsible for executing a single task."""
    worker_prompt_text = """
    You are the Worker. You have one job: execute the single task you are given, perfectly.
    Use your tools, `run_retriever` and `execute_command`, in sequence to complete the task.
    Think step-by-step and show your work using the required XML format.
    Your final answer should be the direct result or observation from executing the command.

    You have access to the following tools:
    {tools}

    You must always use the following format:

    
    <tool>
    <tool_name>the tool to use</tool_name>
    <tool_input>the input to the tool</tool_input>
    </tool>
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", worker_prompt_text),
            ("human", "Complete this task: {input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_xml_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

'''


def create_worker_chain(llm):
    """Creates a custom LCEL chain to act as our Worker."""
    worker_prompt_text = """
    You are the Worker. You have one job: based on the task, decide which tool to use and what its input should be.
    You must output your decision in the specified XML format. Do not add any other text.

    Here are the available tools:
    <tools>
        <tool>
            <name>find_ceph_command</name>
            <description>Finds the single most relevant Ceph command from the knowledge base for a given task description.</description>
        </tool>
        <tool>
            <name>execute_command</name>
            <description>Executes a valid Ceph command string on the remote cluster via SSH.</description>
        </tool>
    </tools>

    You must always use the following format:
    <tool>
    <tool_name>the tool to use</tool_name>
    <tool_input>the input to the tool</tool_input>
    </tool>
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", worker_prompt_text),
            ("human", "Execute this task: {input}"),
        ]
    )
    # This LCEL chain pipes the input to the prompt, then to the LLM, then to the XML parser.
    chain = prompt | llm | XMLAgentOutputParser()
    return chain

# --- MAIN ORCHESTRATION LOGIC ---


def main():
    print("Initializing Hierarchical Agent System...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    llm = ChatOllama(model="granite3.3:8b", temperature=0)
    
        # Step-1: Loading the Database
    vector_store = vectorBuilder(
        json_path="./database/basic_commands.json",
        model_name="all-MiniLM-L6-v2",
        index_path="./faiss_index_store/ceph_faiss.index",
        metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
    )
    
    # Either retrieve the below parameters as it is as it is already 
    # retrieved as part of initialisation
    cephSearch = semanticCephSearch(
            vector_store=vector_store,
            top_k=3,
            threshold=0.9
        )

    cephSearchTool = search_command_tool(cephSearch=cephSearch)

    # Initialize the agenPrompt object:
    agentPromptTool = analyze_response_tool(
        analysePrompt()
    )

    # Setup tools
    cephSearchTool.name = "find_ceph_command"
    worker_tools = [cephSearchTool, execute_command, final_answer_tool]

    # Create the agents -- Create the manager and the new worker chain
    manager = create_manager_agent(llm)
    worker_chain = create_worker_chain(llm)

    while True:
        user_query = input("\nYour Ceph Query (e.g., 'What's wrong with my cluster?'): ").strip()
        if user_query.lower() in ['exit', 'quit']:
            break
    
        # Step 3: Pick which LLM to use
        model_choice = input("Use Ollama or LM Studio? (o/l): ").strip().lower()
        if model_choice not in ['o', 'l']:
            print("Invalid choice. Going to default to 'o' -> Ollama")
            model_choice = 'o'

        cephSearch._set_model_choice(model_choice)

        print("\n MGR 🧠: Thinking and creating a plan...")
        # 1. Manager creates the plan
        plan_str = manager.invoke({"input": user_query}).content
        print(f" MGR 🧠: Here is the plan:\n{plan_str}")

        # Parse the plan into a list of tasks
        tasks = [task.strip() for task in re.findall(r'^\d+\.\s*(.*)', plan_str, re.MULTILINE)]
        if not tasks:
            print(" MGR 🧠: I could not create a plan for that request.")
            continue
        
        # 2. Worker executes each task in the plan
        final_results = []
        # This variable will hold the output of the previous step, to be used by the next.
        previous_step_output = ""

        for i, task in enumerate(tasks):
            print(f"\n--- Executing Task {i+1}/{len(tasks)}: {task} ---")
            try:
                # We add the previous step's output to the task description for context.
                task_with_context = task
                if previous_step_output:
                    task_with_context += f"\n\nContext from previous step: {previous_step_output}"

                # 1. The worker chain decides which tool to use.
                tool_call = worker_chain.invoke({"input": task_with_context})
                
                tool_name = tool_call.tool
                tool_input = tool_call.tool_input
                
                print(f"  ↪ Worker decided to use tool '{tool_name}' with input '{tool_input}'")
                # 2. We execute the tool call manually.
                if tool_name in tools:
                    tool_function = tools[tool_name]
                    result = tool_function.invoke(tool_input)
                    previous_step_output = result # Save result for the next step
                    final_results.append(result)
                    print(f"--- Task Complete. Result: {result} ---")
                else:
                    print(f"--- Task Failed: Worker chose an unknown tool '{tool_name}' ---")
                    previous_step_output = f"Unknown tool: {tool_name}"

            except Exception as e:
                print(f"--- Task Failed: {e} ---")
                previous_step_output = f"Task failed with error: {e}"

        # 3. Present the final collected results
        print("\n\n--- All Tasks Completed ---")
        print("Final Report:")
        for i, report in enumerate(final_results):
            print(f"Task {i+1} Result: {report}")
        print("---------------------------\n")

if __name__ == "__main__":
    main()