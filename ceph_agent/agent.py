from core.agent_logic import analyze_response
from llm.llm_response import run_llm_query_with_ollama
from rag.semantic_search import search_and_select_command_with_llm
from utils.file_ops import load_index
from ceph.executor import execute_command


def main():
    print("Initializing Ceph Agent...")
    print("--------------------------------")
    # Step-1: Loading the Database
    index, metadata, model = load_index("./database/basic_commands.json")

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

        print("Searching for command...")
        print("Results:")
        print("--------------------------------")

        # Step-4: Search & Select the best available command
        vector_results, selected_command = search_and_select_command_with_llm(
            index,
            metadata,
            model,
            user_query,
            model_choice=model_choice,
            llm_model="granite3.3:8b",
            top_k=3,
            threshold=1.1
        )  # Here this search_command can become a tool for MCP to use

        if not selected_command:
            print(
                "Agent Response: I couldn't find a suitable Ceph command for your "
                "query in my knowledge base."
            )
            continue
        # Step-5: Execute the selected command on the ceph cluster
        stdout, stderr, retcode = execute_command(selected_command.strip())

        print(stdout)

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
            description = next((item['description'] for item in vector_results if item['command'] == selected_command.strip()), 'Description not found.')
        
            agent_response = analyze_response(
                user_query,
                selected_command,
                stdout,
                run_llm_query_with_ollama,
                description,
                model_name="granite3.3:8b"
            )
            if not agent_response:
                agent_response = (
                    "Agent Response: I executed the command, but could not extract "
                    "a clear answer from its output based on your query."
                )
            print(f"\nAgent Response: {agent_response}")
            print("\n------------------------------------")


'''
def main():
    user_query = "How many service running in the ceph cluster?"
    print(f"Query: {user_query}")
    print("--------------------------------")

    index, metadata, model = load_index("./database/basic_commands.json")
    print("Searching for command...")
    print("Results:")
    print("--------------------------------")

    filtered_commands = search_command(index, metadata, model, user_query, top_k=3, model_choice)  # Here this search_command can become a tool for MCP to use
    if filtered_commands:
        for r in filtered_commands:
            print(f"[Score: {r['score']:.4f}] ➜ {r['command']}")
            print(f"  → {r['description']}\n")
    else:
        print("No relevant command found with sufficient confidence.")
        sys.exit(1)

    # Step 3: Pick which LLM to use
    model_choice = input("Use Ollama or LM Studio? (o/l): ").strip().lower()

    # Step 4: Use LLM to select best matching command
    if model_choice == 'o':
        selected_command = select_command_with_ollama(user_query, filtered_commands, model="llama3")
    elif model_choice == 'l':
        selected_command = select_command_with_lmstudio(user_query, filtered_commands)
    else:
        print("Invalid choice. Use 'o' for Ollama or 'l' for LM Studio.")
        return

    # Step 5: Output selected command
    print(f"\nLLM Selected Command:\n{selected_command.strip()}")

    # Step-6: Execute the selected command on the ceph cluster

    stdout, stderr, retcode = execute_command(selected_command.strip())

'''

if __name__ == "__main__":
    main()
    print("Done")