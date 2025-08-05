from core.agent_logic import analysePrompt
from rag.semantic_search import semanticCephSearch
from utils.file_ops import vectorBuilder
from ceph.executor import execute_command
import os


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

    #llm_response = llmResponse()
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


if __name__ == "__main__":
    main()