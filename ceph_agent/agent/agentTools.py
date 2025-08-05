from langchain_core.tools import tool
import subprocess

# Here I am reusing all of the tools again that I used earlier
# but with a more better orientation.


@tool
def execute_ceph_command(command: str) -> str:
    print("Execute Ceph Commands")
    # Calls the execute_command function
    try:
        # For safety, ensure the command starts with 'ceph' if not already present.
        # This is a good practice to prevent the LLM from trying to run arbitrary commands.
        if not command.strip().startswith("ceph"):
            full_command = f"ceph {command}"
        else:
            full_command = command

        full_command = "ssh root@130.198.19.212 -i /Users/kritiksachdeva/Downloads/sdf-ssh-key_rsa.prv -- " + full_command
        result = subprocess.run(
            full_command.split(),
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # Add a timeout to prevent hanging
        )
        return f"Command executed successfully:\n---\n{result.stdout}\n---"
    except subprocess.CalledProcessError as e:
        return f"Command failed with an error:\n---\n{e.stderr}\n---\nStdout:\n---\n{e.stdout}\n---"
    except FileNotFoundError:
        return "Error: 'ceph' command not found. Ensure Ceph CLI is installed and in the system's PATH."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."


# Assume you have a function or a class that represents your RAG/vector store retriever.
# This is a placeholder; replace this with your actual implementation.
def run_retriever(query: str) -> str:
    """
    This is your RAG/retriever function.
    It takes a query and returns relevant documents/text from your knowledge base.
    """
    # Example placeholder:
    # docs = your_rag_pipeline_or_vector_store.invoke(query)
    # return "\n\n".join([doc.page_content for doc in docs])
    return f"This is a placeholder for search results related to: '{query}'.\n" \
           "In a real scenario, this would return relevant documents from your vector store."


@tool
def search_command(query: str) -> str:
    """
    Searches an internal knowledge base for relevant documents, explanations, and troubleshooting guides
    related to Ceph issues. Use this when you need background information, definitions, or
    troubleshooting steps that cannot be found by executing a direct Ceph command.

    Args:
        query (str): A natural language question or search query for the internal knowledge base.

    Returns:
        str: A summary of relevant information found in the knowledge base.
    """
    return run_retriever(query)