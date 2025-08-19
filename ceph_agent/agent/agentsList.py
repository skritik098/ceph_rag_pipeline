from ceph.executor import execute_command
from core.agent_logic import analysePrompt


# Definition of different Agents
# --- Agent Definitions ---

class RetrieverAgent:
    """Finds the best command for a given query."""
    def __init__(self, ceph_search_instance):
        self.ceph_search = ceph_search_instance

    def find_command(self, query: str, model_choice: str) -> (str, list):
        print("➡️ RetrieverAgent: Searching for command...")
        vect_results, selected_command = self.ceph_search.search_and_select(
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

        description = next((item['description'] for item in vect_results if item['command'] == command), 'Description not found.')

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
