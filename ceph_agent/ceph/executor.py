#from ast import main
import subprocess
#import paramiko  # Import the Paramiko library
import sys

CEPH_CONF_PATH = '/etc/ceph/ceph.conf'

'''
def execute_command(
    cmd,
    conf=CEPH_CONF_PATH,
    username="client.admin",
    keyring=None,
    ssh_user=None,
    ssh_host=None,
    ssh_port=22,
    ssh_password=None,
    ssh_key_filepath=None,
):
    """
    Executes a Ceph command, either locally or remotely via SSH using Paramiko.

    Args:
        cmd (string): Ceph command to execute.
        conf (string): Configuration path if the default PATH is not available.
        username (string): Username which will perform the execution of the ceph
                          command (Ceph user).
        keyring (string): PATH to the keyring path for Ceph authentication.
        ssh_user (string, optional): Username for SSH connection (e.g., 'root',
                                    'ceph_user'). If provided along with ssh_host,
                                    the command will be executed remotely.
        ssh_host (string, optional): Hostname or IP address of the remote Ceph
                                    admin node. If provided along with ssh_user,
                                    the command will be executed remotely.
        ssh_port (int, optional): SSH port for the remote connection. Defaults to 22.
        ssh_password (string, optional): Password for SSH authentication. Use with
                                        caution; SSH keys are preferred.
        ssh_key_filepath (string, optional): Path to the private SSH key file
                                            (e.g., '~/.ssh/id_rsa').

    Returns:
        tuple: stdout (str), stderr (str), returncode (int)

    Raises:
        Exception: If the command execution failed, including SSH connection or
                  authentication errors.
    """

    try:
        # Build the Ceph-specific arguments for the command
        ceph_args = ""
        if keyring is None:
            print(
                "Taking the default admin keyring listed under /etc/ceph/ "
                "directory (or remote equivalent)"
            )
            ceph_args = f" --conf {conf}"
        else:
            print("Taking the specified keyring, & username")
            ceph_args = f" --conf {conf} --keyring={keyring} --name={username}"

        # Combine the base command with Ceph arguments
        full_ceph_cmd = f"{cmd}{ceph_args}"

        if ssh_user and ssh_host:
            print(
                f"Executing command remotely via Paramiko SSH on "
                f"{ssh_user}@{ssh_host}:{ssh_port}"
            )
            #client = paramiko.SSHClient()
            #client.load_system_host_keys()  # Load known hosts from ~/.ssh/known_hosts
           
            #client.set_missing_host_key_policy(
            #    paramiko.AutoAddPolicy()
            #)  # Auto-add new hosts (use with caution in production)

            #key = paramiko.RSAKey.from_private_key_file(ssh_key_filepath)
            #print(key)
            try:
                if ssh_key_filepath:
                    # Authenticate using an SSH key file
                    full_ceph_cmd = f"ssh {ssh_user}@{ssh_host} -p {ssh_port} -i {ssh_key_filepath} -- " + full_ceph_cmd

                    client.connect(
                        hostname=ssh_host,
                        port=ssh_port,
                        username=ssh_user,
                        key_filename=ssh_key_filepath,
                    )
                    
                elif ssh_password:
                    # Authenticate using a password

                    full_ceph_cmd = f"sshpass -p '{ssh_password}' ssh {ssh_user}@{ssh_host} -p {ssh_port} -- " + full_ceph_cmd
                    
                    client.connect(
                        hostname=ssh_host,
                        port=ssh_port,
                        username=ssh_user,
                        password=ssh_password,
                    )
                    
                else:
                    # Attempt authentication without explicit key/password
                    # (e.g., agent forwarding)

                    full_ceph_cmd = f"ssh {ssh_user}@{ssh_host} -- " + full_ceph_cmd
                    
                    client.connect(
                        hostname=ssh_host, port=ssh_port, username=ssh_user
                    )
                    
                    
                print(f"Remote Command: {full_ceph_cmd}")

                stdout, stderr, return_code = subprocess.run(
                    full_ceph_cmd, capture_output=True, text=True, check=True, shell=True
                    )

                
                stdin, stdout, stderr = client.exec_command(full_ceph_cmd)

                # Read output
                output = stdout.read().decode('utf-8')
                error = stderr.read().decode('utf-8')
                return_code = stdout.channel.recv_exit_status()
                

                if return_code != 0:
                    raise Exception(
                        f"Remote command '{full_ceph_cmd}' failed with exit code "
                        f"{return_code}\nSTDOUT: {output}\nSTDERR: {error}"
                    )

                #return output, error, return_code
                return stdout, stderr, return_code
            
            except paramiko.AuthenticationException as e:
                raise Exception(
                    f"SSH Authentication failed: {e}. Check username, password,"
                    f" or key file."
                )
            except paramiko.SSHException as e:
                raise Exception(
                    f"Could not establish SSH connection: {e}. Check host, port, "
                    f"or network."
                )

            except Exception as e:
                raise Exception(
                    f"An error occurred during remote command execution: {e}"
                )
            finally:
                #client.close()  # Ensure the SSH connection is closed
        else:
            # Otherwise, execute the command locally using subprocess
            print("Executing command locally")
            print(f"Local Command: {full_ceph_cmd}")
            result = subprocess.run(
                full_ceph_cmd, capture_output=True, text=True, check=True, shell=True
            )
            return result.stdout, result.stderr, result.returncode

    except subprocess.CalledProcessError as e:
        # This exception is raised if check=True and the local command returns
        # a non-zero exit code
        print(f"Local command execution failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise Exception(
            f"Local command '{e.cmd}' failed with exit code {e.returncode}"
        )
    except FileNotFoundError:
        # This exception is raised if the local command (e.g., 'ceph') is not found
        print(
            "Error: Local command 'ceph' not found. Make sure 'ceph' is in "
            "your system's PATH."
        )
        raise
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise

'''


def execute_command(cmd, conf=CEPH_CONF_PATH, username="client.admin", keyring=None):
    """
    Short description of what the function does.

    Args:
        cmd (string): Ceph command to execute on the running cluster admin node.
        conf (string): Configuration path if the default PATH is not available.
        username (string): Username which will perform the execution of the ceph command
        keyring (string): PATH to the keyring path 

    Returns:
        return_type: stdout, stderr, returncode

    Raises:
        ExceptionType: If the command execution failed then an exception is raised.
    """

    try:
        if keyring is None:
            print("Taking the default admin keyring listed under /etc/ceph/ directory")
            cmd = cmd + f" --conf {conf}"
        else:
            print("Taking the specified keyring, & username")
            cmd = cmd + f" --conf {conf} --keyring={keyring} --name={username}" 

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    
    cmd = f"ssh root@130.198.19.212 -i /Users/kritiksachdeva/Downloads/sdf-ssh-key_rsa.prv -- '{cmd}'"

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
    return result.stdout, result.stderr, result.returncode