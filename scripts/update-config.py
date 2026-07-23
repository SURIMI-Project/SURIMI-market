#!/usr/bin/env python3

"""
Update the Market's configs with upstream. In its current state, it will
check against the Western Meditteranean reposity of "SURIMI-western_med".

This is an interactive tool that shows the diff to the user, and prompts it to
update the Market's config.
"""

from contextlib import chdir
from datetime import datetime
import getpass
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
import platform
import unicodedata

if __name__ != "__main__":
    print("update-config.py can only be executed as a standalone script.",
          file=sys.stderr)
    sys.exit(1)

# Parts of the URL to craft the HTTP request
GH_BASE = "git@github.com"
REPO = "Official-EwE/SURIMI-data_preprocessing"
DIRECTORY = "SURIMI-market_inputs"
REF_PATH = "refs/heads/main"

# TODO add support for multiple scenarios
SCENARIO_NAME = "northwestern_med" # anchovy-bay, aegean

FILES = (
    "markets.csv",
    "prices.csv",
    "species.csv"
)

# from .env
CONFIG_UPDATE_WORKTREE = ""

# Dependencies
VSCODE_PATH = ""


def set_vscode_path(user: str) -> str:
    def set_path(user):
        if user is None:
            return Path("/usr/bin/code")
        else:
            return Path(
                f"C:/Users/{user}/AppData/Local/Programs/Microsoft VS Code/bin/code.cmd")


    if platform.system() == "Windows":
        # Check username
        try:
            is_username_input = input(f"Is your username: {getpass.getuser()}? [Y/n] ")
            is_username_input = is_username_input.strip().lower()
        except KeyboardInterrupt:
            print("\n\nCancelled. No changes were made.")
            sys.exit(0)

        # Set VSCode path. We ask user to confirm this in a bit.
        if not is_username_input == "" and not is_username_input == "y":
            new_username = input("Enter your username: ")
            return set_path(new_username)
        else:
            return set_path(getpass.getuser())
        print()
    else:
        return set_path(None)


def get_normalized_hash(file_path, hash_algo='md5'):
    """
    Files may have different line endings. Typically, Windows will use CRLF
    while macOS and Linux use LF. Comparing files require normalising them first
    """

    with open(file_path, 'r') as f:
        content = f.read()
    normalized_content = unicodedata.normalize('NFC', content)
    encoded_content = normalized_content.encode('utf-8')
    h = hashlib.new(hash_algo)
    h.update(encoded_content)
    return h.hexdigest()

def execute_command(cmd: list):
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True)

        # Print output in real-time
        for line in proc.stderr:
            print(f"{line}", end="", flush=True)

        proc.wait()
        (stdout, stderr) = proc.communicate()

        if proc.returncode != 0:
            print(stderr)
            print(f"{cmd[0]} exited with a non-zero status code.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(e)
        print(f"Failed to run command {cmd[0]}.")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nCancelled. No changes were made.")
        sys.exit(0)

def get_config_update_worktree() -> str:
    env_file = ".env"
    worktree = ""

    print("Checking .env file...")
    if not os.path.isfile(env_file):
        print(f"{env_file} was not found. Create a new '.env' using the template '.env.template.'", file=sys.stderr)
        sys.exit(1)

    with open(env_file) as f:
        for line in f:
            segments = line.split("=")
            if len(segments) != 2:
                continue
            if segments[0] == "CONFIG_UPDATE_WORKTREE":
                worktree = segments[1]
                worktree = worktree.strip(
                    '"').strip("'")

    # Validate environment variables
    if len(worktree) == 0:
        print("Missing environment variable worktree. It should be in .env", file=sys.stderr)
        sys.exit(1)

    return worktree

def clone_upstream_repo(local_filepath: str):
    if not os.path.isdir(local_filepath):
        print(f"Cloning upstream repository {REPO}...")
        execute_command(["git", "clone", f"{GH_BASE}:{REPO}", local_filepath, "--depth=1"])
    else:
        print(f"Updating repository '{local_filepath}'")
        with chdir(local_filepath):
            try:
                proc = subprocess.Popen(
                    [
                        "git",
                        "pull"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    text=True)

                # Print output in real-time
                for line in proc.stderr:
                    print(f"{line}", end="", flush=True)

                proc.wait()
                (stdout, stderr) = proc.communicate()

                if proc.returncode != 0:
                    print(stderr)
                    print(f"Failed to pull {REPO}")
                    sys.exit(1)

            except subprocess.CalledProcessError as e:
                print(f"Failed to run command 'git pull'.")
                print(e)
                sys.exit(1)

            except KeyboardInterrupt:
                print("\n\nCancelled. No changes were made.")
                sys.exit(0)


def get_current_config(i: int):
    """Helper function to get the appropriate local config file."""
    return Path("config", SCENARIO_NAME, FILES[i])

def create_downstream_config_dir(path):
    if not os.path.isdir(path):
        print(f"Tried to find config files in '{path}', but it was not found.")
        try:
            confirm_create_dirs = input("Would you like to create it? [Y/n] ")
        except KeyboardInterrupt:
            print("Cancelled")
            sys.exit(0)

        confirm_create_dirs = confirm_create_dirs.strip().lower()
        if confirm_create_dirs != "y" and confirm_create_dirs != "":
            print("Cancelled.")
            sys.exit(0)
        else:
            try:
                os.makedirs(path, exist_ok=False)
            except OSError:
                print(f"Something went wrong. Tried to create the existing directory {path}.", file=sys.stderr)
                print("Aborted", file=sys.stderr)
                sys.exit(1)

def find_nonmatching_files(upstream_path) -> list:
    upstream_configs = set()
    for i, file in enumerate(upstream_path.iterdir()):
        upstream_configs.add(file.name)

    upstream = upstream_configs - set(FILES)
    downstream = set(FILES) - upstream_configs
    return list(upstream), list(downstream)

def compare_and_diff(upstream_path) -> list:
    """
    Prompt to compare changed files with VSCode.

    Return missing files, i.e. files in upstream but not in downstream, exclud-
    ing the changed ones.
    """
    missing_files = []
    for i, file in enumerate(upstream_path.iterdir()):
        if os.path.isfile(Path("config/northwestern_med/", FILES[i])):
            downstream_config = Path("config", SCENARIO_NAME, file.name)
            # File exists
            hash_1 = get_normalized_hash(file)
            hash_2 = get_normalized_hash(downstream_config)
            
            if hash_1 != hash_2:
                updated_files.append(file)
            
                try:
                    show_diff = input(f"'{downstream_config.name}' is different. Show diff? [y/N] ")
                except KeyboardInterrupt:
                    print("\n\nCancelled. No changes were made.")
                    sys.exit(0)
            
                show_diff = show_diff.strip().lower()

                if show_diff == "y":
                    print("Opening a new tab in VSCode to show the diff...")
                    print("Come back to the terminal once you have finished reviewing the diff.")
                    execute_command([VSCODE_PATH, "--diff", downstream_config, file])
        else:
            missing_files.append(file)

    return missing_files

def create_new_commit(missing_files: list):
    if len(updated_files) > 0:
        print("The following files are different:")
        for file in updated_files:
            print(f"  - {file}")

        if len(missing_files) > 0:
            print("and t", end="")
    else:
        print("T", end="")

    if len(missing_files) > 0:
        print("he following files do not exist?")
        for file in missing_files:
            print(f"  - {file}")

    to_repeat = True
    to_commit = False
    while to_repeat:
        try:
            confirm_commit = input("Would you like to commit the changes or copy the missing files to the current branch? [y/n] ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        
        confirm_commit = confirm_commit.strip().lower()

        if confirm_commit != "y" and confirm_commit != "n":
            to_repeat = True
        else:
            to_commit = confirm_commit == "y"
            to_repeat = False


    # Commit the changes
    if to_commit:
        execute_command(["git", "pull"])
        # Copy files
        for i, file in enumerate(updated_files):
            downstream_config = Path("config", SCENARIO_NAME, file.name)
            
            print(f"Copying {file} -> {downstream_config}")
            shutil.copy(file, downstream_config)
            execute_command(["git", "add", downstream_config])
        
        # Add missing files
        for i, file in enumerate(missing_files):
            downstream_config = Path("config", SCENARIO_NAME, file.name)
            print(f"Copying {file} -> {downstream_config}")
            shutil.copy(file, downstream_config)
            execute_command(["git", "add", downstream_config])

# Welcome
# -------
print("The interactive tool to update the Market configs.")
print("You can exit this tool at any point with CTRL+C.", end="\n\n")

# Check that the pre-requisites are installed or present
# ------------------------------------------------------
VSCODE_PATH = set_vscode_path(None)
print("Checking dependencies and pre-requisites...")

# Environment variables. Find the variables we need
CONFIG_UPDATE_WORKTREE = get_config_update_worktree()

print("\n-------------------\n")

print("Checking Git...")
try:
    subprocess.check_call(["git", "--version"])
except subprocess.CalledProcessError:
    print("Git is not installed or is not in PATH. Please ensure that it is before executing this script.", file=sys.stderr)
    sys.exit(1)

print("\n-------------------\n")

print("Checking Visual Studio Code...")
try:
    subprocess.check_call([VSCODE_PATH, "--version"])
except subprocess.CalledProcessError:
    print("Visual Studio Code is not installed or is not in PATH. Please ensure that it is before running this script", file=sys.stderr)
    sys.exit(1)

print("\n-------------------\n")

# This is super slow. Let's assume SSH is already in PATH.
print("Checking SSH and connection with GitHub...")
try:
    proc = subprocess.Popen(["ssh", "-T", "git@github.com"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = proc.communicate()

    # For some reason, SSH returning 1 is success. Anything else, including 0,
    # means failure.
    if proc.returncode != 1:
        print("Cannot connect with SSH to GitHub. Please ensure that you have configured your SSH key with GitHub.", file=sys.stderr)
        sys.exit(1)

except subprocess.CalledProcessError as e:
    print(e)
    print("Failed to run command 'ssh'.")
    sys.exit(1)

except FileNotFoundError:
    print("Command SSH is not installed or enabled. Please ensure that it is before running this script.", file=sys.stderr)
    sys.exit(1)

print("\n-------------------\n")

# ------------------------------------------------------------------------------

# Clone the remote project or use the existing local repository
local_filepath = Path(CONFIG_UPDATE_WORKTREE, REPO)
clone_upstream_repo(local_filepath)

# TODO support the two other scenarios
upstream_config_dir = Path(local_filepath, DIRECTORY, "northwestern_med")

updated_files = []
missing_files = []

# If the SURIMI-market repo does not have a `config/` yet, prompt to create it
config_dir_path = Path("config", SCENARIO_NAME)
create_downstream_config_dir(config_dir_path)

# Find non-matching files
upstream, downstream = find_nonmatching_files(upstream_config_dir)
if len(upstream) + len(downstream) > 0:
    print("")
    if len(upstream) > 0:
        print("Non-matching files found in upstream: ", upstream, file=sys.stderr)

    if len(downstream) > 0:
        print("Downstream excessive files: ", downstream, file=sys.stderr)

    print("Aborted")
    sys.exit(1)


# Prompt to diff the changed files
missing_files = compare_and_diff(upstream_config_dir)

# Prompt for committing changes
create_new_commit(missing_files)

date = datetime.today().strftime('%Y-%m-%d')
message = f"""chore: ({date}) sync config files 

This commit is generated by scripts/update-config.py."""

execute_command(["git", "commit", "-m", message])
print("You can sync with remote using\n\n    git push")

print("Script has finished successfully!")
