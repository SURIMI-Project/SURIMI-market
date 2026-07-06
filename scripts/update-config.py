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
REPO = "Official-EwE/SURIMI-western_med"
DIRECTORY = "SURIMI-market_inputs"
REF_PATH = "refs/heads/main"

FILES = (
    "ports.csv",
    "prices.csv",
    "species.csv"
)

# from .env
CONFIG_UPDATE_WORKTREE = ""

# Dependencies
VSCODE_PATH = ""


def set_vscode_path(user: str):
    global VSCODE_PATH
    if user is None:
        VSCODE_PATH = Path("/usr/bin/code")
    else:
        VSCODE_PATH = Path(
            f"C:/Users/{user}/AppData/Local/Programs/Microsoft VS Code/bin/code.cmd")


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

# Welcome
# -------
print("The interactive tool to update the Market configs.")
print("You can exit this tool at any point with CTRL+C.", end="\n\n")

# Check that the pre-requisites are installed or present
# ------------------------------------------------------
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
        set_vscode_path(new_username)
    else:
        set_vscode_path(getpass.getuser())
    print()
else:
    set_vscode_path(None)

print("Checking dependencies and pre-requisites...")

# Environment variables. Find the variables we need
env_file = ".env"
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
            CONFIG_UPDATE_WORKTREE = segments[1]
            CONFIG_UPDATE_WORKTREE = CONFIG_UPDATE_WORKTREE.strip(
                '"').strip("'")

# Validate environment variables
if len(CONFIG_UPDATE_WORKTREE) == 0:
    print("Missing environment variable CONFIG_UPDATE_WORKTREE. It should be in .env", file=sys.stderr)
    sys.exit(1)

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

# TODO support the two other scenarios
local_config_dir = Path(local_filepath, DIRECTORY, "northwestern_med")

updated_files = []

# Validate that the config files exist
for i, file in enumerate(local_config_dir.iterdir()):
    hash_1 = get_normalized_hash(file)
    hash_2 = get_normalized_hash(Path("config", FILES[i]))
    
    if hash_1 != hash_2:
        updated_files.append(file)
    
        try:
            show_diff = input(f"'{FILES[i]}' is different. Show diff? [y/N] ")
        except KeyboardInterrupt:
            print("\n\nCancelled. No changes were made.")
            sys.exit(0)
    
        show_diff = show_diff.strip().lower()

        if show_diff == "y":
            print("Opening a new tab in VSCode to show the diff...")
            print("Come back to the terminal once you have finished reviewing the diff.")
            execute_command([VSCODE_PATH, "--diff", Path("config/", FILES[i]), file])

# Prompt for committing changes
print("The following files are different:")
for file in updated_files:
    print(f"  - {file}")

to_repeat = True
to_commit = False
while to_repeat:
    try:
        confirm_commit = input("Would you like to commit the changes to the current branch? [y/n] ")
    except KeyboardInterrupt:
        print("\n\nCancelled. No changes were made.")
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
    for i, file in enumerate(updated_files):
        print(f"Copying {file} -> {Path("config", FILES[i])}")
        shutil.copy(file, Path("config", FILES[i]))
        execute_command(["git", "add", Path("config", FILES[i])])

date = datetime.today().strftime('%Y-%m-%d')
message = f"""chore: ({date}) sync config files 

This commit is generated by scripts/update-config.py."""

execute_command(["git", "commit", "-m", message])
print("You can sync with remote using\n\n    git push")

print("Script has finished successfully!")
