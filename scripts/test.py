#!/usr/bin/env python

import sys
import subprocess

def execute(cmd):
    try:
        output = subprocess.call(cmd)
    except subprocess.CalledProcessError as e:
        f"Error running command:\n\n{e}"
        sys.exit(1)

    if output != 0:
        print(cmd)
        print("\n[ERROR] Command grpcurl exit with non-zero status code.")
        raise ConnectionError("Command status code is non-zero.")

print("Run unit test and coverage...")
execute("coverage run -m pytest")
execute("coverage report")
execute("coverage html")
