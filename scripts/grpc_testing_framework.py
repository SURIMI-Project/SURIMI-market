#!/usr/bin/env python

import os
import sys
import subprocess
from pathlib import Path
import json
import yaml

# Convert from seconds to datetime
from datetime import datetime

# Constants
HOST = "localhost"
PORT = 5001
STEPS = 20

# Get gRPC messages dir, as used by third-party clients, e.g. Bruno. Strategy:
# find the project's root in the filesystem, and get the messages' parent
# directory.
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root_dir = Path(script_dir).parent.absolute()
GRPC_DIR = os.path.join(project_root_dir, "grpc_requests")

# gRPC messages: Find the YAML file that corresponds to a message.
# The value is a tuple of the method and the corresponding file to read from.
service_name = "surimi.v1.MarketService"
MESSAGE_FILE = {
    "cancel_simulation":
        (f"{service_name}/CancelSimulation", os.path.join(GRPC_DIR, "CancelSimulation.yml")),
    "finalise_simulation":
        (f"{service_name}/FinaliseSimulation", os.path.join(GRPC_DIR, "FinaliseSimulation.yml")),
    "get_protocol_version":
        (f"{service_name}/GetProtocolVersion", os.path.join(GRPC_DIR, "GetProtocolVersion.yml")),
    "get_species_prices":
        (f"{service_name}/GetSpeciesPrices", os.path.join(GRPC_DIR, "GetSpeciesPrices.yml")),
    "initialise_simulation":
        (f"{service_name}/InitialiseSimulation", os.path.join(GRPC_DIR, "InitialiseSimulation.yml")),
    "simulate_step":
        (f"{service_name}/SimulateStep", os.path.join(GRPC_DIR, "SimulateStep.yml")),
    "update_sales":
        (f"{service_name}/UpdateSales", os.path.join(GRPC_DIR, "UpdateSales_small_test_sample.yml"))
}

def convert_seconds_to_datetime(parent: dict, key: str) -> str:
    """
    A helper function that converts seconds to datetime for a given message.

    Arguments
    ---------
    parent : The parent object of the datetime key. This is used to ensure that
             the key indeed exists.
    key : name of the datetime field.

    Returns
    -------
    Date time in format of "1970-01-01T00:00:00.000Z"
    """

    if key in parent:
        dt = datetime.fromtimestamp(parent.get(key).get("seconds"))
        formatted_time = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        parent[key] = formatted_time

def get_message_from_file(message_id: str) -> str:
    """
    Get the gRPC message corresponding to the gRPC method.

    The method uses the same YAML file as third-party clients like Bruno, found
    in "grpc_requests/".

    Arguments
    ---------
    message_id : a key or ID used to identify what message we will send. See
                 `MESSAGE_FILE` for available keys.
    
    Returns
    ------- 
    str : gRPC message is returned as a string. Can be used directly in
          requests.
    """
    assert len(message_id) > 0, "message_id cannot be empty."
    assert message_id, "message_id cannot be None"
    assert message_id in MESSAGE_FILE.keys(), "message_id must be valid."

    file_name = MESSAGE_FILE[message_id][1]
    file_name = os.path.join(GRPC_DIR, file_name)
    
    # This is None, so the caller knows that if something went wrong, data is
    # None.
    data = None

    # Read YAML
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    # Filter the message
    data = data.get("grpc").get("message")

    # grpcurl uses formatted datetime, while our message use seconds and
    # nanoseconds. Thus, we convert.
    data_dict = json.loads(data)

    convert_seconds_to_datetime(data_dict, "start_date_time")
    convert_seconds_to_datetime(data_dict, "current_date_time")
    convert_seconds_to_datetime(data_dict, "end_date_time")
    convert_seconds_to_datetime(data_dict, "date_time")
    
    # Initialise simulation has the datetime in a nested layer
    if "simulation" in data_dict:
        convert_seconds_to_datetime(data_dict.get("simulation"), "start_date_time")
        convert_seconds_to_datetime(data_dict.get("simulation"), "maximum_end_date_time")
    
    # Convert back to string
    data_str = json.dumps(data_dict)

    # Escape quotes
    data_str = data_str.replace('"', '\\"')

    return data_str

def send_grpc_message(message_id):
    """
    Send gRPC message to the Market model.

    Arguments
    ---------
    message_id : a key or ID used to identify what message we will send. See
                 `MESSAGE_FILE` for available keys.

    Returns
    -------
    The program crashes if the request fails.
    """

    print(f"Message ID: {message_id}")

    args = [
        "powershell",
        "grpcurl",
        "-plaintext",
        "-d",
        "\'" + get_message_from_file(message_id) + "\'",
        channel,
        MESSAGE_FILE[message_id][0]
    ]

    # Debug
    # print("Args: " + " ".join(args))

    try:
        output = subprocess.call(args)
    except subprocess.CalledProcessError as e:
        return "", f"Error running command:\n\n{e}"

    if output != 0:
        print(args)
        print("\n[ERROR] Command grpcurl exit with non-zero status code.")
        raise ConnectionError("Command status code is non-zero.")

channel=f"{HOST}:{PORT}"

# send_grpc_message("initialise_simulation")

for i in range(STEPS):
    print(f"Iteration: {i}")
    send_grpc_message("simulate_step")
    send_grpc_message("update_sales")

send_grpc_message("get_species_prices")