import socket

def is_server_running(host="localhost", port=50051):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0  # Returns True if connected

if is_server_running():
    print(" Server is running!")
else:
    print(" Server is NOT running!")
