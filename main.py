import argparse
import threading


def main(trans_func, host, port, block_size: float = 0.3):
    from output import Output
    from record import record

    o = Output(trans_func, host=host, port=port)
    threading.Thread(target=o.transcribe_main, args=[block_size]).start()
    print("Listening...")
    try:
        frame_number = 0
        while not o.stop:
            o.audio_buffer[frame_number] = record(block_size)
            o.newest_frame = frame_number
            frame_number += 1

    except KeyboardInterrupt:
        o.stop = True

    print("Not listening")


def run_server(host, port, model):
    from server import server
    from transcribe import Transcribe

    t = Transcribe(model)
    server(port=port, address=host, trans_func=t.run)


def run_client(
    host,
    port,
):
    from client import client

    main(client, host, port)


def run_local(port, host, model):
    from transcribe import Transcribe

    t = Transcribe(model)
    main(t.run, host, port)


def parse_args():
    parser = argparse.ArgumentParser(description="A whisper dictation system.")

    # Define argument for specifying the role
    parser.add_argument(
        "role", choices=["server", "local", "client"], help="Role to run the script as"
    )

    # Define optional arguments for specifying port and host
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port to use for server/client roles",
        default="8000",
    )
    parser.add_argument(
        "-H", "--host", help="Host to use for server/client role", default="localhost"
    )

    parser.add_argument(
        "-m",
        "--model",
        help="The model to use. {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}",
        default="tiny.en",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    role = args.role
    model = args.model
    host = args.host
    port = args.port

    if role == "server":
        # Run server code with specified port
        print("Running as server with port:", port, "and host:", host)
        run_server(host, port, model)
    elif role == "client":
        # Run client code with specified port and host
        print("Running as client with port:", port, "and host:", host)
        run_client(host, port)
    elif role == "local":
        # Run local code
        print("Running as local")
        run_local(host, port, model)
