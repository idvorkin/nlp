import os
import ell
from icecream import ic
import inspect
import subprocess
import socket
import asyncio


def get_caller_filename():
    current_file = inspect.currentframe().f_code.co_filename
    for frame_info in inspect.stack():
        if frame_info.filename != current_file:
            return os.path.splitext(os.path.basename(frame_info.filename))[0]
    return "unknown"


def get_ell_logdir():
    caller_file = get_caller_filename()
    return os.path.expanduser(f"~/tmp/ell_logdir/{caller_file}")


def init_ell():
    ell.init(store=get_ell_logdir(), autocommit=True)
    ell.models.groq.register()


def get_ell_model(
    openai: bool = False,
    openai_cheap: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
) -> str:
    """
    Select and return the appropriate ELL model based on the provided flags.
    """
    # if more then one is true, exit and fail
    count_true = sum([openai, google, claude, llama, openai_cheap])
    if count_true > 1:
        print("Only one model can be selected")
        exit(1)
    if count_true == 0:
        # default to openai
        openai = True

    if google:
        raise NotImplementedError("google")
    elif claude:
        return "claude-3-5-sonnet-20240620"
    elif llama:
        return "llama-3.2-90b-vision-preview"
    elif openai_cheap:
        return "gpt-4o-mini"
    else:
        return "gpt-4o-2024-08-06"  # Assuming this is the name for gpt4


def is_port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", port))
    sock.close()
    return result == 0


def open_browser(port):
    subprocess.run(["open", f"http://127.0.0.1:{port}"])


def find_available_port(start_port=5000, max_port=65535):
    for port in range(start_port, max_port + 1):
        if not is_port_in_use(port):
            return port
    raise RuntimeError("No available ports found")


async def run_server_and_open_browser(logdir):
    port = find_available_port()
    ic(logdir)

    # Start the server asynchronously with the found port
    server_process = await asyncio.create_subprocess_exec(
        "ell-studio", "--storage", logdir, "--port", str(port)
    )

    # Wait for 2 seconds
    await asyncio.sleep(2)

    # Open the browser with the same port
    open_browser(port)

    # Keep the server running
    await server_process.wait()


def run_studio():
    try:
        # Need to get the logdir from the caller which we
        # can't get once we go asyn
        logdir = get_ell_logdir()
        ic(logdir)
        asyncio.run(run_server_and_open_browser(logdir))
    except RuntimeError as e:
        ic(f"Error: {e}")
    except Exception as e:
        ic(f"An unexpected error occurred: {e}")
