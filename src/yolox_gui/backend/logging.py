import sys


def safe_print(message) -> None:
    """Print backend logs without crashing on Windows console encodings."""
    text = str(message)
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
        stream.flush()


def log_message(callback, message) -> None:
    if callback:
        callback(str(message))
    else:
        safe_print(message)

