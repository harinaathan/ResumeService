from datetime import datetime
import pytz
import os
import threading

_lock = threading.Lock()
def logger(message, tag=""):
    """
    Logs a message with a timestamp to a file, ensuring thread safety.
    """
    # create a log folder
    if not os.path.exists('log'):
        os.mkdir('log')
    # set log file
    logFile = os.path.join('log','dbm.txt')
    # obtain timestamp in IST
    ts = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    # write message by appending
    with _lock:  # thread saftey
        with open(logFile, '+a') as f:
            f.write(f"{ts}_{tag} {message}\n")