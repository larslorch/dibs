import subprocess
import sys
from datetime import datetime

SEPARATOR = '_'

def get_version():
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode(sys.stdout.encoding) 
    return SEPARATOR + git_commit


def get_datetime():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H%M")
    return SEPARATOR + date_time


def get_version_datetime():
    v = get_version()
    d = get_datetime()
    return ''.join([v, d])

