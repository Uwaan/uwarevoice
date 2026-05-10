#!/bin/bash

# Set logging level (0: debug, 1: info, 2: warning, 3: error)
LOG_LEVEL=1

SCRIPT_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd)
cd $SCRIPT_PATH
source ./venv/bin/activate

# (Systemd unit might specify its own logging level)
if [[ " $* " == *" --loglevel"* ]]; then
    exec python u_revoice.py "$@"
else
    exec python u_revoice.py --loglevel=$LOG_LEVEL "$@"
fi
