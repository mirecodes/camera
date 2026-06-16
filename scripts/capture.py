#!/usr/bin/env python3
import os
import sys

# Adjust path to import from functions directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from functions.capture import WebcamCapturer

if __name__ == "__main__":
    capturer = WebcamCapturer()
    # By default, we do NOT record the whole stream to avoid inefficiency.
    # Run with interactive capture enabled.
    capturer.run(record_stream=False)
