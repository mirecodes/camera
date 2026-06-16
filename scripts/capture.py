#!/usr/bin/env python3
import os
import sys

# Adjust path to import from functions directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from functions.capture import WebcamCapturer

if __name__ == "__main__":
    import shutil
    stream_dir = os.path.join(project_root, "stream")
    if os.path.exists(stream_dir):
        try:
            shutil.rmtree(stream_dir)
            print(f"[Capture Clear] Cleared stream directory: {stream_dir}")
        except Exception as e:
            print(f"[Capture Clear] Failed to clear stream directory: {e}")
    os.makedirs(stream_dir, exist_ok=True)

    capturer = WebcamCapturer()
    # By default, we do NOT record the whole stream to avoid inefficiency.
    # Run with interactive capture enabled.
    capturer.run(record_stream=False)
