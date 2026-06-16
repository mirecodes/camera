#!/usr/bin/env python3
import os
import sys
import json
import time
import threading
import yaml
import tkinter as tk
from tkinter import scrolledtext

class OCRMonitorApp:
    def __init__(self, root):
        self.root = root
        
        # Load configuration from cfg/monitor_config.yml
        self.font_family = "Arial"
        self.font_size = 18
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, "cfg", "monitor_config.yml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.font_family = config.get("font_family", self.font_family)
                        self.font_size = int(config.get("font_size", self.font_size))
                print(f"[Monitor] Loaded settings - Font: {self.font_family} {self.font_size}pt")
            except Exception as e:
                print(f"[Monitor] Warning: Failed to parse configuration file: {e}", file=sys.stderr)
        else:
            print("[Monitor] Config file cfg/monitor_config.yml not found. Using defaults.")

        # Path to monitor the JSON payload file
        self.json_path = os.path.join(project_root, "stream", "ocr_result.json")
        print(f"[Monitor] Watching JSON path: {self.json_path}")
        self.last_capture_id = None

        # Configure root window properties
        self.root.title("OCR Text Monitor")
        self.root.geometry("800x650")
        self.root.configure(bg="#FFFFFF")  # Pure white background
        
        self.setup_ui()
        
        # Hotkeys: Escape to clear, Control-S to save log
        self.root.bind("<Escape>", lambda event: self.clear_text())
        self.root.bind("<Control-s>", lambda event: self.save_log())
        
        # Start the file watcher in a daemon background thread
        self.is_running = True
        self.watcher_thread = threading.Thread(target=self.watch_json_file, daemon=True)
        self.watcher_thread.start()

    def setup_ui(self):
        # 1. Top Strip (Minimal status bar showing Mode and File status only)
        # Background: `#F0F0F0` (light gray), border separator at bottom
        status_bar = tk.Frame(self.root, bg="#EAEAEA", height=35)
        status_bar.pack(fill=tk.X, side=tk.TOP)
        status_bar.pack_propagate(False)
        
        # Info labels
        self.info_label = tk.Label(
            status_bar, 
            text="OCR Monitor  |  File-based Mode", 
            font=("Arial", 10, "bold"), 
            fg="#444444", 
            bg="#EAEAEA"
        )
        self.info_label.pack(side=tk.LEFT, padx=15, pady=8)

        self.status_label = tk.Label(
            status_bar, 
            text="● Monitoring", 
            font=("Arial", 10, "italic"), 
            fg="#2E7D32",  # Quiet green
            bg="#EAEAEA"
        )
        self.status_label.pack(side=tk.RIGHT, padx=15, pady=8)

        # 2. Main Content Frame (100% flat scrollable text area below the strip)
        content_frame = tk.Frame(self.root, bg="#FFFFFF")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollable Text display area (pure white background, pure black text)
        self.text_area = scrolledtext.ScrolledText(
            content_frame, 
            wrap=tk.WORD, 
            font=(self.font_family, self.font_size), 
            bg="#FFFFFF", 
            fg="#000000", 
            insertbackground="black",  # Black cursor
            padx=20,
            pady=20,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Initial status message
        self.text_area.insert(tk.END, "System ready. Watching for OCR text updates...\n")
        self.text_area.configure(state=tk.DISABLED)  # Read-only initially

    def update_text(self, payload):
        """Thread-safe UI update to insert newly received OCR text."""
        text = payload.get("text", "")
        
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)  # Overwrite previous screen content
        self.text_area.insert(tk.END, text)
        self.text_area.configure(state=tk.DISABLED)
        self.text_area.see("1.0")  # Scroll to top of the new text for natural reading

    def clear_text(self):
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "Cleared. Waiting for OCR text stream...\n")
        self.text_area.configure(state=tk.DISABLED)

    def save_log(self):
        log_text = self.text_area.get("1.0", tk.END)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        log_path = os.path.join(project_root, "data", "monitor_saved_log.txt")
        
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_text)
            print(f"[Monitor] Log saved to: {log_path}")
        except Exception as e:
            print(f"[Monitor] Failed to save log: {e}", file=sys.stderr)

    def watch_json_file(self):
        """Monitors modifications to ocr_result.json and updates the GUI."""
        last_mtime = 0.0
        
        # If file exists at startup, load its current state
        if os.path.exists(self.json_path):
            try:
                last_mtime = os.path.getmtime(self.json_path)
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                    capture_id = payload.get("capture_id")
                    if capture_id:
                        self.last_capture_id = capture_id
                        self.root.after(0, lambda p=payload: self.update_text(p))
            except Exception as e:
                print(f"[Monitor Watcher] Initial load error: {e}", file=sys.stderr)

        while self.is_running:
            time.sleep(0.1)  # Poll every 100ms
            if os.path.exists(self.json_path):
                try:
                    mtime = os.path.getmtime(self.json_path)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        # Give a tiny buffer for file-write synchronization
                        time.sleep(0.02)
                        with open(self.json_path, 'r', encoding='utf-8') as f:
                            payload = json.load(f)
                        
                        capture_id = payload.get("capture_id")
                        # Only update if the capture_id is actually new/different
                        if capture_id and capture_id != self.last_capture_id:
                            self.last_capture_id = capture_id
                            self.root.after(0, lambda p=payload: self.update_text(p))
                            self.root.after(0, lambda: self.status_label.config(text="● Updated", fg="#2E7D32"))
                            # Revert status text to "Monitoring" after 2 seconds
                            self.root.after(2000, lambda: self.status_label.config(text="● Monitoring", fg="#2E7D32"))
                except Exception as e:
                    # File might be temporarily locked during write, retry on next loop
                    pass

    def close(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
