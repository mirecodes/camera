#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Monitor Program (File-based Stream Consumer)
------------------------------------------------
- Communication: File-based JSON polling (stream/metadata.json, stream/page_xxxx.json)
  (Replaced older TCP socket/port communication to avoid firewall and port conflicts)
- Bug Fix (2026-06-16):
  Fixed a bug where the watcher thread failed to load and display the very first page (page 0)
  when monitor.py started with self.max_page = 0 and metadata.json updated latest_page to 0
  (0 == 0 matched, blocking the trigger). Now self.max_page is initialized to -1, and
  any existing metadata.json is pre-loaded at startup.
"""
import os
import sys
import json
import time
import threading
import yaml
import tkinter as tk
from tkinter import scrolledtext

class OCRMonitorViewApp:
    def __init__(self, root):
        self.root = root
        self.is_running = True
        
        # Load configuration from cfg/monitor_config.yml
        self.font_family = "Arial"
        self.font_size = 18
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(script_dir)
        config_path = os.path.join(self.project_root, "cfg", "monitor_config.yml")
        
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
        
        # Paths
        self.stream_dir = os.path.join(self.project_root, "stream")
        self.meta_path = os.path.join(self.stream_dir, "metadata.json")
        
        # Memory Buffer State
        self.text_buffer = {}  # dict of {page_num: text_string}
        self.current_ptr = 0   # current page pointer
        self.max_page = -1     # highest page index recorded in metadata.json (initialize to -1)
        self.last_meta_mtime = 0.0
        
        # Load initial max_page if metadata exists
        if os.path.exists(self.meta_path):
            try:
                self.last_meta_mtime = os.path.getmtime(self.meta_path)
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.max_page = int(meta.get("latest_page", -1))
            except Exception:
                pass
        
        # Configure root window properties
        self.root.title("Multi-Page OCR Viewer")
        self.root.geometry("800x650")
        self.root.configure(bg="#FFFFFF")  # Pure white background
        
        self.setup_ui()
        self.bind_events()
        
        # Load initial buffer
        self.update_buffer_and_evict()
        self.refresh_display()
        
        # Start background metadata watcher thread (runs every 0.5s)
        self.watcher_thread = threading.Thread(target=self.watch_metadata, daemon=True)
        self.watcher_thread.start()

    def setup_ui(self):
        # 1. Top Strip (Minimal status bar showing Page counter and state)
        status_bar = tk.Frame(self.root, bg="#EAEAEA", height=35)
        status_bar.pack(fill=tk.X, side=tk.TOP)
        status_bar.pack_propagate(False)
        
        # Page indicator and help hints
        self.page_label = tk.Label(
            status_bar, 
            text="Page: 0 / 0", 
            font=("Arial", 10, "bold"), 
            fg="#444444", 
            bg="#EAEAEA"
        )
        self.page_label.pack(side=tk.LEFT, padx=15, pady=8)

        self.hint_label = tk.Label(
            status_bar, 
            text="[Left/Right]: Change Page  |  [+/-]: Change Font Size", 
            font=("Arial", 9, "italic"), 
            fg="#666666", 
            bg="#EAEAEA"
        )
        self.hint_label.pack(side=tk.RIGHT, padx=15, pady=8)

        # 2. Main Content Frame (100% flat scrollable text area)
        content_frame = tk.Frame(self.root, bg="#FFFFFF")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollable Text display area (pure white background, black text)
        self.text_area = scrolledtext.ScrolledText(
            content_frame, 
            wrap=tk.WORD, 
            font=(self.font_family, self.font_size), 
            bg="#FFFFFF", 
            fg="#000000", 
            insertbackground="black", 
            padx=20,
            pady=20,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

    def bind_events(self):
        # Bind page navigation keys
        self.root.bind("<Left>", lambda event: self.navigate_page(-1))
        self.root.bind("<Right>", lambda event: self.navigate_page(1))
        
        # Bind font size scaling keys
        self.root.bind("<plus>", lambda event: self.zoom_font(2))
        self.root.bind("=", lambda event: self.zoom_font(2))  # support '=' key without shift
        self.root.bind("<minus>", lambda event: self.zoom_font(-2))

    def navigate_page(self, direction):
        """Changes the current page pointer and triggers buffer update/refresh."""
        new_ptr = self.current_ptr + direction
        if 0 <= new_ptr <= self.max_page:
            self.current_ptr = new_ptr
            print(f"[Navigate] Moved to page {self.current_ptr}")
            self.update_buffer_and_evict()
            self.refresh_display()
        else:
            print(f"[Navigate] Page boundary reached. Cannot move to {new_ptr}")

    def zoom_font(self, size_change):
        """Resizes the display font dynamically."""
        self.font_size = max(8, min(72, self.font_size + size_change))
        self.text_area.configure(font=(self.font_family, self.font_size))
        print(f"[Font Zoom] Font size changed to {self.font_size}pt")

    def update_buffer_and_evict(self):
        """
        Maintains a sliding window of size 5 around current_ptr:
        [current_ptr - 2, current_ptr - 1, current_ptr, current_ptr + 1, current_ptr + 2].
        Evicts pages outside this window to conserve memory.
        """
        window_pages = set(range(self.current_ptr - 2, self.current_ptr + 3))
        
        # 1. Evict pages outside of the sliding window
        for old_page in list(self.text_buffer.keys()):
            if old_page not in window_pages:
                del self.text_buffer[old_page]
                print(f"[Eviction] Evicted page {old_page} from memory buffer.")
        
        # 2. Pre-load missing pages within the window
        for page in window_pages:
            if 0 <= page <= self.max_page and page not in self.text_buffer:
                page_file = os.path.join(self.stream_dir, f"page_{page:04d}.json")
                if os.path.exists(page_file):
                    try:
                        with open(page_file, 'r', encoding='utf-8') as f:
                            payload = json.load(f)
                            self.text_buffer[page] = payload.get("text", "")
                            print(f"[Buffer Load] Page {page} cached in memory.")
                    except Exception as e:
                        print(f"[Buffer Load Error] Failed to read page {page}: {e}", file=sys.stderr)

    def refresh_display(self):
        """Updates the text widget content and top status bar page count."""
        # Update top page indicator
        display_max = max(0, self.max_page)
        self.page_label.configure(text=f"Page: {self.current_ptr} / {display_max}")
        
        # Retrieve text from buffer or load directly if missing (e.g. fallback)
        text_content = self.text_buffer.get(self.current_ptr)
        
        if text_content is None:
            # Try to load it immediately
            page_file = os.path.join(self.stream_dir, f"page_{self.current_ptr:04d}.json")
            if os.path.exists(page_file):
                try:
                    with open(page_file, 'r', encoding='utf-8') as f:
                        payload = json.load(f)
                        text_content = payload.get("text", "")
                        self.text_buffer[self.current_ptr] = text_content
                except Exception:
                    pass
        
        if text_content is None:
            text_content = f"--- Page {self.current_ptr:04d} Not Captured Yet ---"

        self.text_area.configure(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, text_content)
        self.text_area.configure(state=tk.DISABLED)
        self.text_area.see("1.0")  # Scroll to top of page naturally

    def watch_metadata(self):
        """Periodically polls metadata.json to detect new page captures."""
        while self.is_running:
            time.sleep(0.5)  # Check every 500ms
            if os.path.exists(self.meta_path):
                try:
                    mtime = os.path.getmtime(self.meta_path)
                    if mtime > self.last_meta_mtime:
                        self.last_meta_mtime = mtime
                        
                        # Read metadata file
                        with open(self.meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            new_max_page = int(meta.get("latest_page", self.max_page))
                            
                            if new_max_page != self.max_page:
                                print(f"[Watcher] New page captured. Max page count updated: {self.max_page} -> {new_max_page}")
                                self.max_page = new_max_page
                                
                                # Auto-load buffer for the new state
                                self.root.after(0, self.update_buffer_and_evict)
                                self.root.after(0, self.refresh_display)
                except Exception as e:
                    # File might be temporarily locked, skip this loop
                    pass

    def close(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRMonitorViewApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
