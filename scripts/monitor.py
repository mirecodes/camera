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
- Dynamic Pagination (2026-06-16):
  Decoupled physical scans from display pages. When text exceeds the screen, it paginates
  into viewable screen chunks. Left/Right arrow keys navigate view pages.
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
        self.font_family = "Georgia"
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
        
        # Memory / Pagination State
        self.combined_text = ""
        self.view_pages = []
        self.current_view_idx = 0
        self.max_page = -1     # highest page index recorded in metadata.json (initialize to -1)
        self.last_meta_mtime = 0.0
        self._last_size = (0, 0)
        self._resize_job = None
        
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
        self.root.title("OCR Document Viewer")
        self.root.geometry("800x650")
        self.root.configure(bg="#FFFFFF")  # Pure white background
        
        self.setup_ui()
        self.bind_events()
        
        # Initial load and paginate
        self.rebuild_combined_text()
        
        # Schedule initial pagination slightly later to ensure window has initialized dimensions
        self.root.after(100, self.safe_paginate_and_refresh)
        
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

        # Scrollable Text display area (pure white background, black text, spacious line-spacing)
        # spacing2=10 gives extremely readable vertical line heights for screen reading
        self.text_area = scrolledtext.ScrolledText(
            content_frame, 
            wrap=tk.WORD, 
            font=(self.font_family, self.font_size), 
            bg="#FFFFFF", 
            fg="#000000", 
            insertbackground="black", 
            padx=35,
            pady=30,
            spacing1=int(self.font_size * 0.45), # dynamic space before paragraphs
            spacing2=int(self.font_size * 0.55), # dynamic space between wrapped lines
            spacing3=int(self.font_size * 0.45), # dynamic space after paragraphs
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
        
        # Bind window resizing to recalculate pagination
        self.root.bind("<Configure>", self.on_resize)

    def navigate_page(self, direction):
        """Changes the current page pointer and triggers buffer update/refresh."""
        new_idx = self.current_view_idx + direction
        if 0 <= new_idx < len(self.view_pages):
            self.current_view_idx = new_idx
            print(f"[Navigate] Moved to view page {self.current_view_idx + 1}")
            self.refresh_display()
        else:
            print(f"[Navigate] Page boundary reached. Cannot move to view page {new_idx + 1}")

    def zoom_font(self, size_change):
        """Resizes the display font dynamically and re-paginates."""
        self.font_size = max(8, min(72, self.font_size + size_change))
        self.text_area.configure(
            font=(self.font_family, self.font_size),
            spacing1=int(self.font_size * 0.45),
            spacing2=int(self.font_size * 0.55),
            spacing3=int(self.font_size * 0.45)
        )
        print(f"[Font Zoom] Font size changed to {self.font_size}pt")
        self.safe_paginate_and_refresh()

    def on_resize(self, event):
        """Handles window resizing with a slight debounce to avoid stuttering."""
        if event.widget == self.root:
            new_size = (event.width, event.height)
            if self._last_size != new_size:
                self._last_size = new_size
                if self._resize_job:
                    self.root.after_cancel(self._resize_job)
                self._resize_job = self.root.after(150, self.safe_paginate_and_refresh)

    def rebuild_combined_text(self):
        """Reconstructs the combined document text from all scanned pages in order."""
        texts = []
        for p in range(self.max_page + 1):
            page_file = os.path.join(self.stream_dir, f"page_{p:04d}.json")
            if os.path.exists(page_file):
                try:
                    with open(page_file, 'r', encoding='utf-8') as f:
                        payload = json.load(f)
                        txt = payload.get("text", "").strip()
                        if txt:
                            texts.append(txt)
                except Exception:
                    pass
        self.combined_text = "\n\n".join(texts)

    def paginate_text(self):
        """
        Dynamically paginates self.combined_text into self.view_pages
        based on the actual physical viewport size of the text area.
        """
        # Save character offset of currently viewed page to preserve view position
        current_char_offset = 0
        if self.view_pages and 0 <= self.current_view_idx < len(self.view_pages):
            prev_text = "".join(self.view_pages[:self.current_view_idx])
            current_char_offset = len(prev_text)

        text_to_paginate = self.combined_text
        if not text_to_paginate:
            self.view_pages = ["--- No Text Captured Yet ---"]
            self.current_view_idx = 0
            return

        self.text_area.configure(state=tk.NORMAL)
        
        # Get actual dimensions of the viewport
        width = self.text_area.winfo_width()
        height = self.text_area.winfo_height()
        
        # Fallback if window is not mapped/rendered yet
        if width <= 10 or height <= 10:
            width = 800
            height = 615

        self.view_pages = []
        remaining_text = text_to_paginate
        
        # Pagination loop using Tkinter layout rendering measurements
        while remaining_text:
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", remaining_text)
            self.root.update_idletasks()
            
            # Find the character index rendered at the bottom-right corner of the visible box
            # 25px offset inside the boundaries accounts for paddings
            bottom_index = self.text_area.index(f"@{width - 35},{height - 30}")
            
            # Fetch content up to that index to calculate text length
            visible_text = self.text_area.get("1.0", bottom_index)
            char_offset = len(visible_text)
            
            if char_offset <= 0 or char_offset >= len(remaining_text):
                self.view_pages.append(remaining_text)
                break
                
            # Smart text splitting: avoid cutting mid-word or mid-sentence
            cut_point = char_offset
            lookback = remaining_text[:char_offset]
            
            # Prefer splitting at double newlines (paragraphs) if nearby (within 250 chars)
            para_idx = lookback.rfind('\n\n', max(0, char_offset - 250))
            if para_idx != -1 and para_idx > 0:
                cut_point = para_idx + 2
            else:
                # Fallback to single newline
                newline_idx = lookback.rfind('\n', max(0, char_offset - 150))
                if newline_idx != -1 and newline_idx > 0:
                    cut_point = newline_idx + 1
                else:
                    # Fallback to space
                    space_idx = lookback.rfind(' ', max(0, char_offset - 40))
                    if space_idx != -1 and space_idx > 0:
                        cut_point = space_idx + 1
            
            self.view_pages.append(remaining_text[:cut_point])
            remaining_text = remaining_text[cut_point:]

        # Restore the viewing page containing the previously viewed characters
        if current_char_offset > 0:
            accum = 0
            found_idx = 0
            for idx, page in enumerate(self.view_pages):
                accum += len(page)
                if accum >= current_char_offset:
                    found_idx = idx
                    break
            self.current_view_idx = min(found_idx, len(self.view_pages) - 1)
        else:
            self.current_view_idx = 0
            
        self.text_area.configure(state=tk.DISABLED)

    def safe_paginate_and_refresh(self):
        """Paginates text and refreshes display elements securely."""
        self.paginate_text()
        self.refresh_display()

    def refresh_display(self):
        """Updates the text widget content and top status bar page count."""
        # Update top page indicator
        total_pages = len(self.view_pages)
        display_current = self.current_view_idx + 1 if total_pages > 0 else 0
        self.page_label.configure(text=f"Page: {display_current} / {total_pages}")
        
        # Retrieve text from current page
        text_content = ""
        if 0 <= self.current_view_idx < len(self.view_pages):
            text_content = self.view_pages[self.current_view_idx]
        else:
            text_content = "--- No Text Captured Yet ---"

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
                                
                                # Rebuild and paginate in GUI main thread
                                self.root.after(0, self.rebuild_combined_text)
                                self.root.after(0, self.safe_paginate_and_refresh)
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
