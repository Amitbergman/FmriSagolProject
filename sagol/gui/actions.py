import tkinter as tk
from tkinter import filedialog
import os

STATE = {}


def open_excel_selector(parent):
    excel_paths = filedialog.askopenfilenames(initialdir="/", title="Select excels",
                                              filetypes=(
                                                  ("Excel Files", "*.xls*"), ("Comma Separated Files", "*.csv"),
                                                  ("All files", "*.*")))
    STATE['excel_paths'] = excel_paths

    excel_paths_label = tk.Label(parent, text='\n'.join(excel_paths))
    excel_paths_label.pack()


def open_root_dir_selector(parent):
    root_dir = filedialog.askdirectory(initialdir="/", title="Select root directory")
    STATE['root_dir'] = root_dir
    STATE['tasks_metadata'] = {}
    tasks_list = [dir_name for dir_name in os.listdir(root_dir) if not dir_name.startswith('.') and os.path.isdir(os.path.join(root_dir, dir_name))]
    for task in tasks_list:
        cur_path = os.path.join(root_dir, task)
        contrasts = [dir_name for dir_name in os.listdir(cur_path) if not dir_name.startswith('.') and os.path.isdir(os.path.join(cur_path, dir_name))]
        STATE['tasks_metadata'][task] = contrasts
    root_dir_label = tk.Label(parent, text=root_dir)
    root_dir_label.pack()

