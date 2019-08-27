import tkinter as tk
from tkinter import filedialog


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

    root_dir_label = tk.Label(parent, text=root_dir)
    root_dir_label.pack()
