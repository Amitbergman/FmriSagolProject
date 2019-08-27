import tkinter as tk
from tkinter import filedialog
import os

from sagol.gui.data_filtering_window import open_data_filtering_window
from sagol.gui.globals import STATE


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

    task_names = [dir_name for dir_name in os.listdir(root_dir) if
                  not dir_name.startswith('.') and os.path.isdir(os.path.join(root_dir, dir_name))]
    STATE['task_names'] = task_names

    root_dir_label = tk.Label(parent, text=root_dir)
    root_dir_label.pack()


def display_tasks_selector(parent):
    contrast_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE)
    for task in STATE['task_names']:
        contrast_selector.insert(tk.END, task)
    contrast_selector.pack()
    return contrast_selector


def create_load_data_button(parent, contrasts_selector):
    load_data_button = tk.Button(parent,
                                 text="Load data",
                                 fg="green",
                                 command=lambda: open_data_filtering_window(contrasts_selector))
    load_data_button.pack(side=tk.LEFT)
