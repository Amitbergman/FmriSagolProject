import tkinter as tk

from sagol.gui.actions import open_excel_selector, open_root_dir_selector


def load_initial_window(parent):
    excels_button = tk.Button(parent,
                              text="Choose excels",
                              fg="red",
                              command=lambda: open_excel_selector(parent))
    excels_button.pack(side=tk.LEFT)

    root_dir_button = tk.Button(parent,
                                text="Choose root direcotry",
                                command=lambda: open_root_dir_selector(parent))
    root_dir_button.pack(side=tk.LEFT)
