import tkinter as tk

from sagol.gui.actions import open_excel_selector, open_root_dir_selector, display_tasks_selector, \
    create_load_data_button


def load_initial_window(parent):
    excels_button = tk.Button(parent,
                              text="Choose excels",
                              fg="red",
                              command=lambda: open_excel_selector(parent))
    excels_button.pack(side=tk.LEFT)

    root_dir_button = tk.Button(parent,
                                text="Choose root direcotry",
                                command=lambda: on_root_button_click(parent))
    root_dir_button.pack(side=tk.LEFT)


def on_root_button_click(parent):
    open_root_dir_selector(parent)
    contast_selector = display_tasks_selector(parent)
    create_load_data_button(parent, contast_selector)
