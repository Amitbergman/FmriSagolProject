import tkinter as tk

from sagol.gui.initial_window import load_initial_window
from sagol.gui.globals import MAIN_WINDOW
from sagol.utils import setup_log_handlers

setup_log_handlers()

root = tk.Tk()

MAIN_WINDOW.pack()

load_initial_window(MAIN_WINDOW)

root.mainloop()
