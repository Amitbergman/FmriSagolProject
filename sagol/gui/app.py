import tkinter as tk

from sagol.gui.initial_window import load_initial_window
from sagol.gui.globals import MAIN_WINDOW

root = tk.Tk()

MAIN_WINDOW.pack()

load_initial_window(MAIN_WINDOW)

root.mainloop()
