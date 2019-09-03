import tkinter as tk

from PIL import Image, ImageTk

from sagol.gui.initial_window import load_initial_window
from sagol.utils import setup_log_handlers

setup_log_handlers()

root = tk.Tk()
root.title("Sagol")
root.geometry('1300x700')

load_initial_window(root)

root.mainloop()
