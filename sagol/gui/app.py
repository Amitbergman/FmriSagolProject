import tkinter as tk

from sagol.gui.initial_window import load_initial_window
from sagol.utils import setup_log_handlers
import matplotlib

setup_log_handlers()
matplotlib.rcParams.update({'font.size': 6})

root = tk.Tk()
root.title("Sagol")
root.geometry('1300x700')
load_initial_window(root)

root.mainloop()
