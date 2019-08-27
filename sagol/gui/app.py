import tkinter as tk

from sagol.gui.initial_window import load_initial_window

root = tk.Tk()

frame = tk.Frame()
frame.pack()

load_initial_window(frame)

root.mainloop()