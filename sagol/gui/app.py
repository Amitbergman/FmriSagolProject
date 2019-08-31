import tkinter as tk

from PIL import Image, ImageTk

from sagol.gui.initial_window import load_initial_window
from sagol.utils import setup_log_handlers

setup_log_handlers()

root = tk.Tk()
root.title("Sagol")


background_image = ImageTk.PhotoImage(Image.open('./brain_background.png'))
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
background_label.photo = background_image

load_initial_window(root)

root.mainloop()
