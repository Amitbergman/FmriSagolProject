import tkinter.filedialog

STATE = {}

def open_excel_selector():
    tkinter.filedialog.askopenfilenames()

root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))


def open_root_dir_selector():
    pass