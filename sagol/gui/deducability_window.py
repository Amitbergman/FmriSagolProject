import tkinter as tk
from tkinter import ttk
from sagol.models.utils import DEDUCABILITY_OPTIONS
from sagol.gui.globals import STATE
from sagol.gui.deducability_functions import DEDUCABILITY_CREATORS, DEDUCABILITY_NAMES



class DeducabilityWindow():
    def __init__(self, model_name):
        self.model_name = model_name

    def open(self):
        window = tk.Tk()
        window.geometry('1300x700')
        window.title("Deducability")

        tab_control = ttk.Notebook(window)

        for deducability_option in DEDUCABILITY_OPTIONS[self.model_name]:
            if deducability_option in STATE['unavailable_deducabilities']:
                continue
            tab = DEDUCABILITY_CREATORS[deducability_option](tab_control, self.model_name)
            tab_control.add(tab, text=DEDUCABILITY_NAMES[deducability_option])

        tab_control.pack(expand=True, fill='both')
