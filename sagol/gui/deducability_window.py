import tkinter as tk
from sagol.models.utils import DEDUCABILITY_OPTIONS
from sagol.gui.globals import STATE
from sagol.gui.deducability_functions import DEDUCABILITY_CREATORS, DEDUCABILITY_NAMES
from functools import partial


class DeducabilityWindow(tk.Toplevel):
    def __init__(self, model_name):
        tk.Toplevel.__init__(self)
        self.model_name = model_name
        self.title('Deducability')
        self.grab_set()

        def open_deducability(deducability_option, model_name):
            deducability_window = DEDUCABILITY_CREATORS[deducability_option](model_name)

            def on_closing():
                deducability_window.destroy()
                self.grab_set()
            deducability_window.protocol("WM_DELETE_WINDOW", on_closing)

        for deducability_option in DEDUCABILITY_OPTIONS[self.model_name]:
            if deducability_option in STATE['unavailable_deducabilities']:
                continue
            btn = tk.Button(self, text=DEDUCABILITY_NAMES[deducability_option],
                            command=partial(open_deducability, deducability_option, self.model_name))
            btn.pack()
