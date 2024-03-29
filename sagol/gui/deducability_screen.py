import tkinter as tk
from functools import partial

from PIL import ImageTk, Image

from sagol.deducability import deduce_by_leave_one_roi_out, plot_brain_image_from_nifty_path
from sagol.evaluate_models import Models
from sagol.gui.globals import STATE

image_of_brain = None
import random


class SimpleTable(tk.Toplevel):
    def __init__(self, dictionary):
        # use black background so it "peeks through" to form grid lines
        tk.Toplevel.__init__(self, background="black")
        self.title('ROIs importance')
        self.geometry('700x' + str(30 + 30 * len(dictionary)))
        self.grab_set()
        self._widgets = []

        index = 0
        l = list(dictionary.items())
        l = [('ROI', 'accuracy without the ROI')] + l

        for (key, val) in l:
            if index == 0:
                col_0 = 'roi'
                col_1 = 'test accuracy without the roi'
            else:
                col_0 = key
                col_1 = val
            current_row = []

            roi_button = tk.Button(self, text=col_0, command=partial(self.show_roi, key), width=10)
            if index == 0:
                roi_button["state"] = "disabled"

            roi_button.grid(row=index, column=0, sticky="nsew", padx=1, pady=1)

            current_row.append(roi_button)
            label = tk.Label(self, text=col_1,
                             borderwidth=0, width=10)
            label.grid(row=index, column=1, sticky="nsew", padx=1, pady=1)
            current_row.append(label)
            self._widgets.append(current_row)
            index += 1

        for column in range(2):
            self.grid_columnconfigure(column, weight=1)

    def set(self, row, column, value):
        widget = self._widgets[row][column]
        widget.configure(text=value)

    def show_roi(self, path_of_roi):
        global image_of_brain
        window = tk.Toplevel()
        window.geometry('800x400')
        window.title("Relevant ROIS")
        window.grab_set()

        path_of_image = 'roi_' + str(random.randint(1, 1000200)) + '.jpg'
        plot_brain_image_from_nifty_path(path_of_roi, path_of_image, plotting_func='plot_roi', title='ROI')

        image_of_brain = ImageTk.PhotoImage(Image.open(path_of_image))

        label = tk.Label(window, image=image_of_brain)
        label.image = image_of_brain  # need to keep the reference of your image to avoid garbage collection
        label.pack(side="bottom", fill="both", expand="yes")

        import os
        if os.path.exists(path_of_image):
            os.remove(path_of_image)
            print("deleted temp file")
        else:
            print("Could not delete since the file does not exist")

        def on_closing():
            window.destroy()
            self.grab_set()

        window.protocol("WM_DELETE_WINDOW", on_closing)


def create_deducability_by_leave_one_roi_out_screan(model_name):
    trained_models = STATE['trained_models']
    relevant_model = trained_models.models[model_name]
    models = Models(ylabels=trained_models.ylabels,
                    roi_paths=trained_models.roi_paths,
                    train_scores=trained_models.train_scores,
                    test_scores={},
                    reverse_contrast_mapping=trained_models.reverse_contrast_mapping,
                    residual_plots={},
                    shape=trained_models.shape,
                    models={model_name: relevant_model},
                    parameters=trained_models.parameters)
    d = deduce_by_leave_one_roi_out(models, STATE['experiment_data_after_split'])
    return SimpleTable(d)
