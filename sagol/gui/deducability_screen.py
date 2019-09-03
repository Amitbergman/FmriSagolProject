import tkinter as tk
from sagol.deducability import deduce_by_leave_one_roi_out, plot_brain_image_from_nifty
from sagol.gui.globals import STATE
from sagol.evaluate_models import Models
from PIL import ImageTk, Image


class roi_to_accuracy_table(tk.Tk):
    def __init__(self, dictionary):
        tk.Tk.__init__(self)

        t = SimpleTable(self, dictionary)
        t.pack(side="top", fill="x")


class SimpleTable(tk.Frame):
    def __init__(self, parent, dictionary):
        # use black background so it "peeks through" to
        # form grid lines
        tk.Frame.__init__(self, parent, background="black")
        self._widgets = []
        index = 0
        l = list(dictionary.items())
        l = [('roi', 'accuracy without the roi')] + l
        for (key, val) in l:
            if index ==0:
                col_0 = 'roi'
                col_1 = 'test accuracy without the roi'
            else:
                col_0 = key
                col_1 = val
            current_row = []
            roi_button = tk.Button(self, text=col_0, command=lambda: show_roi(key, self), width=10)
            if index==0:
                roi_button["state"] = "disabled"

            roi_button.grid(row=index, column=0, sticky="nsew", padx=1, pady=1)

            current_row.append(roi_button)
            label = tk.Label(self, text=col_1,
                             borderwidth=0, width=10)
            label.grid(row=index, column=1, sticky="nsew", padx=1, pady=1)
            current_row.append(label)
            self._widgets.append(current_row)
            index +=1

        for column in range(2):
            self.grid_columnconfigure(column, weight=1)


    def set(self, row, column, value):
        widget = self._widgets[row][column]
        widget.configure(text=value)

def show_roi(path_of_roi, frame):

    path_of_image = 'name123.jpg'
    plot_brain_image_from_nifty(path_of_roi, path_of_image)

    img = ImageTk.PhotoImage(Image.open(path_of_image))

    panel = tk.Label(frame, image=img)

    panel.pack(side="bottom", fill="both", expand="yes")


def create_deducability_by_leave_on_roi_out_screan(model_name):
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
    roi_to_accuracy_table(d)
