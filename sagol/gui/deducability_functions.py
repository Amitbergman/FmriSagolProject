from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sagol.deducability import deduce_by_coefs as ded_by_coefs, create_brain_nifty_from_weights
from sagol.gui.globals import STATE
from sagol.gui.deducability_screen import create_deducability_by_leave_one_roi_out_screan
import random
from sagol.deducability import plot_brain_image_from_nifty
from PIL import ImageTk, Image

def deduce_by_coefs_or_from_bagging_regressor(parent, model_name):
    tab = ttk.Frame(parent)
    trained_models = STATE['trained_models']
    flattened_vector_index_to_voxel = STATE['experiment_data_after_split'].flattened_vector_index_to_voxel
    model_importances = ded_by_coefs(models={model_name: trained_models.models[model_name]},
                                     first_index_of_contrast=-len(trained_models.reverse_contrast_mapping),
                                     flattened_vector_index_to_voxel=flattened_vector_index_to_voxel)
    brain_nifty = create_brain_nifty_from_weights(weights=model_importances[model_name], shape=trained_models.shape)
    brain_frame = get_frame_with_nifty(brain_nifty)
    brain_frame['master'] = tab
    return tab


def deduce_by_coefs(parent, model_name):
    return deduce_by_coefs_or_from_bagging_regressor(parent, model_name)


def deduce_from_bagging_regressor(parent, model_name):
    return deduce_by_coefs_or_from_bagging_regressor(parent, model_name)

def deduce_from_leave_one_roi_out(parent, model_name):
    return create_deducability_by_leave_one_roi_out_screan(parent, model_name)


#DEDUCABILITY_CREATORS = {'deduce_by_coefs': deduce_by_coefs, 'deduce_from_bagging_regressor': deduce_from_bagging_regressor, 'deduce_by_leave_one_roi_out': deduce_from_leave_one_roi_out}
DEDUCABILITY_CREATORS = { 'deduce_by_leave_one_roi_out': deduce_from_leave_one_roi_out}

#DEDUCABILITY_NAMES = {'deduce_by_coefs': 'Voxel importance', 'deduce_from_bagging_regressor': 'Voxel importance'}
DEDUCABILITY_NAMES = {'deduce_by_leave_one_roi_out': 'ROIs Importance'}

def get_frame_with_nifty(nifty):
    global image_of_brain
    frame = ttk.Frame()

    path_of_image = 'roi_' + str(random.randint(1, 1000200)) + '.jpg'
    plot_brain_image_from_nifty(nifty, path_of_image)

    image_of_brain = ImageTk.PhotoImage(Image.open(path_of_image))

    label = ttk.Label(frame, image=image_of_brain)
    label.image = image_of_brain  # need to keep the reference of your image to avoid garbage collection
    label.pack(side="bottom", fill="both", expand="yes")

    return frame