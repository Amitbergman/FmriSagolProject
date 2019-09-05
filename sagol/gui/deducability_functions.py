import tkinter as tk
from sagol.deducability import deduce_by_coefs as ded_by_coefs, deduce_from_bagging_regressor as ded_from_bagging_regressor,\
    create_brain_nifty_from_weights
from sagol.gui.globals import STATE
from sagol.gui.deducability_screen import create_deducability_by_leave_one_roi_out_screan
import random
from sagol.deducability import plot_brain_image_from_nifty
from PIL import ImageTk, Image

image_of_brain = None


def deduce_by_coefs_or_from_bagging_regressor(model_name, by_coefs):
    trained_models = STATE['trained_models']
    flattened_vector_index_to_voxel = STATE['experiment_data_after_split'].flattened_vector_index_to_voxel \
        if 'experiment_data_after_split' in STATE else STATE['flattened_vector_index_to_voxel']
    if by_coefs:
        model_importances = ded_by_coefs(models={model_name: trained_models.models[model_name]},
                                         first_index_of_contrast=-len(trained_models.reverse_contrast_mapping),
                                         flattened_vector_index_to_voxel=flattened_vector_index_to_voxel)
    else:
        model_importances = ded_from_bagging_regressor(models={model_name: trained_models.models[model_name]},
                                                       first_index_of_contrast=-len(trained_models.reverse_contrast_mapping),
                                                       flattened_vector_index_to_voxel=flattened_vector_index_to_voxel)

    brain_nifty = create_brain_nifty_from_weights(weights=model_importances[model_name], shape=trained_models.shape)

    window = tk.Toplevel()
    window.geometry('1300x700')
    window.title("Voxel importance")
    window.grab_set()

    path_of_image = 'roi_' + str(random.randint(1, 1000200)) + '.jpg'
    plot_brain_image_from_nifty(brain_nifty, path_of_image, plotting_func='plot_glass_brain', title='')

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


def deduce_by_coefs(model_name):
    deduce_by_coefs_or_from_bagging_regressor(model_name, by_coefs=True)


def deduce_from_bagging_regressor(model_name):
    return deduce_by_coefs_or_from_bagging_regressor(model_name, by_coefs=False)


def deduce_from_leave_one_roi_out(model_name):
    create_deducability_by_leave_one_roi_out_screan(model_name)


DEDUCABILITY_CREATORS = {'deduce_by_coefs': deduce_by_coefs,
                         'deduce_from_bagging_regressor': deduce_from_bagging_regressor,
                         'deduce_by_leave_one_roi_out': deduce_from_leave_one_roi_out}


DEDUCABILITY_NAMES = {'deduce_by_coefs': 'Voxel importance',
                      'deduce_from_bagging_regressor': 'Voxel importance',
                      'deduce_by_leave_one_roi_out': 'ROIs Importance'}
