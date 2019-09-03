import os
import tkinter as tk
from tkinter import filedialog, ttk

from sagol.evaluate_models import Models
from sagol.gui.globals import STATE
from sagol.gui.models_window import ModelsWindow
from sagol.load_data import create_subject_experiment_data
from sagol.rois import get_available_rois
from sagol.run_models import generate_ylabel_weights


def load_initial_window(parent):
    excels_button = tk.Button(parent,
                              text="Choose excels",
                              fg="green",
                              command=lambda: open_excel_selector(parent))
    excels_button.grid(row=0, column=0)

    root_dir_button = tk.Button(parent,
                                text="Choose root direcotry",
                                fg='blue',
                                command=lambda: on_root_button_click(parent))
    root_dir_button.grid(row=0, column=1)

    load_models_button = tk.Button(parent,
                                   text="Load models",
                                   fg="black",
                                   command=lambda: open_load_models_selector())
    load_models_button.grid(row=0, column=2)


def on_root_button_click(parent):
    open_root_dir_selector(parent)
    contast_selector = display_tasks_selector(parent)
    create_load_data_button(parent, contast_selector)


def on_load_data_click(parent, contrasts_selector, btn_text):
    selected_task_names = [contrasts_selector.get(idx) for idx in contrasts_selector.curselection()]
    btn_text.set('Loading data...')

    STATE['experiment_data'] = create_subject_experiment_data(
        excel_paths=STATE['excel_paths'],
        nifty_dirs=[os.path.join(STATE['root_dir'], task) for task in selected_task_names])

    btn_text.set('Loaded data successfully!')

    roi_selector = display_roi_selector(parent)

    choose_roi_button = tk.Button(parent,
                                  text="Choose ROIs",
                                  command=lambda: on_choose_roi_click(parent, roi_selector))
    choose_roi_button.grid(row=5)


def on_choose_roi_click(parent, roi_selector):
    roi_paths = [roi_selector.get(idx) for idx in roi_selector.curselection()]
    STATE['roi_paths'] = roi_paths

    ylabels_frame = ttk.Frame(parent)
    ylabels_frame.grid(row=0, column=5)

    display_ylabel_selector(ylabels_frame)


def open_excel_selector(parent):
    excel_paths = filedialog.askopenfilenames(initialdir="/", title="Select excels",
                                              filetypes=(
                                                  ("Excel Files", "*.xls*"), ("Comma Separated Files", "*.csv"),
                                                  ("All files", "*.*")))
    STATE['excel_paths'] = excel_paths

    excel_paths_label = tk.Label(parent, text='\n'.join(excel_paths))
    excel_paths_label.grid(row=1, column=0)


def open_root_dir_selector(parent):
    root_dir = filedialog.askdirectory(initialdir="/", title="Select root directory")
    STATE['root_dir'] = root_dir

    task_names = [dir_name for dir_name in os.listdir(root_dir) if
                  not dir_name.startswith('.') and os.path.isdir(os.path.join(root_dir, dir_name))]
    STATE['task_names'] = task_names

    root_dir_label = tk.Label(parent, text=root_dir)
    root_dir_label.grid(row=1, column=1)


def open_load_models_selector():
    models_paths = filedialog.askopenfilenames(initialdir="/", title="Select models",
                                               filetypes=(("All files", "*"),))

    models = Models()
    for model_path in models_paths:
        models.load_model(model_path)

    STATE['trained_models'] = models
    STATE['ylabels'] = models.ylabels
    STATE['roi_paths'] = models.roi_paths
    STATE.pop('tasks_and_contrasts', None)
    STATE.pop('experiment_data', None)
    STATE['weights'] = [1 / len(STATE['ylabels']) for _ in range(len(STATE['ylabels']))]
    STATE['is_load'] = True

    model_window = ModelsWindow()
    model_window.open_models_window()


def display_tasks_selector(parent):
    contrast_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE)
    for task in STATE['task_names']:
        contrast_selector.insert(tk.END, task)
    contrast_selector.grid(row=2, column=0)
    return contrast_selector


def create_load_data_button(parent, contrasts_selector):
    btn_text = tk.StringVar()
    btn_text.set('Load Data')
    load_data_button = tk.Button(parent,
                                 textvariable=btn_text,
                                 fg="green",
                                 command=lambda: on_load_data_click(parent, contrasts_selector, btn_text))
    load_data_button.grid(row=3)


def display_roi_selector(parent):
    available_rois = get_available_rois()
    roi_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE, width=max(len(roi) for roi in available_rois))

    for roi in get_available_rois():
        roi_selector.insert(tk.END, roi)
    roi_selector.grid(row=4)
    return roi_selector


def display_ylabel_selector(parent):
    available_ylabels = STATE['experiment_data'].available_ylabels
    y_label_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE,
                                  width=max(len(ylabel) for ylabel in available_ylabels))

    for ylabel in available_ylabels:
        y_label_selector.insert(tk.END, ylabel)
    y_label_selector.grid(row=0, column=0)
    choose_roi_button = tk.Button(parent,
                                  text="Choose Y labels",
                                  command=lambda: on_choose_ylabel_click(parent, y_label_selector))
    choose_roi_button.grid(row=1, column=0)
    return y_label_selector


def on_choose_ylabel_click(parent, ylabels_selector):
    selected_ylabels = [ylabels_selector.get(idx) for idx in ylabels_selector.curselection()]
    STATE['ylabels'] = selected_ylabels

    display_weights_selector(parent, selected_ylabels)


def display_weights_selector(parent, ylabels):
    weights_selector = ttk.Frame(parent)
    weights_selector.grid(row=2, column=0)

    ylabel_entries = []

    i = 0
    for i, ylabel in enumerate(ylabels):
        ylabel_text = tk.Label(parent, text=f'{ylabel}: ')
        ylabel_text.grid(row=i + 2, column=0)
        ylabel_entry = tk.Entry(parent, width=10, validate='focus')
        ylabel_entry.grid(row=i + 2, column=1)
        # Default to all equal weights.
        ylabel_entry.insert(tk.END, str(1 / len(ylabels)))
        ylabel_entries.append((ylabel, ylabel_entry))

    set_ylabel_weights = tk.Button(parent,
                                   text="Set ylabel weights",
                                   fg="green",
                                   command=lambda: on_weights_selector_click(ylabel_entries))
    set_ylabel_weights.grid(row=i + 3, column=0)


def on_weights_selector_click(ylabel_entries):
    STATE['is_load'] = False
    STATE['trained_models'] = Models(ylabels=STATE['ylabels'],
                                     roi_paths=STATE['roi_paths'],
                                     shape=STATE['experiment_data'].shape)
    STATE['tasks_and_contrasts'] = STATE['experiment_data'].tasks_metadata

    ylabels = [ylabel_entry[0] for ylabel_entry in ylabel_entries]
    ylabel_to_weight = {ylabel_entry[0]: float(ylabel_entry[1].get()) for ylabel_entry in ylabel_entries}

    STATE['weights'] = generate_ylabel_weights(ylabels, ylabel_to_weight)
    model_window = ModelsWindow()
    model_window.open_models_window()
