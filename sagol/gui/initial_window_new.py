import tkinter as tk
import os
from sagol.gui.globals import STATE
from sagol.evaluate_models import Models
from sagol.gui.models_window import ModelsWindow
from sagol.load_data import create_subject_experiment_data
from sagol.rois import get_available_rois
from tkinter import ttk
from sagol.pre_processing import generate_ylabel_weights


class InitialWindowNew():
    def __init__(self):
        self.window = None
        self.excel_btn = None
        self.excel_paths = None
        self.excel_paths_label = None
        self.root_dir_btn = None
        self.root_dir = None
        self.root_dir_label = None
        self.left_frame = None
        self.right_frame = None
        self.task_names = None
        self.task_selector = None
        self.load_data_btn = None
        self.experiment_data = None
        self.roi_selector = None
        self.choose_roi_btn = None
        self.roi_paths = None
        self.y_label_selector = None
        self.choose_ylabel_btn = None
        self.ylabels = None
        self.weights_selector_frame = None
        self.ylabel_entries = None
        self.set_ylabel_weights_btn = None
        self.weights = None

    def open_excel_selector(self):
        if self.excel_paths is None:
            excel_paths = tk.filedialog.askopenfilenames(initialdir="/", title="Select excels",
                                                     filetypes=(("Excel Files", "*.xls*"), ("Comma Separated Files", "*.csv"),
                                                                ("All files", "*.*")))
            if excel_paths is None or excel_paths == '':
                return
            self.excel_paths = excel_paths
            self.excel_paths_label['text'] = '\n'.join(excel_paths)
            self.excels_btn['text'] = 'Clear selection'

            if self.excel_paths is not None and self.root_dir is not None:
                self.populate_tasks_and_rois()
        else:
            self.clear_selection()
            self.excel_paths = None
            self.excel_paths_label['text'] = ''
            self.excels_btn['text'] = 'Choose excels'

    def open_root_dir_selector(self):
        if self.root_dir is None:
            root_dir = tk.filedialog.askdirectory(initialdir="/", title="Select root directory")
            if root_dir is None or root_dir == '':
                return
            self.root_dir = root_dir
            self.root_dir_label['text'] = root_dir
            self.root_dir_btn['text'] = 'Clear selection'

            if self.excel_paths is not None and self.root_dir is not None:
                self.populate_tasks_and_rois()
        else:
            self.clear_selection()
            self.root_dir = None
            self.root_dir_label['text'] = ''
            self.root_dir_btn['text'] = 'Choose root direcotry'

    def clear_selection(self):
        self.task_selector.delete(0, tk.END)
        self.task_names = None
        self.load_data_btn['state'] = tk.DISABLED
        self.load_data_btn['text'] = 'Load data'
        self.roi_selector.delete(0, tk.END)
        self.roi_paths = None
        self.choose_roi_btn['state'] = tk.DISABLED
        self.choose_roi_btn['text'] = 'Choose ROIs'
        self.experiment_data = None
        self.clear_ylabels()

    def populate_tasks_and_rois(self):
        self.load_data_btn['state'] = tk.NORMAL
        task_names = [dir_name for dir_name in os.listdir(self.root_dir) if
                      not dir_name.startswith('.') and os.path.isdir(os.path.join(self.root_dir, dir_name))]
        self.task_selector.delete(0, tk.END)
        max_len_item = 0
        for task in task_names:
            self.task_selector.insert(tk.END, task)
            self.task_selector.pack()
            max_len_item = max(max_len_item, len(task))
        self.task_selector.config(width=max_len_item)

        self.choose_roi_btn['state'] = tk.NORMAL
        self.roi_selector.delete(0, tk.END)
        roi_paths = get_available_rois()
        max_len_item = 0
        for roi in roi_paths:
            self.roi_selector.insert(tk.END, roi)
            max_len_item = max(max_len_item, len(roi))
        self.roi_selector.config(width=max_len_item)

    def open_load_models_selector(self):
        models_paths = tk.filedialog.askopenfilenames(initialdir="/", title="Select models",
                                                      filetypes=(("All files", "*"),))
        if models_paths is None or models_paths == '':
            return

        models = Models()
        additional_params = {}
        for model_path in models_paths:
            additional_params = models.load_model(model_path)

        STATE['trained_models'] = models
        STATE['ylabels'] = models.ylabels
        STATE['roi_paths'] = models.roi_paths
        STATE.pop('tasks_and_contrasts', None)
        STATE.pop('experiment_data', None)
        STATE['weights'] = additional_params['weights'] or [1 / len(STATE['ylabels']) for _ in
                                                            range(len(STATE['ylabels']))]
        STATE['flattened_vector_index_to_voxel'] = additional_params['flattened_vector_index_to_voxel']
        STATE['is_load'] = True

        model_window = ModelsWindow()
        model_window.open_models_window()
        self.window.destroy()

    def on_load_data_click(self):
        selected_task_names = [self.task_selector.get(idx) for idx in self.task_selector.curselection()]
        if not selected_task_names:
            return
        self.task_names = selected_task_names
        self.load_data_btn['text'] = 'Loading data...'

        self.clear_ylabels()
        self.experiment_data = create_subject_experiment_data(excel_paths=self.excel_paths,
                                                              nifty_dirs=[os.path.join(self.root_dir, task) for task in
                                                                          selected_task_names])
        self.populate_ylabels()
        self.load_data_btn['text'] = 'Load again'

    def clear_ylabels(self):
        self.y_label_selector.delete(0, tk.END)
        self.ylabels = None
        self.choose_ylabel_btn['state'] = tk.DISABLED
        self.choose_ylabel_btn['text'] = 'Choose y labels'
        self.weights = None
        self.finish_btn['state'] = tk.DISABLED
        if self.weights_selector_frame is not None:
            self.weights_selector_frame.grid_forget()
            self.weights_selector_frame.destroy()
            self.weights_selector_frame = None

    def populate_ylabels(self):
        self.choose_ylabel_btn['state'] = tk.NORMAL
        available_ylabels = self.experiment_data.available_ylabels
        self.y_label_selector.delete(0, tk.END)
        max_len_item = 0
        for ylabel in available_ylabels:
            self.y_label_selector.insert(tk.END, ylabel)
            max_len_item = max(max_len_item, len(ylabel))
        self.y_label_selector.config(width=max_len_item)

    def on_choose_ylabel_click(self):
        selected_ylabels = [self.y_label_selector.get(idx) for idx in self.y_label_selector.curselection()]
        if not selected_ylabels:
            return
        self.ylabels = selected_ylabels

        self.choose_ylabel_btn['text'] = 'Choose y labels again'
        self.weights = None
        self.finish_btn['state'] = tk.DISABLED

        self.display_weights()

    def display_weights(self):
        if self.weights_selector_frame is None:
            self.weights_selector_frame = ttk.Frame(self.right_frame)
            self.weights_selector_frame.pack()
            self.ylabel_entries = []

            for i, ylabel in enumerate(self.ylabels):
                ylabel_text = tk.Label(self.weights_selector_frame, text=f'{ylabel}: ')
                ylabel_text.grid(row=i, column=0)
                ylabel_entry = tk.Entry(self.weights_selector_frame, width=20, validate='focus')
                ylabel_entry.grid(row=i, column=1)
                # Default to all equal weights.
                ylabel_entry.insert(tk.END, str(1 / len(self.ylabels)))
                self.ylabel_entries.append((ylabel, ylabel_entry))

            self.set_ylabel_weights_btn = tk.Button(self.weights_selector_frame, text="Set y label weights", fg="green",
                                               command=self.on_set_ylabel_weights_click)
            self.set_ylabel_weights_btn.grid(row=i+1, column=0)
        else:
            self.weights_selector_frame.grid_forget()
            self.weights_selector_frame.destroy()
            self.weights_selector_frame = None
            self.display_weights()

    def on_set_ylabel_weights_click(self):
        ylabels = [ylabel_entry[0] for ylabel_entry in self.ylabel_entries]
        ylabel_to_weight = {ylabel_entry[0]: float(ylabel_entry[1].get()) for ylabel_entry in self.ylabel_entries}
        self.weights = generate_ylabel_weights(ylabels, ylabel_to_weight)
        self.set_ylabel_weights_btn['text'] = 'Set y label weights again'
        self.finish_btn['state'] = tk.NORMAL

    def on_choose_roi_click(self):
        if self.roi_paths is None:
            self.roi_paths = [self.roi_selector.get(idx) for idx in self.roi_selector.curselection()]
            self.choose_roi_btn['text'] = 'Clear selection'
        else:
            self.roi_paths = None
            self.choose_roi_btn['text'] = 'Choose ROIs'

    def finish(self):
        if self.roi_paths is None:
            message_box = tk.messagebox.askquestion('No ROIs selected',
                                                    'No ROIs selected. Are you sure you want to continue without selecting ROIs?',
                                                    icon='warning')
            if message_box == 'no':
                return
        STATE['is_load'] = False
        STATE['ylabels'] = self.ylabels
        self.roi_paths = [] if self.roi_paths is None else self.roi_paths
        STATE['roi_paths'] = self.roi_paths
        STATE['experiment_data'] = self.experiment_data
        STATE['trained_models'] = Models(ylabels=self.ylabels, roi_paths=self.roi_paths, shape=self.experiment_data.shape)
        STATE['tasks_and_contrasts'] = self.experiment_data.tasks_metadata
        STATE['weights'] = self.weights

        model_window = ModelsWindow()
        self.window.destroy()
        model_window.open_models_window()

    def open(self):
        self.window = tk.Tk()
        self.window.title("Sagol")
        self.window.geometry('1300x700')

        self.excels_btn = tk.Button(self.window, text="Choose excels", fg="green", width=40, height=3,
                                  command=self.open_excel_selector)
        self.excels_btn.grid(row=0, column=0)
        self.excel_paths_label = tk.Label(self.window, text='')
        self.excel_paths_label.grid(row=0, column=1)

        self.root_dir_btn = tk.Button(self.window, text="Choose root direcotry", fg='blue', width=40, height=3,
                                    command=self.open_root_dir_selector)
        self.root_dir_btn.grid(row=1, column=0)
        self.root_dir_label = tk.Label(self.window, text='')
        self.root_dir_label.grid(row=1, column=1)

        load_models_button = tk.Button(self.window, text="Load models", fg="black", width=40, height=3,
                                       command=self.open_load_models_selector)
        load_models_button.grid(row=2, column=0)

        self.left_frame = ttk.Frame(self.window)
        self.left_frame.grid(row=3, column=0)

        self.right_frame = ttk.Frame(self.window)
        self.right_frame.grid(row=3, column=2)

        self.task_selector = tk.Listbox(self.left_frame, selectmode=tk.MULTIPLE)
        self.task_selector.pack(fill=tk.X, expand=True)

        self.load_data_btn = tk.Button(self.left_frame, text='Load data', fg="green", command=self.on_load_data_click)
        self.load_data_btn['state'] = tk.DISABLED
        self.load_data_btn.pack()

        self.roi_selector = tk.Listbox(self.left_frame, selectmode=tk.MULTIPLE)
        self.roi_selector.pack(fill=tk.X, expand=True)

        self.choose_roi_btn = tk.Button(self.left_frame, text="Choose ROIs", command=self.on_choose_roi_click)
        self.choose_roi_btn['state'] = tk.DISABLED
        self.choose_roi_btn.pack()

        self.y_label_selector = tk.Listbox(self.right_frame, selectmode=tk.MULTIPLE)
        self.y_label_selector.pack(fill=tk.X, expand=True)

        self.choose_ylabel_btn = tk.Button(self.right_frame, text='Choose y labels', command=self.on_choose_ylabel_click)
        self.choose_ylabel_btn['state'] = tk.DISABLED
        self.choose_ylabel_btn.pack()

        self.finish_btn = tk.Button(self.window, text='Finish', command=self.finish, width=30, height=2)
        self.finish_btn['state'] = tk.DISABLED
        self.finish_btn.grid(row=7, column=1)

        self.window.mainloop()
