import tkinter as tk
from tkinter import ttk
from sagol.gui.globals import STATE
from sagol.run_models import generate_experiment_data_after_split
from sagol.models.utils import AVAILABLE_MODELS, is_valid_param, get_parameter_remark
from sagol.gui.classes import UntrainedModels
from sagol.gui.utils import load_test_data
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sagol.gui.deducability_window import DeducabilityWindow

INVALID_PARAM_MESSAGE = "At least one of the parameters is invalid"


def prepare_data():
    trained_models = STATE['trained_models']
    STATE['experiment_data_after_split'], STATE['experiment_data_after_split_3d'], \
        STATE['trained_models'].reverse_contrast_mapping = generate_experiment_data_after_split(
        experiment_data=STATE['experiment_data'], roi_paths=trained_models.roi_paths,
        tasks_and_contrasts=STATE['tasks_and_contrasts'], ylabels=trained_models.ylabels, weights=STATE['weights'],
        should_use_rois=not (not trained_models.roi_paths))


class ModelsWindow:
    def __init__(self):
        STATE['untrained_models'] = UntrainedModels()
        if not STATE['is_load']:
            prepare_data()
        self.results_frames = {}
        self.params_valid = {model_name: {} for model_name in AVAILABLE_MODELS}
        self.trained_lately = set()
        self.test_data_loaded_labels = {}
        self.tab_clicked_funcs = {}
        self.button_funcs = {model_name: {} for model_name in AVAILABLE_MODELS}
        self.clicked_button = None
        self.times_test_data_loaded = 0 if STATE['is_load'] else 1
        self.test_score_labels = {}
        self.res_plot_canvases = {}

    def update_trained_lately(self, model_names):
        for name in model_names:
            if name not in self.trained_lately:
                self.trained_lately.add(name)

    def check_params(self, model_name):
        for val in self.params_valid[model_name].values():
            if not val:
                return False
        return True


    # When this function is called, STATE['trained_models'] should be populated by an object of the class
    # 'sagol.evaluate_models.Models'. The attributes 'ylabels', 'roi_paths', 'reverse_contrast_mapping' and 'shape'
    # must be populated, even if no model was trained yet!
    # STATE['experiment_data'] should be populated by an object of the class 'sagol.load_data.ExperimentData',
    # which represent the data that the user chose in the former window, or by None if STATE['is_load'] == True.
    # STATE['tasks_and_contrasts'] should be populated correctly
    # STATE['weights'] should be populated correctly
    # Lastly, STATE['is_load'] should be True iif the user chose to load existing models in the former window.
    def open_models_window(self):
        window = tk.Tk()
        window.geometry('1300x700')
        window.title("Models")

        info_frame = ttk.Frame(window)
        info_frame.pack(expand=True)
        y_labels_lbl = tk.Label(info_frame, text='y labels: ' + ' + '.join(
            [str(STATE['weights'][i]) + '*' + y for
             i, y in enumerate(STATE['trained_models'].ylabels)]))
        y_labels_lbl.pack(expand=True)
        rois_lbl = tk.Label(info_frame, text='ROIs: ' + ', '.join([os.path.basename(
            os.path.normpath(path)) for path in STATE['trained_models'].roi_paths]))
        rois_lbl.pack(expand=True)
        rois_lbl.bind('<Configure>', lambda e: rois_lbl.config(wraplength=rois_lbl.winfo_width()))

        tab_control = ttk.Notebook(window)

        tab_general = ttk.Frame(tab_control)
        tab_control.add(tab_general, text='General')

        self.general_list = tk.Listbox(tab_general, selectmode=tk.MULTIPLE)

        def update_general_list():
            self.list_models = []
            self.general_list.delete(0, tk.END)
            trained_models = STATE['trained_models']
            for i, model_name in enumerate(STATE['untrained_models'].models):
                model_text = ': ' + str(trained_models.parameters[model_name]) if model_name in trained_models.models else ''
                if model_name in trained_models.train_scores:
                    model_text += ', train score: ' + '%.2f' % trained_models.train_scores[model_name]
                if model_name in trained_models.test_scores:
                    model_text += ', test score: ' + '%.2f' % trained_models.test_scores[model_name]
                self.general_list.insert(tk.END, model_name + model_text)
                self.list_models.append(model_name)
                self.general_list.pack(fill=tk.BOTH)
        update_general_list()

        def train_or_train_test_selected_clicked(train_only):
            model_names = []
            for i in self.general_list.curselection():
                model_name = self.list_models[i]
                model_names.append(model_name)
                if not self.check_params(model_name):
                    tk.messagebox.showinfo("Error", INVALID_PARAM_MESSAGE)
                    return
            STATE['untrained_models'].generate_models(model_names, train_only)
            self.update_trained_lately(model_names)
            update_general_list()

        def load_test_general_clicked():
            excel_paths = tk.filedialog.askopenfilenames(initialdir="/", title="Select excels",
                                                         filetypes=(
                                                             ("Excel Files", "*.xls*"),
                                                             ("Comma Separated Files", "*.csv"),
                                                             ("All files", "*.*")))
            if excel_paths is None or excel_paths == '':
                return
            nifty_dir = tk.filedialog.askdirectory(initialdir="/", title="Select test data directory")
            if nifty_dir is None or nifty_dir == '':
                return
            if load_test_data(excel_paths, nifty_dir, True if self.times_test_data_loaded == 0 else False):
                self.times_test_data_loaded += 1
                load_test_general_lbl['text'] = 'Test data presents (' + str(self.times_test_data_loaded) + ')'
            else:
                tk.messagebox.showinfo("Error", 'There is not even one matching task and contrast between ' +
                                       'loaded test data and original training data')

        train_selected_frame = ttk.Frame(tab_general)
        train_selected_frame.pack(expand=True)
        train_selected_btn = tk.Button(train_selected_frame, text='Train selected models',
                                       command=lambda: train_or_train_test_selected_clicked(True))
        train_selected_btn.grid(column=0, row=0)
        train_test_selected_btn = tk.Button(train_selected_frame, text='Train and test selected models',
                                            command=lambda: train_or_train_test_selected_clicked(False))
        train_test_selected_btn.grid(column=2, row=0)
        if STATE['is_load']:
            train_selected_btn['state'] = 'disabled'
            train_test_selected_btn['state'] = 'disabled'
        load_test_general_btn = tk.Button(train_selected_frame, text='Load test data', command=load_test_general_clicked)
        load_test_general_btn.grid(column=1, row=1)
        load_test_general_lbl = tk.Label(train_selected_frame, text='' if self.times_test_data_loaded == 0 else
            'Test data presents (' + str(self.times_test_data_loaded) + ')')
        load_test_general_lbl.grid(column=1, row=2)

        def create_tab(parent, name, untrained_model):
            tab = ttk.Frame(parent)
            parent.add(tab, text=name)
            tab_inner_frame = ttk.Frame(tab)
            tab_inner_frame.pack(side=tk.TOP)
            left_frame = ttk.Frame(tab_inner_frame)
            left_frame.pack(side=tk.LEFT)
            params_frame = ttk.Frame(left_frame)
            params_frame.grid(column=0, row=0)

            def populate_params_frame(parent, params):
                choose_params_lbl = tk.Label(parent, text='Choose parameters:')
                choose_params_lbl.grid(column=0, row=0)
                for i, (p, v) in enumerate(params.items()):

                    def create_param_comp(parent, i, p, v):
                        self.params_valid[name][p] = True
                        param_lbl = tk.Label(parent, text=p + ': ')
                        param_lbl.grid(column=0, row=i + 1)
                        param_entry = tk.Entry(parent, width=10, validate='focus')

                        def update_param():
                            val = param_entry.get()
                            is_valid, value = is_valid_param(name, p, val)
                            if is_valid:
                                STATE['untrained_models'].models[name].parameters[p] = value
                            self.params_valid[name][p] = is_valid
                            param_entry.config(bg='white' if is_valid else 'red')
                            return is_valid

                        param_entry.config(validatecommand=update_param)
                        param_entry.insert(tk.END, v)
                        param_entry.grid(column=1, row=i + 1)
                        remark = get_parameter_remark(name, p)
                        param_remark_lbl = tk.Label(parent, text='' if remark == '' or remark is None else '(' + remark + ')')
                        param_remark_lbl.grid(column=2, row=i + 1)
                        update_param()

                    create_param_comp(parent, i, p, v)

            populate_params_frame(params_frame, untrained_model.parameters)

            def button_clicked(button):
                self.clicked_button = button['text']
                button.focus_set()

            def draw_res_plot(res_plot, results_frame):
                if name in self.res_plot_canvases:
                    self.res_plot_canvases[name].get_tk_widget().destroy()
                res_plot_canvas = FigureCanvasTkAgg(res_plot, master=results_frame)
                res_plot_canvas.get_tk_widget().grid(column=2, row=5)
                res_plot_canvas.draw()
                self.res_plot_canvases[name] = res_plot_canvas

            def create_results_frame():
                results_frame = ttk.Frame(tab_inner_frame)
                self.results_frames[name] = results_frame
                results_frame.pack(side=tk.LEFT)

                trained_model_lbl = tk.Label(results_frame, text='Trained model:')
                trained_model_lbl.grid(column=0, row=0)

                params_chosen_frame = ttk.Frame(results_frame)
                params_chosen_frame.grid(column=2, row=1)
                for i, (p, v) in enumerate(STATE['trained_models'].parameters[name].items()):
                    param_lbl = tk.Label(params_chosen_frame, text=p + ': ' + str(v))
                    param_lbl.grid(column=i % 2, row=i // 2)

                train_score_lbl = tk.Label(results_frame,
                                           text='Train score: ' + STATE['trained_models'].get_train_score(name, True))
                train_score_lbl.grid(column=1, row=2)
                test_score_lbl = tk.Label(results_frame,
                                          text='Test score: ' + STATE['trained_models'].get_test_score(name, True))
                test_score_lbl.grid(column=2, row=2)
                self.test_score_labels[name] = test_score_lbl

                test_data_loaded_lbl = tk.Label(results_frame, text='' if self.times_test_data_loaded == 0 else
                    'Test data presents (' + str(self.times_test_data_loaded) + ')')
                test_data_loaded_lbl.grid(column=2, row=4)
                self.test_data_loaded_labels[name] = test_data_loaded_lbl

                def load_test_data_clicked():
                    excel_paths = tk.filedialog.askopenfilenames(initialdir="/", title="Select excels",
                                                filetypes=(
                                                    ("Excel Files", "*.xls*"), ("Comma Separated Files", "*.csv"),
                                                    ("All files", "*.*")))
                    if excel_paths is None or excel_paths == '':
                        return False
                    nifty_dir = tk.filedialog.askdirectory(initialdir="/", title="Select test data directory")
                    if nifty_dir is None or nifty_dir == '':
                        return False
                    if load_test_data(excel_paths, nifty_dir, True if self.times_test_data_loaded == 0 else False):
                        self.times_test_data_loaded += 1
                        test_data_loaded_lbl['text'] = 'Test data presents (' + str(self.times_test_data_loaded) + ')'
                    else:
                        tk.messagebox.showinfo("Error", 'There is not even one matching task and contrast between ' +
                                                        'loaded test data and original training data')
                        return False
                    return True

                def test_clicked():
                    if self.times_test_data_loaded == 0:
                        if not load_test_data_clicked():
                            return
                    untrained_model = STATE['untrained_models'].models[name]
                    data = STATE['experiment_data_after_split_3d'] if untrained_model.is_3d else STATE[
                        'experiment_data_after_split']
                    test_score, res_plot = STATE['trained_models'].test(name, data.x_test, data.y_test)
                    self.test_score_labels[name]['text'] = test_score
                    draw_res_plot(res_plot, results_frame)

                def open_deducability():
                    STATE['unavailable_deducabilities'] = set()
                    if STATE['is_load']:
                        STATE['unavailable_deducabilities'].add('deduce_by_leave_one_roi_out')
                    if name in {'svr', 'nusvr'} and not STATE['trained_models'].parameters[name]['kernel'] == 'linear':
                        STATE['unavailable_deducabilities'].add('deduce_by_coefs')

                    if 'experiment_data_after_split' in STATE:
                        STATE['flattened_vector_index_to_voxel'] = \
                            STATE['experiment_data_after_split'].flattened_vector_index_to_voxel
                    DeducabilityWindow(model_name=name)

                def save_clicked():
                    file_path = tk.filedialog.asksaveasfile(initialdir="/", title="Save model", mode=tk.W)
                    if file_path is None or file_path == '':
                        return
                    flattened_vector_index_to_voxel = STATE['flattened_vector_index_to_voxel'] \
                        if 'flattened_vector_index_to_voxel' in STATE else STATE['experiment_data_after_split'].flattened_vector_index_to_voxel
                    STATE['trained_models'].save_model(model_name=name, file_path=file_path.name,
                                                       additional_save_dict={'weights': STATE['weights'],
                                                                             'flattened_vector_index_to_voxel': flattened_vector_index_to_voxel})

                load_test_btn = tk.Button(results_frame, text='Load test data', command=load_test_data_clicked)
                load_test_btn.grid(column=1, row=3)

                test_btn = tk.Button(results_frame, text='Test', command=test_clicked)
                test_btn.grid(column=2, row=3)

                deducability_btn = tk.Button(results_frame, text='Deducability', command=open_deducability)
                deducability_btn.grid(column=3, row=3)

                if name in STATE['trained_models'].residual_plots:
                    draw_res_plot(STATE['trained_models'].residual_plots[name], results_frame)

                save_btn = tk.Button(results_frame, text='Save model', command=save_clicked)
                save_btn.grid(column=2, row=6)

            def update_results():
                if name in self.results_frames:
                    self.results_frames[name].destroy()
                create_results_frame()

            def check_params():
                return self.check_params(name)

            def train_or_train_test_clicked(train_only):
                if not check_params():
                    tk.messagebox.showinfo("Error", INVALID_PARAM_MESSAGE)
                    return
                STATE['untrained_models'].generate_models([name], train_only)
                self.update_trained_lately([name])
                update_results()

            def train_clicked():
                train_or_train_test_clicked(True)

            def train_test_clicked():
                train_or_train_test_clicked(False)

            def btn_focus(event):
                clicked_btn = self.clicked_button
                if clicked_btn is not None:
                    self.clicked_button = None
                    self.button_funcs[name][clicked_btn]()

            train_frame = ttk.Frame(left_frame)
            train_frame.grid(column=0, row=1)
            train_btn = tk.Button(train_frame, text='Train on all data', command=lambda: button_clicked(train_btn))
            train_btn.bind("<FocusIn>", btn_focus)
            self.button_funcs[name][train_btn['text']] = train_clicked
            train_btn.grid(column=0, row=0, padx=(0, 100))
            train_test_btn = tk.Button(train_frame, text='Train & test', command=lambda: button_clicked(train_test_btn))
            train_test_btn.bind("<FocusIn>", btn_focus)
            self.button_funcs[name][train_test_btn['text']] = train_test_clicked
            train_test_btn.grid(column=2, row=0)

            if STATE['is_load']:
                train_btn['state'] = 'disabled'
                train_test_btn['state'] = 'disabled'

            if name in STATE['trained_models'].models:
                create_results_frame()

            def tab_clicked():
                if name in self.trained_lately:
                    update_results()
                    self.trained_lately.remove(name)
                if name in STATE['trained_models'].models:
                    self.test_data_loaded_labels[name]['text'] = '' if self.times_test_data_loaded == 0 \
                        else 'Test data presents (' + str(self.times_test_data_loaded) + ')'

            self.tab_clicked_funcs[parent.index(tab)] = tab_clicked

        for name, model in STATE['untrained_models'].models.items():
            create_tab(tab_control, name, model)

        tab_control.pack(expand=True, fill='both')

        def notebook_clicked(event):
            clicked_tab = tab_control.tk.call(tab_control._w, "identify", "tab", event.x, event.y)
            if clicked_tab in self.tab_clicked_funcs:
                self.tab_clicked_funcs[clicked_tab]()
            else:
                update_general_list()
                load_test_general_lbl['text'] = '' if self.times_test_data_loaded == 0 else \
                    'Test data presents (' + str(self.times_test_data_loaded) + ')'
        tab_control.bind('<Button-1>', notebook_clicked)

        window.mainloop()
