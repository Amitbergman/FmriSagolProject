import tkinter as tk
from tkinter import ttk
from sagol.gui.globals import STATE
from sagol.load_data import create_subject_experiment_data
from sagol.run_models import generate_experiment_data_after_split
from sagol.models.utils import AVAILABLE_MODELS
from sagol.evaluate_models import Models
from sagol.rois import get_available_rois
from sagol.gui.classes import UntrainedModels


# א to be removed
def create_data_and_models():
    STATE['experiment_data'] = create_subject_experiment_data(
        ["C:/Users/Liorzlo/FmriSagolProject/data/questionnaires_byTasks_new.xlsx"],
        ["C:/Users/Liorzlo/FmriSagolProject/data/Doors_2ndLev"])
    STATE['trained_models'] = Models(ylabels=['FPES', 'SPIN'], roi_paths=get_available_rois(), shape=(85,101,65))
    STATE['tasks_and_contrasts'] = None
    STATE['weights'] = [0.6, 0.4]
    STATE['is_load'] = False


def prepare_data():
    trained_models = STATE['trained_models']
    STATE['experiment_data_after_split'], STATE['experiment_data_after_split_3d'] = generate_experiment_data_after_split(
        experiment_data=STATE['experiment_data'], roi_paths=trained_models.roi_paths,
        tasks_and_contrasts=STATE['tasks_and_contrasts'], ylabels=trained_models.ylabels, weights=STATE['weights'])


class ModelsWindow:
    def __init__(self):
        create_data_and_models()
        STATE['untrained_models'] = UntrainedModels()
        if not STATE['is_load']:
            prepare_data()
        self.results_frames = {}
        self.params_valid = {model_name: {} for model_name in AVAILABLE_MODELS}
        self.trained_lately = set()
        self.tab_clicked_funcs = {}
        self.button_funcs = {model_name: {} for model_name in AVAILABLE_MODELS}
        self.clicked_button = None

    def update_trained_lately(self, model_names):
        for name in model_names:
            if name not in self.trained_lately:
                self.trained_lately.add(name)

    # When this function is called, STATE['trained_models'] should be populated by an object of the class
    # 'sagol.evaluate_models.Models'. The attributes 'ylabels', 'roi_paths' and 'shape' must be populated,
    # even if no model was trained yet!
    # STATE['experiment_data'] should be populated by an object of the class 'sagol.load_data.ExperimentData',
    # which represent the data that the user chose in the former window.
    # STATE['tasks_and_contrasts'] should be populated correctly
    # STATE['weights'] should be populated should be populated correctly
    # Lastly, STATE['is_load'] should be True iif the user chose to load existing models in the former window.
    def open_models_window(self):
        window = tk.Tk()
        window.geometry('800x400')
        window.title("Models")

        info_frame = ttk.Frame(window)
        info_frame.pack(expand=True)
        y_labels_lbl = tk.Label(info_frame, text='y labels: ' + ', '.join(STATE['trained_models'].ylabels))
        y_labels_lbl.pack(expand=True)
        rois_lbl = tk.Label(info_frame, text='ROIs: ' + ', '.join(STATE['trained_models'].roi_paths))
        rois_lbl.pack(expand=True)

        tab_control = ttk.Notebook(window)

        tab_general = ttk.Frame(tab_control)
        tab_control.add(tab_general, text='General')

        def create_tab(parent, name, untrained_model):
            tab = ttk.Frame(parent)
            parent.add(tab, text=name)
            params_frame = ttk.Frame(tab)
            params_frame.grid(column=0, row=0, padx=(0, 100))

            def populate_params_frame(parent, params):
                choose_params_lbl = tk.Label(parent, text='Choose parameters:')
                choose_params_lbl.grid(column=0, row=0)
                for i, (p, t_v) in enumerate(params.items()):

                    def create_param_comp(parent, i, p, t, v):
                        self.params_valid[name][p] = True
                        param_lbl = tk.Label(parent, text=p + ': ')
                        param_lbl.grid(column=0, row=i + 1)
                        param_entry = tk.Entry(parent, width=10, validate='focus')

                        def update_param():
                            val = param_entry.get()
                            if val == '':
                                STATE['untrained_models'].models[name].parameters[p][1] = ''
                                return_val = True
                            elif t == str:
                                try:
                                    float(val)
                                    return_val = False
                                except ValueError:
                                    STATE['untrained_models'].models[name].parameters[p][1] = val
                                    return_val = True
                            else:
                                try:
                                    converted_val = t(val)
                                    STATE['untrained_models'].models[name].parameters[p][1] = converted_val
                                    return_val = True
                                except ValueError:
                                    return_val = False
                            self.params_valid[name][p] = return_val
                            param_entry.config(bg = 'white' if return_val else 'red')
                            if self.clicked_button is not None:
                                self.button_funcs[name][self.clicked_button]()
                                self.clicked_button = None
                            return return_val

                        param_entry.config(validatecommand=update_param)
                        param_entry.insert(tk.END, v)
                        param_entry.grid(column=1, row=i + 1)

                    create_param_comp(parent, i, p, t_v[0], t_v[1])

            populate_params_frame(params_frame, untrained_model.parameters)

            def button_clicked(button):
                self.clicked_button = button['text']
                button.focus_set()

            # Train & test frame
            def create_results_frame():
                results_frame = ttk.Frame(tab)
                self.results_frames[name] = results_frame
                results_frame.grid(column=1, row=0)

                trained_model_lbl = tk.Label(results_frame, text='Trained model:')
                trained_model_lbl.grid(column=0, row=0)

                params_chosen_frame = ttk.Frame(results_frame)
                params_chosen_frame.grid(column=1, row=1)
                for i, (p, v) in enumerate(STATE['trained_models'].parameters[name].items()):
                    param_lbl = tk.Label(params_chosen_frame, text=p + ': ' + str(v))
                    param_lbl.grid(column=0, row=i)

                train_score_lbl = tk.Label(results_frame,
                                           text='Train score: ' + str(STATE['trained_models'].get_train_score(name)))
                train_score_lbl.grid(column=1, row=2)
                test_score_lbl = tk.Label(results_frame,
                                          text='Test score: ' + str(STATE['trained_models'].get_test_score(name)))
                test_score_lbl.grid(column=2, row=2)

                def test_clicked():
                    return

                def open_deducability():
                    return

                test_btn = tk.Button(results_frame, text='Test', command=test_clicked)
                test_btn.grid(column=1, row=3)
                deducability_btn = tk.Button(results_frame, text='Deducability', command=open_deducability)
                deducability_btn.grid(column=2, row=3)

            def update_results():
                if name in self.results_frames:
                    self.results_frames[name].destroy()
                create_results_frame()

            def check_params():
                for val in self.params_valid[name].values():
                    if not val:
                        return False
                return True

            # Buttons
            def train_clicked():
                if not check_params():
                    tk.messagebox.showinfo("Error", "At least one of the parameters is illegal")
                    return
                STATE['untrained_models'].generate_models([name], True)
                self.update_trained_lately([name])
                update_results()

            def train_test_clicked():
                if not check_params():
                    tk.messagebox.showinfo("Error", "At least one of the parameters is illegal")
                    return
                STATE['untrained_models'].generate_models([name], False)
                self.update_trained_lately([name])
                update_results()

            train_frame = ttk.Frame(tab)
            train_frame.grid(column=0, row=1)
            train_btn = tk.Button(train_frame, text='Train', command=lambda: button_clicked(train_btn))
            self.button_funcs[name][train_btn['text']] = train_clicked
            train_btn.grid(column=0, row=0, padx=(0, 100))
            train_test_btn = tk.Button(train_frame, text='Train & test', command=lambda: button_clicked(train_test_btn))
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

            self.tab_clicked_funcs[parent.index(tab)] = tab_clicked

        for name, model in STATE['untrained_models'].models.items():
            create_tab(tab_control, name, model)

        tab_control.pack(expand=True, fill='both')

        def notebook_clicked(event):
            clicked_tab = tab_control.tk.call(tab_control._w, "identify", "tab", event.x, event.y)
            if clicked_tab in self.tab_clicked_funcs:
                self.tab_clicked_funcs[clicked_tab]()
        tab_control.bind('<Button-1>', notebook_clicked)

        window.mainloop()
