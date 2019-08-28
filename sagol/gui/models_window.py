import tkinter as tk
from tkinter import ttk
from sagol.gui.globals import STATE
from sagol.evaluate_models import Models
from sagol.gui.classes import UntrainedModels
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import BaggingRegressor


models = {'svr': SVR(C=160, gamma=0.00095), 'bagging_regressor': BaggingRegressor()}
train_scores = {'svr': 0.9, 'bagging_regressor': 0.95}
test_scores = {'svr': 0.5, 'bagging_regressor': 0.6}
graphs = {}
params = {'svr':{'kernel': 'rbf', 'C': 160, 'gamma': 0.00095}, 'bagging_regressor':{'n_estimators': 10}}


def populate_STATE():
    STATE['trained_models'] = Models(['FPES', 'BSNE'], ['this/that/roi1', 'this/that/roi2'], (85,101,65), models=models,
                                     test_scores=test_scores, train_scores=train_scores, residual_plots=graphs, parameters=params)
    STATE['untrained_models'] = UntrainedModels()
    STATE['experiment_data'] = None
    STATE['is_load'] = False

# When this function is called, STATE['trained_models'] should be populated by an object of the class
# 'sagol.evaluate_models.Models'. The attributes 'ylabels', 'roi_paths' and 'shape' must be populated,
# even if no model was trained yet!
# In addition, STATE['experiment_data'] should be populated by an object of the class 'sagol.load_data.ExperimentData',
# which represent the data that the user chose in the former window.
# Lastly, STATE['is_load'] should be True iif the user chose to load existing models in the former window.
def open_models_window():
    # To be removed
    populate_STATE()

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
            for i, (p, (t, v)) in enumerate(params.items()):
                param_lbl = tk.Label(parent, text=p + ': ')
                param_lbl.grid(column=0, row=i + 1)
                param_entry = tk.Entry(parent, width=10)
                param_entry.insert(tk.END, v)
                param_entry.grid(column=1, row=i + 1)

        populate_params_frame(params_frame, untrained_model.parameters)

        # Train & test frame
        def create_trained_model_frame():
            results_frame = ttk.Frame(tab)
            results_frame.grid(column=1, row=0)

            trained_model_lbl = tk.Label(results_frame, text='Trained model:')
            trained_model_lbl.grid(column=0, row=0)

            params_chosen_frame = ttk.Frame(results_frame)
            params_chosen_frame.grid(column=1, row=1)
            for i, (p, v) in enumerate(STATE['trained_models'].parameters[name].items()):
                param_lbl = tk.Label(params_chosen_frame, text=p + ': ' + str(v))
                param_lbl.grid(column=0, row=i)

            train_score_lbl = tk.Label(results_frame, text='Train score: ' + str(STATE['trained_models'].get_train_score(name)))
            train_score_lbl.grid(column=1, row=2)
            test_score_lbl = tk.Label(results_frame, text='Test score: ' + str(STATE['trained_models'].get_test_score(name)))
            test_score_lbl.grid(column=3, row=2)

            def open_deducability():
                return

            deducability_btn = tk.Button(results_frame, text='Deducability', command=open_deducability)
            deducability_btn.grid(column=2, row=3)

        # Buttons
        def train_clicked():
            train_btn['text'] = 'Training...'
            create_trained_model_frame()
            train_btn['text'] = 'Train'

        def train_test_clicked():
            train_test_btn['text'] = 'Training & testing...'
            create_trained_model_frame()
            train_test_btn['text'] = 'Train & test'

        train_frame = ttk.Frame(tab)
        train_frame.grid(column=0, row=1)
        train_btn = tk.Button(train_frame, text='Train', command=train_clicked)
        train_btn.grid(column=0, row=0, padx=(0, 100))
        train_test_btn = tk.Button(train_frame, text='Train & test', command=train_test_clicked)
        train_test_btn.grid(column=2, row=0)

        if STATE['is_load']:
            train_btn['state'] = 'disabled'
            train_test_btn['state'] = 'disabled'

        if name in STATE['trained_models'].models:
            create_trained_model_frame()

    for name, model in STATE['untrained_models'].models.items():
        create_tab(tab_control, name, model)

    tab_control.pack(expand=True, fill='both')

    window.mainloop()
