from sagol.gui.globals import STATE
from sagol.models.utils import AVAILABLE_MODELS, AVAILABLE_3D_MODELS
from sagol.run_models import generate_models


class UntrainedModel:
    def __init__(self, name):
        self.name = name
        self.is_3d = name in AVAILABLE_3D_MODELS
        if name in STATE['trained_models'].models:
            self.parameters = {param: [AVAILABLE_MODELS[name][param], v] for
                               param, v in STATE['trained_models'].parameters[name].items()}
        else:
            self.parameters = {param: '' for param in AVAILABLE_MODELS[name]}


class UntrainedModels:
    def __init__(self):
        self.models = {name: UntrainedModel(name) for name in AVAILABLE_MODELS}

    def generate_models(self, model_names, train_only=False):
        data = STATE['experiment_data_after_split']
        data_3d = STATE['experiment_data_after_split_3d']
        trained_models = STATE['trained_models']
        reverse_contrast_mapping = STATE['reverse_contrast_mapping']
        ylabels = trained_models.ylabels
        roi_paths = trained_models.roi_paths
        model_params = {}
        for name in model_names:
            model_params[name] = {}
            for param, val in self.models[name].parameters.items():
                if not val == '':
                    model_params[name][param] = val
        trained_models, _, _ = generate_models(model_names=model_names, experiment_data_after_split=data,
                                               reverse_contrast_mapping=reverse_contrast_mapping,
                                               experiment_data_after_split_3d=data_3d, ylabels=ylabels,
                                               roi_paths=roi_paths, model_params=model_params, train_only=train_only)
        STATE['trained_models'].set_models(trained_models)
