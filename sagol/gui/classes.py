from sagol.gui.globals import STATE
from sagol.run_models import AVAILABLE_MODELS


class UntrainedModel:
    def __init__(self, name):
        self.name = name
        if name in STATE['trained_models'].models:
            self.parameters = {param: (AVAILABLE_MODELS[name][param], v) for
                               param, v in STATE['trained_models'].parameters[name].items()}
        else:
            self.parameters = {param: (t, '') for param, t in AVAILABLE_MODELS[name].items()}


class UntrainedModels:
    def __init__(self):
        self.models = {name: UntrainedModel(name) for name in AVAILABLE_MODELS}
