from base_wrapper import BaseWrapper

class TFWrapper(BaseWrapper):
    """Class for wrapping tensorflow models.
        Supported Functionalities: 
        -> predict -> returns the logit/output returned by the model as a numpy object, moved to the cpu.
        -> .save -> saves the model's state dictionary to specified path.
        -> .load -> loads model weights from specified path for wrapped model.
        -> .get_params -> returns model parameters as a Dict object).
        -> .set_params -> sets model parameters as per input parameters of type **params

        Dependecies = ["tensorflow", "keras"]
    """
    def __init__(self, model, ):
        self.model = model

    def predict(self, X, **kwargs):
        self.model.predict(X, *kwargs)
    
    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        from tensorflow import keras
        self.model = keras.models.load_model(path)

    def get_params(self):
        return {layer.name: layer.get_config() for layer in self.model.layers}
    
    def set_params(self, **params):
        for layer in self.model.layers:
            if layer.name in params:
                layer.set_weights(params[layer.name])