from base_wrapper import BaseWrapper

class TorchWrapper(BaseWrapper):
    """Class for wrapping PyTorch models.\n
        Supported functionalities :\n
        -> .predict -> returns the logit/output returned by the model as a numpy object, moved to the cpu.\n
        -> .save -> saves the model's state dictionary to specified path.\n
        -> .load -> loads model weights from specified path for wrapped model.\n
        -> .get_params -> returns model parameters as a Dict object).\n
        -> .set_params -> sets model parameters as per input parameters of type **params\n

        Dependencies = ["torch"]
    """
    def __init__(self, model, optimizer = None, loss_fn = None):
        import torch
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, X, **kwargs):
        import torch
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype = torch.float32).to(self.device)
            return self.model(X).cpu().numpy()
    
    def save(self, path: str):
        import torch
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        import torch
        self.model.load_state_dict(torch.load(path, map_location = self.device))

    def get_params(self):
        return {"state_dict":  self.model.state_dict}
    
    def set_params(self, **params):
        self.model.load_state_dict(params["state_dict"])