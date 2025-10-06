from base_wrapper import BaseWrapper

class TorchWrapper(BaseWrapper):
    """Class for wrapping PyTorch models.
        Supported functionalities :
        -> .predict -> returns the logit/output returned by the model as a numpy object, moved to the cpu.
        -> .save -> saves the model's state dictionary to specified path.
        -> .load -> loads model weights from specified path for wrapped model.
        -> .get_params -> returns model parameters as a Dict object).
        -> .set_params -> sets model parameters as per input parameters of type **params

        Dependencies = ["torch"]
    """
    def __init__(self, model, optimizer = None, loss_fn = None):
        import torch
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y, epochs=1,**kwargs):
        import torch
        """Simple training loop"""
        self.model.train()
        X = torch.tensor(X,dtype=torch.float32).to(self.device)
        y = torch.tensor(y).to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.loss_fn(outputs,y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X, **kwargs):
        import torch
        self.model.eval()
        with torch.no_grad():
            # in predict, you always convert input to torch.tensor, 
            # which is fine for NumPy arrays but will break if someone passes a 
            # torch.Tensor directly → should handle both. 
            # X = torch.tensor(X, dtype = torch.float32).to(self.device)
            # return self.model(X).cpu().numpy()
            if not isinstance(X,torch.Tensor):
                X = torch.tensor(X,dtype=torch.float32)
            X = X.to(self.device)
            return self.model(X).cpu().numpy()
    
    def predict_proba(self, X, **kwargs):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            logits = self.model(X)

            # Handle binary vs multi-class
            if logits.ndim == 2 and logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()


    def save(self, path: str):
        import torch
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        import torch
        self.model.load_state_dict(torch.load(path, map_location = self.device))

    def get_params(self):
        return {"state_dict":  self.model.state_dict()}
    
    def set_params(self, **params):
        self.model.load_state_dict(params["state_dict"])
    