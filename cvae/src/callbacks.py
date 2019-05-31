# %%
import torch
import operator

class EarlyStopping(object):
    def __init__(self, patience, mode='min'):
        assert mode in ['min', 'max'], "mode must be one of ['min', 'max']"
        self.max_patience = patience
        self.current_patience = 0
        self.best_loss = None
        self.op = self._set_mode(mode)
    
    @staticmethod
    def _set_mode(mode):
        if mode == "min": op = operator.gt
        elif mode == "max": op = operator.lt
        return op
        
    def step(self, loss, verbose):
        if self.best_loss is None:
            self.best_loss = loss
        if self.op(loss,self.best_loss):
            self.current_patience += 1
            if self.current_patience == self.max_patience:
                print("Early stopping yooo!")
                return "break"
            else:
                return "pass"
        else:
            if verbose > 0:
                print(f"Early stopping metric improves from {self.best_loss} to {loss}")
            self.current_patience = 0
            self.best_loss = loss
            return "best"
             

#%%


#%%
