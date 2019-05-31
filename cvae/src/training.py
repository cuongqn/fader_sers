# %%
import os
import sys
import time
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from copy import deepcopy


class AverageBase(object):
    
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
        
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value
    

class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """
    
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count
        
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    
    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value

def acc_logit(y_true, y_pred_logit):
    with torch.no_grad():
        y_pred = y_pred_logit.argmax(dim=1)
        correct = (y_pred == y_true).sum().float()
        accuracy_score = correct/len(y_pred)
    return accuracy_score.cpu().data.numpy()


class Trainer(object):
    def __init__(self, model, criterion, optimizer, metrics=[], scheduler=None, wd_loss=None, device=torch.device('cpu')):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics
        self.train_losses = []
        self.val_losses = []
        self.scheduler = scheduler
        self.weight_decay = wd_loss
    
    def _train_single_epoch(self, train_loader, verbose):
        self.model.train()
        train_loss = MovingAverage()
        if verbose == 2:
            train_loader = tqdm(train_loader)
            
        for x, y_true in train_loader:
            # Move the training data to the GPU
            if isinstance(x, list):
                x = [x_.to(self.device) for x_ in x]
            else:
                x = x.to(self.device)
            y_true = y_true.to(self.device)
            y_pred = self.model(x)
            
            # calculate the loss
            if isinstance(self.criterion, list):
                loss = 0
                for i, criterion in enumerate(self.criterion):
                    loss += criterion(y_pred[:,i], y_true[:,i])
            else:
                loss = self.criterion(y_pred, y_true)
                        
            # clear previous gradient computation and calculate gradients
            self.optimizer.zero_grad()
            loss.backward()
            
            # correct weight decay
            if self.weight_decay:
                for group in self.optimizer.param_groups():
                    for param in group["params"]:
                        param.data = param.data.add(-self.weight_decay * group["lr"], param.data)
            
            # update model weights
            self.optimizer.step()
            train_loss.update(loss.item())
            
            del x, y_true, y_pred, loss
            torch.cuda.empty_cache()
            
        return train_loss.value
    
    @staticmethod
    def _calculate_single_metric(metric, y_true, y_pred):
        if metric == 'acc':
            metric_result = acc_logit(y_true, y_pred)
        else:
            metric_result = metric(y_true.cpu().data.numpy(), y_pred.cpu().data.numpy())
        return metric_result
    
    def _validate_single_epoch(self, val_loader):
        self.model.eval()
        valid_loss = RunningAverage()
        
        # check for specific metrics per output
        if len(self.metrics) == 0:
            metrics = self.metrics
        elif isinstance(self.metrics[0],list):
            metrics = [[np.empty(0) for _ in range(len(m))] for m in self.metrics]
        else:
            metrics = [np.empty(0) for _ in range(len(self.metrics))]
        # keep track of predictions
        y_pred = []

        # We don't need gradients for validation, so wrap in 
        # no_grad to save memory
        with torch.no_grad():

            for x, y_true in val_loader:

                # Move the training data to the GPU
                if isinstance(x, list):
                    x = [x_.to(self.device) for x_ in x]
                else:
                    x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = self.model(x)

                # calculate the loss
                if isinstance(self.criterion, list):
                    loss = 0
                    for i, criterion in enumerate(self.criterion):
                        loss += criterion(y_pred[:,i], y_true[:,i])
                else:
                    loss = self.criterion(y_pred, y_true)
                
                #TODO: make this recursive
                # update metrics
                for i, metric in enumerate(self.metrics):
                    if isinstance(metric,list):
                        for j, m in enumerate(metric):
                            metric_value = self._calculate_single_metric(m, y_true[:,i], y_pred[:,i])
                            metrics[i][j] = np.append(metrics[i][j], metric_value)
                    
                    else:
                        metric_value = self._calculate_single_metric(metric, y_true, y_pred)
                        metrics[i] = np.append(metrics[i], metric_value)
                # update running loss value
                valid_loss.update(loss.item())

        if len(self.metrics) > 0:
            metrics = [np.mean(m) for m in metrics]
        else:
            metrics = valid_loss.value
        
        return valid_loss.value, metrics

    def train_dataloader(self, train_loader,epochs,save_dir=None,callbacks=[], val_loader=None, verbose=0):
        if save_dir is not None:
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            
        self.model = self.model.to(self.device)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs}:", end = "\t")
                
            train_loss_epoch = self._train_single_epoch(train_loader, verbose)
            self.train_losses.append(train_loss_epoch)
            if verbose > 0:
                print(f"loss = {train_loss_epoch}", end="\t")
            
            if val_loader is not None:
                val_loss_epoch, val_metrics_epoch = self._validate_single_epoch(val_loader)
                self.val_losses.append(val_loss_epoch)
                if verbose > 0:
                    print(f"val_loss = {val_loss_epoch}", end="\t")
                    print(f"val_metrics = {val_metrics_epoch}")
                    
                if self.scheduler is not None:
                    self.scheduler.step(val_loss_epoch)
                
                cbs = [cb.step(val_loss_epoch, verbose) for cb in callbacks]
                if "best" in cbs:
                    best_model = deepcopy(self.model)
                
                if "break" in cbs:
                    break
            
            if verbose > 0:
                print(f"Elapsed time: {time.time() - start_time} seconds")
            
            torch.cuda.empty_cache()
            
        if val_loader is None:
            best_model = self.model
            
        return best_model, self.train_losses, self.val_losses
    
    def train_data(self, x, y, dataset, batch_size):
        #TODO: need to implement this
        return NotImplementedError
    

