from copy import deepcopy
from torch import nn
from utils import variable

class EWC():
    
    def __init__(self, eps_model, diffusion, data_loader, importance=1000):
        self.eps_model = eps_model
        self.diffusion = diffusion
        self.importance = importance 

        self.params = {n: p for n, p in self.eps_model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher(data_loader)
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
        

    def _diag_fisher(self, data_loader):
        len_dataloader = int(len(data_loader))
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.eps_model.eval()
        
        for (data, label) in data_loader:
            data, label = data.to('cuda'), label.to('cuda')
            self.eps_model.zero_grad()
            # loss = self.diffusion.loss(data, label)
            loss = self.diffusion.loss(data)
            loss.backward()

            for n, p in self.eps_model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len_dataloader
       
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices
    
    def penalty(self, eps_model: nn.Module):
        loss = 0
        for n, p in eps_model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum() 
        return loss
