import itertools 
import sys 
from torch import autograd
from torch import nn 
from tqdm import tqdm
import os 
from torch.autograd import grad
import numpy as np 
import einops
import torch 


 


class EmpiricalFIM:


    def __init__(self,  device, out_path):
        self.empirical_feat_mat = None
        self.device = device
 
        self.out_path = out_path
    
 
    def create(self, model):
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in model.backbone.named_parameters()
                        if p.requires_grad}

    
    def get(self):
        return self.fisher


    def compute(self, model, trn_loader, task_id):

        self.compute_fisher(model, trn_loader)
  
        
        
        
        
        

    def compute_fisher(self, model, trn_loader):
 
        # Compute fisher information for specified number of samples -- rounded to the batch size
        
        n_samples_batches = len(trn_loader.dataset) // trn_loader.batch_size
        # Do forward and backward pass to compute the fisher information
        model.eval() 
        # ensure that gradients are zero
        model.zero_grad()   

        self.create(model)
        print("Computing Fisher Information")
    
        for images, targets in itertools.islice(trn_loader, n_samples_batches):

            gap_out = model.backbone(images.to(self.device))
            
            outputs = torch.cat([h(gap_out)for h in model.heads], dim=1)
            preds =  outputs.argmax(1).flatten()
            
            
            loss = torch.nn.functional.cross_entropy(outputs, preds)
            
            model.zero_grad()
            
            loss.backward()
            for n, p in model.backbone.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.pow(2) * len(targets)
 
    
            
        n_samples = n_samples_batches * trn_loader.batch_size
        self.fisher = {n: (p / n_samples) for n, p in self.fisher.items()}