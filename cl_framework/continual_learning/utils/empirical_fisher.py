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
from torch.utils.data import DataLoader


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


    def compute(self, model, trn_loader, task_id, criterion_type):

        self.compute_fisher(model, trn_loader, criterion_type)
  
        
        
        
        
        

    def compute_fisher(self, model, trn_loader, criterion_type):
        #create new Dataloader to do sequential sampling
        tmp_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False, num_workers=trn_loader.num_workers)

        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = len(tmp_loader.dataset) // tmp_loader.batch_size
 
        
        
        
        # Do forward and backward pass to compute the fisher information
        model.eval() 
        # ensure that gradients are zero
        model.zero_grad()   

        self.create(model)
        print("Computing Fisher Information")
    
        for images, targets, binarized_targets, _, _ in itertools.islice(trn_loader, n_samples_batches):
            _, gap_out = model(images.to(self.device))
            out = model.heads[0](gap_out)
            #preds =  out.argmax(1).flatten()
            indices_max = torch.argmax(out, dim=1, keepdim=True)
            preds_tens = torch.zeros_like(out)
            preds_tens.scatter_(1, indices_max, 1)

            
            if criterion_type == 'multilabel':
                #loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(out), targets)
                loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(out), preds_tens)
            else:
                loss = torch.nn.functional.cross_entropy(out, targets)
            
            model.zero_grad()
            
            loss.backward()
            for n, p in model.backbone.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.pow(2) * len(targets)
 
    
            
        n_samples = n_samples_batches * trn_loader.batch_size
        self.fisher = {n: (p / n_samples) for n, p in self.fisher.items()}