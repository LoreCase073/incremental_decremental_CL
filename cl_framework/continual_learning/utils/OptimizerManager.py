import torch


class OptimizerManager:
    
    def __init__(self, backbone_lr, head_lr, scheduler_type, approach) -> None:
        
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr 
        self.scheduler_type = scheduler_type
        self.approach = approach
    
    def get_optimizer(self, task_id, model, weight_decay, patience, freeze_bn):
        #freeze batch normalization layers
        if task_id > 0 and freeze_bn == 'yes':
            model.freeze_bn() 

        backbone_params = [p for p in  model.backbone.parameters() if p.requires_grad]
        old_head_params = [p for p in model.heads[:-1].parameters()  if p.requires_grad]
        new_head_params = [p for p in  model.heads[-1].parameters() if p.requires_grad]
        head_params = old_head_params + new_head_params
        
         
       
    
        params = backbone_params + head_params
        if self.backbone_lr == self.head_lr:
            print("Using Adam with a single lr {}".format(self.backbone_lr))
            optimizer =  torch.optim.Adam(params, lr=self.backbone_lr, weight_decay=weight_decay)
        
        else:
            print("Using Adam with two lr. Backbone: {}, Head: {}".format(self.backbone_lr, self.head_lr))
    
            optimizer = torch.optim.Adam([{'params': head_params, 'lr':self.head_lr},
                                            {'params': backbone_params}
                                            ],lr=self.backbone_lr, 
                                            weight_decay=weight_decay)
 
        if self.scheduler_type == "multi_step":
            print("Scheduling lr after 20 and 40 epochs")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[20, 40],
                                                            gamma=0.1, verbose=True
                                                               )
        elif self.scheduler_type == "reduce_plateau":
            print("Scheduling on plateau with patience {}".format(patience))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)
        else:
            print("Fixed Learning Rate")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[1000,2000],
                                                            gamma=0.1, verbose=True
                                                              )
            
        return optimizer, scheduler