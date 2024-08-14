import abc
import os
from sched import scheduler
from tabnanny import verbose
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from continual_learning.utils.OptimizerManager import OptimizerManager

class IncrementalApproach(metaclass=abc.ABCMeta):
    
   def __init__(self, args, device, out_path, class_per_task, task_dict):
      self.device = device
      self.out_path = out_path
      self.class_per_task = class_per_task
      self.task_dict = task_dict 
      
      # Model args
      self.approach = args.approach
      self.backbone = args.backbone
      # Optimizer args
      self.lr_first_task = args.lr_first_task
      self.lr_first_task_head = args.lr_first_task_head
      self.scheduler_type  = args.scheduler_type
      self.plateau_check = args.plateau_check
      self.patience = args.patience
      self.backbone_lr = args.backbone_lr
      self.head_lr = args.head_lr
      self.freeze_bn = args.freeze_bn

      self.batch_size = args.batch_size 
      self.total_epochs = args.epochs 
      self.logger = SummaryWriter(os.path.join(out_path, "tensorboard"))
      self.milestones_first_task = None 
      self.dataset = args.dataset
      self.weight_decay = args.weight_decay
      if self.dataset=="cifar100":
         self.image_size = 32
      elif self.dataset=="tiny-imagenet":
         self.image_size = 64
      elif self.dataset=="imagenet-subset":
         self.image_size = 224
         print("Fixing the backbone lr to 1e-5 and the head lr to 1e-4")
         self.backbone_lr = 1e-5
         self.head_lr == 1e-4
         
      # SELF-ROTATION classifier
      self.auxiliary_classifier = None
      self.optimizer_manager = OptimizerManager(self.backbone_lr, self.head_lr, 
                                                self.scheduler_type, self.approach)
      
      if self.dataset == "imagenet-subset":
         self.milestones_first_task = [80, 120, 150]
      elif self.scheduler_type == "multi_step":
         self.milestones_first_task = [20, 40]
      else:
         self.milestones_first_task = [1000, 2000]

      
      

   def pre_train(self, task_id, *args):
 
      if task_id == 0:
         params_to_optimize = [p for p in self.model.backbone.parameters() if p.requires_grad] + [p for p in self.model.heads.parameters() if p.requires_grad]
         if self.auxiliary_classifier is not None:
            params_to_optimize += [p for p in self.auxiliary_classifier.parameters() if p.requires_grad]
            
         if self.dataset=="imagenet-subset": 
            self.lr_first_task = 0.1 
       
            gamma = 0.1 
            custom_weight_decay = 5e-4 
            custom_momentum = 0.9  
            print("Using SGD Optimizer With PASS setting")
            self.optimizer = torch.optim.SGD(params_to_optimize, lr=self.lr_first_task, momentum=custom_momentum,
                                             weight_decay=custom_weight_decay)
            
            self.reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                         milestones=self.milestones_first_task,
                                                         gamma=gamma, verbose=True
                                                         )
         else:
    
            # self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr_first_task, weight_decay=self.weight_decay)
            backbone_params = [p for p in  self.model.backbone.parameters() if p.requires_grad]
            old_head_params = [p for p in self.model.heads[:-1].parameters()  if p.requires_grad]
            new_head_params = [p for p in  self.model.heads[-1].parameters() if p.requires_grad]
            head_params = old_head_params + new_head_params
            
               
            
         
            params = backbone_params + head_params
            if self.lr_first_task == self.lr_first_task_head:
                  print("Using Adam with a single lr {}".format(self.lr_first_task))
                  self.optimizer =  torch.optim.Adam(params, lr=self.lr_first_task, weight_decay=self.weight_decay)
            
            else:
                  print("Using Adam with two lr. Backbone: {}, Head: {}".format(self.lr_first_task, self.lr_first_task_head))
         
                  self.optimizer = torch.optim.Adam([{'params': head_params, 'lr':self.lr_first_task_head},
                                                {'params': backbone_params}
                                                ],lr=self.lr_first_task, 
                                                weight_decay=self.weight_decay)
                  
                  
            if self.scheduler_type == 'multi_step':
               self.reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                         milestones=self.milestones_first_task,
                                                         gamma=0.1, verbose=True
                                                            )
            elif self.scheduler_type == 'reduce_plateau':
               #with this we decide if looking at mAP or Validation Loss
               if self.plateau_check == "map":
                  self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.patience, verbose=True)
               elif self.plateau_check == "class_loss":
                  self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, verbose=True)
            else:
               self.reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                         milestones=self.milestones_first_task,
                                                         gamma=0.1, verbose=True
                                                            )
      else:            

            self.optimizer, self.reduce_lr_on_plateau = self.optimizer_manager.get_optimizer(task_id, self.model, self.weight_decay, self.patience, self.freeze_bn)
    

   
   @abc.abstractmethod
   def train(self, *args):
      pass 

   @abc.abstractmethod
   def post_train(self, *args):
      pass 

   @abc.abstractmethod
   def eval(self, *args):
      pass

   @abc.abstractmethod
   def log(self, *args):
      pass
   

   def print_running_approach(self):
      print("#"*40 + " --> RUNNING APPROACH")
      print("- approach: {}".format(self.approach))
      print("- backbone: {}".format(self.backbone))
      print("- batch size : {}".format(self.batch_size))
      print("- lr first task: {} with {} scheduler".format(self.lr_first_task, self.scheduler_type))
      print("- incremental phases: backbone lr : {}".format(self.backbone_lr))
      print("- incremental phases: head lr : {}".format(self.head_lr))
      print("- incremental phases: scheduler type  {}".format(self.scheduler_type))
      print()

   
 
   def tag_probabilities(self, outputs):
        tag_output = []
        for key in outputs.keys():
            tag_output.append(outputs[key])
        tag_output = torch.cat(tag_output, dim=1)
        probabilities = torch.nn.Softmax(dim=1)(tag_output)
        return probabilities
   

   def taw_probabilities(self, outputs, head_id):
      probabilities = torch.nn.Softmax(dim=1)(outputs[head_id])
      return probabilities