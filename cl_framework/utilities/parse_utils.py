import argparse
    
def get_args():
    parser = argparse.ArgumentParser()

    """
    Structural hyperparams 
    """
    parser.add_argument("--approach", type=str,default="incdec", choices=["incdec", 'incdec_lwf', 'incdec_fd', 'incdec_ewc'], help="Type of machine learning approach to be followed.")
    parser.add_argument("--pipeline", type=str,default="baseline", choices=["baseline","decremental","incremental_decremental","joint_incremental"], help="Type of pipeline to be follower in the incdec case.") 
    parser.add_argument("--n_accumulation", type=int, default=0, help="To be used in case you want to do gradient accumulation.")
    parser.add_argument("--outpath", "-op",default="./", type=str, help="Output directory where to save results.") 
    parser.add_argument("--seed", type=int, default=0, help="Seed to be initialized to.")
    parser.add_argument("--nw", type=int, default=4, help="num workers for data loader")
    parser.add_argument("--freeze_bn", type=str, default="no", choices=["yes", "no"], help="If training need to be done with the batch normalization in the backbone frozen. Choices: ['yes','no']")
    parser.add_argument("--freeze_backbone", type=str, default="no", choices=["yes", "no"], help="If training need to be done with the backbone frozen. Choices: ['yes','no']")
    parser.add_argument("--early_stopping_val", type=int, default=1000, help="If need to do early stopping, without any scheduling.")
    parser.add_argument("--weight_decay", default=5e-4, type=float)

    """
    EWC Hyperparams
    """
    parser.add_argument("--ewc_lambda", default=500.0, type=float, help="")
    """
    LWF hyperparams 
    """
    parser.add_argument("--lwf_lamb", default=1.0, type=float, help="Coefficient to weight the lwf regularization term.")
    parser.add_argument("--lwf_T", default=1.0, type=float, help="")
    """
    FD hyperparams 
    """
    parser.add_argument("--fd_lamb", default=0.1, type=float, help="")

 
    """
    Training hyperparams 
    """

    parser.add_argument("--stop_first_task", type=str, default="no", choices=["yes", "no"], help="Flag to stop at first task, needed in debugging.") 
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr_first_task", type=float, default=1e-4, help="To control lr for the backbone on the first task.")
    parser.add_argument("--lr_first_task_head", type=float, default=1e-4, help="To control lr for the head on the first task.")
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="fixd", choices=["fixd", "multi_step","reduce_plateau"])
    parser.add_argument("--plateau_check", type=str, default="map", choices=["map", "class_loss"], help="Select the metric to be checked by reduce_plateau scheduler. Mean Average precision 'map' or classification loss 'class_loss'.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for the reduce_plateau scheduler.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--criterion_type", type=str, default="multilabel", choices=["multiclass", "multilabel"], help="Select the type of loss to be used, for multiclass is cross_entropy, for multilabel BCE.")
    
    "Dataset Settings"
    parser.add_argument("--dataset", type=str, default="kinetics", choices=["kinetics"], help="dataset to use") 
    parser.add_argument("--data_path",type=str, default="./Kinetics",help="path where dataset is saved")
    parser.add_argument("--subcategories_csv_path",type=str, default="./Kinetics/Info/subcategories_to_remove.csv",help="path where the csv with the specification of the subcategories to be removed/substituted is stored, for the pipeline decremental/incremental_decremental.")
    parser.add_argument("--subcategories_randomize", type=str, default="yes", choices=["yes", "no"], help="Use it if we want to work with subcategories (subcategories), and in the decremental or incremental/decremental pipeline you want to randomize the order in which are removed/substituted") 
    parser.add_argument("--n_task", type=int, default=6, help="number of tasks, including the initial one")
    parser.add_argument("--sampler", type=str, default="imbalanced", choices=["imbalanced","balanced"], help="Select the type of sampler to ber used by dataloader. imbalance sampler is for class imbalance cases. balanced is the standard one.")
        
    """
    Network Params
    """
    parser.add_argument("--backbone", type=str, default="movinetA0", choices=["movinetA0","movinetA1","movinetA2"])
    parser.add_argument("--pretrained_path", type=str, default="None", help="specify model path if start from a pre-trained model for also task 0")

    
    args = parser.parse_args()

    non_default_args = {
            opt.dest: getattr(args, opt.dest)
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    default_args = {
            opt.dest: opt.default
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest)
    }

    all_args = vars(args)    
    
    return args, all_args, non_default_args, default_args