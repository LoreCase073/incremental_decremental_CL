from utilities.generic_utils import experiment_folder, result_folder, \
                            seed_everything, rollback_model, rollback_model_movinet, \
                            store_model, store_valid_loader, \
                            get_task_dict_incdec, AverageMeter
from utilities.parse_utils import get_args
from utilities.matrix_logger import IncDecLogger
from torch.utils.data.dataloader import DataLoader

# approaches 
from continual_learning.DataIncrementalDecrementalMethod import DataIncrementalDecrementalMethod
from continual_learning.DIDM_lwf import DIDM_lwf
from continual_learning.DIDM_fd import DIDM_fd
from continual_learning.DIDM_ewc import DIDM_ewc
 


# dataset 
from dataset.data_inc_dec_dataset import DataIncDecBaselineDataset, DataIncrementalDecrementalPipelineDataset, JointIncrementalBaselineDataset
from dataset.dataset_utils import get_dataset 
import sys 

from utilities.summary_logger import SummaryLogger
import os 
import math
import time


def get_training_validation_subset_for_tasks(approach, pipeline, train_set,
                                                    n_task, initial_split, 
                                                    total_classes,
                                                    validation_set,
                                                    subcategories_dict = None):
    if pipeline == 'baseline':
        cl_train_val = DataIncDecBaselineDataset(train_set,
                                                n_task, initial_split, 
                                                total_classes,
                                                train=True, validation=validation_set,)
    elif pipeline == 'decremental' or pipeline == 'incremental_decremental':
        cl_train_val = DataIncrementalDecrementalPipelineDataset(train_set, subcategories_dict, 
                n_task, initial_split,
                total_classes, train=True, validation=validation_set)
    elif pipeline == 'joint_incremental':
        cl_train_val = JointIncrementalBaselineDataset(train_set,
                                                n_task, initial_split, 
                                                total_classes,
                                                train=True, validation=validation_set)
        
    return cl_train_val


def get_test_subset_for_tasks(approach, pipeline, test_set,  
                                                    n_task, initial_split, 
                                                    total_classes,
                                                    subcategories_dict = None):
    if pipeline == 'baseline':
        cl_test = DataIncDecBaselineDataset(test_set,
                                                n_task, initial_split, 
                                                total_classes,
                                                train=False, validation=None)
    elif pipeline == 'decremental' or pipeline == 'incremental_decremental':
        cl_test = DataIncrementalDecrementalPipelineDataset(test_set, subcategories_dict, 
                n_task, initial_split,
                total_classes, train=False, validation=None)
    elif pipeline == 'joint_incremental':
        cl_test = JointIncrementalBaselineDataset(test_set,
                                                n_task, initial_split, 
                                                total_classes,
                                                train=False, validation=None)
        
    return cl_test

def get_data_loaders(validation_set, sampler, batch_size, nw, train_dataset_list, val_dataset_list, test_dataset_list):
    train_loaders = []
    valid_loaders = []
    test_loaders = [DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=nw) for test in test_dataset_list]

    if validation_set != None:
        if sampler == 'imbalanced':
            for train in train_dataset_list:
                sampler = cl_train_val.get_weighted_random_sampler(train.indices)
                train_loaders.append(DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=nw, sampler=sampler))
        else:
            train_loaders = [DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=nw) for train in train_dataset_list]
        print("Creating Validation Set")
        
        valid_loaders = [DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=nw) for valid in val_dataset_list]
    
    else:
        print("Not using Validation")
        if sampler == 'imbalanced':
            for train in train_dataset_list:
                sampler = cl_train_val.get_weighted_random_sampler(train.indices)
                train_loaders.append(DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=nw, sampler=sampler))
        else:
            train_loaders = [DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=nw) for train in train_dataset_list]
        valid_loaders = test_loaders

    return train_loaders, valid_loaders, test_loaders
 

if __name__ == "__main__":
    
    # args
    args, all_args, non_default_args, all_default_args = get_args()
    
    print(args.outpath)
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
        
    # device
    device = "cuda:" + str(args.device)
    
     
    # if True create folder exp_folder, else create folder with the name of the approach
    dev_mode = False

    # initial_split specification
    initial_split = 2

    # if we want to stop at the first task to benchmark the first training
    stop_first_task = args.stop_first_task

    # generate output folder and result folders
    out_path, exp_name = experiment_folder(args.outpath, dev_mode, args.approach)
 

    print("Current Seed {}".format(args.seed))
    
    # fix seed
    seed_everything(seed=args.seed)
    
    """
    Dataset Preparation
    """

    
    train_set, test_set, validation_set, total_classes, subcat_dict = get_dataset(args.dataset, args.data_path, args.pretrained_path)
    
      
    # task_dict = {task_id: list_of_class_ids}
    task_dict, subcategories_dict, all_subcategories_dict = get_task_dict_incdec(args.n_task, args.subcategories_csv_path, args.pipeline, args.subcategories_randomize, out_path, subcat_dict)
    
    
        
    """
    Generate Subset For Each Task
    """

    
    cl_train_val = get_training_validation_subset_for_tasks(args.approach, args.pipeline, train_set,
                                                            args.n_task, initial_split,
                                                            total_classes,
                                                            validation_set,
                                                            subcategories_dict)

    
    train_dataset_list, train_sizes, val_dataset_list, val_sizes = cl_train_val.collect()
    print("train_sizes: {}".format(train_sizes))
    print("val_sizes: {}".format(val_sizes))

    
    cl_test = get_test_subset_for_tasks(args.approach, args.pipeline, test_set,
                                                            args.n_task, initial_split, 
                                                            total_classes,subcategories_dict)


    test_dataset_list, test_sizes, _, _  = cl_test.collect()
    


    train_loaders = []
    valid_loaders = []
    test_loaders = []
    
    train_loaders, valid_loaders, test_loaders = get_data_loaders(validation_set, args.sampler, args.batch_size, 
                                                                  args.nw, train_dataset_list, val_dataset_list, test_dataset_list)

    
    """
    Logger Init
    """
    logger = IncDecLogger(out_path=out_path, n_task=args.n_task, task_dict=task_dict, all_subcategories_dict = all_subcategories_dict, class_to_idx= train_set.get_class_to_idx(), num_classes=total_classes, criterion_type=args.criterion_type)
    val_logger = IncDecLogger(out_path=out_path, n_task=args.n_task, task_dict=task_dict, all_subcategories_dict = all_subcategories_dict, class_to_idx= train_set.get_class_to_idx(), num_classes=total_classes, criterion_type=args.criterion_type, validation_mode=True)
    result_folder(out_path, "tensorboard")
    result_folder(out_path, "logger")
    result_folder(out_path, "validation_logger")

    #Average time keeper for training
    avg_time_train = AverageMeter()
 
 

    if args.approach == 'incdec':
        approach = DataIncrementalDecrementalMethod(args=args, device = device,
                    out_path=out_path,
                    task_dict=task_dict,
                    total_classes=total_classes,
                    class_to_idx= train_set.get_class_to_idx(),
                    all_subcategories_dict = all_subcategories_dict,
                    )
    elif args.approach == 'incdec_lwf':
        approach = DIDM_lwf(args=args, device = device,
                            out_path=out_path,
                            task_dict=task_dict,
                            total_classes=total_classes,
                            class_to_idx= train_set.get_class_to_idx(),
                            all_subcategories_dict = all_subcategories_dict,
                            )
    elif args.approach == 'incdec_fd':
        approach = DIDM_fd(args=args, device = device,
                            out_path=out_path,
                            task_dict=task_dict,
                            total_classes=total_classes,
                            class_to_idx= train_set.get_class_to_idx(),
                            all_subcategories_dict = all_subcategories_dict,
                            )
    elif args.approach == 'incdec_ewc':
        approach = DIDM_ewc(args=args, device = device,
                            out_path=out_path,
                            task_dict=task_dict,
                            total_classes=total_classes,
                            class_to_idx= train_set.get_class_to_idx(),
                            all_subcategories_dict = all_subcategories_dict,
                            )

    else:
        sys.exit("Approach not Implemented")

    summary_logger = SummaryLogger(all_args, all_default_args, args.outpath, args.approach)
    summary_logger.summary_parameters(exp_name)
 
    for task_id, train_loader in enumerate(train_loaders):

 
            
        print("#"*40 + " --> TRAINING HEAD/TASK {}".format(task_id))

        """
        Pre-train
        """
        
        if task_id == 0 and args.pretrained_path != "None":
            print("Loading model from path {}".format(args.pretrained_path))
            # Here i substitute the normal head with a 200 size head, to load the pre-trained model on 200 classes
            approach.substitute_head(200)
            rollback_model_movinet(approach, args.pretrained_path, name='checkpoint_adam.pt')
            approach.substitute_head(total_classes)

        approach.pre_train(task_id, train_loader,  valid_loaders[task_id])

        # rolling back to the best model of the past task
        if task_id != 0:
            model_name = os.path.join(out_path,"best_mAP_task_{}_model.pth").format((task_id-1))
            print("Loading model from path: {}".format(model_name))
            rollback_model(approach, model_name, device, name=str(model_name))
            
                
        
        best_taw_accuracy,  best_tag_accuracy, best_accuracy, best_mAP = 0, 0, 0, 0
        best_epoch = 0
        best_loss = math.inf 
        
        n_epochs = args.epochs
        
        
        """
        Main train Loop
        """
        # for early stopping 
        no_decrement_count = 0
        best_loss = float(math.inf)
                
        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch, n_epochs))


            
            if epoch == 0:
                store_model(approach, out_path)

            end_time = time.time()

            approach.train(task_id, train_loader, epoch, n_epochs)
            
            acc, _ , test_loss, _, mean_ap_eval, _, _, _, _,_,_,_,_ = approach.eval(task_id, task_id, valid_loaders[task_id], epoch, verbose=True, testing=None)
            
            previous_lr = approach.optimizer.param_groups[0]["lr"]
            
            if args.scheduler_type == 'reduce_plateau':
                approach.reduce_lr_on_plateau.step(mean_ap_eval)
            else:
                approach.reduce_lr_on_plateau.step()
                
            current_lr = approach.optimizer.param_groups[0]["lr"]


            avg_time_train.update(time.time() - end_time)

            print(f"Last time {avg_time_train.val:.3f} - Average time ({avg_time_train.avg:.3f})\t")

            if current_lr != previous_lr:
                model_name = os.path.join(out_path,"best_mAP_task_{}_model.pth").format(task_id)
                print("Loading model from path: {}".format(model_name))
                rollback_model(approach, model_name, device, name=str(model_name))
            
            if mean_ap_eval > best_mAP:
                old_mAP = best_mAP
                best_mAP = mean_ap_eval
                name_model = "best_mAP_task_" + str(task_id)
                store_model(approach, out_path,name=name_model)
                print(f"  --> from mAP {old_mAP:.3f} to {best_mAP:.3f}")
                best_epoch = epoch
                no_decrement_count = 0
            else:
                no_decrement_count += 1
            if args.scheduler_type == "fixd":
                if no_decrement_count == args.early_stopping_val:
                    print(f"Early stopping because classification loss didn't improve for{args.early_stopping_val} epochs\t")
                    break
            else:
                # Stops if the learning rate is lower than a threshold
                if current_lr < float(1e-5):
                    print(f"Early stopping because learning rate threshold is reached \t")
                    break

            print(f"Current learning rate for the next epoch is: {current_lr}")
                    


            logger.print_best_epoch(best_epoch, task_id)
            val_logger.print_best_epoch(best_epoch, task_id)
            
            
        """
        Test Final Model
        """
        model_name = os.path.join(out_path,"best_mAP_task_{}_model.pth").format(task_id)
        print("Loading model from path: {}".format(model_name))
        rollback_model(approach, model_name, device, name=str(model_name))


        # Here do a validation eval for the best epoch model
        # this is redundant, but here i print metrics of the best model on the validation set...
        vacc_value, vap_value, _, vacc_per_class, vmean_ap, vmap_weighted, vprecision_per_class, vrecall_per_class, vexact_match, vap_per_subcategory, vrecall_per_subcategory, vaccuracy_per_subcategory, vprecision_per_subcategory  = approach.eval(task_id, task_id, valid_loaders[task_id], epoch,  verbose=False, testing='val')
        val_logger.update_accuracy(current_training_task_id=task_id, acc_value=vacc_value, 
                                   ap_value=vap_value, acc_per_class=vacc_per_class, mean_ap=vmean_ap, 
                                   map_weighted=vmap_weighted, precision_per_class=vprecision_per_class, 
                                   recall_per_class=vrecall_per_class, exact_match = vexact_match, 
                                   ap_per_subcategory=vap_per_subcategory, recall_per_subcategory=vrecall_per_subcategory, accuracy_per_subcategory=vaccuracy_per_subcategory, precision_per_subcategory=vprecision_per_subcategory)
        

 
        #For incdec approach for now there is a single test set to be evaluated
        acc_value, ap_value, _, acc_per_class, mean_ap, map_weighted, precision_per_class, recall_per_class, exact_match, ap_per_subcategory, recall_per_subcategory, accuracy_per_subcategory,precision_per_subcategory = approach.eval(task_id, task_id, test_loaders[task_id], epoch,  verbose=False, testing='test')
        logger.update_accuracy(current_training_task_id=task_id, acc_value=acc_value, ap_value=ap_value, acc_per_class=acc_per_class, mean_ap=mean_ap, map_weighted=map_weighted, precision_per_class=precision_per_class, recall_per_class=recall_per_class, exact_match=exact_match, ap_per_subcategory=ap_per_subcategory, recall_per_subcategory=recall_per_subcategory, accuracy_per_subcategory=accuracy_per_subcategory, precision_per_subcategory=precision_per_subcategory)
        logger.update_forgetting(current_training_task_id=task_id)
        logger.print_latest(current_training_task_id=task_id)
        
 
        """
        Post Training
        """
        
        logger.print_file()
        
        val_logger.print_file()
  
        approach.post_train(task_id=task_id, train_loader=train_loader)

        #If i want to stop at the first task
        if stop_first_task == 'yes':
            break

    
    summary_logger = SummaryLogger(all_args, all_default_args, args.outpath, args.approach)
    summary_logger.update_summary(exp_name, logger, avg_time_train.avg)
    store_valid_loader(out_path, valid_loaders, False)