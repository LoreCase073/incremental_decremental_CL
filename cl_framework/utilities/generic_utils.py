import os
import random
import string
import shutil
import json
import random
import numpy as np
import torch
import pandas as pd

from copy import deepcopy


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)


def experiment_folder(root_path, dev_mode, approach_name):
    if os.path.exists(os.path.join(root_path, 'exp_folder')):
        shutil.rmtree(os.path.join(root_path, 'exp_folder'), ignore_errors=True)

    if dev_mode:
        exp_folder = 'exp_folder'
    else:
        exp_folder = approach_name + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    out_path = os.path.join(root_path, exp_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path, exp_folder

def result_folder(out_path, name):
    if not os.path.exists(os.path.join(out_path, name)):
        os.mkdir(os.path.join(out_path, name))


def store_params(out_path, n_epoch, bs, n_task, old_reconstruction, loss_weight):
    params = {}
    params['n_epoch'] = n_epoch
    params['bs'] = bs
    params['n_task'] = n_task
    params['old_reconstruction'] = old_reconstruction
    params['loss_weight'] = loss_weight
    store_dictionary(params, out_path, name='params')


def store_dictionary(d, out_path, name):
    d={str(k): v for k, v in d.items()}
    with open(os.path.join(out_path, name+'.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


def rollback_model(approach, out_path, device, name=None):
    if name is not None:
        approach.model.load_state_dict(torch.load(out_path, map_location=device))
        print("Model Loaded {}".format(out_path))
    else:
        approach.model.load_state_dict(torch.load(os.path.join(out_path,'_model.pth'), map_location=device))

def rollback_model_movinet(approach, out_path, name=None):
    if name is not None:
        checkpoint = torch.load(out_path, map_location="cpu")
        approach.model.load_state_dict(checkpoint["model"])
        print("Model Loaded {}".format(out_path))
    else:
        approach.model.load_state_dict(torch.load(os.path.join(out_path,'_model.pth'), map_location="cpu"))


def store_model(approach, out_path, name=""):
    torch.save(deepcopy(approach.model.state_dict()), os.path.join(out_path, name+"_model.pth"))

#TODO: rimuovere
def remap_targets(train_set, test_set, total_classes):
    l = list(range(total_classes))
    l_sorted = deepcopy(l)
    random.shuffle(l)
    label_mapping = dict(zip(l_sorted, l))
    
    # remap train labels following label_mapping    
    
    for i in range(len(train_set.targets)):
        train_set.targets[i]=label_mapping[train_set.targets[i]]
    
 
    for key in train_set.class_to_idx.keys():
        train_set.class_to_idx[key] = label_mapping[train_set.class_to_idx[key]]
        
    # remap test labels following label_mapping    
    
    for i in range(len(test_set.targets)):
        test_set.targets[i]=label_mapping[test_set.targets[i]]
    
 
    for key in test_set.class_to_idx.keys():
        test_set.class_to_idx[key] = label_mapping[test_set.class_to_idx[key]]
        
        
     
    return train_set, test_set, label_mapping

def store_valid_loader(out_path, valid_loaders, store):
    if store:
        for i, loader in enumerate(valid_loaders):
            torch.save(loader, os.path.join(out_path, 'dataloader_'+str(i)+'.pth'))







def get_task_dict_incdec(n_task, subcategories_csv_path, pipeline, subcategories_randomize, out_path, init_dict):
    starting_data_dict = init_dict

    d = {}

    subcategories_dicts = []
       
    if pipeline == 'baseline' or pipeline == 'joint_incremental':
        for i in range(n_task):
            d[i] = (len(starting_data_dict.keys()))
            subcategories_dicts.append(starting_data_dict)
    elif pipeline == 'decremental':
        
        result_folder(out_path, 'subcategories_task')

        save_path = os.path.join(out_path, 'subcategories_task')

        subcategories_to_remove = pd.read_csv(subcategories_csv_path)
        # i deepcopy just to be sure that i do not change the starting data dict if i need to reuse it
        current_subcategories_dict = deepcopy(starting_data_dict)

        # shuffling columns to do a randomized order of decremental subcategories
        # randomize all but the first row, which should not have any subcategories removed
        if subcategories_randomize == 'yes':
            for column in current_subcategories_dict:
                subcategories_to_remove[column][1:] = np.random.permutation(subcategories_to_remove[column].values[1:])

        for i, row in subcategories_to_remove.iterrows():
            d[i] = (len(starting_data_dict.keys()))

            # iterate the classes to remove them
            for idx_class in current_subcategories_dict.keys():
                # get how many subcategories to remove
                count_to_remove = row[idx_class]
                # remove the subcategories if != 0
                if count_to_remove != 0:
                    for count_idx in range(count_to_remove):
                        # get current subcategories of that class from the current dict of subcategories
                        subcategories_count = len(current_subcategories_dict[idx_class])
                        # now select randomly the index of the subcategory to be removed
                        idx_to_remove = random.randint(0,subcategories_count-1)
                        # check if it's not the last subcategory from some of the classes
                        el_to_remove = current_subcategories_dict[idx_class][idx_to_remove]
                        last_remaining = False
                        for tmp_idx_class in current_subcategories_dict.keys():
                            if len(current_subcategories_dict[tmp_idx_class]) == 1 and el_to_remove in current_subcategories_dict[tmp_idx_class]:
                                last_remaining = True
                        # remove the item from the list if last_remaining not True
                        # make sure to remove it from all of the classes
                        if last_remaining == False:
                            for tmp_idx_class in current_subcategories_dict.keys():
                                if el_to_remove in current_subcategories_dict[tmp_idx_class]:
                                    current_subcategories_dict[tmp_idx_class].remove(el_to_remove)
                        else:
                            print("Removal could not happen because it would be the last remaining subcategory from some of the classes.")
            # add the new dict to the subcategories dicts
            subcategories_dicts.append(deepcopy(current_subcategories_dict))
            csv_subcategories = pd.DataFrame.from_dict(current_subcategories_dict, orient='index')
            csv_subcategories.to_csv(os.path.join(save_path,'task_{}'.format(i)),header=False,index=False)
    elif pipeline == 'incremental_decremental':
        result_folder(out_path, 'subcategories_task')

        save_path = os.path.join(out_path, 'subcategories_task')

        subcategories_to_substitute = pd.read_csv(subcategories_csv_path)
        # get a subset of subcategories to start with, as described in the first row of the csv
        tmp_subcategories_dict = deepcopy(starting_data_dict)

        if subcategories_randomize == 'yes':
            for column in tmp_subcategories_dict:
                subcategories_to_substitute[column][1:] = np.random.permutation(subcategories_to_substitute[column].values[1:])

        first_task_subcategories = subcategories_to_substitute.iloc[0]

        current_subcategories_dict = {}

        for idx_class in tmp_subcategories_dict.keys():
            current_subcategories_dict[idx_class] = []

        # get the subcategories for the first task
        for idx_class in tmp_subcategories_dict.keys():
                # get how many subcategories to remove
                count_to_get = first_task_subcategories[idx_class]
                if count_to_get != 0:
                    for count_idx in range(count_to_get):
                        if len(current_subcategories_dict[idx_class]) < count_to_get:
                            # get current subcategories of that class from the current dict of subcategories
                            subcategories_count = len(tmp_subcategories_dict[idx_class])
                            # now select randomly the index of the subcategory to be added to the first task
                            idx_to_add = random.randint(0,subcategories_count-1)
                            # check if it's not the last element from some of the classes
                            el_to_add_and_remove = tmp_subcategories_dict[idx_class][idx_to_add]
                            last_remaining = False
                            for tmp_idx_class in tmp_subcategories_dict.keys():
                                if len(tmp_subcategories_dict[tmp_idx_class]) == 1 and el_to_add_and_remove in tmp_subcategories_dict[tmp_idx_class]:
                                    last_remaining = True
                            if last_remaining == False:
                                for tmp_idx_class in tmp_subcategories_dict.keys():
                                    if el_to_add_and_remove in tmp_subcategories_dict[tmp_idx_class]:
                                        current_subcategories_dict[tmp_idx_class].append(el_to_add_and_remove)
                                        tmp_subcategories_dict[tmp_idx_class].remove(el_to_add_and_remove)
                            else:
                                print("Removal could not happen because it would be the last remaining subcategory from some of the classes.")
        d[0] = (len(starting_data_dict.keys()))
        # add the first dict to the subcategories dictionaries
        subcategories_dicts.append(deepcopy(current_subcategories_dict))
        # dict with all the subcategories from the first task to be removed
        subcategories_to_remove = deepcopy(current_subcategories_dict)
        csv_subcategories = pd.DataFrame.from_dict(current_subcategories_dict, orient='index')
        csv_subcategories.to_csv(os.path.join(save_path,'task_0'),header=False,index=False)

        for i, row in subcategories_to_substitute.iterrows():
            # skip the first task
            if i != 0:
                d[i] = (len(starting_data_dict.keys()))

                # iterate the classes to remove them
                for idx_class in tmp_subcategories_dict.keys():
                    count_to_get = row[idx_class]
                    if count_to_get != 0:
                        for count_idx in range(count_to_get):
                            subcategories_count_remaining_to_add = len(tmp_subcategories_dict[idx_class])
                            if subcategories_count_remaining_to_add != 0:
                                # now select randomly the index of the subcategory to be added to the task
                                idx_to_add = random.randint(0,subcategories_count_remaining_to_add-1)

                                subcategories_count_remaining_to_remove = len(subcategories_to_remove[idx_class])
                                if subcategories_count_remaining_to_remove != 0:
                                    # now select randomly the index of the subcategory to be removed to the task
                                    idx_to_remove = random.randint(0,subcategories_count_remaining_to_remove-1)
                                    # check if it's not the last subcategory from some of the classes
                                    el_to_remove = subcategories_to_remove[idx_class][idx_to_remove]
                                    el_to_add = tmp_subcategories_dict[idx_class][idx_to_add]
                                    last_remaining = False
                                    for tmp_idx_class in current_subcategories_dict.keys():
                                        if len(current_subcategories_dict[tmp_idx_class]) == 1 and el_to_remove in current_subcategories_dict[tmp_idx_class]:
                                            last_remaining = True
                                    # remove the item from the list if last_remaining not True
                                    # make sure to remove it from all of the classes
                                    # if it's the last in some of them, just add the subcategory to all of them
                                    if last_remaining == False:
                                        for tmp_idx_class in current_subcategories_dict.keys():
                                            if el_to_remove in current_subcategories_dict[tmp_idx_class]:
                                                current_subcategories_dict[tmp_idx_class].remove(el_to_remove)
                                                subcategories_to_remove[tmp_idx_class].remove(el_to_remove)
                                            if el_to_add in tmp_subcategories_dict[tmp_idx_class]:
                                                current_subcategories_dict[tmp_idx_class].append(el_to_add)
                                                tmp_subcategories_dict[tmp_idx_class].remove(el_to_add)
                                    else:
                                        print("Removal could not happen because it would be the last remaining subcategory from some of the classes.")
                                        print("Just adding the new subcategory to all of them.")
                                        # just add the new subcat for all of the classes
                                        for tmp_idx_class in tmp_subcategories_dict.keys():
                                            if el_to_add in tmp_subcategories_dict[tmp_idx_class]:
                                                current_subcategories_dict[tmp_idx_class].append(el_to_add)
                                                tmp_subcategories_dict[tmp_idx_class].remove(el_to_add)
                                else:
                                    print("No more subcategories to remove for this class, but still adding the subcategories if remaining.")
                                    el_to_add = tmp_subcategories_dict[idx_class][idx_to_add]
                                    for tmp_idx_class in current_subcategories_dict.keys():
                                        if el_to_add in tmp_subcategories_dict[tmp_idx_class]:
                                            current_subcategories_dict[tmp_idx_class].append(el_to_add)
                                            tmp_subcategories_dict[tmp_idx_class].remove(el_to_add)
                            else:
                                print("No more subcategories to add for this class.")

                subcategories_dicts.append(deepcopy(current_subcategories_dict))
                csv_subcategories = pd.DataFrame.from_dict(current_subcategories_dict, orient='index')
                csv_subcategories.to_csv(os.path.join(save_path,'task_{}'.format(i)),header=False,index=False)
            num_subcategories_for_task = [len(current_subcategories_dict[key]) for key in current_subcategories_dict]
            print("Subcategories count for current task: {}".format(num_subcategories_for_task))

    return d, subcategories_dicts, starting_data_dict


def old_get_task_dict_incdec(n_task, subcategories_csv_path, pipeline, subcategories_randomize, out_path, init_dict):
    starting_data_dict = init_dict

    d = {}

    subcategories_dicts = []
       
    if pipeline == 'baseline' or pipeline == 'joint_incremental':
        for i in range(n_task):
            d[i] = (len(starting_data_dict.keys()))
            subcategories_dicts.append(starting_data_dict)
    elif pipeline == 'decremental':
        
        result_folder(out_path, 'subcategories_task')

        save_path = os.path.join(out_path, 'subcategories_task')

        subcategories_to_remove = pd.read_csv(subcategories_csv_path)
        # i deepcopy just to be sure that i do not change the starting data dict if i need to reuse it
        current_subcategories_dict = deepcopy(starting_data_dict)

        # shuffling columns to do a randomized order of decremental subcategories
        # randomize all but the first row, which should not have any subcategories removed
        if subcategories_randomize == 'yes':
            for column in current_subcategories_dict:
                subcategories_to_remove[column][1:] = np.random.permutation(subcategories_to_remove[column].values[1:])

        for i, row in subcategories_to_remove.iterrows():
            d[i] = (len(starting_data_dict.keys()))

            # iterate the classes to remove them
            for idx_class in current_subcategories_dict.keys():
                # get how many subcategories to remove
                count_to_remove = row[idx_class]
                # remove the subcategories if != 0
                if count_to_remove != 0:
                    for count_idx in range(count_to_remove):
                        # get current subcategories of that class from the current dict of subcategories
                        subcategories_count = len(current_subcategories_dict[idx_class])
                        # now select randomly the index of the subcategory to be removed
                        idx_to_remove = random.randint(0,subcategories_count-1)
                        # remove the item from the list
                        del current_subcategories_dict[idx_class][idx_to_remove]
            # add the new dict to the subcategories dicts
            subcategories_dicts.append(deepcopy(current_subcategories_dict))
            csv_subcategories = pd.DataFrame.from_dict(current_subcategories_dict, orient='index')
            csv_subcategories.to_csv(os.path.join(save_path,'task_{}'.format(i)),header=False,index=False)
    elif pipeline == 'incremental_decremental':
        result_folder(out_path, 'subcategories_task')

        save_path = os.path.join(out_path, 'subcategories_task')

        subcategories_to_substitute = pd.read_csv(subcategories_csv_path)
        # get a subset of subcategories to start with, as described in the first row of the csv
        tmp_subcategories_dict = deepcopy(starting_data_dict)

        if subcategories_randomize == 'yes':
            for column in tmp_subcategories_dict:
                subcategories_to_substitute[column][1:] = np.random.permutation(subcategories_to_substitute[column].values[1:])

        first_task_subcategories = subcategories_to_substitute.iloc[0]

        current_subcategories_dict = {}

        # get the subcategories for the first task
        for idx_class in tmp_subcategories_dict.keys():
                # get how many subcategories to remove
                count_to_get = first_task_subcategories[idx_class]
                current_subcategories_dict[idx_class] = []
                if count_to_get != 0:
                    for count_idx in range(count_to_get):
                        # get current subcategories of that class from the current dict of subcategories
                        subcategories_count = len(tmp_subcategories_dict[idx_class])
                        # now select randomly the index of the subcategory to be added to the first task
                        idx_to_add = random.randint(0,subcategories_count-1)
                        current_subcategories_dict[idx_class].append(tmp_subcategories_dict[idx_class][idx_to_add])
                        del tmp_subcategories_dict[idx_class][idx_to_add]
        d[0] = (len(starting_data_dict.keys()))
        # add the first dict to the subcategories dictionaries
        subcategories_dicts.append(deepcopy(current_subcategories_dict))
        # dict with all the subcategories from the first task to be removed
        subcategories_to_remove = deepcopy(current_subcategories_dict)

        for i, row in subcategories_to_substitute.iterrows():
            # skip the first task
            if i != 0:
                d[i] = (len(starting_data_dict.keys()))

                # iterate the classes to remove them
                for idx_class in tmp_subcategories_dict.keys():
                    count_to_get = row[idx_class]
                    if count_to_get != 0:
                        for count_idx in range(count_to_get):
                            subcategories_count_remaining_to_add = len(tmp_subcategories_dict[idx_class])
                            # now select randomly the index of the subcategory to be added to the task
                            idx_to_add = random.randint(0,subcategories_count_remaining_to_add-1)

                            subcategories_count_remaining_to_remove = len(subcategories_to_remove[idx_class])
                            # now select randomly the index of the subcategory to be removed to the task
                            idx_to_remove = random.randint(0,subcategories_count_remaining_to_remove-1)
                            current_subcategories_dict[idx_class].append(tmp_subcategories_dict[idx_class][idx_to_add])
                            del tmp_subcategories_dict[idx_class][idx_to_add]
                            current_subcategories_dict[idx_class].remove(subcategories_to_remove[idx_class][idx_to_remove])
                            del subcategories_to_remove[idx_class][idx_to_remove]
                subcategories_dicts.append(deepcopy(current_subcategories_dict))
                csv_subcategories = pd.DataFrame.from_dict(current_subcategories_dict, orient='index')
                csv_subcategories.to_csv(os.path.join(save_path,'task_{}'.format(i)),header=False,index=False)

    return d, subcategories_dicts, starting_data_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count