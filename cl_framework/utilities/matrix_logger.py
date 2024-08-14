import numpy as np
import os
from datetime import datetime
import sys 

class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()



class IncDecLogger():
    def __init__(self, out_path, n_task, task_dict, all_subcategories_dict, class_to_idx, num_classes, criterion_type, print_log_files=True, begin_time=None, validation_mode=False) -> None:
        self.criterion_type = criterion_type
        self.acc = np.zeros((n_task))
        self.mean_ap = np.zeros((n_task))
        self.map_weighted = np.zeros((n_task))
        self.ap = np.zeros((n_task, num_classes))
        self.acc_per_class = np.zeros((n_task, num_classes))
        self.precision_per_class = np.zeros((n_task, num_classes))
        self.recall_per_class = np.zeros((n_task, num_classes))
        self.exact_match = np.zeros((n_task))
        
        self.all_subcategories_dict = all_subcategories_dict
        self.class_to_idx = class_to_idx

        self.num_classes = num_classes
        self.num_subcategories = 0
        self.names_subcategories = []
        for class_name in all_subcategories_dict:
            self.num_subcategories += len(all_subcategories_dict[class_name])
            for idx_subcat in range(len(all_subcategories_dict[class_name])):
                self.names_subcategories.append(all_subcategories_dict[class_name][idx_subcat])
        self.ap_per_subcategory = np.zeros((n_task, self.num_subcategories))
        self.recall_per_subcategory = np.zeros((n_task, self.num_subcategories))
        self.accuracy_per_subcategory = np.zeros((n_task, self.num_subcategories))
        self.precision_per_subcategory = np.zeros((n_task, self.num_subcategories))

        self.forg_acc = np.zeros((n_task))
        self.forg_map = np.zeros((n_task))

        self.forg_ap_per_class = np.zeros((n_task, num_classes))
        self.forg_recall_per_class = np.zeros((n_task, num_classes))
        self.forg_precision_per_class = np.zeros((n_task, num_classes))
        self.forg_accuracy_per_class = np.zeros((n_task, num_classes))

        self.forg_ap_per_subcategory = np.zeros((n_task, self.num_subcategories))
        self.forg_recall_per_subcategory = np.zeros((n_task, self.num_subcategories))
        self.forg_accuracy_per_subcategory = np.zeros((n_task, self.num_subcategories))
        self.forg_precision_per_subcategory = np.zeros((n_task, self.num_subcategories))

        self.best_epoch = np.full(n_task,-1)
        if validation_mode:
            self.out_path = os.path.join(out_path, "validation_logger")
        else:
            self.out_path = os.path.join(out_path, "logger")
        
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time
        if print_log_files:
            self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")
            sys.stdout = FileOutputDuplicator(sys.stdout,
                                            os.path.join(out_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
            sys.stderr = FileOutputDuplicator(sys.stderr,
                                            os.path.join(out_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')
        


    def update_accuracy(self, current_training_task_id, acc_value, ap_value, acc_per_class, mean_ap, map_weighted, precision_per_class, recall_per_class, exact_match, ap_per_subcategory, recall_per_subcategory, accuracy_per_subcategory, precision_per_subcategory):
        self.acc[current_training_task_id] = acc_value * 100
        self.mean_ap[current_training_task_id] = mean_ap * 100
        self.map_weighted[current_training_task_id] = map_weighted * 100
        self.ap[current_training_task_id] = ap_value
        self.acc_per_class[current_training_task_id] = acc_per_class
        self.precision_per_class[current_training_task_id] = precision_per_class
        self.recall_per_class[current_training_task_id] = recall_per_class

        for idx_subcat in range(len(self.names_subcategories)):
            self.ap_per_subcategory[current_training_task_id, idx_subcat] = ap_per_subcategory[self.names_subcategories[idx_subcat]]
            self.recall_per_subcategory[current_training_task_id, idx_subcat] = recall_per_subcategory[self.names_subcategories[idx_subcat]]
            self.accuracy_per_subcategory[current_training_task_id, idx_subcat] = accuracy_per_subcategory[self.names_subcategories[idx_subcat]]
            self.precision_per_subcategory[current_training_task_id, idx_subcat] = precision_per_subcategory[self.names_subcategories[idx_subcat]]

        if self.criterion_type == 'multilabel':
            self.exact_match[current_training_task_id] = exact_match
         
  

    def print_latest(self, current_training_task_id):
        print('\n >>> Test on task {:2d} : acc={:5.1f}%, '
              ' |  <<<'.format(current_training_task_id, 
                                        self.acc[current_training_task_id]))
        

    def update_forgetting(self, current_training_task_id):
        # this is done because in the first task it has to be 0 the forgetting, but doing max like that returns problems
        if current_training_task_id != 0:
            self.forg_acc[current_training_task_id] = np.max(self.acc[:current_training_task_id]) - self.acc[current_training_task_id]
            self.forg_map[current_training_task_id] = np.max(self.mean_ap[:current_training_task_id]) - self.mean_ap[current_training_task_id]

            for idx_class in range(self.num_classes):
                self.forg_ap_per_class[current_training_task_id, idx_class] = self.ap[:current_training_task_id, idx_class].max(0) - self.ap[current_training_task_id, idx_class]
                self.forg_recall_per_class[current_training_task_id, idx_class] = self.recall_per_class[:current_training_task_id, idx_class].max(0) - self.recall_per_class[current_training_task_id, idx_class]
                self.forg_accuracy_per_class[current_training_task_id, idx_class] = self.acc_per_class[:current_training_task_id, idx_class].max(0) - self.acc_per_class[current_training_task_id, idx_class]
                self.forg_precision_per_class[current_training_task_id, idx_class] = self.precision_per_class[:current_training_task_id, idx_class].max(0) - self.precision_per_class[current_training_task_id, idx_class]
                
            for idx_subcat in range(len(self.names_subcategories)):
                self.forg_ap_per_subcategory[current_training_task_id, idx_subcat] = self.ap_per_subcategory[:current_training_task_id, idx_subcat].max(0) - self.ap_per_subcategory[current_training_task_id, idx_subcat]
                self.forg_recall_per_subcategory[current_training_task_id, idx_subcat] = self.recall_per_subcategory[:current_training_task_id, idx_subcat].max(0) - self.recall_per_subcategory[current_training_task_id, idx_subcat]
                self.forg_accuracy_per_subcategory[current_training_task_id, idx_subcat] = self.accuracy_per_subcategory[:current_training_task_id, idx_subcat].max(0) - self.accuracy_per_subcategory[current_training_task_id, idx_subcat]
                self.forg_precision_per_subcategory[current_training_task_id, idx_subcat] = self.precision_per_subcategory[:current_training_task_id, idx_subcat].max(0) - self.precision_per_subcategory[current_training_task_id, idx_subcat]
            
           
          
    def print_file(self):
        np.savetxt(os.path.join(self.out_path, 'acc.out'), self.acc, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'mean_ap.out'), self.mean_ap, delimiter=',', fmt='%.3f')
        
        np.savetxt(os.path.join(self.out_path, 'map_weighted.out'), self.map_weighted, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'ap.out'), self.ap, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'acc_per_class.out'), self.acc_per_class, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'precision_per_class.out'), self.precision_per_class, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'recall_per_class.out'), self.recall_per_class, delimiter=',', fmt='%.3f')

        np.savetxt(os.path.join(self.out_path, 'forg_acc.out'), self.forg_acc, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_mean_ap.out'), self.forg_map, delimiter=',', fmt='%.3f')

        np.savetxt(os.path.join(self.out_path, 'forg_ap.out'), self.forg_ap_per_class, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_recall_per_class.out'), self.forg_recall_per_class, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_precision_per_class.out'), self.forg_precision_per_class, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_accuracy_per_class.out'), self.forg_accuracy_per_class, delimiter=',', fmt='%.3f')

        np.savetxt(os.path.join(self.out_path, 'forg_ap_per_subcategory.out'), self.forg_ap_per_subcategory, header=','.join(self.names_subcategories), delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_recall_per_subcategory.out'), self.forg_recall_per_subcategory, header=','.join(self.names_subcategories), delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_precision_per_subcategory.out'), self.forg_precision_per_subcategory, header=','.join(self.names_subcategories), delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'forg_accuracy_per_subcategory.out'), self.forg_accuracy_per_subcategory, header=','.join(self.names_subcategories), delimiter=',', fmt='%.3f')
        if self.criterion_type == 'multilabel':
            np.savetxt(os.path.join(self.out_path, 'exact_match.out'), self.exact_match, delimiter=',', fmt='%.3f')
        


    def print_best_epoch(self, best_epoch, task_id):
        self.best_epoch[task_id] = best_epoch
        np.savetxt(os.path.join(self.out_path, 'best_epoch_tasks.out'), self.best_epoch, delimiter=',', fmt='%d')
 
