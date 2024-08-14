import sys
from torchvision import transforms
import os
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset
from PIL import Image 
from typing import Tuple,Any 
import torch
import math

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def create_dict_classes_subcategories(classes_csv):
    df = pd.read_csv(classes_csv)
    classes_subcategories = {}

    for _, row in df.iterrows():
        class_name = row['Category']
        subcategory = str(row['Subcategory'])
        
        # Check if the class_name is already in the dictionary, if not, create a new entry
        if class_name not in classes_subcategories:
            classes_subcategories[class_name] = []
        
        # Add the subcategory to the corresponding class_name entry in the dictionary
        classes_subcategories[class_name].append(subcategory)

    return classes_subcategories


class KineticsDataset(Dataset):
    def __init__(self, data_path, transform, dataset_type, fps):

        #In folder_csv are place: train.csv, validation.csv, test.csv and classes.csv
        folder_csv = os.path.join(data_path,'Info')
        if dataset_type == 'train':
            self.data_csv = os.path.join(folder_csv, 'train.csv')
        elif dataset_type == 'validation':
            self.data_csv = os.path.join(folder_csv, 'validation.csv')
        elif dataset_type == 'test':
            self.data_csv = os.path.join(folder_csv, 'test.csv')
        else:
            #This only to get all the data, used for mean and std
            self.data_csv = os.path.join(folder_csv, 'tbdownloaded.csv')


        self.data_folder = os.path.join(data_path,'Videos')

        df = pd.read_csv(self.data_csv)

        self.data = []
        self.targets = []
        self.subcategories = []

        #create a mapping between classes - subcategories
        class_csv = os.path.join(folder_csv, 'classes.csv')
        self.classes_subcategories = create_dict_classes_subcategories(class_csv)

        #create a index for each class -- {class: idx}
        self.class_to_idx = {key: i for i, key in enumerate(self.classes_subcategories.keys())}

        for _, row in df.iterrows():
            #replace to match how the data was called in the folder
            id_data = 'id_' + str(row['youtube_id']) + '_' + '{:06d}'.format(row['time_start']) + '_' + '{:06d}'.format(row['time_end'])
            self.data.append(id_data)

            #retrieve the class - targets from category.csv
            data_dir = os.path.join(self.data_folder, id_data)
            cat_csv_path = os.path.join(data_dir,'category.csv')
            cat_csv = pd.read_csv(cat_csv_path)
            cat_row = next(cat_csv.iterrows())[1]
            matching_class = cat_row['Category']
            #retrieve the behavior from category.csv
            self.targets.append(self.class_to_idx[matching_class])
            matching_behavior = cat_row['Sub-behavior']
            self.subcategories.append(matching_behavior)
        
        self.transform = transform

        self.fps = fps

    def get_class_to_idx(self):
        return self.class_to_idx
    
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_id, target, subcat = self.data[index], self.targets[index], self.subcategories[index]

        video_id_path = os.path.join(self.data_folder,img_id)
        
        images_path = os.path.join(video_id_path,'jpgs')

        video = []
        std_video_len = self.fps*10

        current_video_len = len(os.listdir(images_path))

        for i in range(current_video_len):
            image_name = 'image_{:05d}.jpg'.format((i%current_video_len)+1)
            im_path = os.path.join(images_path,image_name)
            with open(im_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                video.append(img)      
        
        video = torch.stack(video,0).permute(1, 0, 2, 3)
        # repeat video until max frame reach 
        n_repeat = math.ceil(std_video_len/current_video_len)
        video = video.repeat(1, n_repeat, 1, 1)
        # clip the first 50 frames
        video = video[:,:std_video_len, :, :]
        binarized_target = preprocessing.label_binarize([target], classes=[i for i in range(len(self.class_to_idx.keys()))])
        return video, target, binarized_target, subcat, images_path
    


        
        
def get_dataset(dataset_type, data_path, pretrained_path=None):
    if dataset_type == "kinetics":
        
        print("Loading Kinetics")
        
        if pretrained_path == None:
            train_transform = [transforms.Resize(size=(200,200)),
                            
                            transforms.RandomCrop(172),
                            transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                #added normalization factors computed on the actual dataset, training set
                                transforms.Normalize(mean=[0.4516, 0.3883, 0.3569],std=[0.2925, 0.2791, 0.2746])
                            ]
            train_transform = transforms.Compose(train_transform)

            test_transform = [transforms.Resize(size=(172,172)),
                                transforms.CenterCrop(172),
                                transforms.ToTensor(),
                                #added normalization factors computed on the actual dataset, training set
                                transforms.Normalize(mean=[0.4516, 0.3883, 0.3569],std=[0.2925, 0.2791, 0.2746])
                            ]  
            test_transform = transforms.Compose(test_transform)
        else:
            train_transform = [transforms.Resize(size=(200,200)),
                        transforms.RandomCrop(172),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
                            ]
            train_transform = transforms.Compose(train_transform)
        
            test_transform = [transforms.Resize(size=(200,200)),
                            transforms.CenterCrop(172),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
                            ]  
            test_transform = transforms.Compose(test_transform)

        # since we worked with 5 fps for all experiments, here i set it manually, it could be modified to 
        # extract it from outside if different tests will be required with different parameters
        train_set = KineticsDataset(data_path, train_transform, dataset_type='train', fps=5)
        valid_set = KineticsDataset(data_path, test_transform, dataset_type='validation', fps=5)
        test_set = KineticsDataset(data_path, test_transform, dataset_type='test', fps=5)

        # since we worked with fixed 5 classes and always the same subcategories, we fixed them here
        n_classes = 5
        subcat_dict = {
        'food': [
            'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts',
            'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon',
            'sucking lolly', 'tasting beer', 'tasting food', 'tasting wine', 'sipping cup'
        ],
        'phone': [
            'texting', 'talking on cell phone', 'looking at phone'
        ],
        'smoking': [
            'smoking', 'smoking hookah', 'smoking pipe'
        ],
        'fatigue': [
            'sleeping', 'yawning', 'headbanging', 'headbutting', 'shaking head'
        ],
        'selfcare': [
            'scrubbing face', 'putting in contact lenses', 'putting on eyeliner', 'putting on foundation',
            'putting on lipstick', 'putting on mascara', 'brushing hair', 'brushing teeth', 'braiding hair',
            'combing hair', 'dyeing eyebrows', 'dyeing hair'
        ]
        }
    
        
    
    return train_set, test_set, valid_set, n_classes, subcat_dict

 
            
        