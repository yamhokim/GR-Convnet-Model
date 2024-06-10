from torch.utils.data import Dataset 
from typing import Optional
import random
import torch
import os
import glob
import copy

class JacquardGraspCLSDataset(Dataset):
    """
    Parent class for all dataset objects dealing with the Jacquard Dataset. Handles file handling and combining 
    file paths for every instance of training data. Also handles train and test splitting.
    
    Assumes that all jacquard dataset instances have been cleaned using the methods in dataset/preprocess.py

    Child classes must implement methods:
        - get_instance_from_dataset - Gets a single instance from training data given file paths. Handle any augmentations here.
        - get_instance_from_cache - Gets a single instance of training data from the cached files (only use if heavy preprocessing)
        - visualize_instance - visualizes a single instance of training data
        - cache_dataset - caches the entire dataset (only use if heavy preprocessing)

    Cache assumes the following directory structure:
    cache_location
        |__ class_1
            |__ training_data_1.npz
            |__ training_data_2.npz
            ...
        |__ class_2
        ...
    """

    def __init__(
            self, 
            image_size: int, 
            dataset_path: Optional[str] = None, 
            cache_path: Optional[str] = None,
            random_augment: bool = True
        ) -> None:
        assert dataset_path is not None or cache_path is not None, "One of dataset_path or cache_path must be given"
        assert dataset_path is None or cache_path is None, "One of dataset_path or cache_path much be left empty"
        
        self.dataset_path = dataset_path
        self.random_augment = random_augment
        self.cache_path = cache_path
        if self.dataset_path is not None:
            self.class_to_idx, self.idx_to_class = self.get_class_map(self.dataset_path)    
        else:
            self.class_to_idx, self.idx_to_class = self.get_class_map(self.cache_path)
        self.image_size = image_size
        self.individual_file_paths = self.get_all_file_paths(self.dataset_path)

    def extract_test_dataset(self, test_split_ratio: float):
        """
        Extracts test_split_ratio * len(self) number of data points into a test dataset and returns it.
        Also turns off random_augment for the test dataset.
        """
        random.seed(0)
        test_dataset = copy.copy(self)

        self.train_idxs = random.sample(list(range(self.__len__())), k = int((1 - test_split_ratio) * self.__len__()), )
        self.test_idxs = [i for i in range(self.__len__()) if i not in self.train_idxs]

        train_fps = [self.individual_file_paths[i] for i in self.train_idxs]
        test_fps = [self.individual_file_paths[i] for i in self.test_idxs]

        self.individual_file_paths = train_fps
        test_dataset.individual_file_paths = test_fps
        test_dataset.random_augment = False
        return test_dataset

    def get_instance_from_dataset(self, file_paths: list[str], random_augment: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        To be implemented by a child class
        """
        raise NotImplementedError
    
    def get_instance_from_cache(self, file_path: list[str], random_augment: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        To be implemented by a child class
        """
        raise NotImplementedError
    
    def visualize_instance(self, file_paths: list[str]) -> None:
        """
        To be implemented by a child class
        """
        raise NotImplementedError
    
    def cache_dataset(self, cache_location: str):
        """
        To be implemented by a child class
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        file_paths = self.individual_file_paths[idx]
        if self.dataset_path is not None:
            return_values = self.get_instance_from_dataset(file_paths, random_augment=self.random_augment)
        else:
            return_values = self.get_instance_from_cache(file_paths, random_augment=self.random_augment)
        return return_values

    def __len__(self):
        return len(self.individual_file_paths)
    
    def visualize(self, idx) -> None:
        self.visualize_instance(self.individual_file_paths[idx])
    
    def get_all_file_paths(self, dataset_path: str) -> list[list[str]]:
        """
        Returns a list of lists \n
        Each list contains the file paths for the files in the following order:
            - rgb image file paths
            - perfect depth file paths
            - stereo depth file paths
            - grasp file paths
            - mask file paths
            - class of object (not a path)

        If loading from cache, outputs a list containing a single file path
        """
        if self.dataset_path is not None:
            rgb_paths = glob.glob(os.path.join(dataset_path, "*/*", "*RGB.png"))
            output = []
            for rgb_path in rgb_paths:
                assert os.path.isfile(rgb_path)
                instance_data = [rgb_path]

                perfect_depth_path = rgb_path.replace("RGB.png", "perfect_depth.tiff")
                assert os.path.isfile(perfect_depth_path)
                instance_data.append(perfect_depth_path)

                stereo_depth_path = rgb_path.replace("RGB.png", "stereo_depth.tiff")
                assert os.path.isfile(stereo_depth_path)
                instance_data.append(stereo_depth_path)

                grasps_path = rgb_path.replace("RGB.png", "grasps.txt")
                assert os.path.isfile(grasps_path)
                instance_data.append(grasps_path)

                mask_path = rgb_path.replace("RGB.png", "mask.png")
                assert os.path.isfile(mask_path)
                instance_data.append(mask_path)

                instance_data.append(rgb_path.split("/")[-3])
                output.append(instance_data)
            return output
        else:
            return glob.glob(os.path.join(self.cache_path, "*/*"))
            
    def get_class_map(self, dataset_path: str) -> tuple[dict[str, int], dict[int, str]]:
        all_classes = self.listdir(dataset_path)
        class_to_idx = {c:i for i, c in enumerate(sorted(all_classes))}
        idx_to_class = {v:k for k, v in class_to_idx.items()}
        return class_to_idx, idx_to_class

    def listdir(self, folder_path: str) -> list[str]:
        files = os.listdir(folder_path)
        fn = [i for i in files if ".DS_Store" in i]
        if len(fn) > 0:
            files.remove(fn[0])
        return files