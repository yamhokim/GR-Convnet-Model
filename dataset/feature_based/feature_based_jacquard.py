from ..jacquard_grasp_cls import JacquardGraspCLSDataset
from typing import Optional
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from tqdm import tqdm
import random
import math

class FeatureBasedJacquard(JacquardGraspCLSDataset):

    def __init__(
        self, 
        image_size: int, 
        precision: torch.dtype,
        dataset_path: Optional[str] = None, 
        cache_path: Optional[str] = None,
        random_augment: bool = True,
        width_scale_factor: int = 1
    ) -> None:
        super().__init__(image_size, dataset_path, cache_path, random_augment)
        self.width_scale_factor = width_scale_factor
        self.precision = precision

    ###### Image and depth map loading functions
    def load_rgbd_image(self, rgb_image_path: str, depth_image_path: str) -> torch.Tensor:
        T = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        rgb_image = T(Image.open(rgb_image_path))[:3]
        depth_image = T(Image.open(depth_image_path))
        return torch.cat([rgb_image, depth_image], dim=0)
    
    def preprocess_rbgd_image(self, rgbd_image: torch.Tensor) -> torch.Tensor:
        ## Add preprocessing steps here (normalization, etc..)
        mean, std = rgbd_image.mean([1,2]), rgbd_image.std([1,2])
        normalize = transforms.Normalize(mean, std)
        return rgbd_image

    ###### Grasp label loading functions
    def load_grasp_file(self, grasp_path: str) -> np.ndarray:
        """
        Returns in the order [y, x, angle, length, width]
        """
        grasps = []
        with open(grasp_path, "r") as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                grasps.append([x, y, -theta / 180.0 * np.pi, w * self.width_scale_factor, h])
        grasps = np.array(grasps)

        # rescaling values based on image size
        grasps[:, :2] *= (self.image_size / 1024)
        grasps[:, 3:] *= (self.image_size / 1024)
        return grasps
    
    def compute_grasp_rectangle(self, grasp_arr: torch.Tensor) -> torch.Tensor:
        """
        Converts the jacquard dataset grasps into grasp rectangles 
        (returns coordinates of 4 corners of each rectangle)
        """
        x, y, angle, length, width = grasp_arr.transpose(0,1)
        xo = torch.cos(angle)
        yo = torch.sin(angle)

        ya = y + length / 2 * yo
        xa = x - length / 2 * xo
        yb = y - length / 2 * yo
        xb = x + length / 2 * xo

        y1, x1 = ya - width / 2 * xo, xa - width / 2 * yo
        y2, x2 = yb - width / 2 * xo, xb - width / 2 * yo
        y3, x3 = yb + width / 2 * xo, xb + width / 2 * yo
        y4, x4 = ya + width / 2 * xo, xa + width / 2 * yo

        # p1 = np.stack([y1, x1], 1)
        # p2 = np.stack([y2, x2], 1)
        # p3 = np.stack([y3, x3], 1)
        # p4 = np.stack([y4, x4], 1)

        p1 = torch.stack([x1, y1], 1)
        p2 = torch.stack([x2, y2], 1)
        p3 = torch.stack([x3, y3], 1)
        p4 = torch.stack([x4, y4], 1)

        output_arr = torch.stack([p1, p2, p3, p4], 1)
        return output_arr
    
    def rotate_augment(
            self,
            rotation_angle: int, 
            rgbd_image: torch.Tensor, 
            grasp_arr: torch.Tensor,
            grasp_rect: torch.Tensor,
            rotate: Optional[bool] = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.random_augment or rotate:
            grasp_arr[:,2] += math.radians(rotation_angle)
            grasp_arr[:,2] %= (2 * math.pi)

            if rotation_angle == 270:
                rgbd_image = rgbd_image.transpose(-1, -2)
                grasp_rect = grasp_rect.flip(-1)
            elif rotation_angle == 180:
                rgbd_image = rgbd_image.flip(-2)
                grasp_rect[:, :, 1] = self.image_size - grasp_rect[:, :, 1]
            elif rotation_angle == 90:
                rgbd_image = rgbd_image.flip(-2).transpose(-1, -2)
                grasp_rect = grasp_rect.flip(-1)
                grasp_rect[:, :, 0] = self.image_size - grasp_rect[:, :, 0]
            elif rotation_angle == 0:
                pass
            else:
                raise Exception("Invalid rotation angle")
        return rgbd_image, grasp_arr, grasp_rect
    
    def get_random_rotation_angle(self):
        return random.choice([0, 90, 180, 270])
    
    def cast_to_fp_precision(self, rgbd, grasp_arr, grasp_rect, cls_label):
        rgbd = rgbd.to(self.precision)
        grasp_arr = grasp_arr.to(self.precision)
        grasp_rect = grasp_rect.to(self.precision)
        cls_label = cls_label.to(self.precision)
        return rgbd, grasp_arr, grasp_rect, cls_label

    def get_instance_from_dataset(self, file_paths: list[str], random_augment: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_fp, perfect_depth_fp, stereo_depth_fp, grasp_fp, mask_fp, class_label = file_paths

        rgbd_image = self.load_rgbd_image(rgb_fp, perfect_depth_fp)
        rgbd_image = self.preprocess_rbgd_image(rgbd_image)

        cls_label = [0.0,0.0,0.0,0.0,0.0]
        cls_label[self.class_to_idx[class_label]] = 1.0
        cls_label = torch.from_numpy(np.array(cls_label))

        grasp_arr = torch.from_numpy(self.load_grasp_file(grasp_fp))
        grasp_rect_arr = self.compute_grasp_rectangle(grasp_arr)

        rgbd_image, grasp_arr, grasp_rect = self.rotate_augment(self.get_random_rotation_angle(), rgbd_image, grasp_arr, grasp_rect_arr)
        rgbd_image, grasp_arr, grasp_rect, cls_label = self.cast_to_fp_precision(rgbd_image, grasp_arr, grasp_rect, cls_label)

        return rgbd_image, grasp_arr, grasp_rect, cls_label
    
    def get_instance_from_cache(self, file_paths: str, random_augment: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgbd, grasp_arr, grasp_rect, cls_label = np.load(file_paths).values()
        rgbd, grasp_arr, grasp_rect, cls_label = torch.from_numpy(rgbd), torch.from_numpy(grasp_arr), torch.from_numpy(grasp_rect), torch.from_numpy(cls_label)
        rgbd, grasp_arr, grasp_rect = self.rotate_augment(self.get_random_rotation_angle(), rgbd, grasp_arr, grasp_rect)
        rgbd, grasp_arr, grasp_rect, cls_label = self.cast_to_fp_precision(rgbd, grasp_arr, grasp_rect, cls_label)
        return rgbd, grasp_arr, grasp_rect, cls_label
    
    def visualize_instance(self, file_paths: list[str]) -> None:
        if self.dataset_path is not None:
            rgbd_image, (conf, cos, sin, width), grasp_rect,  cls_labels = self.get_instance_from_dataset(file_paths)
        else:
            rgbd_image, (conf, cos, sin, width), grasp_rect, cls_labels = self.get_instance_from_cache(file_paths)
        
        fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
        ax1 = ax1.flatten()

        ax1[0].imshow(rgbd_image[:-1].permute(1, 2, 0))
        ax1[0].axis(False)
        ax1[0].set_title("RGB Image")

        ax1[1].imshow(rgbd_image[-1])
        ax1[1].axis(False)
        ax1[1].set_title("Depth Map")

        ax1[2].imshow(rgbd_image[:-1].permute(1, 2, 0))
        for rect in grasp_rect:
            rect = Polygon(rect, linewidth=1, edgecolor="r", facecolor="none")
            ax1[2].add_patch(rect)
        ax1[2].axis(False)
        ax1[2].set_title("Bounding boxes")
        plt.show()

        fig2, ax2 = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
        ax2 = ax2.flatten()
        for ax, img, name in zip(ax2, (conf, cos, sin, width), ("Confidence map", "cos", "sin", "width")):
            im = ax.imshow(img)
            ax.axis(False)
            ax.set_title(name)
        plt.show()

        fig3, ax3 = plt.subplots(nrows=1, ncols=len(self.class_to_idx), figsize=(20, 6))
        ax3 = ax3.flatten()
        for ax, img, idx in zip(ax3, cls_labels, range(len(self.class_to_idx))):
            im = ax.imshow(img + 1)
            ax.axis(False)
            ax.set_title(self.idx_to_class[idx])
        plt.show()

    def cache_dataset(self, cache_location: str):
        os.mkdir(cache_location)
        for class_name in self.class_to_idx.keys():
            class_path = os.path.join(cache_location, class_name)
            os.mkdir(class_path)

        class_item_indices = {cn: 0 for cn in self.class_to_idx.keys()}
        print("Creating cache...")
        loop = tqdm(range(len(self)))
        self.random_augment = False
        for i in loop:
            rgbd, grasp_arr, grasp_rect_arr, cls_label = self[i]
            class_name = self.individual_file_paths[i][-1]
            save_dir = os.path.join(cache_location, class_name, class_name + "_" + str(class_item_indices[class_name]))
            class_item_indices[class_name] += 1

            rgbd, grasp_arr, cls_label = rgbd.numpy(), grasp_arr.numpy(), cls_label.numpy()
            np.savez_compressed(save_dir, rgbd=rgbd, grasp_arr=grasp_arr, grasp_rect=grasp_rect_arr, cls_label=cls_label)