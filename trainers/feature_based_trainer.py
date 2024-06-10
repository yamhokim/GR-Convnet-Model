from __future__ import annotations
from dataset.feature_based.feature_based_jacquard import FeatureBasedJacquard
from metrics.iou import max_iou_bool
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Optional, Callable
from tqdm import tqdm
import torch
import os
import pickle
import numpy
import torch.nn.functional as F


class FeatureBasedTrainer:

    def __init__(
            self,
            training_mode: str,
            model: nn.Module,
            loss_fn: Any,
            dataset: FeatureBasedJacquard,
            optimizer: optim.Optimizer,
            lr: float,
            train_batch_size: int,
            device: str,
            checkpoint_dir: str,
            scheduler: Callable[[float, int], float] = lambda lr, step: lr,
            test_batch_size: int = 16,
            test_split_ratio: Optional[float] = None
        ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optimizer(model.parameters(), self.lr)
        self.checkpoint_dir = checkpoint_dir
        self.training_mode = training_mode
        self.scheduler = scheduler
        assert self.training_mode == "grasp" or self.training_mode == "cls", "training_mode must be 'grasp' or 'cls'"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.train_dataset = dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.rect_collate)
        self.test_split_ratio = test_split_ratio
        
        if self.test_split_ratio is not None:
            self.test_dataset = self.train_dataset.extract_test_dataset(self.test_split_ratio)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True, collate_fn=self.rect_collate)

        self.step_number = 1

    @staticmethod
    def load_state(load_path: str) -> FeatureBasedTrainer:
        f = open(load_path, "rb")
        return pickle.load(f)
    
    def set_lr(self, lr: float):
        self.lr = lr
        for p in self.optimizer.param_groups:
            p["lr"] = lr 

    def rect_collate(self, batch):
        rgbds = []
        if self.training_mode == "grasp":
            grasp_arrs = []
            rects = []
        cls_labels = []
        for a, b, c, d in batch:
            rgbds.append(a.unsqueeze(0))
            if self.training_mode == "grasp":
                grasp_arrs.append(b.unsqueeze(0))
                rects.append(c.unsqueeze(0))
            cls_labels.append(d.unsqueeze(0))
    
        if self.training_mode == "grasp":
            return torch.cat(rgbds, dim=0), torch.cat(grasp_arrs, dim=0), torch.cat(rects, dim=0), []
        else:
            return torch.cat(rgbds, dim=0), [], [], torch.cat(cls_labels, dim=0)
    
    def grasp_train_step(self):
        self.model = self.model.train()
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)
        loop.set_description(f"Grasp training step {self.step_number}")

        for rgbd_image, _, target_grasp_rect, _ in loop:
            self.optimizer.zero_grad()
            rgbd_image, target_grasp_rect = rgbd_image.to(self.device), target_grasp_rect.to(self.device)
            target_grasp_rect = target_grasp_rect.clamp(min=0, max=self.train_dataset.image_size -1)

            predicted_grasp_rect = self.train_dataset.compute_grasp_rectangle(self.model(rgbd_image))
            predicted_grasp_rect = predicted_grasp_rect.clamp(min=0, max=self.train_dataset.image_size -1)
            predicted_grasp_rect = predicted_grasp_rect.repeat(target_grasp_rect.shape[1], 1, 1).unsqueeze(0)

            predicted_grasp_rect /= self.train_dataset.image_size
            target_grasp_rect /= self.train_dataset.image_size

            loss = self.loss_fn(predicted_grasp_rect, target_grasp_rect)
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss = loss.item())

    def grasp_test_step(self):
        self.model = self.model.eval()
        loop = tqdm(self.test_loader, total=len(self.test_loader), leave=True, position=0)
        loop.set_description(f"Grasp test step {self.step_number}")

        total_loss = 0
        num_correct = 0
        with torch.no_grad():
            for rgbd_image, _, target_grasp_rect, _ in loop:
                rgbd_image, target_grasp_rect = rgbd_image.to(self.device), target_grasp_rect.to(self.device)
                target = target_grasp_rect.cpu().detach().numpy()
                target_grasp_rect = target_grasp_rect.clamp(min=0, max=self.train_dataset.image_size -1)

                predicted_grasp_rect = self.train_dataset.compute_grasp_rectangle(self.model(rgbd_image))
                pred = predicted_grasp_rect.cpu().detach().numpy()
                predicted_grasp_rect = predicted_grasp_rect.clamp(min=0, max=self.train_dataset.image_size -1)
                predicted_grasp_rect = predicted_grasp_rect.repeat(target_grasp_rect.shape[1], 1, 1).unsqueeze(0)

                predicted_grasp_rect /= self.train_dataset.image_size
                target_grasp_rect /= self.train_dataset.image_size

                loss = self.loss_fn(predicted_grasp_rect, target_grasp_rect)
                total_loss += loss.item()

                predicted_grasp_rect *= self.train_dataset.image_size
                target_grasp_rect *= self.train_dataset.image_size
                
                if max_iou_bool(predicted_rect=pred[0], target_rects=target[0], image_size=self.train_dataset.image_size):
                    num_correct += 1


        avg_loss = total_loss / len(self.test_loader)
        avg_acc = num_correct / len(self.test_dataset)
        print(f"Average Loss: {avg_loss} | Accuracy: {avg_acc}")
        return avg_loss, avg_acc
    
    def save_state(self, grasp_or_cls: str, iteration: int, save_loss: float, save_acc: float, decimal_place: int = 6):
        save_name = grasp_or_cls + "_Step_" + str(iteration) + "_Acc_" + str(round(save_acc, decimal_place)) + "_Loss_" + str(round(save_loss, decimal_place)) + ".pth"
        save_path = os.path.join(self.checkpoint_dir, save_name)
        with open(save_path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    def grasp_run(self, num_steps: int = 10):
        while self.step_number <= num_steps:
            if self.step_number == (num_steps // 2):
                print('Unfreezing pretrained layer weights now...')
                self.model.unfreeze_depth_backbone()

            print("-" * 50)
            self.grasp_train_step()
            test_loss, test_acc = self.grasp_test_step()
            self.save_state("Grasp", self.step_number, test_loss, test_acc)
            self.step_number += 1
            self.set_lr(self.scheduler(self.lr, self.step_number))

    def cls_train_step(self):
        self.model = self.model.train()
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)
        loop.set_description(f"Cls training step {self.step_number}")

        count = 0
        for rgbd_image, _, _, cls_labels in loop:
            self.optimizer.zero_grad()
            rgbd_image, cls_labels = rgbd_image.to(self.device), cls_labels.to(self.device)
            predicted_cls_labels = F.softmax(self.model(rgbd_image), dim=1)
            loss = self.loss_fn(predicted_cls_labels, cls_labels)
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss = loss.item())
            count += 1

    def cls_test_step(self):
        self.model = self.model.eval()
        loop = tqdm(self.test_loader, total=len(self.test_loader), leave=True, position=0)
        loop.set_description(f"Cls test step {self.step_number}")

        total_loss = 0
        num_correct = 0
        with torch.no_grad():
            for rgbd_image, _, _, cls_labels in loop:
                rgbd_image, cls_labels = rgbd_image.to(self.device), cls_labels.to(self.device)
                predicted_cls_labels = F.softmax(self.model(rgbd_image), dim=1)
                loss = self.loss_fn(predicted_cls_labels, cls_labels)
                total_loss += loss.item()

                num_correct += (predicted_cls_labels.argmax(1) == cls_labels.argmax(1)).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        avg_acc = num_correct / len(self.test_dataset)
        print(f"Average Loss: {avg_loss} | Accuracy: {avg_acc}")
        return avg_loss, avg_acc
    
    def cls_run(self, num_steps: int = 10):
        #self.model.unfreeze_depth_backbone()
        while self.step_number <= num_steps:
            if self.step_number == (num_steps // 2):
                print('Unfreezing pretrained layer weights now...')
                self.model.unfreeze_depth_backbone()

            print("-" * 50)
            self.cls_train_step()
            test_loss, test_acc = self.cls_test_step()
            self.save_state("Cls", self.step_number, test_loss, test_acc)
            self.step_number += 1
            self.set_lr(self.scheduler(self.lr, self.step_number))
    
    def run(self, num_steps: int = 10):
        if self.training_mode == "grasp":
            self.grasp_run(num_steps)
        elif self.training_mode == "cls":
            self.cls_run(num_steps)