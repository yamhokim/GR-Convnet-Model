from __future__ import annotations
from dataset.map_based.map_based_jacquard import MapBasedJacquardDataset
from torch.utils.tensorboard.writer import SummaryWriter
from metrics.map_based_iou import map_based_iou
from utils.map_based_utils import cls_from_map
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Callable
from tqdm import tqdm
import torch
import os
import pickle

class MapBasedTrainer:
    def __init__(
            self,
            training_mode: str,
            model: nn.Module,
            device: str,
            loss_fn: Any,
            dataset: MapBasedJacquardDataset,
            optimizer: optim.Optimizer,
            lr: float,
            train_batch_size: int,
            test_split_ratio: float,
            checkpoint_dir: str,
            log_dir: str,
            scheduler: Callable[[float, int], float] = lambda lr, step: lr,
            num_accumulate_batches: int = 1,
            test_batch_size: int = 16,
        ):
        """
        Initializer for MapBasedTrainer;
        
        Arguments:
            - training_mode: expected to be either "grasp" or "cls"
            - model: torch model to be trained
            - loss_fn: a callable loss function class that outputs the loss given predicted and target maps
            - dataset: an instance of MapBasedJacquardDataset (cached gives better performance)
            - optimizer: an uninitialized optimizer (eg. optimizer = torch.optim.Adam)
            - log_dir: directory for saving tensorboard logs
            - scheduler: a functions that takes the current lr and step number/epoch and outputs the next lr
            - num_accumulate_batches: if the batch size is small, we may want to accumulate gradients over multiple batches
                before updating model weights. This argument controls the number of batches we accumulate gradients for.
        """
        self.training_mode = training_mode
        assert self.training_mode == "grasp" or self.training_mode == "cls", "training_mode must be 'grasp' or 'cls'"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), self.lr)
        self.num_accumulate_batches = num_accumulate_batches
        self.scheduler = scheduler
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.train_dataset = dataset
        self.test_split_ratio = test_split_ratio

        self.test_dataset = self.train_dataset.extract_test_dataset(self.test_split_ratio)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.rect_collate)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True, collate_fn=self.rect_collate)
        self.step_number = 1

    def run(self, num_steps: int = 10):
        """
        Run function that runs the appropriate train functions based on whether training_mode was set to grasp or cls.
        """
        if self.training_mode == "grasp":
            self.grasp_run(num_steps)
        elif self.training_mode == "cls":
            self.cls_run(num_steps)

    ################## Functions for saving and loading checkpoints ##################
    @staticmethod
    def load_state(load_path: str) -> MapBasedTrainer:
        """
        Returns a MapBasedTrainer object loaded from the checkpoint file located at load_path.
        """
        f = open(load_path, "rb")
        trainer_obj = pickle.load(f)
        trainer_obj.tb_writer = SummaryWriter(log_dir=trainer_obj.log_dir)
        return trainer_obj
    
    def save_state(self, grasp_or_cls: str, iteration: int, save_loss: float, save_acc: float, decimal_place: int = 6):
        """
        Saves the entire MapBasedTrainer class as a serialized pickle object. This saves both model weights
        and training state (iteration number, optimizer state, current lr, etc.).

        File names are generated according to model type (grasp or cls) and test metrics of most recent test step.
        """
        tb_writer = self.tb_writer
        self.tb_writer = None
        save_name = grasp_or_cls + "_Step_" + str(iteration) + "_Acc_" + str(round(save_acc, decimal_place)) + "_Loss_" + str(round(save_loss, decimal_place)) + ".pth"
        save_path = os.path.join(self.checkpoint_dir, save_name)
        with open(save_path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)
        self.tb_writer = tb_writer


    ################## Grasping functions ##################
    def grasp_train_step(self):
        """
        Runs a single grasp train step (trains model on the entire dataset once)
        """
        self.model = self.model.train()
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)
        loop.set_description(f"Grasp training step {self.step_number}")

        for i, (rgbd_image, target_grasp_maps, _, _) in enumerate(loop):
            rgbd_image, target_grasp_maps = rgbd_image.to(self.device), target_grasp_maps.to(self.device)
            predicted_grasp_maps = self.model(rgbd_image)
            loss = self.loss_fn(predicted_grasp_maps, target_grasp_maps)
            loss.backward()
            if i % self.num_accumulate_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loop.set_postfix(loss = loss.item())

    def grasp_test_step(self):
        """
        Runs a single grasp test step (computes test metrics for model on entire test dataset)
        """
        self.model = self.model.eval()
        loop = tqdm(self.test_loader, total=len(self.test_loader), leave=True, position=0)
        loop.set_description(f"Grasp test step {self.step_number}")

        total_loss = 0
        num_correct = 0
        with torch.no_grad():
            for rgbd_image, target_grasp_maps, target_grasp_rects, _ in loop:
                rgbd_image, target_grasp_maps = rgbd_image.to(self.device), target_grasp_maps.to(self.device)
                predicted_maps = self.model(rgbd_image)
                loss = self.loss_fn(predicted_maps, target_grasp_maps)
                total_loss += loss.item()

                for grasp_map, target_rect in zip(predicted_maps, target_grasp_rects):
                    conf, cos, sin, width, length = grasp_map
                    num_correct += map_based_iou(conf, cos, sin, width, length, target_rect)

        avg_loss = total_loss / len(self.test_loader)
        avg_acc = num_correct / len(self.test_dataset)
        print(f"Average Loss: {avg_loss} | Accuracy: {avg_acc}")
        return avg_loss, avg_acc

    def grasp_run(self, num_steps: int = 10):
        """
        This function does the following;
            - Runs a grasp train step
            - Runs a grasp test step
            - Saves model weights and training state
            - Updates lr based on self.scheduler
        """
        while self.step_number <= num_steps:
            print("-" * 50)
            self.grasp_train_step()
            test_loss, test_acc = self.grasp_test_step()

            self.tb_writer.add_scalar("Grasp Test loss", test_loss, self.step_number)
            self.tb_writer.add_scalar("Grasp Test Accuracy", test_acc, self.step_number)

            self.save_state("Grasp", self.step_number, test_loss, test_acc)
            self.step_number += 1
            self.set_lr(self.scheduler(self.lr, self.step_number))
    

    ################## Classification functions ##################
    def cls_train_step(self):
        """
        Runs a single cls train step (trains model on the entire dataset once)
        """
        self.model = self.model.train()
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)
        loop.set_description(f"Cls training step {self.step_number}")

        for i, (rgbd_image, _, _, cls_maps) in enumerate(loop):
            rgbd_image, cls_maps = rgbd_image.to(self.device), cls_maps.to(self.device)
            predicted_cls_maps = self.model(rgbd_image)
            loss = self.loss_fn(predicted_cls_maps, cls_maps)
            loss.backward()
            if i % self.num_accumulate_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loop.set_postfix(loss = loss.item())

    def cls_test_step(self):
        """
        Runs a single cls test step (computes test metrics on entire test dataset)
        """
        self.model = self.model.eval()
        loop = tqdm(self.test_loader, total=len(self.test_loader), leave=True, position=0)
        loop.set_description(f"Cls test step {self.step_number}")

        total_loss = 0
        num_correct = 0
        with torch.no_grad():
            for rgbd_image, _, _, cls_maps in loop:
                rgbd_image, cls_maps = rgbd_image.to(self.device), cls_maps.to(self.device)
                predicted_cls_maps = self.model(rgbd_image)
                loss = self.loss_fn(predicted_cls_maps, cls_maps)
                total_loss += loss.item()

                predicted_labels = cls_from_map(predicted_cls_maps)
                target_labels = cls_from_map(cls_maps)
                num_correct += (predicted_labels == target_labels).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        avg_acc = num_correct / len(self.test_dataset)
        print(f"Average Loss: {avg_loss} | Accuracy: {avg_acc}")
        return avg_loss, avg_acc
    
    def cls_run(self, num_steps: int = 10):
        """
        This function does the following;
            - Runs a cls train step
            - Runs a cls test step
            - Saves model weights and training state
            - Updates lr based on self.scheduler
        """
        while self.step_number <= num_steps:
            print("-" * 50)
            self.cls_train_step()
            test_loss, test_acc = self.cls_test_step()

            self.tb_writer.add_scalar("Cls Test loss", test_loss, self.step_number)
            self.tb_writer.add_scalar("Cls Test Accuracy", test_acc, self.step_number)

            self.save_state("Cls", self.step_number, test_loss, test_acc)
            self.step_number += 1
            self.set_lr(self.scheduler(self.lr, self.step_number))


    ################## Utility functions ##################
    def set_lr(self, lr: float):
        self.lr = lr
        for p in self.optimizer.param_groups:
            p["lr"] = lr

    def rect_collate(self, batch):
        rgbds = []
        grasp_maps = []
        rects = []
        cls_maps = []
        for a, b, c, d in batch:
            rgbds.append(a.unsqueeze(0))
            grasp_maps.append(b.unsqueeze(0))
            rects.append(c)
            cls_maps.append(d.unsqueeze(0))
        return torch.cat(rgbds, dim=0), torch.cat(grasp_maps, dim=0), rects, torch.cat(cls_maps, dim=0)