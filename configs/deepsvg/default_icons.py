from deepsvg.config import _Config
from deepsvg.model.loss import SVGLoss
from deepsvg.model.config import *
from deepsvg.svglib.svg import SVG
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import make_grid
from deepsvg.svglib.geom import Bbox
from deepsvg.utils.utils import batchify, linear

import torchvision.transforms.functional as TF
import torch.optim.lr_scheduler as lr_scheduler
import random


class ModelConfig(Hierarchical):
    """
    Overriding default model_and_dataset config.
    """
    def __init__(self):
        super().__init__()


class Config(_Config):
    """
    Overriding default training config.
    """
    def __init__(self, num_gpus=1):
        super().__init__(num_gpus=num_gpus)

        # Model
        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        # Dataset
        self.filter_category = None

        self.train_ratio = 1.0

        self.max_num_groups = 500
        self.max_total_len = 15000

        # Dataloader
        self.loader_num_workers = 4 * num_gpus

        # Training
        self.num_epochs = 50
        self.val_every = 1000

        # Optimization
        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 60 * num_gpus
        self.grad_clip = 1.0

        self.stats_to_print = {
            "train": ["lr", "time"],
            "val": ["lr", "time"]
        }

    def make_schedulers(self, optimizers, epoch_size):
        optimizer, = optimizers
        return [lr_scheduler.StepLR(optimizer, step_size=3 * epoch_size, gamma=0.6)]


    def make_losses(self):
        return [SVGLoss(self.model_cfg)]

    def get_weights(self, step, epoch):
        return {
            "kl_tolerance": 0.1,
            "loss_kl_weight": linear(0, 10, step, 0, 10000),
            "loss_hierarch_weight": 1.0,
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
            "loss_visibility_weight": 1.0
        }

    def set_train_vars(self, train_vars, dataloader):
        train_vars.x_inputs_train = [dataloader.dataset.get(idx, [*self.model_args, "tensor_grouped"])
                                     for idx in random.sample(range(len(dataloader.dataset)), k=10)]

