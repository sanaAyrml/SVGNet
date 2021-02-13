from .default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=2)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None
        self.train_ratio = 1.0
        
        self.max_num_groups = 120
        self.max_seq_len = 20
        self.max_total_len = 2400



        # Dataloader
        self.loader_num_workers = 20

        # Training
        self.num_epochs = 100

        # Optimization
        self.learning_rate = 1e-4
        self.train_batch_size = 32
        self.val_batch_size = 4

        self.val_every = 1500
        self.log_every = 100
        self.ckpt_every = 100

        self.val_num_steps =100
        self.stats_to_print = {
            "train": ["lr", "time"],
            "val": ["time"]
        }

        self.val_idxs = None
        self.train_idxs = None
        self.step_size = 2.5
        self.gamma = 0.9

        #new
        self.use_agents=True
        self.max_num_agents=15
        self.add_scene_svg=True
        self.add_history_agent_svg=True
        self.add_mlp_history=1
        self.add_mlp_agent=15


        
    def make_schedulers(self, optimizers, epoch_size):
        optimizer, = optimizers
        return [lr_scheduler.StepLR(optimizer, step_size=self.step_size * epoch_size, gamma=self.gamma)]