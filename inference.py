from torch.utils.data import DataLoader, Subset

import importlib
from src.model_and_dataset.models.model_trajectory import ModelTrajectory
import math
import copy
from collections import OrderedDict
from tqdm import tqdm

cfg = importlib.import_module("configs.deepsvg.our_config").Config()

from src.model_and_dataset.svg_dataset import SVGDataset


import os
from torch.utils.data.dataloader import default_collate

import argparse

import torch
import src.argoverse.utils.baseline_utils as baseline_utils
from src.argoverse.utils.transform import *

CV2_SHIFT = 8  # how many bits to shift in drawing
from argoverse.evaluation.competition_util import generate_forecasting_h5
from src.model_and_dataset.models.mlp_added_transformer import MLPTransformer



args = argparse.Namespace()
args.end_epoch=5000
args.joblib_batch_size=100
args.lr=0.001
args.model_path=None
args.normalize=True
args.obs_len=20
args.pred_len=30
args.test=False
args.test_batch_size=512
args.test_features='/work/vita/sadegh/argo/argoverse-forecasting/forecasting_features_test.pkl'
args.train_batch_size=512
args.train_features='/work/vita/sadegh/argo/argoverse-forecasting/forecasting_features_val.pkl'
args.traj_save_path=None
args.use_delta=False
args.use_map=False
args.use_social=False
args.val_batch_size=512
args.val_features='/work/vita/sadegh/argo/argoverse-forecasting/forecasting_features_val.pkl'

# key for getting feature set
    # Get features
if args.use_map and args.use_social:
    baseline_key = "map_social"
elif args.use_map:
    baseline_key = "map"
elif args.use_social:
    baseline_key = "social"
else:
    baseline_key = "none"



def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    # print(batch)
    # batch = filter(lambda x:x is not None, batch)
    batch = list(filter(None, batch))
    # print(batch)
    if len(batch) > 0:
        return default_collate(batch)
    else:
        return


    
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
    print("using cuda")
else:
    device = torch.device("cpu")
    print("using cpu")

# Get data
data_dict = baseline_utils.get_data(args, baseline_key)

model_path="/work/vita/ayromlou/argo_code/logs/test_modes_1/checkpoint-274500"

model = MLPTransformer(model_config=cfg, data_config= None,
                       modes=1,history_num = 20,future_len=30,
                       add_mlp_history=cfg.add_mlp_history,add_mlp_agent=cfg.add_mlp_agent,
                       first_transformer_albert=False,
                       second_transformer_albert=False).to(device)
            

checkpoint = torch.load(model_path,map_location=device)
print(model_path)

new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace("module.", "") # remove `module.`
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
# model = nn.DataParallel(model)

print(cfg.max_num_groups)
dataset = SVGDataset(model_args=cfg.model_args,
                          max_num_groups=cfg.max_num_groups,max_seq_len=cfg.max_seq_len,
                          data_dict=data_dict, args=args, mode="test",use_agents=cfg.use_agents,
                          add_scene_svg=cfg.add_scene_svg,add_history_agent_svg=cfg.add_history_agent_svg,max_num_agents=cfg.max_num_agents)
# print(test_dataset[0])
print(len(dataset))
bs=16

dataloader = DataLoader(dataset, batch_size=bs, shuffle=False,
                        num_workers=40,collate_fn=my_collate)


forecasted_trajectories={}
nums=0

        
progress_bar = tqdm(dataloader)
    
for data in progress_bar:
    model.eval()
    with torch.no_grad():
#         print(data)
        ego_yaw=(-math.pi*data["yaw_deg"]/180).numpy()
        centroids=data["centroid"].numpy()
        
        seq_id=data["seq_id"].numpy()
        model_args = [data["image"][arg].to(device) for arg in cfg.model_args]
        history=data["history_positions"].to(device)

        if cfg.add_mlp_agent>0:
            agents=data["normal_agents_history"].to(device)
            agents_availabilty=data["agents_availabilty"].to(device)
            entery = [[*model_args,history,agents,agents_availabilty, {}, True],history]
        else:
            entery = [[*model_args,history,None,None, {}, True],history]

            
        output,conf = model(entery)
        
        output=output.reshape((data["history_positions"].shape[0],1,30,2)).cpu()
        for i in range(output.shape[0]):
            saved = []
            for j in range(output.shape[1]):
                rot_output=transform_points(output[i][j], yaw_as_rotation33(ego_yaw[i]))+centroids[i]
                saved.append(rot_output)
            forecasted_trajectories[seq_id[i]] = saved

print("here")
output_path = "/work/vita/ayromlou/argo_code/new_multi_modal_code_sadegh/results1"

if os.path.exists(output_path):
    os.rmdir(output_path)
os.mkdir(output_path)

generate_forecasting_h5(forecasted_trajectories, output_path) #this might take awhile
