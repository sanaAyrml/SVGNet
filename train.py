from torch.utils.data import DataLoader, Subset
from deepsvg.config import _Config

import importlib
from src.model_and_dataset.svg_dataset import SVGDataset
from src.model_and_dataset.models.mlp_added_transformer import MLPTransformer
from src.model_and_dataset.utils import Loss
from deepsvg.utils import Stats, TrainVars, Timer
from deepsvg import utils
from datetime import datetime
from tensorboardX import SummaryWriter
from deepsvg.utils.stats import SmoothedValue
import os
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
import argparse
import torch
import torch.nn as nn

import src.argoverse.utils.baseline_utils as baseline_utils
from src.argoverse.utils.evaluation import get_ade,get_ade_6



def my_collate(batch):
    #     "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(None, batch))
    if len(batch) > 0:
        return default_collate(batch)
    else:
        return


def train(model_cfg:_Config, args, model_name, experiment_name="",log_dir="./logs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_map and args.use_social:
        baseline_key = "map_social"
    elif args.use_map:
        baseline_key = "map"
    elif args.use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"
    import numpy as np

    data_dict = baseline_utils.get_data(args, baseline_key)


    # # Get PyTorch Dataset
    train_dataset = SVGDataset(model_args=model_cfg.model_args,
                               max_num_groups=model_cfg.max_num_groups,max_seq_len=model_cfg.max_seq_len,
                               data_dict=data_dict, args=args, mode="train",use_agents=model_cfg.use_agents,
                               add_scene_svg=model_cfg.add_scene_svg,add_history_agent_svg=model_cfg.add_history_agent_svg,max_num_agents=model_cfg.max_num_agents)

    val_dataset = SVGDataset(model_args=model_cfg.model_args,
                             max_num_groups=model_cfg.max_num_groups,max_seq_len=model_cfg.max_seq_len,
                             data_dict=data_dict, args=args, mode="val",use_agents=model_cfg.use_agents,
                             add_scene_svg=model_cfg.add_scene_svg,add_history_agent_svg=model_cfg.add_history_agent_svg,max_num_agents=model_cfg.max_num_agents)
    criterion= get_ade
    old_loss_criterion= nn.MSELoss()
    if args.modes > 1:
        criterion= Loss()
        old_loss_criterion= get_ade_6

    model = MLPTransformer(model_config=model_cfg,data_config= None,
                           modes=args.modes,history_num = 20,future_len=30,
                           add_mlp_history=model_cfg.add_mlp_history,add_mlp_agent=model_cfg.add_mlp_agent,
                           first_transformer_albert=False,
                           second_transformer_albert=False).to(device)




    train_dataloader = DataLoader(train_dataset, batch_size=model_cfg.train_batch_size, shuffle=True,
                                  num_workers=model_cfg.loader_num_workers,collate_fn=my_collate)
    validat_dataloader = DataLoader(val_dataset, batch_size=model_cfg.val_batch_size, shuffle=True,
                                    num_workers=model_cfg.loader_num_workers,collate_fn=my_collate)

    stats = Stats(num_steps=model_cfg.num_steps, num_epochs=model_cfg.num_epochs, steps_per_epoch=len(train_dataloader),
                  stats_to_print=model_cfg.stats_to_print)
    stats.stats['val'] = defaultdict(SmoothedValue)
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)

    # Summary Writer
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    experiment_identifier = f"{model_name}"
    summary_writer = SummaryWriter(os.path.join(log_dir, experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", model_name, experiment_name)


    # Optimizer, lr & warmup schedulers
    optimizers = model_cfg.make_optimizers(model)
    scheduler_lrs = model_cfg.make_schedulers(optimizers, epoch_size=len(train_dataloader))
    scheduler_warmups = model_cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in model_cfg.make_losses()]


    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    epoch_range = utils.infinite_range(stats.epoch) if model_cfg.num_epochs is None else range(stats.epoch, cfg.num_epochs)
    timer.reset()
    for epoch in epoch_range:
        print(f"Epoch {epoch+1}")
        for n_iter, data in enumerate(train_dataloader):
            if data is None:
                continue
            step = n_iter + epoch * len(train_dataloader)

            if model_cfg.num_steps is not None and step > model_cfg.num_steps:
                return

            model.train()
            model_args = [data['image'][arg].to(device) for arg in model_cfg.model_args]
            params_dict, weights_dict = model_cfg.get_params(step, epoch), model_cfg.get_weights(step, epoch)

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, model_cfg.optimizer_starts), 1):
                optimizer.zero_grad()

                history=data["history_positions"].to(device)
                if model_cfg.add_mlp_agent>0:
                    agents=data["normal_agents_history"].to(device)
                    agents_availabilty=data["agents_availabilty"].to(device)
                    entery = [[*model_args,history,agents,agents_availabilty, {}, True],history]
                else:
                    entery = [[*model_args,history,None,None, {}, True],history]
                output,conf = model(entery)
                loss_dict = {}
                #                 print(data['target_positions'].shape,output.shape,conf.shape)
                if args.modes == 1:
                    loss_dict['loss'] = criterion(data['target_positions'].to(device), output.reshape(data['target_positions'].shape),).mean()
                    loss_dict['min_ade'] = loss_dict['loss']
                    loss_dict['old_loss'] = old_loss_criterion(data['target_positions'].to(device),
                                                               output.reshape(data['target_positions'].shape),).mean()
                else:
                    out = dict()
                    data_loss = dict()
                    out["cls"] = conf
                    out["reg"] = output
                    data_loss["gt_preds"] = data['target_positions'].to(device)
                    data_loss["has_preds"] =  torch.ones(output.shape[0],output.shape[2]).long().to(device)
                    loss_dict['loss'] = criterion(out,data_loss)["loss"]
                    loss_dict['min_ade'] = old_loss_criterion(output,data['target_positions'].to(device)).mean()


                if step >= optimizer_start:
                    loss_dict['loss'].backward()
                    if model_cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip)

                    optimizer.step()
                    if scheduler_lr is not None:
                        scheduler_lr.step()
                    if scheduler_warmup is not None:
                        scheduler_warmup.step()

                stats.update_stats_to_print("train", loss_dict)
                stats.update("train", step, epoch, {
                    ("lr" if i == 1 else f"lr_{i}"): optimizer.param_groups[0]['lr'],
                    **loss_dict
                })

            if step % model_cfg.log_every == 0 and step!=0:
                print("log train")
                stats.update("train", step, epoch, {
                    **weights_dict,
                    "time": timer.get_elapsed_time()
                })
                print(stats.get_summary("train"))
                stats.write_tensorboard(summary_writer, "train")
                summary_writer.flush()

            if step % model_cfg.val_every == 0 :
                print("log val")
                timer.reset()
                torch.save(model.state_dict(),log_dir+"/"+experiment_identifier+"/"+"checkpoint"+"-"+str(step))
                validation_train(validat_dataloader, model, model_cfg, device, criterion, epoch, stats, summary_writer, timer,step,
                                 old_loss_criterion,optimizers,loss_fns,scheduler_lrs,scheduler_warmups)


def validation_train(val_dataloader,model,model_cfg,device, criterion, epoch,stats,summary_writer,timer,train_step,old_loss_criterion,optimizers,loss_fns,scheduler_lrs,scheduler_warmups):
    for n_iter, data in enumerate(val_dataloader):
        if data is None:
            continue
        model.train()
        model_args = [data['image'][arg].to(device) for arg in model_cfg.model_args]
        params_dict, weights_dict = model_cfg.get_params(train_step, epoch), model_cfg.get_weights(train_step, epoch)
        step = n_iter

        for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, model_cfg.optimizer_starts), 1):

            optimizer.zero_grad()

            if model_cfg.val_num_steps is not None and step > model_cfg.val_num_steps:
                stats.update("val", train_step, epoch, {
                    **weights_dict,
                    "time": timer.get_elapsed_time()
                })
                print(stats.get_summary("val"))
                stats.write_tensorboard(summary_writer, "val")
                summary_writer.flush()
                return

            history=data["history_positions"].to(device)
            if model_cfg.add_mlp_agent>0:
                agents=data["normal_agents_history"].to(device)
                agents_availabilty=data["agents_availabilty"].to(device)
                entery = [[*model_args,history,agents,agents_availabilty, {}, True],history]
            else:
                entery = [[*model_args,history,None,None, {}, True],history]

            output,conf = model(entery)
            loss_dict = {}
            if args.modes == 1:
                loss_dict['loss'] = criterion(data['target_positions'].to(device),
                                              output.reshape(data['target_positions'].shape),).mean()
                loss_dict['min_ade'] = loss_dict['loss']
                loss_dict['old_loss'] = old_loss_criterion(data['target_positions'].to(device),
                                                           output.reshape(data['target_positions'].shape),).mean()
            else:

                out = dict()
                data_loss = dict()
                out["cls"] = conf
                out["reg"] = output
                data_loss["gt_preds"] = data['target_positions'].to(device)
                data_loss["has_preds"] =  torch.ones(output.shape[0],output.shape[2]).long().to(device)
                loss_dict['loss'] = criterion(out,data_loss)["loss"]
                loss_dict['min_ade'] = old_loss_criterion(output,data['target_positions'].to(device)).mean()

            if step >= optimizer_start:
                loss_dict['loss'].backward()
                if model_cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip)

                optimizer.step()
                if scheduler_lr is not None:
                    scheduler_lr.step()
                if scheduler_warmup is not None:
                    scheduler_warmup.step()

            stats.update_stats_to_print("val", loss_dict)

            stats.update("val", train_step, epoch, {
                **loss_dict
            })




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--modes", type=int, default=3)
    #lyft
    parser.add_argument("--config-data", type=str, required=False)
    parser.add_argument("--val-idxs", type=str, default=None)
    parser.add_argument("--train-idxs", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=False)
    #argo
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )

    args = parser.parse_args()

    cfg = importlib.import_module(args.config_module).Config()
    model_name, experiment_name = args.config_module.split(".")[-2:]
    print(model_name,experiment_name)
    if args.val_idxs is not None:
        cfg.val_idxs = args.val_idxs
    if args.train_idxs is not None:
        cfg.train_idxs = args.train_idxs



    ####best model
    train(model_cfg=cfg, args=args,
          model_name='test_modes_1_new_again_3',experiment_name="test",
          log_dir="/work/vita/ayromlou/argo_code/logs")

