
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from src.lyft.utils import apply_colors

import torch
import torch.utils.data
import numpy as np

from src.argoverse.utils.svg_utils import BaseDataset


class SVGDataset(torch.utils.data.Dataset):
    def __init__(self, model_args, max_num_groups, max_seq_len,
                 data_dict = None, args=None, mode=None,
                 max_total_len=None, PAD_VAL=-1,use_agents=False,add_scene_svg=True,add_history_agent_svg=True,max_num_agents=10):

        super().__init__()
        self.svg = True
        self.svg_cmds = True
        self.data = BaseDataset(data_dict, args, mode,use_agents=use_agents)

        self.MAX_NUM_GROUPS = max_num_groups
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TOTAL_LEN = max_total_len
        self.MAX_NUM_AGENTS = max_num_agents

        if max_total_len is None:
            self.MAX_TOTAL_LEN = max_num_groups * max_seq_len


        self.model_args = model_args

        self.PAD_VAL = PAD_VAL
        self.add_history_agent_svg = add_history_agent_svg
        self.add_scene_svg = add_scene_svg



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get(idx, self.model_args)


    @staticmethod
    def simplify(svg, normalize=True):
        svg.canonicalize(normalize=normalize)
        svg = svg.simplify_heuristic()
        return svg.normalize()

    @staticmethod
    def normalize_history(svg, normalize=True):
        svg.canonicalize(normalize=normalize)
        return svg.normalize()
    
    def get(self, idx=0, model_args=None, random_aug=True, id=None, svg: SVG = None):
        item = self.data[idx]
        if self.svg and self.svg_cmds:
            tens_scene=[]
            tens_path=[]
            if self.add_scene_svg and len(item['path'])!=0:
                tens_scene = self.simplify(SVG.from_tensor(item['path'])).split_paths().to_tensor(concat_groups=False)

            if self.add_history_agent_svg and len(item['history_agent'])!=0:
                tens_path = self.normalize_history(
                    SVG.from_tensor(item['history_agent'])).split_paths().to_tensor(concat_groups=False)

            if self.add_scene_svg and len(item['path_type'])!=0:
                tens_scene = apply_colors(tens_scene, item['path_type'])

            if self.add_history_agent_svg and len(item['history_agent_type'])!=0:
                tens_path = apply_colors(tens_path, item['history_agent_type'])

            MAX_NUM_AGENTS = self.MAX_NUM_AGENTS
            item['agents_availabilty']=np.ones((item['agents_num']),dtype=bool)
            if item['agents_availabilty'].shape[0] < MAX_NUM_AGENTS:
                item['agents_availabilty'] = np.concatenate((item['agents_availabilty'],
                                                       np.zeros((MAX_NUM_AGENTS-item['agents_num']),dtype=bool)))
                
            del item['path']
            del item['path_type']
            del item['history_agent']
            del item['history_agent_type']

            if self.add_history_agent_svg and self.add_scene_svg:
                tens = tens_scene+tens_path
            elif not self.add_history_agent_svg:
                tens = tens_scene
            elif not self.add_scene_svg:
                tens = tens_path
            item['image'] = self.get_data(idx,tens, None, model_args=model_args, label=None)
        if item['image'] is None:
            return None
        return item

    def get_data(self, idx, t_sep, fillings, model_args=None, label=None):
        res = {}
        # max_len_commands = 0
        # len_path = len(t_sep)
        if model_args is None:
            model_args = self.model_args
        if len(t_sep) > self.MAX_NUM_GROUPS:
            return None
        pad_len = max(self.MAX_NUM_GROUPS - len(t_sep), 0)

        t_sep.extend([torch.empty(0, 14)] * pad_len)

        t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=self.PAD_VAL).add_eos().add_sos().pad(
            seq_len=self.MAX_TOTAL_LEN + 2)]
        t_normal = []
        for t in t_sep:
            s = SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL)
            if len(s.commands) > self.MAX_SEQ_LEN:
                return None
            t_normal.append(s.add_eos().add_sos().pad(
                seq_len=self.MAX_SEQ_LEN + 2))

        for arg in set(model_args):
            if "_grouped" in arg:
                arg_ = arg.split("_grouped")[0]
                t_list = t_grouped
            else:
                arg_ = arg
                t_list = t_normal

            if arg_ == "tensor":
                res[arg] = t_list

            if arg_ == "commands":
                res[arg] = torch.stack([t.cmds() for t in t_list])

            if arg_ == "args_rel":
                res[arg] = torch.stack([t.get_relative_args() for t in t_list])
            if arg_ == "args":
                res[arg] = torch.stack([t.args() for t in t_list])

        if "filling" in model_args:
            res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

        if "label" in model_args:
            res["label"] = label
        return res
    
    
    def get_data_history(self, idx, t_sep, fillings, model_args=None, label=None):
            res = {}
            # max_len_commands = 0
            # len_path = len(t_sep)
            if model_args is None:
                model_args = self.model_args
            pad_len = 0

            t_sep.extend([torch.empty(0, 14)] * pad_len)
#             print("t_sep",len(t_sep))

            t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=self.PAD_VAL).add_eos().add_sos()]
            t_normal = []
            for t in t_sep:
                s = SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL)
#                 print(1,len(s.commands))
                j = s.add_eos().add_sos().pad(seq_len=20 + 2)
#                 print(2,len(s.commands))
                t_normal.append(j)

            for arg in set(model_args):
                if "_grouped" in arg:
                    arg_ = arg.split("_grouped")[0]
                    t_list = t_grouped
                else:
                    arg_ = arg
                    t_list = t_normal

                if arg_ == "tensor":
                    res[arg] = t_list

                if arg_ == "commands":
                    res[arg] = torch.stack([t.cmds() for t in t_list])

                if arg_ == "args_rel":
                    res[arg] = torch.stack([t.get_relative_args() for t in t_list])
                if arg_ == "args":
                    res[arg] = torch.stack([t.args() for t in t_list])

            if "filling" in model_args:
                res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

            if "label" in model_args:
                res["label"] = label
            return res

