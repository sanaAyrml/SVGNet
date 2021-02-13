import torch
from deepsvg.model.modified_model_between_added_agents import SVGTransformer



class ModelTrajectory(torch.nn.Module):
    def __init__(self,model_cfg,dim_z=128,mlp=None,add_mlp_history=0,add_mlp_agent=0, first_transformer_albert = False,second_transformer_albert = False):
        super().__init__()
        self.model_cfg = model_cfg
        self.model_cfg.model_cfg.dim_z = dim_z
        self.model_cfg.model_cfg.max_num_groups = self.model_cfg.max_num_groups
        self.model_cfg.model_cfg.max_seq_len = self.model_cfg.max_seq_len

        self.model = SVGTransformer(self.model_cfg.model_cfg,add_mlp_history=add_mlp_history,add_mlp_agent=add_mlp_agent,
                                    first_transformer_albert=first_transformer_albert,second_transformer_albert=second_transformer_albert)


    def forward(self,x):
        commands_enc,args_enc, commands_dec, args_dec,history,agents,agents_validity,params, encode_mode = x
        return self.model(commands_enc = commands_enc, args_enc =args_enc,history=history,
                          agents= agents,agents_validity=agents_validity, params=params,encode_mode=encode_mode)
