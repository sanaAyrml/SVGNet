
from deepsvg.utils.utils import _pack_group_batch, _unpack_group_batch, _make_seq_first

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .basic_blocks import ResNet
from .config import _DefaultConfig
from .utils import (_get_padding_mask, _get_key_padding_mask, _get_group_mask, _get_visibility_mask,
                    _get_key_visibility_mask)

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SVGEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len, rel_args=False, use_group=True, group_len=None):
        super().__init__()

        self.cfg = cfg

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = 2 * cfg.args_dim if rel_args else cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

        if self.use_group:
            nn.init.kaiming_normal_(self.group_embed.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        S, GN = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, GN, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src




class LabelEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.label_embedding = nn.Embedding(cfg.n_labels, cfg.dim_label)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.label_embedding.weight, mode="fan_in")

    def forward(self, label):
        src = self.label_embedding(label)
        return src

class TransformerAlbert(nn.Module):
    def __init__(self, encoder_layer, num_layers,  norm=None):
        super(TransformerAlbert, self).__init__()
        self.layers = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,src, memory2=None, mask=None, src_key_padding_mask=None):
        output = src
        for i in range(self.num_layers):
            output = self.layers(output, memory2=memory2, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output
    
class Encoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig,add_mlp_history=0,add_mlp_agent=0, first_transformer_albert = False,second_transformer_albert = False):
        super().__init__()

        self.cfg = cfg
        self.add_mlp_history = add_mlp_history
        self.add_mlp_agent = add_mlp_agent
        self.first_transformer_albert = first_transformer_albert
        self.second_transformer_albert = second_transformer_albert
        if self.add_mlp_history>0:
            self.history_residual = ResNet(40)
            self.history_block = nn.Linear(40, 256)
        if self.add_mlp_agent>0:
            self.agent_residual = ResNet(40)
            self.agent_block = nn.Linear(40, 256)

        seq_len = cfg.max_seq_len if cfg.encode_stages == 2 else cfg.max_total_len
        self.use_group = cfg.encode_stages == 1
        self.embedding = SVGEmbedding(cfg, seq_len, use_group=self.use_group)
        if cfg.label_condition:
            self.label_embedding = LabelEmbedding(cfg)
        dim_label = cfg.dim_label if cfg.label_condition else None
        if cfg.model_type == "transformer":
            encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout,
                                                            d_global2=dim_label)
            encoder_norm = LayerNorm(cfg.d_model)
            if not self.first_transformer_albert:
                self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)
            else:
                self.encoder = TransformerAlbert(encoder_layer, cfg.n_layers, encoder_norm)
        else:  # "lstm"
            self.encoder = nn.LSTM(cfg.d_model, cfg.d_model // 2, dropout=cfg.dropout, bidirectional=True)
        if cfg.encode_stages == 2:
            if not cfg.self_match:
                self.hierarchical_PE = PositionalEncodingLUT(cfg.d_model,                                                      max_len=cfg.max_num_groups+self.add_mlp_history+self.add_mlp_agent)

            hierarchical_encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward,
                                                                         cfg.dropout, d_global2=dim_label)
            hierarchical_encoder_norm = LayerNorm(cfg.d_model)
            if not self.second_transformer_albert:
                self.hierarchical_encoder = TransformerEncoder(hierarchical_encoder_layer, cfg.n_layers,
                                                               hierarchical_encoder_norm)
            else:
                self.hierarchical_encoder = TransformerAlbert(encoder_layer = hierarchical_encoder_layer, num_layers =cfg.n_layers,
                                                              norm = hierarchical_encoder_norm)

    def forward(self, commands, args,history,agents,agents_validity,label=None):
        S, G, N = commands.shape

        l = self.label_embedding(label).unsqueeze(0).unsqueeze(0).repeat(1, commands.size(1), 1,
                                                                         1) if self.cfg.label_condition else None

        if self.cfg.encode_stages == 2:
            modified = False
            if self.add_mlp_history>0 or self.add_mlp_agent>0:
                modified = True
            visibility_mask, key_visibility_mask = _get_visibility_mask(commands, seq_dim=0,
                                                                        modified=modified,agents_validity= agents_validity), _get_key_visibility_mask(
                commands, seq_dim=0, modified=modified,agents_validity= agents_validity)
        commands, args, l = _pack_group_batch(commands, args, l)
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands,
                                                                                                       seq_dim=0)

        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        src = self.embedding(commands, args, group_mask)

        if self.cfg.model_type == "transformer":
            memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask, memory2=l)
            z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True)
        else:  # "lstm"
            hidden_cell = (src.new_zeros(2, N, self.cfg.d_model // 2),
                           src.new_zeros(2, N, self.cfg.d_model // 2))
            sequence_lengths = padding_mask.sum(dim=0).squeeze(-1)
            x = pack_padded_sequence(src, sequence_lengths, enforce_sorted=False)

            packed_output, _ = self.encoder(x, hidden_cell)

            memory, _ = pad_packed_sequence(packed_output)
            idx = (sequence_lengths - 1).long().view(1, -1, 1).repeat(1, 1, self.cfg.d_model)
            z = memory.gather(dim=0, index=idx)

        z = _unpack_group_batch(N, z)
        if self.add_mlp_history>0:
            h = self.history_block(self.history_residual(torch.flatten(history, start_dim=1))).unsqueeze(0).unsqueeze(0)
            z = torch.cat((z, h), dim=1)
        if self.add_mlp_agent>0:
            agents=agents.permute(1,0,2,3)
            for agent in agents:
                a = self.agent_block(self.agent_residual(torch.flatten(agent.type(torch.cuda.FloatTensor),
                                                                       start_dim=1))).unsqueeze(0).unsqueeze(0)
                z = torch.cat((z, a), dim=1)

        if self.cfg.encode_stages == 2:
            src = z.transpose(0, 1)
            src = _pack_group_batch(src)
            l = self.label_embedding(label).unsqueeze(0) if self.cfg.label_condition else None
            if not self.cfg.self_match:
                src = self.hierarchical_PE(src)
            memory = self.hierarchical_encoder(src, mask=None, src_key_padding_mask=key_visibility_mask, memory2=l)
            z = (memory * visibility_mask).sum(dim=0, keepdim=True) / visibility_mask.sum(dim=0, keepdim=True)
            z = _unpack_group_batch(N, z)

        return z



class Bottleneck(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Linear(cfg.d_model , cfg.dim_z)

    def forward(self, z):
        return self.bottleneck(z)



class SVGTransformer(nn.Module):
    def __init__(self, cfg: _DefaultConfig,add_mlp_history=0,add_mlp_agent=0, first_transformer_albert = False,second_transformer_albert = False):
        super(SVGTransformer, self).__init__()

        self.cfg = cfg
        self.args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1

        if self.cfg.encode_stages > 0:

            self.encoder = Encoder(cfg,add_mlp_history,add_mlp_agent,first_transformer_albert,second_transformer_albert)
            if cfg.use_resnet:
                self.resnet = ResNet(cfg.d_model)

            self.bottleneck = Bottleneck(cfg)

        self.history_mlp = torch.nn.Sequential(torch.nn.Linear(40, 128),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(128, 96),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(96, 64), )
        for i in range(0, len(self.history_mlp), 2):
            nn.init.kaiming_normal_(self.history_mlp[i].weight, mode="fan_in")


    def forward(self, commands_enc, args_enc,  history,agents,agents_validity, label=None,
                z=None, encode_mode=True, params= None):
        commands_enc, args_enc = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        h2 = history.permute(1, 0, 2).unsqueeze(dim=1)

        if z is None:
            z = self.encoder(commands_enc, args_enc, history, agents, agents_validity, label).squeeze(0).squeeze(0)

            if self.cfg.use_resnet:
                z = self.resnet(z)

            if self.cfg.use_vae:
                z, mu, logsigma = self.vae(z)
            else:
                z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        if encode_mode: return z

        return z



