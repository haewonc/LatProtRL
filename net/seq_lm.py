import esm
import torch 
import torch.nn as nn 
from esm import ESM2
    
class VED(nn.Module):
    def __init__(self, cfg, pretrained=None, esm_pretrained=None):
        super().__init__()
        assert pretrained == None or esm_pretrained == None, 'Both VED checkpoint and ESM-2 checkpoint are given'
        
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = cfg.device
        self.esm_num_layers = cfg.num_layers
        self.num_tokens = cfg.num_tokens
        self.cfg = cfg

        self.encoder = ESM2(num_layers=cfg.num_layers, embed_dim=cfg.embed_dim, attention_heads=20, alphabet=self.alphabet, token_dropout=False)
        self.encoder.load_state_dict(self.load_esm_ckpt('ckpt/esm2_t33_650M_UR50D.pt'), strict=False)
        self.encoder.requires_grad_(False)
        self.reduce = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.reduce_dim),
            nn.Tanh()
        )

        self.lm = ESM2(num_layers=cfg.num_layers, embed_dim=cfg.embed_dim, attention_heads=20, alphabet=self.alphabet, token_dropout=False)
        self.lm.requires_grad_(False)
        for i, layer in enumerate(self.lm.layers):
            if i < cfg.num_trainable_layers:
                layer.requires_grad_ = True 
                
        self.rep_recover = nn.Sequential(
            nn.Linear(cfg.reduce_dim, cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )

        if pretrained == None:
            self.lm.load_state_dict(self.load_esm_ckpt(esm_pretrained), strict=False)
        else:
            self.load_state_dict(self.load_ckpt(pretrained))

    def load_esm_ckpt(self, esm_pretrained):
        ckpt = {}
        model_data = torch.load(esm_pretrained)["model"]
        for k in model_data:
            if 'lm_head' in k:
                ckpt[k.replace('encoder.','')] = model_data[k]
            else:
                ckpt[k.replace('encoder.sentence_encoder.','')] = model_data[k]
        return ckpt

    def load_ckpt(self, pretrained):
        ckpt = {}
        model_data = torch.load(pretrained, map_location=self.device)
        for k in model_data:
            ckpt[k.replace('module.','')] = model_data[k]
        return ckpt
    
    def compose_input(self, list_tuple_seq):
        _, _, batch_tokens = self.batch_converter(list_tuple_seq)
        batch_tokens = batch_tokens.to(self.device)
        return batch_tokens 

    def set_wt_tokens(self, wt_seq):
        tokens = self.compose_input([('protein', wt_seq)])
        with torch.no_grad():
            encoded = self.encoder(tokens, set([self.esm_num_layers])) 
            encoded = encoded["representations"][self.esm_num_layers]
            encoded = encoded[:, 0]
            self.wt_encoded = encoded
            
            padding_mask = tokens.eq(self.lm.padding_idx)  # B, T
            self.padding_mask = padding_mask
            x = self.lm.embed_scale * self.lm.embed_tokens(tokens)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
                x = x.transpose(0, 1)
            self.wt_embed = x

    def encode(self, input):
        if isinstance(input, str):
            tokens = self.compose_input([('protein', input)])
        elif isinstance(input, list):
            tokens = self.compose_input([('protein', seq) for seq in input])
        else:
            tokens = input
            
        with torch.no_grad():
            output = self.encoder(tokens, set([self.esm_num_layers])) 
        x = output["representations"][self.esm_num_layers]
        x = x[:, 0]
        x = x - self.wt_encoded
        x = self.reduce(x)
        return x

    def decode(self, repr, to_seq=False, template=None, topk=None):
        sos = self.rep_recover(repr)
        x = torch.clone(self.wt_embed).repeat(1, repr.size(0), 1).to(repr.device)
        x[0] = sos
        for _, layer in enumerate(self.lm.layers):
            x, _ = layer(
                x,
                self_attn_padding_mask=self.padding_mask.to(repr.device).repeat(repr.size(0), 1),
                need_head_weights=False,
            )
        x = self.lm.emb_layer_norm_after(x)
        x = x.transpose(0, 1)
        logits = self.lm.lm_head(x)
        if to_seq: # constrained decoding
            tokens = torch.argmax(logits[:,1:-1,4:24], dim=-1)  
            if topk != None: 
                indices = torch.topk(torch.max(logits[:,1:-1,4:24], dim=-1).values, topk).indices
                indices = indices.flatten().tolist()
                sequences = [''.join([self.alphabet.all_toks[i+4] if i in indices else t for t, i in zip(list(template),sequence.tolist())]) for sequence in list(tokens)]
            else:
                sequences = [''.join([self.alphabet.all_toks[i+4] for i in sequence.tolist()]) for sequence in list(tokens)]
            if len(sequences) == 1:                
                sequences = sequences[0]
            return sequences
        return logits

    def forward(self, input, return_rep=False):
        repr = self.encode(input)
        logits = self.decode(repr)
        if return_rep:
            return logits, repr
        return logits