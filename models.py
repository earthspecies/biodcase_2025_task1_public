import torch
from torch import nn
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class BEATsEncoderAndMLP(nn.Module):
    def __init__(self, beats_checkpoint_fp):
        super().__init__()
        from beats import BEATs, BEATsConfig
        beats_ckpt = torch.load(beats_checkpoint_fp, map_location='cpu')
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        self.encoder = BEATs(beats_cfg)
        self.encoder.load_state_dict(beats_ckpt['model'])
        embedding_dim = self.encoder.cfg.encoder_embed_dim
        self.head = nn.Sequential(nn.Linear(2*embedding_dim,100), nn.ReLU(), nn.Linear(100,100), nn.ReLU(), nn.Linear(100,1))
        
        self.encoder.to(device)
        self.head.to(device)
        
    def embed(self, audio):
        """
        Encode mono audio with BEATs audio encoder
        """
        audio = audio.to(device)
        feats = self.encoder.extract_features(audio, feature_only=True)[0] # [B T C]
        feats = torch.mean(feats, dim = 1) #
        return feats
        
    def forward(self, audio_0, audio_1):
        """
        Compute similarity of two mono audio tensors
        """
        feats_0 = self.embed(audio_0)
        feats_1 = self.embed(audio_1)
        
        feats = torch.cat([feats_0,feats_1],dim=-1)
        similarity = self.head(feats).squeeze(-1)
        return similarity
    
    def freeze_encoder(self):
        """
        Freeze BEATs audio encoder
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
