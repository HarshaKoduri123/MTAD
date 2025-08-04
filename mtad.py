import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalInputEmbedding(nn.Module):
    def __init__(self, d_model=64, max_seq_len=500):
        super(MultiModalInputEmbedding, self).__init__()

        self.d_model = d_model

     
        self.spatial_proj = nn.Linear(2, d_model)    
        self.speed_proj = nn.Linear(1, d_model)     
        self.direction_proj = nn.Linear(2, d_model)  
        self.time_proj = nn.Linear(1, d_model)       

      
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

       
        self.fusion_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        """
        x: tensor shape (batch_size, seq_len, feature_dim)
        feature_dim = 6: [DeltaTime, DeltaLat, DeltaLon, SOG, COG, Heading]

        Expected feature order (example):
        [DeltaTime, DeltaLat, DeltaLon, SOG, COG, Heading]
        """
        batch_size, seq_len, _ = x.size()

        
        delta_time = x[:, :, 0].unsqueeze(-1)   
        delta_lat_lon = x[:, :, 1:3]            
        sog = x[:, :, 3].unsqueeze(-1)          
        cog_heading = x[:, :, 4:6]              

       
        spatial_emb = self.spatial_proj(delta_lat_lon)      
        speed_emb = self.speed_proj(sog)                     
        direction_emb = self.direction_proj(cog_heading)    
        time_emb = self.time_proj(delta_time)                

        pos_emb = self.pos_embedding[:, :seq_len, :]        
        spatial_emb = spatial_emb + pos_emb
        speed_emb = speed_emb + pos_emb
        direction_emb = direction_emb + pos_emb
        time_emb = time_emb + pos_emb

        concat_emb = torch.cat([spatial_emb, speed_emb, direction_emb, time_emb], dim=-1)  # (batch, seq_len, d_model*4)


        fused_emb = self.fusion_proj(concat_emb)  

        return fused_emb

class MultiModalAutoencoder(nn.Module):
    def __init__(self, d_model=64, feature_dim=6, seq_len=100, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(MultiModalAutoencoder, self).__init__()

        self.embedding = MultiModalInputEmbedding(d_model=d_model, max_seq_len=seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent compression (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Transformer Encoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feature_dim)  
        )

    def forward(self, x):
     
        embedded = self.embedding(x)  

        encoded = self.transformer_encoder(embedded)  
        compressed = self.bottleneck(encoded)        

        reconstructed = self.decoder(compressed)     
        return reconstructed
