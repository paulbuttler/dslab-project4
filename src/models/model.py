import torch
import torch.nn as nn
import timm

class MultiTaskDNN(nn.Module):
    def __init__(self, 
                 backbone_name='hrnet_w18.ms_aug_in1k',
                 pretrained=True,
                 num_landmarks=21, # landmark pred 21*3
                 num_pose_params=52,  # pose pred 52*6
                 num_shape_params=10, # shape pred 10
                 num_backbone_features=512, # In full experiment, use 512
                 mlp_head_hidden_dim=512, # 
            ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[4],
            ## Actually out_indices should be a subset of [0, 1, 2, 3, 4] ## Maybe we can exploit output from all backbone stages :)
        )
        
        backbone_feat_dim = self._get_backbone_feat_dim()
        self.feature_adapter = nn.Sequential(
            nn.Linear(backbone_feat_dim, num_backbone_features),
            nn.LeakyReLU()
        )
        
        # MLP head
        self.landmark_head = self._build_head(num_backbone_features, num_landmarks*3, mlp_head_hidden_dim)  # (\mu_x, \mu_y, \logvar)
        self.pose_head = self._build_head(num_backbone_features, num_pose_params*6, mlp_head_hidden_dim) # 6D rotations
        self.shape_head = self._build_head(num_backbone_features, num_shape_params, mlp_head_hidden_dim) # shapes

    def _get_backbone_feat_dim(self):
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy)[0]  # [B, C, H, W]
            return features.shape[1] * features.shape[2] * features.shape[3]  # dim after flattening

    def _build_head(self, input_dim, output_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.backbone(x)[0]  # [B, C, H, W]
        B = features.shape[0]
        
        features = features.view(B, -1)
        hidden_features = self.feature_adapter(features)  # [B, 512]
        
        # multi-task regression
        return {
            'landmarks': self.landmark_head(hidden_features).view((B, -1, 3)), # (B, n_ldmks, 3)
            'pose': self.pose_head(hidden_features).view((B, -1, 6)), # (B, n_pose_params, 6)
            'shape': self.shape_head(hidden_features).view((B, -1)),  # (B, n_shape_params)
        }
    
## FOR TEST ONLY!
if __name__ == "__main__":
    model = MultiTaskDNN(num_landmarks=52, num_pose_params=52, num_shape_params=10)
    dummy_input = torch.randn(2, 3, 224, 224)  # [batch, channel, height, width]
    
    with torch.no_grad():
        backbone_features = model.backbone(dummy_input)[0]
        print(f"Backbone output dim: {backbone_features.shape}")  # should be [2, C, H, W]
        
        features_flat = backbone_features.view(2, -1)
        hidden_features = model.feature_adapter(features_flat)
        print(f"feature adjust layer output dim: {hidden_features.shape}")  # should be [2, 512]
        
        outputs = model(dummy_input)
        print(f"landmark part, head output dim: {outputs['landmarks'].shape}")  # should be (2, 52, 3)
        print(f"pose part, head output dim: {outputs['pose'].shape}")  # should be (2, 52, 6)
        print(f"shape part, head output dim: {outputs['shape'].shape}")  # should be (2, 10)