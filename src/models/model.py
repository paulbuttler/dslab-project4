import torch
import torch.nn as nn
import timm

class MultiTaskDNN(nn.Module):
    """
    Multi-task Deep Neural Network for landmark, pose, and shape prediction.

    Args:
        backbone_name (str): Name of the backbone model from timm.
        pretrained (bool): Whether to use a pre-trained backbone.
        num_landmarks (int): Number of landmarks for prediction.
        num_pose_params (int): Number of pose parameters for prediction.
        num_shape_params (int): Number of shape parameters for prediction.
        mlp_head_hidden_dim (int): Hidden dimension size for MLP heads.
    """

    def __init__(
        self,
        backbone_name="hrnet_w48.ms_in1k",
        pretrained=True,
        num_landmarks=1100,
        num_pose_params=21,
        num_shape_params=10,
        backbone_feat_dim=512,
        mlp_head_hidden_dim=512,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0,  # No classification head
            # The four HRnet feature maps are conbined into a tensor of shape [B, 2048]
        )

        self.compress = nn.Sequential(
            nn.Linear(2048, backbone_feat_dim), nn.LeakyReLU()
        )

        # MLP heads
        self.landmark_head = self._build_head(
            backbone_feat_dim, num_landmarks * 3, mlp_head_hidden_dim
        )  # (\mu_x, \mu_y, \logvar)
        self.pose_head = self._build_head(
            backbone_feat_dim, num_pose_params * 6, mlp_head_hidden_dim
        )  # 6D rotations
        self.shape_head = self._build_head(
            backbone_feat_dim, num_shape_params, mlp_head_hidden_dim
        )  # shapes

    def _build_head(self, input_dim, output_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.compress(features)
        B = features.shape[0]

        # Multi-task regression using compressed features
        return {
            "landmarks": self.landmark_head(features).view((B, -1, 3)),
            "pose": self.pose_head(features).view((B, -1, 6)),
            "shape": self.shape_head(features).view((B, -1)),
        }

## FOR TEST ONLY!
if __name__ == "__main__":
    model = MultiTaskDNN()
    dummy_input = torch.randn(2, 3, 256, 256)  # [batch, channel, height, width]

    with torch.no_grad():
        backbone_features = model.backbone(dummy_input)
        print(
            f"backbone feature dimensions: {backbone_features.shape}"
        )  # should be [2, 2048]
        print(
            f"compressed feature dim: {model.compress(backbone_features).shape}"
        )  # should be [2, 512]
        outputs = model(dummy_input)
        print(
            f"landmark part, head output dim: {outputs['landmarks'].shape}"
        )  # should be [2, 1100, 3]
        print(
            f"pose part, head output dim: {outputs['pose'].shape}"
        )  # should be [2, 21, 6]
        print(
            f"shape part, head output dim: {outputs['shape'].shape}"
        )  # should be [2, 10]
