import torch
import torch as th
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


class CNNTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, nhead=8, num_layers=3):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output dim
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Transformer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=cnn_out_dim, nhead=nhead, dim_feedforward=256, dropout=0.1
        )
        self.transformer = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(cnn_out_dim, features_dim)

    def forward(self, observations):
        # CNN extracts spatial features
        cnn_features = self.cnn(observations)  # Shape: (batch_size, cnn_out_dim)

        # Transformer expects (seq_len, batch_size, dim)
        # We treat each frame as a "sequence of 1" for simplicity
        cnn_features = cnn_features.unsqueeze(0)  # Shape: (1, batch_size, cnn_out_dim)

        # Transformer processes temporal features
        transformer_out = self.transformer(cnn_features)
        transformer_out = transformer_out.squeeze(0)  # Back to (batch_size, cnn_out_dim)

        # Final projection
        return self.fc(transformer_out)


class DartFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dart model that uses decision trees for feature extraction.
    """
    def __init__(self, observation_space, features_dim=64):
        super(DartFeatureExtractor, self).__init__(observation_space, features_dim)

        # Determine input dimension based on observation space
        if isinstance(observation_space, gym.spaces.Dict):
            self.is_dict = True
            self.flatten = nn.Flatten()
            with th.no_grad():
                sample = observation_space.sample()
                sample_tensor = {k: th.as_tensor(v[None]).float() for k, v in sample.items()}
                flattened_dim = sum(v.numel() for v in sample_tensor.values())
        else:
            self.is_dict = False
            with th.no_grad():
                sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
                flattened_dim = sample_tensor.numel()

        # Tree-like layers with residual connections
        self.tree_layer1 = nn.Sequential(
            nn.Linear(flattened_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

        self.tree_layer2 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        if self.is_dict:
            flattened = []
            for key, value in observations.items():
                flattened.append(self.flatten(value))
            x = th.cat(flattened, dim=1)
        else:
            x = observations.flatten(start_dim=1)

        out1 = self.tree_layer1(x)
        out2 = self.tree_layer2(out1)
        return out1 + out2  # Residual connection


class DartPolicy(ActorCriticPolicy):
    """
    Policy network for Dart model with tree-like feature extraction.
    """
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, *args, **kwargs):
        # Remove features_extractor_kwargs from kwargs if present
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        features_extractor_kwargs['features_dim'] = features_extractor_kwargs.get('features_dim', 64)

        super(DartPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch or dict(pi=[64, 64], vf=[64, 64]),
            features_extractor_class=DartFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs
        )

        # Get the actual latent dimension from mlp_extractor
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf

        # Additional tree-like processing in policy head
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim_pi, latent_dim_pi),
            nn.ReLU(),
            nn.Linear(latent_dim_pi, latent_dim_pi),
            nn.ReLU()
        )

        # Value head should output a single value
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim_vf, latent_dim_vf),
            nn.ReLU(),
            nn.Linear(latent_dim_vf, 1),  # Output single value
            nn.ReLU()
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Additional tree-like processing
        latent_pi = self.policy_net(latent_pi)

        # Get value estimate (should be scalar per observation)
        values = self.value_net(latent_vf).squeeze(-1)  # Remove extra dimension

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def predict_values(self, obs):
        """
        Get the value estimates for observations
        """
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf).squeeze(-1)


class ViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ViTFeatureExtractor, self).__init__(observation_space, features_dim)

        # Expecting image observation only (C, H, W)
        channels, height, width = observation_space.shape

        # Load pretrained ViT base model with patch size 16 (adjust as you see fit)
        import timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Adjust input conv if there is channel mismatch
        if channels != 3:
            self.vit.patch_embed.proj = nn.Conv2d(channels, self.vit.embed_dim, kernel_size=16, stride=16)

        # Remove classifier head, keep feature extraction only
        self.vit.head = nn.Identity()

        # Project ViT output dimension to desired features_dim
        self.linear = nn.Linear(self.vit.embed_dim, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Resize input to 224x224 as ViT expects fixed input size
        x = nn.functional.interpolate(observations, size=(224, 224), mode='bilinear', align_corners=False)

        features = self.vit(x)
        return self.linear(features)


class ViTPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(ViTPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=ViTFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels=3, channels=(16, 32, 32)):
        super(ImpalaCNN, self).__init__()
        self.stacks = nn.ModuleList()
        in_channels = input_channels
        for out_channels in channels:
            self.stacks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    ResidualBlock(out_channels),
                    ResidualBlock(out_channels),
                )
            )
            in_channels = out_channels
        # Flatten feature size will depend on input image size. You may add an adaptive pooling here if needed.

    def forward(self, x):
        for stack in self.stacks:
            x = stack(x)
        x = th.flatten(x, start_dim=1)
        return x


class CustomImpalaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomImpalaFeatureExtractor, self).__init__(observation_space, features_dim)
        # observation_space.shape = (C, H, W)
        n_input_channels = observation_space.shape[0]
        self.cnn = ImpalaCNN(input_channels=n_input_channels)

        # Compute the size of the output of the CNN by passing a dummy tensor
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            cnn_output = self.cnn(sample_input)
        self._features_dim = cnn_output.shape[1]

    def forward(self, observations):
        return self.cnn(observations)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Assuming the observation space is images with shape (C, H, W)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
