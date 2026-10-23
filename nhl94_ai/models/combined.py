import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128, cnn_output_dim=None):
        # Extract the observation space dict components
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        if cnn_output_dim is None:
            cnn_output_dim = features_dim

        # Assume observation_space is gym.spaces.Dict with keys "image" and "scalar"
        # Customize according to the actual keys and shapes in your observations

        # CNN for image input
        cnn_input_shape = observation_space.spaces['image'].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass with dummy data
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.spaces['image'].sample()[None]).float()).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

        # For scalar inputs, fully connected layer
        scalar_dim = observation_space.spaces['scalar'].shape[0]
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU()
        )

        # Final combined features dimension
        self._features_dim = cnn_output_dim + 64

    def forward(self, observations):
        # observations is a dict with 'image' and 'scalar'
        img_tensor = observations['image']
        scalar_tensor = observations['scalar']

        cnn_features = self.cnn(img_tensor)
        cnn_features = self.cnn_fc(cnn_features)

        scalar_features = self.scalar_fc(scalar_tensor)

        # Concatenate features
        combined = th.cat([cnn_features, scalar_features], dim=1)
        return combined


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = dict(kwargs.pop('features_extractor_kwargs', {}))
        if 'cnn_output_dim' not in features_extractor_kwargs:
            features_extractor_kwargs['cnn_output_dim'] = features_extractor_kwargs.pop('features_dim', 128)

        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=features_extractor_kwargs
        )
