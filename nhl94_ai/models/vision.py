import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
