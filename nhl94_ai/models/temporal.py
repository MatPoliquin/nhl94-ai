import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class TemporalHybridExtractorBase(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=256,
        temporal_dim=128,
        spatial_hidden_dim=128,
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects a sequence observation space shaped as "
                "(seq_len, feature_dim)"
            )

        self.seq_len, self.input_dim = observation_space.shape
        self.temporal_dim = temporal_dim
        self.spatial_hidden_dim = spatial_hidden_dim

        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.spatial_hidden_dim),
            nn.LayerNorm(self.spatial_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.spatial_hidden_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.ReLU(),
        )

        self.temporal_input_proj = nn.Sequential(
            nn.Linear(self.input_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.ReLU(),
        )

        self.temporal_model = self._build_temporal_model()

        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.ReLU(),
        )

        self.combiner = nn.Sequential(
            nn.Linear(temporal_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def _build_temporal_model(self):
        raise NotImplementedError()

    def _run_temporal_model(self, temporal_input):
        temporal_output = self.temporal_model(temporal_input)
        if isinstance(temporal_output, tuple):
            temporal_output = temporal_output[0]
        return temporal_output

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)

        current_frame = observations[:, -1, :]
        spatial_features = self.spatial_encoder(current_frame)

        temporal_input = self.temporal_input_proj(observations)
        temporal_output = self._run_temporal_model(temporal_input)
        temporal_features = self.temporal_proj(temporal_output[:, -1, :])
        combined = torch.cat([spatial_features, temporal_features], dim=-1)
        return self.combiner(combined)


class HybridMambaMLPExtractor(TemporalHybridExtractorBase):
    def __init__(
        self,
        observation_space,
        features_dim=256,
        mamba_dim=128,
        spatial_hidden_dim=128,
        d_state=64,
        d_conv=4,
        expand=2,
        gru_layers=2,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            temporal_dim=mamba_dim,
            spatial_hidden_dim=spatial_hidden_dim,
        )

    def _build_temporal_model(self):
        try:
            from mambapy.mamba import Mamba, MambaConfig

            config = MambaConfig(
                d_model=self.temporal_dim,
                n_layers=1,
                d_state=self.d_state,
                expand_factor=self.expand,
                d_conv=self.d_conv,
                use_cuda=torch.cuda.is_available(),
            )
            return Mamba(config)
        except ImportError:
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "HybridMambaPolicy requires a Mamba backend. Install mamba-ssm or mambapy, "
                    "or use GRUMlpPolicy instead."
                ) from exc

        return Mamba(
            d_model=self.temporal_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )


class HybridMambaPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {})
        if not isinstance(features_extractor_kwargs, dict):
            raise TypeError("features_extractor_kwargs must be a dict when provided.")
        features_extractor_kwargs = dict(features_extractor_kwargs)
        features_extractor_kwargs.setdefault("features_dim", 256)

        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=HybridMambaMLPExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
        )


class GRUMlpExtractor(TemporalHybridExtractorBase):
    def __init__(
        self,
        observation_space,
        features_dim=256,
        hidden_dim=128,
        spatial_hidden_dim=128,
        gru_layers=2,
    ):
        self.gru_layers = gru_layers
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            temporal_dim=hidden_dim,
            spatial_hidden_dim=spatial_hidden_dim,
        )

    def _build_temporal_model(self):
        return nn.GRU(
            input_size=self.temporal_dim,
            hidden_size=self.temporal_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )

    def _run_temporal_model(self, temporal_input):
        temporal_output, _ = self.temporal_model(temporal_input)
        return temporal_output


class GRUMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {})
        if not isinstance(features_extractor_kwargs, dict):
            raise TypeError("features_extractor_kwargs must be a dict when provided.")
        features_extractor_kwargs = dict(features_extractor_kwargs)
        features_extractor_kwargs.setdefault("features_dim", 256)

        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GRUMlpExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
        )
