"""Export deterministic SB3 policy outputs with a checked inference contract."""
import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import torch as th
from nhl94_ai.artifacts import load_policy, read_metadata


class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy(observation, deterministic=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default='./models/ScoreGoal.zip')
    parser.add_argument('--dest', default='./models/ScoreGoal')
    parser.add_argument('--samples', help='Representative .npy observations or a .npz demo shard')
    return parser


def parse_cmdline(argv):
    return build_parser().parse_args(argv)


def _inputs(model, args):
    shape = model.observation_space.shape
    if shape is None:
        raise ValueError('Export currently requires a Box observation space')
    rng = np.random.default_rng(0)
    samples = [np.zeros((1, *shape), dtype=np.float32), np.ones((1, *shape), dtype=np.float32),
               rng.uniform(-1, 1, (1, *shape)).astype(np.float32)]
    if getattr(args, 'samples', None):
        data = np.load(args.samples, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            try:
                values = data['observations'].copy()
            finally:
                data.close()
        else:
            values = data
        if tuple(values.shape[1:]) != tuple(shape):
            raise ValueError(f'Sample observations must have shape (N, {shape})')
        samples.extend(np.asarray(value, dtype=np.float32)[None] for value in values[:32])
    return samples


def export_onnx(args):
    model = load_policy(args.src, device='cpu')
    if getattr(model, '_nhl94_metadata', {}).get('schema', {}).get('action_type') == 'TARGET_POSITION':
        raise ValueError('TARGET_POSITION export requires a matching runtime controller and is not supported yet')
    model.policy.set_training_mode(False)
    policy = OnnxableSB3Policy(model.policy).eval()
    dummy = th.as_tensor(_inputs(model, args)[0])
    destination = str(Path(args.dest).expanduser()) + '.onnx'
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(policy, dummy, destination, opset_version=17, dynamo=False,
                   input_names=['input'], output_names=['actions', 'values', 'log_prob'])
    return model, policy, dummy, destination


def export_pytorch(args, model, onnx_policy, dummy_input):
    destination = str(Path(args.dest).expanduser()) + '.pt'
    th.jit.save(th.jit.trace(onnx_policy.eval(), dummy_input), destination)
    return destination


def test_models(args, model, dest_onnx_model, dest_pytorch_model):
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(dest_onnx_model))
    session = ort.InferenceSession(dest_onnx_model, providers=['CPUExecutionProvider'])
    traced = th.jit.load(dest_pytorch_model).eval()
    samples = _inputs(model, args)
    for sample in samples:
        with th.no_grad():
            expected = model.policy(th.as_tensor(sample), deterministic=True)
            actual_jit = traced(th.as_tensor(sample))
        actual_onnx = session.run(None, {'input': sample})
        for index, reference in enumerate(expected):
            expected_array = reference.cpu().numpy()
            np.testing.assert_allclose(actual_onnx[index], expected_array, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(actual_jit[index].cpu().numpy(), expected_array, rtol=1e-4, atol=1e-5)
    return len(samples)


def run(args):
    model, policy, dummy, onnx_path = export_onnx(args)
    torch_path = export_pytorch(args, model, policy, dummy)
    count = test_models(args, model, onnx_path, torch_path)
    contract = {
        'format_version': 1, 'artifact_type': 'deterministic-policy-export',
        'source': str(Path(args.src).expanduser()),
        'source_metadata': getattr(model, '_nhl94_metadata', {'compatibility': 'unknown'}),
        'input': {'name': 'input', 'shape': list(dummy.shape), 'dtype': 'float32'},
        'outputs': ['actions', 'values', 'log_prob'], 'opset': 17,
        'batch_size': 1, 'deterministic': True, 'validated_samples': count,
        'action_postprocessing': 'SB3 environment clipping/rescaling is excluded',
        'rtol': 1e-4, 'atol': 1e-5,
    }
    for path in (onnx_path, torch_path):
        Path(path + '.json').write_text(json.dumps(contract, indent=2) + '\n', encoding='utf-8')
    print(f'Exported and verified {count} inputs: {onnx_path}, {torch_path}')
    return contract


def main(argv):
    return run(parse_cmdline(argv[1:]))


if __name__ == '__main__':
    import sys
    main(sys.argv)
