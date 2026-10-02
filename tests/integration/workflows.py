"""Small training, custom-checkpoint, and imitation workflows; requires NHL94 ROM."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run():
    output = Path(tempfile.mkdtemp(prefix='nhl94-workflows-'))
    params = json.loads((ROOT / 'configs/training/default.json').read_text(encoding='utf-8'))
    params.update(n_steps=16, batch_size=8, n_epochs=1, net_arch={'pi': [16], 'vf': [16]},
                  frame_skip=4, residual_mlp={'hidden_dim': 16, 'num_blocks': 1, 'head_dim': 16})
    config = output / 'settings.json'
    config.write_text(json.dumps(params), encoding='utf-8')
    common = ['--env=NHL94-Genesis-v0', '--nn=ResidualMlpPolicy', '--rf=PostPlay',
              '--action_type=FILTERED', f'--hyperparams={config}']

    def command(name, arguments):
        result = subprocess.run([sys.executable, '-B', '-m', 'nhl94_ai', name, *arguments],
                                cwd=output, capture_output=True, text=True, timeout=120, check=False,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                                     'RETRO_VECENV_START_METHOD': 'spawn'})
        (output / f'{name}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed: {result.stdout[-4000:]}\n{result.stderr[-4000:]}')

    command('train', common + ['--num_env=1', '--num_timesteps=16', '--live-eval-episodes=1',
                               f'--output_basedir={output}'])
    checkpoint = next(path for path in output.rglob('*.zip') if '_best_live' not in path.stem)

    command('collect', common + ['--num_episodes=2', '--max_steps=16', f'--output={output / "demos"}'])
    command('bc', common + ['--datasets', str(output / 'demos'), '--epochs=1', '--batch_size=8',
                           '--device=cpu', f'--output_model={output / "bc.zip"}', f'--load_model={checkpoint}'])
    command('dagger', common + ['--model', str(output / 'bc.zip'), '--base_datasets', str(output / 'demos'),
                               '--rounds=1', '--episodes_per_round=1', '--max_steps=8',
                               '--bc_epochs=1', '--bc_batch_size=8', '--device=cpu',
                               f'--output_dir={output / "dagger"}'])
    print(f'PASS: training, custom checkpoint reload, reactive collection, BC and DAgger: {output}')


if __name__ == '__main__':
    run()
