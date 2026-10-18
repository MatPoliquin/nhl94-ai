"""Collect, imitate, train, evaluate and watch the hybrid possession policy."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from nhl94_ai.agents.learned_offense import N_ACTIONS
from nhl94_ai.env.offense import OffenseEnv, SCHEMA, reset_options
from nhl94_ai.env.offense_decisions import DecisionEnv
from nhl94_ai.training.decision_ppo import DecisionPPO
from nhl94_ai.models.offense import (
    OffensePolicy, CurrentOffenseExtractor, ReceiverOrderedExtractor,
    load_offense_model, save_offense_model, model_path, write_json,
)
from nhl94_ai.training.offense_learning import sampling_weights, imitation_metrics, checkpoint_rank, mirror_examples


def env_options(args, *, learner='learned', opponents=None):
    starts = args.starts
    if args.phase == 'train':
        starts = []
        for path in args.starts:
            metadata = json.loads(Path(str(path) + '.json').read_text(encoding='utf-8'))
            source = f"{metadata['state_name']}:{metadata['source_seed']}"
            if not validation_source(source, args.split_seed):
                starts.append(path)
    return dict(states=args.states, opponents=opponents or args.opponents, learner=learner,
                seconds=args.seconds, goalie_policy=args.goalie_policy, side=args.side,
                max_frames=args.max_frames, starts=starts, start_probability=args.start_probability,
                task=args.task, attack_frames=args.attack_frames, control_scope=args.control_scope, gamma=args.gamma)


def make_training_env(options, decision_only=False, gamma=0.997):
    torch.set_num_threads(1)
    env = OffenseEnv(**options)
    if decision_only:
        env = DecisionEnv(env, gamma)
    return Monitor(env)


def new_model(env, args):
    policy_kwargs = dict(features_extractor_kwargs=dict(
        features_dim=args.features_dim, hidden_dim=args.hidden_dim,
        spatial_hidden_dim=args.hidden_dim, gru_layers=1))
    if args.architecture == 'mlp':
        policy_kwargs = dict(features_extractor_class=CurrentOffenseExtractor,
                             features_extractor_kwargs=dict(features_dim=args.features_dim,
                                                            hidden_dim=args.hidden_dim))
    elif args.architecture == 'receiver-gru':
        policy_kwargs['features_extractor_class'] = ReceiverOrderedExtractor
    policy_kwargs['net_arch'] = dict(pi=[args.head_dim], vf=[args.head_dim])
    algorithm = DecisionPPO if args.decision_only else PPO
    return algorithm(OffensePolicy, env, learning_rate=args.learning_rate, n_steps=args.rollout_steps,
               batch_size=args.batch_size, n_epochs=4, gamma=args.gamma, gae_lambda=0.99,
               ent_coef=0.01, target_kl=0.03, seed=args.seed, device=args.device, verbose=1,
               policy_kwargs=policy_kwargs)


def collect(args):
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.rglob('episode-*.npz')):
        raise FileExistsError(f'Collect into a fresh directory: {output}')
    if args.num_envs > 1:
        jobs = []
        for worker in range(min(args.num_envs, args.episodes)):
            options = copy.copy(args)
            options.num_envs = 1
            options.output = str(output / f'worker-{worker:03d}')
            options.episode_indices = list(range(worker, args.episodes, args.num_envs))
            jobs.append(options)
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            summaries = [row for result in pool.map(collect_worker, jobs) for row in result]
        write_json(output / 'collection.json', dict(schema=SCHEMA, episodes=summaries))
        return summaries
    model = load_offense_model(args.model) if args.model else None
    env = OffenseEnv(**env_options(args, learner='learned' if model else 'teacher'))
    summaries = []
    try:
        for episode in getattr(args, 'episode_indices', range(args.episodes)):
            obs, info = env.reset(seed=args.seed + episode, options=reset_options(args, episode))
            rng = np.random.default_rng(args.seed + episode)
            rows, labels, indices, rewards = [], [], [], []
            saved, last_saved_step = 0, -30
            while True:
                before = obs.copy()
                label = None
                if model is not None:
                    # Stateful DAgger oracle observes the learner's actual button
                    # and lifecycle history, not an independent imaginary game.
                    oracle = copy.deepcopy(env.controllers[env.side - 1])
                    view = env.views[env.side - 1]
                    oracle.mode = 'teacher'
                    if oracle.begin_step(view):
                        oracle.frame(view)
                        label = oracle.teacher_command
                    action = model.predict(obs, deterministic=True)[0] if before[-1, -N_ACTIONS:].sum() > 1 else 0
                    if label is not None and rng.random() < args.dagger_beta:
                        action = label
                else:
                    action = 0
                if (args.save_starts and saved < args.save_starts and len(rewards) - last_saved_step >= 30
                        and before[-1, -N_ACTIONS:].sum() > 1):
                    start_path = output / f'episode-{episode:05d}-start-{saved}.pkl.gz'
                    env.save_start(start_path)
                    saved += 1
                    last_saved_step = len(rewards)
                obs, reward, terminated, truncated, info = env.step(action)
                if model is None:
                    label = info['teacher_command']
                if label is not None:
                    if not before[-1, -N_ACTIONS + label]:
                        raise ValueError('Teacher command was not legal in its recorded observation')
                    rows.append(before)
                    labels.append(label)
                    indices.append(len(rewards))
                rewards.append(reward)
                if terminated or truncated:
                    break
            returns, value = np.zeros(len(rewards), np.float32), 0.0
            for step in range(len(rewards) - 1, -1, -1):
                value = rewards[step] + args.gamma * value
                returns[step] = value
            path = output / f'episode-{episode:05d}.npz'
            if path.exists():
                raise FileExistsError(f'Dataset shard already exists: {path}')
            np.savez_compressed(path, observations=np.asarray(rows, np.float32).reshape((-1, *env.observation_space.shape)),
                                actions=np.asarray(labels, np.int64), returns=returns[indices],
                                steps=np.asarray(indices, np.int64), rewards=np.asarray(rewards, np.float32),
                                terminated=np.asarray(terminated), truncated=np.asarray(truncated))
            source = f'{env.state_name}:{env.episode_seed}'
            write_json(str(path) + '.json', dict(schema=SCHEMA, source=source, gamma=args.gamma,
                                               teacher='classic-v1', behavior_model=args.model,
                                               dagger_beta=args.dagger_beta if model else None,
                                               decisions=len(labels), episode=info))
            summaries.append(dict(source=source, decisions=len(labels), **info))
            print(json.dumps(summaries[-1]), flush=True)
    finally:
        env.close()
    write_json(output / 'collection.json', dict(schema=SCHEMA, episodes=summaries))
    return summaries


def collect_worker(args):
    torch.set_num_threads(args.threads)
    return collect(args)


def read_data(paths, gamma):
    observations, actions, returns, valid, sources = [], [], [], [], []
    files = sorted({file for path in paths for file in (
        Path(path).expanduser().rglob('*.npz') if Path(path).expanduser().is_dir() else [Path(path).expanduser()])})
    if not files:
        raise ValueError('No demonstration shards found')
    for path in files:
        metadata = json.loads(Path(str(path) + '.json').read_text(encoding="utf-8"))
        if metadata['schema'] != SCHEMA or metadata['gamma'] != gamma:
            raise ValueError(f'Dataset schema or discount mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            chunk_observations = np.asarray(data['observations'], dtype=np.float32)
            chunk_actions = np.asarray(data['actions'])
            count = len(chunk_actions)
            if not count:
                continue
            if (chunk_observations.shape != (count, SCHEMA['history'], len(SCHEMA['fields']))
                    or not np.isfinite(chunk_observations).all()
                    or not np.issubdtype(chunk_actions.dtype, np.integer)
                    or np.any(chunk_actions < 0) or np.any(chunk_actions >= N_ACTIONS)):
                raise ValueError(f'Invalid demonstration shapes or values: {path}')
            if not np.all(chunk_observations[np.arange(count), -1, -N_ACTIONS + chunk_actions] > 0.5):
                raise ValueError(f'Demonstration labels violate their action masks: {path}')
            observations.append(chunk_observations)
            actions.append(chunk_actions)
            returns.append(data['returns'])
            valid.extend([bool(data['terminated']) and not bool(data['truncated'])] * count)
            sources.extend([metadata['source']] * count)
    if not actions:
        raise ValueError('Dataset contains no offensive decisions')
    return (np.concatenate(observations), np.concatenate(actions), np.concatenate(returns),
            np.asarray(valid), np.asarray(sources))


def validation_source(source, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{source}'.encode()).digest()[:8], 'big') % 5 == 0


def split_sources(sources, seed):
    groups = np.unique(sources)
    if len(groups) < 2:
        raise ValueError('Need at least two source games for a leakage-free validation split')
    # Hash assignment stays stable when DAgger adds games or training uses a
    # different initialization seed. Never move old validation games into train.
    heldout = [source for source in groups if validation_source(source, seed)]
    validation = np.isin(sources, heldout)
    if not validation.any() or validation.all():
        raise ValueError('Need more source games to populate the fixed 80/20 validation split')
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def clone(args):
    obs, actions, returns, valid, sources = read_data(args.data, args.gamma)
    train, validation = split_sources(sources, args.split_seed)
    torch.manual_seed(args.seed)
    env = OffenseEnv(**env_options(args))
    model = load_offense_model(args.model, env) if args.model else new_model(env, args)
    tensors = tuple(torch.as_tensor(x) for x in (obs, actions, returns, valid))
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    counts = np.bincount(actions[train], minlength=N_ACTIONS)
    probabilities = sampling_weights(actions[train], sources[train])
    best, metrics = None, []
    try:
        for epoch in range(args.epochs):
            model.policy.set_training_mode(True)
            shuffled = rng.choice(train, len(train), replace=True, p=probabilities)
            for offset in range(0, len(shuffled), args.batch_size):
                index = shuffled[offset:offset + args.batch_size]
                batch, labels, targets, complete = (tensor[index] for tensor in tensors)
                if args.mirror_augmentation:
                    reflected, reflected_labels = mirror_examples(batch, labels)
                    mask = torch.as_tensor(rng.random(len(index)) < 0.5)
                    batch = torch.where(mask[:, None, None], reflected, batch)
                    labels = torch.where(mask, reflected_labels, labels)
                values, log_prob, _ = model.policy.evaluate_actions(batch, labels)
                loss = -log_prob.mean()
                if complete.any():
                    loss = loss + 0.5 * torch.nn.functional.mse_loss(values.flatten()[complete], targets[complete])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()
            model.policy.set_training_mode(False)
            losses, predictions = [], []
            with torch.no_grad():
                for offset in range(0, len(validation), args.batch_size):
                    index = validation[offset:offset + args.batch_size]
                    distribution = model.policy.get_distribution(tensors[0][index])
                    labels = tensors[1][index]
                    predicted = distribution.get_actions(deterministic=True)
                    losses.extend((-distribution.log_prob(labels)).tolist())
                    predictions.extend(predicted.tolist())
            row = dict(epoch=epoch + 1, **imitation_metrics(
                actions[validation], np.asarray(predictions), np.asarray(losses), counts))
            metrics.append(row)
            print(json.dumps(row), flush=True)
            rank = checkpoint_rank(row)
            if best is None or rank > best:
                best = rank
                save_offense_model(model, args.output, phase='bc', seed=args.seed,
                                   split_seed=args.split_seed,
                                   validation=row, training_action_counts=counts.tolist(),
                                   sampling='equal-intent-and-source-game',
                                   mirror_augmentation=args.mirror_augmentation, weight_decay=args.weight_decay,
                                   training_sources=np.unique(sources[train]).tolist(),
                                   validation_sources=np.unique(sources[validation]).tolist())
    finally:
        env.close()
    write_json(str(model_path(args.output)) + '.metrics.json', dict(
        training_decisions=len(train), validation_decisions=len(validation),
        training_action_counts=counts.tolist(), epochs=metrics))
    return metrics


class OffenseTrainingCallback(BaseCallback):
    """Immutable historical checkpoints, published only for subsequent resets."""
    def __init__(self, directory, pool, *, every, max_snapshots, total_steps, bc_data=None, seed=0,
                 checkpoint_directory=None, checkpoint_every=0, checkpoint_details=None):
        super().__init__()
        self.directory = Path(directory)
        self.anchors = tuple(pool)
        self.every, self.max_snapshots = every, max_snapshots
        self.total_steps, self.bc_data = total_steps, bc_data
        self.seed, self.started, self.last = seed, 0, 0
        self.history = []
        self.episodes = []
        self.active_steps = self.environment_steps = 0
        self.native_frames = 0
        self.executed_commands = 0
        self.rng = np.random.default_rng(seed)
        self.checkpoint_directory = Path(checkpoint_directory) if checkpoint_directory else None
        self.checkpoint_every = checkpoint_every
        self.checkpoint_details = checkpoint_details or {}
        self.checkpoints = []
        self.last_checkpoint = None

    def _on_training_start(self):
        self.started = self.last = self.num_timesteps
        if self.checkpoint_every:
            self._save_checkpoint()

    def _save_checkpoint(self):
        path = self.checkpoint_directory / f'step-{self.num_timesteps:010d}.zip'
        if path.exists():
            raise FileExistsError(f'Checkpoint is immutable: {path}')
        saved = save_offense_model(self.model, path, phase='ppo-checkpoint', seed=self.seed,
                                  active_decisions=self.active_steps, native_frames=self.native_frames,
                                  executed_commands=self.executed_commands,
                                  optimizer_updates=self.model._n_updates,
                                  **self.checkpoint_details)
        self.checkpoints.append(saved)
        self.last_checkpoint = self.num_timesteps
        write_json(self.checkpoint_directory / 'checkpoints.json', dict(paths=self.checkpoints))
        print(json.dumps(dict(checkpoint=saved, timesteps=self.num_timesteps)), flush=True)

    def _on_training_end(self):
        if self.checkpoint_every and self.last_checkpoint != self.num_timesteps:
            self._save_checkpoint()

    def _on_rollout_start(self):
        # SB3 calls rollout_end BEFORE updating the policy. The next rollout
        # starts after that update, so these archives include the labeled data.
        if self.checkpoint_every and self.num_timesteps - self.last_checkpoint >= self.checkpoint_every:
            self._save_checkpoint()
        # Optional decaying BC regularization occurs BEFORE collecting the next
        # on-policy rollout, so the saved old action probabilities stay valid.
        if self.bc_data is not None:
            obs, labels, probabilities = self.bc_data
            strength = 0.1 * max(0, 1 - (self.num_timesteps - self.started) / self.total_steps)
            index = self.rng.choice(len(labels), size=min(256, len(labels)), p=probabilities)
            loss = -strength * self.model.policy.get_distribution(
                torch.as_tensor(obs[index], device=self.model.device)).log_prob(
                    torch.as_tensor(labels[index], device=self.model.device)).mean()
            self.model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
            self.model.policy.optimizer.step()
            self.logger.record('train/imitation_strength', strength)

    def _on_step(self):
        if self.every and self.num_timesteps - self.last >= self.every:
            path = self.directory / f'snapshot-{self.num_timesteps:010d}.zip'
            if path.exists():
                raise FileExistsError(f'Snapshot is immutable: {path}')
            saved = save_offense_model(self.model, path, phase='self-play-snapshot', seed=self.seed)
            self.history.append(saved)
            pool = list(dict.fromkeys([*self.anchors, *self.history[-self.max_snapshots:]]))
            self.training_env.env_method('set_pool', pool)
            write_json(self.directory / 'pool.json', dict(anchors=self.anchors, opponents=pool,
                                                         snapshots=self.history))
            self.last = self.num_timesteps
            print(json.dumps(dict(snapshot=saved, opponents=len(pool))), flush=True)
        for info in self.locals.get('infos', []):
            self.environment_steps += 1
            self.active_steps += bool(info.get('actor_active'))
            self.native_frames += info.get('native_frames', 0)
            self.executed_commands += bool(info.get('command_used'))
            if 'episode' in info:
                row = {key: info[key] for key in ('side', 'opponent', 'seed', 'goals_for',
                                                 'goals_against', 'completed', 'episode')}
                row.update({key: info[key] for key in ('outcome', 'success', 'attempt_frames', 'snapshot', 'opportunity') if key in info})
                self.episodes.append(row)
                if 'outcome' not in row or len(self.episodes) % 100 == 0:
                    print(json.dumps(dict(row, total_episodes=len(self.episodes),
                                          active_decisions=self.active_steps)), flush=True)
        return True


def train(args):
    pool = list(args.opponents)
    if args.self_play and args.model:
        pool.append(str(model_path(args.model)))
    if args.self_play and 'classic' not in pool:
        pool.append('classic')
    options = env_options(args, opponents=pool)
    factories = [partial(make_training_env, options, args.decision_only, args.gamma) for _ in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method='spawn')
    try:
        env.seed(args.seed)
        model = load_offense_model(args.model, env, n_steps=args.rollout_steps, batch_size=args.batch_size,
                                   learning_rate=args.learning_rate, decision_only=args.decision_only,
                                   device=args.device,
                                   expected_control_scope=args.control_scope,
                                   allow_control_transfer=args.allow_control_transfer) if args.model else new_model(env, args)
        model.set_random_seed(args.seed)
        parent_timesteps = model.num_timesteps
        if args.model:
            metadata = json.loads(Path(str(model_path(args.model)) + '.offense.json').read_text(encoding='utf-8'))
            if (metadata.get('algorithm') == 'DecisionPPO') != args.decision_only:
                # Fixed-tick and decision counts cannot be added to one another.
                model.num_timesteps = 0
        if args.reset_value:
            with torch.no_grad():
                model.policy.value_net.weight.zero_()
                model.policy.value_net.bias.zero_()
        if abs(model.gamma - args.gamma) > 1e-12:
            raise ValueError('Checkpoint gamma differs from requested training discount')
        bc_data = None
        if args.data:
            observations, labels, _, _, sources = read_data(args.data, args.gamma)
            train_indices, _ = split_sources(sources, args.split_seed)
            bc_data = (observations[train_indices], labels[train_indices],
                       sampling_weights(labels[train_indices], sources[train_indices]))
        output = model_path(args.output)
        details = dict(options=vars(args), parent_timesteps=parent_timesteps,
                       initialization='checkpoint' if args.model else 'random',
                       training_contract=training_contract(args))
        model.set_logger(configure(str(output.parent / (output.stem + '-logs')), ['stdout', 'csv']))
        callback = OffenseTrainingCallback(output.parent / (output.stem + '-opponents'), pool,
                                            every=args.snapshot_every if args.self_play else 0,
                                            max_snapshots=args.max_snapshots, total_steps=args.timesteps,
                                            bc_data=bc_data, seed=args.seed,
                                            checkpoint_directory=output.parent / (output.stem + '-checkpoints'),
                                            checkpoint_every=args.checkpoint_every, checkpoint_details=details)
        started = time.monotonic()
        model.learn(total_timesteps=args.timesteps, callback=callback, reset_num_timesteps=False)
        path = save_offense_model(model, output, phase='ppo', seed=args.seed, opponents=pool,
                                 elapsed_seconds=time.monotonic() - started, **details,
                                 completed_periods=sum(row['completed'] for row in callback.episodes) if args.task == 'period' else 0,
                                 completed_attempts=sum(row['completed'] for row in callback.episodes) if args.task != 'period' else 0,
                                 active_decisions=callback.active_steps, environment_steps=callback.environment_steps,
                                 executed_commands=callback.executed_commands,
                                 native_frames=callback.native_frames)
        write_json(str(output) + '.training.json', dict(episodes=callback.episodes,
                                                      snapshots=callback.history,
                                                      checkpoints=callback.checkpoints,
                                                      active_decisions=callback.active_steps,
                                                      executed_commands=callback.executed_commands,
                                                      environment_steps=callback.environment_steps,
                                                      native_frames=callback.native_frames))
        return path
    finally:
        env.close()


def evaluate(args):
    if args.num_envs > 1 and args.phase == 'evaluate':
        jobs = []
        for worker in range(min(args.num_envs, args.episodes)):
            options = copy.copy(args)
            options.num_envs = 1
            options.episode_indices = list(range(worker, args.episodes, args.num_envs))
            options.output = str(Path(args.output).with_suffix('')) + f'-worker-{worker}.json'
            jobs.append(options)
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            reports = list(pool.map(evaluate_worker, jobs))
        results = [row for report in reports for row in report['results']]
        return evaluation_report(args, results, tuple(dict.fromkeys(row['agent'] for row in results)))
    model = load_offense_model(args.model, expected_task=args.task,
                               allow_task_transfer=args.allow_task_transfer,
                               expected_control_scope=args.control_scope,
                               allow_control_transfer=args.allow_control_transfer) if args.model else None
    if model is not None and args.task == 'opportunity' and abs(model.gamma - args.gamma) > 1e-12:
        raise ValueError('Opportunity evaluation discount must match the checkpoint')
    agents = args.agents or (('model', 'teacher', 'classic') if model else ('teacher', 'classic'))
    if 'model' in agents and model is None:
        raise ValueError('The model evaluation arm requires --model')
    if args.phase == 'play':
        agents = ('model',) if model else ('classic',)
    results = []
    for opponent in args.opponents:
        for agent in agents:
            env = OffenseEnv(**env_options(args, learner='learned' if agent in ('model', 'shoot', 'coast', 'forward') else agent,
                                          opponents=(opponent,)))
            try:
                for episode in getattr(args, 'episode_indices', range(args.episodes)):
                    options = reset_options(args, episode, opponent)
                    if args.task != 'period':
                        compatible = sorted(p for p in args.starts if env.start_metadata[p]['opponent'] == opponent)
                        options = dict(snapshot=compatible[episode % len(compatible)])
                    obs, _ = env.reset(seed=args.seed + episode, options=options)
                    total = 0.0
                    while True:
                        action = (model.predict(obs, deterministic=True)[0]
                                  if agent == 'model' and obs[-1, -N_ACTIONS:].sum() > 1 else 0)
                        if agent == 'shoot' and obs[-1, -N_ACTIONS + 17]:
                            action = 17
                        if agent == 'forward' and obs[-1, -N_ACTIONS + 1]:
                            action = 1
                        obs, reward, terminated, truncated, info = env.step(action)
                        total += reward
                        if args.phase == 'play':
                            import cv2
                            cv2.imshow('Learned offense', cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
                            if cv2.waitKey(1) == 27:
                                return results
                            time.sleep(4 / 60)
                        if terminated or truncated:
                            break
                    info.update(agent=agent, reward=total)
                    results.append(info)
                    print(json.dumps(info), flush=True)
            finally:
                env.close()
    return evaluation_report(args, results, agents)


def evaluate_worker(args):
    torch.set_num_threads(args.threads)
    return evaluate(args)


def training_contract(args):
    from nhl94_ai.env.opportunity import REWARD_VERSION, REWARD_SPEC
    reward = (REWARD_VERSION if args.task == 'opportunity' else
              'goal-before-loss-v1' if args.task == 'attack' else SCHEMA['reward'])
    contract = dict(task=args.task, attack_frames=args.attack_frames, control_scope=args.control_scope,
                    decision_only=args.decision_only, gamma_unit='four-native-frames', reward=reward,
                    observation_clock='attack-seconds-remaining' if args.task != 'period' else 'game-clock',
                    lambda_unit='decision' if args.decision_only else 'four-native-frames')
    if args.task == 'opportunity':
        contract['reward_weights'] = REWARD_SPEC
    return contract


def evaluation_report(args, results, agents):
    if args.task != 'period':
        from nhl94_ai.evaluation.offense_attempts import attempt_report
        report = attempt_report(results, seed=args.seed)
        report['protocol'] = dict(attack_frames=args.attack_frames, policy_interval_frames=SCHEMA['interval'],
                                  model=args.model, task_transfer=args.allow_task_transfer,
                                  outcome_reward='goal=1; turnover/retreat/timeout/period-end=0',
                                  clock='remaining attack seconds', evaluation='deterministic')
        if args.task == 'opportunity':
            from nhl94_ai.evaluation.opportunity import opportunity_report
            report.update(opportunity_report(results, args.seed))
            report['protocol'].update(control_scope=args.control_scope, task_contract=training_contract(args),
                                      control_transfer=args.allow_control_transfer,
                                      outcome_reward='see task_contract.reward_weights')
        write_json(args.output, report)
        return report
    summary = {}
    for opponent in args.opponents:
        for agent in agents:
            rows = [row for row in results if row['agent'] == agent and row['opponent'] == opponent]
            complete = [row for row in rows if row['completed']]
            summary[f'{agent}:{opponent}'] = dict(
                periods=len(complete), incomplete=len(rows) - len(complete),
                goals_for=sum(row['goals_for'] for row in complete),
                goals_against=sum(row['goals_against'] for row in complete),
                wins=sum(row['goals_for'] > row['goals_against'] for row in complete),
                draws=sum(row['goals_for'] == row['goals_against'] for row in complete),
                losses=sum(row['goals_for'] < row['goals_against'] for row in complete))
    # Bootstrap ROM-seed clusters, keeping both team assignments together.
    paired = {}
    for agent in agents:
        if agent == 'classic':
            continue
        for opponent in args.opponents:
            base = {(r['state'], r['seed'], r['side']): r for r in results
                    if r['agent'] == 'classic' and r['opponent'] == opponent and r['completed']}
            clusters = {}
            for row in results:
                key = row['state'], row['seed'], row['side']
                if row['agent'] != agent or row['opponent'] != opponent or not row['completed'] or key not in base:
                    continue
                ref = base[key]
                delta = row['goals_for'] - row['goals_against'] - ref['goals_for'] + ref['goals_against']
                clusters.setdefault(row['seed'], []).append(delta)
            if clusters:
                means = np.asarray([np.mean(values) for values in clusters.values()])
                draws = np.random.default_rng(args.seed).choice(means, (10000, len(means))).mean(axis=1)
                paired[f'{agent}:{opponent}'] = dict(mean_delta=float(means.mean()),
                                                    seed_clusters=len(means),
                                                    interval95=np.quantile(draws, [0.025, 0.975]).tolist())
    report = dict(schema=SCHEMA, results=results, summary=summary, paired_vs_classic=paired)
    write_json(args.output, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect', 'bc', 'train', 'evaluate', 'play', 'audit'))
    parser.add_argument('--output', required=True, help='Dataset directory, checkpoint, or evaluation JSON')
    parser.add_argument('--model', help='Initial checkpoint; in collect, use learner states for DAgger labels')
    parser.add_argument('--dagger-beta', type=float, default=0.0, help='Probability of executing the teacher during DAgger collection')
    parser.add_argument('--data', nargs='+', default=[], help='Demonstration shards/directories')
    parser.add_argument('--states', nargs='+', default=['MightyDucksVsAllStarCampbell.ManualGoalie.Start'])
    parser.add_argument('--opponents', nargs='+', default=['cpu'], help='cpu, classic, or frozen checkpoint paths')
    parser.add_argument('--goalie-policy', choices=('off', 'selective', 'always'), default='selective')
    parser.add_argument('--side', type=int, choices=(1, 2))
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=41000)
    parser.add_argument('--split-seed', type=int, default=94, help='Fixed data split, independent of training seed')
    parser.add_argument('--architecture', choices=('gru', 'mlp', 'receiver-gru'), default='gru')
    parser.add_argument('--hidden-dim', type=int, default=128, help='MLP/GRU extractor width for a new model')
    parser.add_argument('--features-dim', type=int, default=256, help='Extractor output width for a new model')
    parser.add_argument('--head-dim', type=int, default=128, help='Actor and critic hidden width for a new model')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu', help='PPO training device')
    parser.add_argument('--scratch', action='store_true', help='Require random initialization without a parent or imitation data')
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Save initial, post-update and final policies; interval rounded up to a full rollout (0 disables)')
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--agents', nargs='+', choices=('model', 'teacher', 'classic', 'shoot', 'coast', 'forward'),
                        help='Evaluation arms; shoot is an immediate wrist-shot baseline')
    parser.add_argument('--max-frames', type=int, help='Diagnostic truncation; use complete periods for comparisons')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0, help='AdamW regularization during imitation')
    parser.add_argument('--mirror-augmentation', action='store_true', help='Reflect training examples across the rink center line')
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--rollout-steps', type=int, default=1024)
    parser.add_argument('--timesteps', type=int, default=1000000)
    parser.add_argument('--decision-only', action='store_true', help='PPO steps count offensive decisions with elapsed-time discounting')
    parser.add_argument('--task', choices=('period', 'attack', 'opportunity'), default='period')
    parser.add_argument('--control-scope', choices=('offense', 'carry'), default='offense',
                        help='Carry masks pass/shot actions and delegates all finishing decisions to Classic')
    parser.add_argument('--allow-control-transfer', action='store_true',
                        help='Explicitly adapt/evaluate weights under a different control scope')
    parser.add_argument('--attack-frames', type=int, default=600, help='Attack task horizon at 60 FPS; goals score 1, other endings 0')
    parser.add_argument('--reset-value', action='store_true', help='Reset the value output when adapting to a new reward objective')
    parser.add_argument('--allow-task-transfer', action='store_true',
                        help='Explicitly evaluate weights under a different task/clock contract')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--self-play', action='store_true', help='Add frozen historical policies to the opponent pool')
    parser.add_argument('--snapshot-every', type=int, default=50000)
    parser.add_argument('--max-snapshots', type=int, default=8)
    parser.add_argument('--save-starts', type=int, default=0, help='Save this many trusted local possession snapshots per collected period')
    parser.add_argument('--audit-per-action', type=int, default=32, help='Native execution branches per action')
    parser.add_argument('--starts', nargs='+', default=[])
    parser.add_argument('--start-probability', type=float, default=0.0)
    return parser


def run(args):
    if getattr(args, 'config', None):
        base = Path(args.config).expanduser().resolve().parent
        args.data = [str(base / Path(path).expanduser()) for path in args.data]
        args.starts = [str(base / Path(path).expanduser()) for path in args.starts]
        args.opponents = [name if name in ('cpu', 'classic') else str(base / Path(name).expanduser())
                          for name in args.opponents]
    if min(args.episodes, args.epochs, args.batch_size, args.rollout_steps, args.timesteps,
           args.num_envs, args.threads, args.snapshot_every, args.max_snapshots,
           args.hidden_dim, args.features_dim, args.head_dim) < 1:
        raise ValueError('Counts must be positive')
    if args.checkpoint_every < 0:
        raise ValueError('Checkpoint interval must be nonnegative')
    if args.scratch and (args.phase != 'train' or args.model or args.data):
        raise ValueError('--scratch requires PPO training without --model or --data')
    if args.device != 'cpu' and args.phase != 'train':
        raise ValueError('--device applies to PPO training; other phases use CPU')
    if not 0 < args.gamma < 1 or args.learning_rate <= 0 or args.save_starts < 0 or args.weight_decay < 0:
        raise ValueError('Invalid learning settings')
    if not 0 <= args.dagger_beta <= 1 or args.audit_per_action < 1:
        raise ValueError('DAgger beta must be in [0, 1] and audit counts positive')
    if not 0 <= args.seed < 2**32 - args.episodes or args.batch_size < 2 or args.rollout_steps < 2:
        raise ValueError('Use a uint32 seed range and batch/rollout sizes of at least two')
    if args.phase == 'bc' and not args.data:
        raise ValueError('BC requires --data')
    if args.task != 'period' and (not args.starts or args.phase not in ('train', 'evaluate')):
        raise ValueError('Attack tasks require --starts and train/evaluate mode')
    if args.task != 'period' and args.self_play:
        raise ValueError('Use fixed opponents for snapshot-based attack tasks')
    if args.task == 'opportunity' and args.control_scope != 'carry':
        raise ValueError('Use --control-scope carry for the opportunity task')
    if args.control_scope == 'carry' and (args.phase not in ('train', 'evaluate', 'play') or args.self_play or args.data):
        raise ValueError('Carry currently supports PPO/evaluation/play with fixed opponents and no auxiliary imitation')
    if args.decision_only and args.phase != 'train':
        raise ValueError('--decision-only applies to PPO training')
    if args.model and args.phase in ('bc', 'train') and model_path(args.model) == model_path(args.output):
        raise ValueError('Write the new checkpoint to a different path from its frozen parent')
    torch.set_num_threads(args.threads)
    if args.phase == 'audit':
        from nhl94_ai.evaluation.offense_audit import audit
        return audit(args)
    return {'collect': collect, 'bc': clone, 'train': train,
            'evaluate': evaluate, 'play': evaluate}[args.phase](args)
