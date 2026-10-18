"""Possession-balanced imitation and metrics that expose missing finishers."""
import numpy as np
import torch

from nhl94_ai.agents.learned_offense import N_ACTIONS, ENTITY_FIELDS, CONTEXT_FIELDS


GROUPS = ('movement', 'pass', 'one_timer', 'shot')
ACTION_GROUP = np.asarray([0] * 9 + [1] * 4 + [2] * 4 + [3] * 2)
MIRROR_ACTION = (0, 1, 8, 7, 6, 5, 4, 3, 2, *range(9, N_ACTIONS))


def mirror_examples(observations, actions):
    """Reflect rink geometry, handedness, input history and movement labels.

    Training augmentation only; receiver identities and source splits stay put.
    """
    result = observations.clone()
    width = len(ENTITY_FIELDS)
    for entity in range(12):
        for field in ('x', 'vx', 'rel_puck_x', 'facing_x'):
            result[..., entity * width + ENTITY_FIELDS.index(field)] *= -1
        hand = entity * width + ENTITY_FIELDS.index('handedness')
        result[..., hand] = 1 - result[..., hand]
    base = 12 * width
    for field in ('puck_x', 'puck_vx'):
        result[..., base + CONTEXT_FIELDS.index(field)] *= -1
    left, right = (base + CONTEXT_FIELDS.index(field) for field in ('left', 'right'))
    result[..., left], result[..., right] = observations[..., right], observations[..., left]
    permutation = torch.as_tensor(MIRROR_ACTION, device=observations.device)
    result[..., -N_ACTIONS:] = observations[..., -N_ACTIONS:][..., permutation]
    return result, permutation[actions]


def sampling_weights(actions, sources):
    """Equal intent mass; equal source-game mass inside each intent.

    Receiver classes within a game keep their observed frequency. Balancing
    rare single examples individually would repeatedly memorize them.
    """
    groups = ACTION_GROUP[actions]
    weights = np.zeros(len(actions), dtype=np.float64)
    for group in np.unique(groups):
        group_indices = np.flatnonzero(groups == group)
        _, inverse, counts = np.unique(sources[group_indices], return_inverse=True, return_counts=True)
        weights[group_indices] = 1 / (len(counts) * counts[inverse])
    return weights / weights.sum()


def imitation_metrics(labels, predictions, losses, train_counts):
    """Natural-distribution precision plus recall on each tactical group."""
    actual, predicted = ACTION_GROUP[labels], ACTION_GROUP[predictions]
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (actual, predicted), 1)
    groups = {}
    for i, name in enumerate(GROUPS):
        indices = actual == i
        count, selected, correct = int(indices.sum()), int((predicted == i).sum()), int(confusion[i, i])
        precision = correct / max(selected, 1)
        recall = correct / max(count, 1)
        groups[name] = dict(support=count, predicted=selected, precision=precision, recall=recall,
                           exact_accuracy=float(np.mean(labels[indices] == predictions[indices])) if count else 0.0,
                           f1=2 * precision * recall / max(precision + recall, 1e-12),
                           nll=float(np.mean(losses[indices])) if count else None)
    reasons = []
    for i, name in enumerate(GROUPS):
        item = groups[name]
        training = int(np.asarray(train_counts)[ACTION_GROUP == i].sum())
        if training < 50 or item['support'] < 10:
            reasons.append(f'{name}: fewer than 50 training / 10 validation examples')
        if item['exact_accuracy'] < (0.30 if i == 0 else 0.20):
            reasons.append(f'{name}: insufficient exact-action recall')
        if i and item['precision'] < 0.10:
            reasons.append(f'{name}: intent precision below 10%')
    return dict(groups=groups, confusion=confusion.tolist(),
                macro_f1=float(np.mean([g['f1'] for g in groups.values()])),
                balanced_nll=float(np.mean([g['nll'] for g in groups.values() if g['nll'] is not None])),
                quality_gate=dict(passed=not reasons, reasons=reasons),
                class_total=np.bincount(labels, minlength=N_ACTIONS).tolist(),
                class_correct=np.bincount(labels[labels == predictions], minlength=N_ACTIONS).tolist(),
                accuracy=float(np.mean(labels == predictions)), validation_nll=float(np.mean(losses)))


def checkpoint_rank(metrics):
    """A passing model beats a failed gate; then prefer all-intent F1."""
    return (metrics['quality_gate']['passed'], metrics['macro_f1'], -metrics['balanced_nll'])
