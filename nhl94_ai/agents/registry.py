"""Explicit agent registration without importing neural networks."""
from importlib import import_module
from nhl94_ai.agents.base import ScriptedAgent

CONTROLLERS = {
    'ClassicAIV1': ('nhl94_ai.agents.classic_v1', 'ClassicAIV1Model'),
}
ALIASES = {'classic': 'ClassicAIV1', 'classic-v1': 'ClassicAIV1'}


def create_scripted(name, args, env=None):
    if getattr(args, 'action_type', '').upper() == 'TARGET_POSITION':
        raise ValueError('ClassicAI agents output buttons/intents, not TARGET_POSITION coordinates')
    name = ALIASES.get(name, name)
    if name not in CONTROLLERS:
        raise ValueError(f'Unknown scripted agent: {name}')
    module, attribute = CONTROLLERS[name]
    return ScriptedAgent(getattr(import_module(module), attribute), args, env)
