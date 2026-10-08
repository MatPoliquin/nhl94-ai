"""Explicit agent registration without importing neural networks."""
from importlib import import_module
from nhl94_ai.agents.base import ScriptedAgent

CONTROLLERS = {
    'ClassicAIV1': ('nhl94_ai.agents.classic_v1', 'ClassicAIV1Model'),
}
ALIASES = {'classic': 'ClassicAIV1', 'classic-v1': 'ClassicAIV1'}


def add_classic_arguments(parser):
    parser.add_argument('--offense-lookahead', action='store_true',
                        help='Experimental native carry and receiver-continuation policy (default: off)')
    parser.add_argument('--cross-crease', action='store_true',
                        help='Opt-in full-team Classic held-C cross-crease finishing (default: off)')
    parser.add_argument('--deke', action='store_true',
                        help='Opt-in full-team Classic reaction-based skating dekes (default: off)')
    parser.add_argument('--uncertain-carry', action='store_true',
                        help='Experimental risk-assessed uncertified Classic carries (default: off)')
    parser.add_argument('--chance-creation', action='store_true',
                        help='Experimental action-conditioned Classic passing-window creation (default: off)')
    return parser


def create_scripted(name, args, env=None):
    if getattr(args, 'action_type', '').upper() == 'TARGET_POSITION':
        raise ValueError('ClassicAI agents output buttons/intents, not TARGET_POSITION coordinates')
    name = ALIASES.get(name, name)
    if name not in CONTROLLERS:
        raise ValueError(f'Unknown scripted agent: {name}')
    module, attribute = CONTROLLERS[name]
    return ScriptedAgent(getattr(import_module(module), attribute), args, env)
