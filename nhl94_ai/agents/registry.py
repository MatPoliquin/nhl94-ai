"""Explicit agent registration without importing neural networks."""
from importlib import import_module
from nhl94_ai.agents.base import ScriptedAgent

CONTROLLERS = {
    'ClassicAIV1': ('nhl94_ai.agents.classic_v1', 'ClassicAIV1Model'),
}
ALIASES = {'classic': 'ClassicAIV1', 'classic-v1': 'ClassicAIV1'}


def add_classic_arguments(parser):
    parser.add_argument('--classic-control-interval', type=int, default=0, metavar='N',
                        help='Classic reaction/input cadence: 0 keeps native-frame reactions (default); '
                             'N >= 1 observes and holds raw buttons every N frames (4 = 15 Hz, 10 = 6 Hz, '
                             '20 = 3 Hz). Separate from the tactical frame-skip; FILTERED only.')
    parser.add_argument('--rebound-recovery', action='store_true',
                        help='Replan after a confirmed same-shooter rebound when the native animation unlocks (experimental; default: off)')
    parser.add_argument('--one-timer-execution', choices=('early-cue', 'release-retry'),
                        help='Experimental native one-timer cue or unlaunched-pass retry (FILTERED only; default: off)')
    parser.add_argument('--reception-control', action='store_true',
                        help='Experimental ordinary-pass correction using live stick contact and velocity-aware steering')
    parser.add_argument('--shot-placement', metavar='MODEL.json',
                        help='Aim/hold JSON model or fixed-hold control for existing normal-shot decisions (experimental)')
    parser.add_argument('--receiver-selection', choices=('legacy', 'value', 'all-safe'),
                        help='Rerank eligible receivers after a legacy pass decision (experimental; default: off)')
    parser.add_argument('--possession-ablation',
                        choices=('legacy', 'rank', 'continuations', 'risk', 'continuations-risk'),
                        help='Controlled value-ranking experiment over legacy proposals (default: off)')
    parser.add_argument('--possession-value', action='store_true',
                        help='Rank offensive actions with shared motion-aware possession utility (experimental)')
    parser.add_argument('--classic-refinements', nargs='+', default=[],
                        choices=('pass-timing', 'carry-motion', 'finishing', 'interceptions'),
                        help='Opt-in measured Classic experiments; enable individual components for comparison')
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
