"""Construct Classic components from validated options, without changing history."""
from nhl94_ai.agents.offense import OffenseController
from nhl94_ai.agents.possession import PossessionOffenseController
from nhl94_ai.agents.goalie import GoalieController


def create_offense(options):
    controller = OffenseController if options.offense_lookahead else PossessionOffenseController
    offense = controller(one_timers=options.one_timers, allow_uncertified=options.uncertain_carry,
                         chance_creation=options.chance_creation)
    offense.pass_timing = 'pass-timing' in options.classic_refinements
    offense.carry_motion = 'carry-motion' in options.classic_refinements
    offense.pass_action.stick_control = options.reception_control
    return offense


def create_goalie(options, platform, *, pass_timing):
    """Optional neural dependencies are imported only for a requested checkpoint."""
    policy, interval = options.goalie_policy, options.goalie_decision_interval
    if policy == 'off':
        return None
    if options.goalie_model:
        from nhl94_ai.agents.neural_goalie import NeuralGoalie
        return NeuralGoalie(policy, options.goalie_model, interval=interval, pass_timing=pass_timing)
    if interval and interval != 1:
        from nhl94_ai.agents.learned_goalie import SampledGoalie
        return SampledGoalie(policy, interval=interval, save_timing=options.goalie_save_timing,
                            pass_timing=pass_timing)
    if options.goalie_planner:
        if options.goalie_planner == 'outcome-24':
            from nhl94_ai.agents.goalie_outcome_planner import OutcomeGoalie
            return OutcomeGoalie(policy, pass_timing=pass_timing)
        from nhl94_ai.agents.goalie_planner import PlanningGoalie
        return PlanningGoalie(policy, planning=options.goalie_planner, pass_timing=pass_timing)
    if options.effective_goalie_positioning == 'deep':
        from nhl94_ai.agents.goalie_positioning import DeepGoalie
        return DeepGoalie(policy, pass_timing=pass_timing, save_timing=options.goalie_save_timing)
    if options.effective_goalie_positioning == 'deep-csave':
        from nhl94_ai.agents.goalie_csave import CSaveGoalie
        return CSaveGoalie(policy, pass_timing=pass_timing)
    return GoalieController(policy, pass_timing=pass_timing, save_timing=options.goalie_save_timing,
                            platform=platform)
