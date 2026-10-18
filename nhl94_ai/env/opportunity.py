"""Carry reward using Classic's admitted one-timers and native outcome counters.

Classic's shot value is a heuristic, not a scoring probability. No new pass
geometry, receiver ranking, or opportunity classifier is introduced here.
"""

REWARD_VERSION = 'classic-opportunity-carry-v1'
REWARD_SPEC = dict(setup_quality_scale=100.0, first_setup=0.5, first_release=0.5,
                   goal=2.0, conceded=-2.0, turnover=-1.0, retreat=-1.0,
                   native_frame_cost=0.001, potential_weight=0.25)


def option_quality(option):
    return min(1.0, max(0.0, option.shot_value / REWARD_SPEC['setup_quality_scale'])) if option else 0.0


def classic_opportunity(state, controller):
    """Use the exact admitted/ranked options behind Classic._one_timer_target."""
    if (not controller.available(state) or not controller._one_timers
            or controller.buttons.b_down or controller.scheduler.frames < controller.one_timer.retry_at_frame):
        return 0.0
    choices, _ = controller.offense.passes(state, 'one-timer')
    return option_quality(choices[0]) if choices else 0.0


class OpportunityReward:
    """One credit per setup/release per attempt; follow shots through resolution."""

    def __init__(self, state, controller, info, side, gamma):
        self.side, self.gamma = side, gamma
        self.potential = self.initial_quality = classic_opportunity(state, controller)
        self.peak_quality = self.initial_quality
        self.starts = controller.one_timer.starts
        self.initial_attempts = self.attempts = int(info[f'bench_one_timers{side}'])
        self.initial_goals = int(info[f'bench_one_timer_goals{side}'])
        self.initial_shots = self.shots = int(info[f'bench_shots{side}'])
        self.after_shot = False
        self.setup_count = self.confirmed_releases = 0
        self.setup_quality_sum = 0.0
        self.receiver = None
        self.last_request = None
        self.release_credited = False
        self.movement_decisions = self.eligible_decisions = 0

    def before_frame(self, controller):
        """Called after selecting input, before the native frame executes."""
        reward = 0.0
        request = controller._last_pass_request
        if request is not None and request != self.last_request:
            self.last_request = dict(request)
            self.receiver = None
        if controller.one_timer.starts > self.starts:
            quality = option_quality(controller.one_timer.option)
            self.setup_count += 1
            self.setup_quality_sum += quality
            self.peak_quality = max(self.peak_quality, quality)
            self.receiver = controller.one_timer.pending.receiver
            if self.setup_count == 1:
                reward = REWARD_SPEC['first_setup'] * quality
        self.starts = controller.one_timer.starts
        return reward

    def after_frame(self, info, outcome, *, own_skater=False):
        attempts = int(info[f'bench_one_timers{self.side}'])
        shots = int(info[f'bench_shots{self.side}'])
        reward = -REWARD_SPEC['native_frame_cost']
        if attempts > self.attempts and self.receiver is not None and info['shot_player'] == self.receiver:
            self.confirmed_releases += attempts - self.attempts
            if not self.release_credited:
                reward += REWARD_SPEC['first_release']
                self.release_credited = True
            self.receiver = None
        if attempts > self.attempts or shots > self.shots:
            self.after_shot = True
        elif own_skater:
            self.after_shot = False
        self.attempts, self.shots = attempts, shots
        # A goalie save/recovery after a recorded attempt is a shot resolution,
        # not a carrying turnover. Goal/concession priority comes from the env.
        if outcome == 'turnover' and self.after_shot:
            outcome = 'shot_ended'
        reward += REWARD_SPEC.get(outcome, 0.0)
        return reward, outcome

    def shape(self, state, controller, elapsed, terminal):
        next_potential = 0.0 if terminal else classic_opportunity(state, controller)
        self.peak_quality = max(self.peak_quality, next_potential)
        reward = REWARD_SPEC['potential_weight'] * (self.gamma**elapsed * next_potential - self.potential)
        self.potential = next_potential
        return reward

    def summary(self, info):
        return dict(initial_quality=self.initial_quality, peak_quality=self.peak_quality,
                    setups=self.setup_count, setup_quality_sum=self.setup_quality_sum,
                    confirmed_releases=self.confirmed_releases,
                    native_one_timers=int(info[f'bench_one_timers{self.side}']) - self.initial_attempts,
                    one_timer_goals=int(info[f'bench_one_timer_goals{self.side}']) - self.initial_goals,
                    recorded_shots=int(info[f'bench_shots{self.side}']) - self.initial_shots,
                    movement_decisions=self.movement_decisions, eligible_decisions=self.eligible_decisions)
