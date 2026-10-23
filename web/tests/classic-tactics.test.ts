import { describe, expect, it, vi } from "vitest";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { ClassicController } from "../classic/classic";
import { DEFAULT_CLASSIC_CONFIG } from "../classic/config";
import { DefenseController } from "../classic/defense";
import { ActionEnd, OneTimerLifecycle, PassLifecycle, PassState, ShotLifecycle } from "../classic/lifecycle";
import { carry_future, carry_projection, normal_shot_release_frames, ordinary_shot_conditions, projected_state } from "../classic/offense";
import { _SCENARIOS, evaluate_pass, one_timer_contact_frame, pad_direction, pass_contact, pass_release_frames, rom_pass_vector, selected_receiver } from "../classic/passing";
import { PossessionOffenseController } from "../classic/possession";
import { passive_reception, reception_target } from "../classic/receiving";
import { Buttons, GameState, Player, neutralAction, type Action, type Vec2 } from "../classic/state";

interface TraceRow {
  action: Action;
  decision: string;
  target: Vec2;
  owner: string;
  frames: number;
  decisions: number;
  remaining: number;
  shot_until: number;
  one_timer_retry: number;
  one_timer_metrics: Record<string, number>;
  pass_outcome: string | null;
}
const root = resolve(import.meta.dirname, "../..");
const python = process.env.NHL94_PYTHON ?? resolve(root, "../stable-retro/venv/bin/python");
const TRACE_REFERENCE = String.raw`
import json, sys
from types import SimpleNamespace
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.game.state import NHL94GameState, Player

def player(values):
    p = Player()
    for key, value in values.items():
        setattr(p, key, value)
    return p

def state(values):
    s = NHL94GameState(5)
    for side in ('team1', 'team2'):
        t, data = getattr(s, side), values[side]
        for key, value in data.items():
            if key == 'players':
                t.players = [player(p) for p in value]
            elif key == 'goalie':
                t.goalie = player(value)
            elif key in ('net', 'stats', 'last_stats', 'goalie_stats'):
                for field, content in value.items():
                    setattr(getattr(t, key), field, content)
            else:
                setattr(t, key, value)
    s.puck = player(values['puck'])
    for key, value in values['engine'].items():
        setattr(s.engine, key, value)
    return s

out = []
for scenario in json.load(sys.stdin):
    c = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', goalie_policy='off'))
    rows = []
    for values in scenario:
        action = c.predict_frame(state(values), 4)[0].tolist()
        rows.append(dict(action=action, decision=c._last_decision, target=c._last_target,
                         owner=c.input_owner, frames=c.scheduler.frames, decisions=c.scheduler.decisions,
                         remaining=c.scheduler.remaining, shot_until=c.shot.until_decision,
                         one_timer_retry=c.one_timer.retry_at_frame, one_timer_metrics=c.one_timer.metrics.copy(),
                         pass_outcome=c.offense.pass_action.last_pass.get('outcome') if c.offense.pass_action.last_pass else None))
    out.append(rows)
print(json.dumps(out))
`;

// These fixtures and numerical expectations mirror the Python unit fixtures and
// were checked against the current classic_v1.py, without changing the oracle.
function offense_state(): GameState {
  const state = new GameState();
  state.team1.net.y = -264; state.team2.net.y = 264;
  state.team1.control = 1; state.team1.defense_control = 0;
  state.engine.puck_owner = 0; state.engine.puck_owner_known = true; state.engine.offsides_enabled = false;
  state.team1.pass_attempts = 0;
  for (const team of [state.team1, state.team2]) {
    for (const player of [...team.players, team.goalie]) Object.assign(player, {
      motion_x: 0, motion_y: 0, speed: 20, agility: 20, weight: 64, passing: 20, energy: 4096,
      stick: 20, burst_full_energy: false, role: 4, facing: 0, selection_flags: 0, shot_accuracy: 20,
    });
    team.goalie.role = 0;
  }
  const own: Vec2[] = [[0, -130], [-80, -30], [100, -210], [110, -240], [70, -220]];
  const other: Vec2[] = [[40, -75], [110, 80], [110, 140], [90, 180], [100, 220]];
  state.team1.players.forEach((player, index) => [player.x, player.y] = own[index]!);
  state.team2.players.forEach((player, index) => [player.x, player.y] = other[index]!);
  state.team1.players[0]!.selection_flags = 8;
  state.team1.goalie.y = -250; state.team2.goalie.y = 250;
  Object.assign(state.puck, { x: 0, y: -120, motion_x: 0, motion_y: 0, height: 0 });
  return state;
}

function shooting_state(): GameState {
  const state = new GameState();
  state.team1.control = state.team2.control = 1;
  state.team1.net.y = -264; state.team2.net.y = 264;
  state.team1.players[0]!.x = 30; state.team1.players[0]!.y = 225;
  state.team2.players[0]!.x = -30; state.team2.players[0]!.y = -225;
  state.team2.goalie.x = 10; state.team1.goalie.x = -10;
  state.engine.puck_owner = 0;
  return state;
}

function one_timer_state(): GameState {
  const state = offense_state(), passer = state.team1.players[0]!, receiver = state.team1.players[1]!;
  Object.assign(passer, { x: -50, y: 180, facing: 1 });
  Object.assign(receiver, { x: 35, y: 160, motion_y: 0.7 });
  state.puck.x = -50; state.puck.y = 190;
  for (const other of [...state.team1.players.slice(2), ...state.team2.players]) { other.x = 100; other.y = -180; }
  return state;
}

function reception_state(): GameState {
  const state = offense_state();
  Object.assign(state.team1.players[0]!, {
    x: -32, y: -171, motion_x: -1.7776641845703125, motion_y: 0.8031005859375,
    precise_x: -31.316268920898438, precise_y: -170.67503356933594,
    stick_x: 1, stick_y: 7, facing: 0, facing_phase: 0.074951171875,
    selection_flags: 65, assignment: 19, decision_timer: 20, speed: 15, agility: 15,
    weight: 56, energy: 4096, role: 5, unavailable: 2,
  });
  Object.assign(state.puck, { x: -95, y: -55, motion_x: 0.070037841796875, motion_y: -3.238861083984375,
    height: 0.6746978759765625, motion_z: 0.1984405517578125, friction: 0.998046875 });
  state.team1.defense_control = 3; state.engine.puck_owner = -247;
  return state;
}

function controller(): ClassicController {
  return new ClassicController({ ...DEFAULT_CLASSIC_CONFIG, ai_goalie: false });
}
function mask(action: Action): number { return action.reduce((value, button, index) => value | button << index, 0); }
function waiting_model(): ClassicController {
  const model = controller();
  model.scheduler.frames = 5;
  const pending = new PassState(3, 0, 0, 50, "advance-pass", [-94, -136], 0);
  pending.actual_receiver = 0; pending.launched = pending.flight_observed = true;
  model.offense.pass_action.pending = pending;
  return model;
}

describe("Classic baseline passing oracle", () => {
  it("embeds the exact NumPy common scenarios", () => {
    expect(_SCENARIOS).toEqual([
      [-0.5206581488975943, 0.577386089588275], [0.44408910178056105, -0.9084440085604344],
      [-0.34108614078236954, -0.4757733928037846], [0.12854701686641157, 0.7736261930402584],
      [-0.3888674357851327, 0.9442264172759047], [-0.002655188257643326, 0.6164595111044717],
      [0.7465069468200407, -0.7183455211468097], [-0.0529619730655031, -0.3083051478291041],
    ]);
  });
  it("matches moving contact, rating speed, margin, value and ROM recipient tie order", () => {
    const state = offense_state(), passer = state.team1.players[0]!, receiver = state.team1.players[1]!;
    const [option, details] = evaluate_pass(state, passer, 1, receiver);
    expect(details.status).toBe("safe");
    expect(option).not.toBeNull();
    expect(option!.point).toEqual([-80, -20]);
    expect(option!.flight_frames).toBeCloseTo(36.158819223150196, 12);
    expect(option!.margin).toBeCloseTo(35.362382919017584, 10);
    expect([option!.direction, option!.robustness, option!.forward_gain, option!.bypassed, option!.value]).toEqual([7, 1, 110, 1, 80]);
    expect(option!.contact![3]).toBeCloseTo(3.541666666666667, 14);
    const direction = pad_direction(passer, receiver);
    state.team1.players[2]!.x = receiver.x; state.team1.players[2]!.y = receiver.y;
    expect(selected_receiver(state.team1, state.puck, 0, direction)).toBe(2);
  });
  it("preserves signed word launch arithmetic and overflow rejection", () => {
    expect(rom_pass_vector(80, 90, 0, 0, 20)).toEqual([10922, 12288]);
    expect(rom_pass_vector(-100, 10, -1000, 4000, 29)).toEqual([-19114, 5643]);
    expect(rom_pass_vector(4, -10, 20000, -50000, 0)).toBeNull();
  });
  it("rejects live moving interception and never invents missing feedback", () => {
    const state = offense_state(), passer = state.team1.players[0]!, receiver = state.team1.players[1]!;
    Object.assign(state.team2.players[0]!, { x: 0, y: -70, motion_x: -2 });
    expect(evaluate_pass(state, passer, 1, receiver)[1].status).toBe("moving-interception");
    receiver.motion_x = null;
    expect(evaluate_pass(state, passer, 1, receiver)[1].status).toBe("missing-feedback");
  });
  it("requires the first stick/body contact to follow the tactical cue", () => {
    const state = one_timer_state(), passer = state.team1.players[0]!, receiver = state.team1.players[1]!;
    const contact = pass_contact(state.puck, passer, receiver)!;
    const first = one_timer_contact_frame(state.puck, receiver, contact, pass_release_frames(receiver));
    expect(first).toBeGreaterThan(4);
    expect(first).toBeLessThan(contact[2] + 4);
    const [option, details] = evaluate_pass(state, passer, 1, receiver, "one-timer");
    expect(option).not.toBeNull(); expect(details.cue_frame).toBe(4);
  });
  it("explicitly rejects unsupported tactical profiles", () => {
    const state = offense_state();
    expect(() => evaluate_pass(state, state.team1.players[0]!, 1, state.team1.players[1]!, "advance", { rank_risk: true })).toThrow(/unsupported|not supported/);
    expect(() => new PossessionOffenseController({ chance_creation: true })).toThrow(/unsupported/);
    expect(() => new ClassicController({ ...DEFAULT_CLASSIC_CONFIG, offense_lookahead: true } as typeof DEFAULT_CLASSIC_CONFIG)).toThrow(/supported/);
  });
});

describe("Classic baseline carrying and normal shots", () => {
  it("matches conservative sampled skating and leaves source snapshots unchanged", () => {
    const state = offense_state(), player = state.team1.players[0]!, before = { ...player };
    const motion = carry_projection(player, [35, 235])!;
    expect(motion[0][0]).toBeCloseTo(2.8659145979915652, 12);
    expect(motion[0][1]).toBeCloseTo(-127.13408540200842, 12);
    expect(motion[1][0]).toBeCloseTo(0.428544418471175, 14);
    expect(player).toEqual(before);
    state.team2.players[0]!.role = -1;
    state.team2.players[0]!.x = 119; state.team2.players[0]!.motion_x = 4;
    expect(projected_state(state, [-20, -80])!.team2.players[0]!.x).toBe(119);
  });
  it("holds C four native frames then releases without changing shot aim", () => {
    const state = shooting_state(), model = controller(), masks: number[] = [];
    for (let index = 0; index < 13; index++) {
      if (index === 1) state.team2.goalie.x = -10;
      masks.push(mask(model.predict_frame(state)));
    }
    expect(masks).toEqual([320, 320, 320, 320, 64, 64, 64, 64, 64, 64, 64, 64, 64]);
    expect([model.scheduler.frames, model.scheduler.decisions, model.scheduler.remaining]).toEqual([13, 4, 3]);
    expect(model._last_decision).toBe("shot-follow-through");
  });
  it("retains four-frame shot holds and tactical budgets when planning every native frame", () => {
    const state = shooting_state(), model = controller();
    const masks = Array.from({ length: 13 }, () => mask(model.predict_frame(state, 1)));
    expect(masks).toEqual([320, 320, 320, 320, 64, 64, 64, 64, 64, 64, 64, 64, 64]);
    expect([model.scheduler.frames, model.scheduler.decisions, model.scheduler.remaining]).toEqual([13, 13, 0]);
    expect(model.shot.until_decision).toBe(29);
    expect(model.scheduler.duration(6)).toBe(24);
    expect(model.scheduler.duration(48)).toBe(192);
    expect(model.offense.shot_hold_frames).toBe(4);
  });
  it("keeps forecast integer and precise coordinates coherent without changing live bodies", () => {
    const state = offense_state();
    Object.assign(state.team2.players[0]!, { precise_x: 40.25, precise_y: -74.5, motion_x: 1 });
    Object.assign(state.puck, { precise_x: 0.25, precise_y: -119.5 });
    const before = JSON.stringify(state), future = projected_state(state, [-20, -80])!;
    for (const player of [future.team1.players[0]!, future.team2.players[0]!, future.puck]) {
      expect([player.precise_x, player.precise_y]).toEqual([player.x, player.y]);
    }
    const carrier = Object.assign(new Player(), state.team1.players[0], { x: 15, y: -90 });
    const carried = carry_future(state, state.team1.players[0]!, carrier);
    expect(carried.team1.players[0]).toBe(carrier);
    expect([carried.puck.precise_x, carried.puck.precise_y]).toEqual([carried.puck.x, carried.puck.y]);
    expect(JSON.stringify(state)).toBe(before);
  });
  it("retains the baseline same-shooter rebound window and cancels for teammate recovery", () => {
    const state = shooting_state(), model = controller();
    model.predict_frame(state);
    state.engine.puck_owner = -1; state.engine.shot_player = 0; state.team1.stats.shots++;
    for (let index = 0; index < 8; index++) model.predict_frame(state);
    state.engine.puck_owner = 0;
    expect(model.predict_frame(state)[Buttons.INPUT_C]).toBe(0);
    expect(model._last_decision).toBe("shot-follow-through");
    state.engine.puck_owner = 1;
    expect(model.shot.observe(state, model.scheduler.decisions)).toBe(true);
    expect(model.shot.until_decision).toBe(0);
  });
  it("gates the shot by accuracy, net-relative release geometry and a fresh C edge", () => {
    const state = shooting_state(), player = state.team1.players[0]!;
    player.y = 220; player.shot_accuracy = 0;
    expect(ordinary_shot_conditions(state, player, 4, null)[0]).toBe(false);
    player.shot_accuracy = 30;
    expect(ordinary_shot_conditions(state, player, 4, null)[0]).toBe(true);
    expect(normal_shot_release_frames(player, 4)).toBeGreaterThan(4);
    const model = controller(); model.buttons.c_down = true;
    expect(model.predict_frame(state)[Buttons.INPUT_C]).toBe(0);
    for (let index = 0; index < 3; index++) model.predict_frame(state);
    expect(model.predict_frame(state)[Buttons.INPUT_C]).toBe(1);
  });
});

describe("Classic pass and one-timer lifecycle precedence", () => {
  it("holds ordinary passes four frames before releasing B", () => {
    const state = offense_state(), model = controller(), masks: number[] = [];
    for (let index = 0; index < 13; index++) masks.push(mask(model.predict_frame(state)));
    expect(masks).toEqual([81, 81, 81, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(model._last_decision).toBe("pass-release");
    expect(model.offense.pass_action.pending!.deadline).toBe(58);
    expect(model.offense.pass_at).toBe(25);
  });
  it("prioritizes deadline before simultaneous ordinary reception and completes once", () => {
    const state = offense_state();
    state.engine.puck_owner = 1;
    for (const [frame, outcome] of [[19, "received"], [20, "flight-timeout"], [21, "flight-timeout"]] as const) {
      const lifecycle = new PassLifecycle(), pending = new PassState(0, 1, 0, 20);
      pending.launched = pending.flight_observed = true; lifecycle.pending = pending;
      expect(lifecycle.observe(state, frame)?.reason).toBe(outcome);
      const snapshot = lifecycle.last_pass;
      expect(lifecycle.finish(state, frame + 1, "duplicate")).toBeNull();
      expect(lifecycle.last_pass).toBe(snapshot);
    }
  });
  it("observes native release without fabricating a receiver from controller selection", () => {
    const state = offense_state(), lifecycle = new PassLifecycle();
    lifecycle.pending = new PassState(0, 1, 1, 60, "advance-pass", [-80, -20], 0);
    state.engine.puck_owner = -1; state.engine.last_puck_player = 0;
    lifecycle.observe(state, 2);
    expect(lifecycle.pending!.launched).toBe(true); expect(lifecycle.pending!.actual_receiver).toBeNull();
    state.team1.pass_attempts = 1; state.engine.pass_target = 2;
    lifecycle.observe(state, 3);
    expect(lifecycle.pending!.actual_receiver).toBe(2);
  });
  it("matches one-timer C cue holds, release edges and fixed retry clock", () => {
    const state = one_timer_state(), model = controller(), masks: number[] = [];
    for (let index = 0; index < 13; index++) {
      if (index === 2) {
        state.engine.puck_owner = -1; state.engine.last_puck_player = 0; state.engine.pass_target = 1; state.team1.pass_attempts = 1;
      }
      masks.push(mask(model.predict_frame(state)));
    }
    expect(masks).toEqual([129, 129, 129, 129, 256, 256, 256, 256, 0, 0, 0, 0, 256]);
    expect(model.one_timer.retry_at_frame).toBe(97); expect(model.one_timer.starts).toBe(1);
    expect(model.one_timer.pending!.actual_receiver).toBe(1);
  });
  it("prioritizes one-timer shot confirmation over recovery and timeout, but stoppage over all", () => {
    const state = offense_state(), lifecycle = new OneTimerLifecycle();
    lifecycle.start(state, 1, 5); lifecycle.pending!.launched = true;
    const deadline = lifecycle.pending!.deadline;
    state.engine.shot_player = 1; state.team1.stats.shots++;
    expect(lifecycle.observe(state, deadline)?.reason).toBe("shot-released");
    expect(lifecycle.finish("timeout")).toBeNull(); expect(lifecycle.metrics).toEqual({ "shot-released": 1 });
    lifecycle.start(state, 1, 6); state.engine.clock_stopped = true; state.team1.stats.shots++;
    const end = lifecycle.observe(state, deadline)!;
    expect([end.reason, end.neutral_frame]).toEqual(["play-stopped", true]);
  });
  it("rejects mismatched native recipients and releases an unavailable receiver", () => {
    const state = offense_state(), lifecycle = new OneTimerLifecycle();
    lifecycle.start(state, 1, 1);
    state.team1.pass_attempts = 1; state.engine.pass_target = 2;
    expect((lifecycle.step(state, 5, {}) as ActionEnd).reason).toBe("receiver-mismatch");
    lifecycle.start(state, 1, 6); state.team1.players[1]!.is_falling = 1;
    expect((lifecycle.step(state, 8, {}) as ActionEnd).reason).toBe("receiver-unavailable");
  });
});

describe("Classic native-frame reception and scheduler", () => {
  it("matches the recorded marginal reception forecast and target", () => {
    const state = reception_state(), receiver = state.team1.players[0]!, forecast = passive_reception(state, receiver)!;
    expect(forecast.frames).toBe(29); expect(forecast.gap).toBeCloseTo(0.8619550207579927, 12);
    expect(forecast.point[0]).toBeCloseTo(-92.97218213899555, 12);
    expect(forecast.point[1]).toBeCloseTo(-148.77531041667248, 12);
    const target = reception_target(state, receiver)!;
    expect(target[0]).toBeCloseTo(-91.6675607284657, 12); expect(target[1]).toBeCloseTo(-209.10680275695364, 12);
  });

  describe.skipIf(!existsSync(python))("Classic Python tactical trace parity", () => {
    it("matches 1,152 native inputs, both attacking ends, feedback transitions and all action clocks", () => {
      const cases: GameState[][] = [];
      for (let variation = 0; variation < 48; variation++) {
        const sequence: GameState[] = [];
        for (let frame = 0; frame < 24; frame++) {
          const state = variation % 3 === 0 ? one_timer_state() : variation % 3 === 1 ? shooting_state() : offense_state();
          const player = state.team1.players[0]!;
          if (variation % 3 === 1) { player.y = 190 + variation; player.shot_accuracy = variation % 4 === 0 ? 0 : variation % 4 === 1 ? 30 : 15; }
          if (variation % 3 === 2) { state.team2.players[0]!.x += variation - 24; state.team1.players[1]!.motion_x = (variation - 24) / 40; }
          if (variation % 4 === 0 && frame >= 2 && frame < 12) {
            state.engine.puck_owner = -1; state.engine.last_puck_player = 0; state.engine.pass_target = 1; state.team1.pass_attempts = 1;
          }
          if (variation % 4 === 0 && frame >= 12) {
            state.engine.puck_owner = 1; state.team1.control = 2; state.team1.defense_control = 1;
          }
          if (variation % 4 === 1 && frame >= 6 && frame < 14) {
            state.engine.puck_owner = 6; state.puck.x = -12; state.puck.y = 70;
          }
          if (variation % 4 === 2 && frame >= 8) {
            state.engine.puck_owner = -1; state.engine.shot_player = 0; state.team1.stats.shots = 1;
          }
          if (variation % 4 === 3 && frame === 9) state.engine.clock_stopped = true;
          if (variation % 8 === 0 && frame >= 19) state.engine.faceoff = true;
          if (variation >= 24) {
            state.team1.net.y *= -1; state.team2.net.y *= -1;
            for (const team of [state.team1, state.team2]) for (const p of [...team.players, team.goalie]) {
              p.x *= -1; p.y *= -1; p.vx *= -1; p.vy *= -1;
              if (p.motion_x !== null) p.motion_x *= -1;
              if (p.motion_y !== null) p.motion_y *= -1;
              p.facing = ((p.facing ?? p.orientation) + 4) % 8;
            }
            state.puck.x *= -1; state.puck.y *= -1;
            if (state.puck.motion_x !== null) state.puck.motion_x *= -1;
            if (state.puck.motion_y !== null) state.puck.motion_y *= -1;
          }
          sequence.push(state);
        }
        cases.push(sequence);
      }
      const result = spawnSync(python, ["-c", TRACE_REFERENCE], {
        input: JSON.stringify(cases), encoding: "utf8", timeout: 60000, maxBuffer: 4 * 1024 * 1024,
        env: { ...process.env, PYTHONPATH: `${resolve(root, "../stable-retro")}:${root}` },
      });
      if (result.error) throw result.error;
      expect(result.status, result.stderr).toBe(0);
      const expected = JSON.parse(result.stdout) as TraceRow[][];
      cases.forEach((sequence, scenario) => {
        const model = controller();
        sequence.forEach((state, frame) => {
          const action = model.predict_frame(state);
          const actual: TraceRow = {
            action, decision: model._last_decision, target: model._last_target, owner: model.input_owner,
            frames: model.scheduler.frames, decisions: model.scheduler.decisions, remaining: model.scheduler.remaining,
            shot_until: model.shot.until_decision, one_timer_retry: model.one_timer.retry_at_frame,
            one_timer_metrics: { ...model.one_timer.metrics }, pass_outcome: model.offense.pass_action.last_pass?.outcome ?? null,
          };
          const reference = expected[scenario]![frame]!, label = `scenario ${scenario}, frame ${frame}`;
          expect({ ...actual, target: undefined }, label).toEqual({ ...reference, target: undefined });
          expect(actual.target[0], label).toBeCloseTo(reference.target[0], 10);
          expect(actual.target[1], label).toBeCloseTo(reference.target[1], 10);
        });
      });
    }, 65000);
  });
  it("advances reception frames without tactical decisions and requires confirmed receiver selection", () => {
    const state = reception_state(), model = waiting_model();
    const likely = vi.spyOn(DefenseController, "_likely_switch").mockReturnValue(0);
    try {
      expect(mask(model.predict_frame(state))).toBe(1);
      expect(model._last_decision).toBe("receive-pass-switch");
      expect(mask(model.predict_frame(state))).toBe(0);
      state.team1.defense_control = 0;
      const action = model.predict_frame(state);
      expect(action.slice(4, 8).some(Boolean)).toBe(true);
      expect(action[Buttons.INPUT_B] || action[Buttons.INPUT_C]).toBe(0);
      expect([model.scheduler.frames, model.scheduler.decisions, model.defense.frames]).toEqual([8, 0, 8]);
    } finally { likely.mockRestore(); }
  });
  it("does not observe or mutate commitments or button edges on held frames", () => {
    const state = one_timer_state(), model = controller(), first = model.predict_frame(state);
    const pending = model.one_timer.pending, decisions = model.scheduler.decisions;
    for (let index = 0; index < 3; index++) model.advance_held_frame();
    expect(model.one_timer.pending).toBe(pending);
    expect(model.scheduler.decisions).toBe(decisions);
    expect(model.buttons.b_down).toBe(true);
    expect(model.scheduler.action).toEqual(first);
    expect([model.scheduler.frames, model.scheduler.remaining]).toEqual([4, 0]);
  });
  it("samples offense every four frames but observes defense every native frame", () => {
    const state = offense_state(), model = controller(), calls = vi.spyOn(model.defense, "step");
    model.predict_frame(state); state.engine.puck_owner = 6;
    model.predict_frame(state); model.predict_frame(state); model.predict_frame(state);
    expect(calls).toHaveBeenCalledTimes(3);
    expect(model.scheduler.decisions).toBe(1); expect(model.scheduler.frames).toBe(4);
    expect(model.offense.pass_action.pending).toBeNull();
  });
  it("uses faceoff pulses and resets per-instance commitments without sharing history", () => {
    const state = one_timer_state(), model = controller(), other = controller();
    model.predict_frame(state); state.engine.faceoff = true;
    const masks = Array.from({ length: 8 }, () => mask(model.predict_frame(state)));
    expect(masks).toEqual([1, 1, 0, 0, 0, 0, 1, 1]);
    expect(model.one_timer.pending).toBeNull(); expect(model.one_timer.metrics.stoppage).toBe(1);
    expect(other.one_timer.metrics).toEqual({}); expect(other.scheduler.frames).toBe(0);
    expect(new ShotLifecycle().active(0)).toBe(false);
    expect(neutralAction()).toHaveLength(12);
  });
});
