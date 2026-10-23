import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { GameState, Player, Team, neutralAction, type Vec2 } from "../classic/state";
import { copySign, distance, lexCompare, mod, pointEqual, pyRound, required, shiftRight } from "../classic/numeric";
import { _clear_segment, aim_pass, project_target, route_waypoint, steering_direction } from "../classic/target";
import * as motion from "../classic/motion";
import { grounded_step, skating_route } from "../classic/skating";
import { DefenseController, defensive_steering } from "../classic/defense";
import { GoalieController, GoaliePlan, goalie_arrival, goalie_motion, goalie_steer, goalie_target, moving_body_contact, outlet_options } from "../classic/goalie";
import { DeepGoalie } from "../classic/goalie_positioning";
import { CSaveGoalie, contact_frame } from "../classic/goalie_csave";
import { KeeperMotion, PuckFlight, contact_margin, free_trajectory, friction, signed_word } from "../classic/goalie_physics";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const python = process.env.NHL94_PYTHON ?? resolve(root, "../stable-retro/venv/bin/python");
const pythonPath = `${resolve(root, "../stable-retro")}:${root}`;

// Run the unmodified Python helpers, not a second JavaScript implementation.
const REFERENCE = String.raw`
import json, math, sys
from dataclasses import asdict
from nhl94_ai.game.state import NHL94GameState, Player
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.env.target_control import project_target, route_waypoint, _clear_segment, steering_direction
from nhl94_ai.agents import motion
from nhl94_ai.agents.skating import grounded_step, skating_route
from nhl94_ai.agents.defense import DefenseController, defensive_steering
from nhl94_ai.agents.goalie import (
    GoalieController, GoaliePlan, goalie_target, goalie_motion, goalie_arrival,
    goalie_steer, moving_body_contact, outlet_options,
)
from nhl94_ai.agents.goalie_positioning import DeepGoalie
from nhl94_ai.agents.goalie_csave import CSaveGoalie, contact_frame
from nhl94_ai.agents.goalie_physics import (
    KeeperMotion, PuckFlight, contact_margin, free_trajectory, friction, signed_word,
)
import numpy as np

def player(data):
    p = Player()
    for key, value in data.items():
        setattr(p, key, value)
    return p

def state(data):
    s = NHL94GameState(5)
    for side in ('team1', 'team2'):
        t, source = getattr(s, side), data[side]
        for key in ('controller', 'control', 'defense_control', 'defense_goalie', 'pass_attempts'):
            setattr(t, key, source[key])
        t.players = [player(p) for p in source['players']]
        t.goalie = player(source['goalie'])
        t.net.y = source['net_y']
    s.puck = player(data['puck'])
    for key, value in data['engine'].items():
        setattr(s.engine, key, value)
    return s

def pose(p):
    return {name: getattr(p, name) for name in (
        'x', 'y', 'precise_x', 'precise_y', 'motion_x', 'motion_y', 'facing_phase', 'facing')}

def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value

request = json.load(sys.stdin)
op, cases = request['op'], request['cases']
result = []
for row in cases:
    if op == 'target':
        a, b = tuple(row[0]), tuple(row[1])
        action = np.ones(12, dtype=np.int8)
        aim_pass(action, Player(x=a[0], y=a[1]), Player(x=b[0], y=b[1]))
        result.append([project_target(a), _clear_segment(a, b), route_waypoint(a, project_target(b)), action])
    elif op == 'motion':
        p, target, pad = player(row['player']), tuple(row['target']), tuple(row['pad'])
        other = player(row['other'])
        result.append({
            'velocity': motion.velocity(p), 'coast': motion.coast_projection(p, 9),
            'bounded': motion.bounded_projection(p, 11), 'unbounded': motion.bounded_projection(p, 11, boards=False),
            'skating': motion.skating(p), 'optimistic': motion.skating(p, True),
            'facing': motion.facing(p), 'boost_impulse': motion.boost_impulse(p),
            'burst': motion.burst_velocity(p), 'boost_safe': motion.boost_safe(p, target),
            'stop': motion.stop_projection(p, 12), 'bounded_stop': motion.stop_projection(p, 12, bounded=True),
            'approach': motion.check_approach(p, other),
            'arrival': motion.arrival_time(p, target), 'boost_arrival': motion.arrival_time(p, target, boost=True),
            'optimistic_arrival': motion.arrival_time(p, target, optimistic=True),
            'block': motion.blocks_shot_lane(p, (other.x, other.y), target, 3),
            'skate_step': motion.skate_step(p, ((p.x,p.y),motion.velocity(p)), p.orientation, 3, pad, .08, 2.1),
            'steer': steering_direction(p, target), 'defensive': defensive_steering(p, target),
            'grounded': pose(grounded_step(p,pad)), 'keeper_grounded': pose(grounded_step(p,pad,goalie=True)),
            'braking': pose(grounded_step(p,pad,stop=True)),
            'route': [(t,pose(p)) for t,p in skating_route(p,target,8)],
            'puck_path': motion.puck_path(p,48,sample_interval=3),
        })
    elif op == 'physics':
        k, p = KeeperMotion(*row['keeper']), PuckFlight(*row['puck'])
        frames = []
        for pad in row['pads']:
            k, p = k.step(tuple(pad),save=row['save']), p.step()
            frames.append([asdict(k),asdict(p),contact_margin(k,p),contact_margin(k,p,one_timer=True)])
        result.append([frames,[asdict(p) for p in free_trajectory(p,12,row['sign'])]])
    elif op == 'defense':
        c = DefenseController()
        steps = []
        for snapshot in row:
            s = state(snapshot['state'])
            action = c.step(s, snapshot.get('elapsed',1), frame=snapshot.get('frame'))
            steps.append({'action':action, 'plan':asdict(c.plan) if c.plan else None,
                'diagnostics':c.diagnostics,'frames':c.frames,'switch_at':c.switch_at,'poke_at':c.poke_at,
                'boost_at':c.boost_at,'last_owner':c.last_owner})
        c.idle(7,frame=900)
        result.append([steps,{'frames':c.frames,'diagnostics':c.diagnostics,'plan':c.plan,'switch_at':c.switch_at,
            'poke_at':c.poke_at,'boost_at':c.boost_at,'last_owner':c.last_owner}])
    elif op == 'goalie':
        cls = {'classic':GoalieController,'deep':DeepGoalie,'csave':CSaveGoalie}[row['kind']]
        c = cls('selective')
        steps = []
        for snapshot in row['states']:
            s = state(snapshot['state'])
            action = c.step(s,blocked=snapshot.get('blocked',False))
            steps.append({'action':action,'diagnostics':c.diagnostics,'phase':c.phase,'frames':c.frames,
                'metrics':dict(c.metrics),'pending_save':c.pending_save,'save_at':c.save_at,'outlet':c.outlet,
                'release_until':c.release_until,'retry_at':c.retry_at,'pass_windup':c.pass_windup})
        result.append(steps)
    elif op == 'goalie_helpers':
        s = state(row['state'])
        action = np.asarray(row['action'],dtype=np.int8)
        plan = GoaliePlan(tuple(row['target']),'test',**row['plan'])
        c = CSaveGoalie('selective')
        options, details = outlet_options(s)
        result.append({'target':asdict(goalie_target(s)), 'motion':goalie_motion(s.team1.goalie),
            'arrival':goalie_arrival(s.team1.goalie,tuple(row['target'])),
            'steer':goalie_steer(s.team1.goalie,tuple(row['target'])),
            'contact':contact_frame(s,action,row['horizon']),
            'save_contact':contact_frame(s,action,row['horizon'],save=True),
            'moving_contact':moving_body_contact(s,action,row['horizon']),
            'moving_save_contact':moving_body_contact(s,action,row['horizon'],save=True),
            'keeper':asdict(KeeperMotion.observe(s.team1.goalie)),
            'puck':asdict(PuckFlight.observe(s.puck)),
            'position':c._position(s,plan),'position_diagnostics':c.diagnostics,
            'options':[asdict(o) for o in options],'option_details':details})
print(json.dumps(clean(result),allow_nan=False))
`;

function reference(op: string, cases: readonly unknown[]): unknown {
  const result = spawnSync(python, ["-c", REFERENCE], {
    cwd: root, input: JSON.stringify({ op, cases }), encoding: "utf8",
    env: { ...process.env, PYTHONPATH: pythonPath }, maxBuffer: 16 * 1024 * 1024,
  });
  if (result.error || result.status !== 0) throw new Error(`Python parity reference failed: ${result.error?.message ?? result.stderr}`);
  return JSON.parse(result.stdout);
}

function serial(value: unknown): unknown {
  return JSON.parse(JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "number" && !Number.isFinite(item) ? item === Infinity ? "inf" : item === -Infinity ? "-inf" : "nan" : item));
}

function parity(actual: unknown, expected: unknown, path = "root"): void {
  if (typeof actual === "number" && typeof expected === "number") {
    expect(Math.abs(actual - expected), path).toBeLessThanOrEqual(1e-10 * Math.max(1, Math.abs(expected)));
  } else if (Array.isArray(actual) && Array.isArray(expected)) {
    expect(actual.length, path).toBe(expected.length);
    actual.forEach((item: unknown, index) => parity(item, expected[index], `${path}[${index}]`));
  } else if (typeof actual === "object" && actual !== null && typeof expected === "object" && expected !== null) {
    expect(Object.keys(actual).sort(), path).toEqual(Object.keys(expected).sort());
    for (const key of Object.keys(actual)) {
      parity(Reflect.get(actual, key), Reflect.get(expected, key), `${path}.${key}`);
    }
  } else expect(actual, path).toEqual(expected);
}

function player(values: Partial<Player> = {}): Player {
  return Object.assign(new Player(), {
    speed: 20, agility: 20, weight: 64, energy: 4096, stick: 20, checking: 20, passing: 6, puck_control: 20,
    role: 4, facing: 0, motion_x: 0, motion_y: 0, height: 0, motion_z: 0, selection_flags: 0,
  }, values);
}

function defenseState(sign = -1, controller = 1): GameState {
  const state = new GameState();
  state.team1 = new Team(controller); state.team2 = new Team(controller === 1 ? 2 : 1);
  state.team1.net.y = sign * 264; state.team2.net.y = -sign * 264;
  state.team1.control = state.team2.control = 1;
  state.team1.defense_control = state.team1.skater_scnum_base();
  state.team2.defense_control = state.team2.skater_scnum_base();
  state.engine.puck_owner = state.team2.skater_scnum_base(); state.engine.puck_owner_known = true;
  for (const [team, y] of [[state.team1, sign * 215], [state.team2, -sign * 70]] as const) {
    team.players = Array.from({ length: 5 }, (_, index) => player({ x: -90 + index * 40, y }));
    team.defense_goalie = team.goalie_scnum();
    team.goalie = player({ x: 0, y: team.net.y > 0 ? 246 : -246, role: 0 });
  }
  state.team1.players[0]!.x = 0;
  Object.assign(state.team2.players[0]!, { x: 0, y: sign * 150 });
  state.puck = player({ x: 0, y: sign * 150 });
  return state;
}

function goalieState(selected = false, sign = -1, controller = 1): GameState {
  const state = defenseState(sign, controller), goalie = state.team1.goalie;
  Object.assign(state.engine, { goalie_modes: [0, 0], goalie_hold_counts: [17, 17],
    controller_teams: controller === 1 ? [1, 0] : [2, 0], input_block: 0, clock_stopped: false, camera: [0, sign * 200] });
  Object.assign(goalie, { live_anim: 2, live_state_flags: 0, selection_flags: selected ? 0x88 : 0x80, assignment: 14, stick: 8 });
  Object.assign(state.puck, { x: 10, y: sign * 120, motion_y: sign * 1.3 });
  Object.assign(state.team1.players[0]!, { x: -110, y: -sign * 40 });
  if (selected) state.team1.defense_control = state.team1.goalie_scnum();
  return state;
}

function encode(state: GameState): unknown {
  const team = (t: Team) => ({
    controller: t.controller, control: t.control, defense_control: t.defense_control, defense_goalie: t.defense_goalie,
    pass_attempts: t.pass_attempts, net_y: t.net.y, players: t.players, goalie: t.goalie,
  });
  return { team1: team(state.team1), team2: team(state.team2), puck: state.puck, engine: state.engine };
}
function pose(p: Player): unknown {
  return { x: p.x, y: p.y, precise_x: p.precise_x, precise_y: p.precise_y, motion_x: p.motion_x,
    motion_y: p.motion_y, facing_phase: p.facing_phase, facing: p.facing };
}

describe("Classic numeric contracts", () => {
  it("preserves Python half-to-even rounding, negative modulo and large arithmetic shifts", () => {
    expect([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5].map(pyRound)).toEqual([-4, -2, -2, 0, 0, 2, 2, 4]);
    expect(mod(-9, 8)).toBe(7);
    expect(mod(9, -8)).toBe(-7);
    expect(shiftRight(2 ** 34 + 1234, 16)).toBe(262144);
    expect(shiftRight(-(2 ** 34) - 1234, 16)).toBe(-262145);
    expect(copySign(28, -0)).toBe(-28);
    expect(pointEqual([0, -0], [0, 0])).toBe(true);
    expect(lexCompare([1, 2], [1, 2, 0])).toBeLessThan(0);
    expect(distance([3, 4], [0, 0])).toBe(5);
    expect(signed_word(65535)).toBe(-1);
    expect(friction(-1, 6)).toBe(0);
  });

  it("fails explicitly for unsupported certification and missing required native state", () => {
    expect(() => new DefenseController({ verify_interceptions: true })).toThrow("not included");
    expect(() => GoalieController.validate(new GameState())).toThrow("telemetry");
    expect(() => grounded_step(new Player(), [0, 1])).toThrow("agility");
    expect(PuckFlight.observe(new Player())).toBeNull();
    expect(KeeperMotion.observe(new Player())).toBeNull();
  });
});

describe.skipIf(!existsSync(python))("Classic unmodified Python helper parity (NHL94_PYTHON)", () => {
  it("matches target projection, net routes, tie ordering and eight-way aiming", () => {
    const points: Vec2[] = [[0, 0], [0, 270], [0, -270], [120, 270], [-120, -270], [-35, 260], [35, 260],
      [-35, -260], [35, -260], [-28, 254], [28, -254], [-32, 250], [32, -250], [0, 254], [-0, -270]];
    const cases = points.flatMap((a) => points.map((b) => [a, b] as const));
    const actual = cases.map(([a, b]) => {
      const action = neutralAction(); action.fill(1);
      aim_pass(action, { x: a[0], y: a[1] }, { x: b[0], y: b[1] });
      return [project_target(a), _clear_segment(a, b), route_waypoint(a, project_target(b)), action];
    });
    parity(serial(actual), reference("target", cases));
  });

  it("matches motion, native fixed-point turning/braking, routes and airborne puck samples", () => {
    const cases = Array.from({ length: 36 }, (_, index) => ({
      player: player({ x: index % 3 === 0 ? -30 : index - 18, y: index % 3 === 0 ? -250 : index * 9 - 160,
        precise_x: index % 4 === 0 ? -30.123456 : null, precise_y: index % 4 === 0 ? -250.375 : null,
        motion_x: (index % 9 - 4) * 12001 * motion.VELOCITY_SCALE, motion_y: (index % 7 - 3) * 8003 * motion.VELOCITY_SCALE,
        facing: index % 8, orientation: index % 8, facing_phase: index % 4 === 0 ? -index * 0.125 : null,
        sprite_flipped_x: !!(index % 2), speed: index % 31, agility: (index * 3) % 31, weight: 32 + index * 3,
        energy: index * 137, movement_bonus: index % 5, projection_uncertainty: index % 3,
        height: index % 4 === 0 ? 8 : 0, motion_z: index % 4 === 0 ? 0.4 : 0 }),
      other: player({ x: index - 5, y: index * 7 - 150, motion_x: -0.25, motion_y: 0.375 }),
      target: [index % 2 ? 40 : -40, index % 3 ? 240 : -260] as Vec2,
      pad: [index % 3 - 1, Math.floor(index / 3) % 3 - 1] as Vec2,
    }));
    const actual = cases.map(({ player: p, other, target, pad }) => ({
      velocity: motion.velocity(p), coast: motion.coast_projection(p, 9),
      bounded: motion.bounded_projection(p, 11), unbounded: motion.bounded_projection(p, 11, { boards: false }),
      skating: motion.skating(p), optimistic: motion.skating(p, true), facing: motion.facing(p),
      boost_impulse: motion.boost_impulse(p), burst: motion.burst_velocity(p), boost_safe: motion.boost_safe(p, target),
      stop: motion.stop_projection(p, 12), bounded_stop: motion.stop_projection(p, 12, { bounded: true }),
      approach: motion.check_approach(p, other), arrival: motion.arrival_time(p, target),
      boost_arrival: motion.arrival_time(p, target, { boost: true }), optimistic_arrival: motion.arrival_time(p, target, { optimistic: true }),
      block: motion.blocks_shot_lane(p, [other.x, other.y], target, 3),
      skate_step: motion.skate_step([[p.x, p.y], motion.velocity(p)], p.orientation, 3, pad, 0.08, 2.1),
      steer: steering_direction(p, target), defensive: defensive_steering(p, target),
      grounded: pose(grounded_step(p, pad)), keeper_grounded: pose(grounded_step(p, pad, { goalie: true })),
      braking: pose(grounded_step(p, pad, { stop: true })), route: [...skating_route(p, target, 8)].map(([time, future]) => [time, pose(future)]),
      puck_path: motion.puck_path(p, 48, { sample_interval: 3 }),
    }));
    parity(serial(actual), reference("motion", cases));
  });

  it("matches signed-word integer goalie physics without overflowing native coordinate squares", () => {
    const cases = Array.from({ length: 24 }, (_, index) => ({
      keeper: [(-36 + index * 3) * 65536 + 32767, (index % 2 ? -246 : 246) * 65536,
        index % 2 ? -32767 : 32767, index % 3 ? -1000 : 1000, 50, 8000 ** 2, index % 2 ? 6 : 9, false] as const,
      puck: [(index - 12) * 65536, (index % 2 ? -220 : 220) * 65536, index % 3 ? 0 : 10 * 65536,
        index % 2 ? 31000 : -32000, index % 2 ? -16000 : 16000, index % 3 ? 0 : -32000, 9] as const,
      pads: [[1, 0], [-1, -1], [0, 1], [0, 0], [1, 1], [-1, 0]] as Vec2[],
      save: !!(index % 2), sign: index % 2 ? -1 : 1,
    }));
    const actual = cases.map((row) => {
      let keeper = new KeeperMotion(...row.keeper), puck = new PuckFlight(...row.puck);
      const frames = row.pads.map((pad) => {
        keeper = keeper.step(pad, { save: row.save }); puck = puck.step();
        return [keeper, puck, contact_margin(keeper, puck), contact_margin(keeper, puck, { one_timer: true })];
      });
      return [frames, free_trajectory(puck, 12, row.sign)];
    });
    parity(serial(actual), reference("physics", cases));
  });

  it("matches both-side defensive plans, races, coverage, selection feedback and held-button clocks", () => {
    const states = Array.from({ length: 24 }, (_, index) => {
      const state = defenseState(index % 2 ? 1 : -1, index % 3 ? 1 : 2), sign = state.team1.net.y > 0 ? 1 : -1;
      if (index % 6 === 0) {
        state.engine.puck_owner = -256; Object.assign(state.puck, { x: 0, y: sign * 200 });
        for (const rival of state.team2.players) rival.y = -sign * 180;
      } else if (index % 6 === 1) {
        state.engine.puck_owner = -256; state.engine.last_puck_player = state.team2.skater_scnum_base();
        Object.assign(state.puck, { x: -60, y: sign * 160, motion_x: 3 });
        Object.assign(state.team1.players[0]!, { x: 0, y: sign * 160 });
        Object.assign(state.team2.players[0]!, { x: 80, y: sign * 160 });
      } else if (index % 6 === 2) {
        Object.assign(state.team2.players[0]!, { x: 100 }); state.puck.x = 100;
        Object.assign(state.team1.players[0]!, { x: 65, y: sign * 190 });
      } else if (index % 6 === 3) {
        state.engine.puck_owner = state.team2.goalie_scnum();
      } else if (index % 6 === 4) {
        state.puck.height = null; state.engine.puck_owner = -256; state.team1.defense_control = -1;
      } else {
        Object.assign(state.team1.players[0]!, { x: 0, y: sign * 174, facing: sign > 0 ? 4 : 0, selection_flags: 8, weight: 32 });
        Object.assign(state.team2.players[0]!, { motion_y: sign * 0.5, selection_flags: 0x40 });
        state.puck.y = sign * 160;
      }
      return state;
    });
    const cases = states.map((state) => Array.from({ length: 8 }, (_, index) => {
      const snapshot = { state: encode(state), elapsed: index % 2 ? 4 : 1, frame: index === 7 ? 100 : undefined };
      return { ...snapshot, live: state };
    }));
    const actual = cases.map((sequence) => {
      const controller = new DefenseController();
      const steps = sequence.map(({ live: state, ...snapshot }) => {
        const action = controller.step(state, { elapsed: snapshot.elapsed, frame: snapshot.frame });
        return serial({ action, plan: controller.plan, diagnostics: controller.diagnostics, frames: controller.frames,
          switch_at: controller.switch_at, poke_at: controller.poke_at, boost_at: controller.boost_at, last_owner: controller.last_owner });
      });
      controller.idle(7, { frame: 900 });
      return [steps, { frames: controller.frames, diagnostics: controller.diagnostics, plan: controller.plan,
        switch_at: controller.switch_at, poke_at: controller.poke_at, boost_at: controller.boost_at, last_owner: controller.last_owner }];
    });
    parity(serial(actual), reference("defense", cases.map((sequence) => sequence.map(({ live: _live, ...snapshot }) => snapshot))));
  });

  it("matches goalie crossing, outlet evaluation, deep positioning and C-save native contact checks", () => {
    const cases = Array.from({ length: 24 }, (_, index) => {
      const sign = index % 2 ? 1 : -1, state = goalieState(true, sign, index % 3 ? 1 : 2);
      state.engine.puck_owner = index % 4 ? -256 : state.team2.skater_scnum_base();
      Object.assign(state.puck, { x: index % 5 * 8 - 16, y: sign * (200 + index), motion_y: sign * 4,
        motion_x: index % 3 * 0.25, height: index % 4 * 2, motion_z: index % 3 ? 0 : 0.5 });
      Object.assign(state.team1.goalie, { x: index % 3 * 10 - 10, motion_x: index % 4 * 0.5 - 0.75, motion_y: 0.125 });
      state.engine.word_ffc2f8 = index % 3 ? 0 : 0x200;
      const target: Vec2 = [index % 4 * 10, sign * 246], action = goalie_steer(state.team1.goalie, target);
      const plan = { crossing_frames: index % 2 ? 4 : null, crossing_x: target[0], height: 3 };
      return { live: state, state: encode(state), action, target, plan, horizon: index % 4 + 6 };
    });
    const actual = cases.map(({ live: state, action, target, plan, horizon }) => {
      const controller = new CSaveGoalie(), [options, details] = outlet_options(state);
      const position = controller._position(state, new GoaliePlan(target, "test", plan));
      return { target: goalie_target(state), motion: goalie_motion(state.team1.goalie), arrival: goalie_arrival(state.team1.goalie, target),
        steer: goalie_steer(state.team1.goalie, target), contact: contact_frame(state, action, horizon),
        save_contact: contact_frame(state, action, horizon, { save: true }), moving_contact: moving_body_contact(state, action, horizon),
        moving_save_contact: moving_body_contact(state, action, horizon, { save: true }),
        keeper: KeeperMotion.observe(state.team1.goalie), puck: PuckFlight.observe(state.puck),
        position, position_diagnostics: controller.diagnostics, options, option_details: details };
    });
    parity(serial(actual), reference("goalie_helpers", cases.map(({ live: _live, ...row }) => row)));
  });

  it("matches manual goalie takeovers, edge releases, possession, saves, returns and stoppages frame-for-frame", () => {
    const cases = (["classic", "deep", "csave"] as const).flatMap((kind) => [-1, 1].map((sign) => {
      const states = Array.from({ length: 64 }, (_, index) => {
        const state = goalieState(index >= 5 && index < 48, sign);
        if (index >= 8 && index < 15) {
          state.engine.puck_owner = -256;
          Object.assign(state.puck, { x: 12, y: sign * 220, motion_y: sign * 4 });
        }
        if (index >= 10 && index < 13) { state.team1.goalie.live_anim = 0x250; state.team1.goalie.selection_flags = 0xa8; }
        if (index >= 15 && index < 34) {
          state.engine.puck_owner = state.team1.goalie_scnum(); state.team1.goalie.has_puck = 1;
          state.team1.pass_attempts = 0;
        }
        if (index >= 26 && index < 34) {
          state.engine.puck_owner = -256; state.team1.pass_attempts = 1;
        }
        if (index >= 34 && index < 38) { state.engine.puck_owner = state.team1.skater_scnum_base(); state.team1.pass_attempts = 1; }
        if (index >= 40 && index < 44) state.engine.clock_stopped = true;
        if (index >= 48 && index < 52) state.team1.goalie.live_state_flags = 4;
        return { state: encode(state), live: state, blocked: index < 2 };
      });
      return { kind, states };
    }));
    const actual = cases.map(({ kind, states }) => {
      const controller = kind === "classic" ? new GoalieController() : kind === "deep" ? new DeepGoalie() : new CSaveGoalie();
      return states.map(({ live: state, blocked }) => {
        const action = controller.step(state, { blocked });
        return serial({ action, diagnostics: controller.diagnostics, phase: controller.phase, frames: controller.frames,
          metrics: controller.metrics, pending_save: controller.pending_save, save_at: controller.save_at, outlet: controller.outlet,
          release_until: controller.release_until, retry_at: controller.retry_at, pass_windup: controller.pass_windup });
      });
    });
    parity(serial(actual), reference("goalie", cases.map(({ kind, states }) => ({ kind, states: states.map(({ live: _live, ...state }) => state) }))));
    expect(required(actual[0], "goalie sequence")).toHaveLength(64);
  });

  it("retains a live incoming receiver after the target word clears and drops stale commitments", () => {
    for (const ending of ["touch", "possession", "pass", "stoppage", "period", "expiry", "miss"]) {
      const state = goalieState(true), controller = new GoalieController();
      Object.assign(state.engine, { puck_owner: -256, last_puck_player: 6, pass_target: 7 });
      state.team2.pass_attempts = 1;
      Object.assign(state.puck, { x: -60, y: -175, motion_x: 4, motion_y: 0 });
      Object.assign(state.team2.players[1]!, { x: 30, y: -175 });
      controller.frames = 1;
      expect(controller._retained_receiver(state)).toBeNull();
      expect(controller.incoming_pass?.receiver).toBe(7);
      state.engine.pass_target = -249;
      Object.assign(state.team2.players[1]!, { x: 34, selection_flags: 0x20, live_anim: 0x92e, is_one_timer: 1 });
      const before = JSON.stringify(state);
      controller.frames++;
      expect(controller._retained_receiver(state)).toBe(7);
      expect(JSON.stringify(state)).toBe(before);
      if (ending === "touch") state.engine.last_puck_player = 7;
      if (ending === "possession") state.engine.puck_owner = 0;
      if (ending === "pass") state.team2.pass_attempts++;
      if (ending === "stoppage") state.engine.clock_stopped = true;
      if (ending === "period") state.period++;
      if (ending === "expiry") controller.frames = controller.incoming_pass!.expires + 1;
      if (ending === "miss") state.team2.players[1]!.y = 20;
      expect(controller._retained_receiver(state)).toBeNull();
      expect(controller.incoming_pass).toBeNull();
    }
  });

  it("holds a selected goalie's outlet B until native launch confirmation or timeout", () => {
    const state = goalieState(true), controller = new GoalieController();
    state.engine.puck_owner = state.team1.goalie_scnum(); state.team1.pass_attempts = 0;
    const buttons = neutralAction(); buttons[0] = 1;
    controller.frames = 5;
    controller.outlet = { receiver: 1, started: 1, passes: 0, launched: false, deadline: 60, buttons };
    expect(controller._outlet_feedback(state)).toEqual(buttons);
    state.team1.pass_attempts = 1; state.engine.puck_owner = -256;
    expect(controller._outlet_feedback(state)).toEqual(neutralAction());
    expect(controller.outlet?.launched).toBe(true);
  });
});
