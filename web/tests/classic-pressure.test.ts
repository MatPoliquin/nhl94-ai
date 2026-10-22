import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { DEFAULT_CLASSIC_CONFIG } from "../classic/config";
import { DefenseController } from "../classic/defense";
import { SharedPressure } from "../classic/teammate-pressure";
import { TeammateCoordinator } from "../classic/teammates";
import { Buttons, GameState, neutralAction } from "../classic/state";
import type { Observation } from "../observations";

function state(): GameState {
  const s = new GameState();
  s.time = 300;
  s.team1.net.y = -264; s.team2.net.y = 264;
  s.team1.defense_control = 0; s.team2.defense_control = 6;
  s.team1.goalie.y = -264; s.team2.goalie.y = 264;
  s.engine.puck_owner = 6; s.engine.puck_owner_known = true; s.engine.clock_stopped = false;
  s.team1.pass_attempts = s.team1.one_timer_attempts = 0;
  s.engine.controller_teams = [1, 1, 1, 2];
  for (const team of [s.team1, s.team2]) {
    team.players.forEach((p, i) => Object.assign(p, {
      x: -90 + i * 40, y: team === s.team1 ? -215 : 70, motion_x: 0, motion_y: 0,
      speed: 20, agility: 20, stick: 20, weight: 64, energy: 4096, checking: 20,
      role: -1, facing: 0, selection_flags: 8,
    }));
  }
  [[-24, -24, 1], [24, 24, 5], [0, -72, 0]].forEach(([x, y, facing], i) =>
    Object.assign(s.team1.players[i]!, { x, y, facing, role: 1 }));
  Object.assign(s.team2.players[0]!, { x: 0, y: 0, role: 1, facing: 4, selection_flags: 0x40 });
  Object.assign(s.puck, { x: 0, y: 0, motion_x: 0, motion_y: 0, motion_z: 0, height: 0 });
  return s;
}

function observation(s: GameState): Observation {
  return { frame: 0, seconds: 300, period: 0, score: [0, 0], puck: { x: s.puck.x, y: s.puck.y },
    puckOwner: s.engine.puck_owner, ramHash: "", controllers: [
      { team: 1, slot: 0, player: { x: -24, y: -24 } }, { team: 1, slot: 1, player: { x: 24, y: 24 } },
      { team: 1, slot: 2, player: { x: 0, y: -72 } }, { team: 2, slot: 6, player: { x: 0, y: 0 } },
    ] };
}

describe("shared defensive pressure", () => {
  it("reserves three-bot cover, complementary angles, and one contact owner", () => {
    const s = state(), before = JSON.stringify(s), planner = new SharedPressure();
    const result = planner.choose(s, [0, 1, 2], 1)!;
    expect(result.map((a) => a.plan.mode)).toEqual(["contain", "close", "cover"]);
    expect(result[0]!.plan.target).toEqual([-12, -12]);
    expect(result[1]!.plan.target).toEqual([12, 12]);
    expect(result.filter((a) => a.contact)).toHaveLength(1);
    expect(result[2]!.contact).toBe(false);
    expect(JSON.stringify(s)).toBe(before);
    expect(new SharedPressure().choose(s, [2, 1, 0], 1)!.map((a) => a.plan.mode))
      .toEqual(["cover", "close", "contain"]);
  });

  it("requires observed cover for two bots, including future drift", () => {
    const s = state();
    expect(new SharedPressure().choose(s, [0, 1], 1)).not.toBeNull();
    s.team1.players[2]!.motion_x = 2;
    expect(new SharedPressure().choose(s, [0, 1], 1)).toBeNull();
    s.team1.players[2]!.role = -1;
    expect(new SharedPressure().choose(s, [0, 1], 1)).toBeNull();
  });

  it("declines open receivers, rushing carriers, and obstructed approaches", () => {
    const receiver = state();
    Object.assign(receiver.team2.players[1]!, { role: 1, x: 60, y: -170 });
    expect(new SharedPressure().choose(receiver, [0, 1, 2], 1)).toBeNull();
    const rush = state(); rush.team2.players[0]!.motion_y = -4;
    expect(new SharedPressure().choose(rush, [0, 1, 2], 1)).toBeNull();
    const collision = state();
    Object.assign(collision.team1.players[1]!, { x: -24, y: -24 });
    expect(new SharedPressure().choose(collision, [0, 1, 2], 1)).toBeNull();
  });

  it("declines a trap when a currently wide receiver enters a dangerous lane during the approach", () => {
    const s = state();
    Object.assign(s.team2.players[1]!, { role: 1, x: 100, y: -170, motion_x: -1 });
    expect(new SharedPressure().choose(s, [0, 1, 2], 1)).toBeNull();
  });

  it("uses the boards without targeting outside the rink", () => {
    const s = state(); s.team2.players[0]!.x = s.puck.x = 100;
    [[76, -24], [94, 24], [65, -72]].forEach(([x, y], i) => Object.assign(s.team1.players[i]!, { x, y }));
    const result = new SharedPressure().choose(s, [0, 1, 2], 1)!;
    expect(result).not.toBeNull();
    expect(result[0]!.plan.target[0]).toBeLessThan(100);
    expect(result[1]!.plan.target[0]).toBeLessThan(100);
    expect(result[0]!.plan.target[1]).toBeLessThan(0);
    expect(result[1]!.plan.target[1]).toBeGreaterThan(0);
  });

  it("keeps roles stable but aborts for passes, loose pucks, selections, and falling cover", () => {
    for (const change of ["pass", "loose", "selection", "fall", "unknown"]) {
      const s = state(), planner = new SharedPressure();
      const first = planner.choose(s, [0, 1, 2], 1)!;
      s.team1.players[0]!.x += 1;
      expect(planner.choose(s, [0, 1, 2], 2)!.map((a) => a.plan.mode)).toEqual(first.map((a) => a.plan.mode));
      if (change === "pass") { s.engine.puck_owner = 7; s.team2.players[1]!.role = 1; }
      if (change === "loose") s.engine.puck_owner = -1;
      if (change === "fall") s.team1.players[2]!.is_falling = 1;
      if (change === "unknown") s.engine.puck_owner_known = false;
      expect(planner.choose(s, change === "selection" ? [1, 0, 2] : [0, 1, 2], 3)).toBeNull();
    }
  });

  it("regroups after a trap stalls", () => {
    const s = state(), planner = new SharedPressure();
    expect(planner.choose(s, [0, 1, 2], 1)).not.toBeNull();
    expect(planner.choose(s, [0, 1, 2], 121)).toBeNull();
    expect(planner.choose(s, [0, 1, 2], 140)).toBeNull();
    expect(planner.choose(s, [0, 1, 2], 157)).not.toBeNull();
  });

  it("executes assigned targets without individual switches or unassigned contact", () => {
    const s = state(), planner = new SharedPressure(), controller = new DefenseController();
    const assignment = { ...planner.choose(s, [0, 1, 2], 1)![0]!, contact: false };
    const select = vi.spyOn(controller, "select_player").mockImplementation(() => { throw Error("independent switching"); });
    const action = controller.step(s, { assignment });
    expect(select).not.toHaveBeenCalled();
    expect(action[Buttons.INPUT_B]).toBe(0); expect(action[Buttons.INPUT_C]).toBe(0);
  });

  it("permits covered rear checks while retaining contact safety", () => {
    const s = state(), player = s.team1.players[1]!, controller = new DefenseController();
    Object.assign(player, { x: 0, y: 24, facing: 4, weight: 32 });
    s.team1.players[0]!.x = -60;
    expect(controller._body_check(s, 1, player).status).toBe("not-goal-side");
    expect(controller._body_check(s, 1, player, true).status).toBe("ready");
    player.energy = 0;
    expect(controller._body_check(s, 1, player, true).status).toBe("low-energy");
    player.energy = 4096; Object.assign(s.team1.players[0]!, { x: 0, y: 20 });
    expect(controller._body_check(s, 1, player, true).status).toBe("obstructed-contact");
  });

  it("runs all three assignments and releases held inputs immediately on a pass", () => {
    const s = state(), team = new TeammateCoordinator([0, 1, 2], { ...DEFAULT_CLASSIC_CONFIG, decision_hz: 6 });
    const normal = team.controllers.map((c) => vi.spyOn(c, "predict_frame").mockReturnValue(neutralAction()));
    team.act(observation(s), s);
    expect(team.roles).toEqual(["contain", "close", "cover"]);
    expect(team.controllers.map((c) => c.scheduler.frames)).toEqual([1, 1, 1]);
    expect(normal.every((call) => call.mock.calls.length === 0)).toBe(true);
    team.act(observation(s), s);
    expect(team.controllers.map((c) => c.scheduler.frames)).toEqual([2, 2, 2]);
    s.engine.puck_owner = -1;
    team.act(observation(s), s);
    expect(team.roles).toContain("retrieve");
    expect(team.roles.some((role) => ["contain", "close", "cover"].includes(role))).toBe(false);
    expect(team.controllers.every((c) => c.defense.pending_check === null)).toBe(true);
  });
});

const root = resolve(import.meta.dirname, "../..");
const python = process.env.NHL94_PYTHON ?? resolve(root, "../stable-retro/venv/bin/python");
it.skipIf(!existsSync(python))("matches the Python planner and executor across a trap and abort sequence", () => {
  const script = `
import json
from nhl94_ai.agents.teammate_pressure import SharedPressure
from nhl94_ai.agents.defense import DefenseController
from tests.unit.test_teammate_pressure import pressure_state
s, planner = pressure_state(), SharedPressure()
controllers = [DefenseController() for _ in range(3)]
rows = []
for frame in range(1, 7):
    if frame == 3: s.team1.players[0].x += 1
    if frame == 4: s.engine.puck_owner = -1
    assignments = planner.choose(s, [0, 1, 2], frame)
    actions = []
    if assignments:
        for i, a in enumerate(assignments):
            s.team1.defense_control = i
            actions.append(controllers[i].step(s, frame=frame, assignment=a).tolist())
    rows.append(dict(diagnostics=planner.diagnostics.copy(), actions=actions,
                     targets=[a.plan.target for a in assignments] if assignments else []))
print(json.dumps(rows))
`;
  const oracle = spawnSync(python, ["-c", script], { cwd: root, encoding: "utf8" });
  expect(oracle.status, oracle.stderr).toBe(0);
  const s = state(), planner = new SharedPressure(), controllers = [0, 1, 2].map(() => new DefenseController());
  const rows = [];
  for (let frame = 1; frame <= 6; frame++) {
    if (frame === 3) s.team1.players[0]!.x += 1;
    if (frame === 4) s.engine.puck_owner = -1;
    const assignments = planner.choose(s, [0, 1, 2], frame);
    const actions = assignments?.map((assignment, i) => {
      s.team1.defense_control = i;
      return controllers[i]!.step(s, { frame, assignment });
    }) ?? [];
    rows.push({ diagnostics: { ...planner.diagnostics }, actions, targets: assignments?.map((a) => a.plan.target) ?? [] });
  }
  expect(rows).toEqual(JSON.parse(oracle.stdout));
});
