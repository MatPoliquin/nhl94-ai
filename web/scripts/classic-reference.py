"""Offline Python oracle for the browser Classic port; never shipped to a browser."""
import argparse
import base64
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.control_rate import ControlRateLimit
from nhl94_ai.game.platforms.genesis import GenesisBackend
from nhl94_ai.game.ram import controller_team_info, pass_geometry_info
from nhl94_ai.game.state import NHL94GameState


def fingerprint():
    files = {Path(module.__file__).resolve() for name, module in sys.modules.items()
             if name.startswith("nhl94_ai.") and getattr(module, "__file__", "").endswith(".py")}
    files.add(ROOT / "configs/classic_bot.json")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_relative_to(ROOT)}


class Memory:
    def __init__(self, ram):
        self.ram = ram

    def extract(self, address, kind):
        size = int(kind[-1])
        offset = address - 0xFF0000
        if offset < 0 or offset + size > 65536:
            raise ValueError(f"Reference RAM address out of bounds: {address:#x}")
        value = bytes(self.ram[(offset + index) ^ 1] for index in range(size))
        return int.from_bytes(value, "big", signed="i" in kind)


class Data:
    def __init__(self, ram, variables):
        self.memory = Memory(ram)
        self.variables = dict(variables)

    def set_variable(self, name, value):
        self.variables[name] = value

    def lookup_all(self):
        return {name: self.memory.extract(value["address"], value["type"])
                for name, value in self.variables.items()}


def snapshot(state):
    def team(value):
        return {
            "controller": value.controller, "num_players": value.num_players,
            "players": [asdict(player) for player in value.players],
            "goalie": asdict(value.goalie), "stats": asdict(value.stats),
            "last_stats": asdict(value.last_stats), "goalie_stats": asdict(value.goalie_stats),
            "net": asdict(value.net), "control": value.control,
            "player_haspuck": value.player_haspuck, "goalie_haspuck": value.goalie_haspuck,
            "defense_control": value.defense_control, "defense_goalie": value.defense_goalie,
            "pass_attempts": value.pass_attempts, "one_timer_attempts": value.one_timer_attempts,
        }
    return {
        "team1": team(state.team1), "team2": team(state.team2),
        "puck": asdict(state.puck), "engine": asdict(state.engine),
        "period": state.period, "time": state.time, "last_time": state.last_time,
        "is_shootout_active": state.is_shootout_active, "is_shooting": state.is_shooting,
        "has_released_shot": state.has_released_shot,
    }


def run(request):
    from stable_retro.data import get_file_path
    integration = request.get("integration") or get_file_path("NHL94-Genesis-v0", "data.json")
    variables = json.loads(Path(integration).read_text(encoding="utf-8"))["info"]
    ram = bytearray(65536)
    data = Data(ram, variables)
    backend = GenesisBackend()
    env = SimpleNamespace(data=data, gamename="NHL94-Genesis-v0", pass_geometry_rom=request["rom"])
    env.unwrapped = env
    backend.register(env, 5)
    config = request["config"]
    args = SimpleNamespace(env="NHL94-Genesis-v0", action_type="FILTERED",
                           goalie_policy="selective" if config["ai_goalie"] else "off",
                           player_ratings_consideration=config["player_ratings_consideration"])
    controller = ClassicAIV1Model(args)
    interval = 0 if config["decision_hz"] == 60 else 60 // config["decision_hz"]
    adapter = ControlRateLimit(controller, interval) if interval else controller
    actions, decisions, snapshots = [], [], []
    masks = (1, 512, 0, 8, 16, 32, 64, 128, 256, 0, 0, 0)
    for index, frame in enumerate(request["frames"]):
        if "ram" in frame:
            ram[:] = base64.b64decode(frame["ram"], validate=True)
            if len(ram) != 65536:
                raise ValueError("Reference frame must contain exactly 64 KiB.")
        else:
            changes = frame["changes"]
            if len(changes) % 2:
                raise ValueError("Invalid reference RAM delta.")
            for offset in range(0, len(changes), 2):
                ram[changes[offset]] = changes[offset + 1]
        info = controller_team_info(pass_geometry_info(env, data.lookup_all()))
        state = NHL94GameState(5)
        state.BeginFrame(info, [0] * 6)
        state.EndFrame()
        if index in request.get("snapshots", []):
            snapshots.append({"frame": index, "state": snapshot(state)})
        port = request.get("port", 0)
        side = info[f"defense_team{port + 1}"]
        if side not in (1, 2):
            raise ValueError("Reference bot has no team assignment.")
        if side == 2:
            state.Flip()
        slot = info[f"defense_control{port + 1}"]
        state.team1.defense_control = slot
        state.team1.control = (-1 if slot < 0 else 0 if slot == state.team1.goalie_scnum()
                               else slot - state.team1.skater_scnum_base() + 1)
        state.engine.controlled_is_shooter = float(state.engine.shot_player >= 0
                                                  and state.engine.shot_player == slot)
        action = adapter.predict_frame(state, 4)[0]
        actions.append(sum(mask for mask, pressed in zip(masks, action) if pressed))
        decisions.append(controller._last_decision)
    return {"actions": actions, "decisions": decisions, "snapshots": snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    ClassicAIV1Model(SimpleNamespace(env="NHL94-Genesis-v0", action_type="FILTERED",
                                    goalie_policy="selective"))
    before = fingerprint()
    if args.manifest:
        print(json.dumps({"profile": "Genesis full-team FILTERED baseline; optional experiments off",
                          "sources": before}, indent=2))
        return
    manifest = json.loads((ROOT / "web/classic/reference.json").read_text(encoding="utf-8"))
    if manifest["sources"] != before:
        raise ValueError("Classic Python reference changed; review and deliberately update its fingerprint.")
    payload = sys.stdin.buffer.read()
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    result = run(json.loads(payload))
    if fingerprint() != before:
        # run imports stable-retro, but must not import another nhl94_ai implementation.
        raise ValueError("Classic reference source set changed during comparison.")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
