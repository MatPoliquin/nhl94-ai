import { CONTROLLER_PORTS, type ControllerTeams, type ControllerValues } from "./controllers";

export interface Point { readonly x: number; readonly y: number }
export interface Controller {
  readonly team: number;
  readonly slot: number;
  readonly player: Point | null;
}
export interface Observation {
  readonly frame: number;
  readonly period: number;
  readonly seconds: number;
  readonly score: readonly [number, number];
  readonly puck: Point;
  readonly puckOwner: number;
  readonly controllers: ControllerValues<Controller>;
  readonly transient?: boolean;
  readonly ramHash: string;
}

export function ramDigest(ram: Uint8Array): string {
  let hash = 0x811c9dc5;
  for (const byte of ram) hash = Math.imul(hash ^ byte, 0x01000193);
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function decodeObservation(ram: Uint8Array, frame: number, expected?: ControllerTeams): Observation {
  if (ram.length !== 65536) throw new Error("Genesis work RAM must contain 65536 bytes.");
  // The LSB_FIRST core stores each 68000 word in host byte order.
  const view = new DataView(ram.buffer, ram.byteOffset, ram.byteLength);
  const word = (address: number) => view.getUint16(address - 0xff0000, true);
  const signed = (address: number) => view.getInt16(address - 0xff0000, true);
  const controller = (port: number): Controller => {
    const team = word(0xffc328 + port * 2);
    const slot = signed(0xffc320 + port * 2);
    const assigned = (team === 1 && slot >= 0 && slot < 6) ||
      (team === 2 && slot >= 6 && slot < 12);
    const base = 0xffb04a + slot * 0x80;
    return {
      team, slot,
      player: assigned ? { x: signed(base), y: signed(base + 0x14) } : null,
    };
  };
  let controllers: ControllerValues<Controller> = [controller(0), controller(1), controller(2), controller(3)];
  let transient = false;
  const highlight = !!(ram[0xc2ea ^ 1]! & 0x10);
  if (expected && (expected[2] !== 0 || expected[3] !== 0) && highlight) {
    const unavailable = (port: number): Controller => ({ team: expected[port]!, slot: -1, player: null });
    controllers = [unavailable(0), unavailable(1), unavailable(2), unavailable(3)];
  } else if (expected && (expected[2] !== 0 || expected[3] !== 0)) {
    const active = controllers.filter((entry, port) => expected[port] && entry.slot >= 0).map((entry) => entry.slot);
    const selected = Array.from({ length: 12 }, (_, slot) => slot).filter((slot) =>
      slot % 6 !== 5 && !!(ram[(0xb04a + slot * 0x80 + 0x62) ^ 1]! & 8));
    // The ROM can end a frame while C3/C4 are temporarily aliased into C1/C2.
    transient = controllers.some((entry, port) => entry.team !== expected[port]) ||
      new Set(active).size !== active.length ||
      selected.some((slot) => !active.includes(slot)) ||
      active.some((slot) => slot % 6 !== 5 && !selected.includes(slot));
    if (controllers.some((entry) => entry.team > 2 || entry.slot >= 12)) {
      throw new Error("Invalid native controller ownership.");
    }
    if (transient) {
      const unavailable = (port: number): Controller => ({ team: expected[port]!, slot: -1, player: null });
      controllers = [unavailable(0), unavailable(1), unavailable(2), unavailable(3)];
    } else if (CONTROLLER_PORTS.some((port) =>
      expected[port] && controllers[port].slot >= 0 && !controllers[port].player)) {
      throw new Error("Native controller selected a player on the wrong team.");
    }
  }
  return {
    frame, period: word(0xffc466), seconds: word(0xffc468),
    score: [word(0xffc6da), word(0xffca3e)],
    puck: { x: signed(0xffb74a), y: signed(0xffb75e) },
    puckOwner: signed(0xffb7aa),
    controllers, transient,
    ramHash: ramDigest(ram),
  };
}
