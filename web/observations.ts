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
  readonly controllers: readonly [Controller, Controller];
  readonly ramHash: string;
}

export function ramDigest(ram: Uint8Array): string {
  let hash = 0x811c9dc5;
  for (const byte of ram) hash = Math.imul(hash ^ byte, 0x01000193);
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function decodeObservation(ram: Uint8Array, frame: number): Observation {
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
  return {
    frame, period: word(0xffc466), seconds: word(0xffc468),
    score: [word(0xffc6da), word(0xffca3e)],
    puck: { x: signed(0xffb74a), y: signed(0xffb75e) },
    puckOwner: signed(0xffb7aa),
    controllers: [controller(0), controller(1)],
    ramHash: ramDigest(ram),
  };
}
