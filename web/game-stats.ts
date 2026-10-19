export interface TeamStats {
  readonly score: number;
  readonly shots: number;
  readonly powerPlayGoals: number;
  readonly powerPlayOpportunities: number;
  readonly powerPlaySeconds: number;
  readonly powerPlayShots: number;
  readonly shorthandedGoals: number;
  readonly breakawayGoals: number;
  readonly breakawayAttempts: number;
  readonly oneTimerGoals: number;
  readonly oneTimerAttempts: number;
  readonly penaltyShotGoals: number;
  readonly penaltyShotAttempts: number;
  readonly faceoffsWon: number;
  readonly checks: number;
  readonly penalties: number;
  readonly penaltyMinutes: number;
  readonly attackZoneSeconds: number;
  readonly passesCompleted: number;
  readonly passAttempts: number;
}

export type GameStats = readonly [TeamStats, TeamStats];

export function decodeGameStats(ram: Uint8Array): GameStats {
  if (ram.length !== 65536) throw new Error("Genesis work RAM must contain 65536 bytes.");
  const view = new DataView(ram.buffer, ram.byteOffset, ram.byteLength);
  // The ROM's pause-menu table at 0x0FDFAA reads words from these team records.
  const team = (base: number): TeamStats => {
    const word = (offset: number) => view.getUint16(base + offset, true);
    return {
      score: word(0x0c), shots: word(0x00),
      powerPlayGoals: word(0x02), powerPlayOpportunities: word(0x04),
      powerPlaySeconds: word(0x352), powerPlayShots: word(0x354),
      shorthandedGoals: word(0x356),
      breakawayGoals: word(0x35a), breakawayAttempts: word(0x358),
      oneTimerGoals: word(0x35e), oneTimerAttempts: word(0x35c),
      penaltyShotGoals: word(0x362), penaltyShotAttempts: word(0x360),
      faceoffsWon: word(0x0e), checks: word(0x10),
      penalties: word(0x06), penaltyMinutes: word(0x08),
      attackZoneSeconds: word(0x0a),
      passesCompleted: word(0x14), passAttempts: word(0x12),
    };
  };
  return [team(0xc6ce), team(0xca32)];
}

function time(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function percentage(numerator: number, denominator: number): string {
  // The native renderer skips division when the denominator is zero.
  return `${denominator === 0 ? numerator * 100 : Math.floor(numerator * 100 / denominator)}%`;
}

function ratio(numerator: number, denominator: number): string {
  return `${numerator} / ${denominator}`;
}

export const GAME_STAT_ROWS: readonly {
  id: string;
  label: string;
  format: (stats: TeamStats) => string;
}[] = [
  { id: "score", label: "Score", format: (s) => String(s.score) },
  { id: "shots", label: "Shots on goal", format: (s) => String(s.shots) },
  { id: "shooting", label: "Shooting %", format: (s) => percentage(s.score, s.shots) },
  { id: "power-play", label: "Power play (goals/opportunities)", format: (s) => ratio(s.powerPlayGoals, s.powerPlayOpportunities) },
  { id: "power-play-time", label: "Power play time", format: (s) => time(s.powerPlaySeconds) },
  { id: "power-play-shots", label: "Power play shots", format: (s) => String(s.powerPlayShots) },
  { id: "shorthanded", label: "Shorthanded goals", format: (s) => String(s.shorthandedGoals) },
  { id: "breakaways", label: "Breakaways (goals/attempts)", format: (s) => ratio(s.breakawayGoals, s.breakawayAttempts) },
  { id: "one-timers", label: "One-timers (goals/attempts)", format: (s) => ratio(s.oneTimerGoals, s.oneTimerAttempts) },
  { id: "penalty-shots", label: "Penalty shots (goals/attempts)", format: (s) => ratio(s.penaltyShotGoals, s.penaltyShotAttempts) },
  { id: "faceoffs", label: "Faceoffs won", format: (s) => String(s.faceoffsWon) },
  { id: "checks", label: "Body checks", format: (s) => String(s.checks) },
  { id: "penalties", label: "Penalties / minutes", format: (s) => ratio(s.penalties, s.penaltyMinutes) },
  { id: "attack-zone", label: "Attack zone time", format: (s) => time(s.attackZoneSeconds) },
  { id: "passing", label: "Passes (completed/attempted)", format: (s) =>
    `${ratio(s.passesCompleted, s.passAttempts)} (${percentage(s.passesCompleted, s.passAttempts)})` },
];
