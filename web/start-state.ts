import { sha256 } from "./rom";

export const DEFAULT_STATE_FILE = "SabresVsMightyDucks.ManualGoalie.Start.state";
export const DEFAULT_STATE_HASH = "a091b75562f72a155143fbe3749934e5db805a096a486d356f7eff594e0670f8";
export const DEFAULT_STATE_BYTES = 1036288;

export async function decodeDefaultState(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  if (bytes.length !== 30418 || await sha256(bytes) !== DEFAULT_STATE_HASH) {
    throw new Error("The bundled Sabres/Ducks starting state is missing, damaged or incompatible.");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const snapshot = new Uint8Array(await new Response(stream).arrayBuffer());
  if (snapshot.length !== DEFAULT_STATE_BYTES ||
      new TextDecoder("ascii").decode(snapshot.subarray(0, 16)) !== "GENPLUS-GX 1.7.5") {
    throw new Error("Unsupported Genesis Plus GX starting-state format.");
  }
  return snapshot;
}
