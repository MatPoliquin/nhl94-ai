export const MAX_ROM_BYTES = 4 * 1024 * 1024;
export const ORIGINAL_ROM_HASH =
  "ccc0298c118d580863c877abd470c71186ef0a1b60f1e7c6ff1e5a38f48fd053";

export function inspectRom(bytes: Uint8Array): string {
  if (bytes.length < 512 || bytes.length > MAX_ROM_BYTES) {
    throw new Error("Choose an uncompressed NHL94 Genesis ROM between 512 bytes and 4 MiB.");
  }
  const decoder = new TextDecoder("ascii");
  if (!decoder.decode(bytes.subarray(0x100, 0x110)).startsWith("SEGA")) {
    throw new Error("Not a raw Genesis ROM. ZIP and interleaved SMD files are not supported.");
  }
  const title = decoder.decode(bytes.subarray(0x120, 0x150)).trim();
  if (!/NHL.*94/i.test(title)) throw new Error("This smoke test supports NHL94 only.");
  return title;
}

export async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function identifyRom(bytes: Uint8Array<ArrayBuffer>): Promise<{ title: string; hash: string }> {
  const title = inspectRom(bytes);
  const hash = await sha256(bytes);
  if (hash !== ORIGINAL_ROM_HASH) {
    throw new Error(`Unsupported NHL94 ROM revision (${hash}). This first smoke test requires the original ROM; see README.`);
  }
  return { title, hash };
}
