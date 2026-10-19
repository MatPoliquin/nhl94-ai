export class GenesisRam {
  private readonly view: DataView;

  constructor(private readonly bytes: Uint8Array) {
    if (bytes.length !== 65536) throw new Error("Classic requires exactly 64 KiB of Genesis work RAM.");
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  u8(address: number): number { return this.bytes[(address - 0xff0000) ^ 1]!; }
  i8(address: number): number { const value = this.u8(address); return value < 128 ? value : value - 256; }
  u16(address: number): number { return this.view.getUint16(address - 0xff0000, true); }
  i16(address: number): number { return this.view.getInt16(address - 0xff0000, true); }
  u32(address: number): number { return this.u16(address) * 65536 + this.u16(address + 2); }
  i32(address: number): number {
    const value = this.u32(address);
    return value < 0x80000000 ? value : value - 0x100000000;
  }
}
