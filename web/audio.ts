export class AudioOutput {
  private context: AudioContext | undefined;
  private nextTime = 0;
  private readonly sources = new Set<AudioBufferSourceNode>();

  async enable(sampleRate: number): Promise<void> {
    this.context ??= new AudioContext({ sampleRate });
    await this.context.resume();
    if (this.context.state !== "running") throw new Error("Browser audio could not start.");
  }

  play(samples: Float32Array, sampleRate: number): void {
    if (!this.context || samples.length === 0) return;
    if (this.context.state !== "running") throw new Error("Audio was suspended. Pause and resume playback.");
    const frames = samples.length / 2;
    const buffer = this.context.createBuffer(2, frames, sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const output = buffer.getChannelData(channel);
      for (let frame = 0; frame < frames; frame++) output[frame] = samples[frame * 2 + channel]!;
    }
    const start = Math.max(this.nextTime, this.context.currentTime + 0.025);
    if (start > this.context.currentTime + 0.3) throw new Error("Audio queue is full. Pause or disable sound.");
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => { this.sources.delete(source); source.disconnect(); };
    this.sources.add(source);
    source.start(start);
    this.nextTime = start + buffer.duration;
  }

  clear(): void {
    for (const source of this.sources) source.stop();
    this.sources.clear();
    this.nextTime = 0;
  }
}
