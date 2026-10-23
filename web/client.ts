import type { WorkerCommand, WorkerRequest, WorkerResponse } from "./protocol";

type Success = Exclude<WorkerResponse, { type: "error" }>;

export class EmulatorClient {
  private readonly worker = new Worker(new URL("./emulator.worker.ts", import.meta.url), { type: "module" });
  private nextId = 0;
  private failed = false;
  private readonly pending = new Map<number, {
    resolve: (response: Success) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();

  constructor(private readonly onFailure: (error: Error) => void) {
    this.worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      const request = this.pending.get(response.id);
      if (!request) return;
      clearTimeout(request.timer);
      this.pending.delete(response.id);
      if (response.type === "error") request.reject(new Error(response.message));
      else request.resolve(response);
    });
    this.worker.addEventListener("error", (event) => {
      this.fail(new Error(`Emulator worker failed: ${event.message}`));
    });
    this.worker.addEventListener("messageerror", () => {
      this.fail(new Error("Could not decode an emulator worker response."));
    });
  }

  request(command: WorkerCommand, transfer: Transferable[] = []): Promise<Success> {
    if (this.failed) return Promise.reject(new Error("Emulator stopped. Reload the page to recover."));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.fail(new Error("Emulator timed out. Worker terminated; reload the page to recover."));
      }, 15000);
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.worker.postMessage({ ...command, id } satisfies WorkerRequest, transfer);
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error("Could not send emulator command."));
      }
    });
  }

  terminate(reason = new Error("Emulator disposed.")): void {
    this.failed = true;
    this.worker.terminate();
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(reason);
    }
    this.pending.clear();
  }

  private fail(error: Error): void {
    this.terminate(error);
    this.onFailure(error);
  }
}
