import { readFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";
import type { FramePacket } from "../../protocol";
import { parseReplay } from "../../replay";
import { DEFAULT_STATE_HASH } from "../../start-state";

declare global {
  interface Window { arenaFrame: Pick<FramePacket, "frame" | "observation" | "stats" | "finished"> | undefined }
}

async function openMatch(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event) => {
          if (event.data.type === "frame") {
            const packet = event.data.packet;
            window.arenaFrame = {
              frame: packet.frame, observation: packet.observation, stats: packet.stats, finished: packet.finished,
            };
          }
        });
      }
    };
  });
  await page.goto("/");
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("Sabres vs Mighty Ducks is ready");
}

async function teams(page: Page, expected: number[]): Promise<void> {
  await expect.poll(() => page.evaluate(() => window.arenaFrame?.observation.controllers.map((c) => c.team)))
    .toEqual(expected);
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
}

async function pauseAfter(page: Page, frame: number): Promise<void> {
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.arenaFrame?.frame ?? 0)).toBeGreaterThan(frame);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
}

test("starts on the supplied rink and changes native team ownership while paused", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
  await openMatch(page);
  await teams(page, [1, 2]);
  expect(await page.evaluate(() => window.arenaFrame?.observation.seconds)).toBe(300);
  await expect(page.locator("#frame")).toHaveText("0");
  expect(await page.locator("#screen").evaluate((canvas: HTMLCanvasElement) => {
    const pixels = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
    return pixels.some((value, index) => index % 4 !== 3 && value > 0);
  })).toBe(true);
  await page.locator("#controller-1-team").selectOption("2");
  await teams(page, [2, 2]);
  const slots = await page.evaluate(() => window.arenaFrame?.observation.controllers.map((c) => c.slot));
  expect(slots?.[0]).not.toBe(slots?.[1]);
  await page.locator("#controller-2-team").selectOption("1");
  await teams(page, [2, 1]);
  await page.locator("#controller-1-agent").selectOption("none");
  await teams(page, [0, 1]);
  expect(await page.evaluate(() => window.arenaFrame?.observation.controllers[0].slot)).toBe(-1);
  await page.locator("#controller-2-agent").selectOption("none");
  await teams(page, [0, 0]);
  await expect(page.locator("#frame")).toHaveText("0");
  await expect(page.locator("#error")).toBeHidden();
});

test("replays the saved start and every native assignment change to the same final RAM and stats", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
  await openMatch(page);
  await pauseAfter(page, 60);
  const changeFrame = await page.evaluate(() => window.arenaFrame!.frame);
  await page.locator("#controller-1-team").selectOption("2");
  await teams(page, [2, 2]);
  await page.locator("#controller-2-team").selectOption("1");
  await teams(page, [2, 1]);
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(changeFrame + 1));
  await page.locator("#controller-1-agent").selectOption("none");
  await teams(page, [0, 1]);
  await pauseAfter(page, changeFrame + 80);
  await page.locator("#controller-2-agent").selectOption("none");
  await teams(page, [0, 0]);
  const final = await page.evaluate(() => window.arenaFrame);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const path = await (await downloadPromise).path();
  const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
  expect(replay.format).toBe("nhl94-browser-inputs-v2");
  if (replay.format !== "nhl94-browser-inputs-v2") throw new Error("Expected a saved-state replay.");
  expect(replay.startStateHash).toBe(DEFAULT_STATE_HASH);
  expect(replay.controllerEvents.filter((event) => event.frame === changeFrame).map((event) => event.teams))
    .toEqual([[2, 2], [2, 1]]);
  expect(replay.controllerEvents.at(-1)).toEqual({ frame: replay.inputs.length, teams: [0, 0] });
  await page.locator("#replay").setInputFiles(path!);
  await expect(page.locator("#status")).toContainText("Replay loaded");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#status")).toContainText("Replay finished", { timeout: 15000 });
  const replayed = await page.evaluate(() => window.arenaFrame);
  expect(replayed?.frame).toBe(final?.frame);
  expect(replayed?.observation).toEqual(final?.observation);
  expect(replayed?.stats).toEqual(final?.stats);
  await page.getByRole("button", { name: "Restart match" }).click();
  await expect(page.locator("#frame")).toHaveText("0");
  expect(await page.evaluate(() => window.arenaFrame?.observation.seconds)).toBe(300);
});

test("supports compatible boot replays and rejects a mismatched saved start explicitly", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
  await openMatch(page);
  await page.getByRole("button", { name: "Step one frame" }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const path = await (await downloadPromise).path();
  const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
  await page.locator("#replay").setInputFiles({
    name: "wrong-start.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ ...replay, startStateHash: "0".repeat(64) })),
  });
  await expect(page.locator("#error")).toContainText("starting state differs");
  await page.locator("#replay").setInputFiles({
    name: "compatible-boot.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({
      format: "nhl94-browser-inputs-v1", coreBuild: replay.coreBuild, romHash: replay.romHash, inputs: [[0, 0]],
    })),
  });
  await expect(page.locator("#status")).toContainText("Replay loaded");
  expect(await page.evaluate(() => window.arenaFrame?.observation.seconds)).toBe(0);
  await expect(page.locator("#error")).toBeHidden();
});

test("fails visibly if the bundled state is unavailable instead of falling back to ROM boot", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
  await page.route("**/states/*.state", (route) => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/");
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#error")).toContainText("Starting state load failed");
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeDisabled();
});
