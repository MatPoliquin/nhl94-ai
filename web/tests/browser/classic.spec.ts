import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import type { FramePacket } from "../../protocol";
import { parseReplay } from "../../replay";

declare global {
  interface Window {
    classicFrame: Pick<FramePacket, "frame" | "observation" | "stats" | "finished"> | undefined;
  }
}

test("plays Classic with live settings and replays paused goalie-mode changes exactly", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied original ROM.");
  await page.addInitScript(() => {
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event) => {
          if (event.data.type === "frame") {
            const { frame, observation, stats, finished } = event.data.packet;
            window.classicFrame = { frame, observation, stats, finished };
          }
        });
      }
    };
  });
  await page.goto("/");
  await page.locator("#controller-2-agent").selectOption("classic");
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("Sabres vs Mighty Ducks is ready");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.classicFrame?.frame ?? 0)).toBeGreaterThan(90);
  await page.getByRole("radio", { name: "10 Hz", exact: true }).check();
  await expect(page.locator("#status")).toContainText("no ml bot settings changed");
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  await page.getByRole("checkbox", { name: "Consider player ratings" }).uncheck();
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.classicFrame?.frame ?? 0)).toBeGreaterThan(180);
  await page.getByRole("checkbox", { name: "AI Goalie", exact: true }).uncheck();
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  const final = await page.evaluate(() => window.classicFrame);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const path = await (await downloadPromise).path();
  const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
  expect(replay.format).toBe("nhl94-browser-inputs-v4");
  if (replay.format !== "nhl94-browser-inputs-v4") throw new Error("Expected configuration-aware replay.");
  expect(replay.configurationEvents.at(-1)).toEqual({
    frame: replay.inputs.length, teams: [1, 2, 0, 0], goalieModes: [0, 1],
  });
  expect(replay.inputs.some((inputs) => inputs[1] !== 0)).toBe(true);
  await page.locator("#replay").setInputFiles(path!);
  await expect(page.locator("#status")).toContainText("Replay loaded");
  await expect(page.getByRole("checkbox", { name: "AI Goalie", exact: true })).toBeDisabled();
  await expect(page.getByRole("radio", { name: "10 Hz", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#status")).toContainText("Replay finished", { timeout: 15000 });
  const replayed = await page.evaluate(() => window.classicFrame);
  expect(replayed?.frame).toBe(final?.frame);
  expect(replayed?.observation).toEqual(final?.observation);
  expect(replayed?.stats).toEqual(final?.stats);
  await page.getByRole("button", { name: "Restart match" }).click();
  await expect(page.locator("#frame")).toHaveText("0");
  await expect(page.getByRole("checkbox", { name: "AI Goalie", exact: true })).not.toBeChecked();
  await expect(page.getByRole("radio", { name: "10 Hz", exact: true })).toBeChecked();
  await expect(page.locator("#error")).toBeHidden();
});

test("runs two independent Classic slots with opposite native team assignments", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied original ROM.");
  await page.goto("/");
  await page.locator("#controller-1-agent").selectOption("classic");
  await page.locator("#controller-2-agent").selectOption("classic");
  await page.getByRole("radio", { name: "6 Hz", exact: true }).check();
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("ROM loaded locally");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(async () => Number(await page.locator("#frame").textContent())).toBeGreaterThan(120);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const path = await (await downloadPromise).path();
  const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
  expect(replay.inputs.some((inputs) => inputs[0] !== 0)).toBe(true);
  expect(replay.inputs.some((inputs) => inputs[1] !== 0)).toBe(true);
  await expect(page.locator("#controller-note")).toContainText("No Human slot");
  await expect(page.locator("#error")).toBeHidden();
});
