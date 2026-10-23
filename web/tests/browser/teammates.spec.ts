import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import type { FramePacket } from "../../protocol";
import { Button } from "../../inputs";
import { parseReplay } from "../../replay";

declare global {
  interface Window { teammateFrame: Pick<FramePacket, "frame" | "observation" | "stats"> | undefined }
}

test("runs three coordinated bots against CPU then a fourth-port Human and replays all four ports exactly", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "getGamepads", { value: () => [] });
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event) => {
          if (event.data.type === "frame") {
            const { frame, observation, stats } = event.data.packet;
            window.teammateFrame = { frame, observation, stats };
          }
        });
      }
    };
  });
  await page.goto("/");
  for (const port of [1, 2, 3]) {
    await page.locator(`#controller-${port}-agent`).selectOption("classic");
    await page.locator(`#controller-${port}-team`).selectOption("1");
  }
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("Sabres vs Mighty Ducks is ready");
  expect(await page.evaluate(() => window.teammateFrame?.observation.controllers.map((controller) => controller.team)))
    .toEqual([1, 1, 1, 0]);
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.teammateFrame?.frame ?? 0), { timeout: 15000 }).toBeGreaterThan(360);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  await page.locator("#controller-4-agent").selectOption("human");
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  expect(await page.evaluate(() => window.teammateFrame?.observation.controllers.map((controller) => controller.team)))
    .toEqual([1, 1, 1, 2]);
  const humanStart = await page.evaluate(() => window.teammateFrame!.frame);
  await page.locator("#screen").focus();
  await page.keyboard.down("ArrowRight");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(humanStart + 1));
  await page.keyboard.up("ArrowRight");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.teammateFrame?.frame ?? 0), { timeout: 15000 }).toBeGreaterThan(humanStart + 360);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  const final = await page.evaluate(() => window.teammateFrame);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const path = await (await downloadPromise).path();
  const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
  expect(replay.format).toBe("nhl94-browser-inputs-v4");
  if (replay.format !== "nhl94-browser-inputs-v4") throw new Error("Expected four-port replay.");
  expect(replay.configurationEvents[0]?.teams).toEqual([1, 1, 1, 0]);
  expect(replay.configurationEvents.at(-1)?.teams).toEqual([1, 1, 1, 2]);
  expect(replay.configurationEvents[0]?.goalieModes).toEqual([1, 0]);
  expect(replay.inputs[humanStart]?.[3]).toBe(Button.RIGHT);
  for (const port of [0, 1, 2]) expect(replay.inputs.some((inputs) => inputs[port] !== 0)).toBe(true);
  await page.locator("#replay").setInputFiles(path!);
  await expect(page.locator("#status")).toContainText("Replay loaded");
  for (const port of [1, 2, 3, 4]) await expect(page.locator(`#controller-${port}-agent`)).toBeDisabled();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#status")).toContainText("Replay finished", { timeout: 25000 });
  expect(await page.evaluate(() => window.teammateFrame)).toEqual(final);
  await page.getByRole("button", { name: "Restart match" }).click();
  await expect(page.locator("#frame")).toHaveText("0");
  for (const port of [1, 2, 3, 4]) await expect(page.locator(`#controller-${port}-agent`)).toBeEnabled();
  await expect(page.locator("#error")).toBeHidden();
  expect(errors).toEqual([]);
});

for (const port of [3, 4]) {
  test(`runs a sole Classic bot on controller ${port} with goalie AI enabled`, async ({ page }) => {
    test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
    await page.goto("/");
    await page.locator("#controller-2-agent").selectOption("none");
    await page.locator(`#controller-${port}-agent`).selectOption("classic");
    await page.locator(`#controller-${port}-team`).selectOption("2");
    await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
    await expect(page.locator("#status")).toContainText("Sabres vs Mighty Ducks is ready");
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await expect.poll(async () => Number(await page.locator("#frame").textContent()), { timeout: 15000 }).toBeGreaterThan(360);
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    await expect(page.locator("#error")).toBeHidden();
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download input replay" }).click();
    const path = await (await downloadPromise).path();
    const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
    expect(replay.inputs.some((inputs) => inputs[port - 1] !== 0)).toBe(true);
  });
}

test("runs all four Classic slots as two separately coordinated opposing teams", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied local ROM.");
  await page.goto("/");
  for (const port of [1, 2, 3, 4]) {
    await page.locator(`#controller-${port}-agent`).selectOption("classic");
    await page.locator(`#controller-${port}-team`).selectOption(port <= 2 ? "1" : "2");
  }
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("Sabres vs Mighty Ducks is ready");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(async () => Number(await page.locator("#frame").textContent()), { timeout: 15000 }).toBeGreaterThan(360);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  await expect(page.locator("#error")).toBeHidden();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const path = await (await downloadPromise).path();
  const replay = parseReplay(JSON.parse(await readFile(path!, "utf8")));
  for (const port of [0, 1, 2, 3]) expect(replay.inputs.some((inputs) => inputs[port] !== 0)).toBe(true);
});
