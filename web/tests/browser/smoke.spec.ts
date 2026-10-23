import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { parseReplay } from "../../replay";

test("shows the local arena scope and rejects invalid local files", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Local AI arena.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeDisabled();
  await page.locator("#rom").setInputFiles({
    name: "invalid.bin", mimeType: "application/octet-stream", buffer: Buffer.from("not a ROM"),
  });
  await expect(page.getByRole("alert")).toContainText("between 512 bytes and 4 MiB");
});

test("keeps the complete native frame inside its screen container at every video size", async ({ page }) => {
  await page.goto("/");
  for (const width of [1280, 1043, 850, 390]) {
    await page.setViewportSize({ width, height: 960 });
    for (const [nativeWidth, nativeHeight] of [[256, 224], [320, 224], [320, 240]]) {
      const bounds = await page.locator("#screen").evaluate((canvas: HTMLCanvasElement, size) => {
        canvas.width = size[0]!;
        canvas.height = size[1]!;
        const frame = canvas.getBoundingClientRect(), container = canvas.parentElement!.getBoundingClientRect();
        return { top: frame.top, bottom: frame.bottom, left: frame.left, right: frame.right,
          containerTop: container.top, containerBottom: container.bottom,
          containerLeft: container.left, containerRight: container.right };
      }, [nativeWidth!, nativeHeight!]);
      expect(bounds.top).toBeGreaterThanOrEqual(bounds.containerTop);
      expect(bounds.bottom, `${width}px viewport, ${nativeWidth}x${nativeHeight} native frame`).toBeLessThanOrEqual(bounds.containerBottom);
      expect(bounds.left).toBeGreaterThanOrEqual(bounds.containerLeft);
      expect(bounds.right).toBeLessThanOrEqual(bounds.containerRight);
      await expect(page.locator("#screen")).toHaveCSS("object-fit", "contain");
    }
  }
});

test("runs the actual WASM worker, pauses, steps and replays local inputs", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Set NHL94_ROM to a legally supplied local original ROM.");
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("ROM loaded locally");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText("1");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(async () => Number(await page.locator("#frame").textContent())).toBeGreaterThan(30);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator("#status")).toHaveText("Paused.");
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  await page.getByRole("combobox", { name: "Controller 1 player", exact: true }).selectOption("none");
  await page.getByRole("combobox", { name: "Controller 2 player", exact: true }).selectOption("human");
  await page.locator("#screen").focus();
  const pressedFrame = Number(await page.locator("#frame").textContent()) + 1;
  await page.keyboard.down("x");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(pressedFrame));
  await page.keyboard.up("x");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(pressedFrame + 1));
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const download = await downloadPromise;
  const saved = await download.path();
  expect(saved).not.toBeNull();
  const recorded = parseReplay(JSON.parse(await readFile(saved!, "utf8")));
  expect(recorded.inputs[pressedFrame - 1]).toEqual([0, 1, 0, 0]);
  expect(recorded.inputs[pressedFrame]).toEqual([0, 0, 0, 0]);
  await page.getByRole("button", { name: "Restart match" }).click();
  await expect(page.locator("#frame")).toHaveText("0");
  await page.locator("#replay").setInputFiles(saved!);
  await expect(page.locator("#status")).toContainText("Replay loaded");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#status")).toContainText("Replay finished");
  await expect(page.getByRole("alert")).toBeHidden();
  expect(errors).toEqual([]);
});
