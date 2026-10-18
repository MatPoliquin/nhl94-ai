import { expect, test } from "@playwright/test";

test("shows the honest smoke-test scope and rejects invalid local files", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("This is the emulator test, not Classic AI.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeDisabled();
  await page.locator("#rom").setInputFiles({
    name: "invalid.bin", mimeType: "application/octet-stream", buffer: Buffer.from("not a ROM"),
  });
  await expect(page.getByRole("alert")).toContainText("between 512 bytes and 4 MiB");
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
  await page.locator("#port").selectOption("1");
  await page.locator("#screen").focus();
  const pressedFrame = Number(await page.locator("#frame").textContent()) + 1;
  await page.keyboard.down("x");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(pressedFrame));
  const pressed = JSON.parse(await page.locator("#observations").textContent() ?? "{}");
  expect(pressed.lastInputs).toEqual([0, 1]);
  await page.keyboard.up("x");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(pressedFrame + 1));
  const released = JSON.parse(await page.locator("#observations").textContent() ?? "{}");
  expect(released.lastInputs).toEqual([0, 0]);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const download = await downloadPromise;
  const saved = await download.path();
  expect(saved).not.toBeNull();
  await page.getByRole("button", { name: "Restart ROM" }).click();
  await expect(page.locator("#frame")).toHaveText("0");
  await page.locator("#replay").setInputFiles(saved!);
  await expect(page.locator("#status")).toContainText("Replay loaded");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#status")).toContainText("Replay finished");
  await expect(page.getByRole("alert")).toBeHidden();
  expect(errors).toEqual([]);
});
