import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { parseReplay } from "../../replay";

interface TestPad {
  id: string;
  index: number;
  connected: boolean;
  mapping: "" | "standard";
  buttons: { pressed: boolean; touched: boolean; value: number }[];
  axes: number[];
}

declare global {
  interface Window { testGamepads: (TestPad | null)[] }
}

function receiver(mapping: "" | "standard" = "", index = 0): TestPad {
  return {
    id: mapping ? "Standard gamepad" : "6B controller (Vendor: 0ca3 Product: 0024)",
    index, connected: true, mapping, axes: [0, 0],
    buttons: Array.from({ length: 16 }, () => ({ pressed: false, touched: false, value: 0 })),
  };
}

async function expose(page: Page, pads: (TestPad | null)[]): Promise<void> {
  await page.addInitScript((pads) => {
    window.testGamepads = pads;
    Object.defineProperty(navigator, "getGamepads", { value: () => window.testGamepads });
  }, pads);
  await page.goto("/");
}

async function axis(page: Page, values: number[]): Promise<void> {
  await page.evaluate((values) => {
    const pad = window.testGamepads[0];
    if (!pad) throw new Error("Missing test gamepad.");
    pad.axes = values;
  }, values);
}

async function press(page: Page, index: number, pressed: boolean): Promise<void> {
  await page.evaluate(({ index, pressed }) => {
    const pad = window.testGamepads[0];
    if (!pad) throw new Error("Missing test gamepad.");
    pad.buttons[index] = { pressed, touched: pressed, value: pressed ? 1 : 0 };
  }, { index, pressed });
}

test("explains that an OS device may not yet be exposed by the browser", async ({ page }) => {
  await expose(page, []);
  await expect(page.locator("#gamepad-status")).toContainText("No gamepad exposed by the browser");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(245, 190, 98)");
  await expect(page.locator("#gamepad-status")).toContainText("press a controller button");
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await expect(page.getByRole("button", { name: "Set up buttons / D-pad" })).toBeDisabled();
});

test("uses standard bindings immediately and does not switch pads on disconnect", async ({ page }) => {
  await expose(page, [receiver("standard"), receiver("", 1)]);
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await expect(page.locator("#gamepad-status")).toContainText("standard mapping ready");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(94, 217, 173)");
  await press(page, 0, true);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: B");
  await press(page, 0, false);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: neutral");
  await page.evaluate(() => { window.testGamepads[0] = null; });
  await expect(page.locator("#gamepad-status")).toContainText("Selected gamepad disconnected");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(245, 190, 98)");
  await expect(page.locator("#gamepad-live")).toHaveText("No selected gamepad input.");
});

test("detects a non-standard receiver before ROM load and learns D-pad and face buttons", async ({ page }) => {
  await expose(page, [receiver()]);
  await expect(page.locator("#gamepad option:checked")).toContainText("6B controller");
  await expect(page.locator("#gamepad-status")).toContainText("non-standard mapping");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(245, 190, 98)");
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await press(page, 0, true);
  await expect(page.locator("#gamepad-live")).toContainText("Buttons (16) pressed: 0");
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: neutral");
  await press(page, 0, false);
  await page.getByRole("button", { name: "Set up buttons / D-pad" }).click();
  await expect(page.locator("#gamepad-status")).toContainText("Set UP:");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(98, 200, 245)");
  await axis(page, [0, -1]);
  await expect(page.locator("#gamepad-status")).toContainText("Set DOWN:");
  // Holding Up must not map the next action to the same input.
  await page.waitForTimeout(120);
  await expect(page.locator("#gamepad-status")).toContainText("Set DOWN:");
  await axis(page, [0, 1]);
  await expect(page.locator("#gamepad-status")).toContainText("Set LEFT:");
  await axis(page, [-1, 0]);
  await expect(page.locator("#gamepad-status")).toContainText("Set RIGHT:");
  await axis(page, [1, 0]);
  await expect(page.locator("#gamepad-status")).toContainText("Set A:");
  await axis(page, [0, 0]);
  for (const [index, next] of [[0, "B"], [1, "C"], [2, "START"]] as const) {
    await press(page, index, true);
    await expect(page.locator("#gamepad-status")).toContainText(`Set ${next}:`);
    await press(page, index, false);
    await expect(page.locator("#gamepad-live")).toContainText("Buttons (16) pressed: none");
  }
  await press(page, 9, true);
  await expect(page.locator("#gamepad-status")).toContainText("custom mapping ready");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(94, 217, 173)");
  await press(page, 9, false);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: neutral");
  await press(page, 0, true);
  await axis(page, [-1, 0]);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: LEFT + A");

  if (process.env.NHL94_ROM) {
    await page.locator("#rom").setInputFiles(process.env.NHL94_ROM);
    await expect(page.locator("#status")).toContainText("ROM loaded locally");
    await page.getByRole("button", { name: "Step one frame" }).click();
    await expect(page.locator("#frame")).toHaveText("1");
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download input replay" }).click();
    const replay = await (await downloadPromise).path();
    expect(parseReplay(JSON.parse(await readFile(replay!, "utf8"))).inputs[0]?.[0]).toBe(576);
  }

  await press(page, 0, false);
  await axis(page, [0, 0]);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: neutral");
  await page.getByRole("button", { name: "Reset mapping", exact: true }).click();
  await expect(page.locator("#gamepad-status")).toContainText("non-standard mapping");
});

test("shows blocked or unavailable API access without breaking keyboard UI", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "getGamepads", {
      value: () => { throw new DOMException("Gamepad access blocked", "SecurityError"); },
    });
  });
  await page.goto("/");
  await expect(page.locator("#gamepad-status")).toContainText("Browser gamepad access failed");
  await expect(page.locator("#gamepad-status")).toHaveCSS("color", "rgb(248, 113, 113)");
  await expect(page.locator("#gamepad-live")).toContainText("keyboard controls still work");
  await expect(page.locator("#rom")).toBeEnabled();
});
