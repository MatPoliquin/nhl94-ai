import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { parseReplay } from "../../replay";
import type { FramePacket } from "../../protocol";
import { GAMEPAD_PROFILES_KEY } from "../../gamepad-profiles";

interface TestPad {
  id: string;
  index: number;
  connected: boolean;
  mapping: "" | "standard";
  buttons: { pressed: boolean; touched: boolean; value: number }[];
  axes: number[];
}

declare global {
  interface Window {
    testGamepads: (TestPad | null)[];
    gamepadFrame: Pick<FramePacket, "frame" | "observation" | "inputs"> | undefined;
  }
}

function receiver(mapping: "" | "standard" = "", index = 0): TestPad {
  return {
    id: mapping ? "Standard gamepad" : "6B controller (Vendor: 0ca3 Product: 0024)",
    index, connected: true, mapping, axes: [0, 0],
    buttons: Array.from({ length: 16 }, () => ({ pressed: false, touched: false, value: 0 })),
  };
}

function m30Receiver(): TestPad {
  return { ...receiver(), axes: [0, 0, 0], buttons: receiver().buttons.slice(0, 10) };
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

async function learnButtons(page: Page, buttons: number[]): Promise<void> {
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await page.getByRole("button", { name: "Set up buttons / D-pad" }).click();
  for (const [step, button] of buttons.entries()) {
    await press(page, button, true);
    if (step === 7) await expect(page.locator("#gamepad-status")).toContainText("custom mapping ready");
    else await expect(page.locator("#gamepad-status")).toContainText(`Set ${["DOWN", "LEFT", "RIGHT", "A", "B", "C", "START"][step]}:`);
    await press(page, button, false);
    await expect(page.locator("#gamepad-live")).toContainText("pressed: none");
  }
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

test("automatically maps the M30 6B receiver without running setup", async ({ page }) => {
  await expose(page, [m30Receiver()]);
  await expect(page.locator("#gamepad-status")).toContainText("M30 / 6B receiver mapping ready");
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await expect(page.getByRole("button", { name: "Reset mapping", exact: true })).toBeDisabled();
  for (const [button, action] of [[0, "A"], [1, "B"], [5, "C"], [9, "START"]] as const) {
    await press(page, button, true);
    await expect(page.locator("#gamepad-live")).toContainText(`Genesis input: ${action}`);
    await press(page, button, false);
    await expect(page.locator("#gamepad-live")).toContainText("Genesis input: neutral");
  }
  await axis(page, [-1, -1, 0]);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: UP + LEFT");
  await axis(page, [1, 1, 0]);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: DOWN + RIGHT");
  await axis(page, [0, 0, 0]);
  if (process.env.NHL94_ROM) {
    await page.locator("#controller-2-agent").selectOption("classic");
    await page.locator("#rom").setInputFiles(process.env.NHL94_ROM);
    await expect(page.locator("#status")).toContainText("ROM loaded locally");
    await press(page, 0, true);
    await axis(page, [1, 0, 0]);
    await page.getByRole("button", { name: "Step one frame" }).click();
    await expect(page.locator("#frame")).toHaveText("1");
    await press(page, 0, false);
    await axis(page, [0, 0, 0]);
    await press(page, 5, true);
    await expect(page.locator("#gamepad-live")).toContainText("Genesis input: C");
    await page.getByRole("button", { name: "Step one frame" }).click();
    await expect(page.locator("#frame")).toHaveText("2");
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download input replay" }).click();
    const path = await (await downloadPromise).path();
    const inputs = parseReplay(JSON.parse(await readFile(path!, "utf8"))).inputs;
    expect(inputs[0]?.[0]).toBe(512 | 128);
    expect(inputs[1]?.[0]).toBe(256);
  }
});

test("keeps a custom M30 override across reloads and index changes; reset restores the preset", async ({ page }) => {
  await expose(page, [m30Receiver()]);
  await learnButtons(page, [3, 4, 5, 6, 7, 2, 1, 8]);
  await expect(page.locator("#status")).toHaveText("Gamepad setup complete. Press Play to resume.");
  await press(page, 7, true);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: A");
  await page.reload();
  await expect(page.locator("#gamepad-status")).toContainText("custom mapping ready");
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await press(page, 7, true);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: A");
  await page.evaluate(() => {
    const pad = window.testGamepads[0]!;
    pad.index = 1;
    window.testGamepads = [null, pad];
  });
  await expect(page.locator("#gamepad-status")).toContainText("Selected gamepad disconnected");
  await page.locator("#gamepad").selectOption({ label: "2: 6B controller (Vendor: 0ca3 Product: 0024)" });
  await expect(page.locator("#gamepad-status")).toContainText("custom mapping ready");
  await expect(page.locator("#gamepad-live")).toContainText("A: button 7");
  await page.getByRole("button", { name: "Reset mapping", exact: true }).click();
  await expect(page.locator("#gamepad-status")).toContainText("M30 / 6B receiver mapping ready");
  await page.reload();
  await expect(page.locator("#gamepad-status")).toContainText("M30 / 6B receiver mapping ready");
});

test("warns if mapping storage is blocked while automatic and current-tab custom controls still work", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "localStorage", { get() { throw new DOMException("Storage blocked", "SecurityError"); } });
  });
  await expose(page, [m30Receiver()]);
  await expect(page.locator("#gamepad-status")).toContainText("M30 / 6B receiver mapping ready");
  await expect(page.locator("#gamepad-status")).toContainText("Saved mappings could not be loaded");
  await learnButtons(page, [3, 4, 5, 6, 7, 2, 1, 8]);
  await expect(page.locator("#gamepad-status")).toContainText("custom controls will only last for this tab");
  await press(page, 7, true);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: A");
  await expect(page.locator("#rom")).toBeEnabled();
});

test("reports corrupt saved mappings and allows reset to recover without disabling the M30 preset", async ({ page }) => {
  await page.addInitScript((key) => localStorage.setItem(key, "{"), GAMEPAD_PROFILES_KEY);
  await expose(page, [m30Receiver()]);
  await expect(page.locator("#gamepad-status")).toContainText("Saved mappings could not be loaded");
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await page.getByRole("button", { name: "Reset mapping", exact: true }).click();
  await expect(page.locator("#gamepad-status")).toContainText("M30 / 6B receiver mapping ready");
  await expect(page.locator("#gamepad-status")).not.toContainText("Storage warning");
});

test("routes a configured receiver into the native Human player while Classic controls Away", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Requires a legally supplied original ROM.");
  await page.addInitScript(() => {
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event) => {
          if (event.data.type === "frame") {
            const { frame, observation, inputs } = event.data.packet;
            window.gamepadFrame = { frame, observation, inputs };
          }
        });
      }
    };
  });
  await expose(page, [receiver()]);
  await page.locator("#controller-2-agent").selectOption("classic");
  await page.locator(".gamepad-details").evaluate((details) => details.setAttribute("open", ""));
  await page.getByRole("button", { name: "Set up buttons / D-pad" }).click();
  for (const [step, button] of [12, 13, 14, 15, 0, 1, 2, 9].entries()) {
    await press(page, button, true);
    if (step === 7) await expect(page.locator("#gamepad-status")).toContainText("custom mapping ready");
    else await expect(page.locator("#gamepad-status")).toContainText(`Set ${["DOWN", "LEFT", "RIGHT", "A", "B", "C", "START"][step]}:`);
    await press(page, button, false);
    await expect(page.locator("#gamepad-live")).toContainText("Buttons (16) pressed: none");
  }
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("ROM loaded locally");
  const initial = await page.evaluate(() => window.gamepadFrame!.observation.controllers[0]);
  await press(page, 15, true);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: RIGHT");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText("1");
  expect(await page.evaluate(() => window.gamepadFrame!.inputs[0])).toBe(128);
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.gamepadFrame!.frame)).toBeGreaterThan(240);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const moved = await page.evaluate(() => window.gamepadFrame!.observation.controllers[0]);
  expect(moved.team).toBe(1);
  expect(moved.player).not.toEqual(initial.player);
  await press(page, 15, false);
  await press(page, 2, true);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: C");
  const pressedFrame = await page.evaluate(() => window.gamepadFrame!.frame + 1);
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(pressedFrame));
  expect(await page.evaluate(() => window.gamepadFrame!.inputs[0])).toBe(256);
  await press(page, 2, false);
  await expect(page.locator("#gamepad-live")).toContainText("Genesis input: neutral");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(pressedFrame + 1));
  expect(await page.evaluate(() => window.gamepadFrame!.inputs[0])).toBe(0);
});
