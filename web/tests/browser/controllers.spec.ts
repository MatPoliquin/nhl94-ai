import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { parseReplay } from "../../replay";

test("shows unsupported bot configuration below the game controls", async ({ page }) => {
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Bot Configuration Coming later" });
  await expect(panel).toBeVisible();
  await expect(panel.getByText("Preview only.", { exact: false })).toBeVisible();
  for (const frequency of ["60 Hz", "15 Hz", "10 Hz", "6 Hz"]) {
    const radio = panel.getByRole("radio", { name: frequency, exact: true });
    await expect(radio).toBeDisabled();
    if (frequency === "60 Hz") await expect(radio).toBeChecked();
    else await expect(radio).not.toBeChecked();
  }
  for (const label of ["AI Goalie", "Consider player ratings"]) {
    await expect(panel.getByRole("checkbox", { name: label, exact: true })).toBeDisabled();
    await expect(panel.getByRole("checkbox", { name: label, exact: true })).not.toBeChecked();
  }
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 960 });
    const controls = await page.locator(".virtual-controls").boundingBox();
    const config = await panel.boundingBox();
    expect(controls).not.toBeNull();
    expect(config).not.toBeNull();
    expect(config!.y).toBeGreaterThanOrEqual(controls!.y + controls!.height);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  }
});

test("shows four controller rows, exact occupant order, team choices and two disabled future slots", async ({ page }) => {
  await page.goto("/");
  const defaults = ["human", "diagnostic", "none", "none"];
  for (let number = 1; number <= 4; number++) {
    const player = page.getByRole("combobox", { name: `Controller ${number} player`, exact: true });
    const team = page.getByRole("combobox", { name: `Controller ${number} team`, exact: true });
    await expect(player.locator("option")).toHaveText(["Human", "Diagnostic bot", "None"]);
    await expect(player).toHaveValue(defaults[number - 1]!);
    await expect(team.locator("option")).toHaveText(["Team 1 (Home)", "Team 2 (Away)"]);
    await expect(team).toHaveValue(number % 2 === 1 ? "1" : "2");
    if (number > 2) {
      await expect(player).toBeDisabled();
      await expect(team).toBeDisabled();
    } else {
      await expect(player).toBeEnabled();
      await expect(team).toBeEnabled();
    }
  }
  const controllerPanel = page.locator("section.panel").filter({
    has: page.getByRole("heading", { name: "2. Configure controllers" }),
  });
  const inputPanel = page.locator("section.panel").filter({
    has: page.getByRole("heading", { name: "3. Input device" }),
  });
  await expect(controllerPanel.locator("#gamepad")).toHaveCount(0);
  await expect(inputPanel.getByRole("combobox", { name: "Gamepad device" })).toBeVisible();
  await expect(inputPanel.locator(".gamepad-details")).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "4. Inspect and replay" })).toBeVisible();
});

test("supports None and bot-vs-bot choices without pretending a Human owns the input devices", async ({ page }) => {
  await page.goto("/");
  await page.locator("#controller-1-agent").selectOption("none");
  await expect(page.locator("#controller-1-team")).toBeDisabled();
  await expect(page.locator("#controller-note")).toContainText("No Human slot");
  await page.locator("#controller-1-agent").selectOption("diagnostic");
  await expect(page.locator("#controller-1-team")).toBeEnabled();
  await expect(page.locator("#controller-note")).toContainText("No Human slot");
  await page.locator("#controller-1-agent").selectOption("human");
  await page.locator("#controller-2-agent").selectOption("human");
  await expect(page.locator("#controller-note")).toContainText("Two humans");
  await page.locator("#controller-2-agent").selectOption("none");
  await expect(page.locator("#controller-note")).toContainText("go to the Human slot");
});

test("pauses on configuration changes, clears None input and locks setup during replay", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Set NHL94_ROM to a legally supplied local original ROM.");
  await page.goto("/");
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("ROM loaded locally");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect.poll(async () => Number(await page.locator("#frame").textContent())).toBeGreaterThan(4);
  await page.locator("#controller-1-team").selectOption("2");
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();
  await expect(page.locator("#status")).toContainText("Controller setup changed");
  await expect(page.getByRole("button", { name: "Step one frame" })).toBeEnabled();
  await page.locator("#controller-1-agent").selectOption("none");
  const frame = Number(await page.locator("#frame").textContent());
  await page.locator("#screen").focus();
  await page.keyboard.down("x");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText(String(frame + 1));
  await page.keyboard.up("x");
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download input replay" }).click();
  const replay = await (await downloadPromise).path();
  expect(parseReplay(JSON.parse(await readFile(replay!, "utf8"))).inputs.at(-1)?.[0]).toBe(0);
  await page.locator("#replay").setInputFiles(replay!);
  await expect(page.locator("#status")).toContainText("Replay loaded");
  for (const number of [1, 2]) {
    await expect(page.locator(`#controller-${number}-agent`)).toBeDisabled();
    await expect(page.locator(`#controller-${number}-team`)).toBeDisabled();
  }
  await expect(page.locator("#controller-note")).toContainText("recorded controller inputs");
  await page.getByRole("button", { name: "Restart match" }).click();
  await expect(page.locator("#controller-1-agent")).toBeEnabled();
  await expect(page.locator("#controller-2-agent")).toBeEnabled();
});
