import { expect, test } from "@playwright/test";

test("replaces RAM observations with all pause-menu stats below Bot Configuration", async ({ page }) => {
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Game Stats", exact: true });
  const table = panel.getByRole("table", { name: "Game stats", exact: true });
  await expect(table.getByRole("columnheader")).toHaveText(["Statistic", "Home", "Away"]);
  await expect(table.locator("tbody th")).toHaveText([
    "Score", "Shots on goal", "Shooting %", "Power play (goals/opportunities)",
    "Power play time", "Power play shots", "Shorthanded goals",
    "Breakaways (goals/attempts)", "One-timers (goals/attempts)",
    "Penalty shots (goals/attempts)", "Faceoffs won", "Body checks", "Penalties / minutes",
    "Attack zone time", "Passes (completed/attempted)",
  ]);
  await expect(table.locator("tbody td")).toHaveText(new Array(30).fill("-"));
  await expect(page.getByRole("heading", { name: "Live RAM observations" })).toHaveCount(0);
  await expect(page.locator("#observations")).toHaveCount(0);
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 960 });
    const bot = await page.locator(".bot-configuration").boundingBox(), stats = await panel.boundingBox();
    expect(stats!.y).toBeGreaterThanOrEqual(bot!.y + bot!.height);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  }
});

test("updates native stats through the real worker and resets them with the ROM", async ({ page }) => {
  test.skip(!process.env.NHL94_ROM, "Set NHL94_ROM to a legally supplied local original ROM.");
  await page.goto("/");
  await page.locator("#rom").setInputFiles(process.env.NHL94_ROM!);
  await expect(page.locator("#status")).toContainText("ROM loaded locally");
  await expect(page.locator('#game-stats-body [data-stat="score"] td')).toHaveText(["0", "0"]);
  await expect(page.locator('#game-stats-body [data-stat="passing"] td')).toHaveText(["0 / 0 (0%)", "0 / 0 (0%)"]);
  await expect(page.locator('#game-stats-body [data-stat="attack-zone"] td')).toHaveText(["0:00", "0:00"]);
  await expect(page.locator("#game-stats-status")).toHaveText("Live match statistics.");
  await page.getByRole("button", { name: "Step one frame" }).click();
  await expect(page.locator("#frame")).toHaveText("1");
  await expect(page.getByRole("table", { name: "Game stats", exact: true })).not.toContainText("NaN");
  await page.getByRole("button", { name: "Restart match" }).click();
  await expect(page.locator("#frame")).toHaveText("0");
  await expect(page.locator('#game-stats-body [data-stat="score"] td')).toHaveText(["0", "0"]);
});
