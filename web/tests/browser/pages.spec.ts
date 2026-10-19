import { expect, test } from "@playwright/test";
import { AVAILABLE_BOTS } from "../../bot";

test("loads local heading fonts and keeps body text, controls and stats readable", async ({ page }) => {
  const fontRequests: string[] = [];
  page.on("request", (request) => {
    if (request.resourceType() === "font") fontRequests.push(request.url());
  });
  await page.goto("/");
  const loaded = await page.evaluate(async () => {
    const result: { weight: string; status: string }[] = [];
    for (const weight of ["600", "700"]) {
      const faces = await document.fonts.load(`${weight} 24px "Barlow Condensed"`);
      for (const face of faces) result.push({ weight: face.weight, status: face.status });
    }
    return result;
  });
  expect(loaded).toEqual([{ weight: "600", status: "loaded" }, { weight: "700", status: "loaded" }]);
  expect(fontRequests.length).toBeGreaterThanOrEqual(2);
  expect(fontRequests.every((url) => new URL(url).origin === new URL(page.url()).origin)).toBe(true);
  await expect(page.locator("h1")).toHaveCSS("font-family", /Barlow Condensed/);
  await expect(page.locator("h1")).toHaveCSS("font-weight", "700");
  await expect(page.locator("h2").first()).toHaveCSS("font-family", /Barlow Condensed/);
  await expect(page.locator("h2").first()).toHaveCSS("font-weight", "600");
  await expect(page.locator("html")).toHaveCSS("font-family", "system-ui, sans-serif");
  await expect(page.locator("html")).toHaveCSS("font-size", "15px");
  await expect(page.locator("#play")).toHaveCSS("font-size", "14px");
  await expect(page.locator("#controller-1-agent")).toHaveCSS("font-size", "13px");
  await expect(page.locator(".game-stats table")).toHaveCSS("font-size", "14px");
  await expect(page.locator('[data-stat="score"] td').first()).toHaveCSS("font-variant-numeric", "tabular-nums");
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 960 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  }
  const license = await page.request.get("/fonts/barlow-condensed-LICENSE.txt");
  expect(license.ok()).toBe(true);
  expect(await license.text()).toContain("SIL OPEN FONT LICENSE Version 1.1");
});

test("uses the ice-blue palette consistently across the arena and active navigation", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("html")).toHaveCSS("background-color", "rgb(17, 28, 46)");
  await expect(page.locator("html")).toHaveCSS("color", "rgb(241, 245, 249)");
  await expect(page.locator(".panel").first()).toHaveCSS("background-color", "rgb(28, 45, 69)");
  await expect(page.locator(".panel").first()).toHaveCSS("border-top-color", "rgb(75, 101, 133)");
  await expect(page.locator("#controller-1-agent")).toHaveCSS("background-color", "rgb(42, 64, 96)");
  await expect(page.locator(".intro").first()).toHaveCSS("color", "rgb(168, 184, 204)");
  await expect(page.locator("#play")).toHaveCSS("background-color", "rgb(98, 200, 245)");
  await expect(page.locator("#play")).toHaveCSS("color", "rgb(17, 28, 46)");
  await expect(page.locator("#status")).toHaveCSS("color", "rgb(94, 217, 173)");
  await expect(page.locator(".notice")).toHaveCSS("color", "rgb(245, 190, 98)");
  await expect(page.locator('.game-stats thead th').last()).toHaveCSS("color", "rgb(248, 113, 113)");
  await expect(page.locator('[data-stat="score"] td').first()).toHaveCSS("color", "rgb(98, 200, 245)");
  await expect(page.locator('[data-stat="score"] td').last()).toHaveCSS("color", "rgb(248, 113, 113)");
  await expect(page.locator("#screen")).toHaveCSS("filter", "none");
  await page.goto("/about.html");
  await expect(page.locator('.site-nav [aria-current="page"]')).toHaveCSS("background-color", "rgb(98, 200, 245)");
  await expect(page.locator('.site-nav [aria-current="page"]')).toHaveCSS("color", "rgb(17, 28, 46)");
});

test("opens the top-right information links without unloading the arena", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("NHL94 AI CONTEST");
  await expect(page.getByText("UNDER CONSTRUCTION", { exact: true })).toBeVisible();
  await expect(page.getByText("NHL94 AI / LOCAL DEVELOPMENT", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Genesis Plus GX, one native frame at a time.", { exact: false })).toHaveCount(0);
  if (process.env.NHL94_ROM) {
    await page.locator("#rom").setInputFiles(process.env.NHL94_ROM);
    await expect(page.locator("#status")).toContainText("ROM loaded locally");
    await page.getByRole("button", { name: "Step one frame" }).click();
    await expect(page.locator("#frame")).toHaveText("1");
  }
  const frame = await page.locator("#frame").textContent();
  const navigation = page.getByRole("navigation", { name: "Site navigation" });
  await expect(navigation.getByRole("link")).toHaveText(["Bots", "Help", "About"]);
  for (const name of ["Bots", "Help", "About"]) {
    const popupPromise = page.waitForEvent("popup");
    await navigation.getByRole("link", { name, exact: true }).click();
    const popup = await popupPromise;
    await expect(popup).toHaveTitle(`${name} - NHL94 AI CONTEST`);
    await expect(popup.getByRole("heading", { name, level: 1, exact: true })).toBeVisible();
    await expect(popup).toHaveURL(new RegExp(`/${name.toLowerCase()}\\.html$`));
    await popup.close();
    await expect(page.locator("#frame")).toHaveText(frame!);
  }
});

test("lists the same available bots and descriptions as the controller catalogue", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/diagnostic bot/i)).toHaveCount(0);
  const choices = await page.locator("#controller-1-agent option").allTextContents();
  await page.goto("/bots.html");
  const table = page.getByRole("table", { name: "Available bots" });
  await expect(table.locator("tbody th")).toHaveText(choices.slice(1, -1));
  await expect(table.locator("tbody th")).toHaveText(["ROM AI", "Classic AI"]);
  await expect(table.locator("tbody td")).toHaveText(AVAILABLE_BOTS.map((bot) => bot.description));
  await expect(page.getByText("Classic supports the Bot Configuration controls", { exact: false })).toBeVisible();
  await expect(page.getByText(/diagnostic/i)).toHaveCount(0);
});

test("documents actual controls, setup, unsupported options and replay limits", async ({ page }) => {
  await page.goto("/help.html");
  await expect(page.getByText("controller 2 to Classic AI/Away.", { exact: false })).toBeVisible();
  await expect(page.getByText("ROM AI does not send scripted button inputs.", { exact: false })).toBeVisible();
  await expect(page.getByText(/diagnostic bot/i)).toHaveCount(0);
  for (const heading of [
    "1. Load your game", "2. Configure controllers", "3. Choose an input device",
    "Playback and game statistics", "4. Record and replay", "Current scope",
  ]) await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
  await expect(page.getByText("Bot Configuration applies to Classic AI.", { exact: true })).toBeVisible();
  await expect(page.getByText("108,000 native frames", { exact: false })).toBeVisible();
  await page.getByRole("link", { name: "Back to arena" }).click();
  await expect(page.getByRole("heading", { name: "NHL94 AI CONTEST", exact: false })).toBeVisible();
});

test("shows the requested About content and official resource links", async ({ page }) => {
  await page.goto("/about.html");
  await expect(page.getByText("nhl94ai.com is not affiliated with EA.", { exact: false })).toBeVisible();
  await expect(page.getByText("a site for AI research", { exact: false })).toBeVisible();
  await expect(page.getByRole("link", { name: "mathieu.poliquin@gmail.com" }))
    .toHaveAttribute("href", "mailto:mathieu.poliquin@gmail.com");
  await expect(page.getByRole("link", { name: "stable-retro (Farama website)" }))
    .toHaveAttribute("href", "https://stable-retro.farama.org/");
  await expect(page.getByRole("link", { name: "NHL '94 Discord community" }))
    .toHaveAttribute("href", "https://discord.gg/kvYSWWC");
});

test("keeps page navigation usable without overflow at desktop and mobile widths", async ({ page }) => {
  for (const name of ["bots", "help", "about"]) {
    await page.goto(`/${name}.html`);
    await expect(page.locator('.site-nav [aria-current="page"]')).toHaveAttribute("href", `./${name}.html`);
    for (const width of [1280, 390]) {
      await page.setViewportSize({ width, height: 960 });
      await expect(page.getByRole("navigation", { name: "Site navigation" })).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    }
  }
});

test("loads all static pages and assets under a GitHub Pages-style project subpath", async ({ page, context }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await context.route("**/nhl94-ai/**", async (route) => {
    const url = new URL(route.request().url());
    url.pathname = url.pathname.replace(/^\/nhl94-ai/, "");
    const response = await route.fetch({ url: url.href });
    await route.fulfill({ response });
  });
  await page.goto("/nhl94-ai/about.html");
  await expect(page.getByRole("heading", { name: "About", level: 1 })).toBeVisible();
  expect(await page.evaluate(async () =>
    (await document.fonts.load('700 30px "Barlow Condensed"')).length)).toBe(1);
  await page.getByRole("navigation").getByRole("link", { name: "Bots" }).click();
  await expect(page).toHaveURL(/\/nhl94-ai\/bots\.html$/);
  await expect(page.locator("#bot-list tr")).toHaveCount(AVAILABLE_BOTS.length);
  await page.getByRole("navigation").getByRole("link", { name: "Help" }).click();
  await expect(page).toHaveURL(/\/nhl94-ai\/help\.html$/);
  await page.getByRole("link", { name: "Back to arena" }).click();
  await expect(page).toHaveURL(/\/nhl94-ai\/index\.html$/);
  await expect(page.locator("#controller-1-agent")).toHaveValue("human");
  await expect(page.locator(".site-nav a").first()).toHaveAttribute("href", "./bots.html");
  expect(errors).toEqual([]);
});
