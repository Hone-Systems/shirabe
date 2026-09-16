import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("agent evidence can be explored, filtered, exported and reached directly", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/evidence");
  await expect(
    page.getByRole("heading", { name: "What happened next." }),
  ).toBeVisible();
  await page.getByLabel("Filter research method").selectOption("agent");
  await page.getByLabel("Search papers").fill("Attention Is All You Need");
  await expect(
    page.getByRole("heading", {
      name: "Attention Is All You Need",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".rubric-dimension details")).toHaveCount(20);
  await expect(
    page.getByText("Autonomous agent · sources read directly"),
  ).toBeVisible();
  await page
    .getByText("Research summary & search trail", { exact: true })
    .click();
  expect(await page.locator(".evidence-method ol li").count()).toBeGreaterThan(
    0,
  );
  await page.getByRole("button", { name: "Coverage map", exact: true }).click();
  await expect(page.locator(".evidence-map")).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export filtered evidence" }).click();
  expect((await download).suggestedFilename()).toBe(
    "shirabe-evaluation-data.json",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByLabel("Search papers").fill("First Mo");
  await expect(
    page.getByText("Rubric not applicable", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".evidence-map circle")).toHaveCount(0);
  await expect(page.getByRole("note")).toContainText("Excluded from training");
  await expect(page.locator(".outcome-bounds")).toHaveCount(0);
  expect(errors).toEqual([]);
});
