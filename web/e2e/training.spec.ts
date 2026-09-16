import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("outcome training exposes measured curves, controls, and RL readiness", async ({
  page,
  request,
}) => {
  const report = await (await request.get("/api/outcome-training")).json();
  await page.goto("/training");
  await expect(
    page.getByRole("heading", { name: "Inside the fit." }),
  ).toBeVisible();
  await expect(page.locator(".training-status")).toContainText(
    report.dataset.splits.train.no === 0
      ? "NO NEGATIVE TRAINING LABELS"
      : "NOT PROMOTED",
  );
  await expect(page.locator(".training-chart")).toHaveCount(
    report.experiments?.length ? 9 : 6,
  );
  for (const [id, label] of [
    ["wording", "Wording"],
    ["original", "Original text"],
    ["topic", "Topic control"],
  ]) {
    await page.getByRole("button", { name: label, exact: true }).click();
    const run = report.runs.find((r: { id: string }) => r.id === id);
    await expect(page.locator(".training-metrics")).toContainText(
      run.history.at(-1).train_loss.toFixed(3),
    );
    await page.getByLabel("Inspect epoch").fill("1");
    await expect(page.locator(".training-metrics")).toContainText(
      run.history[0].train_loss.toFixed(3),
    );
    await page.getByRole("button", { name: "Latest", exact: true }).click();
  }
  await page.getByLabel("Evaluation split").selectOption("challenge");
  const baseline = report.runs.find((r: { id: string }) => r.id === "topic")
    .constant_positive_baseline.challenge.brier;
  await expect(
    page.locator(".comparison-bars").first().locator(":scope > div").last(),
  ).toContainText(baseline.toFixed(3));
  expect(
    await page
      .locator(".comparison-bars")
      .first()
      .locator(":scope > div")
      .last()
      .locator("i")
      .evaluate((el) => el.getBoundingClientRect().width),
  ).toBeGreaterThan(0);
  await expect(page.getByText("RL · not run", { exact: true })).toBeVisible();
  await page
    .getByText("Inspect exact measurements & full questions", { exact: true })
    .click();
  await expect(
    page
      .getByRole("region", { name: "Epoch measurements" })
      .locator("tbody tr"),
  ).toHaveCount(
    report.runs.find((r: { id: string }) => r.id === "topic").history.length,
  );
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export training report" }).click();
  expect((await download).suggestedFilename()).toBe(
    "shirabe-outcome-training.json",
  );
  await page
    .getByRole("button", { name: "Citation baseline", exact: true })
    .click();
  await expect(page).toHaveURL(/view=citation/);
  await page.reload();
  await expect(page.locator(".training-title")).toBeVisible();
});
