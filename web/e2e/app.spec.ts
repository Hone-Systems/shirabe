import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";
import path from "node:path";
const specimen = readFileSync(
  path.resolve("../tests/fixtures/abstract.txt"),
  "utf8",
);

test("analysis, representation, export, edit and route preservation", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByText("MODEL READY", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Analyze writing style" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Load example" }).click();
  await page.getByRole("button", { name: "Analyze writing style" }).click();
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /^evidence ·/ }).click();
  await expect(
    page.getByRole("button", { name: /^evidence ·/ }),
  ).toHaveAttribute("aria-pressed", "true");
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByLabel("Mask content words").check();
  await expect(page.locator(".tokens")).toContainText("[content]");
  await page.getByRole("button", { name: /Inspect all .* features/ }).click();
  await expect(page.locator(".formula")).toContainText("sigmoid");
  const dl = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export analysis" }).click();
  expect((await dl).suggestedFilename()).toBe("shirabe-analysis.json");
  await page
    .getByRole("link", { name: "Training notebook", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "The experiment, exposed." }),
  ).toBeFocused();
  await page.getByRole("link", { name: "Paper analysis", exact: true }).click();
  await expect(page.getByLabel("Abstract text")).toHaveValue(specimen);
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Abstract text")
    .fill(
      specimen
        .replaceAll("measured", "studied")
        .replaceAll("results", "findings")
        .replaceAll("estimates", "findings")
        .replaceAll("accounting", "allowing"),
    );
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).not.toBeVisible();
  await expect(page.locator(".notice")).toHaveCount(0);
  await page.getByRole("button", { name: "Analyze writing style" }).click();
  await expect(page.getByRole("button", { name: /^All ·/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByRole("button", { name: "Clear abstract" }).click();
  await expect(page.getByLabel("Abstract text")).toBeEmpty();
  expect(errors).toEqual([]);
});

test("uploads require editable review and malformed inputs recover", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .locator("#paper-upload")
    .setInputFiles(path.resolve("../tests/fixtures/paper.pdf"));
  await expect(page.locator(".notice")).toContainText(
    "Review the extracted text",
  );
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).not.toBeVisible();
  expect(await page.getByLabel("Abstract text").inputValue()).not.toContain(
    "This body should not",
  );
  await page.getByRole("button", { name: "Analyze writing style" }).click();
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).toBeVisible();
  await page.locator("#paper-upload").setInputFiles({
    name: "broken.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("invalid"),
  });
  await expect(page.getByRole("alert")).toContainText("not a valid PDF");
  await page
    .locator("#paper-upload")
    .setInputFiles(path.resolve("../tests/fixtures/abstract.txt"));
  await expect(page.getByLabel("Abstract text")).toHaveValue(specimen);
  await page.getByLabel("Abstract text").fill("Too short.");
  await page.getByRole("button", { name: "Analyze writing style" }).click();
  await expect(page.getByRole("alert")).toContainText("80–800");
});

test("training evidence, filters, charts, exports and accessibility", async ({
  page,
  request,
}) => {
  const report = await (await request.get("/api/report")).json();
  await page.goto("/training");
  await expect(
    page.getByRole("heading", { name: "The experiment, exposed." }),
  ).toBeVisible();
  await expect(page.locator(".stat-grid")).toContainText(
    report.test.roc_auc.toFixed(3),
  );
  await page.getByRole("button", { name: "Optimization", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Optimization", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page
    .getByRole("button", { name: "Inspect chart data" })
    .first()
    .click();
  await expect(page.locator(".chart-data")).toBeVisible();
  await page.getByLabel("Field", { exact: true }).selectOption("17");
  await expect(page.locator(".cohort-table tbody tr")).toHaveCount(6);
  await page
    .locator("summary")
    .filter({ hasText: "MODEL WEIGHTS & REPRODUCIBILITY" })
    .click();
  await expect(page.locator(".details tbody tr")).toHaveCount(
    report.feature_count,
  );
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  const dl = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run" }).click();
  expect((await dl).suggestedFilename()).toBe("shirabe-training-report.json");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("server error and retry, no stale score after failed request", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Load example" }).click();
  await page.route("**/api/predict", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Model is unavailable for this test." }),
    }),
  );
  await page.getByRole("button", { name: "Analyze writing style" }).click();
  await expect(page.getByRole("alert")).toContainText("unavailable");
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).not.toBeVisible();
  await page.unroute("**/api/predict");
  await page.getByRole("button", { name: "Analyze writing style" }).click();
  await expect(
    page.getByText("ANALYSIS COMPLETE", { exact: true }),
  ).toBeVisible();
});
