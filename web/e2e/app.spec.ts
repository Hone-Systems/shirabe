import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";
import path from "node:path";
const specimen = readFileSync(
  path.resolve("../tests/fixtures/abstract.txt"),
  "utf8",
);

test("single-screen analysis, actual data, animation controls and inspection", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Try an example" }),
  ).toBeEnabled();
  expect(
    await page
      .locator(".topbar,.workspace,.visual-stage,.bottom-strip")
      .evaluateAll((elements) =>
        elements.every((el) => {
          const b = el.getBoundingClientRect();
          return b.left >= -1 && b.right <= innerWidth + 1;
        }),
      ),
  ).toBe(true);
  await expect(
    page.getByRole("button", { name: "Analyze paper", exact: true }),
  ).toBeDisabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Try an example" }).click();
  await expect(page.getByText("SIGNAL RESOLVED")).toBeVisible();
  const rail = page.locator(".right-rail");
  if (await rail.isVisible()) {
    const bounds = await rail.boundingBox();
    const exportBounds = await rail
      .getByRole("button", { name: "Export analysis" })
      .boundingBox();
    expect(exportBounds!.y + exportBounds!.height).toBeLessThanOrEqual(
      bounds!.y + bounds!.height + 1,
    );
  }
  const expected = await (
    await request.post("/api/predict", { data: { text: specimen } })
  ).json();
  await expect(page.locator(".score")).toContainText(
    (expected.probability * 100).toFixed(1),
  );
  await expect(page.getByText("Synthetic abstract · demo only")).toBeAttached();
  await page.getByLabel("Mask tokens").check();
  await expect(page.locator(".token-stream")).toContainText("[content]");
  await page.getByRole("button", { name: "Next feature" }).click();
  await expect(page.locator(".feature-readout")).not.toBeEmpty();
  await page.getByRole("button", { name: "Pause animation" }).click();
  await expect(
    page.getByRole("button", { name: "Resume animation" }),
  ).toHaveAttribute("aria-pressed", "true");
  expect(
    await page
      .locator(".signal-particle")
      .first()
      .evaluate((el) => getComputedStyle(el).animationPlayState),
  ).toBe("paused");
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: "Full trace", exact: true }).click();
  await expect(
    page
      .getByRole("region", { name: "All feature contributions" })
      .locator("tbody tr"),
  ).toHaveCount(256);
  await page
    .getByText("Attention values · layers, heads and tokens", { exact: true })
    .click();
  await expect(
    page
      .getByRole("region", { name: "Attention by layer and head" })
      .locator("tbody tr"),
  ).toHaveCount(16);
  await expect(
    page
      .getByRole("region", { name: "Token attention values" })
      .locator("tbody tr"),
  ).toHaveCount(expected.tokens.length);
  const analysisDownload = page.waitForEvent("download");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Export analysis" })
    .click();
  expect((await analysisDownload).suggestedFilename()).toBe(
    "shirabe-analysis.json",
  );
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "About this experiment" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "About this experiment" }),
  ).toBeFocused();
  await page.getByRole("button", { name: "Training", exact: true }).click();
  await expect(page.locator(".training-title")).toBeVisible();
  await page.getByRole("button", { name: "Analyze", exact: true }).click();
  await expect(page.getByText("SIGNAL RESOLVED")).toBeVisible();
  await page
    .getByRole("button", { name: "Edit abstract", exact: true })
    .last()
    .click();
  await expect(page.getByLabel("Abstract text")).toHaveValue(specimen);
  await page.getByLabel("Abstract text").fill("Too short.");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Analyze paper" })
    .click();
  await expect(page.getByRole("dialog").getByRole("alert")).toContainText(
    "80–800",
  );
  await page.getByLabel("Abstract text").fill(specimen);
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Analyze paper" })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByText("SIGNAL RESOLVED")).toBeVisible();
  expect(errors).toEqual([]);
});

test("PDF and TXT uploads are reviewed, errors recover", async ({ page }) => {
  await page.goto("/");
  await page
    .locator("#paper-upload")
    .setInputFiles(path.resolve("../tests/fixtures/paper.pdf"));
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.locator(".notice")).toContainText("Review");
  expect(await page.getByLabel("Abstract text").inputValue()).not.toContain(
    "This body should not",
  );
  await expect(page.getByText("SIGNAL RESOLVED")).toHaveCount(0);
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Analyze paper" })
    .click();
  await expect(page.getByText("SIGNAL RESOLVED")).toBeVisible();
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
  await page.getByRole("button", { name: "Close dialog" }).click();
  await page
    .getByRole("button", { name: "Paste abstract", exact: true })
    .count();
});

test("training stays on screen and full record preserves measured evidence", async ({
  page,
  request,
}) => {
  const report = await (await request.get("/api/report")).json();
  await page.goto("/training");
  await expect(page.locator(".training-title>strong")).toHaveText(
    report.test.roc_auc.toFixed(3),
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "About this experiment" }).click();
  await page.getByRole("button", { name: "Read the experiment" }).click();
  await expect(
    page.getByRole("dialog", { name: "Experiment record" }),
  ).toBeVisible();
  await page.getByLabel("Field", { exact: true }).selectOption("17");
  await expect(
    page.getByRole("region", { name: "Cohort data" }).locator("tbody tr"),
  ).toHaveCount(6);
  await page.getByText("All 256 head weights", { exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Model weights" }).locator("tbody tr"),
  ).toHaveCount(report.feature_count);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  const dl = page.waitForEvent("download");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Export run" })
    .click();
  const download = await dl;
  expect(download.suggestedFilename()).toBe("shirabe-training-report.json");
  const downloaded = JSON.parse(readFileSync((await download.path())!, "utf8"));
  expect(downloaded.test.roc_auc).toBe(report.test.roc_auc);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("failed inference has no stale score and reduced motion stops animation", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.route("**/api/predict", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Model unavailable for this test." }),
    }),
  );
  await page.getByRole("button", { name: "Try an example" }).click();
  await expect(page.getByRole("alert")).toContainText("unavailable");
  await expect(page.getByText("SIGNAL RESOLVED")).toHaveCount(0);
  await page.unroute("**/api/predict");
  await page
    .getByRole("button", { name: "Analyze paper", exact: true })
    .click();
  await expect(page.getByText("SIGNAL RESOLVED")).toBeVisible();
  expect(
    await page
      .locator(".signal-particle")
      .first()
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
});
