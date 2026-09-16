import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";
import path from "node:path";
// Replay real measured telemetry without paid calls. ID alignment is UI-only.
const measured = JSON.parse(
  readFileSync(
    path.resolve("../artifacts/outcome-pilot/attention-outcome-inference.json"),
    "utf8",
  ),
);
const metadata = JSON.parse(
  readFileSync(path.resolve("e2e/fixtures/outcome-metadata.json"), "utf8"),
);
async function fixture(page: Page) {
  await page.route("**/api/outcome-model", (route) =>
    route.fulfill({ json: metadata }),
  );
  await page.route("**/api/outcome-example?paper=attention", (route) =>
    route.fulfill({
      json: {
        title: "Synthetic browser-test paper",
        held_out: true,
        text: "This synthetic browser fixture exercises text review and streaming interactions. Its text is not used to generate the replayed scientific measurements.",
      },
    }),
  );
  await page.addInitScript(() => {
    const native = window.fetch.bind(window);
    let stream: ReadableStreamDefaultController<Uint8Array>;
    (window as any).__outcomeEvent = (value: unknown) =>
      stream.enqueue(new TextEncoder().encode(JSON.stringify(value) + "\n"));
    window.fetch = async (input, init) =>
      String(input).includes("/api/outcome-predict/stream")
        ? new Response(
            new ReadableStream({
              start(c) {
                stream = c;
              },
            }),
            { headers: { "Content-Type": "application/x-ndjson" } },
          )
        : native(input, init);
  });
  await page.goto("/");
  await expect(page.getByText(/11.18M parameters/)).toBeVisible();
  return { ...measured, model_id: metadata.model_id };
}
async function emit(page: Page, data: unknown) {
  await page.evaluate((value) => (window as any).__outcomeEvent(value), data);
}
async function start(page: Page) {
  await page.getByRole("button", { name: "Try Attention" }).click();
  await expect(
    page.getByRole("button", { name: "Analyze wording" }),
  ).toBeDisabled();
  await page
    .getByRole("checkbox", { name: "I reviewed the extracted paper text" })
    .check();
  await page.getByRole("button", { name: "Analyze wording" }).click();
  await expect(
    page.getByRole("button", { name: "Pause motion" }),
  ).toBeVisible();
}
test("stream stages, heatmaps, completion, editing and upload review", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const prediction = await fixture(page);
  await start(page);
  await emit(page, {
    type: "progress",
    stage: "masking",
    model_id: prediction.model_id,
  });
  await expect(
    page.getByRole("status").filter({ hasText: "Masking topic phrases" }),
  ).toBeVisible();
  await expect(page.getByText(/Waiting animation/)).toBeVisible();
  expect(
    await page
      .locator(".oa-progress-track")
      .evaluate((el) => getComputedStyle(el, "::after").position),
  ).toBe("absolute");
  if (info.project.name === "mobile") {
    expect(
      (await page.getByRole("button", { name: "Pause motion" }).boundingBox())!
        .height,
    ).toBeGreaterThanOrEqual(44);
  }
  await page.screenshot({
    path: `../artifacts/outcome-pilot/stream-${info.project.name}-masking.png`,
    fullPage: true,
  });
  await page.getByRole("button", { name: "Pause motion" }).click();
  await expect(page.locator(".oa-working")).toHaveCount(0);
  await expect(page.locator(".oa-spin")).toHaveCount(0);
  await page.getByRole("button", { name: "Resume motion" }).click();
  await emit(page, {
    type: "progress",
    stage: "inference",
    ...prediction,
    completed: 3,
    chunk_summaries: prediction.chunk_summaries.slice(0, 3),
  });
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: `3 / ${prediction.chunks} chunks` }),
  ).toBeVisible();
  await expect(page.locator(".oa-activation-grid span")).toHaveCount(40);
  await expect(page.locator(".oa-entropy-grid span")).toHaveCount(16);
  await expect(page.locator(".oa-chunk-trace span")).toHaveCount(3);
  await page.screenshot({
    path: `../artifacts/outcome-pilot/stream-${info.project.name}-chunks.png`,
    fullPage: true,
  });
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.locator(".oa-inspector > summary").click();
  await expect(page.getByRole("meter")).toHaveCount(4);
  await page.getByText("Exact chunk measurements", { exact: true }).click();
  await expect(
    page.getByRole("table", {
      name: "Mean supported-head probability per chunk",
    }),
  ).toBeVisible();
  await page.locator(".oa-inspector > summary").click();
  await emit(page, {
    type: "progress",
    stage: "aggregating",
    ...prediction,
    completed: prediction.chunks,
  });
  await emit(page, { type: "result", result: prediction });
  await expect(page.locator(".oa-score")).toContainText(
    String(Math.round(prediction.score * 100)),
  );
  await expect(page.getByRole("button", { name: "Pause motion" })).toHaveCount(
    0,
  );
  await expect(page.getByText("All chunks processed")).toBeVisible();
  await page.getByRole("button", { name: "Layer 2", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Layer 2", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.screenshot({
    path: `../artifacts/outcome-pilot/stream-${info.project.name}-result.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page
    .locator("#outcome-paper-text")
    .fill(
      "Edited source with enough text to perform a new analysis. ".repeat(4),
    );
  await expect(page.locator(".oa-score")).not.toContainText(
    String(Math.round(prediction.score * 100)),
  );
  await expect(
    page.getByText("Edited paper text", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Inference measurements" }),
  ).toHaveCount(0);
  await page
    .getByLabel("Upload original paper")
    .setInputFiles(path.resolve("../tests/fixtures/abstract.txt"));
  await expect(
    page.getByRole("checkbox", { name: "I reviewed the extracted paper text" }),
  ).not.toBeChecked();
  await expect(
    page.getByRole("button", { name: "Analyze wording" }),
  ).toBeDisabled();
  expect(errors).toEqual([]);
});
test("stream errors recover and reduced motion disables animation", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const prediction = await fixture(page);
  await start(page);
  await emit(page, {
    type: "progress",
    stage: "masking",
    model_id: prediction.model_id,
  });
  expect(
    await page
      .locator(".oa-stage-nodes")
      .first()
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
  expect(
    await page
      .locator(".oa-progress-track")
      .evaluate((el) => getComputedStyle(el, "::after").animationName),
  ).toBe("none");
  await emit(page, {
    type: "error",
    detail: "Research budget unavailable for test.",
  });
  await expect(page.getByRole("alert")).toContainText(
    "Research budget unavailable",
  );
  await expect(page.locator(".oa-score")).not.toContainText(
    String(Math.round(prediction.score * 100)),
  );
  await expect(
    page.getByRole("button", { name: "Analyze wording" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("region", { name: "Inference measurements" }),
  ).toHaveCount(0);
});
