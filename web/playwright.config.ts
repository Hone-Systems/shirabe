import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  workers: 2,
  timeout: 30000,
  reporter: [
    ["list"],
    ["json", { outputFile: "../artifacts/e2e-results.json" }],
  ],
  use: { baseURL: "http://127.0.0.1:8790", trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command:
      "uv run --directory .. uvicorn shirabe.api:app --host 127.0.0.1 --port 8790",
    url: "http://127.0.0.1:8790/api/health",
    reuseExistingServer: false,
    timeout: 30000,
  },
});
