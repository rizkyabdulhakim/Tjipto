import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, "../..");
const started = [];
const reuseServers = process.env.TJIPTO_SMOKE_REUSE_SERVERS === "1";
const backendUrl = process.env.TJIPTO_SMOKE_BACKEND_URL ?? "http://127.0.0.1:8000";
const frontendUrl = process.env.TJIPTO_SMOKE_FRONTEND_URL ?? "http://127.0.0.1:5173";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(url, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // retry until timeout
    }
    await sleep(300);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function isRunning(url) {
  try {
    await waitFor(url, 1000);
    return true;
  } catch {
    return false;
  }
}

async function ensureBackend() {
  const url = `${backendUrl}/health`;
  if (await isRunning(url)) {
    if (!reuseServers) {
      throw new Error(`Backend is already running on ${backendUrl}; stop it or set TJIPTO_SMOKE_REUSE_SERVERS=1.`);
    }
    return;
  }
  const port = new URL(backendUrl).port || "8000";
  const child = spawn("python", ["-m", "tjipto.runtime.http"], {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONPATH: path.join(repoRoot, "src"),
      TJIPTO_PORT: port,
      TJIPTO_CORS_ORIGINS: frontendUrl,
    },
    stdio: "ignore",
    windowsHide: true,
  });
  started.push(child);
  await waitFor(url);
}

async function ensureFrontend() {
  if (await isRunning(frontendUrl)) {
    if (!reuseServers) {
      throw new Error(`Frontend is already running on ${frontendUrl}; stop it or set TJIPTO_SMOKE_REUSE_SERVERS=1.`);
    }
    return;
  }
  const port = new URL(frontendUrl).port || "5173";
  const command = process.platform === "win32" ? "cmd.exe" : "npm";
  const args = process.platform === "win32" ? ["/d", "/s", "/c", "npm run dev -- --host 127.0.0.1 --port " + port] : ["run", "dev", "--", "--host", "127.0.0.1", "--port", port];
  const child = spawn(command, args, {
    cwd: webRoot,
    env: { ...process.env, VITE_TJIPTO_API_BASE: backendUrl },
    stdio: "ignore",
    windowsHide: true,
  });
  started.push(child);
  await waitFor(frontendUrl);
}

async function runSmoke() {
  await ensureBackend();
  await ensureFrontend();

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  try {
    await page.goto(frontendUrl, { waitUntil: "networkidle" });
    await page.getByText("Tjipto").first().waitFor();

    const composer = page.getByPlaceholder("Tanya UUD 1945...").first();
    await composer.fill("Pasal 1 ayat (3)");
    await page.getByLabel("Send").click();
    await page.getByText("Dukungan sitasi berbasis bukti").waitFor();
    await page.getByText(/SUMBER/).waitFor();

    await page.locator("button", { hasText: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945" }).first().click();
    await page.locator('[data-pdf-document="full"]').waitFor();
    await page.waitForFunction(() => {
      const document = window.document.querySelector('[data-pdf-document="full"]');
      return Number(document?.getAttribute("data-page-count") ?? "0") > 1;
    });
    await page.locator('canvas[aria-label="Halaman sumber 3"][data-rendered="true"]').waitFor();
    await page.locator('[data-bbox-highlight="active"]').first().waitFor();
    const beforeZoom = await page.locator('canvas[aria-label="Halaman sumber 3"]').boundingBox();
    await page.getByRole("button", { name: "Zoom in" }).click();
    await page.waitForFunction((width) => {
      const canvas = window.document.querySelector('canvas[aria-label="Halaman sumber 3"]');
      return canvas instanceof HTMLElement && canvas.getBoundingClientRect().width > Number(width);
    }, beforeZoom?.width ?? 0);
    await page.getByText("PDF asli dirender di frontend melalui akses backend tervalidasi").first().waitFor();
    await page.getByLabel("Simpan bookmark sementara").first().click();

    await page.getByRole("button", { name: "Cari Regulasi" }).click();
    await page.getByRole("heading", { name: "Search UUD" }).waitFor();
    await page.getByPlaceholder("Cari dalam UUD 1945...").fill("Pembukaan");
    await page.getByText(/PEMBUKAAN/).first().waitFor();
    await page.getByLabel(/Buka viewer/).first().click();
    await page.locator('[data-pdf-document="full"]').waitFor();
    await page.locator('canvas[aria-label="Halaman sumber 2"][data-rendered="true"]').waitFor();
    await page.locator('canvas[aria-label="Halaman sumber 3"][data-rendered="true"]').waitFor();
    await page.locator('[data-pdf-page="2"] [data-bbox-highlight]').first().waitFor();
    await page.locator('[data-pdf-page="3"] [data-bbox-highlight]').first().waitFor();

    await page.getByRole("button", { name: "Cari Regulasi" }).click();
    await page.getByRole("heading", { name: "Search UUD" }).waitFor();
    await page.getByPlaceholder("Cari dalam UUD 1945...").fill("BAB XA");
    await page.getByText(/BAB XA/).first().waitFor();
    await page.getByLabel(/Buka viewer/).first().click();
    await page.locator('[data-pdf-document="full"]').waitFor();
    await page.locator('[data-bbox-highlight="active"]').first().waitFor();

    await page.getByRole("button", { name: "Pustaka Hukum" }).click();
    await page.getByRole("heading", { name: "Library" }).waitFor();
    await page.getByText("temporary_process_memory").waitFor();
    await page.getByText("Sample Prompts").waitFor();

    const fallbackPage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    await fallbackPage.route("**/legal/uud/viewer", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "viewer_payload_ready",
          pdf_access_available: false,
          rendering_available: false,
          render_status: "render_unavailable",
          evidence_id: "uud_current_consolidated_final_citation_evidence_00237",
          page_numbers: [3],
          bbox_rectangles: [],
        }),
      }),
    );
    await fallbackPage.goto(frontendUrl, { waitUntil: "networkidle" });
    await fallbackPage.getByPlaceholder("Tanya UUD 1945...").fill("Pasal 1 ayat (3)");
    await fallbackPage.getByLabel("Send").click();
    await fallbackPage.getByText("Dukungan sitasi berbasis bukti").waitFor();
    await fallbackPage.locator("button", { hasText: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945" }).first().click();
    await fallbackPage.getByText("Rendering PDF/BBox belum tersedia").first().waitFor();
  } finally {
    await browser.close();
  }
}

try {
  await runSmoke();
  console.log("smoke:ok");
} finally {
  for (const child of started.reverse()) {
    if (process.platform === "win32") {
      spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
    } else {
      child.kill();
    }
  }
}
