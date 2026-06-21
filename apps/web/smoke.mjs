import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, "../..");
const started = [];
const reuseServers = process.env.TJIPTO_SMOKE_REUSE_SERVERS === "1";

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
  const url = "http://127.0.0.1:8000/health";
  if (await isRunning(url)) {
    if (!reuseServers) {
      throw new Error("Backend is already running on 127.0.0.1:8000; stop it or set TJIPTO_SMOKE_REUSE_SERVERS=1.");
    }
    return;
  }
  const child = spawn("python", ["-m", "tjipto.runtime.http"], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: path.join(repoRoot, "src") },
    stdio: "ignore",
    windowsHide: true,
  });
  started.push(child);
  await waitFor(url);
}

async function ensureFrontend() {
  const url = "http://127.0.0.1:5173";
  if (await isRunning(url)) {
    if (!reuseServers) {
      throw new Error("Frontend is already running on 127.0.0.1:5173; stop it or set TJIPTO_SMOKE_REUSE_SERVERS=1.");
    }
    return;
  }
  const command = process.platform === "win32" ? "cmd.exe" : "npm";
  const args = process.platform === "win32" ? ["/d", "/s", "/c", "npm run dev"] : ["run", "dev"];
  const child = spawn(command, args, {
    cwd: webRoot,
    stdio: "ignore",
    windowsHide: true,
  });
  started.push(child);
  await waitFor(url);
}

async function runSmoke() {
  await ensureBackend();
  await ensureFrontend();

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  try {
    await page.goto("http://127.0.0.1:5173", { waitUntil: "networkidle" });
    await page.getByText("Tjipto").first().waitFor();

    const composer = page.getByPlaceholder("Tanya UUD 1945...").first();
    await composer.fill("Pasal 1 ayat (3)");
    await composer.press("Enter");
    await page.getByText("Dukungan sitasi berbasis bukti").waitFor();
    await page.getByText(/SUMBER/).waitFor();

    await page.locator("button", { hasText: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945" }).first().click();
    await page.getByText("Rendering PDF/BBox belum tersedia").first().waitFor();
    await page.getByLabel("Simpan bookmark sementara").first().click();

    await page.getByRole("button", { name: "Cari Regulasi" }).click();
    await page.getByRole("heading", { name: "Search UUD" }).waitFor();
    await page.getByPlaceholder("Cari dalam UUD 1945...").fill("BAB XA");
    await page.getByText(/BAB XA/).first().waitFor();

    await page.getByRole("button", { name: "Pustaka Hukum" }).click();
    await page.getByRole("heading", { name: "Library" }).waitFor();
    await page.getByText("temporary_process_memory").waitFor();
    await page.getByText("Sample Prompts").waitFor();
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
