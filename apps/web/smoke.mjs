import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, "../..");
const backendUrl = process.env.TJIPTO_SMOKE_BACKEND_URL ?? "http://127.0.0.1:8000";
const frontendUrl = process.env.TJIPTO_SMOKE_FRONTEND_URL ?? "http://127.0.0.1:5173";
const started = [];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const assert = (condition, message) => { if (!condition) throw new Error(message); };

async function waitFor(url) {
  for (let attempts = 0; attempts < 50; attempts += 1) {
    try { if ((await fetch(url)).ok) return; } catch { /* retry */ }
    await sleep(300);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function start(command, args, cwd, env, healthUrl) {
  const child = spawn(command, args, { cwd, env, stdio: "ignore", windowsHide: true });
  started.push(child);
  await waitFor(healthUrl);
}

async function ensureServers() {
  try { await waitFor(`${backendUrl}/health`); } catch {
    await start("python", ["-m", "tjipto.runtime.http"], repoRoot, {
      ...process.env, PYTHONPATH: path.join(repoRoot, "src"), TJIPTO_PORT: new URL(backendUrl).port, TJIPTO_CORS_ORIGINS: frontendUrl,
    }, `${backendUrl}/health`);
  }
  try { await waitFor(frontendUrl); } catch {
    const command = process.platform === "win32" ? "cmd.exe" : "npm";
    const args = process.platform === "win32"
      ? ["/d", "/s", "/c", `npm run preview -- --host 127.0.0.1 --port ${new URL(frontendUrl).port}`]
      : ["run", "preview", "--", "--host", "127.0.0.1", "--port", new URL(frontendUrl).port];
    await start(command, args, webRoot, { ...process.env }, frontendUrl);
  }
}

async function ask(page, query) {
  await page.goto(frontendUrl, { waitUntil: "networkidle" });
  const statuses = page.locator("[data-runtime-status]");
  const before = await statuses.count();
  await page.locator("textarea").fill(query);
  await page.getByLabel("Send").click();
  await page.waitForFunction((count) => document.querySelectorAll("[data-runtime-status]").length > count, before);
}

async function openSupport(page, selector) {
  const group = page.locator(selector).first();
  await group.waitFor();
  if (!await group.evaluate((node) => node.open)) await group.locator("summary").click();
  await group.getByRole("button").first().click();
  await page.locator("[data-evidence-panel]").waitFor();
  await page.locator("[data-bbox-highlight]").first().waitFor();
}

async function run() {
  await ensureServers();
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await context.addInitScript(({ apiBase, corpusId }) => {
    window.__TJIPTO_RUNTIME_CONFIG__ = { apiBase, corpusId };
  }, { apiBase: backendUrl, corpusId: "uud" });
  const page = await context.newPage();
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"], { origin: frontendUrl });
  const payloads = [];
  page.on("response", async (response) => {
    if (response.url().includes("/legal/")) {
      const type = response.headers()["content-type"] ?? "";
      if (type.includes("application/json")) payloads.push(await response.text());
    }
  });
  try {
    await ask(page, "Pasal 1 ayat (3)");
    await page.locator('[data-citation-footer="true"] button').first().click();
    await page.locator('[data-evidence-panel="normal"]').waitFor();
    await page.locator('[data-bbox-highlight="active"]').first().waitFor();
    const quote = page.locator("blockquote").first();
    await page.getByLabel("Salin kutipan relevan").click();
    const fullCopy = await page.evaluate(() => navigator.clipboard.readText());
    assert(fullCopy.length > 0 && !/[*]{1,4}\)/.test(fullCopy), "Full copy is not canonical legal text.");
    await page.evaluate(() => {
      document.addEventListener("copy", (event) => {
        window.__tjiptoCopyTypes = Array.from(event.clipboardData?.types ?? []);
      }, { once: true });
    });
    await quote.selectText();
    await page.keyboard.press(process.platform === "darwin" ? "Meta+C" : "Control+C");
    const selectedCopy = await page.evaluate(() => navigator.clipboard.readText());
    const selectedTypes = await page.evaluate(() => window.__tjiptoCopyTypes);
    assert(selectedCopy === fullCopy && !/[*]{1,4}\)/.test(selectedCopy), "Selected copy is not canonical text/plain.");
    assert(JSON.stringify(selectedTypes) === JSON.stringify(["text/plain"]), "Selected copy published a non-text MIME payload.");

    const partialSelection = await page.evaluate(() => {
      const quote = document.querySelector("blockquote");
      const walker = document.createTreeWalker(quote, NodeFilter.SHOW_TEXT);
      const text = walker.nextNode();
      if (!text) throw new Error("Legal quote has no selectable text.");
      const range = document.createRange();
      range.setStart(text, 0);
      range.setEnd(text, Math.min(12, text.textContent?.length ?? 0));
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      return selection?.toString() ?? "";
    });
    await page.evaluate(() => {
      document.addEventListener("copy", (event) => {
        window.__tjiptoCopyTypes = Array.from(event.clipboardData?.types ?? []);
      }, { once: true });
    });
    await page.keyboard.press(process.platform === "darwin" ? "Meta+C" : "Control+C");
    const partialCopy = await page.evaluate(() => navigator.clipboard.readText());
    const partialTypes = await page.evaluate(() => window.__tjiptoCopyTypes);
    const canonicalSubset = partialSelection.replace(/\r\n?/g, "\n").split("\n").map((line) => line.trimStart()).join("\n").trim();
    assert(partialCopy === canonicalSubset, "Selected subset copy is not the canonical selected text.");
    assert(fullCopy.replace(/\s+/g, " ").includes(partialCopy.replace(/\s+/g, " ")), "Selected subset falls outside canonical legal text.");
    assert(JSON.stringify(partialTypes) === JSON.stringify(["text/plain"]), "Selected subset published a non-text MIME payload.");

    await ask(page, "kapan perubahan pertama ditetapkan");
    assert((await page.locator('[data-citation-footer="true"]').count()) === 0, "Metadata rendered as a legal quotation.");
    await openSupport(page, '[data-support-kind="metadata-support"]');

    await ask(page, "BAB XI agama");
    await openSupport(page, '[data-support-kind="structure-support"]');

    await ask(page, "pasal yang dihapus");
    assert((await page.locator('[data-citation-footer="true"]').count()) === 0, "Relation rendered as a legal quotation.");
    await openSupport(page, '[data-support-kind="trace-support"]');

    await ask(page, "siapa wakil ketua yang tercantum dalam Perubahan Pertama?");
    const disclosure = page.locator('[data-support-kind="metadata-support"]').first();
    const wasOpen = await disclosure.evaluate((node) => node.open);
    await disclosure.locator("summary").focus();
    await page.keyboard.press("Enter");
    assert(await disclosure.evaluate((node) => node.open) !== wasOpen, "Support group is not keyboard-operable.");
    if (!await disclosure.evaluate((node) => node.open)) await disclosure.locator("summary").click();
    assert((await disclosure.getByRole("button").count()) === 7, "Grouped members lost exact viewer targets.");

    const publicText = `${(await page.locator("body").innerText())}\n${payloads.join("\n")}`;
    for (const forbidden of ["evidence_id", "legal_unit_id", "source_document_id", "source_bbox_refs", "bbox_id", "manifest_digest", "artifact_set_digest"]) {
      assert(!publicText.includes(forbidden), `Public surface leaked ${forbidden}.`);
    }
    console.log("smoke:ok");
  } finally {
    await browser.close();
    for (const child of started.reverse()) {
      if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
      else child.kill();
    }
  }
}

await run();
