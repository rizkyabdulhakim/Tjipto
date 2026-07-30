import { spawn, spawnSync } from "node:child_process";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, "../..");
const configuredBackendUrl = process.env.TJIPTO_SMOKE_BACKEND_URL;
const configuredFrontendUrl = process.env.TJIPTO_SMOKE_FRONTEND_URL;
const availablePort = () => new Promise((resolve, reject) => {
  const server = net.createServer();
  server.once("error", reject);
  server.listen(0, "127.0.0.1", () => {
    const { port } = server.address();
    server.close((error) => error ? reject(error) : resolve(port));
  });
});
const backendUrl = configuredBackendUrl ?? `http://127.0.0.1:${await availablePort()}`;
const frontendUrl = configuredFrontendUrl ?? `http://127.0.0.1:${await availablePort()}`;
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
  if (configuredBackendUrl) {
    await waitFor(`${backendUrl}/health`);
  } else {
    await start("python", ["-m", "tjipto.runtime.http"], repoRoot, {
      ...process.env, PYTHONPATH: path.join(repoRoot, "src"), TJIPTO_PORT: new URL(backendUrl).port, TJIPTO_CORS_ORIGINS: frontendUrl,
    }, `${backendUrl}/health`);
  }
  if (configuredFrontendUrl) {
    await waitFor(frontendUrl);
  } else {
    const command = process.platform === "win32" ? "cmd.exe" : "npm";
    const args = process.platform === "win32"
      ? ["/d", "/s", "/c", `npm run preview -- --host 127.0.0.1 --port ${new URL(frontendUrl).port}`]
      : ["run", "preview", "--", "--host", "127.0.0.1", "--port", new URL(frontendUrl).port];
    await start(command, args, webRoot, { ...process.env }, frontendUrl);
  }
}

async function ask(page, query) {
  await page.goto(frontendUrl, { waitUntil: "networkidle" });
  const messages = page.locator('[data-message-role="assistant"]');
  const before = await messages.count();
  await page.locator("textarea").fill(query);
  await page.getByLabel("Kirim").click();
  await page.waitForFunction((count) => document.querySelectorAll('[data-message-role="assistant"]').length > count, before);
  await page.locator('[data-message-role="assistant"]').last().getByLabel("Salin").waitFor();
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
  const payloads = [];
  let catalogSearchCount = 0;
  page.on("response", async (response) => {
    if (response.url().includes("/legal/catalog/search")) catalogSearchCount += 1;
    if (response.url().includes("/legal/")) {
      const type = response.headers()["content-type"] ?? "";
      if (type.includes("application/json")) payloads.push(await response.text());
    }
  });
  try {
    await page.goto(frontendUrl, { waitUntil: "networkidle" });
    await page.getByText("Cari Peraturan", { exact: true }).first().click();
    const searchInput = page.getByPlaceholder("Cari berdasarkan jenis, nomor, tahun, atau judul");
    await searchInput.fill("undang undang dasar negara republik indonesia tahun 1945");
    assert(catalogSearchCount === 0, "Search ran before explicit submission.");
    await page.getByLabel("Status Keberlakuan").selectOption("applicable");
    assert(catalogSearchCount === 0, "Filter change submitted Search implicitly.");
    await page.getByRole("button", { name: "Cari", exact: true }).click();
    await page.waitForFunction(() => document.body.textContent?.includes("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945"));
    assert(catalogSearchCount === 1, "Explicit Search submission did not produce exactly one request.");
    assert(await page.getByLabel("Status Keberlakuan").inputValue() === "applicable", "Submitted filter did not remain visible.");
    await page.getByRole("button", { name: "Atur ulang filter" }).click();
    assert(await page.getByLabel("Status Keberlakuan").inputValue() === "", "Filter reset did not clear selection.");
    assert(catalogSearchCount === 1, "Filter reset submitted Search implicitly.");
    const firstResult = page.getByRole("button", { name: /Buka naskah/ }).first();
    await firstResult.click();
    await page.locator('[data-evidence-panel="normal"]').waitFor();
    await page.getByLabel("Tutup panel").click();

    await ask(page, "Apa ringkasan BAB XA?");
    assert(await page.locator("[data-evidence-panel]").count() === 0, "Answer receipt opened the panel automatically.");
    const legalCitation = page.locator('[data-citation-footer="true"] button').first();
    assert(await legalCitation.getAttribute("data-citation-kind") === "legal_citation", "Legal citation is not typed.");
    await legalCitation.click();
    await page.locator('[data-evidence-panel="normal"]').waitFor();
    await page.locator('[data-bbox-highlight="active"]').first().waitFor();
    assert(await page.locator("blockquote").count() === 0, "Visible quote card remains.");
    assert(await page.getByLabel("Salin kutipan relevan").count() === 0, "Quote copy control remains.");
    const supportPayload = payloads.map((body) => {
      try { return JSON.parse(body); } catch { return null; }
    }).find((body) => body?.supports?.[0]?.authority_kind === "legal_citation");
    assert(supportPayload?.supports[0].citation_final === true, "Public support finality is not typed.");
    assert(typeof supportPayload?.supports[0].citation?.text === "string", "Deterministic citation text is missing.");
    for (const field of ["copy_text", "layout_lines", "legal_citation_available", "relevant_quote_eligible", "panel_section"]) {
      assert(!(field in supportPayload.supports[0]), `Public quote-only field remains: ${field}.`);
    }

    const document = page.locator("[data-pdf-document]").first();
    const cardMetrics = await page.evaluate(() => {
      const pdf = document.querySelector("[data-pdf-card]");
      const status = document.querySelector("[data-legal-status-card]");
      if (!pdf || !status) return null;
      const pdfRect = pdf.getBoundingClientRect();
      const statusRect = status.getBoundingClientRect();
      return {
        widthDelta: Math.abs(pdfRect.width - statusRect.width),
        pdfRadius: getComputedStyle(pdf).borderRadius,
        statusRadius: getComputedStyle(status).borderRadius,
        toolbarHeight: document.querySelector("[data-panel-toolbar]")?.getBoundingClientRect().height,
      };
    });
    assert(cardMetrics?.widthDelta <= 1, "PDF and legal-status cards do not share one content width.");
    assert(cardMetrics?.pdfRadius === cardMetrics?.statusRadius, "PDF and legal-status card radii differ.");
    assert(cardMetrics?.toolbarHeight === 48, "Panel toolbar height is inconsistent.");
    const targetPage = Number(await page.locator('[data-bbox-highlight="active"]').first().locator("xpath=ancestor::*[@data-pdf-page][1]").getAttribute("data-pdf-page"));
    assert(Number(await document.getAttribute("data-first-rendered-page")) === targetPage, "Target page did not render first.");
    const canvas = page.locator(`[data-pdf-page="${targetPage}"] canvas`);
    const at100 = await canvas.evaluate((node) => ({
      logicalWidth: node.clientWidth,
      backingWidth: node.width,
      dpr: window.devicePixelRatio,
      scrollWidth: node.closest("[data-evidence-pdf-area]")?.clientWidth ?? 0,
    }));
    assert(Math.abs(at100.backingWidth - Math.ceil(at100.logicalWidth * at100.dpr)) <= 1, "Canvas backing store does not match DPR.");
    assert(at100.logicalWidth <= at100.scrollWidth && at100.logicalWidth >= at100.scrollWidth - 80, "100% PDF does not fit viewer width.");

    await page.getByLabel("Perkecil tampilan").click();
    await page.waitForFunction((width) => {
      const node = document.querySelector("[data-bbox-highlight='active']")?.closest("[data-pdf-page]")?.querySelector("canvas");
      return node?.dataset.rendered === "true" && node.clientWidth < width * 0.8;
    }, at100.logicalWidth);
    const at75 = await canvas.evaluate((node) => node.clientWidth);
    assert(Math.abs(at75 / at100.logicalWidth - 0.75) < 0.03, "75% zoom did not re-render the PDF.");

    for (let index = 0; index < 5; index += 1) {
      await page.getByLabel("Perbesar tampilan").evaluate((button) => button.click());
    }
    await page.getByText("200%", { exact: true }).waitFor();
    await canvas.waitFor();
    await page.waitForFunction(
      (pageNumber) => document.querySelector(`[data-pdf-page="${pageNumber}"] canvas`)?.dataset.rendered === "true",
      targetPage,
    );
    const at200 = await canvas.evaluate((node) => node.clientWidth);
    assert(Math.abs(at200 / at100.logicalWidth - 2) < 0.03, "200% zoom did not re-render the PDF.");
    await page.getByLabel("Perkecil tampilan").evaluate((button) => button.click());
    await page.getByLabel("Perbesar tampilan").evaluate((button) => button.click());
    await canvas.waitFor({ state: "visible" });
    assert(await page.getByText("Naskah belum dapat ditampilkan").count() === 0, "Cancelled render was reported as a failure.");

    const renderWindowValid = await page.evaluate(() => {
      const root = document.querySelector("[data-evidence-pdf-area]");
      if (!root) return false;
      const rootRect = root.getBoundingClientRect();
      const pages = [...document.querySelectorAll("[data-pdf-page]")];
      const visible = pages.filter((node) => {
        const rect = node.getBoundingClientRect();
        return rect.bottom > rootRect.top && rect.top < rootRect.bottom;
      }).map((node) => Number(node.dataset.pdfPage));
      const canvases = [...document.querySelectorAll("[data-pdf-page] canvas")]
        .map((node) => Number(node.closest("[data-pdf-page]").dataset.pdfPage));
      return canvases.every((pageNumber) => visible.some((visiblePage) => Math.abs(pageNumber - visiblePage) <= 1));
    });
    assert(renderWindowValid, "A canvas exists outside the visible-plus-adjacent page window.");

    const separator = page.getByRole("separator", { name: "Ubah lebar panel sumber" });
    assert(await separator.getAttribute("aria-orientation") === "vertical", "Splitter orientation is missing.");
    assert(await separator.getAttribute("aria-controls") === "tjipto-evidence-panel", "Splitter controlled pane identity is missing.");
    await separator.focus();
    await page.keyboard.press("Home");
    assert(await separator.getAttribute("aria-valuenow") === await separator.getAttribute("aria-valuemin"), "Splitter Home key failed.");
    await page.keyboard.press("End");
    assert(await separator.getAttribute("aria-valuenow") === await separator.getAttribute("aria-valuemax"), "Splitter End key failed.");

    await ask(page, "BAB XA");
    await page.locator('[data-citation-footer="true"] button').first().click();
    await page.locator('[data-evidence-panel="normal"]').waitFor();
    await page.locator('[data-bbox-highlight="active"]').first().waitFor();
    const headingHighlightCount = await page.locator('[data-bbox-highlight="active"]').count();
    assert(headingHighlightCount === 2, `BAB heading did not retain its exact heading geometry (${headingHighlightCount}).`);

    await ask(page, "Apa isi BAB XA?");
    assert(await page.locator('[data-citation-footer="true"] button').count() === 10, "BAB content did not publish every direct Pasal descendant.");

    await ask(page, "kapan perubahan pertama ditetapkan");
    assert((await page.locator('[data-citation-footer="true"]').count()) === 0, "Metadata rendered as a legal quotation.");
    await openSupport(page, '[data-support-kind="metadata-support"]');

    await ask(page, "Apa isi BAB XI agama?");
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
    await disclosure.getByRole("button").first().click();
    await page.locator('[data-evidence-panel="normal"]').waitFor();

    for (const [width, height, mode] of [[390, 844, "drawer"], [768, 1024, "split"], [820, 1180, "split"], [1024, 768, "split"], [1280, 900, "split"]]) {
      await page.setViewportSize({ width, height });
      await page.locator(`[data-evidence-mode="${mode}"]`).waitFor();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
      assert(!overflow, `Viewport ${width} has horizontal content loss.`);
    }
    assert(await page.getByLabel("Panel sempit").count() === 0, "Legacy panel preset controls remain.");

    const highDprContext = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
    await highDprContext.addInitScript(({ apiBase, corpusId }) => {
      window.__TJIPTO_RUNTIME_CONFIG__ = { apiBase, corpusId };
    }, { apiBase: backendUrl, corpusId: "uud" });
    const highDprPage = await highDprContext.newPage();
    await ask(highDprPage, "Pasal 28A");
    await highDprPage.locator('[data-citation-footer="true"] button').first().click();
    const highDprCanvas = highDprPage.locator('[data-bbox-highlight="active"]').first().locator("xpath=ancestor::*[@data-pdf-page][1]").locator("canvas");
    await highDprCanvas.waitFor();
    const highDprMetrics = await highDprCanvas.evaluate((node) => ({
      logicalWidth: node.clientWidth,
      logicalHeight: node.clientHeight,
      backingWidth: node.width,
      backingHeight: node.height,
      dpr: window.devicePixelRatio,
    }));
    assert(highDprMetrics.dpr === 2, "High-DPR smoke context is not DPR 2.");
    assert(Math.abs(highDprMetrics.backingWidth - Math.ceil(highDprMetrics.logicalWidth * 2)) <= 1, "DPR 2 canvas width is incorrect.");
    assert(Math.abs(highDprMetrics.backingHeight - Math.ceil(highDprMetrics.logicalHeight * 2)) <= 1, "DPR 2 canvas height is incorrect.");
    await highDprContext.close();

    const publicText = `${(await page.locator("body").innerText())}\n${payloads.join("\n")}`;
    for (const forbidden of ["evidence_id", "legal_unit_id", "source_document_id", "source_bbox_refs", "bbox_id", "manifest_digest", "artifact_set_digest", "\"source_role\"", "\"route\"", "\"intent\"", "\"reason\"", "reason_code"]) {
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
