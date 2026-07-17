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

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

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
  const args = process.platform === "win32"
    ? ["/d", "/s", "/c", "npm run preview -- --host 127.0.0.1 --port " + port]
    : ["run", "preview", "--", "--host", "127.0.0.1", "--port", port];
  const child = spawn(command, args, {
    cwd: webRoot,
    env: { ...process.env, VITE_TJIPTO_API_BASE: backendUrl },
    stdio: "ignore",
    windowsHide: true,
  });
  started.push(child);
  await waitFor(frontendUrl);
}

async function ask(page, query) {
  await page.goto(frontendUrl, { waitUntil: "networkidle" });
  await page.getByPlaceholder("Tanya UUD 1945...").fill(query);
  await page.getByLabel("Send").click();
  await page.locator("[data-runtime-status]").first().waitFor();
}

async function expectNoExactCitationUi(page) {
  assert((await page.locator('[data-citation-footer="true"]').count()) === 0, "Non-exact support rendered as citation footer.");
  assert((await page.locator("[data-evidence-panel]").count()) === 0, "Non-exact support opened evidence panel.");
}

async function assertHighlightGeometry(page) {
  const result = await page.locator('[data-bbox-highlight="active"]').first().evaluate((highlight) => {
    const pageElement = highlight.closest("[data-pdf-page]");
    const canvas = pageElement?.querySelector('canvas[data-rendered="true"]');
    if (!(highlight instanceof HTMLElement) || !(canvas instanceof HTMLCanvasElement)) return null;
    const highlightBox = highlight.getBoundingClientRect();
    const canvasBox = canvas.getBoundingClientRect();
    const expectedLeft = canvasBox.left + (Number.parseFloat(highlight.style.left) / 100) * canvasBox.width;
    const expectedTop = canvasBox.top + (Number.parseFloat(highlight.style.top) / 100) * canvasBox.height;
    const expectedWidth = (Number.parseFloat(highlight.style.width) / 100) * canvasBox.width;
    const expectedHeight = (Number.parseFloat(highlight.style.height) / 100) * canvasBox.height;
    return {
      page: Number(pageElement?.getAttribute("data-pdf-page")),
      visible: highlightBox.width > 0 && highlightBox.height > 0,
      overlapsPage:
        Math.min(highlightBox.right, canvasBox.right) > Math.max(highlightBox.left, canvasBox.left) &&
        Math.min(highlightBox.bottom, canvasBox.bottom) > Math.max(highlightBox.top, canvasBox.top),
      error: Math.max(
        Math.abs(highlightBox.left - expectedLeft),
        Math.abs(highlightBox.top - expectedTop),
        Math.abs(highlightBox.width - expectedWidth),
        Math.abs(highlightBox.height - expectedHeight),
      ),
    };
  });
  assert(result?.page > 0, "Highlight did not resolve to its declared PDF page.");
  assert(result?.visible, "Highlight rectangle is not visible.");
  assert(result?.overlapsPage, "Highlight rectangle does not overlap its PDF page.");
  assert((result?.error ?? Number.POSITIVE_INFINITY) <= 1, "Rendered highlight geometry exceeds 1 CSS pixel error.");
}

async function runEvidenceContractSmoke(browser) {
  const provenancePage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await ask(provenancePage, "Apa konflik sumber Pasal 25E dan Pasal 25A Perubahan Kedua?");
  await provenancePage.locator('[data-runtime-status="limited_answer"]').waitFor();
  await provenancePage.locator('[data-provenance-note="true"]').waitFor();
  await provenancePage.locator('[data-citation-kind="source_conflict_provenance"]').first().waitFor();
  await provenancePage.locator('[data-citation-footer="true"] button[data-citation-kind="source_conflict_provenance"]').first().click();
  await provenancePage.locator('[data-evidence-panel="normal"]').waitFor();
  await provenancePage.locator('[data-bbox-highlight="active"]').first().waitFor();
  await assertHighlightGeometry(provenancePage);
  await provenancePage.close();

  const metadataPage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await ask(metadataPage, "kapan perubahan pertama ditetapkan");
  await metadataPage.locator('[data-runtime-status="answer_ready"]').waitFor();
  await metadataPage.locator('[data-support-kind="metadata-support"]').waitFor();
  await metadataPage.locator('[data-support-kind="metadata-support"]').getByText("19 Oktober 1999").waitFor();
  await metadataPage.locator('[data-citation-kind="metadata_source"]').first().waitFor();
  await metadataPage.locator('[data-citation-footer="true"] button[data-citation-kind="metadata_source"]').first().click();
  await metadataPage.locator('[data-evidence-panel="normal"]').waitFor();
  await metadataPage.locator('[data-bbox-highlight="active"]').first().waitFor();
  await assertHighlightGeometry(metadataPage);
  await metadataPage.close();

  const tracePage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await ask(tracePage, "Apa konflik sumber Aturan Tambahan Pasal III Perubahan Keempat?");
  await tracePage.locator('[data-runtime-status="limited_answer"]').waitFor();
  await tracePage.locator('[data-provenance-note="true"]').waitFor();
  await tracePage.locator('[data-citation-kind="source_anomaly"]').first().waitFor();
  await tracePage.locator('[data-citation-footer="true"] button[data-citation-kind="source_anomaly"]').first().click();
  await tracePage.locator('[data-evidence-panel="normal"]').waitFor();
  await tracePage.locator('[data-bbox-highlight="active"]').first().waitFor();
  await assertHighlightGeometry(tracePage);
  await tracePage.close();

  const relationPage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await ask(relationPage, "UUD 1945 diubah oleh amandemen berapa");
  await relationPage.locator('[data-runtime-status="answer_ready"]').waitFor();
  await relationPage.locator('[data-support-kind="document-relations"]').waitFor();
  await expectNoExactCitationUi(relationPage);
  await relationPage.close();

  const articleRelationPage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await ask(articleRelationPage, "Pasal 25E menjadi Pasal 25A");
  await articleRelationPage.locator('[data-runtime-status="answer_ready"]').waitFor();
  const relationSupport = articleRelationPage.locator('[data-support-kind="article-relations"]');
  await relationSupport.waitFor();
  await relationSupport.getByText(/Pasal 25E.*Pasal 25A/).waitFor();
  await relationSupport.getByText(/Source proof exact/).waitFor();
  await relationSupport.getByText(/target target_local/).waitFor();
  await articleRelationPage.locator('[data-citation-footer="true"] button').first().click();
  await articleRelationPage.locator('[data-evidence-panel="normal"]').waitFor();
  await articleRelationPage.locator('[data-bbox-highlight="active"]').first().waitFor();
  await assertHighlightGeometry(articleRelationPage);
  await articleRelationPage.close();

  const insufficientPage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  await ask(insufficientPage, "siapa presiden indonesia sekarang?");
  await insufficientPage.locator('[data-runtime-status="insufficient_evidence"]').waitFor();
  await insufficientPage.getByText("Bukti tidak cukup").waitFor();
  await expectNoExactCitationUi(insufficientPage);
  assert((await insufficientPage.locator("[data-support-kind]").count()) === 0, "Insufficient evidence rendered support rows.");
  await insufficientPage.close();
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
    assert((await page.locator("[data-support-kind]").count()) === 0, "Exact citation answer rendered non-exact support rows.");

    await page.locator("button", { hasText: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945" }).first().click();
    await page.locator('[data-pdf-document="full"]').waitFor();
    await page.locator('[data-evidence-panel="normal"]').waitFor();
    await page.waitForFunction(() => {
      const document = window.document.querySelector('[data-pdf-document="full"]');
      return Number(document?.getAttribute("data-page-count") ?? "0") > 1;
    });
    await page.locator('canvas[aria-label="Halaman sumber 3"][data-rendered="true"]').waitFor();
    await page.locator('[data-bbox-highlight="active"]').first().waitFor();
    await assertHighlightGeometry(page);
    await page.waitForFunction(() => !document.body.textContent?.includes("1 / 1"));
    await page.waitForFunction(() => !document.body.textContent?.includes("Hal. 3"));
    await page.locator('[data-evidence-detail-area="normal"]').waitFor();
    const separatedScroll = await page.evaluate(() => {
      const pdfArea = document.querySelector('[data-evidence-pdf-area="normal"]');
      const detailArea = document.querySelector('[data-evidence-detail-area="normal"]');
      if (!(pdfArea instanceof HTMLElement) || !(detailArea instanceof HTMLElement)) return null;
      const beforePdfTop = pdfArea.scrollTop;
      detailArea.scrollTop = 40;
      return {
        pdfScrollable: pdfArea.scrollHeight > pdfArea.clientHeight,
        detailScrollable: detailArea.scrollHeight > detailArea.clientHeight,
        pdfUnchanged: pdfArea.scrollTop === beforePdfTop,
      };
    });
    assert(separatedScroll?.pdfScrollable, "PDF area is not independently scrollable.");
    assert(separatedScroll?.detailScrollable, "Evidence detail area is not independently scrollable.");
    assert(separatedScroll?.pdfUnchanged, "Detail scroll changed the PDF scroll position.");
    const toolbarAlignment = await page.evaluate(() => {
      const pdfArea = document.querySelector('[data-evidence-pdf-area="normal"]');
      const zoomOut = document.querySelector('button[aria-label="Zoom out"]');
      const bookmark = document.querySelector('button[aria-label="Simpan bookmark sementara"]');
      if (!(pdfArea instanceof HTMLElement) || !(zoomOut instanceof HTMLElement) || !(bookmark instanceof HTMLElement)) return null;
      const pdfBox = pdfArea.getBoundingClientRect();
      const zoomBox = zoomOut.getBoundingClientRect();
      const bookmarkBox = bookmark.getBoundingClientRect();
      const padding = Number.parseFloat(getComputedStyle(pdfArea).paddingLeft);
      return {
        leftDelta: Math.abs(zoomBox.left - (pdfBox.left + padding)),
        rightDelta: Math.abs(bookmarkBox.right - (pdfBox.right - padding)),
      };
    });
    assert((toolbarAlignment?.leftDelta ?? 99) < 4, "Zoom controls are not aligned with the PDF viewport left edge.");
    assert((toolbarAlignment?.rightDelta ?? 99) < 4, "Bookmark control is not aligned with the PDF viewport right edge.");
    const panelBeforeResize = await page.locator('[data-evidence-panel="normal"]').boundingBox();
    const resizeHandle = page.locator('[data-evidence-resize-handle="true"]');
    await resizeHandle.waitFor();
    const handleBox = await resizeHandle.boundingBox();
    if (!panelBeforeResize || !handleBox) throw new Error("Evidence panel resize target unavailable.");
    await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + 120);
    await page.mouse.down();
    await page.mouse.move(handleBox.x - 140, handleBox.y + 120, { steps: 6 });
    await page.mouse.up();
    await page.waitForFunction((width) => {
      const panel = window.document.querySelector('[data-evidence-panel="normal"]');
      return panel instanceof HTMLElement && panel.getBoundingClientRect().width > Number(width) + 80;
    }, panelBeforeResize.width);
    await page.locator('[data-bbox-highlight]').first().waitFor();

    const beforeZoom = await page.locator('canvas[aria-label="Halaman sumber 3"]').boundingBox();
    const beforeZoomState = await page.evaluate(() => {
      const pdfArea = document.querySelector('[data-evidence-pdf-area="normal"]');
      const page3 = document.querySelector('[data-pdf-page="3"]');
      if (!(pdfArea instanceof HTMLElement) || !(page3 instanceof HTMLElement)) return null;
      return {
        scrollTop: pdfArea.scrollTop,
        pageTop: page3.getBoundingClientRect().top,
      };
    });
    await page.getByRole("button", { name: "Zoom in" }).click();
    await page.waitForFunction((width) => {
      const canvas = window.document.querySelector('canvas[aria-label="Halaman sumber 3"]');
      return canvas instanceof HTMLElement && canvas.getBoundingClientRect().width > Number(width);
    }, beforeZoom?.width ?? 0);
    await assertHighlightGeometry(page);
    const afterZoomState = await page.evaluate(() => {
      const pdfArea = document.querySelector('[data-evidence-pdf-area="normal"]');
      const page3 = document.querySelector('[data-pdf-page="3"]');
      if (!(pdfArea instanceof HTMLElement) || !(page3 instanceof HTMLElement)) return null;
      return {
        scrollTop: pdfArea.scrollTop,
        pageTop: page3.getBoundingClientRect().top,
      };
    });
    assert((afterZoomState?.scrollTop ?? 0) > 0, "Zoom jumped the PDF scroll to the top.");
    assert(Math.abs((afterZoomState?.pageTop ?? 0) - (beforeZoomState?.pageTop ?? 0)) < 160, "Zoom moved the active page too far from the reading position.");
    const panelAfterResize = await page.locator('[data-evidence-panel="normal"]').boundingBox();
    await page.getByRole("button", { name: "Expand PDF-only mode" }).click();
    await page.locator('[data-evidence-panel="expanded"]').waitFor();
    await page.locator('[data-evidence-pdf-area="expanded"]').waitFor();
    await page.locator('[data-bbox-highlight]').first().waitFor();
    await assertHighlightGeometry(page);
    await page.getByRole("button", { name: "Zoom in" }).click();
    await page.waitForFunction((width) => {
      const canvas = window.document.querySelector('canvas[aria-label="Halaman sumber 3"]');
      return canvas instanceof HTMLElement && canvas.getBoundingClientRect().width > Number(width);
    }, beforeZoom?.width ?? 0);
    await page.getByRole("button", { name: "Exit PDF-only mode" }).click();
    await page.locator('[data-evidence-panel="normal"]').waitFor();
    await page.locator('[data-evidence-pdf-area="normal"]').waitFor();
    await page.locator('[data-bbox-highlight]').first().waitFor();
    await page.waitForFunction((width) => {
      const panel = window.document.querySelector('[data-evidence-panel="normal"]');
      return panel instanceof HTMLElement && Math.abs(panel.getBoundingClientRect().width - Number(width)) < 12;
    }, panelAfterResize?.width ?? 0);
    await page.getByText("PDF asli dirender di frontend melalui akses backend tervalidasi").first().waitFor();
    await page.getByLabel("Simpan bookmark sementara").first().click();

    const mobilePage = await browser.newPage({ viewport: { width: 390, height: 800 } });
    await mobilePage.goto(frontendUrl, { waitUntil: "networkidle" });
    await mobilePage.getByPlaceholder("Tanya UUD 1945...").fill("Pasal 1 ayat (3)");
    await mobilePage.getByLabel("Send").click();
    await mobilePage.getByText("Dukungan sitasi berbasis bukti").waitFor();
    await mobilePage.locator("button", { hasText: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945" }).first().click();
    await mobilePage.locator('[data-evidence-panel="normal"]').waitFor();
    await mobilePage.waitForFunction(() => {
      const handle = window.document.querySelector('[data-evidence-resize-handle="true"]');
      return !handle || window.getComputedStyle(handle).display === "none";
    });
    await mobilePage.close();

    await page.getByRole("button", { name: "Cari Regulasi" }).click();
    await page.getByRole("heading", { name: "Search UUD" }).waitFor();
    await page.getByPlaceholder("Cari dalam UUD 1945...").fill("UUD 1945");
    await page.getByText(/Undang-Undang Dasar Negara Republik Indonesia Tahun 1945/).first().waitFor();
    await page.getByLabel(/Buka viewer/).first().click();
    await page.locator('[data-pdf-document="full"]').waitFor();
    await page.locator('canvas[aria-label="Halaman sumber 1"][data-rendered="true"]').waitFor();
    assert(
      (await page.locator('[data-pdf-document="full"] [data-bbox-highlight]').count()) === 0,
      "Document search viewer should not show default highlights.",
    );

    await page.getByRole("button", { name: "Cari Regulasi" }).click();
    await page.getByRole("heading", { name: "Search UUD" }).waitFor();
    await page.getByPlaceholder("Cari dalam UUD 1945...").fill("Perubahan Ketiga UUD");
    await page.getByText(/Perubahan Ketiga Undang-Undang Dasar/).first().waitFor();
    await page.getByLabel(/Buka viewer/).first().click();
    await page.locator('[data-pdf-document="full"]').waitFor();
    await page.locator('canvas[aria-label="Halaman sumber 1"][data-rendered="true"]').waitFor();
    assert(
      (await page.locator('[data-pdf-document="full"] [data-bbox-highlight]').count()) === 0,
      "Document search viewer should stay unhighlighted.",
    );

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

    await runEvidenceContractSmoke(browser);
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
