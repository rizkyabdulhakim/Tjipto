import { Search, FileText, Clock, Filter } from "lucide-react";
import { useEffect, useState } from "react";
import {
  listLegalBookmarks,
  mapSearchResultToCitation,
  searchLegal,
  type BookmarkPointer,
  type SearchResult,
} from "../../lib/api";
import type { Citation } from "../../lib/types";

const filters = ["Sumber", "Status", "Periode"];

export function SearchRoute({ onOpenCitation }: { onOpenCitation?: (citation: Citation) => void }) {
  const [q, setQ] = useState("UUD 1945");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let ignore = false;
    setStatus("loading");
    searchLegal(q)
      .then((rows) => {
        if (!ignore) {
          setResults(rows);
          setStatus(rows.length ? "ready" : "empty");
        }
      })
      .catch(() => {
        if (!ignore) setStatus("unavailable");
      });
    return () => {
      ignore = true;
    };
  }, [q]);

  return (
    <div className="flex-1 overflow-y-auto tj-scroll">
      <div className="max-w-[960px] mx-auto px-4 sm:px-6 pt-6 sm:pt-10 pb-12 sm:pb-16">
        <h1 className="tracking-tight" style={{ fontSize: 26, fontWeight: 600, color: "var(--tj-text-primary)" }}>
          Search UUD
        </h1>
        <p className="mt-1.5 mb-6" style={{ fontSize: 14, color: "var(--tj-text-secondary)" }}>
          Runtime terverifikasi saat ini hanya memuat UUD. Gunakan chat untuk jawaban final dari /legal/uud/ask.
        </p>

        <div className="relative">
          <Search size={17} className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--tj-text-muted)]" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Cari dalam UUD 1945..."
            className="w-full h-12 rounded-xl border border-[var(--tj-border)] bg-[var(--tj-surface)] pl-11 pr-4 outline-none focus:border-[var(--tj-accent)] focus:shadow-[0_0_0_3px_var(--tj-accent-ring)] transition-all text-[var(--tj-text-primary)] placeholder:text-[var(--tj-text-muted)]"
            style={{ fontSize: 15 }}
          />
        </div>

        <div className="flex items-center gap-2 mt-4 overflow-x-auto tj-scroll -mx-4 px-4 sm:mx-0 sm:px-0 sm:flex-wrap pb-1">
          <span className="flex items-center gap-1.5 mr-1" style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>
            <Filter size={12} /> FILTER
          </span>
          {filters.map((f, i) => (
            <button
              key={f}
              className={`h-8 px-3 rounded-full border transition-colors shrink-0 ${i === 0 ? "bg-[var(--tj-text-primary)] text-[var(--tj-bg)] border-[var(--tj-text-primary)]" : "border-[var(--tj-border)] text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)]"}`}
              style={{ fontSize: 12.5, fontWeight: 500 }}
            >
              {f} {i === 0 && <span className="opacity-70">UUD</span>}
            </button>
          ))}
        </div>

        <div className="mt-6 flex items-center justify-between">
          <span style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>
            Backend Search untuk <span style={{ color: "var(--tj-text-primary)", fontWeight: 500 }}>"{q}"</span>
          </span>
          <span style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>Scope: UUD-only</span>
        </div>

        {status !== "ready" ? (
          <div className="mt-3 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] p-5">
            <p style={{ fontSize: 14, color: "var(--tj-text-secondary)", lineHeight: "22px" }}>
              {status === "loading" ? "Memeriksa evidence backend..." : "Tidak ada evidence publik untuk query ini."}
            </p>
          </div>
        ) : (
          <ul className="mt-3 space-y-2">
            {results.map((row, index) => {
              const citation = mapSearchResultToCitation(row, index);
              const page = row.page_numbers?.[0] ?? row.viewer_ref?.page_numbers?.[0];
              return (
                <li key={row.evidence_id ?? row.source_document_id} className="rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] p-4">
                  <button
                    type="button"
                    aria-label={`Buka viewer ${row.title ?? row.evidence_id}`}
                    className="w-full flex items-start gap-3 text-left disabled:cursor-not-allowed disabled:opacity-70"
                    onClick={() => citation && onOpenCitation?.(citation)}
                    disabled={!citation}
                  >
                    <span className="inline-flex items-center px-2 h-[22px] rounded-md shrink-0 mt-0.5" style={{ fontSize: 11, fontWeight: 600, background: "var(--tj-accent-soft)", color: "var(--tj-accent)" }}>
                      {row.corpus_id.toUpperCase()}
                    </span>
                    <span className="min-w-0">
                      <span className="block" style={{ fontSize: 15, fontWeight: 600, color: "var(--tj-text-primary)" }}>{row.title}</span>
                      <span className="block mt-1.5 line-clamp-3" style={{ fontSize: 13.5, lineHeight: "20px", color: "var(--tj-text-secondary)" }}>{row.snippet}</span>
                      <span className="block mt-3" style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>
                        {page ? `Halaman ${page}` : "Halaman tidak tersedia"} · {sourceStatusLabel(row.source_role, row.temporal_context)}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

export function LibraryRoute() {
  const [bookmarks, setBookmarks] = useState<BookmarkPointer[]>([]);
  const [persistence, setPersistence] = useState("temporary_process_memory");
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    listLegalBookmarks()
      .then((body) => {
        setBookmarks(body.bookmarks);
        setPersistence(body.persistence_label ?? body.persistence ?? "temporary_process_memory");
        setStatus(body.bookmarks.length ? "ready" : "empty");
      })
      .catch(() => setStatus("unavailable"));
  }, []);

  return (
    <div className="flex-1 overflow-y-auto tj-scroll">
      <div className="max-w-[960px] mx-auto px-4 sm:px-6 pt-6 sm:pt-10 pb-12 sm:pb-16">
        <h1
          className="tracking-tight"
          style={{ fontSize: 26, fontWeight: 600, color: "var(--tj-text-primary)" }}
        >
          Library
        </h1>
        <p
          className="mt-1.5"
          style={{ fontSize: 14, color: "var(--tj-text-secondary)" }}
        >
          Korpus runtime terverifikasi saat ini: UUD-only. Bookmark bersifat sementara di memori proses.
        </p>

        <div className="mt-7 flex items-center gap-2">
          <Clock size={13} className="text-[var(--tj-text-secondary)]" />
          <span
            style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.04em", color: "var(--tj-text-secondary)" }}
          >
            LIBRARY STATUS
          </span>
          <span style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>
            {persistence}
          </span>
        </div>

        <div className="mt-3 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] overflow-hidden">
          {status !== "ready" ? (
            <div className="flex items-start gap-3 p-5">
              <div className="w-9 h-11 rounded-md bg-[var(--tj-surface-subtle)] border border-[var(--tj-border-subtle)] flex items-center justify-center shrink-0">
                <FileText size={15} className="text-[var(--tj-text-secondary)]" />
              </div>
              <p style={{ fontSize: 14, color: "var(--tj-text-secondary)", lineHeight: "22px" }}>
                {status === "loading" ? "Memeriksa bookmark backend..." : "Belum ada bookmark evidence sementara tersimpan."}
              </p>
            </div>
          ) : (
            bookmarks.map((row) => (
              <div key={row.bookmark_id} className="flex items-center gap-3 px-4 py-3.5 border-b border-[var(--tj-border-subtle)] last:border-b-0">
                <div className="w-9 h-11 rounded-md bg-[var(--tj-surface-subtle)] border border-[var(--tj-border-subtle)] flex items-center justify-center shrink-0">
                  <FileText size={15} className="text-[var(--tj-text-secondary)]" />
                </div>
                <div className="min-w-0">
                  <div className="truncate" style={{ fontSize: 14, fontWeight: 600, color: "var(--tj-text-primary)" }}>{row.evidence_id}</div>
                  <div style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>{row.status} · {row.legal_unit_id}</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function sourceStatusLabel(sourceRole?: string, temporalContext?: string) {
  const role = sourceRole ?? temporalContext;
  if (role === "current_consolidated") return "Berlaku";
  if (role?.startsWith("amendment_")) return "Historis";
  if (role === "original_historical") return "Historis";
  return "Status sumber tidak tersedia";
}
