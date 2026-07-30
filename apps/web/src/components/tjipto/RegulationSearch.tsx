import { Filter, RotateCcw, Search } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import {
  getCatalogFacets,
  mapSearchResultToCitation,
  searchLegal,
  type CatalogFacet,
  type SearchResult,
} from "../../lib/api";
import type { Citation } from "../../lib/types";

export function RegulationSearch({ onOpenDocument }: { onOpenDocument?: (citation: Citation) => void }) {
  const [input, setInput] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [facets, setFacets] = useState<CatalogFacet[]>([]);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [appliedFilters, setAppliedFilters] = useState<Record<string, string>>({});
  const [results, setResults] = useState<SearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "empty" | "unavailable">("idle");

  useEffect(() => {
    getCatalogFacets().then(setFacets).catch(() => setFacets([]));
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setStatus("loading");
    try {
      const response = await searchLegal(input.trim(), filters);
      setSubmittedQuery(input.trim());
      setAppliedFilters(response.applied_filters);
      setResults(response.results);
      setFacets(response.facets);
      setTotal(response.total);
      setStatus(response.results.length ? "ready" : "empty");
    } catch {
      setStatus("unavailable");
    }
  };

  return (
    <div className="flex-1 overflow-y-auto tj-scroll">
      <div className="mx-auto max-w-[960px] px-4 pb-16 pt-8 sm:px-6 sm:pt-12">
        <div className="max-w-[680px]">
          <p className="mb-2 text-[12px] font-semibold uppercase tracking-[0.14em] text-[var(--tj-accent)]">Katalog Hukum</p>
          <h1 className="tracking-tight text-[28px] font-semibold text-[var(--tj-text-primary)]">Cari Peraturan</h1>
          <p className="mt-2 text-[14px] leading-6 text-[var(--tj-text-secondary)]">
            Temukan identitas dan naskah resmi berdasarkan jenis, nomor, tahun, atau judul.
          </p>
        </div>

        <form className="mt-7" onSubmit={submit} aria-label="Cari Peraturan">
          <div className="flex gap-2">
            <label className="relative flex-1">
              <span className="sr-only">Cari berdasarkan jenis, nomor, tahun, atau judul</span>
              <Search size={17} className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--tj-text-muted)]" />
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Cari berdasarkan jenis, nomor, tahun, atau judul"
                className="h-12 w-full rounded-xl border border-[var(--tj-border)] bg-[var(--tj-surface)] pl-11 pr-4 text-[15px] text-[var(--tj-text-primary)] outline-none transition-all placeholder:text-[var(--tj-text-muted)] focus:border-[var(--tj-accent)] focus:shadow-[0_0_0_3px_var(--tj-accent-ring)]"
              />
            </label>
            <button type="submit" className="h-12 min-w-20 rounded-xl bg-[var(--tj-text-primary)] px-5 text-[14px] font-semibold text-[var(--tj-bg)]">
              Cari
            </button>
          </div>

          {facets.length > 0 && (
            <fieldset className="mt-5 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] p-4">
              <legend className="flex items-center gap-2 px-1 text-[12px] font-semibold text-[var(--tj-text-secondary)]">
                <Filter size={13} /> Filter hukum
              </legend>
              <div className="grid gap-3 pt-2 sm:grid-cols-3">
                {facets.map((facet) => (
                  <label key={facet.name} className="text-[12px] font-medium text-[var(--tj-text-secondary)]">
                    {facet.label}
                    <select
                      aria-label={facet.label}
                      value={filters[facet.name] ?? ""}
                      onChange={(event) => setFilters((current) => {
                        const next = { ...current };
                        if (event.target.value) next[facet.name] = event.target.value;
                        else delete next[facet.name];
                        return next;
                      })}
                      className="mt-1.5 h-10 w-full rounded-lg border border-[var(--tj-border)] bg-[var(--tj-surface)] px-3 text-[13px] text-[var(--tj-text-primary)]"
                    >
                      <option value="">Semua</option>
                      {facet.options.map((option) => (
                        <option key={option.value} value={option.value}>{option.label} ({option.count})</option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setFilters({})}
                className="mt-3 inline-flex min-h-10 items-center gap-2 rounded-lg px-2 text-[12px] font-medium text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)]"
              >
                <RotateCcw size={13} /> Atur ulang filter
              </button>
            </fieldset>
          )}
        </form>

        <div className="mt-7 flex items-center justify-between text-[12px] text-[var(--tj-text-muted)]" aria-live="polite">
          <span>{submittedQuery ? `${total} naskah untuk "${submittedQuery}"` : "Masukkan identitas peraturan untuk memulai."}</span>
          {Object.keys(appliedFilters).length > 0 && <span>{Object.keys(appliedFilters).length} filter diterapkan</span>}
        </div>

        {status === "loading" && <EmptyState text="Mencari naskah resmi..." />}
        {status === "empty" && <EmptyState text="Tidak ada naskah yang sesuai dengan pencarian dan filter hukum tersebut." />}
        {status === "unavailable" && <EmptyState text="Pencarian peraturan belum tersedia." />}
        {status === "ready" && (
          <ul className="mt-3 space-y-2">
            {results.map((row, index) => {
              const citation = mapSearchResultToCitation(row, index);
              return (
                <li key={row.viewer_target?.public_target_id ?? `${row.document_type}-${row.number}-${row.year}`} className="rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] p-4">
                  <button
                    type="button"
                    className="flex w-full items-start gap-3 text-left"
                    onClick={() => citation && onOpenDocument?.(citation)}
                    aria-label={`Buka naskah ${row.short_title}`}
                  >
                    <span className="mt-0.5 inline-flex h-6 shrink-0 items-center rounded-md bg-[var(--tj-accent-soft)] px-2 text-[11px] font-semibold text-[var(--tj-accent)]">
                      {row.document_type}
                    </span>
                    <span className="min-w-0">
                      <span className="block text-[15px] font-semibold text-[var(--tj-text-primary)]">{row.official_title}</span>
                      <span className="mt-1.5 block text-[13px] leading-5 text-[var(--tj-text-secondary)]">
                        {row.document_type} Nomor {row.number} Tahun {row.year}
                      </span>
                      <span className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-[var(--tj-text-muted)]">
                        <span>{row.legal_status}</span>
                        <span>{row.document_role}</span>
                        {row.establishment_date && <span>Ditetapkan {row.establishment_date}</span>}
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

function EmptyState({ text }: { text: string }) {
  return <p className="mt-3 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] p-5 text-[14px] text-[var(--tj-text-secondary)]">{text}</p>;
}
