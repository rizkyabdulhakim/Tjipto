import { Search, FileText, Clock, Filter } from "lucide-react";
import { useState } from "react";

const filters = ["Sumber", "Status", "Periode"];

export function SearchRoute() {
  const [q, setQ] = useState("negara hukum");
  return (
    <div className="flex-1 overflow-y-auto tj-scroll">
      <div className="max-w-[960px] mx-auto px-4 sm:px-6 pt-6 sm:pt-10 pb-12 sm:pb-16">
        <h1 className="tracking-tight" style={{ fontSize: 26, fontWeight: 600, color: "var(--tj-text-primary)" }}>
          Search UUD
        </h1>
        <p className="mt-1.5 mb-6" style={{ fontSize: 14, color: "var(--tj-text-secondary)" }}>
          Runtime terverifikasi saat ini hanya memuat UUD. Gunakan chat untuk jawaban final dari /uud/ask.
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
            Search artifact belum disajikan oleh backend untuk <span style={{ color: "var(--tj-text-primary)", fontWeight: 500 }}>"{q}"</span>
          </span>
          <span style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>Scope: UUD-only</span>
        </div>

        <div className="mt-3 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] p-5">
          <p style={{ fontSize: 14, color: "var(--tj-text-secondary)", lineHeight: "22px" }}>
            Belum ada kontrak backend untuk daftar hasil Search. Gunakan chat UUD untuk jawaban berbasis evidence.
          </p>
        </div>
      </div>
    </div>
  );
}

export function LibraryRoute() {
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
          Korpus runtime terverifikasi saat ini: UUD-only.
        </p>

        <div className="mt-7 flex items-center gap-2">
          <Clock size={13} className="text-[var(--tj-text-secondary)]" />
          <span
            style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.04em", color: "var(--tj-text-secondary)" }}
          >
            LIBRARY STATUS
          </span>
        </div>

        <div className="mt-3 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] p-5">
          <div className="flex items-start gap-3">
            <div className="w-9 h-11 rounded-md bg-[var(--tj-surface-subtle)] border border-[var(--tj-border-subtle)] flex items-center justify-center shrink-0">
              <FileText size={15} className="text-[var(--tj-text-secondary)]" />
            </div>
            <p style={{ fontSize: 14, color: "var(--tj-text-secondary)", lineHeight: "22px" }}>
              Library belum memiliki kontrak backend. Tidak ada dokumen atau bukti yang ditampilkan sebagai hasil terverifikasi di layar ini.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
