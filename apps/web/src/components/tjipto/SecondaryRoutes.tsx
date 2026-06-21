import { Search, FileText, Clock, Filter, X } from "lucide-react";
import { useState } from "react";

const filters = ["Sumber", "Status", "Periode"];

const results = [
  {
    id: "uud_1945_current",
    type: "UUD",
    title: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
    year: 1945,
    status: "Terverifikasi",
    institution: "Runtime UUD",
    snippet: "Korpus UUD terverifikasi. Jawaban final tetap berasal dari /uud/ask.",
  },
  {
    id: "uud_pembukaan",
    type: "UUD",
    title: "Pembukaan UUD 1945",
    year: 1945,
    status: "Terverifikasi",
    institution: "Runtime UUD",
    snippet: "Rujukan pembukaan dan dasar negara dari korpus UUD.",
  },
];

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
              {i === 0 && <X size={11} className="inline ml-1" />}
            </button>
          ))}
        </div>

        <div className="mt-6 flex items-center justify-between">
          <span style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>
            {results.length} rujukan UUD untuk <span style={{ color: "var(--tj-text-primary)", fontWeight: 500 }}>"{q}"</span>
          </span>
          <span style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>Scope: UUD-only</span>
        </div>

        <ul className="mt-3 space-y-2">
          {results.map((r) => (
            <li key={r.id} className="group rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] hover:border-[var(--tj-border)] transition-all p-4">
              <div className="flex items-start gap-3">
                <span className="inline-flex items-center px-2 h-[22px] rounded-md shrink-0 mt-0.5" style={{ fontSize: 11, fontWeight: 600, background: "var(--tj-accent-soft)", color: "var(--tj-accent)" }}>
                  {r.type}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span style={{ fontSize: 15, fontWeight: 600, color: "var(--tj-text-primary)" }}>{r.title}</span>
                    <span className="inline-flex items-center px-1.5 h-[18px] rounded text-[10px]" style={{ fontWeight: 500, background: "rgba(22,163,74,0.1)", color: "#16A34A" }}>
                      {r.status}
                    </span>
                  </div>
                  <p className="mt-1.5" style={{ fontSize: 13.5, lineHeight: "20px", color: "var(--tj-text-secondary)" }}>{r.snippet}</p>
                  <div className="flex items-center gap-3 mt-3" style={{ fontSize: 12 }}>
                    <button className="h-7 px-3 rounded-md border border-[var(--tj-border)] text-[var(--tj-text-primary)] hover:bg-[var(--tj-surface-hover)] transition-colors" style={{ fontWeight: 500 }}>
                      Detail UUD
                    </button>
                    <span className="ml-auto" style={{ color: "var(--tj-text-muted)" }}>{r.institution} · {r.year}</span>
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export function LibraryRoute() {
  const docs = [
    { title: "Undang-Undang Dasar 1945", date: "Dibuka tadi", type: "UUD" },
    { title: "Pembukaan UUD 1945", date: "Hari ini", type: "UUD" },
    { title: "Pasal 1 ayat (3)", date: "Hari ini", type: "UUD" },
  ];
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
            RECENTLY OPENED
          </span>
        </div>

        <ul className="mt-3 rounded-xl border border-[var(--tj-border-subtle)] overflow-hidden">
          {docs.map((d, i) => (
            <li
              key={d.title}
              className={`flex items-center gap-3 px-4 py-3.5 hover:bg-[var(--tj-surface-hover)] transition-colors cursor-pointer ${
                i !== docs.length - 1 ? "border-b border-[var(--tj-border-subtle)]" : ""
              }`}
            >
              <div
                className="w-9 h-11 rounded-md bg-[var(--tj-surface-subtle)] border border-[var(--tj-border-subtle)] flex items-center justify-center"
              >
                <FileText size={15} className="text-[var(--tj-text-secondary)]" />
              </div>
              <div className="flex-1 min-w-0">
                <div
                  className="truncate"
                  style={{ fontSize: 14, fontWeight: 500, color: "var(--tj-text-primary)" }}
                >
                  {d.title}
                </div>
                <div style={{ fontSize: 12, color: "var(--tj-text-muted)" }}>
                  {d.type} · {d.date}
                </div>
              </div>
              <button
                className="px-2.5 h-7 rounded-md border border-[var(--tj-border)] text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface)] transition-colors"
                style={{ fontSize: 12, fontWeight: 500 }}
              >
                Open
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
