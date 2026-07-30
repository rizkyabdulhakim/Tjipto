import { legalIdentity, numberAndYear } from "../../lib/legalPresentation";

export function LegalStatusCard({
  status,
  role,
  title,
  documentType,
  number,
  year,
  issuer,
  establishmentDate,
  promulgationDate,
  effectiveDate,
  officialUrl,
  publication,
}: {
  status: string;
  role?: string;
  title?: string;
  documentType?: string;
  number?: string;
  year?: string;
  issuer?: string;
  establishmentDate?: string;
  promulgationDate?: string;
  effectiveDate?: string;
  officialUrl?: string;
  publication?: string;
}) {
  const identity = legalIdentity({ official_title: title, document_type: documentType, number, year });
  const rows = [
    ["Identitas Resmi", identity],
    ["Kedudukan Naskah", role],
    ["Jenis", documentType],
    ["Nomor dan Tahun", numberAndYear({ document_type: documentType, number, year })],
    ["Penerbit", issuer],
    ["Tanggal Penetapan", establishmentDate],
    ["Tanggal Pengundangan", promulgationDate],
    ["Tanggal Berlaku", effectiveDate],
    ["Identitas Pengundangan", publication],
  ].filter((row): row is [string, string] => Boolean(row[1]));

  return (
    <section className="tj-legal-status-card overflow-hidden rounded-2xl border border-[var(--tj-glass-border)]" data-legal-status-card>
      <div className="flex items-center justify-between gap-4 border-b border-[var(--tj-border-subtle)] px-4 py-3.5">
        <span className="text-[12px] font-semibold uppercase tracking-[0.08em] text-[var(--tj-text-muted)]">Status Keberlakuan</span>
        <span className="rounded-full bg-[var(--tj-accent-soft)] px-2.5 py-1 text-[12px] font-semibold text-[var(--tj-accent)]">{status}</span>
      </div>
      <dl className="divide-y divide-[var(--tj-border-subtle)]">
        {rows.map(([label, value]) => (
          <div key={label} className="flex items-center justify-between gap-4 px-4 py-3">
            <dt className="text-[13px] font-medium text-[var(--tj-text-muted)]">{label}</dt>
            <dd className="min-w-0 truncate text-right text-[13px] font-semibold text-[var(--tj-text-primary)]">{value}</dd>
          </div>
        ))}
      </dl>
      {officialUrl && (
        <a href={officialUrl} target="_blank" rel="noreferrer" className="flex min-h-11 items-center justify-center border-t border-[var(--tj-border-subtle)] px-4 text-[13px] font-semibold text-[var(--tj-accent)]">
          Sumber Resmi
        </a>
      )}
    </section>
  );
}
