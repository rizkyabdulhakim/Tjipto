import type { ReactNode } from "react";
import { legalIdentity, numberAndYear } from "../../lib/legalPresentation";

export function LegalStatusCard({
  status,
  role,
  title,
  documentType,
  number,
  year,
  issuer,
  establishmentPlace,
  signatories,
  establishmentDate,
  promulgationDate,
  effectiveDate,
  officialUrl,
  publication,
  children,
}: {
  status: string;
  role?: string;
  title?: string;
  documentType?: string;
  number?: string;
  year?: string;
  issuer?: string;
  establishmentPlace?: string | null;
  signatories?: string | null;
  establishmentDate?: string;
  promulgationDate?: string;
  effectiveDate?: string;
  officialUrl?: string;
  publication?: string;
  children?: ReactNode;
}) {
  const identity = legalIdentity({ official_title: title, document_type: documentType, number, year });
  const rows = [
    ["Identitas Resmi", identity],
    ["Kedudukan Naskah", role],
    ["Jenis", documentType],
    ["Nomor dan Tahun", numberAndYear({ document_type: documentType, number, year })],
    ["Lembaga/Penerbit", issuer],
    ["Tempat Penetapan", establishmentPlace],
    ["Tanggal Penetapan", establishmentDate],
    ["Tanggal Pengundangan", promulgationDate],
    ["Tanggal Berlaku", effectiveDate],
    ["Penandatangan", signatories],
    ["Identitas Pengundangan", publication],
  ].filter((row): row is [string, string] => Boolean(row[1]));

  return (
    <section className="tj-legal-status-card min-h-full overflow-hidden rounded-2xl" data-legal-status-card>
      <div className="grid grid-cols-[minmax(7.5rem,0.9fr)_minmax(0,1.6fr)] items-start gap-x-4 border-b border-[var(--tj-border-subtle)] px-4 py-3.5">
        <span className="text-[13px] font-medium text-[var(--tj-text-muted)]">Status Keberlakuan</span>
        <span className="min-w-0 text-left text-[13px] font-semibold text-[var(--tj-text-primary)]">{status}</span>
      </div>
      <dl className="divide-y divide-[var(--tj-border-subtle)]">
        {rows.map(([label, value]) => (
          <div key={label} className="grid grid-cols-[minmax(7.5rem,0.9fr)_minmax(0,1.6fr)] items-start gap-x-4 px-4 py-3">
            <dt className="text-[13px] font-medium text-[var(--tj-text-muted)]">{label}</dt>
            <dd className="min-w-0 text-left text-[13px] font-semibold leading-5 text-[var(--tj-text-primary)]">{value}</dd>
          </div>
        ))}
      </dl>
      {children}
      {officialUrl && (
        <div className="grid grid-cols-[minmax(7.5rem,0.9fr)_minmax(0,1.6fr)] items-start gap-x-4 border-t border-[var(--tj-border-subtle)] px-4 py-3">
          <span className="text-[13px] font-medium text-[var(--tj-text-muted)]">Sumber</span>
          <a href={officialUrl} target="_blank" rel="noreferrer" className="min-w-0 break-all text-left text-[12px] leading-5 text-[var(--tj-accent)]">
            {officialUrl}
          </a>
        </div>
      )}
    </section>
  );
}
