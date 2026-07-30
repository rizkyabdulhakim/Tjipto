import type { LegalRelationPayload, OfficialValueConflictPayload, ProvisionEffectPayload, SourceAnnotationPayload } from "../../lib/api";

export function DocumentLegalDetails({
  relations = [],
  provisionEffects = [],
  annotations = [],
  officialTitleConflict,
}: {
  relations?: LegalRelationPayload[];
  provisionEffects?: ProvisionEffectPayload[];
  annotations?: SourceAnnotationPayload[];
  officialTitleConflict?: OfficialValueConflictPayload;
}) {
  const verifiedRelations = relations.filter((item) => isVerified(item.verification_state));
  const verifiedEffects = provisionEffects.filter((item) => isVerified(item.verification_state));
  if (!verifiedRelations.length && !verifiedEffects.length && !annotations.length && !officialTitleConflict) return null;

  return (
    <div className="mt-4 space-y-4" data-document-legal-details>
      <DetailSection title="Hubungan Peraturan" items={verifiedRelations} />
      <DetailSection title="Ketentuan yang Diubah" items={verifiedEffects} />
      {officialTitleConflict && <OfficialValueConflict conflict={officialTitleConflict} />}
      {annotations.length > 0 && (
        <section className="overflow-hidden rounded-2xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)]">
          <h3 className="border-b border-[var(--tj-border-subtle)] px-4 py-3 text-[12px] font-semibold uppercase tracking-[0.08em] text-[var(--tj-text-muted)]">Catatan Sumber</h3>
          <ul className="divide-y divide-[var(--tj-border-subtle)]">
            {annotations.map((item) => (
              <li key={`${item.label}:${item.text}`} className="px-4 py-3">
                <p className="text-[13px] font-semibold text-[var(--tj-text-primary)]">{item.label}</p>
                <p className="mt-1 text-[13px] leading-5 text-[var(--tj-text-secondary)]">{item.text}</p>
                <OfficialSource href={item.source_reference} />
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function OfficialValueConflict({ conflict }: { conflict: OfficialValueConflictPayload }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)]">
      <div className="border-b border-[var(--tj-border-subtle)] px-4 py-3">
        <h3 className="text-[12px] font-semibold uppercase tracking-[0.08em] text-[var(--tj-text-muted)]">{conflict.kind}</h3>
        <p className="mt-1 text-[12px] text-[var(--tj-text-secondary)]">{conflict.state}</p>
      </div>
      <ul className="divide-y divide-[var(--tj-border-subtle)]">
        {conflict.values.map((candidate) => (
          <li key={`${candidate.source_reference}:${candidate.value}`} className="px-4 py-3">
            <p className="text-[13px] font-semibold text-[var(--tj-text-primary)]">{candidate.value}</p>
            {candidate.source_authority && <p className="mt-1 text-[12px] text-[var(--tj-text-secondary)]">{candidate.source_authority}</p>}
            <OfficialSource href={candidate.source_reference} />
          </li>
        ))}
      </ul>
      {conflict.reviewer_decision && <p className="border-t border-[var(--tj-border-subtle)] px-4 py-3 text-[12px] leading-5 text-[var(--tj-text-secondary)]">{conflict.reviewer_decision}</p>}
    </section>
  );
}

function DetailSection({
  title,
  items,
}: {
  title: string;
  items: (LegalRelationPayload | ProvisionEffectPayload)[];
}) {
  if (!items.length) return null;
  return (
    <section className="overflow-hidden rounded-2xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)]">
      <h3 className="border-b border-[var(--tj-border-subtle)] px-4 py-3 text-[12px] font-semibold uppercase tracking-[0.08em] text-[var(--tj-text-muted)]">{title}</h3>
      <ul className="divide-y divide-[var(--tj-border-subtle)]">
        {items.map((item) => (
          <li key={`${item.label}:${item.target}`} className="px-4 py-3">
            <p className="text-[13px] font-semibold text-[var(--tj-text-primary)]">{item.label}</p>
            {"source" in item && (
              <p className="mt-1 text-[12px] text-[var(--tj-text-muted)]">
                {item.source} → {item.target}
              </p>
            )}
            {!('source' in item) && (
              <p className="mt-1 text-[13px] leading-5 text-[var(--tj-text-secondary)]">{item.target}</p>
            )}
            <OfficialSource href={item.source_reference} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function OfficialSource({ href }: { href?: string }) {
  if (!href) return null;
  return <a href={href} target="_blank" rel="noreferrer" className="mt-2 inline-flex text-[12px] font-semibold text-[var(--tj-accent)]">Sumber Resmi</a>;
}

function isVerified(value: string) {
  return value === "verified" || value === "Terverifikasi";
}
