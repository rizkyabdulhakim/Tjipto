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
  const relationGroups = groupByLabel(verifiedRelations);
  const effectGroups = groupByLabel(verifiedEffects);
  if (!verifiedRelations.length && !verifiedEffects.length && !annotations.length && !officialTitleConflict) return null;

  return (
    <div className="divide-y divide-[var(--tj-border-subtle)]" data-document-legal-details>
      {relationGroups.map(([title, items]) => <DetailSection key={`relation:${title}`} title={title} items={items} />)}
      {effectGroups.map(([title, items]) => <DetailSection key={`effect:${title}`} title={title} items={items} />)}
      {officialTitleConflict && <OfficialValueConflict conflict={officialTitleConflict} />}
      {annotations.length > 0 && (
        <section>
          <div className="grid grid-cols-[minmax(7.5rem,0.9fr)_minmax(0,1.6fr)] items-start gap-x-4 px-4 py-3">
            <h3 className="text-[13px] font-medium text-[var(--tj-text-muted)]">Catatan Sumber</h3>
            <div className="min-w-0 text-left">
              <p className="text-[13px] leading-5 text-[var(--tj-text-primary)]">
                {Array.from(new Set(annotations.map((item) => `${item.label}: ${item.text}`))).join("; ")}
              </p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function OfficialValueConflict({ conflict }: { conflict: OfficialValueConflictPayload }) {
  return (
    <section>
      <div className="px-4 py-3">
        <h3 className="text-[13px] font-medium text-[var(--tj-text-muted)]">{conflict.kind}</h3>
        <p className="mt-1 text-[12px] text-[var(--tj-text-secondary)]">{conflict.state}</p>
      </div>
      <ul className="divide-y divide-[var(--tj-border-subtle)]">
        {conflict.values.map((candidate) => (
          <li key={`${candidate.source_reference}:${candidate.value}`} className="px-4 py-3">
            <p className="text-[13px] font-semibold text-[var(--tj-text-primary)]">{candidate.value}</p>
            {candidate.source_authority && <p className="mt-1 text-[12px] text-[var(--tj-text-secondary)]">{candidate.source_authority}</p>}
          </li>
        ))}
      </ul>
      {conflict.reviewer_decision && <p className="px-4 py-3 text-[12px] leading-5 text-[var(--tj-text-secondary)]">{conflict.reviewer_decision}</p>}
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
  const values = Array.from(new Set(items.map((item) => item.target)));
  return (
    <section>
      <div className="grid grid-cols-[minmax(7.5rem,0.9fr)_minmax(0,1.6fr)] items-start gap-x-4 px-4 py-3">
        <h3 className="text-[13px] font-medium text-[var(--tj-text-muted)]">{title}</h3>
        <div className="min-w-0 text-left">
          <p className="text-[13px] leading-5 text-[var(--tj-text-primary)]">{values.join(", ")}</p>
        </div>
      </div>
    </section>
  );
}

function groupByLabel<T extends { label: string }>(items: T[]): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const group = groups.get(item.label);
    if (group) group.push(item);
    else groups.set(item.label, [item]);
  }
  return Array.from(groups.entries());
}

function isVerified(value: string) {
  return value === "verified" || value === "Terverifikasi";
}
