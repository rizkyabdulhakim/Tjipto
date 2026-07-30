import { Bookmark, Check, Maximize2, Minimize2, Minus, Plus } from "lucide-react";

export function PanelToolbar({
  zoom,
  expanded,
  saved,
  canBookmark,
  onZoom,
  onToggleExpanded,
  onBookmark,
}: {
  zoom: number;
  expanded: boolean;
  saved: boolean;
  canBookmark: boolean;
  onZoom: (delta: number) => void;
  onToggleExpanded: () => void;
  onBookmark: () => void;
}) {
  return (
    <div className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)]/40 px-4 sm:px-6" data-panel-toolbar>
      <div className="flex items-center">
        <Control label="Perkecil tampilan" onClick={() => onZoom(-25)} disabled={zoom <= 50}><Minus size={14} /></Control>
        <span className="flex min-w-11 select-none items-center justify-center px-2 text-[12px] font-semibold tabular-nums text-[var(--tj-text-secondary)]">{zoom}%</span>
        <Control label="Perbesar tampilan" onClick={() => onZoom(25)} disabled={zoom >= 200}><Plus size={14} /></Control>
      </div>
      <div className="flex-1" />
      <Control label={expanded ? "Kembali ke panel" : "Tampilkan naskah penuh"} onClick={onToggleExpanded}>
        {expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
      </Control>
      {canBookmark && (
        <Control label="Simpan Penanda" onClick={onBookmark}>
          {saved ? <Check size={15} /> : <Bookmark size={15} />}
        </Control>
      )}
    </div>
  );
}

function Control({ label, children, onClick, disabled = false }: {
  label: string;
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="flex h-10 min-w-10 items-center justify-center rounded-lg text-[var(--tj-text-secondary)] transition-colors hover:bg-[var(--tj-surface-hover)] hover:text-[var(--tj-text-primary)] disabled:opacity-30"
    >
      {children}
    </button>
  );
}
