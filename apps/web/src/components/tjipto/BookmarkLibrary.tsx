import { Bookmark, FileText, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { deleteLegalBookmark, listLegalBookmarks, type BookmarkPointer } from "../../lib/api";
import type { Citation } from "../../lib/types";

export function BookmarkLibrary({ onOpenDocument }: { onOpenDocument?: (citation: Citation) => void }) {
  const [bookmarks, setBookmarks] = useState<BookmarkPointer[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    listLegalBookmarks().then((body) => setBookmarks(body.bookmarks)).finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex-1 overflow-y-auto tj-scroll">
      <div className="mx-auto max-w-[960px] px-4 pb-16 pt-8 sm:px-6 sm:pt-12">
        <h1 className="text-[28px] font-semibold tracking-tight text-[var(--tj-text-primary)]">Penanda</h1>
        <p className="mt-2 text-[14px] text-[var(--tj-text-secondary)]">Naskah dan sumber resmi yang Anda simpan pada sesi ini.</p>
        <div className="mt-7 overflow-hidden rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)]">
          {loading || bookmarks.length === 0 ? (
            <div className="flex items-center gap-3 p-5 text-[14px] text-[var(--tj-text-secondary)]">
              <Bookmark size={17} /> {loading ? "Memuat penanda…" : "Belum ada penanda tersimpan."}
            </div>
          ) : bookmarks.map((row, index) => (
            <div
              key={row.public_bookmark_id}
              className="flex min-h-14 w-full items-center gap-3 border-b border-[var(--tj-border-subtle)] px-4 py-3 text-left last:border-b-0"
            >
              <button
                type="button"
                className="flex min-w-0 flex-1 items-center gap-3 text-left"
                onClick={() => row.public_target_id && onOpenDocument?.({
                  id: index + 1,
                  publicTargetId: row.public_target_id,
                  documentTitle: row.document?.legal_identity || row.document?.title || row.note || "Naskah tersimpan",
                  regulationType: "Dokumen hukum",
                  viewerMode: "document",
                  pageNumber: 1,
                  excerpt: "",
                  legalStatus: row.document?.legal_status,
                  documentRole: row.document?.document_role,
                  establishmentDate: row.document?.establishment_date ?? undefined,
                  promulgationDate: row.document?.promulgation_date ?? undefined,
                  effectiveDate: row.document?.effective_date ?? undefined,
                  officialUrl: row.document?.official_url,
                  issuer: row.document?.issuer,
                })}
              >
                <FileText size={16} className="shrink-0 text-[var(--tj-text-secondary)]" />
                <span className="truncate text-[14px] font-medium text-[var(--tj-text-primary)]">{row.document?.legal_identity || row.document?.title || row.note || "Naskah tersimpan"}</span>
              </button>
              <button
                type="button"
                aria-label="Hapus penanda"
                title="Hapus penanda"
                disabled={deleting === row.public_bookmark_id}
                className="flex min-h-10 min-w-10 items-center justify-center rounded-lg text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)] disabled:opacity-40"
                onClick={async () => {
                  setDeleting(row.public_bookmark_id);
                  try {
                    if (await deleteLegalBookmark(row.public_bookmark_id)) {
                      setBookmarks((current) => current.filter((bookmark) => bookmark.public_bookmark_id !== row.public_bookmark_id));
                    }
                  } finally {
                    setDeleting(null);
                  }
                }}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
