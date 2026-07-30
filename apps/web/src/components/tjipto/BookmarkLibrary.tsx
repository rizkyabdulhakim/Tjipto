import { Bookmark, FileText } from "lucide-react";
import { useEffect, useState } from "react";
import { listLegalBookmarks, type BookmarkPointer } from "../../lib/api";
import type { Citation } from "../../lib/types";

export function BookmarkLibrary({ onOpenDocument }: { onOpenDocument?: (citation: Citation) => void }) {
  const [bookmarks, setBookmarks] = useState<BookmarkPointer[]>([]);
  const [loading, setLoading] = useState(true);

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
            <button
              key={row.public_bookmark_id}
              type="button"
              className="flex min-h-14 w-full items-center gap-3 border-b border-[var(--tj-border-subtle)] px-4 py-3 text-left last:border-b-0"
              onClick={() => row.public_target_id && onOpenDocument?.({
                id: index + 1,
                publicTargetId: row.public_target_id,
                documentTitle: row.note || "Naskah tersimpan",
                regulationType: "Dokumen hukum",
                pageNumber: 1,
                excerpt: "",
              })}
            >
              <FileText size={16} className="text-[var(--tj-text-secondary)]" />
              <span className="text-[14px] font-medium text-[var(--tj-text-primary)]">{row.note || "Naskah tersimpan"}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
