export interface LegalDocumentPresentation {
  legal_identity?: string;
  official_title?: string;
  short_title?: string;
  document_type?: string;
  number?: string;
  year?: string;
  legal_status?: string;
  document_role?: string;
}

export function legalIdentity(document: LegalDocumentPresentation): string {
  if (document.legal_identity?.trim()) return document.legal_identity.trim();
  if (document.official_title?.trim()) return document.official_title.trim();
  if (document.document_type && document.number && document.year) {
    return `${document.document_type} Nomor ${document.number} Tahun ${document.year}`;
  }
  return document.short_title?.trim() || document.document_type?.trim() || "Identitas belum diverifikasi";
}

export function legalStatus(value?: string): string {
  const normalized = value?.trim();
  if (!normalized) return "Belum Diverifikasi";
  return ([
    "Berlaku",
    "Berlaku Sebagian",
    "Tidak Berlaku",
    "Belum Diverifikasi",
    "Konflik Sumber",
  ].includes(normalized) ? normalized : "Belum Diverifikasi");
}

export function documentRole(value?: string): string | undefined {
  const normalized = value?.trim();
  if (!normalized) return undefined;
  return ([
    "Naskah Konsolidasi",
    "Naskah Asli",
    "Amandemen Pertama",
    "Amandemen Kedua",
    "Amandemen Ketiga",
    "Amandemen Keempat",
    "Naskah Pokok",
    "Naskah Perubahan",
    "Amandemen",
  ].includes(normalized) ? normalized : undefined);
}

export function numberAndYear(document: LegalDocumentPresentation): string | undefined {
  return document.number && document.year ? `${document.number} Tahun ${document.year}` : undefined;
}
