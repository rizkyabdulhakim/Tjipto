import { lazy, Suspense, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Sidebar } from "./components/tjipto/Sidebar";
import { EmptyState } from "./components/tjipto/EmptyState";
import { ChatView } from "./components/tjipto/ChatView";
import { SearchRoute, LibraryRoute } from "./components/tjipto/SecondaryRoutes";
import type { Citation, ChatMessage as TMessage } from "./lib/types";
import { conversation } from "./components/tjipto/data";
import {
  answerTextOrFallback,
  askLegal,
  mapAskResponseToCitations,
  mapAskResponseToDocumentSource,
  mapAskResponseToDocumentSources,
  mapAskResponseToSupportGroups,
} from "./lib/api";
import { Menu, SquarePen } from "lucide-react";

type Route = "chat" | "search" | "library";

const EvidencePanel = lazy(() =>
  import("./components/tjipto/EvidencePanel").then((module) => ({ default: module.EvidencePanel })),
);


export default function App() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [route, setRoute] = useState<Route>("chat");
  const [messages, setMessages] = useState<TMessage[]>(conversation);
  const [hasChat, setHasChat] = useState(conversation.length > 0);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [clarification, setClarification] = useState<{ id: string } | null>(null);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [panelCompactNav, setPanelCompactNav] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    const update = () => setPanelCompactNav(Boolean(activeCitation) && window.innerWidth >= 768 && window.innerWidth < 1280);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [activeCitation]);

  // Close on escape
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActiveCitation(null);
        setMobileNavOpen(false);
      }
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);

  // Lock body scroll when mobile drawer is open
  useEffect(() => {
    if (mobileNavOpen) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [mobileNavOpen]);

  const newChat = () => {
    setMessages([]);
    setHasChat(false);
    setRoute("chat");
    setActiveCitation(null);
    setClarification(null);
    setMobileNavOpen(false);
  };

  const submit = async (value: string) => {
    if (!hasChat) setHasChat(true);
    setRoute("chat");
    setActiveCitation(null);
    const userMsg: TMessage = {
      id: "u_" + Date.now(),
      role: "user",
      content: value,
    };
    const asstId = "a_" + Date.now();
    setMessages((prev) => [
      ...prev,
      userMsg,
      {
        id: asstId,
        role: "assistant",
        content: "Menelusuri sumber hukum terverifikasi…",
        status: "streaming",
      },
    ]);
    setIsStreaming(true);

    try {
      const response = await askLegal(value, clarification ?? undefined);
      setClarification(response.clarification ? { id: response.clarification.id } : null);
      const citations = mapAskResponseToCitations(response);
      const documentSource = mapAskResponseToDocumentSource(response);
      const documentCollection = mapAskResponseToDocumentSources(response);
      const supportGroups = mapAskResponseToSupportGroups(response);
      const messageCitations = [...citations, ...documentCollection, ...(documentSource ? [documentSource] : [])];
      const content = response.kind === "document"
        ? response.document?.label ?? "Dokumen sumber tersedia."
        : response.kind === "documents"
          ? `${response.documents?.length ?? 0} dokumen sumber terverifikasi tersedia.`
        : answerTextOrFallback(response);
      setActiveCitation(documentSource);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? {
                ...m,
                content,
                status: "complete",
                citations: messageCitations.length ? messageCitations : undefined,
                documentCollection: documentCollection.length ? documentCollection : undefined,
                supportGroups: supportGroups.length ? supportGroups : undefined,
                researchContext: response.operation || response.source_scopes?.length || response.sufficiency
                  ? {
                      operation: response.operation,
                      sourceScopes: response.source_scopes ?? [],
                      sufficiency: response.sufficiency?.status,
                    }
                  : undefined,
              }
            : m,
        ),
      );
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? {
                ...m,
                content: "Sumber hukum belum dapat diakses. Coba kembali beberapa saat lagi.",
                status: "complete",
              }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const stop = () => setIsStreaming(false);
  const allCitations = messages.flatMap((message) => message.citations ?? []);
  const panelCitations = activeCitation && !allCitations.some((c) => c.publicTargetId === activeCitation.publicTargetId)
    ? [...allCitations, activeCitation]
    : allCitations;

  return (
    <div className={`tj-root size-full ${theme === "dark" ? "tj-dark" : ""}`}>
      <div className="relative flex h-full w-full overflow-hidden bg-[var(--tj-bg)]" data-evidence-workspace>
        {/* Desktop / Tablet sidebar - Collapsible to icon rail */}
        <motion.div
          initial={false}
          animate={{ width: sidebarCollapsed || panelCompactNav ? 68 : 280 }}
          transition={{ type: "spring", stiffness: 360, damping: 38, mass: 0.7 }}
          className="hidden md:block h-full shrink-0 overflow-hidden"
          data-tjipto-navigation
        >
          <Sidebar
            active={route}
            collapsed={sidebarCollapsed || panelCompactNav}
            onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
            onNavigate={(r) => {
              setRoute(r);
              setUserMenuOpen(false);
            }}
            onNewChat={newChat}
            theme={theme}
            onToggleTheme={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
            userMenuOpen={userMenuOpen}
            onToggleUserMenu={() => setUserMenuOpen((v) => !v)}
          />
        </motion.div>

        {/* Mobile sidebar drawer */}
        <AnimatePresence>
          {mobileNavOpen && (
            <>
              <motion.div
                key="backdrop"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                onClick={() => setMobileNavOpen(false)}
                className="md:hidden fixed inset-0 z-40 bg-black/40"
              />
              <motion.div
                key="drawer"
                initial={{ x: "-100%" }}
                animate={{ x: 0 }}
                exit={{ x: "-100%" }}
                transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
                className="md:hidden fixed inset-y-0 left-0 z-50 w-[280px] max-w-[85vw]"
              >
                <div
                  className="h-full tj-glass-strong"
                  style={{
                    boxShadow: "var(--tj-shadow-panel)",
                    background: "color-mix(in srgb, var(--tj-bg) 88%, transparent)",
                    backdropFilter: "blur(32px) saturate(180%)",
                    WebkitBackdropFilter: "blur(32px) saturate(180%)",
                    borderRight: "0.5px solid var(--tj-glass-border)",
                  }}
                >
                  <MobileSidebarWrapper>
                    <Sidebar
                      active={route}
                      collapsed={false}
                      onToggleCollapse={() => setMobileNavOpen(false)}
                      onNavigate={(r) => {
                        setRoute(r);
                        setUserMenuOpen(false);
                        setMobileNavOpen(false);
                      }}
                      onNewChat={newChat}
                      theme={theme}
                      onToggleTheme={() =>
                        setTheme((t) => (t === "light" ? "dark" : "light"))
                      }
                      userMenuOpen={userMenuOpen}
                      onToggleUserMenu={() => setUserMenuOpen((v) => !v)}
                    />
                  </MobileSidebarWrapper>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>

        <main className="flex-1 flex flex-col h-full min-w-0 bg-transparent relative z-0 overflow-hidden">
          {/* Subtle mesh overlay for the main content area */}
          <div className="absolute inset-0 pointer-events-none opacity-20 mix-blend-soft-light z-[-1] overflow-hidden">
            <div className="absolute top-[-10%] right-[-5%] w-[60%] h-[50%] bg-zinc-400/5 blur-[120px] rounded-full" />
            <div className="absolute bottom-[-5%] left-[-10%] w-[50%] h-[40%] bg-zinc-500/5 blur-[100px] rounded-full" />
          </div>
          {/* Mobile top bar - Cleaner, minimal branding */}
          <div className="md:hidden h-14 flex items-center justify-between px-3 border-b border-[var(--tj-glass-border)] shrink-0 tj-glass-strong sticky top-0 z-30">
            <button
              onClick={() => setMobileNavOpen(true)}
              className="w-10 h-10 rounded-xl flex items-center justify-center hover:bg-[var(--tj-surface-hover)] active:scale-90 transition-all"
              aria-label="Buka menu"
            >
              <Menu size={20} className="text-[var(--tj-text-primary)]" />
            </button>
            <div className="flex items-center gap-2" aria-hidden="true" />
            <button
              onClick={newChat}
              className="w-10 h-10 rounded-xl flex items-center justify-center hover:bg-[var(--tj-surface-hover)] active:scale-90 transition-all"
              aria-label="Percakapan baru"
            >
              <SquarePen size={19} className="text-[var(--tj-text-primary)]" />
            </button>
          </div>

          <div className="flex-1 overflow-hidden relative scroll-smooth flex flex-col">
            {route === "chat" &&
              (hasChat ? (
                <ChatView
                  messages={messages}
                  onSubmit={submit}
                  isStreaming={isStreaming}
                  onStop={stop}
                  onCitationClick={setActiveCitation}
                  activeCitationId={activeCitation?.id}
                />
              ) : (
                <EmptyState onSubmit={submit} />
              ))}
            {route === "search" && <SearchRoute onOpenDocument={setActiveCitation} />}
            {route === "library" && <LibraryRoute onOpenDocument={setActiveCitation} />}
          </div>
        </main>

        <Suspense fallback={null}>
          <EvidencePanel
            citation={activeCitation}
            allCitations={panelCitations}
            onClose={() => setActiveCitation(null)}
            onSelect={setActiveCitation}
          />
        </Suspense>
      </div>
    </div>
  );
}

// Force-show the sidebar inside the mobile drawer (sidebar itself is hidden md:flex)
function MobileSidebarWrapper({ children }: { children: React.ReactNode }) {
  return <div className="h-full tj-mobile-sidebar-wrap">{children}</div>;
}
