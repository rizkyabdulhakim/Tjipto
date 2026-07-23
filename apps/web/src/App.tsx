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
  mapAskResponseToSupportGroups,
  mapAskResponseToSupportItems,
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
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

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
    setMobileNavOpen(false);
  };

  const submit = async (value: string, filters?: { source_role: string }, displayValue = value) => {
    if (!hasChat) setHasChat(true);
    setRoute("chat");
    const userMsg: TMessage = {
      id: "u_" + Date.now(),
      role: "user",
      content: displayValue,
    };
    const asstId = "a_" + Date.now();
    setMessages((prev) => [
      ...prev,
      userMsg,
      {
        id: asstId,
        role: "assistant",
        content: "Menghubungi runtime UUD terverifikasi...",
        status: "streaming",
      },
    ]);
    setIsStreaming(true);

    try {
      const response = await askLegal(value, filters);
      const citations = mapAskResponseToCitations(response);
      const documentSource = mapAskResponseToDocumentSource(response);
      const support = mapAskResponseToSupportItems(response);
      const supportGroups = mapAskResponseToSupportGroups(response);
      const content = answerTextOrFallback(response);
      setActiveCitation(documentSource);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? {
                ...m,
                content,
                status: "complete",
                runtimeStatus: response.status,
                citations: citations.length ? citations : undefined,
                metadataSupport: support.metadata.length ? support.metadata : undefined,
                structuralSupport: support.structure.length ? support.structure : undefined,
                traceSupport: support.trace.length ? support.trace : undefined,
                supportGroups: supportGroups.length ? supportGroups : undefined,
                clarificationOptions: (response.clarification_options ?? [])
                  .filter((option) => option.label)
                  .map((option) => ({ sourceRole: option.source_role, label: option.label as string })),
                clarificationQuery: response.status === "clarification_required" ? value : undefined,
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
                content: "Backend UUD belum tersedia. Bukti tidak cukup / database belum tersedia dalam korpus UUD terverifikasi saat ini.",
                status: "complete",
                runtimeStatus: "backend_unavailable",
              }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const clarify = (query: string, sourceRole: string, label: string) =>
    submit(query, { source_role: sourceRole }, `${query}\nKonteks sumber: ${label}`);

  const stop = () => setIsStreaming(false);
  const allCitations = messages.flatMap((message) => message.citations ?? []);
  const panelCitations = activeCitation && !allCitations.some((c) => c.publicTargetId === activeCitation.publicTargetId)
    ? [...allCitations, activeCitation]
    : allCitations;

  return (
    <div className={`tj-root size-full ${theme === "dark" ? "tj-dark" : ""}`}>
      <div className="relative flex h-full w-full overflow-hidden bg-[var(--tj-bg)]">
        {/* Desktop / Tablet sidebar - Collapsible to icon rail */}
        <motion.div
          initial={false}
          animate={{ width: sidebarCollapsed ? 68 : 280 }}
          transition={{ type: "spring", stiffness: 360, damping: 38, mass: 0.7 }}
          className="hidden md:block h-full shrink-0 overflow-hidden"
        >
          <Sidebar
            active={route}
            collapsed={sidebarCollapsed}
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
              aria-label="Open menu"
            >
              <Menu size={20} className="text-[var(--tj-text-primary)]" />
            </button>
            <div className="flex items-center gap-2" aria-hidden="true" />
            <button
              onClick={newChat}
              className="w-10 h-10 rounded-xl flex items-center justify-center hover:bg-[var(--tj-surface-hover)] active:scale-90 transition-all"
              aria-label="New chat"
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
                  onClarify={clarify}
                  isStreaming={isStreaming}
                  onStop={stop}
                  onCitationClick={setActiveCitation}
                  activeCitationId={activeCitation?.id}
                />
              ) : (
                <EmptyState onSubmit={submit} />
              ))}
            {route === "search" && <SearchRoute onOpenCitation={setActiveCitation} />}
            {route === "library" && <LibraryRoute />}
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
