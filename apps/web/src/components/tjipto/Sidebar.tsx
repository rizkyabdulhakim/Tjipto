import { useEffect, useRef } from "react";
import {
  SquarePen,
  Search,
  BookOpen,
  Settings,
  ChevronDown,
  LogOut,
  Sun,
  Moon,
  PanelLeftClose,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

export type Route = "chat" | "search" | "library";

interface SidebarProps {
  active: Route;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onNavigate: (route: Route) => void;
  onNewChat: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  userMenuOpen: boolean;
  onToggleUserMenu: () => void;
}

export function Sidebar({
  active,
  collapsed,
  onNavigate,
  onNewChat,
  onToggleCollapse,
  theme,
  onToggleTheme,
  userMenuOpen,
  onToggleUserMenu,
}: SidebarProps) {
  const userMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!userMenuOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      if (
        userMenuRef.current &&
        !userMenuRef.current.contains(e.target as Node)
      ) {
        onToggleUserMenu();
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [userMenuOpen, onToggleUserMenu]);

  if (collapsed) {
    return (
      <CollapsedSidebar
        active={active}
        onNavigate={onNavigate}
        onNewChat={onNewChat}
        onExpand={onToggleCollapse}
      />
    );
  }

  return (
    <aside
      className="hidden md:flex flex-col shrink-0 h-full overflow-hidden border-r border-[var(--tj-glass-border)] bg-transparent relative z-20"
      style={{ width: 280 }}
    >
      {/* Brand area — cleaner, focus on calmness */}
      <div className="h-[72px] flex items-center shrink-0 px-3 mb-2 gap-2">
        <button
          onClick={onNewChat}
          className="flex items-center gap-3 px-2 py-2 rounded-2xl min-w-0 flex-1 hover:bg-[var(--tj-surface-hover)] transition-colors"
          aria-label="Tjipto home"
        >
          <TjiptoLogo size={36} />
          <div className="flex flex-col items-start leading-tight min-w-0">
            <span
              className="tracking-tight truncate"
              style={{
                fontSize: 18,
                fontWeight: 700,
                color: "var(--tj-text-primary)",
                letterSpacing: "-0.03em",
              }}
            >
              Tjipto
            </span>
          </div>
        </button>
        <button
          onClick={onToggleCollapse}
          className="w-9 h-9 shrink-0 rounded-xl flex items-center justify-center text-[var(--tj-text-secondary)] hover:text-[var(--tj-text-primary)] hover:bg-[var(--tj-surface-hover)] transition-colors"
          aria-label="Tutup sidebar"
          title="Tutup sidebar"
        >
          <PanelLeftClose size={18} />
        </button>
      </div>

      {/* Primary nav — flush-left, smooth active pill */}
      <nav className="flex flex-col gap-1 px-3 mb-6">
        <NavButton
          icon={<SquarePen size={20} />}
          label="Mulai Konsultasi"
          shortcut="⌘N"
          tint="blue"
          onClick={onNewChat}
          active={active === "chat"}
        />
        <NavButton
          icon={<Search size={20} />}
          label="Cari Peraturan"
          tint="indigo"
          active={active === "search"}
          onClick={() => onNavigate("search")}
        />
        <NavButton
          icon={<BookOpen size={20} />}
          label="Penanda"
          tint="teal"
          active={active === "library"}
          onClick={() => onNavigate("library")}
        />
      </nav>

      <div className="flex-1" />

      {/* User menu — cleaner iOS style */}
      <div ref={userMenuRef} className="relative border-t border-[var(--tj-glass-border)] p-3 shrink-0">
        <AnimatePresence>
          {userMenuOpen && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ duration: 0.15, ease: "easeOut" }}
              className="absolute bottom-[76px] left-3 right-3 rounded-2xl bg-[var(--tj-surface)] border border-[var(--tj-border)] overflow-hidden shadow-2xl z-50 backdrop-blur-xl"
            >
              {/* Identity */}
              <div className="px-4 py-3 border-b border-[var(--tj-border-subtle)] flex items-center gap-3">
                <div
                  className="w-9 h-9 rounded-full bg-[linear-gradient(160deg,#ff7bb0_0%,#ff5e9b_45%,#5e5ce6_100%)] text-white flex items-center justify-center shrink-0 shadow-sm"
                  style={{ fontSize: 13, fontWeight: 700 }}
                >
                  AT
                </div>
                <div className="min-w-0">
                  <div
                    className="truncate"
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: "var(--tj-text-primary)",
                    }}
                  >
                    Ahmad Tjipto
                  </div>
                  <div
                    className="truncate"
                    style={{ fontSize: 12, color: "var(--tj-text-muted)" }}
                  >
                    ahmad@firma.id
                  </div>
                </div>
              </div>

              {/* Menu items */}
              <div className="py-1.5">
                <div className="px-4 py-2 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <span
                      className="inline-flex items-center justify-center w-7 h-7 rounded-lg text-white shadow-sm"
                      style={{
                        background:
                          theme === "dark"
                            ? "linear-gradient(160deg,#3a3a55 0%,#1c1c2e 100%)"
                            : "linear-gradient(160deg,#ffd56b 0%,#ff9f0a 100%)",
                      }}
                    >
                      {theme === "dark" ? <Moon size={14} /> : <Sun size={14} />}
                    </span>
                    <span style={{ fontSize: 13.5, fontWeight: 500, color: "var(--tj-text-primary)" }}>
                      Tampilan Gelap
                    </span>
                  </div>
                  <IosSwitch checked={theme === "dark"} onChange={onToggleTheme} />
                </div>

                <button
                  className="w-full h-10 flex items-center gap-3 px-4 hover:bg-[var(--tj-surface-hover)] text-left transition-colors"
                  style={{ fontSize: 13.5, fontWeight: 500, color: "var(--tj-text-primary)" }}
                >
                  <Settings size={15} className="text-[var(--tj-text-secondary)]" />
                  Pengaturan
                </button>

                <div className="h-px bg-[var(--tj-border-subtle)] my-1" />

                <button
                  className="w-full h-10 flex items-center gap-3 px-4 hover:bg-[var(--tj-surface-hover)] text-left transition-colors"
                  style={{ fontSize: 13.5, fontWeight: 500, color: "var(--tj-error)" }}
                >
                  <LogOut size={15} />
                  Keluar
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          onClick={onToggleUserMenu}
          className={`w-full flex items-center rounded-2xl transition-colors h-12 gap-3 px-3 ${userMenuOpen ? "bg-[var(--tj-surface-hover)]" : "hover:bg-[var(--tj-surface-hover)]"}`}
        >
          <div
            className="w-8 h-8 rounded-full bg-[linear-gradient(160deg,#ff7bb0_0%,#ff5e9b_45%,#5e5ce6_100%)] text-white flex items-center justify-center shrink-0 shadow-sm"
            style={{ fontSize: 12, fontWeight: 700 }}
          >
            AT
          </div>
          <div className="flex-1 text-left overflow-hidden">
            <div
              className="truncate"
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: "var(--tj-text-primary)",
                letterSpacing: "-0.01em",
              }}
            >
              Ahmad Tjipto
            </div>
            <div
              className="truncate flex items-center gap-1.5"
              style={{ fontSize: 11, fontWeight: 500, color: "var(--tj-text-muted)" }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
              Pro Account
            </div>
          </div>
          <ChevronDown
            size={14}
            className={`text-[var(--tj-text-muted)] transition-transform duration-200 ${userMenuOpen ? "rotate-180" : ""}`}
          />
        </button>
      </div>
    </aside>
  );
}

function NavButton({
  icon,
  label,
  shortcut,
  badge,
  tint,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  shortcut?: string;
  badge?: string;
  tint?: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group relative rounded-xl text-left flex items-center h-11 gap-3.5 px-3 w-full overflow-hidden"
      style={{ color: "var(--tj-text-primary)" }}
    >
      {active && (
        <span
          aria-hidden
          className="absolute inset-0 rounded-xl pointer-events-none"
          style={{
            background: "var(--tj-glass-fill)",
            border: "0.5px solid var(--tj-glass-border)",
            boxShadow: "var(--tj-shadow-sm)",
            transition: "none",
            animation: "none",
          }}
        />
      )}
      {/* Hover overlay (only when inactive) */}
      {!active && (
        <span
          aria-hidden
          className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none bg-[var(--tj-surface-hover)]"
        />
      )}
      <span
        className="relative shrink-0 flex items-center justify-center w-6 h-6"
        style={{
          color: tint ? (IOS_TINTS_COLOR[tint] || tint) : "inherit",
          filter: "none",
        }}
      >
        {icon}
      </span>
      <span
        className="relative truncate"
        style={{
          fontSize: 14.5,
          fontWeight: 500,
          letterSpacing: "-0.01em",
          color: active ? "var(--tj-text-primary)" : "var(--tj-text-secondary)",
        }}
      >
        {label}
      </span>
      {badge && (
        <span
          className="relative ml-auto inline-flex items-center px-2 h-[18px] rounded-full"
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.02em",
            background: "var(--tj-accent-soft)",
            color: "var(--tj-accent)",
          }}
        >
          {badge}
        </span>
      )}
      {shortcut && !badge && (
        <span className="relative ml-auto text-[10px] text-[var(--tj-text-muted)] font-semibold tracking-tighter opacity-0 group-hover:opacity-100 transition-opacity">
          {shortcut}
        </span>
      )}
    </button>
  );
}

function CollapsedSidebar({
  active,
  onNavigate,
  onNewChat,
  onExpand,
}: {
  active: Route;
  onNavigate: (route: Route) => void;
  onNewChat: () => void;
  onExpand: () => void;
}) {
  return (
    <aside
      className="hidden md:flex flex-col shrink-0 h-full overflow-hidden border-r border-[var(--tj-glass-border)] bg-transparent relative z-20 items-center"
      style={{ width: 68 }}
    >
      {/* Brand: logo that swaps to open-sidebar icon on hover */}
      <div className="h-[72px] flex items-center justify-center shrink-0 mb-2">
        <button
          onClick={onExpand}
          className="group relative w-11 h-11 rounded-2xl flex items-center justify-center hover:bg-[var(--tj-surface-hover)] transition-colors"
          aria-label="Buka sidebar"
          title="Buka sidebar"
        >
          <span className="absolute inset-0 flex items-center justify-center transition-opacity duration-200 group-hover:opacity-0">
            <TjiptoLogo size={36} />
          </span>
          <span className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity duration-200 group-hover:opacity-100 text-[var(--tj-text-primary)]">
            <PanelLeftOpenIcon />
          </span>
        </button>
      </div>

      {/* Icon-only nav */}
      <nav className="flex flex-col gap-1 px-2">
        <CollapsedNavButton
          icon={<SquarePen size={20} />}
          label="Mulai Konsultasi"
          tint="blue"
          active={active === "chat"}
          onClick={onNewChat}
        />
        <CollapsedNavButton
          icon={<Search size={20} />}
          label="Cari Peraturan"
          tint="indigo"
          active={active === "search"}
          onClick={() => onNavigate("search")}
        />
        <CollapsedNavButton
          icon={<BookOpen size={20} />}
          label="Penanda"
          tint="teal"
          active={active === "library"}
          onClick={() => onNavigate("library")}
        />
      </nav>
    </aside>
  );
}

function CollapsedNavButton({
  icon,
  label,
  tint,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  tint?: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className="group relative w-11 h-11 rounded-xl flex items-center justify-center overflow-hidden"
    >
      {active && (
        <span
          aria-hidden
          className="absolute inset-0 rounded-xl pointer-events-none"
          style={{
            background: "var(--tj-glass-fill)",
            border: "0.5px solid var(--tj-glass-border)",
            boxShadow: "var(--tj-shadow-sm)",
            transition: "none",
            animation: "none",
          }}
        />
      )}
      {!active && (
        <span
          aria-hidden
          className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none bg-[var(--tj-surface-hover)]"
        />
      )}
      <span
        className="relative flex items-center justify-center"
        style={{
          color: tint ? (IOS_TINTS_COLOR[tint] || tint) : "inherit",
          filter: "none",
        }}
      >
        {icon}
      </span>
    </button>
  );
}

function PanelLeftOpenIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M9 3v18" />
      <path d="m14 9 3 3-3 3" />
    </svg>
  );
}

const IOS_TINTS_COLOR: Record<string, string> = {
  blue: "#0a84ff",
  indigo: "#5e5ce6",
  teal: "#30d1b8",
  pink: "#ff5e9b",
  graphite: "#8e8e93",
};

/* iOS 26-style switch — pill track, soft thumb, smooth spring. */
function IosSwitch({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className="relative shrink-0 rounded-full transition-colors active:scale-95"
      style={{
        width: 44,
        height: 26,
        background: checked ? "#34c759" : "rgba(120, 120, 128, 0.32)",
        boxShadow:
          "inset 0 0 0 0.5px rgba(0,0,0,0.04), inset 0 1px 2px rgba(0,0,0,0.06)",
        transition: "background 220ms ease",
      }}
    >
      <span
        className="absolute top-[2px] left-[2px] rounded-full bg-white"
        style={{
          width: 22,
          height: 22,
          transform: checked ? "translateX(18px)" : "translateX(0)",
          transition: "transform 240ms cubic-bezier(0.2, 0.8, 0.2, 1)",
          boxShadow:
            "0 2px 4px rgba(0,0,0,0.18), 0 1px 1px rgba(0,0,0,0.1), inset 0 0 0 0.5px rgba(0,0,0,0.04)",
        }}
      />
    </button>
  );
}

/* Dummy brand mark: iOS 26 squircle with stylized "T" pillar + serif crossbar,
   inner specular highlight and outer ambient glow. Pure SVG so it scales crisp. */
export function TjiptoLogo({ size = 36 }: { size?: number }) {
  const id = "tj-logo-" + size;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Tjipto logo"
      style={{
        filter:
          "drop-shadow(0 4px 10px rgba(10,132,255,0.3))",
      }}
    >
      <defs>
        <linearGradient id={`${id}-bg`} x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5fb4ff" />
          <stop offset="0.55" stopColor="#0a84ff" />
          <stop offset="1" stopColor="#5e5ce6" />
        </linearGradient>
        <linearGradient id={`${id}-shine`} x1="0" y1="0" x2="0" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#ffffff" stopOpacity="0.4" />
          <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
        </linearGradient>
        <linearGradient id={`${id}-t`} x1="0" y1="0" x2="0" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#e9efff" />
        </linearGradient>
      </defs>
      {/* squircle base */}
      <path
        d="M32 2c11.6 0 17.4 0 22.4 2.9a16 16 0 0 1 6.7 6.7C64 16.6 64 22.4 64 34v0c0 11.6 0 17.4-2.9 22.4a16 16 0 0 1-6.7 6.7C49.4 64 43.6 64 32 64v0c-11.6 0-17.4 0-22.4-2.9a16 16 0 0 1-6.7-6.7C0 49.4 0 43.6 0 32v0C0 20.4 0 14.6 2.9 9.6A16 16 0 0 1 9.6 2.9C14.6 0 20.4 0 32 0Z"
        fill={`url(#${id}-bg)`}
      />
      {/* top specular sheen */}
      <path
        d="M32 2c11.6 0 17.4 0 22.4 2.9a16 16 0 0 1 6.7 6.7C63.7 16 63.95 21 64 31H0C.05 21 .3 16 2.9 11.6A16 16 0 0 1 9.6 2.9C14.6 0 20.4 0 32 0Z"
        fill={`url(#${id}-shine)`}
        opacity="0.85"
      />
      {/* T mark: capital with serif crossbar */}
      <g fill={`url(#${id}-t)`}>
        <rect x="18" y="24" width="28" height="4.5" rx="1.5" />
        <rect x="29.5" y="24" width="5" height="24" rx="1.5" />
      </g>
    </svg>
  );
}
