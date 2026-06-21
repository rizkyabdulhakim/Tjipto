import { useRef, useEffect, useState, type KeyboardEvent } from "react";
import { Plus, ArrowUp, Mic, Square } from "lucide-react";

interface ComposerProps {
  onSubmit: (value: string) => void;
  isStreaming?: boolean;
  onStop?: () => void;
  compact?: boolean;
}

export function Composer({ onSubmit, isStreaming, onStop, compact }: ComposerProps) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [placeholderIndex, setPlaceholderIndex] = useState(0);
  const [displayText, setDisplayText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);

  const placeholders = [
    "Tanya UUD 1945...",
    "Pasal 1 ayat (3)",
    "Pembukaan",
    "Negara hukum",
    "Hak asasi manusia"
  ];

  // Typing effect for placeholder
  useEffect(() => {
    if (focused || isHovered || value) {
      if (displayText !== "") setDisplayText("");
      return;
    }

    const currentFullText = placeholders[placeholderIndex];
    const speed = isDeleting ? 40 : 80;
    const pause = isDeleting ? 1000 : 2000;

    const timeout = setTimeout(() => {
      if (!isDeleting && displayText === currentFullText) {
        // Pause at the end of typing
        setTimeout(() => setIsDeleting(true), pause);
      } else if (isDeleting && displayText === "") {
        setIsDeleting(false);
        setPlaceholderIndex((prev) => (prev + 1) % placeholders.length);
      } else {
        setDisplayText(
          isDeleting
            ? currentFullText.substring(0, displayText.length - 1)
            : currentFullText.substring(0, displayText.length + 1)
        );
      }
    }, speed);

    return () => clearTimeout(timeout);
  }, [displayText, isDeleting, placeholderIndex, focused, value]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }, [value]);

  const submit = () => {
    const v = value.trim();
    if (!v || isStreaming) return;
    onSubmit(v);
    setValue("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = value.trim().length > 0 && !isStreaming;

  return (
    <div 
      className="w-full max-w-[760px] mx-auto px-3 sm:px-4"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className={`relative rounded-[30px] overflow-hidden transition-all duration-500 ${
          focused ? "scale-[1.015]" : "scale-100"
        }`}
        style={{
          boxShadow: focused ? "var(--tj-accent-glow)" : "var(--tj-shadow-composer)",
        }}
      >
        {/* Animated Border Beam - "Lampu Mengelilingi" */}
        <div className="absolute inset-0 z-0">
          <div className="absolute inset-[-1px] rounded-[30px] border border-transparent overflow-hidden">
             {/* The moving light beam */}
             <div 
               className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[200%] h-[200%]"
               style={{
                 background: "conic-gradient(from 0deg, transparent 0%, transparent 40%, var(--tj-accent) 50%, transparent 60%, transparent 100%)",
                 animation: "tj-spin 4s linear infinite",
                 opacity: focused ? 0.8 : 0.3,
               }}
             />
          </div>
        </div>

        {/* Inner Content Surface */}
        <div className="relative z-10 m-[1.5px] rounded-[28.5px] tj-glass tj-glass-shine border-0 flex items-center min-h-[56px]">
          <div className="absolute left-2.5 flex items-center justify-center">
            {/* Left attachment */}
            <button
              className="w-9 h-9 rounded-full flex items-center justify-center text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)] transition-colors"
              aria-label="Attach"
            >
              <Plus size={18} strokeWidth={2} />
            </button>
          </div>

          <textarea
            ref={ref}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            rows={1}
            placeholder={focused || value ? "Tanya UUD 1945..." : displayText}
            className="w-full resize-none bg-transparent outline-none px-14 py-[14px] placeholder:text-[var(--tj-text-muted)] text-[var(--tj-text-primary)] leading-[24px] tj-scroll self-center"
            style={{ fontSize: 16, minHeight: 52, maxHeight: 180, display: 'flex', alignItems: 'center' }}
          />

          {/* Right cluster */}
          <div className="absolute right-2.5 flex items-center gap-1.5">
            {!isStreaming && !canSend && (
              <button
                className="w-9 h-9 rounded-full flex items-center justify-center text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)] transition-colors"
                aria-label="Dictate"
              >
                <Mic size={16} />
              </button>
            )}
            {isStreaming ? (
              <button
                onClick={onStop}
                className="w-9 h-9 rounded-full flex items-center justify-center text-[var(--tj-text-primary)] transition-colors hover:scale-110"
                aria-label="Stop generating"
              >
                <Square size={12} fill="currentColor" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!canSend}
                className={`w-9 h-9 rounded-full flex items-center justify-center transition-all duration-300 ${
                  canSend
                    ? "text-[#0a84ff] scale-110"
                    : "text-[var(--tj-text-muted)] opacity-30 scale-95"
                }`}
                aria-label="Send"
              >
                <ArrowUp size={18} strokeWidth={2.4} />
              </button>
            )}
          </div>
        </div>
      </div>

      {!compact && (
        <p
          className="text-center mt-2 mb-3"
          style={{ fontSize: 12, color: "var(--tj-text-muted)" }}
        >
          Tjipto dapat membuat kesalahan. Verifikasi sumber hukum penting.
        </p>
      )}
    </div>
  );
}
