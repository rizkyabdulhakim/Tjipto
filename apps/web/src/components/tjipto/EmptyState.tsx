import { motion } from "motion/react";
import { ArrowUpRight } from "lucide-react";
import { Composer } from "./Composer";
import { suggestedPrompts } from "./data";

interface EmptyStateProps {
  onSubmit: (value: string) => void;
}

export function EmptyState({ onSubmit }: EmptyStateProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-3 sm:px-4 min-h-full">
      <div className="w-full max-w-[760px] flex flex-col items-center py-12">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center mb-10"
        >
          <h1
            className="text-center tracking-tight px-4"
            style={{
              fontWeight: 600,
              color: "var(--tj-text-primary)",
            }}
          >
            <span className="tj-h-hero block">Apa yang ingin Anda cari hari ini?</span>
          </h1>
        </motion.div>

        {/* Composer - Centered in Empty State */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="w-full"
        >
          <Composer onSubmit={onSubmit} compact />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.5 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-12 text-center"
          style={{ fontSize: 12, color: "var(--tj-text-muted)" }}
        >
          Tjipto dapat membuat kesalahan. Verifikasi sumber hukum penting.
        </motion.div>
      </div>
    </div>
  );
}
