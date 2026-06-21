import type { ChatMessage, ChatThread } from "../../lib/types";

export const suggestedPrompts = [
  { title: "Pasal 1 ayat (3)", subtitle: "negara hukum" },
  { title: "Pembukaan", subtitle: "dasar negara" },
  { title: "Hak asasi manusia", subtitle: "dalam UUD 1945" },
  { title: "Majelis Permusyawaratan Rakyat", subtitle: "ketentuan UUD" },
];

export const chatHistory: ChatThread[] = [
  { id: "t1", title: "Pasal 1 ayat (3)", group: "Today", active: true },
  { id: "t2", title: "Pembukaan UUD 1945", group: "Today" },
  { id: "t3", title: "Negara hukum", group: "Today" },
  { id: "t4", title: "Hak asasi manusia", group: "Today" },
  { id: "t5", title: "Majelis Permusyawaratan Rakyat", group: "Today" },
];

export const conversation: ChatMessage[] = [];
