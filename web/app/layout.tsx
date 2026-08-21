import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TagPuncher",
  description: "Concept tags for any text, learned from Wikipedia's own links",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
