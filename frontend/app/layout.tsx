import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Unmap.ai",
  description:
    "Unmap.ai - Advanced AI agent security testing and malicious behavior detection platform.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-mono">{children}</body>
    </html>
  );
}
