import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OK Penthouse",
  description:
    "Preparing for a Perplexity Shop & ChatGPT Storefront world. Researching API-first ecommerce and buying agents.",
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
