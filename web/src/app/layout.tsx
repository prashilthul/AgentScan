import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "agentscan · AI governance report",
  description: "Inventory, risk and audit trail of the AI running in an AWS account",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-muted/30 font-sans">{children}</body>
    </html>
  );
}
