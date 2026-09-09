import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  variable: "--font-outfit",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CrediFlux | KNN Loan Prediction & Intelligence Engine",
  description:
    "AI-powered loan assessment and neighbor explainability system powered by an optimized K-Nearest Neighbors classifier.",
  keywords: ["loan prediction", "knn", "machine learning", "credit risk", "nextjs", "streamlit"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
