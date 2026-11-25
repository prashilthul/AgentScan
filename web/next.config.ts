import type { NextConfig } from "next";

// Built as static files (web/out) that `agentscan serve` hosts next to its API.
const nextConfig: NextConfig = {
  output: "export",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
