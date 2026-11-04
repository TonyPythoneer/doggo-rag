import react from "@vitejs/plugin-react";
import { defineConfig } from "vite-plus";

// outDir stays "build": FastAPI serves frontend/build (app/main.py).
export default defineConfig({
  plugins: [react({ compiler: true })],
  server: { port: 3000 },
  build: { outDir: "build" },
});
