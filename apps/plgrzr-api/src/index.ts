import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { auth } from "./lib/auth.js";
import { cors } from "hono/cors";
import { MathpixService } from "./services/mathpixService.js";
import { TextConverter } from "./services/textConverter.js";

import { fromBuffer } from "pdf2pic";
import { existsSync, mkdir } from "fs";

const app = new Hono<{
  Variables: {
    user: any;
    session: any;
  };
}>();

const APP_ID = process.env.APP_ID || "";
const API_KEY = process.env.API_KEY || "";

const mathpixService = new MathpixService(APP_ID, API_KEY);

app.use(
  "*",
  cors({
    origin: ["http://localhost:3000", "https://plgrzr.suryavirkapur.com"],
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["POST", "GET", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
    credentials: true,
  })
);

app.get("/api/auth/*", (c) => auth.handler(c.req.raw));
app.post("/api/auth/*", (c) => auth.handler(c.req.raw));

app.post("/process-pdf", async (c) => {
  const formData = await c.req.formData();

  const file = formData.get("file") as File;

  if (!file || !file.name.endsWith(".pdf")) {
    return c.json({ error: "Only PDF files are allowed" }, 400);
  }

  try {
    const fileBuffer = await file.arrayBuffer();
    const result = await mathpixService.processPdf(fileBuffer);
    const processedResult = new TextConverter(result).convertPagewise();

    return c.json(JSON.parse(processedResult));
  } catch (error) {
    if (error instanceof Error) {
      return c.json({ error: error.message }, 500);
    }
    return c.json({ error: "An unknown error occurred" }, 500);
  }
});

app.post("/pdf2img", async (c) => {
  const formData = await c.req.formData();
  const file = formData.get("file") as File;

  if (!file || !file.name.endsWith(".pdf")) {
    return c.json({ error: "Only PDF files are allowed" }, 400);
  }

  try {
    // Create images directory if it doesn't exist
    const imagesDir = "./images";
    if (!existsSync(imagesDir)) {
      await mkdir(imagesDir, { recursive: true }, () => {});
    }

    const fileBuffer = (await file.bytes()).buffer;
    const buffer = Buffer.from(fileBuffer);
    const options = {
      density: 100,
      saveFilename: "untitled",
      savePath: "./images",
      format: "png",
      width: 600,
      height: 800,
    };

    const convert = fromBuffer(buffer, options);
    const pageToConvertAsImage = 1;

    const imageResult = await convert(pageToConvertAsImage, {
      responseType: "image",
    });

    // Set appropriate headers for image response
    c.header("Content-Type", "image/png");
    return c.json({ finished: true });
  } catch (error) {
    if (error instanceof Error) {
      return c.json({ error: error.message }, 500);
    }
    return c.json({ error: "An unknown error occurred" }, 500);
  }
});

app.get("/health", (c) => {
  return c.json({ status: "healthy" });
});

app.get("/", (c) => {
  return c.text("Hello Hono!");
});

const port = 3001;
console.log(`Server is running on http://localhost:${port}`);

serve({
  fetch: app.fetch,
  port,
});
