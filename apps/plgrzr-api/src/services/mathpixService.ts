import { HTTPException } from "hono/http-exception";

interface MathpixOptions {
  conversion_formats: {
    docx: boolean;
    "tex.zip": boolean;
  };
  math_inline_delimiters: string[];
  rm_spaces: boolean;
}

export class MathpixService {
  private headers: Record<string, string>;
  private options: MathpixOptions;

  constructor(appId: string, apiKey: string) {
    this.headers = {
      app_id: appId,
      app_key: apiKey,
    };
    this.options = {
      conversion_formats: { docx: true, "tex.zip": true },
      math_inline_delimiters: ["$", "$"],
      rm_spaces: true,
    };
  }

  async processPdf(fileContent: ArrayBuffer): Promise<any> {
    try {
      // Initial PDF upload
      const formData = new FormData();
      formData.append("options_json", JSON.stringify(this.options));
      formData.append(
        "file",
        new Blob([fileContent], { type: "application/pdf" }),
        "filename.pdf"
      );

      const response = await fetch("https://api.mathpix.com/v3/pdf", {
        method: "POST",
        headers: this.headers,
        body: formData,
      });

      if (!response.ok) {
        throw new HTTPException(500, {
          message: `Mathpix API request failed: ${await response.text()}`,
        });
      }

      const responseData = await response.json();
      if (!responseData.pdf_id) {
        throw new HTTPException(500, {
          message: `Invalid response from Mathpix: ${JSON.stringify(
            responseData
          )}`,
        });
      }

      const docId = responseData.pdf_id;

      // Poll for completion
      while (true) {
        const statusResponse = await fetch(
          `https://api.mathpix.com/v3/pdf/${docId}`,
          {
            headers: this.headers,
          }
        );

        if (!statusResponse.ok) {
          throw new HTTPException(500, {
            message: `Failed to check status: ${await statusResponse.text()}`,
          });
        }

        const status = await statusResponse.json();
        if (status.status === "completed") {
          break;
        } else if (status.status === "error") {
          throw new HTTPException(500, {
            message: `Mathpix processing error: ${
              status.error || "Unknown error"
            }`,
          });
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      // Get final result
      const result = await fetch(
        `https://api.mathpix.com/v3/pdf/${docId}.lines.json`,
        {
          headers: this.headers,
        }
      );

      if (!result.ok) {
        throw new HTTPException(500, {
          message: `Failed to get results: ${await result.text()}`,
        });
      }

      return await result.json();
    } catch (error) {
      if (error instanceof HTTPException) {
        throw error;
      }
      throw new HTTPException(500, {
        message: `Error processing PDF: ${error}`,
      });
    }
  }
}
