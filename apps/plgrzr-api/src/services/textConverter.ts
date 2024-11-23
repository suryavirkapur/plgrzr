interface Page {
  page: number;
  lines: Array<{ text: string }>;
}

interface MathpixResult {
  pages: Page[];
}

export class TextConverter {
  private result: MathpixResult;

  constructor(result: MathpixResult) {
    this.result = result;
  }

  convertPagewise(): string {
    const processedData: Record<string, string> = {};

    for (const page of this.result.pages) {
      const pageNumber = `Page_${page.page}`;
      const concatText = page.lines.map((line) => line.text).join(" ");
      processedData[pageNumber] = concatText;
    }

    return JSON.stringify(processedData);
  }
}
