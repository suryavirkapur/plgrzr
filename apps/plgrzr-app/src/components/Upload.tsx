import { useState, ChangeEvent } from "react";

interface ProcessingResult {
  [key: string]: any;
}

interface FileState {
  file: File | null;
  result: ProcessingResult | null;
}

/** @type {Array<number>} */
const codes: any[] = [];
/** @type {Array<number>} */
const cache: number[] = [];

function levenshteinEditDistance(
  value: string,
  other: string,
  insensitive: any
) {
  if (value === other) {
    return 0;
  }

  if (value.length === 0) {
    return other.length;
  }

  if (other.length === 0) {
    return value.length;
  }

  if (insensitive) {
    value = value.toLowerCase();
    other = other.toLowerCase();
  }

  let index = 0;

  while (index < value.length) {
    // eslint-disable-next-line unicorn/prefer-code-point
    codes[index] = value.charCodeAt(index);
    cache[index] = ++index;
  }

  let indexOther = 0;
  /** @type {number} */
  let result;

  while (indexOther < other.length) {
    // eslint-disable-next-line unicorn/prefer-code-point
    const code = other.charCodeAt(indexOther);
    let index = -1;
    let distance = indexOther++;
    result = distance;

    while (++index < value.length) {
      const distanceOther = code === codes[index] ? distance : distance + 1;
      distance = cache[index];
      result =
        distance > result
          ? distanceOther > result
            ? result + 1
            : distanceOther
          : distanceOther > distance
          ? distance + 1
          : distanceOther;
      cache[index] = result;
    }
  }

  return result || 0;
}

export default function PdfComparison(): JSX.Element {
  const [files, setFiles] = useState<{ [key: string]: FileState }>({
    file1: { file: null, result: null },
    file2: { file: null, result: null },
  });
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [similarity, setSimilarity] = useState<string | null>(null);

  const handleFileChange =
    (id: string) => (e: ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        setFiles((prev) => ({
          ...prev,
          [id]: { ...prev[id], file: e.target.files![0] },
        }));
        setError(null);
      }
    };

  const processFile = async (file: File): Promise<ProcessingResult> => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(
      "https://jc84c0wcwsskkccocc4gcso8.13.76.121.152.sslip.io/process-pdf",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }

    return await response.json();
  };

  const calculateSimilarity = (
    result1: ProcessingResult,
    result2: ProcessingResult
  ): string => {
    const text1 = Object.values(result1).join(" ");
    const text2 = Object.values(result2).join(" ");

    const editDistance = levenshteinEditDistance(text1, text2, false);
    const maxLength = Math.max(text1.length, text2.length);
    const normalizedSimilarity =
      maxLength === 0 ? 1 : (maxLength - editDistance) / maxLength;

    return `Edit Distance is ${editDistance} and Normalized Similarity is: ${normalizedSimilarity.toFixed(
      4
    )}`;
  };

  const handleSubmit = async (): Promise<void> => {
    if (!files.file1.file || !files.file2.file) {
      setError("Please select both PDF files");
      return;
    }

    if (
      !files.file1.file.name.endsWith(".pdf") ||
      !files.file2.file.name.endsWith(".pdf")
    ) {
      setError("Only PDF files are allowed");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [result1, result2] = await Promise.all([
        processFile(files.file1.file),
        processFile(files.file2.file),
      ]);

      setFiles({
        file1: { ...files.file1, result: result1 },
        file2: { ...files.file2, result: result2 },
      });

      const similarityResult = calculateSimilarity(result1, result2);
      setSimilarity(similarityResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>PDF Comparison</h2>

      <div>
        <div>
          <h3>First PDF</h3>
          <label>Upload PDF 1</label>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange("file1")}
          />
          {files.file1.result && (
            <pre>{JSON.stringify(files.file1.result, null, 2)}</pre>
          )}
        </div>

        <div>
          <h3>Second PDF</h3>
          <label>Upload PDF 2</label>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange("file2")}
          />
          {files.file2.result && (
            <pre>{JSON.stringify(files.file2.result, null, 2)}</pre>
          )}
        </div>
      </div>

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Processing..." : "Compare PDFs"}
      </button>

      {error && <div>{error}</div>}

      {similarity && (
        <div>
          <h3>Similarity Score (Higher is Better, Same is Zero)</h3>
          <p>{similarity}</p>
        </div>
      )}
    </div>
  );
}
