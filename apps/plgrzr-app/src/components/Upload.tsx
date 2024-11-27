import { useState, type ChangeEvent } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Upload } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ProcessingResult {
  [key: string]: any;
}

interface FileState {
  file: File | null;
  result: ProcessingResult | null;
}

const codes: any[] = [];
const cache: number[] = [];

function levenshteinEditDistance(
  value: string,
  other: string,
  insensitive: any
) {
  // ... [keeping the original levenshtein implementation unchanged]
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
    codes[index] = value.charCodeAt(index);
    cache[index] = ++index;
  }

  let indexOther = 0;
  let result;

  while (indexOther < other.length) {
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
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>PDF Comparison Tool</CardTitle>
            <CardDescription>
              Upload two PDF files to compare their content and calculate
              similarity
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* First PDF Upload */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">First PDF</CardTitle>
                </CardHeader>
                <CardContent>
                  {/* biome-ignore lint/a11y/useKeyWithClickEvents: <explanation> */}
                  <div
                    className="border-2 border-dashed rounded-lg p-4 text-center cursor-pointer hover:border-primary"
                    onClick={() => document.getElementById("pdf1")?.click()}
                  >
                    <input
                      id="pdf1"
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange("file1")}
                      className="hidden"
                    />
                    <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                    <p className="text-sm text-gray-600">
                      {files.file1.file
                        ? files.file1.file.name
                        : "Click to upload PDF 1"}
                    </p>
                  </div>
                  {files.file1.result && (
                    <ScrollArea className="h-40 mt-4 rounded-md border p-4">
                      <pre className="text-xs">
                        {JSON.stringify(files.file1.result, null, 2)}
                      </pre>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>

              {/* Second PDF Upload */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Second PDF</CardTitle>
                </CardHeader>
                <CardContent>
                  {/* biome-ignore lint/a11y/useKeyWithClickEvents: <explanation> */}
                  <div
                    className="border-2 border-dashed rounded-lg p-4 text-center cursor-pointer hover:border-primary"
                    onClick={() => document.getElementById("pdf2")?.click()}
                  >
                    <input
                      id="pdf2"
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange("file2")}
                      className="hidden"
                    />
                    <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                    <p className="text-sm text-gray-600">
                      {files.file2.file
                        ? files.file2.file.name
                        : "Click to upload PDF 2"}
                    </p>
                  </div>
                  {files.file2.result && (
                    <ScrollArea className="h-40 mt-4 rounded-md border p-4">
                      <pre className="text-xs">
                        {JSON.stringify(files.file2.result, null, 2)}
                      </pre>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing PDFs...
                </>
              ) : (
                "Compare PDFs"
              )}
            </Button>

            {loading && <Progress value={30} className="w-full" />}

            {similarity && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Similarity Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">{similarity}</p>
                </CardContent>
              </Card>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
