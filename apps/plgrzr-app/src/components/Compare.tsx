import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, CheckCircle2, Upload, ArrowRight } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface SimilarSegment {
  segment: number;
  similarity: number;
}

interface ComparisonResponse {
  anomalies: {
    [key: string]: boolean;
  };
  textual_inconsistencies: {
    [key: string]: boolean;
  };
  visual_similarity: {
    similar_segments: SimilarSegment[];
  };
  semantic_similarity: {
    similar_segments: SimilarSegment[];
  };
}

interface ComparisonWithFiles extends ComparisonResponse {
  file1: string;
  file2: string;
}

interface FileWithPreview extends File {
  preview?: string;
}

const PDFAnalysisDashboard = () => {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [results, setResults] = useState<ComparisonWithFiles[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 3) {
      setError("Please select only up to 3 PDF files");
      return;
    }

    setFiles(
      acceptedFiles.map((file) =>
        Object.assign(file, {
          preview: URL.createObjectURL(file),
        })
      )
    );
    setError(null);
  }, []);

  const analyzePDFs = async () => {
    if (files.length < 2) {
      setError("Please select at least 2 PDF files to compare");
      return;
    }

    setLoading(true);
    setError(null);
    const comparisons: ComparisonWithFiles[] = [];

    try {
      // Compare each PDF with every other PDF (avoiding duplicates)
      for (let i = 0; i < files.length - 1; i++) {
        for (let j = i + 1; j < files.length; j++) {
          const formData = new FormData();
          formData.append("pdf1", files[i]);
          formData.append("pdf2", files[j]);

          const response = await fetch("http://localhost:8001/compare_pdfs", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Error comparing PDFs: ${response.statusText}`);
          }

          const result = await response.json();
          comparisons.push({
            ...result,
            file1: files[i].name,
            file2: files[j].name,
          });
        }
      }

      setResults(comparisons);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An error occurred during analysis"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>PDF Analysis Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* File Upload Area */}
            <div
              className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary"
              onClick={() => document.getElementById("file-upload")?.click()}
            >
              <input
                id="file-upload"
                type="file"
                multiple
                accept=".pdf"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) {
                    onDrop(Array.from(e.target.files));
                  }
                }}
              />
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2">
                Drop up to 3 PDF files here or click to browse
              </p>
            </div>

            {/* Selected Files List */}
            {files.length > 0 && (
              <div className="space-y-2">
                <h3 className="font-semibold">Selected Files:</h3>
                {files.map((file, index) => (
                  <div
                    key={file.name + index}
                    className="flex items-center gap-2"
                  >
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>{file.name}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Analysis Button */}
            <Button
              onClick={analyzePDFs}
              disabled={loading || files.length < 2}
              className="w-full"
            >
              {loading ? "Analyzing..." : "Analyze PDFs"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Loading State */}
      {loading && (
        <Card>
          <CardContent className="py-6">
            <Progress value={30} className="w-full" />
            <p className="text-center mt-2">Analyzing PDFs...</p>
          </CardContent>
        </Card>
      )}

      {/* Results Display */}
      {results.length > 0 && (
        <ScrollArea className="h-[600px]">
          <Card>
            <CardHeader>
              <CardTitle>Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              {results.map((result, index) => (
                <div key={index} className="space-y-6 mb-8">
                  {/* Comparison Header */}
                  <div className="flex items-center gap-2 font-semibold text-lg">
                    <span className="text-muted-foreground">Comparing:</span>
                    <span className="text-primary">{result.file1}</span>
                    <ArrowRight className="h-4 w-4" />
                    <span className="text-primary">{result.file2}</span>
                  </div>

                  {/* Comparison Results Table */}
                  <div className="space-y-4">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Metric</TableHead>
                          <TableHead>Result</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {/* Anomalies */}
                        {Object.entries(result.anomalies).map(
                          ([pdf, hasAnomaly]) => (
                            <TableRow key={pdf}>
                              <TableCell>Anomalies in {pdf}</TableCell>
                              <TableCell>
                                {hasAnomaly ? (
                                  <span className="text-destructive font-medium">
                                    Detected
                                  </span>
                                ) : (
                                  <span className="text-green-500 font-medium">
                                    None
                                  </span>
                                )}
                              </TableCell>
                            </TableRow>
                          )
                        )}

                        {/* Textual Inconsistencies */}
                        {Object.entries(result.textual_inconsistencies).map(
                          ([pdf, hasInconsistency]) => (
                            <TableRow key={`inconsistency-${pdf}`}>
                              <TableCell>
                                Textual Inconsistencies in {pdf}
                              </TableCell>
                              <TableCell>
                                {hasInconsistency ? (
                                  <span className="text-destructive font-medium">
                                    Detected
                                  </span>
                                ) : (
                                  <span className="text-green-500 font-medium">
                                    None
                                  </span>
                                )}
                              </TableCell>
                            </TableRow>
                          )
                        )}

                        {/* Visual Similarity */}
                        {result.visual_similarity.similar_segments.map(
                          (segment) => (
                            <TableRow key={`visual-${segment.segment}`}>
                              <TableCell>
                                Visual Similarity (Segment {segment.segment})
                              </TableCell>
                              <TableCell>
                                <span
                                  className={
                                    segment.similarity > 0.8
                                      ? "text-destructive font-medium"
                                      : segment.similarity > 0.6
                                      ? "text-yellow-600 font-medium dark:text-yellow-500"
                                      : "font-medium"
                                  }
                                >
                                  {(segment.similarity * 100).toFixed(2)}%
                                </span>
                              </TableCell>
                            </TableRow>
                          )
                        )}

                        {/* Semantic Similarity */}
                        {result.semantic_similarity.similar_segments.map(
                          (segment) => (
                            <TableRow key={`semantic-${segment.segment}`}>
                              <TableCell>
                                Semantic Similarity (Segment {segment.segment})
                              </TableCell>
                              <TableCell>
                                <span
                                  className={
                                    segment.similarity > 0.8
                                      ? "text-destructive font-medium"
                                      : segment.similarity > 0.6
                                      ? "text-yellow-600 font-medium dark:text-yellow-500"
                                      : "font-medium"
                                  }
                                >
                                  {(segment.similarity * 100).toFixed(2)}%
                                </span>
                              </TableCell>
                            </TableRow>
                          )
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </ScrollArea>
      )}
    </div>
  );
};

export default PDFAnalysisDashboard;
