import { promises as fs } from "node:fs";
import path from "node:path";

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

interface AnomalyResponse {
	anomaly_detected: boolean;
	textual_inconsistency_detected: boolean;
}

interface ComparisonError {
	error: string;
}

class PDFAnalysisTest {
	private baseUrl: string;

	constructor(baseUrl = "http://localhost:8001") {
		this.baseUrl = baseUrl;
	}

	/**
	 * Compares two PDFs using the comparison service
	 */
	async comparePDFs(
		pdf1Path: string,
		pdf2Path: string,
	): Promise<ComparisonResponse> {
		try {
			// Create FormData and append files
			const formData = new FormData();

			// Read files and create blobs
			const [pdf1Buffer, pdf2Buffer] = await Promise.all([
				fs.readFile(pdf1Path),
				fs.readFile(pdf2Path),
			]);

			// Create Blobs from the file buffers
			const pdf1Blob = new Blob([pdf1Buffer]);
			const pdf2Blob = new Blob([pdf2Buffer]);

			// Append files to FormData with their original filenames
			formData.append("pdf1", pdf1Blob, path.basename(pdf1Path));
			formData.append("pdf2", pdf2Blob, path.basename(pdf2Path));

			// Send request to comparison service
			const response = await fetch(`${this.baseUrl}/compare_pdfs`, {
				method: "POST",
				body: formData,
			});

			if (!response.ok) {
				const errorData = (await response.json()) as ComparisonError;
				throw new Error(
					errorData.error || `HTTP error! status: ${response.status}`,
				);
			}

			const data = (await response.json()) as
				| ComparisonResponse
				| ComparisonError;

			// Check for error response
			if ("error" in data) {
				throw new Error(data.error);
			}

			return data as ComparisonResponse;
		} catch (error) {
			throw error instanceof Error
				? error
				: new Error("An unknown error occurred");
		}
	}

	/**
	 * Detects anomalies in a single PDF
	 */
	async detectAnomalies(pdfPath: string): Promise<AnomalyResponse> {
		try {
			// Create FormData and append file
			const formData = new FormData();

			// Read file and create blob
			const pdfBuffer = await fs.readFile(pdfPath);
			const pdfBlob = new Blob([pdfBuffer]);

			// Append file to FormData with original filename
			formData.append(path.basename(pdfPath), pdfBlob);

			// Send request to anomaly detection service
			const response = await fetch(`${this.baseUrl}/detect_anomalies`, {
				method: "POST",
				body: formData,
			});

			if (!response.ok) {
				const errorData = (await response.json()) as ComparisonError;
				throw new Error(
					errorData.error || `HTTP error! status: ${response.status}`,
				);
			}

			const data = (await response.json()) as AnomalyResponse | ComparisonError;

			// Check for error response
			if ("error" in data) {
				throw new Error(data.error);
			}

			return data as AnomalyResponse;
		} catch (error) {
			throw error instanceof Error
				? error
				: new Error("An unknown error occurred");
		}
	}

	/**
	 * Tests comparison of multiple PDF pairs
	 */
	async runComparisonBatchTest(
		testCases: Array<{ pdf1: string; pdf2: string; description: string }>,
	): Promise<void> {
		console.log("Starting batch PDF comparison test...\n");

		for (const [index, testCase] of testCases.entries()) {
			console.log(`Test Case ${index + 1}: ${testCase.description}`);
			console.log("Comparing:");
			console.log(`  PDF 1: ${path.basename(testCase.pdf1)}`);
			console.log(`  PDF 2: ${path.basename(testCase.pdf2)}`);

			try {
				const startTime = Date.now();
				const result = await this.comparePDFs(testCase.pdf1, testCase.pdf2);
				const duration = Date.now() - startTime;

				console.log("\nResults:");
				console.log("Anomalies Detected:");
				Object.entries(result.anomalies).forEach(([pdf, hasAnomaly]) => {
					console.log(`  ${pdf}: ${hasAnomaly}`);
				});

				console.log("\nTextual Inconsistencies:");
				Object.entries(result.textual_inconsistencies).forEach(
					([pdf, hasInconsistency]) => {
						console.log(`  ${pdf}: ${hasInconsistency}`);
					},
				);

				console.log("\nVisually Similar Segments:");
				result.visual_similarity.similar_segments.forEach((segment) => {
					console.log(
						`  Segment ${segment.segment}: ${(segment.similarity * 100).toFixed(2)}%`,
					);
				});

				console.log("\nSemantically Similar Segments:");
				result.semantic_similarity.similar_segments.forEach((segment) => {
					console.log(
						`  Segment ${segment.segment}: ${(segment.similarity * 100).toFixed(2)}%`,
					);
				});

				console.log(`\nProcessing Time: ${duration}ms`);
			} catch (error) {
				console.error(
					`\nError: ${error instanceof Error ? error.message : "Unknown error"}`,
				);
			}

			console.log(`\n${"-".repeat(50)}\n`);
		}
	}

	/**
	 * Tests anomaly detection on multiple individual PDFs
	 */
	async runAnomalyBatchTest(
		testCases: Array<{ pdf: string; description: string }>,
	): Promise<void> {
		console.log("Starting batch anomaly detection test...\n");

		for (const [index, testCase] of testCases.entries()) {
			console.log(`Test Case ${index + 1}: ${testCase.description}`);
			console.log(`PDF: ${path.basename(testCase.pdf)}`);

			try {
				const startTime = Date.now();
				const result = await this.detectAnomalies(testCase.pdf);
				const duration = Date.now() - startTime;

				console.log("\nResults:");
				console.log(`  Anomaly Detected: ${result.anomaly_detected}`);
				console.log(
					`  Textual Inconsistency: ${result.textual_inconsistency_detected}`,
				);
				console.log(`  Processing Time: ${duration}ms`);
			} catch (error) {
				console.error(
					`\nError: ${error instanceof Error ? error.message : "Unknown error"}`,
				);
			}

			console.log(`\n${"-".repeat(50)}\n`);
		}
	}
}

// Example usage
async function main() {
	const tester = new PDFAnalysisTest();

	// Test PDF comparisons
	const comparisonTestCases = [
		{
			pdf1: "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset/prz_0aczijb4.pdf",
			pdf2: "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset/prz_0aczijb4_copy.pdf",
			description: "Same student.",
		},
		{
			pdf1: "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset/prz_0f1t2uoz.pdf",
			pdf2: "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset/prz_0op8cqv9.pdf",
			description: "Different Students.",
		},
	];

	// Test anomaly detection
	const anomalyTestCases = [
		{
			pdf: "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset/prz_0crona48.pdf",
			description: "PDF 1 Uniform",
		},
		{
			pdf: "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset/prz_6zhsyckk.pdf",
			description: "Checking control document",
		},
	];

	console.log("Running PDF Comparison Tests...");
	await tester.runComparisonBatchTest(comparisonTestCases);

	// console.log("\nRunning Anomaly Detection Tests...");
	// await tester.runAnomalyBatchTest(anomalyTestCases);
}

main().catch((error) => {
	console.error("Test execution failed:", error);
});
