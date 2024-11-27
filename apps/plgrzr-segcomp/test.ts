import { promises as fs } from "node:fs";
import path from "node:path";

interface ComparisonResponse {
	overall_similarity: number;
	region_similarities: number[];
}

interface ComparisonError {
	error: string;
}

class ImageComparisonTest {
	private baseUrl: string;

	constructor(baseUrl = "http://localhost:8000") {
		this.baseUrl = baseUrl;
	}

	/**
	 * Compares two images using the comparison service
	 */
	async compareImages(
		image1Path: string,
		image2Path: string,
	): Promise<ComparisonResponse> {
		try {
			// Create FormData and append files
			const formData = new FormData();

			// Read files and create blobs
			const [image1Buffer, image2Buffer] = await Promise.all([
				fs.readFile(image1Path),
				fs.readFile(image2Path),
			]);

			// Create Blobs from the file buffers
			const image1Blob = new Blob([image1Buffer]);
			const image2Blob = new Blob([image2Buffer]);

			// Append files to FormData with their original filenames
			formData.append("image1", image1Blob, path.basename(image1Path));
			formData.append("image2", image2Blob, path.basename(image2Path));

			// Send request to comparison service
			const response = await fetch(`${this.baseUrl}/compare`, {
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
	 * Tests comparison of multiple image pairs
	 */
	async runBatchTest(
		testCases: Array<{ image1: string; image2: string; description: string }>,
	): Promise<void> {
		console.log("Starting batch image comparison test...\n");

		for (const [index, testCase] of testCases.entries()) {
			console.log(`Test Case ${index + 1}: ${testCase.description}`);
			console.log("Comparing:");
			console.log(`  Image 1: ${path.basename(testCase.image1)}`);
			console.log(`  Image 2: ${path.basename(testCase.image2)}`);

			try {
				const startTime = Date.now();
				const result = await this.compareImages(
					testCase.image1,
					testCase.image2,
				);
				const duration = Date.now() - startTime;

				console.log("\nResults:");
				console.log(
					`  Overall Similarity: ${(result.overall_similarity * 100).toFixed(2)}%`,
				);
				console.log("  Region Similarities:");
				result.region_similarities.forEach((similarity, i) => {
					console.log(`    Region ${i + 1}: ${(similarity * 100).toFixed(2)}%`);
				});
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

// Example usage remains the same
async function main() {
	const tester = new ImageComparisonTest();

	const testCases = [
		{
			image1:
				"/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_output/prz_0bqz4iao/prz_0bqz4iao_1.jpg",
			image2:
				"/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_output/prz_0bqz4iao/prz_0bqz4iao_2.jpg",
			description: "Testing from same PDF",
		},
		{
			image1:
				"/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_output/prz_0bqz4iao/prz_0bqz4iao_1.jpg",
			image2:
				"/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_output/prz_0aczijb4/prz_0aczijb4_1.jpg",
			description: "Testing modified version of same image",
		},
	];

	await tester.runBatchTest(testCases);
}

main().catch((error) => {
	console.error("Test execution failed:", error);
});
