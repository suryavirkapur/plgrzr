import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from pathlib import Path
from sklearn.mixture import GaussianMixture
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import easyocr
from pix2tex.cli import LatexOCR
import os


class ImagePreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image_path):
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path.copy()

        if image is None:
            raise ValueError("Invalid image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        enhanced = self.clahe.apply(denoised)
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)

        edges = cv2.Canny(bilateral, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        return dilated, bilateral, image


class HandwritingFeatureExtractor(nn.Module):
    def __init__(self):
        super(HandwritingFeatureExtractor, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.adaptation = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.adaptation(x)
        return F.normalize(x, p=2, dim=1)


class TextDetector:
    def __init__(self):
        self.ocr = ocr_predictor(pretrained=True)
        self.text_reader = easyocr.Reader(["en"])
        self.latex_reader = LatexOCR()

    def detect_text_regions(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        temp_path = "temp.png"
        pil_image.save(temp_path)

        doc = DocumentFile.from_images(temp_path)
        result = self.ocr(doc)

        os.remove(temp_path)

        regions = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        bbox = word.geometry
                        h, w = image.shape[:2]
                        y1 = max(0, int(bbox[0][1] * h) - 10)
                        x1 = max(0, int(bbox[0][0] * w) - 10)
                        y2 = min(h, int(bbox[1][1] * h) + 10)
                        x2 = min(w, int(bbox[1][0] * w) + 10)

                        if y2 > y1 and x2 > x1:
                            region = image[y1:y2, x1:x2]
                            if region.size > 0:
                                pad = 10
                                padded = cv2.copyMakeBorder(
                                    region, pad, pad, pad, pad, cv2.BORDER_REPLICATE
                                )
                                regions.append(padded)

        return regions[:100] if regions else []


class FeatureExtractor:
    def extract_shape_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        features = []

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            features.extend(
                [
                    area / (image.shape[0] * image.shape[1]),
                    perimeter / (2 * (image.shape[0] + image.shape[1])),
                    cv2.moments(binary)["mu20"],
                    cv2.moments(binary)["mu02"],
                    cv2.moments(binary)["mu11"],
                ]
            )

            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                features.append(solidity)

        return np.array(features)

    def extract_texture_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []

        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            features.extend(
                [
                    np.mean(filtered),
                    np.std(filtered),
                    np.max(filtered),
                    np.min(filtered),
                ]
            )

        glcm = self.compute_glcm(gray)
        features.extend(
            [
                self.compute_contrast(glcm),
                self.compute_homogeneity(glcm),
                self.compute_energy(glcm),
                self.compute_correlation(glcm),
            ]
        )

        return np.array(features)

    def compute_glcm(self, image):
        h, w = image.shape
        glcm = np.zeros((256, 256))

        for i in range(h - 1):
            for j in range(w - 1):
                i_val = image[i, j]
                j_val = image[i, j + 1]
                glcm[i_val, j_val] += 1

        glcm = glcm / glcm.sum()
        return glcm

    def compute_contrast(self, glcm):
        h, w = glcm.shape
        contrast = 0
        for i in range(h):
            for j in range(w):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast

    def compute_homogeneity(self, glcm):
        h, w = glcm.shape
        homogeneity = 0
        for i in range(h):
            for j in range(w):
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        return homogeneity

    def compute_energy(self, glcm):
        return np.sum(glcm**2)

    def compute_correlation(self, glcm):
        h, w = glcm.shape
        i_indices = np.arange(h).reshape(-1, 1)
        j_indices = np.arange(w).reshape(1, -1)

        i_mean = np.sum(i_indices * np.sum(glcm, axis=1))
        j_mean = np.sum(j_indices * np.sum(glcm, axis=0))

        i_var = np.sum(((i_indices - i_mean) ** 2) * np.sum(glcm, axis=1))
        j_var = np.sum(((j_indices - j_mean) ** 2) * np.sum(glcm, axis=0))

        correlation = 0
        for i in range(h):
            for j in range(w):
                correlation += (i - i_mean) * (j - j_mean) * glcm[i, j]

        if i_var * j_var > 0:
            correlation /= np.sqrt(i_var * j_var)

        return correlation


class HandwritingAnalyzer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = HandwritingFeatureExtractor().to(device)
        self.feature_extractor.eval()
        self.text_detector = TextDetector()
        self.feature_computer = FeatureExtractor()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def compare_documents(self, doc1_path, doc2_path):
        try:
            doc1_features = self.process_document(doc1_path)
            doc2_features = self.process_document(doc2_path)

            if not doc1_features or not doc2_features:
                print("No features extracted from documents")
                return {
                    "individual_similarities": [],
                    "average_similarity": 0.0,
                    "max_similarity": 0.0,
                    "min_similarity": 0.0,
                    "error": "No features extracted from documents",
                }

            # Make sure both documents have the same number of pages
            min_pages = min(len(doc1_features), len(doc2_features))
            doc1_features = doc1_features[:min_pages]
            doc2_features = doc2_features[:min_pages]

            similarities = []
            for feat1, feat2 in zip(doc1_features, doc2_features):
                # Check if features are valid (not all zeros)
                if np.any(feat1) and np.any(feat2):
                    sim = F.cosine_similarity(
                        torch.tensor(feat1).unsqueeze(0),
                        torch.tensor(feat2).unsqueeze(0),
                    ).item()
                    similarities.append(sim)

            if not similarities:
                return {
                    "individual_similarities": [],
                    "average_similarity": 0.0,
                    "max_similarity": 0.0,
                    "min_similarity": 0.0,
                    "error": "No valid similarities computed",
                }

            return {
                "individual_similarities": similarities,
                "average_similarity": np.mean(similarities),
                "max_similarity": np.max(similarities),
                "min_similarity": np.min(similarities),
            }

        except Exception as e:
            print(f"Error in document comparison: {str(e)}")
            return {
                "individual_similarities": [],
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "error": str(e),
            }

    def process_document(self, doc_path):
        try:
            doc_path = Path(doc_path)
            if not doc_path.exists():
                print(f"Document path does not exist: {doc_path}")
                return []

            features = []
            image_files = list(doc_path.glob("*.jpg")) + list(doc_path.glob("*.png"))

            if not image_files:
                print(f"No image files found in {doc_path}")
                return []

            for img_path in sorted(image_files):
                try:
                    page_features = self.process_image(str(img_path))
                    if np.any(page_features):  # Check if features are not all zeros
                        features.append(page_features)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            return features

        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")
            return []

    def process_image(self, image_path):
        try:
            dilated, bilateral, original = self.preprocessor.preprocess(image_path)
            regions = self.text_detector.detect_text_regions(original)

            if not regions:
                print(f"No text regions detected in {image_path}")
                return np.zeros(1024)

            features_list = []
            for region in regions:
                try:
                    deep_features = self.extract_deep_features(region)
                    shape_features = self.feature_computer.extract_shape_features(
                        region
                    )
                    texture_features = self.feature_computer.extract_texture_features(
                        region
                    )

                    combined = np.concatenate(
                        [deep_features, shape_features, texture_features]
                    )

                    if np.any(combined):  # Check if features are not all zeros
                        features_list.append(combined)
                except Exception as e:
                    print(f"Error processing region: {e}")
                    continue

            if not features_list:
                print(f"No valid features extracted from {image_path}")
                return np.zeros(1024)

            features_array = np.array(features_list)
            return self.compute_fisher_vector(features_array)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(1024)

    def extract_deep_features(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(tensor)
        return features.cpu().numpy().flatten()

    def compute_fisher_vector(self, features, n_components=4):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=100,
        )
        gmm.fit(features)

        posteriors = gmm.predict_proba(features)

        fv_mu = np.zeros((gmm.n_components, features.shape[1]))
        fv_sigma = np.zeros((gmm.n_components, features.shape[1]))

        for k in range(gmm.n_components):
            diff = (features - gmm.means_[k]) / np.sqrt(gmm.covariances_[k])
            weighted_diff = posteriors[:, k : k + 1] * diff

            fv_mu[k] = np.sum(weighted_diff, axis=0) / np.sqrt(gmm.weights_[k])
            fv_sigma[k] = np.sum(
                posteriors[:, k : k + 1] * (diff**2 - 1) / np.sqrt(2 * gmm.weights_[k]),
                axis=0,
            )

        fv = np.concatenate([fv_mu.ravel(), fv_sigma.ravel()])
        fv = np.sign(fv) * np.sqrt(np.abs(fv))
        fv = fv / (np.linalg.norm(fv) + 1e-4)

        return fv


def main():
    analyzer = HandwritingAnalyzer()

    doc1_path = "../../data/plgrzr_output/prz_0g3lbkdl"
    doc2_path = "../../data/plgrzr_output/prz_0kz7m6jv"

    results = analyzer.compare_documents(doc1_path, doc2_path)

    print("\nDocument Comparison Results:")
    print(f"Average Similarity: {results['average_similarity']:.3f}")
    print(f"Maximum Similarity: {results['max_similarity']:.3f}")
    print(f"Minimum Similarity: {results['min_similarity']:.3f}")

    print("\nPage-by-Page Similarities:")
    for i, sim in enumerate(results["individual_similarities"], 1):
        print(f"Page {i}: {sim:.3f}")


if __name__ == "__main__":
    main()
