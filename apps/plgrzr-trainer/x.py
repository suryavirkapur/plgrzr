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


class MathPaperProcessor:
    def __init__(self):
        self.doctr_model = ocr_predictor(pretrained=True)
        self.text_reader = easyocr.Reader(["en"])
        self.latex_reader = LatexOCR()

    def preprocess_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if img is None:
            raise ValueError("Could not process image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        return denoised, img_rgb

    def segment_document(self, image):
        try:
            pil_image = Image.fromarray(image)
            temp_path = "temp_image.png"
            pil_image.save(temp_path)
            doc = DocumentFile.from_images(temp_path)
            result = self.doctr_model(doc)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            word_regions = []

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            geometry = word.geometry
                            h, w = image.shape[:2]
                            y1 = max(0, int(geometry[0][1] * h))
                            x1 = max(0, int(geometry[0][0] * w))
                            y2 = min(h, int(geometry[1][1] * h))
                            x2 = min(w, int(geometry[1][0] * w))

                            if y2 > y1 and x2 > x1:
                                word_img = image[y1:y2, x1:x2]
                                if word_img.size > 0:
                                    pad = 5
                                    word_img_padded = cv2.copyMakeBorder(
                                        word_img,
                                        pad,
                                        pad,
                                        pad,
                                        pad,
                                        cv2.BORDER_CONSTANT,
                                        value=[255, 255, 255],
                                    )
                                    word_regions.append(word_img_padded)

            return word_regions[:50] if word_regions else []
        except Exception as e:
            print(f"Error segmenting document: {str(e)}")
            return []


def example_word_regions(page_img):
    processor = MathPaperProcessor()
    _, img_rgb = processor.preprocess_image(page_img)
    regions = processor.segment_document(img_rgb)

    if not regions:
        h, w = page_img.shape[:2]
        regions = []
        for _ in range(10):
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 50)
            region = page_img[y : y + 50, x : x + 100]
            regions.append(region)

    return regions[:50]


class HandwritingFeatureExtractor(nn.Module):
    def __init__(self):
        super(HandwritingFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x).squeeze()


class HandwritingAnalyzer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.feature_extractor = HandwritingFeatureExtractor().to(device)
        self.feature_extractor.eval()
        self.n_components = 2
        self.feature_dim = 528

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def compare_pages(self, page1_vector, page2_vector):
        similarity = F.cosine_similarity(
            torch.tensor(page1_vector).unsqueeze(0),
            torch.tensor(page2_vector).unsqueeze(0),
        ).item()
        return similarity

    def extract_contour_features(self, word_img):
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros(8)

        # Handwriting-specific features
        features = []

        # Average stroke width
        stroke_widths = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                stroke_widths.append(
                    2 * area / perimeter
                )  # approximation of stroke width

        # Stroke characteristics
        features.extend(
            [
                np.mean(stroke_widths) if stroke_widths else 0,  # average stroke width
                np.std(stroke_widths)
                if stroke_widths
                else 0,  # stroke width consistency
                np.mean(
                    [cv2.arcLength(cnt, True) for cnt in contours]
                ),  # average stroke length
                np.std(
                    [cv2.arcLength(cnt, True) for cnt in contours]
                ),  # stroke length variation
                np.mean(
                    [cv2.contourArea(cnt) for cnt in contours]
                ),  # average stroke area
                np.std(
                    [cv2.contourArea(cnt) for cnt in contours]
                ),  # stroke area variation
                len(contours),  # number of strokes
                sum([cv2.contourArea(cnt) for cnt in contours])
                / (word_img.shape[0] * word_img.shape[1]),  # ink density
            ]
        )

        return np.array(features)

    def extract_pixel_distribution(self, word_img):
        # Convert to grayscale
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)

        # Get baseline and x-height
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = binary.shape

        # Horizontal projection for baseline analysis
        h_proj = np.sum(binary, axis=1) / w

        features = []
        # Height zones (capturing ascenders, x-height, baseline, descenders)
        n_zones = 4
        zone_h = h // n_zones
        for i in range(n_zones):
            zone = binary[i * zone_h : (i + 1) * zone_h, :]
            features.append(np.sum(zone) / (zone_h * w))

        # Vertical spacing analysis
        v_proj = np.sum(binary, axis=0) / h
        features.append(np.std(v_proj))  # letter spacing consistency

        return np.array(features)

    def estimate_slant(self, word_img):
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)

        if lines is None:
            return np.zeros(4)

        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Focus on vertical strokes for slant analysis
            if 45 <= angle <= 135:
                angles.append(angle - 90)

        if not angles:
            return np.zeros(4)

        # Slant characteristics
        return np.array(
            [
                np.mean(angles),  # average slant
                np.std(angles),  # slant consistency
                np.percentile(angles, 25),  # lower quartile
                np.percentile(angles, 75),  # upper quartile
            ]
        )

    def extract_letter_spacing(self, word_img):
        # New method for letter spacing analysis
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Vertical projection profile
        v_proj = np.sum(binary, axis=0)

        # Find letter spacing through zero crossings
        zero_crossings = np.where(np.diff(np.signbit(v_proj)))[0]
        if len(zero_crossings) > 1:
            spacings = np.diff(zero_crossings)
            return np.array([np.mean(spacings), np.std(spacings)])
        return np.array([0, 0])

    def process_page(self, word_regions):
        if not word_regions:
            return self.get_default_fisher_vector()

        all_features = []
        for word_img in word_regions:
            try:
                contour_feats = self.extract_contour_features(
                    word_img
                )  # stroke characteristics
                pixel_feats = self.extract_pixel_distribution(
                    word_img
                )  # height zones and spacing
                slant_feats = self.estimate_slant(word_img)  # consistency of slant
                spacing_feats = self.extract_letter_spacing(word_img)  # letter spacing

                # Combine all handwriting-specific features
                combined_features = np.concatenate(
                    [
                        contour_feats,  # stroke width, length, consistency
                        pixel_feats,  # zones and density
                        slant_feats,  # writing angle
                        spacing_feats,  # letter spacing
                    ]
                )

                all_features.append(combined_features)
            except Exception as e:
                print(f"Error processing word region: {e}")
                continue

        if not all_features:
            return self.get_default_fisher_vector()

        all_features = np.array(all_features)
        fisher_vector = self.compute_fisher_vector(all_features)

        return fisher_vector

    def extract_cnn_features(self, word_img):
        img_tensor = self.transform(word_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        return features.cpu().numpy()

    def get_default_fisher_vector(self):
        return np.zeros(self.feature_dim * self.n_components * 2)

    def compute_fisher_vector(self, features):
        if len(features) < self.n_components:
            return self.get_default_fisher_vector()

        gmm = GaussianMixture(n_components=self.n_components, covariance_type="diag")
        gmm.fit(features)

        fv = []
        posteriors = gmm.predict_proba(features)

        for k in range(self.n_components):
            diff = (features - gmm.means_[k]) / np.sqrt(gmm.covariances_[k])
            weighted_diff = posteriors[:, k : k + 1] * diff

            fv.extend(
                [
                    np.sum(weighted_diff, axis=0) / np.sqrt(gmm.weights_[k]),
                    np.sum(
                        posteriors[:, k : k + 1]
                        * (diff**2 - 1)
                        / np.sqrt(2 * gmm.weights_[k]),
                        axis=0,
                    ),
                ]
            )

        fv = np.concatenate(fv)
        fv = fv / np.linalg.norm(fv)

        return fv


def process_document(document_path, analyzer):
    document_path = Path(document_path)
    page_vectors = []

    for page_path in sorted(document_path.glob("*.jpg")):
        page_img = cv2.imread(str(page_path))
        word_regions = example_word_regions(page_img)
        page_vector = analyzer.process_page(word_regions)
        page_vectors.append(page_vector)

    return page_vectors


# if __name__ == "__main__":
#     analyzer = HandwritingAnalyzer()
#     doc1_path = "../../data/plgrzr_output/prz_0g3lbkdl"
#     doc2_path = "../../data/plgrzr_output/prz_0g3lbkdl"

#     doc1_vectors = process_document(doc1_path, analyzer)
#     doc2_vectors = process_document(doc2_path, analyzer)

#     similarities = []
#     for i, (vec1, vec2) in enumerate(zip(doc1_vectors, doc2_vectors)):
#         similarity = analyzer.compare_pages(vec1, vec2)
#         print(f"Similarity between page {i+1}: {similarity:.3f}")
#         similarities.append(similarity)

#     avg_similarity = np.mean(similarities) if similarities else 0
#     print(f"\nOverall document similarity: {avg_similarity:.3f}")
# if __name__ == "__main__":
#     analyzer = HandwritingAnalyzer()
#     doc_path = "../../data/plgrzr_output/prz_0g3lbkdl"

#     doc_vectors = process_document(doc_path, analyzer)

#     similarities = []
#     n_pages = len(doc_vectors)

#     for i in range(n_pages):
#         for j in range(i + 1, n_pages):
#             similarity = analyzer.compare_pages(doc_vectors[i], doc_vectors[j])
#             print(f"Similarity between page {i+1} and page {j+1}: {similarity:.3f}")
#             similarities.append(similarity)

#     avg_similarity = np.mean(similarities) if similarities else 0
#     print(f"\nAverage similarity between different pages: {avg_similarity:.3f}")
if __name__ == "__main__":
    analyzer = HandwritingAnalyzer()
    doc1_path = "../../data/plgrzr_output/prz_0g3lbkdl"
    doc2_path = "../../data/plgrzr_output/prz_0kz7m6jv"

    doc1_vectors = process_document(doc1_path, analyzer)
    doc2_vectors = process_document(doc2_path, analyzer)

    n_pages1 = len(doc1_vectors)
    n_pages2 = len(doc2_vectors)
    similarities = []

    print("Cross-document page similarities:")
    for i in range(n_pages1):
        for j in range(n_pages2):
            similarity = analyzer.compare_pages(doc1_vectors[i], doc2_vectors[j])
            print(
                f"Similarity between doc1_page{i+1} and doc2_page{j+1}: {similarity:.3f}"
            )
            similarities.append(similarity)

    avg_similarity = np.mean(similarities) if similarities else 0
    max_similarity = max(similarities) if similarities else 0
    min_similarity = min(similarities) if similarities else 0

    print("\nSimilarity Statistics:")
    print(f"Average similarity between documents: {avg_similarity:.3f}")
    print(f"Maximum similarity found: {max_similarity:.3f}")
    print(f"Minimum similarity found: {min_similarity:.3f}")
