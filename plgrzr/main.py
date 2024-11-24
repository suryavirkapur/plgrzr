import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from pathlib import Path
from sklearn.mixture import GaussianMixture


class HandwritingFeatureExtractor(nn.Module):
    def __init__(self):
        super(HandwritingFeatureExtractor, self).__init__()
        # Load pre-trained ResNet and remove the final classification layer
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze the pre-trained layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x).squeeze()


class HandwritingAnalyzer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.feature_extractor = HandwritingFeatureExtractor().to(device)
        self.feature_extractor.eval()

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

    def extract_contour_features(self, word_img):
        """Extract contour-based features from word image."""
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros(5)

        # Calculate contour features
        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

        features = [
            np.mean(areas),
            np.std(areas),
            np.mean(perimeters),
            np.std(perimeters),
            len(contours),
        ]
        return np.array(features)

    def extract_pixel_distribution(self, word_img):
        """Extract pixel distribution features."""
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Divide image into 3x3 grid and calculate pixel density in each cell
        cell_h, cell_w = h // 3, w // 3
        densities = []

        for i in range(3):
            for j in range(3):
                cell = gray[
                    i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]
                density = np.sum(cell < 128) / (cell_h * cell_w)
                densities.append(density)

        return np.array(densities)

    def estimate_slant(self, word_img):
        """Estimate writing slant and baseline."""
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)

        if lines is None:
            return np.zeros(2)

        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            if 45 <= angle <= 135:  # Consider only near-vertical lines
                angles.append(angle - 90)  # Convert to slant angle

        if not angles:
            return np.zeros(2)

        return np.array([np.mean(angles), np.std(angles)])

    def extract_cnn_features(self, word_img):
        """Extract features using pre-trained CNN."""
        img_tensor = self.transform(word_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        return features.cpu().numpy()

    def compute_fisher_vector(self, features, n_components=2):
        """Compute Fisher Vector representation from features."""
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
        gmm.fit(features)

        # Calculate statistics
        weights = gmm.weights_
        means = gmm.means_
        covars = gmm.covariances_

        # Compute Fisher Vector
        fv = []
        for feature in features:
            posteriors = gmm.predict_proba(feature.reshape(1, -1))
            for k in range(n_components):
                diff = (feature - means[k]) / np.sqrt(covars[k])
                fv.extend(
                    [
                        (posteriors[0, k] * diff / weights[k]).tolist(),
                        (
                            posteriors[0, k] * (diff**2 - 1) / (np.sqrt(2) * weights[k])
                        ).tolist(),
                    ]
                )

        return np.array(fv).flatten()

    def process_page(self, word_regions):
        """Process a single page with multiple word regions."""
        all_features = []

        for word_img in word_regions:
            # Extract all feature types
            contour_feats = self.extract_contour_features(word_img)
            pixel_feats = self.extract_pixel_distribution(word_img)
            slant_feats = self.estimate_slant(word_img)
            cnn_feats = self.extract_cnn_features(word_img)

            # Combine all features
            combined_features = np.concatenate(
                [contour_feats, pixel_feats, slant_feats, cnn_feats]
            )

            all_features.append(combined_features)

        # Convert to numpy array
        all_features = np.array(all_features)

        # Compute Fisher Vector representation for the page
        fisher_vector = self.compute_fisher_vector(all_features)

        return fisher_vector

    def compare_pages(self, page1_vector, page2_vector):
        """Compare two page-level Fisher Vectors."""
        # Compute cosine similarity between Fisher Vectors
        similarity = F.cosine_similarity(
            torch.tensor(page1_vector).unsqueeze(0),
            torch.tensor(page2_vector).unsqueeze(0),
        ).item()

        return similarity


def process_document(document_path, analyzer, word_regions_func):
    """Process an entire document."""
    document_path = Path(document_path)
    page_vectors = []

    for page_path in sorted(document_path.glob("*.jpg")):
        # Load page image
        page_img = cv2.imread(str(page_path))

        # Get word regions (this would be your manual or automated segmentation)
        word_regions = word_regions_func(page_img)

        # Process page and get Fisher Vector
        page_vector = analyzer.process_page(word_regions)
        page_vectors.append(page_vector)

    return page_vectors


# Example usage
def example_word_regions(page_img):
    """Example function to simulate word region extraction.
    In practice, this would be replaced with your manual lassoing or
    automated segmentation method."""
    # This is a placeholder - replace with actual word region extraction
    h, w = page_img.shape[:2]
    regions = []
    # Simulate 10 word regions per page
    for _ in range(10):
        x = np.random.randint(0, w - 100)
        y = np.random.randint(0, h - 50)
        region = page_img[y : y + 50, x : x + 100]
        regions.append(region)
    return regions


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = HandwritingAnalyzer()

    # Process documents
    doc1_vectors = process_document(
        "C:\\Users\\skpc\\Desktop\\Projekts\\temp\\plgrzr\\data\\plgrzr_output\\prz_0brwd85y",
        analyzer,
        example_word_regions,
    )
    doc2_vectors = process_document(
        "C:\\Users\\skpc\\Desktop\\Projekts\\temp\\plgrzr\\data\\plgrzr_output\\prz_0nom6qx3",
        analyzer,
        example_word_regions,
    )

    # Compare pages within and between documents
    for i, vec1 in enumerate(doc1_vectors):
        for j, vec2 in enumerate(doc2_vectors):
            similarity = analyzer.compare_pages(vec1, vec2)
            print(
                f"Similarity between doc1_page{i+1} and doc2_page{j+1}: {similarity:.3f}"
            )
