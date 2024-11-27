from robyn import Robyn, Request, Response, jsonify, ALLOW_CORS
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity


app = Robyn(__file__)
ALLOW_CORS(app, origins=["*"])

# Initialize model and device globally for reuse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
num_classes = 10
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)


# Initialize feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(*list(model.classifier[:4]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


feature_extractor = FeatureExtractor(model)
feature_extractor.eval()
feature_extractor.to(device)

# Image preprocessing
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generate_random_regions(
    image_shape, num_regions=10, region_shape="rectangular", seed=42
):
    if seed is not None:
        random.seed(seed)
    regions = []
    height, width = image_shape[:2]
    for _ in range(num_regions):
        if region_shape == "rectangular":
            x1 = random.randint(0, width - 50)
            y1 = random.randint(0, height - 50)
            x2 = random.randint(x1 + 50, min(x1 + 200, width))
            y2 = random.randint(y1 + 50, min(y1 + 200, height))
            regions.append((x1, y1, x2, y2))
    return regions


def extract_regions(image, regions):
    region_images = []
    for region in regions:
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        region_images.append(roi)
    return region_images


def extract_features(region_image, model, device):
    if region_image.size == 0:
        return np.zeros((4096,))
    try:
        input_tensor = preprocess(region_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(input_batch)
        features = features.cpu().numpy().flatten()
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((4096,))


def process_image_file(image_data, regions, model, device):
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None

    region_images = extract_regions(image, regions)
    features_list = []
    for region_image in region_images:
        features = extract_features(region_image, model, device)
        features_list.append(features)
    return features_list


def compute_similarity(features_list1, features_list2):
    similarities = []
    for feat1, feat2 in zip(features_list1, features_list2):
        sim = cosine_similarity([feat1], [feat2])[0][0]
        similarities.append(sim)
    return similarities


@app.post("/compare")
async def compare_images(request: Request) -> Response:
    try:
        # Get multipart form data
        files = request.files
        file_names = list(files.keys())  # Convert dict_keys to list

        if len(file_names) != 2:
            return jsonify({"error": "Exactly two image files are required"})

        # Read image data from files
        image1_data = files[file_names[0]]
        image2_data = files[file_names[1]]

        # Process first image to get shape for regions
        nparr = np.frombuffer(image1_data, np.uint8)
        image1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image1 is None:
            return jsonify({"error": "Invalid image1 data"})

        # Generate regions based on first image
        regions = generate_random_regions(image1.shape)

        # Process both images
        features_list1 = process_image_file(
            image1_data, regions, feature_extractor, device
        )
        features_list2 = process_image_file(
            image2_data, regions, feature_extractor, device
        )

        if features_list1 is None or features_list2 is None:
            return jsonify({"error": "Feature extraction failed"})

        # Compute similarity
        similarities = compute_similarity(features_list1, features_list2)
        overall_similarity = float(np.mean(similarities))
        print(f"Overall similarity: {overall_similarity:.4f}")

        return jsonify(
            {
                "overall_similarity": overall_similarity,
                "region_similarities": [float(s) for s in similarities],
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.start(port=8000)
