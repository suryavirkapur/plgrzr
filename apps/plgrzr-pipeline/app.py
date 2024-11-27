from robyn import Robyn, Request, Response, jsonify, ALLOW_CORS
import pytesseract
import numpy as np
from pdf2image import convert_from_bytes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import nltk
from sentence_transformers import SentenceTransformer
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Initialize the Robyn app
app = Robyn(__file__)
ALLOW_CORS(app, origins=["*"])

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the CNN model for feature extraction
print("Loading VGG16 model...")
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()

# Remove the last classification layer
model = torch.nn.Sequential(*list(vgg16.children())[:-1]).to(device)

# Image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Initialize the Sentence Transformer model
print("Loading Sentence Transformer model...")
model_name = "all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name, device=device)


# Function to generate fixed grid segments
def generate_grid_segments(img, rows=3, cols=3):
    width, height = img.size
    segment_width = width // cols
    segment_height = height // rows
    segments = []
    for i in range(rows):
        for j in range(cols):
            left = j * segment_width
            upper = i * segment_height
            right = (j + 1) * segment_width if j != cols - 1 else width
            lower = (i + 1) * segment_height if i != rows - 1 else height
            segment = img.crop((left, upper, right, lower))
            segments.append(segment)
    return segments


# Number of rows and columns in the grid
ROWS = 3
COLS = 3


@app.post("/compare_pdfs")
async def compare_pdfs(request: Request) -> Response:
    try:
        files = request.files
        file_names = list(files.keys())
        print(file_names)
        if len(file_names) != 2:
            return jsonify({"error": "Exactly two PDF files are required"})

        pdf1_data = files[file_names[0]]
        pdf2_data = files[file_names[1]]
        results = process_pdfs(pdf1_data, pdf2_data)

        return jsonify(results)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


def process_pdfs(pdf1_data, pdf2_data):
    pdf_texts = {}
    pdf_features = {}
    pdf_datas = [pdf1_data, pdf2_data]
    pdf_names = ["PDF 1", "PDF 2"]

    for pdf_name, pdf_data in zip(pdf_names, pdf_datas):
        images = convert_from_bytes(pdf_data)
        pdf_texts[pdf_name] = []
        pdf_features[pdf_name] = []

        for img in images:
            segments = generate_grid_segments(img, rows=ROWS, cols=COLS)
            for segment in segments:
                text = pytesseract.image_to_string(segment, lang="eng")
                pdf_texts[pdf_name].append(text)

                try:
                    segment_tensor = transform(segment).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(segment_tensor).cpu().numpy().flatten()
                    pdf_features[pdf_name].append(features)
                except Exception as e:
                    print(f"Error extracting features from segment in {pdf_name}: {e}")
                    pdf_features[pdf_name].append(np.zeros(25088))
                    continue

    # Anomaly Detection
    anomalies = {}
    for pdf_name, features_list in pdf_features.items():
        anomaly = detect_handwriting_anomalies(features_list)
        anomalies[pdf_name] = bool(anomaly)

    num_segments = ROWS * COLS
    segment_similarity_matrix = np.zeros(num_segments)
    segment_text_similarity_matrix = np.zeros(num_segments)

    # Calculate similarities
    for seg_idx in range(num_segments):
        # Visual similarity
        feat1 = pdf_features[pdf_names[0]][seg_idx].reshape(1, -1)
        feat2 = pdf_features[pdf_names[1]][seg_idx].reshape(1, -1)
        sim = cosine_similarity(feat1, feat2)[0][0]
        segment_similarity_matrix[seg_idx] = float(sim)

        # Text similarity
        emb1 = sentence_model.encode(pdf_texts[pdf_names[0]][seg_idx])
        emb2 = sentence_model.encode(pdf_texts[pdf_names[1]][seg_idx])
        sim = cosine_similarity([emb1], [emb2])[0][0]
        segment_text_similarity_matrix[seg_idx] = float(sim)

    # Textual Inconsistency Detection
    textual_inconsistencies = {}
    for pdf_name, texts in pdf_texts.items():
        inconsistency = detect_textual_inconsistencies(texts)
        textual_inconsistencies[pdf_name] = bool(inconsistency)

    # Prepare results
    visual_threshold = 0.8
    semantic_threshold = 0.8

    visual_similar_segments = [
        {"segment": seg_idx + 1, "similarity": float(sim)}
        for seg_idx, sim in enumerate(segment_similarity_matrix)
        if sim >= visual_threshold
    ]

    semantic_similar_segments = [
        {"segment": seg_idx + 1, "similarity": float(sim)}
        for seg_idx, sim in enumerate(segment_text_similarity_matrix)
        if sim >= semantic_threshold
    ]

    results = {
        "anomalies": anomalies,
        "textual_inconsistencies": textual_inconsistencies,
        "visual_similarity": {"similar_segments": visual_similar_segments},
        "semantic_similarity": {"similar_segments": semantic_similar_segments},
    }

    return results


def detect_handwriting_anomalies(features_list):
    if len(features_list) < 2:
        return False
    X = np.vstack(features_list)
    X_normalized = normalize(X)
    n_samples, n_features = X_normalized.shape
    n_components = min(10, n_samples, n_features)
    if n_components < 2:
        return False
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_normalized)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced)
    labels = kmeans.labels_
    return bool(len(np.unique(labels)) > 1)


def detect_textual_inconsistencies(texts):
    if len(texts) < 2:
        return False
    embeddings = sentence_model.encode(texts)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    return bool(len(np.unique(labels)) > 1)


@app.post("/detect_anomalies")
async def detect_anomalies(request: Request) -> Response:
    try:
        files = request.files
        file_names = list(files.keys())

        if len(file_names) != 1:
            return jsonify({"error": "Exactly one PDF file is required"})

        pdf_data = files[file_names[0]]
        pdf_texts = []
        pdf_features = []

        images = convert_from_bytes(pdf_data)
        for img in images:
            segments = generate_grid_segments(img, rows=ROWS, cols=COLS)
            for segment in segments:
                text = pytesseract.image_to_string(segment, lang="eng")
                pdf_texts.append(text)

                try:
                    segment_tensor = transform(segment).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(segment_tensor).cpu().numpy().flatten()
                    pdf_features.append(features)
                except Exception as e:
                    print(f"Error extracting features from segment: {e}")
                    pdf_features.append(np.zeros(25088))
                    continue

        anomaly = bool(detect_handwriting_anomalies(pdf_features))
        inconsistency = bool(detect_textual_inconsistencies(pdf_texts))

        results = {
            "anomaly_detected": anomaly,
            "textual_inconsistency_detected": inconsistency,
        }

        return jsonify(results)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.start(port=8001)
