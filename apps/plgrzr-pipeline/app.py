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

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if different
)

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

# Initialize the Sentence Transformer model for semantic analysis
print("Loading Sentence Transformer model...")
model_name = "all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name, device=device)


# Function to generate fixed grid segments
def generate_grid_segments(img, rows=3, cols=3):
    """
    Generate grid segments from an image.

    Args:
        img (PIL.Image): The input image.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        List of PIL.Image segments.
    """
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
ROWS = 3  # Adjust as needed
COLS = 3  # Adjust as needed


# Endpoint to compare two PDFs
@app.post("/compare_pdfs")
async def compare_pdfs(request: Request) -> Response:
    try:
        # Get multipart form data
        files = request.files
        file_names = list(files.keys())

        if len(file_names) != 2:
            return jsonify({"error": "Exactly two PDF files are required"})

        # Read PDF data from files
        pdf1_data = files[file_names[0]]
        pdf2_data = files[file_names[1]]

        # Process both PDFs
        results = process_pdfs(pdf1_data, pdf2_data)

        return jsonify(results)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


# Function to process PDFs
def process_pdfs(pdf1_data, pdf2_data):
    # Dictionaries to store data
    pdf_texts = {}  # Text content per PDF per segment
    pdf_features = {}  # Image features per PDF per segment

    pdf_datas = [pdf1_data, pdf2_data]
    pdf_names = ["PDF 1", "PDF 2"]

    for pdf_name, pdf_data in zip(pdf_names, pdf_datas):
        # Convert PDF to images
        images = convert_from_bytes(pdf_data)
        pdf_texts[pdf_name] = []
        pdf_features[pdf_name] = []
        for img in images:
            segments = generate_grid_segments(img, rows=ROWS, cols=COLS)
            for segment in segments:
                # Extract text using OCR
                text = pytesseract.image_to_string(segment, lang="eng")
                pdf_texts[pdf_name].append(text)
                # Extract image features
                try:
                    # Transform segment
                    segment_tensor = transform(segment).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(segment_tensor).cpu().numpy().flatten()
                    pdf_features[pdf_name].append(features)
                except Exception as e:
                    print(f"Error extracting features from segment in {pdf_name}: {e}")
                    pdf_features[pdf_name].append(np.zeros(25088))  # VGG16 feature size
                    continue

    # Anomaly Detection in Handwriting
    anomalies = {}
    for pdf_name, features_list in pdf_features.items():
        anomaly = detect_handwriting_anomalies(features_list)
        anomalies[pdf_name] = anomaly

    # Visual Similarity Analysis between corresponding segments
    len(pdf_names)
    num_segments = ROWS * COLS

    # Initialize similarity matrices for segments
    segment_similarity_matrix = np.zeros(num_segments)
    for seg_idx in range(num_segments):
        feat1 = pdf_features[pdf_names[0]][seg_idx].reshape(1, -1)
        feat2 = pdf_features[pdf_names[1]][seg_idx].reshape(1, -1)
        sim = cosine_similarity(feat1, feat2)[0][0]
        segment_similarity_matrix[seg_idx] = sim

    # Text Extraction and Semantic Analysis between corresponding segments
    segment_text_similarity_matrix = np.zeros(num_segments)
    for seg_idx in range(num_segments):
        emb1 = sentence_model.encode(pdf_texts[pdf_names[0]][seg_idx])
        emb2 = sentence_model.encode(pdf_texts[pdf_names[1]][seg_idx])
        sim = cosine_similarity([emb1], [emb2])[0][0]
        segment_text_similarity_matrix[seg_idx] = sim

    # Textual Inconsistency Detection
    textual_inconsistencies = {}
    for pdf_name, texts in pdf_texts.items():
        inconsistency = detect_textual_inconsistencies(texts)
        textual_inconsistencies[pdf_name] = inconsistency

    # Prepare the results
    results = {
        "anomalies": anomalies,
        "textual_inconsistencies": textual_inconsistencies,
        "visual_similarity": {},
        "semantic_similarity": {},
    }

    # Thresholds for similarity reporting
    visual_threshold = 0.8
    semantic_threshold = 0.8

    # Visual Similarity Results
    visual_similar_segments = []
    for seg_idx in range(num_segments):
        sim = segment_similarity_matrix[seg_idx]
        if sim >= visual_threshold:
            visual_similar_segments.append({"segment": seg_idx + 1, "similarity": sim})
    results["visual_similarity"]["similar_segments"] = visual_similar_segments

    # Semantic Similarity Results
    semantic_similar_segments = []
    for seg_idx in range(num_segments):
        sim = segment_text_similarity_matrix[seg_idx]
        if sim >= semantic_threshold:
            semantic_similar_segments.append(
                {"segment": seg_idx + 1, "similarity": sim}
            )
    results["semantic_similarity"]["similar_segments"] = semantic_similar_segments

    return results


# Function for handwriting anomaly detection
def detect_handwriting_anomalies(features_list):
    if len(features_list) < 2:
        return False  # Not enough data to detect anomalies
    # Stack features into an array
    X = np.vstack(features_list)
    # Normalize features
    X_normalized = normalize(X)
    # Determine the number of components for PCA
    n_samples, n_features = X_normalized.shape
    n_components = min(10, n_samples, n_features)
    if n_components < 2:
        return False  # Not enough components to perform PCA
    # Reduce dimensionality for clustering
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_normalized)
    # Cluster the features
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced)
    labels = kmeans.labels_
    # Check if there is more than one cluster
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        return True  # Anomaly detected
    else:
        return False


# Function for textual inconsistency detection
def detect_textual_inconsistencies(texts):
    if len(texts) < 2:
        return False  # Not enough data to detect inconsistencies
    # Compute embeddings
    embeddings = sentence_model.encode(texts)
    # Cluster embeddings
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    # Check if there is more than one cluster
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        return True  # Inconsistency detected
    else:
        return False


# Endpoint to detect anomalies in a single PDF
@app.post("/detect_anomalies")
async def detect_anomalies(request: Request) -> Response:
    try:
        # Get multipart form data
        files = request.files
        file_names = list(files.keys())

        if len(file_names) != 1:
            return jsonify({"error": "Exactly one PDF file is required"})

        # Read PDF data from file
        pdf_data = files[file_names[0]]
        pdf_name = "Uploaded PDF"

        # Process the PDF
        pdf_texts = []
        pdf_features = []

        # Convert PDF to images
        images = convert_from_bytes(pdf_data)
        for img in images:
            segments = generate_grid_segments(img, rows=ROWS, cols=COLS)
            for segment in segments:
                # Extract text using OCR
                text = pytesseract.image_to_string(segment, lang="eng")
                pdf_texts.append(text)
                # Extract image features
                try:
                    # Transform segment
                    segment_tensor = transform(segment).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(segment_tensor).cpu().numpy().flatten()
                    pdf_features.append(features)
                except Exception as e:
                    print(f"Error extracting features from segment in {pdf_name}: {e}")
                    pdf_features.append(np.zeros(25088))  # VGG16 feature size
                    continue

        # Anomaly Detection in Handwriting
        anomaly = detect_handwriting_anomalies(pdf_features)

        # Textual Inconsistency Detection
        inconsistency = detect_textual_inconsistencies(pdf_texts)

        # Prepare the results
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
