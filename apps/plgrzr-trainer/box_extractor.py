import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import easyocr
from pix2tex.cli import LatexOCR
import os

class MathPaperProcessor:
    def __init__(self):
        self.doctr_model = ocr_predictor(pretrained=True)
        self.text_reader = easyocr.Reader(['en'])
        self.latex_reader = LatexOCR()
        
    def preprocess_image(self, image_path):
        """Preprocess the scanned image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            return denoised, img_rgb 
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def numpy_to_pil(self, numpy_image):
        """Convert numpy array to PIL Image"""
        if len(numpy_image.shape) == 2: 
            return Image.fromarray(numpy_image)
        else: 
            return Image.fromarray(numpy_image).convert('RGB')

    def segment_document(self, image):
        """Segment document into regions"""
        try:
            pil_image = self.numpy_to_pil(image)
            temp_path = "temp_image.png"
            pil_image.save(temp_path)
            doc = DocumentFile.from_images(temp_path)
            result = self.doctr_model(doc)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

            regions = {
                'header': [],
                'question': [],
                'solution': [],
                'equations': []
            }
            
            result.show()

            for page in result.pages:
                for block in page.blocks:
                    geometry = block.geometry
                    y_center = (geometry[0][1] + geometry[1][1]) / 2
                    block_text = ' '.join([word.value for line in block.lines for word in line.words])

                    if y_center < 0.2:  
                        regions['header'].append({
                            'text': block_text,
                            'geometry': geometry
                        })
                    elif any(math_term in block_text.lower() for math_term in 
                            ['matrix', '=', 'sum', 'integral', 'equation']):
                        regions['equations'].append({
                            'text': block_text,
                            'geometry': geometry
                        })
                    elif y_center < 0.4:
                        regions['question'].append({
                            'text': block_text,
                            'geometry': geometry
                        })
                    else:
                        regions['solution'].append({
                            'text': block_text,
                            'geometry': geometry
                        })
                    
            return regions
        except Exception as e:
            raise Exception(f"Error segmenting document: {str(e)}")

    def process_math_expressions(self, equation_regions, image):
        """Convert mathematical expressions to LaTeX"""
        latex_expressions = []
        
        try:
            for region in equation_regions:
                geometry = region['geometry']
                h, w = image.shape[:2]
                y1 = int(geometry[0][1] * h)
                x1 = int(geometry[0][0] * w)
                y2 = int(geometry[1][1] * h)
                x2 = int(geometry[1][0] * w)
                
                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)
                
                if y2 > y1 and x2 > x1:
                    equation_img = image[y1:y2, x1:x2]
                    if equation_img.size > 0:
                        pil_equation = self.numpy_to_pil(equation_img)
                        latex = self.latex_reader(pil_equation)
                        latex_expressions.append({
                            'latex': latex,
                            'text': region['text']
                        })
                
            return latex_expressions
        except Exception as e:
            raise Exception(f"Error processing math expressions: {str(e)}")

    def extract_handwritten_text(self, solution_regions, image):
        """Extract handwritten text from solution regions"""
        text_segments = []
        
        try:
            for region in solution_regions:
                geometry = region['geometry']
                h, w = image.shape[:2]
                y1 = int(geometry[0][1] * h)
                x1 = int(geometry[0][0] * w)
                y2 = int(geometry[1][1] * h)
                x2 = int(geometry[1][0] * w)

                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)
                
                if y2 > y1 and x2 > x1:
                    text_img = image[y1:y2, x1:x2]
                    if text_img.size > 0:
                        result = self.text_reader.readtext(text_img)
                        text_segments.extend([{
                            'text': text,
                            'confidence': conf,
                            'original': region['text']
                        } for _, text, conf in result if conf > 0.5])
                
            return text_segments
        except Exception as e:
            raise Exception(f"Error extracting handwritten text: {str(e)}")

    def process_paper(self, image_path):
        """Process entire paper"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            processed_image, original_image = self.preprocess_image(image_path)
            regions = self.segment_document(processed_image)
            results = {
                'header': self.extract_handwritten_text(regions['header'], original_image),
                'question': self.extract_handwritten_text(regions['question'], original_image),
                'solution_text': self.extract_handwritten_text(regions['solution'], original_image),
                'equations': self.process_math_expressions(regions['equations'], original_image)
            }
            
            return results
        except Exception as e:
            raise Exception(f"Error processing paper: {str(e)}")