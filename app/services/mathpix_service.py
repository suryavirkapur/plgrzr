import json
import time
import requests
from fastapi import HTTPException
from app.config import settings

class MathpixService:
    def __init__(self):
        self.headers = {
            "app_id": settings.APP_ID,
            "app_key": settings.API_KEY
        }
        self.options = {
            "conversion_formats": {"docx": True, "tex.zip": True},
            "math_inline_delimiters": ["$", "$"],
            "rm_spaces": True
        }

    async def process_pdf(self, file_content):
        try:
            # Initial PDF upload
            response = requests.post(
                "https://api.mathpix.com/v3/pdf",
                headers=self.headers,
                data={"options_json": json.dumps(self.options)},
                files={"file": ("filename.pdf", file_content, "application/pdf")}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Mathpix API request failed: {response.text}"
                )

            response_data = response.json()
            if "pdf_id" not in response_data:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid response from Mathpix: {response_data}"
                )

            doc_id = response_data["pdf_id"]

            # Poll for completion
            while True:
                status_response = requests.get(
                    f"https://api.mathpix.com/v3/pdf/{doc_id}",
                    headers=self.headers
                )
                
                if status_response.status_code != 200:
                    raise HTTPException(
                        status_code=status_response.status_code,
                        detail=f"Failed to check status: {status_response.text}"
                    )
                
                status = status_response.json()
                if status.get('status') == 'completed':
                    break
                elif status.get('status') == 'error':
                    raise HTTPException(
                        status_code=500,
                        detail=f"Mathpix processing error: {status.get('error', 'Unknown error')}"
                    )
                time.sleep(1)

            # Get final result
            result = requests.get(
                f"https://api.mathpix.com/v3/pdf/{doc_id}.lines.json",
                headers=self.headers
            )
            
            if result.status_code != 200:
                raise HTTPException(
                    status_code=result.status_code,
                    detail=f"Failed to get results: {result.text}"
                )

            return result.json()
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Network error: {str(e)}"
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )