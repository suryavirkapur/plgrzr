from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Required fields
    API_KEY: str
    APP_ID: str
    
    # Optional fields with defaults
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()