from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_KEY: str
    APP_ID: str

    class Config:
        env_file = ".env"

settings = Settings()