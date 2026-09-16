from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):

    #Google Secrets
    client_id : str
    project_id: str
    auth_uri : str
    token_uri : str
    auth_provider_x509_cert_url : str
    client_secret: str

    #Posgress
    postgres_host: str 
    postgres_port: int 
    postgres_db: str  
    postgres_user: str 
    postgres_password: str 

    #Embedder Model Name
    embedder_model_name : str

    #ChromaDB
    persist_directory : str
    collection_name : str

    class Config():
        env_file = ".env"

def get_settings():
    return Settings()