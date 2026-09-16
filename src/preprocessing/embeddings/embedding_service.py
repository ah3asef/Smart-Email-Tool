from sentence_transformers import SentenceTransformer
from helpers.config import get_settings

class EmbeddingService:

    def __init__(self,device : str = 'cuda', batch_size : int = 32):

        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(get_settings().embedder_model_name, device=device)

    def embed_texts(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        return embeddings.tolist()