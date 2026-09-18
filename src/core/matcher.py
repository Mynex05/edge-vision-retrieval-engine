import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("EdgeVisionEngine.Matcher")

class VectorMatcher:
    """
    Performs high-performance similarity search over a gallery of feature embeddings.
    """
    def __init__(self):
        self.gallery_embeddings = None
        self.gallery_ids = None

    def add_gallery(self, embeddings: torch.Tensor, ids: list):
        """Registers a gallery of feature vectors for retrieval."""
        self.gallery_embeddings = embeddings
        self.gallery_ids = ids
        logger.info(f"Registered gallery with {len(ids)} items. Embedding shape: {embeddings.shape}")

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> list:
        """
        Computes cosine similarity between the query and all gallery items,
        returning the top-K highest scoring matches.
        """
        if self.gallery_embeddings is None:
            raise ValueError("Gallery is empty. Please add gallery embeddings first.")

        # Ensure L2 normalization for accurate cosine similarity via dot product
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarities = torch.mm(query_embedding, self.gallery_embeddings.t()).squeeze(0)
        
        # Get top-k matches
        top_k = min(top_k, len(self.gallery_ids))
        scores, indices = torch.topk(similarities, k=top_k)

        results = []
        for score, idx in zip(scores, indices):
            item_id = self.gallery_ids[idx.item()]
            results.append({"id": item_id, "score": score.item()})

        logger.info(f"Top match found: ID '{results[0]['id']}' with similarity {results[0]['score']:.4f}")
        return results

if __name__ == "__main__":
    # Quick test of the matcher
    matcher = VectorMatcher()
    
    # Mock gallery: 10 items, 512-dim vectors
    mock_gallery = torch.randn(10, 512)
    mock_gallery = F.normalize(mock_gallery, p=2, dim=1)
    mock_ids = [f"item_{i}" for i in range(10)]
    
    matcher.add_gallery(mock_gallery, mock_ids)
    
    # Mock query embedding
    mock_query = torch.randn(1, 512)
    matches = matcher.search(mock_query, top_k=3)
    print("\nTop Search Results:", matches)