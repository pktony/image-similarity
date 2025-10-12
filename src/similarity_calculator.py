"""
Cosine Similarity Calculator

Calculates cosine similarity between query embeddings and class prototypes
for Pokémon matching with unknown detection via threshold-based logic.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional


class SimilarityCalculator:
    def __init__(self, tau1: float = 0.30, tau2: float = 0.04):
        """
        Initialize Similarity Calculator

        Args:
            tau1: Minimum top similarity score threshold (default: 0.30)
            tau2: Minimum margin threshold between top two scores (default: 0.04)
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.prototypes: Optional[Dict[str, np.ndarray]] = None
        self.class_names: Optional[List[str]] = None

    def load_prototypes_from_dict(self, prototypes_dict: Dict[str, np.ndarray]):
        """
        Load prototypes from dictionary

        Args:
            prototypes_dict: Dictionary mapping class names to prototype vectors
        """
        self.prototypes = prototypes_dict
        self.class_names = sorted(prototypes_dict.keys())
        print(f"✓ Loaded {len(self.class_names)} class prototypes")

    def load_prototypes_from_npz(self, npz_path: str):
        """
        Load prototypes from NPZ file

        Args:
            npz_path: Path to NPZ file with prototype vectors
        """
        data = np.load(npz_path)
        prototypes_dict = {name: data[name] for name in data.files}
        self.load_prototypes_from_dict(prototypes_dict)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector (assumes L2-normalized)
            vec2: Second vector (assumes L2-normalized)

        Returns:
            Cosine similarity score (range: -1 to 1)
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def cosine_similarity_batch(self, query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and multiple prototypes

        Args:
            query: Query vector (1D, L2-normalized)
            prototypes: Prototype matrix (N x dim, L2-normalized)

        Returns:
            Array of similarity scores (N,)
        """
        return prototypes @ query

    def calculate_similarities(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate similarities between query and all class prototypes

        Args:
            query_embedding: Query embedding vector (L2-normalized)

        Returns:
            Dictionary mapping class names to similarity scores
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not loaded. Call load_prototypes_* first.")

        similarities = {}
        for class_name in self.class_names:
            prototype = self.prototypes[class_name]
            similarity = self.cosine_similarity(query_embedding, prototype)
            similarities[class_name] = similarity

        return similarities

    def get_top_k(
        self,
        query_embedding: np.ndarray,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most similar classes

        Args:
            query_embedding: Query embedding vector (L2-normalized)
            k: Number of top results to return (default: 3)

        Returns:
            List of (class_name, similarity_score) tuples sorted by score descending
        """
        similarities = self.calculate_similarities(query_embedding)

        # Sort by similarity descending
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:k]

    def detect_unknown(
        self,
        top_scores: List[Tuple[str, float]]
    ) -> Tuple[bool, float, float]:
        """
        Detect if query should be classified as "unknown"

        Args:
            top_scores: List of (class_name, score) tuples sorted descending

        Returns:
            Tuple of (is_unknown, s1, margin)
            - is_unknown: True if should classify as unknown
            - s1: Top similarity score
            - margin: Difference between top two scores
        """
        if len(top_scores) == 0:
            return True, 0.0, 0.0

        s1 = top_scores[0][1]

        # Calculate margin
        if len(top_scores) >= 2:
            s2 = top_scores[1][1]
            margin = s1 - s2
        else:
            margin = s1  # Only one class, margin = s1

        # Unknown detection logic
        is_unknown = (s1 < self.tau1) or (margin < self.tau2)

        return is_unknown, s1, margin

    def match(
        self,
        query_embedding: np.ndarray,
        k: int = 3
    ) -> Dict:
        """
        Match query embedding to most similar Pokémon classes

        Args:
            query_embedding: Query embedding vector (L2-normalized)
            k: Number of top results to return (default: 3)

        Returns:
            Dictionary with match results:
            {
                "top_k": [(name, score), ...],
                "verdict": str,  # class name or "unknown"
                "s1": float,     # top score
                "margin": float, # s1 - s2
                "is_unknown": bool
            }
        """
        # Get top-k similar classes
        top_k = self.get_top_k(query_embedding, k)

        # Detect unknown
        is_unknown, s1, margin = self.detect_unknown(top_k)

        # Determine verdict
        verdict = "unknown" if is_unknown else top_k[0][0]

        return {
            "top_k": top_k,
            "verdict": verdict,
            "s1": s1,
            "margin": margin,
            "is_unknown": is_unknown
        }

    def get_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold values

        Returns:
            Dictionary with tau1 and tau2 values
        """
        return {
            "tau1": self.tau1,
            "tau2": self.tau2
        }


# Usage example
if __name__ == "__main__":
    import sys
    from embedding_extractor import ImageEmbeddingExtractor

    prototype_path = sys.argv[1] if (len(sys.argv) > 1) else "embedding_output/prototypes.npz"
    input_image = sys.argv[2] if (len(sys.argv) > 2) else "examples/1.jpg"

    # Initialize calculator
    calc = SimilarityCalculator(tau1=0.30, tau2=0.04)

    # Load prototypes from JSON or NPZ

    if prototype_path.endswith('.npz'):
        calc.load_prototypes_from_npz(prototype_path)
    else:
        print("Error: Prototype file must be .npz")
        sys.exit(1)

    extractor = ImageEmbeddingExtractor()
    query_embedding = extractor.extract_single_image(input_image)

    # print("Prototypes loaded:", calc.prototypes)

    # similarities = dict()
    # for class_name in calc.prototypes.keys():
    #     prototype = calc.prototypes[class_name]
    #     similarity = calc.cosine_similarity(query_embedding, prototype)
    #     # print(f"Similarity to {class_name}: {similarity:.4f}")
    #     similarities[class_name] = similarity

    # sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # print(f"Similarities calculated: {sorted_similarities[:4]}")

    result = calc.match(query_embedding)
    print(result)

    # Example: Create dummy query embedding
    # print("\nExample with random query:")
    # dim = len(calc.prototypes[calc.class_names[0]])
    # query = np.random.randn(dim).astype(np.float32)
    # query = query / np.linalg.norm(query)  # L2 normalize

    # Match query
    # result = calc.match(query, k=3)

    # print(f"\nMatch Results:")
    # print(f"  Top-3:")
    # for name, score in result['top_k']:
    #     print(f"    {name}: {score:.4f}")
    # print(f"  Verdict: {result['verdict']}")
    # print(f"  S1: {result['s1']:.4f}")
    # print(f"  Margin: {result['margin']:.4f}")
    # print(f"  Unknown: {result['is_unknown']}")

    # Display threshold info
    # thresholds = calc.get_thresholds()
    # print(f"\nThresholds:")
    # print(f"  tau1 (min score): {thresholds['tau1']}")
    # print(f"  tau2 (min margin): {thresholds['tau2']}")
