"""
CLIP ViT-B/32 Image Embedding Extractor
"""
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Union, List
import os


class ImageEmbeddingExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize CLIP ViT-B/32 model for image embedding extraction

        Args:
            model_name: Hugging Face model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading {model_name} on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print(f"✓ Model loaded successfully")

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path

        Args:
            image_path: Path to image file

        Returns:
            PIL Image in RGB format
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Cannot load image {image_path}: {str(e)}")

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for CLIP model

        Args:
            image: PIL Image or path to image file

        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = self.load_image(image)

        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def extract_embedding(self, image: Union[str, Image.Image], normalize: bool = True) -> np.ndarray:
        """
        Extract embedding from image using CLIP ViT-B/32

        Args:
            image: PIL Image or path to image file
            normalize: Whether to L2-normalize the embedding

        Returns:
            Embedding vector as numpy array (512-dim for ViT-B/32)
        """
        with torch.no_grad():
            pixel_values = self.preprocess_image(image)
            embedding = self.model.get_image_features(pixel_values)

            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().squeeze()
        
    def extract_single_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        if (image is None):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            v = self.model.get_image_features(**inputs)              # [1, D]
            v = torch.nn.functional.normalize(v, dim=-1)         # L2 정규화
        return v.squeeze(0).detach().cpu().numpy()               # [D]

    def extract_batch_embeddings(
        self,
        images: List[Union[str, Image.Image]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings from multiple images in batches

        Args:
            images: List of PIL Images or paths to image files
            normalize: Whether to L2-normalize embeddings
            batch_size: Number of images to process at once

        Returns:
            Array of embeddings (N x 512)
        """
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load and preprocess batch
            pil_images = []
            for img in batch:
                if isinstance(img, str):
                    pil_images.append(self.load_image(img))
                else:
                    pil_images.append(img)

            # Process batch
            with torch.no_grad():
                inputs = self.processor(images=pil_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                batch_embeddings = self.model.get_image_features(pixel_values)

                if normalize:
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def extract_from_file(self, file_path: str, normalize: bool = True, save_path: str = None) -> np.ndarray:
        """
        Extract embedding from a specific image file and optionally save it

        Args:
            file_path: Path to the image file
            normalize: Whether to L2-normalize the embedding
            save_path: Optional path to save the embedding (.npy file)

        Returns:
            Embedding vector as numpy array (512-dim)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Extracting embedding from: {file_path}")
        embedding = self.extract_embedding(file_path, normalize=normalize)

        if save_path:
            np.save(save_path, embedding)
            print(f"✓ Embedding saved to: {save_path}")

        return embedding

    def extract_from_directory(
        self,
        directory: str,
        pattern: str = "*",
        normalize: bool = True,
        save_path: str = None
    ) -> dict:
        """
        Extract embeddings from all images in a directory

        Args:
            directory: Path to directory containing images
            pattern: File pattern to match (e.g., "*.jpg", "*.png")
            normalize: Whether to L2-normalize embeddings
            save_path: Optional path to save embeddings (.npz file)

        Returns:
            Dictionary mapping filenames to embeddings
        """
        from pathlib import Path

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all matching image files
        image_files = []

        for ext in self.IMG_EXTS:
            if pattern == "*":
                image_files.extend(Path(directory).glob(f"*{ext}"))
                image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
            else:
                image_files.extend(Path(directory).glob(pattern))

        if not image_files:
            raise ValueError(f"No images found in {directory} matching pattern: {pattern}")

        print(f"Found {len(image_files)} images in {directory}")

        # Extract embeddings
        results = {}
        errors = []

        for img_path in image_files:
            img_path_str = str(img_path)
            try:
                embedding = self.extract_embedding(img_path_str, normalize=normalize)
                results[img_path.name] = embedding
                print(f"  ✓ {img_path.name}: {embedding.shape}")
            except Exception as e:
                errors.append((img_path.name, str(e)))
                print(f"  ✗ {img_path.name}: {str(e)}")

        if errors:
            print(f"\n⚠ Skipped {len(errors)} files with errors:")
            for filename, error in errors:
                print(f"  - {filename}: {error}")

        if not results:
            raise ValueError(f"No valid images could be processed from {directory}")

        if save_path:
            np.savez(save_path, **results)
            print(f"✓ All embeddings saved to: {save_path}")

        return results

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (512 for ViT-B/32)"""
        return self.model.config.projection_dim
    
    def list_images_in_folder(self, folder: Path):
        """지정된 폴더 내 모든 이미지 파일 리스트 반환"""
        imgs = []
        for ext in self.IMG_EXTS:
            imgs += list(folder.glob(f"*{ext}"))
            imgs += list(folder.glob(f"*{ext.upper()}"))
        return sorted(imgs)

    def build_npz_per_class(
        self,
        images_root="images",
        out_root="embeddings",
        batch_size=64,
        normalize=True
    ):
        images_root = Path(images_root)
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        prototypes = {}   # 클래스별 센트로이드
        counts = {}

        for class_dir in sorted([p for p in images_root.iterdir() if p.is_dir()]):
            class_name = class_dir.name
            image_files = self.list_images_in_folder(class_dir)
            if not image_files:
                print(f"⚠️  {class_name} 폴더에 이미지가 없습니다.")
                continue

            print(f"\n=== {class_name} ({len(image_files)} images) ===")

            # 배치 추론
            embeddings = []
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i + batch_size]
                batch_embs =   batch_embs = self.extract_batch_embeddings(
                    [str(p) for p in batch],  # Path 객체를 문자열로 변환
                    normalize=normalize,
                    batch_size=len(batch)
                )
                embeddings.append(batch_embs)
                print(f"  → {i+len(batch)}/{len(image_files)} done")

            embeddings = np.vstack(embeddings)  # (N, 512)
            filenames = [p.name for p in image_files]

            # .npz로 저장
            out_path = out_root / f"{class_name}.npz"
            np.savez(out_path, embeddings=embeddings, filenames=filenames)
            print(f"✓ saved {out_path.name} ({embeddings.shape})")

            # 센트로이드 계산
            centroid = embeddings.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            prototypes[class_name] = centroid
            counts[class_name] = len(embeddings)

        # 전체 프로토타입 저장
        np.savez(out_root / "prototypes.npz", **prototypes)
        meta = {
            "embedding_dim": next(iter(prototypes.values())).shape[0],
            "classes": list(prototypes.keys()),
            "counts": counts
        }
        with open(out_root / "prototypes.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n✓ prototypes saved to {out_root/'prototypes.npz'}")

# 사용 예제
if __name__ == "__main__":
    import sys

    # 1. 초기화
    extractor = ImageEmbeddingExtractor()
    print(f"Embedding dimension: {extractor.embedding_dim}")

    # 2. 단일 이미지 embedding 추출
    # embedding = extractor.extract_embedding("path/to/image.jpg")
    # print(f"Embedding shape: {embedding.shape}")

    # 3. 특정 파일에서 embedding 추출 및 저장
    # embedding = extractor.extract_from_file(
    #     "path/to/image.jpg",
    #     save_path="output/embedding.npy"
    # )

    # 4. 디렉토리의 모든 이미지에서 embedding 추출
    # embeddings = extractor.extract_from_directory(
    #     "path/to/images/",
    #     pattern="*.jpg",
    #     save_path="output/embeddings.npz"
    # )

    # 5. 커맨드라인에서 사용
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'images'
    save_path = sys.argv[2] if len(sys.argv) > 2 else 'embedding_output/'
    if os.path.isfile(file_path):
        # 단일 파일 처리
        embedding = extractor.extract_from_file(file_path, save_path=save_path)
        print(f"\nEmbedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 10 values: {embedding[:10]}")

    elif os.path.isdir(file_path):
        extractor.build_npz_per_class(file_path, save_path)
        # 디렉토리 처리
        # embeddings = extractor.extract_from_directory(file_path, save_path=save_path)
        # print(f"\nExtracted {len(embeddings)} embeddings")
    else:
        print(f"Error: {file_path} is not a valid file or directory")
