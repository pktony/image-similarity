# Image Similarity


## Image Embedding

hugging face의 transformers CLIPModel을 사용해서 embedding.
클래스 별로 이미지가 여러 장 있는 경우, 벡터의 평균값 (centroid)를 계산해서 전체 프로토타입 파일을 생성함.


#### Model
Pretrained Zero shot image classification 모델 중 
openai/clip-vit-base-patch32

https://github.com/openai/CLIP

포켓몬은 동물 형태를 하고 있는 캐릭터가 많음.
ImageNet-R, ImageNet, Birdsnap 에서 가장 좋은 성적을 냄.


#### Example
1. images/ 에 포켓몬 별로 이미지가 여러 장이 있음.
2. 각 폴더 별로 class를 구분하고, class 별로 embedding 추출 (build_npz_per_class) 
3. 클래스 별로 벡터의 평균값 (centroid)를 계산해서 전체 벡터 embedding 파일을 추출

## Cosine Similarity

이미지를 입력하면 prototype에서 유사한 코사인 벡터를 찾는다.

$$
similarity=\cos(\theta)=\frac{A \cdot B}{\lvert A \lvert \cdot \lvert B \lvert \lvert}
$$