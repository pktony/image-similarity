# Pokémon Look-Alike Matcher

> 사람 얼굴 사진을 업로드하면 CLIP 임베딩과 코사인 유사도를 사용하여 가장 닮은 포켓몬 Top-3를 찾아주는 시스템입니다.

## 🎯 주요 특징

- **학습 불필요**: 사전 학습된 CLIP 모델 사용 (fine-tuning 없음)
- **임베딩 기반**: OpenAI CLIP ViT-B/32 모델로 이미지 임베딩 추출
- **코사인 유사도**: L2 정규화된 벡터 간 유사도 계산
- **Unknown 처리**: 임계값 기반 불확실성 감지
- **FastAPI 백엔드**: 빠르고 안정적인 REST API
- **웹 데모**: 간단한 HTML 인터페이스

## 📦 시스템 요구사항

- Python 3.10 이상
- 4GB+ RAM 권장
- (선택) CUDA 지원 GPU

## 🚀 Quick Start

### 1. 가상환경 생성 및 패키지 설치

```bash
# 저장소 클론 (또는 디렉토리 이동)
cd pokelookalike

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 데이터 준비

포켓몬 클래스별 이미지를 다음과 같이 구성하세요:

```
data/pokemon/
├── pikachu/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── charmander/
│   ├── img1.jpg
│   └── ...
├── squirtle/
└── ...
```

**최소 요구사항**:
- 클래스당 최소 10장 이상의 이미지 권장
- 지원 포맷: `.jpg`, `.jpeg`, `.png`

**참고**: 현재 6개 클래스 폴더가 생성되어 있습니다 (pikachu, charmander, squirtle, bulbasaur, jigglypuff, eevee). 각 폴더에 해당 포켓몬 이미지를 추가하세요.

### 3. 프로토타입 임베딩 생성

```bash
# 기본 실행
python scripts/make_prototypes.py

# 또는 옵션 지정
python scripts/make_prototypes.py \
  --data-root data/pokemon \
  --out artifacts/prototypes.json \
  --image-size 224 \
  --drop-top-percent 10
```

**실행 결과**:
- `artifacts/prototypes.json`: 클래스별 프로토타입 벡터 (JSON)
- `artifacts/prototypes.npy`: NumPy 배열 형식

**처리 과정**:
1. 각 클래스 이미지에서 CLIP 임베딩 추출
2. L2 정규화
3. 중심에서 가장 먼 상위 10% 제거 (outlier removal)
4. 평균을 계산하여 클래스 프로토타입 생성

### 4. API 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

서버가 `http://localhost:8000`에서 실행됩니다.

### 5. API 테스트

#### Health Check

```bash
curl http://localhost:8000/health
```

**응답**:
```json
{
  "status": "ok"
}
```

#### Metadata

```bash
curl http://localhost:8000/metadata
```

**응답 예시**:
```json
{
  "model_name": "openai/clip-vit-base-patch32",
  "embedding_dim": 512,
  "num_classes": 6,
  "tau1": 0.30,
  "tau2": 0.04,
  "top_k": 3,
  "classes": ["pikachu", "charmander", "squirtle", "bulbasaur", "jigglypuff", "eevee"]
}
```

#### Match

```bash
curl -X POST http://localhost:8000/match \
  -F "file=@/path/to/your/image.jpg"
```

**응답 예시**:
```json
{
  "top3": [
    {
      "name": "pikachu",
      "score": 0.42,
      "explanation": "밝은 톤, 둥근 윤곽, 활기찬 인상"
    },
    {
      "name": "jigglypuff",
      "score": 0.39,
      "explanation": "부드러운 핑크 톤, 큰 눈망울, 귀여운 분위기"
    },
    {
      "name": "eevee",
      "score": 0.33,
      "explanation": "따뜻한 브라운 톤, 부드러운 인상, 친근한 느낌"
    }
  ],
  "verdict": "pikachu",
  "s1": 0.42,
  "margin": 0.03
}
```

**Unknown 예시**:
```json
{
  "top3": [...],
  "verdict": "unknown",
  "s1": 0.25,
  "margin": 0.02
}
```

## 🌐 웹 데모 사용

1. API 서버 실행 (위 4번 참조)
2. `web/index.html`을 브라우저에서 열기
3. 사진 선택 버튼을 클릭하여 이미지 업로드
4. 결과 확인

**주의**: CORS 문제가 있을 경우 FastAPI에 CORS 미들웨어를 추가하거나 로컬 서버를 사용하세요.

## 🔧 설정 및 튜닝

### config.yaml 수정

```yaml
model_name: "openai/clip-vit-base-patch32"
image_size: 224
tau1: 0.30      # 최소 Top 유사도 임계값
tau2: 0.04      # 최소 마진 임계값 (s1 - s2)
top_k: 3
prototype_path: "artifacts/prototypes.json"
```

### 환경 변수로 임계값 override

```bash
export POKE_TAU1=0.35
export POKE_TAU2=0.05
uvicorn app.main:app --reload --port 8000
```

### 임계값 분석 도구

사람 얼굴 이미지 데이터셋을 사용하여 최적의 `tau1`, `tau2` 값을 분석할 수 있습니다:

```bash
# 테스트 데이터 준비
# data/test_humans/ 에 사람 얼굴 이미지 배치

# 분석 실행
python scripts/inspect_thresholds.py \
  --test-dir data/test_humans \
  --prototypes artifacts/prototypes.json \
  --out-dir reports
```

**결과**:
- `reports/threshold_distributions.png`: s1 및 margin 분포 그래프
- 콘솔에 통계 및 권장 임계값 출력

**해석**:
- `tau1`: s1 분포의 50-75 백분위수 근처 설정
- `tau2`: margin 분포의 25-50 백분위수 근처 설정

## 🧪 테스트

### 유닛 테스트 실행

```bash
pytest tests/test_similarity.py -v
```

### 통합 테스트 실행

```bash
# 먼저 프로토타입 생성 및 서버 실행 필요
pytest tests/test_api.py -v
```

### 전체 테스트

```bash
pytest -v
```

## 📁 프로젝트 구조

```
pokelookalike/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI 앱: /health, /metadata, /match
│   ├── deps.py           # 모델/프로토타입 싱글톤 로더
│   ├── schemas.py        # Pydantic 모델
│   └── utils.py          # 이미지 전처리, 유사도 계산
├── scripts/
│   ├── make_prototypes.py      # 프로토타입 생성 스크립트
│   └── inspect_thresholds.py   # 임계값 분석 도구
├── data/
│   ├── pokemon/
│   │   ├── pikachu/*.jpg
│   │   ├── charmander/*.jpg
│   │   └── ...
│   └── class_explanations.json
├── artifacts/
│   ├── prototypes.json
│   └── prototypes.npy
├── tests/
│   ├── test_api.py
│   └── test_similarity.py
├── web/
│   ├── index.html
│   └── style.css
├── config.yaml
├── requirements.txt
└── README.md
```

## 🔬 작동 원리

### 1. 프로토타입 생성

각 포켓몬 클래스에 대해:
1. 모든 이미지에서 CLIP 임베딩 추출
2. L2 정규화
3. 클래스 중심으로부터 거리 계산
4. 상위 10% 아웃라이어 제거
5. 평균을 계산하여 단일 프로토타입 벡터 생성

### 2. 추론 과정

1. 입력 이미지 전처리:
   - RGB 변환
   - 중앙 크롭 (정사각형)
   - 224×224 리사이즈
   - 정규화

2. CLIP 임베딩 추출 및 L2 정규화

3. 모든 클래스 프로토타입과 코사인 유사도 계산

4. 내림차순 정렬, Top-k 선택

5. Unknown 판정:
   - `s1 < tau1` 또는 `margin < tau2`이면 "unknown"
   - 그렇지 않으면 Top-1 클래스 반환

### 3. Unknown 감지 규칙

```python
s1 = similarities[0].score
s2 = similarities[1].score
margin = s1 - s2

if s1 < tau1 or margin < tau2:
    verdict = "unknown"
else:
    verdict = similarities[0].name
```

## ⚙️ 고급 사용법

### 다른 CLIP 모델 사용

`config.yaml` 수정:
```yaml
model_name: "openai/clip-vit-large-patch14"
```

프로토타입 재생성 필요:
```bash
python scripts/make_prototypes.py --model-name openai/clip-vit-large-patch14
```

### 커스텀 클래스 설명 추가

`data/class_explanations.json` 수정:
```json
{
  "pikachu": "밝은 톤, 둥근 윤곽, 활기찬 인상",
  "your_class": "설명 추가"
}
```

### API 로깅

로그 레벨 조정은 `app/main.py`에서:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🐛 문제 해결

### 프로토타입 파일이 없다는 에러

```bash
python scripts/make_prototypes.py
```

### CUDA out of memory

CPU 모드로 실행하거나 작은 배치 처리:
```python
# app/deps.py에서 device = "cpu" 강제 설정
```

### 이미지 업로드 실패 (400 에러)

- 지원되는 포맷: JPG, JPEG, PNG
- 파일 크기: 일반적으로 10MB 이하 권장

### CORS 에러 (웹 데모)

`app/main.py`에 CORS 미들웨어 추가:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 성능

- **Cold Start**: 모델 로딩 ~3-5초 (첫 요청)
- **추론 시간**: ~200ms/이미지 (CPU, 224px)
- **GPU 사용 시**: ~50-100ms/이미지

## 🔒 보안 및 프로덕션 배포

프로덕션 배포 시 고려사항:
- Rate limiting 추가
- 파일 크기 제한
- 입력 검증 강화
- HTTPS 사용
- 로깅 및 모니터링 설정

## 📝 라이센스

MIT License

## 🤝 기여

Issues 및 Pull Requests 환영합니다!

## 📧 문의

프로젝트 관련 문의는 Issues를 통해 남겨주세요.

---

**Made with ❤️ and CLIP**
