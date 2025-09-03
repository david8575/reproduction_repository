# Transformer Implementation

## 📖 Paper
### Attention Is All You Need
- https://arxiv.org/abs/1706.03762

## 🚀 Features
- 논문과 동일한 아키텍처 구현
- RTX 3090 최적화
- WMT 2014 English-German 데이터셋 지원
- BLEU 점수 및 Perplexity 측정

## 📁 Project Structure
```
transformer/
├── model/                          # Transformer 모델 구현
│   ├── blocks/                     # Encoder/Decoder 블록
│   ├── embeddings/                 # Token 및 Positional Embedding
│   ├── layers/                     # Attention, FFN, Residual Connection
│   ├── models/                     # 전체 모델 구조
│   └── make_model.py               # 모델 생성 팩토리
├── data/                           # 데이터 처리
│   ├── download_wmt.py             # WMT 데이터 다운로드
│   └── preprocess.py               # 데이터 전처리
├── utils/                          # 유틸리티
│   ├── metrics.py                  # BLEU, Perplexity 계산
│   └── visualization.py            # Attention 가중치 시각화
├── config.py                       # 하이퍼파라미터 설정
├── train.py                        # 메인 학습 스크립트
└── requirements.txt                # 필요한 패키지
```

## 🛠️ Installation

### 1. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. 간단한 테스트 (CPU/GPU)
```bash
python -m model.make_model
```

### 2. WMT 데이터 다운로드
```bash
python data/download_wmt.py
```

### 3. 전체 학습 (RTX 3090 권장)
```bash
python train.py
```

## ⚙️ Configuration

논문과 동일한 하이퍼파라미터:
- **d_model**: 512
- **d_ff**: 2048
- **n_layers**: 6
- **n_heads**: 8
- **batch_size**: 128
- **learning_rate**: 0.0001

## 📊 Expected Performance (RTX 3090)

- **학습 속도**: CPU 대비 15-20배 빠름
- **배치 크기**: 128 (논문과 동일)
- **메모리**: 24GB VRAM으로 충분
- **학습 시간**: WMT 데이터 기준 1-2일

## 🔍 Testing

### CPU/GPU 테스트
```bash
python -m model.make_model
```

### Attention 가중치 시각화
```python
from utils.visualization import plot_attention_weights
# 사용 예시
```

### BLEU 점수 계산
```python
from utils.metrics import calculate_bleu
# 사용 예시
```

## 📝 Notes

- RTX 3090에서 최적 성능을 위해 설계됨
- 논문의 정확한 구현을 목표로 함
- WMT 2014 데이터셋으로 검증 가능

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request