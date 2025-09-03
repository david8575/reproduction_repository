"""
논문과 동일한 하이퍼파라미터 설정
"""

# 모델 설정
MODEL_CONFIG = {
    'd_model': 512,
    'd_ff': 2048,
    'n_layers': 6,
    'n_heads': 8,
    'dropout': 0.1,
    'max_len': 512
}

# 학습 설정
TRAINING_CONFIG = {
    'batch_size': 128,  # 논문의 배치 크기
    'learning_rate': 0.0001,
    'betas': (0.9, 0.98),
    'eps': 1e-9,
    'warmup_steps': 4000,
    'max_epochs': 100,
    'gradient_clip': 1.0
}

# 데이터 설정
DATA_CONFIG = {
    'src_vocab_size': 37000,  # 영어
    'tgt_vocab_size': 32000,  # 독일어
    'min_freq': 2,
    'max_len': 512
}

# 경로 설정
PATHS = {
    'data_dir': 'data',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs'
}
