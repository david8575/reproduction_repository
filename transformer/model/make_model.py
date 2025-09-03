import torch
import torch.nn as nn
import math

from model.embeddings.transformer_embedding import TransformerEmbedding
from model.embeddings.token_embedding import TokenEmbedding
from model.embeddings.positional_encoding import PositionalEncoding
from model.layers.multi_head_attention_layer import MultiHeadAttentionLayer
from model.layers.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from model.layers.generator import Generator
from model.blocks.encode_block import EncoderBlock
from model.blocks.decode_block import DecoderBlock
from model.models.encoder import Encoder
from model.models.decoder import Decoder
from model.models.transformer import Transformer

def make_transformer_model(
    src_vocab_size,
    tgt_vocab_size,
    d_model=512,
    d_ff=2048,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
    max_len=5000,
    device=torch.device("cpu")
):
    """
    Transformer 모델을 생성하는 팩토리 함수
    
    Args:
        src_vocab_size: 소스 어휘 크기
        tgt_vocab_size: 타겟 어휘 크기
        d_model: 모델 차원
        d_ff: Feed Forward 네트워크 차원
        n_layers: 인코더/디코더 레이어 수
        n_heads: 멀티헤드 어텐션 헤드 수
        dropout: 드롭아웃 비율
        max_len: 최대 시퀀스 길이
        device: 디바이스
    
    Returns:
        Transformer 모델
    """
    
    # 1. Embedding 레이어들 생성
    src_token_embedding = TokenEmbedding(d_model, src_vocab_size)
    tgt_token_embedding = TokenEmbedding(d_model, tgt_vocab_size)
    
    src_pos_embedding = PositionalEncoding(d_model, max_len, device)
    tgt_pos_embedding = PositionalEncoding(d_model, max_len, device)
    
    src_embedding = TransformerEmbedding(src_token_embedding, src_pos_embedding)
    tgt_embedding = TransformerEmbedding(tgt_token_embedding, tgt_pos_embedding)
    
    # 2. Multi-Head Attention 레이어들 생성
    src_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
    tgt_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
    cross_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
    
    # 3. Position-wise Feed Forward 레이어들 생성
    src_ff = PositionWiseFeedForwardLayer(d_model, d_ff, dropout)
    tgt_ff = PositionWiseFeedForwardLayer(d_model, d_ff, dropout)
    
    # 4. Encoder/Decoder 블록들 생성
    encoder_block = EncoderBlock(src_attention, src_ff, d_model, dropout)
    decoder_block = DecoderBlock(tgt_attention, cross_attention, tgt_ff, d_model, dropout)
    
    # 5. Encoder/Decoder 생성
    encoder = Encoder(encoder_block, n_layers)
    decoder = Decoder(decoder_block, n_layers)
    
    # 6. Generator 생성
    generator = Generator(d_model, tgt_vocab_size)
    
    # 7. Transformer 모델 생성
    model = Transformer(src_embedding, tgt_embedding, encoder, decoder, generator)
    
    # 8. 가중치 초기화
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

def make_model_for_test():
    """
    테스트용 간단한 모델 생성
    """
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    d_ff = 512
    n_layers = 2
    n_heads = 4
    
    model = make_transformer_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.1
    )
    
    return model

if __name__ == "__main__":
    # 테스트
    model = make_model_for_test()
    print(f"모델 생성 완료!")
    print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 간단한 forward pass 테스트
    src = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    tgt = torch.randint(0, 1000, (2, 8))   # batch_size=2, seq_len=8
    
    try:
        output, decoder_out = model(src, tgt)
        print(f"Forward pass 성공")
        print(f"Output shape: {output.shape}")
        print(f"Decoder output shape: {decoder_out.shape}")
    except Exception as e:
        print(f"Forward pass 실패: {e}")