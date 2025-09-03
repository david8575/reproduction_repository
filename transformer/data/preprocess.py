import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re
from pathlib import Path

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=512):
        self.src_data = self.load_data(src_file)
        self.tgt_data = self.load_data(tgt_file)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]
        
        # 토큰화
        src_tokens = self.tokenize(src_text)
        tgt_tokens = self.tokenize(tgt_text)
        
        # 어휘 인덱스 변환
        src_indices = [self.src_vocab[token] for token in src_tokens]
        tgt_indices = [self.tgt_vocab[token] for token in tgt_tokens]
        
        # 패딩 및 길이 제한
        src_indices = self.pad_sequence(src_indices, self.max_len)
        tgt_indices = self.pad_sequence(tgt_indices, self.max_len)
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
    
    def tokenize(self, text):
        # 논문과 동일한 토큰화 방식
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def pad_sequence(self, tokens, max_len):
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))  # 0은 PAD 토큰
        return tokens

def build_vocab(text_iterator, min_freq=2, special_tokens=['<unk>', '<pad>', '<sos>', '<eos>']):
    """
    어휘 구축
    """
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text.lower())
    
    vocab = build_vocab_from_iterator(
        yield_tokens(text_iterator),
        min_freq=min_freq,
        specials=special_tokens
    )
    
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def create_vocab_files(src_file, tgt_file, min_freq=2):
    """
    소스와 타겟 파일로부터 어휘 파일 생성
    """
    with open(src_file, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f]
    src_vocab = build_vocab(src_texts, min_freq)
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_texts = [line.strip() for line in f]
    tgt_vocab = build_vocab(tgt_texts, min_freq)
    
    return src_vocab, tgt_vocab
