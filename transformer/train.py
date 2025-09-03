import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import time
from pathlib import Path
from model.make_model import make_transformer_model
from data.preprocess import TranslationDataset, create_vocab_files
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS

class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.0001, betas=(0.9, 0.98), eps=1e-9):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 논문과 동일한 optimizer 설정
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD 토큰 무시
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda step: min((step + 1) ** (-0.5), (step + 1) * 4000 ** (-1.5))
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Teacher forcing: tgt[:-1]을 입력으로, tgt[1:]을 타겟으로
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            self.optimizer.zero_grad()
            
            output, _ = self.model(src, tgt_input)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt_target.contiguous().view(-1))
            
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)  # 논문의 gradient clipping
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]
                
                output, _ = self.model(src, tgt_input)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_target.contiguous().view(-1))
                total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

def main():
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 생성 (논문과 동일한 설정)
    model, _ = make_transformer_model(
        src_vocab_size=DATA_CONFIG['src_vocab_size'],
        tgt_vocab_size=DATA_CONFIG['tgt_vocab_size'],
        d_model=MODEL_CONFIG['d_model'],
        d_ff=MODEL_CONFIG['d_ff'],
        n_layers=MODEL_CONFIG['n_layers'],
        n_heads=MODEL_CONFIG['n_heads'],
        dropout=MODEL_CONFIG['dropout'],
        max_len=MODEL_CONFIG['max_len'],
        device=device
    )
    
    model = model.to(device)
    
    # 데이터 로더 생성
    # (실제 데이터 파일 경로로 수정 필요)
    train_loader = None  # 실제 데이터로 생성
    val_loader = None    # 실제 데이터로 생성
    
    # 학습기 생성
    trainer = TransformerTrainer(model, train_loader, val_loader, device)
    
    # 학습 루프
    num_epochs = TRAINING_CONFIG['max_epochs']
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    main()
