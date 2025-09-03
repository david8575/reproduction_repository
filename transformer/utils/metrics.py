import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def calculate_bleu(references, hypotheses):
    """
    BLEU 점수 계산
    """
    return corpus_bleu(references, hypotheses)

def calculate_perplexity(model, data_loader, criterion, device):
    """
    Perplexity 계산
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            output, _ = model(src, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_target.contiguous().view(-1))
            
            total_loss += loss.item()
            total_tokens += tgt_target.ne(0).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    perplexity = np.exp(avg_loss)
    
    return perplexity
