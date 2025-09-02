import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_link_prediction_loss(query_vecs, key_vecs, neg_vecs):
    """
    query_vecs: (B, H) -> 중심 노드(q)
    key_vecs: (B, H) -> 정답 노드(k)
    neg_Vecs: (B, K, H) -> 부정 샘플들(r_1 ~ r_k)
    """
    pos_score = torch.sum(query_vecs * key_vecs, dim=-1, keepdim=True)
    neg_score = torch.einsum("bd,bkd->bk", query_vecs, neg_vecs)
    logits = torch.cat([pos_score, neg_score], dim=1)
    labels = torch.zeros(query_vecs.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for i, batch in enumerate(loop):
        q_input_ids, q_mask, q_rel = batch["query"]
        k_input_ids, k_mask, k_rel = batch["key"]
        neg_input_ids, neg_mask, neg_rel = batch["negatives"]

        B, K, N, T = neg_input_ids.size()

        h_q = model(q_input_ids.to(device), q_mask.to(device), q_rel.to(device))
        h_k = model(k_input_ids.to(device), k_mask.to(device), k_rel.to(device))
        h_neg = model(
            neg_input_ids.view(B * K, N, T).to(device),
            neg_mask.view(B * K, N, T).to(device),
            neg_rel.view(B * K, N, N).to(device)
        ).view(B, K, -1)

        loss = compute_link_prediction_loss(h_q, h_k, h_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # tqdm 진행바에 현재 loss 표시
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)