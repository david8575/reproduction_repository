import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.models import GraphFormerModel
from src.trainer import train
from data.dataset import GraphLinkDataset
from data.load_dblp import load_dblp_json_stream
from data.build_graph import build_graph

print("[step 1: DBLP 데이터 불러오는 중...]")
id_to_title, edges = load_dblp_json_stream("data/dblp_v12.json", max_papers=100_000)
print(f"[논문 수: {len(id_to_title)}, 인용 관계 수: {len(edges)}]")

print("[step 2: 인용 그래프 구축 중...]")
G, center_nodes = build_graph(id_to_title, edges, min_degree=2)
print(f"[중심 노드 수: {len(center_nodes)}]")

print("[step 3: 토크나이저 및 모델 준비 중...]")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = GraphFormerModel()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("[step 4: Dataset / DataLoader 구성 중...]")
dataset = GraphLinkDataset(G, center_nodes, tokenizer, num_negatives=4)
print(f"[dataset 크기: {len(dataset)}]")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("[step 5: 학습 시작]")

for epoch in range(5):
    print(f"\n[epoch {epoch + 1} 시작...]")
    try:
        loss = train(model, dataloader, optimizer, device)
        print(f"[Epoch {epoch + 1} 완료 - 평균 Loss: {loss:.4f}]")
    except Exception as e:
        print(f"[오류 발생 (epoch {epoch + 1}):]", e)
