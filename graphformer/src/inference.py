from transformers import BertTokenizer
import torch

def inference_example(model, tokenizer, center_text, neighbor_texts, device):
    """
    center_text: str
    neighbor_texts: List[str]
    """
    model.eval()

    all_texts = [center_text] + neighbor_texts  # N texts
    encoding = tokenizer(
        all_texts,
        padding='max_length',
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].unsqueeze(0)         # (1, N, T)
    attention_mask = encoding['attention_mask'].unsqueeze(0)

    N = input_ids.size(1)
    relation_matrix = torch.full((1, N, N), 2, dtype=torch.long)  # 기본: neighbor-to-neighbor
    relation_matrix[:, 0, :] = 1  # center to neighbor
    relation_matrix[:, :, 0] = 1  # neighbor to center
    relation_matrix[:, 0, 0] = 0  # self

    with torch.no_grad():
        embedding = model(
            input_ids.to(device),
            attention_mask.to(device),
            relation_matrix.to(device)
        )  # (1, H)

    return embedding.squeeze(0)  # (H,)
