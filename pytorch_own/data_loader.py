from torch_geometric.datasets import Planetoid

def load_data(name="Cora", root='./data'):
    """
    name: 'Cora', 'CiteSeer', 'PubMed'
    root: 데이터 저장 경로
    """
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    print(f"[{name} DataSet]")
    print(f"[Nodes]: {data.num_nodes}")
    print(f"[Edges]: {data.num_edges}")
    print(f"[Features]: {data.num_features}")
    print(f"[Classe]: {dataset.num_classes}")
    print(f"[Train/Val/Test]: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
    return data, dataset.num_classes

if __name__=="__main__":
    data, num_classes = load_data("Cora")
    