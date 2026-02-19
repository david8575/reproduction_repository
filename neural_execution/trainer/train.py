import torch
import torch.nn as nn
from tqdm import tqdm


def train_bfs_only(model, bfs_train, epochs, device, accum_steps=32, lr=0.001):
    # BFS 데이터만의 학습
    bce_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='BFS Only'):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(bfs_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
            loss_out = bce_loss(out, data.y)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(bfs_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"    Epoch {epoch+1} | Loss {total_loss/len(bfs_train):.6f}")


def train_bf_only(model, bf_train, epochs, device, accum_steps=32, lr=0.001):
    # Bellman-Ford 데이터로만 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='BF Only'):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(bf_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
            mask = data.mask

            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0).to(device)

            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(bf_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"    Epoch {epoch+1} | Loss {total_loss/len(bf_train):.6f}")


def train_prim_only(model, prim_train, epochs, device, accum_steps=32, lr=0.001):
    # Prim 데이터로만 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='Prim Only'):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(prim_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='prim')
            mask = data.mask

            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0).to(device)

            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(prim_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"    Epoch {epoch+1} | Loss {total_loss/len(prim_train):.6f}")


def train_joint(model, bfs_train, bf_train, prim_train, epochs, device, accum_steps=32, lr=0.001):
    # BFS + Bellman-Ford + Prim 같이 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='BFS + BF + Prim'):
        model.train()
        total_bfs_loss = 0
        total_bf_loss = 0
        total_prim_loss = 0

        # BFS 학습
        optimizer.zero_grad()
        for i, data in enumerate(bfs_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
            loss_out = bce_loss(out, data.y)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_bfs_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(bfs_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # BF 학습
        optimizer.zero_grad()
        for i, data in enumerate(bf_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
            mask = data.mask

            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0).to(device)

            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_bf_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(bf_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Prim 학습
        optimizer.zero_grad()
        for i, data in enumerate(prim_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='prim')
            mask = data.mask

            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0).to(device)

            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_prim_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(prim_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            avg_bfs = total_bfs_loss / len(bfs_train)
            avg_bf = total_bf_loss / len(bf_train)
            avg_prim = total_prim_loss / len(prim_train)
            tqdm.write(f"    Epoch {epoch+1} | BFS {avg_bfs:.6f} | BF {avg_bf:.6f} | Prim {avg_prim:.6f}")
