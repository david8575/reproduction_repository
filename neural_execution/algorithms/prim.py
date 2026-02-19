def prim_trace(G, source):
    # Prim -> Minimum Spanning Tree
    # 선행 노드를 추적하게 구현
    n = G.number_of_nodes()
    key = [float('inf')] * n
    pred = [-1] * n
    in_mst = [False] * n

    key[source] = 0
    pred[source] = source

    key_traces = [list(key)]
    pred_traces = [list(pred)]

    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
        in_mst[u] = True

        new_key = list(key)
        new_pred = list(pred)

        for v, data in G[u].items():
            w = data['weight']

            if not in_mst[v] and w < new_key[v]:
                new_key[v] = w
                new_pred[v] = u

        if new_key == key:
            break

        key = new_key
        pred = new_pred
        key_traces.append(list(key))
        pred_traces.append(list(pred))

    return key_traces, pred_traces
