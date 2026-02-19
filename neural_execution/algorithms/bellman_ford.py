def bellman_ford_trace(G, source):
    # Bellman-Ford -> Shortest Path
    # 선행 노드를 추적하게 구현
    n = G.number_of_nodes()
    dist = [float('inf')] * n
    pred = [-1] * n
    dist[source] = 0
    pred[source] = source

    dist_traces = []
    pred_traces = []
    dist_traces.append(list(dist))
    pred_traces.append(list(pred))

    for _ in range(n-1):
        new_dist = list(dist)
        new_pred = list(pred)

        for u, v, data in G.edges(data=True):
            w = data['weight']

            if dist[u] + w < new_dist[v]:
                new_dist[v] = dist[u] + w
                new_pred[v] = u

            if dist[v] + w < new_dist[u]:
                new_dist[u] = dist[v] + w
                new_pred[u] = v

        if new_dist == dist:
            break

        dist = new_dist
        pred = new_pred
        dist_traces.append(list(dist))
        pred_traces.append(list(pred))

    return dist_traces, pred_traces
