def bfs_trace(G, source):
    # BFS -> Reachability
    # BFS의 최종 해를 구하는 것이 아닌 각 스텝을 기록하여 GNN이 한 스텝을 모방하도록 학습시키기 위한 데이터셋 생성
    n = G.number_of_nodes()

    visited = [0] * n
    visited[source] = 1

    traces = []
    traces.append(list(visited))

    queue = [source]

    while queue:
        next_queue = []
        for node in queue:
            for neighbor in G.neighbors(node):
                if visited[neighbor] == 0:
                    visited[neighbor] = 1
                    next_queue.append(neighbor)

        if next_queue:
            traces.append(list(visited))

        queue = next_queue

    return traces
