from collections import deque
import heapq

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

def dijkstra(graph, start, target=None, must_pass=None, cannot_pass=None):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if cannot_pass and current_vertex == cannot_pass:
            continue

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            if must_pass and neighbor == must_pass and distances[current_vertex] == float('infinity'):
                continue 

            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def main():
    graph = {
        'a': {'b': 1, 'd': 10},
        'b': {'a': 1, 'c': 2},
        'c': {'b': 2, 'd': 3},
        'd': {'a': 10, 'c': 3, 'e': 4},
        'e': {'d': 4, 'f': 3, 'g': 5, 'h': 12},
        'f': {'h': 4},
        'g': {'e': 5, 'h': 6},
        'h': {'j': 7},
        'j': {'i': 8},
        'i': {'d': 9},
    }   
    
    print("BFS:", end=' ')
    bfs(graph, 'a')
    print()

    print("DFS:", end=' ')
    dfs(graph, 'a')
    print()

    print("Cau a:")
    distances = dijkstra(graph, 'a', must_pass='g')
    print("Distances:", distances)

    print("Cau b:")
    distances_via_g = dijkstra(graph, 'a', target='i', must_pass='g')
    print("Distances:", distances_via_g)

    print("Cau c:")
    distances_without_f = dijkstra(graph, 'a', target='i', cannot_pass='f')
    print("Distances:", distances_without_f)

if __name__ == "__main__":
    main()
