import random
import collections

def tabucol(G, k, iterations=50000, tabu_size=None):
    # Initial random coloring with k colors
    nodes = list(G.nodes())
    coloring = {node: random.randint(0, k-1) for node in nodes}
    
    # Dynamic tabu tenure: Hertz & de Werra (1987) recommend ~0.6 * k
    if tabu_size is None:
        tabu_size = max(7, int(0.6 * k))
    tabu_list = collections.deque(maxlen=tabu_size)  # stores (node, color) pairs
    
    for i in range(iterations):
        conflicts = [edge for edge in G.edges() if coloring[edge[0]] == coloring[edge[1]]]
        
        if not conflicts:
            return True, coloring  # Valid k-coloring found
        
        # Pick a node involved in a conflict
        u, v = random.choice(conflicts)
        node_to_move = u if random.random() > 0.5 else v
        
        best_move = None
        min_conflicts = float('inf')
        
        # Try moving node_to_move to a different color
        for new_color in range(k):
            if new_color == coloring[node_to_move]: continue
            if (node_to_move, new_color) in tabu_list: continue
            
            # Temporary move to count new conflicts
            old_color = coloring[node_to_move]
            coloring[node_to_move] = new_color
            current_count = sum(1 for n in G.neighbors(node_to_move) if coloring[n] == new_color)
            
            if current_count < min_conflicts:
                min_conflicts = current_count
                best_move = new_color
            
            # Backtrack
            coloring[node_to_move] = old_color
            
        if best_move is not None:
            tabu_list.append((node_to_move, coloring[node_to_move]))
            coloring[node_to_move] = best_move
            
    return False, None