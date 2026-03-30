import random
import collections

def tabucol(G, k, iterations=10000, tabu_size=7):
    # Initial random coloring with k colors
    nodes = list(G.nodes())
    coloring = {node: random.randint(0, k-1) for node in nodes}
    
    tabu_list = collections.deque(maxlen=tabu_size) # Tabu list: stores (node, color) to avoid immediate reversals
    
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