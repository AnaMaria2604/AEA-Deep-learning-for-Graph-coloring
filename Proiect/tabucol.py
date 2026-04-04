import random
import collections

def tabucol(G, k, iterations=30000, tabu_size=None):
    nodes = list(G.nodes())
    if not nodes:
        return True, {}
        
    # Initial random coloring with k colors
    coloring = {node: random.randint(0, k-1) for node in nodes}
    
    # Dynamic tabu tenure: Hertz & de Werra (1987) recommend ~0.6 * k
    if tabu_size is None:
        tabu_size = max(7, int(0.6 * k))
        
    # Array tracking until which iteration a color is tabu for a given node
    tabu_until = {node: [0] * k for node in nodes}
    
    # O(1) lookup table for current conflicts: adj_color_table[node][color] = count
    adj_color_table = {node: [0] * k for node in nodes}
    for u, v in G.edges():
        c_u, c_v = coloring[u], coloring[v]
        adj_color_table[u][c_v] += 1
        adj_color_table[v][c_u] += 1

    # Keep track of nodes currently involved in conflicts for O(1) random choice
    conflict_list = []
    conflict_pos = {}
    
    def add_conflict(n):
        if n not in conflict_pos:
            conflict_pos[n] = len(conflict_list)
            conflict_list.append(n)
            
    def remove_conflict(n):
        if n in conflict_pos:
            pos = conflict_pos[n]
            last_node = conflict_list[-1]
            conflict_list[pos] = last_node
            conflict_pos[last_node] = pos
            conflict_list.pop()
            del conflict_pos[n]

    for node in nodes:
        if adj_color_table[node][coloring[node]] > 0:
            add_conflict(node)

    for i in range(iterations):
        if not conflict_list:
            return True, coloring  # Valid k-coloring found
            
        # Pick a node involved in a conflict (uniform random from conflicted nodes)
        node_to_move = random.choice(conflict_list)
        
        best_move = None
        min_conflicts = float('inf')
        
        # Try moving node_to_move to a different color
        for new_color in range(k):
            if new_color == coloring[node_to_move]: continue
            if tabu_until[node_to_move][new_color] > i: continue
            
            # The number of new conflicts is exactly the number of neighbors with new_color
            current_count = adj_color_table[node_to_move][new_color]
            
            if current_count < min_conflicts:
                min_conflicts = current_count
                best_move = new_color
                
        if best_move is not None:
            old_color = coloring[node_to_move]
            
            # Ban moving back to old_color for tabu_size iterations
            tabu_until[node_to_move][old_color] = i + tabu_size
            coloring[node_to_move] = best_move
            
            # Incrementally update neighbor tracking avoiding full recalculation
            for nbr in G.neighbors(node_to_move):
                adj_color_table[nbr][old_color] -= 1
                adj_color_table[nbr][best_move] += 1
                
                # Check if neighbor entered or left conflicting state
                if adj_color_table[nbr][coloring[nbr]] > 0:
                    add_conflict(nbr)
                else:
                    remove_conflict(nbr)
                    
            # Check if node_to_move itself is still conflicting
            if adj_color_table[node_to_move][best_move] > 0:
                add_conflict(node_to_move)
            else:
                remove_conflict(node_to_move)

    return False, None