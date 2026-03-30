import networkx as nx
import collections

def greedy_sdl_coloring(G):
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x))
    coloring = {}
    
    for node in nodes:
        neighbor_colors = {coloring[neighbor] for neighbor in G.neighbors(node) if neighbor in coloring}
        
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[node] = color
        
    return len(set(coloring.values())), coloring