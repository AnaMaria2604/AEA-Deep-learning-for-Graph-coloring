import numpy as np
import networkx as nx
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import copy
import matplotlib.pyplot as plt

def _normalise_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # same
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def _initial_features(G: nx.Graph, nodes: list) -> np.ndarray:
    # same
    n = len(nodes)
    degrees   = dict(G.degree())
    max_deg   = max(degrees.values()) or 1
    clustering = nx.clustering(G)
    triangles  = nx.triangles(G)
    max_tri    = max(triangles.values()) + 1

    try:
        cores    = nx.core_number(G);  max_core = max(cores.values()) + 1
    except Exception:
        cores    = {nd: 0 for nd in nodes};  max_core = 1

    try:
        eig_cent = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eig_cent = {nd: 0.0 for nd in nodes}

    between = (nx.betweenness_centrality(G, normalized=True)
               if n <= 300 else {nd: 0.0 for nd in nodes})

    deg_arr  = np.array([degrees[nd] for nd in nodes], dtype=np.float64)
    nd_norm  = deg_arr / max_deg
    X = np.column_stack([
        nd_norm,
        [clustering.get(nd, 0.0) for nd in nodes],
        np.log(deg_arr + 1) / np.log(max_deg + 1 + 1e-8),
        nd_norm ** 2,
        [triangles.get(nd, 0) / max_tri for nd in nodes],
        [cores.get(nd, 0) / max_core   for nd in nodes],
        [eig_cent.get(nd, 0.0)         for nd in nodes],
        [between.get(nd, 0.0)          for nd in nodes],
    ])
    return X.astype(np.float64)


class GATPyG(nn.Module):
    """
    Graph Attention Network (GAT) built via PyTorch Geometric.
    Uses multi-head attention mechanisms to dynamic-weight neighborhood relevance.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.conv1(x, edge_index))
        z  = F.relu(self.conv2(h1, edge_index))
        return z

    def train_model(self, X: np.ndarray, A: np.ndarray, epochs: int = 80, 
                    lr: float = 0.01, margin: float = 1.0, n_neg: int = 3, 
                    verbose: bool = False) -> np.ndarray:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

        full_edge_idx = np.argwhere(A > 0).T
        edge_index = torch.tensor(full_edge_idx, dtype=torch.long, device=device)

        pos_edge_idx = np.argwhere(A > 0)
        pos_edge_idx = pos_edge_idx[pos_edge_idx[:, 0] < pos_edge_idx[:, 1]].T
        pos_edge_index = torch.tensor(pos_edge_idx, dtype=torch.long, device=device)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        best_loss = float('inf')
        best_model_state = copy.deepcopy(self.state_dict())
        n = X.shape[0]

        deg_np = A.sum(axis=1)
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            Z = self.forward(X_tensor, edge_index)
            loss = torch.tensor(0.0, device=device)

            if pos_edge_index.size(1) > 0:
                u_idx = pos_edge_index[0].cpu().numpy()
                v_idx = pos_edge_index[1].cpu().numpy()
                weights = np.log(deg_np[u_idx] + deg_np[v_idx] + 2.0)
                weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
                
                zi = Z[pos_edge_index[0]]
                zj = Z[pos_edge_index[1]]
                diff = zi - zj
                dist = torch.norm(diff, p=2, dim=1) + 1e-8
                
                hinge = torch.clamp(margin - dist, min=0.0)
                loss = loss + torch.sum(weights_t * (hinge ** 2))

            n_target = pos_edge_index.size(1) * n_neg
            cands = torch.randint(0, n, (n_target * 5, 2), device=device)
            mask = (cands[:, 0] != cands[:, 1]) & (A_tensor[cands[:, 0], cands[:, 1]] == 0)
            neg = cands[mask][:n_target]

            if neg.size(0) > 0:
                zi = Z[neg[:, 0]]
                zj = Z[neg[:, 1]]
                diff = zi - zj
                dist = torch.norm(diff, p=2, dim=1) + 1e-8
                
                hinge = torch.clamp(dist - margin / 2.0, min=0.0)
                loss = loss + 0.3 * torch.sum(hinge ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_model_state = copy.deepcopy(self.state_dict())

            if verbose and epoch % 20 == 0:
                print(f"    [GAT-PyG] epoch {epoch:3d}  loss={loss_val:.4f}")

        self.load_state_dict(best_model_state)
        self.eval()
        with torch.no_grad():
            Z_final = self.forward(X_tensor, edge_index).cpu().numpy()
        
        return Z_final


def _best_color(z_i: np.ndarray, candidates: list, centroids: dict) -> int:
    # same
    known = [c for c in candidates if c in centroids]
    if not known:
        return candidates[0]

    C    = np.array([centroids[c] for c in known])  
    dist = np.linalg.norm(z_i - C, axis=1)          
    return known[int(np.argmax(dist))]


def _single_dsatur_run(G: nx.Graph, nodes: list, Z: np.ndarray, node_to_idx: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    coloring: dict = {}
    centroids: dict = {}
    counts: dict = {}
    
    nbr_colors_set = {nd: set() for nd in nodes}
    degrees = dict(G.degree())
    uncolored = set(nodes)
    
    while uncolored:
        max_sat = -1
        candidates = []
        for nd in uncolored:
            sat = len(nbr_colors_set[nd])
            if sat > max_sat:
                max_sat = sat
                candidates = [nd]
            elif sat == max_sat:
                candidates.append(nd)
                
        if len(candidates) == 1:
            best_nd = candidates[0]
        else:
            degs = np.array([degrees[c] for c in candidates], dtype=np.float32)
            probs = np.exp(degs / (degs.max() + 1e-5))
            probs /= probs.sum()
            best_nd = rng.choice(candidates, p=probs)
                
        uncolored.remove(best_nd)
        forbidden_colors = {coloring[nb] for nb in G.neighbors(best_nd) if nb in coloring}
        
        used = [c for c in centroids if c not in forbidden_colors]
        new_c = (max(centroids) + 1) if centroids else 0
        color_opts = used if used else [new_c]
        if used:
            color_opts.append(new_c)
            
        chosen = _best_color(Z[node_to_idx[best_nd]], color_opts, centroids)
        coloring[best_nd] = chosen
        
        z, cnt = Z[node_to_idx[best_nd]], counts.get(chosen, 0)
        centroids[chosen] = (centroids[chosen] * cnt + z) / (cnt + 1) if chosen in centroids else z.copy()
        counts[chosen] = cnt + 1
        
        for nb in G.neighbors(best_nd):
            if nb in nbr_colors_set:
                nbr_colors_set[nb].add(chosen)
                
    return coloring


def _assign_colors(G: nx.Graph, nodes: list, Z: np.ndarray, starts: int = 15) -> dict:
    """
    Advanced Multi-Start Stochastic DSATUR optimization wrapper.
    Ruan multiple times and preserves the state utilizing the minimum unique palette footprint.
    """
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    best_coloring = None
    min_colors = float('inf')
    
    for i in range(starts):
        current_coloring = _single_dsatur_run(G, nodes, Z, node_to_idx, seed=42 + i)
        num_colors = len(set(current_coloring.values()))
        if num_colors < min_colors:
            min_colors = num_colors
            best_coloring = current_coloring
            
    return best_coloring


def _ils_improve_with_tabu(G: nx.Graph, coloring: dict, max_iter: int = 3000) -> dict:
    nodes = list(G.nodes())
    current_coloring = dict(coloring)
    num_colors = len(set(current_coloring.values()))
    
    best_valid_coloring = dict(current_coloring)
    target_colors = num_colors - 1
    
    while target_colors >= 2:
        for nd in nodes:
            if current_coloring[nd] >= target_colors:
                current_coloring[nd] = np.random.randint(0, target_colors)
                
        tabu_list = {}
        improved = False
        
        for iteration in range(max_iter):
            conflicts = []
            for u, v in G.edges():
                if current_coloring[u] == current_coloring[v]:
                    conflicts.append(u)
                    conflicts.append(v)
            
            if not conflicts:
                best_valid_coloring = dict(current_coloring)
                target_colors -= 1
                improved = True
                break
                
            conf_nodes = [n for n in conflicts if n not in tabu_list or tabu_list[n] < iteration]
            if not conf_nodes:
                conf_nodes = conflicts 
                
            nd_to_move = np.random.choice(conf_nodes)
            
            best_c = current_coloring[nd_to_move]
            min_local_conf = float('inf')
            
            for c in range(target_colors):
                local_conf = sum(1 for nb in G.neighbors(nd_to_move) if current_coloring[nb] == c)
                if local_conf < min_local_conf:
                    min_local_conf = local_conf
                    best_c = c
            
            current_coloring[nd_to_move] = best_c
            tabu_list[nd_to_move] = iteration + np.random.randint(5, 15) # Blocăm nodul temporar
            
        if not improved:
            break
            
    return best_valid_coloring

def _safe_fix(G: nx.Graph, coloring: dict) -> dict:
    # same
    coloring = dict(coloring)
    for nd in G.nodes():
        nbr_colors = {coloring[nb] for nb in G.neighbors(nd) if nb in coloring}
        if coloring.get(nd) in nbr_colors:
            c = 0
            while c in nbr_colors:
                c += 1
            coloring[nd] = c
    return coloring

def plot_colored_graph(G: nx.Graph, coloring: dict, title: str = ""):
    plt.figure(figsize=(10, 8))

    nodes = list(G.nodes())
    node_colors = [coloring[node] for node in nodes]
    
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors, 
                           cmap=plt.cm.rainbow, node_size=600, edgecolors='black')
    
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold")
    
    num_colors = len(set(coloring.values()))
    plt.title(f"{title}\nNumber of colors: {num_colors}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
   
def gnn_coloring(G: nx.Graph,
                  hidden_dim: int = 128,
                  out_dim:    int = 16,
                  epochs:     int = 60,
                  lr:       float = 0.01,
                  margin:   float = 1.0,
                  ils_iter:   int = 500,
                  dsatur_starts: int = 15,
                  seed:       int = 42,
                  verbose:   bool = False,
                ):
    if len(G) == 0:
        return 0, {}

    nodes = sorted(G.nodes())
    n     = len(nodes)

    if verbose:
        print(f"  [GNN] Generating initial feature matrix for {n} nodes...")
    X = _initial_features(G, nodes)                       
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)            

    actual_epochs = epochs
    if n > 500:   actual_epochs = max(100, epochs // 3)
    elif n > 200: actual_epochs = max(200, epochs // 2)

    torch.manual_seed(seed)

    gat_model = GATPyG(in_dim=X.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, heads=8)
    Z = gat_model.train_model(X, A, epochs=actual_epochs, lr=lr, margin=margin, verbose=verbose)

    Z = _normalise_rows(Z)                           
    
    # Runs the advanced multi-start probabilistic DSATUR pipeline
    coloring = _assign_colors(G, nodes, Z, starts=dsatur_starts)
    
    coloring = _ils_improve_with_tabu(G, coloring, max_iter=ils_iter * 2)

    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            coloring = _safe_fix(G, coloring)
            break

    plot_colored_graph(G, coloring, title=f"")
    return len(set(coloring.values())), coloring