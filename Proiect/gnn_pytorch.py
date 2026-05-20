import numpy as np
import networkx as nx
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy


def _normalise_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalizes each row of a 2D matrix (so that each vector has length 1).
    """
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _initial_features(G: nx.Graph, nodes: list) -> np.ndarray:
    """
    Manually computes: degree, triangles, core number, clustering, eigenvector and betweenness centralities.

    Where it is used: In `gnn_coloring` to define a starting matrix that acts as a pure input for the 1st layer of the GNN.
    """
    
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

    # Build feature matrix using numpy array indexing (no per-node scalar loop)
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
    return X.astype(np.float64)   # (n, 8)


class GCNPyG(nn.Module):
    """
    Implementation of a 2-layer Graph Convolutional Network (GCN), 
    written using PyTorch Geometric for optimized message passing and training.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # GCNConv automatically handles self-loops and symmetric normalization internally
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Executes the Forward Pass through the GNN layers using PyG GCNConv.
        """
        h1 = F.relu(self.conv1(x, edge_index))
        z  = F.relu(self.conv2(h1, edge_index))
        return z

    def train_model(self, X: np.ndarray, A: np.ndarray, epochs: int = 80, 
                    lr: float = 0.01, margin: float = 1.0, n_neg: int = 3, 
                    verbose: bool = False) -> np.ndarray:
        """
        Runs the optimized training loop using PyTorch autograd and GPU acceleration if available.
        """
        print("Training GNN pytorch model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Convert numpy inputs to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

        # Build full edge index for GCN message passing (both directions)
        full_edge_idx = np.argwhere(A > 0).T
        edge_index = torch.tensor(full_edge_idx, dtype=torch.long, device=device)

        # Build upper-triangle edge index for positive pairs hinge loss matching original logic
        pos_edge_idx = np.argwhere(A > 0)
        pos_edge_idx = pos_edge_idx[pos_edge_idx[:, 0] < pos_edge_idx[:, 1]].T
        pos_edge_index = torch.tensor(pos_edge_idx, dtype=torch.long, device=device)

        # SGD Optimizer with momentum matching the original numpy implementation
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        best_loss = float('inf')
        best_model_state = copy.deepcopy(self.state_dict())
        n = X.shape[0]

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            Z = self.forward(X_tensor, edge_index)
            loss = torch.tensor(0.0, device=device)

            # ---- positive pairs: vectorized over graph edges ----
            if pos_edge_index.size(1) > 0:
                zi = Z[pos_edge_index[0]]
                zj = Z[pos_edge_index[1]]
                diff = zi - zj
                dist = torch.norm(diff, p=2, dim=1) + 1e-8
                # Penalize adjacent nodes sharing close embeddings (push them apart)
                hinge = torch.clamp(margin - dist, min=0.0)
                loss = loss + torch.sum(hinge ** 2)

            # ---- negative pairs: sample n_neg * |E| random non-edges ----
            n_target = pos_edge_index.size(1) * n_neg
            cands = torch.randint(0, n, (n_target * 5, 2), device=device)
            mask = (cands[:, 0] != cands[:, 1]) & (A_tensor[cands[:, 0], cands[:, 1]] == 0)
            neg = cands[mask][:n_target]

            if neg.size(0) > 0:
                zi = Z[neg[:, 0]]
                zj = Z[neg[:, 1]]
                diff = zi - zj
                dist = torch.norm(diff, p=2, dim=1) + 1e-8
                # Penalize non-adjacent nodes with distant embeddings (pull closer)
                hinge = torch.clamp(dist - margin / 2.0, min=0.0)
                loss = loss + 0.3 * torch.sum(hinge ** 2)

            # Automatic differentiation backprop
            loss.backward()

            # Gradient clipping by total norm
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_model_state = copy.deepcopy(self.state_dict())

            if verbose and epoch % 20 == 0:
                print(f"    [GCN] epoch {epoch:3d}  loss={loss_val:.4f}")

        # Restore best weights and extract final embeddings on CPU
        self.load_state_dict(best_model_state)
        self.eval()
        with torch.no_grad():
            Z_final = self.forward(X_tensor, edge_index).cpu().numpy()
        
        return Z_final


def _priority_scores(G: nx.Graph, nodes: list, Z: np.ndarray) -> np.ndarray:
    """
    Computes a priority score based on the degree and difficulty of a node mixed with the localized semantic complexity of the centroids 
        derived in the network, to establish that higher-scoring nodes need to be determined earlier.
    """
    A       = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)
    deg_arr = A.sum(axis=1)                          
    Z_norm  = _normalise_rows(Z)                     
    
    nbr_sim_sum = A @ Z_norm                         
    nbr_sim     = (nbr_sim_sum * Z_norm).sum(axis=1)
    deg_norm    = deg_arr / (deg_arr.max() + 1)
    with np.errstate(invalid='ignore'):
        mean_sim = np.where(deg_arr > 0, nbr_sim / deg_arr, 0.0)
    return 0.7 * deg_norm + 0.3 * mean_sim


def _best_color(z_i: np.ndarray, candidates: list, centroids: dict) -> int:
    """
    Taking a given vector `z_i`, matrix-checks the maximum euclidean distance towards the centroids of the 
        different subsets of already established colors; geometrically chooses the farthest label. 
    """
    known = [c for c in candidates if c in centroids]
    if not known:
        return candidates[0]

    C    = np.array([centroids[c] for c in known])  
    dist = np.linalg.norm(z_i - C, axis=1)          
    return known[int(np.argmax(dist))]


def _assign_colors(G: nx.Graph, nodes: list, Z: np.ndarray) -> dict:
    """
    Uses the fully trained GNN model and established priorities to ultimately allocate a valid integer representing 
        the color, totally guided by the spatial position and the center of mass of the centroids generated by the network.        
    """
    scores  = _priority_scores(G, nodes, Z)
    ordered = [nodes[i] for i in np.argsort(-scores)]   
    idx     = {nd: i for i, nd in enumerate(nodes)}

    coloring: dict  = {}
    centroids: dict = {}   
    counts:    dict = {}

    for nd in ordered:
        nbr_colors = {coloring[nb] for nb in G.neighbors(nd) if nb in coloring}
        used        = [c for c in centroids if c not in nbr_colors]
        new_c       = (max(centroids) + 1) if centroids else 0
        candidates  = used if used else [new_c]
        if used:
            candidates.append(new_c)   

        chosen = _best_color(Z[idx[nd]], candidates, centroids)
        coloring[nd] = chosen

        z, cnt = Z[idx[nd]], counts.get(chosen, 0)
        centroids[chosen] = (centroids[chosen] * cnt + z) / (cnt + 1) if chosen in centroids else z.copy()
        counts[chosen]    = cnt + 1

    return coloring


def _ils_improve(G: nx.Graph, coloring: dict, Z: np.ndarray,
                  idx: dict, max_iter: int = 2000) -> dict:
    """
    Modified Iterated Local Search - Locates the color with the rarest members and, by forcing successive valid moves based 
        on distance from the embedding centroids, tries to distribute them to other families, completely dissolving that color from the graph.
    """
    coloring = dict(coloring)
    use      = defaultdict(int, {c: 0 for c in set(coloring.values())})
    for c in coloring.values():
        use[c] += 1

    cents: dict = {};  cnts: dict = {}
    for nd, c in coloring.items():
        z, cnt = Z[idx[nd]], cnts.get(c, 0)
        cents[c] = (cents[c] * cnt + z) / (cnt + 1) if c in cents else z.copy()
        cnts[c]  = cnt + 1

    for _ in range(max_iter):
        if not use:
            break
        rare   = min(use, key=lambda c: use[c])
        r_nds  = [nd for nd, c in coloring.items() if c == rare]

        moved_all = True
        for nd in r_nds:
            nbr_c  = {coloring[nb] for nb in G.neighbors(nd)}
            cands  = [c for c in use if c != rare and c not in nbr_c]
            if not cands:
                moved_all = False;  break

            chosen = _best_color(Z[idx[nd]], cands, cents)

            z       = Z[idx[nd]]
            old_cnt = cnts[rare]
            if old_cnt > 1:
                cents[rare] = (cents[rare] * old_cnt - z) / (old_cnt - 1)
                cnts[rare]  = old_cnt - 1
            new_cnt       = cnts.get(chosen, 0)
            cents[chosen] = (cents[chosen] * new_cnt + z) / (new_cnt + 1) if chosen in cents else z.copy()
            cnts[chosen]  = new_cnt + 1

            coloring[nd]  = chosen
            use[rare]    -= 1
            use[chosen]   = use.get(chosen, 0) + 1

        if moved_all:
            del use[rare]
        else:
            break

    palette = sorted(set(coloring.values()))
    remap   = {old: new for new, old in enumerate(palette)}
    return {nd: remap[c] for nd, c in coloring.items()}


def _safe_fix(G: nx.Graph, coloring: dict) -> dict:
    """
    Brute-force safety validation fail-safe. 
    """
    coloring = dict(coloring)
    for nd in G.nodes():
        nbr_colors = {coloring[nb] for nb in G.neighbors(nd) if nb in coloring}
        if coloring.get(nd) in nbr_colors:
            c = 0
            while c in nbr_colors:
                c += 1
            coloring[nd] = c
    return coloring


def gnn_coloring(G: nx.Graph,
                  hidden_dim: int = 32,
                  out_dim:    int = 16,
                  epochs:     int = 60,
                  lr:       float = 0.01,
                  margin:   float = 1.0,
                  ils_iter:   int = 500,
                  seed:       int = 42,
                  verbose:   bool = False):
    """
    The main API function that links all the methods in the file.
    """
    if len(G) == 0:
        return 0, {}

    nodes = sorted(G.nodes())
    n     = len(nodes)

    if verbose:
        print(f"  [GNN] Building features for {n} nodes …")
    X = _initial_features(G, nodes)                       
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)            

    actual_epochs = epochs
    if n > 500:   actual_epochs = max(20, epochs // 3)
    elif n > 200: actual_epochs = max(30, epochs // 2)

    # Set reproducibility seed for PyTorch
    torch.manual_seed(seed)

    # Instantiate and train the PyTorch Geometric model
    gcn = GCNPyG(X.shape[1], hidden_dim, out_dim)
    Z = gcn.train_model(X, A, epochs=actual_epochs, lr=lr, margin=margin, verbose=verbose)

    Z = _normalise_rows(Z)                           
    
    coloring = _assign_colors(G, nodes, Z)
    idx = {nd: i for i, nd in enumerate(nodes)}
    coloring = _ils_improve(G, coloring, Z, idx, max_iter=ils_iter)

    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            coloring = _safe_fix(G, coloring)
            break

    return len(set(coloring.values())), coloring