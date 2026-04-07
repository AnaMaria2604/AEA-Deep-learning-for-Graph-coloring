import numpy as np
import networkx as nx
from collections import defaultdict


def _normalise_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalizes each row of a 2D matrix (so that each vector has length 1).
    """
    
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _normalised_adjacency(G: nx.Graph, nodes: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the simple adjacency matrix (A) and its normalized variant (A_hat) 
    """

    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)
    A_tilde = A + np.eye(len(nodes))
    d_inv_sqrt = np.where(A_tilde.sum(1) > 0, A_tilde.sum(1) ** -0.5, 0.0)

    # outer product broadcast: D^{-1/2} @ A_tilde @ D^{-1/2}
    A_hat = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]
    return A, A_hat


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
    idx      = {nd: i for i, nd in enumerate(nodes)}
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


class GCNNumpy:
    """
    Implementation of a 2-layer Graph Convolutional Network (GCN), written purely using NumPy operations.

    The mechanism is self-supervised: it directly learns spatial coordinates by pushing adjacent nodes apart 
        and pulling totally disparate nodes closer.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42):
        rng  = np.random.default_rng(seed)
        lim1 = np.sqrt(6.0 / (in_dim + hidden_dim))
        lim2 = np.sqrt(6.0 / (hidden_dim + out_dim))
        self.W1 = rng.uniform(-lim1, lim1, (in_dim, hidden_dim))
        self.W2 = rng.uniform(-lim2, lim2, (hidden_dim, out_dim))

    def forward(self, A_hat: np.ndarray, X: np.ndarray):
        """
        Executes the "Forward Pass" through the GNN, evaluating mathematical formulas directly from matrices. 
        """

        H1 = np.maximum(0.0, A_hat @ X  @ self.W1)   # (n, hidden_dim)
        Z  = np.maximum(0.0, A_hat @ H1 @ self.W2)   # (n, out_dim)
        return Z, H1

    def _loss_and_grads(self, A_hat, X, A, edges, n_neg=3, margin=1.0):
        """
        Computes the vectorized Loss function (hinge loss) that pulls distant neighbors 
            and the Gradients for Backpropagation aimed at modifying the GNN iteratively.
        """

        Z, H1 = self.forward(A_hat, X)
        n     = Z.shape[0]
        dL_dZ = np.zeros_like(Z)
        loss  = 0.0

        # ---- positive pairs: vectorised over edges ----
        if len(edges):
            zi   = Z[edges[:, 0]]            # (E, d)
            zj   = Z[edges[:, 1]]            # (E, d)
            diff = zi - zj                   # (E, d)
            dist = np.linalg.norm(diff, axis=1) + 1e-8  # (E,)
            # want dist >= margin  →  penalise when dist < margin
            hinge = np.maximum(0.0, margin - dist)       # (E,)
            loss  += float(np.sum(hinge ** 2))

            # gradient w.r.t. Z[i] and Z[j] for active hinges
            scale = -2.0 * hinge / dist                  # (E,)  zero where hinge=0
            grad  = scale[:, None] * diff                # (E, d)
            np.add.at(dL_dZ, edges[:, 0],  grad)
            np.add.at(dL_dZ, edges[:, 1], -grad)

        # ---- negative pairs: sample n_neg * |E| random non-edges ----
        rng      = np.random.default_rng()
        n_target = len(edges) * n_neg

        # over-sample and filter
        cands    = rng.integers(0, n, size=(n_target * 5, 2))
        mask     = (cands[:, 0] != cands[:, 1]) & (A[cands[:, 0], cands[:, 1]] == 0)
        neg      = cands[mask][:n_target]
        if len(neg):
            zi   = Z[neg[:, 0]]
            zj   = Z[neg[:, 1]]
            diff = zi - zj
            dist = np.linalg.norm(diff, axis=1) + 1e-8

            # want dist <= margin/2  →  penalise when dist > margin/2
            hinge = np.maximum(0.0, dist - margin / 2.0)
            loss  += 0.3 * float(np.sum(hinge ** 2))
            scale  = 0.3 * 2.0 * hinge / dist
            grad   = scale[:, None] * diff
            
            np.add.at(dL_dZ, neg[:, 0],  grad)
            np.add.at(dL_dZ, neg[:, 1], -grad)

        # ---- backprop through layer 2 ----
        dZ_pre  = dL_dZ * (Z > 0)
        dL_dW2  = H1.T @ (A_hat.T @ dZ_pre)
        dL_dH1  = (A_hat.T @ dZ_pre) @ self.W2.T

        # ---- backprop through layer 1 ----
        dH1_pre = dL_dH1 * (H1 > 0)
        dL_dW1  = X.T @ (A_hat.T @ dH1_pre)

        return loss, dL_dW1, dL_dW2

    def train(self, A_hat, X, A, epochs=80, lr=0.01, margin=1.0, verbose=False):
        """
        Runs the training for the GCNNumpy object in a loop for the given epochs. 
        
        Decreases the Loss by calculating the optimal weights using Backpropagation.
        """

        # Pre-extract upper-triangle edge indices once (reused every epoch)
        edge_idx = np.argwhere(A > 0)
        edge_idx = edge_idx[edge_idx[:, 0] < edge_idx[:, 1]]

        v1, v2   = np.zeros_like(self.W1), np.zeros_like(self.W2)
        momentum = 0.9
        best_loss, best_W1, best_W2 = float('inf'), self.W1.copy(), self.W2.copy()

        for epoch in range(epochs):
            loss, gW1, gW2 = self._loss_and_grads(A_hat, X, A, edge_idx, margin=margin)

            # gradient clipping
            for g in (gW1, gW2):
                gnorm = np.linalg.norm(g)
                if gnorm > 5.0:
                    g *= 5.0 / gnorm

            v1 = momentum * v1 - lr * gW1;  self.W1 += v1
            v2 = momentum * v2 - lr * gW2;  self.W2 += v2

            if loss < best_loss:
                best_loss = loss;  best_W1 = self.W1.copy();  best_W2 = self.W2.copy()

            if verbose and epoch % 20 == 0:
                print(f"    [GCN] epoch {epoch:3d}  loss={loss:.4f}")

        self.W1, self.W2 = best_W1, best_W2


def _priority_scores(G: nx.Graph, nodes: list, Z: np.ndarray) -> np.ndarray:
    """
    Computes a priority score based on the degree and difficulty of a node mixed with the localized semantic complexity of the centroids 
        derived in the network, to establish that higher-scoring nodes need to be determined earlier.
        
    Where it is used: Only in the `_assign_colors` function, with the strict role of deciding the descending sorting order.
    """
    A       = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)
    deg_arr = A.sum(axis=1)                          
    Z_norm  = _normalise_rows(Z)                     
    
    # (A @ Z_norm)[i] = sum of normalised embeddings of i's neighbours
    # dotted with Z_norm[i] gives total cosine similarity, /deg gives mean
    nbr_sim_sum = A @ Z_norm                         
    
    # element-wise dot of each row with corresponding Z_norm row → (n,)
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
            candidates.append(new_c)   # always allow opening a fresh colour

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

    # Build per-colour centroid embeddings
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

            # update centroids (incremental remove / add)
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
    idx   = {nd: i for i, nd in enumerate(nodes)}

    if verbose:
        print(f"  [GNN] Building features for {n} nodes …")
    X = _initial_features(G, nodes)                       

    A, A_hat = _normalised_adjacency(G, nodes)            

    actual_epochs = epochs
    if n > 500:   actual_epochs = max(20, epochs // 3)
    elif n > 200: actual_epochs = max(30, epochs // 2)

    gcn = GCNNumpy(X.shape[1], hidden_dim, out_dim, seed=seed)
    gcn.train(A_hat, X, A, epochs=actual_epochs, lr=lr, margin=margin, verbose=verbose)

    Z, _ = gcn.forward(A_hat, X)
    Z     = _normalise_rows(Z)                           

    
    coloring = _assign_colors(G, nodes, Z)
    coloring = _ils_improve(G, coloring, Z, idx, max_iter=ils_iter)

    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            coloring = _safe_fix(G, coloring)
            break

    return len(set(coloring.values())), coloring