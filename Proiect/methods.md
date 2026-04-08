# Explanation of Graph Coloring Methods


## 1. Graph Neural Network (GNN) Method

### What it is
- The GNN method leverages a **2-layer Graph Convolutional Network (GCN)** designed to produce low-dimensional embeddings for each node in a graph.

-  Rather than predicting colors directly, **it learns to map the graph's structural topology into a semantic embedding space**. 
- The training is entirely **self-supervised**, meaning it does not require pre-colored graphs as ground truth data. Instead, it relies on a custom **geometric loss function** that actively pulls non-adjacent nodes closer together (encouraging same colors) while pushing adjacent nodes apart by a unified margin (forcing different colors).

- **Once the network generates these spatial embeddings, a downstream processing pipeline uses these continuous coordinate representations to cluster the nodes and safely allocate integer colors, essentially reducing the combinatorial coloring problem into geometric distance clustering.**

### Iterations
- **GNN Training (Epochs):** To avoid unnecessary computation and prevent overfitting, **the system scales** the learning epochs dynamically based on the topological size of the graph
- **Iterated Local Search (ILS):** After the initial color alignment, a subsequent refinement phase (Iterated Local Search) runs for up to **500 iterations**. During this loop, the algorithm aggressively targets the rarest color class in the graph and attempts to dissolve it entirely by reassigning its nodes to other legal colors.

### Optimizations Applied and Why
- **Vectorized Hinge Loss over Edges:** 
  - *What:* The backpropagation computes the hinge loss for every single edge simultaneously without looping through individual node pairs. It also samples negative random subsets (non-edges) identically.
  - *Why:* Iterating node-to-node would choke the computation time. Vectorizing the cost over the adjacency matrix provides massive continuous chunks of arithmetic for the process scheduler.


---

## 2. TabuCol Method

### What it is
- TabuCol is an application of **Tabu Search**, a highly successful metaheuristic algorithm tailored specifically for the vertex coloring problem. 

- The algorithm continually endeavors to discover a 0-conflict state utilizing exactly *k* colors. 

- The process initiates with a random assignment across *k* colors. Naturally, this generates numerous edge conflicts (where adjacent nodes erroneously share identical colors). 

- TabuCol iteratively cycles through conflicting nodes, flipping their colors to locally minimize the immediate number of conflicts. 

- To evade infinite loops where the algorithm alternates back and forth between two identical states (getting trapped in local optima), a short-term memory mechanism called the "Tabu List" records flipped colors and forbids a node from accepting a recently discarded color for a certain duration.

### Iterations
- **Core Loop Processing:** Every attempt to format the graph with *k* colors runs for up to **50,000 iterations** without halting. If 0 conflicts are reached earlier, it terminates immediately.
- **Independent Restarts:** If the iterations are exhausted and conflicts remain, the algorithm automatically enacts up to **3 totally random independent restarts** on the exact same *k* before officially declaring it unsolvable. 
- **Sequential Optimization Outer Loop:** The architecture progressively decrements *k* (seeded from a rapid greedy theoretical upper limit bound estimation), executing the iterations at each target *k* until it can no longer compress the chromatic number.

### Optimizations Applied and Why
- **Dynamic Tabu Tenure Formula:** 
  - *What:* The lifespan of how long a color remains forbidden ("tabu") is dynamically tied to the number of colors *k* (explicitly targeting a recommended factor `0.6 * k`). 
  - *Why:* Establishing a stagnant integer for the tabu list yields poor results. Scaling proportionally against *k* grants exact equilibrium between the exploitation of good paths and the forced exploration of alternative color combinations without overly constricting the maneuverability.
- **O(1) Adjacency Color Lookup Map:** 
  - *What:* TabuCol utilizes an overarching state grid (`adj_color_table`) tracking the precise quantity of surrounding colors attached to every localized node.
  - *Why:* Traditionally, deducing neighbor conflict frequencies imposes `O(Neighbor-Degree)` time complexity. Establishing a pre-maintained matrix simplifies every query to instant `O(1)` memory lookups, which heavily compresses processing time given the operation functions millions of times per run.
- **O(1) Live Conflict Isolation Array:** 
  - *What:* Contains an exclusively managed active array map dictating exactly which targeted nodes are participating in unresolved conflicts.
  - *Why:* In each iteration, TabuCol uniformely selects a random node that is in a conflict natively. Without an actively managed subset logic, the entire matrix space would be iterated every tick to manually discern conflict participations. Structuring index arrays directly slashes the operation cost drastically.
