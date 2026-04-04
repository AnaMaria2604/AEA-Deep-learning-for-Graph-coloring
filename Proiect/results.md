# Graph Coloring Benchmark Results

> **GNN** — 2-layer GCN (numpy, self-supervised coloring loss )  
> **Tabu** — TabuCol local search, independent upper bound from nx largest-first greedy  
> All values averaged over 15 independent runs.

## Set1-myciel

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| myciel3 | 11 | 20 | 4.00 | 4 | 4 | 0.00 | 0.0155 | 4.00 | 4 | 4 | 0.00 | 0.9780 | 4 |
| myciel4 | 23 | 71 | 5.00 | 5 | 5 | 0.00 | 0.0241 | 5.00 | 5 | 5 | 0.00 | 2.7716 | 5 |
| myciel5 | 47 | 236 | 6.00 | 6 | 6 | 0.00 | 0.0525 | 6.00 | 6 | 6 | 0.00 | 7.4057 | 6 |
| myciel6 | 95 | 755 | 7.00 | 7 | 7 | 0.00 | 0.1842 | 7.00 | 7 | 7 | 0.00 | 20.4277 | 7 |
| myciel7 | 191 | 2360 | 8.07 | 8 | 9 | 0.25 | 0.5340 | 8.00 | 8 | 8 | 0.00 | 58.8443 | 8 |

