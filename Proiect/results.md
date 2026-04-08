# Graph Coloring Benchmark Results

> **GNN** — 2-layer GCN (numpy, self-supervised coloring loss )  
> **Tabu** — TabuCol local search, independent upper bound from nx largest-first greedy  
> All values averaged over 15 independent runs.

## Set1-myciel

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| myciel3 | 11 | 20 | 4.00 | 4 | 4 | 0.00 | 0.0157 | 4.00 | 4 | 4 | 0.00 | 0.2864 | 4 |
| myciel4 | 23 | 71 | 5.00 | 5 | 5 | 0.00 | 0.0243 | 5.00 | 5 | 5 | 0.00 | 0.4291 | 5 |
| myciel5 | 47 | 236 | 6.00 | 6 | 6 | 0.00 | 0.0528 | 6.00 | 6 | 6 | 0.00 | 0.6091 | 6 |
| myciel6 | 95 | 755 | 7.00 | 7 | 7 | 0.00 | 0.1830 | 7.00 | 7 | 7 | 0.00 | 0.9516 | 7 |
| myciel7 | 191 | 2360 | 8.00 | 8 | 8 | 0.00 | 0.5788 | 8.00 | 8 | 8 | 0.00 | 1.4826 | 8 |

## Set2-queen

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| queen5_5 | 25 | 160 | 6.13 | 5 | 8 | 1.26 | 0.0384 | 5.33 | 5 | 6 | 0.47 | 0.3026 | 5 |
| queen6_6 | 36 | 290 | 10.00 | 10 | 10 | 0.00 | 0.0601 | 7.73 | 7 | 8 | 0.44 | 0.7252 | 7 |
| queen7_7 | 49 | 476 | 12.53 | 11 | 15 | 1.26 | 0.0890 | 8.00 | 8 | 8 | 0.00 | 0.8997 | 7 |

## Set3

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anna | 138 | 493 | 11.00 | 11 | 11 | 0.00 | 0.1299 | 11.00 | 11 | 11 | 0.00 | 1.4365 | 11 |
| david | 87 | 406 | 11.00 | 11 | 11 | 0.00 | 0.0918 | 11.00 | 11 | 11 | 0.00 | 1.2292 | 11 |
| jean | 80 | 254 | 10.00 | 10 | 10 | 0.00 | 0.0648 | 10.00 | 10 | 10 | 0.00 | 0.6908 | 10 |
| huck | 74 | 301 | 11.00 | 11 | 11 | 0.00 | 0.0686 | 11.00 | 11 | 11 | 0.00 | 0.6434 | 11 |
| homer | 561 | 1629 | 13.00 | 13 | 13 | 0.00 | 0.2577 | 13.00 | 13 | 13 | 0.00 | 1.1747 | 13 |
| games120 | 120 | 638 | 9.13 | 9 | 10 | 0.34 | 0.1453 | 9.00 | 9 | 9 | 0.00 | 0.6182 | 9 |

## Set4-medium

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat300_28_0 | 300 | 21695 | 47.27 | 45 | 49 | 1.06 | 4.2284 | 35.80 | 35 | 36 | 0.40 | 8.7718 | 28 |
| dsjc250.5 | 250 | 15668 | 42.13 | 41 | 44 | 0.81 | 2.2571 | 32.13 | 32 | 33 | 0.34 | 6.2456 | 28 |
| le450_25c | 450 | 17343 | 33.07 | 32 | 34 | 0.57 | 3.0228 | 29.00 | 29 | 29 | 0.00 | 3.9931 | 25 |

## Set5-miles

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| miles250 | 128 | 387 | 8.33 | 8 | 9 | 0.47 | 0.0949 | 8.00 | 8 | 8 | 0.00 | 0.6660 | 8 |
| miles500 | 128 | 1170 | 20.07 | 20 | 21 | 0.25 | 0.2361 | 20.00 | 20 | 20 | 0.00 | 1.3744 | 20 |
| miles750 | 128 | 2113 | 32.00 | 32 | 32 | 0.00 | 0.3967 | 31.33 | 31 | 32 | 0.47 | 1.2297 | 31 |
| miles1000 | 128 | 3216 | 44.60 | 44 | 45 | 0.49 | 0.5896 | 42.67 | 42 | 43 | 0.47 | 2.1177 | 42 |
| miles1500 | 128 | 5198 | 73.00 | 73 | 73 | 0.00 | 0.9148 | 73.00 | 73 | 73 | 0.00 | 3.8054 | 73 |

## Set6-large

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dsjc500.9 | 500 | 112437 | 177.47 | 174 | 181 | 1.86 | 13.0093 | 139.47 | 138 | 141 | 0.81 | 54.0298 | 126 |
| dsjc1000.5 | 1000 | 249826 | 131.33 | 130 | 135 | 1.35 | 40.3504 | 102.67 | 101 | 104 | 0.70 | 63.5095 | 85 |

