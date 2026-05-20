# Graph Coloring Benchmark Results

> **GNN** � 2-layer GCN (numpy, self-supervised coloring loss )  
> **Tabu** � TabuCol local search, independent upper bound from nx largest-first greedy  
> All values averaged over 15 independent runs.

## Set1-myciel

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| myciel3 | 11 | 20 | 4.00 | 4 | 4 | 0.00 | 0.2696 | 4 |
| myciel4 | 23 | 71 | 5.00 | 5 | 5 | 0.00 | 0.2450 | 5 |
| myciel5 | 47 | 236 | 6.00 | 6 | 6 | 0.00 | 0.3294 | 6 |
| myciel6 | 95 | 755 | 7.00 | 7 | 7 | 0.00 | 0.6492 | 7 |
| myciel7 | 191 | 2360 | 8.00 | 8 | 8 | 0.00 | 1.2921 | 8 |

## Set2-queen

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| queen5_5 | 25 | 160 | 5.00 | 5 | 5 | 0.00 | 0.3176 | 5 |
| queen6_6 | 36 | 290 | 8.00 | 8 | 8 | 0.00 | 0.3716 | 7 |
| queen7_7 | 49 | 476 | 8.80 | 8 | 9 | 0.40 | 0.4501 | 7 |

## Set3

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| anna | 138 | 493 | 11.00 | 11 | 11 | 0.00 | 0.5572 | 11 |
| david | 87 | 406 | 11.00 | 11 | 11 | 0.00 | 0.5218 | 11 |
| jean | 80 | 254 | 10.00 | 10 | 10 | 0.00 | 0.4092 | 10 |
| huck | 74 | 301 | 11.00 | 11 | 11 | 0.00 | 0.4193 | 11 |
| homer | 561 | 1629 | 13.00 | 13 | 13 | 0.00 | 2.0427 | 13 |
| games120 | 120 | 638 | 9.00 | 9 | 9 | 0.00 | 0.5756 | 9 |

## Set4-medium

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| flat300_28_0 | 300 | 21695 | 40.67 | 40 | 42 | 0.60 | 39.7848 | 28 |
| dsjc250.5 | 250 | 15668 | 36.60 | 36 | 37 | 0.49 | 21.2318 | 28 |
| le450_25c | 450 | 17343 | 28.27 | 28 | 29 | 0.44 | 37.2790 | 25 |

## Set5-miles

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| miles250 | 128 | 387 | 8.00 | 8 | 8 | 0.00 | 2.1158 | 8 |
| miles500 | 128 | 1170 | 20.00 | 20 | 20 | 0.00 | 3.4271 | 20 |
| miles750 | 128 | 2113 | 31.00 | 31 | 31 | 0.00 | 4.9393 | 31 |
| miles1000 | 128 | 3216 | 42.00 | 42 | 42 | 0.00 | 3.6998 | 42 |
| miles1500 | 128 | 5198 | 73.00 | 73 | 73 | 0.00 | 5.0852 | 73 |

## Set6-large

| Instance | Nodes | Edges | GNN mean k | GNN min k | GNN max k | GNN stdev | GNN time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| dsjc500.9 | 500 | 112437 | 160.33 | 160 | 161 | 0.47 | 107.3679 | 126 |
| dsjc1000.5 | 1000 | 249826 | 114.07 | 113 | 115 | 0.68 | 139.2819 | 85 |

