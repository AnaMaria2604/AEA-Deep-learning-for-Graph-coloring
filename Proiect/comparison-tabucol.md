# Graph Coloring Benchmark Results

> **GNN** — 2-layer GCN (numpy, self-supervised coloring loss )  
> **Tabu** — TabuCol local search, independent upper bound from nx largest-first greedy  
> All values averaged over 15 independent runs.

## Set1-myciel

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| myciel3 | 11 | 20 | 4.00 | 4 | 4 | 0.00 | 0.3966 | 4.00 | 4 | 4 | 0.00 | 1.5390 | 4 |
| myciel4 | 23 | 71 | 5.00 | 5 | 5 | 0.00 | 0.4094 | 5.00 | 5 | 5 | 0.00 | 2.9801 | 5 |
| myciel5 | 47 | 236 | 6.00 | 6 | 6 | 0.00 | 0.5914 | 6.00 | 6 | 6 | 0.00 | 6.8712 | 6 |
| myciel6 | 95 | 755 | 7.00 | 7 | 7 | 0.00 | 0.9240 | 7.00 | 7 | 7 | 0.00 | 19.3267 | 7 |
| myciel7 | 191 | 2360 | 8.00 | 8 | 8 | 0.00 | 1.5428 | 8.00 | 8 | 8 | 0.00 | 51.9272 | 8 |

## Set2-queen

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| queen5_5 | 25 | 160 | 5.00 | 5 | 5 | 0.00 | 0.4519 | 5.00 | 5 | 5 | 0.00 | 2.7056 | 5 |
| queen6_6 | 36 | 290 | 8.00 | 8 | 8 | 0.00 | 0.7187 | 8.00 | 8 | 8 | 0.00 | 7.0411 | 7 |
| queen7_7 | 49 | 476 | 8.00 | 8 | 8 | 0.00 | 0.8438 | 8.00 | 8 | 8 | 0.00 | 9.6809 | 7 |

## Set3

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anna | 138 | 493 | 11.00 | 11 | 11 | 0.00 | 1.3796 | 11.00 | 11 | 11 | 0.00 | 18.4599 | 11 |
| david | 87 | 406 | 11.00 | 11 | 11 | 0.00 | 1.1506 | 11.00 | 11 | 11 | 0.00 | 13.5350 | 11 |
| jean | 80 | 254 | 10.00 | 10 | 10 | 0.00 | 0.6558 | 10.00 | 10 | 10 | 0.00 | 9.7445 | 10 |
| huck | 74 | 301 | 11.00 | 11 | 11 | 0.00 | 0.6336 | 11.00 | 11 | 11 | 0.00 | 10.4280 | 11 |
| homer | 561 | 1629 | 13.00 | 13 | 13 | 0.00 | 1.3098 | 13.00 | 13 | 13 | 0.00 | 67.6890 | 13 |
| games120 | 120 | 638 | 9.00 | 9 | 9 | 0.00 | 0.6012 | 9.00 | 9 | 9 | 0.00 | 18.5230 | 9 |

## Set4-medium

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat300_28_0 | 300 | 21695 | 36.00 | 36 | 36 | 0.00 | 7.1751 | 35.00 | 35 | 35 | 0.00 | 559.6576 | 28 |
| dsjc250.5 | 250 | 15668 | 32.00 | 32 | 32 | 0.00 | 5.6188 | 31.00 | 31 | 31 | 0.00 | 512.8320 | 28 |
| le450_25c | 450 | 17343 | 29.00 | 29 | 29 | 0.00 | 3.9444 | 29.00 | 29 | 29 | 0.00 | 397.6999 | 25 |

## Set5-miles

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| miles250 | 128 | 387 | 8.00 | 8 | 8 | 0.00 | 0.6295 | 8.00 | 8 | 8 | 0.00 | 14.2179 | 8 |
| miles500 | 128 | 1170 | 20.00 | 20 | 20 | 0.00 | 1.3294 | 20.00 | 20 | 20 | 0.00 | 31.4741 | 20 |
| miles750 | 128 | 2113 | 31.00 | 31 | 31 | 0.00 | 0.0894 | 32.00 | 32 | 32 | 0.00 | 52.6383 | 31 |
| miles1000 | 128 | 3216 | 43.00 | 43 | 43 | 0.00 | 2.6846 | 43.00 | 43 | 43 | 0.00 | 80.1818 | 42 |
| miles1500 | 128 | 5198 | 73.00 | 73 | 73 | 0.00 | 3.6828 | 73.00 | 73 | 73 | 0.00 | 137.7784 | 73 |

## Set6-large

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dsjc500.9 | 500 | 112437 | 139.00 | 139 | 139 | 0.00 | 71.0719 | 138.00 | 138 | 138 | 0.00 | 5088.0499 | 126 |
| dsjc1000.5 | 1000 | 249826 | 103.00 | 103 | 103 | 0.00 | 49.5291 | 99.00 | 99 | 99 | 0.00 | 12831.3922 | 85 |

