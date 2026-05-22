# Graph Coloring Benchmark Results

> **GNN** — 2-layer GCN (numpy, self-supervised coloring loss )  
> **Tabu** — TabuCol local search, independent upper bound from nx largest-first greedy  
> All values averaged over 15 independent runs.

## Set1-myciel

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| myciel3 | 11 | 20 | 4.00 | 4 | 4 | 0.00 | 0.2676 | 4.00 | 4 | 4 | 0.00 | 1.1707 | 4 |
| myciel4 | 23 | 71 | 5.00 | 5 | 5 | 0.00 | 0.4010 | 5.00 | 5 | 5 | 0.00 | 2.7836 | 5 |
| myciel5 | 47 | 236 | 6.00 | 6 | 6 | 0.00 | 0.5932 | 6.00 | 6 | 6 | 0.00 | 7.0554 | 6 |
| myciel6 | 95 | 755 | 7.00 | 7 | 7 | 0.00 | 0.9081 | 7.00 | 7 | 7 | 0.00 | 18.5445 | 7 |
| myciel7 | 191 | 2360 | 8.00 | 8 | 8 | 0.00 | 1.5506 | 8.00 | 8 | 8 | 0.00 | 52.5811 | 8 |

## Set2-queen

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| queen5_5 | 25 | 160 | 6.00 | 6 | 6 | 0.00 | 0.5803 | 5.00 | 5 | 5 | 0.00 | 0.3756 | 5 |
| queen6_6 | 36 | 290 | 7.00 | 7 | 7 | 0.00 | 0.4938 | 7.00 | 7 | 7 | 0.00 | 5.4600 | 7 |
| queen7_7 | 49 | 476 | 8.00 | 8 | 8 | 0.00 | 0.8719 | 8.00 | 8 | 8 | 0.00 | 9.5863 | 7 |

## Set3

| Instance | Nodes | Edges | Tabu mean k | Tabu min k | Tabu max k | Tabu stdev | Tabu time (s) | TabuBasic mean k | TabuBasic min k | TabuBasic max k | TabuBasic stdev | TabuBasic time (s) | Best k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anna | 138 | 493 | 11.00 | 11 | 11 | 0.00 | 1.3755 | 11.00 | 11 | 11 | 0.00 | 18.7052 | 11 |
| david | 87 | 406 | 11.00 | 11 | 11 | 0.00 | 1.1383 | 11.00 | 11 | 11 | 0.00 | 13.5073 | 11 |
| jean | 80 | 254 | 10.00 | 10 | 10 | 0.00 | 0.6563 | 10.00 | 10 | 10 | 0.00 | 9.8583 | 10 |
| huck | 74 | 301 | 11.00 | 11 | 11 | 0.00 | 0.6106 | 11.00 | 11 | 11 | 0.00 | 10.3450 | 11 |
| homer | 561 | 1629 | 13.00 | 13 | 13 | 0.00 | 1.3170 | 13.00 | 13 | 13 | 0.00 | 68.4941 | 13 |
| games120 | 120 | 638 | 9.00 | 9 | 9 | 0.00 | 0.5929 | 9.00 | 9 | 9 | 0.00 | 19.3542 | 9 |


## Set4-medium

| Instance | Nodes | Edges | Greedy UB | Tabucol improved best | TabucolBasic best |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat300_28_0 | 300 | 21695 | 45 | 36 | 35 |
| dsjc250.5 | 250 | 15668 | 41 | 32 | 31 |
| le450_25c | 450 | 17343 | 29 | 29 | 29 |

## Set5-miles

| Instance | Nodes | Edges | Greedy UB | Tabucol improved best | TabucolBasic best |
| --- | ---: | ---: | ---: | ---: | ---: |
| miles250 | 128 | 387 | 8 | 8 | 8 |
| miles500 | 128 | 1170 | 20 | 20 | 20 |
| miles750 | 128 | 2113 | 32 | 32 | 32 |
| miles1000 | 128 | 3216 | 43 | 43 | 43 |
| miles1500 | 128 | 5198 | 73 | 73 | 73 |

## Set6-large

| Instance | Nodes | Edges | Greedy UB | Tabucol improved best | TabucolBasic best |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsjc500.9 | 500 | 112437 | 169 | 139 | 136 |
| dsjc1000.5 | 1000 | 249826 | 121 | 103 | 99 |
