/*********************************************
 * OPL 22.1.2.0 Model
 * Author: monic
 * Creation Date: Mar 7, 2026 at 2:16:26 PM
 *********************************************/
int n = 15;
int m = 5;

range N = 1..n;
range M = 1..m;

int r[N]  = [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25];
int w[N] = [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36];
int cap[M]    = [100, 100, 100, 100, 100];

dvar boolean x[N][M];

maximize
  sum(i in N, j in M) r[i] * x[i][j];

subject to {
  forall(j in M)
    sum(i in N) w[i] * x[i][j] <= cap[j];

  forall(i in N)
    sum(j in M) x[i][j] <= 1;
}

execute {
  writeln("Solution:");
  for(var j in M) {
    writeln("Knapsack ", j, ":");
    for(var i in N)
      if (x[i][j] > 0)
        writeln("  item ", i, " selected w=", w[i], ", r=", r[i]);
  }
  writeln("Total value = ", cplex.getObjValue());
}

// solution (optimal) with objective 395
//Solution:
//Knapsack 1:
//  item 8 selected w=42, r=40
//  item 13 selected w=42, r=20
//Knapsack 2:
//  item 2 selected w=30, r=30
//  item 11 selected w=30, r=45
//  item 15 selected w=36, r=25
//Knapsack 3:
//  item 4 selected w=36, r=50
//  item 10 selected w=24, r=35
//  item 14 selected w=36, r=30
//Knapsack 4:
//  item 3 selected w=42, r=25
//  item 6 selected w=48, r=30
//Knapsack 5:
//  item 5 selected w=36, r=35
//  item 9 selected w=36, r=30
//Total value = 395