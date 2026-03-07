/*********************************************
 * OPL 22.1.2.0 Model
 * Author: monic
 * Creation Date: Mar 7, 2026 at 1:59:06 PM
 *********************************************/
int n = 4;
range R = 1..n;

int r[R]  = [10, 6, 8, 7];
int w[R] = [ 5, 4, 6, 3];
int cap = 10;

dvar int+ x[R];

maximize
  sum(i in R) r[i] * x[i];

subject to {
  ctCapacity:
    sum(i in R) w[i] * x[i] <= cap;
}

execute {
  writeln("Solution:");
  for(var i in R)
    writeln("x[", i, "] = ", x[i]);
  writeln("Total value = ", cplex.getObjValue());
}

// solution (optimal) with objective 21
//Solution:
//x[1] = 0
//x[2] = 0
//x[3] = 0
//x[4] = 3
//Total value = 21