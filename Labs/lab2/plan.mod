/*********************************************
 * OPL 22.1.2.0 Model
 * Author: monic
 * Creation Date: Mar 7, 2026 at 1:52:08 PM
 *********************************************/
dvar float+ Gas;
dvar float+ Chloride;

maximize
  40 * Gas + 50 * Chloride;

subject to {
  ctMaxTotal:
    Gas + Chloride <= 50;

  ctMaxTotal2:
    3 * Gas + 4 * Chloride <= 180;

  ctMaxTotal3:
    Chloride <= 40;
}

execute {
  writeln("Gas = ", Gas);
  writeln("Chloride = ", Chloride);
  writeln("Profit = ", cplex.getObjValue());
}

// solution (optimal) with objective 2300
//Gas = 20
//Chloride = 30
//Profit = 2300