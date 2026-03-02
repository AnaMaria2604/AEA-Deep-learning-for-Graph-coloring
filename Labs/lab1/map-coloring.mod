using CP;
range r = 0..3;
string Names[r] =["blue", "red", "yellow", "green"];

dvar int Belgium in r;
dvar int Denmark in r;
dvar int France in r;
dvar int Germany in r;
dvar int Luxembourg in r;
dvar int Netherlands in r;
dvar int Switzerland in r;

subject to {
  Belgium != Netherlands;
  Belgium != Luxembourg;
  Belgium != Germany;
  Belgium != France;
  
// comented because Denmark and Germany have the same color
//  Denmark != Germany;	
  
  France != Luxembourg;
  France != Germany;
  
  Germany != Luxembourg;
  Germany != Netherlands;
  
  Switzerland != France;
  Switzerland != Germany;
  }
  
 execute {
  writeln("Belgium:	", Names[Belgium]);
  writeln("Denmark:	", Names[Denmark]);
  writeln("France:	", Names[France]);
  writeln("Germany:	", Names[Germany]);
  writeln("Luxembourg:	", Names[Luxembourg]);
  writeln("Netherlands:	", Names[Netherlands]);
   }
   
// a solution
//Belgium:	    yellow
//Denmark:	    blue
//France:	    green
//Germany:	    blue
//Luxembourg:	red
//Netherlands:	green