NAME MasterProb
ROWS
 N  OBJ
 L  IB      
 L  RIB     
COLUMNS
    X[1][(1,1)]  OBJ       6.7111325293890346e+01
    X[1][(1,1)]  IB        600
    X[1][(2,1)]  OBJ       6.7111325293890346e+01
    X[1][(2,1)]  RIB       600
    X[1][(1,2)]  OBJ       3.1094914052835861e+02
    X[1][(1,2)]  IB        2780
    X[1][(2,2)]  OBJ       3.1094914052835861e+02
    X[1][(2,2)]  RIB       2780
    X[1][(1,3)]  OBJ       5.5926104411575295e+01
    X[1][(1,3)]  IB        500
    X[1][(2,3)]  OBJ       5.5926104411575295e+01
    X[1][(2,3)]  RIB       500
    eta       OBJ       1
RHS
    RHS1      IB        250000
    RHS1      RIB       125000
BOUNDS
 LO BND1      X[1][(1,1)]  20.5
 UP BND1      X[1][(1,1)]  100
 UP BND1      X[1][(2,1)]  50
 LO BND1      X[1][(1,2)]  20.9
 UP BND1      X[1][(1,2)]  100
 UP BND1      X[1][(2,2)]  40
 LO BND1      X[1][(1,3)]  10.1
 UP BND1      X[1][(1,3)]  60
 UP BND1      X[1][(2,3)]  30
 LO BND1      eta       -100
ENDATA
