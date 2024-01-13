NAME MasterProb
ROWS
 N  OBJ
 L  IB      
 L  RIB     
COLUMNS
    X[1,1,1]  OBJ       2.9081574294019152e+01
    X[1,1,1]  IB        260
    X[1,1,2]  OBJ       2.7291938952848742e+01
    X[1,1,2]  IB        244
    X[1,1,3]  OBJ       5.5926104411575288e+00
    X[1,1,3]  IB        50
    X[2,1,1]  OBJ       2.9081574294019152e+01
    X[2,1,1]  RIB       260
    X[2,1,2]  OBJ       2.7291938952848742e+01
    X[2,1,2]  RIB       244
    X[2,1,3]  OBJ       5.5926104411575288e+00
    X[2,1,3]  RIB       50
    eta       OBJ       1
RHS
    RHS1      IB        250000
    RHS1      RIB       125000
BOUNDS
 LO BND1      X[1,1,1]  1
 UP BND1      X[1,1,1]  100.1
 LO BND1      X[1,1,2]  1
 UP BND1      X[1,1,2]  100.1
 LO BND1      X[1,1,3]  1
 UP BND1      X[1,1,3]  60.1
 UP BND1      X[2,1,1]  50.1
 UP BND1      X[2,1,2]  40.1
 UP BND1      X[2,1,3]  30.1
 LO BND1      eta       -1000000
ENDATA
