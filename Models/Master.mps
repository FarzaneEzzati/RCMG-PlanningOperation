NAME MasterProb
ROWS
 N  OBJ
 G  R0      
 G  R1      
 G  R2      
 G  R3      
 G  R4      
 G  R5      
 G  R6      
 G  R7      
 G  R8      
 G  R9      
 G  R10     
 G  R11     
 G  R12     
 G  R13     
COLUMNS
    X[1,1,1]  OBJ       1.2303742970546564e+01
    X[1,1,1]  R0        -110
    X[1,1,1]  R2        -1
    X[1,1,1]  R8        1
    X[1,1,2]  OBJ       1.2303742970546564e+01
    X[1,1,2]  R0        -110
    X[1,1,2]  R3        -1
    X[1,1,2]  R9        1
    X[1,1,3]  OBJ       2.2370441764630118e+00
    X[1,1,3]  R0        -20
    X[1,1,3]  R4        -1
    X[1,1,3]  R10       1
    X[2,1,1]  OBJ       1.2303742970546564e+01
    X[2,1,1]  R1        -110
    X[2,1,1]  R5        -1
    X[2,1,1]  R11       1
    X[2,1,2]  OBJ       1.2303742970546564e+01
    X[2,1,2]  R1        -110
    X[2,1,2]  R6        -1
    X[2,1,2]  R12       1
    X[2,1,3]  OBJ       2.2370441764630118e+00
    X[2,1,3]  R1        -20
    X[2,1,3]  R7        -1
    X[2,1,3]  R13       1
    eta       OBJ       1
RHS
    RHS1      R0        -250000
    RHS1      R1        -125000
    RHS1      R2        -200.6
    RHS1      R3        -200.8
    RHS1      R4        -100
    RHS1      R5        -50.1
    RHS1      R6        -40.1
    RHS1      R7        -30.1
    RHS1      R8        10
    RHS1      R9        10
    RHS1      R10       2
BOUNDS
 LO BND1      X[1,1,1]  10
 UP BND1      X[1,1,1]  200.6
 LO BND1      X[1,1,2]  10
 UP BND1      X[1,1,2]  200.8
 LO BND1      X[1,1,3]  2
 UP BND1      X[1,1,3]  100
 UP BND1      X[2,1,1]  50.1
 UP BND1      X[2,1,2]  40.1
 UP BND1      X[2,1,3]  30.1
 LO BND1      eta       -10000000
ENDATA
