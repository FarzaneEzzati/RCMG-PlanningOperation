NAME MasterProb
ROWS
 N  OBJ
 L  UpB01   
 G  LoB01   
 L  UpB11   
 G  LoB11   
 L  UpB02   
 G  LoB02   
 L  UpB12   
 G  LoB12   
 L  UpB03   
 G  LoB03   
 L  UpB13   
 G  LoB13   
 L  IB      
 L  RIB     
COLUMNS
    X[0,1]    OBJ       6.7111325293890346e+01
    X[0,1]    UpB01     1
    X[0,1]    LoB01     1
    X[0,1]    IB        600
    X[0,2]    OBJ       3.1094914052835861e+02
    X[0,2]    UpB02     1
    X[0,2]    LoB02     1
    X[0,2]    IB        2780
    X[0,3]    OBJ       5.5926104411575295e+01
    X[0,3]    UpB03     1
    X[0,3]    LoB03     1
    X[0,3]    IB        500
    X[1,1]    OBJ       6.7111325293890346e+01
    X[1,1]    UpB11     1
    X[1,1]    LoB11     1
    X[1,1]    RIB       600
    X[1,2]    OBJ       3.1094914052835861e+02
    X[1,2]    UpB12     1
    X[1,2]    LoB12     1
    X[1,2]    RIB       2780
    X[1,3]    OBJ       5.5926104411575295e+01
    X[1,3]    UpB13     1
    X[1,3]    LoB13     1
    X[1,3]    RIB       500
    eta       OBJ       1
RHS
    RHS1      UpB01     100
    RHS1      LoB01     20
    RHS1      UpB11     50
    RHS1      UpB02     100
    RHS1      LoB02     20
    RHS1      UpB12     40
    RHS1      UpB03     60
    RHS1      LoB03     10
    RHS1      UpB13     30
    RHS1      IB        250000
    RHS1      RIB       125000
BOUNDS
 FR BND1      eta     
ENDATA
