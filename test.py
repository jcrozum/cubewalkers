import cupy as cp
import cubewalkers as cw
from timeit import default_timer as timer

cp.set_printoptions(edgeitems=5)

if __name__=='__main__':
    print("INITIALIZING")
    rules="""#
ABA* = ABA
ABI1* = not PA and not RCARs and not ROS and pHc
ABI2* = not RCARs and not ROS and not PA
Actin_Reorganization* = not AtRAC1
AnionEM* = SLAC1 or (QUAC1 and SLAH3)
AtRAC1* = not ABA or ABI1
CIS* = InsP3_6 or cADPR
CPK3_21* = Ca2c
Ca2c* = (CaIM or CIS) and not Ca2_ATPase
CaIM* = Actin_Reorganization or GHR1
Ca2_ATPase* = Ca2c
Closure* = Microtubule_Depolymerization and H2O_Efflux
DAG* = PLC
Depolarization* = (AnionEM or Ca2c or KEV) and (not H_ATPase or not K_Efflux)
GHR1* = not ABI2 and ROS
H2O_Efflux* = AnionEM and OST1 and K_Efflux and not Malate
HAB1* = not RCARs and not ROS
H_ATPase* = not pHc and not Ca2c and not ROS
InsP3_6* = PLC
KEV* = Vacuolar_Acidification or Ca2c
KOUT* = (not NO or not ROS or pHc) and Depolarization
K_Efflux* = KEV and KOUT
MPK9_12* = Ca2c
Malate* = PEPC and not ABA and not AnionEM
Microtubule_Depolymerization* = TCTP
NIA1_2* = ROS
NO* = NIA1_2
cGMP* = NO
OST1* = (not ABI1 and not HAB1) or (not PP2CA and not ABI2) or (not ABI1 and not ABI2) or (not HAB1 and not PP2CA) or (not HAB1 and not ABI2) or (not ABI1 and not PP2CA)
PA* = PLDdelta or PLDalpha or DAG
PEPC* = not ABA
V_PPase* = ABA
PLC* = Ca2c
PLDalpha* = S1P and Ca2c
PLDdelta* = NO or ROS
PP2CA* = not RCARs and not ROS
QUAC1* = OST1 and Ca2c
ROS* = pHc and not ABI1 and OST1 and S1P and PA
RCARs* = ABA
SLAC1* = MPK9_12 and OST1 and GHR1 and not ABI1 and not PP2CA and not ABI2 and pHc
SLAH3* = CPK3_21 and not ABI1
S1P* = PA or ABA
TCTP* = Ca2c
V_ATPase* = Ca2c
Vacuolar_Acidification* = V_PPase or V_ATPase
cADPR* = cGMP and ROS and NO
pHc* = ((OST1 and not ABI2 and not ABI1) or Ca2c) and Vacuolar_Acidification
#"""

    # uncomment if more variables are desired for testing
    # from io import StringIO
    # rules = rules + '\n' + ''.join([('x{}'.format(i)).join(list(StringIO(rules))[0:-1]) for i in range(10)]) # make the system bigger

    experiment_string="""#
ABA,0,5,0
ABA,6,10,1
ABA,11,inf, !ABA
#"""
    rules ="""#
ABA* = ABA
C* = 0
D* = 1
# E*=1 with prob 0.3, E*=0 with prob 0.7
E* = 1 & (0<<=0.3) | 0 & (0.3<<=1)
#"""

    experiment_string="""#
ABA,3,5,1
ABA,9,inf,!ABA
#"""

    myexperiment=cw.Experiment(experiment_string)
    mymodel=cw.Model(rules,experiment=myexperiment)
    print(mymodel.code)
    N=mymodel.n_variables
    for T,W in [(10000,1),(1000,10),(100,100),(10,1000),(10000,10000)]:
    #for T,W in [(15,16)]:
        averages_only = (T*W > 1e5)
        start = timer()
        mymodel.simulate_ensemble(n_time_steps=T,n_walkers=W,
                                         averages_only=averages_only,
                                         maskfunction='synchronous_PBN',
                                         threads_per_block=(32,32))
        end = timer()
        print(f'{averages_only=}, {T=}, {W=}, {N=}, {end-start}s')
        #print(mymodel.trajectories[:,0,:])
        #quit()

        if averages_only:
            print(mymodel.trajectories[:,mymodel.vardict['E']].T)
        else:
            print(cp.mean(mymodel.trajectories[:,mymodel.vardict['E'],:],axis=1).T)
            
            #for w in range(W):
            #    print(mymodel.trajectories[:,mymodel.vardict['E'],w])
            
