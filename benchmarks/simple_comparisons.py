import random
import re
import time
import cubewalkers as cw
import boolean2
import cana.boolean_network as bn

rules = """ABA*=ABA
ABI1*=not PA and not RCARs and not ROS and pHc
ABI2*=not RCARs and not ROS and not PA
Actin_Reorganization*=not AtRAC1
AnionEM*=SLAC1 or (QUAC1 and SLAH3)
AtRAC1*=not ABA or ABI1
CIS*=InsP3_6 or cADPR
CPK3_21*=Ca2c
Ca2c*=(CaIM or CIS) and not Ca2_ATPase
CaIM*=Actin_Reorganization or GHR1
Ca2_ATPase*=Ca2c
Closure*=Microtubule_Depolymerization and H2O_Efflux
DAG*=PLC
Depolarization*=(AnionEM or Ca2c or KEV) and (not H_ATPase or not K_Efflux)
GHR1*=not ABI2 and ROS
H2O_Efflux*=AnionEM and OST1 and K_Efflux and not Malate
HAB1*=not RCARs and not ROS
H_ATPase*=not pHc and not Ca2c and not ROS
InsP3_6*=PLC
KEV*=Vacuolar_Acidification or Ca2c
KOUT*=(not NO or not ROS or pHc) and Depolarization
K_Efflux*=KEV and KOUT
MPK9_12*=Ca2c
Malate*=PEPC and not ABA and not AnionEM
Microtubule_Depolymerization*=TCTP
NIA1_2*=ROS
NO*=NIA1_2
cGMP*=NO
OST1*=(not ABI1 and not HAB1) or (not PP2CA and not ABI2) or (not ABI1 and not ABI2) or (not HAB1 and not PP2CA) or (not HAB1 and not ABI2) or (not ABI1 and not PP2CA)
PA*=PLDdelta or PLDalpha or DAG
PEPC*=not ABA
V_PPase*=ABA
PLC*=Ca2c
PLDalpha*=S1P and Ca2c
PLDdelta*=NO or ROS
PP2CA*=not RCARs and not ROS
QUAC1*=OST1 and Ca2c
ROS*=pHc and not ABI1 and OST1 and S1P and PA
RCARs*=ABA
SLAC1*=MPK9_12 and OST1 and GHR1 and not ABI1 and not PP2CA and not ABI2 and pHc
SLAH3*=CPK3_21 and not ABI1
S1P*=PA or ABA
TCTP*=Ca2c
V_ATPase*=Ca2c
Vacuolar_Acidification*=V_PPase or V_ATPase
cADPR*=cGMP and ROS and NO
pHc*=((OST1 and not ABI2 and not ABI1) or Ca2c) and Vacuolar_Acidification"""

T = 1000
W = 500

print("start cana")
ti = time.perf_counter()
model_cana = bn.BooleanNetwork.from_string_boolean(rules)
initial = "".join(random.choices(["0", "1"], k=model_cana.Nnodes))
for _ in range(W):
    model_cana.trajectory(initial, length=T)
print(f"done cana: {time.perf_counter()-ti}s")

T = 10_000
W = 2500
print("start cw")
ti = time.perf_counter()
model_cw = cw.Model(rules, n_time_steps=T, n_walkers=W)
model_cw.simulate_ensemble()
print(f"done cw: {time.perf_counter()-ti}s")

T = 100
W = 50
print("start bn")
ti = time.perf_counter()
model_bn2 = boolean2.Model(text=rules, mode="sync")
model_bn2.initialize(missing=lambda x: random.choice([0, 1]))
for _ in range(W):
    model_bn2.iterate(steps=T)  # type: ignore
print(f"done bn: {time.perf_counter()-ti}s")
