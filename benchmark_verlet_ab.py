import time, copy, yaml, numpy as np, sys
sys.path.insert(0, ".")
import src.simulator.numpy_simulator as ns
import src.physics.forces as F

cfg = {"chain":{"N":30,"dimension":3},"bond":{"type":"harmonic","k_bond":100.0,"r0":1.0},
 "fene":{"k":30.0,"R0":1.5},"wca":{"epsilon":1.0,"sigma":1.0},
 "thermostat":{"kBT":1.0,"gamma":1.0,"mass":1.0},"integrator":{"type":"overdamped","dt":0.001},
 "simulation":{"T_steps":20000,"n_burnin":0,"save_every":100},
 "initialization":{"method":"linear","spacing":1.0},"box":{"L":100.0,"periodic":True},"seed":42}

orig_caf = ns.compute_all_forces
def run(use_list, N=30, steps=20000):
    c = copy.deepcopy(cfg); c["chain"]["N"]=N; c["simulation"]["T_steps"]=steps
    ns.compute_all_forces = orig_caf if use_list else (lambda *a, neighbor_pairs=None, **k: orig_caf(*a, **k))
    t=time.perf_counter(); tr = ns.NumpySimulator(c).run(verbose=use_list); return tr, time.perf_counter()-t

a, ta = run(False)
b, tb = run(True)
same = all(np.array_equal(fa["positions"], fb["positions"]) and np.array_equal(fa["forces"], fb["forces"])
           and fa["potential_energy_nonbond"]==fb["potential_energy_nonbond"] for fa,fb in zip(a["frames"], b["frames"]))
print("N=30 frames:", len(a["frames"]), "bitwise identical trajectories:", same)
print(f"N=30  all-pairs {ta:.2f}s  verlet {tb:.2f}s  speedup {ta/tb:.2f}x")
for N in (100, 300):
    _,ta = run(False, N, 4000); _,tb = run(True, N, 4000)
    print(f"N={N} (4k steps) all-pairs {ta:.2f}s  verlet {tb:.2f}s  speedup {ta/tb:.2f}x")