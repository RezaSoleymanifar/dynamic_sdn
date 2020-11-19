from gen_dataset import run_sim
from gen_opt_locs import find_opt

seed1 = 1
seed2 = 2
bound = 1
centers = 3
compactness = 19
speed_var = 5e-3
n_samples = 500
out_coeff = 1.5
delta_1 = -1
delta_2 = -1
case = 'exp'
T = 70
gamma = 0.1
t_min = 1e-3
t_reset = 1e-1
alpha = 0.8

# Save generating arguments
with open('info.txt', 'w+') as file:
    file.write("""seed1 = {0}, seed2 = {1}, bound = {2},
    centers= {3}, compactness= {4}, speed_var= {5},
    n_samples={6}, out_coeff= {7}, case = {8},
     T = {9}, gamma = {10}, t_min={11}, t_reset = {12},
      alpha = {13}, delta_1 = {14}, delta_2 = {15}""".
               format(seed1, seed2, bound, centers,
                      compactness, speed_var, n_samples,
                      out_coeff, case, T, gamma, t_min,
                      t_reset, alpha, delta_1, delta_2))

#Generate datasets
run_sim(seed1 = seed1, seed2 = seed2, bound = bound,
                    centers= centers, compactness= compactness,
                    speed_var= speed_var, n_samples=n_samples,
                    out_coeff= out_coeff, case = case, T = T, delta_1 = delta_1, delta_2 = delta_2)

find_opt(gamma = gamma, t_min = 1e-4, t_reset = 1e-1, alpha = 0.8)