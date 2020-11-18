import gen_dataset
import gen_opt_locs
# import RCP
# import test

gen_dataset.run_sim(2, 7, centers= 10, compactness= 10, variation= 7e-3, n_samples=500, out_coeff= 2.5)
gen_opt_locs.find_opt()
# RCP.rcp()
# test.show_sim()
