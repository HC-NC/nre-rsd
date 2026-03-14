import argparse
from math import *
import pandas as pd
import numpy as np
from progress.bar import IncrementalBar
from multiprocessing import Pool

SQRT5 = sqrt(5)

data_path = "test.xlsx"

blur_coef_opt_columns = ["C", "W(c)"]

def main():
	global sample
	sample = pd.read_excel(data_path)

	print(sample, '\n')
	print(sample.describe(), '\n')

	global bar
	bar = IncrementalBar('Calc W(c) optimization', max = len(sample) * 100, suffix = '%(percent).1f%% (%(eta_td)s)')

	with Pool(2) as p:
		blur_coefs = list(p.map(VToWcOpt, list(range(1, 101))))

	bar.finish()

	blur_coefs_df = pd.DataFrame(blur_coefs, columns = blur_coef_opt_columns)

	print('\n', blur_coefs_df, '\n')
	print(blur_coefs_df.describe(), '\n')

	blur_coefs_df.to_excel("wc_opt_" + data_path)
	

def EpanechnikovFunction(u):
	return 3 / (4 * SQRT5) - 3 * pow(u, 2) / (20 * SQRT5) if abs(u) < SQRT5 else 0

def W(c):
	tmp_sum = 0

	for j in range(len(sample)):
		up_sum = 0
		down_sum = 0

		for i in range(len(sample)):
			if i == j:
				continue

			tmp_mult = 1

			for key in sample.keys():
				if key == 'Y':
					continue

				tmp_mult *= EpanechnikovFunction((sample[key][j] - sample[key][i]) / (c * sample[key].std()))

			up_sum += sample.Y[i] * tmp_mult
			down_sum += tmp_mult

		if down_sum != 0:
			tmp_sum += pow(sample.Y[j] - up_sum / down_sum, 2)

		bar.next()

	return tmp_sum / len(sample)

def VToWcOpt(v):
	print(sample)
	c = v * 0.01
	return [c, W(c)]		

if __name__ == "__main__":
	main()
