import numpy as np
import math

SQRT5 = math.sqrt(5)

# --- Определяем ядра на верхнем уровне модуля ---

def kernel_epanechnikov(u):
	res = (3 / (4 * SQRT5)) * (1 - (u**2) / 5)
	return np.where(np.abs(u) < SQRT5, res, 0)

def kernel_gaussian(u):
	return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def kernel_triangular(u):
	return np.where(np.abs(u) < 1, 1 - np.abs(u), 0)

def kernel_uniform(u):
	return np.where(np.abs(u) < 1, 0.5, 0)

def get_kernel_func(name):
	"""Безопасный способ получить функцию ядра по имени"""
	kernels = {
		'epanechnikov': kernel_epanechnikov,
		'gaussian': kernel_gaussian,
		'triangular': kernel_triangular,
		'uniform': kernel_uniform
	}
	return kernels.get(name, kernel_epanechnikov)

def get_kernels():
	return ['epanechnikov', 'gaussian', 'triangular', 'uniform']

# --- Глобальные переменные для worker-процессов ---
_X, _Y, _S, _K_FUNC = None, None, None, None

def _init_loo(x, y, s, k_name):
	global _X, _Y, _S, _K_FUNC
	_X, _Y, _S, _K_FUNC = x, y, s, get_kernel_func(k_name)

def _loo_unit(j_and_c):
	j, c = j_and_c
	# Используем c как массив для поддержки векторного режима
	diffs = (_X[j] - _X) / (np.array(c) * _S)
	weights = np.prod(_K_FUNC(diffs), axis=1)
	weights[j] = 0 # Исключаем саму точку
	sw = np.sum(weights)
	y_hat = np.sum(weights * _Y) / sw if sw > 1e-15 else np.mean(_Y)
	return (_Y[j] - y_hat)**2

def calc_loo_error_parallel(c, x, y, s, k_name, pool):
	n = len(y)
	tasks = [(j, c) for j in range(n)]
	errors = pool.map(_loo_unit, tasks)
	return sum(errors) / n

def predict_nre(x_target, x_train, y_train, stds, c, k_func, disable_nodata=False):
	c = np.array(c)
	diffs = (x_target - x_train) / (c * stds)
	weights = np.prod(k_func(diffs), axis=-1)
	sw = np.sum(weights)
	nodata = np.mean(y_train) if disable_nodata else None
	return np.sum(weights * y_train) / sw if sw > 1e-15 else nodata