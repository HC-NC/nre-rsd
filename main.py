import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats import ks_2samp
from scipy.optimize import minimize
import nre_lib

# Словари локализации
STRINGS = {
	'ru': {
		'desc': 'nrer: Инструментарий непараметрической регрессии для ДЗЗ',
		'h_train': 'Обучение модели (поиск коэффициента размытости c)',
		'h_apply': 'Восстановление данных в формате GeoTIFF',
		'h_plot': 'Визуализация зависимостей',
		'h_view': 'Просмотр GeoTIFF файла',
		'h_config': 'Настройка приложения',
		'h_data': 'Путь к Excel файлу выборки',
		'h_mode': 'Режим: single (один c) или multi (вектор c)',
		'h_kern': 'Тип функции ядра',
		'h_out': 'Путь к выходному файлу',
		'h_step': 'Количество шагов при поиске по сетке',
		'h_proc': 'Количество задействованных ядер процессора',
		'h_crop': 'Доля фрагмента изображения (от 0.1 до 1.0)',
		'h_model': 'Путь к JSON файлу с параметрами модели',
		'h_input': 'Путь к исходному GeoTIFF файлу',
		'h_lang': 'Установка языка интерфейса (ru/en)',
		'h_tile': 'Размер тайла для обработки',
		'msg_opt': 'Запуск многоядерной оптимизации...',
		'msg_apply': 'Обработка изображения...',
		'err_chan': 'Ошибка: Количество каналов в TIFF ({}) не совпадает с данными модели ({})!',
		'err_file': 'Ошибка: Файл не найден: {}',
		'done': 'Операция завершена успешно.'
	},
	'en': {
		'desc': 'nrer: Non-parametric regression toolkit for Remote Sensing',
		'h_train': 'Model training (optimization of blur coefficient c)',
		'h_apply': 'Data recovery for GeoTIFF files',
		'h_plot': 'Dependency visualization',
		'h_view': 'Preview GeoTIFF file',
		'h_config': 'Application configuration',
		'h_data': 'Path to Excel sample file',
		'h_mode': 'Mode: single (one c) or multi (vector c)',
		'h_kern': 'Kernel function type',
		'h_out': 'Output file path',
		'h_step': 'Number of grid search steps',
		'h_proc': 'Number of CPU cores to use',
		'h_crop': 'Image crop fraction (0.1 to 1.0)',
		'h_model': 'Path to JSON model file',
		'h_input': 'Path to source GeoTIFF file',
		'h_lang': 'Set interface language (ru/en)',
		'h_tile': 'Processing tile size',
		'msg_opt': 'Starting multi-core optimization...',
		'msg_apply': 'Processing image...',
		'err_chan': 'Error: TIFF channels count ({}) does not match model data ({})!',
		'err_file': 'Error: File not found: {}',
		'done': 'Operation completed successfully.'
	}
}

CONFIG_FILE = 'config.json'

def load_config():
	defaults = {'lang': 'en', 'proc': os.cpu_count()}
	if os.path.exists(CONFIG_FILE):
		try:
			with open(CONFIG_FILE, 'r') as f:
				defaults.update(json.load(f))
		except: pass
	return defaults

def save_config(conf):
	with open(CONFIG_FILE, 'w') as f:
		json.dump(conf, f, indent=4)

def get_txt(key):
	conf = load_config()
	lang = conf.get('lang', 'en')
	return STRINGS[lang].get(key, key)

# --- Глобальные функции для Pool при Apply ---
_APPLY_DATA = None

def _init_apply_worker(x, y, s, c, k_name):
    global _APPLY_DATA
    # Используем библиотечную функцию для получения объекта функции по имени
    _APPLY_DATA = (x, y, s, c, nre_lib.get_kernel_func(k_name))

def _apply_unit(chunk):
	x_t, y_t, s_t, c_t, k_f = _APPLY_DATA
	return [nre_lib.predict_nre(p, x_t, y_t, s_t, c_t, k_f) for p in chunk]

class NRERApp:
	def __init__(self):
		conf = load_config()
		# Использование ArgumentDefaultsHelpFormatter позволяет выводить default значения в help
		self.parser = argparse.ArgumentParser(
			prog="nrer", 
			description=get_txt('desc'),
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
		subparsers = self.parser.add_subparsers(dest="command")

		# --- Команда TRAIN ---
		tr = subparsers.add_parser('train', help=get_txt('h_train'), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		tr.add_argument('-d', '--data', required=True, help=get_txt('h_data'))
		tr.add_argument('-m', '--mode', choices=['single', 'multi'], default='single', help=get_txt('h_mode'))
		tr.add_argument('-k', '--kernel', default='epanechnikov', choices=nre_lib.get_kernels(), help=get_txt('h_kern'))
		tr.add_argument('-s', '--steps', type=int, default=100, help=get_txt('h_step'))
		tr.add_argument('-p', '--proc', type=int, default=conf['proc'], help=get_txt('h_proc'))
		tr.add_argument('-o', '--output', default='model.json', help=get_txt('h_out'))

		# --- Команда APPLY ---
		ap = subparsers.add_parser('apply', help=get_txt('h_apply'), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		ap.add_argument('-i', '--input', required=True, help=get_txt('h_input'))
		ap.add_argument('-m', '--model', required=True, help=get_txt('h_model'))
		ap.add_argument('-c', '--crop', type=float, default=1.0, help=get_txt('h_crop'))
		ap.add_argument('-p', '--proc', type=int, default=conf['proc'], help=get_txt('h_proc'))
		ap.add_argument('-t', '--tile', type=int, default=10000, help=get_txt('h_tile'))
		ap.add_argument('-o', '--output', default='result.tif', help=get_txt('h_out'))

		# --- Команда PLOT ---
		pl = subparsers.add_parser('plot', help=get_txt('h_plot'))
		pl.add_argument('-m', '--model', required=True, help=get_txt('h_model'))

		# --- Команда VIEW ---
		vi = subparsers.add_parser('view', help=get_txt('h_view'))
		vi.add_argument('-i', '--input', required=True, help=get_txt('h_input'))

		# --- Команда CONFIG ---
		cf = subparsers.add_parser('config', help=get_txt('h_config'))
		cf.add_argument('-l', '--lang', choices=['ru', 'en'], help=get_txt('h_lang'))
		cf.add_argument('-p', '--proc', type=int, help=get_txt('h_proc'))

	def run(self):
		args = self.parser.parse_args()
		if args.command == 'train': self.cmd_train(args)
		elif args.command == 'apply': self.cmd_apply(args)
		elif args.command == 'plot': self.cmd_plot(args)
		elif args.command == 'view': self.cmd_view(args)
		elif args.command == 'config': self.cmd_config(args)
		else: self.parser.print_help()

	def cmd_train(self, args):
		if not os.path.exists(args.data):
			print(get_txt('err_file').format(args.data)); return
			
		df = pd.read_excel(args.data)
		y, x = df['Y'].values.astype(float), df.drop(columns=['Y']).values.astype(float)
		stds = np.std(x, axis=0); stds[stds == 0] = 1.0
		k_func = nre_lib.get_kernel_func(args.kernel)

		print(get_txt('msg_opt'))
		with Pool(args.proc, nre_lib._init_loo, (x, y, stds, args.kernel)) as pool:
			if args.mode == 'single':
				grid = np.linspace(0.01, 2.0, args.steps)
				# Передаем имя ядра args.kernel
				errors = [nre_lib.calc_loo_error_parallel(c, x, y, stds, args.kernel, pool) for c in tqdm(grid)]
				best_c = float(grid[np.argmin(errors)])
			else:
				def obj(c_vec): 
					return nre_lib.calc_loo_error_parallel(c_vec, x, y, stds, args.kernel, pool)
				res = minimize(obj, np.ones(x.shape[1])*0.5, bounds=[(0.01, 5.0)]*x.shape[1], method='L-BFGS-B')
				best_c = res.x.tolist()

		# Метрики
		y_pred = np.array([nre_lib.predict_nre(pt, x, y, stds, best_c, k_func) for pt in x])
		r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
		ks_p = ks_2samp(y, y_pred).pvalue

		model_data = {
			"data": os.path.abspath(args.data), "best_c": best_c, "kernel": args.kernel,
			"chan_count": x.shape[1], "r2": r2, "ks_p": ks_p
		}
		with open(args.output, 'w') as f: json.dump(model_data, f, indent=4)
		print(f"R2: {r2:.4f}, KS p-value: {ks_p:.4f}")
		print(get_txt('done'))

	def cmd_apply(self, args):
		if not os.path.exists(args.model):
			print(get_txt('err_file').format(args.model)); return
			
		with open(args.model, 'r') as f: m = json.load(f)
		df = pd.read_excel(m['data'])
		y_t, x_t = df['Y'].values, df.drop(columns=['Y']).values
		stds_t = np.std(x_t, axis=0); stds_t[stds_t == 0] = 1.0

		with rasterio.open(args.input) as src:
			if src.count != m['chan_count']:
				print(get_txt('err_chan').format(src.count, m['chan_count'])); return

			meta = src.meta.copy()
			if args.crop < 1.0:
				h, w = int(src.height * args.crop), int(src.width * args.crop)
				win = Window((src.width-w)//2, (src.height-h)//2, w, h)
				meta.update({'height': h, 'width': w, 'transform': src.window_transform(win)})
			else:
				win = None; h, w = src.height, src.width

			meta.update(count=1, dtype='float32', nodata=-999)
			print(get_txt('msg_apply'))
			
			with rasterio.open(args.output, 'w', **meta) as dst:
				data = src.read(window=win).reshape(src.count, -1).T
				chunks = [data[i:i+args.tile] for i in range(0, len(data), args.tile)]
				
				with Pool(args.proc, _init_apply_worker, (x_t, y_t, stds_t, m['best_c'], m['kernel'])) as p:
					results = list(tqdm(p.imap(_apply_unit, chunks), total=len(chunks)))
				
				res_img = np.concatenate(results).reshape(h, w).astype('float32')
				dst.write(res_img, 1)
		print(get_txt('done'))

	def cmd_plot(self, args):
		with open(args.model, 'r') as f: m = json.load(f)
		df = pd.read_excel(m['data'])
		y, x = df['Y'].values, df.drop(columns=['Y']).values
		stds, c = np.std(x, axis=0), np.array(m['best_c'])
		k_f = nre_lib.get_kernel_func(m['kernel'])

		fig, axes = plt.subplots(1, x.shape[1], figsize=(14, 4))
		if x.shape[1] == 1: axes = [axes]
		for i in range(x.shape[1]):
			x_fix = np.mean(x, axis=0)
			line_x = np.linspace(x[:,i].min(), x[:,i].max(), 100)
			line_y = []
			for v in line_x:
				pt = x_fix.copy(); pt[i] = v
				line_y.append(nre_lib.predict_nre(pt, x, y, stds, c, k_f))
			axes[i].scatter(x[:,i], y, alpha=0.2, s=5, color='gray')
			axes[i].plot(line_x, line_y, color='red')
			axes[i].set_title(f"Channel {i+1}")
		plt.tight_layout(); plt.show()

	def cmd_view(self, args):
		with rasterio.open(args.input) as src:
			plt.imshow(src.read(1), cmap='magma'); plt.colorbar(); plt.show()

	def cmd_config(self, args):
		conf = load_config()
		if args.lang: conf['lang'] = args.lang
		if args.proc: conf['proc'] = args.proc
		save_config(conf)
		print(get_txt('done'))

if __name__ == "__main__":
	NRERApp().run()