import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import argparse
import json
import math
import os
from multiprocessing import Pool
from tqdm import tqdm

# Глобальные переменные для дочерних процессов
SQRT5 = math.sqrt(5)
x_train = None
y_train = None
train_stds = None
best_c = None

def init_worker(x_p, y_p, s_p, c_p):
	global x_train, y_train, train_stds, best_c
	x_train, y_train, train_stds, best_c = x_p, y_p, s_p, c_p

def epanechnikov_kernel(u):
	res = (3 / (4 * SQRT5)) * (1 - (u**2) / 5)
	return np.where(np.abs(u) < SQRT5, res, 0)

def predict_pixel_block(block):
	results = []
	for pixel_x in block:
		# Реализация формулы 5.5
		diffs = (pixel_x - x_train) / (best_c * train_stds)
		weights = np.prod(epanechnikov_kernel(diffs), axis=1)
		sum_w = np.sum(weights)
		
		if sum_w > 1e-15:
			results.append(np.sum(weights * y_train) / sum_w)
		else:
			results.append(np.nan) 
	return results

def main():
	parser = argparse.ArgumentParser(description="Восстановление Y по GeoTIFF")
	parser.add_argument("-i", "--input", type=str, required=True, help="Исходный GeoTIFF")
	parser.add_argument("-m", "--model", type=str, required=True, help="JSON файл параметров")
	parser.add_argument("-o", "--output", type=str, default="reconstructed_Y.tif", help="Выходной файл")
	parser.add_argument("-c", "--crop", type=float, default=1.0, help="Доля изображения (напр. 0.2)")
	parser.add_argument("-p", "--proc", type=int, default=4, help="Кол-во процессов")
	args = parser.parse_args()

	# 1. Загрузка параметров из JSON
	with open(args.model, "r", encoding='utf-8') as f:
		model_params = json.load(f)
	
	c_val = model_params["best_c"]
	train_path = model_params["data"] # Берем путь из JSON

	# 2. Загрузка обучающей выборки
	print(f"Загрузка обучающих данных по пути: {train_path}")
	if not os.path.exists(train_path):
		print(f"Ошибка: Файл данных не найден по указанному в JSON пути!")
		return

	df_train = pd.read_excel(train_path)
	y_vals = df_train['Y'].values.astype(float)
	x_vals = df_train.drop(columns=['Y']).values.astype(float)
	s_vals = np.std(x_vals, axis=0)
	s_vals[s_vals == 0] = 1.0

	# 3. Работа с GeoTIFF
	with rasterio.open(args.input) as src:
		meta = src.meta.copy()
		
		if args.crop < 1.0:
			h_size, w_size = int(src.height * args.crop), int(src.width * args.crop)
			off_h, off_w = (src.height - h_size) // 2, (src.width - w_size) // 2
			read_window = Window(off_w, off_h, w_size, h_size)
			meta.update({
				'height': h_size, 'width': w_size,
				'transform': src.window_transform(read_window)
			})
		else:
			read_window = None

		print(f"Чтение каналов B1-B4...")
		img_data = src.read([1, 2, 3, 4], window=read_window)
		_, h, w = img_data.shape
		flat_img = img_data.reshape(4, -1).T 

	# 4. Расчет
	print(f"Запуск восстановления (c={c_val:.4f}). Обработка {h*w} пикселей...")
	chunk_size = max(1, len(flat_img) // (args.proc * 10))
	chunks = [flat_img[i:i + chunk_size] for i in range(0, len(flat_img), chunk_size)]

	with Pool(processes=args.proc, initializer=init_worker, initargs=(x_vals, y_vals, s_vals, c_val)) as p:
		raw_results = list(tqdm(p.imap(predict_pixel_block, chunks), total=len(chunks)))

	reconstructed_img = np.concatenate(raw_results).reshape(h, w).astype(np.float32)

	# 5. Сохранение
	meta.update(count=1, dtype='float32', nodata=np.nan)
	with rasterio.open(args.output, 'w', **meta) as dst:
		dst.write(reconstructed_img, 1)

	print(f"Готово. Результат: {args.output}")

if __name__ == "__main__":
	main()