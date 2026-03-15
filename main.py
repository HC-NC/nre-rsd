import pandas as pd
import numpy as np
import argparse
import math
import json
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats import ks_2samp

# Константы
SQRT5 = math.sqrt(5)

# Глобальные переменные для дочерних процессов
x_train = None
y_train = None
train_stds = None

def init_worker(x_p, y_p, s_p):
    global x_train, y_train, train_stds
    x_train = x_p
    y_train = y_p
    train_stds = s_p

def epanechnikov_kernel(u):
    """Ядро Эпанечникова Ф(u)"""
    res = (3 / (4 * SQRT5)) * (1 - (u**2) / 5)
    return np.where(np.abs(u) < SQRT5, res, 0)

def predict_point(x_target, c):
    """Реализация формулы 5.5 для одной точки x"""
    # (x_target - x_i) / (c * sigma)
    diffs = (x_target - x_train) / (c * train_stds)
    # П Ф(...)
    weights = np.prod(epanechnikov_kernel(diffs), axis=1)
    
    sum_w = np.sum(weights)
    if sum_w > 1e-15:
        return np.sum(weights * y_train) / sum_w
    else:
        return np.mean(y_train)

def W_loo(c):
    """Формула 5.15: Оценка среднеквадратичной ошибки (Leave-One-Out)"""
    n = len(y_train)
    total_error = 0
    for j in range(n):
        diffs = (x_train[j] - x_train) / (c * train_stds)
        weights = np.prod(epanechnikov_kernel(diffs), axis=1)
        weights[j] = 0 # Исключаем саму точку
        
        sum_w = np.sum(weights)
        y_hat = np.sum(weights * y_train) / sum_w if sum_w > 1e-15 else np.mean(y_train)
        total_error += (y_train[j] - y_hat)**2
    return total_error / n

def wrapper_opt(v):
    c = v * 0.001 # Шаг сетки
    return [c, W_loo(c)]

def main():
    parser = argparse.ArgumentParser(description="Непараметрическая регрессия")
    parser.add_argument("-d", "--data", type=str, required=True, help="Путь к Excel файлу")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Кол-во шагов оптимизации c")
    parser.add_argument("-p", "--proc", type=int, default=4, help="Кол-во процессов")
    parser.add_argument("-o", "--output", type=str, default="model_params.json", help="Файл для сохранения параметров")
    args = parser.parse_args()

    # 1. Загрузка данных
    try:
        df = pd.read_excel(args.data)
        y_data = df['Y'].values.astype(float)
        x_data = df.drop(columns=['Y']).values.astype(float)
        stds = np.std(x_data, axis=0)
        stds[stds == 0] = 1.0 # Защита от деления на 0
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    # 2. Оптимизация коэффициента размытости c (поиск минимума W)
    print(f"--- Шаг 1: Оптимизация c (шагов: {args.steps}) ---")
    with Pool(processes=args.proc, initializer=init_worker, initargs=(x_data, y_data, stds)) as p:
        opt_results = list(tqdm(p.imap(wrapper_opt, range(1, args.steps + 1)), total=args.steps))
    
    best_c, min_mse = min(opt_results, key=lambda x: x[1])
    print(f"Оптимальное c = {best_c:.4f} (MSE = {min_mse:.6f})")

    # 3. Расчет финальной зависимости и параметров (R^2, СКО, Колмогоров)
    print("--- Шаг 2: Расчет метрик достоверности ---")
    init_worker(x_data, y_data, stds)
    y_pred = np.array([predict_point(x, best_c) for x in x_data])

    # Коэффициент детерминации R^2
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Среднеквадратичное отклонение (RMSE)
    rmse = math.sqrt(ss_res / len(y_data))

    # Критерий Смирнова-Колмогорова (сравнение распределений y_true и y_pred)
    ks_stat, ks_p = ks_2samp(y_data, y_pred)

    # 4. Вывод и сохранение
    results = {
        "best_c": best_c,
        "r_squared": r_squared,
        "rmse": rmse,
        "kolmogorov_stat": ks_stat,
        "kolmogorov_p_value": ks_p,
        "status": "Reliable" if ks_p > 0.05 else "Distributions differ"
    }

    print("\nПараметры модели:")
    print(f"  R^2 (детерминация): {r_squared:.4f}")
    print(f"  СКО (RMSE): {rmse:.4f}")
    print(f"  Критерий Колм.-Смирнова: стат={ks_stat:.4f}, p-value={ks_p:.4f}")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nПараметры сохранены в {args.output}")

if __name__ == "__main__":
    main()