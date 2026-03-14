import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import math

# Константы
SQRT5 = math.sqrt(5)
DATA_PATH = "1.xlsx"
BLUR_COEF_OPT_COLUMNS = ["C", "W(c)"]

# Глобальная переменная для дочерних процессов
sample = None

def init_worker(shared_df):
    """Эта функция вызывается при старте каждого процесса в Pool"""
    global sample
    sample = shared_df

def epanechnikov_vectorized(u):
    """Векторная версия ядра Эпанечникова"""
    # Результат: 3/(4*sqrt(5)) * (1 - u^2/5) если |u| < sqrt(5), иначе 0
    res = (3 / (4 * SQRT5)) * (1 - (u**2) / 5)
    return np.where(np.abs(u) < SQRT5, res, 0)

def W(c):
    """Оптимизированный расчет W(c) через Numpy"""
    global sample
    
    # Выделяем целевую переменную Y и предикторы X
    y_values = sample['Y'].values
    # Берем все колонки кроме Y
    x_df = sample.drop(columns=['Y'])
    
    # Стандартное отклонение для каждой колонки
    stds = x_df.std().values
    x_values = x_df.values
    
    n = len(y_values)
    total_error = 0

    # Цикл Leave-One-Out (увы, совсем без него сложно для памяти, 
    # но внутренние расчеты теперь векторные)
    for j in range(n):
        # Вычисляем разности (x_j - x_i) для всех i != j
        # Нормируем на (c * std)
        diffs = (x_values[j] - x_values) / (c * stds)
        
        # Применяем ядро к каждой компоненте вектора разностей
        kernels = epanechnikov_vectorized(diffs)
        
        # Перемножаем ядра по строкам (продуктивное ядро)
        # Исключаем текущий индекс j, зануляя его вес
        weights = np.prod(kernels, axis=1)
        weights[j] = 0 
        
        sum_weights = np.sum(weights)
        
        if sum_weights != 0:
            y_hat = np.sum(weights * y_values) / sum_weights
            total_error += (y_values[j] - y_hat) ** 2
            
    return total_error / n

def VToWcOpt(v):
    """Функция-обертка для Pool"""
    c = v * 0.01
    return [c, W(c)]

def main():
    global sample
    
    try:
        # Загрузка данных
        sample = pd.read_excel(DATA_PATH)
        print("Данные загружены:")
        print(sample.head(), '\n')
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    # Настраиваем Pool
    # initializer гарантирует, что переменная sample будет во всех процессах
    num_steps = 100
    tasks = list(range(1, num_steps + 1))
    
    print(f"Запуск расчетов в 2 потока...")
    with Pool(processes=2, initializer=init_worker, initargs=(sample,)) as p:
        # Используем tqdm для красивого прогресс-бара
        blur_coefs = list(tqdm(p.imap(VToWcOpt, tasks), total=num_steps))

    # Сбор результатов
    blur_coefs_df = pd.DataFrame(blur_coefs, columns=BLUR_COEF_OPT_COLUMNS)

    print('\nРезультаты оптимизации:')
    print(blur_coefs_df.describe(), '\n')

    # Сохранение
    output_name = "wc_opt_" + DATA_PATH
    blur_coefs_df.to_excel(output_name, index=False)
    print(f"Файл сохранен как: {output_name}")

if __name__ == "__main__":
    main()