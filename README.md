# nre-rsd: Non-parametric Regression Toolkit for Remote Sensing Data Recovery

A comprehensive toolkit implementing non-parametric regression methods for stochastic dependency recovery in remote sensing applications.

---

## English

### Project Description

**nre-rsd** (Nonparametric regression estimation in the problem of restoring stochastic dependencies) is a specialized toolkit designed for non-parametric regression analysis in remote sensing. It implements advanced kernel-based methods to recover stochastic dependencies in geographical data, particularly useful for processing GeoTIFF files and restoring missing or degraded values in satellite imagery and remote sensing datasets.

The project uses Leave-One-Out (LOO) cross-validation to optimize blur coefficients (bandwidth parameters) for non-parametric regression, supporting both single and multi-dimensional optimization modes.

### Dependencies

The project requires the following Python libraries:

- [NumPy](https://numpy.org/) - Numerical computing library
  ```bash
  pip install numpy
  ```

- [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation
  ```bash
  pip install pandas
  ```

- [Matplotlib](https://matplotlib.org/) - Data visualization
  ```bash
  pip install matplotlib
  ```

- [Rasterio](https://rasterio.readthedocs.io/) - Read and write geospatial raster data
  ```bash
  pip install rasterio
  ```

- [SciPy](https://scipy.org/) - Scientific computing tools
  ```bash
  pip install scipy
  ```

- [tqdm](https://github.com/tqdm/tqdm) - Progress bar library
  ```bash
  pip install tqdm
  ```

**Install all dependencies at once:**
```bash
pip install numpy pandas matplotlib rasterio scipy tqdm
```

### Installation

#### From Repository (Cloning)

To clone the repository and set up the project locally:

```bash
git clone https://github.com/HC-NC/nre-rsd.git
cd nre-rsd
pip install numpy pandas matplotlib rasterio scipy tqdm
```

#### From Releases

Download the latest release from the [releases page](https://github.com/HC-NC/nre-rsd/releases):

1. Visit the releases page
2. Download the latest stable version
3. Extract the archive

### User Guide

#### General Usage

The application uses command-line interface with the following structure:

##### For source code

```bash
python main.py [COMMAND] [OPTIONS]
```

##### For nrer.exe

```bash
nrer.exe [COMMAND] [OPTIONS]
```

#### Commands

##### 1. **train** - Model Training

Train a non-parametric regression model by optimizing the blur coefficient(s).

**Usage:**
```bash
python main.py train <data_file> [OPTIONS]
```

**Positional Arguments:**
- `data_file` - Path to Excel file containing sample data (required)

**Options:**
- `-m, --mode {single, multi}` - Optimization mode (default: `single`)
  - `single`: Optimize a single blur coefficient for all dimensions
  - `multi`: Optimize a vector of blur coefficients (one per dimension)
  
- `-k, --kernel {epanechnikov, gaussian, triangular, uniform}` - Kernel function type (default: `epanechnikov`)
  - `epanechnikov`: Epanechnikov kernel
  - `gaussian`: Gaussian (RBF) kernel
  - `triangular`: Triangular kernel
  - `uniform`: Uniform kernel
  
- `-s, --steps <int>` - Number of grid search steps for single mode (default: `100`)

- `-p, --proc <int>` - Number of CPU cores to use (default: auto-detect)

- `-o, --output <path>` - Output JSON model file path (default: `model.json`)

**Example:**
```bash
python main.py train data.xlsx -m single -k gaussian -s 150 -o my_model.json
```

---

##### 2. **apply** - Data Recovery/Prediction

Apply trained model to GeoTIFF images for data recovery and value prediction.

**Usage:**
```bash
python main.py apply <input_file> <model_file> [OPTIONS]
```

**Positional Arguments:**
- `input_file` - Path to source GeoTIFF file (required)
- `model_file` - Path to JSON model file (required)

**Options:**
- `-c, --crop <float>` - Image crop fraction (default: `1.0`)
  - Range: 0.1 to 1.0
  - Crops the center of the image to specified fraction
  - Example: `-c 0.5` processes only the center 50% of the image
  
- `-p, --proc <int>` - Number of CPU cores to use (default: auto-detect)

- `-t, --tile <int>` - Processing tile size for memory efficiency (default: `10000`)
  - Larger values use more memory but may be faster
  - Smaller values use less memory
  
- `-o, --output <path>` - Output GeoTIFF file path (default: `result.tif`)

- `--disable-nodata` - Enable nodata handling (replaces nodata values with mean)

**Example:**
```bash
python main.py apply satellite.tif my_model.json -c 0.8 -o recovered.tif -p 8
```

---

##### 3. **plot** - Visualization

Generate visualizations of regression dependencies for model analysis.

**Usage:**
```bash
python main.py plot <model_file> [OPTIONS]
```

**Positional Arguments:**
- `model_file` - Path to JSON model file (required)

**Options:**
- `--disable-nodata` - Enable nodata handling

**Output:**
- Displays scatter plots with fitted regression curves
- One subplot per data dimension
- Red line shows non-parametric regression fit
- Gray points show original data

**Example:**
```bash
python main.py plot my_model.json
```

---

##### 4. **view** - GeoTIFF Preview

View and visualize GeoTIFF files before processing.

**Usage:**
```bash
python main.py view <input_file>
```

**Positional Arguments:**
- `input_file` - Path to GeoTIFF file to view (required)

**Output:**
- Displays the first band of the GeoTIFF with color mapping
- Includes color bar for value reference

**Example:**
```bash
python main.py view satellite.tif
```

---

##### 5. **config** - Configuration

Manage application-wide settings stored in `config.json`.

**Usage:**
```bash
python main.py config [OPTIONS]
```

**Options:**
- `-l, --lang {ru, en}` - Set interface language
  - `ru`: Russian interface
  - `en`: English interface (default)

- `-p, --proc <int>` - Set default number of CPU cores to use

**Example:**
```bash
python main.py config -l ru -p 4
```

---

#### Model File Format

The model JSON file contains:
```json
{
    "data": "path/to/excel/file.xlsx",
    "best_c": 0.5 or [0.5, 0.3, 0.2],
    "kernel": "epanechnikov",
    "band_count": 3,
    "r2": 0.85,
    "rmse": 1.23,
    "ks_stat": 0.15,
    "ks_p": 0.42,
    "status": "Reliable"
}
```

---

#### Data Format

Input Excel files must contain:
- Column named `Y` with target values
- Remaining columns are independent variables (features)
- All values must be numeric

Example:
```
Y         | X1    | X2    | X3
----------|-------|-------|-------
10.5      | 1.2   | 2.3   | 3.1
11.2      | 1.5   | 2.1   | 3.5
12.1      | 1.8   | 2.5   | 3.2
```

---

## Русский

### Описание проекта

**nre-rsd** реализует методы непараметрической регрессии для восстановления стохастических зависимостей в дистанционном зондировании (ДЗЗ). Инструментарий использует ядерные методы для анализа геоданных, восстановления пропущенных значений в спутниковых снимках и данных дистанционного зондирования.

Проект использует кросс-валидацию с исключением одного объекта (Leave-One-Out) для оптимизации коэффициентов размытости (параметров ядра) в режимах одномерной и многомерной оптимизации.

### Зависимости

Проект требует следующие библиотеки Python:

- [NumPy](https://numpy.org/) - Библиотека численных вычислений
  ```bash
  pip install numpy
  ```

- [Pandas](https://pandas.pydata.org/) - Анализ и обработка данных
  ```bash
  pip install pandas
  ```

- [Matplotlib](https://matplotlib.org/) - Визуализация данных
  ```bash
  pip install matplotlib
  ```

- [Rasterio](https://rasterio.readthedocs.io/) - Работа с геопространственными растровыми данными
  ```bash
  pip install rasterio
  ```

- [SciPy](https://scipy.org/) - Инструменты научных вычислений
  ```bash
  pip install scipy
  ```

- [tqdm](https://github.com/tqdm/tqdm) - Библиотека прогресс-баров
  ```bash
  pip install tqdm
  ```

**Установите все зависимости одной командой:**
```bash
pip install numpy pandas matplotlib rasterio scipy tqdm
```

### Установка

#### Из репозитория (Клонирование)

Чтобы клонировать репозиторий и установить проект локально:

```bash
git clone https://github.com/HC-NC/nre-rsd.git
cd nre-rsd
pip install numpy pandas matplotlib rasterio scipy tqdm
```

#### Из релизов

Загрузите последнюю версию со [страницы релизов](https://github.com/HC-NC/nre-rsd/releases):

1. Перейдите на страницу релизов
2. Загрузите последнюю стабильную версию
3. Распакуйте архив

### Пользовательское руководство

#### Общее использование

Приложение использует командную строку со следующей структурой:

##### Для исходного кода

```bash
python main.py [КОМАНДА] [ОПЦИИ]
```

##### Для nrer.exe

```bash
nrer.exe [КОМАНДА] [ОПЦИИ]
```

#### Команды

##### 1. **train** - Обучение модели

Обучить модель непараметрической регрессии путём оптимизации коэффициента(ов) размытости.

**Использование:**
```bash
python main.py train <файл_данных> [ОПЦИИ]
```

**Позиционные аргументы:**
- `файл_данных` - Путь к файлу Excel с выборкой данных (обязательно)

**Опции:**
- `-m, --mode {single, multi}` - Режим оптимизации (по умолчанию: `single`)
  - `single`: Оптимизировать один коэффициент для всех измерений
  - `multi`: Оптимизировать вектор коэффициентов (по одному на измерение)
  
- `-k, --kernel {epanechnikov, gaussian, triangular, uniform}` - Тип функции ядра (по умолчанию: `epanechnikov`)
  - `epanechnikov`: Ядро Епанечникова
  - `gaussian`: Гауссовское ядро (RBF)
  - `triangular`: Треугольное ядро
  - `uniform`: Равномерное ядро
  
- `-s, --steps <int>` - Количество шагов поиска по сетке для режима single (по умолчанию: `100`)

- `-p, --proc <int>` - Количество ядер процессора (по умолчанию: автоопределение)

- `-o, --output <путь>` - Путь к выходному файлу JSON модели (по умолчанию: `model.json`)

**Пример:**
```bash
python main.py train данные.xlsx -m single -k gaussian -s 150 -o моя_модель.json
```

---

##### 2. **apply** - Восстановление данных

Применить обученную модель к GeoTIFF для восстановления и предсказания значений.

**Использование:**
```bash
python main.py apply <входной_файл> <файл_модели> [ОПЦИИ]
```

**Позиционные аргументы:**
- `входной_файл` - Путь к исходному GeoTIFF файлу (обязательно)
- `файл_модели` - Путь к JSON файлу модели (обязательно)

**Опции:**
- `-c, --crop <float>` - Доля фрагмента изображения (по умолчанию: `1.0`)
  - Диапазон: 0.1 до 1.0
  - Обрезает центр изображения на указанную долю
  - Пример: `-c 0.5` обрабатывает только центральные 50% изображения
  
- `-p, --proc <int>` - Количество ядер процессора (по умолчанию: автоопределение)

- `-t, --tile <int>` - Размер плитки для обработки (по умолчанию: `10000`)
  - Больше значения = больше памяти, но быстрее
  - Меньше значения = меньше памяти, но медленнее
  
- `-o, --output <путь>` - Путь к выходному GeoTIFF файлу (по умолчанию: `result.tif`)

- `--disable-nodata` - Включить обработку пропущенных значений

**Пример:**
```bash
python main.py apply спутник.tif моя_модель.json -c 0.8 -o восстановленный.tif -p 8
```

---

##### 3. **plot** - Визуализация

Создать визуализацию регрессионных зависимостей для анализа модели.

**Использование:**
```bash
python main.py plot <файл_модели> [ОПЦИИ]
```

**Позиционные аргументы:**
- `файл_модели` - Путь к JSON файлу модели (обязательно)

**Опции:**
- `--disable-nodata` - Включить обработку пропущенных значений

**Вывод:**
- Диаграммы рассеяния с подобранными кривыми регрессии
- Один подграфик на каждое измерение данных
- Красная линия показывает непараметрическую регрессию
- Серые точки показывают исходные данные

**Пример:**
```bash
python main.py plot моя_модель.json
```

---

##### 4. **view** - Просмотр GeoTIFF

Просмотреть и визуализировать GeoTIFF файлы перед обработкой.

**Использование:**
```bash
python main.py view <входной_файл>
```

**Позиционные аргументы:**
- `входной_файл` - Путь к GeoTIFF файлу для просмотра (обязательно)

**Вывод:**
- Отображает первый канал GeoTIFF с цветовой картой
- Включает цветовую шкалу для справки по значениям

**Пример:**
```bash
python main.py view спутник.tif
```

---

##### 5. **config** - Конфигурация

Управлять параметрами приложения, сохранёнными в `config.json`.

**Использование:**
```bash
python main.py config [ОПЦИИ]
```

**Опции:**
- `-l, --lang {ru, en}` - Установить язык интерфейса
  - `ru`: Русский интерфейс
  - `en`: Английский интерфейс (по умолчанию)

- `-p, --proc <int>` - Установить количество ядер процессора по умолчанию

**Пример:**
```bash
python main.py config -l ru -p 4
```

---

#### Формат файла модели

Файл JSON модели содержит:
```json
{
    "data": "путь/к/файлу/excel.xlsx",
    "best_c": 0.5 или [0.5, 0.3, 0.2],
    "kernel": "epanechnikov",
    "band_count": 3,
    "r2": 0.85,
    "rmse": 1.23,
    "ks_stat": 0.15,
    "ks_p": 0.42,
    "status": "Reliable"
}
```

---

#### Формат входных данных

Файлы Excel должны содержать:
- Столбец с именем `Y` целевыми значениями
- Остальные столбцы - независимые переменные (признаки)
- Все значения должны быть числовыми

Пример:
```
Y         | X1    | X2    | X3
----------|-------|-------|-------
10.5      | 1.2   | 2.3   | 3.1
11.2      | 1.5   | 2.1   | 3.5
12.1      | 1.8   | 2.5   | 3.2
```

---

## License

This project is licensed under the terms specified in the LICENSE file. Please refer to [LICENSE](LICENSE) for more information.