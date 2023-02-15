# Wprowadzenie do Systemów Wizyjnych

## Politechnika Poznańska, Instytut Robotyki i Inteligencji Maszynowej

<p align="center">
  <img width="180" height="180" src="./readme_files/logo.svg">
</p>

# **Projekt zaliczeniowy: zliczanie cukierków**

## Zadanie

Zadanie projektowe polega na przygotowaniu algorytmu wykrywania i zliczania kolorowych cukierków znajdujących się na zdjęciach. Dla uproszczenia zadania w zbiorze danych występują jedynie 4 kolory cukierków:
- czerwony
- żółty
- zielony
- fioletowy

Wszystkie zdjęcia zostały zarejestrowane "z góry", ale z różnej wysokości i pod różnym kątem. Ponadto obrazy różnią się między sobą poziomem oświetlenia oraz oczywiście ilością cukierków.

Poniżej przedstawione zostało przykładowe zdjęcie ze zbioru danych i poprawny wynik detekcji dla niego:

<p align="center">
  <img width="750" height="500" src="./data/37.jpg">
</p>


```bash
{
  ...,
  "37.jpg": {
    "red": 2,
    "yellow": 2,
    "green": 2,
    "purple": 2
  },
  ...
}
```

## Struktura projektu

Szablon projektu zliczania cukierków na zdjęciach dostępny jest w serwisie [GitHub](https://github.com/PUTvision/WDPOProject) i ma następującą strukturę:

```bash
.
├── data
│   ├── 00.jpg
│   ├── 01.jpg
│   └── 02.jpg
├── readme_files
├── detect.py
├── README.md
└── requirements.txt
```

Katalog [`data`](./data) zawiera przykłady, na podstawie których w pliku [`detect.py`](./detect.py) przygotowany został algorytm zliczania cukierków. 

## Wywołanie programu

Skrypt `detect.py` przyjmuje 2 parametry wejściowe:
- `data_path` - ścieżkę do folderu z danymi (zdjęciami)
- `output_file_path` - ścieżkę do pliku z wynikami

```bash
$ python3 detect.py --help

Options:
  -p, --data_path TEXT         Path to data directory
  -o, --output_file_path TEXT  Path to output file
  --help                       Show this message and exit.
```

W konsoli systemu Linux skrypt można wywołać z katalogu projektu w następujący sposób:

```bash
python3 detect.py -p ./data -o ./results.json
```

## Ewaluacja rozwiązań

Algortymu zliczania cukierków jest oceniany na podstawie poniższeo wzoru:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\bg_white&space;Mean&space;Absolute&space;Relative&space;Percentage&space;Error&space;[%]&space;=&space;\frac{100}{n}\sum_{t=0}^{n-1}\frac{\left|y_{r}-\widehat{y_{r}}\right|&space;&plus;&space;\left|y_{y}-\widehat{y_{y}}\right|&space;&plus;&space;\left|y_{g}-\widehat{y_{g}}\right|&space;&plus;&space;\left|y_{p}-\widehat{y_{p}}\right|}{y_{r}&plus;y_{y}&plus;y_{g}&plus;y_{p}}" title="\bg_white Mean Absolute Relative Percentage Error [%] = \frac{100}{n}\sum_{t=0}^{n-1}\frac{\left|y_{a}-\widehat{y_{a}}\right| + \left|y_{b}-\widehat{y_{b}}\right| + \left|y_{o}-\widehat{y_{o}}\right|}{y_{a}+y_{b}+y_{o}}" style="background-color: white"/>
</p>

Gdzie:
- ![](https://render.githubusercontent.com/render/math?math=n) oznacza liczbę obrazów
- ![](https://render.githubusercontent.com/render/math?math=y_x) oznacza rzeczywistą ilość danego koloru
- ![](https://render.githubusercontent.com/render/math?math=\widehat{y_x}) oznacza przewidzianą ilość danego koloru
