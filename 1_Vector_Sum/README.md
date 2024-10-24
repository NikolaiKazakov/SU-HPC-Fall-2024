# Лабораторная работа 1 - Сложение векторов

## Описание
В данной лабораторной работе реализовано сложение векторов с использованием CUDA на GPU, что позволяет значительно ускорить процесс по сравнению с суммированием на CPU.

Программа генерирует векторы фиксированных размеров со случайными значениями (от 0 до 99), выполняет сложение векторов на CPU и GPU, измеряет время суммирования и вычисляет ускорение.

## О реализации
- Программа генерирует векторы фиксированных размеров со случайными значениями (от 0 до 99), выполняет сложение векторов на CPU и GPU, измеряет время суммирования и вычисляет ускорение.
- **CUDA-ядро (`vectorSum`)** отвечает за параллельное суммирование элементов вектора. Каждый блок использует общую память (`sharedData`) для хранения части вектора, что снижает задержку доступа. Используется алгоритм редукции для объединения элементов, с синхронизацией через `__syncthreads()`.
- Для сравнения представлена **CPU-функция (`cpuVectorSum`)**, которая суммирует значения вектора последовательно.


## Результаты

В таблице ниже показано сравнение времени выполнения кода на CPU и GPU. Как видно из таблицы, распаралеливание дает значительный прирост в скорости, выгода заметно выше при больших размерах векторов. Время указано в секундах.

Speedup = Time(CPU)/Time(GPU).

Вычисления выполнялись в Google Colab.

| Vector Size | CPU Time   | GPU Time   | Speedup  | Vector Sum       |
|-------------|------------|------------|----------|------------------|
| 1000        | 0.000013   | 0.000252   | 0.052991 | 48381            |
| 1400        | 0.000014   | 0.000018   | 0.814995 | 70287            |
| 2100        | 0.000022   | 0.000016   | 1.356950 | 101914           |
| 3000        | 0.000031   | 0.000015   | 2.017784 | 148718           |
| 4300        | 0.000047   | 0.000015   | 3.138024 | 206873           |
| 6200        | 0.000064   | 0.000016   | 4.039522 | 308456           |
| 8900        | 0.000093   | 0.000018   | 5.284819 | 441161           |
| 12700       | 0.000132   | 0.000016   | 8.364555 | 628232           |
| 18300       | 0.000189   | 0.000022   | 8.653709 | 904609           |
| 26400       | 0.000273   | 0.000026   | 10.498482| 1308259          |
| 37900       | 0.000395   | 0.000027   | 14.501285| 1868924          |
| 54600       | 0.000582   | 0.000038   | 15.127643| 2690219          |
| 78500       | 0.000828   | 0.000049   | 16.898635| 3892944          |
| 112900      | 0.001177   | 0.000062   | 19.095881| 5592203          |
| 162400      | 0.001693   | 0.000085   | 19.961681| 8031684          |
| 233600      | 0.002422   | 0.000114   | 21.332238| 11546203         |
| 336000      | 0.003510   | 0.000098   | 35.656033| 16621377         |
| 483300      | 0.005126   | 0.000133   | 38.499970| 23918981         |
| 695200      | 0.007436   | 0.000133   | 55.997236| 34411414         |
| 1000000     | 0.010664   | 0.000177   | 60.239111| 49444922         |

## Ниже представлен график со средним значением ускорения на основе 10-ти запусков программы:
![](vecsum_chart.png)
