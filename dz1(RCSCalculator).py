import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
import csv

class RCSCalculator:
    def __init__(self, diameter):
        self.radius = diameter / 2
        self.c = 3e8

    def calculate_rcs(self, freq):
        lam = self.c / freq
        k = 2 * np.pi / lam
        x = k * self.radius
        total = 0j
        n = 1
        tolerance = 1e-10
        max_iter = 1000

        while n <= max_iter:
            jn = spherical_jn(n, x)
            yn = spherical_yn(n, x)
            hn = jn + 1j * yn
            
            jn_deriv = spherical_jn(n, x, derivative=True)
            yn_deriv = spherical_yn(n, x, derivative=True)
            hn_deriv = jn_deriv + 1j * yn_deriv

            an = jn / hn
            numerator = jn + x * jn_deriv
            denominator = hn + x * hn_deriv
            bn = numerator / denominator

            term = (-1)**n * (n + 0.5) * (bn - an)
            total += term

            if abs(term) < tolerance:
                break
            n += 1

        rcs = (lam**2 / np.pi) * abs(total)**2
        return rcs

class ResultWriter:
    def __init__(self, filename, format_type):
        self.filename = filename
        self.format_type = format_type

    def write(self, freqs, rcs_values):
        if self.format_type == 2:
            with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Frequency (Hz)', 'RCS (m2)'])
                for i, (f, rcs) in enumerate(zip(freqs, rcs_values)):
                    writer.writerow([i + 1, f, rcs])

def read_parameters_from_file(filename, variant):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.strip().startswith('#'):
            continue
        
        data = line.split()
        if len(data) >= 4:
            try:
                var_num = int(data[0])
                if var_num == variant:
                    D = float(data[1])
                    fmin = float(data[2])
                    fmax = float(data[3])
                    return D, fmin, fmax
            except ValueError:
                continue
    
    raise ValueError(f"Вариант {variant} не найден в файле {filename}")

def main():
    try:
        D, fmin, fmax = read_parameters_from_file('task_rcs_02.txt', 8)
        print(f"Прочитаны параметры: D={D}, fmin={fmin}, fmax={fmax}")
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return

    output_format = 2

    calculator = RCSCalculator(D)

    freqs = np.linspace(fmin, fmax, 1000)
    rcs_values = [calculator.calculate_rcs(f) for f in freqs]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, rcs_values)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RCS (m²)')
    plt.title(f'ЭПР идеально проводящей сферы (D = {D} м)')
    plt.grid(True)
    plt.savefig('rcs_plot.png')
    plt.show()

    writer = ResultWriter('result.csv', output_format)
    writer.write(freqs, rcs_values)
    print("Результаты сохранены в result.csv")

if __name__ == '__main__':
    main()