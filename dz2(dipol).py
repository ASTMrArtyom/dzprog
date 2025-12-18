import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

class DipoleAntenna:
    def __init__(self, frequency, length_ratio):
        self.frequency = frequency
        self.length_ratio = length_ratio
        self.c = 3e8
        self.Z0 = 120 * np.pi
        
        self.wavelength = self.c / (frequency * 1e9)
        self.k = 2 * np.pi / self.wavelength
        self.l = self.length_ratio * self.wavelength / 2
        
    def radiation_pattern(self, theta):
        if theta == 0 or theta == np.pi:
            return 0
        
        kl = self.k * self.l
        numerator = np.cos(kl * np.cos(theta)) - np.cos(kl)
        denominator = np.sin(theta) * (1 - np.cos(kl))
        
        if abs(denominator) < 1e-10:
            return 0
        
        F_theta = numerator / denominator
        return F_theta
    
    def calculate_Dmax(self):
        def integrand(theta):
            F = self.radiation_pattern(theta)
            return F**2 * np.sin(theta)
        
        integral, error = quad(integrand, 0, np.pi)
        
        if integral == 0:
            return 1
        
        Dmax = 4 * np.pi / (2 * np.pi * integral)
        return Dmax
    
    def directivity(self, theta):
        F_theta = self.radiation_pattern(theta)
        Dmax = self.calculate_Dmax()
        return F_theta**2 * Dmax

def read_cst_data(filename):
    theta_cst = []
    D_theta_cst = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Пропускаем первые две строки (заголовок)
        for line in lines[2:]:
            parts = line.strip().split()
            
            if len(parts) >= 3:
                try:
                    theta_val = float(parts[0])
                    D_val = float(parts[2])  # Abs(Dir.)
                    # Преобразуем значение направленности
                    theta_cst.append(theta_val)
                    D_theta_cst.append(D_val)
                except ValueError:
                    # Если не удалось преобразовать в число, пропускаем строку
                    continue
    
    return np.array(theta_cst), np.array(D_theta_cst)

def plot_results(antenna, theta_deg, D_theta, D_theta_db, cst_data=None):
    theta_rad = np.radians(theta_deg)
    
    fig = plt.figure(figsize=(18, 10))
    
    # График 1: Линейный масштаб
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(theta_deg, D_theta, 'b-', linewidth=2, label='Аналитический расчет')
    
    if cst_data is not None:
        theta_cst, D_theta_cst = cst_data
        # Масштабируем CST данные, чтобы их максимум совпал с максимумом аналитического графика
        scale_factor = np.max(D_theta) / np.max(D_theta_cst)
        D_theta_cst_scaled = D_theta_cst * scale_factor
        ax1.plot(theta_cst, D_theta_cst_scaled, 'ro', markersize=4, alpha=0.7, label='CST данные')
        ax1.legend()
    
    ax1.set_xlabel('θ, градусы')
    ax1.set_ylabel('D(θ)')
    ax1.set_title('Диаграмма направленности (линейный масштаб)')
    ax1.grid(True)
    ax1.set_xlim(0, 180)
    
    # График 2: Логарифмический масштаб
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(theta_deg, D_theta_db, 'b-', linewidth=2, label='Аналитический расчет')
    
    if cst_data is not None:
        theta_cst, D_theta_cst = cst_data
        # Масштабируем CST данные для логарифмического графика
        scale_factor = np.max(D_theta) / np.max(D_theta_cst)
        D_theta_cst_scaled = D_theta_cst * scale_factor
        D_theta_cst_scaled_db = 10 * np.log10(np.maximum(D_theta_cst_scaled, 1e-10))
        ax2.plot(theta_cst, D_theta_cst_scaled_db, 'ro', markersize=4, alpha=0.7, label='CST данные')
        ax2.legend()
    
    ax2.set_xlabel('θ, градусы')
    ax2.set_ylabel('D(θ), дБ')
    ax2.set_title('Диаграмма направленности (логарифмический масштаб)')
    ax2.grid(True)
    ax2.set_xlim(0, 180)
    
    # График 3: Полярные координаты (линейный масштаб)
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(theta_rad, D_theta, 'b-', linewidth=2, label='Аналитический расчет')
    
    # Добавляем точки из CST в полярные координаты
    if cst_data is not None:
        theta_cst_rad = np.radians(theta_cst)
        # Используем масштабированные данные
        scale_factor = np.max(D_theta) / np.max(D_theta_cst)
        D_theta_cst_scaled = D_theta_cst * scale_factor
        ax3.plot(theta_cst_rad, D_theta_cst_scaled, 'ro', markersize=4, alpha=0.7, label='CST данные')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
    
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_title('Диаграмма направленности (линейный масштаб)')
    
    # График 4: Полярные координаты (логарифмический масштаб)
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    # Нормируем для полярных координат (для лучшей визуализации)
    D_theta_db_normalized = D_theta_db - np.min(D_theta_db)
    if np.max(D_theta_db_normalized) > 0:
        D_theta_db_normalized = D_theta_db_normalized / np.max(D_theta_db_normalized)
    
    ax4.plot(theta_rad, D_theta_db_normalized, 'b-', linewidth=2, label='Аналитический расчет')

    if cst_data is not None:
        theta_cst_rad = np.radians(theta_cst)
        # Масштабируем CST данные
        scale_factor = np.max(D_theta) / np.max(D_theta_cst)
        D_theta_cst_scaled = D_theta_cst * scale_factor
        D_theta_cst_scaled_db = 10 * np.log10(np.maximum(D_theta_cst_scaled, 1e-10))
        # Нормируем для полярных координат
        D_theta_cst_scaled_db_normalized = D_theta_cst_scaled_db - np.min(D_theta_cst_scaled_db)
        if np.max(D_theta_cst_scaled_db_normalized) > 0:
            D_theta_cst_scaled_db_normalized = D_theta_cst_scaled_db_normalized / np.max(D_theta_cst_scaled_db_normalized)
        
        ax4.plot(theta_cst_rad, D_theta_cst_scaled_db_normalized, 'ro', markersize=4, alpha=0.7, label='CST данные')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
    
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_title('Диаграмма направленности (нормированная дБ шкала)')
    
    plt.tight_layout()
    fig.savefig('dipole_pattern_combined_with_cst_scaled.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    frequency = 8.0
    length_ratio = 1.2
    
    print(f"Расчет для варианта 8:")
    print(f"Частота: {frequency} ГГц")
    print(f"Отношение длины вибратора к длине волны: 2l/λ = {length_ratio}")
    
    antenna = DipoleAntenna(frequency, length_ratio)
    
    print(f"Длина волны: {antenna.wavelength:.6f} м")
    print(f"Длина плеча вибратора: {antenna.l:.6f} м")
    print(f"Волновое число: {antenna.k:.6f} рад/м")
    
    Dmax = antenna.calculate_Dmax()
    Dmax_db = 10 * math.log10(Dmax)
    
    print(f"\nМаксимальный КНД:")
    print(f"Dmax = {Dmax:.4f} раз")
    print(f"Dmax = {Dmax_db:.4f} дБ")
    
    theta_deg = np.linspace(0, 180, 361)
    theta_rad = np.radians(theta_deg)
    
    D_theta = np.zeros_like(theta_deg)
    for i, theta in enumerate(theta_rad):
        D_theta[i] = antenna.directivity(theta)
    
    print(f"\nМаксимум аналитического графика: {np.max(D_theta):.4f}")
    
    D_theta_db = 10 * np.log10(np.maximum(D_theta, 1e-10))
    
    # Чтение данных из файла CST
    print("\nЧтение данных из файла CST...")
    try:
        cst_theta, cst_D = read_cst_data('data.txt')
        print(f"Прочитано {len(cst_theta)} точек из файла CST")
        print(f"Диапазон углов: от {np.min(cst_theta)} до {np.max(cst_theta)} градусов")
        print(f"Диапазон значений D(θ): от {np.min(cst_D)} до {np.max(cst_D)}")
        print(f"Максимум CST данных: {np.max(cst_D):.4f}")
        
        # Вычисляем масштабирующий коэффициент
        scale_factor = np.max(D_theta) / np.max(cst_D)
        print(f"Масштабирующий коэффициент для CST данных: {scale_factor:.4f}")
        print(f"Максимум CST после масштабирования: {np.max(cst_D) * scale_factor:.4f}")
        
    except Exception as e:
        print(f"Ошибка при чтении файла CST: {e}")
        cst_data = None
    else:
        cst_data = (cst_theta, cst_D)
    
    plot_results(antenna, theta_deg, D_theta, D_theta_db, cst_data)
    
    save_data(theta_deg, D_theta, D_theta_db)

def save_data(theta_deg, D_theta, D_theta_db):
    with open('dipole_pattern_data.csv', 'w') as f:
        f.write("Theta(deg),D(linear),D(dB)\n")
        for i in range(len(theta_deg)):
            f.write(f"{theta_deg[i]},{D_theta[i]},{D_theta_db[i]}\n")
    
    print("\nДанные сохранены в файл 'dipole_pattern_data.csv'")

if __name__ == '__main__':
    main()