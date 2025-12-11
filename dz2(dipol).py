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

def plot_results(antenna, theta_deg, D_theta, D_theta_db):
    
    fig1 = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(theta_deg, D_theta)
    plt.xlabel('θ, градусы')
    plt.ylabel('D(θ)')
    plt.title('Диаграмма направленности (линейный масштаб)')
    plt.grid(True)
    plt.xlim(0, 180)
    
    plt.subplot(1, 2, 2)
    plt.plot(theta_deg, D_theta_db)
    plt.xlabel('θ, градусы')
    plt.ylabel('D(θ), дБ')
    plt.title('Диаграмма направленности (логарифмический масштаб)')
    plt.grid(True)
    plt.xlim(0, 180)
    
    plt.tight_layout()
    fig1.savefig('dipole_pattern_cartesian.png', dpi=300)
    
    theta_rad = np.radians(theta_deg)
    
    fig2 = plt.figure(figsize=(12, 5))
    
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    ax1.plot(theta_rad, D_theta)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Диаграмма направленности (линейный масштаб)')
    
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    
    D_theta_db_normalized = D_theta_db - np.min(D_theta_db)
    if np.max(D_theta_db_normalized) > 0:
        D_theta_db_normalized = D_theta_db_normalized / np.max(D_theta_db_normalized)
    
    ax2.plot(theta_rad, D_theta_db_normalized)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('Диаграмма направленности (нормированная дБ шкала)')
    
    plt.tight_layout()
    fig2.savefig('dipole_pattern_polar.png', dpi=300)
    
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
    
    D_theta_db = 10 * np.log10(np.maximum(D_theta, 1e-10))
    
    plot_results(antenna, theta_deg, D_theta, D_theta_db)
    
    save_data(theta_deg, D_theta, D_theta_db)

def save_data(theta_deg, D_theta, D_theta_db):
    with open('dipole_pattern_data.csv', 'w') as f:
        f.write("Theta(deg),D(linear),D(dB)\n")
        for i in range(len(theta_deg)):
            f.write(f"{theta_deg[i]},{D_theta[i]},{D_theta_db[i]}\n")
    
    print("\nДанные сохранены в файл 'dipole_pattern_data.csv'")

if __name__ == '__main__':
    main()