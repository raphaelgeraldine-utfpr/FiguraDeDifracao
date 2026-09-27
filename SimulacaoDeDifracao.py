import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def simular_difracao(caminho_imagem, lamb, Z, altura_objeto_m):
    img = Image.open(caminho_imagem).convert('L')
    campo_entrada = np.array(img, dtype=float) / 255.0
    
    Ny, Nx = campo_entrada.shape
    
    dy = altura_objeto_m / Ny
    dx = dy 
    
    espectro_complexo = np.fft.fftshift(np.fft.fft2(campo_entrada))
    
    intensidade = np.abs(espectro_complexo)**2
    intensidade_log = np.log1p(intensidade)
    
    freq_x = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    
    X_anteparo_cm = (freq_x * lamb * Z) * 100
    Y_anteparo_cm = (freq_y * lamb * Z) * 100
    
    limites_anteparo = [X_anteparo_cm[0], X_anteparo_cm[-1], Y_anteparo_cm[0], Y_anteparo_cm[-1]]
    limites_objeto = [-Nx*dx*1000/2, Nx*dx*1000/2, -Ny*dy*1000/2, Ny*dy*1000/2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(campo_entrada, cmap='gray', extent=limites_objeto)
    axes[0].set_title("Objeto / Abertura de Entrada")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    
    im = axes[1].imshow(intensidade_log, cmap='inferno', extent=limites_anteparo)
    axes[1].set_title(f"Figura de Difração no Anteparo (Z = {Z} m)")
    axes[1].set_xlabel("X no anteparo (cm)")
    axes[1].set_ylabel("Y no anteparo (cm)")
    fig.colorbar(im, ax=axes[1], label="Intensidade (Escala Logarítmica)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    CAMINHO = "assets/img.png"
    LAMBDA = 632e-9     # 632 nm
    DISTANCIA_Z = 2.0    # 2 m
    ALTURA_A = 1e-3      # 1 mm
    
    simular_difracao(caminho_imagem=CAMINHO, lamb=LAMBDA, Z=DISTANCIA_Z, altura_objeto_m=ALTURA_A)




