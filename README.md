
# 🌟 Gauss Yumuşatma Filtresi Uygulaması 🎨

Bu Python programı, bir Excel dosyasındaki verilere Gauss yumuşatma filtresi uygular ve sonuçları yeni bir Excel dosyasına kaydeder.

📄 **İçindekiler**
- [Nasıl Çalışır?](#-nasıl-çalışır)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Kod Açıklaması](#-kod-açıklaması)
- [Çıktı](#-çıktı)
- [Katkıda Bulunanlar](#-katkıda-bulunanlar)

## 🎯 Nasıl Çalışır?

1. **Veriyi Yükleme:** Excel dosyasındaki veriler pandas DataFrame olarak yüklenir ve numpy array'e dönüştürülür.
2. **Gauss Çekirdeği Oluşturma:** Belirtilen boyutta ve standart sapmada bir Gauss çekirdeği oluşturulur.
3. **Gauss Filtresi Uygulama:** Çekirdek, verilerin her bir parçasına uygulanır.
4. **Sonuçları Kaydetme:** Elde edilen sonuçlar yeni bir Excel dosyasına kaydedilir.

## 🛠️ Kurulum

### Python Kurulumu:

Python'u bilgisayarınıza yükleyin (eğer yüklü değilse): [Python İndirme Sayfası](https://www.python.org/downloads/)

### Projeyi Başlatma:

Projeyi bilgisayarınıza indirin veya kopyalayın.
Terminal veya komut istemcisinde projenin bulunduğu dizine gidin.

### Gerekli Kütüphanelerin Kurulumu:

Gerekli kütüphaneleri yüklemek için şu komutları çalıştırın:

```sh
pip install numpy pandas openpyxl
```

## 🎮 Kullanım

1. **Excel dosyasını hazırlayın:** `excel_files` klasörüne `soru3_data.xlsx` adında bir Excel dosyası yerleştirin. Verilerin ilk sayfada ve başlıksız olması gerekmektedir.
2. **Python betiğini çalıştırın:** 
    ```sh
    python gauss_blur.py
    ```
3. **Sonuçları inceleyin:** Sonuçlar `excel_files` klasöründe `gaussian_blur_results.xlsx` dosyasına kaydedilecektir.

## 📜 Kod Açıklaması

### Gauss Çekirdeği Oluşturma Fonksiyonu

```python
def gauss_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)
```

### Gauss Filtresi Uygulama

```python
for i in range(output_size):
    for j in range(output_size):
        region = data[i:i + kernel_size, j:j + kernel_size]
        output_data[i, j] = np.sum(region * kernel)
```

### Tam Kod

```python
import numpy as np
import pandas as pd

# Dosya yolunu tam olarak belirtiyoruz
file_path = 'excel_files/soru3_data.xlsx'

# Dosyayı yükle
df = pd.read_excel(file_path, header=None)

# Veriyi numpy array'e çevir
data = df.to_numpy()

# Gauss yumuşatma filtresini oluştur
def gauss_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)

# 3x3 Gauss filtresi
kernel_size = 3
sigma = 1
kernel = gauss_kernel(kernel_size, sigma)

# Yeni matris (30x30 boyutunda)
output_size = data.shape[0] - kernel_size + 1
output_data = np.zeros((output_size, output_size))

# İç içe döngülerle Gauss filtresini uygulama
for i in range(output_size):
    for j in range(output_size):
        # Görüntü parçası
        region = data[i:i + kernel_size, j:j + kernel_size]
        # Filtreyi uygula
        output_data[i, j] = np.sum(region * kernel)

# Ondalık kısımları atarak tam sayıya çevir
output_data = np.round(output_data).astype(int)

# Sonucu gösterme
print("Gauss Yumuşatma Filtresi Sonucu:\n", output_data)

# Sonucu bir DataFrame olarak oluşturma
output_df = pd.DataFrame(output_data)

# Sonucu bir Excel dosyasına kaydetme
output_file_path = 'excel_files/gaussian_blur_results.xlsx'
output_df.to_excel(output_file_path, index=False, header=False)
print(f'Sonuç başarıyla {output_file_path} dosyasına kaydedildi.')
```

## 📈 Çıktı

Sonuçlar `excel_files/gaussian_blur_results.xlsx` dosyasına kaydedilecektir. Bu dosya, Gauss yumuşatma filtresinin uygulanmış halini içerecektir.

