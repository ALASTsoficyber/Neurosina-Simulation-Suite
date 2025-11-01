import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# شبیه‌سازی داده QEEG برای 5 شبکه فعال
fs = 500  # نرخ نمونه‌برداری (Hz)
t = np.linspace(0, 10, fs*10)  # زمان 10 ثانیه

# تعریف فرکانس‌های غالب هر شبکه
dmn_freq = 10  # آلفا
acn_freq = 4   # تتا
tpj_freq = 8   # آلفا-تتا
metacog_freq = 15  # بتا
gnw_freq = 40  # گاما

# تولید سیگنال برای هر شبکه با نویز سفید
np.random.seed(42)
dmn_signal = np.sin(2*np.pi*dmn_freq*t) + 0.5*np.random.randn(len(t))
acn_signal = np.sin(2*np.pi*acn_freq*t) + 0.5*np.random.randn(len(t))
tpj_signal = np.sin(2*np.pi*tpj_freq*t) + 0.5*np.random.randn(len(t))
metacog_signal = np.sin(2*np.pi*metacog_freq*t) + 0.5*np.random.randn(len(t))
gnw_signal = np.sin(2*np.pi*gnw_freq*t) + 0.5*np.random.randn(len(t))

# ترکیب سیگنال‌ها در 5 کانال EEG
channels = np.vstack([dmn_signal, acn_signal, tpj_signal, metacog_signal, gnw_signal])

# رسم QEEG (قدرت طیفی)
plt.figure(figsize=(14, 8))
for i, sig in enumerate(channels):
    f, Pxx = welch(sig, fs, nperseg=1024)
    plt.semilogy(f, Pxx, label=f'Channel {i+1}')

plt.xlim(0, 60)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (uV^2/Hz)')
plt.title('Simulated QEEG Power Spectrum of 5 Integrated Networks')
plt.legend()
plt.grid(True)
plt.show()
