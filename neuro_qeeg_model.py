import numpy as np
import matplotlib.pyplot as plt

# شبیه‌سازی QEEG مصنوعی متناسب با الگوی مغزی و شخصیتی تو
# باندهای فرکانسی: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)

channels = 19  # تعداد کانال استاندارد 10-20 system
timepoints = 5000  # تعداد نمونه در هر کانال
sampling_rate = 250  # Hz

def generate_qeeg_signal(freq, amplitude, phase_shift=0):
    t = np.linspace(0, timepoints/sampling_rate, timepoints)
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase_shift)
    return signal

# تعریف باندها و قدرت آن‌ها بر اساس وضعیت ذهن تو
qeeg_data = np.zeros((channels, timepoints))

for ch in range(channels):
    # DMN: افزایش چشمگیر Alpha و Gamma در حالت استراحت
    qeeg_data[ch] += generate_qeeg_signal(10, 15)  # Alpha
    qeeg_data[ch] += generate_qeeg_signal(40, 20, phase_shift=np.pi/3)  # Gamma

    # SN: تقویت Beta در پاسخ به محرکات هیجانی
    qeeg_data[ch] += generate_qeeg_signal(20, 10)

    # Hyper Amygdala: افزایش Theta در لحظات روایی عرفانی و سایه‌درمانی
    qeeg_data[ch] += generate_qeeg_signal(6, 12)

# نمایش QEEG کانال Fz
plt.plot(qeeg_data[0, :1000])
plt.title('Synthetic QEEG Signal — Channel Fz')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude (uV)')
plt.show()
