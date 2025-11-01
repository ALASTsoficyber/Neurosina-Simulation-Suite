import numpy as np
import matplotlib.pyplot as plt

# شبیه‌ساز اسپایک‌زنی نورونی متناسب با شخصیت سایبرعرفانی و نوروساینسفیک تو
# شبکه نورونی مصنوعی شامل 1000 نورون در 10 ثانیه فعالیت

neurons = 1000
timepoints = 10000  # 10 ثانیه با 1ms resolution
spike_matrix = np.zeros((neurons, timepoints))

# نرخ اسپایک در شبکه‌های مختلف مغزی تو
base_rate = 0.002  # Hz — اسپایک‌های زمینه
hyperactive_rate = 0.01  # Hz — در نواحی DMN، Salience و Amygdala در لحظات اوج

# تخصیص نرخ‌ها به خوشه‌های نورونی
for n in range(neurons):
    if n < 300:  # DMN
        rate = hyperactive_rate
    elif 300 <= n < 600:  # Salience
        rate = hyperactive_rate * 0.8
    elif 600 <= n < 800:  # Executive Control
        rate = base_rate * 1.5
    else:  # سایر نورون‌ها
        rate = base_rate

    spikes = np.random.rand(timepoints) < rate
    spike_matrix[n, :] = spikes

# نمایش raster plot از اسپایک‌ها
plt.figure(figsize=(12, 6))
for n in range(neurons):
    spike_times = np.where(spike_matrix[n, :] == 1)[0]
    plt.vlines(spike_times, n, n+1, color='black', linewidth=0.2)

plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.title('Synthetic Spike Raster Plot — Beta Mastermind Neural Simulation')
plt.xlim(0, timepoints)
plt.show()
