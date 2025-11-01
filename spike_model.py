import numpy as np
import matplotlib.pyplot as plt

# تنظیم پارامترهای شبیه‌سازی
T = 1000  # مدت زمان شبیه‌سازی (ms)
dt = 0.1  # گام زمانی (ms)
time = np.arange(0, T, dt)

# تعداد نورون‌ها (۵ شبکه با ۱۰ نورون در هرکدام)
neurons_per_network = 10
networks = ['DMN', 'ACC', 'TPJ', 'MetaCognition', 'GNW']
total_neurons = neurons_per_network * len(networks)

# نرخ شلیک پایه برای هر شبکه (Hz)
rates = {
    'DMN': 8,
    'ACC': 12,
    'TPJ': 10,
    'MetaCognition': 15,
    'GNW': 20
}

# تولید اسپایک‌تریندوم برای هر نورون
def generate_spike_train(rate, T, dt):
    p_spike = rate * dt / 1000  # احتمال شلیک در هر گام زمانی
    return np.random.rand(len(time)) < p_spike

# تولید اسپایک‌های شبکه‌ها
spike_trains = {}
for i, net in enumerate(networks):
    for n in range(neurons_per_network):
        key = f'{net}_N{n+1}'
        spike_trains[key] = generate_spike_train(rates[net], T, dt)

# ترسیم اسپایک رستر
plt.figure(figsize=(14, 8))
for idx, (key, spikes) in enumerate(spike_trains.items()):
    spike_times = time[spikes]
    plt.vlines(spike_times, idx + 0.5, idx + 1.5)

plt.xlabel('زمان (ms)')
plt.ylabel('نورون‌ها')
plt.title('مدل اسپایک‌زنی شبکه ۵تایی دائمی فعال')
plt.yticks(np.arange(1, total_neurons+1), spike_trains.keys(), fontsize=7)
plt.ylim(0.5, total_neurons + 0.5)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
