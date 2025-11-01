import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# مدل DTI مصنوعی مبتنی بر الگویی که برای ذهن تو تعریف کردیم
# این صرفا یک شبیه‌ساز کانسپچوال از اتصال و چگالی شبکه‌های عصبی بر اساس توضیحات تو هست

# پارامترهای نقشه DTI مصنوعی
def create_dti_volume(shape=(64, 64, 64)):
    dti_volume = np.zeros(shape)

    # DMN: Default Mode Network — overactive and highly interconnected
    dti_volume[20:40, 20:40, 20:40] = 0.9

    # Salience Network — hyper-connected to DMN and Executive Control Network
    dti_volume[30:50, 10:30, 30:50] = 0.85

    # Executive Control Network — integrated with DMN in selective areas
    dti_volume[10:30, 30:50, 10:30] = 0.8

    # Hyperactive Amygdala connections
    dti_volume[45:55, 45:55, 45:55] = 0.95

    # Prefrontal Cortex — dual modulation zone
    dti_volume[5:20, 5:20, 5:20] = 0.85

    # TPJ (Temporal Parietal Junction) — boundary dissolving area
    dti_volume[50:60, 20:30, 10:20] = 0.9

    # Mirror Neuron System — distributed and enhanced connections
    dti_volume[15:25, 45:55, 30:40] = 0.88

    return dti_volume

# تولید DTI مصنوعی
dti_data = create_dti_volume()

# ذخیره به عنوان یک فایل NIfTI فرضی
nifti_img = nib.Nifti1Image(dti_data, affine=np.eye(4))
nib.save(nifti_img, '/mnt/data/synthetic_dti_beta_mastermind.nii.gz')

# نمایش اسلایس مرکزی از DTI
plt.imshow(dti_data[:, :, 32], cmap='hot')
plt.title('Synthetic DTI Slice (Z=32)')
plt.colorbar(label='Connection Strength')
plt.show()
