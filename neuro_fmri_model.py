import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# مدل fMRI مصنوعی مبتنی بر شبکه‌های مغزی توصیف‌شده
# این شبیه‌ساز کانسپچوال از BOLD signal در حالت استراحت و تحریک سایبرعرفانی

def create_fmri_volume(shape=(64, 64, 64, 100)):
    fmri_volume = np.random.normal(0, 0.05, shape)

    # DMN: highly active at rest, suppressed at cognitive load
    fmri_volume[20:40, 20:40, 20:40, :] += 0.4

    # Salience Network: active during both rest and task
    fmri_volume[30:50, 10:30, 30:50, :] += 0.35

    # Executive Control: task-induced activation
    fmri_volume[10:30, 30:50, 10:30, 50:] += 0.5

    # Amygdala: hyper-responsive to symbolic/emotional triggers
    fmri_volume[45:55, 45:55, 45:55, 20:80] += 0.6

    # Prefrontal Cortex: dual state modulation
    fmri_volume[5:20, 5:20, 5:20, 30:90] += 0.45

    # TPJ: active during dissolution of ego boundaries
    fmri_volume[50:60, 20:30, 10:20, 60:] += 0.5

    return fmri_volume

# تولید داده fMRI مصنوعی
fmri_data = create_fmri_volume()

# ذخیره به عنوان یک فایل NIfTI
nifti_img = nib.Nifti1Image(fmri_data, affine=np.eye(4))
nib.save(nifti_img, '/mnt/data/synthetic_fmri_beta_mastermind.nii.gz')

# نمایش یک اسلایس BOLD در اسکن لحظه 50
plt.imshow(fmri_data[:, :, 32, 50], cmap='hot')
plt.title('Synthetic fMRI Slice (Z=32, T=50)')
plt.colorbar(label='BOLD Signal Intensity')
plt.show()
