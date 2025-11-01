import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, image

# شبیه‌سازی داده fMRI 4D: (64, 64, 30, 180) ابعاد (x, y, z, time)
data = np.random.rand(64, 64, 30, 180)

# اعمال activation pattern بر اساس 5 شبکه توصیف شده
def activate_network(data, coords, strength=5.0):
    for coord in coords:
        x, y, z = coord
        data[x-2:x+2, y-2:y+2, z-1:z+1, :] += strength
    return data

# مختصات تقریبی شبکه‌ها (در فضای فرضی)
dmn_coords = [(32, 32, 15)]
tpj_coords = [(40, 30, 12)]
gnw_coords = [(30, 28, 16)]
acc_coords = [(34, 34, 18)]
meta_coords = [(28, 36, 14)]

# فعالسازی همزمان
for coords in [dmn_coords, tpj_coords, gnw_coords, acc_coords, meta_coords]:
    data = activate_network(data, coords)

# ساخت فایل NIfTI
img = nib.Nifti1Image(data, affine=np.eye(4))
nib.save(img, 'simulated_activation_five_networks.nii.gz')

# ویژوالایز یک اسلایس
display = plotting.plot_epi(img.slicer[:, :, 15, 90], title='Activation Map - Timepoint 90')
plt.show()

# ویژوالایز activation map به شکل interactive
plotting.view_img(img, threshold=3.0, title='Simulated fMRI Activation - 5 Networks')
