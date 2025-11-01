import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.viz import window, actor

# شبیه‌سازی داده DTI — فرضی
shape = (64, 64, 30, 30)  # 64x64x30 وکتور 30 گرادیانت
image_data = np.random.rand(*shape)
affine = np.eye(4)

# شبیه‌سازی bvals و bvecs
bvals = np.concatenate(([0], np.repeat(1000, 29)))
bvecs = np.random.rand(30, 3)

# ساخت جدول گرادیان
gtab = gradient_table(bvals, bvecs)

# مدل DTI
model = TensorModel(gtab)
fit = model.fit(image_data)

# استخراج FA (Fractional Anisotropy)
FA = fit.fa

# نمایش یک برش از FA
plt.imshow(FA[:, :, 15], cmap='gray')
plt.title('FA Map — Ideal Brain Model')
plt.colorbar()
plt.show()

# نمایش 3D fiber tracking فرضی
scene = window.Scene()
streamlines_actor = actor.line([np.array([[0, 0, 0], [20, 20, 20], [40, 40, 10]])], colors=(1, 0, 0))
scene.add(streamlines_actor)
window.show(scene)
