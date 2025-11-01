# DTI_Processing_Pipeline.py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def run_dti_processing(dti_file):
    img = nib.load(dti_file)
    data = img.get_fdata()
    fa_map = np.mean(data, axis=3)
    plt.imshow(fa_map[:, :, data.shape[2]//2], cmap='gray')
    plt.title('Fractional Anisotropy Map')
    plt.show()


# fMRI_Activation_Mapping.py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def run_fmri_activation(fmri_file):
    img = nib.load(fmri_file)
    data = img.get_fdata()
    mean_img = np.mean(data, axis=3)
    plt.imshow(mean_img[:, :, data.shape[2]//2], cmap='hot')
    plt.title('Mean fMRI Activation Map')
    plt.show()


# QEEG_Spectral_Analysis.py
import mne

def run_qeeg_analysis(eeg_file):
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    raw.plot_psd(fmax=60)


# Spike_Detection_Tool.py
import mne

def detect_spikes(eeg_file):
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    raw.filter(0.5, 40)
    events = mne.find_events(raw, stim_channel='STI 014')
    print(f"Detected {len(events)} spikes/events")


# BCI_CSP_SVM_Classifier.py
import mne
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_bci_motor_imagery(eeg_file):
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    raw.filter(8., 30.)
    events = mne.find_events(raw, stim_channel='STI 014')
    event_id = dict(left=1, right=2)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=3, picks='eeg', preload=True)
    labels = epochs.events[:, -1]
    csp = CSP(n_components=4, log=True)
    clf = Pipeline([('CSP', csp), ('Scaler', StandardScaler()), ('SVM', SVC(kernel='linear'))])
    X = epochs.get_data()
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"BCI Classification Accuracy: {acc*100:.2f}%")


# Quantum_Personalized_Medicine_Sim.py
import numpy as np

def quantum_personalized_medicine(genome_features):
    quantum_scores = np.sin(genome_features) + np.random.rand(len(genome_features)) * 0.1
    risk_score = np.mean(quantum_scores)
    print(f"Quantum Health Risk Score: {risk_score:.2f}")


# CRISPR_GeneEdit_AI_Sim.py
def crispr_edit_simulation(target_genes):
    edits = {gene: f"{gene}_knockout" for gene in target_genes}
    print(f"Simulated CRISPR Edits: {edits}")


# PharmacoNeuroinformatics_AI.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def drug_response_prediction(drug_features):
    rf = RandomForestClassifier()
    X = np.random.rand(100, len(drug_features))
    y = np.random.randint(0, 2, 100)
    rf.fit(X, y)
    pred = rf.predict([drug_features])
    print(f"Predicted Neuro Response: {'Positive' if pred[0] == 1 else 'Negative'}")


# Neurosina_Multitool_FullPipeline.py
import numpy as np
import mne
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

def demo_neurosina_suite():
    quantum_personalized_medicine(np.array([0.8, 0.5, 0.3, 0.9]))
    crispr_edit_simulation(['APP', 'HTT', 'SNCA'])
    drug_response_prediction([0.7, 0.3, 0.6, 0.9, 0.5])

if __name__ == "__main__":
    demo_neurosina_suite()
