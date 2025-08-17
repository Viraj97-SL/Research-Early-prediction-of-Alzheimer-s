import ants
import os
import nibabel as nib
import numpy as np
import pandas as pd
from google.colab import drive

# --- Configuration ---
drive.mount('/content/drive', force_remount=True)
drive_project_path = '/content/drive/My Drive/ADNI_NewDS/'
base_image_directory = os.path.join(drive_project_path, 'unzipped_data/ADNI')
results_directory = os.path.join(drive_project_path, 'results')
cleaned_data_path = os.path.join(results_directory, 'project_data_cleaned.csv')

# --- MODIFICATION: New output directory for Swin Transformer data ---
processed_mri_directory = os.path.join(results_directory, 'processed_mri_scans_swin')
if not os.path.exists(processed_mri_directory):
    os.makedirs(processed_mri_directory)
    print(f"Created directory: {processed_mri_directory}")

# --- Load Standard Template ---
print("Loading MNI152 template...")
template = ants.image_read(ants.get_ants_data('mni'))

# --- UPGRADED Preprocessing Function ---
def preprocess_mri_for_swin(patient_id, input_dir, output_dir, template_image):
    """
    Finds the best MRI, registers it, RESIZES it for the Swin Transformer,
    normalizes intensity, and saves the output.
    """
    try:
        patient_folder = os.path.join(input_dir, patient_id)
        if not os.path.isdir(patient_folder):
            print(f"Skipping {patient_id} as it's not a directory.")
            return

        # (Robust File Search Logic remains the same)
        nii_files_found = []
        for root, dirs, files in os.walk(patient_folder):
            for file in files:
                if file.endswith('.nii'):
                    nii_files_found.append(os.path.join(root, file))

        if not nii_files_found:
            print(f"Warning: No .nii files found for patient {patient_id}. Skipping.")
            return

        best_file = None
        highest_score = -1
        priority_keywords = ["GradWarp", "B1_Correction", "N3", "Scaled"]
        for file_path in nii_files_found:
            score = sum(keyword in file_path for keyword in priority_keywords)
            if score > highest_score:
                highest_score = score
                best_file = file_path
        
        if best_file is None:
            best_file = nii_files_found[0]
        
        nii_file_path = best_file
        print(f"Processing patient: {patient_id} (using file: ...{nii_file_path[-80:]})")
        
        moving_image = ants.image_read(nii_file_path)

        registration_output = ants.registration(
            fixed=template_image,
            moving=moving_image,
            type_of_transform='SyN'
        )
        warped_image = registration_output['warpedmovout']

        # --- MODIFICATION: Add Resampling Step for Swin Transformer ---
        # This resizes the 3D image to the exact dimensions the model requires.
        target_size = (96, 128, 96)
        resampled_image = ants.resample_image(warped_image, target_size, use_voxels=True, interp_type=4)
        # ----------------------------------------------------------------

        # Intensity Scaling
        warped_image_np = resampled_image.numpy()
        warped_image_np = (warped_image_np - np.min(warped_image_np)) / (np.max(warped_image_np) - np.min(warped_image_np))
        
        # Save the processed 3D image
        output_path = os.path.join(output_dir, f"{patient_id}_processed.npy")
        np.save(output_path, warped_image_np)
        print(f"--> Saved RESIZED scan to: {output_path}")

    except Exception as e:
        print(f"ERROR processing patient {patient_id}: {e}")

# --- Execution Loop ---
print(f"Loading patient list from: {cleaned_data_path}")
df_cleaned = pd.read_csv(cleaned_data_path)
patient_ids_from_csv = df_cleaned['PTID'].unique()

print(f"\nStarting Swin Transformer preprocessing for {len(patient_ids_from_csv)} patients...")
for patient_id in patient_ids_from_csv:
    output_file = os.path.join(processed_mri_directory, f"{patient_id}_processed.npy")
    if os.path.exists(output_file):
        print(f"Patient {patient_id} already processed for Swin. Skipping.")
        continue
    preprocess_mri_for_swin(patient_id, base_image_directory, processed_mri_directory, template)

print("\nSwin Transformer MRI Preprocessing Complete.")
