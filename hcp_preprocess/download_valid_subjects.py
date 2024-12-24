import boto3
import os

aws_access_key = ""
aws_secret_key = ""

bucket_name = "hcp-openaccess"
tasks = ["WM", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "EMOTION"]
prefix_template = "HCP_1200/{subject_id}/MNINonLinear/Results/tfMRI_{task}_LR/tfMRI_{task}_LR_Atlas.dtseries.nii"

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

def download_timeseries(subject_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for task in tasks:
        prefix = prefix_template.format(subject_id=subject_id, task=task)
        original_file_name = prefix.split("/")[-1]  # e.g., 'tfMRI_WM_LR_Atlas.dtseries.nii'
        
        # Attach subject ID to the filename to avoid overwriting
        subject_specific_file_name = f"{subject_id}_{original_file_name}"
        
        local_path = os.path.join(output_dir, subject_specific_file_name)

        try:
            print(f"Downloading {subject_specific_file_name} for subject {subject_id}, task {task}...")
            s3.download_file(bucket_name, prefix, local_path)
            print(f"  Downloaded: {local_path}")
        except Exception as e:
            print(f"  Failed to download {subject_specific_file_name} for subject {subject_id}, task {task}: {e}")

def download_for_subject_list(subject_list_file, output_dir, max_subjects=None):
    # Read subject IDs from file, stripping empty lines
    with open(subject_list_file, "r") as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    print(f"Found {len(subject_ids)} subjects in {subject_list_file}.")

    count = 0
    for subject_id in subject_ids:
        if max_subjects is not None and count >= max_subjects:
            print(f"Reached the maximum limit of {max_subjects} subjects. Stopping.")
            break
        
        print(f"\n=== Downloading data for subject {subject_id} (#{count+1}) ===")
        download_timeseries(subject_id, output_dir)
        count += 1
    
    print(f"\nFinished downloading data for {count} subjects.")

if __name__ == "__main__":
    subject_list_file = "valid_subjects.txt"   
    output_dir = "data"                       
    max_subjects = 100

    download_for_subject_list(subject_list_file, output_dir, max_subjects)
