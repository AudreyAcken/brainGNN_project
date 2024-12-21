import boto3
import os

# AWS Credentials (as in your original script)
aws_access_key = ""
aws_secret_key = ""

# HCP S3 bucket and tasks
bucket_name = "hcp-openaccess"
tasks = ["WM", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "EMOTION"]
prefix_template = "HCP_1200/{subject_id}/MNINonLinear/Results/tfMRI_{task}_LR/tfMRI_{task}_LR_Atlas.dtseries.nii"

# Initialize S3 Client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

def download_timeseries(subject_id, output_dir):
    """
    Download timeseries data for all tasks for a single subject.

    :param subject_id: Subject ID (e.g., '100307')
    :param output_dir: Directory to save the downloaded files
    """
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
    """
    Download data for each subject listed in 'subject_list_file'.
    If max_subjects is set, stop after downloading that many subjects.

    :param subject_list_file: Path to file containing one subject ID per line
    :param output_dir: Directory to store downloaded files
    :param max_subjects: Optional limit on how many subjects to download
    """
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
    # Example usage:
    subject_list_file = "valid_subjects.txt"   # the file you mentioned
    output_dir = "data"                        # directory to store all downloaded files
    max_subjects = 50  # set an integer limit, or None to download all

    download_for_subject_list(subject_list_file, output_dir, max_subjects)
