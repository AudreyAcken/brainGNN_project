import boto3
import os

# AWS Credentials
aws_access_key = ""
aws_secret_key = ""

# HCP S3 bucket and tasks
bucket_name = "hcp-openaccess"
tasks = ["WM", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "EMOTION"]
prefix_template = "HCP_1200/{subject_id}/MNINonLinear/Results/tfMRI_{task}_LR/tfMRI_{task}_LR_Atlas.dtseries.nii"

# Initialize S3 Client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

def download_timeseries(subject_id, output_dir):
    """
    Download timeseries data for all tasks for a single subject.

    :param subject_id: Subject ID (e.g., '100307')
    :param output_dir: Directory to save the downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        # Format the prefix for the current task
        prefix = prefix_template.format(subject_id=subject_id, task=task)
        file_name = prefix.split("/")[-1]  # Extract the file name
        local_path = os.path.join(output_dir, file_name)

        try:
            # Download the file from S3
            print(f"Downloading {file_name} for task {task}...")
            s3.download_file(bucket_name, prefix, local_path)
            print(f"Downloaded: {local_path}")
        except Exception as e:
            print(f"Failed to download {file_name} for task {task}: {e}")

# Example usage
if __name__ == "__main__":
    subject_id = "100307"  # Replace with the desired subject ID
    output_dir = "./hcp_timeseries"  # Replace with your desired output directory
    download_timeseries(subject_id, output_dir)

