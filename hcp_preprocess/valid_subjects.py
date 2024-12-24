import boto3

# AWS Credentials
aws_access_key = ""
aws_secret_key = ""

# HCP S3 bucket
bucket_name = "hcp-openaccess"
prefix = "HCP_1200/"

# Required tasks
required_tasks = ["tfMRI_WM", "tfMRI_GAMBLING", "tfMRI_LANGUAGE", 
                  "tfMRI_MOTOR", "tfMRI_RELATIONAL", "tfMRI_SOCIAL", "tfMRI_EMOTION"]

# Initialize S3 Client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

# Fetch subject IDs
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
subject_ids = [prefix['Prefix'].split('/')[-2] for prefix in response.get('CommonPrefixes', [])]

valid_subjects = []

# Check for task directories
for subject_id in subject_ids:
    task_paths = [f"{prefix}{subject_id}/MNINonLinear/Results/{task}/" for task in required_tasks]
    all_tasks_present = True

    for task_path in task_paths:
        task_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=task_path)
        if 'Contents' not in task_response:
            all_tasks_present = False
            break

    if all_tasks_present:
        print(f"Subject {subject_id} has all tasks.")
        valid_subjects.append(subject_id)
    else:
        print(f"Subject {subject_id} is missing some tasks.")

# Save valid subjects to a file
with open("valid_subjects.txt", "w") as f:
    for subject in valid_subjects:
        f.write(subject + "\n")

print("Valid subjects saved to valid_subjects.txt.")

