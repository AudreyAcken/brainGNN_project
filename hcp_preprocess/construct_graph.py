import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img, new_img_like
import networkx as nx
import os


def load_cifti_data(cifti_path):
    """Load a CIFTI `.dtseries.nii` file."""
    cifti_img = nib.load(cifti_path)
    data = cifti_img.get_fdata(dtype=np.float32)
    return data, cifti_img


def convert_to_nifti_like(cifti_data, cifti_img, reference_img):
    """Convert CIFTI data to a NIfTI-like volume for atlas compatibility."""
    # Create a new NIfTI image based on the reference volume
    nifti_like_img = new_img_like(reference_img, cifti_data)
    return nifti_like_img


def extract_regional_time_series_volumetric(data_img, atlas_img, mask_img=None):
    """Extract regional time series using a volumetric atlas."""
    masker = NiftiLabelsMasker(labels_img=atlas_img, mask_img=mask_img, standardize=True)
    regional_time_series = masker.fit_transform(data_img)
    return regional_time_series


def construct_connectivity_graph(regional_time_series, method='correlation'):
    """Construct a graph from regional time series using a specified method."""
    if method == 'correlation':
        # Compute the Pearson correlation matrix
        connectivity_matrix = np.corrcoef(regional_time_series.T)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Build a graph using NetworkX
    graph = nx.Graph()
    n_regions = connectivity_matrix.shape[0]

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            weight = connectivity_matrix[i, j]
            if not np.isnan(weight):
                graph.add_edge(i, j, weight=weight)

    return graph


# Paths and Configuration
cifti_path = "hcp_timeseries/tfMRI_EMOTION_LR_Atlas.dtseries.nii"
atlas_path = "AAL.nii"  # E.g., AAL or Harvard-Oxford atlas
reference_path = "path/to/reference_volume.nii.gz"  # Reference NIfTI volume
mask_path = "path/to/brain_mask.nii.gz"  # Optional: Use a brain mask

# Load Data
print("Loading CIFTI data...")
cifti_data, cifti_img = load_cifti_data(cifti_path)

print("Loading atlas and reference volume...")
atlas_img = nib.load(atlas_path)
reference_img = nib.load(reference_path)

# Convert to NIfTI-like format
print("Converting CIFTI to NIfTI-like format...")
data_img = convert_to_nifti_like(cifti_data, cifti_img, reference_img)

# Resample atlas to match the data (if needed)
print("Resampling atlas to match data...")
atlas_img_resampled = resample_to_img(atlas_img, reference_img, interpolation='nearest')

# Extract Regional Time Series
print("Extracting regional time series...")
regional_time_series = extract_regional_time_series_volumetric(data_img, atlas_img_resampled, mask_img=None)

# Construct Graph
print("Constructing graph...")
graph = construct_connectivity_graph(regional_time_series, method='correlation')

# Save Graph
output_path = "output_graph.gml"
print(f"Saving graph to {output_path}...")
nx.write_gml(graph, output_path)

print("Graph construction complete!")
