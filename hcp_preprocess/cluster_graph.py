import os
import glob
import numpy as np
import nibabel as nib
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans

def load_cifti_data(cifti_path):
    cifti_img = nib.load(cifti_path)
    cifti_data = cifti_img.get_fdata(dtype=np.float32)  # [time, n_vertices]
    hdr = cifti_img.header

    brain_models = hdr.get_index_map(1).brain_models
    left_hemi = None
    right_hemi = None
    for bm in brain_models:
        if bm.brain_structure.strip() == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_hemi = (bm.index_offset, bm.index_offset + bm.index_count)
        elif bm.brain_structure.strip() == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_hemi = (bm.index_offset, bm.index_offset + bm.index_count)

    if left_hemi is None or right_hemi is None:
        raise ValueError(f"Could not find both hemispheres in {cifti_path}")

    # Extract LH, RH data
    left_data = cifti_data[:, left_hemi[0]:left_hemi[1]]
    right_data = cifti_data[:, right_hemi[0]:right_hemi[1]]
    combined_data = np.hstack((left_data, right_data))  # shape [time, LH+RH_vertices]
    return combined_data

def build_region_time_series(surface_data, labels):
    n_clusters = labels.max() + 1
    timepoints = surface_data.shape[0]
    region_ts = np.zeros((timepoints, n_clusters), dtype=np.float32)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            region_ts[:, c] = surface_data[:, idx].mean(axis=1)
    return region_ts

def compute_top_percent_graph(region_ts, top_percent=10):
    corr_mat = np.corrcoef(region_ts.T)
    np.fill_diagonal(corr_mat, 0)

    n = corr_mat.shape[0]
    upper_vals = []
    for i in range(n):
        for j in range(i+1, n):
            val = corr_mat[i, j]
            if val > 0:
                upper_vals.append(val)

    if len(upper_vals) == 0:
        print("Warning: no positive correlations found.")
        threshold = 0
    else:
        threshold = np.percentile(upper_vals, 100 - top_percent)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i+1, n):
            val = corr_mat[i, j]
            if val >= threshold:
                G.add_edge(i, j, weight=val)
    return corr_mat, G

def networkx_to_pyg_data(G, node_features=None):
    edges = []
    weights = []
    for (u, v, d) in G.edges(data=True):
        edges.append([u, v])
        edges.append([v, u])  # undirected => both directions
        w = d.get('weight', 1.0)
        weights.append(w)
        weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape [2, E]
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(-1)  # [E, 1]

    n_nodes = G.number_of_nodes()
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float)
    else:
        # fallback: each node has a 1D feature = node_index
        x = torch.arange(n_nodes, dtype=torch.float).unsqueeze(-1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Group-level K-means for HCP dtseries => PyG graphs.")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Folder containing downloaded .dtseries.nii files.")
    parser.add_argument("--out_dir", type=str, default="pyg_graphs",
                        help="Folder to store output PyG graph files.")
    parser.add_argument("--n_clusters", type=int, default=100,
                        help="Number of K-means clusters (nodes) for parcellation.")
    parser.add_argument("--top_percent", type=float, default=10.0,
                        help="Keep top X% positive correlations as edges.")
    parser.add_argument("--do_centroid_loc", action="store_true",
                        help="(Optional) If you have vertex coords, store cluster centroid coords.")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    n_clusters = args.n_clusters
    top_percent = args.top_percent
    os.makedirs(out_dir, exist_ok=True)

    # Gather all dtseries in data_dir
    dtseries_paths = sorted(glob.glob(os.path.join(data_dir, "*_tfMRI_*_Atlas.dtseries.nii")))
    if len(dtseries_paths) == 0:
        print(f"No dtseries files found in {data_dir} matching *_tfMRI_*_Atlas.dtseries.nii")
        exit(0)

    print(f"Found {len(dtseries_paths)} dtseries files. Building group-level K-means with n_clusters={n_clusters}...")

    all_means = []
    for cifti_path in dtseries_paths:
        print(f"Loading {os.path.basename(cifti_path)} for group-level K-means...")
        surface_data = load_cifti_data(cifti_path)  # [time, vertices]
        vertex_mean = np.mean(surface_data, axis=0) # shape [vertices]
        all_means.append(vertex_mean)

    all_means = np.vstack(all_means)  # shape [num_dtseries, n_vertices]
    # reshape => [num_dtseries * n_vertices, 1]
    num_dtseries, n_vertices = all_means.shape
    all_means = all_means.reshape(-1, 1)

    print(f"Concatenated shape for K-means: {all_means.shape}, now fitting...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_means)  # learns cluster_centers_

    print("Group-level K-means done! Centroid shape:", kmeans.cluster_centers_.shape)

    centroid_save_path = os.path.join(out_dir, f"group_kmeans_{n_clusters}_centroids.npy")
    np.save(centroid_save_path, kmeans.cluster_centers_)
    print(f"Saved cluster centroids to {centroid_save_path}")

    for cifti_path in dtseries_paths:
        fname = os.path.basename(cifti_path)
        print(f"\nProcessing subject/task file: {fname}")

        surface_data = load_cifti_data(cifti_path)
        vertex_mean = np.mean(surface_data, axis=0).reshape(-1,1)
        
        labels = kmeans.predict(vertex_mean.ravel().reshape(-1,1))  # shape [n_vertices]

        label_save_path = os.path.join(out_dir, fname.replace(".dtseries.nii", "_labels.npy"))
        np.save(label_save_path, labels)
        print(f"  Saved cluster labels to {label_save_path}")

        region_ts = build_region_time_series(surface_data, labels)
        print(f"  region_ts shape: {region_ts.shape}")

        # Build correlation graph
        corr_mat, G = compute_top_percent_graph(region_ts, top_percent=top_percent)
        print(f"  Graph => {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Convert Nx => PyG
        data = networkx_to_pyg_data(G, node_features=None)

        # Save .pt
        out_name = fname.replace(".dtseries.nii", f"_kmeans{n_clusters}.pt")
        out_path = os.path.join(out_dir, out_name)
        torch.save(data, out_path)
        print(f"  Saved PyG data => {out_path}")

    print("\nAll dtseries processed. PyG graphs + label arrays saved in", out_dir)
