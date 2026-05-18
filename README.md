# Open3dRenderDemo
An example of processing the Eagle demo data from Open3d and rendering it

## Setting up the environment

Create and activate a Python virtual environment, then install Open3D:

(Note: This uses venv and pip, but feel free to use any env and package manager that you prefer.)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install --install-hooks
```


## Content of the demo

The demo will load the example EaglePointCloud dataset from Open3d.

It will then perform the following steps:

- Apply a down-sampling filter to reduce the number of points while preserving
geometric structure.
- Estimate surface normals for the cropped point cloud.
- Perform Euclidean clustering to separate the scene into individual components.
- Highlight and save each cluster separately or color them for visualization.
- Save renders of intermediate results (downsampled cloud, normals, clusters) as
image files (linked to this Readme).

## Running the demo
