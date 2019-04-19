# Overview

This repo contains the files for the benthic species identification task for the ROV MATE Competition of 2019.

# Setup

```bash
# Clone the repo and change directories into the folder
git clone https://github.com/dvc-rov/shape-detection.git
cd shape-detection

# Create the environment from the .yml file and activate it
conda env create -f environment.yml
conda activate dvc-rov
```

# Usage
1. `src/detect_shapes.py` - Contains the entire logic for detecting shapes (i.e. benthic species)
2. `src/test_pictures.py` - Test code for running the shape detection against the two example pictures in the `test/` folder. Press 'q' at any time to stop the script.
3. `src/test_webcam.py` - Test code for running the shape detection against your webcam. You'll need to change the video source to make sure you're getting the right one. Press 'q' at any time to freeze the camera frame and again to stop the script.

**NOTE**: DO NOT click the 'X' on the window, since it won't terminate the python script and you won't be able to Ctrl+C the script. Press 'q' to gracefully end the script (twice if testing with the webcam).