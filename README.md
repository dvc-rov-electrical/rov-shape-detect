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
2. `src/test_pictures.py` - Test code for running the shape detection against the two example pictures in the `test/` folder. You can specify an image file for recognizing, or omit it to default to loading `sample_complex.png`. Press 'q' at any time to stop the script.
3. `src/test_webcam.py` - Test code for running the shape detection against your webcam. You can specify a video port number or omit it to load the first one it can find. Press 'q' at any time to freeze the camera frame and again to stop the script.
4. `src/test_screencap.py` - Test code for running the shape detection against your screen. It'll only look at the left half of your screen to avoid the recursive screen effect. Press 'q' at any time to freeze the screen frame and again to stop the script.

**NOTE**: DO NOT click the 'X' on the window, since it won't terminate the python script and you won't be able to Ctrl+C the script. Press 'q' to gracefully end the script (twice if testing with the webcam or screen).