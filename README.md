# Gradient Health Tools
[![Python Package](https://github.com/gradienthealth/gh-tools/actions/workflows/main.yml/badge.svg)](https://github.com/gradienthealth/gh-tools/actions/workflows/main.yml)

Tools to be used primarily in colab training environment and using wasabi storage for logging/data.
Also includes tools for common dicom preprocessing steps.

# Callbacks

### gh_tools.callbacks.keras_storage
Used for syncing models/logs into s3 file system

### gh_tools.callbacks.log_code
Used to snapshot notebooks into s3 file system

# Utils

### gh_tools.utils.image
Used for various 2D image transforms see (projection matrix)[https://en.wikipedia.org/wiki/Transformation_matrix]

### gh_tools.utils.colab
Used for checking and saving notebooks in colab programmatically

### gh_tools.utils.version_service
Bundles callbacks related to versioning and logging.

# Contributing and Issues
Contributions are welcome in the form of pull requests.
For issues, please provide a minimal example of code which results in an error.
 
## Software License
Copyright 2020 Ouwen Huang, Gradient Health Inc.
