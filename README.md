# GRA4157-Electricity-markets-project


## Env setup:

### Create a new conda environment with Python
conda create -n gra4157 python=3.11 -y

### Activate the environment
conda activate gra4157

### Install core data science packages
conda install pandas numpy matplotlib requests -y

### Install geopandas (easier via conda due to GDAL dependencies)
conda install geopandas -y

### Install scikit-learn
conda install scikit-learn -y

### Install holidays and other packages via pip
pip install holidays pyarrow

### Command	Description
conda activate gra4157 -- Activate your environment
conda deactivate -- Deactivate current environment
conda env list	-- List all environments
conda list	-- List installed packages
