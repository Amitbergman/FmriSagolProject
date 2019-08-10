## Setup:

1. Create a conda env + jupyter kernel
```
conda create -n sagol python=3.7
conda activate sagol
```

2. Install dependencies:

conda install -y pandas scipy numpy scikit-learn nilearn seaborn matplotlib 

3. Create jupyter kernel (recommended):
```
conda install -y ipykernel jupyter_client
python -m ipykernel install --user --name sagol
```
