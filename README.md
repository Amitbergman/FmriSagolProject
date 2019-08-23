## Setup:

* Create a conda env + jupyter kernel
```
conda create -n sagol python=3.7
conda activate sagol
```

* Install dependencies:

```
python setup.py devlelop
pip install -r requirements.txt
```

* Create jupyter kernel (recommended):
```
conda install -y ipykernel jupyter_client
python -m ipykernel install --user --name sagol
```
