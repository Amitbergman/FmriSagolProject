## Setup:

* Create a conda env + pytorch
```
conda create -n sagol python=3.7
conda activate sagol

MacOS:
conda install pytorch torchvision -c pytorch
Windows:
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

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

## Run:
In CMD:
```
conda activate sagol
python sagol/gui/app.py
```
