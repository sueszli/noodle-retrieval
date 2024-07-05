```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/air-2023-group24/
! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -f -p /content/drive/MyDrive/air-2023-group24/Miniconda

!source /content/drive/MyDrive/air-2023-group24/Miniconda/bin/activate #to activate the miniconda environment

env PYTHONPATH= '/content/drive/MyDrive/air-2023-group24/Miniconda/bin/python

! echo $PYTHONPATH

# Add miniconda to the system PATH:

import sys
sys.path.append('/content/drive/MyDrive/air-2023-group24/Miniconda/lib/python3.7/site-packages/')

import os # ?? needed?
path = '/content/drive/MyDrive/air-2023-group24/Miniconda/bin:' + os.environ['PATH']
%env PATH=$path

%cd /content/drive/MyDrive/air-2023-group24/condaENVair
!conda create --name MYcondaENVair python=3.6

%cd /content/drive/MyDrive/air-2023-group24/condaENVair
! source activate MYcondaENVair
```

```bash
pip install allennlp==1.2.2
pip install blingfire==0.1.7
pip install PyYAML==5.4
pip install transformers==3.5.1
```
