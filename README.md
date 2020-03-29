# Deep Learning with Pytorch Essentials
This repository describes how to build, train and test deep learning networks in pytorch. 
New models can be added to the models directory and loaded in main.py.

## Installation
Create a virtual environment using venv

```bash
python3 -m venv env
```

Source the virtual environment

```bash
source env/bin/activate
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py  --mode=[Train/Test] --resume=[True/False]
```
There are more flags that can be found in the main function.
## License
[MIT](https://choosealicense.com/licenses/mit/)
