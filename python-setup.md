conda env --name py3 python=3
source activate py3
conda install tensorflow
python -m ipykernel install --user --name py3 --display-name "Tensorflow"
jupyter notebook