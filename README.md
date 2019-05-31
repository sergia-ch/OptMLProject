# OptMLProject
Frank-Wolfe Optimization method for NNs with minima sharpness analysis.

<a href="mailto:mariam.hakobyan@epfl.ch">Mariam Hakobyan</a>, <a href="mailto:sergei.volodin@epfl.ch">Sergei Volodin</a>. <a href="http://epfl.ch">Swiss Federal Institute of Technology in Lausanne</a> (EPFL)

We train small fully-connected networks on MNIST using (Frank-Wolfe, Adam, SGD) and measure minima sharpness via Hessian eigenvalues.

Mini-Project for Optimization for Machine Learning CS-439 at EPFL, 2019

## Optimizers
We consider SGD, Adam and Frank-Wolfe, with and without averaging. See our <a href="https://www.overleaf.com/read/hsvzyfcxkrhc
">report</a> for more details

## How to run experiments
Tested on Ubuntu 16.04.5 LTS with 12 CPU, 60GB of RAM and 2x GPU NVidia GeForce 1080.

1. Install <a href="https://docs.conda.io/en/latest/miniconda.html">Anaconda</a> (Python 3.7 option)
2. Create and activate an environment
3. Clone/download: `git clone https://github.com/sergeivolodin/OptMLProject.git; cd OptMLProject`
4. Install requirements: `pip install -r requirements.txt`. Install tensorflow-gpu by `conda install -c anaconda tensorflow-gpu`
5. Run Jupyter notebook `create_analyze_runs.ipynb`
6. Select the setting in the first cells (small/medium/big/batch size)
7. Generate `.sh` file in `output/` (next cells)
8. Navigate to `output/` and run your `.sh` file
9. It will produce `output/*.output` files and `output/figures/*.pdf` files
10. Run the rest of the notebook `create_analyze_runs` to aggregate the results

## Project structure
1. `experiment.py` the main file containing one experiment (loading optimizer, training, computing Hessian, computing metrics)
2. `helpers.py` contains a definition of a Fully-Connected Network `FCModelConcat()` with variables as a single tensor (needed to compute the Hessian). In addition, it contains our own implementation of the <a href="https://arxiv.org/abs/1804.09554">Stochastic Frank-Wolfe</a> method `StochasticFrankWolfe()`. This file also contains helper functions required in the experiments, such as training code, dataset loaders, Hessian calculation
3. `create_analyze_runs.ipynb` is the main notebook to create experiments and analyze them
4. `create_analyze_runs_helpers.py` is the helper file for the previous notebook containing code to make the results nice
5. `output/*.sh` files consist of many lines of the form `python ../experiment.py --param1 v1 --param2 v2 ...`, running at most 4 processes in total (2 per GPU)
6. `output/*.output` files contain outputs of `experiment.py` (one run corresponds to one file)
7. `output/figures` contains generated figures
10. Other files are not used
