# Formulating Multistage Cutting Stock Problems as QUBO

[Marcel Seelbach Benkner<sup>1,2</sup>](https://www.vsa.informatik.uni-siegen.de/en/seelbach-marcel),  [Sebastian Nagies,<sup>3,4</sup>](https://scholar.google.com/citations?user=O23y3A8AAAAJ&hl=it),   [Chiara Capecci<sup>3,4</sup>](https://scholar.google.com/citations?user=8-cjSkAAAAAJ&hl=it), [Javed Akram<sup>2</sup>](https://eleqtron.com/), [Sebastian Rubbert<sup>2</sup>](https://eleqtron.com/), [Dimitrios Bantounas<sup>2</sup>](https://eleqtron.com/), [Philipp Hauke<sup>3,4</sup>](https://hauke-group.physics.unitn.it/authors/hauke/), [Michael Johanning<sup>2</sup>](https://eleqtron.com/) and [Michael Möller<sup>1</sup>](https://sites.google.com/site/michaelmoellermath/)

 <sup>1</sup>University of Siegen, <sup>2</sup>eleQtron GmbH,  <sup>3</sup>Pitaevskii BEC Center and Department of Physics, <sup>4</sup>INFN-TIFPA, Trento Institute for Fundamental Physics and Applications.

This is the official repository for the publication "Formulating Multistage Cutting Stock Problems as QUBO".
The work presents a QUBO formulation of the multstage 2D cutting stock problem. The QUBOs could for example be solved via quantum annealing, but in this repository the simulated annealing sampler are used for experimental tests.
The preprint can be found at [TBD](https://arxiv.org/abs/2507.12536).




[![arXiv](http://img.shields.io/badge/arXiv-2507.12536-b31b1b.svg)](https://arxiv.org/abs/2507.12536)
![Python](http://img.shields.io/badge/python-%3E%3D3.8-blue)



## Getting Started
-The repository can be cloned with <br/>
`git clone https://github.com/MSeelbach/2DCS_QUBOs/` <br/>
-We recommend the user to set up a conda environment with
```
conda create -n 2DCS_QUBOs-env python=3.8
conda activate 2DCS_QUBOs-env
```
-After this execute <br/>
`conda install pandas, matplotlib` <br/>


## How to access benchmark instances
To conduct expirements on literature problem instances we recommend a download from https://github.com/henriquebecker91/phd/tree/master/instances. 

## Code description
In the following we give a description about the content of the folders:

- Benchmarking contains code to reproduce the Numerical Experiments with Simulated Annealing Sampler. The folder SmallSimulation contains benchmark instances to test quantum annealing protocol simulations.

- mainPlot.py is the main class to load problem instances solve them and plot the solutions.

- In QUBOSA.py the Augmented Lagrangian Method from the publication is written down.

- The utilityclasses folder should make using other solvers easier in our context: The class Highssolver is for solving the linear programming formulation via the [HiGHs](https://highs.dev/) solver and the NealSolver class uses [D-Wave neal](https://docs.ocean.dwavesys.com/projects/neal/en/latest/) to solve QUBOs.




## License
The MIT License (MIT)

Copyright (c) 2026 Marcel Seelbach Benkner

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
