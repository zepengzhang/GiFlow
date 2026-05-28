<div align="center">

<h1>Spatiotemporal Imputation with Graph-Informed Flow Matching</h1>

<div>
    <a href='https://zepengzhang.com/' target='_blank'>Zepeng Zhang</a><sup>1</sup>&emsp;
    <a href='https://arefeinizade.github.io/' target='_blank'>Aref Einizade</a><sup>2</sup>&emsp;
    <a href='https://jhonygiraldo.github.io/' target='_blank'>Jhony H. Giraldo</a><sup>3</sup>&emsp;
    <a href='https://people.epfl.ch/olga.fink?lang=en' target='_blank'>Olga Fink</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>EPFL, <sup>2</sup>Télécom SudParis, Institut Polytechnique de Paris, <sup>3</sup> Télécom Paris, Institut Polytechnique de Paris
</div>

<div>
    <h4 align="center">
        • <a>ICML 2026</a> •
    </h4>
</div>

</div>

## Abstract
Missing data is a common challenge in spatiotemporal systems, arising in applications such as air quality monitoring and urban traffic management. 
Traditional machine learning approaches, like recurrent and graph neural networks, rely on iterative propagation, which tends to accumulate errors over time and space. 
Recent diffusion-based methods mitigate error propagation but require iterative sampling and often depend on problem-agnostic Gaussian priors, limiting both efficiency and effectiveness.
To address these limitations, we propose GiFlow, a Graph-Informed Flow Matching framework for spatiotemporal imputation. 
GiFlow replaces the typical Gaussian prior with a graph-informed prior constructed via spatiotemporal filtering of observable signals, which better aligns the source distribution to the target and thereby simplifies the generation trajectory.
The flow field is parameterized by a hybrid vector field model that integrates spatial attention, temporal attention, and spatiotemporal propagation, enabling joint modeling of spatial and temporal dependencies. 
Extensive experiments on both synthetic and real-world datasets demonstrate that the proposed GiFlow outperforms the state-of-the-art approaches in spatiotemporal imputation.

## Code
The code will be available soon.

## Citation

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{zhang2026giflow,
  title={Spatiotemporal Imputation with Graph-Informed Flow Matching},
  author={Zhang, Zepeng and Einizade, Aref and Giraldo, Jhony H. and Fink, Olga},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
