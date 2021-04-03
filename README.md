# Sentiment-driven statistical causality in multimodal systems

This repository contains the code for the paper "Sentiment-driven statistical causality in multimodal systems", by Ioannis Chalkiadakis, Anna Zaremba, Gareth W. Peters and Michael J. Chantler.

Please cite the paper as follows:

_Chalkiadakis, Ioannis and Zaremba, Anna and Peters, Gareth and Chantler, Michael John, Sentiment-Driven Statistical Causality in Multimodal Systems (December 8, 2020). Available at SSRN: https://ssrn.com/abstract=3742063_

_Chalkiadakis, Ioannis and Zaremba, Anna and Peters, Gareth and Chantler, Michael John, Sentiment-driven statistical causality in multimodal systems: Supplementary appendix (April 3, 2021). Available at SSRN: https://ssrn.com/abstract=3818944

If you use the MatLab code for the Multiple-Output Gaussian Process Statistical Causal model, please also cite the following paper:

_Zaremba, Anna and Peters, Gareth, Statistical Causality for Multivariate Non-Linear Time Series via Gaussian Processes (May 24, 2020). Available at SSRN: https://ssrn.com/abstract=3609497 or http://dx.doi.org/10.2139/ssrn.3609497

The available sentiment data contain the sentiment entropy index per news source, as well as the combined weighted sentiment indices that were used in the studies.

Contents of the current repository:

  1. Folder "appendix" contains the supplementary appendix with extra studies we conducted.
  
  2. Folder data contains the financial time-series (subfolder "finance"), the dictionaries (subfolder "dictionaries"), the text time-series per asset and news source (subfolder "sentiment"), and the constructed sentiment indices we used in the studies (subfolders "NLP1 - 2 - 3", each for the corresponding index as described in the paper).

  3. Folder "python" contains the Python code for constructing the sentiment indices.

  4. Folder "matlab" contains MatLab code for the Multiple-Output Gaussian Process Statistical Causal model


Tested with Python v3.6.7 and the <cite><a href="http://www.gaussianprocess.org/gpml/code/matlab/doc/">Gaussian Process Toolbox</a></cite> v3.5.

Copyright @ Authors April 2021

