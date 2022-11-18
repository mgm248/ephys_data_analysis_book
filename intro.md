# Analysis of Electrophysiological Data in Neuroscience
*Tutorials and scripts for analyzing electrophysiological (LFP and spike train) data.* 
## Introduction
In neuroscience, electrophysiological data is commonly collected in order to analyze neural spiking and local field potential activity. Much of the material covers frequently used analyses in the field, such as power spectra / spectrograms, coherence, spike raster / rate plots, and spike-lfp coupling. In addition, these tutorials contain cover less frequently applied analyses, such as measures of directionality between LFP signals, and measures of spike synchrony. 
##  Contents
```{tableofcontents}
```
## Requirements
matplotlib, numpy, scipy, mne$^{1}$, math, itertools, random, time, elephant$^{2}$, viziphant$^{2}$, neo$^{3}$, quantaties, sklearn, seaborn

Noted packages are dedicated to electrophysiological data analysis, including:
1. [MNE](https://mne.tools/stable/index.html): Primarily designed for analyzing EEG+MEG data, this package still is a great resource for analyzing LFP data
2. [Elephant](https://elephant.readthedocs.io/en/latest/index.html) + [viziphant](https://viziphant.readthedocs.io/en/latest/) (visualization companion package): Excellent package for spike train analyses, in addition to some LFP analyses
3. [Neo](https://neo.readthedocs.io/en/stable/): Supports ephys object structures
## Authors
This material was created by the [Computational Physiology Laboratory](http://cplab.net/) at Cornell University. Michael Mariscal wrote and published this work, with support from PI's Thomas Cleland and Christiane Linster, and additional help from Jesse Werth and Matt Einhorn.  