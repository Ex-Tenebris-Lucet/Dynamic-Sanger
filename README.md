\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{Self-Sizing, Forgetting-Proof Online PCA: Norm-Driven Sanger}
\author{Bryan Jensen}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Sanger's PCA has served for decades as a clean way to converge k neurons to the first k Principal Components of a data stream, but typically finds friction with the needs of real-world implementations.
The number of potential PCs must be chosen in advance, the learning rates must be tuned carefully per application, and the core algorithm defines no mechanism to stop learning once converged, allowing the representations to wander even under steady stream conditions. 
This paper presents four interconnected mechanisms that make Sanger's online PCA self-sizing, sample-efficient, and forgetting-proof, achieving ~99.5% of optimal PCA without tuning, while remaining stable across distribution shifts.
\end{abstract}


\section{Introduction}
\label{sec:intro}

When representations need to shift completely to adapt to changing input streams, ordinary Sanger, and related algorithms, are already near optimal. 
However, in real-world applications, it is typically important for learned representations to remain consistent throughout the lifetime of the system. 
In most real systems that use PCA, downstream components require stable representations from the compression, otherwise they're forced to continually relearn as the mapping gradually shifts and rotates. 
Typically this requires a variety of compromises: either PCA must be run first and then frozen, which is not always possible or convenient in online systems, or the system as a whole must be designed to remain functional over unstable foundations.

CCIPCA requires an amnesic parameter, with no mechanisms that allow the stabilization of converged neurons, while also allowing for adaptation to new data; it must either remain plastic and unstable, or settle after an arbitrary number of samples.
AdaOja introduces adaptive learning rates based on gradient histories, which enables hyperparameter-free learning, but does not index on true PCA convergence. 
Migenda et al. allows for dynamic k size selection through estimation of eigenvalue spectrum, but makes potentially fragile assumptions about the expected shape of data, and does not retain stable representations as data shifts.

This work provides a combination of four mechanisms which optimize the Sanger algorithm for real-world application: norm-based adaptive learning rates, sequential gating, input energy normalization, and dynamic k discovery.
Together, these refinements create an implementation of online PCA which is stable, persistent, requires little tuning, and uses very little additional computation compared to standard Sanger.

\section{Background}
\label{sec:background}

PCA is a form of compression through dimensionality reduction, which preserves maximal variance from the input space.
Sanger's rule, also known as the Generalized Hebbian Algorithm, is an online implementation that allows the discovery of Principal Components from data as it is streamed through the layer. 
\begin{equation}
\Delta w_{ij} = \eta \, y_i \left( x_j - \sum_{k=1}^{i} w_{kj} \, y_k \right)
\label{eq:sanger}
\end{equation}
The individual weight vectors converge towards unit norm, and this work shows that the unit convergence is a useful signal.

\section{Method}
\label{sec:method}

\subsection{Norm-Based Adaptive Learning Rate}
\label{sec:norm-lr}

Each neuron gets a learning rate modifier set by
\begin{equation}
\alpha_i = \min\!\left(1,\; \sqrt{\left|\,\|\mathbf{w}_i\|^2 - 1\,\right|}\right)
\label{eq:own_factor}
\end{equation} 
This creates a function where the learning rate is higher the further the neuron's weights are from unit norm, and falls into a deep well as it closes in on the correct eigenvector. 
When the vector length reaches 1, the learning factor becomes 0, and the neuron never changes again.
Because each neuron stabilizes onto a true eigenvector present in the data, then stops updating once converged, coordinates within the converged PC space are consistent throughout the lifetime of the system.

\subsection{Sequential Gating}
\label{sec:gating}

Unit norm driven learning rate is powerful because it allows dramatic early steps in correct directions; however the nature of GHA ultimately causes the steps to be noisy until all of a neuron's predecessors are relatively converged.
The remaining signal, once the earlier components are settled, is necessarily in the correct direction.
For this reason, it is beneficial to use sequential gating, driven by the norm of the previous neuron's weights, determined by
\begin{equation}
\beta_i = \max\!\left(0,\; 1 - \left(\|\mathbf{w}_{i-1}\|^2 - 1\right)^2\right)
\label{eq:pred_gate}
\end{equation}
This allows the PCs to be extracted quickly and correctly, with each neuron gradually unlocked as its predecessor converges.

\subsection{Input Energy Normalization}
\label{sec:energy-norm}

The step sizes determined by GHA are fundamentally driven by the scale of the data being learned, despite the useful fact that the neurons always converge to unit weight norm.
In order to stabilize learning across data regimes, it is necessary to normalize the magnitude of weight updates by the square of the input vector norm. 
Combining the three factors, the update rule becomes
\begin{equation}
\Delta \mathbf{w}_i = \frac{\alpha_i \, \beta_i \, \eta}{{\|\mathbf{x}\|^2 + \epsilon}} \; y_i \left( \mathbf{x} - \sum_{k=1}^{i} y_k \, \mathbf{w}_k \right)
\label{eq:full_update}
\end{equation}
Each neuron learns quickly, in sequential order, at a consistent speed regardless of input magnitude.
The input is also EMA centered to allow for implicit bias correction on data which is not zero-mean.

\subsection{Dynamic Rank Selection}
\label{sec:dynamic}

The three mechanisms described above create a stable learning paradigm where an arbitrary number of neurons will correctly converge on the true PC structure of the input data streams, with excess neurons sitting nearly silent.
Additionally, this creates an opportunity for a clean and simple dynamic sizing mechanism.
Experiments showed that neurons which are not capturing true variance in the data rarely reach $\|w\| = 0.05$, while neurons picking up real variance quickly grow towards the unit norm. 
By starting at $k=2$, and adding an additional neuron when the last neuron reaches a norm of $0.25$, the Sanger layer inherently grows itself to match the intrinsic dimensionality of the input stream, settling on $n + 1$, where $n$ is the number of principal components with significant variance.
The final neuron acts as a listener ready to pick up any new dimensionality, should the structure of the data change.

\section{Experiments}
\label{sec:experiments}

\subsection{Synthetic: Convergence and Quality}
\label{sec:synthetic}

Insofar as convergence speed and explained variance, there are no real differences of note between all compared approaches.
Ordinary Sanger, CCIPCA, AdaOja, and this work all converge quickly and correctly. 
The primary advantages of this set of mechanisms are twofold; no tuning is required across tests and regimes, even when the dimensionality of the data shifts, and the learned representations show zero drift once settled, allowing online PCA to be integrated as a stable part of a complete system.
Over several thousand samples of persistent data within a steady regime, the representations formed by pure Sanger rotated around 45 degrees on average, CCIPCA rotated around 5 degrees, and AdaOja rotated around 12 degrees. 
In contrast, this dynamic Sanger implementation was self-sizing, and each neuron which captured true variance and converged was automatically frozen.

\subsection{Continual Learning: Atari}
\label{sec:atari}

As a test of the online capabilities of this variant, it was run on visual input from 32 Gabor filters applied over the 26 games in the Atari 100k benchmark. 
A tunable threshold was used to stop learning once a given portion of input variance was explained. 
As is expected with real-world data, some level of structure is present across the full dimensionality of the input, and running without a threshold eventually grows as many neurons as input dimensions. 
The threshold allows a design decision about how precise the compression should be, without hardcoding the number of principal components. 
With a threshold of 0.95, 17 fully converged neurons explained >95% of variance in the samples. 
With a threshold of 0.99, 25 fully converged neurons explained >99% of variance. 
In both cases, no new neurons were grown during a second pass over the data, and the representations stayed stable for all converged neurons. 

\section{Related Work}
\label{sec:related}

Oja's rule \cite{oja1982} established the foundation for online PCA via a single-neuron Hebbian update.
Sanger \cite{sanger1989} extended this to multiple components via hierarchical subtraction, producing the Generalized Hebbian Algorithm.
CCIPCA \cite{weng2003ccipca} maintains unnormalized weight vectors whose norms encode eigenvalue estimates, updated through a weighted running average.
AdaOja \cite{henriksen2019adaoja} applies AdaGrad-style gradient accumulation to Oja's rule, removing the need to tune learning rates.
However, it adapts based on gradient history rather than convergence state, and cannot distinguish a converged neuron from one that is receiving small gradients.
All of these algorithms require predefinition of the k parameter, where this work allows for the definition of precision instead. 

INO-PCA \cite{demir2025inopca} studies the relationship between weight vector norms and convergence for single-component Oja, showing that the norm encodes useful information about signal strength.
This work extends that insight to the multi-component setting, using norms as explicit per-neuron learning rate gates with sequential dependencies between neurons.

Migenda et al. \cite{migenda2021adaptive} address dynamic rank selection by extrapolating the eigenvalue spectrum from a small number of tracked components.
This requires assumptions about the spectral shape of the data.
Lv et al. \cite{lv2007determination} propose an improved GHA with lateral connections for adaptive rank determination.
In contrast, the norm-based growth mechanism presented here makes no assumptions about spectral shape and uses a direct observation --- whether a neuron has found real and substantial variance --- as the growth signal.

The biologically plausible similarity matching framework of Pehlevan and Chklovskii \cite{pehlevan2018bioplausible} derives online PCA from a different optimization principle and achieves faster convergence than GHA in some settings, but does not address representation stability or dynamic sizing.

\section{Discussion}
\label{sec:discussion}

This work provides refinements and combinations of abilities which are not present all together in previous work, specifically the ability to continue learning as data shifts, without letting converged representations wander.
These traits allow for the application of online PCA as part of a continual learning system, but naturally exclude this work from relevance when tracking a rotating input space with fixed capacity is a necessary behavior. 
As previously mentioned, this work is not completely free from hyperparameters, but instead changes what the architect has control over, namely the ability to choose precision with a variance capture threshold, instead of explicitly defining the number of neurons available. 
The variance capture threshold also provides the opportunity to skip learning on adequately represented inputs, saving compute without disabling learning outright. 
Code is available at \url{https://github.com/Ex-Tenebris-Lucet/Dynamic-Sanger}

\bibliographystyle{plain}
\bibliography{refs}

\end{document}
