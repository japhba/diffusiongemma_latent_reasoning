# Does DiffusionGemma have latent reasoning?

## TL;DR

Google DeepMind's recent model DiffusionGemma (DG) works differently from regular language models, including by passing distribution vectors in addition to tokens between generation steps. A priori, this allows it to pass information that is illegible to monitors, sometimes called "latent reasoning". A recent paper found that DG nevertheless maintains high monitorability, but at the same time found that ablating the passed distribution degrades performance.
Here, we show that this performance degradation is largely a sampler artifact, supporting the case for high monitorability. Nevertheless, we find some rare cases where the distribution vector is load-bearing computationally, however in a way that is easily interpretable.
Overall, this underlines the paper's conclusion that DiffusionGemma remains highly monitorable, while nevertheless showing that there are cases where models can learn to use vector-valued information.

## Introduction

DiffusionGemma is a text-generation model, based on the Gemma architecture. In short, generation looks like the following. Let $p$ be the prompt, and let $x_0\in\mathcal{V}^{C}$ be the noise-initialized token canvas comprising $C$ positions. The self-conditioning state $\mathbf{S}_0\in\mathbb{R}^{C\times |\mathcal{V}|}$ is initialised uninformatively — there is no model output to feed back at the first step. Let $f$ denote a single forward pass through the transformer stack, which is a finetune of Gemma. Roughly, the final output $x_T$ then is obtained via

$$
\begin{aligned}
&\textbf{for } t = 0, \dots, T-1: \\[2pt]
&\qquad \mathbf{S}_{t+1} = f(p,\, x_t;\, \mathbf{S}_t) \\[2pt]
&\qquad x_{t+1} = \mathrm{sample}(\mathbf{S}_{t+1})
\end{aligned}
$$

where $T$ is the number of diffusion steps. Importantly, $x_t$ attends bidirectionally to itself, and causally to $p$. $\mathbf{S}_t[x_t]$ also functions as a confidence score for any token $x_t$: unless a confidence threshold is passed, $x_t$ gets replaced with a random token at every step $t$, facilitating exploration and correction. Two consequences matter for this post: at such open positions the canvas carries no information from one step to the next — $\mathbf{S}_t$ is the only memory the model has there — and the pace of commitment is set by this confidence gate, so anything that artificially sharpens $\mathbf{S}_t$ (such as truncating it to its top-k entries) makes the gate commit more positions, earlier.

For a visual and more detailed introduction to DiffusionGemma, see [this post](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma).

Thus, generation in DiffusionGemma mainly differs in the following ways:

1. Generation happens as reverse diffusion, not as token-by-token autoregression. This is reminiscent of a looped transformer ([Giannou et al., 2023](https://arxiv.org/abs/2301.13196)).
2. Attention is bidirectional.
3. Between every diffusion step, not only the current text output is sampled, but the output distributions $\mathbf{S}_t$ across all positions are also passed to the next diffusion step $t+1$.

The last point is especially interesting, since it passes the vector-valued object between diffusion steps, in addition to the tokens $x_t$ lacking the $|\mathcal{V}|$ axis. A priori, this allows the model to transport drastically more information between diffusion steps in an illegible way, hindering monitorability.

## Performance degradation from top-k truncation largely is a sampler artifact

Despite this a-priori risk, [Engels et al.](https://arxiv.org/abs/2606.20560) found that DiffusionGemma scores similar monitorability to Gemma. [maybe give more deets] However, when truncating $\mathbf{S}_t$ to just its top-k entries, performance of DG significantly dropped. This was in conflict with their other results suggesting high monitorability, since somehow the information in $\mathbf{S}_t$ seemed to have been essential to DiffusionGemma.

When replicating their experiments, we observed that the model will often fall into a "degenerate loop", outputting the same token over and over, never reaching the final answer.

We found that adopting a gentler sampler largely prevents this failure mode, suggesting that in fact the distribution is not essential to solve these problems. However, this does not rule out that there is a functional necessity in other tasks that the paper had not investigated.

![GPQA truncation failure modes](figs/fig1_gpqa_trunc_failures.png)

*A gentler sampler prevents the degenerate loop that caused performance degradation on top-k truncating the distributional state $\mathbf{S}_t$ observed in [Engels et al.](https://arxiv.org/abs/2606.20560)*.

## A case study for using the distribution computationally: letter arithmetic

We next investigated whether there may still be some other tasks where $\mathbf{S}_t$ in fact is essential. Note that in principle, there is no need for the model to use $\mathbf{S}_t$ whatsoever to satisfy its training objective (indeed, most of the phenomena in [Engels et al.](https://arxiv.org/abs/2606.20560) are explained by bidirectional attention+looping). However, it may facilitate trainability and exploration.

We therefore looked for tasks where the model plausibly would hold several "hypotheses" in superposition. Note that superposition in a simple form is already present in the pretraining data ("Today the weather is _", with "rainy" and "sunny" both plausible), so that we were especially interested in cases where this may have been induced by the model's generalization. [+compuaiton]

We consider a task that requires DG to make an unspecified choice, and to do a compution on that choice. Specifically, we consider _letter arithmetic_:

```Pick any uppercase letter``` (the operand $x$) ```between A and W, write it, then write the letter``` (the target $x'$) ```3 ```(the increment $k$)``` positions later in the alphabet.```

DG answers these correctly on its own (e.g. ```Letters: G, J```). We then capture at some intermediate denoising step $t$, add probability mass $\varepsilon$ on a *different* source letter $x$ at the operand position. Importantly, we choose the injection such that $\mathbf{S}^t[x]+\epsilon$ is still not the top logit. This is important, because we would like to measure what DG does to states that are not the most likely ones.

Then, we measure the response to that perturbation $\mathbf{R}[x'_{i+1}\vert\mathrm{pert}(x_i)] = \log_{10}\big(\bar{\mathbf{S}}_{\mathrm{pert}}^{t+1}[x'_{i+1}]\, / \,\bar{\mathbf{S}}_{\mathrm{base}}^{t+1}[x'_{i+1}]\big)$, where $\bar{ \mathbf{S}}$ averages over the paired draws.

![Letter-arithmetic transfer maps](figs/fig2a_transfer_map.png)

*Perturbing subleading tokens triggers a response at corresponding target tokens.*

For illustration, here is a single intervention in full:

![Example intervention](figs/fig2a_example_intervention.png)

Example of one intervention: a subleading injection on ```H``` (the leader ```G``` keeps rank 1) swings the answer slot from ```J```  to ```K```=```H+3```  one step later.

### Parallelism

This simple response behavior begs the question whether it is possible to perturb $n>1$ source letters _simultaneously_ (again, keeping them subleading) and see whether the corresponding target images respond. To this end, we measure responses $\langle R\rangle_{T}$ and $\langle R\rangle_{N}$ averaged over the target and non-target sets, respectively, and introduce the _effect_ $E = \langle R\rangle_{T} - \langle R\rangle_{N}$. Because we now inject mass at multiple positions and observe proportionally scaled response, we calculate a _normalized effect_ $NE$.

![Parallelism triptych](figs/fig2b_triptych.png)

_DiffusionGemma does simultaneous letter arithmetic to about a capacity of $n=4$._

This behavior is plausible considering the computation in question: a shift is easily implemented by a linear rotation in representation space. Therefore, superpositions of letters will be transported to superpositions of responses. Overall, this leaves an interpretable picture.

## Conclusion

In this post, we have studied whether DiffusionGemma has latent reasoning, and found a capability to do  (parallel) computation on the vector-valued state passed between diffusion steps.

The fact that top-k truncation largely preserves accuracy suggests that no _significant_ latent reasoning is used in typical reasoning-focussed task.

More broadly, DiffusionGemma is only one testbed for latent reasoning. Going forward, there are many more architectures and forms where it can occur. 

## Appendix

The findings in Engels et al. and in this post have focussed on DiffusionGemma's behavior. We here study whether the model's representation supports the behavioral finding of high monitorability.

### Representation similarity (cosine & CKA)

We first ask about how the representation changes between Gemma and DiffusionGemma. A simple way is to just measure overlap between pairs of inputs, where each element of the pair is fed through Gemma or DiffusionGemma, respectively. We here compare a simple cosine similarity, and centered kernel analysis (CKA).

![RSA cosine and CKA](figs/figA1_rsa_cosine_cka.png)

*Representational similarity falls with layers, and is significantly driven by bare model difference and bidirectional attention.*

A priori, this rather large drop suggests a significant change in how the representation is organized, which we then went on to test more specifically.

### Probe retention

Probing allows to study how well a model separates concepts. We use the data from XXX and train on XXX, optimizing the layer via heldout AUC. We then test this probe on DG.

![Probe retention](figs/figA2_probe_retention.png)

*Top: mean held-out AUC (56 concepts, source-CV layer) — cross-model reading costs ~0.03 in both directions. Bottom: a held-out pair read by the same gemma-trained probe on both models' activations.*

#### DiffusionGemmas representation is more separable

### Steering retention

![Steering retention](figs/figA3_steer_retention.png)

*Top: blind-pair judge accuracy (judge must identify +steer vs −steer), 11 RepE tasks — every cell steers (0.70–0.85); bidirectional-mode directions are the weakest sources. Bottom: the same gemma-fit happiness direction applied to gemma-4 and DiffusionGemma side by side (±steer on one carrier); judge digests in italics.*

### J-Lens retention

![J-Lens retention](figs/figA4_jlens_retention.png)

J-lens

### A5: Autonomous computational usage of $\mathbf{S}^t$

In the _letter arithmetic_ task introduced in the main text, DiffusionGemma did respond to modifications of the distributional state in the way we expected. While suggestive, it is unclear if the model also _autonomously_ would make use of its state.

To investigate this, we searched for a non-trivial (i.e., not just a binary choice) task where DG will use a hypothesis subleading in its distribution autonomously. An instance we found is a word-level _palindrome_ task. When asked ```Please write a word-level palindrome```, DG maintains two competing completions of the same canvas: a literal _seasonal_ phrase (```fall leaves as soon as leaves fall```) and the famous _idiom_ (```all for one and one for all```).

The canvas will read the _seaonal_ answer for the first few diffusion steps, after which it flips and stays at _idiom_. Interestingly, this is accompanied by _dynamics_ in $\mathbf{S}^t$: _idiom_ will start at ~0, and progressively gain weight, replacing _seasonal_.

We validated that the emergence of _idiom_ is indeed causal: when ablating _idiom_'s tokens, the transition can be prevented. Note however that early and late ablation do not have this effect.

![Seasonal vs idiom: ember kill](figs/figA5_seasonal_ember_kill.png)

*Left: sheet mass of the two completions at the contested slots (black: idiom, purple: seasonal; seed s3) — the base run (top, separated), then single-step ablations of the idiom's $\mathbf{S}$-mass (red ×): an early kill regrows, a mid-ramp kill destroys the escape, a late kill is a no-op. Right: final outcome across all arms × ablation steps — single-step kills, persistent kills (grey = collapse into a third, degenerate basin), and incumbent-kills on the trapped seed s0, where silencing the seasonal incumbent for a single step rescues the idiom at any time. Cell text = draft-flip step (— = never); outlined cells = the panels at left.*
