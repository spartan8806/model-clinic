"""References and educational links for model-clinic conditions.

Maps each condition to relevant papers, blog posts, and guides that
explain the underlying issue and remediation strategies.
"""


CONDITION_REFERENCES = {
    # ── Static conditions ─────────────────────────────────────────────
    "dead_neurons": [
        {
            "title": "Delving Deep into Rectifiers (He et al., 2015)",
            "url": "https://arxiv.org/abs/1502.01852",
            "note": "Kaiming initialization — prevents dead neurons in ReLU networks",
        },
        {
            "title": "Deep Sparse Rectifier Neural Networks (Glorot et al., 2011)",
            "url": "https://proceedings.mlr.press/v15/glorot11a.html",
            "note": "Analysis of sparsity and dead units in deep networks with ReLU",
        },
    ],
    "stuck_gate_closed": [
        {
            "title": "Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)",
            "url": "https://doi.org/10.1162/neco.1997.9.8.1735",
            "note": "Original LSTM paper — gate initialization affects learning dynamics",
        },
        {
            "title": "Learning to Forget (Gers et al., 2000)",
            "url": "https://doi.org/10.1162/089976600300015015",
            "note": "Forget gate bias initialization to prevent stuck-closed gates",
        },
    ],
    "stuck_gate_open": [
        {
            "title": "Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)",
            "url": "https://doi.org/10.1162/neco.1997.9.8.1735",
            "note": "Gate saturation prevents gradient flow and blocks learning",
        },
        {
            "title": "An Empirical Exploration of Recurrent Network Architectures (Jozefowicz et al., 2015)",
            "url": "https://proceedings.mlr.press/v37/jozefowicz15.html",
            "note": "Systematic study of gate initialization and bias strategies",
        },
    ],
    "exploding_norm": [
        {
            "title": "Understanding the Difficulty of Training Deep Feedforward Neural Networks (Glorot & Bengio, 2010)",
            "url": "https://proceedings.mlr.press/v9/glorot10a.html",
            "note": "Xavier initialization — stabilizes norms across layers",
        },
        {
            "title": "On the Difficulty of Training Recurrent Neural Networks (Pascanu et al., 2013)",
            "url": "https://arxiv.org/abs/1211.5063",
            "note": "Gradient clipping as a remedy for exploding gradients",
        },
    ],
    "vanishing_norm": [
        {
            "title": "Understanding the Difficulty of Training Deep Feedforward Neural Networks (Glorot & Bengio, 2010)",
            "url": "https://proceedings.mlr.press/v9/glorot10a.html",
            "note": "Proper initialization prevents vanishing signals in deep networks",
        },
        {
            "title": "Deep Residual Learning for Image Recognition (He et al., 2016)",
            "url": "https://arxiv.org/abs/1512.03385",
            "note": "Skip connections alleviate vanishing gradients in very deep models",
        },
    ],
    "heavy_tails": [
        {
            "title": "Implicit Self-Regularization in Deep Neural Networks (Martin & Mahoney, 2019)",
            "url": "https://arxiv.org/abs/1901.08276",
            "note": "Heavy-tailed weight distributions as a sign of implicit regularization",
        },
        {
            "title": "Heavy-Tailed Universality Predicts Trends in Test Accuracies (Martin & Mahoney, 2020)",
            "url": "https://arxiv.org/abs/1901.08278",
            "note": "Weight matrix spectral analysis and heavy-tail metrics for model quality",
        },
    ],
    "norm_drift": [
        {
            "title": "Layer Normalization (Ba et al., 2016)",
            "url": "https://arxiv.org/abs/1607.06450",
            "note": "LayerNorm stabilizes training — drift from 1.0 indicates training issues",
        },
        {
            "title": "Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)",
            "url": "https://arxiv.org/abs/1910.07467",
            "note": "RMSNorm variant — norm weight drift affects model stability",
        },
    ],
    "saturated_weights": [
        {
            "title": "Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy, 2015)",
            "url": "https://arxiv.org/abs/1502.03167",
            "note": "Weight saturation reduces effective dynamic range; normalization helps",
        },
    ],
    "nan_inf": [
        {
            "title": "Mixed Precision Training (Micikevicius et al., 2018)",
            "url": "https://arxiv.org/abs/1710.03740",
            "note": "Loss scaling prevents NaN/Inf in reduced-precision training",
        },
        {
            "title": "BFloat16: The Secret to High Performance on Cloud TPUs (Google, 2019)",
            "url": "https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus",
            "note": "bfloat16 reduces overflow risk compared to float16",
        },
    ],
    "identical_rows": [
        {
            "title": "Exact Solutions to the Nonlinear Dynamics of Learning (Saxe et al., 2014)",
            "url": "https://arxiv.org/abs/1312.6120",
            "note": "Symmetry breaking in weight initialization — identical rows block learning",
        },
    ],
    "attention_imbalance": [
        {
            "title": "Attention is All You Need (Vaswani et al., 2017)",
            "url": "https://arxiv.org/abs/1706.03762",
            "note": "Multi-head attention — imbalanced heads waste model capacity",
        },
        {
            "title": "Are Sixteen Heads Really Better than One? (Michel et al., 2019)",
            "url": "https://arxiv.org/abs/1905.10650",
            "note": "Many attention heads are redundant and can be pruned",
        },
    ],
    "dtype_mismatch": [
        {
            "title": "Mixed Precision Training (Micikevicius et al., 2018)",
            "url": "https://arxiv.org/abs/1710.03740",
            "note": "Consistent dtype across the model avoids precision-related instability",
        },
    ],
    "weight_corruption": [
        {
            "title": "Mixed Precision Training (Micikevicius et al., 2018)",
            "url": "https://arxiv.org/abs/1710.03740",
            "note": "Numerical instability from dtype issues can corrupt weights",
        },
    ],
    "head_redundancy": [
        {
            "title": "Are Sixteen Heads Really Better than One? (Michel et al., 2019)",
            "url": "https://arxiv.org/abs/1905.10650",
            "note": "Redundant heads can be pruned with minimal accuracy loss",
        },
        {
            "title": "A Multiscale Visualization of Attention in the Transformer Model (Vig, 2019)",
            "url": "https://arxiv.org/abs/1906.05714",
            "note": "Tools for visualizing and understanding attention head behavior",
        },
    ],
    "positional_encoding_issues": [
        {
            "title": "Attention is All You Need (Vaswani et al., 2017)",
            "url": "https://arxiv.org/abs/1706.03762",
            "note": "Sinusoidal positional encodings — corruption breaks sequence modeling",
        },
        {
            "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)",
            "url": "https://arxiv.org/abs/2104.09864",
            "note": "Rotary position embeddings — issues here affect long-context performance",
        },
    ],
    "token_collapse": [
        {
            "title": "Understanding Dimensional Collapse in Contrastive Self-Supervised Learning (Jing et al., 2022)",
            "url": "https://arxiv.org/abs/2110.09348",
            "note": "Dimensional collapse causes token representations to converge",
        },
    ],
    "gradient_noise": [
        {
            "title": "An Empirical Model of Large-Batch Training (McCandlish et al., 2018)",
            "url": "https://arxiv.org/abs/1812.06162",
            "note": "Gradient noise scale determines optimal batch size and training stability",
        },
        {
            "title": "Don't Decay the Learning Rate, Increase the Batch Size (Smith et al., 2018)",
            "url": "https://arxiv.org/abs/1711.00489",
            "note": "Relationship between gradient noise, batch size, and learning rate",
        },
    ],
    "representation_drift": [
        {
            "title": "Catastrophic Forgetting in Neural Networks (French, 1999)",
            "url": "https://doi.org/10.1016/S1364-6613(99)01294-2",
            "note": "Representation drift across layers signals catastrophic forgetting",
        },
    ],
    "moe_router_collapse": [
        {
            "title": "Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2022)",
            "url": "https://arxiv.org/abs/2101.03961",
            "note": "Load-balancing loss prevents router collapse in MoE models",
        },
        {
            "title": "ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022)",
            "url": "https://arxiv.org/abs/2202.08906",
            "note": "Router z-loss and auxiliary losses for stable expert utilization",
        },
    ],
    "lora_merge_artifacts": [
        {
            "title": "LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022)",
            "url": "https://arxiv.org/abs/2106.09685",
            "note": "Merging LoRA adapters can leave rank-deficient artifacts in weight matrices",
        },
    ],

    "causal_outlier": [
        {
            "title": "Locating and Editing Factual Associations in GPT (Meng et al., 2022)",
            "url": "https://arxiv.org/abs/2202.05262",
            "note": "Causal tracing identifies which layers store and recall factual knowledge",
        },
        {
            "title": "Understanding the Difficulty of Training Deep Feedforward Neural Networks (Glorot & Bengio, 2010)",
            "url": "https://proceedings.mlr.press/v9/glorot10a.html",
            "note": "Norm outliers across layers indicate initialization or training instability",
        },
    ],
    "layer_isolation": [
        {
            "title": "Locating and Editing Factual Associations in GPT (Meng et al., 2022)",
            "url": "https://arxiv.org/abs/2202.05262",
            "note": "Causal tracing reveals isolated layers that dominate or block information flow",
        },
        {
            "title": "Catastrophic Forgetting in Neural Networks (French, 1999)",
            "url": "https://doi.org/10.1016/S1364-6613(99)01294-2",
            "note": "Isolated layers with divergent norms may indicate catastrophic forgetting",
        },
    ],

    # ── Runtime conditions ────────────────────────────────────────────
    "generation_collapse": [
        {
            "title": "The Curious Case of Neural Text Degeneration (Holtzman et al., 2020)",
            "url": "https://arxiv.org/abs/1904.09751",
            "note": "Nucleus sampling and analysis of repetitive/degenerate text generation",
        },
        {
            "title": "A Theoretical Analysis of the Repetition Problem in Text Generation (Fu et al., 2021)",
            "url": "https://arxiv.org/abs/2012.14660",
            "note": "Why models get stuck in repetition loops during generation",
        },
    ],
    "low_coherence": [
        {
            "title": "The Curious Case of Neural Text Degeneration (Holtzman et al., 2020)",
            "url": "https://arxiv.org/abs/1904.09751",
            "note": "Analysis of coherence failures in neural text generation",
        },
    ],
    "activation_nan": [
        {
            "title": "Mixed Precision Training (Micikevicius et al., 2018)",
            "url": "https://arxiv.org/abs/1710.03740",
            "note": "NaN activations often stem from overflow in reduced-precision compute",
        },
    ],
    "activation_inf": [
        {
            "title": "Mixed Precision Training (Micikevicius et al., 2018)",
            "url": "https://arxiv.org/abs/1710.03740",
            "note": "Inf activations from overflow — loss scaling and dtype casts help",
        },
    ],
    "activation_explosion": [
        {
            "title": "On the Difficulty of Training Recurrent Neural Networks (Pascanu et al., 2013)",
            "url": "https://arxiv.org/abs/1211.5063",
            "note": "Exploding activations share root causes with exploding gradients",
        },
        {
            "title": "Fixup Initialization: Residual Learning Without Normalization (Zhang et al., 2019)",
            "url": "https://arxiv.org/abs/1901.09321",
            "note": "Proper residual scaling prevents activation explosion in deep models",
        },
    ],
    "activation_collapse": [
        {
            "title": "Deep Residual Learning for Image Recognition (He et al., 2016)",
            "url": "https://arxiv.org/abs/1512.03385",
            "note": "Residual connections prevent activation collapse in deep networks",
        },
    ],
    "residual_explosion": [
        {
            "title": "Fixup Initialization: Residual Learning Without Normalization (Zhang et al., 2019)",
            "url": "https://arxiv.org/abs/1901.09321",
            "note": "Residual branch scaling prevents explosion in deep residual networks",
        },
        {
            "title": "On the Difficulty of Training Recurrent Neural Networks (Pascanu et al., 2013)",
            "url": "https://arxiv.org/abs/1211.5063",
            "note": "Gradient clipping as a practical remedy for exploding signals",
        },
    ],
    "residual_collapse": [
        {
            "title": "Deep Residual Learning for Image Recognition (He et al., 2016)",
            "url": "https://arxiv.org/abs/1512.03385",
            "note": "Skip connections are essential — collapse means residuals are not contributing",
        },
    ],
}


def get_references(condition: str) -> list:
    """Get references for a condition. Returns empty list if none."""
    return CONDITION_REFERENCES.get(condition, [])


def format_references(condition: str) -> str:
    """Format references for CLI display."""
    refs = get_references(condition)
    if not refs:
        return ""
    lines = [f"  References for {condition}:"]
    for ref in refs:
        lines.append(f"    - {ref['title']}")
        if ref.get("url"):
            lines.append(f"      {ref['url']}")
        if ref.get("note"):
            lines.append(f"      {ref['note']}")
    return "\n".join(lines)
