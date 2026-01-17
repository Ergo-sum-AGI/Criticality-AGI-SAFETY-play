# RG-Φ: Renormalization Group Framework for AGI Safety & Criticality Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/Ergo-sum-AGI/Criticality-AGI-SAFETY-play?style=social)](https://github.com/Ergo-sum-AGI/Criticality-AGI-SAFETY-play)

> "Thought must never submit to dogma, to a party, to a passion, to an interest, to a preconception, or to anything other than facts themselves."  
> — Henri Poincaré (1909)

**RG-Φ** applies tools from theoretical physics — **Renormalization Group (RG) theory**, ε-expansion, Callan-Symanzik equations, Operator Product Expansion (OPE), and critical exponents — to rigorously analyze AGI safety risks: phase transitions, emergent capabilities, scaling laws, alignment decay, and self-model instability.

This repo contains a single executable script (`RG-AGI-SAFETY-PLAY.py` / `rg_phi_agi_complete.py`) that computes and visualizes the full apparatus, generating 7 research-grade plots.

## Why RG for AGI Safety?

Intelligence systems may behave like physical systems near criticality:
- **Phase transitions** → sudden capability jumps (fast takeoff)
- **Fixed points** → equilibrium intelligence levels
- **Relevant/irrelevant operators** → which features dominate or vanish at scale
- **OPE** → predictable emergence from primitive combinations
- **Self-reference loops** → Gödelian limits on self-transparency

These analogies yield **sharp, testable theorems** — not metaphors.

## Four Core Safety Theorems

1. **Capability Discontinuity**  
   Near ε ≈ 0, capabilities may jump discontinuously:  
   `C(μ) ~ A + B·(μ - μ_c)^β` (β = ν(d-2+η)).  
   → **Risk**: First-order transitions = no warning.

2. **Universality of Emergence**  
   Primitives A, B with OPE coeff. C_AB^E > 0 produce emergent E:  
   `P(E | A ∧ B) ≥ |C_AB^E|² · ρ(A) · ρ(B)`  
   → **Safety**: Suppress dangerous OPE channels.

3. **Self-Model Instability**  
   Measurement back-reaction decays self-models:  
   `F(t) = F₀ / (1 + κ·I(M)·F₀·t)`  
   → **Gödelian bound**: Perfect transparency is unstable.

4. **Alignment Scaling Law**  
   If alignment is irrelevant (Δ_align > Δ_cap):  
   `R(μ) ~ μ^(Δ_align - Δ_cap) → 0` as μ → ∞  
   → **Catastrophic**: Alignment dies at scale unless engineered relevant.

## Quick Start

```bash
# Clone repo
git clone https://github.com/Ergo-sum-AGI/Criticality-AGI-SAFETY-play.git
cd Criticality-AGI-SAFETY-play

# Install dependencies
pip install mpmath numpy scipy matplotlib

# Run (generates 7 plots + console output)
python rg_phi_agi_complete.py
