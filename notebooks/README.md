# AnalyticLoadFlow workshop notebooks

These notebooks run on Google Colab, no local installation required. The
first cell installs AnalyticLoadFlow from GitHub (branch `main`) into a fresh
temporary environment (takes a few minutes); a commented line in the same
cell switches to the latest registered release.

| Notebook | What it covers | Open |
|---|---|---|
| [workshop_verification.ipynb](workshop_verification.ipynb) | (Newcomer to Advanced) The two-bus and four-bus hand calculations of theory Section 7 checked digit by digit against the solver, the old recursion without the conjugation for comparison, the PV kernels on the same network | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SOPTIM/AnalyticLoadFlow.jl/blob/main/notebooks/workshop_verification.ipynb) |
| [workshop_tour.ipynb](workshop_tour.ipynb) | (Newcomer to Advanced) A case from branch data, the series coefficients, Taylor vs Padé, the Padé-pole stability indicator, the three germ variants, PV buses and reactive limits, branch flows, the sparse path | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SOPTIM/AnalyticLoadFlow.jl/blob/main/notebooks/workshop_tour.ipynb) |
| [workshop_pst.ipynb](workshop_pst.ipynb) | (Advanced) Transformer branch model, why a PST breaks the flat germ, the two embeddings of theory Section 6.5 worked by hand on a 4-bus network, PST angle and ratio sweeps on the 9-bus ring, a regulated PST via the outer loop | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SOPTIM/AnalyticLoadFlow.jl/blob/main/notebooks/workshop_pst.ipynb) |
| [workshop_large_network.ipynb](workshop_large_network.ipynb) | (Advanced) PEGASE 2869 from a MATPOWER file: transformer convention detection (radians, sign, ratio), sparse solve and timing, deviation vs no-load embedding on a large network, reactive limits, branch flows | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SOPTIM/AnalyticLoadFlow.jl/blob/main/notebooks/workshop_large_network.ipynb) |

The same content is rendered in the Notebooks section of the documentation.

**Do not edit the `.ipynb` files directly**: they are generated. Edit the
Literate.jl source in [`docs/lit/`](../docs/lit/) and regenerate with
`julia --project=docs docs/generate_notebooks.jl`.
