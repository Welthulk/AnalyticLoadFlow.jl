# Copyright 2026 SOPTIM AG                                                    #src
#                                                                             #src
# Licensed under the Apache License, Version 2.0 (the "License");             #src
# you may not use this file except in compliance with the License.            #src
# You may obtain a copy of the License at                                     #src
#                                                                             #src
#     https://www.apache.org/licenses/LICENSE-2.0                             #src
#                                                                             #src
# Unless required by applicable law or agreed to in writing, software         #src
# distributed under the License is distributed on an "AS IS" BASIS,           #src
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #src
# See the License for the specific language governing permissions and         #src
# limitations under the License.                                              #src
#                                                                             #src
# file: docs/lit/workshop_tour.jl                                             #src
# purpose: Literate.jl source of the APSLF tour notebook and its Documenter   #src
#          page. Regenerate with `julia --project=docs docs/generate_notebooks.jl`. #src

# # APSLF tour: from a Y-bus to a power series
#
# > **Level: Newcomer to Advanced.**
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SOPTIM/AnalyticLoadFlow.jl/blob/main/notebooks/workshop_tour.ipynb)
#
# > **Note:** This workshop was created with AI assistance and is reviewed
# > and curated by the maintainer; it is not a fully machine-generated text.
#
# The Analytic Power Series Load Flow (APSLF) does not iterate. It embeds the
# load-flow problem into a family of problems with a parameter $s$, computes
# the bus voltages as a power series in $s$, and evaluates that series at
# $s = 1$ with a Padé approximant. This notebook walks through the pieces of
# [AnalyticLoadFlow.jl](https://github.com/SOPTIM/AnalyticLoadFlow.jl): the
# Y-bus data contract, the series coefficients, the choice of the germ
# (the order-0 state), PV buses with reactive limits, and the sparse path for
# larger networks. The theory article in the documentation is referenced by
# section number.
#
# > **Note:** On Google Colab the install cell takes a few minutes on a
# > fresh session. This notebook targets Julia ≥ 1.12.

#nb # ## Setup (Colab)
#nb # This cell installs AnalyticLoadFlow from GitHub (branch `main`) into a
#nb # fresh temporary environment. Run it first, once per session.
#nb using Pkg
#nb Pkg.activate(temp = true)
#nb Pkg.add(url = "https://github.com/SOPTIM/AnalyticLoadFlow.jl", rev = "main")
#nb # For the latest registered release use: Pkg.add("AnalyticLoadFlow")

# ## Warm-up and shared helpers
#
# Julia compiles each function on first use; this cell loads the package and
# defines two small helpers used throughout.

using AnalyticLoadFlow
using LinearAlgebra
using SparseArrays
using Printf
const A = AnalyticLoadFlow

## largest active/reactive mismatch of a solution on the physical Y-bus
mismatch(case, res) = maximum(A.compute_demo_mismatch(case, res))

## compact voltage table
function show_voltages(case, V; rows = 12)
   for i = 1:min(rows, length(V))
      @printf("%-6s %-5s |V| = %.5f pu   angle = %8.4f°\n", case.labels[i], case.bustype[i], abs(V[i]), rad2deg(angle(V[i])))
   end
   length(V) > rows && println("... $(length(V) - rows) more buses")
end
solve_pf_apslf(A.demo_case_9bus(); order = 20)   # warm-up
println("ready")

# ## 1. A case from branch data
#
# The solver works on a bus admittance matrix `Y` and a small NamedTuple data
# contract: `bustype` (`:slack`, `:pv`, `:pq`), `Pspec`/`Qspec` injections in
# pu (load negative), `Vm` voltage setpoints, `Qmin`/`Qmax` for PV buses and
# the `slack` index. Branches are stamped with [`build_ybus`](@ref): π-lines
# via [`pi_branch`](@ref), transformers via [`transformer_branch`](@ref).
# The 9-bus teaching case ships with the package.

branches = demo_9bus_branches()
Y = build_ybus(9, branches)
case = A.demo_case_9bus()
println("Y is $(size(Y, 1))×$(size(Y, 2)), symmetric: ", Y ≈ transpose(Y))
println("bus types: ", join(string.(case.bustype), " "))

# The solver returns the complex voltages, the final bus types (PV buses may
# have been switched to PQ), the reactive injections and diagnostics.

res = solve_pf_apslf(case; mode = :direct, order = 40, use_pade = true, nr_polish = false)
println("converged = $(res.converged), effective mode = $(res.effective_mode), outer iterations = $(res.outer_iters)")
@printf("max mismatch on the physical Y-bus: %.2e pu (no Newton polish involved)\n", mismatch(case, res))
show_voltages(case, res.V)

# ## 2. The series behind the numbers
#
# APSLF scales every injection with $s$: at $s=0$ nothing is injected, at
# $s=1$ the specified loads and generation act (theory Section 2.2). The
# voltages are power series $V_i(s) = \sum_n V_i^{(n)} s^n$; the coefficients
# come from one linear solve per order with a constant matrix (Section 4).
# With `return_coeffs = true` the coefficient matrix is returned.

res = solve_pf_apslf(case; order = 40, nr_polish = false, return_coeffs = true)
Vc = res.Vcoeff
println("coefficient matrix: ", size(Vc), " (bus × order+1)")
for n in (0, 1, 2, 5, 10, 20, 40)
   @printf("n = %2d   max_i |V_i^(n)| = %.2e\n", n, maximum(abs.(Vc[:, n+1])))
end

# The coefficients decay geometrically, so a direct Taylor sum would already
# converge here. Padé approximants (Section 5.2) turn the polynomial into a
# rational function and extend the reach of the series to cases where the
# Taylor sum does not converge at $s=1$. Both evaluations are available via
# [`evaluate_series`](@ref); the difference is negligible for a well-behaved case.

c5 = Vc[5, :]
taylor = A.evaluate_series(c5, A.APSLFEvaluationOptions(mode = :taylor)).voltage
pade = A.evaluate_series(c5, A.APSLFEvaluationOptions(mode = :pade)).voltage
@printf("bus 5:  Taylor %.8f ∠ %.5f°   Padé %.8f ∠ %.5f°   |Δ| = %.1e\n", abs(taylor), rad2deg(angle(taylor)), abs(pade), rad2deg(angle(pade)), abs(taylor - pade))

# The poles of the Padé denominator are a heuristic distance-to-collapse
# indicator (Section 5.3): a pole close to $s = 1$ means the network is near
# its loadability limit. `st_level` maps the distance to a traffic-light label.

st = A.stability_from_Vcoeff(Vc; slack = case.slack, order = 40)
@printf("nearest pole to s = 1: distance %.3f at bus %d, level %s\n", st.dmin, st.bus, A.st_level(st.dmin))

# Scaling all injections up moves the pole towards $s = 1$:

for factor in (1.0, 1.5, 2.0, 2.5)
   heavy = merge(case, (Pspec = factor .* case.Pspec, Qspec = factor .* case.Qspec, Qmin = fill(-1e9, 9), Qmax = fill(1e9, 9)))
   r = solve_pf_apslf(heavy; order = 40, nr_polish = false, return_coeffs = true)
   s = A.stability_from_Vcoeff(r.Vcoeff; slack = 1, order = 40)
   @printf("load × %.1f: converged = %-5s  min |V| = %.4f  pole distance = %.3f  %s\n", factor, r.converged, minimum(abs.(r.V)), s.dmin, A.st_level(s.dmin))
end

# ## 3. The germ: why the order-0 state matters
#
# At $s = 0$ the recursion needs an exact solution, the *germ*. The flat germ
# $V^{(0)} = 1$ is exact only when the constant matrix has zero row sums
# (Section 2.2): all buses at the same voltage, no current anywhere. Line
# charging, bus shunts, transformer taps and a slack voltage other than
# 1 pu break this property. The row sums of the 9-bus matrix show it:

rs = A.apslf_row_sums(Y)
for i in (1, 4, 5)
   @printf("row sum at bus %d: %s\n", i, rs[i] == 0 ? "0" : @sprintf("%.4f%+.4fim", real(rs[i]), imag(rs[i])))
end

# Theory Section 6.5 gives two consistent ways out, both implemented through
# the `germ` keyword:
#
# * `:deviation` (default): keep the flat germ and embed the deviation
#   $Y - Y_0$ with $s$, where $Y_0 = Y - \mathrm{diag}(Y\mathbf{1})$ has zero
#   row sums by construction.
# * `:noload`: keep the full $Y$ and use the linear no-load solution as germ.
# * `:flat`: the legacy flat germ on the full $Y$. Not exact; it needs the
#   Newton polish to arrive at a solution.
#
# Both exact variants describe the same function at $s = 1$ but follow
# different paths in $s$ and therefore have different convergence radii.

for germ in (:deviation, :noload, :flat)
   r = solve_pf_apslf(case; order = 40, nr_polish = false, germ = germ)
   @printf("germ = %-10s converged = %-5s  max mismatch = %.2e pu\n", germ, r.converged, mismatch(case, r))
end

# The optional Newton polish (Section 6.4) starts from the APSLF result and is
# a safety net, not part of the method. With an exact embedding it has nothing
# left to do:

r = solve_pf_apslf(case; order = 40, nr_polish = true)
@printf("with polish: score before %.1e, after %.1e, improved = %s\n", r.nr_polish_score_before, r.nr_polish_score_after, r.nr_polish_improved)

# ## 4. PV buses and reactive limits
#
# PV buses can be handled in two ways (Section 6): an outer loop that adjusts
# the reactive injection of each PV bus until its voltage magnitude is met
# (`mode = :outer`, PQ-only inner solves), or the direct formulation with one
# augmented real linear system per order (`mode = :direct`). Reactive limits
# are enforced by switching PV→PQ at the violated limit and re-solving.

for mode in (:direct, :outer)
   r = solve_pf_apslf(case; mode = mode, order = 40, nr_polish = false)
   sw = get(r, :switch_log, ())
   @printf("mode = %-7s converged = %-5s outer iterations = %d  PV→PQ switches = %d  Q(bus 2) = %.4f pu\n", mode, r.converged, r.outer_iters, length(sw), r.Q[2])
   for e in sw
      @printf("   outer %d: bus %d hit its %s limit (Q = %.4f pu)\n", e.outer, e.bus, e.side, e.qinj)
   end
end

# Without limit enforcement the generator at bus 3 would run outside its band:

r = solve_pf_apslf(case; order = 40, nr_polish = false, enforce_q_limits = false)
@printf("enforce_q_limits = false: Q(bus 3) = %.4f pu, band [%.2f, %.2f]\n", r.Q[3], case.Qmin[3], case.Qmax[3])

# ## 5. Branch flows and losses
#
# [`branch_flows`](@ref) evaluates every branch (lines and transformers) from
# the voltage vector; [`print_branch_flows`](@ref) prints a table.

r = solve_pf_apslf(case; order = 40, nr_polish = false)
flows = branch_flows(r.V, branches; baseMVA = case.baseMVA)
print_branch_flows(flows; labels = case.labels)
@printf("total losses: %.3f MW\n", sum(f.Ploss_MW for f in flows))

# ## 6. Sparse matrices
#
# For larger networks pass a sparse `Y`. `solve_pf_apslf` then selects the
# sparse direct PV kernel automatically (`use_sparse = :auto` switches at
# 110 buses or when `Y` is already sparse); the recursion, the germ and the
# Padé evaluation are identical. The synthetic 118-bus case illustrates it.

case118 = A.demo_case_118bus_synthetic()
Ys = sparse(case118.Y)
sparse_case = merge(case118, (Y = Ys,))
r = solve_pf_apslf(sparse_case; order = 40, nr_polish = false)
t = @elapsed r = solve_pf_apslf(sparse_case; order = 40, nr_polish = false)
@printf("118 buses, nnz(Y) = %d: converged = %s, mode = %s, max mismatch = %.1e pu, %.3f s\n", nnz(Ys), r.converged, r.effective_mode, mismatch(case118, r), t)

# ## Where to go next
#
# * `workshop_pst.ipynb`: transformers and phase shifters, the two embeddings
#   worked by hand, angle sweeps and a regulated PST.
# * `workshop_large_network.ipynb`: a 2869-bus PEGASE case from a MATPOWER
#   file, convention detection, timing and diagnostics.
