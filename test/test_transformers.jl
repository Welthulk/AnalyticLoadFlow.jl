# =============================================================================
# File: test/test_transformers.jl
# Date: 2026-09-17
# Author: Udo Schmitz
# Organization: SOPTIM AG
# Purpose: Tests for the no-load germ, the reflected-reciprocal PQ recursion,
#          the PV kernel sign consistency, transformer/PST branches, the
#          regulated phase shifter loop and the MATPOWER importer.
#
# Copyright 2026 SOPTIM AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

using Test
using LinearAlgebra
using SparseArrays
using AnalyticLoadFlow

const A = AnalyticLoadFlow

function _series_3bus()
   Y = build_ybus(3, [pi_branch(1, 2; r = 0.02, x = 0.10), pi_branch(2, 3; r = 0.03, x = 0.15), pi_branch(1, 3; r = 0.01, x = 0.08)])
   return Y
end

@testset "PQ recursion uses the reflected reciprocal (pure APSLF is exact)" begin
   Y = _series_3bus()
   S = ComplexF64[0, -0.8 - 0.3im, -0.6 - 0.2im]
   for germ in (:flat, :noload)
      V, Vc, Wc = A.apslf_pq(Y, S; slack = 1, order = 30, use_pade = true, germ = germ)
      Sinj = A.calc_injections(Y, V)
      @test maximum(abs.(Sinj[2:3] .- S[2:3])) < 1e-12
   end
   # workspace path
   ws = A.build_apslf_pq_workspace(Y; slack = 1, order = 30, germ = :noload)
   V, _, _ = A.apslf_pq_solve!(ws, S; germ = :noload)
   @test maximum(abs.(A.calc_injections(Y, V)[2:3] .- S[2:3])) < 1e-12
end

@testset "direct PV kernels are exact without NR polish (dense and sparse, flat and no-load germ)" begin
   Y = _series_3bus()
   bt = [:slack, :pv, :pq]
   P = [0.0, 0.5, -0.6]
   Q = [0.0, 0.0, -0.2]
   for (Vsl, germ) in ((1.0, :flat), (1.0, :noload), (1.05, :noload)), kern in (A.apslf_pf_pv_direct, A.apslf_pf_pv_direct_sparse)
      Vm = [Vsl, 1.02, 1.0]
      V, Qpv, _, _, _ = kern(Y, bt, P, Q, Vm; slack = 1, Vslack = ComplexF64(Vsl, 0), order = 30, germ = germ, self_check = false)
      Sinj = A.calc_injections(Y, V)
      @test maximum(abs.(real.(Sinj[2:3]) .- P[2:3])) < 1e-12
      @test abs(imag(Sinj[3]) - Q[3]) < 1e-12
      @test abs(abs(V[2]) - 1.02) < 1e-12
      @test abs(Qpv[1] - imag(Sinj[2])) < 1e-10   # internal Q convention equals the physical injection
   end
end

@testset "no-load germ makes the 9-bus case (shunts, Vslack≠1) exact without NR polish" begin
   case = A.demo_case_9bus()
   for Ymat in (case.Y, sparse(case.Y)), inner in (:pq, :direct_pv, :direct_pv_sparse)
      res = A.solve_pf_apslf_with_pv_q_limits(
         Ymat, case.bustype, case.Pspec, case.Qspec, case.Vm, case.Qmin, case.Qmax;
         slack = 1, Vslack = ComplexF64(case.Vm[1], 0), inner = inner, order = 40, use_pade = true,
         nr_polish = false, germ = :noload, max_outer = 30, use_sparse = issparse(Ymat),
      )
      maxP, maxQ = A.compute_demo_mismatch(case, res)
      @test res.converged
      @test max(maxP, maxQ) < 1e-8
   end
   # flat germ is not exact here (non-zero row sums, Vslack = 1.04)
   res_flat = A.solve_pf_apslf_with_pv_q_limits(
      case.Y, case.bustype, case.Pspec, case.Qspec, case.Vm, case.Qmin, case.Qmax;
      slack = 1, Vslack = ComplexF64(case.Vm[1], 0), inner = :direct_pv, order = 40, nr_polish = false, germ = :flat, max_outer = 5,
   )
   maxP_flat, _ = A.compute_demo_mismatch(case, res_flat)
   @test maxP_flat > 1e-3
   @test_throws ArgumentError A.apslf_germ(case.Y, [2, 3], 1, 1.0 + 0im, :nonsense)
end

@testset "branch builders and Y-bus stamping" begin
   br = demo_9bus_branches()
   @test length(br) == 9
   Y = build_ybus(9, br)
   @test maximum(abs.(Y .- A.demo_case_9bus().Y)) < 1e-14
   Ys = build_ybus(9, br; sparse_output = true)
   @test Ys isa SparseMatrixCSC{ComplexF64,Int}
   @test maximum(abs.(Matrix(Ys) .- Y)) < 1e-14
   Y0 = build_ybus(9, br; nominal = true)
   @test maximum(abs.(sum(Y0, dims = 2))) < 1e-12        # zero row sums
   # transformer with ratio and shift: non-symmetric, theory Section 6.5 entries
   t = 0.98 * cis(deg2rad(7.0))
   y = inv(complex(0.01, 0.1))
   tb = transformer_branch(1, 2; r = 0.01, x = 0.1, ratio = 0.98, shift_deg = 7.0)
   yii, yij, yji, yjj = A.branch_admittances(tb)
   @test yii ≈ y / abs2(t)
   @test yij ≈ -y / conj(t)
   @test yji ≈ -y / t
   @test yjj ≈ y
   Yt = build_ybus(2, [tb])
   @test Yt[1, 2] != Yt[2, 1]
   @test A.is_phase_shifter(tb) && A.is_transformer(tb) && !A.is_transformer(pi_branch(1, 2; r = 0.1, x = 0.2))
   sh = build_ybus(2, [tb]; bus_shunts = [(bus = 2, g = 0.0, b = 0.05)])
   @test sh[2, 2] ≈ Yt[2, 2] + 0.05im
   @test_throws ArgumentError build_ybus(2, [pi_branch(1, 3; r = 0.1, x = 0.2)])
end

@testset "PST case: exact solution, flows and angle sweep" begin
   c0 = demo_case_9bus_pst()
   @test maximum(abs.(c0.Y .- A.demo_case_9bus().Y)) < 1e-14
   P45 = Float64[]
   for φ in (-10.0, 0.0, 10.0)
      c = demo_case_9bus_pst(shift_deg = φ, enforce_q_limits = false, sparse_output = true)
      res = solve_pf_apslf(c; mode = :direct, order = 40, nr_polish = false)
      maxP, maxQ = A.compute_demo_mismatch(c, res)
      @test res.converged
      @test max(maxP, maxQ) < 1e-8
      flows = branch_flows(res.V, c.branches; baseMVA = c.baseMVA)
      @test length(flows) == 9
      # power balance: generation - load = losses
      Sinj = A.calc_injections(c.Y, res.V)
      @test abs(real(sum(Sinj)) - sum(f.Ploss_MW for f in flows) / c.baseMVA) < 1e-8
      push!(P45, branch_active_power(res.V, c.branches, 4, 5))
      @test branch_active_power(res.V, c.branches, 5, 4) ≈ real(flows[2].Sji_pu)
   end
   @test P45[1] > P45[2] > P45[3]   # negative shift on the 4-side pushes more power 4→5
   @test_throws ArgumentError branch_active_power(ones(ComplexF64, 9), c0.branches, 1, 9)
end

@testset "regulated PST outer loop" begin
   reg = solve_pf_pst_regulated(φ -> demo_case_9bus_pst(shift_deg = φ, enforce_q_limits = false), 4, 5, 0.6; order = 40, nr_polish = false)
   @test reg.converged
   @test abs(reg.P_pu - 0.6) < 1e-5
   @test reg.iterations <= 8
   @test reg.case.pst.shift_deg == reg.shift_deg
   # angle limit pins the result
   pinned = solve_pf_pst_regulated(φ -> demo_case_9bus_pst(shift_deg = φ, enforce_q_limits = false), 4, 5, 0.6; shift_min = -3.0, shift_max = 3.0, order = 40, nr_polish = false)
   @test !pinned.converged
   @test pinned.shift_deg == -3.0
end

@testset "MATPOWER importer with convention detection" begin
   # 4-bus case written in MATPOWER syntax: bus 2 fed through a PST (angle in radians, sign flipped),
   # bus 4 through an OLTC with the inverse ratio. The stored solution is produced by APSLF itself.
   base_branches = [
      pi_branch(1, 2; r = 0.01, x = 0.10, b = 0.02),
      transformer_branch(1, 3; r = 0.005, x = 0.05, shift_deg = 5.0),
      pi_branch(2, 3; r = 0.02, x = 0.15, b = 0.03),
      transformer_branch(3, 4; r = 0.0, x = 0.08, ratio = 0.95),
   ]
   shunts = [(bus = 2, g = 0.0, b = 0.05)]
   Y = build_ybus(4, base_branches; bus_shunts = shunts)
   spec = (Y = Y, bustype = [:slack, :pq, :pv, :pq], Pspec = [0.0, -0.5, 0.4, -0.3], Qspec = [0.0, -0.2, 0.0, -0.1],
      Vm = [1.02, 1.0, 1.01, 1.0], Qmin = [-1e9, -1e9, -1.0, -1e9], Qmax = [1e9, 1e9, 1.0, 1e9], slack = 1)
   res = solve_pf_apslf(spec; order = 40, nr_polish = true)
   @test res.converged
   Sinj = A.calc_injections(Y, res.V)
   path = joinpath(mktempdir(), "case4test.m")
   open(path, "w") do io
      println(io, "function mpc = case4test")
      println(io, "%   comment line ; with ] brackets")
      println(io, "mpc.version = '2';")
      println(io, "mpc.baseMVA = 100;")
      println(io, "%\tbus_i\ttype\tPd\tQd\tGs\tBs\tarea\tVm\tVa\tbaseKV\tzone\tVmax\tVmin")
      println(io, "mpc.bus = [")
      types = [3, 1, 2, 1, 4]
      for k = 1:4
         Pd = k == 3 ? 0.0 : -spec.Pspec[k] * 100
         Qd = k == 3 ? 0.0 : -spec.Qspec[k] * 100
         Bs = k == 2 ? 5.0 : 0.0
         println(io, "\t$(10k)\t$(types[k])\t$(Pd)\t$(Qd)\t0\t$(Bs)\t1\t$(abs(res.V[k]))\t$(rad2deg(angle(res.V[k])))\t110\t1\t1.1\t0.9;")
      end
      println(io, "\t99\t4\t0\t0\t0\t0\t1\t1\t0\t110\t1\t1.1\t0.9;   % isolated bus, dropped")
      println(io, "];")
      println(io, "mpc.gen = [")
      println(io, "\t10\t0\t0\t999\t-999\t1.02\t100\t1\t999\t-999\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;")
      println(io, "\t30\t40\t0\t100\t-100\t1.01\t100\t1\t999\t-999\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;")
      println(io, "\t30\t0\t0\t0\t0\t1.01\t100\t0\t999\t-999\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;   % out of service")
      println(io, "];")
      println(io, "mpc.branch = [")
      println(io, "\t10\t20\t0.01\t0.10\t0.02\t0\t0\t0\t0\t0\t1\t-360\t360;")
      println(io, "\t10\t30\t0.005\t0.05\t0\t0\t0\t0\t0\t$(-deg2rad(5.0))\t1\t-360\t360;")
      println(io, "\t20\t30\t0.02\t0.15\t0.03\t0\t0\t0\t0\t0\t1\t-360\t360;")
      println(io, "\t30\t40\t0\t0.08\t0\t0\t0\t0\t$(1/0.95)\t0\t1\t-360\t360;")
      println(io, "\t30\t40\t0\t0.08\t0\t0\t0\t0\t1\t0\t0\t-360\t360;   % out of service")
      println(io, "\t30\t99\t0\t0.08\t0\t0\t0\t0\t1\t0\t1\t-360\t360;   % to isolated bus, dropped")
      println(io, "];")
   end
   mp = parse_matpower_m(path)
   @test mp.baseMVA == 100.0
   @test size(mp.bus) == (5, 13) && size(mp.gen) == (3, 21) && size(mp.branch) == (6, 13)
   c = matpower_case(path)
   @test size(c.Y) == (4, 4) && issparse(c.Y)
   @test c.bustype == [:slack, :pq, :pv, :pq]
   @test c.slack == 1
   @test c.labels == ["10", "20", "30", "40"]
   @test length(c.branches) == 4
   @test c.conventions.angle_unit == :rad
   @test c.conventions.angle_sign == -1
   @test c.conventions.ratio_convention == :inverse
   @test c.conventions.ref_mismatch_pu < 1e-6
   @test length(c.conventions.trials) == 8
   @test c.Pspec ≈ spec.Pspec && c.Qspec ≈ spec.Qspec
   @test c.Qmin[3] ≈ -1.0 && c.Qmax[3] ≈ 1.0
   @test c.Vm[3] ≈ 1.01 && c.Vm[1] ≈ 1.02
   @test c.bus_shunts == [(bus = 2, g = 0.0, b = 0.05)]
   @test maximum(abs.(Matrix(c.Y) .- Y)) < 1e-12
   res2 = solve_pf_apslf(c; order = 40, nr_polish = false)
   @test res2.converged
   @test maximum(abs.(res2.V .- res.V)) < 1e-6
   # forced (wrong) convention is honored
   c_deg = matpower_case(path; angle_unit = :deg, angle_sign = 1, ratio_convention = :matpower)
   @test c_deg.conventions.angle_unit == :deg
   @test c_deg.conventions.ref_mismatch_pu > 1e-3
   @test_throws ArgumentError matpower_case(path; angle_unit = :grad)
   @test_throws ArgumentError parse_matpower_m(joinpath(dirname(path), "missing.m"))
end

@testset "deviation embedding (default germ) is exact in all kernels" begin
   case = A.demo_case_9bus()
   for Ymat in (case.Y, sparse(case.Y)), inner in (:pq, :direct_pv, :direct_pv_sparse)
      res = A.solve_pf_apslf_with_pv_q_limits(
         Ymat, case.bustype, case.Pspec, case.Qspec, case.Vm, case.Qmin, case.Qmax;
         slack = 1, Vslack = ComplexF64(case.Vm[1], 0), inner = inner, order = 40, nr_polish = false, max_outer = 30, use_sparse = issparse(Ymat),
      )
      @test res.converged
      @test max(A.compute_demo_mismatch(case, res)...) < 1e-8
      @test res.apslf_germ == :deviation
   end
   Y = _series_3bus()
   @test A.apslf_germ(Y, [2, 3], 1, 1.05 + 0im, :deviation) == fill(1.05 + 0im, 2)
   @test A.apslf_row_sums(Y) ≈ vec(sum(Y, dims = 2))
   ws = A.build_apslf_pq_workspace(Y; slack = 1, order = 20, germ = :deviation)
   @test_throws ArgumentError A.apslf_pq_solve!(ws, ComplexF64[0, -0.1, -0.1]; germ = :noload)
   c = demo_case_9bus_pst(shift_deg = 8.0, enforce_q_limits = false, sparse_output = true)
   res = solve_pf_apslf(c; order = 40, nr_polish = false)
   @test res.converged && max(A.compute_demo_mismatch(c, res)...) < 1e-10
   res_nl = solve_pf_apslf(c; order = 40, nr_polish = false, germ = :noload)
   @test maximum(abs.(res.V .- res_nl.V)) < 1e-8   # both variants describe the same function at s = 1
end

const _PEGASE_1354 = normpath(joinpath(@__DIR__, "..", "data", "_downloaded", "case1354pegase.m"))
if isfile(_PEGASE_1354)
   @testset "PEGASE 1354 (local download): conventions and pure APSLF solution" begin
      c = matpower_case(_PEGASE_1354)
      @test size(c.Y) == (1354, 1354)
      @test c.conventions.angle_unit == :rad
      @test c.conventions.angle_sign == -1
      @test c.conventions.ratio_convention == :matpower
      res = solve_pf_apslf(c; order = 40, nr_polish = false, enforce_q_limits = false)
      @test res.converged
      @test res.effective_mode == :direct
      @test max(A.compute_demo_mismatch(c, res)...) < 1e-8
      @test maximum(abs.(res.V .- c.V_ref)) < 2e-2
   end
else
   @info "PEGASE test skipped (no data/_downloaded/case1354pegase.m); run examples/pegase_matpower_demo.jl to download it"
end
