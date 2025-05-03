import Hammer
import QuerySMT
import Mathlib.Data.Nat.Cast.Defs
import Mathlib.Algebra.Order.Hom.Monoid
import Mathlib.AlgebraicGeometry.EllipticCurve.Weierstrass
import Mathlib.Analysis.Analytic.Composition
import Mathlib.Analysis.Convex.Strict

set_option trace.auto.lamFOL2SMT true in
set_option trace.auto.mono.match true in
set_option trace.auto.mono.printLemmaInst true in
set_option trace.auto.mono true in
set_option trace.auto.mono.printResult true in
example (x y : Nat) (h : x = match y with | 0 => 0 | Nat.succ ys => ys) : x < y := by
  querySMT -- Investigating how lean-auto processes match statements
  sorry

--------------------------------------------------------------------------------------------------------
-- Original theorem from `Mathlib.Data.Nat.Cast.Defs`
theorem cast_one [AddMonoidWithOne R] : ((1 : ℕ) : R) = 1 := by
  rw [Nat.cast_succ, Nat.cast_zero, zero_add]

-- Hammer fails when given ground truth
set_option trace.auto.tptp.printQuery true in
theorem cast_one_test1 [AddMonoidWithOne R] : ((1 : ℕ) : R) = 1 := by
  hammer [Nat.cast_succ, Nat.cast_zero, zero_add] {simpTarget := no_target}

/-
With only `Mathlib.Data.Nat.Cast.Defs` imported:
hammer successfully translated the problem to TPTP, but the external prover was unable to solve it. Error: runAutoGetHints :: External TPTP solver unable to solve the goal
[auto.tptp.printQuery]
    thf(sortdecl_nat, type, s_nat: $tType).
    thf(sortdecl_int, type, s_int: $tType).
    thf(sortdecl_string, type, s_string: $tType).
    thf(sortdecl_empty, type, s_empty: $tType).
    thf(sortdecl_0, type, s_a0: $tType).

    thf(typedecl_icst_t_natVal_1, type, t_natVal_1: s_nat).
    thf(typedecl_icst_t_natVal_0, type, t_natVal_0: s_nat).
    thf(typedecl_icst_t_nadd, type, t_nadd: (s_nat > (s_nat > s_nat))).
    thf(typedecl_t_a3, type, t_a3: s_a0).
    thf(typedecl_t_a2, type, t_a2: (s_nat > s_a0)).
    thf(typedecl_t_a0, type, t_a0: (s_a0 > (s_a0 > s_a0))).

    thf(fact0, axiom, (! [X0 : s_a0] : (((^ [L : s_a0, R : s_a0] : L = R) @ ((t_a0 @ (t_a2 @ t_natVal_0)) @ X0)) @ X0))).
    thf(fact1, axiom, (! [X0 : s_nat] : (((^ [L : s_a0, R : s_a0] : L = R) @ (t_a2 @ ((t_nadd @ X0) @ t_natVal_1))) @ ((t_a0 @ (t_a2 @ X0)) @ t_a3)))).
    thf(fact2, axiom, ((~) @ (((^ [L : s_a0, R : s_a0] : L = R) @ (t_a2 @ t_natVal_1)) @ t_a3))).

With all imports:
hammer successfully translated the problem to TPTP and obtained an unsat core from an external prover, but was unable to reconstruct the proof. Error: getDuperCoreSMTLemmas :: Unable to use hints from external solver to reconstruct proof. Duper threw the following error:

Duper encountered a (deterministic) timeout. The maximum number of heartbeats 200000 has been reached (use 'set_option maxHeartbeats <num>' to set the limit)
[auto.tptp.printQuery]
    thf(sortdecl_nat, type, s_nat: $tType).
    thf(sortdecl_int, type, s_int: $tType).
    thf(sortdecl_string, type, s_string: $tType).
    thf(sortdecl_empty, type, s_empty: $tType).
    thf(sortdecl_0, type, s_a0: $tType).

    thf(typedecl_icst_t_natVal_1, type, t_natVal_1: s_nat).
    thf(typedecl_icst_t_natVal_0, type, t_natVal_0: s_nat).
    thf(typedecl_icst_t_nadd, type, t_nadd: (s_nat > (s_nat > s_nat))).
    thf(typedecl_t_a4, type, t_a4: s_a0).
    thf(typedecl_t_a3, type, t_a3: (s_nat > s_a0)).
    thf(typedecl_t_a2, type, t_a2: (s_nat > s_nat)).
    thf(typedecl_t_a0, type, t_a0: (s_a0 > (s_a0 > s_a0))).

    thf(fact0, axiom, (! [X0 : s_nat] : (((^ [L : s_nat, R : s_nat] : L = R) @ ((t_nadd @ t_natVal_0) @ X0)) @ X0))).
    thf(fact1, axiom, (! [X0 : s_a0] : (((^ [L : s_a0, R : s_a0] : L = R) @ ((t_a0 @ (t_a3 @ t_natVal_0)) @ X0)) @ X0))).
    thf(fact2, axiom, (((^ [L : s_nat, R : s_nat] : L = R) @ (t_a2 @ t_natVal_0)) @ t_natVal_0)).
    thf(fact3, axiom, (! [X0 : s_nat] : (((^ [L : s_nat, R : s_nat] : L = R) @ (t_a2 @ ((t_nadd @ X0) @ t_natVal_1))) @ ((t_nadd @ (t_a2 @ X0)) @ t_natVal_1)))).
    thf(fact4, axiom, (! [X0 : s_nat] : (((^ [L : s_a0, R : s_a0] : L = R) @ (t_a3 @ ((t_nadd @ X0) @ t_natVal_1))) @ ((t_a0 @ (t_a3 @ X0)) @ t_a4)))).
    thf(fact5, axiom, ((~) @ (((^ [L : s_a0, R : s_a0] : L = R) @ (t_a3 @ t_natVal_1)) @ t_a4))).

-/

def one_eq_succ_zero : 1 = Nat.succ 0 := by rfl

-- External prover succeeds when given the additional fact: `1 = 0.succ` which is used implicitly
-- in the originally proof (Duper fails to reconstruct the proof, but that's an unrelated issue)
theorem cast_one_test2 [AddMonoidWithOne R] : ((1 : ℕ) : R) = 1 := by
  hammer [one_eq_succ_zero, Nat.cast_succ, Nat.cast_zero, zero_add] {simpTarget := no_target}
  -- Note: `hammer` behaves differently depending on set of imports!!!

#check zero_add
def my_zero_add [AddMonoidWithOne R] (a : R) : 0 + a = a := zero_add a

-- Note: `hammer` genuinely succeeds in finding a proof, but the suggestion it creates fails.
-- Need to figure out why (but that's independent of the premise selection work)
theorem cast_one_test3 [AddMonoidWithOne R] : ((1 : ℕ) : R) = 1 := by
  sorry -- hammer [one_eq_succ_zero, Nat.cast_succ, Nat.cast_zero, my_zero_add] {simpTarget := no_target}
  -- Additional note: This `hammer` call succeeds when the only imports are `Hammer`,
  -- `Mathlib.Data.Nat.Cast.Defs`, and `Mathlib.Algebra.Order.Hom.Monoid`. After
  -- `Mathlib.AlgebraicGeometry.EllipticCurve.Weierstrass` is added, `hammer` starts to fail

--------------------------------------------------------------------------------------------------------
namespace OrderMonoidWithZeroHom

variable [Preorder α] [Preorder β] [Preorder γ] [Preorder δ] [MulZeroOneClass α] [MulZeroOneClass β]
  [MulZeroOneClass γ] [MulZeroOneClass δ] {f g : α →*₀o β}

-- Original theorem from `Mathlib.Algebra.Order.Hom.Monoid`
theorem toOrderMonoidHom_injective_original : Function.Injective (toOrderMonoidHom : _ → α →*o β) := fun f g h =>
  OrderMonoidWithZeroHom.ext <| by convert DFunLike.ext_iff.1 h using 0

-- Note `convert ... using 0` can be replaced by `apply ...`
theorem toOrderMonoidHom_injective_apply : Function.Injective (toOrderMonoidHom : _ → α →*o β) := fun f g h =>
  OrderMonoidWithZeroHom.ext <| by apply DFunLike.ext_iff.1 h

-- Note that taking the suggestion from `show_term apply ...` fails
theorem toOrderMonoidHom_injective_exact : Function.Injective (toOrderMonoidHom : _ → α →*o β) := fun f g h =>
  OrderMonoidWithZeroHom.ext <| by
  sorry -- exact DFunLike.ext_iff.mp h -- This is what `show_term apply DFunLike.ext_iff.1 h` suggests

-- Hammer fails when given ground truth
theorem toOrderMonoidHom_injective_test1 : Function.Injective (toOrderMonoidHom : _ → α →*o β) := fun f g h =>
  OrderMonoidWithZeroHom.ext <| by
  sorry -- hammer [*, DFunLike.ext_iff] {simpTarget := no_target}

-- Could only get Hammer to succeed with sufficient massaging
theorem toOrderMonoidHom_injective_test2 : Function.Injective (toOrderMonoidHom : _ → α →*o β) := fun f g h =>
  OrderMonoidWithZeroHom.ext <| by
  unfold DFunLike.coe instFunLike toMonoidWithZeroHom ZeroHom.toFun
  unfold toOrderMonoidHom at h
  simp only [OrderMonoidHom.mk.injEq, MonoidHom.mk.injEq, OneHom.mk.injEq] at h
  hammer [*, DFunLike.ext_iff] {simpTarget := no_target} -- Note `DFunLike.ext_iff` isn't used

end OrderMonoidWithZeroHom

--------------------------------------------------------------------------------------------------------
namespace WeierstrassCurve

variable {R : Type u} [CommRing R] (W : WeierstrassCurve R)

-- Original theorem from `Mathlib.AlgebraicGeometry.EllipticCurve.Weierstrass`
lemma ofJ1728_c₄_original : (ofJ1728 R).c₄ = -48 := by
  rw [ofJ1728, c₄, b₂, b₄]
  norm_num1 -- We don't expect hammer to get anything that invokes `norm_num`

end WeierstrassCurve

--------------------------------------------------------------------------------------------------------
namespace Composition

#check length_gather

-- Original theorem from `Mathlib.Analysis.Analytic.Composition`
theorem length_gather_original (a : Composition n) (b : Composition a.length) :
    length (a.gather b) = b.length :=
  show (List.map List.sum (a.blocks.splitWrtComposition b)).length = b.blocks.length by
    rw [List.length_map, List.length_splitWrtComposition]

#print Composition.gather

-- Hammer can solve this with ground truth (but evaluation said it failed)
-- Evaluation says it fails because `show` is doing a fair amount of work
theorem length_gather_test1 (a : Composition n) (b : Composition a.length) :
    length (a.gather b) = b.length :=
  show (List.map List.sum (a.blocks.splitWrtComposition b)).length = b.blocks.length by
    hammer [*, List.length_map, List.length_splitWrtComposition] {simpTarget := no_target}

#print Composition

theorem length_gather_test2 (a : Composition n) (b : Composition a.length) :
  length (a.gather b) = b.length := by
  unfold Composition.gather
  hammer [*, List.length_map, List.length_splitWrtComposition] {simpTarget := no_target}

theorem length_gather_test3 (a : Composition n) (b : Composition a.length) :
  length (a.gather b) = b.length := by -- Original recommendation
  hammer [*, List.length_map, List.length_splitWrtComposition] {simpTarget := no_target}

theorem length_gather_test4 (a : Composition n) (b : Composition a.length) :
  length (a.gather b) = b.length := by -- Original recommendation + `Composition.gather`
  hammer [*, Composition.gather, List.length_map, List.length_splitWrtComposition] {simpTarget := no_target}
  -- Fails to process with `Composition.gather`. But `length_gather_test2` shows that unfolding `Composition.gather`
  -- should be sufficient, so look into improving this situation.

end Composition
--------------------------------------------------------------------------------------------------------
section

variable [OrderedRing 𝕜] [TopologicalSpace E] [TopologicalSpace F]
variable [AddCommGroup E] [AddCommGroup F] [Module 𝕜 E] [Module 𝕜 F] {s t : Set E} {x y : E}

theorem StrictConvex.affine_preimage_original {s : Set F} (hs : StrictConvex 𝕜 s) {f : E →ᵃ[𝕜] F}
    (hf : Continuous f) (hfinj : Function.Injective f) : StrictConvex 𝕜 (f ⁻¹' s) := by
  intro x hx y hy hxy a b ha hb hab
  refine' preimage_interior_subset_interior_preimage hf _
  rw [Set.mem_preimage, Convex.combo_affine_apply hab]
  exact hs hx hy (hfinj.ne hxy) ha hb hab

-- Hammer fails on ground truth (I expect at least part of the issue has to do with unfolding `StrictConvex`,
-- but this example is complicated enough that I wasn't able to determine if there was a set of constants
-- that could be unfolded to make Hammer succeed)
theorem StrictConvex.affine_preimage_test1 {s : Set F} (hs : StrictConvex 𝕜 s) {f : E →ᵃ[𝕜] F}
  (hf : Continuous f) (hfinj : Function.Injective f) : StrictConvex 𝕜 (f ⁻¹' s) := by
  sorry
  -- hammer [*, Function.Injective.ne, preimage_interior_subset_interior_preimage,
  --   Set.mem_preimage, Convex.combo_affine_apply] {simpTarget := no_target}

end

--------------------------------------------------------------------------------------------------------

-- Using `Unitization.ind` from `Mathlib.Algebra.Algebra.Unitization` creates problems because the output type
-- is dependent on one of the inputs
/-
@[elab_as_elim, induction_eliminator, cases_eliminator]
theorem ind {R A} [AddZeroClass R] [AddZeroClass A] {P : Unitization R A → Prop}
    (inl_add_inr : ∀ (r : R) (a : A), P (inl r + (a : Unitization R A))) (x) : P x :=
  inl_fst_add_inr_snd_eq x ▸ inl_add_inr x.1 x.2
-/

--------------------------------------------------------------------------------------------------------

-- Using `NonUnitalStarAlgHomClass.toStarHomClass` and `NonnegSpectrumClass.quasispectrum_nonneg_of_nonneg` in
-- `Mathlib.Algebra.Algebra.Quasispectrum` creates similar problems also because the output type is dependent
-- on one of the inputs
