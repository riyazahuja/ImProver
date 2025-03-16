open Topology in
/-- If `f : α → ℂ` is summable, then so is `n ↦ log (1 - f n)`. -/
lemma Summable.clog_one_sub {α  : Type*} {f : α → ℂ} (hsum : Summable f) :
    Summable fun n ↦ log (1 - f n) := by
  have hg : DifferentiableAt ℂ (fun z ↦ log (1 - z)) 0 := by
    have : 1 - 0 ∈ slitPlane := (sub_zero (1 : ℂ)).symm ▸ one_mem_slitPlane
    fun_prop (disch := assumption)
  have : (fun z ↦ log (1 - z)) =O[𝓝 0] id := by
    simpa only [sub_zero, log_one] using hg.isBigO_sub
  /-
    α : Type u_1
    f : α → Complex
    hsum : Summable f
    hg : DifferentiableAt Complex (fun z => Complex.log (HSub.hSub 1 z)) 0
    this : Asymptotics.IsBigO (nhds 0) (fun z => Complex.log (HSub.hSub 1 z)) id
    ⊢ Summable fun n => Complex.log (HSub.hSub 1 (f n))
  -/
  exact this.comp_summable hsum
  /-
    🎉 no goals
  -/


/-- A variant of the Euler Product formula in terms of the exponential of a sum of logarithms. -/
theorem exp_tsum_primes_log_eq_tsum {f : ℕ →*₀ ℂ} (hsum : Summable (‖f ·‖)) :
    exp (∑' p : Nat.Primes, -log (1 - f p)) = ∑' n : ℕ, f n := by
  /-
    f : MonoidWithZeroHom Nat Complex
    hsum : Summable fun x => Norm.norm (f x)
    ⊢ Eq (Complex.exp (tsum fun p => Neg.neg (Complex.log (HSub.hSub 1 (f ↑p)))))  …
  -/
  have hs {p : ℕ} (hp : 1 < p) : ‖f p‖ < 1 := hsum.of_norm.norm_lt_one (f := f.toMonoidHom) hp
  have hp (p : Nat.Primes) : 1 - f p ≠ 0 :=
    fun h ↦ (norm_one (α := ℂ) ▸ (sub_eq_zero.mp h) ▸ hs p.prop.one_lt).false
  /-
    f : MonoidWithZeroHom Nat Complex
    hsum : Summable fun x => Norm.norm (f x)
    hs : ∀ {p : Nat}, LT.lt 1 p → LT.lt (Norm.norm (f p)) 1
    hp : ∀ (p : Nat.Primes), Ne (HSub.hSub 1 (f ↑p)) 0
    ⊢ Eq (Complex.exp (tsum fun p => Neg.neg (Complex.log (HSub.hSub 1 (f ↑p)))))  …
  -/
  have H := hsum.of_norm.clog_one_sub.neg.subtype {p | p.Prime} |>.hasSum.cexp.tprod_eq
  /-
    f : MonoidWithZeroHom Nat Complex
    hsum : Summable fun x => Norm.norm (f x)
    hs : ∀ {p : Nat}, LT.lt 1 p → LT.lt (Norm.norm (f p)) 1
    hp : ∀ (p : Nat.Primes), Ne (HSub.hSub 1 (f ↑p)) 0
    H : Eq (tprod fun b => Function.comp Complex.exp (Function.comp (fun b => Neg. …
    ⊢ Eq (Complex.exp (tsum fun p => Neg.neg (Complex.log (HSub.hSub 1 (f ↑p)))))  …
  -/
  simp only [Set.coe_setOf, Set.mem_setOf_eq, Function.comp_apply, exp_neg, exp_log (hp _)] at H
  /-
    f : MonoidWithZeroHom Nat Complex
    hsum : Summable fun x => Norm.norm (f x)
    hs : ∀ {p : Nat}, LT.lt 1 p → LT.lt (Norm.norm (f p)) 1
    hp : ∀ (p : Nat.Primes), Ne (HSub.hSub 1 (f ↑p)) 0
    H : Eq (tprod fun b => Inv.inv (HSub.hSub 1 (f ↑b))) (Complex.exp (tsum fun b  …
    ⊢ Eq (Complex.exp (tsum fun p => Neg.neg (Complex.log (HSub.hSub 1 (f ↑p)))))  …
  -/
  exact H.symm.trans <| eulerProduct_completely_multiplicative_tprod hsum
  /-
    🎉 no goals
  -/


