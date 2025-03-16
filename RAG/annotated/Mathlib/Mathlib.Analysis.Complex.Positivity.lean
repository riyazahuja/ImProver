/-- A function that is holomorphic on the open disk around `c` with radius `r` and whose iterated
derivatives at `c` are all nonnegative real has nonnegative real values on `c + [0,r)`. -/
theorem nonneg_of_iteratedDeriv_nonneg {f : ℂ → ℂ} {c : ℂ} {r : ℝ}
    (hf : DifferentiableOn ℂ f (Metric.ball c r)) (h : ∀ n, 0 ≤ iteratedDeriv n f c) ⦃z : ℂ⦄
    (hz₁ : c ≤ z) (hz₂ : z ∈ Metric.ball c r):
    0 ≤ f z := by
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₁ : LE.le c z
    hz₂ : Membership.mem (Metric.ball c r) z
    ⊢ LE.le 0 (f z)
  -/
  have H := taylorSeries_eq_on_ball' hz₂ hf
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₁ : LE.le c z
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    ⊢ LE.le 0 (f z)
  -/
  rw [← sub_nonneg] at hz₁
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₁ : LE.le 0 (HSub.hSub z c)
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    ⊢ LE.le 0 (f z)
  -/
  have hz' := eq_re_of_ofReal_le hz₁
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₁ : LE.le 0 (HSub.hSub z c)
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    hz' : Eq (HSub.hSub z c) ↑(HSub.hSub z c).re
    ⊢ LE.le 0 (f z)
  -/
  rw [hz'] at hz₁ H
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₁ : LE.le 0 ↑(HSub.hSub z c).re
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    hz' : Eq (HSub.hSub z c) ↑(HSub.hSub z c).re
    ⊢ LE.le 0 (f z)
  -/
  refine H ▸ tsum_nonneg fun n ↦ ?_
  rw [← ofReal_natCast, ← ofReal_pow, ← ofReal_inv, eq_re_of_ofReal_le (h n), ← ofReal_mul,
    ← ofReal_mul]
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₁ : LE.le 0 ↑(HSub.hSub z c).re
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    hz' : Eq (HSub.hSub z c) ↑(HSub.hSub z c).re
    n : Nat
    ⊢ LE.le 0 ↑(HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv n f c). …
  -/
  norm_cast at hz₁ ⊢
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    hz' : Eq (HSub.hSub z c) ↑(HSub.hSub z c).re
    n : Nat
    hz₁ : LE.le 0 (HSub.hSub z c).re
    ⊢ LE.le 0 (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv n f c).r …
  -/
  have := zero_re ▸ (Complex.le_def.mp (h n)).1
  /-
    f : Complex → Complex
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz₂ : Membership.mem (Metric.ball c r) z
    H : Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDer …
    hz' : Eq (HSub.hSub z c) ↑(HSub.hSub z c).re
    n : Nat
    hz₁ : LE.le 0 (HSub.hSub z c).re
    this : LE.le 0 (iteratedDeriv n f c).re
    ⊢ LE.le 0 (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv n f c).r …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- An entire function whose iterated derivatives at `c` are all nonnegative real has nonnegative
real values on `c + ℝ≥0`. -/
theorem nonneg_of_iteratedDeriv_nonneg {f : ℂ → ℂ} (hf : Differentiable ℂ f) {c : ℂ}
    (h : ∀ n, 0 ≤ iteratedDeriv n f c) ⦃z : ℂ⦄ (hz : c ≤ z) :
    0 ≤ f z := by
  /-
    f : Complex → Complex
    hf : Differentiable Complex f
    c : Complex
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz : LE.le c z
    ⊢ LE.le 0 (f z)
  -/
  refine hf.differentiableOn.nonneg_of_iteratedDeriv_nonneg (r := (z - c).re + 1) h hz ?_
  /-
    f : Complex → Complex
    hf : Differentiable Complex f
    c : Complex
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz : LE.le c z
    ⊢ Membership.mem (Metric.ball c (HAdd.hAdd (HSub.hSub z c).re 1)) z
  -/
  rw [← sub_nonneg] at hz
  /-
    f : Complex → Complex
    hf : Differentiable Complex f
    c : Complex
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz : LE.le 0 (HSub.hSub z c)
    ⊢ Membership.mem (Metric.ball c (HAdd.hAdd (HSub.hSub z c).re 1)) z
  -/
  rw [Metric.mem_ball, dist_eq, eq_re_of_ofReal_le hz]
  /-
    f : Complex → Complex
    hf : Differentiable Complex f
    c : Complex
    h : ∀ (n : Nat), LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz : LE.le 0 (HSub.hSub z c)
    ⊢ LT.lt (Complex.abs ↑(HSub.hSub z c).re) (HAdd.hAdd (↑(HSub.hSub z c).re).re 1)
  -/
  simpa only [Complex.abs_of_nonneg (nonneg_iff.mp hz).1] using lt_add_one _
  /-
    🎉 no goals
  -/


/-- An entire function whose iterated derivatives at `c` are all nonnegative real (except
possibly the value itself) has values of the form `f c + nonneg. real` on the set `c + ℝ≥0`. -/
theorem apply_le_of_iteratedDeriv_nonneg {f : ℂ → ℂ} {c : ℂ} (hf : Differentiable ℂ f)
    (h : ∀ n ≠ 0, 0 ≤ iteratedDeriv n f c) ⦃z : ℂ⦄ (hz : c ≤ z) :
    f c ≤ f z := by
  have h' (n : ℕ) : 0 ≤ iteratedDeriv n (f · - f c) c := by
    cases n with
    | zero => simp only [iteratedDeriv_zero, sub_self, le_refl]
    | succ n =>
      specialize h (n + 1) n.succ_ne_zero
      rw [iteratedDeriv_succ'] at h ⊢
      rwa [funext fun x ↦ deriv_sub_const (f := f) (x := x) (f c)]
  /-
    f : Complex → Complex
    c : Complex
    hf : Differentiable Complex f
    h : ∀ (n : Nat), Ne n 0 → LE.le 0 (iteratedDeriv n f c)
    z : Complex
    hz : LE.le c z
    h' : ∀ (n : Nat), LE.le 0 (iteratedDeriv n (fun x => HSub.hSub (f x) (f c)) c)
    ⊢ LE.le (f c) (f z)
  -/
  exact sub_nonneg.mp <| nonneg_of_iteratedDeriv_nonneg (hf.sub_const _) h' hz
  /-
    🎉 no goals
  -/


/-- An entire function whose iterated derivatives at `c` are all real with alternating signs
(except possibly the value itself) has values of the form `f c + nonneg. real` along the
set `c - ℝ≥0`. -/
theorem apply_le_of_iteratedDeriv_alternating {f : ℂ → ℂ} {c : ℂ} (hf : Differentiable ℂ f)
    (h : ∀ n ≠ 0, 0 ≤ (-1) ^ n * iteratedDeriv n f c) ⦃z : ℂ⦄ (hz : z ≤ c) :
    f c ≤ f z := by
  convert apply_le_of_iteratedDeriv_nonneg (f := fun z ↦ f (-z))
    (hf.comp <| differentiable_neg) (fun n hn ↦ ?_) (neg_le_neg_iff.mpr hz) using 1
    /-
      case h.e'_3
      f : Complex → Complex
      c : Complex
      hf : Differentiable Complex f
      h : ∀ (n : Nat), Ne n 0 → LE.le 0 (HMul.hMul (HPow.hPow (-1) n) (iteratedDeriv …
      z : Complex
      hz : LE.le z c
      ⊢ Eq (f c) (f (Neg.neg (Neg.neg c)))
    -/
  · simp only [neg_neg]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      f : Complex → Complex
      c : Complex
      hf : Differentiable Complex f
      h : ∀ (n : Nat), Ne n 0 → LE.le 0 (HMul.hMul (HPow.hPow (-1) n) (iteratedDeriv …
      z : Complex
      hz : LE.le z c
      ⊢ Eq (f z) (f (Neg.neg (Neg.neg z)))
    -/
  · simp only [neg_neg]
    /-
      🎉 no goals
    -/
    /-
      f : Complex → Complex
      c : Complex
      hf : Differentiable Complex f
      h : ∀ (n : Nat), Ne n 0 → LE.le 0 (HMul.hMul (HPow.hPow (-1) n) (iteratedDeriv …
      z : Complex
      hz : LE.le z c
      n : Nat
      hn : Ne n 0
      ⊢ LE.le 0 (iteratedDeriv n (fun z => f (Neg.neg z)) (Neg.neg c))
    -/
  · simpa only [iteratedDeriv_comp_neg, neg_neg, smul_eq_mul] using h n hn
    /-
      🎉 no goals
    -/


