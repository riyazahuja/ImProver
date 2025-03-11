/-- The `n`th term of the L-series of `f` evaluated at `s`. We set it to zero when `n = 0`. -/
noncomputable
def term (f : ℕ → ℂ) (s : ℂ) (n : ℕ) : ℂ :=
  if n = 0 then 0 else f n / n ^ s


lemma term_def (f : ℕ → ℂ) (s : ℂ) (n : ℕ) :
    term f s n = if n = 0 then 0 else f n / n ^ s :=
  rfl


@[simp]
lemma term_zero (f : ℕ → ℂ) (s : ℂ) : term f s 0 = 0 := rfl

-- We put `hn` first for convnience, so that we can write `rw [LSeries.term_of_ne_zero hn]` etc.

@[simp]
lemma term_of_ne_zero {n : ℕ} (hn : n ≠ 0) (f : ℕ → ℂ) (s : ℂ) :
    term f s n = f n / n ^ s :=
  if_neg hn


/--
If `s ≠ 0`, then the `if .. then .. else` construction in `LSeries.term` isn't needed, since
`0 ^ s = 0`.
-/
lemma term_of_ne_zero' {s : ℂ} (hs : s ≠ 0) (f : ℕ → ℂ) (n : ℕ) :
    term f s n = f n / n ^ s := by
  /-
    s : Complex
    hs : Ne s 0
    f : Nat → Complex
    n : Nat
    ⊢ Eq (LSeries.term f s n) (HDiv.hDiv (f n) (HPow.hPow (↑n) s))
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      s : Complex
      hs : Ne s 0
      f : Nat → Complex
      ⊢ Eq (LSeries.term f s 0) (HDiv.hDiv (f 0) (HPow.hPow (↑0) s))
    -/
  · rw [term_zero, Nat.cast_zero, zero_cpow hs, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Complex
      hs : Ne s 0
      f : Nat → Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (LSeries.term f s n) (HDiv.hDiv (f n) (HPow.hPow (↑n) s))
    -/
  · rw [term_of_ne_zero hn]
    /-
      🎉 no goals
    -/


lemma term_congr {f g : ℕ → ℂ} (h : ∀ {n}, n ≠ 0 → f n = g n) (s : ℂ) (n : ℕ) :
    term f s n = term g s n := by
  /-
    f g : Nat → Complex
    h : ∀ {n : Nat}, Ne n 0 → Eq (f n) (g n)
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term f s n) (LSeries.term g s n)
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases eq_or_ne n 0 with hn | hn <;> simp [hn, h]
                                       /-
                                         🎉 no goals
                                       -/


lemma pow_mul_term_eq (f : ℕ → ℂ) (s : ℂ) (n : ℕ) :
    (n + 1) ^ s * term f s (n + 1) = f (n + 1) := by
  /-
    f : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (↑n) 1) s) (LSeries.term f s (HAdd.hAdd  …
  -/
  simp [term, natCast_add_one_cpow_ne_zero n _, mul_comm (f _), mul_div_assoc']
  /-
    🎉 no goals
  -/


lemma norm_term_eq (f : ℕ → ℂ) (s : ℂ) (n : ℕ) :
    ‖term f s n‖ = if n = 0 then 0 else ‖f n‖ / n ^ s.re := by
  /-
    f : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (Norm.norm (LSeries.term f s n)) (ite (Eq n 0) 0 (HDiv.hDiv (Norm.norm (f …
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      f : Nat → Complex
      s : Complex
      ⊢ Eq (Norm.norm (LSeries.term f s 0)) (ite (Eq 0 0) 0 (HDiv.hDiv (Norm.norm (f …
    -/
  · simp only [term_zero, norm_zero, ↓reduceIte]
    /-
      🎉 no goals
    -/
    /-
      case inr
      f : Nat → Complex
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (Norm.norm (LSeries.term f s n)) (ite (Eq n 0) 0 (HDiv.hDiv (Norm.norm (f …
    -/
  · rw [if_neg hn, term_of_ne_zero hn, norm_div, norm_natCast_cpow_of_pos <| Nat.pos_of_ne_zero hn]
    /-
      🎉 no goals
    -/


lemma norm_term_le {f g : ℕ → ℂ} (s : ℂ) {n : ℕ} (h : ‖f n‖ ≤ ‖g n‖) :
    ‖term f s n‖ ≤ ‖term g s n‖ := by
  /-
    f g : Nat → Complex
    s : Complex
    n : Nat
    h : LE.le (Norm.norm (f n)) (Norm.norm (g n))
    ⊢ LE.le (Norm.norm (LSeries.term f s n)) (Norm.norm (LSeries.term g s n))
  -/
  simp only [norm_term_eq]
  /-
    f g : Nat → Complex
    s : Complex
    n : Nat
    h : LE.le (Norm.norm (f n)) (Norm.norm (g n))
    ⊢ LE.le (ite (Eq n 0) 0 (HDiv.hDiv (Norm.norm (f n)) (HPow.hPow (↑n) s.re))) ( …
  -/
  split
    /-
      case isTrue
      f g : Nat → Complex
      s : Complex
      n : Nat
      h : LE.le (Norm.norm (f n)) (Norm.norm (g n))
      h✝ : Eq n 0
      ⊢ LE.le 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case isFalse
      f g : Nat → Complex
      s : Complex
      n : Nat
      h : LE.le (Norm.norm (f n)) (Norm.norm (g n))
      h✝ : Not (Eq n 0)
      ⊢ LE.le (HDiv.hDiv (Norm.norm (f n)) (HPow.hPow (↑n) s.re)) (HDiv.hDiv (Norm.n …
    -/
  · gcongr
    /-
      🎉 no goals
    -/


lemma norm_term_le_of_re_le_re (f : ℕ → ℂ) {s s' : ℂ} (h : s.re ≤ s'.re) (n : ℕ) :
    ‖term f s' n‖ ≤ ‖term f s n‖ := by
  /-
    f : Nat → Complex
    s s' : Complex
    h : LE.le s.re s'.re
    n : Nat
    ⊢ LE.le (Norm.norm (LSeries.term f s' n)) (Norm.norm (LSeries.term f s n))
  -/
  simp only [norm_term_eq]
  /-
    f : Nat → Complex
    s s' : Complex
    h : LE.le s.re s'.re
    n : Nat
    ⊢ LE.le (ite (Eq n 0) 0 (HDiv.hDiv (Norm.norm (f n)) (HPow.hPow (↑n) s'.re)))  …
  -/
  split
  /-
    case isTrue
    f : Nat → Complex
    s s' : Complex
    h : LE.le s.re s'.re
    n : Nat
    h✝ : Eq n 0
    ⊢ LE.le 0 0
  -/
  next => rfl
  /-
    case isFalse
    f : Nat → Complex
    s s' : Complex
    h : LE.le s.re s'.re
    n : Nat
    h✝ : Not (Eq n 0)
    ⊢ LE.le (HDiv.hDiv (Norm.norm (f n)) (HPow.hPow (↑n) s'.re)) (HDiv.hDiv (Norm. …
  -/
  next hn => gcongr; exact Nat.one_le_cast.mpr <| Nat.one_le_iff_ne_zero.mpr hn
  /-
    🎉 no goals
  -/


lemma term_nonneg {a : ℕ → ℂ} {n : ℕ} (h : 0 ≤ a n) (x : ℝ) : 0 ≤ term a x n := by
  /-
    a : Nat → Complex
    n : Nat
    h : LE.le 0 (a n)
    x : Real
    ⊢ LE.le 0 (LSeries.term a (↑x) n)
  -/
  rw [term_def]
  /-
    a : Nat → Complex
    n : Nat
    h : LE.le 0 (a n)
    x : Real
    ⊢ LE.le 0 (ite (Eq n 0) 0 (HDiv.hDiv (a n) (HPow.hPow ↑n ↑x)))
  -/
  split_ifs with hn
  /-
    case pos
    a : Nat → Complex
    n : Nat
    h : LE.le 0 (a n)
    x : Real
    hn : Eq n 0
    ⊢ LE.le 0 0
  -/
  exacts [le_rfl, mul_nonneg h (inv_natCast_cpow_ofReal_pos hn x).le]
  /-
    🎉 no goals
  -/


lemma term_pos {a : ℕ → ℂ} {n : ℕ} (hn : n ≠ 0) (h : 0 < a n) (x : ℝ) : 0 < term a x n := by
  /-
    a : Nat → Complex
    n : Nat
    hn : Ne n 0
    h : LT.lt 0 (a n)
    x : Real
    ⊢ LT.lt 0 (LSeries.term a (↑x) n)
  -/
  simpa only [term_of_ne_zero hn] using mul_pos h <| inv_natCast_cpow_ofReal_pos hn x
  /-
    🎉 no goals
  -/


/-- The value of the L-series of the sequence `f` at the point `s`
if it converges absolutely there, and `0` otherwise. -/
noncomputable
def LSeries (f : ℕ → ℂ) (s : ℂ) : ℂ :=
  ∑' n, term f s n

-- TODO: change argument order in `LSeries_congr` to have `s` last.

lemma LSeries_congr {f g : ℕ → ℂ} (s : ℂ) (h : ∀ {n}, n ≠ 0 → f n = g n) :
    LSeries f s = LSeries g s :=
  tsum_congr <| term_congr h s


/-- `LSeriesSummable f s` indicates that the L-series of `f` converges absolutely at `s`. -/
def LSeriesSummable (f : ℕ → ℂ) (s : ℂ) : Prop :=
  Summable (term f s)


lemma LSeriesSummable_congr {f g : ℕ → ℂ} (s : ℂ) (h : ∀ {n}, n ≠ 0 → f n = g n) :
    LSeriesSummable f s ↔ LSeriesSummable g s :=
  summable_congr <| term_congr h s


open Filter in
/-- If `f` and `g` agree on large `n : ℕ` and the `LSeries` of `f` converges at `s`,
then so does that of `g`. -/
lemma LSeriesSummable.congr' {f g : ℕ → ℂ} (s : ℂ) (h : f =ᶠ[atTop] g) (hf : LSeriesSummable f s) :
    LSeriesSummable g s := by
  /-
    f g : Nat → Complex
    s : Complex
    h : Filter.atTop.EventuallyEq f g
    hf : LSeriesSummable f s
    ⊢ LSeriesSummable g s
  -/
  rw [← Nat.cofinite_eq_atTop] at h
  /-
    f g : Nat → Complex
    s : Complex
    h : Filter.cofinite.EventuallyEq f g
    hf : LSeriesSummable f s
    ⊢ LSeriesSummable g s
  -/
  refine (summable_norm_iff.mpr hf).of_norm_bounded_eventually _ ?_
  have : term f s =ᶠ[cofinite] term g s := by
    rw [eventuallyEq_iff_exists_mem] at h ⊢
    obtain ⟨S, hS, hS'⟩ := h
    refine ⟨S \ {0}, diff_mem hS <| (Set.finite_singleton 0).compl_mem_cofinite, fun n hn ↦ ?_⟩
    simp only [Set.mem_diff, Set.mem_singleton_iff] at hn
    simp only [term_of_ne_zero hn.2, hS' hn.1]
  /-
    f g : Nat → Complex
    s : Complex
    h : Filter.cofinite.EventuallyEq f g
    hf : LSeriesSummable f s
    this : Filter.cofinite.EventuallyEq (LSeries.term f s) (LSeries.term g s)
    ⊢ Filter.Eventually (fun i => LE.le (Norm.norm (LSeries.term g s i)) (Norm.nor …
  -/
  exact Eventually.mono this.symm fun n hn ↦ by simp only [hn, le_rfl]
  /-
    🎉 no goals
  -/


open Filter in
/-- If `f` and `g` agree on large `n : ℕ`, then the `LSeries` of `f` converges at `s`
if and only if that of `g` does. -/
lemma LSeriesSummable_congr' {f g : ℕ → ℂ} (s : ℂ) (h : f =ᶠ[atTop] g) :
    LSeriesSummable f s ↔ LSeriesSummable g s :=
  ⟨fun H ↦ H.congr' s h, fun H ↦ H.congr' s h.symm⟩


theorem LSeries.eq_zero_of_not_LSeriesSummable (f : ℕ → ℂ) (s : ℂ) :
    ¬ LSeriesSummable f s → LSeries f s = 0 :=
  tsum_eq_zero_of_not_summable


@[simp]
theorem LSeriesSummable_zero {s : ℂ} : LSeriesSummable 0 s := by
  simp only [LSeriesSummable, funext (term_def 0 s), Pi.zero_apply, zero_div, ite_self,
    summable_zero]


/-- This states that the L-series of the sequence `f` converges absolutely at `s` and that
the value there is `a`. -/
def LSeriesHasSum (f : ℕ → ℂ) (s a : ℂ) : Prop :=
  HasSum (term f s) a


lemma LSeriesHasSum.LSeriesSummable {f : ℕ → ℂ} {s a : ℂ}
    (h : LSeriesHasSum f s a) : LSeriesSummable f s :=
  h.summable


lemma LSeriesHasSum.LSeries_eq {f : ℕ → ℂ} {s a : ℂ}
    (h : LSeriesHasSum f s a) : LSeries f s = a :=
  h.tsum_eq


lemma LSeriesSummable.LSeriesHasSum {f : ℕ → ℂ} {s : ℂ} (h : LSeriesSummable f s) :
    LSeriesHasSum f s (LSeries f s) :=
  h.hasSum


lemma LSeriesHasSum_iff {f : ℕ → ℂ} {s a : ℂ} :
    LSeriesHasSum f s a ↔ LSeriesSummable f s ∧ LSeries f s = a :=
  ⟨fun H ↦ ⟨H.LSeriesSummable, H.LSeries_eq⟩, fun ⟨H₁, H₂⟩ ↦ H₂ ▸ H₁.LSeriesHasSum⟩


lemma LSeriesHasSum_congr {f g : ℕ → ℂ} (s a : ℂ) (h : ∀ {n}, n ≠ 0 → f n = g n) :
    LSeriesHasSum f s a ↔ LSeriesHasSum g s a := by
  /-
    f g : Nat → Complex
    s a : Complex
    h : ∀ {n : Nat}, Ne n 0 → Eq (f n) (g n)
    ⊢ Iff (LSeriesHasSum f s a) (LSeriesHasSum g s a)
  -/
  simp only [LSeriesHasSum_iff, LSeriesSummable_congr s h, LSeries_congr s h]
  /-
    🎉 no goals
  -/


lemma LSeriesSummable.of_re_le_re {f : ℕ → ℂ} {s s' : ℂ} (h : s.re ≤ s'.re)
    (hf : LSeriesSummable f s) : LSeriesSummable f s' := by
  /-
    f : Nat → Complex
    s s' : Complex
    h : LE.le s.re s'.re
    hf : LSeriesSummable f s
    ⊢ LSeriesSummable f s'
  -/
  rw [LSeriesSummable, ← summable_norm_iff] at hf ⊢
  /-
    f : Nat → Complex
    s s' : Complex
    h : LE.le s.re s'.re
    hf : Summable fun x => Norm.norm (LSeries.term f s x)
    ⊢ Summable fun x => Norm.norm (LSeries.term f s' x)
  -/
  exact hf.of_nonneg_of_le (fun _ ↦ norm_nonneg _) (norm_term_le_of_re_le_re f h)
  /-
    🎉 no goals
  -/


theorem LSeriesSummable_iff_of_re_eq_re {f : ℕ → ℂ} {s s' : ℂ} (h : s.re = s'.re) :
    LSeriesSummable f s ↔ LSeriesSummable f s' :=
  ⟨fun H ↦ H.of_re_le_re h.le, fun H ↦ H.of_re_le_re h.symm.le⟩


/-- The indicator function of `{1} ⊆ ℕ` with values in `ℂ`. -/
def LSeries.delta (n : ℕ) : ℂ :=
  if n = 1 then 1 else 0


@[inherit_doc]
scoped[LSeries.notation] notation "L" => LSeries


/-- We introduce notation `↗f` for `f` interpreted as a function `ℕ → ℂ`.

Let `R` be a ring with a coercion to `ℂ`. Then we can write `↗χ` when `χ : DirichletCharacter R`
or `↗f` when `f : ArithmeticFunction R` or simply `f : N → R` with a coercion from `ℕ` to `N`
as an argument to `LSeries`, `LSeriesHasSum`, `LSeriesSummable` etc. -/
scoped[LSeries.notation] notation:max "↗" f:max => fun n : ℕ ↦ (f n : ℂ)


@[inherit_doc]
scoped[LSeries.notation] notation "δ" => delta


@[simp]
lemma LSeries_zero : LSeries 0 = 0 := by
  /-
    ⊢ Eq (LSeries 0) 0
  -/
  ext
  /-
    case h
    x✝ : Complex
    ⊢ Eq (LSeries 0 x✝) (0 x✝)
  -/
  simp only [LSeries, LSeries.term, Pi.zero_apply, zero_div, ite_self, tsum_zero]
  /-
    🎉 no goals
  -/


lemma term_delta (s : ℂ) (n : ℕ) : term δ s n = if n = 1 then 1 else 0 := by
  /-
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term LSeries.delta s n) (ite (Eq n 1) 1 0)
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      s : Complex
      ⊢ Eq (LSeries.term LSeries.delta s 0) (ite (Eq 0 1) 1 0)
    -/
  · simp only [term_zero, zero_ne_one, ↓reduceIte]
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (LSeries.term LSeries.delta s n) (ite (Eq n 1) 1 0)
    -/
  · simp only [ne_eq, hn, not_false_eq_true, term_of_ne_zero, delta]
    /-
      case inr
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (HDiv.hDiv (ite (Eq n 1) 1 0) (HPow.hPow (↑n) s)) (ite (Eq n 1) 1 0)
    -/
    rcases eq_or_ne n 1 with rfl | hn'
      /-
        case inr.inl
        s : Complex
        hn : Ne 1 0
        ⊢ Eq (HDiv.hDiv (ite (Eq 1 1) 1 0) (HPow.hPow (↑1) s)) (ite (Eq 1 1) 1 0)
      -/
    · simp only [↓reduceIte, cast_one, one_cpow, ne_eq, one_ne_zero, not_false_eq_true, div_self]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        s : Complex
        n : Nat
        hn : Ne n 0
        hn' : Ne n 1
        ⊢ Eq (HDiv.hDiv (ite (Eq n 1) 1 0) (HPow.hPow (↑n) s)) (ite (Eq n 1) 1 0)
      -/
    · simp only [hn', ↓reduceIte, zero_div]
      /-
        🎉 no goals
      -/


lemma mul_delta_eq_smul_delta {f : ℕ → ℂ} : f * δ = f 1 • δ := by
  /-
    f : Nat → Complex
    ⊢ Eq (HMul.hMul f LSeries.delta) (HSMul.hSMul (f 1) LSeries.delta)
  -/
  ext n
  /-
    case h
    f : Nat → Complex
    n : Nat
    ⊢ Eq (HMul.hMul f LSeries.delta n) (HSMul.hSMul (f 1) LSeries.delta n)
  -/
  simp only [Pi.mul_apply, delta, mul_ite, mul_one, mul_zero, Pi.smul_apply, smul_eq_mul]
  /-
    case h
    f : Nat → Complex
    n : Nat
    ⊢ Eq (ite (Eq n 1) (f n) 0) (ite (Eq n 1) (f 1) 0)
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hn <;> simp only [hn]
                        /-
                          🎉 no goals
                        -/


lemma mul_delta {f : ℕ → ℂ} (h : f 1 = 1) : f * δ = δ := by
  /-
    f : Nat → Complex
    h : Eq (f 1) 1
    ⊢ Eq (HMul.hMul f LSeries.delta) LSeries.delta
  -/
  rw [mul_delta_eq_smul_delta, h, one_smul]
  /-
    🎉 no goals
  -/


lemma delta_mul_eq_smul_delta {f : ℕ → ℂ} : δ * f = f 1 • δ :=
  mul_comm δ f ▸ mul_delta_eq_smul_delta


lemma delta_mul {f : ℕ → ℂ} (h : f 1 = 1) : δ * f = δ :=
  mul_comm δ f ▸ mul_delta h


/-- The L-series of `δ` is the constant function `1`. -/
lemma LSeries_delta : LSeries δ = 1 := by
  /-
    ⊢ Eq (LSeries LSeries.delta) 1
  -/
  ext
  /-
    case h
    x✝ : Complex
    ⊢ Eq (LSeries LSeries.delta x✝) (1 x✝)
  -/
  simp only [LSeries, LSeries.term_delta, tsum_ite_eq, Pi.one_apply]
  /-
    🎉 no goals
  -/


/-- If the `LSeries` of `f` is summable at `s`, then `f n` is bounded in absolute value
by a constant times `n^(re s)`. -/
lemma LSeriesSummable.le_const_mul_rpow {f : ℕ → ℂ} {s : ℂ} (h : LSeriesSummable f s) :
    ∃ C, ∀ n ≠ 0, ‖f n‖ ≤ C * n ^ s.re := by
  /-
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    ⊢ Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C ( …
  -/
  replace h := h.norm
  /-
    f : Nat → Complex
    s : Complex
    h : Summable fun x => Norm.norm (LSeries.term f s x)
    ⊢ Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C ( …
  -/
  by_contra! H
  /-
    f : Nat → Complex
    s : Complex
    h : Summable fun x => Norm.norm (LSeries.term f s x)
    H : ∀ (C : Real), Exists fun n => And (Ne n 0) (LT.lt (HMul.hMul C (HPow.hPow  …
    ⊢ False
  -/
  obtain ⟨n, hn₀, hn⟩ := H (tsum fun n ↦ ‖term f s n‖)
  /-
    case intro.intro
    f : Nat → Complex
    s : Complex
    h : Summable fun x => Norm.norm (LSeries.term f s x)
    H : ∀ (C : Real), Exists fun n => And (Ne n 0) (LT.lt (HMul.hMul C (HPow.hPow  …
    n : Nat
    hn₀ : Ne n 0
    hn : LT.lt (HMul.hMul (tsum fun n => Norm.norm (LSeries.term f s n)) (HPow.hPo …
    ⊢ False
  -/
  have := le_tsum h n fun _ _ ↦ norm_nonneg _
  rw [norm_term_eq, if_neg hn₀,
    div_le_iff₀ <| Real.rpow_pos_of_pos (Nat.cast_pos.mpr <| Nat.pos_of_ne_zero hn₀) _] at this
  /-
    case intro.intro
    f : Nat → Complex
    s : Complex
    h : Summable fun x => Norm.norm (LSeries.term f s x)
    H : ∀ (C : Real), Exists fun n => And (Ne n 0) (LT.lt (HMul.hMul C (HPow.hPow  …
    n : Nat
    hn₀ : Ne n 0
    hn : LT.lt (HMul.hMul (tsum fun n => Norm.norm (LSeries.term f s n)) (HPow.hPo …
    this : LE.le (Norm.norm (f n)) (HMul.hMul (tsum fun i => Norm.norm (LSeries.te …
    ⊢ False
  -/
  exact (this.trans_lt hn).false.elim
  /-
    🎉 no goals
  -/


open Filter in
/-- If the `LSeries` of `f` is summable at `s`, then `f = O(n^(re s))`. -/
lemma LSeriesSummable.isBigO_rpow {f : ℕ → ℂ} {s : ℂ} (h : LSeriesSummable f s) :
    f =O[atTop] fun n ↦ (n : ℝ) ^ s.re := by
  /-
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    ⊢ Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) s.re
  -/
  obtain ⟨C, hC⟩ := h.le_const_mul_rpow
  /-
    case intro
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    ⊢ Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) s.re
  -/
  refine Asymptotics.IsBigO.of_bound C <| eventually_atTop.mpr ⟨1, fun n hn ↦ ?_⟩
  /-
    case intro
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    n : Nat
    hn : GE.ge n 1
    ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C (Norm.norm (HPow.hPow (↑n) s.re)))
  -/
  convert hC n (Nat.pos_iff_ne_zero.mp hn) using 2
  /-
    case h.e'_4.h.e'_6
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    n : Nat
    hn : GE.ge n 1
    ⊢ Eq (Norm.norm (HPow.hPow (↑n) s.re)) (HPow.hPow (↑n) s.re)
  -/
  rw [Real.norm_eq_abs, Real.abs_rpow_of_nonneg n.cast_nonneg, _root_.abs_of_nonneg n.cast_nonneg]
  /-
    🎉 no goals
  -/


/-- If `f n` is bounded in absolute value by a constant times `n^(x-1)` and `re s > x`,
then the `LSeries` of `f` is summable at `s`. -/
lemma LSeriesSummable_of_le_const_mul_rpow {f : ℕ → ℂ} {x : ℝ} {s : ℂ} (hs : x < s.re)
    (h : ∃ C, ∀ n ≠ 0, ‖f n‖ ≤ C * n ^ (x - 1)) :
    LSeriesSummable f s := by
  /-
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Exists fun C => ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C …
    ⊢ LSeriesSummable f s
  -/
  obtain ⟨C, hC⟩ := h
  have hC₀ : 0 ≤ C := by
    specialize hC 1 one_ne_zero
    simp only [Nat.cast_one, Real.one_rpow, mul_one] at hC
    exact (norm_nonneg _).trans hC
  have hsum : Summable fun n : ℕ ↦ ‖(C : ℂ) / n ^ (s + (1 - x))‖ := by
    simp_rw [div_eq_mul_inv, norm_mul, ← cpow_neg]
    have hsx : -s.re + x - 1 < -1 := by linarith only [hs]
    refine Summable.mul_left _ <|
      Summable.of_norm_bounded_eventually_nat (fun n ↦ (n : ℝ) ^ (-s.re + x - 1)) ?_ ?_
    · simp only [Real.summable_nat_rpow, hsx]
    · simp only [neg_add_rev, neg_sub, norm_norm, Filter.eventually_atTop]
      refine ⟨1, fun n hn ↦ ?_⟩
      simp only [norm_natCast_cpow_of_pos hn, add_re, sub_re, neg_re, ofReal_re, one_re]
      convert le_refl ?_ using 2
      ring
  /-
    case intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    hC₀ : LE.le 0 C
    hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
    ⊢ LSeriesSummable f s
  -/
  refine Summable.of_norm <| hsum.of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun n ↦ ?_)
  /-
    case intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    hC₀ : LE.le 0 C
    hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
    n : Nat
    ⊢ LE.le (Norm.norm (LSeries.term f s n)) (Norm.norm (HDiv.hDiv (↑C) (HPow.hPow …
  -/
  rcases n.eq_zero_or_pos with rfl | hn
    /-
      case intro.inl
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      C : Real
      hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
      hC₀ : LE.le 0 C
      hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
      ⊢ LE.le (Norm.norm (LSeries.term f s 0)) (Norm.norm (HDiv.hDiv (↑C) (HPow.hPow …
    -/
  · simp only [term_zero, norm_zero]
    /-
      case intro.inl
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      C : Real
      hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
      hC₀ : LE.le 0 C
      hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
      ⊢ LE.le 0 (Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑0) (HAdd.hAdd s (HSub.hSub 1 …
    -/
    exact norm_nonneg _
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    hC₀ : LE.le 0 C
    hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
    n : Nat
    hn : GT.gt n 0
    ⊢ LE.le (Norm.norm (LSeries.term f s n)) (Norm.norm (HDiv.hDiv (↑C) (HPow.hPow …
  -/
  have hn' : 0 < (n : ℝ) ^ s.re := Real.rpow_pos_of_pos (Nat.cast_pos.mpr hn) _
  simp_rw [term_of_ne_zero hn.ne', norm_div, norm_natCast_cpow_of_pos hn, div_le_iff₀ hn',
    norm_eq_abs (C : ℂ), abs_ofReal, _root_.abs_of_nonneg hC₀, div_eq_mul_inv, mul_assoc,
    ← Real.rpow_neg <| Nat.cast_nonneg _, ← Real.rpow_add <| Nat.cast_pos.mpr hn]
  /-
    case intro.inr
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    hC₀ : LE.le 0 C
    hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
    n : Nat
    hn : GT.gt n 0
    hn' : LT.lt 0 (HPow.hPow (↑n) s.re)
    ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n) (HAdd.hAdd (Neg.neg (HA …
  -/
  simp only [add_re, sub_re, one_re, ofReal_re, neg_add_rev, neg_sub, neg_add_cancel_right]
  /-
    case intro.inr
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    C : Real
    hC : ∀ (n : Nat), Ne n 0 → LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n …
    hC₀ : LE.le 0 C
    hsum : Summable fun n => Norm.norm (HDiv.hDiv (↑C) (HPow.hPow (↑n) (HAdd.hAdd  …
    n : Nat
    hn : GT.gt n 0
    hn' : LT.lt 0 (HPow.hPow (↑n) s.re)
    ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow (↑n) (HSub.hSub x 1)))
  -/
  exact hC n <| Nat.pos_iff_ne_zero.mp hn
  /-
    🎉 no goals
  -/


open Filter Finset Real Nat in
/-- If `f = O(n^(x-1))` and `re s > x`, then the `LSeries` of `f` is summable at `s`. -/
lemma LSeriesSummable_of_isBigO_rpow {f : ℕ → ℂ} {x : ℝ} {s : ℂ} (hs : x < s.re)
    (h : f =O[atTop] fun n ↦ (n : ℝ) ^ (x - 1)) :
    LSeriesSummable f s := by
  /-
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
    ⊢ LSeriesSummable f s
  -/
  obtain ⟨C, hC⟩ := Asymptotics.isBigO_iff.mp h
  /-
    case intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
    C : Real
    hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
    ⊢ LSeriesSummable f s
  -/
  obtain ⟨m, hm⟩ := eventually_atTop.mp hC
  let C' := max C (max' (insert 0 (image (fun n : ℕ ↦ ‖f n‖ / (n : ℝ) ^ (x - 1)) (range m)))
    (insert_nonempty 0 _))
  /-
    case intro.intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
    C : Real
    hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
    m : Nat
    hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
    C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
    ⊢ LSeriesSummable f s
  -/
  have hC'₀ : 0 ≤ C' := (le_max' _ _ (mem_insert.mpr (Or.inl rfl))).trans <| le_max_right ..
  /-
    case intro.intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
    C : Real
    hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
    m : Nat
    hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
    C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
    hC'₀ : LE.le 0 C'
    ⊢ LSeriesSummable f s
  -/
  have hCC' : C ≤ C' := le_max_left ..
  /-
    case intro.intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
    C : Real
    hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
    m : Nat
    hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
    C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
    hC'₀ : LE.le 0 C'
    hCC' : LE.le C C'
    ⊢ LSeriesSummable f s
  -/
  refine LSeriesSummable_of_le_const_mul_rpow hs ⟨C', fun n hn₀ ↦ ?_⟩
  /-
    case intro.intro
    f : Nat → Complex
    x : Real
    s : Complex
    hs : LT.lt x s.re
    h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
    C : Real
    hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
    m : Nat
    hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
    C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
    hC'₀ : LE.le 0 C'
    hCC' : LE.le C C'
    n : Nat
    hn₀ : Ne n 0
    ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C' (HPow.hPow (↑n) (HSub.hSub x 1)))
  -/
  rcases le_or_lt m n with hn | hn
    /-
      case intro.intro.inl
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀ : Ne n 0
      hn : LE.le m n
      ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C' (HPow.hPow (↑n) (HSub.hSub x 1)))
    -/
  · refine (hm n hn).trans ?_
    /-
      case intro.intro.inl
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀ : Ne n 0
      hn : LE.le m n
      ⊢ LE.le (HMul.hMul C (Norm.norm (HPow.hPow (↑n) (HSub.hSub x 1)))) (HMul.hMul  …
    -/
    have hn₀ : (0 : ℝ) ≤ n := cast_nonneg _
    /-
      case intro.intro.inl
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀✝ : Ne n 0
      hn : LE.le m n
      hn₀ : LE.le 0 ↑n
      ⊢ LE.le (HMul.hMul C (Norm.norm (HPow.hPow (↑n) (HSub.hSub x 1)))) (HMul.hMul  …
    -/
    gcongr
    /-
      case intro.intro.inl.h₂
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀✝ : Ne n 0
      hn : LE.le m n
      hn₀ : LE.le 0 ↑n
      ⊢ LE.le (Norm.norm (HPow.hPow (↑n) (HSub.hSub x 1))) (HPow.hPow (↑n) (HSub.hSu …
    -/
    rw [Real.norm_eq_abs, abs_rpow_of_nonneg hn₀, _root_.abs_of_nonneg hn₀]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀ : Ne n 0
      hn : LT.lt n m
      ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C' (HPow.hPow (↑n) (HSub.hSub x 1)))
    -/
  · have hn' : 0 < n := Nat.pos_of_ne_zero hn₀
    /-
      case intro.intro.inr
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀ : Ne n 0
      hn : LT.lt n m
      hn' : LT.lt 0 n
      ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C' (HPow.hPow (↑n) (HSub.hSub x 1)))
    -/
    refine (div_le_iff₀ <| rpow_pos_of_pos (cast_pos.mpr hn') _).mp ?_
    /-
      case intro.intro.inr
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀ : Ne n 0
      hn : LT.lt n m
      hn' : LT.lt 0 n
      ⊢ LE.le (HDiv.hDiv (Norm.norm (f n)) (HPow.hPow (↑n) (HSub.hSub x 1))) C'
    -/
    refine (le_max' _ _ <| mem_insert_of_mem ?_).trans <| le_max_right ..
    /-
      case intro.intro.inr
      f : Nat → Complex
      x : Real
      s : Complex
      hs : LT.lt x s.re
      h : Asymptotics.IsBigO Filter.atTop f fun n => HPow.hPow (↑n) (HSub.hSub x 1)
      C : Real
      hC : Filter.Eventually (fun x_1 => LE.le (Norm.norm (f x_1)) (HMul.hMul C (Nor …
      m : Nat
      hm : ∀ (b : Nat), GE.ge b m → LE.le (Norm.norm (f b)) (HMul.hMul C (Norm.norm  …
      C' : Real := Max.max C ((Insert.insert 0 (Finset.image (fun n => HDiv.hDiv (No …
      hC'₀ : LE.le 0 C'
      hCC' : LE.le C C'
      n : Nat
      hn₀ : Ne n 0
      hn : LT.lt n m
      hn' : LT.lt 0 n
      ⊢ Membership.mem (Finset.image (fun n => HDiv.hDiv (Norm.norm (f n)) (HPow.hPo …
    -/
    exact mem_image.mpr ⟨n, mem_range.mpr hn, rfl⟩
    /-
      🎉 no goals
    -/


/-- If `f` is bounded, then its `LSeries` is summable at `s` when `re s > 1`. -/
theorem LSeriesSummable_of_bounded_of_one_lt_re {f : ℕ → ℂ} {m : ℝ}
    (h : ∀ n ≠ 0, Complex.abs (f n) ≤ m) {s : ℂ} (hs : 1 < s.re) :
    LSeriesSummable f s := by
  /-
    f : Nat → Complex
    m : Real
    h : ∀ (n : Nat), Ne n 0 → LE.le (Complex.abs (f n)) m
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ LSeriesSummable f s
  -/
  refine LSeriesSummable_of_le_const_mul_rpow hs ⟨m, fun n hn ↦ ?_⟩
  /-
    f : Nat → Complex
    m : Real
    h : ∀ (n : Nat), Ne n 0 → LE.le (Complex.abs (f n)) m
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    hn : Ne n 0
    ⊢ LE.le (Norm.norm (f n)) (HMul.hMul m (HPow.hPow (↑n) (HSub.hSub 1 1)))
  -/
  simp only [norm_eq_abs, sub_self, Real.rpow_zero, mul_one, h n hn]
  /-
    🎉 no goals
  -/


/-- If `f` is bounded, then its `LSeries` is summable at `s : ℝ` when `s > 1`. -/
theorem LSeriesSummable_of_bounded_of_one_lt_real {f : ℕ → ℂ} {m : ℝ}
    (h : ∀ n ≠ 0, Complex.abs (f n) ≤ m) {s : ℝ} (hs : 1 < s) :
    LSeriesSummable f s :=
                                                  /-
                                                    f : Nat → Complex
                                                    m : Real
                                                    h : ∀ (n : Nat), Ne n 0 → LE.le (Complex.abs (f n)) m
                                                    s : Real
                                                    hs : LT.lt 1 s
                                                    ⊢ LT.lt 1 (↑s).re
                                                  -/
  LSeriesSummable_of_bounded_of_one_lt_re h <| by simp only [ofReal_re, hs]
                                                  /-
                                                    🎉 no goals
                                                  -/

