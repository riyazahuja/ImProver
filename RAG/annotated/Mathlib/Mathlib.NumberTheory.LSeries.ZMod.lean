/-- If `Φ` is a periodic function, then the L-series of `Φ` converges for `1 < re s`. -/
lemma LSeriesSummable_of_one_lt_re (Φ : ZMod N → ℂ) {s : ℂ} (hs : 1 < re s) :
    LSeriesSummable (Φ ·) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ LSeriesSummable (fun x => Φ ↑x) s
  -/
  let c := max' _ <| univ_nonempty.image (Complex.abs ∘ Φ)
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : LT.lt 1 s.re
    c : Real := (Finset.image (Function.comp (⇑Complex.abs) Φ) Finset.univ).max' ⋯
    ⊢ LSeriesSummable (fun x => Φ ↑x) s
  -/
  refine LSeriesSummable_of_bounded_of_one_lt_re (fun n _ ↦ le_max' _ _ ?_) (m := c) hs
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : LT.lt 1 s.re
    c : Real := (Finset.image (Function.comp (⇑Complex.abs) Φ) Finset.univ).max' ⋯
    n : Nat
    x✝ : Ne n 0
    ⊢ Membership.mem (Finset.image (Function.comp (⇑Complex.abs) Φ) Finset.univ) ( …
  -/
  exact mem_image_of_mem _ (mem_univ _)
  /-
    🎉 no goals
  -/


/--
The unique meromorphic function `ℂ → ℂ` which agrees with `∑' n : ℕ, Φ n / n ^ s` wherever the
latter is convergent. This is constructed as a linear combination of Hurwitz zeta functions.

Note that this is not the same as `LSeries Φ`: they agree in the convergence range, but
`LSeries Φ s` is defined to be `0` if `re s ≤ 1`.
 -/
noncomputable def LFunction (Φ : ZMod N → ℂ) (s : ℂ) : ℂ :=
  N ^ (-s) * ∑ j : ZMod N, Φ j * hurwitzZeta (toAddCircle j) s


/-- The L-function of a function on `ZMod 1` is a scalar multiple of the Riemann zeta function. -/
lemma LFunction_modOne_eq (Φ : ZMod 1 → ℂ) (s : ℂ) :
    LFunction Φ s = Φ 0 * riemannZeta s := by
  simp only [LFunction, Nat.cast_one, one_cpow, ← singleton_eq_univ (0 : ZMod 1), sum_singleton,
    map_zero, hurwitzZeta_zero, one_mul]


/-- For `1 < re s` the congruence L-function agrees with the sum of the Dirichlet series. -/
lemma LFunction_eq_LSeries (Φ : ZMod N → ℂ) {s : ℂ} (hs : 1 < re s) :
    LFunction Φ s = LSeries (Φ ·) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (ZMod.LFunction Φ s) (LSeries (fun x => Φ ↑x) s)
  -/
  rw [LFunction, LSeries, mul_sum, Nat.sumByResidueClasses (LSeriesSummable_of_one_lt_re Φ hs) N]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (HMul.hM …
  -/
  congr 1 with j
  have : (j.val / N : ℝ) ∈ Set.Icc 0 1 := mem_Icc.mpr ⟨by positivity,
    (div_le_one (Nat.cast_pos.mpr <| NeZero.pos _)).mpr <| Nat.cast_le.mpr (val_lt j).le⟩
  rw [toAddCircle_apply, ← (hasSum_hurwitzZeta_of_one_lt_re this hs).tsum_eq, ← mul_assoc,
    ← tsum_mul_left]
  /-
    case e_f.h
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : LT.lt 1 s.re
    j : ZMod N
    this : Membership.mem (Set.Icc 0 1) (HDiv.hDiv ↑j.val ↑N)
    ⊢ Eq (tsum fun x => HMul.hMul (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (Φ j)) ( …
  -/
  congr 1 with m
  -- The following manipulation is slightly delicate because `(x * y) ^ s = x ^ s * y ^ s` is
  -- false for general complex `x`, `y`, but it is true if `x` and `y` are non-negative reals, so
  -- we have to carefully juggle coercions `ℕ → ℝ → ℂ`.
  calc N ^ (-s) * Φ j * (1 / (m + (j.val / N : ℝ)) ^ s)
  _ = Φ j * (N ^ (-s) * (1 / (m + (j.val / N : ℝ)) ^ s)) := by
    rw [← mul_assoc, mul_comm _ (Φ _)]
  _ = Φ j * (1 / (N : ℝ) ^ s * (1 / ((j.val + N * m) / N : ℝ) ^ s)) := by
    simp only [cpow_neg, ← one_div, ofReal_div, ofReal_natCast, add_comm, add_div, ofReal_add,
      ofReal_mul, mul_div_cancel_left₀ (m : ℂ) (Nat.cast_ne_zero.mpr (NeZero.ne N))]
  _ = Φ j / ((N : ℝ) * ((j.val + N * m) / N : ℝ)) ^ s := by -- this is the delicate step!
    rw [one_div_mul_one_div, mul_one_div, mul_cpow_ofReal_nonneg] <;> positivity
  _ = Φ j / (N * (j.val + N * m) / N) ^ s := by
    simp only [ofReal_natCast, ofReal_div, ofReal_add, ofReal_mul, mul_div_assoc]
  _ = Φ j / (j.val + N * m) ^ s := by
    rw [mul_div_cancel_left₀ _ (Nat.cast_ne_zero.mpr (NeZero.ne N))]
  _ = Φ ↑(j.val + N * m) / (↑(j.val + N * m)) ^ s := by
    simp only [Nat.cast_add, Nat.cast_mul, natCast_zmod_val, natCast_self, zero_mul, add_zero]
  _ = LSeries.term (Φ ·) s (j.val + N * m) := by
    rw [LSeries.term_of_ne_zero' (ne_zero_of_one_lt_re hs)]


lemma differentiableAt_LFunction (Φ : ZMod N → ℂ) (s : ℂ) (hs : s ≠ 1 ∨ ∑ j, Φ j = 0) :
    DifferentiableAt ℂ (LFunction Φ) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
    ⊢ DifferentiableAt Complex (ZMod.LFunction Φ) s
  -/
  refine .mul (by fun_prop) ?_
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
    ⊢ DifferentiableAt Complex (fun y => Finset.univ.sum fun j => HMul.hMul (Φ j)  …
  -/
  rcases ne_or_eq s 1 with hs' | rfl
    /-
      case inl
      N : Nat
      inst✝ : NeZero N
      Φ : ZMod N → Complex
      s : Complex
      hs : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
      hs' : Ne s 1
      ⊢ DifferentiableAt Complex (fun y => Finset.univ.sum fun j => HMul.hMul (Φ j)  …
    -/
  · exact .sum fun j _ ↦ (differentiableAt_hurwitzZeta _ hs').const_mul _
    /-
      🎉 no goals
    -/
  · have := DifferentiableAt.sum (u := univ) fun j _ ↦
      (differentiableAt_hurwitzZeta_sub_one_div (toAddCircle j)).const_mul (Φ j)
    /-
      case inr
      N : Nat
      inst✝ : NeZero N
      Φ : ZMod N → Complex
      hs : Or (Ne 1 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
      this : DifferentiableAt Complex (fun y => Finset.univ.sum fun i => HMul.hMul ( …
      ⊢ DifferentiableAt Complex (fun y => Finset.univ.sum fun j => HMul.hMul (Φ j)  …
    -/
    simpa only [mul_sub, sum_sub_distrib, ← sum_mul, hs.neg_resolve_left rfl, zero_mul, sub_zero]
    /-
      🎉 no goals
    -/


lemma differentiable_LFunction_of_sum_zero {Φ : ZMod N → ℂ} (hΦ : ∑ j, Φ j = 0) :
    Differentiable ℂ (LFunction Φ) :=
  fun s ↦ differentiableAt_LFunction Φ s (Or.inr hΦ)


/-- The L-function of `Φ` has a residue at `s = 1` equal to the average value of `Φ`. -/
lemma LFunction_residue_one (Φ : ZMod N → ℂ) :
    Tendsto (fun s ↦ (s - 1) * LFunction Φ s) (𝓝[≠] 1) (𝓝 (∑ j, Φ j / N)) := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (ZMod.LFunction Φ s)) (nh …
  -/
  simp only [sum_div, LFunction, mul_sum]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    ⊢ Filter.Tendsto (fun s => Finset.univ.sum fun i => HMul.hMul (HSub.hSub s 1)  …
  -/
  refine tendsto_finset_sum _ fun j _ ↦ ?_
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    j : ZMod N
    x✝ : Membership.mem Finset.univ j
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (HMul.hMul (HPow.hPow (↑N …
  -/
  rw [(by ring : Φ j / N = Φ j * (1 / N * 1)), one_div, ← cpow_neg_one]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    j : ZMod N
    x✝ : Membership.mem Finset.univ j
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (HMul.hMul (HPow.hPow (↑N …
  -/
  simp only [show ∀ a b c d : ℂ, a * (b * (c * d)) = c * (b * (a * d)) by intros; ring]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    j : ZMod N
    x✝ : Membership.mem Finset.univ j
    ⊢ Filter.Tendsto (fun s => HMul.hMul (Φ j) (HMul.hMul (HPow.hPow (↑N) (Neg.neg …
  -/
  refine tendsto_const_nhds.mul (.mul ?_ <| hurwitzZeta_residue_one _)
  exact ((continuous_neg.const_cpow (Or.inl <| NeZero.ne _)).tendsto _).mono_left
    nhdsWithin_le_nhds


local notation "𝕖" => stdAddChar


/--
The `LFunction` of the function `x ↦ e (j * x)`, where `e : ZMod N → ℂ` is the standard additive
character, agrees with `expZeta (j / N)` on `1 < re s`. Private since it is a stepping-stone to
the more general result `LFunction_stdAddChar_eq_expZeta` below.
-/
private lemma LFunction_stdAddChar_eq_expZeta_of_one_lt_re (j : ZMod N) {s : ℂ} (hs : 1 < s.re) :
    LFunction (fun k ↦ 𝕖 (j * k)) s = expZeta (ZMod.toAddCircle j) s := by
  rw [toAddCircle_apply, ← (hasSum_expZeta_of_one_lt_re (j.val / N) hs).tsum_eq,
    LFunction_eq_LSeries _ hs, LSeries]
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (tsum fun n => LSeries.term (fun x => ZMod.stdAddChar (HMul.hMul j ↑x)) s …
  -/
  congr 1 with n
  rw [LSeries.term_of_ne_zero' (ne_zero_of_one_lt_re hs), ofReal_div, ofReal_natCast,
    ofReal_natCast, mul_assoc, div_mul_eq_mul_div, stdAddChar_apply]
  /-
    case e_f.h
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv (↑(ZMod.toCircle (HMul.hMul j ↑n))) (HPow.hPow (↑n) s)) (HDiv. …
  -/
  have := ZMod.toCircle_intCast (N := N) (j.val * n)
  /-
    case e_f.h
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    this : Eq (↑(ZMod.toCircle ↑(HMul.hMul ↑j.val ↑n))) (Complex.exp (HDiv.hDiv (H …
    ⊢ Eq (HDiv.hDiv (↑(ZMod.toCircle (HMul.hMul j ↑n))) (HPow.hPow (↑n) s)) (HDiv. …
  -/
  conv_rhs at this => rw [Int.cast_mul, Int.cast_natCast, Int.cast_natCast, mul_div_assoc]
  /-
    case e_f.h
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    this : Eq (↑(ZMod.toCircle ↑(HMul.hMul ↑j.val ↑n))) (Complex.exp (HMul.hMul (H …
    ⊢ Eq (HDiv.hDiv (↑(ZMod.toCircle (HMul.hMul j ↑n))) (HPow.hPow (↑n) s)) (HDiv. …
  -/
  rw [← this, Int.cast_mul, Int.cast_natCast, Int.cast_natCast, natCast_zmod_val]
  /-
    🎉 no goals
  -/


/--
The `LFunction` of the function `x ↦ e (j * x)`, where `e : ZMod N → ℂ` is the standard additive
character, is `expZeta (j / N)`.

Note this is not at all obvious from the definitions, and we prove it by analytic continuation
from the convergence range.
-/
lemma LFunction_stdAddChar_eq_expZeta (j : ZMod N) (s : ℂ) (hjs : j ≠ 0 ∨ s ≠ 1) :
    LFunction (fun k ↦ 𝕖 (j * k)) s = expZeta (ZMod.toAddCircle j) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  let U := if j = 0 then {z : ℂ | z ≠ 1} else univ -- region of analyticity of both functions
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  let V := {z : ℂ | 1 < re z} -- convergence region
  have hUo : IsOpen U := by
    by_cases h : j = 0
    · simpa only [h, ↓reduceIte, U] using isOpen_compl_singleton
    · simp only [h, ↓reduceIte, isOpen_univ, U]
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  let f := LFunction (fun k ↦ stdAddChar (j * k))
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  let g := expZeta (toAddCircle j)
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    g : Complex → Complex := HurwitzZeta.expZeta (ZMod.toAddCircle j)
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  have hU {u} : u ∈ U ↔ u ≠ 1 ∨ j ≠ 0 := by simp only [mem_ite_univ_right, U]; tauto
  -- hypotheses for uniqueness of analytic continuation
  have hf : AnalyticOnNhd ℂ f U := by
    refine DifferentiableOn.analyticOnNhd (fun u hu ↦ ?_) hUo
    refine (differentiableAt_LFunction _ _ ((hU.mp hu).imp_right fun h ↦ ?_)).differentiableWithinAt
    simp only [mul_comm j, AddChar.sum_mulShift _ (isPrimitive_stdAddChar _), h,
      ↓reduceIte, CharP.cast_eq_zero, or_true]
  have hg : AnalyticOnNhd ℂ g U := by
    refine DifferentiableOn.analyticOnNhd (fun u hu ↦ ?_) hUo
    refine (differentiableAt_expZeta _ _ ((hU.mp hu).imp_right fun h ↦ ?_)).differentiableWithinAt
    rwa [ne_eq, toAddCircle_eq_zero]
  have hUc : IsPreconnected U := by
    by_cases h : j = 0
    · simpa only [h, ↓reduceIte, U] using
        (isConnected_compl_singleton_of_one_lt_rank (by simp) _).isPreconnected
    · simpa only [h, ↓reduceIte, U] using isPreconnected_univ
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    g : Complex → Complex := HurwitzZeta.expZeta (ZMod.toAddCircle j)
    hU : ∀ {u : Complex}, Iff (Membership.mem U u) (Or (Ne u 1) (Ne j 0))
    hf : AnalyticOnNhd Complex f U
    hg : AnalyticOnNhd Complex g U
    hUc : IsPreconnected U
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  have hV : V ∈ 𝓝 2 := (continuous_re.isOpen_preimage _ isOpen_Ioi).mem_nhds (by simp)
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    g : Complex → Complex := HurwitzZeta.expZeta (ZMod.toAddCircle j)
    hU : ∀ {u : Complex}, Iff (Membership.mem U u) (Or (Ne u 1) (Ne j 0))
    hf : AnalyticOnNhd Complex f U
    hg : AnalyticOnNhd Complex g U
    hUc : IsPreconnected U
    hV : Membership.mem (nhds 2) V
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  have hUmem : 2 ∈ U := by simp [U]
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    g : Complex → Complex := HurwitzZeta.expZeta (ZMod.toAddCircle j)
    hU : ∀ {u : Complex}, Iff (Membership.mem U u) (Or (Ne u 1) (Ne j 0))
    hf : AnalyticOnNhd Complex f U
    hg : AnalyticOnNhd Complex g U
    hUc : IsPreconnected U
    hV : Membership.mem (nhds 2) V
    hUmem : Membership.mem U 2
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  have hUmem' : s ∈ U := hU.mpr hjs.symm
  -- apply uniqueness result
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    g : Complex → Complex := HurwitzZeta.expZeta (ZMod.toAddCircle j)
    hU : ∀ {u : Complex}, Iff (Membership.mem U u) (Or (Ne u 1) (Ne j 0))
    hf : AnalyticOnNhd Complex f U
    hg : AnalyticOnNhd Complex g U
    hUc : IsPreconnected U
    hV : Membership.mem (nhds 2) V
    hUmem : Membership.mem U 2
    hUmem' : Membership.mem U s
    ⊢ Eq (ZMod.LFunction (fun k => ZMod.stdAddChar (HMul.hMul j k)) s) (HurwitzZet …
  -/
  refine hf.eqOn_of_preconnected_of_eventuallyEq hg hUc hUmem ?_ hUmem'
  -- now remains to prove equality on `1 < re s`
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    s : Complex
    hjs : Or (Ne j 0) (Ne s 1)
    U : Set Complex := ite (Eq j 0) (setOf fun z => Ne z 1) Set.univ
    V : Set Complex := setOf fun z => LT.lt 1 z.re
    hUo : IsOpen U
    f : Complex → Complex := ZMod.LFunction fun k => ZMod.stdAddChar (HMul.hMul j k)
    g : Complex → Complex := HurwitzZeta.expZeta (ZMod.toAddCircle j)
    hU : ∀ {u : Complex}, Iff (Membership.mem U u) (Or (Ne u 1) (Ne j 0))
    hf : AnalyticOnNhd Complex f U
    hg : AnalyticOnNhd Complex g U
    hUc : IsPreconnected U
    hV : Membership.mem (nhds 2) V
    hUmem : Membership.mem U 2
    hUmem' : Membership.mem U s
    ⊢ (nhds 2).EventuallyEq f g
  -/
  filter_upwards [hV] with z using LFunction_stdAddChar_eq_expZeta_of_one_lt_re _
  /-
    🎉 no goals
  -/


/-- Explicit formula for the L-function of `𝓕 Φ`, where `𝓕` is the discrete Fourier transform. -/
lemma LFunction_dft (Φ : ZMod N → ℂ) {s : ℂ} (hs : Φ 0 = 0 ∨ s ≠ 1) :
    LFunction (𝓕 Φ) s = ∑ j : ZMod N, Φ j * expZeta (toAddCircle (-j)) s := by
  have (j : ZMod N) : Φ j * LFunction (fun k ↦ 𝕖 (-j * k)) s =
      Φ j * expZeta (toAddCircle (-j)) s := by
    by_cases h : -j ≠ 0 ∨ s ≠ 1
    · rw [LFunction_stdAddChar_eq_expZeta _ _ h]
    · simp only [neg_ne_zero, not_or, not_not] at h
      rw [h.1, show Φ 0 = 0 by tauto, zero_mul, zero_mul]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (ZMod.LFunction (fun k => ZMod.stdA …
    ⊢ Eq (ZMod.LFunction (ZMod.dft Φ) s) (Finset.univ.sum fun j => HMul.hMul (Φ j) …
  -/
  simp only [LFunction, ← this, mul_sum]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (ZMod.LFunction (fun k => ZMod.stdA …
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (HMul.hM …
  -/
  rw [dft_def, sum_comm]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (ZMod.LFunction (fun k => ZMod.stdA …
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (HMul.hM …
  -/
  simp only [sum_mul, mul_sum, Circle.smul_def, smul_eq_mul, stdAddChar_apply, ← mul_assoc]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (ZMod.LFunction (fun k => ZMod.stdA …
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HMul.hMul (HMul.hMul …
  -/
  congr 1 with j
  /-
    case e_f.h
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (ZMod.LFunction (fun k => ZMod.stdA …
    j : ZMod N
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (↑N) …
  -/
  congr 1 with k
  /-
    case e_f.h.e_f.h
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (ZMod.LFunction (fun k => ZMod.stdA …
    j k : ZMod N
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) ↑(ZMod.toCi …
  -/
  rw [mul_assoc (Φ _), mul_comm (Φ _), neg_mul]
  /-
    🎉 no goals
  -/


/-- Functional equation for `ZMod` L-functions, in terms of discrete Fourier transform. -/
theorem LFunction_one_sub (Φ : ZMod N → ℂ) {s : ℂ}
    (hs : ∀ (n : ℕ), s ≠ -n) (hs' : Φ 0 = 0 ∨ s ≠ 1) :
    LFunction Φ (1 - s) = N ^ (s - 1) * (2 * π) ^ (-s) * Gamma s *
      (cexp (π * I * s / 2) * LFunction (𝓕 Φ) s
       + cexp (-π * I * s / 2) * LFunction (𝓕 fun x ↦ Φ (-x)) s) := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Eq (Φ 0) 0) (Ne s 1)
    ⊢ Eq (ZMod.LFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HMul.hMul (HPow …
  -/
  rw [LFunction]
  have (j : ZMod N) :  Φ j * hurwitzZeta (toAddCircle j) (1 - s) = Φ j *
      ((2 * π) ^ (-s) * Gamma s * (cexp (-π * I * s / 2) *
      expZeta (toAddCircle j) s + cexp (π * I * s / 2) * expZeta (-toAddCircle j) s)) := by
    rcases eq_or_ne j 0 with rfl | hj
    · rcases hs' with hΦ | hs'
      · simp only [hΦ, zero_mul]
      · rw [hurwitzZeta_one_sub _ hs (Or.inr hs')]
    · rw [hurwitzZeta_one_sub _ hs (Or.inl <| toAddCircle_eq_zero.not.mpr hj)]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (HurwitzZeta.hurwitzZeta (ZMod.toAd …
    ⊢ Eq (HMul.hMul (HPow.hPow (↑N) (Neg.neg (HSub.hSub 1 s))) (Finset.univ.sum fu …
  -/
  simp only [this, mul_assoc _ _ (Gamma s)]
  -- get rid of Gamma terms and power of N
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (HurwitzZeta.hurwitzZeta (ZMod.toAd …
    ⊢ Eq (HMul.hMul (HPow.hPow (↑N) (Neg.neg (HSub.hSub 1 s))) (Finset.univ.sum fu …
  -/
  generalize (2 * π) ^ (-s) * Gamma s = C
  simp_rw [← mul_assoc, mul_comm _ C, mul_assoc, ← mul_sum, ← mul_assoc, mul_comm _ C, mul_assoc,
    neg_sub]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (HurwitzZeta.hurwitzZeta (ZMod.toAd …
    C : Complex
    ⊢ Eq (HMul.hMul C (HMul.hMul (HPow.hPow (↑N) (HSub.hSub s 1)) (Finset.univ.sum …
  -/
  congr 2
  -- now gather sum terms
  /-
    case e_a.e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (HurwitzZeta.hurwitzZeta (ZMod.toAd …
    C : Complex
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (Φ x) (HAdd.hAdd (HMul.hMul (Complex. …
  -/
  rw [LFunction_dft _ hs', LFunction_dft _ (hs'.imp_left <| by simp only [neg_zero, imp_self])]
  /-
    case e_a.e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Eq (Φ 0) 0) (Ne s 1)
    this : ∀ (j : ZMod N), Eq (HMul.hMul (Φ j) (HurwitzZeta.hurwitzZeta (ZMod.toAd …
    C : Complex
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (Φ x) (HAdd.hAdd (HMul.hMul (Complex. …
  -/
  conv_rhs => enter [2, 2]; rw [← (Equiv.neg _).sum_comp _ _ (by simp), Equiv.neg_apply]
  simp_rw [neg_neg, mul_sum, ← sum_add_distrib, ← mul_assoc, mul_comm _ (Φ _), mul_assoc,
    ← mul_add, map_neg, add_comm]


lemma LFunction_def_even (hΦ : Φ.Even) (s : ℂ) :
    LFunction Φ s = N ^ (-s) * ∑ j : ZMod N, Φ j * hurwitzZetaEven (toAddCircle j) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    ⊢ Eq (ZMod.LFunction Φ s) (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (Finset.univ …
  -/
  simp only [LFunction, hurwitzZeta, mul_add (Φ _), sum_add_distrib]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    ⊢ Eq (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (HAdd.hAdd (Finset.univ.sum fun x …
  -/
  congr 1
  /-
    case e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun x => HMul.hMul (Φ x) (HurwitzZeta.hurwitz …
  -/
  simp only [add_right_eq_self, ← neg_eq_self ℂ, ← sum_neg_distrib]
  /-
    case e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    ⊢ Eq (Finset.univ.sum fun x => Neg.neg (HMul.hMul (Φ x) (HurwitzZeta.hurwitzZe …
  -/
  refine Fintype.sum_equiv (.neg _) _ _ fun i ↦ ?_
  /-
    case e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    i : ZMod N
    ⊢ Eq (Neg.neg (HMul.hMul (Φ i) (HurwitzZeta.hurwitzZetaOdd (ZMod.toAddCircle i …
  -/
  simp only [Equiv.neg_apply, hΦ i, map_neg, hurwitzZetaOdd_neg, mul_neg]
  /-
    🎉 no goals
  -/


lemma LFunction_def_odd (hΦ : Φ.Odd) (s : ℂ) :
    LFunction Φ s = N ^ (-s) * ∑ j : ZMod N, Φ j * hurwitzZetaOdd (toAddCircle j) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    ⊢ Eq (ZMod.LFunction Φ s) (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (Finset.univ …
  -/
  simp only [LFunction, hurwitzZeta, mul_add (Φ _), sum_add_distrib]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    ⊢ Eq (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (HAdd.hAdd (Finset.univ.sum fun x …
  -/
  congr 1
  /-
    case e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun x => HMul.hMul (Φ x) (HurwitzZeta.hurwitz …
  -/
  simp only [add_left_eq_self, ← neg_eq_self ℂ, ← sum_neg_distrib]
  /-
    case e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    ⊢ Eq (Finset.univ.sum fun x => Neg.neg (HMul.hMul (Φ x) (HurwitzZeta.hurwitzZe …
  -/
  refine Fintype.sum_equiv (.neg _) _ _ fun i ↦ ?_
  /-
    case e_a
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    i : ZMod N
    ⊢ Eq (Neg.neg (HMul.hMul (Φ i) (HurwitzZeta.hurwitzZetaEven (ZMod.toAddCircle  …
  -/
  simp only [Equiv.neg_apply, hΦ i, map_neg, hurwitzZetaEven_neg, neg_mul]
  /-
    🎉 no goals
  -/


/-- Explicit formula for `LFunction Φ 0` when `Φ` is even. -/
@[simp] lemma LFunction_apply_zero_of_even (hΦ : Φ.Even) :
    LFunction Φ 0 = -Φ 0 / 2 := by
  simp only [LFunction_def_even hΦ, neg_zero, cpow_zero, hurwitzZetaEven_apply_zero,
    toAddCircle_eq_zero, mul_ite, mul_div, mul_neg_one, mul_zero, sum_ite_eq', Finset.mem_univ,
    ↓reduceIte, one_mul]


/-- The L-function of an even function vanishes at negative even integers. -/
@[simp] lemma LFunction_neg_two_mul_nat_add_one (hΦ : Φ.Even) (n : ℕ) :
    LFunction Φ (-(2 * (n + 1))) = 0 := by
  simp only [LFunction_def_even hΦ, hurwitzZetaEven_neg_two_mul_nat_add_one, mul_zero,
    sum_const_zero, ← neg_mul]


/-- The L-function of an odd function vanishes at negative odd integers. -/
@[simp] lemma LFunction_neg_two_mul_nat_sub_one (hΦ : Φ.Odd) (n : ℕ) :
    LFunction Φ (-(2 * n) - 1) = 0 := by
  simp only [LFunction_def_odd hΦ, hurwitzZetaOdd_neg_two_mul_nat_sub_one, mul_zero, ← neg_mul,
    sum_const_zero]


/--
The completed L-function of a function `Φ : ZMod N → ℂ`.

This is only mathematically meaningful if `Φ` is either even, or odd; here we extend this to all `Φ`
by linearity.
-/
noncomputable def completedLFunction (Φ : ZMod N → ℂ) (s : ℂ) : ℂ :=
  N ^ (-s) * ∑ j, Φ j * completedHurwitzZetaEven (toAddCircle j) s
  + N ^ (-s) * ∑ j, Φ j * completedHurwitzZetaOdd (toAddCircle j) s


@[simp] lemma completedLFunction_zero (s : ℂ) : completedLFunction (0 : ZMod N → ℂ) s = 0 := by
  /-
    N : Nat
    inst✝ : NeZero N
    s : Complex
    ⊢ Eq (ZMod.completedLFunction 0 s) 0
  -/
  simp only [completedLFunction, Pi.zero_apply, zero_mul, sum_const_zero, mul_zero, zero_add]
  /-
    🎉 no goals
  -/


lemma completedLFunction_const_mul (a : ℂ) (Φ : ZMod N → ℂ) (s : ℂ) :
    completedLFunction (fun j ↦ a * Φ j) s = a * completedLFunction Φ s := by
  /-
    N : Nat
    inst✝ : NeZero N
    a : Complex
    Φ : ZMod N → Complex
    s : Complex
    ⊢ Eq (ZMod.completedLFunction (fun j => HMul.hMul a (Φ j)) s) (HMul.hMul a (ZM …
  -/
  simp only [completedLFunction, mul_add, mul_sum]
  /-
    N : Nat
    inst✝ : NeZero N
    a : Complex
    Φ : ZMod N → Complex
    s : Complex
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HMul.hMul (HPow.hPow (↑N) (Neg.neg s …
  -/
                   /-
                     🎉 no goals
                   -/
  congr with i <;> ring
                   /-
                     🎉 no goals
                   -/


lemma completedLFunction_def_even (hΦ : Φ.Even) (s : ℂ) :
    completedLFunction Φ s = N ^ (-s) * ∑ j, Φ j * completedHurwitzZetaEven (toAddCircle j) s := by
  suffices ∑ j, Φ j * completedHurwitzZetaOdd (toAddCircle j) s = 0 by
    rw [completedLFunction, this, mul_zero, add_zero]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul (Φ j) (HurwitzZeta.completedHurwitzZe …
  -/
  refine (hΦ.mul_odd fun j ↦ ?_).sum_eq_zero
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    j : ZMod N
    ⊢ Eq (HurwitzZeta.completedHurwitzZetaOdd (ZMod.toAddCircle (Neg.neg j)) s) (N …
  -/
  rw [map_neg, completedHurwitzZetaOdd_neg]
  /-
    🎉 no goals
  -/


lemma completedLFunction_def_odd (hΦ : Φ.Odd) (s : ℂ) :
    completedLFunction Φ s = N ^ (-s) * ∑ j, Φ j * completedHurwitzZetaOdd (toAddCircle j) s := by
  suffices ∑ j, Φ j * completedHurwitzZetaEven (toAddCircle j) s = 0 by
    rw [completedLFunction, this, mul_zero, zero_add]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul (Φ j) (HurwitzZeta.completedHurwitzZe …
  -/
  refine (hΦ.mul_even fun j ↦ ?_).sum_eq_zero
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    j : ZMod N
    ⊢ Eq (HurwitzZeta.completedHurwitzZetaEven (ZMod.toAddCircle (Neg.neg j)) s) ( …
  -/
  rw [map_neg, completedHurwitzZetaEven_neg]
  /-
    🎉 no goals
  -/


/--
The completed L-function of a function `ZMod 1 → ℂ` is a scalar multiple of the completed Riemann
zeta function.
-/
lemma completedLFunction_modOne_eq (Φ : ZMod 1 → ℂ) (s : ℂ) :
    completedLFunction Φ s = Φ 1 * completedRiemannZeta s := by
  rw [completedLFunction_def_even (show Φ.Even from fun _ ↦ congr_arg Φ (Subsingleton.elim ..)),
    Nat.cast_one, one_cpow, one_mul, ← singleton_eq_univ 0, sum_singleton, map_zero,
    completedHurwitzZetaEven_zero, Subsingleton.elim 0 1]


/--
The completed L-function of a function `ZMod N → ℂ`, modified by adding multiples of `N ^ (-s) / s`
and `N ^ (-s) / (1 - s)` to make it entire.
-/
noncomputable def completedLFunction₀ (Φ : ZMod N → ℂ) (s : ℂ) : ℂ :=
  N ^ (-s) * ∑ j : ZMod N, Φ j * completedHurwitzZetaEven₀ (toAddCircle j) s
  + N ^ (-s) * ∑ j : ZMod N, Φ j * completedHurwitzZetaOdd (toAddCircle j) s


/-- The function `completedLFunction₀ Φ` is differentiable. -/
lemma differentiable_completedLFunction₀ (Φ : ZMod N → ℂ) :
    Differentiable ℂ (completedLFunction₀ Φ) := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    ⊢ Differentiable Complex (ZMod.completedLFunction₀ Φ)
  -/
  refine .add ?_ ?_ <;>
  /-
    case refine_1
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    ⊢ Differentiable Complex fun y => HMul.hMul (HPow.hPow (↑N) (Neg.neg y)) (Fins …
  -/
  refine .mul (by fun_prop) (.sum fun i _ ↦ .const_mul ?_ _)
  /-
    case refine_1
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    i : ZMod N
    x✝ : Membership.mem Finset.univ i
    ⊢ Differentiable Complex (HurwitzZeta.completedHurwitzZetaEven₀ (ZMod.toAddCir …
  -/
  exacts [differentiable_completedHurwitzZetaEven₀ _, differentiable_completedHurwitzZetaOdd _]
  /-
    🎉 no goals
  -/


lemma completedLFunction_eq (Φ : ZMod N → ℂ) (s : ℂ) :
    completedLFunction Φ s =
      completedLFunction₀ Φ s - N ^ (-s) * Φ 0 / s - N ^ (-s) * (∑ j, Φ j) / (1 - s) := by
  simp only [completedLFunction, completedHurwitzZetaEven_eq, toAddCircle_eq_zero, div_eq_mul_inv,
    ite_mul, one_mul, zero_mul, mul_sub, mul_ite, mul_zero, sum_sub_distrib, Fintype.sum_ite_eq',
    ← sum_mul, completedLFunction₀, mul_assoc]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (HPow.hPow (↑N) (Neg.neg s))  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/--
The completed L-function of a function `ZMod N → ℂ` is differentiable, with the following
exceptions: at `s = 1` if `∑ j, Φ j ≠ 0`; and at `s = 0` if `Φ 0 ≠ 0`.
-/
lemma differentiableAt_completedLFunction (Φ : ZMod N → ℂ) (s : ℂ) (hs₀ : s ≠ 0 ∨ Φ 0 = 0)
    (hs₁ : s ≠ 1 ∨ ∑ j, Φ j = 0) : DifferentiableAt ℂ (completedLFunction Φ) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
    hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
    ⊢ DifferentiableAt Complex (ZMod.completedLFunction Φ) s
  -/
  simp only [funext (completedLFunction_eq Φ), mul_div_assoc]
  -- We know `completedLFunction₀` is differentiable everywhere, so it suffices to show that the
  -- correction terms from `completedLFunction_eq` are differentiable at `s`.
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
    hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
    ⊢ DifferentiableAt Complex (fun x => HSub.hSub (HSub.hSub (ZMod.completedLFunc …
  -/
  refine ((differentiable_completedLFunction₀ _ _).sub ?_).sub ?_
  · -- term with `1 / s`
    /-
      case refine_1
      N : Nat
      inst✝ : NeZero N
      Φ : ZMod N → Complex
      s : Complex
      hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
      hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
      ⊢ DifferentiableAt Complex (fun x => HMul.hMul (HPow.hPow (↑N) (Neg.neg x)) (H …
    -/
    refine .mul (by fun_prop) (hs₀.elim ?_ ?_)
      /-
        case refine_1.refine_1
        N : Nat
        inst✝ : NeZero N
        Φ : ZMod N → Complex
        s : Complex
        hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
        hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
        ⊢ Ne s 0 → DifferentiableAt Complex (HDiv.hDiv (Φ 0)) s
      -/
    · exact fun h ↦ (differentiableAt_const _).div differentiableAt_id h
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        N : Nat
        inst✝ : NeZero N
        Φ : ZMod N → Complex
        s : Complex
        hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
        hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
        ⊢ Eq (Φ 0) 0 → DifferentiableAt Complex (HDiv.hDiv (Φ 0)) s
      -/
    · exact fun h ↦ by simp only [h, funext zero_div, differentiableAt_const]
      /-
        🎉 no goals
      -/
  · -- term with `1 / (1 - s)`
    /-
      case refine_2
      N : Nat
      inst✝ : NeZero N
      Φ : ZMod N → Complex
      s : Complex
      hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
      hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
      ⊢ DifferentiableAt Complex (fun x => HMul.hMul (HPow.hPow (↑N) (Neg.neg x)) (H …
    -/
    refine .mul (by fun_prop) (hs₁.elim ?_ ?_)
      /-
        case refine_2.refine_1
        N : Nat
        inst✝ : NeZero N
        Φ : ZMod N → Complex
        s : Complex
        hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
        hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
        ⊢ Ne s 1 → DifferentiableAt Complex (fun x => HDiv.hDiv (Finset.univ.sum fun j …
      -/
    · exact fun h ↦ .div (by fun_prop) (by fun_prop) (by rwa [sub_ne_zero, ne_comm])
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        N : Nat
        inst✝ : NeZero N
        Φ : ZMod N → Complex
        s : Complex
        hs₀ : Or (Ne s 0) (Eq (Φ 0) 0)
        hs₁ : Or (Ne s 1) (Eq (Finset.univ.sum fun j => Φ j) 0)
        ⊢ Eq (Finset.univ.sum fun j => Φ j) 0 → DifferentiableAt Complex (fun x => HDi …
      -/
    · exact fun h ↦ by simp only [h, zero_div, differentiableAt_const]
      /-
        🎉 no goals
      -/


/--
Special case of `differentiableAt_completedLFunction` asserting differentiability everywhere
under suitable hypotheses.
-/
lemma differentiable_completedLFunction (hΦ₂ : Φ 0 = 0) (hΦ₃ : ∑ j, Φ j = 0) :
    Differentiable ℂ (completedLFunction Φ) :=
  fun s ↦ differentiableAt_completedLFunction Φ s (.inr hΦ₂) (.inr hΦ₃)


/--
Relation between the completed L-function and the usual one (even case).
We state it this way around so it holds at the poles of the gamma factor as well
(except at `s = 0`, where it is genuinely false if `N > 1` and `Φ 0 ≠ 0`).
-/
lemma LFunction_eq_completed_div_gammaFactor_even (hΦ : Φ.Even) (s : ℂ) (hs : s ≠ 0 ∨ Φ 0 = 0) :
    LFunction Φ s = completedLFunction Φ s / Gammaℝ s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs : Or (Ne s 0) (Eq (Φ 0) 0)
    ⊢ Eq (ZMod.LFunction Φ s) (HDiv.hDiv (ZMod.completedLFunction Φ s) s.Gammaℝ)
  -/
  simp only [completedLFunction_def_even hΦ, LFunction_def_even hΦ, mul_div_assoc, sum_div]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs : Or (Ne s 0) (Eq (Φ 0) 0)
    ⊢ Eq (HMul.hMul (HPow.hPow (↑N) (Neg.neg s)) (Finset.univ.sum fun j => HMul.hM …
  -/
  congr 2 with i
  /-
    case e_a.e_f.h
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs : Or (Ne s 0) (Eq (Φ 0) 0)
    i : ZMod N
    ⊢ Eq (HMul.hMul (Φ i) (HurwitzZeta.hurwitzZetaEven (ZMod.toAddCircle i) s)) (H …
  -/
  rcases ne_or_eq i 0 with hi | rfl
    /-
      case e_a.e_f.h.inl
      N : Nat
      inst✝ : NeZero N
      Φ : ZMod N → Complex
      hΦ : Function.Even Φ
      s : Complex
      hs : Or (Ne s 0) (Eq (Φ 0) 0)
      i : ZMod N
      hi : Ne i 0
      ⊢ Eq (HMul.hMul (Φ i) (HurwitzZeta.hurwitzZetaEven (ZMod.toAddCircle i) s)) (H …
    -/
  · rw [hurwitzZetaEven_def_of_ne_or_ne (.inl (hi ∘ toAddCircle_eq_zero.mp))]
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_f.h.inr
      N : Nat
      inst✝ : NeZero N
      Φ : ZMod N → Complex
      hΦ : Function.Even Φ
      s : Complex
      hs : Or (Ne s 0) (Eq (Φ 0) 0)
      ⊢ Eq (HMul.hMul (Φ 0) (HurwitzZeta.hurwitzZetaEven (ZMod.toAddCircle 0) s)) (H …
    -/
  · rcases hs with hs | hΦ'
      /-
        case e_a.e_f.h.inr.inl
        N : Nat
        inst✝ : NeZero N
        Φ : ZMod N → Complex
        hΦ : Function.Even Φ
        s : Complex
        hs : Ne s 0
        ⊢ Eq (HMul.hMul (Φ 0) (HurwitzZeta.hurwitzZetaEven (ZMod.toAddCircle 0) s)) (H …
      -/
    · rw [hurwitzZetaEven_def_of_ne_or_ne (.inr hs)]
      /-
        🎉 no goals
      -/
      /-
        case e_a.e_f.h.inr.inr
        N : Nat
        inst✝ : NeZero N
        Φ : ZMod N → Complex
        hΦ : Function.Even Φ
        s : Complex
        hΦ' : Eq (Φ 0) 0
        ⊢ Eq (HMul.hMul (Φ 0) (HurwitzZeta.hurwitzZetaEven (ZMod.toAddCircle 0) s)) (H …
      -/
    · simp only [hΦ', map_zero, zero_mul]
      /-
        🎉 no goals
      -/


/--
Relation between the completed L-function and the usual one (odd case).
We state it this way around so it holds at the poles of the gamma factor as well.
-/
lemma LFunction_eq_completed_div_gammaFactor_odd (hΦ : Φ.Odd) (s : ℂ) :
    LFunction Φ s = completedLFunction Φ s / Gammaℝ (s + 1) := by
  simp only [LFunction_def_odd hΦ, completedLFunction_def_odd hΦ, hurwitzZetaOdd, mul_div_assoc,
    sum_div]


/--
First form of functional equation for completed L-functions (even case).

Private because it is superseded by `completedLFunction_one_sub_even` below, which is valid for a
much wider range of `s`.
-/
private lemma completedLFunction_one_sub_of_one_lt_even (hΦ : Φ.Even) {s : ℂ} (hs : 1 < re s) :
    completedLFunction Φ (1 - s) = N ^ (s - 1) * completedLFunction (𝓕 Φ) s := by
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  have hs₀ : s ≠ 0 := ne_zero_of_one_lt_re hs
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs : LT.lt 1 s.re
    hs₀ : Ne s 0
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  have hs₁ : s ≠ 1 := (lt_irrefl _ <| one_re ▸ · ▸ hs)
  -- strip down to the key equality:
  suffices ∑ x, Φ x * completedCosZeta (toAddCircle x) s = completedLFunction (𝓕 Φ) s by
    simp only [completedLFunction_def_even hΦ, neg_sub, completedHurwitzZetaEven_one_sub, this]
  -- reduce to equality with un-completed L-functions:
  suffices ∑ x, Φ x * cosZeta (toAddCircle x) s = LFunction (𝓕 Φ) s by
    simpa only [cosZeta, Function.update_of_ne hs₀, ← mul_div_assoc, ← sum_div,
      LFunction_eq_completed_div_gammaFactor_even (dft_even_iff.mpr hΦ) _ (.inl hs₀),
      div_left_inj' (Gammaℝ_ne_zero_of_re_pos (zero_lt_one.trans hs))]
  -- expand out `LFunction (𝓕 Φ)` and use parity:
  simp only [cosZeta_eq, ← mul_div_assoc _ _ (2 : ℂ), mul_add, ← sum_div, sum_add_distrib,
    LFunction_dft Φ (.inr hs₁), map_neg, div_eq_iff (two_ne_zero' ℂ), mul_two, add_left_inj]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs : LT.lt 1 s.re
    hs₀ : Ne s 0
    hs₁ : Ne s 1
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (Φ x) (HurwitzZeta.expZeta (ZMod.toAd …
  -/
  exact Fintype.sum_equiv (.neg _) _ _ (by simp [hΦ _])
  /-
    🎉 no goals
  -/


/--
First form of functional equation for completed L-functions (odd case).

Private because it is superseded by `completedLFunction_one_sub_odd` below, which is valid for a
much wider range of `s`.
-/
private lemma completedLFunction_one_sub_of_one_lt_odd (hΦ : Φ.Odd) {s : ℂ} (hs : 1 < re s) :
    completedLFunction Φ (1 - s) = N ^ (s - 1) * I * completedLFunction (𝓕 Φ) s := by
  -- strip down to the key equality:
  suffices ∑ x, Φ x * completedSinZeta (toAddCircle x) s = I * completedLFunction (𝓕 Φ) s by
    simp only [completedLFunction_def_odd hΦ, neg_sub, completedHurwitzZetaOdd_one_sub, this,
      mul_assoc]
  -- reduce to equality with un-completed L-functions:
  suffices ∑ x, Φ x * sinZeta (toAddCircle x) s = I * LFunction (𝓕 Φ) s by
    have hs' : 0 < re (s + 1) := by simp only [add_re, one_re]; linarith
    simpa only [sinZeta, ← mul_div_assoc, ← sum_div, div_left_inj' (Gammaℝ_ne_zero_of_re_pos hs'),
      LFunction_eq_completed_div_gammaFactor_odd (dft_odd_iff.mpr hΦ)]
  -- now calculate:
  calc ∑ x, Φ x * sinZeta (toAddCircle x) s
  _ = (∑ x, Φ x * expZeta (toAddCircle x) s) / (2 * I)
      - (∑ x, Φ x * expZeta (toAddCircle (-x)) s) / (2 * I) := by
    simp only [sinZeta_eq, ← mul_div_assoc, mul_sub, sub_div, sum_sub_distrib, sum_div, map_neg]
  _ = (∑ x, Φ (-x) * expZeta (toAddCircle (-x)) s) / (_) - (_) := by
    congrm ?_ / _ - _
    exact (Fintype.sum_equiv (.neg _) _ _ fun x ↦ by rfl).symm
  _ = -I⁻¹ * LFunction (𝓕 Φ) s := by
    simp only [hΦ _, neg_mul, sum_neg_distrib, LFunction_dft Φ (.inl hΦ.map_zero)]
    ring
  _ = I * LFunction (𝓕 Φ) s := by rw [inv_I, neg_neg]


/--
Functional equation for completed L-functions (even case), valid at all points of differentiability.
-/
theorem completedLFunction_one_sub_even (hΦ : Φ.Even) (s : ℂ)
    (hs₀ : s ≠ 0 ∨ ∑ j, Φ j = 0) (hs₁ : s ≠ 1 ∨ Φ 0 = 0) :
    completedLFunction Φ (1 - s) = N ^ (s - 1) * completedLFunction (𝓕 Φ) s := by
  -- We prove this using `AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq`, so we need to
  -- gather up the ingredients for this big theorem.
  -- First set up some notations:
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  let F (t) := completedLFunction Φ (1 - t)
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  let G (t) := ↑N ^ (t - 1) * completedLFunction (𝓕 Φ) t
  -- Set on which F, G are analytic:
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HPow.hPow (↑N) (HSub.hSub t 1)) ( …
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  let U := {t : ℂ | (t ≠ 0 ∨ ∑ j, Φ j = 0) ∧ (t ≠ 1 ∨ Φ 0 = 0)}
  -- Properties of U:
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HPow.hPow (↑N) (HSub.hSub t 1)) ( …
    U : Set Complex := setOf fun t => And (Or (Ne t 0) (Eq (Finset.univ.sum fun j  …
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  have hsU : s ∈ U := ⟨hs₀, hs₁⟩
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HPow.hPow (↑N) (HSub.hSub t 1)) ( …
    U : Set Complex := setOf fun t => And (Or (Ne t 0) (Eq (Finset.univ.sum fun j  …
    hsU : Membership.mem U s
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  have h2U : 2 ∈ U := ⟨.inl two_ne_zero, .inl (OfNat.ofNat_ne_one _)⟩
  have hUo : IsOpen U := (isOpen_compl_singleton.union isOpen_const).inter
    (isOpen_compl_singleton.union isOpen_const)
  have hUp : IsPreconnected U := by
    -- need to write `U` as the complement of an obviously countable set
    let Uc : Set ℂ := (if ∑ j, Φ j = 0 then ∅ else {0}) ∪ (if Φ 0 = 0 then ∅ else {1})
    have : Uc.Countable := by
      apply Countable.union <;>
      split_ifs <;>
      simp only [countable_singleton, countable_empty]
    convert (this.isConnected_compl_of_one_lt_rank ?_).isPreconnected using 1
    · ext x
      by_cases h : Φ 0 = 0 <;>
      by_cases h' : ∑ j, Φ j = 0 <;>
      simp [U, Uc, h, h', and_comm]
    · simp only [rank_real_complex, Nat.one_lt_ofNat]
  -- Analyticity on U:
  have hF : AnalyticOnNhd ℂ F U := by
    refine DifferentiableOn.analyticOnNhd
      (fun t ht ↦ DifferentiableAt.differentiableWithinAt ?_) hUo
    refine (differentiableAt_completedLFunction Φ _ ?_ ?_).comp t (differentiableAt_id.const_sub 1)
    exacts [ht.2.imp_left (sub_ne_zero.mpr ∘ Ne.symm), ht.1.imp_left sub_eq_self.not.mpr]
  have hG : AnalyticOnNhd ℂ G U := by
    refine DifferentiableOn.analyticOnNhd
      (fun t ht ↦ DifferentiableAt.differentiableWithinAt ?_) hUo
    apply ((differentiableAt_id.sub_const 1).const_cpow (.inl (NeZero.ne _))).mul
    apply differentiableAt_completedLFunction _ _ (ht.1.imp_right fun h ↦ dft_apply_zero Φ ▸ h)
    exact ht.2.imp_right (fun h ↦ by simp only [← dft_apply_zero, dft_dft, neg_zero, h, smul_zero])
  -- set where we know equality
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HPow.hPow (↑N) (HSub.hSub t 1)) ( …
    U : Set Complex := setOf fun t => And (Or (Ne t 0) (Eq (Finset.univ.sum fun j  …
    hsU : Membership.mem U s
    h2U : Membership.mem U 2
    hUo : IsOpen U
    hUp : IsPreconnected U
    hF : AnalyticOnNhd Complex F U
    hG : AnalyticOnNhd Complex G U
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  have hV : {z | 1 < re z} ∈ 𝓝 2 := (continuous_re.isOpen_preimage _ isOpen_Ioi).mem_nhds (by simp)
  have hFG : F =ᶠ[𝓝 2] G := eventually_of_mem hV <| fun t ht ↦ by
    simpa only [F, G, pow_zero, mul_one] using completedLFunction_one_sub_of_one_lt_even hΦ ht
  -- now apply the big hammer to finish
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Even Φ
    s : Complex
    hs₀ : Or (Ne s 0) (Eq (Finset.univ.sum fun j => Φ j) 0)
    hs₁ : Or (Ne s 1) (Eq (Φ 0) 0)
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HPow.hPow (↑N) (HSub.hSub t 1)) ( …
    U : Set Complex := setOf fun t => And (Or (Ne t 0) (Eq (Finset.univ.sum fun j  …
    hsU : Membership.mem U s
    h2U : Membership.mem U 2
    hUo : IsOpen U
    hUp : IsPreconnected U
    hF : AnalyticOnNhd Complex F U
    hG : AnalyticOnNhd Complex G U
    hV : Membership.mem (nhds 2) (setOf fun z => LT.lt 1 z.re)
    hFG : (nhds 2).EventuallyEq F G
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HPow.hPow (↑N) (H …
  -/
  exact hF.eqOn_of_preconnected_of_eventuallyEq hG hUp h2U hFG hsU
  /-
    🎉 no goals
  -/


/-- Functional equation for completed L-functions (odd case), valid for all `s`. -/
theorem completedLFunction_one_sub_odd (hΦ : Φ.Odd) (s : ℂ) :
    completedLFunction Φ (1 - s) = N ^ (s - 1) * I * completedLFunction (𝓕 Φ) s := by
  -- This is much easier than the even case since both functions are entire.
  -- First set up some notations:
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HPow.h …
  -/
  let F (t) := completedLFunction Φ (1 - t)
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HPow.h …
  -/
  let G (t) := ↑N ^ (t - 1) * I * completedLFunction (𝓕 Φ) t
  -- check F, G globally differentiable
  have hF : Differentiable ℂ F := (differentiable_completedLFunction hΦ.map_zero
    hΦ.sum_eq_zero).comp (differentiable_id.const_sub 1)
  have hG : Differentiable ℂ G := by
    apply (((differentiable_id.sub_const 1).const_cpow (.inl (NeZero.ne _))).mul_const _).mul
    rw [← dft_odd_iff] at hΦ
    exact differentiable_completedLFunction hΦ.map_zero hΦ.sum_eq_zero
  -- set where we know equality
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HMul.hMul (HPow.hPow (↑N) (HSub.h …
    hF : Differentiable Complex F
    hG : Differentiable Complex G
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HPow.h …
  -/
  have : {z | 1 < re z} ∈ 𝓝 2 := (continuous_re.isOpen_preimage _ isOpen_Ioi).mem_nhds (by simp)
  have hFG : F =ᶠ[𝓝 2] G := by filter_upwards [this] with t ht
    using completedLFunction_one_sub_of_one_lt_odd hΦ ht
  -- now apply the big hammer to finish
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HMul.hMul (HPow.hPow (↑N) (HSub.h …
    hF : Differentiable Complex F
    hG : Differentiable Complex G
    this : Membership.mem (nhds 2) (setOf fun z => LT.lt 1 z.re)
    hFG : (nhds 2).EventuallyEq F G
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HPow.h …
  -/
  rw [← analyticOnNhd_univ_iff_differentiable] at hF hG
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    hΦ : Function.Odd Φ
    s : Complex
    F : Complex → Complex := fun t => ZMod.completedLFunction Φ (HSub.hSub 1 t)
    G : Complex → Complex := fun t => HMul.hMul (HMul.hMul (HPow.hPow (↑N) (HSub.h …
    hF : AnalyticOnNhd Complex F Set.univ
    hG : AnalyticOnNhd Complex G Set.univ
    this : Membership.mem (nhds 2) (setOf fun z => LT.lt 1 z.re)
    hFG : (nhds 2).EventuallyEq F G
    ⊢ Eq (ZMod.completedLFunction Φ (HSub.hSub 1 s)) (HMul.hMul (HMul.hMul (HPow.h …
  -/
  exact congr_fun (hF.eq_of_eventuallyEq hG hFG) s
  /-
    🎉 no goals
  -/


