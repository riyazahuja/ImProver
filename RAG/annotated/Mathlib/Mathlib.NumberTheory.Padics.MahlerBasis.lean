/-- Bound for norms of ascending Pochhammer symbols. -/
lemma norm_ascPochhammer_le (k : ℕ) (x : ℤ_[p]) :
    ‖(ascPochhammer ℤ_[p] k).eval x‖ ≤ ‖(k.factorial : ℤ_[p])‖ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    x : PadicInt p
    ⊢ LE.le (Norm.norm (Polynomial.eval x (ascPochhammer (PadicInt p) k))) (Norm.n …
  -/
  let f := (ascPochhammer ℤ_[p] k).eval
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    x : PadicInt p
    f : PadicInt p → PadicInt p := fun a => Polynomial.eval a (ascPochhammer (Padi …
    ⊢ LE.le (Norm.norm (Polynomial.eval x (ascPochhammer (PadicInt p) k))) (Norm.n …
  -/
  change ‖f x‖ ≤ ‖_‖
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    x : PadicInt p
    f : PadicInt p → PadicInt p := fun a => Polynomial.eval a (ascPochhammer (Padi …
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm ↑k.factorial)
  -/
  have hC : (k.factorial : ℤ_[p]) ≠ 0 := Nat.cast_ne_zero.mpr k.factorial_ne_zero
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    x : PadicInt p
    f : PadicInt p → PadicInt p := fun a => Polynomial.eval a (ascPochhammer (Padi …
    hC : Ne (↑k.factorial) 0
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm ↑k.factorial)
  -/
  have hf : ContinuousAt f x := Polynomial.continuousAt _
  -- find `n : ℕ` such that `‖f x - f n‖ ≤ ‖k!‖`
  obtain ⟨n, hn⟩ : ∃ n : ℕ, ‖f x - f n‖ ≤ ‖(k.factorial : ℤ_[p])‖ := by
    obtain ⟨δ, hδp, hδ⟩ := Metric.continuousAt_iff.mp hf _ (norm_pos_iff.mpr hC)
    obtain ⟨n, hn'⟩ := PadicInt.denseRange_natCast.exists_dist_lt x hδp
    simpa only [← dist_eq_norm_sub'] using ⟨n, (hδ (dist_comm x n ▸ hn')).le⟩
  -- use ultrametric property to show that `‖f n‖ ≤ ‖k!‖` implies `‖f x‖ ≤ ‖k!‖`
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    x : PadicInt p
    f : PadicInt p → PadicInt p := fun a => Polynomial.eval a (ascPochhammer (Padi …
    hC : Ne (↑k.factorial) 0
    hf : ContinuousAt f x
    n : Nat
    hn : LE.le (Norm.norm (HSub.hSub (f x) (f ↑n))) (Norm.norm ↑k.factorial)
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm ↑k.factorial)
  -/
  refine sub_add_cancel (f x) _ ▸ (IsUltrametricDist.norm_add_le_max _ (f n)).trans (max_le hn ?_)
  -- finish using the fact that `n.multichoose k ∈ ℤ`
  simp_rw [f, ← ascPochhammer_eval_cast, Polynomial.eval_eq_smeval,
    ← Ring.factorial_nsmul_multichoose_eq_ascPochhammer, smul_eq_mul, Nat.cast_mul, norm_mul]
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    x : PadicInt p
    f : PadicInt p → PadicInt p := fun a => Polynomial.eval a (ascPochhammer (Padi …
    hC : Ne (↑k.factorial) 0
    hf : ContinuousAt f x
    n : Nat
    hn : LE.le (Norm.norm (HSub.hSub (f x) (f ↑n))) (Norm.norm ↑k.factorial)
    ⊢ LE.le (HMul.hMul (Norm.norm ↑k.factorial) (Norm.norm ↑(Ring.multichoose n k) …
  -/
  exact mul_le_of_le_one_right (norm_nonneg _) (norm_le_one _)
  /-
    🎉 no goals
  -/


/-- The p-adic integers are a binomial ring, i.e. a ring where binomial coefficients make sense. -/
noncomputable instance instBinomialRing : BinomialRing ℤ_[p] where
  nsmul_right_injective hn := smul_right_injective ℤ_[p] hn
  -- We define `multichoose` as a fraction in `ℚ_[p]` together with a proof that its norm is `≤ 1`.
  multichoose x k := ⟨(ascPochhammer ℤ_[p] k).eval x / (k.factorial : ℚ_[p]), by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      k : Nat
      ⊢ LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval x (ascPochhammer (PadicInt p)  …
    -/
    rw [norm_div, div_le_one (by simpa using k.factorial_ne_zero)]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      k : Nat
      ⊢ LE.le (Norm.norm ↑(Polynomial.eval x (ascPochhammer (PadicInt p) k))) (Norm. …
    -/
    exact x.norm_ascPochhammer_le k⟩
    /-
      🎉 no goals
    -/
  factorial_nsmul_multichoose x k := by rw [← Subtype.coe_inj, nsmul_eq_mul, PadicInt.coe_mul,
    PadicInt.coe_natCast, mul_div_cancel₀ _ (mod_cast k.factorial_ne_zero), Subtype.coe_inj,
    Polynomial.eval_eq_smeval, Polynomial.ascPochhammer_smeval_cast]


@[fun_prop]
lemma continuous_multichoose (k : ℕ) : Continuous (fun x : ℤ_[p] ↦ Ring.multichoose x k) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ⊢ Continuous fun x => Ring.multichoose x k
  -/
  simp only [Ring.multichoose, BinomialRing.multichoose, continuous_induced_rng]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ⊢ Continuous (Function.comp Subtype.val fun x => ⟨HDiv.hDiv ↑(Polynomial.eval  …
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[fun_prop]
lemma continuous_choose (k : ℕ) : Continuous (fun x : ℤ_[p] ↦ Ring.choose x k) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ⊢ Continuous fun x => Ring.choose x k
  -/
  simp only [Ring.choose]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ⊢ Continuous fun x => Ring.multichoose (HAdd.hAdd (HSub.hSub x ↑k) 1) k
  -/
  fun_prop
  /-
    🎉 no goals
  -/


/--
The `k`-th Mahler basis function, i.e. the unique continuous function `ℤ_[p] → ℚ_[p]`
agreeing with `n ↦ n.choose k` for `n ∈ ℕ`. See [colmez2010], §1.2.1.
-/
noncomputable def mahler (k : ℕ) : C(ℤ_[p], ℚ_[p]) where
  toFun x := ↑(Ring.choose x k)
  continuous_toFun := continuous_induced_rng.mp (PadicInt.continuous_choose k)


lemma mahler_apply (k : ℕ) (x : ℤ_[p]) : mahler k x = Ring.choose x k := rfl


/-- The function `mahler k` extends `n ↦ n.choose k` on `ℕ`. -/
lemma mahler_natCast_eq (k n : ℕ) : mahler k (n : ℤ_[p]) = n.choose k := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k n : Nat
    ⊢ Eq ((mahler k) ↑n) ↑(n.choose k)
  -/
  simp only [mahler_apply, Ring.choose_natCast, PadicInt.coe_natCast]
  /-
    🎉 no goals
  -/


/--
The uniform norm of the `k`-th Mahler basis function is 1, for every `k`.
-/
@[simp] lemma norm_mahler_eq (k : ℕ) : ‖(mahler k : C(ℤ_[p], ℚ_[p]))‖ = 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Nat
    ⊢ Eq (Norm.norm (mahler k)) 1
  -/
  apply le_antisymm
  · -- Show all values have norm ≤ 1
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Nat
      ⊢ LE.le (Norm.norm (mahler k)) 1
    -/
    exact (mahler k).norm_le_of_nonempty.mpr (fun _ ↦ PadicInt.norm_le_one _)
    /-
      🎉 no goals
    -/
  · -- Show norm 1 is attained at `x = k`
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Nat
      ⊢ LE.le 1 (Norm.norm (mahler k))
    -/
    refine (le_of_eq ?_).trans ((mahler k).norm_coe_le_norm k)
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Nat
      ⊢ Eq 1 (Norm.norm ((mahler k) ↑k))
    -/
    rw [mahler_natCast_eq, Nat.choose_self, Nat.cast_one, norm_one]
    /-
      🎉 no goals
    -/


/-- Bound for iterated forward differences of a continuous function from a compact space to a
nonarchimedean seminormed group. -/
lemma IsUltrametricDist.norm_fwdDiff_iter_apply_le [TopologicalSpace M] [CompactSpace M]
    [AddCommMonoid M] [SeminormedAddCommGroup G] [IsUltrametricDist G]
    (h : M) (f : C(M, G)) (m : M) (n : ℕ) : ‖Δ_[h]^[n] f m‖ ≤ ‖f‖ := by
  -- A proof by induction on `n` would be possible but would involve some messing around to
  -- define `Δ_[h]` as an operator on continuous maps (not just on bare functions). So instead we
  -- use the formula for `Δ_[h]^[n] f` as a sum.
  /-
    M : Type u_1
    G : Type u_2
    inst✝⁴ : TopologicalSpace M
    inst✝³ : CompactSpace M
    inst✝² : AddCommMonoid M
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : IsUltrametricDist G
    h : M
    f : ContinuousMap M G
    m : M
    n : Nat
    ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff h) n (⇑f) m)) (Norm.norm f)
  -/
  rw [fwdDiff_iter_eq_sum_shift]
  /-
    M : Type u_1
    G : Type u_2
    inst✝⁴ : TopologicalSpace M
    inst✝³ : CompactSpace M
    inst✝² : AddCommMonoid M
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : IsUltrametricDist G
    h : M
    f : ContinuousMap M G
    m : M
    n : Nat
    ⊢ LE.le (Norm.norm ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (H …
  -/
  refine norm_sum_le_of_forall_le_of_nonneg (norm_nonneg f) fun i _ ↦ ?_
  /-
    M : Type u_1
    G : Type u_2
    inst✝⁴ : TopologicalSpace M
    inst✝³ : CompactSpace M
    inst✝² : AddCommMonoid M
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : IsUltrametricDist G
    h : M
    f : ContinuousMap M G
    m : M
    n i : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ LE.le (Norm.norm (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) (HSub.hSub n i)) ↑( …
  -/
  exact (norm_zsmul_le _ _).trans (f.norm_coe_le_norm _)
  /-
    🎉 no goals
  -/


/-- First step in Bojanić's proof of Mahler's theorem (equation (10) of [bojanic74]): rewrite
`Δ^[n + R] f 0` in a shape that makes it easy to bound `p`-adically. -/
private lemma bojanic_mahler_step1 [AddCommMonoidWithOne M] [AddCommGroup G] (f : M → G)
    (n : ℕ) {R : ℕ} (hR : 1 ≤ R) :
    Δ_[1]^[n + R] f 0 = -∑ j ∈ range (R - 1), R.choose (j + 1) • Δ_[1]^[n + (j + 1)] f 0 +
      ∑ k ∈ range (n + 1), ((-1 : ℤ) ^ (n - k) * n.choose k) • (f (k + R) - f k) := by
  have aux : Δ_[1]^[n + R] f 0 = R.choose (R - 1 + 1) • Δ_[1]^[n + R] f 0 := by
    rw [Nat.sub_add_cancel hR, Nat.choose_self, one_smul]
  rw [neg_add_eq_sub, eq_sub_iff_add_eq, add_comm, aux, (by omega : n + R = (n + ((R - 1) + 1))),
    ← sum_range_succ, Nat.sub_add_cancel hR,
    ← sub_eq_iff_eq_add.mpr (sum_range_succ' (fun x ↦ R.choose x • Δ_[1]^[n + x] f 0) R), add_zero,
    Nat.choose_zero_right, one_smul]
  have : ∑ k ∈ Finset.range (R + 1), R.choose k • Δ_[1]^[n + k] f 0 = Δ_[1]^[n] f R := by
    simpa only [← Function.iterate_add_apply, add_comm, nsmul_one, add_zero] using
      (shift_eq_sum_fwdDiff_iter 1 (Δ_[1]^[n] f) R 0).symm
  simp only [this, fwdDiff_iter_eq_sum_shift (1 : M) f n, mul_comm, nsmul_one, mul_smul, add_comm,
    add_zero, smul_sub, sum_sub_distrib]


/--
Second step in Bojanić's proof of Mahler's theorem (equation (11) of [bojanic74]): show that values
`Δ_[1]^[n + p ^ t] f 0` for large enough `n` are bounded by the max of `(‖f‖ / p ^ s)` and `1 / p`
times a sup over values for smaller `n`.

We use `nnnorm`s on the RHS since `Finset.sup` requires an order with a bottom element.
-/
private lemma bojanic_mahler_step2 {f : C(ℤ_[p], E)} {s t : ℕ}
    (hst : ∀ x y : ℤ_[p], ‖x - y‖ ≤ p ^ (-t : ℤ) → ‖f x - f y‖ ≤ ‖f‖ / p ^ s) (n : ℕ) :
    ‖Δ_[1]^[n + p ^ t] f 0‖ ≤ max ↑((Finset.range (p ^ t - 1)).sup
      fun j ↦ ‖Δ_[1]^[n + (j + 1)] f 0‖₊ / p) (‖f‖ / p ^ s) := by
  -- Use previous lemma to rewrite in a convenient form.
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    s t : Nat
    hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
    n : Nat
    ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAdd n (HPow.hPow p t)) (⇑f) …
  -/
  rw [bojanic_mahler_step1 _ _ (one_le_pow₀ hp.out.one_le)]
  -- Now use ultrametric property and bound each term separately.
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    s t : Nat
    hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
    n : Nat
    ⊢ LE.le (Norm.norm (HAdd.hAdd (Neg.neg ((Finset.range (HSub.hSub (HPow.hPow p  …
  -/
  refine (norm_add_le_max _ _).trans (max_le_max ?_ ?_)
  · -- Bounding the sum over `range (p ^ t - 1)`: every term involves a value `Δ_[1]^[·] f 0` and
    -- a binomial coefficient which is divisible by `p`
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n : Nat
      ⊢ LE.le (Norm.norm (Neg.neg ((Finset.range (HSub.hSub (HPow.hPow p t) 1)).sum  …
    -/
    rw [norm_neg, ← coe_nnnorm, coe_le_coe]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n : Nat
      ⊢ LE.le (NNNorm.nnnorm ((Finset.range (HSub.hSub (HPow.hPow p t) 1)).sum fun j …
    -/
    refine nnnorm_sum_le_of_forall_le (fun i hi ↦ Finset.le_sup_of_le hi ?_)
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n i : Nat
      hi : Membership.mem (Finset.range (HSub.hSub (HPow.hPow p t) 1)) i
      ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul ((HPow.hPow p t).choose (HAdd.hAdd i 1)) ( …
    -/
    rw [mem_range] at hi
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n i : Nat
      hi : LT.lt i (HSub.hSub (HPow.hPow p t) 1)
      ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul ((HPow.hPow p t).choose (HAdd.hAdd i 1)) ( …
    -/
    rw [← Nat.cast_smul_eq_nsmul ℚ_[p], nnnorm_smul, div_eq_inv_mul]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n i : Nat
      hi : LT.lt i (HSub.hSub (HPow.hPow p t) 1)
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm ↑((HPow.hPow p t).choose (HAdd.hAdd i 1))) ( …
    -/
    refine mul_le_mul_of_nonneg_right ?_ (by simp only [zero_le])
    -- remains to show norm of binomial coeff is `≤ p⁻¹`
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n i : Nat
      hi : LT.lt i (HSub.hSub (HPow.hPow p t) 1)
      ⊢ LE.le (NNNorm.nnnorm ↑((HPow.hPow p t).choose (HAdd.hAdd i 1))) (Inv.inv ↑p)
    -/
    have : 0 < (p ^ t).choose (i + 1) := Nat.choose_pos (by omega)
    rw [← zpow_neg_one, ← coe_le_coe, coe_nnnorm, Padic.norm_eq_zpow_neg_valuation
      (mod_cast this.ne'), coe_zpow, NNReal.coe_natCast,
      zpow_le_zpow_iff_right₀ (mod_cast hp.out.one_lt), neg_le_neg_iff, Padic.valuation_natCast,
      Nat.one_le_cast]
    /-
      case refine_1
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n i : Nat
      hi : LT.lt i (HSub.hSub (HPow.hPow p t) 1)
      this : LT.lt 0 ((HPow.hPow p t).choose (HAdd.hAdd i 1))
      ⊢ LE.le 1 (padicValNat p ((HPow.hPow p t).choose (HAdd.hAdd i 1)))
    -/
    exact one_le_padicValNat_of_dvd this <| hp.out.dvd_choose_pow (by omega) (by omega)
    /-
      🎉 no goals
    -/
  · -- Bounding the sum over `range (n + 1)`: every term is small by the choice of `t`
    /-
      case refine_2
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n : Nat
      ⊢ LE.le (Norm.norm ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (H …
    -/
    refine norm_sum_le_of_forall_le_of_nonempty nonempty_range_succ (fun i _ ↦ ?_)
    calc ‖((-1 : ℤ) ^ (n - i) * n.choose i) • (f (i + ↑(p ^ t)) - f i)‖
    _ ≤ ‖f (i + ↑(p ^ t)) - f i‖ := by
      rw [← Int.cast_smul_eq_zsmul ℚ_[p], norm_smul]
      apply mul_le_of_le_one_left (norm_nonneg _)
      simpa only [← coe_intCast] using norm_le_one _
    _ ≤ ‖f‖ / p ^ s := by
      apply hst
      rw [Nat.cast_pow, add_sub_cancel_left, norm_pow, norm_p, inv_pow, zpow_neg, zpow_natCast]


/--
Explicit bound for the decay rate of the Mahler coefficients of a continuous function on `ℤ_[p]`.
This will be used to prove Mahler's theorem.
 -/
lemma fwdDiff_iter_le_of_forall_le {f : C(ℤ_[p], E)} {s t : ℕ}
    (hst : ∀ x y : ℤ_[p], ‖x - y‖ ≤ p ^ (-t : ℤ) → ‖f x - f y‖ ≤ ‖f‖ / p ^ s) (n : ℕ) :
    ‖Δ_[1]^[n + s * p ^ t] f 0‖ ≤ ‖f‖ / p ^ s := by
  -- We show the following more general statement by induction on `k`:
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    s t : Nat
    hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
    n : Nat
    ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAdd n (HMul.hMul s (HPow.hP …
  -/
  suffices ∀ {k : ℕ}, k ≤ s → ‖Δ_[1]^[n + k * p ^ t] f 0‖ ≤ ‖f‖ / p ^ k from this le_rfl
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    s t : Nat
    hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
    n : Nat
    ⊢ ∀ {k : Nat}, LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAd …
  -/
  intro k hk
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    s t : Nat
    hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
    n k : Nat
    hk : LE.le k s
    ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAdd n (HMul.hMul k (HPow.hP …
  -/
  induction' k with k IH generalizing n
  · -- base case just says that `‖Δ^[·] (⇑f) 0‖` is bounded by `‖f‖`
    /-
      case zero
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      n : Nat
      hk : LE.le 0 s
      ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAdd n (HMul.hMul 0 (HPow.hP …
    -/
    simpa only [zero_mul, pow_zero, add_zero, div_one] using norm_fwdDiff_iter_apply_le 1 f 0 n
    /-
      🎉 no goals
    -/
  · -- induction is the "step 2" lemma above
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      k : Nat
      IH : ∀ (n : Nat), LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd. …
      n : Nat
      hk : LE.le (HAdd.hAdd k 1) s
      ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAdd n (HMul.hMul (HAdd.hAdd …
    -/
    rw [add_mul, one_mul, ← add_assoc]
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace (Padic p) E
      inst✝ : IsUltrametricDist E
      f : ContinuousMap (PadicInt p) E
      s t : Nat
      hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
      k : Nat
      IH : ∀ (n : Nat), LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd. …
      n : Nat
      hk : LE.le (HAdd.hAdd k 1) s
      ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd.hAdd (HAdd.hAdd n (HMul.hMul …
    -/
    refine (bojanic_mahler_step2 hst (n + k * p ^ t)).trans (max_le ?_ ?_)
      /-
        case succ.refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace (Padic p) E
        inst✝ : IsUltrametricDist E
        f : ContinuousMap (PadicInt p) E
        s t : Nat
        hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
        k : Nat
        IH : ∀ (n : Nat), LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd. …
        n : Nat
        hk : LE.le (HAdd.hAdd k 1) s
        ⊢ LE.le (↑((Finset.range (HSub.hSub (HPow.hPow p t) 1)).sup fun j => HDiv.hDiv …
      -/
    · rw [← coe_nnnorm, ← NNReal.coe_natCast, ← NNReal.coe_pow, ← NNReal.coe_div, NNReal.coe_le_coe]
      /-
        case succ.refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace (Padic p) E
        inst✝ : IsUltrametricDist E
        f : ContinuousMap (PadicInt p) E
        s t : Nat
        hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
        k : Nat
        IH : ∀ (n : Nat), LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd. …
        n : Nat
        hk : LE.le (HAdd.hAdd k 1) s
        ⊢ LE.le ((Finset.range (HSub.hSub (HPow.hPow p t) 1)).sup fun j => HDiv.hDiv ( …
      -/
      refine Finset.sup_le fun j _ ↦ ?_
      /-
        case succ.refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace (Padic p) E
        inst✝ : IsUltrametricDist E
        f : ContinuousMap (PadicInt p) E
        s t : Nat
        hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
        k : Nat
        IH : ∀ (n : Nat), LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd. …
        n : Nat
        hk : LE.le (HAdd.hAdd k 1) s
        j : Nat
        x✝ : Membership.mem (Finset.range (HSub.hSub (HPow.hPow p t) 1)) j
        ⊢ LE.le (HDiv.hDiv (NNNorm.nnnorm (Nat.iterate (fwdDiff 1) (HAdd.hAdd (HAdd.hA …
      -/
      rw [pow_succ, ← div_div, div_le_div_iff_of_pos_right (mod_cast hp.out.pos), add_right_comm]
      /-
        case succ.refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace (Padic p) E
        inst✝ : IsUltrametricDist E
        f : ContinuousMap (PadicInt p) E
        s t : Nat
        hst : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p)  …
        k : Nat
        IH : ∀ (n : Nat), LE.le k s → LE.le (Norm.norm (Nat.iterate (fwdDiff 1) (HAdd. …
        n : Nat
        hk : LE.le (HAdd.hAdd k 1) s
        j : Nat
        x✝ : Membership.mem (Finset.range (HSub.hSub (HPow.hPow p t) 1)) j
        ⊢ LE.le (NNNorm.nnnorm (Nat.iterate (fwdDiff 1) (HAdd.hAdd (HAdd.hAdd n (HAdd. …
      -/
      exact_mod_cast IH (n + (j + 1)) (by omega)
      /-
        🎉 no goals
      -/
    · exact div_le_div_of_nonneg_left (norm_nonneg _)
        (mod_cast pow_pos hp.out.pos _) (mod_cast pow_le_pow_right₀ hp.out.one_le hk)


/-- Key lemma for Mahler's theorem: for `f` a continuous function on `ℤ_[p]`, the sequence
`n ↦ Δ^[n] f 0` tends to 0. See `PadicInt.fwdDiff_iter_le_of_forall_le` for an explicit
estimate of the decay rate. -/
lemma fwdDiff_tendsto_zero (f : C(ℤ_[p], E)) : Tendsto (Δ_[1]^[·] f 0) atTop (𝓝 0) := by
  -- first extract an `s`
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    ⊢ Filter.Tendsto (fun x => Nat.iterate (fwdDiff 1) x (⇑f) 0) Filter.atTop (nhd …
  -/
  refine NormedAddCommGroup.tendsto_nhds_zero.mpr (fun ε hε ↦ ?_)
  have : Tendsto (fun s ↦ ‖f‖ / p ^ s) _ _ := tendsto_const_nhds.div_atTop
    (tendsto_pow_atTop_atTop_of_one_lt (mod_cast hp.out.one_lt))
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    ε : Real
    hε : GT.gt ε 0
    this : Filter.Tendsto (fun s => HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) Fi …
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (Nat.iterate (fwdDiff 1) x (⇑f) …
  -/
  obtain ⟨s, hs⟩ := (this.eventually_lt_const hε).exists
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    ε : Real
    hε : GT.gt ε 0
    this : Filter.Tendsto (fun s => HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) Fi …
    s : Nat
    hs : LT.lt (HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) ε
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (Nat.iterate (fwdDiff 1) x (⇑f) …
  -/
  refine .mp ?_ (.of_forall fun x hx ↦ lt_of_le_of_lt hx hs)
  -- use uniform continuity to find `t`
  obtain ⟨t, ht⟩ : ∃ t : ℕ, ∀ x y, ‖x - y‖ ≤ p ^ (-t : ℤ) → ‖f x - f y‖ ≤ ‖f‖ / p ^ s := by
    rcases eq_or_ne f 0 with rfl | hf
    · -- silly case : f = 0
      simp
    have : 0 < ‖f‖ / p ^ s := div_pos (norm_pos_iff.mpr hf) (mod_cast pow_pos hp.out.pos _)
    obtain ⟨δ, hδpos, hδf⟩ := f.uniform_continuity _ this
    obtain ⟨t, ht⟩ := PadicInt.exists_pow_neg_lt p hδpos
    exact ⟨t, fun x y hxy ↦  by simpa only [dist_eq_norm_sub] using (hδf (hxy.trans_lt ht)).le⟩
  /-
    case intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    ε : Real
    hε : GT.gt ε 0
    this : Filter.Tendsto (fun s => HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) Fi …
    s : Nat
    hs : LT.lt (HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) ε
    t : Nat
    ht : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p) ( …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (Nat.iterate (fwdDiff 1) x (⇑f) …
  -/
  filter_upwards [eventually_ge_atTop (s * p ^ t)] with m hm
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace (Padic p) E
    inst✝ : IsUltrametricDist E
    f : ContinuousMap (PadicInt p) E
    ε : Real
    hε : GT.gt ε 0
    this : Filter.Tendsto (fun s => HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) Fi …
    s : Nat
    hs : LT.lt (HDiv.hDiv (Norm.norm f) (HPow.hPow (↑p) s)) ε
    t : Nat
    ht : ∀ (x y : PadicInt p), LE.le (Norm.norm (HSub.hSub x y)) (HPow.hPow (↑p) ( …
    m : Nat
    hm : LE.le (HMul.hMul s (HPow.hPow p t)) m
    ⊢ LE.le (Norm.norm (Nat.iterate (fwdDiff 1) m (⇑f) 0)) (HDiv.hDiv (Norm.norm f …
  -/
  simpa only [Nat.sub_add_cancel hm] using fwdDiff_iter_le_of_forall_le ht (m - s * p ^ t)
  /-
    🎉 no goals
  -/


/--
A single term of a Mahler series, given by the product of the scalar-valued continuous map
`mahler n : ℤ_[p] → ℚ_[p]` with a constant vector in some normed `ℚ_[p]`-vector space.
-/
noncomputable def mahlerTerm : C(ℤ_[p], E) := (mahler n : C(_, ℚ_[p])) • .const _ a


lemma mahlerTerm_apply : mahlerTerm a n x = mahler n x • a := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace (Padic p) E
    a : E
    n : Nat
    x : PadicInt p
    ⊢ Eq ((PadicInt.mahlerTerm a n) x) (HSMul.hSMul ((mahler n) x) a)
  -/
  simp only [mahlerTerm, ContinuousMap.smul_apply', ContinuousMap.const_apply]
  /-
    🎉 no goals
  -/


lemma norm_mahlerTerm : ‖(mahlerTerm a n : C(ℤ_[p], E))‖ = ‖a‖ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace (Padic p) E
    a : E
    n : Nat
    ⊢ Eq (Norm.norm (PadicInt.mahlerTerm a n)) (Norm.norm a)
  -/
  simp only [mahlerTerm, ContinuousMap.norm_smul_const, norm_mahler_eq, one_mul]
  /-
    🎉 no goals
  -/


/-- A series of the form considered in Mahler's theorem. -/
noncomputable def mahlerSeries (a : ℕ → E) : C(ℤ_[p], E) := ∑' n, mahlerTerm (a n) n


/-- A Mahler series whose coefficients tend to 0 is convergent. -/
lemma hasSum_mahlerSeries (ha : Tendsto a atTop (𝓝 0)) :
    HasSum (fun n ↦ mahlerTerm (a n) n) (mahlerSeries a : C(ℤ_[p], E)) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace (Padic p) E
    inst✝¹ : IsUltrametricDist E
    inst✝ : CompleteSpace E
    a : Nat → E
    ha : Filter.Tendsto a Filter.atTop (nhds 0)
    ⊢ HasSum (fun n => PadicInt.mahlerTerm (a n) n) (PadicInt.mahlerSeries a)
  -/
  refine (NonarchimedeanAddGroup.summable_of_tendsto_cofinite_zero ?_).hasSum
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace (Padic p) E
    inst✝¹ : IsUltrametricDist E
    inst✝ : CompleteSpace E
    a : Nat → E
    ha : Filter.Tendsto a Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun n => PadicInt.mahlerTerm (a n) n) Filter.cofinite (nhds 0)
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero] at ha ⊢
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace (Padic p) E
    inst✝¹ : IsUltrametricDist E
    inst✝ : CompleteSpace E
    a : Nat → E
    ha : Filter.Tendsto (fun x => Norm.norm (a x)) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun x => Norm.norm (PadicInt.mahlerTerm (a x) x)) Filter.cof …
  -/
  simpa only [norm_mahlerTerm, Nat.cofinite_eq_atTop] using ha
  /-
    🎉 no goals
  -/


/-- Evaluation of a Mahler series is just the pointwise sum. -/
lemma mahlerSeries_apply (ha : Tendsto a atTop (𝓝 0)) (x : ℤ_[p]) :
    mahlerSeries a x = ∑' n, mahler n x • a n := by
  simp only [mahlerSeries, ← ContinuousMap.tsum_apply (hasSum_mahlerSeries ha).summable,
    mahlerTerm_apply]


/--
The value of a Mahler series at a natural number `n` is given by the finite sum of the first `m`
terms, for any `n ≤ m`.
-/
lemma mahlerSeries_apply_nat (ha : Tendsto a atTop (𝓝 0)) {m n : ℕ} (hmn : m ≤ n) :
    mahlerSeries a (m : ℤ_[p]) = ∑ i in range (n + 1), m.choose i • a i := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace (Padic p) E
    inst✝¹ : IsUltrametricDist E
    inst✝ : CompleteSpace E
    a : Nat → E
    ha : Filter.Tendsto a Filter.atTop (nhds 0)
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq ((PadicInt.mahlerSeries a) ↑m) ((Finset.range (HAdd.hAdd n 1)).sum fun i  …
  -/
  have h_van (i) : m.choose (i + (n + 1)) = 0 := Nat.choose_eq_zero_of_lt (by omega)
  have aux : Summable fun i ↦ m.choose (i + (n + 1)) • a (i + (n + 1)) := by
    simpa only [h_van, zero_smul] using summable_zero
  simp only [mahlerSeries_apply ha, mahler_natCast_eq, Nat.cast_smul_eq_nsmul, add_zero,
    ← sum_add_tsum_nat_add' (f := fun i ↦ m.choose i • a i) aux, h_van, zero_smul, tsum_zero]


/--
The coefficients of a Mahler series can be recovered from the sum by taking forward differences at
`0`.
-/
lemma fwdDiff_mahlerSeries (ha : Tendsto a atTop (𝓝 0)) (n) :
    Δ_[1]^[n] (mahlerSeries a) (0 : ℤ_[p]) = a n :=
  calc Δ_[1]^[n] (mahlerSeries a) 0
  -- throw away terms after the n'th
  _ = Δ_[1]^[n] (fun k ↦ ∑ j ∈ range (n + 1), k.choose j • (a j)) 0 := by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      a : Nat → E
      ha : Filter.Tendsto a Filter.atTop (nhds 0)
      n : Nat
      ⊢ Eq (Nat.iterate (fwdDiff 1) n (⇑(PadicInt.mahlerSeries a)) 0) (Nat.iterate ( …
    -/
    simp only [fwdDiff_iter_eq_sum_shift, zero_add]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      a : Nat → E
      ha : Filter.Tendsto a Filter.atTop (nhds 0)
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HSMul.hSMul (HMul.hMul (HPow …
    -/
    refine Finset.sum_congr rfl fun j hj ↦ ?_
    rw [nsmul_one, nsmul_one,
      mahlerSeries_apply_nat ha (Nat.lt_succ.mp <| Finset.mem_range.mp hj), Nat.cast_id]
  -- bring `Δ_[1]` inside sum
  _ = ∑ j ∈ range (n + 1), Δ_[1]^[n] (fun k ↦ k.choose j • (a j)) 0 := by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      a : Nat → E
      ha : Filter.Tendsto a Filter.atTop (nhds 0)
      n : Nat
      ⊢ Eq (Nat.iterate (fwdDiff 1) n (fun k => (Finset.range (HAdd.hAdd n 1)).sum f …
    -/
    simp only [fwdDiff_iter_eq_sum_shift, smul_sum]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      a : Nat → E
      ha : Filter.Tendsto a Filter.atTop (nhds 0)
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => (Finset.range (HAdd.hAdd n 1 …
    -/
    rw [sum_comm]
    /-
      🎉 no goals
    -/
  -- bring `Δ_[1]` inside scalar-mult
  _ = ∑ j ∈ range (n + 1), (Δ_[1]^[n] (fun k ↦ k.choose j : ℕ → ℤ) 0) • (a j) := by
    simp only [fwdDiff_iter_eq_sum_shift, zero_add, sum_smul, smul_assoc, Nat.cast_id,
      natCast_zsmul]
  -- finish using `fwdDiff_iter_choose_zero`
  _ = a n := by
    simp only [fwdDiff_iter_choose_zero, ite_smul, one_smul, zero_smul, sum_ite_eq,
      Finset.mem_range, lt_add_iff_pos_right, zero_lt_one, ↓reduceIte]


/--
**Mahler's theorem**: for any continuous function `f` from `ℤ_[p]` to a `p`-adic Banach space, the
Mahler series with coefficients `n ↦ Δ_[1]^[n] f 0` converges to the original function `f`.
-/
lemma hasSum_mahler (f : C(ℤ_[p], E)) : HasSum (fun n ↦ mahlerTerm (Δ_[1]^[n] f 0) n) f := by
  -- First show `∑' n, mahler_term f n` converges to *something*.
  have : HasSum (fun n ↦ mahlerTerm (Δ_[1]^[n] f 0) n)
      (mahlerSeries (Δ_[1]^[·] f 0) : C(ℤ_[p], E)) :=
    hasSum_mahlerSeries (PadicInt.fwdDiff_tendsto_zero f)
  -- Now show that the sum of the Mahler terms must equal `f` on a dense set, so it is actually `f`.
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace (Padic p) E
    inst✝¹ : IsUltrametricDist E
    inst✝ : CompleteSpace E
    f : ContinuousMap (PadicInt p) E
    this : HasSum (fun n => PadicInt.mahlerTerm (Nat.iterate (fwdDiff 1) n (⇑f) 0) …
    ⊢ HasSum (fun n => PadicInt.mahlerTerm (Nat.iterate (fwdDiff 1) n (⇑f) 0) n) f
  -/
  convert this using 1
  refine ContinuousMap.coe_injective (PadicInt.denseRange_natCast.equalizer
    (map_continuous f) (map_continuous _) (funext fun n ↦ ?_))
  simpa only [Function.comp_apply, mahlerSeries_apply_nat (fwdDiff_tendsto_zero f) le_rfl,
    zero_add, sum_apply, Pi.smul_apply, nsmul_one] using (shift_eq_sum_fwdDiff_iter 1 f n 0)


variable (E) in
/--
The isometric equivalence from `C(ℤ_[p], E)` to the space of sequences in `E` tending to `0` given
by Mahler's theorem, for `E` a nonarchimedean `ℚ_[p]`-Banach space.
-/
noncomputable def mahlerEquiv : C(ℤ_[p], E) ≃ₗᵢ[ℚ_[p]] C₀(ℕ, E) where
  toFun f := ⟨⟨(Δ_[1]^[·] f 0), continuous_of_discreteTopology⟩,
    cocompact_eq_atTop (α := ℕ) ▸ fwdDiff_tendsto_zero f⟩
  invFun a := mahlerSeries a
  map_add' f g := by
    /-
      p✝ : Nat
      hp✝ : Fact (Nat.Prime p✝)
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      f g : ContinuousMap (PadicInt p) E
      ⊢ Eq ((fun f => { toFun := fun x => Nat.iterate (fwdDiff 1) x (⇑f) 0, continuo …
    -/
    ext x
    simp only [ContinuousMap.coe_add, fwdDiff_iter_add, Pi.add_apply,
      ZeroAtInftyContinuousMap.coe_mk, ZeroAtInftyContinuousMap.coe_add]
  map_smul' r f := by
    /-
      p✝ : Nat
      hp✝ : Fact (Nat.Prime p✝)
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      r : Padic p
      f : ContinuousMap (PadicInt p) E
      ⊢ Eq ({ toFun := fun f => { toFun := fun x => Nat.iterate (fwdDiff 1) x (⇑f) 0 …
    -/
    ext n
    simp only [ContinuousMap.coe_smul, RingHom.id_apply, ZeroAtInftyContinuousMap.coe_mk,
      ZeroAtInftyContinuousMap.coe_smul, Pi.smul_apply, fwdDiff_iter_const_smul]
  left_inv f := (hasSum_mahler f).tsum_eq
  right_inv a := ZeroAtInftyContinuousMap.ext <|
    fwdDiff_mahlerSeries (cocompact_eq_atTop (α := ℕ) ▸ zero_at_infty a)
  norm_map' f := by
    /-
      p✝ : Nat
      hp✝ : Fact (Nat.Prime p✝)
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      f : ContinuousMap (PadicInt p) E
      ⊢ Eq (Norm.norm ({ toFun := fun f => { toFun := fun x => Nat.iterate (fwdDiff  …
    -/
    simp only [LinearEquiv.coe_mk, ← ZeroAtInftyContinuousMap.norm_toBCF_eq_norm]
    /-
      p✝ : Nat
      hp✝ : Fact (Nat.Prime p✝)
      p : Nat
      hp : Fact (Nat.Prime p)
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace (Padic p) E
      inst✝¹ : IsUltrametricDist E
      inst✝ : CompleteSpace E
      f : ContinuousMap (PadicInt p) E
      ⊢ Eq (Norm.norm { toFun := fun x => Nat.iterate (fwdDiff 1) x (⇑f) 0, continuo …
    -/
    apply le_antisymm
    · exact BoundedContinuousFunction.norm_le_of_nonempty.mpr
        (fun n ↦ norm_fwdDiff_iter_apply_le 1 f 0 n)
      /-
        case a
        p✝ : Nat
        hp✝ : Fact (Nat.Prime p✝)
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace (Padic p) E
        inst✝¹ : IsUltrametricDist E
        inst✝ : CompleteSpace E
        f : ContinuousMap (PadicInt p) E
        ⊢ LE.le (Norm.norm f) (Norm.norm { toFun := fun x => Nat.iterate (fwdDiff 1) x …
      -/
    · rw [← (hasSum_mahler f).tsum_eq]
      /-
        case a
        p✝ : Nat
        hp✝ : Fact (Nat.Prime p✝)
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace (Padic p) E
        inst✝¹ : IsUltrametricDist E
        inst✝ : CompleteSpace E
        f : ContinuousMap (PadicInt p) E
        ⊢ LE.le (Norm.norm (tsum fun b => PadicInt.mahlerTerm (Nat.iterate (fwdDiff 1) …
      -/
      refine (norm_tsum_le _).trans (ciSup_le fun n ↦ ?_)
      /-
        case a
        p✝ : Nat
        hp✝ : Fact (Nat.Prime p✝)
        p : Nat
        hp : Fact (Nat.Prime p)
        E : Type u_1
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace (Padic p) E
        inst✝¹ : IsUltrametricDist E
        inst✝ : CompleteSpace E
        f : ContinuousMap (PadicInt p) E
        n : Nat
        ⊢ LE.le (Norm.norm (PadicInt.mahlerTerm (Nat.iterate (fwdDiff 1) n (⇑f) 0) n)) …
      -/
      refine le_trans (le_of_eq ?_) (BoundedContinuousFunction.norm_coe_le_norm _ n)
      simp only [ZeroAtInftyContinuousMap.toBCF_apply, ZeroAtInftyContinuousMap.coe_mk,
        norm_mahlerTerm, (hasSum_mahler f).tsum_eq]


lemma mahlerEquiv_apply (f : C(ℤ_[p], E)) : mahlerEquiv E f = fun n ↦ Δ_[1]^[n] f 0 := rfl


lemma mahlerEquiv_symm_apply (a : C₀(ℕ, E)) : (mahlerEquiv E).symm a = (mahlerSeries (p := p) a) :=
  rfl


