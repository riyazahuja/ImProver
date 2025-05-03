/-- The `k`th coefficient of the Taylor polynomial. -/
noncomputable def taylorCoeffWithin (f : ℝ → E) (k : ℕ) (s : Set ℝ) (x₀ : ℝ) : E :=
  (k ! : ℝ)⁻¹ • iteratedDerivWithin k f s x₀


/-- The Taylor polynomial with derivatives inside of a set `s`.

The Taylor polynomial is given by
$$∑_{k=0}^n \frac{(x - x₀)^k}{k!} f^{(k)}(x₀),$$
where $f^{(k)}(x₀)$ denotes the iterated derivative in the set `s`. -/
noncomputable def taylorWithin (f : ℝ → E) (n : ℕ) (s : Set ℝ) (x₀ : ℝ) : PolynomialModule ℝ E :=
  (Finset.range (n + 1)).sum fun k =>
    PolynomialModule.comp (Polynomial.X - Polynomial.C x₀)
      (PolynomialModule.single ℝ k (taylorCoeffWithin f k s x₀))


/-- The Taylor polynomial with derivatives inside of a set `s` considered as a function `ℝ → E`-/
noncomputable def taylorWithinEval (f : ℝ → E) (n : ℕ) (s : Set ℝ) (x₀ x : ℝ) : E :=
  PolynomialModule.eval x (taylorWithin f n s x₀)


theorem taylorWithin_succ (f : ℝ → E) (n : ℕ) (s : Set ℝ) (x₀ : ℝ) :
    taylorWithin f (n + 1) s x₀ = taylorWithin f n s x₀ +
      PolynomialModule.comp (Polynomial.X - Polynomial.C x₀)
      (PolynomialModule.single ℝ (n + 1) (taylorCoeffWithin f (n + 1) s x₀)) := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    n : Nat
    s : Set Real
    x₀ : Real
    ⊢ Eq (taylorWithin f (HAdd.hAdd n 1) s x₀) (HAdd.hAdd (taylorWithin f n s x₀)  …
  -/
  dsimp only [taylorWithin]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    n : Nat
    s : Set Real
    x₀ : Real
    ⊢ Eq ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun k => (PolynomialMod …
  -/
  rw [Finset.sum_range_succ]
  /-
    🎉 no goals
  -/


@[simp]
theorem taylorWithinEval_succ (f : ℝ → E) (n : ℕ) (s : Set ℝ) (x₀ x : ℝ) :
    taylorWithinEval f (n + 1) s x₀ x = taylorWithinEval f n s x₀ x +
      (((n + 1 : ℝ) * n !)⁻¹ * (x - x₀) ^ (n + 1)) • iteratedDerivWithin (n + 1) f s x₀ := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    n : Nat
    s : Set Real
    x₀ x : Real
    ⊢ Eq (taylorWithinEval f (HAdd.hAdd n 1) s x₀ x) (HAdd.hAdd (taylorWithinEval  …
  -/
  simp_rw [taylorWithinEval, taylorWithin_succ, LinearMap.map_add, PolynomialModule.comp_eval]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    n : Nat
    s : Set Real
    x₀ x : Real
    ⊢ Eq (HAdd.hAdd ((PolynomialModule.eval x) (taylorWithin f n s x₀)) ((Polynomi …
  -/
  congr
  simp only [Polynomial.eval_sub, Polynomial.eval_X, Polynomial.eval_C,
    PolynomialModule.eval_single, mul_inv_rev]
  /-
    case e_a
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    n : Nat
    s : Set Real
    x₀ x : Real
    ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub x x₀) (HAdd.hAdd n 1)) (taylorCoeffWit …
  -/
  dsimp only [taylorCoeffWithin]
  rw [← mul_smul, mul_comm, Nat.factorial_succ, Nat.cast_mul, Nat.cast_add, Nat.cast_one,
    mul_inv_rev]


/-- The Taylor polynomial of order zero evaluates to `f x`. -/
@[simp]
theorem taylor_within_zero_eval (f : ℝ → E) (s : Set ℝ) (x₀ x : ℝ) :
    taylorWithinEval f 0 s x₀ x = f x₀ := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    s : Set Real
    x₀ x : Real
    ⊢ Eq (taylorWithinEval f 0 s x₀ x) (f x₀)
  -/
  dsimp only [taylorWithinEval]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    s : Set Real
    x₀ x : Real
    ⊢ Eq ((PolynomialModule.eval x) (taylorWithin f 0 s x₀)) (f x₀)
  -/
  dsimp only [taylorWithin]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    s : Set Real
    x₀ x : Real
    ⊢ Eq ((PolynomialModule.eval x) ((Finset.range (HAdd.hAdd 0 1)).sum fun k => ( …
  -/
  dsimp only [taylorCoeffWithin]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    s : Set Real
    x₀ x : Real
    ⊢ Eq ((PolynomialModule.eval x) ((Finset.range (HAdd.hAdd 0 1)).sum fun k => ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Evaluating the Taylor polynomial at `x = x₀` yields `f x`. -/
@[simp]
theorem taylorWithinEval_self (f : ℝ → E) (n : ℕ) (s : Set ℝ) (x₀ : ℝ) :
    taylorWithinEval f n s x₀ x₀ = f x₀ := by
  induction n with
  | zero => exact taylor_within_zero_eval _ _ _ _
  | succ k hk => simp [hk]


theorem taylor_within_apply (f : ℝ → E) (n : ℕ) (s : Set ℝ) (x₀ x : ℝ) :
    taylorWithinEval f n s x₀ x =
      ∑ k ∈ Finset.range (n + 1), ((k ! : ℝ)⁻¹ * (x - x₀) ^ k) • iteratedDerivWithin k f s x₀ := by
  induction n with
  | zero => simp
  | succ k hk =>
    rw [taylorWithinEval_succ, Finset.sum_range_succ, hk]
    simp [Nat.factorial]


/-- If `f` is `n` times continuous differentiable on a set `s`, then the Taylor polynomial
  `taylorWithinEval f n s x₀ x` is continuous in `x₀`. -/
theorem continuousOn_taylorWithinEval {f : ℝ → E} {x : ℝ} {n : ℕ} {s : Set ℝ}
    (hs : UniqueDiffOn ℝ s) (hf : ContDiffOn ℝ n f s) :
    ContinuousOn (fun t => taylorWithinEval f n s t x) s := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    hf : ContDiffOn Real (↑n) f s
    ⊢ ContinuousOn (fun t => taylorWithinEval f n s t x) s
  -/
  simp_rw [taylor_within_apply]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    hf : ContDiffOn Real (↑n) f s
    ⊢ ContinuousOn (fun t => (Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSM …
  -/
  refine continuousOn_finset_sum (Finset.range (n + 1)) fun i hi => ?_
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    hf : ContDiffOn Real (↑n) f s
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ ContinuousOn (fun t => HSMul.hSMul (HMul.hMul (Inv.inv ↑i.factorial) (HPow.h …
  -/
  refine (continuousOn_const.mul ((continuousOn_const.sub continuousOn_id).pow _)).smul ?_
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    hf : ContDiffOn Real (↑n) f s
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ ContinuousOn (iteratedDerivWithin i f s) s
  -/
  rw [contDiffOn_nat_iff_continuousOn_differentiableOn_deriv hs] at hf
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    hf : And (∀ (m : Nat), LE.le m n → ContinuousOn (iteratedDerivWithin m f s) s) …
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ ContinuousOn (iteratedDerivWithin i f s) s
  -/
  cases' hf with hf_left
  /-
    case intro
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    hf_left : ∀ (m : Nat), LE.le m n → ContinuousOn (iteratedDerivWithin m f s) s
    right✝ : ∀ (m : Nat), LT.lt m n → DifferentiableOn Real (iteratedDerivWithin m …
    ⊢ ContinuousOn (iteratedDerivWithin i f s) s
  -/
  specialize hf_left i
  /-
    case intro
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    right✝ : ∀ (m : Nat), LT.lt m n → DifferentiableOn Real (iteratedDerivWithin m …
    hf_left : LE.le i n → ContinuousOn (iteratedDerivWithin i f s) s
    ⊢ ContinuousOn (iteratedDerivWithin i f s) s
  -/
  simp only [Finset.mem_range] at hi
  /-
    case intro
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    i : Nat
    right✝ : ∀ (m : Nat), LT.lt m n → DifferentiableOn Real (iteratedDerivWithin m …
    hf_left : LE.le i n → ContinuousOn (iteratedDerivWithin i f s) s
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ ContinuousOn (iteratedDerivWithin i f s) s
  -/
  refine hf_left ?_
  /-
    case intro
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    n : Nat
    s : Set Real
    hs : UniqueDiffOn Real s
    i : Nat
    right✝ : ∀ (m : Nat), LT.lt m n → DifferentiableOn Real (iteratedDerivWithin m …
    hf_left : LE.le i n → ContinuousOn (iteratedDerivWithin i f s) s
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ LE.le i n
  -/
  simp only [WithTop.coe_le_coe, Nat.cast_le, Nat.lt_succ_iff.mp hi]
  /-
    🎉 no goals
  -/


/-- Helper lemma for calculating the derivative of the monomial that appears in Taylor
expansions. -/
theorem monomial_has_deriv_aux (t x : ℝ) (n : ℕ) :
    HasDerivAt (fun y => (x - y) ^ (n + 1)) (-(n + 1) * (x - t) ^ n) t := by
  /-
    t x : Real
    n : Nat
    ⊢ HasDerivAt (fun y => HPow.hPow (HSub.hSub x y) (HAdd.hAdd n 1)) (HMul.hMul ( …
  -/
  simp_rw [sub_eq_neg_add]
  /-
    t x : Real
    n : Nat
    ⊢ HasDerivAt (fun y => HPow.hPow (HAdd.hAdd (Neg.neg y) x) (HAdd.hAdd n 1)) (H …
  -/
  rw [← neg_one_mul, mul_comm (-1 : ℝ), mul_assoc, mul_comm (-1 : ℝ), ← mul_assoc]
  /-
    t x : Real
    n : Nat
    ⊢ HasDerivAt (fun y => HPow.hPow (HAdd.hAdd (Neg.neg y) x) (HAdd.hAdd n 1)) (H …
  -/
  convert HasDerivAt.pow (n + 1) ((hasDerivAt_id t).neg.add_const x)
  /-
    case h.e'_9.h.e'_5.h.e'_5
    t x : Real
    n : Nat
    ⊢ Eq (HAdd.hAdd (↑n) 1) ↑(HAdd.hAdd n 1)
  -/
  simp only [Nat.cast_add, Nat.cast_one]
  /-
    🎉 no goals
  -/


theorem hasDerivWithinAt_taylor_coeff_within {f : ℝ → E} {x y : ℝ} {k : ℕ} {s t : Set ℝ}
    (ht : UniqueDiffWithinAt ℝ t y) (hs : s ∈ 𝓝[t] y)
    (hf : DifferentiableWithinAt ℝ (iteratedDerivWithin (k + 1) f s) s y) :
    HasDerivWithinAt
      (fun z => (((k + 1 : ℝ) * k !)⁻¹ * (x - z) ^ (k + 1)) • iteratedDerivWithin (k + 1) f s z)
      ((((k + 1 : ℝ) * k !)⁻¹ * (x - y) ^ (k + 1)) • iteratedDerivWithin (k + 2) f s y -
        ((k ! : ℝ)⁻¹ * (x - y) ^ k) • iteratedDerivWithin (k + 1) f s y) t y := by
  replace hf :
    HasDerivWithinAt (iteratedDerivWithin (k + 1) f s) (iteratedDerivWithin (k + 2) f s y) t y := by
    convert (hf.mono_of_mem_nhdsWithin hs).hasDerivWithinAt using 1
    rw [iteratedDerivWithin_succ (ht.mono_nhds (nhdsWithin_le_iff.mpr hs))]
    exact (derivWithin_of_mem_nhdsWithin hs ht hf).symm
  have : HasDerivWithinAt (fun t => ((k + 1 : ℝ) * k !)⁻¹ * (x - t) ^ (k + 1))
      (-((k ! : ℝ)⁻¹ * (x - y) ^ k)) t y := by
    -- Commuting the factors:
    have : -((k ! : ℝ)⁻¹ * (x - y) ^ k) = ((k + 1 : ℝ) * k !)⁻¹ * (-(k + 1) * (x - y) ^ k) := by
      field_simp; ring
    rw [this]
    exact (monomial_has_deriv_aux y x _).hasDerivWithinAt.const_mul _
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x y : Real
    k : Nat
    s t : Set Real
    ht : UniqueDiffWithinAt Real t y
    hs : Membership.mem (nhdsWithin y t) s
    hf : HasDerivWithinAt (iteratedDerivWithin (HAdd.hAdd k 1) f s) (iteratedDeriv …
    this : HasDerivWithinAt (fun t => HMul.hMul (Inv.inv (HMul.hMul (HAdd.hAdd (↑k …
    ⊢ HasDerivWithinAt (fun z => HSMul.hSMul (HMul.hMul (Inv.inv (HMul.hMul (HAdd. …
  -/
  convert this.smul hf using 1
  /-
    case h.e'_9
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x y : Real
    k : Nat
    s t : Set Real
    ht : UniqueDiffWithinAt Real t y
    hs : Membership.mem (nhdsWithin y t) s
    hf : HasDerivWithinAt (iteratedDerivWithin (HAdd.hAdd k 1) f s) (iteratedDeriv …
    this : HasDerivWithinAt (fun t => HMul.hMul (Inv.inv (HMul.hMul (HAdd.hAdd (↑k …
    ⊢ Eq (HSub.hSub (HSMul.hSMul (HMul.hMul (Inv.inv (HMul.hMul (HAdd.hAdd (↑k) 1) …
  -/
  field_simp
  /-
    case h.e'_9
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x y : Real
    k : Nat
    s t : Set Real
    ht : UniqueDiffWithinAt Real t y
    hs : Membership.mem (nhdsWithin y t) s
    hf : HasDerivWithinAt (iteratedDerivWithin (HAdd.hAdd k 1) f s) (iteratedDeriv …
    this : HasDerivWithinAt (fun t => HMul.hMul (Inv.inv (HMul.hMul (HAdd.hAdd (↑k …
    ⊢ Eq (HSub.hSub (HSMul.hSMul (HDiv.hDiv (HPow.hPow (HSub.hSub x y) (HAdd.hAdd  …
  -/
  rw [neg_div, neg_smul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- Calculate the derivative of the Taylor polynomial with respect to `x₀`.

Version for arbitrary sets -/
theorem hasDerivWithinAt_taylorWithinEval {f : ℝ → E} {x y : ℝ} {n : ℕ} {s s' : Set ℝ}
    (hs'_unique : UniqueDiffWithinAt ℝ s' y) (hs_unique : UniqueDiffOn ℝ s) (hs' : s' ∈ 𝓝[s] y)
    (hy : y ∈ s') (h : s' ⊆ s) (hf : ContDiffOn ℝ n f s)
    (hf' : DifferentiableWithinAt ℝ (iteratedDerivWithin n f s) s y) :
    HasDerivWithinAt (fun t => taylorWithinEval f n s t x)
      (((n ! : ℝ)⁻¹ * (x - y) ^ n) • iteratedDerivWithin (n + 1) f s y) s' y := by
  induction n with
  | zero =>
    simp only [taylor_within_zero_eval, Nat.factorial_zero, Nat.cast_one, inv_one, pow_zero,
      mul_one, zero_add, one_smul]
    simp only [iteratedDerivWithin_zero] at hf'
    rw [iteratedDerivWithin_one (hs_unique _ (h hy))]
    exact hf'.hasDerivWithinAt.mono h
  | succ k hk =>
    simp_rw [Nat.add_succ, taylorWithinEval_succ]
    simp only [add_zero, Nat.factorial_succ, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
    have coe_lt_succ : (k : WithTop ℕ) < k.succ := Nat.cast_lt.2 k.lt_succ_self
    have hdiff : DifferentiableOn ℝ (iteratedDerivWithin k f s) s' :=
      (hf.differentiableOn_iteratedDerivWithin (mod_cast coe_lt_succ) hs_unique).mono h
    specialize hk hf.of_succ ((hdiff y hy).mono_of_mem_nhdsWithin hs')
    convert hk.add (hasDerivWithinAt_taylor_coeff_within hs'_unique
      (nhdsWithin_mono _ h self_mem_nhdsWithin) hf') using 1
    exact (add_sub_cancel _ _).symm


/-- Calculate the derivative of the Taylor polynomial with respect to `x₀`.

Version for open intervals -/
theorem taylorWithinEval_hasDerivAt_Ioo {f : ℝ → E} {a b t : ℝ} (x : ℝ) {n : ℕ} (hx : a < b)
    (ht : t ∈ Ioo a b) (hf : ContDiffOn ℝ n f (Icc a b))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Icc a b)) (Ioo a b)) :
    HasDerivAt (fun y => taylorWithinEval f n (Icc a b) y x)
      (((n ! : ℝ)⁻¹ * (x - t) ^ n) • iteratedDerivWithin (n + 1) f (Icc a b) t) t :=
  have h_nhds : Ioo a b ∈ 𝓝 t := isOpen_Ioo.mem_nhds ht
  have h_nhds' : Ioo a b ∈ 𝓝[Icc a b] t := nhdsWithin_le_nhds h_nhds
  (hasDerivWithinAt_taylorWithinEval (uniqueDiffWithinAt_Ioo ht) (uniqueDiffOn_Icc hx) h_nhds' ht
    Ioo_subset_Icc_self hf <| (hf' t ht).mono_of_mem_nhdsWithin h_nhds').hasDerivAt h_nhds


/-- Calculate the derivative of the Taylor polynomial with respect to `x₀`.

Version for closed intervals -/
theorem hasDerivWithinAt_taylorWithinEval_at_Icc {f : ℝ → E} {a b t : ℝ} (x : ℝ) {n : ℕ}
    (hx : a < b) (ht : t ∈ Icc a b) (hf : ContDiffOn ℝ n f (Icc a b))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Icc a b)) (Icc a b)) :
    HasDerivWithinAt (fun y => taylorWithinEval f n (Icc a b) y x)
      (((n ! : ℝ)⁻¹ * (x - t) ^ n) • iteratedDerivWithin (n + 1) f (Icc a b) t) (Icc a b) t :=
  hasDerivWithinAt_taylorWithinEval (uniqueDiffOn_Icc hx t ht) (uniqueDiffOn_Icc hx)
    self_mem_nhdsWithin ht rfl.subset hf (hf' t ht)


/-- **Taylor's theorem** with the general mean value form of the remainder.

We assume that `f` is `n+1`-times continuously differentiable in the closed set `Icc x₀ x` and
`n+1`-times differentiable on the open set `Ioo x₀ x`, and `g` is a differentiable function on
`Ioo x₀ x` and continuous on `Icc x₀ x`. Then there exists an `x' ∈ Ioo x₀ x` such that
$$f(x) - (P_n f)(x₀, x) = \frac{(x - x')^n}{n!} \frac{g(x) - g(x₀)}{g' x'},$$
where $P_n f$ denotes the Taylor polynomial of degree $n$. -/
theorem taylor_mean_remainder {f : ℝ → ℝ} {g g' : ℝ → ℝ} {x x₀ : ℝ} {n : ℕ} (hx : x₀ < x)
    (hf : ContDiffOn ℝ n f (Icc x₀ x))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Icc x₀ x)) (Ioo x₀ x))
    (gcont : ContinuousOn g (Icc x₀ x))
    (gdiff : ∀ x_1 : ℝ, x_1 ∈ Ioo x₀ x → HasDerivAt g (g' x_1) x_1)
    (g'_ne : ∀ x_1 : ℝ, x_1 ∈ Ioo x₀ x → g' x_1 ≠ 0) :
    ∃ x' ∈ Ioo x₀ x, f x - taylorWithinEval f n (Icc x₀ x) x₀ x =
    ((x - x') ^ n / n ! * (g x - g x₀) / g' x') • iteratedDerivWithin (n + 1) f (Icc x₀ x) x' := by
  -- We apply the mean value theorem
  rcases exists_ratio_hasDerivAt_eq_ratio_slope (fun t => taylorWithinEval f n (Icc x₀ x) t x)
      (fun t => ((n ! : ℝ)⁻¹ * (x - t) ^ n) • iteratedDerivWithin (n + 1) f (Icc x₀ x) t) hx
      (continuousOn_taylorWithinEval (uniqueDiffOn_Icc hx) hf)
      (fun _ hy => taylorWithinEval_hasDerivAt_Ioo x hx hy hf hf') g g' gcont gdiff with ⟨y, hy, h⟩
  /-
    case intro.intro
    f g g' : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn g (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt g (g' x …
    g'_ne : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → Ne (g' x_1) 0
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HMul.hMul (HSub.hSub (g x) (g x₀)) (HSMul.hSMul (HMul.hMul (Inv.inv ↑n …
    ⊢ Exists fun x' => And (Membership.mem (Set.Ioo x₀ x) x') (Eq (HSub.hSub (f x) …
  -/
  use y, hy
  -- The rest is simplifications and trivial calculations
  /-
    case right
    f g g' : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn g (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt g (g' x …
    g'_ne : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → Ne (g' x_1) 0
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HMul.hMul (HSub.hSub (g x) (g x₀)) (HSMul.hSMul (HMul.hMul (Inv.inv ↑n …
    ⊢ Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSMul …
  -/
  simp only [taylorWithinEval_self] at h
  /-
    case right
    f g g' : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn g (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt g (g' x …
    g'_ne : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → Ne (g' x_1) 0
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HMul.hMul (HSub.hSub (g x) (g x₀)) (HSMul.hSMul (HMul.hMul (Inv.inv ↑n …
    ⊢ Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSMul …
  -/
  rw [mul_comm, ← div_left_inj' (g'_ne y hy), mul_div_cancel_right₀ _ (g'_ne y hy)] at h
  /-
    case right
    f g g' : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn g (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt g (g' x …
    g'_ne : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → Ne (g' x_1) 0
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HDiv.hDiv (HMul.hMul (HSMul.hSMul (HMul.hMul (Inv.inv ↑n.factorial) (H …
    ⊢ Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSMul …
  -/
  rw [← h]
  /-
    case right
    f g g' : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn g (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt g (g' x …
    g'_ne : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → Ne (g' x_1) 0
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HDiv.hDiv (HMul.hMul (HSMul.hSMul (HMul.hMul (Inv.inv ↑n.factorial) (H …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HSMul.hSMul (HMul.hMul (Inv.inv ↑n.factorial) (HPo …
  -/
  field_simp [g'_ne y hy]
  /-
    case right
    f g g' : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn g (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt g (g' x …
    g'_ne : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → Ne (g' x_1) 0
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HDiv.hDiv (HMul.hMul (HSMul.hSMul (HMul.hMul (Inv.inv ↑n.factorial) (H …
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (HSub.hSub x y) n) (iteratedDerivWithin  …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Taylor's theorem** with the Lagrange form of the remainder.

We assume that `f` is `n+1`-times continuously differentiable in the closed set `Icc x₀ x` and
`n+1`-times differentiable on the open set `Ioo x₀ x`. Then there exists an `x' ∈ Ioo x₀ x` such
that $$f(x) - (P_n f)(x₀, x) = \frac{f^{(n+1)}(x') (x - x₀)^{n+1}}{(n+1)!},$$
where $P_n f$ denotes the Taylor polynomial of degree $n$ and $f^{(n+1)}$ is the $n+1$-th iterated
derivative. -/
theorem taylor_mean_remainder_lagrange {f : ℝ → ℝ} {x x₀ : ℝ} {n : ℕ} (hx : x₀ < x)
    (hf : ContDiffOn ℝ n f (Icc x₀ x))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Icc x₀ x)) (Ioo x₀ x)) :
    ∃ x' ∈ Ioo x₀ x, f x - taylorWithinEval f n (Icc x₀ x) x₀ x =
      iteratedDerivWithin (n + 1) f (Icc x₀ x) x' * (x - x₀) ^ (n + 1) / (n + 1)! := by
  /-
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    ⊢ Exists fun x' => And (Membership.mem (Set.Ioo x₀ x) x') (Eq (HSub.hSub (f x) …
  -/
  have gcont : ContinuousOn (fun t : ℝ => (x - t) ^ (n + 1)) (Icc x₀ x) := by fun_prop
  have xy_ne : ∀ y : ℝ, y ∈ Ioo x₀ x → (x - y) ^ n ≠ 0 := by
    intro y hy
    refine pow_ne_zero _ ?_
    rw [mem_Ioo] at hy
    rw [sub_ne_zero]
    exact hy.2.ne'
  have hg' : ∀ y : ℝ, y ∈ Ioo x₀ x → -(↑n + 1) * (x - y) ^ n ≠ 0 := fun y hy =>
    mul_ne_zero (neg_ne_zero.mpr (Nat.cast_add_one_ne_zero n)) (xy_ne y hy)
  -- We apply the general theorem with g(t) = (x - t)^(n+1)
  rcases taylor_mean_remainder hx hf hf' gcont (fun y _ => monomial_has_deriv_aux y x _) hg' with
    ⟨y, hy, h⟩
  /-
    case intro.intro
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn (fun t => HPow.hPow (HSub.hSub x t) (HAdd.hAdd n 1)) (Set …
    xy_ne : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HPow.hPow (HSub.hS …
    hg' : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HMul.hMul (Neg.neg ( …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Exists fun x' => And (Membership.mem (Set.Ioo x₀ x) x') (Eq (HSub.hSub (f x) …
  -/
  use y, hy
  /-
    case right
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn (fun t => HPow.hPow (HSub.hSub x t) (HAdd.hAdd n 1)) (Set …
    xy_ne : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HPow.hPow (HSub.hS …
    hg' : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HMul.hMul (Neg.neg ( …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HDiv.hDiv ( …
  -/
  simp only [sub_self, zero_pow, Ne, Nat.succ_ne_zero, not_false_iff, zero_sub, mul_neg] at h
  /-
    case right
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn (fun t => HPow.hPow (HSub.hSub x t) (HAdd.hAdd n 1)) (Set …
    xy_ne : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HPow.hPow (HSub.hS …
    hg' : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HMul.hMul (Neg.neg ( …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HDiv.hDiv ( …
  -/
  rw [h, neg_div, ← div_neg, neg_mul, neg_neg]
  /-
    case right
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn (fun t => HPow.hPow (HSub.hSub x t) (HAdd.hAdd n 1)) (Set …
    xy_ne : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HPow.hPow (HSub.hS …
    hg' : ∀ (y : Real), Membership.mem (Set.Ioo x₀ x) y → Ne (HMul.hMul (Neg.neg ( …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HPow.hPow (HSub.hSub x y)  …
  -/
  field_simp [xy_ne y hy, Nat.factorial]; ring
                                          /-
                                            🎉 no goals
                                          -/


/-- **Taylor's theorem** with the Cauchy form of the remainder.

We assume that `f` is `n+1`-times continuously differentiable on the closed set `Icc x₀ x` and
`n+1`-times differentiable on the open set `Ioo x₀ x`. Then there exists an `x' ∈ Ioo x₀ x` such
that $$f(x) - (P_n f)(x₀, x) = \frac{f^{(n+1)}(x') (x - x')^n (x-x₀)}{n!},$$
where $P_n f$ denotes the Taylor polynomial of degree $n$ and $f^{(n+1)}$ is the $n+1$-th iterated
derivative. -/
theorem taylor_mean_remainder_cauchy {f : ℝ → ℝ} {x x₀ : ℝ} {n : ℕ} (hx : x₀ < x)
    (hf : ContDiffOn ℝ n f (Icc x₀ x))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Icc x₀ x)) (Ioo x₀ x)) :
    ∃ x' ∈ Ioo x₀ x, f x - taylorWithinEval f n (Icc x₀ x) x₀ x =
      iteratedDerivWithin (n + 1) f (Icc x₀ x) x' * (x - x') ^ n / n ! * (x - x₀) := by
  /-
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    ⊢ Exists fun x' => And (Membership.mem (Set.Ioo x₀ x) x') (Eq (HSub.hSub (f x) …
  -/
  have gcont : ContinuousOn id (Icc x₀ x) := by fun_prop
  have gdiff : ∀ x_1 : ℝ, x_1 ∈ Ioo x₀ x → HasDerivAt id ((fun _ : ℝ => (1 : ℝ)) x_1) x_1 :=
    fun _ _ => hasDerivAt_id _
  -- We apply the general theorem with g = id
  /-
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn id (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt id ((fu …
    ⊢ Exists fun x' => And (Membership.mem (Set.Ioo x₀ x) x') (Eq (HSub.hSub (f x) …
  -/
  rcases taylor_mean_remainder hx hf hf' gcont gdiff fun _ _ => by simp with ⟨y, hy, h⟩
  /-
    case intro.intro
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn id (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt id ((fu …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Exists fun x' => And (Membership.mem (Set.Ioo x₀ x) x') (Eq (HSub.hSub (f x) …
  -/
  use y, hy
  /-
    case right
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn id (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt id ((fu …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HMul.hMul ( …
  -/
  rw [h]
  /-
    case right
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn id (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt id ((fu …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HPow.hPow (HSub.hSub x y)  …
  -/
  field_simp [n.factorial_ne_zero]
  /-
    case right
    f : Real → Real
    x x₀ : Real
    n : Nat
    hx : LT.lt x₀ x
    hf : ContDiffOn Real (↑n) f (Set.Icc x₀ x)
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc x₀ x)) (Set.Ioo  …
    gcont : ContinuousOn id (Set.Icc x₀ x)
    gdiff : ∀ (x_1 : Real), Membership.mem (Set.Ioo x₀ x) x_1 → HasDerivAt id ((fu …
    y : Real
    hy : Membership.mem (Set.Ioo x₀ x) y
    h : Eq (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc x₀ x) x₀ x)) (HSMul.hSM …
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (HSub.hSub x y) n) (HSub.hSub x x₀)) (it …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Taylor's theorem** with a polynomial bound on the remainder

We assume that `f` is `n+1`-times continuously differentiable on the closed set `Icc a b`.
The difference of `f` and its `n`-th Taylor polynomial can be estimated by
`C * (x - a)^(n+1) / n!` where `C` is a bound for the `n+1`-th iterated derivative of `f`. -/
theorem taylor_mean_remainder_bound {f : ℝ → E} {a b C x : ℝ} {n : ℕ} (hab : a ≤ b)
    (hf : ContDiffOn ℝ (n + 1) f (Icc a b)) (hx : x ∈ Icc a b)
    (hC : ∀ y ∈ Icc a b, ‖iteratedDerivWithin (n + 1) f (Icc a b) y‖ ≤ C) :
    ‖f x - taylorWithinEval f n (Icc a b) a x‖ ≤ C * (x - a) ^ (n + 1) / n ! := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C x : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    hx : Membership.mem (Set.Icc a b) x
    hC : ∀ (y : Real), Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iterated …
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a x))) …
  -/
  rcases eq_or_lt_of_le hab with (rfl | h)
    /-
      case inl
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a C x : Real
      n : Nat
      hab : LE.le a a
      hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a a)
      hx : Membership.mem (Set.Icc a a) x
      hC : ∀ (y : Real), Membership.mem (Set.Icc a a) y → LE.le (Norm.norm (iterated …
      ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a a) a x))) …
    -/
  · rw [Icc_self, mem_singleton_iff] at hx
    /-
      case inl
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a C x : Real
      n : Nat
      hab : LE.le a a
      hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a a)
      hx : Eq x a
      hC : ∀ (y : Real), Membership.mem (Set.Icc a a) y → LE.le (Norm.norm (iterated …
      ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a a) a x))) …
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
  -- The nth iterated derivative is differentiable
  have hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Icc a b)) (Icc a b) :=
    hf.differentiableOn_iteratedDerivWithin (mod_cast n.lt_succ_self)
      (uniqueDiffOn_Icc h)
  -- We can uniformly bound the derivative of the Taylor polynomial
  have h' : ∀ y ∈ Ico a x,
      ‖((n ! : ℝ)⁻¹ * (x - y) ^ n) • iteratedDerivWithin (n + 1) f (Icc a b) y‖ ≤
        (n ! : ℝ)⁻¹ * |x - a| ^ n * C := by
    rintro y ⟨hay, hyx⟩
    rw [norm_smul, Real.norm_eq_abs]
    gcongr
    · rw [abs_mul, abs_pow, abs_inv, Nat.abs_cast]
      gcongr
      exact sub_nonneg.2 hyx.le
    -- Estimate the iterated derivative by `C`
    · exact hC y ⟨hay, hyx.le.trans hx.2⟩
  -- Apply the mean value theorem for vector valued functions:
  have A : ∀ t ∈ Icc a x, HasDerivWithinAt (fun y => taylorWithinEval f n (Icc a b) y x)
      (((↑n !)⁻¹ * (x - t) ^ n) • iteratedDerivWithin (n + 1) f (Icc a b) t) (Icc a x) t := by
    intro t ht
    have I : Icc a x ⊆ Icc a b := Icc_subset_Icc_right hx.2
    exact (hasDerivWithinAt_taylorWithinEval_at_Icc x h (I ht) hf.of_succ hf').mono I
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C x : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    hx : Membership.mem (Set.Icc a b) x
    hC : ∀ (y : Real), Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iterated …
    h : LT.lt a b
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc a b)) (Set.Icc a …
    h' : ∀ (y : Real), Membership.mem (Set.Ico a x) y → LE.le (Norm.norm (HSMul.hS …
    A : ∀ (t : Real), Membership.mem (Set.Icc a x) t → HasDerivWithinAt (fun y =>  …
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a x))) …
  -/
  have := norm_image_sub_le_of_norm_deriv_le_segment' A h' x (right_mem_Icc.2 hx.1)
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C x : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    hx : Membership.mem (Set.Icc a b) x
    hC : ∀ (y : Real), Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iterated …
    h : LT.lt a b
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc a b)) (Set.Icc a …
    h' : ∀ (y : Real), Membership.mem (Set.Ico a x) y → LE.le (Norm.norm (HSMul.hS …
    A : ∀ (t : Real), Membership.mem (Set.Icc a x) t → HasDerivWithinAt (fun y =>  …
    this : LE.le (Norm.norm (HSub.hSub (taylorWithinEval f n (Set.Icc a b) x x) (t …
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a x))) …
  -/
  simp only [taylorWithinEval_self] at this
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C x : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    hx : Membership.mem (Set.Icc a b) x
    hC : ∀ (y : Real), Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iterated …
    h : LT.lt a b
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc a b)) (Set.Icc a …
    h' : ∀ (y : Real), Membership.mem (Set.Ico a x) y → LE.le (Norm.norm (HSMul.hS …
    A : ∀ (t : Real), Membership.mem (Set.Icc a x) t → HasDerivWithinAt (fun y =>  …
    this : LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a …
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a x))) …
  -/
  refine this.trans_eq ?_
  -- The rest is a trivial calculation
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C x : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    hx : Membership.mem (Set.Icc a b) x
    hC : ∀ (y : Real), Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iterated …
    h : LT.lt a b
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc a b)) (Set.Icc a …
    h' : ∀ (y : Real), Membership.mem (Set.Ico a x) y → LE.le (Norm.norm (HSMul.hS …
    A : ∀ (t : Real), Membership.mem (Set.Icc a x) t → HasDerivWithinAt (fun y =>  …
    this : LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (HPow.hPow (abs ( …
  -/
  rw [abs_of_nonneg (sub_nonneg.mpr hx.1)]
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C x : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    hx : Membership.mem (Set.Icc a b) x
    hC : ∀ (y : Real), Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iterated …
    h : LT.lt a b
    hf' : DifferentiableOn Real (iteratedDerivWithin n f (Set.Icc a b)) (Set.Icc a …
    h' : ∀ (y : Real), Membership.mem (Set.Ico a x) y → LE.le (Norm.norm (HSMul.hS …
    A : ∀ (t : Real), Membership.mem (Set.Icc a x) t → HasDerivWithinAt (fun y =>  …
    this : LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (HPow.hPow (HSub. …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Taylor's theorem** with a polynomial bound on the remainder

We assume that `f` is `n+1`-times continuously differentiable on the closed set `Icc a b`.
There exists a constant `C` such that for all `x ∈ Icc a b` the difference of `f` and its `n`-th
Taylor polynomial can be estimated by `C * (x - a)^(n+1)`. -/
theorem exists_taylor_mean_remainder_bound {f : ℝ → E} {a b : ℝ} {n : ℕ} (hab : a ≤ b)
    (hf : ContDiffOn ℝ (n + 1) f (Icc a b)) :
    ∃ C, ∀ x ∈ Icc a b, ‖f x - taylorWithinEval f n (Icc a b) a x‖ ≤ C * (x - a) ^ (n + 1) := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    ⊢ Exists fun C => ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.n …
  -/
  rcases eq_or_lt_of_le hab with (rfl | h)
    /-
      case inl
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a : Real
      n : Nat
      hab : LE.le a a
      hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a a)
      ⊢ Exists fun C => ∀ (x : Real), Membership.mem (Set.Icc a a) x → LE.le (Norm.n …
    -/
  · refine ⟨0, fun x hx => ?_⟩
    /-
      case inl
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a : Real
      n : Nat
      hab : LE.le a a
      hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a a)
      x : Real
      hx : Membership.mem (Set.Icc a a) x
      ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a a) a x))) …
    -/
    have : x = a := by simpa [← le_antisymm_iff] using hx
    /-
      case inl
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a : Real
      n : Nat
      hab : LE.le a a
      hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a a)
      x : Real
      hx : Membership.mem (Set.Icc a a) x
      this : Eq x a
      ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a a) a x))) …
    -/
    simp [← this]
    /-
      🎉 no goals
    -/
  -- We estimate by the supremum of the norm of the iterated derivative
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    h : LT.lt a b
    ⊢ Exists fun C => ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.n …
  -/
  let g : ℝ → ℝ := fun y => ‖iteratedDerivWithin (n + 1) f (Icc a b) y‖
  /-
    case inr
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    h : LT.lt a b
    g : Real → Real := fun y => Norm.norm (iteratedDerivWithin (HAdd.hAdd n 1) f ( …
    ⊢ Exists fun C => ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.n …
  -/
  use SupSet.sSup (g '' Icc a b) / (n !)
  /-
    case h
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    h : LT.lt a b
    g : Real → Real := fun y => Norm.norm (iteratedDerivWithin (HAdd.hAdd n 1) f ( …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  intro x hx
  /-
    case h
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    h : LT.lt a b
    g : Real → Real := fun y => Norm.norm (iteratedDerivWithin (HAdd.hAdd n 1) f ( …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a x))) …
  -/
  rw [div_mul_eq_mul_div₀]
  /-
    case h
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    h : LT.lt a b
    g : Real → Real := fun y => Norm.norm (iteratedDerivWithin (HAdd.hAdd n 1) f ( …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (taylorWithinEval f n (Set.Icc a b) a x))) …
  -/
  refine taylor_mean_remainder_bound hab hf hx fun y => ?_
  /-
    case h
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    n : Nat
    hab : LE.le a b
    hf : ContDiffOn Real (HAdd.hAdd (↑n) 1) f (Set.Icc a b)
    h : LT.lt a b
    g : Real → Real := fun y => Norm.norm (iteratedDerivWithin (HAdd.hAdd n 1) f ( …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    y : Real
    ⊢ Membership.mem (Set.Icc a b) y → LE.le (Norm.norm (iteratedDerivWithin (HAdd …
  -/
  exact (hf.continuousOn_iteratedDerivWithin rfl.le <| uniqueDiffOn_Icc h).norm.le_sSup_image_Icc
  /-
    🎉 no goals
  -/

