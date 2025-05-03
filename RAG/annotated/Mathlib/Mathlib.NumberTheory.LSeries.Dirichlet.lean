/-- `δ` is the function underlying the arithmetic function `1`. -/
lemma ArithmeticFunction.one_eq_delta : ↗(1 : ArithmeticFunction ℂ) = δ := by
  /-
    ⊢ Eq (fun n => 1 n) LSeries.delta
  -/
  ext
  /-
    case h
    x✝ : Nat
    ⊢ Eq (1 x✝) (LSeries.delta x✝)
  -/
  simp only [one_apply, LSeries.delta]
  /-
    🎉 no goals
  -/



lemma not_LSeriesSummable_moebius_at_one : ¬ LSeriesSummable ↗μ 1 := by
  /-
    ⊢ Not (LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) 1)
  -/
  intro h
  /-
    h : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) 1
    ⊢ False
  -/
  refine not_summable_one_div_on_primes <| summable_ofReal.mp <| Summable.of_neg ?_
  /-
    h : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) 1
    ⊢ Summable fun b => Neg.neg ↑((setOf fun p => Nat.Prime p).indicator (fun n => …
  -/
  simp only [← Pi.neg_def, Set.indicator_comp_of_zero ofReal_zero, ofReal_inv, ofReal_natCast]
  /-
    h : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) 1
    ⊢ Summable (Neg.neg fun i => ↑((setOf fun p => Nat.Prime p).indicator (fun n = …
  -/
  refine (h.indicator {n | n.Prime}).congr (fun n ↦ ?_)
  /-
    h : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) 1
    n : Nat
    ⊢ Eq ((setOf fun n => Nat.Prime n).indicator (LSeries.term (fun n => ↑(Arithme …
  -/
  by_cases hn : n ∈ {p | p.Prime}
  · simp only [Pi.neg_apply, Set.indicator_of_mem hn, term_of_ne_zero hn.ne_zero,
      moebius_apply_prime hn, cpow_one, push_cast, neg_div]
    /-
      case neg
      h : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) 1
      n : Nat
      hn : Not (Membership.mem (setOf fun p => Nat.Prime p) n)
      ⊢ Eq ((setOf fun n => Nat.Prime n).indicator (LSeries.term (fun n => ↑(Arithme …
    -/
  · simp only [one_div, Pi.neg_apply, Set.indicator_of_not_mem hn, ofReal_zero, neg_zero]
    /-
      🎉 no goals
    -/


/-- The L-series of the Möbius function converges absolutely at `s` if and only if `re s > 1`. -/
lemma LSeriesSummable_moebius_iff {s : ℂ} : LSeriesSummable ↗μ s ↔ 1 < s.re := by
  /-
    s : Complex
    ⊢ Iff (LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) s) (LT.lt 1  …
  -/
  refine ⟨fun H ↦ ?_, LSeriesSummable_of_bounded_of_one_lt_re (m := 1) fun n _ ↦ ?_⟩
    /-
      case refine_1
      s : Complex
      H : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) s
      ⊢ LT.lt 1 s.re
    -/
  · by_contra! h
    /-
      case refine_1
      s : Complex
      H : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) s
      h : LE.le s.re 1
      ⊢ False
    -/
    have h' : s.re ≤ (1 : ℂ).re := by simp only [one_re, h]
    /-
      case refine_1
      s : Complex
      H : LSeriesSummable (fun n => ↑(ArithmeticFunction.moebius n)) s
      h : LE.le s.re 1
      h' : LE.le s.re (Complex.re 1)
      ⊢ False
    -/
    exact not_LSeriesSummable_moebius_at_one <| LSeriesSummable.of_re_le_re h' H
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      s : Complex
      n : Nat
      x✝ : Ne n 0
      ⊢ LE.le (Complex.abs ↑(ArithmeticFunction.moebius n)) 1
    -/
  · rw [abs_intCast] -- not done by `norm_cast`
    /-
      case refine_2
      s : Complex
      n : Nat
      x✝ : Ne n 0
      ⊢ LE.le (abs ↑(ArithmeticFunction.moebius n)) 1
    -/
    norm_cast
    /-
      case refine_2
      s : Complex
      n : Nat
      x✝ : Ne n 0
      ⊢ LE.le (abs (ArithmeticFunction.moebius n)) 1
    -/
    exact abs_moebius_le_one
    /-
      🎉 no goals
    -/


/-- The abscissa of absolute convergence of the L-series of the Möbius function is `1`. -/
lemma abscissaOfAbsConv_moebius : abscissaOfAbsConv ↗μ = 1 := by
  simpa only [abscissaOfAbsConv, LSeriesSummable_moebius_iff, ofReal_re, Set.Ioi_def,
    EReal.image_coe_Ioi, EReal.coe_one] using csInf_Ioo <| EReal.coe_lt_top _


open scoped ArithmeticFunction.zeta in
lemma ArithmeticFunction.const_one_eq_zeta {R : Type*} [Semiring R] {n : ℕ} (hn : n ≠ 0) :
    (1 : ℕ → R) n = (ζ ·) n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    hn : Ne n 0
    ⊢ Eq (1 n) ↑((fun x => ArithmeticFunction.zeta x) n)
  -/
  simp only [Pi.one_apply, zeta_apply, hn, ↓reduceIte, cast_one]
  /-
    🎉 no goals
  -/


lemma LSeries.one_convolution_eq_zeta_convolution {R : Type*} [Semiring R] (f : ℕ → R) :
    (1 : ℕ → R) ⍟ f = ((ArithmeticFunction.zeta ·) : ℕ → R) ⍟ f :=
  convolution_congr ArithmeticFunction.const_one_eq_zeta fun _ ↦ rfl


lemma LSeries.convolution_one_eq_convolution_zeta {R : Type*} [Semiring R] (f : ℕ → R) :
    f ⍟ (1 : ℕ → R) = f ⍟ ((ArithmeticFunction.zeta ·) : ℕ → R) :=
  convolution_congr (fun _ ↦ rfl) ArithmeticFunction.const_one_eq_zeta


/-- `χ₁` is (local) notation for the (necessarily trivial) Dirichlet character modulo `1`. -/
local notation (name := Dchar_one) "χ₁" => (1 : DirichletCharacter ℂ 1)


open ArithmeticFunction in
/-- The arithmetic function associated to a Dirichlet character is multiplicative. -/
lemma isMultiplicative_toArithmeticFunction {N : ℕ} {R : Type*} [CommMonoidWithZero R]
    (χ : DirichletCharacter R N) :
    (toArithmeticFunction (χ ·)).IsMultiplicative := by
  /-
    N : Nat
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    χ : DirichletCharacter R N
    ⊢ (toArithmeticFunction fun x => χ ↑x).IsMultiplicative
  -/
  refine IsMultiplicative.iff_ne_zero.mpr ⟨?_, fun {m} {n} hm hn _ ↦ ?_⟩
    /-
      case refine_1
      N : Nat
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      χ : DirichletCharacter R N
      ⊢ Eq ((toArithmeticFunction fun x => χ ↑x) 1) 1
    -/
  · simp only [toArithmeticFunction, coe_mk, one_ne_zero, ↓reduceIte, Nat.cast_one, map_one]
    /-
      🎉 no goals
    -/
  · simp only [toArithmeticFunction, coe_mk, mul_eq_zero, hm, hn, false_or, Nat.cast_mul, map_mul,
      if_false]


lemma apply_eq_toArithmeticFunction_apply {N : ℕ} {R : Type*} [CommMonoidWithZero R]
    (χ : DirichletCharacter R N) {n : ℕ} (hn : n ≠ 0) :
    χ n = toArithmeticFunction (χ ·) n := by
  /-
    N : Nat
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    χ : DirichletCharacter R N
    n : Nat
    hn : Ne n 0
    ⊢ Eq (χ ↑n) ((toArithmeticFunction fun x => χ ↑x) n)
  -/
  simp only [toArithmeticFunction, ArithmeticFunction.coe_mk, hn, ↓reduceIte]
  /-
    🎉 no goals
  -/


/-- Twisting by a Dirichlet character `χ` distributes over convolution. -/
lemma mul_convolution_distrib {R : Type*} [CommSemiring R] {n : ℕ} (χ : DirichletCharacter R n)
    (f g : ℕ → R) :
    (((χ ·) : ℕ → R) * f) ⍟ (((χ ·) : ℕ → R) * g) = ((χ ·) : ℕ → R) * (f ⍟ g) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    n : Nat
    χ : DirichletCharacter R n
    f g : Nat → R
    ⊢ Eq (LSeries.convolution (HMul.hMul (fun x => χ ↑x) f) (HMul.hMul (fun x => χ …
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    χ : DirichletCharacter R n✝
    f g : Nat → R
    n : Nat
    ⊢ Eq (LSeries.convolution (HMul.hMul (fun x => χ ↑x) f) (HMul.hMul (fun x => χ …
  -/
  simp only [Pi.mul_apply, LSeries.convolution_def, Finset.mul_sum]
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    χ : DirichletCharacter R n✝
    f g : Nat → R
    n : Nat
    ⊢ Eq (n.divisorsAntidiagonal.sum fun x => HMul.hMul (HMul.hMul (χ ↑x.1) (f x.1 …
  -/
  refine Finset.sum_congr rfl fun p hp ↦ ?_
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    χ : DirichletCharacter R n✝
    f g : Nat → R
    n : Nat
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    ⊢ Eq (HMul.hMul (HMul.hMul (χ ↑p.1) (f p.1)) (HMul.hMul (χ ↑p.2) (g p.2))) (HM …
  -/
  rw [(mem_divisorsAntidiagonal.mp hp).1.symm, cast_mul, map_mul]
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n✝ : Nat
    χ : DirichletCharacter R n✝
    f g : Nat → R
    n : Nat
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    ⊢ Eq (HMul.hMul (HMul.hMul (χ ↑p.1) (f p.1)) (HMul.hMul (χ ↑p.2) (g p.2))) (HM …
  -/
  exact mul_mul_mul_comm ..
  /-
    🎉 no goals
  -/


lemma mul_delta {n : ℕ} (χ : DirichletCharacter ℂ n) : ↗χ * δ = δ :=
                          /-
                            n : Nat
                            χ : DirichletCharacter Complex n
                            ⊢ Eq (χ ↑1) 1
                          -/
  LSeries.mul_delta <| by rw [cast_one, map_one]
                          /-
                            🎉 no goals
                          -/


lemma delta_mul {n : ℕ} (χ : DirichletCharacter ℂ n) : δ * ↗χ = δ :=
  mul_comm δ _ ▸ mul_delta ..


open ArithmeticFunction in
/-- The convolution of a Dirichlet character `χ` with the twist `χ * μ` is `δ`,
the indicator function of `{1}`. -/
lemma convolution_mul_moebius {n : ℕ} (χ : DirichletCharacter ℂ n) : ↗χ ⍟ (↗χ * ↗μ) = δ := by
  have : (1 : ℕ → ℂ) ⍟ (μ ·) = δ := by
    rw [one_convolution_eq_zeta_convolution, ← one_eq_delta]
    simp_rw [← natCoe_apply, ← intCoe_apply, coe_mul, coe_zeta_mul_coe_moebius]
  /-
    n : Nat
    χ : DirichletCharacter Complex n
    this : Eq (LSeries.convolution 1 fun x => ↑(ArithmeticFunction.moebius x)) LSe …
    ⊢ Eq (LSeries.convolution (fun n_1 => χ ↑n_1) (HMul.hMul (fun n_1 => χ ↑n_1) f …
  -/
  nth_rewrite 1 [← mul_one ↗χ]
  /-
    n : Nat
    χ : DirichletCharacter Complex n
    this : Eq (LSeries.convolution 1 fun x => ↑(ArithmeticFunction.moebius x)) LSe …
    ⊢ Eq (LSeries.convolution (HMul.hMul (fun n_1 => χ ↑n_1) 1) (HMul.hMul (fun n_ …
  -/
  simpa only [mul_convolution_distrib χ 1 ↗μ, this] using mul_delta _
  /-
    🎉 no goals
  -/


/-- The Dirichlet character mod `0` corresponds to `δ`. -/
lemma modZero_eq_delta {χ : DirichletCharacter ℂ 0} : ↗χ = δ := by
  /-
    χ : DirichletCharacter Complex 0
    ⊢ Eq (fun n => χ ↑n) LSeries.delta
  -/
  ext n
  /-
    case h
    χ : DirichletCharacter Complex 0
    n : Nat
    ⊢ Eq (χ ↑n) (LSeries.delta n)
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case h.inl
      χ : DirichletCharacter Complex 0
      ⊢ Eq (χ ↑0) (LSeries.delta 0)
    -/
  · simp_rw [cast_zero, χ.map_nonunit not_isUnit_zero, delta, reduceCtorEq, if_false]
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    χ : DirichletCharacter Complex 0
    n : Nat
    hn : Ne n 0
    ⊢ Eq (χ ↑n) (LSeries.delta n)
  -/
  rcases eq_or_ne n 1 with rfl | hn'
    /-
      case h.inr.inl
      χ : DirichletCharacter Complex 0
      hn : Ne 1 0
      ⊢ Eq (χ ↑1) (LSeries.delta 1)
    -/
  · simp only [cast_one, map_one, delta, ↓reduceIte]
    /-
      🎉 no goals
    -/
  /-
    case h.inr.inr
    χ : DirichletCharacter Complex 0
    n : Nat
    hn : Ne n 0
    hn' : Ne n 1
    ⊢ Eq (χ ↑n) (LSeries.delta n)
  -/
  have : ¬ IsUnit (n : ZMod 0) := fun h ↦ hn' <| ZMod.eq_one_of_isUnit_natCast h
  /-
    case h.inr.inr
    χ : DirichletCharacter Complex 0
    n : Nat
    hn : Ne n 0
    hn' : Ne n 1
    this : Not (IsUnit ↑n)
    ⊢ Eq (χ ↑n) (LSeries.delta n)
  -/
  simp only [χ.map_nonunit this, delta, hn', ↓reduceIte]
  /-
    🎉 no goals
  -/


/-- The Dirichlet character mod `1` corresponds to the constant function `1`. -/
lemma modOne_eq_one {R : Type*} [CommSemiring R] {χ : DirichletCharacter R 1} :
    ((χ ·) : ℕ → R) = 1 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    χ : DirichletCharacter R 1
    ⊢ Eq (fun x => χ ↑x) 1
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    χ : DirichletCharacter R 1
    x✝ : Nat
    ⊢ Eq (χ ↑x✝) (1 x✝)
  -/
  rw [χ.level_one, MulChar.one_apply (isUnit_of_subsingleton _), Pi.one_apply]
  /-
    🎉 no goals
  -/


lemma LSeries_modOne_eq : L ↗χ₁ = L 1 :=
  congr_arg L modOne_eq_one


/-- The L-series of a Dirichlet character mod `N > 0` does not converge absolutely at `s = 1`. -/
lemma not_LSeriesSummable_at_one {N : ℕ} (hN : N ≠ 0) (χ : DirichletCharacter ℂ N) :
    ¬ LSeriesSummable ↗χ 1 := by
  /-
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    ⊢ Not (LSeriesSummable (fun n => χ ↑n) 1)
  -/
  refine fun h ↦ (Real.not_summable_indicator_one_div_natCast hN 1) ?_
  refine h.norm.of_nonneg_of_le (fun m ↦ Set.indicator_apply_nonneg (fun _ ↦ by positivity))
    (fun n ↦ ?_)
  /-
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    h : LSeriesSummable (fun n => χ ↑n) 1
    n : Nat
    ⊢ LE.le ((setOf fun n => Eq (↑n) 1).indicator (fun n => HDiv.hDiv 1 ↑n) n) (No …
  -/
  rw [norm_term_eq, one_re, Real.rpow_one, Set.indicator]
  /-
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    h : LSeriesSummable (fun n => χ ↑n) 1
    n : Nat
    ⊢ LE.le (ite (Membership.mem (setOf fun n => Eq (↑n) 1) n) (HDiv.hDiv 1 ↑n) 0) …
  -/
  split_ifs with h₁ h₂
    /-
      case pos
      N : Nat
      hN : Ne N 0
      χ : DirichletCharacter Complex N
      h : LSeriesSummable (fun n => χ ↑n) 1
      n : Nat
      h₁ : Membership.mem (setOf fun n => Eq (↑n) 1) n
      h₂ : Eq n 0
      ⊢ LE.le (HDiv.hDiv 1 ↑n) 0
    -/
  · rw [h₂, cast_zero, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      N : Nat
      hN : Ne N 0
      χ : DirichletCharacter Complex N
      h : LSeriesSummable (fun n => χ ↑n) 1
      n : Nat
      h₁ : Membership.mem (setOf fun n => Eq (↑n) 1) n
      h₂ : Not (Eq n 0)
      ⊢ LE.le (HDiv.hDiv 1 ↑n) (HDiv.hDiv (Norm.norm (χ ↑n)) ↑n)
    -/
  · rw [h₁, χ.map_one, norm_one]
    /-
      🎉 no goals
    -/
  /-
    case pos
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    h : LSeriesSummable (fun n => χ ↑n) 1
    n : Nat
    h₁ : Not (Membership.mem (setOf fun n => Eq (↑n) 1) n)
    h✝ : Eq n 0
    ⊢ LE.le 0 0
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


/-- The L-series of a Dirichlet character converges absolutely at `s` if `re s > 1`. -/
lemma LSeriesSummable_of_one_lt_re {N : ℕ} (χ : DirichletCharacter ℂ N) {s : ℂ} (hs : 1 < s.re) :
    LSeriesSummable ↗χ s :=
  LSeriesSummable_of_bounded_of_one_lt_re (fun _ _ ↦ χ.norm_le_one _) hs


/-- The L-series of a Dirichlet character mod `N > 0` converges absolutely at `s` if and only if
`re s > 1`. -/
lemma LSeriesSummable_iff {N : ℕ} (hN : N ≠ 0) (χ : DirichletCharacter ℂ N) {s : ℂ} :
    LSeriesSummable ↗χ s ↔ 1 < s.re := by
  /-
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    s : Complex
    ⊢ Iff (LSeriesSummable (fun n => χ ↑n) s) (LT.lt 1 s.re)
  -/
  refine ⟨fun H ↦ ?_, LSeriesSummable_of_one_lt_re χ⟩
  /-
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    s : Complex
    H : LSeriesSummable (fun n => χ ↑n) s
    ⊢ LT.lt 1 s.re
  -/
  by_contra! h
  /-
    N : Nat
    hN : Ne N 0
    χ : DirichletCharacter Complex N
    s : Complex
    H : LSeriesSummable (fun n => χ ↑n) s
    h : LE.le s.re 1
    ⊢ False
  -/
  exact not_LSeriesSummable_at_one hN χ <| LSeriesSummable.of_re_le_re (by simp only [one_re, h]) H
  /-
    🎉 no goals
  -/


/-- The abscissa of absolute convergence of the L-series of a Dirichlet character mod `N > 0`
is `1`. -/
lemma absicssaOfAbsConv_eq_one {N : ℕ} (hn : N ≠ 0) (χ : DirichletCharacter ℂ N) :
    abscissaOfAbsConv ↗χ = 1 := by
  simpa only [abscissaOfAbsConv, LSeriesSummable_iff hn χ, ofReal_re, Set.Ioi_def,
    EReal.image_coe_Ioi, EReal.coe_one] using csInf_Ioo <| EReal.coe_lt_top _


/-- The L-series of the twist of `f` by a Dirichlet character converges at `s` if the L-series
of `f` does. -/
lemma LSeriesSummable_mul {N : ℕ} (χ : DirichletCharacter ℂ N) {f : ℕ → ℂ} {s : ℂ}
    (h : LSeriesSummable f s) :
    LSeriesSummable (↗χ * f) s := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    ⊢ LSeriesSummable (HMul.hMul (fun n => χ ↑n) f) s
  -/
  refine .of_norm <| h.norm.of_nonneg_of_le (fun _ ↦ norm_nonneg _) fun n ↦ norm_term_le s ?_
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    n : Nat
    ⊢ LE.le (Norm.norm (HMul.hMul (fun n => χ ↑n) f n)) (Norm.norm (f n))
  -/
  rw [Pi.mul_apply, norm_mul]
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    f : Nat → Complex
    s : Complex
    h : LSeriesSummable f s
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm (χ ↑n)) (Norm.norm (f n))) (Norm.norm (f n))
  -/
  exact mul_le_of_le_one_left (norm_nonneg _) <| norm_le_one ..
  /-
    🎉 no goals
  -/


open scoped ArithmeticFunction.Moebius in
/-- The L-series of a Dirichlet character `χ` and of the twist of `μ` by `χ` are multiplicative
inverses. -/
lemma LSeries.mul_mu_eq_one {N : ℕ} (χ : DirichletCharacter ℂ N) {s : ℂ}
    (hs : 1 < s.re) : L ↗χ s * L (↗χ * ↗μ) s = 1 := by
  rw [← LSeries_convolution' (LSeriesSummable_of_one_lt_re χ hs) <|
          LSeriesSummable_mul χ <| ArithmeticFunction.LSeriesSummable_moebius_iff.mpr hs,
    convolution_mul_moebius, LSeries_delta, Pi.one_apply]



/-- The L-series of a Dirichlet character does not vanish on the right half-plane `re s > 1`. -/
lemma LSeries_ne_zero_of_one_lt_re {N : ℕ} (χ : DirichletCharacter ℂ N) {s : ℂ} (hs : 1 < s.re) :
    L ↗χ s ≠ 0 :=
             /-
               N : Nat
               χ : DirichletCharacter Complex N
               s : Complex
               hs : LT.lt 1 s.re
               h : Eq (LSeries (fun n => χ ↑n) s) 0
               ⊢ False
             -/
  fun h ↦ by simpa only [h, zero_mul, zero_ne_one] using LSeries.mul_mu_eq_one χ hs
             /-
               🎉 no goals
             -/


/-- The abscissa of (absolute) convergence of the constant sequence `1` is `1`. -/
lemma LSeries.abscissaOfAbsConv_one : abscissaOfAbsConv 1 = 1 :=
  modOne_eq_one (χ := χ₁) ▸ absicssaOfAbsConv_eq_one one_ne_zero χ₁


/-- The `LSeries` of the constant sequence `1` converges at `s` if and only if `re s > 1`. -/
theorem LSeriesSummable_one_iff {s : ℂ} : LSeriesSummable 1 s ↔ 1 < s.re :=
  modOne_eq_one (χ := χ₁) ▸ LSeriesSummable_iff one_ne_zero χ₁



/-- The `LSeries` of the arithmetic function `ζ` is the same as the `LSeries` associated
to the constant sequence `1`. -/
lemma LSeries_zeta_eq : L ↗ζ = L 1 := by
  /-
    ⊢ Eq (LSeries fun n => ↑(ArithmeticFunction.zeta n)) (LSeries 1)
  -/
  ext s
  /-
    case h
    s : Complex
    ⊢ Eq (LSeries (fun n => ↑(ArithmeticFunction.zeta n)) s) (LSeries 1 s)
  -/
  exact (LSeries_congr s const_one_eq_zeta).symm
  /-
    🎉 no goals
  -/


/-- The `LSeries` associated to the arithmetic function `ζ` converges at `s` if and only if
`re s > 1`. -/
theorem LSeriesSummable_zeta_iff {s : ℂ} : LSeriesSummable (ζ ·) s ↔ 1 < s.re :=
  (LSeriesSummable_congr s const_one_eq_zeta).symm.trans <| LSeriesSummable_one_iff


@[deprecated (since := "2024-03-29")]
alias zeta_LSeriesSummable_iff_one_lt_re := LSeriesSummable_zeta_iff


/-- The abscissa of (absolute) convergence of the arithmetic function `ζ` is `1`. -/
lemma abscissaOfAbsConv_zeta : abscissaOfAbsConv ↗ζ = 1 := by
  /-
    ⊢ Eq (LSeries.abscissaOfAbsConv fun n => ↑(ArithmeticFunction.zeta n)) 1
  -/
  rw [abscissaOfAbsConv_congr (g := 1) fun hn ↦ by simp [hn], abscissaOfAbsConv_one]
  /-
    🎉 no goals
  -/


/-- The L-series of the arithmetic function `ζ` equals the Riemann Zeta Function on its
domain of convergence `1 < re s`. -/
lemma LSeries_zeta_eq_riemannZeta {s : ℂ} (hs : 1 < s.re) : L ↗ζ s = riemannZeta s := by
  simp only [LSeries, natCoe_apply, zeta_apply, cast_ite, cast_zero, cast_one,
    zeta_eq_tsum_one_div_nat_cpow hs]
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (tsum fun n => LSeries.term (fun n => ite (Eq n 0) 0 1) s n) (tsum fun n  …
  -/
  refine tsum_congr fun n ↦ ?_
  /-
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (LSeries.term (fun n => ite (Eq n 0) 0 1) s n) (HDiv.hDiv 1 (HPow.hPow (↑ …
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      s : Complex
      hs : LT.lt 1 s.re
      ⊢ Eq (LSeries.term (fun n => ite (Eq n 0) 0 1) s 0) (HDiv.hDiv 1 (HPow.hPow (↑ …
    -/
  · simp only [term_zero, cast_zero, zero_cpow (ne_zero_of_one_lt_re hs), div_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Complex
      hs : LT.lt 1 s.re
      n : Nat
      hn : Ne n 0
      ⊢ Eq (LSeries.term (fun n => ite (Eq n 0) 0 1) s n) (HDiv.hDiv 1 (HPow.hPow (↑ …
    -/
  · simp only [term_of_ne_zero hn, hn, ↓reduceIte, one_div]
    /-
      🎉 no goals
    -/


/-- The L-series of the arithmetic function `ζ` equals the Riemann Zeta Function on its
domain of convergence `1 < re s`. -/
lemma LSeriesHasSum_zeta {s : ℂ} (hs : 1 < s.re) : LSeriesHasSum ↗ζ s (riemannZeta s) :=
  LSeries_zeta_eq_riemannZeta hs ▸ (LSeriesSummable_zeta_iff.mpr hs).LSeriesHasSum


/-- The L-series of the arithmetic function `ζ` and of the Möbius function are inverses. -/
lemma LSeries_zeta_mul_Lseries_moebius {s : ℂ} (hs : 1 < s.re) : L ↗ζ s * L ↗μ s = 1 := by
  rw [← LSeries_convolution' (LSeriesSummable_zeta_iff.mpr hs)
    (LSeriesSummable_moebius_iff.mpr hs)]
  simp only [← natCoe_apply, ← intCoe_apply, coe_mul, coe_zeta_mul_coe_moebius, one_eq_delta,
    LSeries_delta, Pi.one_apply]


/-- The L-series of the arithmetic function `ζ` does not vanish on the right half-plane
`re s > 1`. -/
lemma LSeries_zeta_ne_zero_of_one_lt_re {s : ℂ} (hs : 1 < s.re) : L ↗ζ s ≠ 0 :=
             /-
               s : Complex
               hs : LT.lt 1 s.re
               h : Eq (LSeries (fun n => ↑(ArithmeticFunction.zeta n)) s) 0
               ⊢ False
             -/
  fun h ↦ by simpa only [h, zero_mul, zero_ne_one] using LSeries_zeta_mul_Lseries_moebius hs
             /-
               🎉 no goals
             -/


/-- The L-series of the constant sequence `1` equals the Riemann Zeta Function on its
domain of convergence `1 < re s`. -/
lemma LSeries_one_eq_riemannZeta {s : ℂ} (hs : 1 < s.re) : L 1 s = riemannZeta s :=
  LSeries_zeta_eq ▸ LSeries_zeta_eq_riemannZeta hs


/-- The L-series of the constant sequence `1` equals the Riemann zeta function on its
domain of convergence `1 < re s`. -/
lemma LSeriesHasSum_one {s : ℂ} (hs : 1 < s.re) : LSeriesHasSum 1 s (riemannZeta s) :=
  LSeries_one_eq_riemannZeta hs ▸ (LSeriesSummable_one_iff.mpr hs).LSeriesHasSum


/-- The L-series of the constant sequence `1` and of the Möbius function are inverses. -/
lemma LSeries_one_mul_Lseries_moebius {s : ℂ} (hs : 1 < s.re) : L 1 s * L ↗μ s = 1 :=
  LSeries_zeta_eq ▸ LSeries_zeta_mul_Lseries_moebius hs


/-- The L-series of the constant sequence `1` does not vanish on the right half-plane
`re s > 1`. -/
lemma LSeries_one_ne_zero_of_one_lt_re {s : ℂ} (hs : 1 < s.re) : L 1 s ≠ 0 :=
  LSeries_zeta_eq ▸ LSeries_zeta_ne_zero_of_one_lt_re hs


/-- The Riemann Zeta Function does not vanish on the half-plane `re s > 1`. -/
lemma riemannZeta_ne_zero_of_one_lt_re {s : ℂ} (hs : 1 < s.re) : riemannZeta s ≠ 0 :=
  LSeries_one_eq_riemannZeta hs ▸ LSeries_one_ne_zero_of_one_lt_re hs


/-- A translation of the relation `Λ * ↑ζ = log` of (real-valued) arithmetic functions
to an equality of complex sequences. -/
lemma convolution_vonMangoldt_zeta : ↗Λ ⍟ ↗ζ = ↗Complex.log := by
  /-
    ⊢ Eq (LSeries.convolution (fun n => ↑(ArithmeticFunction.vonMangoldt n)) fun n …
  -/
  ext n
  simpa only [zeta_apply, apply_ite, cast_zero, cast_one, LSeries.convolution_def, mul_zero,
    mul_one, mul_apply, natCoe_apply, ofReal_sum, ofReal_zero, log_apply, ofReal_log n.cast_nonneg]
    using congr_arg (ofReal <| · n) vonMangoldt_mul_zeta


lemma convolution_vonMangoldt_const_one : ↗Λ ⍟ 1 = ↗Complex.log :=
  (convolution_one_eq_convolution_zeta _).trans convolution_vonMangoldt_zeta


/-- The L-series of the von Mangoldt function `Λ` converges at `s` when `re s > 1`. -/
lemma LSeriesSummable_vonMangoldt {s : ℂ} (hs : 1 < s.re) : LSeriesSummable ↗Λ s := by
  have hf := LSeriesSummable_logMul_of_lt_re
    (show abscissaOfAbsConv 1 < s.re by rw [abscissaOfAbsConv_one]; exact_mod_cast hs)
  /-
    s : Complex
    hs : LT.lt 1 s.re
    hf : LSeriesSummable (LSeries.logMul 1) s
    ⊢ LSeriesSummable (fun n => ↑(ArithmeticFunction.vonMangoldt n)) s
  -/
  rw [LSeriesSummable, ← summable_norm_iff] at hf ⊢
  /-
    s : Complex
    hs : LT.lt 1 s.re
    hf : Summable fun x => Norm.norm (LSeries.term (LSeries.logMul 1) s x)
    ⊢ Summable fun x => Norm.norm (LSeries.term (fun n => ↑(ArithmeticFunction.von …
  -/
  refine Summable.of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun n ↦ norm_term_le s ?_) hf
  have hΛ : ‖↗Λ n‖ ≤ ‖Complex.log n‖ := by
    simp only [norm_eq_abs, abs_ofReal, _root_.abs_of_nonneg vonMangoldt_nonneg,
      ← Complex.natCast_log, _root_.abs_of_nonneg <| Real.log_natCast_nonneg n]
    exact ArithmeticFunction.vonMangoldt_le_log
  /-
    s : Complex
    hs : LT.lt 1 s.re
    hf : Summable fun x => Norm.norm (LSeries.term (LSeries.logMul 1) s x)
    n : Nat
    hΛ : LE.le (Norm.norm ((fun n => ↑(ArithmeticFunction.vonMangoldt n)) n)) (Nor …
    ⊢ LE.le (Norm.norm ↑(ArithmeticFunction.vonMangoldt n)) (Norm.norm (LSeries.lo …
  -/
  exact hΛ.trans <| by simp only [norm_eq_abs, norm_mul, Pi.one_apply, norm_one, mul_one, le_refl]
  /-
    🎉 no goals
  -/


/-- A twisted version of the relation `Λ * ↑ζ = log` in terms of complex sequences. -/
lemma convolution_twist_vonMangoldt {N : ℕ} (χ : DirichletCharacter ℂ N) :
    (↗χ * ↗Λ) ⍟ ↗χ = ↗χ * ↗Complex.log := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    ⊢ Eq (LSeries.convolution (HMul.hMul (fun n => χ ↑n) fun n => ↑(ArithmeticFunc …
  -/
  rw [← convolution_vonMangoldt_const_one, ← χ.mul_convolution_distrib, mul_one]
  /-
    🎉 no goals
  -/


/-- The L-series of the twist of the von Mangoldt function `Λ` by a Dirichlet character `χ`
converges at `s` when `re s > 1`. -/
lemma LSeriesSummable_twist_vonMangoldt {N : ℕ} (χ : DirichletCharacter ℂ N) {s : ℂ}
    (hs : 1 < s.re) :
    LSeriesSummable (↗χ * ↗Λ) s :=
  LSeriesSummable_mul χ <| LSeriesSummable_vonMangoldt hs


/-- The L-series of the twist of the von Mangoldt function `Λ` by a Dirichlet character `χ` at `s`
equals the negative logarithmic derivative of the L-series of `χ` when `re s > 1`. -/
lemma LSeries_twist_vonMangoldt_eq {N : ℕ} (χ : DirichletCharacter ℂ N) {s : ℂ} (hs : 1 < s.re) :
    L (↗χ * ↗Λ) s = - deriv (L ↗χ) s / L ↗χ s := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (LSeries (HMul.hMul (fun n => χ ↑n) fun n => ↑(ArithmeticFunction.vonMang …
  -/
  rcases eq_or_ne N 0 with rfl | hN
  · simpa only [modZero_eq_delta, delta_mul_eq_smul_delta, vonMangoldt_apply_one, ofReal_zero,
      zero_smul, LSeries_zero, Pi.zero_apply, LSeries_delta, Pi.one_apply, div_one, zero_eq_neg]
      using deriv_const s 1
  -- now `N ≠ 0`
  /-
    case inr
    N : Nat
    χ : DirichletCharacter Complex N
    s : Complex
    hs : LT.lt 1 s.re
    hN : Ne N 0
    ⊢ Eq (LSeries (HMul.hMul (fun n => χ ↑n) fun n => ↑(ArithmeticFunction.vonMang …
  -/
  have hχ : LSeriesSummable ↗χ s := (LSeriesSummable_iff hN χ).mpr hs
  have hs' : abscissaOfAbsConv ↗χ < s.re := by
    rwa [absicssaOfAbsConv_eq_one hN, ← EReal.coe_one, EReal.coe_lt_coe_iff]
  /-
    case inr
    N : Nat
    χ : DirichletCharacter Complex N
    s : Complex
    hs : LT.lt 1 s.re
    hN : Ne N 0
    hχ : LSeriesSummable (fun n => χ ↑n) s
    hs' : LT.lt (LSeries.abscissaOfAbsConv fun n => χ ↑n) ↑s.re
    ⊢ Eq (LSeries (HMul.hMul (fun n => χ ↑n) fun n => ↑(ArithmeticFunction.vonMang …
  -/
  have hΛ : LSeriesSummable (↗χ * ↗Λ) s := LSeriesSummable_twist_vonMangoldt χ hs
  rw [eq_div_iff <| LSeries_ne_zero_of_one_lt_re χ hs, ← LSeries_convolution' hΛ hχ,
    convolution_twist_vonMangoldt, LSeries_deriv hs', neg_neg]
  /-
    case inr
    N : Nat
    χ : DirichletCharacter Complex N
    s : Complex
    hs : LT.lt 1 s.re
    hN : Ne N 0
    hχ : LSeriesSummable (fun n => χ ↑n) s
    hs' : LT.lt (LSeries.abscissaOfAbsConv fun n => χ ↑n) ↑s.re
    hΛ : LSeriesSummable (HMul.hMul (fun n => χ ↑n) fun n => ↑(ArithmeticFunction. …
    ⊢ Eq (LSeries (HMul.hMul (fun n => χ ↑n) fun n => Complex.log ↑n) s) (LSeries  …
  -/
  exact LSeries_congr s fun _ ↦ by simp only [Pi.mul_apply, mul_comm, logMul]
  /-
    🎉 no goals
  -/


open DirichletCharacter in
/-- The L-series of the von Mangoldt function `Λ` equals the negative logarithmic derivative
of the L-series of the constant sequence `1` on its domain of convergence `re s > 1`. -/
lemma LSeries_vonMangoldt_eq {s : ℂ} (hs : 1 < s.re) : L ↗Λ s = - deriv (L 1) s / L 1 s := by
  refine (LSeries_congr s fun {n} _ ↦ ?_).trans <|
    LSeries_modOne_eq ▸ LSeries_twist_vonMangoldt_eq χ₁ hs
  /-
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    x✝ : Ne n 0
    ⊢ Eq (↑(ArithmeticFunction.vonMangoldt n)) (HMul.hMul (fun n => 1 ↑n) (fun n = …
  -/
  simp only [Subsingleton.eq_one (n : ZMod 1), map_one, Pi.mul_apply, one_mul]
  /-
    🎉 no goals
  -/


/-- The L-series of the von Mangoldt function `Λ` equals the negative logarithmic derivative
of the Riemann zeta function on its domain of convergence `re s > 1`. -/
lemma LSeries_vonMangoldt_eq_deriv_riemannZeta_div {s : ℂ} (hs : 1 < s.re) :
    L ↗Λ s = - deriv riemannZeta s / riemannZeta s := by
  suffices deriv (L 1) s = deriv riemannZeta s by
    rw [LSeries_vonMangoldt_eq hs, ← LSeries_one_eq_riemannZeta hs, this]
  /-
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (deriv (LSeries 1) s) (deriv riemannZeta s)
  -/
  refine Filter.EventuallyEq.deriv_eq <| Filter.eventuallyEq_iff_exists_mem.mpr ?_
  exact ⟨{z | 1 < z.re}, (isOpen_lt continuous_const continuous_re).mem_nhds hs,
    fun _ ↦ LSeries_one_eq_riemannZeta⟩


