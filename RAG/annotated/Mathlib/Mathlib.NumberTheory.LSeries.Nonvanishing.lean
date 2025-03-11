/-- The complex-valued arithmetic function that is the convolution of the constant
function `1` with `χ`. -/
def zetaMul (χ : DirichletCharacter ℂ N) : ArithmeticFunction ℂ :=
  .zeta * toArithmeticFunction (χ ·)


/-- The arithmetic function `zetaMul χ` is multiplicative. -/
lemma isMultiplicative_zetaMul (χ : DirichletCharacter ℂ N) : χ.zetaMul.IsMultiplicative :=
  isMultiplicative_zeta.natCast.mul <| isMultiplicative_toArithmeticFunction χ


lemma LSeriesSummable_zetaMul (χ : DirichletCharacter ℂ N) {s : ℂ} (hs : 1 < s.re) :
    LSeriesSummable χ.zetaMul s := by
  refine ArithmeticFunction.LSeriesSummable_mul (LSeriesSummable_zeta_iff.mpr hs) <|
    LSeriesSummable_of_bounded_of_one_lt_re (m := 1) (fun n hn ↦ ?_) hs
  simpa only [toArithmeticFunction, coe_mk, hn, ↓reduceIte, ← Complex.norm_eq_abs]
  using norm_le_one χ _


lemma zetaMul_prime_pow_nonneg {χ : DirichletCharacter ℂ N} (hχ : χ ^ 2 = 1) {p : ℕ}
    (hp : p.Prime) (k : ℕ) :
    0 ≤ zetaMul χ (p ^ k) := by
  simp only [zetaMul, toArithmeticFunction, coe_zeta_mul_apply, coe_mk,
    Nat.sum_divisors_prime_pow hp, pow_eq_zero_iff', hp.ne_zero, ne_eq, false_and, ↓reduceIte,
    Nat.cast_pow, map_pow]
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    hχ : Eq (HPow.hPow χ 2) 1
    p : Nat
    hp : Nat.Prime p
    k : Nat
    ⊢ LE.le 0 ((Finset.range (HAdd.hAdd k 1)).sum fun x => HPow.hPow (χ ↑p) x)
  -/
  rcases MulChar.isQuadratic_iff_sq_eq_one.mpr hχ p with h | h | h
    /-
      case inl
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) 0
      ⊢ LE.le 0 ((Finset.range (HAdd.hAdd k 1)).sum fun x => HPow.hPow (χ ↑p) x)
    -/
  · refine Finset.sum_nonneg fun i _ ↦ ?_
    /-
      case inl
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) 0
      i : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) i
      ⊢ LE.le 0 (HPow.hPow (χ ↑p) i)
    -/
    simp only [h, le_refl, pow_nonneg]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) 1
      ⊢ LE.le 0 ((Finset.range (HAdd.hAdd k 1)).sum fun x => HPow.hPow (χ ↑p) x)
    -/
  · refine Finset.sum_nonneg fun i _ ↦ ?_
    /-
      case inr.inl
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) 1
      i : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) i
      ⊢ LE.le 0 (HPow.hPow (χ ↑p) i)
    -/
    simp only [h, one_pow, zero_le_one]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) (-1)
      ⊢ LE.le 0 ((Finset.range (HAdd.hAdd k 1)).sum fun x => HPow.hPow (χ ↑p) x)
    -/
  · simp only [h, neg_one_geom_sum]
    /-
      case inr.inr
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) (-1)
      ⊢ LE.le 0 (ite (_root_.Even (HAdd.hAdd k 1)) 0 1)
    -/
    split_ifs
    /-
      case pos
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      p : Nat
      hp : Nat.Prime p
      k : Nat
      h : Eq (χ ↑p) (-1)
      h✝ : _root_.Even (HAdd.hAdd k 1)
      ⊢ LE.le 0 0
    -/
    exacts [le_rfl, zero_le_one]
    /-
      🎉 no goals
    -/


/-- `zetaMul χ` takes nonnegative real values when `χ` is a quadratic character. -/
lemma zetaMul_nonneg {χ : DirichletCharacter ℂ N} (hχ : χ ^ 2 = 1) (n : ℕ) :
    0 ≤ zetaMul χ n := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    hχ : Eq (HPow.hPow χ 2) 1
    n : Nat
    ⊢ LE.le 0 (χ.zetaMul n)
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      N : Nat
      χ : DirichletCharacter Complex N
      hχ : Eq (HPow.hPow χ 2) 1
      ⊢ LE.le 0 (χ.zetaMul 0)
    -/
  · simp only [ArithmeticFunction.map_zero, le_refl]
    /-
      🎉 no goals
    -/
  · simpa only [χ.isMultiplicative_zetaMul.multiplicative_factorization _ hn] using
      Finset.prod_nonneg
        fun p hp ↦ zetaMul_prime_pow_nonneg hχ (Nat.prime_of_mem_primeFactors hp) _

/-
### "Bad" Dirichlet characters

Our goal is to show that `L(χ, 1) ≠ 0` when `χ` is a (nontrivial) quadratic Dirichlet character.
To do that, we package the contradictory properties in a (private) structure
`DirichletCharacter.BadChar` and derive further statements eventually leading to a contradiction.

This entire section is private.
-/


/-- The object we're trying to show doesn't exist: A nontrivial quadratic Dirichlet character
whose L-function vanishes at `s = 1`. -/
private structure BadChar (N : ℕ) [NeZero N] where
  /-- The character we want to show cannot exist. -/
  χ : DirichletCharacter ℂ N
  χ_ne : χ ≠ 1
  χ_sq : χ ^ 2 = 1
  hχ : χ.LFunction 1 = 0


/-- The product of the Riemann zeta function with the L-function of `B.χ`.
We will show that `B.F (-2) = 0` but also that `B.F (-2)` must be positive,
giving the desired contradiction. -/
private noncomputable
def F (B : BadChar N) : ℂ → ℂ :=
  Function.update (fun s : ℂ ↦ riemannZeta s * LFunction B.χ s) 1 (deriv (LFunction B.χ) 1)


private lemma F_differentiableAt_of_ne (B : BadChar N) {s : ℂ} (hs : s ≠ 1) :
    DifferentiableAt ℂ B.F s := by
  /-
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    s : Complex
    hs : Ne s 1
    ⊢ DifferentiableAt Complex B.F s
  -/
  apply DifferentiableAt.congr_of_eventuallyEq
    /-
      case h
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : Ne s 1
      ⊢ DifferentiableAt Complex ?f s
    -/
  · exact (differentiableAt_riemannZeta hs).mul <| differentiableAt_LFunction B.χ s (.inl hs)
    /-
      🎉 no goals
    -/
    /-
      case hL
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : Ne s 1
      ⊢ (nhds s).EventuallyEq B.F fun y => HMul.hMul (riemannZeta y) (DirichletChara …
    -/
  · filter_upwards [eventually_ne_nhds hs] with t ht using Function.update_of_ne ht ..
    /-
      🎉 no goals
    -/


/-- `B.F` agrees with the L-series of `zetaMul χ` on `1 < s.re`. -/
private lemma F_eq_LSeries (B : BadChar N) {s : ℂ} (hs : 1 < s.re) :
    B.F s = LSeries B.χ.zetaMul s := by
  /-
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (B.F s) (LSeries (⇑B.χ.zetaMul) s)
  -/
  rw [F, zetaMul, ← coe_mul, LSeries_convolution']
    /-
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : LT.lt 1 s.re
      ⊢ Eq (Function.update (fun s => HMul.hMul (riemannZeta s) (DirichletCharacter. …
    -/
  · have hs' : s ≠ 1 := fun h ↦ by simp only [h, one_re, lt_self_iff_false] at hs
    /-
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : LT.lt 1 s.re
      hs' : Ne s 1
      ⊢ Eq (Function.update (fun s => HMul.hMul (riemannZeta s) (DirichletCharacter. …
    -/
    simp only [ne_eq, hs', not_false_eq_true, Function.update_of_ne, B.χ.LFunction_eq_LSeries hs]
    /-
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : LT.lt 1 s.re
      hs' : Ne s 1
      ⊢ Eq (HMul.hMul (riemannZeta s) (LSeries (fun x => B.χ ↑x) s)) (HMul.hMul (LSe …
    -/
    congr 1
      /-
        case e_a
        N : Nat
        inst✝ : NeZero N
        B : DirichletCharacter.BadChar N
        s : Complex
        hs : LT.lt 1 s.re
        hs' : Ne s 1
        ⊢ Eq (riemannZeta s) (LSeries (⇑↑ArithmeticFunction.zeta) s)
      -/
    · simp_rw [← LSeries_zeta_eq_riemannZeta hs, ← natCoe_apply]
      /-
        🎉 no goals
      -/
      /-
        case e_a
        N : Nat
        inst✝ : NeZero N
        B : DirichletCharacter.BadChar N
        s : Complex
        hs : LT.lt 1 s.re
        hs' : Ne s 1
        ⊢ Eq (LSeries (fun x => B.χ ↑x) s) (LSeries (⇑(toArithmeticFunction fun x => B …
      -/
    · exact LSeries_congr s B.χ.apply_eq_toArithmeticFunction_apply
      /-
        🎉 no goals
      -/
  -- summability side goals from `LSeries_convolution'`
    /-
      case hf
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : LT.lt 1 s.re
      ⊢ LSeriesSummable (⇑↑ArithmeticFunction.zeta) s
    -/
  · exact LSeriesSummable_zeta_iff.mpr hs
    /-
      🎉 no goals
    -/
  · exact (LSeriesSummable_congr _ fun h ↦ (B.χ.apply_eq_toArithmeticFunction_apply h).symm).mpr <|
      ZMod.LSeriesSummable_of_one_lt_re B.χ hs


/-- If `χ` is a bad character, then `F` is an entire function. -/
private lemma F_differentiable (B : BadChar N) : Differentiable ℂ B.F := by
  /-
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    ⊢ Differentiable Complex B.F
  -/
  intro s
  /-
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    s : Complex
    ⊢ DifferentiableAt Complex B.F s
  -/
  rcases ne_or_eq s 1 with hs | rfl
    /-
      case inl
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      s : Complex
      hs : Ne s 1
      ⊢ DifferentiableAt Complex B.F s
    -/
  · exact B.F_differentiableAt_of_ne hs
    /-
      🎉 no goals
    -/
  -- now need to deal with `s = 1`
  /-
    case inr
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    ⊢ DifferentiableAt Complex B.F 1
  -/
  refine (analyticAt_of_differentiable_on_punctured_nhds_of_continuousAt ?_ ?_).differentiableAt
    /-
      case inr.refine_1
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      ⊢ Filter.Eventually (fun z => DifferentiableAt Complex B.F z) (nhdsWithin 1 (H …
    -/
  · filter_upwards [self_mem_nhdsWithin] with t ht
    /-
      case h
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      t : Complex
      ht : Membership.mem (HasCompl.compl (Singleton.singleton 1)) t
      ⊢ DifferentiableAt Complex B.F t
    -/
    exact B.F_differentiableAt_of_ne ht
    /-
      🎉 no goals
    -/
  -- now reduced to showing *continuity* at s = 1
  /-
    case inr.refine_2
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    ⊢ ContinuousAt B.F 1
  -/
  let G := Function.update (fun s ↦ (s - 1) * riemannZeta s) 1 1
  let H := Function.update (fun s ↦ (B.χ.LFunction s - B.χ.LFunction 1) / (s - 1)) 1
    (deriv B.χ.LFunction 1)
  have : B.F = G * H := by
    ext1 t
    rcases eq_or_ne t 1 with rfl | ht
    · simp only [F, G, H, Pi.mul_apply, one_mul, Function.update_self]
    · simp only [F, G, H, Function.update_of_ne ht, mul_comm _ (riemannZeta _), B.hχ, sub_zero,
      Pi.mul_apply, mul_assoc, mul_div_cancel₀ _ (sub_ne_zero.mpr ht)]
  /-
    case inr.refine_2
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    G : Complex → Complex := Function.update (fun s => HMul.hMul (HSub.hSub s 1) ( …
    H : Complex → Complex := Function.update (fun s => HDiv.hDiv (HSub.hSub (Diric …
    this : Eq B.F (HMul.hMul G H)
    ⊢ ContinuousAt B.F 1
  -/
  rw [this]
  /-
    case inr.refine_2
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    G : Complex → Complex := Function.update (fun s => HMul.hMul (HSub.hSub s 1) ( …
    H : Complex → Complex := Function.update (fun s => HDiv.hDiv (HSub.hSub (Diric …
    this : Eq B.F (HMul.hMul G H)
    ⊢ ContinuousAt (HMul.hMul G H) 1
  -/
  apply ContinuousAt.mul
    /-
      case inr.refine_2.hf
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      G : Complex → Complex := Function.update (fun s => HMul.hMul (HSub.hSub s 1) ( …
      H : Complex → Complex := Function.update (fun s => HDiv.hDiv (HSub.hSub (Diric …
      this : Eq B.F (HMul.hMul G H)
      ⊢ ContinuousAt G 1
    -/
  · simpa only [G, continuousAt_update_same] using riemannZeta_residue_one
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2.hg
      N : Nat
      inst✝ : NeZero N
      B : DirichletCharacter.BadChar N
      G : Complex → Complex := Function.update (fun s => HMul.hMul (HSub.hSub s 1) ( …
      H : Complex → Complex := Function.update (fun s => HDiv.hDiv (HSub.hSub (Diric …
      this : Eq B.F (HMul.hMul G H)
      ⊢ ContinuousAt H 1
    -/
  · exact (B.χ.differentiableAt_LFunction 1 (.inr B.χ_ne)).hasDerivAt.continuousAt_div
    /-
      🎉 no goals
    -/


/-- The trivial zero at `s = -2` of the zeta function gives that `F (-2) = 0`.
This is used later to obtain a contradction. -/
private lemma F_neg_two (B : BadChar N) : B.F (-2 : ℝ) = 0 := by
  /-
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    ⊢ Eq (B.F ↑(-2)) 0
  -/
  have := riemannZeta_neg_two_mul_nat_add_one 0
  /-
    N : Nat
    inst✝ : NeZero N
    B : DirichletCharacter.BadChar N
    this : Eq (riemannZeta (HMul.hMul (-2) (HAdd.hAdd (↑0) 1))) 0
    ⊢ Eq (B.F ↑(-2)) 0
  -/
  rw [Nat.cast_zero, zero_add, mul_one] at this
  rw [F, ofReal_neg, ofReal_ofNat, Function.update_of_ne (mod_cast (by omega : (-2 : ℤ) ≠ 1)),
    this, zero_mul]


/-- If `χ` is a nontrivial quadratic Dirichlet character, then `L(χ, 1) ≠ 0`. This is private
since it is later superseded by `LFunction_apply_one_ne_zero`. -/
private theorem LFunction_apply_one_ne_zero_of_quadratic {χ : DirichletCharacter ℂ N}
    (hχ : χ ^ 2 = 1) (χ_ne : χ ≠ 1) :
    χ.LFunction 1 ≠ 0 := by
  /-
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    hχ : Eq (HPow.hPow χ 2) 1
    χ_ne : Ne χ 1
    ⊢ Ne (DirichletCharacter.LFunction χ 1) 0
  -/
  intro hL
  -- construct a "bad character" and put together a contradiction.
  /-
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    hχ : Eq (HPow.hPow χ 2) 1
    χ_ne : Ne χ 1
    hL : Eq (DirichletCharacter.LFunction χ 1) 0
    ⊢ False
  -/
  let B : BadChar N := {χ := χ, χ_sq := hχ, hχ := hL, χ_ne := χ_ne}
  /-
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    hχ : Eq (HPow.hPow χ 2) 1
    χ_ne : Ne χ 1
    hL : Eq (DirichletCharacter.LFunction χ 1) 0
    B : DirichletCharacter.BadChar N := { χ := χ, χ_ne := χ_ne, χ_sq := hχ, hχ :=  …
    ⊢ False
  -/
  refine B.F_neg_two.not_gt ?_
  refine ArithmeticFunction.LSeries_positive_of_differentiable_of_eqOn (zetaMul_nonneg hχ)
    (χ.isMultiplicative_zetaMul.map_one ▸ zero_lt_one) B.F_differentiable ?_
    (fun _ ↦ B.F_eq_LSeries) _
  exact LSeries.abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable
    fun _ a ↦ χ.LSeriesSummable_zetaMul a


private lemma re_log_comb_nonneg' {a : ℝ} (ha₀ : 0 ≤ a) (ha₁ : a < 1) {z : ℂ} (hz : ‖z‖ = 1) :
      0 ≤ 3 * (-log (1 - a)).re + 4 * (-log (1 - a * z)).re + (-log (1 - a * z ^ 2)).re := by
  have hac₀ : ‖(a : ℂ)‖ < 1 := by
    simp only [norm_eq_abs, abs_ofReal, _root_.abs_of_nonneg ha₀, ha₁]
  /-
    a : Real
    ha₀ : LE.le 0 a
    ha₁ : LT.lt a 1
    z : Complex
    hz : Eq (Norm.norm z) 1
    hac₀ : LT.lt (Norm.norm ↑a) 1
    ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub  …
  -/
  have hac₁ : ‖a * z‖ < 1 := by rwa [norm_mul, hz, mul_one]
  /-
    a : Real
    ha₀ : LE.le 0 a
    ha₁ : LT.lt a 1
    z : Complex
    hz : Eq (Norm.norm z) 1
    hac₀ : LT.lt (Norm.norm ↑a) 1
    hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
    ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub  …
  -/
  have hac₂ : ‖a * z ^ 2‖ < 1 := by rwa [norm_mul, norm_pow, hz, one_pow, mul_one]
  rw [← ((hasSum_re <| hasSum_taylorSeries_neg_log hac₀).mul_left 3).add
    ((hasSum_re <| hasSum_taylorSeries_neg_log hac₁).mul_left 4) |>.add
    (hasSum_re <| hasSum_taylorSeries_neg_log hac₂) |>.tsum_eq]
  /-
    a : Real
    ha₀ : LE.le 0 a
    ha₁ : LT.lt a 1
    z : Complex
    hz : Eq (Norm.norm z) 1
    hac₀ : LT.lt (Norm.norm ↑a) 1
    hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
    hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
    ⊢ LE.le 0 (tsum fun b => HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HDiv.hDiv (HPow.hP …
  -/
  refine tsum_nonneg fun n ↦ ?_
  simp only [← ofReal_pow, div_natCast_re, ofReal_re, mul_pow, mul_re, ofReal_im, zero_mul,
    sub_zero]
  /-
    a : Real
    ha₀ : LE.le 0 a
    ha₁ : LT.lt a 1
    z : Complex
    hz : Eq (Norm.norm z) 1
    hac₀ : LT.lt (Norm.norm ↑a) 1
    hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
    hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
    n : Nat
    ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HDiv.hDiv (HPow.hPow a n) ↑n)) ( …
  -/
  rcases n.eq_zero_or_pos with rfl | hn
    /-
      case inl
      a : Real
      ha₀ : LE.le 0 a
      ha₁ : LT.lt a 1
      z : Complex
      hz : Eq (Norm.norm z) 1
      hac₀ : LT.lt (Norm.norm ↑a) 1
      hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
      hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HDiv.hDiv (HPow.hPow a 0) ↑0)) ( …
    -/
  · simp only [pow_zero, Nat.cast_zero, div_zero, mul_zero, one_re, mul_one, add_zero, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Real
      ha₀ : LE.le 0 a
      ha₁ : LT.lt a 1
      z : Complex
      hz : Eq (Norm.norm z) 1
      hac₀ : LT.lt (Norm.norm ↑a) 1
      hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
      hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HDiv.hDiv (HPow.hPow a n) ↑n)) ( …
    -/
  · simp only [← mul_div_assoc, ← add_div]
    /-
      case inr
      a : Real
      ha₀ : LE.le 0 a
      ha₁ : LT.lt a 1
      z : Complex
      hz : Eq (Norm.norm z) 1
      hac₀ : LT.lt (Norm.norm ↑a) 1
      hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
      hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le 0 (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow a n)) (HMul …
    -/
    refine div_nonneg ?_ n.cast_nonneg
    rw [← pow_mul, pow_mul', sq, mul_re, ← sq, ← sq, ← sq_abs_sub_sq_re, ← norm_eq_abs, norm_pow,
      hz]
    /-
      case inr
      a : Real
      ha₀ : LE.le 0 a
      ha₁ : LT.lt a 1
      z : Complex
      hz : Eq (Norm.norm z) 1
      hac₀ : LT.lt (Norm.norm ↑a) 1
      hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
      hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow a n)) (HMul.hMul 4 (HM …
    -/
    convert (show 0 ≤ 2 * a ^ n * ((z ^ n).re + 1) ^ 2 by positivity) using 1
    /-
      case h.e'_4
      a : Real
      ha₀ : LE.le 0 a
      ha₁ : LT.lt a 1
      z : Complex
      hz : Eq (Norm.norm z) 1
      hac₀ : LT.lt (Norm.norm ↑a) 1
      hac₁ : LT.lt (Norm.norm (HMul.hMul (↑a) z)) 1
      hac₂ : LT.lt (Norm.norm (HMul.hMul (↑a) (HPow.hPow z 2))) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow a n)) (HMul.hMul 4 (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/

-- This is the version of the technical positivity lemma for logarithms of Euler factors.

private lemma re_log_comb_nonneg {n : ℕ} (hn : 2 ≤ n) {x : ℝ} (hx : 1 < x) (y : ℝ) :
    0 ≤ 3 * (-log (1 - (1 : DirichletCharacter ℂ N) n * n ^ (-x : ℂ))).re +
          4 * (-log (1 - χ n * n ^ (-(x + I * y)))).re +
          (-log (1 - (χ n ^ 2) * n ^ (-(x + 2 * I * y)))).re := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    n : Nat
    hn : LE.le 2 n
    x : Real
    hx : LT.lt 1 x
    y : Real
    ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub  …
  -/
  by_cases hn' : IsUnit (n : ZMod N)
    /-
      case pos
      N : Nat
      χ : DirichletCharacter Complex N
      n : Nat
      hn : LE.le 2 n
      x : Real
      hx : LT.lt 1 x
      y : Real
      hn' : IsUnit ↑n
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub  …
    -/
  · have ha₀ : 0 ≤ (n : ℝ) ^ (-x) := Real.rpow_nonneg n.cast_nonneg _
    have ha₁ : (n : ℝ) ^ (-x) < 1 := by
      rw [Real.rpow_neg (Nat.cast_nonneg n), inv_lt_one_iff₀]
      exact .inr <| Real.one_lt_rpow (mod_cast one_lt_two.trans_le hn) <| zero_lt_one.trans hx
    have hz : ‖χ n * (n : ℂ) ^ (-(I * y))‖ = 1 := by
      rw [norm_mul, ← hn'.unit_spec, DirichletCharacter.unit_norm_eq_one χ hn'.unit,
        norm_eq_abs, ← ofReal_natCast, abs_cpow_eq_rpow_re_of_pos (mod_cast by omega)]
      simp only [neg_re, mul_re, I_re, ofReal_re, zero_mul, I_im, ofReal_im, mul_zero, sub_self,
        neg_zero, Real.rpow_zero, one_mul]
    /-
      case pos
      N : Nat
      χ : DirichletCharacter Complex N
      n : Nat
      hn : LE.le 2 n
      x : Real
      hx : LT.lt 1 x
      y : Real
      hn' : IsUnit ↑n
      ha₀ : LE.le 0 (HPow.hPow (↑n) (Neg.neg x))
      ha₁ : LT.lt (HPow.hPow (↑n) (Neg.neg x)) 1
      hz : Eq (Norm.norm (HMul.hMul (χ ↑n) (HPow.hPow (↑n) (Neg.neg (HMul.hMul Compl …
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub  …
    -/
    rw [MulChar.one_apply hn', one_mul]
    /-
      case pos
      N : Nat
      χ : DirichletCharacter Complex N
      n : Nat
      hn : LE.le 2 n
      x : Real
      hx : LT.lt 1 x
      y : Real
      hn' : IsUnit ↑n
      ha₀ : LE.le 0 (HPow.hPow (↑n) (Neg.neg x))
      ha₁ : LT.lt (HPow.hPow (↑n) (Neg.neg x)) 1
      hz : Eq (Norm.norm (HMul.hMul (χ ↑n) (HPow.hPow (↑n) (Neg.neg (HMul.hMul Compl …
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub  …
    -/
    convert re_log_comb_nonneg' ha₀ ha₁ hz using 6
      /-
        case h.e'_4.h.e'_5.h.e'_5.h.e'_6.h.e'_1.h.e'_3
        N : Nat
        χ : DirichletCharacter Complex N
        n : Nat
        hn : LE.le 2 n
        x : Real
        hx : LT.lt 1 x
        y : Real
        hn' : IsUnit ↑n
        ha₀ : LE.le 0 (HPow.hPow (↑n) (Neg.neg x))
        ha₁ : LT.lt (HPow.hPow (↑n) (Neg.neg x)) 1
        hz : Eq (Norm.norm (HMul.hMul (χ ↑n) (HPow.hPow (↑n) (Neg.neg (HMul.hMul Compl …
        ⊢ Eq (Complex.log (HSub.hSub 1 (HPow.hPow (↑n) (Neg.neg ↑x)))) (Complex.log (H …
      -/
    · simp only [ofReal_cpow n.cast_nonneg (-x), ofReal_natCast, ofReal_neg]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4.h.e'_5.h.e'_6.h.e'_6.h.e'_1.h.e'_3
        N : Nat
        χ : DirichletCharacter Complex N
        n : Nat
        hn : LE.le 2 n
        x : Real
        hx : LT.lt 1 x
        y : Real
        hn' : IsUnit ↑n
        ha₀ : LE.le 0 (HPow.hPow (↑n) (Neg.neg x))
        ha₁ : LT.lt (HPow.hPow (↑n) (Neg.neg x)) 1
        hz : Eq (Norm.norm (HMul.hMul (χ ↑n) (HPow.hPow (↑n) (Neg.neg (HMul.hMul Compl …
        ⊢ Eq (Complex.log (HSub.hSub 1 (HMul.hMul (χ ↑n) (HPow.hPow (↑n) (Neg.neg (HAd …
      -/
    · congr 2
      rw [neg_add, cpow_add _ _ <| mod_cast by omega, ← ofReal_neg, ofReal_cpow n.cast_nonneg (-x),
        ofReal_natCast, mul_left_comm]
    · rw [neg_add, cpow_add _ _ <| mod_cast by omega, ← ofReal_neg, ofReal_cpow n.cast_nonneg (-x),
        ofReal_natCast, show -(2 * I * y) = (2 : ℕ) * -(I * y) by ring, cpow_nat_mul, mul_pow,
        mul_left_comm]
  · simp only [MulChar.map_nonunit _ hn', zero_mul, sub_zero, log_one, neg_zero, zero_re, mul_zero,
      neg_add_rev, add_zero, pow_two, le_refl]


/-- The logarithms of the Euler factors of a Dirichlet L-series form a summable sequence. -/
lemma summable_neg_log_one_sub_mul_prime_cpow {s : ℂ} (hs : 1 < s.re) :
    Summable fun p : Nat.Primes ↦ -log (1 - χ p * (p : ℂ) ^ (-s)) := by
  have (p : Nat.Primes) : ‖χ p * (p : ℂ) ^ (-s)‖ ≤ (p : ℝ) ^ (-s).re := by
    simpa only [norm_mul, norm_natCast_cpow_of_re_ne_zero _ <| re_neg_ne_zero_of_one_lt_re hs]
      using mul_le_of_le_one_left (by positivity) (χ.norm_le_one _)
  refine (Nat.Primes.summable_rpow.mpr ?_).of_nonneg_of_le (fun _ ↦ norm_nonneg _) this
    |>.of_norm.clog_one_sub.neg
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    s : Complex
    hs : LT.lt 1 s.re
    this : ∀ (p : Nat.Primes), LE.le (Norm.norm (HMul.hMul (χ ↑↑p) (HPow.hPow (↑↑p …
    ⊢ LT.lt (Neg.neg s).re (-1)
  -/
  simp only [neg_re, neg_lt_neg_iff, hs]
  /-
    🎉 no goals
  -/


private lemma one_lt_re_one_add {x : ℝ} (hx : 0 < x) (y : ℝ) :
    1 < (1 + x : ℂ).re ∧ 1 < (1 + x + I * y).re ∧ 1 < (1 + x + 2 * I * y).re := by
  simp only [add_re, one_re, ofReal_re, lt_add_iff_pos_right, hx, mul_re, I_re, zero_mul, I_im,
    ofReal_im, mul_zero, sub_self, add_zero, re_ofNat, im_ofNat, mul_one, mul_im, and_self]


open scoped LSeries.notation in
/-- For positive `x` and nonzero `y` and a Dirichlet character `χ` we have that
`|L(χ^0, 1 + x)^3 L(χ, 1 + x + I * y)^4 L(χ^2, 1 + x + 2 * I * y)| ≥ 1. -/
lemma norm_LSeries_product_ge_one {x : ℝ} (hx : 0 < x) (y : ℝ) :
    ‖L ↗(1 : DirichletCharacter ℂ N) (1 + x) ^ 3 * L ↗χ (1 + x + I * y) ^ 4 *
      L ↗(χ ^ 2 :) (1 + x + 2 * I * y)‖ ≥ 1 := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have ⟨h₀, h₁, h₂⟩ := one_lt_re_one_add hx y
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have H₀ := summable_neg_log_one_sub_mul_prime_cpow (N := N) 1 h₀
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    H₀ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (1 ↑↑p) (H …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have H₁ := summable_neg_log_one_sub_mul_prime_cpow χ h₁
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    H₀ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (1 ↑↑p) (H …
    H₁ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (χ ↑↑p) (H …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have H₂ := summable_neg_log_one_sub_mul_prime_cpow (χ ^ 2) h₂
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    H₀ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (1 ↑↑p) (H …
    H₁ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (χ ↑↑p) (H …
    H₂ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul ((HPow.hPo …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have hsum₀ := (hasSum_re H₀.hasSum).summable.mul_left 3
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    H₀ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (1 ↑↑p) (H …
    H₁ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (χ ↑↑p) (H …
    H₂ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul ((HPow.hPo …
    hsum₀ : Summable fun i => HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub 1 (HMul …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have hsum₁ := (hasSum_re H₁.hasSum).summable.mul_left 4
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    H₀ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (1 ↑↑p) (H …
    H₁ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul (χ ↑↑p) (H …
    H₂ : Summable fun p => Neg.neg (Complex.log (HSub.hSub 1 (HMul.hMul ((HPow.hPo …
    hsum₀ : Summable fun i => HMul.hMul 3 (Neg.neg (Complex.log (HSub.hSub 1 (HMul …
    hsum₁ : Summable fun i => HMul.hMul 4 (Neg.neg (Complex.log (HSub.hSub 1 (HMul …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun n => 1 ↑n) ( …
  -/
  have hsum₂ := (hasSum_re H₂.hasSum).summable
  rw [← LSeries_eulerProduct_exp_log _ h₀, ← LSeries_eulerProduct_exp_log χ h₁,
    ← LSeries_eulerProduct_exp_log _ h₂]
  simp only [← exp_nat_mul, Nat.cast_ofNat, ← exp_add, norm_eq_abs, abs_exp, add_re, mul_re,
    re_ofNat, im_ofNat, zero_mul, sub_zero, Real.one_le_exp_iff]
  rw [re_tsum H₀, re_tsum H₁, re_tsum H₂, ← tsum_mul_left, ← tsum_mul_left,
    ← tsum_add hsum₀ hsum₁, ← tsum_add (hsum₀.add hsum₁) hsum₂]
  simpa only [neg_add_rev, neg_re, mul_neg, χ.pow_apply' two_ne_zero, ge_iff_le, add_re, one_re,
    ofReal_re, ofReal_add, ofReal_one] using
      tsum_nonneg fun (p : Nat.Primes) ↦ χ.re_log_comb_nonneg p.prop.two_le h₀ y


/-- A variant of `DirichletCharacter.norm_LSeries_product_ge_one` in terms of the L-functions. -/
lemma norm_LFunction_product_ge_one {x : ℝ} (hx : 0 < x) (y : ℝ) :
    ‖LFunctionTrivChar N (1 + x) ^ 3 * LFunction χ (1 + x + I * y) ^ 4 *
      LFunction (χ ^ 2) (1 + x + 2 * I * y)‖ ≥ 1 := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    x : Real
    hx : LT.lt 0 x
    y : Real
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (DirichletCharacter.LFunct …
  -/
  have ⟨h₀, h₁, h₂⟩ := one_lt_re_one_add hx y
  rw [LFunctionTrivChar, DirichletCharacter.LFunction_eq_LSeries 1 h₀,
    χ.LFunction_eq_LSeries h₁, (χ ^ 2).LFunction_eq_LSeries h₂]
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    x : Real
    hx : LT.lt 0 x
    y : Real
    h₀ : LT.lt 1 (HAdd.hAdd 1 ↑x).re
    h₁ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul Complex.I ↑y)).re
    h₂ : LT.lt 1 (HAdd.hAdd (HAdd.hAdd 1 ↑x) (HMul.hMul (HMul.hMul 2 Complex.I) ↑y …
    ⊢ GE.ge (Norm.norm (HMul.hMul (HMul.hMul (HPow.hPow (LSeries (fun x => 1 ↑x) ( …
  -/
  exact norm_LSeries_product_ge_one χ hx y
  /-
    🎉 no goals
  -/


lemma LFunctionTrivChar_isBigO_near_one_horizontal :
    (fun x : ℝ ↦ LFunctionTrivChar N (1 + x)) =O[𝓝[>] 0] fun x ↦ (1 : ℂ) / x := by
  have : (fun w : ℂ ↦ LFunctionTrivChar N (1 + w)) =O[𝓝[≠] 0] (1 / ·) := by
    have H : Tendsto (fun w ↦ w * LFunctionTrivChar N (1 + w)) (𝓝[≠] 0)
        (𝓝 <| ∏ p ∈ N.primeFactors, (1 - (p : ℂ)⁻¹)) := by
      convert (LFunctionTrivChar_residue_one (N := N)).comp (f := fun w ↦ 1 + w) ?_ using 1
      · simp only [Function.comp_def, add_sub_cancel_left]
      · simpa only [tendsto_iff_comap, Homeomorph.coe_addLeft, add_zero, map_le_iff_le_comap] using
          ((Homeomorph.addLeft (1 : ℂ)).map_punctured_nhds_eq 0).le
    exact (isBigO_mul_iff_isBigO_div eventually_mem_nhdsWithin).mp <| H.isBigO_one ℂ
  /-
    N : Nat
    inst✝ : NeZero N
    this : Asymptotics.IsBigO (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0 …
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => DirichletCharacter.L …
  -/
  exact (isBigO_comp_ofReal_nhds_ne this).mono <| nhdsGT_le_nhdsNE 0
  /-
    🎉 no goals
  -/


omit [NeZero N] in
private lemma one_add_I_mul_ne_one_or {y : ℝ} (hy : y ≠ 0 ∨ χ ≠ 1) :
    1 + I * y ≠ 1 ∨ χ ≠ 1:= by
  simpa only [ne_eq, add_right_eq_self, _root_.mul_eq_zero, I_ne_zero, ofReal_eq_zero, false_or]
    using hy


lemma LFunction_isBigO_horizontal {y : ℝ} (hy : y ≠ 0 ∨ χ ≠ 1) :
    (fun x : ℝ ↦ LFunction χ (1 + x + I * y)) =O[𝓝[>] 0] fun _ ↦ (1 : ℂ) := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => DirichletCharacter.L …
  -/
  refine IsBigO.mono ?_ nhdsWithin_le_nhds
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => DirichletCharacter.LFunction χ (HAdd.h …
  -/
  simp_rw [add_comm (1 : ℂ), add_assoc]
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => DirichletCharacter.LFunction χ (HAdd.h …
  -/
  have := (χ.differentiableAt_LFunction _ <| one_add_I_mul_ne_one_or χ hy).continuousAt
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    this : ContinuousAt (DirichletCharacter.LFunction χ) (HAdd.hAdd 1 (HMul.hMul C …
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => DirichletCharacter.LFunction χ (HAdd.h …
  -/
  rw [← zero_add (1 + _)] at this
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    this : ContinuousAt (DirichletCharacter.LFunction χ) (HAdd.hAdd 0 (HAdd.hAdd 1 …
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x => DirichletCharacter.LFunction χ (HAdd.h …
  -/
  exact this.comp (f := fun x : ℝ ↦ x + (1 + I * y)) (x := 0) (by fun_prop) |>.tendsto.isBigO_one ℂ
  /-
    🎉 no goals
  -/


private lemma LFunction_isBigO_horizontal_of_eq_zero {y : ℝ} (hy : y ≠ 0 ∨ χ ≠ 1)
    (h : LFunction χ (1 + I * y) = 0) :
    (fun x : ℝ ↦ LFunction χ (1 + x + I * y)) =O[𝓝[>] 0] fun x : ℝ ↦ (x : ℂ) := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    h : Eq (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑y))) 0
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => DirichletCharacter.L …
  -/
  simp_rw [add_comm (1 : ℂ), add_assoc]
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    h : Eq (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑y))) 0
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => DirichletCharacter.L …
  -/
  have := (χ.differentiableAt_LFunction _ <| one_add_I_mul_ne_one_or χ hy).hasDerivAt
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    y : Real
    hy : Or (Ne y 0) (Ne χ 1)
    h : Eq (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑y))) 0
    this : HasDerivAt (DirichletCharacter.LFunction χ) (deriv (DirichletCharacter. …
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => DirichletCharacter.L …
  -/
  rw [← zero_add (1 + _)] at this
  simpa only [zero_add, h, sub_zero]
    using (Complex.isBigO_comp_ofReal_nhds
      (this.comp_add_const 0 _).differentiableAt.isBigO_sub) |>.mono nhdsWithin_le_nhds

-- intermediate statement, special case of the next theorem

private lemma LFunction_ne_zero_of_not_quadratic_or_ne_one {t : ℝ} (h : χ ^ 2 ≠ 1 ∨ t ≠ 0) :
    LFunction χ (1 + I * t) ≠ 0 := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    t : Real
    h : Or (Ne (HPow.hPow χ 2) 1) (Ne t 0)
    ⊢ Ne (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑t))) 0
  -/
  intro Hz
  have hz₁ : t ≠ 0 ∨ χ ≠ 1 := by
    refine h.symm.imp_right (fun h H ↦ ?_)
    simp only [H, one_pow, ne_eq, not_true_eq_false] at h
  have hz₂ : 2 * t ≠ 0 ∨ χ ^ 2 ≠ 1 :=
    h.symm.imp_left <| mul_ne_zero two_ne_zero
  have help (x : ℝ) : ((1 / x) ^ 3 * x ^ 4 * 1 : ℂ) = x := by
    rcases eq_or_ne x 0 with rfl | h
    · rw [ofReal_zero, zero_pow (by omega), mul_zero, mul_one]
    · rw [one_div, inv_pow, pow_succ _ 3, ← mul_assoc,
        inv_mul_cancel₀ <| pow_ne_zero 3 (ofReal_ne_zero.mpr h), one_mul, mul_one]
  -- put together the various `IsBigO` statements and `norm_LFunction_product_ge_one`
  -- to derive a contradiction
  have H₀ : (fun _ : ℝ ↦ (1 : ℝ)) =O[𝓝[>] 0]
      fun x ↦ LFunctionTrivChar N (1 + x) ^ 3 * LFunction χ (1 + x + I * t) ^ 4 *
                   LFunction (χ ^ 2) (1 + x + 2 * I * t) :=
    IsBigO.of_bound' <| eventually_nhdsWithin_of_forall
      fun _ hx ↦ (norm_one (α := ℝ)).symm ▸ (χ.norm_LFunction_product_ge_one hx t).le
  have H := (LFunctionTrivChar_isBigO_near_one_horizontal (N := N)).pow 3 |>.mul <|
    (χ.LFunction_isBigO_horizontal_of_eq_zero hz₁ Hz).pow 4 |>.mul <|
    LFunction_isBigO_horizontal _ hz₂
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    t : Real
    h : Or (Ne (HPow.hPow χ 2) 1) (Ne t 0)
    Hz : Eq (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑t)) …
    hz₁ : Or (Ne t 0) (Ne χ 1)
    hz₂ : Or (Ne (HMul.hMul 2 t) 0) (Ne (HPow.hPow χ 2) 1)
    help : ∀ (x : Real), Eq (HMul.hMul (HMul.hMul (HPow.hPow (HDiv.hDiv 1 ↑x) 3) ( …
    H₀ : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => 1) fun x => HMul. …
    H : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HMul.hMul (HPow.hP …
    ⊢ False
  -/
  simp only [ofReal_mul, ofReal_ofNat, mul_left_comm I, ← mul_assoc, help] at H
  -- go via absolute value to translate into a statement over `ℝ`
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    t : Real
    h : Or (Ne (HPow.hPow χ 2) 1) (Ne t 0)
    Hz : Eq (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑t)) …
    hz₁ : Or (Ne t 0) (Ne χ 1)
    hz₂ : Or (Ne (HMul.hMul 2 t) 0) (Ne (HPow.hPow χ 2) 1)
    help : ∀ (x : Real), Eq (HMul.hMul (HMul.hMul (HPow.hPow (HDiv.hDiv 1 ↑x) 3) ( …
    H₀ : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => 1) fun x => HMul. …
    H : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HMul.hMul (HMul.hM …
    ⊢ False
  -/
  replace H := (H₀.trans H).norm_right
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    t : Real
    h : Or (Ne (HPow.hPow χ 2) 1) (Ne t 0)
    Hz : Eq (DirichletCharacter.LFunction χ (HAdd.hAdd 1 (HMul.hMul Complex.I ↑t)) …
    hz₁ : Or (Ne t 0) (Ne χ 1)
    hz₂ : Or (Ne (HMul.hMul 2 t) 0) (Ne (HPow.hPow χ 2) 1)
    help : ∀ (x : Real), Eq (HMul.hMul (HMul.hMul (HPow.hPow (HDiv.hDiv 1 ↑x) 3) ( …
    H₀ : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => 1) fun x => HMul. …
    H : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => 1) fun x => Norm.n …
    ⊢ False
  -/
  simp only [norm_eq_abs, abs_ofReal] at H
  #adaptation_note
  /--
  After https://github.com/leanprover/lean4/pull/6024
  we needed to add `(F' := ℝ)` to `H.of_norm_right`.
  -/
  exact isLittleO_irrefl (.of_forall (fun _ ↦ one_ne_zero)) <|
    (H.of_norm_right (F' := ℝ)).trans_isLittleO <| isLittleO_id_one.mono nhdsWithin_le_nhds


/-- If `χ` is a Dirichlet character, then `L(χ, s)` does not vanish when `s.re = 1`
except when `χ` is trivial and `s = 1` (then `L(χ, s)` has a simple pole at `s = 1`). -/
theorem LFunction_ne_zero_of_re_eq_one {s : ℂ} (hs : s.re = 1) (hχs : χ ≠ 1 ∨ s ≠ 1) :
    LFunction χ s ≠ 0 := by
  /-
    N : Nat
    χ : DirichletCharacter Complex N
    inst✝ : NeZero N
    s : Complex
    hs : Eq s.re 1
    hχs : Or (Ne χ 1) (Ne s 1)
    ⊢ Ne (DirichletCharacter.LFunction χ s) 0
  -/
  by_cases h : χ ^ 2 = 1 ∧ s = 1
    /-
      case pos
      N : Nat
      χ : DirichletCharacter Complex N
      inst✝ : NeZero N
      s : Complex
      hs : Eq s.re 1
      hχs : Or (Ne χ 1) (Ne s 1)
      h : And (Eq (HPow.hPow χ 2) 1) (Eq s 1)
      ⊢ Ne (DirichletCharacter.LFunction χ s) 0
    -/
  · exact h.2 ▸ LFunction_apply_one_ne_zero_of_quadratic h.1 <| hχs.neg_resolve_right h.2
    /-
      🎉 no goals
    -/
  · have hs' : s = 1 + I * s.im := by
      conv_lhs => rw [← re_add_im s, hs, ofReal_one, mul_comm]
    /-
      case neg
      N : Nat
      χ : DirichletCharacter Complex N
      inst✝ : NeZero N
      s : Complex
      hs : Eq s.re 1
      hχs : Or (Ne χ 1) (Ne s 1)
      h : Not (And (Eq (HPow.hPow χ 2) 1) (Eq s 1))
      hs' : Eq s (HAdd.hAdd 1 (HMul.hMul Complex.I ↑s.im))
      ⊢ Ne (DirichletCharacter.LFunction χ s) 0
    -/
    rw [not_and_or, ← ne_eq, ← ne_eq, hs', add_right_ne_self] at h
    replace h : χ ^ 2 ≠ 1 ∨ s.im ≠ 0 :=
      h.imp_right (fun H ↦ by exact_mod_cast right_ne_zero_of_mul H)
    /-
      case neg
      N : Nat
      χ : DirichletCharacter Complex N
      inst✝ : NeZero N
      s : Complex
      hs : Eq s.re 1
      hχs : Or (Ne χ 1) (Ne s 1)
      hs' : Eq s (HAdd.hAdd 1 (HMul.hMul Complex.I ↑s.im))
      h : Or (Ne (HPow.hPow χ 2) 1) (Ne s.im 0)
      ⊢ Ne (DirichletCharacter.LFunction χ s) 0
    -/
    exact hs'.symm ▸ χ.LFunction_ne_zero_of_not_quadratic_or_ne_one h
    /-
      🎉 no goals
    -/


/-- If `χ` is a Dirichlet character, then `L(χ, s)` does not vanish for `s.re ≥ 1`
except when `χ` is trivial and `s = 1` (then `L(χ, s)` has a simple pole at `s = 1`). -/
theorem LFunction_ne_zero_of_one_le_re ⦃s : ℂ⦄ (hχs : χ ≠ 1 ∨ s ≠ 1) (hs : 1 ≤ s.re) :
    LFunction χ s ≠ 0 :=
  hs.eq_or_lt.casesOn (fun hs ↦ LFunction_ne_zero_of_re_eq_one χ hs.symm hχs)
    fun hs ↦ LFunction_eq_LSeries χ hs ▸ LSeries_ne_zero_of_one_lt_re χ hs

-- Interesting special case:

variable {χ} in
/-- The L-function of a nontrivial Dirichlet character does not vanish at `s = 1`. -/
theorem LFunction_apply_one_ne_zero (hχ : χ ≠ 1) : LFunction χ 1 ≠ 0 :=
  LFunction_ne_zero_of_one_le_re χ (.inl hχ) <| one_re ▸ le_rfl


/-- The Riemann Zeta Function does not vanish on the closed half-plane `re s ≥ 1`.
(Note that the value at `s = 1` is a junk value, which happens to be nonzero.) -/
lemma _root_.riemannZeta_ne_zero_of_one_le_re ⦃s : ℂ⦄ (hs : 1 ≤ s.re) :
    riemannZeta s ≠ 0 := by
  /-
    s : Complex
    hs : LE.le 1 s.re
    ⊢ Ne (riemannZeta s) 0
  -/
  rcases eq_or_ne s 1 with rfl | hs₀
    /-
      case inl
      hs : LE.le 1 (Complex.re 1)
      ⊢ Ne (riemannZeta 1) 0
    -/
  · exact riemannZeta_one_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Complex
      hs : LE.le 1 s.re
      hs₀ : Ne s 1
      ⊢ Ne (riemannZeta s) 0
    -/
  · exact LFunction_modOne_eq (χ := 1) ▸ LFunction_ne_zero_of_one_le_re _ (.inr hs₀) hs
    /-
      🎉 no goals
    -/


