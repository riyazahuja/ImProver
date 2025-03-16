/-- If all values of a `ℂ`-valued arithmetic function are nonnegative reals and `x` is a
real number in the domain of absolute convergence, then the `n`th iterated derivative
of the associated L-series is nonnegative real when `n` is even and nonpositive real
when `n` is odd. -/
lemma iteratedDeriv_alternating {a : ℕ → ℂ} (hn : 0 ≤ a) {x : ℝ}
    (h : LSeries.abscissaOfAbsConv a < x) (n : ℕ) :
    0 ≤ (-1) ^ n * iteratedDeriv n (LSeries a) x := by
  rw [LSeries_iteratedDeriv _ h, LSeries, ← mul_assoc, ← pow_add, Even.neg_one_pow ⟨n, rfl⟩,
    one_mul]
  /-
    a : Nat → Complex
    hn : LE.le 0 a
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
    n : Nat
    ⊢ LE.le 0 (tsum fun n_1 => LSeries.term (Nat.iterate LSeries.logMul n a) (↑x)  …
  -/
  refine tsum_nonneg fun k ↦ ?_
  /-
    a : Nat → Complex
    hn : LE.le 0 a
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
    n k : Nat
    ⊢ LE.le 0 (LSeries.term (Nat.iterate LSeries.logMul n a) (↑x) k)
  -/
  rw [LSeries.term_def]
  /-
    a : Nat → Complex
    hn : LE.le 0 a
    x : Real
    h : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
    n k : Nat
    ⊢ LE.le 0 (ite (Eq k 0) 0 (HDiv.hDiv (Nat.iterate LSeries.logMul n a k) (HPow. …
  -/
  split
    /-
      case isTrue
      a : Nat → Complex
      hn : LE.le 0 a
      x : Real
      h : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
      n k : Nat
      h✝ : Eq k 0
      ⊢ LE.le 0 0
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case isFalse
      a : Nat → Complex
      hn : LE.le 0 a
      x : Real
      h : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
      n k : Nat
      h✝ : Not (Eq k 0)
      ⊢ LE.le 0 (HDiv.hDiv (Nat.iterate LSeries.logMul n a k) (HPow.hPow ↑k ↑x))
    -/
  · refine mul_nonneg ?_ <| (inv_natCast_cpow_ofReal_pos (by assumption) x).le
    induction n with
    | zero => simpa only [Function.iterate_zero, id_eq] using hn k
    | succ n IH =>
        rw [Function.iterate_succ_apply']
        refine mul_nonneg ?_ IH
        simp only [← natCast_log, zero_le_real, Real.log_natCast_nonneg]


/-- If all values of `a : ℕ → ℂ` are nonnegative reals and `a 1` is positive,
then `L a x` is positive real for all real `x` larger than `abscissaOfAbsConv a`. -/
lemma positive {a : ℕ → ℂ} (ha₀ : 0 ≤ a) (ha₁ : 0 < a 1) {x : ℝ} (hx : abscissaOfAbsConv a < x) :
    0 < LSeries a x := by
  /-
    a : Nat → Complex
    ha₀ : LE.le 0 a
    ha₁ : LT.lt 0 (a 1)
    x : Real
    hx : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
    ⊢ LT.lt 0 (LSeries a ↑x)
  -/
  rw [LSeries]
  /-
    a : Nat → Complex
    ha₀ : LE.le 0 a
    ha₁ : LT.lt 0 (a 1)
    x : Real
    hx : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
    ⊢ LT.lt 0 (tsum fun n => LSeries.term a (↑x) n)
  -/
  refine tsum_pos ?_ (fun n ↦ term_nonneg (ha₀ n) x) 1 <| term_pos one_ne_zero ha₁ x
  /-
    a : Nat → Complex
    ha₀ : LE.le 0 a
    ha₁ : LT.lt 0 (a 1)
    x : Real
    hx : LT.lt (LSeries.abscissaOfAbsConv a) ↑x
    ⊢ Summable (LSeries.term a ↑x)
  -/
  exact LSeriesSummable_of_abscissaOfAbsConv_lt_re <| by simpa only [ofReal_re] using hx
  /-
    🎉 no goals
  -/


/-- If all values of `a : ℕ → ℂ` are nonnegative reals and `a 1`
is positive, and the L-series of `a` agrees with an entire function `f` on some open
right half-plane where it converges, then `f` is real and positive on `ℝ`. -/
lemma positive_of_differentiable_of_eqOn {a : ℕ → ℂ} (ha₀ : 0 ≤ a) (ha₁ : 0 < a 1) {f : ℂ → ℂ}
    (hf : Differentiable ℂ f) {x : ℝ} (hx : abscissaOfAbsConv a ≤ x)
    (hf' : {s | x < s.re}.EqOn f (LSeries a)) (y : ℝ) :
    0 < f y := by
  /-
    a : Nat → Complex
    ha₀ : LE.le 0 a
    ha₁ : LT.lt 0 (a 1)
    f : Complex → Complex
    hf : Differentiable Complex f
    x : Real
    hx : LE.le (LSeries.abscissaOfAbsConv a) ↑x
    hf' : Set.EqOn f (LSeries a) (setOf fun s => LT.lt x s.re)
    y : Real
    ⊢ LT.lt 0 (f ↑y)
  -/
  have hxy : x < max x y + 1 := (le_max_left x y).trans_lt (lt_add_one _)
  /-
    a : Nat → Complex
    ha₀ : LE.le 0 a
    ha₁ : LT.lt 0 (a 1)
    f : Complex → Complex
    hf : Differentiable Complex f
    x : Real
    hx : LE.le (LSeries.abscissaOfAbsConv a) ↑x
    hf' : Set.EqOn f (LSeries a) (setOf fun s => LT.lt x s.re)
    y : Real
    hxy : LT.lt x (HAdd.hAdd (Max.max x y) 1)
    ⊢ LT.lt 0 (f ↑y)
  -/
  have hxy' : abscissaOfAbsConv a < max x y + 1 := hx.trans_lt <| mod_cast hxy
  have hys : (max x y + 1 : ℂ) ∈ {s | x < s.re} := by
    simp only [Set.mem_setOf_eq, add_re, ofReal_re, one_re, hxy]
  have hfx : 0 < f (max x y + 1) := by
    simpa only [hf' hys, ofReal_add, ofReal_one] using positive ha₀ ha₁ hxy'
  /-
    a : Nat → Complex
    ha₀ : LE.le 0 a
    ha₁ : LT.lt 0 (a 1)
    f : Complex → Complex
    hf : Differentiable Complex f
    x : Real
    hx : LE.le (LSeries.abscissaOfAbsConv a) ↑x
    hf' : Set.EqOn f (LSeries a) (setOf fun s => LT.lt x s.re)
    y : Real
    hxy : LT.lt x (HAdd.hAdd (Max.max x y) 1)
    hxy' : LT.lt (LSeries.abscissaOfAbsConv a) (HAdd.hAdd (↑(Max.max x y)) 1)
    hys : Membership.mem (setOf fun s => LT.lt x s.re) (HAdd.hAdd (↑(Max.max x y)) …
    hfx : LT.lt 0 (f (HAdd.hAdd (↑(Max.max x y)) 1))
    ⊢ LT.lt 0 (f ↑y)
  -/
  refine (hfx.trans_le <| hf.apply_le_of_iteratedDeriv_alternating (fun n _ ↦ ?_) ?_)
    /-
      case refine_1
      a : Nat → Complex
      ha₀ : LE.le 0 a
      ha₁ : LT.lt 0 (a 1)
      f : Complex → Complex
      hf : Differentiable Complex f
      x : Real
      hx : LE.le (LSeries.abscissaOfAbsConv a) ↑x
      hf' : Set.EqOn f (LSeries a) (setOf fun s => LT.lt x s.re)
      y : Real
      hxy : LT.lt x (HAdd.hAdd (Max.max x y) 1)
      hxy' : LT.lt (LSeries.abscissaOfAbsConv a) (HAdd.hAdd (↑(Max.max x y)) 1)
      hys : Membership.mem (setOf fun s => LT.lt x s.re) (HAdd.hAdd (↑(Max.max x y)) …
      hfx : LT.lt 0 (f (HAdd.hAdd (↑(Max.max x y)) 1))
      n : Nat
      x✝ : Ne n 0
      ⊢ LE.le 0 (HMul.hMul (HPow.hPow (-1) n) (iteratedDeriv n f (HAdd.hAdd (↑(Max.m …
    -/
  · have hs : IsOpen {s : ℂ | x < s.re} := continuous_re.isOpen_preimage _ isOpen_Ioi
    simpa only [hf'.iteratedDeriv_of_isOpen hs n hys, ofReal_add, ofReal_one] using
      iteratedDeriv_alternating ha₀ hxy' n
    /-
      case refine_2
      a : Nat → Complex
      ha₀ : LE.le 0 a
      ha₁ : LT.lt 0 (a 1)
      f : Complex → Complex
      hf : Differentiable Complex f
      x : Real
      hx : LE.le (LSeries.abscissaOfAbsConv a) ↑x
      hf' : Set.EqOn f (LSeries a) (setOf fun s => LT.lt x s.re)
      y : Real
      hxy : LT.lt x (HAdd.hAdd (Max.max x y) 1)
      hxy' : LT.lt (LSeries.abscissaOfAbsConv a) (HAdd.hAdd (↑(Max.max x y)) 1)
      hys : Membership.mem (setOf fun s => LT.lt x s.re) (HAdd.hAdd (↑(Max.max x y)) …
      hfx : LT.lt 0 (f (HAdd.hAdd (↑(Max.max x y)) 1))
      ⊢ LE.le (↑y) (HAdd.hAdd (↑(Max.max x y)) 1)
    -/
  · exact_mod_cast (le_max_right x y).trans (lt_add_one _).le
    /-
      🎉 no goals
    -/


/-- If all values of a `ℂ`-valued arithmetic function are nonnegative reals and `x` is a
real number in the domain of absolute convergence, then the `n`th iterated derivative
of the associated L-series is nonnegative real when `n` is even and nonpositive real
when `n` is odd. -/
lemma iteratedDeriv_LSeries_alternating (a : ArithmeticFunction ℂ) (hn : ∀ n, 0 ≤ a n) {x : ℝ}
    (h : LSeries.abscissaOfAbsConv a < x) (n : ℕ) :
    0 ≤ (-1) ^ n * iteratedDeriv n (LSeries (a ·)) x :=
  LSeries.iteratedDeriv_alternating hn h n


/-- If all values of a `ℂ`-valued arithmetic function `a` are nonnegative reals and `a 1` is
positive, then `L a x` is positive real for all real `x` larger than `abscissaOfAbsConv a`. -/
lemma LSeries_positive {a : ℕ → ℂ} (ha₀ : 0 ≤ a) (ha₁ : 0 < a 1) {x : ℝ}
    (hx : LSeries.abscissaOfAbsConv a < x) :
    0 < LSeries a x :=
  LSeries.positive ha₀ ha₁ hx


/-- If all values of a `ℂ`-valued arithmetic function `a` are nonnegative reals and `a 1`
is positive, and the L-series of `a` agrees with an entire function `f` on some open
right half-plane where it converges, then `f` is real and positive on `ℝ`. -/
lemma LSeries_positive_of_differentiable_of_eqOn {a : ArithmeticFunction ℂ} (ha₀ : 0 ≤ (a ·))
    (ha₁ : 0 < a 1) {f : ℂ → ℂ} (hf : Differentiable ℂ f) {x : ℝ}
    (hx : LSeries.abscissaOfAbsConv a ≤ x) (hf' : {s | x < s.re}.EqOn f (LSeries a)) (y : ℝ) :
    0 < f y :=
  LSeries.positive_of_differentiable_of_eqOn ha₀ ha₁ hf hx hf' y


