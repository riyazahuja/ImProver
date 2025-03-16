local macro "C_simp" : tactic =>
  `(tactic| simp only [map_ofNat, C_0, C_1, C_neg, C_add, C_sub, C_mul, C_pow])


local macro "derivative_simp" : tactic =>
  `(tactic| simp only [derivative_C, derivative_X, derivative_X_pow, derivative_neg, derivative_add,
    derivative_sub, derivative_mul, derivative_sq])


local macro "eval_simp" : tactic =>
  `(tactic| simp only [eval_C, eval_X, eval_neg, eval_add, eval_sub, eval_mul, eval_pow, evalEval])


local macro "map_simp" : tactic =>
  `(tactic| simp only [map_ofNat, map_neg, map_add, map_sub, map_mul, map_pow, map_div₀,
    Polynomial.map_ofNat, map_C, map_X, Polynomial.map_neg, Polynomial.map_add, Polynomial.map_sub,
    Polynomial.map_mul, Polynomial.map_pow, Polynomial.map_div, coe_mapRingHom,
    WeierstrassCurve.map])


/-- An abbreviation for a Weierstrass curve in affine coordinates. -/
abbrev WeierstrassCurve.Affine (R : Type u) : Type u :=
  WeierstrassCurve R


/-- The coercion to a Weierstrass curve in affine coordinates. -/
abbrev WeierstrassCurve.toAffine {R : Type u} (W : WeierstrassCurve R) : Affine R :=
  W


/-- The polynomial $W(X, Y) := Y^2 + a_1XY + a_3Y - (X^3 + a_2X^2 + a_4X + a_6)$ associated to a
Weierstrass curve `W` over `R`. For ease of polynomial manipulation, this is represented as a term
of type `R[X][X]`, where the inner variable represents $X$ and the outer variable represents $Y$.
For clarity, the alternative notations `Y` and `R[X][Y]` are provided in the `Polynomial`
scope to represent the outer variable and the bivariate polynomial ring `R[X][X]` respectively. -/
noncomputable def polynomial : R[X][Y] :=
  Y ^ 2 + C (C W.a₁ * X + C W.a₃) * Y - C (X ^ 3 + C W.a₂ * X ^ 2 + C W.a₄ * X + C W.a₆)


lemma polynomial_eq : W.polynomial =
    Cubic.toPoly
      ⟨0, 1, Cubic.toPoly ⟨0, 0, W.a₁, W.a₃⟩, Cubic.toPoly ⟨-1, -W.a₂, -W.a₄, -W.a₆⟩⟩ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq W.polynomial { a := 0, b := 1, c := { a := 0, b := 0, c := W.a₁, d := W.a …
  -/
  simp only [polynomial, Cubic.toPoly]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow Polynomial.X 2) (HMul.hMul (Polynomial.C …
  -/
  C_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow Polynomial.X 2) (HMul.hMul (HAdd.hAdd (H …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma polynomial_ne_zero [Nontrivial R] : W.polynomial ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : Nontrivial R
    ⊢ Ne W.polynomial 0
  -/
  rw [polynomial_eq]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : Nontrivial R
    ⊢ Ne { a := 0, b := 1, c := { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly, d …
  -/
  exact Cubic.ne_zero_of_b_ne_zero one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma degree_polynomial [Nontrivial R] : W.polynomial.degree = 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : Nontrivial R
    ⊢ Eq W.polynomial.degree 2
  -/
  rw [polynomial_eq]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : Nontrivial R
    ⊢ Eq { a := 0, b := 1, c := { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly, d …
  -/
  exact Cubic.degree_of_b_ne_zero' one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma natDegree_polynomial [Nontrivial R] : W.polynomial.natDegree = 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : Nontrivial R
    ⊢ Eq W.polynomial.natDegree 2
  -/
  rw [polynomial_eq]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : Nontrivial R
    ⊢ Eq { a := 0, b := 1, c := { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly, d …
  -/
  exact Cubic.natDegree_of_b_ne_zero' one_ne_zero
  /-
    🎉 no goals
  -/


lemma monic_polynomial : W.polynomial.Monic := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ W.polynomial.Monic
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    a✝ : Nontrivial R
    ⊢ W.polynomial.Monic
  -/
  simpa only [polynomial_eq] using Cubic.monic_of_b_eq_one'
  /-
    🎉 no goals
  -/


lemma irreducible_polynomial [IsDomain R] : Irreducible W.polynomial := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    ⊢ Irreducible W.polynomial
  -/
  by_contra h
  rcases (W.monic_polynomial.not_irreducible_iff_exists_add_mul_eq_coeff W.natDegree_polynomial).mp
    h with ⟨f, g, h0, h1⟩
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq (W.polynomial.coeff 0) (HMul.hMul f g)
    h1 : Eq (W.polynomial.coeff 1) (HAdd.hAdd f g)
    ⊢ False
  -/
  simp only [polynomial_eq, Cubic.coeff_eq_c, Cubic.coeff_eq_d] at h0 h1
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq { a := -1, b := Neg.neg W.a₂, c := Neg.neg W.a₄, d := Neg.neg W.a₆ }.t …
    h1 : Eq { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly (HAdd.hAdd f g)
    ⊢ False
  -/
  apply_fun degree at h0 h1
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq { a := -1, b := Neg.neg W.a₂, c := Neg.neg W.a₄, d := Neg.neg W.a₆ }.t …
    h1 : Eq { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly.degree (HAdd.hAdd f g) …
    ⊢ False
  -/
  rw [Cubic.degree_of_a_ne_zero' <| neg_ne_zero.mpr <| one_ne_zero' R, degree_mul] at h0
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq 3 (HAdd.hAdd f.degree g.degree)
    h1 : Eq { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly.degree (HAdd.hAdd f g) …
    ⊢ False
  -/
  apply (h1.symm.le.trans Cubic.degree_of_b_eq_zero').not_lt
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq 3 (HAdd.hAdd f.degree g.degree)
    h1 : Eq { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly.degree (HAdd.hAdd f g) …
    ⊢ LT.lt 1 (HAdd.hAdd f g).degree
  -/
  rcases Nat.WithBot.add_eq_three_iff.mp h0.symm with h | h | h | h
  -- Porting note: replaced two `any_goals` proofs with two `iterate 2` proofs
  /-
    case intro.intro.intro.inl
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h✝ : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq 3 (HAdd.hAdd f.degree g.degree)
    h1 : Eq { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly.degree (HAdd.hAdd f g) …
    h : And (Eq f.degree 0) (Eq g.degree 3)
    ⊢ LT.lt 1 (HAdd.hAdd f g).degree
  -/
  iterate 2 rw [degree_add_eq_right_of_degree_lt] <;> simp only [h] <;> decide
  /-
    case intro.intro.intro.inr.inr.inl
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    inst✝ : IsDomain R
    h✝ : Not (Irreducible W.polynomial)
    f g : Polynomial R
    h0 : Eq 3 (HAdd.hAdd f.degree g.degree)
    h1 : Eq { a := 0, b := 0, c := W.a₁, d := W.a₃ }.toPoly.degree (HAdd.hAdd f g) …
    h : And (Eq f.degree 2) (Eq g.degree 1)
    ⊢ LT.lt 1 (HAdd.hAdd f g).degree
  -/
  iterate 2 rw [degree_add_eq_left_of_degree_lt] <;> simp only [h] <;> decide
  /-
    🎉 no goals
  -/


lemma evalEval_polynomial (x y : R) : W.polynomial.evalEval x y =
    y ^ 2 + W.a₁ * x * y + W.a₃ * y - (x ^ 3 + W.a₂ * x ^ 2 + W.a₄ * x + W.a₆) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y W.polynomial) (HSub.hSub (HAdd.hAdd (HAdd.hAdd ( …
  -/
  simp only [polynomial]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y (HSub.hSub (HAdd.hAdd (HPow.hPow Polynomial.X 2) …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow y 2) (HMul.hMul (HAdd.hAdd (HMul.hMul W. …
  -/
  rw [add_mul, ← add_assoc]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_polynomial_zero : W.polynomial.evalEval 0 0 = -W.a₆ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (Polynomial.evalEval 0 0 W.polynomial) (Neg.neg W.a₆)
  -/
  simp only [evalEval_polynomial, zero_add, zero_sub, mul_zero, zero_pow <| Nat.succ_ne_zero _]
  /-
    🎉 no goals
  -/


/-- The proposition that an affine point $(x, y)$ lies in `W`. In other words, $W(x, y) = 0$. -/
def Equation (x y : R) : Prop :=
  W.polynomial.evalEval x y = 0


lemma equation_iff' (x y : R) : W.Equation x y ↔
    y ^ 2 + W.a₁ * x * y + W.a₃ * y - (x ^ 3 + W.a₂ * x ^ 2 + W.a₄ * x + W.a₆) = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Equation x y) (Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow y 2) (H …
  -/
  rw [Equation, evalEval_polynomial]
  /-
    🎉 no goals
  -/


lemma equation_iff (x y : R) :
    W.Equation x y ↔ y ^ 2 + W.a₁ * x * y + W.a₃ * y = x ^ 3 + W.a₂ * x ^ 2 + W.a₄ * x + W.a₆ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Equation x y) (Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y 2) (HMul.hMul (H …
  -/
  rw [equation_iff', sub_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma equation_zero : W.Equation 0 0 ↔ W.a₆ = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Iff (W.Equation 0 0) (Eq W.a₆ 0)
  -/
  rw [Equation, evalEval_polynomial_zero, neg_eq_zero]
  /-
    🎉 no goals
  -/


lemma equation_iff_variableChange (x y : R) :
    W.Equation x y ↔ (W.variableChange ⟨1, x, 0, y⟩).toAffine.Equation 0 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Equation x y) ((WeierstrassCurve.variableChange W { u := 1, r := x, s …
  -/
  rw [equation_iff', ← neg_eq_zero, equation_zero, variableChange_a₆, inv_one, Units.val_one]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (Eq (Neg.neg (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow y 2) (HMul.hMul …
  -/
  congr! 1
  /-
    case a.h.e'_2
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Neg.neg (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow y 2) (HMul.hMul (HMu …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The partial derivative $W_X(X, Y)$ of $W(X, Y)$ with respect to $X$.

TODO: define this in terms of `Polynomial.derivative`. -/
noncomputable def polynomialX : R[X][Y] :=
  C (C W.a₁) * Y - C (C 3 * X ^ 2 + C (2 * W.a₂) * X + C W.a₄)


lemma evalEval_polynomialX (x y : R) :
    W.polynomialX.evalEval x y = W.a₁ * y - (3 * x ^ 2 + 2 * W.a₂ * x + W.a₄) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y W.polynomialX) (HSub.hSub (HMul.hMul W.a₁ y) (HA …
  -/
  simp only [polynomialX]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y (HSub.hSub (HMul.hMul (Polynomial.C (Polynomial. …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_polynomialX_zero : W.polynomialX.evalEval 0 0 = -W.a₄ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (Polynomial.evalEval 0 0 W.polynomialX) (Neg.neg W.a₄)
  -/
  simp only [evalEval_polynomialX, zero_add, zero_sub, mul_zero, zero_pow <| Nat.succ_ne_zero _]
  /-
    🎉 no goals
  -/


/-- The partial derivative $W_Y(X, Y)$ of $W(X, Y)$ with respect to $Y$.

TODO: define this in terms of `Polynomial.derivative`. -/
noncomputable def polynomialY : R[X][Y] :=
  C (C 2) * Y + C (C W.a₁ * X + C W.a₃)


lemma evalEval_polynomialY (x y : R) :
    W.polynomialY.evalEval x y = 2 * y + W.a₁ * x + W.a₃ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y W.polynomialY) (HAdd.hAdd (HAdd.hAdd (HMul.hMul  …
  -/
  simp only [polynomialY]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y (HAdd.hAdd (HMul.hMul (Polynomial.C (Polynomial. …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 y) (HAdd.hAdd (HMul.hMul W.a₁ x) W.a₃)) (HAdd.hAd …
  -/
  rw [← add_assoc]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_polynomialY_zero : W.polynomialY.evalEval 0 0 = W.a₃ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (Polynomial.evalEval 0 0 W.polynomialY) W.a₃
  -/
  simp only [evalEval_polynomialY, zero_add, mul_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-19")] alias eval_polynomial := evalEval_polynomial

@[deprecated (since := "2024-06-19")] alias eval_polynomial_zero := evalEval_polynomial_zero

@[deprecated (since := "2024-06-19")] alias eval_polynomialX := evalEval_polynomialX

@[deprecated (since := "2024-06-19")] alias eval_polynomialX_zero := evalEval_polynomialX_zero

@[deprecated (since := "2024-06-19")] alias eval_polynomialY := evalEval_polynomialY

@[deprecated (since := "2024-06-19")] alias eval_polynomialY_zero := evalEval_polynomialY_zero


/-- The proposition that an affine point $(x, y)$ in `W` is nonsingular.
In other words, either $W_X(x, y) \ne 0$ or $W_Y(x, y) \ne 0$.

Note that this definition is only mathematically accurate for fields.
TODO: generalise this definition to be mathematically accurate for a larger class of rings. -/
def Nonsingular (x y : R) : Prop :=
  W.Equation x y ∧ (W.polynomialX.evalEval x y ≠ 0 ∨ W.polynomialY.evalEval x y ≠ 0)


lemma nonsingular_iff' (x y : R) : W.Nonsingular x y ↔ W.Equation x y ∧
    (W.a₁ * y - (3 * x ^ 2 + 2 * W.a₂ * x + W.a₄) ≠ 0 ∨ 2 * y + W.a₁ * x + W.a₃ ≠ 0) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Nonsingular x y) (And (W.Equation x y) (Or (Ne (HSub.hSub (HMul.hMul  …
  -/
  rw [Nonsingular, equation_iff', evalEval_polynomialX, evalEval_polynomialY]
  /-
    🎉 no goals
  -/


lemma nonsingular_iff (x y : R) : W.Nonsingular x y ↔
    W.Equation x y ∧ (W.a₁ * y ≠ 3 * x ^ 2 + 2 * W.a₂ * x + W.a₄ ∨ y ≠ -y - W.a₁ * x - W.a₃) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Nonsingular x y) (And (W.Equation x y) (Or (Ne (HMul.hMul W.a₁ y) (HA …
  -/
  rw [nonsingular_iff', sub_ne_zero, ← sub_ne_zero (a := y)]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (And (W.Equation x y) (Or (Ne (HMul.hMul W.a₁ y) (HAdd.hAdd (HAdd.hAdd ( …
  -/
  congr! 3
  /-
    case a.h.e'_2.h.e'_2.h.e'_2
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 y) (HMul.hMul W.a₁ x)) W.a₃) (HSub.hSu …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma nonsingular_zero : W.Nonsingular 0 0 ↔ W.a₆ = 0 ∧ (W.a₃ ≠ 0 ∨ W.a₄ ≠ 0) := by
  rw [Nonsingular, equation_zero, evalEval_polynomialX_zero, neg_ne_zero, evalEval_polynomialY_zero,
    or_comm]


lemma nonsingular_iff_variableChange (x y : R) :
    W.Nonsingular x y ↔ (W.variableChange ⟨1, x, 0, y⟩).toAffine.Nonsingular 0 0 := by
  rw [nonsingular_iff', equation_iff_variableChange, equation_zero, ← neg_ne_zero, or_comm,
    nonsingular_zero, variableChange_a₃, variableChange_a₄, inv_one, Units.val_one]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (And (Eq (WeierstrassCurve.variableChange W { u := 1, r := x, s := 0, t  …
  -/
  simp only [variableChange]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (And (Eq (HMul.hMul (HPow.hPow (↑(Inv.inv 1)) 6) (HSub.hSub (HSub.hSub ( …
  -/
               /-
                 🎉 no goals
               -/
  congr! 3 <;> ring1
               /-
                 🎉 no goals
               -/


lemma nonsingular_zero_of_Δ_ne_zero (h : W.Equation 0 0) (hΔ : W.Δ ≠ 0) : W.Nonsingular 0 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    h : W.Equation 0 0
    hΔ : Ne (WeierstrassCurve.Δ W) 0
    ⊢ W.Nonsingular 0 0
  -/
  simp only [equation_zero, nonsingular_zero] at *
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    hΔ : Ne (WeierstrassCurve.Δ W) 0
    h : Eq W.a₆ 0
    ⊢ And (Eq W.a₆ 0) (Or (Ne W.a₃ 0) (Ne W.a₄ 0))
  -/
  contrapose! hΔ
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    h : Eq W.a₆ 0
    hΔ : Eq W.a₆ 0 → And (Eq W.a₃ 0) (Eq W.a₄ 0)
    ⊢ Eq (WeierstrassCurve.Δ W) 0
  -/
  simp only [b₂, b₄, b₆, b₈, Δ, h, hΔ]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    h : Eq W.a₆ 0
    hΔ : Eq W.a₆ 0 → And (Eq W.a₃ 0) (Eq W.a₄ 0)
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HAdd.hAd …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- A Weierstrass curve is nonsingular at every point if its discriminant is non-zero. -/
lemma nonsingular_of_Δ_ne_zero {x y : R} (h : W.Equation x y) (hΔ : W.Δ ≠ 0) : W.Nonsingular x y :=
  (W.nonsingular_iff_variableChange x y).mpr <|
    nonsingular_zero_of_Δ_ne_zero _ ((W.equation_iff_variableChange x y).mp h) <| by
      /-
        R : Type u
        inst✝ : CommRing R
        W : WeierstrassCurve.Affine R
        x y : R
        h : W.Equation x y
        hΔ : Ne (WeierstrassCurve.Δ W) 0
        ⊢ Ne (WeierstrassCurve.Δ (WeierstrassCurve.variableChange W { u := 1, r := x,  …
      -/
      rwa [variableChange_Δ, inv_one, Units.val_one, one_pow, one_mul]
      /-
        🎉 no goals
      -/


/-- The polynomial $-Y - a_1X - a_3$ associated to negation. -/
noncomputable def negPolynomial : R[X][Y] :=
  -(Y : R[X][Y]) - C (C W.a₁ * X + C W.a₃)


lemma Y_sub_polynomialY : Y - W.polynomialY = W.negPolynomial := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (HSub.hSub Polynomial.X W.polynomialY) W.negPolynomial
  -/
  rw [polynomialY, negPolynomial]; C_simp; ring
                                           /-
                                             🎉 no goals
                                           -/


lemma Y_sub_negPolynomial : Y - W.negPolynomial = W.polynomialY := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (HSub.hSub Polynomial.X W.negPolynomial) W.polynomialY
  -/
  rw [← Y_sub_polynomialY, sub_sub_cancel]
  /-
    🎉 no goals
  -/


/-- The $Y$-coordinate of the negation of an affine point in `W`.

This depends on `W`, and has argument order: $x$, $y$. -/
@[simp]
def negY (x y : R) : R :=
  -y - W.a₁ * x - W.a₃


lemma negY_negY (x y : R) : W.negY x (W.negY x y) = y := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (W.negY x (W.negY x y)) y
  -/
  simp only [negY]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HSub.hSub (HSub.hSub (Neg.neg y) (HMul.hM …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_negPolynomial (x y : R) : W.negPolynomial.evalEval x y = W.negY x y := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y W.negPolynomial) (W.negY x y)
  -/
  rw [negY, sub_sub, negPolynomial]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y (HSub.hSub (Neg.neg Polynomial.X) (Polynomial.C  …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


/-- The polynomial $L(X - x) + y$ associated to the line $Y = L(X - x) + y$,
with a slope of $L$ that passes through an affine point $(x, y)$.

This does not depend on `W`, and has argument order: $x$, $y$, $L$. -/
noncomputable def linePolynomial (x y L : R) : R[X] :=
  C L * (X - C x) + C y


/-- The polynomial obtained by substituting the line $Y = L*(X - x) + y$, with a slope of $L$
that passes through an affine point $(x, y)$, into the polynomial $W(X, Y)$ associated to `W`.
If such a line intersects `W` at another point $(x', y')$, then the roots of this polynomial are
precisely $x$, $x'$, and the $X$-coordinate of the addition of $(x, y)$ and $(x', y')$.

This depends on `W`, and has argument order: $x$, $y$, $L$. -/
noncomputable def addPolynomial (x y L : R) : R[X] :=
  W.polynomial.eval <| linePolynomial x y L


lemma C_addPolynomial (x y L : R) : C (W.addPolynomial x y L) =
    (Y - C (linePolynomial x y L)) * (W.negPolynomial - C (linePolynomial x y L)) +
      W.polynomial := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (Polynomial.C (W.addPolynomial x y L)) (HAdd.hAdd (HMul.hMul (HSub.hSub P …
  -/
  rw [addPolynomial, linePolynomial, polynomial, negPolynomial]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (Polynomial.C (Polynomial.eval (HAdd.hAdd (HMul.hMul (Polynomial.C L) (HS …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (Polynomial.C (HSub.hSub (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul (Pol …
  -/
  C_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul (Polynomial.C (Pol …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addPolynomial_eq (x y L : R) : W.addPolynomial x y L = -Cubic.toPoly
    ⟨1, -L ^ 2 - W.a₁ * L + W.a₂,
      2 * x * L ^ 2 + (W.a₁ * x - 2 * y - W.a₃) * L + (-W.a₁ * y + W.a₄),
      -x ^ 2 * L ^ 2 + (2 * x * y + W.a₃ * x) * L - (y ^ 2 + W.a₃ * y - W.a₆)⟩ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (W.addPolynomial x y L) (Neg.neg { a := 1, b := HAdd.hAdd (HSub.hSub (Neg …
  -/
  rw [addPolynomial, linePolynomial, polynomial, Cubic.toPoly]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (Polynomial.eval (HAdd.hAdd (HMul.hMul (Polynomial.C L) (HSub.hSub Polyno …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul (Polynomial.C L) ( …
  -/
  C_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y L : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul (Polynomial.C L) ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The $X$-coordinate of the addition of two affine points $(x_1, y_1)$ and $(x_2, y_2)$ in `W`,
where the line through them is not vertical and has a slope of $L$.

This depends on `W`, and has argument order: $x_1$, $x_2$, $L$. -/
@[simp]
def addX (x₁ x₂ L : R) : R :=
  L ^ 2 + W.a₁ * L - W.a₂ - x₁ - x₂


/-- The $Y$-coordinate of the negated addition of two affine points $(x_1, y_1)$ and $(x_2, y_2)$,
where the line through them is not vertical and has a slope of $L$.

This depends on `W`, and has argument order: $x_1$, $x_2$, $y_1$, $L$. -/
@[simp]
def negAddY (x₁ x₂ y₁ L : R) : R :=
  L * (W.addX x₁ x₂ L - x₁) + y₁


/-- The $Y$-coordinate of the addition of two affine points $(x_1, y_1)$ and $(x_2, y_2)$ in `W`,
where the line through them is not vertical and has a slope of $L$.

This depends on `W`, and has argument order: $x_1$, $x_2$, $y_1$, $L$. -/
@[simp]
def addY (x₁ x₂ y₁ L : R) : R :=
  W.negY (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)


lemma equation_neg_iff (x y : R) : W.Equation x (W.negY x y) ↔ W.Equation x y := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Equation x (W.negY x y)) (W.Equation x y)
  -/
  rw [equation_iff, equation_iff, negY]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow (HSub.hSub (HSub.hSub (Neg.neg y) ( …
  -/
  congr! 1
  /-
    case a.h.e'_2
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow (HSub.hSub (HSub.hSub (Neg.neg y) (HMul. …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma nonsingular_neg_iff (x y : R) : W.Nonsingular x (W.negY x y) ↔ W.Nonsingular x y := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x y : R
    ⊢ Iff (W.Nonsingular x (W.negY x y)) (W.Nonsingular x y)
  -/
  rw [nonsingular_iff, equation_neg_iff, ← negY, negY_negY, ← @ne_comm _ y, nonsingular_iff]
  exact and_congr_right' <| (iff_congr not_and_or.symm not_and_or.symm).mpr <|
    not_congr <| and_congr_left fun h => by rw [← h]


lemma equation_add_iff (x₁ x₂ y₁ L : R) :
    W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L) ↔
      (W.addPolynomial x₁ y₁ L).eval (W.addX x₁ x₂ L) = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    ⊢ Iff (W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)) (Eq (Polynomial.eva …
  -/
  rw [Equation, negAddY, addPolynomial, linePolynomial, polynomial]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    ⊢ Iff (Eq (Polynomial.evalEval (W.addX x₁ x₂ L) (HAdd.hAdd (HMul.hMul L (HSub. …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma equation_neg_of {x y : R} (h : W.Equation x <| W.negY x y) : W.Equation x y :=
  (W.equation_neg_iff ..).mp h


/-- The negation of an affine point in `W` lies in `W`. -/
lemma equation_neg {x y : R} (h : W.Equation x y) : W.Equation x <| W.negY x y :=
  (W.equation_neg_iff ..).mpr h


lemma nonsingular_neg_of {x y : R} (h : W.Nonsingular x <| W.negY x y) : W.Nonsingular x y :=
  (W.nonsingular_neg_iff ..).mp h


/-- The negation of a nonsingular affine point in `W` is nonsingular. -/
lemma nonsingular_neg {x y : R} (h : W.Nonsingular x y) : W.Nonsingular x <| W.negY x y :=
  (W.nonsingular_neg_iff ..).mpr h


lemma nonsingular_negAdd_of_eval_derivative_ne_zero {x₁ x₂ y₁ L : R}
    (hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L))
    (hx : (W.addPolynomial x₁ y₁ L).derivative.eval (W.addX x₁ x₂ L) ≠ 0) :
    W.Nonsingular (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : Ne (Polynomial.eval (W.addX x₁ x₂ L) (Polynomial.derivative (W.addPolynom …
    ⊢ W.Nonsingular (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
  -/
  rw [Nonsingular, and_iff_right hx', negAddY, polynomialX, polynomialY]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : Ne (Polynomial.eval (W.addX x₁ x₂ L) (Polynomial.derivative (W.addPolynom …
    ⊢ Or (Ne (Polynomial.evalEval (W.addX x₁ x₂ L) (HAdd.hAdd (HMul.hMul L (HSub.h …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : Ne (Polynomial.eval (W.addX x₁ x₂ L) (Polynomial.derivative (W.addPolynom …
    ⊢ Or (Ne (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W.addX …
  -/
  contrapose! hx
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : And (Eq (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W. …
    ⊢ Eq (Polynomial.eval (W.addX x₁ x₂ L) (Polynomial.derivative (W.addPolynomial …
  -/
  rw [addPolynomial, linePolynomial, polynomial]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : And (Eq (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W. …
    ⊢ Eq (Polynomial.eval (W.addX x₁ x₂ L) (Polynomial.derivative (Polynomial.eval …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : And (Eq (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W. …
    ⊢ Eq (Polynomial.eval (W.addX x₁ x₂ L) (Polynomial.derivative (HSub.hSub (HAdd …
  -/
  derivative_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : And (Eq (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W. …
    ⊢ Eq (Polynomial.eval (W.addX x₁ x₂ L) (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul. …
  -/
  simp only [zero_add, add_zero, sub_zero, zero_mul, mul_one]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : And (Eq (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W. …
    ⊢ Eq (Polynomial.eval (W.addX x₁ x₂ L) (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul. …
  -/
  eval_simp
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    x₁ x₂ y₁ L : R
    hx' : W.Equation (W.addX x₁ x₂ L) (W.negAddY x₁ x₂ y₁ L)
    hx : And (Eq (HSub.hSub (HMul.hMul W.a₁ (HAdd.hAdd (HMul.hMul L (HSub.hSub (W. …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (HMul.hMul L (HS …
  -/
  linear_combination (norm := (norm_num1; ring1)) hx.left + L * hx.right
  /-
    🎉 no goals
  -/


open Classical in
/-- The slope of the line through two affine points $(x_1, y_1)$ and $(x_2, y_2)$ in `W`.
If $x_1 \ne x_2$, then this line is the secant of `W` through $(x_1, y_1)$ and $(x_2, y_2)$,
and has slope $(y_1 - y_2) / (x_1 - x_2)$. Otherwise, if $y_1 \ne -y_1 - a_1x_1 - a_3$,
then this line is the tangent of `W` at $(x_1, y_1) = (x_2, y_2)$, and has slope
$(3x_1^2 + 2a_2x_1 + a_4 - a_1y_1) / (2y_1 + a_1x_1 + a_3)$. Otherwise, this line is vertical,
and has undefined slope, in which case this function returns the value 0.

This depends on `W`, and has argument order: $x_1$, $x_2$, $y_1$, $y_2$. -/
noncomputable def slope {F : Type u} [Field F] (W : Affine F) (x₁ x₂ y₁ y₂ : F) : F :=
  if x₁ = x₂ then if y₁ = W.negY x₂ y₂ then 0
    else (3 * x₁ ^ 2 + 2 * W.a₂ * x₁ + W.a₄ - W.a₁ * y₁) / (y₁ - W.negY x₁ y₁)
  else (y₁ - y₂) / (x₁ - x₂)


@[simp]
lemma slope_of_Y_eq {x₁ x₂ y₁ y₂ : F} (hx : x₁ = x₂) (hy : y₁ = W.negY x₂ y₂) :
    W.slope x₁ x₂ y₁ y₂ = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Eq x₁ x₂
    hy : Eq y₁ (W.negY x₂ y₂)
    ⊢ Eq (W.slope x₁ x₂ y₁ y₂) 0
  -/
  rw [slope, if_pos hx, if_pos hy]
  /-
    🎉 no goals
  -/


@[simp]
lemma slope_of_Y_ne {x₁ x₂ y₁ y₂ : F} (hx : x₁ = x₂) (hy : y₁ ≠ W.negY x₂ y₂) :
    W.slope x₁ x₂ y₁ y₂ =
      (3 * x₁ ^ 2 + 2 * W.a₂ * x₁ + W.a₄ - W.a₁ * y₁) / (y₁ - W.negY x₁ y₁) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Eq x₁ x₂
    hy : Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (W.slope x₁ x₂ y₁ y₂) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  rw [slope, if_pos hx, if_neg hy]
  /-
    🎉 no goals
  -/


@[simp]
lemma slope_of_X_ne {x₁ x₂ y₁ y₂ : F} (hx : x₁ ≠ x₂) :
    W.slope x₁ x₂ y₁ y₂ = (y₁ - y₂) / (x₁ - x₂) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (W.slope x₁ x₂ y₁ y₂) (HDiv.hDiv (HSub.hSub y₁ y₂) (HSub.hSub x₁ x₂))
  -/
  rw [slope, if_neg hx]
  /-
    🎉 no goals
  -/


lemma slope_of_Y_ne_eq_eval {x₁ x₂ y₁ y₂ : F} (hx : x₁ = x₂) (hy : y₁ ≠ W.negY x₂ y₂) :
    W.slope x₁ x₂ y₁ y₂ = -W.polynomialX.evalEval x₁ y₁ / W.polynomialY.evalEval x₁ y₁ := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Eq x₁ x₂
    hy : Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (W.slope x₁ x₂ y₁ y₂) (HDiv.hDiv (Neg.neg (Polynomial.evalEval x₁ y₁ W.po …
  -/
  rw [slope_of_Y_ne hx hy, evalEval_polynomialX, neg_sub]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Eq x₁ x₂
    hy : Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow x₁ 2) …
  -/
  congr 1
  /-
    case e_a
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Eq x₁ x₂
    hy : Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (HSub.hSub y₁ (W.negY x₁ y₁)) (Polynomial.evalEval x₁ y₁ W.polynomialY)
  -/
  rw [negY, evalEval_polynomialY]
  /-
    case e_a
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Eq x₁ x₂
    hy : Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMul W.a₁ x₁)) W.a …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma Y_eq_of_X_eq {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂)
    (hx : x₁ = x₂) : y₁ = y₂ ∨ y₁ = W.negY x₂ y₂ := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hx : Eq x₁ x₂
    ⊢ Or (Eq y₁ y₂) (Eq y₁ (W.negY x₂ y₂))
  -/
  rw [equation_iff] at h₁ h₂
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
    h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
    hx : Eq x₁ x₂
    ⊢ Or (Eq y₁ y₂) (Eq y₁ (W.negY x₂ y₂))
  -/
  rw [← sub_eq_zero, ← sub_eq_zero (a := y₁), ← mul_eq_zero, negY]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
    h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
    hx : Eq x₁ x₂
    ⊢ Eq (HMul.hMul (HSub.hSub y₁ y₂) (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg …
  -/
  linear_combination (norm := (rw [hx]; ring1)) h₁ - h₂
  /-
    🎉 no goals
  -/


lemma Y_eq_of_Y_ne {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂) (hx : x₁ = x₂)
    (hy : y₁ ≠ W.negY x₂ y₂) : y₁ = y₂ :=
  (Y_eq_of_X_eq h₁ h₂ hx).resolve_right hy


lemma addPolynomial_slope {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) : W.addPolynomial x₁ y₁ (W.slope x₁ x₂ y₁ y₂) =
      -((X - C x₁) * (X - C x₂) * (X - C (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂))) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (W.addPolynomial x₁ y₁ (W.slope x₁ x₂ y₁ y₂)) (Neg.neg (HMul.hMul (HMul.h …
  -/
  rw [addPolynomial_eq, neg_inj, Cubic.prod_X_sub_C_eq, Cubic.toPoly_injective]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (W.slope x₁ x₂ y₁ …
  -/
  by_cases hx : x₁ = x₂
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx : Eq x₁ x₂
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (W.slope x₁ x₂ y₁ …
    -/
  · rcases hx, Y_eq_of_Y_ne h₁ h₂ hx (hxy hx) with ⟨rfl, rfl⟩
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ h₂ : W.Equation x₁ y₁
      hxy : Eq x₁ x₁ → Ne y₁ (W.negY x₁ y₁)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (W.slope x₁ x₁ y₁ …
    -/
    rw [equation_iff] at h₁ h₂
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
      hxy : Eq x₁ x₁ → Ne y₁ (W.negY x₁ y₁)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (W.slope x₁ x₁ y₁ …
    -/
    rw [slope_of_Y_ne rfl <| hxy rfl]
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
      hxy : Eq x₁ x₁ → Ne y₁ (W.negY x₁ y₁)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
    -/
    rw [negY, ← sub_ne_zero] at hxy
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ y₁ : F
      h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
      hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
    -/
    ext
      /-
        case pos.a
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
        hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos.b
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
        hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · simp only [addX]
      /-
        case pos.b
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
        hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
        ⊢ Eq (HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub.hSub (HAdd.hAd …
      -/
      ring1
      /-
        🎉 no goals
      -/
      /-
        case pos.c
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
        hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · field_simp [hxy rfl]
      /-
        case pos.c
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
        hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul 2 x₁) (HPow.hPow ( …
      -/
      ring1
      /-
        🎉 no goals
      -/
      /-
        case pos.d
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ y₁ : F
        h₁ h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x …
        hxy : Eq x₁ x₁ → Ne (HSub.hSub y₁ (HSub.hSub (HSub.hSub (Neg.neg y₁) (HMul.hMu …
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · linear_combination (norm := (field_simp [hxy rfl]; ring1)) -h₁
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Equation x₁ y₁
      h₂ : W.Equation x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx : Not (Eq x₁ x₂)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (W.slope x₁ x₂ y₁ …
    -/
  · rw [equation_iff] at h₁ h₂
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
      h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx : Not (Eq x₁ x₂)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (W.slope x₁ x₂ y₁ …
    -/
    rw [slope_of_X_ne hx]
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
      h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx : Not (Eq x₁ x₂)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
    -/
    rw [← sub_eq_zero] at hx
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
      h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx : Not (Eq (HSub.hSub x₁ x₂) 0)
      ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
    -/
    ext
      /-
        case neg.a
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg.b
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · simp only [addX]
      /-
        case neg.b
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq (HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub.hSub y₁ y₂) (H …
      -/
      ring1
      /-
        🎉 no goals
      -/
      /-
        case neg.c
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · apply mul_right_injective₀ hx
      /-
        case neg.c.a
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq ((fun x => HMul.hMul (HSub.hSub x₁ x₂) x) { a := 1, b := HAdd.hAdd (HSub. …
      -/
      linear_combination (norm := (field_simp [hx]; ring1)) h₂ - h₁
      /-
        🎉 no goals
      -/
      /-
        case neg.d
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq { a := 1, b := HAdd.hAdd (HSub.hSub (Neg.neg (HPow.hPow (HDiv.hDiv (HSub. …
      -/
    · apply mul_right_injective₀ hx
      /-
        case neg.d.a
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₁ 2) (HMul.hMul (HMul.hMul W.a₁ x₁)  …
        h₂ : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow y₂ 2) (HMul.hMul (HMul.hMul W.a₁ x₂)  …
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx : Not (Eq (HSub.hSub x₁ x₂) 0)
        ⊢ Eq ((fun x => HMul.hMul (HSub.hSub x₁ x₂) x) { a := 1, b := HAdd.hAdd (HSub. …
      -/
      linear_combination (norm := (field_simp [hx]; ring1)) x₂ * h₁ - x₁ * h₂
      /-
        🎉 no goals
      -/


/-- The negated addition of two affine points in `W` on a sloped line lies in `W`. -/
lemma equation_negAdd {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) : W.Equation
      (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂) (W.negAddY x₁ x₂ y₁ <| W.slope x₁ x₂ y₁ y₂) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ W.Equation (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.slope …
  -/
  rw [equation_add_iff, addPolynomial_slope h₁ h₂ hxy]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (Polynomial.eval (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (Neg.neg (HMul.hMul …
  -/
  eval_simp
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (Neg.neg (HMul.hMul (HMul.hMul (HSub.hSub (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ …
  -/
  rw [neg_eq_zero, sub_self, mul_zero]
  /-
    🎉 no goals
  -/


/-- The addition of two affine points in `W` on a sloped line lies in `W`. -/
lemma equation_add {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁) (h₂ : W.Equation x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) :
    W.Equation (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂) (W.addY x₁ x₂ y₁ <| W.slope x₁ x₂ y₁ y₂) :=
  equation_neg <| equation_negAdd h₁ h₂ hxy


lemma derivative_addPolynomial_slope {x₁ x₂ y₁ y₂ : F} (h₁ : W.Equation x₁ y₁)
    (h₂ : W.Equation x₂ y₂) (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) :
    derivative (W.addPolynomial x₁ y₁ <| W.slope x₁ x₂ y₁ y₂) =
      -((X - C x₁) * (X - C x₂) + (X - C x₁) * (X - C (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂)) +
          (X - C x₂) * (X - C (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂))) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (Polynomial.derivative (W.addPolynomial x₁ y₁ (W.slope x₁ x₂ y₁ y₂))) (Ne …
  -/
  rw [addPolynomial_slope h₁ h₂ hxy]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (Polynomial.derivative (Neg.neg (HMul.hMul (HMul.hMul (HSub.hSub Polynomi …
  -/
  derivative_simp
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Equation x₁ y₁
    h₂ : W.Equation x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ Eq (Neg.neg (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul (HSub.hSub 1 0) (HSu …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The negated addition of two nonsingular affine points in `W` on a sloped line is nonsingular. -/
lemma nonsingular_negAdd {x₁ x₂ y₁ y₂ : F} (h₁ : W.Nonsingular x₁ y₁) (h₂ : W.Nonsingular x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) : W.Nonsingular
      (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂) (W.negAddY x₁ x₂ y₁ <| W.slope x₁ x₂ y₁ y₂) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Nonsingular x₁ y₁
    h₂ : W.Nonsingular x₂ y₂
    hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
    ⊢ W.Nonsingular (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.sl …
  -/
  by_cases hx₁ : W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂) = x₁
    /-
      case pos
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Nonsingular x₁ y₁
      h₂ : W.Nonsingular x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx₁ : Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁
      ⊢ W.Nonsingular (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.sl …
    -/
  · rwa [negAddY, hx₁, sub_self, mul_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u
      inst✝ : Field F
      W : WeierstrassCurve.Affine F
      x₁ x₂ y₁ y₂ : F
      h₁ : W.Nonsingular x₁ y₁
      h₂ : W.Nonsingular x₂ y₂
      hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
      hx₁ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁)
      ⊢ W.Nonsingular (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.sl …
    -/
  · by_cases hx₂ : W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂) = x₂
      /-
        case pos
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : W.Nonsingular x₁ y₁
        h₂ : W.Nonsingular x₂ y₂
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx₁ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁)
        hx₂ : Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₂
        ⊢ W.Nonsingular (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.sl …
      -/
    · by_cases hx : x₁ = x₂
        /-
          case pos
          F : Type u
          inst✝ : Field F
          W : WeierstrassCurve.Affine F
          x₁ x₂ y₁ y₂ : F
          h₁ : W.Nonsingular x₁ y₁
          h₂ : W.Nonsingular x₂ y₂
          hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
          hx₁ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁)
          hx₂ : Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₂
          hx : Eq x₁ x₂
          ⊢ W.Nonsingular (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.sl …
        -/
      · subst hx
        /-
          case pos
          F : Type u
          inst✝ : Field F
          W : WeierstrassCurve.Affine F
          x₁ y₁ y₂ : F
          h₁ : W.Nonsingular x₁ y₁
          h₂ : W.Nonsingular x₁ y₂
          hxy : Eq x₁ x₁ → Ne y₁ (W.negY x₁ y₂)
          hx₁ : Not (Eq (W.addX x₁ x₁ (W.slope x₁ x₁ y₁ y₂)) x₁)
          hx₂ : Eq (W.addX x₁ x₁ (W.slope x₁ x₁ y₁ y₂)) x₁
          ⊢ W.Nonsingular (W.addX x₁ x₁ (W.slope x₁ x₁ y₁ y₂)) (W.negAddY x₁ x₁ y₁ (W.sl …
        -/
        contradiction
        /-
          🎉 no goals
        -/
      · rwa [negAddY, ← neg_sub, mul_neg, hx₂, slope_of_X_ne hx,
          div_mul_cancel₀ _ <| sub_ne_zero_of_ne hx, neg_sub, sub_add_cancel]
      /-
        case neg
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : W.Nonsingular x₁ y₁
        h₂ : W.Nonsingular x₂ y₂
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx₁ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁)
        hx₂ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₂)
        ⊢ W.Nonsingular (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (W.negAddY x₁ x₂ y₁ (W.sl …
      -/
    · apply nonsingular_negAdd_of_eval_derivative_ne_zero <| equation_negAdd h₁.1 h₂.1 hxy
      /-
        case neg
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : W.Nonsingular x₁ y₁
        h₂ : W.Nonsingular x₂ y₂
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx₁ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁)
        hx₂ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₂)
        ⊢ Ne (Polynomial.eval (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (Polynomial.derivat …
      -/
      rw [derivative_addPolynomial_slope h₁.left h₂.left hxy]
      /-
        case neg
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        x₁ x₂ y₁ y₂ : F
        h₁ : W.Nonsingular x₁ y₁
        h₂ : W.Nonsingular x₂ y₂
        hxy : Eq x₁ x₂ → Ne y₁ (W.negY x₂ y₂)
        hx₁ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₁)
        hx₂ : Not (Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) x₂)
        ⊢ Ne (Polynomial.eval (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (Neg.neg (HAdd.hAdd …
      -/
      eval_simp
      simpa only [neg_ne_zero, sub_self, mul_zero, add_zero] using
        mul_ne_zero (sub_ne_zero_of_ne hx₁) (sub_ne_zero_of_ne hx₂)


/-- The addition of two nonsingular affine points in `W` on a sloped line is nonsingular. -/
lemma nonsingular_add {x₁ x₂ y₁ y₂ : F} (h₁ : W.Nonsingular x₁ y₁) (h₂ : W.Nonsingular x₂ y₂)
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) :
    W.Nonsingular (W.addX x₁ x₂ <| W.slope x₁ x₂ y₁ y₂) (W.addY x₁ x₂ y₁ <| W.slope x₁ x₂ y₁ y₂) :=
  nonsingular_neg <| nonsingular_negAdd h₁ h₂ hxy


/-- The formula x(P₁ + P₂) = x(P₁ - P₂) - ψ(P₁)ψ(P₂) / (x(P₂) - x(P₁))²,
where ψ(x,y) = 2y + a₁x + a₃. -/
lemma addX_eq_addX_negY_sub (hx : x₁ ≠ x₂) :
    W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂) = W.addX x₁ x₂ (W.slope x₁ x₂ y₁ (W.negY x₂ y₂))
      - (y₁ - W.negY x₁ y₁) * (y₂ - W.negY x₂ y₂) / (x₂ - x₁) ^ 2 := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)) (HSub.hSub (W.addX x₁ x₂ (W.slope x₁ …
  -/
  simp_rw [slope_of_X_ne hx, addX, negY, ← neg_sub x₁, neg_sq]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HSub.hSub (Neg.neg (HSub.hSub x₁ (HSub.hSub (HAdd.hAdd (HPow.hPow (HDiv. …
  -/
  field_simp [sub_ne_zero.mpr hx]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HMul.hMul (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The formula y(P₁)(x(P₂) - x(P₃)) + y(P₂)(x(P₃) - x(P₁)) + y(P₃)(x(P₁) - x(P₂)) = 0,
assuming that P₁ + P₂ + P₃ = O. -/
lemma cyclic_sum_Y_mul_X_sub_X (hx : x₁ ≠ x₂) :
    letI x₃ := W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)
    y₁ * (x₂ - x₃) + y₂ * (x₃ - x₁) + W.negAddY x₁ x₂ y₁ (W.slope x₁ x₂ y₁ y₂) * (x₁ - x₂) = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul y₁ (HSub.hSub x₂ (W.addX x₁ x₂ (W.slope  …
  -/
  simp_rw [slope_of_X_ne hx, negAddY, addX]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul y₁ (HSub.hSub x₂ (HSub.hSub (HSub.hSub ( …
  -/
  field_simp [sub_ne_zero.mpr hx]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul y₁ (HSub.hSub (HMul.hMul x₂ ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The formula
ψ(P₁ + P₂) = (ψ(P₂)(x(P₁) - x(P₃)) - ψ(P₁)(x(P₂) - x(P₃))) / (x(P₂) - x(P₁)),
where ψ(x,y) = 2y + a₁x + a₃. -/
lemma addY_sub_negY_addY (hx : x₁ ≠ x₂) :
    letI x₃ := W.addX x₁ x₂ (W.slope x₁ x₂ y₁ y₂)
    letI y₃ := W.addY x₁ x₂ y₁ (W.slope x₁ x₂ y₁ y₂)
    y₃ - W.negY x₃ y₃ =
      ((y₂ - W.negY x₂ y₂) * (x₁ - x₃) - (y₁ - W.negY x₁ y₁) * (x₂ - x₃)) / (x₂ - x₁) := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HSub.hSub (W.addY x₁ x₂ y₁ (W.slope x₁ x₂ y₁ y₂)) (W.negY (W.addX x₁ x₂  …
  -/
  simp_rw [addY, negY, eq_div_iff (sub_ne_zero.mpr hx.symm)]
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    hx : Ne x₁ x₂
    ⊢ Eq (HMul.hMul (HSub.hSub (HSub.hSub (HSub.hSub (Neg.neg (W.negAddY x₁ x₂ y₁  …
  -/
  linear_combination 2 * cyclic_sum_Y_mul_X_sub_X y₁ y₂ hx
  /-
    🎉 no goals
  -/


/-- A nonsingular rational point on a Weierstrass curve `W` in affine coordinates. This is either
the unique point at infinity `WeierstrassCurve.Affine.Point.zero` or the nonsingular affine points
`WeierstrassCurve.Affine.Point.some` $(x, y)$ satisfying the Weierstrass equation of `W`. -/
inductive Point
  | zero
  | some {x y : R} (h : W.Nonsingular x y)


/-- For an algebraic extension `S` of `R`, the type of nonsingular `S`-rational points on `W`. -/
scoped notation3:max W "⟮" S "⟯" => Affine.Point <| baseChange W S


instance : Inhabited W.Point :=
  ⟨zero⟩


instance : Zero W.Point :=
  ⟨zero⟩


lemma zero_def : (zero : W.Point) = 0 :=
  rfl


                                                                        /-
                                                                          R : Type u
                                                                          inst✝ : CommRing R
                                                                          W : WeierstrassCurve.Affine R
                                                                          x y : R
                                                                          h : W.Nonsingular x y
                                                                          ⊢ Ne (WeierstrassCurve.Affine.Point.some h) 0
                                                                        -/
lemma some_ne_zero {x y : R} (h : W.Nonsingular x y) : some h ≠ 0 := by rintro (_|_)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The negation of a nonsingular rational point on `W`.

Given a nonsingular rational point `P` on `W`, use `-P` instead of `neg P`. -/
def neg : W.Point → W.Point
  | 0 => 0
  | some h => some <| nonsingular_neg h


instance : Neg W.Point :=
  ⟨neg⟩


lemma neg_def (P : W.Point) : P.neg = -P :=
  rfl


@[simp]
lemma neg_zero : (-0 : W.Point) = 0 :=
  rfl


@[simp]
lemma neg_some {x y : R} (h : W.Nonsingular x y) : -some h = some (nonsingular_neg h) :=
  rfl


instance : InvolutiveNeg W.Point :=
      /-
        R : Type u
        inst✝ : CommRing R
        W : WeierstrassCurve.Affine R
        ⊢ ∀ (x : W.Point), Eq (Neg.neg (Neg.neg x)) x
      -/
                         /-
                           🎉 no goals
                         -/
  ⟨by rintro (_ | _) <;> simp [zero_def]; ring1⟩
                                          /-
                                            🎉 no goals
                                          -/


open Classical in
/-- The addition of two nonsingular rational points on `W`.

Given two nonsingular rational points `P` and `Q` on `W`, use `P + Q` instead of `add P Q`. -/
noncomputable def add : W.Point → W.Point → W.Point
  | 0, P => P
  | P, 0 => P
  | @some _ _ _ x₁ y₁ h₁, @some _ _ _ x₂ y₂ h₂ =>
    if h : x₁ = x₂ ∧ y₁ = W.negY x₂ y₂ then 0
    else some (nonsingular_add h₁ h₂ fun hx hy ↦ h ⟨hx, hy⟩)


noncomputable instance instAddPoint : Add W.Point :=
  ⟨add⟩


lemma add_def (P Q : W.Point) : P.add Q = P + Q :=
  rfl


noncomputable instance instAddZeroClassPoint : AddZeroClass W.Point :=
      /-
        R : Type u
        inst✝¹ : CommRing R
        W✝ : WeierstrassCurve.Affine R
        F : Type u
        inst✝ : Field F
        W : WeierstrassCurve.Affine F
        ⊢ ∀ (a : W.Point), Eq (HAdd.hAdd 0 a) a
      -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  ⟨by rintro (_ | _) <;> rfl, by rintro (_ | _) <;> rfl⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
lemma add_of_Y_eq {x₁ x₂ y₁ y₂ : F} {h₁ : W.Nonsingular x₁ y₁} {h₂ : W.Nonsingular x₂ y₂}
    (hx : x₁ = x₂) (hy : y₁ = W.negY x₂ y₂) : some h₁ + some h₂ = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    W : WeierstrassCurve.Affine F
    x₁ x₂ y₁ y₂ : F
    h₁ : W.Nonsingular x₁ y₁
    h₂ : W.Nonsingular x₂ y₂
    hx : Eq x₁ x₂
    hy : Eq y₁ (W.negY x₂ y₂)
    ⊢ Eq (HAdd.hAdd (WeierstrassCurve.Affine.Point.some h₁) (WeierstrassCurve.Affi …
  -/
  simp_rw [← add_def, add]; exact dif_pos ⟨hx, hy⟩
                            /-
                              🎉 no goals
                            -/


@[simp]
lemma add_self_of_Y_eq {x₁ y₁ : F} {h₁ : W.Nonsingular x₁ y₁} (hy : y₁ = W.negY x₁ y₁) :
    some h₁ + some h₁ = 0 :=
  add_of_Y_eq rfl hy


@[simp]
lemma add_of_imp {x₁ x₂ y₁ y₂ : F} {h₁ : W.Nonsingular x₁ y₁} {h₂ : W.Nonsingular x₂ y₂}
    (hxy : x₁ = x₂ → y₁ ≠ W.negY x₂ y₂) : some h₁ + some h₂ = some (nonsingular_add h₁ h₂ hxy) :=
  dif_neg fun hn ↦ hxy hn.1 hn.2


@[simp]
lemma add_of_Y_ne {x₁ x₂ y₁ y₂ : F} {h₁ : W.Nonsingular x₁ y₁} {h₂ : W.Nonsingular x₂ y₂}
    (hy : y₁ ≠ W.negY x₂ y₂) :
    some h₁ + some h₂ = some (nonsingular_add h₁ h₂ fun _ ↦ hy) :=
  add_of_imp fun _ ↦ hy


lemma add_of_Y_ne' {x₁ x₂ y₁ y₂ : F} {h₁ : W.Nonsingular x₁ y₁} {h₂ : W.Nonsingular x₂ y₂}
    (hy : y₁ ≠ W.negY x₂ y₂) :
    some h₁ + some h₂ = -some (nonsingular_negAdd h₁ h₂ fun _ ↦ hy) :=
  add_of_Y_ne hy


@[simp]
lemma add_self_of_Y_ne {x₁ y₁ : F} {h₁ : W.Nonsingular x₁ y₁} (hy : y₁ ≠ W.negY x₁ y₁) :
    some h₁ + some h₁ = some (nonsingular_add h₁ h₁ fun _ => hy) :=
  add_of_Y_ne hy


lemma add_self_of_Y_ne' {x₁ y₁ : F} {h₁ : W.Nonsingular x₁ y₁} (hy : y₁ ≠ W.negY x₁ y₁) :
    some h₁ + some h₁ = -some (nonsingular_negAdd h₁ h₁ fun _ => hy) :=
  add_of_Y_ne hy


@[simp]
lemma add_of_X_ne {x₁ x₂ y₁ y₂ : F} {h₁ : W.Nonsingular x₁ y₁} {h₂ : W.Nonsingular x₂ y₂}
    (hx : x₁ ≠ x₂) : some h₁ + some h₂ = some (nonsingular_add h₁ h₂ fun h => (hx h).elim) :=
  add_of_imp fun h ↦ (hx h).elim


lemma add_of_X_ne' {x₁ x₂ y₁ y₂ : F} {h₁ : W.Nonsingular x₁ y₁} {h₂ : W.Nonsingular x₂ y₂}
    (hx : x₁ ≠ x₂) : some h₁ + some h₂ = -some (nonsingular_negAdd h₁ h₂ fun h => (hx h).elim) :=
  add_of_X_ne hx


@[deprecated (since := "2024-06-03")] alias some_add_some_of_Yeq := add_of_Y_eq

@[deprecated (since := "2024-06-03")] alias some_add_self_of_Yeq := add_self_of_Y_eq

@[deprecated (since := "2024-06-03")] alias some_add_some_of_Yne := add_of_Y_ne

@[deprecated (since := "2024-06-03")] alias some_add_some_of_Yne' := add_of_Y_ne'

@[deprecated (since := "2024-06-03")] alias some_add_self_of_Yne := add_self_of_Y_ne

@[deprecated (since := "2024-06-03")] alias some_add_self_of_Yne' := add_self_of_Y_ne'

@[deprecated (since := "2024-06-03")] alias some_add_some_of_Xne := add_of_X_ne

@[deprecated (since := "2024-06-03")] alias some_add_some_of_Xne' := add_of_X_ne'


lemma map_polynomial : (W.map f).toAffine.polynomial = W.polynomial.map (mapRingHom f) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W f).toAffine.polynomial (Polynomial.map (Polynomia …
  -/
  simp only [polynomial]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow Polynomial.X 2) (HMul.hMul (Polynomial.C …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma evalEval_baseChange_polynomial_X_Y :
    (W.baseChange R[X][Y]).toAffine.polynomial.evalEval (C X) Y = W.polynomial := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve.Affine R
    ⊢ Eq (Polynomial.evalEval (Polynomial.C Polynomial.X) Polynomial.X (Weierstras …
  -/
  rw [baseChange, toAffine, map_polynomial, evalEval, eval_map, eval_C_X_eval₂_map_C_X]
  /-
    🎉 no goals
  -/


variable {W} in
lemma Equation.map {x y : R} (h : W.Equation x y) : Equation (W.map f) (f x) (f y) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x y : R
    h : W.Equation x y
    ⊢ WeierstrassCurve.Affine.Equation (WeierstrassCurve.map W f) (f x) (f y)
  -/
  rw [Equation, map_polynomial, map_mapRingHom_evalEval, ← f.map_zero]; exact congr_arg f h
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


variable {f} in
lemma map_equation (hf : Function.Injective f) (x y : R) :
    (W.map f).toAffine.Equation (f x) (f y) ↔ W.Equation x y := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Injective ⇑f
    x y : R
    ⊢ Iff ((WeierstrassCurve.map W f).toAffine.Equation (f x) (f y)) (W.Equation x …
  -/
  simp only [Equation, map_polynomial, map_mapRingHom_evalEval, map_eq_zero_iff f hf]
  /-
    🎉 no goals
  -/


lemma map_polynomialX : (W.map f).toAffine.polynomialX = W.polynomialX.map (mapRingHom f) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W f).toAffine.polynomialX (Polynomial.map (Polynomi …
  -/
  simp only [polynomialX]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (HSub.hSub (HMul.hMul (Polynomial.C (Polynomial.C (WeierstrassCurve.map W …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_polynomialY : (W.map f).toAffine.polynomialY = W.polynomialY.map (mapRingHom f) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W f).toAffine.polynomialY (Polynomial.map (Polynomi …
  -/
  simp only [polynomialY]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (Polynomial.C 2)) Polynomial.X) (Poly …
  -/
  map_simp
  /-
    🎉 no goals
  -/


variable {f} in
lemma map_nonsingular (hf : Function.Injective f) (x y : R) :
    (W.map f).toAffine.Nonsingular (f x) (f y) ↔ W.Nonsingular x y := by
  simp only [Nonsingular, evalEval, W.map_equation hf, map_polynomialX,
    map_polynomialY, map_mapRingHom_evalEval, map_ne_zero_iff f hf]


lemma map_negPolynomial :
    (W.map f).toAffine.negPolynomial = W.negPolynomial.map (mapRingHom f) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W f).toAffine.negPolynomial (Polynomial.map (Polyno …
  -/
  simp only [negPolynomial]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (HSub.hSub (Neg.neg Polynomial.X) (Polynomial.C (HAdd.hAdd (HMul.hMul (Po …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_negY (x y : R) : (W.map f).toAffine.negY (f x) (f y) = f (W.negY x y) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x y : R
    ⊢ Eq ((WeierstrassCurve.map W f).toAffine.negY (f x) (f y)) (f (W.negY x y))
  -/
  simp only [negY]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x y : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (f y)) (HMul.hMul (WeierstrassCurve.map W  …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_linePolynomial (x y L : R) :
    linePolynomial (f x) (f y) (f L) = (linePolynomial x y L).map f := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x y L : R
    ⊢ Eq (WeierstrassCurve.Affine.linePolynomial (f x) (f y) (f L)) (Polynomial.ma …
  -/
  simp only [linePolynomial]
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x y L : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (f L)) (HSub.hSub Polynomial.X (Polyn …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_addPolynomial (x y L : R) :
    (W.map f).toAffine.addPolynomial (f x) (f y) (f L) = (W.addPolynomial x y L).map f := by
  rw [addPolynomial, map_polynomial, eval_map, linePolynomial, addPolynomial, ← coe_mapRingHom,
    ← eval₂_hom, linePolynomial]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x y L : R
    ⊢ Eq (Polynomial.eval₂ (Polynomial.mapRingHom f) (HAdd.hAdd (HMul.hMul (Polyno …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_addX (x₁ x₂ L : R) :
    (W.map f).toAffine.addX (f x₁) (f x₂) (f L) = f (W.addX x₁ x₂ L) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x₁ x₂ L : R
    ⊢ Eq ((WeierstrassCurve.map W f).toAffine.addX (f x₁) (f x₂) (f L)) (f (W.addX …
  -/
  simp only [addX]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x₁ x₂ L : R
    ⊢ Eq (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HPow.hPow (f L) 2) (HMul.hMu …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_negAddY (x₁ x₂ y₁ L : R) :
    (W.map f).toAffine.negAddY (f x₁) (f x₂) (f y₁) (f L) = f (W.negAddY x₁ x₂ y₁ L) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x₁ x₂ y₁ L : R
    ⊢ Eq ((WeierstrassCurve.map W f).toAffine.negAddY (f x₁) (f x₂) (f y₁) (f L))  …
  -/
  simp only [negAddY, map_addX]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x₁ x₂ y₁ L : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (f L) (HSub.hSub (f (W.addX x₁ x₂ L)) (f x₁))) (f y …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma map_addY (x₁ x₂ y₁ L : R) :
    (W.map f).toAffine.addY (f x₁) (f x₂) (f y₁) (f L) = f (W.toAffine.addY x₁ x₂ y₁ L) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    x₁ x₂ y₁ L : R
    ⊢ Eq ((WeierstrassCurve.map W f).toAffine.addY (f x₁) (f x₂) (f y₁) (f L)) (f  …
  -/
  simp only [addY, map_negAddY, map_addX, map_negY]
  /-
    🎉 no goals
  -/


lemma map_slope {F : Type u} [Field F] (W : Affine F) {K : Type v} [Field K] (f : F →+* K)
    (x₁ x₂ y₁ y₂ : F) : (W.map f).toAffine.slope (f x₁) (f x₂) (f y₁) (f y₂) =
      f (W.slope x₁ x₂ y₁ y₂) := by
  /-
    F : Type u
    inst✝¹ : Field F
    W : WeierstrassCurve.Affine F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    x₁ x₂ y₁ y₂ : F
    ⊢ Eq ((WeierstrassCurve.map W f).toAffine.slope (f x₁) (f x₂) (f y₁) (f y₂)) ( …
  -/
  by_cases hx : x₁ = x₂
    /-
      case pos
      F : Type u
      inst✝¹ : Field F
      W : WeierstrassCurve.Affine F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      x₁ x₂ y₁ y₂ : F
      hx : Eq x₁ x₂
      ⊢ Eq ((WeierstrassCurve.map W f).toAffine.slope (f x₁) (f x₂) (f y₁) (f y₂)) ( …
    -/
  · by_cases hy : y₁ = W.negY x₂ y₂
      /-
        case pos
        F : Type u
        inst✝¹ : Field F
        W : WeierstrassCurve.Affine F
        K : Type v
        inst✝ : Field K
        f : RingHom F K
        x₁ x₂ y₁ y₂ : F
        hx : Eq x₁ x₂
        hy : Eq y₁ (W.negY x₂ y₂)
        ⊢ Eq ((WeierstrassCurve.map W f).toAffine.slope (f x₁) (f x₂) (f y₁) (f y₂)) ( …
      -/
    · rw [slope_of_Y_eq (congr_arg f hx) <| by rw [hy, map_negY], slope_of_Y_eq hx hy, map_zero]
      /-
        🎉 no goals
      -/
    · rw [slope_of_Y_ne (congr_arg f hx) <| W.map_negY f x₂ y₂ ▸ fun h => hy <| f.injective h,
        map_negY, slope_of_Y_ne hx hy]
      /-
        case neg
        F : Type u
        inst✝¹ : Field F
        W : WeierstrassCurve.Affine F
        K : Type v
        inst✝ : Field K
        f : RingHom F K
        x₁ x₂ y₁ y₂ : F
        hx : Eq x₁ x₂
        hy : Not (Eq y₁ (W.negY x₂ y₂))
        ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow (f x₁ …
      -/
      map_simp
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type u
      inst✝¹ : Field F
      W : WeierstrassCurve.Affine F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      x₁ x₂ y₁ y₂ : F
      hx : Not (Eq x₁ x₂)
      ⊢ Eq ((WeierstrassCurve.map W f).toAffine.slope (f x₁) (f x₂) (f y₁) (f y₂)) ( …
    -/
  · rw [slope_of_X_ne fun h => hx <| f.injective h, slope_of_X_ne hx]
    /-
      case neg
      F : Type u
      inst✝¹ : Field F
      W : WeierstrassCurve.Affine F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      x₁ x₂ y₁ y₂ : F
      hx : Not (Eq x₁ x₂)
      ⊢ Eq (HDiv.hDiv (HSub.hSub (f y₁) (f y₂)) (HSub.hSub (f x₁) (f x₂))) (f (HDiv. …
    -/
    map_simp
    /-
      🎉 no goals
    -/


lemma baseChange_polynomial : (W.baseChange B).toAffine.polynomial =
    (W.baseChange A).toAffine.polynomial.map (mapRingHom f) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    ⊢ Eq (WeierstrassCurve.baseChange W B).toAffine.polynomial (Polynomial.map (Po …
  -/
  rw [← map_polynomial, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_equation (hf : Function.Injective f) (x y : A) :
    (W.baseChange B).toAffine.Equation (f x) (f y) ↔ (W.baseChange A).toAffine.Equation x y := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    hf : Function.Injective ⇑f
    x y : A
    ⊢ Iff ((WeierstrassCurve.baseChange W B).toAffine.Equation (f x) (f y)) ((Weie …
  -/
  erw [← map_equation _ hf, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    hf : Function.Injective ⇑f
    x y : A
    ⊢ Iff ((WeierstrassCurve.baseChange W B).toAffine.Equation (f x) (f y)) ((Weie …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_polynomialX : (W.baseChange B).toAffine.polynomialX =
    (W.baseChange A).toAffine.polynomialX.map (mapRingHom f) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    ⊢ Eq (WeierstrassCurve.baseChange W B).toAffine.polynomialX (Polynomial.map (P …
  -/
  rw [← map_polynomialX, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_polynomialY : (W.baseChange B).toAffine.polynomialY =
    (W.baseChange A).toAffine.polynomialY.map (mapRingHom f) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    ⊢ Eq (WeierstrassCurve.baseChange W B).toAffine.polynomialY (Polynomial.map (P …
  -/
  rw [← map_polynomialY, map_baseChange]
  /-
    🎉 no goals
  -/


variable {f} in
lemma baseChange_nonsingular (hf : Function.Injective f) (x y : A) :
    (W.baseChange B).toAffine.Nonsingular (f x) (f y) ↔
      (W.baseChange A).toAffine.Nonsingular x y := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    hf : Function.Injective ⇑f
    x y : A
    ⊢ Iff ((WeierstrassCurve.baseChange W B).toAffine.Nonsingular (f x) (f y)) ((W …
  -/
  erw [← map_nonsingular _ hf, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    hf : Function.Injective ⇑f
    x y : A
    ⊢ Iff ((WeierstrassCurve.baseChange W B).toAffine.Nonsingular (f x) (f y)) ((W …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_negPolynomial :
    (W.baseChange B).toAffine.negPolynomial =
      (W.baseChange A).toAffine.negPolynomial.map (mapRingHom f) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    ⊢ Eq (WeierstrassCurve.baseChange W B).toAffine.negPolynomial (Polynomial.map  …
  -/
  rw [← map_negPolynomial, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_negY (x y : A) :
    (W.baseChange B).toAffine.negY (f x) (f y) = f ((W.baseChange A).toAffine.negY x y) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x y : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.negY (f x) (f y)) (f ((Weiers …
  -/
  erw [← map_negY, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x y : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.negY (f x) (f y)) ((Weierstra …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_addPolynomial (x y L : A) :
    (W.baseChange B).toAffine.addPolynomial (f x) (f y) (f L) =
      ((W.baseChange A).toAffine.addPolynomial x y L).map f := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x y L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.addPolynomial (f x) (f y) (f  …
  -/
  rw [← map_addPolynomial, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x y L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.addPolynomial (f x) (f y) (f  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_addX (x₁ x₂ L : A) :
    (W.baseChange B).toAffine.addX (f x₁) (f x₂) (f L) =
      f ((W.baseChange A).toAffine.addX x₁ x₂ L) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x₁ x₂ L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.addX (f x₁) (f x₂) (f L)) (f  …
  -/
  erw [← map_addX, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x₁ x₂ L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.addX (f x₁) (f x₂) (f L)) ((W …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_negAddY (x₁ x₂ y₁ L : A) :
    (W.baseChange B).toAffine.negAddY (f x₁) (f x₂) (f y₁) (f L) =
      f ((W.baseChange A).toAffine.negAddY x₁ x₂ y₁ L) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x₁ x₂ y₁ L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.negAddY (f x₁) (f x₂) (f y₁)  …
  -/
  erw [← map_negAddY, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x₁ x₂ y₁ L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.negAddY (f x₁) (f x₂) (f y₁)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_addY (x₁ x₂ y₁ L : A) :
    (W.baseChange B).toAffine.addY (f x₁) (f x₂) (f y₁) (f L) =
      f ((W.baseChange A).toAffine.addY x₁ x₂ y₁ L) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x₁ x₂ y₁ L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.addY (f x₁) (f x₂) (f y₁) (f  …
  -/
  erw [← map_addY, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra S A
    inst✝⁴ : IsScalarTower R S A
    B : Type v
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    inst✝¹ : Algebra S B
    inst✝ : IsScalarTower R S B
    f : AlgHom S A B
    x₁ x₂ y₁ L : A
    ⊢ Eq ((WeierstrassCurve.baseChange W B).toAffine.addY (f x₁) (f x₂) (f y₁) (f  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma baseChange_slope (x₁ x₂ y₁ y₂ : F) :
    (W.baseChange K).toAffine.slope (f x₁) (f x₂) (f y₁) (f y₂) =
      f ((W.baseChange F).toAffine.slope x₁ x₂ y₁ y₂) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    F : Type u
    inst✝⁷ : Field F
    inst✝⁶ : Algebra R F
    inst✝⁵ : Algebra S F
    inst✝⁴ : IsScalarTower R S F
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : Algebra S K
    inst✝ : IsScalarTower R S K
    f : AlgHom S F K
    x₁ x₂ y₁ y₂ : F
    ⊢ Eq ((WeierstrassCurve.baseChange W K).toAffine.slope (f x₁) (f x₂) (f y₁) (f …
  -/
  erw [← map_slope, map_baseChange]
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    F : Type u
    inst✝⁷ : Field F
    inst✝⁶ : Algebra R F
    inst✝⁵ : Algebra S F
    inst✝⁴ : IsScalarTower R S F
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : Algebra S K
    inst✝ : IsScalarTower R S K
    f : AlgHom S F K
    x₁ x₂ y₁ y₂ : F
    ⊢ Eq ((WeierstrassCurve.baseChange W K).toAffine.slope (f x₁) (f x₂) (f y₁) (f …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The function from `W⟮F⟯` to `W⟮K⟯` induced by an algebra homomorphism `f : F →ₐ[S] K`,
where `W` is defined over a subring of a ring `S`, and `F` and `K` are field extensions of `S`. -/
def mapFun : W⟮F⟯ → W⟮K⟯
  | 0 => 0
  | some h => some <| (W.baseChange_nonsingular f.injective ..).mpr h


/-- The group homomorphism from `W⟮F⟯` to `W⟮K⟯` induced by an algebra homomorphism `f : F →ₐ[S] K`,
where `W` is defined over a subring of a ring `S`, and `F` and `K` are field extensions of `S`. -/
def map : W⟮F⟯ →+ W⟮K⟯ where
  toFun := mapFun W f
  map_zero' := rfl
  map_add' := by
    /-
      R✝ : Type u
      inst✝²³ : CommRing R✝
      W✝ : WeierstrassCurve.Affine R✝
      R : Type r
      inst✝²² : CommRing R
      W : WeierstrassCurve.Affine R
      S : Type s
      inst✝²¹ : CommRing S
      inst✝²⁰ : Algebra R S
      A : Type u
      inst✝¹⁹ : CommRing A
      inst✝¹⁸ : Algebra R A
      inst✝¹⁷ : Algebra S A
      inst✝¹⁶ : IsScalarTower R S A
      B : Type v
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Algebra R B
      inst✝¹³ : Algebra S B
      inst✝¹² : IsScalarTower R S B
      f✝ : AlgHom S A B
      F : Type u
      inst✝¹¹ : Field F
      inst✝¹⁰ : Algebra R F
      inst✝⁹ : Algebra S F
      inst✝⁸ : IsScalarTower R S F
      K : Type v
      inst✝⁷ : Field K
      inst✝⁶ : Algebra R K
      inst✝⁵ : Algebra S K
      inst✝⁴ : IsScalarTower R S K
      f : AlgHom S F K
      L : Type w
      inst✝³ : Field L
      inst✝² : Algebra R L
      inst✝¹ : Algebra S L
      inst✝ : IsScalarTower R S L
      g : AlgHom S K L
      ⊢ ∀ (x y : WeierstrassCurve.Affine.Point (WeierstrassCurve.baseChange W F)), E …
    -/
    rintro (_ | @⟨x₁, y₁, _⟩) (_ | @⟨x₂, y₂, _⟩)
    /-
      case zero.zero
      R✝ : Type u
      inst✝²³ : CommRing R✝
      W✝ : WeierstrassCurve.Affine R✝
      R : Type r
      inst✝²² : CommRing R
      W : WeierstrassCurve.Affine R
      S : Type s
      inst✝²¹ : CommRing S
      inst✝²⁰ : Algebra R S
      A : Type u
      inst✝¹⁹ : CommRing A
      inst✝¹⁸ : Algebra R A
      inst✝¹⁷ : Algebra S A
      inst✝¹⁶ : IsScalarTower R S A
      B : Type v
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Algebra R B
      inst✝¹³ : Algebra S B
      inst✝¹² : IsScalarTower R S B
      f✝ : AlgHom S A B
      F : Type u
      inst✝¹¹ : Field F
      inst✝¹⁰ : Algebra R F
      inst✝⁹ : Algebra S F
      inst✝⁸ : IsScalarTower R S F
      K : Type v
      inst✝⁷ : Field K
      inst✝⁶ : Algebra R K
      inst✝⁵ : Algebra S K
      inst✝⁴ : IsScalarTower R S K
      f : AlgHom S F K
      L : Type w
      inst✝³ : Field L
      inst✝² : Algebra R L
      inst✝¹ : Algebra S L
      inst✝ : IsScalarTower R S L
      g : AlgHom S K L
      ⊢ Eq ({ toFun := WeierstrassCurve.Affine.Point.mapFun W f, map_zero' := ⋯ }.to …
    -/
    any_goals rfl
    /-
      case some.some
      R✝ : Type u
      inst✝²³ : CommRing R✝
      W✝ : WeierstrassCurve.Affine R✝
      R : Type r
      inst✝²² : CommRing R
      W : WeierstrassCurve.Affine R
      S : Type s
      inst✝²¹ : CommRing S
      inst✝²⁰ : Algebra R S
      A : Type u
      inst✝¹⁹ : CommRing A
      inst✝¹⁸ : Algebra R A
      inst✝¹⁷ : Algebra S A
      inst✝¹⁶ : IsScalarTower R S A
      B : Type v
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Algebra R B
      inst✝¹³ : Algebra S B
      inst✝¹² : IsScalarTower R S B
      f✝ : AlgHom S A B
      F : Type u
      inst✝¹¹ : Field F
      inst✝¹⁰ : Algebra R F
      inst✝⁹ : Algebra S F
      inst✝⁸ : IsScalarTower R S F
      K : Type v
      inst✝⁷ : Field K
      inst✝⁶ : Algebra R K
      inst✝⁵ : Algebra S K
      inst✝⁴ : IsScalarTower R S K
      f : AlgHom S F K
      L : Type w
      inst✝³ : Field L
      inst✝² : Algebra R L
      inst✝¹ : Algebra S L
      inst✝ : IsScalarTower R S L
      g : AlgHom S K L
      x₁ y₁ : F
      h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
      x₂ y₂ : F
      h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
      ⊢ Eq ({ toFun := WeierstrassCurve.Affine.Point.mapFun W f, map_zero' := ⋯ }.to …
    -/
    have inj : Function.Injective f := f.injective
    /-
      case some.some
      R✝ : Type u
      inst✝²³ : CommRing R✝
      W✝ : WeierstrassCurve.Affine R✝
      R : Type r
      inst✝²² : CommRing R
      W : WeierstrassCurve.Affine R
      S : Type s
      inst✝²¹ : CommRing S
      inst✝²⁰ : Algebra R S
      A : Type u
      inst✝¹⁹ : CommRing A
      inst✝¹⁸ : Algebra R A
      inst✝¹⁷ : Algebra S A
      inst✝¹⁶ : IsScalarTower R S A
      B : Type v
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Algebra R B
      inst✝¹³ : Algebra S B
      inst✝¹² : IsScalarTower R S B
      f✝ : AlgHom S A B
      F : Type u
      inst✝¹¹ : Field F
      inst✝¹⁰ : Algebra R F
      inst✝⁹ : Algebra S F
      inst✝⁸ : IsScalarTower R S F
      K : Type v
      inst✝⁷ : Field K
      inst✝⁶ : Algebra R K
      inst✝⁵ : Algebra S K
      inst✝⁴ : IsScalarTower R S K
      f : AlgHom S F K
      L : Type w
      inst✝³ : Field L
      inst✝² : Algebra R L
      inst✝¹ : Algebra S L
      inst✝ : IsScalarTower R S L
      g : AlgHom S K L
      x₁ y₁ : F
      h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
      x₂ y₂ : F
      h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
      inj : Function.Injective ⇑f
      ⊢ Eq ({ toFun := WeierstrassCurve.Affine.Point.mapFun W f, map_zero' := ⋯ }.to …
    -/
    by_cases h : x₁ = x₂ ∧ y₁ = negY (W.baseChange F) x₂ y₂
      /-
        case pos
        R✝ : Type u
        inst✝²³ : CommRing R✝
        W✝ : WeierstrassCurve.Affine R✝
        R : Type r
        inst✝²² : CommRing R
        W : WeierstrassCurve.Affine R
        S : Type s
        inst✝²¹ : CommRing S
        inst✝²⁰ : Algebra R S
        A : Type u
        inst✝¹⁹ : CommRing A
        inst✝¹⁸ : Algebra R A
        inst✝¹⁷ : Algebra S A
        inst✝¹⁶ : IsScalarTower R S A
        B : Type v
        inst✝¹⁵ : CommRing B
        inst✝¹⁴ : Algebra R B
        inst✝¹³ : Algebra S B
        inst✝¹² : IsScalarTower R S B
        f✝ : AlgHom S A B
        F : Type u
        inst✝¹¹ : Field F
        inst✝¹⁰ : Algebra R F
        inst✝⁹ : Algebra S F
        inst✝⁸ : IsScalarTower R S F
        K : Type v
        inst✝⁷ : Field K
        inst✝⁶ : Algebra R K
        inst✝⁵ : Algebra S K
        inst✝⁴ : IsScalarTower R S K
        f : AlgHom S F K
        L : Type w
        inst✝³ : Field L
        inst✝² : Algebra R L
        inst✝¹ : Algebra S L
        inst✝ : IsScalarTower R S L
        g : AlgHom S K L
        x₁ y₁ : F
        h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
        x₂ y₂ : F
        h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
        inj : Function.Injective ⇑f
        h : And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve.base …
        ⊢ Eq ({ toFun := WeierstrassCurve.Affine.Point.mapFun W f, map_zero' := ⋯ }.to …
      -/
    · simp only [add_of_Y_eq h.1 h.2, mapFun]
      /-
        case pos
        R✝ : Type u
        inst✝²³ : CommRing R✝
        W✝ : WeierstrassCurve.Affine R✝
        R : Type r
        inst✝²² : CommRing R
        W : WeierstrassCurve.Affine R
        S : Type s
        inst✝²¹ : CommRing S
        inst✝²⁰ : Algebra R S
        A : Type u
        inst✝¹⁹ : CommRing A
        inst✝¹⁸ : Algebra R A
        inst✝¹⁷ : Algebra S A
        inst✝¹⁶ : IsScalarTower R S A
        B : Type v
        inst✝¹⁵ : CommRing B
        inst✝¹⁴ : Algebra R B
        inst✝¹³ : Algebra S B
        inst✝¹² : IsScalarTower R S B
        f✝ : AlgHom S A B
        F : Type u
        inst✝¹¹ : Field F
        inst✝¹⁰ : Algebra R F
        inst✝⁹ : Algebra S F
        inst✝⁸ : IsScalarTower R S F
        K : Type v
        inst✝⁷ : Field K
        inst✝⁶ : Algebra R K
        inst✝⁵ : Algebra S K
        inst✝⁴ : IsScalarTower R S K
        f : AlgHom S F K
        L : Type w
        inst✝³ : Field L
        inst✝² : Algebra R L
        inst✝¹ : Algebra S L
        inst✝ : IsScalarTower R S L
        g : AlgHom S K L
        x₁ y₁ : F
        h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
        x₂ y₂ : F
        h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
        inj : Function.Injective ⇑f
        h : And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve.base …
        ⊢ Eq 0 (HAdd.hAdd (WeierstrassCurve.Affine.Point.some ⋯) (WeierstrassCurve.Aff …
      -/
      rw [add_of_Y_eq congr(f $(h.1))]
      /-
        case pos
        R✝ : Type u
        inst✝²³ : CommRing R✝
        W✝ : WeierstrassCurve.Affine R✝
        R : Type r
        inst✝²² : CommRing R
        W : WeierstrassCurve.Affine R
        S : Type s
        inst✝²¹ : CommRing S
        inst✝²⁰ : Algebra R S
        A : Type u
        inst✝¹⁹ : CommRing A
        inst✝¹⁸ : Algebra R A
        inst✝¹⁷ : Algebra S A
        inst✝¹⁶ : IsScalarTower R S A
        B : Type v
        inst✝¹⁵ : CommRing B
        inst✝¹⁴ : Algebra R B
        inst✝¹³ : Algebra S B
        inst✝¹² : IsScalarTower R S B
        f✝ : AlgHom S A B
        F : Type u
        inst✝¹¹ : Field F
        inst✝¹⁰ : Algebra R F
        inst✝⁹ : Algebra S F
        inst✝⁸ : IsScalarTower R S F
        K : Type v
        inst✝⁷ : Field K
        inst✝⁶ : Algebra R K
        inst✝⁵ : Algebra S K
        inst✝⁴ : IsScalarTower R S K
        f : AlgHom S F K
        L : Type w
        inst✝³ : Field L
        inst✝² : Algebra R L
        inst✝¹ : Algebra S L
        inst✝ : IsScalarTower R S L
        g : AlgHom S K L
        x₁ y₁ : F
        h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
        x₂ y₂ : F
        h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
        inj : Function.Injective ⇑f
        h : And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve.base …
        ⊢ Eq (f y₁) (WeierstrassCurve.Affine.negY (WeierstrassCurve.baseChange W K) (f …
      -/
      rw [baseChange_negY, inj.eq_iff]
      /-
        case pos
        R✝ : Type u
        inst✝²³ : CommRing R✝
        W✝ : WeierstrassCurve.Affine R✝
        R : Type r
        inst✝²² : CommRing R
        W : WeierstrassCurve.Affine R
        S : Type s
        inst✝²¹ : CommRing S
        inst✝²⁰ : Algebra R S
        A : Type u
        inst✝¹⁹ : CommRing A
        inst✝¹⁸ : Algebra R A
        inst✝¹⁷ : Algebra S A
        inst✝¹⁶ : IsScalarTower R S A
        B : Type v
        inst✝¹⁵ : CommRing B
        inst✝¹⁴ : Algebra R B
        inst✝¹³ : Algebra S B
        inst✝¹² : IsScalarTower R S B
        f✝ : AlgHom S A B
        F : Type u
        inst✝¹¹ : Field F
        inst✝¹⁰ : Algebra R F
        inst✝⁹ : Algebra S F
        inst✝⁸ : IsScalarTower R S F
        K : Type v
        inst✝⁷ : Field K
        inst✝⁶ : Algebra R K
        inst✝⁵ : Algebra S K
        inst✝⁴ : IsScalarTower R S K
        f : AlgHom S F K
        L : Type w
        inst✝³ : Field L
        inst✝² : Algebra R L
        inst✝¹ : Algebra S L
        inst✝ : IsScalarTower R S L
        g : AlgHom S K L
        x₁ y₁ : F
        h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
        x₂ y₂ : F
        h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
        inj : Function.Injective ⇑f
        h : And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve.base …
        ⊢ Eq y₁ ((WeierstrassCurve.baseChange W F).toAffine.negY (Mathlib.Tactic.TermC …
      -/
      exact h.2
      /-
        🎉 no goals
      -/
      /-
        case neg
        R✝ : Type u
        inst✝²³ : CommRing R✝
        W✝ : WeierstrassCurve.Affine R✝
        R : Type r
        inst✝²² : CommRing R
        W : WeierstrassCurve.Affine R
        S : Type s
        inst✝²¹ : CommRing S
        inst✝²⁰ : Algebra R S
        A : Type u
        inst✝¹⁹ : CommRing A
        inst✝¹⁸ : Algebra R A
        inst✝¹⁷ : Algebra S A
        inst✝¹⁶ : IsScalarTower R S A
        B : Type v
        inst✝¹⁵ : CommRing B
        inst✝¹⁴ : Algebra R B
        inst✝¹³ : Algebra S B
        inst✝¹² : IsScalarTower R S B
        f✝ : AlgHom S A B
        F : Type u
        inst✝¹¹ : Field F
        inst✝¹⁰ : Algebra R F
        inst✝⁹ : Algebra S F
        inst✝⁸ : IsScalarTower R S F
        K : Type v
        inst✝⁷ : Field K
        inst✝⁶ : Algebra R K
        inst✝⁵ : Algebra S K
        inst✝⁴ : IsScalarTower R S K
        f : AlgHom S F K
        L : Type w
        inst✝³ : Field L
        inst✝² : Algebra R L
        inst✝¹ : Algebra S L
        inst✝ : IsScalarTower R S L
        g : AlgHom S K L
        x₁ y₁ : F
        h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
        x₂ y₂ : F
        h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
        inj : Function.Injective ⇑f
        h : Not (And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve …
        ⊢ Eq ({ toFun := WeierstrassCurve.Affine.Point.mapFun W f, map_zero' := ⋯ }.to …
      -/
    · simp only [add_of_imp fun hx hy ↦ h ⟨hx, hy⟩, mapFun]
      /-
        case neg
        R✝ : Type u
        inst✝²³ : CommRing R✝
        W✝ : WeierstrassCurve.Affine R✝
        R : Type r
        inst✝²² : CommRing R
        W : WeierstrassCurve.Affine R
        S : Type s
        inst✝²¹ : CommRing S
        inst✝²⁰ : Algebra R S
        A : Type u
        inst✝¹⁹ : CommRing A
        inst✝¹⁸ : Algebra R A
        inst✝¹⁷ : Algebra S A
        inst✝¹⁶ : IsScalarTower R S A
        B : Type v
        inst✝¹⁵ : CommRing B
        inst✝¹⁴ : Algebra R B
        inst✝¹³ : Algebra S B
        inst✝¹² : IsScalarTower R S B
        f✝ : AlgHom S A B
        F : Type u
        inst✝¹¹ : Field F
        inst✝¹⁰ : Algebra R F
        inst✝⁹ : Algebra S F
        inst✝⁸ : IsScalarTower R S F
        K : Type v
        inst✝⁷ : Field K
        inst✝⁶ : Algebra R K
        inst✝⁵ : Algebra S K
        inst✝⁴ : IsScalarTower R S K
        f : AlgHom S F K
        L : Type w
        inst✝³ : Field L
        inst✝² : Algebra R L
        inst✝¹ : Algebra S L
        inst✝ : IsScalarTower R S L
        g : AlgHom S K L
        x₁ y₁ : F
        h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
        x₂ y₂ : F
        h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
        inj : Function.Injective ⇑f
        h : Not (And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve …
        ⊢ Eq (WeierstrassCurve.Affine.Point.some ⋯) (HAdd.hAdd (WeierstrassCurve.Affin …
      -/
      rw [add_of_imp]
        /-
          case neg
          R✝ : Type u
          inst✝²³ : CommRing R✝
          W✝ : WeierstrassCurve.Affine R✝
          R : Type r
          inst✝²² : CommRing R
          W : WeierstrassCurve.Affine R
          S : Type s
          inst✝²¹ : CommRing S
          inst✝²⁰ : Algebra R S
          A : Type u
          inst✝¹⁹ : CommRing A
          inst✝¹⁸ : Algebra R A
          inst✝¹⁷ : Algebra S A
          inst✝¹⁶ : IsScalarTower R S A
          B : Type v
          inst✝¹⁵ : CommRing B
          inst✝¹⁴ : Algebra R B
          inst✝¹³ : Algebra S B
          inst✝¹² : IsScalarTower R S B
          f✝ : AlgHom S A B
          F : Type u
          inst✝¹¹ : Field F
          inst✝¹⁰ : Algebra R F
          inst✝⁹ : Algebra S F
          inst✝⁸ : IsScalarTower R S F
          K : Type v
          inst✝⁷ : Field K
          inst✝⁶ : Algebra R K
          inst✝⁵ : Algebra S K
          inst✝⁴ : IsScalarTower R S K
          f : AlgHom S F K
          L : Type w
          inst✝³ : Field L
          inst✝² : Algebra R L
          inst✝¹ : Algebra S L
          inst✝ : IsScalarTower R S L
          g : AlgHom S K L
          x₁ y₁ : F
          h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
          x₂ y₂ : F
          h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
          inj : Function.Injective ⇑f
          h : Not (And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve …
          ⊢ Eq (WeierstrassCurve.Affine.Point.some ⋯) (WeierstrassCurve.Affine.Point.som …
        -/
      · simp only [some.injEq, ← baseChange_addX, ← baseChange_addY, ← baseChange_slope]
        /-
          🎉 no goals
        -/
        /-
          case neg
          R✝ : Type u
          inst✝²³ : CommRing R✝
          W✝ : WeierstrassCurve.Affine R✝
          R : Type r
          inst✝²² : CommRing R
          W : WeierstrassCurve.Affine R
          S : Type s
          inst✝²¹ : CommRing S
          inst✝²⁰ : Algebra R S
          A : Type u
          inst✝¹⁹ : CommRing A
          inst✝¹⁸ : Algebra R A
          inst✝¹⁷ : Algebra S A
          inst✝¹⁶ : IsScalarTower R S A
          B : Type v
          inst✝¹⁵ : CommRing B
          inst✝¹⁴ : Algebra R B
          inst✝¹³ : Algebra S B
          inst✝¹² : IsScalarTower R S B
          f✝ : AlgHom S A B
          F : Type u
          inst✝¹¹ : Field F
          inst✝¹⁰ : Algebra R F
          inst✝⁹ : Algebra S F
          inst✝⁸ : IsScalarTower R S F
          K : Type v
          inst✝⁷ : Field K
          inst✝⁶ : Algebra R K
          inst✝⁵ : Algebra S K
          inst✝⁴ : IsScalarTower R S K
          f : AlgHom S F K
          L : Type w
          inst✝³ : Field L
          inst✝² : Algebra R L
          inst✝¹ : Algebra S L
          inst✝ : IsScalarTower R S L
          g : AlgHom S K L
          x₁ y₁ : F
          h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₁ …
          x₂ y₂ : F
          h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x₂ y₂
          inj : Function.Injective ⇑f
          h : Not (And (Eq x₁ x₂) (Eq y₁ (WeierstrassCurve.Affine.negY (WeierstrassCurve …
          ⊢ Eq (f x₁) (f x₂) → Ne (f y₁) (WeierstrassCurve.Affine.negY (WeierstrassCurve …
        -/
      · push_neg at h; rwa [baseChange_negY, inj.eq_iff, inj.ne_iff]
                       /-
                         🎉 no goals
                       -/


lemma map_zero : map W f (0 : W⟮F⟯) = 0 :=
  rfl


lemma map_some {x y : F} (h : (W.baseChange F).toAffine.Nonsingular x y) :
    map W f (some h) = some ((W.baseChange_nonsingular f.injective ..).mpr h) :=
  rfl


lemma map_id (P : W⟮F⟯) : map W (Algebra.ofId F F) P = P := by
  /-
    R : Type r
    inst✝² : CommRing R
    W : WeierstrassCurve.Affine R
    F : Type u
    inst✝¹ : Field F
    inst✝ : Algebra R F
    P : WeierstrassCurve.Affine.Point (WeierstrassCurve.baseChange W F)
    ⊢ Eq ((WeierstrassCurve.Affine.Point.map W (Algebra.ofId F F)) P) P
  -/
              /-
                🎉 no goals
              -/
  cases P <;> rfl
              /-
                🎉 no goals
              -/


lemma map_map (P : W⟮F⟯) : map W g (map W f P) = map W (g.comp f) P := by
  /-
    R : Type r
    inst✝¹⁴ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    F : Type u
    inst✝¹¹ : Field F
    inst✝¹⁰ : Algebra R F
    inst✝⁹ : Algebra S F
    inst✝⁸ : IsScalarTower R S F
    K : Type v
    inst✝⁷ : Field K
    inst✝⁶ : Algebra R K
    inst✝⁵ : Algebra S K
    inst✝⁴ : IsScalarTower R S K
    f : AlgHom S F K
    L : Type w
    inst✝³ : Field L
    inst✝² : Algebra R L
    inst✝¹ : Algebra S L
    inst✝ : IsScalarTower R S L
    g : AlgHom S K L
    P : WeierstrassCurve.Affine.Point (WeierstrassCurve.baseChange W F)
    ⊢ Eq ((WeierstrassCurve.Affine.Point.map W g) ((WeierstrassCurve.Affine.Point. …
  -/
              /-
                🎉 no goals
              -/
  cases P <;> rfl
              /-
                🎉 no goals
              -/


lemma map_injective : Function.Injective <| map W f := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    F : Type u
    inst✝⁷ : Field F
    inst✝⁶ : Algebra R F
    inst✝⁵ : Algebra S F
    inst✝⁴ : IsScalarTower R S F
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : Algebra S K
    inst✝ : IsScalarTower R S K
    f : AlgHom S F K
    ⊢ Function.Injective ⇑(WeierstrassCurve.Affine.Point.map W f)
  -/
  rintro (_ | _) (_ | _) h
  /-
    case zero.zero
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    S : Type s
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    F : Type u
    inst✝⁷ : Field F
    inst✝⁶ : Algebra R F
    inst✝⁵ : Algebra S F
    inst✝⁴ : IsScalarTower R S F
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : Algebra S K
    inst✝ : IsScalarTower R S K
    f : AlgHom S F K
    h : Eq ((WeierstrassCurve.Affine.Point.map W f) WeierstrassCurve.Affine.Point. …
    ⊢ Eq WeierstrassCurve.Affine.Point.zero WeierstrassCurve.Affine.Point.zero
  -/
  any_goals contradiction
    /-
      case zero.zero
      R : Type r
      inst✝¹⁰ : CommRing R
      W : WeierstrassCurve.Affine R
      S : Type s
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      F : Type u
      inst✝⁷ : Field F
      inst✝⁶ : Algebra R F
      inst✝⁵ : Algebra S F
      inst✝⁴ : IsScalarTower R S F
      K : Type v
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : Algebra S K
      inst✝ : IsScalarTower R S K
      f : AlgHom S F K
      h : Eq ((WeierstrassCurve.Affine.Point.map W f) WeierstrassCurve.Affine.Point. …
      ⊢ Eq WeierstrassCurve.Affine.Point.zero WeierstrassCurve.Affine.Point.zero
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case some.some
      R : Type r
      inst✝¹⁰ : CommRing R
      W : WeierstrassCurve.Affine R
      S : Type s
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      F : Type u
      inst✝⁷ : Field F
      inst✝⁶ : Algebra R F
      inst✝⁵ : Algebra S F
      inst✝⁴ : IsScalarTower R S F
      K : Type v
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : Algebra S K
      inst✝ : IsScalarTower R S K
      f : AlgHom S F K
      x✝¹ y✝¹ : F
      h✝¹ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x✝ …
      x✝ y✝ : F
      h✝ : WeierstrassCurve.Affine.Nonsingular (WeierstrassCurve.baseChange W F) x✝ y✝
      h : Eq ((WeierstrassCurve.Affine.Point.map W f) (WeierstrassCurve.Affine.Point …
      ⊢ Eq (WeierstrassCurve.Affine.Point.some h✝¹) (WeierstrassCurve.Affine.Point.s …
    -/
  · simpa only [some.injEq] using ⟨f.injective (some.inj h).left, f.injective (some.inj h).right⟩
    /-
      🎉 no goals
    -/


variable (F K) in
/-- The group homomorphism from `W⟮F⟯` to `W⟮K⟯` induced by the base change from `F` to `K`,
where `W` is defined over a subring of a ring `S`, and `F` and `K` are field extensions of `S`. -/
abbrev baseChange [Algebra F K] [IsScalarTower R F K] : W⟮F⟯ →+ W⟮K⟯ :=
  map W <| Algebra.ofId F K


lemma map_baseChange [Algebra F K] [IsScalarTower R F K] [Algebra F L] [IsScalarTower R F L]
    (f : K →ₐ[F] L) (P : W⟮F⟯) : map W f (baseChange W F K P) = baseChange W F L P := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    F : Type u
    inst✝⁹ : Field F
    inst✝⁸ : Algebra R F
    K : Type v
    inst✝⁷ : Field K
    inst✝⁶ : Algebra R K
    L : Type w
    inst✝⁵ : Field L
    inst✝⁴ : Algebra R L
    inst✝³ : Algebra F K
    inst✝² : IsScalarTower R F K
    inst✝¹ : Algebra F L
    inst✝ : IsScalarTower R F L
    f : AlgHom F K L
    P : WeierstrassCurve.Affine.Point (WeierstrassCurve.baseChange W F)
    ⊢ Eq ((WeierstrassCurve.Affine.Point.map W f) ((WeierstrassCurve.Affine.Point. …
  -/
  have : Subsingleton (F →ₐ[F] L) := inferInstance
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W : WeierstrassCurve.Affine R
    F : Type u
    inst✝⁹ : Field F
    inst✝⁸ : Algebra R F
    K : Type v
    inst✝⁷ : Field K
    inst✝⁶ : Algebra R K
    L : Type w
    inst✝⁵ : Field L
    inst✝⁴ : Algebra R L
    inst✝³ : Algebra F K
    inst✝² : IsScalarTower R F K
    inst✝¹ : Algebra F L
    inst✝ : IsScalarTower R F L
    f : AlgHom F K L
    P : WeierstrassCurve.Affine.Point (WeierstrassCurve.baseChange W F)
    this : Subsingleton (AlgHom F F L)
    ⊢ Eq ((WeierstrassCurve.Affine.Point.map W f) ((WeierstrassCurve.Affine.Point. …
  -/
  convert map_map W (Algebra.ofId F K) f P
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-03")] alias addY' := negAddY

@[deprecated (since := "2024-06-03")] alias
  nonsingular_add_of_eval_derivative_ne_zero := nonsingular_negAdd_of_eval_derivative_ne_zero

@[deprecated (since := "2024-06-03")] alias slope_of_Yeq := slope_of_Y_eq

@[deprecated (since := "2024-06-03")] alias slope_of_Yne := slope_of_Y_ne

@[deprecated (since := "2024-06-03")] alias slope_of_Xne := slope_of_X_ne

@[deprecated (since := "2024-06-03")] alias slope_of_Yne_eq_eval := slope_of_Y_ne_eq_eval

@[deprecated (since := "2024-06-03")] alias Yeq_of_Xeq := Y_eq_of_X_eq

@[deprecated (since := "2024-06-03")] alias Yeq_of_Yne := Y_eq_of_Y_ne

@[deprecated (since := "2024-06-03")] alias equation_add' := equation_negAdd

@[deprecated (since := "2024-06-03")] alias nonsingular_add' := nonsingular_negAdd

@[deprecated (since := "2024-06-03")] alias baseChange_addY' := baseChange_negAddY

@[deprecated (since := "2024-06-03")] alias map_addY' := map_negAddY


lemma nonsingular [Nontrivial R] {x y : R} (h : E.toAffine.Equation x y) :
    E.toAffine.Nonsingular x y :=
  E.toAffine.nonsingular_of_Δ_ne_zero h <| E.coe_Δ' ▸ E.Δ'.ne_zero


