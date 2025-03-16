local notation3 "x" => (0 : Fin 3)


local notation3 "y" => (1 : Fin 3)


local notation3 "z" => (2 : Fin 3)


local macro "matrix_simp" : tactic =>
  `(tactic| simp only [Matrix.head_cons, Matrix.tail_cons, Matrix.smul_empty, Matrix.smul_cons,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two])


/-- An abbreviation for a Weierstrass curve in Jacobian coordinates. -/
abbrev WeierstrassCurve.Jacobian (R : Type u) : Type u :=
  WeierstrassCurve R


/-- The coercion to a Weierstrass curve in Jacobian coordinates. -/
abbrev WeierstrassCurve.toJacobian {R : Type u} (W : WeierstrassCurve R) : Jacobian R :=
  W


local macro "eval_simp" : tactic =>
  `(tactic| simp only [eval_C, eval_X, eval_add, eval_sub, eval_mul, eval_pow])


local macro "pderiv_simp" : tactic =>
  `(tactic| simp only [map_ofNat, map_neg, map_add, map_sub, map_mul, pderiv_mul, pderiv_pow,
    pderiv_C, pderiv_X_self, pderiv_X_of_ne one_ne_zero, pderiv_X_of_ne one_ne_zero.symm,
    pderiv_X_of_ne (by decide : z ≠ x), pderiv_X_of_ne (by decide : x ≠ z),
    pderiv_X_of_ne (by decide : z ≠ y), pderiv_X_of_ne (by decide : y ≠ z)])


lemma fin3_def (P : Fin 3 → R) : ![P x, P y, P z] = P := by
  /-
    R : Type u
    P : Fin 3 → R
    ⊢ Eq (Matrix.vecCons (P 0) (Matrix.vecCons (P 1) (Matrix.vecCons (P 2) Matrix. …
  -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
  ext n; fin_cases n <;> rfl
                         /-
                           🎉 no goals
                         -/


lemma fin3_def_ext (X Y Z : R) : ![X, Y, Z] x = X ∧ ![X, Y, Z] y = Y ∧ ![X, Y, Z] z = Z :=
  ⟨rfl, rfl, rfl⟩


lemma comp_fin3 {S} (f : R → S) (X Y Z : R) : f ∘ ![X, Y, Z] = ![f X, f Y, f Z] :=
  (FinVec.map_eq _ _).symm


/-- The scalar multiplication on a point representative. -/
scoped instance instSMulPoint : SMul R <| Fin 3 → R :=
  ⟨fun u P => ![u ^ 2 * P x, u ^ 3 * P y, u * P z]⟩


lemma smul_fin3 (P : Fin 3 → R) (u : R) : u • P = ![u ^ 2 * P x, u ^ 3 * P y, u * P z] :=
  rfl


lemma smul_fin3_ext (P : Fin 3 → R) (u : R) :
    (u • P) x = u ^ 2 * P x ∧ (u • P) y = u ^ 3 * P y ∧ (u • P) z = u * P z :=
  ⟨rfl, rfl, rfl⟩


/-- The multiplicative action on a point representative. -/
scoped instance instMulActionPoint : MulAction R <| Fin 3 → R where
                   /-
                     R : Type u
                     W' : WeierstrassCurve.Jacobian R
                     F : Type v
                     inst✝¹ : Field F
                     W : WeierstrassCurve.Jacobian F
                     inst✝ : CommRing R
                     x✝ : Fin 3 → R
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by simp_rw [smul_fin3, one_pow, one_mul, fin3_def]
                   /-
                     🎉 no goals
                   -/
                       /-
                         R : Type u
                         W' : WeierstrassCurve.Jacobian R
                         F : Type v
                         inst✝¹ : Field F
                         W : WeierstrassCurve.Jacobian F
                         inst✝ : CommRing R
                         x✝² x✝¹ : R
                         x✝ : Fin 3 → R
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  mul_smul _ _ _ := by simp_rw [smul_fin3, mul_pow, mul_assoc, fin3_def_ext]
                       /-
                         🎉 no goals
                       -/


/-- The equivalence setoid for a point representative. -/
scoped instance instSetoidPoint : Setoid <| Fin 3 → R :=
  MulAction.orbitRel Rˣ <| Fin 3 → R


variable (R) in
/-- The equivalence class of a point representative. -/
abbrev PointClass : Type u :=
  MulAction.orbitRel.Quotient Rˣ <| Fin 3 → R


lemma smul_equiv (P : Fin 3 → R) {u : R} (hu : IsUnit u) : u • P ≈ P :=
  ⟨hu.unit, rfl⟩


@[simp]
lemma smul_eq (P : Fin 3 → R) {u : R} (hu : IsUnit u) : (⟦u • P⟧ : PointClass R) = ⟦P⟧ :=
  Quotient.eq.mpr <| smul_equiv P hu


variable (W') in
/-- The coercion to a Weierstrass curve in affine coordinates. -/
abbrev toAffine : Affine R :=
  W'


lemma equiv_iff_eq_of_Z_eq' {P Q : Fin 3 → R} (hz : P z = Q z) (mem : Q z ∈ nonZeroDivisors R) :
    P ≈ Q ↔ P = Q := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hz : Eq (P 2) (Q 2)
    mem : Membership.mem (nonZeroDivisors R) (Q 2)
    ⊢ Iff (HasEquiv.Equiv P Q) (Eq P Q)
  -/
  refine ⟨?_, by rintro rfl; exact Setoid.refl _⟩
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hz : Eq (P 2) (Q 2)
    mem : Membership.mem (nonZeroDivisors R) (Q 2)
    ⊢ HasEquiv.Equiv P Q → Eq P Q
  -/
  rintro ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    mem : Membership.mem (nonZeroDivisors R) (Q 2)
    u : Units R
    hz : Eq ((fun m => HSMul.hSMul m Q) u 2) (Q 2)
    ⊢ Eq ((fun m => HSMul.hSMul m Q) u) Q
  -/
  rw [← one_mul (Q z)] at hz
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    mem : Membership.mem (nonZeroDivisors R) (Q 2)
    u : Units R
    hz : Eq ((fun m => HSMul.hSMul m Q) u 2) (HMul.hMul 1 (Q 2))
    ⊢ Eq ((fun m => HSMul.hSMul m Q) u) Q
  -/
  simp_rw [Units.smul_def, (mul_cancel_right_mem_nonZeroDivisors mem).mp hz, one_smul]
  /-
    🎉 no goals
  -/


lemma equiv_iff_eq_of_Z_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hz : P z = Q z) (hQz : Q z ≠ 0) :
    P ≈ Q ↔ P = Q :=
  equiv_iff_eq_of_Z_eq' hz (mem_nonZeroDivisors_of_ne_zero hQz)


lemma Z_eq_zero_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : P z = 0 ↔ Q z = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Iff (Eq (P 2) 0) (Eq (Q 2) 0)
  -/
  rcases h with ⟨_, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    w✝ : Units R
    ⊢ Iff (Eq ((fun m => HSMul.hSMul m Q) w✝ 2) 0) (Eq (Q 2) 0)
  -/
  simp only [Units.smul_def, smul_fin3_ext, Units.mul_right_eq_zero]
  /-
    🎉 no goals
  -/


lemma X_eq_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : P x * Q z ^ 2 = Q x * P z ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2) 2))
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul ((fun m => HSMul.hSMul m Q) u 0) (HPow.hPow (Q 2) 2)) (HMul.hM …
  -/
  simp only [Units.smul_def, smul_fin3_ext]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑u) 2) (Q 0)) (HPow.hPow (Q 2) 2)) (HMu …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma Y_eq_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : P y * Q z ^ 3 = Q y * P z ^ 3 := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2) 3))
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul ((fun m => HSMul.hSMul m Q) u 1) (HPow.hPow (Q 2) 3)) (HMul.hM …
  -/
  simp only [Units.smul_def, smul_fin3_ext]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑u) 3) (Q 1)) (HPow.hPow (Q 2) 3)) (HMu …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma not_equiv_of_Z_eq_zero_left {P Q : Fin 3 → R} (hPz : P z = 0) (hQz : Q z ≠ 0) : ¬P ≈ Q :=
  fun h => hQz <| (Z_eq_zero_of_equiv h).mp hPz


lemma not_equiv_of_Z_eq_zero_right {P Q : Fin 3 → R} (hPz : P z ≠ 0) (hQz : Q z = 0) : ¬P ≈ Q :=
  fun h => hPz <| (Z_eq_zero_of_equiv h).mpr hQz


lemma not_equiv_of_X_ne {P Q : Fin 3 → R} (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) : ¬P ≈ Q :=
  hx.comp X_eq_of_equiv


lemma not_equiv_of_Y_ne {P Q : Fin 3 → R} (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) : ¬P ≈ Q :=
  hy.comp Y_eq_of_equiv


lemma equiv_of_X_eq_of_Y_eq {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3) : P ≈ Q := by
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    ⊢ HasEquiv.Equiv P Q
  -/
  use Units.mk0 _ hPz / Units.mk0 _ hQz
  simp only [Units.smul_def, smul_fin3, Units.val_div_eq_div_val, Units.val_mk0, div_pow, mul_comm,
    mul_div, ← hx, ← hy, mul_div_cancel_right₀ _ <| pow_ne_zero _ hQz, mul_div_cancel_right₀ _ hQz,
    fin3_def]


lemma equiv_some_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    P ≈ ![P x / P z ^ 2, P y / P z ^ 3, 1] :=
  equiv_of_X_eq_of_Y_eq hPz one_ne_zero
        /-
          F : Type v
          inst✝ : Field F
          P : Fin 3 → F
          hPz : Ne (P 2) 0
          ⊢ Eq (HMul.hMul (P 0) (HPow.hPow (Matrix.vecCons (HDiv.hDiv (P 0) (HPow.hPow ( …
        -/
    (by linear_combination (norm := (matrix_simp; ring1)) -P x * div_self (pow_ne_zero 2 hPz))
        /-
          🎉 no goals
        -/
        /-
          F : Type v
          inst✝ : Field F
          P : Fin 3 → F
          hPz : Ne (P 2) 0
          ⊢ Eq (HMul.hMul (P 1) (HPow.hPow (Matrix.vecCons (HDiv.hDiv (P 0) (HPow.hPow ( …
        -/
    (by linear_combination (norm := (matrix_simp; ring1)) -P y * div_self (pow_ne_zero 3 hPz))
        /-
          🎉 no goals
        -/


lemma X_eq_iff {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) :
    P x * Q z ^ 2 = Q x * P z ^ 2 ↔ P x / P z ^ 2 = Q x / Q z ^ 2 :=
  (div_eq_div_iff (pow_ne_zero 2 hPz) (pow_ne_zero 2 hQz)).symm


lemma Y_eq_iff {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) :
    P y * Q z ^ 3 = Q y * P z ^ 3 ↔ P y / P z ^ 3 = Q y / Q z ^ 3 :=
  (div_eq_div_iff (pow_ne_zero 3 hPz) (pow_ne_zero 3 hQz)).symm


variable (W') in
/-- The polynomial $W(X, Y, Z) := Y^2 + a_1XYZ + a_3YZ^3 - (X^3 + a_2X^2Z^2 + a_4XZ^4 + a_6Z^6)$
associated to a Weierstrass curve `W'` over `R`. This is represented as a term of type
`MvPolynomial (Fin 3) R`, where `X 0`, `X 1`, and `X 2` represent $X$, $Y$, and $Z$ respectively. -/
noncomputable def polynomial : MvPolynomial (Fin 3) R :=
  X 1 ^ 2 + C W'.a₁ * X 0 * X 1 * X 2 + C W'.a₃ * X 1 * X 2 ^ 3
    - (X 0 ^ 3 + C W'.a₂ * X 0 ^ 2 * X 2 ^ 2 + C W'.a₄ * X 0 * X 2 ^ 4 + C W'.a₆ * X 2 ^ 6)


lemma eval_polynomial (P : Fin 3 → R) : eval P W'.polynomial =
    P y ^ 2 + W'.a₁ * P x * P y * P z + W'.a₃ * P y * P z ^ 3
      - (P x ^ 3 + W'.a₂ * P x ^ 2 * P z ^ 2 + W'.a₄ * P x * P z ^ 4 + W'.a₆ * P z ^ 6) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomial) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (H …
  -/
  rw [polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (MvPol …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma eval_polynomial_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) : eval P W.polynomial / P z ^ 6 =
    W.toAffine.polynomial.evalEval (P x / P z ^ 2) (P y / P z ^ 3) := by
  linear_combination (norm := (rw [eval_polynomial, Affine.evalEval_polynomial]; ring1))
    W.a₁ * P x * P y / P z ^ 5 * div_self hPz + W.a₃ * P y / P z ^ 3 * div_self (pow_ne_zero 3 hPz)
      - W.a₂ * P x ^ 2 / P z ^ 4 * div_self (pow_ne_zero 2 hPz)
      - W.a₄ * P x / P z ^ 2 * div_self (pow_ne_zero 4 hPz) - W.a₆ * div_self (pow_ne_zero 6 hPz)


variable (W') in
/-- The proposition that a point representative $(x, y, z)$ lies in `W'`.
In other words, $W(x, y, z) = 0$. -/
def Equation (P : Fin 3 → R) : Prop :=
  eval P W'.polynomial = 0


lemma equation_iff (P : Fin 3 → R) : W'.Equation P ↔
    P y ^ 2 + W'.a₁ * P x * P y * P z + W'.a₃ * P y * P z ^ 3
      - (P x ^ 3 + W'.a₂ * P x ^ 2 * P z ^ 2 + W'.a₄ * P x * P z ^ 4 + W'.a₆ * P z ^ 6) = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Iff (W'.Equation P) (Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (P 1) 2) …
  -/
  rw [Equation, eval_polynomial]
  /-
    🎉 no goals
  -/


lemma equation_smul (P : Fin 3 → R) {u : R} (hu : IsUnit u) : W'.Equation (u • P) ↔ W'.Equation P :=
  have (u : R) {P : Fin 3 → R} (hP : W'.Equation P) : W'.Equation <| u • P := by
    /-
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu : IsUnit u✝
      u : R
      P : Fin 3 → R
      hP : W'.Equation P
      ⊢ W'.Equation (HSMul.hSMul u P)
    -/
    rw [equation_iff] at hP ⊢
    /-
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu : IsUnit u✝
      u : R
      P : Fin 3 → R
      hP : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (P 1) 2) (HMul.hMul (HMul. …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (HSMul.hSMul u P 1) 2) (HMul. …
    -/
    linear_combination (norm := (simp only [smul_fin3_ext]; ring1)) u ^ 6 * hP
    /-
      🎉 no goals
    -/
               /-
                 R : Type u
                 W' : WeierstrassCurve.Jacobian R
                 inst✝ : CommRing R
                 P : Fin 3 → R
                 u : R
                 hu : IsUnit u
                 this : ∀ (u : R) {P : Fin 3 → R}, W'.Equation P → W'.Equation (HSMul.hSMul u P)
                 h : W'.Equation (HSMul.hSMul u P)
                 ⊢ W'.Equation P
               -/
  ⟨fun h => by convert this hu.unit.inv h; erw [smul_smul, hu.val_inv_mul, one_smul], this u⟩
                                           /-
                                             🎉 no goals
                                           -/


lemma equation_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.Equation P ↔ W'.Equation Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Iff (W'.Equation P) (W'.Equation Q)
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Iff (W'.Equation ((fun m => HSMul.hSMul m Q) u)) (W'.Equation Q)
  -/
  exact equation_smul Q u.isUnit
  /-
    🎉 no goals
  -/


lemma equation_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) :
    W'.Equation P ↔ P y ^ 2 = P x ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Iff (W'.Equation P) (Eq (HPow.hPow (P 1) 2) (HPow.hPow (P 0) 3))
  -/
  simp only [equation_iff, hPz, add_zero, mul_zero, zero_pow <| OfNat.ofNat_ne_zero _, sub_eq_zero]
  /-
    🎉 no goals
  -/


lemma equation_zero : W'.Equation ![1, 1, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ W'.Equation (Matrix.vecCons 1 (Matrix.vecCons 1 (Matrix.vecCons 0 Matrix.vec …
  -/
  simp only [equation_of_Z_eq_zero, fin3_def_ext, one_pow]
  /-
    🎉 no goals
  -/


lemma equation_some (X Y : R) : W'.Equation ![X, Y, 1] ↔ W'.toAffine.Equation X Y := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    X Y : R
    ⊢ Iff (W'.Equation (Matrix.vecCons X (Matrix.vecCons Y (Matrix.vecCons 1 Matri …
  -/
  simp only [equation_iff, Affine.equation_iff', fin3_def_ext, one_pow, mul_one]
  /-
    🎉 no goals
  -/


lemma equation_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.Equation P ↔ W.toAffine.Equation (P x / P z ^ 2) (P y / P z ^ 3) :=
  (equation_of_equiv <| equiv_some_of_Z_ne_zero hPz).trans <| equation_some ..


variable (W') in
/-- The partial derivative $W_X(X, Y, Z)$ of $W(X, Y, Z)$ with respect to $X$. -/
noncomputable def polynomialX : MvPolynomial (Fin 3) R :=
  pderiv x W'.polynomial


lemma polynomialX_eq : W'.polynomialX =
    C W'.a₁ * X 1 * X 2 - (C 3 * X 0 ^ 2 + C (2 * W'.a₂) * X 0 * X 2 ^ 2 + C W'.a₄ * X 2 ^ 4) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq W'.polynomialX (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C W'.a₁) (M …
  -/
  rw [polynomialX, polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq ((MvPolynomial.pderiv 0) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (MvP …
  -/
  pderiv_simp
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (↑2) (HPow.hPow (M …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_polynomialX (P : Fin 3 → R) : eval P W'.polynomialX =
    W'.a₁ * P y * P z - (3 * P x ^ 2 + 2 * W'.a₂ * P x * P z ^ 2 + W'.a₄ * P z ^ 4) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomialX) (HSub.hSub (HMul.hMul (HMul.hMul W …
  -/
  rw [polynomialX_eq]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C W …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma eval_polynomialX_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    eval P W.polynomialX / P z ^ 4 =
      W.toAffine.polynomialX.evalEval (P x / P z ^ 2) (P y / P z ^ 3) := by
  linear_combination (norm := (rw [eval_polynomialX, Affine.evalEval_polynomialX]; ring1))
    W.a₁ * P y / P z ^ 3 * div_self hPz - 2 * W.a₂ * P x / P z ^ 2 * div_self (pow_ne_zero 2 hPz)
      - W.a₄ * div_self (pow_ne_zero 4 hPz)


variable (W') in
/-- The partial derivative $W_Y(X, Y, Z)$ of $W(X, Y, Z)$ with respect to $Y$. -/
noncomputable def polynomialY : MvPolynomial (Fin 3) R :=
  pderiv y W'.polynomial


lemma polynomialY_eq : W'.polynomialY = C 2 * X 1 + C W'.a₁ * X 0 * X 2 + C W'.a₃ * X 2 ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq W'.polynomialY (HAdd.hAdd (HAdd.hAdd (HMul.hMul (MvPolynomial.C 2) (MvPol …
  -/
  rw [polynomialY, polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq ((MvPolynomial.pderiv 1) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (MvP …
  -/
  pderiv_simp
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (↑2) (HPow.hPow (M …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_polynomialY (P : Fin 3 → R) :
    eval P W'.polynomialY = 2 * P y + W'.a₁ * P x * P z + W'.a₃ * P z ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomialY) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 …
  -/
  rw [polynomialY_eq]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (MvPolynomial.C 2 …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma eval_polynomialY_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    eval P W.polynomialY / P z ^ 3 =
      W.toAffine.polynomialY.evalEval (P x / P z ^ 2) (P y / P z ^ 3) := by
  linear_combination (norm := (rw [eval_polynomialY, Affine.evalEval_polynomialY]; ring1))
    W.a₁ * P x / P z ^ 2 * div_self hPz + W.a₃ * div_self (pow_ne_zero 3 hPz)


variable (W') in
/-- The partial derivative $W_Z(X, Y, Z)$ of $W(X, Y, Z)$ with respect to $Z$. -/
noncomputable def polynomialZ : MvPolynomial (Fin 3) R :=
  pderiv z W'.polynomial


lemma polynomialZ_eq : W'.polynomialZ = C W'.a₁ * X 0 * X 1 + C (3 * W'.a₃) * X 1 * X 2 ^ 2 -
    (C (2 * W'.a₂) * X 0 ^ 2 * X 2 + C (4 * W'.a₄) * X 0 * X 2 ^ 3 + C (6 * W'.a₆) * X 2 ^ 5) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq W'.polynomialZ (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul (MvPolynomial. …
  -/
  rw [polynomialZ, polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq ((MvPolynomial.pderiv 2) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (MvP …
  -/
  pderiv_simp
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (↑2) (HPow.hPow (M …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_polynomialZ (P : Fin 3 → R) : eval P W'.polynomialZ =
    W'.a₁ * P x * P y + 3 * W'.a₃ * P y * P z ^ 2 -
      (2 * W'.a₂ * P x ^ 2 * P z + 4 * W'.a₄ * P x * P z ^ 3 + 6 * W'.a₆ * P z ^ 5) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomialZ) (HSub.hSub (HAdd.hAdd (HMul.hMul ( …
  -/
  rw [polynomialZ_eq]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul (MvPol …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


variable (W') in
/-- The proposition that a point representative $(x, y, z)$ in `W'` is nonsingular.
In other words, either $W_X(x, y, z) \ne 0$, $W_Y(x, y, z) \ne 0$, or $W_Z(x, y, z) \ne 0$.

Note that this definition is only mathematically accurate for fields. -/
-- TODO: generalise this definition to be mathematically accurate for a larger class of rings.
def Nonsingular (P : Fin 3 → R) : Prop :=
  W'.Equation P ∧
    (eval P W'.polynomialX ≠ 0 ∨ eval P W'.polynomialY ≠ 0 ∨ eval P W'.polynomialZ ≠ 0)


lemma nonsingular_iff (P : Fin 3 → R) : W'.Nonsingular P ↔ W'.Equation P ∧
    (W'.a₁ * P y * P z - (3 * P x ^ 2 + 2 * W'.a₂ * P x * P z ^ 2 + W'.a₄ * P z ^ 4) ≠ 0 ∨
      2 * P y + W'.a₁ * P x * P z + W'.a₃ * P z ^ 3 ≠ 0 ∨
      W'.a₁ * P x * P y + 3 * W'.a₃ * P y * P z ^ 2
        - (2 * W'.a₂ * P x ^ 2 * P z + 4 * W'.a₄ * P x * P z ^ 3 + 6 * W'.a₆ * P z ^ 5) ≠ 0) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Iff (W'.Nonsingular P) (And (W'.Equation P) (Or (Ne (HSub.hSub (HMul.hMul (H …
  -/
  rw [Nonsingular, eval_polynomialX, eval_polynomialY, eval_polynomialZ]
  /-
    🎉 no goals
  -/


lemma nonsingular_smul (P : Fin 3 → R) {u : R} (hu : IsUnit u) :
    W'.Nonsingular (u • P) ↔ W'.Nonsingular P :=
  have {u : R} (hu : IsUnit u) {P : Fin 3 → R} (hP : W'.Nonsingular <| u • P) :
      W'.Nonsingular P := by
    /-
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu✝ : IsUnit u✝
      u : R
      hu : IsUnit u
      P : Fin 3 → R
      hP : W'.Nonsingular (HSMul.hSMul u P)
      ⊢ W'.Nonsingular P
    -/
    rcases (nonsingular_iff _).mp hP with ⟨hP, hP'⟩
    /-
      case intro
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu✝ : IsUnit u✝
      u : R
      hu : IsUnit u
      P : Fin 3 → R
      hP✝ : W'.Nonsingular (HSMul.hSMul u P)
      hP : W'.Equation (HSMul.hSMul u P)
      hP' : Or (Ne (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (HSMul.hSMul u P 1)) (HSMu …
      ⊢ W'.Nonsingular P
    -/
    refine (nonsingular_iff P).mpr ⟨(equation_smul P hu).mp hP, ?_⟩
    /-
      case intro
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu✝ : IsUnit u✝
      u : R
      hu : IsUnit u
      P : Fin 3 → R
      hP✝ : W'.Nonsingular (HSMul.hSMul u P)
      hP : W'.Equation (HSMul.hSMul u P)
      hP' : Or (Ne (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (HSMul.hSMul u P 1)) (HSMu …
      ⊢ Or (Ne (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (P 1)) (P 2)) (HAdd.hAdd (HAdd …
    -/
    contrapose! hP'
    /-
      case intro
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu✝ : IsUnit u✝
      u : R
      hu : IsUnit u
      P : Fin 3 → R
      hP✝ : W'.Nonsingular (HSMul.hSMul u P)
      hP : W'.Equation (HSMul.hSMul u P)
      hP' : And (Eq (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (P 1)) (P 2)) (HAdd.hAdd  …
      ⊢ And (Eq (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (HSMul.hSMul u P 1)) (HSMul.h …
    -/
    simp only [smul_fin3_ext]
    exact ⟨by linear_combination (norm := ring1) u ^ 4 * hP'.left,
      by linear_combination (norm := ring1) u ^ 3 * hP'.right.left,
      by linear_combination (norm := ring1) u ^ 5 * hP'.right.right⟩
                                                 /-
                                                   R : Type u
                                                   W' : WeierstrassCurve.Jacobian R
                                                   inst✝ : CommRing R
                                                   P : Fin 3 → R
                                                   u : R
                                                   hu : IsUnit u
                                                   this : ∀ {u : R}, IsUnit u → ∀ {P : Fin 3 → R}, W'.Nonsingular (HSMul.hSMul u  …
                                                   h : W'.Nonsingular P
                                                   ⊢ W'.Nonsingular (HSMul.hSMul (↑(Inv.inv hu.unit)) (HSMul.hSMul u P))
                                                 -/
  ⟨this hu, fun h => this hu.unit⁻¹.isUnit <| by rwa [smul_smul, hu.val_inv_mul, one_smul]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma nonsingular_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.Nonsingular P ↔ W'.Nonsingular Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Iff (W'.Nonsingular P) (W'.Nonsingular Q)
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Iff (W'.Nonsingular ((fun m => HSMul.hSMul m Q) u)) (W'.Nonsingular Q)
  -/
  exact nonsingular_smul Q u.isUnit
  /-
    🎉 no goals
  -/


lemma nonsingular_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) :
    W'.Nonsingular P ↔ W'.Equation P ∧ (3 * P x ^ 2 ≠ 0 ∨ 2 * P y ≠ 0 ∨ W'.a₁ * P x * P y ≠ 0) := by
  simp only [nonsingular_iff, hPz, add_zero, sub_zero, zero_sub, mul_zero,
    zero_pow <| OfNat.ofNat_ne_zero _, neg_ne_zero]


lemma nonsingular_zero [Nontrivial R] : W'.Nonsingular ![1, 1, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ W'.Nonsingular (Matrix.vecCons 1 (Matrix.vecCons 1 (Matrix.vecCons 0 Matrix. …
  -/
  simp only [nonsingular_of_Z_eq_zero, equation_zero, true_and, fin3_def_ext, ← not_and_or]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Not (And (Eq (HMul.hMul 3 (HPow.hPow 1 2)) 0) (And (Eq (HMul.hMul 2 1) 0) (E …
  -/
  exact fun h => one_ne_zero <| by linear_combination (norm := ring1) h.1 - h.2.1
  /-
    🎉 no goals
  -/


lemma nonsingular_some (X Y : R) : W'.Nonsingular ![X, Y, 1] ↔ W'.toAffine.Nonsingular X Y := by
  simp_rw [nonsingular_iff, equation_some, fin3_def_ext, Affine.nonsingular_iff',
    Affine.equation_iff', and_congr_right_iff, ← not_and_or, not_iff_not, one_pow, mul_one,
    and_congr_right_iff, Iff.comm, iff_self_and]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    X Y : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow Y 2) (HMul.hMul (HMul.hMul W' …
  -/
  intro h hX hY
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    X Y : R
    h : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow Y 2) (HMul.hMul (HMul.hMul  …
    hX : Eq (HSub.hSub (HMul.hMul W'.a₁ Y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPo …
    hY : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 Y) (HMul.hMul W'.toAffine.a₁ X)) W' …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul W'.a₁ X) Y) (HMul.hMul (HMul. …
  -/
  linear_combination (norm := ring1) 6 * h - 2 * X * hX - 3 * Y * hY
  /-
    🎉 no goals
  -/


lemma nonsingular_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.Nonsingular P ↔ W.toAffine.Nonsingular (P x / P z ^ 2) (P y / P z ^ 3) :=
  (nonsingular_of_equiv <| equiv_some_of_Z_ne_zero hPz).trans <| nonsingular_some ..


lemma nonsingular_iff_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.Nonsingular P ↔ W.Equation P ∧ (eval P W.polynomialX ≠ 0 ∨ eval P W.polynomialY ≠ 0) := by
  rw [nonsingular_of_Z_ne_zero hPz, Affine.Nonsingular, ← equation_of_Z_ne_zero hPz,
    ← eval_polynomialX_of_Z_ne_zero hPz, div_ne_zero_iff, and_iff_left <| pow_ne_zero 4 hPz,
    ← eval_polynomialY_of_Z_ne_zero hPz, div_ne_zero_iff, and_iff_left <| pow_ne_zero 3 hPz]


lemma X_ne_zero_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Nonsingular P)
    (hPz : P z = 0) : P x ≠ 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Nonsingular P
    hPz : Eq (P 2) 0
    ⊢ Ne (P 0) 0
  -/
  intro hPx
  simp only [nonsingular_of_Z_eq_zero hPz, equation_of_Z_eq_zero hPz, hPx, mul_zero, zero_mul,
    zero_pow <| OfNat.ofNat_ne_zero _, ne_self_iff_false, or_false, false_or] at hP
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    hPx : Eq (P 0) 0
    hP : And (Eq (HPow.hPow (P 1) 2) 0) (Ne (HMul.hMul 2 (P 1)) 0)
    ⊢ False
  -/
  rwa [pow_eq_zero_iff two_ne_zero, hP.left, eq_self, true_and, mul_zero, ne_self_iff_false] at hP
  /-
    🎉 no goals
  -/


lemma isUnit_X_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) : IsUnit (P x) :=
  (X_ne_zero_of_Z_eq_zero hP hPz).isUnit


lemma Y_ne_zero_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Nonsingular P)
    (hPz : P z = 0) : P y ≠ 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Nonsingular P
    hPz : Eq (P 2) 0
    ⊢ Ne (P 1) 0
  -/
  have hPx : P x ≠ 0 := X_ne_zero_of_Z_eq_zero hP hPz
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Nonsingular P
    hPz : Eq (P 2) 0
    hPx : Ne (P 0) 0
    ⊢ Ne (P 1) 0
  -/
  intro hPy
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Nonsingular P
    hPz : Eq (P 2) 0
    hPx : Ne (P 0) 0
    hPy : Eq (P 1) 0
    ⊢ False
  -/
  rw [nonsingular_of_Z_eq_zero hPz, equation_of_Z_eq_zero hPz, hPy, zero_pow two_ne_zero] at hP
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : And (Eq 0 (HPow.hPow (P 0) 3)) (Or (Ne (HMul.hMul 3 (HPow.hPow (P 0) 2))  …
    hPz : Eq (P 2) 0
    hPx : Ne (P 0) 0
    hPy : Eq (P 1) 0
    ⊢ False
  -/
  exact hPx <| pow_eq_zero hP.left.symm
  /-
    🎉 no goals
  -/


lemma isUnit_Y_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) : IsUnit (P y) :=
  (Y_ne_zero_of_Z_eq_zero hP hPz).isUnit


lemma equiv_of_Z_eq_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Nonsingular Q)
    (hPz : P z = 0) (hQz : Q z = 0) : P ≈ Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ HasEquiv.Equiv P Q
  -/
  have hPx : IsUnit <| P x := isUnit_X_of_Z_eq_zero hP hPz
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    ⊢ HasEquiv.Equiv P Q
  -/
  have hPy : IsUnit <| P y := isUnit_Y_of_Z_eq_zero hP hPz
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    hPy : IsUnit (P 1)
    ⊢ HasEquiv.Equiv P Q
  -/
  have hQx : IsUnit <| Q x := isUnit_X_of_Z_eq_zero hQ hQz
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    hPy : IsUnit (P 1)
    hQx : IsUnit (Q 0)
    ⊢ HasEquiv.Equiv P Q
  -/
  have hQy : IsUnit <| Q y := isUnit_Y_of_Z_eq_zero hQ hQz
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    hPy : IsUnit (P 1)
    hQx : IsUnit (Q 0)
    hQy : IsUnit (Q 1)
    ⊢ HasEquiv.Equiv P Q
  -/
  simp only [nonsingular_of_Z_eq_zero, equation_of_Z_eq_zero, hPz, hQz] at hP hQ
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    hPy : IsUnit (P 1)
    hQx : IsUnit (Q 0)
    hQy : IsUnit (Q 1)
    hP : And (Eq (HPow.hPow (P 1) 2) (HPow.hPow (P 0) 3)) (Or (Ne (HMul.hMul 3 (HP …
    hQ : And (Eq (HPow.hPow (Q 1) 2) (HPow.hPow (Q 0) 3)) (Or (Ne (HMul.hMul 3 (HP …
    ⊢ HasEquiv.Equiv P Q
  -/
  use (hPy.unit / hPx.unit) * (hQx.unit / hQy.unit)
  simp only [Units.smul_def, smul_fin3, Units.val_mul, Units.val_div_eq_div_val, IsUnit.unit_spec,
    mul_pow, div_pow, hQz, mul_zero]
  /-
    case h
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    hPy : IsUnit (P 1)
    hQx : IsUnit (Q 0)
    hQy : IsUnit (Q 1)
    hP : And (Eq (HPow.hPow (P 1) 2) (HPow.hPow (P 0) 3)) (Or (Ne (HMul.hMul 3 (HP …
    hQ : And (Eq (HPow.hPow (Q 1) 2) (HPow.hPow (Q 0) 3)) (Or (Ne (HMul.hMul 3 (HP …
    ⊢ Eq (Matrix.vecCons (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.hPow (P 1) 2) (HPo …
  -/
  conv_rhs => rw [← fin3_def P, hPz]
  /-
    case h
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    hPx : IsUnit (P 0)
    hPy : IsUnit (P 1)
    hQx : IsUnit (Q 0)
    hQy : IsUnit (Q 1)
    hP : And (Eq (HPow.hPow (P 1) 2) (HPow.hPow (P 0) 3)) (Or (Ne (HMul.hMul 3 (HP …
    hQ : And (Eq (HPow.hPow (Q 1) 2) (HPow.hPow (Q 0) 3)) (Or (Ne (HMul.hMul 3 (HP …
    ⊢ Eq (Matrix.vecCons (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.hPow (P 1) 2) (HPo …
  -/
  congr! 2
  · rw [hP.left, pow_succ, (hPx.pow 2).mul_div_cancel_left, hQ.left, pow_succ _ 2,
      (hQx.pow 2).div_mul_cancel_left, hQx.inv_mul_cancel_right]
  · rw [← hP.left, pow_succ, (hPy.pow 2).mul_div_cancel_left, ← hQ.left, pow_succ _ 2,
      (hQy.pow 2).div_mul_cancel_left, hQy.inv_mul_cancel_right]


lemma equiv_zero_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) :
    P ≈ ![1, 1, 0] :=
  equiv_of_Z_eq_zero hP nonsingular_zero hPz rfl


variable (W') in
/-- The proposition that a point class on `W'` is nonsingular. If `P` is a point representative,
then `W'.NonsingularLift ⟦P⟧` is definitionally equivalent to `W'.Nonsingular P`. -/
def NonsingularLift (P : PointClass R) : Prop :=
  P.lift W'.Nonsingular fun _ _ => propext ∘ nonsingular_of_equiv


lemma nonsingularLift_iff (P : Fin 3 → R) : W'.NonsingularLift ⟦P⟧ ↔ W'.Nonsingular P :=
  Iff.rfl


lemma nonsingularLift_zero [Nontrivial R] : W'.NonsingularLift ⟦![1, 1, 0]⟧ :=
  nonsingular_zero


lemma nonsingularLift_some (X Y : R) :
    W'.NonsingularLift ⟦![X, Y, 1]⟧ ↔ W'.toAffine.Nonsingular X Y :=
  nonsingular_some X Y


variable (W') in
/-- The $Y$-coordinate of the negation of a point representative. -/
def negY (P : Fin 3 → R) : R :=
  -P y - W'.a₁ * P x * P z - W'.a₃ * P z ^ 3


lemma negY_smul (P : Fin 3 → R) {u : R} : W'.negY (u • P) = u ^ 3 * W'.negY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.negY (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 3) (W'.negY P))
  -/
  simp only [negY, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow u 3) (P 1))) (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negY_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) : W'.negY P = -P y := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.negY P) (Neg.neg (P 1))
  -/
  simp only [negY, hPz, sub_zero, mul_zero, zero_pow three_ne_zero]
  /-
    🎉 no goals
  -/


lemma negY_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.negY P / P z ^ 3 = W.toAffine.negY (P x / P z ^ 2) (P y / P z ^ 3) := by
  linear_combination (norm := (rw [negY, Affine.negY]; ring1))
    -W.a₁ * P x / P z ^ 2 * div_self hPz - W.a₃ * div_self (pow_ne_zero 3 hPz)


lemma Y_sub_Y_mul_Y_sub_negY {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) :
    (P y * Q z ^ 3 - Q y * P z ^ 3) * (P y * Q z ^ 3 - W'.negY Q * P z ^ 3) = 0 := by
  linear_combination' (norm := (rw [negY]; ring1)) Q z ^ 6 * (equation_iff P).mp hP
    - P z ^ 6 * (equation_iff Q).mp hQ + hx * hx * hx + W'.a₂ * P z ^ 2 * Q z ^ 2 * hx * hx
    + (W'.a₄ * P z ^ 4 * Q z ^ 4 - W'.a₁ * P y * P z * Q z ^ 4) * hx


lemma Y_eq_of_Y_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) :
    P y * Q z ^ 3 = W.negY Q * P z ^ 3 :=
  eq_of_sub_eq_zero <| (mul_eq_zero.mp <| Y_sub_Y_mul_Y_sub_negY hP hQ hx).resolve_left <|
    sub_ne_zero_of_ne hy


lemma Y_eq_of_Y_ne' {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    P y * Q z ^ 3 = Q y * P z ^ 3 :=
  eq_of_sub_eq_zero <| (mul_eq_zero.mp <| Y_sub_Y_mul_Y_sub_negY hP hQ hx).resolve_right <|
    sub_ne_zero_of_ne hy


lemma Y_eq_iff' {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) :
    P y * Q z ^ 3 = W.negY Q * P z ^ 3 ↔
      P y / P z ^ 3 = W.toAffine.negY (Q x / Q z ^ 2) (Q y / Q z ^ 3) :=
  negY_of_Z_ne_zero hQz ▸ (div_eq_div_iff (pow_ne_zero 3 hPz) (pow_ne_zero 3 hQz)).symm


lemma Y_sub_Y_add_Y_sub_negY (P Q : Fin 3 → R) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) :
    (P y * Q z ^ 3 - Q y * P z ^ 3) + (P y * Q z ^ 3 - W'.negY Q * P z ^ 3) =
      (P y - W'.negY P) * Q z ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q …
  -/
  linear_combination (norm := (rw [negY, negY]; ring1)) -W'.a₁ * P z * Q z * hx
  /-
    🎉 no goals
  -/


lemma Y_ne_negY_of_Y_ne [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) :
    P y ≠ W'.negY P := by
  have hy' : P y * Q z ^ 3 - W'.negY Q * P z ^ 3 = 0 :=
    (mul_eq_zero.mp <| Y_sub_Y_mul_Y_sub_negY hP hQ hx).resolve_left <| sub_ne_zero_of_ne hy
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Ne (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY  …
    ⊢ Ne (P 1) (W'.negY P)
  -/
  contrapose! hy
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY  …
    hy : Eq (P 1) (W'.negY P)
    ⊢ Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2) 3))
  -/
  linear_combination (norm := ring1) Y_sub_Y_add_Y_sub_negY P Q hx + Q z ^ 3 * hy - hy'
  /-
    🎉 no goals
  -/


lemma Y_ne_negY_of_Y_ne' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy : P y * Q z ^ 3 ≠ W'.negY Q * P z ^ 3) : P y ≠ W'.negY P := by
  have hy' : P y * Q z ^ 3 - Q y * P z ^ 3 = 0 :=
    (mul_eq_zero.mp <| Y_sub_Y_mul_Y_sub_negY hP hQ hx).resolve_right <| sub_ne_zero_of_ne hy
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Ne (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hPo …
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HP …
    ⊢ Ne (P 1) (W'.negY P)
  -/
  contrapose! hy
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HP …
    hy : Eq (P 1) (W'.negY P)
    ⊢ Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hPow ( …
  -/
  linear_combination (norm := ring1) Y_sub_Y_add_Y_sub_negY P Q hx + Q z ^ 3 * hy - hy'
  /-
    🎉 no goals
  -/


lemma Y_eq_negY_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W'.negY Q * P z ^ 3) : P y = W'.negY P :=
  mul_left_injective₀ (pow_ne_zero 3 hQz) <| by
    /-
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      P Q : Fin 3 → R
      hQz : Ne (Q 2) 0
      hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
      hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
      hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
      ⊢ Eq ((fun a => HMul.hMul a (HPow.hPow (Q 2) 3)) (P 1)) ((fun a => HMul.hMul a …
    -/
    linear_combination (norm := ring1) -Y_sub_Y_add_Y_sub_negY P Q hx + hy + hy'
    /-
      🎉 no goals
    -/


lemma nonsingular_iff_of_Y_eq_negY {P : Fin 3 → F} (hPz : P z ≠ 0) (hy : P y = W.negY P) :
    W.Nonsingular P ↔ W.Equation P ∧ eval P W.polynomialX ≠ 0 := by
  have : eval P W.polynomialY = P y - W.negY P := by
    rw [negY, eval_polynomialY]; ring1
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    hy : Eq (P 1) (W.negY P)
    this : Eq ((MvPolynomial.eval P) W.polynomialY) (HSub.hSub (P 1) (W.negY P))
    ⊢ Iff (W.Nonsingular P) (And (W.Equation P) (Ne ((MvPolynomial.eval P) W.polyn …
  -/
  rw [nonsingular_iff_of_Z_ne_zero hPz, this, hy, sub_self, ne_self_iff_false, or_false]
  /-
    🎉 no goals
  -/


variable (W') in
/-- The unit associated to the doubling of a 2-torsion point.
More specifically, the unit `u` such that `W.add P P = u • ![1, 1, 0]` where `P = W.neg P`. -/
noncomputable def dblU (P : Fin 3 → R) : R :=
  eval P W'.polynomialX


lemma dblU_eq (P : Fin 3 → R) : W'.dblU P =
    W'.a₁ * P y * P z - (3 * P x ^ 2 + 2 * W'.a₂ * P x * P z ^ 2 + W'.a₄ * P z ^ 4) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.dblU P) (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (P 1)) (P 2)) (HAdd.hA …
  -/
  rw [dblU, eval_polynomialX]
  /-
    🎉 no goals
  -/


lemma dblU_smul (P : Fin 3 → R) (u : R) : W'.dblU (u • P) = u ^ 4 * W'.dblU P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblU (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W'.dblU P))
  -/
  simp only [dblU_eq, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (HMul.hMul (HPow.hPow u 3) (P 1))) …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblU_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) : W'.dblU P = -3 * P x ^ 2 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.dblU P) (HMul.hMul (-3) (HPow.hPow (P 0) 2))
  -/
  rw [dblU_eq, hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul W'.a₁ (P 1)) 0) (HAdd.hAdd (HAdd.hAdd (H …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblU_ne_zero_of_Y_eq {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W.negY Q * P z ^ 3) : W.dblU P ≠ 0 :=
  ((nonsingular_iff_of_Y_eq_negY hPz <| Y_eq_negY_of_Y_eq hQz hx hy hy').mp hP).right


lemma isUnit_dblU_of_Y_eq {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W.negY Q * P z ^ 3) : IsUnit (W.dblU P) :=
  (dblU_ne_zero_of_Y_eq hP hPz hQz hx hy hy').isUnit


variable (W') in
/-- The $Z$-coordinate of the doubling of a point representative. -/
def dblZ (P : Fin 3 → R) : R :=
  P z * (P y - W'.negY P)


lemma dblZ_smul (P : Fin 3 → R) (u : R) : W'.dblZ (u • P) = u ^ 4 * W'.dblZ P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblZ (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W'.dblZ P))
  -/
  simp only [dblZ, negY_smul, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HMul.hMul (HMul.hMul u (P 2)) (HSub.hSub (HMul.hMul (HPow.hPow u 3) (P 1 …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblZ_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) : W'.dblZ P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.dblZ P) 0
  -/
  rw [dblZ, hPz, zero_mul]
  /-
    🎉 no goals
  -/


lemma dblZ_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W'.negY Q * P z ^ 3) : W'.dblZ P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
    ⊢ Eq (W'.dblZ P) 0
  -/
  rw [dblZ, Y_eq_negY_of_Y_eq hQz hx hy hy', sub_self, mul_zero]
  /-
    🎉 no goals
  -/


lemma dblZ_ne_zero_of_Y_ne [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hPz : P z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) : W'.dblZ P ≠ 0 :=
  mul_ne_zero hPz <| sub_ne_zero_of_ne <| Y_ne_negY_of_Y_ne hP hQ hx hy


lemma isUnit_dblZ_of_Y_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) : IsUnit (W.dblZ P) :=
  (dblZ_ne_zero_of_Y_ne hP hQ hPz hx hy).isUnit


lemma dblZ_ne_zero_of_Y_ne' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hPz : P z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy : P y * Q z ^ 3 ≠ W'.negY Q * P z ^ 3) : W'.dblZ P ≠ 0 :=
  mul_ne_zero hPz <| sub_ne_zero_of_ne <| Y_ne_negY_of_Y_ne' hP hQ hx hy


lemma isUnit_dblZ_of_Y_ne' {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    IsUnit (W.dblZ P) :=
  (dblZ_ne_zero_of_Y_ne' hP hQ hPz hx hy).isUnit


private lemma toAffine_slope_of_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3) =
      -W.dblU P / W.dblZ P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Ne (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W.negY Q) (HPow.hPow …
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0)  …
  -/
  have hPy : P y - W.negY P ≠ 0 := sub_ne_zero_of_ne <| Y_ne_negY_of_Y_ne' hP hQ hx hy
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Ne (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W.negY Q) (HPow.hPow …
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0)  …
  -/
  simp only [mul_comm <| P z ^ _, X_eq_iff hPz hQz, ne_eq, Y_eq_iff' hPz hQz] at hx hy
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    hx : Eq (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0) (HPow.hPow (Q 2 …
    hy : Not (Eq (HDiv.hDiv (P 1) (HPow.hPow (P 2) 3)) (W.toAffine.negY (HDiv.hDiv …
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0)  …
  -/
  rw [Affine.slope_of_Y_ne hx <| negY_of_Z_ne_zero hQz ▸ hy, ← negY_of_Z_ne_zero hPz, dblU_eq, dblZ]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    hx : Eq (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0) (HPow.hPow (Q 2 …
    hy : Not (Eq (HDiv.hDiv (P 1) (HPow.hPow (P 2) 3)) (W.toAffine.negY (HDiv.hDiv …
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow (HDiv …
  -/
  field_simp [pow_ne_zero 2 hPz]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    hx : Eq (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0) (HPow.hPow (Q 2 …
    hy : Not (Eq (HDiv.hDiv (P 1) (HPow.hPow (P 2) 3)) (W.toAffine.negY (HDiv.hDiv …
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


variable (W') in
/-- The $X$-coordinate of the doubling of a point representative. -/
noncomputable def dblX (P : Fin 3 → R) : R :=
  W'.dblU P ^ 2 - W'.a₁ * W'.dblU P * P z * (P y - W'.negY P)
    - W'.a₂ * P z ^ 2 * (P y - W'.negY P) ^ 2 - 2 * P x * (P y - W'.negY P) ^ 2


lemma dblX_smul (P : Fin 3 → R) (u : R) : W'.dblX (u • P) = (u ^ 4) ^ 2 * W'.dblX P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblX (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow (HPow.hPow u 4) 2) (W'. …
  -/
  simp_rw [dblX, dblU_smul, negY_smul, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HSub.hSub (HSub.hSub (HPow.hPow (HMul.hMul (HPow.hPow u 4) (W …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblX_of_Z_eq_zero {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.dblX P = (P x ^ 2) ^ 2 := by
  linear_combination (norm := (rw [dblX, dblU_of_Z_eq_zero hPz, negY_of_Z_eq_zero hPz, hPz]; ring1))
    -8 * P x * (equation_of_Z_eq_zero hPz).mp hP


lemma dblX_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W'.negY Q * P z ^ 3) : W'.dblX P = W'.dblU P ^ 2 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
    ⊢ Eq (W'.dblX P) (HPow.hPow (W'.dblU P) 2)
  -/
  rw [dblX, Y_eq_negY_of_Y_eq hQz hx hy hy']
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
    ⊢ Eq (HSub.hSub (HSub.hSub (HSub.hSub (HPow.hPow (W'.dblU P) 2) (HMul.hMul (HM …
  -/
  ring1
  /-
    🎉 no goals
  -/


private lemma toAffine_addX_of_eq {P : Fin 3 → F} {n d : F} (hPz : P z ≠ 0) (hd : d ≠ 0) :
    W.toAffine.addX (P x / P z ^ 2) (P x / P z ^ 2) (-n / (P z * d)) =
      (n ^ 2 - W.a₁ * n * P z * d - W.a₂ * P z ^ 2 * d ^ 2 - 2 * P x * d ^ 2) / (P z * d) ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    n d : F
    hPz : Ne (P 2) 0
    hd : Ne d 0
    ⊢ Eq (W.toAffine.addX (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (P 0) ( …
  -/
  field_simp [mul_ne_zero hPz hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    n d : F
    hPz : Ne (P 2) 0
    hd : Ne d 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblX_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    W.dblX P / W.dblZ P ^ 2 = W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
      (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)) := by
  rw [dblX, toAffine_slope_of_eq hP hQ hPz hQz hx hy, dblZ, ← (X_eq_iff hPz hQz).mp hx,
    toAffine_addX_of_eq hPz <| sub_ne_zero_of_ne <| Y_ne_negY_of_Y_ne' hP hQ hx hy]


variable (W') in
/-- The $Y$-coordinate of the negated doubling of a point representative. -/
noncomputable def negDblY (P : Fin 3 → R) : R :=
  -W'.dblU P * (W'.dblX P - P x * (P y - W'.negY P) ^ 2) + P y * (P y - W'.negY P) ^ 3


lemma negDblY_smul (P : Fin 3 → R) (u : R) : W'.negDblY (u • P) = (u ^ 4) ^ 3 * W'.negDblY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.negDblY (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow (HPow.hPow u 4) 3) ( …
  -/
  simp only [negDblY, dblU_smul, dblX_smul, negY_smul, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (HMul.hMul (HPow.hPow u 4) (W'.dblU P))) ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negDblY_of_Z_eq_zero {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.negDblY P = -(P x ^ 2) ^ 3 := by
  linear_combination' (norm :=
      (rw [negDblY, dblU_of_Z_eq_zero hPz, dblX_of_Z_eq_zero hP hPz, negY_of_Z_eq_zero hPz]; ring1))
    (8 * (equation_of_Z_eq_zero hPz).mp hP - 12 * P x ^ 3) * (equation_of_Z_eq_zero hPz).mp hP


lemma negDblY_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W'.negY Q * P z ^ 3) : W'.negDblY P = (-W'.dblU P) ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
    ⊢ Eq (W'.negDblY P) (HPow.hPow (Neg.neg (W'.dblU P)) 3)
  -/
  rw [negDblY, dblX_of_Y_eq hQz hx hy hy', Y_eq_negY_of_Y_eq hQz hx hy hy']
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (W'.dblU P)) (HSub.hSub (HPow.hPow (W'.dbl …
  -/
  ring1
  /-
    🎉 no goals
  -/


private lemma toAffine_negAddY_of_eq {P : Fin 3 → F} {n d : F} (hPz : P z ≠ 0) (hd : d ≠ 0) :
    W.toAffine.negAddY (P x / P z ^ 2) (P x / P z ^ 2) (P y / P z ^ 3) (-n / (P z * d)) =
      (-n * (n ^ 2 - W.a₁ * n * P z * d - W.a₂ * P z ^ 2 * d ^ 2 - 2 * P x * d ^ 2 - P x * d ^ 2)
        + P y * d ^ 3) / (P z * d) ^ 3 := by
  linear_combination (norm := (rw [Affine.negAddY, toAffine_addX_of_eq hPz hd]; ring1))
    -n * P x / (P z ^ 3 * d) * div_self (pow_ne_zero 2 hd)
      - P y / P z ^ 3 * div_self (pow_ne_zero 3 hd)


lemma negDblY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) : W.negDblY P / W.dblZ P ^ 3 =
    W.toAffine.negAddY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
      (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)) := by
  rw [negDblY, dblX, toAffine_slope_of_eq hP hQ hPz hQz hx hy, dblZ, ← (X_eq_iff hPz hQz).mp hx,
    toAffine_negAddY_of_eq hPz <| sub_ne_zero_of_ne <| Y_ne_negY_of_Y_ne' hP hQ hx hy]


variable (W') in
/-- The $Y$-coordinate of the doubling of a point representative. -/
noncomputable def dblY (P : Fin 3 → R) : R :=
  W'.negY ![W'.dblX P, W'.negDblY P, W'.dblZ P]


lemma dblY_smul (P : Fin 3 → R) (u : R) : W'.dblY (u • P) = (u ^ 4) ^ 3 * W'.dblY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblY (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow (HPow.hPow u 4) 3) (W'. …
  -/
  simp only [dblY, negY, dblX_smul, negDblY_smul, dblZ_smul, fin3_def_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow (HPow.hPow u 4) 3) ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblY_of_Z_eq_zero {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.dblY P = (P x ^ 2) ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.dblY P) (HPow.hPow (HPow.hPow (P 0) 2) 3)
  -/
  erw [dblY, negDblY_of_Z_eq_zero hP hPz, dblZ_of_Z_eq_zero hPz, negY_of_Z_eq_zero rfl, neg_neg]
  /-
    🎉 no goals
  -/


lemma dblY_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W'.negY Q * P z ^ 3) : W'.dblY P = W'.dblU P ^ 3 := by
  erw [dblY, dblZ_of_Y_eq hQz hx hy hy', negY_of_Z_eq_zero rfl, negDblY_of_Y_eq hQz hx hy hy',
    ← Odd.neg_pow <| by decide, neg_neg]


lemma dblY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    W.dblY P / W.dblZ P ^ 3 = W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
      (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)) := by
  erw [dblY, negY_of_Z_ne_zero <| dblZ_ne_zero_of_Y_ne' hP hQ hPz hx hy,
    dblX_of_Z_ne_zero hP hQ hPz hQz hx hy, negDblY_of_Z_ne_zero hP hQ hPz hQz hx hy, Affine.addY]


variable (W') in
/-- The coordinates of the doubling of a point representative. -/
noncomputable def dblXYZ (P : Fin 3 → R) : Fin 3 → R :=
  ![W'.dblX P, W'.dblY P, W'.dblZ P]


lemma dblXYZ_smul (P : Fin 3 → R) (u : R) : W'.dblXYZ (u • P) = (u ^ 4) • W'.dblXYZ P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblXYZ (HSMul.hSMul u P)) (HSMul.hSMul (HPow.hPow u 4) (W'.dblXYZ P))
  -/
  rw [dblXYZ, dblX_smul, dblY_smul, dblZ_smul]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (Matrix.vecCons (HMul.hMul (HPow.hPow (HPow.hPow u 4) 2) (W'.dblX P)) (Ma …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma dblXYZ_of_Z_eq_zero {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.dblXYZ P = P x ^ 2 • ![1, 1, 0] := by
  erw [dblXYZ, dblX_of_Z_eq_zero hP hPz, dblY_of_Z_eq_zero hP hPz, dblZ_of_Z_eq_zero hPz, smul_fin3,
    mul_one, mul_one, mul_zero]


lemma dblXYZ_of_Y_eq' [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W'.negY Q * P z ^ 3) :
    W'.dblXYZ P = ![W'.dblU P ^ 2, W'.dblU P ^ 3, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W'.negY Q) (HPow.hP …
    ⊢ Eq (W'.dblXYZ P) (Matrix.vecCons (HPow.hPow (W'.dblU P) 2) (Matrix.vecCons ( …
  -/
  rw [dblXYZ, dblX_of_Y_eq hQz hx hy hy', dblY_of_Y_eq hQz hx hy hy', dblZ_of_Y_eq hQz hx hy hy']
  /-
    🎉 no goals
  -/


lemma dblXYZ_of_Y_eq {P Q : Fin 3 → F} (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy : P y * Q z ^ 3 = Q y * P z ^ 3) (hy' : P y * Q z ^ 3 = W.negY Q * P z ^ 3) :
    W.dblXYZ P = W.dblU P • ![1, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W.negY Q) (HPow.hPo …
    ⊢ Eq (W.dblXYZ P) (HSMul.hSMul (W.dblU P) (Matrix.vecCons 1 (Matrix.vecCons 1  …
  -/
  erw [dblXYZ_of_Y_eq' hQz hx hy hy', smul_fin3, mul_one, mul_one, mul_zero]
  /-
    🎉 no goals
  -/


lemma dblXYZ_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    W.dblXYZ P = W.dblZ P •
      ![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Ne (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W.negY Q) (HPow.hPow …
    ⊢ Eq (W.dblXYZ P) (HSMul.hSMul (W.dblZ P) (Matrix.vecCons (W.toAffine.addX (HD …
  -/
  have hZ {n : ℕ} : IsUnit <| W.dblZ P ^ n := (isUnit_dblZ_of_Y_ne' hP hQ hPz hx hy).pow n
  erw [dblXYZ, smul_fin3, ← dblX_of_Z_ne_zero hP hQ hPz hQz hx hy, hZ.mul_div_cancel,
    ← dblY_of_Z_ne_zero hP hQ hPz hQz hx hy, hZ.mul_div_cancel, mul_one]


/-- The unit associated to the addition of a non-2-torsion point with its negation.
More specifically, the unit `u` such that `W.add P Q = u • ![1, 1, 0]` where
`P x / P z ^ 2 = Q x / Q z ^ 2` but `P ≠ W.neg P`. -/
def addU (P Q : Fin 3 → F) : F :=
  -((P y * Q z ^ 3 - Q y * P z ^ 3) / (P z * Q z))


lemma addU_smul {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) {u v : F} (hu : u ≠ 0)
    (hv : v ≠ 0) : addU (u • P) (v • Q) = (u * v) ^ 2 * addU P Q := by
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    u v : F
    hu : Ne u 0
    hv : Ne v 0
    ⊢ Eq (WeierstrassCurve.Jacobian.addU (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMu …
  -/
  field_simp [addU, smul_fin3_ext]
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    u v : F
    hu : Ne u 0
    hv : Ne v 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow v 3) (Q 1)) (HPow. …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addU_of_Z_eq_zero_left {P Q : Fin 3 → F} (hPz : P z = 0) : addU P Q = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hPz : Eq (P 2) 0
    ⊢ Eq (WeierstrassCurve.Jacobian.addU P Q) 0
  -/
  rw [addU, hPz, zero_mul, div_zero, neg_zero]
  /-
    🎉 no goals
  -/


lemma addU_of_Z_eq_zero_right {P Q : Fin 3 → F} (hQz : Q z = 0) : addU P Q = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hQz : Eq (Q 2) 0
    ⊢ Eq (WeierstrassCurve.Jacobian.addU P Q) 0
  -/
  rw [addU, hQz, mul_zero, div_zero, neg_zero]
  /-
    🎉 no goals
  -/


lemma addU_ne_zero_of_Y_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) : addU P Q ≠ 0 :=
  neg_ne_zero.mpr <| div_ne_zero (sub_ne_zero_of_ne hy) <| mul_ne_zero hPz hQz


lemma isUnit_addU_of_Y_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) : IsUnit (addU P Q) :=
  (addU_ne_zero_of_Y_ne hPz hQz hy).isUnit


/-- The $Z$-coordinate of the addition of two distinct point representatives. -/
def addZ (P Q : Fin 3 → R) : R :=
  P x * Q z ^ 2 - Q x * P z ^ 2


lemma addZ_self {P : Fin 3 → R} : addZ P P = 0 := sub_self _


lemma addZ_smul (P Q : Fin 3 → R) (u v : R) : addZ (u • P) (v • Q) = (u * v) ^ 2 * addZ P Q := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (WeierstrassCurve.Jacobian.addZ (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMu …
  -/
  simp only [addZ, smul_fin3_ext]
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow u 2) (P 0)) (HPow.hPow (HMul. …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_of_Z_eq_zero_left {P Q : Fin 3 → R} (hPz : P z = 0) : addZ P Q = P x * Q z * Q z := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (WeierstrassCurve.Jacobian.addZ P Q) (HMul.hMul (HMul.hMul (P 0) (Q 2)) ( …
  -/
  rw [addZ, hPz]
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_of_Z_eq_zero_right {P Q : Fin 3 → R} (hQz : Q z = 0) :
    addZ P Q = -(Q x * P z) * P z := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hQz : Eq (Q 2) 0
    ⊢ Eq (WeierstrassCurve.Jacobian.addZ P Q) (HMul.hMul (Neg.neg (HMul.hMul (Q 0) …
  -/
  rw [addZ, hQz]
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hQz : Eq (Q 2) 0
    ⊢ Eq (HSub.hSub (HMul.hMul (P 0) (HPow.hPow 0 2)) (HMul.hMul (Q 0) (HPow.hPow  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_of_X_eq {P Q : Fin 3 → R} (hx : P x * Q z ^ 2 = Q x * P z ^ 2) : addZ P Q = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    ⊢ Eq (WeierstrassCurve.Jacobian.addZ P Q) 0
  -/
  rw [addZ, hx, sub_self]
  /-
    🎉 no goals
  -/


lemma addZ_ne_zero_of_X_ne {P Q : Fin 3 → R} (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) : addZ P Q ≠ 0 :=
  sub_ne_zero_of_ne hx


lemma isUnit_addZ_of_X_ne {P Q : Fin 3 → F} (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) :
    IsUnit <| addZ P Q :=
  (addZ_ne_zero_of_X_ne hx).isUnit


private lemma toAffine_slope_of_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) :
    W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3) =
      (P y * Q z ^ 3 - Q y * P z ^ 3) / (P z * Q z * addZ P Q) := by
  rw [Affine.slope_of_X_ne <| by rwa [ne_eq, ← X_eq_iff hPz hQz],
    div_sub_div _ _ (pow_ne_zero 2 hPz) (pow_ne_zero 2 hQz), mul_comm <| _ ^ 2, addZ]
  field_simp [mul_ne_zero (mul_ne_zero hPz hQz) <| sub_ne_zero_of_ne hx,
    mul_ne_zero (mul_ne_zero (pow_ne_zero 3 hPz) (pow_ne_zero 3 hQz)) <| sub_ne_zero_of_ne hx]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (H …
  -/
  ring1
  /-
    🎉 no goals
  -/


variable (W') in
/-- The $X$-coordinate of the addition of two distinct point representatives. -/
def addX (P Q : Fin 3 → R) : R :=
  P x * Q x ^ 2 * P z ^ 2 - 2 * P y * Q y * P z * Q z + P x ^ 2 * Q x * Q z ^ 2
    - W'.a₁ * P x * Q y * P z ^ 2 * Q z - W'.a₁ * P y * Q x * P z * Q z ^ 2
    + 2 * W'.a₂ * P x * Q x * P z ^ 2 * Q z ^ 2 - W'.a₃ * Q y * P z ^ 4 * Q z
    - W'.a₃ * P y * P z * Q z ^ 4 + W'.a₄ * Q x * P z ^ 4 * Q z ^ 2
    + W'.a₄ * P x * P z ^ 2 * Q z ^ 4 + 2 * W'.a₆ * P z ^ 4 * Q z ^ 4


lemma addX_self {P : Fin 3 → R} (hP : W'.Equation P) : W'.addX P P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addX P P) 0
  -/
  linear_combination (norm := (rw [addX]; ring1)) -2 * P z ^ 2 * (equation_iff _).mp hP
  /-
    🎉 no goals
  -/


lemma addX_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q) :
    W'.addX P Q * (P z * Q z) ^ 2 =
      (P y * Q z ^ 3 - Q y * P z ^ 3) ^ 2
        + W'.a₁ * (P y * Q z ^ 3 - Q y * P z ^ 3) * P z * Q z * addZ P Q
        - W'.a₂ * P z ^ 2 * Q z ^ 2 * addZ P Q ^ 2 - P x * Q z ^ 2 * addZ P Q ^ 2
        - Q x * P z ^ 2 * addZ P Q ^ 2 := by
  linear_combination (norm := (rw [addX, addZ]; ring1)) -Q z ^ 6 * (equation_iff P).mp hP
    - P z ^ 6 * (equation_iff Q).mp hQ


lemma addX_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) : W.addX P Q =
      ((P y * Q z ^ 3 - Q y * P z ^ 3) ^ 2
        + W.a₁ * (P y * Q z ^ 3 - Q y * P z ^ 3) * P z * Q z * addZ P Q
        - W.a₂ * P z ^ 2 * Q z ^ 2 * addZ P Q ^ 2 - P x * Q z ^ 2 * addZ P Q ^ 2
        - Q x * P z ^ 2 * addZ P Q ^ 2) / (P z * Q z) ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W.addX P Q) (HDiv.hDiv (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HPow …
  -/
  rw [← addX_eq' hP hQ, mul_div_cancel_right₀ _ <| pow_ne_zero 2 <| mul_ne_zero hPz hQz]
  /-
    🎉 no goals
  -/


lemma addX_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addX (u • P) (v • Q) = ((u * v) ^ 2) ^ 2 * W'.addX P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addX (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (HPow …
  -/
  simp only [addX, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_Z_eq_zero_left {P Q : Fin 3 → R} (hPz : P z = 0) :
    W'.addX P Q = (P x * Q z) ^ 2 * Q x := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.addX P Q) (HMul.hMul (HPow.hPow (HMul.hMul (P 0) (Q 2)) 2) (Q 0))
  -/
  rw [addX, hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_Z_eq_zero_right {P Q : Fin 3 → R} (hQz : Q z = 0) :
    W'.addX P Q = (-(Q x * P z)) ^ 2 * P x := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hQz : Eq (Q 2) 0
    ⊢ Eq (W'.addX P Q) (HMul.hMul (HPow.hPow (Neg.neg (HMul.hMul (Q 0) (P 2))) 2)  …
  -/
  rw [addX, hQz]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hQz : Eq (Q 2) 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_X_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) :
    W'.addX P Q * (P z * Q z) ^ 2 = (P y * Q z ^ 3 - Q y * P z ^ 3) ^ 2 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    ⊢ Eq (HMul.hMul (W'.addX P Q) (HPow.hPow (HMul.hMul (P 2) (Q 2)) 2)) (HPow.hPo …
  -/
  simp only [addX_eq' hP hQ, addZ_of_X_eq hx, add_zero, sub_zero, mul_zero, zero_pow two_ne_zero]
  /-
    🎉 no goals
  -/


lemma addX_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) : W.addX P Q = addU P Q ^ 2 := by
  rw [addU, neg_sq, div_pow, ← addX_of_X_eq' hP hQ hx,
    mul_div_cancel_right₀ _ <| pow_ne_zero 2 <| mul_ne_zero hPz hQz]


private lemma toAffine_addX_of_ne {P Q : Fin 3 → F} {n d : F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hd : d ≠ 0) : W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2) (n / (P z * Q z * d)) =
      (n ^ 2 + W.a₁ * n * P z * Q z * d - W.a₂ * P z ^ 2 * Q z ^ 2 * d ^ 2 - P x * Q z ^ 2 * d ^ 2
        - Q x * P z ^ 2 * d ^ 2) / (P z * Q z * d) ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    n d : F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hd : Ne d 0
    ⊢ Eq (W.toAffine.addX (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0) ( …
  -/
  field_simp [mul_ne_zero (mul_ne_zero hPz hQz) hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    n d : F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hd : Ne d 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) :
    W.addX P Q / addZ P Q ^ 2 = W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
      (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)) := by
  rw [addX_eq hP hQ hPz hQz, div_div, ← mul_pow, toAffine_slope_of_ne hPz hQz hx,
    toAffine_addX_of_ne hPz hQz <| addZ_ne_zero_of_X_ne hx]


variable (W') in
/-- The $Y$-coordinate of the negated addition of two distinct point representatives. -/
def negAddY (P Q : Fin 3 → R) : R :=
  -P y * Q x ^ 3 * P z ^ 3 + 2 * P y * Q y ^ 2 * P z ^ 3 - 3 * P x ^ 2 * Q x * Q y * P z ^ 2 * Q z
    + 3 * P x * P y * Q x ^ 2 * P z * Q z ^ 2 + P x ^ 3 * Q y * Q z ^ 3
    - 2 * P y ^ 2 * Q y * Q z ^ 3 + W'.a₁ * P x * Q y ^ 2 * P z ^ 4
    + W'.a₁ * P y * Q x * Q y * P z ^ 3 * Q z - W'.a₁ * P x * P y * Q y * P z * Q z ^ 3
    - W'.a₁ * P y ^ 2 * Q x * Q z ^ 4 - 2 * W'.a₂ * P x * Q x * Q y * P z ^ 4 * Q z
    + 2 * W'.a₂ * P x * P y * Q x * P z * Q z ^ 4 + W'.a₃ * Q y ^ 2 * P z ^ 6
    - W'.a₃ * P y ^ 2 * Q z ^ 6 - W'.a₄ * Q x * Q y * P z ^ 6 * Q z
    - W'.a₄ * P x * Q y * P z ^ 4 * Q z ^ 3 + W'.a₄ * P y * Q x * P z ^ 3 * Q z ^ 4
    + W'.a₄ * P x * P y * P z * Q z ^ 6 - 2 * W'.a₆ * Q y * P z ^ 6 * Q z ^ 3
    + 2 * W'.a₆ * P y * P z ^ 3 * Q z ^ 6


                                                              /-
                                                                R : Type u
                                                                W' : WeierstrassCurve.Jacobian R
                                                                inst✝ : CommRing R
                                                                P : Fin 3 → R
                                                                ⊢ Eq (W'.negAddY P P) 0
                                                              -/
lemma negAddY_self {P : Fin 3 → R} : W'.negAddY P P = 0 := by rw [negAddY]; ring
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


lemma negAddY_eq' {P Q : Fin 3 → R} : W'.negAddY P Q * (P z * Q z) ^ 3 =
    (P y * Q z ^ 3 - Q y * P z ^ 3) * (W'.addX P Q * (P z * Q z) ^ 2 - P x * Q z ^ 2 * addZ P Q ^ 2)
      + P y * Q z ^ 3 * addZ P Q ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    ⊢ Eq (HMul.hMul (W'.negAddY P Q) (HPow.hPow (HMul.hMul (P 2) (Q 2)) 3)) (HAdd. …
  -/
  rw [negAddY, addX, addZ]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_eq {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) : W.negAddY P Q =
    ((P y * Q z ^ 3 - Q y * P z ^ 3) * (W.addX P Q * (P z * Q z) ^ 2 - P x * Q z ^ 2 * addZ P Q ^ 2)
      + P y * Q z ^ 3 * addZ P Q ^ 3) / (P z * Q z) ^ 3 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W.negAddY P Q) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (P …
  -/
  rw [← negAddY_eq', mul_div_cancel_right₀ _ <| pow_ne_zero 3 <| mul_ne_zero hPz hQz]
  /-
    🎉 no goals
  -/


lemma negAddY_smul (P Q : Fin 3 → R) (u v : R) :
    W'.negAddY (u • P) (v • Q) = ((u * v) ^ 2) ^ 3 * W'.negAddY P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.negAddY (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (H …
  -/
  simp only [negAddY, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_of_Z_eq_zero_left {P Q : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.negAddY P Q = (P x * Q z) ^ 3 * W'.negY Q := by
  linear_combination (norm := (rw [negAddY, negY, hPz]; ring1))
    (W'.negY Q - Q y) * Q z ^ 3 * (equation_of_Z_eq_zero hPz).mp hP


lemma negAddY_of_Z_eq_zero_right {P Q : Fin 3 → R} (hQ : W'.Equation Q) (hQz : Q z = 0) :
    W'.negAddY P Q = (-(Q x * P z)) ^ 3 * W'.negY P := by
  linear_combination (norm := (rw [negAddY, negY, hQz]; ring1))
    (P y - W'.negY P) * P z ^ 3 * (equation_of_Z_eq_zero hQz).mp hQ


lemma negAddY_of_X_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) :
    W'.negAddY P Q * (P z * Q z) ^ 3 = (P y * Q z ^ 3 - Q y * P z ^ 3) ^ 3 := by
  simp only [negAddY_eq', addX_eq' hP hQ, addZ_of_X_eq hx, add_zero, sub_zero, mul_zero,
    zero_pow <| OfNat.ofNat_ne_zero _, ← pow_succ']


lemma negAddY_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) : W.negAddY P Q = (-addU P Q) ^ 3 := by
  rw [addU, neg_neg, div_pow, ← negAddY_of_X_eq' hP hQ hx,
    mul_div_cancel_right₀ _ <| pow_ne_zero 3 <| mul_ne_zero hPz hQz]


private lemma toAffine_negAddY_of_ne {P Q : Fin 3 → F} {n d : F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hd : d ≠ 0) :
    W.toAffine.negAddY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (n / (P z * Q z * d)) =
      (n * (n ^ 2 + W.a₁ * n * P z * Q z * d - W.a₂ * P z ^ 2 * Q z ^ 2 * d ^ 2
          - P x * Q z ^ 2 * d ^ 2 - Q x * P z ^ 2 * d ^ 2 - P x * Q z ^ 2 * d ^ 2)
        + P y * Q z ^ 3 * d ^ 3) / (P z * Q z * d) ^ 3 := by
  linear_combination (norm := (rw [Affine.negAddY, toAffine_addX_of_ne hPz hQz hd]; ring1))
    n * P x / (P z ^ 3 * Q z * d) * div_self (pow_ne_zero 2 <| mul_ne_zero hQz hd)
      - P y / P z ^ 3 * div_self (pow_ne_zero 3 <| mul_ne_zero hQz hd)


lemma negAddY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) : W.negAddY P Q / addZ P Q ^ 3 =
      W.toAffine.negAddY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
        (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)) := by
  rw [negAddY_eq hPz hQz, addX_eq' hP hQ, div_div, ← mul_pow _ _ 3, toAffine_slope_of_ne hPz hQz hx,
    toAffine_negAddY_of_ne hPz hQz <| addZ_ne_zero_of_X_ne hx]


variable (W') in
/-- The $Y$-coordinate of the addition of two distinct point representatives. -/
def addY (P Q : Fin 3 → R) : R :=
  W'.negY ![W'.addX P Q, W'.negAddY P Q, addZ P Q]


lemma addY_self {P : Fin 3 → R} (hP : W'.Equation P) : W'.addY P P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addY P P) 0
  -/
  erw [addY, addX_self hP, negAddY_self, addZ_self, negY_of_Z_eq_zero rfl, neg_zero]
  /-
    🎉 no goals
  -/


lemma addY_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addY (u • P) (v • Q) = ((u * v) ^ 2) ^ 3 * W'.addY P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addY (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (HPow …
  -/
  simp only [addY, negY, addX_smul, negAddY_smul, addZ_smul, fin3_def_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow (HPow.hPow (HMul.hMu …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addY_of_Z_eq_zero_left {P Q : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.addY P Q = (P x * Q z) ^ 3 * Q y := by
  simp only [addY, addX_of_Z_eq_zero_left hPz, negAddY_of_Z_eq_zero_left hP hPz,
    addZ_of_Z_eq_zero_left hPz, negY, fin3_def_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow (HMul.hMul (P 0) (Q  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addY_of_Z_eq_zero_right {P Q : Fin 3 → R} (hQ : W'.Equation Q) (hQz : Q z = 0) :
    W'.addY P Q = (-(Q x * P z)) ^ 3 * P y := by
  simp only [addY, addX_of_Z_eq_zero_right hQz, negAddY_of_Z_eq_zero_right hQ hQz,
    addZ_of_Z_eq_zero_right hQz, negY, fin3_def_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow (Neg.neg (HMul.hMul  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addY_of_X_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) :
    W'.addY P Q * (P z * Q z) ^ 3 = (-(P y * Q z ^ 3 - Q y * P z ^ 3)) ^ 3 := by
  erw [addY, negY, addZ_of_X_eq hx, mul_zero, sub_zero, zero_pow three_ne_zero, mul_zero, sub_zero,
    neg_mul, negAddY_of_X_eq' hP hQ hx, Odd.neg_pow <| by decide]


lemma addY_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) : W.addY P Q = addU P Q ^ 3 := by
  rw [addU, ← neg_div, div_pow, ← addY_of_X_eq' hP hQ hx,
    mul_div_cancel_right₀ _ <| pow_ne_zero 3 <| mul_ne_zero hPz hQz]


lemma addY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) :
    W.addY P Q / addZ P Q ^ 3 = W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
      (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)) := by
  erw [addY, negY_of_Z_ne_zero <| addZ_ne_zero_of_X_ne hx, addX_of_Z_ne_zero hP hQ hPz hQz hx,
    negAddY_of_Z_ne_zero hP hQ hPz hQz hx, Affine.addY]


variable (W') in
/-- The coordinates of the addition of two distinct point representatives. -/
noncomputable def addXYZ (P Q : Fin 3 → R) : Fin 3 → R :=
  ![W'.addX P Q, W'.addY P Q, addZ P Q]


lemma addXYZ_self {P : Fin 3 → R} (hP : W'.Equation P) : W'.addXYZ P P = ![0, 0, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addXYZ P P) (Matrix.vecCons 0 (Matrix.vecCons 0 (Matrix.vecCons 0 Mat …
  -/
  rw [addXYZ, addX_self hP, addY_self hP, addZ_self]
  /-
    🎉 no goals
  -/


lemma addXYZ_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addXYZ (u • P) (v • Q) = (u * v) ^ 2 • W'.addXYZ P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addXYZ (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HSMul.hSMul (HPow.hPow ( …
  -/
  rw [addXYZ, addX_smul, addY_smul, addZ_smul]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (Matrix.vecCons (HMul.hMul (HPow.hPow (HPow.hPow (HMul.hMul u v) 2) 2) (W …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma addXYZ_of_Z_eq_zero_left {P Q : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.addXYZ P Q = (P x * Q z) • Q := by
  rw [addXYZ, addX_of_Z_eq_zero_left hPz, addY_of_Z_eq_zero_left hP hPz, addZ_of_Z_eq_zero_left hPz,
    smul_fin3]


lemma addXYZ_of_Z_eq_zero_right {P Q : Fin 3 → R} (hQ : W'.Equation Q) (hQz : Q z = 0) :
    W'.addXYZ P Q = -(Q x * P z) • P := by
  rw [addXYZ, addX_of_Z_eq_zero_right hQz, addY_of_Z_eq_zero_right hQ hQz,
    addZ_of_Z_eq_zero_right hQz, smul_fin3]


lemma addXYZ_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) :
    W.addXYZ P Q = addU P Q • ![1, 1, 0] := by
  erw [addXYZ, addX_of_X_eq hP hQ hPz hQz hx, addY_of_X_eq hP hQ hPz hQz hx, addZ_of_X_eq hx,
    smul_fin3, mul_one, mul_one, mul_zero]


lemma addXYZ_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) :
    W.addXYZ P Q = addZ P Q •
      ![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    ⊢ Eq (W.addXYZ P Q) (HSMul.hSMul (WeierstrassCurve.Jacobian.addZ P Q) (Matrix. …
  -/
  have hZ {n : ℕ} : IsUnit <| addZ P Q ^ n := (isUnit_addZ_of_X_ne hx).pow n
  erw [addXYZ, smul_fin3, ← addX_of_Z_ne_zero hP hQ hPz hQz hx, hZ.mul_div_cancel,
    ← addY_of_Z_ne_zero hP hQ hPz hQz hx, hZ.mul_div_cancel, mul_one]


variable (W') in
/-- The negation of a point representative. -/
def neg (P : Fin 3 → R) : Fin 3 → R :=
  ![P x, W'.negY P, P z]


lemma neg_smul (P : Fin 3 → R) (u : R) : W'.neg (u • P) = u • W'.neg P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.neg (HSMul.hSMul u P)) (HSMul.hSMul u (W'.neg P))
  -/
  rw [neg, negY_smul]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (Matrix.vecCons (HSMul.hSMul u P 0) (Matrix.vecCons (HMul.hMul (HPow.hPow …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma neg_smul_equiv (P : Fin 3 → R) {u : R} (hu : IsUnit u) : W'.neg (u • P) ≈ W'.neg P :=
  ⟨hu.unit, (neg_smul ..).symm⟩


lemma neg_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.neg P ≈ W'.neg Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ HasEquiv.Equiv (W'.neg P) (W'.neg Q)
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ HasEquiv.Equiv (W'.neg ((fun m => HSMul.hSMul m Q) u)) (W'.neg Q)
  -/
  exact neg_smul_equiv Q u.isUnit
  /-
    🎉 no goals
  -/


lemma neg_of_Z_eq_zero' {P : Fin 3 → R} (hPz : P z = 0) : W'.neg P = ![P x, -P y, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.neg P) (Matrix.vecCons (P 0) (Matrix.vecCons (Neg.neg (P 1)) (Matrix. …
  -/
  rw [neg, negY_of_Z_eq_zero hPz, hPz]
  /-
    🎉 no goals
  -/


lemma neg_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) :
    W.neg P = -(P y / P x) • ![1, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hP : W.Nonsingular P
    hPz : Eq (P 2) 0
    ⊢ Eq (W.neg P) (HSMul.hSMul (Neg.neg (HDiv.hDiv (P 1) (P 0))) (Matrix.vecCons  …
  -/
  have hX {n : ℕ} : IsUnit <| P x ^ n := (isUnit_X_of_Z_eq_zero hP hPz).pow n
  erw [neg_of_Z_eq_zero' hPz, smul_fin3, neg_sq, div_pow, (equation_of_Z_eq_zero hPz).mp hP.left,
    pow_succ, hX.mul_div_cancel_left, mul_one, Odd.neg_pow <| by decide, div_pow, pow_succ,
    (equation_of_Z_eq_zero hPz).mp hP.left, hX.mul_div_cancel_left, mul_one, mul_zero]


lemma neg_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.neg P = P z • ![P x / P z ^ 2, W.toAffine.negY (P x / P z ^ 2) (P y / P z ^ 3), 1] := by
  erw [neg, smul_fin3, mul_div_cancel₀ _ <| pow_ne_zero 2 hPz, ← negY_of_Z_ne_zero hPz,
    mul_div_cancel₀ _ <| pow_ne_zero 3 hPz, mul_one]


private lemma nonsingular_neg_of_Z_ne_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) :
    W.Nonsingular ![P x / P z ^ 2, W.toAffine.negY (P x / P z ^ 2) (P y / P z ^ 3), 1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hP : W.Nonsingular P
    hPz : Ne (P 2) 0
    ⊢ W.Nonsingular (Matrix.vecCons (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (Matrix. …
  -/
  exact (nonsingular_some ..).mpr <| Affine.nonsingular_neg <| (nonsingular_of_Z_ne_zero hPz).mp hP
  /-
    🎉 no goals
  -/


lemma nonsingular_neg {P : Fin 3 → F} (hP : W.Nonsingular P) : W.Nonsingular <| W.neg P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hP : W.Nonsingular P
    ⊢ W.Nonsingular (W.neg P)
  -/
  by_cases hPz : P z = 0
  · simp only [neg_of_Z_eq_zero hP hPz, nonsingular_smul _
        ((isUnit_Y_of_Z_eq_zero hP hPz).div <| isUnit_X_of_Z_eq_zero hP hPz).neg, nonsingular_zero]
  · simp only [neg_of_Z_ne_zero hPz, nonsingular_smul _ <| Ne.isUnit hPz,
      nonsingular_neg_of_Z_ne_zero hP hPz]


lemma addZ_neg {P : Fin 3 → R} : addZ P (W'.neg P) = 0 := addZ_of_X_eq rfl


lemma addX_neg {P : Fin 3 → R} (hP : W'.Equation P) : W'.addX P (W'.neg P) = W'.dblZ P ^ 2 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addX P (W'.neg P)) (HPow.hPow (W'.dblZ P) 2)
  -/
  simp only [addX, neg, dblZ, negY, fin3_def_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.h …
  -/
  linear_combination -2 * P z ^ 2 * (equation_iff _).mp hP
  /-
    🎉 no goals
  -/


lemma negAddY_neg {P : Fin 3 → R} (hP : W'.Equation P) :
    W'.negAddY P (W'.neg P) = W'.dblZ P ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.negAddY P (W'.neg P)) (HPow.hPow (W'.dblZ P) 3)
  -/
  simp only [negAddY, neg, dblZ, negY, fin3_def_ext]
  linear_combination -2 * (2 * P y * P z ^ 3 + W'.a₁ * P x * P z ^ 4 + W'.a₃ * P z ^ 6)
    * (equation_iff _).mp hP


lemma addY_neg {P : Fin 3 → R} (hP : W'.Equation P) : W'.addY P (W'.neg P) = -W'.dblZ P ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addY P (W'.neg P)) (Neg.neg (HPow.hPow (W'.dblZ P) 3))
  -/
  rw [addY, addX_neg hP, negAddY_neg hP, negY_of_Z_eq_zero addZ_neg]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma addXYZ_neg {P : Fin 3 → R} (hP : W'.Equation P) :
    W'.addXYZ P (W'.neg P) = -W'.dblZ P • ![1, 1, 0] := by
  erw [addXYZ, addX_neg hP, addY_neg hP, addZ_neg, smul_fin3, neg_sq, mul_one,
    Odd.neg_pow <| by decide, mul_one, mul_zero]


variable (W') in
/-- The negation of a point class. If `P` is a point representative,
then `W'.negMap ⟦P⟧` is definitionally equivalent to `W'.neg P`. -/
def negMap (P : PointClass R) : PointClass R :=
  P.map W'.neg fun _ _ => neg_equiv


lemma negMap_eq {P : Fin 3 → R} : W'.negMap ⟦P⟧ = ⟦W'.neg P⟧ :=
  rfl


lemma negMap_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) :
    W.negMap ⟦P⟧ = ⟦![1, 1, 0]⟧ := by
  rw [negMap_eq, neg_of_Z_eq_zero hP hPz,
    smul_eq _ ((isUnit_Y_of_Z_eq_zero hP hPz).div <| isUnit_X_of_Z_eq_zero hP hPz).neg]


lemma negMap_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.negMap ⟦P⟧ = ⟦![P x / P z ^ 2, W.toAffine.negY (P x / P z ^ 2) (P y / P z ^ 3), 1]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    ⊢ Eq (W.negMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P)) (Qu …
  -/
  rw [negMap_eq, neg_of_Z_ne_zero hPz, smul_eq _ <| Ne.isUnit hPz]
  /-
    🎉 no goals
  -/


lemma nonsingularLift_negMap {P : PointClass F} (hP : W.NonsingularLift P) :
    W.NonsingularLift <| W.negMap P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : WeierstrassCurve.Jacobian.PointClass F
    hP : W.NonsingularLift P
    ⊢ W.NonsingularLift (W.negMap P)
  -/
  rcases P with ⟨_⟩
  /-
    case mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : WeierstrassCurve.Jacobian.PointClass F
    a✝ : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    ⊢ W.NonsingularLift (W.negMap (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3  …
  -/
  exact nonsingular_neg hP
  /-
    🎉 no goals
  -/


open Classical in
variable (W') in
/-- The addition of two point representatives. -/
noncomputable def add (P Q : Fin 3 → R) : Fin 3 → R :=
  if P ≈ Q then W'.dblXYZ P else W'.addXYZ P Q


lemma add_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.add P Q = W'.dblXYZ P :=
  if_pos h


lemma add_smul_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) {u v : R} (hu : IsUnit u) (hv : IsUnit v) :
    W'.add (u • P) (v • Q) = u ^ 4 • W'.add P Q := by
  have smul : P ≈ Q ↔ u • P ≈ v • Q := by
    erw [← Quotient.eq_iff_equiv, ← Quotient.eq_iff_equiv, smul_eq P hu, smul_eq Q hv]
    rfl
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    smul : Iff (HasEquiv.Equiv P Q) (HasEquiv.Equiv (HSMul.hSMul u P) (HSMul.hSMul …
    ⊢ Eq (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HSMul.hSMul (HPow.hPow u 4) …
  -/
  rw [add_of_equiv <| smul.mp h, dblXYZ_smul, add_of_equiv h]
  /-
    🎉 no goals
  -/


lemma add_self (P : Fin 3 → R) : W'.add P P = W'.dblXYZ P :=
  add_of_equiv <| Setoid.refl _


lemma add_of_eq {P Q : Fin 3 → R} (h : P = Q) : W'.add P Q = W'.dblXYZ P :=
  h ▸ add_self P


lemma add_of_not_equiv {P Q : Fin 3 → R} (h : ¬P ≈ Q) : W'.add P Q = W'.addXYZ P Q :=
  if_neg h


lemma add_smul_of_not_equiv {P Q : Fin 3 → R} (h : ¬P ≈ Q) {u v : R} (hu : IsUnit u)
    (hv : IsUnit v) : W'.add (u • P) (v • Q) = (u * v) ^ 2 • W'.add P Q := by
  have smul : P ≈ Q ↔ u • P ≈ v • Q := by
    erw [← Quotient.eq_iff_equiv, ← Quotient.eq_iff_equiv, smul_eq P hu, smul_eq Q hv]
    rfl
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : Not (HasEquiv.Equiv P Q)
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    smul : Iff (HasEquiv.Equiv P Q) (HasEquiv.Equiv (HSMul.hSMul u P) (HSMul.hSMul …
    ⊢ Eq (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HSMul.hSMul (HPow.hPow (HMu …
  -/
  rw [add_of_not_equiv <| h.comp smul.mpr, addXYZ_smul, add_of_not_equiv h]
  /-
    🎉 no goals
  -/


lemma add_smul_equiv (P Q : Fin 3 → R) {u v : R} (hu : IsUnit u) (hv : IsUnit v) :
    W'.add (u • P) (v • Q) ≈ W'.add P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    ⊢ HasEquiv.Equiv (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (W'.add P Q)
  -/
  by_cases h : P ≈ Q
    /-
      case pos
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P Q : Fin 3 → R
      u v : R
      hu : IsUnit u
      hv : IsUnit v
      h : HasEquiv.Equiv P Q
      ⊢ HasEquiv.Equiv (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (W'.add P Q)
    -/
  · exact ⟨hu.unit ^ 4, by convert (add_smul_of_equiv h hu hv).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      inst✝ : CommRing R
      P Q : Fin 3 → R
      u v : R
      hu : IsUnit u
      hv : IsUnit v
      h : Not (HasEquiv.Equiv P Q)
      ⊢ HasEquiv.Equiv (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (W'.add P Q)
    -/
  · exact ⟨(hu.unit * hv.unit) ^ 2, by convert (add_smul_of_not_equiv h hu hv).symm⟩
    /-
      🎉 no goals
    -/


lemma add_equiv {P P' Q Q' : Fin 3 → R} (hP : P ≈ P') (hQ : Q ≈ Q') :
    W'.add P Q ≈ W'.add P' Q' := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P P' Q Q' : Fin 3 → R
    hP : HasEquiv.Equiv P P'
    hQ : HasEquiv.Equiv Q Q'
    ⊢ HasEquiv.Equiv (W'.add P Q) (W'.add P' Q')
  -/
  rcases hP, hQ with ⟨⟨u, rfl⟩, ⟨v, rfl⟩⟩
  /-
    case intro.intro
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P' Q' : Fin 3 → R
    u v : Units R
    ⊢ HasEquiv.Equiv (W'.add ((fun m => HSMul.hSMul m P') u) ((fun m => HSMul.hSMu …
  -/
  exact add_smul_equiv P' Q' u.isUnit v.isUnit
  /-
    🎉 no goals
  -/


lemma add_of_Z_eq_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Nonsingular Q)
    (hPz : P z = 0) (hQz : Q z = 0) : W.add P Q = P x ^ 2 • ![1, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ Eq (W.add P Q) (HSMul.hSMul (HPow.hPow (P 0) 2) (Matrix.vecCons 1 (Matrix.ve …
  -/
  rw [add, if_pos <| equiv_of_Z_eq_zero hP hQ hPz hQz, dblXYZ_of_Z_eq_zero hP.left hPz]
  /-
    🎉 no goals
  -/


lemma add_of_Z_eq_zero_left {P Q : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) (hQz : Q z ≠ 0) :
    W'.add P Q = (P x * Q z) • Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W'.add P Q) (HSMul.hSMul (HMul.hMul (P 0) (Q 2)) Q)
  -/
  rw [add, if_neg <| not_equiv_of_Z_eq_zero_left hPz hQz, addXYZ_of_Z_eq_zero_left hP hPz]
  /-
    🎉 no goals
  -/


lemma add_of_Z_eq_zero_right {P Q : Fin 3 → R} (hQ : W'.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z = 0) : W'.add P Q = -(Q x * P z) • P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ Eq (W'.add P Q) (HSMul.hSMul (Neg.neg (HMul.hMul (Q 0) (P 2))) P)
  -/
  rw [add, if_neg <| not_equiv_of_Z_eq_zero_right hPz hQz, addXYZ_of_Z_eq_zero_right hQ hQz]
  /-
    🎉 no goals
  -/


lemma add_of_Y_eq {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 = Q y * P z ^ 3)
    (hy' : P y * Q z ^ 3 = W.negY Q * P z ^ 3) : W.add P Q = W.dblU P • ![1, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W.negY Q) (HPow.hPo …
    ⊢ Eq (W.add P Q) (HSMul.hSMul (W.dblU P) (Matrix.vecCons 1 (Matrix.vecCons 1 ( …
  -/
  rw [add, if_pos <| equiv_of_X_eq_of_Y_eq hPz hQz hx hy, dblXYZ_of_Y_eq hQz hx hy hy']
  /-
    🎉 no goals
  -/


lemma add_of_Y_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ Q y * P z ^ 3) :
    W.add P Q = addU P Q • ![1, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy : Ne (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (Q 1) (HPow.hPow (P 2 …
    ⊢ Eq (W.add P Q) (HSMul.hSMul (WeierstrassCurve.Jacobian.addU P Q) (Matrix.vec …
  -/
  rw [add, if_neg <| not_equiv_of_Y_ne hy, addXYZ_of_X_eq hP hQ hPz hQz hx]
  /-
    🎉 no goals
  -/


lemma add_of_Y_ne' {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2) (hy : P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    W.add P Q = W.dblZ P •
      ![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        1] := by
  rw [add, if_pos <| equiv_of_X_eq_of_Y_eq hPz hQz hx <| Y_eq_of_Y_ne' hP hQ hx hy,
    dblXYZ_of_Z_ne_zero hP hQ hPz hQz hx hy]


lemma add_of_X_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 ≠ Q x * P z ^ 2) :
    W.add P Q = addZ P Q •
      ![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    ⊢ Eq (W.add P Q) (HSMul.hSMul (WeierstrassCurve.Jacobian.addZ P Q) (Matrix.vec …
  -/
  rw [add, if_neg <| not_equiv_of_X_ne hx, addXYZ_of_Z_ne_zero hP hQ hPz hQz hx]
  /-
    🎉 no goals
  -/


private lemma nonsingular_add_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P)
    (hQ : W.Nonsingular Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hxy : P x * Q z ^ 2 = Q x * P z ^ 2 → P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) : W.Nonsingular
      ![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)), 1] :=
  (nonsingular_some ..).mpr <| Affine.nonsingular_add ((nonsingular_of_Z_ne_zero hPz).mp hP)
                                               /-
                                                 F : Type v
                                                 inst✝ : Field F
                                                 W : WeierstrassCurve.Jacobian F
                                                 P Q : Fin 3 → F
                                                 hP : W.Nonsingular P
                                                 hQ : W.Nonsingular Q
                                                 hPz : Ne (P 2) 0
                                                 hQz : Ne (Q 2) 0
                                                 hxy : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P  …
                                                 ⊢ Eq (HDiv.hDiv (P 0) (HPow.hPow (P 2) 2)) (HDiv.hDiv (Q 0) (HPow.hPow (Q 2) 2 …
                                               -/
    ((nonsingular_of_Z_ne_zero hQz).mp hQ) (by rwa [← X_eq_iff hPz hQz, ne_eq, ← Y_eq_iff' hPz hQz])
                                               /-
                                                 🎉 no goals
                                               -/


lemma nonsingular_add {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Nonsingular Q) :
    W.Nonsingular <| W.add P Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    ⊢ W.Nonsingular (W.add P Q)
  -/
  by_cases hPz : P z = 0
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Eq (P 2) 0
      ⊢ W.Nonsingular (W.add P Q)
    -/
  · by_cases hQz : Q z = 0
    · simp only [add_of_Z_eq_zero hP hQ hPz hQz,
        nonsingular_smul _ <| (isUnit_X_of_Z_eq_zero hP hPz).pow 2, nonsingular_zero]
    · simpa only [add_of_Z_eq_zero_left hP.left hPz hQz,
        nonsingular_smul _ <| (isUnit_X_of_Z_eq_zero hP hPz).mul <| Ne.isUnit hQz]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Not (Eq (P 2) 0)
      ⊢ W.Nonsingular (W.add P Q)
    -/
  · by_cases hQz : Q z = 0
    · simpa only [add_of_Z_eq_zero_right hQ.left hPz hQz,
        nonsingular_smul _ ((isUnit_X_of_Z_eq_zero hQ hQz).mul <| Ne.isUnit hPz).neg]
      /-
        case neg
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Jacobian F
        P Q : Fin 3 → F
        hP : W.Nonsingular P
        hQ : W.Nonsingular Q
        hPz : Not (Eq (P 2) 0)
        hQz : Not (Eq (Q 2) 0)
        ⊢ W.Nonsingular (W.add P Q)
      -/
    · by_cases hxy : P x * Q z ^ 2 = Q x * P z ^ 2 → P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3
        /-
          case pos
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Jacobian F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P  …
          ⊢ W.Nonsingular (W.add P Q)
        -/
      · by_cases hx : P x * Q z ^ 2 = Q x * P z ^ 2
        · simp only [add_of_Y_ne' hP.left hQ.left hPz hQz hx <| hxy hx,
            nonsingular_smul _ <| isUnit_dblZ_of_Y_ne' hP.left hQ.left hPz hx <| hxy hx,
            nonsingular_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        · simp only [add_of_X_ne hP.left hQ.left hPz hQz hx,
            nonsingular_smul _ <| isUnit_addZ_of_X_ne hx,
            nonsingular_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Jacobian F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Not (Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPo …
          ⊢ W.Nonsingular (W.add P Q)
        -/
      · rw [_root_.not_imp, not_ne_iff] at hxy
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Jacobian F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : And (Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPo …
          ⊢ W.Nonsingular (W.add P Q)
        -/
        by_cases hy : P y * Q z ^ 3 = Q y * P z ^ 3
        · simp only [add_of_Y_eq hPz hQz hxy.left hy hxy.right, nonsingular_smul _ <|
              isUnit_dblU_of_Y_eq hP hPz hQz hxy.left hy hxy.right, nonsingular_zero]
        · simp only [add_of_Y_ne hP.left hQ.left hPz hQz hxy.left hy,
            nonsingular_smul _ <| isUnit_addU_of_Y_ne hPz hQz hy, nonsingular_zero]


variable (W') in
/-- The addition of two point classes. If `P` is a point representative,
then `W.addMap ⟦P⟧ ⟦Q⟧` is definitionally equivalent to `W.add P Q`. -/
noncomputable def addMap (P Q : PointClass R) : PointClass R :=
  Quotient.map₂ W'.add (fun _ _ hP _ _ hQ => add_equiv hP hQ) P Q


lemma addMap_eq (P Q : Fin 3 → R) : W'.addMap ⟦P⟧ ⟦Q⟧ = ⟦W'.add P Q⟧ :=
  rfl


lemma addMap_of_Z_eq_zero_left {P : Fin 3 → F} {Q : PointClass F} (hP : W.Nonsingular P)
    (hQ : W.NonsingularLift Q) (hPz : P z = 0) : W.addMap ⟦P⟧ Q = Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    Q : WeierstrassCurve.Jacobian.PointClass F
    hP : W.Nonsingular P
    hQ : W.NonsingularLift Q
    hPz : Eq (P 2) 0
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) Q) Q
  -/
  rcases Q with ⟨Q⟩
  /-
    case mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    Q✝ : WeierstrassCurve.Jacobian.PointClass F
    hP : W.Nonsingular P
    hPz : Eq (P 2) 0
    Q : Fin 3 → F
    hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) Q)
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
  -/
  by_cases hQz : Q z = 0
  · erw [addMap_eq, add_of_Z_eq_zero hP hQ hPz hQz,
      smul_eq _ <| (isUnit_X_of_Z_eq_zero hP hPz).pow 2, Quotient.eq]
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P : Fin 3 → F
      Q✝ : WeierstrassCurve.Jacobian.PointClass F
      hP : W.Nonsingular P
      hPz : Eq (P 2) 0
      Q : Fin 3 → F
      hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) Q)
      hQz : Eq (Q 2) 0
      ⊢ (MulAction.orbitRel (Units F) (Fin 3 → F)) (Matrix.vecCons 1 (Matrix.vecCons …
    -/
    exact Setoid.symm <| equiv_zero_of_Z_eq_zero hQ hQz
    /-
      🎉 no goals
    -/
  · erw [addMap_eq, add_of_Z_eq_zero_left hP.left hPz hQz,
      smul_eq _ <| (isUnit_X_of_Z_eq_zero hP hPz).mul <| Ne.isUnit hQz]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P : Fin 3 → F
      Q✝ : WeierstrassCurve.Jacobian.PointClass F
      hP : W.Nonsingular P
      hPz : Eq (P 2) 0
      Q : Fin 3 → F
      hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) Q)
      hQz : Not (Eq (Q 2) 0)
      ⊢ Eq (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) Q) (Quot.mk (⇑(Mu …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma addMap_of_Z_eq_zero_right {P : PointClass F} {Q : Fin 3 → F} (hP : W.NonsingularLift P)
    (hQ : W.Nonsingular Q) (hQz : Q z = 0) : W.addMap P ⟦Q⟧ = P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : WeierstrassCurve.Jacobian.PointClass F
    Q : Fin 3 → F
    hP : W.NonsingularLift P
    hQ : W.Nonsingular Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (W.addMap P (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) Q)) P
  -/
  rcases P with ⟨P⟩
  /-
    case mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P✝ : WeierstrassCurve.Jacobian.PointClass F
    Q : Fin 3 → F
    hQ : W.Nonsingular Q
    hQz : Eq (Q 2) 0
    P : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
    ⊢ Eq (W.addMap (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P) (Quot …
  -/
  by_cases hPz : P z = 0
  · erw [addMap_eq, add_of_Z_eq_zero hP hQ hPz hQz,
      smul_eq _ <| (isUnit_X_of_Z_eq_zero hP hPz).pow 2, Quotient.eq]
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P✝ : WeierstrassCurve.Jacobian.PointClass F
      Q : Fin 3 → F
      hQ : W.Nonsingular Q
      hQz : Eq (Q 2) 0
      P : Fin 3 → F
      hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
      hPz : Eq (P 2) 0
      ⊢ (MulAction.orbitRel (Units F) (Fin 3 → F)) (Matrix.vecCons 1 (Matrix.vecCons …
    -/
    exact Setoid.symm <| equiv_zero_of_Z_eq_zero hP hPz
    /-
      🎉 no goals
    -/
  · erw [addMap_eq, add_of_Z_eq_zero_right hQ.left hPz hQz,
      smul_eq _ ((isUnit_X_of_Z_eq_zero hQ hQz).mul <| Ne.isUnit hPz).neg]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P✝ : WeierstrassCurve.Jacobian.PointClass F
      Q : Fin 3 → F
      hQ : W.Nonsingular Q
      hQz : Eq (Q 2) 0
      P : Fin 3 → F
      hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
      hPz : Not (Eq (P 2) 0)
      ⊢ Eq (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quot.mk (⇑(Mu …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma addMap_of_Y_eq {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ^ 2 = Q x * P z ^ 2)
    (hy' : P y * Q z ^ 3 = W.negY Q * P z ^ 3) : W.addMap ⟦P⟧ ⟦Q⟧ = ⟦![1, 1, 0]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P 2 …
    hy' : Eq (HMul.hMul (P 1) (HPow.hPow (Q 2) 3)) (HMul.hMul (W.negY Q) (HPow.hPo …
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
  -/
  by_cases hy : P y * Q z ^ 3 = Q y * P z ^ 3
  · rw [addMap_eq, add_of_Y_eq hPz hQz hx hy hy',
      smul_eq _ <| isUnit_dblU_of_Y_eq hP hPz hQz hx hy hy']
  · rw [addMap_eq, add_of_Y_ne hP.left hQ hPz hQz hx hy,
      smul_eq _ <| isUnit_addU_of_Y_ne hPz hQz hy]


lemma addMap_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hxy : P x * Q z ^ 2 = Q x * P z ^ 2 → P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    W.addMap ⟦P⟧ ⟦Q⟧ =
      ⟦![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        1]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hxy : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P  …
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
  -/
  by_cases hx : P x * Q z ^ 2 = Q x * P z ^ 2
  · rw [addMap_eq, add_of_Y_ne' hP hQ hPz hQz hx <| hxy hx,
      smul_eq _ <| isUnit_dblZ_of_Y_ne' hP hQ hPz hx <| hxy hx]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P Q : Fin 3 → F
      hP : W.Equation P
      hQ : W.Equation Q
      hPz : Ne (P 2) 0
      hQz : Ne (Q 2) 0
      hxy : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P  …
      hx : Not (Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow …
      ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
    -/
  · rw [addMap_eq, add_of_X_ne hP hQ hPz hQz hx, smul_eq _ <| isUnit_addZ_of_X_ne hx]
    /-
      🎉 no goals
    -/


lemma nonsingularLift_addMap {P Q : PointClass F} (hP : W.NonsingularLift P)
    (hQ : W.NonsingularLift Q) : W.NonsingularLift <| W.addMap P Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : WeierstrassCurve.Jacobian.PointClass F
    hP : W.NonsingularLift P
    hQ : W.NonsingularLift Q
    ⊢ W.NonsingularLift (W.addMap P Q)
  -/
  rcases P; rcases Q
  /-
    case mk.mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : WeierstrassCurve.Jacobian.PointClass F
    a✝¹ : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    a✝ : Fin 3 → F
    hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    ⊢ W.NonsingularLift (W.addMap (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3  …
  -/
  exact nonsingular_add hP hQ
  /-
    🎉 no goals
  -/


variable (W') in
/-- A nonsingular rational point on `W'`. -/
@[ext]
structure Point where
  /-- The point class underlying a nonsingular rational point on `W'`. -/
  {point : PointClass R}
  /-- The nonsingular condition underlying a nonsingular rational point on `W'`. -/
  (nonsingular : W'.NonsingularLift point)


lemma mk_point {P : PointClass R} (h : W'.NonsingularLift P) : (mk h).point = P :=
  rfl


instance instZeroPoint [Nontrivial R] : Zero W'.Point :=
  ⟨⟨nonsingularLift_zero⟩⟩


lemma zero_def [Nontrivial R] : (0 : W'.Point) = ⟨nonsingularLift_zero⟩ :=
  rfl


lemma zero_point [Nontrivial R] : (0 : W'.Point).point = ⟦![1, 1, 0]⟧ :=
  rfl


/-- The map from a nonsingular rational point on a Weierstrass curve `W'` in affine coordinates
to the corresponding nonsingular rational point on `W'` in Jacobian coordinates. -/
def fromAffine [Nontrivial R] : W'.toAffine.Point → W'.Point
  | 0 => 0
  | .some h => ⟨(nonsingularLift_some ..).mpr h⟩


lemma fromAffine_zero [Nontrivial R] : fromAffine 0 = (0 : W'.Point) :=
  rfl


lemma fromAffine_some [Nontrivial R] {X Y : R} (h : W'.toAffine.Nonsingular X Y) :
    fromAffine (.some h) = ⟨(nonsingularLift_some ..).mpr h⟩ :=
  rfl


lemma fromAffine_ne_zero [Nontrivial R] {X Y : R} (h : W'.toAffine.Nonsingular X Y) :
    fromAffine (.some h) ≠ 0 := fun h0 ↦ by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    X Y : R
    h : W'.toAffine.Nonsingular X Y
    h0 : Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Affine.P …
    ⊢ False
  -/
  obtain ⟨u, eq⟩ := Quotient.eq.mp <| (Point.ext_iff ..).mp h0
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    X Y : R
    h : W'.toAffine.Nonsingular X Y
    h0 : Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Affine.P …
    u : Units R
    eq : Eq ((fun m => HSMul.hSMul m (Matrix.vecCons 1 (Matrix.vecCons 1 (Matrix.v …
    ⊢ False
  -/
  simpa [Units.smul_def, smul_fin3] using congr_fun eq z
  /-
    🎉 no goals
  -/


/-- The negation of a nonsingular rational point on `W`.
Given a nonsingular rational point `P` on `W`, use `-P` instead of `neg P`. -/
def neg (P : W.Point) : W.Point :=
  ⟨nonsingularLift_negMap P.nonsingular⟩


instance instNegPoint : Neg W.Point :=
  ⟨neg⟩


lemma neg_def (P : W.Point) : -P = P.neg :=
  rfl


lemma neg_point (P : W.Point) : (-P).point = W.negMap P.point :=
  rfl


/-- The addition of two nonsingular rational points on `W`.
Given two nonsingular rational points `P` and `Q` on `W`, use `P + Q` instead of `add P Q`. -/
noncomputable def add (P Q : W.Point) : W.Point :=
  ⟨nonsingularLift_addMap P.nonsingular Q.nonsingular⟩


noncomputable instance instAddPoint : Add W.Point :=
  ⟨add⟩


lemma add_def (P Q : W.Point) : P + Q = P.add Q :=
  rfl


lemma add_point (P Q : W.Point) : (P + Q).point = W.addMap P.point Q.point :=
  rfl


open Classical in
variable (W) in
/-- The map from a point representative that is nonsingular on a Weierstrass curve `W` in Jacobian
coordinates to the corresponding nonsingular rational point on `W` in affine coordinates. -/
noncomputable def toAffine (P : Fin 3 → F) : W.toAffine.Point :=
  if hP : W.Nonsingular P ∧ P z ≠ 0 then .some <| (nonsingular_of_Z_ne_zero hP.2).mp hP.1 else 0


lemma toAffine_of_singular {P : Fin 3 → F} (hP : ¬W.Nonsingular P) : toAffine W P = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hP : Not (W.Nonsingular P)
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W P) 0
  -/
  rw [toAffine, dif_neg <| not_and_of_not_left _ hP]
  /-
    🎉 no goals
  -/


lemma toAffine_of_Z_eq_zero {P : Fin 3 → F} (hPz : P z = 0) :
    toAffine W P = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hPz : Eq (P 2) 0
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W P) 0
  -/
  rw [toAffine, dif_neg <| not_and_not_right.mpr fun _ => hPz]
  /-
    🎉 no goals
  -/


lemma toAffine_zero : toAffine W ![1, 1, 0] = 0 :=
  toAffine_of_Z_eq_zero rfl


lemma toAffine_of_Z_ne_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) :
    toAffine W P = .some ((nonsingular_of_Z_ne_zero hPz).mp hP) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hP : W.Nonsingular P
    hPz : Ne (P 2) 0
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W P) (WeierstrassCurve.Affine.P …
  -/
  rw [toAffine, dif_pos ⟨hP, hPz⟩]
  /-
    🎉 no goals
  -/


lemma toAffine_some {X Y : F} (h : W.Nonsingular ![X, Y, 1]) :
    toAffine W ![X, Y, 1] = .some ((nonsingular_some ..).mp h) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    X Y : F
    h : W.Nonsingular (Matrix.vecCons X (Matrix.vecCons Y (Matrix.vecCons 1 Matrix …
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (Matrix.vecCons X (Matrix.vec …
  -/
  simp only [toAffine_of_Z_ne_zero h one_ne_zero, fin3_def_ext, one_pow, div_one]
  /-
    🎉 no goals
  -/


lemma toAffine_smul (P : Fin 3 → F) {u : F} (hu : IsUnit u) :
    toAffine W (u • P) = toAffine W P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    u : F
    hu : IsUnit u
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (HSMul.hSMul u P)) (Weierstra …
  -/
  by_cases hP : W.Nonsingular P
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P : Fin 3 → F
      u : F
      hu : IsUnit u
      hP : W.Nonsingular P
      ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (HSMul.hSMul u P)) (Weierstra …
    -/
  · by_cases hPz : P z = 0
      /-
        case pos
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Jacobian F
        P : Fin 3 → F
        u : F
        hu : IsUnit u
        hP : W.Nonsingular P
        hPz : Eq (P 2) 0
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (HSMul.hSMul u P)) (Weierstra …
      -/
    · rw [toAffine_of_Z_eq_zero <| mul_eq_zero_of_right u hPz, toAffine_of_Z_eq_zero hPz]
      /-
        🎉 no goals
      -/
    · rw [toAffine_of_Z_ne_zero ((nonsingular_smul P hu).mpr hP) <| mul_ne_zero hu.ne_zero hPz,
        toAffine_of_Z_ne_zero hP hPz, Affine.Point.some.injEq]
      /-
        case neg
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Jacobian F
        P : Fin 3 → F
        u : F
        hu : IsUnit u
        hP : W.Nonsingular P
        hPz : Not (Eq (P 2) 0)
        ⊢ And (Eq (HDiv.hDiv (HSMul.hSMul u P 0) (HPow.hPow (HSMul.hSMul u P 2) 2)) (H …
      -/
      simp only [smul_fin3_ext, mul_pow, mul_div_mul_left _ _ (hu.pow _).ne_zero, and_self]
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P : Fin 3 → F
      u : F
      hu : IsUnit u
      hP : Not (W.Nonsingular P)
      ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (HSMul.hSMul u P)) (Weierstra …
    -/
  · rw [toAffine_of_singular <| hP.comp (nonsingular_smul P hu).mp, toAffine_of_singular hP]
    /-
      🎉 no goals
    -/


lemma toAffine_of_equiv {P Q : Fin 3 → F} (h : P ≈ Q) : toAffine W P = toAffine W Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    h : HasEquiv.Equiv P Q
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W P) (WeierstrassCurve.Jacobian …
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    Q : Fin 3 → F
    u : Units F
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W ((fun m => HSMul.hSMul m Q) u …
  -/
  exact toAffine_smul Q u.isUnit
  /-
    🎉 no goals
  -/


lemma toAffine_neg {P : Fin 3 → F} (hP : W.Nonsingular P) :
    toAffine W (W.neg P) = -toAffine W P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : Fin 3 → F
    hP : W.Nonsingular P
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.neg P)) (Neg.neg (Weierstr …
  -/
  by_cases hPz : P z = 0
  · rw [neg_of_Z_eq_zero hP hPz,
      toAffine_smul _ ((isUnit_Y_of_Z_eq_zero hP hPz).div <| isUnit_X_of_Z_eq_zero hP hPz).neg,
      toAffine_zero, toAffine_of_Z_eq_zero hPz, Affine.Point.neg_zero]
  · rw [neg_of_Z_ne_zero hPz, toAffine_smul _ <| Ne.isUnit hPz, toAffine_some <|
        (nonsingular_smul _ <| Ne.isUnit hPz).mp <| neg_of_Z_ne_zero hPz ▸ nonsingular_neg hP,
      toAffine_of_Z_ne_zero hP hPz, Affine.Point.neg_some]


private lemma toAffine_add_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P)
    (hQ : W.Nonsingular Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hxy : P x * Q z ^ 2 = Q x * P z ^ 2 → P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3) :
    toAffine W
      ![W.toAffine.addX (P x / P z ^ 2) (Q x / Q z ^ 2)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        W.toAffine.addY (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3)
          (W.toAffine.slope (P x / P z ^ 2) (Q x / Q z ^ 2) (P y / P z ^ 3) (Q y / Q z ^ 3)),
        1] = toAffine W P + toAffine W Q := by
  rw [toAffine_some <| nonsingular_add_of_Z_ne_zero hP hQ hPz hQz hxy, toAffine_of_Z_ne_zero hP hPz,
    toAffine_of_Z_ne_zero hQ hQz,
    Affine.Point.add_of_imp <| by rwa [← X_eq_iff hPz hQz, ne_eq, ← Y_eq_iff' hPz hQz]]


lemma toAffine_add {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Nonsingular Q) :
    toAffine W (W.add P Q) = toAffine W P + toAffine W Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (HAdd.hAdd (Weie …
  -/
  by_cases hPz : P z = 0
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Eq (P 2) 0
      ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (HAdd.hAdd (Weie …
    -/
  · rw [toAffine_of_Z_eq_zero hPz, zero_add]
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Eq (P 2) 0
      ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (WeierstrassCurv …
    -/
    by_cases hQz : Q z = 0
    · rw [add_of_Z_eq_zero hP hQ hPz hQz, toAffine_smul _ <| (isUnit_X_of_Z_eq_zero hP hPz).pow 2,
        toAffine_zero, toAffine_of_Z_eq_zero hQz]
    · rw [add_of_Z_eq_zero_left hP.left hPz hQz,
        toAffine_smul _ <| (isUnit_X_of_Z_eq_zero hP hPz).mul <| Ne.isUnit hQz]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Jacobian F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Not (Eq (P 2) 0)
      ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (HAdd.hAdd (Weie …
    -/
  · by_cases hQz : Q z = 0
    · rw [add_of_Z_eq_zero_right hQ.left hPz hQz,
        toAffine_smul _ ((isUnit_X_of_Z_eq_zero hQ hQz).mul <| Ne.isUnit hPz).neg,
        toAffine_of_Z_eq_zero hQz, add_zero]
      /-
        case neg
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Jacobian F
        P Q : Fin 3 → F
        hP : W.Nonsingular P
        hQ : W.Nonsingular Q
        hPz : Not (Eq (P 2) 0)
        hQz : Not (Eq (Q 2) 0)
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (HAdd.hAdd (Weie …
      -/
    · by_cases hxy : P x * Q z ^ 2 = Q x * P z ^ 2 → P y * Q z ^ 3 ≠ W.negY Q * P z ^ 3
        /-
          case pos
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Jacobian F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPow (P  …
          ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (HAdd.hAdd (Weie …
        -/
      · by_cases hx : P x * Q z ^ 2 = Q x * P z ^ 2
        · rw [add_of_Y_ne' hP.left hQ.left hPz hQz hx <| hxy hx,
            toAffine_smul _ <| isUnit_dblZ_of_Y_ne' hP.left hQ.left hPz hx <| hxy hx,
            toAffine_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        · rw [add_of_X_ne hP.left hQ.left hPz hQz hx, toAffine_smul _ <| isUnit_addZ_of_X_ne hx,
            toAffine_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Jacobian F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Not (Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPo …
          ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) (HAdd.hAdd (Weie …
        -/
      · rw [_root_.not_imp, not_ne_iff] at hxy
        rw [toAffine_of_Z_ne_zero hP hPz, toAffine_of_Z_ne_zero hQ hQz, Affine.Point.add_of_Y_eq
            ((X_eq_iff hPz hQz).mp hxy.left) ((Y_eq_iff' hPz hQz).mp hxy.right)]
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Jacobian F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : And (Eq (HMul.hMul (P 0) (HPow.hPow (Q 2) 2)) (HMul.hMul (Q 0) (HPow.hPo …
          ⊢ Eq (WeierstrassCurve.Jacobian.Point.toAffine W (W.add P Q)) 0
        -/
        by_cases hy : P y * Q z ^ 3 = Q y * P z ^ 3
        · rw [add_of_Y_eq hPz hQz hxy.left hy hxy.right,
            toAffine_smul _ <| isUnit_dblU_of_Y_eq hP hPz hQz hxy.left hy hxy.right, toAffine_zero]
        · rw [add_of_Y_ne hP.left hQ.left hPz hQz hxy.left hy,
            toAffine_smul _ <| isUnit_addU_of_Y_ne hPz hQz hy, toAffine_zero]


/-- The map from a nonsingular rational point on a Weierstrass curve `W` in Jacobian coordinates
to the corresponding nonsingular rational point on `W` in affine coordinates. -/
noncomputable def toAffineLift (P : W.Point) : W.toAffine.Point :=
  P.point.lift _ fun _ _ => toAffine_of_equiv


lemma toAffineLift_eq {P : Fin 3 → F} (hP : W.NonsingularLift ⟦P⟧) :
    toAffineLift ⟨hP⟩ = toAffine W P :=
  rfl


lemma toAffineLift_of_Z_eq_zero {P : Fin 3 → F} (hP : W.NonsingularLift ⟦P⟧) (hPz : P z = 0) :
    toAffineLift ⟨hP⟩ = 0 :=
  toAffine_of_Z_eq_zero hPz


lemma toAffineLift_zero : toAffineLift (0 : W.Point) = 0 :=
  toAffine_zero


lemma toAffineLift_of_Z_ne_zero {P : Fin 3 → F} {hP : W.NonsingularLift ⟦P⟧} (hPz : P z ≠ 0) :
    toAffineLift ⟨hP⟩ = .some ((nonsingular_of_Z_ne_zero hPz).mp hP) :=
  toAffine_of_Z_ne_zero hP hPz


lemma toAffineLift_some {X Y : F} (h : W.NonsingularLift ⟦![X, Y, 1]⟧) :
    toAffineLift ⟨h⟩ = .some ((nonsingular_some ..).mp h) :=
  toAffine_some h


lemma toAffineLift_neg (P : W.Point) : (-P).toAffineLift = -P.toAffineLift := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P : W.Point
    ⊢ Eq (Neg.neg P).toAffineLift (Neg.neg P.toAffineLift)
  -/
  rcases P with @⟨⟨_⟩, hP⟩
  /-
    case mk.mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    point✝ : WeierstrassCurve.Jacobian.PointClass F
    a✝ : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    ⊢ Eq (Neg.neg (WeierstrassCurve.Jacobian.Point.mk hP)).toAffineLift (Neg.neg ( …
  -/
  exact toAffine_neg hP
  /-
    🎉 no goals
  -/


lemma toAffineLift_add (P Q : W.Point) :
    (P + Q).toAffineLift = P.toAffineLift + Q.toAffineLift := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    P Q : W.Point
    ⊢ Eq (HAdd.hAdd P Q).toAffineLift (HAdd.hAdd P.toAffineLift Q.toAffineLift)
  -/
  rcases P, Q with ⟨@⟨⟨_⟩, hP⟩, @⟨⟨_⟩, hQ⟩⟩
  /-
    case mk.mk.mk.mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Jacobian F
    point✝¹ : WeierstrassCurve.Jacobian.PointClass F
    a✝¹ : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    point✝ : WeierstrassCurve.Jacobian.PointClass F
    a✝ : Fin 3 → F
    hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    ⊢ Eq (HAdd.hAdd (WeierstrassCurve.Jacobian.Point.mk hP) (WeierstrassCurve.Jaco …
  -/
  exact toAffine_add hP hQ
  /-
    🎉 no goals
  -/


variable (W) in
/-- The equivalence between the nonsingular rational points on a Weierstrass curve `W` in Jacobian
coordinates with the nonsingular rational points on `W` in affine coordinates. -/
@[simps]
noncomputable def toAffineAddEquiv : W.Point ≃+ W.toAffine.Point where
  toFun := toAffineLift
  invFun := fromAffine
  left_inv := by
    /-
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Jacobian F
      inst✝ : CommRing R
      ⊢ Function.LeftInverse WeierstrassCurve.Jacobian.Point.fromAffine WeierstrassC …
    -/
    rintro @⟨⟨P⟩, hP⟩
    /-
      case mk.mk
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Jacobian F
      inst✝ : CommRing R
      point✝ : WeierstrassCurve.Jacobian.PointClass F
      P : Fin 3 → F
      hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
      ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Jacobian.Po …
    -/
    by_cases hPz : P z = 0
      /-
        case pos
        R : Type u
        W' : WeierstrassCurve.Jacobian R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Jacobian F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Jacobian.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Eq (P 2) 0
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Jacobian.Po …
      -/
    · rw [Point.ext_iff, toAffineLift_eq, toAffine_of_Z_eq_zero hPz]
      /-
        case pos
        R : Type u
        W' : WeierstrassCurve.Jacobian R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Jacobian F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Jacobian.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Eq (P 2) 0
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine 0).point (WeierstrassCurve.Ja …
      -/
      exact Quotient.eq.mpr <| Setoid.symm <| equiv_zero_of_Z_eq_zero hP hPz
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        W' : WeierstrassCurve.Jacobian R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Jacobian F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Jacobian.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Not (Eq (P 2) 0)
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Jacobian.Po …
      -/
    · rw [Point.ext_iff, toAffineLift_eq, toAffine_of_Z_ne_zero hP hPz]
      /-
        case neg
        R : Type u
        W' : WeierstrassCurve.Jacobian R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Jacobian F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Jacobian.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Not (Eq (P 2) 0)
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Affine.Poin …
      -/
      exact Quotient.eq.mpr <| Setoid.symm <| equiv_some_of_Z_ne_zero hPz
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      R : Type u
      W' : WeierstrassCurve.Jacobian R
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Jacobian F
      inst✝ : CommRing R
      ⊢ Function.RightInverse WeierstrassCurve.Jacobian.Point.fromAffine Weierstrass …
    -/
    rintro (_ | _)
      /-
        case zero
        R : Type u
        W' : WeierstrassCurve.Jacobian R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Jacobian F
        inst✝ : CommRing R
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine WeierstrassCurve.Affine.Point …
      -/
    · erw [fromAffine_zero, toAffineLift_zero, Affine.Point.zero_def]
      /-
        🎉 no goals
      -/
      /-
        case some
        R : Type u
        W' : WeierstrassCurve.Jacobian R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Jacobian F
        inst✝ : CommRing R
        x✝ y✝ : F
        h✝ : W.toAffine.Nonsingular x✝ y✝
        ⊢ Eq (WeierstrassCurve.Jacobian.Point.fromAffine (WeierstrassCurve.Affine.Poin …
      -/
    · rw [fromAffine_some, toAffineLift_some]
      /-
        🎉 no goals
      -/
  map_add' := toAffineLift_add


protected lemma map_smul (u : R) : f ∘ (u • P) = f u • (f ∘ P) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    u : R
    ⊢ Eq (Function.comp (⇑f) (HSMul.hSMul u P)) (HSMul.hSMul (f u) (Function.comp  …
  -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
  ext i; fin_cases i <;> simp [smul_fin3]
                         /-
                           🎉 no goals
                         -/


                                                                   /-
                                                                     R : Type u
                                                                     inst✝¹ : CommRing R
                                                                     S : Type u_1
                                                                     inst✝ : CommRing S
                                                                     f : RingHom R S
                                                                     P Q : Fin 3 → R
                                                                     ⊢ Eq (WeierstrassCurve.Jacobian.addZ (Function.comp (⇑f) P) (Function.comp (⇑f …
                                                                   -/
@[simp] lemma map_addZ : addZ (f ∘ P) (f ∘ Q) = f (addZ P Q) := by simp [addZ]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

@[simp] lemma map_addX : addX (W'.map f) (f ∘ P) (f ∘ Q) = f (W'.addX P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.addX (WeierstrassCurve.map W' f) (Function.com …
  -/
  simp [map_ofNat, addX]
  /-
    🎉 no goals
  -/

@[simp] lemma map_negAddY : negAddY (W'.map f) (f ∘ P) (f ∘ Q) = f (W'.negAddY P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.negAddY (WeierstrassCurve.map W' f) (Function. …
  -/
  simp [map_ofNat, negAddY]
  /-
    🎉 no goals
  -/

                                                                       /-
                                                                         R : Type u
                                                                         W' : WeierstrassCurve.Jacobian R
                                                                         inst✝¹ : CommRing R
                                                                         S : Type u_1
                                                                         inst✝ : CommRing S
                                                                         f : RingHom R S
                                                                         P : Fin 3 → R
                                                                         ⊢ Eq (WeierstrassCurve.Jacobian.negY (WeierstrassCurve.map W' f) (Function.com …
                                                                       -/
@[simp] lemma map_negY : negY (W'.map f) (f ∘ P) = f (W'.negY P) := by simp [negY]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp] protected lemma map_neg : neg (W'.map f) (f ∘ P) = f ∘ W'.neg P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.neg (WeierstrassCurve.map W' f) (Function.comp …
  -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
  ext i; fin_cases i <;> simp [neg]
                         /-
                           🎉 no goals
                         -/


@[simp] lemma map_addY : addY (W'.map f) (f ∘ P) (f ∘ Q) = f (W'.addY P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.addY (WeierstrassCurve.map W' f) (Function.com …
  -/
  simp [addY, ← comp_fin3]
  /-
    🎉 no goals
  -/


@[simp] lemma map_addXYZ : addXYZ (W'.map f) (f ∘ P) (f ∘ Q) = f ∘ addXYZ W' P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.addXYZ (WeierstrassCurve.map W' f) (Function.c …
  -/
  simp_rw [addXYZ, comp_fin3, map_addX, map_addY, map_addZ]
  /-
    🎉 no goals
  -/


lemma map_polynomial : (W'.map f).toJacobian.polynomial = MvPolynomial.map f W'.polynomial := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toJacobian.polynomial ((MvPolynomial.map f) W …
  -/
  simp [polynomial]
  /-
    🎉 no goals
  -/


lemma map_polynomialX : (W'.map f).toJacobian.polynomialX = MvPolynomial.map f W'.polynomialX := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toJacobian.polynomialX ((MvPolynomial.map f)  …
  -/
  simp [polynomialX, map_polynomial, pderiv_map]
  /-
    🎉 no goals
  -/


lemma map_polynomialY : (W'.map f).toJacobian.polynomialY = MvPolynomial.map f W'.polynomialY := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toJacobian.polynomialY ((MvPolynomial.map f)  …
  -/
  simp [polynomialY, map_polynomial, pderiv_map]
  /-
    🎉 no goals
  -/


lemma map_polynomialZ : (W'.map f).toJacobian.polynomialZ = MvPolynomial.map f W'.polynomialZ := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toJacobian.polynomialZ ((MvPolynomial.map f)  …
  -/
  simp [polynomialZ, map_polynomial, pderiv_map]
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         R : Type u
                                                                         W' : WeierstrassCurve.Jacobian R
                                                                         inst✝¹ : CommRing R
                                                                         S : Type u_1
                                                                         inst✝ : CommRing S
                                                                         f : RingHom R S
                                                                         P : Fin 3 → R
                                                                         ⊢ Eq (WeierstrassCurve.Jacobian.dblZ (WeierstrassCurve.map W' f) (Function.com …
                                                                       -/
@[simp] lemma map_dblZ : dblZ (W'.map f) (f ∘ P) = f (W'.dblZ P) := by simp [dblZ]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

@[simp] lemma map_dblU : dblU (W'.map f) (f ∘ P) = f (W'.dblU P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.dblU (WeierstrassCurve.map W' f) (Function.com …
  -/
  simp [dblU, map_polynomialX, ← eval₂_id, eval₂_comp_left]
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         R : Type u
                                                                         W' : WeierstrassCurve.Jacobian R
                                                                         inst✝¹ : CommRing R
                                                                         S : Type u_1
                                                                         inst✝ : CommRing S
                                                                         f : RingHom R S
                                                                         P : Fin 3 → R
                                                                         ⊢ Eq (WeierstrassCurve.Jacobian.dblX (WeierstrassCurve.map W' f) (Function.com …
                                                                       -/
@[simp] lemma map_dblX : dblX (W'.map f) (f ∘ P) = f (W'.dblX P) := by simp [map_ofNat, dblX]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

                                                                                /-
                                                                                  R : Type u
                                                                                  W' : WeierstrassCurve.Jacobian R
                                                                                  inst✝¹ : CommRing R
                                                                                  S : Type u_1
                                                                                  inst✝ : CommRing S
                                                                                  f : RingHom R S
                                                                                  P : Fin 3 → R
                                                                                  ⊢ Eq (WeierstrassCurve.Jacobian.negDblY (WeierstrassCurve.map W' f) (Function. …
                                                                                -/
@[simp] lemma map_negDblY : negDblY (W'.map f) (f ∘ P) = f (W'.negDblY P) := by simp [negDblY]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                       /-
                                                                         R : Type u
                                                                         W' : WeierstrassCurve.Jacobian R
                                                                         inst✝¹ : CommRing R
                                                                         S : Type u_1
                                                                         inst✝ : CommRing S
                                                                         f : RingHom R S
                                                                         P : Fin 3 → R
                                                                         ⊢ Eq (WeierstrassCurve.Jacobian.dblY (WeierstrassCurve.map W' f) (Function.com …
                                                                       -/
@[simp] lemma map_dblY : dblY (W'.map f) (f ∘ P) = f (W'.dblY P) := by simp [dblY, ← comp_fin3]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp] lemma map_dblXYZ : dblXYZ (W'.map f) (f ∘ P) = f ∘ dblXYZ W' P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Jacobian R
    inst✝¹ : CommRing R
    S : Type u_1
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (WeierstrassCurve.Jacobian.dblXYZ (WeierstrassCurve.map W' f) (Function.c …
  -/
  simp_rw [dblXYZ, comp_fin3, map_dblX, map_dblY, map_dblZ]
  /-
    🎉 no goals
  -/


/-- An abbreviation for `WeierstrassCurve.Jacobian.Point.fromAffine` for dot notation. -/
abbrev WeierstrassCurve.Affine.Point.toJacobian {R : Type u} [CommRing R]
    [Nontrivial R] {W : Affine R} (P : W.Point) : W.toJacobian.Point :=
  Jacobian.Point.fromAffine P


