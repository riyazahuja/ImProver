local notation3 "x" => (0 : Fin 3)


local notation3 "y" => (1 : Fin 3)


local notation3 "z" => (2 : Fin 3)


local macro "matrix_simp" : tactic =>
  `(tactic| simp only [Matrix.head_cons, Matrix.tail_cons, Matrix.smul_empty, Matrix.smul_cons,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two])


/-- An abbreviation for a Weierstrass curve in projective coordinates. -/
abbrev WeierstrassCurve.Projective (R : Type u) : Type u :=
  WeierstrassCurve R


/-- The coercion to a Weierstrass curve in projective coordinates. -/
abbrev WeierstrassCurve.toProjective {R : Type u} (W : WeierstrassCurve R) : Projective R :=
  W


local macro "eval_simp" : tactic =>
  `(tactic| simp only [eval_C, eval_X, eval_add, eval_sub, eval_mul, eval_pow])


local macro "map_simp" : tactic =>
  `(tactic| simp only [map_ofNat, map_C, map_X, map_neg, map_add, map_sub, map_mul, map_pow,
    map_div₀, WeierstrassCurve.map, Function.comp_apply])


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


lemma comp_fin3 {S : Type v} (f : R → S) (X Y Z : R) : f ∘ ![X, Y, Z] = ![f X, f Y, f Z] :=
  (FinVec.map_eq ..).symm


lemma smul_fin3 (P : Fin 3 → R) (u : R) : u • P = ![u * P x, u * P y, u * P z] := by
  /-
    R : Type u
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSMul.hSMul u P) (Matrix.vecCons (HMul.hMul u (P 0)) (Matrix.vecCons (HM …
  -/
  simp [← List.ofFn_inj]
  /-
    🎉 no goals
  -/


lemma smul_fin3_ext (P : Fin 3 → R) (u : R) :
    (u • P) x = u * P x ∧ (u • P) y = u * P y ∧ (u • P) z = u * P z :=
  ⟨rfl, rfl, rfl⟩


lemma comp_smul {S : Type v} [CommRing S] (f : R →+* S) (P : Fin 3 → R) (u : R) :
    f ∘ (u • P) = f u • f ∘ P := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
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
  ext n; fin_cases n <;> simp only [smul_fin3, comp_fin3] <;> map_simp
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


lemma smul_equiv_smul (P Q : Fin 3 → R) {u v : R} (hu : IsUnit u) (hv : IsUnit v) :
    u • P ≈ v • Q ↔ P ≈ Q := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    ⊢ Iff (HasEquiv.Equiv (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HasEquiv.Equiv P Q)
  -/
  erw [← Quotient.eq_iff_equiv, ← Quotient.eq_iff_equiv, smul_eq P hu, smul_eq Q hv]
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    ⊢ Iff (Eq (Quotient.mk (MulAction.orbitRel (Units R) (Fin 3 → R)) P) (Quotient …
  -/
  rfl
  /-
    🎉 no goals
  -/


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


lemma X_eq_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : P x * Q z = Q x * P z := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul ((fun m => HSMul.hSMul m Q) u 0) (Q 2)) (HMul.hMul (Q 0) ((fun …
  -/
  simp only [Units.smul_def, smul_fin3_ext]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul (HMul.hMul (↑u) (Q 0)) (Q 2)) (HMul.hMul (Q 0) (HMul.hMul (↑u) …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma Y_eq_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : P y * Q z = Q y * P z := by
  /-
    R : Type u
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul ((fun m => HSMul.hSMul m Q) u 1) (Q 2)) (HMul.hMul (Q 1) ((fun …
  -/
  simp only [Units.smul_def, smul_fin3_ext]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Eq (HMul.hMul (HMul.hMul (↑u) (Q 1)) (Q 2)) (HMul.hMul (Q 1) (HMul.hMul (↑u) …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma not_equiv_of_Z_eq_zero_left {P Q : Fin 3 → R} (hPz : P z = 0) (hQz : Q z ≠ 0) : ¬P ≈ Q :=
  fun h => hQz <| (Z_eq_zero_of_equiv h).mp hPz


lemma not_equiv_of_Z_eq_zero_right {P Q : Fin 3 → R} (hPz : P z ≠ 0) (hQz : Q z = 0) : ¬P ≈ Q :=
  fun h => hPz <| (Z_eq_zero_of_equiv h).mpr hQz


lemma not_equiv_of_X_ne {P Q : Fin 3 → R} (hx : P x * Q z ≠ Q x * P z) : ¬P ≈ Q :=
  hx.comp X_eq_of_equiv


lemma not_equiv_of_Y_ne {P Q : Fin 3 → R} (hy : P y * Q z ≠ Q y * P z) : ¬P ≈ Q :=
  hy.comp Y_eq_of_equiv


lemma equiv_of_X_eq_of_Y_eq {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) : P ≈ Q := by
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    ⊢ HasEquiv.Equiv P Q
  -/
  use Units.mk0 _ hPz / Units.mk0 _ hQz
  simp only [Units.smul_def, smul_fin3, Units.val_div_eq_div_val, Units.val_mk0, mul_comm, mul_div,
    ← hx, ← hy, mul_div_cancel_right₀ _ hQz, fin3_def]


lemma equiv_some_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) : P ≈ ![P x / P z, P y / P z, 1] :=
  equiv_of_X_eq_of_Y_eq hPz one_ne_zero
        /-
          F : Type v
          inst✝ : Field F
          P : Fin 3 → F
          hPz : Ne (P 2) 0
          ⊢ Eq (HMul.hMul (P 0) (Matrix.vecCons (HDiv.hDiv (P 0) (P 2)) (Matrix.vecCons  …
        -/
    (by linear_combination (norm := (matrix_simp; ring1)) -P x * div_self hPz)
        /-
          🎉 no goals
        -/
        /-
          F : Type v
          inst✝ : Field F
          P : Fin 3 → F
          hPz : Ne (P 2) 0
          ⊢ Eq (HMul.hMul (P 1) (Matrix.vecCons (HDiv.hDiv (P 0) (P 2)) (Matrix.vecCons  …
        -/
    (by linear_combination (norm := (matrix_simp; ring1)) -P y * div_self hPz)
        /-
          🎉 no goals
        -/


lemma X_eq_iff {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) :
    P x * Q z = Q x * P z ↔ P x / P z = Q x / Q z :=
  (div_eq_div_iff hPz hQz).symm


lemma Y_eq_iff {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) :
    P y * Q z = Q y * P z ↔ P y / P z = Q y / Q z :=
  (div_eq_div_iff hPz hQz).symm


variable (W') in
/-- The polynomial $W(X, Y, Z) := Y^2Z + a_1XYZ + a_3YZ^2 - (X^3 + a_2X^2Z + a_4XZ^2 + a_6Z^3)$
associated to a Weierstrass curve `W'` over `R`. This is represented as a term of type
`MvPolynomial (Fin 3) R`, where `X 0`, `X 1`, and `X 2` represent $X$, $Y$, and $Z$ respectively. -/
noncomputable def polynomial : MvPolynomial (Fin 3) R :=
  X 1 ^ 2 * X 2 + C W'.a₁ * X 0 * X 1 * X 2 + C W'.a₃ * X 1 * X 2 ^ 2
    - (X 0 ^ 3 + C W'.a₂ * X 0 ^ 2 * X 2 + C W'.a₄ * X 0 * X 2 ^ 2 + C W'.a₆ * X 2 ^ 3)


lemma eval_polynomial (P : Fin 3 → R) : eval P W'.polynomial =
    P y ^ 2 * P z + W'.a₁ * P x * P y * P z + W'.a₃ * P y * P z ^ 2
      - (P x ^ 3 + W'.a₂ * P x ^ 2 * P z + W'.a₄ * P x * P z ^ 2 + W'.a₆ * P z ^ 3) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomial) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (H …
  -/
  rw [polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow. …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma eval_polynomial_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) : eval P W.polynomial / P z ^ 3 =
    W.toAffine.polynomial.evalEval (P x / P z) (P y / P z) := by
  linear_combination (norm := (rw [eval_polynomial, Affine.evalEval_polynomial]; ring1))
    P y ^ 2 / P z ^ 2 * div_self hPz + W.a₁ * P x * P y / P z ^ 2 * div_self hPz
      + W.a₃ * P y / P z * div_self (pow_ne_zero 2 hPz) - W.a₂ * P x ^ 2 / P z ^ 2 * div_self hPz
      - W.a₄ * P x / P z * div_self (pow_ne_zero 2 hPz) - W.a₆ * div_self (pow_ne_zero 3 hPz)


variable (W') in
/-- The proposition that a point representative $(x, y, z)$ lies in `W'`.
In other words, $W(x, y, z) = 0$. -/
def Equation (P : Fin 3 → R) : Prop :=
  eval P W'.polynomial = 0


lemma equation_iff (P : Fin 3 → R) : W'.Equation P ↔
    P y ^ 2 * P z + W'.a₁ * P x * P y * P z + W'.a₃ * P y * P z ^ 2
      - (P x ^ 3 + W'.a₂ * P x ^ 2 * P z + W'.a₄ * P x * P z ^ 2 + W'.a₆ * P z ^ 3) = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Iff (W'.Equation P) (Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hP …
  -/
  rw [Equation, eval_polynomial, sub_eq_zero]
  /-
    🎉 no goals
  -/


lemma equation_smul (P : Fin 3 → R) {u : R} (hu : IsUnit u) : W'.Equation (u • P) ↔ W'.Equation P :=
  have hP (u : R) {P : Fin 3 → R} (hP : W'.Equation P) : W'.Equation <| u • P := by
    /-
      R : Type u
      W' : WeierstrassCurve.Projective R
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
      W' : WeierstrassCurve.Projective R
      inst✝ : CommRing R
      P✝ : Fin 3 → R
      u✝ : R
      hu : IsUnit u✝
      u : R
      P : Fin 3 → R
      hP : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (P 1) 2) (P 2)) …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (HSMul.hSMul u P 1 …
    -/
    linear_combination (norm := (simp only [smul_fin3_ext]; ring1)) u ^ 3 * hP
    /-
      🎉 no goals
    -/
               /-
                 R : Type u
                 W' : WeierstrassCurve.Projective R
                 inst✝ : CommRing R
                 P : Fin 3 → R
                 u : R
                 hu : IsUnit u
                 hP : ∀ (u : R) {P : Fin 3 → R}, W'.Equation P → W'.Equation (HSMul.hSMul u P)
                 h : W'.Equation (HSMul.hSMul u P)
                 ⊢ W'.Equation P
               -/
  ⟨fun h => by convert hP hu.unit.inv h; erw [smul_smul, hu.val_inv_mul, one_smul], hP u⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma equation_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.Equation P ↔ W'.Equation Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Iff (W'.Equation P) (W'.Equation Q)
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ Iff (W'.Equation ((fun m => HSMul.hSMul m Q) u)) (W'.Equation Q)
  -/
  exact equation_smul Q u.isUnit
  /-
    🎉 no goals
  -/


lemma equation_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) : W'.Equation P ↔ P x ^ 3 = 0 := by
  simp only [equation_iff, hPz, add_zero, zero_sub, mul_zero, zero_pow <| OfNat.ofNat_ne_zero _,
    neg_eq_zero]


lemma equation_zero : W'.Equation ![0, 1, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ W'.Equation (Matrix.vecCons 0 (Matrix.vecCons 1 (Matrix.vecCons 0 Matrix.vec …
  -/
  simp only [equation_of_Z_eq_zero, fin3_def_ext, zero_pow three_ne_zero]
  /-
    🎉 no goals
  -/


lemma equation_some (X Y : R) : W'.Equation ![X, Y, 1] ↔ W'.toAffine.Equation X Y := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    X Y : R
    ⊢ Iff (W'.Equation (Matrix.vecCons X (Matrix.vecCons Y (Matrix.vecCons 1 Matri …
  -/
  simp only [equation_iff, Affine.equation_iff', fin3_def_ext, one_pow, mul_one]
  /-
    🎉 no goals
  -/


lemma equation_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.Equation P ↔ W.toAffine.Equation (P x / P z) (P y / P z) :=
  (equation_of_equiv <| equiv_some_of_Z_ne_zero hPz).trans <| equation_some ..


lemma X_eq_zero_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) : P x = 0 :=
  pow_eq_zero <| (equation_of_Z_eq_zero hPz).mp hP


variable (W') in
/-- The partial derivative $W_X(X, Y, Z)$ of $W(X, Y, Z)$ with respect to $X$. -/
noncomputable def polynomialX : MvPolynomial (Fin 3) R :=
  pderiv x W'.polynomial


lemma polynomialX_eq : W'.polynomialX =
    C W'.a₁ * X 1 * X 2 - (C 3 * X 0 ^ 2 + C (2 * W'.a₂) * X 0 * X 2 + C W'.a₄ * X 2 ^ 2) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq W'.polynomialX (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C W'.a₁) (M …
  -/
  rw [polynomialX, polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq ((MvPolynomial.pderiv 0) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPo …
  -/
  pderiv_simp
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_polynomialX (P : Fin 3 → R) : eval P W'.polynomialX =
    W'.a₁ * P y * P z - (3 * P x ^ 2 + 2 * W'.a₂ * P x * P z + W'.a₄ * P z ^ 2) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomialX) (HSub.hSub (HMul.hMul (HMul.hMul W …
  -/
  rw [polynomialX_eq]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C W …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma eval_polynomialX_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    eval P W.polynomialX / P z ^ 2 = W.toAffine.polynomialX.evalEval (P x / P z) (P y / P z) := by
  linear_combination (norm := (rw [eval_polynomialX, Affine.evalEval_polynomialX]; ring1))
    W.a₁ * P y / P z * div_self hPz - 2 * W.a₂ * P x / P z * div_self hPz
      - W.a₄ * div_self (pow_ne_zero 2 hPz)


variable (W') in
/-- The partial derivative $W_Y(X, Y, Z)$ of $W(X, Y, Z)$ with respect to $Y$. -/
noncomputable def polynomialY : MvPolynomial (Fin 3) R :=
  pderiv y W'.polynomial


lemma polynomialY_eq : W'.polynomialY =
    C 2 * X 1 * X 2 + C W'.a₁ * X 0 * X 2 + C W'.a₃ * X 2 ^ 2 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq W'.polynomialY (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (MvPolynomial. …
  -/
  rw [polynomialY, polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq ((MvPolynomial.pderiv 1) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPo …
  -/
  pderiv_simp
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_polynomialY (P : Fin 3 → R) :
    eval P W'.polynomialY = 2 * P y * P z + W'.a₁ * P x * P z + W'.a₃ * P z ^ 2 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomialY) (HAdd.hAdd (HAdd.hAdd (HMul.hMul ( …
  -/
  rw [polynomialY_eq]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (MvPol …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


lemma eval_polynomialY_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    eval P W.polynomialY / P z ^ 2 = W.toAffine.polynomialY.evalEval (P x / P z) (P y / P z) := by
  linear_combination (norm := (rw [eval_polynomialY, Affine.evalEval_polynomialY]; ring1))
    2 * P y / P z * div_self hPz + W.a₁ * P x / P z * div_self hPz
      + W.a₃ * div_self (pow_ne_zero 2 hPz)


variable (W') in
/-- The partial derivative $W_Z(X, Y, Z)$ of $W(X, Y, Z)$ with respect to $Z$. -/
noncomputable def polynomialZ : MvPolynomial (Fin 3) R :=
  pderiv z W'.polynomial


lemma polynomialZ_eq : W'.polynomialZ =
    X 1 ^ 2 + C W'.a₁ * X 0 * X 1 + C (2 * W'.a₃) * X 1 * X 2
      - (C W'.a₂ * X 0 ^ 2 + C (2 * W'.a₄) * X 0 * X 2 + C (3 * W'.a₆) * X 2 ^ 2) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq W'.polynomialZ (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (MvPolynomial. …
  -/
  rw [polynomialZ, polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq ((MvPolynomial.pderiv 2) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPo …
  -/
  pderiv_simp
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma eval_polynomialZ (P : Fin 3 → R) : eval P W'.polynomialZ =
    P y ^ 2 + W'.a₁ * P x * P y + 2 * W'.a₃ * P y * P z
      - (W'.a₂ * P x ^ 2 + 2 * W'.a₄ * P x * P z + 3 * W'.a₆ * P z ^ 2) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) W'.polynomialZ) (HSub.hSub (HAdd.hAdd (HAdd.hAdd ( …
  -/
  rw [polynomialZ_eq]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq ((MvPolynomial.eval P) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow (MvPol …
  -/
  eval_simp
  /-
    🎉 no goals
  -/


/-- Euler's homogeneous function theorem. -/
theorem polynomial_relation (P : Fin 3 → R) : 3 * eval P W'.polynomial =
    P x * eval P W'.polynomialX + P y * eval P W'.polynomialY + P z * eval P W'.polynomialZ := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HMul.hMul 3 ((MvPolynomial.eval P) W'.polynomial)) (HAdd.hAdd (HAdd.hAdd …
  -/
  rw [eval_polynomial, eval_polynomialX, eval_polynomialY, eval_polynomialZ]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HMul.hMul 3 (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (P 1) …
  -/
  ring1
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
    (W'.a₁ * P y * P z - (3 * P x ^ 2 + 2 * W'.a₂ * P x * P z + W'.a₄ * P z ^ 2) ≠ 0 ∨
      2 * P y * P z + W'.a₁ * P x * P z + W'.a₃ * P z ^ 2 ≠ 0 ∨
      P y ^ 2 + W'.a₁ * P x * P y + 2 * W'.a₃ * P y * P z
        - (W'.a₂ * P x ^ 2 + 2 * W'.a₄ * P x * P z + 3 * W'.a₆ * P z ^ 2) ≠ 0) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
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
  have hP {u : R} (hu : IsUnit u) {P : Fin 3 → R} (hP : W'.Nonsingular <| u • P) :
      W'.Nonsingular P := by
    /-
      R : Type u
      W' : WeierstrassCurve.Projective R
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
      W' : WeierstrassCurve.Projective R
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
      W' : WeierstrassCurve.Projective R
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
      W' : WeierstrassCurve.Projective R
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
    exact ⟨by linear_combination (norm := ring1) u ^ 2 * hP'.left,
      by linear_combination (norm := ring1) u ^ 2 * hP'.right.left,
      by linear_combination (norm := ring1) u ^ 2 * hP'.right.right⟩
                                             /-
                                               R : Type u
                                               W' : WeierstrassCurve.Projective R
                                               inst✝ : CommRing R
                                               P : Fin 3 → R
                                               u : R
                                               hu : IsUnit u
                                               hP : ∀ {u : R}, IsUnit u → ∀ {P : Fin 3 → R}, W'.Nonsingular (HSMul.hSMul u P) …
                                               h : W'.Nonsingular P
                                               ⊢ W'.Nonsingular (HSMul.hSMul (↑(Inv.inv hu.unit)) (HSMul.hSMul u P))
                                             -/
  ⟨hP hu, fun h => hP hu.unit⁻¹.isUnit <| by rwa [smul_smul, hu.val_inv_mul, one_smul]⟩
                                             /-
                                               🎉 no goals
                                             -/


lemma nonsingular_of_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.Nonsingular P ↔ W'.Nonsingular Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ Iff (W'.Nonsingular P) (W'.Nonsingular Q)
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Projective R
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
    W'.Nonsingular P ↔
      W'.Equation P ∧ (3 * P x ^ 2 ≠ 0 ∨ P y ^ 2 + W'.a₁ * P x * P y - W'.a₂ * P x ^ 2 ≠ 0) := by
  simp only [nonsingular_iff, hPz, add_zero, sub_zero, zero_sub, mul_zero,
    zero_pow <| OfNat.ofNat_ne_zero _, neg_ne_zero, ne_self_iff_false, false_or]


lemma nonsingular_zero [Nontrivial R] : W'.Nonsingular ![0, 1, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ W'.Nonsingular (Matrix.vecCons 0 (Matrix.vecCons 1 (Matrix.vecCons 0 Matrix. …
  -/
  simp only [nonsingular_of_Z_eq_zero, equation_zero, true_and, fin3_def_ext, ← not_and_or]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Not (And (Eq (HMul.hMul 3 (HPow.hPow 0 2)) 0) (Eq (HSub.hSub (HAdd.hAdd (HPo …
  -/
  exact fun h => one_ne_zero <| by linear_combination (norm := ring1) h.right
  /-
    🎉 no goals
  -/


lemma nonsingular_some (X Y : R) : W'.Nonsingular ![X, Y, 1] ↔ W'.toAffine.Nonsingular X Y := by
  simp_rw [nonsingular_iff, equation_some, fin3_def_ext, Affine.nonsingular_iff',
    Affine.equation_iff', and_congr_right_iff, ← not_and_or, not_iff_not, one_pow, mul_one,
    and_congr_right_iff, Iff.comm, iff_self_and]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    X Y : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow Y 2) (HMul.hMul (HMul.hMul W' …
  -/
  intro h hX hY
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    X Y : R
    h : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow Y 2) (HMul.hMul (HMul.hMul  …
    hX : Eq (HSub.hSub (HMul.hMul W'.a₁ Y) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPo …
    hY : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 Y) (HMul.hMul W'.toAffine.a₁ X)) W' …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow Y 2) (HMul.hMul (HMul.hMul W' …
  -/
  linear_combination (norm := ring1) 3 * h - X * hX - Y * hY
  /-
    🎉 no goals
  -/


lemma nonsingular_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.Nonsingular P ↔ W.toAffine.Nonsingular (P x / P z) (P y / P z) :=
  (nonsingular_of_equiv <| equiv_some_of_Z_ne_zero hPz).trans <| nonsingular_some ..


lemma nonsingular_iff_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.Nonsingular P ↔ W.Equation P ∧ (eval P W.polynomialX ≠ 0 ∨ eval P W.polynomialY ≠ 0) := by
  rw [nonsingular_of_Z_ne_zero hPz, Affine.Nonsingular, ← equation_of_Z_ne_zero hPz,
    ← eval_polynomialX_of_Z_ne_zero hPz, div_ne_zero_iff, and_iff_left <| pow_ne_zero 2 hPz,
    ← eval_polynomialY_of_Z_ne_zero hPz, div_ne_zero_iff, and_iff_left <| pow_ne_zero 2 hPz]


lemma Y_ne_zero_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Nonsingular P)
    (hPz : P z = 0) : P y ≠ 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Nonsingular P
    hPz : Eq (P 2) 0
    ⊢ Ne (P 1) 0
  -/
  intro hPy
  simp only [nonsingular_of_Z_eq_zero hPz, X_eq_zero_of_Z_eq_zero hP.left hPz, hPy, add_zero,
    sub_zero, mul_zero, zero_pow two_ne_zero, or_self, ne_self_iff_false, and_false] at hP


lemma isUnit_Y_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) : IsUnit (P y) :=
  (Y_ne_zero_of_Z_eq_zero hP hPz).isUnit


lemma equiv_of_Z_eq_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Nonsingular Q)
    (hPz : P z = 0) (hQz : Q z = 0) : P ≈ Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ HasEquiv.Equiv P Q
  -/
  use (isUnit_Y_of_Z_eq_zero hP hPz).unit / (isUnit_Y_of_Z_eq_zero hQ hQz).unit
  simp only [Units.smul_def, smul_fin3, X_eq_zero_of_Z_eq_zero hQ.left hQz, hQz, mul_zero,
    Units.val_div_eq_div_val, IsUnit.unit_spec, (isUnit_Y_of_Z_eq_zero hQ hQz).div_mul_cancel]
  /-
    case h
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ Eq (Matrix.vecCons 0 (Matrix.vecCons (P 1) (Matrix.vecCons 0 Matrix.vecEmpty …
  -/
  conv_rhs => rw [← fin3_def P, X_eq_zero_of_Z_eq_zero hP.left hPz, hPz]
  /-
    🎉 no goals
  -/


lemma equiv_zero_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) :
    P ≈ ![0, 1, 0] :=
  equiv_of_Z_eq_zero hP nonsingular_zero hPz rfl


lemma comp_equiv_comp {K : Type v} [Field K] (f : F →+* K) {P Q : Fin 3 → F} (hP : W.Nonsingular P)
    (hQ : W.Nonsingular Q): f ∘ P ≈ f ∘ Q ↔ P ≈ Q := by
  /-
    F : Type v
    inst✝¹ : Field F
    W : WeierstrassCurve.Projective F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    ⊢ Iff (HasEquiv.Equiv (Function.comp (⇑f) P) (Function.comp (⇑f) Q)) (HasEquiv …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      h : HasEquiv.Equiv (Function.comp (⇑f) P) (Function.comp (⇑f) Q)
      ⊢ HasEquiv.Equiv P Q
    -/
  · by_cases hz : f (P z) = 0
    · exact equiv_of_Z_eq_zero hP hQ ((map_eq_zero_iff f f.injective).mp hz) <|
        (map_eq_zero_iff f f.injective).mp <| (Z_eq_zero_of_equiv h).mp hz
    · refine equiv_of_X_eq_of_Y_eq ((map_ne_zero_iff f f.injective).mp hz)
        ((map_ne_zero_iff f f.injective).mp <| hz.comp (Z_eq_zero_of_equiv h).mpr) ?_ ?_
      /-
        case neg.refine_1
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        K : Type v
        inst✝ : Field K
        f : RingHom F K
        P Q : Fin 3 → F
        hP : W.Nonsingular P
        hQ : W.Nonsingular Q
        h : HasEquiv.Equiv (Function.comp (⇑f) P) (Function.comp (⇑f) Q)
        hz : Not (Eq (f (P 2)) 0)
        ⊢ Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
      -/
      all_goals apply f.injective; map_simp
      /-
        case neg.refine_1.a
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        K : Type v
        inst✝ : Field K
        f : RingHom F K
        P Q : Fin 3 → F
        hP : W.Nonsingular P
        hQ : W.Nonsingular Q
        h : HasEquiv.Equiv (Function.comp (⇑f) P) (Function.comp (⇑f) Q)
        hz : Not (Eq (f (P 2)) 0)
        ⊢ Eq (HMul.hMul (f (P 0)) (f (Q 2))) (HMul.hMul (f (Q 0)) (f (P 2)))
      -/
      exacts [X_eq_of_equiv h, Y_eq_of_equiv h]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      h : HasEquiv.Equiv P Q
      ⊢ HasEquiv.Equiv (Function.comp (⇑f) P) (Function.comp (⇑f) Q)
    -/
  · rcases h with ⟨u, rfl⟩
    /-
      case refine_2.intro
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      Q : Fin 3 → F
      hQ : W.Nonsingular Q
      u : Units F
      hP : W.Nonsingular ((fun m => HSMul.hSMul m Q) u)
      ⊢ HasEquiv.Equiv (Function.comp (⇑f) ((fun m => HSMul.hSMul m Q) u)) (Function …
    -/
    exact ⟨Units.map f u, (comp_smul ..).symm⟩
    /-
      🎉 no goals
    -/


variable (W') in
/-- The proposition that a point class on `W'` is nonsingular. If `P` is a point representative,
then `W.NonsingularLift ⟦P⟧` is definitionally equivalent to `W.Nonsingular P`. -/
def NonsingularLift (P : PointClass R) : Prop :=
  P.lift W'.Nonsingular fun _ _ => propext ∘ nonsingular_of_equiv


lemma nonsingularLift_iff (P : Fin 3 → R) : W'.NonsingularLift ⟦P⟧ ↔ W'.Nonsingular P :=
  Iff.rfl


lemma nonsingularLift_zero [Nontrivial R] : W'.NonsingularLift ⟦![0, 1, 0]⟧ :=
  nonsingular_zero


lemma nonsingularLift_some (X Y : R) :
    W'.NonsingularLift ⟦![X, Y, 1]⟧ ↔ W'.toAffine.Nonsingular X Y :=
  nonsingular_some X Y


@[deprecated (since := "2024-08-27")] alias equation_smul_iff := equation_smul

@[deprecated (since := "2024-08-27")] alias nonsingularLift_zero' := nonsingularLift_zero

@[deprecated (since := "2024-08-27")]
alias nonsingular_affine_of_Z_ne_zero := nonsingular_of_Z_ne_zero

@[deprecated (since := "2024-08-27")]
alias nonsingular_iff_affine_of_Z_ne_zero := nonsingular_of_Z_ne_zero

@[deprecated (since := "2024-08-27")]
alias nonsingular_of_affine_of_Z_ne_zero := nonsingular_of_Z_ne_zero

@[deprecated (since := "2024-08-27")] alias nonsingular_smul_iff := nonsingular_smul

@[deprecated (since := "2024-08-27")] alias nonsingular_zero' := nonsingular_zero


variable (W') in
/-- The $Y$-coordinate of a representative of `-P` for a point `P`. -/
def negY (P : Fin 3 → R) : R :=
  -P y - W'.a₁ * P x - W'.a₃ * P z


lemma negY_eq (X Y Z : R) : W'.negY ![X, Y, Z] = -Y - W'.a₁ * X - W'.a₃ * Z :=
  rfl


lemma negY_smul (P : Fin 3 → R) (u : R) : W'.negY (u • P) = u * W'.negY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.negY (HSMul.hSMul u P)) (HMul.hMul u (W'.negY P))
  -/
  simp only [negY, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul u (P 1))) (HMul.hMul W'.a₁ (HMu …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negY_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.negY P = -P y := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.negY P) (Neg.neg (P 1))
  -/
  rw [negY, hPz, X_eq_zero_of_Z_eq_zero hP hPz, mul_zero, sub_zero, mul_zero, sub_zero]
  /-
    🎉 no goals
  -/


lemma negY_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.negY P / P z = W.toAffine.negY (P x / P z) (P y / P z) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    ⊢ Eq (HDiv.hDiv (W.negY P) (P 2)) (W.toAffine.negY (HDiv.hDiv (P 0) (P 2)) (HD …
  -/
  linear_combination (norm := (rw [negY, Affine.negY]; ring1)) -W.a₃ * div_self hPz
  /-
    🎉 no goals
  -/


lemma Y_sub_Y_mul_Y_sub_negY {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hx : P x * Q z = Q x * P z) :
    P z * Q z * (P y * Q z - Q y * P z) * (P y * Q z - W'.negY Q * P z) = 0 := by
  linear_combination' (norm := (rw [negY]; ring1)) Q z ^ 3 * (equation_iff P).mp hP
    - P z ^ 3 * (equation_iff Q).mp hQ + hx * hx * hx + W'.a₂ * P z * Q z * hx * hx
    + (W'.a₄ * P z ^ 2 * Q z ^ 2 - W'.a₁ * P y * P z * Q z ^ 2) * hx


lemma Y_eq_of_Y_ne [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ Q y * P z) :
    P y * Q z = W'.negY Q * P z :=
  sub_eq_zero.mp <| (mul_eq_zero.mp <| Y_sub_Y_mul_Y_sub_negY hP hQ hx).resolve_left <|
    mul_ne_zero (mul_ne_zero hPz hQz) <| sub_ne_zero.mpr hy


lemma Y_eq_of_Y_ne' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z)
    (hy : P y * Q z ≠ W'.negY Q * P z) : P y * Q z = Q y * P z :=
  sub_eq_zero.mp <| (mul_eq_zero.mp <| (mul_eq_zero.mp <| Y_sub_Y_mul_Y_sub_negY hP hQ hx
    ).resolve_right <| sub_ne_zero.mpr hy).resolve_left <| mul_ne_zero hPz hQz


lemma Y_eq_iff' {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) :
    P y * Q z = W.negY Q * P z ↔ P y / P z = W.toAffine.negY (Q x / Q z) (Q y / Q z) :=
  negY_of_Z_ne_zero hQz ▸ (div_eq_div_iff hPz hQz).symm


lemma Y_sub_Y_add_Y_sub_negY {P Q : Fin 3 → R} (hx : P x * Q z = Q x * P z) :
    (P y * Q z - Q y * P z) + (P y * Q z - W'.negY Q * P z) = (P y - W'.negY P) * Q z := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))) (H …
  -/
  linear_combination (norm := (rw [negY, negY]; ring1)) -W'.a₁ * hx
  /-
    🎉 no goals
  -/


lemma Y_ne_negY_of_Y_ne [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z)
    (hy : P y * Q z ≠ Q y * P z) : P y ≠ W'.negY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    ⊢ Ne (P 1) (W'.negY P)
  -/
  have hy' : P y * Q z - W'.negY Q * P z = 0 := sub_eq_zero.mpr <| Y_eq_of_Y_ne hP hQ hPz hQz hx hy
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))) 0
    ⊢ Ne (P 1) (W'.negY P)
  -/
  contrapose! hy
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))) 0
    hy : Eq (P 1) (W'.negY P)
    ⊢ Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
  -/
  linear_combination (norm := ring1) Y_sub_Y_add_Y_sub_negY hx + Q z * hy - hy'
  /-
    🎉 no goals
  -/


lemma Y_ne_negY_of_Y_ne' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z)
    (hy : P y * Q z ≠ W'.negY Q * P z) : P y ≠ W'.negY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Ne (P 1) (W'.negY P)
  -/
  have hy' : P y * Q z - Q y * P z = 0 := sub_eq_zero.mpr <| Y_eq_of_Y_ne' hP hQ hPz hQz hx hy
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))) 0
    ⊢ Ne (P 1) (W'.negY P)
  -/
  contrapose! hy
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy' : Eq (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))) 0
    hy : Eq (P 1) (W'.negY P)
    ⊢ Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
  -/
  linear_combination (norm := ring1) Y_sub_Y_add_Y_sub_negY hx + Q z * hy - hy'
  /-
    🎉 no goals
  -/


lemma Y_eq_negY_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W'.negY Q * P z) :
    P y = W'.negY P :=
  mul_left_injective₀ hQz <| by
    /-
      R : Type u
      W' : WeierstrassCurve.Projective R
      inst✝¹ : CommRing R
      inst✝ : NoZeroDivisors R
      P Q : Fin 3 → R
      hQz : Ne (Q 2) 0
      hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
      hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
      hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
      ⊢ Eq ((fun a => HMul.hMul a (Q 2)) (P 1)) ((fun a => HMul.hMul a (Q 2)) (W'.ne …
    -/
    linear_combination (norm := ring1) -Y_sub_Y_add_Y_sub_negY hx + hy + hy'
    /-
      🎉 no goals
    -/


lemma nonsingular_iff_of_Y_eq_negY {P : Fin 3 → F} (hPz : P z ≠ 0) (hy : P y = W.negY P) :
    W.Nonsingular P ↔ W.Equation P ∧ eval P W.polynomialX ≠ 0 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    hy : Eq (P 1) (W.negY P)
    ⊢ Iff (W.Nonsingular P) (And (W.Equation P) (Ne ((MvPolynomial.eval P) W.polyn …
  -/
  have hy' : eval P W.polynomialY = (P y - W.negY P) * P z := by rw [negY, eval_polynomialY]; ring1
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    hy : Eq (P 1) (W.negY P)
    hy' : Eq ((MvPolynomial.eval P) W.polynomialY) (HMul.hMul (HSub.hSub (P 1) (W. …
    ⊢ Iff (W.Nonsingular P) (And (W.Equation P) (Ne ((MvPolynomial.eval P) W.polyn …
  -/
  rw [nonsingular_iff_of_Z_ne_zero hPz, hy', hy, sub_self, zero_mul, ne_self_iff_false, or_false]
  /-
    🎉 no goals
  -/


variable (W) in
/-- The unit associated to the doubling of a 2-torsion point `P`.
More specifically, the unit `u` such that `W.add P P = u • ![0, 1, 0]` where `P = W.neg P`. -/
noncomputable def dblU (P : Fin 3 → F) : F :=
  eval P W.polynomialX ^ 3 / P z ^ 2


lemma dblU_eq (P : Fin 3 → F) : W.dblU P =
    (W.a₁ * P y * P z - (3 * P x ^ 2 + 2 * W.a₂ * P x * P z + W.a₄ * P z ^ 2)) ^ 3 / P z ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    ⊢ Eq (W.dblU P) (HDiv.hDiv (HPow.hPow (HSub.hSub (HMul.hMul (HMul.hMul W.a₁ (P …
  -/
  rw [dblU, eval_polynomialX]
  /-
    🎉 no goals
  -/


lemma dblU_smul {P : Fin 3 → F} (hPz : P z ≠ 0) {u : F} (hu : u ≠ 0) :
    W.dblU (u • P) = u ^ 4 * W.dblU P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    u : F
    hu : Ne u 0
    ⊢ Eq (W.dblU (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W.dblU P))
  -/
  field_simp [dblU_eq, smul_fin3_ext]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    u : F
    hu : Ne u 0
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub (HMul.hMul (HMul.hMul W.a₁ (HMul.hMul u  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblU_of_Z_eq_zero {P : Fin 3 → F} (hPz : P z = 0) : W.dblU P = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Eq (P 2) 0
    ⊢ Eq (W.dblU P) 0
  -/
  rw [dblU_eq, hPz, zero_pow two_ne_zero, div_zero]
  /-
    🎉 no goals
  -/


lemma dblU_ne_zero_of_Y_eq {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W.negY Q * P z) :
    W.dblU P ≠ 0 :=
  div_ne_zero (pow_ne_zero 3
    ((nonsingular_iff_of_Y_eq_negY hPz <| Y_eq_negY_of_Y_eq hQz hx hy hy').mp hP).right) <|
    pow_ne_zero 2 hPz


lemma isUnit_dblU_of_Y_eq {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W.negY Q * P z) :
    IsUnit (W.dblU P) :=
  (dblU_ne_zero_of_Y_eq hP hPz hQz hx hy hy').isUnit


variable (W') in
/-- The $Z$-coordinate of a representative of `2 • P` for a point `P`. -/
def dblZ (P : Fin 3 → R) : R :=
  P z * (P y - W'.negY P) ^ 3


lemma dblZ_smul (P : Fin 3 → R) (u : R) : W'.dblZ (u • P) = u ^ 4 * W'.dblZ P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblZ (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W'.dblZ P))
  -/
  simp only [dblZ, negY_smul, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HMul.hMul (HMul.hMul u (P 2)) (HPow.hPow (HSub.hSub (HMul.hMul u (P 1))  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblZ_of_Z_eq_zero {P : Fin 3 → R} (hPz : P z = 0) : W'.dblZ P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.dblZ P) 0
  -/
  rw [dblZ, hPz, zero_mul]
  /-
    🎉 no goals
  -/


lemma dblZ_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z)
    (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W'.negY Q * P z) : W'.dblZ P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Eq (W'.dblZ P) 0
  -/
  rw [dblZ, Y_eq_negY_of_Y_eq hQz hx hy hy', sub_self, zero_pow three_ne_zero, mul_zero]
  /-
    🎉 no goals
  -/


lemma dblZ_ne_zero_of_Y_ne [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z)
    (hy : P y * Q z ≠ Q y * P z) : W'.dblZ P ≠ 0 :=
  mul_ne_zero hPz <| pow_ne_zero 3 <| sub_ne_zero.mpr <| Y_ne_negY_of_Y_ne hP hQ hPz hQz hx hy


lemma isUnit_dblZ_of_Y_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ Q y * P z) : IsUnit (W.dblZ P) :=
  (dblZ_ne_zero_of_Y_ne hP hQ hPz hQz hx hy).isUnit


lemma dblZ_ne_zero_of_Y_ne' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z)
    (hy : P y * Q z ≠ W'.negY Q * P z) : W'.dblZ P ≠ 0 :=
  mul_ne_zero hPz <| pow_ne_zero 3 <| sub_ne_zero.mpr <| Y_ne_negY_of_Y_ne' hP hQ hPz hQz hx hy


lemma isUnit_dblZ_of_Y_ne' {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    IsUnit (W.dblZ P) :=
  (dblZ_ne_zero_of_Y_ne' hP hQ hPz hQz hx hy).isUnit


private lemma toAffine_slope_of_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z) =
      -eval P W.polynomialX / P z / (P y - W.negY P) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W.negY Q) (P 2))
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) (HDiv.h …
  -/
  have hPy : P y - W.negY P ≠ 0 := sub_ne_zero.mpr <| Y_ne_negY_of_Y_ne' hP hQ hPz hQz hx hy
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W.negY Q) (P 2))
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) (HDiv.h …
  -/
  simp only [X_eq_iff hPz hQz, ne_eq, Y_eq_iff' hPz hQz] at hx hy
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    hx : Eq (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2))
    hy : Not (Eq (HDiv.hDiv (P 1) (P 2)) (W.toAffine.negY (HDiv.hDiv (Q 0) (Q 2))  …
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) (HDiv.h …
  -/
  rw [Affine.slope_of_Y_ne hx <| negY_of_Z_ne_zero hQz ▸ hy, ← negY_of_Z_ne_zero hPz]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    hx : Eq (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2))
    hy : Not (Eq (HDiv.hDiv (P 1) (P 2)) (W.toAffine.negY (HDiv.hDiv (Q 0) (Q 2))  …
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow (HDiv …
  -/
  field_simp [eval_polynomialX, hPz]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hPy : Ne (HSub.hSub (P 1) (W.negY P)) 0
    hx : Eq (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2))
    hy : Not (Eq (HDiv.hDiv (P 1) (P 2)) (W.toAffine.negY (HDiv.hDiv (Q 0) (Q 2))  …
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


variable (W') in
/-- The $X$-coordinate of a representative of `2 • P` for a point `P`. -/
noncomputable def dblX (P : Fin 3 → R) : R :=
  2 * P x * P y ^ 3 + 3 * W'.a₁ * P x ^ 2 * P y ^ 2 + 6 * W'.a₂ * P x ^ 3 * P y
    - 8 * W'.a₂ * P y ^ 3 * P z + 9 * W'.a₃ * P x ^ 4 - 6 * W'.a₃ * P x * P y ^ 2 * P z
    - 6 * W'.a₄ * P x ^ 2 * P y * P z - 18 * W'.a₆ * P x * P y * P z ^ 2
    + 3 * W'.a₁ ^ 2 * P x ^ 3 * P y - 2 * W'.a₁ ^ 2 * P y ^ 3 * P z + 3 * W'.a₁ * W'.a₂ * P x ^ 4
    - 12 * W'.a₁ * W'.a₂ * P x * P y ^ 2 * P z - 9 * W'.a₁ * W'.a₃ * P x ^ 2 * P y * P z
    - 3 * W'.a₁ * W'.a₄ * P x ^ 3 * P z - 9 * W'.a₁ * W'.a₆ * P x ^ 2 * P z ^ 2
    + 8 * W'.a₂ ^ 2 * P x ^ 2 * P y * P z + 12 * W'.a₂ * W'.a₃ * P x ^ 3 * P z
    - 12 * W'.a₂ * W'.a₃ * P y ^ 2 * P z ^ 2 + 8 * W'.a₂ * W'.a₄ * P x * P y * P z ^ 2
    - 12 * W'.a₃ ^ 2 * P x * P y * P z ^ 2 + 6 * W'.a₃ * W'.a₄ * P x ^ 2 * P z ^ 2
    + 2 * W'.a₄ ^ 2 * P y * P z ^ 3 + W'.a₁ ^ 3 * P x ^ 4 - 3 * W'.a₁ ^ 3 * P x * P y ^ 2 * P z
    - 2 * W'.a₁ ^ 2 * W'.a₂ * P x ^ 2 * P y * P z - 3 * W'.a₁ ^ 2 * W'.a₃ * P y ^ 2 * P z ^ 2
    + 2 * W'.a₁ ^ 2 * W'.a₄ * P x * P y * P z ^ 2 + 4 * W'.a₁ * W'.a₂ ^ 2 * P x ^ 3 * P z
    - 8 * W'.a₁ * W'.a₂ * W'.a₃ * P x * P y * P z ^ 2
    + 4 * W'.a₁ * W'.a₂ * W'.a₄ * P x ^ 2 * P z ^ 2 - 3 * W'.a₁ * W'.a₃ ^ 2 * P x ^ 2 * P z ^ 2
    + 2 * W'.a₁ * W'.a₃ * W'.a₄ * P y * P z ^ 3 + W'.a₁ * W'.a₄ ^ 2 * P x * P z ^ 3
    + 4 * W'.a₂ ^ 2 * W'.a₃ * P x ^ 2 * P z ^ 2 - 6 * W'.a₂ * W'.a₃ ^ 2 * P y * P z ^ 3
    + 4 * W'.a₂ * W'.a₃ * W'.a₄ * P x * P z ^ 3 - 2 * W'.a₃ ^ 3 * P x * P z ^ 3
    + W'.a₃ * W'.a₄ ^ 2 * P z ^ 4 - W'.a₁ ^ 4 * P x ^ 2 * P y * P z
    + W'.a₁ ^ 3 * W'.a₂ * P x ^ 3 * P z - 2 * W'.a₁ ^ 3 * W'.a₃ * P x * P y * P z ^ 2
    + W'.a₁ ^ 3 * W'.a₄ * P x ^ 2 * P z ^ 2 + W'.a₁ ^ 2 * W'.a₂ * W'.a₃ * P x ^ 2 * P z ^ 2
    - W'.a₁ ^ 2 * W'.a₃ ^ 2 * P y * P z ^ 3 + 2 * W'.a₁ ^ 2 * W'.a₃ * W'.a₄ * P x * P z ^ 3
    - W'.a₁ * W'.a₂ * W'.a₃ ^ 2 * P x * P z ^ 3 - W'.a₂ * W'.a₃ ^ 3 * P z ^ 4
    + W'.a₁ * W'.a₃ ^ 2 * W'.a₄ * P z ^ 4


lemma dblX_eq' {P : Fin 3 → R} (hP : W'.Equation P) : W'.dblX P * P z =
    (eval P W'.polynomialX ^ 2 - W'.a₁ * eval P W'.polynomialX * P z * (P y - W'.negY P)
      - W'.a₂ * P z ^ 2 * (P y - W'.negY P) ^ 2 - 2 * P x * P z * (P y - W'.negY P) ^ 2)
      * (P y - W'.negY P) := by
  linear_combination (norm := (rw [dblX, eval_polynomialX, negY]; ring1))
    9 * (W'.a₁ * P x ^ 2 + 2 * P x * P y) * (equation_iff _).mp hP


lemma dblX_eq {P : Fin 3 → F} (hP : W.Equation P) (hPz : P z ≠ 0) : W.dblX P =
    ((eval P W.polynomialX ^ 2 - W.a₁ * eval P W.polynomialX * P z * (P y - W.negY P)
      - W.a₂ * P z ^ 2 * (P y - W.negY P) ^ 2 - 2 * P x * P z * (P y - W.negY P) ^ 2)
      * (P y - W.negY P)) / P z := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : W.Equation P
    hPz : Ne (P 2) 0
    ⊢ Eq (W.dblX P) (HDiv.hDiv (HMul.hMul (HSub.hSub (HSub.hSub (HSub.hSub (HPow.h …
  -/
  rw [← dblX_eq' hP, mul_div_cancel_right₀ _ hPz]
  /-
    🎉 no goals
  -/


lemma dblX_smul (P : Fin 3 → R) (u : R) : W'.dblX (u • P) = u ^ 4 * W'.dblX P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblX (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W'.dblX P))
  -/
  simp only [dblX, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblX_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.dblX P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.dblX P) 0
  -/
  rw [dblX, hPz, X_eq_zero_of_Z_eq_zero hP hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblX_of_Y_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z)
    (hy' : P y * Q z = W'.negY Q * P z) : W'.dblX P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Eq (W'.dblX P) 0
  -/
  apply eq_zero_of_ne_zero_of_mul_right_eq_zero hPz
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Eq (HMul.hMul (W'.dblX P) (P 2)) 0
  -/
  rw [dblX_eq' hP, Y_eq_negY_of_Y_eq hQz hx hy hy']
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Eq (HMul.hMul (HSub.hSub (HSub.hSub (HSub.hSub (HPow.hPow ((MvPolynomial.eva …
  -/
  ring1
  /-
    🎉 no goals
  -/


private lemma toAffine_addX_of_eq {P : Fin 3 → F} (hPz : P z ≠ 0) {n d : F} (hd : d ≠ 0) :
    W.toAffine.addX (P x / P z) (P x / P z) (-n / P z / d) =
      (n ^ 2 - W.a₁ * n * P z * d - W.a₂ * P z ^ 2 * d ^ 2 - 2 * P x * P z * d ^ 2) * d / P z
        / (P z * d ^ 3) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (W.toAffine.addX (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (P 0) (P 2)) (HDiv.hD …
  -/
  field_simp [mul_ne_zero hPz hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblX_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    W.dblX P / W.dblZ P = W.toAffine.addX (P x / P z) (Q x / Q z)
      (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)) := by
  rw [dblX_eq hP hPz, dblZ, toAffine_slope_of_eq hP hQ hPz hQz hx hy, ← (X_eq_iff hPz hQz).mp hx,
    toAffine_addX_of_eq hPz <| sub_ne_zero.mpr <| Y_ne_negY_of_Y_ne' hP hQ hPz hQz hx hy]


variable (W') in
/-- The $Y$-coordinate of a representative of `-(2 • P)` for a point `P`. -/
noncomputable def negDblY (P : Fin 3 → R) : R :=
  -P y ^ 4 - 3 * W'.a₁ * P x * P y ^ 3 - 9 * W'.a₃ * P x ^ 3 * P y + 3 * W'.a₃ * P y ^ 3 * P z
    - 3 * W'.a₄ * P x * P y ^ 2 * P z - 27 * W'.a₆ * P x ^ 3 * P z + 9 * W'.a₆ * P y ^ 2 * P z ^ 2
    - 3 * W'.a₁ ^ 2 * P x ^ 2 * P y ^ 2 + 4 * W'.a₁ * W'.a₂ * P y ^ 3 * P z
    - 3 * W'.a₁ * W'.a₂ * P x ^ 3 * P y - 9 * W'.a₁ * W'.a₃ * P x ^ 4
    + 6 * W'.a₁ * W'.a₃ * P x * P y ^ 2 * P z + 18 * W'.a₁ * W'.a₆ * P x * P y * P z ^ 2
    + 9 * W'.a₂ ^ 2 * P x ^ 4 - 8 * W'.a₂ ^ 2 * P x * P y ^ 2 * P z
    - 9 * W'.a₂ * W'.a₃ * P x ^ 2 * P y * P z + 9 * W'.a₂ * W'.a₄ * P x ^ 3 * P z
    - 4 * W'.a₂ * W'.a₄ * P y ^ 2 * P z ^ 2 - 27 * W'.a₂ * W'.a₆ * P x ^ 2 * P z ^ 2
    - 9 * W'.a₃ ^ 2 * P x ^ 3 * P z + 6 * W'.a₃ ^ 2 * P y ^ 2 * P z ^ 2
    - 12 * W'.a₃ * W'.a₄ * P x * P y * P z ^ 2 + 9 * W'.a₄ ^ 2 * P x ^ 2 * P z ^ 2
    - 2 * W'.a₁ ^ 3 * P x ^ 3 * P y + W'.a₁ ^ 3 * P y ^ 3 * P z + 3 * W'.a₁ ^ 2 * W'.a₂ * P x ^ 4
    + 2 * W'.a₁ ^ 2 * W'.a₂ * P x * P y ^ 2 * P z + 3 * W'.a₁ ^ 2 * W'.a₃ * P x ^ 2 * P y * P z
    + 3 * W'.a₁ ^ 2 * W'.a₄ * P x ^ 3 * P z - W'.a₁ ^ 2 * W'.a₄ * P y ^ 2 * P z ^ 2
    - 12 * W'.a₁ * W'.a₂ ^ 2 * P x ^ 2 * P y * P z - 6 * W'.a₁ * W'.a₂ * W'.a₃ * P x ^ 3 * P z
    + 4 * W'.a₁ * W'.a₂ * W'.a₃ * P y ^ 2 * P z ^ 2
    - 8 * W'.a₁ * W'.a₂ * W'.a₄ * P x * P y * P z ^ 2 + 6 * W'.a₁ * W'.a₃ ^ 2 * P x * P y * P z ^ 2
    - W'.a₁ * W'.a₄ ^ 2 * P y * P z ^ 3 + 8 * W'.a₂ ^ 3 * P x ^ 3 * P z
    - 8 * W'.a₂ ^ 2 * W'.a₃ * P x * P y * P z ^ 2 + 12 * W'.a₂ ^ 2 * W'.a₄ * P x ^ 2 * P z ^ 2
    - 9 * W'.a₂ * W'.a₃ ^ 2 * P x ^ 2 * P z ^ 2 - 4 * W'.a₂ * W'.a₃ * W'.a₄ * P y * P z ^ 3
    + 6 * W'.a₂ * W'.a₄ ^ 2 * P x * P z ^ 3 + W'.a₃ ^ 3 * P y * P z ^ 3
    - 3 * W'.a₃ ^ 2 * W'.a₄ * P x * P z ^ 3 + W'.a₄ ^ 3 * P z ^ 4 + W'.a₁ ^ 4 * P x * P y ^ 2 * P z
    - 3 * W'.a₁ ^ 3 * W'.a₂ * P x ^ 2 * P y * P z + W'.a₁ ^ 3 * W'.a₃ * P y ^ 2 * P z ^ 2
    - 2 * W'.a₁ ^ 3 * W'.a₄ * P x * P y * P z ^ 2 + 2 * W'.a₁ ^ 2 * W'.a₂ ^ 2 * P x ^ 3 * P z
    - 2 * W'.a₁ ^ 2 * W'.a₂ * W'.a₃ * P x * P y * P z ^ 2
    + 3 * W'.a₁ ^ 2 * W'.a₂ * W'.a₄ * P x ^ 2 * P z ^ 2
    - 2 * W'.a₁ ^ 2 * W'.a₃ * W'.a₄ * P y * P z ^ 3 + W'.a₁ ^ 2 * W'.a₄ ^ 2 * P x * P z ^ 3
    + W'.a₁ * W'.a₂ * W'.a₃ ^ 2 * P y * P z ^ 3 + 2 * W'.a₁ * W'.a₂ * W'.a₃ * W'.a₄ * P x * P z ^ 3
    + W'.a₁ * W'.a₃ * W'.a₄ ^ 2 * P z ^ 4 - 2 * W'.a₂ ^ 2 * W'.a₃ ^ 2 * P x * P z ^ 3
    - W'.a₂ * W'.a₃ ^ 2 * W'.a₄ * P z ^ 4


lemma negDblY_eq' {P : Fin 3 → R} (hP : W'.Equation P) : W'.negDblY P * P z ^ 2 =
    -eval P W'.polynomialX * (eval P W'.polynomialX ^ 2
      - W'.a₁ * eval P W'.polynomialX * P z * (P y - W'.negY P)
      - W'.a₂ * P z ^ 2 * (P y - W'.negY P) ^ 2 - 2 * P x * P z * (P y - W'.negY P) ^ 2
      - P x * P z * (P y - W'.negY P) ^ 2) + P y * P z ^ 2 * (P y - W'.negY P) ^ 3 := by
  linear_combination (norm := (rw [negDblY, eval_polynomialX, negY]; ring1))
    -9 * (P y ^ 2 * P z + 2 * W'.a₁ * P x * P y * P z - 3 * P x ^ 3 - 3 * W'.a₂ * P x ^ 2 * P z)
      * (equation_iff _).mp hP


lemma negDblY_eq {P : Fin 3 → F} (hP : W.Equation P) (hPz : P z ≠ 0) : W.negDblY P =
    (-eval P W.polynomialX * (eval P W.polynomialX ^ 2
      - W.a₁ * eval P W.polynomialX * P z * (P y - W.negY P)
      - W.a₂ * P z ^ 2 * (P y - W.negY P) ^ 2 - 2 * P x * P z * (P y - W.negY P) ^ 2
      - P x * P z * (P y - W.negY P) ^ 2) + P y * P z ^ 2 * (P y - W.negY P) ^ 3) / P z ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : W.Equation P
    hPz : Ne (P 2) 0
    ⊢ Eq (W.negDblY P) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (Neg.neg ((MvPolynomial.ev …
  -/
  rw [← negDblY_eq' hP, mul_div_cancel_right₀ _ <| pow_ne_zero 2 hPz]
  /-
    🎉 no goals
  -/


lemma negDblY_smul (P : Fin 3 → R) (u : R) : W'.negDblY (u • P) = u ^ 4 * W'.negDblY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.negDblY (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W'.negDblY P))
  -/
  simp only [negDblY, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negDblY_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.negDblY P = -P y ^ 4 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.negDblY P) (Neg.neg (HPow.hPow (P 1) 4))
  -/
  rw [negDblY, hPz, X_eq_zero_of_Z_eq_zero hP hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negDblY_of_Y_eq' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W'.negY Q * P z) :
    W'.negDblY P * P z ^ 2 = -eval P W'.polynomialX ^ 3 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Eq (HMul.hMul (W'.negDblY P) (HPow.hPow (P 2) 2)) (Neg.neg (HPow.hPow ((MvPo …
  -/
  rw [negDblY_eq' hP, Y_eq_negY_of_Y_eq hQz hx hy hy']
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W'.negY Q) (P 2))
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg ((MvPolynomial.eval P) W'.polynomialX)) (H …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negDblY_of_Y_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W.negY Q * P z) :
    W.negDblY P = -W.dblU P := by
  rw [dblU, ← neg_div, ← negDblY_of_Y_eq' hP hQz hx hy hy',
    mul_div_cancel_right₀ _ <| pow_ne_zero 2 hPz]


private lemma toAffine_negAddY_of_eq {P : Fin 3 → F} (hPz : P z ≠ 0) {n d : F} (hd : d ≠ 0) :
    W.toAffine.negAddY (P x / P z) (P x / P z) (P y / P z) (-n / P z / d) =
      (-n * (n ^ 2 - W.a₁ * n * P z * d - W.a₂ * P z ^ 2 * d ^ 2 - 2 * P x * P z * d ^ 2
          - P x * P z * d ^ 2) + P y * P z ^ 2 * d ^ 3) / P z ^ 2 / (P z * d ^ 3) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (W.toAffine.negAddY (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (P 0) (P 2)) (HDiv …
  -/
  rw [Affine.negAddY, toAffine_addX_of_eq hPz hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HDiv.hDiv (Neg.neg n) (P 2)) d) (HSub.h …
  -/
  field_simp [mul_ne_zero hPz <| mul_ne_zero hPz <| pow_ne_zero 3 hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Ne (P 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul n (HSub.hSub (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negDblY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    W.negDblY P / W.dblZ P = W.toAffine.negAddY (P x / P z) (Q x / Q z) (P y / P z)
      (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)) := by
  rw [negDblY_eq hP hPz, dblZ, toAffine_slope_of_eq hP hQ hPz hQz hx hy, ← (X_eq_iff hPz hQz).mp hx,
    toAffine_negAddY_of_eq hPz <| sub_ne_zero.mpr <| Y_ne_negY_of_Y_ne' hP hQ hPz hQz hx hy]


variable (W') in
/-- The $Y$-coordinate of a representative of `2 • P` for a point `P`. -/
noncomputable def dblY (P : Fin 3 → R) : R :=
  W'.negY ![W'.dblX P, W'.negDblY P, W'.dblZ P]


lemma dblY_smul (P : Fin 3 → R) (u : R) : W'.dblY (u • P) = u ^ 4 * W'.dblY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblY (HSMul.hSMul u P)) (HMul.hMul (HPow.hPow u 4) (W'.dblY P))
  -/
  simp only [dblY, negY_eq, negDblY_smul, dblX_smul, dblZ_smul]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow u 4) (W'.negDblY P)) …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblY_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.dblY P = P y ^ 4 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.dblY P) (HPow.hPow (P 1) 4)
  -/
  rw [dblY, negY_eq, negDblY_of_Z_eq_zero hP hPz, dblX_of_Z_eq_zero hP hPz, dblZ_of_Z_eq_zero hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (Neg.neg (HPow.hPow (P 1) 4))) (HMul.hMul  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma dblY_of_Y_eq' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z)
    (hy' : P y * Q z = W'.negY Q * P z) : W'.dblY P * P z ^ 2 = eval P W'.polynomialX ^ 3 := by
  linear_combination (norm := (rw [dblY, negY_eq, dblX_of_Y_eq hP hPz hQz hx hy hy',
    dblZ_of_Y_eq hQz hx hy hy']; ring1)) -negDblY_of_Y_eq' hP hQz hx hy hy'


lemma dblY_of_Y_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W.negY Q * P z) :
    W.dblY P = W.dblU P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W.negY Q) (P 2))
    ⊢ Eq (W.dblY P) (W.dblU P)
  -/
  rw [dblU, ← dblY_of_Y_eq' hP hPz hQz hx hy hy', mul_div_cancel_right₀ _ <| pow_ne_zero 2 hPz]
  /-
    🎉 no goals
  -/


lemma dblY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    W.dblY P / W.dblZ P = W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
      (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)) := by
  erw [dblY, negY_of_Z_ne_zero <| dblZ_ne_zero_of_Y_ne' hP hQ hPz hQz hx hy,
    dblX_of_Z_ne_zero hP hQ hPz hQz hx hy, negDblY_of_Z_ne_zero hP hQ hPz hQz hx hy, Affine.addY]


variable (W') in
/-- The coordinates of a representative of `2 • P` for a point `P`. -/
noncomputable def dblXYZ (P : Fin 3 → R) : Fin 3 → R :=
  ![W'.dblX P, W'.dblY P, W'.dblZ P]


lemma dblXYZ_X (P : Fin 3 → R) : W'.dblXYZ P x = W'.dblX P :=
  rfl


lemma dblXYZ_Y (P : Fin 3 → R) : W'.dblXYZ P y = W'.dblY P :=
  rfl


lemma dblXYZ_Z (P : Fin 3 → R) : W'.dblXYZ P z = W'.dblZ P :=
  rfl


lemma dblXYZ_smul (P : Fin 3 → R) (u : R) : W'.dblXYZ (u • P) = u ^ 4 • W'.dblXYZ P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.dblXYZ (HSMul.hSMul u P)) (HSMul.hSMul (HPow.hPow u 4) (W'.dblXYZ P))
  -/
  rw [dblXYZ, dblX_smul, dblY_smul, dblZ_smul, smul_fin3, dblXYZ_X, dblXYZ_Y, dblXYZ_Z]
  /-
    🎉 no goals
  -/


lemma dblXYZ_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.dblXYZ P = P y ^ 4 • ![0, 1, 0] := by
  erw [dblXYZ, dblX_of_Z_eq_zero hP hPz, dblY_of_Z_eq_zero hP hPz, dblZ_of_Z_eq_zero hPz, smul_fin3,
    mul_zero, mul_one]


lemma dblXYZ_of_Y_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W.negY Q * P z) :
    W.dblXYZ P = W.dblU P • ![0, 1, 0] := by
  erw [dblXYZ, dblX_of_Y_eq hP hPz hQz hx hy hy', dblY_of_Y_eq hP hPz hQz hx hy hy',
    dblZ_of_Y_eq hQz hx hy hy', smul_fin3, mul_zero, mul_one]


lemma dblXYZ_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    W.dblXYZ P = W.dblZ P •
      ![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)), 1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W.negY Q) (P 2))
    ⊢ Eq (W.dblXYZ P) (HSMul.hSMul (W.dblZ P) (Matrix.vecCons (W.toAffine.addX (HD …
  -/
  have hZ : IsUnit <| W.dblZ P := isUnit_dblZ_of_Y_ne' hP hQ hPz hQz hx hy
  erw [dblXYZ, smul_fin3, ← dblX_of_Z_ne_zero hP hQ hPz hQz hx hy, hZ.mul_div_cancel,
    ← dblY_of_Z_ne_zero hP hQ hPz hQz hx hy, hZ.mul_div_cancel, mul_one]


/-- The unit associated to the addition of a non-2-torsion point `P` with its negation.
More specifically, the unit `u` such that `W.add P Q = u • ![0, 1, 0]` where `P x / P z = Q x / Q z`
but `P ≠ W.neg P`. -/
def addU (P Q : Fin 3 → F) : F :=
  -(P y * Q z - Q y * P z) ^ 3 / (P z * Q z)


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
    ⊢ Eq (WeierstrassCurve.Projective.addU (HSMul.hSMul u P) (HSMul.hSMul v Q)) (H …
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
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub (HMul.hMul (HMul.hMul u (P 1)) (HMul.hMu …
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
    ⊢ Eq (WeierstrassCurve.Projective.addU P Q) 0
  -/
  rw [addU, hPz, zero_mul, div_zero]
  /-
    🎉 no goals
  -/


lemma addU_of_Z_eq_zero_right {P Q : Fin 3 → F} (hQz : Q z = 0) : addU P Q = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    P Q : Fin 3 → F
    hQz : Eq (Q 2) 0
    ⊢ Eq (WeierstrassCurve.Projective.addU P Q) 0
  -/
  rw [addU, hQz, mul_zero <| P z, div_zero]
  /-
    🎉 no goals
  -/


lemma addU_ne_zero_of_Y_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hy : P y * Q z ≠ Q y * P z) : addU P Q ≠ 0 :=
  div_ne_zero (neg_ne_zero.mpr <| pow_ne_zero 3 <| sub_ne_zero.mpr hy) <| mul_ne_zero hPz hQz


lemma isUnit_addU_of_Y_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hy : P y * Q z ≠ Q y * P z) : IsUnit (addU P Q) :=
  (addU_ne_zero_of_Y_ne hPz hQz hy).isUnit


variable (W') in
/-- The $Z$-coordinate of a representative of `P + Q` for two distinct points `P` and `Q`.
Note that this returns the value 0 if the representatives of `P` and `Q` are equal. -/
def addZ (P Q : Fin 3 → R) : R :=
  -3 * P x ^ 2 * Q x * Q z + 3 * P x * Q x ^ 2 * P z + P y ^ 2 * Q z ^ 2 - Q y ^ 2 * P z ^ 2
    + W'.a₁ * P x * P y * Q z ^ 2 - W'.a₁ * Q x * Q y * P z ^ 2 - W'.a₂ * P x ^ 2 * Q z ^ 2
    + W'.a₂ * Q x ^ 2 * P z ^ 2 + W'.a₃ * P y * P z * Q z ^ 2 - W'.a₃ * Q y * P z ^ 2 * Q z
    - W'.a₄ * P x * P z * Q z ^ 2 + W'.a₄ * Q x * P z ^ 2 * Q z


lemma addZ_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q) :
    W'.addZ P Q * (P z * Q z) = (P x * Q z - Q x * P z) ^ 3 := by
  linear_combination (norm := (rw [addZ]; ring1))
    Q z ^ 3 * (equation_iff _).mp hP - P z ^ 3 * (equation_iff _).mp hQ


lemma addZ_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) : W.addZ P Q = (P x * Q z - Q x * P z) ^ 3 / (P z * Q z) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W.addZ P Q) (HDiv.hDiv (HPow.hPow (HSub.hSub (HMul.hMul (P 0) (Q 2)) (HM …
  -/
  rw [← addZ_eq' hP hQ, mul_div_cancel_right₀ _ <| mul_ne_zero hPz hQz]
  /-
    🎉 no goals
  -/


lemma addZ_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addZ (u • P) (v • Q) = (u * v) ^ 2 * W'.addZ P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addZ (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (HMul …
  -/
  simp only [addZ, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_self (P : Fin 3 → R) : W'.addZ P P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.addZ P P) 0
  -/
  rw [addZ]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_of_Z_eq_zero_left [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) : W'.addZ P Q = P y ^ 2 * Q z * Q z := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.addZ P Q) (HMul.hMul (HMul.hMul (HPow.hPow (P 1) 2) (Q 2)) (Q 2))
  -/
  rw [addZ, hPz, X_eq_zero_of_Z_eq_zero hP hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_of_Z_eq_zero_right [NoZeroDivisors R] {P Q : Fin 3 → R} (hQ : W'.Equation Q)
    (hQz : Q z = 0) : W'.addZ P Q = -(Q y ^ 2 * P z) * P z := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (W'.addZ P Q) (HMul.hMul (Neg.neg (HMul.hMul (HPow.hPow (Q 1) 2) (P 2)))  …
  -/
  rw [addZ, hQz, X_eq_zero_of_Z_eq_zero hQ hQz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addZ_of_X_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) : W'.addZ P Q = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (W'.addZ P Q) 0
  -/
  apply eq_zero_of_ne_zero_of_mul_right_eq_zero <| mul_ne_zero hPz hQz
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HMul.hMul (W'.addZ P Q) (HMul.hMul (P 2) (Q 2))) 0
  -/
  rw [addZ_eq' hP hQ, hx, sub_self, zero_pow three_ne_zero]
  /-
    🎉 no goals
  -/


lemma addZ_ne_zero_of_X_ne [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hQ : W'.Equation Q) (hx : P x * Q z ≠ Q x * P z) : W'.addZ P Q ≠ 0 :=
  addZ_eq' hP hQ ▸ left_ne_zero_of_mul <| pow_ne_zero 3 <| sub_ne_zero.mpr hx


lemma isUnit_addZ_of_X_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q)
    (hx : P x * Q z ≠ Q x * P z) : IsUnit <| W.addZ P Q :=
  (addZ_ne_zero_of_X_ne hP hQ hx).isUnit


private lemma toAffine_slope_of_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z ≠ Q x * P z) :
    W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z) =
      (P y * Q z - Q y * P z) / (P x * Q z - Q x * P z) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (W.toAffine.slope (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) (HDiv.h …
  -/
  field_simp [Affine.slope_of_X_ne <| by rwa [ne_eq, ← X_eq_iff hPz hQz]]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (P 2) (Q 1))) (H …
  -/
  ring1
  /-
    🎉 no goals
  -/


variable (W') in
/-- The $X$-coordinate of a representative of `P + Q` for two distinct points `P` and `Q`.
Note that this returns the value 0 if the representatives of `P` and `Q` are equal. -/
def addX (P Q : Fin 3 → R) : R :=
  -P x * Q y ^ 2 * P z + Q x * P y ^ 2 * Q z - 2 * P x * P y * Q y * Q z + 2 * Q x * P y * Q y * P z
    - W'.a₁ * P x ^ 2 * Q y * Q z + W'.a₁ * Q x ^ 2 * P y * P z + W'.a₂ * P x ^ 2 * Q x * Q z
    - W'.a₂ * P x * Q x ^ 2 * P z - W'.a₃ * P x * P y * Q z ^ 2 + W'.a₃ * Q x * Q y * P z ^ 2
    - 2 * W'.a₃ * P x * Q y * P z * Q z + 2 * W'.a₃ * Q x * P y * P z * Q z
    + W'.a₄ * P x ^ 2 * Q z ^ 2 - W'.a₄ * Q x ^ 2 * P z ^ 2 + 3 * W'.a₆ * P x * P z * Q z ^ 2
    - 3 * W'.a₆ * Q x * P z ^ 2 * Q z


lemma addX_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q) :
    W'.addX P Q * (P z * Q z) ^ 2 =
      ((P y * Q z - Q y * P z) ^ 2 * P z * Q z
        + W'.a₁ * (P y * Q z - Q y * P z) * P z * Q z * (P x * Q z - Q x * P z)
        - W'.a₂ * P z * Q z * (P x * Q z - Q x * P z) ^ 2 - P x * Q z * (P x * Q z - Q x * P z) ^ 2
        - Q x * P z * (P x * Q z - Q x * P z) ^ 2) * (P x * Q z - Q x * P z) := by
  linear_combination (norm := (rw [addX]; ring1))
    (2 * Q x * P z * Q z ^ 3 - P x * Q z ^ 4) * (equation_iff _).mp hP
      + (Q x * P z ^ 4 - 2 * P x * P z ^ 3 * Q z) * (equation_iff _).mp hQ


lemma addX_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) : W.addX P Q =
      ((P y * Q z - Q y * P z) ^ 2 * P z * Q z
        + W.a₁ * (P y * Q z - Q y * P z) * P z * Q z * (P x * Q z - Q x * P z)
        - W.a₂ * P z * Q z * (P x * Q z - Q x * P z) ^ 2 - P x * Q z * (P x * Q z - Q x * P z) ^ 2
        - Q x * P z * (P x * Q z - Q x * P z) ^ 2) * (P x * Q z - Q x * P z) / (P z * Q z) ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W.addX P Q) (HDiv.hDiv (HMul.hMul (HSub.hSub (HSub.hSub (HSub.hSub (HAdd …
  -/
  rw [← addX_eq' hP hQ, mul_div_cancel_right₀ _ <| pow_ne_zero 2 <| mul_ne_zero hPz hQz]
  /-
    🎉 no goals
  -/


lemma addX_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addX (u • P) (v • Q) = (u * v) ^ 2 * W'.addX P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addX (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (HMul …
  -/
  simp only [addX, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_self (P : Fin 3 → R) : W'.addX P P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.addX P P) 0
  -/
  rw [addX]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_Z_eq_zero_left [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) : W'.addX P Q = P y ^ 2 * Q z * Q x := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.addX P Q) (HMul.hMul (HMul.hMul (HPow.hPow (P 1) 2) (Q 2)) (Q 0))
  -/
  rw [addX, hPz, X_eq_zero_of_Z_eq_zero hP hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_Z_eq_zero_right [NoZeroDivisors R] {P Q : Fin 3 → R} (hQ : W'.Equation Q)
    (hQz : Q z = 0) : W'.addX P Q = -(Q y ^ 2 * P z) * P x := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (W'.addX P Q) (HMul.hMul (Neg.neg (HMul.hMul (HPow.hPow (Q 1) 2) (P 2)))  …
  -/
  rw [addX, hQz, X_eq_zero_of_Z_eq_zero hQ hQz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_X_eq [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) : W'.addX P Q = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (W'.addX P Q) 0
  -/
  apply eq_zero_of_ne_zero_of_mul_right_eq_zero <| pow_ne_zero 2 <| mul_ne_zero hPz hQz
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HMul.hMul (W'.addX P Q) (HPow.hPow (HMul.hMul (P 2) (Q 2)) 2)) 0
  -/
  rw [addX_eq' hP hQ, hx]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HMul.hMul (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


private lemma toAffine_addX_of_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) {n d : F}
    (hd : d ≠ 0) : W.toAffine.addX (P x / P z) (Q x / Q z) (n / d) =
      (n ^ 2 * P z * Q z + W.a₁ * n * P z * Q z * d - W.a₂ * P z * Q z * d ^ 2 - P x * Q z * d ^ 2
        - Q x * P z * d ^ 2) * d / (P z * Q z) ^ 2 / (d ^ 3 / (P z * Q z)) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (W.toAffine.addX (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) (HDiv.hD …
  -/
  field_simp [hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HMul.hMul (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ≠ Q x * P z) : W.addX P Q / W.addZ P Q =
    W.toAffine.addX (P x / P z) (Q x / Q z)
      (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)) := by
  rw [addX_eq hP hQ hPz hQz, addZ_eq hP hQ hPz hQz, toAffine_slope_of_ne hPz hQz hx,
    toAffine_addX_of_ne hPz hQz <| sub_ne_zero.mpr hx]


variable (W') in
/-- The $Y$-coordinate of a representative of `-(P + Q)` for two distinct points `P` and `Q`.
Note that this returns the value 0 if the representatives of `P` and `Q` are equal. -/
def negAddY (P Q : Fin 3 → R) : R :=
  -3 * P x ^ 2 * Q x * Q y + 3 * P x * Q x ^ 2 * P y - P y ^ 2 * Q y * Q z + P y * Q y ^ 2 * P z
    + W'.a₁ * P x * Q y ^ 2 * P z - W'.a₁ * Q x * P y ^ 2 * Q z - W'.a₂ * P x ^ 2 * Q y * Q z
    + W'.a₂ * Q x ^ 2 * P y * P z + 2 * W'.a₂ * P x * Q x * P y * Q z
    - 2 * W'.a₂ * P x * Q x * Q y * P z - W'.a₃ * P y ^ 2 * Q z ^ 2 + W'.a₃ * Q y ^ 2 * P z ^ 2
    + W'.a₄ * P x * P y * Q z ^ 2 - 2 * W'.a₄ * P x * Q y * P z * Q z
    + 2 * W'.a₄ * Q x * P y * P z * Q z - W'.a₄ * Q x * Q y * P z ^ 2
    + 3 * W'.a₆ * P y * P z * Q z ^ 2 - 3 * W'.a₆ * Q y * P z ^ 2 * Q z


lemma negAddY_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q) :
    W'.negAddY P Q * (P z * Q z) ^ 2 =
      (P y * Q z - Q y * P z) * ((P y * Q z - Q y * P z) ^ 2 * P z * Q z
        + W'.a₁ * (P y * Q z - Q y * P z) * P z * Q z * (P x * Q z - Q x * P z)
        - W'.a₂ * P z * Q z * (P x * Q z - Q x * P z) ^ 2 - P x * Q z * (P x * Q z - Q x * P z) ^ 2
        - Q x * P z * (P x * Q z - Q x * P z) ^ 2 - P x * Q z * (P x * Q z - Q x * P z) ^ 2)
        + P y * Q z * (P x * Q z - Q x * P z) ^ 3 := by
  linear_combination (norm := (rw [negAddY]; ring1))
    (2 * Q y * P z * Q z ^ 3 - P y * Q z ^ 4) * (equation_iff _).mp hP
      + (Q y * P z ^ 4 - 2 * P y * P z ^ 3 * Q z) * (equation_iff _).mp hQ


lemma negAddY_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) : W.negAddY P Q =
      ((P y * Q z - Q y * P z) * ((P y * Q z - Q y * P z) ^ 2 * P z * Q z
        + W.a₁ * (P y * Q z - Q y * P z) * P z * Q z * (P x * Q z - Q x * P z)
        - W.a₂ * P z * Q z * (P x * Q z - Q x * P z) ^ 2 - P x * Q z * (P x * Q z - Q x * P z) ^ 2
        - Q x * P z * (P x * Q z - Q x * P z) ^ 2 - P x * Q z * (P x * Q z - Q x * P z) ^ 2)
        + P y * Q z * (P x * Q z - Q x * P z) ^ 3) / (P z * Q z) ^ 2 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W.negAddY P Q) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (P …
  -/
  rw [← negAddY_eq' hP hQ, mul_div_cancel_right₀ _ <| pow_ne_zero 2 <| mul_ne_zero hPz hQz]
  /-
    🎉 no goals
  -/


lemma negAddY_smul (P Q : Fin 3 → R) (u v : R) :
    W'.negAddY (u • P) (v • Q) = (u * v) ^ 2 * W'.negAddY P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.negAddY (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (H …
  -/
  simp only [negAddY, smul_fin3_ext]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_self (P : Fin 3 → R) : W'.negAddY P P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.negAddY P P) 0
  -/
  rw [negAddY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_of_Z_eq_zero_left [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) : W'.negAddY P Q = P y ^ 2 * Q z * W'.negY Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (W'.negAddY P Q) (HMul.hMul (HMul.hMul (HPow.hPow (P 1) 2) (Q 2)) (W'.neg …
  -/
  rw [negAddY, hPz, X_eq_zero_of_Z_eq_zero hP hPz, negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_of_Z_eq_zero_right [NoZeroDivisors R] {P Q : Fin 3 → R} (hQ : W'.Equation Q)
    (hQz : Q z = 0) : W'.negAddY P Q = -(Q y ^ 2 * P z) * W'.negY P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (W'.negAddY P Q) (HMul.hMul (Neg.neg (HMul.hMul (HPow.hPow (Q 1) 2) (P 2) …
  -/
  rw [negAddY, hQz, X_eq_zero_of_Z_eq_zero hQ hQz, negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_of_X_eq' {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hx : P x * Q z = Q x * P z) :
    W'.negAddY P Q * (P z * Q z) ^ 2 = (P y * Q z - Q y * P z) ^ 3 * (P z * Q z) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HMul.hMul (W'.negAddY P Q) (HPow.hPow (HMul.hMul (P 2) (Q 2)) 2)) (HMul. …
  -/
  rw [negAddY_eq' hP hQ, hx]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hQ : W'.Equation Q
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) : W.negAddY P Q = -addU P Q := by
  rw [addU, neg_div, neg_neg, ← mul_div_mul_right _ _ <| mul_ne_zero hPz hQz,
    ← negAddY_of_X_eq' hP hQ hx, ← sq,
    mul_div_cancel_right₀ _ <| pow_ne_zero 2 <| mul_ne_zero hPz hQz]


private lemma toAffine_negAddY_of_ne {P Q : Fin 3 → F} (hPz : P z ≠ 0) (hQz : Q z ≠ 0) {n d : F}
    (hd : d ≠ 0) : W.toAffine.negAddY (P x / P z) (Q x / Q z) (P y / P z) (n / d) =
      (n * (n ^ 2 * P z * Q z + W.a₁ * n * P z * Q z * d - W.a₂ * P z * Q z * d ^ 2
        - P x * Q z * d ^ 2 - Q x * P z * d ^ 2 - P x * Q z * d ^ 2) + P y * Q z * d ^ 3)
        / (P z * Q z) ^ 2 / (d ^ 3 / (P z * Q z)) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (W.toAffine.negAddY (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) (HDiv …
  -/
  rw [Affine.negAddY, toAffine_addX_of_ne hPz hQz hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv n d) (HSub.hSub (HDiv.hDiv (HDiv.hDiv (H …
  -/
  field_simp [mul_ne_zero (pow_ne_zero 2 <| mul_ne_zero hPz hQz) <| pow_ne_zero 3 hd]
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    n d : F
    hd : Ne d 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HMul.hMul n (HSub.hSub (HMul.hMul (HMul …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ≠ Q x * P z) : W.negAddY P Q / W.addZ P Q =
      W.toAffine.negAddY (P x / P z) (Q x / Q z) (P y / P z)
        (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)) := by
  rw [negAddY_eq hP hQ hPz hQz, addZ_eq hP hQ hPz hQz, toAffine_slope_of_ne hPz hQz hx,
    toAffine_negAddY_of_ne hPz hQz <| sub_ne_zero.mpr hx]


variable (W') in
/-- The $Y$-coordinate of a representative of `P + Q` for two distinct points `P` and `Q`.
Note that this returns the value 0 if the representatives of `P` and `Q` are equal. -/
def addY (P Q : Fin 3 → R) : R :=
  W'.negY ![W'.addX P Q, W'.negAddY P Q, W'.addZ P Q]


lemma addY_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addY (u • P) (v • Q) = (u * v) ^ 2 * W'.addY P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addY (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HMul.hMul (HPow.hPow (HMul …
  -/
  simp only [addY, negY_eq, negAddY_smul, addX_smul, addZ_smul]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HPow.hPow (HMul.hMul u v) 2) ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addY_self (P : Fin 3 → R) : W'.addY P P = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.addY P P) 0
  -/
  simp only [addY, negY_eq, negAddY_self, addX_self, addZ_self, neg_zero, mul_zero, sub_zero]
  /-
    🎉 no goals
  -/


lemma addY_of_Z_eq_zero_left [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) : W'.addY P Q = P y ^ 2 * Q z * Q y := by
  rw [addY, negY_eq, negAddY_of_Z_eq_zero_left hP hPz, negY, addX_of_Z_eq_zero_left hP hPz,
    addZ_of_Z_eq_zero_left hP hPz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (HMul.hMul (HPow.hPow (P 1) 2)  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addY_of_Z_eq_zero_right [NoZeroDivisors R] {P Q : Fin 3 → R} (hQ : W'.Equation Q)
    (hQz : Q z = 0) : W'.addY P Q = -(Q y ^ 2 * P z) * P y := by
  rw [addY, negY_eq, negAddY_of_Z_eq_zero_right hQ hQz, negY, addX_of_Z_eq_zero_right hQ hQz,
    addZ_of_Z_eq_zero_right hQ hQz]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hQz : Eq (Q 2) 0
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (HMul.hMul (Neg.neg (HMul.hMul (HPow.hPow  …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addY_of_X_eq' [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P) (hQ : W'.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) :
    W'.addY P Q * (P z * Q z) ^ 3 = -(P y * Q z - Q y * P z) ^ 3 * (P z * Q z) ^ 2 := by
  linear_combination (norm := (rw [addY, negY_eq, addX_of_X_eq hP hQ hPz hQz hx,
    addZ_of_X_eq hP hQ hPz hQz hx]; ring1)) -(P z * Q z) * negAddY_of_X_eq' hP hQ hx


lemma addY_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) : W.addY P Q = addU P Q := by
  rw [addU, ← mul_div_mul_right _ _ <| pow_ne_zero 2 <| mul_ne_zero hPz hQz,
    ← addY_of_X_eq' hP hQ hPz hQz hx, ← pow_succ',
    mul_div_cancel_right₀ _ <| pow_ne_zero 3 <| mul_ne_zero hPz hQz]


lemma addY_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ≠ Q x * P z) : W.addY P Q / W.addZ P Q =
      W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
        (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)) := by
  erw [addY, negY_of_Z_ne_zero <| addZ_ne_zero_of_X_ne hP hQ hx, addX_of_Z_ne_zero hP hQ hPz hQz hx,
    negAddY_of_Z_ne_zero hP hQ hPz hQz hx, Affine.addY]


variable (W') in
/-- The coordinates of a representative of `P + Q` for two distinct points `P` and `Q`.
Note that this returns the value `![0, 0, 0]` if the representatives of `P` and `Q` are equal. -/
noncomputable def addXYZ (P Q : Fin 3 → R) : Fin 3 → R :=
  ![W'.addX P Q, W'.addY P Q, W'.addZ P Q]


lemma addXYZ_X (P Q : Fin 3 → R) : W'.addXYZ P Q x = W'.addX P Q :=
  rfl


lemma addXYZ_Y (P Q : Fin 3 → R) : W'.addXYZ P Q y = W'.addY P Q :=
  rfl


lemma addXYZ_Z (P Q : Fin 3 → R) : W'.addXYZ P Q z = W'.addZ P Q :=
  rfl


lemma addXYZ_smul (P Q : Fin 3 → R) (u v : R) :
    W'.addXYZ (u • P) (v • Q) = (u * v) ^ 2 • W'.addXYZ P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    u v : R
    ⊢ Eq (W'.addXYZ (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HSMul.hSMul (HPow.hPow ( …
  -/
  rw [addXYZ, addX_smul, addY_smul, addZ_smul, smul_fin3, addXYZ_X, addXYZ_Y, addXYZ_Z]
  /-
    🎉 no goals
  -/


lemma addXYZ_self (P : Fin 3 → R) : W'.addXYZ P P = ![0, 0, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.addXYZ P P) (Matrix.vecCons 0 (Matrix.vecCons 0 (Matrix.vecCons 0 Mat …
  -/
  rw [addXYZ, addX_self, addY_self, addZ_self]
  /-
    🎉 no goals
  -/


lemma addXYZ_of_Z_eq_zero_left [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) : W'.addXYZ P Q = (P y ^ 2 * Q z) • Q := by
  rw [addXYZ, addX_of_Z_eq_zero_left hP hPz, addY_of_Z_eq_zero_left hP hPz,
    addZ_of_Z_eq_zero_left hP hPz, smul_fin3]


lemma addXYZ_of_Z_eq_zero_right [NoZeroDivisors R] {P Q : Fin 3 → R} (hQ : W'.Equation Q)
    (hQz : Q z = 0) : W'.addXYZ P Q = -(Q y ^ 2 * P z) • P := by
  rw [addXYZ, addX_of_Z_eq_zero_right hQ hQz, addY_of_Z_eq_zero_right hQ hQz,
    addZ_of_Z_eq_zero_right hQ hQz, smul_fin3]


lemma addXYZ_of_X_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) : W.addXYZ P Q = addU P Q • ![0, 1, 0] := by
  erw [addXYZ, addX_of_X_eq hP hQ hPz hQz hx, addY_of_X_eq hP hQ hPz hQz hx,
    addZ_of_X_eq hP hQ hPz hQz hx, smul_fin3, mul_zero, mul_one]


lemma addXYZ_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ≠ Q x * P z) : W.addXYZ P Q = W.addZ P Q •
      ![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)), 1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (W.addXYZ P Q) (HSMul.hSMul (W.addZ P Q) (Matrix.vecCons (W.toAffine.addX …
  -/
  have hZ : IsUnit <| W.addZ P Q := isUnit_addZ_of_X_ne hP hQ hx
  erw [addXYZ, smul_fin3, ← addX_of_Z_ne_zero hP hQ hPz hQz hx, hZ.mul_div_cancel,
    ← addY_of_Z_ne_zero hP hQ hPz hQz hx, hZ.mul_div_cancel, mul_one]


variable (W') in
/-- The negation of a point representative. -/
def neg (P : Fin 3 → R) : Fin 3 → R :=
  ![P x, W'.negY P, P z]


lemma neg_X (P : Fin 3 → R) : W'.neg P x = P x :=
  rfl


lemma neg_Y (P : Fin 3 → R) : W'.neg P y = W'.negY P :=
  rfl


lemma neg_Z (P : Fin 3 → R) : W'.neg P z = P z :=
  rfl


lemma neg_smul (P : Fin 3 → R) (u : R) : W'.neg (u • P) = u • W'.neg P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    u : R
    ⊢ Eq (W'.neg (HSMul.hSMul u P)) (HSMul.hSMul u (W'.neg P))
  -/
  simpa only [neg, negY_smul] using (smul_fin3 (W'.neg P) u).symm
  /-
    🎉 no goals
  -/


lemma neg_smul_equiv (P : Fin 3 → R) {u : R} (hu : IsUnit u) : W'.neg (u • P) ≈ W'.neg P :=
  ⟨hu.unit, (neg_smul ..).symm⟩


lemma neg_equiv {P Q : Fin 3 → R} (h : P ≈ Q) : W'.neg P ≈ W'.neg Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    ⊢ HasEquiv.Equiv (W'.neg P) (W'.neg Q)
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    Q : Fin 3 → R
    u : Units R
    ⊢ HasEquiv.Equiv (W'.neg ((fun m => HSMul.hSMul m Q) u)) (W'.neg Q)
  -/
  exact neg_smul_equiv Q u.isUnit
  /-
    🎉 no goals
  -/


lemma neg_of_Z_eq_zero [NoZeroDivisors R] {P : Fin 3 → R} (hP : W'.Equation P) (hPz : P z = 0) :
    W'.neg P = -P y • ![0, 1, 0] := by
  erw [neg, X_eq_zero_of_Z_eq_zero hP hPz, negY_of_Z_eq_zero hP hPz, hPz, smul_fin3, mul_zero,
    mul_one]


lemma neg_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.neg P = P z • ![P x / P z, W.toAffine.negY (P x / P z) (P y / P z), 1] := by
  erw [neg, smul_fin3, mul_div_cancel₀ _ hPz, ← negY_of_Z_ne_zero hPz, mul_div_cancel₀ _ hPz,
    mul_one]


private lemma nonsingular_neg_of_Z_ne_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) :
    W.Nonsingular ![P x / P z, W.toAffine.negY (P x / P z) (P y / P z), 1] :=
  (nonsingular_some ..).mpr <| Affine.nonsingular_neg <| (nonsingular_of_Z_ne_zero hPz).mp hP


lemma nonsingular_neg {P : Fin 3 → F} (hP : W.Nonsingular P) : W.Nonsingular <| W.neg P := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : W.Nonsingular P
    ⊢ W.Nonsingular (W.neg P)
  -/
  by_cases hPz : P z = 0
  · simp only [neg_of_Z_eq_zero hP.left hPz, nonsingular_smul _ (isUnit_Y_of_Z_eq_zero hP hPz).neg,
      nonsingular_zero]
  · simp only [neg_of_Z_ne_zero hPz, nonsingular_smul _ <| Ne.isUnit hPz,
      nonsingular_neg_of_Z_ne_zero hP hPz]


lemma addZ_neg (P : Fin 3 → R) : W'.addZ P (W'.neg P) = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.addZ P (W'.neg P)) 0
  -/
  rw [addZ, neg_X, neg_Y, neg_Z, negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma addX_neg (P : Fin 3 → R) : W'.addX P (W'.neg P) = 0 := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (W'.addX P (W'.neg P)) 0
  -/
  rw [addX, neg_X, neg_Y, neg_Z, negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma negAddY_neg {P : Fin 3 → R} (hP : W'.Equation P) : W'.negAddY P (W'.neg P) = W'.dblZ P := by
  linear_combination (norm := (rw [negAddY, neg_X, neg_Y, neg_Z, dblZ, negY]; ring1))
    -3 * (P y - W'.negY P) * (equation_iff _).mp hP


lemma addY_neg {P : Fin 3 → R} (hP : W'.Equation P) : W'.addY P (W'.neg P) = -W'.dblZ P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addY P (W'.neg P)) (Neg.neg (W'.dblZ P))
  -/
  rw [addY, negY_eq, addX_neg, negAddY_neg hP, addZ_neg, mul_zero, sub_zero, mul_zero, sub_zero]
  /-
    🎉 no goals
  -/


lemma addXYZ_neg {P : Fin 3 → R} (hP : W'.Equation P) :
    W'.addXYZ P (W'.neg P) = -W'.dblZ P • ![0, 1, 0] := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P : Fin 3 → R
    hP : W'.Equation P
    ⊢ Eq (W'.addXYZ P (W'.neg P)) (HSMul.hSMul (Neg.neg (W'.dblZ P)) (Matrix.vecCo …
  -/
  erw [addXYZ, addX_neg, addY_neg hP, addZ_neg, smul_fin3, mul_zero, mul_one]
  /-
    🎉 no goals
  -/


variable (W') in
/-- The negation of a point class. If `P` is a point representative,
then `W'.negMap ⟦P⟧` is definitionally equivalent to `W'.neg P`. -/
def negMap (P : PointClass R) : PointClass R :=
  P.map W'.neg fun _ _ => neg_equiv


lemma negMap_eq (P : Fin 3 → R) : W'.negMap ⟦P⟧ = ⟦W'.neg P⟧ :=
  rfl


lemma negMap_of_Z_eq_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z = 0) :
    W.negMap ⟦P⟧ = ⟦![0, 1, 0]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : W.Nonsingular P
    hPz : Eq (P 2) 0
    ⊢ Eq (W.negMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P)) (Qu …
  -/
  rw [negMap_eq, neg_of_Z_eq_zero hP.left hPz, smul_eq _ (isUnit_Y_of_Z_eq_zero hP hPz).neg]
  /-
    🎉 no goals
  -/


lemma negMap_of_Z_ne_zero {P : Fin 3 → F} (hPz : P z ≠ 0) :
    W.negMap ⟦P⟧ = ⟦![P x / P z, W.toAffine.negY (P x / P z) (P y / P z), 1]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
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
    W : WeierstrassCurve.Projective F
    P : WeierstrassCurve.Projective.PointClass F
    hP : W.NonsingularLift P
    ⊢ W.NonsingularLift (W.negMap P)
  -/
  rcases P with ⟨_⟩
  /-
    case mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : WeierstrassCurve.Projective.PointClass F
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
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : HasEquiv.Equiv P Q
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    ⊢ Eq (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HSMul.hSMul (HPow.hPow u 4) …
  -/
  rw [add_of_equiv <| (smul_equiv_smul P Q hu hv).mpr h, dblXYZ_smul, add_of_equiv h]
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
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝ : CommRing R
    P Q : Fin 3 → R
    h : Not (HasEquiv.Equiv P Q)
    u v : R
    hu : IsUnit u
    hv : IsUnit v
    ⊢ Eq (W'.add (HSMul.hSMul u P) (HSMul.hSMul v Q)) (HSMul.hSMul (HPow.hPow (HMu …
  -/
  rw [add_of_not_equiv <| h.comp (smul_equiv_smul P Q hu hv).mp, addXYZ_smul, add_of_not_equiv h]
  /-
    🎉 no goals
  -/


lemma add_smul_equiv (P Q : Fin 3 → R) {u v : R} (hu : IsUnit u) (hv : IsUnit v) :
    W'.add (u • P) (v • Q) ≈ W'.add P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
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
      W' : WeierstrassCurve.Projective R
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
      W' : WeierstrassCurve.Projective R
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
    W' : WeierstrassCurve.Projective R
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
    W' : WeierstrassCurve.Projective R
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
    (hPz : P z = 0) (hQz : Q z = 0) : W.add P Q = P y ^ 4 • ![0, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    hPz : Eq (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ Eq (W.add P Q) (HSMul.hSMul (HPow.hPow (P 1) 4) (Matrix.vecCons 0 (Matrix.ve …
  -/
  rw [add, if_pos <| equiv_of_Z_eq_zero hP hQ hPz hQz, dblXYZ_of_Z_eq_zero hP.left hPz]
  /-
    🎉 no goals
  -/


lemma add_of_Z_eq_zero_left [NoZeroDivisors R] {P Q : Fin 3 → R} (hP : W'.Equation P)
    (hPz : P z = 0) (hQz : Q z ≠ 0) : W'.add P Q = (P y ^ 2 * Q z) • Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hP : W'.Equation P
    hPz : Eq (P 2) 0
    hQz : Ne (Q 2) 0
    ⊢ Eq (W'.add P Q) (HSMul.hSMul (HMul.hMul (HPow.hPow (P 1) 2) (Q 2)) Q)
  -/
  rw [add, if_neg <| not_equiv_of_Z_eq_zero_left hPz hQz, addXYZ_of_Z_eq_zero_left hP hPz]
  /-
    🎉 no goals
  -/


lemma add_of_Z_eq_zero_right [NoZeroDivisors R] {P Q : Fin 3 → R} (hQ : W'.Equation Q)
    (hPz : P z ≠ 0) (hQz : Q z = 0) : W'.add P Q = -(Q y ^ 2 * P z) • P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : NoZeroDivisors R
    P Q : Fin 3 → R
    hQ : W'.Equation Q
    hPz : Ne (P 2) 0
    hQz : Eq (Q 2) 0
    ⊢ Eq (W'.add P Q) (HSMul.hSMul (Neg.neg (HMul.hMul (HPow.hPow (Q 1) 2) (P 2))) …
  -/
  rw [add, if_neg <| not_equiv_of_Z_eq_zero_right hPz hQz, addXYZ_of_Z_eq_zero_right hQ hQz]
  /-
    🎉 no goals
  -/


lemma add_of_Y_eq {P Q : Fin 3 → F} (hP : W.Equation P) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hx : P x * Q z = Q x * P z) (hy : P y * Q z = Q y * P z) (hy' : P y * Q z = W.negY Q * P z) :
    W.add P Q = W.dblU P • ![0, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W.negY Q) (P 2))
    ⊢ Eq (W.add P Q) (HSMul.hSMul (W.dblU P) (Matrix.vecCons 0 (Matrix.vecCons 1 ( …
  -/
  rw [add, if_pos <| equiv_of_X_eq_of_Y_eq hPz hQz hx hy, dblXYZ_of_Y_eq hP hPz hQz hx hy hy']
  /-
    🎉 no goals
  -/


lemma add_of_Y_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ Q y * P z) :
    W.add P Q = addU P Q • ![0, 1, 0] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy : Ne (HMul.hMul (P 1) (Q 2)) (HMul.hMul (Q 1) (P 2))
    ⊢ Eq (W.add P Q) (HSMul.hSMul (WeierstrassCurve.Projective.addU P Q) (Matrix.v …
  -/
  rw [add, if_neg <| not_equiv_of_Y_ne hy, addXYZ_of_X_eq hP hQ hPz hQz hx]
  /-
    🎉 no goals
  -/


lemma add_of_Y_ne' {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy : P y * Q z ≠ W.negY Q * P z) :
    W.add P Q = W.dblZ P •
      ![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)), 1] := by
  rw [add, if_pos <| equiv_of_X_eq_of_Y_eq hPz hQz hx <| Y_eq_of_Y_ne' hP hQ hPz hQz hx hy,
    dblXYZ_of_Z_ne_zero hP hQ hPz hQz hx hy]


lemma add_of_X_ne {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hx : P x * Q z ≠ Q x * P z) : W.add P Q = W.addZ P Q •
      ![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)), 1] := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Ne (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    ⊢ Eq (W.add P Q) (HSMul.hSMul (W.addZ P Q) (Matrix.vecCons (W.toAffine.addX (H …
  -/
  rw [add, if_neg <| not_equiv_of_X_ne hx, addXYZ_of_Z_ne_zero hP hQ hPz hQz hx]
  /-
    🎉 no goals
  -/


private lemma nonsingular_add_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P)
    (hQ : W.Nonsingular Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hxy : P x * Q z = Q x * P z → P y * Q z ≠ W.negY Q * P z) : W.Nonsingular
      ![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)), 1] :=
  (nonsingular_some ..).mpr <| Affine.nonsingular_add ((nonsingular_of_Z_ne_zero hPz).mp hP)
                                               /-
                                                 F : Type v
                                                 inst✝ : Field F
                                                 W : WeierstrassCurve.Projective F
                                                 P Q : Fin 3 → F
                                                 hP : W.Nonsingular P
                                                 hQ : W.Nonsingular Q
                                                 hPz : Ne (P 2) 0
                                                 hQz : Ne (Q 2) 0
                                                 hxy : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul (P 1) …
                                                 ⊢ Eq (HDiv.hDiv (P 0) (P 2)) (HDiv.hDiv (Q 0) (Q 2)) → Ne (HDiv.hDiv (P 1) (P  …
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
    W : WeierstrassCurve.Projective F
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
      W : WeierstrassCurve.Projective F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Eq (P 2) 0
      ⊢ W.Nonsingular (W.add P Q)
    -/
  · by_cases hQz : Q z = 0
    · simp only [add_of_Z_eq_zero hP hQ hPz hQz,
        nonsingular_smul _ <| (isUnit_Y_of_Z_eq_zero hP hPz).pow 4, nonsingular_zero]
    · simpa only [add_of_Z_eq_zero_left hP.left hPz hQz,
        nonsingular_smul _ <| ((isUnit_Y_of_Z_eq_zero hP hPz).pow 2).mul <| Ne.isUnit hQz]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Not (Eq (P 2) 0)
      ⊢ W.Nonsingular (W.add P Q)
    -/
  · by_cases hQz : Q z = 0
    · simpa only [add_of_Z_eq_zero_right hQ.left hPz hQz,
        nonsingular_smul _ (((isUnit_Y_of_Z_eq_zero hQ hQz).pow 2).mul <| Ne.isUnit hPz).neg]
      /-
        case neg
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Projective F
        P Q : Fin 3 → F
        hP : W.Nonsingular P
        hQ : W.Nonsingular Q
        hPz : Not (Eq (P 2) 0)
        hQz : Not (Eq (Q 2) 0)
        ⊢ W.Nonsingular (W.add P Q)
      -/
    · by_cases hxy : P x * Q z = Q x * P z → P y * Q z ≠ W.negY Q * P z
        /-
          case pos
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Projective F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul (P 1) …
          ⊢ W.Nonsingular (W.add P Q)
        -/
      · by_cases hx : P x * Q z = Q x * P z
        · simp only [add_of_Y_ne' hP.left hQ.left hPz hQz hx <| hxy hx,
            nonsingular_smul _ <| isUnit_dblZ_of_Y_ne' hP.left hQ.left hPz hQz hx <| hxy hx,
            nonsingular_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        · simp only [add_of_X_ne hP.left hQ.left hPz hQz hx,
            nonsingular_smul _ <| isUnit_addZ_of_X_ne hP.left hQ.left hx,
            nonsingular_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Projective F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Not (Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul  …
          ⊢ W.Nonsingular (W.add P Q)
        -/
      · rw [_root_.not_imp, not_ne_iff] at hxy
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Projective F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : And (Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))) (Eq (HMul.hMul  …
          ⊢ W.Nonsingular (W.add P Q)
        -/
        by_cases hy : P y * Q z = Q y * P z
        · simp only [add_of_Y_eq hP.left hPz hQz hxy.left hy hxy.right, nonsingular_smul _ <|
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
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    Q : WeierstrassCurve.Projective.PointClass F
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
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    Q✝ : WeierstrassCurve.Projective.PointClass F
    hP : W.Nonsingular P
    hPz : Eq (P 2) 0
    Q : Fin 3 → F
    hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) Q)
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
  -/
  by_cases hQz : Q z = 0
  · erw [addMap_eq, add_of_Z_eq_zero hP hQ hPz hQz,
      smul_eq _ <| (isUnit_Y_of_Z_eq_zero hP hPz).pow 4, Quotient.eq]
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P : Fin 3 → F
      Q✝ : WeierstrassCurve.Projective.PointClass F
      hP : W.Nonsingular P
      hPz : Eq (P 2) 0
      Q : Fin 3 → F
      hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) Q)
      hQz : Eq (Q 2) 0
      ⊢ (MulAction.orbitRel (Units F) (Fin 3 → F)) (Matrix.vecCons 0 (Matrix.vecCons …
    -/
    exact Setoid.symm <| equiv_zero_of_Z_eq_zero hQ hQz
    /-
      🎉 no goals
    -/
  · erw [addMap_eq, add_of_Z_eq_zero_left hP.left hPz hQz,
      smul_eq _ <| ((isUnit_Y_of_Z_eq_zero hP hPz).pow 2).mul <| Ne.isUnit hQz]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P : Fin 3 → F
      Q✝ : WeierstrassCurve.Projective.PointClass F
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
    W : WeierstrassCurve.Projective F
    P : WeierstrassCurve.Projective.PointClass F
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
    W : WeierstrassCurve.Projective F
    P✝ : WeierstrassCurve.Projective.PointClass F
    Q : Fin 3 → F
    hQ : W.Nonsingular Q
    hQz : Eq (Q 2) 0
    P : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
    ⊢ Eq (W.addMap (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P) (Quot …
  -/
  by_cases hPz : P z = 0
  · erw [addMap_eq, add_of_Z_eq_zero hP hQ hPz hQz,
      smul_eq _ <| (isUnit_Y_of_Z_eq_zero hP hPz).pow 4, Quotient.eq]
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P✝ : WeierstrassCurve.Projective.PointClass F
      Q : Fin 3 → F
      hQ : W.Nonsingular Q
      hQz : Eq (Q 2) 0
      P : Fin 3 → F
      hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
      hPz : Eq (P 2) 0
      ⊢ (MulAction.orbitRel (Units F) (Fin 3 → F)) (Matrix.vecCons 0 (Matrix.vecCons …
    -/
    exact Setoid.symm <| equiv_zero_of_Z_eq_zero hP hPz
    /-
      🎉 no goals
    -/
  · erw [addMap_eq, add_of_Z_eq_zero_right hQ.left hPz hQz,
      smul_eq _ (((isUnit_Y_of_Z_eq_zero hQ hQz).pow 2).mul <| Ne.isUnit hPz).neg]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P✝ : WeierstrassCurve.Projective.PointClass F
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
    (hQz : Q z ≠ 0) (hx : P x * Q z = Q x * P z) (hy' : P y * Q z = W.negY Q * P z) :
    W.addMap ⟦P⟧ ⟦Q⟧ = ⟦![0, 1, 0]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hx : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))
    hy' : Eq (HMul.hMul (P 1) (Q 2)) (HMul.hMul (W.negY Q) (P 2))
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
  -/
  by_cases hy : P y * Q z = Q y * P z
  · rw [addMap_eq, add_of_Y_eq hP.left hPz hQz hx hy hy',
      smul_eq _ <| isUnit_dblU_of_Y_eq hP hPz hQz hx hy hy']
  · rw [addMap_eq, add_of_Y_ne hP.left hQ hPz hQz hx hy,
      smul_eq _ <| isUnit_addU_of_Y_ne hPz hQz hy]


lemma addMap_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Equation P) (hQ : W.Equation Q) (hPz : P z ≠ 0)
    (hQz : Q z ≠ 0) (hxy : P x * Q z = Q x * P z → P y * Q z ≠ W.negY Q * P z) : W.addMap ⟦P⟧ ⟦Q⟧ =
      ⟦![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)), 1]⟧ := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Equation P
    hQ : W.Equation Q
    hPz : Ne (P 2) 0
    hQz : Ne (Q 2) 0
    hxy : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul (P 1) …
    ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
  -/
  by_cases hx : P x * Q z = Q x * P z
  · rw [addMap_eq, add_of_Y_ne' hP hQ hPz hQz hx <| hxy hx,
      smul_eq _ <| isUnit_dblZ_of_Y_ne' hP hQ hPz hQz hx <| hxy hx]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P Q : Fin 3 → F
      hP : W.Equation P
      hQ : W.Equation Q
      hPz : Ne (P 2) 0
      hQz : Ne (Q 2) 0
      hxy : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul (P 1) …
      hx : Not (Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)))
      ⊢ Eq (W.addMap (Quotient.mk (MulAction.orbitRel (Units F) (Fin 3 → F)) P) (Quo …
    -/
  · rw [addMap_eq, add_of_X_ne hP hQ hPz hQz hx, smul_eq _ <| isUnit_addZ_of_X_ne hP hQ hx]
    /-
      🎉 no goals
    -/


lemma nonsingularLift_addMap {P Q : PointClass F} (hP : W.NonsingularLift P)
    (hQ : W.NonsingularLift Q) : W.NonsingularLift <| W.addMap P Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : WeierstrassCurve.Projective.PointClass F
    hP : W.NonsingularLift P
    hQ : W.NonsingularLift Q
    ⊢ W.NonsingularLift (W.addMap P Q)
  -/
  rcases P; rcases Q
  /-
    case mk.mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : WeierstrassCurve.Projective.PointClass F
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


lemma zero_point [Nontrivial R] : (0 : W'.Point).point = ⟦![0, 1, 0]⟧ :=
  rfl


/-- The map from a nonsingular rational point on a Weierstrass curve `W'` in affine coordinates
to the corresponding nonsingular rational point on `W'` in projective coordinates. -/
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
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    X Y : R
    h : W'.toAffine.Nonsingular X Y
    h0 : Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Affine …
    ⊢ False
  -/
  obtain ⟨u, eq⟩ := Quotient.eq.mp <| (Point.ext_iff ..).mp h0
  /-
    case intro
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    X Y : R
    h : W'.toAffine.Nonsingular X Y
    h0 : Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Affine …
    u : Units R
    eq : Eq ((fun m => HSMul.hSMul m (Matrix.vecCons 0 (Matrix.vecCons 1 (Matrix.v …
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
/-- The map from a point representative that is nonsingular on a Weierstrass curve `W` in projective
coordinates to the corresponding nonsingular rational point on `W` in affine coordinates. -/
noncomputable def toAffine (P : Fin 3 → F) : W.toAffine.Point :=
  if hP : W.Nonsingular P ∧ P z ≠ 0 then .some <| (nonsingular_of_Z_ne_zero hP.2).mp hP.1 else 0


lemma toAffine_of_singular {P : Fin 3 → F} (hP : ¬W.Nonsingular P) : toAffine W P = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : Not (W.Nonsingular P)
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W P) 0
  -/
  rw [toAffine, dif_neg <| not_and_of_not_left _ hP]
  /-
    🎉 no goals
  -/


lemma toAffine_of_Z_eq_zero {P : Fin 3 → F} (hPz : P z = 0) : toAffine W P = 0 := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hPz : Eq (P 2) 0
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W P) 0
  -/
  rw [toAffine, dif_neg <| not_and_not_right.mpr fun _ => hPz]
  /-
    🎉 no goals
  -/


lemma toAffine_zero : toAffine W ![0, 1, 0] = 0 :=
  toAffine_of_Z_eq_zero rfl


lemma toAffine_of_Z_ne_zero {P : Fin 3 → F} (hP : W.Nonsingular P) (hPz : P z ≠ 0) :
    toAffine W P = .some ((nonsingular_of_Z_ne_zero hPz).mp hP) := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : W.Nonsingular P
    hPz : Ne (P 2) 0
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W P) (WeierstrassCurve.Affine …
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
    W : WeierstrassCurve.Projective F
    X Y : F
    h : W.Nonsingular (Matrix.vecCons X (Matrix.vecCons Y (Matrix.vecCons 1 Matrix …
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (Matrix.vecCons X (Matrix.v …
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
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    u : F
    hu : IsUnit u
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (HSMul.hSMul u P)) (Weierst …
  -/
  by_cases hP : W.Nonsingular P
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P : Fin 3 → F
      u : F
      hu : IsUnit u
      hP : W.Nonsingular P
      ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (HSMul.hSMul u P)) (Weierst …
    -/
  · by_cases hPz : P z = 0
      /-
        case pos
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Projective F
        P : Fin 3 → F
        u : F
        hu : IsUnit u
        hP : W.Nonsingular P
        hPz : Eq (P 2) 0
        ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (HSMul.hSMul u P)) (Weierst …
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
        W : WeierstrassCurve.Projective F
        P : Fin 3 → F
        u : F
        hu : IsUnit u
        hP : W.Nonsingular P
        hPz : Not (Eq (P 2) 0)
        ⊢ And (Eq (HDiv.hDiv (HSMul.hSMul u P 0) (HSMul.hSMul u P 2)) (HDiv.hDiv (P 0) …
      -/
      simp only [smul_fin3_ext, mul_div_mul_left _ _ hu.ne_zero, and_self]
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P : Fin 3 → F
      u : F
      hu : IsUnit u
      hP : Not (W.Nonsingular P)
      ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (HSMul.hSMul u P)) (Weierst …
    -/
  · rw [toAffine_of_singular <| hP.comp (nonsingular_smul P hu).mp, toAffine_of_singular hP]
    /-
      🎉 no goals
    -/


lemma toAffine_of_equiv {P Q : Fin 3 → F} (h : P ≈ Q) : toAffine W P = toAffine W Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    h : HasEquiv.Equiv P Q
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W P) (WeierstrassCurve.Projec …
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    Q : Fin 3 → F
    u : Units F
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W ((fun m => HSMul.hSMul m Q) …
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
    W : WeierstrassCurve.Projective F
    P : Fin 3 → F
    hP : W.Nonsingular P
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.neg P)) (Neg.neg (Weiers …
  -/
  by_cases hPz : P z = 0
  · rw [neg_of_Z_eq_zero hP.left hPz, toAffine_smul _ (isUnit_Y_of_Z_eq_zero hP hPz).neg,
      toAffine_zero, toAffine_of_Z_eq_zero hPz, Affine.Point.neg_zero]
  · rw [neg_of_Z_ne_zero hPz, toAffine_smul _ <| Ne.isUnit hPz, toAffine_some <|
        (nonsingular_smul _ <| Ne.isUnit hPz).mp <| neg_of_Z_ne_zero hPz ▸ nonsingular_neg hP,
      toAffine_of_Z_ne_zero hP hPz, Affine.Point.neg_some]


private lemma toAffine_add_of_Z_ne_zero {P Q : Fin 3 → F} (hP : W.Nonsingular P)
    (hQ : W.Nonsingular Q) (hPz : P z ≠ 0) (hQz : Q z ≠ 0)
    (hxy : P x * Q z = Q x * P z → P y * Q z ≠ W.negY Q * P z) : toAffine W
      ![W.toAffine.addX (P x / P z) (Q x / Q z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        W.toAffine.addY (P x / P z) (Q x / Q z) (P y / P z)
          (W.toAffine.slope (P x / P z) (Q x / Q z) (P y / P z) (Q y / Q z)),
        1] = toAffine W P + toAffine W Q := by
  rw [toAffine_some <| nonsingular_add_of_Z_ne_zero hP hQ hPz hQz hxy, toAffine_of_Z_ne_zero hP hPz,
    toAffine_of_Z_ne_zero hQ hQz,
    Affine.Point.add_of_imp <| by rwa [← X_eq_iff hPz hQz, ne_eq, ← Y_eq_iff' hPz hQz]]


lemma toAffine_add {P Q : Fin 3 → F} (hP : W.Nonsingular P) (hQ : W.Nonsingular Q) :
    toAffine W (W.add P Q) = toAffine W P + toAffine W Q := by
  /-
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (HAdd.hAdd (We …
  -/
  by_cases hPz : P z = 0
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Eq (P 2) 0
      ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (HAdd.hAdd (We …
    -/
  · rw [toAffine_of_Z_eq_zero hPz, zero_add]
    /-
      case pos
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Eq (P 2) 0
      ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (WeierstrassCu …
    -/
    by_cases hQz : Q z = 0
    · rw [add_of_Z_eq_zero hP hQ hPz hQz, toAffine_smul _ <| (isUnit_Y_of_Z_eq_zero hP hPz).pow 4,
        toAffine_zero, toAffine_of_Z_eq_zero hQz]
    · rw [add_of_Z_eq_zero_left hP.left hPz hQz,
        toAffine_smul _ <| ((isUnit_Y_of_Z_eq_zero hP hPz).pow 2).mul <| Ne.isUnit hQz]
    /-
      case neg
      F : Type v
      inst✝ : Field F
      W : WeierstrassCurve.Projective F
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      hPz : Not (Eq (P 2) 0)
      ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (HAdd.hAdd (We …
    -/
  · by_cases hQz : Q z = 0
    · rw [add_of_Z_eq_zero_right hQ.left hPz hQz,
        toAffine_smul _ (((isUnit_Y_of_Z_eq_zero hQ hQz).pow 2).mul <| Ne.isUnit hPz).neg,
        toAffine_of_Z_eq_zero hQz, add_zero]
      /-
        case neg
        F : Type v
        inst✝ : Field F
        W : WeierstrassCurve.Projective F
        P Q : Fin 3 → F
        hP : W.Nonsingular P
        hQ : W.Nonsingular Q
        hPz : Not (Eq (P 2) 0)
        hQz : Not (Eq (Q 2) 0)
        ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (HAdd.hAdd (We …
      -/
    · by_cases hxy : P x * Q z = Q x * P z → P y * Q z ≠ W.negY Q * P z
        /-
          case pos
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Projective F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul (P 1) …
          ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (HAdd.hAdd (We …
        -/
      · by_cases hx : P x * Q z = Q x * P z
        · rw [add_of_Y_ne' hP.left hQ.left hPz hQz hx <| hxy hx,
            toAffine_smul _ <| isUnit_dblZ_of_Y_ne' hP.left hQ.left hPz hQz hx <| hxy hx,
            toAffine_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        · rw [add_of_X_ne hP.left hQ.left hPz hQz hx, toAffine_smul _ <|
              isUnit_addZ_of_X_ne hP.left hQ.left hx, toAffine_add_of_Z_ne_zero hP hQ hPz hQz hxy]
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Projective F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : Not (Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2)) → Ne (HMul.hMul  …
          ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) (HAdd.hAdd (We …
        -/
      · rw [_root_.not_imp, not_ne_iff] at hxy
        rw [toAffine_of_Z_ne_zero hP hPz, toAffine_of_Z_ne_zero hQ hQz, Affine.Point.add_of_Y_eq
            ((X_eq_iff hPz hQz).mp hxy.left) ((Y_eq_iff' hPz hQz).mp hxy.right)]
        /-
          case neg
          F : Type v
          inst✝ : Field F
          W : WeierstrassCurve.Projective F
          P Q : Fin 3 → F
          hP : W.Nonsingular P
          hQ : W.Nonsingular Q
          hPz : Not (Eq (P 2) 0)
          hQz : Not (Eq (Q 2) 0)
          hxy : And (Eq (HMul.hMul (P 0) (Q 2)) (HMul.hMul (Q 0) (P 2))) (Eq (HMul.hMul  …
          ⊢ Eq (WeierstrassCurve.Projective.Point.toAffine W (W.add P Q)) 0
        -/
        by_cases hy : P y * Q z = Q y * P z
        · rw [add_of_Y_eq hP.left hPz hQz hxy.left hy hxy.right,
            toAffine_smul _ <| isUnit_dblU_of_Y_eq hP hPz hQz hxy.left hy hxy.right, toAffine_zero]
        · rw [add_of_Y_ne hP.left hQ.left hPz hQz hxy.left hy,
            toAffine_smul _ <| isUnit_addU_of_Y_ne hPz hQz hy, toAffine_zero]


/-- The map from a nonsingular rational point on a Weierstrass curve `W` in projective coordinates
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
    W : WeierstrassCurve.Projective F
    P : W.Point
    ⊢ Eq (Neg.neg P).toAffineLift (Neg.neg P.toAffineLift)
  -/
  rcases P with @⟨⟨_⟩, hP⟩
  /-
    case mk.mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    point✝ : WeierstrassCurve.Projective.PointClass F
    a✝ : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    ⊢ Eq (Neg.neg (WeierstrassCurve.Projective.Point.mk hP)).toAffineLift (Neg.neg …
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
    W : WeierstrassCurve.Projective F
    P Q : W.Point
    ⊢ Eq (HAdd.hAdd P Q).toAffineLift (HAdd.hAdd P.toAffineLift Q.toAffineLift)
  -/
  rcases P, Q with ⟨@⟨⟨_⟩, hP⟩, @⟨⟨_⟩, hQ⟩⟩
  /-
    case mk.mk.mk.mk
    F : Type v
    inst✝ : Field F
    W : WeierstrassCurve.Projective F
    point✝¹ : WeierstrassCurve.Projective.PointClass F
    a✝¹ : Fin 3 → F
    hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    point✝ : WeierstrassCurve.Projective.PointClass F
    a✝ : Fin 3 → F
    hQ : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F)))  …
    ⊢ Eq (HAdd.hAdd (WeierstrassCurve.Projective.Point.mk hP) (WeierstrassCurve.Pr …
  -/
  exact toAffine_add hP hQ
  /-
    🎉 no goals
  -/


variable (W) in
/-- The equivalence between the nonsingular rational points on a Weierstrass curve `W` in Projective
coordinates with the nonsingular rational points on `W` in affine coordinates. -/
@[simps]
noncomputable def toAffineAddEquiv : W.Point ≃+ W.toAffine.Point where
  toFun := toAffineLift
  invFun := fromAffine
  left_inv := by
    /-
      R : Type u
      W' : WeierstrassCurve.Projective R
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      inst✝ : CommRing R
      ⊢ Function.LeftInverse WeierstrassCurve.Projective.Point.fromAffine Weierstras …
    -/
    rintro @⟨⟨P⟩, hP⟩
    /-
      case mk.mk
      R : Type u
      W' : WeierstrassCurve.Projective R
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      inst✝ : CommRing R
      point✝ : WeierstrassCurve.Projective.PointClass F
      P : Fin 3 → F
      hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
      ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Projectiv …
    -/
    by_cases hPz : P z = 0
      /-
        case pos
        R : Type u
        W' : WeierstrassCurve.Projective R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Projective.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Eq (P 2) 0
        ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Projectiv …
      -/
    · rw [Point.ext_iff, toAffineLift_eq, toAffine_of_Z_eq_zero hPz]
      /-
        case pos
        R : Type u
        W' : WeierstrassCurve.Projective R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Projective.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Eq (P 2) 0
        ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine 0).point (WeierstrassCurve. …
      -/
      exact Quotient.eq.mpr <| Setoid.symm <| equiv_zero_of_Z_eq_zero hP hPz
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        W' : WeierstrassCurve.Projective R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Projective.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Not (Eq (P 2) 0)
        ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Projectiv …
      -/
    · rw [Point.ext_iff, toAffineLift_eq, toAffine_of_Z_ne_zero hP hPz]
      /-
        case neg
        R : Type u
        W' : WeierstrassCurve.Projective R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        inst✝ : CommRing R
        point✝ : WeierstrassCurve.Projective.PointClass F
        P : Fin 3 → F
        hP : W.NonsingularLift (Quot.mk (⇑(MulAction.orbitRel (Units F) (Fin 3 → F))) P)
        hPz : Not (Eq (P 2) 0)
        ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Affine.Po …
      -/
      exact Quotient.eq.mpr <| Setoid.symm <| equiv_some_of_Z_ne_zero hPz
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      R : Type u
      W' : WeierstrassCurve.Projective R
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      inst✝ : CommRing R
      ⊢ Function.RightInverse WeierstrassCurve.Projective.Point.fromAffine Weierstra …
    -/
    rintro (_ | _)
      /-
        case zero
        R : Type u
        W' : WeierstrassCurve.Projective R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        inst✝ : CommRing R
        ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine WeierstrassCurve.Affine.Poi …
      -/
    · erw [fromAffine_zero, toAffineLift_zero, Affine.Point.zero_def]
      /-
        🎉 no goals
      -/
      /-
        case some
        R : Type u
        W' : WeierstrassCurve.Projective R
        F : Type v
        inst✝¹ : Field F
        W : WeierstrassCurve.Projective F
        inst✝ : CommRing R
        x✝ y✝ : F
        h✝ : W.toAffine.Nonsingular x✝ y✝
        ⊢ Eq (WeierstrassCurve.Projective.Point.fromAffine (WeierstrassCurve.Affine.Po …
      -/
    · rw [fromAffine_some, toAffineLift_some]
      /-
        🎉 no goals
      -/
  map_add' := toAffineLift_add


@[simp]
lemma map_polynomial : (W'.map f).toProjective.polynomial = MvPolynomial.map f W'.polynomial := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toProjective.polynomial ((MvPolynomial.map f) …
  -/
  simp only [polynomial]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (MvPolynomial.X 1) …
  -/
  map_simp
  /-
    🎉 no goals
  -/


lemma Equation.map {P : Fin 3 → R} (h : W'.Equation P) :
    (W'.map f).toProjective.Equation (f ∘ P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    h : W'.Equation P
    ⊢ (WeierstrassCurve.map W' f).toProjective.Equation (Function.comp (⇑f) P)
  -/
  rw [Equation, map_polynomial, eval_map, ← eval₂_comp, ← map_zero f]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    h : W'.Equation P
    ⊢ Eq (f ((MvPolynomial.eval P) W'.polynomial)) (f 0)
  -/
  exact congr_arg f h
  /-
    🎉 no goals
  -/


variable {f} in
@[simp]
lemma map_equation (hf : Function.Injective f) (P : Fin 3 → R) :
    (W'.map f).toProjective.Equation (f ∘ P) ↔ W'.Equation P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Injective ⇑f
    P : Fin 3 → R
    ⊢ Iff ((WeierstrassCurve.map W' f).toProjective.Equation (Function.comp (⇑f) P …
  -/
  simp only [Equation, map_polynomial, eval_map, ← eval₂_comp, map_eq_zero_iff f hf]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_polynomialX :
    (W'.map f).toProjective.polynomialX = MvPolynomial.map f W'.polynomialX := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toProjective.polynomialX ((MvPolynomial.map f …
  -/
  simp only [polynomialX, map_polynomial, pderiv_map]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_polynomialY :
    (W'.map f).toProjective.polynomialY = MvPolynomial.map f W'.polynomialY := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toProjective.polynomialY ((MvPolynomial.map f …
  -/
  simp only [polynomialY, map_polynomial, pderiv_map]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_polynomialZ :
    (W'.map f).toProjective.polynomialZ = MvPolynomial.map f W'.polynomialZ := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (WeierstrassCurve.map W' f).toProjective.polynomialZ ((MvPolynomial.map f …
  -/
  simp only [polynomialZ, map_polynomial, pderiv_map]
  /-
    🎉 no goals
  -/


variable {f} in
@[simp]
lemma map_nonsingular (hf : Function.Injective f) (P : Fin 3 → R) :
    (W'.map f).toProjective.Nonsingular (f ∘ P) ↔ W'.Nonsingular P := by
  simp only [Nonsingular, map_equation hf, map_polynomialX, map_polynomialY, map_polynomialZ,
    eval_map, ← eval₂_comp, map_ne_zero_iff f hf]


@[simp]
lemma map_negY (P : Fin 3 → R) : (W'.map f).toProjective.negY (f ∘ P) = f (W'.negY P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.negY (Function.comp (⇑f) P)) (f …
  -/
  simp only [negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (Function.comp (⇑f) P 1)) (HMul.hMul (Weie …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
protected lemma map_neg (P : Fin 3 → R) : (W'.map f).toProjective.neg (f ∘ P) = f ∘ W'.neg P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.neg (Function.comp (⇑f) P)) (Fu …
  -/
  simp only [neg, map_negY, comp_fin3]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (Matrix.vecCons (Function.comp (⇑f) P 0) (Matrix.vecCons (f (W'.negY P))  …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_dblU {K : Type v} [Field K] (f : F →+* K) (P : Fin 3 → F) :
    (W.map f).toProjective.dblU (f ∘ P) = f (W.dblU P) := by
  /-
    F : Type v
    inst✝¹ : Field F
    W : WeierstrassCurve.Projective F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    P : Fin 3 → F
    ⊢ Eq ((WeierstrassCurve.map W f).toProjective.dblU (Function.comp (⇑f) P)) (f  …
  -/
  simp only [dblU_eq]
  /-
    F : Type v
    inst✝¹ : Field F
    W : WeierstrassCurve.Projective F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    P : Fin 3 → F
    ⊢ Eq (HDiv.hDiv (HPow.hPow (HSub.hSub (HMul.hMul (HMul.hMul (WeierstrassCurve. …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_dblZ (P : Fin 3 → R) : (W'.map f).toProjective.dblZ (f ∘ P) = f (W'.dblZ P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.dblZ (Function.comp (⇑f) P)) (f …
  -/
  simp only [dblZ, negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (HMul.hMul (Function.comp (⇑f) P 2) (HPow.hPow (HSub.hSub (Function.comp  …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_dblX (P : Fin 3 → R) : (W'.map f).toProjective.dblX (f ∘ P) = f (W'.dblX P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.dblX (Function.comp (⇑f) P)) (f …
  -/
  simp only [dblX, map_dblU, map_negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_negDblY (P : Fin 3 → R) : (W'.map f).toProjective.negDblY (f ∘ P) = f (W'.negDblY P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.negDblY (Function.comp (⇑f) P)) …
  -/
  simp only [negDblY, map_dblU, map_dblX, map_negY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSub.h …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_dblY (P : Fin 3 → R) : (W'.map f).toProjective.dblY (f ∘ P) = f (W'.dblY P) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.dblY (Function.comp (⇑f) P)) (f …
  -/
  simp only [dblY, negY_eq, map_negDblY, map_dblX, map_dblZ]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (f (W'.negDblY P))) (HMul.hMul (Weierstras …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_dblXYZ (P : Fin 3 → R) : (W'.map f).toProjective.dblXYZ (f ∘ P) = f ∘ dblXYZ W' P := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.dblXYZ (Function.comp (⇑f) P))  …
  -/
  simp only [dblXYZ, map_dblX, map_dblY, map_dblZ, comp_fin3]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_addU {K : Type v} [Field K] (f : F →+* K) (P Q : Fin 3 → F) :
    addU (f ∘ P) (f ∘ Q) = f (addU P Q) := by
  /-
    F : Type v
    inst✝¹ : Field F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    P Q : Fin 3 → F
    ⊢ Eq (WeierstrassCurve.Projective.addU (Function.comp (⇑f) P) (Function.comp ( …
  -/
  simp only [addU]
  /-
    F : Type v
    inst✝¹ : Field F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    P Q : Fin 3 → F
    ⊢ Eq (HDiv.hDiv (Neg.neg (HPow.hPow (HSub.hSub (HMul.hMul (Function.comp (⇑f)  …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_addZ (P Q : Fin 3 → R) :
    (W'.map f).toProjective.addZ (f ∘ P) (f ∘ Q) = f (W'.addZ P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.addZ (Function.comp (⇑f) P) (Fu …
  -/
  simp only [addZ]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.h …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_addX (P Q : Fin 3 → R) :
    (W'.map f).toProjective.addX (f ∘ P) (f ∘ Q) = f (W'.addX P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.addX (Function.comp (⇑f) P) (Fu …
  -/
  simp only [addX]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.h …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_negAddY (P Q : Fin 3 → R) :
    (W'.map f).toProjective.negAddY (f ∘ P) (f ∘ Q) = f (W'.negAddY P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.negAddY (Function.comp (⇑f) P)  …
  -/
  simp only [negAddY]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_addY (P Q : Fin 3 → R) :
    (W'.map f).toProjective.addY (f ∘ P) (f ∘ Q) = f (W'.addY P Q) := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.addY (Function.comp (⇑f) P) (Fu …
  -/
  simp only [addY, negY_eq, map_negAddY, map_addX, map_addZ]
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq (HSub.hSub (HSub.hSub (Neg.neg (f (W'.negAddY P Q))) (HMul.hMul (Weierstr …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_addXYZ (P Q : Fin 3 → R) :
    (W'.map f).toProjective.addXYZ (f ∘ P) (f ∘ Q) = f ∘ addXYZ W' P Q := by
  /-
    R : Type u
    W' : WeierstrassCurve.Projective R
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    P Q : Fin 3 → R
    ⊢ Eq ((WeierstrassCurve.map W' f).toProjective.addXYZ (Function.comp (⇑f) P) ( …
  -/
  simp only [addXYZ, map_addX, map_addY, map_addZ, comp_fin3]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma map_add {K : Type v} [Field K] (f : F →+* K) {P Q : Fin 3 → F}
    (hP : W.Nonsingular P) (hQ : W.Nonsingular Q) :
    (W.map f).toProjective.add (f ∘ P) (f ∘ Q) = f ∘ W.add P Q := by
  /-
    F : Type v
    inst✝¹ : Field F
    W : WeierstrassCurve.Projective F
    K : Type v
    inst✝ : Field K
    f : RingHom F K
    P Q : Fin 3 → F
    hP : W.Nonsingular P
    hQ : W.Nonsingular Q
    ⊢ Eq ((WeierstrassCurve.map W f).toProjective.add (Function.comp (⇑f) P) (Func …
  -/
  by_cases h : P ≈ Q
    /-
      case pos
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      h : HasEquiv.Equiv P Q
      ⊢ Eq ((WeierstrassCurve.map W f).toProjective.add (Function.comp (⇑f) P) (Func …
    -/
  · rw [add_of_equiv <| (comp_equiv_comp f hP hQ).mpr h, add_of_equiv h, map_dblXYZ]
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type v
      inst✝¹ : Field F
      W : WeierstrassCurve.Projective F
      K : Type v
      inst✝ : Field K
      f : RingHom F K
      P Q : Fin 3 → F
      hP : W.Nonsingular P
      hQ : W.Nonsingular Q
      h : Not (HasEquiv.Equiv P Q)
      ⊢ Eq ((WeierstrassCurve.map W f).toProjective.add (Function.comp (⇑f) P) (Func …
    -/
  · rw [add_of_not_equiv <| h.comp (comp_equiv_comp f hP hQ).mp, add_of_not_equiv h, map_addXYZ]
    /-
      🎉 no goals
    -/


lemma baseChange_polynomial : (W'.baseChange B).toProjective.polynomial =
    MvPolynomial.map f (W'.baseChange A).toProjective.polynomial := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    ⊢ Eq (WeierstrassCurve.baseChange W' B).toProjective.polynomial ((MvPolynomial …
  -/
  rw [← map_polynomial, map_baseChange]
  /-
    🎉 no goals
  -/


variable {f} in
lemma baseChange_equation (hf : Function.Injective f) (P : Fin 3 → A) :
    (W'.baseChange B).toProjective.Equation (f ∘ P) ↔
      (W'.baseChange A).toProjective.Equation P := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Iff ((WeierstrassCurve.baseChange W' B).toProjective.Equation (Function.comp …
  -/
  rw [← RingHom.coe_coe, ← map_equation hf, AlgHom.toRingHom_eq_coe, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_polynomialX : (W'.baseChange B).toProjective.polynomialX =
    MvPolynomial.map f (W'.baseChange A).toProjective.polynomialX := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    ⊢ Eq (WeierstrassCurve.baseChange W' B).toProjective.polynomialX ((MvPolynomia …
  -/
  rw [← map_polynomialX, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_polynomialY : (W'.baseChange B).toProjective.polynomialY =
    MvPolynomial.map f (W'.baseChange A).toProjective.polynomialY := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    ⊢ Eq (WeierstrassCurve.baseChange W' B).toProjective.polynomialY ((MvPolynomia …
  -/
  rw [← map_polynomialY, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_polynomialZ : (W'.baseChange B).toProjective.polynomialZ =
    MvPolynomial.map f (W'.baseChange A).toProjective.polynomialZ := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    ⊢ Eq (WeierstrassCurve.baseChange W' B).toProjective.polynomialZ ((MvPolynomia …
  -/
  rw [← map_polynomialZ, map_baseChange]
  /-
    🎉 no goals
  -/


variable {f} in
lemma baseChange_nonsingular (hf : Function.Injective f) (P : Fin 3 → A) :
    (W'.baseChange B).toProjective.Nonsingular (f ∘ P) ↔
      (W'.baseChange A).toProjective.Nonsingular P := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Iff ((WeierstrassCurve.baseChange W' B).toProjective.Nonsingular (Function.c …
  -/
  rw [← RingHom.coe_coe, ← map_nonsingular hf, AlgHom.toRingHom_eq_coe, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_negY (P : Fin 3 → A) :
    (W'.baseChange B).toProjective.negY (f ∘ P) = f ((W'.baseChange A).toProjective.negY P) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.negY (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_negY, map_baseChange]
  /-
    🎉 no goals
  -/


protected lemma baseChange_neg (P : Fin 3 → A) :
    (W'.baseChange B).toProjective.neg (f ∘ P) = f ∘ (W'.baseChange A).toProjective.neg P := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.neg (Function.comp (⇑f)  …
  -/
  rw [← RingHom.coe_coe, ← WeierstrassCurve.Projective.map_neg, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_dblZ (P : Fin 3 → A) : (W'.baseChange B).toProjective.dblZ (f ∘ P) =
    f ((W'.baseChange A).toProjective.dblZ P) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.dblZ (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_dblZ, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_dblX (P : Fin 3 → A) : (W'.baseChange B).toProjective.dblX (f ∘ P) =
    f ((W'.baseChange A).toProjective.dblX P) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.dblX (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_dblX, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_negDblY (P : Fin 3 → A) : (W'.baseChange B).toProjective.negDblY (f ∘ P) =
    f ((W'.baseChange A).toProjective.negDblY P) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.negDblY (Function.comp ( …
  -/
  rw [← RingHom.coe_coe, ← map_negDblY, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_dblY (P : Fin 3 → A) : (W'.baseChange B).toProjective.dblY (f ∘ P) =
    f ((W'.baseChange A).toProjective.dblY P) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.dblY (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_dblY, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_dblXYZ (P : Fin 3 → A) : (W'.baseChange B).toProjective.dblXYZ (f ∘ P) =
    f ∘ (W'.baseChange A).toProjective.dblXYZ P := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.dblXYZ (Function.comp (⇑ …
  -/
  rw [← RingHom.coe_coe, ← map_dblXYZ, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_addX (P Q : Fin 3 → A) : (W'.baseChange B).toProjective.addX (f ∘ P) (f ∘ Q) =
    f ((W'.baseChange A).toProjective.addX P Q) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P Q : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.addX (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_addX, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_negAddY (P Q : Fin 3 → A) :
    (W'.baseChange B).toProjective.negAddY (f ∘ P) (f ∘ Q) =
      f ((W'.baseChange A).toProjective.negAddY P Q) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P Q : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.negAddY (Function.comp ( …
  -/
  rw [← RingHom.coe_coe, ← map_negAddY, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_addY (P Q : Fin 3 → A) : (W'.baseChange B).toProjective.addY (f ∘ P) (f ∘ Q) =
    f ((W'.baseChange A).toProjective.addY P Q) := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P Q : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.addY (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_addY, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_addXYZ (P Q : Fin 3 → A) : (W'.baseChange B).toProjective.addXYZ (f ∘ P) (f ∘ Q) =
    f ∘ (W'.baseChange A).toProjective.addXYZ P Q := by
  /-
    R : Type r
    inst✝¹⁰ : CommRing R
    W' : WeierstrassCurve.Projective R
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
    P Q : Fin 3 → A
    ⊢ Eq ((WeierstrassCurve.baseChange W' B).toProjective.addXYZ (Function.comp (⇑ …
  -/
  rw [← RingHom.coe_coe, ← map_addXYZ, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_dblU (P : Fin 3 → F) : (W'.baseChange K).toProjective.dblU (f ∘ P) =
    f ((W'.baseChange F).toProjective.dblU P) := by
  /-
    F : Type v
    inst✝¹⁰ : Field F
    R : Type r
    inst✝⁹ : CommRing R
    W' : WeierstrassCurve.Projective R
    S : Type s
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra R F
    inst✝⁵ : Algebra S F
    inst✝⁴ : IsScalarTower R S F
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : Algebra S K
    inst✝ : IsScalarTower R S K
    f : AlgHom S F K
    P : Fin 3 → F
    ⊢ Eq ((WeierstrassCurve.baseChange W' K).toProjective.dblU (Function.comp (⇑f) …
  -/
  rw [← RingHom.coe_coe, ← map_dblU, map_baseChange]
  /-
    🎉 no goals
  -/


lemma baseChange_add {P Q : Fin 3 → F} (hP : (W'.baseChange F).toProjective.Nonsingular P)
    (hQ : (W'.baseChange F).toProjective.Nonsingular Q) :
    (W'.baseChange K).toProjective.add (f ∘ P) (f ∘ Q) =
      f ∘ (W'.baseChange F).toProjective.add P Q := by
  /-
    F : Type v
    inst✝¹⁰ : Field F
    R : Type r
    inst✝⁹ : CommRing R
    W' : WeierstrassCurve.Projective R
    S : Type s
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra R F
    inst✝⁵ : Algebra S F
    inst✝⁴ : IsScalarTower R S F
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : Algebra S K
    inst✝ : IsScalarTower R S K
    f : AlgHom S F K
    P Q : Fin 3 → F
    hP : (WeierstrassCurve.baseChange W' F).toProjective.Nonsingular P
    hQ : (WeierstrassCurve.baseChange W' F).toProjective.Nonsingular Q
    ⊢ Eq ((WeierstrassCurve.baseChange W' K).toProjective.add (Function.comp (⇑f)  …
  -/
  rw [← RingHom.coe_coe, ← WeierstrassCurve.Projective.map_add f hP hQ (K := K), map_baseChange]
  /-
    🎉 no goals
  -/


/-- An abbreviation for `WeierstrassCurve.Projective.Point.fromAffine` for dot notation. -/
abbrev WeierstrassCurve.Affine.Point.toProjective {R : Type u} [CommRing R] [Nontrivial R]
    {W : Affine R} (P : W.Point) : W.toProjective.Point :=
  Projective.Point.fromAffine P


