local macro "map_simp" : tactic =>
  `(tactic| simp only [map_ofNat, map_neg, map_add, map_sub, map_mul, map_pow])


/-- An admissible linear change of variables of Weierstrass curves defined over a ring `R` given by
a tuple $(u, r, s, t)$ for some $u \in R^\times$ and some $r, s, t \in R$. As a matrix, it is
$\begin{pmatrix} u^2 & 0 & r \cr u^2s & u^3 & t \cr 0 & 0 & 1 \end{pmatrix}$.
In other words, this is the change of variables $(X, Y) \mapsto (u^2X + r, u^3Y + u^2sX + t)$.
When `R` is a field, any two isomorphic Weierstrass equations are related by this. -/
@[ext]
structure VariableChange (R : Type u) [CommRing R] where
  /-- The `u` coefficient of an admissible linear change of variables, which must be a unit. -/
  u : Rˣ
  /-- The `r` coefficient of an admissible linear change of variables. -/
  r : R
  /-- The `s` coefficient of an admissible linear change of variables. -/
  s : R
  /-- The `t` coefficient of an admissible linear change of variables. -/
  t : R


/-- The identity linear change of variables given by the identity matrix. -/
def id : VariableChange R :=
  ⟨1, 0, 0, 0⟩


/-- The composition of two linear changes of variables given by matrix multiplication. -/
def comp : VariableChange R where
  u := C.u * C'.u
  r := C.r * C'.u ^ 2 + C'.r
  s := C'.u * C.s + C'.s
  t := C.t * C'.u ^ 3 + C.r * C'.s * C'.u ^ 2 + C'.t


/-- The inverse of a linear change of variables given by matrix inversion. -/
def inv : VariableChange R where
  u := C.u⁻¹
  r := -C.r * C.u⁻¹ ^ 2
  s := -C.s * C.u⁻¹
  t := (C.r * C.s - C.t) * C.u⁻¹ ^ 3


lemma id_comp (C : VariableChange R) : comp id C = C := by
  /-
    R : Type u
    inst✝ : CommRing R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (WeierstrassCurve.VariableChange.id.comp C) C
  -/
  simp only [comp, id, zero_add, zero_mul, mul_zero, one_mul]
  /-
    🎉 no goals
  -/


lemma comp_id (C : VariableChange R) : comp C id = C := by
  /-
    R : Type u
    inst✝ : CommRing R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (C.comp WeierstrassCurve.VariableChange.id) C
  -/
  simp only [comp, id, add_zero, mul_zero, one_mul, mul_one, one_pow, Units.val_one]
  /-
    🎉 no goals
  -/


lemma comp_left_inv (C : VariableChange R) : comp (inv C) C = id := by
  /-
    R : Type u
    inst✝ : CommRing R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (C.inv.comp C) WeierstrassCurve.VariableChange.id
  -/
  rw [comp, id, inv]
  /-
    R : Type u
    inst✝ : CommRing R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq { u := HMul.hMul { u := Inv.inv C.u, r := HMul.hMul (Neg.neg C.r) (HPow.h …
  -/
  ext <;> dsimp only
    /-
      case u.a
      R : Type u
      inst✝ : CommRing R
      C : WeierstrassCurve.VariableChange R
      ⊢ Eq ↑(HMul.hMul (Inv.inv C.u) C.u) ↑1
    -/
  · exact C.u.inv_mul
    /-
      🎉 no goals
    -/
    /-
      case r
      R : Type u
      inst✝ : CommRing R
      C : WeierstrassCurve.VariableChange R
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (Neg.neg C.r) (HPow.hPow (↑(Inv.inv C.u) …
    -/
  · linear_combination -C.r * pow_mul_pow_eq_one 2 C.u.inv_mul
    /-
      🎉 no goals
    -/
    /-
      case s
      R : Type u
      inst✝ : CommRing R
      C : WeierstrassCurve.VariableChange R
      ⊢ Eq (HAdd.hAdd (HMul.hMul (↑C.u) (HMul.hMul (Neg.neg C.s) ↑(Inv.inv C.u))) C. …
    -/
  · linear_combination -C.s * C.u.inv_mul
    /-
      🎉 no goals
    -/
  · linear_combination (C.r * C.s - C.t) * pow_mul_pow_eq_one 3 C.u.inv_mul
      + -C.r * C.s * pow_mul_pow_eq_one 2 C.u.inv_mul


lemma comp_assoc (C C' C'' : VariableChange R) : comp (comp C C') C'' = comp C (comp C' C'') := by
  /-
    R : Type u
    inst✝ : CommRing R
    C C' C'' : WeierstrassCurve.VariableChange R
    ⊢ Eq ((C.comp C').comp C'') (C.comp (C'.comp C''))
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
  ext <;> simp only [comp, Units.val_mul] <;> ring1
                                              /-
                                                🎉 no goals
                                              -/


instance instGroup : Group (VariableChange R) where
  one := id
  inv := inv
  mul := comp
  one_mul := id_comp
  mul_one := comp_id
  inv_mul_cancel := comp_left_inv
  mul_assoc := comp_assoc


/-- The Weierstrass curve over `R` induced by an admissible linear change of variables
$(X, Y) \mapsto (u^2X + r, u^3Y + u^2sX + t)$ for some $u \in R^\times$ and some $r, s, t \in R$. -/
@[simps]
def variableChange : WeierstrassCurve R where
  a₁ := C.u⁻¹ * (W.a₁ + 2 * C.s)
  a₂ := C.u⁻¹ ^ 2 * (W.a₂ - C.s * W.a₁ + 3 * C.r - C.s ^ 2)
  a₃ := C.u⁻¹ ^ 3 * (W.a₃ + C.r * W.a₁ + 2 * C.t)
  a₄ := C.u⁻¹ ^ 4 * (W.a₄ - C.s * W.a₃ + 2 * C.r * W.a₂ - (C.t + C.r * C.s) * W.a₁ + 3 * C.r ^ 2
    - 2 * C.s * C.t)
  a₆ := C.u⁻¹ ^ 6 * (W.a₆ + C.r * W.a₄ + C.r ^ 2 * W.a₂ + C.r ^ 3 - C.t * W.a₃ - C.t ^ 2
    - C.r * C.t * W.a₁)


lemma variableChange_id : W.variableChange VariableChange.id = W := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (W.variableChange WeierstrassCurve.VariableChange.id) W
  -/
  rw [VariableChange.id, variableChange, inv_one, Units.val_one]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq { a₁ := HMul.hMul 1 (HAdd.hAdd W.a₁ (HMul.hMul 2 { u := 1, r := 0, s := 0 …
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
                       /-
                         🎉 no goals
                       -/
  ext <;> (dsimp only; ring1)
                       /-
                         🎉 no goals
                       -/


lemma variableChange_comp (C C' : VariableChange R) (W : WeierstrassCurve R) :
    W.variableChange (C.comp C') = (W.variableChange C').variableChange C := by
  /-
    R : Type u
    inst✝ : CommRing R
    C C' : WeierstrassCurve.VariableChange R
    W : WeierstrassCurve R
    ⊢ Eq (W.variableChange (C.comp C')) ((W.variableChange C').variableChange C)
  -/
  simp only [VariableChange.comp, variableChange]
  /-
    R : Type u
    inst✝ : CommRing R
    C C' : WeierstrassCurve.VariableChange R
    W : WeierstrassCurve R
    ⊢ Eq { a₁ := HMul.hMul (↑(Inv.inv (HMul.hMul C.u C'.u))) (HAdd.hAdd W.a₁ (HMul …
  -/
  ext <;> simp only [mul_inv, Units.val_mul]
    /-
      case a₁
      R : Type u
      inst✝ : CommRing R
      C C' : WeierstrassCurve.VariableChange R
      W : WeierstrassCurve R
      ⊢ Eq (HMul.hMul (HMul.hMul ↑(Inv.inv C.u) ↑(Inv.inv C'.u)) (HAdd.hAdd W.a₁ (HM …
    -/
  · linear_combination ↑C.u⁻¹ * C.s * 2 * C'.u.inv_mul
    /-
      🎉 no goals
    -/
  · linear_combination
      C.s * (-C'.s * 2 - W.a₁) * C.u⁻¹ ^ 2 * ↑C'.u⁻¹ * C'.u.inv_mul
        + (C.r * 3 - C.s ^ 2) * C.u⁻¹ ^ 2 * pow_mul_pow_eq_one 2 C'.u.inv_mul
  · linear_combination
      C.r * (C'.s * 2 + W.a₁) * C.u⁻¹ ^ 3 * ↑C'.u⁻¹ * pow_mul_pow_eq_one 2 C'.u.inv_mul
        + C.t * 2 * C.u⁻¹ ^ 3 * pow_mul_pow_eq_one 3 C'.u.inv_mul
  · linear_combination
      C.s * (-W.a₃ - C'.r * W.a₁ - C'.t * 2) * C.u⁻¹ ^ 4 * C'.u⁻¹ ^ 3 * C'.u.inv_mul
        + C.u⁻¹ ^ 4 * C'.u⁻¹ ^ 2 * (C.r * C'.r * 6 + C.r * W.a₂ * 2 - C'.s * C.r * W.a₁ * 2
          - C'.s ^ 2 * C.r * 2) * pow_mul_pow_eq_one 2 C'.u.inv_mul
        - C.u⁻¹ ^ 4 * ↑C'.u⁻¹ * (C.s * C'.s * C.r * 2 + C.s * C.r * W.a₁ + C'.s * C.t * 2
          + C.t * W.a₁) * pow_mul_pow_eq_one 3 C'.u.inv_mul
        + C.u⁻¹ ^ 4 * (C.r ^ 2 * 3 - C.s * C.t * 2) * pow_mul_pow_eq_one 4 C'.u.inv_mul
  · linear_combination
      C.r * C.u⁻¹ ^ 6 * C'.u⁻¹ ^ 4 * (C'.r * W.a₂ * 2 - C'.r * C'.s * W.a₁ + C'.r ^ 2 * 3 + W.a₄
          - C'.s * C'.t * 2 - C'.s * W.a₃ - C'.t * W.a₁) * pow_mul_pow_eq_one 2 C'.u.inv_mul
        - C.u⁻¹ ^ 6 * C'.u⁻¹ ^ 3 * C.t * (C'.r * W.a₁ + C'.t * 2 + W.a₃)
          * pow_mul_pow_eq_one 3 C'.u.inv_mul
        + C.r ^ 2 * C.u⁻¹ ^ 6 * C'.u⁻¹ ^ 2 * (C'.r * 3 + W.a₂ - C'.s * W.a₁ - C'.s ^ 2)
          * pow_mul_pow_eq_one 4 C'.u.inv_mul
        - C.r * C.t * C.u⁻¹ ^ 6 * ↑C'.u⁻¹ * (C'.s * 2 + W.a₁) * pow_mul_pow_eq_one 5 C'.u.inv_mul
        + C.u⁻¹ ^ 6 * (C.r ^ 3 - C.t ^ 2) * pow_mul_pow_eq_one 6 C'.u.inv_mul


instance instMulActionVariableChange : MulAction (VariableChange R) (WeierstrassCurve R) where
  smul := fun C W => W.variableChange C
  one_smul := variableChange_id
  mul_smul := variableChange_comp


@[simp]
lemma variableChange_b₂ : (W.variableChange C).b₂ = C.u⁻¹ ^ 2 * (W.b₂ + 12 * C.r) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (W.variableChange C).b₂ (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 2) (HAdd.h …
  -/
  simp only [b₂, variableChange_a₁, variableChange_a₂]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul (↑(Inv.inv C.u)) (HAdd.hAdd W.a₁ (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_b₄ :
    (W.variableChange C).b₄ = C.u⁻¹ ^ 4 * (W.b₄ + C.r * W.b₂ + 6 * C.r ^ 2) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (W.variableChange C).b₄ (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 4) (HAdd.h …
  -/
  simp only [b₂, b₄, variableChange_a₁, variableChange_a₃, variableChange_a₄]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 4) (HSub.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_b₆ : (W.variableChange C).b₆ =
    C.u⁻¹ ^ 6 * (W.b₆ + 2 * C.r * W.b₄ + C.r ^ 2 * W.b₂ + 4 * C.r ^ 3) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (W.variableChange C).b₆ (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 6) (HAdd.h …
  -/
  simp only [b₂, b₄, b₆, variableChange_a₃, variableChange_a₆]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 3) (HAdd.hAd …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_b₈ : (W.variableChange C).b₈ = C.u⁻¹ ^ 8 *
    (W.b₈ + 3 * C.r * W.b₆ + 3 * C.r ^ 2 * W.b₄ + C.r ^ 3 * W.b₂ + 3 * C.r ^ 4) := by
  simp only [b₂, b₄, b₆, b₈, variableChange_a₁, variableChange_a₂, variableChange_a₃,
    variableChange_a₄, variableChange_a₆]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_c₄ : (W.variableChange C).c₄ = C.u⁻¹ ^ 4 * W.c₄ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (W.variableChange C).c₄ (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 4) W.c₄)
  -/
  simp only [c₄, variableChange_b₂, variableChange_b₄]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HSub.hSub (HPow.hPow (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 2) (HAdd.hAd …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_c₆ : (W.variableChange C).c₆ = C.u⁻¹ ^ 6 * W.c₆ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (W.variableChange C).c₆ (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 6) W.c₆)
  -/
  simp only [c₆, variableChange_b₂, variableChange_b₄, variableChange_b₆]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow (HMul.hMul (HPow.hPow (↑(Inv.in …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_Δ : (W.variableChange C).Δ = C.u⁻¹ ^ 12 * W.Δ := by
  simp only [b₂, b₄, b₆, b₈, Δ, variableChange_a₁, variableChange_a₂, variableChange_a₃,
    variableChange_a₄, variableChange_a₆]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HAdd.hAd …
  -/
  ring1
  /-
    🎉 no goals
  -/


instance : (W.variableChange C).IsElliptic := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    inst✝ : W.IsElliptic
    ⊢ (W.variableChange C).IsElliptic
  -/
  rw [isElliptic_iff, variableChange_Δ]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    inst✝ : W.IsElliptic
    ⊢ IsUnit (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 12) W.Δ)
  -/
  exact (C.u⁻¹.isUnit.pow 12).mul W.isUnit_Δ
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
@[simp]
lemma variableChange_Δ' : (W.variableChange C).Δ' = C.u⁻¹ ^ 12 * W.Δ' := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    inst✝ : W.IsElliptic
    ⊢ Eq (W.variableChange C).Δ' (HMul.hMul (HPow.hPow (Inv.inv C.u) 12) W.Δ')
  -/
  simp_rw [Units.ext_iff, Units.val_mul, coe_Δ', variableChange_Δ, Units.val_pow_eq_pow_val]
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
lemma coe_variableChange_Δ' : ((W.variableChange C).Δ' : R) = C.u⁻¹ ^ 12 * W.Δ' := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    inst✝ : W.IsElliptic
    ⊢ Eq (↑(W.variableChange C).Δ') (HMul.hMul (HPow.hPow (↑(Inv.inv C.u)) 12) ↑W. …
  -/
  simp_rw [coe_Δ', variableChange_Δ]
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
lemma inv_variableChange_Δ' : (W.variableChange C).Δ'⁻¹ = C.u ^ 12 * W.Δ'⁻¹ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    inst✝ : W.IsElliptic
    ⊢ Eq (Inv.inv (W.variableChange C).Δ') (HMul.hMul (HPow.hPow C.u 12) (Inv.inv  …
  -/
  rw [variableChange_Δ', mul_inv, inv_pow, inv_inv]
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
lemma coe_inv_variableChange_Δ' : (↑(W.variableChange C).Δ'⁻¹ : R) = C.u ^ 12 * W.Δ'⁻¹ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    C : WeierstrassCurve.VariableChange R
    inst✝ : W.IsElliptic
    ⊢ Eq (↑(Inv.inv (W.variableChange C).Δ')) (HMul.hMul (HPow.hPow (↑C.u) 12) ↑(I …
  -/
  rw [inv_variableChange_Δ', Units.val_mul, Units.val_pow_eq_pow_val]
  /-
    🎉 no goals
  -/


@[simp]
lemma variableChange_j : (W.variableChange C).j = W.j := by
  rw [j, coe_inv_variableChange_Δ', variableChange_c₄, j, mul_pow, ← pow_mul, ← mul_assoc,
    mul_right_comm (C.u.val ^ 12), ← mul_pow, C.u.mul_inv, one_pow, one_mul]


/-- The change of variables mapped over a ring homomorphism `φ : R →+* A`. -/
@[simps]
def map : VariableChange A :=
  ⟨Units.map φ C.u, φ C.r, φ C.s, φ C.t⟩


/-- The change of variables base changed to an algebra `A` over `R`. -/
abbrev baseChange [Algebra R A] : VariableChange A :=
  C.map <| algebraMap R A


@[simp]
lemma map_id : C.map (RingHom.id R) = C :=
  rfl


lemma map_map {A : Type v} [CommRing A] (φ : R →+* A) {B : Type w} [CommRing B] (ψ : A →+* B) :
    (C.map φ).map ψ = C.map (ψ.comp φ) :=
  rfl


@[simp]
lemma map_baseChange {S : Type s} [CommRing S] [Algebra R S] {A : Type v} [CommRing A] [Algebra R A]
    [Algebra S A] [IsScalarTower R S A] {B : Type w} [CommRing B] [Algebra R B] [Algebra S B]
    [IsScalarTower R S B] (ψ : A →ₐ[S] B) : (C.baseChange A).map ψ = C.baseChange B :=
  congr_arg C.map <| ψ.comp_algebraMap_of_tower R


lemma map_injective {φ : R →+* A} (hφ : Function.Injective φ) :
    Function.Injective <| map (φ := φ) := fun _ _ h => by
  /-
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    hφ : Function.Injective ⇑φ
    x✝¹ x✝ : WeierstrassCurve.VariableChange R
    h : Eq (WeierstrassCurve.VariableChange.map φ x✝¹) (WeierstrassCurve.VariableC …
    ⊢ Eq x✝¹ x✝
  -/
  rcases mk.inj h with ⟨h, _, _, _⟩
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    hφ : Function.Injective ⇑φ
    x✝¹ x✝ : WeierstrassCurve.VariableChange R
    h✝ : Eq (WeierstrassCurve.VariableChange.map φ x✝¹) (WeierstrassCurve.Variable …
    h : Eq ((Units.map ↑φ) x✝¹.u) ((Units.map ↑φ) x✝.u)
    left✝¹ : Eq (φ x✝¹.r) (φ x✝.r)
    left✝ : Eq (φ x✝¹.s) (φ x✝.s)
    right✝ : Eq (φ x✝¹.t) (φ x✝.t)
    ⊢ Eq x✝¹ x✝
  -/
  replace h := (Units.mk.inj h).left
  /-
    case intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    hφ : Function.Injective ⇑φ
    x✝¹ x✝ : WeierstrassCurve.VariableChange R
    h✝ : Eq (WeierstrassCurve.VariableChange.map φ x✝¹) (WeierstrassCurve.Variable …
    left✝¹ : Eq (φ x✝¹.r) (φ x✝.r)
    left✝ : Eq (φ x✝¹.s) (φ x✝.s)
    right✝ : Eq (φ x✝¹.t) (φ x✝.t)
    h : Eq (↑φ ↑x✝¹.u) (↑φ ↑x✝.u)
    ⊢ Eq x✝¹ x✝
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
  ext <;> apply_fun _ using hφ <;> assumption
                                   /-
                                     🎉 no goals
                                   -/


private lemma id_map : (id : VariableChange R).map φ = id := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (WeierstrassCurve.VariableChange.map φ WeierstrassCurve.VariableChange.id …
  -/
  simp only [id, map]
  /-
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq { u := (Units.map ↑φ) 1, r := φ 0, s := φ 0, t := φ 0 } { u := 1, r := 0, …
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
  ext <;> simp only [map_one, Units.val_one, map_zero]
          /-
            🎉 no goals
          -/


private lemma comp_map (C' : VariableChange R) : (C.comp C').map φ = (C.map φ).comp (C'.map φ) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    C C' : WeierstrassCurve.VariableChange R
    ⊢ Eq (WeierstrassCurve.VariableChange.map φ (C.comp C')) ((WeierstrassCurve.Va …
  -/
  simp only [comp, map]
  /-
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    C C' : WeierstrassCurve.VariableChange R
    ⊢ Eq { u := (Units.map ↑φ) (HMul.hMul C.u C'.u), r := φ (HAdd.hAdd (HMul.hMul  …
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
  ext <;> map_simp <;> simp only [Units.coe_map, Units.coe_map_inv, MonoidHom.coe_coe]
                       /-
                         🎉 no goals
                       -/


/-- The map over a ring homomorphism of a change of variables is a group homomorphism. -/
def mapHom : VariableChange R →* VariableChange A where
  toFun := map φ
  map_one' := id_map φ
  map_mul' := comp_map φ


lemma map_variableChange (C : VariableChange R) :
    (W.map φ).variableChange (C.map φ) = (W.variableChange C).map φ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq ((W.map φ).variableChange (WeierstrassCurve.VariableChange.map φ C)) ((W. …
  -/
  simp only [map, variableChange, VariableChange.map]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    C : WeierstrassCurve.VariableChange R
    ⊢ Eq { a₁ := HMul.hMul (↑(Inv.inv ((Units.map ↑φ) C.u))) (HAdd.hAdd (φ W.a₁) ( …
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
                       /-
                         🎉 no goals
                       -/
  ext <;> map_simp <;> simp only [Units.coe_map, Units.coe_map_inv, MonoidHom.coe_coe]
                       /-
                         🎉 no goals
                       -/


