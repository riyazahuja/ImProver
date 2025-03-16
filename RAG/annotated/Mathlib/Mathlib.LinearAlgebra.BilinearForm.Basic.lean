@[deprecated "No deprecation message was provided." (since := "2024-04-14")]
theorem coeFn_congr : ∀ {x x' y y' : M}, x = x' → y = y' → B x y = B x' y'
  | _, _, _, _, rfl, rfl => rfl


theorem add_left (x y z : M) : B (x + y) z = B x z + B y z := map_add₂ _ _ _ _


theorem smul_left (a : R) (x y : M) : B (a • x) y = a * B x y := map_smul₂ _ _ _ _


theorem add_right (x y z : M) : B x (y + z) = B x y + B x z := map_add _ _ _


theorem smul_right (a : R) (x y : M) : B x (a • y) = a * B x y := map_smul _ _ _


theorem zero_left (x : M) : B 0 x = 0 := map_zero₂ _ _


theorem zero_right (x : M) : B x 0 = 0 := map_zero _


theorem neg_left (x y : M₁) : B₁ (-x) y = -B₁ x y := map_neg₂ _ _ _


theorem neg_right (x y : M₁) : B₁ x (-y) = -B₁ x y := map_neg _ _


theorem sub_left (x y z : M₁) : B₁ (x - y) z = B₁ x z - B₁ y z := map_sub₂ _ _ _ _


theorem sub_right (x y z : M₁) : B₁ x (y - z) = B₁ x y - B₁ x z := map_sub _ _ _


lemma smul_left_of_tower (r : S) (x y : M) : B (r • x) y = r • B x y := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    S : Type u_3
    inst✝³ : CommSemiring S
    inst✝² : Algebra S R
    inst✝¹ : Module S M
    inst✝ : IsScalarTower S R M
    B : LinearMap.BilinForm R M
    r : S
    x y : M
    ⊢ Eq ((B (HSMul.hSMul r x)) y) (HSMul.hSMul r ((B x) y))
  -/
  rw [← IsScalarTower.algebraMap_smul R r, smul_left, Algebra.smul_def]
  /-
    🎉 no goals
  -/


lemma smul_right_of_tower (r : S) (x y : M) : B x (r • y) = r • B x y := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    S : Type u_3
    inst✝³ : CommSemiring S
    inst✝² : Algebra S R
    inst✝¹ : Module S M
    inst✝ : IsScalarTower S R M
    B : LinearMap.BilinForm R M
    r : S
    x y : M
    ⊢ Eq ((B x) (HSMul.hSMul r y)) (HSMul.hSMul r ((B x) y))
  -/
  rw [← IsScalarTower.algebraMap_smul R r, smul_right, Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem coe_injective : Function.Injective ((fun B x y => B x y) : BilinForm R M → M → M → R) :=
  fun B D h => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B D : LinearMap.BilinForm R M
      h : Eq ((fun B x y => (B x) y) B) ((fun B x y => (B x) y) D)
      ⊢ Eq B D
    -/
    ext x y
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B D : LinearMap.BilinForm R M
      h : Eq ((fun B x y => (B x) y) B) ((fun B x y => (B x) y) D)
      x y : M
      ⊢ Eq ((B x) y) ((D x) y)
    -/
    apply congrFun₂ h
    /-
      🎉 no goals
    -/


@[ext]
theorem ext (H : ∀ x y : M, B x y = D x y) : B = D := ext₂ H


theorem congr_fun (h : B = D) (x y : M) : B x y = D x y := congr_fun₂ h _ _


@[deprecated "No deprecation message was provided." (since := "2024-04-14")]
theorem coe_zero : ⇑(0 : BilinForm R M) = 0 :=
  rfl


@[simp]
theorem zero_apply (x y : M) : (0 : BilinForm R M) x y = 0 :=
  rfl


@[deprecated "No deprecation message was provided." (since := "2024-04-14")]
theorem coe_add : ⇑(B + D) = B + D :=
  rfl


@[simp]
theorem add_apply (x y : M) : (B + D) x y = B x y + D x y :=
  rfl


@[deprecated "No deprecation message was provided." (since := "2024-04-14")]
theorem coe_neg : ⇑(-B₁) = -B₁ :=
  rfl


@[simp]
theorem neg_apply (x y : M₁) : (-B₁) x y = -B₁ x y :=
  rfl


@[deprecated "No deprecation message was provided." (since := "2024-04-14")]
theorem coe_sub : ⇑(B₁ - D₁) = B₁ - D₁ :=
  rfl


@[simp]
theorem sub_apply (x y : M₁) : (B₁ - D₁) x y = B₁ x y - D₁ x y :=
  rfl


/-- `coeFn` as an `AddMonoidHom` -/
def coeFnAddMonoidHom : BilinForm R M →+ M → M → R where
  toFun := fun B x y => B x y
  map_zero' := rfl
  map_add' _ _ := rfl


/-- Auxiliary construction for the flip of a bilinear form, obtained by exchanging the left and
right arguments. This version is a `LinearMap`; it is later upgraded to a `LinearEquiv`
in `flipHom`. -/
def flipHomAux : (BilinForm R M) →ₗ[R] (BilinForm R M) where
  toFun A := A.flip
  map_add' A₁ A₂ := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : Module R M
      S : Type u_3
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : IsScalarTower S R M
      R₁ : Type u_4
      M₁ : Type u_5
      inst✝⁵ : CommRing R₁
      inst✝⁴ : AddCommGroup M₁
      inst✝³ : Module R₁ M₁
      V : Type u_6
      K : Type u_7
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm R M
      B₁ : LinearMap.BilinForm R₁ M₁
      D : LinearMap.BilinForm R M
      D₁ : LinearMap.BilinForm R₁ M₁
      A₁ A₂ : LinearMap.BilinForm R M
      ⊢ Eq ((fun A => LinearMap.flip A) (HAdd.hAdd A₁ A₂)) (HAdd.hAdd ((fun A => Lin …
    -/
    ext
    /-
      case H
      R : Type u_1
      M : Type u_2
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : Module R M
      S : Type u_3
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : IsScalarTower S R M
      R₁ : Type u_4
      M₁ : Type u_5
      inst✝⁵ : CommRing R₁
      inst✝⁴ : AddCommGroup M₁
      inst✝³ : Module R₁ M₁
      V : Type u_6
      K : Type u_7
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm R M
      B₁ : LinearMap.BilinForm R₁ M₁
      D : LinearMap.BilinForm R M
      D₁ : LinearMap.BilinForm R₁ M₁
      A₁ A₂ : LinearMap.BilinForm R M
      x✝ y✝ : M
      ⊢ Eq ((((fun A => LinearMap.flip A) (HAdd.hAdd A₁ A₂)) x✝) y✝) (((HAdd.hAdd (( …
    -/
    simp only [LinearMap.flip_apply, LinearMap.add_apply]
    /-
      🎉 no goals
    -/
  map_smul' c A := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : Module R M
      S : Type u_3
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : IsScalarTower S R M
      R₁ : Type u_4
      M₁ : Type u_5
      inst✝⁵ : CommRing R₁
      inst✝⁴ : AddCommGroup M₁
      inst✝³ : Module R₁ M₁
      V : Type u_6
      K : Type u_7
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm R M
      B₁ : LinearMap.BilinForm R₁ M₁
      D : LinearMap.BilinForm R M
      D₁ : LinearMap.BilinForm R₁ M₁
      c : R
      A : LinearMap.BilinForm R M
      ⊢ Eq ({ toFun := fun A => LinearMap.flip A, map_add' := ⋯ }.toFun (HSMul.hSMul …
    -/
    ext
    /-
      case H
      R : Type u_1
      M : Type u_2
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : Module R M
      S : Type u_3
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : IsScalarTower S R M
      R₁ : Type u_4
      M₁ : Type u_5
      inst✝⁵ : CommRing R₁
      inst✝⁴ : AddCommGroup M₁
      inst✝³ : Module R₁ M₁
      V : Type u_6
      K : Type u_7
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm R M
      B₁ : LinearMap.BilinForm R₁ M₁
      D : LinearMap.BilinForm R M
      D₁ : LinearMap.BilinForm R₁ M₁
      c : R
      A : LinearMap.BilinForm R M
      x✝ y✝ : M
      ⊢ Eq ((({ toFun := fun A => LinearMap.flip A, map_add' := ⋯ }.toFun (HSMul.hSM …
    -/
    simp only [LinearMap.flip_apply, LinearMap.smul_apply, RingHom.id_apply]
    /-
      🎉 no goals
    -/


theorem flip_flip_aux (A : BilinForm R M) :
    flipHomAux (M := M) (flipHomAux (M := M) A) = A := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : LinearMap.BilinForm R M
    ⊢ Eq (LinearMap.BilinForm.flipHomAux (LinearMap.BilinForm.flipHomAux A)) A
  -/
  ext A
  /-
    case H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A✝ : LinearMap.BilinForm R M
    A y✝ : M
    ⊢ Eq (((LinearMap.BilinForm.flipHomAux (LinearMap.BilinForm.flipHomAux A✝)) A) …
  -/
  simp [flipHomAux]
  /-
    🎉 no goals
  -/


/-- The flip of a bilinear form, obtained by exchanging the left and right arguments. -/
def flipHom : BilinForm R M ≃ₗ[R] BilinForm R M :=
  { flipHomAux with
    invFun := flipHomAux (M := M)
    left_inv := flip_flip_aux
    right_inv := flip_flip_aux }


@[simp]
theorem flip_apply (A : BilinForm R M) (x y : M) : flipHom A x y = A y x :=
  rfl


theorem flip_flip :
    flipHom.trans flipHom = LinearEquiv.refl R (BilinForm R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (LinearMap.BilinForm.flipHom.trans LinearMap.BilinForm.flipHom) (LinearEq …
  -/
  ext A
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : LinearMap.BilinForm R M
    x✝ y✝ : M
    ⊢ Eq ((((LinearMap.BilinForm.flipHom.trans LinearMap.BilinForm.flipHom) A) x✝) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The `flip` of a bilinear form over a commutative ring, obtained by exchanging the left and
right arguments. -/
abbrev flip (B : BilinForm R M) :=
  flipHom B


/-- The restriction of a bilinear form on a submodule. -/
@[simps! apply]
def restrict (B : BilinForm R M) (W : Submodule R M) : BilinForm R W :=
  LinearMap.domRestrict₁₂ B W W


