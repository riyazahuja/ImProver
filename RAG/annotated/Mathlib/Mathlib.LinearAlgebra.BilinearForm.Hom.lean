/-- Auxiliary definition to define `toLinHom`; see below. -/
def toLinHomAux₁ (A : BilinForm R M) (x : M) : M →ₗ[R] R := A x


/-- Auxiliary definition to define `toLinHom`; see below. -/
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
def toLinHomAux₂ (A : BilinForm R M) : M →ₗ[R] M →ₗ[R] R := A


/-- The linear map obtained from a `BilinForm` by fixing the left co-ordinate and evaluating in
the right. -/
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
def toLinHom : BilinForm R M →ₗ[R] M →ₗ[R] M →ₗ[R] R := LinearMap.id


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
theorem toLin'_apply (A : BilinForm R M) (x : M) : toLinHom (M := M) A x = A x :=
  rfl


theorem sum_left {α} (t : Finset α) (g : α → M) (w : M) :
    B (∑ i ∈ t, g i) w = ∑ i ∈ t, B (g i) w :=
  B.map_sum₂ t g w


theorem sum_right {α} (t : Finset α) (w : M) (g : α → M) :
    B w (∑ i ∈ t, g i) = ∑ i ∈ t, B w (g i) := map_sum _ _ _


theorem sum_apply {α} (t : Finset α) (B : α → BilinForm R M) (v w : M) :
    (∑ i ∈ t, B i) v w = ∑ i ∈ t, B i v w := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type u_7
    t : Finset α
    B : α → LinearMap.BilinForm R M
    v w : M
    ⊢ Eq (((t.sum fun i => B i) v) w) (t.sum fun i => ((B i) v) w)
  -/
  simp only [coeFn_sum, Finset.sum_apply]
  /-
    🎉 no goals
  -/


/-- The linear map obtained from a `BilinForm` by fixing the right co-ordinate and evaluating in
the left. -/
def toLinHomFlip : BilinForm R M →ₗ[R] M →ₗ[R] M →ₗ[R] R :=
  flipHom.toLinearMap


theorem toLin'Flip_apply (A : BilinForm R M) (x : M) : toLinHomFlip (M := M) A x = fun y => A y x :=
  rfl


/-- A map with two arguments that is linear in both is a bilinear form.

This is an auxiliary definition for the full linear equivalence `LinearMap.toBilin`.
-/
def LinearMap.toBilinAux (f : M →ₗ[R] M →ₗ[R] R) : BilinForm R M := f


set_option linter.deprecated false in
/-- Bilinear forms are linearly equivalent to maps with two arguments that are linear in both. -/
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
def LinearMap.BilinForm.toLin : BilinForm R M ≃ₗ[R] M →ₗ[R] M →ₗ[R] R :=
  { BilinForm.toLinHom with
    invFun := LinearMap.toBilinAux
    left_inv := fun _ => rfl
    right_inv := fun _ => rfl }


set_option linter.deprecated false in
/-- A map with two arguments that is linear in both is linearly equivalent to bilinear form. -/
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
def LinearMap.toBilin : (M →ₗ[R] M →ₗ[R] R) ≃ₗ[R] BilinForm R M :=
  BilinForm.toLin.symm


@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
theorem LinearMap.toBilinAux_eq (f : M →ₗ[R] M →ₗ[R] R) :
    LinearMap.toBilinAux f = f :=
  rfl


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
theorem LinearMap.toBilin_symm :
    (LinearMap.toBilin.symm : BilinForm R M ≃ₗ[R] _) = BilinForm.toLin :=
  rfl


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
theorem BilinForm.toLin_symm :
    (BilinForm.toLin.symm : _ ≃ₗ[R] BilinForm R M) = LinearMap.toBilin :=
  LinearMap.toBilin.symm_symm


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
theorem LinearMap.toBilin_apply (f : M →ₗ[R] M →ₗ[R] R) (x y : M) :
    toBilin f x y = f x y :=
  rfl


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-04-26")]
theorem BilinForm.toLin_apply (x : M) : BilinForm.toLin B x = B x :=
  rfl


/-- Apply a linear map on the output of a bilinear form. -/
@[simps!]
def compBilinForm (f : R →ₗ[R'] R') (B : BilinForm R M) : BilinForm R' M :=
  compr₂ (restrictScalars₁₂ R' R' B) f


/-- Apply a linear map on the left and right argument of a bilinear form. -/
def comp (B : BilinForm R M') (l r : M →ₗ[R] M') : BilinForm R M := B.compl₁₂ l r


/-- Apply a linear map to the left argument of a bilinear form. -/
def compLeft (B : BilinForm R M) (f : M →ₗ[R] M) : BilinForm R M :=
  B.comp f LinearMap.id


/-- Apply a linear map to the right argument of a bilinear form. -/
def compRight (B : BilinForm R M) (f : M →ₗ[R] M) : BilinForm R M :=
  B.comp LinearMap.id f


theorem comp_comp {M'' : Type*} [AddCommMonoid M''] [Module R M''] (B : BilinForm R M'')
    (l r : M →ₗ[R] M') (l' r' : M' →ₗ[R] M'') :
    (B.comp l' r').comp l r = B.comp (l'.comp l) (r'.comp r) :=
  rfl


@[simp]
theorem compLeft_compRight (B : BilinForm R M) (l r : M →ₗ[R] M) :
    (B.compLeft l).compRight r = B.comp l r :=
  rfl


@[simp]
theorem compRight_compLeft (B : BilinForm R M) (l r : M →ₗ[R] M) :
    (B.compRight r).compLeft l = B.comp l r :=
  rfl


@[simp]
theorem comp_apply (B : BilinForm R M') (l r : M →ₗ[R] M') (v w) : B.comp l r v w = B (l v) (r w) :=
  rfl


@[simp]
theorem compLeft_apply (B : BilinForm R M) (f : M →ₗ[R] M) (v w) : B.compLeft f v w = B (f v) w :=
  rfl


@[simp]
theorem compRight_apply (B : BilinForm R M) (f : M →ₗ[R] M) (v w) : B.compRight f v w = B v (f w) :=
  rfl


@[simp]
theorem comp_id_left (B : BilinForm R M) (r : M →ₗ[R] M) :
    B.comp LinearMap.id r = B.compRight r := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    r : LinearMap (RingHom.id R) M M
    ⊢ Eq (B.comp LinearMap.id r) (B.compRight r)
  -/
  ext
  /-
    case H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    r : LinearMap (RingHom.id R) M M
    x✝ y✝ : M
    ⊢ Eq (((B.comp LinearMap.id r) x✝) y✝) (((B.compRight r) x✝) y✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_id_right (B : BilinForm R M) (l : M →ₗ[R] M) :
    B.comp l LinearMap.id = B.compLeft l := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    l : LinearMap (RingHom.id R) M M
    ⊢ Eq (B.comp l LinearMap.id) (B.compLeft l)
  -/
  ext
  /-
    case H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    l : LinearMap (RingHom.id R) M M
    x✝ y✝ : M
    ⊢ Eq (((B.comp l LinearMap.id) x✝) y✝) (((B.compLeft l) x✝) y✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem compLeft_id (B : BilinForm R M) : B.compLeft LinearMap.id = B := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ⊢ Eq (B.compLeft LinearMap.id) B
  -/
  ext
  /-
    case H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    x✝ y✝ : M
    ⊢ Eq (((B.compLeft LinearMap.id) x✝) y✝) ((B x✝) y✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem compRight_id (B : BilinForm R M) : B.compRight LinearMap.id = B := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ⊢ Eq (B.compRight LinearMap.id) B
  -/
  ext
  /-
    case H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    x✝ y✝ : M
    ⊢ Eq (((B.compRight LinearMap.id) x✝) y✝) ((B x✝) y✝)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Shortcut for `comp_id_{left,right}` followed by `comp{Right,Left}_id`,
-- Needs higher priority to be applied

@[simp high]
theorem comp_id_id (B : BilinForm R M) : B.comp LinearMap.id LinearMap.id = B := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ⊢ Eq (B.comp LinearMap.id LinearMap.id) B
  -/
  ext
  /-
    case H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    x✝ y✝ : M
    ⊢ Eq (((B.comp LinearMap.id LinearMap.id) x✝) y✝) ((B x✝) y✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comp_inj (B₁ B₂ : BilinForm R M') {l r : M →ₗ[R] M'} (hₗ : Function.Surjective l)
    (hᵣ : Function.Surjective r) : B₁.comp l r = B₂.comp l r ↔ B₁ = B₂ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type w
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    B₁ B₂ : LinearMap.BilinForm R M'
    l r : LinearMap (RingHom.id R) M M'
    hₗ : Function.Surjective ⇑l
    hᵣ : Function.Surjective ⇑r
    ⊢ Iff (Eq (B₁.comp l r) (B₂.comp l r)) (Eq B₁ B₂)
  -/
  constructor <;> intro h
  · -- B₁.comp l r = B₂.comp l r → B₁ = B₂
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq (B₁.comp l r) (B₂.comp l r)
      ⊢ Eq B₁ B₂
    -/
    ext x y
    /-
      case mp.H
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq (B₁.comp l r) (B₂.comp l r)
      x y : M'
      ⊢ Eq ((B₁ x) y) ((B₂ x) y)
    -/
    cases' hₗ x with x' hx
    /-
      case mp.H.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq (B₁.comp l r) (B₂.comp l r)
      x y : M'
      x' : M
      hx : Eq (l x') x
      ⊢ Eq ((B₁ x) y) ((B₂ x) y)
    -/
    subst hx
    /-
      case mp.H.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq (B₁.comp l r) (B₂.comp l r)
      y : M'
      x' : M
      ⊢ Eq ((B₁ (l x')) y) ((B₂ (l x')) y)
    -/
    cases' hᵣ y with y' hy
    /-
      case mp.H.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq (B₁.comp l r) (B₂.comp l r)
      y : M'
      x' y' : M
      hy : Eq (r y') y
      ⊢ Eq ((B₁ (l x')) y) ((B₂ (l x')) y)
    -/
    subst hy
    /-
      case mp.H.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq (B₁.comp l r) (B₂.comp l r)
      x' y' : M
      ⊢ Eq ((B₁ (l x')) (r y')) ((B₂ (l x')) (r y'))
    -/
    rw [← comp_apply, ← comp_apply, h]
    /-
      🎉 no goals
    -/
  · -- B₁ = B₂ → B₁.comp l r = B₂.comp l r
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type w
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B₁ B₂ : LinearMap.BilinForm R M'
      l r : LinearMap (RingHom.id R) M M'
      hₗ : Function.Surjective ⇑l
      hᵣ : Function.Surjective ⇑r
      h : Eq B₁ B₂
      ⊢ Eq (B₁.comp l r) (B₂.comp l r)
    -/
    rw [h]
    /-
      🎉 no goals
    -/


/-- Apply a linear equivalence on the arguments of a bilinear form. -/
def congr (e : M ≃ₗ[R] M') : BilinForm R M ≃ₗ[R] BilinForm R M' :=
  LinearEquiv.congrRight (LinearEquiv.congrLeft _ _ e) ≪≫ₗ LinearEquiv.congrLeft _ _ e


@[simp]
theorem congr_apply (e : M ≃ₗ[R] M') (B : BilinForm R M) (x y : M') :
    congr e B x y = B (e.symm x) (e.symm y) :=
  rfl


@[simp]
theorem congr_symm (e : M ≃ₗ[R] M') : (congr e).symm = congr e.symm := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_7
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    e : LinearEquiv (RingHom.id R) M M'
    ⊢ Eq (LinearMap.BilinForm.congr e).symm (LinearMap.BilinForm.congr e.symm)
  -/
  ext
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_7
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    e : LinearEquiv (RingHom.id R) M M'
    x✝¹ : LinearMap.BilinForm R M'
    x✝ y✝ : M
    ⊢ Eq ((((LinearMap.BilinForm.congr e).symm x✝¹) x✝) y✝) ((((LinearMap.BilinFor …
  -/
  simp only [congr_apply, LinearEquiv.symm_symm]
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_7
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    e : LinearEquiv (RingHom.id R) M M'
    x✝¹ : LinearMap.BilinForm R M'
    x✝ y✝ : M
    ⊢ Eq ((((LinearMap.BilinForm.congr e).symm x✝¹) x✝) y✝) ((x✝¹ (e x✝)) (e y✝))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem congr_refl : congr (LinearEquiv.refl R M) = LinearEquiv.refl R _ :=
  LinearEquiv.ext fun _ => ext₂ fun _ _ => rfl


theorem congr_trans (e : M ≃ₗ[R] M') (f : M' ≃ₗ[R] M'') :
    (congr e).trans (congr f) = congr (e.trans f) :=
  rfl


theorem congr_congr (e : M' ≃ₗ[R] M'') (f : M ≃ₗ[R] M') (B : BilinForm R M) :
    congr e (congr f B) = congr (f.trans e) B :=
  rfl


theorem congr_comp (e : M ≃ₗ[R] M') (B : BilinForm R M) (l r : M'' →ₗ[R] M') :
    (congr e B).comp l r =
      B.comp (LinearMap.comp (e.symm : M' →ₗ[R] M) l)
        (LinearMap.comp (e.symm : M' →ₗ[R] M) r) :=
  rfl


theorem comp_congr (e : M' ≃ₗ[R] M'') (B : BilinForm R M) (l r : M' →ₗ[R] M) :
    congr e (B.comp l r) =
      B.comp (l.comp (e.symm : M'' →ₗ[R] M')) (r.comp (e.symm : M'' →ₗ[R] M')) :=
  rfl


/-- When `N₁` and `N₂` are equivalent, bilinear maps on `M` into `N₁` are equivalent to bilinear
maps into `N₂`. -/
def _root_.LinearEquiv.congrRight₂ (e : N₁ ≃ₗ[R] N₂) : BilinMap R M N₁ ≃ₗ[R] BilinMap R M N₂ :=
  LinearEquiv.congrRight (LinearEquiv.congrRight e)


@[simp]
theorem _root_.LinearEquiv.congrRight₂_apply (e : N₁ ≃ₗ[R] N₂) (B : BilinMap R M N₁) :
    LinearEquiv.congrRight₂ e B = compr₂ B e := rfl


@[simp]
theorem _root_.LinearEquiv.congrRight₂_refl :
    LinearEquiv.congrRight₂ (.refl R N₁) = .refl R (BilinMap R M N₁) := rfl


@[simp]
theorem _root_.LinearEquiv.congrRight_symm (e : N₁ ≃ₗ[R] N₂) :
    (LinearEquiv.congrRight₂ e (M := M)).symm = LinearEquiv.congrRight₂ e.symm :=
  rfl


theorem _root_.LinearEquiv.congrRight₂_trans (e₁₂ : N₁ ≃ₗ[R] N₂) (e₂₃ : N₂ ≃ₗ[R] N₃) :
    LinearEquiv.congrRight₂ (M := M) (e₁₂ ≪≫ₗ e₂₃) =
    LinearEquiv.congrRight₂ e₁₂ ≪≫ₗ LinearEquiv.congrRight₂ e₂₃ :=
  rfl


/-- `linMulLin f g` is the bilinear form mapping `x` and `y` to `f x * g y` -/
def linMulLin (f g : M →ₗ[R] R) : BilinForm R M := (LinearMap.mul R R).compl₁₂ f g


@[simp]
theorem linMulLin_apply (x y) : linMulLin f g x y = f x * g y :=
  rfl


@[simp]
theorem linMulLin_comp (l r : M' →ₗ[R] M) :
    (linMulLin f g).comp l r = linMulLin (f.comp l) (g.comp r) :=
  rfl


@[simp]
theorem linMulLin_compLeft (l : M →ₗ[R] M) :
    (linMulLin f g).compLeft l = linMulLin (f.comp l) g :=
  rfl


@[simp]
theorem linMulLin_compRight (r : M →ₗ[R] M) :
    (linMulLin f g).compRight r = linMulLin f (g.comp r) :=
  rfl


/-- Two bilinear forms are equal when they are equal on all basis vectors. -/
theorem ext_basis (h : ∀ i j, B (b i) (b j) = F₂ (b i) (b j)) : B = F₂ :=
  b.ext fun i => b.ext fun j => h i j


/-- Write out `B x y` as a sum over `B (b i) (b j)` if `b` is a basis. -/
theorem sum_repr_mul_repr_mul (x y : M) :
    ((b.repr x).sum fun i xi => (b.repr y).sum fun j yj => xi • yj • B (b i) (b j)) = B x y := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ι : Type u_9
    b : Basis ι R M
    x y : M
    ⊢ Eq ((b.repr x).sum fun i xi => (b.repr y).sum fun j yj => HSMul.hSMul xi (HS …
  -/
  conv_rhs => rw [← b.linearCombination_repr x, ← b.linearCombination_repr y]
  simp_rw [Finsupp.linearCombination_apply, Finsupp.sum, sum_left, sum_right, smul_left, smul_right,
    smul_eq_mul]


