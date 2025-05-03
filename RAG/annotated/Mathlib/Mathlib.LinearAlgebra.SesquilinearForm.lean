/-- The proposition that two elements of a sesquilinear map space are orthogonal -/
def IsOrtho (B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M) (x : M₁) (y : M₂) : Prop :=
  B x y = 0


theorem isOrtho_def {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} {x y} : B.IsOrtho x y ↔ B x y = 0 :=
  Iff.rfl


theorem isOrtho_zero_left (B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M) (x) : IsOrtho B (0 : M₁) x := by
  /-
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring R₁
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : Module R₁ M₁
    inst✝⁴ : CommSemiring R₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₂ M₂
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ : RingHom R₁ R
    I₂ : RingHom R₂ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
    x : M₂
    ⊢ B.IsOrtho 0 x
  -/
  dsimp only [IsOrtho]
  /-
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring R₁
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : Module R₁ M₁
    inst✝⁴ : CommSemiring R₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₂ M₂
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ : RingHom R₁ R
    I₂ : RingHom R₂ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
    x : M₂
    ⊢ Eq ((B 0) x) 0
  -/
  rw [map_zero B, zero_apply]
  /-
    🎉 no goals
  -/


theorem isOrtho_zero_right (B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M) (x) : IsOrtho B x (0 : M₂) :=
  map_zero (B x)


theorem isOrtho_flip {B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₁'] M} {x y} : B.IsOrtho x y ↔ B.flip.IsOrtho y x := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ I₁' : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
    x y : M₁
    ⊢ Iff (B.IsOrtho x y) (B.flip.IsOrtho y x)
  -/
  simp_rw [isOrtho_def, flip_apply]
  /-
    🎉 no goals
  -/


/-- A set of vectors `v` is orthogonal with respect to some bilinear map `B` if and only
if for all `i ≠ j`, `B (v i) (v j) = 0`. For orthogonality between two elements, use
`BilinForm.isOrtho` -/
def IsOrthoᵢ (B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₁'] M) (v : n → M₁) : Prop :=
  Pairwise (B.IsOrtho on v)


theorem isOrthoᵢ_def {B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₁'] M} {v : n → M₁} :
    B.IsOrthoᵢ v ↔ ∀ i j : n, i ≠ j → B (v i) (v j) = 0 :=
  Iff.rfl


theorem isOrthoᵢ_flip (B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₁'] M) {v : n → M₁} :
    B.IsOrthoᵢ v ↔ B.flip.IsOrthoᵢ v := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ I₁' : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
    v : n → M₁
    ⊢ Iff (B.IsOrthoᵢ v) (B.flip.IsOrthoᵢ v)
  -/
  simp_rw [isOrthoᵢ_def]
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ I₁' : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
    v : n → M₁
    ⊢ Iff (∀ (i j : n), Ne i j → Eq ((B (v i)) (v j)) 0) (∀ (i j : n), Ne i j → Eq …
  -/
  constructor <;> intro h i j hij
    /-
      case mp
      R : Type u_1
      R₁ : Type u_2
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring R₁
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R₁ M₁
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I₁ I₁' : RingHom R₁ R
      B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
      v : n → M₁
      h : ∀ (i j : n), Ne i j → Eq ((B (v i)) (v j)) 0
      i j : n
      hij : Ne i j
      ⊢ Eq ((B.flip (v i)) (v j)) 0
    -/
  · rw [flip_apply]
    /-
      case mp
      R : Type u_1
      R₁ : Type u_2
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring R₁
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R₁ M₁
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I₁ I₁' : RingHom R₁ R
      B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
      v : n → M₁
      h : ∀ (i j : n), Ne i j → Eq ((B (v i)) (v j)) 0
      i j : n
      hij : Ne i j
      ⊢ Eq ((B (v j)) (v i)) 0
    -/
    exact h j i (Ne.symm hij)
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ I₁' : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
    v : n → M₁
    h : ∀ (i j : n), Ne i j → Eq ((B.flip (v i)) (v j)) 0
    i j : n
    hij : Ne i j
    ⊢ Eq ((B (v i)) (v j)) 0
  -/
  simp_rw [flip_apply] at h
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I₁ I₁' : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₁' M₁ M)
    v : n → M₁
    h : ∀ (i j : n), Ne i j → Eq ((B (v j)) (v i)) 0
    i j : n
    hij : Ne i j
    ⊢ Eq ((B (v i)) (v j)) 0
  -/
  exact h j i (Ne.symm hij)
  /-
    🎉 no goals
  -/


theorem ortho_smul_left {B : V₁ →ₛₗ[I₁] V₂ →ₛₗ[I₂] V} {x y} {a : K₁} (ha : a ≠ 0) :
    IsOrtho B x y ↔ IsOrtho B (a • x) y := by
  /-
    K : Type u_13
    K₁ : Type u_14
    K₂ : Type u_15
    V : Type u_16
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    inst✝⁵ : Field K₁
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K₁ V₁
    inst✝² : Field K₂
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K₂ V₂
    I₁ : RingHom K₁ K
    I₂ : RingHom K₂ K
    B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
    x : V₁
    y : V₂
    a : K₁
    ha : Ne a 0
    ⊢ Iff (B.IsOrtho x y) (B.IsOrtho (HSMul.hSMul a x) y)
  -/
  dsimp only [IsOrtho]
  /-
    K : Type u_13
    K₁ : Type u_14
    K₂ : Type u_15
    V : Type u_16
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    inst✝⁵ : Field K₁
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K₁ V₁
    inst✝² : Field K₂
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K₂ V₂
    I₁ : RingHom K₁ K
    I₂ : RingHom K₂ K
    B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
    x : V₁
    y : V₂
    a : K₁
    ha : Ne a 0
    ⊢ Iff (Eq ((B x) y) 0) (Eq ((B (HSMul.hSMul a x)) y) 0)
  -/
  constructor <;> intro H
    /-
      case mp
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      inst✝⁸ : Field K
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module K V
      inst✝⁵ : Field K₁
      inst✝⁴ : AddCommGroup V₁
      inst✝³ : Module K₁ V₁
      inst✝² : Field K₂
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K₂ V₂
      I₁ : RingHom K₁ K
      I₂ : RingHom K₂ K
      B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
      x : V₁
      y : V₂
      a : K₁
      ha : Ne a 0
      H : Eq ((B x) y) 0
      ⊢ Eq ((B (HSMul.hSMul a x)) y) 0
    -/
  · rw [map_smulₛₗ₂, H, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      inst✝⁸ : Field K
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module K V
      inst✝⁵ : Field K₁
      inst✝⁴ : AddCommGroup V₁
      inst✝³ : Module K₁ V₁
      inst✝² : Field K₂
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K₂ V₂
      I₁ : RingHom K₁ K
      I₂ : RingHom K₂ K
      B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
      x : V₁
      y : V₂
      a : K₁
      ha : Ne a 0
      H : Eq ((B (HSMul.hSMul a x)) y) 0
      ⊢ Eq ((B x) y) 0
    -/
  · rw [map_smulₛₗ₂, smul_eq_zero] at H
    /-
      case mpr
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      inst✝⁸ : Field K
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module K V
      inst✝⁵ : Field K₁
      inst✝⁴ : AddCommGroup V₁
      inst✝³ : Module K₁ V₁
      inst✝² : Field K₂
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K₂ V₂
      I₁ : RingHom K₁ K
      I₂ : RingHom K₂ K
      B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
      x : V₁
      y : V₂
      a : K₁
      ha : Ne a 0
      H : Or (Eq (I₁ a) 0) (Eq ((B x) y) 0)
      ⊢ Eq ((B x) y) 0
    -/
    cases' H with H H
      /-
        case mpr.inl
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₁
        ha : Ne a 0
        H : Eq (I₁ a) 0
        ⊢ Eq ((B x) y) 0
      -/
    · rw [map_eq_zero I₁] at H
      /-
        case mpr.inl
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₁
        ha : Ne a 0
        H : Eq a 0
        ⊢ Eq ((B x) y) 0
      -/
      trivial
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₁
        ha : Ne a 0
        H : Eq ((B x) y) 0
        ⊢ Eq ((B x) y) 0
      -/
    · exact H
      /-
        🎉 no goals
      -/

-- todo: this also holds for [CommRing R] [IsDomain R] when J₂ is invertible

theorem ortho_smul_right {B : V₁ →ₛₗ[I₁] V₂ →ₛₗ[I₂] V} {x y} {a : K₂} {ha : a ≠ 0} :
    IsOrtho B x y ↔ IsOrtho B x (a • y) := by
  /-
    K : Type u_13
    K₁ : Type u_14
    K₂ : Type u_15
    V : Type u_16
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    inst✝⁵ : Field K₁
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K₁ V₁
    inst✝² : Field K₂
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K₂ V₂
    I₁ : RingHom K₁ K
    I₂ : RingHom K₂ K
    B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
    x : V₁
    y : V₂
    a : K₂
    ha : Ne a 0
    ⊢ Iff (B.IsOrtho x y) (B.IsOrtho x (HSMul.hSMul a y))
  -/
  dsimp only [IsOrtho]
  /-
    K : Type u_13
    K₁ : Type u_14
    K₂ : Type u_15
    V : Type u_16
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    inst✝⁵ : Field K₁
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K₁ V₁
    inst✝² : Field K₂
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K₂ V₂
    I₁ : RingHom K₁ K
    I₂ : RingHom K₂ K
    B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
    x : V₁
    y : V₂
    a : K₂
    ha : Ne a 0
    ⊢ Iff (Eq ((B x) y) 0) (Eq ((B x) (HSMul.hSMul a y)) 0)
  -/
  constructor <;> intro H
    /-
      case mp
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      inst✝⁸ : Field K
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module K V
      inst✝⁵ : Field K₁
      inst✝⁴ : AddCommGroup V₁
      inst✝³ : Module K₁ V₁
      inst✝² : Field K₂
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K₂ V₂
      I₁ : RingHom K₁ K
      I₂ : RingHom K₂ K
      B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
      x : V₁
      y : V₂
      a : K₂
      ha : Ne a 0
      H : Eq ((B x) y) 0
      ⊢ Eq ((B x) (HSMul.hSMul a y)) 0
    -/
  · rw [map_smulₛₗ, H, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      inst✝⁸ : Field K
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module K V
      inst✝⁵ : Field K₁
      inst✝⁴ : AddCommGroup V₁
      inst✝³ : Module K₁ V₁
      inst✝² : Field K₂
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K₂ V₂
      I₁ : RingHom K₁ K
      I₂ : RingHom K₂ K
      B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
      x : V₁
      y : V₂
      a : K₂
      ha : Ne a 0
      H : Eq ((B x) (HSMul.hSMul a y)) 0
      ⊢ Eq ((B x) y) 0
    -/
  · rw [map_smulₛₗ, smul_eq_zero] at H
    /-
      case mpr
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      inst✝⁸ : Field K
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module K V
      inst✝⁵ : Field K₁
      inst✝⁴ : AddCommGroup V₁
      inst✝³ : Module K₁ V₁
      inst✝² : Field K₂
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K₂ V₂
      I₁ : RingHom K₁ K
      I₂ : RingHom K₂ K
      B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
      x : V₁
      y : V₂
      a : K₂
      ha : Ne a 0
      H : Or (Eq (I₂ a) 0) (Eq ((B x) y) 0)
      ⊢ Eq ((B x) y) 0
    -/
    cases' H with H H
      /-
        case mpr.inl
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₂
        ha : Ne a 0
        H : Eq (I₂ a) 0
        ⊢ Eq ((B x) y) 0
      -/
    · simp only [map_eq_zero] at H
      /-
        case mpr.inl
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₂
        ha : Ne a 0
        H : Eq a 0
        ⊢ Eq ((B x) y) 0
      -/
      exfalso
      /-
        case mpr.inl
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₂
        ha : Ne a 0
        H : Eq a 0
        ⊢ False
      -/
      exact ha H
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        K : Type u_13
        K₁ : Type u_14
        K₂ : Type u_15
        V : Type u_16
        V₁ : Type u_17
        V₂ : Type u_18
        inst✝⁸ : Field K
        inst✝⁷ : AddCommGroup V
        inst✝⁶ : Module K V
        inst✝⁵ : Field K₁
        inst✝⁴ : AddCommGroup V₁
        inst✝³ : Module K₁ V₁
        inst✝² : Field K₂
        inst✝¹ : AddCommGroup V₂
        inst✝ : Module K₂ V₂
        I₁ : RingHom K₁ K
        I₂ : RingHom K₂ K
        B : LinearMap I₁ V₁ (LinearMap I₂ V₂ V)
        x : V₁
        y : V₂
        a : K₂
        ha : Ne a 0
        H : Eq ((B x) y) 0
        ⊢ Eq ((B x) y) 0
      -/
    · exact H
      /-
        🎉 no goals
      -/


/-- A set of orthogonal vectors `v` with respect to some sesquilinear map `B` is linearly
  independent if for all `i`, `B (v i) (v i) ≠ 0`. -/
theorem linearIndependent_of_isOrthoᵢ {B : V₁ →ₛₗ[I₁] V₁ →ₛₗ[I₁'] V} {v : n → V₁}
    (hv₁ : B.IsOrthoᵢ v) (hv₂ : ∀ i, ¬B.IsOrtho (v i) (v i)) : LinearIndependent K₁ v := by
  classical
    rw [linearIndependent_iff']
    intro s w hs i hi
    have : B (s.sum fun i : n ↦ w i • v i) (v i) = 0 := by rw [hs, map_zero, zero_apply]
    have hsum : (s.sum fun j : n ↦ I₁ (w j) • B (v j) (v i)) = I₁ (w i) • B (v i) (v i) := by
      apply Finset.sum_eq_single_of_mem i hi
      intro j _hj hij
      rw [isOrthoᵢ_def.1 hv₁ _ _ hij, smul_zero]
    simp_rw [B.map_sum₂, map_smulₛₗ₂, hsum] at this
    apply (map_eq_zero I₁).mp
    exact (smul_eq_zero.mp this).elim _root_.id (hv₂ i · |>.elim)


/-- The proposition that a sesquilinear map is reflexive -/
def IsRefl (B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₂] M) : Prop :=
  ∀ x y, B x y = 0 → B y x = 0


theorem eq_zero : ∀ {x y}, B x y = 0 → B y x = 0 := fun {x y} ↦ H x y


theorem eq_iff {x y} : B x y = 0 ↔ B y x = 0 := ⟨H x y, H y x⟩


theorem ortho_comm {x y} : IsOrtho B x y ↔ IsOrtho B y x :=
  ⟨eq_zero H, eq_zero H⟩


theorem domRestrict (p : Submodule R₁ M₁) : (B.domRestrict₁₂ p p).IsRefl :=
  fun _ _ ↦ by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    p : Submodule R₁ M₁
    x✝¹ x✝ : Subtype fun x => Membership.mem p x
    ⊢ Eq (((B.domRestrict₁₂ p p) x✝¹) x✝) 0 → Eq (((B.domRestrict₁₂ p p) x✝) x✝¹) 0
  -/
  simp_rw [domRestrict₁₂_apply]
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    p : Submodule R₁ M₁
    x✝¹ x✝ : Subtype fun x => Membership.mem p x
    ⊢ Eq ((B ↑x✝¹) ↑x✝) 0 → Eq ((B ↑x✝) ↑x✝¹) 0
  -/
  exact H _ _
  /-
    🎉 no goals
  -/

@[simp]
theorem flip_isRefl_iff : B.flip.IsRefl ↔ B.IsRefl :=
  ⟨fun h x y H ↦ h y x ((B.flip_apply _ _).trans H), fun h x y ↦ h y x⟩


theorem ker_flip_eq_bot (H : B.IsRefl) (h : LinearMap.ker B = ⊥) : LinearMap.ker B.flip = ⊥ := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    h : Eq (LinearMap.ker B) Bot.bot
    ⊢ Eq (LinearMap.ker B.flip) Bot.bot
  -/
  refine ker_eq_bot'.mpr fun _ hx ↦ ker_eq_bot'.mp h _ ?_
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    h : Eq (LinearMap.ker B) Bot.bot
    x✝ : M₁
    hx : Eq (B.flip x✝) 0
    ⊢ Eq (B x✝) 0
  -/
  ext
  /-
    case h
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    h : Eq (LinearMap.ker B) Bot.bot
    x✝¹ : M₁
    hx : Eq (B.flip x✝¹) 0
    x✝ : M₁
    ⊢ Eq ((B x✝¹) x✝) (0 x✝)
  -/
  exact H _ _ (LinearMap.congr_fun hx _)
  /-
    🎉 no goals
  -/


theorem ker_eq_bot_iff_ker_flip_eq_bot (H : B.IsRefl) :
    LinearMap.ker B = ⊥ ↔ LinearMap.ker B.flip = ⊥ := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    ⊢ Iff (Eq (LinearMap.ker B) Bot.bot) (Eq (LinearMap.ker B.flip) Bot.bot)
  -/
  refine ⟨ker_flip_eq_bot H, fun h ↦ ?_⟩
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsRefl
    h : Eq (LinearMap.ker B.flip) Bot.bot
    ⊢ Eq (LinearMap.ker B) Bot.bot
  -/
  exact (congr_arg _ B.flip_flip.symm).trans (ker_flip_eq_bot (flip_isRefl_iff.mpr H) h)
  /-
    🎉 no goals
  -/


/-- The proposition that a sesquilinear form is symmetric -/
def IsSymm (B : M →ₛₗ[I] M →ₗ[R] R) : Prop :=
  ∀ x y, I (B x y) = B y x


protected theorem eq (H : B.IsSymm) (x y) : I (B x y) = B y x :=
  H x y


theorem isRefl (H : B.IsSymm) : B.IsRefl := fun x y H1 ↦ by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : RingHom R R
    B : LinearMap I M (LinearMap (RingHom.id R) M R)
    H : B.IsSymm
    x y : M
    H1 : Eq ((B x) y) 0
    ⊢ Eq ((B y) x) 0
  -/
  rw [← H.eq]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : RingHom R R
    B : LinearMap I M (LinearMap (RingHom.id R) M R)
    H : B.IsSymm
    x y : M
    H1 : Eq ((B x) y) 0
    ⊢ Eq (I ((B x) y)) 0
  -/
  simp [H1]
  /-
    🎉 no goals
  -/


theorem ortho_comm (H : B.IsSymm) {x y} : IsOrtho B x y ↔ IsOrtho B y x :=
  H.isRefl.ortho_comm


theorem domRestrict (H : B.IsSymm) (p : Submodule R M) : (B.domRestrict₁₂ p p).IsSymm :=
  fun _ _ ↦ by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : RingHom R R
    B : LinearMap I M (LinearMap (RingHom.id R) M R)
    H : B.IsSymm
    p : Submodule R M
    x✝¹ x✝ : Subtype fun x => Membership.mem p x
    ⊢ Eq (I (((B.domRestrict₁₂ p p) x✝¹) x✝)) (((B.domRestrict₁₂ p p) x✝) x✝¹)
  -/
  simp_rw [domRestrict₁₂_apply]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : RingHom R R
    B : LinearMap I M (LinearMap (RingHom.id R) M R)
    H : B.IsSymm
    p : Submodule R M
    x✝¹ x✝ : Subtype fun x => Membership.mem p x
    ⊢ Eq (I ((B ↑x✝¹) ↑x✝)) ((B ↑x✝) ↑x✝¹)
  -/
  exact H _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem isSymm_zero : (0 : M →ₛₗ[I] M →ₗ[R] R).IsSymm := fun _ _ => map_zero _


theorem isSymm_iff_eq_flip {B : LinearMap.BilinForm R M} : B.IsSymm ↔ B = B.flip := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ⊢ Iff (LinearMap.IsSymm B) (Eq B (LinearMap.flip B))
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      M : Type u_5
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : LinearMap.IsSymm B
      ⊢ Eq B (LinearMap.flip B)
    -/
  · ext
    /-
      case mp.h.h
      R : Type u_1
      M : Type u_5
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : LinearMap.IsSymm B
      x✝¹ x✝ : M
      ⊢ Eq ((B x✝¹) x✝) (((LinearMap.flip B) x✝¹) x✝)
    -/
    rw [← h, flip_apply, RingHom.id_apply]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    h : Eq B (LinearMap.flip B)
    ⊢ LinearMap.IsSymm B
  -/
  intro x y
  /-
    case mpr
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    h : Eq B (LinearMap.flip B)
    x y : M
    ⊢ Eq ((RingHom.id R) ((B x) y)) ((B y) x)
  -/
  conv_lhs => rw [h]
  /-
    case mpr
    R : Type u_1
    M : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    h : Eq B (LinearMap.flip B)
    x y : M
    ⊢ Eq ((RingHom.id R) (((LinearMap.flip B) x) y)) ((B y) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The proposition that a sesquilinear map is alternating -/
def IsAlt (B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₂] M) : Prop :=
  ∀ x, B x x = 0


theorem IsAlt.self_eq_zero (x : M₁) : B x x = 0 :=
  H x


theorem IsAlt.eq_of_add_add_eq_zero [IsCancelAdd M] {a b c : M₁} (hAdd : a + b + c = 0) :
    B a b = B b c := by
  have : B a a + B a b + B a c = B a c + B b c + B c c := by
    simp_rw [← map_add, ← map_add₂, hAdd, map_zero, LinearMap.zero_apply]
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : CommSemiring R₁
    inst✝² : AddCommMonoid M₁
    inst✝¹ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    inst✝ : IsCancelAdd M
    a b c : M₁
    hAdd : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
    this : Eq (HAdd.hAdd (HAdd.hAdd ((B a) a) ((B a) b)) ((B a) c)) (HAdd.hAdd (HA …
    ⊢ Eq ((B a) b) ((B b) c)
  -/
  rw [H, H, zero_add, add_zero, add_comm] at this
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : CommSemiring R₁
    inst✝² : AddCommMonoid M₁
    inst✝¹ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    inst✝ : IsCancelAdd M
    a b c : M₁
    hAdd : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
    this : Eq (HAdd.hAdd ((B a) c) ((B a) b)) (HAdd.hAdd ((B a) c) ((B b) c))
    ⊢ Eq ((B a) b) ((B b) c)
  -/
  exact add_left_cancel this
  /-
    🎉 no goals
  -/


theorem neg (H : B.IsAlt) (x y : M₁) : -B x y = B y x := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    x y : M₁
    ⊢ Eq (Neg.neg ((B x) y)) ((B y) x)
  -/
  have H1 : B (y + x) (y + x) = 0 := self_eq_zero H (y + x)
  simp? [map_add, self_eq_zero H] at H1 says
    simp only [map_add, add_apply, self_eq_zero H, zero_add, add_zero] at H1
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    x y : M₁
    H1 : Eq (HAdd.hAdd ((B x) y) ((B y) x)) 0
    ⊢ Eq (Neg.neg ((B x) y)) ((B y) x)
  -/
  rw [add_eq_zero_iff_neg_eq] at H1
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    x y : M₁
    H1 : Eq (Neg.neg ((B x) y)) ((B y) x)
    ⊢ Eq (Neg.neg ((B x) y)) ((B y) x)
  -/
  exact H1
  /-
    🎉 no goals
  -/


theorem isRefl (H : B.IsAlt) : B.IsRefl := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    ⊢ B.IsRefl
  -/
  intro x y h
  /-
    R : Type u_1
    R₁ : Type u_2
    M : Type u_5
    M₁ : Type u_6
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : CommSemiring R₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    I₁ I₂ : RingHom R₁ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
    H : B.IsAlt
    x y : M₁
    h : Eq ((B x) y) 0
    ⊢ Eq ((B y) x) 0
  -/
  rw [← neg H, h, neg_zero]
  /-
    🎉 no goals
  -/


theorem ortho_comm (H : B.IsAlt) {x y} : IsOrtho B x y ↔ IsOrtho B y x :=
  H.isRefl.ortho_comm


theorem isAlt_iff_eq_neg_flip [NoZeroDivisors R] [CharZero R] {B : M₁ →ₛₗ[I] M₁ →ₛₗ[I] R} :
    B.IsAlt ↔ B = -B.flip := by
  /-
    R : Type u_1
    R₁ : Type u_2
    M₁ : Type u_6
    inst✝⁵ : CommRing R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    I : RingHom R₁ R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    B : LinearMap I M₁ (LinearMap I M₁ R)
    ⊢ Iff B.IsAlt (Eq B (Neg.neg B.flip))
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      R₁ : Type u_2
      M₁ : Type u_6
      inst✝⁵ : CommRing R
      inst✝⁴ : CommSemiring R₁
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R₁ M₁
      I : RingHom R₁ R
      inst✝¹ : NoZeroDivisors R
      inst✝ : CharZero R
      B : LinearMap I M₁ (LinearMap I M₁ R)
      h : B.IsAlt
      ⊢ Eq B (Neg.neg B.flip)
    -/
  · ext
    /-
      case mp.h.h
      R : Type u_1
      R₁ : Type u_2
      M₁ : Type u_6
      inst✝⁵ : CommRing R
      inst✝⁴ : CommSemiring R₁
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R₁ M₁
      I : RingHom R₁ R
      inst✝¹ : NoZeroDivisors R
      inst✝ : CharZero R
      B : LinearMap I M₁ (LinearMap I M₁ R)
      h : B.IsAlt
      x✝¹ x✝ : M₁
      ⊢ Eq ((B x✝¹) x✝) (((Neg.neg B.flip) x✝¹) x✝)
    -/
    simp_rw [neg_apply, flip_apply]
    /-
      case mp.h.h
      R : Type u_1
      R₁ : Type u_2
      M₁ : Type u_6
      inst✝⁵ : CommRing R
      inst✝⁴ : CommSemiring R₁
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R₁ M₁
      I : RingHom R₁ R
      inst✝¹ : NoZeroDivisors R
      inst✝ : CharZero R
      B : LinearMap I M₁ (LinearMap I M₁ R)
      h : B.IsAlt
      x✝¹ x✝ : M₁
      ⊢ Eq ((B x✝¹) x✝) (Neg.neg ((B x✝) x✝¹))
    -/
    exact (h.neg _ _).symm
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    M₁ : Type u_6
    inst✝⁵ : CommRing R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    I : RingHom R₁ R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    B : LinearMap I M₁ (LinearMap I M₁ R)
    h : Eq B (Neg.neg B.flip)
    ⊢ B.IsAlt
  -/
  intro x
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    M₁ : Type u_6
    inst✝⁵ : CommRing R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    I : RingHom R₁ R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    B : LinearMap I M₁ (LinearMap I M₁ R)
    h : Eq B (Neg.neg B.flip)
    x : M₁
    ⊢ Eq ((B x) x) 0
  -/
  let h' := congr_fun₂ h x x
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    M₁ : Type u_6
    inst✝⁵ : CommRing R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    I : RingHom R₁ R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    B : LinearMap I M₁ (LinearMap I M₁ R)
    h : Eq B (Neg.neg B.flip)
    x : M₁
    h' : Eq ((B x) x) (((Neg.neg B.flip) x) x) := LinearMap.congr_fun₂ h x x
    ⊢ Eq ((B x) x) 0
  -/
  simp only [neg_apply, flip_apply, ← add_eq_zero_iff_eq_neg] at h'
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    M₁ : Type u_6
    inst✝⁵ : CommRing R
    inst✝⁴ : CommSemiring R₁
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R₁ M₁
    I : RingHom R₁ R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    B : LinearMap I M₁ (LinearMap I M₁ R)
    h : Eq B (Neg.neg B.flip)
    x : M₁
    h' : Eq (HAdd.hAdd ((B x) x) ((B x) x)) 0
    ⊢ Eq ((B x) x) 0
  -/
  exact add_self_eq_zero.mp h'
  /-
    🎉 no goals
  -/


/-- The orthogonal complement of a submodule `N` with respect to some bilinear map is the set of
elements `x` which are orthogonal to all elements of `N`; i.e., for all `y` in `N`, `B x y = 0`.

Note that for general (neither symmetric nor antisymmetric) bilinear maps this definition has a
chirality; in addition to this "left" orthogonal complement one could define a "right" orthogonal
complement for which, for all `y` in `N`, `B y x = 0`.  This variant definition is not currently
provided in mathlib. -/
def orthogonalBilin (N : Submodule R₁ M₁) (B : M₁ →ₛₗ[I₁] M₁ →ₛₗ[I₂] M) : Submodule R₁ M₁ where
  carrier := { m | ∀ n ∈ N, B.IsOrtho n m }
  zero_mem' x _ := B.isOrtho_zero_right x
  add_mem' hx hy n hn := by
    rw [LinearMap.IsOrtho, map_add, show B n _ = 0 from hx n hn, show B n _ = 0 from hy n hn,
      zero_add]
  smul_mem' c x hx n hn := by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      Mₗ₁ : Type u_9
      Mₗ₁' : Type u_10
      Mₗ₂ : Type u_11
      Mₗ₂' : Type u_12
      K : Type u_13
      K₁ : Type u_14
      K₂ : Type u_15
      V : Type u_16
      V₁ : Type u_17
      V₂ : Type u_18
      n✝ : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing R₁
      inst✝³ : AddCommGroup M₁
      inst✝² : Module R₁ M₁
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I₁ I₂ : RingHom R₁ R
      B✝ : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
      N : Submodule R₁ M₁
      B : LinearMap I₁ M₁ (LinearMap I₂ M₁ M)
      c : R₁
      x : M₁
      hx : Membership.mem { carrier := setOf fun m => ∀ (n : M₁), Membership.mem N n …
      n : M₁
      hn : Membership.mem N n
      ⊢ B.IsOrtho n (HSMul.hSMul c x)
    -/
    rw [LinearMap.IsOrtho, LinearMap.map_smulₛₗ, show B n x = 0 from hx n hn, smul_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_orthogonalBilin_iff {m : M₁} : m ∈ N.orthogonalBilin B ↔ ∀ n ∈ N, B.IsOrtho n m :=
  Iff.rfl


theorem orthogonalBilin_le (h : N ≤ L) : L.orthogonalBilin B ≤ N.orthogonalBilin B :=
  fun _ hn l hl ↦ hn l (h hl)


theorem le_orthogonalBilin_orthogonalBilin (b : B.IsRefl) :
    N ≤ (N.orthogonalBilin B).orthogonalBilin B := fun n hn _m hm ↦ b _ _ (hm n hn)


theorem span_singleton_inf_orthogonal_eq_bot (B : V₁ →ₛₗ[J₁] V₁ →ₛₗ[J₁'] V₂) (x : V₁)
    (hx : ¬B.IsOrtho x x) : (K₁ ∙ x) ⊓ Submodule.orthogonalBilin (K₁ ∙ x) B = ⊥ := by
  /-
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Min.min (Submodule.span K₁ (Singleton.singleton x)) ((Submodule.span K₁  …
  -/
  rw [← Finset.coe_singleton]
  /-
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Min.min (Submodule.span K₁ ↑(Singleton.singleton x)) ((Submodule.span K₁ …
  -/
  refine eq_bot_iff.2 fun y h ↦ ?_
  /-
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    y : V₁
    h : Membership.mem (Min.min (Submodule.span K₁ ↑(Singleton.singleton x)) ((Sub …
    ⊢ Membership.mem Bot.bot y
  -/
  rcases mem_span_finset.1 h.1 with ⟨μ, rfl⟩
  /-
    case intro
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    μ : V₁ → K₁
    h : Membership.mem (Min.min (Submodule.span K₁ ↑(Singleton.singleton x)) ((Sub …
    ⊢ Membership.mem Bot.bot ((Singleton.singleton x).sum fun i => HSMul.hSMul (μ  …
  -/
  replace h := h.2 x (by simp [Submodule.mem_span] : x ∈ Submodule.span K₁ ({x} : Finset V₁))
  /-
    case intro
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    μ : V₁ → K₁
    h : B.IsOrtho x ((Singleton.singleton x).sum fun i => HSMul.hSMul (μ i) i)
    ⊢ Membership.mem Bot.bot ((Singleton.singleton x).sum fun i => HSMul.hSMul (μ  …
  -/
  rw [Finset.sum_singleton] at h ⊢
  /-
    case intro
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    μ : V₁ → K₁
    h : B.IsOrtho x (HSMul.hSMul (μ x) x)
    ⊢ Membership.mem Bot.bot (HSMul.hSMul (μ x) x)
  -/
  suffices hμzero : μ x = 0 by rw [hμzero, zero_smul, Submodule.mem_bot]
  /-
    case intro
    K : Type u_13
    K₁ : Type u_14
    V₁ : Type u_17
    V₂ : Type u_18
    inst✝⁵ : Field K
    inst✝⁴ : Field K₁
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K₁ V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J₁ J₁' : RingHom K₁ K
    B : LinearMap J₁ V₁ (LinearMap J₁' V₁ V₂)
    x : V₁
    hx : Not (B.IsOrtho x x)
    μ : V₁ → K₁
    h : B.IsOrtho x (HSMul.hSMul (μ x) x)
    ⊢ Eq (μ x) 0
  -/
  rw [isOrtho_def, map_smulₛₗ] at h
  exact Or.elim (smul_eq_zero.mp h)
      (fun y ↦ by simpa using y)
      (fun hfalse ↦ False.elim <| hx hfalse)

-- ↓ This lemma only applies in fields since we use the `mul_eq_zero`

theorem orthogonal_span_singleton_eq_to_lin_ker {B : V →ₗ[K] V →ₛₗ[J] V₂} (x : V) :
    Submodule.orthogonalBilin (K ∙ x) B = LinearMap.ker (B x) := by
  /-
    K : Type u_13
    V : Type u_16
    V₂ : Type u_18
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J : RingHom K K
    B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
    x : V
    ⊢ Eq ((Submodule.span K (Singleton.singleton x)).orthogonalBilin B) (LinearMap …
  -/
  ext y
  /-
    case h
    K : Type u_13
    V : Type u_16
    V₂ : Type u_18
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J : RingHom K K
    B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
    x y : V
    ⊢ Iff (Membership.mem ((Submodule.span K (Singleton.singleton x)).orthogonalBi …
  -/
  simp_rw [Submodule.mem_orthogonalBilin_iff, LinearMap.mem_ker, Submodule.mem_span_singleton]
  /-
    case h
    K : Type u_13
    V : Type u_16
    V₂ : Type u_18
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    J : RingHom K K
    B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
    x y : V
    ⊢ Iff (∀ (n : V), (Exists fun a => Eq (HSMul.hSMul a x) n) → B.IsOrtho n y) (E …
  -/
  constructor
    /-
      case h.mp
      K : Type u_13
      V : Type u_16
      V₂ : Type u_18
      inst✝⁴ : Field K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K V₂
      J : RingHom K K
      B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
      x y : V
      ⊢ (∀ (n : V), (Exists fun a => Eq (HSMul.hSMul a x) n) → B.IsOrtho n y) → Eq ( …
    -/
  · exact fun h ↦ h x ⟨1, one_smul _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      K : Type u_13
      V : Type u_16
      V₂ : Type u_18
      inst✝⁴ : Field K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K V₂
      J : RingHom K K
      B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
      x y : V
      ⊢ Eq ((B x) y) 0 → ∀ (n : V), (Exists fun a => Eq (HSMul.hSMul a x) n) → B.IsO …
    -/
  · rintro h _ ⟨z, rfl⟩
    /-
      case h.mpr.intro
      K : Type u_13
      V : Type u_16
      V₂ : Type u_18
      inst✝⁴ : Field K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K V₂
      J : RingHom K K
      B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
      x y : V
      h : Eq ((B x) y) 0
      z : K
      ⊢ B.IsOrtho (HSMul.hSMul z x) y
    -/
    rw [isOrtho_def, map_smulₛₗ₂, smul_eq_zero]
    /-
      case h.mpr.intro
      K : Type u_13
      V : Type u_16
      V₂ : Type u_18
      inst✝⁴ : Field K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V₂
      inst✝ : Module K V₂
      J : RingHom K K
      B : LinearMap (RingHom.id K) V (LinearMap J V V₂)
      x y : V
      h : Eq ((B x) y) 0
      z : K
      ⊢ Or (Eq ((RingHom.id K) z) 0) (Eq ((B x) y) 0)
    -/
    exact Or.intro_right _ h
    /-
      🎉 no goals
    -/

-- todo: Generalize this to sesquilinear maps

theorem span_singleton_sup_orthogonal_eq_top {B : V →ₗ[K] V →ₗ[K] K} {x : V} (hx : ¬B.IsOrtho x x) :
    (K ∙ x) ⊔ Submodule.orthogonalBilin (N := K ∙ x) (B := B) = ⊤ := by
  /-
    K : Type u_13
    V : Type u_16
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) V K)
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Max.max (Submodule.span K (Singleton.singleton x)) ((Submodule.span K (S …
  -/
  rw [orthogonal_span_singleton_eq_to_lin_ker]
  /-
    K : Type u_13
    V : Type u_16
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) V K)
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Max.max (Submodule.span K (Singleton.singleton x)) (LinearMap.ker (B x)) …
  -/
  exact (B x).span_singleton_sup_ker_eq_top hx
  /-
    🎉 no goals
  -/

-- todo: Generalize this to sesquilinear maps

/-- Given a bilinear form `B` and some `x` such that `B x x ≠ 0`, the span of the singleton of `x`
  is complement to its orthogonal complement. -/
theorem isCompl_span_singleton_orthogonal {B : V →ₗ[K] V →ₗ[K] K} {x : V} (hx : ¬B.IsOrtho x x) :
    IsCompl (K ∙ x) (Submodule.orthogonalBilin (N := K ∙ x) (B := B)) :=
  { disjoint := disjoint_iff.2 <| span_singleton_inf_orthogonal_eq_bot B x hx
    codisjoint := codisjoint_iff.2 <| span_singleton_sup_orthogonal_eq_top hx }


/-- Given a pair of modules equipped with bilinear maps, this is the condition for a pair of
maps between them to be mutually adjoint. -/
def IsAdjointPair (f : M → M₁) (g : M₁ → M) :=
  ∀ x y, B' (f x) y = B x (g y)


theorem isAdjointPair_iff_comp_eq_compl₂ : IsAdjointPair B B' f g ↔ B'.comp f = B.compl₂ g := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₃ : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    I : RingHom R R
    B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
    B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
    f : LinearMap (RingHom.id R) M M₁
    g : LinearMap (RingHom.id R) M₁ M
    ⊢ Iff (B.IsAdjointPair B' ⇑f ⇑g) (Eq (B'.comp f) (B.compl₂ g))
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      M₃ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R M₁
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      I : RingHom R R
      B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
      B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
      f : LinearMap (RingHom.id R) M M₁
      g : LinearMap (RingHom.id R) M₁ M
      h : B.IsAdjointPair B' ⇑f ⇑g
      ⊢ Eq (B'.comp f) (B.compl₂ g)
    -/
  · ext x y
    /-
      case mp.h.h
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      M₃ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R M₁
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      I : RingHom R R
      B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
      B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
      f : LinearMap (RingHom.id R) M M₁
      g : LinearMap (RingHom.id R) M₁ M
      h : B.IsAdjointPair B' ⇑f ⇑g
      x : M
      y : M₁
      ⊢ Eq (((B'.comp f) x) y) (((B.compl₂ g) x) y)
    -/
    rw [comp_apply, compl₂_apply]
    /-
      case mp.h.h
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      M₃ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R M₁
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      I : RingHom R R
      B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
      B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
      f : LinearMap (RingHom.id R) M M₁
      g : LinearMap (RingHom.id R) M₁ M
      h : B.IsAdjointPair B' ⇑f ⇑g
      x : M
      y : M₁
      ⊢ Eq ((B' (f x)) y) ((B x) (g y))
    -/
    exact h x y
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      M₃ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R M₁
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      I : RingHom R R
      B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
      B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
      f : LinearMap (RingHom.id R) M M₁
      g : LinearMap (RingHom.id R) M₁ M
      h : Eq (B'.comp f) (B.compl₂ g)
      ⊢ B.IsAdjointPair B' ⇑f ⇑g
    -/
  · intro _ _
    /-
      case mpr
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      M₃ : Type u_8
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₁
      inst✝² : Module R M₁
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      I : RingHom R R
      B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
      B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
      f : LinearMap (RingHom.id R) M M₁
      g : LinearMap (RingHom.id R) M₁ M
      h : Eq (B'.comp f) (B.compl₂ g)
      x✝ : M
      y✝ : M₁
      ⊢ Eq ((B' (f x✝)) y✝) ((B x✝) (g y✝))
    -/
    rw [← compl₂_apply, ← comp_apply, h]
    /-
      🎉 no goals
    -/


theorem isAdjointPair_zero : IsAdjointPair B B' 0 0 := fun _ _ ↦ by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₃ : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    I : RingHom R R
    B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
    B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
    x✝¹ : M
    x✝ : M₁
    ⊢ Eq ((B' (0 x✝¹)) x✝) ((B x✝¹) (0 x✝))
  -/
  simp only [Pi.zero_apply, map_zero, zero_apply]
  /-
    🎉 no goals
  -/


theorem isAdjointPair_id : IsAdjointPair B B (_root_.id : M → M) (_root_.id : M → M) :=
  fun _ _ ↦ rfl


theorem isAdjointPair_one : IsAdjointPair B B (1 : Module.End R M) (1 : Module.End R M) :=
  isAdjointPair_id


theorem IsAdjointPair.add {f f' : M → M₁} {g g' : M₁ → M} (h : IsAdjointPair B B' f g)
    (h' : IsAdjointPair B B' f' g') :
    IsAdjointPair B B' (f + f') (g + g') := fun x _ ↦ by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₃ : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    I : RingHom R R
    B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
    B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
    f f' : M → M₁
    g g' : M₁ → M
    h : B.IsAdjointPair B' f g
    h' : B.IsAdjointPair B' f' g'
    x : M
    x✝ : M₁
    ⊢ Eq ((B' (HAdd.hAdd f f' x)) x✝) ((B x) (HAdd.hAdd g g' x✝))
  -/
  rw [Pi.add_apply, Pi.add_apply, B'.map_add₂, (B x).map_add, h, h']
  /-
    🎉 no goals
  -/


theorem IsAdjointPair.comp {f : M → M₁} {g : M₁ → M} {f' : M₁ → M₂} {g' : M₂ → M₁}
    (h : IsAdjointPair B B' f g) (h' : IsAdjointPair B' B'' f' g') :
    IsAdjointPair B B'' (f' ∘ f) (g ∘ g') := fun _ _ ↦ by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : Module R M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    I : RingHom R R
    B : LinearMap (RingHom.id R) M (LinearMap I M M₃)
    B' : LinearMap (RingHom.id R) M₁ (LinearMap I M₁ M₃)
    B'' : LinearMap (RingHom.id R) M₂ (LinearMap I M₂ M₃)
    f : M → M₁
    g : M₁ → M
    f' : M₁ → M₂
    g' : M₂ → M₁
    h : B.IsAdjointPair B' f g
    h' : B'.IsAdjointPair B'' f' g'
    x✝¹ : M
    x✝ : M₂
    ⊢ Eq ((B'' (Function.comp f' f x✝¹)) x✝) ((B x✝¹) (Function.comp g g' x✝))
  -/
  rw [Function.comp_def, Function.comp_def, h', h]
  /-
    🎉 no goals
  -/


theorem IsAdjointPair.mul {f g f' g' : Module.End R M} (h : IsAdjointPair B B f g)
    (h' : IsAdjointPair B B f' g') : IsAdjointPair B B (f * f') (g' * g) :=
  h'.comp h


theorem IsAdjointPair.sub (h : IsAdjointPair B B' f g) (h' : IsAdjointPair B B' f' g') :
    IsAdjointPair B B' (f - f') (g - g') := fun x _ ↦ by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
    B' : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₁ M₂)
    f f' : M → M₁
    g g' : M₁ → M
    h : B.IsAdjointPair B' f g
    h' : B.IsAdjointPair B' f' g'
    x : M
    x✝ : M₁
    ⊢ Eq ((B' (HSub.hSub f f' x)) x✝) ((B x) (HSub.hSub g g' x✝))
  -/
  rw [Pi.sub_apply, Pi.sub_apply, B'.map_sub₂, (B x).map_sub, h, h']
  /-
    🎉 no goals
  -/


theorem IsAdjointPair.smul (c : R) (h : IsAdjointPair B B' f g) :
    IsAdjointPair B B' (c • f) (c • g) := fun _ _ ↦ by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
    B' : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₁ M₂)
    f : M → M₁
    g : M₁ → M
    c : R
    h : B.IsAdjointPair B' f g
    x✝¹ : M
    x✝ : M₁
    ⊢ Eq ((B' (HSMul.hSMul c f x✝¹)) x✝) ((B x✝¹) (HSMul.hSMul c g x✝))
  -/
  simp [h _]
  /-
    🎉 no goals
  -/


/-- A linear transformation `f` is orthogonal with respect to a bilinear form `B` if `B` is
bi-invariant with respect to `f`. -/
def IsOrthogonal : Prop :=
  ∀ x y, B (f x) (f y) = B x y


@[simp]
lemma _root_.LinearEquiv.isAdjointPair_symm_iff {f : M ≃ M} :
    LinearMap.IsAdjointPair B B f f.symm ↔ B.IsOrthogonal f :=
                   /-
                     R : Type u_20
                     M : Type u_21
                     inst✝² : CommRing R
                     inst✝¹ : AddCommGroup M
                     inst✝ : Module R M
                     B : LinearMap.BilinForm R M
                     f : Equiv M M
                     hf : LinearMap.IsAdjointPair B B ⇑f ⇑f.symm
                     x y : M
                     ⊢ Eq ((B (f x)) (f y)) ((B x) y)
                   -/
                   /-
                     🎉 no goals
                   -/
  ⟨fun hf x y ↦ by simpa using hf x (f y), fun hf x y ↦ by simpa using hf x (f.symm y)⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma isOrthogonal_of_forall_apply_same {F : Type*} [FunLike F M M] [LinearMapClass F R M M]
    (f : F) (h : IsLeftRegular (2 : R)) (hB : B.IsSymm) (hf : ∀ x, B (f x) (f x) = B x x) :
    B.IsOrthogonal f := by
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    ⊢ LinearMap.IsOrthogonal B ⇑f
  -/
  intro x y
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    ⊢ Eq ((B (f x)) (f y)) ((B x) y)
  -/
  suffices 2 * B (f x) (f y) = 2 * B x y from h this
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    ⊢ Eq (HMul.hMul 2 ((B (f x)) (f y))) (HMul.hMul 2 ((B x) y))
  -/
  have := hf (x + y)
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    this : Eq ((B (f (HAdd.hAdd x y))) (f (HAdd.hAdd x y))) ((B (HAdd.hAdd x y)) ( …
    ⊢ Eq (HMul.hMul 2 ((B (f x)) (f y))) (HMul.hMul 2 ((B x) y))
  -/
  simp only [map_add, LinearMap.add_apply, hf x, hf y, show B y x = B x y from hB.eq y x] at this
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    this : Eq (HAdd.hAdd (HAdd.hAdd ((B x) x) ((B (f y)) (f x))) (HAdd.hAdd ((B (f …
    ⊢ Eq (HMul.hMul 2 ((B (f x)) (f y))) (HMul.hMul 2 ((B x) y))
  -/
  rw [show B (f y) (f x) = B (f x) (f y) from hB.eq (f y) (f x)] at this
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    this : Eq (HAdd.hAdd (HAdd.hAdd ((B x) x) ((B (f x)) (f y))) (HAdd.hAdd ((B (f …
    ⊢ Eq (HMul.hMul 2 ((B (f x)) (f y))) (HMul.hMul 2 ((B x) y))
  -/
  simp only [add_assoc, add_right_inj] at this
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    this : Eq (HAdd.hAdd ((B (f x)) (f y)) (HAdd.hAdd ((B (f x)) (f y)) ((B y) y)) …
    ⊢ Eq (HMul.hMul 2 ((B (f x)) (f y))) (HMul.hMul 2 ((B x) y))
  -/
  simp only [← add_assoc, add_left_inj] at this
  /-
    R : Type u_20
    M : Type u_21
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    B : LinearMap.BilinForm R M
    F : Type u_22
    inst✝¹ : FunLike F M M
    inst✝ : LinearMapClass F R M M
    f : F
    h : IsLeftRegular 2
    hB : LinearMap.IsSymm B
    hf : ∀ (x : M), Eq ((B (f x)) (f x)) ((B x) x)
    x y : M
    this : Eq (HAdd.hAdd ((B (f x)) (f y)) ((B (f x)) (f y))) (HAdd.hAdd ((B x) y) …
    ⊢ Eq (HMul.hMul 2 ((B (f x)) (f y))) (HMul.hMul 2 ((B x) y))
  -/
  simpa only [← two_mul] using this
  /-
    🎉 no goals
  -/


/-- The condition for an endomorphism to be "self-adjoint" with respect to a pair of bilinear maps
on the underlying module. In the case that these two maps are identical, this is the usual concept
of self adjointness. In the case that one of the maps is the negation of the other, this is the
usual concept of skew adjointness. -/
def IsPairSelfAdjoint (f : M → M) :=
  IsAdjointPair B F f f


/-- An endomorphism of a module is self-adjoint with respect to a bilinear map if it serves as an
adjoint for itself. -/
protected def IsSelfAdjoint (f : M → M) :=
  IsAdjointPair B B f f


/-- The set of pair-self-adjoint endomorphisms are a submodule of the type of all endomorphisms. -/
def isPairSelfAdjointSubmodule : Submodule R (Module.End R M) where
  carrier := { f | IsPairSelfAdjoint B F f }
  zero_mem' := isAdjointPair_zero
  add_mem' hf hg := hf.add hg
  smul_mem' c _ h := h.smul c


/-- An endomorphism of a module is skew-adjoint with respect to a bilinear map if its negation
serves as an adjoint. -/
def IsSkewAdjoint (f : M → M) :=
  IsAdjointPair B B f (-f)


/-- The set of self-adjoint endomorphisms of a module with bilinear map is a submodule. (In fact
it is a Jordan subalgebra.) -/
def selfAdjointSubmodule :=
  isPairSelfAdjointSubmodule B B


/-- The set of skew-adjoint endomorphisms of a module with bilinear map is a submodule. (In fact
it is a Lie subalgebra.) -/
def skewAdjointSubmodule :=
  isPairSelfAdjointSubmodule (-B) B


@[simp]
theorem mem_isPairSelfAdjointSubmodule (f : Module.End R M) :
    f ∈ isPairSelfAdjointSubmodule B F ↔ IsPairSelfAdjoint B F f :=
  Iff.rfl


theorem isPairSelfAdjoint_equiv (e : M₁ ≃ₗ[R] M) (f : Module.End R M) :
    IsPairSelfAdjoint B F f ↔
      IsPairSelfAdjoint (B.compl₁₂ e e) (F.compl₁₂ e e) (e.symm.conj f) := by
  have hₗ :
    (F.compl₁₂ (↑e : M₁ →ₗ[R] M) (↑e : M₁ →ₗ[R] M)).comp (e.symm.conj f) =
      (F.comp f).compl₁₂ (↑e : M₁ →ₗ[R] M) (↑e : M₁ →ₗ[R] M) := by
    ext
    simp only [LinearEquiv.symm_conj_apply, coe_comp, LinearEquiv.coe_coe, compl₁₂_apply,
      LinearEquiv.apply_symm_apply, Function.comp_apply]
  have hᵣ :
    (B.compl₁₂ (↑e : M₁ →ₗ[R] M) (↑e : M₁ →ₗ[R] M)).compl₂ (e.symm.conj f) =
      (B.compl₂ f).compl₁₂ (↑e : M₁ →ₗ[R] M) (↑e : M₁ →ₗ[R] M) := by
    ext
    simp only [LinearEquiv.symm_conj_apply, compl₂_apply, coe_comp, LinearEquiv.coe_coe,
      compl₁₂_apply, LinearEquiv.apply_symm_apply, Function.comp_apply]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    B F : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
    e : LinearEquiv (RingHom.id R) M₁ M
    f : Module.End R M
    hₗ : Eq ((F.compl₁₂ ↑e ↑e).comp (e.symm.conj f)) ((F.comp f).compl₁₂ ↑e ↑e)
    hᵣ : Eq ((B.compl₁₂ ↑e ↑e).compl₂ (e.symm.conj f)) ((B.compl₂ f).compl₁₂ ↑e ↑e)
    ⊢ Iff (B.IsPairSelfAdjoint F ⇑f) ((B.compl₁₂ ↑e ↑e).IsPairSelfAdjoint (F.compl …
  -/
  have he : Function.Surjective (⇑(↑e : M₁ →ₗ[R] M) : M₁ → M) := e.surjective
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    B F : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
    e : LinearEquiv (RingHom.id R) M₁ M
    f : Module.End R M
    hₗ : Eq ((F.compl₁₂ ↑e ↑e).comp (e.symm.conj f)) ((F.comp f).compl₁₂ ↑e ↑e)
    hᵣ : Eq ((B.compl₁₂ ↑e ↑e).compl₂ (e.symm.conj f)) ((B.compl₂ f).compl₁₂ ↑e ↑e)
    he : Function.Surjective ⇑↑e
    ⊢ Iff (B.IsPairSelfAdjoint F ⇑f) ((B.compl₁₂ ↑e ↑e).IsPairSelfAdjoint (F.compl …
  -/
  simp_rw [IsPairSelfAdjoint, isAdjointPair_iff_comp_eq_compl₂, hₗ, hᵣ, compl₁₂_inj he he]
  /-
    🎉 no goals
  -/


theorem isSkewAdjoint_iff_neg_self_adjoint (f : M → M) :
    B.IsSkewAdjoint f ↔ IsAdjointPair (-B) B f f :=
                                                                              /-
                                                                                R : Type u_1
                                                                                M : Type u_5
                                                                                M₂ : Type u_7
                                                                                inst✝⁴ : CommRing R
                                                                                inst✝³ : AddCommGroup M
                                                                                inst✝² : Module R M
                                                                                inst✝¹ : AddCommGroup M₂
                                                                                inst✝ : Module R M₂
                                                                                B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
                                                                                f : M → M
                                                                                ⊢ Iff (∀ (x y : M), Eq ((B (f x)) y) ((B x) (Neg.neg f y))) (∀ (x y : M), Eq ( …
                                                                              -/
  show (∀ x y, B (f x) y = B x ((-f) y)) ↔ ∀ x y, B (f x) y = (-B) x (f y) by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem mem_selfAdjointSubmodule (f : Module.End R M) :
    f ∈ B.selfAdjointSubmodule ↔ B.IsSelfAdjoint f :=
  Iff.rfl


@[simp]
theorem mem_skewAdjointSubmodule (f : Module.End R M) :
    f ∈ B.skewAdjointSubmodule ↔ B.IsSkewAdjoint f := by
  /-
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
    f : Module.End R M
    ⊢ Iff (Membership.mem B.skewAdjointSubmodule f) (B.IsSkewAdjoint ⇑f)
  -/
  rw [isSkewAdjoint_iff_neg_self_adjoint]
  /-
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₂)
    f : Module.End R M
    ⊢ Iff (Membership.mem B.skewAdjointSubmodule f) ((Neg.neg B).IsAdjointPair B ⇑ …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


/-- A bilinear map is called left-separating if
the only element that is left-orthogonal to every other element is `0`; i.e.,
for every nonzero `x` in `M₁`, there exists `y` in `M₂` with `B x y ≠ 0`. -/
def SeparatingLeft (B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M) : Prop :=
  ∀ x : M₁, (∀ y : M₂, B x y = 0) → x = 0


/-- In a non-trivial module, zero is not non-degenerate. -/
theorem not_separatingLeft_zero [Nontrivial M₁] : ¬(0 : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M).SeparatingLeft :=
  let ⟨m, hm⟩ := exists_ne (0 : M₁)
  fun h ↦ hm (h m fun _n ↦ rfl)


theorem SeparatingLeft.ne_zero [Nontrivial M₁] {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M}
    (h : B.SeparatingLeft) : B ≠ 0 := fun h0 ↦ not_separatingLeft_zero M₁ M₂ I₁ I₂ <| h0 ▸ h


theorem SeparatingLeft.congr (h : B.SeparatingLeft) :
    (e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M)) B).SeparatingLeft := by
  /-
    R : Type u_1
    M : Type u_5
    Mₗ₁ : Type u_9
    Mₗ₁' : Type u_10
    Mₗ₂ : Type u_11
    Mₗ₂' : Type u_12
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid Mₗ₁
    inst✝⁶ : AddCommMonoid Mₗ₂
    inst✝⁵ : AddCommMonoid Mₗ₁'
    inst✝⁴ : AddCommMonoid Mₗ₂'
    inst✝³ : Module R Mₗ₁
    inst✝² : Module R Mₗ₂
    inst✝¹ : Module R Mₗ₁'
    inst✝ : Module R Mₗ₂'
    B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
    e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
    e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
    h : B.SeparatingLeft
    ⊢ ((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M))) B).SeparatingLeft
  -/
  intro x hx
  /-
    R : Type u_1
    M : Type u_5
    Mₗ₁ : Type u_9
    Mₗ₁' : Type u_10
    Mₗ₂ : Type u_11
    Mₗ₂' : Type u_12
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid Mₗ₁
    inst✝⁶ : AddCommMonoid Mₗ₂
    inst✝⁵ : AddCommMonoid Mₗ₁'
    inst✝⁴ : AddCommMonoid Mₗ₂'
    inst✝³ : Module R Mₗ₁
    inst✝² : Module R Mₗ₂
    inst✝¹ : Module R Mₗ₁'
    inst✝ : Module R Mₗ₂'
    B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
    e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
    e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
    h : B.SeparatingLeft
    x : Mₗ₁'
    hx : ∀ (y : Mₗ₂'), Eq ((((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M)) …
    ⊢ Eq x 0
  -/
  rw [← e₁.symm.map_eq_zero_iff]
  /-
    R : Type u_1
    M : Type u_5
    Mₗ₁ : Type u_9
    Mₗ₁' : Type u_10
    Mₗ₂ : Type u_11
    Mₗ₂' : Type u_12
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid Mₗ₁
    inst✝⁶ : AddCommMonoid Mₗ₂
    inst✝⁵ : AddCommMonoid Mₗ₁'
    inst✝⁴ : AddCommMonoid Mₗ₂'
    inst✝³ : Module R Mₗ₁
    inst✝² : Module R Mₗ₂
    inst✝¹ : Module R Mₗ₁'
    inst✝ : Module R Mₗ₂'
    B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
    e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
    e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
    h : B.SeparatingLeft
    x : Mₗ₁'
    hx : ∀ (y : Mₗ₂'), Eq ((((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M)) …
    ⊢ Eq (e₁.symm x) 0
  -/
  refine h (e₁.symm x) fun y ↦ ?_
  /-
    R : Type u_1
    M : Type u_5
    Mₗ₁ : Type u_9
    Mₗ₁' : Type u_10
    Mₗ₂ : Type u_11
    Mₗ₂' : Type u_12
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid Mₗ₁
    inst✝⁶ : AddCommMonoid Mₗ₂
    inst✝⁵ : AddCommMonoid Mₗ₁'
    inst✝⁴ : AddCommMonoid Mₗ₂'
    inst✝³ : Module R Mₗ₁
    inst✝² : Module R Mₗ₂
    inst✝¹ : Module R Mₗ₁'
    inst✝ : Module R Mₗ₂'
    B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
    e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
    e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
    h : B.SeparatingLeft
    x : Mₗ₁'
    hx : ∀ (y : Mₗ₂'), Eq ((((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M)) …
    y : Mₗ₂
    ⊢ Eq ((B (e₁.symm x)) y) 0
  -/
  specialize hx (e₂ y)
  simp only [LinearEquiv.arrowCongr_apply, LinearEquiv.symm_apply_apply,
    LinearEquiv.map_eq_zero_iff] at hx
  /-
    R : Type u_1
    M : Type u_5
    Mₗ₁ : Type u_9
    Mₗ₁' : Type u_10
    Mₗ₂ : Type u_11
    Mₗ₂' : Type u_12
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid Mₗ₁
    inst✝⁶ : AddCommMonoid Mₗ₂
    inst✝⁵ : AddCommMonoid Mₗ₁'
    inst✝⁴ : AddCommMonoid Mₗ₂'
    inst✝³ : Module R Mₗ₁
    inst✝² : Module R Mₗ₂
    inst✝¹ : Module R Mₗ₁'
    inst✝ : Module R Mₗ₂'
    B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
    e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
    e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
    h : B.SeparatingLeft
    x : Mₗ₁'
    y : Mₗ₂
    hx : Eq ((B (e₁.symm x)) y) 0
    ⊢ Eq ((B (e₁.symm x)) y) 0
  -/
  exact hx
  /-
    🎉 no goals
  -/


@[simp]
theorem separatingLeft_congr_iff :
    (e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M)) B).SeparatingLeft ↔ B.SeparatingLeft :=
  ⟨fun h ↦ by
    /-
      R : Type u_1
      M : Type u_5
      Mₗ₁ : Type u_9
      Mₗ₁' : Type u_10
      Mₗ₂ : Type u_11
      Mₗ₂' : Type u_12
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid Mₗ₁
      inst✝⁶ : AddCommMonoid Mₗ₂
      inst✝⁵ : AddCommMonoid Mₗ₁'
      inst✝⁴ : AddCommMonoid Mₗ₂'
      inst✝³ : Module R Mₗ₁
      inst✝² : Module R Mₗ₂
      inst✝¹ : Module R Mₗ₁'
      inst✝ : Module R Mₗ₂'
      B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
      e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
      e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
      h : ((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M))) B).SeparatingLeft
      ⊢ B.SeparatingLeft
    -/
    convert h.congr e₁.symm e₂.symm
    /-
      case h.e'_18
      R : Type u_1
      M : Type u_5
      Mₗ₁ : Type u_9
      Mₗ₁' : Type u_10
      Mₗ₂ : Type u_11
      Mₗ₂' : Type u_12
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid Mₗ₁
      inst✝⁶ : AddCommMonoid Mₗ₂
      inst✝⁵ : AddCommMonoid Mₗ₁'
      inst✝⁴ : AddCommMonoid Mₗ₂'
      inst✝³ : Module R Mₗ₁
      inst✝² : Module R Mₗ₂
      inst✝¹ : Module R Mₗ₁'
      inst✝ : Module R Mₗ₂'
      B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
      e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
      e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
      h : ((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M))) B).SeparatingLeft
      ⊢ Eq B ((e₁.symm.arrowCongr (e₂.symm.arrowCongr (LinearEquiv.refl R M))) ((e₁. …
    -/
    ext x y
    /-
      case h.e'_18.h.h
      R : Type u_1
      M : Type u_5
      Mₗ₁ : Type u_9
      Mₗ₁' : Type u_10
      Mₗ₂ : Type u_11
      Mₗ₂' : Type u_12
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid Mₗ₁
      inst✝⁶ : AddCommMonoid Mₗ₂
      inst✝⁵ : AddCommMonoid Mₗ₁'
      inst✝⁴ : AddCommMonoid Mₗ₂'
      inst✝³ : Module R Mₗ₁
      inst✝² : Module R Mₗ₂
      inst✝¹ : Module R Mₗ₁'
      inst✝ : Module R Mₗ₂'
      B : LinearMap (RingHom.id R) Mₗ₁ (LinearMap (RingHom.id R) Mₗ₂ M)
      e₁ : LinearEquiv (RingHom.id R) Mₗ₁ Mₗ₁'
      e₂ : LinearEquiv (RingHom.id R) Mₗ₂ Mₗ₂'
      h : ((e₁.arrowCongr (e₂.arrowCongr (LinearEquiv.refl R M))) B).SeparatingLeft
      x : Mₗ₁
      y : Mₗ₂
      ⊢ Eq ((B x) y) ((((e₁.symm.arrowCongr (e₂.symm.arrowCongr (LinearEquiv.refl R  …
    -/
    simp,
    /-
      🎉 no goals
    -/
   SeparatingLeft.congr e₁ e₂⟩


/-- A bilinear map is called right-separating if
the only element that is right-orthogonal to every other element is `0`; i.e.,
for every nonzero `y` in `M₂`, there exists `x` in `M₁` with `B x y ≠ 0`. -/
def SeparatingRight (B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M) : Prop :=
  ∀ y : M₂, (∀ x : M₁, B x y = 0) → y = 0


/-- A bilinear map is called non-degenerate if it is left-separating and right-separating. -/
def Nondegenerate (B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M) : Prop :=
  SeparatingLeft B ∧ SeparatingRight B


@[simp]
theorem flip_separatingRight {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} :
    B.flip.SeparatingRight ↔ B.SeparatingLeft :=
  ⟨fun hB x hy ↦ hB x hy, fun hB x hy ↦ hB x hy⟩


@[simp]
theorem flip_separatingLeft {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} :
                                                    /-
                                                      R : Type u_1
                                                      R₁ : Type u_2
                                                      R₂ : Type u_3
                                                      M : Type u_5
                                                      M₁ : Type u_6
                                                      M₂ : Type u_7
                                                      inst✝⁸ : CommSemiring R
                                                      inst✝⁷ : AddCommMonoid M
                                                      inst✝⁶ : Module R M
                                                      inst✝⁵ : CommSemiring R₁
                                                      inst✝⁴ : AddCommMonoid M₁
                                                      inst✝³ : Module R₁ M₁
                                                      inst✝² : CommSemiring R₂
                                                      inst✝¹ : AddCommMonoid M₂
                                                      inst✝ : Module R₂ M₂
                                                      I₁ : RingHom R₁ R
                                                      I₂ : RingHom R₂ R
                                                      B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
                                                      ⊢ Iff B.flip.SeparatingLeft B.SeparatingRight
                                                    -/
    B.flip.SeparatingLeft ↔ SeparatingRight B := by rw [← flip_separatingRight, flip_flip]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem flip_nondegenerate {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} : B.flip.Nondegenerate ↔ B.Nondegenerate :=
  Iff.trans and_comm (and_congr flip_separatingRight flip_separatingLeft)


theorem separatingLeft_iff_linear_nontrivial {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} :
    B.SeparatingLeft ↔ ∀ x : M₁, B x = 0 → x = 0 := by
  /-
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : CommSemiring R₁
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : Module R₁ M₁
    inst✝² : CommSemiring R₂
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    I₁ : RingHom R₁ R
    I₂ : RingHom R₂ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
    ⊢ Iff B.SeparatingLeft (∀ (x : M₁), Eq (B x) 0 → Eq x 0)
  -/
  constructor <;> intro h x hB
    /-
      case mp
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : CommSemiring R₁
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : Module R₁ M₁
      inst✝² : CommSemiring R₂
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R₂ M₂
      I₁ : RingHom R₁ R
      I₂ : RingHom R₂ R
      B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
      h : B.SeparatingLeft
      x : M₁
      hB : Eq (B x) 0
      ⊢ Eq x 0
    -/
  · simpa only [hB, zero_apply, eq_self_iff_true, forall_const] using h x
    /-
      🎉 no goals
    -/
  have h' : B x = 0 := by
    ext
    rw [zero_apply]
    exact hB _
  /-
    case mpr
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : CommSemiring R₁
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : Module R₁ M₁
    inst✝² : CommSemiring R₂
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    I₁ : RingHom R₁ R
    I₂ : RingHom R₂ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
    h : ∀ (x : M₁), Eq (B x) 0 → Eq x 0
    x : M₁
    hB : ∀ (y : M₂), Eq ((B x) y) 0
    h' : Eq (B x) 0
    ⊢ Eq x 0
  -/
  exact h x h'
  /-
    🎉 no goals
  -/


theorem separatingRight_iff_linear_flip_nontrivial {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} :
    B.SeparatingRight ↔ ∀ y : M₂, B.flip y = 0 → y = 0 := by
  /-
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : CommSemiring R₁
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : Module R₁ M₁
    inst✝² : CommSemiring R₂
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    I₁ : RingHom R₁ R
    I₂ : RingHom R₂ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
    ⊢ Iff B.SeparatingRight (∀ (y : M₂), Eq (B.flip y) 0 → Eq y 0)
  -/
  rw [← flip_separatingLeft, separatingLeft_iff_linear_nontrivial]
  /-
    🎉 no goals
  -/


/-- A bilinear map is left-separating if and only if it has a trivial kernel. -/
theorem separatingLeft_iff_ker_eq_bot {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} :
    B.SeparatingLeft ↔ LinearMap.ker B = ⊥ :=
  Iff.trans separatingLeft_iff_linear_nontrivial LinearMap.ker_eq_bot'.symm


/-- A bilinear map is right-separating if and only if its flip has a trivial kernel. -/
theorem separatingRight_iff_flip_ker_eq_bot {B : M₁ →ₛₗ[I₁] M₂ →ₛₗ[I₂] M} :
    B.SeparatingRight ↔ LinearMap.ker B.flip = ⊥ := by
  /-
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : CommSemiring R₁
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : Module R₁ M₁
    inst✝² : CommSemiring R₂
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R₂ M₂
    I₁ : RingHom R₁ R
    I₂ : RingHom R₂ R
    B : LinearMap I₁ M₁ (LinearMap I₂ M₂ M)
    ⊢ Iff B.SeparatingRight (Eq (LinearMap.ker B.flip) Bot.bot)
  -/
  rw [← flip_separatingLeft, separatingLeft_iff_ker_eq_bot]
  /-
    🎉 no goals
  -/


theorem IsRefl.nondegenerate_iff_separatingLeft {B : M →ₗ[R] M →ₗ[R] M₁} (hB : B.IsRefl) :
    B.Nondegenerate ↔ B.SeparatingLeft := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    ⊢ Iff B.Nondegenerate B.SeparatingLeft
  -/
  refine ⟨fun h ↦ h.1, fun hB' ↦ ⟨hB', ?_⟩⟩
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    hB' : B.SeparatingLeft
    ⊢ B.SeparatingRight
  -/
  rw [separatingRight_iff_flip_ker_eq_bot, hB.ker_eq_bot_iff_ker_flip_eq_bot.mp]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    hB' : B.SeparatingLeft
    ⊢ Eq (LinearMap.ker B) Bot.bot
  -/
  rwa [← separatingLeft_iff_ker_eq_bot]
  /-
    🎉 no goals
  -/


theorem IsRefl.nondegenerate_iff_separatingRight {B : M →ₗ[R] M →ₗ[R] M₁} (hB : B.IsRefl) :
    B.Nondegenerate ↔ B.SeparatingRight := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    ⊢ Iff B.Nondegenerate B.SeparatingRight
  -/
  refine ⟨fun h ↦ h.2, fun hB' ↦ ⟨?_, hB'⟩⟩
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    hB' : B.SeparatingRight
    ⊢ B.SeparatingLeft
  -/
  rw [separatingLeft_iff_ker_eq_bot, hB.ker_eq_bot_iff_ker_flip_eq_bot.mpr]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    hB' : B.SeparatingRight
    ⊢ Eq (LinearMap.ker B.flip) Bot.bot
  -/
  rwa [← separatingRight_iff_flip_ker_eq_bot]
  /-
    🎉 no goals
  -/


lemma disjoint_ker_of_nondegenerate_restrict {B : M →ₗ[R] M →ₗ[R] M₁} {W : Submodule R M}
    (hW : (B.domRestrict₁₂ W W).Nondegenerate) :
    Disjoint W (LinearMap.ker B) := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    ⊢ Disjoint W (LinearMap.ker B)
  -/
  refine Submodule.disjoint_def.mpr fun x hx hx' ↦ ?_
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (LinearMap.ker B) x
    ⊢ Eq x 0
  -/
  let x' : W := ⟨x, hx⟩
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (LinearMap.ker B) x
    x' : Subtype fun x => Membership.mem W x := ⟨x, hx⟩
    ⊢ Eq x 0
  -/
  suffices x' = 0 by simpa [x']
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (LinearMap.ker B) x
    x' : Subtype fun x => Membership.mem W x := ⟨x, hx⟩
    ⊢ Eq x' 0
  -/
  apply hW.1 x'
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (LinearMap.ker B) x
    x' : Subtype fun x => Membership.mem W x := ⟨x, hx⟩
    ⊢ ∀ (y : Subtype fun x => Membership.mem W x), Eq (((B.domRestrict₁₂ W W) x')  …
  -/
  simp_rw [Subtype.forall, domRestrict₁₂_apply]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (LinearMap.ker B) x
    x' : Subtype fun x => Membership.mem W x := ⟨x, hx⟩
    ⊢ ∀ (a : M), Membership.mem W a → Eq ((B x) a) 0
  -/
  intro y hy
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (LinearMap.ker B) x
    x' : Subtype fun x => Membership.mem W x := ⟨x, hx⟩
    y : M
    hy : Membership.mem W y
    ⊢ Eq ((B x) y) 0
  -/
  rw [mem_ker] at hx'
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    W : Submodule R M
    hW : (B.domRestrict₁₂ W W).Nondegenerate
    x : M
    hx : Membership.mem W x
    hx' : Eq (B x) 0
    x' : Subtype fun x => Membership.mem W x := ⟨x, hx⟩
    y : M
    hy : Membership.mem W y
    ⊢ Eq ((B x) y) 0
  -/
  simp [x', hx']
  /-
    🎉 no goals
  -/


lemma IsSymm.nondegenerate_restrict_of_isCompl_ker {B : M →ₗ[R] M →ₗ[R] R} (hB : B.IsSymm)
    {W : Submodule R M} (hW : IsCompl W (LinearMap.ker B)) :
    (B.domRestrict₁₂ W W).Nondegenerate := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    hB : B.IsSymm
    W : Submodule R M
    hW : IsCompl W (LinearMap.ker B)
    ⊢ (B.domRestrict₁₂ W W).Nondegenerate
  -/
  have hB' : (B.domRestrict₁₂ W W).IsRefl := fun x y ↦ hB.isRefl (W.subtype x) (W.subtype y)
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    hB : B.IsSymm
    W : Submodule R M
    hW : IsCompl W (LinearMap.ker B)
    hB' : (B.domRestrict₁₂ W W).IsRefl
    ⊢ (B.domRestrict₁₂ W W).Nondegenerate
  -/
  rw [LinearMap.IsRefl.nondegenerate_iff_separatingLeft hB']
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    hB : B.IsSymm
    W : Submodule R M
    hW : IsCompl W (LinearMap.ker B)
    hB' : (B.domRestrict₁₂ W W).IsRefl
    ⊢ (B.domRestrict₁₂ W W).SeparatingLeft
  -/
  intro ⟨x, hx⟩ hx'
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    hB : B.IsSymm
    W : Submodule R M
    hW : IsCompl W (LinearMap.ker B)
    hB' : (B.domRestrict₁₂ W W).IsRefl
    x : M
    hx : Membership.mem W x
    hx' : ∀ (y : Subtype fun x => Membership.mem W x), Eq (((B.domRestrict₁₂ W W)  …
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  simp only [Submodule.mk_eq_zero]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    hB : B.IsSymm
    W : Submodule R M
    hW : IsCompl W (LinearMap.ker B)
    hB' : (B.domRestrict₁₂ W W).IsRefl
    x : M
    hx : Membership.mem W x
    hx' : ∀ (y : Subtype fun x => Membership.mem W x), Eq (((B.domRestrict₁₂ W W)  …
    ⊢ Eq x 0
  -/
  replace hx' : ∀ y ∈ W, B x y = 0 := by simpa [Subtype.forall] using hx'
  replace hx' : x ∈ W ⊓ ker B := by
    refine ⟨hx, ?_⟩
    ext y
    obtain ⟨u, hu, v, hv, rfl⟩ : ∃ u ∈ W, ∃ v ∈ ker B, u + v = y := by
      rw [← Submodule.mem_sup, hW.sup_eq_top]; exact Submodule.mem_top
    suffices B x u = 0 by rw [mem_ker] at hv; simpa [← hB.eq v, hv]
    exact hx' u hu
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    hB : B.IsSymm
    W : Submodule R M
    hW : IsCompl W (LinearMap.ker B)
    hB' : (B.domRestrict₁₂ W W).IsRefl
    x : M
    hx : Membership.mem W x
    hx' : Membership.mem (Min.min W (LinearMap.ker B)) x
    ⊢ Eq x 0
  -/
  simpa [hW.inf_eq_bot] using hx'
  /-
    🎉 no goals
  -/


/-- The restriction of a reflexive bilinear map `B` onto a submodule `W` is
nondegenerate if `W` has trivial intersection with its orthogonal complement,
that is `Disjoint W (W.orthogonalBilin B)`. -/
theorem nondegenerate_restrict_of_disjoint_orthogonal {B : M →ₗ[R] M →ₗ[R] M₁} (hB : B.IsRefl)
    {W : Submodule R M} (hW : Disjoint W (W.orthogonalBilin B)) :
    (B.domRestrict₁₂ W W).Nondegenerate := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    ⊢ (B.domRestrict₁₂ W W).Nondegenerate
  -/
  rw [(hB.domRestrict W).nondegenerate_iff_separatingLeft]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    ⊢ (B.domRestrict₁₂ W W).SeparatingLeft
  -/
  rintro ⟨x, hx⟩ b₁
  /-
    case mk
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    x : M
    hx : Membership.mem W x
    b₁ : ∀ (y : Subtype fun x => Membership.mem W x), Eq (((B.domRestrict₁₂ W W) ⟨ …
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  rw [Submodule.mk_eq_zero, ← Submodule.mem_bot R]
  /-
    case mk
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    x : M
    hx : Membership.mem W x
    b₁ : ∀ (y : Subtype fun x => Membership.mem W x), Eq (((B.domRestrict₁₂ W W) ⟨ …
    ⊢ Membership.mem Bot.bot x
  -/
  refine hW.le_bot ⟨hx, fun y hy ↦ ?_⟩
  /-
    case mk
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    x : M
    hx : Membership.mem W x
    b₁ : ∀ (y : Subtype fun x => Membership.mem W x), Eq (((B.domRestrict₁₂ W W) ⟨ …
    y : M
    hy : Membership.mem W y
    ⊢ B.IsOrtho y x
  -/
  specialize b₁ ⟨y, hy⟩
  /-
    case mk
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    x : M
    hx : Membership.mem W x
    y : M
    hy : Membership.mem W y
    b₁ : Eq (((B.domRestrict₁₂ W W) ⟨x, hx⟩) ⟨y, hy⟩) 0
    ⊢ B.IsOrtho y x
  -/
  simp_rw [domRestrict₁₂_apply] at b₁
  /-
    case mk
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    x : M
    hx : Membership.mem W x
    y : M
    hy : Membership.mem W y
    b₁ : Eq ((B x) y) 0
    ⊢ B.IsOrtho y x
  -/
  rw [hB.ortho_comm]
  /-
    case mk
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    hB : B.IsRefl
    W : Submodule R M
    hW : Disjoint W (W.orthogonalBilin B)
    x : M
    hx : Membership.mem W x
    y : M
    hy : Membership.mem W y
    b₁ : Eq ((B x) y) 0
    ⊢ B.IsOrtho x y
  -/
  exact b₁
  /-
    🎉 no goals
  -/


/-- An orthogonal basis with respect to a left-separating bilinear map has no self-orthogonal
elements. -/
theorem IsOrthoᵢ.not_isOrtho_basis_self_of_separatingLeft [Nontrivial R]
    {B : M →ₛₗ[I] M →ₛₗ[I'] M₁} {v : Basis n R M} (h : B.IsOrthoᵢ v) (hB : B.SeparatingLeft)
    (i : n) : ¬B.IsOrtho (v i) (v i) := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ⊢ Not (B.IsOrtho (v i) (v i))
  -/
  intro ho
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    ⊢ False
  -/
  refine v.ne_zero i (hB (v i) fun m ↦ ?_)
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    m : M
    ⊢ Eq ((B (v i)) m) 0
  -/
  obtain ⟨vi, rfl⟩ := v.repr.symm.surjective m
  /-
    case intro
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    ⊢ Eq ((B (v i)) (v.repr.symm vi)) 0
  -/
  rw [Basis.repr_symm_apply, Finsupp.linearCombination_apply, Finsupp.sum, map_sum]
  /-
    case intro
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    ⊢ Eq (vi.support.sum fun x => (B (v i)) (HSMul.hSMul (vi x) (v x))) 0
  -/
  apply Finset.sum_eq_zero
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    ⊢ ∀ (x : n), Membership.mem vi.support x → Eq ((B (v i)) (HSMul.hSMul (vi x) ( …
  -/
  rintro j -
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    j : n
    ⊢ Eq ((B (v i)) (HSMul.hSMul (vi j) (v j))) 0
  -/
  rw [map_smulₛₗ]
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    j : n
    ⊢ Eq (HSMul.hSMul (I' (vi j)) ((B (v i)) (v j))) 0
  -/
  suffices B (v i) (v j) = 0 by rw [this, smul_zero]
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingLeft
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    j : n
    ⊢ Eq ((B (v i)) (v j)) 0
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case intro.h.inl
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      I I' : RingHom R R
      inst✝ : Nontrivial R
      B : LinearMap I M (LinearMap I' M M₁)
      v : Basis n R M
      h : B.IsOrthoᵢ ⇑v
      hB : B.SeparatingLeft
      i : n
      ho : B.IsOrtho (v i) (v i)
      vi : Finsupp n R
      ⊢ Eq ((B (v i)) (v i)) 0
    -/
  · exact ho
    /-
      🎉 no goals
    -/
    /-
      case intro.h.inr
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      I I' : RingHom R R
      inst✝ : Nontrivial R
      B : LinearMap I M (LinearMap I' M M₁)
      v : Basis n R M
      h : B.IsOrthoᵢ ⇑v
      hB : B.SeparatingLeft
      i : n
      ho : B.IsOrtho (v i) (v i)
      vi : Finsupp n R
      j : n
      hij : Ne i j
      ⊢ Eq ((B (v i)) (v j)) 0
    -/
  · exact h hij
    /-
      🎉 no goals
    -/


/-- An orthogonal basis with respect to a right-separating bilinear map has no self-orthogonal
elements. -/
theorem IsOrthoᵢ.not_isOrtho_basis_self_of_separatingRight [Nontrivial R]
    {B : M →ₛₗ[I] M →ₛₗ[I'] M₁} {v : Basis n R M} (h : B.IsOrthoᵢ v) (hB : B.SeparatingRight)
    (i : n) : ¬B.IsOrtho (v i) (v i) := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.IsOrthoᵢ ⇑v
    hB : B.SeparatingRight
    i : n
    ⊢ Not (B.IsOrtho (v i) (v i))
  -/
  rw [isOrthoᵢ_flip] at h
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.flip.IsOrthoᵢ ⇑v
    hB : B.SeparatingRight
    i : n
    ⊢ Not (B.IsOrtho (v i) (v i))
  -/
  rw [isOrtho_flip]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    I I' : RingHom R R
    inst✝ : Nontrivial R
    B : LinearMap I M (LinearMap I' M M₁)
    v : Basis n R M
    h : B.flip.IsOrthoᵢ ⇑v
    hB : B.SeparatingRight
    i : n
    ⊢ Not (B.flip.IsOrtho (v i) (v i))
  -/
  exact h.not_isOrtho_basis_self_of_separatingLeft (flip_separatingLeft.mpr hB) i
  /-
    🎉 no goals
  -/


/-- Given an orthogonal basis with respect to a bilinear map, the bilinear map is left-separating if
the basis has no elements which are self-orthogonal. -/
theorem IsOrthoᵢ.separatingLeft_of_not_isOrtho_basis_self [NoZeroSMulDivisors R M₁]
    {B : M →ₗ[R] M →ₗ[R] M₁} (v : Basis n R M) (hO : B.IsOrthoᵢ v)
    (h : ∀ i, ¬B.IsOrtho (v i) (v i)) : B.SeparatingLeft := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    ⊢ B.SeparatingLeft
  -/
  intro m hB
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    m : M
    hB : ∀ (y : M), Eq ((B m) y) 0
    ⊢ Eq m 0
  -/
  obtain ⟨vi, rfl⟩ := v.repr.symm.surjective m
  /-
    case intro
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (y : M), Eq ((B (v.repr.symm vi)) y) 0
    ⊢ Eq (v.repr.symm vi) 0
  -/
  rw [LinearEquiv.map_eq_zero_iff]
  /-
    case intro
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (y : M), Eq ((B (v.repr.symm vi)) y) 0
    ⊢ Eq vi 0
  -/
  ext i
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (y : M), Eq ((B (v.repr.symm vi)) y) 0
    i : n
    ⊢ Eq (vi i) (0 i)
  -/
  rw [Finsupp.zero_apply]
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (y : M), Eq ((B (v.repr.symm vi)) y) 0
    i : n
    ⊢ Eq (vi i) 0
  -/
  specialize hB (v i)
  simp_rw [Basis.repr_symm_apply, Finsupp.linearCombination_apply, Finsupp.sum, map_sum₂,
           map_smulₛₗ₂] at hB
  /-
    case intro.h
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    i : n
    hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
    ⊢ Eq (vi i) 0
  -/
  rw [Finset.sum_eq_single i] at hB
    /-
      case intro.h
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (HSMul.hSMul ((RingHom.id R) (vi i)) ((B (v i)) (v i))) 0
      ⊢ Eq (vi i) 0
    -/
  · exact (smul_eq_zero.mp hB).elim _root_.id (h i).elim
    /-
      🎉 no goals
    -/
    /-
      case intro.h.h₀
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
      ⊢ ∀ (b : n), Membership.mem vi.support b → Ne b i → Eq (HSMul.hSMul ((RingHom. …
    -/
  · intro j _hj hij
    /-
      case intro.h.h₀
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
      j : n
      _hj : Membership.mem vi.support j
      hij : Ne j i
      ⊢ Eq (HSMul.hSMul ((RingHom.id R) (vi j)) ((B (v j)) (v i))) 0
    -/
    replace hij : B (v j) (v i) = 0 := hO hij
    /-
      case intro.h.h₀
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
      j : n
      _hj : Membership.mem vi.support j
      hij : Eq ((B (v j)) (v i)) 0
      ⊢ Eq (HSMul.hSMul ((RingHom.id R) (vi j)) ((B (v j)) (v i))) 0
    -/
    rw [hij, RingHom.id_apply, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.h.h₁
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
      ⊢ Not (Membership.mem vi.support i) → Eq (HSMul.hSMul ((RingHom.id R) (vi i))  …
    -/
  · intro hi
    /-
      case intro.h.h₁
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
      hi : Not (Membership.mem vi.support i)
      ⊢ Eq (HSMul.hSMul ((RingHom.id R) (vi i)) ((B (v i)) (v i))) 0
    -/
    replace hi : vi i = 0 := Finsupp.not_mem_support_iff.mp hi
    /-
      case intro.h.h₁
      R : Type u_1
      M : Type u_5
      M₁ : Type u_6
      n : Type u_19
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₁
      inst✝¹ : Module R M₁
      inst✝ : NoZeroSMulDivisors R M₁
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
      v : Basis n R M
      hO : B.IsOrthoᵢ ⇑v
      h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HSMul.hSMul ((RingHom.id R) (vi x)) ((B (v x) …
      hi : Eq (vi i) 0
      ⊢ Eq (HSMul.hSMul ((RingHom.id R) (vi i)) ((B (v i)) (v i))) 0
    -/
    rw [hi, RingHom.id_apply, zero_smul]
    /-
      🎉 no goals
    -/


/-- Given an orthogonal basis with respect to a bilinear map, the bilinear map is right-separating
if the basis has no elements which are self-orthogonal. -/
theorem IsOrthoᵢ.separatingRight_iff_not_isOrtho_basis_self [NoZeroSMulDivisors R M₁]
    {B : M →ₗ[R] M →ₗ[R] M₁} (v : Basis n R M) (hO : B.IsOrthoᵢ v)
    (h : ∀ i, ¬B.IsOrtho (v i) (v i)) : B.SeparatingRight := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    ⊢ B.SeparatingRight
  -/
  rw [isOrthoᵢ_flip] at hO
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.flip.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    ⊢ B.SeparatingRight
  -/
  rw [← flip_separatingLeft]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.flip.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    ⊢ B.flip.SeparatingLeft
  -/
  refine IsOrthoᵢ.separatingLeft_of_not_isOrtho_basis_self v hO fun i ↦ ?_
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.flip.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    i : n
    ⊢ Not (B.flip.IsOrtho (v i) (v i))
  -/
  rw [isOrtho_flip]
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    n : Type u_19
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : NoZeroSMulDivisors R M₁
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M M₁)
    v : Basis n R M
    hO : B.flip.IsOrthoᵢ ⇑v
    h : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    i : n
    ⊢ Not (B.flip.flip.IsOrtho (v i) (v i))
  -/
  exact h i
  /-
    🎉 no goals
  -/


/-- Given an orthogonal basis with respect to a bilinear map, the bilinear map is nondegenerate
if the basis has no elements which are self-orthogonal. -/
theorem IsOrthoᵢ.nondegenerate_of_not_isOrtho_basis_self [NoZeroSMulDivisors R M₁]
    {B : M →ₗ[R] M →ₗ[R] M₁} (v : Basis n R M) (hO : B.IsOrthoᵢ v)
    (h : ∀ i, ¬B.IsOrtho (v i) (v i)) : B.Nondegenerate :=
  ⟨IsOrthoᵢ.separatingLeft_of_not_isOrtho_basis_self v hO h,
    IsOrthoᵢ.separatingRight_iff_not_isOrtho_basis_self v hO h⟩


lemma apply_smul_sub_smul_sub_eq [CommRing R] [AddCommGroup M] [Module R M]
    (B : LinearMap.BilinForm R M) (x y : M) :
    B ((B x y) • x - (B x x) • y) ((B x y) • x - (B x x) • y) =
      (B x x) * ((B x x) * (B y y) - (B x y) * (B y x)) := by
  simp only [map_sub, map_smul, sub_apply, smul_apply, smul_eq_mul, mul_sub,
    mul_comm (B x y) (B x x), mul_left_comm (B x y) (B x x)]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    x y : M
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul ((B x) x) (HMul.hMul ((B x) y) ((B x) y) …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- The **Cauchy-Schwarz inequality** for positive semidefinite forms. -/
lemma apply_mul_apply_le_of_forall_zero_le (hs : ∀ x, 0 ≤ B x x) (x y : M) :
    (B x y) * (B y x) ≤ (B x x) * (B y y) := by
  have aux (x y : M) : 0 ≤ (B x x) * ((B x x) * (B y y) - (B x y) * (B y x)) := by
    rw [← apply_smul_sub_smul_sub_eq B x y]
    exact hs (B x y • x - B x x • y)
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    x y : M
    aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
    ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
  -/
  rcases lt_or_le 0 (B x x) with hx | hx
    /-
      case inl
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hs : ∀ (x : M), LE.le 0 ((B x) x)
      x y : M
      aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
      hx : LT.lt 0 ((B x) x)
      ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
    -/
  · exact sub_nonneg.mp <| nonneg_of_mul_nonneg_right (aux x y) hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hs : ∀ (x : M), LE.le 0 ((B x) x)
      x y : M
      aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
      hx : LE.le ((B x) x) 0
      ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
    -/
  · replace hx : B x x = 0 := le_antisymm hx (hs x)
    /-
      case inr
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hs : ∀ (x : M), LE.le 0 ((B x) x)
      x y : M
      aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
      hx : Eq ((B x) x) 0
      ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
    -/
    rcases lt_or_le 0 (B y y) with hy | hy
      /-
        case inr.inl
        R : Type u_1
        M : Type u_5
        inst✝² : LinearOrderedCommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        B : LinearMap.BilinForm R M
        hs : ∀ (x : M), LE.le 0 ((B x) x)
        x y : M
        aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
        hx : Eq ((B x) x) 0
        hy : LT.lt 0 ((B y) y)
        ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      -/
    · rw [mul_comm (B x y), mul_comm (B x x)]
      /-
        case inr.inl
        R : Type u_1
        M : Type u_5
        inst✝² : LinearOrderedCommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        B : LinearMap.BilinForm R M
        hs : ∀ (x : M), LE.le 0 ((B x) x)
        x y : M
        aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
        hx : Eq ((B x) x) 0
        hy : LT.lt 0 ((B y) y)
        ⊢ LE.le (HMul.hMul ((B y) x) ((B x) y)) (HMul.hMul ((B y) y) ((B x) x))
      -/
      exact sub_nonneg.mp <| nonneg_of_mul_nonneg_right (aux y x) hy
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        R : Type u_1
        M : Type u_5
        inst✝² : LinearOrderedCommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        B : LinearMap.BilinForm R M
        hs : ∀ (x : M), LE.le 0 ((B x) x)
        x y : M
        aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
        hx : Eq ((B x) x) 0
        hy : LE.le ((B y) y) 0
        ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      -/
    · replace hy : B y y = 0 := le_antisymm hy (hs y)
      /-
        case inr.inr
        R : Type u_1
        M : Type u_5
        inst✝² : LinearOrderedCommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        B : LinearMap.BilinForm R M
        hs : ∀ (x : M), LE.le 0 ((B x) x)
        x y : M
        aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
        hx : Eq ((B x) x) 0
        hy : Eq ((B y) y) 0
        ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      -/
      suffices B x y = - B y x by simpa [this, hx, hy] using mul_self_nonneg (B y x)
      /-
        case inr.inr
        R : Type u_1
        M : Type u_5
        inst✝² : LinearOrderedCommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        B : LinearMap.BilinForm R M
        hs : ∀ (x : M), LE.le 0 ((B x) x)
        x y : M
        aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
        hx : Eq ((B x) x) 0
        hy : Eq ((B y) y) 0
        ⊢ Eq ((B x) y) (Neg.neg ((B y) x))
      -/
      rw [eq_neg_iff_add_eq_zero]
      /-
        case inr.inr
        R : Type u_1
        M : Type u_5
        inst✝² : LinearOrderedCommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        B : LinearMap.BilinForm R M
        hs : ∀ (x : M), LE.le 0 ((B x) x)
        x y : M
        aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
        hx : Eq ((B x) x) 0
        hy : Eq ((B y) y) 0
        ⊢ Eq (HAdd.hAdd ((B x) y) ((B y) x)) 0
      -/
      apply le_antisymm
        /-
          case inr.inr.a
          R : Type u_1
          M : Type u_5
          inst✝² : LinearOrderedCommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          B : LinearMap.BilinForm R M
          hs : ∀ (x : M), LE.le 0 ((B x) x)
          x y : M
          aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
          hx : Eq ((B x) x) 0
          hy : Eq ((B y) y) 0
          ⊢ LE.le (HAdd.hAdd ((B x) y) ((B y) x)) 0
        -/
      · simpa [hx, hy, le_neg_iff_add_nonpos_left] using hs (x - y)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.a
          R : Type u_1
          M : Type u_5
          inst✝² : LinearOrderedCommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          B : LinearMap.BilinForm R M
          hs : ∀ (x : M), LE.le 0 ((B x) x)
          x y : M
          aux : ∀ (x y : M), LE.le 0 (HMul.hMul ((B x) x) (HSub.hSub (HMul.hMul ((B x) x …
          hx : Eq ((B x) x) 0
          hy : Eq ((B y) y) 0
          ⊢ LE.le 0 (HAdd.hAdd ((B x) y) ((B y) x))
        -/
      · simpa [hx, hy] using hs (x + y)
        /-
          🎉 no goals
        -/


/-- The **Cauchy-Schwarz inequality** for positive semidefinite symmetric forms. -/
lemma apply_sq_le_of_symm (hs : ∀ x, 0 ≤ B x x) (hB : B.IsSymm) (x y : M) :
    (B x y) ^ 2 ≤ (B x x) * (B y y) := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x y : M
    ⊢ LE.le (HPow.hPow ((B x) y) 2) (HMul.hMul ((B x) x) ((B y) y))
  -/
  rw [show (B x y) ^ 2 = (B x y) * (B y x) by rw [sq, ← hB, RingHom.id_apply]]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x y : M
    ⊢ LE.le (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
  -/
  exact apply_mul_apply_le_of_forall_zero_le B hs x y
  /-
    🎉 no goals
  -/


/-- The equality case of **Cauchy-Schwarz**. -/
lemma not_linearIndependent_of_apply_mul_apply_eq (hp : ∀ x, x ≠ 0 → 0 < B x x)
    (x y : M) (he : (B x y) * (B y x) = (B x x) * (B y y)) :
    ¬ LinearIndependent R ![x, y] := by
  have hz : (B x y) • x - (B x x) • y = 0 := by
    by_contra hc
    exact (ne_of_lt (hp ((B x) y • x - (B x) x • y) hc)).symm <|
      (apply_smul_sub_smul_sub_eq B x y).symm ▸ (mul_eq_zero_of_right ((B x) x)
      (sub_eq_zero_of_eq he.symm))
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
    x y : M
    he : Eq (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
    hz : Eq (HSub.hSub (HSMul.hSMul ((B x) y) x) (HSMul.hSMul ((B x) x) y)) 0
    ⊢ Not (LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty …
  -/
  by_contra hL
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
    x y : M
    he : Eq (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
    hz : Eq (HSub.hSub (HSMul.hSMul ((B x) y) x) (HSMul.hSMul ((B x) x) y)) 0
    hL : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    ⊢ False
  -/
  by_cases hx : x = 0
    /-
      case pos
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      he : Eq (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      hz : Eq (HSub.hSub (HSMul.hSMul ((B x) y) x) (HSMul.hSMul ((B x) x) y)) 0
      hL : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
      hx : Eq x 0
      ⊢ False
    -/
  · simpa [hx] using LinearIndependent.ne_zero 0 hL
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      he : Eq (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      hz : Eq (HSub.hSub (HSMul.hSMul ((B x) y) x) (HSMul.hSMul ((B x) x) y)) 0
      hL : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
      hx : Not (Eq x 0)
      ⊢ False
    -/
  · have h := sub_eq_zero.mpr (sub_eq_zero.mp hz).symm
    /-
      case neg
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      he : Eq (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      hz : Eq (HSub.hSub (HSMul.hSMul ((B x) y) x) (HSMul.hSMul ((B x) x) y)) 0
      hL : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
      hx : Not (Eq x 0)
      h : Eq (HSub.hSub (HSMul.hSMul ((B x) x) y) (HSMul.hSMul ((B x) y) x)) 0
      ⊢ False
    -/
    rw [sub_eq_add_neg, ← neg_smul, add_comm] at h
    /-
      case neg
      R : Type u_1
      M : Type u_5
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      he : Eq (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y))
      hz : Eq (HSub.hSub (HSMul.hSMul ((B x) y) x) (HSMul.hSMul ((B x) x) y)) 0
      hL : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
      hx : Not (Eq x 0)
      h : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg ((B x) y)) x) (HSMul.hSMul ((B x) x) y …
      ⊢ False
    -/
    exact (Ne.symm (ne_of_lt (hp x hx))) (LinearIndependent.eq_zero_of_pair hL h).2
    /-
      🎉 no goals
    -/


/-- Strict **Cauchy-Schwarz** is equivalent to linear independence for positive definite forms. -/
lemma apply_mul_apply_lt_iff_linearIndependent [NoZeroSMulDivisors R M]
    (hp : ∀ x, x ≠ 0 → 0 < B x x) (x y : M) :
    (B x y) * (B y x) < (B x x) * (B y y) ↔ LinearIndependent R ![x, y] := by
  have hle : ∀ z, 0 ≤ B z z := by
    intro z
    by_cases hz : z = 0; simp [hz]
    exact le_of_lt (hp z hz)
  /-
    R : Type u_1
    M : Type u_5
    inst✝³ : LinearOrderedCommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    B : LinearMap.BilinForm R M
    inst✝ : NoZeroSMulDivisors R M
    hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
    x y : M
    hle : ∀ (z : M), LE.le 0 ((B z) z)
    ⊢ Iff (LT.lt (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y)))  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      ⊢ LT.lt (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y)) → Line …
    -/
  · contrapose!
    /-
      case mp
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      ⊢ Not (LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty …
    -/
    intro h
    /-
      case mp
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      h : Not (LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmp …
      ⊢ LE.le (HMul.hMul ((B x) x) ((B y) y)) (HMul.hMul ((B x) y) ((B y) x))
    -/
    rw [LinearIndependent.pair_iff] at h
    /-
      case mp
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      h : Not (∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → A …
      ⊢ LE.le (HMul.hMul ((B x) x) ((B y) y)) (HMul.hMul ((B x) y) ((B y) x))
    -/
    push_neg at h
    /-
      case mp
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      h : Exists fun s => Exists fun t => And (Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMu …
      ⊢ LE.le (HMul.hMul ((B x) x) ((B y) y)) (HMul.hMul ((B x) y) ((B y) x))
    -/
    obtain ⟨r, s, hl, h0⟩ := h
    /-
      case mp.intro.intro.intro
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      r s : R
      hl : Eq (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul s y)) 0
      h0 : Eq r 0 → Ne s 0
      ⊢ LE.le (HMul.hMul ((B x) x) ((B y) y)) (HMul.hMul ((B x) y) ((B y) x))
    -/
    by_cases hr : r = 0; · simp_all
                           /-
                             🎉 no goals
                           -/
    /-
      case neg
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      r s : R
      hl : Eq (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul s y)) 0
      h0 : Eq r 0 → Ne s 0
      hr : Not (Eq r 0)
      ⊢ LE.le (HMul.hMul ((B x) x) ((B y) y)) (HMul.hMul ((B x) y) ((B y) x))
    -/
    by_cases hs : s = 0; · simp_all
                           /-
                             🎉 no goals
                           -/
    suffices
        (B (r • x) (r • x)) * (B (s • y) (s • y)) = (B (r • x) (s • y)) * (B (s • y) (r • x)) by
      simp only [map_smul, smul_apply, smul_eq_mul] at this
      rw [show r * (r * (B x) x) * (s * (s * (B y) y)) = (r * r * s * s) * ((B x) x * (B y) y) by
        ring, show s * (r * (B x) y) * (r * (s * (B y) x)) = (r * r * s * s) * ((B x) y * (B y) x)
        by ring] at this
      have hrs : r * r * s * s ≠ 0 := by simp [hr, hs]
      exact le_of_eq <| mul_right_injective₀ hrs this
    /-
      case neg
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      r s : R
      hl : Eq (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul s y)) 0
      h0 : Eq r 0 → Ne s 0
      hr : Not (Eq r 0)
      hs : Not (Eq s 0)
      ⊢ Eq (HMul.hMul ((B (HSMul.hSMul r x)) (HSMul.hSMul r x)) ((B (HSMul.hSMul s y …
    -/
    simp [show s • y = - r • x by rwa [neg_smul, ← add_eq_zero_iff_eq_neg']]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      ⊢ LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty)) →  …
    -/
  · contrapose!
    /-
      case mpr
      R : Type u_1
      M : Type u_5
      inst✝³ : LinearOrderedCommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      B : LinearMap.BilinForm R M
      inst✝ : NoZeroSMulDivisors R M
      hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
      x y : M
      hle : ∀ (z : M), LE.le 0 ((B z) z)
      ⊢ LE.le (HMul.hMul ((B x) x) ((B y) y)) (HMul.hMul ((B x) y) ((B y) x)) → Not  …
    -/
    intro h
    refine not_linearIndependent_of_apply_mul_apply_eq B hp x y (le_antisymm
      (apply_mul_apply_le_of_forall_zero_le B hle x y) h)


/-- Strict **Cauchy-Schwarz** is equivalent to linear independence for positive definite symmetric
forms. -/
lemma apply_sq_lt_iff_linearIndependent_of_symm [NoZeroSMulDivisors R M]
    (hp : ∀ x, x ≠ 0 → 0 < B x x) (hB: B.IsSymm) (x y : M) :
    (B x y) ^ 2 < (B x x) * (B y y) ↔ LinearIndependent R ![x, y] := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝³ : LinearOrderedCommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    B : LinearMap.BilinForm R M
    inst✝ : NoZeroSMulDivisors R M
    hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x y : M
    ⊢ Iff (LT.lt (HPow.hPow ((B x) y) 2) (HMul.hMul ((B x) x) ((B y) y))) (LinearI …
  -/
  rw [show (B x y) ^ 2 = (B x y) * (B y x) by rw [sq, ← hB, RingHom.id_apply]]
  /-
    R : Type u_1
    M : Type u_5
    inst✝³ : LinearOrderedCommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    B : LinearMap.BilinForm R M
    inst✝ : NoZeroSMulDivisors R M
    hp : ∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x y : M
    ⊢ Iff (LT.lt (HMul.hMul ((B x) y) ((B y) x)) (HMul.hMul ((B x) x) ((B y) y)))  …
  -/
  exact apply_mul_apply_lt_iff_linearIndependent B hp x y
  /-
    🎉 no goals
  -/


lemma apply_apply_same_eq_zero_iff (hs : ∀ x, 0 ≤ B x x) (hB : B.IsSymm) {x : M} :
    B x x = 0 ↔ x ∈ LinearMap.ker B := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x : M
    ⊢ Iff (Eq ((B x) x) 0) (Membership.mem (LinearMap.ker B) x)
  -/
  rw [LinearMap.mem_ker]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x : M
    ⊢ Iff (Eq ((B x) x) 0) (Eq (B x) 0)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simp [h]⟩
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x : M
    h : Eq ((B x) x) 0
    ⊢ Eq (B x) 0
  -/
  ext y
  /-
    case h
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x : M
    h : Eq ((B x) x) 0
    y : M
    ⊢ Eq ((B x) y) (0 y)
  -/
  have := B.apply_sq_le_of_symm hs hB x y
  /-
    case h
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x : M
    h : Eq ((B x) x) 0
    y : M
    this : LE.le (HPow.hPow ((B x) y) 2) (HMul.hMul ((B x) x) ((B y) y))
    ⊢ Eq ((B x) y) (0 y)
  -/
  simp only [h, zero_mul] at this
  /-
    case h
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    x : M
    h : Eq ((B x) x) 0
    y : M
    this : LE.le (HPow.hPow ((B x) y) 2) 0
    ⊢ Eq ((B x) y) (0 y)
  -/
  exact pow_eq_zero <| le_antisymm this (sq_nonneg (B x y))
  /-
    🎉 no goals
  -/


lemma nondegenerate_iff (hs : ∀ x, 0 ≤ B x x) (hB : B.IsSymm) :
    B.Nondegenerate ↔ ∀ x, B x x = 0 ↔ x = 0 := by
  simp_rw [hB.isRefl.nondegenerate_iff_separatingLeft, separatingLeft_iff_ker_eq_bot,
    Submodule.eq_bot_iff, B.apply_apply_same_eq_zero_iff hs hB, mem_ker]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    ⊢ Iff (∀ (x : M), Eq (B x) 0 → Eq x 0) (∀ (x : M), Iff (Eq (B x) 0) (Eq x 0))
  -/
  exact forall_congr' fun x ↦ by aesop
  /-
    🎉 no goals
  -/


/-- A convenience variant of `LinearMap.BilinForm.nondegenerate_iff` characterising nondegeneracy as
positive definiteness. -/
lemma nondegenerate_iff' (hs : ∀ x, 0 ≤ B x x) (hB : B.IsSymm) :
    B.Nondegenerate ↔ ∀ x, x ≠ 0 → 0 < B x x := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    ⊢ Iff (LinearMap.Nondegenerate B) (∀ (x : M), Ne x 0 → LT.lt 0 ((B x) x))
  -/
  rw [B.nondegenerate_iff hs hB, ← not_iff_not]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    ⊢ Iff (Not (∀ (x : M), Iff (Eq ((B x) x) 0) (Eq x 0))) (Not (∀ (x : M), Ne x 0 …
  -/
  push_neg
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    ⊢ Iff (Exists fun x => Or (And (Eq ((B x) x) 0) (Ne x 0)) (And (Ne ((B x) x) 0 …
  -/
  exact exists_congr fun x ↦ ⟨by aesop, fun ⟨h₀, h⟩ ↦ Or.inl ⟨le_antisymm h (hs x), h₀⟩⟩
  /-
    🎉 no goals
  -/


lemma nondegenerate_restrict_iff_disjoint_ker (hs : ∀ x, 0 ≤ B x x) (hB : B.IsSymm)
    {W : Submodule R M} :
    (B.domRestrict₁₂ W W).Nondegenerate ↔ Disjoint W (LinearMap.ker B) := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    ⊢ Iff (LinearMap.domRestrict₁₂ B W W).Nondegenerate (Disjoint W (LinearMap.ker …
  -/
  refine ⟨disjoint_ker_of_nondegenerate_restrict, fun hW ↦ ?_⟩
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    ⊢ (LinearMap.domRestrict₁₂ B W W).Nondegenerate
  -/
  have hB' : (B.domRestrict₁₂ W W).IsRefl := fun x y ↦ hB.isRefl (W.subtype x) (W.subtype y)
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    ⊢ (LinearMap.domRestrict₁₂ B W W).Nondegenerate
  -/
  rw [IsRefl.nondegenerate_iff_separatingLeft hB']
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    ⊢ (LinearMap.domRestrict₁₂ B W W).SeparatingLeft
  -/
  intro ⟨x, hx⟩ h
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    x : M
    hx : Membership.mem W x
    h : ∀ (y : Subtype fun x => Membership.mem W x), Eq (((LinearMap.domRestrict₁₂ …
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  simp_rw [Subtype.forall, domRestrict₁₂_apply] at h
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    x : M
    hx : Membership.mem W x
    h : ∀ (a : M), Membership.mem W a → Eq ((B x) a) 0
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  specialize h x hx
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    x : M
    hx : Membership.mem W x
    h : Eq ((B x) x) 0
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  rw [B.apply_apply_same_eq_zero_iff hs hB] at h
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    x : M
    hx : Membership.mem W x
    h : Membership.mem (LinearMap.ker B) x
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  have key : x ∈ W ⊓ LinearMap.ker B := ⟨hx, h⟩
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hs : ∀ (x : M), LE.le 0 ((B x) x)
    hB : LinearMap.IsSymm B
    W : Submodule R M
    hW : Disjoint W (LinearMap.ker B)
    hB' : (LinearMap.domRestrict₁₂ B W W).IsRefl
    x : M
    hx : Membership.mem W x
    h : Membership.mem (LinearMap.ker B) x
    key : Membership.mem (Min.min W (LinearMap.ker B)) x
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  simpa [hW.eq_bot] using key
  /-
    🎉 no goals
  -/


