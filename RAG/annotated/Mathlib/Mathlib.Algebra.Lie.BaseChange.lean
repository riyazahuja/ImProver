/-- The Lie bracket on the extension of a Lie algebra `L` over `R` by an algebra `A` over `R`. -/
private def bracket' : A ⊗[R] L →ₗ[A] A ⊗[R] M →ₗ[A] A ⊗[R] M :=
  TensorProduct.curry <|
    TensorProduct.AlgebraTensorModule.map
        (LinearMap.mul' A A) (LieModule.toModuleHom R L M : L ⊗[R] M →ₗ[R] M) ∘ₗ
      (TensorProduct.AlgebraTensorModule.tensorTensorTensorComm R A A L A M).toLinearMap


@[simp]
private theorem bracket'_tmul (s t : A) (x : L) (m : M) :
    bracket' R A L M (s ⊗ₜ[R] x) (t ⊗ₜ[R] m) = (s * t) ⊗ₜ ⁅x, m⁆ := rfl


instance : Bracket (A ⊗[R] L) (A ⊗[R] M) where bracket x m := bracket' R A L M x m


private theorem bracket_def (x : A ⊗[R] L) (m : A ⊗[R] M) : ⁅x, m⁆ = bracket' R A L M x m :=
  rfl


@[simp]
theorem bracket_tmul (s t : A) (x : L) (y : M) : ⁅s ⊗ₜ[R] x, t ⊗ₜ[R] y⁆ = (s * t) ⊗ₜ ⁅x, y⁆ := rfl


private theorem bracket_lie_self (x : A ⊗[R] L) : ⁅x, x⁆ = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : TensorProduct R A L
    ⊢ Eq (Bracket.bracket x x) 0
  -/
  simp only [bracket_def]
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : TensorProduct R A L
    ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) x) x) 0
  -/
  refine x.induction_on ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : TensorProduct R A L
      ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) 0) 0) 0
    -/
  · simp only [LinearMap.map_zero, eq_self_iff_true, LinearMap.zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : TensorProduct R A L
      ⊢ ∀ (x : A) (y : L), Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) (TensorP …
    -/
  · intro a l
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : TensorProduct R A L
      a : A
      l : L
      ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) (TensorProduct.tmul R a l)) …
    -/
    simp only [bracket'_tmul, TensorProduct.tmul_zero, eq_self_iff_true, lie_self]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : TensorProduct R A L
      ⊢ ∀ (x y : TensorProduct R A L), Eq (((LieAlgebra.ExtendScalars.bracket' R A L …
    -/
  · intro z₁ z₂ h₁ h₂
    suffices bracket' R A L L z₁ z₂ + bracket' R A L L z₂ z₁ = 0 by
      rw [LinearMap.map_add, LinearMap.map_add, LinearMap.add_apply, LinearMap.add_apply, h₁, h₂,
        zero_add, add_zero, add_comm, this]
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      L : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x z₁ z₂ : TensorProduct R A L
      h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
      h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
      ⊢ Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₂) (((LieAl …
    -/
    refine z₁.induction_on ?_ ?_ ?_
      /-
        case refine_3.refine_1
        R : Type u_1
        A : Type u_2
        L : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x z₁ z₂ : TensorProduct R A L
        h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
        h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
        ⊢ Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) 0) z₂) (((LieAlg …
      -/
    · simp only [LinearMap.map_zero, add_zero, LinearMap.zero_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_3.refine_2
        R : Type u_1
        A : Type u_2
        L : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x z₁ z₂ : TensorProduct R A L
        h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
        h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
        ⊢ ∀ (x : A) (y : L), Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L  …
      -/
    · intro a₁ l₁; refine z₂.induction_on ?_ ?_ ?_
        /-
          case refine_3.refine_2.refine_1
          R : Type u_1
          A : Type u_2
          L : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : CommRing A
          inst✝² : Algebra R A
          inst✝¹ : LieRing L
          inst✝ : LieAlgebra R L
          x z₁ z₂ : TensorProduct R A L
          h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
          h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
          a₁ : A
          l₁ : L
          ⊢ Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) (TensorProduct.t …
        -/
      · simp only [LinearMap.map_zero, add_zero, LinearMap.zero_apply]
        /-
          🎉 no goals
        -/
        /-
          case refine_3.refine_2.refine_2
          R : Type u_1
          A : Type u_2
          L : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : CommRing A
          inst✝² : Algebra R A
          inst✝¹ : LieRing L
          inst✝ : LieAlgebra R L
          x z₁ z₂ : TensorProduct R A L
          h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
          h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
          a₁ : A
          l₁ : L
          ⊢ ∀ (x : A) (y : L), Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L  …
        -/
      · intro a₂ l₂
        simp only [← lie_skew l₂ l₁, mul_comm a₁ a₂, TensorProduct.tmul_neg, bracket'_tmul,
          add_neg_cancel]
        /-
          case refine_3.refine_2.refine_3
          R : Type u_1
          A : Type u_2
          L : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : CommRing A
          inst✝² : Algebra R A
          inst✝¹ : LieRing L
          inst✝ : LieAlgebra R L
          x z₁ z₂ : TensorProduct R A L
          h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
          h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
          a₁ : A
          l₁ : L
          ⊢ ∀ (x y : TensorProduct R A L), Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bra …
        -/
      · intro y₁ y₂ hy₁ hy₂
        /-
          case refine_3.refine_2.refine_3
          R : Type u_1
          A : Type u_2
          L : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : CommRing A
          inst✝² : Algebra R A
          inst✝¹ : LieRing L
          inst✝ : LieAlgebra R L
          x z₁ z₂ : TensorProduct R A L
          h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
          h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
          a₁ : A
          l₁ : L
          y₁ y₂ : TensorProduct R A L
          hy₁ : Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) (TensorProdu …
          hy₂ : Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) (TensorProdu …
          ⊢ Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) (TensorProduct.t …
        -/
        simp only [hy₁, hy₂, add_add_add_comm, add_zero, LinearMap.add_apply, LinearMap.map_add]
        /-
          🎉 no goals
        -/
      /-
        case refine_3.refine_3
        R : Type u_1
        A : Type u_2
        L : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x z₁ z₂ : TensorProduct R A L
        h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
        h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
        ⊢ ∀ (x y : TensorProduct R A L), Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bra …
      -/
    · intro y₁ y₂ hy₁ hy₂
      /-
        case refine_3.refine_3
        R : Type u_1
        A : Type u_2
        L : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Algebra R A
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x z₁ z₂ : TensorProduct R A L
        h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₁) z₁) 0
        h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L L) z₂) z₂) 0
        y₁ y₂ : TensorProduct R A L
        hy₁ : Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) y₁) z₂) (((L …
        hy₂ : Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) y₂) z₂) (((L …
        ⊢ Eq (HAdd.hAdd (((LieAlgebra.ExtendScalars.bracket' R A L L) (HAdd.hAdd y₁ y₂ …
      -/
      simp only [add_add_add_comm, hy₁, hy₂, add_zero, LinearMap.add_apply, LinearMap.map_add]
      /-
        🎉 no goals
      -/


private theorem bracket_leibniz_lie (x y : A ⊗[R] L) (z : A ⊗[R] M) :
    ⁅x, ⁅y, z⁆⁆ = ⁅⁅x, y⁆, z⁆ + ⁅y, ⁅x, z⁆⁆ := by
  -- Porting note: replaced some `simp`s by `rw`s to avoid raising heartbeats
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x y : TensorProduct R A L
    z : TensorProduct R A M
    ⊢ Eq (Bracket.bracket x (Bracket.bracket y z)) (HAdd.hAdd (Bracket.bracket (Br …
  -/
  simp only [bracket_def]
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x y : TensorProduct R A L
    z : TensorProduct R A M
    ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) x) (((LieAlgebra.ExtendScal …
  -/
  refine x.induction_on ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing A
      inst✝⁶ : Algebra R A
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y : TensorProduct R A L
      z : TensorProduct R A M
      ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) 0) (((LieAlgebra.ExtendScal …
    -/
  · simp only [LinearMap.map_zero, add_zero, LinearMap.zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing A
      inst✝⁶ : Algebra R A
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y : TensorProduct R A L
      z : TensorProduct R A M
      ⊢ ∀ (x : A) (y_1 : L), Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (Tenso …
    -/
  · intro a₁ l₁
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing A
      inst✝⁶ : Algebra R A
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y : TensorProduct R A L
      z : TensorProduct R A M
      a₁ : A
      l₁ : L
      ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorProduct.tmul R a₁ l₁ …
    -/
    refine y.induction_on ?_ ?_ ?_
      /-
        case refine_2.refine_1
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing A
        inst✝⁶ : Algebra R A
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y : TensorProduct R A L
        z : TensorProduct R A M
        a₁ : A
        l₁ : L
        ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorProduct.tmul R a₁ l₁ …
      -/
    · simp only [LinearMap.map_zero, add_zero, LinearMap.zero_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing A
        inst✝⁶ : Algebra R A
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y : TensorProduct R A L
        z : TensorProduct R A M
        a₁ : A
        l₁ : L
        ⊢ ∀ (x : A) (y : L), Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorP …
      -/
    · intro a₂ l₂
      /-
        case refine_2.refine_2
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing A
        inst✝⁶ : Algebra R A
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y : TensorProduct R A L
        z : TensorProduct R A M
        a₁ : A
        l₁ : L
        a₂ : A
        l₂ : L
        ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorProduct.tmul R a₁ l₁ …
      -/
      refine z.induction_on ?_ ?_ ?_
      · rw [LinearMap.map_zero, LinearMap.map_zero, LinearMap.map_zero, LinearMap.map_zero,
          add_zero]
        /-
          case refine_2.refine_2.refine_2
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing A
          inst✝⁶ : Algebra R A
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra R L
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          x y : TensorProduct R A L
          z : TensorProduct R A M
          a₁ : A
          l₁ : L
          a₂ : A
          l₂ : L
          ⊢ ∀ (x : A) (y : M), Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorP …
        -/
      · intro a₃ l₃; simp only [bracket'_tmul]
        /-
          case refine_2.refine_2.refine_2
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing A
          inst✝⁶ : Algebra R A
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra R L
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          x y : TensorProduct R A L
          z : TensorProduct R A M
          a₁ : A
          l₁ : L
          a₂ : A
          l₂ : L
          a₃ : A
          l₃ : M
          ⊢ Eq (TensorProduct.tmul R (HMul.hMul a₁ (HMul.hMul a₂ a₃)) (Bracket.bracket l …
        -/
        rw [mul_left_comm a₂ a₁ a₃, mul_assoc, leibniz_lie, TensorProduct.tmul_add]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2.refine_3
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing A
          inst✝⁶ : Algebra R A
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra R L
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          x y : TensorProduct R A L
          z : TensorProduct R A M
          a₁ : A
          l₁ : L
          a₂ : A
          l₂ : L
          ⊢ ∀ (x y : TensorProduct R A M), Eq (((LieAlgebra.ExtendScalars.bracket' R A L …
        -/
      · intro u₁ u₂ h₁ h₂
        /-
          case refine_2.refine_2.refine_3
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing A
          inst✝⁶ : Algebra R A
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra R L
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          x y : TensorProduct R A L
          z : TensorProduct R A M
          a₁ : A
          l₁ : L
          a₂ : A
          l₂ : L
          u₁ u₂ : TensorProduct R A M
          h₁ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorProduct.tmul R a₁ …
          h₂ : Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorProduct.tmul R a₁ …
          ⊢ Eq (((LieAlgebra.ExtendScalars.bracket' R A L M) (TensorProduct.tmul R a₁ l₁ …
        -/
        rw [map_add, map_add, map_add, map_add, map_add, h₁, h₂, add_add_add_comm]
        /-
          🎉 no goals
        -/
      /-
        case refine_2.refine_3
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing A
        inst✝⁶ : Algebra R A
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y : TensorProduct R A L
        z : TensorProduct R A M
        a₁ : A
        l₁ : L
        ⊢ ∀ (x y : TensorProduct R A L), Eq (((LieAlgebra.ExtendScalars.bracket' R A L …
      -/
    · intro u₁ u₂ h₁ h₂
      rw [map_add, LinearMap.add_apply, LinearMap.add_apply, map_add, map_add, map_add,
        LinearMap.add_apply, h₁, h₂, add_add_add_comm]
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing A
      inst✝⁶ : Algebra R A
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y : TensorProduct R A L
      z : TensorProduct R A M
      ⊢ ∀ (x y_1 : TensorProduct R A L), Eq (((LieAlgebra.ExtendScalars.bracket' R A …
    -/
  · intro u₁ u₂ h₁ h₂
    rw [map_add, LinearMap.add_apply, LinearMap.add_apply, map_add, map_add, LinearMap.add_apply,
      map_add, LinearMap.add_apply, h₁, h₂, add_add_add_comm]


instance instLieRing : LieRing (A ⊗[R] L) where
                      /-
                        R : Type u_1
                        A : Type u_2
                        L : Type u_3
                        M : Type u_4
                        inst✝⁸ : CommRing R
                        inst✝⁷ : CommRing A
                        inst✝⁶ : Algebra R A
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        x y z : TensorProduct R A L
                        ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) z) (HAdd.hAdd (Bracket.bracket x z) (Bra …
                      -/
  add_lie x y z := by simp only [bracket_def, LinearMap.add_apply, LinearMap.map_add]
                      /-
                        🎉 no goals
                      -/
                      /-
                        R : Type u_1
                        A : Type u_2
                        L : Type u_3
                        M : Type u_4
                        inst✝⁸ : CommRing R
                        inst✝⁷ : CommRing A
                        inst✝⁶ : Algebra R A
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        x y z : TensorProduct R A L
                        ⊢ Eq (Bracket.bracket x (HAdd.hAdd y z)) (HAdd.hAdd (Bracket.bracket x y) (Bra …
                      -/
  lie_add x y z := by simp only [bracket_def, LinearMap.map_add]
                      /-
                        🎉 no goals
                      -/
  lie_self := bracket_lie_self R A L
  leibniz_lie := bracket_leibniz_lie R A L L


instance instLieAlgebra : LieAlgebra A (A ⊗[R] L) where lie_smul _a _x _y := map_smul _ _ _


instance instLieRingModule : LieRingModule (A ⊗[R] L) (A ⊗[R] M) where
                      /-
                        R : Type u_1
                        A : Type u_2
                        L : Type u_3
                        M : Type u_4
                        inst✝⁸ : CommRing R
                        inst✝⁷ : CommRing A
                        inst✝⁶ : Algebra R A
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        x y : TensorProduct R A L
                        z : TensorProduct R A M
                        ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) z) (HAdd.hAdd (Bracket.bracket x z) (Bra …
                      -/
  add_lie x y z := by simp only [bracket_def, LinearMap.add_apply, LinearMap.map_add]
                      /-
                        🎉 no goals
                      -/
                      /-
                        R : Type u_1
                        A : Type u_2
                        L : Type u_3
                        M : Type u_4
                        inst✝⁸ : CommRing R
                        inst✝⁷ : CommRing A
                        inst✝⁶ : Algebra R A
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        x : TensorProduct R A L
                        y z : TensorProduct R A M
                        ⊢ Eq (Bracket.bracket x (HAdd.hAdd y z)) (HAdd.hAdd (Bracket.bracket x y) (Bra …
                      -/
  lie_add x y z := by simp only [bracket_def, LinearMap.map_add]
                      /-
                        🎉 no goals
                      -/
  leibniz_lie := bracket_leibniz_lie R A L M


instance instLieModule : LieModule A (A ⊗[R] L) (A ⊗[R] M) where
                       /-
                         R : Type u_1
                         A : Type u_2
                         L : Type u_3
                         M : Type u_4
                         inst✝⁸ : CommRing R
                         inst✝⁷ : CommRing A
                         inst✝⁶ : Algebra R A
                         inst✝⁵ : LieRing L
                         inst✝⁴ : LieAlgebra R L
                         inst✝³ : AddCommGroup M
                         inst✝² : Module R M
                         inst✝¹ : LieRingModule L M
                         inst✝ : LieModule R L M
                         t : A
                         x : TensorProduct R A L
                         m : TensorProduct R A M
                         ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
                       -/
  smul_lie t x m := by simp only [bracket_def, map_smul, LinearMap.smul_apply]
                       /-
                         🎉 no goals
                       -/
  lie_smul _ _ _ := map_smul _ _ _


instance : LieRing (RestrictScalars R A L) :=
  h


instance lieAlgebra [CommRing R] [Algebra R A] : LieAlgebra R (RestrictScalars R A L) where
  lie_smul t x y := (lie_smul (algebraMap R A t) (RestrictScalars.addEquiv R A L x)
    (RestrictScalars.addEquiv R A L y) : _)


@[simp]
lemma LieModule.toEnd_baseChange (x : L) :
    toEnd A (A ⊗[R] L) (A ⊗[R] M) (1 ⊗ₜ x) = (toEnd R L M x).baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    x : L
    ⊢ Eq ((LieModule.toEnd A (TensorProduct R A L) (TensorProduct R A M)) (TensorP …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


variable {R L M} in
/-- If `A` is an `R`-algebra, any Lie submodule of a Lie module `M` with coefficients in `R` may be
pushed forward to a Lie submodule of `A ⊗ M` with coefficients in `A`.

This "base change" operation is also known as "extension of scalars". -/
def baseChange : LieSubmodule A (A ⊗[R] L) (A ⊗[R] M) :=
  { (N : Submodule R M).baseChange A with
    lie_mem := by
      /-
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        N : LieSubmodule R L M
        ⊢ ∀ {x : TensorProduct R A L} {m : TensorProduct R A M}, Membership.mem __src✝ …
      -/
      intro x m hm
      simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
        Submodule.mem_toAddSubmonoid] at hm ⊢
      /-
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        N : LieSubmodule R L M
        x : TensorProduct R A L
        m : TensorProduct R A M
        hm : Membership.mem (Submodule.baseChange A ↑N) m
        ⊢ Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket x m)
      -/
      obtain ⟨c, rfl⟩ := (Finsupp.mem_span_iff_linearCombination _ _ _).mp hm
      /-
        case intro
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        N : LieSubmodule R L M
        x : TensorProduct R A L
        c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
        hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
        ⊢ Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket x ((Finsupp.line …
      -/
      refine x.induction_on (by simp) (fun a y ↦ ?_) (fun y z hy hz ↦ ?_)
        /-
          case intro.refine_1
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          N : LieSubmodule R L M
          x : TensorProduct R A L
          c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
          hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
          a : A
          y : L
          ⊢ Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket (TensorProduct.t …
        -/
      · change toEnd A (A ⊗[R] L) (A ⊗[R] M) _ _ ∈ _
        /-
          case intro.refine_1
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          N : LieSubmodule R L M
          x : TensorProduct R A L
          c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
          hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
          a : A
          y : L
          ⊢ Membership.mem (Submodule.baseChange A ↑N) (((LieModule.toEnd A (TensorProdu …
        -/
        simp_rw [Finsupp.linearCombination_apply, Finsupp.sum, map_sum, map_smul, toEnd_apply_apply]
        suffices ∀ n : (N : Submodule R M).map (TensorProduct.mk R A M 1),
            ⁅a ⊗ₜ[R] y, (n : A ⊗[R] M)⁆ ∈ (N : Submodule R M).baseChange A by
          exact Submodule.sum_mem _ fun n _ ↦ Submodule.smul_mem _ _ (this n)
        /-
          case intro.refine_1
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          N : LieSubmodule R L M
          x : TensorProduct R A L
          c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
          hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
          a : A
          y : L
          ⊢ ∀ (n : Subtype fun x => Membership.mem (Submodule.map ((TensorProduct.mk R A …
        -/
        rintro ⟨-, ⟨n : M, hn : n ∈ N, rfl⟩⟩
        /-
          case intro.refine_1.mk.intro.intro
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          N : LieSubmodule R L M
          x : TensorProduct R A L
          c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
          hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
          a : A
          y : L
          n : M
          hn : Membership.mem N n
          ⊢ Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket (TensorProduct.t …
        -/
        exact Submodule.tmul_mem_baseChange_of_mem _ (N.lie_mem hn)
        /-
          🎉 no goals
        -/
        /-
          case intro.refine_2
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          N : LieSubmodule R L M
          x : TensorProduct R A L
          c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
          hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
          y z : TensorProduct R A L
          hy : Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket y ((Finsupp.l …
          hz : Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket z ((Finsupp.l …
          ⊢ Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket (HAdd.hAdd y z)  …
        -/
      · rw [add_lie]
        /-
          case intro.refine_2
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          N : LieSubmodule R L M
          x : TensorProduct R A L
          c : Finsupp (↑↑(Submodule.map ((TensorProduct.mk R A M) 1) ↑N)) A
          hm : Membership.mem (Submodule.baseChange A ↑N) ((Finsupp.linearCombination A  …
          y z : TensorProduct R A L
          hy : Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket y ((Finsupp.l …
          hz : Membership.mem (Submodule.baseChange A ↑N) (Bracket.bracket z ((Finsupp.l …
          ⊢ Membership.mem (Submodule.baseChange A ↑N) (HAdd.hAdd (Bracket.bracket y ((F …
        -/
        exact ((N : Submodule R M).baseChange A).add_mem hy hz }
        /-
          🎉 no goals
        -/


@[simp]
lemma coe_baseChange :
    (N.baseChange A : Submodule A (A ⊗[R] M)) = (N : Submodule R M).baseChange A :=
  rfl


variable {R A L M} in
lemma tmul_mem_baseChange_of_mem (a : A) {m : M} (hm : m ∈ N) :
    a ⊗ₜ[R] m ∈ N.baseChange A :=
  (N : Submodule R M).tmul_mem_baseChange_of_mem a hm


lemma mem_baseChange_iff {m : A ⊗[R] M} :
    m ∈ N.baseChange A ↔
    m ∈ Submodule.span A ((N : Submodule R M).map (TensorProduct.mk R A M 1)) :=
  Iff.rfl


@[simp]
lemma baseChange_bot : (⊥ : LieSubmodule R L M).baseChange A = ⊥ := by
  simp only [baseChange, bot_toSubmodule, Submodule.baseChange_bot,
    Submodule.bot_toAddSubmonoid]
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq { toSubmodule := Bot.bot, lie_mem := ⋯ } Bot.bot
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma baseChange_top : (⊤ : LieSubmodule R L M).baseChange A = ⊤ := by
  simp only [baseChange, top_toSubmodule, Submodule.baseChange_top,
    Submodule.bot_toAddSubmonoid]
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq { toSubmodule := Top.top, lie_mem := ⋯ } Top.top
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma lie_baseChange {I : LieIdeal R L} {N : LieSubmodule R L M} :
    ⁅I, N⁆.baseChange A = ⁅I.baseChange A, N.baseChange A⁆ := by
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    I : LieIdeal R L
    N : LieSubmodule R L M
    ⊢ Eq (LieSubmodule.baseChange A (Bracket.bracket I N)) (Bracket.bracket (LieSu …
  -/
  set s : Set (A ⊗[R] M) := { m | ∃ x ∈ I, ∃ n ∈ N, 1 ⊗ₜ ⁅x, n⁆ = m}
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    I : LieIdeal R L
    N : LieSubmodule R L M
    s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
    ⊢ Eq (LieSubmodule.baseChange A (Bracket.bracket I N)) (Bracket.bracket (LieSu …
  -/
  have : (TensorProduct.mk R A M 1) '' {m | ∃ x ∈ I, ∃ n ∈ N, ⁅x, n⁆ = m} = s := by ext; simp [s]
  rw [← toSubmodule_inj, coe_baseChange, lieIdeal_oper_eq_linear_span',
    Submodule.baseChange_span, this, lieIdeal_oper_eq_linear_span']
  /-
    R : Type u_1
    A : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    I : LieIdeal R L
    N : LieSubmodule R L M
    s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
    this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
    ⊢ Eq (Submodule.span A s) (Submodule.span A (setOf fun m => Exists fun x => An …
  -/
  refine le_antisymm (Submodule.span_mono ?_) (Submodule.span_le.mpr ?_)
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      I : LieIdeal R L
      N : LieSubmodule R L M
      s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
      this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
      ⊢ HasSubset.Subset s (setOf fun m => Exists fun x => And (Membership.mem (LieS …
    -/
  · rintro - ⟨x, hx, m, hm, rfl⟩
    exact ⟨1 ⊗ₜ x, tmul_mem_baseChange_of_mem 1 hx,
           1 ⊗ₜ m, tmul_mem_baseChange_of_mem 1 hm, by simp⟩
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      I : LieIdeal R L
      N : LieSubmodule R L M
      s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
      this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
      ⊢ HasSubset.Subset (setOf fun m => Exists fun x => And (Membership.mem (LieSub …
    -/
  · rintro - ⟨x, hx, m, hm, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro
      R : Type u_1
      A : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      I : LieIdeal R L
      N : LieSubmodule R L M
      s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
      this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
      x : TensorProduct R A L
      hx : Membership.mem (LieSubmodule.baseChange A I) x
      m : TensorProduct R A M
      hm : Membership.mem (LieSubmodule.baseChange A N) m
      ⊢ Membership.mem (↑(Submodule.span A s)) (Bracket.bracket x m)
    -/
    revert m
    apply Submodule.span_induction
      (p := fun x' _ ↦ ∀ m' ∈ N.baseChange A, ⁅x', m'⁆ ∈ Submodule.span A s) (hx := hx)
      /-
        case refine_2.intro.intro.intro.intro.mem
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x : TensorProduct R A L
        hx : Membership.mem (LieSubmodule.baseChange A I) x
        ⊢ ∀ (x : TensorProduct R A L) (h : Membership.mem (↑(Submodule.map ((TensorPro …
      -/
    · rintro _ ⟨y : L, hy : y ∈ I, rfl⟩ m hm
      apply Submodule.span_induction
        (p := fun m' _ ↦ ⁅(1 : A) ⊗ₜ[R] y, m'⁆ ∈ Submodule.span A s) (hx := hm)
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.mem
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          ⊢ ∀ (x : TensorProduct R A M) (h : Membership.mem (↑(Submodule.map ((TensorPro …
        -/
      · rintro - ⟨m', hm' : m' ∈ N, rfl⟩
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.mem.intro.intro
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          m' : M
          hm' : Membership.mem N m'
          ⊢ Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul R 1 …
        -/
        rw [TensorProduct.mk_apply, LieAlgebra.ExtendScalars.bracket_tmul, mul_one]
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.mem.intro.intro
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          m' : M
          hm' : Membership.mem N m'
          ⊢ Membership.mem (Submodule.span A s) (TensorProduct.tmul R 1 (Bracket.bracket …
        -/
        apply Submodule.subset_span
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.mem.intro.intro.a
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          m' : M
          hm' : Membership.mem N m'
          ⊢ Membership.mem s (TensorProduct.tmul R 1 (Bracket.bracket y m'))
        -/
        exact ⟨y, hy, m', hm', rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.zero
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          ⊢ Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul R 1 …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.add
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          ⊢ ∀ (x y_1 : TensorProduct R A M) (hx : Membership.mem (Submodule.span A ↑(Sub …
        -/
      · intro u v _ _ hu hv
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.add
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          u v : TensorProduct R A M
          hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
          hy✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
          hu : Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul  …
          hv : Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul  …
          ⊢ Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul R 1 …
        -/
        rw [lie_add]
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.add
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          u v : TensorProduct R A M
          hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
          hy✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
          hu : Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul  …
          hv : Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul  …
          ⊢ Membership.mem (Submodule.span A s) (HAdd.hAdd (Bracket.bracket (TensorProdu …
        -/
        exact Submodule.add_mem _ hu hv
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.smul
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          ⊢ ∀ (a : A) (x : TensorProduct R A M) (hx : Membership.mem (Submodule.span A ↑ …
        -/
      · intro a u _ hu
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.smul
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          a : A
          u : TensorProduct R A M
          hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
          hu : Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul  …
          ⊢ Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul R 1 …
        -/
        rw [lie_smul]
        /-
          case refine_2.intro.intro.intro.intro.mem.intro.intro.smul
          R : Type u_1
          A : Type u_2
          L : Type u_3
          M : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : LieRing L
          inst✝⁶ : LieAlgebra R L
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : LieModule R L M
          inst✝¹ : CommRing A
          inst✝ : Algebra R A
          I : LieIdeal R L
          N : LieSubmodule R L M
          s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
          this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
          x : TensorProduct R A L
          hx : Membership.mem (LieSubmodule.baseChange A I) x
          y : L
          hy : Membership.mem I y
          m : TensorProduct R A M
          hm : Membership.mem (LieSubmodule.baseChange A N) m
          a : A
          u : TensorProduct R A M
          hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
          hu : Membership.mem (Submodule.span A s) (Bracket.bracket (TensorProduct.tmul  …
          ⊢ Membership.mem (Submodule.span A s) (HSMul.hSMul a (Bracket.bracket (TensorP …
        -/
        exact Submodule.smul_mem _ a hu
        /-
          🎉 no goals
        -/
      /-
        case refine_2.intro.intro.intro.intro.zero
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x : TensorProduct R A L
        hx : Membership.mem (LieSubmodule.baseChange A I) x
        ⊢ ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N) m …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.add
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x : TensorProduct R A L
        hx : Membership.mem (LieSubmodule.baseChange A I) x
        ⊢ ∀ (x y : TensorProduct R A L) (hx : Membership.mem (Submodule.span A ↑(Submo …
      -/
    · intro x y _ _ hx hy m' hm'
      /-
        case refine_2.intro.intro.intro.intro.add
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x✝ : TensorProduct R A L
        hx✝¹ : Membership.mem (LieSubmodule.baseChange A I) x✝
        x y : TensorProduct R A L
        hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
        hy✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
        hx : ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N …
        hy : ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N …
        m' : TensorProduct R A M
        hm' : Membership.mem (LieSubmodule.baseChange A N) m'
        ⊢ Membership.mem (Submodule.span A s) (Bracket.bracket (HAdd.hAdd x y) m')
      -/
      rw [add_lie]
      /-
        case refine_2.intro.intro.intro.intro.add
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x✝ : TensorProduct R A L
        hx✝¹ : Membership.mem (LieSubmodule.baseChange A I) x✝
        x y : TensorProduct R A L
        hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
        hy✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
        hx : ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N …
        hy : ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N …
        m' : TensorProduct R A M
        hm' : Membership.mem (LieSubmodule.baseChange A N) m'
        ⊢ Membership.mem (Submodule.span A s) (HAdd.hAdd (Bracket.bracket x m') (Brack …
      -/
      exact Submodule.add_mem _ (hx _ hm') (hy _ hm')
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.smul
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x : TensorProduct R A L
        hx : Membership.mem (LieSubmodule.baseChange A I) x
        ⊢ ∀ (a : A) (x : TensorProduct R A L) (hx : Membership.mem (Submodule.span A ↑ …
      -/
    · intro a x _ hx m' hm'
      /-
        case refine_2.intro.intro.intro.intro.smul
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x✝ : TensorProduct R A L
        hx✝¹ : Membership.mem (LieSubmodule.baseChange A I) x✝
        a : A
        x : TensorProduct R A L
        hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
        hx : ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N …
        m' : TensorProduct R A M
        hm' : Membership.mem (LieSubmodule.baseChange A N) m'
        ⊢ Membership.mem (Submodule.span A s) (Bracket.bracket (HSMul.hSMul a x) m')
      -/
      rw [smul_lie]
      /-
        case refine_2.intro.intro.intro.intro.smul
        R : Type u_1
        A : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        I : LieIdeal R L
        N : LieSubmodule R L M
        s : Set (TensorProduct R A M) := setOf fun m => Exists fun x => And (Membershi …
        this : Eq (Set.image (⇑((TensorProduct.mk R A M) 1)) (setOf fun m => Exists fu …
        x✝ : TensorProduct R A L
        hx✝¹ : Membership.mem (LieSubmodule.baseChange A I) x✝
        a : A
        x : TensorProduct R A L
        hx✝ : Membership.mem (Submodule.span A ↑(Submodule.map ((TensorProduct.mk R A  …
        hx : ∀ (m' : TensorProduct R A M), Membership.mem (LieSubmodule.baseChange A N …
        m' : TensorProduct R A M
        hm' : Membership.mem (LieSubmodule.baseChange A N) m'
        ⊢ Membership.mem (Submodule.span A s) (HSMul.hSMul a (Bracket.bracket x m'))
      -/
      exact Submodule.smul_mem _ a (hx _ hm')
      /-
        🎉 no goals
      -/


