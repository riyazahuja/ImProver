/-- The inclusion of locally-constant functions into continuous functions as a multiplicative
monoid hom. -/
@[to_additive (attr := simps)
"The inclusion of locally-constant functions into continuous functions as an additive monoid hom."]
def toContinuousMapMonoidHom [Monoid Y] [ContinuousMul Y] : LocallyConstant X Y →* C(X, Y) where
  toFun := (↑)
  map_one' := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : Monoid Y
      inst✝ : ContinuousMul Y
      ⊢ Eq (↑1) 1
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : Monoid Y
      inst✝ : ContinuousMul Y
      a✝ : X
      ⊢ Eq (↑1 a✝) (1 a✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : Monoid Y
      inst✝ : ContinuousMul Y
      x y : LocallyConstant X Y
      ⊢ Eq ({ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯ }.toFun (HMul. …
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : Monoid Y
      inst✝ : ContinuousMul Y
      x y : LocallyConstant X Y
      a✝ : X
      ⊢ Eq (({ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯ }.toFun (HMul …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The inclusion of locally-constant functions into continuous functions as a linear map. -/
@[simps]
def toContinuousMapLinearMap (R : Type*) [Semiring R] [AddCommMonoid Y] [Module R Y]
    [ContinuousAdd Y] [ContinuousConstSMul R Y] : LocallyConstant X Y →ₗ[R] C(X, Y) where
  toFun := (↑)
  map_add' x y := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      R : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid Y
      inst✝² : Module R Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : ContinuousConstSMul R Y
      x y : LocallyConstant X Y
      ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd ↑x ↑y)
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      R : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid Y
      inst✝² : Module R Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : ContinuousConstSMul R Y
      x y : LocallyConstant X Y
      a✝ : X
      ⊢ Eq (↑(HAdd.hAdd x y) a✝) ((HAdd.hAdd ↑x ↑y) a✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  map_smul' x y := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      R : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid Y
      inst✝² : Module R Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : ContinuousConstSMul R Y
      x : R
      y : LocallyConstant X Y
      ⊢ Eq ({ toFun := LocallyConstant.toContinuousMap, map_add' := ⋯ }.toFun (HSMul …
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      R : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid Y
      inst✝² : Module R Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : ContinuousConstSMul R Y
      x : R
      y : LocallyConstant X Y
      a✝ : X
      ⊢ Eq (({ toFun := LocallyConstant.toContinuousMap, map_add' := ⋯ }.toFun (HSMu …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The inclusion of locally-constant functions into continuous functions as an algebra map. -/
@[simps]
def toContinuousMapAlgHom (R : Type*) [CommSemiring R] [Semiring Y] [Algebra R Y]
    [TopologicalSemiring Y] : LocallyConstant X Y →ₐ[R] C(X, Y) where
  toFun := (↑)
  map_one' := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      ⊢ Eq (↑1) 1
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      a✝ : X
      ⊢ Eq (↑1 a✝) (1 a✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      x y : LocallyConstant X Y
      ⊢ Eq ({ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯ }.toFun (HMul. …
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      x y : LocallyConstant X Y
      a✝ : X
      ⊢ Eq (({ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯ }.toFun (HMul …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_zero' := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      ⊢ Eq ((↑{ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯, map_mul' := …
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      a✝ : X
      ⊢ Eq (((↑{ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯, map_mul' : …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' x y := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      x y : LocallyConstant X Y
      ⊢ Eq ((↑{ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯, map_mul' := …
    -/
    ext
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      x y : LocallyConstant X Y
      a✝ : X
      ⊢ Eq (((↑{ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯, map_mul' : …
    -/
    simp
    /-
      🎉 no goals
    -/
  commutes' r := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      r : R
      ⊢ Eq ((↑↑{ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯, map_mul' : …
    -/
    ext x
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      R : Type u_3
      inst✝³ : CommSemiring R
      inst✝² : Semiring Y
      inst✝¹ : Algebra R Y
      inst✝ : TopologicalSemiring Y
      r : R
      x : X
      ⊢ Eq (((↑↑{ toFun := LocallyConstant.toContinuousMap, map_one' := ⋯, map_mul'  …
    -/
    simp [Algebra.smul_def]
    /-
      🎉 no goals
    -/


