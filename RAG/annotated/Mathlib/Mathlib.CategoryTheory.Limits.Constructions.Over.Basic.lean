/-- Make sure we can derive pullbacks in `Over B`. -/
instance {B : C} [HasPullbacks C] : HasPullbacks (Over B) := by
  letI : HasLimitsOfShape (ULiftHom.{v} (ULift.{v} WalkingCospan)) C :=
    hasLimitsOfShape_of_equivalence (ULiftHomULiftCategory.equiv.{v} _)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X B : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    this : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.ULiftHom (ULift. …
    ⊢ CategoryTheory.Limits.HasPullbacks (CategoryTheory.Over B)
  -/
  letI : Category (ULiftHom.{v} (ULift.{v} WalkingCospan)) := inferInstance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X B : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    this✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.ULiftHom (ULift …
    this : CategoryTheory.Category.{v, v} (CategoryTheory.ULiftHom (ULift.{v, 0} C …
    ⊢ CategoryTheory.Limits.HasPullbacks (CategoryTheory.Over B)
  -/
  exact hasLimitsOfShape_of_equivalence (ULiftHomULiftCategory.equiv.{v, v} _).symm
  /-
    🎉 no goals
  -/


/-- Make sure we can derive equalizers in `Over B`. -/
instance {B : C} [HasEqualizers C] : HasEqualizers (Over B) := by
  letI : HasLimitsOfShape (ULiftHom.{v} (ULift.{v} WalkingParallelPair)) C :=
    hasLimitsOfShape_of_equivalence (ULiftHomULiftCategory.equiv.{v} _)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X B : C
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    this : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.ULiftHom (ULift. …
    ⊢ CategoryTheory.Limits.HasEqualizers (CategoryTheory.Over B)
  -/
  letI : Category (ULiftHom.{v} (ULift.{v} WalkingParallelPair)) := inferInstance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X B : C
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    this✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.ULiftHom (ULift …
    this : CategoryTheory.Category.{v, v} (CategoryTheory.ULiftHom (ULift.{v, 0} C …
    ⊢ CategoryTheory.Limits.HasEqualizers (CategoryTheory.Over B)
  -/
  exact hasLimitsOfShape_of_equivalence (ULiftHomULiftCategory.equiv.{v, v} _).symm
  /-
    🎉 no goals
  -/


instance hasFiniteLimits {B : C} [HasFiniteWidePullbacks C] : HasFiniteLimits (Over B) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X B : C
    inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
    ⊢ CategoryTheory.Limits.HasFiniteLimits (CategoryTheory.Over B)
  -/
  apply @hasFiniteLimits_of_hasEqualizers_and_finite_products _ _ ?_ ?_
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
      ⊢ CategoryTheory.Limits.HasFiniteProducts (CategoryTheory.Over B)
    -/
  · exact ConstructProducts.over_finiteProducts_of_finiteWidePullbacks
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
      ⊢ CategoryTheory.Limits.HasEqualizers (CategoryTheory.Over B)
    -/
  · apply @hasEqualizers_of_hasPullbacks_and_binary_products _ _ ?_ _
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
      ⊢ CategoryTheory.Limits.HasBinaryProducts (CategoryTheory.Over B)
    -/
    haveI : HasPullbacks C := ⟨inferInstance⟩
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasFiniteWidePullbacks C
      this : CategoryTheory.Limits.HasPullbacks C
      ⊢ CategoryTheory.Limits.HasBinaryProducts (CategoryTheory.Over B)
    -/
    exact ConstructProducts.over_binaryProduct_of_pullback
    /-
      🎉 no goals
    -/


instance hasLimits {B : C} [HasWidePullbacks.{w} C] : HasLimitsOfSize.{w, w} (Over B) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X B : C
    inst✝ : CategoryTheory.Limits.HasWidePullbacks C
    ⊢ CategoryTheory.Limits.HasLimitsOfSize.{w, w, v, max u v} (CategoryTheory.Ove …
  -/
  apply @has_limits_of_hasEqualizers_and_products _ _ ?_ ?_
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      ⊢ CategoryTheory.Limits.HasProducts (CategoryTheory.Over B)
    -/
  · exact ConstructProducts.over_products_of_widePullbacks
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      ⊢ CategoryTheory.Limits.HasEqualizers (CategoryTheory.Over B)
    -/
  · apply @hasEqualizers_of_hasPullbacks_and_binary_products _ _ ?_ _
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      ⊢ CategoryTheory.Limits.HasBinaryProducts (CategoryTheory.Over B)
    -/
    haveI : HasPullbacks C := ⟨inferInstance⟩
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X B : C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      this : CategoryTheory.Limits.HasPullbacks C
      ⊢ CategoryTheory.Limits.HasBinaryProducts (CategoryTheory.Over B)
    -/
    exact ConstructProducts.over_binaryProduct_of_pullback
    /-
      🎉 no goals
    -/


