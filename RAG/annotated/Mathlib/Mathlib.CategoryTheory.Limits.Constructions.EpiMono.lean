/-- If `F` preserves pullbacks, then it preserves monomorphisms. -/
theorem preserves_mono_of_preservesLimit {X Y : C} (f : X ⟶ Y) [PreservesLimit (cospan f f) F]
    [Mono f] : Mono (F.map f) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.Mono (F.map f)
  -/
  have := isLimitPullbackConeMapOfIsLimit F _ (PullbackCone.isLimitMkIdId f)
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝ : CategoryTheory.Mono f
    this : letFun ⋯ fun this => CategoryTheory.Limits.IsLimit (CategoryTheory.Limi …
    ⊢ CategoryTheory.Mono (F.map f)
  -/
  simp_rw [F.map_id] at this
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f  …
    inst✝ : CategoryTheory.Mono f
    this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (C …
    ⊢ CategoryTheory.Mono (F.map f)
  -/
  apply PullbackCone.mono_of_isLimitMkIdId _ this
  /-
    🎉 no goals
  -/


instance (priority := 100) preservesMonomorphisms_of_preservesLimitsOfShape
    [PreservesLimitsOfShape WalkingCospan F] : F.PreservesMonomorphisms where
  preserves f _ := preserves_mono_of_preservesLimit F f


/-- If `F` reflects pullbacks, then it reflects monomorphisms. -/
theorem reflects_mono_of_reflectsLimit {X Y : C} (f : X ⟶ Y) [ReflectsLimit (cospan f f) F]
    [Mono (F.map f)] : Mono f := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan f f …
    inst✝ : CategoryTheory.Mono (F.map f)
    ⊢ CategoryTheory.Mono f
  -/
  have := PullbackCone.isLimitMkIdId (F.map f)
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan f f …
    inst✝ : CategoryTheory.Mono (F.map f)
    this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (C …
    ⊢ CategoryTheory.Mono f
  -/
  simp_rw [← F.map_id] at this
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan f f …
    inst✝ : CategoryTheory.Mono (F.map f)
    this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (F …
    ⊢ CategoryTheory.Mono f
  -/
  apply PullbackCone.mono_of_isLimitMkIdId _ (isLimitOfIsLimitPullbackConeMap F _ this)
  /-
    🎉 no goals
  -/


instance (priority := 100) reflectsMonomorphisms_of_reflectsLimitsOfShape
    [ReflectsLimitsOfShape WalkingCospan F] : F.ReflectsMonomorphisms where
  reflects f _ := reflects_mono_of_reflectsLimit F f


/-- If `F` preserves pushouts, then it preserves epimorphisms. -/
theorem preserves_epi_of_preservesColimit {X Y : C} (f : X ⟶ Y) [PreservesColimit (span f f) F]
    [Epi f] : Epi (F.map f) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.Epi (F.map f)
  -/
  have := isColimitPushoutCoconeMapOfIsColimit F _ (PushoutCocone.isColimitMkIdId f)
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝ : CategoryTheory.Epi f
    this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk …
    ⊢ CategoryTheory.Epi (F.map f)
  -/
  simp_rw [F.map_id] at this
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.span f  …
    inst✝ : CategoryTheory.Epi f
    this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk …
    ⊢ CategoryTheory.Epi (F.map f)
  -/
  apply PushoutCocone.epi_of_isColimitMkIdId _ this
  /-
    🎉 no goals
  -/


instance (priority := 100) preservesEpimorphisms_of_preservesColimitsOfShape
    [PreservesColimitsOfShape WalkingSpan F] : F.PreservesEpimorphisms where
  preserves f _ := preserves_epi_of_preservesColimit F f


/-- If `F` reflects pushouts, then it reflects epimorphisms. -/
theorem reflects_epi_of_reflectsColimit {X Y : C} (f : X ⟶ Y) [ReflectsColimit (span f f) F]
    [Epi (F.map f)] : Epi f := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Limits.span f f …
    inst✝ : CategoryTheory.Epi (F.map f)
    ⊢ CategoryTheory.Epi f
  -/
  have := PushoutCocone.isColimitMkIdId (F.map f)
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Limits.span f f …
    inst✝ : CategoryTheory.Epi (F.map f)
    this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk …
    ⊢ CategoryTheory.Epi f
  -/
  simp_rw [← F.map_id] at this
  apply
    PushoutCocone.epi_of_isColimitMkIdId _
      (isColimitOfIsColimitPushoutCoconeMap F _ this)


instance (priority := 100) reflectsEpimorphisms_of_reflectsColimitsOfShape
    [ReflectsColimitsOfShape WalkingSpan F] : F.ReflectsEpimorphisms where
  reflects f _ := reflects_epi_of_reflectsColimit F f


