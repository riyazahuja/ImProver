/-- A morphism is proper if it is separated, universally closed and locally of finite type. -/
@[mk_iff]
class IsProper extends IsSeparated f, UniversallyClosed f, LocallyOfFiniteType f : Prop where


lemma isProper_eq : @IsProper =
    (@IsSeparated ⊓ @UniversallyClosed : MorphismProperty Scheme) ⊓ @LocallyOfFiniteType := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsProper) (Min.min (Min.min @AlgebraicGeometry.IsSepa …
  -/
  ext X Y f
  /-
    case h.h.h.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.IsProper f) (Min.min (Min.min @AlgebraicGeometry.IsSe …
  -/
  rw [isProper_iff, ← and_assoc]
  /-
    case h.h.h.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (And (And (AlgebraicGeometry.IsSeparated f) (AlgebraicGeometry.Universal …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : MorphismProperty.RespectsIso @IsProper := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.MorphismProperty.RespectsIso @AlgebraicGeometry.IsProper
  -/
  rw [isProper_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ (Min.min (Min.min @AlgebraicGeometry.IsSeparated @AlgebraicGeometry.Universa …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance stableUnderComposition : MorphismProperty.IsStableUnderComposition @IsProper := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderComposition @AlgebraicGeometry. …
  -/
  rw [isProper_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ (Min.min (Min.min @AlgebraicGeometry.IsSeparated @AlgebraicGeometry.Universa …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : MorphismProperty.IsMultiplicative @IsProper := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.MorphismProperty.IsMultiplicative @AlgebraicGeometry.IsProper
  -/
  rw [isProper_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ (Min.min (Min.min @AlgebraicGeometry.IsSeparated @AlgebraicGeometry.Universa …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (priority := 900) [IsClosedImmersion f] : IsProper f where


instance isStableUnderBaseChange : MorphismProperty.IsStableUnderBaseChange @IsProper := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.I …
  -/
  rw [isProper_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ (Min.min (Min.min @AlgebraicGeometry.IsSeparated @AlgebraicGeometry.Universa …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : IsLocalAtTarget @IsProper := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsLocalAtTarget @AlgebraicGeometry.IsProper
  -/
  rw [isProper_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsLocalAtTarget (Min.min (Min.min @AlgebraicGeometry.IsSep …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


