instance [HasFiniteLimits A] [HasColimitsOfShape K A] [HasExactColimitsOfShape K A]
    [PreservesColimitsOfShape K (sheafToPresheaf J A)] : HasExactColimitsOfShape K (Sheaf J A) :=
  HasExactColimitsOfShape.domain_of_functor K (sheafToPresheaf J A)


instance [HasFiniteColimits A] [HasLimitsOfShape K A] [HasExactLimitsOfShape K A]
    [PreservesFiniteColimits (sheafToPresheaf J A)] : HasExactLimitsOfShape K (Sheaf J A) :=
  HasExactLimitsOfShape.domain_of_functor K (sheafToPresheaf J A)


instance hasFilteredColimitsOfSize
    [HasSheafify J A] [HasFilteredColimitsOfSize.{v₂, u₂} A] :
    HasFilteredColimitsOfSize.{v₂, u₂} (Sheaf J A) where
                             /-
                               C : Type u
                               A : Type u₁
                               K✝ : Type u₂
                               inst✝⁴ : CategoryTheory.Category.{v, u} C
                               inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                               inst✝² : CategoryTheory.Category.{v₂, u₂} K✝
                               J : CategoryTheory.GrothendieckTopology C
                               inst✝¹ : CategoryTheory.HasSheafify J A
                               inst✝ : CategoryTheory.Limits.HasFilteredColimitsOfSize.{v₂, u₂, v₁, u₁} A
                               K : Type u₂
                               ⊢ ∀ [inst : CategoryTheory.Category.{v₂, u₂} K] [inst_1 : CategoryTheory.IsFil …
                             -/
  HasColimitsOfShape K := by infer_instance
                             /-
                               🎉 no goals
                             -/


instance hasExactColimitsOfShape [HasFiniteLimits A] [HasSheafify J A]
    [HasColimitsOfShape K A] [HasExactColimitsOfShape K A] :
    HasExactColimitsOfShape K (Sheaf J A) :=
  (sheafificationAdjunction J A).hasExactColimitsOfShape K


instance ab5ofSize [HasFiniteLimits A] [HasSheafify J A]
    [HasFilteredColimitsOfSize.{v₂, u₂} A] [AB5OfSize.{v₂, u₂} A] :
    AB5OfSize.{v₂, u₂} (Sheaf J A) where
                      /-
                        C : Type u
                        A : Type u₁
                        K✝ : Type u₂
                        inst✝⁶ : CategoryTheory.Category.{v, u} C
                        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K✝
                        J : CategoryTheory.GrothendieckTopology C
                        inst✝³ : CategoryTheory.Limits.HasFiniteLimits A
                        inst✝² : CategoryTheory.HasSheafify J A
                        inst✝¹ : CategoryTheory.Limits.HasFilteredColimitsOfSize.{v₂, u₂, v₁, u₁} A
                        inst✝ : CategoryTheory.AB5OfSize.{v₂, u₂, v₁, u₁} A
                        K : Type u₂
                        x✝¹ : CategoryTheory.Category.{v₂, u₂} K
                        x✝ : CategoryTheory.IsFiltered K
                        ⊢ CategoryTheory.HasExactColimitsOfShape K (CategoryTheory.Sheaf J A)
                      -/
  ofShape K _ _ := by infer_instance
                      /-
                        🎉 no goals
                      -/


instance {C : Type v} [SmallCategory.{v} C] (J : GrothendieckTopology C)
    (A : Type u₁) [Category.{v} A] [Abelian A] [IsGrothendieckAbelian.{v} A]
    [HasSheafify J A] : IsGrothendieckAbelian.{v} (Sheaf J A) where


