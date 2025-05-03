/--
If `C` is an abelian category, we shall say that it satisfies `IsGrothendieckAbelian.{w} C`
if it is locally small (relative to `w`), has exact filtered colimits of size `w` (AB5) and has a
separator.
If `[Category.{v} C]` and `w = v`, this means that `C` satisfies `AB5` and has a separator;
general results about Grothendieck abelian categories can be
reduced to this case using the instance `ShrinkHoms.isGrothendieckAbelian` below.

The introduction of the auxiliary universe `w` shall be needed for certain
applications to categories of sheaves. That the present definition still preserves essential
properties of Grothendieck categories is ensured by `IsGrothendieckAbelian.of_equivalence`,
which shows that every instance for `C` implies an instance for `ShrinkHoms C` with hom sets in
`Type w`.
-/
@[stacks 079B]
class IsGrothendieckAbelian [Abelian C] : Prop where
  locallySmall : LocallySmall.{w} C := by infer_instance
  hasFilteredColimitsOfSize : HasFilteredColimitsOfSize.{w, w} C := by infer_instance
  ab5OfSize : AB5OfSize.{w, w} C := by infer_instance
  hasSeparator : HasSeparator C := by infer_instance


variable {C} {D} in
theorem IsGrothendieckAbelian.of_equivalence [Abelian C] [Abelian D]
    [IsGrothendieckAbelian.{w} C] (α : C ≌ D) : IsGrothendieckAbelian.{w} D := by
  have hasFilteredColimits : HasFilteredColimitsOfSize.{w, w, v₂, u₂} D :=
    ⟨fun _ _ _ => Adjunction.hasColimitsOfShape_of_equivalence α.inverse⟩
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Abelian D
    inst✝ : CategoryTheory.IsGrothendieckAbelian C
    α : CategoryTheory.Equivalence C D
    hasFilteredColimits : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v …
    ⊢ CategoryTheory.IsGrothendieckAbelian D
  -/
  refine ⟨?_, hasFilteredColimits, ?_, ?_⟩
    /-
      case refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Abelian D
      inst✝ : CategoryTheory.IsGrothendieckAbelian C
      α : CategoryTheory.Equivalence C D
      hasFilteredColimits : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v …
      ⊢ CategoryTheory.LocallySmall.{w, v₂, u₂} D
    -/
  · exact locallySmall_of_faithful α.inverse
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Abelian D
      inst✝ : CategoryTheory.IsGrothendieckAbelian C
      α : CategoryTheory.Equivalence C D
      hasFilteredColimits : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v …
      ⊢ CategoryTheory.AB5OfSize.{w, w, v₂, u₂} D
    -/
  · refine ⟨fun _ _ _ => ?_⟩
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Abelian D
      inst✝ : CategoryTheory.IsGrothendieckAbelian C
      α : CategoryTheory.Equivalence C D
      hasFilteredColimits : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v …
      x✝² : Type w
      x✝¹ : CategoryTheory.Category.{w, w} x✝²
      x✝ : CategoryTheory.IsFiltered x✝²
      ⊢ CategoryTheory.HasExactColimitsOfShape x✝² D
    -/
    exact HasExactColimitsOfShape.of_codomain_equivalence _ α
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Abelian D
      inst✝ : CategoryTheory.IsGrothendieckAbelian C
      α : CategoryTheory.Equivalence C D
      hasFilteredColimits : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v …
      ⊢ CategoryTheory.HasSeparator D
    -/
  · exact HasSeparator.of_equivalence α
    /-
      🎉 no goals
    -/


instance ShrinkHoms.isGrothendieckAbelian [Abelian C] [IsGrothendieckAbelian.{w} C] :
    IsGrothendieckAbelian.{w, w} (ShrinkHoms C) :=
  IsGrothendieckAbelian.of_equivalence <| ShrinkHoms.equivalence C


instance IsGrothendieckAbelian.hasColimits : HasColimitsOfSize.{w, w} C :=
  has_colimits_of_finite_and_filtered


instance IsGrothendieckAbelian.hasLimits : HasLimitsOfSize.{w, w} C :=
  have : HasLimits.{w, u} (ShrinkHoms C) := hasLimits_of_hasColimits_of_hasSeparator
  Adjunction.has_limits_of_equivalence (ShrinkHoms.equivalence C |>.functor)


instance IsGrothendieckAbelian.wellPowered : WellPowered.{w} C :=
  wellPowered_of_equiv.{w} (ShrinkHoms.equivalence.{w} C).symm


