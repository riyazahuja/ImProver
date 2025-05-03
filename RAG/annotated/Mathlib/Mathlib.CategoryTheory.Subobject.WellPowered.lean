/--
A category (with morphisms in `Type v`) is well-powered relative to a universe `w`
if it is locally small and `Subobject X` is `w`-small for every `X`.

We show in `wellPowered_of_essentiallySmall_monoOver` and `essentiallySmall_monoOver`
that this is the case if and only if `MonoOver X` is `w`-essentially small for every `X`.
-/
@[pp_with_univ]
class WellPowered [LocallySmall.{w} C] : Prop where
  subobject_small : ∀ X : C, Small.{w} (Subobject X) := by infer_instance


instance small_subobject [LocallySmall.{w} C] [WellPowered C] (X : C) :
    Small.{w} (Subobject X) :=
  WellPowered.subobject_small X


instance (priority := 100) wellPowered_of_smallCategory (C : Type u₁) [SmallCategory C] :
    WellPowered.{u₁} C where


theorem essentiallySmall_monoOver_iff_small_subobject (X : C) :
    EssentiallySmall.{w} (MonoOver X) ↔ Small.{w} (Subobject X) :=
  essentiallySmall_iff_of_thin


theorem wellPowered_of_essentiallySmall_monoOver [LocallySmall.{w} C]
    (h : ∀ X : C, EssentiallySmall.{w} (MonoOver X)) :
    WellPowered.{w} C :=
  { subobject_small := fun X => (essentiallySmall_monoOver_iff_small_subobject X).mp (h X) }


instance essentiallySmall_monoOver (X : C) : EssentiallySmall.{w} (MonoOver X) :=
  (essentiallySmall_monoOver_iff_small_subobject X).mpr (WellPowered.subobject_small X)


theorem wellPowered_of_equiv (e : C ≌ D) [LocallySmall.{w} C] [LocallySmall.{w} D]
    [WellPowered.{w} C] : WellPowered.{w} D :=
  wellPowered_of_essentiallySmall_monoOver fun X =>
                                                               /-
                                                                 C : Type u₁
                                                                 inst✝⁴ : CategoryTheory.Category.{v, u₁} C
                                                                 D : Type u₂
                                                                 inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                                 e : CategoryTheory.Equivalence C D
                                                                 inst✝² : CategoryTheory.LocallySmall.{w, v, u₁} C
                                                                 inst✝¹ : CategoryTheory.LocallySmall.{w, v₂, u₂} D
                                                                 inst✝ : CategoryTheory.WellPowered.{w, v, u₁} C
                                                                 X : D
                                                                 ⊢ CategoryTheory.EssentiallySmall.{w, v, max u₁ v} (CategoryTheory.MonoOver (e …
                                                               -/
    (essentiallySmall_congr (MonoOver.congr X e.symm)).2 <| by infer_instance
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- Being well-powered is preserved by equivalences. -/
theorem wellPowered_congr (e : C ≌ D) [LocallySmall.{w} C] [LocallySmall.{w} D] :
    WellPowered.{w} C ↔ WellPowered.{w} D :=
  ⟨fun _ => wellPowered_of_equiv e, fun _ => wellPowered_of_equiv e.symm⟩


instance [LocallySmall.{w} C] [WellPowered.{w} C] :
    WellPowered.{w, w} (ShrinkHoms C) :=
  wellPowered_of_equiv.{w} (ShrinkHoms.equivalence.{w} C)


