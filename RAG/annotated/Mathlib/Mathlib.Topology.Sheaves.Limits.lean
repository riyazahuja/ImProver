instance [HasLimits C] (X : TopCat.{v}) : HasLimits.{v} (Presheaf C X) :=
  Limits.functorCategoryHasLimitsOfSize.{v, v}


instance [HasColimits.{v, u} C] (X : TopCat.{w}) : HasColimitsOfSize.{v, v} (Presheaf C X) :=
  Limits.functorCategoryHasColimitsOfSize


instance [HasLimits C] (X : TopCat) : CreatesLimits.{v, v} (Sheaf.forget C X) :=
  Sheaf.createsLimits.{u, v, v}


instance [HasLimits C] (X : TopCat.{v}) : HasLimitsOfSize.{v, v} (Sheaf.{v} C X) :=
  hasLimits_of_hasLimits_createsLimits (Sheaf.forget C X)


theorem isSheaf_of_isLimit [HasLimits C] {X : TopCat} (F : J ⥤ Presheaf.{v} C X)
    (H : ∀ j, (F.obj j).IsSheaf) {c : Cone F} (hc : IsLimit c) : c.pt.IsSheaf := by
  let F' : J ⥤ Sheaf C X :=
    { obj := fun j => ⟨F.obj j, H j⟩
      map := fun f => ⟨F.map f⟩ }
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimits C
    X : TopCat
    F : CategoryTheory.Functor J (TopCat.Presheaf C X)
    H : ∀ (j : J), (F.obj j).IsSheaf
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    F' : CategoryTheory.Functor J (TopCat.Sheaf C X) := { obj := fun j => { val := …
    ⊢ c.pt.IsSheaf
  -/
  let e : F' ⋙ Sheaf.forget C X ≅ F := NatIso.ofComponents fun _ => Iso.refl _
  exact Presheaf.isSheaf_of_iso
    ((isLimitOfPreserves (Sheaf.forget C X) (limit.isLimit F')).conePointsIsoOfNatIso hc e)
    (limit F').2


theorem limit_isSheaf [HasLimits C] {X : TopCat} (F : J ⥤ Presheaf.{v} C X)
    (H : ∀ j, (F.obj j).IsSheaf) : (limit F).IsSheaf :=
  isSheaf_of_isLimit F H (limit.isLimit F)


