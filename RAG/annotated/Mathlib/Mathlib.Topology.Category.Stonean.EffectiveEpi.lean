open List in
theorem effectiveEpi_tfae
    {B X : Stonean.{u}} (π : X ⟶ B) :
    TFAE
    [ EffectiveEpi π
    , Epi π
    , Function.Surjective π
    ] := by
  /-
    B X : Stonean
    π : Quiver.Hom X B
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_have 1 → 2 := fun _ ↦ inferInstance
  /-
    B X : Stonean
    π : Quiver.Hom X B
    tfae_1_to_2 : CategoryTheory.EffectiveEpi π → CategoryTheory.Epi π
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_have 2 ↔ 3 := epi_iff_surjective π
  /-
    B X : Stonean
    π : Quiver.Hom X B
    tfae_1_to_2 : CategoryTheory.EffectiveEpi π → CategoryTheory.Epi π
    tfae_2_iff_3 : Iff (CategoryTheory.Epi π) (Function.Surjective ⇑π)
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_have 3 → 1 := fun hπ ↦ ⟨⟨effectiveEpiStruct π hπ⟩⟩
  /-
    B X : Stonean
    π : Quiver.Hom X B
    tfae_1_to_2 : CategoryTheory.EffectiveEpi π → CategoryTheory.Epi π
    tfae_2_iff_3 : Iff (CategoryTheory.Epi π) (Function.Surjective ⇑π)
    tfae_3_to_1 : Function.Surjective ⇑π → CategoryTheory.EffectiveEpi π
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


instance : Stonean.toCompHaus.PreservesEffectiveEpis where
  preserves f h :=
     /-
       X✝ Y✝ : Stonean
       f : Quiver.Hom X✝ Y✝
       h : CategoryTheory.EffectiveEpi f
       ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (Stonean.toCompHaus.map f)) (Lis …
     -/
     /-
       🎉 no goals
     -/
    ((CompHaus.effectiveEpi_tfae (Stonean.toCompHaus.map f)).out 0 2).mpr
     /-
       🎉 no goals
     -/
        /-
          X✝ Y✝ : Stonean
          f : Quiver.Hom X✝ Y✝
          h : CategoryTheory.EffectiveEpi f
          ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi f) (List.cons (CategoryTheory.Ep …
        -/
        /-
          🎉 no goals
        -/
      (((Stonean.effectiveEpi_tfae f).out 0 2).mp h)
        /-
          🎉 no goals
        -/


instance : Stonean.toCompHaus.ReflectsEffectiveEpis where
  reflects f h :=
     /-
       X✝ Y✝ : Stonean
       f : Quiver.Hom X✝ Y✝
       h : CategoryTheory.EffectiveEpi (Stonean.toCompHaus.map f)
       ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi f) (List.cons (CategoryTheory.Ep …
     -/
     /-
       🎉 no goals
     -/
    ((Stonean.effectiveEpi_tfae f).out 0 2).mpr
     /-
       🎉 no goals
     -/
        /-
          X✝ Y✝ : Stonean
          f : Quiver.Hom X✝ Y✝
          h : CategoryTheory.EffectiveEpi (Stonean.toCompHaus.map f)
          ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (Stonean.toCompHaus.map f)) (Lis …
        -/
        /-
          🎉 no goals
        -/
      (((CompHaus.effectiveEpi_tfae (Stonean.toCompHaus.map f)).out 0 2).mp h)
        /-
          🎉 no goals
        -/


/--
An effective presentation of an `X : CompHaus` with respect to the inclusion functor from `Stonean`
-/
noncomputable def stoneanToCompHausEffectivePresentation (X : CompHaus) :
    Stonean.toCompHaus.EffectivePresentation X where
  p := X.presentation
  f := CompHaus.presentation.π X
                   /-
                     X : CompHaus
                     ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi ?m.1854) (List.cons (CategoryThe …
                   -/
                   /-
                     🎉 no goals
                   -/
  effectiveEpi := ((CompHaus.effectiveEpi_tfae _).out 0 1).mpr (inferInstance : Epi _)
                   /-
                     🎉 no goals
                   -/


instance : Stonean.toCompHaus.EffectivelyEnough where
  presentation X := ⟨stoneanToCompHausEffectivePresentation X⟩


instance : Preregular Stonean := Stonean.toCompHaus.reflects_preregular


open List in
theorem effectiveEpiFamily_tfae
    {α : Type} [Finite α] {B : Stonean.{u}}
    (X : α → Stonean.{u}) (π : (a : α) → (X a ⟶ B)) :
    TFAE
    [ EffectiveEpiFamily X π
    , Epi (Sigma.desc π)
    , ∀ b : B, ∃ (a : α) (x : X a), π a x = b
    ] := by
  tfae_have 2 → 1
  | _ => by
    simpa [← effectiveEpi_desc_iff_effectiveEpiFamily, (effectiveEpi_tfae (Sigma.desc π)).out 0 1]
  /-
    α : Type
    inst✝ : Finite α
    B : Stonean
    X : α → Stonean
    π : (a : α) → Quiver.Hom (X a) B
    tfae_2_to_1 : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc π) → Catego …
    ⊢ (List.cons (CategoryTheory.EffectiveEpiFamily X π) (List.cons (CategoryTheor …
  -/
  tfae_have 1 → 2 := fun _ ↦ inferInstance
  tfae_have 3 ↔ 1 := by
    erw [((CompHaus.effectiveEpiFamily_tfae
      (fun a ↦ Stonean.toCompHaus.obj (X a)) (fun a ↦ Stonean.toCompHaus.map (π a))).out 2 0 : )]
    exact ⟨fun h ↦ Stonean.toCompHaus.finite_effectiveEpiFamily_of_map _ _ h,
      fun _ ↦ inferInstance⟩
  /-
    α : Type
    inst✝ : Finite α
    B : Stonean
    X : α → Stonean
    π : (a : α) → Quiver.Hom (X a) B
    tfae_2_to_1 : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc π) → Catego …
    tfae_1_to_2 : CategoryTheory.EffectiveEpiFamily X π → CategoryTheory.Epi (Cate …
    tfae_3_iff_1 : Iff (∀ (b : ↑B.toTop), Exists fun a => Exists fun x => Eq ((π a …
    ⊢ (List.cons (CategoryTheory.EffectiveEpiFamily X π) (List.cons (CategoryTheor …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem effectiveEpiFamily_of_jointly_surjective
    {α : Type} [Finite α] {B : Stonean.{u}}
    (X : α → Stonean.{u}) (π : (a : α) → (X a ⟶ B))
    (surj : ∀ b : B, ∃ (a : α) (x : X a), π a x = b) :
    EffectiveEpiFamily X π :=
   /-
     α : Type
     inst✝ : Finite α
     B : Stonean
     X : α → Stonean
     π : (a : α) → Quiver.Hom (X a) B
     surj : ∀ (b : ↑B.toTop), Exists fun a => Exists fun x => Eq ((π a) x) b
     ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpiFamily X π) (List.cons (CategoryT …
   -/
   /-
     🎉 no goals
   -/
  ((effectiveEpiFamily_tfae X π).out 2 0).mp surj
   /-
     🎉 no goals
   -/


