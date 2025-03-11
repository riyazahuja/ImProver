open List in
theorem effectiveEpi_tfae
    {B X : CompHaus.{u}} (π : X ⟶ B) :
    TFAE
    [ EffectiveEpi π
    , Epi π
    , Function.Surjective π
    ] := by
  /-
    B X : CompHaus
    π : Quiver.Hom X B
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_have 1 → 2 := fun _ ↦ inferInstance
  /-
    B X : CompHaus
    π : Quiver.Hom X B
    tfae_1_to_2 : CategoryTheory.EffectiveEpi π → CategoryTheory.Epi π
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_have 2 ↔ 3 := epi_iff_surjective π
  /-
    B X : CompHaus
    π : Quiver.Hom X B
    tfae_1_to_2 : CategoryTheory.EffectiveEpi π → CategoryTheory.Epi π
    tfae_2_iff_3 : Iff (CategoryTheory.Epi π) (Function.Surjective ⇑π)
    ⊢ (List.cons (CategoryTheory.EffectiveEpi π) (List.cons (CategoryTheory.Epi π) …
  -/
  tfae_have 3 → 1 := fun hπ ↦ ⟨⟨effectiveEpiStruct π hπ⟩⟩
  /-
    B X : CompHaus
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


instance : Preregular CompHaus :=
                          /-
                            x✝² x✝¹ : CompHausLike fun x => True
                            x✝ : Quiver.Hom x✝² x✝¹
                            ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (?m.964 x✝² x✝¹ x✝)) (List.cons  …
                          -/
                          /-
                            🎉 no goals
                          -/
  preregular fun _ _ _ ↦ ((effectiveEpi_tfae _).out 0 2).mp
                          /-
                            🎉 no goals
                          -/


open List in
theorem effectiveEpiFamily_tfae
    {α : Type} [Finite α] {B : CompHaus.{u}}
    (X : α → CompHaus.{u}) (π : (a : α) → (X a ⟶ B)) :
    TFAE
    [ EffectiveEpiFamily X π
    , Epi (Sigma.desc π)
    , ∀ b : B, ∃ (a : α) (x : X a), π a x = b
    ] := by
  tfae_have 2 → 1
  | _ => by
    simpa [← effectiveEpi_desc_iff_effectiveEpiFamily, (effectiveEpi_tfae (Sigma.desc π)).out 0 1]
  tfae_have 1 → 2
  | _ => inferInstance
  tfae_have 3 → 2
  | e => by
    rw [epi_iff_surjective]
    intro b
    obtain ⟨t, x, h⟩ := e b
    refine ⟨Sigma.ι X t x, ?_⟩
    change (Sigma.ι X t ≫ Sigma.desc π) x = _
    simpa using h
  tfae_have 2 → 3
  | e => by
    rw [epi_iff_surjective] at e
    let i : ∐ X ≅ finiteCoproduct X :=
      (colimit.isColimit _).coconePointUniqueUpToIso (finiteCoproduct.isColimit _)
    intro b
    obtain ⟨t, rfl⟩ := e b
    let q := i.hom t
    refine ⟨q.1,q.2,?_⟩
    have : t = i.inv (i.hom t) := show t = (i.hom ≫ i.inv) t by simp only [i.hom_inv_id]; rfl
    rw [this]
    show _ = (i.inv ≫ Sigma.desc π) (i.hom t)
    suffices i.inv ≫ Sigma.desc π = finiteCoproduct.desc X π by
      rw [this]; rfl
    rw [Iso.inv_comp_eq]
    apply colimit.hom_ext
    rintro ⟨a⟩
    simp only [i, Discrete.functor_obj, colimit.ι_desc, Cofan.mk_pt, Cofan.mk_ι_app,
      colimit.comp_coconePointUniqueUpToIso_hom_assoc]
    ext; rfl
  /-
    α : Type
    inst✝ : Finite α
    B : CompHaus
    X : α → CompHaus
    π : (a : α) → Quiver.Hom (X a) B
    tfae_2_to_1 : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc π) → Catego …
    tfae_1_to_2 : CategoryTheory.EffectiveEpiFamily X π → CategoryTheory.Epi (Cate …
    tfae_3_to_2 : (∀ (b : ↑B.toTop), Exists fun a => Exists fun x => Eq ((π a) x)  …
    tfae_2_to_3 : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc π) → ∀ (b : …
    ⊢ (List.cons (CategoryTheory.EffectiveEpiFamily X π) (List.cons (CategoryTheor …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem effectiveEpiFamily_of_jointly_surjective
    {α : Type} [Finite α] {B : CompHaus.{u}}
    (X : α → CompHaus.{u}) (π : (a : α) → (X a ⟶ B))
    (surj : ∀ b : B, ∃ (a : α) (x : X a), π a x = b) :
    EffectiveEpiFamily X π :=
   /-
     α : Type
     inst✝ : Finite α
     B : CompHaus
     X : α → CompHaus
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


