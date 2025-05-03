/-- If a functor preserves span and cospan, then it preserves images.
-/
@[simps!]
def iso {X Y : A} (f : X ⟶ Y) : image (L.map f) ≅ L.obj (image f) :=
  let aux1 : StrongEpiMonoFactorisation (L.map f) :=
    { I := L.obj (Limits.image f)
      m := L.map <| Limits.image.ι _
      m_mono := preserves_mono_of_preservesLimit _ _
      e := L.map <| factorThruImage _
      e_strong_epi := @strongEpi_of_epi B _ _ _ _ _ (preserves_epi_of_preservesColimit L _)
                /-
                  A : Type u₁
                  B : Type u₂
                  inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
                  inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
                  inst✝⁵ : CategoryTheory.Limits.HasEqualizers A
                  inst✝⁴ : CategoryTheory.Limits.HasImages A
                  inst✝³ : CategoryTheory.StrongEpiCategory B
                  inst✝² : CategoryTheory.Limits.HasImages B
                  L : CategoryTheory.Functor A B
                  inst✝¹ : ∀ {X Y Z : A} (f : Quiver.Hom X Z) (g : Quiver.Hom Y Z), CategoryTheo …
                  inst✝ : ∀ {X Y Z : A} (f : Quiver.Hom X Y) (g : Quiver.Hom X Z), CategoryTheor …
                  X Y : A
                  f : Quiver.Hom X Y
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Limits.factorT …
                -/
      fac := by rw [← L.map_comp, Limits.image.fac] }
                /-
                  🎉 no goals
                -/
  IsImage.isoExt (Image.isImage (L.map f)) aux1.toMonoIsImage


@[reassoc]
theorem factorThruImage_comp_hom {X Y : A} (f : X ⟶ Y) :
                                                                                /-
                                                                                  A : Type u₁
                                                                                  B : Type u₂
                                                                                  inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
                                                                                  inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
                                                                                  inst✝⁵ : CategoryTheory.Limits.HasEqualizers A
                                                                                  inst✝⁴ : CategoryTheory.Limits.HasImages A
                                                                                  inst✝³ : CategoryTheory.StrongEpiCategory B
                                                                                  inst✝² : CategoryTheory.Limits.HasImages B
                                                                                  L : CategoryTheory.Functor A B
                                                                                  inst✝¹ : ∀ {X Y Z : A} (f : Quiver.Hom X Z) (g : Quiver.Hom Y Z), CategoryTheo …
                                                                                  inst✝ : ∀ {X Y Z : A} (f : Quiver.Hom X Y) (g : Quiver.Hom X Z), CategoryTheor …
                                                                                  X Y : A
                                                                                  f : Quiver.Hom X Y
                                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                                                                                -/
    factorThruImage (L.map f) ≫ (iso L f).hom = L.map (factorThruImage f) := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[reassoc]
theorem hom_comp_map_image_ι {X Y : A} (f : X ⟶ Y) :
                                                                /-
                                                                  A : Type u₁
                                                                  B : Type u₂
                                                                  inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
                                                                  inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
                                                                  inst✝⁵ : CategoryTheory.Limits.HasEqualizers A
                                                                  inst✝⁴ : CategoryTheory.Limits.HasImages A
                                                                  inst✝³ : CategoryTheory.StrongEpiCategory B
                                                                  inst✝² : CategoryTheory.Limits.HasImages B
                                                                  L : CategoryTheory.Functor A B
                                                                  inst✝¹ : ∀ {X Y Z : A} (f : Quiver.Hom X Z) (g : Quiver.Hom Y Z), CategoryTheo …
                                                                  inst✝ : ∀ {X Y Z : A} (f : Quiver.Hom X Y) (g : Quiver.Hom X Z), CategoryTheor …
                                                                  X Y : A
                                                                  f : Quiver.Hom X Y
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.PreservesImage.iso L  …
                                                                -/
    (iso L f).hom ≫ L.map (image.ι f) = image.ι (L.map f) := by rw [iso_hom, image.lift_fac]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
theorem inv_comp_image_ι_map {X Y : A} (f : X ⟶ Y) :
                                                                /-
                                                                  A : Type u₁
                                                                  B : Type u₂
                                                                  inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
                                                                  inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
                                                                  inst✝⁵ : CategoryTheory.Limits.HasEqualizers A
                                                                  inst✝⁴ : CategoryTheory.Limits.HasImages A
                                                                  inst✝³ : CategoryTheory.StrongEpiCategory B
                                                                  inst✝² : CategoryTheory.Limits.HasImages B
                                                                  L : CategoryTheory.Functor A B
                                                                  inst✝¹ : ∀ {X Y Z : A} (f : Quiver.Hom X Z) (g : Quiver.Hom Y Z), CategoryTheo …
                                                                  inst✝ : ∀ {X Y Z : A} (f : Quiver.Hom X Y) (g : Quiver.Hom X Z), CategoryTheor …
                                                                  X Y : A
                                                                  f : Quiver.Hom X Y
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.PreservesImage.iso L  …
                                                                -/
    (iso L f).inv ≫ image.ι (L.map f) = L.map (image.ι f) := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


