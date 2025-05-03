/-- A cocone in the category `PresheafOfModules R` is colimit if it is so after the application
of the functors `evaluation R X` for all `X`. -/
def evaluationJointlyReflectsColimits (c : Cocone F)
    (hc : ∀ (X : Cᵒᵖ), IsColimit ((evaluation R X).mapCocone c)) : IsColimit c where
  desc s :=
    { app := fun X => (hc X).desc ((evaluation R X).mapCocone s)
      naturality := fun {X Y} f ↦ (hc X).hom_ext (fun j ↦ by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
          c : CategoryTheory.Limits.Cocone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
          s : CategoryTheory.Limits.Cocone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.evaluation R X). …
        -/
        rw [(hc X).fac_assoc ((evaluation R X).mapCocone s) j]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
          c : CategoryTheory.Limits.Cocone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
          s : CategoryTheory.Limits.Cocone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.evaluation R X). …
        -/
        have h₁ := (c.ι.app j).naturality f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
          c : CategoryTheory.Limits.Cocone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
          s : CategoryTheory.Limits.Cocone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((ModuleCat.rest …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.evaluation R X). …
        -/
        have h₂ := (hc Y).fac ((evaluation R Y).mapCocone s)
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
          c : CategoryTheory.Limits.Cocone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
          s : CategoryTheory.Limits.Cocone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((ModuleCat.rest …
          h₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.ev …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.evaluation R X). …
        -/
        dsimp at h₁ h₂ ⊢
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
          c : CategoryTheory.Limits.Cocone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
          s : CategoryTheory.Limits.Cocone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((ModuleCat.rest …
          h₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((c.ι.app j).app Y) ((h …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c.ι.app j).app X) (CategoryTheory.C …
        -/
        simp only [← reassoc_of% h₁, ← Functor.map_comp, h₂, Hom.naturality]) }
        /-
          🎉 no goals
        -/
  fac s j := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { app := fun X …
    -/
    ext1 X
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      j : J
      X : Opposite C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { app := fun  …
    -/
    exact (hc X).fac ((evaluation R X).mapCocone s) j
    /-
      🎉 no goals
    -/
  uniq s m hm := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      ⊢ Eq m ((fun s => { app := fun X => (hc X).desc ((PresheafOfModules.evaluation …
    -/
    ext1 X
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      X : Opposite C
      ⊢ Eq (m.app X) (((fun s => { app := fun X => (hc X).desc ((PresheafOfModules.e …
    -/
    apply (hc X).uniq ((evaluation R X).mapCocone s)
    /-
      case h.x
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      X : Opposite C
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.evalu …
    -/
    intro j
    /-
      case h.x
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((PresheafOfModules.evaluation R X). …
    -/
    dsimp
    /-
      case h.x
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c.ι.app j).app X) (m.app X)) ((s.ι. …
    -/
    rw [← hm]
    /-
      case h.x
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Prese …
      c : CategoryTheory.Limits.Cocone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsColimit ((PresheafOfModules.ev …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c.ι.app j).app X) (m.app X)) ((Cate …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    HasColimit (F ⋙ evaluation R Y ⋙ (ModuleCat.restrictScalars (R.map f).hom)) :=
  ⟨_, isColimitOfPreserves (ModuleCat.restrictScalars (R.map f).hom)
    (colimit.isColimit (F ⋙ evaluation R Y))⟩


/-- Given `F : J ⥤ PresheafOfModules.{v} R`, this is the presheaf of modules obtained by
taking a colimit in the category of modules over `R.obj X` for all `X`. -/
@[simps]
noncomputable def colimitPresheafOfModules : PresheafOfModules R where
  obj X := colimit (F ⋙ evaluation R X)
  map {_ Y} f := colimMap (whiskerLeft F (restriction R f)) ≫
    (preservesColimitIso (ModuleCat.restrictScalars (R.map f).hom) (F ⋙ evaluation R Y)).inv
  map_id X := colimit.hom_ext (fun j => by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    rw [ι_colimMap_assoc, whiskerLeft_app, restriction_app]
    erw [ι_preservesColimitIso_inv (G := ModuleCat.restrictScalars (R.map (𝟙 X)).hom),
      ModuleCat.restrictScalarsId'App_inv_naturality]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map (CategoryTheory.Catego …
    -/
    rw [map_id]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.restrictScalarsId' (R.map …
    -/
    dsimp)
    /-
      🎉 no goals
    -/
  map_comp {X Y Z} f g := colimit.hom_ext (fun j => by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    rw [ι_colimMap_assoc, whiskerLeft_app, restriction_app, assoc, ι_colimMap_assoc]
    erw [ι_preservesColimitIso_inv (G := ModuleCat.restrictScalars (R.map (f ≫ g)).hom),
      ι_preservesColimitIso_inv_assoc (G := ModuleCat.restrictScalars (R.map f).hom)]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map (CategoryTheory.Catego …
    -/
    rw [← Functor.map_comp_assoc, ι_colimMap_assoc]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map (CategoryTheory.Catego …
    -/
    erw [ι_preservesColimitIso_inv (G := ModuleCat.restrictScalars (R.map g).hom)]
    rw [map_comp, ModuleCat.restrictScalarsComp'_inv_app, assoc, assoc,
      whiskerLeft_app, whiskerLeft_app, restriction_app, restriction_app]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) (CategoryTheory.Cat …
    -/
    simp only [Functor.map_comp, assoc]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
      inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) (CategoryTheory.Cat …
    -/
    rfl)
    /-
      🎉 no goals
    -/


/-- The (colimit) cocone for `F : J ⥤ PresheafOfModules.{v} R` that is constructed from
the colimit of `F ⋙ evaluation R X` for all `X`. -/
@[simps]
noncomputable def colimitCocone : Cocone F where
  pt := colimitPresheafOfModules F
  ι :=
    { app := fun j ↦
        { app := fun X ↦ colimit.ι (F ⋙ evaluation R X) j
          naturality := fun {X Y} f ↦ by
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              R : CategoryTheory.Functor (Opposite C) RingCat
              J : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} J
              F : CategoryTheory.Functor J (PresheafOfModules R)
              inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
              inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
              j : J
              X Y : Opposite C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((ModuleCat.restric …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              R : CategoryTheory.Functor (Opposite C) RingCat
              J : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} J
              F : CategoryTheory.Functor J (PresheafOfModules R)
              inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
              inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
              j : J
              X Y : Opposite C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((ModuleCat.restric …
            -/
            erw [colimit.ι_desc_assoc, assoc, ← ι_preservesColimitIso_inv]
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              R : CategoryTheory.Functor (Opposite C) RingCat
              J : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} J
              F : CategoryTheory.Functor J (PresheafOfModules R)
              inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
              inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
              j : J
              X Y : Opposite C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) (CategoryTheory.Cat …
            -/
            rfl }
            /-
              🎉 no goals
            -/
      naturality := fun {X Y} f ↦ by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
          inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { app := fun X = …
        -/
        ext1 X
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝¹ : ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), CategoryTheory.Limits.Pres …
          inst✝ : ∀ (X : Opposite C), CategoryTheory.Limits.HasColimit (F.comp (Presheaf …
          X✝ Y : J
          f : Quiver.Hom X✝ Y
          X : Opposite C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { app := fun X  …
        -/
        simpa using colimit.w (F ⋙ evaluation R X) f }
        /-
          🎉 no goals
        -/


/-- The cocone `colimitCocone F` is colimit for any `F : J ⥤ PresheafOfModules.{v} R`. -/
noncomputable def isColimitColimitCocone : IsColimit (colimitCocone F) :=
  evaluationJointlyReflectsColimits _ _ (fun _ => colimit.isColimit _)


instance hasColimit : HasColimit F := ⟨_, isColimitColimitCocone F⟩


instance evaluation_preservesColimit (X : Cᵒᵖ) :
    PreservesColimit F (evaluation R X) :=
  preservesColimit_of_preserves_colimit_cocone (isColimitColimitCocone F) (colimit.isColimit _)


instance toPresheaf_preservesColimit :
    PreservesColimit F (toPresheaf R) :=
  preservesColimit_of_preserves_colimit_cocone (isColimitColimitCocone F)
    (Limits.evaluationJointlyReflectsColimits _
      (fun X => isColimitOfPreserves (evaluation R X ⋙ forget₂ _ AddCommGrp)
        (isColimitColimitCocone F)))


instance hasColimitsOfShape : HasColimitsOfShape J (PresheafOfModules.{v} R) where


noncomputable instance evaluation_preservesColimitsOfShape (X : Cᵒᵖ) :
    PreservesColimitsOfShape J (evaluation R X : PresheafOfModules.{v} R ⥤ _) where


noncomputable instance toPresheaf_preservesColimitsOfShape :
    PreservesColimitsOfShape J (toPresheaf.{v} R) where


instance hasFiniteColimits : HasFiniteColimits (PresheafOfModules.{v} R) :=
  ⟨fun _ => inferInstance⟩


noncomputable instance evaluation_preservesFiniteColimits (X : Cᵒᵖ) :
    PreservesFiniteColimits (evaluation.{v} R X) where


noncomputable instance toPresheaf_preservesFiniteColimits :
    PreservesFiniteColimits (toPresheaf R) where


