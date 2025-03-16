/-- A cone in the category `PresheafOfModules R` is limit if it is so after the application
of the functors `evaluation R X` for all `X`. -/
def evaluationJointlyReflectsLimits (c : Cone F)
    (hc : ∀ (X : Cᵒᵖ), IsLimit ((evaluation R X).mapCone c)) : IsLimit c where
  lift s :=
    { app := fun X => (hc X).lift ((evaluation R X).mapCone s)
      naturality := fun {X Y} f ↦ by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.map f) ((ModuleCat.restrictScal …
        -/
        apply (isLimitOfPreserves (ModuleCat.restrictScalars (R.map f).hom) (hc Y)).hom_ext
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
        -/
        intro j
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have h₁ := (c.π.app j).naturality f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Functor.const J …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have h₂ := (hc X).fac ((evaluation R X).mapCone s) j
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Functor.const J …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp ((hc X).lift ((PresheafOfModules.e …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Functor.mapCone_π_app, assoc, assoc, ← Functor.map_comp, IsLimit.fac]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Functor.const J …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp ((hc X).lift ((PresheafOfModules.e …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.map f) ((ModuleCat.restrictScal …
        -/
        dsimp at h₁ h₂ ⊢
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          c : CategoryTheory.Limits.Cone F
          hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
          s : CategoryTheory.Limits.Cone F
          X Y : Opposite C
          f : Quiver.Hom X Y
          j : J
          h₁ : Eq (CategoryTheory.CategoryStruct.comp (c.pt.map f) ((ModuleCat.restrictS …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp ((hc X).lift ((PresheafOfModules.e …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.map f) ((ModuleCat.restrictScal …
        -/
        rw [h₁, reassoc_of% h₂, Hom.naturality] }
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
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => { app := fun X => (hc X).l …
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
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      j : J
      X : Opposite C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun s => { app := fun X => (hc X). …
    -/
    exact (hc X).fac ((evaluation R X).mapCone s) j
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
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      ⊢ Eq m ((fun s => { app := fun X => (hc X).lift ((PresheafOfModules.evaluation …
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
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      X : Opposite C
      ⊢ Eq (m.app X) (((fun s => { app := fun X => (hc X).lift ((PresheafOfModules.e …
    -/
    apply (hc X).uniq ((evaluation R X).mapCone s)
    /-
      case h.x
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      X : Opposite C
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (m.app X) (((PresheafOfMod …
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
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (m.app X) (((PresheafOfModules.evalua …
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
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((PresheafOfModules.eval …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (m.app X) ((c.π.app j).app X)) ((s.π. …
    -/
    rw [← hm, comp_app]
    /-
      🎉 no goals
    -/


instance {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    HasLimit (F ⋙ evaluation R Y ⋙ ModuleCat.restrictScalars (R.map f).hom) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    J : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
    F : CategoryTheory.Functor J (PresheafOfModules R)
    inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
    X Y : Opposite C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Limits.HasLimit (F.comp ((PresheafOfModules.evaluation R Y).c …
  -/
  change HasLimit ((F ⋙ evaluation R Y) ⋙ ModuleCat.restrictScalars (R.map f).hom)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    J : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
    F : CategoryTheory.Functor J (PresheafOfModules R)
    inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
    X Y : Opposite C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Limits.HasLimit ((F.comp (PresheafOfModules.evaluation R Y)). …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Given `F : J ⥤ PresheafOfModules.{v} R`, this is the presheaf of modules obtained by
taking a limit in the category of modules over `R.obj X` for all `X`. -/
@[simps]
noncomputable def limitPresheafOfModules : PresheafOfModules R where
  obj X := limit (F ⋙ evaluation R X)
  map {_ Y} f := limMap (whiskerLeft F (restriction R f)) ≫
    (preservesLimitIso (ModuleCat.restrictScalars (R.map f).hom) (F ⋙ evaluation R Y)).inv
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      ⊢ Eq ((fun {x Y} f => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (Catego …
    -/
    rw [← cancel_mono (preservesLimitIso _ _).hom, assoc, Iso.inv_hom_id, comp_id]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      ⊢ Eq (CategoryTheory.Limits.limMap (CategoryTheory.whiskerLeft F (PresheafOfMo …
    -/
    apply limit.hom_ext
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.lim …
    -/
    intro j
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (Catego …
    -/
    dsimp
    simp only [limMap_π, Functor.comp_obj, evaluation_obj, whiskerLeft_app,
      restriction_app, assoc]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    erw [preservesLimitIso_hom_π]
    rw [← ModuleCat.restrictScalarsId'App_inv_naturality, map_id,
      ModuleCat.restrictScalarsId'_inv_app]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X : Opposite C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    dsimp
    /-
      🎉 no goals
    -/
  map_comp {X Y Z} f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ((fun {x Y} f => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
    -/
    dsimp
    rw [← cancel_mono (preservesLimitIso _ _).hom, assoc, assoc, assoc, assoc, Iso.inv_hom_id,
      comp_id]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.Limits.limMap (CategoryTheory.whiskerLeft F (PresheafOfMo …
    -/
    apply limit.hom_ext
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.lim …
    -/
    intro j
    simp only [Functor.comp_obj, evaluation_obj, limMap_π, whiskerLeft_app, restriction_app,
      map_comp, ModuleCat.restrictScalarsComp'_inv_app, Functor.map_comp, assoc]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    erw [preservesLimitIso_hom_π]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    rw [← ModuleCat.restrictScalarsComp'App_inv_naturality]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    dsimp
    rw [← Functor.map_comp_assoc, ← Functor.map_comp_assoc, assoc,
      preservesLimitIso_inv_π]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    erw [limMap_π]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    dsimp
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    simp only [Functor.map_comp, assoc, preservesLimitIso_inv_π_assoc]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    erw [limMap_π_assoc]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      R : CategoryTheory.Functor (Opposite C) RingCat
      J : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
      F : CategoryTheory.Functor J (PresheafOfModules R)
      inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
    -/
    dsimp
    /-
      🎉 no goals
    -/


/-- The (limit) cone for `F : J ⥤ PresheafOfModules.{v} R` that is constructed from the limit
of `F ⋙ evaluation R X` for all `X`. -/
@[simps]
noncomputable def limitCone : Cone F where
  pt := limitPresheafOfModules F
  π :=
    { app := fun j ↦
        { app := fun X ↦ limit.π (F ⋙ evaluation R X) j
          naturality := fun {X Y} f ↦ by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              R : CategoryTheory.Functor (Opposite C) RingCat
              J : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
              F : CategoryTheory.Functor J (PresheafOfModules R)
              inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
              j : J
              X Y : Opposite C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Functor.const J).o …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              R : CategoryTheory.Functor (Opposite C) RingCat
              J : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
              F : CategoryTheory.Functor J (PresheafOfModules R)
              inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
              j : J
              X Y : Opposite C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
            -/
            simp only [assoc, preservesLimitIso_inv_π]
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              R : CategoryTheory.Functor (Opposite C) RingCat
              J : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
              F : CategoryTheory.Functor J (PresheafOfModules R)
              inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
              j : J
              X Y : Opposite C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (Catego …
            -/
            apply limMap_π }
            /-
              🎉 no goals
            -/
      naturality := fun {j j'} f ↦ by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          R : CategoryTheory.Functor (Opposite C) RingCat
          J : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} J
          F : CategoryTheory.Functor J (PresheafOfModules R)
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
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
          inst✝ : ∀ (X : Opposite C), Small.{v, max u₂ v} ↑((F.comp (PresheafOfModules.e …
          j j' : J
          f : Quiver.Hom j j'
          X : Opposite C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).o …
        -/
        simpa using (limit.w (F ⋙ evaluation R X) f).symm }
        /-
          🎉 no goals
        -/


/-- The cone `limitCone F` is limit for any `F : J ⥤ PresheafOfModules.{v} R`. -/
noncomputable def isLimitLimitCone : IsLimit (limitCone F) :=
  evaluationJointlyReflectsLimits _ _ (fun _ => limit.isLimit _)


instance hasLimit : HasLimit F := ⟨_, isLimitLimitCone F⟩


noncomputable instance evaluation_preservesLimit (X : Cᵒᵖ) :
    PreservesLimit F (evaluation R X) :=
  preservesLimit_of_preserves_limit_cone (isLimitLimitCone F) (limit.isLimit _)


noncomputable instance toPresheaf_preservesLimit :
    PreservesLimit F (toPresheaf R) :=
  preservesLimit_of_preserves_limit_cone (isLimitLimitCone F)
    (Limits.evaluationJointlyReflectsLimits _
      (fun X => isLimitOfPreserves (evaluation R X ⋙ forget₂ _ AddCommGrp)
        (isLimitLimitCone F)))


instance hasLimitsOfShape : HasLimitsOfShape J (PresheafOfModules.{v} R) where


noncomputable instance evaluation_preservesLimitsOfShape (X : Cᵒᵖ) :
    PreservesLimitsOfShape J (evaluation R X : PresheafOfModules.{v} R ⥤ _) where


noncomputable instance toPresheaf_preservesLimitsOfShape :
    PreservesLimitsOfShape J (toPresheaf.{v} R) where


instance hasFiniteLimits : HasFiniteLimits (PresheafOfModules.{v} R) :=
  ⟨fun _ => inferInstance⟩


noncomputable instance evaluation_preservesFiniteLimits (X : Cᵒᵖ) :
    PreservesFiniteLimits (evaluation.{v} R X) where


noncomputable instance toPresheaf_preservesFiniteLimits :
    PreservesFiniteLimits (toPresheaf.{v} R) where


