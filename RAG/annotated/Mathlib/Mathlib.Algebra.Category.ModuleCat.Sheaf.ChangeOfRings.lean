/-- The restriction of scalars functor `SheafOfModules R' ⥤ SheafOfModules R`
induced by a morphism of sheaves of rings `R ⟶ R'`. -/
@[simps]
noncomputable def restrictScalars :
    SheafOfModules.{v} R' ⥤ SheafOfModules.{v} R where
  obj M' :=
    { val := (PresheafOfModules.restrictScalars α.val).obj M'.val
      isSheaf := M'.isSheaf }
  map φ := { val := (PresheafOfModules.restrictScalars α.val).map φ.val }


instance : (restrictScalars.{v} α).Additive where


/-- The functor `PresheafOfModules.restrictScalars α` induces bijections on
morphisms if `α` is locally surjective and the target presheaf is a sheaf. -/
noncomputable def restrictHomEquivOfIsLocallySurjective
    (hM₂ : Presheaf.IsSheaf J M₂.presheaf) [Presheaf.IsLocallySurjective J α] :
    (M₁ ⟶ M₂) ≃ ((restrictScalars α).obj M₁ ⟶ (restrictScalars α).obj M₂) where
  toFun f := (restrictScalars α).map f
  invFun g := homMk ((toPresheaf R).map g) (fun X r' m ↦ by
    /-
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      ⊢ Eq ((((PresheafOfModules.toPresheaf R).map g).app X) (HSMul.hSMul r' m)) (HS …
    -/
    apply hM₂.isSeparated _ _ (Presheaf.imageSieve_mem J α r')
    -- Type-ascript `hr` so it uses `RingCat.Hom.hom` instead of `ConcreteCategory.instFunLike`
    /-
      case a
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      ⊢ ∀ (Y : C) (f : Quiver.Hom Y (Opposite.unop X)), (CategoryTheory.Presheaf.ima …
    -/
    rintro Y p ⟨r : R.obj _, (hr : α.app (Opposite.op Y) r = R'.map p.op r')⟩
    have hg : ∀ (z : M₁.obj X), g.app _ (M₁.map p.op z) = M₂.map p.op (g.app X z) :=
      fun z ↦ congr_fun ((forget _).congr_map (g.naturality p.op)) z
    /-
      case a.intro
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      Y : C
      p : Quiver.Hom Y (Opposite.unop X)
      r : ↑(R.obj { unop := Y })
      hr : Eq ((α.app { unop := Y }).hom r) ((R'.map p.op).hom r')
      hg : ∀ (z : ↑(M₁.obj X)), Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom z)) …
      ⊢ Eq ((M₂.presheaf.map p.op) ((((PresheafOfModules.toPresheaf R).map g).app X) …
    -/
    change M₂.map p.op (g.app X (r' • m)) = M₂.map p.op (r' • show M₂.obj X from g.app X m)
    /-
      case a.intro
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      Y : C
      p : Quiver.Hom Y (Opposite.unop X)
      r : ↑(R.obj { unop := Y })
      hr : Eq ((α.app { unop := Y }).hom r) ((R'.map p.op).hom r')
      hg : ∀ (z : ↑(M₁.obj X)), Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom z)) …
      ⊢ Eq ((M₂.map p.op).hom ((g.app X).hom (HSMul.hSMul r' m))) ((M₂.map p.op).hom …
    -/
    dsimp at hg ⊢
    /-
      case a.intro
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      Y : C
      p : Quiver.Hom Y (Opposite.unop X)
      r : ↑(R.obj { unop := Y })
      hr : Eq ((α.app { unop := Y }).hom r) ((R'.map p.op).hom r')
      hg : ∀ (z : ↑(M₁.obj X)), Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom z)) …
      ⊢ Eq ((M₂.map p.op).hom ((g.app X).hom (HSMul.hSMul r' m))) ((M₂.map p.op).hom …
    -/
    rw [← hg, M₂.map_smul, ← hg, ← hr]
    /-
      case a.intro
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      Y : C
      p : Quiver.Hom Y (Opposite.unop X)
      r : ↑(R.obj { unop := Y })
      hr : Eq ((α.app { unop := Y }).hom r) ((R'.map p.op).hom r')
      hg : ∀ (z : ↑(M₁.obj X)), Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom z)) …
      ⊢ Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom (HSMul.hSMul r' m))) (HSMul …
    -/
    erw [← (g.app _).hom.map_smul]
    /-
      case a.intro
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      Y : C
      p : Quiver.Hom Y (Opposite.unop X)
      r : ↑(R.obj { unop := Y })
      hr : Eq ((α.app { unop := Y }).hom r) ((R'.map p.op).hom r')
      hg : ∀ (z : ↑(M₁.obj X)), Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom z)) …
      ⊢ Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom (HSMul.hSMul r' m))) ((g.ap …
    -/
    rw [M₁.map_smul, ← hr]
    /-
      case a.intro
      C : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R R' : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R R'
      M₁ M₂ : PresheafOfModules R'
      hM₂ : CategoryTheory.Presheaf.IsSheaf J M₂.presheaf
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J α
      g : Quiver.Hom ((PresheafOfModules.restrictScalars α).obj M₁) ((PresheafOfModu …
      X : Opposite C
      r' : ↑(R'.obj X)
      m : ↑(M₁.obj X)
      Y : C
      p : Quiver.Hom Y (Opposite.unop X)
      r : ↑(R.obj { unop := Y })
      hr : Eq ((α.app { unop := Y }).hom r) ((R'.map p.op).hom r')
      hg : ∀ (z : ↑(M₁.obj X)), Eq ((g.app { unop := Y }).hom ((M₁.map p.op).hom z)) …
      ⊢ Eq ((g.app { unop := Y }).hom (HSMul.hSMul ((α.app { unop := Y }).hom r) ((M …
    -/
    rfl)
    /-
      🎉 no goals
    -/
  left_inv _ := rfl
  right_inv _ := rfl


