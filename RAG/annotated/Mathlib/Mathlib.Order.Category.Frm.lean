/-- The category of frames. -/
def Frm :=
  Bundled Frame


instance : CoeSort Frm Type* :=
  Bundled.coeSort


instance (X : Frm) : Frame X :=
  X.str


/-- Construct a bundled `Frm` from a `Frame`. -/
def of (α : Type*) [Frame α] : Frm :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [Frame α] : ↥(of α) = α := rfl


instance : Inhabited Frm :=
  ⟨of PUnit⟩


/-- An abbreviation of `FrameHom` that assumes `Frame` instead of the weaker `CompleteLattice`.
Necessary for the category theory machinery. -/
abbrev Hom (α β : Type*) [Frame α] [Frame β] : Type _ :=
  FrameHom α β


instance bundledHom : BundledHom Hom where
  toFun {α β} _ _ := ((↑) : FrameHom α β → α → β)
  id {α} _ := FrameHom.id α
  comp _ _ _ := FrameHom.comp
  hom_ext _ _ := DFunLike.coe_injective

-- Porting note: Originally `deriving instance LargeCategory, ConcreteCategory for Frm`
-- see https://github.com/leanprover-community/mathlib4/issues/5020

deriving instance LargeCategory, Category for Frm


instance : ConcreteCategory Frm := by
  /-
    ⊢ CategoryTheory.ConcreteCategory Frm
  -/
  unfold Frm
  /-
    ⊢ CategoryTheory.ConcreteCategory (CategoryTheory.Bundled Order.Frame)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance hasForgetToLat : HasForget₂ Frm Lat where
  forget₂ :=
    { obj := fun X => ⟨X, _⟩
      map := fun {_ _} => FrameHom.toLatticeHom }


/-- Constructs an isomorphism of frames from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : Frm.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : FrameHom _ _)
  inv := (e.symm : FrameHom _ _)
  hom_inv_id := by
    /-
      α β : Frm
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_inf' := ⋯, map_top …
    -/
    ext
    /-
      case w
      α β : Frm
      e : OrderIso ↑α ↑β
      x✝ : (CategoryTheory.forget Frm).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_inf' := ⋯, map_to …
    -/
    exact e.symm_apply_apply _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : Frm
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_inf' := ⋯, ma …
    -/
    ext
    /-
      case w
      α β : Frm
      e : OrderIso ↑α ↑β
      x✝ : (CategoryTheory.forget Frm).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_inf' := ⋯, m …
    -/
    exact e.apply_symm_apply _
    /-
      🎉 no goals
    -/


/-- The forgetful functor from `TopCatᵒᵖ` to `Frm`. -/
@[simps]
def topCatOpToFrm : TopCatᵒᵖ ⥤ Frm where
  obj X := Frm.of (Opens (unop X : TopCat))
  map f := Opens.comap <| Quiver.Hom.unop f
  map_id _ := Opens.comap_id

-- Note, `CompHaus` is too strong. We only need `T0Space`.

instance CompHausOpToFrame.faithful : (compHausToTop.op ⋙ topCatOpToFrm.{u}).Faithful :=
  ⟨fun h => Quiver.Hom.unop_inj <| Opens.comap_injective h⟩

