/-- The category of locales. -/
def Locale :=
  Frmᵒᵖ deriving LargeCategory


instance : CoeSort Locale Type* :=
  ⟨fun X => X.unop⟩


instance (X : Locale) : Frame X :=
  X.unop.str


/-- Construct a bundled `Locale` from a `Frame`. -/
def of (α : Type*) [Frame α] : Locale :=
  op <| Frm.of α


@[simp]
theorem coe_of (α : Type*) [Frame α] : ↥(of α) = α :=
  rfl


instance : Inhabited Locale :=
  ⟨of PUnit⟩


/-- The forgetful functor from `Top` to `Locale` which forgets that the space has "enough points".
-/
@[simps!]
def topToLocale : TopCat ⥤ Locale :=
  topCatOpToFrm.rightOp

-- Note, `CompHaus` is too strong. We only need `T0Space`.

instance CompHausToLocale.faithful : (compHausToTop ⋙ topToLocale.{u}).Faithful :=
  ⟨fun h => by
    /-
      X✝ Y✝ : CompHaus
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((compHausToTop.comp topToLocale).map a₁✝) ((compHausToTop.comp topToLo …
      ⊢ Eq a₁✝ a₂✝
    -/
    dsimp at h
    /-
      X✝ Y✝ : CompHaus
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq (Quiver.Hom.op (TopologicalSpace.Opens.comap a₁✝)) (Quiver.Hom.op (Topo …
      ⊢ Eq a₁✝ a₂✝
    -/
    exact Opens.comap_injective (Quiver.Hom.op_inj h)⟩
    /-
      🎉 no goals
    -/

