/-- The category of lattices. -/
def Lat :=
  Bundled Lattice


instance : CoeSort Lat Type* :=
  Bundled.coeSort


instance (X : Lat) : Lattice X :=
  X.str


/-- Construct a bundled `Lat` from a `Lattice`. -/
def of (α : Type*) [Lattice α] : Lat :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [Lattice α] : ↥(of α) = α :=
  rfl


instance : Inhabited Lat :=
  ⟨of Bool⟩


instance : BundledHom @LatticeHom where
  toFun _ _ f := f.toFun
  id := @LatticeHom.id
  comp := @LatticeHom.comp
  hom_ext _ _ _ _ h := DFunLike.coe_injective h


instance : LargeCategory.{u} Lat :=
  BundledHom.category LatticeHom


instance : ConcreteCategory Lat :=
  BundledHom.concreteCategory LatticeHom


instance hasForgetToPartOrd : HasForget₂ Lat PartOrd where
  forget₂ :=
    { obj := fun X => Bundled.mk X inferInstance
      map := fun {X Y} (f : LatticeHom X Y) => (f : OrderHom X Y) }


/-- Constructs an isomorphism of lattices from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : Lat.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : LatticeHom _ _)
  inv := (e.symm : LatticeHom _ _)
  hom_inv_id := by
    /-
      α β : Lat
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_sup' := ⋯, map_inf …
    -/
    ext
    /-
      case w
      α β : Lat
      e : OrderIso ↑α ↑β
      x✝ : (CategoryTheory.forget Lat).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_sup' := ⋯, map_in …
    -/
    exact e.symm_apply_apply _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : Lat
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_sup' := ⋯, ma …
    -/
    ext
    /-
      case w
      α β : Lat
      e : OrderIso ↑α ↑β
      x✝ : (CategoryTheory.forget Lat).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_sup' := ⋯, m …
    -/
    exact e.apply_symm_apply _
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : Lat ⥤ Lat where
  obj X := of Xᵒᵈ
  map := LatticeHom.dual


/-- The equivalence between `Lat` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : Lat ≌ Lat where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : Lat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : Lat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem Lat_dual_comp_forget_to_partOrd :
    Lat.dual ⋙ forget₂ Lat PartOrd = forget₂ Lat PartOrd ⋙ PartOrd.dual :=
  rfl

