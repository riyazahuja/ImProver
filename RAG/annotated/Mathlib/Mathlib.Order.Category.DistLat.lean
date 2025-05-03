/-- The category of distributive lattices. -/
def DistLat :=
  Bundled DistribLattice


instance : CoeSort DistLat Type* :=
  Bundled.coeSort


instance (X : DistLat) : DistribLattice X :=
  X.str


/-- Construct a bundled `DistLat` from a `DistribLattice` underlying type and typeclass. -/
def of (α : Type*) [DistribLattice α] : DistLat :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [DistribLattice α] : ↥(of α) = α :=
  rfl


instance : Inhabited DistLat :=
  ⟨of PUnit⟩


instance : BundledHom.ParentProjection @DistribLattice.toLattice :=
  ⟨⟩


deriving instance LargeCategory for DistLat


instance : ConcreteCategory DistLat :=
  BundledHom.concreteCategory _


instance hasForgetToLat : HasForget₂ DistLat Lat :=
  BundledHom.forget₂ _ _


/-- Constructs an equivalence between distributive lattices from an order isomorphism between them.
-/
@[simps]
def Iso.mk {α β : DistLat.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : LatticeHom α β)
  inv := (e.symm : LatticeHom β α)
  hom_inv_id := by
    /-
      α β : DistLat
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_sup' := ⋯, map_inf …
    -/
    ext
    /-
      case w
      α β : DistLat
      e : OrderIso ↑α ↑β
      x✝ : (CategoryTheory.forget DistLat).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_sup' := ⋯, map_in …
    -/
    exact e.symm_apply_apply _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : DistLat
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_sup' := ⋯, ma …
    -/
    ext
    /-
      case w
      α β : DistLat
      e : OrderIso ↑α ↑β
      x✝ : (CategoryTheory.forget DistLat).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_sup' := ⋯, m …
    -/
    exact e.apply_symm_apply _
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : DistLat ⥤ DistLat where
  obj X := of Xᵒᵈ
  map := LatticeHom.dual


/-- The equivalence between `DistLat` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : DistLat ≌ DistLat where
  functor := dual
  inverse := dual
  unitIso := NatIso.ofComponents (fun X => Iso.mk <| OrderIso.dualDual X) fun _ => rfl
  counitIso := NatIso.ofComponents (fun X => Iso.mk <| OrderIso.dualDual X) fun _ => rfl


theorem distLat_dual_comp_forget_to_Lat :
    DistLat.dual ⋙ forget₂ DistLat Lat = forget₂ DistLat Lat ⋙ Lat.dual :=
  rfl

