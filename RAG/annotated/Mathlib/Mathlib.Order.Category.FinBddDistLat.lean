/-- The category of finite distributive lattices with bounded lattice morphisms. -/
structure FinBddDistLat where
  toBddDistLat : BddDistLat
  [isFintype : Fintype toBddDistLat]


instance : CoeSort FinBddDistLat Type* :=
  ⟨fun X => X.toBddDistLat⟩


instance (X : FinBddDistLat) : DistribLattice X :=
  X.toBddDistLat.toDistLat.str


instance (X : FinBddDistLat) : BoundedOrder X :=
  X.toBddDistLat.isBoundedOrder


/-- Construct a bundled `FinBddDistLat` from a `Nonempty` `BoundedOrder` `DistribLattice`. -/
def of (α : Type*) [DistribLattice α] [BoundedOrder α] [Fintype α] : FinBddDistLat :=
  -- Porting note: was `⟨⟨⟨α⟩⟩⟩`
  -- see https://github.com/leanprover-community/mathlib4/issues/4998
  ⟨⟨{α := α}⟩⟩


/-- Construct a bundled `FinBddDistLat` from a `Nonempty` `BoundedOrder` `DistribLattice`. -/
def of' (α : Type*) [DistribLattice α] [Fintype α] [Nonempty α] : FinBddDistLat :=
  haveI := Fintype.toBoundedOrder α
  -- Porting note: was `⟨⟨⟨α⟩⟩⟩`
  -- see https://github.com/leanprover-community/mathlib4/issues/4998
  ⟨⟨{α := α}⟩⟩


instance : Inhabited FinBddDistLat :=
  ⟨of PUnit⟩


instance largeCategory : LargeCategory FinBddDistLat :=
  InducedCategory.category toBddDistLat


instance concreteCategory : ConcreteCategory FinBddDistLat :=
  InducedCategory.concreteCategory toBddDistLat


instance hasForgetToBddDistLat : HasForget₂ FinBddDistLat BddDistLat :=
  InducedCategory.hasForget₂ FinBddDistLat.toBddDistLat


instance hasForgetToFinPartOrd : HasForget₂ FinBddDistLat FinPartOrd where
  forget₂.obj X := FinPartOrd.of X
  forget₂.map {X Y} f := (show BoundedLatticeHom X Y from f : X →o Y)


/-- Constructs an equivalence between finite distributive lattices from an order isomorphism
between them. -/
@[simps]
def Iso.mk {α β : FinBddDistLat.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : BoundedLatticeHom α β)
  inv := (e.symm : BoundedLatticeHom β α)
                   /-
                     α β : FinBddDistLat
                     e : OrderIso ↑α.toBddDistLat.toDistLat ↑β.toBddDistLat.toDistLat
                     ⊢ Eq
                         (CategoryTheory.CategoryStruct.comp
                           (let __src := { toFun := ⇑e, map_sup' := ⋯, map_inf' := ⋯ };
                           { toFun := ⇑e, map_sup' := ⋯, map_inf' := ⋯, map_top' := ⋯, map_bot' :=  …
                           (let __src := { toFun := ⇑e.symm, map_sup' := ⋯, map_inf' := ⋯ };
                           { toFun := ⇑e.symm, map_sup' := ⋯, map_inf' := ⋯, map_top' := ⋯, map_bot …
                         (CategoryTheory.CategoryStruct.id α)
                   -/
  hom_inv_id := by ext; exact e.symm_apply_apply _
                        /-
                          🎉 no goals
                        -/
                   /-
                     α β : FinBddDistLat
                     e : OrderIso ↑α.toBddDistLat.toDistLat ↑β.toBddDistLat.toDistLat
                     ⊢ Eq
                         (CategoryTheory.CategoryStruct.comp
                           (let __src := { toFun := ⇑e.symm, map_sup' := ⋯, map_inf' := ⋯ };
                           { toFun := ⇑e.symm, map_sup' := ⋯, map_inf' := ⋯, map_top' := ⋯, map_bot …
                           (let __src := { toFun := ⇑e, map_sup' := ⋯, map_inf' := ⋯ };
                           { toFun := ⇑e, map_sup' := ⋯, map_inf' := ⋯, map_top' := ⋯, map_bot' :=  …
                         (CategoryTheory.CategoryStruct.id β)
                   -/
  inv_hom_id := by ext; exact e.apply_symm_apply _
                        /-
                          🎉 no goals
                        -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : FinBddDistLat ⥤ FinBddDistLat where
  obj X := of Xᵒᵈ
  map {_ _} := BoundedLatticeHom.dual


/-- The equivalence between `FinBddDistLat` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : FinBddDistLat ≌ FinBddDistLat where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : FinBddDistLat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStr …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : FinBddDistLat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStr …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem finBddDistLat_dual_comp_forget_to_bddDistLat :
    FinBddDistLat.dual ⋙ forget₂ FinBddDistLat BddDistLat =
      forget₂ FinBddDistLat BddDistLat ⋙ BddDistLat.dual :=
  rfl

