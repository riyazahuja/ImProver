/-- The category of bounded distributive lattices with bounded lattice morphisms. -/
structure BddDistLat where
  /-- The underlying distrib lattice of a bounded distributive lattice. -/
  toDistLat : DistLat
  [isBoundedOrder : BoundedOrder toDistLat]


instance : CoeSort BddDistLat Type* :=
  ⟨fun X => X.toDistLat⟩


instance (X : BddDistLat) : DistribLattice X :=
  X.toDistLat.str


/-- Construct a bundled `BddDistLat` from a `BoundedOrder` `DistribLattice`. -/
def of (α : Type*) [DistribLattice α] [BoundedOrder α] : BddDistLat :=
  -- Porting note: was `⟨⟨α⟩⟩`
  -- see https://github.com/leanprover-community/mathlib4/issues/4998
  ⟨{α := α}⟩


@[simp]
theorem coe_of (α : Type*) [DistribLattice α] [BoundedOrder α] : ↥(of α) = α :=
  rfl


instance : Inhabited BddDistLat :=
  ⟨of PUnit⟩


/-- Turn a `BddDistLat` into a `BddLat` by forgetting it is distributive. -/
def toBddLat (X : BddDistLat) : BddLat :=
  BddLat.of X


@[simp]
theorem coe_toBddLat (X : BddDistLat) : ↥X.toBddLat = ↥X :=
  rfl


instance : LargeCategory.{u} BddDistLat :=
  InducedCategory.category toBddLat


instance : ConcreteCategory BddDistLat :=
  InducedCategory.concreteCategory toBddLat


instance hasForgetToDistLat : HasForget₂ BddDistLat DistLat where
  forget₂ :=
    -- Porting note: was `⟨X⟩`
    -- see https://github.com/leanprover-community/mathlib4/issues/4998
    { obj := fun X => { α := X }
      map := fun {_ _} => BoundedLatticeHom.toLatticeHom }


instance hasForgetToBddLat : HasForget₂ BddDistLat BddLat :=
  InducedCategory.hasForget₂ toBddLat


theorem forget_bddLat_lat_eq_forget_distLat_lat :
    forget₂ BddDistLat BddLat ⋙ forget₂ BddLat Lat =
      forget₂ BddDistLat DistLat ⋙ forget₂ DistLat Lat :=
  rfl


/-- Constructs an equivalence between bounded distributive lattices from an order isomorphism
between them. -/
@[simps]
def Iso.mk {α β : BddDistLat.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : BoundedLatticeHom α β)
  inv := (e.symm : BoundedLatticeHom β α)
                   /-
                     α β : BddDistLat
                     e : OrderIso ↑α.toDistLat ↑β.toDistLat
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
                     α β : BddDistLat
                     e : OrderIso ↑α.toDistLat ↑β.toDistLat
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
def dual : BddDistLat ⥤ BddDistLat where
  obj X := of Xᵒᵈ
  map {_ _} := BoundedLatticeHom.dual


/-- The equivalence between `BddDistLat` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : BddDistLat ≌ BddDistLat where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : BddDistLat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : BddDistLat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem bddDistLat_dual_comp_forget_to_distLat :
    BddDistLat.dual ⋙ forget₂ BddDistLat DistLat =
      forget₂ BddDistLat DistLat ⋙ DistLat.dual :=
  rfl

