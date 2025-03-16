/-- The category of finite boolean algebras with bounded lattice morphisms. -/
structure FinBoolAlg where
  toBoolAlg : BoolAlg
  [isFintype : Fintype toBoolAlg]


instance : CoeSort FinBoolAlg Type* :=
  ⟨fun X => X.toBoolAlg⟩


instance (X : FinBoolAlg) : BooleanAlgebra X :=
  X.toBoolAlg.str


/-- Construct a bundled `FinBoolAlg` from `BooleanAlgebra` + `Fintype`. -/
def of (α : Type*) [BooleanAlgebra α] [Fintype α] : FinBoolAlg :=
  ⟨{α := α}⟩


@[simp]
theorem coe_of (α : Type*) [BooleanAlgebra α] [Fintype α] : ↥(of α) = α :=
  rfl


instance : Inhabited FinBoolAlg :=
  ⟨of PUnit⟩


instance largeCategory : LargeCategory FinBoolAlg :=
  InducedCategory.category FinBoolAlg.toBoolAlg


instance concreteCategory : ConcreteCategory FinBoolAlg :=
  InducedCategory.concreteCategory FinBoolAlg.toBoolAlg


instance instFunLike {X Y : FinBoolAlg} : FunLike (X ⟶ Y) X Y :=
  BoundedLatticeHom.instFunLike

-- Porting note: added
-- TODO: in all of the earlier bundled order categories,
-- we should be constructing instances analogous to this,
-- rather than directly coercions to functions.

instance instBoundedLatticeHomClass {X Y : FinBoolAlg} : BoundedLatticeHomClass (X ⟶ Y) X Y :=
  BoundedLatticeHom.instBoundedLatticeHomClass


instance hasForgetToBoolAlg : HasForget₂ FinBoolAlg BoolAlg :=
  InducedCategory.hasForget₂ FinBoolAlg.toBoolAlg


instance hasForgetToFinBddDistLat : HasForget₂ FinBoolAlg FinBddDistLat where
  forget₂.obj X := FinBddDistLat.of X
  forget₂.map f := f
  forget_comp := rfl


instance forgetToBoolAlg_full : (forget₂ FinBoolAlg BoolAlg).Full :=
  InducedCategory.full _


instance forgetToBoolAlgFaithful : (forget₂ FinBoolAlg BoolAlg).Faithful :=
  InducedCategory.faithful _


@[simps]
instance hasForgetToFinPartOrd : HasForget₂ FinBoolAlg FinPartOrd where
  forget₂.obj X := FinPartOrd.of X
  forget₂.map {X Y} f := show OrderHom X Y from ↑(show BoundedLatticeHom X Y from f)


instance forgetToFinPartOrdFaithful : (forget₂ FinBoolAlg FinPartOrd).Faithful :=
  -- Porting note: original code
  -- ⟨fun {X Y} f g h =>
  --   haveI := congr_arg (coeFn : _ → X → Y) h
  --   DFunLike.coe_injective this⟩
  -- Porting note: the coercions to functions for the various bundled order categories
  -- are quite inconsistent. We need to go back through and make all these files uniform.
  ⟨fun {X Y} f g h => by
    /-
      X Y : FinBoolAlg
      f g : Quiver.Hom X Y
      h : Eq ((CategoryTheory.forget₂ FinBoolAlg FinPartOrd).map f) ((CategoryTheory …
      ⊢ Eq f g
    -/
    dsimp at *
    /-
      X Y : FinBoolAlg
      f g : Quiver.Hom X Y
      h : Eq ↑f ↑g
      ⊢ Eq f g
    -/
    apply DFunLike.coe_injective
    /-
      case a
      X Y : FinBoolAlg
      f g : Quiver.Hom X Y
      h : Eq ↑f ↑g
      ⊢ Eq ((fun f => ⇑f) f) ((fun f => ⇑f) g)
    -/
    dsimp
    /-
      case a
      X Y : FinBoolAlg
      f g : Quiver.Hom X Y
      h : Eq ↑f ↑g
      ⊢ Eq ⇑f ⇑g
    -/
    ext x
    /-
      case a.h
      X Y : FinBoolAlg
      f g : Quiver.Hom X Y
      h : Eq ↑f ↑g
      x : ↑X.toBoolAlg
      ⊢ Eq (f x) (g x)
    -/
    apply_fun (fun f => f x) at h
    /-
      case a.h
      X Y : FinBoolAlg
      f g : Quiver.Hom X Y
      x : ↑X.toBoolAlg
      h : Eq (↑f x) (↑g x)
      ⊢ Eq (f x) (g x)
    -/
    exact h ⟩
    /-
      🎉 no goals
    -/


/-- Constructs an equivalence between finite Boolean algebras from an order isomorphism between
them. -/
@[simps]
def Iso.mk {α β : FinBoolAlg.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : BoundedLatticeHom α β)
  inv := (e.symm : BoundedLatticeHom β α)
                   /-
                     α β : FinBoolAlg
                     e : OrderIso ↑α.toBoolAlg ↑β.toBoolAlg
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
                     α β : FinBoolAlg
                     e : OrderIso ↑α.toBoolAlg ↑β.toBoolAlg
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
def dual : FinBoolAlg ⥤ FinBoolAlg where
  obj X := of Xᵒᵈ
  map {_ _} := BoundedLatticeHom.dual


/-- The equivalence between `FinBoolAlg` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : FinBoolAlg ≌ FinBoolAlg where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : FinBoolAlg} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : FinBoolAlg} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem finBoolAlg_dual_comp_forget_to_finBddDistLat :
    FinBoolAlg.dual ⋙ forget₂ FinBoolAlg FinBddDistLat =
      forget₂ FinBoolAlg FinBddDistLat ⋙ FinBddDistLat.dual :=
  rfl


/-- The powerset functor. `Set` as a functor. -/
@[simps]
def fintypeToFinBoolAlgOp : FintypeCat ⥤ FinBoolAlgᵒᵖ where
  obj X := op <| FinBoolAlg.of (Set X)
  map {X Y} f :=
    Quiver.Hom.op <| (CompleteLatticeHom.setPreimage f : BoundedLatticeHom (Set Y) (Set X))

