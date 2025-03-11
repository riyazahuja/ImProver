/-- The category of linear orders. -/
def LinOrd :=
  Bundled LinearOrder


instance : BundledHom.ParentProjection @LinearOrder.toPartialOrder :=
  ⟨⟩


deriving instance LargeCategory for LinOrd

-- Porting note: Probably see https://github.com/leanprover-community/mathlib4/issues/5020

instance : ConcreteCategory LinOrd :=
  BundledHom.concreteCategory _


instance : CoeSort LinOrd Type* :=
  Bundled.coeSort


/-- Construct a bundled `LinOrd` from the underlying type and typeclass. -/
def of (α : Type*) [LinearOrder α] : LinOrd :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [LinearOrder α] : ↥(of α) = α :=
  rfl


instance : Inhabited LinOrd :=
  ⟨of PUnit⟩


instance (α : LinOrd) : LinearOrder α :=
  α.str


instance hasForgetToLat : HasForget₂ LinOrd Lat where
  forget₂ :=
    { obj := fun X => Lat.of X
      map := fun {X Y} (f : OrderHom _ _) => OrderHomClass.toLatticeHom X Y f }


/-- Constructs an equivalence between linear orders from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : LinOrd.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : OrderHom _ _)
  inv := (e.symm : OrderHom _ _)
  hom_inv_id := by
    /-
      α β : LinOrd
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : LinOrd
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget LinOrd).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) x) ((CategoryTheory.Cate …
    -/
    exact e.symm_apply_apply x
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : LinOrd
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : LinOrd
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget LinOrd).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) x) ((CategoryTheory.Cate …
    -/
    exact e.apply_symm_apply x
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : LinOrd ⥤ LinOrd where
  obj X := of Xᵒᵈ
  map := OrderHom.dual


/-- The equivalence between `LinOrd` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : LinOrd ≌ LinOrd where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : LinOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : LinOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem linOrd_dual_comp_forget_to_Lat :
    LinOrd.dual ⋙ forget₂ LinOrd Lat = forget₂ LinOrd Lat ⋙ Lat.dual :=
  rfl

