/-- The category of finite partial orders with monotone functions. -/
structure FinPartOrd where
  toPartOrd : PartOrd
  [isFintype : Fintype toPartOrd]


instance : CoeSort FinPartOrd Type* :=
  ⟨fun X => X.toPartOrd⟩


instance (X : FinPartOrd) : PartialOrder X :=
  X.toPartOrd.str


/-- Construct a bundled `FinPartOrd` from `PartialOrder` + `Fintype`. -/
def of (α : Type*) [PartialOrder α] [Fintype α] : FinPartOrd :=
  ⟨⟨α, inferInstance⟩⟩


@[simp]
theorem coe_of (α : Type*) [PartialOrder α] [Fintype α] : ↥(of α) = α := rfl


instance : Inhabited FinPartOrd :=
  ⟨of PUnit⟩


instance largeCategory : LargeCategory FinPartOrd :=
  InducedCategory.category FinPartOrd.toPartOrd


instance concreteCategory : ConcreteCategory FinPartOrd :=
  InducedCategory.concreteCategory FinPartOrd.toPartOrd


instance hasForgetToPartOrd : HasForget₂ FinPartOrd PartOrd :=
  InducedCategory.hasForget₂ FinPartOrd.toPartOrd


instance hasForgetToFintype : HasForget₂ FinPartOrd FintypeCat where
  forget₂ :=
    { obj := fun X => ⟨X, inferInstance⟩
      -- Porting note: Originally `map := fun X Y => coeFn`
      map := fun {X Y} (f : OrderHom X Y) => ⇑f }


/-- Constructs an isomorphism of finite partial orders from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : FinPartOrd.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : OrderHom _ _)
  inv := (e.symm : OrderHom _ _)
  hom_inv_id := by
    /-
      α β : FinPartOrd
      e : OrderIso ↑α.toPartOrd ↑β.toPartOrd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) (CategoryTheory.CategoryS …
    -/
    ext
    /-
      case w
      α β : FinPartOrd
      e : OrderIso ↑α.toPartOrd ↑β.toPartOrd
      x✝ : (CategoryTheory.forget FinPartOrd).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) x✝) ((CategoryTheory.Cat …
    -/
    exact e.symm_apply_apply _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : FinPartOrd
      e : OrderIso ↑α.toPartOrd ↑β.toPartOrd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) (CategoryTheory.CategoryS …
    -/
    ext
    /-
      case w
      α β : FinPartOrd
      e : OrderIso ↑α.toPartOrd ↑β.toPartOrd
      x✝ : (CategoryTheory.forget FinPartOrd).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) x✝) ((CategoryTheory.Cat …
    -/
    exact e.apply_symm_apply _
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : FinPartOrd ⥤ FinPartOrd where
  obj X := of Xᵒᵈ
  map {_ _} := OrderHom.dual


/-- The equivalence between `FinPartOrd` and itself induced by `OrderDual` both ways. -/
@[simps]
def dualEquiv : FinPartOrd ≌ FinPartOrd where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : FinPartOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : FinPartOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem FinPartOrd_dual_comp_forget_to_partOrd :
    FinPartOrd.dual ⋙ forget₂ FinPartOrd PartOrd =
      forget₂ FinPartOrd PartOrd ⋙ PartOrd.dual := rfl

