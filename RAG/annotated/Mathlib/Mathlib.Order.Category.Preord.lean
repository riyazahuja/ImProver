/-- The category of preorders. -/
def Preord :=
  Bundled Preorder


instance : BundledHom @OrderHom where
  toFun := @OrderHom.toFun
  id := @OrderHom.id
  comp := @OrderHom.comp
  hom_ext := @OrderHom.ext


deriving instance LargeCategory for Preord

-- Porting note: probably see https://github.com/leanprover-community/mathlib4/issues/5020

instance : ConcreteCategory Preord :=
  BundledHom.concreteCategory _


instance : CoeSort Preord Type* :=
  Bundled.coeSort


/-- Construct a bundled Preord from the underlying type and typeclass. -/
def of (α : Type*) [Preorder α] : Preord :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [Preorder α] : ↥(of α) = α :=
  rfl


instance : Inhabited Preord :=
  ⟨of PUnit⟩


instance (α : Preord) : Preorder α :=
  α.str


/-- Constructs an equivalence between preorders from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : Preord.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : OrderHom α β)
  inv := (e.symm : OrderHom β α)
  hom_inv_id := by
    /-
      α β : Preord
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : Preord
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget Preord).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) x) ((CategoryTheory.Cate …
    -/
    exact e.symm_apply_apply x
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : Preord
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : Preord
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget Preord).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) x) ((CategoryTheory.Cate …
    -/
    exact e.apply_symm_apply x
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : Preord ⥤ Preord where
  obj X := of Xᵒᵈ
  map := OrderHom.dual


/-- The equivalence between `Preord` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : Preord ≌ Preord where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : Preord} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : Preord} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


/-- The embedding of `Preord` into `Cat`.
-/
@[simps]
def preordToCat : Preord.{u} ⥤ Cat where
  obj X := Cat.of X.1
  map f := f.monotone.functor


instance : preordToCat.{u}.Faithful where
                        /-
                          X✝ Y✝ : Preord
                          a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                          h : Eq (preordToCat.map a₁✝) (preordToCat.map a₂✝)
                          ⊢ Eq a₁✝ a₂✝
                        -/
  map_injective h := by ext x; exact Functor.congr_obj h x
                               /-
                                 🎉 no goals
                               -/


instance : preordToCat.{u}.Full where
  map_surjective {X Y} f := ⟨⟨f.obj, @CategoryTheory.Functor.monotone X Y _ _ f⟩, rfl⟩

