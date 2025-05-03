/-- The category of complete lattices. -/
def CompleteLat :=
  Bundled CompleteLattice


instance : CoeSort CompleteLat Type* :=
  Bundled.coeSort


instance (X : CompleteLat) : CompleteLattice X :=
  X.str


/-- Construct a bundled `CompleteLat` from a `CompleteLattice`. -/
def of (α : Type*) [CompleteLattice α] : CompleteLat :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [CompleteLattice α] : ↥(of α) = α :=
  rfl


instance : Inhabited CompleteLat :=
  ⟨of PUnit⟩


instance : BundledHom @CompleteLatticeHom where
  toFun _ _ f := f.toFun
  id := @CompleteLatticeHom.id
  comp := @CompleteLatticeHom.comp
  hom_ext _ _ _ _ h := DFunLike.coe_injective h


deriving instance LargeCategory for CompleteLat


instance : ConcreteCategory CompleteLat := by
  /-
    ⊢ CategoryTheory.ConcreteCategory CompleteLat
  -/
  dsimp [CompleteLat]; infer_instance
                       /-
                         🎉 no goals
                       -/


instance hasForgetToBddLat : HasForget₂ CompleteLat BddLat where
  forget₂ :=
    { obj := fun X => BddLat.of X
      map := fun {_ _} => CompleteLatticeHom.toBoundedLatticeHom }
  forget_comp := rfl


/-- Constructs an isomorphism of complete lattices from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : CompleteLat.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : CompleteLatticeHom _ _) -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO, wrong?
  inv := (e.symm : CompleteLatticeHom _ _)
                   /-
                     α β : CompleteLat
                     e : OrderIso ↑α ↑β
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_sInf' := ⋯, map_sS …
                   -/
  hom_inv_id := by ext; exact e.symm_apply_apply _
                        /-
                          🎉 no goals
                        -/
                   /-
                     α β : CompleteLat
                     e : OrderIso ↑α ↑β
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_sInf' := ⋯, m …
                   -/
  inv_hom_id := by ext; exact e.apply_symm_apply _
                        /-
                          🎉 no goals
                        -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : CompleteLat ⥤ CompleteLat where
  obj X := of Xᵒᵈ
  map {_ _} := CompleteLatticeHom.dual


/-- The equivalence between `CompleteLat` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : CompleteLat ≌ CompleteLat where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : CompleteLat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruc …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : CompleteLat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruc …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


theorem completeLat_dual_comp_forget_to_bddLat :
    CompleteLat.dual ⋙ forget₂ CompleteLat BddLat =
    forget₂ CompleteLat BddLat ⋙ BddLat.dual :=
  rfl

