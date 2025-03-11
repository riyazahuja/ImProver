/-- The category of Heyting algebras. -/
def HeytAlg :=
  Bundled HeytingAlgebra


instance : CoeSort HeytAlg Type* :=
  Bundled.coeSort


instance (X : HeytAlg) : HeytingAlgebra X :=
  X.str


/-- Construct a bundled `HeytAlg` from a `HeytingAlgebra`. -/
def of (α : Type*) [HeytingAlgebra α] : HeytAlg :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [HeytingAlgebra α] : ↥(of α) = α :=
  rfl


instance : Inhabited HeytAlg :=
  ⟨of PUnit⟩


instance bundledHom : BundledHom HeytingHom where
  toFun α β [HeytingAlgebra α] [HeytingAlgebra β] := (DFunLike.coe : HeytingHom α β → α → β)
  id := @HeytingHom.id
  comp := @HeytingHom.comp
  hom_ext α β [HeytingAlgebra α] [HeytingAlgebra β] := DFunLike.coe_injective


deriving instance LargeCategory for HeytAlg

-- Porting note: deriving failed.
-- see https://github.com/leanprover-community/mathlib4/issues/5020

instance : ConcreteCategory HeytAlg := by
  /-
    ⊢ CategoryTheory.ConcreteCategory HeytAlg
  -/
  dsimp [HeytAlg]
  /-
    ⊢ CategoryTheory.ConcreteCategory (CategoryTheory.Bundled HeytingAlgebra)
  -/
  infer_instance
  /-
    🎉 no goals
  -/

-- Porting note: No idea why it does not find this instance...

instance {X Y : HeytAlg.{u}} : FunLike (X ⟶ Y) ↑X ↑Y :=
  HeytingHom.instFunLike

-- Porting note: No idea why it does not find this instance...

instance {X Y : HeytAlg.{u}} : HeytingHomClass (X ⟶ Y) ↑X ↑Y :=
  HeytingHom.instHeytingHomClass


@[simps]
instance hasForgetToLat : HasForget₂ HeytAlg BddDistLat where
  forget₂ :=
    { obj := fun X => BddDistLat.of X
      map := fun {X Y} f => (f : BoundedLatticeHom X Y) }


/-- Constructs an isomorphism of Heyting algebras from an order isomorphism between them. -/
@[simps]
def Iso.mk {α β : HeytAlg.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : HeytingHom _ _)
  inv := (e.symm : HeytingHom _ _)
                   /-
                     α β : HeytAlg
                     e : OrderIso ↑α ↑β
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e, map_sup' := ⋯, map_inf …
                   -/
  hom_inv_id := by ext; exact e.symm_apply_apply _
                        /-
                          🎉 no goals
                        -/
                   /-
                     α β : HeytAlg
                     e : OrderIso ↑α ↑β
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑e.symm, map_sup' := ⋯, ma …
                   -/
  inv_hom_id := by ext; exact e.apply_symm_apply _
                        /-
                          🎉 no goals
                        -/


