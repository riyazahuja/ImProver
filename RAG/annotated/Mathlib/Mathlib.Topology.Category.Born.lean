/-- The category of bornologies. -/
def Born :=
  Bundled Bornology


instance : CoeSort Born Type* :=
  Bundled.coeSort


instance (X : Born) : Bornology X :=
  X.str


/-- Construct a bundled `Born` from a `Bornology`. -/
def of (α : Type*) [Bornology α] : Born :=
  Bundled.of α


instance : Inhabited Born :=
  ⟨of PUnit⟩


instance : BundledHom @LocallyBoundedMap where
  id := @LocallyBoundedMap.id
  comp := @LocallyBoundedMap.comp
  hom_ext _ _ := DFunLike.coe_injective


instance : LargeCategory.{u} Born :=
  BundledHom.category LocallyBoundedMap


instance : ConcreteCategory Born :=
  BundledHom.concreteCategory LocallyBoundedMap


