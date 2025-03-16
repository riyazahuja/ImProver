/-- The group of ring automorphisms. -/
abbrev RingAut (R : Type*) [Mul R] [Add R] :=
  RingEquiv R R


/-- The group operation on automorphisms of a ring is defined by
`fun g h => RingEquiv.trans h g`.
This means that multiplication agrees with composition, `(g*h)(x) = g (h x)`. -/
instance : Group (RingAut R) where
  mul g h := RingEquiv.trans h g
  one := RingEquiv.refl R
  inv := RingEquiv.symm
  mul_assoc _ _ _ := rfl
  one_mul _ := rfl
  mul_one _ := rfl
  inv_mul_cancel := RingEquiv.self_trans_symm


instance : Inhabited (RingAut R) :=
  ⟨1⟩


/-- Monoid homomorphism from ring automorphisms to additive automorphisms. -/
def toAddAut : RingAut R →* AddAut R where
  toFun := RingEquiv.toAddEquiv
  map_one' := rfl
  map_mul' _ _ := rfl


/-- Monoid homomorphism from ring automorphisms to multiplicative automorphisms. -/
def toMulAut : RingAut R →* MulAut R where
  toFun := RingEquiv.toMulEquiv
  map_one' := rfl
  map_mul' _ _ := rfl


/-- Monoid homomorphism from ring automorphisms to permutations. -/
def toPerm : RingAut R →* Equiv.Perm R where
  toFun := RingEquiv.toEquiv
  map_one' := rfl
  map_mul' _ _ := rfl


/-- The tautological action by the group of automorphism of a ring `R` on `R`. -/
instance applyMulSemiringAction :
    MulSemiringAction (RingAut R) R where
  smul := (· <| ·)
  smul_zero := RingEquiv.map_zero
  smul_add := RingEquiv.map_add
  smul_one := RingEquiv.map_one
  smul_mul := RingEquiv.map_mul
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp]
protected theorem smul_def (f : RingAut R) (r : R) : f • r = f r :=
  rfl


instance apply_faithfulSMul : FaithfulSMul (RingAut R) R :=
  ⟨RingEquiv.ext⟩


/-- Each element of the group defines a ring automorphism.

This is a stronger version of `DistribMulAction.toAddAut` and
`MulDistribMulAction.toMulAut`. -/
@[simps]
def _root_.MulSemiringAction.toRingAut [MulSemiringAction G R] :
    G →* RingAut R where
  toFun := MulSemiringAction.toRingEquiv G R
  map_mul' g h := RingEquiv.ext <| mul_smul g h
  map_one' := RingEquiv.ext <| one_smul _


