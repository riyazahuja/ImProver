/-- The subgroup of positive units of a linear ordered semiring. -/
def Units.posSubgroup (R : Type*) [LinearOrderedSemiring R] : Subgroup Rˣ :=
  { (Submonoid.pos R).comap (Units.coeHom R) with
    carrier := { x | (0 : R) < x }
    inv_mem' := Units.inv_pos.mpr }


@[simp]
theorem Units.mem_posSubgroup {R : Type*} [LinearOrderedSemiring R] (u : Rˣ) :
    u ∈ Units.posSubgroup R ↔ (0 : R) < u :=
  Iff.rfl

