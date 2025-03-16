instance wellPowered_addCommGrp : WellPowered.{u} AddCommGrp.{u} :=
  wellPowered_of_equiv (forget₂ (ModuleCat.{u} ℤ) AddCommGrp.{u}).asEquivalence


