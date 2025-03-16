instance : EnoughInjectives (ModuleCat.{v} ℤ) :=
  EnoughInjectives.of_equivalence (forget₂ (ModuleCat ℤ) AddCommGrp)


lemma ModuleCat.enoughInjectives : EnoughInjectives (ModuleCat.{max v u} R) :=
  EnoughInjectives.of_adjunction (ModuleCat.restrictCoextendScalarsAdj.{max v u} (algebraMap ℤ R))


open ModuleCat in
instance [UnivLE.{u,v}] : EnoughInjectives (ModuleCat.{v} R) :=
  letI := (equivShrink.{v} R).symm.ring
  letI := enoughInjectives.{v} (Shrink.{v} R)
  EnoughInjectives.of_equivalence (restrictScalars (equivShrink R).symm.ringEquiv.toRingHom)

