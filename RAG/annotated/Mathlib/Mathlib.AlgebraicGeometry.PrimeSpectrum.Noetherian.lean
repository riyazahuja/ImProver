instance : NoetherianSpace (PrimeSpectrum R) :=
   /-
     R : Type u
     inst✝¹ : CommRing R
     inst✝ : IsNoetherianRing R
     ⊢ Eq ((List.cons (TopologicalSpace.NoetherianSpace (PrimeSpectrum R)) (List.co …
   -/
   /-
     🎉 no goals
   -/
  ((noetherianSpace_TFAE <| PrimeSpectrum R).out 0 1).mpr (closedsEmbedding R).dual.wellFoundedLT
   /-
     🎉 no goals
   -/


lemma _root_.minimalPrimes.finite_of_isNoetherianRing : (minimalPrimes R).Finite :=
  minimalPrimes.equivIrreducibleComponents R
    |>.set_finite_iff
    |>.mpr NoetherianSpace.finite_irreducibleComponents


lemma finite_setOf_isMin :
    {x : PrimeSpectrum R | IsMin x }.Finite := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ (setOf fun x => IsMin x).Finite
  -/
  have : Function.Injective (asIdeal (R := R)) := @PrimeSpectrum.ext _ _
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    this : Function.Injective PrimeSpectrum.asIdeal
    ⊢ (setOf fun x => IsMin x).Finite
  -/
  refine Set.Finite.of_finite_image (f := asIdeal) ?_ this.injOn
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    this : Function.Injective PrimeSpectrum.asIdeal
    ⊢ (Set.image PrimeSpectrum.asIdeal (setOf fun x => IsMin x)).Finite
  -/
  simp_rw [isMin_iff]
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    this : Function.Injective PrimeSpectrum.asIdeal
    ⊢ (Set.image PrimeSpectrum.asIdeal (setOf fun x => Membership.mem (minimalPrim …
  -/
  exact (minimalPrimes.finite_of_isNoetherianRing R).subset (Set.image_preimage_subset _ _)
  /-
    🎉 no goals
  -/


