instance : DecompositionMonoid R := MulEquiv.decompositionMonoid (equivPi R)


instance : DecompositionMonoid (Polynomial R) :=
  MulEquiv.decompositionMonoid <| (Polynomial.mapEquiv <| equivPi R).trans (Polynomial.piEquiv _)


theorem isSemisimpleRing_of_isReduced : IsSemisimpleRing R := (equivPi R).symm.isSemisimpleRing


