/-- The ring topology on `Localization M` coinduced from the natural homomorphism sending `x : R`
to the equivalence class of `(x, 1)`. -/
def Localization.ringTopology : RingTopology (Localization M) :=
  RingTopology.coinduced (Localization.monoidOf M).toFun


instance : TopologicalSpace (Localization M) :=
  Localization.ringTopology.toTopologicalSpace


instance : TopologicalRing (Localization M) :=
  Localization.ringTopology.toTopologicalRing

