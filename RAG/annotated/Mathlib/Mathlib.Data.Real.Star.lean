/-- The real numbers are a `*`-ring, with the trivial `*`-structure. -/
instance : StarRing ℝ :=
  starRingOfComm


instance : TrivialStar ℝ :=
  ⟨fun _ => rfl⟩

