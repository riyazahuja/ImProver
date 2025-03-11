instance (priority := 900) RCLike.measurableSpace {𝕜 : Type*} [RCLike 𝕜] : MeasurableSpace 𝕜 :=
  borel 𝕜


instance (priority := 900) RCLike.borelSpace {𝕜 : Type*} [RCLike 𝕜] : BorelSpace 𝕜 :=
  ⟨rfl⟩


instance Complex.measurableSpace : MeasurableSpace ℂ :=
  borel ℂ


instance Complex.borelSpace : BorelSpace ℂ :=
  ⟨rfl⟩

