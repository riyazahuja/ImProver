theorem smooth_barycentric_coord (b : AffineBasis ι 𝕜 E) (i : ι) : ContDiff 𝕜 ⊤ (b.coord i) :=
  (⟨b.coord i, continuous_barycentric_coord b i⟩ : E →ᴬ[𝕜] 𝕜).contDiff

