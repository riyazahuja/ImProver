theorem mdifferentiable_jacobiTheta : MDifferentiable 𝓘(ℂ) 𝓘(ℂ) (jacobiTheta ∘ (↑) : ℍ → ℂ) :=
  fun τ => (differentiableAt_jacobiTheta τ.2).mdifferentiableAt.comp τ τ.mdifferentiable_coe

