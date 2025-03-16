local macro:max "ℝ"n:superscript(term) : term => `(EuclideanSpace ℝ (Fin $(⟨n.raw[0]⟩)))

local macro:max "𝕊"n:superscript(term) : term =>
  `(sphere (0 : EuclideanSpace ℝ (Fin ($(⟨n.raw[0]⟩) + 1))) 1)


/-- The smooth Poincaré conjecture; true for n = 1, 2, 3, 5, 6, 12, 56, and 61,
open for n = 4, and it is conjectured that there are no other n > 4 for which it is true
(Conjecture 1.17, https://annals.math.princeton.edu/2017/186-2/p03). -/
def ContinuousMap.HomotopyEquiv.NonemptyDiffeomorphSphere (n : ℕ) : Prop :=
  ∀ (_ : ChartedSpace ℝⁿ M) (_ : SmoothManifoldWithCorners (𝓡 n) M),
    M ≃ₕ 𝕊ⁿ → Nonempty (M ≃ₘ⟮𝓡 n, 𝓡 n⟯ 𝕊ⁿ)


