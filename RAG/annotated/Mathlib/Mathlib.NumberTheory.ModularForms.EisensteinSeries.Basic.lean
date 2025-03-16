/-- This defines Eisenstein series as modular forms of weight `k`, level `Γ(N)` and congruence
condition given by `a: Fin 2 → ZMod N`. -/
def eisensteinSeries_MF {k : ℤ} {N : ℕ+} (hk : 3 ≤ k) (a : Fin 2 → ZMod N) :
    ModularForm (Gamma N) k where
  toFun := eisensteinSeries_SIF a k
  slash_action_eq' := (eisensteinSeries_SIF a k).slash_action_eq'
  holo' := eisensteinSeries_SIF_MDifferentiable hk a
  bdd_at_infty' := isBoundedAtImInfty_eisensteinSeries_SIF a hk


