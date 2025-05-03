/-- The uniform structure coming from an absolute value. -/
def uniformSpace : UniformSpace R :=
                                      /-
                                        𝕜 : Type u_1
                                        inst✝¹ : LinearOrderedField 𝕜
                                        R : Type u_2
                                        inst✝ : CommRing R
                                        abv : AbsoluteValue R 𝕜
                                        ⊢ ∀ (x : R), Eq ((fun x y => abv (HSub.hSub y x)) x x) 0
                                      -/
  .ofFun (fun x y => abv (y - x)) (by simp) (fun x y => abv.map_sub y x)
                                      /-
                                        🎉 no goals
                                      -/
    (fun _ _ _ => (abv.sub_le _ _ _).trans_eq (add_comm _ _))
    fun ε ε0 => ⟨ε / 2, half_pos ε0, fun _ h₁ _ h₂ => (add_lt_add h₁ h₂).trans_eq (add_halves ε)⟩


theorem hasBasis_uniformity :
    𝓤[abv.uniformSpace].HasBasis ((0 : 𝕜) < ·) fun ε => { p : R × R | abv (p.2 - p.1) < ε } :=
  UniformSpace.hasBasis_ofFun (exists_gt _) _ _ _ _ _


