/-- `Set α` has distributive negation if `α` has. -/
protected noncomputable def hasDistribNeg [Mul α] [HasDistribNeg α] : HasDistribNeg (Set α) where
  __ := Set.involutiveNeg
                    /-
                      α : Type u_1
                      inst✝¹ : Mul α
                      inst✝ : HasDistribNeg α
                      x✝¹ x✝ : Set α
                      ⊢ Eq (HMul.hMul (Neg.neg x✝¹) x✝) (Neg.neg (HMul.hMul x✝¹ x✝))
                    -/
  neg_mul _ _ := by simp_rw [← image_neg_eq_neg]; exact image2_image_left_comm neg_mul
                                                  /-
                                                    🎉 no goals
                                                  -/
                    /-
                      α : Type u_1
                      inst✝¹ : Mul α
                      inst✝ : HasDistribNeg α
                      x✝¹ x✝ : Set α
                      ⊢ Eq (HMul.hMul x✝¹ (Neg.neg x✝)) (Neg.neg (HMul.hMul x✝¹ x✝))
                    -/
  mul_neg _ _ := by simp_rw [← image_neg_eq_neg]; exact image_image2_right_comm mul_neg
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma mul_add_subset : s * (t + u) ⊆ s * t + s * u := image2_distrib_subset_left mul_add

lemma add_mul_subset : (s + t) * u ⊆ s * u + t * u := image2_distrib_subset_right add_mul


