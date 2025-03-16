/-- `Finset α` has distributive negation if `α` has. -/
protected def distribNeg [DecidableEq α] [Mul α] [HasDistribNeg α] : HasDistribNeg (Finset α) :=
  coe_injective.hasDistribNeg _ coe_neg coe_mul


lemma mul_add_subset : s * (t + u) ⊆ s * t + s * u :=
  image₂_distrib_subset_left mul_add


lemma add_mul_subset : (s + t) * u ⊆ s * u + t * u :=
  image₂_distrib_subset_right add_mul


@[simp]
lemma neg_smul_finset : -a • t = -(a • t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Ring α
    inst✝² : AddCommGroup β
    inst✝¹ : Module α β
    inst✝ : DecidableEq β
    t : Finset β
    a : α
    ⊢ Eq (HSMul.hSMul (Neg.neg a) t) (Neg.neg (HSMul.hSMul a t))
  -/
  simp only [← image_smul, ← image_neg_eq_neg, image_image, neg_smul, Function.comp_def]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma neg_smul [DecidableEq α] : -s • t = -(s • t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : Ring α
    inst✝³ : AddCommGroup β
    inst✝² : Module α β
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq α
    ⊢ Eq (HSMul.hSMul (Neg.neg s) t) (Neg.neg (HSMul.hSMul s t))
  -/
  simp_rw [← image_neg_eq_neg]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : Ring α
    inst✝³ : AddCommGroup β
    inst✝² : Module α β
    inst✝¹ : DecidableEq β
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq α
    ⊢ Eq (HSMul.hSMul (Finset.image (fun x => Neg.neg x) s) t) (Finset.image (fun  …
  -/
  exact image₂_image_left_comm neg_smul
  /-
    🎉 no goals
  -/


