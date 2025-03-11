/-- Translation in the domain does not change the derivative. -/
lemma HasDerivAt.comp_const_add (a x : 𝕜) (hf : HasDerivAt f f' (a + x)) :
    HasDerivAt (fun x ↦ f (a + x)) f' x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    a x : 𝕜
    hf : HasDerivAt f f' (HAdd.hAdd a x)
    ⊢ HasDerivAt (fun x => f (HAdd.hAdd a x)) f' x
  -/
  simpa [Function.comp_def] using HasDerivAt.scomp (𝕜 := 𝕜) x hf <| hasDerivAt_id' x |>.const_add a
  /-
    🎉 no goals
  -/


/-- Translation in the domain does not change the derivative. -/
lemma HasDerivAt.comp_add_const (x a : 𝕜) (hf : HasDerivAt f f' (x + a)) :
    HasDerivAt (fun x ↦ f (x + a)) f' x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x a : 𝕜
    hf : HasDerivAt f f' (HAdd.hAdd x a)
    ⊢ HasDerivAt (fun x => f (HAdd.hAdd x a)) f' x
  -/
  simpa [Function.comp_def] using HasDerivAt.scomp (𝕜 := 𝕜) x hf <| hasDerivAt_id' x |>.add_const a
  /-
    🎉 no goals
  -/


/-- Translation in the domain does not change the derivative. -/
lemma HasDerivAt.comp_const_sub (a x : 𝕜) (hf : HasDerivAt f f' (a - x)) :
    HasDerivAt (fun x ↦ f (a - x)) (-f') x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    a x : 𝕜
    hf : HasDerivAt f f' (HSub.hSub a x)
    ⊢ HasDerivAt (fun x => f (HSub.hSub a x)) (Neg.neg f') x
  -/
  simpa [Function.comp_def] using HasDerivAt.scomp (𝕜 := 𝕜) x hf <| hasDerivAt_id' x |>.const_sub a
  /-
    🎉 no goals
  -/


/-- Translation in the domain does not change the derivative. -/
lemma HasDerivAt.comp_sub_const (x a : 𝕜) (hf : HasDerivAt f f' (x - a)) :
    HasDerivAt (fun x ↦ f (x - a)) f' x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x a : 𝕜
    hf : HasDerivAt f f' (HSub.hSub x a)
    ⊢ HasDerivAt (fun x => f (HSub.hSub x a)) f' x
  -/
  simpa [Function.comp_def] using HasDerivAt.scomp (𝕜 := 𝕜) x hf <| hasDerivAt_id' x |>.sub_const a
  /-
    🎉 no goals
  -/


/-- The derivative of `x ↦ f (-x)` at `a` is the negative of the derivative of `f` at `-a`. -/
lemma deriv_comp_neg : deriv (fun x ↦ f (-x)) x = -deriv f (-x) := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (deriv (fun x => f (Neg.neg x)) x) (Neg.neg (deriv f (Neg.neg x)))
  -/
  by_cases f : DifferentiableAt 𝕜 f (-x)
    /-
      case pos
      𝕜 : Type u_1
      F : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f✝ : 𝕜 → F
      x : 𝕜
      f : DifferentiableAt 𝕜 f✝ (Neg.neg x)
      ⊢ Eq (deriv (fun x => f✝ (Neg.neg x)) x) (Neg.neg (deriv f✝ (Neg.neg x)))
    -/
  · simpa only [deriv_neg, neg_one_smul] using deriv.scomp _ f (differentiable_neg _)
    /-
      🎉 no goals
    -/
  · rw [deriv_zero_of_not_differentiableAt (differentiableAt_comp_neg.not.2 f),
      deriv_zero_of_not_differentiableAt f, neg_zero]


/-- Translation in the domain does not change the derivative. -/
lemma deriv_comp_const_add : deriv (fun x ↦ f (a + x)) x = deriv f (a + x) := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a x : 𝕜
    ⊢ Eq (deriv (fun x => f (HAdd.hAdd a x)) x) (deriv f (HAdd.hAdd a x))
  -/
  by_cases hf : DifferentiableAt 𝕜 f (a + x)
    /-
      case pos
      𝕜 : Type u_1
      F : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      a x : 𝕜
      hf : DifferentiableAt 𝕜 f (HAdd.hAdd a x)
      ⊢ Eq (deriv (fun x => f (HAdd.hAdd a x)) x) (deriv f (HAdd.hAdd a x))
    -/
  · exact HasDerivAt.deriv hf.hasDerivAt.comp_const_add
    /-
      🎉 no goals
    -/
  · rw [deriv_zero_of_not_differentiableAt (differentiableAt_comp_const_add.not.2 hf),
      deriv_zero_of_not_differentiableAt hf]


/-- Translation in the domain does not change the derivative. -/
lemma deriv_comp_add_const : deriv (fun x ↦ f (x + a)) x = deriv f (x + a) := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a x : 𝕜
    ⊢ Eq (deriv (fun x => f (HAdd.hAdd x a)) x) (deriv f (HAdd.hAdd x a))
  -/
  simpa [add_comm] using deriv_comp_const_add f a x
  /-
    🎉 no goals
  -/


lemma deriv_comp_const_sub : deriv (fun x ↦ f (a - x)) x = -deriv f (a - x) := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a x : 𝕜
    ⊢ Eq (deriv (fun x => f (HSub.hSub a x)) x) (Neg.neg (deriv f (HSub.hSub a x)))
  -/
  simp_rw [sub_eq_add_neg, deriv_comp_neg (f <| a + ·), deriv_comp_const_add]
  /-
    🎉 no goals
  -/


lemma deriv_comp_sub_const : deriv (fun x ↦ f (x - a)) x = deriv f (x - a) := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a x : 𝕜
    ⊢ Eq (deriv (fun x => f (HSub.hSub x a)) x) (deriv f (HSub.hSub x a))
  -/
  simp_rw [sub_eq_add_neg, deriv_comp_add_const]
  /-
    🎉 no goals
  -/

