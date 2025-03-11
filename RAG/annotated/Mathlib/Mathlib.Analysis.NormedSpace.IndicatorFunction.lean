theorem norm_indicator_eq_indicator_norm : ‖indicator s f a‖ = indicator s (fun a => ‖f a‖) a :=
  flip congr_fun a (indicator_comp_of_zero norm_zero).symm


theorem nnnorm_indicator_eq_indicator_nnnorm :
    ‖indicator s f a‖₊ = indicator s (fun a => ‖f a‖₊) a :=
  flip congr_fun a (indicator_comp_of_zero nnnorm_zero).symm


theorem norm_indicator_le_of_subset (h : s ⊆ t) (f : α → E) (a : α) :
    ‖indicator s f a‖ ≤ ‖indicator t f a‖ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : SeminormedAddCommGroup E
    s t : Set α
    h : HasSubset.Subset s t
    f : α → E
    a : α
    ⊢ LE.le (Norm.norm (s.indicator f a)) (Norm.norm (t.indicator f a))
  -/
  simp only [norm_indicator_eq_indicator_norm]
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : SeminormedAddCommGroup E
    s t : Set α
    h : HasSubset.Subset s t
    f : α → E
    a : α
    ⊢ LE.le (s.indicator (fun a => Norm.norm (f a)) a) (t.indicator (fun a => Norm …
  -/
  exact indicator_le_indicator_of_subset ‹_› (fun _ => norm_nonneg _) _
  /-
    🎉 no goals
  -/


theorem indicator_norm_le_norm_self : indicator s (fun a => ‖f a‖) a ≤ ‖f a‖ :=
  indicator_le_self' (fun _ _ => norm_nonneg _) a


theorem norm_indicator_le_norm_self : ‖indicator s f a‖ ≤ ‖f a‖ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : SeminormedAddCommGroup E
    s : Set α
    f : α → E
    a : α
    ⊢ LE.le (Norm.norm (s.indicator f a)) (Norm.norm (f a))
  -/
  rw [norm_indicator_eq_indicator_norm]
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : SeminormedAddCommGroup E
    s : Set α
    f : α → E
    a : α
    ⊢ LE.le (s.indicator (fun a => Norm.norm (f a)) a) (Norm.norm (f a))
  -/
  apply indicator_norm_le_norm_self
  /-
    🎉 no goals
  -/

