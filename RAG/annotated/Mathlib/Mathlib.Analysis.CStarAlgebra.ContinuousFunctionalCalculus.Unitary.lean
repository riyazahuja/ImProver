lemma cfc_unitary_iff (f : R → R) (a : A) (ha : p a := by cfc_tac)
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) :
    cfc f a ∈ unitary A ↔ ∀ x ∈ spectrum R a, star (f x) * f x = 1 := by
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommRing R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalRing R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ha : autoParam (p a) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ Iff (Membership.mem (unitary A) (cfc f a)) (∀ (x : R), Membership.mem (spect …
  -/
  simp only [unitary, Submonoid.mem_mk, Subsemigroup.mem_mk, Set.mem_setOf_eq]
  rw [← IsStarNormal.cfc_map (p := p) f a |>.star_comm_self |>.eq, and_self, ← cfc_one R a,
    ← cfc_star, ← cfc_mul .., cfc_eq_cfc_iff_eqOn]
  /-
    R : Type u_1
    A : Type u_2
    p : A → Prop
    inst✝⁹ : CommRing R
    inst✝⁸ : StarRing R
    inst✝⁷ : MetricSpace R
    inst✝⁶ : TopologicalRing R
    inst✝⁵ : ContinuousStar R
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : ContinuousFunctionalCalculus R p
    f : R → R
    a : A
    ha : autoParam (p a) _auto✝
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    ⊢ Iff (Set.EqOn (fun x => HMul.hMul (Star.star (f x)) (f x)) 1 (spectrum R a)) …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


lemma unitary_iff_isStarNormal_and_spectrum_subset_unitary {u : A} :
    u ∈ unitary A ↔ IsStarNormal u ∧ spectrum ℂ u ⊆ unitary ℂ := by
  /-
    A : Type u_1
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    u : A
    ⊢ Iff (Membership.mem (unitary A) u) (And (IsStarNormal u) (HasSubset.Subset ( …
  -/
  rw [← and_iff_right_of_imp isStarNormal_of_mem_unitary]
  /-
    A : Type u_1
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    u : A
    ⊢ Iff (And (IsStarNormal u) (Membership.mem (unitary A) u)) (And (IsStarNormal …
  -/
  refine and_congr_right fun hu ↦ ?_
  /-
    A : Type u_1
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    u : A
    hu : IsStarNormal u
    ⊢ Iff (Membership.mem (unitary A) u) (HasSubset.Subset (spectrum Complex u) ↑( …
  -/
  nth_rw 1 [← cfc_id ℂ u]
  /-
    A : Type u_1
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    u : A
    hu : IsStarNormal u
    ⊢ Iff (Membership.mem (unitary A) (cfc id u)) (HasSubset.Subset (spectrum Comp …
  -/
  rw [cfc_unitary_iff id u, Set.subset_def]
  /-
    A : Type u_1
    inst✝⁴ : TopologicalSpace A
    inst✝³ : Ring A
    inst✝² : StarRing A
    inst✝¹ : Algebra Complex A
    inst✝ : ContinuousFunctionalCalculus Complex IsStarNormal
    u : A
    hu : IsStarNormal u
    ⊢ Iff (∀ (x : Complex), Membership.mem (spectrum Complex u) x → Eq (HMul.hMul  …
  -/
  simp only [id_eq, RCLike.star_def, SetLike.mem_coe, unitary.mem_iff_star_mul_self]
  /-
    🎉 no goals
  -/


lemma mem_unitary_of_spectrum_subset_unitary {u : A}
    [IsStarNormal u] (hu : spectrum ℂ u ⊆ unitary ℂ) : u ∈ unitary A :=
  unitary_iff_isStarNormal_and_spectrum_subset_unitary.mpr ⟨‹_›, hu⟩


lemma spectrum_subset_unitary_of_mem_unitary {u : A} (hu : u ∈ unitary A) :
    spectrum ℂ u ⊆ unitary ℂ :=
  unitary_iff_isStarNormal_and_spectrum_subset_unitary.mp hu |>.right


