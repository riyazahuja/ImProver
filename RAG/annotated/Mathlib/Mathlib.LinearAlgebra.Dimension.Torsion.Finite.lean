lemma rank_eq_zero_iff_isTorsion: Module.rank R M = 0 ↔ Module.IsTorsion R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Eq (Module.rank R M) 0) (Module.IsTorsion R M)
  -/
  rw [Module.IsTorsion, rank_eq_zero_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)) (∀ ⦃x …
  -/
  simp [mem_nonZeroDivisors_iff_ne_zero]
  /-
    🎉 no goals
  -/


/-- The `StrongRankCondition` is automatic. See `commRing_strongRankCondition`. -/
theorem Module.finrank_eq_zero_iff_isTorsion [StrongRankCondition R] [Module.Finite R M] :
    finrank R M = 0 ↔ Module.IsTorsion R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ Iff (Eq (Module.finrank R M) 0) (Module.IsTorsion R M)
  -/
  rw [← rank_eq_zero_iff_isTorsion (R := R), ← finrank_eq_rank]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ Iff (Eq (Module.finrank R M) 0) (Eq (↑(Module.finrank R M)) 0)
  -/
  norm_cast
  /-
    🎉 no goals
  -/

