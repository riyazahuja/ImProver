/-- A proper submodule `S : Submodule R M` is primary iff
  `r • x ∈ S` implies `x ∈ S` or `∃ n : ℕ, r ^ n • (⊤ : Submodule R M) ≤ S`.
  This generalizes `Ideal.IsPrimary`. -/
protected def IsPrimary (S : Submodule R M) : Prop :=
  S ≠ ⊤ ∧ ∀ {r : R} {x : M}, r • x ∈ S → x ∈ S ∨ ∃ n : ℕ, (r ^ n • ⊤ : Submodule R M) ≤ S


lemma IsPrimary.ne_top (h : S.IsPrimary) : S ≠ ⊤ := h.left


lemma isPrimary_iff_zero_divisor_quotient_imp_nilpotent_smul :
    S.IsPrimary ↔ S ≠ ⊤ ∧ ∀ (r : R) (x : M ⧸ S), x ≠ 0 → r • x = 0 →
      ∃ n : ℕ, r ^ n • (⊤ : Submodule R (M ⧸ S)) = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    ⊢ Iff S.IsPrimary (And (Ne S Top.top) (∀ (r : R) (x : HasQuotient.Quotient M S …
  -/
  refine (and_congr_right fun _ ↦ ?_)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    x✝ : Ne S Top.top
    ⊢ Iff (∀ {r : R} {x : M}, Membership.mem S (HSMul.hSMul r x) → Or (Membership. …
  -/
  simp_rw [S.mkQ_surjective.forall, ← map_smul, ne_eq, ← LinearMap.mem_ker, ker_mkQ]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    x✝ : Ne S Top.top
    ⊢ Iff (∀ {r : R} {x : M}, Membership.mem S (HSMul.hSMul r x) → Or (Membership. …
  -/
  congr! 2
  rw [forall_comm, ← or_iff_not_imp_left,
    ← LinearMap.range_eq_top.mpr S.mkQ_surjective, ← map_top]
  /-
    case a.h.h.a
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    x✝ : Ne S Top.top
    a✝¹ : R
    a✝ : M
    ⊢ Iff (Membership.mem S (HSMul.hSMul a✝¹ a✝) → Or (Membership.mem S a✝) (Exist …
  -/
  simp_rw [eq_bot_iff, ← map_pointwise_smul, map_le_iff_le_comap, comap_bot, ker_mkQ]
  /-
    🎉 no goals
  -/


