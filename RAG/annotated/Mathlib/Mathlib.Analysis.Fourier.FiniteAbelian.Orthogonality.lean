lemma expect_eq_ite (ψ : AddChar G R) : 𝔼 a, ψ a = if ψ = 0 then 1 else 0 := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝⁴ : AddGroup G
    inst✝³ : Fintype G
    inst✝² : Semifield R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    ψ : AddChar G R
    ⊢ Eq (Finset.univ.expect fun a => ψ a) (ite (Eq ψ 0) 1 0)
  -/
  simp [Fintype.expect_eq_sum_div_card, sum_eq_ite, ite_div]
  /-
    🎉 no goals
  -/


lemma expect_eq_zero_iff_ne_zero : 𝔼 x, ψ x = 0 ↔ ψ ≠ 0 := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝⁴ : AddGroup G
    inst✝³ : Fintype G
    inst✝² : Semifield R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    ψ : AddChar G R
    ⊢ Iff (Eq (Finset.univ.expect fun x => ψ x) 0) (Ne ψ 0)
  -/
  rw [expect_eq_ite, one_ne_zero.ite_eq_right_iff]
  /-
    🎉 no goals
  -/


lemma expect_ne_zero_iff_eq_zero : 𝔼 x, ψ x ≠ 0 ↔ ψ = 0 := expect_eq_zero_iff_ne_zero.not_left


lemma wInner_cWeight_self (ψ : AddChar G R) : ⟪(ψ : G → R), ψ⟫ₙ_[R] = 1 := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝² : AddGroup G
    inst✝¹ : RCLike R
    inst✝ : Fintype G
    ψ : AddChar G R
    ⊢ Eq (RCLike.wInner RCLike.cWeight ⇑ψ ⇑ψ) 1
  -/
  simp [wInner_cWeight_eq_expect, ψ.norm_apply, RCLike.conj_mul]
  /-
    🎉 no goals
  -/


lemma wInner_cWeight_eq_boole [Fintype G] (ψ₁ ψ₂ : AddChar G R) :
    ⟪(ψ₁ : G → R), ψ₂⟫ₙ_[R] = if ψ₁ = ψ₂ then 1 else 0 := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    inst✝ : Fintype G
    ψ₁ ψ₂ : AddChar G R
    ⊢ Eq (RCLike.wInner RCLike.cWeight ⇑ψ₁ ⇑ψ₂) (ite (Eq ψ₁ ψ₂) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      G : Type u_1
      R : Type u_3
      inst✝² : AddCommGroup G
      inst✝¹ : RCLike R
      inst✝ : Fintype G
      ψ₁ ψ₂ : AddChar G R
      h : Eq ψ₁ ψ₂
      ⊢ Eq (RCLike.wInner RCLike.cWeight ⇑ψ₁ ⇑ψ₂) 1
    -/
  · rw [h, wInner_cWeight_self]
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    inst✝ : Fintype G
    ψ₁ ψ₂ : AddChar G R
    h : Not (Eq ψ₁ ψ₂)
    ⊢ Eq (RCLike.wInner RCLike.cWeight ⇑ψ₁ ⇑ψ₂) 0
  -/
  have : ψ₁⁻¹ * ψ₂ ≠ 1 := by rwa [Ne, inv_mul_eq_one]
  /-
    case neg
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    inst✝ : Fintype G
    ψ₁ ψ₂ : AddChar G R
    h : Not (Eq ψ₁ ψ₂)
    this : Ne (HMul.hMul (Inv.inv ψ₁) ψ₂) 1
    ⊢ Eq (RCLike.wInner RCLike.cWeight ⇑ψ₁ ⇑ψ₂) 0
  -/
  simp_rw [wInner_cWeight_eq_expect, RCLike.inner_apply, ← inv_apply_eq_conj]
  /-
    case neg
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    inst✝ : Fintype G
    ψ₁ ψ₂ : AddChar G R
    h : Not (Eq ψ₁ ψ₂)
    this : Ne (HMul.hMul (Inv.inv ψ₁) ψ₂) 1
    ⊢ Eq (Finset.univ.expect fun i => HMul.hMul (Inv.inv (ψ₁ i)) (ψ₂ i)) 0
  -/
  simpa [map_neg_eq_inv] using expect_eq_zero_iff_ne_zero.2 this
  /-
    🎉 no goals
  -/


lemma wInner_cWeight_eq_zero_iff_ne [Fintype G] : ⟪(ψ₁ : G → R), ψ₂⟫ₙ_[R] = 0 ↔ ψ₁ ≠ ψ₂ := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    ψ₁ ψ₂ : AddChar G R
    inst✝ : Fintype G
    ⊢ Iff (Eq (RCLike.wInner RCLike.cWeight ⇑ψ₁ ⇑ψ₂) 0) (Ne ψ₁ ψ₂)
  -/
  rw [wInner_cWeight_eq_boole, one_ne_zero.ite_eq_right_iff]
  /-
    🎉 no goals
  -/


lemma wInner_cWeight_eq_one_iff_eq [Fintype G] : ⟪(ψ₁ : G → R), ψ₂⟫ₙ_[R] = 1 ↔ ψ₁ = ψ₂ := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    ψ₁ ψ₂ : AddChar G R
    inst✝ : Fintype G
    ⊢ Iff (Eq (RCLike.wInner RCLike.cWeight ⇑ψ₁ ⇑ψ₂) 1) (Eq ψ₁ ψ₂)
  -/
  rw [wInner_cWeight_eq_boole, one_ne_zero.ite_eq_left_iff]
  /-
    🎉 no goals
  -/


protected lemma linearIndependent [Finite G] : LinearIndependent R ((⇑) : AddChar G R → G → R) := by
  /-
    G : Type u_1
    R : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : RCLike R
    inst✝ : Finite G
    ⊢ LinearIndependent R DFunLike.coe
  -/
  cases nonempty_fintype G
  exact linearIndependent_of_ne_zero_of_wInner_cWeight_eq_zero coe_ne_zero
    fun ψ₁ ψ₂ ↦ wInner_cWeight_eq_zero_iff_ne.2


noncomputable instance instFintype [Finite G] : Fintype (AddChar G R) :=
  @Fintype.ofFinite _ (AddChar.linearIndependent G R).finite


@[simp] lemma card_addChar_le [Fintype G] : card (AddChar G R) ≤ card G := by
  simpa only [Module.finrank_fintype_fun_eq_card] using
    (AddChar.linearIndependent G R).fintype_card_le_finrank


