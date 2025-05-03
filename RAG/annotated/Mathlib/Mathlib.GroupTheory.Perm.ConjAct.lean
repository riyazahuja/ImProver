/-- `a : α` belongs to the support of `k • g` iff
  `k⁻¹ * a` belongs to the support of `g` -/
theorem mem_conj_support (k : ConjAct (Perm α)) (g : Perm α) (a : α) :
    a ∈ (k • g).support ↔ ConjAct.ofConjAct k⁻¹ a ∈ g.support := by
  simp only [mem_support, ConjAct.smul_def, not_iff_not, coe_mul,
    Function.comp_apply, ConjAct.ofConjAct_inv]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g : Equiv.Perm α
    a : α
    ⊢ Iff (Eq ((ConjAct.ofConjAct k) (g ((Inv.inv (ConjAct.ofConjAct k)) a))) a) ( …
  -/
  apply Equiv.apply_eq_iff_eq_symm_apply
  /-
    🎉 no goals
  -/


theorem cycleFactorsFinset_conj (g k : Perm α) :
    (ConjAct.toConjAct k • g).cycleFactorsFinset =
      Finset.map (MulAut.conj k).toEquiv.toEmbedding g.cycleFactorsFinset := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k : Equiv.Perm α
    ⊢ Eq (HSMul.hSMul (ConjAct.toConjAct k) g).cycleFactorsFinset (Finset.map (Mul …
  -/
  ext c
  rw [ConjAct.smul_def, ConjAct.ofConjAct_toConjAct, Finset.mem_map_equiv,
    ← mem_cycleFactorsFinset_conj g k]
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    ⊢ Iff (Membership.mem (HMul.hMul (HMul.hMul k g) (Inv.inv k)).cycleFactorsFins …
  -/
  simp only [MulEquiv.toEquiv_eq_coe, MulEquiv.coe_toEquiv_symm, MulAut.conj_symm_apply]
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g k c : Equiv.Perm α
    ⊢ Iff (Membership.mem (HMul.hMul (HMul.hMul k g) (Inv.inv k)).cycleFactorsFins …
  -/
  group
  /-
    🎉 no goals
  -/


/-- A permutation `c` is a cycle of `g` iff `k • c` is a cycle of `k • g` -/
@[simp]
theorem mem_cycleFactorsFinset_conj'
    (k : ConjAct (Perm α)) (g c : Perm α) :
    k • c ∈ (k • g).cycleFactorsFinset ↔ c ∈ g.cycleFactorsFinset := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g c : Equiv.Perm α
    ⊢ Iff (Membership.mem (HSMul.hSMul k g).cycleFactorsFinset (HSMul.hSMul k c))  …
  -/
  simp only [ConjAct.smul_def]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g c : Equiv.Perm α
    ⊢ Iff (Membership.mem (HMul.hMul (HMul.hMul (ConjAct.ofConjAct k) g) (Inv.inv  …
  -/
  apply mem_cycleFactorsFinset_conj g k
  /-
    🎉 no goals
  -/


theorem cycleFactorsFinset_conj_eq
    (k : ConjAct (Perm α)) (g : Perm α) :
    cycleFactorsFinset (k • g) = k • cycleFactorsFinset g := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g : Equiv.Perm α
    ⊢ Eq (HSMul.hSMul k g).cycleFactorsFinset (HSMul.hSMul k g.cycleFactorsFinset)
  -/
  ext c
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g c : Equiv.Perm α
    ⊢ Iff (Membership.mem (HSMul.hSMul k g).cycleFactorsFinset c) (Membership.mem  …
  -/
  rw [← mem_cycleFactorsFinset_conj' k⁻¹ (k • g) c]
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g c : Equiv.Perm α
    ⊢ Iff (Membership.mem (HSMul.hSMul (Inv.inv k) (HSMul.hSMul k g)).cycleFactors …
  -/
  simp only [inv_smul_smul]
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : ConjAct (Equiv.Perm α)
    g c : Equiv.Perm α
    ⊢ Iff (Membership.mem g.cycleFactorsFinset (HSMul.hSMul (Inv.inv k) c)) (Membe …
  -/
  exact Finset.inv_smul_mem_iff
  /-
    🎉 no goals
  -/


