@[to_additive]
private lemma smul_stabilizer_of_no_doubling_aux (hA : #(A * A) ≤ #A) (ha : a ∈ A) :
    a •> (stabilizer G A : Set G) = A ∧ (stabilizer G A : Set G) <• a = A := by
  have smul_A {a} (ha : a ∈ A) : a •> A = A * A :=
    eq_of_subset_of_card_le (smul_finset_subset_mul ha) (by simpa)
  have A_smul {a} (ha : a ∈ A) : A <• a = A * A :=
    eq_of_subset_of_card_le (op_smul_finset_subset_mul ha) (by simpa)
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A : Finset G
    a : G
    hA : LE.le (HMul.hMul A A).card A.card
    ha : Membership.mem A a
    smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
    A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
    ⊢ And (Eq (HSMul.hSMul a ↑(MulAction.stabilizer G A)) ↑A) (Eq (HSMul.hSMul (Mu …
  -/
  have smul_A_eq_A_smul {a} (ha : a ∈ A) : a •> A = A <• a := by rw [smul_A ha, A_smul ha]
  have mul_mem_A_comm {x a} (ha : a ∈ A) : x * a ∈ A ↔ a * x ∈ A := by
    rw [← smul_mem_smul_finset_iff a, smul_A_eq_A_smul ha, ← op_smul_eq_mul, smul_comm,
      smul_mem_smul_finset_iff, smul_eq_mul]
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A : Finset G
    a : G
    hA : LE.le (HMul.hMul A A).card A.card
    ha : Membership.mem A a
    smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
    A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
    smul_A_eq_A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HSMul …
    mul_mem_A_comm : ∀ {x a : G}, Membership.mem A a → Iff (Membership.mem A (HMul …
    ⊢ And (Eq (HSMul.hSMul a ↑(MulAction.stabilizer G A)) ↑A) (Eq (HSMul.hSMul (Mu …
  -/
  let H := stabilizer G A
  have inv_smul_A {a} (ha : a ∈ A) : a⁻¹ • (A : Set G) = H := by
    ext x
    rw [Set.mem_inv_smul_set_iff, smul_eq_mul]
    refine ⟨fun hx ↦ ?_, fun hx ↦ ?_⟩
    · simpa [← smul_A ha, mul_smul] using smul_A hx
    · norm_cast
      rwa [← mul_mem_A_comm ha, ← smul_eq_mul, ← mem_inv_smul_finset_iff, inv_mem hx]
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A : Finset G
    a : G
    hA : LE.le (HMul.hMul A A).card A.card
    ha : Membership.mem A a
    smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
    A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
    smul_A_eq_A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HSMul …
    mul_mem_A_comm : ∀ {x a : G}, Membership.mem A a → Iff (Membership.mem A (HMul …
    H : Subgroup G := MulAction.stabilizer G A
    inv_smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (Inv.inv a) ↑A) ↑H
    ⊢ And (Eq (HSMul.hSMul a ↑(MulAction.stabilizer G A)) ↑A) (Eq (HSMul.hSMul (Mu …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      A : Finset G
      a : G
      hA : LE.le (HMul.hMul A A).card A.card
      ha : Membership.mem A a
      smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
      A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
      smul_A_eq_A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HSMul …
      mul_mem_A_comm : ∀ {x a : G}, Membership.mem A a → Iff (Membership.mem A (HMul …
      H : Subgroup G := MulAction.stabilizer G A
      inv_smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (Inv.inv a) ↑A) ↑H
      ⊢ Eq (HSMul.hSMul a ↑(MulAction.stabilizer G A)) ↑A
    -/
  · rw [← inv_smul_A ha, smul_inv_smul]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      A : Finset G
      a : G
      hA : LE.le (HMul.hMul A A).card A.card
      ha : Membership.mem A a
      smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
      A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
      smul_A_eq_A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HSMul …
      mul_mem_A_comm : ∀ {x a : G}, Membership.mem A a → Iff (Membership.mem A (HMul …
      H : Subgroup G := MulAction.stabilizer G A
      inv_smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (Inv.inv a) ↑A) ↑H
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) ↑(MulAction.stabilizer G A)) ↑A
    -/
  · rw [← inv_smul_A ha, smul_comm]
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      A : Finset G
      a : G
      hA : LE.le (HMul.hMul A A).card A.card
      ha : Membership.mem A a
      smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
      A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
      smul_A_eq_A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HSMul …
      mul_mem_A_comm : ∀ {x a : G}, Membership.mem A a → Iff (Membership.mem A (HMul …
      H : Subgroup G := MulAction.stabilizer G A
      inv_smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (Inv.inv a) ↑A) ↑H
      ⊢ Eq (HSMul.hSMul (Inv.inv a) (HSMul.hSMul (MulOpposite.op a) ↑A)) ↑A
    -/
    norm_cast
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      A : Finset G
      a : G
      hA : LE.le (HMul.hMul A A).card A.card
      ha : Membership.mem A a
      smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HMul.hMul A A)
      A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (MulOpposite.op a) A) …
      smul_A_eq_A_smul : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul a A) (HSMul …
      mul_mem_A_comm : ∀ {x a : G}, Membership.mem A a → Iff (Membership.mem A (HMul …
      H : Subgroup G := MulAction.stabilizer G A
      inv_smul_A : ∀ {a : G}, Membership.mem A a → Eq (HSMul.hSMul (Inv.inv a) ↑A) ↑H
      ⊢ Eq (HSMul.hSMul (Inv.inv a) (HSMul.hSMul (MulOpposite.op a) A)) A
    -/
    rw [← smul_A_eq_A_smul ha, inv_smul_smul]
    /-
      🎉 no goals
    -/


/-- A non-empty set with no doubling is the left translate of its stabilizer. -/
@[to_additive "A non-empty set with no doubling is the left-translate of its stabilizer."]
lemma smul_stabilizer_of_no_doubling (hA : #(A * A) ≤ #A) (ha : a ∈ A) :
    a •> (stabilizer G A : Set G) = A := (smul_stabilizer_of_no_doubling_aux hA ha).1


/-- A non-empty set with no doubling is the right translate of its stabilizer. -/
@[to_additive "A non-empty set with no doubling is the right translate of its stabilizer."]
lemma op_smul_stabilizer_of_no_doubling (hA : #(A * A) ≤ #A) (ha : a ∈ A) :
    (stabilizer G A : Set G) <• a = A := (smul_stabilizer_of_no_doubling_aux hA ha).2


