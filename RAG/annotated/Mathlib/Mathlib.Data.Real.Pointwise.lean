theorem Real.sInf_smul_of_nonneg (ha : 0 ≤ a) (s : Set ℝ) : sInf (a • s) = a • sInf s := by
  /-
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : MulActionWithZero α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le 0 a
    s : Set Real
    ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le 0 a
      ⊢ Eq (InfSet.sInf (HSMul.hSMul a EmptyCollection.emptyCollection)) (HSMul.hSMu …
    -/
  · rw [smul_set_empty, Real.sInf_empty, smul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : MulActionWithZero α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le 0 a
    s : Set Real
    hs : s.Nonempty
    ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (InfSet.sInf (HSMul.hSMul 0 s)) (HSMul.hSMul 0 (InfSet.sInf s))
    -/
  · rw [zero_smul_set hs, zero_smul]
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (InfSet.sInf 0) 0
    -/
    exact csInf_singleton 0
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : MulActionWithZero α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le 0 a
    s : Set Real
    hs : s.Nonempty
    ha' : LT.lt 0 a
    ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
  -/
  by_cases h : BddBelow s
    /-
      case pos
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le 0 a
      s : Set Real
      hs : s.Nonempty
      ha' : LT.lt 0 a
      h : BddBelow s
      ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
    -/
  · exact ((OrderIso.smulRight ha').map_csInf' hs h).symm
    /-
      🎉 no goals
    -/
  · rw [Real.sInf_of_not_bddBelow (mt (bddBelow_smul_iff_of_pos ha').1 h),
        Real.sInf_of_not_bddBelow h, smul_zero]


theorem Real.smul_iInf_of_nonneg (ha : 0 ≤ a) (f : ι → ℝ) : (a • ⨅ i, f i) = ⨅ i, a • f i :=
  (Real.sInf_smul_of_nonneg ha _).symm.trans <| congr_arg sInf <| (range_comp _ _).symm


theorem Real.sSup_smul_of_nonneg (ha : 0 ≤ a) (s : Set ℝ) : sSup (a • s) = a • sSup s := by
  /-
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : MulActionWithZero α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le 0 a
    s : Set Real
    ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le 0 a
      ⊢ Eq (SupSet.sSup (HSMul.hSMul a EmptyCollection.emptyCollection)) (HSMul.hSMu …
    -/
  · rw [smul_set_empty, Real.sSup_empty, smul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : MulActionWithZero α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le 0 a
    s : Set Real
    hs : s.Nonempty
    ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (SupSet.sSup (HSMul.hSMul 0 s)) (HSMul.hSMul 0 (SupSet.sSup s))
    -/
  · rw [zero_smul_set hs, zero_smul]
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (SupSet.sSup 0) 0
    -/
    exact csSup_singleton 0
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : MulActionWithZero α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le 0 a
    s : Set Real
    hs : s.Nonempty
    ha' : LT.lt 0 a
    ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
  -/
  by_cases h : BddAbove s
    /-
      case pos
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : MulActionWithZero α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le 0 a
      s : Set Real
      hs : s.Nonempty
      ha' : LT.lt 0 a
      h : BddAbove s
      ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
    -/
  · exact ((OrderIso.smulRight ha').map_csSup' hs h).symm
    /-
      🎉 no goals
    -/
  · rw [Real.sSup_of_not_bddAbove (mt (bddAbove_smul_iff_of_pos ha').1 h),
        Real.sSup_of_not_bddAbove h, smul_zero]


theorem Real.smul_iSup_of_nonneg (ha : 0 ≤ a) (f : ι → ℝ) : (a • ⨆ i, f i) = ⨆ i, a • f i :=
  (Real.sSup_smul_of_nonneg ha _).symm.trans <| congr_arg sSup <| (range_comp _ _).symm


theorem Real.sInf_smul_of_nonpos (ha : a ≤ 0) (s : Set ℝ) : sInf (a • s) = a • sSup s := by
  /-
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Module α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le a 0
    s : Set Real
    ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le a 0
      ⊢ Eq (InfSet.sInf (HSMul.hSMul a EmptyCollection.emptyCollection)) (HSMul.hSMu …
    -/
  · rw [smul_set_empty, Real.sInf_empty, Real.sSup_empty, smul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Module α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le a 0
    s : Set Real
    hs : s.Nonempty
    ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (InfSet.sInf (HSMul.hSMul 0 s)) (HSMul.hSMul 0 (SupSet.sSup s))
    -/
  · rw [zero_smul_set hs, zero_smul]
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (InfSet.sInf 0) 0
    -/
    exact csInf_singleton 0
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Module α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le a 0
    s : Set Real
    hs : s.Nonempty
    ha' : LT.lt a 0
    ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
  -/
  by_cases h : BddAbove s
    /-
      case pos
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le a 0
      s : Set Real
      hs : s.Nonempty
      ha' : LT.lt a 0
      h : BddAbove s
      ⊢ Eq (InfSet.sInf (HSMul.hSMul a s)) (HSMul.hSMul a (SupSet.sSup s))
    -/
  · exact ((OrderIso.smulRightDual ℝ ha').map_csSup' hs h).symm
    /-
      🎉 no goals
    -/
  · rw [Real.sInf_of_not_bddBelow (mt (bddBelow_smul_iff_of_neg ha').1 h),
        Real.sSup_of_not_bddAbove h, smul_zero]


theorem Real.smul_iSup_of_nonpos (ha : a ≤ 0) (f : ι → ℝ) : (a • ⨆ i, f i) = ⨅ i, a • f i :=
  (Real.sInf_smul_of_nonpos ha _).symm.trans <| congr_arg sInf <| (range_comp _ _).symm


theorem Real.sSup_smul_of_nonpos (ha : a ≤ 0) (s : Set ℝ) : sSup (a • s) = a • sInf s := by
  /-
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Module α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le a 0
    s : Set Real
    ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le a 0
      ⊢ Eq (SupSet.sSup (HSMul.hSMul a EmptyCollection.emptyCollection)) (HSMul.hSMu …
    -/
  · rw [smul_set_empty, Real.sSup_empty, Real.sInf_empty, smul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Module α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le a 0
    s : Set Real
    hs : s.Nonempty
    ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (SupSet.sSup (HSMul.hSMul 0 s)) (HSMul.hSMul 0 (InfSet.sInf s))
    -/
  · rw [zero_smul_set hs, zero_smul]
    /-
      case inr.inl
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      s : Set Real
      hs : s.Nonempty
      ha : LE.le 0 0
      ⊢ Eq (SupSet.sSup 0) 0
    -/
    exact csSup_singleton 0
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Module α Real
    inst✝ : OrderedSMul α Real
    a : α
    ha : LE.le a 0
    s : Set Real
    hs : s.Nonempty
    ha' : LT.lt a 0
    ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
  -/
  by_cases h : BddBelow s
    /-
      case pos
      α : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Module α Real
      inst✝ : OrderedSMul α Real
      a : α
      ha : LE.le a 0
      s : Set Real
      hs : s.Nonempty
      ha' : LT.lt a 0
      h : BddBelow s
      ⊢ Eq (SupSet.sSup (HSMul.hSMul a s)) (HSMul.hSMul a (InfSet.sInf s))
    -/
  · exact ((OrderIso.smulRightDual ℝ ha').map_csInf' hs h).symm
    /-
      🎉 no goals
    -/
  · rw [Real.sSup_of_not_bddAbove (mt (bddAbove_smul_iff_of_neg ha').1 h),
        Real.sInf_of_not_bddBelow h, smul_zero]


theorem Real.smul_iInf_of_nonpos (ha : a ≤ 0) (f : ι → ℝ) : (a • ⨅ i, f i) = ⨆ i, a • f i :=
  (Real.sSup_smul_of_nonpos ha _).symm.trans <| congr_arg sSup <| (range_comp _ _).symm


theorem Real.mul_iInf_of_nonneg (ha : 0 ≤ r) (f : ι → ℝ) : (r * ⨅ i, f i) = ⨅ i, r * f i :=
  Real.smul_iInf_of_nonneg ha f


theorem Real.mul_iSup_of_nonneg (ha : 0 ≤ r) (f : ι → ℝ) : (r * ⨆ i, f i) = ⨆ i, r * f i :=
  Real.smul_iSup_of_nonneg ha f


theorem Real.mul_iInf_of_nonpos (ha : r ≤ 0) (f : ι → ℝ) : (r * ⨅ i, f i) = ⨆ i, r * f i :=
  Real.smul_iInf_of_nonpos ha f


theorem Real.mul_iSup_of_nonpos (ha : r ≤ 0) (f : ι → ℝ) : (r * ⨆ i, f i) = ⨅ i, r * f i :=
  Real.smul_iSup_of_nonpos ha f


theorem Real.iInf_mul_of_nonneg (ha : 0 ≤ r) (f : ι → ℝ) : (⨅ i, f i) * r = ⨅ i, f i * r := by
  /-
    ι : Sort u_1
    r : Real
    ha : LE.le 0 r
    f : ι → Real
    ⊢ Eq (HMul.hMul (iInf fun i => f i) r) (iInf fun i => HMul.hMul (f i) r)
  -/
  simp only [Real.mul_iInf_of_nonneg ha, mul_comm]
  /-
    🎉 no goals
  -/


theorem Real.iSup_mul_of_nonneg (ha : 0 ≤ r) (f : ι → ℝ) : (⨆ i, f i) * r = ⨆ i, f i * r := by
  /-
    ι : Sort u_1
    r : Real
    ha : LE.le 0 r
    f : ι → Real
    ⊢ Eq (HMul.hMul (iSup fun i => f i) r) (iSup fun i => HMul.hMul (f i) r)
  -/
  simp only [Real.mul_iSup_of_nonneg ha, mul_comm]
  /-
    🎉 no goals
  -/


theorem Real.iInf_mul_of_nonpos (ha : r ≤ 0) (f : ι → ℝ) : (⨅ i, f i) * r = ⨆ i, f i * r := by
  /-
    ι : Sort u_1
    r : Real
    ha : LE.le r 0
    f : ι → Real
    ⊢ Eq (HMul.hMul (iInf fun i => f i) r) (iSup fun i => HMul.hMul (f i) r)
  -/
  simp only [Real.mul_iInf_of_nonpos ha, mul_comm]
  /-
    🎉 no goals
  -/


theorem Real.iSup_mul_of_nonpos (ha : r ≤ 0) (f : ι → ℝ) : (⨆ i, f i) * r = ⨅ i, f i * r := by
  /-
    ι : Sort u_1
    r : Real
    ha : LE.le r 0
    f : ι → Real
    ⊢ Eq (HMul.hMul (iSup fun i => f i) r) (iInf fun i => HMul.hMul (f i) r)
  -/
  simp only [Real.mul_iSup_of_nonpos ha, mul_comm]
  /-
    🎉 no goals
  -/


