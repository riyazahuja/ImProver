instance PosSMulMono.nnrat_of_rat [Preorder α] [MulAction ℚ α] [PosSMulMono ℚ α] :
    PosSMulMono ℚ≥0 α where elim _q hq _a₁ _a₂ ha := smul_le_smul_of_nonneg_left (α := ℚ) ha hq


instance PosSMulStrictMono.nnrat_of_rat [Preorder α] [MulAction ℚ α] [PosSMulStrictMono ℚ α] :
    PosSMulStrictMono ℚ≥0 α where elim _q hq _a₁ _a₂ ha := smul_lt_smul_of_pos_left (α := ℚ) ha hq


@[simp] lemma abs_nnqsmul [DistribMulAction ℚ≥0 α] [PosSMulMono ℚ≥0 α] (q : ℚ≥0) (a : α) :
    |q • a| = q • |a| := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : DistribMulAction NNRat α
    inst✝ : PosSMulMono NNRat α
    q : NNRat
    a : α
    ⊢ Eq (abs (HSMul.hSMul q a)) (HSMul.hSMul q (abs a))
  -/
  obtain ha | ha := le_total a 0 <;>
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrderedAddCommGroup α
      inst✝¹ : DistribMulAction NNRat α
      inst✝ : PosSMulMono NNRat α
      q : NNRat
      a : α
      ha : LE.le a 0
      ⊢ Eq (abs (HSMul.hSMul q a)) (HSMul.hSMul q (abs a))
    -/
    /-
      🎉 no goals
    -/
    simp [*, abs_of_nonneg, abs_of_nonpos, smul_nonneg, smul_nonpos_of_nonneg_of_nonpos]
    /-
      🎉 no goals
    -/


@[simp] lemma abs_qsmul [Module ℚ α] [PosSMulMono ℚ α] (q : ℚ) (a : α) :
    |q • a| = |q| • |a| := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : Module Rat α
    inst✝ : PosSMulMono Rat α
    q : Rat
    a : α
    ⊢ Eq (abs (HSMul.hSMul q a)) (HSMul.hSMul (abs q) (abs a))
  -/
  obtain ha | ha := le_total a 0 <;> obtain hq | hq := le_total q 0 <;>
    simp [*, abs_of_nonneg, abs_of_nonpos, smul_nonneg, smul_nonpos_of_nonneg_of_nonpos,
      smul_nonpos_of_nonpos_of_nonneg, smul_nonneg_of_nonpos_of_nonpos]


instance LinearOrderedSemifield.toPosSMulStrictMono_rat : PosSMulStrictMono ℚ≥0 α where
  elim q hq a b hab := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      q : NNRat
      hq : LT.lt 0 q
      a b : α
      hab : LT.lt a b
      ⊢ LT.lt (HSMul.hSMul q a) (HSMul.hSMul q b)
    -/
    rw [NNRat.smul_def, NNRat.smul_def]; exact mul_lt_mul_of_pos_left hab <| NNRat.cast_pos.2 hq
                                         /-
                                           🎉 no goals
                                         -/


instance LinearOrderedField.toPosSMulStrictMono_rat : PosSMulStrictMono ℚ α where
  elim q hq a b hab := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      q : Rat
      hq : LT.lt 0 q
      a b : α
      hab : LT.lt a b
      ⊢ LT.lt (HSMul.hSMul q a) (HSMul.hSMul q b)
    -/
    rw [Rat.smul_def, Rat.smul_def]; exact mul_lt_mul_of_pos_left hab <| Rat.cast_pos.2 hq
                                     /-
                                       🎉 no goals
                                     -/


