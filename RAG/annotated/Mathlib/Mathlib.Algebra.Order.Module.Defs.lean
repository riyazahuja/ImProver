/-- Typeclass for monotonicity of scalar multiplication by nonnegative elements on the left,
namely `b₁ ≤ b₂ → a • b₁ ≤ a • b₂` if `0 ≤ a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class PosSMulMono : Prop where
  /-- Do not use this. Use `smul_le_smul_of_nonneg_left` instead. -/
  protected elim ⦃a : α⦄ (ha : 0 ≤ a) ⦃b₁ b₂ : β⦄ (hb : b₁ ≤ b₂) : a • b₁ ≤ a • b₂


/-- Typeclass for strict monotonicity of scalar multiplication by positive elements on the left,
namely `b₁ < b₂ → a • b₁ < a • b₂` if `0 < a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class PosSMulStrictMono : Prop where
  /-- Do not use this. Use `smul_lt_smul_of_pos_left` instead. -/
  protected elim ⦃a : α⦄ (ha : 0 < a) ⦃b₁ b₂ : β⦄ (hb : b₁ < b₂) : a • b₁ < a • b₂


/-- Typeclass for strict reverse monotonicity of scalar multiplication by nonnegative elements on
the left, namely `a • b₁ < a • b₂ → b₁ < b₂` if `0 ≤ a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class PosSMulReflectLT : Prop where
  /-- Do not use this. Use `lt_of_smul_lt_smul_left` instead. -/
  protected elim ⦃a : α⦄ (ha : 0 ≤ a) ⦃b₁ b₂ : β⦄ (hb : a • b₁ < a • b₂) : b₁ < b₂


/-- Typeclass for reverse monotonicity of scalar multiplication by positive elements on the left,
namely `a • b₁ ≤ a • b₂ → b₁ ≤ b₂` if `0 < a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class PosSMulReflectLE : Prop where
  /-- Do not use this. Use `le_of_smul_lt_smul_left` instead. -/
  protected elim ⦃a : α⦄ (ha : 0 < a) ⦃b₁ b₂ : β⦄ (hb : a • b₁ ≤ a • b₂) : b₁ ≤ b₂


/-- Typeclass for monotonicity of scalar multiplication by nonnegative elements on the left,
namely `a₁ ≤ a₂ → a₁ • b ≤ a₂ • b` if `0 ≤ b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class SMulPosMono : Prop where
  /-- Do not use this. Use `smul_le_smul_of_nonneg_right` instead. -/
  protected elim ⦃b : β⦄ (hb : 0 ≤ b) ⦃a₁ a₂ : α⦄ (ha : a₁ ≤ a₂) : a₁ • b ≤ a₂ • b


/-- Typeclass for strict monotonicity of scalar multiplication by positive elements on the left,
namely `a₁ < a₂ → a₁ • b < a₂ • b` if `0 < b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class SMulPosStrictMono : Prop where
  /-- Do not use this. Use `smul_lt_smul_of_pos_right` instead. -/
  protected elim ⦃b : β⦄ (hb : 0 < b) ⦃a₁ a₂ : α⦄ (ha : a₁ < a₂) : a₁ • b < a₂ • b


/-- Typeclass for strict reverse monotonicity of scalar multiplication by nonnegative elements on
the left, namely `a₁ • b < a₂ • b → a₁ < a₂` if `0 ≤ b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class SMulPosReflectLT : Prop where
  /-- Do not use this. Use `lt_of_smul_lt_smul_right` instead. -/
  protected elim ⦃b : β⦄ (hb : 0 ≤ b) ⦃a₁ a₂ : α⦄ (hb : a₁ • b < a₂ • b) : a₁ < a₂


/-- Typeclass for reverse monotonicity of scalar multiplication by positive elements on the left,
namely `a₁ • b ≤ a₂ • b → a₁ ≤ a₂` if `0 < b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSMul`. -/
class SMulPosReflectLE : Prop where
  /-- Do not use this. Use `le_of_smul_lt_smul_right` instead. -/
  protected elim ⦃b : β⦄ (hb : 0 < b) ⦃a₁ a₂ : α⦄ (hb : a₁ • b ≤ a₂ • b) : a₁ ≤ a₂


instance (priority := 100) PosMulMono.toPosSMulMono [PosMulMono α] : PosSMulMono α α where
  elim _a ha _b₁ _b₂ hb := mul_le_mul_of_nonneg_left hb ha

-- See note [lower instance priority]

instance (priority := 100) PosMulStrictMono.toPosSMulStrictMono [PosMulStrictMono α] :
    PosSMulStrictMono α α where
  elim _a ha _b₁ _b₂ hb := mul_lt_mul_of_pos_left hb ha

-- See note [lower instance priority]

instance (priority := 100) PosMulReflectLT.toPosSMulReflectLT [PosMulReflectLT α] :
    PosSMulReflectLT α α where
  elim _a ha _b₁ _b₂ h := lt_of_mul_lt_mul_left h ha

-- See note [lower instance priority]

instance (priority := 100) PosMulReflectLE.toPosSMulReflectLE [PosMulReflectLE α] :
    PosSMulReflectLE α α where
  elim _a ha _b₁ _b₂ h := le_of_mul_le_mul_left h ha

-- See note [lower instance priority]

instance (priority := 100) MulPosMono.toSMulPosMono [MulPosMono α] : SMulPosMono α α where
  elim _b hb _a₁ _a₂ ha := mul_le_mul_of_nonneg_right ha hb

-- See note [lower instance priority]

instance (priority := 100) MulPosStrictMono.toSMulPosStrictMono [MulPosStrictMono α] :
    SMulPosStrictMono α α where
  elim _b hb _a₁ _a₂ ha := mul_lt_mul_of_pos_right ha hb

-- See note [lower instance priority]

instance (priority := 100) MulPosReflectLT.toSMulPosReflectLT [MulPosReflectLT α] :
    SMulPosReflectLT α α where
  elim _b hb _a₁ _a₂ h := lt_of_mul_lt_mul_right h hb

-- See note [lower instance priority]

instance (priority := 100) MulPosReflectLE.toSMulPosReflectLE [MulPosReflectLE α] :
    SMulPosReflectLE α α where
  elim _b hb _a₁ _a₂ h := le_of_mul_le_mul_right h hb


lemma monotone_smul_left_of_nonneg [PosSMulMono α β] (ha : 0 ≤ a) : Monotone ((a • ·) : β → β) :=
  PosSMulMono.elim ha


lemma strictMono_smul_left_of_pos [PosSMulStrictMono α β] (ha : 0 < a) :
    StrictMono ((a • ·) : β → β) := PosSMulStrictMono.elim ha


@[gcongr] lemma smul_le_smul_of_nonneg_left [PosSMulMono α β] (hb : b₁ ≤ b₂) (ha : 0 ≤ a) :
    a • b₁ ≤ a • b₂ := monotone_smul_left_of_nonneg ha hb


@[gcongr] lemma smul_lt_smul_of_pos_left [PosSMulStrictMono α β] (hb : b₁ < b₂) (ha : 0 < a) :
    a • b₁ < a • b₂ := strictMono_smul_left_of_pos ha hb


lemma lt_of_smul_lt_smul_left [PosSMulReflectLT α β] (h : a • b₁ < a • b₂) (ha : 0 ≤ a) : b₁ < b₂ :=
  PosSMulReflectLT.elim ha h


lemma le_of_smul_le_smul_left [PosSMulReflectLE α β] (h : a • b₁ ≤ a • b₂) (ha : 0 < a) : b₁ ≤ b₂ :=
  PosSMulReflectLE.elim ha h


alias lt_of_smul_lt_smul_of_nonneg_left := lt_of_smul_lt_smul_left

alias le_of_smul_le_smul_of_pos_left := le_of_smul_le_smul_left


@[simp]
lemma smul_le_smul_iff_of_pos_left [PosSMulMono α β] [PosSMulReflectLE α β] (ha : 0 < a) :
    a • b₁ ≤ a • b₂ ↔ b₁ ≤ b₂ :=
  ⟨fun h ↦ le_of_smul_le_smul_left h ha, fun h ↦ smul_le_smul_of_nonneg_left h ha.le⟩


@[simp]
lemma smul_lt_smul_iff_of_pos_left [PosSMulStrictMono α β] [PosSMulReflectLT α β] (ha : 0 < a) :
    a • b₁ < a • b₂ ↔ b₁ < b₂ :=
  ⟨fun h ↦ lt_of_smul_lt_smul_left h ha.le, fun hb ↦ smul_lt_smul_of_pos_left hb ha⟩


lemma monotone_smul_right_of_nonneg [SMulPosMono α β] (hb : 0 ≤ b) : Monotone ((· • b) : α → β) :=
  SMulPosMono.elim hb


lemma strictMono_smul_right_of_pos [SMulPosStrictMono α β] (hb : 0 < b) :
    StrictMono ((· • b) : α → β) := SMulPosStrictMono.elim hb


@[gcongr] lemma smul_le_smul_of_nonneg_right [SMulPosMono α β] (ha : a₁ ≤ a₂) (hb : 0 ≤ b) :
    a₁ • b ≤ a₂ • b := monotone_smul_right_of_nonneg hb ha


@[gcongr] lemma smul_lt_smul_of_pos_right [SMulPosStrictMono α β] (ha : a₁ < a₂) (hb : 0 < b) :
    a₁ • b < a₂ • b := strictMono_smul_right_of_pos hb ha


lemma lt_of_smul_lt_smul_right [SMulPosReflectLT α β] (h : a₁ • b < a₂ • b) (hb : 0 ≤ b) :
    a₁ < a₂ := SMulPosReflectLT.elim hb h


lemma le_of_smul_le_smul_right [SMulPosReflectLE α β] (h : a₁ • b ≤ a₂ • b) (hb : 0 < b) :
    a₁ ≤ a₂ := SMulPosReflectLE.elim hb h


alias lt_of_smul_lt_smul_of_nonneg_right := lt_of_smul_lt_smul_right

alias le_of_smul_le_smul_of_pos_right := le_of_smul_le_smul_right


@[simp]
lemma smul_le_smul_iff_of_pos_right [SMulPosMono α β] [SMulPosReflectLE α β] (hb : 0 < b) :
    a₁ • b ≤ a₂ • b ↔ a₁ ≤ a₂ :=
  ⟨fun h ↦ le_of_smul_le_smul_right h hb, fun ha ↦ smul_le_smul_of_nonneg_right ha hb.le⟩


@[simp]
lemma smul_lt_smul_iff_of_pos_right [SMulPosStrictMono α β] [SMulPosReflectLT α β] (hb : 0 < b) :
    a₁ • b < a₂ • b ↔ a₁ < a₂ :=
  ⟨fun h ↦ lt_of_smul_lt_smul_right h hb.le, fun ha ↦ smul_lt_smul_of_pos_right ha hb⟩


lemma smul_lt_smul_of_le_of_lt [PosSMulStrictMono α β] [SMulPosMono α β] (ha : a₁ ≤ a₂)
    (hb : b₁ < b₂) (h₁ : 0 < a₁) (h₂ : 0 ≤ b₂) : a₁ • b₁ < a₂ • b₂ :=
  (smul_lt_smul_of_pos_left hb h₁).trans_le (smul_le_smul_of_nonneg_right ha h₂)


lemma smul_lt_smul_of_le_of_lt' [PosSMulStrictMono α β] [SMulPosMono α β] (ha : a₁ ≤ a₂)
    (hb : b₁ < b₂) (h₂ : 0 < a₂) (h₁ : 0 ≤ b₁) : a₁ • b₁ < a₂ • b₂ :=
  (smul_le_smul_of_nonneg_right ha h₁).trans_lt (smul_lt_smul_of_pos_left hb h₂)


lemma smul_lt_smul_of_lt_of_le [PosSMulMono α β] [SMulPosStrictMono α β] (ha : a₁ < a₂)
    (hb : b₁ ≤ b₂) (h₁ : 0 ≤ a₁) (h₂ : 0 < b₂) : a₁ • b₁ < a₂ • b₂ :=
  (smul_le_smul_of_nonneg_left hb h₁).trans_lt (smul_lt_smul_of_pos_right ha h₂)


lemma smul_lt_smul_of_lt_of_le' [PosSMulMono α β] [SMulPosStrictMono α β] (ha : a₁ < a₂)
    (hb : b₁ ≤ b₂) (h₂ : 0 ≤ a₂) (h₁ : 0 < b₁) : a₁ • b₁ < a₂ • b₂ :=
  (smul_lt_smul_of_pos_right ha h₁).trans_le (smul_le_smul_of_nonneg_left hb h₂)


lemma smul_lt_smul [PosSMulStrictMono α β] [SMulPosStrictMono α β] (ha : a₁ < a₂) (hb : b₁ < b₂)
    (h₁ : 0 < a₁) (h₂ : 0 < b₂) : a₁ • b₁ < a₂ • b₂ :=
  (smul_lt_smul_of_pos_left hb h₁).trans (smul_lt_smul_of_pos_right ha h₂)


lemma smul_lt_smul' [PosSMulStrictMono α β] [SMulPosStrictMono α β] (ha : a₁ < a₂) (hb : b₁ < b₂)
    (h₂ : 0 < a₂) (h₁ : 0 < b₁) : a₁ • b₁ < a₂ • b₂ :=
  (smul_lt_smul_of_pos_right ha h₁).trans (smul_lt_smul_of_pos_left hb h₂)


lemma smul_le_smul [PosSMulMono α β] [SMulPosMono α β] (ha : a₁ ≤ a₂) (hb : b₁ ≤ b₂)
    (h₁ : 0 ≤ a₁) (h₂ : 0 ≤ b₂) : a₁ • b₁ ≤ a₂ • b₂ :=
  (smul_le_smul_of_nonneg_left hb h₁).trans (smul_le_smul_of_nonneg_right ha h₂)


lemma smul_le_smul' [PosSMulMono α β] [SMulPosMono α β] (ha : a₁ ≤ a₂) (hb : b₁ ≤ b₂) (h₂ : 0 ≤ a₂)
    (h₁ : 0 ≤ b₁) : a₁ • b₁ ≤ a₂ • b₂ :=
  (smul_le_smul_of_nonneg_right ha h₁).trans (smul_le_smul_of_nonneg_left hb h₂)


instance (priority := 100) PosSMulStrictMono.toPosSMulReflectLE [PosSMulStrictMono α β] :
    PosSMulReflectLE α β where
  elim _a ha _b₁ _b₂ := (strictMono_smul_left_of_pos ha).le_iff_le.1


lemma PosSMulReflectLE.toPosSMulStrictMono [PosSMulReflectLE α β] : PosSMulStrictMono α β where
  elim _a ha _b₁ _b₂ hb := not_le.1 fun h ↦ hb.not_le <| le_of_smul_le_smul_left h ha


lemma posSMulStrictMono_iff_PosSMulReflectLE : PosSMulStrictMono α β ↔ PosSMulReflectLE α β :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ PosSMulReflectLE.toPosSMulStrictMono⟩


instance PosSMulMono.toPosSMulReflectLT [PosSMulMono α β] : PosSMulReflectLT α β where
  elim _a ha _b₁ _b₂ := (monotone_smul_left_of_nonneg ha).reflect_lt


lemma PosSMulReflectLT.toPosSMulMono [PosSMulReflectLT α β] : PosSMulMono α β where
  elim _a ha _b₁ _b₂ hb := not_lt.1 fun h ↦ hb.not_lt <| lt_of_smul_lt_smul_left h ha


lemma posSMulMono_iff_posSMulReflectLT : PosSMulMono α β ↔ PosSMulReflectLT α β :=
  ⟨fun _ ↦ PosSMulMono.toPosSMulReflectLT, fun _ ↦ PosSMulReflectLT.toPosSMulMono⟩


lemma smul_max_of_nonneg [PosSMulMono α β] (ha : 0 ≤ a) (b₁ b₂ : β) :
    a • max b₁ b₂ = max (a • b₁) (a • b₂) := (monotone_smul_left_of_nonneg ha).map_max


lemma smul_min_of_nonneg [PosSMulMono α β] (ha : 0 ≤ a) (b₁ b₂ : β) :
    a • min b₁ b₂ = min (a • b₁) (a • b₂) := (monotone_smul_left_of_nonneg ha).map_min


lemma SMulPosReflectLE.toSMulPosStrictMono [SMulPosReflectLE α β] : SMulPosStrictMono α β where
  elim _b hb _a₁ _a₂ ha := not_le.1 fun h ↦ ha.not_le <| le_of_smul_le_smul_of_pos_right h hb


lemma SMulPosReflectLT.toSMulPosMono [SMulPosReflectLT α β] : SMulPosMono α β where
  elim _b hb _a₁ _a₂ ha := not_lt.1 fun h ↦ ha.not_lt <| lt_of_smul_lt_smul_right h hb


instance (priority := 100) SMulPosStrictMono.toSMulPosReflectLE [SMulPosStrictMono α β] :
    SMulPosReflectLE α β where
  elim _b hb _a₁ _a₂ h := not_lt.1 fun ha ↦ h.not_lt <| smul_lt_smul_of_pos_right ha hb


lemma SMulPosMono.toSMulPosReflectLT [SMulPosMono α β] : SMulPosReflectLT α β where
  elim _b hb _a₁ _a₂ h := not_le.1 fun ha ↦ h.not_le <| smul_le_smul_of_nonneg_right ha hb


lemma smulPosStrictMono_iff_SMulPosReflectLE : SMulPosStrictMono α β ↔ SMulPosReflectLE α β :=
  ⟨fun _ ↦ SMulPosStrictMono.toSMulPosReflectLE, fun _ ↦ SMulPosReflectLE.toSMulPosStrictMono⟩


lemma smulPosMono_iff_smulPosReflectLT : SMulPosMono α β ↔ SMulPosReflectLT α β :=
  ⟨fun _ ↦ SMulPosMono.toSMulPosReflectLT, fun _ ↦ SMulPosReflectLT.toSMulPosMono⟩


lemma smul_pos [PosSMulStrictMono α β] (ha : 0 < a) (hb : 0 < b) : 0 < a • b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulZeroClass α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : PosSMulStrictMono α β
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ LT.lt 0 (HSMul.hSMul a b)
  -/
  simpa only [smul_zero] using smul_lt_smul_of_pos_left hb ha
  /-
    🎉 no goals
  -/


lemma smul_neg_of_pos_of_neg [PosSMulStrictMono α β] (ha : 0 < a) (hb : b < 0) : a • b < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulZeroClass α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : PosSMulStrictMono α β
    ha : LT.lt 0 a
    hb : LT.lt b 0
    ⊢ LT.lt (HSMul.hSMul a b) 0
  -/
  simpa only [smul_zero] using smul_lt_smul_of_pos_left hb ha
  /-
    🎉 no goals
  -/


@[simp]
lemma smul_pos_iff_of_pos_left [PosSMulStrictMono α β] [PosSMulReflectLT α β] (ha : 0 < a) :
    0 < a • b ↔ 0 < b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulZeroClass α β
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : PosSMulReflectLT α β
    ha : LT.lt 0 a
    ⊢ Iff (LT.lt 0 (HSMul.hSMul a b)) (LT.lt 0 b)
  -/
  simpa only [smul_zero] using smul_lt_smul_iff_of_pos_left ha (b₁ := 0) (b₂ := b)
  /-
    🎉 no goals
  -/


lemma smul_neg_iff_of_pos_left [PosSMulStrictMono α β] [PosSMulReflectLT α β] (ha : 0 < a) :
    a • b < 0 ↔ b < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulZeroClass α β
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : PosSMulReflectLT α β
    ha : LT.lt 0 a
    ⊢ Iff (LT.lt (HSMul.hSMul a b) 0) (LT.lt b 0)
  -/
  simpa only [smul_zero]  using smul_lt_smul_iff_of_pos_left ha (b₂ := (0 : β))
  /-
    🎉 no goals
  -/


lemma smul_nonneg [PosSMulMono α β] (ha : 0 ≤ a) (hb : 0 ≤ b₁) : 0 ≤ a • b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulZeroClass α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : PosSMulMono α β
    ha : LE.le 0 a
    hb : LE.le 0 b₁
    ⊢ LE.le 0 (HSMul.hSMul a b₁)
  -/
  simpa only [smul_zero] using smul_le_smul_of_nonneg_left hb ha
  /-
    🎉 no goals
  -/


lemma smul_nonpos_of_nonneg_of_nonpos [PosSMulMono α β] (ha : 0 ≤ a) (hb : b ≤ 0) : a • b ≤ 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulZeroClass α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : PosSMulMono α β
    ha : LE.le 0 a
    hb : LE.le b 0
    ⊢ LE.le (HSMul.hSMul a b) 0
  -/
  simpa only [smul_zero] using smul_le_smul_of_nonneg_left hb ha
  /-
    🎉 no goals
  -/


lemma pos_of_smul_pos_left [PosSMulReflectLT α β] (h : 0 < a • b) (ha : 0 ≤ a) : 0 < b :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                a : α
                                b : β
                                inst✝⁵ : Zero α
                                inst✝⁴ : Zero β
                                inst✝³ : SMulZeroClass α β
                                inst✝² : Preorder α
                                inst✝¹ : Preorder β
                                inst✝ : PosSMulReflectLT α β
                                h : LT.lt 0 (HSMul.hSMul a b)
                                ha : LE.le 0 a
                                ⊢ LT.lt (HSMul.hSMul a 0) (HSMul.hSMul a b)
                              -/
  lt_of_smul_lt_smul_left (by rwa [smul_zero]) ha
                              /-
                                🎉 no goals
                              -/


lemma neg_of_smul_neg_left [PosSMulReflectLT α β] (h : a • b < 0) (ha : 0 ≤ a) : b < 0 :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                a : α
                                b : β
                                inst✝⁵ : Zero α
                                inst✝⁴ : Zero β
                                inst✝³ : SMulZeroClass α β
                                inst✝² : Preorder α
                                inst✝¹ : Preorder β
                                inst✝ : PosSMulReflectLT α β
                                h : LT.lt (HSMul.hSMul a b) 0
                                ha : LE.le 0 a
                                ⊢ LT.lt (HSMul.hSMul a b) (HSMul.hSMul a 0)
                              -/
  lt_of_smul_lt_smul_left (by rwa [smul_zero]) ha
                              /-
                                🎉 no goals
                              -/


lemma smul_pos' [SMulPosStrictMono α β] (ha : 0 < a) (hb : 0 < b) : 0 < a • b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulWithZero α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosStrictMono α β
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ LT.lt 0 (HSMul.hSMul a b)
  -/
  simpa only [zero_smul] using smul_lt_smul_of_pos_right ha hb
  /-
    🎉 no goals
  -/


lemma smul_neg_of_neg_of_pos [SMulPosStrictMono α β] (ha : a < 0) (hb : 0 < b) : a • b < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulWithZero α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosStrictMono α β
    ha : LT.lt a 0
    hb : LT.lt 0 b
    ⊢ LT.lt (HSMul.hSMul a b) 0
  -/
  simpa only [zero_smul] using smul_lt_smul_of_pos_right ha hb
  /-
    🎉 no goals
  -/


@[simp]
lemma smul_pos_iff_of_pos_right [SMulPosStrictMono α β] [SMulPosReflectLT α β] (hb : 0 < b) :
    0 < a • b ↔ 0 < a := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : SMulPosStrictMono α β
    inst✝ : SMulPosReflectLT α β
    hb : LT.lt 0 b
    ⊢ Iff (LT.lt 0 (HSMul.hSMul a b)) (LT.lt 0 a)
  -/
  simpa only [zero_smul] using smul_lt_smul_iff_of_pos_right hb (a₁ := 0) (a₂ := a)
  /-
    🎉 no goals
  -/


lemma smul_nonneg' [SMulPosMono α β] (ha : 0 ≤ a) (hb : 0 ≤ b₁) : 0 ≤ a • b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulWithZero α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosMono α β
    ha : LE.le 0 a
    hb : LE.le 0 b₁
    ⊢ LE.le 0 (HSMul.hSMul a b₁)
  -/
  simpa only [zero_smul] using smul_le_smul_of_nonneg_right ha hb
  /-
    🎉 no goals
  -/


lemma smul_nonpos_of_nonpos_of_nonneg [SMulPosMono α β] (ha : a ≤ 0) (hb : 0 ≤ b) : a • b ≤ 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Zero α
    inst✝⁴ : Zero β
    inst✝³ : SMulWithZero α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosMono α β
    ha : LE.le a 0
    hb : LE.le 0 b
    ⊢ LE.le (HSMul.hSMul a b) 0
  -/
  simpa only [zero_smul] using smul_le_smul_of_nonneg_right ha hb
  /-
    🎉 no goals
  -/


lemma pos_of_smul_pos_right [SMulPosReflectLT α β] (h : 0 < a • b) (hb : 0 ≤ b) : 0 < a :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 a : α
                                 b : β
                                 inst✝⁵ : Zero α
                                 inst✝⁴ : Zero β
                                 inst✝³ : SMulWithZero α β
                                 inst✝² : Preorder α
                                 inst✝¹ : Preorder β
                                 inst✝ : SMulPosReflectLT α β
                                 h : LT.lt 0 (HSMul.hSMul a b)
                                 hb : LE.le 0 b
                                 ⊢ LT.lt (HSMul.hSMul 0 b) (HSMul.hSMul a b)
                               -/
  lt_of_smul_lt_smul_right (by rwa [zero_smul]) hb
                               /-
                                 🎉 no goals
                               -/


lemma neg_of_smul_neg_right [SMulPosReflectLT α β] (h : a • b < 0) (hb : 0 ≤ b) : a < 0 :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 a : α
                                 b : β
                                 inst✝⁵ : Zero α
                                 inst✝⁴ : Zero β
                                 inst✝³ : SMulWithZero α β
                                 inst✝² : Preorder α
                                 inst✝¹ : Preorder β
                                 inst✝ : SMulPosReflectLT α β
                                 h : LT.lt (HSMul.hSMul a b) 0
                                 hb : LE.le 0 b
                                 ⊢ LT.lt (HSMul.hSMul a b) (HSMul.hSMul 0 b)
                               -/
  lt_of_smul_lt_smul_right (by rwa [zero_smul]) hb
                               /-
                                 🎉 no goals
                               -/


lemma pos_iff_pos_of_smul_pos [PosSMulReflectLT α β] [SMulPosReflectLT α β] (hab : 0 < a • b) :
    0 < a ↔ 0 < b :=
  ⟨pos_of_smul_pos_left hab ∘ le_of_lt, pos_of_smul_pos_right hab ∘ le_of_lt⟩


/-- A constructor for `PosSMulMono` requiring you to prove `b₁ ≤ b₂ → a • b₁ ≤ a • b₂` only when
`0 < a`-/
lemma PosSMulMono.of_pos (h₀ : ∀ a : α, 0 < a → ∀ b₁ b₂ : β, b₁ ≤ b₂ → a • b₁ ≤ a • b₂) :
    PosSMulMono α β where
  elim a ha b₁ b₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Zero α
      inst✝³ : Zero β
      inst✝² : SMulWithZero α β
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      h₀ : ∀ (a : α), LT.lt 0 a → ∀ (b₁ b₂ : β), LE.le b₁ b₂ → LE.le (HSMul.hSMul a  …
      a : α
      ha : LE.le 0 a
      b₁ b₂ : β
      h : LE.le b₁ b₂
      ⊢ LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
    -/
    obtain ha | ha := ha.eq_or_lt
      /-
        case inl
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        h₀ : ∀ (a : α), LT.lt 0 a → ∀ (b₁ b₂ : β), LE.le b₁ b₂ → LE.le (HSMul.hSMul a  …
        a : α
        ha✝ : LE.le 0 a
        b₁ b₂ : β
        h : LE.le b₁ b₂
        ha : Eq 0 a
        ⊢ LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
      -/
    · simp [← ha]
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        h₀ : ∀ (a : α), LT.lt 0 a → ∀ (b₁ b₂ : β), LE.le b₁ b₂ → LE.le (HSMul.hSMul a  …
        a : α
        ha✝ : LE.le 0 a
        b₁ b₂ : β
        h : LE.le b₁ b₂
        ha : LT.lt 0 a
        ⊢ LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
      -/
    · exact h₀ _ ha _ _ h
      /-
        🎉 no goals
      -/


/-- A constructor for `PosSMulReflectLT` requiring you to prove `a • b₁ < a • b₂ → b₁ < b₂` only
when `0 < a`-/
lemma PosSMulReflectLT.of_pos (h₀ : ∀ a : α, 0 < a → ∀ b₁ b₂ : β, a • b₁ < a • b₂ → b₁ < b₂) :
    PosSMulReflectLT α β where
  elim a ha b₁ b₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Zero α
      inst✝³ : Zero β
      inst✝² : SMulWithZero α β
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      h₀ : ∀ (a : α), LT.lt 0 a → ∀ (b₁ b₂ : β), LT.lt (HSMul.hSMul a b₁) (HSMul.hSM …
      a : α
      ha : LE.le 0 a
      b₁ b₂ : β
      h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
      ⊢ LT.lt b₁ b₂
    -/
    obtain ha | ha := ha.eq_or_lt
      /-
        case inl
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        h₀ : ∀ (a : α), LT.lt 0 a → ∀ (b₁ b₂ : β), LT.lt (HSMul.hSMul a b₁) (HSMul.hSM …
        a : α
        ha✝ : LE.le 0 a
        b₁ b₂ : β
        h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
        ha : Eq 0 a
        ⊢ LT.lt b₁ b₂
      -/
    · simp [← ha] at h
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        h₀ : ∀ (a : α), LT.lt 0 a → ∀ (b₁ b₂ : β), LT.lt (HSMul.hSMul a b₁) (HSMul.hSM …
        a : α
        ha✝ : LE.le 0 a
        b₁ b₂ : β
        h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
        ha : LT.lt 0 a
        ⊢ LT.lt b₁ b₂
      -/
    · exact h₀ _ ha _ _ h
      /-
        🎉 no goals
      -/


/-- A constructor for `SMulPosMono` requiring you to prove `a₁ ≤ a₂ → a₁ • b ≤ a₂ • b` only when
`0 < b`-/
lemma SMulPosMono.of_pos (h₀ : ∀ b : β, 0 < b → ∀ a₁ a₂ : α, a₁ ≤ a₂ → a₁ • b ≤ a₂ • b) :
    SMulPosMono α β where
  elim b hb a₁ a₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Zero α
      inst✝³ : Zero β
      inst✝² : SMulWithZero α β
      inst✝¹ : Preorder α
      inst✝ : PartialOrder β
      h₀ : ∀ (b : β), LT.lt 0 b → ∀ (a₁ a₂ : α), LE.le a₁ a₂ → LE.le (HSMul.hSMul a₁ …
      b : β
      hb : LE.le 0 b
      a₁ a₂ : α
      h : LE.le a₁ a₂
      ⊢ LE.le (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
    -/
    obtain hb | hb := hb.eq_or_lt
      /-
        case inl
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : Preorder α
        inst✝ : PartialOrder β
        h₀ : ∀ (b : β), LT.lt 0 b → ∀ (a₁ a₂ : α), LE.le a₁ a₂ → LE.le (HSMul.hSMul a₁ …
        b : β
        hb✝ : LE.le 0 b
        a₁ a₂ : α
        h : LE.le a₁ a₂
        hb : Eq 0 b
        ⊢ LE.le (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
      -/
    · simp [← hb]
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : Preorder α
        inst✝ : PartialOrder β
        h₀ : ∀ (b : β), LT.lt 0 b → ∀ (a₁ a₂ : α), LE.le a₁ a₂ → LE.le (HSMul.hSMul a₁ …
        b : β
        hb✝ : LE.le 0 b
        a₁ a₂ : α
        h : LE.le a₁ a₂
        hb : LT.lt 0 b
        ⊢ LE.le (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
      -/
    · exact h₀ _ hb _ _ h
      /-
        🎉 no goals
      -/


/-- A constructor for `SMulPosReflectLT` requiring you to prove `a₁ • b < a₂ • b → a₁ < a₂` only
when `0 < b`-/
lemma SMulPosReflectLT.of_pos (h₀ : ∀ b : β, 0 < b → ∀ a₁ a₂ : α, a₁ • b < a₂ • b → a₁ < a₂) :
    SMulPosReflectLT α β where
  elim b hb a₁ a₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Zero α
      inst✝³ : Zero β
      inst✝² : SMulWithZero α β
      inst✝¹ : Preorder α
      inst✝ : PartialOrder β
      h₀ : ∀ (b : β), LT.lt 0 b → ∀ (a₁ a₂ : α), LT.lt (HSMul.hSMul a₁ b) (HSMul.hSM …
      b : β
      hb : LE.le 0 b
      a₁ a₂ : α
      h : LT.lt (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
      ⊢ LT.lt a₁ a₂
    -/
    obtain hb | hb := hb.eq_or_lt
      /-
        case inl
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : Preorder α
        inst✝ : PartialOrder β
        h₀ : ∀ (b : β), LT.lt 0 b → ∀ (a₁ a₂ : α), LT.lt (HSMul.hSMul a₁ b) (HSMul.hSM …
        b : β
        hb✝ : LE.le 0 b
        a₁ a₂ : α
        h : LT.lt (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
        hb : Eq 0 b
        ⊢ LT.lt a₁ a₂
      -/
    · simp [← hb] at h
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        β : Type u_2
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : SMulWithZero α β
        inst✝¹ : Preorder α
        inst✝ : PartialOrder β
        h₀ : ∀ (b : β), LT.lt 0 b → ∀ (a₁ a₂ : α), LT.lt (HSMul.hSMul a₁ b) (HSMul.hSM …
        b : β
        hb✝ : LE.le 0 b
        a₁ a₂ : α
        h : LT.lt (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
        hb : LT.lt 0 b
        ⊢ LT.lt a₁ a₂
      -/
    · exact h₀ _ hb _ _ h
      /-
        🎉 no goals
      -/


instance (priority := 100) PosSMulStrictMono.toPosSMulMono [PosSMulStrictMono α β] :
    PosSMulMono α β :=
  PosSMulMono.of_pos fun _a ha ↦ (strictMono_smul_left_of_pos ha).monotone

-- See note [lower instance priority]

instance (priority := 100) SMulPosStrictMono.toSMulPosMono [SMulPosStrictMono α β] :
    SMulPosMono α β :=
  SMulPosMono.of_pos fun _b hb ↦ (strictMono_smul_right_of_pos hb).monotone

-- See note [lower instance priority]

instance (priority := 100) PosSMulReflectLE.toPosSMulReflectLT [PosSMulReflectLE α β] :
    PosSMulReflectLT α β :=
  PosSMulReflectLT.of_pos fun a ha b₁ b₂ h ↦
                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              a✝ a₁ a₂ : α
                                                              b b₁✝ b₂✝ : β
                                                              inst✝⁵ : Zero α
                                                              inst✝⁴ : Zero β
                                                              inst✝³ : SMulWithZero α β
                                                              inst✝² : PartialOrder α
                                                              inst✝¹ : PartialOrder β
                                                              inst✝ : PosSMulReflectLE α β
                                                              a : α
                                                              ha : LT.lt 0 a
                                                              b₁ b₂ : β
                                                              h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
                                                              ⊢ Ne b₁ b₂
                                                            -/
    (le_of_smul_le_smul_of_pos_left h.le ha).lt_of_ne <| by rintro rfl; simp at h
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

-- See note [lower instance priority]

instance (priority := 100) SMulPosReflectLE.toSMulPosReflectLT [SMulPosReflectLE α β] :
    SMulPosReflectLT α β :=
  SMulPosReflectLT.of_pos fun b hb a₁ a₂ h ↦
                                                             /-
                                                               α : Type u_1
                                                               β : Type u_2
                                                               a a₁✝ a₂✝ : α
                                                               b✝ b₁ b₂ : β
                                                               inst✝⁵ : Zero α
                                                               inst✝⁴ : Zero β
                                                               inst✝³ : SMulWithZero α β
                                                               inst✝² : PartialOrder α
                                                               inst✝¹ : PartialOrder β
                                                               inst✝ : SMulPosReflectLE α β
                                                               b : β
                                                               hb : LT.lt 0 b
                                                               a₁ a₂ : α
                                                               h : LT.lt (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
                                                               ⊢ Ne a₁ a₂
                                                             -/
    (le_of_smul_le_smul_of_pos_right h.le hb).lt_of_ne <| by rintro rfl; simp at h
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma smul_eq_smul_iff_eq_and_eq_of_pos [PosSMulStrictMono α β] [SMulPosStrictMono α β]
    (ha : a₁ ≤ a₂) (hb : b₁ ≤ b₂) (h₁ : 0 < a₁) (h₂ : 0 < b₂) :
    a₁ • b₁ = a₂ • b₂ ↔ a₁ = a₂ ∧ b₁ = b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    a₁ a₂ : α
    b₁ b₂ : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : SMulPosStrictMono α β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h₁ : LT.lt 0 a₁
    h₂ : LT.lt 0 b₂
    ⊢ Iff (Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)) (And (Eq a₁ a₂) (Eq b₁ b₂))
  -/
  refine ⟨fun h ↦ ?_, by rintro ⟨rfl, rfl⟩; rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    a₁ a₂ : α
    b₁ b₂ : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : SMulPosStrictMono α β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h₁ : LT.lt 0 a₁
    h₂ : LT.lt 0 b₂
    h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    ⊢ And (Eq a₁ a₂) (Eq b₁ b₂)
  -/
  simp only [eq_iff_le_not_lt, ha, hb, true_and]
  /-
    α : Type u_1
    β : Type u_2
    a₁ a₂ : α
    b₁ b₂ : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : SMulPosStrictMono α β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h₁ : LT.lt 0 a₁
    h₂ : LT.lt 0 b₂
    h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    ⊢ And (Not (LT.lt a₁ a₂)) (Not (LT.lt b₁ b₂))
  -/
  refine ⟨fun ha ↦ h.not_lt ?_, fun hb ↦ h.not_lt ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      a₁ a₂ : α
      b₁ b₂ : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : PosSMulStrictMono α β
      inst✝ : SMulPosStrictMono α β
      ha✝ : LE.le a₁ a₂
      hb : LE.le b₁ b₂
      h₁ : LT.lt 0 a₁
      h₂ : LT.lt 0 b₂
      h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
      ha : LT.lt a₁ a₂
      ⊢ LT.lt (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    -/
  · exact (smul_le_smul_of_nonneg_left hb h₁.le).trans_lt (smul_lt_smul_of_pos_right ha h₂)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      a₁ a₂ : α
      b₁ b₂ : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : PosSMulStrictMono α β
      inst✝ : SMulPosStrictMono α β
      ha : LE.le a₁ a₂
      hb✝ : LE.le b₁ b₂
      h₁ : LT.lt 0 a₁
      h₂ : LT.lt 0 b₂
      h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
      hb : LT.lt b₁ b₂
      ⊢ LT.lt (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    -/
  · exact (smul_lt_smul_of_pos_left hb h₁).trans_le (smul_le_smul_of_nonneg_right ha h₂.le)
    /-
      🎉 no goals
    -/


lemma smul_eq_smul_iff_eq_and_eq_of_pos' [PosSMulStrictMono α β] [SMulPosStrictMono α β]
    (ha : a₁ ≤ a₂) (hb : b₁ ≤ b₂) (h₂ : 0 < a₂) (h₁ : 0 < b₁) :
    a₁ • b₁ = a₂ • b₂ ↔ a₁ = a₂ ∧ b₁ = b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    a₁ a₂ : α
    b₁ b₂ : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : SMulPosStrictMono α β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h₂ : LT.lt 0 a₂
    h₁ : LT.lt 0 b₁
    ⊢ Iff (Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)) (And (Eq a₁ a₂) (Eq b₁ b₂))
  -/
  refine ⟨fun h ↦ ?_, by rintro ⟨rfl, rfl⟩; rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    a₁ a₂ : α
    b₁ b₂ : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : SMulPosStrictMono α β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h₂ : LT.lt 0 a₂
    h₁ : LT.lt 0 b₁
    h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    ⊢ And (Eq a₁ a₂) (Eq b₁ b₂)
  -/
  simp only [eq_iff_le_not_lt, ha, hb, true_and]
  /-
    α : Type u_1
    β : Type u_2
    a₁ a₂ : α
    b₁ b₂ : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : SMulPosStrictMono α β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h₂ : LT.lt 0 a₂
    h₁ : LT.lt 0 b₁
    h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    ⊢ And (Not (LT.lt a₁ a₂)) (Not (LT.lt b₁ b₂))
  -/
  refine ⟨fun ha ↦ h.not_lt ?_, fun hb ↦ h.not_lt ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      a₁ a₂ : α
      b₁ b₂ : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : PosSMulStrictMono α β
      inst✝ : SMulPosStrictMono α β
      ha✝ : LE.le a₁ a₂
      hb : LE.le b₁ b₂
      h₂ : LT.lt 0 a₂
      h₁ : LT.lt 0 b₁
      h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
      ha : LT.lt a₁ a₂
      ⊢ LT.lt (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    -/
  · exact (smul_lt_smul_of_pos_right ha h₁).trans_le (smul_le_smul_of_nonneg_left hb h₂.le)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      a₁ a₂ : α
      b₁ b₂ : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : PosSMulStrictMono α β
      inst✝ : SMulPosStrictMono α β
      ha : LE.le a₁ a₂
      hb✝ : LE.le b₁ b₂
      h₂ : LT.lt 0 a₂
      h₁ : LT.lt 0 b₁
      h : Eq (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
      hb : LT.lt b₁ b₂
      ⊢ LT.lt (HSMul.hSMul a₁ b₁) (HSMul.hSMul a₂ b₂)
    -/
  · exact (smul_le_smul_of_nonneg_right ha h₁.le).trans_lt (smul_lt_smul_of_pos_left hb h₂)
    /-
      🎉 no goals
    -/


lemma pos_and_pos_or_neg_and_neg_of_smul_pos [PosSMulMono α β] [SMulPosMono α β] (hab : 0 < a • b) :
    0 < a ∧ 0 < b ∨ a < 0 ∧ b < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁶ : Zero α
    inst✝⁵ : Zero β
    inst✝⁴ : SMulWithZero α β
    inst✝³ : LinearOrder α
    inst✝² : LinearOrder β
    inst✝¹ : PosSMulMono α β
    inst✝ : SMulPosMono α β
    hab : LT.lt 0 (HSMul.hSMul a b)
    ⊢ Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))
  -/
  obtain ha | rfl | ha := lt_trichotomy a 0
    /-
      case inl
      α : Type u_1
      β : Type u_2
      a : α
      b : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : LinearOrder α
      inst✝² : LinearOrder β
      inst✝¹ : PosSMulMono α β
      inst✝ : SMulPosMono α β
      hab : LT.lt 0 (HSMul.hSMul a b)
      ha : LT.lt a 0
      ⊢ Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))
    -/
  · refine Or.inr ⟨ha, lt_imp_lt_of_le_imp_le (fun hb ↦ ?_) hab⟩
    /-
      case inl
      α : Type u_1
      β : Type u_2
      a : α
      b : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : LinearOrder α
      inst✝² : LinearOrder β
      inst✝¹ : PosSMulMono α β
      inst✝ : SMulPosMono α β
      hab : LT.lt 0 (HSMul.hSMul a b)
      ha : LT.lt a 0
      hb : LE.le 0 b
      ⊢ LE.le (HSMul.hSMul a b) 0
    -/
    exact smul_nonpos_of_nonpos_of_nonneg ha.le hb
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      b : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : LinearOrder α
      inst✝² : LinearOrder β
      inst✝¹ : PosSMulMono α β
      inst✝ : SMulPosMono α β
      hab : LT.lt 0 (HSMul.hSMul 0 b)
      ⊢ Or (And (LT.lt 0 0) (LT.lt 0 b)) (And (LT.lt 0 0) (LT.lt b 0))
    -/
  · rw [zero_smul] at hab
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      b : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : LinearOrder α
      inst✝² : LinearOrder β
      inst✝¹ : PosSMulMono α β
      inst✝ : SMulPosMono α β
      hab : LT.lt 0 0
      ⊢ Or (And (LT.lt 0 0) (LT.lt 0 b)) (And (LT.lt 0 0) (LT.lt b 0))
    -/
    exact hab.false.elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      a : α
      b : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : LinearOrder α
      inst✝² : LinearOrder β
      inst✝¹ : PosSMulMono α β
      inst✝ : SMulPosMono α β
      hab : LT.lt 0 (HSMul.hSMul a b)
      ha : LT.lt 0 a
      ⊢ Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))
    -/
  · refine Or.inl ⟨ha, lt_imp_lt_of_le_imp_le (fun hb ↦ ?_) hab⟩
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      a : α
      b : β
      inst✝⁶ : Zero α
      inst✝⁵ : Zero β
      inst✝⁴ : SMulWithZero α β
      inst✝³ : LinearOrder α
      inst✝² : LinearOrder β
      inst✝¹ : PosSMulMono α β
      inst✝ : SMulPosMono α β
      hab : LT.lt 0 (HSMul.hSMul a b)
      ha : LT.lt 0 a
      hb : LE.le b 0
      ⊢ LE.le (HSMul.hSMul a b) 0
    -/
    exact smul_nonpos_of_nonneg_of_nonpos ha.le hb
    /-
      🎉 no goals
    -/


lemma neg_of_smul_pos_right [PosSMulMono α β] [SMulPosMono α β] (h : 0 < a • b) (ha : a ≤ 0) :
    b < 0 := ((pos_and_pos_or_neg_and_neg_of_smul_pos h).resolve_left fun h ↦ h.1.not_le ha).2


lemma neg_of_smul_pos_left [PosSMulMono α β] [SMulPosMono α β] (h : 0 < a • b) (ha : b ≤ 0) :
    a < 0 := ((pos_and_pos_or_neg_and_neg_of_smul_pos h).resolve_left fun h ↦ h.2.not_le ha).1


lemma neg_iff_neg_of_smul_pos [PosSMulMono α β] [SMulPosMono α β] (hab : 0 < a • b) :
    a < 0 ↔ b < 0 :=
  ⟨neg_of_smul_pos_right hab ∘ le_of_lt, neg_of_smul_pos_left hab ∘ le_of_lt⟩


lemma neg_of_smul_neg_left' [SMulPosMono α β] (h : a • b < 0) (ha : 0 ≤ a) : b < 0 :=
  lt_of_not_ge fun hb ↦ (smul_nonneg' ha hb).not_lt h


lemma neg_of_smul_neg_right' [PosSMulMono α β] (h : a • b < 0) (hb : 0 ≤ b) : a < 0 :=
  lt_of_not_ge fun ha ↦ (smul_nonneg ha hb).not_lt h


@[simp]
lemma le_smul_iff_one_le_left [SMulPosMono α β] [SMulPosReflectLE α β] (hb : 0 < b) :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         a : α
                                         b : β
                                         inst✝⁶ : Monoid α
                                         inst✝⁵ : Zero β
                                         inst✝⁴ : MulAction α β
                                         inst✝³ : Preorder α
                                         inst✝² : Preorder β
                                         inst✝¹ : SMulPosMono α β
                                         inst✝ : SMulPosReflectLE α β
                                         hb : LT.lt 0 b
                                         ⊢ Iff (LE.le b (HSMul.hSMul a b)) (LE.le (HSMul.hSMul 1 b) (HSMul.hSMul a b))
                                       -/
    b ≤ a • b ↔ 1 ≤ a := Iff.trans (by rw [one_smul]) (smul_le_smul_iff_of_pos_right hb)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
lemma lt_smul_iff_one_lt_left [SMulPosStrictMono α β] [SMulPosReflectLT α β] (hb : 0 < b) :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         a : α
                                         b : β
                                         inst✝⁶ : Monoid α
                                         inst✝⁵ : Zero β
                                         inst✝⁴ : MulAction α β
                                         inst✝³ : Preorder α
                                         inst✝² : Preorder β
                                         inst✝¹ : SMulPosStrictMono α β
                                         inst✝ : SMulPosReflectLT α β
                                         hb : LT.lt 0 b
                                         ⊢ Iff (LT.lt b (HSMul.hSMul a b)) (LT.lt (HSMul.hSMul 1 b) (HSMul.hSMul a b))
                                       -/
    b < a • b ↔ 1 < a := Iff.trans (by rw [one_smul]) (smul_lt_smul_iff_of_pos_right hb)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
lemma smul_le_iff_le_one_left [SMulPosMono α β] [SMulPosReflectLE α β] (hb : 0 < b) :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         a : α
                                         b : β
                                         inst✝⁶ : Monoid α
                                         inst✝⁵ : Zero β
                                         inst✝⁴ : MulAction α β
                                         inst✝³ : Preorder α
                                         inst✝² : Preorder β
                                         inst✝¹ : SMulPosMono α β
                                         inst✝ : SMulPosReflectLE α β
                                         hb : LT.lt 0 b
                                         ⊢ Iff (LE.le (HSMul.hSMul a b) b) (LE.le (HSMul.hSMul a b) (HSMul.hSMul 1 b))
                                       -/
    a • b ≤ b ↔ a ≤ 1 := Iff.trans (by rw [one_smul]) (smul_le_smul_iff_of_pos_right hb)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
lemma smul_lt_iff_lt_one_left [SMulPosStrictMono α β] [SMulPosReflectLT α β] (hb : 0 < b) :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         a : α
                                         b : β
                                         inst✝⁶ : Monoid α
                                         inst✝⁵ : Zero β
                                         inst✝⁴ : MulAction α β
                                         inst✝³ : Preorder α
                                         inst✝² : Preorder β
                                         inst✝¹ : SMulPosStrictMono α β
                                         inst✝ : SMulPosReflectLT α β
                                         hb : LT.lt 0 b
                                         ⊢ Iff (LT.lt (HSMul.hSMul a b) b) (LT.lt (HSMul.hSMul a b) (HSMul.hSMul 1 b))
                                       -/
    a • b < b ↔ a < 1 := Iff.trans (by rw [one_smul]) (smul_lt_smul_iff_of_pos_right hb)
                                       /-
                                         🎉 no goals
                                       -/


lemma smul_le_of_le_one_left [SMulPosMono α β] (hb : 0 ≤ b) (h : a ≤ 1) : a • b ≤ b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Monoid α
    inst✝⁴ : Zero β
    inst✝³ : MulAction α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosMono α β
    hb : LE.le 0 b
    h : LE.le a 1
    ⊢ LE.le (HSMul.hSMul a b) b
  -/
  simpa only [one_smul] using smul_le_smul_of_nonneg_right h hb
  /-
    🎉 no goals
  -/


lemma le_smul_of_one_le_left [SMulPosMono α β] (hb : 0 ≤ b) (h : 1 ≤ a) : b ≤ a • b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Monoid α
    inst✝⁴ : Zero β
    inst✝³ : MulAction α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosMono α β
    hb : LE.le 0 b
    h : LE.le 1 a
    ⊢ LE.le b (HSMul.hSMul a b)
  -/
  simpa only [one_smul] using smul_le_smul_of_nonneg_right h hb
  /-
    🎉 no goals
  -/


lemma smul_lt_of_lt_one_left [SMulPosStrictMono α β] (hb : 0 < b) (h : a < 1) : a • b < b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Monoid α
    inst✝⁴ : Zero β
    inst✝³ : MulAction α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosStrictMono α β
    hb : LT.lt 0 b
    h : LT.lt a 1
    ⊢ LT.lt (HSMul.hSMul a b) b
  -/
  simpa only [one_smul] using smul_lt_smul_of_pos_right h hb
  /-
    🎉 no goals
  -/


lemma lt_smul_of_one_lt_left [SMulPosStrictMono α β] (hb : 0 < b) (h : 1 < a) : b < a • b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁵ : Monoid α
    inst✝⁴ : Zero β
    inst✝³ : MulAction α β
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : SMulPosStrictMono α β
    hb : LT.lt 0 b
    h : LT.lt 1 a
    ⊢ LT.lt b (HSMul.hSMul a b)
  -/
  simpa only [one_smul] using smul_lt_smul_of_pos_right h hb
  /-
    🎉 no goals
  -/


lemma PosSMulMono.toPosSMulStrictMono [PosSMulMono α β] : PosSMulStrictMono α β :=
  ⟨fun _a ha _b₁ _b₂ hb ↦ (smul_le_smul_of_nonneg_left hb.le ha.le).lt_of_ne <|
    (smul_right_injective _ ha.ne').ne hb.ne⟩


instance PosSMulReflectLT.toPosSMulReflectLE [PosSMulReflectLT α β] : PosSMulReflectLE α β :=
  ⟨fun _a ha _b₁ _b₂ h ↦ h.eq_or_lt.elim (fun h ↦ (smul_right_injective _ ha.ne' h).le) fun h' ↦
    (lt_of_smul_lt_smul_left h' ha.le).le⟩


lemma posSMulMono_iff_posSMulStrictMono : PosSMulMono α β ↔ PosSMulStrictMono α β :=
  ⟨fun _ ↦ PosSMulMono.toPosSMulStrictMono, fun _ ↦ inferInstance⟩


lemma PosSMulReflectLE_iff_posSMulReflectLT : PosSMulReflectLE α β ↔ PosSMulReflectLT α β :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ PosSMulReflectLT.toPosSMulReflectLE⟩


lemma SMulPosMono.toSMulPosStrictMono [SMulPosMono α β] : SMulPosStrictMono α β :=
  ⟨fun _b hb _a₁ _a₂ ha ↦ (smul_le_smul_of_nonneg_right ha.le hb.le).lt_of_ne <|
    (smul_left_injective _ hb.ne').ne ha.ne⟩


lemma smulPosMono_iff_smulPosStrictMono : SMulPosMono α β ↔ SMulPosStrictMono α β :=
  ⟨fun _ ↦ SMulPosMono.toSMulPosStrictMono, fun _ ↦ inferInstance⟩


lemma SMulPosReflectLT.toSMulPosReflectLE [SMulPosReflectLT α β] : SMulPosReflectLE α β :=
  ⟨fun _b hb _a₁ _a₂ h ↦ h.eq_or_lt.elim (fun h ↦ (smul_left_injective _ hb.ne' h).le) fun h' ↦
    (lt_of_smul_lt_smul_right h' hb.le).le⟩


lemma SMulPosReflectLE_iff_smulPosReflectLT : SMulPosReflectLE α β ↔ SMulPosReflectLT α β :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ SMulPosReflectLT.toSMulPosReflectLE⟩


lemma inv_smul_le_iff_of_pos [PosSMulMono α β] [PosSMulReflectLE α β] (ha : 0 < a) :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        a : α
                                        b₁ b₂ : β
                                        inst✝⁵ : GroupWithZero α
                                        inst✝⁴ : Preorder α
                                        inst✝³ : Preorder β
                                        inst✝² : MulAction α β
                                        inst✝¹ : PosSMulMono α β
                                        inst✝ : PosSMulReflectLE α β
                                        ha : LT.lt 0 a
                                        ⊢ Iff (LE.le (HSMul.hSMul (Inv.inv a) b₁) b₂) (LE.le b₁ (HSMul.hSMul a b₂))
                                      -/
    a⁻¹ • b₁ ≤ b₂ ↔ b₁ ≤ a • b₂ := by rw [← smul_le_smul_iff_of_pos_left ha, smul_inv_smul₀ ha.ne']
                                      /-
                                        🎉 no goals
                                      -/


lemma le_inv_smul_iff_of_pos [PosSMulMono α β] [PosSMulReflectLE α β] (ha : 0 < a) :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        a : α
                                        b₁ b₂ : β
                                        inst✝⁵ : GroupWithZero α
                                        inst✝⁴ : Preorder α
                                        inst✝³ : Preorder β
                                        inst✝² : MulAction α β
                                        inst✝¹ : PosSMulMono α β
                                        inst✝ : PosSMulReflectLE α β
                                        ha : LT.lt 0 a
                                        ⊢ Iff (LE.le b₁ (HSMul.hSMul (Inv.inv a) b₂)) (LE.le (HSMul.hSMul a b₁) b₂)
                                      -/
    b₁ ≤ a⁻¹ • b₂ ↔ a • b₁ ≤ b₂ := by rw [← smul_le_smul_iff_of_pos_left ha, smul_inv_smul₀ ha.ne']
                                      /-
                                        🎉 no goals
                                      -/


lemma inv_smul_lt_iff_of_pos [PosSMulStrictMono α β] [PosSMulReflectLT α β] (ha : 0 < a) :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        a : α
                                        b₁ b₂ : β
                                        inst✝⁵ : GroupWithZero α
                                        inst✝⁴ : Preorder α
                                        inst✝³ : Preorder β
                                        inst✝² : MulAction α β
                                        inst✝¹ : PosSMulStrictMono α β
                                        inst✝ : PosSMulReflectLT α β
                                        ha : LT.lt 0 a
                                        ⊢ Iff (LT.lt (HSMul.hSMul (Inv.inv a) b₁) b₂) (LT.lt b₁ (HSMul.hSMul a b₂))
                                      -/
    a⁻¹ • b₁ < b₂ ↔ b₁ < a • b₂ := by rw [← smul_lt_smul_iff_of_pos_left ha, smul_inv_smul₀ ha.ne']
                                      /-
                                        🎉 no goals
                                      -/


lemma lt_inv_smul_iff_of_pos [PosSMulStrictMono α β] [PosSMulReflectLT α β] (ha : 0 < a) :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        a : α
                                        b₁ b₂ : β
                                        inst✝⁵ : GroupWithZero α
                                        inst✝⁴ : Preorder α
                                        inst✝³ : Preorder β
                                        inst✝² : MulAction α β
                                        inst✝¹ : PosSMulStrictMono α β
                                        inst✝ : PosSMulReflectLT α β
                                        ha : LT.lt 0 a
                                        ⊢ Iff (LT.lt b₁ (HSMul.hSMul (Inv.inv a) b₂)) (LT.lt (HSMul.hSMul a b₁) b₂)
                                      -/
    b₁ < a⁻¹ • b₂ ↔ a • b₁ < b₂ := by rw [← smul_lt_smul_iff_of_pos_left ha, smul_inv_smul₀ ha.ne']
                                      /-
                                        🎉 no goals
                                      -/


/-- Right scalar multiplication as an order isomorphism. -/
@[simps!]
def OrderIso.smulRight [PosSMulMono α β] [PosSMulReflectLE α β] {a : α} (ha : 0 < a) : β ≃o β where
  toEquiv := Equiv.smulRight ha.ne'
  map_rel_iff' := smul_le_smul_iff_of_pos_left ha


instance instPosSMulMono [PosSMulMono α β] : PosSMulMono α βᵒᵈ where
  elim _a ha _b₁ _b₂ hb := smul_le_smul_of_nonneg_left (β := β) hb ha

instance instPosSMulStrictMono [PosSMulStrictMono α β] : PosSMulStrictMono α βᵒᵈ where
  elim _a ha _b₁ _b₂ hb := smul_lt_smul_of_pos_left (β := β) hb ha

instance instPosSMulReflectLT [PosSMulReflectLT α β] : PosSMulReflectLT α βᵒᵈ where
  elim _a ha _b₁ _b₂ h := lt_of_smul_lt_smul_of_nonneg_left (β := β) h ha

instance instPosSMulReflectLE [PosSMulReflectLE α β] : PosSMulReflectLE α βᵒᵈ where
  elim _a ha _b₁ _b₂ h := le_of_smul_le_smul_of_pos_left (β := β) h ha


instance instSMulPosMono [SMulPosMono α β] : SMulPosMono α βᵒᵈ where
  elim _b hb a₁ a₂ ha := by
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosMono α β
      _b : OrderDual β
      hb : LE.le 0 _b
      a₁ a₂ : α
      ha : LE.le a₁ a₂
      ⊢ LE.le (HSMul.hSMul a₁ _b) (HSMul.hSMul a₂ _b)
    -/
    rw [← neg_le_neg_iff, ← smul_neg, ← smul_neg]
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosMono α β
      _b : OrderDual β
      hb : LE.le 0 _b
      a₁ a₂ : α
      ha : LE.le a₁ a₂
      ⊢ LE.le (HSMul.hSMul a₂ (Neg.neg _b)) (HSMul.hSMul a₁ (Neg.neg _b))
    -/
    exact smul_le_smul_of_nonneg_right (β := β) ha <| neg_nonneg.2 hb
    /-
      🎉 no goals
    -/


instance instSMulPosStrictMono [SMulPosStrictMono α β] : SMulPosStrictMono α βᵒᵈ where
  elim _b hb a₁ a₂ ha := by
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosStrictMono α β
      _b : OrderDual β
      hb : LT.lt 0 _b
      a₁ a₂ : α
      ha : LT.lt a₁ a₂
      ⊢ LT.lt (HSMul.hSMul a₁ _b) (HSMul.hSMul a₂ _b)
    -/
    rw [← neg_lt_neg_iff, ← smul_neg, ← smul_neg]
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosStrictMono α β
      _b : OrderDual β
      hb : LT.lt 0 _b
      a₁ a₂ : α
      ha : LT.lt a₁ a₂
      ⊢ LT.lt (HSMul.hSMul a₂ (Neg.neg _b)) (HSMul.hSMul a₁ (Neg.neg _b))
    -/
    exact smul_lt_smul_of_pos_right (β := β) ha <| neg_pos.2 hb
    /-
      🎉 no goals
    -/


instance instSMulPosReflectLT [SMulPosReflectLT α β] : SMulPosReflectLT α βᵒᵈ where
  elim _b hb a₁ a₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosReflectLT α β
      _b : OrderDual β
      hb : LE.le 0 _b
      a₁ a₂ : α
      h : LT.lt (HSMul.hSMul a₁ _b) (HSMul.hSMul a₂ _b)
      ⊢ LT.lt a₁ a₂
    -/
    rw [← neg_lt_neg_iff, ← smul_neg, ← smul_neg] at h
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosReflectLT α β
      _b : OrderDual β
      hb : LE.le 0 _b
      a₁ a₂ : α
      h : LT.lt (HSMul.hSMul a₂ (Neg.neg _b)) (HSMul.hSMul a₁ (Neg.neg _b))
      ⊢ LT.lt a₁ a₂
    -/
    exact lt_of_smul_lt_smul_right (β := β) h <| neg_nonneg.2 hb
    /-
      🎉 no goals
    -/


instance instSMulPosReflectLE [SMulPosReflectLE α β] : SMulPosReflectLE α βᵒᵈ where
  elim _b hb a₁ a₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosReflectLE α β
      _b : OrderDual β
      hb : LT.lt 0 _b
      a₁ a₂ : α
      h : LE.le (HSMul.hSMul a₁ _b) (HSMul.hSMul a₂ _b)
      ⊢ LE.le a₁ a₂
    -/
    rw [← neg_le_neg_iff, ← smul_neg, ← smul_neg] at h
    /-
      α : Type u_1
      β : Type u_2
      a a₁✝ a₂✝ : α
      b b₁ b₂ : β
      inst✝⁴ : Preorder α
      inst✝³ : Monoid α
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : DistribMulAction α β
      inst✝ : SMulPosReflectLE α β
      _b : OrderDual β
      hb : LT.lt 0 _b
      a₁ a₂ : α
      h : LE.le (HSMul.hSMul a₂ (Neg.neg _b)) (HSMul.hSMul a₁ (Neg.neg _b))
      ⊢ LE.le a₁ a₂
    -/
    exact le_of_smul_le_smul_right (β := β) h <| neg_pos.2 hb
    /-
      🎉 no goals
    -/


/-- Binary **rearrangement inequality**. -/
lemma smul_add_smul_le_smul_add_smul (ha : a₁ ≤ a₂) (hb : b₁ ≤ b₂) :
    a₁ • b₂ + a₂ • b₁ ≤ a₁ • b₁ + a₂ • b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulMono α β
    a₁ a₂ : α
    b₁ b₂ : β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.hSMul a₂ b₁)) (HAdd.hAdd (HSMul. …
  -/
  obtain ⟨a, ha₀, rfl⟩ := exists_nonneg_add_of_le ha
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulMono α β
    a₁ : α
    b₁ b₂ : β
    hb : LE.le b₁ b₂
    a : α
    ha₀ : LE.le 0 a
    ha : LE.le a₁ (HAdd.hAdd a₁ a)
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.hSMul (HAdd.hAdd a₁ a) b₁)) (HAd …
  -/
  rw [add_smul, add_smul, add_left_comm]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulMono α β
    a₁ : α
    b₁ b₂ : β
    hb : LE.le b₁ b₂
    a : α
    ha₀ : LE.le 0 a
    ha : LE.le a₁ (HAdd.hAdd a₁ a)
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a₁ b₁) (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.h …
  -/
  gcongr
  /-
    🎉 no goals
  -/


/-- Binary **rearrangement inequality**. -/
lemma smul_add_smul_le_smul_add_smul' (ha : a₂ ≤ a₁) (hb : b₂ ≤ b₁) :
    a₁ • b₂ + a₂ • b₁ ≤ a₁ • b₁ + a₂ • b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulMono α β
    a₁ a₂ : α
    b₁ b₂ : β
    ha : LE.le a₂ a₁
    hb : LE.le b₂ b₁
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.hSMul a₂ b₁)) (HAdd.hAdd (HSMul. …
  -/
  simp_rw [add_comm (a₁ • _)]; exact smul_add_smul_le_smul_add_smul ha hb
                               /-
                                 🎉 no goals
                               -/


/-- Binary strict **rearrangement inequality**. -/
lemma smul_add_smul_lt_smul_add_smul (ha : a₁ < a₂) (hb : b₁ < b₂) :
    a₁ • b₂ + a₂ • b₁ < a₁ • b₁ + a₂ • b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a₁ a₂ : α
    b₁ b₂ : β
    ha : LT.lt a₁ a₂
    hb : LT.lt b₁ b₂
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.hSMul a₂ b₁)) (HAdd.hAdd (HSMul. …
  -/
  obtain ⟨a, ha₀, rfl⟩ := lt_iff_exists_pos_add.1 ha
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a₁ : α
    b₁ b₂ : β
    hb : LT.lt b₁ b₂
    a : α
    ha₀ : LT.lt 0 a
    ha : LT.lt a₁ (HAdd.hAdd a₁ a)
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.hSMul (HAdd.hAdd a₁ a) b₁)) (HAd …
  -/
  rw [add_smul, add_smul, add_left_comm]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a₁ : α
    b₁ b₂ : β
    hb : LT.lt b₁ b₂
    a : α
    ha₀ : LT.lt 0 a
    ha : LT.lt a₁ (HAdd.hAdd a₁ a)
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a₁ b₁) (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.h …
  -/
  gcongr
  /-
    🎉 no goals
  -/


/-- Binary strict **rearrangement inequality**. -/
lemma smul_add_smul_lt_smul_add_smul' (ha : a₂ < a₁) (hb : b₂ < b₁) :
    a₁ • b₂ + a₂ • b₁ < a₁ • b₁ + a₂ • b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : StrictOrderedSemiring α
    inst✝³ : ExistsAddOfLE α
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a₁ a₂ : α
    b₁ b₂ : β
    ha : LT.lt a₂ a₁
    hb : LT.lt b₂ b₁
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a₁ b₂) (HSMul.hSMul a₂ b₁)) (HAdd.hAdd (HSMul. …
  -/
  simp_rw [add_comm (a₁ • _)]; exact smul_add_smul_lt_smul_add_smul ha hb
                               /-
                                 🎉 no goals
                               -/


lemma smul_le_smul_of_nonpos_left (h : b₁ ≤ b₂) (ha : a ≤ 0) : a • b₂ ≤ a • b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulMono α β
    h : LE.le b₁ b₂
    ha : LE.le a 0
    ⊢ LE.le (HSMul.hSMul a b₂) (HSMul.hSMul a b₁)
  -/
  rw [← neg_neg a, neg_smul, neg_smul (-a), neg_le_neg_iff]
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulMono α β
    h : LE.le b₁ b₂
    ha : LE.le a 0
    ⊢ LE.le (HSMul.hSMul (Neg.neg a) b₁) (HSMul.hSMul (Neg.neg a) b₂)
  -/
  exact smul_le_smul_of_nonneg_left h (neg_nonneg_of_nonpos ha)
  /-
    🎉 no goals
  -/


lemma antitone_smul_left (ha : a ≤ 0) : Antitone ((a • ·) : β → β) :=
  fun _ _ h ↦ smul_le_smul_of_nonpos_left h ha


instance PosSMulMono.toSMulPosMono : SMulPosMono α β where
                            /-
                              α : Type u_1
                              β : Type u_2
                              a a₁✝ a₂✝ : α
                              b b₁ b₂ : β
                              inst✝³ : OrderedRing α
                              inst✝² : OrderedAddCommGroup β
                              inst✝¹ : Module α β
                              inst✝ : PosSMulMono α β
                              _b : β
                              hb : LE.le 0 _b
                              a₁ a₂ : α
                              ha : LE.le a₁ a₂
                              ⊢ LE.le (HSMul.hSMul a₁ _b) (HSMul.hSMul a₂ _b)
                            -/
  elim _b hb a₁ a₂ ha := by rw [← sub_nonneg, ← sub_smul]; exact smul_nonneg (sub_nonneg.2 ha) hb
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma smul_lt_smul_of_neg_left (hb : b₁ < b₂) (ha : a < 0) : a • b₂ < a • b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    hb : LT.lt b₁ b₂
    ha : LT.lt a 0
    ⊢ LT.lt (HSMul.hSMul a b₂) (HSMul.hSMul a b₁)
  -/
  rw [← neg_neg a, neg_smul, neg_smul (-a), neg_lt_neg_iff]
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    hb : LT.lt b₁ b₂
    ha : LT.lt a 0
    ⊢ LT.lt (HSMul.hSMul (Neg.neg a) b₁) (HSMul.hSMul (Neg.neg a) b₂)
  -/
  exact smul_lt_smul_of_pos_left hb (neg_pos_of_neg ha)
  /-
    🎉 no goals
  -/


lemma strictAnti_smul_left (ha : a < 0) : StrictAnti ((a • ·) : β → β) :=
  fun _ _ h ↦ smul_lt_smul_of_neg_left h ha


instance PosSMulStrictMono.toSMulPosStrictMono : SMulPosStrictMono α β where
                            /-
                              α : Type u_1
                              β : Type u_2
                              a a₁✝ a₂✝ : α
                              b b₁ b₂ : β
                              inst✝³ : OrderedRing α
                              inst✝² : OrderedAddCommGroup β
                              inst✝¹ : Module α β
                              inst✝ : PosSMulStrictMono α β
                              _b : β
                              hb : LT.lt 0 _b
                              a₁ a₂ : α
                              ha : LT.lt a₁ a₂
                              ⊢ LT.lt (HSMul.hSMul a₁ _b) (HSMul.hSMul a₂ _b)
                            -/
  elim _b hb a₁ a₂ ha := by rw [← sub_pos, ← sub_smul]; exact smul_pos (sub_pos.2 ha) hb
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma le_of_smul_le_smul_of_neg [PosSMulReflectLE α β] (h : a • b₁ ≤ a • b₂) (ha : a < 0) :
    b₂ ≤ b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulReflectLE α β
    h : LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
    ha : LT.lt a 0
    ⊢ LE.le b₂ b₁
  -/
  rw [← neg_neg a, neg_smul, neg_smul (-a), neg_le_neg_iff] at h
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulReflectLE α β
    h : LE.le (HSMul.hSMul (Neg.neg a) b₂) (HSMul.hSMul (Neg.neg a) b₁)
    ha : LT.lt a 0
    ⊢ LE.le b₂ b₁
  -/
  exact le_of_smul_le_smul_of_pos_left h <| neg_pos.2 ha
  /-
    🎉 no goals
  -/


lemma lt_of_smul_lt_smul_of_nonpos [PosSMulReflectLT α β] (h : a • b₁ < a • b₂) (ha : a ≤ 0) :
    b₂ < b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulReflectLT α β
    h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
    ha : LE.le a 0
    ⊢ LT.lt b₂ b₁
  -/
  rw [← neg_neg a, neg_smul, neg_smul (-a), neg_lt_neg_iff] at h
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝³ : OrderedRing α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulReflectLT α β
    h : LT.lt (HSMul.hSMul (Neg.neg a) b₂) (HSMul.hSMul (Neg.neg a) b₁)
    ha : LE.le a 0
    ⊢ LT.lt b₂ b₁
  -/
  exact lt_of_smul_lt_smul_of_nonneg_left h (neg_nonneg_of_nonpos ha)
  /-
    🎉 no goals
  -/


lemma smul_nonneg_of_nonpos_of_nonpos [SMulPosMono α β] (ha : a ≤ 0) (hb : b ≤ 0) : 0 ≤ a • b :=
  smul_nonpos_of_nonpos_of_nonneg (β := βᵒᵈ) ha hb


lemma smul_le_smul_iff_of_neg_left [PosSMulMono α β] [PosSMulReflectLE α β] (ha : a < 0) :
    a • b₁ ≤ a • b₂ ↔ b₂ ≤ b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝⁴ : OrderedRing α
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module α β
    inst✝¹ : PosSMulMono α β
    inst✝ : PosSMulReflectLE α β
    ha : LT.lt a 0
    ⊢ Iff (LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)) (LE.le b₂ b₁)
  -/
  rw [← neg_neg a, neg_smul, neg_smul (-a), neg_le_neg_iff]
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝⁴ : OrderedRing α
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module α β
    inst✝¹ : PosSMulMono α β
    inst✝ : PosSMulReflectLE α β
    ha : LT.lt a 0
    ⊢ Iff (LE.le (HSMul.hSMul (Neg.neg a) b₂) (HSMul.hSMul (Neg.neg a) b₁)) (LE.le …
  -/
  exact smul_le_smul_iff_of_pos_left (neg_pos_of_neg ha)
  /-
    🎉 no goals
  -/


lemma smul_lt_smul_iff_of_neg_left (ha : a < 0) : a • b₁ < a • b₂ ↔ b₂ < b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝⁴ : OrderedRing α
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module α β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : PosSMulReflectLT α β
    ha : LT.lt a 0
    ⊢ Iff (LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)) (LT.lt b₂ b₁)
  -/
  rw [← neg_neg a, neg_smul, neg_smul (-a), neg_lt_neg_iff]
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b₁ b₂ : β
    inst✝⁴ : OrderedRing α
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module α β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : PosSMulReflectLT α β
    ha : LT.lt a 0
    ⊢ Iff (LT.lt (HSMul.hSMul (Neg.neg a) b₂) (HSMul.hSMul (Neg.neg a) b₁)) (LT.lt …
  -/
  exact smul_lt_smul_iff_of_pos_left (neg_pos_of_neg ha)
  /-
    🎉 no goals
  -/


lemma smul_pos_iff_of_neg_left (ha : a < 0) : 0 < a • b ↔ b < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁴ : OrderedRing α
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module α β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : PosSMulReflectLT α β
    ha : LT.lt a 0
    ⊢ Iff (LT.lt 0 (HSMul.hSMul a b)) (LT.lt b 0)
  -/
  simpa only [smul_zero] using smul_lt_smul_iff_of_neg_left ha (b₁ := (0 : β))
  /-
    🎉 no goals
  -/


alias ⟨_, smul_pos_of_neg_of_neg⟩ := smul_pos_iff_of_neg_left


lemma smul_neg_iff_of_neg_left (ha : a < 0) : a • b < 0 ↔ 0 < b := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    inst✝⁴ : OrderedRing α
    inst✝³ : OrderedAddCommGroup β
    inst✝² : Module α β
    inst✝¹ : PosSMulStrictMono α β
    inst✝ : PosSMulReflectLT α β
    ha : LT.lt a 0
    ⊢ Iff (LT.lt (HSMul.hSMul a b) 0) (LT.lt 0 b)
  -/
  simpa only [smul_zero] using smul_lt_smul_iff_of_neg_left ha (b₂ := (0 : β))
  /-
    🎉 no goals
  -/


lemma smul_max_of_nonpos (ha : a ≤ 0) (b₁ b₂ : β) : a • max b₁ b₂ = min (a • b₁) (a • b₂) :=
  (antitone_smul_left ha : Antitone (_ : β → β)).map_max


lemma smul_min_of_nonpos (ha : a ≤ 0) (b₁ b₂ : β) : a • min b₁ b₂ = max (a • b₁) (a • b₂) :=
  (antitone_smul_left ha : Antitone (_ : β → β)).map_min


lemma nonneg_and_nonneg_or_nonpos_and_nonpos_of_smul_nonneg (hab : 0 ≤ a • b) :
    0 ≤ a ∧ 0 ≤ b ∨ a ≤ 0 ∧ b ≤ 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    hab : LE.le 0 (HSMul.hSMul a b)
    ⊢ Or (And (LE.le 0 a) (LE.le 0 b)) (And (LE.le a 0) (LE.le b 0))
  -/
  simp only [Decidable.or_iff_not_and_not, not_and, not_le]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    hab : LE.le 0 (HSMul.hSMul a b)
    ⊢ (LE.le 0 a → LT.lt b 0) → Not (LE.le a 0 → LT.lt 0 b)
  -/
  refine fun ab nab ↦ hab.not_lt ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    hab : LE.le 0 (HSMul.hSMul a b)
    ab : LE.le 0 a → LT.lt b 0
    nab : LE.le a 0 → LT.lt 0 b
    ⊢ LT.lt (HSMul.hSMul a b) 0
  -/
  obtain ha | rfl | ha := lt_trichotomy 0 a
  exacts [smul_neg_of_pos_of_neg ha (ab ha.le), ((ab le_rfl).asymm (nab le_rfl)).elim,
    smul_neg_of_neg_of_pos ha (nab ha.le)]


lemma smul_nonneg_iff : 0 ≤ a • b ↔ 0 ≤ a ∧ 0 ≤ b ∨ a ≤ 0 ∧ b ≤ 0 :=
  ⟨nonneg_and_nonneg_or_nonpos_and_nonpos_of_smul_nonneg,
    fun h ↦ h.elim (and_imp.2 smul_nonneg) (and_imp.2 smul_nonneg_of_nonpos_of_nonpos)⟩


lemma smul_nonpos_iff : a • b ≤ 0 ↔ 0 ≤ a ∧ b ≤ 0 ∨ a ≤ 0 ∧ 0 ≤ b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    ⊢ Iff (LE.le (HSMul.hSMul a b) 0) (Or (And (LE.le 0 a) (LE.le b 0)) (And (LE.l …
  -/
  rw [← neg_nonneg, ← smul_neg, smul_nonneg_iff, neg_nonneg, neg_nonpos]
  /-
    🎉 no goals
  -/


lemma smul_nonneg_iff_pos_imp_nonneg : 0 ≤ a • b ↔ (0 < a → 0 ≤ b) ∧ (0 < b → 0 ≤ a) :=
  smul_nonneg_iff.trans <| by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedRing α
      inst✝² : LinearOrderedAddCommGroup β
      inst✝¹ : Module α β
      inst✝ : PosSMulStrictMono α β
      a : α
      b : β
      ⊢ Iff (Or (And (LE.le 0 a) (LE.le 0 b)) (And (LE.le a 0) (LE.le b 0))) (And (L …
    -/
    simp_rw [← not_le, ← or_iff_not_imp_left]; have := le_total a 0; have := le_total b 0; tauto
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


lemma smul_nonneg_iff_neg_imp_nonpos : 0 ≤ a • b ↔ (a < 0 → b ≤ 0) ∧ (b < 0 → a ≤ 0) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    ⊢ Iff (LE.le 0 (HSMul.hSMul a b)) (And (LT.lt a 0 → LE.le b 0) (LT.lt b 0 → LE …
  -/
  rw [← neg_smul_neg, smul_nonneg_iff_pos_imp_nonneg]; simp only [neg_pos, neg_nonneg]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma smul_nonpos_iff_pos_imp_nonpos : a • b ≤ 0 ↔ (0 < a → b ≤ 0) ∧ (b < 0 → 0 ≤ a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    ⊢ Iff (LE.le (HSMul.hSMul a b) 0) (And (LT.lt 0 a → LE.le b 0) (LT.lt b 0 → LE …
  -/
  rw [← neg_nonneg, ← smul_neg, smul_nonneg_iff_pos_imp_nonneg]; simp only [neg_pos, neg_nonneg]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma smul_nonpos_iff_neg_imp_nonneg : a • b ≤ 0 ↔ (a < 0 → 0 ≤ b) ∧ (0 < b → a ≤ 0) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : PosSMulStrictMono α β
    a : α
    b : β
    ⊢ Iff (LE.le (HSMul.hSMul a b) 0) (And (LT.lt a 0 → LE.le 0 b) (LT.lt 0 b → LE …
  -/
  rw [← neg_nonneg, ← neg_smul, smul_nonneg_iff_pos_imp_nonneg]; simp only [neg_pos, neg_nonneg]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance (priority := 100) PosSMulMono.toPosSMulReflectLE [MulAction α β] [PosSMulMono α β] :
    PosSMulReflectLE α β where
  elim _a ha b₁ b₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      a a₁ a₂ : α
      b b₁✝ b₂✝ : β
      inst✝⁴ : LinearOrderedSemifield α
      inst✝³ : AddCommGroup β
      inst✝² : PartialOrder β
      inst✝¹ : MulAction α β
      inst✝ : PosSMulMono α β
      _a : α
      ha : LT.lt 0 _a
      b₁ b₂ : β
      h : LE.le (HSMul.hSMul _a b₁) (HSMul.hSMul _a b₂)
      ⊢ LE.le b₁ b₂
    -/
    simpa [ha.ne'] using smul_le_smul_of_nonneg_left h <| inv_nonneg.2 ha.le
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) PosSMulStrictMono.toPosSMulReflectLT [MulActionWithZero α β]
    [PosSMulStrictMono α β] : PosSMulReflectLT α β :=
  PosSMulReflectLT.of_pos fun a ha b₁ b₂ h ↦ by
    /-
      α : Type u_1
      β : Type u_2
      a✝ a₁ a₂ : α
      b b₁✝ b₂✝ : β
      inst✝⁴ : LinearOrderedSemifield α
      inst✝³ : AddCommGroup β
      inst✝² : PartialOrder β
      inst✝¹ : MulActionWithZero α β
      inst✝ : PosSMulStrictMono α β
      a : α
      ha : LT.lt 0 a
      b₁ b₂ : β
      h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
      ⊢ LT.lt b₁ b₂
    -/
    simpa [ha.ne'] using smul_lt_smul_of_pos_left h <| inv_pos.2 ha
    /-
      🎉 no goals
    -/


lemma inv_smul_le_iff_of_neg (h : a < 0) : a⁻¹ • b₁ ≤ b₂ ↔ a • b₂ ≤ b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    a : α
    b₁ b₂ : β
    inst✝ : PosSMulMono α β
    h : LT.lt a 0
    ⊢ Iff (LE.le (HSMul.hSMul (Inv.inv a) b₁) b₂) (LE.le (HSMul.hSMul a b₂) b₁)
  -/
  rw [← smul_le_smul_iff_of_neg_left h, smul_inv_smul₀ h.ne]
  /-
    🎉 no goals
  -/


lemma smul_inv_le_iff_of_neg (h : a < 0) : b₁ ≤ a⁻¹ • b₂ ↔ b₂ ≤ a • b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    a : α
    b₁ b₂ : β
    inst✝ : PosSMulMono α β
    h : LT.lt a 0
    ⊢ Iff (LE.le b₁ (HSMul.hSMul (Inv.inv a) b₂)) (LE.le b₂ (HSMul.hSMul a b₁))
  -/
  rw [← smul_le_smul_iff_of_neg_left h, smul_inv_smul₀ h.ne]
  /-
    🎉 no goals
  -/


/-- Left scalar multiplication as an order isomorphism. -/
@[simps!]
def OrderIso.smulRightDual (ha : a < 0) : β ≃o βᵒᵈ where
  toEquiv := (Equiv.smulRight ha.ne).trans toDual
  map_rel_iff' := (@OrderDual.toDual_le_toDual β).trans <| smul_le_smul_iff_of_neg_left ha


lemma inv_smul_lt_iff_of_neg (h : a < 0) : a⁻¹ • b₁ < b₂ ↔ a • b₂ < b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    a : α
    b₁ b₂ : β
    inst✝ : PosSMulStrictMono α β
    h : LT.lt a 0
    ⊢ Iff (LT.lt (HSMul.hSMul (Inv.inv a) b₁) b₂) (LT.lt (HSMul.hSMul a b₂) b₁)
  -/
  rw [← smul_lt_smul_iff_of_neg_left h, smul_inv_smul₀ h.ne]
  /-
    🎉 no goals
  -/


lemma smul_inv_lt_iff_of_neg (h : a < 0) : b₁ < a⁻¹ • b₂ ↔ b₂ < a • b₁ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module α β
    a : α
    b₁ b₂ : β
    inst✝ : PosSMulStrictMono α β
    h : LT.lt a 0
    ⊢ Iff (LT.lt b₁ (HSMul.hSMul (Inv.inv a) b₂)) (LT.lt b₂ (HSMul.hSMul a b₁))
  -/
  rw [← smul_lt_smul_iff_of_neg_left h, smul_inv_smul₀ h.ne]
  /-
    🎉 no goals
  -/


instance instPosSMulMono [∀ i, PosSMulMono α (β i)] : PosSMulMono α (∀ i, β i) where
  elim _a ha _b₁ _b₂ hb i := smul_le_smul_of_nonneg_left (hb i) ha


instance instSMulPosMono [∀ i, SMulPosMono α (β i)] : SMulPosMono α (∀ i, β i) where
  elim _b hb _a₁ _a₂ ha i := smul_le_smul_of_nonneg_right ha (hb i)


instance instPosSMulReflectLE [∀ i, PosSMulReflectLE α (β i)] : PosSMulReflectLE α (∀ i, β i) where
  elim _a ha _b₁ _b₂ h i := le_of_smul_le_smul_left (h i) ha


instance instSMulPosReflectLE [∀ i, SMulPosReflectLE α (β i)] : SMulPosReflectLE α (∀ i, β i) where
  elim _b hb _a₁ _a₂ h := by
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : Preorder α
      inst✝² : (i : ι) → Preorder (β i)
      inst✝¹ : (i : ι) → SMulZeroClass α (β i)
      inst✝ : ∀ (i : ι), SMulPosReflectLE α (β i)
      _b : (i : ι) → β i
      hb : LT.lt 0 _b
      _a₁ _a₂ : α
      h : LE.le (HSMul.hSMul _a₁ _b) (HSMul.hSMul _a₂ _b)
      ⊢ LE.le _a₁ _a₂
    -/
    obtain ⟨-, i, hi⟩ := lt_def.1 hb; exact le_of_smul_le_smul_right (h _) hi
                                      /-
                                        🎉 no goals
                                      -/


instance instPosSMulStrictMono [∀ i, PosSMulStrictMono α (β i)] :
    PosSMulStrictMono α (∀ i, β i) where
  elim := by
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), PosSMulStrictMono α (β i)
      ⊢ ∀ ⦃a : α⦄, LT.lt 0 a → ∀ ⦃b₁ b₂ : (i : ι) → β i⦄, LT.lt b₁ b₂ → LT.lt (HSMul …
    -/
    simp_rw [lt_def]
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), PosSMulStrictMono α (β i)
      ⊢ ∀ ⦃a : α⦄, LT.lt 0 a → ∀ ⦃b₁ b₂ : (i : ι) → β i⦄, And (LE.le b₁ b₂) (Exists  …
    -/
    rintro _a ha _b₁ _b₂ ⟨hb, i, hi⟩
    /-
      case intro.intro
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), PosSMulStrictMono α (β i)
      _a : α
      ha : LT.lt 0 _a
      _b₁ _b₂ : (i : ι) → β i
      hb : LE.le _b₁ _b₂
      i : ι
      hi : LT.lt (_b₁ i) (_b₂ i)
      ⊢ And (LE.le (HSMul.hSMul _a _b₁) (HSMul.hSMul _a _b₂)) (Exists fun i => LT.lt …
    -/
    exact ⟨smul_le_smul_of_nonneg_left hb ha.le, i, smul_lt_smul_of_pos_left hi ha⟩
    /-
      🎉 no goals
    -/


instance instSMulPosStrictMono [∀ i, SMulPosStrictMono α (β i)] :
    SMulPosStrictMono α (∀ i, β i) where
  elim := by
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), SMulPosStrictMono α (β i)
      ⊢ ∀ ⦃b : (i : ι) → β i⦄, LT.lt 0 b → ∀ ⦃a₁ a₂ : α⦄, LT.lt a₁ a₂ → LT.lt (HSMul …
    -/
    simp_rw [lt_def]
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), SMulPosStrictMono α (β i)
      ⊢ ∀ ⦃b : (i : ι) → β i⦄, And (LE.le 0 b) (Exists fun i => LT.lt (0 i) (b i)) → …
    -/
    rintro a ⟨ha, i, hi⟩ _b₁ _b₂ hb
    /-
      case intro.intro
      α : Type u_1
      β✝ : Type u_2
      a✝ a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), SMulPosStrictMono α (β i)
      a : (i : ι) → β i
      ha : LE.le 0 a
      i : ι
      hi : LT.lt (0 i) (a i)
      _b₁ _b₂ : α
      hb : LT.lt _b₁ _b₂
      ⊢ And (LE.le (HSMul.hSMul _b₁ a) (HSMul.hSMul _b₂ a)) (Exists fun i => LT.lt ( …
    -/
    exact ⟨smul_le_smul_of_nonneg_right hb.le ha, i, smul_lt_smul_of_pos_right hb hi⟩
    /-
      🎉 no goals
    -/

-- Note: There is no interesting instance for `PosSMulReflectLT α (∀ i, β i)` that's not already
-- implied by the other instances


instance instSMulPosReflectLT [∀ i, SMulPosReflectLT α (β i)] : SMulPosReflectLT α (∀ i, β i) where
  elim := by
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), SMulPosReflectLT α (β i)
      ⊢ ∀ ⦃b : (i : ι) → β i⦄, LE.le 0 b → ∀ ⦃a₁ a₂ : α⦄, LT.lt (HSMul.hSMul a₁ b) ( …
    -/
    simp_rw [lt_def]
    /-
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), SMulPosReflectLT α (β i)
      ⊢ ∀ ⦃b : (i : ι) → β i⦄, LE.le 0 b → ∀ ⦃a₁ a₂ : α⦄, And (LE.le (HSMul.hSMul a₁ …
    -/
    rintro b hb _a₁ _a₂ ⟨-, i, hi⟩
    /-
      case intro.intro
      α : Type u_1
      β✝ : Type u_2
      a a₁ a₂ : α
      b✝ b₁ b₂ : β✝
      ι : Type u_3
      β : ι → Type u_4
      inst✝⁵ : Zero α
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : PartialOrder α
      inst✝² : (i : ι) → PartialOrder (β i)
      inst✝¹ : (i : ι) → SMulWithZero α (β i)
      inst✝ : ∀ (i : ι), SMulPosReflectLT α (β i)
      b : (i : ι) → β i
      hb : LE.le 0 b
      _a₁ _a₂ : α
      i : ι
      hi : LT.lt (HSMul.hSMul _a₁ b i) (HSMul.hSMul _a₂ b i)
      ⊢ LT.lt _a₁ _a₂
    -/
    exact lt_of_smul_lt_smul_right hi <| hb _
    /-
      🎉 no goals
    -/


lemma PosSMulMono.lift [PosSMulMono α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b) : PosSMulMono α β where
                           /-
                             α : Type u_1
                             β : Type u_2
                             γ : Type u_3
                             inst✝⁶ : Preorder α
                             inst✝⁵ : Preorder β
                             inst✝⁴ : Preorder γ
                             inst✝³ : SMul α β
                             inst✝² : SMul α γ
                             f : β → γ
                             inst✝¹ : Zero α
                             inst✝ : PosSMulMono α γ
                             hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
                             smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
                             a : α
                             ha : LE.le 0 a
                             b₁ b₂ : β
                             hb : LE.le b₁ b₂
                             ⊢ LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
                           -/
  elim a ha b₁ b₂ hb := by simp only [← hf, smul] at *; exact smul_le_smul_of_nonneg_left hb ha
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma PosSMulStrictMono.lift [PosSMulStrictMono α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b) : PosSMulStrictMono α β where
  elim a ha b₁ b₂ hb := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁶ : Preorder α
      inst✝⁵ : Preorder β
      inst✝⁴ : Preorder γ
      inst✝³ : SMul α β
      inst✝² : SMul α γ
      f : β → γ
      inst✝¹ : Zero α
      inst✝ : PosSMulStrictMono α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      a : α
      ha : LT.lt 0 a
      b₁ b₂ : β
      hb : LT.lt b₁ b₂
      ⊢ LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
    -/
    simp only [← lt_iff_lt_of_le_iff_le' hf hf, smul] at *; exact smul_lt_smul_of_pos_left hb ha
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma PosSMulReflectLE.lift [PosSMulReflectLE α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b) : PosSMulReflectLE α β where
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             γ : Type u_3
                                                             inst✝⁶ : Preorder α
                                                             inst✝⁵ : Preorder β
                                                             inst✝⁴ : Preorder γ
                                                             inst✝³ : SMul α β
                                                             inst✝² : SMul α γ
                                                             f : β → γ
                                                             inst✝¹ : Zero α
                                                             inst✝ : PosSMulReflectLE α γ
                                                             hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
                                                             smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
                                                             a : α
                                                             ha : LT.lt 0 a
                                                             b₁ b₂ : β
                                                             h : LE.le (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
                                                             ⊢ LE.le (HSMul.hSMul a (f b₁)) (HSMul.hSMul a (f b₂))
                                                           -/
  elim a ha b₁ b₂ h := hf.1 <| le_of_smul_le_smul_left (by simpa only [smul] using hf.2 h) ha
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma PosSMulReflectLT.lift [PosSMulReflectLT α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b) : PosSMulReflectLT α β where
  elim a ha b₁ b₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁶ : Preorder α
      inst✝⁵ : Preorder β
      inst✝⁴ : Preorder γ
      inst✝³ : SMul α β
      inst✝² : SMul α γ
      f : β → γ
      inst✝¹ : Zero α
      inst✝ : PosSMulReflectLT α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      a : α
      ha : LE.le 0 a
      b₁ b₂ : β
      h : LT.lt (HSMul.hSMul a b₁) (HSMul.hSMul a b₂)
      ⊢ LT.lt b₁ b₂
    -/
    simp only [← lt_iff_lt_of_le_iff_le' hf hf, smul] at *; exact lt_of_smul_lt_smul_left h ha
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma SMulPosMono.lift [SMulPosMono α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b)
    (zero : f 0 = 0) : SMulPosMono α β where
  elim b hb a₁ a₂ ha := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosMono α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      zero : Eq (f 0) 0
      b : β
      hb : LE.le 0 b
      a₁ a₂ : α
      ha : LE.le a₁ a₂
      ⊢ LE.le (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
    -/
    simp only [← hf, zero, smul] at *; exact smul_le_smul_of_nonneg_right ha hb
                                       /-
                                         🎉 no goals
                                       -/


lemma SMulPosStrictMono.lift [SMulPosStrictMono α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b)
    (zero : f 0 = 0) : SMulPosStrictMono α β where
  elim b hb a₁ a₂ ha := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosStrictMono α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      zero : Eq (f 0) 0
      b : β
      hb : LT.lt 0 b
      a₁ a₂ : α
      ha : LT.lt a₁ a₂
      ⊢ LT.lt (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
    -/
    simp only [← lt_iff_lt_of_le_iff_le' hf hf, zero, smul] at *
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosStrictMono α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      b : β
      a₁ a₂ : α
      ha : LT.lt a₁ a₂
      smul : α → β → True
      zero : True
      hb : LT.lt 0 (f b)
      ⊢ LT.lt (HSMul.hSMul a₁ (f b)) (HSMul.hSMul a₂ (f b))
    -/
    exact smul_lt_smul_of_pos_right ha hb
    /-
      🎉 no goals
    -/


lemma SMulPosReflectLE.lift [SMulPosReflectLE α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b)
    (zero : f 0 = 0) : SMulPosReflectLE α β where
  elim b hb a₁ a₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosReflectLE α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      zero : Eq (f 0) 0
      b : β
      hb : LT.lt 0 b
      a₁ a₂ : α
      h : LE.le (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
      ⊢ LE.le a₁ a₂
    -/
    simp only [← hf, ← lt_iff_lt_of_le_iff_le' hf hf, zero, smul] at *
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosReflectLE α γ
      b : β
      a₁ a₂ : α
      hf : ∀ {b₁ b₂ : β}, True
      smul : α → β → True
      zero : True
      hb : LT.lt 0 (f b)
      h : LE.le (HSMul.hSMul a₁ (f b)) (HSMul.hSMul a₂ (f b))
      ⊢ LE.le a₁ a₂
    -/
    exact le_of_smul_le_smul_right h hb
    /-
      🎉 no goals
    -/


lemma SMulPosReflectLT.lift [SMulPosReflectLT α γ]
    (hf : ∀ {b₁ b₂}, f b₁ ≤ f b₂ ↔ b₁ ≤ b₂)
    (smul : ∀ (a : α) b, f (a • b) = a • f b)
    (zero : f 0 = 0) : SMulPosReflectLT α β where
  elim b hb a₁ a₂ h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosReflectLT α γ
      hf : ∀ {b₁ b₂ : β}, Iff (LE.le (f b₁) (f b₂)) (LE.le b₁ b₂)
      smul : ∀ (a : α) (b : β), Eq (f (HSMul.hSMul a b)) (HSMul.hSMul a (f b))
      zero : Eq (f 0) 0
      b : β
      hb : LE.le 0 b
      a₁ a₂ : α
      h : LT.lt (HSMul.hSMul a₁ b) (HSMul.hSMul a₂ b)
      ⊢ LT.lt a₁ a₂
    -/
    simp only [← hf, ← lt_iff_lt_of_le_iff_le' hf hf, zero, smul] at *
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : SMul α β
      inst✝³ : SMul α γ
      f : β → γ
      inst✝² : Zero β
      inst✝¹ : Zero γ
      inst✝ : SMulPosReflectLT α γ
      b : β
      a₁ a₂ : α
      hf : ∀ {b₁ b₂ : β}, True
      smul : α → β → True
      zero : True
      hb : LE.le 0 (f b)
      h : LT.lt (HSMul.hSMul a₁ (f b)) (HSMul.hSMul a₂ (f b))
      ⊢ LT.lt a₁ a₂
    -/
    exact lt_of_smul_lt_smul_right h hb
    /-
      🎉 no goals
    -/


instance OrderedSemiring.toPosSMulMonoNat [OrderedSemiring α] : PosSMulMono ℕ α where
  elim _n _ _a _b hab := nsmul_le_nsmul_right hab _


instance OrderedSemiring.toSMulPosMonoNat [OrderedSemiring α] : SMulPosMono ℕ α where
  elim _a ha _m _n hmn := nsmul_le_nsmul_left ha hmn


instance StrictOrderedSemiring.toPosSMulStrictMonoNat [StrictOrderedSemiring α] :
    PosSMulStrictMono ℕ α where
  elim _n hn _a _b hab := nsmul_right_strictMono hn.ne' hab


instance StrictOrderedSemiring.toSMulPosStrictMonoNat [StrictOrderedSemiring α] :
    SMulPosStrictMono ℕ α where
  elim _a ha _m _n hmn := nsmul_lt_nsmul_left ha hmn


private theorem smul_nonneg_of_pos_of_nonneg (ha : 0 < a) (hb : 0 ≤ b) : 0 ≤ a • b :=
  smul_nonneg ha.le hb


private theorem smul_nonneg_of_nonneg_of_pos (ha : 0 ≤ a) (hb : 0 < b) : 0 ≤ a • b :=
  smul_nonneg ha hb.le


private theorem smul_ne_zero_of_pos_of_ne_zero [Preorder α] (ha : 0 < a) (hb : b ≠ 0) : a • b ≠ 0 :=
  smul_ne_zero ha.ne' hb


private theorem smul_ne_zero_of_ne_zero_of_pos [Preorder β] (ha : a ≠ 0) (hb : 0 < b) : a • b ≠ 0 :=
  smul_ne_zero ha hb.ne'


/-- Positivity extension for HSMul, i.e. (_ • _). -/
@[positivity HSMul.hSMul _ _]
def evalHSMul : PositivityExt where eval {_u α} zα pα (e : Q($α)) := do
  let .app (.app (.app (.app (.app (.app
        (.const ``HSMul.hSMul [u1, _, _]) (β : Q(Type u1))) _) _) _)
          (a : Q($β))) (b : Q($α)) ← whnfR e | throwError "failed to match hSMul"
  let zM : Q(Zero $β) ← synthInstanceQ q(Zero $β)
  let pM : Q(PartialOrder $β) ← synthInstanceQ q(PartialOrder $β)
  -- Using `q()` here would be impractical, as we would have to manually `synthInstanceQ` all the
  -- required typeclasses. Ideally we could tell `q()` to do this automatically.
  match ← core zM pM a, ← core zα pα b with
  | .positive pa, .positive pb =>
      pure (.positive (← mkAppM ``smul_pos #[pa, pb]))
  | .positive pa, .nonnegative pb =>
      pure (.nonnegative (← mkAppM ``smul_nonneg_of_pos_of_nonneg #[pa, pb]))
  | .nonnegative pa, .positive pb =>
      pure (.nonnegative (← mkAppM ``smul_nonneg_of_nonneg_of_pos #[pa, pb]))
  | .nonnegative pa, .nonnegative pb =>
      pure (.nonnegative (← mkAppM ``smul_nonneg #[pa, pb]))
  | .positive pa, .nonzero pb =>
      pure (.nonzero (← mkAppM ``smul_ne_zero_of_pos_of_ne_zero #[pa, pb]))
  | .nonzero pa, .positive pb =>
      pure (.nonzero (← mkAppM ``smul_ne_zero_of_ne_zero_of_pos #[pa, pb]))
  | .nonzero pa, .nonzero pb =>
      pure (.nonzero (← mkAppM ``smul_ne_zero #[pa, pb]))
  | _, _ => pure .none


