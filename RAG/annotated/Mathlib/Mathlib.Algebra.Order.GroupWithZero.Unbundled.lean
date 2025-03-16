/-- Local notation for the nonnegative elements of a type `α`. -/
local notation3 "α≥0" => { x : α // 0 ≤ x }


/-- Local notation for the positive elements of a type `α`. -/
local notation3 "α>0" => { x : α // 0 < x }


/-- Typeclass for monotonicity of multiplication by nonnegative elements on the left,
namely `b₁ ≤ b₂ → a * b₁ ≤ a * b₂` if `0 ≤ a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSemiring`. -/
abbrev PosMulMono : Prop :=
  CovariantClass α≥0 α (fun x y => x * y) (· ≤ ·)


/-- Typeclass for monotonicity of multiplication by nonnegative elements on the right,
namely `a₁ ≤ a₂ → a₁ * b ≤ a₂ * b` if `0 ≤ b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedSemiring`. -/
abbrev MulPosMono : Prop :=
  CovariantClass α≥0 α (fun x y => y * x) (· ≤ ·)


/-- Typeclass for strict monotonicity of multiplication by positive elements on the left,
namely `b₁ < b₂ → a * b₁ < a * b₂` if `0 < a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`StrictOrderedSemiring`. -/
abbrev PosMulStrictMono : Prop :=
  CovariantClass α>0 α (fun x y => x * y) (· < ·)


/-- Typeclass for strict monotonicity of multiplication by positive elements on the right,
namely `a₁ < a₂ → a₁ * b < a₂ * b` if `0 < b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`StrictOrderedSemiring`. -/
abbrev MulPosStrictMono : Prop :=
  CovariantClass α>0 α (fun x y => y * x) (· < ·)


/-- Typeclass for strict reverse monotonicity of multiplication by nonnegative elements on
the left, namely `a * b₁ < a * b₂ → b₁ < b₂` if `0 ≤ a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`LinearOrderedSemiring`. -/
abbrev PosMulReflectLT : Prop :=
  ContravariantClass α≥0 α (fun x y => x * y) (· < ·)


/-- Typeclass for strict reverse monotonicity of multiplication by nonnegative elements on
the right, namely `a₁ * b < a₂ * b → a₁ < a₂` if `0 ≤ b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`LinearOrderedSemiring`. -/
abbrev MulPosReflectLT : Prop :=
  ContravariantClass α≥0 α (fun x y => y * x) (· < ·)


/-- Typeclass for reverse monotonicity of multiplication by positive elements on the left,
namely `a * b₁ ≤ a * b₂ → b₁ ≤ b₂` if `0 < a`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`LinearOrderedSemiring`. -/
abbrev PosMulReflectLE : Prop :=
  ContravariantClass α>0 α (fun x y => x * y) (· ≤ ·)


/-- Typeclass for reverse monotonicity of multiplication by positive elements on the right,
namely `a₁ * b ≤ a₂ * b → a₁ ≤ a₂` if `0 < b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`LinearOrderedSemiring`. -/
abbrev MulPosReflectLE : Prop :=
  ContravariantClass α>0 α (fun x y => y * x) (· ≤ ·)


instance PosMulMono.to_covariantClass_pos_mul_le [PosMulMono α] :
    CovariantClass α>0 α (fun x y => x * y) (· ≤ ·) :=
  ⟨fun a _ _ bc => @CovariantClass.elim α≥0 α (fun x y => x * y) (· ≤ ·) _ ⟨_, a.2.le⟩ _ _ bc⟩


instance MulPosMono.to_covariantClass_pos_mul_le [MulPosMono α] :
    CovariantClass α>0 α (fun x y => y * x) (· ≤ ·) :=
  ⟨fun a _ _ bc => @CovariantClass.elim α≥0 α (fun x y => y * x) (· ≤ ·) _ ⟨_, a.2.le⟩ _ _ bc⟩


instance PosMulReflectLT.to_contravariantClass_pos_mul_lt [PosMulReflectLT α] :
    ContravariantClass α>0 α (fun x y => x * y) (· < ·) :=
  ⟨fun a _ _ bc => @ContravariantClass.elim α≥0 α (fun x y => x * y) (· < ·) _ ⟨_, a.2.le⟩ _ _ bc⟩


instance MulPosReflectLT.to_contravariantClass_pos_mul_lt [MulPosReflectLT α] :
    ContravariantClass α>0 α (fun x y => y * x) (· < ·) :=
  ⟨fun a _ _ bc => @ContravariantClass.elim α≥0 α (fun x y => y * x) (· < ·) _ ⟨_, a.2.le⟩ _ _ bc⟩


instance (priority := 100) MulLeftMono.toPosMulMono [MulLeftMono α] :
    PosMulMono α where elim _ _ := ‹MulLeftMono α›.elim _


instance (priority := 100) MulRightMono.toMulPosMono [MulRightMono α] :
    MulPosMono α where elim _ _ := ‹MulRightMono α›.elim _


instance (priority := 100) MulLeftStrictMono.toPosMulStrictMono [MulLeftStrictMono α] :
    PosMulStrictMono α where elim _ _ := ‹MulLeftStrictMono α›.elim _


instance (priority := 100) MulRightStrictMono.toMulPosStrictMono [MulRightStrictMono α] :
    MulPosStrictMono α where elim _ _ := ‹MulRightStrictMono α›.elim _


instance (priority := 100) MulLeftMono.toPosMulReflectLT [MulLeftReflectLT α] :
   PosMulReflectLT α where elim _ _ := ‹MulLeftReflectLT α›.elim _


instance (priority := 100) MulRightMono.toMulPosReflectLT [MulRightReflectLT α] :
   MulPosReflectLT α where elim _ _ := ‹MulRightReflectLT α›.elim _


instance (priority := 100) MulLeftStrictMono.toPosMulReflectLE [MulLeftReflectLE α] :
   PosMulReflectLE α where elim _ _ := ‹MulLeftReflectLE α›.elim _


instance (priority := 100) MulRightStrictMono.toMulPosReflectLE [MulRightReflectLE α] :
   MulPosReflectLE α where elim _ _ := ‹MulRightReflectLE α›.elim _


@[gcongr]
theorem mul_le_mul_of_nonneg_left [PosMulMono α] (h : b ≤ c) (a0 : 0 ≤ a) : a * b ≤ a * c :=
  @CovariantClass.elim α≥0 α (fun x y => x * y) (· ≤ ·) _ ⟨a, a0⟩ _ _ h


@[gcongr]
theorem mul_le_mul_of_nonneg_right [MulPosMono α] (h : b ≤ c) (a0 : 0 ≤ a) : b * a ≤ c * a :=
  @CovariantClass.elim α≥0 α (fun x y => y * x) (· ≤ ·) _ ⟨a, a0⟩ _ _ h


@[gcongr]
theorem mul_lt_mul_of_pos_left [PosMulStrictMono α] (bc : b < c) (a0 : 0 < a) : a * b < a * c :=
  @CovariantClass.elim α>0 α (fun x y => x * y) (· < ·) _ ⟨a, a0⟩ _ _ bc


@[gcongr]
theorem mul_lt_mul_of_pos_right [MulPosStrictMono α] (bc : b < c) (a0 : 0 < a) : b * a < c * a :=
  @CovariantClass.elim α>0 α (fun x y => y * x) (· < ·) _ ⟨a, a0⟩ _ _ bc


theorem lt_of_mul_lt_mul_left [PosMulReflectLT α] (h : a * b < a * c) (a0 : 0 ≤ a) : b < c :=
  @ContravariantClass.elim α≥0 α (fun x y => x * y) (· < ·) _ ⟨a, a0⟩ _ _ h


theorem lt_of_mul_lt_mul_right [MulPosReflectLT α] (h : b * a < c * a) (a0 : 0 ≤ a) : b < c :=
  @ContravariantClass.elim α≥0 α (fun x y => y * x) (· < ·) _ ⟨a, a0⟩ _ _ h


theorem le_of_mul_le_mul_left [PosMulReflectLE α] (bc : a * b ≤ a * c) (a0 : 0 < a) : b ≤ c :=
  @ContravariantClass.elim α>0 α (fun x y => x * y) (· ≤ ·) _ ⟨a, a0⟩ _ _ bc


theorem le_of_mul_le_mul_right [MulPosReflectLE α] (bc : b * a ≤ c * a) (a0 : 0 < a) : b ≤ c :=
  @ContravariantClass.elim α>0 α (fun x y => y * x) (· ≤ ·) _ ⟨a, a0⟩ _ _ bc


alias lt_of_mul_lt_mul_of_nonneg_left := lt_of_mul_lt_mul_left

alias lt_of_mul_lt_mul_of_nonneg_right := lt_of_mul_lt_mul_right

alias le_of_mul_le_mul_of_pos_left := le_of_mul_le_mul_left

alias le_of_mul_le_mul_of_pos_right := le_of_mul_le_mul_right


@[simp]
theorem mul_lt_mul_left [PosMulStrictMono α] [PosMulReflectLT α] (a0 : 0 < a) :
    a * b < a * c ↔ b < c :=
  @rel_iff_cov α>0 α (fun x y => x * y) (· < ·) _ _ ⟨a, a0⟩ _ _


@[simp]
theorem mul_lt_mul_right [MulPosStrictMono α] [MulPosReflectLT α] (a0 : 0 < a) :
    b * a < c * a ↔ b < c :=
  @rel_iff_cov α>0 α (fun x y => y * x) (· < ·) _ _ ⟨a, a0⟩ _ _


@[simp]
theorem mul_le_mul_left [PosMulMono α] [PosMulReflectLE α] (a0 : 0 < a) : a * b ≤ a * c ↔ b ≤ c :=
  @rel_iff_cov α>0 α (fun x y => x * y) (· ≤ ·) _ _ ⟨a, a0⟩ _ _


@[simp]
theorem mul_le_mul_right [MulPosMono α] [MulPosReflectLE α] (a0 : 0 < a) : b * a ≤ c * a ↔ b ≤ c :=
  @rel_iff_cov α>0 α (fun x y => y * x) (· ≤ ·) _ _ ⟨a, a0⟩ _ _


alias mul_le_mul_iff_of_pos_left := mul_le_mul_left

alias mul_le_mul_iff_of_pos_right := mul_le_mul_right

alias mul_lt_mul_iff_of_pos_left := mul_lt_mul_left

alias mul_lt_mul_iff_of_pos_right := mul_lt_mul_right


theorem mul_le_mul_of_nonneg [PosMulMono α] [MulPosMono α]
    (h₁ : a ≤ b) (h₂ : c ≤ d) (a0 : 0 ≤ a) (d0 : 0 ≤ d) : a * c ≤ b * d :=
  (mul_le_mul_of_nonneg_left h₂ a0).trans (mul_le_mul_of_nonneg_right h₁ d0)


@[deprecated (since := "2024-07-13")]
alias mul_le_mul_of_le_of_le := mul_le_mul_of_nonneg


theorem mul_le_mul_of_nonneg' [PosMulMono α] [MulPosMono α]
    (h₁ : a ≤ b) (h₂ : c ≤ d) (c0 : 0 ≤ c) (b0 : 0 ≤ b) : a * c ≤ b * d :=
  (mul_le_mul_of_nonneg_right h₁ c0).trans (mul_le_mul_of_nonneg_left h₂ b0)


theorem mul_lt_mul_of_le_of_lt_of_pos_of_nonneg [PosMulStrictMono α] [MulPosMono α]
    (h₁ : a ≤ b) (h₂ : c < d) (a0 : 0 < a) (d0 : 0 ≤ d) : a * c < b * d :=
  (mul_lt_mul_of_pos_left h₂ a0).trans_le (mul_le_mul_of_nonneg_right h₁ d0)


alias mul_lt_mul_of_pos_of_nonneg := mul_lt_mul_of_le_of_lt_of_pos_of_nonneg


theorem mul_lt_mul_of_le_of_lt_of_nonneg_of_pos [PosMulStrictMono α] [MulPosMono α]
    (h₁ : a ≤ b) (h₂ : c < d) (c0 : 0 ≤ c) (b0 : 0 < b) : a * c < b * d :=
  (mul_le_mul_of_nonneg_right h₁ c0).trans_lt (mul_lt_mul_of_pos_left h₂ b0)


alias mul_lt_mul_of_nonneg_of_pos' := mul_lt_mul_of_le_of_lt_of_nonneg_of_pos


@[deprecated (since := "2024-07-13")]
alias mul_lt_mul_of_le_of_le' := mul_lt_mul_of_le_of_lt_of_nonneg_of_pos


theorem mul_lt_mul_of_lt_of_le_of_nonneg_of_pos [PosMulMono α] [MulPosStrictMono α]
    (h₁ : a < b) (h₂ : c ≤ d) (a0 : 0 ≤ a) (d0 : 0 < d) : a * c < b * d :=
  (mul_le_mul_of_nonneg_left h₂ a0).trans_lt (mul_lt_mul_of_pos_right h₁ d0)


alias mul_lt_mul_of_nonneg_of_pos := mul_lt_mul_of_lt_of_le_of_nonneg_of_pos


theorem mul_lt_mul_of_lt_of_le_of_pos_of_nonneg [PosMulMono α] [MulPosStrictMono α]
    (h₁ : a < b) (h₂ : c ≤ d) (c0 : 0 < c) (b0 : 0 ≤ b) : a * c < b * d :=
  (mul_lt_mul_of_pos_right h₁ c0).trans_le (mul_le_mul_of_nonneg_left h₂ b0)


alias mul_lt_mul_of_pos_of_nonneg' := mul_lt_mul_of_lt_of_le_of_pos_of_nonneg


@[deprecated (since := "2024-07-13")]
alias mul_lt_mul_of_le_of_lt' := mul_lt_mul_of_lt_of_le_of_pos_of_nonneg


theorem mul_lt_mul_of_pos [PosMulStrictMono α] [MulPosStrictMono α]
    (h₁ : a < b) (h₂ : c < d) (a0 : 0 < a) (d0 : 0 < d) : a * c < b * d :=
  (mul_lt_mul_of_pos_left h₂ a0).trans (mul_lt_mul_of_pos_right h₁ d0)


@[deprecated (since := "2024-07-13")]
alias mul_lt_mul_of_pos_of_pos := mul_lt_mul_of_pos


theorem mul_lt_mul_of_pos' [PosMulStrictMono α] [MulPosStrictMono α]
    (h₁ : a < b) (h₂ : c < d) (c0 : 0 < c) (b0 : 0 < b) : a * c < b * d :=
  (mul_lt_mul_of_pos_right h₁ c0).trans (mul_lt_mul_of_pos_left h₂ b0)


@[deprecated (since := "2024-07-13")]
alias mul_lt_mul_of_lt_of_lt' := mul_lt_mul_of_pos'


alias mul_le_mul := mul_le_mul_of_nonneg'

alias mul_lt_mul := mul_lt_mul_of_pos_of_nonneg'


alias mul_lt_mul' := mul_lt_mul_of_nonneg_of_pos'


theorem mul_le_of_mul_le_of_nonneg_left [PosMulMono α] (h : a * b ≤ c) (hle : d ≤ b) (a0 : 0 ≤ a) :
    a * d ≤ c :=
  (mul_le_mul_of_nonneg_left hle a0).trans h


theorem mul_lt_of_mul_lt_of_nonneg_left [PosMulMono α] (h : a * b < c) (hle : d ≤ b) (a0 : 0 ≤ a) :
    a * d < c :=
  (mul_le_mul_of_nonneg_left hle a0).trans_lt h


theorem le_mul_of_le_mul_of_nonneg_left [PosMulMono α] (h : a ≤ b * c) (hle : c ≤ d) (b0 : 0 ≤ b) :
    a ≤ b * d :=
  h.trans (mul_le_mul_of_nonneg_left hle b0)


theorem lt_mul_of_lt_mul_of_nonneg_left [PosMulMono α] (h : a < b * c) (hle : c ≤ d) (b0 : 0 ≤ b) :
    a < b * d :=
  h.trans_le (mul_le_mul_of_nonneg_left hle b0)


theorem mul_le_of_mul_le_of_nonneg_right [MulPosMono α] (h : a * b ≤ c) (hle : d ≤ a) (b0 : 0 ≤ b) :
    d * b ≤ c :=
  (mul_le_mul_of_nonneg_right hle b0).trans h


theorem mul_lt_of_mul_lt_of_nonneg_right [MulPosMono α] (h : a * b < c) (hle : d ≤ a) (b0 : 0 ≤ b) :
    d * b < c :=
  (mul_le_mul_of_nonneg_right hle b0).trans_lt h


theorem le_mul_of_le_mul_of_nonneg_right [MulPosMono α] (h : a ≤ b * c) (hle : b ≤ d) (c0 : 0 ≤ c) :
    a ≤ d * c :=
  h.trans (mul_le_mul_of_nonneg_right hle c0)


theorem lt_mul_of_lt_mul_of_nonneg_right [MulPosMono α] (h : a < b * c) (hle : b ≤ d) (c0 : 0 ≤ c) :
    a < d * c :=
  h.trans_le (mul_le_mul_of_nonneg_right hle c0)


instance (priority := 100) PosMulStrictMono.toPosMulReflectLE [PosMulStrictMono α] :
    PosMulReflectLE α :=
  ⟨(covariant_lt_iff_contravariant_le _ _ _).1 CovariantClass.elim⟩

-- see Note [lower instance priority]

instance (priority := 100) MulPosStrictMono.toMulPosReflectLE [MulPosStrictMono α] :
    MulPosReflectLE α :=
  ⟨(covariant_lt_iff_contravariant_le _ _ _).1 CovariantClass.elim⟩


theorem PosMulReflectLE.toPosMulStrictMono [PosMulReflectLE α] : PosMulStrictMono α :=
  ⟨(covariant_lt_iff_contravariant_le _ _ _).2 ContravariantClass.elim⟩


theorem MulPosReflectLE.toMulPosStrictMono [MulPosReflectLE α] : MulPosStrictMono α :=
  ⟨(covariant_lt_iff_contravariant_le _ _ _).2 ContravariantClass.elim⟩


theorem posMulStrictMono_iff_posMulReflectLE : PosMulStrictMono α ↔ PosMulReflectLE α :=
  ⟨@PosMulStrictMono.toPosMulReflectLE _ _ _ _, @PosMulReflectLE.toPosMulStrictMono _ _ _ _⟩


theorem mulPosStrictMono_iff_mulPosReflectLE : MulPosStrictMono α ↔ MulPosReflectLE α :=
  ⟨@MulPosStrictMono.toMulPosReflectLE _ _ _ _, @MulPosReflectLE.toMulPosStrictMono _ _ _ _⟩


theorem PosMulReflectLT.toPosMulMono [PosMulReflectLT α] : PosMulMono α :=
  ⟨(covariant_le_iff_contravariant_lt _ _ _).2 ContravariantClass.elim⟩


theorem MulPosReflectLT.toMulPosMono [MulPosReflectLT α] : MulPosMono α :=
  ⟨(covariant_le_iff_contravariant_lt _ _ _).2 ContravariantClass.elim⟩


theorem PosMulMono.toPosMulReflectLT [PosMulMono α] : PosMulReflectLT α :=
  ⟨(covariant_le_iff_contravariant_lt _ _ _).1 CovariantClass.elim⟩


theorem MulPosMono.toMulPosReflectLT [MulPosMono α] : MulPosReflectLT α :=
  ⟨(covariant_le_iff_contravariant_lt _ _ _).1 CovariantClass.elim⟩

/- TODO: Currently, only one in four of the above are made instances; we could consider making
  both directions of `covariant_le_iff_contravariant_lt` and `covariant_lt_iff_contravariant_le`
  instances, then all of the above become redundant instances, but there are performance issues. -/


theorem posMulMono_iff_posMulReflectLT : PosMulMono α ↔ PosMulReflectLT α :=
  ⟨@PosMulMono.toPosMulReflectLT _ _ _ _, @PosMulReflectLT.toPosMulMono _ _ _ _⟩


theorem mulPosMono_iff_mulPosReflectLT : MulPosMono α ↔ MulPosReflectLT α :=
  ⟨@MulPosMono.toMulPosReflectLT _ _ _ _, @MulPosReflectLT.toMulPosMono _ _ _ _⟩


/-- Assumes left covariance. -/
theorem Left.mul_pos [PosMulStrictMono α] (ha : 0 < a) (hb : 0 < b) : 0 < a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : PosMulStrictMono α
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ LT.lt 0 (HMul.hMul a b)
  -/
  simpa only [mul_zero] using mul_lt_mul_of_pos_left hb ha
  /-
    🎉 no goals
  -/


alias mul_pos := Left.mul_pos


theorem mul_neg_of_pos_of_neg [PosMulStrictMono α] (ha : 0 < a) (hb : b < 0) : a * b < 0 := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : PosMulStrictMono α
    ha : LT.lt 0 a
    hb : LT.lt b 0
    ⊢ LT.lt (HMul.hMul a b) 0
  -/
  simpa only [mul_zero] using mul_lt_mul_of_pos_left hb ha
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_pos_iff_of_pos_left [PosMulStrictMono α] [PosMulReflectLT α] (h : 0 < a) :
                            /-
                              α : Type u_3
                              a b : α
                              inst✝³ : MulZeroClass α
                              inst✝² : Preorder α
                              inst✝¹ : PosMulStrictMono α
                              inst✝ : PosMulReflectLT α
                              h : LT.lt 0 a
                              ⊢ Iff (LT.lt 0 (HMul.hMul a b)) (LT.lt 0 b)
                            -/
    0 < a * b ↔ 0 < b := by simpa using mul_lt_mul_left (b := 0) h
                            /-
                              🎉 no goals
                            -/


/-- Assumes right covariance. -/
theorem Right.mul_pos [MulPosStrictMono α] (ha : 0 < a) (hb : 0 < b) : 0 < a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : MulPosStrictMono α
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ LT.lt 0 (HMul.hMul a b)
  -/
  simpa only [zero_mul] using mul_lt_mul_of_pos_right ha hb
  /-
    🎉 no goals
  -/


theorem mul_neg_of_neg_of_pos [MulPosStrictMono α] (ha : a < 0) (hb : 0 < b) : a * b < 0 := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : MulPosStrictMono α
    ha : LT.lt a 0
    hb : LT.lt 0 b
    ⊢ LT.lt (HMul.hMul a b) 0
  -/
  simpa only [zero_mul] using mul_lt_mul_of_pos_right ha hb
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_pos_iff_of_pos_right [MulPosStrictMono α] [MulPosReflectLT α] (h : 0 < b) :
                            /-
                              α : Type u_3
                              a b : α
                              inst✝³ : MulZeroClass α
                              inst✝² : Preorder α
                              inst✝¹ : MulPosStrictMono α
                              inst✝ : MulPosReflectLT α
                              h : LT.lt 0 b
                              ⊢ Iff (LT.lt 0 (HMul.hMul a b)) (LT.lt 0 a)
                            -/
    0 < a * b ↔ 0 < a := by simpa using mul_lt_mul_right (b := 0) h
                            /-
                              🎉 no goals
                            -/


/-- Assumes left covariance. -/
theorem Left.mul_nonneg [PosMulMono α] (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : PosMulMono α
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le 0 (HMul.hMul a b)
  -/
  simpa only [mul_zero] using mul_le_mul_of_nonneg_left hb ha
  /-
    🎉 no goals
  -/


alias mul_nonneg := Left.mul_nonneg


theorem mul_nonpos_of_nonneg_of_nonpos [PosMulMono α] (ha : 0 ≤ a) (hb : b ≤ 0) : a * b ≤ 0 := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : PosMulMono α
    ha : LE.le 0 a
    hb : LE.le b 0
    ⊢ LE.le (HMul.hMul a b) 0
  -/
  simpa only [mul_zero] using mul_le_mul_of_nonneg_left hb ha
  /-
    🎉 no goals
  -/


/-- Assumes right covariance. -/
theorem Right.mul_nonneg [MulPosMono α] (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : MulPosMono α
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le 0 (HMul.hMul a b)
  -/
  simpa only [zero_mul] using mul_le_mul_of_nonneg_right ha hb
  /-
    🎉 no goals
  -/


theorem mul_nonpos_of_nonpos_of_nonneg [MulPosMono α] (ha : a ≤ 0) (hb : 0 ≤ b) : a * b ≤ 0 := by
  /-
    α : Type u_3
    a b : α
    inst✝² : MulZeroClass α
    inst✝¹ : Preorder α
    inst✝ : MulPosMono α
    ha : LE.le a 0
    hb : LE.le 0 b
    ⊢ LE.le (HMul.hMul a b) 0
  -/
  simpa only [zero_mul] using mul_le_mul_of_nonneg_right ha hb
  /-
    🎉 no goals
  -/


theorem pos_of_mul_pos_right [PosMulReflectLT α] (h : 0 < a * b) (ha : 0 ≤ a) : 0 < b :=
  lt_of_mul_lt_mul_left ((mul_zero a).symm ▸ h : a * 0 < a * b) ha


theorem pos_of_mul_pos_left [MulPosReflectLT α] (h : 0 < a * b) (hb : 0 ≤ b) : 0 < a :=
  lt_of_mul_lt_mul_right ((zero_mul b).symm ▸ h : 0 * b < a * b) hb


theorem pos_iff_pos_of_mul_pos [PosMulReflectLT α] [MulPosReflectLT α] (hab : 0 < a * b) :
    0 < a ↔ 0 < b :=
  ⟨pos_of_mul_pos_right hab ∘ le_of_lt, pos_of_mul_pos_left hab ∘ le_of_lt⟩


/-- Assumes left strict covariance. -/
theorem Left.mul_lt_mul_of_nonneg [PosMulStrictMono α] [MulPosMono α]
    (h₁ : a < b) (h₂ : c < d) (a0 : 0 ≤ a) (c0 : 0 ≤ c) : a * c < b * d :=
  mul_lt_mul_of_le_of_lt_of_nonneg_of_pos h₁.le h₂ c0 (a0.trans_lt h₁)


/-- Assumes right strict covariance. -/
theorem Right.mul_lt_mul_of_nonneg [PosMulMono α] [MulPosStrictMono α]
    (h₁ : a < b) (h₂ : c < d) (a0 : 0 ≤ a) (c0 : 0 ≤ c) : a * c < b * d :=
  mul_lt_mul_of_lt_of_le_of_nonneg_of_pos h₁ h₂.le a0 (c0.trans_lt h₂)


alias mul_lt_mul_of_nonneg := Left.mul_lt_mul_of_nonneg


alias mul_lt_mul'' := Left.mul_lt_mul_of_nonneg

theorem mul_self_le_mul_self [PosMulMono α] [MulPosMono α] (ha : 0 ≤ a) (hab : a ≤ b) :
    a * a ≤ b * b :=
  mul_le_mul hab hab ha <| ha.trans hab


theorem posMulMono_iff_covariant_pos :
    PosMulMono α ↔ CovariantClass α>0 α (fun x y => x * y) (· ≤ ·) :=
  ⟨@PosMulMono.to_covariantClass_pos_mul_le _ _ _ _, fun h =>
    ⟨fun a b c h => by
      /-
        α : Type u_3
        inst✝¹ : MulZeroClass α
        inst✝ : PartialOrder α
        h✝ : CovariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul (↑x)  …
        a : Subtype fun x => LE.le 0 x
        b c : α
        h : LE.le b c
        ⊢ LE.le (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
      -/
      obtain ha | ha := a.prop.eq_or_lt
        /-
          case inl
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : CovariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul (↑x)  …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LE.le b c
          ha : Eq 0 ↑a
          ⊢ LE.le (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
        -/
      · simp [← ha]
        /-
          🎉 no goals
        -/
        /-
          case inr
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : CovariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul (↑x)  …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LE.le b c
          ha : LT.lt 0 ↑a
          ⊢ LE.le (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
        -/
      · exact @CovariantClass.elim α>0 α (fun x y => x * y) (· ≤ ·) _ ⟨_, ha⟩ _ _ h ⟩⟩
        /-
          🎉 no goals
        -/


theorem mulPosMono_iff_covariant_pos :
    MulPosMono α ↔ CovariantClass α>0 α (fun x y => y * x) (· ≤ ·) :=
  ⟨@MulPosMono.to_covariantClass_pos_mul_le _ _ _ _, fun h =>
    ⟨fun a b c h => by
      /-
        α : Type u_3
        inst✝¹ : MulZeroClass α
        inst✝ : PartialOrder α
        h✝ : CovariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul y ↑x) …
        a : Subtype fun x => LE.le 0 x
        b c : α
        h : LE.le b c
        ⊢ LE.le (HMul.hMul b ↑a) (HMul.hMul c ↑a)
      -/
      obtain ha | ha := a.prop.eq_or_lt
        /-
          case inl
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : CovariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul y ↑x) …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LE.le b c
          ha : Eq 0 ↑a
          ⊢ LE.le (HMul.hMul b ↑a) (HMul.hMul c ↑a)
        -/
      · simp [← ha]
        /-
          🎉 no goals
        -/
        /-
          case inr
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : CovariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul y ↑x) …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LE.le b c
          ha : LT.lt 0 ↑a
          ⊢ LE.le (HMul.hMul b ↑a) (HMul.hMul c ↑a)
        -/
      · exact @CovariantClass.elim α>0 α (fun x y => y * x) (· ≤ ·) _ ⟨_, ha⟩ _ _ h ⟩⟩
        /-
          🎉 no goals
        -/


theorem posMulReflectLT_iff_contravariant_pos :
    PosMulReflectLT α ↔ ContravariantClass α>0 α (fun x y => x * y) (· < ·) :=
  ⟨@PosMulReflectLT.to_contravariantClass_pos_mul_lt _ _ _ _, fun h =>
    ⟨fun a b c h => by
      /-
        α : Type u_3
        inst✝¹ : MulZeroClass α
        inst✝ : PartialOrder α
        h✝ : ContravariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul ( …
        a : Subtype fun x => LE.le 0 x
        b c : α
        h : LT.lt (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
        ⊢ LT.lt b c
      -/
      obtain ha | ha := a.prop.eq_or_lt
        /-
          case inl
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : ContravariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul ( …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LT.lt (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
          ha : Eq 0 ↑a
          ⊢ LT.lt b c
        -/
      · simp [← ha] at h
        /-
          🎉 no goals
        -/
        /-
          case inr
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : ContravariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul ( …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LT.lt (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
          ha : LT.lt 0 ↑a
          ⊢ LT.lt b c
        -/
      · exact @ContravariantClass.elim α>0 α (fun x y => x * y) (· < ·) _ ⟨_, ha⟩ _ _ h ⟩⟩
        /-
          🎉 no goals
        -/


theorem mulPosReflectLT_iff_contravariant_pos :
    MulPosReflectLT α ↔ ContravariantClass α>0 α (fun x y => y * x) (· < ·) :=
  ⟨@MulPosReflectLT.to_contravariantClass_pos_mul_lt _ _ _ _, fun h =>
    ⟨fun a b c h => by
      /-
        α : Type u_3
        inst✝¹ : MulZeroClass α
        inst✝ : PartialOrder α
        h✝ : ContravariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul y …
        a : Subtype fun x => LE.le 0 x
        b c : α
        h : LT.lt (HMul.hMul b ↑a) (HMul.hMul c ↑a)
        ⊢ LT.lt b c
      -/
      obtain ha | ha := a.prop.eq_or_lt
        /-
          case inl
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : ContravariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul y …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LT.lt (HMul.hMul b ↑a) (HMul.hMul c ↑a)
          ha : Eq 0 ↑a
          ⊢ LT.lt b c
        -/
      · simp [← ha] at h
        /-
          🎉 no goals
        -/
        /-
          case inr
          α : Type u_3
          inst✝¹ : MulZeroClass α
          inst✝ : PartialOrder α
          h✝ : ContravariantClass (Subtype fun x => LT.lt 0 x) α (fun x y => HMul.hMul y …
          a : Subtype fun x => LE.le 0 x
          b c : α
          h : LT.lt (HMul.hMul b ↑a) (HMul.hMul c ↑a)
          ha : LT.lt 0 ↑a
          ⊢ LT.lt b c
        -/
      · exact @ContravariantClass.elim α>0 α (fun x y => y * x) (· < ·) _ ⟨_, ha⟩ _ _ h ⟩⟩
        /-
          🎉 no goals
        -/

-- Porting note: mathlib3 proofs would look like `StrictMono.monotone <| @CovariantClass.elim ..`
-- but implicit argument handling causes that to break
-- see Note [lower instance priority]

instance (priority := 100) PosMulStrictMono.toPosMulMono [PosMulStrictMono α] : PosMulMono α :=
  posMulMono_iff_covariant_pos.2 (covariantClass_le_of_lt _ _ _)

-- Porting note: mathlib3 proofs would look like `StrictMono.monotone <| @CovariantClass.elim ..`
-- but implicit argument handling causes that to break
-- see Note [lower instance priority]

instance (priority := 100) MulPosStrictMono.toMulPosMono [MulPosStrictMono α] : MulPosMono α :=
  mulPosMono_iff_covariant_pos.2 (covariantClass_le_of_lt _ _ _)

-- see Note [lower instance priority]

instance (priority := 100) PosMulReflectLE.toPosMulReflectLT [PosMulReflectLE α] :
    PosMulReflectLT α :=
  posMulReflectLT_iff_contravariant_pos.2
    ⟨fun a b c h =>
      (le_of_mul_le_mul_of_pos_left h.le a.2).lt_of_ne <| by
        /-
          M₀ : Type u_1
          G₀ : Type u_2
          α : Type u_3
          a✝ b✝ c✝ d : α
          inst✝² : MulZeroClass α
          inst✝¹ : PartialOrder α
          inst✝ : PosMulReflectLE α
          a : Subtype fun x => LT.lt 0 x
          b c : α
          h : LT.lt (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
          ⊢ Ne b c
        -/
        rintro rfl
        /-
          M₀ : Type u_1
          G₀ : Type u_2
          α : Type u_3
          a✝ b✝ c d : α
          inst✝² : MulZeroClass α
          inst✝¹ : PartialOrder α
          inst✝ : PosMulReflectLE α
          a : Subtype fun x => LT.lt 0 x
          b : α
          h : LT.lt (HMul.hMul (↑a) b) (HMul.hMul (↑a) b)
          ⊢ False
        -/
        simp at h⟩
        /-
          🎉 no goals
        -/

-- see Note [lower instance priority]

instance (priority := 100) MulPosReflectLE.toMulPosReflectLT [MulPosReflectLE α] :
    MulPosReflectLT α :=
  mulPosReflectLT_iff_contravariant_pos.2
    ⟨fun a b c h =>
      (le_of_mul_le_mul_of_pos_right h.le a.2).lt_of_ne <| by
        /-
          M₀ : Type u_1
          G₀ : Type u_2
          α : Type u_3
          a✝ b✝ c✝ d : α
          inst✝² : MulZeroClass α
          inst✝¹ : PartialOrder α
          inst✝ : MulPosReflectLE α
          a : Subtype fun x => LT.lt 0 x
          b c : α
          h : LT.lt (HMul.hMul b ↑a) (HMul.hMul c ↑a)
          ⊢ Ne b c
        -/
        rintro rfl
        /-
          M₀ : Type u_1
          G₀ : Type u_2
          α : Type u_3
          a✝ b✝ c d : α
          inst✝² : MulZeroClass α
          inst✝¹ : PartialOrder α
          inst✝ : MulPosReflectLE α
          a : Subtype fun x => LT.lt 0 x
          b : α
          h : LT.lt (HMul.hMul b ↑a) (HMul.hMul b ↑a)
          ⊢ False
        -/
        simp at h⟩
        /-
          🎉 no goals
        -/


theorem mul_left_cancel_iff_of_pos [PosMulReflectLE α] (a0 : 0 < a) : a * b = a * c ↔ b = c :=
  ⟨fun h => (le_of_mul_le_mul_of_pos_left h.le a0).antisymm <|
    le_of_mul_le_mul_of_pos_left h.ge a0, congr_arg _⟩


theorem mul_right_cancel_iff_of_pos [MulPosReflectLE α] (b0 : 0 < b) : a * b = c * b ↔ a = c :=
  ⟨fun h => (le_of_mul_le_mul_of_pos_right h.le b0).antisymm <|
    le_of_mul_le_mul_of_pos_right h.ge b0, congr_arg (· * b)⟩


theorem mul_eq_mul_iff_eq_and_eq_of_pos [PosMulStrictMono α] [MulPosStrictMono α]
    (hab : a ≤ b) (hcd : c ≤ d) (a0 : 0 < a) (d0 : 0 < d) :
    a * c = b * d ↔ a = b ∧ c = d := by
  /-
    α : Type u_3
    a b c d : α
    inst✝³ : MulZeroClass α
    inst✝² : PartialOrder α
    inst✝¹ : PosMulStrictMono α
    inst✝ : MulPosStrictMono α
    hab : LE.le a b
    hcd : LE.le c d
    a0 : LT.lt 0 a
    d0 : LT.lt 0 d
    ⊢ Iff (Eq (HMul.hMul a c) (HMul.hMul b d)) (And (Eq a b) (Eq c d))
  -/
  refine ⟨fun h ↦ ?_, by rintro ⟨rfl, rfl⟩; rfl⟩
  /-
    α : Type u_3
    a b c d : α
    inst✝³ : MulZeroClass α
    inst✝² : PartialOrder α
    inst✝¹ : PosMulStrictMono α
    inst✝ : MulPosStrictMono α
    hab : LE.le a b
    hcd : LE.le c d
    a0 : LT.lt 0 a
    d0 : LT.lt 0 d
    h : Eq (HMul.hMul a c) (HMul.hMul b d)
    ⊢ And (Eq a b) (Eq c d)
  -/
  simp only [eq_iff_le_not_lt, hab, hcd, true_and]
  /-
    α : Type u_3
    a b c d : α
    inst✝³ : MulZeroClass α
    inst✝² : PartialOrder α
    inst✝¹ : PosMulStrictMono α
    inst✝ : MulPosStrictMono α
    hab : LE.le a b
    hcd : LE.le c d
    a0 : LT.lt 0 a
    d0 : LT.lt 0 d
    h : Eq (HMul.hMul a c) (HMul.hMul b d)
    ⊢ And (Not (LT.lt a b)) (Not (LT.lt c d))
  -/
  refine ⟨fun hab ↦ h.not_lt ?_, fun hcd ↦ h.not_lt ?_⟩
    /-
      case refine_1
      α : Type u_3
      a b c d : α
      inst✝³ : MulZeroClass α
      inst✝² : PartialOrder α
      inst✝¹ : PosMulStrictMono α
      inst✝ : MulPosStrictMono α
      hab✝ : LE.le a b
      hcd : LE.le c d
      a0 : LT.lt 0 a
      d0 : LT.lt 0 d
      h : Eq (HMul.hMul a c) (HMul.hMul b d)
      hab : LT.lt a b
      ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
    -/
  · exact (mul_le_mul_of_nonneg_left hcd a0.le).trans_lt (mul_lt_mul_of_pos_right hab d0)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      a b c d : α
      inst✝³ : MulZeroClass α
      inst✝² : PartialOrder α
      inst✝¹ : PosMulStrictMono α
      inst✝ : MulPosStrictMono α
      hab : LE.le a b
      hcd✝ : LE.le c d
      a0 : LT.lt 0 a
      d0 : LT.lt 0 d
      h : Eq (HMul.hMul a c) (HMul.hMul b d)
      hcd : LT.lt c d
      ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
    -/
  · exact (mul_lt_mul_of_pos_left hcd a0).trans_le (mul_le_mul_of_nonneg_right hab d0.le)
    /-
      🎉 no goals
    -/


theorem mul_eq_mul_iff_eq_and_eq_of_pos' [PosMulStrictMono α] [MulPosStrictMono α]
    (hab : a ≤ b) (hcd : c ≤ d) (b0 : 0 < b) (c0 : 0 < c) :
    a * c = b * d ↔ a = b ∧ c = d := by
  /-
    α : Type u_3
    a b c d : α
    inst✝³ : MulZeroClass α
    inst✝² : PartialOrder α
    inst✝¹ : PosMulStrictMono α
    inst✝ : MulPosStrictMono α
    hab : LE.le a b
    hcd : LE.le c d
    b0 : LT.lt 0 b
    c0 : LT.lt 0 c
    ⊢ Iff (Eq (HMul.hMul a c) (HMul.hMul b d)) (And (Eq a b) (Eq c d))
  -/
  refine ⟨fun h ↦ ?_, by rintro ⟨rfl, rfl⟩; rfl⟩
  /-
    α : Type u_3
    a b c d : α
    inst✝³ : MulZeroClass α
    inst✝² : PartialOrder α
    inst✝¹ : PosMulStrictMono α
    inst✝ : MulPosStrictMono α
    hab : LE.le a b
    hcd : LE.le c d
    b0 : LT.lt 0 b
    c0 : LT.lt 0 c
    h : Eq (HMul.hMul a c) (HMul.hMul b d)
    ⊢ And (Eq a b) (Eq c d)
  -/
  simp only [eq_iff_le_not_lt, hab, hcd, true_and]
  /-
    α : Type u_3
    a b c d : α
    inst✝³ : MulZeroClass α
    inst✝² : PartialOrder α
    inst✝¹ : PosMulStrictMono α
    inst✝ : MulPosStrictMono α
    hab : LE.le a b
    hcd : LE.le c d
    b0 : LT.lt 0 b
    c0 : LT.lt 0 c
    h : Eq (HMul.hMul a c) (HMul.hMul b d)
    ⊢ And (Not (LT.lt a b)) (Not (LT.lt c d))
  -/
  refine ⟨fun hab ↦ h.not_lt ?_, fun hcd ↦ h.not_lt ?_⟩
    /-
      case refine_1
      α : Type u_3
      a b c d : α
      inst✝³ : MulZeroClass α
      inst✝² : PartialOrder α
      inst✝¹ : PosMulStrictMono α
      inst✝ : MulPosStrictMono α
      hab✝ : LE.le a b
      hcd : LE.le c d
      b0 : LT.lt 0 b
      c0 : LT.lt 0 c
      h : Eq (HMul.hMul a c) (HMul.hMul b d)
      hab : LT.lt a b
      ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
    -/
  · exact (mul_lt_mul_of_pos_right hab c0).trans_le (mul_le_mul_of_nonneg_left hcd b0.le)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      a b c d : α
      inst✝³ : MulZeroClass α
      inst✝² : PartialOrder α
      inst✝¹ : PosMulStrictMono α
      inst✝ : MulPosStrictMono α
      hab : LE.le a b
      hcd✝ : LE.le c d
      b0 : LT.lt 0 b
      c0 : LT.lt 0 c
      h : Eq (HMul.hMul a c) (HMul.hMul b d)
      hcd : LT.lt c d
      ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
    -/
  · exact (mul_le_mul_of_nonneg_right hab c0.le).trans_lt (mul_lt_mul_of_pos_left hcd b0)
    /-
      🎉 no goals
    -/


theorem pos_and_pos_or_neg_and_neg_of_mul_pos [PosMulMono α] [MulPosMono α] (hab : 0 < a * b) :
    0 < a ∧ 0 < b ∨ a < 0 ∧ b < 0 := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulZeroClass α
    inst✝² : LinearOrder α
    inst✝¹ : PosMulMono α
    inst✝ : MulPosMono α
    hab : LT.lt 0 (HMul.hMul a b)
    ⊢ Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))
  -/
  rcases lt_trichotomy a 0 with (ha | rfl | ha)
    /-
      case inl
      α : Type u_3
      a b : α
      inst✝³ : MulZeroClass α
      inst✝² : LinearOrder α
      inst✝¹ : PosMulMono α
      inst✝ : MulPosMono α
      hab : LT.lt 0 (HMul.hMul a b)
      ha : LT.lt a 0
      ⊢ Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))
    -/
  · refine Or.inr ⟨ha, lt_imp_lt_of_le_imp_le (fun hb => ?_) hab⟩
    /-
      case inl
      α : Type u_3
      a b : α
      inst✝³ : MulZeroClass α
      inst✝² : LinearOrder α
      inst✝¹ : PosMulMono α
      inst✝ : MulPosMono α
      hab : LT.lt 0 (HMul.hMul a b)
      ha : LT.lt a 0
      hb : LE.le 0 b
      ⊢ LE.le (HMul.hMul a b) 0
    -/
    exact mul_nonpos_of_nonpos_of_nonneg ha.le hb
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_3
      b : α
      inst✝³ : MulZeroClass α
      inst✝² : LinearOrder α
      inst✝¹ : PosMulMono α
      inst✝ : MulPosMono α
      hab : LT.lt 0 (HMul.hMul 0 b)
      ⊢ Or (And (LT.lt 0 0) (LT.lt 0 b)) (And (LT.lt 0 0) (LT.lt b 0))
    -/
  · rw [zero_mul] at hab
    /-
      case inr.inl
      α : Type u_3
      b : α
      inst✝³ : MulZeroClass α
      inst✝² : LinearOrder α
      inst✝¹ : PosMulMono α
      inst✝ : MulPosMono α
      hab : LT.lt 0 0
      ⊢ Or (And (LT.lt 0 0) (LT.lt 0 b)) (And (LT.lt 0 0) (LT.lt b 0))
    -/
    exact hab.false.elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_3
      a b : α
      inst✝³ : MulZeroClass α
      inst✝² : LinearOrder α
      inst✝¹ : PosMulMono α
      inst✝ : MulPosMono α
      hab : LT.lt 0 (HMul.hMul a b)
      ha : LT.lt 0 a
      ⊢ Or (And (LT.lt 0 a) (LT.lt 0 b)) (And (LT.lt a 0) (LT.lt b 0))
    -/
  · refine Or.inl ⟨ha, lt_imp_lt_of_le_imp_le (fun hb => ?_) hab⟩
    /-
      case inr.inr
      α : Type u_3
      a b : α
      inst✝³ : MulZeroClass α
      inst✝² : LinearOrder α
      inst✝¹ : PosMulMono α
      inst✝ : MulPosMono α
      hab : LT.lt 0 (HMul.hMul a b)
      ha : LT.lt 0 a
      hb : LE.le b 0
      ⊢ LE.le (HMul.hMul a b) 0
    -/
    exact mul_nonpos_of_nonneg_of_nonpos ha.le hb
    /-
      🎉 no goals
    -/


theorem neg_of_mul_pos_right [PosMulMono α] [MulPosMono α] (h : 0 < a * b) (ha : a ≤ 0) : b < 0 :=
  ((pos_and_pos_or_neg_and_neg_of_mul_pos h).resolve_left fun h => h.1.not_le ha).2


theorem neg_of_mul_pos_left [PosMulMono α] [MulPosMono α] (h : 0 < a * b) (ha : b ≤ 0) : a < 0 :=
  ((pos_and_pos_or_neg_and_neg_of_mul_pos h).resolve_left fun h => h.2.not_le ha).1


theorem neg_iff_neg_of_mul_pos [PosMulMono α] [MulPosMono α] (hab : 0 < a * b) : a < 0 ↔ b < 0 :=
  ⟨neg_of_mul_pos_right hab ∘ le_of_lt, neg_of_mul_pos_left hab ∘ le_of_lt⟩


theorem Left.neg_of_mul_neg_right [PosMulMono α] (h : a * b < 0) (a0 : 0 ≤ a) : b < 0 :=
  lt_of_not_ge fun b0 : b ≥ 0 => (Left.mul_nonneg a0 b0).not_lt h


alias neg_of_mul_neg_right := Left.neg_of_mul_neg_right


theorem Right.neg_of_mul_neg_right [MulPosMono α] (h : a * b < 0) (a0 : 0 ≤ a) : b < 0 :=
  lt_of_not_ge fun b0 : b ≥ 0 => (Right.mul_nonneg a0 b0).not_lt h


theorem Left.neg_of_mul_neg_left [PosMulMono α] (h : a * b < 0) (b0 : 0 ≤ b) : a < 0 :=
  lt_of_not_ge fun a0 : a ≥ 0 => (Left.mul_nonneg a0 b0).not_lt h


alias neg_of_mul_neg_left := Left.neg_of_mul_neg_left


theorem Right.neg_of_mul_neg_left [MulPosMono α] (h : a * b < 0) (b0 : 0 ≤ b) : a < 0 :=
  lt_of_not_ge fun a0 : a ≥ 0 => (Right.mul_nonneg a0 b0).not_lt h


lemma one_lt_of_lt_mul_left₀ [PosMulReflectLT α] (ha : 0 ≤ a) (h : a < a * b) : 1 < b :=
                            /-
                              α : Type u_3
                              a b : α
                              inst✝³ : MulOneClass α
                              inst✝² : Zero α
                              inst✝¹ : Preorder α
                              inst✝ : PosMulReflectLT α
                              ha : LE.le 0 a
                              h : LT.lt a (HMul.hMul a b)
                              ⊢ LT.lt (HMul.hMul a 1) (HMul.hMul a b)
                            -/
  lt_of_mul_lt_mul_left (by simpa) ha
                            /-
                              🎉 no goals
                            -/


lemma one_lt_of_lt_mul_right₀ [MulPosReflectLT α] (hb : 0 ≤ b) (h : b < a * b) : 1 < a :=
                             /-
                               α : Type u_3
                               a b : α
                               inst✝³ : MulOneClass α
                               inst✝² : Zero α
                               inst✝¹ : Preorder α
                               inst✝ : MulPosReflectLT α
                               hb : LE.le 0 b
                               h : LT.lt b (HMul.hMul a b)
                               ⊢ LT.lt (HMul.hMul 1 b) (HMul.hMul a b)
                             -/
  lt_of_mul_lt_mul_right (by simpa) hb
                             /-
                               🎉 no goals
                             -/


lemma one_le_of_le_mul_left₀ [PosMulReflectLE α] (ha : 0 < a) (h : a ≤ a * b) : 1 ≤ b :=
                            /-
                              α : Type u_3
                              a b : α
                              inst✝³ : MulOneClass α
                              inst✝² : Zero α
                              inst✝¹ : Preorder α
                              inst✝ : PosMulReflectLE α
                              ha : LT.lt 0 a
                              h : LE.le a (HMul.hMul a b)
                              ⊢ LE.le (HMul.hMul a 1) (HMul.hMul a b)
                            -/
  le_of_mul_le_mul_left (by simpa) ha
                            /-
                              🎉 no goals
                            -/


lemma one_le_of_le_mul_right₀ [MulPosReflectLE α] (hb : 0 < b) (h : b ≤ a * b) : 1 ≤ a :=
                             /-
                               α : Type u_3
                               a b : α
                               inst✝³ : MulOneClass α
                               inst✝² : Zero α
                               inst✝¹ : Preorder α
                               inst✝ : MulPosReflectLE α
                               hb : LT.lt 0 b
                               h : LE.le b (HMul.hMul a b)
                               ⊢ LE.le (HMul.hMul 1 b) (HMul.hMul a b)
                             -/
  le_of_mul_le_mul_right (by simpa) hb
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma le_mul_iff_one_le_right [PosMulMono α] [PosMulReflectLE α] (a0 : 0 < a) : a ≤ a * b ↔ 1 ≤ b :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : PosMulMono α
                  inst✝ : PosMulReflectLE α
                  a0 : LT.lt 0 a
                  ⊢ Iff (LE.le a (HMul.hMul a b)) (LE.le (HMul.hMul a 1) (HMul.hMul a b))
                -/
  Iff.trans (by rw [mul_one]) (mul_le_mul_left a0)
                /-
                  🎉 no goals
                -/


@[simp]
theorem lt_mul_iff_one_lt_right [PosMulStrictMono α] [PosMulReflectLT α] (a0 : 0 < a) :
    a < a * b ↔ 1 < b :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : PosMulStrictMono α
                  inst✝ : PosMulReflectLT α
                  a0 : LT.lt 0 a
                  ⊢ Iff (LT.lt a (HMul.hMul a b)) (LT.lt (HMul.hMul a 1) (HMul.hMul a b))
                -/
  Iff.trans (by rw [mul_one]) (mul_lt_mul_left a0)
                /-
                  🎉 no goals
                -/


@[simp]
lemma mul_le_iff_le_one_right [PosMulMono α] [PosMulReflectLE α] (a0 : 0 < a) : a * b ≤ a ↔ b ≤ 1 :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : PosMulMono α
                  inst✝ : PosMulReflectLE α
                  a0 : LT.lt 0 a
                  ⊢ Iff (LE.le (HMul.hMul a b) a) (LE.le (HMul.hMul a b) (HMul.hMul a 1))
                -/
  Iff.trans (by rw [mul_one]) (mul_le_mul_left a0)
                /-
                  🎉 no goals
                -/


@[simp]
theorem mul_lt_iff_lt_one_right [PosMulStrictMono α] [PosMulReflectLT α] (a0 : 0 < a) :
    a * b < a ↔ b < 1 :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : PosMulStrictMono α
                  inst✝ : PosMulReflectLT α
                  a0 : LT.lt 0 a
                  ⊢ Iff (LT.lt (HMul.hMul a b) a) (LT.lt (HMul.hMul a b) (HMul.hMul a 1))
                -/
  Iff.trans (by rw [mul_one]) (mul_lt_mul_left a0)
                /-
                  🎉 no goals
                -/


@[simp]
lemma le_mul_iff_one_le_left [MulPosMono α] [MulPosReflectLE α] (a0 : 0 < a) : a ≤ b * a ↔ 1 ≤ b :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : MulPosMono α
                  inst✝ : MulPosReflectLE α
                  a0 : LT.lt 0 a
                  ⊢ Iff (LE.le a (HMul.hMul b a)) (LE.le (HMul.hMul 1 a) (HMul.hMul b a))
                -/
  Iff.trans (by rw [one_mul]) (mul_le_mul_right a0)
                /-
                  🎉 no goals
                -/


@[simp]
theorem lt_mul_iff_one_lt_left [MulPosStrictMono α] [MulPosReflectLT α] (a0 : 0 < a) :
    a < b * a ↔ 1 < b :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : MulPosStrictMono α
                  inst✝ : MulPosReflectLT α
                  a0 : LT.lt 0 a
                  ⊢ Iff (LT.lt a (HMul.hMul b a)) (LT.lt (HMul.hMul 1 a) (HMul.hMul b a))
                -/
  Iff.trans (by rw [one_mul]) (mul_lt_mul_right a0)
                /-
                  🎉 no goals
                -/


@[simp]
lemma mul_le_iff_le_one_left [MulPosMono α] [MulPosReflectLE α] (b0 : 0 < b) : a * b ≤ b ↔ a ≤ 1 :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : MulPosMono α
                  inst✝ : MulPosReflectLE α
                  b0 : LT.lt 0 b
                  ⊢ Iff (LE.le (HMul.hMul a b) b) (LE.le (HMul.hMul a b) (HMul.hMul 1 b))
                -/
  Iff.trans (by rw [one_mul]) (mul_le_mul_right b0)
                /-
                  🎉 no goals
                -/


@[simp]
theorem mul_lt_iff_lt_one_left [MulPosStrictMono α] [MulPosReflectLT α] (b0 : 0 < b) :
    a * b < b ↔ a < 1 :=
                /-
                  α : Type u_3
                  a b : α
                  inst✝⁴ : MulOneClass α
                  inst✝³ : Zero α
                  inst✝² : Preorder α
                  inst✝¹ : MulPosStrictMono α
                  inst✝ : MulPosReflectLT α
                  b0 : LT.lt 0 b
                  ⊢ Iff (LT.lt (HMul.hMul a b) b) (LT.lt (HMul.hMul a b) (HMul.hMul 1 b))
                -/
  Iff.trans (by rw [one_mul]) (mul_lt_mul_right b0)
                /-
                  🎉 no goals
                -/


theorem mul_le_of_le_one_left [MulPosMono α] (hb : 0 ≤ b) (h : a ≤ 1) : a * b ≤ b := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : MulPosMono α
    hb : LE.le 0 b
    h : LE.le a 1
    ⊢ LE.le (HMul.hMul a b) b
  -/
  simpa only [one_mul] using mul_le_mul_of_nonneg_right h hb
  /-
    🎉 no goals
  -/


theorem le_mul_of_one_le_left [MulPosMono α] (hb : 0 ≤ b) (h : 1 ≤ a) : b ≤ a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : MulPosMono α
    hb : LE.le 0 b
    h : LE.le 1 a
    ⊢ LE.le b (HMul.hMul a b)
  -/
  simpa only [one_mul] using mul_le_mul_of_nonneg_right h hb
  /-
    🎉 no goals
  -/


theorem mul_le_of_le_one_right [PosMulMono α] (ha : 0 ≤ a) (h : b ≤ 1) : a * b ≤ a := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : PosMulMono α
    ha : LE.le 0 a
    h : LE.le b 1
    ⊢ LE.le (HMul.hMul a b) a
  -/
  simpa only [mul_one] using mul_le_mul_of_nonneg_left h ha
  /-
    🎉 no goals
  -/


theorem le_mul_of_one_le_right [PosMulMono α] (ha : 0 ≤ a) (h : 1 ≤ b) : a ≤ a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : PosMulMono α
    ha : LE.le 0 a
    h : LE.le 1 b
    ⊢ LE.le a (HMul.hMul a b)
  -/
  simpa only [mul_one] using mul_le_mul_of_nonneg_left h ha
  /-
    🎉 no goals
  -/


theorem mul_lt_of_lt_one_left [MulPosStrictMono α] (hb : 0 < b) (h : a < 1) : a * b < b := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : MulPosStrictMono α
    hb : LT.lt 0 b
    h : LT.lt a 1
    ⊢ LT.lt (HMul.hMul a b) b
  -/
  simpa only [one_mul] using mul_lt_mul_of_pos_right h hb
  /-
    🎉 no goals
  -/


theorem lt_mul_of_one_lt_left [MulPosStrictMono α] (hb : 0 < b) (h : 1 < a) : b < a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : MulPosStrictMono α
    hb : LT.lt 0 b
    h : LT.lt 1 a
    ⊢ LT.lt b (HMul.hMul a b)
  -/
  simpa only [one_mul] using mul_lt_mul_of_pos_right h hb
  /-
    🎉 no goals
  -/


theorem mul_lt_of_lt_one_right [PosMulStrictMono α] (ha : 0 < a) (h : b < 1) : a * b < a := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : PosMulStrictMono α
    ha : LT.lt 0 a
    h : LT.lt b 1
    ⊢ LT.lt (HMul.hMul a b) a
  -/
  simpa only [mul_one] using mul_lt_mul_of_pos_left h ha
  /-
    🎉 no goals
  -/


theorem lt_mul_of_one_lt_right [PosMulStrictMono α] (ha : 0 < a) (h : 1 < b) : a < a * b := by
  /-
    α : Type u_3
    a b : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : Preorder α
    inst✝ : PosMulStrictMono α
    ha : LT.lt 0 a
    h : LT.lt 1 b
    ⊢ LT.lt a (HMul.hMul a b)
  -/
  simpa only [mul_one] using mul_lt_mul_of_pos_left h ha
  /-
    🎉 no goals
  -/


theorem mul_le_of_le_of_le_one_of_nonneg [PosMulMono α] (h : b ≤ c) (ha : a ≤ 1) (hb : 0 ≤ b) :
    b * a ≤ c :=
  (mul_le_of_le_one_right hb ha).trans h


theorem mul_lt_of_le_of_lt_one_of_pos [PosMulStrictMono α] (bc : b ≤ c) (ha : a < 1) (b0 : 0 < b) :
    b * a < c :=
  (mul_lt_of_lt_one_right b0 ha).trans_le bc


theorem mul_lt_of_lt_of_le_one_of_nonneg [PosMulMono α] (h : b < c) (ha : a ≤ 1) (hb : 0 ≤ b) :
    b * a < c :=
  (mul_le_of_le_one_right hb ha).trans_lt h


/-- Assumes left covariance. -/
theorem Left.mul_le_one_of_le_of_le [PosMulMono α] (ha : a ≤ 1) (hb : b ≤ 1) (a0 : 0 ≤ a) :
    a * b ≤ 1 :=
  mul_le_of_le_of_le_one_of_nonneg ha hb a0


/-- Assumes left covariance. -/
theorem Left.mul_lt_of_le_of_lt_one_of_pos [PosMulStrictMono α] (ha : a ≤ 1) (hb : b < 1)
    (a0 : 0 < a) : a * b < 1 :=
  _root_.mul_lt_of_le_of_lt_one_of_pos ha hb a0


/-- Assumes left covariance. -/
theorem Left.mul_lt_of_lt_of_le_one_of_nonneg [PosMulMono α] (ha : a < 1) (hb : b ≤ 1)
    (a0 : 0 ≤ a) : a * b < 1 :=
  _root_.mul_lt_of_lt_of_le_one_of_nonneg ha hb a0


theorem mul_le_of_le_of_le_one' [PosMulMono α] [MulPosMono α] (bc : b ≤ c) (ha : a ≤ 1) (a0 : 0 ≤ a)
    (c0 : 0 ≤ c) : b * a ≤ c :=
  (mul_le_mul_of_nonneg_right bc a0).trans <| mul_le_of_le_one_right c0 ha


theorem mul_lt_of_lt_of_le_one' [PosMulMono α] [MulPosStrictMono α] (bc : b < c) (ha : a ≤ 1)
    (a0 : 0 < a) (c0 : 0 ≤ c) : b * a < c :=
  (mul_lt_mul_of_pos_right bc a0).trans_le <| mul_le_of_le_one_right c0 ha


theorem mul_lt_of_le_of_lt_one' [PosMulStrictMono α] [MulPosMono α] (bc : b ≤ c) (ha : a < 1)
    (a0 : 0 ≤ a) (c0 : 0 < c) : b * a < c :=
  (mul_le_mul_of_nonneg_right bc a0).trans_lt <| mul_lt_of_lt_one_right c0 ha


theorem mul_lt_of_lt_of_lt_one_of_pos [PosMulMono α] [MulPosStrictMono α] (bc : b < c) (ha : a ≤ 1)
    (a0 : 0 < a) (c0 : 0 ≤ c) : b * a < c :=
  (mul_lt_mul_of_pos_right bc a0).trans_le <| mul_le_of_le_one_right c0 ha


theorem le_mul_of_le_of_one_le_of_nonneg [PosMulMono α] (h : b ≤ c) (ha : 1 ≤ a) (hc : 0 ≤ c) :
    b ≤ c * a :=
  h.trans <| le_mul_of_one_le_right hc ha


theorem lt_mul_of_le_of_one_lt_of_pos [PosMulStrictMono α] (bc : b ≤ c) (ha : 1 < a) (c0 : 0 < c) :
    b < c * a :=
  bc.trans_lt <| lt_mul_of_one_lt_right c0 ha


theorem lt_mul_of_lt_of_one_le_of_nonneg [PosMulMono α] (h : b < c) (ha : 1 ≤ a) (hc : 0 ≤ c) :
    b < c * a :=
  h.trans_le <| le_mul_of_one_le_right hc ha


/-- Assumes left covariance. -/
theorem Left.one_le_mul_of_le_of_le [PosMulMono α] (ha : 1 ≤ a) (hb : 1 ≤ b) (a0 : 0 ≤ a) :
    1 ≤ a * b :=
  le_mul_of_le_of_one_le_of_nonneg ha hb a0


/-- Assumes left covariance. -/
theorem Left.one_lt_mul_of_le_of_lt_of_pos [PosMulStrictMono α] (ha : 1 ≤ a) (hb : 1 < b)
    (a0 : 0 < a) : 1 < a * b :=
  lt_mul_of_le_of_one_lt_of_pos ha hb a0


/-- Assumes left covariance. -/
theorem Left.lt_mul_of_lt_of_one_le_of_nonneg [PosMulMono α] (ha : 1 < a) (hb : 1 ≤ b)
    (a0 : 0 ≤ a) : 1 < a * b :=
  _root_.lt_mul_of_lt_of_one_le_of_nonneg ha hb a0


theorem le_mul_of_le_of_one_le' [PosMulMono α] [MulPosMono α] (bc : b ≤ c) (ha : 1 ≤ a)
    (a0 : 0 ≤ a) (b0 : 0 ≤ b) : b ≤ c * a :=
  (le_mul_of_one_le_right b0 ha).trans <| mul_le_mul_of_nonneg_right bc a0


theorem lt_mul_of_le_of_one_lt' [PosMulStrictMono α] [MulPosMono α] (bc : b ≤ c) (ha : 1 < a)
    (a0 : 0 ≤ a) (b0 : 0 < b) : b < c * a :=
  (lt_mul_of_one_lt_right b0 ha).trans_le <| mul_le_mul_of_nonneg_right bc a0


theorem lt_mul_of_lt_of_one_le' [PosMulMono α] [MulPosStrictMono α] (bc : b < c) (ha : 1 ≤ a)
    (a0 : 0 < a) (b0 : 0 ≤ b) : b < c * a :=
  (le_mul_of_one_le_right b0 ha).trans_lt <| mul_lt_mul_of_pos_right bc a0


theorem lt_mul_of_lt_of_one_lt_of_pos [PosMulStrictMono α] [MulPosStrictMono α] (bc : b < c)
    (ha : 1 < a) (a0 : 0 < a) (b0 : 0 < b) : b < c * a :=
  (lt_mul_of_one_lt_right b0 ha).trans <| mul_lt_mul_of_pos_right bc a0


theorem mul_le_of_le_one_of_le_of_nonneg [MulPosMono α] (ha : a ≤ 1) (h : b ≤ c) (hb : 0 ≤ b) :
    a * b ≤ c :=
  (mul_le_of_le_one_left hb ha).trans h


theorem mul_lt_of_lt_one_of_le_of_pos [MulPosStrictMono α] (ha : a < 1) (h : b ≤ c) (hb : 0 < b) :
    a * b < c :=
  (mul_lt_of_lt_one_left hb ha).trans_le h


theorem mul_lt_of_le_one_of_lt_of_nonneg [MulPosMono α] (ha : a ≤ 1) (h : b < c) (hb : 0 ≤ b) :
    a * b < c :=
  (mul_le_of_le_one_left hb ha).trans_lt h


/-- Assumes right covariance. -/
theorem Right.mul_lt_one_of_lt_of_le_of_pos [MulPosStrictMono α] (ha : a < 1) (hb : b ≤ 1)
    (b0 : 0 < b) : a * b < 1 :=
  mul_lt_of_lt_one_of_le_of_pos ha hb b0


/-- Assumes right covariance. -/
theorem Right.mul_lt_one_of_le_of_lt_of_nonneg [MulPosMono α] (ha : a ≤ 1) (hb : b < 1)
    (b0 : 0 ≤ b) : a * b < 1 :=
  mul_lt_of_le_one_of_lt_of_nonneg ha hb b0


theorem mul_lt_of_lt_one_of_lt_of_pos [PosMulStrictMono α] [MulPosStrictMono α] (ha : a < 1)
    (bc : b < c) (a0 : 0 < a) (c0 : 0 < c) : a * b < c :=
  (mul_lt_mul_of_pos_left bc a0).trans <| mul_lt_of_lt_one_left c0 ha


/-- Assumes right covariance. -/
theorem Right.mul_le_one_of_le_of_le [MulPosMono α] (ha : a ≤ 1) (hb : b ≤ 1) (b0 : 0 ≤ b) :
    a * b ≤ 1 :=
  mul_le_of_le_one_of_le_of_nonneg ha hb b0


theorem mul_le_of_le_one_of_le' [PosMulMono α] [MulPosMono α] (ha : a ≤ 1) (bc : b ≤ c) (a0 : 0 ≤ a)
    (c0 : 0 ≤ c) : a * b ≤ c :=
  (mul_le_mul_of_nonneg_left bc a0).trans <| mul_le_of_le_one_left c0 ha


theorem mul_lt_of_lt_one_of_le' [PosMulMono α] [MulPosStrictMono α] (ha : a < 1) (bc : b ≤ c)
    (a0 : 0 ≤ a) (c0 : 0 < c) : a * b < c :=
  (mul_le_mul_of_nonneg_left bc a0).trans_lt <| mul_lt_of_lt_one_left c0 ha


theorem mul_lt_of_le_one_of_lt' [PosMulStrictMono α] [MulPosMono α] (ha : a ≤ 1) (bc : b < c)
    (a0 : 0 < a) (c0 : 0 ≤ c) : a * b < c :=
  (mul_lt_mul_of_pos_left bc a0).trans_le <| mul_le_of_le_one_left c0 ha


theorem lt_mul_of_one_lt_of_le_of_pos [MulPosStrictMono α] (ha : 1 < a) (h : b ≤ c) (hc : 0 < c) :
    b < a * c :=
  h.trans_lt <| lt_mul_of_one_lt_left hc ha


theorem lt_mul_of_one_le_of_lt_of_nonneg [MulPosMono α] (ha : 1 ≤ a) (h : b < c) (hc : 0 ≤ c) :
    b < a * c :=
  h.trans_le <| le_mul_of_one_le_left hc ha


theorem lt_mul_of_one_lt_of_lt_of_pos [MulPosStrictMono α] (ha : 1 < a) (h : b < c) (hc : 0 < c) :
    b < a * c :=
  h.trans <| lt_mul_of_one_lt_left hc ha


/-- Assumes right covariance. -/
theorem Right.one_lt_mul_of_lt_of_le_of_pos [MulPosStrictMono α] (ha : 1 < a) (hb : 1 ≤ b)
    (b0 : 0 < b) : 1 < a * b :=
  lt_mul_of_one_lt_of_le_of_pos ha hb b0


/-- Assumes right covariance. -/
theorem Right.one_lt_mul_of_le_of_lt_of_nonneg [MulPosMono α] (ha : 1 ≤ a) (hb : 1 < b)
    (b0 : 0 ≤ b) : 1 < a * b :=
  lt_mul_of_one_le_of_lt_of_nonneg ha hb b0


/-- Assumes right covariance. -/
theorem Right.one_lt_mul_of_lt_of_lt [MulPosStrictMono α] (ha : 1 < a) (hb : 1 < b) (b0 : 0 < b) :
    1 < a * b :=
  lt_mul_of_one_lt_of_lt_of_pos ha hb b0


theorem lt_mul_of_one_lt_of_lt_of_nonneg [MulPosMono α] (ha : 1 ≤ a) (h : b < c) (hc : 0 ≤ c) :
    b < a * c :=
  h.trans_le <| le_mul_of_one_le_left hc ha


theorem lt_of_mul_lt_of_one_le_of_nonneg_left [PosMulMono α] (h : a * b < c) (hle : 1 ≤ b)
    (ha : 0 ≤ a) : a < c :=
  (le_mul_of_one_le_right ha hle).trans_lt h


theorem lt_of_lt_mul_of_le_one_of_nonneg_left [PosMulMono α] (h : a < b * c) (hc : c ≤ 1)
    (hb : 0 ≤ b) : a < b :=
  h.trans_le <| mul_le_of_le_one_right hb hc


theorem lt_of_lt_mul_of_le_one_of_nonneg_right [MulPosMono α] (h : a < b * c) (hb : b ≤ 1)
    (hc : 0 ≤ c) : a < c :=
  h.trans_le <| mul_le_of_le_one_left hc hb


theorem le_mul_of_one_le_of_le_of_nonneg [MulPosMono α] (ha : 1 ≤ a) (bc : b ≤ c) (c0 : 0 ≤ c) :
    b ≤ a * c :=
  bc.trans <| le_mul_of_one_le_left c0 ha


/-- Assumes right covariance. -/
theorem Right.one_le_mul_of_le_of_le [MulPosMono α] (ha : 1 ≤ a) (hb : 1 ≤ b) (b0 : 0 ≤ b) :
    1 ≤ a * b :=
  le_mul_of_one_le_of_le_of_nonneg ha hb b0


theorem le_of_mul_le_of_one_le_of_nonneg_left [PosMulMono α] (h : a * b ≤ c) (hb : 1 ≤ b)
    (ha : 0 ≤ a) : a ≤ c :=
  (le_mul_of_one_le_right ha hb).trans h


theorem le_of_le_mul_of_le_one_of_nonneg_left [PosMulMono α] (h : a ≤ b * c) (hc : c ≤ 1)
    (hb : 0 ≤ b) : a ≤ b :=
  h.trans <| mul_le_of_le_one_right hb hc


theorem le_of_mul_le_of_one_le_nonneg_right [MulPosMono α] (h : a * b ≤ c) (ha : 1 ≤ a)
    (hb : 0 ≤ b) : b ≤ c :=
  (le_mul_of_one_le_left hb ha).trans h


theorem le_of_le_mul_of_le_one_of_nonneg_right [MulPosMono α] (h : a ≤ b * c) (hb : b ≤ 1)
    (hc : 0 ≤ c) : a ≤ c :=
  h.trans <| mul_le_of_le_one_left hc hb


theorem exists_square_le' [PosMulStrictMono α] (a0 : 0 < a) : ∃ b : α, b * b ≤ a := by
  /-
    α : Type u_3
    a : α
    inst✝³ : MulOneClass α
    inst✝² : Zero α
    inst✝¹ : LinearOrder α
    inst✝ : PosMulStrictMono α
    a0 : LT.lt 0 a
    ⊢ Exists fun b => LE.le (HMul.hMul b b) a
  -/
  obtain ha | ha := lt_or_le a 1
    /-
      case inl
      α : Type u_3
      a : α
      inst✝³ : MulOneClass α
      inst✝² : Zero α
      inst✝¹ : LinearOrder α
      inst✝ : PosMulStrictMono α
      a0 : LT.lt 0 a
      ha : LT.lt a 1
      ⊢ Exists fun b => LE.le (HMul.hMul b b) a
    -/
  · exact ⟨a, (mul_lt_of_lt_one_right a0 ha).le⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_3
      a : α
      inst✝³ : MulOneClass α
      inst✝² : Zero α
      inst✝¹ : LinearOrder α
      inst✝ : PosMulStrictMono α
      a0 : LT.lt 0 a
      ha : LE.le 1 a
      ⊢ Exists fun b => LE.le (HMul.hMul b b) a
    -/
  · exact ⟨1, by rwa [mul_one]⟩
    /-
      🎉 no goals
    -/


@[simp] lemma pow_nonneg [ZeroLEOneClass M₀] [PosMulMono M₀] (ha : 0 ≤ a) : ∀ n, 0 ≤ a ^ n
  | 0 => pow_zero a ▸ zero_le_one
  | n + 1 => pow_succ a n ▸ mul_nonneg (pow_nonneg ha _) ha


lemma zero_pow_le_one [ZeroLEOneClass M₀] : ∀ n : ℕ, (0 : M₀) ^ n ≤ 1
  | 0 => (pow_zero _).le
                /-
                  M₀ : Type u_1
                  inst✝² : MonoidWithZero M₀
                  inst✝¹ : Preorder M₀
                  inst✝ : ZeroLEOneClass M₀
                  n : Nat
                  ⊢ LE.le (HPow.hPow 0 (HAdd.hAdd n 1)) 1
                -/
  | n + 1 => by rw [zero_pow n.succ_ne_zero]; exact zero_le_one
                                              /-
                                                🎉 no goals
                                              -/


lemma pow_le_pow_of_le_one [ZeroLEOneClass M₀] [PosMulMono M₀] [MulPosMono M₀] (ha₀ : 0 ≤ a)
    (ha₁ : a ≤ 1) : ∀ {m n : ℕ}, m ≤ n → a ^ n ≤ a ^ m
  | _, _, Nat.le.refl => le_rfl
  | _, _, Nat.le.step h => by
    /-
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : Preorder M₀
      a : M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulMono M₀
      inst✝ : MulPosMono M₀
      ha₀ : LE.le 0 a
      ha₁ : LE.le a 1
      n✝ m✝ : Nat
      h : n✝.le m✝
      ⊢ LE.le (HPow.hPow a m✝.succ) (HPow.hPow a n✝)
    -/
    rw [pow_succ']
    /-
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : Preorder M₀
      a : M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulMono M₀
      inst✝ : MulPosMono M₀
      ha₀ : LE.le 0 a
      ha₁ : LE.le a 1
      n✝ m✝ : Nat
      h : n✝.le m✝
      ⊢ LE.le (HMul.hMul a (HPow.hPow a m✝)) (HPow.hPow a n✝)
    -/
    exact (mul_le_of_le_one_left (pow_nonneg ha₀ _) ha₁).trans <| pow_le_pow_of_le_one ha₀ ha₁ h
    /-
      🎉 no goals
    -/


lemma pow_le_of_le_one [ZeroLEOneClass M₀] [PosMulMono M₀] [MulPosMono M₀] (h₀ : 0 ≤ a) (h₁ : a ≤ 1)
    (hn : n ≠ 0) : a ^ n ≤ a :=
  (pow_one a).subst (pow_le_pow_of_le_one h₀ h₁ (Nat.pos_of_ne_zero hn))


lemma sq_le [ZeroLEOneClass M₀] [PosMulMono M₀] [MulPosMono M₀] (h₀ : 0 ≤ a) (h₁ : a ≤ 1) :
    a ^ 2 ≤ a := pow_le_of_le_one h₀ h₁ two_ne_zero


lemma one_le_mul_of_one_le_of_one_le [ZeroLEOneClass M₀] [PosMulMono M₀] (ha : 1 ≤ a) (hb : 1 ≤ b) :
    (1 : M₀) ≤ a * b := Left.one_le_mul_of_le_of_le ha hb <| zero_le_one.trans ha


lemma one_lt_mul_of_le_of_lt [ZeroLEOneClass M₀] [MulPosMono M₀] (ha : 1 ≤ a) (hb : 1 < b) :
    1 < a * b := hb.trans_le <| le_mul_of_one_le_left (zero_le_one.trans hb.le) ha


lemma one_lt_mul_of_lt_of_le [ZeroLEOneClass M₀] [PosMulMono M₀] (ha : 1 < a) (hb : 1 ≤ b) :
    1 < a * b := ha.trans_le <| le_mul_of_one_le_right (zero_le_one.trans ha.le) hb


alias one_lt_mul := one_lt_mul_of_le_of_lt


lemma mul_lt_one_of_nonneg_of_lt_one_left [PosMulMono M₀] (ha₀ : 0 ≤ a) (ha : a < 1) (hb : b ≤ 1) :
    a * b < 1 := (mul_le_of_le_one_right ha₀ hb).trans_lt ha


lemma mul_lt_one_of_nonneg_of_lt_one_right [MulPosMono M₀] (ha : a ≤ 1) (hb₀ : 0 ≤ b) (hb : b < 1) :
    a * b < 1 := (mul_le_of_le_one_left hb₀ ha).trans_lt hb


@[bound]
protected lemma Bound.one_lt_mul : 1 ≤ a ∧ 1 < b ∨ 1 < a ∧ 1 ≤ b → 1 < a * b := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : Preorder M₀
    a b : M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulMono M₀
    inst✝ : MulPosMono M₀
    ⊢ Or (And (LE.le 1 a) (LT.lt 1 b)) (And (LT.lt 1 a) (LE.le 1 b)) → LT.lt 1 (HM …
  -/
  rintro (⟨ha, hb⟩ | ⟨ha, hb⟩); exacts [one_lt_mul ha hb, one_lt_mul_of_lt_of_le ha hb]
                                /-
                                  🎉 no goals
                                -/


@[bound]
lemma mul_le_one₀ (ha : a ≤ 1) (hb₀ : 0 ≤ b) (hb : b ≤ 1) : a * b ≤ 1 :=
  one_mul (1 : M₀) ▸ mul_le_mul ha hb hb₀ zero_le_one


lemma pow_le_one₀ : ∀ {n : ℕ}, 0 ≤ a → a ≤ 1 → a ^ n ≤ 1
  | 0, _, _ => (pow_zero a).le
  | n + 1, h₀, h₁ => (pow_succ a n).le.trans (mul_le_one₀ (pow_le_one₀ h₀ h₁) h₀ h₁)


lemma pow_lt_one₀ (h₀ : 0 ≤ a) (h₁ : a < 1) : ∀ {n : ℕ}, n ≠ 0 → a ^ n < 1
  | 0, h => (h rfl).elim
  | n + 1, _ => by
    /-
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : Preorder M₀
      a : M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulMono M₀
      inst✝ : MulPosMono M₀
      h₀ : LE.le 0 a
      h₁ : LT.lt a 1
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ LT.lt (HPow.hPow a (HAdd.hAdd n 1)) 1
    -/
    rw [pow_succ']; exact mul_lt_one_of_nonneg_of_lt_one_left h₀ h₁ (pow_le_one₀ h₀ h₁.le)
                    /-
                      🎉 no goals
                    -/


lemma one_le_pow₀ (ha : 1 ≤ a) : ∀ {n : ℕ}, 1 ≤ a ^ n
            /-
              M₀ : Type u_1
              inst✝⁴ : MonoidWithZero M₀
              inst✝³ : Preorder M₀
              a : M₀
              inst✝² : ZeroLEOneClass M₀
              inst✝¹ : PosMulMono M₀
              inst✝ : MulPosMono M₀
              ha : LE.le 1 a
              ⊢ LE.le 1 (HPow.hPow a 0)
            -/
  | 0 => by rw [pow_zero]
            /-
              🎉 no goals
            -/
  | n + 1 => by
    simpa only [pow_succ', mul_one]
      using mul_le_mul ha (one_le_pow₀ ha) zero_le_one (zero_le_one.trans ha)


lemma one_lt_pow₀ (ha : 1 < a) : ∀ {n : ℕ}, n ≠ 0 → 1 < a ^ n
  | 0, h => (h rfl).elim
                   /-
                     M₀ : Type u_1
                     inst✝⁴ : MonoidWithZero M₀
                     inst✝³ : Preorder M₀
                     a : M₀
                     inst✝² : ZeroLEOneClass M₀
                     inst✝¹ : PosMulMono M₀
                     inst✝ : MulPosMono M₀
                     ha : LT.lt 1 a
                     n : Nat
                     x✝ : Ne (HAdd.hAdd n 1) 0
                     ⊢ LT.lt 1 (HPow.hPow a (HAdd.hAdd n 1))
                   -/
  | n + 1, _ => by rw [pow_succ']; exact one_lt_mul_of_lt_of_le ha (one_le_pow₀ ha.le)
                                   /-
                                     🎉 no goals
                                   -/


lemma pow_right_mono₀ (h : 1 ≤ a) : Monotone (a ^ ·) :=
  monotone_nat_of_le_succ fun n => by
    /-
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : Preorder M₀
      a : M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulMono M₀
      inst✝ : MulPosMono M₀
      h : LE.le 1 a
      n : Nat
      ⊢ LE.le (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
    -/
    rw [pow_succ']; exact le_mul_of_one_le_left (pow_nonneg (zero_le_one.trans h) _) h
                    /-
                      🎉 no goals
                    -/


/-- `bound` lemma for branching on `1 ≤ a ∨ a ≤ 1` when proving `a ^ n ≤ a ^ m` -/
@[bound]
lemma Bound.pow_le_pow_right_of_le_one_or_one_le (h : 1 ≤ a ∧ n ≤ m ∨ 0 ≤ a ∧ a ≤ 1 ∧ m ≤ n) :
    a ^ n ≤ a ^ m := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : Preorder M₀
    a : M₀
    m n : Nat
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulMono M₀
    inst✝ : MulPosMono M₀
    h : Or (And (LE.le 1 a) (LE.le n m)) (And (LE.le 0 a) (And (LE.le a 1) (LE.le  …
    ⊢ LE.le (HPow.hPow a n) (HPow.hPow a m)
  -/
  obtain ⟨a1, nm⟩ | ⟨a0, a1, mn⟩ := h
    /-
      case inl.intro
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : Preorder M₀
      a : M₀
      m n : Nat
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulMono M₀
      inst✝ : MulPosMono M₀
      a1 : LE.le 1 a
      nm : LE.le n m
      ⊢ LE.le (HPow.hPow a n) (HPow.hPow a m)
    -/
  · exact pow_right_mono₀ a1 nm
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : Preorder M₀
      a : M₀
      m n : Nat
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulMono M₀
      inst✝ : MulPosMono M₀
      a0 : LE.le 0 a
      a1 : LE.le a 1
      mn : LE.le m n
      ⊢ LE.le (HPow.hPow a n) (HPow.hPow a m)
    -/
  · exact pow_le_pow_of_le_one a0 a1 mn
    /-
      🎉 no goals
    -/


@[gcongr]
lemma pow_le_pow_right₀ (ha : 1 ≤ a) (hmn : m ≤ n) : a ^ m ≤ a ^ n := pow_right_mono₀ ha hmn


lemma le_self_pow₀ (ha : 1 ≤ a) (hn : n ≠ 0) : a ≤ a ^ n := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : Preorder M₀
    a : M₀
    n : Nat
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulMono M₀
    inst✝ : MulPosMono M₀
    ha : LE.le 1 a
    hn : Ne n 0
    ⊢ LE.le a (HPow.hPow a n)
  -/
  simpa only [pow_one] using pow_le_pow_right₀ ha <| Nat.pos_iff_ne_zero.2 hn
  /-
    🎉 no goals
  -/


/-- The `bound` tactic can't handle `m ≠ 0` goals yet, so we express as `0 < m` -/
@[bound]
lemma Bound.le_self_pow_of_pos (ha : 1 ≤ a) (hn : 0 < n) : a ≤ a ^ n := le_self_pow₀ ha hn.ne'


@[mono, gcongr, bound]
theorem pow_le_pow_left₀ (ha : 0 ≤ a) (hab : a ≤ b) : ∀ n, a ^ n ≤ b ^ n
            /-
              M₀ : Type u_1
              inst✝⁴ : MonoidWithZero M₀
              inst✝³ : Preorder M₀
              a b : M₀
              inst✝² : ZeroLEOneClass M₀
              inst✝¹ : PosMulMono M₀
              inst✝ : MulPosMono M₀
              ha : LE.le 0 a
              hab : LE.le a b
              ⊢ LE.le (HPow.hPow a 0) (HPow.hPow b 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by simpa only [pow_succ']
      using mul_le_mul hab (pow_le_pow_left₀ ha hab _) (pow_nonneg ha _) (ha.trans hab)


lemma pow_left_monotoneOn : MonotoneOn (fun a : M₀ ↦ a ^ n) {x | 0 ≤ x} :=
  fun _a ha _b _ hab ↦ pow_le_pow_left₀ ha hab _


lemma monotone_mul_left_of_nonneg [PosMulMono M₀] (ha : 0 ≤ a) : Monotone fun x ↦ a * x :=
  fun _ _ h ↦ mul_le_mul_of_nonneg_left h ha


lemma monotone_mul_right_of_nonneg [MulPosMono M₀] (ha : 0 ≤ a) : Monotone fun x ↦ x * a :=
  fun _ _ h ↦ mul_le_mul_of_nonneg_right h ha


lemma Monotone.mul_const [MulPosMono M₀] (hf : Monotone f) (ha : 0 ≤ a) :
    Monotone fun x ↦ f x * a := (monotone_mul_right_of_nonneg ha).comp hf


lemma Monotone.const_mul [PosMulMono M₀] (hf : Monotone f) (ha : 0 ≤ a) :
    Monotone fun x ↦ a * f x := (monotone_mul_left_of_nonneg ha).comp hf


lemma Antitone.mul_const [MulPosMono M₀] (hf : Antitone f) (ha : 0 ≤ a) :
    Antitone fun x ↦ f x * a := (monotone_mul_right_of_nonneg ha).comp_antitone hf


lemma Antitone.const_mul [PosMulMono M₀] (hf : Antitone f) (ha : 0 ≤ a) :
    Antitone fun x ↦ a * f x := (monotone_mul_left_of_nonneg ha).comp_antitone hf


lemma Monotone.mul [PosMulMono M₀] [MulPosMono M₀] (hf : Monotone f) (hg : Monotone g)
    (hf₀ : ∀ x, 0 ≤ f x) (hg₀ : ∀ x, 0 ≤ g x) : Monotone (f * g) :=
  fun _ _ h ↦ mul_le_mul (hf h) (hg h) (hg₀ _) (hf₀ _)


lemma mul_self_lt_mul_self [PosMulStrictMono M₀] [MulPosMono M₀] (ha : 0 ≤ a) (hab : a < b) :
    a * a < b * b := mul_lt_mul' hab.le hab ha <| ha.trans_lt hab

-- In the next lemma, we used to write `Set.Ici 0` instead of `{x | 0 ≤ x}`.
-- As this lemma is not used outside this file,
-- and the import for `Set.Ici` is not otherwise needed until later,
-- we choose not to use it here.

lemma strictMonoOn_mul_self [PosMulStrictMono M₀] [MulPosMono M₀] :
    StrictMonoOn (fun x ↦ x * x) {x : M₀ | 0 ≤ x} := fun _ hx _ _ hxy ↦ mul_self_lt_mul_self hx hxy

-- See Note [decidable namespace]

protected lemma Decidable.mul_lt_mul'' [PosMulMono M₀] [PosMulStrictMono M₀] [MulPosStrictMono M₀]
    [DecidableRel (α := M₀) (· ≤ ·)] (h1 : a < c) (h2 : b < d)
    (h3 : 0 ≤ a) (h4 : 0 ≤ b) : a * b < c * d :=
  h4.lt_or_eq_dec.elim (fun b0 ↦ mul_lt_mul h1 h2.le b0 <| h3.trans h1.le) fun b0 ↦ by
    /-
      M₀ : Type u_1
      inst✝⁵ : MonoidWithZero M₀
      inst✝⁴ : PartialOrder M₀
      a b c d : M₀
      inst✝³ : PosMulMono M₀
      inst✝² : PosMulStrictMono M₀
      inst✝¹ : MulPosStrictMono M₀
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      h1 : LT.lt a c
      h2 : LT.lt b d
      h3 : LE.le 0 a
      h4 : LE.le 0 b
      b0 : Eq 0 b
      ⊢ LT.lt (HMul.hMul a b) (HMul.hMul c d)
    -/
    rw [← b0, mul_zero]; exact mul_pos (h3.trans_lt h1) (h4.trans_lt h2)
                         /-
                           🎉 no goals
                         -/


lemma lt_mul_left [MulPosStrictMono M₀] (ha : 0 < a) (hb : 1 < b) : a < b * a := by
  /-
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : PartialOrder M₀
    a b : M₀
    inst✝ : MulPosStrictMono M₀
    ha : LT.lt 0 a
    hb : LT.lt 1 b
    ⊢ LT.lt a (HMul.hMul b a)
  -/
  simpa using mul_lt_mul_of_pos_right hb ha
  /-
    🎉 no goals
  -/


lemma lt_mul_right [PosMulStrictMono M₀] (ha : 0 < a) (hb : 1 < b) : a < a * b := by
  /-
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : PartialOrder M₀
    a b : M₀
    inst✝ : PosMulStrictMono M₀
    ha : LT.lt 0 a
    hb : LT.lt 1 b
    ⊢ LT.lt a (HMul.hMul a b)
  -/
  simpa using mul_lt_mul_of_pos_left hb ha
  /-
    🎉 no goals
  -/


lemma lt_mul_self [ZeroLEOneClass M₀] [MulPosStrictMono M₀] (ha : 1 < a) : a < a * a :=
  lt_mul_left (ha.trans_le' zero_le_one) ha


@[simp] lemma pow_pos (ha : 0 < a) : ∀ n, 0 < a ^ n
            /-
              M₀ : Type u_1
              inst✝³ : MonoidWithZero M₀
              inst✝² : PartialOrder M₀
              a : M₀
              inst✝¹ : ZeroLEOneClass M₀
              inst✝ : PosMulStrictMono M₀
              ha : LT.lt 0 a
              ⊢ LT.lt 0 (HPow.hPow a 0)
            -/
  | 0 => by nontriviality; rw [pow_zero]; exact zero_lt_one
                                          /-
                                            🎉 no goals
                                          -/
  | _ + 1 => pow_succ a _ ▸ mul_pos (pow_pos ha _) ha


lemma sq_pos_of_pos (ha : 0 < a) : 0 < a ^ 2 := pow_pos ha _


@[gcongr, bound]
lemma pow_lt_pow_left₀ (hab : a < b)
    (ha : 0 ≤ a) : ∀ {n : ℕ}, n ≠ 0 → a ^ n < b ^ n
  | n + 1, _ => by
    simpa only [pow_succ] using mul_lt_mul_of_le_of_lt_of_nonneg_of_pos
      (pow_le_pow_left₀ ha hab.le _) hab ha (pow_pos (ha.trans_lt hab) _)


/-- See also `pow_left_strictMono₀` and `Nat.pow_left_strictMono`. -/
lemma pow_left_strictMonoOn₀ (hn : n ≠ 0) : StrictMonoOn (· ^ n : M₀ → M₀) {a | 0 ≤ a} :=
  fun _a ha _b _ hab ↦ pow_lt_pow_left₀ hab ha hn


/-- See also `pow_right_strictMono'`. -/
lemma pow_right_strictMono₀ (h : 1 < a) : StrictMono (a ^ ·) :=
  have : 0 < a := zero_le_one.trans_lt h
  strictMono_nat_of_lt_succ fun n => by
    /-
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : PartialOrder M₀
      a : M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulStrictMono M₀
      inst✝ : MulPosStrictMono M₀
      h : LT.lt 1 a
      this : LT.lt 0 a
      n : Nat
      ⊢ LT.lt (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
    -/
    simpa only [one_mul, pow_succ'] using mul_lt_mul h (le_refl (a ^ n)) (pow_pos this _) this.le
    /-
      🎉 no goals
    -/


@[gcongr]
lemma pow_lt_pow_right₀ (h : 1 < a) (hmn : m < n) : a ^ m < a ^ n := pow_right_strictMono₀ h hmn


lemma pow_lt_pow_iff_right₀ (h : 1 < a) : a ^ n < a ^ m ↔ n < m :=
  (pow_right_strictMono₀ h).lt_iff_lt


lemma pow_le_pow_iff_right₀ (h : 1 < a) : a ^ n ≤ a ^ m ↔ n ≤ m :=
  (pow_right_strictMono₀ h).le_iff_le


lemma lt_self_pow₀ (h : 1 < a) (hm : 1 < m) : a < a ^ m := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : PartialOrder M₀
    a : M₀
    m : Nat
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    h : LT.lt 1 a
    hm : LT.lt 1 m
    ⊢ LT.lt a (HPow.hPow a m)
  -/
  simpa only [pow_one] using pow_lt_pow_right₀ h hm
  /-
    🎉 no goals
  -/


lemma pow_right_strictAnti₀ (h₀ : 0 < a) (h₁ : a < 1) : StrictAnti (a ^ ·) :=
  strictAnti_nat_of_succ_lt fun n => by
    /-
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : PartialOrder M₀
      a : M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulStrictMono M₀
      inst✝ : MulPosStrictMono M₀
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      n : Nat
      ⊢ LT.lt (HPow.hPow a (HAdd.hAdd n 1)) (HPow.hPow a n)
    -/
    simpa only [pow_succ', one_mul] using mul_lt_mul h₁ le_rfl (pow_pos h₀ n) zero_le_one
    /-
      🎉 no goals
    -/


lemma pow_lt_pow_iff_right_of_lt_one₀ (h₀ : 0 < a) (h₁ : a < 1) : a ^ m < a ^ n ↔ n < m :=
  (pow_right_strictAnti₀ h₀ h₁).lt_iff_lt


lemma pow_lt_pow_right_of_lt_one₀ (h₀ : 0 < a) (h₁ : a < 1) (hmn : m < n) : a ^ n < a ^ m :=
  (pow_lt_pow_iff_right_of_lt_one₀ h₀ h₁).2 hmn


lemma pow_lt_self_of_lt_one₀ (h₀ : 0 < a) (h₁ : a < 1) (hn : 1 < n) : a ^ n < a := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : PartialOrder M₀
    a : M₀
    n : Nat
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    h₀ : LT.lt 0 a
    h₁ : LT.lt a 1
    hn : LT.lt 1 n
    ⊢ LT.lt (HPow.hPow a n) a
  -/
  simpa only [pow_one] using pow_lt_pow_right_of_lt_one₀ h₀ h₁ hn
  /-
    🎉 no goals
  -/


lemma strictMono_mul_left_of_pos [PosMulStrictMono M₀] (ha : 0 < a) :
    StrictMono fun x ↦ a * x := fun _ _ b_lt_c ↦ mul_lt_mul_of_pos_left b_lt_c ha


lemma strictMono_mul_right_of_pos [MulPosStrictMono M₀] (ha : 0 < a) :
    StrictMono fun x ↦ x * a := fun _ _ b_lt_c ↦ mul_lt_mul_of_pos_right b_lt_c ha


lemma StrictMono.mul_const [MulPosStrictMono M₀] (hf : StrictMono f) (ha : 0 < a) :
    StrictMono fun x ↦ f x * a := (strictMono_mul_right_of_pos ha).comp hf


lemma StrictMono.const_mul [PosMulStrictMono M₀] (hf : StrictMono f) (ha : 0 < a) :
    StrictMono fun x ↦ a * f x := (strictMono_mul_left_of_pos ha).comp hf


lemma StrictAnti.mul_const [MulPosStrictMono M₀] (hf : StrictAnti f) (ha : 0 < a) :
    StrictAnti fun x ↦ f x * a := (strictMono_mul_right_of_pos ha).comp_strictAnti hf


lemma StrictAnti.const_mul [PosMulStrictMono M₀] (hf : StrictAnti f) (ha : 0 < a) :
    StrictAnti fun x ↦ a * f x := (strictMono_mul_left_of_pos ha).comp_strictAnti hf


lemma StrictMono.mul_monotone [PosMulMono M₀] [MulPosStrictMono M₀] (hf : StrictMono f)
    (hg : Monotone g) (hf₀ : ∀ x, 0 ≤ f x) (hg₀ : ∀ x, 0 < g x) :
    StrictMono (f * g) := fun _ _ h ↦ mul_lt_mul (hf h) (hg h.le) (hg₀ _) (hf₀ _)


lemma Monotone.mul_strictMono [PosMulStrictMono M₀] [MulPosMono M₀] (hf : Monotone f)
    (hg : StrictMono g) (hf₀ : ∀ x, 0 < f x) (hg₀ : ∀ x, 0 ≤ g x) :
    StrictMono (f * g) := fun _ _ h ↦ mul_lt_mul' (hf h.le) (hg h) (hg₀ _) (hf₀ _)


lemma StrictMono.mul [PosMulStrictMono M₀] [MulPosStrictMono M₀] (hf : StrictMono f)
    (hg : StrictMono g) (hf₀ : ∀ x, 0 ≤ f x) (hg₀ : ∀ x, 0 ≤ g x) :
    StrictMono (f * g) := fun _ _ h ↦ mul_lt_mul'' (hf h) (hg h) (hf₀ _) (hg₀ _)


lemma pow_le_pow_iff_left₀ (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : n ≠ 0) : a ^ n ≤ b ^ n ↔ a ≤ b :=
  (pow_left_strictMonoOn₀ hn).le_iff_le ha hb


lemma pow_lt_pow_iff_left₀ (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : n ≠ 0) : a ^ n < b ^ n ↔ a < b :=
  (pow_left_strictMonoOn₀ hn).lt_iff_lt ha hb


@[simp]
lemma pow_left_inj₀ (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : n ≠ 0) : a ^ n = b ^ n ↔ a = b :=
  (pow_left_strictMonoOn₀ hn).eq_iff_eq ha hb


lemma pow_right_injective₀ (ha₀ : 0 < a) (ha₁ : a ≠ 1) : Injective (a ^ ·) := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a : M₀
    ha₀ : LT.lt 0 a
    ha₁ : Ne a 1
    ⊢ Function.Injective fun x => HPow.hPow a x
  -/
  obtain ha₁ | ha₁ := ha₁.lt_or_lt
    /-
      case inl
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : LinearOrder M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulStrictMono M₀
      inst✝ : MulPosStrictMono M₀
      a : M₀
      ha₀ : LT.lt 0 a
      ha₁✝ : Ne a 1
      ha₁ : LT.lt a 1
      ⊢ Function.Injective fun x => HPow.hPow a x
    -/
  · exact (pow_right_strictAnti₀ ha₀ ha₁).injective
    /-
      🎉 no goals
    -/
    /-
      case inr
      M₀ : Type u_1
      inst✝⁴ : MonoidWithZero M₀
      inst✝³ : LinearOrder M₀
      inst✝² : ZeroLEOneClass M₀
      inst✝¹ : PosMulStrictMono M₀
      inst✝ : MulPosStrictMono M₀
      a : M₀
      ha₀ : LT.lt 0 a
      ha₁✝ : Ne a 1
      ha₁ : LT.lt 1 a
      ⊢ Function.Injective fun x => HPow.hPow a x
    -/
  · exact (pow_right_strictMono₀ ha₁).injective
    /-
      🎉 no goals
    -/


@[simp]
lemma pow_right_inj₀ (ha₀ : 0 < a) (ha₁ : a ≠ 1) : a ^ m = a ^ n ↔ m = n :=
  (pow_right_injective₀ ha₀ ha₁).eq_iff


lemma pow_le_one_iff_of_nonneg (ha : 0 ≤ a) (hn : n ≠ 0) : a ^ n ≤ 1 ↔ a ≤ 1 := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a : M₀
    n : Nat
    ha : LE.le 0 a
    hn : Ne n 0
    ⊢ Iff (LE.le (HPow.hPow a n) 1) (LE.le a 1)
  -/
  simpa only [one_pow] using pow_le_pow_iff_left₀ ha zero_le_one hn
  /-
    🎉 no goals
  -/


lemma one_le_pow_iff_of_nonneg (ha : 0 ≤ a) (hn : n ≠ 0) : 1 ≤ a ^ n ↔ 1 ≤ a := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a : M₀
    n : Nat
    ha : LE.le 0 a
    hn : Ne n 0
    ⊢ Iff (LE.le 1 (HPow.hPow a n)) (LE.le 1 a)
  -/
  simpa only [one_pow] using pow_le_pow_iff_left₀ zero_le_one ha hn
  /-
    🎉 no goals
  -/


lemma pow_lt_one_iff_of_nonneg (ha : 0 ≤ a) (hn : n ≠ 0) : a ^ n < 1 ↔ a < 1 :=
  lt_iff_lt_of_le_iff_le (one_le_pow_iff_of_nonneg ha hn)


lemma one_lt_pow_iff_of_nonneg (ha : 0 ≤ a) (hn : n ≠ 0) : 1 < a ^ n ↔ 1 < a := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a : M₀
    n : Nat
    ha : LE.le 0 a
    hn : Ne n 0
    ⊢ Iff (LT.lt 1 (HPow.hPow a n)) (LT.lt 1 a)
  -/
  simpa only [one_pow] using pow_lt_pow_iff_left₀ zero_le_one ha hn
  /-
    🎉 no goals
  -/


lemma pow_eq_one_iff_of_nonneg (ha : 0 ≤ a) (hn : n ≠ 0) : a ^ n = 1 ↔ a = 1 := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a : M₀
    n : Nat
    ha : LE.le 0 a
    hn : Ne n 0
    ⊢ Iff (Eq (HPow.hPow a n) 1) (Eq a 1)
  -/
  simpa only [one_pow] using pow_left_inj₀ ha zero_le_one hn
  /-
    🎉 no goals
  -/


lemma sq_le_one_iff₀ (ha : 0 ≤ a) : a ^ 2 ≤ 1 ↔ a ≤ 1 :=
  pow_le_one_iff_of_nonneg ha (Nat.succ_ne_zero _)


lemma sq_lt_one_iff₀ (ha : 0 ≤ a) : a ^ 2 < 1 ↔ a < 1 :=
  pow_lt_one_iff_of_nonneg ha (Nat.succ_ne_zero _)


lemma one_le_sq_iff₀ (ha : 0 ≤ a) : 1 ≤ a ^ 2 ↔ 1 ≤ a :=
  one_le_pow_iff_of_nonneg ha (Nat.succ_ne_zero _)


lemma one_lt_sq_iff₀ (ha : 0 ≤ a) : 1 < a ^ 2 ↔ 1 < a :=
  one_lt_pow_iff_of_nonneg ha (Nat.succ_ne_zero _)


lemma lt_of_pow_lt_pow_left₀ (n : ℕ) (hb : 0 ≤ b) (h : a ^ n < b ^ n) : a < b :=
  lt_of_not_ge fun hn => not_lt_of_ge (pow_le_pow_left₀ hb hn _) h


lemma le_of_pow_le_pow_left₀ (hn : n ≠ 0) (hb : 0 ≤ b) (h : a ^ n ≤ b ^ n) : a ≤ b :=
  le_of_not_lt fun h1 => not_le_of_lt (pow_lt_pow_left₀ h1 hb hn) h


@[simp]
                                                                                             /-
                                                                                               M₀ : Type u_1
                                                                                               inst✝⁴ : MonoidWithZero M₀
                                                                                               inst✝³ : LinearOrder M₀
                                                                                               inst✝² : ZeroLEOneClass M₀
                                                                                               inst✝¹ : PosMulStrictMono M₀
                                                                                               inst✝ : MulPosStrictMono M₀
                                                                                               a b : M₀
                                                                                               ha : LE.le 0 a
                                                                                               hb : LE.le 0 b
                                                                                               ⊢ Ne 2 0
                                                                                             -/
lemma sq_eq_sq₀ (ha : 0 ≤ a) (hb : 0 ≤ b) : a ^ 2 = b ^ 2 ↔ a = b := pow_left_inj₀ ha hb (by decide)
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


lemma lt_of_mul_self_lt_mul_self₀ (hb : 0 ≤ b) : a * a < b * b → a < b := by
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a b : M₀
    hb : LE.le 0 b
    ⊢ LT.lt (HMul.hMul a a) (HMul.hMul b b) → LT.lt a b
  -/
  simp_rw [← sq]
  /-
    M₀ : Type u_1
    inst✝⁴ : MonoidWithZero M₀
    inst✝³ : LinearOrder M₀
    inst✝² : ZeroLEOneClass M₀
    inst✝¹ : PosMulStrictMono M₀
    inst✝ : MulPosStrictMono M₀
    a b : M₀
    hb : LE.le 0 b
    ⊢ LT.lt (HPow.hPow a 2) (HPow.hPow b 2) → LT.lt a b
  -/
  exact lt_of_pow_lt_pow_left₀ _ hb
  /-
    🎉 no goals
  -/


lemma sq_lt_sq₀ (ha : 0 ≤ a) (hb : 0 ≤ b) : a ^ 2 < b ^ 2 ↔ a < b :=
  pow_lt_pow_iff_left₀ ha hb two_ne_zero


lemma sq_le_sq₀ (ha : 0 ≤ a) (hb : 0 ≤ b) : a ^ 2 ≤ b ^ 2 ↔ a ≤ b :=
  pow_le_pow_iff_left₀ ha hb two_ne_zero


theorem PosMulMono.toPosMulStrictMono [PosMulMono α] : PosMulStrictMono α :=
  ⟨fun x _ _ h => (mul_le_mul_of_nonneg_left h.le x.2.le).lt_of_ne
    (h.ne ∘ mul_left_cancel₀ x.2.ne')⟩


theorem posMulMono_iff_posMulStrictMono : PosMulMono α ↔ PosMulStrictMono α :=
  ⟨@PosMulMono.toPosMulStrictMono α _ _, @PosMulStrictMono.toPosMulMono α _ _⟩


theorem MulPosMono.toMulPosStrictMono [MulPosMono α] : MulPosStrictMono α :=
  ⟨fun x _ _ h => (mul_le_mul_of_nonneg_right h.le x.2.le).lt_of_ne
    (h.ne ∘ mul_right_cancel₀ x.2.ne')⟩


theorem mulPosMono_iff_mulPosStrictMono : MulPosMono α ↔ MulPosStrictMono α :=
  ⟨@MulPosMono.toMulPosStrictMono α _ _, @MulPosStrictMono.toMulPosMono α _ _⟩


theorem PosMulReflectLT.toPosMulReflectLE [PosMulReflectLT α] : PosMulReflectLE α :=
  ⟨fun x _ _ h =>
    h.eq_or_lt.elim (le_of_eq ∘ mul_left_cancel₀ x.2.ne.symm) fun h' =>
      (lt_of_mul_lt_mul_left h' x.2.le).le⟩


theorem posMulReflectLE_iff_posMulReflectLT : PosMulReflectLE α ↔ PosMulReflectLT α :=
  ⟨@PosMulReflectLE.toPosMulReflectLT α _ _, @PosMulReflectLT.toPosMulReflectLE α _ _⟩


theorem MulPosReflectLT.toMulPosReflectLE [MulPosReflectLT α] : MulPosReflectLE α :=
  ⟨fun x _ _ h => h.eq_or_lt.elim (le_of_eq ∘ mul_right_cancel₀ x.2.ne.symm) fun h' =>
    (lt_of_mul_lt_mul_right h' x.2.le).le⟩


theorem mulPosReflectLE_iff_mulPosReflectLT : MulPosReflectLE α ↔ MulPosReflectLT α :=
  ⟨@MulPosReflectLE.toMulPosReflectLT α _ _, @MulPosReflectLT.toMulPosReflectLE α _ _⟩


/-- See `div_self` for the version with equality when `a ≠ 0`. -/
                                                 /-
                                                   G₀ : Type u_2
                                                   inst✝² : GroupWithZero G₀
                                                   inst✝¹ : Preorder G₀
                                                   inst✝ : ZeroLEOneClass G₀
                                                   a : G₀
                                                   ⊢ LE.le (HDiv.hDiv a a) 1
                                                 -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
lemma div_self_le_one (a : G₀) : a / a ≤ 1 := by obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp] lemma inv_pos : 0 < a⁻¹ ↔ 0 < a :=
  suffices ∀ a : G₀, 0 < a → 0 < a⁻¹ from ⟨fun h ↦ inv_inv a ▸ this _ h, this a⟩
                                                    /-
                                                      G₀ : Type u_2
                                                      inst✝³ : GroupWithZero G₀
                                                      inst✝² : PartialOrder G₀
                                                      inst✝¹ : ZeroLEOneClass G₀
                                                      inst✝ : PosMulReflectLT G₀
                                                      a✝ a : G₀
                                                      ha : LT.lt 0 a
                                                      ⊢ LT.lt (HMul.hMul a 0) (HMul.hMul a (Inv.inv a))
                                                    -/
  fun a ha ↦ flip lt_of_mul_lt_mul_left ha.le <| by simp [ne_of_gt ha, zero_lt_one]
                                                    /-
                                                      🎉 no goals
                                                    -/


alias ⟨_, inv_pos_of_pos⟩ := inv_pos


                                                 /-
                                                   G₀ : Type u_2
                                                   inst✝³ : GroupWithZero G₀
                                                   inst✝² : PartialOrder G₀
                                                   inst✝¹ : ZeroLEOneClass G₀
                                                   inst✝ : PosMulReflectLT G₀
                                                   a : G₀
                                                   ⊢ Iff (LE.le 0 (Inv.inv a)) (LE.le 0 a)
                                                 -/
@[simp] lemma inv_nonneg : 0 ≤ a⁻¹ ↔ 0 ≤ a := by simp only [le_iff_eq_or_lt, inv_pos, zero_eq_inv]
                                                 /-
                                                   🎉 no goals
                                                 -/


alias ⟨_, inv_nonneg_of_nonneg⟩ := inv_nonneg


lemma one_div_pos : 0 < 1 / a ↔ 0 < a := one_div a ▸ inv_pos

lemma one_div_nonneg : 0 ≤ 1 / a ↔ 0 ≤ a := one_div a ▸ inv_nonneg


lemma div_pos [PosMulStrictMono G₀] (ha : 0 < a) (hb : 0 < b) : 0 < a / b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ LT.lt 0 (HDiv.hDiv a b)
  -/
  rw [div_eq_mul_inv]; exact mul_pos ha (inv_pos.2 hb)
                       /-
                         🎉 no goals
                       -/


lemma div_nonneg [PosMulMono G₀] (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a / b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le 0 (HDiv.hDiv a b)
  -/
  rw [div_eq_mul_inv]; exact mul_nonneg ha (inv_nonneg.2 hb)
                       /-
                         🎉 no goals
                       -/


lemma div_nonpos_of_nonpos_of_nonneg [MulPosMono G₀] (ha : a ≤ 0) (hb : 0 ≤ b) : a / b ≤ 0 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : MulPosMono G₀
    ha : LE.le a 0
    hb : LE.le 0 b
    ⊢ LE.le (HDiv.hDiv a b) 0
  -/
  rw [div_eq_mul_inv]; exact mul_nonpos_of_nonpos_of_nonneg ha (inv_nonneg.2 hb)
                       /-
                         🎉 no goals
                       -/


lemma zpow_nonneg [PosMulMono G₀] (ha : 0 ≤ a) : ∀ n : ℤ, 0 ≤ a ^ n
                  /-
                    G₀ : Type u_2
                    inst✝⁴ : GroupWithZero G₀
                    inst✝³ : PartialOrder G₀
                    inst✝² : ZeroLEOneClass G₀
                    inst✝¹ : PosMulReflectLT G₀
                    a : G₀
                    inst✝ : PosMulMono G₀
                    ha : LE.le 0 a
                    n : Nat
                    ⊢ LE.le 0 (HPow.hPow a ↑n)
                  -/
  | (n : ℕ) => by rw [zpow_natCast]; exact pow_nonneg ha _
                                     /-
                                       🎉 no goals
                                     -/
                      /-
                        G₀ : Type u_2
                        inst✝⁴ : GroupWithZero G₀
                        inst✝³ : PartialOrder G₀
                        inst✝² : ZeroLEOneClass G₀
                        inst✝¹ : PosMulReflectLT G₀
                        a : G₀
                        inst✝ : PosMulMono G₀
                        ha : LE.le 0 a
                        n : Nat
                        ⊢ LE.le 0 (HPow.hPow a (Neg.neg ↑(HAdd.hAdd n 1)))
                      -/
  |-(n + 1 : ℕ) => by rw [zpow_neg, inv_nonneg, zpow_natCast]; exact pow_nonneg ha _
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma zpow_pos [PosMulStrictMono G₀] (ha : 0 < a) : ∀ n : ℤ, 0 < a ^ n
                  /-
                    G₀ : Type u_2
                    inst✝⁴ : GroupWithZero G₀
                    inst✝³ : PartialOrder G₀
                    inst✝² : ZeroLEOneClass G₀
                    inst✝¹ : PosMulReflectLT G₀
                    a : G₀
                    inst✝ : PosMulStrictMono G₀
                    ha : LT.lt 0 a
                    n : Nat
                    ⊢ LT.lt 0 (HPow.hPow a ↑n)
                  -/
  | (n : ℕ) => by rw [zpow_natCast]; exact pow_pos ha _
                                     /-
                                       🎉 no goals
                                     -/
                      /-
                        G₀ : Type u_2
                        inst✝⁴ : GroupWithZero G₀
                        inst✝³ : PartialOrder G₀
                        inst✝² : ZeroLEOneClass G₀
                        inst✝¹ : PosMulReflectLT G₀
                        a : G₀
                        inst✝ : PosMulStrictMono G₀
                        ha : LT.lt 0 a
                        n : Nat
                        ⊢ LT.lt 0 (HPow.hPow a (Neg.neg ↑(HAdd.hAdd n 1)))
                      -/
  |-(n + 1 : ℕ) => by rw [zpow_neg, inv_pos, zpow_natCast]; exact pow_pos ha _
                                                            /-
                                                              🎉 no goals
                                                            -/


@[deprecated (since := "2024-10-08")] alias zpow_pos_of_pos := zpow_pos


/-- See `le_inv_mul_iff₀'` for a version with multiplication on the other side. -/
lemma le_inv_mul_iff₀ (hc : 0 < c) : a ≤ c⁻¹ * b ↔ c * a ≤ b where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : PosMulMono G₀
               hc : LT.lt 0 c
               h : LE.le a (HMul.hMul (Inv.inv c) b)
               ⊢ LE.le (HMul.hMul c a) b
             -/
  mp h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_left h hc.le
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : PosMulMono G₀
                hc : LT.lt 0 c
                h : LE.le (HMul.hMul c a) b
                ⊢ LE.le a (HMul.hMul (Inv.inv c) b)
              -/
  mpr h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_left h (inv_nonneg.2 hc.le)
              /-
                🎉 no goals
              -/


/-- See `inv_mul_le_iff₀'` for a version with multiplication on the other side. -/
lemma inv_mul_le_iff₀ (hc : 0 < c) : c⁻¹ * b ≤ a ↔ b ≤ c * a where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : PosMulMono G₀
               hc : LT.lt 0 c
               h : LE.le (HMul.hMul (Inv.inv c) b) a
               ⊢ LE.le b (HMul.hMul c a)
             -/
  mp h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_left h hc.le
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : PosMulMono G₀
                hc : LT.lt 0 c
                h : LE.le b (HMul.hMul c a)
                ⊢ LE.le (HMul.hMul (Inv.inv c) b) a
              -/
  mpr h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_left h (inv_nonneg.2 hc.le)
              /-
                🎉 no goals
              -/


                                                               /-
                                                                 G₀ : Type u_2
                                                                 inst✝⁴ : GroupWithZero G₀
                                                                 inst✝³ : PartialOrder G₀
                                                                 inst✝² : ZeroLEOneClass G₀
                                                                 inst✝¹ : PosMulReflectLT G₀
                                                                 a b : G₀
                                                                 inst✝ : PosMulMono G₀
                                                                 ha : LT.lt 0 a
                                                                 ⊢ Iff (LE.le 1 (HMul.hMul (Inv.inv a) b)) (LE.le a b)
                                                               -/
lemma one_le_inv_mul₀ (ha : 0 < a) : 1 ≤ a⁻¹ * b ↔ a ≤ b := by rw [le_inv_mul_iff₀ ha, mul_one]
                                                               /-
                                                                 🎉 no goals
                                                               -/

                                                               /-
                                                                 G₀ : Type u_2
                                                                 inst✝⁴ : GroupWithZero G₀
                                                                 inst✝³ : PartialOrder G₀
                                                                 inst✝² : ZeroLEOneClass G₀
                                                                 inst✝¹ : PosMulReflectLT G₀
                                                                 a b : G₀
                                                                 inst✝ : PosMulMono G₀
                                                                 ha : LT.lt 0 a
                                                                 ⊢ Iff (LE.le (HMul.hMul (Inv.inv a) b) 1) (LE.le b a)
                                                               -/
lemma inv_mul_le_one₀ (ha : 0 < a) : a⁻¹ * b ≤ 1 ↔ b ≤ a := by rw [inv_mul_le_iff₀ ha, mul_one]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- See `inv_le_iff_one_le_mul₀` for a version with multiplication on the other side. -/
lemma inv_le_iff_one_le_mul₀' (ha : 0 < a) : a⁻¹ ≤ b ↔ 1 ≤ a * b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : PosMulMono G₀
    ha : LT.lt 0 a
    ⊢ Iff (LE.le (Inv.inv a) b) (LE.le 1 (HMul.hMul a b))
  -/
  rw [← inv_mul_le_iff₀ ha, mul_one]
  /-
    🎉 no goals
  -/


                                                       /-
                                                         G₀ : Type u_2
                                                         inst✝⁴ : GroupWithZero G₀
                                                         inst✝³ : PartialOrder G₀
                                                         inst✝² : ZeroLEOneClass G₀
                                                         inst✝¹ : PosMulReflectLT G₀
                                                         a : G₀
                                                         inst✝ : PosMulMono G₀
                                                         ha : LT.lt 0 a
                                                         ⊢ Iff (LE.le 1 (Inv.inv a)) (LE.le a 1)
                                                       -/
lemma one_le_inv₀ (ha : 0 < a) : 1 ≤ a⁻¹ ↔ a ≤ 1 := by simpa using one_le_inv_mul₀ ha (b := 1)
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                       /-
                                                         G₀ : Type u_2
                                                         inst✝⁴ : GroupWithZero G₀
                                                         inst✝³ : PartialOrder G₀
                                                         inst✝² : ZeroLEOneClass G₀
                                                         inst✝¹ : PosMulReflectLT G₀
                                                         a : G₀
                                                         inst✝ : PosMulMono G₀
                                                         ha : LT.lt 0 a
                                                         ⊢ Iff (LE.le (Inv.inv a) 1) (LE.le 1 a)
                                                       -/
lemma inv_le_one₀ (ha : 0 < a) : a⁻¹ ≤ 1 ↔ 1 ≤ a := by simpa using inv_mul_le_one₀ ha (b := 1)
                                                       /-
                                                         🎉 no goals
                                                       -/


@[bound] alias ⟨_, Bound.one_le_inv₀⟩ := one_le_inv₀


@[bound]
lemma inv_le_one_of_one_le₀ (ha : 1 ≤ a) : a⁻¹ ≤ 1 := (inv_le_one₀ <| zero_lt_one.trans_le ha).2 ha


lemma one_le_inv_iff₀ : 1 ≤ a⁻¹ ↔ 0 < a ∧ a ≤ 1 where
  mp h := ⟨inv_pos.1 (zero_lt_one.trans_le h),
    inv_inv a ▸ (inv_le_one₀ <| zero_lt_one.trans_le h).2 h⟩
  mpr h := (one_le_inv₀ h.1).2 h.2


/-- One direction of `le_inv_mul_iff₀` where `c` is allowed to be `0` (but `b` must be nonnegative).
-/
lemma mul_le_of_le_inv_mul₀ (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a ≤ c⁻¹ * b) : c * a ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : PosMulMono G₀
    hb : LE.le 0 b
    hc : LE.le 0 c
    h : LE.le a (HMul.hMul (Inv.inv c) b)
    ⊢ LE.le (HMul.hMul c a) b
  -/
  obtain rfl | hc := hc.eq_or_lt
    /-
      case inl
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a b : G₀
      inst✝ : PosMulMono G₀
      hb : LE.le 0 b
      hc : LE.le 0 0
      h : LE.le a (HMul.hMul (Inv.inv 0) b)
      ⊢ LE.le (HMul.hMul 0 a) b
    -/
  · simpa using hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a b c : G₀
      inst✝ : PosMulMono G₀
      hb : LE.le 0 b
      hc✝ : LE.le 0 c
      h : LE.le a (HMul.hMul (Inv.inv c) b)
      hc : LT.lt 0 c
      ⊢ LE.le (HMul.hMul c a) b
    -/
  · rwa [le_inv_mul_iff₀ hc] at h
    /-
      🎉 no goals
    -/


/-- One direction of `inv_mul_le_iff₀` where `b` is allowed to be `0` (but `c` must be nonnegative).
-/
lemma inv_mul_le_of_le_mul₀ (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a ≤ b * c) : b⁻¹ * a ≤ c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : PosMulMono G₀
    hb : LE.le 0 b
    hc : LE.le 0 c
    h : LE.le a (HMul.hMul b c)
    ⊢ LE.le (HMul.hMul (Inv.inv b) a) c
  -/
  obtain rfl | hb := hb.eq_or_lt
    /-
      case inl
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a c : G₀
      inst✝ : PosMulMono G₀
      hc : LE.le 0 c
      hb : LE.le 0 0
      h : LE.le a (HMul.hMul 0 c)
      ⊢ LE.le (HMul.hMul (Inv.inv 0) a) c
    -/
  · simp [hc]
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a b c : G₀
      inst✝ : PosMulMono G₀
      hb✝ : LE.le 0 b
      hc : LE.le 0 c
      h : LE.le a (HMul.hMul b c)
      hb : LT.lt 0 b
      ⊢ LE.le (HMul.hMul (Inv.inv b) a) c
    -/
  · rwa [inv_mul_le_iff₀ hb]
    /-
      🎉 no goals
    -/


@[bound]
lemma inv_mul_le_one_of_le₀ (h : a ≤ b) (hb : 0 ≤ b) : b⁻¹ * a ≤ 1 :=
                                             /-
                                               G₀ : Type u_2
                                               inst✝⁴ : GroupWithZero G₀
                                               inst✝³ : PartialOrder G₀
                                               inst✝² : ZeroLEOneClass G₀
                                               inst✝¹ : PosMulReflectLT G₀
                                               a b : G₀
                                               inst✝ : PosMulMono G₀
                                               h : LE.le a b
                                               hb : LE.le 0 b
                                               ⊢ LE.le a (HMul.hMul b 1)
                                             -/
  inv_mul_le_of_le_mul₀ hb zero_le_one <| by rwa [mul_one]
                                             /-
                                               🎉 no goals
                                             -/


lemma zpow_right_mono₀ (ha : 1 ≤ a) : Monotone fun n : ℤ ↦ a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 1 a
    ⊢ Monotone fun n => HPow.hPow a n
  -/
  refine monotone_int_of_le_succ fun n ↦ ?_
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 1 a
    n : Int
    ⊢ LE.le (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
  -/
  rw [zpow_add_one₀ (zero_lt_one.trans_le ha).ne']
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 1 a
    n : Int
    ⊢ LE.le (HPow.hPow a n) (HMul.hMul (HPow.hPow a n) a)
  -/
  exact le_mul_of_one_le_right (zpow_nonneg (zero_le_one.trans ha) _) ha
  /-
    🎉 no goals
  -/


lemma zpow_right_anti₀ (ha₀ : 0 < a) (ha₁ : a ≤ 1) : Antitone fun n : ℤ ↦ a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : LE.le a 1
    ⊢ Antitone fun n => HPow.hPow a n
  -/
  refine antitone_int_of_succ_le fun n ↦ ?_
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : LE.le a 1
    n : Int
    ⊢ LE.le (HPow.hPow a (HAdd.hAdd n 1)) (HPow.hPow a n)
  -/
  rw [zpow_add_one₀ ha₀.ne']
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : LE.le a 1
    n : Int
    ⊢ LE.le (HMul.hMul (HPow.hPow a n) a) (HPow.hPow a n)
  -/
  exact mul_le_of_le_one_right (zpow_nonneg ha₀.le _) ha₁
  /-
    🎉 no goals
  -/


@[gcongr]
lemma zpow_le_zpow_right₀ (ha : 1 ≤ a) (hmn : m ≤ n) : a ^ m ≤ a ^ n := zpow_right_mono₀ ha hmn


@[gcongr]
lemma zpow_le_zpow_right_of_le_one₀ (ha₀ : 0 < a) (ha₁ : a ≤ 1) (hmn : m ≤ n) : a ^ n ≤ a ^ m :=
  zpow_right_anti₀ ha₀ ha₁ hmn


                                                               /-
                                                                 G₀ : Type u_2
                                                                 inst✝⁴ : GroupWithZero G₀
                                                                 inst✝³ : PartialOrder G₀
                                                                 inst✝² : ZeroLEOneClass G₀
                                                                 inst✝¹ : PosMulReflectLT G₀
                                                                 a : G₀
                                                                 inst✝ : PosMulMono G₀
                                                                 n : Int
                                                                 ha : LE.le 1 a
                                                                 hn : LE.le 0 n
                                                                 ⊢ LE.le 1 (HPow.hPow a n)
                                                               -/
lemma one_le_zpow₀ (ha : 1 ≤ a) (hn : 0 ≤ n) : 1 ≤ a ^ n := by simpa using zpow_right_mono₀ ha hn
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma zpow_le_one₀ (ha₀ : 0 < a) (ha₁ : a ≤ 1) (hn : 0 ≤ n) : a ^ n ≤ 1 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LE.le a 1
    hn : LE.le 0 n
    ⊢ LE.le (HPow.hPow a n) 1
  -/
  simpa using zpow_right_anti₀ ha₀ ha₁ hn
  /-
    🎉 no goals
  -/


lemma zpow_le_one_of_nonpos₀ (ha : 1 ≤ a) (hn : n ≤ 0) : a ^ n ≤ 1 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    n : Int
    ha : LE.le 1 a
    hn : LE.le n 0
    ⊢ LE.le (HPow.hPow a n) 1
  -/
  simpa using zpow_right_mono₀ ha hn
  /-
    🎉 no goals
  -/


lemma one_le_zpow_of_nonpos₀ (ha₀ : 0 < a) (ha₁ : a ≤ 1) (hn : n ≤ 0) : 1 ≤ a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LE.le a 1
    hn : LE.le n 0
    ⊢ LE.le 1 (HPow.hPow a n)
  -/
  simpa using zpow_right_anti₀ ha₀ ha₁ hn
  /-
    🎉 no goals
  -/


/-- See `le_mul_inv_iff₀'` for a version with multiplication on the other side. -/
lemma le_mul_inv_iff₀ (hc : 0 < c) : a ≤ b * c⁻¹ ↔ a * c ≤ b where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : MulPosMono G₀
               hc : LT.lt 0 c
               h : LE.le a (HMul.hMul b (Inv.inv c))
               ⊢ LE.le (HMul.hMul a c) b
             -/
  mp h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_right h hc.le
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : MulPosMono G₀
                hc : LT.lt 0 c
                h : LE.le (HMul.hMul a c) b
                ⊢ LE.le a (HMul.hMul b (Inv.inv c))
              -/
  mpr h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_right h (inv_nonneg.2 hc.le)
              /-
                🎉 no goals
              -/


/-- See `mul_inv_le_iff₀'` for a version with multiplication on the other side. -/
lemma mul_inv_le_iff₀ (hc : 0 < c) : b * c⁻¹ ≤ a ↔ b ≤ a * c where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : MulPosMono G₀
               hc : LT.lt 0 c
               h : LE.le (HMul.hMul b (Inv.inv c)) a
               ⊢ LE.le b (HMul.hMul a c)
             -/
  mp h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_right h hc.le
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : MulPosMono G₀
                hc : LT.lt 0 c
                h : LE.le b (HMul.hMul a c)
                ⊢ LE.le (HMul.hMul b (Inv.inv c)) a
              -/
  mpr h := by simpa [hc.ne'] using mul_le_mul_of_nonneg_right h (inv_nonneg.2 hc.le)
              /-
                🎉 no goals
              -/


/-- See `le_div_iff₀'` for a version with multiplication on the other side. -/
lemma le_div_iff₀ (hc : 0 < c) : a ≤ b / c ↔ a * c ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosMono G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le a (HDiv.hDiv b c)) (LE.le (HMul.hMul a c) b)
  -/
  rw [div_eq_mul_inv, le_mul_inv_iff₀ hc]
  /-
    🎉 no goals
  -/


/-- See `div_le_iff₀'` for a version with multiplication on the other side. -/
lemma div_le_iff₀ (hc : 0 < c) : b / c ≤ a ↔ b ≤ a * c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosMono G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le (HDiv.hDiv b c) a) (LE.le b (HMul.hMul a c))
  -/
  rw [div_eq_mul_inv, mul_inv_le_iff₀ hc]
  /-
    🎉 no goals
  -/


/-- See `inv_le_iff_one_le_mul₀'` for a version with multiplication on the other side. -/
lemma inv_le_iff_one_le_mul₀ (ha : 0 < a) : a⁻¹ ≤ b ↔ 1 ≤ b * a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : MulPosMono G₀
    ha : LT.lt 0 a
    ⊢ Iff (LE.le (Inv.inv a) b) (LE.le 1 (HMul.hMul b a))
  -/
  rw [← mul_inv_le_iff₀ ha, one_mul]
  /-
    🎉 no goals
  -/


                                                         /-
                                                           G₀ : Type u_2
                                                           inst✝⁴ : GroupWithZero G₀
                                                           inst✝³ : PartialOrder G₀
                                                           inst✝² : ZeroLEOneClass G₀
                                                           inst✝¹ : PosMulReflectLT G₀
                                                           a b : G₀
                                                           inst✝ : MulPosMono G₀
                                                           hb : LT.lt 0 b
                                                           ⊢ Iff (LE.le 1 (HDiv.hDiv a b)) (LE.le b a)
                                                         -/
lemma one_le_div₀ (hb : 0 < b) : 1 ≤ a / b ↔ b ≤ a := by rw [le_div_iff₀ hb, one_mul]
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           G₀ : Type u_2
                                                           inst✝⁴ : GroupWithZero G₀
                                                           inst✝³ : PartialOrder G₀
                                                           inst✝² : ZeroLEOneClass G₀
                                                           inst✝¹ : PosMulReflectLT G₀
                                                           a b : G₀
                                                           inst✝ : MulPosMono G₀
                                                           hb : LT.lt 0 b
                                                           ⊢ Iff (LE.le (HDiv.hDiv a b) 1) (LE.le a b)
                                                         -/
lemma div_le_one₀ (hb : 0 < b) : a / b ≤ 1 ↔ a ≤ b := by rw [div_le_iff₀ hb, one_mul]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- One direction of `le_mul_inv_iff₀` where `c` is allowed to be `0` (but `b` must be nonnegative).
-/
lemma mul_le_of_le_mul_inv₀ (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a ≤ b * c⁻¹) : a * c ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosMono G₀
    hb : LE.le 0 b
    hc : LE.le 0 c
    h : LE.le a (HMul.hMul b (Inv.inv c))
    ⊢ LE.le (HMul.hMul a c) b
  -/
  obtain rfl | hc := hc.eq_or_lt
    /-
      case inl
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a b : G₀
      inst✝ : MulPosMono G₀
      hb : LE.le 0 b
      hc : LE.le 0 0
      h : LE.le a (HMul.hMul b (Inv.inv 0))
      ⊢ LE.le (HMul.hMul a 0) b
    -/
  · simpa using hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a b c : G₀
      inst✝ : MulPosMono G₀
      hb : LE.le 0 b
      hc✝ : LE.le 0 c
      h : LE.le a (HMul.hMul b (Inv.inv c))
      hc : LT.lt 0 c
      ⊢ LE.le (HMul.hMul a c) b
    -/
  · rwa [le_mul_inv_iff₀ hc] at h
    /-
      🎉 no goals
    -/


/-- One direction of `mul_inv_le_iff₀` where `b` is allowed to be `0` (but `c` must be nonnegative).
-/
lemma mul_inv_le_of_le_mul₀ (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a ≤ c * b) : a * b⁻¹ ≤ c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosMono G₀
    hb : LE.le 0 b
    hc : LE.le 0 c
    h : LE.le a (HMul.hMul c b)
    ⊢ LE.le (HMul.hMul a (Inv.inv b)) c
  -/
  obtain rfl | hb := hb.eq_or_lt
    /-
      case inl
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a c : G₀
      inst✝ : MulPosMono G₀
      hc : LE.le 0 c
      hb : LE.le 0 0
      h : LE.le a (HMul.hMul c 0)
      ⊢ LE.le (HMul.hMul a (Inv.inv 0)) c
    -/
  · simp [hc]
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝⁴ : GroupWithZero G₀
      inst✝³ : PartialOrder G₀
      inst✝² : ZeroLEOneClass G₀
      inst✝¹ : PosMulReflectLT G₀
      a b c : G₀
      inst✝ : MulPosMono G₀
      hb✝ : LE.le 0 b
      hc : LE.le 0 c
      h : LE.le a (HMul.hMul c b)
      hb : LT.lt 0 b
      ⊢ LE.le (HMul.hMul a (Inv.inv b)) c
    -/
  · rwa [mul_inv_le_iff₀ hb]
    /-
      🎉 no goals
    -/


/-- One direction of `le_div_iff₀` where `c` is allowed to be `0` (but `b` must be nonnegative). -/
lemma mul_le_of_le_div₀ (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a ≤ b / c) : a * c ≤ b :=
  mul_le_of_le_mul_inv₀ hb hc (div_eq_mul_inv b _ ▸ h)


/-- One direction of `div_le_iff₀` where `b` is allowed to be `0` (but `c` must be nonnegative). -/
lemma div_le_of_le_mul₀ (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a ≤ c * b) : a / b ≤ c :=
  div_eq_mul_inv a _ ▸ mul_inv_le_of_le_mul₀ hb hc h


@[bound]
lemma mul_inv_le_one_of_le₀ (h : a ≤ b) (hb : 0 ≤ b) : a * b⁻¹ ≤ 1 :=
                                             /-
                                               G₀ : Type u_2
                                               inst✝⁴ : GroupWithZero G₀
                                               inst✝³ : PartialOrder G₀
                                               inst✝² : ZeroLEOneClass G₀
                                               inst✝¹ : PosMulReflectLT G₀
                                               a b : G₀
                                               inst✝ : MulPosMono G₀
                                               h : LE.le a b
                                               hb : LE.le 0 b
                                               ⊢ LE.le a (HMul.hMul 1 b)
                                             -/
  mul_inv_le_of_le_mul₀ hb zero_le_one <| by rwa [one_mul]
                                             /-
                                               🎉 no goals
                                             -/


@[bound]
lemma div_le_one_of_le₀ (h : a ≤ b) (hb : 0 ≤ b) : a / b ≤ 1 :=
                                         /-
                                           G₀ : Type u_2
                                           inst✝⁴ : GroupWithZero G₀
                                           inst✝³ : PartialOrder G₀
                                           inst✝² : ZeroLEOneClass G₀
                                           inst✝¹ : PosMulReflectLT G₀
                                           a b : G₀
                                           inst✝ : MulPosMono G₀
                                           h : LE.le a b
                                           hb : LE.le 0 b
                                           ⊢ LE.le a (HMul.hMul 1 b)
                                         -/
  div_le_of_le_mul₀ hb zero_le_one <| by rwa [one_mul]
                                         /-
                                           🎉 no goals
                                         -/


@[mono, gcongr, bound]
lemma div_le_div_of_nonneg_right (hab : a ≤ b) (hc : 0 ≤ c) : a / c ≤ b / c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosMono G₀
    hab : LE.le a b
    hc : LE.le 0 c
    ⊢ LE.le (HDiv.hDiv a c) (HDiv.hDiv b c)
  -/
  rw [div_eq_mul_one_div a c, div_eq_mul_one_div b c]
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosMono G₀
    hab : LE.le a b
    hc : LE.le 0 c
    ⊢ LE.le (HMul.hMul a (HDiv.hDiv 1 c)) (HMul.hMul b (HDiv.hDiv 1 c))
  -/
  exact mul_le_mul_of_nonneg_right hab (one_div_nonneg.2 hc)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-21")] alias le_div_iff := le_div_iff₀

@[deprecated (since := "2024-08-21")] alias div_le_iff := div_le_iff₀


/-- See `inv_anti₀` for the implication from right-to-left with one fewer assumption. -/
lemma inv_le_inv₀ (ha : 0 < a) (hb : 0 < b) : a⁻¹ ≤ b⁻¹ ↔ b ≤ a := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b : G₀
    inst✝¹ : MulPosMono G₀
    inst✝ : PosMulMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Iff (LE.le (Inv.inv a) (Inv.inv b)) (LE.le b a)
  -/
  rw [inv_le_iff_one_le_mul₀' ha, le_mul_inv_iff₀ hb, one_mul]
  /-
    🎉 no goals
  -/


@[gcongr, bound]
lemma inv_anti₀ (hb : 0 < b) (hba : b ≤ a) : a⁻¹ ≤ b⁻¹ := (inv_le_inv₀ (hb.trans_le hba) hb).2 hba


/-- See also `inv_le_of_inv_le₀` for a one-sided implication with one fewer assumption. -/
lemma inv_le_comm₀ (ha : 0 < a) (hb : 0 < b) : a⁻¹ ≤ b ↔ b⁻¹ ≤ a := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b : G₀
    inst✝¹ : MulPosMono G₀
    inst✝ : PosMulMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Iff (LE.le (Inv.inv a) b) (LE.le (Inv.inv b) a)
  -/
  rw [← inv_le_inv₀ hb (inv_pos.2 ha), inv_inv]
  /-
    🎉 no goals
  -/


lemma inv_le_of_inv_le₀ (ha : 0 < a) (h : a⁻¹ ≤ b) : b⁻¹ ≤ a :=
  (inv_le_comm₀ ha <| (inv_pos.2 ha).trans_le h).1 h


/-- See also `le_inv_of_le_inv₀` for a one-sided implication with one fewer assumption. -/
lemma le_inv_comm₀ (ha : 0 < a) (hb : 0 < b) : a ≤ b⁻¹ ↔ b ≤ a⁻¹ := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b : G₀
    inst✝¹ : MulPosMono G₀
    inst✝ : PosMulMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Iff (LE.le a (Inv.inv b)) (LE.le b (Inv.inv a))
  -/
  rw [← inv_le_inv₀ (inv_pos.2 hb) ha, inv_inv]
  /-
    🎉 no goals
  -/


lemma le_inv_of_le_inv₀ (ha : 0 < a) (h : a ≤ b⁻¹) : b ≤ a⁻¹ :=
  (le_inv_comm₀ ha <| inv_pos.1 <| ha.trans_le h).1 h

-- Not a `mono` lemma b/c `div_le_div₀` is strictly more general

@[gcongr]
lemma div_le_div_of_nonneg_left (ha : 0 ≤ a) (hc : 0 < c) (h : c ≤ b) : a / b ≤ a / c := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b c : G₀
    inst✝¹ : MulPosMono G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 0 a
    hc : LT.lt 0 c
    h : LE.le c b
    ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv a c)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b c : G₀
    inst✝¹ : MulPosMono G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 0 a
    hc : LT.lt 0 c
    h : LE.le c b
    ⊢ LE.le (HMul.hMul a (Inv.inv b)) (HMul.hMul a (Inv.inv c))
  -/
  exact mul_le_mul_of_nonneg_left ((inv_le_inv₀ (hc.trans_le h) hc).mpr h) ha
  /-
    🎉 no goals
  -/


/-- See `lt_inv_mul_iff₀'` for a version with multiplication on the other side. -/
lemma lt_inv_mul_iff₀ (hc : 0 < c) : a < c⁻¹ * b ↔ c * a < b where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : PosMulStrictMono G₀
               hc : LT.lt 0 c
               h : LT.lt a (HMul.hMul (Inv.inv c) b)
               ⊢ LT.lt (HMul.hMul c a) b
             -/
  mp h := by simpa [hc.ne'] using mul_lt_mul_of_pos_left h hc
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : PosMulStrictMono G₀
                hc : LT.lt 0 c
                h : LT.lt (HMul.hMul c a) b
                ⊢ LT.lt a (HMul.hMul (Inv.inv c) b)
              -/
  mpr h := by simpa [hc.ne'] using mul_lt_mul_of_pos_left h (inv_pos.2 hc)
              /-
                🎉 no goals
              -/


/-- See `inv_mul_lt_iff₀'` for a version with multiplication on the other side. -/
lemma inv_mul_lt_iff₀ (hc : 0 < c) : c⁻¹ * b < a ↔ b < c * a where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : PosMulStrictMono G₀
               hc : LT.lt 0 c
               h : LT.lt (HMul.hMul (Inv.inv c) b) a
               ⊢ LT.lt b (HMul.hMul c a)
             -/
  mp h := by simpa [hc.ne'] using mul_lt_mul_of_pos_left h hc
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : PosMulStrictMono G₀
                hc : LT.lt 0 c
                h : LT.lt b (HMul.hMul c a)
                ⊢ LT.lt (HMul.hMul (Inv.inv c) b) a
              -/
  mpr h := by simpa [hc.ne'] using mul_lt_mul_of_pos_left h (inv_pos.2 hc)
              /-
                🎉 no goals
              -/


/-- See `inv_lt_iff_one_lt_mul₀` for a version with multiplication on the other side. -/
lemma inv_lt_iff_one_lt_mul₀' (ha : 0 < a) : a⁻¹ < b ↔ 1 < a * b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    ⊢ Iff (LT.lt (Inv.inv a) b) (LT.lt 1 (HMul.hMul a b))
  -/
  rw [← inv_mul_lt_iff₀ ha, mul_one]
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 G₀ : Type u_2
                                                                 inst✝⁴ : GroupWithZero G₀
                                                                 inst✝³ : PartialOrder G₀
                                                                 inst✝² : ZeroLEOneClass G₀
                                                                 inst✝¹ : PosMulReflectLT G₀
                                                                 a b : G₀
                                                                 inst✝ : PosMulStrictMono G₀
                                                                 ha : LT.lt 0 a
                                                                 ⊢ Iff (LT.lt 1 (HMul.hMul (Inv.inv a) b)) (LT.lt a b)
                                                               -/
lemma one_lt_inv_mul₀ (ha : 0 < a) : 1 < a⁻¹ * b ↔ a < b := by rw [lt_inv_mul_iff₀ ha, mul_one]
                                                               /-
                                                                 🎉 no goals
                                                               -/

                                                               /-
                                                                 G₀ : Type u_2
                                                                 inst✝⁴ : GroupWithZero G₀
                                                                 inst✝³ : PartialOrder G₀
                                                                 inst✝² : ZeroLEOneClass G₀
                                                                 inst✝¹ : PosMulReflectLT G₀
                                                                 a b : G₀
                                                                 inst✝ : PosMulStrictMono G₀
                                                                 ha : LT.lt 0 a
                                                                 ⊢ Iff (LT.lt (HMul.hMul (Inv.inv a) b) 1) (LT.lt b a)
                                                               -/
lemma inv_mul_lt_one₀ (ha : 0 < a) : a⁻¹ * b < 1 ↔ b < a := by rw [inv_mul_lt_iff₀ ha, mul_one]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                       /-
                                                         G₀ : Type u_2
                                                         inst✝⁴ : GroupWithZero G₀
                                                         inst✝³ : PartialOrder G₀
                                                         inst✝² : ZeroLEOneClass G₀
                                                         inst✝¹ : PosMulReflectLT G₀
                                                         a : G₀
                                                         inst✝ : PosMulStrictMono G₀
                                                         ha : LT.lt 0 a
                                                         ⊢ Iff (LT.lt 1 (Inv.inv a)) (LT.lt a 1)
                                                       -/
lemma one_lt_inv₀ (ha : 0 < a) : 1 < a⁻¹ ↔ a < 1 := by simpa using one_lt_inv_mul₀ ha (b := 1)
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                       /-
                                                         G₀ : Type u_2
                                                         inst✝⁴ : GroupWithZero G₀
                                                         inst✝³ : PartialOrder G₀
                                                         inst✝² : ZeroLEOneClass G₀
                                                         inst✝¹ : PosMulReflectLT G₀
                                                         a : G₀
                                                         inst✝ : PosMulStrictMono G₀
                                                         ha : LT.lt 0 a
                                                         ⊢ Iff (LT.lt (Inv.inv a) 1) (LT.lt 1 a)
                                                       -/
lemma inv_lt_one₀ (ha : 0 < a) : a⁻¹ < 1 ↔ 1 < a := by simpa using inv_mul_lt_one₀ ha (b := 1)
                                                       /-
                                                         🎉 no goals
                                                       -/


@[bound]
lemma inv_lt_one_of_one_lt₀ (ha : 1 < a) : a⁻¹ < 1 := (inv_lt_one₀ <| zero_lt_one.trans ha).2 ha


lemma one_lt_inv_iff₀ : 1 < a⁻¹ ↔ 0 < a ∧ a < 1 where
  mp h := ⟨inv_pos.1 (zero_lt_one.trans h), inv_inv a ▸ (inv_lt_one₀ <| zero_lt_one.trans h).2 h⟩
  mpr h := (one_lt_inv₀ h.1).2 h.2


lemma zpow_right_strictMono₀ (ha : 1 < a) : StrictMono fun n : ℤ ↦ a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 1 a
    ⊢ StrictMono fun n => HPow.hPow a n
  -/
  refine strictMono_int_of_lt_succ fun n ↦ ?_
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 1 a
    n : Int
    ⊢ LT.lt (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
  -/
  rw [zpow_add_one₀ (zero_lt_one.trans ha).ne']
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 1 a
    n : Int
    ⊢ LT.lt (HPow.hPow a n) (HMul.hMul (HPow.hPow a n) a)
  -/
  exact lt_mul_of_one_lt_right (zpow_pos (zero_lt_one.trans ha) _) ha
  /-
    🎉 no goals
  -/


lemma zpow_right_strictAnti₀ (ha₀ : 0 < a) (ha₁ : a < 1) : StrictAnti fun n : ℤ ↦ a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    ⊢ StrictAnti fun n => HPow.hPow a n
  -/
  refine strictAnti_int_of_succ_lt fun n ↦ ?_
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    n : Int
    ⊢ LT.lt (HPow.hPow a (HAdd.hAdd n 1)) (HPow.hPow a n)
  -/
  rw [zpow_add_one₀ ha₀.ne']
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    n : Int
    ⊢ LT.lt (HMul.hMul (HPow.hPow a n) a) (HPow.hPow a n)
  -/
  exact mul_lt_of_lt_one_right (zpow_pos ha₀ _) ha₁
  /-
    🎉 no goals
  -/


@[gcongr]
lemma zpow_lt_zpow_right₀ (ha : 1 < a) (hmn : m < n) : a ^ m < a ^ n :=
  zpow_right_strictMono₀ ha hmn


@[gcongr]
lemma zpow_lt_zpow_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) (hmn : m < n) : a ^ n < a ^ m :=
  zpow_right_strictAnti₀ ha₀ ha₁ hmn


lemma one_lt_zpow₀ (ha : 1 < a) (hn : 0 < n) : 1 < a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha : LT.lt 1 a
    hn : LT.lt 0 n
    ⊢ LT.lt 1 (HPow.hPow a n)
  -/
  simpa using zpow_right_strictMono₀ ha hn
  /-
    🎉 no goals
  -/


lemma zpow_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) (hn : 0 < n) : a ^ n < 1 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    hn : LT.lt 0 n
    ⊢ LT.lt (HPow.hPow a n) 1
  -/
  simpa using zpow_right_strictAnti₀ ha₀ ha₁ hn
  /-
    🎉 no goals
  -/


lemma zpow_lt_one_of_neg₀ (ha : 1 < a) (hn : n < 0) : a ^ n < 1 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha : LT.lt 1 a
    hn : LT.lt n 0
    ⊢ LT.lt (HPow.hPow a n) 1
  -/
  simpa using zpow_right_strictMono₀ ha hn
  /-
    🎉 no goals
  -/


lemma one_lt_zpow_of_neg₀ (ha₀ : 0 < a) (ha₁ : a < 1) (hn : n < 0) : 1 < a ^ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    hn : LT.lt n 0
    ⊢ LT.lt 1 (HPow.hPow a n)
  -/
  simpa using zpow_right_strictAnti₀ ha₀ ha₁ hn
  /-
    🎉 no goals
  -/


@[simp] lemma zpow_le_zpow_iff_right₀ (ha : 1 < a) : a ^ m ≤ a ^ n ↔ m ≤ n :=
  (zpow_right_strictMono₀ ha).le_iff_le


@[simp] lemma zpow_lt_zpow_iff_right₀ (ha : 1 < a) : a ^ m < a ^ n ↔ m < n :=
  (zpow_right_strictMono₀ ha).lt_iff_lt


lemma zpow_le_zpow_iff_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) :
    a ^ m ≤ a ^ n ↔ n ≤ m := (zpow_right_strictAnti₀ ha₀ ha₁).le_iff_le


lemma zpow_lt_zpow_iff_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) :
    a ^ m < a ^ n ↔ n < m := (zpow_right_strictAnti₀ ha₀ ha₁).lt_iff_lt


@[simp] lemma one_le_zpow_iff_right₀ (ha : 1 < a) : 1 ≤ a ^ n ↔ 0 ≤ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha : LT.lt 1 a
    ⊢ Iff (LE.le 1 (HPow.hPow a n)) (LE.le 0 n)
  -/
  simp [← zpow_le_zpow_iff_right₀ ha]
  /-
    🎉 no goals
  -/


@[simp] lemma one_lt_zpow_iff_right₀ (ha : 1 < a) : 1 < a ^ n ↔ 0 < n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha : LT.lt 1 a
    ⊢ Iff (LT.lt 1 (HPow.hPow a n)) (LT.lt 0 n)
  -/
  simp [← zpow_lt_zpow_iff_right₀ ha]
  /-
    🎉 no goals
  -/


@[simp] lemma one_le_zpow_iff_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) : 1 ≤ a ^ n ↔ n ≤ 0 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    ⊢ Iff (LE.le 1 (HPow.hPow a n)) (LE.le n 0)
  -/
  simp [← zpow_le_zpow_iff_right_of_lt_one₀ ha₀ ha₁]
  /-
    🎉 no goals
  -/


@[simp] lemma one_lt_zpow_iff_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) : 1 < a ^ n ↔ n < 0 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    ⊢ Iff (LT.lt 1 (HPow.hPow a n)) (LT.lt n 0)
  -/
  simp [← zpow_lt_zpow_iff_right_of_lt_one₀ ha₀ ha₁]
  /-
    🎉 no goals
  -/


@[simp] lemma zpow_le_one_iff_right₀ (ha : 1 < a) : a ^ n ≤ 1 ↔ n ≤ 0 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha : LT.lt 1 a
    ⊢ Iff (LE.le (HPow.hPow a n) 1) (LE.le n 0)
  -/
  simp [← zpow_le_zpow_iff_right₀ ha]
  /-
    🎉 no goals
  -/


@[simp] lemma zpow_lt_one_iff_right₀ (ha : 1 < a) : a ^ n < 1 ↔ n < 0 := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha : LT.lt 1 a
    ⊢ Iff (LT.lt (HPow.hPow a n) 1) (LT.lt n 0)
  -/
  simp [← zpow_lt_zpow_iff_right₀ ha]
  /-
    🎉 no goals
  -/


@[simp] lemma zpow_le_one_iff_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) : a ^ n ≤ 1 ↔ 0 ≤ n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    ⊢ Iff (LE.le (HPow.hPow a n) 1) (LE.le 0 n)
  -/
  simp [← zpow_le_zpow_iff_right_of_lt_one₀ ha₀ ha₁]
  /-
    🎉 no goals
  -/


@[simp] lemma zpow_lt_one_iff_right_of_lt_one₀ (ha₀ : 0 < a) (ha₁ : a < 1) : a ^ n < 1 ↔ 0 < n := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    n : Int
    ha₀ : LT.lt 0 a
    ha₁ : LT.lt a 1
    ⊢ Iff (LT.lt (HPow.hPow a n) 1) (LT.lt 0 n)
  -/
  simp [← zpow_lt_zpow_iff_right_of_lt_one₀ ha₀ ha₁]
  /-
    🎉 no goals
  -/


/-- See `lt_mul_inv_iff₀'` for a version with multiplication on the other side. -/
lemma lt_mul_inv_iff₀ (hc : 0 < c) : a < b * c⁻¹ ↔ a * c < b where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : MulPosStrictMono G₀
               hc : LT.lt 0 c
               h : LT.lt a (HMul.hMul b (Inv.inv c))
               ⊢ LT.lt (HMul.hMul a c) b
             -/
  mp h := by simpa [hc.ne'] using mul_lt_mul_of_pos_right h hc
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : MulPosStrictMono G₀
                hc : LT.lt 0 c
                h : LT.lt (HMul.hMul a c) b
                ⊢ LT.lt a (HMul.hMul b (Inv.inv c))
              -/
  mpr h := by simpa [hc.ne'] using mul_lt_mul_of_pos_right h (inv_pos.2 hc)
              /-
                🎉 no goals
              -/


/-- See `mul_inv_lt_iff₀'` for a version with multiplication on the other side. -/
lemma mul_inv_lt_iff₀ (hc : 0 < c) : b * c⁻¹ < a ↔ b < a * c where
             /-
               G₀ : Type u_2
               inst✝⁴ : GroupWithZero G₀
               inst✝³ : PartialOrder G₀
               inst✝² : ZeroLEOneClass G₀
               inst✝¹ : PosMulReflectLT G₀
               a b c : G₀
               inst✝ : MulPosStrictMono G₀
               hc : LT.lt 0 c
               h : LT.lt (HMul.hMul b (Inv.inv c)) a
               ⊢ LT.lt b (HMul.hMul a c)
             -/
  mp h := by simpa [hc.ne'] using mul_lt_mul_of_pos_right h hc
             /-
               🎉 no goals
             -/
              /-
                G₀ : Type u_2
                inst✝⁴ : GroupWithZero G₀
                inst✝³ : PartialOrder G₀
                inst✝² : ZeroLEOneClass G₀
                inst✝¹ : PosMulReflectLT G₀
                a b c : G₀
                inst✝ : MulPosStrictMono G₀
                hc : LT.lt 0 c
                h : LT.lt b (HMul.hMul a c)
                ⊢ LT.lt (HMul.hMul b (Inv.inv c)) a
              -/
  mpr h := by simpa [hc.ne'] using mul_lt_mul_of_pos_right h (inv_pos.2 hc)
              /-
                🎉 no goals
              -/


/-- See `lt_div_iff₀'` for a version with multiplication on the other side. -/
lemma lt_div_iff₀ (hc : 0 < c) : a < b / c ↔ a * c < b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosStrictMono G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt a (HDiv.hDiv b c)) (LT.lt (HMul.hMul a c) b)
  -/
  rw [div_eq_mul_inv, lt_mul_inv_iff₀ hc]
  /-
    🎉 no goals
  -/


/-- See `div_lt_iff₀'` for a version with multiplication on the other side. -/
lemma div_lt_iff₀ (hc : 0 < c) : b / c < a ↔ b < a * c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosStrictMono G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt (HDiv.hDiv b c) a) (LT.lt b (HMul.hMul a c))
  -/
  rw [div_eq_mul_inv, mul_inv_lt_iff₀ hc]
  /-
    🎉 no goals
  -/


/-- See `inv_lt_iff_one_lt_mul₀'` for a version with multiplication on the other side. -/
lemma inv_lt_iff_one_lt_mul₀ (ha : 0 < a) : a⁻¹ < b ↔ 1 < b * a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b : G₀
    inst✝ : MulPosStrictMono G₀
    ha : LT.lt 0 a
    ⊢ Iff (LT.lt (Inv.inv a) b) (LT.lt 1 (HMul.hMul b a))
  -/
  rw [← mul_inv_lt_iff₀ ha, one_mul]
  /-
    🎉 no goals
  -/


@[gcongr, bound]
lemma div_lt_div_of_pos_right (h : a < b) (hc : 0 < c) : a / c < b / c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosStrictMono G₀
    h : LT.lt a b
    hc : LT.lt 0 c
    ⊢ LT.lt (HDiv.hDiv a c) (HDiv.hDiv b c)
  -/
  rw [div_eq_mul_one_div a c, div_eq_mul_one_div b c]
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    a b c : G₀
    inst✝ : MulPosStrictMono G₀
    h : LT.lt a b
    hc : LT.lt 0 c
    ⊢ LT.lt (HMul.hMul a (HDiv.hDiv 1 c)) (HMul.hMul b (HDiv.hDiv 1 c))
  -/
  exact mul_lt_mul_of_pos_right h (one_div_pos.2 hc)
  /-
    🎉 no goals
  -/


/-- See `inv_strictAnti₀` for the implication from right-to-left with one fewer assumption. -/
lemma inv_lt_inv₀ (ha : 0 < a) (hb : 0 < b) : a⁻¹ < b⁻¹ ↔ b < a := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b : G₀
    inst✝¹ : MulPosStrictMono G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Iff (LT.lt (Inv.inv a) (Inv.inv b)) (LT.lt b a)
  -/
  rw [inv_lt_iff_one_lt_mul₀' ha, lt_mul_inv_iff₀ hb, one_mul]
  /-
    🎉 no goals
  -/


@[gcongr, bound]
lemma inv_strictAnti₀ (hb : 0 < b) (hba : b < a) : a⁻¹ < b⁻¹ :=
  (inv_lt_inv₀ (hb.trans hba) hb).2 hba


/-- See also `inv_lt_of_inv_lt₀` for a one-sided implication with one fewer assumption. -/
lemma inv_lt_comm₀ (ha : 0 < a) (hb : 0 < b) : a⁻¹ < b ↔ b⁻¹ < a := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b : G₀
    inst✝¹ : MulPosStrictMono G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Iff (LT.lt (Inv.inv a) b) (LT.lt (Inv.inv b) a)
  -/
  rw [← inv_lt_inv₀ hb (inv_pos.2 ha), inv_inv]
  /-
    🎉 no goals
  -/


lemma inv_lt_of_inv_lt₀ (ha : 0 < a) (h : a⁻¹ < b) : b⁻¹ < a :=
  (inv_lt_comm₀ ha <| (inv_pos.2 ha).trans h).1 h


/-- See also `lt_inv_of_lt_inv₀` for a one-sided implication with one fewer assumption. -/
lemma lt_inv_comm₀ (ha : 0 < a) (hb : 0 < b) : a < b⁻¹ ↔ b < a⁻¹ := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b : G₀
    inst✝¹ : MulPosStrictMono G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Iff (LT.lt a (Inv.inv b)) (LT.lt b (Inv.inv a))
  -/
  rw [← inv_lt_inv₀ (inv_pos.2 hb) ha, inv_inv]
  /-
    🎉 no goals
  -/


lemma lt_inv_of_lt_inv₀ (ha : 0 < a) (h : a < b⁻¹) : b < a⁻¹ :=
  (lt_inv_comm₀ ha <| inv_pos.1 <| ha.trans h).1 h


@[gcongr, bound]
lemma div_lt_div_of_pos_left (ha : 0 < a) (hc : 0 < c) (h : c < b) : a / b < a / c := by
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b c : G₀
    inst✝¹ : MulPosStrictMono G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    h : LT.lt c b
    ⊢ LT.lt (HDiv.hDiv a b) (HDiv.hDiv a c)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  /-
    G₀ : Type u_2
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : PartialOrder G₀
    inst✝³ : ZeroLEOneClass G₀
    inst✝² : PosMulReflectLT G₀
    a b c : G₀
    inst✝¹ : MulPosStrictMono G₀
    inst✝ : PosMulStrictMono G₀
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    h : LT.lt c b
    ⊢ LT.lt (HMul.hMul a (Inv.inv b)) (HMul.hMul a (Inv.inv c))
  -/
  exact mul_lt_mul_of_pos_left ((inv_lt_inv₀ (hc.trans h) hc).2 h) ha
  /-
    🎉 no goals
  -/


@[simp] lemma inv_neg'' : a⁻¹ < 0 ↔ a < 0 := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ⊢ Iff (LT.lt (Inv.inv a) 0) (LT.lt a 0)
  -/
  have := PosMulMono.toPosMulReflectLT (α := G₀); simp only [← not_le, inv_nonneg]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp] lemma inv_nonpos : a⁻¹ ≤ 0 ↔ a ≤ 0 := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulMono G₀
    ⊢ Iff (LE.le (Inv.inv a) 0) (LE.le a 0)
  -/
  have := PosMulMono.toPosMulReflectLT (α := G₀); simp only [← not_lt, inv_pos]
                                                  /-
                                                    🎉 no goals
                                                  -/


alias inv_lt_zero := inv_neg''


lemma one_div_neg : 1 / a < 0 ↔ a < 0 := one_div a ▸ inv_neg''

lemma one_div_nonpos : 1 / a ≤ 0 ↔ a ≤ 0 := one_div a ▸ inv_nonpos


lemma div_nonpos_of_nonneg_of_nonpos (ha : 0 ≤ a) (hb : b ≤ 0) : a / b ≤ 0 := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a b : G₀
    inst✝ : PosMulMono G₀
    ha : LE.le 0 a
    hb : LE.le b 0
    ⊢ LE.le (HDiv.hDiv a b) 0
  -/
  rw [div_eq_mul_inv]; exact mul_nonpos_of_nonneg_of_nonpos ha (inv_nonpos.2 hb)
                       /-
                         🎉 no goals
                       -/


lemma neg_of_div_neg_right (h : a / b < 0) (ha : 0 ≤ a) : b < 0 :=
  have := PosMulMono.toPosMulReflectLT (α := G₀)
  lt_of_not_ge fun hb ↦ (div_nonneg ha hb).not_lt h


lemma neg_of_div_neg_left (h : a / b < 0) (hb : 0 ≤ b) : a < 0 :=
  have := PosMulMono.toPosMulReflectLT (α := G₀)
  lt_of_not_ge fun ha ↦ (div_nonneg ha hb).not_lt h


lemma inv_lt_one_iff₀ : a⁻¹ < 1 ↔ a ≤ 0 ∨ 1 < a := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ⊢ Iff (LT.lt (Inv.inv a) 1) (Or (LE.le a 0) (LT.lt 1 a))
  -/
  simp_rw [← not_le, one_le_inv_iff₀, not_and_or, not_lt]
  /-
    🎉 no goals
  -/


lemma inv_le_one_iff₀ : a⁻¹ ≤ 1 ↔ a ≤ 0 ∨ 1 ≤ a := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ⊢ Iff (LE.le (Inv.inv a) 1) (Or (LE.le a 0) (LE.le 1 a))
  -/
  simp only [← not_lt, one_lt_inv_iff₀, not_and_or]
  /-
    🎉 no goals
  -/


lemma zpow_right_injective₀ (ha₀ : 0 < a) (ha₁ : a ≠ 1) : Injective fun n : ℤ ↦ a ^ n := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha₀ : LT.lt 0 a
    ha₁ : Ne a 1
    ⊢ Function.Injective fun n => HPow.hPow a n
  -/
  obtain ha₁ | ha₁ := ha₁.lt_or_lt
    /-
      case inl
      G₀ : Type u_2
      inst✝³ : GroupWithZero G₀
      inst✝² : LinearOrder G₀
      inst✝¹ : ZeroLEOneClass G₀
      a : G₀
      inst✝ : PosMulStrictMono G₀
      ha₀ : LT.lt 0 a
      ha₁✝ : Ne a 1
      ha₁ : LT.lt a 1
      ⊢ Function.Injective fun n => HPow.hPow a n
    -/
  · exact (zpow_right_strictAnti₀ ha₀ ha₁).injective
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_2
      inst✝³ : GroupWithZero G₀
      inst✝² : LinearOrder G₀
      inst✝¹ : ZeroLEOneClass G₀
      a : G₀
      inst✝ : PosMulStrictMono G₀
      ha₀ : LT.lt 0 a
      ha₁✝ : Ne a 1
      ha₁ : LT.lt 1 a
      ⊢ Function.Injective fun n => HPow.hPow a n
    -/
  · exact (zpow_right_strictMono₀ ha₁).injective
    /-
      🎉 no goals
    -/


@[simp] lemma zpow_right_inj₀ (ha₀ : 0 < a) (ha₁ : a ≠ 1) : a ^ m = a ^ n ↔ m = n :=
  (zpow_right_injective₀ ha₀ ha₁).eq_iff


lemma zpow_eq_one_iff_right₀ (ha₀ : 0 ≤ a) (ha₁ : a ≠ 1) {n : ℤ} : a ^ n = 1 ↔ n = 0 := by
  /-
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha₀ : LE.le 0 a
    ha₁ : Ne a 1
    n : Int
    ⊢ Iff (Eq (HPow.hPow a n) 1) (Eq n 0)
  -/
  obtain rfl | ha₀ := ha₀.eq_or_lt
    /-
      case inl
      G₀ : Type u_2
      inst✝³ : GroupWithZero G₀
      inst✝² : LinearOrder G₀
      inst✝¹ : ZeroLEOneClass G₀
      inst✝ : PosMulStrictMono G₀
      n : Int
      ha₀ : LE.le 0 0
      ha₁ : Ne 0 1
      ⊢ Iff (Eq (HPow.hPow 0 n) 1) (Eq n 0)
    -/
  · exact zero_zpow_eq_one₀
    /-
      🎉 no goals
    -/
  /-
    case inr
    G₀ : Type u_2
    inst✝³ : GroupWithZero G₀
    inst✝² : LinearOrder G₀
    inst✝¹ : ZeroLEOneClass G₀
    a : G₀
    inst✝ : PosMulStrictMono G₀
    ha₀✝ : LE.le 0 a
    ha₁ : Ne a 1
    n : Int
    ha₀ : LT.lt 0 a
    ⊢ Iff (Eq (HPow.hPow a n) 1) (Eq n 0)
  -/
  simpa using zpow_right_inj₀ ha₀ ha₁ (n := 0)
  /-
    🎉 no goals
  -/


lemma div_le_div_iff_of_pos_right (hc : 0 < c) : a / c ≤ b / c ↔ a ≤ b where
  mp := le_imp_le_of_lt_imp_lt fun hab ↦ div_lt_div_of_pos_right hab hc
  mpr hab := div_le_div_of_nonneg_right hab hc.le


lemma div_lt_div_iff_of_pos_right (hc : 0 < c) : a / c < b / c ↔ a < b :=
  lt_iff_lt_of_le_iff_le <| div_le_div_iff_of_pos_right hc


lemma div_lt_div_iff_of_pos_left (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
                                /-
                                  G₀ : Type u_2
                                  inst✝⁴ : GroupWithZero G₀
                                  inst✝³ : LinearOrder G₀
                                  inst✝² : ZeroLEOneClass G₀
                                  a b c : G₀
                                  inst✝¹ : PosMulStrictMono G₀
                                  inst✝ : MulPosStrictMono G₀
                                  ha : LT.lt 0 a
                                  hb : LT.lt 0 b
                                  hc : LT.lt 0 c
                                  ⊢ Iff (LT.lt (HDiv.hDiv a b) (HDiv.hDiv a c)) (LT.lt c b)
                                -/
    a / b < a / c ↔ c < b := by simp only [div_eq_mul_inv, mul_lt_mul_left ha, inv_lt_inv₀ hb hc]
                                /-
                                  🎉 no goals
                                -/


lemma div_le_div_iff_of_pos_left (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a / b ≤ a / c ↔ c ≤ b :=
  le_iff_le_iff_lt_iff_lt.2 (div_lt_div_iff_of_pos_left ha hc hb)


@[mono, gcongr, bound]
lemma div_le_div₀ (hc : 0 ≤ c) (hac : a ≤ c) (hd : 0 < d) (hdb : d ≤ b) : a / b ≤ c / d := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : LinearOrder G₀
    inst✝² : ZeroLEOneClass G₀
    a b c d : G₀
    inst✝¹ : PosMulStrictMono G₀
    inst✝ : MulPosStrictMono G₀
    hc : LE.le 0 c
    hac : LE.le a c
    hd : LT.lt 0 d
    hdb : LE.le d b
    ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv c d)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul hac ((inv_le_inv₀ (hd.trans_le hdb) hd).2 hdb)
    (inv_nonneg.2 <| hd.le.trans hdb) hc


@[gcongr]
lemma div_lt_div₀ (hac : a < c) (hdb : d ≤ b) (hc : 0 ≤ c) (hd : 0 < d) : a / b < c / d := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : LinearOrder G₀
    inst✝² : ZeroLEOneClass G₀
    a b c d : G₀
    inst✝¹ : PosMulStrictMono G₀
    inst✝ : MulPosStrictMono G₀
    hac : LT.lt a c
    hdb : LE.le d b
    hc : LE.le 0 c
    hd : LT.lt 0 d
    ⊢ LT.lt (HDiv.hDiv a b) (HDiv.hDiv c d)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : LinearOrder G₀
    inst✝² : ZeroLEOneClass G₀
    a b c d : G₀
    inst✝¹ : PosMulStrictMono G₀
    inst✝ : MulPosStrictMono G₀
    hac : LT.lt a c
    hdb : LE.le d b
    hc : LE.le 0 c
    hd : LT.lt 0 d
    ⊢ LT.lt (HMul.hMul a (Inv.inv b)) (HMul.hMul c (Inv.inv d))
  -/
  exact mul_lt_mul hac ((inv_le_inv₀ (hd.trans_le hdb) hd).2 hdb) (inv_pos.2 <| hd.trans_le hdb) hc
  /-
    🎉 no goals
  -/


lemma div_lt_div₀' (hac : a ≤ c) (hdb : d < b) (hc : 0 < c) (hd : 0 < d) : a / b < c / d := by
  /-
    G₀ : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : LinearOrder G₀
    inst✝² : ZeroLEOneClass G₀
    a b c d : G₀
    inst✝¹ : PosMulStrictMono G₀
    inst✝ : MulPosStrictMono G₀
    hac : LE.le a c
    hdb : LT.lt d b
    hc : LT.lt 0 c
    hd : LT.lt 0 d
    ⊢ LT.lt (HDiv.hDiv a b) (HDiv.hDiv c d)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  exact mul_lt_mul' hac ((inv_lt_inv₀ (hd.trans hdb) hd).2 hdb)
    (inv_nonneg.2 <| hd.le.trans hdb.le) hc


theorem posMulStrictMono_iff_mulPosStrictMono : PosMulStrictMono α ↔ MulPosStrictMono α := by
  /-
    α : Type u_3
    inst✝³ : Mul α
    inst✝² : Std.Commutative fun x1 x2 => HMul.hMul x1 x2
    inst✝¹ : Zero α
    inst✝ : Preorder α
    ⊢ Iff (PosMulStrictMono α) (MulPosStrictMono α)
  -/
  simp only [PosMulStrictMono, MulPosStrictMono, Std.Commutative.comm]
  /-
    🎉 no goals
  -/


theorem posMulReflectLT_iff_mulPosReflectLT : PosMulReflectLT α ↔ MulPosReflectLT α := by
  /-
    α : Type u_3
    inst✝³ : Mul α
    inst✝² : Std.Commutative fun x1 x2 => HMul.hMul x1 x2
    inst✝¹ : Zero α
    inst✝ : Preorder α
    ⊢ Iff (PosMulReflectLT α) (MulPosReflectLT α)
  -/
  simp only [PosMulReflectLT, MulPosReflectLT, Std.Commutative.comm]
  /-
    🎉 no goals
  -/


theorem posMulMono_iff_mulPosMono : PosMulMono α ↔ MulPosMono α := by
  /-
    α : Type u_3
    inst✝³ : Mul α
    inst✝² : Std.Commutative fun x1 x2 => HMul.hMul x1 x2
    inst✝¹ : Zero α
    inst✝ : Preorder α
    ⊢ Iff (PosMulMono α) (MulPosMono α)
  -/
  simp only [PosMulMono, MulPosMono, Std.Commutative.comm]
  /-
    🎉 no goals
  -/


theorem posMulReflectLE_iff_mulPosReflectLE : PosMulReflectLE α ↔ MulPosReflectLE α := by
  /-
    α : Type u_3
    inst✝³ : Mul α
    inst✝² : Std.Commutative fun x1 x2 => HMul.hMul x1 x2
    inst✝¹ : Zero α
    inst✝ : Preorder α
    ⊢ Iff (PosMulReflectLE α) (MulPosReflectLE α)
  -/
  simp only [PosMulReflectLE, MulPosReflectLE, Std.Commutative.comm]
  /-
    🎉 no goals
  -/


/-- See `le_inv_mul_iff₀` for a version with multiplication on the other side. -/
lemma le_inv_mul_iff₀' (hc : 0 < c) : a ≤ c⁻¹ * b ↔ c * a ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le a (HMul.hMul (Inv.inv c) b)) (LE.le (HMul.hMul c a) b)
  -/
  rw [le_inv_mul_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `inv_mul_le_iff₀` for a version with multiplication on the other side. -/
lemma inv_mul_le_iff₀' (hc : 0 < c) : c⁻¹ * b ≤ a ↔ b ≤ a * c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le (HMul.hMul (Inv.inv c) b) a) (LE.le b (HMul.hMul a c))
  -/
  rw [inv_mul_le_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `le_mul_inv_iff₀` for a version with multiplication on the other side. -/
lemma le_mul_inv_iff₀' (hc : 0 < c) : a ≤ b * c⁻¹ ↔ c * a ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le a (HMul.hMul b (Inv.inv c))) (LE.le (HMul.hMul c a) b)
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosMono G₀
    ⊢ Iff (LE.le a (HMul.hMul b (Inv.inv c))) (LE.le (HMul.hMul c a) b)
  -/
  rw [le_mul_inv_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `mul_inv_le_iff₀` for a version with multiplication on the other side. -/
lemma mul_inv_le_iff₀' (hc : 0 < c) : b * c⁻¹ ≤ a ↔ b ≤ c * a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le (HMul.hMul b (Inv.inv c)) a) (LE.le b (HMul.hMul c a))
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosMono G₀
    ⊢ Iff (LE.le (HMul.hMul b (Inv.inv c)) a) (LE.le b (HMul.hMul c a))
  -/
  rw [mul_inv_le_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


lemma div_le_div_iff₀ (hb : 0 < b) (hd : 0 < d) : a / b ≤ c / d ↔ a * d ≤ c * b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c d : G₀
    hb : LT.lt 0 b
    hd : LT.lt 0 d
    ⊢ Iff (LE.le (HDiv.hDiv a b) (HDiv.hDiv c d)) (LE.le (HMul.hMul a d) (HMul.hMu …
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c d : G₀
    hb : LT.lt 0 b
    hd : LT.lt 0 d
    this : MulPosMono G₀
    ⊢ Iff (LE.le (HDiv.hDiv a b) (HDiv.hDiv c d)) (LE.le (HMul.hMul a d) (HMul.hMu …
  -/
  rw [div_le_iff₀ hb, ← mul_div_right_comm, le_div_iff₀ hd]
  /-
    🎉 no goals
  -/


/-- See `le_div_iff₀` for a version with multiplication on the other side. -/
lemma le_div_iff₀' (hc : 0 < c) : a ≤ b / c ↔ c * a ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le a (HDiv.hDiv b c)) (LE.le (HMul.hMul c a) b)
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosMono G₀
    ⊢ Iff (LE.le a (HDiv.hDiv b c)) (LE.le (HMul.hMul c a) b)
  -/
  rw [le_div_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `div_le_iff₀` for a version with multiplication on the other side. -/
lemma div_le_iff₀' (hc : 0 < c) : b / c ≤ a ↔ b ≤ c * a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LE.le (HDiv.hDiv b c) a) (LE.le b (HMul.hMul c a))
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosMono G₀
    ⊢ Iff (LE.le (HDiv.hDiv b c) a) (LE.le b (HMul.hMul c a))
  -/
  rw [div_le_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


lemma le_div_comm₀ (ha : 0 < a) (hc : 0 < c) : a ≤ b / c ↔ c ≤ b / a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    ⊢ Iff (LE.le a (HDiv.hDiv b c)) (LE.le c (HDiv.hDiv b a))
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    this : MulPosMono G₀
    ⊢ Iff (LE.le a (HDiv.hDiv b c)) (LE.le c (HDiv.hDiv b a))
  -/
  rw [le_div_iff₀ ha, le_div_iff₀' hc]
  /-
    🎉 no goals
  -/


lemma div_le_comm₀ (hb : 0 < b) (hc : 0 < c) : a / b ≤ c ↔ a / c ≤ b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hb : LT.lt 0 b
    hc : LT.lt 0 c
    ⊢ Iff (LE.le (HDiv.hDiv a b) c) (LE.le (HDiv.hDiv a c) b)
  -/
  have := posMulMono_iff_mulPosMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulMono G₀
    a b c : G₀
    hb : LT.lt 0 b
    hc : LT.lt 0 c
    this : MulPosMono G₀
    ⊢ Iff (LE.le (HDiv.hDiv a b) c) (LE.le (HDiv.hDiv a c) b)
  -/
  rw [div_le_iff₀ hb, div_le_iff₀' hc]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-21")] alias le_div_iff' := le_div_iff₀'

@[deprecated (since := "2024-08-21")] alias div_le_iff' := div_le_iff₀'


/-- See `lt_inv_mul_iff₀` for a version with multiplication on the other side. -/
lemma lt_inv_mul_iff₀' (hc : 0 < c) : a < c⁻¹ * b ↔ a * c < b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt a (HMul.hMul (Inv.inv c) b)) (LT.lt (HMul.hMul a c) b)
  -/
  rw [lt_inv_mul_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `inv_mul_lt_iff₀` for a version with multiplication on the other side. -/
lemma inv_mul_lt_iff₀' (hc : 0 < c) : c⁻¹ * b < a ↔ b < a * c := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt (HMul.hMul (Inv.inv c) b) a) (LT.lt b (HMul.hMul a c))
  -/
  rw [inv_mul_lt_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `lt_mul_inv_iff₀` for a version with multiplication on the other side. -/
lemma lt_mul_inv_iff₀' (hc : 0 < c) : a < b * c⁻¹ ↔ c * a < b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt a (HMul.hMul b (Inv.inv c))) (LT.lt (HMul.hMul c a) b)
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt a (HMul.hMul b (Inv.inv c))) (LT.lt (HMul.hMul c a) b)
  -/
  rw [lt_mul_inv_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `mul_inv_lt_iff₀` for a version with multiplication on the other side. -/
lemma mul_inv_lt_iff₀' (hc : 0 < c) : b * c⁻¹ < a ↔ b < c * a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt (HMul.hMul b (Inv.inv c)) a) (LT.lt b (HMul.hMul c a))
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt (HMul.hMul b (Inv.inv c)) a) (LT.lt b (HMul.hMul c a))
  -/
  rw [mul_inv_lt_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


lemma div_lt_div_iff₀ (hb : 0 < b) (hd : 0 < d) : a / b < c / d ↔ a * d < c * b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c d : G₀
    hb : LT.lt 0 b
    hd : LT.lt 0 d
    ⊢ Iff (LT.lt (HDiv.hDiv a b) (HDiv.hDiv c d)) (LT.lt (HMul.hMul a d) (HMul.hMu …
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c d : G₀
    hb : LT.lt 0 b
    hd : LT.lt 0 d
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt (HDiv.hDiv a b) (HDiv.hDiv c d)) (LT.lt (HMul.hMul a d) (HMul.hMu …
  -/
  rw [div_lt_iff₀ hb, ← mul_div_right_comm, lt_div_iff₀ hd]
  /-
    🎉 no goals
  -/


/-- See `lt_div_iff₀` for a version with multiplication on the other side. -/
lemma lt_div_iff₀' (hc : 0 < c) : a < b / c ↔ c * a < b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt a (HDiv.hDiv b c)) (LT.lt (HMul.hMul c a) b)
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt a (HDiv.hDiv b c)) (LT.lt (HMul.hMul c a) b)
  -/
  rw [lt_div_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


/-- See `div_lt_iff₀` for a version with multiplication on the other side. -/
lemma div_lt_iff₀' (hc : 0 < c) : b / c < a ↔ b < c * a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt (HDiv.hDiv b c) a) (LT.lt b (HMul.hMul c a))
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hc : LT.lt 0 c
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt (HDiv.hDiv b c) a) (LT.lt b (HMul.hMul c a))
  -/
  rw [div_lt_iff₀ hc, mul_comm]
  /-
    🎉 no goals
  -/


lemma lt_div_comm₀ (ha : 0 < a) (hc : 0 < c) : a < b / c ↔ c < b / a := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt a (HDiv.hDiv b c)) (LT.lt c (HDiv.hDiv b a))
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt a (HDiv.hDiv b c)) (LT.lt c (HDiv.hDiv b a))
  -/
  rw [lt_div_iff₀ ha, lt_div_iff₀' hc]
  /-
    🎉 no goals
  -/


lemma div_lt_comm₀ (hb : 0 < b) (hc : 0 < c) : a / b < c ↔ a / c < b := by
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hb : LT.lt 0 b
    hc : LT.lt 0 c
    ⊢ Iff (LT.lt (HDiv.hDiv a b) c) (LT.lt (HDiv.hDiv a c) b)
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    G₀ : Type u_2
    inst✝⁴ : CommGroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : ZeroLEOneClass G₀
    inst✝¹ : PosMulReflectLT G₀
    inst✝ : PosMulStrictMono G₀
    a b c : G₀
    hb : LT.lt 0 b
    hc : LT.lt 0 c
    this : MulPosStrictMono G₀
    ⊢ Iff (LT.lt (HDiv.hDiv a b) c) (LT.lt (HDiv.hDiv a c) b)
  -/
  rw [div_lt_iff₀ hb, div_lt_iff₀' hc]
  /-
    🎉 no goals
  -/


