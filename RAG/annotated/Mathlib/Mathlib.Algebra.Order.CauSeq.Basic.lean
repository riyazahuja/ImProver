theorem rat_add_continuous_lemma {ε : α} (ε0 : 0 < ε) :
    ∃ δ > 0, ∀ {a₁ a₂ b₁ b₂ : β}, abv (a₁ - b₁) < δ → abv (a₂ - b₂) < δ →
      abv (a₁ + a₂ - (b₁ + b₂)) < ε :=
  ⟨ε / 2, half_pos ε0, fun {a₁ a₂ b₁ b₂} h₁ h₂ => by
    simpa [add_halves, sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using
      lt_of_le_of_lt (abv_add abv _ _) (add_lt_add h₁ h₂)⟩


theorem rat_mul_continuous_lemma {ε K₁ K₂ : α} (ε0 : 0 < ε) :
    ∃ δ > 0, ∀ {a₁ a₂ b₁ b₂ : β}, abv a₁ < K₁ → abv b₂ < K₂ → abv (a₁ - b₁) < δ →
      abv (a₂ - b₂) < δ → abv (a₁ * a₂ - b₁ * b₂) < ε := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ {a₁ a₂ b₁ b₂ : β}, LT.lt (abv a₁) K₁ → LT …
  -/
  have K0 : (0 : α) < max 1 (max K₁ K₂) := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 (Max.max 1 (Max.max K₁ K₂))
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ {a₁ a₂ b₁ b₂ : β}, LT.lt (abv a₁) K₁ → LT …
  -/
  have εK := div_pos (half_pos ε0) K0
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 (Max.max 1 (Max.max K₁ K₂))
    εK : LT.lt 0 (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max.max K₁ K₂)))
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ {a₁ a₂ b₁ b₂ : β}, LT.lt (abv a₁) K₁ → LT …
  -/
  refine ⟨_, εK, fun {a₁ a₂ b₁ b₂} ha₁ hb₂ h₁ h₂ => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 (Max.max 1 (Max.max K₁ K₂))
    εK : LT.lt 0 (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max.max K₁ K₂)))
    a₁ a₂ b₁ b₂ : β
    ha₁ : LT.lt (abv a₁) K₁
    hb₂ : LT.lt (abv b₂) K₂
    h₁ : LT.lt (abv (HSub.hSub a₁ b₁)) (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max. …
    h₂ : LT.lt (abv (HSub.hSub a₂ b₂)) (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max. …
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))) ε
  -/
  replace ha₁ := lt_of_lt_of_le ha₁ (le_trans (le_max_left _ K₂) (le_max_right 1 _))
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 (Max.max 1 (Max.max K₁ K₂))
    εK : LT.lt 0 (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max.max K₁ K₂)))
    a₁ a₂ b₁ b₂ : β
    hb₂ : LT.lt (abv b₂) K₂
    h₁ : LT.lt (abv (HSub.hSub a₁ b₁)) (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max. …
    h₂ : LT.lt (abv (HSub.hSub a₂ b₂)) (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max. …
    ha₁ : LT.lt (abv a₁) (Max.max 1 (Max.max K₁ K₂))
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))) ε
  -/
  replace hb₂ := lt_of_lt_of_le hb₂ (le_trans (le_max_right K₁ _) (le_max_right 1 _))
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 (Max.max 1 (Max.max K₁ K₂))
    εK : LT.lt 0 (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max.max K₁ K₂)))
    a₁ a₂ b₁ b₂ : β
    h₁ : LT.lt (abv (HSub.hSub a₁ b₁)) (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max. …
    h₂ : LT.lt (abv (HSub.hSub a₂ b₂)) (HDiv.hDiv (HDiv.hDiv ε 2) (Max.max 1 (Max. …
    ha₁ : LT.lt (abv a₁) (Max.max 1 (Max.max K₁ K₂))
    hb₂ : LT.lt (abv b₂) (Max.max 1 (Max.max K₁ K₂))
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))) ε
  -/
  set M := max 1 (max K₁ K₂)
  have : abv (a₁ - b₁) * abv b₂ + abv (a₂ - b₂) * abv a₁ < ε / 2 / M * M + ε / 2 / M * M := by
    gcongr
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K₁ K₂ : α
    ε0 : LT.lt 0 ε
    a₁ a₂ b₁ b₂ : β
    M : α := Max.max 1 (Max.max K₁ K₂)
    K0 : LT.lt 0 M
    εK : LT.lt 0 (HDiv.hDiv (HDiv.hDiv ε 2) M)
    h₁ : LT.lt (abv (HSub.hSub a₁ b₁)) (HDiv.hDiv (HDiv.hDiv ε 2) M)
    h₂ : LT.lt (abv (HSub.hSub a₂ b₂)) (HDiv.hDiv (HDiv.hDiv ε 2) M)
    ha₁ : LT.lt (abv a₁) M
    hb₂ : LT.lt (abv b₂) M
    this : LT.lt (HAdd.hAdd (HMul.hMul (abv (HSub.hSub a₁ b₁)) (abv b₂)) (HMul.hMu …
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))) ε
  -/
  rw [← abv_mul abv, mul_comm, div_mul_cancel₀ _ (ne_of_gt K0), ← abv_mul abv, add_halves] at this
  simpa [sub_eq_add_neg, mul_add, add_mul, add_left_comm] using
    lt_of_le_of_lt (abv_add abv _ _) this


theorem rat_inv_continuous_lemma {β : Type*} [DivisionRing β] (abv : β → α) [IsAbsoluteValue abv]
    {ε K : α} (ε0 : 0 < ε) (K0 : 0 < K) :
    ∃ δ > 0, ∀ {a b : β}, K ≤ abv a → K ≤ abv b → abv (a - b) < δ → abv (a⁻¹ - b⁻¹) < ε := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ {a b : β}, LE.le K (abv a) → LE.le K (abv …
  -/
  refine ⟨K * ε * K, mul_pos (mul_pos K0 ε0) K0, fun {a b} ha hb h => ?_⟩
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    a b : β
    ha : LE.le K (abv a)
    hb : LE.le K (abv b)
    h : LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul K ε) K)
    ⊢ LT.lt (abv (HSub.hSub (Inv.inv a) (Inv.inv b))) ε
  -/
  have a0 := K0.trans_le ha
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    a b : β
    ha : LE.le K (abv a)
    hb : LE.le K (abv b)
    h : LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul K ε) K)
    a0 : LT.lt 0 (abv a)
    ⊢ LT.lt (abv (HSub.hSub (Inv.inv a) (Inv.inv b))) ε
  -/
  have b0 := K0.trans_le hb
  rw [inv_sub_inv' ((abv_pos abv).1 a0) ((abv_pos abv).1 b0), abv_mul abv, abv_mul abv, abv_inv abv,
    abv_inv abv, abv_sub abv]
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    a b : β
    ha : LE.le K (abv a)
    hb : LE.le K (abv b)
    h : LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul K ε) K)
    a0 : LT.lt 0 (abv a)
    b0 : LT.lt 0 (abv b)
    ⊢ LT.lt (HMul.hMul (HMul.hMul (Inv.inv (abv a)) (abv (HSub.hSub a b))) (Inv.in …
  -/
  refine lt_of_mul_lt_mul_left (lt_of_mul_lt_mul_right ?_ b0.le) a0.le
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    a b : β
    ha : LE.le K (abv a)
    hb : LE.le K (abv b)
    h : LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul K ε) K)
    a0 : LT.lt 0 (abv a)
    b0 : LT.lt 0 (abv b)
    ⊢ LT.lt (HMul.hMul (HMul.hMul (abv a) (HMul.hMul (HMul.hMul (Inv.inv (abv a))  …
  -/
  rw [mul_assoc, inv_mul_cancel_right₀ b0.ne', ← mul_assoc, mul_inv_cancel₀ a0.ne', one_mul]
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    a b : β
    ha : LE.le K (abv a)
    hb : LE.le K (abv b)
    h : LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul K ε) K)
    a0 : LT.lt 0 (abv a)
    b0 : LT.lt 0 (abv b)
    ⊢ LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul (abv a) ε) (abv b))
  -/
  refine h.trans_le ?_
  /-
    α : Type u_1
    inst✝² : LinearOrderedField α
    β : Type u_3
    inst✝¹ : DivisionRing β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    ε K : α
    ε0 : LT.lt 0 ε
    K0 : LT.lt 0 K
    a b : β
    ha : LE.le K (abv a)
    hb : LE.le K (abv b)
    h : LT.lt (abv (HSub.hSub a b)) (HMul.hMul (HMul.hMul K ε) K)
    a0 : LT.lt 0 (abv a)
    b0 : LT.lt 0 (abv b)
    ⊢ LE.le (HMul.hMul (HMul.hMul K ε) K) (HMul.hMul (HMul.hMul (abv a) ε) (abv b))
  -/
  gcongr
  /-
    🎉 no goals
  -/


/-- A sequence is Cauchy if the distance between its entries tends to zero. -/
def IsCauSeq {α : Type*} [LinearOrderedField α] {β : Type*} [Ring β] (abv : β → α) (f : ℕ → β) :
    Prop :=
  ∀ ε > 0, ∃ i, ∀ j ≥ i, abv (f j - f i) < ε


theorem cauchy₂ (hf : IsCauSeq abv f) {ε : α} (ε0 : 0 < ε) :
    ∃ i, ∀ j ≥ i, ∀ k ≥ i, abv (f j - f k) < ε := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    ε : α
    ε0 : LT.lt 0 ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k i → LT.lt (abv …
  -/
  refine (hf _ (half_pos ε0)).imp fun i hi j ij k ik => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    ε : α
    ε0 : LT.lt 0 ε
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) (HDiv.hDiv ε …
    j : Nat
    ij : GE.ge j i
    k : Nat
    ik : GE.ge k i
    ⊢ LT.lt (abv (HSub.hSub (f j) (f k))) ε
  -/
  rw [← add_halves ε]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    ε : α
    ε0 : LT.lt 0 ε
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) (HDiv.hDiv ε …
    j : Nat
    ij : GE.ge j i
    k : Nat
    ik : GE.ge k i
    ⊢ LT.lt (abv (HSub.hSub (f j) (f k))) (HAdd.hAdd (HDiv.hDiv ε 2) (HDiv.hDiv ε  …
  -/
  refine lt_of_le_of_lt (abv_sub_le abv _ _ _) (add_lt_add (hi _ ij) ?_)
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    ε : α
    ε0 : LT.lt 0 ε
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) (HDiv.hDiv ε …
    j : Nat
    ij : GE.ge j i
    k : Nat
    ik : GE.ge k i
    ⊢ LT.lt (abv (HSub.hSub (f i) (f k))) (HDiv.hDiv ε 2)
  -/
  rw [abv_sub abv]; exact hi _ ik
                    /-
                      🎉 no goals
                    -/


theorem cauchy₃ (hf : IsCauSeq abv f) {ε : α} (ε0 : 0 < ε) :
    ∃ i, ∀ j ≥ i, ∀ k ≥ j, abv (f k - f j) < ε :=
  let ⟨i, H⟩ := hf.cauchy₂ ε0
  ⟨i, fun _ ij _ jk => H _ (le_trans ij jk) _ ij⟩


lemma bounded (hf : IsCauSeq abv f) : ∃ r, ∀ i, abv (f i) < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    ⊢ Exists fun r => ∀ (i : Nat), LT.lt (abv (f i)) r
  -/
  obtain ⟨i, h⟩ := hf _ zero_lt_one
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    i : Nat
    h : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) 1
    ⊢ Exists fun r => ∀ (i : Nat), LT.lt (abv (f i)) r
  -/
  set R : ℕ → α := @Nat.rec (fun _ => α) (abv (f 0)) fun i c => max c (abv (f i.succ)) with hR
  have : ∀ i, ∀ j ≤ i, abv (f j) ≤ R i := by
    refine Nat.rec (by simp [hR]) ?_
    rintro i hi j (rfl | hj)
    · simp [R]
    · exact (hi j hj).trans (le_max_left _ _)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    i : Nat
    h : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) 1
    R : Nat → α := Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ))
    hR : Eq R (Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ)))
    this : ∀ (i j : Nat), LE.le j i → LE.le (abv (f j)) (R i)
    ⊢ Exists fun r => ∀ (i : Nat), LT.lt (abv (f i)) r
  -/
  refine ⟨R i + 1, fun j ↦ ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    hf : IsCauSeq abv f
    i : Nat
    h : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) 1
    R : Nat → α := Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ))
    hR : Eq R (Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ)))
    this : ∀ (i j : Nat), LE.le j i → LE.le (abv (f j)) (R i)
    j : Nat
    ⊢ LT.lt (abv (f j)) (HAdd.hAdd (R i) 1)
  -/
  obtain hji | hij := le_total j i
    /-
      case intro.inl
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Ring β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      f : Nat → β
      hf : IsCauSeq abv f
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) 1
      R : Nat → α := Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ))
      hR : Eq R (Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ)))
      this : ∀ (i j : Nat), LE.le j i → LE.le (abv (f j)) (R i)
      j : Nat
      hji : LE.le j i
      ⊢ LT.lt (abv (f j)) (HAdd.hAdd (R i) 1)
    -/
  · exact (this i _ hji).trans_lt (lt_add_one _)
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Ring β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      f : Nat → β
      hf : IsCauSeq abv f
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (f j) (f i))) 1
      R : Nat → α := Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ))
      hR : Eq R (Nat.rec (abv (f 0)) fun i c => Max.max c (abv (f i.succ)))
      this : ∀ (i j : Nat), LE.le j i → LE.le (abv (f j)) (R i)
      j : Nat
      hij : LE.le i j
      ⊢ LT.lt (abv (f j)) (HAdd.hAdd (R i) 1)
    -/
  · simpa using (abv_add abv _ _).trans_lt <| add_lt_add_of_le_of_lt (this i _ le_rfl) (h _ hij)
    /-
      🎉 no goals
    -/


lemma bounded' (hf : IsCauSeq abv f) (x : α) : ∃ r > x, ∀ i, abv (f i) < r :=
  let ⟨r, h⟩ := hf.bounded
  ⟨max r (x + 1), (lt_add_one x).trans_le (le_max_right _ _),
    fun i ↦ (h i).trans_le (le_max_left _ _)⟩


lemma const (x : β) : IsCauSeq abv fun _ ↦ x :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 inst✝² : LinearOrderedField α
                                 inst✝¹ : Ring β
                                 abv : β → α
                                 inst✝ : IsAbsoluteValue abv
                                 x : β
                                 ε : α
                                 ε0 : GT.gt ε 0
                                 j : Nat
                                 x✝ : GE.ge j 0
                                 ⊢ LT.lt (abv (HSub.hSub ((fun x_1 => x) j) ((fun x_1 => x) 0))) ε
                               -/
  fun ε ε0 ↦ ⟨0, fun j _ => by simpa [abv_zero] using ε0⟩
                               /-
                                 🎉 no goals
                               -/


theorem add (hf : IsCauSeq abv f) (hg : IsCauSeq abv g) : IsCauSeq abv (f + g) := fun _ ε0 =>
  let ⟨_, δ0, Hδ⟩ := rat_add_continuous_lemma abv ε0
  let ⟨i, H⟩ := exists_forall_ge_and (hf.cauchy₃ δ0) (hg.cauchy₃ δ0)
  ⟨i, fun _ ij =>
    let ⟨H₁, H₂⟩ := H _ le_rfl
    Hδ (H₁ _ ij) (H₂ _ ij)⟩


lemma mul (hf : IsCauSeq abv f) (hg : IsCauSeq abv g) : IsCauSeq abv (f * g) := fun _ ε0 =>
  let ⟨_, _, hF⟩ := hf.bounded' 0
  let ⟨_, _, hG⟩ := hg.bounded' 0
  let ⟨_, δ0, Hδ⟩ := rat_mul_continuous_lemma abv ε0
  let ⟨i, H⟩ := exists_forall_ge_and (hf.cauchy₃ δ0) (hg.cauchy₃ δ0)
  ⟨i, fun j ij =>
    let ⟨H₁, H₂⟩ := H _ le_rfl
    Hδ (hF j) (hG i) (H₁ _ ij) (H₂ _ ij)⟩


@[simp] lemma _root_.isCauSeq_neg : IsCauSeq abv (-f) ↔ IsCauSeq abv f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    ⊢ Iff (IsCauSeq abv (Neg.neg f)) (IsCauSeq abv f)
  -/
  simp only [IsCauSeq, Pi.neg_apply, ← neg_sub', abv_neg]
  /-
    🎉 no goals
  -/


protected alias ⟨of_neg, neg⟩ := isCauSeq_neg


/-- `CauSeq β abv` is the type of `β`-valued Cauchy sequences, with respect to the absolute value
function `abv`. -/
def CauSeq {α : Type*} [LinearOrderedField α] (β : Type*) [Ring β] (abv : β → α) : Type _ :=
  { f : ℕ → β // IsCauSeq abv f }


instance : CoeFun (CauSeq β abv) fun _ => ℕ → β :=
  ⟨Subtype.val⟩

-- Porting note: Remove coeFn theorem
/-@[simp]
theorem mk_to_fun (f) (hf : IsCauSeq abv f) : @coeFn (CauSeq β abv) _ _ ⟨f, hf⟩ = f :=
  rfl -/


@[ext]
theorem ext {f g : CauSeq β abv} (h : ∀ i, f i = g i) : f = g := Subtype.eq (funext h)


theorem isCauSeq (f : CauSeq β abv) : IsCauSeq abv f :=
  f.2


theorem cauchy (f : CauSeq β abv) : ∀ {ε}, 0 < ε → ∃ i, ∀ j ≥ i, abv (f j - f i) < ε := @f.2


/-- Given a Cauchy sequence `f`, create a Cauchy sequence from a sequence `g` with
the same values as `f`. -/
def ofEq (f : CauSeq β abv) (g : ℕ → β) (e : ∀ i, f i = g i) : CauSeq β abv :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝¹ : LinearOrderedField α
                    inst✝ : Ring β
                    abv : β → α
                    f : CauSeq β abv
                    g : Nat → β
                    e : ∀ (i : Nat), Eq (↑f i) (g i)
                    ε : α
                    ⊢ GT.gt ε 0 → Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub ( …
                  -/
  ⟨g, fun ε => by rw [show g = f from (funext e).symm]; exact f.cauchy⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem cauchy₂ (f : CauSeq β abv) {ε} :
    0 < ε → ∃ i, ∀ j ≥ i, ∀ k ≥ i, abv (f j - f k) < ε :=
  f.2.cauchy₂


theorem cauchy₃ (f : CauSeq β abv) {ε} : 0 < ε → ∃ i, ∀ j ≥ i, ∀ k ≥ j, abv (f k - f j) < ε :=
  f.2.cauchy₃


theorem bounded (f : CauSeq β abv) : ∃ r, ∀ i, abv (f i) < r := f.2.bounded


theorem bounded' (f : CauSeq β abv) (x : α) : ∃ r > x, ∀ i, abv (f i) < r := f.2.bounded' x


instance : Add (CauSeq β abv) :=
  ⟨fun f g => ⟨f + g, f.2.add g.2⟩⟩


@[simp, norm_cast]
theorem coe_add (f g : CauSeq β abv) : ⇑(f + g) = (f : ℕ → β) + g :=
  rfl


@[simp, norm_cast]
theorem add_apply (f g : CauSeq β abv) (i : ℕ) : (f + g) i = f i + g i :=
  rfl


/-- The constant Cauchy sequence. -/
def const (x : β) : CauSeq β abv := ⟨fun _ ↦ x, IsCauSeq.const _⟩


/-- The constant Cauchy sequence -/
local notation "const" => const abv


@[simp, norm_cast]
theorem coe_const (x : β) : (const x : ℕ → β) = Function.const ℕ x :=
  rfl


@[simp, norm_cast]
theorem const_apply (x : β) (i : ℕ) : (const x : ℕ → β) i = x :=
  rfl


theorem const_inj {x y : β} : (const x : CauSeq β abv) = const y ↔ x = y :=
  ⟨fun h => congr_arg (fun f : CauSeq β abv => (f : ℕ → β) 0) h, congr_arg _⟩


instance : Zero (CauSeq β abv) :=
  ⟨const 0⟩


instance : One (CauSeq β abv) :=
  ⟨const 1⟩


instance : Inhabited (CauSeq β abv) :=
  ⟨0⟩


@[simp, norm_cast]
theorem coe_zero : ⇑(0 : CauSeq β abv) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_one : ⇑(1 : CauSeq β abv) = 1 :=
  rfl


@[simp, norm_cast]
theorem zero_apply (i) : (0 : CauSeq β abv) i = 0 :=
  rfl


@[simp, norm_cast]
theorem one_apply (i) : (1 : CauSeq β abv) i = 1 :=
  rfl


@[simp]
theorem const_zero : const 0 = 0 :=
  rfl


@[simp]
theorem const_one : const 1 = 1 :=
  rfl


theorem const_add (x y : β) : const (x + y) = const x + const y :=
  rfl


instance : Mul (CauSeq β abv) := ⟨fun f g ↦ ⟨f * g, f.2.mul g.2⟩⟩


@[simp, norm_cast]
theorem coe_mul (f g : CauSeq β abv) : ⇑(f * g) = (f : ℕ → β) * g :=
  rfl


@[simp, norm_cast]
theorem mul_apply (f g : CauSeq β abv) (i : ℕ) : (f * g) i = f i * g i :=
  rfl


theorem const_mul (x y : β) : const (x * y) = const x * const y :=
  rfl


instance : Neg (CauSeq β abv) := ⟨fun f ↦ ⟨-f, f.2.neg⟩⟩


@[simp, norm_cast]
theorem coe_neg (f : CauSeq β abv) : ⇑(-f) = -f :=
  rfl


@[simp, norm_cast]
theorem neg_apply (f : CauSeq β abv) (i) : (-f) i = -f i :=
  rfl


theorem const_neg (x : β) : const (-x) = -const x :=
  rfl


instance : Sub (CauSeq β abv) :=
                                                             /-
                                                               α : Type u_1
                                                               β : Type u_2
                                                               inst✝² : LinearOrderedField α
                                                               inst✝¹ : Ring β
                                                               abv : β → α
                                                               inst✝ : IsAbsoluteValue abv
                                                               f g : CauSeq β abv
                                                               i : Nat
                                                               ⊢ Eq (↑(HAdd.hAdd f (Neg.neg g)) i) ((fun x => HSub.hSub (↑f x) (↑g x)) i)
                                                             -/
  ⟨fun f g => ofEq (f + -g) (fun x => f x - g x) fun i => by simp [sub_eq_add_neg]⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp, norm_cast]
theorem coe_sub (f g : CauSeq β abv) : ⇑(f - g) = (f : ℕ → β) - g :=
  rfl


@[simp, norm_cast]
theorem sub_apply (f g : CauSeq β abv) (i : ℕ) : (f - g) i = f i - g i :=
  rfl


theorem const_sub (x y : β) : const (x - y) = const x - const y :=
  rfl


instance : SMul G (CauSeq β abv) :=
  ⟨fun a f => (ofEq (const (a • (1 : β)) * f) (a • (f : ℕ → β))) fun _ => smul_one_mul _ _⟩


@[simp, norm_cast]
theorem coe_smul (a : G) (f : CauSeq β abv) : ⇑(a • f) = a • (f : ℕ → β) :=
  rfl


@[simp, norm_cast]
theorem smul_apply (a : G) (f : CauSeq β abv) (i : ℕ) : (a • f) i = a • f i :=
  rfl


theorem const_smul (a : G) (x : β) : const (a • x) = a • const x :=
  rfl


instance : IsScalarTower G (CauSeq β abv) (CauSeq β abv) :=
  ⟨fun a f g => Subtype.ext <| smul_assoc a (f : ℕ → β) (g : ℕ → β)⟩


instance addGroup : AddGroup (CauSeq β abv) :=
  Function.Injective.addGroup Subtype.val Subtype.val_injective rfl coe_add coe_neg coe_sub
    (fun _ _ => coe_smul _ _) fun _ _ => coe_smul _ _


instance instNatCast : NatCast (CauSeq β abv) := ⟨fun n => const n⟩


instance instIntCast : IntCast (CauSeq β abv) := ⟨fun n => const n⟩


instance addGroupWithOne : AddGroupWithOne (CauSeq β abv) :=
  Function.Injective.addGroupWithOne Subtype.val Subtype.val_injective rfl rfl
  coe_add coe_neg coe_sub
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        ⊢ ∀ (n : Nat) (x : CauSeq β abv), Eq (↑(HSMul.hSMul n x)) (HSMul.hSMul n ↑x)
      -/
  (by intros; rfl)
              /-
                🎉 no goals
              -/
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        ⊢ ∀ (n : Int) (x : CauSeq β abv), Eq (↑(HSMul.hSMul n x)) (HSMul.hSMul n ↑x)
      -/
  (by intros; rfl)
              /-
                🎉 no goals
              -/
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        ⊢ ∀ (n : Nat), Eq ↑↑n ↑n
      -/
  (by intros; rfl)
              /-
                🎉 no goals
              -/
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        ⊢ ∀ (n : Int), Eq ↑↑n ↑n
      -/
  (by intros; rfl)
              /-
                🎉 no goals
              -/


instance : Pow (CauSeq β abv) ℕ :=
  ⟨fun f n =>
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  inst✝² : LinearOrderedField α
                                                  inst✝¹ : Ring β
                                                  abv : β → α
                                                  inst✝ : IsAbsoluteValue abv
                                                  f : CauSeq β abv
                                                  n : Nat
                                                  ⊢ ∀ (i : Nat), Eq (↑(npowRec n f) i) (HPow.hPow (↑f i) n)
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (ofEq (npowRec n f) fun i => f i ^ n) <| by induction n <;> simp [*, npowRec, pow_succ]⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp, norm_cast]
theorem coe_pow (f : CauSeq β abv) (n : ℕ) : ⇑(f ^ n) = (f : ℕ → β) ^ n :=
  rfl


@[simp, norm_cast]
theorem pow_apply (f : CauSeq β abv) (n i : ℕ) : (f ^ n) i = f i ^ n :=
  rfl


theorem const_pow (x : β) (n : ℕ) : const (x ^ n) = const x ^ n :=
  rfl


instance ring : Ring (CauSeq β abv) :=
  Function.Injective.ring Subtype.val Subtype.val_injective rfl rfl coe_add coe_mul coe_neg coe_sub
    (fun _ _ => coe_smul _ _) (fun _ _ => coe_smul _ _) coe_pow (fun _ => rfl) fun _ => rfl


instance {β : Type*} [CommRing β] {abv : β → α} [IsAbsoluteValue abv] : CommRing (CauSeq β abv) :=
  { CauSeq.ring with
                                           /-
                                             α : Type u_1
                                             β✝ : Type u_2
                                             inst✝⁴ : LinearOrderedField α
                                             inst✝³ : Ring β✝
                                             abv✝ : β✝ → α
                                             inst✝² : IsAbsoluteValue abv✝
                                             β : Type u_3
                                             inst✝¹ : CommRing β
                                             abv : β → α
                                             inst✝ : IsAbsoluteValue abv
                                             a b : CauSeq β abv
                                             n : Nat
                                             ⊢ Eq (↑(HMul.hMul a b) n) (↑(HMul.hMul b a) n)
                                           -/
    mul_comm := fun a b => ext fun n => by simp [mul_left_comm, mul_comm] }
                                           /-
                                             🎉 no goals
                                           -/


/-- `LimZero f` holds when `f` approaches 0. -/
def LimZero {abv : β → α} (f : CauSeq β abv) : Prop :=
  ∀ ε > 0, ∃ i, ∀ j ≥ i, abv (f j) < ε


theorem add_limZero {f g : CauSeq β abv} (hf : LimZero f) (hg : LimZero g) : LimZero (f + g)
  | ε, ε0 =>
    (exists_forall_ge_and (hf _ <| half_pos ε0) (hg _ <| half_pos ε0)).imp fun _ H j ij => by
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f g : CauSeq β abv
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abv (↑f j)) (HDiv.hDiv ε 2)) (LT.lt  …
        j : Nat
        ij : GE.ge j x✝
        ⊢ LT.lt (abv (↑(HAdd.hAdd f g) j)) ε
      -/
      let ⟨H₁, H₂⟩ := H _ ij
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f g : CauSeq β abv
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abv (↑f j)) (HDiv.hDiv ε 2)) (LT.lt  …
        j : Nat
        ij : GE.ge j x✝
        H₁ : LT.lt (abv (↑f j)) (HDiv.hDiv ε 2)
        H₂ : LT.lt (abv (↑g j)) (HDiv.hDiv ε 2)
        ⊢ LT.lt (abv (↑(HAdd.hAdd f g) j)) ε
      -/
      simpa [add_halves ε] using lt_of_le_of_lt (abv_add abv _ _) (add_lt_add H₁ H₂)
      /-
        🎉 no goals
      -/


theorem mul_limZero_right (f : CauSeq β abv) {g} (hg : LimZero g) : LimZero (f * g)
  | ε, ε0 =>
    let ⟨F, F0, hF⟩ := f.bounded' 0
    (hg _ <| div_pos ε0 F0).imp fun _ H j ij => by
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f g : CauSeq β abv
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        F : α
        F0 : GT.gt F 0
        hF : ∀ (i : Nat), LT.lt (abv (↑f i)) F
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → LT.lt (abv (↑g j)) (HDiv.hDiv ε F)
        j : Nat
        ij : GE.ge j x✝
        ⊢ LT.lt (abv (↑(HMul.hMul f g) j)) ε
      -/
      have := mul_lt_mul' (le_of_lt <| hF j) (H _ ij) (abv_nonneg abv _) F0
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f g : CauSeq β abv
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        F : α
        F0 : GT.gt F 0
        hF : ∀ (i : Nat), LT.lt (abv (↑f i)) F
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → LT.lt (abv (↑g j)) (HDiv.hDiv ε F)
        j : Nat
        ij : GE.ge j x✝
        this : LT.lt (HMul.hMul (abv (↑f j)) (abv (↑g j))) (HMul.hMul F (HDiv.hDiv ε F))
        ⊢ LT.lt (abv (↑(HMul.hMul f g) j)) ε
      -/
      rwa [mul_comm F, div_mul_cancel₀ _ (ne_of_gt F0), ← abv_mul] at this
      /-
        🎉 no goals
      -/


theorem mul_limZero_left {f} (g : CauSeq β abv) (hg : LimZero f) : LimZero (f * g)
  | ε, ε0 =>
    let ⟨G, G0, hG⟩ := g.bounded' 0
    (hg _ <| div_pos ε0 G0).imp fun _ H j ij => by
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f g : CauSeq β abv
        hg : f.LimZero
        ε : α
        ε0 : GT.gt ε 0
        G : α
        G0 : GT.gt G 0
        hG : ∀ (i : Nat), LT.lt (abv (↑g i)) G
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → LT.lt (abv (↑f j)) (HDiv.hDiv ε G)
        j : Nat
        ij : GE.ge j x✝
        ⊢ LT.lt (abv (↑(HMul.hMul f g) j)) ε
      -/
      have := mul_lt_mul'' (H _ ij) (hG j) (abv_nonneg abv _) (abv_nonneg abv _)
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f g : CauSeq β abv
        hg : f.LimZero
        ε : α
        ε0 : GT.gt ε 0
        G : α
        G0 : GT.gt G 0
        hG : ∀ (i : Nat), LT.lt (abv (↑g i)) G
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → LT.lt (abv (↑f j)) (HDiv.hDiv ε G)
        j : Nat
        ij : GE.ge j x✝
        this : LT.lt (HMul.hMul (abv (↑f j)) (abv (↑g j))) (HMul.hMul (HDiv.hDiv ε G) G)
        ⊢ LT.lt (abv (↑(HMul.hMul f g) j)) ε
      -/
      rwa [div_mul_cancel₀ _ (ne_of_gt G0), ← abv_mul] at this
      /-
        🎉 no goals
      -/


theorem neg_limZero {f : CauSeq β abv} (hf : LimZero f) : LimZero (-f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : f.LimZero
    ⊢ (Neg.neg f).LimZero
  -/
  rw [← neg_one_mul f]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : f.LimZero
    ⊢ (HMul.hMul (-1) f).LimZero
  -/
  exact mul_limZero_right _ hf
  /-
    🎉 no goals
  -/


theorem sub_limZero {f g : CauSeq β abv} (hf : LimZero f) (hg : LimZero g) : LimZero (f - g) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : f.LimZero
    hg : g.LimZero
    ⊢ (HSub.hSub f g).LimZero
  -/
  simpa only [sub_eq_add_neg] using add_limZero hf (neg_limZero hg)
  /-
    🎉 no goals
  -/


theorem limZero_sub_rev {f g : CauSeq β abv} (hfg : LimZero (f - g)) : LimZero (g - f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hfg : (HSub.hSub f g).LimZero
    ⊢ (HSub.hSub g f).LimZero
  -/
  simpa using neg_limZero hfg
  /-
    🎉 no goals
  -/


theorem zero_limZero : LimZero (0 : CauSeq β abv)
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 inst✝² : LinearOrderedField α
                                 inst✝¹ : Ring β
                                 abv : β → α
                                 inst✝ : IsAbsoluteValue abv
                                 ε : α
                                 ε0 : GT.gt ε 0
                                 j : Nat
                                 x✝ : GE.ge j 0
                                 ⊢ LT.lt (abv (↑0 j)) ε
                               -/
  | ε, ε0 => ⟨0, fun j _ => by simpa [abv_zero abv] using ε0⟩
                               /-
                                 🎉 no goals
                               -/


theorem const_limZero {x : β} : LimZero (const x) ↔ x = 0 :=
  ⟨fun H =>
    (abv_eq_zero abv).1 <|
      (eq_of_le_of_forall_le_of_dense (abv_nonneg abv _)) fun _ ε0 =>
        let ⟨_, hi⟩ := H _ ε0
        le_of_lt <| hi _ le_rfl,
    fun e => e.symm ▸ zero_limZero⟩


instance equiv : Setoid (CauSeq β abv) :=
  ⟨fun f g => LimZero (f - g),
                 /-
                   α : Type u_1
                   β : Type u_2
                   inst✝² : LinearOrderedField α
                   inst✝¹ : Ring β
                   abv : β → α
                   inst✝ : IsAbsoluteValue abv
                   f : CauSeq β abv
                   ⊢ (HSub.hSub f f).LimZero
                 -/
    ⟨fun f => by simp [zero_limZero],
                 /-
                   🎉 no goals
                 -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝² : LinearOrderedField α
                       inst✝¹ : Ring β
                       abv : β → α
                       inst✝ : IsAbsoluteValue abv
                       x✝ y✝ : CauSeq β abv
                       f : (HSub.hSub x✝ y✝).LimZero
                       ε : α
                       hε : GT.gt ε 0
                       ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (↑(HSub.hSub y✝ x✝) j)) ε
                     -/
    fun f ε hε => by simpa using neg_limZero f ε hε,
                     /-
                       🎉 no goals
                     -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      inst✝² : LinearOrderedField α
                      inst✝¹ : Ring β
                      abv : β → α
                      inst✝ : IsAbsoluteValue abv
                      x✝ y✝ z✝ : CauSeq β abv
                      fg : (HSub.hSub x✝ y✝).LimZero
                      gh : (HSub.hSub y✝ z✝).LimZero
                      ⊢ (HSub.hSub x✝ z✝).LimZero
                    -/
    fun fg gh => by simpa using add_limZero fg gh⟩⟩
                    /-
                      🎉 no goals
                    -/


theorem add_equiv_add {f1 f2 g1 g2 : CauSeq β abv} (hf : f1 ≈ f2) (hg : g1 ≈ g2) :
                            /-
                              α : Type u_1
                              β : Type u_2
                              inst✝² : LinearOrderedField α
                              inst✝¹ : Ring β
                              abv : β → α
                              inst✝ : IsAbsoluteValue abv
                              f1 f2 g1 g2 : CauSeq β abv
                              hf : HasEquiv.Equiv f1 f2
                              hg : HasEquiv.Equiv g1 g2
                              ⊢ HasEquiv.Equiv (HAdd.hAdd f1 g1) (HAdd.hAdd f2 g2)
                            -/
    f1 + g1 ≈ f2 + g2 := by simpa only [← add_sub_add_comm] using add_limZero hf hg
                            /-
                              🎉 no goals
                            -/


theorem neg_equiv_neg {f g : CauSeq β abv} (hf : f ≈ g) : -f ≈ -g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : HasEquiv.Equiv f g
    ⊢ HasEquiv.Equiv (Neg.neg f) (Neg.neg g)
  -/
  simpa only [neg_sub'] using neg_limZero hf
  /-
    🎉 no goals
  -/


theorem sub_equiv_sub {f1 f2 g1 g2 : CauSeq β abv} (hf : f1 ≈ f2) (hg : g1 ≈ g2) :
                            /-
                              α : Type u_1
                              β : Type u_2
                              inst✝² : LinearOrderedField α
                              inst✝¹ : Ring β
                              abv : β → α
                              inst✝ : IsAbsoluteValue abv
                              f1 f2 g1 g2 : CauSeq β abv
                              hf : HasEquiv.Equiv f1 f2
                              hg : HasEquiv.Equiv g1 g2
                              ⊢ HasEquiv.Equiv (HSub.hSub f1 g1) (HSub.hSub f2 g2)
                            -/
    f1 - g1 ≈ f2 - g2 := by simpa only [sub_eq_add_neg] using add_equiv_add hf (neg_equiv_neg hg)
                            /-
                              🎉 no goals
                            -/


theorem equiv_def₃ {f g : CauSeq β abv} (h : f ≈ g) {ε : α} (ε0 : 0 < ε) :
    ∃ i, ∀ j ≥ i, ∀ k ≥ j, abv (f k - g j) < ε :=
  (exists_forall_ge_and (h _ <| half_pos ε0) (f.cauchy₃ <| half_pos ε0)).imp fun _ H j ij k jk => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Ring β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      f g : CauSeq β abv
      h : HasEquiv.Equiv f g
      ε : α
      ε0 : LT.lt 0 ε
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abv (↑(HSub.hSub f g) j)) (HDiv.hDiv …
      j : Nat
      ij : GE.ge j x✝
      k : Nat
      jk : GE.ge k j
      ⊢ LT.lt (abv (HSub.hSub (↑f k) (↑g j))) ε
    -/
    let ⟨h₁, h₂⟩ := H _ ij
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Ring β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      f g : CauSeq β abv
      h : HasEquiv.Equiv f g
      ε : α
      ε0 : LT.lt 0 ε
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abv (↑(HSub.hSub f g) j)) (HDiv.hDiv …
      j : Nat
      ij : GE.ge j x✝
      k : Nat
      jk : GE.ge k j
      h₁ : LT.lt (abv (↑(HSub.hSub f g) j)) (HDiv.hDiv ε 2)
      h₂ : ∀ (k : Nat), GE.ge k j → LT.lt (abv (HSub.hSub (↑f k) (↑f j))) (HDiv.hDiv …
      ⊢ LT.lt (abv (HSub.hSub (↑f k) (↑g j))) ε
    -/
    have := lt_of_le_of_lt (abv_add abv (f j - g j) _) (add_lt_add h₁ (h₂ _ jk))
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Ring β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      f g : CauSeq β abv
      h : HasEquiv.Equiv f g
      ε : α
      ε0 : LT.lt 0 ε
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abv (↑(HSub.hSub f g) j)) (HDiv.hDiv …
      j : Nat
      ij : GE.ge j x✝
      k : Nat
      jk : GE.ge k j
      h₁ : LT.lt (abv (↑(HSub.hSub f g) j)) (HDiv.hDiv ε 2)
      h₂ : ∀ (k : Nat), GE.ge k j → LT.lt (abv (HSub.hSub (↑f k) (↑f j))) (HDiv.hDiv …
      this : LT.lt (abv (HAdd.hAdd (HSub.hSub (↑f j) (↑g j)) (HSub.hSub (↑f k) (↑f j …
      ⊢ LT.lt (abv (HSub.hSub (↑f k) (↑g j))) ε
    -/
    rwa [sub_add_sub_cancel', add_halves] at this
    /-
      🎉 no goals
    -/


theorem limZero_congr {f g : CauSeq β abv} (h : f ≈ g) : LimZero f ↔ LimZero g :=
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝² : LinearOrderedField α
                 inst✝¹ : Ring β
                 abv : β → α
                 inst✝ : IsAbsoluteValue abv
                 f g : CauSeq β abv
                 h : HasEquiv.Equiv f g
                 l : f.LimZero
                 ⊢ g.LimZero
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun l => by simpa using add_limZero (Setoid.symm h) l, fun l => by simpa using add_limZero h l⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem abv_pos_of_not_limZero {f : CauSeq β abv} (hf : ¬LimZero f) :
    ∃ K > 0, ∃ i, ∀ j ≥ i, K ≤ abv (f j) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    ⊢ Exists fun K => And (GT.gt K 0) (Exists fun i => ∀ (j : Nat), GE.ge j i → LE …
  -/
  haveI := Classical.propDecidable
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this : (a : Prop) → Decidable a
    ⊢ Exists fun K => And (GT.gt K 0) (Exists fun i => ∀ (j : Nat), GE.ge j i → LE …
  -/
  by_contra nk
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this : (a : Prop) → Decidable a
    nk : Not (Exists fun K => And (GT.gt K 0) (Exists fun i => ∀ (j : Nat), GE.ge  …
    ⊢ False
  -/
  refine hf fun ε ε0 => ?_
  simp? [not_forall] at nk says
    simp only [gt_iff_lt, ge_iff_le, not_exists, not_and, not_forall, Classical.not_imp,
      not_le] at nk
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this : (a : Prop) → Decidable a
    ε : α
    ε0 : GT.gt ε 0
    nk : ∀ (x : α), LT.lt 0 x → ∀ (x_1 : Nat), Exists fun x_2 => Exists fun h => L …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (↑f j)) ε
  -/
  cases' f.cauchy₃ (half_pos ε0) with i hi
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this : (a : Prop) → Decidable a
    ε : α
    ε0 : GT.gt ε 0
    nk : ∀ (x : α), LT.lt 0 x → ∀ (x_1 : Nat), Exists fun x_2 => Exists fun h => L …
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (abv (HSub.hSub ( …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (↑f j)) ε
  -/
  rcases nk _ (half_pos ε0) i with ⟨j, ij, hj⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this : (a : Prop) → Decidable a
    ε : α
    ε0 : GT.gt ε 0
    nk : ∀ (x : α), LT.lt 0 x → ∀ (x_1 : Nat), Exists fun x_2 => Exists fun h => L …
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (abv (HSub.hSub ( …
    j : Nat
    ij : LE.le i j
    hj : LT.lt (abv (↑f j)) (HDiv.hDiv ε 2)
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (↑f j)) ε
  -/
  refine ⟨j, fun k jk => ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this : (a : Prop) → Decidable a
    ε : α
    ε0 : GT.gt ε 0
    nk : ∀ (x : α), LT.lt 0 x → ∀ (x_1 : Nat), Exists fun x_2 => Exists fun h => L …
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (abv (HSub.hSub ( …
    j : Nat
    ij : LE.le i j
    hj : LT.lt (abv (↑f j)) (HDiv.hDiv ε 2)
    k : Nat
    jk : GE.ge k j
    ⊢ LT.lt (abv (↑f k)) ε
  -/
  have := lt_of_le_of_lt (abv_add abv _ _) (add_lt_add (hi j ij k jk) hj)
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not f.LimZero
    this✝ : (a : Prop) → Decidable a
    ε : α
    ε0 : GT.gt ε 0
    nk : ∀ (x : α), LT.lt 0 x → ∀ (x_1 : Nat), Exists fun x_2 => Exists fun h => L …
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (abv (HSub.hSub ( …
    j : Nat
    ij : LE.le i j
    hj : LT.lt (abv (↑f j)) (HDiv.hDiv ε 2)
    k : Nat
    jk : GE.ge k j
    this : LT.lt (abv (HAdd.hAdd (HSub.hSub (↑f k) (↑f j)) (↑f j))) (HAdd.hAdd (HD …
    ⊢ LT.lt (abv (↑f k)) ε
  -/
  rwa [sub_add_cancel, add_halves] at this
  /-
    🎉 no goals
  -/


theorem of_near (f : ℕ → β) (g : CauSeq β abv) (h : ∀ ε > 0, ∃ i, ∀ j ≥ i, abv (f j - g j) < ε) :
    IsCauSeq abv f
  | ε, ε0 =>
    let ⟨i, hi⟩ := exists_forall_ge_and (h _ (half_pos <| half_pos ε0)) (g.cauchy₃ <| half_pos ε0)
    ⟨i, fun j ij => by
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f : Nat → β
        g : CauSeq β abv
        h : ∀ (ε : α), GT.gt ε 0 → Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv …
        ε : α
        ε0 : GT.gt ε 0
        i : Nat
        hi : ∀ (j : Nat), GE.ge j i → And (LT.lt (abv (HSub.hSub (f j) (↑g j))) (HDiv. …
        j : Nat
        ij : GE.ge j i
        ⊢ LT.lt (abv (HSub.hSub (f j) (f i))) ε
      -/
      cases' hi _ le_rfl with h₁ h₂; rw [abv_sub abv] at h₁
      /-
        case intro
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f : Nat → β
        g : CauSeq β abv
        h : ∀ (ε : α), GT.gt ε 0 → Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv …
        ε : α
        ε0 : GT.gt ε 0
        i : Nat
        hi : ∀ (j : Nat), GE.ge j i → And (LT.lt (abv (HSub.hSub (f j) (↑g j))) (HDiv. …
        j : Nat
        ij : GE.ge j i
        h₁ : LT.lt (abv (HSub.hSub (↑g i) (f i))) (HDiv.hDiv (HDiv.hDiv ε 2) 2)
        h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abv (HSub.hSub (↑g k) (↑g i))) (HDiv.hDiv …
        ⊢ LT.lt (abv (HSub.hSub (f j) (f i))) ε
      -/
      have := lt_of_le_of_lt (abv_add abv _ _) (add_lt_add (hi _ ij).1 h₁)
      /-
        case intro
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f : Nat → β
        g : CauSeq β abv
        h : ∀ (ε : α), GT.gt ε 0 → Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv …
        ε : α
        ε0 : GT.gt ε 0
        i : Nat
        hi : ∀ (j : Nat), GE.ge j i → And (LT.lt (abv (HSub.hSub (f j) (↑g j))) (HDiv. …
        j : Nat
        ij : GE.ge j i
        h₁ : LT.lt (abv (HSub.hSub (↑g i) (f i))) (HDiv.hDiv (HDiv.hDiv ε 2) 2)
        h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abv (HSub.hSub (↑g k) (↑g i))) (HDiv.hDiv …
        this : LT.lt (abv (HAdd.hAdd (HSub.hSub (f j) (↑g j)) (HSub.hSub (↑g i) (f i)) …
        ⊢ LT.lt (abv (HSub.hSub (f j) (f i))) ε
      -/
      have := lt_of_le_of_lt (abv_add abv _ _) (add_lt_add this (h₂ _ ij))
      /-
        case intro
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : Ring β
        abv : β → α
        inst✝ : IsAbsoluteValue abv
        f : Nat → β
        g : CauSeq β abv
        h : ∀ (ε : α), GT.gt ε 0 → Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv …
        ε : α
        ε0 : GT.gt ε 0
        i : Nat
        hi : ∀ (j : Nat), GE.ge j i → And (LT.lt (abv (HSub.hSub (f j) (↑g j))) (HDiv. …
        j : Nat
        ij : GE.ge j i
        h₁ : LT.lt (abv (HSub.hSub (↑g i) (f i))) (HDiv.hDiv (HDiv.hDiv ε 2) 2)
        h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abv (HSub.hSub (↑g k) (↑g i))) (HDiv.hDiv …
        this✝ : LT.lt (abv (HAdd.hAdd (HSub.hSub (f j) (↑g j)) (HSub.hSub (↑g i) (f i) …
        this : LT.lt (abv (HAdd.hAdd (HAdd.hAdd (HSub.hSub (f j) (↑g j)) (HSub.hSub (↑ …
        ⊢ LT.lt (abv (HSub.hSub (f j) (f i))) ε
      -/
      rwa [add_halves, add_halves, add_right_comm, sub_add_sub_cancel, sub_add_sub_cancel] at this⟩
      /-
        🎉 no goals
      -/


theorem not_limZero_of_not_congr_zero {f : CauSeq _ abv} (hf : ¬f ≈ 0) : ¬LimZero f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    ⊢ Not f.LimZero
  -/
  intro h
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    h : f.LimZero
    ⊢ False
  -/
  have : LimZero (f - 0) := by simp [h]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    h : f.LimZero
    this : (HSub.hSub f 0).LimZero
    ⊢ False
  -/
  exact hf this
  /-
    🎉 no goals
  -/


theorem mul_equiv_zero (g : CauSeq _ abv) {f : CauSeq _ abv} (hf : f ≈ 0) : g * f ≈ 0 :=
  have : LimZero (f - 0) := hf
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        inst✝² : LinearOrderedField α
                                                        inst✝¹ : Ring β
                                                        abv : β → α
                                                        inst✝ : IsAbsoluteValue abv
                                                        g f : CauSeq β abv
                                                        hf : HasEquiv.Equiv f 0
                                                        this : (HSub.hSub f 0).LimZero
                                                        ⊢ f.LimZero
                                                      -/
  have : LimZero (g * f) := mul_limZero_right _ <| by simpa
                                                      /-
                                                        🎉 no goals
                                                      -/
                              /-
                                α : Type u_1
                                β : Type u_2
                                inst✝² : LinearOrderedField α
                                inst✝¹ : Ring β
                                abv : β → α
                                inst✝ : IsAbsoluteValue abv
                                g f : CauSeq β abv
                                hf : HasEquiv.Equiv f 0
                                this✝ : (HSub.hSub f 0).LimZero
                                this : (HMul.hMul g f).LimZero
                                ⊢ (HSub.hSub (HMul.hMul g f) 0).LimZero
                              -/
  show LimZero (g * f - 0) by simpa
                              /-
                                🎉 no goals
                              -/


theorem mul_equiv_zero' (g : CauSeq _ abv) {f : CauSeq _ abv} (hf : f ≈ 0) : f * g ≈ 0 :=
  have : LimZero (f - 0) := hf
                                                     /-
                                                       α : Type u_1
                                                       β : Type u_2
                                                       inst✝² : LinearOrderedField α
                                                       inst✝¹ : Ring β
                                                       abv : β → α
                                                       inst✝ : IsAbsoluteValue abv
                                                       g f : CauSeq β abv
                                                       hf : HasEquiv.Equiv f 0
                                                       this : (HSub.hSub f 0).LimZero
                                                       ⊢ f.LimZero
                                                     -/
  have : LimZero (f * g) := mul_limZero_left _ <| by simpa
                                                     /-
                                                       🎉 no goals
                                                     -/
                              /-
                                α : Type u_1
                                β : Type u_2
                                inst✝² : LinearOrderedField α
                                inst✝¹ : Ring β
                                abv : β → α
                                inst✝ : IsAbsoluteValue abv
                                g f : CauSeq β abv
                                hf : HasEquiv.Equiv f 0
                                this✝ : (HSub.hSub f 0).LimZero
                                this : (HMul.hMul f g).LimZero
                                ⊢ (HSub.hSub (HMul.hMul f g) 0).LimZero
                              -/
  show LimZero (f * g - 0) by simpa
                              /-
                                🎉 no goals
                              -/


theorem mul_not_equiv_zero {f g : CauSeq _ abv} (hf : ¬f ≈ 0) (hg : ¬g ≈ 0) : ¬f * g ≈ 0 :=
  fun (this : LimZero (f * g - 0)) => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this : (HSub.hSub (HMul.hMul f g) 0).LimZero
    ⊢ False
  -/
  have hlz : LimZero (f * g) := by simpa
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    ⊢ False
  -/
  have hf' : ¬LimZero f := by simpa using show ¬LimZero (f - 0) from hf
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    ⊢ False
  -/
  have hg' : ¬LimZero g := by simpa using show ¬LimZero (g - 0) from hg
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    ⊢ False
  -/
  rcases abv_pos_of_not_limZero hf' with ⟨a1, ha1, N1, hN1⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    ⊢ False
  -/
  rcases abv_pos_of_not_limZero hg' with ⟨a2, ha2, N2, hN2⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    ⊢ False
  -/
  have : 0 < a1 * a2 := mul_pos ha1 ha2
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    ⊢ False
  -/
  cases' hlz _ this with N hN
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    ⊢ False
  -/
  let i := max N (max N1 N2)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    ⊢ False
  -/
  have hN' := hN i (le_max_left _ _)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    hN' : LT.lt (abv (↑(HMul.hMul f g) i)) (HMul.hMul a1 a2)
    ⊢ False
  -/
  have hN1' := hN1 i (le_trans (le_max_left _ _) (le_max_right _ _))
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    hN' : LT.lt (abv (↑(HMul.hMul f g) i)) (HMul.hMul a1 a2)
    hN1' : LE.le a1 (abv (↑f i))
    ⊢ False
  -/
  have hN1' := hN2 i (le_trans (le_max_right _ _) (le_max_right _ _))
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    hN' : LT.lt (abv (↑(HMul.hMul f g) i)) (HMul.hMul a1 a2)
    hN1'✝ : LE.le a1 (abv (↑f i))
    hN1' : LE.le a2 (abv (↑g i))
    ⊢ False
  -/
  apply not_le_of_lt hN'
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    hN' : LT.lt (abv (↑(HMul.hMul f g) i)) (HMul.hMul a1 a2)
    hN1'✝ : LE.le a1 (abv (↑f i))
    hN1' : LE.le a2 (abv (↑g i))
    ⊢ LE.le (HMul.hMul a1 a2) (abv (↑(HMul.hMul f g) i))
  -/
  change _ ≤ abv (_ * _)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    hN' : LT.lt (abv (↑(HMul.hMul f g) i)) (HMul.hMul a1 a2)
    hN1'✝ : LE.le a1 (abv (↑f i))
    hN1' : LE.le a2 (abv (↑g i))
    ⊢ LE.le (HMul.hMul a1 a2) (abv (HMul.hMul (↑f i) (↑g i)))
  -/
  rw [abv_mul abv]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : CauSeq β abv
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    this✝ : (HSub.hSub (HMul.hMul f g) 0).LimZero
    hlz : (HMul.hMul f g).LimZero
    hf' : Not f.LimZero
    hg' : Not g.LimZero
    a1 : α
    ha1 : GT.gt a1 0
    N1 : Nat
    hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le a1 (abv (↑f j))
    a2 : α
    ha2 : GT.gt a2 0
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → LE.le a2 (abv (↑g j))
    this : LT.lt 0 (HMul.hMul a1 a2)
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (abv (↑(HMul.hMul f g) j)) (HMul.hMul a1 a2)
    i : Nat := Max.max N (Max.max N1 N2)
    hN' : LT.lt (abv (↑(HMul.hMul f g) i)) (HMul.hMul a1 a2)
    hN1'✝ : LE.le a1 (abv (↑f i))
    hN1' : LE.le a2 (abv (↑g i))
    ⊢ LE.le (HMul.hMul a1 a2) (HMul.hMul (abv (↑f i)) (abv (↑g i)))
  -/
  gcongr
  /-
    🎉 no goals
  -/


theorem const_equiv {x y : β} : const x ≈ const y ↔ x = y :=
                        /-
                          α : Type u_1
                          β : Type u_2
                          inst✝² : LinearOrderedField α
                          inst✝¹ : Ring β
                          abv : β → α
                          inst✝ : IsAbsoluteValue abv
                          x y : β
                          ⊢ Iff (HSub.hSub (CauSeq.const abv x) (CauSeq.const abv y)).LimZero (Eq x y)
                        -/
  show LimZero _ ↔ _ by rw [← const_sub, const_limZero, sub_eq_zero]
                        /-
                          🎉 no goals
                        -/


theorem mul_equiv_mul {f1 f2 g1 g2 : CauSeq β abv} (hf : f1 ≈ f2) (hg : g1 ≈ g2) :
    f1 * g1 ≈ f2 * g2 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f1 f2 g1 g2 : CauSeq β abv
    hf : HasEquiv.Equiv f1 f2
    hg : HasEquiv.Equiv g1 g2
    ⊢ HasEquiv.Equiv (HMul.hMul f1 g1) (HMul.hMul f2 g2)
  -/
  change LimZero (f1 * g1 - f2 * g2)
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f1 f2 g1 g2 : CauSeq β abv
    hf : HasEquiv.Equiv f1 f2
    hg : HasEquiv.Equiv g1 g2
    ⊢ (HSub.hSub (HMul.hMul f1 g1) (HMul.hMul f2 g2)).LimZero
  -/
  convert add_limZero (mul_limZero_left g1 hf) (mul_limZero_right f2 hg) using 1
  /-
    case h.e'_6
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f1 f2 g1 g2 : CauSeq β abv
    hf : HasEquiv.Equiv f1 f2
    hg : HasEquiv.Equiv g1 g2
    ⊢ Eq (HSub.hSub (HMul.hMul f1 g1) (HMul.hMul f2 g2)) (HAdd.hAdd (HMul.hMul (HS …
  -/
  rw [mul_sub, sub_mul]
  -- Porting note: doesn't work with `rw`, but did in Lean 3
  /-
    case h.e'_6
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f1 f2 g1 g2 : CauSeq β abv
    hf : HasEquiv.Equiv f1 f2
    hg : HasEquiv.Equiv g1 g2
    ⊢ Eq (HSub.hSub (HMul.hMul f1 g1) (HMul.hMul f2 g2)) (HAdd.hAdd (HSub.hSub (HM …
  -/
  exact (sub_add_sub_cancel (f1*g1) (f2*g1) (f2*g2)).symm
  /-
    🎉 no goals
  -/
  -- Porting note: was
  /-
  simpa only [mul_sub, sub_mul, sub_add_sub_cancel] using
    add_lim_zero (mul_limZero_left g1 hf) (mul_limZero_right f2 hg)
  -/


theorem smul_equiv_smul {G : Type*} [SMul G β] [IsScalarTower G β β] {f1 f2 : CauSeq β abv} (c : G)
    (hf : f1 ≈ f2) : c • f1 ≈ c • f2 := by
  simpa [const_smul, smul_one_mul _ _] using
    mul_equiv_mul (const_equiv.mpr <| Eq.refl <| c • (1 : β)) hf


theorem pow_equiv_pow {f1 f2 : CauSeq β abv} (hf : f1 ≈ f2) (n : ℕ) : f1 ^ n ≈ f2 ^ n := by
  induction n with
  | zero => simp only [pow_zero, Setoid.refl]
  | succ n ih => simpa only [pow_succ'] using mul_equiv_mul hf ih


theorem one_not_equiv_zero : ¬const abv 1 ≈ const abv 0 := fun h =>
  have : ∀ ε > 0, ∃ i, ∀ k, i ≤ k → abv (1 - 0) < ε := h
  have h1 : abv 1 ≤ 0 :=
    le_of_not_gt fun h2 : 0 < abv 1 =>
                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      inst✝³ : LinearOrderedField α
                                                                      inst✝² : Ring β
                                                                      inst✝¹ : IsDomain β
                                                                      abv : β → α
                                                                      inst✝ : IsAbsoluteValue abv
                                                                      h : HasEquiv.Equiv (CauSeq.const abv 1) (CauSeq.const abv 0)
                                                                      this : ∀ (ε : α), GT.gt ε 0 → Exists fun i => ∀ (k : Nat), LE.le i k → LT.lt ( …
                                                                      h2 : LT.lt 0 (abv 1)
                                                                      i : Nat
                                                                      hi : ∀ (k : Nat), LE.le i k → LT.lt (abv (HSub.hSub 1 0)) (abv 1)
                                                                      ⊢ LT.lt (abv 1) (abv 1)
                                                                    -/
      (Exists.elim (this _ h2)) fun i hi => lt_irrefl (abv 1) <| by simpa using hi _ le_rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  have h2 : 0 ≤ abv 1 := abv_nonneg abv _
  have : abv 1 = 0 := le_antisymm h1 h2
  have : (1 : β) = 0 := (abv_eq_zero abv).mp this
  absurd this one_ne_zero


theorem inv_aux {f : CauSeq β abv} (hf : ¬LimZero f) :
    ∀ ε > 0, ∃ i, ∀ j ≥ i, abv ((f j)⁻¹ - (f i)⁻¹) < ε
  | _, ε0 =>
    let ⟨_, K0, HK⟩ := abv_pos_of_not_limZero hf
    let ⟨_, δ0, Hδ⟩ := rat_inv_continuous_lemma abv ε0 K0
    let ⟨i, H⟩ := exists_forall_ge_and HK (f.cauchy₃ δ0)
    ⟨i, fun _ ij =>
      let ⟨iK, H'⟩ := H _ le_rfl
      Hδ (H _ ij).1 iK (H' _ ij)⟩


/-- Given a Cauchy sequence `f` with nonzero limit, create a Cauchy sequence with values equal to
the inverses of the values of `f`. -/
def inv (f : CauSeq β abv) (hf : ¬LimZero f) : CauSeq β abv :=
  ⟨_, inv_aux hf⟩


@[simp, norm_cast]
theorem coe_inv {f : CauSeq β abv} (hf) : ⇑(inv f hf) = (f : ℕ → β)⁻¹ :=
  rfl


@[simp, norm_cast]
theorem inv_apply {f : CauSeq β abv} (hf i) : inv f hf i = (f i)⁻¹ :=
  rfl


theorem inv_mul_cancel {f : CauSeq β abv} (hf) : inv f hf * f ≈ 1 := fun ε ε0 =>
  let ⟨K, K0, i, H⟩ := abv_pos_of_not_limZero hf
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝² : LinearOrderedField α
                       inst✝¹ : DivisionRing β
                       abv : β → α
                       inst✝ : IsAbsoluteValue abv
                       f : CauSeq β abv
                       hf : Not f.LimZero
                       ε : α
                       ε0 : GT.gt ε 0
                       K : α
                       K0 : GT.gt K 0
                       i : Nat
                       H : ∀ (j : Nat), GE.ge j i → LE.le K (abv (↑f j))
                       j : Nat
                       ij : GE.ge j i
                       ⊢ LT.lt (abv (↑(HSub.hSub (HMul.hMul (f.inv hf) f) 1) j)) ε
                     -/
  ⟨i, fun j ij => by simpa [(abv_pos abv).1 (lt_of_lt_of_le K0 (H _ ij)), abv_zero abv] using ε0⟩
                     /-
                       🎉 no goals
                     -/


theorem mul_inv_cancel {f : CauSeq β abv} (hf) : f * inv f hf ≈ 1 := fun ε ε0 =>
  let ⟨K, K0, i, H⟩ := abv_pos_of_not_limZero hf
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝² : LinearOrderedField α
                       inst✝¹ : DivisionRing β
                       abv : β → α
                       inst✝ : IsAbsoluteValue abv
                       f : CauSeq β abv
                       hf : Not f.LimZero
                       ε : α
                       ε0 : GT.gt ε 0
                       K : α
                       K0 : GT.gt K 0
                       i : Nat
                       H : ∀ (j : Nat), GE.ge j i → LE.le K (abv (↑f j))
                       j : Nat
                       ij : GE.ge j i
                       ⊢ LT.lt (abv (↑(HSub.hSub (HMul.hMul f (f.inv hf)) 1) j)) ε
                     -/
  ⟨i, fun j ij => by simpa [(abv_pos abv).1 (lt_of_lt_of_le K0 (H _ ij)), abv_zero abv] using ε0⟩
                     /-
                       🎉 no goals
                     -/


theorem const_inv {x : β} (hx : x ≠ 0) :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            inst✝² : LinearOrderedField α
                                            inst✝¹ : DivisionRing β
                                            abv : β → α
                                            inst✝ : IsAbsoluteValue abv
                                            x : β
                                            hx : Ne x 0
                                            ⊢ Not (CauSeq.const abv x).LimZero
                                          -/
    const abv x⁻¹ = inv (const abv x) (by rwa [const_limZero]) :=
                                          /-
                                            🎉 no goals
                                          -/
  rfl


/-- The constant Cauchy sequence -/
local notation "const" => const abs


/-- The entries of a positive Cauchy sequence eventually have a positive lower bound. -/
def Pos (f : CauSeq α abs) : Prop :=
  ∃ K > 0, ∃ i, ∀ j ≥ i, K ≤ f j


theorem not_limZero_of_pos {f : CauSeq α abs} : Pos f → ¬LimZero f
  | ⟨_, F0, hF⟩, H =>
    let ⟨_, h⟩ := exists_forall_ge_and hF (H _ F0)
    let ⟨h₁, h₂⟩ := h _ le_rfl
    not_lt_of_le h₁ (abs_lt.1 h₂).2


theorem const_pos {x : α} : Pos (const x) ↔ 0 < x :=
  ⟨fun ⟨_, K0, _, h⟩ => lt_of_lt_of_le K0 (h _ le_rfl), fun h => ⟨x, h, 0, fun _ _ => le_rfl⟩⟩


theorem add_pos {f g : CauSeq α abs} : Pos f → Pos g → Pos (f + g)
  | ⟨_, F0, hF⟩, ⟨_, G0, hG⟩ =>
    let ⟨i, h⟩ := exists_forall_ge_and hF hG
    ⟨_, _root_.add_pos F0 G0, i, fun _ ij =>
      let ⟨h₁, h₂⟩ := h _ ij
      add_le_add h₁ h₂⟩


theorem pos_add_limZero {f g : CauSeq α abs} : Pos f → LimZero g → Pos (f + g)
  | ⟨F, F0, hF⟩, H =>
    let ⟨i, h⟩ := exists_forall_ge_and hF (H _ (half_pos F0))
    ⟨_, half_pos F0, i, fun j ij => by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        F : α
        F0 : GT.gt F 0
        hF : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le F (↑f j)
        H : g.LimZero
        i : Nat
        h : ∀ (j : Nat), GE.ge j i → And (LE.le F (↑f j)) (LT.lt (abs (↑g j)) (HDiv.hD …
        j : Nat
        ij : GE.ge j i
        ⊢ LE.le (HDiv.hDiv F 2) (↑(HAdd.hAdd f g) j)
      -/
      cases' h j ij with h₁ h₂
      /-
        case intro
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        F : α
        F0 : GT.gt F 0
        hF : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le F (↑f j)
        H : g.LimZero
        i : Nat
        h : ∀ (j : Nat), GE.ge j i → And (LE.le F (↑f j)) (LT.lt (abs (↑g j)) (HDiv.hD …
        j : Nat
        ij : GE.ge j i
        h₁ : LE.le F (↑f j)
        h₂ : LT.lt (abs (↑g j)) (HDiv.hDiv F 2)
        ⊢ LE.le (HDiv.hDiv F 2) (↑(HAdd.hAdd f g) j)
      -/
      have := add_le_add h₁ (le_of_lt (abs_lt.1 h₂).1)
      /-
        case intro
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        F : α
        F0 : GT.gt F 0
        hF : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le F (↑f j)
        H : g.LimZero
        i : Nat
        h : ∀ (j : Nat), GE.ge j i → And (LE.le F (↑f j)) (LT.lt (abs (↑g j)) (HDiv.hD …
        j : Nat
        ij : GE.ge j i
        h₁ : LE.le F (↑f j)
        h₂ : LT.lt (abs (↑g j)) (HDiv.hDiv F 2)
        this : LE.le (HAdd.hAdd F (Neg.neg (HDiv.hDiv F 2))) (HAdd.hAdd (↑f j) (↑g j))
        ⊢ LE.le (HDiv.hDiv F 2) (↑(HAdd.hAdd f g) j)
      -/
      rwa [← sub_eq_add_neg, sub_self_div_two] at this⟩
      /-
        🎉 no goals
      -/


protected theorem mul_pos {f g : CauSeq α abs} : Pos f → Pos g → Pos (f * g)
  | ⟨_, F0, hF⟩, ⟨_, G0, hG⟩ =>
    let ⟨i, h⟩ := exists_forall_ge_and hF hG
    ⟨_, mul_pos F0 G0, i, fun _ ij =>
      let ⟨h₁, h₂⟩ := h _ ij
      mul_le_mul h₁ h₂ (le_of_lt G0) (le_trans (le_of_lt F0) h₁)⟩


theorem trichotomy (f : CauSeq α abs) : Pos f ∨ LimZero f ∨ Pos (-f) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    f : CauSeq α abs
    ⊢ Or f.Pos (Or f.LimZero (Neg.neg f).Pos)
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  cases' Classical.em (LimZero f) with h h <;> simp [*]
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrderedField α
    f : CauSeq α abs
    h : Not f.LimZero
    ⊢ Or f.Pos (Neg.neg f).Pos
  -/
  rcases abv_pos_of_not_limZero h with ⟨K, K0, hK⟩
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    f : CauSeq α abs
    h : Not f.LimZero
    K : α
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
    ⊢ Or f.Pos (Neg.neg f).Pos
  -/
  rcases exists_forall_ge_and hK (f.cauchy₃ K0) with ⟨i, hi⟩
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    f : CauSeq α abs
    h : Not f.LimZero
    K : α
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
    ⊢ Or f.Pos (Neg.neg f).Pos
  -/
  refine (le_total 0 (f i)).imp ?_ ?_ <;>
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      ⊢ LE.le 0 (↑f i) → f.Pos
    -/
    refine fun h => ⟨K, K0, i, fun j ij => ?_⟩ <;>
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le 0 (↑f i)
      j : Nat
      ij : GE.ge j i
      ⊢ LE.le K (↑f j)
    -/
    have := (hi _ ij).1 <;>
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le 0 (↑f i)
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      ⊢ LE.le K (↑f j)
    -/
    cases' hi _ le_rfl with h₁ h₂
    /-
      case inr.intro.intro.intro.refine_1.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le 0 (↑f i)
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      h₁ : LE.le K (abs (↑f i))
      h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abs (HSub.hSub (↑f k) (↑f i))) K
      ⊢ LE.le K (↑f j)
    -/
  · rwa [abs_of_nonneg] at this
    /-
      case inr.intro.intro.intro.refine_1.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le 0 (↑f i)
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      h₁ : LE.le K (abs (↑f i))
      h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abs (HSub.hSub (↑f k) (↑f i))) K
      ⊢ LE.le 0 (↑f j)
    -/
    rw [abs_of_nonneg h] at h₁
    exact
      (le_add_iff_nonneg_right _).1
        (le_trans h₁ <| neg_le_sub_iff_le_add'.1 <| le_of_lt (abs_lt.1 <| h₂ _ ij).1)
    /-
      case inr.intro.intro.intro.refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le (↑f i) 0
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      h₁ : LE.le K (abs (↑f i))
      h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abs (HSub.hSub (↑f k) (↑f i))) K
      ⊢ LE.le K (↑(Neg.neg f) j)
    -/
  · rwa [abs_of_nonpos] at this
    /-
      case inr.intro.intro.intro.refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le (↑f i) 0
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      h₁ : LE.le K (abs (↑f i))
      h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abs (HSub.hSub (↑f k) (↑f i))) K
      ⊢ LE.le (↑f j) 0
    -/
    rw [abs_of_nonpos h] at h₁
    /-
      case inr.intro.intro.intro.refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le (↑f i) 0
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      h₁ : LE.le K (Neg.neg (↑f i))
      h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abs (HSub.hSub (↑f k) (↑f i))) K
      ⊢ LE.le (↑f j) 0
    -/
    rw [← sub_le_sub_iff_right, zero_sub]
    /-
      case inr.intro.intro.intro.refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      h✝ : Not f.LimZero
      K : α
      K0 : GT.gt K 0
      hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (abs (↑f j))
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → And (LE.le K (abs (↑f j))) (∀ (k : Nat), GE.ge k …
      h : LE.le (↑f i) 0
      j : Nat
      ij : GE.ge j i
      this : LE.le K (abs (↑f j))
      h₁ : LE.le K (Neg.neg (↑f i))
      h₂ : ∀ (k : Nat), GE.ge k i → LT.lt (abs (HSub.hSub (↑f k) (↑f i))) K
      ⊢ LE.le (HSub.hSub (↑f j) ?inr.intro.intro.intro.refine_2.intro) (Neg.neg ?inr …
    -/
    exact le_trans (le_of_lt (abs_lt.1 <| h₂ _ ij).2) h₁
    /-
      🎉 no goals
    -/


instance : LT (CauSeq α abs) :=
  ⟨fun f g => Pos (g - f)⟩


instance : LE (CauSeq α abs) :=
  ⟨fun f g => f < g ∨ f ≈ g⟩


theorem lt_of_lt_of_eq {f g h : CauSeq α abs} (fg : f < g) (gh : g ≈ h) : f < h :=
  show Pos (h - f) by
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      f g h : CauSeq α abs
      fg : LT.lt f g
      gh : HasEquiv.Equiv g h
      ⊢ (HSub.hSub h f).Pos
    -/
    convert pos_add_limZero fg (neg_limZero gh) using 1
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : LinearOrderedField α
      f g h : CauSeq α abs
      fg : LT.lt f g
      gh : HasEquiv.Equiv g h
      ⊢ Eq (HSub.hSub h f) (HAdd.hAdd (HSub.hSub g f) (Neg.neg (HSub.hSub g h)))
    -/
    simp
    /-
      🎉 no goals
    -/


theorem lt_of_eq_of_lt {f g h : CauSeq α abs} (fg : f ≈ g) (gh : g < h) : f < h := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    f g h : CauSeq α abs
    fg : HasEquiv.Equiv f g
    gh : LT.lt g h
    ⊢ LT.lt f h
  -/
  have := pos_add_limZero gh (neg_limZero fg)
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    f g h : CauSeq α abs
    fg : HasEquiv.Equiv f g
    gh : LT.lt g h
    this : (HAdd.hAdd (HSub.hSub h g) (Neg.neg (HSub.hSub f g))).Pos
    ⊢ LT.lt f h
  -/
  rwa [← sub_eq_add_neg, sub_sub_sub_cancel_right] at this
  /-
    🎉 no goals
  -/


theorem lt_trans {f g h : CauSeq α abs} (fg : f < g) (gh : g < h) : f < h :=
  show Pos (h - f) by
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      f g h : CauSeq α abs
      fg : LT.lt f g
      gh : LT.lt g h
      ⊢ (HSub.hSub h f).Pos
    -/
    convert add_pos fg gh using 1
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : LinearOrderedField α
      f g h : CauSeq α abs
      fg : LT.lt f g
      gh : LT.lt g h
      ⊢ Eq (HSub.hSub h f) (HAdd.hAdd (HSub.hSub g f) (HSub.hSub h g))
    -/
    simp
    /-
      🎉 no goals
    -/


theorem lt_irrefl {f : CauSeq α abs} : ¬f < f
                                  /-
                                    α : Type u_1
                                    inst✝ : LinearOrderedField α
                                    f : CauSeq α abs
                                    x✝ : LT.lt f f
                                    h : LT.lt f f := x✝
                                    ⊢ (HSub.hSub f f).LimZero
                                  -/
  | h => not_limZero_of_pos h (by simp [zero_limZero])
                                  /-
                                    🎉 no goals
                                  -/


theorem le_of_eq_of_le {f g h : CauSeq α abs} (hfg : f ≈ g) (hgh : g ≤ h) : f ≤ h :=
  hgh.elim (Or.inl ∘ CauSeq.lt_of_eq_of_lt hfg) (Or.inr ∘ Setoid.trans hfg)


theorem le_of_le_of_eq {f g h : CauSeq α abs} (hfg : f ≤ g) (hgh : g ≈ h) : f ≤ h :=
  hfg.elim (fun h => Or.inl (CauSeq.lt_of_lt_of_eq h hgh)) fun h => Or.inr (Setoid.trans h hgh)


instance : Preorder (CauSeq α abs) where
  lt := (· < ·)
  le f g := f < g ∨ f ≈ g
  le_refl _ := Or.inr (Setoid.refl _)
  le_trans _ _ _ fg gh :=
    match fg, gh with
    | Or.inl fg, Or.inl gh => Or.inl <| lt_trans fg gh
    | Or.inl fg, Or.inr gh => Or.inl <| lt_of_lt_of_eq fg gh
    | Or.inr fg, Or.inl gh => Or.inl <| lt_of_eq_of_lt fg gh
    | Or.inr fg, Or.inr gh => Or.inr <| Setoid.trans fg gh
  lt_iff_le_not_le _ _ :=
    ⟨fun h => ⟨Or.inl h, not_or_intro (mt (lt_trans h) lt_irrefl) (not_limZero_of_pos h)⟩,
      fun ⟨h₁, h₂⟩ => h₁.resolve_right (mt (fun h => Or.inr (Setoid.symm h)) h₂)⟩


theorem le_antisymm {f g : CauSeq α abs} (fg : f ≤ g) (gf : g ≤ f) : f ≈ g :=
  fg.resolve_left (not_lt_of_le gf)


theorem lt_total (f g : CauSeq α abs) : f < g ∨ f ≈ g ∨ g < f :=
  (trichotomy (g - f)).imp_right fun h =>
                                               /-
                                                 α : Type u_1
                                                 inst✝ : LinearOrderedField α
                                                 f g : CauSeq α abs
                                                 h✝ : Or (HSub.hSub g f).LimZero (Neg.neg (HSub.hSub g f)).Pos
                                                 h : (Neg.neg (HSub.hSub g f)).Pos
                                                 ⊢ LT.lt g f
                                               -/
    h.imp (fun h => Setoid.symm h) fun h => by rwa [neg_sub] at h
                                               /-
                                                 🎉 no goals
                                               -/


theorem le_total (f g : CauSeq α abs) : f ≤ g ∨ g ≤ f :=
  (or_assoc.2 (lt_total f g)).imp_right Or.inl


theorem const_lt {x y : α} : const x < const y ↔ x < y :=
                    /-
                      α : Type u_1
                      inst✝ : LinearOrderedField α
                      x y : α
                      ⊢ Iff (HSub.hSub (CauSeq.const abs y) (CauSeq.const abs x)).Pos (LT.lt x y)
                    -/
  show Pos _ ↔ _ by rw [← const_sub, const_pos, sub_pos]
                    /-
                      🎉 no goals
                    -/


theorem const_le {x y : α} : const x ≤ const y ↔ x ≤ y := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    x y : α
    ⊢ Iff (LE.le (CauSeq.const abs x) (CauSeq.const abs y)) (LE.le x y)
  -/
  rw [le_iff_lt_or_eq]; exact or_congr const_lt const_equiv
                        /-
                          🎉 no goals
                        -/


theorem le_of_exists {f g : CauSeq α abs} (h : ∃ i, ∀ j ≥ i, f j ≤ g j) : f ≤ g :=
  let ⟨i, hi⟩ := h
  (or_assoc.2 (CauSeq.lt_total f g)).elim id fun hgf =>
    False.elim
      (let ⟨_, hK0, j, hKj⟩ := hgf
      not_lt_of_ge (hi (max i j) (le_max_left _ _))
        (sub_pos.1 (lt_of_lt_of_le hK0 (hKj _ (le_max_right _ _)))))


theorem exists_gt (f : CauSeq α abs) : ∃ a : α, f < const a :=
  let ⟨K, H⟩ := f.bounded
  ⟨K + 1, 1, zero_lt_one, 0, fun i _ => by
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      K : α
      H : ∀ (i : Nat), LT.lt (abs (↑f i)) K
      i : Nat
      x✝ : GE.ge i 0
      ⊢ LE.le 1 (↑(HSub.hSub (CauSeq.const abs (HAdd.hAdd K 1)) f) i)
    -/
    rw [sub_apply, const_apply, le_sub_iff_add_le', add_le_add_iff_right]
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      f : CauSeq α abs
      K : α
      H : ∀ (i : Nat), LT.lt (abs (↑f i)) K
      i : Nat
      x✝ : GE.ge i 0
      ⊢ LE.le (↑f i) K
    -/
    exact le_of_lt (abs_lt.1 (H _)).2⟩
    /-
      🎉 no goals
    -/


theorem exists_lt (f : CauSeq α abs) : ∃ a : α, const a < f :=
  let ⟨a, h⟩ := (-f).exists_gt
                     /-
                       α : Type u_1
                       inst✝ : LinearOrderedField α
                       f : CauSeq α abs
                       a : α
                       h : LT.lt (Neg.neg f) (CauSeq.const abs a)
                       ⊢ (HSub.hSub f (CauSeq.const abs (Neg.neg a))).Pos
                     -/
  ⟨-a, show Pos _ by rwa [const_neg, sub_neg_eq_add, add_comm, ← sub_neg_eq_add]⟩
                     /-
                       🎉 no goals
                     -/

-- so named to match `rat_add_continuous_lemma`

theorem rat_sup_continuous_lemma {ε : α} {a₁ a₂ b₁ b₂ : α} :
    abs (a₁ - b₁) < ε → abs (a₂ - b₂) < ε → abs (a₁ ⊔ a₂ - b₁ ⊔ b₂) < ε := fun h₁ h₂ =>
  (abs_max_sub_max_le_max _ _ _ _).trans_lt (max_lt h₁ h₂)

-- so named to match `rat_add_continuous_lemma`

theorem rat_inf_continuous_lemma {ε : α} {a₁ a₂ b₁ b₂ : α} :
    abs (a₁ - b₁) < ε → abs (a₂ - b₂) < ε → abs (a₁ ⊓ a₂ - b₁ ⊓ b₂) < ε := fun h₁ h₂ =>
  (abs_min_sub_min_le_max _ _ _ _).trans_lt (max_lt h₁ h₂)


instance : Max (CauSeq α abs) :=
  ⟨fun f g =>
    ⟨f ⊔ g, fun _ ε0 =>
      (exists_forall_ge_and (f.cauchy₃ ε0) (g.cauchy₃ ε0)).imp fun _ H _ ij =>
        let ⟨H₁, H₂⟩ := H _ le_rfl
        rat_sup_continuous_lemma (H₁ _ ij) (H₂ _ ij)⟩⟩


instance : Min (CauSeq α abs) :=
  ⟨fun f g =>
    ⟨f ⊓ g, fun _ ε0 =>
      (exists_forall_ge_and (f.cauchy₃ ε0) (g.cauchy₃ ε0)).imp fun _ H _ ij =>
        let ⟨H₁, H₂⟩ := H _ le_rfl
        rat_inf_continuous_lemma (H₁ _ ij) (H₂ _ ij)⟩⟩


@[simp, norm_cast]
theorem coe_sup (f g : CauSeq α abs) : ⇑(f ⊔ g) = (f : ℕ → α) ⊔ g :=
  rfl


@[simp, norm_cast]
theorem coe_inf (f g : CauSeq α abs) : ⇑(f ⊓ g) = (f : ℕ → α) ⊓ g :=
  rfl


theorem sup_limZero {f g : CauSeq α abs} (hf : LimZero f) (hg : LimZero g) : LimZero (f ⊔ g)
  | ε, ε0 =>
    (exists_forall_ge_and (hf _ ε0) (hg _ ε0)).imp fun _ H j ij => by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abs (↑f j)) ε) (LT.lt (abs (↑g j)) ε)
        j : Nat
        ij : GE.ge j x✝
        ⊢ LT.lt (abs (↑(Max.max f g) j)) ε
      -/
      let ⟨H₁, H₂⟩ := H _ ij
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abs (↑f j)) ε) (LT.lt (abs (↑g j)) ε)
        j : Nat
        ij : GE.ge j x✝
        H₁ : LT.lt (abs (↑f j)) ε
        H₂ : LT.lt (abs (↑g j)) ε
        ⊢ LT.lt (abs (↑(Max.max f g) j)) ε
      -/
      rw [abs_lt] at H₁ H₂ ⊢
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abs (↑f j)) ε) (LT.lt (abs (↑g j)) ε)
        j : Nat
        ij : GE.ge j x✝
        H₁ : And (LT.lt (Neg.neg ε) (↑f j)) (LT.lt (↑f j) ε)
        H₂ : And (LT.lt (Neg.neg ε) (↑g j)) (LT.lt (↑g j) ε)
        ⊢ And (LT.lt (Neg.neg ε) (↑(Max.max f g) j)) (LT.lt (↑(Max.max f g) j) ε)
      -/
      exact ⟨lt_sup_iff.mpr (Or.inl H₁.1), sup_lt_iff.mpr ⟨H₁.2, H₂.2⟩⟩
      /-
        🎉 no goals
      -/


theorem inf_limZero {f g : CauSeq α abs} (hf : LimZero f) (hg : LimZero g) : LimZero (f ⊓ g)
  | ε, ε0 =>
    (exists_forall_ge_and (hf _ ε0) (hg _ ε0)).imp fun _ H j ij => by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abs (↑f j)) ε) (LT.lt (abs (↑g j)) ε)
        j : Nat
        ij : GE.ge j x✝
        ⊢ LT.lt (abs (↑(Min.min f g) j)) ε
      -/
      let ⟨H₁, H₂⟩ := H _ ij
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abs (↑f j)) ε) (LT.lt (abs (↑g j)) ε)
        j : Nat
        ij : GE.ge j x✝
        H₁ : LT.lt (abs (↑f j)) ε
        H₂ : LT.lt (abs (↑g j)) ε
        ⊢ LT.lt (abs (↑(Min.min f g) j)) ε
      -/
      rw [abs_lt] at H₁ H₂ ⊢
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        f g : CauSeq α abs
        hf : f.LimZero
        hg : g.LimZero
        ε : α
        ε0 : GT.gt ε 0
        x✝ : Nat
        H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (abs (↑f j)) ε) (LT.lt (abs (↑g j)) ε)
        j : Nat
        ij : GE.ge j x✝
        H₁ : And (LT.lt (Neg.neg ε) (↑f j)) (LT.lt (↑f j) ε)
        H₂ : And (LT.lt (Neg.neg ε) (↑g j)) (LT.lt (↑g j) ε)
        ⊢ And (LT.lt (Neg.neg ε) (↑(Min.min f g) j)) (LT.lt (↑(Min.min f g) j) ε)
      -/
      exact ⟨lt_inf_iff.mpr ⟨H₁.1, H₂.1⟩, inf_lt_iff.mpr (Or.inl H₁.2)⟩
      /-
        🎉 no goals
      -/


theorem sup_equiv_sup {a₁ b₁ a₂ b₂ : CauSeq α abs} (ha : a₁ ≈ a₂) (hb : b₁ ≈ b₂) :
    a₁ ⊔ b₁ ≈ a₂ ⊔ b₂ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a₁ b₁ a₂ b₂ : CauSeq α abs
    ha : HasEquiv.Equiv a₁ a₂
    hb : HasEquiv.Equiv b₁ b₂
    ⊢ HasEquiv.Equiv (Max.max a₁ b₁) (Max.max a₂ b₂)
  -/
  intro ε ε0
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a₁ b₁ a₂ b₂ : CauSeq α abs
    ha : HasEquiv.Equiv a₁ a₂
    hb : HasEquiv.Equiv b₁ b₂
    ε : α
    ε0 : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub (Max.max a₁ …
  -/
  obtain ⟨ai, hai⟩ := ha ε ε0
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a₁ b₁ a₂ b₂ : CauSeq α abs
    ha : HasEquiv.Equiv a₁ a₂
    hb : HasEquiv.Equiv b₁ b₂
    ε : α
    ε0 : GT.gt ε 0
    ai : Nat
    hai : ∀ (j : Nat), GE.ge j ai → LT.lt (abs (↑(HSub.hSub a₁ a₂) j)) ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub (Max.max a₁ …
  -/
  obtain ⟨bi, hbi⟩ := hb ε ε0
  exact
    ⟨ai ⊔ bi, fun i hi =>
      (abs_max_sub_max_le_max (a₁ i) (b₁ i) (a₂ i) (b₂ i)).trans_lt
        (max_lt (hai i (sup_le_iff.mp hi).1) (hbi i (sup_le_iff.mp hi).2))⟩


theorem inf_equiv_inf {a₁ b₁ a₂ b₂ : CauSeq α abs} (ha : a₁ ≈ a₂) (hb : b₁ ≈ b₂) :
    a₁ ⊓ b₁ ≈ a₂ ⊓ b₂ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a₁ b₁ a₂ b₂ : CauSeq α abs
    ha : HasEquiv.Equiv a₁ a₂
    hb : HasEquiv.Equiv b₁ b₂
    ⊢ HasEquiv.Equiv (Min.min a₁ b₁) (Min.min a₂ b₂)
  -/
  intro ε ε0
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a₁ b₁ a₂ b₂ : CauSeq α abs
    ha : HasEquiv.Equiv a₁ a₂
    hb : HasEquiv.Equiv b₁ b₂
    ε : α
    ε0 : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub (Min.min a₁ …
  -/
  obtain ⟨ai, hai⟩ := ha ε ε0
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a₁ b₁ a₂ b₂ : CauSeq α abs
    ha : HasEquiv.Equiv a₁ a₂
    hb : HasEquiv.Equiv b₁ b₂
    ε : α
    ε0 : GT.gt ε 0
    ai : Nat
    hai : ∀ (j : Nat), GE.ge j ai → LT.lt (abs (↑(HSub.hSub a₁ a₂) j)) ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub (Min.min a₁ …
  -/
  obtain ⟨bi, hbi⟩ := hb ε ε0
  exact
    ⟨ai ⊔ bi, fun i hi =>
      (abs_min_sub_min_le_max (a₁ i) (b₁ i) (a₂ i) (b₂ i)).trans_lt
        (max_lt (hai i (sup_le_iff.mp hi).1) (hbi i (sup_le_iff.mp hi).2))⟩


protected theorem sup_lt {a b c : CauSeq α abs} (ha : a < c) (hb : b < c) : a ⊔ b < c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    ha : LT.lt a c
    hb : LT.lt b c
    ⊢ LT.lt (Max.max a b) c
  -/
  obtain ⟨⟨εa, εa0, ia, ha⟩, ⟨εb, εb0, ib, hb⟩⟩ := ha, hb
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    εa : α
    εa0 : GT.gt εa 0
    ia : Nat
    ha : ∀ (j : Nat), GE.ge j ia → LE.le εa (↑(HSub.hSub c a) j)
    εb : α
    εb0 : GT.gt εb 0
    ib : Nat
    hb : ∀ (j : Nat), GE.ge j ib → LE.le εb (↑(HSub.hSub c b) j)
    ⊢ LT.lt (Max.max a b) c
  -/
  refine ⟨εa ⊓ εb, lt_inf_iff.mpr ⟨εa0, εb0⟩, ia ⊔ ib, fun i hi => ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    εa : α
    εa0 : GT.gt εa 0
    ia : Nat
    ha : ∀ (j : Nat), GE.ge j ia → LE.le εa (↑(HSub.hSub c a) j)
    εb : α
    εb0 : GT.gt εb 0
    ib : Nat
    hb : ∀ (j : Nat), GE.ge j ib → LE.le εb (↑(HSub.hSub c b) j)
    i : Nat
    hi : GE.ge i (Max.max ia ib)
    ⊢ LE.le (Min.min εa εb) (↑(HSub.hSub c (Max.max a b)) i)
  -/
  have := min_le_min (ha _ (sup_le_iff.mp hi).1) (hb _ (sup_le_iff.mp hi).2)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    εa : α
    εa0 : GT.gt εa 0
    ia : Nat
    ha : ∀ (j : Nat), GE.ge j ia → LE.le εa (↑(HSub.hSub c a) j)
    εb : α
    εb0 : GT.gt εb 0
    ib : Nat
    hb : ∀ (j : Nat), GE.ge j ib → LE.le εb (↑(HSub.hSub c b) j)
    i : Nat
    hi : GE.ge i (Max.max ia ib)
    this : LE.le (Min.min εa εb) (Min.min (↑(HSub.hSub c a) i) (↑(HSub.hSub c b) i))
    ⊢ LE.le (Min.min εa εb) (↑(HSub.hSub c (Max.max a b)) i)
  -/
  exact this.trans_eq (min_sub_sub_left _ _ _)
  /-
    🎉 no goals
  -/


protected theorem lt_inf {a b c : CauSeq α abs} (hb : a < b) (hc : a < c) : a < b ⊓ c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    hb : LT.lt a b
    hc : LT.lt a c
    ⊢ LT.lt a (Min.min b c)
  -/
  obtain ⟨⟨εb, εb0, ib, hb⟩, ⟨εc, εc0, ic, hc⟩⟩ := hb, hc
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    εb : α
    εb0 : GT.gt εb 0
    ib : Nat
    hb : ∀ (j : Nat), GE.ge j ib → LE.le εb (↑(HSub.hSub b a) j)
    εc : α
    εc0 : GT.gt εc 0
    ic : Nat
    hc : ∀ (j : Nat), GE.ge j ic → LE.le εc (↑(HSub.hSub c a) j)
    ⊢ LT.lt a (Min.min b c)
  -/
  refine ⟨εb ⊓ εc, lt_inf_iff.mpr ⟨εb0, εc0⟩, ib ⊔ ic, fun i hi => ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    εb : α
    εb0 : GT.gt εb 0
    ib : Nat
    hb : ∀ (j : Nat), GE.ge j ib → LE.le εb (↑(HSub.hSub b a) j)
    εc : α
    εc0 : GT.gt εc 0
    ic : Nat
    hc : ∀ (j : Nat), GE.ge j ic → LE.le εc (↑(HSub.hSub c a) j)
    i : Nat
    hi : GE.ge i (Max.max ib ic)
    ⊢ LE.le (Min.min εb εc) (↑(HSub.hSub (Min.min b c) a) i)
  -/
  have := min_le_min (hb _ (sup_le_iff.mp hi).1) (hc _ (sup_le_iff.mp hi).2)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    εb : α
    εb0 : GT.gt εb 0
    ib : Nat
    hb : ∀ (j : Nat), GE.ge j ib → LE.le εb (↑(HSub.hSub b a) j)
    εc : α
    εc0 : GT.gt εc 0
    ic : Nat
    hc : ∀ (j : Nat), GE.ge j ic → LE.le εc (↑(HSub.hSub c a) j)
    i : Nat
    hi : GE.ge i (Max.max ib ic)
    this : LE.le (Min.min εb εc) (Min.min (↑(HSub.hSub b a) i) (↑(HSub.hSub c a) i))
    ⊢ LE.le (Min.min εb εc) (↑(HSub.hSub (Min.min b c) a) i)
  -/
  exact this.trans_eq (min_sub_sub_right _ _ _)
  /-
    🎉 no goals
  -/


@[simp]
protected theorem sup_idem (a : CauSeq α abs) : a ⊔ a = a := Subtype.ext (sup_idem _)


@[simp]
protected theorem inf_idem (a : CauSeq α abs) : a ⊓ a = a := Subtype.ext (inf_idem _)


protected theorem sup_comm (a b : CauSeq α abs) : a ⊔ b = b ⊔ a := Subtype.ext (sup_comm _ _)


protected theorem inf_comm (a b : CauSeq α abs) : a ⊓ b = b ⊓ a := Subtype.ext (inf_comm _ _)


protected theorem sup_eq_right {a b : CauSeq α abs} (h : a ≤ b) : a ⊔ b ≈ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b : CauSeq α abs
    h : LE.le a b
    ⊢ HasEquiv.Equiv (Max.max a b) b
  -/
  obtain ⟨ε, ε0 : _ < _, i, h⟩ | h := h
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ⊢ HasEquiv.Equiv (Max.max a b) b
    -/
  · intro _ _
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub (Max.max a  …
    -/
    refine ⟨i, fun j hj => ?_⟩
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (abs (↑(HSub.hSub (Max.max a b) b) j)) ε✝
    -/
    dsimp
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (abs (HSub.hSub (Max.max (↑a j) (↑b j)) (↑b j))) ε✝
    -/
    rw [← max_sub_sub_right]
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (abs (Max.max (HSub.hSub (↑a j) (↑b j)) (HSub.hSub (↑b j) (↑b j)))) ε✝
    -/
    rwa [sub_self, max_eq_right, abs_zero]
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LE.le (HSub.hSub (↑a j) (↑b j)) 0
    -/
    rw [sub_nonpos, ← sub_nonneg]
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub b a) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LE.le 0 (HSub.hSub (↑b j) (↑a j))
    -/
    exact ε0.le.trans (h _ hj)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      h : HasEquiv.Equiv a b
      ⊢ HasEquiv.Equiv (Max.max a b) b
    -/
  · refine Setoid.trans (sup_equiv_sup h (Setoid.refl _)) ?_
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      h : HasEquiv.Equiv a b
      ⊢ HasEquiv.Equiv (Max.max b b) b
    -/
    rw [CauSeq.sup_idem]
    /-
      🎉 no goals
    -/


protected theorem inf_eq_right {a b : CauSeq α abs} (h : b ≤ a) : a ⊓ b ≈ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b : CauSeq α abs
    h : LE.le b a
    ⊢ HasEquiv.Equiv (Min.min a b) b
  -/
  obtain ⟨ε, ε0 : _ < _, i, h⟩ | h := h
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub a b) j)
      ⊢ HasEquiv.Equiv (Min.min a b) b
    -/
  · intro _ _
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub a b) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub (Min.min a  …
    -/
    refine ⟨i, fun j hj => ?_⟩
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub a b) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (abs (↑(HSub.hSub (Min.min a b) b) j)) ε✝
    -/
    dsimp
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub a b) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (abs (HSub.hSub (Min.min (↑a j) (↑b j)) (↑b j))) ε✝
    -/
    rw [← min_sub_sub_right]
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub a b) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (abs (Min.min (HSub.hSub (↑a j) (↑b j)) (HSub.hSub (↑b j) (↑b j)))) ε✝
    -/
    rwa [sub_self, min_eq_right, abs_zero]
    /-
      case inl.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      ε : α
      ε0 : LT.lt 0 ε
      i : Nat
      h : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub a b) j)
      ε✝ : α
      a✝ : GT.gt ε✝ 0
      j : Nat
      hj : GE.ge j i
      ⊢ LE.le 0 (HSub.hSub (↑a j) (↑b j))
    -/
    exact ε0.le.trans (h _ hj)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      h : HasEquiv.Equiv b a
      ⊢ HasEquiv.Equiv (Min.min a b) b
    -/
  · refine Setoid.trans (inf_equiv_inf (Setoid.symm h) (Setoid.refl _)) ?_
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : CauSeq α abs
      h : HasEquiv.Equiv b a
      ⊢ HasEquiv.Equiv (Min.min b b) b
    -/
    rw [CauSeq.inf_idem]
    /-
      🎉 no goals
    -/


protected theorem sup_eq_left {a b : CauSeq α abs} (h : b ≤ a) : a ⊔ b ≈ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b : CauSeq α abs
    h : LE.le b a
    ⊢ HasEquiv.Equiv (Max.max a b) a
  -/
  simpa only [CauSeq.sup_comm] using CauSeq.sup_eq_right h
  /-
    🎉 no goals
  -/


protected theorem inf_eq_left {a b : CauSeq α abs} (h : a ≤ b) : a ⊓ b ≈ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b : CauSeq α abs
    h : LE.le a b
    ⊢ HasEquiv.Equiv (Min.min a b) a
  -/
  simpa only [CauSeq.inf_comm] using CauSeq.inf_eq_right h
  /-
    🎉 no goals
  -/


protected theorem le_sup_left {a b : CauSeq α abs} : a ≤ a ⊔ b :=
  le_of_exists ⟨0, fun _ _ => le_sup_left⟩


protected theorem inf_le_left {a b : CauSeq α abs} : a ⊓ b ≤ a :=
  le_of_exists ⟨0, fun _ _ => inf_le_left⟩


protected theorem le_sup_right {a b : CauSeq α abs} : b ≤ a ⊔ b :=
  le_of_exists ⟨0, fun _ _ => le_sup_right⟩


protected theorem inf_le_right {a b : CauSeq α abs} : a ⊓ b ≤ b :=
  le_of_exists ⟨0, fun _ _ => inf_le_right⟩


protected theorem sup_le {a b c : CauSeq α abs} (ha : a ≤ c) (hb : b ≤ c) : a ⊔ b ≤ c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    ha : LE.le a c
    hb : LE.le b c
    ⊢ LE.le (Max.max a b) c
  -/
  cases' ha with ha ha
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      hb : LE.le b c
      ha : LT.lt a c
      ⊢ LE.le (Max.max a b) c
    -/
  · cases' hb with hb hb
      /-
        case inl.inl
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        ha : LT.lt a c
        hb : LT.lt b c
        ⊢ LE.le (Max.max a b) c
      -/
    · exact Or.inl (CauSeq.sup_lt ha hb)
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        ha : LT.lt a c
        hb : HasEquiv.Equiv b c
        ⊢ LE.le (Max.max a b) c
      -/
    · replace ha := le_of_le_of_eq ha.le (Setoid.symm hb)
      /-
        case inl.inr
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        hb : HasEquiv.Equiv b c
        ha : LE.le a b
        ⊢ LE.le (Max.max a b) c
      -/
      refine le_of_le_of_eq (Or.inr ?_) hb
      /-
        case inl.inr
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        hb : HasEquiv.Equiv b c
        ha : LE.le a b
        ⊢ HasEquiv.Equiv (Max.max a b) b
      -/
      exact CauSeq.sup_eq_right ha
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      hb : LE.le b c
      ha : HasEquiv.Equiv a c
      ⊢ LE.le (Max.max a b) c
    -/
  · replace hb := le_of_le_of_eq hb (Setoid.symm ha)
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      ha : HasEquiv.Equiv a c
      hb : LE.le b a
      ⊢ LE.le (Max.max a b) c
    -/
    refine le_of_le_of_eq (Or.inr ?_) ha
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      ha : HasEquiv.Equiv a c
      hb : LE.le b a
      ⊢ HasEquiv.Equiv (Max.max a b) a
    -/
    exact CauSeq.sup_eq_left hb
    /-
      🎉 no goals
    -/


protected theorem le_inf {a b c : CauSeq α abs} (hb : a ≤ b) (hc : a ≤ c) : a ≤ b ⊓ c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : CauSeq α abs
    hb : LE.le a b
    hc : LE.le a c
    ⊢ LE.le a (Min.min b c)
  -/
  cases' hb with hb hb
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      hc : LE.le a c
      hb : LT.lt a b
      ⊢ LE.le a (Min.min b c)
    -/
  · cases' hc with hc hc
      /-
        case inl.inl
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        hb : LT.lt a b
        hc : LT.lt a c
        ⊢ LE.le a (Min.min b c)
      -/
    · exact Or.inl (CauSeq.lt_inf hb hc)
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        hb : LT.lt a b
        hc : HasEquiv.Equiv a c
        ⊢ LE.le a (Min.min b c)
      -/
    · replace hb := le_of_eq_of_le (Setoid.symm hc) hb.le
      /-
        case inl.inr
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        hc : HasEquiv.Equiv a c
        hb : LE.le c b
        ⊢ LE.le a (Min.min b c)
      -/
      refine le_of_eq_of_le hc (Or.inr ?_)
      /-
        case inl.inr
        α : Type u_1
        inst✝ : LinearOrderedField α
        a b c : CauSeq α abs
        hc : HasEquiv.Equiv a c
        hb : LE.le c b
        ⊢ HasEquiv.Equiv c (Min.min b c)
      -/
      exact Setoid.symm (CauSeq.inf_eq_right hb)
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      hc : LE.le a c
      hb : HasEquiv.Equiv a b
      ⊢ LE.le a (Min.min b c)
    -/
  · replace hc := le_of_eq_of_le (Setoid.symm hb) hc
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      hb : HasEquiv.Equiv a b
      hc : LE.le b c
      ⊢ LE.le a (Min.min b c)
    -/
    refine le_of_eq_of_le hb (Or.inr ?_)
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : CauSeq α abs
      hb : HasEquiv.Equiv a b
      hc : LE.le b c
      ⊢ HasEquiv.Equiv b (Min.min b c)
    -/
    exact Setoid.symm (CauSeq.inf_eq_left hc)
    /-
      🎉 no goals
    -/


protected theorem sup_inf_distrib_left (a b c : CauSeq α abs) : a ⊔ b ⊓ c = (a ⊔ b) ⊓ (a ⊔ c) :=
  ext fun _ ↦ max_min_distrib_left _ _ _


protected theorem sup_inf_distrib_right (a b c : CauSeq α abs) : a ⊓ b ⊔ c = (a ⊔ c) ⊓ (b ⊔ c) :=
  ext fun _ ↦ max_min_distrib_right _ _ _


