@[to_additive] lemma mabs_pow (n : ℕ) (a : α) : |a ^ n|ₘ = |a|ₘ ^ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    n : Nat
    a : α
    ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) n)
  -/
  obtain ha | ha := le_total a 1
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Nat
      a : α
      ha : LE.le a 1
      ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) n)
    -/
  · rw [mabs_of_le_one ha, ← mabs_inv, ← inv_pow, mabs_of_one_le]
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Nat
      a : α
      ha : LE.le a 1
      ⊢ LE.le 1 (HPow.hPow (Inv.inv a) n)
    -/
    exact one_le_pow_of_one_le' (one_le_inv'.2 ha) n
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Nat
      a : α
      ha : LE.le 1 a
      ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) n)
    -/
  · rw [mabs_of_one_le ha, mabs_of_one_le (one_le_pow_of_one_le' ha n)]
    /-
      🎉 no goals
    -/


@[to_additive] private lemma mabs_mul_eq_mul_mabs_le (hab : a ≤ b) :
    |a * b|ₘ = |a|ₘ * |b|ₘ ↔ 1 ≤ a ∧ 1 ≤ b ∨ a ≤ 1 ∧ b ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b : α
    hab : LE.le a b
    ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
  -/
  obtain ha | ha := le_or_lt 1 a <;> obtain hb | hb := le_or_lt 1 b
    /-
      case inl.inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LE.le 1 a
      hb : LE.le 1 b
      ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
    -/
  · simp [ha, hb, mabs_of_one_le, one_le_mul ha hb]
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LE.le 1 a
      hb : LT.lt b 1
      ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
    -/
  · exact (lt_irrefl (1 : α) <| ha.trans_lt <| hab.trans_lt hb).elim
    /-
      🎉 no goals
    -/
  /-
    case inr.inl
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b : α
    hab : LE.le a b
    ha : LT.lt a 1
    hb : LE.le 1 b
    ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
  -/
  swap
    /-
      case inr.inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LT.lt a 1
      hb : LT.lt b 1
      ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
    -/
  · simp [ha.le, hb.le, mabs_of_le_one, mul_le_one', mul_comm]
    /-
      🎉 no goals
    -/
  have : (|a * b|ₘ = a⁻¹ * b ↔ b ≤ 1) ↔
    (|a * b|ₘ = |a|ₘ * |b|ₘ ↔ 1 ≤ a ∧ 1 ≤ b ∨ a ≤ 1 ∧ b ≤ 1) := by
    simp [ha.le, ha.not_le, hb, mabs_of_le_one, mabs_of_one_le]
  /-
    case inr.inl
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b : α
    hab : LE.le a b
    ha : LT.lt a 1
    hb : LE.le 1 b
    this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
    ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
  -/
  refine this.mp ⟨fun h ↦ ?_, fun h ↦ by simp only [h.antisymm hb, mabs_of_lt_one ha, mul_one]⟩
  /-
    case inr.inl
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b : α
    hab : LE.le a b
    ha : LT.lt a 1
    hb : LE.le 1 b
    this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
    h : Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)
    ⊢ LE.le b 1
  -/
  obtain ab | ab := le_or_lt (a * b) 1
    /-
      case inr.inl.inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LT.lt a 1
      hb : LE.le 1 b
      this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
      h : Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)
      ab : LE.le (HMul.hMul a b) 1
      ⊢ LE.le b 1
    -/
  · refine (eq_one_of_inv_eq' ?_).le
    /-
      case inr.inl.inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LT.lt a 1
      hb : LE.le 1 b
      this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
      h : Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)
      ab : LE.le (HMul.hMul a b) 1
      ⊢ Eq (Inv.inv b) b
    -/
    rwa [mabs_of_le_one ab, mul_inv_rev, mul_comm, mul_right_inj] at h
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LT.lt a 1
      hb : LE.le 1 b
      this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
      h : Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)
      ab : LT.lt 1 (HMul.hMul a b)
      ⊢ LE.le b 1
    -/
  · rw [mabs_of_one_lt ab, mul_left_inj] at h
    /-
      case inr.inl.inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LT.lt a 1
      hb : LE.le 1 b
      this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
      h : Eq a (Inv.inv a)
      ab : LT.lt 1 (HMul.hMul a b)
      ⊢ LE.le b 1
    -/
    rw [eq_one_of_inv_eq' h.symm] at ha
    /-
      case inr.inl.inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      hab : LE.le a b
      ha : LT.lt 1 1
      hb : LE.le 1 b
      this : Iff (Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (Inv.inv a) b)) (LE.le b …
      h : Eq a (Inv.inv a)
      ab : LT.lt 1 (HMul.hMul a b)
      ⊢ LE.le b 1
    -/
    cases ha.false
    /-
      🎉 no goals
    -/


@[to_additive] lemma mabs_mul_eq_mul_mabs_iff (a b : α) :
    |a * b|ₘ = |a|ₘ * |b|ₘ ↔ 1 ≤ a ∧ 1 ≤ b ∨ a ≤ 1 ∧ b ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b : α
    ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
  -/
  obtain ab | ab := le_total a b
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      ab : LE.le a b
      ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
    -/
  · exact mabs_mul_eq_mul_mabs_le ab
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a b : α
      ab : LE.le b a
      ⊢ Iff (Eq (mabs (HMul.hMul a b)) (HMul.hMul (mabs a) (mabs b))) (Or (And (LE.l …
    -/
  · simpa only [mul_comm, and_comm] using mabs_mul_eq_mul_mabs_le ab
    /-
      🎉 no goals
    -/


                                                /-
                                                  α : Type u_1
                                                  inst✝ : LinearOrderedAddCommGroup α
                                                  a b : α
                                                  ⊢ Iff (LE.le (abs a) b) (And (LE.le (Neg.neg b) a) (LE.le a b))
                                                -/
theorem abs_le : |a| ≤ b ↔ -b ≤ a ∧ a ≤ b := by rw [abs_le', and_comm, @neg_le α]
                                                /-
                                                  🎉 no goals
                                                -/


                                                 /-
                                                   α : Type u_1
                                                   inst✝ : LinearOrderedAddCommGroup α
                                                   a b : α
                                                   ⊢ Iff (LE.le a (abs b)) (Or (LE.le b (Neg.neg a)) (LE.le a b))
                                                 -/
theorem le_abs' : a ≤ |b| ↔ b ≤ -a ∨ a ≤ b := by rw [le_abs, or_comm, @le_neg α]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem neg_le_of_abs_le (h : |a| ≤ b) : -b ≤ a :=
  (abs_le.mp h).1


theorem le_of_abs_le (h : |a| ≤ b) : a ≤ b :=
  (abs_le.mp h).2


@[to_additive]
theorem apply_abs_le_mul_of_one_le' {β : Type*} [MulOneClass β] [Preorder β]
    [MulLeftMono β] [MulRightMono β] {f : α → β}
    {a : α} (h₁ : 1 ≤ f a) (h₂ : 1 ≤ f (-a)) : f |a| ≤ f a * f (-a) :=
  (le_total a 0).rec (fun ha => (abs_of_nonpos ha).symm ▸ le_mul_of_one_le_left' h₁) fun ha =>
    (abs_of_nonneg ha).symm ▸ le_mul_of_one_le_right' h₂


@[to_additive]
theorem apply_abs_le_mul_of_one_le {β : Type*} [MulOneClass β] [Preorder β]
    [MulLeftMono β] [MulRightMono β] {f : α → β}
    (h : ∀ x, 1 ≤ f x) (a : α) : f |a| ≤ f a * f (-a) :=
  apply_abs_le_mul_of_one_le' (h _) (h _)


/-- The **triangle inequality** in `LinearOrderedAddCommGroup`s. -/
theorem abs_add (a b : α) : |a + b| ≤ |a| + |b| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    ⊢ LE.le (abs (HAdd.hAdd a b)) (HAdd.hAdd (abs a) (abs b))
  -/
  rw [abs_le, neg_add]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    ⊢ And (LE.le (HAdd.hAdd (Neg.neg (abs a)) (Neg.neg (abs b))) (HAdd.hAdd a b))  …
  -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
  constructor <;> gcongr <;> apply_rules [neg_abs_le, le_abs_self]
                             /-
                               🎉 no goals
                             -/


                                                       /-
                                                         α : Type u_1
                                                         inst✝ : LinearOrderedAddCommGroup α
                                                         a b : α
                                                         ⊢ LE.le (abs a) (HAdd.hAdd (abs b) (abs (HAdd.hAdd b a)))
                                                       -/
theorem abs_add' (a b : α) : |a| ≤ |b| + |b + a| := by simpa using abs_add (-b) (b + a)
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem abs_sub (a b : α) : |a - b| ≤ |a| + |b| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    ⊢ LE.le (abs (HSub.hSub a b)) (HAdd.hAdd (abs a) (abs b))
  -/
  rw [sub_eq_add_neg, ← abs_neg b]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    ⊢ LE.le (abs (HAdd.hAdd a (Neg.neg b))) (HAdd.hAdd (abs a) (abs (Neg.neg b)))
  -/
  exact abs_add a _
  /-
    🎉 no goals
  -/


theorem abs_sub_le_iff : |a - b| ≤ c ↔ a - b ≤ c ∧ b - a ≤ c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Iff (LE.le (abs (HSub.hSub a b)) c) (And (LE.le (HSub.hSub a b) c) (LE.le (H …
  -/
  rw [abs_le, neg_le_sub_iff_le_add, sub_le_iff_le_add', and_comm, sub_le_iff_le_add']
  /-
    🎉 no goals
  -/


theorem abs_sub_lt_iff : |a - b| < c ↔ a - b < c ∧ b - a < c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Iff (LT.lt (abs (HSub.hSub a b)) c) (And (LT.lt (HSub.hSub a b) c) (LT.lt (H …
  -/
  rw [@abs_lt α, neg_lt_sub_iff_lt_add', sub_lt_iff_lt_add', and_comm, sub_lt_iff_lt_add']
  /-
    🎉 no goals
  -/


theorem sub_le_of_abs_sub_le_left (h : |a - b| ≤ c) : b - c ≤ a :=
  sub_le_comm.1 <| (abs_sub_le_iff.1 h).2


theorem sub_le_of_abs_sub_le_right (h : |a - b| ≤ c) : a - c ≤ b :=
  sub_le_of_abs_sub_le_left (abs_sub_comm a b ▸ h)


theorem sub_lt_of_abs_sub_lt_left (h : |a - b| < c) : b - c < a :=
  sub_lt_comm.1 <| (abs_sub_lt_iff.1 h).2


theorem sub_lt_of_abs_sub_lt_right (h : |a - b| < c) : a - c < b :=
  sub_lt_of_abs_sub_lt_left (abs_sub_comm a b ▸ h)


theorem abs_sub_abs_le_abs_sub (a b : α) : |a| - |b| ≤ |a - b| :=
  sub_le_iff_le_add.2 <|
    calc
                              /-
                                α : Type u_1
                                inst✝ : LinearOrderedAddCommGroup α
                                a b : α
                                ⊢ Eq (abs a) (abs (HAdd.hAdd (HSub.hSub a b) b))
                              -/
      |a| = |a - b + b| := by rw [sub_add_cancel]
                              /-
                                🎉 no goals
                              -/
      _ ≤ |a - b| + |b| := abs_add _ _


theorem abs_abs_sub_abs_le_abs_sub (a b : α) : |(|a| - |b|)| ≤ |a - b| :=
  abs_sub_le_iff.2
                                    /-
                                      α : Type u_1
                                      inst✝ : LinearOrderedAddCommGroup α
                                      a b : α
                                      ⊢ LE.le (HSub.hSub (abs b) (abs a)) (abs (HSub.hSub a b))
                                    -/
    ⟨abs_sub_abs_le_abs_sub _ _, by rw [abs_sub_comm]; apply abs_sub_abs_le_abs_sub⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- `|a - b| ≤ n` if `0 ≤ a ≤ n` and `0 ≤ b ≤ n`. -/
theorem abs_sub_le_of_nonneg_of_le {a b n : α} (a_nonneg : 0 ≤ a) (a_le_n : a ≤ n)
    (b_nonneg : 0 ≤ b) (b_le_n : b ≤ n) : |a - b| ≤ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b n : α
    a_nonneg : LE.le 0 a
    a_le_n : LE.le a n
    b_nonneg : LE.le 0 b
    b_le_n : LE.le b n
    ⊢ LE.le (abs (HSub.hSub a b)) n
  -/
  rw [abs_sub_le_iff, sub_le_iff_le_add, sub_le_iff_le_add]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b n : α
    a_nonneg : LE.le 0 a
    a_le_n : LE.le a n
    b_nonneg : LE.le 0 b
    b_le_n : LE.le b n
    ⊢ And (LE.le a (HAdd.hAdd n b)) (LE.le b (HAdd.hAdd n a))
  -/
  exact ⟨le_add_of_le_of_nonneg a_le_n b_nonneg, le_add_of_le_of_nonneg b_le_n a_nonneg⟩
  /-
    🎉 no goals
  -/


/-- `|a - b| < n` if `0 ≤ a < n` and `0 ≤ b < n`. -/
theorem abs_sub_lt_of_nonneg_of_lt {a b n : α} (a_nonneg : 0 ≤ a) (a_lt_n : a < n)
    (b_nonneg : 0 ≤ b) (b_lt_n : b < n) : |a - b| < n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b n : α
    a_nonneg : LE.le 0 a
    a_lt_n : LT.lt a n
    b_nonneg : LE.le 0 b
    b_lt_n : LT.lt b n
    ⊢ LT.lt (abs (HSub.hSub a b)) n
  -/
  rw [abs_sub_lt_iff, sub_lt_iff_lt_add, sub_lt_iff_lt_add]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b n : α
    a_nonneg : LE.le 0 a
    a_lt_n : LT.lt a n
    b_nonneg : LE.le 0 b
    b_lt_n : LT.lt b n
    ⊢ And (LT.lt a (HAdd.hAdd n b)) (LT.lt b (HAdd.hAdd n a))
  -/
  exact ⟨lt_add_of_lt_of_nonneg a_lt_n b_nonneg, lt_add_of_lt_of_nonneg b_lt_n a_nonneg⟩
  /-
    🎉 no goals
  -/


theorem abs_eq (hb : 0 ≤ b) : |a| = b ↔ a = b ∨ a = -b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    hb : LE.le 0 b
    ⊢ Iff (Eq (abs a) b) (Or (Eq a b) (Eq a (Neg.neg b)))
  -/
  refine ⟨eq_or_eq_neg_of_abs_eq, ?_⟩
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    hb : LE.le 0 b
    ⊢ Or (Eq a b) (Eq a (Neg.neg b)) → Eq (abs a) b
  -/
                         /-
                           🎉 no goals
                         -/
  rintro (rfl | rfl) <;> simp only [abs_neg, abs_of_nonneg hb]
                         /-
                           🎉 no goals
                         -/


theorem abs_le_max_abs_abs (hab : a ≤ b) (hbc : b ≤ c) : |b| ≤ max |a| |c| :=
  abs_le'.2
        /-
          α : Type u_1
          inst✝ : LinearOrderedAddCommGroup α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          ⊢ LE.le b (Max.max (abs a) (abs c))
        -/
    ⟨by simp [hbc.trans (le_abs_self c)], by
        /-
          🎉 no goals
        -/
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        a b c : α
        hab : LE.le a b
        hbc : LE.le b c
        ⊢ LE.le (Neg.neg b) (Max.max (abs a) (abs c))
      -/
      simp [((@neg_le_neg_iff α ..).mpr hab).trans (neg_le_abs a)]⟩
      /-
        🎉 no goals
      -/


theorem min_abs_abs_le_abs_max : min |a| |b| ≤ |max a b| :=
  (le_total a b).elim (fun h => (min_le_right _ _).trans_eq <| congr_arg _ (max_eq_right h).symm)
    fun h => (min_le_left _ _).trans_eq <| congr_arg _ (max_eq_left h).symm


theorem min_abs_abs_le_abs_min : min |a| |b| ≤ |min a b| :=
  (le_total a b).elim (fun h => (min_le_left _ _).trans_eq <| congr_arg _ (min_eq_left h).symm)
    fun h => (min_le_right _ _).trans_eq <| congr_arg _ (min_eq_right h).symm


theorem abs_max_le_max_abs_abs : |max a b| ≤ max |a| |b| :=
  (le_total a b).elim (fun h => (congr_arg _ <| max_eq_right h).trans_le <| le_max_right _ _)
    fun h => (congr_arg _ <| max_eq_left h).trans_le <| le_max_left _ _


theorem abs_min_le_max_abs_abs : |min a b| ≤ max |a| |b| :=
  (le_total a b).elim (fun h => (congr_arg _ <| min_eq_left h).trans_le <| le_max_left _ _) fun h =>
    (congr_arg _ <| min_eq_right h).trans_le <| le_max_right _ _


theorem eq_of_abs_sub_eq_zero {a b : α} (h : |a - b| = 0) : a = b :=
  sub_eq_zero.1 <| (abs_eq_zero (α := α)).1 h


theorem abs_sub_le (a b c : α) : |a - c| ≤ |a - b| + |b - c| :=
  calc
                                      /-
                                        α : Type u_1
                                        inst✝ : LinearOrderedAddCommGroup α
                                        a b c : α
                                        ⊢ Eq (abs (HSub.hSub a c)) (abs (HAdd.hAdd (HSub.hSub a b) (HSub.hSub b c)))
                                      -/
    |a - c| = |a - b + (b - c)| := by rw [sub_add_sub_cancel]
                                      /-
                                        🎉 no goals
                                      -/
    _ ≤ |a - b| + |b - c| := abs_add _ _


theorem abs_add_three (a b c : α) : |a + b + c| ≤ |a| + |b| + |c| :=
  (abs_add _ _).trans (add_le_add_right (abs_add _ _) _)


theorem dist_bdd_within_interval {a b lb ub : α} (hal : lb ≤ a) (hau : a ≤ ub) (hbl : lb ≤ b)
    (hbu : b ≤ ub) : |a - b| ≤ ub - lb :=
  abs_sub_le_iff.2 ⟨sub_le_sub hau hbl, sub_le_sub hbu hal⟩


theorem eq_of_abs_sub_nonpos (h : |a - b| ≤ 0) : a = b :=
  eq_of_abs_sub_eq_zero (le_antisymm h (abs_nonneg (a - b)))


theorem abs_sub_nonpos : |a - b| ≤ 0 ↔ a = b :=
                            /-
                              α : Type u_1
                              inst✝ : LinearOrderedAddCommGroup α
                              a b : α
                              ⊢ Eq a b → LE.le (abs (HSub.hSub a b)) 0
                            -/
  ⟨eq_of_abs_sub_nonpos, by rintro rfl; rw [sub_self, abs_zero]⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem abs_sub_pos : 0 < |a - b| ↔ a ≠ b :=
  not_le.symm.trans abs_sub_nonpos.not


@[simp]
theorem abs_eq_self : |a| = a ↔ 0 ≤ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a : α
    ⊢ Iff (Eq (abs a) a) (LE.le 0 a)
  -/
  rw [abs_eq_max_neg, max_eq_left_iff, neg_le_self_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem abs_eq_neg_self : |a| = -a ↔ a ≤ 0 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a : α
    ⊢ Iff (Eq (abs a) (Neg.neg a)) (LE.le a 0)
  -/
  rw [abs_eq_max_neg, max_eq_right_iff, le_neg_self_iff]
  /-
    🎉 no goals
  -/


/-- For an element `a` of a linear ordered ring, either `abs a = a` and `0 ≤ a`,
    or `abs a = -a` and `a < 0`.
    Use cases on this lemma to automate linarith in inequalities -/
theorem abs_cases (a : α) : |a| = a ∧ 0 ≤ a ∨ |a| = -a ∧ a < 0 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a : α
    ⊢ Or (And (Eq (abs a) a) (LE.le 0 a)) (And (Eq (abs a) (Neg.neg a)) (LT.lt a 0))
  -/
  by_cases h : 0 ≤ a
    /-
      case pos
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a : α
      h : LE.le 0 a
      ⊢ Or (And (Eq (abs a) a) (LE.le 0 a)) (And (Eq (abs a) (Neg.neg a)) (LT.lt a 0))
    -/
  · left
    /-
      case pos.h
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a : α
      h : LE.le 0 a
      ⊢ And (Eq (abs a) a) (LE.le 0 a)
    -/
    exact ⟨abs_eq_self.mpr h, h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a : α
      h : Not (LE.le 0 a)
      ⊢ Or (And (Eq (abs a) a) (LE.le 0 a)) (And (Eq (abs a) (Neg.neg a)) (LT.lt a 0))
    -/
  · right
    /-
      case neg.h
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a : α
      h : Not (LE.le 0 a)
      ⊢ And (Eq (abs a) (Neg.neg a)) (LT.lt a 0)
    -/
    push_neg at h
    /-
      case neg.h
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a : α
      h : LT.lt a 0
      ⊢ And (Eq (abs a) (Neg.neg a)) (LT.lt a 0)
    -/
    exact ⟨abs_eq_neg_self.mpr (le_of_lt h), h⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem max_zero_add_max_neg_zero_eq_abs_self (a : α) : max a 0 + max (-a) 0 = |a| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a : α
    ⊢ Eq (HAdd.hAdd (Max.max a 0) (Max.max (Neg.neg a) 0)) (abs a)
  -/
  symm
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a : α
    ⊢ Eq (abs a) (HAdd.hAdd (Max.max a 0) (Max.max (Neg.neg a) 0))
  -/
                                         /-
                                           🎉 no goals
                                         -/
  rcases le_total 0 a with (ha | ha) <;> simp [ha]
                                         /-
                                           🎉 no goals
                                         -/


