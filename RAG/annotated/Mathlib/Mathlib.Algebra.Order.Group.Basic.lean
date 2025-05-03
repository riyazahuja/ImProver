@[to_additive zsmul_left_strictMono]
lemma zpow_right_strictMono (ha : 1 < a) : StrictMono fun n : ℤ ↦ a ^ n := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LT.lt 1 a
    ⊢ StrictMono fun n => HPow.hPow a n
  -/
  refine strictMono_int_of_lt_succ fun n ↦ ?_
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LT.lt 1 a
    n : Int
    ⊢ LT.lt (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
  -/
  rw [zpow_add_one]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LT.lt 1 a
    n : Int
    ⊢ LT.lt (HPow.hPow a n) (HMul.hMul (HPow.hPow a n) a)
  -/
  exact lt_mul_of_one_lt_right' (a ^ n) ha
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-19")] alias zsmul_strictMono_left := zsmul_left_strictMono


@[to_additive zsmul_pos] lemma one_lt_zpow (ha : 1 < a) (hn : 0 < n) : 1 < a ^ n := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    n : Int
    a : α
    ha : LT.lt 1 a
    hn : LT.lt 0 n
    ⊢ LT.lt 1 (HPow.hPow a n)
  -/
  simpa using zpow_right_strictMono ha hn
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-13")] alias one_lt_zpow' := one_lt_zpow


@[to_additive zsmul_left_strictAnti]
lemma zpow_right_strictAnti (ha : a < 1) : StrictAnti fun n : ℤ ↦ a ^ n := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LT.lt a 1
    ⊢ StrictAnti fun n => HPow.hPow a n
  -/
  refine strictAnti_int_of_succ_lt fun n ↦ ?_
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LT.lt a 1
    n : Int
    ⊢ LT.lt (HPow.hPow a (HAdd.hAdd n 1)) (HPow.hPow a n)
  -/
  rw [zpow_add_one]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LT.lt a 1
    n : Int
    ⊢ LT.lt (HMul.hMul (HPow.hPow a n) a) (HPow.hPow a n)
  -/
  exact mul_lt_of_lt_one_right' (a ^ n) ha
  /-
    🎉 no goals
  -/


@[to_additive zsmul_left_inj]
lemma zpow_right_inj (ha : 1 < a) {m n : ℤ} : a ^ m = a ^ n ↔ m = n :=
  (zpow_right_strictMono ha).injective.eq_iff


@[to_additive zsmul_left_mono]
lemma zpow_right_mono (ha : 1 ≤ a) : Monotone fun n : ℤ ↦ a ^ n := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LE.le 1 a
    ⊢ Monotone fun n => HPow.hPow a n
  -/
  refine monotone_int_of_le_succ fun n ↦ ?_
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LE.le 1 a
    n : Int
    ⊢ LE.le (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
  -/
  rw [zpow_add_one]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a : α
    ha : LE.le 1 a
    n : Int
    ⊢ LE.le (HPow.hPow a n) (HMul.hMul (HPow.hPow a n) a)
  -/
  exact le_mul_of_one_le_right' ha
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-13")] alias zpow_mono_right := zpow_right_mono


@[to_additive (attr := gcongr) zsmul_le_zsmul_left]
lemma zpow_le_zpow_right (ha : 1 ≤ a) (h : m ≤ n) : a ^ m ≤ a ^ n := zpow_right_mono ha h


@[deprecated (since := "2024-11-13")] alias zpow_le_zpow := zpow_le_zpow_right


@[to_additive (attr := gcongr) zsmul_lt_zsmul_left]
lemma zpow_lt_zpow_right (ha : 1 < a) (h : m < n) : a ^ m < a ^ n := zpow_right_strictMono ha h


@[deprecated (since := "2024-11-13")] alias zpow_lt_zpow := zpow_lt_zpow_right


@[to_additive zsmul_le_zsmul_iff_left]
lemma zpow_le_zpow_iff_right (ha : 1 < a) : a ^ m ≤ a ^ n ↔ m ≤ n :=
  (zpow_right_strictMono ha).le_iff_le


@[deprecated (since := "2024-11-13")] alias zpow_le_zpow_iff := zpow_le_zpow_iff_right


@[to_additive zsmul_lt_zsmul_iff_left]
lemma zpow_lt_zpow_iff_right (ha : 1 < a) : a ^ m < a ^ n ↔ m < n :=
  (zpow_right_strictMono ha).lt_iff_lt


@[deprecated (since := "2024-11-13")] alias zpow_lt_zpow_iff := zpow_lt_zpow_iff_right


@[to_additive zsmul_strictMono_right]
lemma zpow_left_strictMono (hn : 0 < n) : StrictMono ((· ^ n) : α → α) := fun a b hab => by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    n : Int
    hn : LT.lt 0 n
    a b : α
    hab : LT.lt a b
    ⊢ LT.lt ((fun x => HPow.hPow x n) a) ((fun x => HPow.hPow x n) b)
  -/
  rw [← one_lt_div', ← div_zpow]; exact one_lt_zpow (one_lt_div'.2 hab) hn
                                  /-
                                    🎉 no goals
                                  -/


@[deprecated (since := "2024-11-13")] alias zpow_strictMono_left := zpow_left_strictMono


@[to_additive zsmul_mono_right]
lemma zpow_left_mono (hn : 0 ≤ n) : Monotone ((· ^ n) : α → α) := fun a b hab => by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    n : Int
    hn : LE.le 0 n
    a b : α
    hab : LE.le a b
    ⊢ LE.le ((fun x => HPow.hPow x n) a) ((fun x => HPow.hPow x n) b)
  -/
  rw [← one_le_div', ← div_zpow]; exact one_le_zpow (one_le_div'.2 hab) hn
                                  /-
                                    🎉 no goals
                                  -/


@[deprecated (since := "2024-11-13")] alias zpow_mono_left := zpow_left_mono


@[to_additive (attr := gcongr) zsmul_le_zsmul_right]
lemma zpow_le_zpow_left (hn : 0 ≤ n) (h : a ≤ b) : a ^ n ≤ b ^ n := zpow_left_mono α hn h


@[deprecated (since := "2024-11-13")] alias zpow_le_zpow' := zpow_le_zpow_left


@[to_additive (attr := gcongr) zsmul_lt_zsmul_right]
lemma zpow_lt_zpow_left (hn : 0 < n) (h : a < b) : a ^ n < b ^ n := zpow_left_strictMono α hn h


@[deprecated (since := "2024-11-13")] alias zpow_lt_zpow' := zpow_lt_zpow_left


@[to_additive zsmul_le_zsmul_iff_right]
lemma zpow_le_zpow_iff_left (hn : 0 < n) : a ^ n ≤ b ^ n ↔ a ≤ b :=
  (zpow_left_strictMono α hn).le_iff_le


@[deprecated (since := "2024-11-13")] alias zpow_le_zpow_iff' := zpow_le_zpow_iff_left


@[to_additive zsmul_lt_zsmul_iff_right]
lemma zpow_lt_zpow_iff_left (hn : 0 < n) : a ^ n < b ^ n ↔ a < b :=
  (zpow_left_strictMono α hn).lt_iff_lt


@[deprecated (since := "2024-11-13")] alias zpow_lt_zpow_iff' := zpow_lt_zpow_iff_left


@[to_additive zsmul_right_injective
"See also `smul_right_injective`. TODO: provide a `NoZeroSMulDivisors` instance. We can't do
that here because importing that definition would create import cycles."]
lemma zpow_left_injective (hn : n ≠ 0) : Injective ((· ^ n) : α → α) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    n : Int
    hn : Ne n 0
    ⊢ Function.Injective fun x => HPow.hPow x n
  -/
  obtain hn | hn := hn.lt_or_lt
  · refine fun a b (hab : a ^ n = b ^ n) ↦
      (zpow_left_strictMono _ <| Int.neg_pos_of_neg hn).injective ?_
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Int
      hn✝ : Ne n 0
      hn : LT.lt n 0
      a b : α
      hab : Eq (HPow.hPow a n) (HPow.hPow b n)
      ⊢ Eq (HPow.hPow a (Neg.neg n)) (HPow.hPow b (Neg.neg n))
    -/
    rw [zpow_neg, zpow_neg, hab]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Int
      hn✝ : Ne n 0
      hn : LT.lt 0 n
      ⊢ Function.Injective fun x => HPow.hPow x n
    -/
  · exact (zpow_left_strictMono _ hn).injective
    /-
      🎉 no goals
    -/


@[to_additive zsmul_right_inj]
lemma zpow_left_inj (hn : n ≠ 0) : a ^ n = b ^ n ↔ a = b := (zpow_left_injective hn).eq_iff


/-- Alias of `zpow_left_inj`, for ease of discovery alongside `zsmul_le_zsmul_iff'` and
`zsmul_lt_zsmul_iff'`. -/
@[to_additive "Alias of `zsmul_right_inj`, for ease of discovery alongside `zsmul_le_zsmul_iff'` and
`zsmul_lt_zsmul_iff'`."]
lemma zpow_eq_zpow_iff' (hn : n ≠ 0) : a ^ n = b ^ n ↔ a = b := zpow_left_inj hn


variable (α) in
/-- A nontrivial densely linear ordered commutative group can't be a cyclic group. -/
@[to_additive
  "A nontrivial densely linear ordered additive commutative group can't be a cyclic group."]
theorem not_isCyclic_of_denselyOrdered [DenselyOrdered α] [Nontrivial α] : ¬IsCyclic α := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedCommGroup α
    inst✝¹ : DenselyOrdered α
    inst✝ : Nontrivial α
    ⊢ Not (IsCyclic α)
  -/
  intro h
  /-
    α : Type u_1
    inst✝² : LinearOrderedCommGroup α
    inst✝¹ : DenselyOrdered α
    inst✝ : Nontrivial α
    h : IsCyclic α
    ⊢ False
  -/
  rcases exists_zpow_surjective α with ⟨a, ha⟩
  /-
    case intro
    α : Type u_1
    inst✝² : LinearOrderedCommGroup α
    inst✝¹ : DenselyOrdered α
    inst✝ : Nontrivial α
    h : IsCyclic α
    a : α
    ha : Function.Surjective fun x => HPow.hPow a x
    ⊢ False
  -/
  rcases lt_trichotomy a 1 with hlt | rfl | hlt
    /-
      case intro.inl
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt a 1
      ⊢ False
    -/
  · rcases exists_between hlt with ⟨b, hab, hb⟩
    /-
      case intro.inl.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt a 1
      b : α
      hab : LT.lt a b
      hb : LT.lt b 1
      ⊢ False
    -/
    rcases ha b with ⟨k, rfl⟩
    /-
      case intro.inl.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt a 1
      k : Int
      hab : LT.lt a ((fun x => HPow.hPow a x) k)
      hb : LT.lt ((fun x => HPow.hPow a x) k) 1
      ⊢ False
    -/
    suffices 0 < k ∧ k < 1 by omega
    /-
      case intro.inl.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt a 1
      k : Int
      hab : LT.lt a ((fun x => HPow.hPow a x) k)
      hb : LT.lt ((fun x => HPow.hPow a x) k) 1
      ⊢ And (LT.lt 0 k) (LT.lt k 1)
    -/
    rw [← one_lt_inv'] at hlt
    /-
      case intro.inl.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 (Inv.inv a)
      k : Int
      hab : LT.lt a ((fun x => HPow.hPow a x) k)
      hb : LT.lt ((fun x => HPow.hPow a x) k) 1
      ⊢ And (LT.lt 0 k) (LT.lt k 1)
    -/
    simp_rw [← zpow_lt_zpow_iff_right hlt]
    /-
      case intro.inl.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 (Inv.inv a)
      k : Int
      hab : LT.lt a ((fun x => HPow.hPow a x) k)
      hb : LT.lt ((fun x => HPow.hPow a x) k) 1
      ⊢ And (LT.lt (HPow.hPow (Inv.inv a) 0) (HPow.hPow (Inv.inv a) k)) (LT.lt (HPow …
    -/
    simp_all
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inl
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      ha : Function.Surjective fun x => HPow.hPow 1 x
      ⊢ False
    -/
  · rcases exists_ne (1 : α) with ⟨b, hb⟩
    /-
      case intro.inr.inl.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      ha : Function.Surjective fun x => HPow.hPow 1 x
      b : α
      hb : Ne b 1
      ⊢ False
    -/
    simpa [hb.symm] using ha b
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inr
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 a
      ⊢ False
    -/
  · rcases exists_between hlt with ⟨b, hb, hba⟩
    /-
      case intro.inr.inr.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 a
      b : α
      hb : LT.lt 1 b
      hba : LT.lt b a
      ⊢ False
    -/
    rcases ha b with ⟨k, rfl⟩
    /-
      case intro.inr.inr.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 a
      k : Int
      hb : LT.lt 1 ((fun x => HPow.hPow a x) k)
      hba : LT.lt ((fun x => HPow.hPow a x) k) a
      ⊢ False
    -/
    suffices 0 < k ∧ k < 1 by omega
    /-
      case intro.inr.inr.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 a
      k : Int
      hb : LT.lt 1 ((fun x => HPow.hPow a x) k)
      hba : LT.lt ((fun x => HPow.hPow a x) k) a
      ⊢ And (LT.lt 0 k) (LT.lt k 1)
    -/
    simp_rw [← zpow_lt_zpow_iff_right hlt]
    /-
      case intro.inr.inr.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommGroup α
      inst✝¹ : DenselyOrdered α
      inst✝ : Nontrivial α
      h : IsCyclic α
      a : α
      ha : Function.Surjective fun x => HPow.hPow a x
      hlt : LT.lt 1 a
      k : Int
      hb : LT.lt 1 ((fun x => HPow.hPow a x) k)
      hba : LT.lt ((fun x => HPow.hPow a x) k) a
      ⊢ And (LT.lt (HPow.hPow a 0) (HPow.hPow a k)) (LT.lt (HPow.hPow a k) (HPow.hPo …
    -/
    simp_all
    /-
      🎉 no goals
    -/


