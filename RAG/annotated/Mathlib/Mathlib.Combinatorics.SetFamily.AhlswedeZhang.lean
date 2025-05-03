private lemma binomial_sum_eq (h : n < m) :
    ∑ i ∈ range (n + 1), (n.choose i * (m - n) / ((m - i) * m.choose i) : ℚ) = 1 := by
  /-
    m n : Nat
    h : LT.lt n m
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HDiv.hDiv (HMul.hMul (↑(n.ch …
  -/
  set f : ℕ → ℚ := fun i ↦ n.choose i * (m.choose i : ℚ)⁻¹ with hf
  suffices ∀ i ∈ range (n + 1), f i - f (i + 1) = n.choose i * (m - n) / ((m - i) * m.choose i) by
    rw [← sum_congr rfl this, sum_range_sub', hf]
    simp [choose_self, choose_zero_right, choose_eq_zero_of_lt h]
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    ⊢ ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → Eq (HSub.hSub …
  -/
  intro i h₁
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁ : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  rw [mem_range] at h₁
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁ : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  have h₁ := le_of_lt_succ h₁
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  have h₂ := h₁.trans_lt h
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  have h₃ := h₂.le
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  have hi₄ : (i + 1 : ℚ) ≠ 0 := i.cast_add_one_ne_zero
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  have := congr_arg ((↑) : ℕ → ℚ) (choose_succ_right_eq m i)
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this : Eq ↑(HMul.hMul (m.choose (HAdd.hAdd i 1)) (HAdd.hAdd i 1)) ↑(HMul.hMul  …
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  push_cast at this
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul.h …
    ⊢ Eq (HSub.hSub (f i) (f (HAdd.hAdd i 1))) (HDiv.hDiv (HMul.hMul (↑(n.choose i …
  -/
  dsimp [f, hf]
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul.h …
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  rw [(eq_mul_inv_iff_mul_eq₀ hi₄).mpr this]
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul.h …
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  have := congr_arg ((↑) : ℕ → ℚ) (choose_succ_right_eq n i)
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this✝ : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul. …
    this : Eq ↑(HMul.hMul (n.choose (HAdd.hAdd i 1)) (HAdd.hAdd i 1)) ↑(HMul.hMul  …
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  push_cast at this
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this✝ : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul. …
    this : Eq (HMul.hMul (↑(n.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul.h …
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  rw [(eq_mul_inv_iff_mul_eq₀ hi₄).mpr this]
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this✝ : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul. …
    this : Eq (HMul.hMul (↑(n.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul.h …
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  have : (m - i : ℚ) ≠ 0 := sub_ne_zero_of_ne (cast_lt.mpr h₂).ne'
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this✝¹ : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul …
    this✝ : Eq (HMul.hMul (↑(n.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul. …
    this : Ne (HSub.hSub ↑m ↑i) 0
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  have : (m.choose i : ℚ) ≠ 0 := cast_ne_zero.2 (choose_pos h₂.le).ne'
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this✝² : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul …
    this✝¹ : Eq (HMul.hMul (↑(n.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul …
    this✝ : Ne (HSub.hSub ↑m ↑i) 0
    this : Ne (↑(m.choose i)) 0
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))) (HMul.hMul …
  -/
  field_simp
  /-
    m n : Nat
    h : LT.lt n m
    f : Nat → Rat := fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    hf : Eq f fun i => HMul.hMul (↑(n.choose i)) (Inv.inv ↑(m.choose i))
    i : Nat
    h₁✝ : LT.lt i (HAdd.hAdd n 1)
    h₁ : LE.le i n
    h₂ : LT.lt i m
    h₃ : LE.le i m
    hi₄ : Ne (HAdd.hAdd (↑i) 1) 0
    this✝² : Eq (HMul.hMul (↑(m.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul …
    this✝¹ : Eq (HMul.hMul (↑(n.choose (HAdd.hAdd i 1))) (HAdd.hAdd (↑i) 1)) (HMul …
    this✝ : Ne (HSub.hSub ↑m ↑i) 0
    this : Ne (↑(m.choose i)) 0
    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (↑(n.choose i)) (HMul.hMul (↑(m.choose i …
  -/
  ring
  /-
    🎉 no goals
  -/


private lemma Fintype.sum_div_mul_card_choose_card :
    ∑ s : Finset α, (card α / ((card α - #s) * (card α).choose #s) : ℚ) =
      card α * ∑ k ∈ range (card α), (↑k)⁻¹ + 1 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    ⊢ Eq (Finset.univ.sum fun s => HDiv.hDiv (↑(Fintype.card α)) (HMul.hMul (HSub. …
  -/
  rw [← powerset_univ, powerset_card_disjiUnion, sum_disjiUnion]
  have : ∀ {x : ℕ}, ∀ s ∈ powersetCard x (univ : Finset α),
    (card α / ((card α - #s) * (card α).choose #s) : ℚ) =
      card α / ((card α - x) * (card α).choose x) := by
    intros n s hs
    rw [mem_powersetCard_univ.1 hs]
  simp_rw [sum_congr rfl this, sum_const, card_powersetCard, card_univ, nsmul_eq_mul, mul_div,
    mul_comm, ← mul_div]
  rw [← mul_sum, ← mul_inv_cancel₀ (cast_ne_zero.mpr card_ne_zero : (card α : ℚ) ≠ 0), ← mul_add,
    add_comm _ ((card α)⁻¹ : ℚ), ← sum_insert (f := fun x : ℕ ↦ (x⁻¹ : ℚ)) not_mem_range_self,
    ← range_succ]
  have (n) (hn : n ∈ range (card α + 1)) :
      ((card α).choose n / ((card α - n) * (card α).choose n) : ℚ) = (card α - n : ℚ)⁻¹ := by
    rw [div_mul_cancel_right₀]
    exact cast_ne_zero.2 (choose_pos <| mem_range_succ_iff.1 hn).ne'
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    this✝ : ∀ {x : Nat} (s : Finset α), Membership.mem (Finset.powersetCard x Fins …
    this : ∀ (n : Nat), Membership.mem (Finset.range (HAdd.hAdd (Fintype.card α) 1 …
    ⊢ Eq (HMul.hMul (↑(Fintype.card α)) ((Finset.range (HAdd.hAdd (Fintype.card α) …
  -/
  simp only [sum_congr rfl this, mul_eq_mul_left_iff, cast_eq_zero]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    this✝ : ∀ {x : Nat} (s : Finset α), Membership.mem (Finset.powersetCard x Fins …
    this : ∀ (n : Nat), Membership.mem (Finset.range (HAdd.hAdd (Fintype.card α) 1 …
    ⊢ Or (Eq ((Finset.range (HAdd.hAdd (Fintype.card α) 1)).sum fun x => Inv.inv ( …
  -/
  convert Or.inl <| sum_range_reflect _ _ with a ha
  /-
    case h.e'_1.h.e'_2.a
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    this✝ : ∀ {x : Nat} (s : Finset α), Membership.mem (Finset.powersetCard x Fins …
    this : ∀ (n : Nat), Membership.mem (Finset.range (HAdd.hAdd (Fintype.card α) 1 …
    a : Nat
    ha : Membership.mem (Finset.range (HAdd.hAdd (Fintype.card α) 1)) a
    ⊢ Eq (Inv.inv (HSub.hSub ↑(Fintype.card α) ↑a)) (Inv.inv ↑(HSub.hSub (HSub.hSu …
  -/
  rw [add_tsub_cancel_right, cast_sub (mem_range_succ_iff.mp ha)]
  /-
    🎉 no goals
  -/


private lemma sup_aux [DecidableRel (α := α) (· ≤ ·)] :
    a ∈ lowerClosure s → {b ∈ s | a ≤ b}.Nonempty :=
  fun ⟨b, hb, hab⟩ ↦ ⟨b, mem_filter.2 ⟨hb, hab⟩⟩


private lemma lower_aux [DecidableEq α] :
    a ∈ lowerClosure ↑(s ∪ t) ↔ a ∈ lowerClosure s ∨ a ∈ lowerClosure t := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeSup α
    s t : Finset α
    a : α
    inst✝ : DecidableEq α
    ⊢ Iff (Membership.mem (lowerClosure ↑(Union.union s t)) a) (Or (Membership.mem …
  -/
  rw [coe_union, lowerClosure_union, LowerSet.mem_sup_iff]
  /-
    🎉 no goals
  -/


/-- The supremum of the elements of `s` less than `a` if there are some, otherwise `⊤`. -/
def truncatedSup (s : Finset α) (a : α) : α :=
  if h : a ∈ lowerClosure s then {b ∈ s | a ≤ b}.sup' (sup_aux h) id else ⊤


lemma truncatedSup_of_mem (h : a ∈ lowerClosure s) :
    truncatedSup s a = {b ∈ s | a ≤ b}.sup' (sup_aux h) id := dif_pos h


lemma truncatedSup_of_not_mem (h : a ∉ lowerClosure s) : truncatedSup s a = ⊤ := dif_neg h


                                                                                               /-
                                                                                                 α : Type u_1
                                                                                                 inst✝² : SemilatticeSup α
                                                                                                 inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                                                                                 inst✝ : OrderTop α
                                                                                                 a : α
                                                                                                 ⊢ Not (Membership.mem (lowerClosure ↑EmptyCollection.emptyCollection) a)
                                                                                               -/
@[simp] lemma truncatedSup_empty (a : α) : truncatedSup ∅ a = ⊤ := truncatedSup_of_not_mem (by simp)
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[simp] lemma truncatedSup_singleton (b a : α) : truncatedSup {b} a = if a ≤ b then b else ⊤ := by
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    b a : α
    ⊢ Eq ((Singleton.singleton b).truncatedSup a) (ite (LE.le a b) b Top.top)
  -/
                                     /-
                                       🎉 no goals
                                     -/
  simp [truncatedSup]; split_ifs <;> simp [Finset.filter_true_of_mem, *]
                                     /-
                                       🎉 no goals
                                     -/


lemma le_truncatedSup : a ≤ truncatedSup s a := by
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    ⊢ LE.le a (s.truncatedSup a)
  -/
  rw [truncatedSup]
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    ⊢ LE.le a (dite (Membership.mem (lowerClosure ↑s) a) (fun h => (Finset.filter  …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝² : SemilatticeSup α
      s : Finset α
      a : α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : OrderTop α
      h : Membership.mem (lowerClosure ↑s) a
      ⊢ LE.le a ((Finset.filter (fun b => LE.le a b) s).sup' ⋯ id)
    -/
  · obtain ⟨ℬ, hb, h⟩ := h
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝² : SemilatticeSup α
      s : Finset α
      a : α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : OrderTop α
      ℬ : α
      hb : Membership.mem (↑s) ℬ
      h : LE.le a ℬ
      ⊢ LE.le a ((Finset.filter (fun b => LE.le a b) s).sup' ⋯ id)
    -/
    exact h.trans <| le_sup' id <| mem_filter.2 ⟨hb, h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : SemilatticeSup α
      s : Finset α
      a : α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : OrderTop α
      h : Not (Membership.mem (lowerClosure ↑s) a)
      ⊢ LE.le a Top.top
    -/
  · exact le_top
    /-
      🎉 no goals
    -/


lemma map_truncatedSup [DecidableRel (α := β) (· ≤ ·)] (e : α ≃o β) (s : Finset α) (a : α) :
    e (truncatedSup s a) = truncatedSup (s.map e.toEquiv.toEmbedding) (e a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeSup α
    inst✝⁴ : SemilatticeSup β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : OrderTop α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    e : OrderIso α β
    s : Finset α
    a : α
    ⊢ Eq (e (s.truncatedSup a)) ((Finset.map e.toEmbedding s).truncatedSup (e a))
  -/
  have : e a ∈ lowerClosure (s.map e.toEquiv.toEmbedding : Set β) ↔ a ∈ lowerClosure s := by simp
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeSup α
    inst✝⁴ : SemilatticeSup β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : OrderTop α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    e : OrderIso α β
    s : Finset α
    a : α
    this : Iff (Membership.mem (lowerClosure ↑(Finset.map e.toEmbedding s)) (e a)) …
    ⊢ Eq (e (s.truncatedSup a)) ((Finset.map e.toEmbedding s).truncatedSup (e a))
  -/
  simp_rw [truncatedSup, apply_dite e, map_finset_sup', map_top, this]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeSup α
    inst✝⁴ : SemilatticeSup β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : OrderTop α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    e : OrderIso α β
    s : Finset α
    a : α
    this : Iff (Membership.mem (lowerClosure ↑(Finset.map e.toEmbedding s)) (e a)) …
    ⊢ Eq (dite (Membership.mem (lowerClosure ↑s) a) (fun h => (Finset.filter (fun  …
  -/
  congr with h
  simp only [filter_map, Function.comp_def, Equiv.coe_toEmbedding, RelIso.coe_fn_toEquiv,
    OrderIso.le_iff_le, id]
  /-
    case e_t.h
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeSup α
    inst✝⁴ : SemilatticeSup β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : OrderTop α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    e : OrderIso α β
    s : Finset α
    a : α
    this : Iff (Membership.mem (lowerClosure ↑(Finset.map e.toEmbedding s)) (e a)) …
    h : Membership.mem (lowerClosure ↑s) a
    ⊢ Eq ((Finset.filter (fun b => LE.le a b) s).sup' ⋯ fun x => e x) ((Finset.map …
  -/
  rw [sup'_map]
  -- TODO: Why can't `simp` use `Finset.sup'_map`?
  /-
    case e_t.h
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeSup α
    inst✝⁴ : SemilatticeSup β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : OrderTop α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    e : OrderIso α β
    s : Finset α
    a : α
    this : Iff (Membership.mem (lowerClosure ↑(Finset.map e.toEmbedding s)) (e a)) …
    h : Membership.mem (lowerClosure ↑s) a
    ⊢ Eq ((Finset.filter (fun b => LE.le a b) s).sup' ⋯ fun x => e x) ((Finset.fil …
  -/
  simp only [sup'_map, Equiv.coe_toEmbedding, RelIso.coe_fn_toEquiv, Function.comp_apply]
  /-
    🎉 no goals
  -/


lemma truncatedSup_of_isAntichain (hs : IsAntichain (· ≤ ·) (s : Set α)) (ha : a ∈ s) :
    truncatedSup s a = a := by
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    ⊢ Eq (s.truncatedSup a) a
  -/
  refine le_antisymm ?_ le_truncatedSup
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    ⊢ LE.le (s.truncatedSup a) a
  -/
  simp_rw [truncatedSup_of_mem (subset_lowerClosure ha), sup'_le_iff, mem_filter]
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    ⊢ ∀ (b : α), And (Membership.mem s b) (LE.le a b) → LE.le (id b) a
  -/
  rintro b ⟨hb, hab⟩
  /-
    case intro
    α : Type u_1
    inst✝² : SemilatticeSup α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : OrderTop α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LE.le a b
    ⊢ LE.le (id b) a
  -/
  exact (hs.eq ha hb hab).ge
  /-
    🎉 no goals
  -/


lemma truncatedSup_union (hs : a ∈ lowerClosure s) (ht : a ∈ lowerClosure t) :
    truncatedSup (s ∪ t) a = truncatedSup s a ⊔ truncatedSup t a := by
  simpa only [truncatedSup_of_mem, hs, ht, lower_aux.2 (Or.inl hs), filter_union] using
    sup'_union _ _ _


lemma truncatedSup_union_left (hs : a ∈ lowerClosure s) (ht : a ∉ lowerClosure t) :
    truncatedSup (s ∪ t) a = truncatedSup s a := by
  /-
    α : Type u_1
    inst✝³ : SemilatticeSup α
    s t : Finset α
    a : α
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : OrderTop α
    inst✝ : DecidableEq α
    hs : Membership.mem (lowerClosure ↑s) a
    ht : Not (Membership.mem (lowerClosure ↑t) a)
    ⊢ Eq ((Union.union s t).truncatedSup a) (s.truncatedSup a)
  -/
  simp only [mem_lowerClosure, mem_coe, exists_prop, not_exists, not_and] at ht
  simp only [truncatedSup_of_mem, hs, filter_union, filter_false_of_mem ht, union_empty,
    lower_aux.2 (Or.inl hs), ht]


lemma truncatedSup_union_right (hs : a ∉ lowerClosure s) (ht : a ∈ lowerClosure t) :
                                                    /-
                                                      α : Type u_1
                                                      inst✝³ : SemilatticeSup α
                                                      s t : Finset α
                                                      a : α
                                                      inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
                                                      inst✝¹ : OrderTop α
                                                      inst✝ : DecidableEq α
                                                      hs : Not (Membership.mem (lowerClosure ↑s) a)
                                                      ht : Membership.mem (lowerClosure ↑t) a
                                                      ⊢ Eq ((Union.union s t).truncatedSup a) (t.truncatedSup a)
                                                    -/
    truncatedSup (s ∪ t) a = truncatedSup t a := by rw [union_comm, truncatedSup_union_left ht hs]
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma truncatedSup_union_of_not_mem (hs : a ∉ lowerClosure s) (ht : a ∉ lowerClosure t) :
    truncatedSup (s ∪ t) a = ⊤ := truncatedSup_of_not_mem fun h ↦ (lower_aux.1 h).elim hs ht


private lemma inf_aux [DecidableRel (α := α) (· ≤ ·)] :
    a ∈ upperClosure s → {b ∈ s | b ≤ a}.Nonempty :=
  fun ⟨b, hb, hab⟩ ↦ ⟨b, mem_filter.2 ⟨hb, hab⟩⟩


private lemma upper_aux [DecidableEq α] :
    a ∈ upperClosure ↑(s ∪ t) ↔ a ∈ upperClosure s ∨ a ∈ upperClosure t := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    s t : Finset α
    a : α
    inst✝ : DecidableEq α
    ⊢ Iff (Membership.mem (upperClosure ↑(Union.union s t)) a) (Or (Membership.mem …
  -/
  rw [coe_union, upperClosure_union, UpperSet.mem_inf_iff]
  /-
    🎉 no goals
  -/


/-- The infimum of the elements of `s` less than `a` if there are some, otherwise `⊥`. -/
def truncatedInf (s : Finset α) (a : α) : α :=
  if h : a ∈ upperClosure s then {b ∈ s | b ≤ a}.inf' (inf_aux h) id else ⊥


lemma truncatedInf_of_mem (h : a ∈ upperClosure s) :
    truncatedInf s a = {b ∈ s | b ≤ a}.inf' (inf_aux h) id := dif_pos h


lemma truncatedInf_of_not_mem (h : a ∉ upperClosure s) : truncatedInf s a = ⊥ := dif_neg h


lemma truncatedInf_le : truncatedInf s a ≤ a := by
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    ⊢ LE.le (s.truncatedInf a) a
  -/
  unfold truncatedInf
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    ⊢ LE.le (dite (Membership.mem (upperClosure ↑s) a) (fun h => (Finset.filter (f …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝² : SemilatticeInf α
      s : Finset α
      a : α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : BoundedOrder α
      h : Membership.mem (upperClosure ↑s) a
      ⊢ LE.le ((Finset.filter (fun b => LE.le b a) s).inf' ⋯ id) a
    -/
  · obtain ⟨b, hb, hba⟩ := h
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝² : SemilatticeInf α
      s : Finset α
      a : α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : BoundedOrder α
      b : α
      hb : Membership.mem (↑s) b
      hba : LE.le b a
      ⊢ LE.le ((Finset.filter (fun b => LE.le b a) s).inf' ⋯ id) a
    -/
    exact hba.trans' <| inf'_le id <| mem_filter.2 ⟨hb, ‹_›⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : SemilatticeInf α
      s : Finset α
      a : α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : BoundedOrder α
      h : Not (Membership.mem (upperClosure ↑s) a)
      ⊢ LE.le Bot.bot a
    -/
  · exact bot_le
    /-
      🎉 no goals
    -/


                                                                                               /-
                                                                                                 α : Type u_1
                                                                                                 inst✝² : SemilatticeInf α
                                                                                                 inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                                                                                 inst✝ : BoundedOrder α
                                                                                                 a : α
                                                                                                 ⊢ Not (Membership.mem (upperClosure ↑EmptyCollection.emptyCollection) a)
                                                                                               -/
@[simp] lemma truncatedInf_empty (a : α) : truncatedInf ∅ a = ⊥ := truncatedInf_of_not_mem (by simp)
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[simp] lemma truncatedInf_singleton (b a : α) : truncatedInf {b} a = if b ≤ a then b else ⊥ := by
  simp only [truncatedInf, coe_singleton, upperClosure_singleton, UpperSet.mem_Ici_iff,
    filter_congr_decidable, id_eq]
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    b a : α
    ⊢ Eq (dite (LE.le b a) (fun h => (Finset.filter (fun b => LE.le b a) (Singleto …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [Finset.filter_true_of_mem, *]
                /-
                  🎉 no goals
                -/


lemma map_truncatedInf (e : α ≃o β) (s : Finset α) (a : α) :
    e (truncatedInf s a) = truncatedInf (s.map e.toEquiv.toEmbedding) (e a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeInf α
    inst✝⁴ : SemilatticeInf β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    e : OrderIso α β
    s : Finset α
    a : α
    ⊢ Eq (e (s.truncatedInf a)) ((Finset.map e.toEmbedding s).truncatedInf (e a))
  -/
  have : e a ∈ upperClosure (s.map e.toEquiv.toEmbedding) ↔ a ∈ upperClosure s := by simp
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeInf α
    inst✝⁴ : SemilatticeInf β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    e : OrderIso α β
    s : Finset α
    a : α
    this : Iff (Membership.mem (upperClosure ↑(Finset.map e.toEmbedding s)) (e a)) …
    ⊢ Eq (e (s.truncatedInf a)) ((Finset.map e.toEmbedding s).truncatedInf (e a))
  -/
  simp_rw [truncatedInf, apply_dite e, map_finset_inf', map_bot, this]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : SemilatticeInf α
    inst✝⁴ : SemilatticeInf β
    inst✝³ : BoundedOrder β
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    e : OrderIso α β
    s : Finset α
    a : α
    this : Iff (Membership.mem (upperClosure ↑(Finset.map e.toEmbedding s)) (e a)) …
    ⊢ Eq (dite (Membership.mem (upperClosure ↑s) a) (fun h => (Finset.filter (fun  …
  -/
  congr with h
  simp only [filter_map, Function.comp_def, Equiv.coe_toEmbedding, RelIso.coe_fn_toEquiv,
    OrderIso.le_iff_le, id, inf'_map]


lemma truncatedInf_of_isAntichain (hs : IsAntichain (· ≤ ·) (s : Set α)) (ha : a ∈ s) :
    truncatedInf s a = a := by
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    ⊢ Eq (s.truncatedInf a) a
  -/
  refine le_antisymm truncatedInf_le ?_
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    ⊢ LE.le a (s.truncatedInf a)
  -/
  simp_rw [truncatedInf_of_mem (subset_upperClosure ha), le_inf'_iff, mem_filter]
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    ⊢ ∀ (b : α), And (Membership.mem s b) (LE.le b a) → LE.le a (id b)
  -/
  rintro b ⟨hb, hba⟩
  /-
    case intro
    α : Type u_1
    inst✝² : SemilatticeInf α
    s : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) ↑s
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hba : LE.le b a
    ⊢ LE.le a (id b)
  -/
  exact (hs.eq hb ha hba).ge
  /-
    🎉 no goals
  -/


lemma truncatedInf_union (hs : a ∈ upperClosure s) (ht : a ∈ upperClosure t) :
    truncatedInf (s ∪ t) a = truncatedInf s a ⊓ truncatedInf t a := by
  simpa only [truncatedInf_of_mem, hs, ht, upper_aux.2 (Or.inl hs), filter_union] using
    inf'_union _ _ _


lemma truncatedInf_union_left (hs : a ∈ upperClosure s) (ht : a ∉ upperClosure t) :
    truncatedInf (s ∪ t) a = truncatedInf s a := by
  /-
    α : Type u_1
    inst✝³ : SemilatticeInf α
    s t : Finset α
    a : α
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq α
    hs : Membership.mem (upperClosure ↑s) a
    ht : Not (Membership.mem (upperClosure ↑t) a)
    ⊢ Eq ((Union.union s t).truncatedInf a) (s.truncatedInf a)
  -/
  simp only [mem_upperClosure, mem_coe, exists_prop, not_exists, not_and] at ht
  simp only [truncatedInf_of_mem, hs, filter_union, filter_false_of_mem ht, union_empty,
    upper_aux.2 (Or.inl hs), ht]


lemma truncatedInf_union_right (hs : a ∉ upperClosure s) (ht : a ∈ upperClosure t) :
    truncatedInf (s ∪ t) a = truncatedInf t a := by
  /-
    α : Type u_1
    inst✝³ : SemilatticeInf α
    s t : Finset α
    a : α
    inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq α
    hs : Not (Membership.mem (upperClosure ↑s) a)
    ht : Membership.mem (upperClosure ↑t) a
    ⊢ Eq ((Union.union s t).truncatedInf a) (t.truncatedInf a)
  -/
  rw [union_comm, truncatedInf_union_left ht hs]
  /-
    🎉 no goals
  -/


lemma truncatedInf_union_of_not_mem (hs : a ∉ upperClosure s) (ht : a ∉ upperClosure t) :
    truncatedInf (s ∪ t) a = ⊥ :=
                                /-
                                  α : Type u_1
                                  inst✝³ : SemilatticeInf α
                                  s t : Finset α
                                  a : α
                                  inst✝² : DecidableRel fun x1 x2 => LE.le x1 x2
                                  inst✝¹ : BoundedOrder α
                                  inst✝ : DecidableEq α
                                  hs : Not (Membership.mem (upperClosure ↑s) a)
                                  ht : Not (Membership.mem (upperClosure ↑t) a)
                                  ⊢ Not (Membership.mem (upperClosure ↑(Union.union s t)) a)
                                -/
  truncatedInf_of_not_mem <| by rw [coe_union, upperClosure_union]; exact fun h ↦ h.elim hs ht
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


private lemma infs_aux : a ∈ lowerClosure ↑(s ⊼ t) ↔ a ∈ lowerClosure s ∧ a ∈ lowerClosure t := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Iff (Membership.mem (lowerClosure ↑(HasInfs.infs s t)) a) (And (Membership.m …
  -/
  rw [coe_infs, lowerClosure_infs, LowerSet.mem_inf_iff]
  /-
    🎉 no goals
  -/


private lemma sups_aux : a ∈ upperClosure ↑(s ⊻ t) ↔ a ∈ upperClosure s ∧ a ∈ upperClosure t := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Iff (Membership.mem (upperClosure ↑(HasSups.sups s t)) a) (And (Membership.m …
  -/
  rw [coe_sups, upperClosure_sups, UpperSet.mem_sup_iff]
  /-
    🎉 no goals
  -/


lemma truncatedSup_infs (hs : a ∈ lowerClosure s) (ht : a ∈ lowerClosure t) :
    truncatedSup (s ⊼ t) a = truncatedSup s a ⊓ truncatedSup t a := by
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (lowerClosure ↑s) a
    ht : Membership.mem (lowerClosure ↑t) a
    ⊢ Eq ((HasInfs.infs s t).truncatedSup a) (Min.min (s.truncatedSup a) (t.trunca …
  -/
  simp only [truncatedSup_of_mem, hs, ht, infs_aux.2 ⟨hs, ht⟩, sup'_inf_sup', filter_infs_le]
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (lowerClosure ↑s) a
    ht : Membership.mem (lowerClosure ↑t) a
    ⊢ Eq ((HasInfs.infs (Finset.filter (fun b => LE.le a b) s) (Finset.filter (fun …
  -/
  simp_rw [← image_inf_product]
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (lowerClosure ↑s) a
    ht : Membership.mem (lowerClosure ↑t) a
    ⊢ Eq ((Finset.image (Function.uncurry fun x1 x2 => Min.min x1 x2) (SProd.sprod …
  -/
  rw [sup'_image]
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (lowerClosure ↑s) a
    ht : Membership.mem (lowerClosure ↑t) a
    ⊢ Eq ((SProd.sprod (Finset.filter (fun b => LE.le a b) s) (Finset.filter (fun  …
  -/
  simp [Function.uncurry_def]
  /-
    🎉 no goals
  -/


lemma truncatedInf_sups (hs : a ∈ upperClosure s) (ht : a ∈ upperClosure t) :
    truncatedInf (s ⊻ t) a = truncatedInf s a ⊔ truncatedInf t a := by
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (upperClosure ↑s) a
    ht : Membership.mem (upperClosure ↑t) a
    ⊢ Eq ((HasSups.sups s t).truncatedInf a) (Max.max (s.truncatedInf a) (t.trunca …
  -/
  simp only [truncatedInf_of_mem, hs, ht, sups_aux.2 ⟨hs, ht⟩, inf'_sup_inf', filter_sups_le]
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (upperClosure ↑s) a
    ht : Membership.mem (upperClosure ↑t) a
    ⊢ Eq ((HasSups.sups (Finset.filter (fun b => LE.le b a) s) (Finset.filter (fun …
  -/
  simp_rw [← image_sup_product]
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (upperClosure ↑s) a
    ht : Membership.mem (upperClosure ↑t) a
    ⊢ Eq ((Finset.image (Function.uncurry fun x1 x2 => Max.max x1 x2) (SProd.sprod …
  -/
  rw [inf'_image]
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : DecidableEq α
    s t : Finset α
    a : α
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : BoundedOrder α
    hs : Membership.mem (upperClosure ↑s) a
    ht : Membership.mem (upperClosure ↑t) a
    ⊢ Eq ((SProd.sprod (Finset.filter (fun b => LE.le b a) s) (Finset.filter (fun  …
  -/
  simp [Function.uncurry_def]
  /-
    🎉 no goals
  -/


lemma truncatedSup_infs_of_not_mem (ha : a ∉ lowerClosure s ⊓ lowerClosure t) :
    truncatedSup (s ⊼ t) a = ⊤ :=
                                /-
                                  α : Type u_1
                                  inst✝³ : DistribLattice α
                                  inst✝² : DecidableEq α
                                  s t : Finset α
                                  a : α
                                  inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                  inst✝ : BoundedOrder α
                                  ha : Not (Membership.mem (Min.min (lowerClosure ↑s) (lowerClosure ↑t)) a)
                                  ⊢ Not (Membership.mem (lowerClosure ↑(HasInfs.infs s t)) a)
                                -/
  truncatedSup_of_not_mem <| by rwa [coe_infs, lowerClosure_infs]
                                /-
                                  🎉 no goals
                                -/


lemma truncatedInf_sups_of_not_mem (ha : a ∉ upperClosure s ⊔ upperClosure t) :
    truncatedInf (s ⊻ t) a = ⊥ :=
                                /-
                                  α : Type u_1
                                  inst✝³ : DistribLattice α
                                  inst✝² : DecidableEq α
                                  s t : Finset α
                                  a : α
                                  inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                  inst✝ : BoundedOrder α
                                  ha : Not (Membership.mem (Max.max (upperClosure ↑s) (upperClosure ↑t)) a)
                                  ⊢ Not (Membership.mem (upperClosure ↑(HasSups.sups s t)) a)
                                -/
  truncatedInf_of_not_mem <| by rwa [coe_sups, upperClosure_sups]
                                /-
                                  🎉 no goals
                                -/


@[simp] lemma compl_truncatedSup (s : Finset α) (a : α) :
    (truncatedSup s a)ᶜ = truncatedInf sᶜˢ aᶜ := map_truncatedSup (OrderIso.compl α) _ _


@[simp] lemma compl_truncatedInf (s : Finset α) (a : α) :
    (truncatedInf s a)ᶜ = truncatedSup sᶜˢ aᶜ := map_truncatedInf (OrderIso.compl α) _ _


lemma card_truncatedSup_union_add_card_truncatedSup_infs (𝒜 ℬ : Finset (Finset α)) (s : Finset α) :
    #(truncatedSup (𝒜 ∪ ℬ) s) + #(truncatedSup (𝒜 ⊼ ℬ) s) =
      #(truncatedSup 𝒜 s) + #(truncatedSup ℬ s) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    ⊢ Eq (HAdd.hAdd ((Union.union 𝒜 ℬ).truncatedSup s).card ((HasInfs.infs 𝒜 ℬ).tr …
  -/
  by_cases h𝒜 : s ∈ lowerClosure (𝒜 : Set <| Finset α) <;>
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      h𝒜 : Membership.mem (lowerClosure ↑𝒜) s
      ⊢ Eq (HAdd.hAdd ((Union.union 𝒜 ℬ).truncatedSup s).card ((HasInfs.infs 𝒜 ℬ).tr …
    -/
    by_cases hℬ : s ∈ lowerClosure (ℬ : Set <| Finset α)
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      h𝒜 : Membership.mem (lowerClosure ↑𝒜) s
      hℬ : Membership.mem (lowerClosure ↑ℬ) s
      ⊢ Eq (HAdd.hAdd ((Union.union 𝒜 ℬ).truncatedSup s).card ((HasInfs.infs 𝒜 ℬ).tr …
    -/
  · rw [truncatedSup_union h𝒜 hℬ, truncatedSup_infs h𝒜 hℬ]
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      h𝒜 : Membership.mem (lowerClosure ↑𝒜) s
      hℬ : Membership.mem (lowerClosure ↑ℬ) s
      ⊢ Eq (HAdd.hAdd (Max.max (𝒜.truncatedSup s) (ℬ.truncatedSup s)).card (Min.min  …
    -/
    exact card_union_add_card_inter _ _
    /-
      🎉 no goals
    -/
  · rw [truncatedSup_union_left h𝒜 hℬ, truncatedSup_of_not_mem hℬ,
      truncatedSup_infs_of_not_mem fun h ↦ hℬ h.2]
  · rw [truncatedSup_union_right h𝒜 hℬ, truncatedSup_of_not_mem h𝒜,
      truncatedSup_infs_of_not_mem fun h ↦ h𝒜 h.1, add_comm]
  · rw [truncatedSup_of_not_mem h𝒜, truncatedSup_of_not_mem hℬ,
      truncatedSup_union_of_not_mem h𝒜 hℬ, truncatedSup_infs_of_not_mem fun h ↦ h𝒜 h.1]


lemma card_truncatedInf_union_add_card_truncatedInf_sups (𝒜 ℬ : Finset (Finset α)) (s : Finset α) :
    #(truncatedInf (𝒜 ∪ ℬ) s) + #(truncatedInf (𝒜 ⊻ ℬ) s) =
      #(truncatedInf 𝒜 s) + #(truncatedInf ℬ s) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    ⊢ Eq (HAdd.hAdd ((Union.union 𝒜 ℬ).truncatedInf s).card ((HasSups.sups 𝒜 ℬ).tr …
  -/
  by_cases h𝒜 : s ∈ upperClosure (𝒜 : Set <| Finset α) <;>
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      h𝒜 : Membership.mem (upperClosure ↑𝒜) s
      ⊢ Eq (HAdd.hAdd ((Union.union 𝒜 ℬ).truncatedInf s).card ((HasSups.sups 𝒜 ℬ).tr …
    -/
    by_cases hℬ : s ∈ upperClosure (ℬ : Set <| Finset α)
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      h𝒜 : Membership.mem (upperClosure ↑𝒜) s
      hℬ : Membership.mem (upperClosure ↑ℬ) s
      ⊢ Eq (HAdd.hAdd ((Union.union 𝒜 ℬ).truncatedInf s).card ((HasSups.sups 𝒜 ℬ).tr …
    -/
  · rw [truncatedInf_union h𝒜 hℬ, truncatedInf_sups h𝒜 hℬ]
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      h𝒜 : Membership.mem (upperClosure ↑𝒜) s
      hℬ : Membership.mem (upperClosure ↑ℬ) s
      ⊢ Eq (HAdd.hAdd (Min.min (𝒜.truncatedInf s) (ℬ.truncatedInf s)).card (Max.max  …
    -/
    exact card_inter_add_card_union _ _
    /-
      🎉 no goals
    -/
  · rw [truncatedInf_union_left h𝒜 hℬ, truncatedInf_of_not_mem hℬ,
      truncatedInf_sups_of_not_mem fun h ↦ hℬ h.2]
  · rw [truncatedInf_union_right h𝒜 hℬ, truncatedInf_of_not_mem h𝒜,
      truncatedInf_sups_of_not_mem fun h ↦ h𝒜 h.1, add_comm]
  · rw [truncatedInf_of_not_mem h𝒜, truncatedInf_of_not_mem hℬ,
      truncatedInf_union_of_not_mem h𝒜 hℬ, truncatedInf_sups_of_not_mem fun h ↦ h𝒜 h.1]


/-- Weighted sum of the size of the truncated infima of a set family. Relevant to the
Ahlswede-Zhang identity. -/
def infSum (𝒜 : Finset (Finset α)) : ℚ :=
  ∑ s, #(truncatedInf 𝒜 s) / (#s * (card α).choose #s)


/-- Weighted sum of the size of the truncated suprema of a set family. Relevant to the
Ahlswede-Zhang identity. -/
def supSum (𝒜 : Finset (Finset α)) : ℚ :=
  ∑ s, #(truncatedSup 𝒜 s) / ((card α - #s) * (card α).choose #s)


lemma supSum_union_add_supSum_infs (𝒜 ℬ : Finset (Finset α)) :
    supSum (𝒜 ∪ ℬ) + supSum (𝒜 ⊼ ℬ) = supSum 𝒜 + supSum ℬ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (AhlswedeZhang.supSum (Union.union 𝒜 ℬ)) (AhlswedeZhang.supSum …
  -/
  unfold supSum
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun s => HDiv.hDiv (↑((Union.union 𝒜 ℬ).trunc …
  -/
  rw [← sum_add_distrib, ← sum_add_distrib, sum_congr rfl fun s _ ↦ _]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ ∀ (s : Finset α), Membership.mem Finset.univ s → Eq (HAdd.hAdd (HDiv.hDiv (↑ …
  -/
  simp_rw [div_add_div_same, ← Nat.cast_add, card_truncatedSup_union_add_card_truncatedSup_infs]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ ∀ (s : Finset α), Membership.mem Finset.univ s → True
  -/
  simp
  /-
    🎉 no goals
  -/


lemma infSum_union_add_infSum_sups (𝒜 ℬ : Finset (Finset α)) :
    infSum (𝒜 ∪ ℬ) + infSum (𝒜 ⊻ ℬ) = infSum 𝒜 + infSum ℬ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (AhlswedeZhang.infSum (Union.union 𝒜 ℬ)) (AhlswedeZhang.infSum …
  -/
  unfold infSum
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun s => HDiv.hDiv (↑((Union.union 𝒜 ℬ).trunc …
  -/
  rw [← sum_add_distrib, ← sum_add_distrib, sum_congr rfl fun s _ ↦ _]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ ∀ (s : Finset α), Membership.mem Finset.univ s → Eq (HAdd.hAdd (HDiv.hDiv (↑ …
  -/
  simp_rw [div_add_div_same, ← Nat.cast_add, card_truncatedInf_union_add_card_truncatedInf_sups]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 ℬ : Finset (Finset α)
    ⊢ ∀ (s : Finset α), Membership.mem Finset.univ s → True
  -/
  simp
  /-
    🎉 no goals
  -/


lemma IsAntichain.le_infSum (h𝒜 : IsAntichain (· ⊆ ·) (𝒜 : Set (Finset α))) (h𝒜₀ : ∅ ∉ 𝒜) :
    ∑ s ∈ 𝒜, ((card α).choose #s : ℚ)⁻¹ ≤ infSum 𝒜 := by
  calc
    _ = ∑ s ∈ 𝒜, #(truncatedInf 𝒜 s) / (#s * (card α).choose #s : ℚ) := ?_
    _ ≤ _ := sum_le_univ_sum_of_nonneg fun s ↦ by positivity
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
    h𝒜₀ : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    ⊢ Eq (𝒜.sum fun s => Inv.inv ↑((Fintype.card α).choose s.card)) (𝒜.sum fun s = …
  -/
  refine sum_congr rfl fun s hs ↦ ?_
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
    h𝒜₀ : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    s : Finset α
    hs : Membership.mem 𝒜 s
    ⊢ Eq (Inv.inv ↑((Fintype.card α).choose s.card)) (HDiv.hDiv (↑(𝒜.truncatedInf  …
  -/
  rw [truncatedInf_of_isAntichain h𝒜 hs, div_mul_cancel_left₀]
  /-
    case ha
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
    h𝒜₀ : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    s : Finset α
    hs : Membership.mem 𝒜 s
    ⊢ Ne (↑s.card) 0
  -/
  have := (nonempty_iff_ne_empty.2 <| ne_of_mem_of_not_mem hs h𝒜₀).card_pos
  /-
    case ha
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
    h𝒜₀ : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    s : Finset α
    hs : Membership.mem 𝒜 s
    this : LT.lt 0 s.card
    ⊢ Ne (↑s.card) 0
  -/
  positivity
  /-
    🎉 no goals
  -/


@[simp] lemma supSum_singleton (hs : s ≠ univ) :
    supSum ({s} : Finset (Finset α)) = card α * ∑ k ∈ range (card α), (k : ℚ)⁻¹ := by
  have : ∀ t : Finset α,
    (card α - #(truncatedSup {s} t) : ℚ) / ((card α - #t) * (card α).choose #t) =
    if t ⊆ s then (card α - #s : ℚ) / ((card α - #t) * (card α).choose #t) else 0 := by
    rintro t
    simp_rw [truncatedSup_singleton, le_iff_subset]
    split_ifs <;> simp [card_univ]
  simp_rw [← sub_eq_of_eq_add (Fintype.sum_div_mul_card_choose_card α), eq_sub_iff_add_eq,
    ← eq_sub_iff_add_eq', supSum, ← sum_sub_distrib, ← sub_div]
  rw [sum_congr rfl fun t _ ↦ this t, sum_ite, sum_const_zero, add_zero, filter_subset_univ,
    sum_powerset, ← binomial_sum_eq ((card_lt_iff_ne_univ _).2 hs), eq_comm]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    s : Finset α
    inst✝ : Nonempty α
    hs : Ne s Finset.univ
    this : ∀ (t : Finset α), Eq (HDiv.hDiv (HSub.hSub ↑(Fintype.card α) ↑((Singlet …
    ⊢ Eq ((Finset.range (HAdd.hAdd s.card 1)).sum fun j => (Finset.powersetCard j  …
  -/
  refine sum_congr rfl fun n _ ↦ ?_
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    s : Finset α
    inst✝ : Nonempty α
    hs : Ne s Finset.univ
    this : ∀ (t : Finset α), Eq (HDiv.hDiv (HSub.hSub ↑(Fintype.card α) ↑((Singlet …
    n : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) n
    ⊢ Eq ((Finset.powersetCard n s).sum fun t => HDiv.hDiv (HSub.hSub ↑(Fintype.ca …
  -/
  rw [mul_div_assoc, ← nsmul_eq_mul]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    s : Finset α
    inst✝ : Nonempty α
    hs : Ne s Finset.univ
    this : ∀ (t : Finset α), Eq (HDiv.hDiv (HSub.hSub ↑(Fintype.card α) ↑((Singlet …
    n : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd s.card 1)) n
    ⊢ Eq ((Finset.powersetCard n s).sum fun t => HDiv.hDiv (HSub.hSub ↑(Fintype.ca …
  -/
  exact sum_powersetCard n s fun m ↦ (card α - #s : ℚ) / ((card α - m) * (card α).choose m)
  /-
    🎉 no goals
  -/


/-- The **Ahlswede-Zhang Identity**. -/
lemma infSum_compls_add_supSum (𝒜 : Finset (Finset α)) :
    infSum 𝒜ᶜˢ + supSum 𝒜 = card α * ∑ k ∈ range (card α), (k : ℚ)⁻¹ + 1 := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    𝒜 : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (AhlswedeZhang.infSum 𝒜.compls) (AhlswedeZhang.supSum 𝒜)) (HAd …
  -/
  unfold infSum supSum
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    𝒜 : Finset (Finset α)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun s => HDiv.hDiv (↑(𝒜.compls.truncatedInf s …
  -/
  rw [← @map_univ_of_surjective (Finset α) _ _ _ ⟨compl, compl_injective⟩ compl_surjective, sum_map]
  simp only [Function.Embedding.coeFn_mk, univ_map_embedding, ← compl_truncatedSup,
    ← sum_add_distrib, card_compl, cast_sub (card_le_univ _), choose_symm (card_le_univ _),
    div_add_div_same, sub_add_cancel, Fintype.sum_div_mul_card_choose_card]


lemma supSum_of_not_univ_mem (h𝒜₁ : 𝒜.Nonempty) (h𝒜₂ : univ ∉ 𝒜) :
    supSum 𝒜 = card α * ∑ k ∈ range (card α), (k : ℚ)⁻¹ := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Nonempty α
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  set m := 𝒜.card with hm
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Nonempty α
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    m : Nat := 𝒜.card
    hm : Eq m 𝒜.card
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  clear_value m
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Nonempty α
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    m : Nat
    hm : Eq m 𝒜.card
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  induction' m using Nat.strong_induction_on with m ih generalizing 𝒜
  /-
    case h
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    m : Nat
    ih : ∀ (m_1 : Nat), LT.lt m_1 m → ∀ {𝒜 : Finset (Finset α)}, 𝒜.Nonempty → Not  …
    𝒜 : Finset (Finset α)
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    hm : Eq m 𝒜.card
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  replace ih := fun 𝒜 h𝒜 h𝒜₁ h𝒜₂ ↦ @ih _ h𝒜 𝒜 h𝒜₁ h𝒜₂ rfl
  /-
    case h
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    m : Nat
    𝒜 : Finset (Finset α)
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    hm : Eq m 𝒜.card
    ih : ∀ (𝒜 : Finset (Finset α)), LT.lt 𝒜.card m → 𝒜.Nonempty → Not (Membership. …
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  obtain ⟨a, rfl⟩ | h𝒜₃ := h𝒜₁.exists_eq_singleton_or_nontrivial
    /-
      case h.inl.intro
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      m : Nat
      ih : ∀ (𝒜 : Finset (Finset α)), LT.lt 𝒜.card m → 𝒜.Nonempty → Not (Membership. …
      a : Finset α
      h𝒜₁ : (Singleton.singleton a).Nonempty
      h𝒜₂ : Not (Membership.mem (Singleton.singleton a) Finset.univ)
      hm : Eq m (Singleton.singleton a).card
      ⊢ Eq (AhlswedeZhang.supSum (Singleton.singleton a)) (HMul.hMul (↑(Fintype.card …
    -/
  · refine supSum_singleton ?_
    /-
      case h.inl.intro
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      m : Nat
      ih : ∀ (𝒜 : Finset (Finset α)), LT.lt 𝒜.card m → 𝒜.Nonempty → Not (Membership. …
      a : Finset α
      h𝒜₁ : (Singleton.singleton a).Nonempty
      h𝒜₂ : Not (Membership.mem (Singleton.singleton a) Finset.univ)
      hm : Eq m (Singleton.singleton a).card
      ⊢ Ne a Finset.univ
    -/
    simpa [eq_comm] using h𝒜₂
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    m : Nat
    𝒜 : Finset (Finset α)
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    hm : Eq m 𝒜.card
    ih : ∀ (𝒜 : Finset (Finset α)), LT.lt 𝒜.card m → 𝒜.Nonempty → Not (Membership. …
    h𝒜₃ : 𝒜.Nontrivial
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  cases m
    /-
      case h.inr.zero
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      𝒜 : Finset (Finset α)
      h𝒜₁ : 𝒜.Nonempty
      h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
      h𝒜₃ : 𝒜.Nontrivial
      hm : Eq 0 𝒜.card
      ih : ∀ (𝒜 : Finset (Finset α)), LT.lt 𝒜.card 0 → 𝒜.Nonempty → Not (Membership. …
      ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
    -/
  · cases h𝒜₁.card_pos.ne hm
    /-
      🎉 no goals
    -/
  /-
    case h.inr.succ
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    𝒜 : Finset (Finset α)
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₂ : Not (Membership.mem 𝒜 Finset.univ)
    h𝒜₃ : 𝒜.Nontrivial
    n✝ : Nat
    hm : Eq (HAdd.hAdd n✝ 1) 𝒜.card
    ih : ∀ (𝒜 : Finset (Finset α)), LT.lt 𝒜.card (HAdd.hAdd n✝ 1) → 𝒜.Nonempty → N …
    ⊢ Eq (AhlswedeZhang.supSum 𝒜) (HMul.hMul (↑(Fintype.card α)) ((Finset.range (F …
  -/
  obtain ⟨s, 𝒜, hs, rfl, rfl⟩ := card_eq_succ.1 hm.symm
  /-
    case h.inr.succ.intro.intro.intro.intro
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    s : Finset α
    𝒜 : Finset (Finset α)
    hs : Not (Membership.mem 𝒜 s)
    h𝒜₁ : (Insert.insert s 𝒜).Nonempty
    h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
    h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
    ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
    hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
    ⊢ Eq (AhlswedeZhang.supSum (Insert.insert s 𝒜)) (HMul.hMul (↑(Fintype.card α)) …
  -/
  have h𝒜 : 𝒜.Nonempty := nonempty_iff_ne_empty.2 (by rintro rfl; simp at h𝒜₃)
  rw [insert_eq, eq_sub_of_add_eq (supSum_union_add_supSum_infs _ _), singleton_infs,
    supSum_singleton (ne_of_mem_of_not_mem (mem_insert_self _ _) h𝒜₂), ih, ih, add_sub_cancel_right]
    /-
      case h.inr.succ.intro.intro.intro.intro.h𝒜
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset α
      𝒜 : Finset (Finset α)
      hs : Not (Membership.mem 𝒜 s)
      h𝒜₁ : (Insert.insert s 𝒜).Nonempty
      h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
      h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
      ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
      hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
      h𝒜 : 𝒜.Nonempty
      ⊢ LT.lt (Finset.image (fun x => Min.min s x) 𝒜).card (HAdd.hAdd 𝒜.card 1)
    -/
  · exact card_image_le.trans_lt (lt_add_one _)
    /-
      🎉 no goals
    -/
    /-
      case h.inr.succ.intro.intro.intro.intro.h𝒜₁
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset α
      𝒜 : Finset (Finset α)
      hs : Not (Membership.mem 𝒜 s)
      h𝒜₁ : (Insert.insert s 𝒜).Nonempty
      h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
      h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
      ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
      hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
      h𝒜 : 𝒜.Nonempty
      ⊢ (Finset.image (fun x => Min.min s x) 𝒜).Nonempty
    -/
  · exact h𝒜.image _
    /-
      🎉 no goals
    -/
    /-
      case h.inr.succ.intro.intro.intro.intro.h𝒜₂
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset α
      𝒜 : Finset (Finset α)
      hs : Not (Membership.mem 𝒜 s)
      h𝒜₁ : (Insert.insert s 𝒜).Nonempty
      h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
      h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
      ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
      hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
      h𝒜 : 𝒜.Nonempty
      ⊢ Not (Membership.mem (Finset.image (fun x => Min.min s x) 𝒜) Finset.univ)
    -/
  · simpa using fun _ ↦ ne_of_mem_of_not_mem (mem_insert_self _ _) h𝒜₂
    /-
      🎉 no goals
    -/
    /-
      case h.inr.succ.intro.intro.intro.intro.h𝒜
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset α
      𝒜 : Finset (Finset α)
      hs : Not (Membership.mem 𝒜 s)
      h𝒜₁ : (Insert.insert s 𝒜).Nonempty
      h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
      h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
      ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
      hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
      h𝒜 : 𝒜.Nonempty
      ⊢ LT.lt 𝒜.card (HAdd.hAdd 𝒜.card 1)
    -/
  · exact lt_add_one _
    /-
      🎉 no goals
    -/
    /-
      case h.inr.succ.intro.intro.intro.intro.h𝒜₁
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset α
      𝒜 : Finset (Finset α)
      hs : Not (Membership.mem 𝒜 s)
      h𝒜₁ : (Insert.insert s 𝒜).Nonempty
      h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
      h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
      ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
      hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
      h𝒜 : 𝒜.Nonempty
      ⊢ 𝒜.Nonempty
    -/
  · exact h𝒜
    /-
      🎉 no goals
    -/
    /-
      case h.inr.succ.intro.intro.intro.intro.h𝒜₂
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset α
      𝒜 : Finset (Finset α)
      hs : Not (Membership.mem 𝒜 s)
      h𝒜₁ : (Insert.insert s 𝒜).Nonempty
      h𝒜₂ : Not (Membership.mem (Insert.insert s 𝒜) Finset.univ)
      h𝒜₃ : (Insert.insert s 𝒜).Nontrivial
      ih : ∀ (𝒜_1 : Finset (Finset α)), LT.lt 𝒜_1.card (HAdd.hAdd 𝒜.card 1) → 𝒜_1.No …
      hm : Eq (HAdd.hAdd 𝒜.card 1) (Insert.insert s 𝒜).card
      h𝒜 : 𝒜.Nonempty
      ⊢ Not (Membership.mem 𝒜 Finset.univ)
    -/
  · exact fun h ↦ h𝒜₂ (mem_insert_of_mem h)
    /-
      🎉 no goals
    -/


/-- The **Ahlswede-Zhang Identity**. -/
lemma infSum_eq_one (h𝒜₁ : 𝒜.Nonempty) (h𝒜₀ : ∅ ∉ 𝒜) : infSum 𝒜 = 1 := by
  rw [← compls_compls 𝒜, eq_sub_of_add_eq (infSum_compls_add_supSum _),
    supSum_of_not_univ_mem h𝒜₁.compls, add_sub_cancel_left]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    𝒜 : Finset (Finset α)
    inst✝ : Nonempty α
    h𝒜₁ : 𝒜.Nonempty
    h𝒜₀ : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    ⊢ Not (Membership.mem 𝒜.compls Finset.univ)
  -/
  simpa
  /-
    🎉 no goals
  -/


