local notation "G " n:80 => ∑ i ∈ range n, g i


/-- **Summation by parts**, also known as **Abel's lemma** or an **Abel transformation** -/
theorem sum_Ico_by_parts (hmn : m < n) :
    ∑ i ∈ Ico m n, f i • g i =
      f (n - 1) • G n - f m • G m - ∑ i ∈ Ico m (n - 1), (f (i + 1) - f i) • G (i + 1) := by
  have h₁ : (∑ i ∈ Ico (m + 1) n, f i • G i) = ∑ i ∈ Ico m (n - 1), f (i + 1) • G (i + 1) := by
    rw [← Nat.sub_add_cancel (Nat.one_le_of_lt hmn), ← sum_Ico_add']
    simp only [tsub_le_iff_right, add_le_iff_nonpos_left, nonpos_iff_eq_zero,
      tsub_eq_zero_iff_le, add_tsub_cancel_right]
  have h₂ :
    (∑ i ∈ Ico (m + 1) n, f i • G (i + 1)) =
      (∑ i ∈ Ico m (n - 1), f i • G (i + 1)) + f (n - 1) • G n - f m • G (m + 1) := by
    rw [← sum_Ico_sub_bot _ hmn, ← sum_Ico_succ_sub_top _ (Nat.le_sub_one_of_lt hmn),
      Nat.sub_add_cancel (pos_of_gt hmn), sub_add_cancel]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    m n : Nat
    hmn : LT.lt m n
    h₁ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    h₂ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    ⊢ Eq ((Finset.Ico m n).sum fun i => HSMul.hSMul (f i) (g i)) (HSub.hSub (HSub. …
  -/
  rw [sum_eq_sum_Ico_succ_bot hmn]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    m n : Nat
    hmn : LT.lt m n
    h₁ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    h₂ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (f m) (g m)) ((Finset.Ico (HAdd.hAdd m 1) n).sum  …
  -/
  conv in (occs := 3) (f _ • g _) => rw [← sum_range_succ_sub_sum g]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    m n : Nat
    hmn : LT.lt m n
    h₁ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    h₂ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (f m) (g m)) ((Finset.Ico (HAdd.hAdd m 1) n).sum  …
  -/
  simp_rw [smul_sub, sum_sub_distrib, h₂, h₁]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    m n : Nat
    hmn : LT.lt m n
    h₁ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    h₂ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (f m) (g m)) (HSub.hSub (HSub.hSub (HAdd.hAdd ((F …
  -/
  conv_lhs => congr; rfl; rw [← add_sub, add_comm, ← add_sub, ← sum_sub_distrib]
  have : ∀ i, f i • G (i + 1) - f (i + 1) • G (i + 1) = -((f (i + 1) - f i) • G (i + 1)) := by
    intro i
    rw [sub_smul]
    abel
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    m n : Nat
    hmn : LT.lt m n
    h₁ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    h₂ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    this : ∀ (i : Nat), Eq (HSub.hSub (HSMul.hSMul (f i) ((Finset.range (HAdd.hAdd …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (f m) (g m)) (HAdd.hAdd (HSub.hSub (HSMul.hSMul ( …
  -/
  simp_rw [this, sum_neg_distrib, sum_range_succ, smul_add]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    m n : Nat
    hmn : LT.lt m n
    h₁ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    h₂ : Eq ((Finset.Ico (HAdd.hAdd m 1) n).sum fun i => HSMul.hSMul (f i) ((Finse …
    this : ∀ (i : Nat), Eq (HSub.hSub (HSMul.hSMul (f i) ((Finset.range (HAdd.hAdd …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (f m) (g m)) (HAdd.hAdd (HSub.hSub (HSMul.hSMul ( …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem sum_Ioc_by_parts (hmn : m < n) :
    ∑ i ∈ Ioc m n, f i • g i =
      f n • G (n + 1) - f (m + 1) • G (m + 1)
        - ∑ i ∈ Ioc m (n - 1), (f (i + 1) - f i) • G (i + 1) := by
  simpa only [← Nat.Ico_succ_succ, Nat.succ_eq_add_one, Nat.sub_add_cancel (Nat.one_le_of_lt hmn),
    add_tsub_cancel_right] using sum_Ico_by_parts f g (Nat.succ_lt_succ hmn)


/-- **Summation by parts** for ranges -/
theorem sum_range_by_parts :
    ∑ i ∈ range n, f i • g i =
      f (n - 1) • G n - ∑ i ∈ range (n - 1), (f (i + 1) - f i) • G (i + 1) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → R
    g : Nat → M
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HSMul.hSMul (f i) (g i)) (HSub.hSub (HSMul …
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Nat → R
      g : Nat → M
      n : Nat
      hn : Eq n 0
      ⊢ Eq ((Finset.range n).sum fun i => HSMul.hSMul (f i) (g i)) (HSub.hSub (HSMul …
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  · rw [range_eq_Ico, sum_Ico_by_parts f g (Nat.pos_of_ne_zero hn), sum_range_zero, smul_zero,
      sub_zero, range_eq_Ico]


