/-- A successor pre-limit is a value that doesn't cover any other.

It's so named because in a successor order, a successor pre-limit can't be the successor of anything
smaller.

Use `IsSuccLimit` if you want to exclude the case of a minimal element. -/
def IsSuccPrelimit (a : α) : Prop :=
  ∀ b, ¬b ⋖ a


theorem not_isSuccPrelimit_iff_exists_covBy (a : α) : ¬IsSuccPrelimit a ↔ ∃ b, b ⋖ a := by
  /-
    α : Type u_1
    inst✝ : LT α
    a : α
    ⊢ Iff (Not (Order.IsSuccPrelimit a)) (Exists fun b => CovBy b a)
  -/
  simp [IsSuccPrelimit]
  /-
    🎉 no goals
  -/


@[deprecated not_isSuccPrelimit_iff_exists_covBy (since := "2024-09-05")]
alias not_isSuccLimit_iff_exists_covBy := not_isSuccPrelimit_iff_exists_covBy


@[simp]
theorem IsSuccPrelimit.of_dense [DenselyOrdered α] (a : α) : IsSuccPrelimit a := fun _ => not_covBy


@[deprecated (since := "2024-09-30")] alias isSuccPrelimit_of_dense := IsSuccPrelimit.of_dense

@[deprecated (since := "2024-09-05")] alias isSuccLimit_of_dense := IsSuccPrelimit.of_dense


/-- A successor limit is a value that isn't minimal and doesn't cover any other.

It's so named because in a successor order, a successor limit can't be the successor of anything
smaller.

This previously allowed the element to be minimal. This usage is now covered by `IsSuccPrelimit`. -/
def IsSuccLimit (a : α) : Prop :=
  ¬ IsMin a ∧ IsSuccPrelimit a


protected theorem IsSuccLimit.not_isMin (h : IsSuccLimit a) : ¬ IsMin a := h.1

protected theorem IsSuccLimit.isSuccPrelimit (h : IsSuccLimit a) : IsSuccPrelimit a := h.2


theorem IsSuccPrelimit.isSuccLimit_of_not_isMin (h : IsSuccPrelimit a) (ha : ¬ IsMin a) :
    IsSuccLimit a :=
  ⟨ha, h⟩


theorem IsSuccPrelimit.isSuccLimit [NoMinOrder α] (h : IsSuccPrelimit a) : IsSuccLimit a :=
  h.isSuccLimit_of_not_isMin (not_isMin a)


theorem isSuccPrelimit_iff_isSuccLimit_of_not_isMin (h : ¬ IsMin a) :
    IsSuccPrelimit a ↔ IsSuccLimit a :=
  ⟨fun ha ↦ ha.isSuccLimit_of_not_isMin h, IsSuccLimit.isSuccPrelimit⟩


theorem isSuccPrelimit_iff_isSuccLimit [NoMinOrder α] : IsSuccPrelimit a ↔ IsSuccLimit a :=
  isSuccPrelimit_iff_isSuccLimit_of_not_isMin (not_isMin a)


protected theorem _root_.IsMin.not_isSuccLimit (h : IsMin a) : ¬ IsSuccLimit a :=
  fun ha ↦ ha.not_isMin h


protected theorem _root_.IsMin.isSuccPrelimit : IsMin a → IsSuccPrelimit a := fun h _ hab =>
  not_isMin_of_lt hab.lt h


@[deprecated _root_.IsMin.isSuccPrelimit (since := "2024-09-05")]
alias _root_.IsMin.isSuccLimit := _root_.IsMin.isSuccPrelimit


theorem isSuccPrelimit_bot [OrderBot α] : IsSuccPrelimit (⊥ : α) :=
  isMin_bot.isSuccPrelimit


theorem not_isSuccLimit_bot [OrderBot α] : ¬ IsSuccLimit (⊥ : α) :=
  isMin_bot.not_isSuccLimit


theorem IsSuccLimit.ne_bot [OrderBot α] (h : IsSuccLimit a) : a ≠ ⊥ := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    h : Order.IsSuccLimit a
    ⊢ Ne a Bot.bot
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    h : Order.IsSuccLimit Bot.bot
    ⊢ False
  -/
  exact not_isSuccLimit_bot h
  /-
    🎉 no goals
  -/


@[deprecated isSuccPrelimit_bot (since := "2024-09-05")]
alias isSuccLimit_bot := isSuccPrelimit_bot


theorem not_isSuccLimit_iff : ¬ IsSuccLimit a ↔ IsMin a ∨ ¬ IsSuccPrelimit a := by
  /-
    α : Type u_1
    a : α
    inst✝ : Preorder α
    ⊢ Iff (Not (Order.IsSuccLimit a)) (Or (IsMin a) (Not (Order.IsSuccPrelimit a)))
  -/
  rw [IsSuccLimit, not_and_or, not_not]
  /-
    🎉 no goals
  -/


protected theorem IsSuccPrelimit.isMax (h : IsSuccPrelimit (succ a)) : IsMax a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    h : Order.IsSuccPrelimit (Order.succ a)
    ⊢ IsMax a
  -/
  by_contra H
  /-
    α : Type u_1
    a : α
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    h : Order.IsSuccPrelimit (Order.succ a)
    H : Not (IsMax a)
    ⊢ False
  -/
  exact h a (covBy_succ_of_not_isMax H)
  /-
    🎉 no goals
  -/


protected theorem IsSuccLimit.isMax (h : IsSuccLimit (succ a)) : IsMax a :=
  h.isSuccPrelimit.isMax


theorem not_isSuccPrelimit_succ_of_not_isMax (ha : ¬ IsMax a) : ¬ IsSuccPrelimit (succ a) :=
  mt IsSuccPrelimit.isMax ha


theorem not_isSuccLimit_succ_of_not_isMax (ha : ¬ IsMax a) : ¬ IsSuccLimit (succ a) :=
  mt IsSuccLimit.isMax ha


/-- Given `j < i` with `i` a prelimit, `IsSuccPrelimit.mid` picks an arbitrary element strictly
between `j` and `i`. -/
noncomputable def IsSuccPrelimit.mid {i j : α} (hi : IsSuccPrelimit i) (hj : j < i) :
    Ioo j i :=
  Classical.indefiniteDescription _ ((not_covBy_iff hj).mp <| hi j)


theorem IsSuccPrelimit.succ_ne (h : IsSuccPrelimit a) (b : α) : succ b ≠ a := by
  /-
    α : Type u_1
    a : α
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : NoMaxOrder α
    h : Order.IsSuccPrelimit a
    b : α
    ⊢ Ne (Order.succ b) a
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : NoMaxOrder α
    b : α
    h : Order.IsSuccPrelimit (Order.succ b)
    ⊢ False
  -/
  exact not_isMax _ h.isMax
  /-
    🎉 no goals
  -/


theorem IsSuccLimit.succ_ne (h : IsSuccLimit a) (b : α) : succ b ≠ a :=
  h.isSuccPrelimit.succ_ne b


@[simp]
theorem not_isSuccPrelimit_succ (a : α) : ¬IsSuccPrelimit (succ a) := fun h => h.succ_ne _ rfl


@[simp]
theorem not_isSuccLimit_succ (a : α) : ¬IsSuccLimit (succ a) := fun h => h.succ_ne _ rfl


theorem IsSuccPrelimit.isMin_of_noMax (h : IsSuccPrelimit a) : IsMin a := by
  /-
    α : Type u_1
    a : α
    inst✝³ : Preorder α
    inst✝² : SuccOrder α
    inst✝¹ : IsSuccArchimedean α
    inst✝ : NoMaxOrder α
    h : Order.IsSuccPrelimit a
    ⊢ IsMin a
  -/
  intro b hb
  /-
    α : Type u_1
    a : α
    inst✝³ : Preorder α
    inst✝² : SuccOrder α
    inst✝¹ : IsSuccArchimedean α
    inst✝ : NoMaxOrder α
    h : Order.IsSuccPrelimit a
    b : α
    hb : LE.le b a
    ⊢ LE.le a b
  -/
  rcases hb.exists_succ_iterate with ⟨_ | n, rfl⟩
    /-
      case intro.zero
      α : Type u_1
      inst✝³ : Preorder α
      inst✝² : SuccOrder α
      inst✝¹ : IsSuccArchimedean α
      inst✝ : NoMaxOrder α
      b : α
      h : Order.IsSuccPrelimit (Nat.iterate Order.succ 0 b)
      hb : LE.le b (Nat.iterate Order.succ 0 b)
      ⊢ LE.le (Nat.iterate Order.succ 0 b) b
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.succ
      α : Type u_1
      inst✝³ : Preorder α
      inst✝² : SuccOrder α
      inst✝¹ : IsSuccArchimedean α
      inst✝ : NoMaxOrder α
      b : α
      n : Nat
      h : Order.IsSuccPrelimit (Nat.iterate Order.succ (HAdd.hAdd n 1) b)
      hb : LE.le b (Nat.iterate Order.succ (HAdd.hAdd n 1) b)
      ⊢ LE.le (Nat.iterate Order.succ (HAdd.hAdd n 1) b) b
    -/
  · rw [iterate_succ_apply'] at h
    /-
      case intro.succ
      α : Type u_1
      inst✝³ : Preorder α
      inst✝² : SuccOrder α
      inst✝¹ : IsSuccArchimedean α
      inst✝ : NoMaxOrder α
      b : α
      n : Nat
      h : Order.IsSuccPrelimit (Order.succ (Nat.iterate Order.succ n b))
      hb : LE.le b (Nat.iterate Order.succ (HAdd.hAdd n 1) b)
      ⊢ LE.le (Nat.iterate Order.succ (HAdd.hAdd n 1) b) b
    -/
    exact (not_isSuccPrelimit_succ _ h).elim
    /-
      🎉 no goals
    -/


@[deprecated IsSuccPrelimit.isMin_of_noMax (since := "2024-09-05")]
alias IsSuccLimit.isMin_of_noMax := IsSuccPrelimit.isMin_of_noMax


@[simp]
theorem isSuccPrelimit_iff_of_noMax : IsSuccPrelimit a ↔ IsMin a :=
  ⟨IsSuccPrelimit.isMin_of_noMax, IsMin.isSuccPrelimit⟩


@[deprecated isSuccPrelimit_iff_of_noMax (since := "2024-09-05")]
alias isSuccLimit_iff_of_noMax := isSuccPrelimit_iff_of_noMax


@[simp]
theorem not_isSuccLimit_of_noMax : ¬ IsSuccLimit a :=
  fun h ↦ h.not_isMin h.isSuccPrelimit.isMin_of_noMax


                                                                              /-
                                                                                α : Type u_1
                                                                                a : α
                                                                                inst✝⁴ : Preorder α
                                                                                inst✝³ : SuccOrder α
                                                                                inst✝² : IsSuccArchimedean α
                                                                                inst✝¹ : NoMaxOrder α
                                                                                inst✝ : NoMinOrder α
                                                                                ⊢ Not (Order.IsSuccPrelimit a)
                                                                              -/
theorem not_isSuccPrelimit_of_noMax [NoMinOrder α] : ¬ IsSuccPrelimit a := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem isSuccLimit_iff [OrderBot α] : IsSuccLimit a ↔ a ≠ ⊥ ∧ IsSuccPrelimit a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    ⊢ Iff (Order.IsSuccLimit a) (And (Ne a Bot.bot) (Order.IsSuccPrelimit a))
  -/
  rw [IsSuccLimit, isMin_iff_eq_bot]
  /-
    🎉 no goals
  -/


theorem IsSuccLimit.bot_lt [OrderBot α] (h : IsSuccLimit a) : ⊥ < a :=
  h.ne_bot.bot_lt


theorem isSuccPrelimit_of_succ_ne (h : ∀ b, succ b ≠ a) : IsSuccPrelimit a := fun b hba =>
  h b (CovBy.succ_eq hba)


@[deprecated isSuccPrelimit_of_succ_ne (since := "2024-09-05")]
alias isSuccLimit_of_succ_ne := isSuccPrelimit_of_succ_ne


theorem not_isSuccPrelimit_iff : ¬ IsSuccPrelimit a ↔ ∃ b, ¬ IsMax b ∧ succ b = a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    ⊢ Iff (Not (Order.IsSuccPrelimit a)) (Exists fun b => And (Not (IsMax b)) (Eq  …
  -/
  rw [not_isSuccPrelimit_iff_exists_covBy]
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    ⊢ Iff (Exists fun b => CovBy b a) (Exists fun b => And (Not (IsMax b)) (Eq (Or …
  -/
  refine exists_congr fun b => ⟨fun hba => ⟨hba.lt.not_isMax, (CovBy.succ_eq hba)⟩, ?_⟩
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    b : α
    ⊢ And (Not (IsMax b)) (Eq (Order.succ b) a) → CovBy b a
  -/
  rintro ⟨h, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    b : α
    h : Not (IsMax b)
    ⊢ CovBy b (Order.succ b)
  -/
  exact covBy_succ_of_not_isMax h
  /-
    🎉 no goals
  -/


/-- See `not_isSuccPrelimit_iff` for a version that states that `a` is a successor of a value other
than itself. -/
theorem mem_range_succ_of_not_isSuccPrelimit (h : ¬ IsSuccPrelimit a) :
    a ∈ range (succ : α → α) := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    h : Not (Order.IsSuccPrelimit a)
    ⊢ Membership.mem (Set.range Order.succ) a
  -/
  obtain ⟨b, hb⟩ := not_isSuccPrelimit_iff.1 h
  /-
    case intro
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    h : Not (Order.IsSuccPrelimit a)
    b : α
    hb : And (Not (IsMax b)) (Eq (Order.succ b) a)
    ⊢ Membership.mem (Set.range Order.succ) a
  -/
  exact ⟨b, hb.2⟩
  /-
    🎉 no goals
  -/


@[deprecated mem_range_succ_of_not_isSuccPrelimit (since := "2024-09-05")]
alias mem_range_succ_of_not_isSuccLimit := mem_range_succ_of_not_isSuccPrelimit


theorem mem_range_succ_or_isSuccPrelimit (a) : a ∈ range (succ : α → α) ∨ IsSuccPrelimit a :=
  or_iff_not_imp_right.2 <| mem_range_succ_of_not_isSuccPrelimit


@[deprecated mem_range_succ_or_isSuccPrelimit (since := "2024-09-05")]
alias mem_range_succ_or_isSuccLimit := mem_range_succ_or_isSuccPrelimit


theorem isMin_or_mem_range_succ_or_isSuccLimit (a) :
    IsMin a ∨ a ∈ range (succ : α → α) ∨ IsSuccLimit a := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    a : α
    ⊢ Or (IsMin a) (Or (Membership.mem (Set.range Order.succ) a) (Order.IsSuccLimi …
  -/
  rw [IsSuccLimit]
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    a : α
    ⊢ Or (IsMin a) (Or (Membership.mem (Set.range Order.succ) a) (And (Not (IsMin  …
  -/
  have := mem_range_succ_or_isSuccPrelimit a
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    a : α
    this : Or (Membership.mem (Set.range Order.succ) a) (Order.IsSuccPrelimit a)
    ⊢ Or (IsMin a) (Or (Membership.mem (Set.range Order.succ) a) (And (Not (IsMin  …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem isSuccPrelimit_of_succ_lt (H : ∀ a < b, succ a < b) : IsSuccPrelimit b := fun a hab =>
  (H a hab.lt).ne (CovBy.succ_eq hab)


@[deprecated isSuccPrelimit_of_succ_lt (since := "2024-09-05")]
alias isSuccLimit_of_succ_lt := isSuccPrelimit_of_succ_lt


theorem IsSuccPrelimit.succ_lt (hb : IsSuccPrelimit b) (ha : a < b) : succ a < b := by
  /-
    α : Type u_1
    a b : α
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    hb : Order.IsSuccPrelimit b
    ha : LT.lt a b
    ⊢ LT.lt (Order.succ a) b
  -/
  by_cases h : IsMax a
    /-
      case pos
      α : Type u_1
      a b : α
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      hb : Order.IsSuccPrelimit b
      ha : LT.lt a b
      h : IsMax a
      ⊢ LT.lt (Order.succ a) b
    -/
  · rwa [h.succ_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      a b : α
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      hb : Order.IsSuccPrelimit b
      ha : LT.lt a b
      h : Not (IsMax a)
      ⊢ LT.lt (Order.succ a) b
    -/
  · rw [lt_iff_le_and_ne, succ_le_iff_of_not_isMax h]
    /-
      case neg
      α : Type u_1
      a b : α
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      hb : Order.IsSuccPrelimit b
      ha : LT.lt a b
      h : Not (IsMax a)
      ⊢ And (LT.lt a b) (Ne (Order.succ a) b)
    -/
    refine ⟨ha, fun hab => ?_⟩
    /-
      case neg
      α : Type u_1
      a b : α
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      hb : Order.IsSuccPrelimit b
      ha : LT.lt a b
      h : Not (IsMax a)
      hab : Eq (Order.succ a) b
      ⊢ False
    -/
    subst hab
    /-
      case neg
      α : Type u_1
      a : α
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      h : Not (IsMax a)
      hb : Order.IsSuccPrelimit (Order.succ a)
      ha : LT.lt a (Order.succ a)
      ⊢ False
    -/
    exact (h hb.isMax).elim
    /-
      🎉 no goals
    -/


theorem IsSuccLimit.succ_lt (hb : IsSuccLimit b) (ha : a < b) : succ a < b :=
  hb.isSuccPrelimit.succ_lt ha


theorem IsSuccPrelimit.succ_lt_iff (hb : IsSuccPrelimit b) : succ a < b ↔ a < b :=
  ⟨fun h => (le_succ a).trans_lt h, hb.succ_lt⟩


theorem IsSuccLimit.succ_lt_iff (hb : IsSuccLimit b) : succ a < b ↔ a < b :=
  hb.isSuccPrelimit.succ_lt_iff


theorem isSuccPrelimit_iff_succ_lt : IsSuccPrelimit b ↔ ∀ a < b, succ a < b :=
  ⟨fun hb _ => hb.succ_lt, isSuccPrelimit_of_succ_lt⟩


@[deprecated isSuccPrelimit_iff_succ_lt (since := "2024-09-05")]
alias isSuccLimit_iff_succ_lt := isSuccPrelimit_iff_succ_lt


theorem isSuccPrelimit_iff_succ_ne : IsSuccPrelimit a ↔ ∀ b, succ b ≠ a :=
  ⟨IsSuccPrelimit.succ_ne, isSuccPrelimit_of_succ_ne⟩


@[deprecated isSuccPrelimit_iff_succ_ne (since := "2024-09-05")]
alias isSuccLimit_iff_succ_ne := isSuccPrelimit_iff_succ_ne


theorem not_isSuccPrelimit_iff' : ¬ IsSuccPrelimit a ↔ a ∈ range (succ : α → α) := by
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : NoMaxOrder α
    ⊢ Iff (Not (Order.IsSuccPrelimit a)) (Membership.mem (Set.range Order.succ) a)
  -/
  simp_rw [isSuccPrelimit_iff_succ_ne, not_forall, not_ne_iff, mem_range]
  /-
    🎉 no goals
  -/


@[deprecated not_isSuccPrelimit_iff' (since := "2024-09-05")]
alias not_isSuccLimit_iff' := not_isSuccPrelimit_iff'


protected theorem IsSuccPrelimit.isMin (h : IsSuccPrelimit a) : IsMin a := fun b hb => by
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    h : Order.IsSuccPrelimit a
    b : α
    hb : LE.le b a
    ⊢ LE.le a b
  -/
  revert h
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    b : α
    hb : LE.le b a
    ⊢ Order.IsSuccPrelimit a → LE.le a b
  -/
  refine Succ.rec (fun _ => le_rfl) (fun c _ H hc => ?_) hb
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    b : α
    hb : LE.le b a
    c : α
    x✝ : LE.le b c
    H : Order.IsSuccPrelimit c → LE.le c b
    hc : Order.IsSuccPrelimit (Order.succ c)
    ⊢ LE.le (Order.succ c) b
  -/
  have := hc.isMax.succ_eq
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    b : α
    hb : LE.le b a
    c : α
    x✝ : LE.le b c
    H : Order.IsSuccPrelimit c → LE.le c b
    hc : Order.IsSuccPrelimit (Order.succ c)
    this : Eq (Order.succ c) c
    ⊢ LE.le (Order.succ c) b
  -/
  rw [this] at hc ⊢
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    b : α
    hb : LE.le b a
    c : α
    x✝ : LE.le b c
    H : Order.IsSuccPrelimit c → LE.le c b
    hc : Order.IsSuccPrelimit c
    this : Eq (Order.succ c) c
    ⊢ LE.le c b
  -/
  exact H hc
  /-
    🎉 no goals
  -/


@[simp]
theorem isSuccPrelimit_iff : IsSuccPrelimit a ↔ IsMin a :=
  ⟨IsSuccPrelimit.isMin, IsMin.isSuccPrelimit⟩


@[simp]
theorem not_isSuccLimit : ¬ IsSuccLimit a :=
  fun h ↦ h.not_isMin <| h.isSuccPrelimit.isMin


                                                                     /-
                                                                       α : Type u_1
                                                                       a : α
                                                                       inst✝³ : PartialOrder α
                                                                       inst✝² : SuccOrder α
                                                                       inst✝¹ : IsSuccArchimedean α
                                                                       inst✝ : NoMinOrder α
                                                                       ⊢ Not (Order.IsSuccPrelimit a)
                                                                     -/
theorem not_isSuccPrelimit [NoMinOrder α] : ¬ IsSuccPrelimit a := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem IsSuccPrelimit.le_iff_forall_le (h : IsSuccPrelimit a) : a ≤ b ↔ ∀ c < a, c ≤ b := by
  /-
    α : Type u_1
    a b : α
    inst✝ : LinearOrder α
    h : Order.IsSuccPrelimit a
    ⊢ Iff (LE.le a b) (∀ (c : α), LT.lt c a → LE.le c b)
  -/
  use fun ha c hc ↦ hc.le.trans ha
  /-
    case mpr
    α : Type u_1
    a b : α
    inst✝ : LinearOrder α
    h : Order.IsSuccPrelimit a
    ⊢ (∀ (c : α), LT.lt c a → LE.le c b) → LE.le a b
  -/
  intro H
  /-
    case mpr
    α : Type u_1
    a b : α
    inst✝ : LinearOrder α
    h : Order.IsSuccPrelimit a
    H : ∀ (c : α), LT.lt c a → LE.le c b
    ⊢ LE.le a b
  -/
  by_contra! ha
  /-
    case mpr
    α : Type u_1
    a b : α
    inst✝ : LinearOrder α
    h : Order.IsSuccPrelimit a
    H : ∀ (c : α), LT.lt c a → LE.le c b
    ha : LT.lt b a
    ⊢ False
  -/
  exact h b ⟨ha, fun c hb hc ↦ (H c hc).not_lt hb⟩
  /-
    🎉 no goals
  -/


theorem IsSuccLimit.le_iff_forall_le (h : IsSuccLimit a) : a ≤ b ↔ ∀ c < a, c ≤ b :=
  h.isSuccPrelimit.le_iff_forall_le


theorem IsSuccPrelimit.lt_iff_exists_lt (h : IsSuccPrelimit b) : a < b ↔ ∃ c < b, a < c := by
  /-
    α : Type u_1
    a b : α
    inst✝ : LinearOrder α
    h : Order.IsSuccPrelimit b
    ⊢ Iff (LT.lt a b) (Exists fun c => And (LT.lt c b) (LT.lt a c))
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    a b : α
    inst✝ : LinearOrder α
    h : Order.IsSuccPrelimit b
    ⊢ Iff (Not (LT.lt a b)) (Not (Exists fun c => And (LT.lt c b) (LT.lt a c)))
  -/
  simp [h.le_iff_forall_le]
  /-
    🎉 no goals
  -/


theorem IsSuccLimit.lt_iff_exists_lt (h : IsSuccLimit b) : a < b ↔ ∃ c < b, a < c :=
  h.isSuccPrelimit.lt_iff_exists_lt


theorem IsSuccPrelimit.le_succ_iff (hb : IsSuccPrelimit b) : b ≤ succ a ↔ b ≤ a :=
  le_iff_le_iff_lt_iff_lt.2 hb.succ_lt_iff


theorem IsSuccLimit.le_succ_iff (hb : IsSuccLimit b) : b ≤ succ a ↔ b ≤ a :=
  hb.isSuccPrelimit.le_succ_iff


/-- A predecessor pre-limit is a value that isn't covered by any other.

It's so named because in a predecessor order, a predecessor pre-limit can't be the predecessor of
anything smaller.

Use `IsPredLimit` to exclude the case of a maximal element. -/
def IsPredPrelimit (a : α) : Prop :=
  ∀ b, ¬ a ⋖ b


theorem not_isPredPrelimit_iff_exists_covBy (a : α) : ¬IsPredPrelimit a ↔ ∃ b, a ⋖ b := by
  /-
    α : Type u_1
    inst✝ : LT α
    a : α
    ⊢ Iff (Not (Order.IsPredPrelimit a)) (Exists fun b => CovBy a b)
  -/
  simp [IsPredPrelimit]
  /-
    🎉 no goals
  -/


@[deprecated not_isPredPrelimit_iff_exists_covBy (since := "2024-09-05")]
alias not_isPredLimit_iff_exists_covBy := not_isPredPrelimit_iff_exists_covBy


@[simp]
theorem IsPredPrelimit.of_dense [DenselyOrdered α] (a : α) : IsPredPrelimit a := fun _ => not_covBy


@[deprecated (since := "2024-09-30")] alias isPredPrelimit_of_dense := IsPredPrelimit.of_dense

@[deprecated (since := "2024-09-05")] alias isPredLimit_of_dense := IsPredPrelimit.of_dense


@[simp]
theorem isSuccPrelimit_toDual_iff : IsSuccPrelimit (toDual a) ↔ IsPredPrelimit a := by
  /-
    α : Type u_1
    a : α
    inst✝ : LT α
    ⊢ Iff (Order.IsSuccPrelimit (OrderDual.toDual a)) (Order.IsPredPrelimit a)
  -/
  simp [IsSuccPrelimit, IsPredPrelimit]
  /-
    🎉 no goals
  -/


@[simp]
theorem isPredPrelimit_toDual_iff : IsPredPrelimit (toDual a) ↔ IsSuccPrelimit a := by
  /-
    α : Type u_1
    a : α
    inst✝ : LT α
    ⊢ Iff (Order.IsPredPrelimit (OrderDual.toDual a)) (Order.IsSuccPrelimit a)
  -/
  simp [IsSuccPrelimit, IsPredPrelimit]
  /-
    🎉 no goals
  -/


alias ⟨_, IsPredPrelimit.dual⟩ := isSuccPrelimit_toDual_iff

alias ⟨_, IsSuccPrelimit.dual⟩ := isPredPrelimit_toDual_iff

@[deprecated IsPredPrelimit.dual (since := "2024-09-05")]
alias isPredLimit.dual := IsPredPrelimit.dual

@[deprecated IsSuccPrelimit.dual (since := "2024-09-05")]
alias isSuccLimit.dual := IsSuccPrelimit.dual


/-- A predecessor limit is a value that isn't maximal and doesn't cover any other.

It's so named because in a predecessor order, a predecessor limit can't be the predecessor of
anything larger.

This previously allowed the element to be maximal. This usage is now covered by `IsPredPreLimit`. -/
def IsPredLimit (a : α) : Prop :=
  ¬ IsMax a ∧ IsPredPrelimit a


protected theorem IsPredLimit.not_isMax (h : IsPredLimit a) : ¬ IsMax a := h.1

protected theorem IsPredLimit.isPredPrelimit (h : IsPredLimit a) : IsPredPrelimit a := h.2


@[simp]
theorem isSuccLimit_toDual_iff : IsSuccLimit (toDual a) ↔ IsPredLimit a := by
  /-
    α : Type u_1
    a : α
    inst✝ : Preorder α
    ⊢ Iff (Order.IsSuccLimit (OrderDual.toDual a)) (Order.IsPredLimit a)
  -/
  simp [IsSuccLimit, IsPredLimit]
  /-
    🎉 no goals
  -/


@[simp]
theorem isPredLimit_toDual_iff : IsPredLimit (toDual a) ↔ IsSuccLimit a := by
  /-
    α : Type u_1
    a : α
    inst✝ : Preorder α
    ⊢ Iff (Order.IsPredLimit (OrderDual.toDual a)) (Order.IsSuccLimit a)
  -/
  simp [IsSuccLimit, IsPredLimit]
  /-
    🎉 no goals
  -/


alias ⟨_, IsPredLimit.dual⟩ := isSuccLimit_toDual_iff

alias ⟨_, IsSuccLimit.dual⟩ := isPredLimit_toDual_iff


theorem IsPredPrelimit.isPredLimit_of_not_isMax (h : IsPredPrelimit a) (ha : ¬ IsMax a) :
    IsPredLimit a :=
  ⟨ha, h⟩


theorem IsPredPrelimit.isPredLimit [NoMaxOrder α] (h : IsPredPrelimit a) : IsPredLimit a :=
  h.isPredLimit_of_not_isMax (not_isMax a)


theorem isPredPrelimit_iff_isPredLimit_of_not_isMax (h : ¬ IsMax a) :
    IsPredPrelimit a ↔ IsPredLimit a :=
  ⟨fun ha ↦ ha.isPredLimit_of_not_isMax h, IsPredLimit.isPredPrelimit⟩


theorem isPredPrelimit_iff_isPredLimit [NoMaxOrder α] : IsPredPrelimit a ↔ IsPredLimit a :=
  isPredPrelimit_iff_isPredLimit_of_not_isMax (not_isMax a)


protected theorem _root_.IsMax.not_isPredLimit (h : IsMax a) : ¬ IsPredLimit a :=
  fun ha ↦ ha.not_isMax h


protected theorem _root_.IsMax.isPredPrelimit : IsMax a → IsPredPrelimit a := fun h _ hab =>
  not_isMax_of_lt hab.lt h


@[deprecated _root_.IsMax.isPredPrelimit (since := "2024-09-05")]
alias _root_.IsMax.isPredLimit := _root_.IsMax.isPredPrelimit


theorem isPredPrelimit_top [OrderTop α] : IsPredPrelimit (⊤ : α) :=
  isMax_top.isPredPrelimit


@[deprecated isPredPrelimit_top (since := "2024-09-05")]
alias isPredLimit_top := isPredPrelimit_top


theorem not_isPredLimit_top [OrderTop α] : ¬ IsPredLimit (⊤ : α) :=
  isMax_top.not_isPredLimit


theorem IsPredLimit.ne_top [OrderTop α] (h : IsPredLimit a) : a ≠ ⊤ :=
  h.dual.ne_bot


theorem not_isPredLimit_iff : ¬ IsPredLimit a ↔ IsMax a ∨ ¬ IsPredPrelimit a := by
  /-
    α : Type u_1
    a : α
    inst✝ : Preorder α
    ⊢ Iff (Not (Order.IsPredLimit a)) (Or (IsMax a) (Not (Order.IsPredPrelimit a)))
  -/
  rw [IsPredLimit, not_and_or, not_not]
  /-
    🎉 no goals
  -/


theorem not_isPredLimit_of_not_isPredPrelimit (h : ¬ IsPredPrelimit a) : ¬ IsPredLimit a :=
  not_isPredLimit_iff.2 (Or.inr h)


protected theorem IsPredPrelimit.isMin (h : IsPredPrelimit (pred a)) : IsMin a :=
  h.dual.isMax


protected theorem IsPredLimit.isMin (h : IsPredLimit (pred a)) : IsMin a :=
  h.dual.isMax


theorem not_isPredPrelimit_pred_of_not_isMin (ha : ¬ IsMin a) : ¬ IsPredPrelimit (pred a) :=
  mt IsPredPrelimit.isMin ha


theorem not_isPredLimit_pred_of_not_isMin (ha : ¬ IsMin a) : ¬ IsPredLimit (pred a) :=
  mt IsPredLimit.isMin ha


theorem IsPredPrelimit.pred_ne (h : IsPredPrelimit a) (b : α) : pred b ≠ a :=
  h.dual.succ_ne b


theorem IsPredLimit.pred_ne (h : IsPredLimit a) (b : α) : pred b ≠ a :=
  h.isPredPrelimit.pred_ne b


@[simp]
theorem not_isPredPrelimit_pred (a : α) : ¬ IsPredPrelimit (pred a) := fun h => h.pred_ne _ rfl


@[simp]
theorem not_isPredLimit_pred (a : α) : ¬ IsPredLimit (pred a) := fun h => h.pred_ne _ rfl


theorem IsPredPrelimit.isMax_of_noMin (h : IsPredPrelimit a) : IsMax a :=
  h.dual.isMin_of_noMax


@[deprecated IsPredPrelimit.isMax_of_noMin (since := "2024-09-05")]
alias IsPredLimit.isMax_of_noMin := IsPredPrelimit.isMax_of_noMin


@[simp]
theorem isPredPrelimit_iff_of_noMin : IsPredPrelimit a ↔ IsMax a :=
  ⟨IsPredPrelimit.isMax_of_noMin, IsMax.isPredPrelimit⟩


@[deprecated isPredPrelimit_iff_of_noMin (since := "2024-09-05")]
alias isPredLimit_iff_of_noMin := isPredPrelimit_iff_of_noMin


                                                                              /-
                                                                                α : Type u_1
                                                                                a : α
                                                                                inst✝⁴ : Preorder α
                                                                                inst✝³ : PredOrder α
                                                                                inst✝² : IsPredArchimedean α
                                                                                inst✝¹ : NoMinOrder α
                                                                                inst✝ : NoMaxOrder α
                                                                                ⊢ Not (Order.IsPredPrelimit a)
                                                                              -/
theorem not_isPredPrelimit_of_noMin [NoMaxOrder α] : ¬ IsPredPrelimit a := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem not_isPredLimit_of_noMin : ¬ IsPredLimit a :=
  fun h ↦ h.not_isMax h.isPredPrelimit.isMax_of_noMin


theorem isPredLimit_iff [OrderTop α] : IsPredLimit a ↔ a ≠ ⊤ ∧ IsPredPrelimit a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : OrderTop α
    ⊢ Iff (Order.IsPredLimit a) (And (Ne a Top.top) (Order.IsPredPrelimit a))
  -/
  rw [IsPredLimit, isMax_iff_eq_top]
  /-
    🎉 no goals
  -/


theorem IsPredLimit.lt_top [OrderTop α] (h : IsPredLimit a) : a < ⊤ :=
  h.ne_top.lt_top


theorem isPredPrelimit_of_pred_ne (h : ∀ b, pred b ≠ a) : IsPredPrelimit a := fun b hba =>
  h b (CovBy.pred_eq hba)


@[deprecated isPredPrelimit_of_pred_ne (since := "2024-09-05")]
alias isPredLimit_of_pred_ne := isPredPrelimit_of_pred_ne


theorem not_isPredPrelimit_iff : ¬ IsPredPrelimit a ↔ ∃ b, ¬ IsMin b ∧ pred b = a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : PredOrder α
    ⊢ Iff (Not (Order.IsPredPrelimit a)) (Exists fun b => And (Not (IsMin b)) (Eq  …
  -/
  rw [← isSuccPrelimit_toDual_iff]
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : PredOrder α
    ⊢ Iff (Not (Order.IsSuccPrelimit (OrderDual.toDual a))) (Exists fun b => And ( …
  -/
  exact not_isSuccPrelimit_iff
  /-
    🎉 no goals
  -/


/-- See `not_isPredPrelimit_iff` for a version that states that `a` is a successor of a value other
than itself. -/
theorem mem_range_pred_of_not_isPredPrelimit (h : ¬ IsPredPrelimit a) :
    a ∈ range (pred : α → α) := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : PredOrder α
    h : Not (Order.IsPredPrelimit a)
    ⊢ Membership.mem (Set.range Order.pred) a
  -/
  obtain ⟨b, hb⟩ := not_isPredPrelimit_iff.1 h
  /-
    case intro
    α : Type u_1
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : PredOrder α
    h : Not (Order.IsPredPrelimit a)
    b : α
    hb : And (Not (IsMin b)) (Eq (Order.pred b) a)
    ⊢ Membership.mem (Set.range Order.pred) a
  -/
  exact ⟨b, hb.2⟩
  /-
    🎉 no goals
  -/


@[deprecated mem_range_pred_of_not_isPredPrelimit (since := "2024-09-05")]
alias mem_range_pred_of_not_isPredLimit := mem_range_pred_of_not_isPredPrelimit


theorem mem_range_pred_or_isPredPrelimit (a) : a ∈ range (pred : α → α) ∨ IsPredPrelimit a :=
  or_iff_not_imp_right.2 <| mem_range_pred_of_not_isPredPrelimit


@[deprecated mem_range_pred_or_isPredPrelimit (since := "2024-09-05")]
alias mem_range_pred_or_isPredLimit := mem_range_pred_or_isPredPrelimit


theorem isPredPrelimit_of_pred_lt (H : ∀ b > a, a < pred b) : IsPredPrelimit a := fun a hab =>
  (H a hab.lt).ne (CovBy.pred_eq hab).symm


@[deprecated isPredPrelimit_of_pred_lt (since := "2024-09-05")]
alias isPredLimit_of_pred_lt := isPredPrelimit_of_pred_lt


theorem IsPredPrelimit.lt_pred (ha : IsPredPrelimit a) (hb : a < b) : a < pred b :=
  ha.dual.succ_lt hb


theorem IsPredLimit.lt_pred (ha : IsPredLimit a) (hb : a < b) : a < pred b :=
  ha.isPredPrelimit.lt_pred hb


theorem IsPredPrelimit.lt_pred_iff (ha : IsPredPrelimit a) : a < pred b ↔ a < b :=
  ha.dual.succ_lt_iff


theorem IsPredLimit.lt_pred_iff (ha : IsPredLimit a) : a < pred b ↔ a < b :=
  ha.dual.succ_lt_iff


theorem isPredPrelimit_iff_lt_pred : IsPredPrelimit a ↔ ∀ b > a, a < pred b :=
  ⟨fun hb _ => hb.lt_pred, isPredPrelimit_of_pred_lt⟩


@[deprecated isPredPrelimit_iff_lt_pred (since := "2024-09-05")]
alias isPredLimit_iff_lt_pred := isPredPrelimit_iff_lt_pred


theorem isPredPrelimit_iff_pred_ne : IsPredPrelimit a ↔ ∀ b, pred b ≠ a :=
  ⟨IsPredPrelimit.pred_ne, isPredPrelimit_of_pred_ne⟩


theorem not_isPredPrelimit_iff' : ¬ IsPredPrelimit a ↔ a ∈ range (pred : α → α) := by
  /-
    α : Type u_1
    a : α
    inst✝² : PartialOrder α
    inst✝¹ : PredOrder α
    inst✝ : NoMinOrder α
    ⊢ Iff (Not (Order.IsPredPrelimit a)) (Membership.mem (Set.range Order.pred) a)
  -/
  simp_rw [isPredPrelimit_iff_pred_ne, not_forall, not_ne_iff, mem_range]
  /-
    🎉 no goals
  -/


protected theorem IsPredPrelimit.isMax (h : IsPredPrelimit a) : IsMax a :=
  h.dual.isMin


@[deprecated IsPredPrelimit.isMax (since := "2024-09-05")]
alias IsPredLimit.isMax := IsPredPrelimit.isMax


@[simp]
theorem isPredPrelimit_iff : IsPredPrelimit a ↔ IsMax a :=
  ⟨IsPredPrelimit.isMax, IsMax.isPredPrelimit⟩


@[simp]
theorem not_isPredLimit : ¬ IsPredLimit a :=
  fun h ↦ h.not_isMax <| h.isPredPrelimit.isMax


                                                                     /-
                                                                       α : Type u_1
                                                                       a : α
                                                                       inst✝³ : PartialOrder α
                                                                       inst✝² : PredOrder α
                                                                       inst✝¹ : IsPredArchimedean α
                                                                       inst✝ : NoMaxOrder α
                                                                       ⊢ Not (Order.IsPredPrelimit a)
                                                                     -/
theorem not_isPredPrelimit [NoMaxOrder α] : ¬ IsPredPrelimit a := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem IsPredPrelimit.le_iff_forall_le (h : IsPredPrelimit a) : b ≤ a ↔ ∀ ⦃c⦄, a < c → b ≤ c :=
  h.dual.le_iff_forall_le


theorem IsPredLimit.le_iff_forall_le (h : IsPredLimit a) : b ≤ a ↔ ∀ ⦃c⦄, a < c → b ≤ c :=
  h.dual.le_iff_forall_le


theorem IsPredPrelimit.lt_iff_exists_lt (h : IsPredPrelimit b) : b < a ↔ ∃ c, b < c ∧ c < a :=
  h.dual.lt_iff_exists_lt


theorem IsPredLimit.lt_iff_exists_lt (h : IsPredLimit b) : b < a ↔ ∃ c, b < c ∧ c < a :=
  h.dual.lt_iff_exists_lt


theorem IsPredPrelimit.pred_le_iff (hb : IsPredPrelimit b) : pred a ≤ b ↔ a ≤ b :=
  hb.dual.le_succ_iff


theorem IsPredLimit.pred_le_iff (hb : IsPredLimit b) : pred a ≤ b ↔ a ≤ b :=
  hb.dual.le_succ_iff


variable (b) in
open Classical in
/-- A value can be built by building it on successors and successor pre-limits. -/
@[elab_as_elim]
noncomputable def isSuccPrelimitRecOn : C b :=
  if hb : IsSuccPrelimit b then hl b hb else
    haveI H := Classical.choose_spec (not_isSuccPrelimit_iff.1 hb)
    cast (congr_arg C H.2) (hs _ H.1)


theorem isSuccPrelimitRecOn_of_isSuccPrelimit (hb : IsSuccPrelimit b) :
    isSuccPrelimitRecOn b hs hl = hl b hb :=
  dif_pos hb


@[deprecated isSuccPrelimitRecOn_of_isSuccPrelimit (since := "2024-09-05")]
alias isSuccLimitRecOn_limit := isSuccPrelimitRecOn_of_isSuccPrelimit

@[deprecated isSuccPrelimitRecOn_of_isSuccPrelimit (since := "2024-09-14")]
alias isSuccPrelimitRecOn_limit := isSuccPrelimitRecOn_of_isSuccPrelimit


theorem isSuccPrelimitRecOn_succ_of_not_isMax (hb : ¬ IsMax b) :
    isSuccPrelimitRecOn (succ b) hs hl = hs b hb := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → C a
    hb : Not (IsMax b)
    ⊢ Eq (Order.isSuccPrelimitRecOn (Order.succ b) hs hl) (hs b hb)
  -/
  have hb' := not_isSuccPrelimit_succ_of_not_isMax hb
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → C a
    hb : Not (IsMax b)
    hb' : Not (Order.IsSuccPrelimit (Order.succ b))
    ⊢ Eq (Order.isSuccPrelimitRecOn (Order.succ b) hs hl) (hs b hb)
  -/
  have H := Classical.choose_spec (not_isSuccPrelimit_iff.1 hb')
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → C a
    hb : Not (IsMax b)
    hb' : Not (Order.IsSuccPrelimit (Order.succ b))
    H : And (Not (IsMax (Classical.choose ⋯))) (Eq (Order.succ (Classical.choose ⋯ …
    ⊢ Eq (Order.isSuccPrelimitRecOn (Order.succ b) hs hl) (hs b hb)
  -/
  rw [isSuccPrelimitRecOn, dif_neg hb', cast_eq_iff_heq]
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → C a
    hb : Not (IsMax b)
    hb' : Not (Order.IsSuccPrelimit (Order.succ b))
    H : And (Not (IsMax (Classical.choose ⋯))) (Eq (Order.succ (Classical.choose ⋯ …
    ⊢ HEq (hs (Classical.choose ⋯) ⋯) (hs b hb)
  -/
  congr
  /-
    case e_1
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → C a
    hb : Not (IsMax b)
    hb' : Not (Order.IsSuccPrelimit (Order.succ b))
    H : And (Not (IsMax (Classical.choose ⋯))) (Eq (Order.succ (Classical.choose ⋯ …
    ⊢ Eq (Classical.choose ⋯) b
  -/
  exacts [(succ_eq_succ_iff_of_not_isMax H.1 hb).1 H.2, proof_irrel_heq _ _]
  /-
    🎉 no goals
  -/


@[deprecated isSuccPrelimitRecOn_succ_of_not_isMax (since := "2024-09-05")]
alias isSuccLimitRecOn_succ' := isSuccPrelimitRecOn_succ_of_not_isMax

@[deprecated isSuccPrelimitRecOn_succ_of_not_isMax (since := "2024-09-14")]
alias isSuccPrelimitRecOn_succ' := isSuccPrelimitRecOn_succ_of_not_isMax


@[simp]
theorem isSuccPrelimitRecOn_succ [NoMaxOrder α] (b : α) :
    isSuccPrelimitRecOn (succ b) hs hl = hs b (not_isMax b) :=
  isSuccPrelimitRecOn_succ_of_not_isMax _ _ _


variable (b) in
/-- A value can be built by building it on predecessors and predecessor pre-limits. -/
@[elab_as_elim]
noncomputable def isPredPrelimitRecOn : C b :=
  isSuccPrelimitRecOn (α := αᵒᵈ) b hs (fun a ha ↦ hl a ha.dual)


theorem isPredPrelimitRecOn_of_isPredPrelimit (hb : IsPredPrelimit b) :
    isPredPrelimitRecOn b hs hl = hl b hb :=
  isSuccPrelimitRecOn_of_isSuccPrelimit _ _ hb.dual


@[deprecated isPredPrelimitRecOn_of_isPredPrelimit (since := "2024-09-05")]
alias isPredLimitRecOn_limit := isPredPrelimitRecOn_of_isPredPrelimit

@[deprecated isPredPrelimitRecOn_of_isPredPrelimit (since := "2024-09-14")]
alias isPredPrelimitRecOn_limit := isPredPrelimitRecOn_of_isPredPrelimit


theorem isPredPrelimitRecOn_pred_of_not_isMin (hb : ¬ IsMin b) :
    isPredPrelimitRecOn (pred b) hs hl = hs b hb :=
  isSuccPrelimitRecOn_succ_of_not_isMax (α := αᵒᵈ) _ _ _


@[deprecated isPredPrelimitRecOn_pred_of_not_isMin (since := "2024-09-05")]
alias isPredLimitRecOn_pred' := isPredPrelimitRecOn_pred_of_not_isMin

@[deprecated isPredPrelimitRecOn_pred_of_not_isMin (since := "2024-09-14")]
alias isPredPrelimitRecOn_pred' := isPredPrelimitRecOn_pred_of_not_isMin


@[simp]
theorem isPredPrelimitRecOn_pred [NoMinOrder α] (b : α) :
    isPredPrelimitRecOn (pred b) hs hl = hs b (not_isMin b) :=
  isPredPrelimitRecOn_pred_of_not_isMin _ _ _


variable (b) in
open Classical in
/-- A value can be built by building it on minimal elements, successors, and successor limits. -/
@[elab_as_elim]
noncomputable def isSuccLimitRecOn : C b :=
  isSuccPrelimitRecOn b hs fun a ha ↦
    if h : IsMin a then hm a h else hl a (ha.isSuccLimit_of_not_isMin h)


@[simp]
theorem isSuccLimitRecOn_of_isSuccLimit (hb : IsSuccLimit b) :
    isSuccLimitRecOn b hm hs hl = hl b hb := by
  rw [isSuccLimitRecOn, isSuccPrelimitRecOn_of_isSuccPrelimit _ _ hb.isSuccPrelimit,
    dif_neg hb.not_isMin]


theorem isSuccLimitRecOn_succ_of_not_isMax (hb : ¬ IsMax b) :
    isSuccLimitRecOn (succ b) hm hs hl = hs b hb := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hm : (a : α) → IsMin a → C a
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccLimit a → C a
    hb : Not (IsMax b)
    ⊢ Eq (Order.isSuccLimitRecOn (Order.succ b) hm hs hl) (hs b hb)
  -/
  rw [isSuccLimitRecOn, isSuccPrelimitRecOn_succ_of_not_isMax]
  /-
    🎉 no goals
  -/


@[simp]
theorem isSuccLimitRecOn_succ [NoMaxOrder α] (b : α) :
    isSuccLimitRecOn (succ b) hm hs hl = hs b (not_isMax b) :=
  isSuccLimitRecOn_succ_of_not_isMax hm hs hl _


theorem isSuccLimitRecOn_of_isMin (hb : IsMin b) : isSuccLimitRecOn b hm hs hl = hm b hb := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    hm : (a : α) → IsMin a → C a
    hs : (a : α) → Not (IsMax a) → C (Order.succ a)
    hl : (a : α) → Order.IsSuccLimit a → C a
    hb : IsMin b
    ⊢ Eq (Order.isSuccLimitRecOn b hm hs hl) (hm b hb)
  -/
  rw [isSuccLimitRecOn, isSuccPrelimitRecOn_of_isSuccPrelimit _ _ hb.isSuccPrelimit, dif_pos hb]
  /-
    🎉 no goals
  -/


variable (b) in
/-- A value can be built by building it on maximal elements, predecessors,
and predecessor limits. -/
@[elab_as_elim]
noncomputable def isPredLimitRecOn : C b :=
  isSuccLimitRecOn (α := αᵒᵈ) b hm hs (fun a ha => hl a ha.dual)


@[simp]
theorem isPredLimitRecOn_of_isPredLimit (hb : IsPredLimit b) :
    isPredLimitRecOn b hm hs hl = hl b hb :=
  isSuccLimitRecOn_of_isSuccLimit (α := αᵒᵈ) hm hs _ hb.dual


theorem isPredLimitRecOn_pred_of_not_isMin (hb : ¬ IsMin b) :
    isPredLimitRecOn (pred b) hm hs hl = hs b hb :=
  isSuccLimitRecOn_succ_of_not_isMax (α := αᵒᵈ) hm hs _ hb


@[simp]
theorem isPredLimitRecOn_pred [NoMinOrder α] :
    isPredLimitRecOn (pred b) hm hs hl = hs b (not_isMin b) :=
  isSuccLimitRecOn_succ (α := αᵒᵈ) hm hs _ b


theorem isPredLimitRecOn_of_isMax (hb : IsMax b) : isPredLimitRecOn b hm hs hl = hm b hb :=
  isSuccLimitRecOn_of_isMin (α := αᵒᵈ) hm hs _ hb


variable (b) in
open Classical in
/-- Recursion principle on a well-founded partial `SuccOrder`. -/
@[elab_as_elim] noncomputable def prelimitRecOn : C b :=
  wellFounded_lt.fix
    (fun a IH ↦ if h : IsSuccPrelimit a then hl a h IH else
      haveI H := Classical.choose_spec (not_isSuccPrelimit_iff.1 h)
      cast (congr_arg C H.2) (hs _ H.1 <| IH _ <| H.2.subst <| lt_succ_of_not_isMax H.1))
    b


@[simp]
theorem prelimitRecOn_of_isSuccPrelimit (hb : IsSuccPrelimit b) :
    prelimitRecOn b hs hl = hl b hb fun x _ ↦ SuccOrder.prelimitRecOn x hs hl := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Order.IsSuccPrelimit b
    ⊢ Eq (SuccOrder.prelimitRecOn b hs hl) (hl b hb fun x x_1 => SuccOrder.prelimi …
  -/
  rw [prelimitRecOn, WellFounded.fix_eq, dif_pos hb]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


@[deprecated prelimitRecOn_of_isSuccPrelimit (since := "2024-09-05")]
alias limitRecOn_limit := prelimitRecOn_of_isSuccPrelimit

@[deprecated prelimitRecOn_of_isSuccPrelimit (since := "2024-09-14")]
alias prelimitRecOn_limit := prelimitRecOn_of_isSuccPrelimit


theorem prelimitRecOn_succ_of_not_isMax (hb : ¬ IsMax b) :
    prelimitRecOn (Order.succ b) hs hl = hs b hb (prelimitRecOn b hs hl) := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Not (IsMax b)
    ⊢ Eq (SuccOrder.prelimitRecOn (Order.succ b) hs hl) (hs b hb (SuccOrder.prelim …
  -/
  have h := not_isSuccPrelimit_succ_of_not_isMax hb
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Not (IsMax b)
    h : Not (Order.IsSuccPrelimit (Order.succ b))
    ⊢ Eq (SuccOrder.prelimitRecOn (Order.succ b) hs hl) (hs b hb (SuccOrder.prelim …
  -/
  have H := Classical.choose_spec (not_isSuccPrelimit_iff.1 h)
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Not (IsMax b)
    h : Not (Order.IsSuccPrelimit (Order.succ b))
    H : And (Not (IsMax (Classical.choose ⋯))) (Eq (Order.succ (Classical.choose ⋯ …
    ⊢ Eq (SuccOrder.prelimitRecOn (Order.succ b) hs hl) (hs b hb (SuccOrder.prelim …
  -/
  rw [prelimitRecOn, WellFounded.fix_eq, dif_neg h]
  have {a c : α} {ha hc} {x : ∀ a, C a} (h : a = c) :
    cast (congr_arg (C ∘ succ) h) (hs a ha (x a)) = hs c hc (x c) := by subst h; rfl
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccPrelimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Not (IsMax b)
    h : Not (Order.IsSuccPrelimit (Order.succ b))
    H : And (Not (IsMax (Classical.choose ⋯))) (Eq (Order.succ (Classical.choose ⋯ …
    this : ∀ {a c : α} {ha : Not (IsMax a)} {hc : Not (IsMax c)} {x : (a : α) → C  …
    ⊢ Eq (cast ⋯ (hs (Classical.choose ⋯) ⋯ ((fun y x => ⋯.fix (fun a IH => dite ( …
  -/
  exact this <| (succ_eq_succ_iff_of_not_isMax H.1 hb).1 H.2
  /-
    🎉 no goals
  -/


@[deprecated prelimitRecOn_succ_of_not_isMax (since := "2024-09-05")]
alias limitRecOn_succ' := prelimitRecOn_succ_of_not_isMax

@[deprecated prelimitRecOn_succ_of_not_isMax (since := "2024-09-14")]
alias prelimitRecOn_succ' := prelimitRecOn_succ_of_not_isMax


@[simp]
theorem prelimitRecOn_succ [NoMaxOrder α] (b : α) :
    prelimitRecOn (Order.succ b) hs hl = hs b (not_isMax b) (prelimitRecOn b hs hl) :=
  prelimitRecOn_succ_of_not_isMax _ _ _


variable (b) in
open Classical in
/-- Recursion principle on a well-founded partial `SuccOrder`, separating out the case of a
minimal element. -/
@[elab_as_elim] noncomputable def limitRecOn : C b :=
  prelimitRecOn b hs fun a ha IH ↦
    if h : IsMin a then hm a h else hl a (ha.isSuccLimit_of_not_isMin h) IH


@[simp]
theorem limitRecOn_isMin (hb : IsMin b) : limitRecOn b hm hs hl = hm b hb := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hm : (a : α) → IsMin a → C a
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccLimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : IsMin b
    ⊢ Eq (SuccOrder.limitRecOn b hm hs hl) (hm b hb)
  -/
  rw [limitRecOn, prelimitRecOn_of_isSuccPrelimit _ _ hb.isSuccPrelimit, dif_pos hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem limitRecOn_of_isSuccLimit (hb : IsSuccLimit b) :
    limitRecOn b hm hs hl = hl b hb fun x _ ↦ limitRecOn x hm hs hl := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hm : (a : α) → IsMin a → C a
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccLimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Order.IsSuccLimit b
    ⊢ Eq (SuccOrder.limitRecOn b hm hs hl) (hl b hb fun x x_1 => SuccOrder.limitRe …
  -/
  rw [limitRecOn, prelimitRecOn_of_isSuccPrelimit _ _ hb.isSuccPrelimit, dif_neg hb.not_isMin]; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem limitRecOn_succ_of_not_isMax (hb : ¬ IsMax b) :
    limitRecOn (Order.succ b) hm hs hl = hs b hb (limitRecOn b hm hs hl) := by
  /-
    α : Type u_1
    b : α
    C : α → Sort u_2
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : WellFoundedLT α
    hm : (a : α) → IsMin a → C a
    hs : (a : α) → Not (IsMax a) → C a → C (Order.succ a)
    hl : (a : α) → Order.IsSuccLimit a → ((b : α) → LT.lt b a → C b) → C a
    hb : Not (IsMax b)
    ⊢ Eq (SuccOrder.limitRecOn (Order.succ b) hm hs hl) (hs b hb (SuccOrder.limitR …
  -/
  rw [limitRecOn, prelimitRecOn_succ_of_not_isMax]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem limitRecOn_succ [NoMaxOrder α] (b : α) :
    limitRecOn (Order.succ b) hm hs hl = hs b (not_isMax b) (limitRecOn b hm hs hl) :=
  limitRecOn_succ_of_not_isMax hm hs hl _


variable (b) in
/-- Recursion principle on a well-founded partial `PredOrder`. -/
@[elab_as_elim] noncomputable def prelimitRecOn : C b :=
  SuccOrder.prelimitRecOn (α := αᵒᵈ) b hp (fun a ha => hl a ha.dual)


@[simp]
theorem prelimitRecOn_of_isPredPrelimit (hb : IsPredPrelimit b) :
    prelimitRecOn b hp hl = hl b hb fun x _ ↦ prelimitRecOn x hp hl :=
  SuccOrder.prelimitRecOn_of_isSuccPrelimit _ _ hb.dual


@[deprecated prelimitRecOn_of_isPredPrelimit (since := "2024-09-05")]
alias limitRecOn_limit := prelimitRecOn_of_isPredPrelimit

@[deprecated prelimitRecOn_of_isPredPrelimit (since := "2024-09-14")]
alias prelimitRecOn_limit := prelimitRecOn_of_isPredPrelimit


theorem prelimitRecOn_pred_of_not_isMin (hb : ¬ IsMin b) :
    prelimitRecOn (Order.pred b) hp hl = hp b hb (prelimitRecOn b hp hl) :=
  SuccOrder.prelimitRecOn_succ_of_not_isMax _ _ _


@[deprecated prelimitRecOn_pred_of_not_isMin (since := "2024-09-05")]
alias limitRecOn_pred' := prelimitRecOn_pred_of_not_isMin

@[deprecated prelimitRecOn_pred_of_not_isMin (since := "2024-09-14")]
alias prelimitRecOn_pred' := prelimitRecOn_pred_of_not_isMin


@[simp]
theorem prelimitRecOn_pred [NoMinOrder α] (b : α) :
    prelimitRecOn (Order.pred b) hp hl = hp b (not_isMin b) (prelimitRecOn b hp hl) :=
  prelimitRecOn_pred_of_not_isMin _ _ _


variable (b) in
open Classical in
/-- Recursion principle on a well-founded partial `PredOrder`, separating out the case of a
maximal element. -/
@[elab_as_elim] noncomputable def limitRecOn : C b :=
  SuccOrder.limitRecOn (α := αᵒᵈ) b hm hs (fun a ha => hl a ha.dual)


@[simp]
theorem limitRecOn_isMax (hb : IsMax b) : limitRecOn b hm hs hl = hm b hb :=
  SuccOrder.limitRecOn_isMin (α := αᵒᵈ) hm hs _ hb


@[simp]
theorem limitRecOn_of_isPredLimit (hb : IsPredLimit b) :
    limitRecOn b hm hs hl = hl b hb fun x _ ↦ limitRecOn x hm hs hl :=
  SuccOrder.limitRecOn_of_isSuccLimit (α := αᵒᵈ) hm hs _ hb.dual


theorem limitRecOn_pred_of_not_isMin (hb : ¬ IsMin b) :
    limitRecOn (Order.pred b) hm hs hl = hs b hb (limitRecOn b hm hs hl) :=
  SuccOrder.limitRecOn_succ_of_not_isMax (α := αᵒᵈ) hm hs _ hb


@[simp]
theorem limitRecOn_pred [NoMinOrder α] (b : α) :
    limitRecOn (Order.pred b) hm hs hl = hs b (not_isMin b) (limitRecOn b hm hs hl) :=
  SuccOrder.limitRecOn_succ (α := αᵒᵈ) hm hs _ b


