/-- A set is equitable if no element value is more than one bigger than another. -/
def EquitableOn [LE β] [Add β] [One β] (s : Set α) (f : α → β) : Prop :=
  ∀ ⦃a₁ a₂⦄, a₁ ∈ s → a₂ ∈ s → f a₁ ≤ f a₂ + 1


@[simp]
theorem equitableOn_empty [LE β] [Add β] [One β] (f : α → β) : EquitableOn ∅ f := fun a _ ha =>
  (Set.not_mem_empty a ha).elim


theorem equitableOn_iff_exists_le_le_add_one {s : Set α} {f : α → ℕ} :
    s.EquitableOn f ↔ ∃ b, ∀ a ∈ s, b ≤ f a ∧ f a ≤ b + 1 := by
  /-
    α : Type u_1
    s : Set α
    f : α → Nat
    ⊢ Iff (s.EquitableOn f) (Exists fun b => ∀ (a : α), Membership.mem s a → And ( …
  -/
  refine ⟨?_, fun ⟨b, hb⟩ x y hx hy => (hb x hx).2.trans (add_le_add_right (hb y hy).1 _)⟩
  /-
    α : Type u_1
    s : Set α
    f : α → Nat
    ⊢ s.EquitableOn f → Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le …
  -/
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      f : α → Nat
      ⊢ EmptyCollection.emptyCollection.EquitableOn f → Exists fun b => ∀ (a : α), M …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    ⊢ s.EquitableOn f → Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le …
  -/
  intro hs
  /-
    case inr.intro
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    hs : s.EquitableOn f
    ⊢ Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le b (f a)) (LE.le ( …
  -/
  by_cases h : ∀ y ∈ s, f x ≤ f y
    /-
      case pos
      α : Type u_1
      s : Set α
      f : α → Nat
      x : α
      hx : Membership.mem s x
      hs : s.EquitableOn f
      h : ∀ (y : α), Membership.mem s y → LE.le (f x) (f y)
      ⊢ Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le b (f a)) (LE.le ( …
    -/
  · exact ⟨f x, fun y hy => ⟨h _ hy, hs hy hx⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    hs : s.EquitableOn f
    h : Not (∀ (y : α), Membership.mem s y → LE.le (f x) (f y))
    ⊢ Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le b (f a)) (LE.le ( …
  -/
  push_neg at h
  /-
    case neg
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    hs : s.EquitableOn f
    h : Exists fun y => And (Membership.mem s y) (LT.lt (f y) (f x))
    ⊢ Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le b (f a)) (LE.le ( …
  -/
  obtain ⟨w, hw, hwx⟩ := h
  /-
    case neg.intro.intro
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    hs : s.EquitableOn f
    w : α
    hw : Membership.mem s w
    hwx : LT.lt (f w) (f x)
    ⊢ Exists fun b => ∀ (a : α), Membership.mem s a → And (LE.le b (f a)) (LE.le ( …
  -/
  refine ⟨f w, fun y hy => ⟨Nat.le_of_succ_le_succ ?_, hs hy hw⟩⟩
  /-
    case neg.intro.intro
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    hs : s.EquitableOn f
    w : α
    hw : Membership.mem s w
    hwx : LT.lt (f w) (f x)
    y : α
    hy : Membership.mem s y
    ⊢ LE.le (f w).succ (f y).succ
  -/
  rw [(Nat.succ_le_of_lt hwx).antisymm (hs hx hw)]
  /-
    case neg.intro.intro
    α : Type u_1
    s : Set α
    f : α → Nat
    x : α
    hx : Membership.mem s x
    hs : s.EquitableOn f
    w : α
    hw : Membership.mem s w
    hwx : LT.lt (f w) (f x)
    y : α
    hy : Membership.mem s y
    ⊢ LE.le (f x) (f y).succ
  -/
  exact hs hx hy
  /-
    🎉 no goals
  -/


theorem equitableOn_iff_exists_image_subset_icc {s : Set α} {f : α → ℕ} :
    s.EquitableOn f ↔ ∃ b, f '' s ⊆ Icc b (b + 1) := by
  /-
    α : Type u_1
    s : Set α
    f : α → Nat
    ⊢ Iff (s.EquitableOn f) (Exists fun b => HasSubset.Subset (Set.image f s) (Set …
  -/
  simpa only [image_subset_iff] using equitableOn_iff_exists_le_le_add_one
  /-
    🎉 no goals
  -/


theorem equitableOn_iff_exists_eq_eq_add_one {s : Set α} {f : α → ℕ} :
    s.EquitableOn f ↔ ∃ b, ∀ a ∈ s, f a = b ∨ f a = b + 1 := by
  /-
    α : Type u_1
    s : Set α
    f : α → Nat
    ⊢ Iff (s.EquitableOn f) (Exists fun b => ∀ (a : α), Membership.mem s a → Or (E …
  -/
  simp_rw [equitableOn_iff_exists_le_le_add_one, Nat.le_and_le_add_one_iff]
  /-
    🎉 no goals
  -/


@[simp]
lemma not_equitableOn : ¬s.EquitableOn f ↔ ∃ a ∈ s, ∃ b ∈ s, f b + 1 < f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder β
    inst✝¹ : Add β
    inst✝ : One β
    s : Set α
    f : α → β
    ⊢ Iff (Not (s.EquitableOn f)) (Exists fun a => And (Membership.mem s a) (Exist …
  -/
  simp [EquitableOn]
  /-
    🎉 no goals
  -/


theorem Subsingleton.equitableOn {s : Set α} (hs : s.Subsingleton) (f : α → β) : s.EquitableOn f :=
  fun i j hi hj => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : OrderedSemiring β
    s : Set α
    hs : s.Subsingleton
    f : α → β
    i j : α
    hi : Membership.mem s i
    hj : Membership.mem s j
    ⊢ LE.le (f i) (HAdd.hAdd (f j) 1)
  -/
  rw [hs hi hj]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : OrderedSemiring β
    s : Set α
    hs : s.Subsingleton
    f : α → β
    i j : α
    hi : Membership.mem s i
    hj : Membership.mem s j
    ⊢ LE.le (f j) (HAdd.hAdd (f j) 1)
  -/
  exact le_add_of_nonneg_right zero_le_one
  /-
    🎉 no goals
  -/


theorem equitableOn_singleton (a : α) (f : α → β) : Set.EquitableOn {a} f :=
  Set.subsingleton_singleton.equitableOn f


theorem equitableOn_iff_le_le_add_one :
    EquitableOn (s : Set α) f ↔
      ∀ a ∈ s, (∑ i ∈ s, f i) / s.card ≤ f a ∧ f a ≤ (∑ i ∈ s, f i) / s.card + 1 := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Nat
    ⊢ Iff ((↑s).EquitableOn f) (∀ (a : α), Membership.mem s a → And (LE.le (HDiv.h …
  -/
  rw [Set.equitableOn_iff_exists_le_le_add_one]
  /-
    α : Type u_1
    s : Finset α
    f : α → Nat
    ⊢ Iff (Exists fun b => ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a))  …
  -/
  refine ⟨?_, fun h => ⟨_, h⟩⟩
  /-
    α : Type u_1
    s : Finset α
    f : α → Nat
    ⊢ (Exists fun b => ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE. …
  -/
  rintro ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    s : Finset α
    f : α → Nat
    b : Nat
    hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
    ⊢ ∀ (a : α), Membership.mem s a → And (LE.le (HDiv.hDiv (s.sum fun i => f i) s …
  -/
  by_cases h : ∀ a ∈ s, f a = b + 1
    /-
      case pos
      α : Type u_1
      s : Finset α
      f : α → Nat
      b : Nat
      hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
      h : ∀ (a : α), Membership.mem s a → Eq (f a) (HAdd.hAdd b 1)
      ⊢ ∀ (a : α), Membership.mem s a → And (LE.le (HDiv.hDiv (s.sum fun i => f i) s …
    -/
  · intro a ha
    /-
      case pos
      α : Type u_1
      s : Finset α
      f : α → Nat
      b : Nat
      hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
      h : ∀ (a : α), Membership.mem s a → Eq (f a) (HAdd.hAdd b 1)
      a : α
      ha : Membership.mem s a
      ⊢ And (LE.le (HDiv.hDiv (s.sum fun i => f i) s.card) (f a)) (LE.le (f a) (HAdd …
    -/
    rw [h _ ha, sum_const_nat h, Nat.mul_div_cancel_left _ (card_pos.2 ⟨a, ha⟩)]
    /-
      case pos
      α : Type u_1
      s : Finset α
      f : α → Nat
      b : Nat
      hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
      h : ∀ (a : α), Membership.mem s a → Eq (f a) (HAdd.hAdd b 1)
      a : α
      ha : Membership.mem s a
      ⊢ And (LE.le (HAdd.hAdd b 1) (HAdd.hAdd b 1)) (LE.le (HAdd.hAdd b 1) (HAdd.hAd …
    -/
    exact ⟨le_rfl, Nat.le_succ _⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    s : Finset α
    f : α → Nat
    b : Nat
    hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
    h : Not (∀ (a : α), Membership.mem s a → Eq (f a) (HAdd.hAdd b 1))
    ⊢ ∀ (a : α), Membership.mem s a → And (LE.le (HDiv.hDiv (s.sum fun i => f i) s …
  -/
  push_neg at h
  /-
    case neg
    α : Type u_1
    s : Finset α
    f : α → Nat
    b : Nat
    hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
    h : Exists fun a => And (Membership.mem s a) (Ne (f a) (HAdd.hAdd b 1))
    ⊢ ∀ (a : α), Membership.mem s a → And (LE.le (HDiv.hDiv (s.sum fun i => f i) s …
  -/
  obtain ⟨x, hx₁, hx₂⟩ := h
  suffices h : b = (∑ i ∈ s, f i) / s.card by
    simp_rw [← h]
    apply hb
  /-
    case neg.intro.intro
    α : Type u_1
    s : Finset α
    f : α → Nat
    b : Nat
    hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
    x : α
    hx₁ : Membership.mem s x
    hx₂ : Ne (f x) (HAdd.hAdd b 1)
    ⊢ Eq b (HDiv.hDiv (s.sum fun i => f i) s.card)
  -/
  symm
  refine
    Nat.div_eq_of_lt_le (le_trans (by simp [mul_comm]) (sum_le_sum fun a ha => (hb a ha).1))
      ((sum_lt_sum (fun a ha => (hb a ha).2) ⟨_, hx₁, (hb _ hx₁).2.lt_of_ne hx₂⟩).trans_le ?_)
  /-
    case neg.intro.intro
    α : Type u_1
    s : Finset α
    f : α → Nat
    b : Nat
    hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
    x : α
    hx₁ : Membership.mem s x
    hx₂ : Ne (f x) (HAdd.hAdd b 1)
    ⊢ LE.le (s.sum fun i => HAdd.hAdd b 1) (HMul.hMul (HAdd.hAdd b 1) s.card)
  -/
  rw [mul_comm, sum_const_nat]
  /-
    case neg.intro.intro
    α : Type u_1
    s : Finset α
    f : α → Nat
    b : Nat
    hb : ∀ (a : α), Membership.mem (↑s) a → And (LE.le b (f a)) (LE.le (f a) (HAdd …
    x : α
    hx₁ : Membership.mem s x
    hx₂ : Ne (f x) (HAdd.hAdd b 1)
    ⊢ ∀ (x : α), Membership.mem s x → Eq (HAdd.hAdd b 1) (HAdd.hAdd b 1)
  -/
  exact fun _ _ => rfl
  /-
    🎉 no goals
  -/


theorem EquitableOn.le (h : EquitableOn (s : Set α) f) (ha : a ∈ s) :
    (∑ i ∈ s, f i) / s.card ≤ f a :=
  (equitableOn_iff_le_le_add_one.1 h a ha).1


theorem EquitableOn.le_add_one (h : EquitableOn (s : Set α) f) (ha : a ∈ s) :
    f a ≤ (∑ i ∈ s, f i) / s.card + 1 :=
  (equitableOn_iff_le_le_add_one.1 h a ha).2


theorem equitableOn_iff :
    EquitableOn (s : Set α) f ↔
      ∀ a ∈ s, f a = (∑ i ∈ s, f i) / s.card ∨ f a = (∑ i ∈ s, f i) / s.card + 1 := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Nat
    ⊢ Iff ((↑s).EquitableOn f) (∀ (a : α), Membership.mem s a → Or (Eq (f a) (HDiv …
  -/
  simp_rw [equitableOn_iff_le_le_add_one, Nat.le_and_le_add_one_iff]
  /-
    🎉 no goals
  -/


