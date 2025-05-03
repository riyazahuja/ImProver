/-- `divisors n` is the `Finset` of divisors of `n`. As a special case, `divisors 0 = ∅`. -/
def divisors : Finset ℕ := {d ∈ Ico 1 (n + 1) | d ∣ n}


/-- `properDivisors n` is the `Finset` of divisors of `n`, other than `n`.
  As a special case, `properDivisors 0 = ∅`. -/
def properDivisors : Finset ℕ := {d ∈ Ico 1 n | d ∣ n}


/-- `divisorsAntidiagonal n` is the `Finset` of pairs `(x,y)` such that `x * y = n`.
  As a special case, `divisorsAntidiagonal 0 = ∅`. -/
def divisorsAntidiagonal : Finset (ℕ × ℕ) :=
  {x ∈ Ico 1 (n + 1) ×ˢ Ico 1 (n + 1) | x.fst * x.snd = n}


@[simp]
theorem filter_dvd_eq_divisors (h : n ≠ 0) : {d ∈ range n.succ | d ∣ n} = n.divisors := by
  /-
    n : Nat
    h : Ne n 0
    ⊢ Eq (Finset.filter (fun d => Dvd.dvd d n) (Finset.range n.succ)) n.divisors
  -/
  ext
  /-
    case h
    n : Nat
    h : Ne n 0
    a✝ : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun d => Dvd.dvd d n) (Finset.range n.su …
  -/
  simp only [divisors, mem_filter, mem_range, mem_Ico, and_congr_left_iff, iff_and_self]
  /-
    case h
    n : Nat
    h : Ne n 0
    a✝ : Nat
    ⊢ Dvd.dvd a✝ n → LT.lt a✝ n.succ → LE.le 1 a✝
  -/
  exact fun ha _ => succ_le_iff.mpr (pos_of_dvd_of_pos ha h.bot_lt)
  /-
    🎉 no goals
  -/


@[simp]
theorem filter_dvd_eq_properDivisors (h : n ≠ 0) : {d ∈ range n | d ∣ n} = n.properDivisors := by
  /-
    n : Nat
    h : Ne n 0
    ⊢ Eq (Finset.filter (fun d => Dvd.dvd d n) (Finset.range n)) n.properDivisors
  -/
  ext
  /-
    case h
    n : Nat
    h : Ne n 0
    a✝ : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun d => Dvd.dvd d n) (Finset.range n))  …
  -/
  simp only [properDivisors, mem_filter, mem_range, mem_Ico, and_congr_left_iff, iff_and_self]
  /-
    case h
    n : Nat
    h : Ne n 0
    a✝ : Nat
    ⊢ Dvd.dvd a✝ n → LT.lt a✝ n → LE.le 1 a✝
  -/
  exact fun ha _ => succ_le_iff.mpr (pos_of_dvd_of_pos ha h.bot_lt)
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    n : Nat
                                                                    ⊢ Not (Membership.mem n.properDivisors n)
                                                                  -/
theorem properDivisors.not_self_mem : ¬n ∈ properDivisors n := by simp [properDivisors]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem mem_properDivisors {m : ℕ} : n ∈ properDivisors m ↔ n ∣ m ∧ n < m := by
  /-
    n m : Nat
    ⊢ Iff (Membership.mem m.properDivisors n) (And (Dvd.dvd n m) (LT.lt n m))
  -/
  rcases eq_or_ne m 0 with (rfl | hm); · simp [properDivisors]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    n m : Nat
    hm : Ne m 0
    ⊢ Iff (Membership.mem m.properDivisors n) (And (Dvd.dvd n m) (LT.lt n m))
  -/
  simp only [and_comm, ← filter_dvd_eq_properDivisors hm, mem_filter, mem_range]
  /-
    🎉 no goals
  -/


theorem insert_self_properDivisors (h : n ≠ 0) : insert n (properDivisors n) = divisors n := by
  rw [divisors, properDivisors, Ico_succ_right_eq_insert_Ico (one_le_iff_ne_zero.2 h),
    Finset.filter_insert, if_pos (dvd_refl n)]


theorem cons_self_properDivisors (h : n ≠ 0) :
    cons n (properDivisors n) properDivisors.not_self_mem = divisors n := by
  /-
    n : Nat
    h : Ne n 0
    ⊢ Eq (Finset.cons n n.properDivisors ⋯) n.divisors
  -/
  rw [cons_eq_insert, insert_self_properDivisors h]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_divisors {m : ℕ} : n ∈ divisors m ↔ n ∣ m ∧ m ≠ 0 := by
  /-
    n m : Nat
    ⊢ Iff (Membership.mem m.divisors n) (And (Dvd.dvd n m) (Ne m 0))
  -/
  rcases eq_or_ne m 0 with (rfl | hm); · simp [divisors]
                                         /-
                                           🎉 no goals
                                         -/
  simp only [hm, Ne, not_false_iff, and_true, ← filter_dvd_eq_divisors hm, mem_filter,
    mem_range, and_iff_right_iff_imp, Nat.lt_succ_iff]
  /-
    case inr
    n m : Nat
    hm : Ne m 0
    ⊢ Dvd.dvd n m → LE.le n m
  -/
  exact le_of_dvd hm.bot_lt
  /-
    🎉 no goals
  -/


                                                        /-
                                                          n : Nat
                                                          ⊢ Iff (Membership.mem n.divisors 1) (Ne n 0)
                                                        -/
theorem one_mem_divisors : 1 ∈ divisors n ↔ n ≠ 0 := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem mem_divisors_self (n : ℕ) (h : n ≠ 0) : n ∈ n.divisors :=
  mem_divisors.2 ⟨dvd_rfl, h⟩


theorem dvd_of_mem_divisors {m : ℕ} (h : n ∈ divisors m) : n ∣ m := by
  /-
    n m : Nat
    h : Membership.mem m.divisors n
    ⊢ Dvd.dvd n m
  -/
  cases m
    /-
      case zero
      n : Nat
      h : Membership.mem (Nat.divisors 0) n
      ⊢ Dvd.dvd n 0
    -/
  · apply dvd_zero
    /-
      🎉 no goals
    -/
    /-
      case succ
      n n✝ : Nat
      h : Membership.mem (HAdd.hAdd n✝ 1).divisors n
      ⊢ Dvd.dvd n (HAdd.hAdd n✝ 1)
    -/
  · simp [mem_divisors.1 h]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_divisorsAntidiagonal {x : ℕ × ℕ} :
    x ∈ divisorsAntidiagonal n ↔ x.fst * x.snd = n ∧ n ≠ 0 := by
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (Membership.mem n.divisorsAntidiagonal x) (And (Eq (HMul.hMul x.1 x.2) n …
  -/
  simp only [divisorsAntidiagonal, Finset.mem_Ico, Ne, Finset.mem_filter, Finset.mem_product]
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (And (And (And (LE.le 1 x.1) (LT.lt x.1 (HAdd.hAdd n 1))) (And (LE.le 1  …
  -/
  rw [and_comm]
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (And (Eq (HMul.hMul x.1 x.2) n) (And (And (LE.le 1 x.1) (LT.lt x.1 (HAdd …
  -/
  apply and_congr_right
  /-
    case h
    n : Nat
    x : Prod Nat Nat
    ⊢ Eq (HMul.hMul x.1 x.2) n → Iff (And (And (LE.le 1 x.1) (LT.lt x.1 (HAdd.hAdd …
  -/
  rintro rfl
  /-
    case h
    x : Prod Nat Nat
    ⊢ Iff (And (And (LE.le 1 x.1) (LT.lt x.1 (HAdd.hAdd (HMul.hMul x.1 x.2) 1))) ( …
  -/
  constructor <;> intro h
    /-
      case h.mp
      x : Prod Nat Nat
      h : And (And (LE.le 1 x.1) (LT.lt x.1 (HAdd.hAdd (HMul.hMul x.1 x.2) 1))) (And …
      ⊢ Not (Eq (HMul.hMul x.1 x.2) 0)
    -/
  · contrapose! h
    /-
      case h.mp
      x : Prod Nat Nat
      h : Eq (HMul.hMul x.1 x.2) 0
      ⊢ And (LE.le 1 x.1) (LT.lt x.1 (HAdd.hAdd (HMul.hMul x.1 x.2) 1)) → LE.le 1 x. …
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      x : Prod Nat Nat
      h : Not (Eq (HMul.hMul x.1 x.2) 0)
      ⊢ And (And (LE.le 1 x.1) (LT.lt x.1 (HAdd.hAdd (HMul.hMul x.1 x.2) 1))) (And ( …
    -/
  · rw [Nat.lt_add_one_iff, Nat.lt_add_one_iff]
    /-
      case h.mpr
      x : Prod Nat Nat
      h : Not (Eq (HMul.hMul x.1 x.2) 0)
      ⊢ And (And (LE.le 1 x.1) (LE.le x.1 (HMul.hMul x.1 x.2))) (And (LE.le 1 x.2) ( …
    -/
    rw [mul_eq_zero, not_or] at h
    simp only [succ_le_of_lt (Nat.pos_of_ne_zero h.1), succ_le_of_lt (Nat.pos_of_ne_zero h.2),
      true_and]
    exact
      ⟨Nat.le_mul_of_pos_right _ (Nat.pos_of_ne_zero h.2),
        Nat.le_mul_of_pos_left _ (Nat.pos_of_ne_zero h.1)⟩


lemma ne_zero_of_mem_divisorsAntidiagonal {p : ℕ × ℕ} (hp : p ∈ n.divisorsAntidiagonal) :
    p.1 ≠ 0 ∧ p.2 ≠ 0 := by
  /-
    n : Nat
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    ⊢ And (Ne p.1 0) (Ne p.2 0)
  -/
  obtain ⟨hp₁, hp₂⟩ := Nat.mem_divisorsAntidiagonal.mp hp
  /-
    case intro
    n : Nat
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    hp₁ : Eq (HMul.hMul p.1 p.2) n
    hp₂ : Ne n 0
    ⊢ And (Ne p.1 0) (Ne p.2 0)
  -/
  exact mul_ne_zero_iff.mp (hp₁.symm ▸ hp₂)
  /-
    🎉 no goals
  -/


lemma left_ne_zero_of_mem_divisorsAntidiagonal {p : ℕ × ℕ} (hp : p ∈ n.divisorsAntidiagonal) :
    p.1 ≠ 0 :=
  (ne_zero_of_mem_divisorsAntidiagonal hp).1


lemma right_ne_zero_of_mem_divisorsAntidiagonal {p : ℕ × ℕ} (hp : p ∈ n.divisorsAntidiagonal) :
    p.2 ≠ 0 :=
  (ne_zero_of_mem_divisorsAntidiagonal hp).2


theorem divisor_le {m : ℕ} : n ∈ divisors m → n ≤ m := by
  /-
    n m : Nat
    ⊢ Membership.mem m.divisors n → LE.le n m
  -/
  cases' m with m
    /-
      case zero
      n : Nat
      ⊢ Membership.mem (Nat.divisors 0) n → LE.le n 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n m : Nat
      ⊢ Membership.mem (HAdd.hAdd m 1).divisors n → LE.le n (HAdd.hAdd m 1)
    -/
  · simp only [mem_divisors, Nat.succ_ne_zero m, and_true, Ne, not_false_iff]
    /-
      case succ
      n m : Nat
      ⊢ Dvd.dvd n (HAdd.hAdd m 1) → LE.le n (HAdd.hAdd m 1)
    -/
    exact Nat.le_of_dvd (Nat.succ_pos m)
    /-
      🎉 no goals
    -/


theorem divisors_subset_of_dvd {m : ℕ} (hzero : n ≠ 0) (h : m ∣ n) : divisors m ⊆ divisors n :=
  Finset.subset_iff.2 fun _x hx => Nat.mem_divisors.mpr ⟨(Nat.mem_divisors.mp hx).1.trans h, hzero⟩


theorem card_divisors_le_self (n : ℕ) : #n.divisors ≤ n := calc
  _ ≤ #(Ico 1 (n + 1)) := by
    /-
      n : Nat
      ⊢ LE.le n.divisors.card (Finset.Ico 1 (HAdd.hAdd n 1)).card
    -/
    apply card_le_card
    /-
      case a
      n : Nat
      ⊢ HasSubset.Subset n.divisors (Finset.Ico 1 (HAdd.hAdd n 1))
    -/
    simp only [divisors, filter_subset]
    /-
      🎉 no goals
    -/
              /-
                n : Nat
                ⊢ Eq (Finset.Ico 1 (HAdd.hAdd n 1)).card n
              -/
  _ = n := by rw [card_Ico, add_tsub_cancel_right]
              /-
                🎉 no goals
              -/


theorem divisors_subset_properDivisors {m : ℕ} (hzero : n ≠ 0) (h : m ∣ n) (hdiff : m ≠ n) :
    divisors m ⊆ properDivisors n := by
  /-
    n m : Nat
    hzero : Ne n 0
    h : Dvd.dvd m n
    hdiff : Ne m n
    ⊢ HasSubset.Subset m.divisors n.properDivisors
  -/
  apply Finset.subset_iff.2
  /-
    n m : Nat
    hzero : Ne n 0
    h : Dvd.dvd m n
    hdiff : Ne m n
    ⊢ ∀ ⦃x : Nat⦄, Membership.mem m.divisors x → Membership.mem n.properDivisors x
  -/
  intro x hx
  exact
    Nat.mem_properDivisors.2
      ⟨(Nat.mem_divisors.1 hx).1.trans h,
        lt_of_le_of_lt (divisor_le hx)
          (lt_of_le_of_ne (divisor_le (Nat.mem_divisors.2 ⟨h, hzero⟩)) hdiff)⟩


lemma divisors_filter_dvd_of_dvd {n m : ℕ} (hn : n ≠ 0) (hm : m ∣ n) :
    {d ∈ n.divisors | d ∣ m} = m.divisors := by
  /-
    n m : Nat
    hn : Ne n 0
    hm : Dvd.dvd m n
    ⊢ Eq (Finset.filter (fun d => Dvd.dvd d m) n.divisors) m.divisors
  -/
  ext k
  /-
    case h
    n m : Nat
    hn : Ne n 0
    hm : Dvd.dvd m n
    k : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun d => Dvd.dvd d m) n.divisors) k) (Me …
  -/
  simp_rw [mem_filter, mem_divisors]
  /-
    case h
    n m : Nat
    hn : Ne n 0
    hm : Dvd.dvd m n
    k : Nat
    ⊢ Iff (And (And (Dvd.dvd k n) (Ne n 0)) (Dvd.dvd k m)) (And (Dvd.dvd k m) (Ne  …
  -/
  exact ⟨fun ⟨_, hkm⟩ ↦ ⟨hkm, ne_zero_of_dvd_ne_zero hn hm⟩, fun ⟨hk, _⟩ ↦ ⟨⟨hk.trans hm, hn⟩, hk⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem divisors_zero : divisors 0 = ∅ := by
  /-
    ⊢ Eq (Nat.divisors 0) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    a✝ : Nat
    ⊢ Iff (Membership.mem (Nat.divisors 0) a✝) (Membership.mem EmptyCollection.emp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem properDivisors_zero : properDivisors 0 = ∅ := by
  /-
    ⊢ Eq (Nat.properDivisors 0) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    a✝ : Nat
    ⊢ Iff (Membership.mem (Nat.properDivisors 0) a✝) (Membership.mem EmptyCollecti …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma nonempty_divisors : (divisors n).Nonempty ↔ n ≠ 0 :=
                       /-
                         n : Nat
                         x✝ : n.divisors.Nonempty
                         hn : Eq n 0
                         m : Nat
                         hm : Membership.mem n.divisors m
                         ⊢ False
                       -/
  ⟨fun ⟨m, hm⟩ hn ↦ by simp [hn] at hm, fun hn ↦ ⟨1, one_mem_divisors.2 hn⟩⟩
                       /-
                         🎉 no goals
                       -/


@[simp]
lemma divisors_eq_empty : divisors n = ∅ ↔ n = 0 :=
  not_nonempty_iff_eq_empty.symm.trans nonempty_divisors.not_left


theorem properDivisors_subset_divisors : properDivisors n ⊆ divisors n :=
  filter_subset_filter _ <| Ico_subset_Ico_right n.le_succ


@[simp]
theorem divisors_one : divisors 1 = {1} := by
  /-
    ⊢ Eq (Nat.divisors 1) (Singleton.singleton 1)
  -/
  ext
  /-
    case h
    a✝ : Nat
    ⊢ Iff (Membership.mem (Nat.divisors 1) a✝) (Membership.mem (Singleton.singleto …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          ⊢ Eq (Nat.properDivisors 1) EmptyCollection.emptyCollection
                                                        -/
theorem properDivisors_one : properDivisors 1 = ∅ := by rw [properDivisors, Ico_self, filter_empty]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem pos_of_mem_divisors {m : ℕ} (h : m ∈ n.divisors) : 0 < m := by
  /-
    n m : Nat
    h : Membership.mem n.divisors m
    ⊢ LT.lt 0 m
  -/
  cases m
    /-
      case zero
      n : Nat
      h : Membership.mem n.divisors 0
      ⊢ LT.lt 0 0
    -/
  · rw [mem_divisors, zero_dvd_iff (a := n)] at h
    /-
      case zero
      n : Nat
      h : And (Eq n 0) (Ne n 0)
      ⊢ LT.lt 0 0
    -/
    cases h.2 h.1
    /-
      🎉 no goals
    -/
  /-
    case succ
    n n✝ : Nat
    h : Membership.mem n.divisors (HAdd.hAdd n✝ 1)
    ⊢ LT.lt 0 (HAdd.hAdd n✝ 1)
  -/
  apply Nat.succ_pos
  /-
    🎉 no goals
  -/


theorem pos_of_mem_properDivisors {m : ℕ} (h : m ∈ n.properDivisors) : 0 < m :=
  pos_of_mem_divisors (properDivisors_subset_divisors h)


theorem one_mem_properDivisors_iff_one_lt : 1 ∈ n.properDivisors ↔ 1 < n := by
  /-
    n : Nat
    ⊢ Iff (Membership.mem n.properDivisors 1) (LT.lt 1 n)
  -/
  rw [mem_properDivisors, and_iff_right (one_dvd _)]
  /-
    🎉 no goals
  -/


@[simp]
lemma sup_divisors_id (n : ℕ) : n.divisors.sup id = n := by
  /-
    n : Nat
    ⊢ Eq (n.divisors.sup id) n
  -/
  refine le_antisymm (Finset.sup_le fun _ ↦ divisor_le) ?_
  /-
    n : Nat
    ⊢ LE.le n (n.divisors.sup id)
  -/
  rcases Decidable.eq_or_ne n 0 with rfl | hn
    /-
      case inl
      ⊢ LE.le 0 ((Nat.divisors 0).sup id)
    -/
  · apply zero_le
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : Ne n 0
      ⊢ LE.le n (n.divisors.sup id)
    -/
  · exact Finset.le_sup (f := id) <| mem_divisors_self n hn
    /-
      🎉 no goals
    -/


lemma one_lt_of_mem_properDivisors {m n : ℕ} (h : m ∈ n.properDivisors) : 1 < n :=
  lt_of_le_of_lt (pos_of_mem_properDivisors h) (mem_properDivisors.1 h).2


lemma one_lt_div_of_mem_properDivisors {m n : ℕ} (h : m ∈ n.properDivisors) :
    1 < n / m := by
  /-
    m n : Nat
    h : Membership.mem n.properDivisors m
    ⊢ LT.lt 1 (HDiv.hDiv n m)
  -/
  obtain ⟨h_dvd, h_lt⟩ := mem_properDivisors.mp h
  /-
    case intro
    m n : Nat
    h : Membership.mem n.properDivisors m
    h_dvd : Dvd.dvd m n
    h_lt : LT.lt m n
    ⊢ LT.lt 1 (HDiv.hDiv n m)
  -/
  rwa [Nat.lt_div_iff_mul_lt' h_dvd, mul_one]
  /-
    🎉 no goals
  -/


/-- See also `Nat.mem_properDivisors`. -/
lemma mem_properDivisors_iff_exists {m n : ℕ} (hn : n ≠ 0) :
    m ∈ n.properDivisors ↔ ∃ k > 1, n = m * k := by
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Iff (Membership.mem n.properDivisors m) (Exists fun k => And (GT.gt k 1) (Eq …
  -/
  refine ⟨fun h ↦ ⟨n / m, one_lt_div_of_mem_properDivisors h, ?_⟩, ?_⟩
    /-
      case refine_1
      m n : Nat
      hn : Ne n 0
      h : Membership.mem n.properDivisors m
      ⊢ Eq n (HMul.hMul m (HDiv.hDiv n m))
    -/
  · exact (Nat.mul_div_cancel' (mem_properDivisors.mp h).1).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      m n : Nat
      hn : Ne n 0
      ⊢ (Exists fun k => And (GT.gt k 1) (Eq n (HMul.hMul m k))) → Membership.mem n. …
    -/
  · rintro ⟨k, hk, rfl⟩
    /-
      case refine_2.intro.intro
      m k : Nat
      hk : GT.gt k 1
      hn : Ne (HMul.hMul m k) 0
      ⊢ Membership.mem (HMul.hMul m k).properDivisors m
    -/
    rw [mul_ne_zero_iff] at hn
    /-
      case refine_2.intro.intro
      m k : Nat
      hk : GT.gt k 1
      hn : And (Ne m 0) (Ne k 0)
      ⊢ Membership.mem (HMul.hMul m k).properDivisors m
    -/
    exact mem_properDivisors.mpr ⟨⟨k, rfl⟩, lt_mul_of_one_lt_right (Nat.pos_of_ne_zero hn.1) hk⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma nonempty_properDivisors : n.properDivisors.Nonempty ↔ 1 < n :=
  ⟨fun ⟨_m, hm⟩ ↦ one_lt_of_mem_properDivisors hm, fun hn ↦
    ⟨1, one_mem_properDivisors_iff_one_lt.2 hn⟩⟩


@[simp]
lemma properDivisors_eq_empty : n.properDivisors = ∅ ↔ n ≤ 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.properDivisors EmptyCollection.emptyCollection) (LE.le n 1)
  -/
  rw [← not_nonempty_iff_eq_empty, nonempty_properDivisors, not_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem divisorsAntidiagonal_zero : divisorsAntidiagonal 0 = ∅ := by
  /-
    ⊢ Eq (Nat.divisorsAntidiagonal 0) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    a✝ : Prod Nat Nat
    ⊢ Iff (Membership.mem (Nat.divisorsAntidiagonal 0) a✝) (Membership.mem EmptyCo …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem divisorsAntidiagonal_one : divisorsAntidiagonal 1 = {(1, 1)} := by
  /-
    ⊢ Eq (Nat.divisorsAntidiagonal 1) (Singleton.singleton { fst := 1, snd := 1 })
  -/
  ext
  /-
    case h
    a✝ : Prod Nat Nat
    ⊢ Iff (Membership.mem (Nat.divisorsAntidiagonal 1) a✝) (Membership.mem (Single …
  -/
  simp [mul_eq_one, Prod.ext_iff]
  /-
    🎉 no goals
  -/

/- Porting note: simpnf linter; added aux lemma below
Left-hand side simplifies from
  Prod.swap x ∈ Nat.divisorsAntidiagonal n
to
  x.snd * x.fst = n ∧ ¬n = 0-/
-- @[simp]

theorem swap_mem_divisorsAntidiagonal {x : ℕ × ℕ} :
    x.swap ∈ divisorsAntidiagonal n ↔ x ∈ divisorsAntidiagonal n := by
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (Membership.mem n.divisorsAntidiagonal x.swap) (Membership.mem n.divisor …
  -/
  rw [mem_divisorsAntidiagonal, mem_divisorsAntidiagonal, mul_comm, Prod.swap]
  /-
    🎉 no goals
  -/

-- Porting note: added below thm to replace the simp from the previous thm

@[simp]
theorem swap_mem_divisorsAntidiagonal_aux {x : ℕ × ℕ} :
    x.snd * x.fst = n ∧ ¬n = 0 ↔ x ∈ divisorsAntidiagonal n := by
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (And (Eq (HMul.hMul x.2 x.1) n) (Not (Eq n 0))) (Membership.mem n.diviso …
  -/
  rw [mem_divisorsAntidiagonal, mul_comm]
  /-
    🎉 no goals
  -/


theorem fst_mem_divisors_of_mem_antidiagonal {x : ℕ × ℕ} (h : x ∈ divisorsAntidiagonal n) :
    x.fst ∈ divisors n := by
  /-
    n : Nat
    x : Prod Nat Nat
    h : Membership.mem n.divisorsAntidiagonal x
    ⊢ Membership.mem n.divisors x.1
  -/
  rw [mem_divisorsAntidiagonal] at h
  /-
    n : Nat
    x : Prod Nat Nat
    h : And (Eq (HMul.hMul x.1 x.2) n) (Ne n 0)
    ⊢ Membership.mem n.divisors x.1
  -/
  simp [Dvd.intro _ h.1, h.2]
  /-
    🎉 no goals
  -/


theorem snd_mem_divisors_of_mem_antidiagonal {x : ℕ × ℕ} (h : x ∈ divisorsAntidiagonal n) :
    x.snd ∈ divisors n := by
  /-
    n : Nat
    x : Prod Nat Nat
    h : Membership.mem n.divisorsAntidiagonal x
    ⊢ Membership.mem n.divisors x.2
  -/
  rw [mem_divisorsAntidiagonal] at h
  /-
    n : Nat
    x : Prod Nat Nat
    h : And (Eq (HMul.hMul x.1 x.2) n) (Ne n 0)
    ⊢ Membership.mem n.divisors x.2
  -/
  simp [Dvd.intro_left _ h.1, h.2]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_swap_divisorsAntidiagonal :
    (divisorsAntidiagonal n).map (Equiv.prodComm _ _).toEmbedding = divisorsAntidiagonal n := by
  rw [← coe_inj, coe_map, Equiv.coe_toEmbedding, Equiv.coe_prodComm,
    Set.image_swap_eq_preimage_swap]
  /-
    n : Nat
    ⊢ Eq (Set.preimage Prod.swap ↑n.divisorsAntidiagonal) ↑n.divisorsAntidiagonal
  -/
  ext
  /-
    case h
    n : Nat
    x✝ : Prod Nat Nat
    ⊢ Iff (Membership.mem (Set.preimage Prod.swap ↑n.divisorsAntidiagonal) x✝) (Me …
  -/
  exact swap_mem_divisorsAntidiagonal
  /-
    🎉 no goals
  -/


@[simp]
theorem image_fst_divisorsAntidiagonal : (divisorsAntidiagonal n).image Prod.fst = divisors n := by
  /-
    n : Nat
    ⊢ Eq (Finset.image Prod.fst n.divisorsAntidiagonal) n.divisors
  -/
  ext
  /-
    case h
    n a✝ : Nat
    ⊢ Iff (Membership.mem (Finset.image Prod.fst n.divisorsAntidiagonal) a✝) (Memb …
  -/
  simp [Dvd.dvd, @eq_comm _ n (_ * _)]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_snd_divisorsAntidiagonal : (divisorsAntidiagonal n).image Prod.snd = divisors n := by
  /-
    n : Nat
    ⊢ Eq (Finset.image Prod.snd n.divisorsAntidiagonal) n.divisors
  -/
  rw [← map_swap_divisorsAntidiagonal, map_eq_image, image_image]
  /-
    n : Nat
    ⊢ Eq (Finset.image (Function.comp Prod.snd ⇑(Equiv.prodComm Nat Nat).toEmbeddi …
  -/
  exact image_fst_divisorsAntidiagonal
  /-
    🎉 no goals
  -/


theorem map_div_right_divisors :
    n.divisors.map ⟨fun d => (d, n / d), fun _ _ => congr_arg Prod.fst⟩ =
      n.divisorsAntidiagonal := by
  /-
    n : Nat
    ⊢ Eq (Finset.map { toFun := fun d => { fst := d, snd := HDiv.hDiv n d }, inj'  …
  -/
  ext ⟨d, nd⟩
  simp only [mem_map, mem_divisorsAntidiagonal, Function.Embedding.coeFn_mk, mem_divisors,
    Prod.ext_iff, exists_prop, and_left_comm, exists_eq_left]
  /-
    case h.mk
    n d nd : Nat
    ⊢ Iff (And (And (Dvd.dvd d n) (Ne n 0)) (Eq (HDiv.hDiv n d) nd)) (And (Eq (HMu …
  -/
  constructor
    /-
      case h.mk.mp
      n d nd : Nat
      ⊢ And (And (Dvd.dvd d n) (Ne n 0)) (Eq (HDiv.hDiv n d) nd) → And (Eq (HMul.hMu …
    -/
  · rintro ⟨⟨⟨k, rfl⟩, hn⟩, rfl⟩
    /-
      case h.mk.mp.intro.intro.intro
      d k : Nat
      hn : Ne (HMul.hMul d k) 0
      ⊢ And (Eq (HMul.hMul d (HDiv.hDiv (HMul.hMul d k) d)) (HMul.hMul d k)) (Ne (HM …
    -/
    rw [Nat.mul_div_cancel_left _ (left_ne_zero_of_mul hn).bot_lt]
    /-
      case h.mk.mp.intro.intro.intro
      d k : Nat
      hn : Ne (HMul.hMul d k) 0
      ⊢ And (Eq (HMul.hMul d k) (HMul.hMul d k)) (Ne (HMul.hMul d k) 0)
    -/
    exact ⟨rfl, hn⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mpr
      n d nd : Nat
      ⊢ And (Eq (HMul.hMul d nd) n) (Ne n 0) → And (And (Dvd.dvd d n) (Ne n 0)) (Eq  …
    -/
  · rintro ⟨rfl, hn⟩
    /-
      case h.mk.mpr.intro
      d nd : Nat
      hn : Ne (HMul.hMul d nd) 0
      ⊢ And (And (Dvd.dvd d (HMul.hMul d nd)) (Ne (HMul.hMul d nd) 0)) (Eq (HDiv.hDi …
    -/
    exact ⟨⟨dvd_mul_right _ _, hn⟩, Nat.mul_div_cancel_left _ (left_ne_zero_of_mul hn).bot_lt⟩
    /-
      🎉 no goals
    -/


theorem map_div_left_divisors :
    n.divisors.map ⟨fun d => (n / d, d), fun _ _ => congr_arg Prod.snd⟩ =
      n.divisorsAntidiagonal := by
  /-
    n : Nat
    ⊢ Eq (Finset.map { toFun := fun d => { fst := HDiv.hDiv n d, snd := d }, inj'  …
  -/
  apply Finset.map_injective (Equiv.prodComm _ _).toEmbedding
  /-
    case a
    n : Nat
    ⊢ Eq (Finset.map (Equiv.prodComm Nat Nat).toEmbedding (Finset.map { toFun := f …
  -/
  ext
  /-
    case a.h
    n : Nat
    a✝ : Prod Nat Nat
    ⊢ Iff (Membership.mem (Finset.map (Equiv.prodComm Nat Nat).toEmbedding (Finset …
  -/
  rw [map_swap_divisorsAntidiagonal, ← map_div_right_divisors, Finset.map_map]
  /-
    case a.h
    n : Nat
    a✝ : Prod Nat Nat
    ⊢ Iff (Membership.mem (Finset.map ({ toFun := fun d => { fst := HDiv.hDiv n d, …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sum_divisors_eq_sum_properDivisors_add_self :
    ∑ i ∈ divisors n, i = (∑ i ∈ properDivisors n, i) + n := by
  /-
    n : Nat
    ⊢ Eq (n.divisors.sum fun i => i) (HAdd.hAdd (n.properDivisors.sum fun i => i) n)
  -/
  rcases Decidable.eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      ⊢ Eq ((Nat.divisors 0).sum fun i => i) (HAdd.hAdd ((Nat.properDivisors 0).sum  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : Ne n 0
      ⊢ Eq (n.divisors.sum fun i => i) (HAdd.hAdd (n.properDivisors.sum fun i => i) n)
    -/
  · rw [← cons_self_properDivisors hn, Finset.sum_cons, add_comm]
    /-
      🎉 no goals
    -/


/-- `n : ℕ` is perfect if and only the sum of the proper divisors of `n` is `n` and `n`
  is positive. -/
def Perfect (n : ℕ) : Prop :=
  ∑ i ∈ properDivisors n, i = n ∧ 0 < n


theorem perfect_iff_sum_properDivisors (h : 0 < n) : Perfect n ↔ ∑ i ∈ properDivisors n, i = n :=
  and_iff_left h


theorem perfect_iff_sum_divisors_eq_two_mul (h : 0 < n) :
    Perfect n ↔ ∑ i ∈ divisors n, i = 2 * n := by
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Iff n.Perfect (Eq (n.divisors.sum fun i => i) (HMul.hMul 2 n))
  -/
  rw [perfect_iff_sum_properDivisors h, sum_divisors_eq_sum_properDivisors_add_self, two_mul]
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Iff (Eq (n.properDivisors.sum fun i => i) n) (Eq (HAdd.hAdd (n.properDivisor …
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      h✝ : LT.lt 0 n
      h : Eq (n.properDivisors.sum fun i => i) n
      ⊢ Eq (HAdd.hAdd (n.properDivisors.sum fun i => i) n) (HAdd.hAdd n n)
    -/
  · rw [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      h✝ : LT.lt 0 n
      h : Eq (HAdd.hAdd (n.properDivisors.sum fun i => i) n) (HAdd.hAdd n n)
      ⊢ Eq (n.properDivisors.sum fun i => i) n
    -/
  · apply add_right_cancel h
    /-
      🎉 no goals
    -/


theorem mem_divisors_prime_pow {p : ℕ} (pp : p.Prime) (k : ℕ) {x : ℕ} :
    x ∈ divisors (p ^ k) ↔ ∃ j ≤ k, x = p ^ j := by
  /-
    p : Nat
    pp : Nat.Prime p
    k x : Nat
    ⊢ Iff (Membership.mem (HPow.hPow p k).divisors x) (Exists fun j => And (LE.le  …
  -/
  rw [mem_divisors, Nat.dvd_prime_pow pp, and_iff_left (ne_of_gt (pow_pos pp.pos k))]
  /-
    🎉 no goals
  -/


theorem Prime.divisors {p : ℕ} (pp : p.Prime) : divisors p = {1, p} := by
  /-
    p : Nat
    pp : Nat.Prime p
    ⊢ Eq p.divisors (Insert.insert 1 (Singleton.singleton p))
  -/
  ext
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    a✝ : Nat
    ⊢ Iff (Membership.mem p.divisors a✝) (Membership.mem (Insert.insert 1 (Singlet …
  -/
  rw [mem_divisors, dvd_prime pp, and_iff_left pp.ne_zero, Finset.mem_insert, Finset.mem_singleton]
  /-
    🎉 no goals
  -/


theorem Prime.properDivisors {p : ℕ} (pp : p.Prime) : properDivisors p = {1} := by
  rw [← erase_insert properDivisors.not_self_mem, insert_self_properDivisors pp.ne_zero,
    pp.divisors, pair_comm, erase_insert fun con => pp.ne_one (mem_singleton.1 con)]


theorem divisors_prime_pow {p : ℕ} (pp : p.Prime) (k : ℕ) :
    divisors (p ^ k) = (Finset.range (k + 1)).map ⟨(p ^ ·), Nat.pow_right_injective pp.two_le⟩ := by
  /-
    p : Nat
    pp : Nat.Prime p
    k : Nat
    ⊢ Eq (HPow.hPow p k).divisors (Finset.map { toFun := fun x => HPow.hPow p x, i …
  -/
  ext a
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k a : Nat
    ⊢ Iff (Membership.mem (HPow.hPow p k).divisors a) (Membership.mem (Finset.map  …
  -/
  rw [mem_divisors_prime_pow pp]
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k a : Nat
    ⊢ Iff (Exists fun j => And (LE.le j k) (Eq a (HPow.hPow p j))) (Membership.mem …
  -/
  simp [Nat.lt_succ, eq_comm]
  /-
    🎉 no goals
  -/


theorem divisors_injective : Function.Injective divisors :=
  Function.LeftInverse.injective sup_divisors_id


@[simp]
theorem divisors_inj {a b : ℕ} : a.divisors = b.divisors ↔ a = b :=
  divisors_injective.eq_iff


theorem eq_properDivisors_of_subset_of_sum_eq_sum {s : Finset ℕ} (hsub : s ⊆ n.properDivisors) :
    ((∑ x ∈ s, x) = ∑ x ∈ n.properDivisors, x) → s = n.properDivisors := by
  /-
    n : Nat
    s : Finset Nat
    hsub : HasSubset.Subset s n.properDivisors
    ⊢ Eq (s.sum fun x => x) (n.properDivisors.sum fun x => x) → Eq s n.properDivis …
  -/
  cases n
    /-
      case zero
      s : Finset Nat
      hsub : HasSubset.Subset s (Nat.properDivisors 0)
      ⊢ Eq (s.sum fun x => x) ((Nat.properDivisors 0).sum fun x => x) → Eq s (Nat.pr …
    -/
  · rw [properDivisors_zero, subset_empty] at hsub
    /-
      case zero
      s : Finset Nat
      hsub : Eq s EmptyCollection.emptyCollection
      ⊢ Eq (s.sum fun x => x) ((Nat.properDivisors 0).sum fun x => x) → Eq s (Nat.pr …
    -/
    simp [hsub]
    /-
      🎉 no goals
    -/
  classical
    rw [← sum_sdiff hsub]
    intro h
    apply Subset.antisymm hsub
    rw [← sdiff_eq_empty_iff_subset]
    contrapose h
    rw [← Ne, ← nonempty_iff_ne_empty] at h
    apply ne_of_lt
    rw [← zero_add (∑ x ∈ s, x), ← add_assoc, add_zero]
    apply add_lt_add_right
    have hlt :=
      sum_lt_sum_of_nonempty h fun x hx => pos_of_mem_properDivisors (sdiff_subset hx)
    simp only [sum_const_zero] at hlt
    apply hlt


theorem sum_properDivisors_dvd (h : (∑ x ∈ n.properDivisors, x) ∣ n) :
    ∑ x ∈ n.properDivisors, x = 1 ∨ ∑ x ∈ n.properDivisors, x = n := by
  /-
    n : Nat
    h : Dvd.dvd (n.properDivisors.sum fun x => x) n
    ⊢ Or (Eq (n.properDivisors.sum fun x => x) 1) (Eq (n.properDivisors.sum fun x  …
  -/
  cases' n with n
    /-
      case zero
      h : Dvd.dvd ((Nat.properDivisors 0).sum fun x => x) 0
      ⊢ Or (Eq ((Nat.properDivisors 0).sum fun x => x) 1) (Eq ((Nat.properDivisors 0 …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      h : Dvd.dvd ((HAdd.hAdd n 1).properDivisors.sum fun x => x) (HAdd.hAdd n 1)
      ⊢ Or (Eq ((HAdd.hAdd n 1).properDivisors.sum fun x => x) 1) (Eq ((HAdd.hAdd n  …
    -/
  · cases' n with n
      /-
        case succ.zero
        h : Dvd.dvd ((HAdd.hAdd 0 1).properDivisors.sum fun x => x) (HAdd.hAdd 0 1)
        ⊢ Or (Eq ((HAdd.hAdd 0 1).properDivisors.sum fun x => x) 1) (Eq ((HAdd.hAdd 0  …
      -/
    · simp at h
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        n : Nat
        h : Dvd.dvd ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) (HAd …
        ⊢ Or (Eq ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) 1) (Eq  …
      -/
    · rw [or_iff_not_imp_right]
      /-
        case succ.succ
        n : Nat
        h : Dvd.dvd ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) (HAd …
        ⊢ Not (Eq ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) (HAdd. …
      -/
      intro ne_n
      have hlt : ∑ x ∈ n.succ.succ.properDivisors, x < n.succ.succ :=
        lt_of_le_of_ne (Nat.le_of_dvd (Nat.succ_pos _) h) ne_n
      /-
        case succ.succ
        n : Nat
        h : Dvd.dvd ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) (HAd …
        ne_n : Not (Eq ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) ( …
        hlt : LT.lt (n.succ.succ.properDivisors.sum fun x => x) n.succ.succ
        ⊢ Eq ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) 1
      -/
      symm
      rw [← mem_singleton, eq_properDivisors_of_subset_of_sum_eq_sum (singleton_subset_iff.2
        (mem_properDivisors.2 ⟨h, hlt⟩)) (sum_singleton _ _), mem_properDivisors]
      /-
        case succ.succ
        n : Nat
        h : Dvd.dvd ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) (HAd …
        ne_n : Not (Eq ((HAdd.hAdd (HAdd.hAdd n 1) 1).properDivisors.sum fun x => x) ( …
        hlt : LT.lt (n.succ.succ.properDivisors.sum fun x => x) n.succ.succ
        ⊢ And (Dvd.dvd 1 (HAdd.hAdd (HAdd.hAdd n 1) 1)) (LT.lt 1 (HAdd.hAdd (HAdd.hAdd …
      -/
      exact ⟨one_dvd _, Nat.succ_lt_succ (Nat.succ_pos _)⟩
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem Prime.prod_properDivisors {α : Type*} [CommMonoid α] {p : ℕ} {f : ℕ → α} (h : p.Prime) :
                                            /-
                                              α : Type u_1
                                              inst✝ : CommMonoid α
                                              p : Nat
                                              f : Nat → α
                                              h : Nat.Prime p
                                              ⊢ Eq (p.properDivisors.prod fun x => f x) (f 1)
                                            -/
    ∏ x ∈ p.properDivisors, f x = f 1 := by simp [h.properDivisors]
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive (attr := simp)]
theorem Prime.prod_divisors {α : Type*} [CommMonoid α] {p : ℕ} {f : ℕ → α} (h : p.Prime) :
    ∏ x ∈ p.divisors, f x = f p * f 1 := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p : Nat
    f : Nat → α
    h : Nat.Prime p
    ⊢ Eq (p.divisors.prod fun x => f x) (HMul.hMul (f p) (f 1))
  -/
  rw [← cons_self_properDivisors h.ne_zero, prod_cons, h.prod_properDivisors]
  /-
    🎉 no goals
  -/


theorem properDivisors_eq_singleton_one_iff_prime : n.properDivisors = {1} ↔ n.Prime := by
  /-
    n : Nat
    ⊢ Iff (Eq n.properDivisors (Singleton.singleton 1)) (Nat.Prime n)
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      n : Nat
      ⊢ Eq n.properDivisors (Singleton.singleton 1) → Nat.Prime n
    -/
  · intro h
    /-
      case refine_1
      n : Nat
      h : Eq n.properDivisors (Singleton.singleton 1)
      ⊢ Nat.Prime n
    -/
    refine Nat.prime_def.mpr ⟨?_, fun m hdvd => ?_⟩
    · match n with
      | 0 => contradiction
      | 1 => contradiction
      | Nat.succ (Nat.succ n) => simp [succ_le_succ]
      /-
        case refine_1.refine_2
        n : Nat
        h : Eq n.properDivisors (Singleton.singleton 1)
        m : Nat
        hdvd : Dvd.dvd m n
        ⊢ Or (Eq m 1) (Eq m n)
      -/
    · rw [← mem_singleton, ← h, mem_properDivisors]
      /-
        case refine_1.refine_2
        n : Nat
        h : Eq n.properDivisors (Singleton.singleton 1)
        m : Nat
        hdvd : Dvd.dvd m n
        ⊢ Or (And (Dvd.dvd m n) (LT.lt m n)) (Eq m n)
      -/
      have := Nat.le_of_dvd ?_ hdvd
        /-
          case refine_1.refine_2.refine_2
          n : Nat
          h : Eq n.properDivisors (Singleton.singleton 1)
          m : Nat
          hdvd : Dvd.dvd m n
          this : LE.le m n
          ⊢ Or (And (Dvd.dvd m n) (LT.lt m n)) (Eq m n)
        -/
      · simpa [hdvd, this] using (le_iff_eq_or_lt.mp this).symm
        /-
          🎉 no goals
        -/
        /-
          case refine_1.refine_2.refine_1
          n : Nat
          h : Eq n.properDivisors (Singleton.singleton 1)
          m : Nat
          hdvd : Dvd.dvd m n
          ⊢ LT.lt 0 n
        -/
      · by_contra!
        /-
          case refine_1.refine_2.refine_1
          n : Nat
          h : Eq n.properDivisors (Singleton.singleton 1)
          m : Nat
          hdvd : Dvd.dvd m n
          this : LE.le n 0
          ⊢ False
        -/
        simp only [nonpos_iff_eq_zero.mp this, this] at h
        /-
          case refine_1.refine_2.refine_1
          n m : Nat
          hdvd : Dvd.dvd m n
          this : LE.le n 0
          h : Eq (Nat.properDivisors 0) (Singleton.singleton 1)
          ⊢ False
        -/
        contradiction
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      n : Nat
      ⊢ Nat.Prime n → Eq n.properDivisors (Singleton.singleton 1)
    -/
  · exact fun h => Prime.properDivisors h
    /-
      🎉 no goals
    -/


theorem sum_properDivisors_eq_one_iff_prime : ∑ x ∈ n.properDivisors, x = 1 ↔ n.Prime := by
  /-
    n : Nat
    ⊢ Iff (Eq (n.properDivisors.sum fun x => x) 1) (Nat.Prime n)
  -/
  cases' n with n
    /-
      case zero
      ⊢ Iff (Eq ((Nat.properDivisors 0).sum fun x => x) 1) (Nat.Prime 0)
    -/
  · simp [Nat.not_prime_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ⊢ Iff (Eq ((HAdd.hAdd n 1).properDivisors.sum fun x => x) 1) (Nat.Prime (HAdd. …
    -/
  · cases n
      /-
        case succ.zero
        ⊢ Iff (Eq ((HAdd.hAdd 0 1).properDivisors.sum fun x => x) 1) (Nat.Prime (HAdd. …
      -/
    · simp [Nat.not_prime_one]
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        n✝ : Nat
        ⊢ Iff (Eq ((HAdd.hAdd (HAdd.hAdd n✝ 1) 1).properDivisors.sum fun x => x) 1) (N …
      -/
    · rw [← properDivisors_eq_singleton_one_iff_prime]
      /-
        case succ.succ
        n✝ : Nat
        ⊢ Iff (Eq ((HAdd.hAdd (HAdd.hAdd n✝ 1) 1).properDivisors.sum fun x => x) 1) (E …
      -/
      refine ⟨fun h => ?_, fun h => h.symm ▸ sum_singleton _ _⟩
      /-
        case succ.succ
        n✝ : Nat
        h : Eq ((HAdd.hAdd (HAdd.hAdd n✝ 1) 1).properDivisors.sum fun x => x) 1
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1).properDivisors (Singleton.singleton 1)
      -/
      rw [@eq_comm (Finset ℕ) _ _]
      apply
        eq_properDivisors_of_subset_of_sum_eq_sum
          (singleton_subset_iff.2
            (one_mem_properDivisors_iff_one_lt.2 (succ_lt_succ (Nat.succ_pos _))))
          ((sum_singleton _ _).trans h.symm)


theorem mem_properDivisors_prime_pow {p : ℕ} (pp : p.Prime) (k : ℕ) {x : ℕ} :
    x ∈ properDivisors (p ^ k) ↔ ∃ (j : ℕ) (_ : j < k), x = p ^ j := by
  /-
    p : Nat
    pp : Nat.Prime p
    k x : Nat
    ⊢ Iff (Membership.mem (HPow.hPow p k).properDivisors x) (Exists fun j => Exist …
  -/
  rw [mem_properDivisors, Nat.dvd_prime_pow pp, ← exists_and_right]
  /-
    p : Nat
    pp : Nat.Prime p
    k x : Nat
    ⊢ Iff (Exists fun x_1 => And (And (LE.le x_1 k) (Eq x (HPow.hPow p x_1))) (LT. …
  -/
  simp only [exists_prop, and_assoc]
  /-
    p : Nat
    pp : Nat.Prime p
    k x : Nat
    ⊢ Iff (Exists fun x_1 => And (LE.le x_1 k) (And (Eq x (HPow.hPow p x_1)) (LT.l …
  -/
  apply exists_congr
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k x : Nat
    ⊢ ∀ (a : Nat), Iff (And (LE.le a k) (And (Eq x (HPow.hPow p a)) (LT.lt x (HPow …
  -/
  intro a
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k x a : Nat
    ⊢ Iff (And (LE.le a k) (And (Eq x (HPow.hPow p a)) (LT.lt x (HPow.hPow p k)))) …
  -/
  constructor <;> intro h
    /-
      case h.mp
      p : Nat
      pp : Nat.Prime p
      k x a : Nat
      h : And (LE.le a k) (And (Eq x (HPow.hPow p a)) (LT.lt x (HPow.hPow p k)))
      ⊢ And (LT.lt a k) (Eq x (HPow.hPow p a))
    -/
  · rcases h with ⟨_h_left, rfl, h_right⟩
    /-
      case h.mp.intro.intro
      p : Nat
      pp : Nat.Prime p
      k a : Nat
      _h_left : LE.le a k
      h_right : LT.lt (HPow.hPow p a) (HPow.hPow p k)
      ⊢ And (LT.lt a k) (Eq (HPow.hPow p a) (HPow.hPow p a))
    -/
    rw [Nat.pow_lt_pow_iff_right pp.one_lt] at h_right
    /-
      case h.mp.intro.intro
      p : Nat
      pp : Nat.Prime p
      k a : Nat
      _h_left : LE.le a k
      h_right : LT.lt a k
      ⊢ And (LT.lt a k) (Eq (HPow.hPow p a) (HPow.hPow p a))
    -/
    exact ⟨h_right, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      p : Nat
      pp : Nat.Prime p
      k x a : Nat
      h : And (LT.lt a k) (Eq x (HPow.hPow p a))
      ⊢ And (LE.le a k) (And (Eq x (HPow.hPow p a)) (LT.lt x (HPow.hPow p k)))
    -/
  · rcases h with ⟨h_left, rfl⟩
    /-
      case h.mpr.intro
      p : Nat
      pp : Nat.Prime p
      k a : Nat
      h_left : LT.lt a k
      ⊢ And (LE.le a k) (And (Eq (HPow.hPow p a) (HPow.hPow p a)) (LT.lt (HPow.hPow  …
    -/
    rw [Nat.pow_lt_pow_iff_right pp.one_lt]
    /-
      case h.mpr.intro
      p : Nat
      pp : Nat.Prime p
      k a : Nat
      h_left : LT.lt a k
      ⊢ And (LE.le a k) (And (Eq (HPow.hPow p a) (HPow.hPow p a)) (LT.lt a k))
    -/
    simp [h_left, le_of_lt]
    /-
      🎉 no goals
    -/


theorem properDivisors_prime_pow {p : ℕ} (pp : p.Prime) (k : ℕ) :
    properDivisors (p ^ k) = (Finset.range k).map ⟨(p ^ ·), Nat.pow_right_injective pp.two_le⟩ := by
  /-
    p : Nat
    pp : Nat.Prime p
    k : Nat
    ⊢ Eq (HPow.hPow p k).properDivisors (Finset.map { toFun := fun x => HPow.hPow  …
  -/
  ext a
  simp only [mem_properDivisors, Nat.isUnit_iff, mem_map, mem_range, Function.Embedding.coeFn_mk,
    pow_eq]
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k a : Nat
    ⊢ Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists fun  …
  -/
  have := mem_properDivisors_prime_pow pp k (x := a)
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k a : Nat
    this : Iff (Membership.mem (HPow.hPow p k).properDivisors a) (Exists fun j =>  …
    ⊢ Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists fun  …
  -/
  rw [mem_properDivisors] at this
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k a : Nat
    this : Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists …
    ⊢ Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists fun  …
  -/
  rw [this]
  /-
    case h
    p : Nat
    pp : Nat.Prime p
    k a : Nat
    this : Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists …
    ⊢ Iff (Exists fun j => Exists fun x => Eq a (HPow.hPow p j)) (Exists fun a_1 = …
  -/
  refine ⟨?_, ?_⟩
    /-
      case h.refine_1
      p : Nat
      pp : Nat.Prime p
      k a : Nat
      this : Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists …
      ⊢ (Exists fun j => Exists fun x => Eq a (HPow.hPow p j)) → Exists fun a_2 => A …
    -/
  · intro h; rcases h with ⟨j, hj, hap⟩; use j; tauto
                                                /-
                                                  🎉 no goals
                                                -/
    /-
      case h.refine_2
      p : Nat
      pp : Nat.Prime p
      k a : Nat
      this : Iff (And (Dvd.dvd a (HPow.hPow p k)) (LT.lt a (HPow.hPow p k))) (Exists …
      ⊢ (Exists fun a_1 => And (LT.lt a_1 k) (Eq (HPow.hPow p a_1) a)) → Exists fun  …
    -/
  · tauto
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem prod_properDivisors_prime_pow {α : Type*} [CommMonoid α] {k p : ℕ} {f : ℕ → α}
    (h : p.Prime) : (∏ x ∈ (p ^ k).properDivisors, f x) = ∏ x ∈ range k, f (p ^ x) := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    k p : Nat
    f : Nat → α
    h : Nat.Prime p
    ⊢ Eq ((HPow.hPow p k).properDivisors.prod fun x => f x) ((Finset.range k).prod …
  -/
  simp [h, properDivisors_prime_pow]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) sum_divisors_prime_pow]
theorem prod_divisors_prime_pow {α : Type*} [CommMonoid α] {k p : ℕ} {f : ℕ → α} (h : p.Prime) :
    (∏ x ∈ (p ^ k).divisors, f x) = ∏ x ∈ range (k + 1), f (p ^ x) := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    k p : Nat
    f : Nat → α
    h : Nat.Prime p
    ⊢ Eq ((HPow.hPow p k).divisors.prod fun x => f x) ((Finset.range (HAdd.hAdd k  …
  -/
  simp [h, divisors_prime_pow]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_divisorsAntidiagonal {M : Type*} [CommMonoid M] (f : ℕ → ℕ → M) {n : ℕ} :
    ∏ i ∈ n.divisorsAntidiagonal, f i.1 i.2 = ∏ i ∈ n.divisors, f i (n / i) := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    ⊢ Eq (n.divisorsAntidiagonal.prod fun i => f i.1 i.2) (n.divisors.prod fun i = …
  -/
  rw [← map_div_right_divisors, Finset.prod_map]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    ⊢ Eq (n.divisors.prod fun x => f ({ toFun := fun d => { fst := d, snd := HDiv. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_divisorsAntidiagonal' {M : Type*} [CommMonoid M] (f : ℕ → ℕ → M) {n : ℕ} :
    ∏ i ∈ n.divisorsAntidiagonal, f i.1 i.2 = ∏ i ∈ n.divisors, f (n / i) i := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    ⊢ Eq (n.divisorsAntidiagonal.prod fun i => f i.1 i.2) (n.divisors.prod fun i = …
  -/
  rw [← map_swap_divisorsAntidiagonal, Finset.prod_map]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    f : Nat → Nat → M
    n : Nat
    ⊢ Eq (n.divisorsAntidiagonal.prod fun x => f ((Equiv.prodComm Nat Nat).toEmbed …
  -/
  exact prod_divisorsAntidiagonal fun i j => f j i
  /-
    🎉 no goals
  -/


/-- The factors of `n` are the prime divisors -/
theorem primeFactors_eq_to_filter_divisors_prime (n : ℕ) :
    n.primeFactors = {p ∈ divisors n | p.Prime} := by
  /-
    n : Nat
    ⊢ Eq n.primeFactors (Finset.filter (fun p => Nat.Prime p) n.divisors)
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      ⊢ Eq (Nat.primeFactors 0) (Finset.filter (fun p => Nat.Prime p) (Nat.divisors  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : GT.gt n 0
      ⊢ Eq n.primeFactors (Finset.filter (fun p => Nat.Prime p) n.divisors)
    -/
  · ext q
    /-
      case inr.h
      n : Nat
      hn : GT.gt n 0
      q : Nat
      ⊢ Iff (Membership.mem n.primeFactors q) (Membership.mem (Finset.filter (fun p  …
    -/
    simpa [hn, hn.ne', mem_primeFactorsList] using and_comm
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-17")]
alias prime_divisors_eq_to_filter_divisors_prime := primeFactors_eq_to_filter_divisors_prime


lemma primeFactors_filter_dvd_of_dvd {m n : ℕ} (hn : n ≠ 0) (hmn : m ∣ n) :
    {p ∈ n.primeFactors | p ∣ m} = m.primeFactors := by
  simp_rw [primeFactors_eq_to_filter_divisors_prime, filter_comm,
    divisors_filter_dvd_of_dvd hn hmn]


@[deprecated (since := "2024-07-17")]
alias prime_divisors_filter_dvd_of_dvd := primeFactors_filter_dvd_of_dvd


@[simp]
theorem image_div_divisors_eq_divisors (n : ℕ) :
    image (fun x : ℕ => n / x) n.divisors = n.divisors := by
  /-
    n : Nat
    ⊢ Eq (Finset.image (fun x => HDiv.hDiv n x) n.divisors) n.divisors
  -/
  by_cases hn : n = 0
    /-
      case pos
      n : Nat
      hn : Eq n 0
      ⊢ Eq (Finset.image (fun x => HDiv.hDiv n x) n.divisors) n.divisors
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (Finset.image (fun x => HDiv.hDiv n x) n.divisors) n.divisors
  -/
  ext a
  /-
    case neg.h
    n : Nat
    hn : Not (Eq n 0)
    a : Nat
    ⊢ Iff (Membership.mem (Finset.image (fun x => HDiv.hDiv n x) n.divisors) a) (M …
  -/
  constructor
    /-
      case neg.h.mp
      n : Nat
      hn : Not (Eq n 0)
      a : Nat
      ⊢ Membership.mem (Finset.image (fun x => HDiv.hDiv n x) n.divisors) a → Member …
    -/
  · rw [mem_image]
    /-
      case neg.h.mp
      n : Nat
      hn : Not (Eq n 0)
      a : Nat
      ⊢ (Exists fun a_1 => And (Membership.mem n.divisors a_1) (Eq (HDiv.hDiv n a_1) …
    -/
    rintro ⟨x, hx1, hx2⟩
    /-
      case neg.h.mp.intro.intro
      n : Nat
      hn : Not (Eq n 0)
      a x : Nat
      hx1 : Membership.mem n.divisors x
      hx2 : Eq (HDiv.hDiv n x) a
      ⊢ Membership.mem n.divisors a
    -/
    rw [mem_divisors] at *
    /-
      case neg.h.mp.intro.intro
      n : Nat
      hn : Not (Eq n 0)
      a x : Nat
      hx1 : And (Dvd.dvd x n) (Ne n 0)
      hx2 : Eq (HDiv.hDiv n x) a
      ⊢ And (Dvd.dvd a n) (Ne n 0)
    -/
    refine ⟨?_, hn⟩
    /-
      case neg.h.mp.intro.intro
      n : Nat
      hn : Not (Eq n 0)
      a x : Nat
      hx1 : And (Dvd.dvd x n) (Ne n 0)
      hx2 : Eq (HDiv.hDiv n x) a
      ⊢ Dvd.dvd a n
    -/
    rw [← hx2]
    /-
      case neg.h.mp.intro.intro
      n : Nat
      hn : Not (Eq n 0)
      a x : Nat
      hx1 : And (Dvd.dvd x n) (Ne n 0)
      hx2 : Eq (HDiv.hDiv n x) a
      ⊢ Dvd.dvd (HDiv.hDiv n x) n
    -/
    exact div_dvd_of_dvd hx1.1
    /-
      🎉 no goals
    -/
    /-
      case neg.h.mpr
      n : Nat
      hn : Not (Eq n 0)
      a : Nat
      ⊢ Membership.mem n.divisors a → Membership.mem (Finset.image (fun x => HDiv.hD …
    -/
  · rw [mem_divisors, mem_image]
    /-
      case neg.h.mpr
      n : Nat
      hn : Not (Eq n 0)
      a : Nat
      ⊢ And (Dvd.dvd a n) (Ne n 0) → Exists fun a_2 => And (Membership.mem n.divisor …
    -/
    rintro ⟨h1, -⟩
    /-
      case neg.h.mpr.intro
      n : Nat
      hn : Not (Eq n 0)
      a : Nat
      h1 : Dvd.dvd a n
      ⊢ Exists fun a_1 => And (Membership.mem n.divisors a_1) (Eq (HDiv.hDiv n a_1) a)
    -/
    exact ⟨n / a, mem_divisors.mpr ⟨div_dvd_of_dvd h1, hn⟩, Nat.div_div_self h1 hn⟩
    /-
      🎉 no goals
    -/

/- Porting note: Removed simp; simp_nf linter:
Left-hand side does not simplify, when using the simp lemma on itself.
This usually means that it will never apply. -/

@[to_additive sum_div_divisors]
theorem prod_div_divisors {α : Type*} [CommMonoid α] (n : ℕ) (f : ℕ → α) :
    (∏ d ∈ n.divisors, f (n / d)) = n.divisors.prod f := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    n : Nat
    f : Nat → α
    ⊢ Eq (n.divisors.prod fun d => f (HDiv.hDiv n d)) (n.divisors.prod f)
  -/
  by_cases hn : n = 0; · simp [hn]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝ : CommMonoid α
    n : Nat
    f : Nat → α
    hn : Not (Eq n 0)
    ⊢ Eq (n.divisors.prod fun d => f (HDiv.hDiv n d)) (n.divisors.prod f)
  -/
  rw [← prod_image]
    /-
      case neg
      α : Type u_1
      inst✝ : CommMonoid α
      n : Nat
      f : Nat → α
      hn : Not (Eq n 0)
      ⊢ Eq ((Finset.image (HDiv.hDiv n) n.divisors).prod fun x => f x) (n.divisors.p …
    -/
  · exact prod_congr (image_div_divisors_eq_divisors n) (by simp)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : CommMonoid α
      n : Nat
      f : Nat → α
      hn : Not (Eq n 0)
      ⊢ ∀ (x : Nat), Membership.mem n.divisors x → ∀ (y : Nat), Membership.mem n.div …
    -/
  · intro x hx y hy h
    /-
      case neg
      α : Type u_1
      inst✝ : CommMonoid α
      n : Nat
      f : Nat → α
      hn : Not (Eq n 0)
      x : Nat
      hx : Membership.mem n.divisors x
      y : Nat
      hy : Membership.mem n.divisors y
      h : Eq (HDiv.hDiv n x) (HDiv.hDiv n y)
      ⊢ Eq x y
    -/
    rw [mem_divisors] at hx hy
    /-
      case neg
      α : Type u_1
      inst✝ : CommMonoid α
      n : Nat
      f : Nat → α
      hn : Not (Eq n 0)
      x : Nat
      hx : And (Dvd.dvd x n) (Ne n 0)
      y : Nat
      hy : And (Dvd.dvd y n) (Ne n 0)
      h : Eq (HDiv.hDiv n x) (HDiv.hDiv n y)
      ⊢ Eq x y
    -/
    exact (div_eq_iff_eq_of_dvd_dvd hn hx.1 hy.1).mp h
    /-
      🎉 no goals
    -/


