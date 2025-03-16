lemma even_of_val {n : ℕ} {k : Fin n} (h : Even k.val) : Even k := by
  /-
    n : Nat
    k : Fin n
    h : Even ↑k
    ⊢ Even k
  -/
  have : NeZero n := ⟨k.pos.ne'⟩
  /-
    n : Nat
    k : Fin n
    h : Even ↑k
    this : NeZero n
    ⊢ Even k
  -/
  rw [← Fin.cast_val_eq_self k]
  /-
    n : Nat
    k : Fin n
    h : Even ↑k
    this : NeZero n
    ⊢ Even ↑↑k
  -/
  exact h.natCast
  /-
    🎉 no goals
  -/


lemma odd_of_val {n : ℕ} [NeZero n] {k : Fin n} (h : Odd k.val) : Odd k := by
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    h : Odd ↑k
    ⊢ Odd k
  -/
  rw [← Fin.cast_val_eq_self k]
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    h : Odd ↑k
    ⊢ Odd ↑↑k
  -/
  exact h.natCast
  /-
    🎉 no goals
  -/


lemma even_of_odd {n : ℕ} (hn : Odd n) (k : Fin n) : Even k := by
  /-
    n : Nat
    hn : Odd n
    k : Fin n
    ⊢ Even k
  -/
  have : NeZero n := ⟨k.pos.ne'⟩
  /-
    n : Nat
    hn : Odd n
    k : Fin n
    this : NeZero n
    ⊢ Even k
  -/
  rcases k.val.even_or_odd with hk | hk
    /-
      case inl
      n : Nat
      hn : Odd n
      k : Fin n
      this : NeZero n
      hk : Even ↑k
      ⊢ Even k
    -/
  · exact even_of_val hk
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : Odd n
      k : Fin n
      this : NeZero n
      hk : Odd ↑k
      ⊢ Even k
    -/
  · simpa using (hk.add_odd hn).natCast (α := Fin n)
    /-
      🎉 no goals
    -/



lemma odd_of_odd {n : ℕ} [NeZero n] (hn : Odd n) (k : Fin n) : Odd k := by
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Odd n
    k : Fin n
    ⊢ Odd k
  -/
  rcases k.val.even_or_odd with hk | hk
    /-
      case inl
      n : Nat
      inst✝ : NeZero n
      hn : Odd n
      k : Fin n
      hk : Even ↑k
      ⊢ Odd k
    -/
  · simpa using (Even.add_odd hk hn).natCast (R := Fin n)
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      inst✝ : NeZero n
      hn : Odd n
      k : Fin n
      hk : Odd ↑k
      ⊢ Odd k
    -/
  · exact odd_of_val hk
    /-
      🎉 no goals
    -/


lemma even_iff_of_even {n : ℕ} (hn : Even n) {k : Fin n} : Even k ↔ Even k.val := by
  /-
    n : Nat
    hn : Even n
    k : Fin n
    ⊢ Iff (Even k) (Even ↑k)
  -/
  rcases hn with ⟨n, rfl⟩
  /-
    case intro
    n : Nat
    k : Fin (HAdd.hAdd n n)
    ⊢ Iff (Even k) (Even ↑k)
  -/
  refine ⟨?_, even_of_val⟩
  /-
    case intro
    n : Nat
    k : Fin (HAdd.hAdd n n)
    ⊢ Even k → Even ↑k
  -/
  rintro ⟨l, rfl⟩
  /-
    case intro.intro
    n : Nat
    l : Fin (HAdd.hAdd n n)
    ⊢ Even ↑(HAdd.hAdd l l)
  -/
  rw [val_add_eq_ite]
  /-
    case intro.intro
    n : Nat
    l : Fin (HAdd.hAdd n n)
    ⊢ Even (ite (LE.le (HAdd.hAdd n n) (HAdd.hAdd ↑l ↑l)) (HSub.hSub (HAdd.hAdd ↑l …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [Nat.even_sub, *]
                       /-
                         🎉 no goals
                       -/


lemma odd_iff_of_even {n : ℕ} [NeZero n] (hn : Even n) {k : Fin n} : Odd k ↔ Odd k.val := by
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Odd k) (Odd ↑k)
  -/
  rcases hn with ⟨n, rfl⟩
  /-
    case intro
    n : Nat
    inst✝ : NeZero (HAdd.hAdd n n)
    k : Fin (HAdd.hAdd n n)
    ⊢ Iff (Odd k) (Odd ↑k)
  -/
  refine ⟨?_, odd_of_val⟩
  /-
    case intro
    n : Nat
    inst✝ : NeZero (HAdd.hAdd n n)
    k : Fin (HAdd.hAdd n n)
    ⊢ Odd k → Odd ↑k
  -/
  rintro ⟨l, rfl⟩
  /-
    case intro.intro
    n : Nat
    inst✝ : NeZero (HAdd.hAdd n n)
    l : Fin (HAdd.hAdd n n)
    ⊢ Odd ↑(HAdd.hAdd (HMul.hMul 2 l) 1)
  -/
  rw [val_add, val_mul, val_one', show Fin.val 2 = 2 % _ from rfl]
  /-
    case intro.intro
    n : Nat
    inst✝ : NeZero (HAdd.hAdd n n)
    l : Fin (HAdd.hAdd n n)
    ⊢ Odd (HMod.hMod (HAdd.hAdd (HMod.hMod (HMul.hMul (HMod.hMod 2 (HAdd.hAdd n n) …
  -/
  simp only [Nat.mod_mul_mod, Nat.add_mod_mod, Nat.mod_add_mod, Nat.odd_iff]
  rw [Nat.mod_mod_of_dvd _ ⟨n, (two_mul n).symm⟩, ← Nat.odd_iff, Nat.odd_add_one,
    Nat.not_odd_iff_even]
  /-
    case intro.intro
    n : Nat
    inst✝ : NeZero (HAdd.hAdd n n)
    l : Fin (HAdd.hAdd n n)
    ⊢ Even (HMul.hMul 2 ↑l)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- In `Fin n`, all elements are even for odd `n`,
otherwise an element is even iff its `Fin.val` value is even. -/
lemma even_iff {n : ℕ} {k : Fin n} : Even k ↔ (Odd n ∨ Even k.val) := by
  /-
    n : Nat
    k : Fin n
    ⊢ Iff (Even k) (Or (Odd n) (Even ↑k))
  -/
  refine ⟨fun hk ↦ ?_, or_imp.mpr ⟨(even_of_odd · k), even_of_val⟩⟩
  /-
    n : Nat
    k : Fin n
    hk : Even k
    ⊢ Or (Odd n) (Even ↑k)
  -/
  rw [← Nat.not_even_iff_odd, ← imp_iff_not_or]
  /-
    n : Nat
    k : Fin n
    hk : Even k
    ⊢ Even n → Even ↑k
  -/
  exact fun hn ↦ (even_iff_of_even hn).mp hk
  /-
    🎉 no goals
  -/


lemma even_iff_imp {n : ℕ} {k : Fin n} : Even k ↔ (Even n → Even k.val) := by
  /-
    n : Nat
    k : Fin n
    ⊢ Iff (Even k) (Even n → Even ↑k)
  -/
  rw [imp_iff_not_or, Nat.not_even_iff_odd]
  /-
    n : Nat
    k : Fin n
    ⊢ Iff (Even k) (Or (Odd n) (Even ↑k))
  -/
  exact even_iff
  /-
    🎉 no goals
  -/


/-- In `Fin n`, all elements are odd for odd `n`,
otherwise an element is odd iff its `Fin.val` value is odd. -/
lemma odd_iff {n : ℕ} [NeZero n] {k : Fin n} : Odd k ↔ Odd n ∨ Odd k.val := by
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    ⊢ Iff (Odd k) (Or (Odd n) (Odd ↑k))
  -/
  refine ⟨fun hk ↦ ?_, or_imp.mpr ⟨(odd_of_odd · k), odd_of_val⟩⟩
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    hk : Odd k
    ⊢ Or (Odd n) (Odd ↑k)
  -/
  rw [← Nat.not_even_iff_odd, ← imp_iff_not_or]
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    hk : Odd k
    ⊢ Even n → Odd ↑k
  -/
  exact fun hn ↦ (odd_iff_of_even hn).mp hk
  /-
    🎉 no goals
  -/


lemma odd_iff_imp {n : ℕ} [NeZero n] {k : Fin n} : Odd k ↔ (Even n → Odd k.val) := by
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    ⊢ Iff (Odd k) (Even n → Odd ↑k)
  -/
  rw [imp_iff_not_or, Nat.not_even_iff_odd]
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    ⊢ Iff (Odd k) (Or (Odd n) (Odd ↑k))
  -/
  exact odd_iff
  /-
    🎉 no goals
  -/


lemma even_iff_mod_of_even {n : ℕ} (hn : Even n) {k : Fin n} : Even k ↔ k.val % 2 = 0 := by
  /-
    n : Nat
    hn : Even n
    k : Fin n
    ⊢ Iff (Even k) (Eq (HMod.hMod (↑k) 2) 0)
  -/
  rw [even_iff_of_even hn]
  /-
    n : Nat
    hn : Even n
    k : Fin n
    ⊢ Iff (Even ↑k) (Eq (HMod.hMod (↑k) 2) 0)
  -/
  exact Nat.even_iff
  /-
    🎉 no goals
  -/


lemma odd_iff_mod_of_even {n : ℕ} [NeZero n] (hn : Even n) {k : Fin n} : Odd k ↔ k.val % 2 = 1 := by
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Odd k) (Eq (HMod.hMod (↑k) 2) 1)
  -/
  rw [odd_iff_of_even hn]
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Odd ↑k) (Eq (HMod.hMod (↑k) 2) 1)
  -/
  exact Nat.odd_iff
  /-
    🎉 no goals
  -/


lemma not_odd_iff_even_of_even {n : ℕ} [NeZero n] (hn : Even n) {k : Fin n} : ¬Odd k ↔ Even k := by
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Not (Odd k)) (Even k)
  -/
  rw [even_iff_of_even hn, odd_iff_of_even hn]
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Not (Odd ↑k)) (Even ↑k)
  -/
  exact Nat.not_odd_iff_even
  /-
    🎉 no goals
  -/


lemma not_even_iff_odd_of_even {n : ℕ} [NeZero n] (hn : Even n) {k : Fin n} : ¬Even k ↔ Odd k := by
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Not (Even k)) (Odd k)
  -/
  rw [even_iff_of_even hn, odd_iff_of_even hn]
  /-
    n : Nat
    inst✝ : NeZero n
    hn : Even n
    k : Fin n
    ⊢ Iff (Not (Even ↑k)) (Odd ↑k)
  -/
  exact Nat.not_even_iff_odd
  /-
    🎉 no goals
  -/


lemma odd_add_one_iff_even {n : ℕ} [NeZero n] {k : Fin n} : Odd (k + 1) ↔ Even k :=
  ⟨fun ⟨k, hk⟩ ↦ add_right_cancel hk ▸ even_two_mul k, Even.add_one⟩


lemma even_add_one_iff_odd {n : ℕ} [NeZero n] {k : Fin n} : Even (k + 1) ↔ Odd k :=
  ⟨fun ⟨k, hk⟩ ↦ eq_sub_iff_add_eq.mpr hk ▸ (Even.add_self k).sub_odd odd_one, Odd.add_one⟩


