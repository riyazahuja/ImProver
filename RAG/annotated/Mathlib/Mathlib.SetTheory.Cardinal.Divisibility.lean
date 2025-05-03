@[simp]
theorem isUnit_iff : IsUnit a ↔ a = 1 := by
  refine
    ⟨fun h => ?_, by
      rintro rfl
      exact isUnit_one⟩
  /-
    a : Cardinal.{u}
    h : IsUnit a
    ⊢ Eq a 1
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      h : IsUnit 0
      ⊢ Eq 0 1
    -/
  · exact (not_isUnit_zero h).elim
    /-
      🎉 no goals
    -/
  /-
    case inr
    a : Cardinal.{u}
    h : IsUnit a
    ha : Ne a 0
    ⊢ Eq a 1
  -/
  rw [isUnit_iff_forall_dvd] at h
  /-
    case inr
    a : Cardinal.{u}
    h : ∀ (y : Cardinal.{u}), Dvd.dvd a y
    ha : Ne a 0
    ⊢ Eq a 1
  -/
  cases' h 1 with t ht
  /-
    case inr.intro
    a : Cardinal.{u}
    h : ∀ (y : Cardinal.{u}), Dvd.dvd a y
    ha : Ne a 0
    t : Cardinal.{u}
    ht : Eq 1 (HMul.hMul a t)
    ⊢ Eq a 1
  -/
  rw [eq_comm, mul_eq_one_iff_of_one_le] at ht
    /-
      case inr.intro
      a : Cardinal.{u}
      h : ∀ (y : Cardinal.{u}), Dvd.dvd a y
      ha : Ne a 0
      t : Cardinal.{u}
      ht : And (Eq a 1) (Eq t 1)
      ⊢ Eq a 1
    -/
  · exact ht.1
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.ha
      a : Cardinal.{u}
      h : ∀ (y : Cardinal.{u}), Dvd.dvd a y
      ha : Ne a 0
      t : Cardinal.{u}
      ht : Eq (HMul.hMul a t) 1
      ⊢ LE.le 1 a
    -/
  · exact one_le_iff_ne_zero.mpr ha
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.hb
      a : Cardinal.{u}
      h : ∀ (y : Cardinal.{u}), Dvd.dvd a y
      ha : Ne a 0
      t : Cardinal.{u}
      ht : Eq (HMul.hMul a t) 1
      ⊢ LE.le 1 t
    -/
  · apply one_le_iff_ne_zero.mpr
    /-
      case inr.intro.hb
      a : Cardinal.{u}
      h : ∀ (y : Cardinal.{u}), Dvd.dvd a y
      ha : Ne a 0
      t : Cardinal.{u}
      ht : Eq (HMul.hMul a t) 1
      ⊢ Ne t 0
    -/
    intro h
    /-
      case inr.intro.hb
      a : Cardinal.{u}
      h✝ : ∀ (y : Cardinal.{u}), Dvd.dvd a y
      ha : Ne a 0
      t : Cardinal.{u}
      ht : Eq (HMul.hMul a t) 1
      h : Eq t 0
      ⊢ False
    -/
    rw [h, mul_zero] at ht
    /-
      case inr.intro.hb
      a : Cardinal.{u}
      h✝ : ∀ (y : Cardinal.{u}), Dvd.dvd a y
      ha : Ne a 0
      t : Cardinal.{u}
      ht : Eq 0 1
      h : Eq t 0
      ⊢ False
    -/
    exact zero_ne_one ht
    /-
      🎉 no goals
    -/


instance : Unique Cardinal.{u}ˣ where
  default := 1
  uniq a := Units.val_eq_one.mp <| isUnit_iff.mp a.isUnit


theorem le_of_dvd : ∀ {a b : Cardinal}, b ≠ 0 → a ∣ b → a ≤ b
  | a, x, b0, ⟨b, hab⟩ => by
    simpa only [hab, mul_one] using
      mul_le_mul_left' (one_le_iff_ne_zero.2 fun h : b = 0 => b0 (by rwa [h, mul_zero] at hab)) a


theorem dvd_of_le_of_aleph0_le (ha : a ≠ 0) (h : a ≤ b) (hb : ℵ₀ ≤ b) : a ∣ b :=
  ⟨b, (mul_eq_right hb h ha).symm⟩


@[simp]
theorem prime_of_aleph0_le (ha : ℵ₀ ≤ a) : Prime a := by
  /-
    a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    ⊢ Prime a
  -/
  refine ⟨(aleph0_pos.trans_le ha).ne', ?_, fun b c hbc => ?_⟩
    /-
      case refine_1
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      ⊢ Not (IsUnit a)
    -/
  · rw [isUnit_iff]
    /-
      case refine_1
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      ⊢ Not (Eq a 1)
    -/
    exact (one_lt_aleph0.trans_le ha).ne'
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    b c : Cardinal.{u}
    hbc : Dvd.dvd a (HMul.hMul b c)
    ⊢ Or (Dvd.dvd a b) (Dvd.dvd a c)
  -/
  rcases eq_or_ne (b * c) 0 with hz | hz
    /-
      case refine_2.inl
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      b c : Cardinal.{u}
      hbc : Dvd.dvd a (HMul.hMul b c)
      hz : Eq (HMul.hMul b c) 0
      ⊢ Or (Dvd.dvd a b) (Dvd.dvd a c)
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  · rcases mul_eq_zero.mp hz with (rfl | rfl) <;> simp
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case refine_2.inr
    a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    b c : Cardinal.{u}
    hbc : Dvd.dvd a (HMul.hMul b c)
    hz : Ne (HMul.hMul b c) 0
    ⊢ Or (Dvd.dvd a b) (Dvd.dvd a c)
  -/
  wlog h : c ≤ b
    /-
      case refine_2.inr.inr
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      b c : Cardinal.{u}
      hbc : Dvd.dvd a (HMul.hMul b c)
      hz : Ne (HMul.hMul b c) 0
      this : ∀ {a : Cardinal.{u}}, LE.le Cardinal.aleph0 a → ∀ (b c : Cardinal.{u}), …
      h : Not (LE.le c b)
      ⊢ Or (Dvd.dvd a b) (Dvd.dvd a c)
    -/
  · cases le_total c b <;> [solve_by_elim; rw [or_comm]]
    /-
      case refine_2.inr.inr.inr
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      b c : Cardinal.{u}
      hbc : Dvd.dvd a (HMul.hMul b c)
      hz : Ne (HMul.hMul b c) 0
      this : ∀ {a : Cardinal.{u}}, LE.le Cardinal.aleph0 a → ∀ (b c : Cardinal.{u}), …
      h : Not (LE.le c b)
      h✝ : LE.le b c
      ⊢ Or (Dvd.dvd a c) (Dvd.dvd a b)
    -/
    apply_assumption
    /-
      case refine_2.inr.inr.inr.ha
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      b c : Cardinal.{u}
      hbc : Dvd.dvd a (HMul.hMul b c)
      hz : Ne (HMul.hMul b c) 0
      this : ∀ {a : Cardinal.{u}}, LE.le Cardinal.aleph0 a → ∀ (b c : Cardinal.{u}), …
      h : Not (LE.le c b)
      h✝ : LE.le b c
      ⊢ LE.le Cardinal.aleph0 a
    -/
    assumption'
    /-
      case refine_2.inr.inr.inr.hbc
      a : Cardinal.{u}
      ha : LE.le Cardinal.aleph0 a
      b c : Cardinal.{u}
      hbc : Dvd.dvd a (HMul.hMul b c)
      hz : Ne (HMul.hMul b c) 0
      this : ∀ {a : Cardinal.{u}}, LE.le Cardinal.aleph0 a → ∀ (b c : Cardinal.{u}), …
      h : Not (LE.le c b)
      h✝ : LE.le b c
      ⊢ Dvd.dvd a (HMul.hMul c b)
    -/
    all_goals rwa [mul_comm]
    /-
      🎉 no goals
    -/
  /-
    a✝ a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    b c : Cardinal.{u}
    hbc : Dvd.dvd a (HMul.hMul b c)
    hz : Ne (HMul.hMul b c) 0
    h : LE.le c b
    ⊢ Or (Dvd.dvd a b) (Dvd.dvd a c)
  -/
  left
  /-
    case h
    a✝ a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    b c : Cardinal.{u}
    hbc : Dvd.dvd a (HMul.hMul b c)
    hz : Ne (HMul.hMul b c) 0
    h : LE.le c b
    ⊢ Dvd.dvd a b
  -/
  have habc := le_of_dvd hz hbc
  /-
    case h
    a✝ a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    b c : Cardinal.{u}
    hbc : Dvd.dvd a (HMul.hMul b c)
    hz : Ne (HMul.hMul b c) 0
    h : LE.le c b
    habc : LE.le a (HMul.hMul b c)
    ⊢ Dvd.dvd a b
  -/
  rwa [mul_eq_max' <| ha.trans <| habc, max_def', if_pos h] at hbc
  /-
    🎉 no goals
  -/


theorem not_irreducible_of_aleph0_le (ha : ℵ₀ ≤ a) : ¬Irreducible a := by
  /-
    a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    ⊢ Not (Irreducible a)
  -/
  rw [irreducible_iff, not_and_or]
  /-
    a : Cardinal.{u}
    ha : LE.le Cardinal.aleph0 a
    ⊢ Or (Not (Not (IsUnit a))) (Not (∀ (a_1 b : Cardinal.{u}), Eq a (HMul.hMul a_ …
  -/
  refine Or.inr fun h => ?_
  simpa [mul_aleph0_eq ha, isUnit_iff, (one_lt_aleph0.trans_le ha).ne', one_lt_aleph0.ne'] using
    h a ℵ₀


@[simp, norm_cast]
theorem nat_coe_dvd_iff : (n : Cardinal) ∣ m ↔ n ∣ m := by
  /-
    n m : Nat
    ⊢ Iff (Dvd.dvd ↑n ↑m) (Dvd.dvd n m)
  -/
  refine ⟨?_, fun ⟨h, ht⟩ => ⟨h, mod_cast ht⟩⟩
  /-
    n m : Nat
    ⊢ Dvd.dvd ↑n ↑m → Dvd.dvd n m
  -/
  rintro ⟨k, hk⟩
  /-
    case intro
    n m : Nat
    k : Cardinal.{u_1}
    hk : Eq (↑m) (HMul.hMul (↑n) k)
    ⊢ Dvd.dvd n m
  -/
  have : ↑m < ℵ₀ := nat_lt_aleph0 m
  /-
    case intro
    n m : Nat
    k : Cardinal.{u_1}
    hk : Eq (↑m) (HMul.hMul (↑n) k)
    this : LT.lt (↑m) Cardinal.aleph0
    ⊢ Dvd.dvd n m
  -/
  rw [hk, mul_lt_aleph0_iff] at this
  /-
    case intro
    n m : Nat
    k : Cardinal.{u_1}
    hk : Eq (↑m) (HMul.hMul (↑n) k)
    this : Or (Eq (↑n) 0) (Or (Eq k 0) (And (LT.lt (↑n) Cardinal.aleph0) (LT.lt k  …
    ⊢ Dvd.dvd n m
  -/
  rcases this with (h | h | ⟨-, hk'⟩)
  /-
    case intro.inl
    n m : Nat
    k : Cardinal.{u_1}
    hk : Eq (↑m) (HMul.hMul (↑n) k)
    h : Eq (↑n) 0
    ⊢ Dvd.dvd n m
  -/
  iterate 2 simp only [h, mul_zero, zero_mul, Nat.cast_eq_zero] at hk; simp [hk]
  /-
    case intro.inr.inr.intro
    n m : Nat
    k : Cardinal.{u_1}
    hk : Eq (↑m) (HMul.hMul (↑n) k)
    hk' : LT.lt k Cardinal.aleph0
    ⊢ Dvd.dvd n m
  -/
  lift k to ℕ using hk'
  /-
    case intro.inr.inr.intro.intro
    n m k : Nat
    hk : Eq (↑m) (HMul.hMul ↑n ↑k)
    ⊢ Dvd.dvd n m
  -/
  exact ⟨k, mod_cast hk⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem nat_is_prime_iff : Prime (n : Cardinal) ↔ n.Prime := by
  /-
    n : Nat
    ⊢ Iff (Prime ↑n) (Nat.Prime n)
  -/
  simp only [Prime, Nat.prime_iff]
  /-
    n : Nat
    ⊢ Iff (And (Ne (↑n) 0) (And (Not (IsUnit ↑n)) (∀ (a b : Cardinal.{u_1}), Dvd.d …
  -/
  refine and_congr (by simp) (and_congr ?_ ⟨fun h b c hbc => ?_, fun h b c hbc => ?_⟩)
    /-
      case refine_1
      n : Nat
      ⊢ Iff (Not (IsUnit ↑n)) (Not (IsUnit n))
    -/
  · simp only [isUnit_iff, Nat.isUnit_iff]
    /-
      case refine_1
      n : Nat
      ⊢ Iff (Not (Eq (↑n) 1)) (Not (Eq n 1))
    -/
    exact mod_cast Iff.rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      h : ∀ (a b : Cardinal.{u_1}), Dvd.dvd (↑n) (HMul.hMul a b) → Or (Dvd.dvd (↑n)  …
      b c : Nat
      hbc : Dvd.dvd n (HMul.hMul b c)
      ⊢ Or (Dvd.dvd n b) (Dvd.dvd n c)
    -/
  · exact mod_cast h b c (mod_cast hbc)
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    n : Nat
    h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
    b c : Cardinal.{u_1}
    hbc : Dvd.dvd (↑n) (HMul.hMul b c)
    ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
  -/
  cases' lt_or_le (b * c) ℵ₀ with h' h'
    /-
      case refine_3.inl
      n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      b c : Cardinal.{u_1}
      hbc : Dvd.dvd (↑n) (HMul.hMul b c)
      h' : LT.lt (HMul.hMul b c) Cardinal.aleph0
      ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
    -/
  · rcases mul_lt_aleph0_iff.mp h' with (rfl | rfl | ⟨hb, hc⟩)
      /-
        case refine_3.inl.inl
        n : Nat
        h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
        c : Cardinal.{u_1}
        hbc : Dvd.dvd (↑n) (HMul.hMul 0 c)
        h' : LT.lt (HMul.hMul 0 c) Cardinal.aleph0
        ⊢ Or (Dvd.dvd (↑n) 0) (Dvd.dvd (↑n) c)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_3.inl.inr.inl
        n : Nat
        h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
        b : Cardinal.{u_1}
        hbc : Dvd.dvd (↑n) (HMul.hMul b 0)
        h' : LT.lt (HMul.hMul b 0) Cardinal.aleph0
        ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) 0)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case refine_3.inl.inr.inr.intro
      n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      b c : Cardinal.{u_1}
      hbc : Dvd.dvd (↑n) (HMul.hMul b c)
      h' : LT.lt (HMul.hMul b c) Cardinal.aleph0
      hb : LT.lt b Cardinal.aleph0
      hc : LT.lt c Cardinal.aleph0
      ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
    -/
    lift b to ℕ using hb
    /-
      case refine_3.inl.inr.inr.intro.intro
      n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      c : Cardinal.{u_1}
      hc : LT.lt c Cardinal.aleph0
      b : Nat
      hbc : Dvd.dvd (↑n) (HMul.hMul (↑b) c)
      h' : LT.lt (HMul.hMul (↑b) c) Cardinal.aleph0
      ⊢ Or (Dvd.dvd ↑n ↑b) (Dvd.dvd (↑n) c)
    -/
    lift c to ℕ using hc
    /-
      case refine_3.inl.inr.inr.intro.intro.intro
      n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      b c : Nat
      hbc : Dvd.dvd (↑n) (HMul.hMul ↑b ↑c)
      h' : LT.lt (HMul.hMul ↑b ↑c) Cardinal.aleph0
      ⊢ Or (Dvd.dvd ↑n ↑b) (Dvd.dvd ↑n ↑c)
    -/
    exact mod_cast h b c (mod_cast hbc)
    /-
      🎉 no goals
    -/
  /-
    case refine_3.inr
    n : Nat
    h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
    b c : Cardinal.{u_1}
    hbc : Dvd.dvd (↑n) (HMul.hMul b c)
    h' : LE.le Cardinal.aleph0 (HMul.hMul b c)
    ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
  -/
  rcases aleph0_le_mul_iff.mp h' with ⟨hb, hc, hℵ₀⟩
  have hn : (n : Cardinal) ≠ 0 := by
    intro h
    rw [h, zero_dvd_iff, mul_eq_zero] at hbc
    cases hbc <;> contradiction
  /-
    case refine_3.inr.intro.intro
    n : Nat
    h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
    b c : Cardinal.{u_1}
    hbc : Dvd.dvd (↑n) (HMul.hMul b c)
    h' : LE.le Cardinal.aleph0 (HMul.hMul b c)
    hb : Ne b 0
    hc : Ne c 0
    hℵ₀ : Or (LE.le Cardinal.aleph0 b) (LE.le Cardinal.aleph0 c)
    hn : Ne (↑n) 0
    ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
  -/
  wlog hℵ₀b : ℵ₀ ≤ b
  /-
    case refine_3.inr.intro.intro.inr
    n : Nat
    h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
    b c : Cardinal.{u_1}
    hbc : Dvd.dvd (↑n) (HMul.hMul b c)
    h' : LE.le Cardinal.aleph0 (HMul.hMul b c)
    hb : Ne b 0
    hc : Ne c 0
    hℵ₀ : Or (LE.le Cardinal.aleph0 b) (LE.le Cardinal.aleph0 c)
    hn : Ne (↑n) 0
    this : ∀ {n : Nat}, (∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n  …
    hℵ₀b : Not (LE.le Cardinal.aleph0 b)
    ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
  -/
  apply (this h c b _ _ hc hb hℵ₀.symm hn (hℵ₀.resolve_left hℵ₀b)).symm <;> try assumption
    /-
      n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      b c : Cardinal.{u_1}
      hbc : Dvd.dvd (↑n) (HMul.hMul b c)
      h' : LE.le Cardinal.aleph0 (HMul.hMul b c)
      hb : Ne b 0
      hc : Ne c 0
      hℵ₀ : Or (LE.le Cardinal.aleph0 b) (LE.le Cardinal.aleph0 c)
      hn : Ne (↑n) 0
      this : ∀ {n : Nat}, (∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n  …
      hℵ₀b : Not (LE.le Cardinal.aleph0 b)
      ⊢ Dvd.dvd (↑n) (HMul.hMul c b)
    -/
  · rwa [mul_comm] at hbc
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      b c : Cardinal.{u_1}
      hbc : Dvd.dvd (↑n) (HMul.hMul b c)
      h' : LE.le Cardinal.aleph0 (HMul.hMul b c)
      hb : Ne b 0
      hc : Ne c 0
      hℵ₀ : Or (LE.le Cardinal.aleph0 b) (LE.le Cardinal.aleph0 c)
      hn : Ne (↑n) 0
      this : ∀ {n : Nat}, (∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n  …
      hℵ₀b : Not (LE.le Cardinal.aleph0 b)
      ⊢ LE.le Cardinal.aleph0 (HMul.hMul c b)
    -/
  · rwa [mul_comm] at h'
    /-
      🎉 no goals
    -/
    /-
      n✝ n : Nat
      h : ∀ (a b : Nat), Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
      b c : Cardinal.{u_1}
      hbc : Dvd.dvd (↑n) (HMul.hMul b c)
      h' : LE.le Cardinal.aleph0 (HMul.hMul b c)
      hb : Ne b 0
      hc : Ne c 0
      hℵ₀ : Or (LE.le Cardinal.aleph0 b) (LE.le Cardinal.aleph0 c)
      hn : Ne (↑n) 0
      hℵ₀b : LE.le Cardinal.aleph0 b
      ⊢ Or (Dvd.dvd (↑n) b) (Dvd.dvd (↑n) c)
    -/
  · exact Or.inl (dvd_of_le_of_aleph0_le hn ((nat_lt_aleph0 n).le.trans hℵ₀b) hℵ₀b)
    /-
      🎉 no goals
    -/


theorem is_prime_iff {a : Cardinal} : Prime a ↔ ℵ₀ ≤ a ∨ ∃ p : ℕ, a = p ∧ p.Prime := by
  /-
    a : Cardinal.{u_1}
    ⊢ Iff (Prime a) (Or (LE.le Cardinal.aleph0 a) (Exists fun p => And (Eq a ↑p) ( …
  -/
  rcases le_or_lt ℵ₀ a with h | h
    /-
      case inl
      a : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 a
      ⊢ Iff (Prime a) (Or (LE.le Cardinal.aleph0 a) (Exists fun p => And (Eq a ↑p) ( …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a : Cardinal.{u_1}
    h : LT.lt a Cardinal.aleph0
    ⊢ Iff (Prime a) (Or (LE.le Cardinal.aleph0 a) (Exists fun p => And (Eq a ↑p) ( …
  -/
  lift a to ℕ using id h
  /-
    case inr.intro
    a : Nat
    h : LT.lt (↑a) Cardinal.aleph0
    ⊢ Iff (Prime ↑a) (Or (LE.le Cardinal.aleph0 ↑a) (Exists fun p => And (Eq ↑a ↑p …
  -/
  simp [not_le.mpr h]
  /-
    🎉 no goals
  -/


theorem isPrimePow_iff {a : Cardinal} : IsPrimePow a ↔ ℵ₀ ≤ a ∨ ∃ n : ℕ, a = n ∧ IsPrimePow n := by
  /-
    a : Cardinal.{u_1}
    ⊢ Iff (IsPrimePow a) (Or (LE.le Cardinal.aleph0 a) (Exists fun n => And (Eq a  …
  -/
  by_cases h : ℵ₀ ≤ a
    /-
      case pos
      a : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 a
      ⊢ Iff (IsPrimePow a) (Or (LE.le Cardinal.aleph0 a) (Exists fun n => And (Eq a  …
    -/
  · simp [h, (prime_of_aleph0_le h).isPrimePow]
    /-
      🎉 no goals
    -/
  /-
    case neg
    a : Cardinal.{u_1}
    h : Not (LE.le Cardinal.aleph0 a)
    ⊢ Iff (IsPrimePow a) (Or (LE.le Cardinal.aleph0 a) (Exists fun n => And (Eq a  …
  -/
  simp only [h, Nat.cast_inj, exists_eq_left', false_or, isPrimePow_nat_iff]
  /-
    case neg
    a : Cardinal.{u_1}
    h : Not (LE.le Cardinal.aleph0 a)
    ⊢ Iff (IsPrimePow a) (Exists fun n => And (Eq a ↑n) (Exists fun p => Exists fu …
  -/
  lift a to ℕ using not_le.mp h
  /-
    case neg.intro
    a : Nat
    h : Not (LE.le Cardinal.aleph0 ↑a)
    ⊢ Iff (IsPrimePow ↑a) (Exists fun n => And (Eq ↑a ↑n) (Exists fun p => Exists  …
  -/
  rw [isPrimePow_def]
  refine
    ⟨?_, fun ⟨n, han, p, k, hp, hk, h⟩ =>
          ⟨p, k, nat_is_prime_iff.2 hp, hk, by rw [han]; exact mod_cast h⟩⟩
  /-
    case neg.intro
    a : Nat
    h : Not (LE.le Cardinal.aleph0 ↑a)
    ⊢ (Exists fun p => Exists fun k => And (Prime p) (And (LT.lt 0 k) (Eq (HPow.hP …
  -/
  rintro ⟨p, k, hp, hk, hpk⟩
  have key : p ^ (1 : Cardinal) ≤ ↑a := by
    rw [← hpk]; apply power_le_power_left hp.ne_zero; exact mod_cast hk
  /-
    case neg.intro.intro.intro.intro.intro
    a : Nat
    h : Not (LE.le Cardinal.aleph0 ↑a)
    p : Cardinal.{u_1}
    k : Nat
    hp : Prime p
    hk : LT.lt 0 k
    hpk : Eq (HPow.hPow p k) ↑a
    key : LE.le (HPow.hPow p 1) ↑a
    ⊢ Exists fun n => And (Eq ↑a ↑n) (Exists fun p => Exists fun k => And (Nat.Pri …
  -/
  rw [power_one] at key
  /-
    case neg.intro.intro.intro.intro.intro
    a : Nat
    h : Not (LE.le Cardinal.aleph0 ↑a)
    p : Cardinal.{u_1}
    k : Nat
    hp : Prime p
    hk : LT.lt 0 k
    hpk : Eq (HPow.hPow p k) ↑a
    key : LE.le p ↑a
    ⊢ Exists fun n => And (Eq ↑a ↑n) (Exists fun p => Exists fun k => And (Nat.Pri …
  -/
  lift p to ℕ using key.trans_lt (nat_lt_aleph0 a)
  /-
    case neg.intro.intro.intro.intro.intro.intro
    a : Nat
    h : Not (LE.le Cardinal.aleph0 ↑a)
    k : Nat
    hk : LT.lt 0 k
    p : Nat
    hp : Prime ↑p
    hpk : Eq (HPow.hPow (↑p) k) ↑a
    key : LE.le ↑p ↑a
    ⊢ Exists fun n => And (Eq ↑a ↑n) (Exists fun p => Exists fun k => And (Nat.Pri …
  -/
  exact ⟨a, rfl, p, k, nat_is_prime_iff.mp hp, hk, mod_cast hpk⟩
  /-
    🎉 no goals
  -/


