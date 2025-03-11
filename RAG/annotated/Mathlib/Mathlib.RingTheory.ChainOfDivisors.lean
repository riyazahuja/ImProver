theorem Associates.isAtom_iff {p : Associates M} (h₁ : p ≠ 0) : IsAtom p ↔ Irreducible p :=
  ⟨fun hp =>
        /-
          M : Type u_1
          inst✝ : CancelCommMonoidWithZero M
          p : Associates M
          h₁ : Ne p 0
          hp : IsAtom p
          ⊢ Not (IsUnit p)
        -/
    ⟨by simpa only [Associates.isUnit_iff_eq_one] using hp.1, fun a b h =>
        /-
          🎉 no goals
        -/
      (hp.le_iff.mp ⟨_, h⟩).casesOn (fun ha => Or.inl (a.isUnit_iff_eq_one.mpr ha)) fun ha =>
        Or.inr
          (show IsUnit b by
            /-
              M : Type u_1
              inst✝ : CancelCommMonoidWithZero M
              p : Associates M
              h₁ : Ne p 0
              hp : IsAtom p
              a b : Associates M
              h : Eq p (HMul.hMul a b)
              ha : Eq a p
              ⊢ IsUnit b
            -/
            rw [ha] at h
            /-
              M : Type u_1
              inst✝ : CancelCommMonoidWithZero M
              p : Associates M
              h₁ : Ne p 0
              hp : IsAtom p
              a b : Associates M
              h : Eq p (HMul.hMul p b)
              ha : Eq a p
              ⊢ IsUnit b
            -/
            apply isUnit_of_associated_mul (show Associated (p * b) p by conv_rhs => rw [h]) h₁)⟩,
            /-
              🎉 no goals
            -/
    fun hp =>
        /-
          M : Type u_1
          inst✝ : CancelCommMonoidWithZero M
          p : Associates M
          h₁ : Ne p 0
          hp : Irreducible p
          ⊢ Ne p Bot.bot
        -/
    ⟨by simpa only [Associates.isUnit_iff_eq_one, Associates.bot_eq_one] using hp.1,
        /-
          🎉 no goals
        -/
      fun b ⟨⟨a, hab⟩, hb⟩ =>
      (hp.isUnit_or_isUnit hab).casesOn
                                 /-
                                   M : Type u_1
                                   inst✝ : CancelCommMonoidWithZero M
                                   p : Associates M
                                   h₁ : Ne p 0
                                   hp : Irreducible p
                                   b : Associates M
                                   x✝ : LT.lt b p
                                   a : Associates M
                                   hab : Eq p (HMul.hMul b a)
                                   hb✝ : Not (Dvd.dvd p b)
                                   hb : IsUnit b
                                   ⊢ Eq b Bot.bot
                                 -/
        (fun hb => show b = ⊥ by rwa [Associates.isUnit_iff_eq_one, ← Associates.bot_eq_one] at hb)
                                 /-
                                   🎉 no goals
                                 -/
        fun ha =>
        absurd
          (show p ∣ b from
                                       /-
                                         M : Type u_1
                                         inst✝ : CancelCommMonoidWithZero M
                                         p : Associates M
                                         h₁ : Ne p 0
                                         hp : Irreducible p
                                         b : Associates M
                                         x✝ : LT.lt b p
                                         a : Associates M
                                         hab : Eq p (HMul.hMul b a)
                                         hb : Not (Dvd.dvd p b)
                                         ha : IsUnit a
                                         ⊢ Eq b (HMul.hMul p ↑(Inv.inv ha.unit))
                                       -/
            ⟨(ha.unit⁻¹ : Units _), by rw [hab, mul_assoc, IsUnit.mul_val_inv ha, mul_one]⟩)
                                       /-
                                         🎉 no goals
                                       -/
          hb⟩⟩


theorem exists_chain_of_prime_pow {p : Associates M} {n : ℕ} (hn : n ≠ 0) (hp : Prime p) :
    ∃ c : Fin (n + 1) → Associates M,
      c 1 = p ∧ StrictMono c ∧ ∀ {r : Associates M}, r ≤ p ^ n ↔ ∃ i, r = c i := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : Associates M
    n : Nat
    hn : Ne n 0
    hp : Prime p
    ⊢ Exists fun c => And (Eq (c 1) p) (And (StrictMono c) (∀ {r : Associates M},  …
  -/
  refine ⟨fun i => p ^ (i : ℕ), ?_, fun n m h => ?_, @fun y => ⟨fun h => ?_, ?_⟩⟩
    /-
      case refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      ⊢ Eq ((fun i => HPow.hPow p ↑i) 1) p
    -/
  · dsimp only
    /-
      case refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      ⊢ Eq (HPow.hPow p ↑1) p
    -/
    rw [Fin.val_one', Nat.mod_eq_of_lt, pow_one]
    /-
      case refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      ⊢ LT.lt 1 (HAdd.hAdd n 1)
    -/
    exact Nat.lt_succ_of_le (Nat.one_le_iff_ne_zero.mpr hn)
    /-
      🎉 no goals
    -/
  · exact Associates.dvdNotUnit_iff_lt.mp
        ⟨pow_ne_zero n hp.ne_zero, p ^ (m - n : ℕ),
          not_isUnit_of_not_isUnit_dvd hp.not_unit (dvd_pow dvd_rfl (Nat.sub_pos_of_lt h).ne'),
          (pow_mul_pow_sub p h.le).symm⟩
    /-
      case refine_3
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      y : Associates M
      h : LE.le y (HPow.hPow p n)
      ⊢ Exists fun i => Eq y ((fun i => HPow.hPow p ↑i) i)
    -/
  · obtain ⟨i, i_le, hi⟩ := (dvd_prime_pow hp n).1 h
    /-
      case refine_3.intro.intro
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      y : Associates M
      h : LE.le y (HPow.hPow p n)
      i : Nat
      i_le : LE.le i n
      hi : Associated y (HPow.hPow p i)
      ⊢ Exists fun i => Eq y ((fun i => HPow.hPow p ↑i) i)
    -/
    rw [associated_iff_eq] at hi
    /-
      case refine_3.intro.intro
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      y : Associates M
      h : LE.le y (HPow.hPow p n)
      i : Nat
      i_le : LE.le i n
      hi : Eq y (HPow.hPow p i)
      ⊢ Exists fun i => Eq y ((fun i => HPow.hPow p ↑i) i)
    -/
    exact ⟨⟨i, Nat.lt_succ_of_le i_le⟩, hi⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      y : Associates M
      ⊢ (Exists fun i => Eq y ((fun i => HPow.hPow p ↑i) i)) → LE.le y (HPow.hPow p n)
    -/
  · rintro ⟨i, rfl⟩
    /-
      case refine_4.intro
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : Associates M
      n : Nat
      hn : Ne n 0
      hp : Prime p
      i : Fin (HAdd.hAdd n 1)
      ⊢ LE.le ((fun i => HPow.hPow p ↑i) i) (HPow.hPow p n)
    -/
    exact ⟨p ^ (n - i : ℕ), (pow_mul_pow_sub p (Nat.succ_le_succ_iff.mp i.2)).symm⟩
    /-
      🎉 no goals
    -/


theorem element_of_chain_not_isUnit_of_index_ne_zero {n : ℕ} {i : Fin (n + 1)} (i_pos : i ≠ 0)
    {c : Fin (n + 1) → Associates M} (h₁ : StrictMono c) : ¬IsUnit (c i) :=
  DvdNotUnit.not_unit
    (Associates.dvdNotUnit_iff_lt.2
      (h₁ <| show (0 : Fin (n + 1)) < i from Fin.pos_iff_ne_zero.mpr i_pos))


theorem first_of_chain_isUnit {q : Associates M} {n : ℕ} {c : Fin (n + 1) → Associates M}
    (h₁ : StrictMono c) (h₂ : ∀ {r}, r ≤ q ↔ ∃ i, r = c i) : IsUnit (c 0) := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    n : Nat
    c : Fin (HAdd.hAdd n 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    ⊢ IsUnit (c 0)
  -/
  obtain ⟨i, hr⟩ := h₂.mp Associates.one_le
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    n : Nat
    c : Fin (HAdd.hAdd n 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    i : Fin (HAdd.hAdd n 1)
    hr : Eq 1 (c i)
    ⊢ IsUnit (c 0)
  -/
  rw [Associates.isUnit_iff_eq_one, ← Associates.le_one_iff, hr]
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    n : Nat
    c : Fin (HAdd.hAdd n 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    i : Fin (HAdd.hAdd n 1)
    hr : Eq 1 (c i)
    ⊢ LE.le (c 0) (c i)
  -/
  exact h₁.monotone (Fin.zero_le i)
  /-
    🎉 no goals
  -/


/-- The second element of a chain is irreducible. -/
theorem second_of_chain_is_irreducible {q : Associates M} {n : ℕ} (hn : n ≠ 0)
    {c : Fin (n + 1) → Associates M} (h₁ : StrictMono c) (h₂ : ∀ {r}, r ≤ q ↔ ∃ i, r = c i)
    (hq : q ≠ 0) : Irreducible (c 1) := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    n : Nat
    hn : Ne n 0
    c : Fin (HAdd.hAdd n 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    hq : Ne q 0
    ⊢ Irreducible (c 1)
  -/
  cases' n with n; · contradiction
                     /-
                       🎉 no goals
                     -/
  /-
    case succ
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    hq : Ne q 0
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    ⊢ Irreducible (c 1)
  -/
  refine (Associates.isAtom_iff (ne_zero_of_dvd_ne_zero hq (h₂.2 ⟨1, rfl⟩))).mp ⟨?_, fun b hb => ?_⟩
    /-
      case succ.refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q : Associates M
      hq : Ne q 0
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      ⊢ Ne (c 1) Bot.bot
    -/
  · exact ne_bot_of_gt (h₁ (show (0 : Fin (n + 2)) < 1 from Fin.one_pos))
    /-
      🎉 no goals
    -/
  /-
    case succ.refine_2
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    hq : Ne q 0
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    b : Associates M
    hb : LT.lt b (c 1)
    ⊢ Eq b Bot.bot
  -/
  obtain ⟨⟨i, hi⟩, rfl⟩ := h₂.1 (hb.le.trans (h₂.2 ⟨1, rfl⟩))
  /-
    case succ.refine_2.intro.mk
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : Associates M
    hq : Ne q 0
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    i : Nat
    hi : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hb : LT.lt (c ⟨i, hi⟩) (c 1)
    ⊢ Eq (c ⟨i, hi⟩) Bot.bot
  -/
  cases i
    /-
      case succ.refine_2.intro.mk.zero
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q : Associates M
      hq : Ne q 0
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hb : LT.lt (c ⟨0, hi⟩) (c 1)
      ⊢ Eq (c ⟨0, hi⟩) Bot.bot
    -/
  · exact (Associates.isUnit_iff_eq_one _).mp (first_of_chain_isUnit h₁ @h₂)
    /-
      🎉 no goals
    -/
    /-
      case succ.refine_2.intro.mk.succ
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q : Associates M
      hq : Ne q 0
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      n✝ : Nat
      hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hb : LT.lt (c ⟨HAdd.hAdd n✝ 1, hi⟩) (c 1)
      ⊢ Eq (c ⟨HAdd.hAdd n✝ 1, hi⟩) Bot.bot
    -/
  · simpa [Fin.lt_iff_val_lt_val] using h₁.lt_iff_lt.mp hb
    /-
      🎉 no goals
    -/


theorem eq_second_of_chain_of_prime_dvd {p q r : Associates M} {n : ℕ} (hn : n ≠ 0)
    {c : Fin (n + 1) → Associates M} (h₁ : StrictMono c)
    (h₂ : ∀ {r : Associates M}, r ≤ q ↔ ∃ i, r = c i) (hp : Prime p) (hr : r ∣ q) (hp' : p ∣ r) :
    p = c 1 := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q r : Associates M
    n : Nat
    hn : Ne n 0
    c : Fin (HAdd.hAdd n 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    hp : Prime p
    hr : Dvd.dvd r q
    hp' : Dvd.dvd p r
    ⊢ Eq p (c 1)
  -/
  cases' n with n
    /-
      case zero
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p q r : Associates M
      hp : Prime p
      hr : Dvd.dvd r q
      hp' : Dvd.dvd p r
      hn : Ne 0 0
      c : Fin (HAdd.hAdd 0 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      ⊢ Eq p (c 1)
    -/
  · contradiction
    /-
      🎉 no goals
    -/
  /-
    case succ
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q r : Associates M
    hp : Prime p
    hr : Dvd.dvd r q
    hp' : Dvd.dvd p r
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    ⊢ Eq p (c 1)
  -/
  obtain ⟨i, rfl⟩ := h₂.1 (dvd_trans hp' hr)
  /-
    case succ.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q r : Associates M
    hr : Dvd.dvd r q
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hp : Prime (c i)
    hp' : Dvd.dvd (c i) r
    ⊢ Eq (c i) (c 1)
  -/
  refine congr_arg c (eq_of_ge_of_not_gt ?_ fun hi => ?_)
  · rw [Fin.le_iff_val_le_val, Fin.val_one, Nat.succ_le_iff, ← Fin.val_zero' (n.succ + 1), ←
      Fin.lt_iff_val_lt_val, Fin.pos_iff_ne_zero]
    /-
      case succ.intro.refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q r : Associates M
      hr : Dvd.dvd r q
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hp : Prime (c i)
      hp' : Dvd.dvd (c i) r
      ⊢ Ne i 0
    -/
    rintro rfl
    /-
      case succ.intro.refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q r : Associates M
      hr : Dvd.dvd r q
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      hp : Prime (c 0)
      hp' : Dvd.dvd (c 0) r
      ⊢ False
    -/
    exact hp.not_unit (first_of_chain_isUnit h₁ @h₂)
    /-
      🎉 no goals
    -/
  /-
    case succ.intro.refine_2
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q r : Associates M
    hr : Dvd.dvd r q
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
    h₁ : StrictMono c
    h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
    i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hp : Prime (c i)
    hp' : Dvd.dvd (c i) r
    hi : LT.lt 1 i
    ⊢ False
  -/
  obtain rfl | ⟨j, rfl⟩ := i.eq_zero_or_eq_succ
    /-
      case succ.intro.refine_2.inl
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q r : Associates M
      hr : Dvd.dvd r q
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      hp : Prime (c 0)
      hp' : Dvd.dvd (c 0) r
      hi : LT.lt 1 0
      ⊢ False
    -/
  · cases hi
    /-
      🎉 no goals
    -/
  refine
    not_irreducible_of_not_unit_dvdNotUnit
      (DvdNotUnit.not_unit
        (Associates.dvdNotUnit_iff_lt.2 (h₁ (show (0 : Fin (n + 2)) < j from ?_))))
      ?_ hp.irreducible
    /-
      case succ.intro.refine_2.inr.intro.refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q r : Associates M
      hr : Dvd.dvd r q
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      j : Fin (HAdd.hAdd n 1)
      hp : Prime (c j.succ)
      hp' : Dvd.dvd (c j.succ) r
      hi : LT.lt 1 j.succ
      ⊢ LT.lt 0 ↑↑j
    -/
  · simpa [Fin.succ_lt_succ_iff, Fin.lt_iff_val_lt_val] using hi
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.refine_2.inr.intro.refine_2
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q r : Associates M
      hr : Dvd.dvd r q
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      j : Fin (HAdd.hAdd n 1)
      hp : Prime (c j.succ)
      hp' : Dvd.dvd (c j.succ) r
      hi : LT.lt 1 j.succ
      ⊢ DvdNotUnit (c ↑↑j) (c j.succ)
    -/
  · refine Associates.dvdNotUnit_iff_lt.2 (h₁ ?_)
    /-
      case succ.intro.refine_2.inr.intro.refine_2
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q r : Associates M
      hr : Dvd.dvd r q
      n : Nat
      hn : Ne (HAdd.hAdd n 1) 0
      c : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → Associates M
      h₁ : StrictMono c
      h₂ : ∀ {r : Associates M}, Iff (LE.le r q) (Exists fun i => Eq r (c i))
      j : Fin (HAdd.hAdd n 1)
      hp : Prime (c j.succ)
      hp' : Dvd.dvd (c j.succ) r
      hi : LT.lt 1 j.succ
      ⊢ LT.lt (↑↑j) j.succ
    -/
    simpa only [Fin.coe_eq_castSucc] using Fin.lt_succ
    /-
      🎉 no goals
    -/


theorem card_subset_divisors_le_length_of_chain {q : Associates M} {n : ℕ}
    {c : Fin (n + 1) → Associates M} (h₂ : ∀ {r}, r ≤ q ↔ ∃ i, r = c i) {m : Finset (Associates M)}
    (hm : ∀ r, r ∈ m → r ≤ q) : m.card ≤ n + 1 := by
  classical
    have mem_image : ∀ r : Associates M, r ≤ q → r ∈ Finset.univ.image c := by
      intro r hr
      obtain ⟨i, hi⟩ := h₂.1 hr
      exact Finset.mem_image.2 ⟨i, Finset.mem_univ _, hi.symm⟩
    rw [← Finset.card_fin (n + 1)]
    exact (Finset.card_le_card fun x hx => mem_image x <| hm x hx).trans Finset.card_image_le


theorem element_of_chain_eq_pow_second_of_chain {q r : Associates M} {n : ℕ} (hn : n ≠ 0)
    {c : Fin (n + 1) → Associates M} (h₁ : StrictMono c) (h₂ : ∀ {r}, r ≤ q ↔ ∃ i, r = c i)
    (hr : r ∣ q) (hq : q ≠ 0) : ∃ i : Fin (n + 1), r = c 1 ^ (i : ℕ) := by
  classical
    let i := Multiset.card (normalizedFactors r)
    have hi : normalizedFactors r = Multiset.replicate i (c 1) := by
      apply Multiset.eq_replicate_of_mem
      intro b hb
      refine
        eq_second_of_chain_of_prime_dvd hn h₁ (@fun r' => h₂) (prime_of_normalized_factor b hb) hr
          (dvd_of_mem_normalizedFactors hb)
    have H : r = c 1 ^ i := by
      have := UniqueFactorizationMonoid.prod_normalizedFactors (ne_zero_of_dvd_ne_zero hq hr)
      rw [associated_iff_eq, hi, Multiset.prod_replicate] at this
      rw [this]
    refine ⟨⟨i, ?_⟩, H⟩
    have : (Finset.univ.image fun m : Fin (i + 1) => c 1 ^ (m : ℕ)).card = i + 1 := by
      conv_rhs => rw [← Finset.card_fin (i + 1)]
      cases n
      · contradiction
      rw [Finset.card_image_iff]
      refine Set.injOn_of_injective (fun m m' h => Fin.ext ?_)
      refine
        pow_injective_of_not_isUnit (element_of_chain_not_isUnit_of_index_ne_zero (by simp) h₁) ?_ h
      exact Irreducible.ne_zero (second_of_chain_is_irreducible hn h₁ (@h₂) hq)
    suffices H' : ∀ r ∈ Finset.univ.image fun m : Fin (i + 1) => c 1 ^ (m : ℕ), r ≤ q by
      simp only [← Nat.succ_le_iff, Nat.succ_eq_add_one, ← this]
      apply card_subset_divisors_le_length_of_chain (@h₂) H'
    simp only [Finset.mem_image]
    rintro r ⟨a, _, rfl⟩
    refine dvd_trans ?_ hr
    use c 1 ^ (i - (a : ℕ))
    rw [pow_mul_pow_sub (c 1)]
    · exact H
    · exact Nat.succ_le_succ_iff.mp a.2


theorem eq_pow_second_of_chain_of_has_chain {q : Associates M} {n : ℕ} (hn : n ≠ 0)
    {c : Fin (n + 1) → Associates M} (h₁ : StrictMono c)
    (h₂ : ∀ {r : Associates M}, r ≤ q ↔ ∃ i, r = c i) (hq : q ≠ 0) : q = c 1 ^ n := by
  classical
    obtain ⟨i, hi'⟩ := element_of_chain_eq_pow_second_of_chain hn h₁ (@fun r => h₂) (dvd_refl q) hq
    convert hi'
    refine (Nat.lt_succ_iff.1 i.prop).antisymm' (Nat.le_of_succ_le_succ ?_)
    calc
      n + 1 = (Finset.univ : Finset (Fin (n + 1))).card := (Finset.card_fin _).symm
      _ = (Finset.univ.image c).card := (Finset.card_image_iff.mpr h₁.injective.injOn).symm
      _ ≤ (Finset.univ.image fun m : Fin (i + 1) => c 1 ^ (m : ℕ)).card :=
        (Finset.card_le_card ?_)
      _ ≤ (Finset.univ : Finset (Fin (i + 1))).card := Finset.card_image_le
      _ = i + 1 := Finset.card_fin _
    intro r hr
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.1 hr
    have := h₂.2 ⟨j, rfl⟩
    rw [hi'] at this
    have h := (dvd_prime_pow (show Prime (c 1) from ?_) i).1 this
    · rcases h with ⟨u, hu, hu'⟩
      refine Finset.mem_image.mpr ⟨u, Finset.mem_univ _, ?_⟩
      rw [associated_iff_eq] at hu'
      rw [Fin.val_cast_of_lt (Nat.lt_succ_of_le hu), hu']
    · rw [← irreducible_iff_prime]
      exact second_of_chain_is_irreducible hn h₁ (@h₂) hq


theorem isPrimePow_of_has_chain {q : Associates M} {n : ℕ} (hn : n ≠ 0)
    {c : Fin (n + 1) → Associates M} (h₁ : StrictMono c)
    (h₂ : ∀ {r : Associates M}, r ≤ q ↔ ∃ i, r = c i) (hq : q ≠ 0) : IsPrimePow q :=
  ⟨c 1, n, irreducible_iff_prime.mp (second_of_chain_is_irreducible hn h₁ (@h₂) hq),
    zero_lt_iff.mpr hn, (eq_pow_second_of_chain_of_has_chain hn h₁ (@h₂) hq).symm⟩


theorem factor_orderIso_map_one_eq_bot {m : Associates M} {n : Associates N}
    (d : { l : Associates M // l ≤ m } ≃o { l : Associates N // l ≤ n }) :
    (d ⟨1, one_dvd m⟩ : Associates N) = 1 := by
  /-
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝ : CancelCommMonoidWithZero N
    m : Associates M
    n : Associates N
    d : OrderIso (Subtype fun l => LE.le l m) (Subtype fun l => LE.le l n)
    ⊢ Eq (↑(d ⟨1, ⋯⟩)) 1
  -/
  letI : OrderBot { l : Associates M // l ≤ m } := Subtype.orderBot bot_le
  /-
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝ : CancelCommMonoidWithZero N
    m : Associates M
    n : Associates N
    d : OrderIso (Subtype fun l => LE.le l m) (Subtype fun l => LE.le l n)
    this : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
    ⊢ Eq (↑(d ⟨1, ⋯⟩)) 1
  -/
  letI : OrderBot { l : Associates N // l ≤ n } := Subtype.orderBot bot_le
  /-
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝ : CancelCommMonoidWithZero N
    m : Associates M
    n : Associates N
    d : OrderIso (Subtype fun l => LE.le l m) (Subtype fun l => LE.le l n)
    this✝ : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
    this : OrderBot (Subtype fun l => LE.le l n) := Subtype.orderBot ⋯
    ⊢ Eq (↑(d ⟨1, ⋯⟩)) 1
  -/
  simp only [← Associates.bot_eq_one, Subtype.mk_bot, bot_le, Subtype.coe_eq_bot_iff]
  /-
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝ : CancelCommMonoidWithZero N
    m : Associates M
    n : Associates N
    d : OrderIso (Subtype fun l => LE.le l m) (Subtype fun l => LE.le l n)
    this✝ : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
    this : OrderBot (Subtype fun l => LE.le l n) := Subtype.orderBot ⋯
    ⊢ Eq (d Bot.bot) Bot.bot
  -/
  letI : BotHomClass ({ l // l ≤ m } ≃o { l // l ≤ n }) _ _ := OrderIsoClass.toBotHomClass
  /-
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝ : CancelCommMonoidWithZero N
    m : Associates M
    n : Associates N
    d : OrderIso (Subtype fun l => LE.le l m) (Subtype fun l => LE.le l n)
    this✝¹ : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
    this✝ : OrderBot (Subtype fun l => LE.le l n) := Subtype.orderBot ⋯
    this : BotHomClass (OrderIso (Subtype fun l => LE.le l m) (Subtype fun l => LE …
    ⊢ Eq (d Bot.bot) Bot.bot
  -/
  exact map_bot d
  /-
    🎉 no goals
  -/


theorem coe_factor_orderIso_map_eq_one_iff {m u : Associates M} {n : Associates N} (hu' : u ≤ m)
    (d : Set.Iic m ≃o Set.Iic n) : (d ⟨u, hu'⟩ : Associates N) = 1 ↔ u = 1 :=
  ⟨fun hu => by
    rw [show u = (d.symm ⟨d ⟨u, hu'⟩, (d ⟨u, hu'⟩).prop⟩) by
        simp only [Subtype.coe_eta, OrderIso.symm_apply_apply, Subtype.coe_mk]]
    /-
      M : Type u_1
      inst✝¹ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝ : CancelCommMonoidWithZero N
      m u : Associates M
      n : Associates N
      hu' : LE.le u m
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hu : Eq (↑(d ⟨u, hu'⟩)) 1
      ⊢ Eq (↑(d.symm ⟨↑(d ⟨u, hu'⟩), ⋯⟩)) 1
    -/
    conv_rhs => rw [← factor_orderIso_map_one_eq_bot d.symm]
    /-
      M : Type u_1
      inst✝¹ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝ : CancelCommMonoidWithZero N
      m u : Associates M
      n : Associates N
      hu' : LE.le u m
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hu : Eq (↑(d ⟨u, hu'⟩)) 1
      ⊢ Eq ↑(d.symm ⟨↑(d ⟨u, hu'⟩), ⋯⟩) ↑(d.symm ⟨1, ⋯⟩)
    -/
    congr, fun hu => by
    /-
      🎉 no goals
    -/
    /-
      M : Type u_1
      inst✝¹ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝ : CancelCommMonoidWithZero N
      m u : Associates M
      n : Associates N
      hu' : LE.le u m
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hu : Eq u 1
      ⊢ Eq (↑(d ⟨u, hu'⟩)) 1
    -/
    simp_rw [hu]
    /-
      M : Type u_1
      inst✝¹ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝ : CancelCommMonoidWithZero N
      m u : Associates M
      n : Associates N
      hu' : LE.le u m
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hu : Eq u 1
      ⊢ Eq (↑(d ⟨1, ⋯⟩)) 1
    -/
    conv_rhs => rw [← factor_orderIso_map_one_eq_bot d]
    /-
      M : Type u_1
      inst✝¹ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝ : CancelCommMonoidWithZero N
      m u : Associates M
      n : Associates N
      hu' : LE.le u m
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hu : Eq u 1
      ⊢ Eq ↑(d ⟨1, ⋯⟩) ↑(d ⟨1, ⋯⟩)
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


theorem pow_image_of_prime_by_factor_orderIso_dvd
    {m p : Associates M} {n : Associates N} (hn : n ≠ 0) (hp : p ∈ normalizedFactors m)
    (d : Set.Iic m ≃o Set.Iic n) {s : ℕ} (hs' : p ^ s ≤ m) :
    (d ⟨p, dvd_of_mem_normalizedFactors hp⟩ : Associates N) ^ s ≤ n := by
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    s : Nat
    hs' : LE.le (HPow.hPow p s) m
    ⊢ LE.le (HPow.hPow (↑(d ⟨p, ⋯⟩)) s) n
  -/
  by_cases hs : s = 0
    /-
      case pos
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs' : LE.le (HPow.hPow p s) m
      hs : Eq s 0
      ⊢ LE.le (HPow.hPow (↑(d ⟨p, ⋯⟩)) s) n
    -/
  · simp [← Associates.bot_eq_one, hs]
    /-
      🎉 no goals
    -/
  suffices (d ⟨p, dvd_of_mem_normalizedFactors hp⟩ : Associates N) ^ s =
      (d ⟨p ^ s, hs'⟩) by
    rw [this]
    apply Subtype.prop (d ⟨p ^ s, hs'⟩)
  /-
    case neg
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    s : Nat
    hs' : LE.le (HPow.hPow p s) m
    hs : Not (Eq s 0)
    ⊢ Eq (HPow.hPow (↑(d ⟨p, ⋯⟩)) s) ↑(d ⟨HPow.hPow p s, hs'⟩)
  -/
  obtain ⟨c₁, rfl, hc₁', hc₁''⟩ := exists_chain_of_prime_pow hs (prime_of_normalized_factor p hp)
  /-
    case neg.intro.intro.intro
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m : Associates M
    n : Associates N
    hn : Ne n 0
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    s : Nat
    hs : Not (Eq s 0)
    c₁ : Fin (HAdd.hAdd s 1) → Associates M
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
    hs' : LE.le (HPow.hPow (c₁ 1) s) m
    hc₁' : StrictMono c₁
    hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
    ⊢ Eq (HPow.hPow (↑(d ⟨c₁ 1, ⋯⟩)) s) ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
  -/
  let c₂ : Fin (s + 1) → Associates N := fun t => d ⟨c₁ t, le_trans (hc₁''.2 ⟨t, by simp⟩) hs'⟩
  /-
    case neg.intro.intro.intro
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m : Associates M
    n : Associates N
    hn : Ne n 0
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    s : Nat
    hs : Not (Eq s 0)
    c₁ : Fin (HAdd.hAdd s 1) → Associates M
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
    hs' : LE.le (HPow.hPow (c₁ 1) s) m
    hc₁' : StrictMono c₁
    hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
    c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
    ⊢ Eq (HPow.hPow (↑(d ⟨c₁ 1, ⋯⟩)) s) ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
  -/
  have c₂_def : ∀ t, c₂ t = d ⟨c₁ t, _⟩ := fun t => rfl
  /-
    case neg.intro.intro.intro
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m : Associates M
    n : Associates N
    hn : Ne n 0
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    s : Nat
    hs : Not (Eq s 0)
    c₁ : Fin (HAdd.hAdd s 1) → Associates M
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
    hs' : LE.le (HPow.hPow (c₁ 1) s) m
    hc₁' : StrictMono c₁
    hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
    c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
    c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
    ⊢ Eq (HPow.hPow (↑(d ⟨c₁ 1, ⋯⟩)) s) ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
  -/
  rw [← c₂_def]
  refine (eq_pow_second_of_chain_of_has_chain hs (fun t u h => ?_)
    (@fun r => ⟨@fun hr => ?_, ?_⟩) ?_).symm
    /-
      case neg.intro.intro.intro.refine_1
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      t u : Fin (HAdd.hAdd s 1)
      h : LT.lt t u
      ⊢ LT.lt (c₂ t) (c₂ u)
    -/
  · rw [c₂_def, c₂_def, Subtype.coe_lt_coe, d.lt_iff_lt, Subtype.mk_lt_mk, hc₁'.lt_iff_lt]
    /-
      case neg.intro.intro.intro.refine_1
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      t u : Fin (HAdd.hAdd s 1)
      h : LT.lt t u
      ⊢ LT.lt t u
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.refine_2
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      hr : LE.le r ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
      ⊢ Exists fun i => Eq r (c₂ i)
    -/
  · have : r ≤ n := hr.trans (d ⟨c₁ 1 ^ s, _⟩).2
    suffices d.symm ⟨r, this⟩ ≤ ⟨c₁ 1 ^ s, hs'⟩ by
      obtain ⟨i, hi⟩ := hc₁''.1 this
      use i
      simp only [c₂_def, ← hi, d.apply_symm_apply, Subtype.coe_eta, Subtype.coe_mk]
    /-
      case neg.intro.intro.intro.refine_2
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      hr : LE.le r ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
      this : LE.le r n
      ⊢ LE.le (d.symm ⟨r, this⟩) ⟨HPow.hPow (c₁ 1) s, hs'⟩
    -/
    conv_rhs => rw [← d.symm_apply_apply ⟨c₁ 1 ^ s, hs'⟩]
    /-
      case neg.intro.intro.intro.refine_2
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      hr : LE.le r ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
      this : LE.le r n
      ⊢ LE.le (d.symm ⟨r, this⟩) (d.symm (d ⟨HPow.hPow (c₁ 1) s, hs'⟩))
    -/
    rw [d.symm.le_iff_le]
    /-
      case neg.intro.intro.intro.refine_2
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      hr : LE.le r ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
      this : LE.le r n
      ⊢ LE.le ⟨r, this⟩ (d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
    -/
    simpa only [← Subtype.coe_le_coe, Subtype.coe_mk] using hr
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.refine_3
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      ⊢ (Exists fun i => Eq r (c₂ i)) → LE.le r ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
    -/
  · rintro ⟨i, hr⟩
    /-
      case neg.intro.intro.intro.refine_3.intro
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      i : Fin (HAdd.hAdd s 1)
      hr : Eq r (c₂ i)
      ⊢ LE.le r ↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)
    -/
    rw [hr, c₂_def, Subtype.coe_le_coe, d.le_iff_le]
    /-
      case neg.intro.intro.intro.refine_3.intro
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      s : Nat
      hs : Not (Eq s 0)
      c₁ : Fin (HAdd.hAdd s 1) → Associates M
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
      hs' : LE.le (HPow.hPow (c₁ 1) s) m
      hc₁' : StrictMono c₁
      hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
      c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
      c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
      r : Associates N
      i : Fin (HAdd.hAdd s 1)
      hr : Eq r (c₂ i)
      ⊢ LE.le ⟨c₁ i, ⋯⟩ ⟨HPow.hPow (c₁ 1) s, hs'⟩
    -/
    simpa [Subtype.mk_le_mk] using hc₁''.2 ⟨i, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case neg.intro.intro.intro.refine_4
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m : Associates M
    n : Associates N
    hn : Ne n 0
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    s : Nat
    hs : Not (Eq s 0)
    c₁ : Fin (HAdd.hAdd s 1) → Associates M
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) (c₁ 1)
    hs' : LE.le (HPow.hPow (c₁ 1) s) m
    hc₁' : StrictMono c₁
    hc₁'' : ∀ {r : Associates M}, Iff (LE.le r (HPow.hPow (c₁ 1) s)) (Exists fun i …
    c₂ : Fin (HAdd.hAdd s 1) → Associates N := fun t => ↑(d ⟨c₁ t, ⋯⟩)
    c₂_def : ∀ (t : Fin (HAdd.hAdd s 1)), Eq (c₂ t) ↑(d ⟨c₁ t, ⋯⟩)
    ⊢ Ne (↑(d ⟨HPow.hPow (c₁ 1) s, hs'⟩)) 0
  -/
  exact ne_zero_of_dvd_ne_zero hn (Subtype.prop (d ⟨c₁ 1 ^ s, _⟩))
  /-
    🎉 no goals
  -/


theorem map_prime_of_factor_orderIso {m p : Associates M} {n : Associates N} (hn : n ≠ 0)
    (hp : p ∈ normalizedFactors m) (d : Set.Iic m ≃o Set.Iic n) :
    Prime (d ⟨p, dvd_of_mem_normalizedFactors hp⟩ : Associates N) := by
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    ⊢ Prime ↑(d ⟨p, ⋯⟩)
  -/
  rw [← irreducible_iff_prime]
  refine (Associates.isAtom_iff <|
    ne_zero_of_dvd_ne_zero hn (d ⟨p, _⟩).prop).mp ⟨?_, fun b hb => ?_⟩
  · rw [Ne, ← Associates.isUnit_iff_eq_bot, Associates.isUnit_iff_eq_one,
      coe_factor_orderIso_map_eq_one_iff _ d]
    /-
      case refine_1
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      ⊢ Not (Eq p 1)
    -/
    rintro rfl
    /-
      case refine_1
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m : Associates M
      n : Associates N
      hn : Ne n 0
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) 1
      ⊢ False
    -/
    exact (prime_of_normalized_factor 1 hp).not_unit isUnit_one
    /-
      🎉 no goals
    -/
  · obtain ⟨x, hx⟩ :=
      d.surjective ⟨b, le_trans (le_of_lt hb) (d ⟨p, dvd_of_mem_normalizedFactors hp⟩).prop⟩
    /-
      case refine_2.intro
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      b : Associates N
      hb : LT.lt b ↑(d ⟨p, ⋯⟩)
      x : ↑(Set.Iic m)
      hx : Eq (d x) ⟨b, ⋯⟩
      ⊢ Eq b Bot.bot
    -/
    rw [← Subtype.coe_mk b _, ← hx] at hb
    /-
      case refine_2.intro
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      b : Associates N
      hb✝ : LT.lt b ↑(d ⟨p, ⋯⟩)
      x : ↑(Set.Iic m)
      hb : LT.lt ↑(d x) ↑(d ⟨p, ⋯⟩)
      hx : Eq (d x) ⟨b, ⋯⟩
      ⊢ Eq b Bot.bot
    -/
    letI : OrderBot { l : Associates M // l ≤ m } := Subtype.orderBot bot_le
    /-
      case refine_2.intro
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      b : Associates N
      hb✝ : LT.lt b ↑(d ⟨p, ⋯⟩)
      x : ↑(Set.Iic m)
      hb : LT.lt ↑(d x) ↑(d ⟨p, ⋯⟩)
      hx : Eq (d x) ⟨b, ⋯⟩
      this : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
      ⊢ Eq b Bot.bot
    -/
    letI : OrderBot { l : Associates N // l ≤ n } := Subtype.orderBot bot_le
    suffices x = ⊥ by
      rw [this, OrderIso.map_bot d] at hx
      refine (Subtype.mk_eq_bot_iff ?_ _).mp hx.symm
      simp
    /-
      case refine_2.intro
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      b : Associates N
      hb✝ : LT.lt b ↑(d ⟨p, ⋯⟩)
      x : ↑(Set.Iic m)
      hb : LT.lt ↑(d x) ↑(d ⟨p, ⋯⟩)
      hx : Eq (d x) ⟨b, ⋯⟩
      this✝ : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
      this : OrderBot (Subtype fun l => LE.le l n) := Subtype.orderBot ⋯
      ⊢ Eq x Bot.bot
    -/
    obtain ⟨a, ha⟩ := x
    /-
      case refine_2.intro.mk
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      b : Associates N
      hb✝ : LT.lt b ↑(d ⟨p, ⋯⟩)
      this✝ : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
      this : OrderBot (Subtype fun l => LE.le l n) := Subtype.orderBot ⋯
      a : Associates M
      ha : Membership.mem (Set.Iic m) a
      hb : LT.lt ↑(d ⟨a, ha⟩) ↑(d ⟨p, ⋯⟩)
      hx : Eq (d ⟨a, ha⟩) ⟨b, ⋯⟩
      ⊢ Eq ⟨a, ha⟩ Bot.bot
    -/
    rw [Subtype.mk_eq_bot_iff]
    · exact
        ((Associates.isAtom_iff <| Prime.ne_zero <| prime_of_normalized_factor p hp).mpr <|
              irreducible_of_normalized_factor p hp).right
          a (Subtype.mk_lt_mk.mp <| d.lt_iff_lt.mp hb)
    /-
      case refine_2.intro.mk.hbot
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hn : Ne n 0
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      b : Associates N
      hb✝ : LT.lt b ↑(d ⟨p, ⋯⟩)
      this✝ : OrderBot (Subtype fun l => LE.le l m) := Subtype.orderBot ⋯
      this : OrderBot (Subtype fun l => LE.le l n) := Subtype.orderBot ⋯
      a : Associates M
      ha : Membership.mem (Set.Iic m) a
      hb : LT.lt ↑(d ⟨a, ha⟩) ↑(d ⟨p, ⋯⟩)
      hx : Eq (d ⟨a, ha⟩) ⟨b, ⋯⟩
      ⊢ Membership.mem (Set.Iic m) Bot.bot
    -/
    simp
    /-
      🎉 no goals
    -/


theorem mem_normalizedFactors_factor_orderIso_of_mem_normalizedFactors {m p : Associates M}
    {n : Associates N} (hn : n ≠ 0) (hp : p ∈ normalizedFactors m) (d : Set.Iic m ≃o Set.Iic n) :
    (d ⟨p, dvd_of_mem_normalizedFactors hp⟩ : Associates N) ∈ normalizedFactors n := by
  obtain ⟨q, hq, hq'⟩ :=
    exists_mem_normalizedFactors_of_dvd hn (map_prime_of_factor_orderIso hn hp d).irreducible
      (d ⟨p, dvd_of_mem_normalizedFactors hp⟩).prop
  /-
    case intro.intro
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    q : Associates N
    hq : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n) q
    hq' : Associated (↑(d ⟨p, ⋯⟩)) q
    ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors n) ↑(d ⟨p, ⋯⟩)
  -/
  rw [associated_iff_eq] at hq'
  /-
    case intro.intro
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    q : Associates N
    hq : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n) q
    hq' : Eq (↑(d ⟨p, ⋯⟩)) q
    ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors n) ↑(d ⟨p, ⋯⟩)
  -/
  rwa [hq']
  /-
    🎉 no goals
  -/


theorem emultiplicity_prime_le_emultiplicity_image_by_factor_orderIso {m p : Associates M}
    {n : Associates N} (hp : p ∈ normalizedFactors m) (d : Set.Iic m ≃o Set.Iic n) :
    emultiplicity p m ≤ emultiplicity (↑(d ⟨p, dvd_of_mem_normalizedFactors hp⟩)) n := by
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    ⊢ LE.le (emultiplicity p m) (emultiplicity (↑(d ⟨p, ⋯⟩)) n)
  -/
  by_cases hn : n = 0
    /-
      case pos
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hn : Eq n 0
      ⊢ LE.le (emultiplicity p m) (emultiplicity (↑(d ⟨p, ⋯⟩)) n)
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  /-
    case neg
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    hn : Not (Eq n 0)
    ⊢ LE.le (emultiplicity p m) (emultiplicity (↑(d ⟨p, ⋯⟩)) n)
  -/
  by_cases hm : m = 0
    /-
      case pos
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : UniqueFactorizationMonoid N
      inst✝ : UniqueFactorizationMonoid M
      m p : Associates M
      n : Associates N
      hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
      d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
      hn : Not (Eq n 0)
      hm : Eq m 0
      ⊢ LE.le (emultiplicity p m) (emultiplicity (↑(d ⟨p, ⋯⟩)) n)
    -/
  · simp [hm] at hp
    /-
      🎉 no goals
    -/
  rw [FiniteMultiplicity.of_prime_left (prime_of_normalized_factor p hp) hm
    |>.emultiplicity_eq_multiplicity, ← pow_dvd_iff_le_emultiplicity]
  /-
    case neg
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    hn : Not (Eq n 0)
    hm : Not (Eq m 0)
    ⊢ Dvd.dvd (HPow.hPow (↑(d ⟨p, ⋯⟩)) (multiplicity p m)) n
  -/
  apply pow_image_of_prime_by_factor_orderIso_dvd hn hp d (pow_multiplicity_dvd ..)
  /-
    🎉 no goals
  -/


theorem emultiplicity_prime_eq_emultiplicity_image_by_factor_orderIso {m p : Associates M}
    {n : Associates N} (hn : n ≠ 0) (hp : p ∈ normalizedFactors m) (d : Set.Iic m ≃o Set.Iic n) :
    emultiplicity p m = emultiplicity (↑(d ⟨p, dvd_of_mem_normalizedFactors hp⟩)) n := by
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    ⊢ Eq (emultiplicity p m) (emultiplicity (↑(d ⟨p, ⋯⟩)) n)
  -/
  refine le_antisymm (emultiplicity_prime_le_emultiplicity_image_by_factor_orderIso hp d) ?_
  suffices emultiplicity (↑(d ⟨p, dvd_of_mem_normalizedFactors hp⟩)) n ≤
      emultiplicity (↑(d.symm (d ⟨p, dvd_of_mem_normalizedFactors hp⟩))) m by
    rw [d.symm_apply_apply ⟨p, dvd_of_mem_normalizedFactors hp⟩, Subtype.coe_mk] at this
    exact this
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝² : CancelCommMonoidWithZero N
    inst✝¹ : UniqueFactorizationMonoid N
    inst✝ : UniqueFactorizationMonoid M
    m p : Associates M
    n : Associates N
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : OrderIso ↑(Set.Iic m) ↑(Set.Iic n)
    ⊢ LE.le (emultiplicity (↑(d ⟨p, ⋯⟩)) n) (emultiplicity (↑(d.symm (d ⟨p, ⋯⟩))) m)
  -/
  letI := Classical.decEq (Associates N)
  simpa only [Subtype.coe_eta] using
    emultiplicity_prime_le_emultiplicity_image_by_factor_orderIso
      (mem_normalizedFactors_factor_orderIso_of_mem_normalizedFactors hn hp d) d.symm


/-- The order isomorphism between the factors of `mk m` and the factors of `mk n` induced by a
  bijection between the factors of `m` and the factors of `n` that preserves `∣`. -/
@[simps]
def mkFactorOrderIsoOfFactorDvdEquiv {m : M} {n : N} {d : { l : M // l ∣ m } ≃ { l : N // l ∣ n }}
    (hd : ∀ l l', (d l : N) ∣ d l' ↔ (l : M) ∣ (l' : M)) :
    Set.Iic (Associates.mk m) ≃o Set.Iic (Associates.mk n) where
  toFun l :=
    ⟨Associates.mk
        (d
          ⟨associatesEquivOfUniqueUnits ↑l, by
            /-
              M : Type u_1
              inst✝³ : CancelCommMonoidWithZero M
              N : Type u_2
              inst✝² : CancelCommMonoidWithZero N
              inst✝¹ : Subsingleton (Units M)
              inst✝ : Subsingleton (Units N)
              m : M
              n : N
              d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
              hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
              l : ↑(Set.Iic (Associates.mk m))
              ⊢ Dvd.dvd (associatesEquivOfUniqueUnits ↑l) m
            -/
            obtain ⟨x, hx⟩ := l
            /-
              case mk
              M : Type u_1
              inst✝³ : CancelCommMonoidWithZero M
              N : Type u_2
              inst✝² : CancelCommMonoidWithZero N
              inst✝¹ : Subsingleton (Units M)
              inst✝ : Subsingleton (Units N)
              m : M
              n : N
              d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
              hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
              x : Associates M
              hx : Membership.mem (Set.Iic (Associates.mk m)) x
              ⊢ Dvd.dvd (associatesEquivOfUniqueUnits ↑⟨x, hx⟩) m
            -/
            rw [Subtype.coe_mk, associatesEquivOfUniqueUnits_apply, out_dvd_iff]
            /-
              case mk
              M : Type u_1
              inst✝³ : CancelCommMonoidWithZero M
              N : Type u_2
              inst✝² : CancelCommMonoidWithZero N
              inst✝¹ : Subsingleton (Units M)
              inst✝ : Subsingleton (Units N)
              m : M
              n : N
              d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
              hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
              x : Associates M
              hx : Membership.mem (Set.Iic (Associates.mk m)) x
              ⊢ LE.le x (Associates.mk m)
            -/
            exact hx⟩),
            /-
              🎉 no goals
            -/
      mk_le_mk_iff_dvd.mpr (Subtype.prop (d ⟨associatesEquivOfUniqueUnits ↑l, _⟩))⟩
  invFun l :=
    ⟨Associates.mk
        (d.symm
          ⟨associatesEquivOfUniqueUnits ↑l, by
            /-
              M : Type u_1
              inst✝³ : CancelCommMonoidWithZero M
              N : Type u_2
              inst✝² : CancelCommMonoidWithZero N
              inst✝¹ : Subsingleton (Units M)
              inst✝ : Subsingleton (Units N)
              m : M
              n : N
              d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
              hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
              l : ↑(Set.Iic (Associates.mk n))
              ⊢ Dvd.dvd (associatesEquivOfUniqueUnits ↑l) n
            -/
            obtain ⟨x, hx⟩ := l
            /-
              case mk
              M : Type u_1
              inst✝³ : CancelCommMonoidWithZero M
              N : Type u_2
              inst✝² : CancelCommMonoidWithZero N
              inst✝¹ : Subsingleton (Units M)
              inst✝ : Subsingleton (Units N)
              m : M
              n : N
              d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
              hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
              x : Associates N
              hx : Membership.mem (Set.Iic (Associates.mk n)) x
              ⊢ Dvd.dvd (associatesEquivOfUniqueUnits ↑⟨x, hx⟩) n
            -/
            rw [Subtype.coe_mk, associatesEquivOfUniqueUnits_apply, out_dvd_iff]
            /-
              case mk
              M : Type u_1
              inst✝³ : CancelCommMonoidWithZero M
              N : Type u_2
              inst✝² : CancelCommMonoidWithZero N
              inst✝¹ : Subsingleton (Units M)
              inst✝ : Subsingleton (Units N)
              m : M
              n : N
              d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
              hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
              x : Associates N
              hx : Membership.mem (Set.Iic (Associates.mk n)) x
              ⊢ LE.le x (Associates.mk n)
            -/
            exact hx⟩),
            /-
              🎉 no goals
            -/
      mk_le_mk_iff_dvd.mpr (Subtype.prop (d.symm ⟨associatesEquivOfUniqueUnits ↑l, _⟩))⟩
  left_inv := fun ⟨l, hl⟩ => by
    simp only [Subtype.coe_eta, Equiv.symm_apply_apply, Subtype.coe_mk,
      associatesEquivOfUniqueUnits_apply, mk_out, out_mk, normalize_eq]
  right_inv := fun ⟨l, hl⟩ => by
    simp only [Subtype.coe_eta, Equiv.apply_symm_apply, Subtype.coe_mk,
      associatesEquivOfUniqueUnits_apply, out_mk, normalize_eq, mk_out]
  map_rel_iff' := by
    /-
      M : Type u_1
      inst✝³ : CancelCommMonoidWithZero M
      N : Type u_2
      inst✝² : CancelCommMonoidWithZero N
      inst✝¹ : Subsingleton (Units M)
      inst✝ : Subsingleton (Units N)
      m : M
      n : N
      d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
      hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
      ⊢ ∀ {a b : ↑(Set.Iic (Associates.mk m))}, Iff (LE.le ({ toFun := fun l => ⟨Ass …
    -/
    rintro ⟨a, ha⟩ ⟨b, hb⟩
    simp only [Equiv.coe_fn_mk, Subtype.mk_le_mk, Associates.mk_le_mk_iff_dvd, hd,
        Subtype.coe_mk, associatesEquivOfUniqueUnits_apply, out_dvd_iff, mk_out]


theorem mem_normalizedFactors_factor_dvd_iso_of_mem_normalizedFactors {m p : M} {n : N} (hm : m ≠ 0)
    (hn : n ≠ 0) (hp : p ∈ normalizedFactors m) {d : { l : M // l ∣ m } ≃ { l : N // l ∣ n }}
    (hd : ∀ l l', (d l : N) ∣ d l' ↔ (l : M) ∣ (l' : M)) :
    ↑(d ⟨p, dvd_of_mem_normalizedFactors hp⟩) ∈ normalizedFactors n := by
  suffices
    Prime (d ⟨associatesEquivOfUniqueUnits (associatesEquivOfUniqueUnits.symm p), by
            simp [dvd_of_mem_normalizedFactors hp]⟩ : N) by
    simp only [associatesEquivOfUniqueUnits_apply, out_mk, normalize_eq,
      associatesEquivOfUniqueUnits_symm_apply] at this
    obtain ⟨q, hq, hq'⟩ :=
      exists_mem_normalizedFactors_of_dvd hn this.irreducible
        (d ⟨p, by apply dvd_of_mem_normalizedFactors; convert hp⟩).prop
    rwa [associated_iff_eq.mp hq']
  have :
    Associates.mk
        (d ⟨associatesEquivOfUniqueUnits (associatesEquivOfUniqueUnits.symm p), by
              simp only [dvd_of_mem_normalizedFactors hp, associatesEquivOfUniqueUnits_apply,
                out_mk, normalize_eq, associatesEquivOfUniqueUnits_symm_apply]⟩ : N) =
      ↑(mkFactorOrderIsoOfFactorDvdEquiv hd
          ⟨associatesEquivOfUniqueUnits.symm p, by
            simp only [associatesEquivOfUniqueUnits_symm_apply]
            exact mk_dvd_mk.mpr (dvd_of_mem_normalizedFactors hp)⟩) := by
    rw [mkFactorOrderIsoOfFactorDvdEquiv_apply_coe]
  /-
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    this : Eq (Associates.mk ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOfU …
    ⊢ Prime ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOfUniqueUnits.symm p …
  -/
  rw [← Associates.prime_mk, this]
  /-
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    this : Eq (Associates.mk ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOfU …
    ⊢ Prime ↑((mkFactorOrderIsoOfFactorDvdEquiv hd) ⟨associatesEquivOfUniqueUnits. …
  -/
  letI := Classical.decEq (Associates M)
  /-
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    this✝ : Eq (Associates.mk ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOf …
    this : DecidableEq (Associates M) := Classical.decEq (Associates M)
    ⊢ Prime ↑((mkFactorOrderIsoOfFactorDvdEquiv hd) ⟨associatesEquivOfUniqueUnits. …
  -/
  refine map_prime_of_factor_orderIso (mk_ne_zero.mpr hn) ?_ _
  obtain ⟨q, hq, hq'⟩ :=
    exists_mem_normalizedFactors_of_dvd (mk_ne_zero.mpr hm)
      (prime_mk.mpr (prime_of_normalized_factor p (by convert hp))).irreducible
      (mk_le_mk_of_dvd (dvd_of_mem_normalizedFactors hp))
  /-
    case intro.intro
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    this✝ : Eq (Associates.mk ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOf …
    this : DecidableEq (Associates M) := Classical.decEq (Associates M)
    q : Associates M
    hq : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Associates.m …
    hq' : Associated (Associates.mk p) q
    ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Associates.mk m …
  -/
  simpa only [associated_iff_eq.mp hq', associatesEquivOfUniqueUnits_symm_apply] using hq
  /-
    🎉 no goals
  -/


theorem emultiplicity_factor_dvd_iso_eq_emultiplicity_of_mem_normalizedFactors {m p : M} {n : N}
    (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ∈ normalizedFactors m)
    {d : { l : M // l ∣ m } ≃ { l : N // l ∣ n }} (hd : ∀ l l', (d l : N) ∣ d l' ↔ (l : M) ∣ l') :
    emultiplicity (d ⟨p, dvd_of_mem_normalizedFactors hp⟩ : N) n = emultiplicity p m := by
  /-
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    ⊢ Eq (emultiplicity (↑(d ⟨p, ⋯⟩)) n) (emultiplicity p m)
  -/
  apply Eq.symm
  suffices emultiplicity (Associates.mk p) (Associates.mk m) = emultiplicity (Associates.mk
    ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOfUniqueUnits.symm p), by
      simp [dvd_of_mem_normalizedFactors hp]⟩)) (Associates.mk n) by
    simpa only [emultiplicity_mk_eq_emultiplicity, associatesEquivOfUniqueUnits_symm_apply,
      associatesEquivOfUniqueUnits_apply, out_mk, normalize_eq] using this
  have : Associates.mk (d ⟨associatesEquivOfUniqueUnits (associatesEquivOfUniqueUnits.symm p), by
    simp only [dvd_of_mem_normalizedFactors hp, associatesEquivOfUniqueUnits_symm_apply,
      associatesEquivOfUniqueUnits_apply, out_mk, normalize_eq]⟩ : N) =
    ↑(mkFactorOrderIsoOfFactorDvdEquiv hd ⟨associatesEquivOfUniqueUnits.symm p, by
      rw [associatesEquivOfUniqueUnits_symm_apply]
      exact mk_le_mk_of_dvd (dvd_of_mem_normalizedFactors hp)⟩) := by
    rw [mkFactorOrderIsoOfFactorDvdEquiv_apply_coe]
  /-
    case h
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    this : Eq (Associates.mk ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOfU …
    ⊢ Eq (emultiplicity (Associates.mk p) (Associates.mk m)) (emultiplicity (Assoc …
  -/
  rw [this]
  refine
    emultiplicity_prime_eq_emultiplicity_image_by_factor_orderIso (mk_ne_zero.mpr hn) ?_
      (mkFactorOrderIsoOfFactorDvdEquiv hd)
  obtain ⟨q, hq, hq'⟩ :=
    exists_mem_normalizedFactors_of_dvd (mk_ne_zero.mpr hm)
      (prime_mk.mpr (prime_of_normalized_factor p hp)).irreducible
      (mk_le_mk_of_dvd (dvd_of_mem_normalizedFactors hp))
  /-
    case h.intro.intro
    M : Type u_1
    inst✝⁵ : CancelCommMonoidWithZero M
    N : Type u_2
    inst✝⁴ : CancelCommMonoidWithZero N
    inst✝³ : Subsingleton (Units M)
    inst✝² : Subsingleton (Units N)
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : UniqueFactorizationMonoid N
    m p : M
    n : N
    hm : Ne m 0
    hn : Ne n 0
    hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors m) p
    d : Equiv (Subtype fun l => Dvd.dvd l m) (Subtype fun l => Dvd.dvd l n)
    hd : ∀ (l l' : Subtype fun l => Dvd.dvd l m), Iff (Dvd.dvd ↑(d l) ↑(d l')) (Dv …
    this : Eq (Associates.mk ↑(d ⟨associatesEquivOfUniqueUnits (associatesEquivOfU …
    q : Associates M
    hq : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Associates.m …
    hq' : Associated (Associates.mk p) q
    ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Associates.mk m …
  -/
  rwa [associated_iff_eq.mp hq']
  /-
    🎉 no goals
  -/

