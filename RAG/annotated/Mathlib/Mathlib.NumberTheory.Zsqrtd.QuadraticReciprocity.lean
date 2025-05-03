local notation "ℤ[i]" => GaussianInt


theorem mod_four_eq_three_of_nat_prime_of_prime (p : ℕ) [hp : Fact p.Prime]
    (hpi : Prime (p : ℤ[i])) : p % 4 = 3 :=
  hp.1.eq_two_or_odd.elim
    (fun hp2 =>
      absurd hpi
        (mt irreducible_iff_prime.2 fun ⟨_, h⟩ => by
          /-
            p : Nat
            hp : Fact (Nat.Prime p)
            hpi : Prime ↑p
            hp2 : Eq p 2
            x✝ : Irreducible ↑p
            not_unit✝ : Not (IsUnit ↑p)
            h : ∀ (a b : GaussianInt), Eq (↑p) (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
            ⊢ False
          -/
          have := h ⟨1, 1⟩ ⟨1, -1⟩ (hp2.symm ▸ rfl)
          /-
            p : Nat
            hp : Fact (Nat.Prime p)
            hpi : Prime ↑p
            hp2 : Eq p 2
            x✝ : Irreducible ↑p
            not_unit✝ : Not (IsUnit ↑p)
            h : ∀ (a b : GaussianInt), Eq (↑p) (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
            this : Or (IsUnit { re := 1, im := 1 }) (IsUnit { re := 1, im := -1 })
            ⊢ False
          -/
          rw [← norm_eq_one_iff, ← norm_eq_one_iff] at this
          /-
            p : Nat
            hp : Fact (Nat.Prime p)
            hpi : Prime ↑p
            hp2 : Eq p 2
            x✝ : Irreducible ↑p
            not_unit✝ : Not (IsUnit ↑p)
            h : ∀ (a b : GaussianInt), Eq (↑p) (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
            this : Or (Eq { re := 1, im := 1 }.norm.natAbs 1) (Eq { re := 1, im := -1 }.no …
            ⊢ False
          -/
          exact absurd this (by decide)))
          /-
            🎉 no goals
          -/
    fun hp1 =>
    by_contradiction fun hp3 : p % 4 ≠ 3 => by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        ⊢ False
      -/
      have hp41 : p % 4 = 1 := by omega
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        hp41 : Eq (HMod.hMod p 4) 1
        ⊢ False
      -/
      let ⟨k, hk⟩ := (ZMod.exists_sq_eq_neg_one_iff (p := p)).2 <| by rw [hp41]; decide
      obtain ⟨k, k_lt_p, rfl⟩ : ∃ (k' : ℕ) (_ : k' < p), (k' : ZMod p) = k := by
        exact ⟨k.val, k.val_lt, ZMod.natCast_zmod_val k⟩
      have hpk : p ∣ k ^ 2 + 1 := by
        rw [pow_two, ← CharP.cast_eq_zero_iff (ZMod p) p, Nat.cast_add, Nat.cast_mul, Nat.cast_one,
          ← hk, neg_add_cancel]
      /-
        case intro.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        hp41 : Eq (HMod.hMod p 4) 1
        k : Nat
        k_lt_p : LT.lt k p
        hk : Eq (-1) (HMul.hMul ↑k ↑k)
        hpk : Dvd.dvd p (HAdd.hAdd (HPow.hPow k 2) 1)
        ⊢ False
      -/
      have hkmul : (k ^ 2 + 1 : ℤ[i]) = ⟨k, 1⟩ * ⟨k, -1⟩ := by ext <;> simp [sq]
      have hkltp : 1 + k * k < p * p :=
        calc
          1 + k * k ≤ k + k * k := by
            apply add_le_add_right
            exact (Nat.pos_of_ne_zero fun (hk0 : k = 0) => by clear_aux_decl; simp_all [pow_succ'])
          _ = k * (k + 1) := by simp [add_comm, mul_add]
          _ < p * p := mul_lt_mul k_lt_p k_lt_p (Nat.succ_pos _) (Nat.zero_le _)
      have hpk₁ : ¬(p : ℤ[i]) ∣ ⟨k, -1⟩ := fun ⟨x, hx⟩ =>
        lt_irrefl (p * x : ℤ[i]).norm.natAbs <|
          calc
            (norm (p * x : ℤ[i])).natAbs = (Zsqrtd.norm ⟨k, -1⟩).natAbs := by rw [hx]
            _ < (norm (p : ℤ[i])).natAbs := by simpa [add_comm, Zsqrtd.norm] using hkltp
            _ ≤ (norm (p * x : ℤ[i])).natAbs :=
              norm_le_norm_mul_left _ fun hx0 =>
                show (-1 : ℤ) ≠ 0 by decide <| by simpa [hx0] using congr_arg Zsqrtd.im hx
      have hpk₂ : ¬(p : ℤ[i]) ∣ ⟨k, 1⟩ := fun ⟨x, hx⟩ =>
        lt_irrefl (p * x : ℤ[i]).norm.natAbs <|
          calc
            (norm (p * x : ℤ[i])).natAbs = (Zsqrtd.norm ⟨k, 1⟩).natAbs := by rw [hx]
            _ < (norm (p : ℤ[i])).natAbs := by simpa [add_comm, Zsqrtd.norm] using hkltp
            _ ≤ (norm (p * x : ℤ[i])).natAbs :=
              norm_le_norm_mul_left _ fun hx0 =>
                show (1 : ℤ) ≠ 0 by decide <| by simpa [hx0] using congr_arg Zsqrtd.im hx
      /-
        case intro.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        hp41 : Eq (HMod.hMod p 4) 1
        k : Nat
        k_lt_p : LT.lt k p
        hk : Eq (-1) (HMul.hMul ↑k ↑k)
        hpk : Dvd.dvd p (HAdd.hAdd (HPow.hPow k 2) 1)
        hkmul : Eq (HAdd.hAdd (HPow.hPow (↑k) 2) 1) (HMul.hMul { re := ↑k, im := 1 } { …
        hkltp : LT.lt (HAdd.hAdd 1 (HMul.hMul k k)) (HMul.hMul p p)
        hpk₁ : Not (Dvd.dvd ↑p { re := ↑k, im := -1 })
        hpk₂ : Not (Dvd.dvd ↑p { re := ↑k, im := 1 })
        ⊢ False
      -/
      obtain ⟨y, hy⟩ := hpk
      /-
        case intro.intro.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        hp41 : Eq (HMod.hMod p 4) 1
        k : Nat
        k_lt_p : LT.lt k p
        hk : Eq (-1) (HMul.hMul ↑k ↑k)
        hkmul : Eq (HAdd.hAdd (HPow.hPow (↑k) 2) 1) (HMul.hMul { re := ↑k, im := 1 } { …
        hkltp : LT.lt (HAdd.hAdd 1 (HMul.hMul k k)) (HMul.hMul p p)
        hpk₁ : Not (Dvd.dvd ↑p { re := ↑k, im := -1 })
        hpk₂ : Not (Dvd.dvd ↑p { re := ↑k, im := 1 })
        y : Nat
        hy : Eq (HAdd.hAdd (HPow.hPow k 2) 1) (HMul.hMul p y)
        ⊢ False
      -/
      have := hpi.2.2 ⟨k, 1⟩ ⟨k, -1⟩ ⟨y, by rw [← hkmul, ← Nat.cast_mul p, ← hy]; simp⟩
      /-
        case intro.intro.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        hp41 : Eq (HMod.hMod p 4) 1
        k : Nat
        k_lt_p : LT.lt k p
        hk : Eq (-1) (HMul.hMul ↑k ↑k)
        hkmul : Eq (HAdd.hAdd (HPow.hPow (↑k) 2) 1) (HMul.hMul { re := ↑k, im := 1 } { …
        hkltp : LT.lt (HAdd.hAdd 1 (HMul.hMul k k)) (HMul.hMul p p)
        hpk₁ : Not (Dvd.dvd ↑p { re := ↑k, im := -1 })
        hpk₂ : Not (Dvd.dvd ↑p { re := ↑k, im := 1 })
        y : Nat
        hy : Eq (HAdd.hAdd (HPow.hPow k 2) 1) (HMul.hMul p y)
        this : Or (Dvd.dvd ↑p { re := ↑k, im := 1 }) (Dvd.dvd ↑p { re := ↑k, im := -1 })
        ⊢ False
      -/
      clear_aux_decl
      /-
        case intro.intro.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Prime ↑p
        hp1 : Eq (HMod.hMod p 2) 1
        hp3 : Ne (HMod.hMod p 4) 3
        hp41 : Eq (HMod.hMod p 4) 1
        k : Nat
        k_lt_p : LT.lt k p
        hk : Eq (-1) (HMul.hMul ↑k ↑k)
        hkmul : Eq (HAdd.hAdd (HPow.hPow (↑k) 2) 1) (HMul.hMul { re := ↑k, im := 1 } { …
        hkltp : LT.lt (HAdd.hAdd 1 (HMul.hMul k k)) (HMul.hMul p p)
        hpk₁ : Not (Dvd.dvd ↑p { re := ↑k, im := -1 })
        hpk₂ : Not (Dvd.dvd ↑p { re := ↑k, im := 1 })
        y : Nat
        hy : Eq (HAdd.hAdd (HPow.hPow k 2) 1) (HMul.hMul p y)
        this : Or (Dvd.dvd ↑p { re := ↑k, im := 1 }) (Dvd.dvd ↑p { re := ↑k, im := -1 })
        ⊢ False
      -/
      tauto
      /-
        🎉 no goals
      -/


theorem prime_of_nat_prime_of_mod_four_eq_three (p : ℕ) [Fact p.Prime] (hp3 : p % 4 = 3) :
    Prime (p : ℤ[i]) :=
  irreducible_iff_prime.1 <|
    by_contradiction fun hpi =>
      let ⟨a, b, hab⟩ := sq_add_sq_of_nat_prime_of_not_irreducible p hpi
      have : ∀ a b : ZMod 4, a ^ 2 + b ^ 2 ≠ (p : ZMod 4) := by
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          hp3 : Eq (HMod.hMod p 4) 3
          hpi : Not (Irreducible ↑p)
          a b : Nat
          hab : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) p
          ⊢ ∀ (a b : ZMod 4), Ne (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) ↑p
        -/
        rw [← ZMod.natCast_mod p 4, hp3]; decide
                                          /-
                                            🎉 no goals
                                          -/
                         /-
                           p : Nat
                           inst✝ : Fact (Nat.Prime p)
                           hp3 : Eq (HMod.hMod p 4) 3
                           hpi : Not (Irreducible ↑p)
                           a b : Nat
                           hab : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) p
                           this : ∀ (a b : ZMod 4), Ne (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) ↑p
                           ⊢ Eq (HAdd.hAdd (HPow.hPow (↑a) 2) (HPow.hPow (↑b) 2)) ↑(HAdd.hAdd (HPow.hPow  …
                         -/
      this a b (hab ▸ by simp)
                         /-
                           🎉 no goals
                         -/


/-- A prime natural number is prime in `ℤ[i]` if and only if it is `3` mod `4` -/
theorem prime_iff_mod_four_eq_three_of_nat_prime (p : ℕ) [Fact p.Prime] :
    Prime (p : ℤ[i]) ↔ p % 4 = 3 :=
  ⟨mod_four_eq_three_of_nat_prime_of_prime p, prime_of_nat_prime_of_mod_four_eq_three p⟩


