/-- If `a^(p-1) = 1 mod p`, but `a^((p-1)/q) ≠ 1 mod p` for all prime factors `q` of `p-1`, then `p`
is prime. This is true because `a` has order `p-1` in the multiplicative group mod `p`, so this
group must itself have order `p-1`, which only happens when `p` is prime.
-/
theorem lucas_primality (p : ℕ) (a : ZMod p) (ha : a ^ (p - 1) = 1)
    (hd : ∀ q : ℕ, q.Prime → q ∣ p - 1 → a ^ ((p - 1) / q) ≠ 1) : p.Prime := by
  have h : p ≠ 0 ∧ p ≠ 1 := by
    constructor <;> rintro rfl <;> exact hd 2 Nat.prime_two (dvd_zero _) (pow_zero _)
  /-
    p : Nat
    a : ZMod p
    ha : Eq (HPow.hPow a (HSub.hSub p 1)) 1
    hd : ∀ (q : Nat), Nat.Prime q → Dvd.dvd q (HSub.hSub p 1) → Ne (HPow.hPow a (H …
    h : And (Ne p 0) (Ne p 1)
    ⊢ Nat.Prime p
  -/
  have hp1 : 1 < p := Nat.one_lt_iff_ne_zero_and_ne_one.2 h
  /-
    p : Nat
    a : ZMod p
    ha : Eq (HPow.hPow a (HSub.hSub p 1)) 1
    hd : ∀ (q : Nat), Nat.Prime q → Dvd.dvd q (HSub.hSub p 1) → Ne (HPow.hPow a (H …
    h : And (Ne p 0) (Ne p 1)
    hp1 : LT.lt 1 p
    ⊢ Nat.Prime p
  -/
  have : NeZero p := ⟨h.1⟩
  /-
    p : Nat
    a : ZMod p
    ha : Eq (HPow.hPow a (HSub.hSub p 1)) 1
    hd : ∀ (q : Nat), Nat.Prime q → Dvd.dvd q (HSub.hSub p 1) → Ne (HPow.hPow a (H …
    h : And (Ne p 0) (Ne p 1)
    hp1 : LT.lt 1 p
    this : NeZero p
    ⊢ Nat.Prime p
  -/
  rw [Nat.prime_iff_card_units]
  /-
    p : Nat
    a : ZMod p
    ha : Eq (HPow.hPow a (HSub.hSub p 1)) 1
    hd : ∀ (q : Nat), Nat.Prime q → Dvd.dvd q (HSub.hSub p 1) → Ne (HPow.hPow a (H …
    h : And (Ne p 0) (Ne p 1)
    hp1 : LT.lt 1 p
    this : NeZero p
    ⊢ Eq (Fintype.card (Units (ZMod p))) (HSub.hSub p 1)
  -/
  apply (Nat.card_units_zmod_lt_sub_one hp1).antisymm
  /-
    p : Nat
    a : ZMod p
    ha : Eq (HPow.hPow a (HSub.hSub p 1)) 1
    hd : ∀ (q : Nat), Nat.Prime q → Dvd.dvd q (HSub.hSub p 1) → Ne (HPow.hPow a (H …
    h : And (Ne p 0) (Ne p 1)
    hp1 : LT.lt 1 p
    this : NeZero p
    ⊢ LE.le (HSub.hSub p 1) (Fintype.card (Units (ZMod p)))
  -/
  let a' : (ZMod p)ˣ := Units.mkOfMulEqOne a _ (by rwa [← pow_succ', tsub_add_eq_add_tsub hp1])
  calc p - 1 = orderOf a := (orderOf_eq_of_pow_and_pow_div_prime (tsub_pos_of_lt hp1) ha hd).symm
    _ = orderOf a' := orderOf_injective (Units.coeHom _) Units.ext a'
    _ ≤ Fintype.card (ZMod p)ˣ := orderOf_le_card_univ


/-- If `p` is prime, then there exists an `a` such that `a^(p-1) = 1 mod p`
and `a^((p-1)/q) ≠ 1 mod p` for all prime factors `q` of `p-1`.
The multiplicative group mod `p` is cyclic, so `a` can be any generator of the group
(which must have order `p-1`).
-/
theorem reverse_lucas_primality (p : ℕ) (hP : p.Prime) :
    ∃ a : ZMod p, a ^ (p - 1) = 1 ∧ ∀ q : ℕ, q.Prime → q ∣ p - 1 → a ^ ((p - 1) / q) ≠ 1 := by
  /-
    p : Nat
    hP : Nat.Prime p
    ⊢ Exists fun a => And (Eq (HPow.hPow a (HSub.hSub p 1)) 1) (∀ (q : Nat), Nat.P …
  -/
  have : Fact p.Prime := ⟨hP⟩
  /-
    p : Nat
    hP : Nat.Prime p
    this : Fact (Nat.Prime p)
    ⊢ Exists fun a => And (Eq (HPow.hPow a (HSub.hSub p 1)) 1) (∀ (q : Nat), Nat.P …
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := (ZMod p)ˣ)
  have h1 : orderOf g = p - 1 := by
    rwa [orderOf_eq_card_of_forall_mem_zpowers hg, Nat.card_eq_fintype_card,
      ← Nat.prime_iff_card_units]
  /-
    case intro
    p : Nat
    hP : Nat.Prime p
    this : Fact (Nat.Prime p)
    g : Units (ZMod p)
    hg : ∀ (x : Units (ZMod p)), Membership.mem (Subgroup.zpowers g) x
    h1 : Eq (orderOf g) (HSub.hSub p 1)
    ⊢ Exists fun a => And (Eq (HPow.hPow a (HSub.hSub p 1)) 1) (∀ (q : Nat), Nat.P …
  -/
  have h2 := tsub_pos_iff_lt.2 hP.one_lt
  /-
    case intro
    p : Nat
    hP : Nat.Prime p
    this : Fact (Nat.Prime p)
    g : Units (ZMod p)
    hg : ∀ (x : Units (ZMod p)), Membership.mem (Subgroup.zpowers g) x
    h1 : Eq (orderOf g) (HSub.hSub p 1)
    h2 : LT.lt 0 (HSub.hSub p 1)
    ⊢ Exists fun a => And (Eq (HPow.hPow a (HSub.hSub p 1)) 1) (∀ (q : Nat), Nat.P …
  -/
  rw [← orderOf_injective (Units.coeHom _) Units.ext _, orderOf_eq_iff h2] at h1
  /-
    case intro
    p : Nat
    hP : Nat.Prime p
    this : Fact (Nat.Prime p)
    g : Units (ZMod p)
    hg : ∀ (x : Units (ZMod p)), Membership.mem (Subgroup.zpowers g) x
    h1 : And (Eq (HPow.hPow ((Units.coeHom (ZMod p)) g) (HSub.hSub p 1)) 1) (∀ (m  …
    h2 : LT.lt 0 (HSub.hSub p 1)
    ⊢ Exists fun a => And (Eq (HPow.hPow a (HSub.hSub p 1)) 1) (∀ (q : Nat), Nat.P …
  -/
  refine ⟨g, h1.1, fun q hq hqd ↦ ?_⟩
  /-
    case intro
    p : Nat
    hP : Nat.Prime p
    this : Fact (Nat.Prime p)
    g : Units (ZMod p)
    hg : ∀ (x : Units (ZMod p)), Membership.mem (Subgroup.zpowers g) x
    h1 : And (Eq (HPow.hPow ((Units.coeHom (ZMod p)) g) (HSub.hSub p 1)) 1) (∀ (m  …
    h2 : LT.lt 0 (HSub.hSub p 1)
    q : Nat
    hq : Nat.Prime q
    hqd : Dvd.dvd q (HSub.hSub p 1)
    ⊢ Ne (HPow.hPow (↑g) (HDiv.hDiv (HSub.hSub p 1) q)) 1
  -/
  replace hq := hq.one_lt
  /-
    case intro
    p : Nat
    hP : Nat.Prime p
    this : Fact (Nat.Prime p)
    g : Units (ZMod p)
    hg : ∀ (x : Units (ZMod p)), Membership.mem (Subgroup.zpowers g) x
    h1 : And (Eq (HPow.hPow ((Units.coeHom (ZMod p)) g) (HSub.hSub p 1)) 1) (∀ (m  …
    h2 : LT.lt 0 (HSub.hSub p 1)
    q : Nat
    hqd : Dvd.dvd q (HSub.hSub p 1)
    hq : LT.lt 1 q
    ⊢ Ne (HPow.hPow (↑g) (HDiv.hDiv (HSub.hSub p 1) q)) 1
  -/
  exact h1.2 _ (Nat.div_lt_self h2 hq) (Nat.div_pos (Nat.le_of_dvd h2 hqd) (zero_lt_one.trans hq))
  /-
    🎉 no goals
  -/


/-- A number `p` is prime if and only if there exists an `a` such that
`a^(p-1) = 1 mod p` and `a^((p-1)/q) ≠ 1 mod p` for all prime factors `q` of `p-1`.
-/
theorem lucas_primality_iff (p : ℕ) : p.Prime ↔
    ∃ a : ZMod p, a ^ (p - 1) = 1 ∧ ∀ q : ℕ, q.Prime → q ∣ p - 1 → a ^ ((p - 1) / q) ≠ 1 :=
  ⟨reverse_lucas_primality p, fun ⟨a, ⟨ha, hb⟩⟩ ↦ lucas_primality p a ha hb⟩

