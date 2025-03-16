/-- Shorthand for three non-zero integers `a`, `b`, and `c` satisfying `a ^ 4 + b ^ 4 = c ^ 2`.
We will show that no integers satisfy this equation. Clearly Fermat's Last theorem for n = 4
follows. -/
def Fermat42 (a b c : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ a ^ 4 + b ^ 4 = c ^ 2


theorem comm {a b c : ℤ} : Fermat42 a b c ↔ Fermat42 b a c := by
  /-
    a b c : Int
    ⊢ Iff (Fermat42 a b c) (Fermat42 b a c)
  -/
  delta Fermat42
  /-
    a b c : Int
    ⊢ Iff (And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b  …
  -/
  rw [add_comm]
  /-
    a b c : Int
    ⊢ Iff (And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow b 4) (HPow.hPow a  …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem mul {a b c k : ℤ} (hk0 : k ≠ 0) :
    Fermat42 a b c ↔ Fermat42 (k * a) (k * b) (k ^ 2 * c) := by
  /-
    a b c k : Int
    hk0 : Ne k 0
    ⊢ Iff (Fermat42 a b c) (Fermat42 (HMul.hMul k a) (HMul.hMul k b) (HMul.hMul (H …
  -/
  delta Fermat42
  /-
    a b c k : Int
    hk0 : Ne k 0
    ⊢ Iff (And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b  …
  -/
  constructor
    /-
      case mp
      a b c k : Int
      hk0 : Ne k 0
      ⊢ And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) ( …
    -/
  · intro f42
    /-
      case mp
      a b c k : Int
      hk0 : Ne k 0
      f42 : And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4 …
      ⊢ And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (HPow. …
    -/
    constructor
      /-
        case mp.left
        a b c k : Int
        hk0 : Ne k 0
        f42 : And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4 …
        ⊢ Ne (HMul.hMul k a) 0
      -/
    · exact mul_ne_zero hk0 f42.1
      /-
        🎉 no goals
      -/
    /-
      case mp.right
      a b c k : Int
      hk0 : Ne k 0
      f42 : And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4 …
      ⊢ And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (HPow.hPow (HMul.hMul k a) 4) (HPo …
    -/
    constructor
      /-
        case mp.right.left
        a b c k : Int
        hk0 : Ne k 0
        f42 : And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4 …
        ⊢ Ne (HMul.hMul k b) 0
      -/
    · exact mul_ne_zero hk0 f42.2.1
      /-
        🎉 no goals
      -/
      /-
        case mp.right.right
        a b c k : Int
        hk0 : Ne k 0
        f42 : And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4 …
        ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul k a) 4) (HPow.hPow (HMul.hMul k b) 4)) ( …
      -/
    · have H : a ^ 4 + b ^ 4 = c ^ 2 := f42.2.2
      /-
        case mp.right.right
        a b c k : Int
        hk0 : Ne k 0
        f42 : And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4 …
        H : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
        ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul k a) 4) (HPow.hPow (HMul.hMul k b) 4)) ( …
      -/
      linear_combination k ^ 4 * H
      /-
        🎉 no goals
      -/
    /-
      case mpr
      a b c k : Int
      hk0 : Ne k 0
      ⊢ And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (HPow. …
    -/
  · intro f42
    /-
      case mpr
      a b c k : Int
      hk0 : Ne k 0
      f42 : And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (H …
      ⊢ And (Ne a 0) (And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) ( …
    -/
    constructor
      /-
        case mpr.left
        a b c k : Int
        hk0 : Ne k 0
        f42 : And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (H …
        ⊢ Ne a 0
      -/
    · exact right_ne_zero_of_mul f42.1
      /-
        🎉 no goals
      -/
    /-
      case mpr.right
      a b c k : Int
      hk0 : Ne k 0
      f42 : And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (H …
      ⊢ And (Ne b 0) (Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2))
    -/
    constructor
      /-
        case mpr.right.left
        a b c k : Int
        hk0 : Ne k 0
        f42 : And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (H …
        ⊢ Ne b 0
      -/
    · exact right_ne_zero_of_mul f42.2.1
      /-
        🎉 no goals
      -/
    /-
      case mpr.right.right
      a b c k : Int
      hk0 : Ne k 0
      f42 : And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (H …
      ⊢ Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
    -/
    apply (mul_right_inj' (pow_ne_zero 4 hk0)).mp
    /-
      case mpr.right.right
      a b c k : Int
      hk0 : Ne k 0
      f42 : And (Ne (HMul.hMul k a) 0) (And (Ne (HMul.hMul k b) 0) (Eq (HAdd.hAdd (H …
      ⊢ Eq (HMul.hMul (HPow.hPow k 4) (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4))) ( …
    -/
    linear_combination f42.2.2
    /-
      🎉 no goals
    -/


theorem ne_zero {a b c : ℤ} (h : Fermat42 a b c) : c ≠ 0 := by
  /-
    a b c : Int
    h : Fermat42 a b c
    ⊢ Ne c 0
  -/
  apply ne_zero_pow two_ne_zero _; apply ne_of_gt
  /-
    case h
    a b c : Int
    h : Fermat42 a b c
    ⊢ LT.lt 0 (HPow.hPow c 2)
  -/
  rw [← h.2.2, (by ring : a ^ 4 + b ^ 4 = (a ^ 2) ^ 2 + (b ^ 2) ^ 2)]
  exact
    add_pos (sq_pos_of_ne_zero (pow_ne_zero 2 h.1)) (sq_pos_of_ne_zero (pow_ne_zero 2 h.2.1))


/-- We say a solution to `a ^ 4 + b ^ 4 = c ^ 2` is minimal if there is no other solution with
a smaller `c` (in absolute value). -/
def Minimal (a b c : ℤ) : Prop :=
  Fermat42 a b c ∧ ∀ a1 b1 c1 : ℤ, Fermat42 a1 b1 c1 → Int.natAbs c ≤ Int.natAbs c1


/-- if we have a solution to `a ^ 4 + b ^ 4 = c ^ 2` then there must be a minimal one. -/
theorem exists_minimal {a b c : ℤ} (h : Fermat42 a b c) : ∃ a0 b0 c0, Minimal a0 b0 c0 := by
  classical
  let S : Set ℕ := { n | ∃ s : ℤ × ℤ × ℤ, Fermat42 s.1 s.2.1 s.2.2 ∧ n = Int.natAbs s.2.2 }
  have S_nonempty : S.Nonempty := by
    use Int.natAbs c
    rw [Set.mem_setOf_eq]
    use ⟨a, ⟨b, c⟩⟩
  let m : ℕ := Nat.find S_nonempty
  have m_mem : m ∈ S := Nat.find_spec S_nonempty
  rcases m_mem with ⟨s0, hs0, hs1⟩
  use s0.1, s0.2.1, s0.2.2, hs0
  intro a1 b1 c1 h1
  rw [← hs1]
  apply Nat.find_min'
  use ⟨a1, ⟨b1, c1⟩⟩


/-- a minimal solution to `a ^ 4 + b ^ 4 = c ^ 2` must have `a` and `b` coprime. -/
theorem coprime_of_minimal {a b c : ℤ} (h : Minimal a b c) : IsCoprime a b := by
  /-
    a b c : Int
    h : Fermat42.Minimal a b c
    ⊢ IsCoprime a b
  -/
  apply Int.gcd_eq_one_iff_coprime.mp
  /-
    a b c : Int
    h : Fermat42.Minimal a b c
    ⊢ Eq (a.gcd b) 1
  -/
  by_contra hab
  /-
    a b c : Int
    h : Fermat42.Minimal a b c
    hab : Not (Eq (a.gcd b) 1)
    ⊢ False
  -/
  obtain ⟨p, hp, hpa, hpb⟩ := Nat.Prime.not_coprime_iff_dvd.mp hab
  /-
    case intro.intro.intro
    a b c : Int
    h : Fermat42.Minimal a b c
    hab : Not (Eq (a.gcd b) 1)
    p : Nat
    hp : Nat.Prime p
    hpa : Dvd.dvd p a.natAbs
    hpb : Dvd.dvd p b.natAbs
    ⊢ False
  -/
  obtain ⟨a1, rfl⟩ := Int.natCast_dvd.mpr hpa
  /-
    case intro.intro.intro.intro
    b c : Int
    p : Nat
    hp : Nat.Prime p
    hpb : Dvd.dvd p b.natAbs
    a1 : Int
    h : Fermat42.Minimal (HMul.hMul (↑p) a1) b c
    hab : Not (Eq ((HMul.hMul (↑p) a1).gcd b) 1)
    hpa : Dvd.dvd p (HMul.hMul (↑p) a1).natAbs
    ⊢ False
  -/
  obtain ⟨b1, rfl⟩ := Int.natCast_dvd.mpr hpb
  have hpc : (p : ℤ) ^ 2 ∣ c := by
    rw [← Int.pow_dvd_pow_iff two_ne_zero, ← h.1.2.2]
    apply Dvd.intro (a1 ^ 4 + b1 ^ 4)
    ring
  /-
    case intro.intro.intro.intro.intro
    c : Int
    p : Nat
    hp : Nat.Prime p
    a1 : Int
    hpa : Dvd.dvd p (HMul.hMul (↑p) a1).natAbs
    b1 : Int
    hpb : Dvd.dvd p (HMul.hMul (↑p) b1).natAbs
    h : Fermat42.Minimal (HMul.hMul (↑p) a1) (HMul.hMul (↑p) b1) c
    hab : Not (Eq ((HMul.hMul (↑p) a1).gcd (HMul.hMul (↑p) b1)) 1)
    hpc : Dvd.dvd (HPow.hPow (↑p) 2) c
    ⊢ False
  -/
  obtain ⟨c1, rfl⟩ := hpc
  have hf : Fermat42 a1 b1 c1 :=
    (Fermat42.mul (Int.natCast_ne_zero.mpr (Nat.Prime.ne_zero hp))).mpr h.1
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    hp : Nat.Prime p
    a1 : Int
    hpa : Dvd.dvd p (HMul.hMul (↑p) a1).natAbs
    b1 : Int
    hpb : Dvd.dvd p (HMul.hMul (↑p) b1).natAbs
    hab : Not (Eq ((HMul.hMul (↑p) a1).gcd (HMul.hMul (↑p) b1)) 1)
    c1 : Int
    h : Fermat42.Minimal (HMul.hMul (↑p) a1) (HMul.hMul (↑p) b1) (HMul.hMul (HPow. …
    hf : Fermat42 a1 b1 c1
    ⊢ False
  -/
  apply Nat.le_lt_asymm (h.2 _ _ _ hf)
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    hp : Nat.Prime p
    a1 : Int
    hpa : Dvd.dvd p (HMul.hMul (↑p) a1).natAbs
    b1 : Int
    hpb : Dvd.dvd p (HMul.hMul (↑p) b1).natAbs
    hab : Not (Eq ((HMul.hMul (↑p) a1).gcd (HMul.hMul (↑p) b1)) 1)
    c1 : Int
    h : Fermat42.Minimal (HMul.hMul (↑p) a1) (HMul.hMul (↑p) b1) (HMul.hMul (HPow. …
    hf : Fermat42 a1 b1 c1
    ⊢ LT.lt c1.natAbs (HMul.hMul (HPow.hPow (↑p) 2) c1).natAbs
  -/
  rw [Int.natAbs_mul, lt_mul_iff_one_lt_left, Int.natAbs_pow, Int.natAbs_ofNat]
    /-
      case intro.intro.intro.intro.intro.intro
      p : Nat
      hp : Nat.Prime p
      a1 : Int
      hpa : Dvd.dvd p (HMul.hMul (↑p) a1).natAbs
      b1 : Int
      hpb : Dvd.dvd p (HMul.hMul (↑p) b1).natAbs
      hab : Not (Eq ((HMul.hMul (↑p) a1).gcd (HMul.hMul (↑p) b1)) 1)
      c1 : Int
      h : Fermat42.Minimal (HMul.hMul (↑p) a1) (HMul.hMul (↑p) b1) (HMul.hMul (HPow. …
      hf : Fermat42 a1 b1 c1
      ⊢ LT.lt 1 (HPow.hPow p 2)
    -/
  · exact Nat.one_lt_pow two_ne_zero (Nat.Prime.one_lt hp)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro
      p : Nat
      hp : Nat.Prime p
      a1 : Int
      hpa : Dvd.dvd p (HMul.hMul (↑p) a1).natAbs
      b1 : Int
      hpb : Dvd.dvd p (HMul.hMul (↑p) b1).natAbs
      hab : Not (Eq ((HMul.hMul (↑p) a1).gcd (HMul.hMul (↑p) b1)) 1)
      c1 : Int
      h : Fermat42.Minimal (HMul.hMul (↑p) a1) (HMul.hMul (↑p) b1) (HMul.hMul (HPow. …
      hf : Fermat42 a1 b1 c1
      ⊢ LT.lt 0 c1.natAbs
    -/
  · exact Nat.pos_of_ne_zero (Int.natAbs_ne_zero.2 (ne_zero hf))
    /-
      🎉 no goals
    -/


/-- We can swap `a` and `b` in a minimal solution to `a ^ 4 + b ^ 4 = c ^ 2`. -/
theorem minimal_comm {a b c : ℤ} : Minimal a b c → Minimal b a c := fun ⟨h1, h2⟩ =>
  ⟨Fermat42.comm.mp h1, h2⟩


/-- We can assume that a minimal solution to `a ^ 4 + b ^ 4 = c ^ 2` has positive `c`. -/
theorem neg_of_minimal {a b c : ℤ} : Minimal a b c → Minimal a b (-c) := by
  /-
    a b c : Int
    ⊢ Fermat42.Minimal a b c → Fermat42.Minimal a b (Neg.neg c)
  -/
  rintro ⟨⟨ha, hb, heq⟩, h2⟩
  /-
    case intro.intro.intro
    a b c : Int
    h2 : ∀ (a1 b1 c1 : Int), Fermat42 a1 b1 c1 → LE.le c.natAbs c1.natAbs
    ha : Ne a 0
    hb : Ne b 0
    heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
    ⊢ Fermat42.Minimal a b (Neg.neg c)
  -/
  constructor
    /-
      case intro.intro.intro.left
      a b c : Int
      h2 : ∀ (a1 b1 c1 : Int), Fermat42 a1 b1 c1 → LE.le c.natAbs c1.natAbs
      ha : Ne a 0
      hb : Ne b 0
      heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
      ⊢ Fermat42 a b (Neg.neg c)
    -/
  · apply And.intro ha (And.intro hb _)
    /-
      a b c : Int
      h2 : ∀ (a1 b1 c1 : Int), Fermat42 a1 b1 c1 → LE.le c.natAbs c1.natAbs
      ha : Ne a 0
      hb : Ne b 0
      heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
      ⊢ Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow (Neg.neg c) 2)
    -/
    rw [heq]
    /-
      a b c : Int
      h2 : ∀ (a1 b1 c1 : Int), Fermat42 a1 b1 c1 → LE.le c.natAbs c1.natAbs
      ha : Ne a 0
      hb : Ne b 0
      heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
      ⊢ Eq (HPow.hPow c 2) (HPow.hPow (Neg.neg c) 2)
    -/
    exact (neg_sq c).symm
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.right
    a b c : Int
    h2 : ∀ (a1 b1 c1 : Int), Fermat42 a1 b1 c1 → LE.le c.natAbs c1.natAbs
    ha : Ne a 0
    hb : Ne b 0
    heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
    ⊢ ∀ (a1 b1 c1 : Int), Fermat42 a1 b1 c1 → LE.le (Neg.neg c).natAbs c1.natAbs
  -/
  rwa [Int.natAbs_neg c]
  /-
    🎉 no goals
  -/


/-- We can assume that a minimal solution to `a ^ 4 + b ^ 4 = c ^ 2` has `a` odd. -/
theorem exists_odd_minimal {a b c : ℤ} (h : Fermat42 a b c) :
    ∃ a0 b0 c0, Minimal a0 b0 c0 ∧ a0 % 2 = 1 := by
  /-
    a b c : Int
    h : Fermat42 a b c
    ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
  -/
  obtain ⟨a0, b0, c0, hf⟩ := exists_minimal h
  /-
    case intro.intro.intro
    a b c : Int
    h : Fermat42 a b c
    a0 b0 c0 : Int
    hf : Fermat42.Minimal a0 b0 c0
    ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
  -/
  cases' Int.emod_two_eq_zero_or_one a0 with hap hap
    /-
      case intro.intro.intro.inl
      a b c : Int
      h : Fermat42 a b c
      a0 b0 c0 : Int
      hf : Fermat42.Minimal a0 b0 c0
      hap : Eq (HMod.hMod a0 2) 0
      ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
    -/
  · cases' Int.emod_two_eq_zero_or_one b0 with hbp hbp
      /-
        case intro.intro.intro.inl.inl
        a b c : Int
        h : Fermat42 a b c
        a0 b0 c0 : Int
        hf : Fermat42.Minimal a0 b0 c0
        hap : Eq (HMod.hMod a0 2) 0
        hbp : Eq (HMod.hMod b0 2) 0
        ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
      -/
    · exfalso
      have h1 : 2 ∣ (Int.gcd a0 b0 : ℤ) :=
        Int.dvd_gcd (Int.dvd_of_emod_eq_zero hap) (Int.dvd_of_emod_eq_zero hbp)
      /-
        case intro.intro.intro.inl.inl
        a b c : Int
        h : Fermat42 a b c
        a0 b0 c0 : Int
        hf : Fermat42.Minimal a0 b0 c0
        hap : Eq (HMod.hMod a0 2) 0
        hbp : Eq (HMod.hMod b0 2) 0
        h1 : Dvd.dvd 2 ↑(a0.gcd b0)
        ⊢ False
      -/
      rw [Int.gcd_eq_one_iff_coprime.mpr (coprime_of_minimal hf)] at h1
      /-
        case intro.intro.intro.inl.inl
        a b c : Int
        h : Fermat42 a b c
        a0 b0 c0 : Int
        hf : Fermat42.Minimal a0 b0 c0
        hap : Eq (HMod.hMod a0 2) 0
        hbp : Eq (HMod.hMod b0 2) 0
        h1 : Dvd.dvd 2 ↑1
        ⊢ False
      -/
      revert h1
      /-
        case intro.intro.intro.inl.inl
        a b c : Int
        h : Fermat42 a b c
        a0 b0 c0 : Int
        hf : Fermat42.Minimal a0 b0 c0
        hap : Eq (HMod.hMod a0 2) 0
        hbp : Eq (HMod.hMod b0 2) 0
        ⊢ Dvd.dvd 2 ↑1 → False
      -/
      decide
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.inl.inr
        a b c : Int
        h : Fermat42 a b c
        a0 b0 c0 : Int
        hf : Fermat42.Minimal a0 b0 c0
        hap : Eq (HMod.hMod a0 2) 0
        hbp : Eq (HMod.hMod b0 2) 1
        ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
      -/
    · exact ⟨b0, ⟨a0, ⟨c0, minimal_comm hf, hbp⟩⟩⟩
      /-
        🎉 no goals
      -/
  /-
    case intro.intro.intro.inr
    a b c : Int
    h : Fermat42 a b c
    a0 b0 c0 : Int
    hf : Fermat42.Minimal a0 b0 c0
    hap : Eq (HMod.hMod a0 2) 1
    ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
  -/
  exact ⟨a0, ⟨b0, ⟨c0, hf, hap⟩⟩⟩
  /-
    🎉 no goals
  -/


/-- We can assume that a minimal solution to `a ^ 4 + b ^ 4 = c ^ 2` has
`a` odd and `c` positive. -/
theorem exists_pos_odd_minimal {a b c : ℤ} (h : Fermat42 a b c) :
    ∃ a0 b0 c0, Minimal a0 b0 c0 ∧ a0 % 2 = 1 ∧ 0 < c0 := by
  /-
    a b c : Int
    h : Fermat42 a b c
    ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
  -/
  obtain ⟨a0, b0, c0, hf, hc⟩ := exists_odd_minimal h
  /-
    case intro.intro.intro.intro
    a b c : Int
    h : Fermat42 a b c
    a0 b0 c0 : Int
    hf : Fermat42.Minimal a0 b0 c0
    hc : Eq (HMod.hMod a0 2) 1
    ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
  -/
  rcases lt_trichotomy 0 c0 with (h1 | h1 | h1)
    /-
      case intro.intro.intro.intro.inl
      a b c : Int
      h : Fermat42 a b c
      a0 b0 c0 : Int
      hf : Fermat42.Minimal a0 b0 c0
      hc : Eq (HMod.hMod a0 2) 1
      h1 : LT.lt 0 c0
      ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
    -/
  · use a0, b0, c0
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inl
      a b c : Int
      h : Fermat42 a b c
      a0 b0 c0 : Int
      hf : Fermat42.Minimal a0 b0 c0
      hc : Eq (HMod.hMod a0 2) 1
      h1 : Eq 0 c0
      ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
    -/
  · exfalso
    /-
      case intro.intro.intro.intro.inr.inl
      a b c : Int
      h : Fermat42 a b c
      a0 b0 c0 : Int
      hf : Fermat42.Minimal a0 b0 c0
      hc : Eq (HMod.hMod a0 2) 1
      h1 : Eq 0 c0
      ⊢ False
    -/
    exact ne_zero hf.1 h1.symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inr
      a b c : Int
      h : Fermat42 a b c
      a0 b0 c0 : Int
      hf : Fermat42.Minimal a0 b0 c0
      hc : Eq (HMod.hMod a0 2) 1
      h1 : LT.lt c0 0
      ⊢ Exists fun a0 => Exists fun b0 => Exists fun c0 => And (Fermat42.Minimal a0  …
    -/
  · use a0, b0, -c0, neg_of_minimal hf, hc
    /-
      case right
      a b c : Int
      h : Fermat42 a b c
      a0 b0 c0 : Int
      hf : Fermat42.Minimal a0 b0 c0
      hc : Eq (HMod.hMod a0 2) 1
      h1 : LT.lt c0 0
      ⊢ LT.lt 0 (Neg.neg c0)
    -/
    exact neg_pos.mpr h1
    /-
      🎉 no goals
    -/


theorem Int.coprime_of_sq_sum {r s : ℤ} (h2 : IsCoprime s r) : IsCoprime (r ^ 2 + s ^ 2) r := by
  /-
    r s : Int
    h2 : IsCoprime s r
    ⊢ IsCoprime (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2)) r
  -/
  rw [sq, sq]
  /-
    r s : Int
    h2 : IsCoprime s r
    ⊢ IsCoprime (HAdd.hAdd (HMul.hMul r r) (HMul.hMul s s)) r
  -/
  exact (IsCoprime.mul_left h2 h2).mul_add_left_left r
  /-
    🎉 no goals
  -/


theorem Int.coprime_of_sq_sum' {r s : ℤ} (h : IsCoprime r s) :
    IsCoprime (r ^ 2 + s ^ 2) (r * s) := by
  /-
    r s : Int
    h : IsCoprime r s
    ⊢ IsCoprime (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2)) (HMul.hMul r s)
  -/
  apply IsCoprime.mul_right (Int.coprime_of_sq_sum (isCoprime_comm.mp h))
  /-
    r s : Int
    h : IsCoprime r s
    ⊢ IsCoprime (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2)) s
  -/
  rw [add_comm]; apply Int.coprime_of_sq_sum h
                 /-
                   🎉 no goals
                 -/


theorem not_minimal {a b c : ℤ} (h : Minimal a b c) (ha2 : a % 2 = 1) (hc : 0 < c) : False := by
  -- Use the fact that a ^ 2, b ^ 2, c form a pythagorean triple to obtain m and n such that
  -- a ^ 2 = m ^ 2 - n ^ 2, b ^ 2 = 2 * m * n and c = m ^ 2 + n ^ 2
  -- first the formula:
  have ht : PythagoreanTriple (a ^ 2) (b ^ 2) c := by
    delta PythagoreanTriple
    linear_combination h.1.2.2
  -- coprime requirement:
  /-
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    ⊢ False
  -/
  have h2 : Int.gcd (a ^ 2) (b ^ 2) = 1 := Int.gcd_eq_one_iff_coprime.mpr (coprime_of_minimal h).pow
  -- in order to reduce the possibilities we get from the classification of pythagorean triples
  -- it helps if we know the parity of a ^ 2 (and the sign of c):
  have ha22 : a ^ 2 % 2 = 1 := by
    rw [sq, Int.mul_emod, ha2]
    decide
  /-
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    ⊢ False
  -/
  obtain ⟨m, n, ht1, ht2, ht3, ht4, ht5, ht6⟩ := ht.coprime_classification' h2 ha22 hc
  -- Now a, n, m form a pythagorean triple and so we can obtain r and s such that
  -- a = r ^ 2 - s ^ 2, n = 2 * r * s and m = r ^ 2 + s ^ 2
  -- formula:
  have htt : PythagoreanTriple a n m := by
    delta PythagoreanTriple
    linear_combination ht1
  -- a and n are coprime, because a ^ 2 = m ^ 2 - n ^ 2 and m and n are coprime.
  have h3 : Int.gcd a n = 1 := by
    apply Int.gcd_eq_one_iff_coprime.mpr
    apply @IsCoprime.of_mul_left_left _ _ _ a
    rw [← sq, ht1, (by ring : m ^ 2 - n ^ 2 = m ^ 2 + -n * n)]
    exact (Int.gcd_eq_one_iff_coprime.mp ht4).pow_left.add_mul_right_left (-n)
  -- m is positive because b is non-zero and b ^ 2 = 2 * m * n and we already have 0 ≤ m.
  /-
    case intro.intro.intro.intro.intro.intro.intro
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    ⊢ False
  -/
  have hb20 : b ^ 2 ≠ 0 := mt pow_eq_zero h.1.2.1
  have h4 : 0 < m := by
    apply lt_of_le_of_ne ht6
    rintro rfl
    revert hb20
    rw [ht2]
    simp
  /-
    case intro.intro.intro.intro.intro.intro.intro
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    ⊢ False
  -/
  obtain ⟨r, s, _, htt2, htt3, htt4, htt5, htt6⟩ := htt.coprime_classification' h3 ha2 h4
  -- Now use the fact that (b / 2) ^ 2 = m * r * s, and m, r and s are pairwise coprime to obtain
  -- i, j and k such that m = i ^ 2, r = j ^ 2 and s = k ^ 2.
  -- m and r * s are coprime because m = r ^ 2 + s ^ 2 and r and s are coprime.
  have hcp : Int.gcd m (r * s) = 1 := by
    rw [htt3]
    exact
      Int.gcd_eq_one_iff_coprime.mpr (Int.coprime_of_sq_sum' (Int.gcd_eq_one_iff_coprime.mp htt4))
  -- b is even because b ^ 2 = 2 * m * n.
  have hb2 : 2 ∣ b := by
    apply @Int.Prime.dvd_pow' _ 2 _ Nat.prime_two
    rw [ht2, mul_assoc]
    exact dvd_mul_right 2 (m * n)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq (m.gcd (HMul.hMul r s)) 1
    hb2 : Dvd.dvd 2 b
    ⊢ False
  -/
  cases' hb2 with b' hb2'
  have hs : b' ^ 2 = m * (r * s) := by
    apply (mul_right_inj' (by norm_num : (4 : ℤ) ≠ 0)).mp
    linear_combination (-b - 2 * b') * hb2' + ht2 + 2 * m * htt2
  have hrsz : r * s ≠ 0 := by
    -- because b ^ 2 is not zero and (b / 2) ^ 2 = m * (r * s)
    by_contra hrsz
    revert hb20
    rw [ht2, htt2, mul_assoc, @mul_assoc _ _ _ r s, hrsz]
    simp
  have h2b0 : b' ≠ 0 := by
    apply ne_zero_pow two_ne_zero
    rw [hs]
    apply mul_ne_zero
    · exact ne_of_gt h4
    · exact hrsz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq (m.gcd (HMul.hMul r s)) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul m (HMul.hMul r s))
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := Int.sq_of_gcd_eq_one hcp hs.symm
  -- use m is positive to exclude m = - i ^ 2
  have hi' : ¬m = -i ^ 2 := by
    by_contra h1
    have hit : -i ^ 2 ≤ 0 := neg_nonpos.mpr (sq_nonneg i)
    rw [← h1] at hit
    apply absurd h4 (not_lt.mpr hit)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq (m.gcd (HMul.hMul r s)) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul m (HMul.hMul r s))
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi : Or (Eq m (HPow.hPow i 2)) (Eq m (Neg.neg (HPow.hPow i 2)))
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    ⊢ False
  -/
  replace hi : m = i ^ 2 := Or.resolve_right hi hi'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq (m.gcd (HMul.hMul r s)) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul m (HMul.hMul r s))
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    ⊢ False
  -/
  rw [mul_comm] at hs
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq (m.gcd (HMul.hMul r s)) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    ⊢ False
  -/
  rw [Int.gcd_comm] at hcp
  -- obtain d such that r * s = d ^ 2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    ⊢ False
  -/
  obtain ⟨d, hd⟩ := Int.sq_of_gcd_eq_one hcp hs.symm
  -- (b / 2) ^ 2 and m are positive so r * s is positive
  have hd' : ¬r * s = -d ^ 2 := by
    by_contra h1
    rw [h1] at hs
    have h2 : b' ^ 2 ≤ 0 := by
      rw [hs, (by ring : -d ^ 2 * m = -(d ^ 2 * m))]
      exact neg_nonpos.mpr ((mul_nonneg_iff_of_pos_right h4).mpr (sq_nonneg d))
    have h2' : 0 ≤ b' ^ 2 := by apply sq_nonneg b'
    exact absurd (lt_of_le_of_ne h2' (Ne.symm (pow_ne_zero _ h2b0))) (not_lt.mpr h2)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd : Or (Eq (HMul.hMul r s) (HPow.hPow d 2)) (Eq (HMul.hMul r s) (Neg.neg (HPo …
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    ⊢ False
  -/
  replace hd : r * s = d ^ 2 := Or.resolve_right hd hd'
  -- r = +/- j ^ 2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    hd : Eq (HMul.hMul r s) (HPow.hPow d 2)
    ⊢ False
  -/
  obtain ⟨j, hj⟩ := Int.sq_of_gcd_eq_one htt4 hd
  have hj0 : j ≠ 0 := by
    intro h0
    rw [h0, zero_pow two_ne_zero, neg_zero, or_self_iff] at hj
    apply left_ne_zero_of_mul hrsz hj
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    hd : Eq (HMul.hMul r s) (HPow.hPow d 2)
    j : Int
    hj : Or (Eq r (HPow.hPow j 2)) (Eq r (Neg.neg (HPow.hPow j 2)))
    hj0 : Ne j 0
    ⊢ False
  -/
  rw [mul_comm] at hd
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (r.gcd s) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    hd : Eq (HMul.hMul s r) (HPow.hPow d 2)
    j : Int
    hj : Or (Eq r (HPow.hPow j 2)) (Eq r (Neg.neg (HPow.hPow j 2)))
    hj0 : Ne j 0
    ⊢ False
  -/
  rw [Int.gcd_comm] at htt4
  -- s = +/- k ^ 2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (s.gcd r) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    hd : Eq (HMul.hMul s r) (HPow.hPow d 2)
    j : Int
    hj : Or (Eq r (HPow.hPow j 2)) (Eq r (Neg.neg (HPow.hPow j 2)))
    hj0 : Ne j 0
    ⊢ False
  -/
  obtain ⟨k, hk⟩ := Int.sq_of_gcd_eq_one htt4 hd
  have hk0 : k ≠ 0 := by
    intro h0
    rw [h0, zero_pow two_ne_zero, neg_zero, or_self_iff] at hk
    apply right_ne_zero_of_mul hrsz hk
  have hj2 : r ^ 2 = j ^ 4 := by
    cases' hj with hjp hjp <;>
      · rw [hjp]
        ring
  have hk2 : s ^ 2 = k ^ 4 := by
    cases' hk with hkp hkp <;>
      · rw [hkp]
        ring
  -- from m = r ^ 2 + s ^ 2 we now get a new solution to a ^ 4 + b ^ 4 = c ^ 2:
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (s.gcd r) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    hd : Eq (HMul.hMul s r) (HPow.hPow d 2)
    j : Int
    hj : Or (Eq r (HPow.hPow j 2)) (Eq r (Neg.neg (HPow.hPow j 2)))
    hj0 : Ne j 0
    k : Int
    hk : Or (Eq s (HPow.hPow k 2)) (Eq s (Neg.neg (HPow.hPow k 2)))
    hk0 : Ne k 0
    hj2 : Eq (HPow.hPow r 2) (HPow.hPow j 4)
    hk2 : Eq (HPow.hPow s 2) (HPow.hPow k 4)
    ⊢ False
  -/
  have hh : i ^ 2 = j ^ 4 + k ^ 4 := by rw [← hi, htt3, hj2, hk2]
  have hn : n ≠ 0 := by
    rw [ht2] at hb20
    apply right_ne_zero_of_mul hb20
  -- and it has a smaller c: from c = m ^ 2 + n ^ 2 we see that m is smaller than c, and i ^ 2 = m.
  have hic : Int.natAbs i < Int.natAbs c := by
    apply Int.ofNat_lt.mp
    rw [← Int.eq_natAbs_of_zero_le (le_of_lt hc)]
    apply gt_of_gt_of_ge _ (Int.natAbs_le_self_sq i)
    rw [← hi, ht3]
    apply gt_of_gt_of_ge _ (Int.le_self_sq m)
    exact lt_add_of_pos_right (m ^ 2) (sq_pos_of_ne_zero hn)
  have hic' : Int.natAbs c ≤ Int.natAbs i := by
    apply h.2 j k i
    exact ⟨hj0, hk0, hh.symm⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    a b c : Int
    h : Fermat42.Minimal a b c
    ha2 : Eq (HMod.hMod a 2) 1
    hc : LT.lt 0 c
    ht : PythagoreanTriple (HPow.hPow a 2) (HPow.hPow b 2) c
    h2 : Eq ((HPow.hPow a 2).gcd (HPow.hPow b 2)) 1
    ha22 : Eq (HMod.hMod (HPow.hPow a 2) 2) 1
    m n : Int
    ht1 : Eq (HPow.hPow a 2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    ht2 : Eq (HPow.hPow b 2) (HMul.hMul (HMul.hMul 2 m) n)
    ht3 : Eq c (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ht4 : Eq (m.gcd n) 1
    ht5 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ht6 : LE.le 0 m
    htt : PythagoreanTriple a n m
    h3 : Eq (a.gcd n) 1
    hb20 : Ne (HPow.hPow b 2) 0
    h4 : LT.lt 0 m
    r s : Int
    left✝ : Eq a (HSub.hSub (HPow.hPow r 2) (HPow.hPow s 2))
    htt2 : Eq n (HMul.hMul (HMul.hMul 2 r) s)
    htt3 : Eq m (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    htt4 : Eq (s.gcd r) 1
    htt5 : Or (And (Eq (HMod.hMod r 2) 0) (Eq (HMod.hMod s 2) 1)) (And (Eq (HMod.h …
    htt6 : LE.le 0 r
    hcp : Eq ((HMul.hMul r s).gcd m) 1
    b' : Int
    hb2' : Eq b (HMul.hMul 2 b')
    hs : Eq (HPow.hPow b' 2) (HMul.hMul (HMul.hMul r s) m)
    hrsz : Ne (HMul.hMul r s) 0
    h2b0 : Ne b' 0
    i : Int
    hi' : Not (Eq m (Neg.neg (HPow.hPow i 2)))
    hi : Eq m (HPow.hPow i 2)
    d : Int
    hd' : Not (Eq (HMul.hMul r s) (Neg.neg (HPow.hPow d 2)))
    hd : Eq (HMul.hMul s r) (HPow.hPow d 2)
    j : Int
    hj : Or (Eq r (HPow.hPow j 2)) (Eq r (Neg.neg (HPow.hPow j 2)))
    hj0 : Ne j 0
    k : Int
    hk : Or (Eq s (HPow.hPow k 2)) (Eq s (Neg.neg (HPow.hPow k 2)))
    hk0 : Ne k 0
    hj2 : Eq (HPow.hPow r 2) (HPow.hPow j 4)
    hk2 : Eq (HPow.hPow s 2) (HPow.hPow k 4)
    hh : Eq (HPow.hPow i 2) (HAdd.hAdd (HPow.hPow j 4) (HPow.hPow k 4))
    hn : Ne n 0
    hic : LT.lt i.natAbs c.natAbs
    hic' : LE.le c.natAbs i.natAbs
    ⊢ False
  -/
  apply absurd (not_le_of_lt hic) (not_not.mpr hic')
  /-
    🎉 no goals
  -/


theorem not_fermat_42 {a b c : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) : a ^ 4 + b ^ 4 ≠ c ^ 2 := by
  /-
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Ne (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
  -/
  intro h
  obtain ⟨a0, b0, c0, ⟨hf, h2, hp⟩⟩ :=
    Fermat42.exists_pos_odd_minimal (And.intro ha (And.intro hb h))
  /-
    case intro.intro.intro.intro.intro
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    h : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 2)
    a0 b0 c0 : Int
    hf : Fermat42.Minimal a0 b0 c0
    h2 : Eq (HMod.hMod a0 2) 1
    hp : LT.lt 0 c0
    ⊢ False
  -/
  apply Fermat42.not_minimal hf h2 hp
  /-
    🎉 no goals
  -/


/--
Fermat's Last Theorem for $n=4$: if `a b c : ℕ` are all non-zero
then `a ^ 4 + b ^ 4 ≠ c ^ 4`.
-/
theorem fermatLastTheoremFour : FermatLastTheoremFor 4 := by
  /-
    ⊢ FermatLastTheoremFor 4
  -/
  rw [fermatLastTheoremFor_iff_int]
  /-
    ⊢ FermatLastTheoremWith Int 4
  -/
  intro a b c ha hb _ heq
  /-
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    a✝ : Ne c 0
    heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 4)
    ⊢ False
  -/
  apply @not_fermat_42 _ _ (c ^ 2) ha hb
  /-
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    a✝ : Ne c 0
    heq : Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow c 4)
    ⊢ Eq (HAdd.hAdd (HPow.hPow a 4) (HPow.hPow b 4)) (HPow.hPow (HPow.hPow c 2) 2)
  -/
  rw [heq]; ring
            /-
              🎉 no goals
            -/


/--
To prove Fermat's Last Theorem, it suffices to prove it for odd prime exponents.
-/
theorem FermatLastTheorem.of_odd_primes
    (hprimes : ∀ p : ℕ, Nat.Prime p → Odd p → FermatLastTheoremFor p) : FermatLastTheorem := by
  /-
    hprimes : ∀ (p : Nat), Nat.Prime p → Odd p → FermatLastTheoremFor p
    ⊢ FermatLastTheorem
  -/
  intro n h
  /-
    hprimes : ∀ (p : Nat), Nat.Prime p → Odd p → FermatLastTheoremFor p
    n : Nat
    h : GE.ge n 3
    ⊢ FermatLastTheoremFor n
  -/
  obtain hdvd|⟨p, hpprime, hdvd, hpodd⟩ := Nat.four_dvd_or_exists_odd_prime_and_dvd_of_two_lt h <;>
    /-
      case inl
      hprimes : ∀ (p : Nat), Nat.Prime p → Odd p → FermatLastTheoremFor p
      n : Nat
      h : GE.ge n 3
      hdvd : Dvd.dvd 4 n
      ⊢ FermatLastTheoremFor n
    -/
    apply FermatLastTheoremWith.mono hdvd
    /-
      case inl
      hprimes : ∀ (p : Nat), Nat.Prime p → Odd p → FermatLastTheoremFor p
      n : Nat
      h : GE.ge n 3
      hdvd : Dvd.dvd 4 n
      ⊢ FermatLastTheoremWith Nat 4
    -/
  · exact fermatLastTheoremFour
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro
      hprimes : ∀ (p : Nat), Nat.Prime p → Odd p → FermatLastTheoremFor p
      n : Nat
      h : GE.ge n 3
      p : Nat
      hpprime : Nat.Prime p
      hdvd : Dvd.dvd p n
      hpodd : Odd p
      ⊢ FermatLastTheoremWith Nat p
    -/
  · exact hprimes p hpprime hpodd
    /-
      🎉 no goals
    -/

