/-- For a natural number `n`, the pre-game `powHalf (n + 1)` is recursively defined as
`{0 | powHalf n}`. These are the explicit expressions of powers of `1 / 2`. By definition, we have
`powHalf 0 = 1` and `powHalf 1 ≈ 1 / 2` and we prove later on that
`powHalf (n + 1) + powHalf (n + 1) ≈ powHalf n`. -/
def powHalf : ℕ → PGame
  | 0 => 1
  | n + 1 => ⟨PUnit, PUnit, 0, fun _ => powHalf n⟩


@[simp]
theorem powHalf_zero : powHalf 0 = 1 :=
  rfl


                                                                    /-
                                                                      n : Nat
                                                                      ⊢ Eq (SetTheory.PGame.powHalf n).LeftMoves PUnit.{u_1 + 1}
                                                                    -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
theorem powHalf_leftMoves (n) : (powHalf n).LeftMoves = PUnit := by cases n <;> rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem powHalf_zero_rightMoves : (powHalf 0).RightMoves = PEmpty :=
  rfl


theorem powHalf_succ_rightMoves (n) : (powHalf (n + 1)).RightMoves = PUnit :=
  rfl


@[simp]
                                                                  /-
                                                                    n : Nat
                                                                    i : (SetTheory.PGame.powHalf n).LeftMoves
                                                                    ⊢ Eq ((SetTheory.PGame.powHalf n).moveLeft i) 0
                                                                  -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
theorem powHalf_moveLeft (n i) : (powHalf n).moveLeft i = 0 := by cases n <;> cases i <;> rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
theorem powHalf_succ_moveRight (n i) : (powHalf (n + 1)).moveRight i = powHalf n :=
  rfl


instance uniquePowHalfLeftMoves (n) : Unique (powHalf n).LeftMoves := by
  /-
    n : Nat
    ⊢ Unique (SetTheory.PGame.powHalf n).LeftMoves
  -/
              /-
                🎉 no goals
              -/
  cases n <;> exact PUnit.instUnique
              /-
                🎉 no goals
              -/


instance isEmpty_powHalf_zero_rightMoves : IsEmpty (powHalf 0).RightMoves :=
  inferInstanceAs (IsEmpty PEmpty)


instance uniquePowHalfSuccRightMoves (n) : Unique (powHalf (n + 1)).RightMoves :=
  PUnit.instUnique


@[simp]
theorem birthday_half : birthday (powHalf 1) = 2 := by
  /-
    ⊢ Eq (SetTheory.PGame.powHalf 1).birthday 2
  -/
  rw [birthday_def]; simp
                     /-
                       🎉 no goals
                     -/


/-- For all natural numbers `n`, the pre-games `powHalf n` are numeric. -/
theorem numeric_powHalf (n) : (powHalf n).Numeric := by
  induction n with
  | zero => exact numeric_one
  | succ n hn =>
    constructor
    · simpa using hn.moveLeft_lt default
    · exact ⟨fun _ => numeric_zero, fun _ => hn⟩


theorem powHalf_succ_lt_powHalf (n : ℕ) : powHalf (n + 1) < powHalf n :=
  (numeric_powHalf (n + 1)).lt_moveRight default


theorem powHalf_succ_le_powHalf (n : ℕ) : powHalf (n + 1) ≤ powHalf n :=
  (powHalf_succ_lt_powHalf n).le


theorem powHalf_le_one (n : ℕ) : powHalf n ≤ 1 := by
  induction n with
  | zero => exact le_rfl
  | succ n hn => exact (powHalf_succ_le_powHalf n).trans hn


theorem powHalf_succ_lt_one (n : ℕ) : powHalf (n + 1) < 1 :=
  (powHalf_succ_lt_powHalf n).trans_le <| powHalf_le_one n


theorem powHalf_pos (n : ℕ) : 0 < powHalf n := by
  /-
    n : Nat
    ⊢ LT.lt 0 (SetTheory.PGame.powHalf n)
  -/
  rw [← lf_iff_lt numeric_zero (numeric_powHalf n), zero_lf_le]; simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem zero_le_powHalf (n : ℕ) : 0 ≤ powHalf n :=
  (powHalf_pos n).le


theorem add_powHalf_succ_self_eq_powHalf (n) : powHalf (n + 1) + powHalf (n + 1) ≈ powHalf n := by
  /-
    n : Nat
    ⊢ HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd n 1)) (SetTheo …
  -/
  induction' n using Nat.strong_induction_on with n hn
  /-
    case h
    n : Nat
    hn : ∀ (m : Nat), LT.lt m n → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
    ⊢ HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd n 1)) (SetTheo …
  -/
  constructor <;> rw [le_iff_forall_lf] <;> constructor
    /-
      case h.left.left
      n : Nat
      hn : ∀ (m : Nat), LT.lt m n → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
      ⊢ ∀ (i : (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd n 1)) (SetTheory.PGame …
    -/
  · rintro (⟨⟨⟩⟩ | ⟨⟨⟩⟩) <;> apply lf_of_lt
    · calc
        0 + powHalf n.succ ≈ powHalf n.succ := zero_add_equiv _
        _ < powHalf n := powHalf_succ_lt_powHalf n
    · calc
        powHalf n.succ + 0 ≈ powHalf n.succ := add_zero_equiv _
        _ < powHalf n := powHalf_succ_lt_powHalf n
    /-
      case h.left.right
      n : Nat
      hn : ∀ (m : Nat), LT.lt m n → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
      ⊢ ∀ (j : (SetTheory.PGame.powHalf n).RightMoves), (HAdd.hAdd (SetTheory.PGame. …
    -/
  · cases' n with n
      /-
        case h.left.right.zero
        hn : ∀ (m : Nat), LT.lt m 0 → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
        ⊢ ∀ (j : (SetTheory.PGame.powHalf 0).RightMoves), (HAdd.hAdd (SetTheory.PGame. …
      -/
    · rintro ⟨⟩
      /-
        🎉 no goals
      -/
    /-
      case h.left.right.succ
      n : Nat
      hn : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → HasEquiv.Equiv (HAdd.hAdd (SetTheo …
      ⊢ ∀ (j : (SetTheory.PGame.powHalf (HAdd.hAdd n 1)).RightMoves), (HAdd.hAdd (Se …
    -/
    rintro ⟨⟩
    /-
      case h.left.right.succ.unit
      n : Nat
      hn : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → HasEquiv.Equiv (HAdd.hAdd (SetTheo …
      ⊢ (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd (HAdd.hAdd n 1) 1)) (SetTheor …
    -/
    apply lf_of_moveRight_le
    /-
      case h.left.right.succ.unit.h
      n : Nat
      hn : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → HasEquiv.Equiv (HAdd.hAdd (SetTheo …
      ⊢ LE.le ((HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd (HAdd.hAdd n 1) 1)) (S …
    -/
    swap
      /-
        case h.left.right.succ.unit.j
        n : Nat
        hn : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → HasEquiv.Equiv (HAdd.hAdd (SetTheo …
        ⊢ (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd (HAdd.hAdd n 1) 1)) (SetTheor …
      -/
    · exact Sum.inl default
      /-
        🎉 no goals
      -/
    calc
      powHalf n.succ + powHalf (n.succ + 1) ≤ powHalf n.succ + powHalf n.succ :=
        add_le_add_left (powHalf_succ_le_powHalf _) _
      _ ≈ powHalf n := hn _ (Nat.lt_succ_self n)
    /-
      case h.right.left
      n : Nat
      hn : ∀ (m : Nat), LT.lt m n → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
      ⊢ ∀ (i : (SetTheory.PGame.powHalf n).LeftMoves), ((SetTheory.PGame.powHalf n). …
    -/
  · simp only [powHalf_moveLeft, forall_const]
    /-
      case h.right.left
      n : Nat
      hn : ∀ (m : Nat), LT.lt m n → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
      ⊢ SetTheory.PGame.LF 0 (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd n 1)) (S …
    -/
    apply lf_of_lt
    calc
      0 ≈ 0 + 0 := Equiv.symm (add_zero_equiv 0)
      _ ≤ powHalf n.succ + 0 := add_le_add_right (zero_le_powHalf _) _
      _ < powHalf n.succ + powHalf n.succ := add_lt_add_left (powHalf_pos _) _
    /-
      case h.right.right
      n : Nat
      hn : ∀ (m : Nat), LT.lt m n → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.powHa …
      ⊢ ∀ (j : (HAdd.hAdd (SetTheory.PGame.powHalf (HAdd.hAdd n 1)) (SetTheory.PGame …
    -/
  · rintro (⟨⟨⟩⟩ | ⟨⟨⟩⟩) <;> apply lf_of_lt
    · calc
        powHalf n ≈ powHalf n + 0 := Equiv.symm (add_zero_equiv _)
        _ < powHalf n + powHalf n.succ := add_lt_add_left (powHalf_pos _) _
    · calc
        powHalf n ≈ 0 + powHalf n := Equiv.symm (zero_add_equiv _)
        _ < powHalf n.succ + powHalf n := add_lt_add_right (powHalf_pos _) _


theorem half_add_half_equiv_one : powHalf 1 + powHalf 1 ≈ 1 :=
  add_powHalf_succ_self_eq_powHalf 0


/-- Powers of the surreal number `half`. -/
def powHalf (n : ℕ) : Surreal :=
  ⟦⟨PGame.powHalf n, PGame.numeric_powHalf n⟩⟧


@[simp]
theorem double_powHalf_succ_eq_powHalf (n : ℕ) : 2 * powHalf (n + 1) = powHalf n := by
  /-
    n : Nat
    ⊢ Eq (HMul.hMul 2 (Surreal.powHalf (HAdd.hAdd n 1))) (Surreal.powHalf n)
  -/
  rw [two_mul]; exact Quotient.sound (PGame.add_powHalf_succ_self_eq_powHalf n)
                /-
                  🎉 no goals
                -/


@[simp]
theorem nsmul_pow_two_powHalf (n : ℕ) : 2 ^ n * powHalf n = 1 := by
  /-
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf n)) 1
  -/
  induction' n with n hn
    /-
      case zero
      ⊢ Eq (HMul.hMul (HPow.hPow 2 0) (Surreal.powHalf 0)) 1
    -/
  · simp only [pow_zero, powHalf_zero, mul_one]
    /-
      🎉 no goals
    -/
  · rw [← hn, ← double_powHalf_succ_eq_powHalf n, ← mul_assoc (2 ^ n) 2 (powHalf (n + 1)),
      pow_succ', mul_comm 2 (2 ^ n)]


@[simp]
theorem nsmul_pow_two_powHalf' (n k : ℕ) : 2 ^ n * powHalf (n + k) = powHalf k := by
  /-
    n k : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd n k))) (Surreal.po …
  -/
  induction' k with k hk
  · simp only [add_zero, Surreal.nsmul_pow_two_powHalf, eq_self_iff_true,
      Surreal.powHalf_zero]
  · rw [← double_powHalf_succ_eq_powHalf (n + k), ← double_powHalf_succ_eq_powHalf k,
      ← mul_assoc, mul_comm (2 ^ n) 2, mul_assoc] at hk
    /-
      case succ
      n k : Nat
      hk : Eq (HMul.hMul 2 (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd (H …
      ⊢ Eq (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd n (HAdd.hAdd k 1)) …
    -/
    rw [← zsmul_eq_zsmul_iff' two_ne_zero]
    /-
      case succ
      n k : Nat
      hk : Eq (HMul.hMul 2 (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd (H …
      ⊢ Eq (HSMul.hSMul 2 (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd n ( …
    -/
    simpa only [zsmul_eq_mul, Int.cast_ofNat]
    /-
      🎉 no goals
    -/


theorem zsmul_pow_two_powHalf (m : ℤ) (n k : ℕ) :
    (m * 2 ^ n) * powHalf (n + k) = m * powHalf k := by
  /-
    m : Int
    n k : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (↑m) (HPow.hPow 2 n)) (Surreal.powHalf (HAdd.hAdd n …
  -/
  rw [mul_assoc]
  /-
    m : Int
    n k : Nat
    ⊢ Eq (HMul.hMul (↑m) (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd n  …
  -/
  congr
  /-
    case e_a
    m : Int
    n k : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow 2 n) (Surreal.powHalf (HAdd.hAdd n k))) (Surreal.po …
  -/
  exact nsmul_pow_two_powHalf' n k
  /-
    🎉 no goals
  -/


theorem dyadic_aux {m₁ m₂ : ℤ} {y₁ y₂ : ℕ} (h₂ : m₁ * 2 ^ y₁ = m₂ * 2 ^ y₂) :
    m₁ * powHalf y₂ = m₂ * powHalf y₁ := by
  /-
    m₁ m₂ : Int
    y₁ y₂ : Nat
    h₂ : Eq (HMul.hMul m₁ (HPow.hPow 2 y₁)) (HMul.hMul m₂ (HPow.hPow 2 y₂))
    ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf y₂)) (HMul.hMul (↑m₂) (Surreal.powHalf  …
  -/
  revert m₁ m₂
  /-
    y₁ y₂ : Nat
    ⊢ ∀ {m₁ m₂ : Int}, Eq (HMul.hMul m₁ (HPow.hPow 2 y₁)) (HMul.hMul m₂ (HPow.hPow …
  -/
  wlog h : y₁ ≤ y₂
    /-
      case inr
      y₁ y₂ : Nat
      this : ∀ {y₁ y₂ : Nat}, LE.le y₁ y₂ → ∀ {m₁ m₂ : Int}, Eq (HMul.hMul m₁ (HPow. …
      h : Not (LE.le y₁ y₂)
      ⊢ ∀ {m₁ m₂ : Int}, Eq (HMul.hMul m₁ (HPow.hPow 2 y₁)) (HMul.hMul m₂ (HPow.hPow …
    -/
  · intro m₁ m₂ aux; exact (this (le_of_not_le h) aux.symm).symm
                     /-
                       🎉 no goals
                     -/
  /-
    y₁ y₂ : Nat
    h : LE.le y₁ y₂
    ⊢ ∀ {m₁ m₂ : Int}, Eq (HMul.hMul m₁ (HPow.hPow 2 y₁)) (HMul.hMul m₂ (HPow.hPow …
  -/
  intro m₁ m₂ h₂
  /-
    y₁ y₂ : Nat
    h : LE.le y₁ y₂
    m₁ m₂ : Int
    h₂ : Eq (HMul.hMul m₁ (HPow.hPow 2 y₁)) (HMul.hMul m₂ (HPow.hPow 2 y₂))
    ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf y₂)) (HMul.hMul (↑m₂) (Surreal.powHalf  …
  -/
  obtain ⟨c, rfl⟩ := le_iff_exists_add.mp h
  /-
    case intro
    y₁ : Nat
    m₁ m₂ : Int
    c : Nat
    h : LE.le y₁ (HAdd.hAdd y₁ c)
    h₂ : Eq (HMul.hMul m₁ (HPow.hPow 2 y₁)) (HMul.hMul m₂ (HPow.hPow 2 (HAdd.hAdd  …
    ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (HAdd.hAdd y₁ c))) (HMul.hMul (↑m₂) (Su …
  -/
  rw [add_comm, pow_add, ← mul_assoc, mul_eq_mul_right_iff] at h₂
  /-
    case intro
    y₁ : Nat
    m₁ m₂ : Int
    c : Nat
    h : LE.le y₁ (HAdd.hAdd y₁ c)
    h₂ : Or (Eq m₁ (HMul.hMul m₂ (HPow.hPow 2 c))) (Eq (HPow.hPow 2 y₁) 0)
    ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (HAdd.hAdd y₁ c))) (HMul.hMul (↑m₂) (Su …
  -/
  cases' h₂ with h₂ h₂
    /-
      case intro.inl
      y₁ : Nat
      m₁ m₂ : Int
      c : Nat
      h : LE.le y₁ (HAdd.hAdd y₁ c)
      h₂ : Eq m₁ (HMul.hMul m₂ (HPow.hPow 2 c))
      ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (HAdd.hAdd y₁ c))) (HMul.hMul (↑m₂) (Su …
    -/
  · rw [h₂, add_comm]
    /-
      case intro.inl
      y₁ : Nat
      m₁ m₂ : Int
      c : Nat
      h : LE.le y₁ (HAdd.hAdd y₁ c)
      h₂ : Eq m₁ (HMul.hMul m₂ (HPow.hPow 2 c))
      ⊢ Eq (HMul.hMul (↑(HMul.hMul m₂ (HPow.hPow 2 c))) (Surreal.powHalf (HAdd.hAdd  …
    -/
    simp_rw [Int.cast_mul, Int.cast_pow, Int.cast_ofNat, zsmul_pow_two_powHalf m₂ c y₁]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      y₁ : Nat
      m₁ m₂ : Int
      c : Nat
      h : LE.le y₁ (HAdd.hAdd y₁ c)
      h₂ : Eq (HPow.hPow 2 y₁) 0
      ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (HAdd.hAdd y₁ c))) (HMul.hMul (↑m₂) (Su …
    -/
  · have := Nat.one_le_pow y₁ 2 Nat.succ_pos'
    /-
      case intro.inr
      y₁ : Nat
      m₁ m₂ : Int
      c : Nat
      h : LE.le y₁ (HAdd.hAdd y₁ c)
      h₂ : Eq (HPow.hPow 2 y₁) 0
      this : LE.le 1 (HPow.hPow 2 y₁)
      ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (HAdd.hAdd y₁ c))) (HMul.hMul (↑m₂) (Su …
    -/
    norm_cast at h₂; omega
                     /-
                       🎉 no goals
                     -/


/-- The additive monoid morphism `dyadicMap` sends ⟦⟨m, 2^n⟩⟧ to m • half ^ n. -/
noncomputable def dyadicMap : Localization.Away (2 : ℤ) →+ Surreal where
  toFun x :=
    (Localization.liftOn x fun x y => x * powHalf (Submonoid.log y)) <| by
      /-
        x : Localization.Away 2
        ⊢ ∀ {a c : Int} {b d : Subtype fun x => Membership.mem (Submonoid.powers 2) x} …
      -/
      intro m₁ m₂ n₁ n₂ h₁
      /-
        x : Localization.Away 2
        m₁ m₂ : Int
        n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
        h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
        ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
      -/
      obtain ⟨⟨n₃, y₃, hn₃⟩, h₂⟩ := Localization.r_iff_exists.mp h₁
      /-
        case intro.mk.intro
        x : Localization.Away 2
        m₁ m₂ : Int
        n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
        h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
        n₃ : Int
        y₃ : Nat
        hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
        h₂ : Eq (HMul.hMul (↑⟨n₃, ⋯⟩) (HMul.hMul ↑{ fst := m₂, snd := n₂ }.2 { fst :=  …
        ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
      -/
      simp only [Subtype.coe_mk, mul_eq_mul_left_iff] at h₂
      /-
        case intro.mk.intro
        x : Localization.Away 2
        m₁ m₂ : Int
        n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
        h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
        n₃ : Int
        y₃ : Nat
        hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
        h₂ : Or (Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)) (Eq n₃ 0)
        ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
      -/
      cases h₂
        /-
          case intro.mk.intro.inl
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
      · obtain ⟨a₁, ha₁⟩ := n₁.prop
        /-
          case intro.mk.intro.inl.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq ((fun x => HPow.hPow 2 x) a₁) ↑n₁
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        obtain ⟨a₂, ha₂⟩ := n₂.prop
        /-
          case intro.mk.intro.inl.intro.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq ((fun x => HPow.hPow 2 x) a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq ((fun x => HPow.hPow 2 x) a₂) ↑n₂
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        simp only at ha₁ ha₂ ⊢
        /-
          case intro.mk.intro.inl.intro.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq (HPow.hPow 2 a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq (HPow.hPow 2 a₂) ↑n₂
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        have hn₁ : n₁ = Submonoid.pow 2 a₁ := Subtype.ext ha₁.symm
        /-
          case intro.mk.intro.inl.intro.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq (HPow.hPow 2 a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq (HPow.hPow 2 a₂) ↑n₂
          hn₁ : Eq n₁ (Submonoid.pow 2 a₁)
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        have hn₂ : n₂ = Submonoid.pow 2 a₂ := Subtype.ext ha₂.symm
        /-
          case intro.mk.intro.inl.intro.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq (HPow.hPow 2 a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq (HPow.hPow 2 a₂) ↑n₂
          hn₁ : Eq n₁ (Submonoid.pow 2 a₁)
          hn₂ : Eq n₂ (Submonoid.pow 2 a₂)
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        have h₂ : 1 < (2 : ℤ).natAbs := one_lt_two
        /-
          case intro.mk.intro.inl.intro.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq (HPow.hPow 2 a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq (HPow.hPow 2 a₂) ↑n₂
          hn₁ : Eq n₁ (Submonoid.pow 2 a₁)
          hn₂ : Eq n₂ (Submonoid.pow 2 a₂)
          h₂ : LT.lt 1 (Int.natAbs 2)
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        rw [hn₁, hn₂, Submonoid.log_pow_int_eq_self h₂, Submonoid.log_pow_int_eq_self h₂]
        /-
          case intro.mk.intro.inl.intro.intro
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq (HPow.hPow 2 a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq (HPow.hPow 2 a₂) ↑n₂
          hn₁ : Eq n₁ (Submonoid.pow 2 a₁)
          hn₂ : Eq n₂ (Submonoid.pow 2 a₂)
          h₂ : LT.lt 1 (Int.natAbs 2)
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf a₁)) (HMul.hMul (↑m₂) (Surreal.powHalf  …
        -/
        apply dyadic_aux
        /-
          case intro.mk.intro.inl.intro.intro.h₂
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq (HMul.hMul (↑n₂) m₁) (HMul.hMul (↑n₁) m₂)
          a₁ : Nat
          ha₁ : Eq (HPow.hPow 2 a₁) ↑n₁
          a₂ : Nat
          ha₂ : Eq (HPow.hPow 2 a₂) ↑n₂
          hn₁ : Eq n₁ (Submonoid.pow 2 a₁)
          hn₂ : Eq n₂ (Submonoid.pow 2 a₂)
          h₂ : LT.lt 1 (Int.natAbs 2)
          ⊢ Eq (HMul.hMul m₁ (HPow.hPow 2 a₂)) (HMul.hMul m₂ (HPow.hPow 2 a₁))
        -/
        rwa [ha₁, ha₂, mul_comm, mul_comm m₂]
        /-
          🎉 no goals
        -/
        /-
          case intro.mk.intro.inr
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq n₃ 0
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
      · have : (1 : ℤ) ≤ 2 ^ y₃ := mod_cast Nat.one_le_pow y₃ 2 Nat.succ_pos'
        /-
          case intro.mk.intro.inr
          x : Localization.Away 2
          m₁ m₂ : Int
          n₁ n₂ : Subtype fun x => Membership.mem (Submonoid.powers 2) x
          h₁ : (Localization.r (Submonoid.powers 2)) { fst := m₁, snd := n₁ } { fst := m …
          n₃ : Int
          y₃ : Nat
          hn₃ : Eq ((fun x => HPow.hPow 2 x) y₃) n₃
          h✝ : Eq n₃ 0
          this : LE.le 1 (HPow.hPow 2 y₃)
          ⊢ Eq (HMul.hMul (↑m₁) (Surreal.powHalf (Submonoid.log n₁))) (HMul.hMul (↑m₂) ( …
        -/
        linarith
        /-
          🎉 no goals
        -/
                  /-
                    ⊢ Eq ((fun x => Localization.liftOn x (fun x y => HMul.hMul (↑x) (Surreal.powH …
                  -/
  map_zero' := by simp_rw [Localization.liftOn_zero _ _, Int.cast_zero, zero_mul]
                  /-
                    🎉 no goals
                  -/
  map_add' x y :=
    Localization.induction_on₂ x y <| by
      /-
        x y : Localization.Away 2
        ⊢ ∀ (x y : Prod Int (Subtype fun x => Membership.mem (Submonoid.powers 2) x)), …
      -/
      rintro ⟨a, ⟨b, ⟨b', rfl⟩⟩⟩ ⟨c, ⟨d, ⟨d', rfl⟩⟩⟩
      /-
        case mk.mk.intro.mk.mk.intro
        x y : Localization.Away 2
        a : Int
        b' : Nat
        c : Int
        d' : Nat
        ⊢ Eq ({ toFun := fun x => Localization.liftOn x (fun x y => HMul.hMul (↑x) (Su …
      -/
      have h₂ : 1 < (2 : ℤ).natAbs := one_lt_two
      /-
        case mk.mk.intro.mk.mk.intro
        x y : Localization.Away 2
        a : Int
        b' : Nat
        c : Int
        d' : Nat
        h₂ : LT.lt 1 (Int.natAbs 2)
        ⊢ Eq ({ toFun := fun x => Localization.liftOn x (fun x y => HMul.hMul (↑x) (Su …
      -/
      have hpow₂ := Submonoid.log_pow_int_eq_self h₂
      /-
        case mk.mk.intro.mk.mk.intro
        x y : Localization.Away 2
        a : Int
        b' : Nat
        c : Int
        d' : Nat
        h₂ : LT.lt 1 (Int.natAbs 2)
        hpow₂ : ∀ (m : Nat), Eq (Submonoid.log (Submonoid.pow 2 m)) m
        ⊢ Eq ({ toFun := fun x => Localization.liftOn x (fun x y => HMul.hMul (↑x) (Su …
      -/
      simp_rw [Submonoid.pow_apply] at hpow₂
      simp_rw [Localization.add_mk, Localization.liftOn_mk,
        Submonoid.log_mul (Int.pow_right_injective h₂), hpow₂]
      /-
        case mk.mk.intro.mk.mk.intro
        x y : Localization.Away 2
        a : Int
        b' : Nat
        c : Int
        d' : Nat
        h₂ : LT.lt 1 (Int.natAbs 2)
        hpow₂ : ∀ (m : Nat), Eq (Submonoid.log ⟨HPow.hPow 2 m, ⋯⟩) m
        ⊢ Eq (HMul.hMul (↑(HAdd.hAdd (HMul.hMul (HPow.hPow 2 b') c) (HMul.hMul (HPow.h …
      -/
      simp only [Int.cast_add, Int.cast_mul, Int.cast_pow, Int.cast_ofNat]
      calc
        (2 ^ b' * c + 2 ^ d' * a) * powHalf (b' + d') =
            (c * 2 ^ b') * powHalf (b' + d') + (a * 2 ^ d') * powHalf (d' + b') := by
          simp only [right_distrib, mul_comm, add_comm]
        _ = c * powHalf d' + a * powHalf b' := by simp only [zsmul_pow_two_powHalf]
        _ = a * powHalf b' + c * powHalf d' := add_comm _ _


@[simp]
theorem dyadicMap_apply (m : ℤ) (p : Submonoid.powers (2 : ℤ)) :
    dyadicMap (IsLocalization.mk' (Localization (Submonoid.powers 2)) m p) =
      m * powHalf (Submonoid.log p) := by
  /-
    m : Int
    p : Subtype fun x => Membership.mem (Submonoid.powers 2) x
    ⊢ Eq (Surreal.dyadicMap (IsLocalization.mk' (Localization (Submonoid.powers 2) …
  -/
  rw [← Localization.mk_eq_mk']; rfl
                                 /-
                                   🎉 no goals
                                 -/

-- @[simp] -- Porting note: simp normal form is `dyadicMap_apply_pow'`

theorem dyadicMap_apply_pow (m : ℤ) (n : ℕ) :
    dyadicMap (IsLocalization.mk' (Localization (Submonoid.powers 2)) m (Submonoid.pow 2 n)) =
      m • powHalf n := by
  /-
    m : Int
    n : Nat
    ⊢ Eq (Surreal.dyadicMap (IsLocalization.mk' (Localization (Submonoid.powers 2) …
  -/
  rw [dyadicMap_apply, @Submonoid.log_pow_int_eq_self 2 one_lt_two]
  /-
    m : Int
    n : Nat
    ⊢ Eq (HMul.hMul (↑m) (Surreal.powHalf n)) (HSMul.hSMul m (Surreal.powHalf n))
  -/
  simp only [zsmul_eq_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem dyadicMap_apply_pow' (m : ℤ) (n : ℕ) :
    m * Surreal.powHalf (Submonoid.log (Submonoid.pow (2 : ℤ) n)) = m * powHalf n := by
  /-
    m : Int
    n : Nat
    ⊢ Eq (HMul.hMul (↑m) (Surreal.powHalf (Submonoid.log (Submonoid.pow 2 n)))) (H …
  -/
  rw [@Submonoid.log_pow_int_eq_self 2 one_lt_two]
  /-
    🎉 no goals
  -/


/-- We define dyadic surreals as the range of the map `dyadicMap`. -/
def dyadic : Set Surreal :=
  Set.range dyadicMap

-- We conclude with some ideas for further work on surreals; these would make fun projects.
-- TODO show that the map from dyadic rationals to surreals is injective
-- TODO map the reals into the surreals, using dyadic Dedekind cuts
-- TODO show this is a group homomorphism, and injective
-- TODO show the maps from the dyadic rationals and from the reals
-- into the surreals are multiplicative

