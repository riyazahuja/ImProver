lemma cast_mono : Monotone (Int.cast : ℤ → R) := by
  /-
    R : Type u_1
    inst✝³ : AddCommGroupWithOne R
    inst✝² : PartialOrder R
    inst✝¹ : AddLeftMono R
    inst✝ : ZeroLEOneClass R
    ⊢ Monotone Int.cast
  -/
  intro m n h
  /-
    R : Type u_1
    inst✝³ : AddCommGroupWithOne R
    inst✝² : PartialOrder R
    inst✝¹ : AddLeftMono R
    inst✝ : ZeroLEOneClass R
    m n : Int
    h : LE.le m n
    ⊢ LE.le ↑m ↑n
  -/
  rw [← sub_nonneg] at h
  /-
    R : Type u_1
    inst✝³ : AddCommGroupWithOne R
    inst✝² : PartialOrder R
    inst✝¹ : AddLeftMono R
    inst✝ : ZeroLEOneClass R
    m n : Int
    h : LE.le 0 (HSub.hSub n m)
    ⊢ LE.le ↑m ↑n
  -/
  lift n - m to ℕ using h with k hk
  /-
    case intro
    R : Type u_1
    inst✝³ : AddCommGroupWithOne R
    inst✝² : PartialOrder R
    inst✝¹ : AddLeftMono R
    inst✝ : ZeroLEOneClass R
    m n : Int
    k : Nat
    hk : Eq (↑k) (HSub.hSub n m)
    ⊢ LE.le ↑m ↑n
  -/
  rw [← sub_nonneg, ← cast_sub, ← hk, cast_natCast]
  /-
    case intro
    R : Type u_1
    inst✝³ : AddCommGroupWithOne R
    inst✝² : PartialOrder R
    inst✝¹ : AddLeftMono R
    inst✝ : ZeroLEOneClass R
    m n : Int
    k : Nat
    hk : Eq (↑k) (HSub.hSub n m)
    ⊢ LE.le 0 ↑k
  -/
  exact k.cast_nonneg'
  /-
    🎉 no goals
  -/


@[gcongr] protected lemma GCongr.intCast_mono {m n : ℤ} (hmn : m ≤ n) : (m : R) ≤ n := cast_mono hmn


@[simp] lemma cast_nonneg : ∀ {n : ℤ}, (0 : R) ≤ n ↔ 0 ≤ n
                  /-
                    R : Type u_1
                    inst✝⁴ : AddCommGroupWithOne R
                    inst✝³ : PartialOrder R
                    inst✝² : AddLeftMono R
                    inst✝¹ : ZeroLEOneClass R
                    inst✝ : NeZero 1
                    n : Nat
                    ⊢ Iff (LE.le 0 ↑↑n) (LE.le 0 ↑n)
                  -/
  | (n : ℕ) => by simp
                  /-
                    🎉 no goals
                  -/
  | -[n+1] => by
    /-
      R : Type u_1
      inst✝⁴ : AddCommGroupWithOne R
      inst✝³ : PartialOrder R
      inst✝² : AddLeftMono R
      inst✝¹ : ZeroLEOneClass R
      inst✝ : NeZero 1
      n : Nat
      ⊢ Iff (LE.le 0 ↑(Int.negSucc n)) (LE.le 0 (Int.negSucc n))
    -/
    have : -(n : R) < 1 := lt_of_le_of_lt (by simp) zero_lt_one
    /-
      R : Type u_1
      inst✝⁴ : AddCommGroupWithOne R
      inst✝³ : PartialOrder R
      inst✝² : AddLeftMono R
      inst✝¹ : ZeroLEOneClass R
      inst✝ : NeZero 1
      n : Nat
      this : LT.lt (Neg.neg ↑n) 1
      ⊢ Iff (LE.le 0 ↑(Int.negSucc n)) (LE.le 0 (Int.negSucc n))
    -/
    simpa [(negSucc_lt_zero n).not_le, ← sub_eq_add_neg, le_neg] using this.not_le
    /-
      🎉 no goals
    -/


@[simp, norm_cast] lemma cast_le : (m : R) ≤ n ↔ m ≤ n := by
  /-
    R : Type u_1
    inst✝⁴ : AddCommGroupWithOne R
    inst✝³ : PartialOrder R
    inst✝² : AddLeftMono R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : NeZero 1
    m n : Int
    ⊢ Iff (LE.le ↑m ↑n) (LE.le m n)
  -/
  rw [← sub_nonneg, ← cast_sub, cast_nonneg, sub_nonneg]
  /-
    🎉 no goals
  -/


lemma cast_strictMono : StrictMono (fun x : ℤ => (x : R)) :=
  strictMono_of_le_iff_le fun _ _ => cast_le.symm


@[simp, norm_cast] lemma cast_lt : (m : R) < n ↔ m < n := cast_strictMono.lt_iff_lt


@[gcongr] protected alias ⟨_, GCongr.intCast_strictMono⟩ := Int.cast_lt


                                                      /-
                                                        R : Type u_1
                                                        inst✝⁴ : AddCommGroupWithOne R
                                                        inst✝³ : PartialOrder R
                                                        inst✝² : AddLeftMono R
                                                        inst✝¹ : ZeroLEOneClass R
                                                        inst✝ : NeZero 1
                                                        n : Int
                                                        ⊢ Iff (LE.le (↑n) 0) (LE.le n 0)
                                                      -/
@[simp] lemma cast_nonpos : (n : R) ≤ 0 ↔ n ≤ 0 := by rw [← cast_zero, cast_le]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                   /-
                                                     R : Type u_1
                                                     inst✝⁴ : AddCommGroupWithOne R
                                                     inst✝³ : PartialOrder R
                                                     inst✝² : AddLeftMono R
                                                     inst✝¹ : ZeroLEOneClass R
                                                     inst✝ : NeZero 1
                                                     n : Int
                                                     ⊢ Iff (LT.lt 0 ↑n) (LT.lt 0 n)
                                                   -/
@[simp] lemma cast_pos : (0 : R) < n ↔ 0 < n := by rw [← cast_zero, cast_lt]
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                       /-
                                                         R : Type u_1
                                                         inst✝⁴ : AddCommGroupWithOne R
                                                         inst✝³ : PartialOrder R
                                                         inst✝² : AddLeftMono R
                                                         inst✝¹ : ZeroLEOneClass R
                                                         inst✝ : NeZero 1
                                                         n : Int
                                                         ⊢ Iff (LT.lt (↑n) 0) (LT.lt n 0)
                                                       -/
@[simp] lemma cast_lt_zero : (n : R) < 0 ↔ n < 0 := by rw [← cast_zero, cast_lt]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp, norm_cast]
lemma cast_min : ↑(min a b) = (min a b : R) := Monotone.map_min cast_mono


@[simp, norm_cast]
lemma cast_max : (↑(max a b) : R) = max (a : R) (b : R) := Monotone.map_max cast_mono


@[simp, norm_cast]
                                              /-
                                                R : Type u_1
                                                inst✝ : LinearOrderedRing R
                                                a : Int
                                                ⊢ Eq (↑(abs a)) (abs ↑a)
                                              -/
lemma cast_abs : (↑|a| : R) = |(a : R)| := by simp [abs_eq_max_neg]
                                              /-
                                                🎉 no goals
                                              -/


lemma cast_one_le_of_pos (h : 0 < a) : (1 : R) ≤ a := mod_cast Int.add_one_le_of_lt h


lemma cast_le_neg_one_of_neg (h : a < 0) : (a : R) ≤ -1 := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedRing R
    a : Int
    h : LT.lt a 0
    ⊢ LE.le (↑a) (-1)
  -/
  rw [← Int.cast_one, ← Int.cast_neg, cast_le]
  /-
    R : Type u_1
    inst✝ : LinearOrderedRing R
    a : Int
    h : LT.lt a 0
    ⊢ LE.le a (-1)
  -/
  exact Int.le_sub_one_of_lt h
  /-
    🎉 no goals
  -/


variable (R) in
lemma cast_le_neg_one_or_one_le_cast_of_ne_zero (hn : n ≠ 0) : (n : R) ≤ -1 ∨ 1 ≤ (n : R) :=
  hn.lt_or_lt.imp cast_le_neg_one_of_neg cast_one_le_of_pos


lemma nneg_mul_add_sq_of_abs_le_one (n : ℤ) (hx : |x| ≤ 1) : (0 : R) ≤ n * x + n * n := by
  have hnx : 0 < n → 0 ≤ x + n := fun hn => by
    have := _root_.add_le_add (neg_le_of_abs_le hx) (cast_one_le_of_pos hn)
    rwa [neg_add_cancel] at this
  have hnx' : n < 0 → x + n ≤ 0 := fun hn => by
    have := _root_.add_le_add (le_of_abs_le hx) (cast_le_neg_one_of_neg hn)
    rwa [add_neg_cancel] at this
  /-
    R : Type u_1
    inst✝ : LinearOrderedRing R
    x : R
    n : Int
    hx : LE.le (abs x) 1
    hnx : LT.lt 0 n → LE.le 0 (HAdd.hAdd x ↑n)
    hnx' : LT.lt n 0 → LE.le (HAdd.hAdd x ↑n) 0
    ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) x) (HMul.hMul ↑n ↑n))
  -/
  rw [← mul_add, mul_nonneg_iff]
  /-
    R : Type u_1
    inst✝ : LinearOrderedRing R
    x : R
    n : Int
    hx : LE.le (abs x) 1
    hnx : LT.lt 0 n → LE.le 0 (HAdd.hAdd x ↑n)
    hnx' : LT.lt n 0 → LE.le (HAdd.hAdd x ↑n) 0
    ⊢ Or (And (LE.le 0 ↑n) (LE.le 0 (HAdd.hAdd x ↑n))) (And (LE.le (↑n) 0) (LE.le  …
  -/
  rcases lt_trichotomy n 0 with (h | rfl | h)
    /-
      case inl
      R : Type u_1
      inst✝ : LinearOrderedRing R
      x : R
      n : Int
      hx : LE.le (abs x) 1
      hnx : LT.lt 0 n → LE.le 0 (HAdd.hAdd x ↑n)
      hnx' : LT.lt n 0 → LE.le (HAdd.hAdd x ↑n) 0
      h : LT.lt n 0
      ⊢ Or (And (LE.le 0 ↑n) (LE.le 0 (HAdd.hAdd x ↑n))) (And (LE.le (↑n) 0) (LE.le  …
    -/
  · exact Or.inr ⟨mod_cast h.le, hnx' h⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝ : LinearOrderedRing R
      x : R
      hx : LE.le (abs x) 1
      hnx : LT.lt 0 0 → LE.le 0 (HAdd.hAdd x ↑0)
      hnx' : LT.lt 0 0 → LE.le (HAdd.hAdd x ↑0) 0
      ⊢ Or (And (LE.le 0 ↑0) (LE.le 0 (HAdd.hAdd x ↑0))) (And (LE.le (↑0) 0) (LE.le  …
    -/
  · simp [le_total 0 x]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝ : LinearOrderedRing R
      x : R
      n : Int
      hx : LE.le (abs x) 1
      hnx : LT.lt 0 n → LE.le 0 (HAdd.hAdd x ↑n)
      hnx' : LT.lt n 0 → LE.le (HAdd.hAdd x ↑n) 0
      h : LT.lt 0 n
      ⊢ Or (And (LE.le 0 ↑n) (LE.le 0 (HAdd.hAdd x ↑n))) (And (LE.le (↑n) 0) (LE.le  …
    -/
  · exact Or.inl ⟨mod_cast h.le, hnx h⟩
    /-
      🎉 no goals
    -/


lemma cast_natAbs : (n.natAbs : R) = |n| := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedRing R
    n : Int
    ⊢ Eq ↑n.natAbs ↑(abs n)
  -/
  cases n
    /-
      case ofNat
      R : Type u_1
      inst✝ : LinearOrderedRing R
      a✝ : Nat
      ⊢ Eq ↑(Int.ofNat a✝).natAbs ↑(abs (Int.ofNat a✝))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      R : Type u_1
      inst✝ : LinearOrderedRing R
      a✝ : Nat
      ⊢ Eq ↑(Int.negSucc a✝).natAbs ↑(abs (Int.negSucc a✝))
    -/
  · rw [abs_eq_natAbs, natAbs_negSucc, cast_succ, cast_natCast, cast_succ]
    /-
      🎉 no goals
    -/


instance instIntCast             [IntCast R]             : IntCast Rᵒᵈ             := ‹_›

instance instAddGroupWithOne     [AddGroupWithOne R]     : AddGroupWithOne Rᵒᵈ     := ‹_›

instance instAddCommGroupWithOne [AddCommGroupWithOne R] : AddCommGroupWithOne Rᵒᵈ := ‹_›


@[simp] lemma toDual_intCast [IntCast R] (n : ℤ) : toDual (n : R) = n := rfl


@[simp] lemma ofDual_intCast [IntCast R] (n : ℤ) : (ofDual n : R) = n := rfl


instance instIntCast             [IntCast R]             : IntCast (Lex R)             := ‹_›

instance instAddGroupWithOne     [AddGroupWithOne R]     : AddGroupWithOne (Lex R)     := ‹_›

instance instAddCommGroupWithOne [AddCommGroupWithOne R] : AddCommGroupWithOne (Lex R) := ‹_›


@[simp] lemma toLex_intCast [IntCast R] (n : ℤ) : toLex (n : R) = n := rfl


@[simp] lemma ofLex_intCast [IntCast R] (n : ℤ) : (ofLex n : R) = n := rfl

