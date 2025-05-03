instance : FloorSemiring ℚ≥0 where
  floor q := ⌊q.val⌋₊
  ceil q := ⌈q.val⌉₊
                       /-
                         a✝ : NNRat
                         h : LT.lt a✝ 0
                         ⊢ Eq ((fun q => Nat.floor ↑q) a✝) 0
                       -/
  floor_of_neg h := by simpa using h.trans zero_lt_one
                       /-
                         🎉 no goals
                       -/
                         /-
                           a : NNRat
                           n : Nat
                           h : LE.le 0 a
                           ⊢ Iff (LE.le n ((fun q => Nat.floor ↑q) a)) (LE.le (↑n) a)
                         -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  gc_floor {a n} h := by rw [← NNRat.coe_le_coe, Nat.le_floor_iff] <;> norm_cast
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                      /-
                        a : NNRat
                        b : Nat
                        ⊢ Iff (LE.le ((fun q => Nat.ceil ↑q) a) b) (LE.le a ↑b)
                      -/
  gc_ceil {a b} := by rw [← NNRat.coe_le_coe, Nat.ceil_le]; norm_cast
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp, norm_cast]
theorem floor_coe (q : ℚ≥0) : ⌊(q : ℚ)⌋₊ = ⌊q⌋₊ := rfl


@[simp, norm_cast]
theorem ceil_coe (q : ℚ≥0) : ⌈(q : ℚ)⌉₊ = ⌈q⌉₊ := rfl


@[simp, norm_cast]
theorem coe_floor (q : ℚ≥0) : ↑⌊q⌋₊ = ⌊(q : ℚ)⌋ := Int.natCast_floor_eq_floor q.coe_nonneg


@[simp, norm_cast]
theorem coe_ceil (q : ℚ≥0) : ↑⌈q⌉₊ = ⌈(q : ℚ)⌉ := Int.natCast_ceil_eq_ceil q.coe_nonneg


protected theorem floor_def (q : ℚ≥0) : ⌊q⌋₊ = q.num / q.den := by
  /-
    q : NNRat
    ⊢ Eq (Nat.floor q) (HDiv.hDiv q.num q.den)
  -/
  rw [← Int.natCast_inj, NNRat.coe_floor, Rat.floor_def, Int.ofNat_ediv, den_coe, num_coe]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem floor_cast (x : ℚ≥0) : ⌊(x : K)⌋₊ = ⌊x⌋₊ :=
  (Nat.floor_eq_iff x.cast_nonneg).2 (mod_cast (Nat.floor_eq_iff x.cast_nonneg).1 (Eq.refl ⌊x⌋₊))


@[simp, norm_cast]
theorem ceil_cast (x : ℚ≥0) : ⌈(x : K)⌉₊ = ⌈x⌉₊ := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedSemifield K
    inst✝ : FloorSemiring K
    x : NNRat
    ⊢ Eq (Nat.ceil ↑x) (Nat.ceil x)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      K : Type u_1
      inst✝¹ : LinearOrderedSemifield K
      inst✝ : FloorSemiring K
      ⊢ Eq (Nat.ceil ↑0) (Nat.ceil 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝¹ : LinearOrderedSemifield K
      inst✝ : FloorSemiring K
      x : NNRat
      hx : Ne x 0
      ⊢ Eq (Nat.ceil ↑x) (Nat.ceil x)
    -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  · refine (Nat.ceil_eq_iff ?_).2 (mod_cast (Nat.ceil_eq_iff ?_).1 (Eq.refl ⌈x⌉₊)) <;> simpa
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp, norm_cast]
theorem intFloor_cast (x : ℚ≥0) : ⌊(x : K)⌋ = ⌊(x : ℚ)⌋ := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    x : NNRat
    ⊢ Eq (Int.floor ↑x) (Int.floor ↑x)
  -/
  rw [Int.floor_eq_iff (α := K), ← coe_floor]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    x : NNRat
    ⊢ And (LE.le ↑↑(Nat.floor x) ↑x) (LT.lt (↑x) (HAdd.hAdd (↑↑(Nat.floor x)) 1))
  -/
  norm_cast
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    x : NNRat
    ⊢ And (LE.le ↑(Nat.floor x) ↑x) (LT.lt ↑x ↑(HAdd.hAdd (Nat.floor x) 1))
  -/
  norm_cast
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    x : NNRat
    ⊢ And (LE.le (↑(Nat.floor x)) x) (LT.lt x ↑(HAdd.hAdd (Nat.floor x) 1))
  -/
  rw [Nat.cast_add_one, ← Nat.floor_eq_iff (zero_le _)]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem intCeil_cast (x : ℚ≥0) : ⌈(x : K)⌉ = ⌈(x : ℚ)⌉ := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    x : NNRat
    ⊢ Eq (Int.ceil ↑x) (Int.ceil ↑x)
  -/
  rw [Int.ceil_eq_iff, ← coe_ceil, sub_lt_iff_lt_add]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    x : NNRat
    ⊢ And (LT.lt (↑↑(Nat.ceil x)) (HAdd.hAdd (↑x) 1)) (LE.le ↑x ↑↑(Nat.ceil x))
  -/
  constructor
    /-
      case left
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      x : NNRat
      ⊢ LT.lt (↑↑(Nat.ceil x)) (HAdd.hAdd (↑x) 1)
    -/
  · have := NNRat.cast_strictMono (K := K) <| Nat.ceil_lt_add_one <| zero_le x
    /-
      case left
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      x : NNRat
      this : LT.lt ↑↑(Nat.ceil x) ↑(HAdd.hAdd x 1)
      ⊢ LT.lt (↑↑(Nat.ceil x)) (HAdd.hAdd (↑x) 1)
    -/
    rw [NNRat.cast_add, NNRat.cast_one] at this
    /-
      case left
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      x : NNRat
      this : LT.lt (↑↑(Nat.ceil x)) (HAdd.hAdd (↑x) 1)
      ⊢ LT.lt (↑↑(Nat.ceil x)) (HAdd.hAdd (↑x) 1)
    -/
    refine Eq.trans_lt ?_ this
    /-
      case left
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      x : NNRat
      this : LT.lt (↑↑(Nat.ceil x)) (HAdd.hAdd (↑x) 1)
      ⊢ Eq ↑↑(Nat.ceil x) ↑↑(Nat.ceil x)
    -/
    norm_cast
    /-
      🎉 no goals
    -/
    /-
      case right
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      x : NNRat
      ⊢ LE.le ↑x ↑↑(Nat.ceil x)
    -/
  · rw [Int.cast_natCast, NNRat.cast_le_natCast]
    /-
      case right
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      x : NNRat
      ⊢ LE.le x ↑(Nat.ceil x)
    -/
    exact Nat.le_ceil _
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem floor_natCast_div_natCast (n d : ℕ) : ⌊(↑n / ↑d : ℚ≥0)⌋₊ = n / d :=
  Rat.natFloor_natCast_div_natCast n d


