/-- Given a nonzero `e : ℝ≥0`, this is the map `ℤₘ₀ → ℝ≥0` sending `0 ↦ 0` and
  `x ↦ e^(WithZero.unzero hx).toAdd` when `x ≠ 0` as a `MonoidWithZeroHom`. -/
def toNNReal {e : ℝ≥0} (he : e ≠ 0) : ℤₘ₀ →*₀ ℝ≥0 where
  toFun := fun x ↦ if hx : x = 0 then 0 else e ^ (WithZero.unzero hx).toAdd
  map_zero' := rfl
  map_one' := by
    /-
      e : NNReal
      he : Ne e 0
      ⊢ Eq ({ toFun := fun x => dite (Eq x 0) (fun hx => 0) fun hx => HPow.hPow e (M …
    -/
    simp only [dif_neg one_ne_zero]
    /-
      e : NNReal
      he : Ne e 0
      ⊢ Eq (HPow.hPow e (Multiplicative.toAdd (WithZero.unzero ⋯))) 1
    -/
    erw [toAdd_one, zpow_zero]
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    /-
      e : NNReal
      he : Ne e 0
      x y : WithZero (Multiplicative Int)
      ⊢ Eq ({ toFun := fun x => dite (Eq x 0) (fun hx => 0) fun hx => HPow.hPow e (M …
    -/
    simp only
    /-
      e : NNReal
      he : Ne e 0
      x y : WithZero (Multiplicative Int)
      ⊢ Eq (dite (Eq (HMul.hMul x y) 0) (fun hx => 0) fun hx => HPow.hPow e (Multipl …
    -/
    by_cases hxy : x * y = 0
      /-
        case pos
        e : NNReal
        he : Ne e 0
        x y : WithZero (Multiplicative Int)
        hxy : Eq (HMul.hMul x y) 0
        ⊢ Eq (dite (Eq (HMul.hMul x y) 0) (fun hx => 0) fun hx => HPow.hPow e (Multipl …
      -/
    · cases' zero_eq_mul.mp (Eq.symm hxy) with hx hy
      --either x = 0 or y = 0
        /-
          case pos.inl
          e : NNReal
          he : Ne e 0
          x y : WithZero (Multiplicative Int)
          hxy : Eq (HMul.hMul x y) 0
          hx : Eq x 0
          ⊢ Eq (dite (Eq (HMul.hMul x y) 0) (fun hx => 0) fun hx => HPow.hPow e (Multipl …
        -/
      · rw [dif_pos hxy, dif_pos hx, MulZeroClass.zero_mul]
        /-
          🎉 no goals
        -/
        /-
          case pos.inr
          e : NNReal
          he : Ne e 0
          x y : WithZero (Multiplicative Int)
          hxy : Eq (HMul.hMul x y) 0
          hy : Eq y 0
          ⊢ Eq (dite (Eq (HMul.hMul x y) 0) (fun hx => 0) fun hx => HPow.hPow e (Multipl …
        -/
      · rw [dif_pos hxy, dif_pos hy, MulZeroClass.mul_zero]
        /-
          🎉 no goals
        -/
      /-
        case neg
        e : NNReal
        he : Ne e 0
        x y : WithZero (Multiplicative Int)
        hxy : Not (Eq (HMul.hMul x y) 0)
        ⊢ Eq (dite (Eq (HMul.hMul x y) 0) (fun hx => 0) fun hx => HPow.hPow e (Multipl …
      -/
    · cases' mul_ne_zero_iff.mp hxy with hx hy
      --  x Equiv≠ 0 and y ≠ 0
      /-
        case neg.intro
        e : NNReal
        he : Ne e 0
        x y : WithZero (Multiplicative Int)
        hxy : Not (Eq (HMul.hMul x y) 0)
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Eq (dite (Eq (HMul.hMul x y) 0) (fun hx => 0) fun hx => HPow.hPow e (Multipl …
      -/
      rw [dif_neg hxy, dif_neg hx, dif_neg hy, ← zpow_add' (Or.inl he), ← toAdd_mul]
      /-
        case neg.intro
        e : NNReal
        he : Ne e 0
        x y : WithZero (Multiplicative Int)
        hxy : Not (Eq (HMul.hMul x y) 0)
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Eq (HPow.hPow e (Multiplicative.toAdd (WithZero.unzero hxy))) (HPow.hPow e ( …
      -/
      congr
      /-
        case neg.intro.e_a.h.e_6.h
        e : NNReal
        he : Ne e 0
        x y : WithZero (Multiplicative Int)
        hxy : Not (Eq (HMul.hMul x y) 0)
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Eq (WithZero.unzero hxy) (HMul.hMul (WithZero.unzero hx) (WithZero.unzero hy))
      -/
      rw [← WithZero.coe_inj, WithZero.coe_mul, coe_unzero hx, coe_unzero hy, coe_unzero hxy]
      /-
        🎉 no goals
      -/


theorem toNNReal_pos_apply {e : ℝ≥0} (he : e ≠ 0) {x : ℤₘ₀} (hx : x = 0) :
    toNNReal he x = 0 := by
  /-
    e : NNReal
    he : Ne e 0
    x : WithZero (Multiplicative Int)
    hx : Eq x 0
    ⊢ Eq ((WithZeroMulInt.toNNReal he) x) 0
  -/
  simp only [toNNReal, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    e : NNReal
    he : Ne e 0
    x : WithZero (Multiplicative Int)
    hx : Eq x 0
    ⊢ Eq (dite (Eq x 0) (fun hx => 0) fun hx => HPow.hPow e (Multiplicative.toAdd  …
  -/
  split_ifs; rfl
             /-
               🎉 no goals
             -/


theorem toNNReal_neg_apply {e : ℝ≥0} (he : e ≠ 0) {x : ℤₘ₀} (hx : x ≠ 0) :
    toNNReal he x = e ^ (WithZero.unzero hx).toAdd := by
  /-
    e : NNReal
    he : Ne e 0
    x : WithZero (Multiplicative Int)
    hx : Ne x 0
    ⊢ Eq ((WithZeroMulInt.toNNReal he) x) (HPow.hPow e (Multiplicative.toAdd (With …
  -/
  simp only [toNNReal, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    e : NNReal
    he : Ne e 0
    x : WithZero (Multiplicative Int)
    hx : Ne x 0
    ⊢ Eq (dite (Eq x 0) (fun hx => 0) fun hx => HPow.hPow e (Multiplicative.toAdd  …
  -/
  split_ifs
    /-
      case pos
      e : NNReal
      he : Ne e 0
      x : WithZero (Multiplicative Int)
      hx : Ne x 0
      h✝ : Eq x 0
      ⊢ Eq 0 (HPow.hPow e (Multiplicative.toAdd (WithZero.unzero hx)))
    -/
  · tauto
    /-
      🎉 no goals
    -/
    /-
      case neg
      e : NNReal
      he : Ne e 0
      x : WithZero (Multiplicative Int)
      hx : Ne x 0
      h✝ : Not (Eq x 0)
      ⊢ Eq (HPow.hPow e (Multiplicative.toAdd (WithZero.unzero h✝))) (HPow.hPow e (M …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- `toNNReal` sends nonzero elements to nonzero elements. -/
theorem toNNReal_ne_zero {e : ℝ≥0} {m : ℤₘ₀} (he : e ≠ 0) (hm : m ≠ 0) : toNNReal he m ≠ 0 := by
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : Ne e 0
    hm : Ne m 0
    ⊢ Ne ((WithZeroMulInt.toNNReal he) m) 0
  -/
  simp only [ne_eq, map_eq_zero, hm, not_false_eq_true]
  /-
    🎉 no goals
  -/


/-- `toNNReal` sends nonzero elements to positive elements. -/
theorem toNNReal_pos {e : ℝ≥0} {m : ℤₘ₀} (he : e ≠ 0) (hm : m ≠ 0) : 0 < toNNReal he m :=
  lt_of_le_of_ne zero_le' (toNNReal_ne_zero he hm).symm


/-- The map `toNNReal` is strictly monotone whenever `1 < e`. -/
theorem toNNReal_strictMono {e : ℝ≥0} (he : 1 < e) :
    StrictMono (toNNReal (ne_zero_of_lt he)) := by
  /-
    e : NNReal
    he : LT.lt 1 e
    ⊢ StrictMono ⇑(WithZeroMulInt.toNNReal ⋯)
  -/
  intro x y hxy
  /-
    e : NNReal
    he : LT.lt 1 e
    x y : WithZero (Multiplicative Int)
    hxy : LT.lt x y
    ⊢ LT.lt ((WithZeroMulInt.toNNReal ⋯) x) ((WithZeroMulInt.toNNReal ⋯) y)
  -/
  simp only [toNNReal, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    e : NNReal
    he : LT.lt 1 e
    x y : WithZero (Multiplicative Int)
    hxy : LT.lt x y
    ⊢ LT.lt (dite (Eq x 0) (fun hx => 0) fun hx => HPow.hPow e (Multiplicative.toA …
  -/
  split_ifs with hx hy hy
    /-
      case pos
      e : NNReal
      he : LT.lt 1 e
      x y : WithZero (Multiplicative Int)
      hxy : LT.lt x y
      hx : Eq x 0
      hy : Eq y 0
      ⊢ LT.lt 0 0
    -/
  · simp only [hy, not_lt_zero'] at hxy
    /-
      🎉 no goals
    -/
    /-
      case neg
      e : NNReal
      he : LT.lt 1 e
      x y : WithZero (Multiplicative Int)
      hxy : LT.lt x y
      hx : Eq x 0
      hy : Not (Eq y 0)
      ⊢ LT.lt 0 (HPow.hPow e (Multiplicative.toAdd (WithZero.unzero hy)))
    -/
  · exact zpow_pos he.bot_lt _
    /-
      🎉 no goals
    -/
    /-
      case pos
      e : NNReal
      he : LT.lt 1 e
      x y : WithZero (Multiplicative Int)
      hxy : LT.lt x y
      hx : Not (Eq x 0)
      hy : Eq y 0
      ⊢ LT.lt (HPow.hPow e (Multiplicative.toAdd (WithZero.unzero hx))) 0
    -/
  · simp only [hy, not_lt_zero'] at hxy
    /-
      🎉 no goals
    -/
  · rw [zpow_lt_zpow_iff_right₀ he, Multiplicative.toAdd_lt, ← coe_lt_coe, coe_unzero hx,
      WithZero.coe_unzero hy]
    /-
      case neg
      e : NNReal
      he : LT.lt 1 e
      x y : WithZero (Multiplicative Int)
      hxy : LT.lt x y
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      ⊢ LT.lt x y
    -/
    exact hxy
    /-
      🎉 no goals
    -/


theorem toNNReal_eq_one_iff {e : ℝ≥0} (m : ℤₘ₀) (he0 : e ≠ 0) (he1 : e ≠ 1) :
    toNNReal he0 m = 1 ↔ m = 1 := by
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he0 : Ne e 0
    he1 : Ne e 1
    ⊢ Iff (Eq ((WithZeroMulInt.toNNReal he0) m) 1) (Eq m 1)
  -/
  by_cases hm : m = 0
    /-
      case pos
      e : NNReal
      m : WithZero (Multiplicative Int)
      he0 : Ne e 0
      he1 : Ne e 1
      hm : Eq m 0
      ⊢ Iff (Eq ((WithZeroMulInt.toNNReal he0) m) 1) (Eq m 1)
    -/
  · simp only [hm, map_zero, zero_ne_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      e : NNReal
      m : WithZero (Multiplicative Int)
      he0 : Ne e 0
      he1 : Ne e 1
      hm : Not (Eq m 0)
      ⊢ Iff (Eq ((WithZeroMulInt.toNNReal he0) m) 1) (Eq m 1)
    -/
  · refine ⟨fun h1 ↦ ?_, fun h1 ↦ h1 ▸ map_one _⟩
    /-
      case neg
      e : NNReal
      m : WithZero (Multiplicative Int)
      he0 : Ne e 0
      he1 : Ne e 1
      hm : Not (Eq m 0)
      h1 : Eq ((WithZeroMulInt.toNNReal he0) m) 1
      ⊢ Eq m 1
    -/
    rw [toNNReal_neg_apply he0 hm, zpow_eq_one_iff_right₀ (zero_le e) he1, toAdd_eq_zero] at h1
    /-
      case neg
      e : NNReal
      m : WithZero (Multiplicative Int)
      he0 : Ne e 0
      he1 : Ne e 1
      hm : Not (Eq m 0)
      h1 : Eq (WithZero.unzero hm) 1
      ⊢ Eq m 1
    -/
    rw [← WithZero.coe_unzero hm, h1, coe_one]
    /-
      🎉 no goals
    -/


theorem toNNReal_lt_one_iff {e : ℝ≥0} {m : ℤₘ₀} (he : 1 < e) :
    toNNReal (ne_zero_of_lt he) m < 1 ↔ m < 1 := by
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : LT.lt 1 e
    ⊢ Iff (LT.lt ((WithZeroMulInt.toNNReal ⋯) m) 1) (LT.lt m 1)
  -/
  have : 1 = (toNNReal (ne_zero_of_lt he)) 1 := rfl
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : LT.lt 1 e
    this : Eq 1 ((WithZeroMulInt.toNNReal ⋯) 1)
    ⊢ Iff (LT.lt ((WithZeroMulInt.toNNReal ⋯) m) 1) (LT.lt m 1)
  -/
  simp_rw [this]
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : LT.lt 1 e
    this : Eq 1 ((WithZeroMulInt.toNNReal ⋯) 1)
    ⊢ Iff (LT.lt ((WithZeroMulInt.toNNReal ⋯) m) ((WithZeroMulInt.toNNReal ⋯) 1))  …
  -/
  exact StrictMono.lt_iff_lt (toNNReal_strictMono he)
  /-
    🎉 no goals
  -/


theorem toNNReal_le_one_iff {e : ℝ≥0} {m : ℤₘ₀} (he : 1 < e) :
    toNNReal (ne_zero_of_lt he) m ≤ 1 ↔ m ≤ 1 := by
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : LT.lt 1 e
    ⊢ Iff (LE.le ((WithZeroMulInt.toNNReal ⋯) m) 1) (LE.le m 1)
  -/
  have : 1 = (toNNReal (ne_zero_of_lt he)) 1 := rfl
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : LT.lt 1 e
    this : Eq 1 ((WithZeroMulInt.toNNReal ⋯) 1)
    ⊢ Iff (LE.le ((WithZeroMulInt.toNNReal ⋯) m) 1) (LE.le m 1)
  -/
  simp_rw [this]
  /-
    e : NNReal
    m : WithZero (Multiplicative Int)
    he : LT.lt 1 e
    this : Eq 1 ((WithZeroMulInt.toNNReal ⋯) 1)
    ⊢ Iff (LE.le ((WithZeroMulInt.toNNReal ⋯) m) ((WithZeroMulInt.toNNReal ⋯) 1))  …
  -/
  exact StrictMono.le_iff_le (toNNReal_strictMono he)
  /-
    🎉 no goals
  -/


