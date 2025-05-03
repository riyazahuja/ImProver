theorem exists_coeff_ne_zero_iff_ne_zero : (∃ n : ℕ, coeff R n φ ≠ 0) ↔ φ ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (Exists fun n => Ne ((PowerSeries.coeff R n) φ) 0) (Ne φ 0)
  -/
  refine not_iff_not.mp ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (Not (Exists fun n => Ne ((PowerSeries.coeff R n) φ) 0)) (Not (Ne φ 0))
  -/
  push_neg
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (∀ (n : Nat), Eq ((PowerSeries.coeff R n) φ) 0) (Eq φ 0)
  -/
  simp [(coeff R _).map_zero]
  /-
    🎉 no goals
  -/


/-- The order of a formal power series `φ` is the greatest `n : PartENat`
such that `X^n` divides `φ`. The order is `⊤` if and only if `φ = 0`. -/
def order (φ : R⟦X⟧) : ℕ∞ :=
  letI := Classical.decEq R
  letI := Classical.decEq R⟦X⟧
  if h : φ = 0 then ⊤ else Nat.find (exists_coeff_ne_zero_iff_ne_zero.mpr h)


/-- The order of the `0` power series is infinite. -/
@[simp]
theorem order_zero : order (0 : R⟦X⟧) = ⊤ :=
  dif_pos rfl


theorem order_finite_iff_ne_zero : (order φ < ⊤) ↔ φ ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (LT.lt φ.order Top.top) (Ne φ 0)
  -/
  simp only [order]
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (LT.lt (dite (Eq φ 0) (fun h => Top.top) fun h => ↑(Nat.find ⋯)) Top.top …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      ⊢ LT.lt (dite (Eq φ 0) (fun h => Top.top) fun h => ↑(Nat.find ⋯)) Top.top → Ne …
    -/
  · split_ifs with h <;> intro H
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        φ : PowerSeries R
        h : Eq φ 0
        H : LT.lt Top.top Top.top
        ⊢ Ne φ 0
      -/
    · simp at H
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        φ : PowerSeries R
        h : Not (Eq φ 0)
        H : LT.lt (↑(Nat.find ⋯)) Top.top
        ⊢ Ne φ 0
      -/
    · exact h
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      ⊢ Ne φ 0 → LT.lt (dite (Eq φ 0) (fun h => Top.top) fun h => ↑(Nat.find ⋯)) Top …
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : Ne φ 0
      ⊢ LT.lt (dite (Eq φ 0) (fun h => Top.top) fun h => ↑(Nat.find ⋯)) Top.top
    -/
    simp [h]
    /-
      🎉 no goals
    -/


/-- If the order of a formal power series is finite,
then the coefficient indexed by the order is nonzero. -/
theorem coeff_order (h : order φ < ⊤) : coeff R (φ.order.lift h) φ ≠ 0 := by
  classical
  simp only [order, order_finite_iff_ne_zero.mp h, not_false_iff, dif_neg]
  generalize_proofs h
  exact Nat.find_spec h


/-- If the `n`th coefficient of a formal power series is nonzero,
then the order of the power series is less than or equal to `n`. -/
theorem order_le (n : ℕ) (h : coeff R n φ ≠ 0) : order φ ≤ n := by
  classical
  rw [order, dif_neg]
  · simpa using ⟨n, le_rfl, h⟩
  · exact exists_coeff_ne_zero_iff_ne_zero.mp ⟨n, h⟩


/-- The `n`th coefficient of a formal power series is `0` if `n` is strictly
smaller than the order of the power series. -/
theorem coeff_of_lt_order (n : ℕ) (h : ↑n < order φ) : coeff R n φ = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) φ.order
    ⊢ Eq ((PowerSeries.coeff R n) φ) 0
  -/
  contrapose! h
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    n : Nat
    h : Ne ((PowerSeries.coeff R n) φ) 0
    ⊢ LE.le φ.order ↑n
  -/
  exact order_le _ h
  /-
    🎉 no goals
  -/


/-- The `0` power series is the unique power series with infinite order. -/
@[simp]
theorem order_eq_top {φ : R⟦X⟧} : φ.order = ⊤ ↔ φ = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (Eq φ.order Top.top) (Eq φ 0)
  -/
  simpa using order_finite_iff_ne_zero.not_left
  /-
    🎉 no goals
  -/


/-- The order of a formal power series is at least `n` if
the `i`th coefficient is `0` for all `i < n`. -/
theorem nat_le_order (φ : R⟦X⟧) (n : ℕ) (h : ∀ i < n, coeff R i φ = 0) : ↑n ≤ order φ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    n : Nat
    h : ∀ (i : Nat), LT.lt i n → Eq ((PowerSeries.coeff R i) φ) 0
    ⊢ LE.le (↑n) φ.order
  -/
  by_contra H; rw [not_le] at H
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    n : Nat
    h : ∀ (i : Nat), LT.lt i n → Eq ((PowerSeries.coeff R i) φ) 0
    H : LT.lt φ.order ↑n
    ⊢ False
  -/
  have lt_top : order φ < ⊤ := lt_top_of_lt H
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    n : Nat
    h : ∀ (i : Nat), LT.lt i n → Eq ((PowerSeries.coeff R i) φ) 0
    H : LT.lt φ.order ↑n
    lt_top : LT.lt φ.order Top.top
    ⊢ False
  -/
  replace H : (order φ).lift lt_top < n := by simpa
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    n : Nat
    h : ∀ (i : Nat), LT.lt i n → Eq ((PowerSeries.coeff R i) φ) 0
    lt_top : LT.lt φ.order Top.top
    H : LT.lt (φ.order.lift lt_top) n
    ⊢ False
  -/
  exact coeff_order lt_top (h _ H)
  /-
    🎉 no goals
  -/


/-- The order of a formal power series is at least `n` if
the `i`th coefficient is `0` for all `i < n`. -/
theorem le_order (φ : R⟦X⟧) (n : ℕ∞) (h : ∀ i : ℕ, ↑i < n → coeff R i φ = 0) :
    n ≤ order φ := by
  cases n with
  | top => simpa using ext (by simpa using h)
  | coe n =>
    convert nat_le_order φ n _
    simpa using h


/-- The order of a formal power series is exactly `n` if the `n`th coefficient is nonzero,
and the `i`th coefficient is `0` for all `i < n`. -/
theorem order_eq_nat {φ : R⟦X⟧} {n : ℕ} :
    order φ = n ↔ coeff R n φ ≠ 0 ∧ ∀ i, i < n → coeff R i φ = 0 := by
  classical
  rcases eq_or_ne φ 0 with (rfl | hφ)
  · simp
  simp [order, dif_neg hφ, Nat.find_eq_iff]


/-- The order of a formal power series is exactly `n` if the `n`th coefficient is nonzero,
and the `i`th coefficient is `0` for all `i < n`. -/
theorem order_eq {φ : R⟦X⟧} {n : ℕ∞} :
    order φ = n ↔ (∀ i : ℕ, ↑i = n → coeff R i φ ≠ 0) ∧ ∀ i : ℕ, ↑i < n → coeff R i φ = 0 := by
  cases n with
  | top => simp [ext_iff]
  | coe n => simp [order_eq_nat]



/-- The order of the sum of two formal power series
 is at least the minimum of their orders. -/
theorem min_order_le_order_add (φ ψ : R⟦X⟧) : min (order φ) (order ψ) ≤ order (φ + ψ) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    ⊢ LE.le (Min.min φ.order ψ.order) (HAdd.hAdd φ ψ).order
  -/
  refine le_order _ _ ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    ⊢ ∀ (i : Nat), LT.lt (↑i) (Min.min φ.order ψ.order) → Eq ((PowerSeries.coeff R …
  -/
  simp +contextual [coeff_of_lt_order]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")] alias le_order_add := min_order_le_order_add


private theorem order_add_of_order_eq.aux (φ ψ : R⟦X⟧) (_h : order φ ≠ order ψ)
    (H : order φ < order ψ) : order (φ + ψ) ≤ order φ ⊓ order ψ := by
  suffices order (φ + ψ) = order φ by
    rw [le_inf_iff, this]
    exact ⟨le_rfl, le_of_lt H⟩
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    _h : Ne φ.order ψ.order
    H : LT.lt φ.order ψ.order
    ⊢ Eq (HAdd.hAdd φ ψ).order φ.order
  -/
  rw [order_eq]
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    _h : Ne φ.order ψ.order
    H : LT.lt φ.order ψ.order
    ⊢ And (∀ (i : Nat), Eq (↑i) φ.order → Ne ((PowerSeries.coeff R i) (HAdd.hAdd φ …
  -/
  constructor
    /-
      case left
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      _h : Ne φ.order ψ.order
      H : LT.lt φ.order ψ.order
      ⊢ ∀ (i : Nat), Eq (↑i) φ.order → Ne ((PowerSeries.coeff R i) (HAdd.hAdd φ ψ)) 0
    -/
  · intro i hi
    /-
      case left
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      _h : Ne φ.order ψ.order
      H : LT.lt φ.order ψ.order
      i : Nat
      hi : Eq (↑i) φ.order
      ⊢ Ne ((PowerSeries.coeff R i) (HAdd.hAdd φ ψ)) 0
    -/
    rw [← hi] at H
    /-
      case left
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      _h : Ne φ.order ψ.order
      i : Nat
      H : LT.lt (↑i) ψ.order
      hi : Eq (↑i) φ.order
      ⊢ Ne ((PowerSeries.coeff R i) (HAdd.hAdd φ ψ)) 0
    -/
    rw [(coeff _ _).map_add, coeff_of_lt_order i H, add_zero]
    /-
      case left
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      _h : Ne φ.order ψ.order
      i : Nat
      H : LT.lt (↑i) ψ.order
      hi : Eq (↑i) φ.order
      ⊢ Ne ((PowerSeries.coeff R i) φ) 0
    -/
    exact (order_eq_nat.1 hi.symm).1
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      _h : Ne φ.order ψ.order
      H : LT.lt φ.order ψ.order
      ⊢ ∀ (i : Nat), LT.lt (↑i) φ.order → Eq ((PowerSeries.coeff R i) (HAdd.hAdd φ ψ …
    -/
  · intro i hi
    rw [(coeff _ _).map_add, coeff_of_lt_order i hi, coeff_of_lt_order i (lt_trans hi H),
      zero_add]


/-- The order of the sum of two formal power series
 is the minimum of their orders if their orders differ. -/
theorem order_add_of_order_eq (φ ψ : R⟦X⟧) (h : order φ ≠ order ψ) :
    order (φ + ψ) = order φ ⊓ order ψ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    h : Ne φ.order ψ.order
    ⊢ Eq (HAdd.hAdd φ ψ).order (Min.min φ.order ψ.order)
  -/
  refine le_antisymm ?_ (min_order_le_order_add _ _)
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    h : Ne φ.order ψ.order
    ⊢ LE.le (HAdd.hAdd φ ψ).order (Min.min φ.order ψ.order)
  -/
  by_cases H₁ : order φ < order ψ
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      h : Ne φ.order ψ.order
      H₁ : LT.lt φ.order ψ.order
      ⊢ LE.le (HAdd.hAdd φ ψ).order (Min.min φ.order ψ.order)
    -/
  · apply order_add_of_order_eq.aux _ _ h H₁
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    h : Ne φ.order ψ.order
    H₁ : Not (LT.lt φ.order ψ.order)
    ⊢ LE.le (HAdd.hAdd φ ψ).order (Min.min φ.order ψ.order)
  -/
  by_cases H₂ : order ψ < order φ
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      h : Ne φ.order ψ.order
      H₁ : Not (LT.lt φ.order ψ.order)
      H₂ : LT.lt ψ.order φ.order
      ⊢ LE.le (HAdd.hAdd φ ψ).order (Min.min φ.order ψ.order)
    -/
  · simpa only [add_comm, inf_comm] using order_add_of_order_eq.aux _ _ h.symm H₂
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    h : Ne φ.order ψ.order
    H₁ : Not (LT.lt φ.order ψ.order)
    H₂ : Not (LT.lt ψ.order φ.order)
    ⊢ LE.le (HAdd.hAdd φ ψ).order (Min.min φ.order ψ.order)
  -/
  exfalso; exact h (le_antisymm (not_lt.1 H₂) (not_lt.1 H₁))
           /-
             🎉 no goals
           -/


/-- The order of the product of two formal power series
 is at least the sum of their orders. -/
theorem le_order_mul (φ ψ : R⟦X⟧) : order φ + order ψ ≤ order (φ * ψ) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    ⊢ LE.le (HAdd.hAdd φ.order ψ.order) (HMul.hMul φ ψ).order
  -/
  apply le_order
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    ⊢ ∀ (i : Nat), LT.lt (↑i) (HAdd.hAdd φ.order ψ.order) → Eq ((PowerSeries.coeff …
  -/
  intro n hn; rw [coeff_mul, Finset.sum_eq_zero]
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
  -/
  rintro ⟨i, j⟩ hij
  /-
    case h.mk
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
    ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
  -/
  by_cases hi : ↑i < order φ
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      n : Nat
      hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
      i j : Nat
      hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
      hi : LT.lt (↑i) φ.order
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
    -/
  · rw [coeff_of_lt_order i hi, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
    hi : Not (LT.lt (↑i) φ.order)
    ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
  -/
  by_cases hj : ↑j < order ψ
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      n : Nat
      hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
      i j : Nat
      hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
      hi : Not (LT.lt (↑i) φ.order)
      hj : LT.lt (↑j) ψ.order
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
    -/
  · rw [coeff_of_lt_order j hj, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
    hi : Not (LT.lt (↑i) φ.order)
    hj : Not (LT.lt (↑j) ψ.order)
    ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
  -/
  rw [not_lt] at hi hj; rw [mem_antidiagonal] at hij
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    i j : Nat
    hij : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) n
    hi : LE.le φ.order ↑i
    hj : LE.le ψ.order ↑j
    ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) φ) ((PowerSeri …
  -/
  exfalso
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    i j : Nat
    hij : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) n
    hi : LE.le φ.order ↑i
    hj : LE.le ψ.order ↑j
    ⊢ False
  -/
  apply ne_of_lt (lt_of_lt_of_le hn <| add_le_add hi hj)
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    hn : LT.lt (↑n) (HAdd.hAdd φ.order ψ.order)
    i j : Nat
    hij : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) n
    hi : LE.le φ.order ↑i
    hj : LE.le ψ.order ↑j
    ⊢ Eq (↑n) (HAdd.hAdd ↑i ↑j)
  -/
  rw [← Nat.cast_add, hij]
  /-
    🎉 no goals
  -/


alias order_mul_ge := le_order_mul


/-- The order of the monomial `a*X^n` is infinite if `a = 0` and `n` otherwise. -/
theorem order_monomial (n : ℕ) (a : R) [Decidable (a = 0)] :
    order (monomial R n a) = if a = 0 then (⊤ : ℕ∞) else n := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    n : Nat
    a : R
    inst✝ : Decidable (Eq a 0)
    ⊢ Eq ((PowerSeries.monomial R n) a).order (ite (Eq a 0) Top.top ↑n)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝¹ : Semiring R
      n : Nat
      a : R
      inst✝ : Decidable (Eq a 0)
      h : Eq a 0
      ⊢ Eq ((PowerSeries.monomial R n) a).order Top.top
    -/
  · rw [h, order_eq_top, LinearMap.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : Semiring R
      n : Nat
      a : R
      inst✝ : Decidable (Eq a 0)
      h : Not (Eq a 0)
      ⊢ Eq ((PowerSeries.monomial R n) a).order ↑n
    -/
  · rw [order_eq]
    /-
      case neg
      R : Type u_1
      inst✝¹ : Semiring R
      n : Nat
      a : R
      inst✝ : Decidable (Eq a 0)
      h : Not (Eq a 0)
      ⊢ And (∀ (i : Nat), Eq ↑i ↑n → Ne ((PowerSeries.coeff R i) ((PowerSeries.monom …
    -/
    constructor <;> intro i hi
      /-
        case neg.left
        R : Type u_1
        inst✝¹ : Semiring R
        n : Nat
        a : R
        inst✝ : Decidable (Eq a 0)
        h : Not (Eq a 0)
        i : Nat
        hi : Eq ↑i ↑n
        ⊢ Ne ((PowerSeries.coeff R i) ((PowerSeries.monomial R n) a)) 0
      -/
    · simp only [Nat.cast_inj] at hi
      /-
        case neg.left
        R : Type u_1
        inst✝¹ : Semiring R
        n : Nat
        a : R
        inst✝ : Decidable (Eq a 0)
        h : Not (Eq a 0)
        i : Nat
        hi : Eq i n
        ⊢ Ne ((PowerSeries.coeff R i) ((PowerSeries.monomial R n) a)) 0
      -/
      rwa [hi, coeff_monomial_same]
      /-
        🎉 no goals
      -/
      /-
        case neg.right
        R : Type u_1
        inst✝¹ : Semiring R
        n : Nat
        a : R
        inst✝ : Decidable (Eq a 0)
        h : Not (Eq a 0)
        i : Nat
        hi : LT.lt ↑i ↑n
        ⊢ Eq ((PowerSeries.coeff R i) ((PowerSeries.monomial R n) a)) 0
      -/
    · simp only [Nat.cast_lt] at hi
      /-
        case neg.right
        R : Type u_1
        inst✝¹ : Semiring R
        n : Nat
        a : R
        inst✝ : Decidable (Eq a 0)
        h : Not (Eq a 0)
        i : Nat
        hi : LT.lt i n
        ⊢ Eq ((PowerSeries.coeff R i) ((PowerSeries.monomial R n) a)) 0
      -/
      rw [coeff_monomial, if_neg]
      /-
        case neg.right.hnc
        R : Type u_1
        inst✝¹ : Semiring R
        n : Nat
        a : R
        inst✝ : Decidable (Eq a 0)
        h : Not (Eq a 0)
        i : Nat
        hi : LT.lt i n
        ⊢ Not (Eq i n)
      -/
      exact ne_of_lt hi
      /-
        🎉 no goals
      -/


/-- The order of the monomial `a*X^n` is `n` if `a ≠ 0`. -/
theorem order_monomial_of_ne_zero (n : ℕ) (a : R) (h : a ≠ 0) : order (monomial R n a) = n := by
  classical
  rw [order_monomial, if_neg h]


/-- If `n` is strictly smaller than the order of `ψ`, then the `n`th coefficient of its product
with any other power series is `0`. -/
theorem coeff_mul_of_lt_order {φ ψ : R⟦X⟧} {n : ℕ} (h : ↑n < ψ.order) :
    coeff R n (φ * ψ) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul φ ψ)) 0
  -/
  suffices coeff R n (φ * ψ) = ∑ p ∈ antidiagonal n, 0 by rw [this, Finset.sum_const_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul φ ψ)) ((Finset.HasAntidiagonal.antidi …
  -/
  rw [coeff_mul]
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun p => HMul.hMul ((PowerSe …
  -/
  apply Finset.sum_congr rfl
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    x : Prod Nat Nat
    hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) x
    ⊢ Eq (HMul.hMul ((PowerSeries.coeff R x.1) φ) ((PowerSeries.coeff R x.2) ψ)) 0
  -/
  refine mul_eq_zero_of_right (coeff R x.fst φ) (coeff_of_lt_order x.snd (lt_of_le_of_lt ?_ h))
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    x : Prod Nat Nat
    hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) x
    ⊢ LE.le ↑x.2 ↑n
  -/
  rw [mem_antidiagonal] at hx
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    x : Prod Nat Nat
    hx : Eq (HAdd.hAdd x.1 x.2) n
    ⊢ LE.le ↑x.2 ↑n
  -/
  norm_cast
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    x : Prod Nat Nat
    hx : Eq (HAdd.hAdd x.1 x.2) n
    ⊢ LE.le x.2 n
  -/
  omega
  /-
    🎉 no goals
  -/


theorem coeff_mul_one_sub_of_lt_order {R : Type*} [CommRing R] {φ ψ : R⟦X⟧} (n : ℕ)
    (h : ↑n < ψ.order) : coeff R n (φ * (1 - ψ)) = coeff R n φ := by
  /-
    R : Type u_2
    inst✝ : CommRing R
    φ ψ : PowerSeries R
    n : Nat
    h : LT.lt (↑n) ψ.order
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul φ (HSub.hSub 1 ψ))) ((PowerSeries.coe …
  -/
  simp [coeff_mul_of_lt_order h, mul_sub]
  /-
    🎉 no goals
  -/


theorem coeff_mul_prod_one_sub_of_lt_order {R ι : Type*} [CommRing R] (k : ℕ) (s : Finset ι)
    (φ : R⟦X⟧) (f : ι → R⟦X⟧) :
    (∀ i ∈ s, ↑k < (f i).order) → coeff R k (φ * ∏ i ∈ s, (1 - f i)) = coeff R k φ := by
  classical
  induction' s using Finset.induction_on with a s ha ih t
  · simp
  · intro t
    simp only [Finset.mem_insert, forall_eq_or_imp] at t
    rw [Finset.prod_insert ha, ← mul_assoc, mul_right_comm, coeff_mul_one_sub_of_lt_order _ t.1]
    exact ih t.2

-- TODO: link with `X_pow_dvd_iff`

theorem X_pow_order_dvd (h : order φ < ⊤) : X ^ (order φ).lift h ∣ φ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    h : LT.lt φ.order Top.top
    ⊢ Dvd.dvd (HPow.hPow PowerSeries.X (φ.order.lift h)) φ
  -/
  refine ⟨PowerSeries.mk fun n => coeff R (n + (order φ).lift h) φ, ?_⟩
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    h : LT.lt φ.order Top.top
    ⊢ Eq φ (HMul.hMul (HPow.hPow PowerSeries.X (φ.order.lift h)) (PowerSeries.mk f …
  -/
  ext n
  simp only [coeff_mul, coeff_X_pow, coeff_mk, boole_mul, Finset.sum_ite,
    Finset.sum_const_zero, add_zero]
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    h : LT.lt φ.order Top.top
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) φ) ((Finset.filter (fun x => Eq x.1 (φ.order.lif …
  -/
  rw [Finset.filter_fst_eq_antidiagonal n ((order φ).lift h)]
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    h : LT.lt φ.order Top.top
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) φ) ((ite (LE.le (φ.order.lift h) n) (Singleton.s …
  -/
  split_ifs with hn
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : LT.lt φ.order Top.top
      n : Nat
      hn : LE.le (φ.order.lift h) n
      ⊢ Eq ((PowerSeries.coeff R n) φ) ((Singleton.singleton { fst := φ.order.lift h …
    -/
  · simp [tsub_add_cancel_of_le hn]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : LT.lt φ.order Top.top
      n : Nat
      hn : Not (LE.le (φ.order.lift h) n)
      ⊢ Eq ((PowerSeries.coeff R n) φ) (EmptyCollection.emptyCollection.sum fun x => …
    -/
  · simp only [Finset.sum_empty]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : LT.lt φ.order Top.top
      n : Nat
      hn : Not (LE.le (φ.order.lift h) n)
      ⊢ Eq ((PowerSeries.coeff R n) φ) 0
    -/
    refine coeff_of_lt_order _ ?_
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : LT.lt φ.order Top.top
      n : Nat
      hn : Not (LE.le (φ.order.lift h) n)
      ⊢ LT.lt (↑n) φ.order
    -/
    simpa using hn
    /-
      🎉 no goals
    -/


theorem order_eq_emultiplicity_X {R : Type*} [Semiring R] (φ : R⟦X⟧) :
    order φ = emultiplicity X φ := by
  classical
  rcases eq_or_ne φ 0 with (rfl | hφ)
  · simp
  cases ho : order φ with
  | top => simp [hφ] at ho
  | coe n =>
    have hn : φ.order.lift (order_finite_iff_ne_zero.mpr hφ) = n := by simp [ho]
    rw [← hn, eq_comm]
    apply le_antisymm _
    · apply le_emultiplicity_of_pow_dvd
      apply X_pow_order_dvd
    · apply Order.le_of_lt_add_one
      rw [← not_le, ← Nat.cast_one, ← Nat.cast_add, ← pow_dvd_iff_le_emultiplicity]
      rintro ⟨ψ, H⟩
      have := congr_arg (coeff R n) H
      rw [← (ψ.commute_X.pow_right _).eq, coeff_mul_of_lt_order, ← hn] at this
      · exact coeff_order _ this
      · rw [X_pow_eq, order_monomial]
        split_ifs
        · simp
        · rw [← hn, ENat.coe_lt_coe]
          simp


/-- Given a non-zero power series `f`, `divided_by_X_pow_order f` is the power series obtained by
  dividing out the largest power of X that divides `f`, that is its order -/
def divided_by_X_pow_order {f : PowerSeries R} (hf : f ≠ 0) : R⟦X⟧ :=
  (exists_eq_mul_right_of_dvd (X_pow_order_dvd (order_finite_iff_ne_zero.2 hf))).choose


theorem self_eq_X_pow_order_mul_divided_by_X_pow_order {f : R⟦X⟧} (hf : f ≠ 0) :
    X ^ f.order.lift (order_finite_iff_ne_zero.mpr hf) * divided_by_X_pow_order hf = f :=
  haveI dvd := X_pow_order_dvd (order_finite_iff_ne_zero.mpr hf)
  (exists_eq_mul_right_of_dvd dvd).choose_spec.symm


/-- The order of the formal power series `1` is `0`. -/
@[simp]
theorem order_one : order (1 : R⟦X⟧) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    ⊢ Eq (PowerSeries.order 1) 0
  -/
  simpa using order_monomial_of_ne_zero 0 (1 : R) one_ne_zero
  /-
    🎉 no goals
  -/


/-- The order of an invertible power series is `0`. -/
theorem order_zero_of_unit {f : PowerSeries R} : IsUnit f → f.order = 0 := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    f : PowerSeries R
    ⊢ IsUnit f → Eq f.order 0
  -/
  rintro ⟨⟨u, v, hu, hv⟩, hf⟩
  /-
    case intro.mk
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    f u v : PowerSeries R
    hu : Eq (HMul.hMul u v) 1
    hv : Eq (HMul.hMul v u) 1
    hf : Eq (↑{ val := u, inv := v, val_inv := hu, inv_val := hv }) f
    ⊢ Eq f.order 0
  -/
  apply And.left
  /-
    case intro.mk.self
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    f u v : PowerSeries R
    hu : Eq (HMul.hMul u v) 1
    hv : Eq (HMul.hMul v u) 1
    hf : Eq (↑{ val := u, inv := v, val_inv := hu, inv_val := hv }) f
    ⊢ And (Eq f.order 0) ?intro.mk.b
  -/
  rw [← add_eq_zero, ← hf, ← nonpos_iff_eq_zero, ← @order_one R _ _, ← hu]
  /-
    case intro.mk.self
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    f u v : PowerSeries R
    hu : Eq (HMul.hMul u v) 1
    hv : Eq (HMul.hMul v u) 1
    hf : Eq (↑{ val := u, inv := v, val_inv := hu, inv_val := hv }) f
    ⊢ LE.le (HAdd.hAdd (↑{ val := u, inv := v, val_inv := hu, inv_val := hv }).ord …
  -/
  exact order_mul_ge _ _
  /-
    🎉 no goals
  -/


/-- The order of the formal power series `X` is `1`. -/
@[simp]
theorem order_X : order (X : R⟦X⟧) = 1 := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    ⊢ Eq PowerSeries.X.order 1
  -/
  simpa only [Nat.cast_one] using order_monomial_of_ne_zero 1 (1 : R) one_ne_zero
  /-
    🎉 no goals
  -/


/-- The order of the formal power series `X^n` is `n`. -/
@[simp]
theorem order_X_pow (n : ℕ) : order ((X : R⟦X⟧) ^ n) = n := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (HPow.hPow PowerSeries.X n).order ↑n
  -/
  rw [X_pow_eq, order_monomial_of_ne_zero]
  /-
    case h
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Ne 1 0
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


/-- The order of the product of two formal power series over an integral domain
 is the sum of their orders. -/
theorem order_mul (φ ψ : R⟦X⟧) : order (φ * ψ) = order φ + order ψ := by
  classical
  simp only [order_eq_emultiplicity_X]
  rw [emultiplicity_mul X_prime]

-- Dividing `X` by the maximal power of `X` dividing it leaves `1`.

@[simp]
theorem divided_by_X_pow_order_of_X_eq_one : divided_by_X_pow_order X_ne_zero = (1 : R⟦X⟧) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq (PowerSeries.divided_by_X_pow_order ⋯) 1
  -/
  rw [← mul_eq_left₀ X_ne_zero]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq (HMul.hMul PowerSeries.X (PowerSeries.divided_by_X_pow_order ⋯)) PowerSer …
  -/
  simpa using self_eq_X_pow_order_mul_divided_by_X_pow_order (@X_ne_zero R _ _)
  /-
    🎉 no goals
  -/

-- Dividing a power series by the maximal power of `X` dividing it, respects multiplication.

theorem divided_by_X_pow_orderMul {f g : R⟦X⟧} (hf : f ≠ 0) (hg : g ≠ 0) :
    divided_by_X_pow_order hf * divided_by_X_pow_order hg =
      divided_by_X_pow_order (mul_ne_zero hf hg) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    ⊢ Eq (HMul.hMul (PowerSeries.divided_by_X_pow_order hf) (PowerSeries.divided_b …
  -/
  set df := f.order.lift (order_finite_iff_ne_zero.mpr hf)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    df : Nat := f.order.lift ⋯
    ⊢ Eq (HMul.hMul (PowerSeries.divided_by_X_pow_order hf) (PowerSeries.divided_b …
  -/
  set dg := g.order.lift (order_finite_iff_ne_zero.mpr hg)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    df : Nat := f.order.lift ⋯
    dg : Nat := g.order.lift ⋯
    ⊢ Eq (HMul.hMul (PowerSeries.divided_by_X_pow_order hf) (PowerSeries.divided_b …
  -/
  set dfg := (f * g).order.lift (order_finite_iff_ne_zero.mpr (mul_ne_zero hf hg))
  have H_add_d : df + dg = dfg := by
    simp_all [df, dg, dfg, order_mul f g]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    df : Nat := f.order.lift ⋯
    dg : Nat := g.order.lift ⋯
    dfg : Nat := (HMul.hMul f g).order.lift ⋯
    H_add_d : Eq (HAdd.hAdd df dg) dfg
    ⊢ Eq (HMul.hMul (PowerSeries.divided_by_X_pow_order hf) (PowerSeries.divided_b …
  -/
  have H := self_eq_X_pow_order_mul_divided_by_X_pow_order (mul_ne_zero hf hg)
  have : f * g = X ^ dfg * (divided_by_X_pow_order hf * divided_by_X_pow_order hg) := by
    calc
      f * g = X ^ df * divided_by_X_pow_order hf * (X ^ dg * divided_by_X_pow_order hg) := by
        rw [self_eq_X_pow_order_mul_divided_by_X_pow_order,
          self_eq_X_pow_order_mul_divided_by_X_pow_order]
      _ = X ^ df * X ^ dg * divided_by_X_pow_order hf * divided_by_X_pow_order hg := by ring
      _ = X ^ (df + dg) * divided_by_X_pow_order hf * divided_by_X_pow_order hg := by rw [pow_add]
      _ = X ^ dfg * divided_by_X_pow_order hf * divided_by_X_pow_order hg := by rw [H_add_d]
      _ = X ^ dfg * (divided_by_X_pow_order hf * divided_by_X_pow_order hg) := by rw [mul_assoc]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    df : Nat := f.order.lift ⋯
    dg : Nat := g.order.lift ⋯
    dfg : Nat := (HMul.hMul f g).order.lift ⋯
    H_add_d : Eq (HAdd.hAdd df dg) dfg
    H : Eq (HMul.hMul (HPow.hPow PowerSeries.X ((HMul.hMul f g).order.lift ⋯)) (Po …
    this : Eq (HMul.hMul f g) (HMul.hMul (HPow.hPow PowerSeries.X dfg) (HMul.hMul  …
    ⊢ Eq (HMul.hMul (PowerSeries.divided_by_X_pow_order hf) (PowerSeries.divided_b …
  -/
  refine (IsLeftCancelMulZero.mul_left_cancel_of_ne_zero (pow_ne_zero dfg X_ne_zero) ?_).symm
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    df : Nat := f.order.lift ⋯
    dg : Nat := g.order.lift ⋯
    dfg : Nat := (HMul.hMul f g).order.lift ⋯
    H_add_d : Eq (HAdd.hAdd df dg) dfg
    H : Eq (HMul.hMul (HPow.hPow PowerSeries.X ((HMul.hMul f g).order.lift ⋯)) (Po …
    this : Eq (HMul.hMul f g) (HMul.hMul (HPow.hPow PowerSeries.X dfg) (HMul.hMul  …
    ⊢ Eq (HMul.hMul (HPow.hPow PowerSeries.X dfg) (PowerSeries.divided_by_X_pow_or …
  -/
  simp only [this] at H
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    f g : PowerSeries R
    hf : Ne f 0
    hg : Ne g 0
    df : Nat := f.order.lift ⋯
    dg : Nat := g.order.lift ⋯
    dfg : Nat := (HMul.hMul f g).order.lift ⋯
    H_add_d : Eq (HAdd.hAdd df dg) dfg
    this : Eq (HMul.hMul f g) (HMul.hMul (HPow.hPow PowerSeries.X dfg) (HMul.hMul  …
    H : Eq (HMul.hMul (HPow.hPow PowerSeries.X ((HMul.hMul (HPow.hPow PowerSeries. …
    ⊢ Eq (HMul.hMul (HPow.hPow PowerSeries.X dfg) (PowerSeries.divided_by_X_pow_or …
  -/
  convert H
  /-
    🎉 no goals
  -/


