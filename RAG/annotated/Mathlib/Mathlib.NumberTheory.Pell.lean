/-- An element of `ℤ√d` has norm one (i.e., `a.re^2 - d*a.im^2 = 1`) if and only if
it is contained in the submonoid of unitary elements.

TODO: merge this result with `Pell.isPell_iff_mem_unitary`. -/
theorem is_pell_solution_iff_mem_unitary {d : ℤ} {a : ℤ√d} :
    a.re ^ 2 - d * a.im ^ 2 = 1 ↔ a ∈ unitary (ℤ√d) := by
  /-
    d : Int
    a : Zsqrtd d
    ⊢ Iff (Eq (HSub.hSub (HPow.hPow a.re 2) (HMul.hMul d (HPow.hPow a.im 2))) 1) ( …
  -/
  rw [← norm_eq_one_iff_mem_unitary, norm_def, sq, sq, ← mul_assoc]
  /-
    🎉 no goals
  -/

-- We use `solution₁ d` to allow for a more general structure `solution d m` that
-- encodes solutions to `x^2 - d*y^2 = m` to be added later.

/-- `Pell.Solution₁ d` is the type of solutions to the Pell equation `x^2 - d*y^2 = 1`.
We define this in terms of elements of `ℤ√d` of norm one.
-/
def Solution₁ (d : ℤ) : Type :=
  ↥(unitary (ℤ√d))


instance instCommGroup : CommGroup (Solution₁ d) :=
  inferInstanceAs (CommGroup (unitary (ℤ√d)))


instance instHasDistribNeg : HasDistribNeg (Solution₁ d) :=
  inferInstanceAs (HasDistribNeg (unitary (ℤ√d)))


instance instInhabited : Inhabited (Solution₁ d) :=
  inferInstanceAs (Inhabited (unitary (ℤ√d)))


instance : Coe (Solution₁ d) (ℤ√d) where coe := Subtype.val


/-- The `x` component of a solution to the Pell equation `x^2 - d*y^2 = 1` -/
protected def x (a : Solution₁ d) : ℤ :=
  (a : ℤ√d).re


/-- The `y` component of a solution to the Pell equation `x^2 - d*y^2 = 1` -/
protected def y (a : Solution₁ d) : ℤ :=
  (a : ℤ√d).im


/-- The proof that `a` is a solution to the Pell equation `x^2 - d*y^2 = 1` -/
theorem prop (a : Solution₁ d) : a.x ^ 2 - d * a.y ^ 2 = 1 :=
  is_pell_solution_iff_mem_unitary.mpr a.property


/-- An alternative form of the equation, suitable for rewriting `x^2`. -/
                                                                   /-
                                                                     d : Int
                                                                     a : Pell.Solution₁ d
                                                                     ⊢ Eq (HPow.hPow a.x 2) (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow a.y 2)))
                                                                   -/
theorem prop_x (a : Solution₁ d) : a.x ^ 2 = 1 + d * a.y ^ 2 := by rw [← a.prop]; ring
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- An alternative form of the equation, suitable for rewriting `d * y^2`. -/
                                                                   /-
                                                                     d : Int
                                                                     a : Pell.Solution₁ d
                                                                     ⊢ Eq (HMul.hMul d (HPow.hPow a.y 2)) (HSub.hSub (HPow.hPow a.x 2) 1)
                                                                   -/
theorem prop_y (a : Solution₁ d) : d * a.y ^ 2 = a.x ^ 2 - 1 := by rw [← a.prop]; ring
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- Two solutions are equal if their `x` and `y` components are equal. -/
@[ext]
theorem ext {a b : Solution₁ d} (hx : a.x = b.x) (hy : a.y = b.y) : a = b :=
  Subtype.ext <| Zsqrtd.ext hx hy


/-- Construct a solution from `x`, `y` and a proof that the equation is satisfied. -/
def mk (x y : ℤ) (prop : x ^ 2 - d * y ^ 2 = 1) : Solution₁ d where
  val := ⟨x, y⟩
  property := is_pell_solution_iff_mem_unitary.mp prop


@[simp]
theorem x_mk (x y : ℤ) (prop : x ^ 2 - d * y ^ 2 = 1) : (mk x y prop).x = x :=
  rfl


@[simp]
theorem y_mk (x y : ℤ) (prop : x ^ 2 - d * y ^ 2 = 1) : (mk x y prop).y = y :=
  rfl


@[simp]
theorem coe_mk (x y : ℤ) (prop : x ^ 2 - d * y ^ 2 = 1) : (↑(mk x y prop) : ℤ√d) = ⟨x, y⟩ :=
  Zsqrtd.ext (x_mk x y prop) (y_mk x y prop)


@[simp]
theorem x_one : (1 : Solution₁ d).x = 1 :=
  rfl


@[simp]
theorem y_one : (1 : Solution₁ d).y = 0 :=
  rfl


@[simp]
theorem x_mul (a b : Solution₁ d) : (a * b).x = a.x * b.x + d * (a.y * b.y) := by
  /-
    d : Int
    a b : Pell.Solution₁ d
    ⊢ Eq (HMul.hMul a b).x (HAdd.hAdd (HMul.hMul a.x b.x) (HMul.hMul d (HMul.hMul  …
  -/
  rw [← mul_assoc]
  /-
    d : Int
    a b : Pell.Solution₁ d
    ⊢ Eq (HMul.hMul a b).x (HAdd.hAdd (HMul.hMul a.x b.x) (HMul.hMul (HMul.hMul d  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem y_mul (a b : Solution₁ d) : (a * b).y = a.x * b.y + a.y * b.x :=
  rfl


@[simp]
theorem x_inv (a : Solution₁ d) : a⁻¹.x = a.x :=
  rfl


@[simp]
theorem y_inv (a : Solution₁ d) : a⁻¹.y = -a.y :=
  rfl


@[simp]
theorem x_neg (a : Solution₁ d) : (-a).x = -a.x :=
  rfl


@[simp]
theorem y_neg (a : Solution₁ d) : (-a).y = -a.y :=
  rfl


/-- When `d` is negative, then `x` or `y` must be zero in a solution. -/
theorem eq_zero_of_d_neg (h₀ : d < 0) (a : Solution₁ d) : a.x = 0 ∨ a.y = 0 := by
  /-
    d : Int
    h₀ : LT.lt d 0
    a : Pell.Solution₁ d
    ⊢ Or (Eq a.x 0) (Eq a.y 0)
  -/
  have h := a.prop
  /-
    d : Int
    h₀ : LT.lt d 0
    a : Pell.Solution₁ d
    h : Eq (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
    ⊢ Or (Eq a.x 0) (Eq a.y 0)
  -/
  contrapose! h
  /-
    d : Int
    h₀ : LT.lt d 0
    a : Pell.Solution₁ d
    h : And (Ne a.x 0) (Ne a.y 0)
    ⊢ Ne (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
  -/
  have h1 := sq_pos_of_ne_zero h.1
  /-
    d : Int
    h₀ : LT.lt d 0
    a : Pell.Solution₁ d
    h : And (Ne a.x 0) (Ne a.y 0)
    h1 : LT.lt 0 (HPow.hPow a.x 2)
    ⊢ Ne (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
  -/
  have h2 := sq_pos_of_ne_zero h.2
  /-
    d : Int
    h₀ : LT.lt d 0
    a : Pell.Solution₁ d
    h : And (Ne a.x 0) (Ne a.y 0)
    h1 : LT.lt 0 (HPow.hPow a.x 2)
    h2 : LT.lt 0 (HPow.hPow a.y 2)
    ⊢ Ne (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
  -/
  nlinarith
  /-
    🎉 no goals
  -/


/-- A solution has `x ≠ 0`. -/
theorem x_ne_zero (h₀ : 0 ≤ d) (a : Solution₁ d) : a.x ≠ 0 := by
  /-
    d : Int
    h₀ : LE.le 0 d
    a : Pell.Solution₁ d
    ⊢ Ne a.x 0
  -/
  intro hx
  /-
    d : Int
    h₀ : LE.le 0 d
    a : Pell.Solution₁ d
    hx : Eq a.x 0
    ⊢ False
  -/
  have h : 0 ≤ d * a.y ^ 2 := mul_nonneg h₀ (sq_nonneg _)
  /-
    d : Int
    h₀ : LE.le 0 d
    a : Pell.Solution₁ d
    hx : Eq a.x 0
    h : LE.le 0 (HMul.hMul d (HPow.hPow a.y 2))
    ⊢ False
  -/
  rw [a.prop_y, hx, sq, zero_mul, zero_sub] at h
  /-
    d : Int
    h₀ : LE.le 0 d
    a : Pell.Solution₁ d
    hx : Eq a.x 0
    h : LE.le 0 (-1)
    ⊢ False
  -/
  exact not_le.mpr (neg_one_lt_zero : (-1 : ℤ) < 0) h
  /-
    🎉 no goals
  -/


/-- A solution with `x > 1` must have `y ≠ 0`. -/
theorem y_ne_zero_of_one_lt_x {a : Solution₁ d} (ha : 1 < a.x) : a.y ≠ 0 := by
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    ⊢ Ne a.y 0
  -/
  intro hy
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    hy : Eq a.y 0
    ⊢ False
  -/
  have prop := a.prop
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    hy : Eq a.y 0
    prop : Eq (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
    ⊢ False
  -/
  rw [hy, sq (0 : ℤ), zero_mul, mul_zero, sub_zero] at prop
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    hy : Eq a.y 0
    prop : Eq (HPow.hPow a.x 2) 1
    ⊢ False
  -/
  exact lt_irrefl _ (((one_lt_sq_iff₀ <| zero_le_one.trans ha.le).mpr ha).trans_eq prop)
  /-
    🎉 no goals
  -/


/-- If a solution has `x > 1`, then `d` is positive. -/
theorem d_pos_of_one_lt_x {a : Solution₁ d} (ha : 1 < a.x) : 0 < d := by
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    ⊢ LT.lt 0 d
  -/
  refine pos_of_mul_pos_left ?_ (sq_nonneg a.y)
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    ⊢ LT.lt 0 (HMul.hMul d (HPow.hPow a.y 2))
  -/
  rw [a.prop_y, sub_pos]
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    ⊢ LT.lt 1 (HPow.hPow a.x 2)
  -/
  exact one_lt_pow₀ ha two_ne_zero
  /-
    🎉 no goals
  -/


/-- If a solution has `x > 1`, then `d` is not a square. -/
theorem d_nonsquare_of_one_lt_x {a : Solution₁ d} (ha : 1 < a.x) : ¬IsSquare d := by
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    ⊢ Not (IsSquare d)
  -/
  have hp := a.prop
  /-
    d : Int
    a : Pell.Solution₁ d
    ha : LT.lt 1 a.x
    hp : Eq (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
    ⊢ Not (IsSquare d)
  -/
  rintro ⟨b, rfl⟩
  /-
    case intro
    b : Int
    a : Pell.Solution₁ (HMul.hMul b b)
    ha : LT.lt 1 a.x
    hp : Eq (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul (HMul.hMul b b) (HPow.hPow a.y …
    ⊢ False
  -/
  simp_rw [← sq, ← mul_pow, sq_sub_sq, Int.mul_eq_one_iff_eq_one_or_neg_one] at hp
  /-
    case intro
    b : Int
    a : Pell.Solution₁ (HMul.hMul b b)
    ha : LT.lt 1 a.x
    hp : Or (And (Eq (HAdd.hAdd a.x (HMul.hMul b a.y)) 1) (Eq (HSub.hSub a.x (HMul …
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


/-- A solution with `x = 1` is trivial. -/
theorem eq_one_of_x_eq_one (h₀ : d ≠ 0) {a : Solution₁ d} (ha : a.x = 1) : a = 1 := by
  /-
    d : Int
    h₀ : Ne d 0
    a : Pell.Solution₁ d
    ha : Eq a.x 1
    ⊢ Eq a 1
  -/
  have prop := a.prop_y
  /-
    d : Int
    h₀ : Ne d 0
    a : Pell.Solution₁ d
    ha : Eq a.x 1
    prop : Eq (HMul.hMul d (HPow.hPow a.y 2)) (HSub.hSub (HPow.hPow a.x 2) 1)
    ⊢ Eq a 1
  -/
  rw [ha, one_pow, sub_self, mul_eq_zero, or_iff_right h₀, sq_eq_zero_iff] at prop
  /-
    d : Int
    h₀ : Ne d 0
    a : Pell.Solution₁ d
    ha : Eq a.x 1
    prop : Eq a.y 0
    ⊢ Eq a 1
  -/
  exact ext ha prop
  /-
    🎉 no goals
  -/


/-- A solution is `1` or `-1` if and only if `y = 0`. -/
theorem eq_one_or_neg_one_iff_y_eq_zero {a : Solution₁ d} : a = 1 ∨ a = -1 ↔ a.y = 0 := by
  /-
    d : Int
    a : Pell.Solution₁ d
    ⊢ Iff (Or (Eq a 1) (Eq a (-1))) (Eq a.y 0)
  -/
  refine ⟨fun H => H.elim (fun h => by simp [h]) fun h => by simp [h], fun H => ?_⟩
  /-
    d : Int
    a : Pell.Solution₁ d
    H : Eq a.y 0
    ⊢ Or (Eq a 1) (Eq a (-1))
  -/
  have prop := a.prop
  /-
    d : Int
    a : Pell.Solution₁ d
    H : Eq a.y 0
    prop : Eq (HSub.hSub (HPow.hPow a.x 2) (HMul.hMul d (HPow.hPow a.y 2))) 1
    ⊢ Or (Eq a 1) (Eq a (-1))
  -/
  rw [H, sq (0 : ℤ), mul_zero, mul_zero, sub_zero, sq_eq_one_iff] at prop
  /-
    d : Int
    a : Pell.Solution₁ d
    H : Eq a.y 0
    prop : Or (Eq a.x 1) (Eq a.x (-1))
    ⊢ Or (Eq a 1) (Eq a (-1))
  -/
  exact prop.imp (fun h => ext h H) fun h => ext h H
  /-
    🎉 no goals
  -/


/-- The set of solutions with `x > 0` is closed under multiplication. -/
theorem x_mul_pos {a b : Solution₁ d} (ha : 0 < a.x) (hb : 0 < b.x) : 0 < (a * b).x := by
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : LT.lt 0 a.x
    hb : LT.lt 0 b.x
    ⊢ LT.lt 0 (HMul.hMul a b).x
  -/
  simp only [x_mul]
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : LT.lt 0 a.x
    hb : LT.lt 0 b.x
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul a.x b.x) (HMul.hMul d (HMul.hMul a.y b.y)))
  -/
  refine neg_lt_iff_pos_add'.mp (abs_lt.mp ?_).1
  rw [← abs_of_pos ha, ← abs_of_pos hb, ← abs_mul, ← sq_lt_sq, mul_pow a.x, a.prop_x, b.prop_x, ←
    sub_pos]
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : LT.lt 0 a.x
    hb : LT.lt 0 b.x
    ⊢ LT.lt 0 (HSub.hSub (HMul.hMul (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow a.y 2)))  …
  -/
  ring_nf
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : LT.lt 0 a.x
    hb : LT.lt 0 b.x
    ⊢ LT.lt 0 (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow a.y 2))) (HMul.hMul  …
  -/
  rcases le_or_lt 0 d with h | h
    /-
      case inl
      d : Int
      a b : Pell.Solution₁ d
      ha : LT.lt 0 a.x
      hb : LT.lt 0 b.x
      h : LE.le 0 d
      ⊢ LT.lt 0 (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow a.y 2))) (HMul.hMul  …
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case inr
      d : Int
      a b : Pell.Solution₁ d
      ha : LT.lt 0 a.x
      hb : LT.lt 0 b.x
      h : LT.lt d 0
      ⊢ LT.lt 0 (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow a.y 2))) (HMul.hMul  …
    -/
  · rw [(eq_zero_of_d_neg h a).resolve_left ha.ne', (eq_zero_of_d_neg h b).resolve_left hb.ne']
    -- Porting note: was
    -- rw [zero_pow two_ne_zero, zero_add, zero_mul, zero_add]
    -- exact one_pos
    -- but this relied on the exact output of `ring_nf`
    /-
      case inr
      d : Int
      a b : Pell.Solution₁ d
      ha : LT.lt 0 a.x
      hb : LT.lt 0 b.x
      h : LT.lt d 0
      ⊢ LT.lt 0 (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow 0 2))) (HMul.hMul d  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The set of solutions with `x` and `y` positive is closed under multiplication. -/
theorem y_mul_pos {a b : Solution₁ d} (hax : 0 < a.x) (hay : 0 < a.y) (hbx : 0 < b.x)
    (hby : 0 < b.y) : 0 < (a * b).y := by
  /-
    d : Int
    a b : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    hbx : LT.lt 0 b.x
    hby : LT.lt 0 b.y
    ⊢ LT.lt 0 (HMul.hMul a b).y
  -/
  simp only [y_mul]
  /-
    d : Int
    a b : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    hbx : LT.lt 0 b.x
    hby : LT.lt 0 b.y
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul a.x b.y) (HMul.hMul a.y b.x))
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- If `(x, y)` is a solution with `x` positive, then all its powers with natural exponents
have positive `x`. -/
theorem x_pow_pos {a : Solution₁ d} (hax : 0 < a.x) (n : ℕ) : 0 < (a ^ n).x := by
  /-
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    n : Nat
    ⊢ LT.lt 0 (HPow.hPow a n).x
  -/
  induction' n with n ih
    /-
      case zero
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      ⊢ LT.lt 0 (HPow.hPow a 0).x
    -/
  · simp only [pow_zero, x_one, zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      n : Nat
      ih : LT.lt 0 (HPow.hPow a n).x
      ⊢ LT.lt 0 (HPow.hPow a (HAdd.hAdd n 1)).x
    -/
  · rw [pow_succ]
    /-
      case succ
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      n : Nat
      ih : LT.lt 0 (HPow.hPow a n).x
      ⊢ LT.lt 0 (HMul.hMul (HPow.hPow a n) a).x
    -/
    exact x_mul_pos ih hax
    /-
      🎉 no goals
    -/


/-- If `(x, y)` is a solution with `x` and `y` positive, then all its powers with positive
natural exponents have positive `y`. -/
theorem y_pow_succ_pos {a : Solution₁ d} (hax : 0 < a.x) (hay : 0 < a.y) (n : ℕ) :
    0 < (a ^ n.succ).y := by
  /-
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    n : Nat
    ⊢ LT.lt 0 (HPow.hPow a n.succ).y
  -/
  induction' n with n ih
    /-
      case zero
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      ⊢ LT.lt 0 (HPow.hPow a (Nat.succ 0)).y
    -/
  · simp only [pow_one, hay]
    /-
      🎉 no goals
    -/
    /-
      case succ
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      n : Nat
      ih : LT.lt 0 (HPow.hPow a n.succ).y
      ⊢ LT.lt 0 (HPow.hPow a (HAdd.hAdd n 1).succ).y
    -/
  · rw [pow_succ']
    /-
      case succ
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      n : Nat
      ih : LT.lt 0 (HPow.hPow a n.succ).y
      ⊢ LT.lt 0 (HMul.hMul a (HPow.hPow a (HAdd.hAdd n 1))).y
    -/
    exact y_mul_pos hax hay (x_pow_pos hax _) ih
    /-
      🎉 no goals
    -/


/-- If `(x, y)` is a solution with `x` and `y` positive, then all its powers with positive
exponents have positive `y`. -/
theorem y_zpow_pos {a : Solution₁ d} (hax : 0 < a.x) (hay : 0 < a.y) {n : ℤ} (hn : 0 < n) :
    0 < (a ^ n).y := by
  /-
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    n : Int
    hn : LT.lt 0 n
    ⊢ LT.lt 0 (HPow.hPow a n).y
  -/
  lift n to ℕ using hn.le
  /-
    case intro
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    n : Nat
    hn : LT.lt 0 ↑n
    ⊢ LT.lt 0 (HPow.hPow a ↑n).y
  -/
  norm_cast at hn ⊢
  /-
    case intro
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    n : Nat
    hn : LT.lt 0 n
    ⊢ LT.lt 0 (HPow.hPow a n).y
  -/
  rw [← Nat.succ_pred_eq_of_pos hn]
  /-
    case intro
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    n : Nat
    hn : LT.lt 0 n
    ⊢ LT.lt 0 (HPow.hPow a n.pred.succ).y
  -/
  exact y_pow_succ_pos hax hay _
  /-
    🎉 no goals
  -/


/-- If `(x, y)` is a solution with `x` positive, then all its powers have positive `x`. -/
theorem x_zpow_pos {a : Solution₁ d} (hax : 0 < a.x) (n : ℤ) : 0 < (a ^ n).x := by
  cases n with
  | ofNat n =>
    rw [Int.ofNat_eq_coe, zpow_natCast]
    exact x_pow_pos hax n
  | negSucc n =>
    rw [zpow_negSucc]
    exact x_pow_pos hax (n + 1)


/-- If `(x, y)` is a solution with `x` and `y` positive, then the `y` component of any power
has the same sign as the exponent. -/
theorem sign_y_zpow_eq_sign_of_x_pos_of_y_pos {a : Solution₁ d} (hax : 0 < a.x) (hay : 0 < a.y)
    (n : ℤ) : (a ^ n).y.sign = n.sign := by
  /-
    d : Int
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LT.lt 0 a.y
    n : Int
    ⊢ Eq (HPow.hPow a n).y.sign n.sign
  -/
  rcases n with ((_ | n) | n)
    /-
      case ofNat.zero
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      ⊢ Eq (HPow.hPow a (Int.ofNat 0)).y.sign (Int.ofNat 0).sign
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case ofNat.succ
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      n : Nat
      ⊢ Eq (HPow.hPow a (Int.ofNat (HAdd.hAdd n 1))).y.sign (Int.ofNat (HAdd.hAdd n  …
    -/
  · rw [Int.ofNat_eq_coe, zpow_natCast]
    /-
      case ofNat.succ
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      n : Nat
      ⊢ Eq (HPow.hPow a (HAdd.hAdd n 1)).y.sign (↑(HAdd.hAdd n 1)).sign
    -/
    exact Int.sign_eq_one_of_pos (y_pow_succ_pos hax hay n)
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      n : Nat
      ⊢ Eq (HPow.hPow a (Int.negSucc n)).y.sign (Int.negSucc n).sign
    -/
  · rw [zpow_negSucc]
    /-
      case negSucc
      d : Int
      a : Pell.Solution₁ d
      hax : LT.lt 0 a.x
      hay : LT.lt 0 a.y
      n : Nat
      ⊢ Eq (Inv.inv (HPow.hPow a (HAdd.hAdd n 1))).y.sign (Int.negSucc n).sign
    -/
    exact Int.sign_eq_neg_one_of_neg (neg_neg_of_pos (y_pow_succ_pos hax hay n))
    /-
      🎉 no goals
    -/


/-- If `a` is any solution, then one of `a`, `a⁻¹`, `-a`, `-a⁻¹` has
positive `x` and nonnegative `y`. -/
theorem exists_pos_variant (h₀ : 0 < d) (a : Solution₁ d) :
    ∃ b : Solution₁ d, 0 < b.x ∧ 0 ≤ b.y ∧ a ∈ ({b, b⁻¹, -b, -b⁻¹} : Set (Solution₁ d)) := by
  refine
        (lt_or_gt_of_ne (a.x_ne_zero h₀.le)).elim
          ((le_total 0 a.y).elim (fun hy hx => ⟨-a⁻¹, ?_, ?_, ?_⟩) fun hy hx => ⟨-a, ?_, ?_, ?_⟩)
          ((le_total 0 a.y).elim (fun hy hx => ⟨a, hx, hy, ?_⟩) fun hy hx => ⟨a⁻¹, hx, ?_, ?_⟩) <;>
      simp only [neg_neg, inv_inv, neg_inv, Set.mem_insert_iff, Set.mem_singleton_iff, true_or,
        eq_self_iff_true, x_neg, x_inv, y_neg, y_inv, neg_pos, neg_nonneg, or_true] <;>
    /-
      case refine_1
      d : Int
      h₀ : LT.lt 0 d
      a : Pell.Solution₁ d
      hy : LE.le 0 a.y
      hx : LT.lt a.x 0
      ⊢ LT.lt a.x 0
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    assumption
    /-
      🎉 no goals
    -/


/-- If `d` is a positive integer that is not a square, then there is a nontrivial solution
to the Pell equation `x^2 - d*y^2 = 1`. -/
theorem exists_of_not_isSquare (h₀ : 0 < d) (hd : ¬IsSquare d) :
    ∃ x y : ℤ, x ^ 2 - d * y ^ 2 = 1 ∧ y ≠ 0 := by
  /-
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ⊢ Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMu …
  -/
  let ξ : ℝ := √d
  have hξ : Irrational ξ := by
    refine irrational_nrt_of_notint_nrt 2 d (sq_sqrt <| Int.cast_nonneg.mpr h₀.le) ?_ two_pos
    rintro ⟨x, hx⟩
    refine hd ⟨x, @Int.cast_injective ℝ _ _ d (x * x) ?_⟩
    rw [← sq_sqrt <| Int.cast_nonneg.mpr h₀.le, Int.cast_mul, ← hx, sq]
  /-
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ξ : Real := (↑d).sqrt
    hξ : Irrational ξ
    ⊢ Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMu …
  -/
  obtain ⟨M, hM₁⟩ := exists_int_gt (2 * |ξ| + 1)
  have hM : {q : ℚ | |q.1 ^ 2 - d * (q.2 : ℤ) ^ 2| < M}.Infinite := by
    refine Infinite.mono (fun q h => ?_) (infinite_rat_abs_sub_lt_one_div_den_sq_of_irrational hξ)
    have h0 : 0 < (q.2 : ℝ) ^ 2 := pow_pos (Nat.cast_pos.mpr q.pos) 2
    have h1 : (q.num : ℝ) / (q.den : ℝ) = q := mod_cast q.num_div_den
    rw [mem_setOf, abs_sub_comm, ← @Int.cast_lt ℝ,
      ← div_lt_div_iff_of_pos_right (abs_pos_of_pos h0)]
    push_cast
    rw [← abs_div, abs_sq, sub_div, mul_div_cancel_right₀ _ h0.ne', ← div_pow, h1, ←
      sq_sqrt (Int.cast_pos.mpr h₀).le, sq_sub_sq, abs_mul, ← mul_one_div]
    refine mul_lt_mul'' (((abs_add ξ q).trans ?_).trans_lt hM₁) h (abs_nonneg _) (abs_nonneg _)
    rw [two_mul, add_assoc, add_le_add_iff_left, ← sub_le_iff_le_add']
    rw [mem_setOf, abs_sub_comm] at h
    refine (abs_sub_abs_le_abs_sub (q : ℝ) ξ).trans (h.le.trans ?_)
    rw [div_le_one h0, one_le_sq_iff_one_le_abs, Nat.abs_cast, Nat.one_le_cast]
    exact q.pos
  obtain ⟨m, hm⟩ : ∃ m : ℤ, {q : ℚ | q.1 ^ 2 - d * (q.den : ℤ) ^ 2 = m}.Infinite := by
    contrapose! hM
    simp only [not_infinite] at hM ⊢
    refine (congr_arg _ (ext fun x => ?_)).mp (Finite.biUnion (finite_Ioo (-M) M) fun m _ => hM m)
    simp only [abs_lt, mem_setOf, mem_Ioo, mem_iUnion, exists_prop, exists_eq_right']
  have hm₀ : m ≠ 0 := by
    rintro rfl
    obtain ⟨q, hq⟩ := hm.nonempty
    rw [mem_setOf, sub_eq_zero, mul_comm] at hq
    obtain ⟨a, ha⟩ := (Int.pow_dvd_pow_iff two_ne_zero).mp ⟨d, hq⟩
    rw [ha, mul_pow, mul_right_inj' (pow_pos (Int.natCast_pos.mpr q.pos) 2).ne'] at hq
    exact hd ⟨a, sq a ▸ hq.symm⟩
  /-
    case intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ξ : Real := (↑d).sqrt
    hξ : Irrational ξ
    M : Int
    hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
    hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
    m : Int
    hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
    hm₀ : Ne m 0
    ⊢ Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMu …
  -/
  haveI := neZero_iff.mpr (Int.natAbs_ne_zero.mpr hm₀)
  /-
    case intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ξ : Real := (↑d).sqrt
    hξ : Irrational ξ
    M : Int
    hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
    hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
    m : Int
    hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
    hm₀ : Ne m 0
    this : NeZero m.natAbs
    ⊢ Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMu …
  -/
  let f : ℚ → ZMod m.natAbs × ZMod m.natAbs := fun q => (q.num, q.den)
  obtain ⟨q₁, h₁ : q₁.num ^ 2 - d * (q₁.den : ℤ) ^ 2 = m,
      q₂, h₂ : q₂.num ^ 2 - d * (q₂.den : ℤ) ^ 2 = m, hne, hqf⟩ :=
    hm.exists_ne_map_eq_of_mapsTo (mapsTo_univ f _) finite_univ
  obtain ⟨hq1 : (q₁.num : ZMod m.natAbs) = q₂.num, hq2 : (q₁.den : ZMod m.natAbs) = q₂.den⟩ :=
    Prod.ext_iff.mp hqf
  have hd₁ : m ∣ q₁.num * q₂.num - d * (q₁.den * q₂.den) := by
    rw [← Int.natAbs_dvd, ← ZMod.intCast_zmod_eq_zero_iff_dvd]
    push_cast
    rw [hq1, hq2, ← sq, ← sq]
    norm_cast
    rw [ZMod.intCast_zmod_eq_zero_iff_dvd, Int.natAbs_dvd, Nat.cast_pow, ← h₂]
  have hd₂ : m ∣ q₁.num * q₂.den - q₂.num * q₁.den := by
    rw [← Int.natAbs_dvd, ← ZMod.intCast_eq_intCast_iff_dvd_sub]
    push_cast
    rw [hq1, hq2]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ξ : Real := (↑d).sqrt
    hξ : Irrational ξ
    M : Int
    hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
    hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
    m : Int
    hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
    hm₀ : Ne m 0
    this : NeZero m.natAbs
    f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
    q₁ : Rat
    h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
    q₂ : Rat
    h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
    hne : Ne q₁ q₂
    hqf : Eq (f q₁) (f q₂)
    hq1 : Eq ↑q₁.num ↑q₂.num
    hq2 : Eq ↑q₁.den ↑q₂.den
    hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
    hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
    ⊢ Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMu …
  -/
  replace hm₀ : (m : ℚ) ≠ 0 := Int.cast_ne_zero.mpr hm₀
  refine ⟨(q₁.num * q₂.num - d * (q₁.den * q₂.den)) / m, (q₁.num * q₂.den - q₂.num * q₁.den) / m,
      ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Eq (HSub.hSub (HPow.hPow (HDiv.hDiv (HSub.hSub (HMul.hMul q₁.num q₂.num) (HM …
    -/
  · qify [hd₁, hd₂]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Eq (HSub.hSub (HPow.hPow (HDiv.hDiv (HSub.hSub (HMul.hMul ↑q₁.num ↑q₂.num) ( …
    -/
    field_simp [hm₀]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Eq (HSub.hSub (HPow.hPow (HSub.hSub (HMul.hMul ↑q₁.num ↑q₂.num) (HMul.hMul ( …
    -/
    norm_cast
    conv_rhs =>
      rw [sq]
      congr
      · rw [← h₁]
      · rw [← h₂]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Eq (HSub.hSub (HPow.hPow (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d ↑ …
    -/
    push_cast
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Eq (HSub.hSub (HPow.hPow (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d ( …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Ne (HDiv.hDiv (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
    -/
  · qify [hd₂]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Ne (HDiv.hDiv (HSub.hSub (HMul.hMul ↑q₁.num ↑q₂.den) (HMul.hMul ↑q₂.num ↑q₁. …
    -/
    refine div_ne_zero_iff.mpr ⟨?_, hm₀⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      ξ : Real := (↑d).sqrt
      hξ : Irrational ξ
      M : Int
      hM₁ : LT.lt (HAdd.hAdd (HMul.hMul 2 (abs ξ)) 1) ↑M
      hM : (setOf fun q => LT.lt (abs (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (H …
      m : Int
      hm : (setOf fun q => Eq (HSub.hSub (HPow.hPow q.num 2) (HMul.hMul d (HPow.hPow …
      this : NeZero m.natAbs
      f : Rat → Prod (ZMod m.natAbs) (ZMod m.natAbs) := fun q => { fst := ↑q.num, sn …
      q₁ : Rat
      h₁ : Eq (HSub.hSub (HPow.hPow q₁.num 2) (HMul.hMul d (HPow.hPow (↑q₁.den) 2))) m
      q₂ : Rat
      h₂ : Eq (HSub.hSub (HPow.hPow q₂.num 2) (HMul.hMul d (HPow.hPow (↑q₂.den) 2))) m
      hne : Ne q₁ q₂
      hqf : Eq (f q₁) (f q₂)
      hq1 : Eq ↑q₁.num ↑q₂.num
      hq2 : Eq ↑q₁.den ↑q₂.den
      hd₁ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num q₂.num) (HMul.hMul d (HMul.hMul ↑ …
      hd₂ : Dvd.dvd m (HSub.hSub (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.de …
      hm₀ : Ne (↑m) 0
      ⊢ Ne (HSub.hSub (HMul.hMul ↑q₁.num ↑q₂.den) (HMul.hMul ↑q₂.num ↑q₁.den)) 0
    -/
    exact mod_cast mt sub_eq_zero.mp (mt Rat.eq_iff_mul_eq_mul.mpr hne)
    /-
      🎉 no goals
    -/


/-- If `d` is a positive integer, then there is a nontrivial solution
to the Pell equation `x^2 - d*y^2 = 1` if and only if `d` is not a square. -/
theorem exists_iff_not_isSquare (h₀ : 0 < d) :
    (∃ x y : ℤ, x ^ 2 - d * y ^ 2 = 1 ∧ y ≠ 0) ↔ ¬IsSquare d := by
  /-
    d : Int
    h₀ : LT.lt 0 d
    ⊢ Iff (Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMu …
  -/
  refine ⟨?_, exists_of_not_isSquare h₀⟩
  /-
    d : Int
    h₀ : LT.lt 0 d
    ⊢ (Exists fun x => Exists fun y => And (Eq (HSub.hSub (HPow.hPow x 2) (HMul.hM …
  -/
  rintro ⟨x, y, hxy, hy⟩ ⟨a, rfl⟩
  /-
    case intro.intro.intro.intro
    x y : Int
    hy : Ne y 0
    a : Int
    h₀ : LT.lt 0 (HMul.hMul a a)
    hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (HMul.hMul a a) (HPow.hPow y 2) …
    ⊢ False
  -/
  rw [← sq, ← mul_pow, sq_sub_sq] at hxy
  /-
    case intro.intro.intro.intro
    x y : Int
    hy : Ne y 0
    a : Int
    h₀ : LT.lt 0 (HMul.hMul a a)
    hxy : Eq (HMul.hMul (HAdd.hAdd x (HMul.hMul a y)) (HSub.hSub x (HMul.hMul a y) …
    ⊢ False
  -/
  simpa [hy, mul_self_pos.mp h₀, sub_eq_add_neg, eq_neg_self_iff] using Int.eq_of_mul_eq_one hxy
  /-
    🎉 no goals
  -/


/-- If `d` is a positive integer that is not a square, then there exists a nontrivial solution
to the Pell equation `x^2 - d*y^2 = 1`. -/
theorem exists_nontrivial_of_not_isSquare (h₀ : 0 < d) (hd : ¬IsSquare d) :
    ∃ a : Solution₁ d, a ≠ 1 ∧ a ≠ -1 := by
  /-
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ⊢ Exists fun a => And (Ne a 1) (Ne a (-1))
  -/
  obtain ⟨x, y, prop, hy⟩ := exists_of_not_isSquare h₀ hd
  /-
    case intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    x y : Int
    prop : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul d (HPow.hPow y 2))) 1
    hy : Ne y 0
    ⊢ Exists fun a => And (Ne a 1) (Ne a (-1))
  -/
  refine ⟨mk x y prop, fun H => ?_, fun H => ?_⟩ <;> apply_fun Solution₁.y at H <;>
    /-
      case intro.intro.intro.refine_1
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      x y : Int
      prop : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul d (HPow.hPow y 2))) 1
      hy : Ne y 0
      H : Eq (Pell.Solution₁.mk x y prop).y (Pell.Solution₁.y 1)
      ⊢ False
    -/
    /-
      🎉 no goals
    -/
    simp [hy] at H
    /-
      🎉 no goals
    -/


/-- If `d` is a positive integer that is not a square, then there exists a solution
to the Pell equation `x^2 - d*y^2 = 1` with `x > 1` and `y > 0`. -/
theorem exists_pos_of_not_isSquare (h₀ : 0 < d) (hd : ¬IsSquare d) :
    ∃ a : Solution₁ d, 1 < a.x ∧ 0 < a.y := by
  /-
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ⊢ Exists fun a => And (LT.lt 1 a.x) (LT.lt 0 a.y)
  -/
  obtain ⟨x, y, h, hy⟩ := exists_of_not_isSquare h₀ hd
  /-
    case intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    x y : Int
    h : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul d (HPow.hPow y 2))) 1
    hy : Ne y 0
    ⊢ Exists fun a => And (LT.lt 1 a.x) (LT.lt 0 a.y)
  -/
  refine ⟨mk |x| |y| (by rwa [sq_abs, sq_abs]), ?_, abs_pos.mpr hy⟩
  /-
    case intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    x y : Int
    h : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul d (HPow.hPow y 2))) 1
    hy : Ne y 0
    ⊢ LT.lt 1 (Pell.Solution₁.mk (abs x) (abs y) ⋯).x
  -/
  rw [x_mk, ← one_lt_sq_iff_one_lt_abs, eq_add_of_sub_eq h, lt_add_iff_pos_right]
  /-
    case intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    x y : Int
    h : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul d (HPow.hPow y 2))) 1
    hy : Ne y 0
    ⊢ LT.lt 0 (HMul.hMul d (HPow.hPow y 2))
  -/
  exact mul_pos h₀ (sq_pos_of_ne_zero hy)
  /-
    🎉 no goals
  -/


/-- We define a solution to be *fundamental* if it has `x > 1` and `y > 0`
and its `x` is the smallest possible among solutions with `x > 1`. -/
def IsFundamental (a : Solution₁ d) : Prop :=
  1 < a.x ∧ 0 < a.y ∧ ∀ {b : Solution₁ d}, 1 < b.x → a.x ≤ b.x


/-- A fundamental solution has positive `x`. -/
theorem x_pos {a : Solution₁ d} (h : IsFundamental a) : 0 < a.x :=
  zero_lt_one.trans h.1


/-- If a fundamental solution exists, then `d` must be positive. -/
theorem d_pos {a : Solution₁ d} (h : IsFundamental a) : 0 < d :=
  d_pos_of_one_lt_x h.1


/-- If a fundamental solution exists, then `d` must be a non-square. -/
theorem d_nonsquare {a : Solution₁ d} (h : IsFundamental a) : ¬IsSquare d :=
  d_nonsquare_of_one_lt_x h.1


/-- If there is a fundamental solution, it is unique. -/
theorem subsingleton {a b : Solution₁ d} (ha : IsFundamental a) (hb : IsFundamental b) : a = b := by
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : Pell.IsFundamental a
    hb : Pell.IsFundamental b
    ⊢ Eq a b
  -/
  have hx := le_antisymm (ha.2.2 hb.1) (hb.2.2 ha.1)
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : Pell.IsFundamental a
    hb : Pell.IsFundamental b
    hx : Eq a.x b.x
    ⊢ Eq a b
  -/
  refine Solution₁.ext hx ?_
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : Pell.IsFundamental a
    hb : Pell.IsFundamental b
    hx : Eq a.x b.x
    ⊢ Eq a.y b.y
  -/
  have : d * a.y ^ 2 = d * b.y ^ 2 := by rw [a.prop_y, b.prop_y, hx]
  /-
    d : Int
    a b : Pell.Solution₁ d
    ha : Pell.IsFundamental a
    hb : Pell.IsFundamental b
    hx : Eq a.x b.x
    this : Eq (HMul.hMul d (HPow.hPow a.y 2)) (HMul.hMul d (HPow.hPow b.y 2))
    ⊢ Eq a.y b.y
  -/
  exact (sq_eq_sq₀ ha.2.1.le hb.2.1.le).mp (Int.eq_of_mul_eq_mul_left ha.d_pos.ne' this)
  /-
    🎉 no goals
  -/


/-- If `d` is positive and not a square, then a fundamental solution exists. -/
theorem exists_of_not_isSquare (h₀ : 0 < d) (hd : ¬IsSquare d) :
    ∃ a : Solution₁ d, IsFundamental a := by
  /-
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ⊢ Exists fun a => Pell.IsFundamental a
  -/
  obtain ⟨a, ha₁, ha₂⟩ := exists_pos_of_not_isSquare h₀ hd
  -- convert to `x : ℕ` to be able to use `Nat.find`
  have P : ∃ x' : ℕ, 1 < x' ∧ ∃ y' : ℤ, 0 < y' ∧ (x' : ℤ) ^ 2 - d * y' ^ 2 = 1 := by
    have hax := a.prop
    lift a.x to ℕ using by positivity with ax
    norm_cast at ha₁
    exact ⟨ax, ha₁, a.y, ha₂, hax⟩
  classical
  -- to avoid having to show that the predicate is decidable
  let x₁ := Nat.find P
  obtain ⟨hx, y₁, hy₀, hy₁⟩ := Nat.find_spec P
  refine ⟨mk x₁ y₁ hy₁, by rw [x_mk]; exact mod_cast hx, hy₀, fun {b} hb => ?_⟩
  rw [x_mk]
  have hb' := (Int.toNat_of_nonneg <| zero_le_one.trans hb.le).symm
  have hb'' := hb
  rw [hb'] at hb ⊢
  norm_cast at hb ⊢
  refine Nat.find_min' P ⟨hb, |b.y|, abs_pos.mpr <| y_ne_zero_of_one_lt_x hb'', ?_⟩
  rw [← hb', sq_abs]
  exact b.prop


/-- The map sending an integer `n` to the `y`-coordinate of `a^n` for a fundamental
solution `a` is stritcly increasing. -/
theorem y_strictMono {a : Solution₁ d} (h : IsFundamental a) :
    StrictMono fun n : ℤ => (a ^ n).y := by
  have H : ∀ n : ℤ, 0 ≤ n → (a ^ n).y < (a ^ (n + 1)).y := by
    intro n hn
    rw [← sub_pos, zpow_add, zpow_one, y_mul, add_sub_assoc]
    rw [show (a ^ n).y * a.x - (a ^ n).y = (a ^ n).y * (a.x - 1) by ring]
    refine
      add_pos_of_pos_of_nonneg (mul_pos (x_zpow_pos h.x_pos _) h.2.1)
        (mul_nonneg ?_ (by rw [sub_nonneg]; exact h.1.le))
    rcases hn.eq_or_lt with (rfl | hn)
    · simp only [zpow_zero, y_one, le_refl]
    · exact (y_zpow_pos h.x_pos h.2.1 hn).le
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
    ⊢ StrictMono fun n => (HPow.hPow a n).y
  -/
  refine strictMono_int_of_lt_succ fun n => ?_
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
    n : Int
    ⊢ LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n 1)).y
  -/
  rcases le_or_lt 0 n with hn | hn
    /-
      case inl
      d : Int
      a : Pell.Solution₁ d
      h : Pell.IsFundamental a
      H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
      n : Int
      hn : LE.le 0 n
      ⊢ LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n 1)).y
    -/
  · exact H n hn
    /-
      🎉 no goals
    -/
    /-
      case inr
      d : Int
      a : Pell.Solution₁ d
      h : Pell.IsFundamental a
      H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
      n : Int
      hn : LT.lt n 0
      ⊢ LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n 1)).y
    -/
  · let m : ℤ := -n - 1
    /-
      case inr
      d : Int
      a : Pell.Solution₁ d
      h : Pell.IsFundamental a
      H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
      n : Int
      hn : LT.lt n 0
      m : Int := HSub.hSub (Neg.neg n) 1
      ⊢ LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n 1)).y
    -/
    have hm : n = -m - 1 := by simp only [m, neg_sub, sub_neg_eq_add, add_tsub_cancel_left]
    /-
      case inr
      d : Int
      a : Pell.Solution₁ d
      h : Pell.IsFundamental a
      H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
      n : Int
      hn : LT.lt n 0
      m : Int := HSub.hSub (Neg.neg n) 1
      hm : Eq n (HSub.hSub (Neg.neg m) 1)
      ⊢ LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n 1)).y
    -/
    rw [hm, sub_add_cancel, ← neg_add', zpow_neg, zpow_neg, y_inv, y_inv, neg_lt_neg_iff]
    /-
      case inr
      d : Int
      a : Pell.Solution₁ d
      h : Pell.IsFundamental a
      H : ∀ (n : Int), LE.le 0 n → LT.lt (HPow.hPow a n).y (HPow.hPow a (HAdd.hAdd n …
      n : Int
      hn : LT.lt n 0
      m : Int := HSub.hSub (Neg.neg n) 1
      hm : Eq n (HSub.hSub (Neg.neg m) 1)
      ⊢ LT.lt (HPow.hPow a m).y (HPow.hPow a (HAdd.hAdd m 1)).y
    -/
    exact H _ (by omega)
    /-
      🎉 no goals
    -/


/-- If `a` is a fundamental solution, then `(a^m).y < (a^n).y` if and only if `m < n`. -/
theorem zpow_y_lt_iff_lt {a : Solution₁ d} (h : IsFundamental a) (m n : ℤ) :
    (a ^ m).y < (a ^ n).y ↔ m < n := by
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    m n : Int
    ⊢ Iff (LT.lt (HPow.hPow a m).y (HPow.hPow a n).y) (LT.lt m n)
  -/
  refine ⟨fun H => ?_, fun H => h.y_strictMono H⟩
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    m n : Int
    H : LT.lt (HPow.hPow a m).y (HPow.hPow a n).y
    ⊢ LT.lt m n
  -/
  contrapose! H
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    m n : Int
    H : LE.le n m
    ⊢ LE.le (HPow.hPow a n).y (HPow.hPow a m).y
  -/
  exact h.y_strictMono.monotone H
  /-
    🎉 no goals
  -/


/-- The `n`th power of a fundamental solution is trivial if and only if `n = 0`. -/
theorem zpow_eq_one_iff {a : Solution₁ d} (h : IsFundamental a) (n : ℤ) : a ^ n = 1 ↔ n = 0 := by
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n : Int
    ⊢ Iff (Eq (HPow.hPow a n) 1) (Eq n 0)
  -/
  rw [← zpow_zero a]
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n : Int
    ⊢ Iff (Eq (HPow.hPow a n) (HPow.hPow a 0)) (Eq n 0)
  -/
  exact ⟨fun H => h.y_strictMono.injective (congr_arg Solution₁.y H), fun H => H ▸ rfl⟩
  /-
    🎉 no goals
  -/


/-- A power of a fundamental solution is never equal to the negative of a power of this
fundamental solution. -/
theorem zpow_ne_neg_zpow {a : Solution₁ d} (h : IsFundamental a) {n n' : ℤ} : a ^ n ≠ -a ^ n' := by
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n n' : Int
    ⊢ Ne (HPow.hPow a n) (Neg.neg (HPow.hPow a n'))
  -/
  intro hf
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n n' : Int
    hf : Eq (HPow.hPow a n) (Neg.neg (HPow.hPow a n'))
    ⊢ False
  -/
  apply_fun Solution₁.x at hf
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n n' : Int
    hf : Eq (HPow.hPow a n).x (Neg.neg (HPow.hPow a n')).x
    ⊢ False
  -/
  have H := x_zpow_pos h.x_pos n
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n n' : Int
    hf : Eq (HPow.hPow a n).x (Neg.neg (HPow.hPow a n')).x
    H : LT.lt 0 (HPow.hPow a n).x
    ⊢ False
  -/
  rw [hf, x_neg, lt_neg, neg_zero] at H
  /-
    d : Int
    a : Pell.Solution₁ d
    h : Pell.IsFundamental a
    n n' : Int
    hf : Eq (HPow.hPow a n).x (Neg.neg (HPow.hPow a n')).x
    H : LT.lt (HPow.hPow a n').x 0
    ⊢ False
  -/
  exact lt_irrefl _ ((x_zpow_pos h.x_pos n').trans H)
  /-
    🎉 no goals
  -/


/-- The `x`-coordinate of a fundamental solution is a lower bound for the `x`-coordinate
of any positive solution. -/
theorem x_le_x {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d} (hax : 1 < a.x) :
    a₁.x ≤ a.x :=
  h.2.2 hax


/-- The `y`-coordinate of a fundamental solution is a lower bound for the `y`-coordinate
of any positive solution. -/
theorem y_le_y {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d} (hax : 1 < a.x)
    (hay : 0 < a.y) : a₁.y ≤ a.y := by
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LE.le a₁.y a.y
  -/
  have H : d * (a₁.y ^ 2 - a.y ^ 2) = a₁.x ^ 2 - a.x ^ 2 := by rw [a.prop_x, a₁.prop_x]; ring
  rw [← abs_of_pos hay, ← abs_of_pos h.2.1, ← sq_le_sq, ← mul_le_mul_left h.d_pos, ← sub_nonpos, ←
    mul_sub, H, sub_nonpos, sq_le_sq, abs_of_pos (zero_lt_one.trans h.1),
    abs_of_pos (zero_lt_one.trans hax)]
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    H : Eq (HMul.hMul d (HSub.hSub (HPow.hPow a₁.y 2) (HPow.hPow a.y 2))) (HSub.hS …
    ⊢ LE.le a₁.x a.x
  -/
  exact h.x_le_x hax
  /-
    🎉 no goals
  -/

-- helper lemma for the next three results

theorem x_mul_y_le_y_mul_x {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d}
    (hax : 1 < a.x) (hay : 0 < a.y) : a.x * a₁.y ≤ a.y * a₁.x := by
  rw [← abs_of_pos <| zero_lt_one.trans hax, ← abs_of_pos hay, ← abs_of_pos h.x_pos, ←
    abs_of_pos h.2.1, ← abs_mul, ← abs_mul, ← sq_le_sq, mul_pow, mul_pow, a.prop_x, a₁.prop_x, ←
    sub_nonneg]
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LE.le 0 (HSub.hSub (HMul.hMul (HPow.hPow a.y 2) (HAdd.hAdd 1 (HMul.hMul d (H …
  -/
  ring_nf
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LE.le 0 (HSub.hSub (HPow.hPow a.y 2) (HPow.hPow a₁.y 2))
  -/
  rw [sub_nonneg, sq_le_sq, abs_of_pos hay, abs_of_pos h.2.1]
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LE.le a₁.y a.y
  -/
  exact h.y_le_y hax hay
  /-
    🎉 no goals
  -/


/-- If we multiply a positive solution with the inverse of a fundamental solution,
the `y`-coordinate remains nonnegative. -/
theorem mul_inv_y_nonneg {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d} (hax : 1 < a.x)
    (hay : 0 < a.y) : 0 ≤ (a * a₁⁻¹).y := by
  simpa only [y_inv, mul_neg, y_mul, le_neg_add_iff_add_le, add_zero] using
    h.x_mul_y_le_y_mul_x hax hay


/-- If we multiply a positive solution with the inverse of a fundamental solution,
the `x`-coordinate stays positive. -/
theorem mul_inv_x_pos {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d} (hax : 1 < a.x)
    (hay : 0 < a.y) : 0 < (a * a₁⁻¹).x := by
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt 0 (HMul.hMul a (Inv.inv a₁)).x
  -/
  simp only [x_mul, x_inv, y_inv, mul_neg, lt_add_neg_iff_add_lt, zero_add]
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul d (HMul.hMul a.y a₁.y)) (HMul.hMul a.x a₁.x)
  -/
  refine (mul_lt_mul_left <| zero_lt_one.trans hax).mp ?_
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul a.x (HMul.hMul d (HMul.hMul a.y a₁.y))) (HMul.hMul a.x (HMu …
  -/
  rw [(by ring : a.x * (d * (a.y * a₁.y)) = d * a.y * (a.x * a₁.y))]
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul (HMul.hMul d a.y) (HMul.hMul a.x a₁.y)) (HMul.hMul a.x (HMu …
  -/
  refine ((mul_le_mul_left <| mul_pos h.d_pos hay).mpr <| x_mul_y_le_y_mul_x h hax hay).trans_lt ?_
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul (HMul.hMul d a.y) (HMul.hMul a.y a₁.x)) (HMul.hMul a.x (HMu …
  -/
  rw [← mul_assoc, mul_assoc d, ← sq, a.prop_y, ← sub_pos]
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt 0 (HSub.hSub (HMul.hMul a.x (HMul.hMul a.x a₁.x)) (HMul.hMul (HSub.hSu …
  -/
  ring_nf
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt 0 a₁.x
  -/
  exact zero_lt_one.trans h.1
  /-
    🎉 no goals
  -/


/-- If we multiply a positive solution with the inverse of a fundamental solution,
the `x`-coordinate decreases. -/
theorem mul_inv_x_lt_x {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d} (hax : 1 < a.x)
    (hay : 0 < a.y) : (a * a₁⁻¹).x < a.x := by
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul a (Inv.inv a₁)).x a.x
  -/
  simp only [x_mul, x_inv, y_inv, mul_neg, add_neg_lt_iff_le_add']
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul a.x a₁.x) (HAdd.hAdd (HMul.hMul d (HMul.hMul a.y a₁.y)) a.x)
  -/
  refine (mul_lt_mul_left h.2.1).mp ?_
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul a₁.y (HMul.hMul a.x a₁.x)) (HMul.hMul a₁.y (HAdd.hAdd (HMul …
  -/
  rw [(by ring : a₁.y * (a.x * a₁.x) = a.x * a₁.y * a₁.x)]
  refine
    ((mul_le_mul_right <| zero_lt_one.trans h.1).mpr <| x_mul_y_le_y_mul_x h hax hay).trans_lt ?_
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HMul.hMul (HMul.hMul a.y a₁.x) a₁.x) (HMul.hMul a₁.y (HAdd.hAdd (HMul …
  -/
  rw [mul_assoc, ← sq, a₁.prop_x, ← sub_neg]
  -- Porting note: was `ring_nf`
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 1 a.x
    hay : LT.lt 0 a.y
    ⊢ LT.lt (HSub.hSub (HMul.hMul a.y (HAdd.hAdd 1 (HMul.hMul d (HPow.hPow a₁.y 2) …
  -/
  suffices a.y - a.x * a₁.y < 0 by convert this using 1; ring
  rw [sub_neg, ← abs_of_pos hay, ← abs_of_pos h.2.1, ← abs_of_pos <| zero_lt_one.trans hax, ←
    abs_mul, ← sq_lt_sq, mul_pow, a.prop_x]
  calc
    a.y ^ 2 = 1 * a.y ^ 2 := (one_mul _).symm
    _ ≤ d * a.y ^ 2 := (mul_le_mul_right <| sq_pos_of_pos hay).mpr h.d_pos
    _ < d * a.y ^ 2 + 1 := lt_add_one _
    _ = (1 + d * a.y ^ 2) * 1 := by rw [add_comm, mul_one]
    _ ≤ (1 + d * a.y ^ 2) * a₁.y ^ 2 :=
      (mul_le_mul_left (by have := h.d_pos; positivity)).mpr (sq_pos_of_pos h.2.1)


/-- Any nonnegative solution is a power with nonnegative exponent of a fundamental solution. -/
theorem eq_pow_of_nonneg {a₁ : Solution₁ d} (h : IsFundamental a₁) {a : Solution₁ d} (hax : 0 < a.x)
    (hay : 0 ≤ a.y) : ∃ n : ℕ, a = a₁ ^ n := by
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hax : LT.lt 0 a.x
    hay : LE.le 0 a.y
    ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
  -/
  lift a.x to ℕ using hax.le with ax hax'
  -- Porting note: added
  /-
    case intro
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hay : LE.le 0 a.y
    ax : Nat
    hax' : Eq (↑ax) a.x
    hax✝ hax : LT.lt 0 ↑ax
    ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
  -/
  clear hax
  /-
    case intro
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    hay : LE.le 0 a.y
    ax : Nat
    hax' : Eq (↑ax) a.x
    hax : LT.lt 0 ↑ax
    ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
  -/
  induction' ax using Nat.strong_induction_on with x ih generalizing a
  /-
    case intro.h
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    x : Nat
    ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
    a : Pell.Solution₁ d
    hay : LE.le 0 a.y
    hax' : Eq (↑x) a.x
    hax : LT.lt 0 ↑x
    ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
  -/
  rcases hay.eq_or_lt with hy | hy
  · -- case 1: `a = 1`
    /-
      case intro.h.inl
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : Eq 0 a.y
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    refine ⟨0, ?_⟩
    /-
      case intro.h.inl
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : Eq 0 a.y
      ⊢ Eq a (HPow.hPow a₁ 0)
    -/
    simp only [pow_zero]
    /-
      case intro.h.inl
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : Eq 0 a.y
      ⊢ Eq a 1
    -/
    ext <;> simp only [x_one, y_one]
      /-
        case intro.h.inl.hx
        d : Int
        a₁ : Pell.Solution₁ d
        h : Pell.IsFundamental a₁
        x : Nat
        ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
        a : Pell.Solution₁ d
        hay : LE.le 0 a.y
        hax' : Eq (↑x) a.x
        hax : LT.lt 0 ↑x
        hy : Eq 0 a.y
        ⊢ Eq a.x 1
      -/
    · have prop := a.prop
      rw [← hy, sq (0 : ℤ), zero_mul, mul_zero, sub_zero,
        sq_eq_one_iff] at prop
      /-
        case intro.h.inl.hx
        d : Int
        a₁ : Pell.Solution₁ d
        h : Pell.IsFundamental a₁
        x : Nat
        ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
        a : Pell.Solution₁ d
        hay : LE.le 0 a.y
        hax' : Eq (↑x) a.x
        hax : LT.lt 0 ↑x
        hy : Eq 0 a.y
        prop : Or (Eq a.x 1) (Eq a.x (-1))
        ⊢ Eq a.x 1
      -/
      refine prop.resolve_right fun hf => ?_
      /-
        case intro.h.inl.hx
        d : Int
        a₁ : Pell.Solution₁ d
        h : Pell.IsFundamental a₁
        x : Nat
        ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
        a : Pell.Solution₁ d
        hay : LE.le 0 a.y
        hax' : Eq (↑x) a.x
        hax : LT.lt 0 ↑x
        hy : Eq 0 a.y
        prop : Or (Eq a.x 1) (Eq a.x (-1))
        hf : Eq a.x (-1)
        ⊢ False
      -/
      have := (hax.trans_eq hax').le.trans_eq hf
      /-
        case intro.h.inl.hx
        d : Int
        a₁ : Pell.Solution₁ d
        h : Pell.IsFundamental a₁
        x : Nat
        ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
        a : Pell.Solution₁ d
        hay : LE.le 0 a.y
        hax' : Eq (↑x) a.x
        hax : LT.lt 0 ↑x
        hy : Eq 0 a.y
        prop : Or (Eq a.x 1) (Eq a.x (-1))
        hf : Eq a.x (-1)
        this : LE.le 0 (-1)
        ⊢ False
      -/
      norm_num at this
      /-
        🎉 no goals
      -/
      /-
        case intro.h.inl.hy
        d : Int
        a₁ : Pell.Solution₁ d
        h : Pell.IsFundamental a₁
        x : Nat
        ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
        a : Pell.Solution₁ d
        hay : LE.le 0 a.y
        hax' : Eq (↑x) a.x
        hax : LT.lt 0 ↑x
        hy : Eq 0 a.y
        ⊢ Eq a.y 0
      -/
    · exact hy.symm
      /-
        🎉 no goals
      -/
  · -- case 2: `a ≥ a₁`
    /-
      case intro.h.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    have hx₁ : 1 < a.x := by nlinarith [a.prop, h.d_pos]
    /-
      case intro.h.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      hx₁ : LT.lt 1 a.x
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    have hxx₁ := h.mul_inv_x_pos hx₁ hy
    /-
      case intro.h.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      hx₁ : LT.lt 1 a.x
      hxx₁ : LT.lt 0 (HMul.hMul a (Inv.inv a₁)).x
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    have hxx₂ := h.mul_inv_x_lt_x hx₁ hy
    /-
      case intro.h.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      hx₁ : LT.lt 1 a.x
      hxx₁ : LT.lt 0 (HMul.hMul a (Inv.inv a₁)).x
      hxx₂ : LT.lt (HMul.hMul a (Inv.inv a₁)).x a.x
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    have hyy := h.mul_inv_y_nonneg hx₁ hy
    /-
      case intro.h.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      hx₁ : LT.lt 1 a.x
      hxx₁ : LT.lt 0 (HMul.hMul a (Inv.inv a₁)).x
      hxx₂ : LT.lt (HMul.hMul a (Inv.inv a₁)).x a.x
      hyy : LE.le 0 (HMul.hMul a (Inv.inv a₁)).y
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    lift (a * a₁⁻¹).x to ℕ using hxx₁.le with x' hx'
    -- Porting note: `ih` has its arguments in a different order compared to lean 3.
    /-
      case intro.h.inr.intro
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      hx₁ : LT.lt 1 a.x
      hyy : LE.le 0 (HMul.hMul a (Inv.inv a₁)).y
      x' : Nat
      hx' : Eq (↑x') (HMul.hMul a (Inv.inv a₁)).x
      hxx₁✝ hxx₁ : LT.lt 0 ↑x'
      hxx₂✝ hxx₂ : LT.lt (↑x') a.x
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    obtain ⟨n, hn⟩ := ih x' (mod_cast hxx₂.trans_eq hax'.symm) hyy hx' hxx₁
    /-
      case intro.h.inr.intro.intro
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {a : Pell.Solution₁ d}, LE.le 0 a.y → Eq (↑m)  …
      a : Pell.Solution₁ d
      hay : LE.le 0 a.y
      hax' : Eq (↑x) a.x
      hax : LT.lt 0 ↑x
      hy : LT.lt 0 a.y
      hx₁ : LT.lt 1 a.x
      hyy : LE.le 0 (HMul.hMul a (Inv.inv a₁)).y
      x' : Nat
      hx' : Eq (↑x') (HMul.hMul a (Inv.inv a₁)).x
      hxx₁✝ hxx₁ : LT.lt 0 ↑x'
      hxx₂✝ hxx₂ : LT.lt (↑x') a.x
      n : Nat
      hn : Eq (HMul.hMul a (Inv.inv a₁)) (HPow.hPow a₁ n)
      ⊢ Exists fun n => Eq a (HPow.hPow a₁ n)
    -/
    exact ⟨n + 1, by rw [pow_succ', ← hn, mul_comm a, ← mul_assoc, mul_inv_cancel, one_mul]⟩
    /-
      🎉 no goals
    -/


/-- Every solution is, up to a sign, a power of a given fundamental solution. -/
theorem eq_zpow_or_neg_zpow {a₁ : Solution₁ d} (h : IsFundamental a₁) (a : Solution₁ d) :
    ∃ n : ℤ, a = a₁ ^ n ∨ a = -a₁ ^ n := by
  /-
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    ⊢ Exists fun n => Or (Eq a (HPow.hPow a₁ n)) (Eq a (Neg.neg (HPow.hPow a₁ n)))
  -/
  obtain ⟨b, hbx, hby, hb⟩ := exists_pos_variant h.d_pos a
  /-
    case intro.intro.intro
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a b : Pell.Solution₁ d
    hbx : LT.lt 0 b.x
    hby : LE.le 0 b.y
    hb : Membership.mem (Insert.insert b (Insert.insert (Inv.inv b) (Insert.insert …
    ⊢ Exists fun n => Or (Eq a (HPow.hPow a₁ n)) (Eq a (Neg.neg (HPow.hPow a₁ n)))
  -/
  obtain ⟨n, hn⟩ := h.eq_pow_of_nonneg hbx hby
  /-
    case intro.intro.intro.intro
    d : Int
    a₁ : Pell.Solution₁ d
    h : Pell.IsFundamental a₁
    a b : Pell.Solution₁ d
    hbx : LT.lt 0 b.x
    hby : LE.le 0 b.y
    hb : Membership.mem (Insert.insert b (Insert.insert (Inv.inv b) (Insert.insert …
    n : Nat
    hn : Eq b (HPow.hPow a₁ n)
    ⊢ Exists fun n => Or (Eq a (HPow.hPow a₁ n)) (Eq a (Neg.neg (HPow.hPow a₁ n)))
  -/
  rcases hb with (rfl | rfl | rfl | hb)
    /-
      case intro.intro.intro.intro.inl
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      a : Pell.Solution₁ d
      n : Nat
      hbx : LT.lt 0 a.x
      hby : LE.le 0 a.y
      hn : Eq a (HPow.hPow a₁ n)
      ⊢ Exists fun n => Or (Eq a (HPow.hPow a₁ n)) (Eq a (Neg.neg (HPow.hPow a₁ n)))
    -/
  · exact ⟨n, Or.inl (mod_cast hn)⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inl
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      b : Pell.Solution₁ d
      hbx : LT.lt 0 b.x
      hby : LE.le 0 b.y
      n : Nat
      hn : Eq b (HPow.hPow a₁ n)
      ⊢ Exists fun n => Or (Eq (Inv.inv b) (HPow.hPow a₁ n)) (Eq (Inv.inv b) (Neg.ne …
    -/
  · exact ⟨-n, Or.inl (by simp [hn])⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inr.inl
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      b : Pell.Solution₁ d
      hbx : LT.lt 0 b.x
      hby : LE.le 0 b.y
      n : Nat
      hn : Eq b (HPow.hPow a₁ n)
      ⊢ Exists fun n => Or (Eq (Neg.neg b) (HPow.hPow a₁ n)) (Eq (Neg.neg b) (Neg.ne …
    -/
  · exact ⟨n, Or.inr (by simp [hn])⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inr.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      a b : Pell.Solution₁ d
      hbx : LT.lt 0 b.x
      hby : LE.le 0 b.y
      n : Nat
      hn : Eq b (HPow.hPow a₁ n)
      hb : Membership.mem (Singleton.singleton (Neg.neg (Inv.inv b))) a
      ⊢ Exists fun n => Or (Eq a (HPow.hPow a₁ n)) (Eq a (Neg.neg (HPow.hPow a₁ n)))
    -/
  · rw [Set.mem_singleton_iff] at hb
    /-
      case intro.intro.intro.intro.inr.inr.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      a b : Pell.Solution₁ d
      hbx : LT.lt 0 b.x
      hby : LE.le 0 b.y
      n : Nat
      hn : Eq b (HPow.hPow a₁ n)
      hb : Eq a (Neg.neg (Inv.inv b))
      ⊢ Exists fun n => Or (Eq a (HPow.hPow a₁ n)) (Eq a (Neg.neg (HPow.hPow a₁ n)))
    -/
    rw [hb]
    /-
      case intro.intro.intro.intro.inr.inr.inr
      d : Int
      a₁ : Pell.Solution₁ d
      h : Pell.IsFundamental a₁
      a b : Pell.Solution₁ d
      hbx : LT.lt 0 b.x
      hby : LE.le 0 b.y
      n : Nat
      hn : Eq b (HPow.hPow a₁ n)
      hb : Eq a (Neg.neg (Inv.inv b))
      ⊢ Exists fun n => Or (Eq (Neg.neg (Inv.inv b)) (HPow.hPow a₁ n)) (Eq (Neg.neg  …
    -/
    exact ⟨-n, Or.inr (by simp [hn])⟩
    /-
      🎉 no goals
    -/


/-- When `d` is positive and not a square, then the group of solutions to the Pell equation
`x^2 - d*y^2 = 1` has a unique positive generator (up to sign). -/
theorem existsUnique_pos_generator (h₀ : 0 < d) (hd : ¬IsSquare d) :
    ∃! a₁ : Solution₁ d,
      1 < a₁.x ∧ 0 < a₁.y ∧ ∀ a : Solution₁ d, ∃ n : ℤ, a = a₁ ^ n ∨ a = -a₁ ^ n := by
  /-
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ⊢ ExistsUnique fun a₁ => And (LT.lt 1 a₁.x) (And (LT.lt 0 a₁.y) (∀ (a : Pell.S …
  -/
  obtain ⟨a₁, ha₁⟩ := IsFundamental.exists_of_not_isSquare h₀ hd
  /-
    case intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    ⊢ ExistsUnique fun a₁ => And (LT.lt 1 a₁.x) (And (LT.lt 0 a₁.y) (∀ (a : Pell.S …
  -/
  refine ⟨a₁, ⟨ha₁.1, ha₁.2.1, ha₁.eq_zpow_or_neg_zpow⟩, fun a (H : 1 < _ ∧ _) => ?_⟩
  /-
    case intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    H : And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (a_1 : Pell.Solution₁ d), Exists f …
    ⊢ Eq a a₁
  -/
  obtain ⟨Hx, Hy, H⟩ := H
  /-
    case intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    Hx : LT.lt 1 a.x
    Hy : LT.lt 0 a.y
    H : ∀ (a_1 : Pell.Solution₁ d), Exists fun n => Or (Eq a_1 (HPow.hPow a n)) (E …
    ⊢ Eq a a₁
  -/
  obtain ⟨n₁, hn₁⟩ := H a₁
  /-
    case intro.intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    Hx : LT.lt 1 a.x
    Hy : LT.lt 0 a.y
    H : ∀ (a_1 : Pell.Solution₁ d), Exists fun n => Or (Eq a_1 (HPow.hPow a n)) (E …
    n₁ : Int
    hn₁ : Or (Eq a₁ (HPow.hPow a n₁)) (Eq a₁ (Neg.neg (HPow.hPow a n₁)))
    ⊢ Eq a a₁
  -/
  obtain ⟨n₂, hn₂⟩ := ha₁.eq_zpow_or_neg_zpow a
  /-
    case intro.intro.intro.intro.intro
    d : Int
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    a : Pell.Solution₁ d
    Hx : LT.lt 1 a.x
    Hy : LT.lt 0 a.y
    H : ∀ (a_1 : Pell.Solution₁ d), Exists fun n => Or (Eq a_1 (HPow.hPow a n)) (E …
    n₁ : Int
    hn₁ : Or (Eq a₁ (HPow.hPow a n₁)) (Eq a₁ (Neg.neg (HPow.hPow a n₁)))
    n₂ : Int
    hn₂ : Or (Eq a (HPow.hPow a₁ n₂)) (Eq a (Neg.neg (HPow.hPow a₁ n₂)))
    ⊢ Eq a a₁
  -/
  rcases hn₂ with (rfl | rfl)
  · rw [← zpow_mul, eq_comm, @eq_comm _ a₁, ← mul_inv_eq_one, ← @mul_inv_eq_one _ _ _ a₁, ←
      zpow_neg_one, neg_mul, ← zpow_add, ← sub_eq_add_neg] at hn₁
    /-
      case intro.intro.intro.intro.intro.inl
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      a₁ : Pell.Solution₁ d
      ha₁ : Pell.IsFundamental a₁
      n₁ n₂ : Int
      Hx : LT.lt 1 (HPow.hPow a₁ n₂).x
      Hy : LT.lt 0 (HPow.hPow a₁ n₂).y
      H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (HPow.hPow a …
      hn₁ : Or (Eq (HPow.hPow a₁ (HSub.hSub (HMul.hMul n₂ n₁) 1)) 1) (Eq (Neg.neg (H …
      ⊢ Eq (HPow.hPow a₁ n₂) a₁
    -/
    cases' hn₁ with hn₁ hn₁
    · rcases Int.isUnit_iff.mp
          (isUnit_of_mul_eq_one _ _ <|
            sub_eq_zero.mp <| (ha₁.zpow_eq_one_iff (n₂ * n₁ - 1)).mp hn₁) with
        (rfl | rfl)
        /-
          case intro.intro.intro.intro.intro.inl.inl.inl
          d : Int
          h₀ : LT.lt 0 d
          hd : Not (IsSquare d)
          a₁ : Pell.Solution₁ d
          ha₁ : Pell.IsFundamental a₁
          n₁ : Int
          Hx : LT.lt 1 (HPow.hPow a₁ 1).x
          Hy : LT.lt 0 (HPow.hPow a₁ 1).y
          H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (HPow.hPow a …
          hn₁ : Eq (HPow.hPow a₁ (HSub.hSub (HMul.hMul 1 n₁) 1)) 1
          ⊢ Eq (HPow.hPow a₁ 1) a₁
        -/
      · rw [zpow_one]
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.intro.inl.inl.inr
          d : Int
          h₀ : LT.lt 0 d
          hd : Not (IsSquare d)
          a₁ : Pell.Solution₁ d
          ha₁ : Pell.IsFundamental a₁
          n₁ : Int
          Hx : LT.lt 1 (HPow.hPow a₁ (-1)).x
          Hy : LT.lt 0 (HPow.hPow a₁ (-1)).y
          H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (HPow.hPow a …
          hn₁ : Eq (HPow.hPow a₁ (HSub.hSub (HMul.hMul (-1) n₁) 1)) 1
          ⊢ Eq (HPow.hPow a₁ (-1)) a₁
        -/
      · rw [zpow_neg_one, y_inv, lt_neg, neg_zero] at Hy
        /-
          case intro.intro.intro.intro.intro.inl.inl.inr
          d : Int
          h₀ : LT.lt 0 d
          hd : Not (IsSquare d)
          a₁ : Pell.Solution₁ d
          ha₁ : Pell.IsFundamental a₁
          n₁ : Int
          Hx : LT.lt 1 (HPow.hPow a₁ (-1)).x
          Hy : LT.lt a₁.y 0
          H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (HPow.hPow a …
          hn₁ : Eq (HPow.hPow a₁ (HSub.hSub (HMul.hMul (-1) n₁) 1)) 1
          ⊢ Eq (HPow.hPow a₁ (-1)) a₁
        -/
        exact False.elim (lt_irrefl _ <| ha₁.2.1.trans Hy)
        /-
          🎉 no goals
        -/
      /-
        case intro.intro.intro.intro.intro.inl.inr
        d : Int
        h₀ : LT.lt 0 d
        hd : Not (IsSquare d)
        a₁ : Pell.Solution₁ d
        ha₁ : Pell.IsFundamental a₁
        n₁ n₂ : Int
        Hx : LT.lt 1 (HPow.hPow a₁ n₂).x
        Hy : LT.lt 0 (HPow.hPow a₁ n₂).y
        H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (HPow.hPow a …
        hn₁ : Eq (Neg.neg (HPow.hPow a₁ (HSub.hSub (HMul.hMul n₂ n₁) 1))) 1
        ⊢ Eq (HPow.hPow a₁ n₂) a₁
      -/
    · rw [← zpow_zero a₁, eq_comm] at hn₁
      /-
        case intro.intro.intro.intro.intro.inl.inr
        d : Int
        h₀ : LT.lt 0 d
        hd : Not (IsSquare d)
        a₁ : Pell.Solution₁ d
        ha₁ : Pell.IsFundamental a₁
        n₁ n₂ : Int
        Hx : LT.lt 1 (HPow.hPow a₁ n₂).x
        Hy : LT.lt 0 (HPow.hPow a₁ n₂).y
        H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (HPow.hPow a …
        hn₁ : Eq (HPow.hPow a₁ 0) (Neg.neg (HPow.hPow a₁ (HSub.hSub (HMul.hMul n₂ n₁)  …
        ⊢ Eq (HPow.hPow a₁ n₂) a₁
      -/
      exact False.elim (ha₁.zpow_ne_neg_zpow hn₁)
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.inr
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      a₁ : Pell.Solution₁ d
      ha₁ : Pell.IsFundamental a₁
      n₁ n₂ : Int
      Hx : LT.lt 1 (Neg.neg (HPow.hPow a₁ n₂)).x
      Hy : LT.lt 0 (Neg.neg (HPow.hPow a₁ n₂)).y
      H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (Neg.neg (HP …
      hn₁ : Or (Eq a₁ (HPow.hPow (Neg.neg (HPow.hPow a₁ n₂)) n₁)) (Eq a₁ (Neg.neg (H …
      ⊢ Eq (Neg.neg (HPow.hPow a₁ n₂)) a₁
    -/
  · rw [x_neg, lt_neg] at Hx
    /-
      case intro.intro.intro.intro.intro.inr
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      a₁ : Pell.Solution₁ d
      ha₁ : Pell.IsFundamental a₁
      n₁ n₂ : Int
      Hx : LT.lt (HPow.hPow a₁ n₂).x (-1)
      Hy : LT.lt 0 (Neg.neg (HPow.hPow a₁ n₂)).y
      H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (Neg.neg (HP …
      hn₁ : Or (Eq a₁ (HPow.hPow (Neg.neg (HPow.hPow a₁ n₂)) n₁)) (Eq a₁ (Neg.neg (H …
      ⊢ Eq (Neg.neg (HPow.hPow a₁ n₂)) a₁
    -/
    have := (x_zpow_pos (zero_lt_one.trans ha₁.1) n₂).trans Hx
    /-
      case intro.intro.intro.intro.intro.inr
      d : Int
      h₀ : LT.lt 0 d
      hd : Not (IsSquare d)
      a₁ : Pell.Solution₁ d
      ha₁ : Pell.IsFundamental a₁
      n₁ n₂ : Int
      Hx : LT.lt (HPow.hPow a₁ n₂).x (-1)
      Hy : LT.lt 0 (Neg.neg (HPow.hPow a₁ n₂)).y
      H : ∀ (a : Pell.Solution₁ d), Exists fun n => Or (Eq a (HPow.hPow (Neg.neg (HP …
      hn₁ : Or (Eq a₁ (HPow.hPow (Neg.neg (HPow.hPow a₁ n₂)) n₁)) (Eq a₁ (Neg.neg (H …
      this : LT.lt 0 (-1)
      ⊢ Eq (Neg.neg (HPow.hPow a₁ n₂)) a₁
    -/
    norm_num at this
    /-
      🎉 no goals
    -/


/-- A positive solution is a generator (up to sign) of the group of all solutions to the
Pell equation `x^2 - d*y^2 = 1` if and only if it is a fundamental solution. -/
theorem pos_generator_iff_fundamental (a : Solution₁ d) :
    (1 < a.x ∧ 0 < a.y ∧ ∀ b : Solution₁ d, ∃ n : ℤ, b = a ^ n ∨ b = -a ^ n) ↔ IsFundamental a := by
  /-
    d : Int
    a : Pell.Solution₁ d
    ⊢ Iff (And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (b : Pell.Solution₁ d), Exists  …
  -/
  refine ⟨fun h => ?_, fun H => ⟨H.1, H.2.1, H.eq_zpow_or_neg_zpow⟩⟩
  /-
    d : Int
    a : Pell.Solution₁ d
    h : And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (b : Pell.Solution₁ d), Exists fun …
    ⊢ Pell.IsFundamental a
  -/
  have h₀ := d_pos_of_one_lt_x h.1
  /-
    d : Int
    a : Pell.Solution₁ d
    h : And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (b : Pell.Solution₁ d), Exists fun …
    h₀ : LT.lt 0 d
    ⊢ Pell.IsFundamental a
  -/
  have hd := d_nonsquare_of_one_lt_x h.1
  /-
    d : Int
    a : Pell.Solution₁ d
    h : And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (b : Pell.Solution₁ d), Exists fun …
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    ⊢ Pell.IsFundamental a
  -/
  obtain ⟨a₁, ha₁⟩ := IsFundamental.exists_of_not_isSquare h₀ hd
  /-
    case intro
    d : Int
    a : Pell.Solution₁ d
    h : And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (b : Pell.Solution₁ d), Exists fun …
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    ⊢ Pell.IsFundamental a
  -/
  obtain ⟨b, -, hb₂⟩ := existsUnique_pos_generator h₀ hd
  /-
    case intro.intro.intro
    d : Int
    a : Pell.Solution₁ d
    h : And (LT.lt 1 a.x) (And (LT.lt 0 a.y) (∀ (b : Pell.Solution₁ d), Exists fun …
    h₀ : LT.lt 0 d
    hd : Not (IsSquare d)
    a₁ : Pell.Solution₁ d
    ha₁ : Pell.IsFundamental a₁
    b : Pell.Solution₁ d
    hb₂ : ∀ (y : Pell.Solution₁ d), (fun a₁ => And (LT.lt 1 a₁.x) (And (LT.lt 0 a₁ …
    ⊢ Pell.IsFundamental a
  -/
  rwa [hb₂ a h, ← hb₂ a₁ ⟨ha₁.1, ha₁.2.1, ha₁.eq_zpow_or_neg_zpow⟩]
  /-
    🎉 no goals
  -/


