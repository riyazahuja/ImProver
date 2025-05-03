/-- The real-valued function sending `x ∈ R` to the supremum of  `f(x*y)/f(y)`, where `y` runs over
the elements of `R`.-/
def seminormFromBounded' : R → ℝ := fun x ↦ iSup fun y : R ↦ f (x * y) / f y


/-- If `f : R → ℝ` is a nonzero, nonnegative, multiplicatively bounded function, then `f 1 ≠ 0`. -/
theorem map_one_ne_zero (f_ne_zero : f ≠ 0) (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) : f 1 ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ Ne (f 1) 0
  -/
  intro h1
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    h1 : Eq (f 1) 0
    ⊢ False
  -/
  specialize f_mul 1
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    h1 : Eq (f 1) 0
    f_mul : ∀ (y : R), LE.le (f (HMul.hMul 1 y)) (HMul.hMul (HMul.hMul c (f 1)) (f …
    ⊢ False
  -/
  simp_rw [h1, one_mul, mul_zero, zero_mul] at f_mul
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    h1 : Eq (f 1) 0
    f_mul : ∀ (y : R), LE.le (f y) 0
    ⊢ False
  -/
  obtain ⟨z, hz⟩ := Function.ne_iff.mp f_ne_zero
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    h1 : Eq (f 1) 0
    f_mul : ∀ (y : R), LE.le (f y) 0
    z : R
    hz : Ne (f z) (0 z)
    ⊢ False
  -/
  exact hz <| (f_mul z).antisymm (f_nonneg z)
  /-
    🎉 no goals
  -/


/-- If `f : R → ℝ` is a nonnegative multiplicatively bounded function and `x : R` is a unit with
  `f x ≠ 0`, then for every `n : ℕ`, we have `f (x ^ n) ≠ 0`. -/
theorem map_pow_ne_zero (f_nonneg : 0 ≤ f) {x : R} (hx : IsUnit x) (hfx : f x ≠ 0) (n : ℕ)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) : f (x ^ n) ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    x : R
    hx : IsUnit x
    hfx : Ne (f x) 0
    n : Nat
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ Ne (f (HPow.hPow x n)) 0
  -/
  have h1 : f 1 ≠ 0 := map_one_ne_zero (Function.ne_iff.mpr ⟨x, hfx⟩) f_nonneg f_mul
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    x : R
    hx : IsUnit x
    hfx : Ne (f x) 0
    n : Nat
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    h1 : Ne (f 1) 0
    ⊢ Ne (f (HPow.hPow x n)) 0
  -/
  intro hxn
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    x : R
    hx : IsUnit x
    hfx : Ne (f x) 0
    n : Nat
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    h1 : Ne (f 1) 0
    hxn : Eq (f (HPow.hPow x n)) 0
    ⊢ False
  -/
  have : f 1 ≤ 0 := by simpa [← mul_pow, hxn] using f_mul (x ^ n) (hx.unit⁻¹ ^ n)
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    x : R
    hx : IsUnit x
    hfx : Ne (f x) 0
    n : Nat
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    h1 : Ne (f 1) 0
    hxn : Eq (f (HPow.hPow x n)) 0
    this : LE.le (f 1) 0
    ⊢ False
  -/
  exact h1 <| this.antisymm (f_nonneg 1)
  /-
    🎉 no goals
  -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then given `x y : R` with
  `f x = 0`, we have `f (x * y) = 0`. -/
theorem map_mul_zero_of_map_zero (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) {x : R} (hx : f x = 0)
    (y : R) : f (x * y) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : Eq (f x) 0
    y : R
    ⊢ Eq (f (HMul.hMul x y)) 0
  -/
  replace f_mul : f (x * y) ≤ 0 := by simpa [hx] using f_mul x y
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    x : R
    hx : Eq (f x) 0
    y : R
    f_mul : LE.le (f (HMul.hMul x y)) 0
    ⊢ Eq (f (HMul.hMul x y)) 0
  -/
  exact le_antisymm f_mul (f_nonneg _)
  /-
    🎉 no goals
  -/


/-- `seminormFromBounded' f` preserves `0`. -/
theorem seminormFromBounded_zero (f_zero : f 0 = 0) : seminormFromBounded' f (0 : R) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_zero : Eq (f 0) 0
    ⊢ Eq (seminormFromBounded' f 0) 0
  -/
  simp_rw [seminormFromBounded', zero_mul, f_zero, zero_div, ciSup_const]
  /-
    🎉 no goals
  -/


theorem seminormFromBounded_aux (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) (x : R) : 0 ≤ c * f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ LE.le 0 (HMul.hMul c (f x))
  -/
  rcases (f_nonneg x).eq_or_gt with hx | hx
    /-
      case inl
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      hx : Eq (f x) (0 x)
      ⊢ LE.le 0 (HMul.hMul c (f x))
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      hx : LT.lt (0 x) (f x)
      ⊢ LE.le 0 (HMul.hMul c (f x))
    -/
  · change 0 < f x at hx
    have hc : 0 ≤ c := by
      specialize f_mul x 1
      rw [mul_one, show c * f x * f 1 = c * f 1 * f x by ring, le_mul_iff_one_le_left hx] at f_mul
      replace f_nonneg : 0 ≤ f 1 := f_nonneg 1
      rcases f_nonneg.eq_or_gt with h1 | h1
      · linarith [show (1 : ℝ) ≤ 0 by simpa [h1] using f_mul]
      · rw [← div_le_iff₀ h1] at f_mul
        linarith [one_div_pos.mpr h1]
    /-
      case inr
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      hx : LT.lt 0 (f x)
      hc : LE.le 0 c
      ⊢ LE.le 0 (HMul.hMul c (f x))
    -/
    positivity
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then for every `x : R`,
  the image of `y ↦ f (x * y) / f y` is bounded above. -/
theorem seminormFromBounded_bddAbove_range (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) (x : R) :
    BddAbove (Set.range fun y ↦ f (x * y) / f y) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ BddAbove (Set.range fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y))
  -/
  use c * f x
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ Membership.mem (upperBounds (Set.range fun y => HDiv.hDiv (f (HMul.hMul x y) …
  -/
  rintro r ⟨y, rfl⟩
  /-
    case h.intro
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x y : R
    ⊢ LE.le ((fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y)) y) (HMul.hMul c (f x))
  -/
  rcases (f_nonneg y).eq_or_gt with hy0 | hy0
    /-
      case h.intro.inl
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy0 : Eq (f y) (0 y)
      ⊢ LE.le ((fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y)) y) (HMul.hMul c (f x))
    -/
  · simpa [hy0] using seminormFromBounded_aux f_nonneg f_mul x
    /-
      🎉 no goals
    -/
    /-
      case h.intro.inr
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy0 : LT.lt (0 y) (f y)
      ⊢ LE.le ((fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y)) y) (HMul.hMul c (f x))
    -/
  · simpa [div_le_iff₀ hy0] using f_mul x y
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then for every `x : R`,
  `seminormFromBounded' f x` is bounded above by some multiple of `f x`. -/
theorem seminormFromBounded_le (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) (x : R) :
    seminormFromBounded' f x ≤ c * f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ LE.le (seminormFromBounded' f x) (HMul.hMul c (f x))
  -/
  refine ciSup_le (fun y ↦ ?_)
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x y : R
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x y)) (f y)) (HMul.hMul c (f x))
  -/
  rcases (f_nonneg y).eq_or_gt with hy | hy
    /-
      case inl
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Eq (f y) (0 y)
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x y)) (f y)) (HMul.hMul c (f x))
    -/
  · simpa [hy] using seminormFromBounded_aux f_nonneg f_mul x
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : LT.lt (0 y) (f y)
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x y)) (f y)) (HMul.hMul c (f x))
    -/
  · rw [div_le_iff₀ hy]
    /-
      case inr
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : LT.lt (0 y) (f y)
      ⊢ LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x)) (f y))
    -/
    apply f_mul
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then for every `x : R`,
  `f x ≤ f 1 * seminormFromBounded' f x`. -/
theorem seminormFromBounded_ge (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) (x : R) :
    f x ≤ f 1 * seminormFromBounded' f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
  -/
  by_cases h1 : f 1 = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h1 : Eq (f 1) 0
      ⊢ LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
    -/
  · specialize f_mul x 1
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      x : R
      h1 : Eq (f 1) 0
      f_mul : LE.le (f (HMul.hMul x 1)) (HMul.hMul (HMul.hMul c (f x)) (f 1))
      ⊢ LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
    -/
    rw [mul_one, h1, mul_zero] at f_mul
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      x : R
      h1 : Eq (f 1) 0
      f_mul : LE.le (f x) 0
      ⊢ LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
    -/
    have hx0 : f x = 0 := f_mul.antisymm (f_nonneg _)
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      x : R
      h1 : Eq (f 1) 0
      f_mul : LE.le (f x) 0
      hx0 : Eq (f x) 0
      ⊢ LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
    -/
    rw [hx0, h1, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h1 : Not (Eq (f 1) 0)
      ⊢ LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
    -/
  · rw [mul_comm, ← div_le_iff₀ (lt_of_le_of_ne' (f_nonneg _) h1)]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h1 : Not (Eq (f 1) 0)
      ⊢ LE.le (HDiv.hDiv (f x) (f 1)) (seminormFromBounded' f x)
    -/
    conv_lhs => rw [← mul_one x]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h1 : Not (Eq (f 1) 0)
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x 1)) (f 1)) (seminormFromBounded' f x)
    -/
    exact le_ciSup (seminormFromBounded_bddAbove_range f_nonneg f_mul x) (1 : R)
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then
  `seminormFromBounded' f` is nonnegative. -/
theorem seminormFromBounded_nonneg (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y)  :
    0 ≤ seminormFromBounded' f := fun x ↦
  le_csSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul x) ⟨1, rfl⟩
    (div_nonneg (f_nonneg _) (f_nonneg _))


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then
  `seminormFromBounded' f x = 0` if and only if `f x = 0`. -/
theorem seminormFromBounded_eq_zero_iff (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) (x : R) :
    seminormFromBounded' f x = 0 ↔ f x = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ Iff (Eq (seminormFromBounded' f x) 0) (Eq (f x) 0)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h : Eq (seminormFromBounded' f x) 0
      ⊢ Eq (f x) 0
    -/
  · have hf := seminormFromBounded_ge f_nonneg f_mul x
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h : Eq (seminormFromBounded' f x) 0
      hf : LE.le (f x) (HMul.hMul (f 1) (seminormFromBounded' f x))
      ⊢ Eq (f x) 0
    -/
    rw [h, mul_zero] at hf
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h : Eq (seminormFromBounded' f x) 0
      hf : LE.le (f x) 0
      ⊢ Eq (f x) 0
    -/
    exact hf.antisymm (f_nonneg _)
    /-
      🎉 no goals
    -/
  · have hf : seminormFromBounded' f x ≤ c * f x :=
      seminormFromBounded_le f_nonneg f_mul x
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h : Eq (f x) 0
      hf : LE.le (seminormFromBounded' f x) (HMul.hMul c (f x))
      ⊢ Eq (seminormFromBounded' f x) 0
    -/
    rw [h, mul_zero] at hf
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      h : Eq (f x) 0
      hf : LE.le (seminormFromBounded' f x) 0
      ⊢ Eq (seminormFromBounded' f x) 0
    -/
    exact hf.antisymm (seminormFromBounded_nonneg f_nonneg f_mul x)
    /-
      🎉 no goals
    -/


/-- If `f` is invariant under negation of `x`, then so is `seminormFromBounded'`.-/
theorem seminormFromBounded_neg (f_neg : ∀ x : R, f (-x) = f x) (x : R) :
    seminormFromBounded' f (-x) = seminormFromBounded' f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
    x : R
    ⊢ Eq (seminormFromBounded' f (Neg.neg x)) (seminormFromBounded' f x)
  -/
  suffices ⨆ y, f (-x * y) / f y = ⨆ y, f (x * y) / f y by simpa only [seminormFromBounded']
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
    x : R
    ⊢ Eq (iSup fun y => HDiv.hDiv (f (HMul.hMul (Neg.neg x) y)) (f y)) (iSup fun y …
  -/
  congr
  /-
    case e_s
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
    x : R
    ⊢ Eq (fun y => HDiv.hDiv (f (HMul.hMul (Neg.neg x) y)) (f y)) fun y => HDiv.hD …
  -/
  ext y
  /-
    case e_s.h
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
    x y : R
    ⊢ Eq (HDiv.hDiv (f (HMul.hMul (Neg.neg x) y)) (f y)) (HDiv.hDiv (f (HMul.hMul  …
  -/
  rw [neg_mul, f_neg]
  /-
    🎉 no goals
  -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then
  `seminormFromBounded' f` is submultiplicative. -/
theorem seminormFromBounded_mul (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) (x y : R) :
    seminormFromBounded' f (x * y) ≤ seminormFromBounded' f x * seminormFromBounded' f y := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x y : R
    ⊢ LE.le (seminormFromBounded' f (HMul.hMul x y)) (HMul.hMul (seminormFromBound …
  -/
  apply ciSup_le
  /-
    case H
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x y : R
    ⊢ ∀ (x_1 : R), LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) x_1)) (f x_1)) ( …
  -/
  by_cases hy : seminormFromBounded' f y = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Eq (seminormFromBounded' f y) 0
      ⊢ ∀ (x_1 : R), LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) x_1)) (f x_1)) ( …
    -/
  · rw [seminormFromBounded_eq_zero_iff f_nonneg f_mul] at hy
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Eq (f y) 0
      ⊢ ∀ (x_1 : R), LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) x_1)) (f x_1)) ( …
    -/
    intro z
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Eq (f y) 0
      z : R
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (HMul.hMul (semino …
    -/
    rw [mul_comm x y, mul_assoc, map_mul_zero_of_map_zero f_nonneg f_mul hy (x * z), zero_div]
    exact mul_nonneg (seminormFromBounded_nonneg f_nonneg f_mul x)
      (seminormFromBounded_nonneg f_nonneg f_mul y)
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Not (Eq (seminormFromBounded' f y) 0)
      ⊢ ∀ (x_1 : R), LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) x_1)) (f x_1)) ( …
    -/
  · intro z
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Not (Eq (seminormFromBounded' f y) 0)
      z : R
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (HMul.hMul (semino …
    -/
    rw [← div_le_iff₀ (lt_of_le_of_ne' (seminormFromBounded_nonneg f_nonneg f_mul _) hy)]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Not (Eq (seminormFromBounded' f y) 0)
      z : R
      ⊢ LE.le (HDiv.hDiv (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (semino …
    -/
    apply le_ciSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul x) z
    rw [div_le_iff₀ (lt_of_le_of_ne' (seminormFromBounded_nonneg f_nonneg f_mul _) hy),
      div_mul_eq_mul_div]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x y : R
      hy : Not (Eq (seminormFromBounded' f y) 0)
      z : R
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (HDiv.hDiv (HMul.h …
    -/
    by_cases hz : f z = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x y : R
        hy : Not (Eq (seminormFromBounded' f y) 0)
        z : R
        hz : Eq (f z) 0
        ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (HDiv.hDiv (HMul.h …
      -/
    · have hxyz : f (z * (x * y)) = 0 := map_mul_zero_of_map_zero f_nonneg f_mul hz _
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x y : R
        hy : Not (Eq (seminormFromBounded' f y) 0)
        z : R
        hz : Eq (f z) 0
        hxyz : Eq (f (HMul.hMul z (HMul.hMul x y))) 0
        ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (HDiv.hDiv (HMul.h …
      -/
      simp_rw [mul_comm, hxyz, zero_div]
      exact div_nonneg (mul_nonneg (seminormFromBounded_nonneg f_nonneg f_mul y) (f_nonneg _))
        (f_nonneg _)
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x y : R
        hy : Not (Eq (seminormFromBounded' f y) 0)
        z : R
        hz : Not (Eq (f z) 0)
        ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f z)) (HDiv.hDiv (HMul.h …
      -/
    · rw [div_le_div_iff_of_pos_right (lt_of_le_of_ne' (f_nonneg _) hz), mul_comm (f (x * z))]
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x y : R
        hy : Not (Eq (seminormFromBounded' f y) 0)
        z : R
        hz : Not (Eq (f z) 0)
        ⊢ LE.le (f (HMul.hMul (HMul.hMul x y) z)) (HMul.hMul (seminormFromBounded' f y …
      -/
      by_cases hxz : f (x * z) = 0
        /-
          case pos
          R : Type u_1
          inst✝ : CommRing R
          f : R → Real
          c : Real
          f_nonneg : LE.le 0 f
          f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
          x y : R
          hy : Not (Eq (seminormFromBounded' f y) 0)
          z : R
          hz : Not (Eq (f z) 0)
          hxz : Eq (f (HMul.hMul x z)) 0
          ⊢ LE.le (f (HMul.hMul (HMul.hMul x y) z)) (HMul.hMul (seminormFromBounded' f y …
        -/
      · rw [mul_comm x y, mul_assoc, mul_comm y, map_mul_zero_of_map_zero f_nonneg f_mul hxz y]
        /-
          case pos
          R : Type u_1
          inst✝ : CommRing R
          f : R → Real
          c : Real
          f_nonneg : LE.le 0 f
          f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
          x y : R
          hy : Not (Eq (seminormFromBounded' f y) 0)
          z : R
          hz : Not (Eq (f z) 0)
          hxz : Eq (f (HMul.hMul x z)) 0
          ⊢ LE.le 0 (HMul.hMul (seminormFromBounded' f y) (f (HMul.hMul x z)))
        -/
        exact mul_nonneg (seminormFromBounded_nonneg f_nonneg f_mul y) (f_nonneg _)
        /-
          🎉 no goals
        -/
        /-
          case neg
          R : Type u_1
          inst✝ : CommRing R
          f : R → Real
          c : Real
          f_nonneg : LE.le 0 f
          f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
          x y : R
          hy : Not (Eq (seminormFromBounded' f y) 0)
          z : R
          hz : Not (Eq (f z) 0)
          hxz : Not (Eq (f (HMul.hMul x z)) 0)
          ⊢ LE.le (f (HMul.hMul (HMul.hMul x y) z)) (HMul.hMul (seminormFromBounded' f y …
        -/
      · rw [← div_le_iff₀ (lt_of_le_of_ne' (f_nonneg _) hxz)]
        /-
          case neg
          R : Type u_1
          inst✝ : CommRing R
          f : R → Real
          c : Real
          f_nonneg : LE.le 0 f
          f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
          x y : R
          hy : Not (Eq (seminormFromBounded' f y) 0)
          z : R
          hz : Not (Eq (f z) 0)
          hxz : Not (Eq (f (HMul.hMul x z)) 0)
          ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f (HMul.hMul x z))) (sem …
        -/
        apply le_ciSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul y) (x * z)
        /-
          case neg
          R : Type u_1
          inst✝ : CommRing R
          f : R → Real
          c : Real
          f_nonneg : LE.le 0 f
          f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
          x y : R
          hy : Not (Eq (seminormFromBounded' f y) 0)
          z : R
          hz : Not (Eq (f z) 0)
          hxz : Not (Eq (f (HMul.hMul x z)) 0)
          ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x y) z)) (f (HMul.hMul x z))) (HDi …
        -/
        rw [div_le_div_iff_of_pos_right (lt_of_le_of_ne' (f_nonneg _) hxz), mul_comm x y, mul_assoc]
        /-
          🎉 no goals
        -/


/-- If `f : R → ℝ` is a nonzero, nonnegative, multiplicatively bounded function, then
  `seminormFromBounded' f 1 = 1`. -/
theorem seminormFromBounded_one (f_ne_zero : f ≠ 0) (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) :
    seminormFromBounded' f 1 = 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ Eq (seminormFromBounded' f 1) 1
  -/
  simp_rw [seminormFromBounded', one_mul]
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ Eq (iSup fun y => HDiv.hDiv (f y) (f y)) 1
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_ne_zero : Ne f 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      ⊢ LE.le (iSup fun y => HDiv.hDiv (f y) (f y)) 1
    -/
  · refine ciSup_le (fun x ↦ ?_)
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_ne_zero : Ne f 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      ⊢ LE.le (HDiv.hDiv (f x) (f x)) 1
    -/
    by_cases hx : f x = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_ne_zero : Ne f 0
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : Eq (f x) 0
        ⊢ LE.le (HDiv.hDiv (f x) (f x)) 1
      -/
    · rw [hx, div_zero]; exact zero_le_one
                         /-
                           🎉 no goals
                         -/
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_ne_zero : Ne f 0
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : Not (Eq (f x) 0)
        ⊢ LE.le (HDiv.hDiv (f x) (f x)) 1
      -/
    · rw [div_self hx]
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_ne_zero : Ne f 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      ⊢ LE.le 1 (iSup fun y => HDiv.hDiv (f y) (f y))
    -/
  · rw [← div_self (map_one_ne_zero f_ne_zero f_nonneg f_mul)]
    have h_bdd : BddAbove (Set.range fun y ↦ f y / f y) := by
      use (1 : ℝ)
      rintro r ⟨y, rfl⟩
      by_cases hy : f y = 0
      · simp only [hy, div_zero, zero_le_one]
      · simp only [div_self hy, le_refl]
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_ne_zero : Ne f 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      h_bdd : BddAbove (Set.range fun y => HDiv.hDiv (f y) (f y))
      ⊢ LE.le (HDiv.hDiv (f 1) (f 1)) (iSup fun y => HDiv.hDiv (f y) (f y))
    -/
    exact le_ciSup h_bdd (1 : R)
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then
  `seminormFromBounded' f 1 ≤ 1`. -/
theorem seminormFromBounded_one_le (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) :
    seminormFromBounded' f 1 ≤ 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ LE.le (seminormFromBounded' f 1) 1
  -/
  by_cases f_ne_zero : f ≠ 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_ne_zero : Ne f 0
      ⊢ LE.le (seminormFromBounded' f 1) 1
    -/
  · exact le_of_eq (seminormFromBounded_one f_ne_zero f_nonneg f_mul)
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_ne_zero : Not (Ne f 0)
      ⊢ LE.le (seminormFromBounded' f 1) 1
    -/
  · simp_rw [seminormFromBounded', one_mul]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_ne_zero : Not (Ne f 0)
      ⊢ LE.le (iSup fun y => HDiv.hDiv (f y) (f y)) 1
    -/
    refine ciSup_le (fun _ ↦ ?_)
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_ne_zero : Not (Ne f 0)
      x✝ : R
      ⊢ LE.le (HDiv.hDiv (f x✝) (f x✝)) 1
    -/
    push_neg at f_ne_zero
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x✝ : R
      f_ne_zero : Eq f 0
      ⊢ LE.le (HDiv.hDiv (f x✝) (f x✝)) 1
    -/
    simp only [f_ne_zero, Pi.zero_apply, div_zero, zero_le_one]
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded, subadditive function, then
  `seminormFromBounded' f` is subadditive. -/
theorem seminormFromBounded_add (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y)
    (f_add : ∀ a b, f (a + b) ≤ f a + f b) (x y : R) :
    seminormFromBounded' f (x + y) ≤ seminormFromBounded' f x + seminormFromBounded' f y := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
    x y : R
    ⊢ LE.le (seminormFromBounded' f (HAdd.hAdd x y)) (HAdd.hAdd (seminormFromBound …
  -/
  refine ciSup_le (fun z ↦ ?_)
  suffices hf : f ((x + y) * z) / f z ≤ f (x * z) / f z + f (y * z) / f z by
    exact le_trans hf (add_le_add
      (le_ciSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul x) z (le_refl _))
      (le_ciSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul y) z (le_refl _)))
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
    x y z : R
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) z)) (f z)) (HAdd.hAdd (HDiv.h …
  -/
  by_cases hz : f z = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      x y z : R
      hz : Eq (f z) 0
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) z)) (f z)) (HAdd.hAdd (HDiv.h …
    -/
  · simp only [hz, div_zero, zero_add, le_refl, or_self_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      x y z : R
      hz : Not (Eq (f z) 0)
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) z)) (f z)) (HAdd.hAdd (HDiv.h …
    -/
  · rw [div_add_div_same, div_le_div_iff_of_pos_right (lt_of_le_of_ne' (f_nonneg _) hz), add_mul]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      x y z : R
      hz : Not (Eq (f z) 0)
      ⊢ LE.le (f (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))) (HAdd.hAdd (f (HMul.hM …
    -/
    exact f_add _ _
    /-
      🎉 no goals
    -/


/-- `seminormFromBounded'` is a ring seminorm on `R`. -/
def seminormFromBounded (f_zero : f 0 = 0) (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y)
    (f_add : ∀ a b, f (a + b) ≤ f a + f b) (f_neg : ∀ x : R, f (-x) = f x) : RingSeminorm R where
  toFun     := seminormFromBounded' f
  map_zero' := seminormFromBounded_zero f_zero
  add_le'   := seminormFromBounded_add f_nonneg f_mul f_add
  mul_le'   := seminormFromBounded_mul f_nonneg f_mul
  neg'      := seminormFromBounded_neg f_neg


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded, nonarchimedean function, then
  `seminormFromBounded' f` is nonarchimedean. -/
theorem seminormFromBounded_isNonarchimedean (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y)
    (hna : IsNonarchimedean f) : IsNonarchimedean (seminormFromBounded' f) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    hna : IsNonarchimedean f
    ⊢ IsNonarchimedean (seminormFromBounded' f)
  -/
  refine fun x y ↦ ciSup_le (fun z ↦ ?_)
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    hna : IsNonarchimedean f
    x y z : R
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) z)) (f z)) (Max.max (seminorm …
  -/
  rw [le_max_iff]
  suffices hf : f ((x + y) * z) / f z ≤ f (x * z) / f z ∨ f ((x + y) * z) / f z ≤ f (y * z) / f z by
    rcases hf with hfx | hfy
    · exact Or.inl <| le_ciSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul x) z hfx
    · exact Or.inr <| le_ciSup_of_le (seminormFromBounded_bddAbove_range f_nonneg f_mul y) z hfy
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    hna : IsNonarchimedean f
    x y z : R
    ⊢ Or (LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) z)) (f z)) (HDiv.hDiv (f  …
  -/
  by_cases hz : f z = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      hna : IsNonarchimedean f
      x y z : R
      hz : Eq (f z) 0
      ⊢ Or (LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) z)) (f z)) (HDiv.hDiv (f  …
    -/
  · simp only [hz, div_zero, le_refl, or_self_iff]
    /-
      🎉 no goals
    -/
  · rw [div_le_div_iff_of_pos_right (lt_of_le_of_ne' (f_nonneg _) hz),
      div_le_div_iff_of_pos_right (lt_of_le_of_ne' (f_nonneg _) hz), add_mul, ← le_max_iff]
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      hna : IsNonarchimedean f
      x y z : R
      hz : Not (Eq (f z) 0)
      ⊢ LE.le (f (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))) (Max.max (f (HMul.hMul …
    -/
    exact hna _ _
    /-
      🎉 no goals
    -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function and `x : R` is
  multiplicative for `f`, then `seminormFromBounded' f x = f x`. -/
theorem seminormFromBounded_of_mul_apply (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) {x : R}
    (hx : ∀ y : R, f (x * y) = f x * f y) : seminormFromBounded' f x = f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    ⊢ Eq (seminormFromBounded' f x) (f x)
  -/
  simp_rw [seminormFromBounded', hx, ← mul_div_assoc']
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    ⊢ Eq (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y))) (f x)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      ⊢ LE.le (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y))) (f x)
    -/
  · refine ciSup_le (fun x ↦ ?_)
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x✝ : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x✝ y)) (HMul.hMul (f x✝) (f y))
      x : R
      ⊢ LE.le (HMul.hMul (f x✝) (HDiv.hDiv (f x) (f x))) (f x✝)
    -/
    by_cases hx : f x = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x✝ : R
        hx✝ : ∀ (y : R), Eq (f (HMul.hMul x✝ y)) (HMul.hMul (f x✝) (f y))
        x : R
        hx : Eq (f x) 0
        ⊢ LE.le (HMul.hMul (f x✝) (HDiv.hDiv (f x) (f x))) (f x✝)
      -/
    · rw [hx, div_zero, mul_zero]; exact f_nonneg _
                                   /-
                                     🎉 no goals
                                   -/
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x✝ : R
        hx✝ : ∀ (y : R), Eq (f (HMul.hMul x✝ y)) (HMul.hMul (f x✝) (f y))
        x : R
        hx : Not (Eq (f x) 0)
        ⊢ LE.le (HMul.hMul (f x✝) (HDiv.hDiv (f x) (f x))) (f x✝)
      -/
    · rw [div_self hx, mul_one]
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      x : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      ⊢ LE.le (f x) (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y)))
    -/
  · by_cases f_ne_zero : f ≠ 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        f_ne_zero : Ne f 0
        ⊢ LE.le (f x) (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y)))
      -/
    · conv_lhs => rw [← mul_one (f x)]
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        f_ne_zero : Ne f 0
        ⊢ LE.le (HMul.hMul (f x) 1) (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f …
      -/
      rw [← div_self (map_one_ne_zero f_ne_zero f_nonneg f_mul)]
      have h_bdd : BddAbove (Set.range fun y ↦ f x * (f y / f y)) := by
        use f x
        rintro r ⟨y, rfl⟩
        by_cases hy0 : f y = 0
        · simp only [hy0, div_zero, mul_zero]; exact f_nonneg _
        · simp only [div_self hy0, mul_one, le_refl]
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        f_ne_zero : Ne f 0
        h_bdd : BddAbove (Set.range fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y)))
        ⊢ LE.le (HMul.hMul (f x) (HDiv.hDiv (f 1) (f 1))) (iSup fun y => HMul.hMul (f  …
      -/
      exact le_ciSup h_bdd (1 : R)
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        f_ne_zero : Not (Ne f 0)
        ⊢ LE.le (f x) (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y)))
      -/
    · push_neg at f_ne_zero
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        c : Real
        f_nonneg : LE.le 0 f
        f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
        x : R
        hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        f_ne_zero : Eq f 0
        ⊢ LE.le (f x) (iSup fun y => HMul.hMul (f x) (HDiv.hDiv (f y) (f y)))
      -/
      simp_rw [f_ne_zero, Pi.zero_apply, zero_div, zero_mul, ciSup_const]; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- If `f : R → ℝ` is a nonnegative function and `x : R` is submultiplicative for `f`, then
  `seminormFromBounded' f x = f x`. -/
theorem seminormFromBounded_of_mul_le (f_nonneg : 0 ≤ f) {x : R}
    (hx : ∀ y : R, f (x * y) ≤ f x * f y) (h_one : f 1 ≤ 1) : seminormFromBounded' f x = f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_nonneg : LE.le 0 f
    x : R
    hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    h_one : LE.le (f 1) 1
    ⊢ Eq (seminormFromBounded' f x) (f x)
  -/
  simp_rw [seminormFromBounded']
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    f_nonneg : LE.le 0 f
    x : R
    hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    h_one : LE.le (f 1) 1
    ⊢ Eq (iSup fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y)) (f x)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      f_nonneg : LE.le 0 f
      x : R
      hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      h_one : LE.le (f 1) 1
      ⊢ LE.le (iSup fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y)) (f x)
    -/
  · refine ciSup_le (fun y ↦ ?_)
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      f_nonneg : LE.le 0 f
      x : R
      hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      h_one : LE.le (f 1) 1
      y : R
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x y)) (f y)) (f x)
    -/
    by_cases hy : f y = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        f_nonneg : LE.le 0 f
        x : R
        hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        h_one : LE.le (f 1) 1
        y : R
        hy : Eq (f y) 0
        ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x y)) (f y)) (f x)
      -/
    · rw [hy, div_zero]; exact f_nonneg _
                         /-
                           🎉 no goals
                         -/
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        f_nonneg : LE.le 0 f
        x : R
        hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        h_one : LE.le (f 1) 1
        y : R
        hy : Not (Eq (f y) 0)
        ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x y)) (f y)) (f x)
      -/
    · rw [div_le_iff₀ (lt_of_le_of_ne' (f_nonneg _) hy)]; exact hx _
                                                          /-
                                                            🎉 no goals
                                                          -/
  · have h_bdd : BddAbove (Set.range fun y ↦ f (x * y) / f y) := by
      use f x
      rintro r ⟨y, rfl⟩
      by_cases hy0 : f y = 0
      · simp only [hy0, div_zero]
        exact f_nonneg _
      · rw [← mul_one (f x), ← div_self hy0, ← mul_div_assoc,
          div_le_iff₀ (lt_of_le_of_ne' (f_nonneg _) hy0), mul_div_assoc, div_self hy0, mul_one]
        exact hx y
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      f_nonneg : LE.le 0 f
      x : R
      hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      h_one : LE.le (f 1) 1
      h_bdd : BddAbove (Set.range fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y))
      ⊢ LE.le (f x) (iSup fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y))
    -/
    convert le_ciSup h_bdd (1 : R)
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      f_nonneg : LE.le 0 f
      x : R
      hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      h_one : LE.le (f 1) 1
      h_bdd : BddAbove (Set.range fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y))
      ⊢ Eq (f x) (HDiv.hDiv (f (HMul.hMul x 1)) (f 1))
    -/
    by_cases h0 : f x = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        f_nonneg : LE.le 0 f
        x : R
        hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        h_one : LE.le (f 1) 1
        h_bdd : BddAbove (Set.range fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y))
        h0 : Eq (f x) 0
        ⊢ Eq (f x) (HDiv.hDiv (f (HMul.hMul x 1)) (f 1))
      -/
    · rw [mul_one, h0, zero_div]
      /-
        🎉 no goals
      -/
    · have heq : f 1 = 1 := by
        apply h_one.antisymm
        specialize hx 1
        rw [mul_one, le_mul_iff_one_le_right (lt_of_le_of_ne (f_nonneg _) (Ne.symm h0))] at hx
        exact hx
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        f : R → Real
        f_nonneg : LE.le 0 f
        x : R
        hx : ∀ (y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        h_one : LE.le (f 1) 1
        h_bdd : BddAbove (Set.range fun y => HDiv.hDiv (f (HMul.hMul x y)) (f y))
        h0 : Not (Eq (f x) 0)
        heq : Eq (f 1) 1
        ⊢ Eq (f x) (HDiv.hDiv (f (HMul.hMul x 1)) (f 1))
      -/
      rw [heq, mul_one, div_one]
      /-
        🎉 no goals
      -/


/-- If `f : R → ℝ` is a nonzero, nonnegative, multiplicatively bounded function, then
  `seminormFromBounded' f` is nonzero. -/
theorem seminormFromBounded_nonzero (f_ne_zero : f ≠ 0) (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) :
    seminormFromBounded' f ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ Ne (seminormFromBounded' f) 0
  -/
  obtain ⟨x, hx⟩ := Function.ne_iff.mp f_ne_zero
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : Ne (f x) (0 x)
    ⊢ Ne (seminormFromBounded' f) 0
  -/
  rw [Function.ne_iff]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : Ne (f x) (0 x)
    ⊢ Exists fun a => Ne (seminormFromBounded' f a) (0 a)
  -/
  use x
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : Ne (f x) (0 x)
    ⊢ Ne (seminormFromBounded' f x) (0 x)
  -/
  rw [ne_eq, Pi.zero_apply, seminormFromBounded_eq_zero_iff f_nonneg f_mul x]
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_ne_zero : Ne f 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : Ne (f x) (0 x)
    ⊢ Not (Eq (f x) 0)
  -/
  exact hx
  /-
    🎉 no goals
  -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function, then the kernel of
  `seminormFromBounded' f` equals the kernel of `f`. -/
theorem seminormFromBounded_ker (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) :
    seminormFromBounded' f ⁻¹' {0} = f ⁻¹' {0} := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    ⊢ Eq (Set.preimage (seminormFromBounded' f) (Singleton.singleton 0)) (Set.prei …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    ⊢ Iff (Membership.mem (Set.preimage (seminormFromBounded' f) (Singleton.single …
  -/
  exact seminormFromBounded_eq_zero_iff f_nonneg f_mul x
  /-
    🎉 no goals
  -/


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded, subadditive function that preserves
  zero and negation, then `seminormFromBounded' f` is a norm if and only if `f⁻¹' {0} = {0}`. -/
theorem seminormFromBounded_is_norm_iff (f_zero : f 0 = 0) (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y)
    (f_add : ∀ a b, f (a + b) ≤ f a + f b) (f_neg : ∀ x : R, f (-x) = f x) :
    (∀ x : R, (seminormFromBounded f_zero f_nonneg f_mul f_add f_neg).toFun x = 0 → x = 0) ↔
      f ⁻¹' {0} = {0} := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_zero : Eq (f 0) 0
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
    f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
    ⊢ Iff (∀ (x : R), Eq ((seminormFromBounded f_zero f_nonneg f_mul f_add f_neg). …
  -/
  refine ⟨fun h0 ↦ ?_, fun h_ker x hx ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_zero : Eq (f 0) 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
      h0 : ∀ (x : R), Eq ((seminormFromBounded f_zero f_nonneg f_mul f_add f_neg).to …
      ⊢ Eq (Set.preimage f (Singleton.singleton 0)) (Singleton.singleton 0)
    -/
  · rw [← seminormFromBounded_ker f_nonneg f_mul]
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_zero : Eq (f 0) 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
      h0 : ∀ (x : R), Eq ((seminormFromBounded f_zero f_nonneg f_mul f_add f_neg).to …
      ⊢ Eq (Set.preimage (seminormFromBounded' f) (Singleton.singleton 0)) (Singleto …
    -/
    ext x
    /-
      case refine_1.h
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_zero : Eq (f 0) 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
      h0 : ∀ (x : R), Eq ((seminormFromBounded f_zero f_nonneg f_mul f_add f_neg).to …
      x : R
      ⊢ Iff (Membership.mem (Set.preimage (seminormFromBounded' f) (Singleton.single …
    -/
    simp only [Set.mem_preimage, Set.mem_singleton_iff]
    /-
      case refine_1.h
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_zero : Eq (f 0) 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
      h0 : ∀ (x : R), Eq ((seminormFromBounded f_zero f_nonneg f_mul f_add f_neg).to …
      x : R
      ⊢ Iff (Eq (seminormFromBounded' f x) 0) (Eq x 0)
    -/
    exact ⟨fun h ↦ h0 x h, fun h ↦ by rw [h]; exact seminormFromBounded_zero f_zero⟩
    /-
      🎉 no goals
    -/
  · rw [← Set.mem_singleton_iff, ← h_ker, Set.mem_preimage, Set.mem_singleton_iff,
      ← seminormFromBounded_eq_zero_iff f_nonneg f_mul x]
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      f : R → Real
      c : Real
      f_zero : Eq (f 0) 0
      f_nonneg : LE.le 0 f
      f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
      f_add : ∀ (a b : R), LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
      f_neg : ∀ (x : R), Eq (f (Neg.neg x)) (f x)
      h_ker : Eq (Set.preimage f (Singleton.singleton 0)) (Singleton.singleton 0)
      x : R
      hx : Eq ((seminormFromBounded f_zero f_nonneg f_mul f_add f_neg).toFun x) 0
      ⊢ Eq (seminormFromBounded' f x) 0
    -/
    exact hx
    /-
      🎉 no goals
    -/


/-- `seminormFromBounded' f` as a `RingNorm` on `R`, provided that `f` is nonnegative,
  multiplicatively bounded and subadditive, that it preserves `0` and negation, and that `f` has
  trivial kernel. -/
def normFromBounded (f_zero : f 0 = 0) (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y)
    (f_add : ∀ a b, f (a + b) ≤ f a + f b) (f_neg : ∀ x : R, f (-x) = f x)
    (f_ker : f ⁻¹' {0} = {0}) : RingNorm R :=
  { seminormFromBounded f_zero f_nonneg f_mul f_add f_neg with
    eq_zero_of_map_eq_zero' :=
      (seminormFromBounded_is_norm_iff f_zero f_nonneg f_mul f_add f_neg).mpr f_ker }


/-- If `f : R → ℝ` is a nonnegative, multiplicatively bounded function and `x : R` is
  multiplicative for `f`, then `x` is multiplicative for `seminormFromBounded' f`. -/
theorem seminormFromBounded_of_mul_is_mul (f_nonneg : 0 ≤ f)
    (f_mul : ∀ x y : R, f (x * y) ≤ c * f x * f y) {x : R}
    (hx : ∀ y : R, f (x * y) = f x * f y) (y : R) :
    seminormFromBounded' f (x * y) = seminormFromBounded' f x * seminormFromBounded' f y := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : R → Real
    c : Real
    f_nonneg : LE.le 0 f
    f_mul : ∀ (x y : R), LE.le (f (HMul.hMul x y)) (HMul.hMul (HMul.hMul c (f x))  …
    x : R
    hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    y : R
    ⊢ Eq (seminormFromBounded' f (HMul.hMul x y)) (HMul.hMul (seminormFromBounded' …
  -/
  rw [seminormFromBounded_of_mul_apply f_nonneg f_mul hx]
  simp only [seminormFromBounded', mul_assoc, hx, mul_div_assoc,
    Real.mul_iSup_of_nonneg (f_nonneg _)]


