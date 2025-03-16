/-- If `f : 𝕜 → 𝕜` is convex, then for any three points `x < y < z` the slope of the secant line of
`f` on `[x, y]` is less than the slope of the secant line of `f` on `[y, z]`. -/
theorem ConvexOn.slope_mono_adjacent (hf : ConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x < y) (hyz : y < z) : (f y - f x) / (y - x) ≤ (f z - f y) / (z - y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz := hxy.trans hyz
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxz : LT.lt x z
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  rw [← sub_pos] at hxy hxz hyz
  suffices f y / (y - x) + f y / (z - y) ≤ f x / (y - x) + f z / (z - y) by
    ring_nf at this ⊢
    linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  set a := (z - y) / (z - x)
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  set b := (y - x) / (z - x)
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  have hy : a • x + b • z = y := by field_simp [a, b]; ring
  have key :=
    hf.2 hx hz (show 0 ≤ a by apply div_nonneg <;> linarith)
      (show 0 ≤ b by apply div_nonneg <;> linarith)
      (show a + b = 1 by field_simp [a, b])
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z))) (HAdd.hAdd (HS …
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  rw [hy] at key
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LE.le (f y) (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f z)))
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  replace key := mul_le_mul_of_nonneg_left key hxz.le
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LE.le (HMul.hMul (HSub.hSub z x) (f y)) (HMul.hMul (HSub.hSub z x) (HAdd …
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  field_simp [a, b, mul_comm (z - x) _] at key ⊢
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LE.le (HMul.hMul (f y) (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub …
    ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HMul.hMul (f y) (HSub.hSub z y)) (HMul.hMul (f  …
  -/
  rw [div_le_div_iff_of_pos_right]
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt 0 (HSub.hSub y x)
      hyz : LT.lt 0 (HSub.hSub z y)
      hxz : LT.lt 0 (HSub.hSub z x)
      a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
      b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
      hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
      key : LE.le (HMul.hMul (f y) (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub …
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (f y) (HSub.hSub z y)) (HMul.hMul (f y) (HSub.hS …
    -/
  · linarith
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt 0 (HSub.hSub y x)
      hyz : LT.lt 0 (HSub.hSub z y)
      hxz : LT.lt 0 (HSub.hSub z x)
      a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
      b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
      hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
      key : LE.le (HMul.hMul (f y) (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub …
      ⊢ LT.lt 0 (HMul.hMul (HSub.hSub y x) (HSub.hSub z y))
    -/
  · positivity
    /-
      🎉 no goals
    -/


/-- If `f : 𝕜 → 𝕜` is concave, then for any three points `x < y < z` the slope of the secant line of
`f` on `[x, y]` is greater than the slope of the secant line of `f` on `[y, z]`. -/
theorem ConcaveOn.slope_anti_adjacent (hf : ConcaveOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x < y) (hyz : y < z) : (f z - f y) / (z - y) ≤ (f y - f x) / (y - x) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConcaveOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (HSub.h …
  -/
  have := neg_le_neg (ConvexOn.slope_mono_adjacent hf.neg hx hz hxy hyz)
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConcaveOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    this : LE.le (Neg.neg (HDiv.hDiv (HSub.hSub (Neg.neg f z) (Neg.neg f y)) (HSub …
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (HSub.h …
  -/
  simp only [Pi.neg_apply, ← neg_div, neg_sub', neg_neg] at this
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConcaveOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    this : LE.le (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (H …
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (HSub.h …
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- If `f : 𝕜 → 𝕜` is strictly convex, then for any three points `x < y < z` the slope of the
secant line of `f` on `[x, y]` is strictly less than the slope of the secant line of `f` on
`[y, z]`. -/
theorem StrictConvexOn.slope_strict_mono_adjacent (hf : StrictConvexOn 𝕜 s f) {x y z : 𝕜}
    (hx : x ∈ s) (hz : z ∈ s) (hxy : x < y) (hyz : y < z) :
    (f y - f x) / (y - x) < (f z - f y) / (z - y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz := hxy.trans hyz
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxz : LT.lt x z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz' := hxz.ne
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxz : LT.lt x z
    hxz' : Ne x z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  rw [← sub_pos] at hxy hxz hyz
  suffices f y / (y - x) + f y / (z - y) < f x / (y - x) + f z / (z - y) by
    ring_nf at this ⊢
    linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  set a := (z - y) / (z - x)
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  set b := (y - x) / (z - x)
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  have hy : a • x + b • z = y := by field_simp [a, b]; ring
  have key :=
    hf.2 hx hz hxz' (div_pos hyz hxz) (div_pos hxy hxz)
      (show a + b = 1 by field_simp [a, b])
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LT.lt (f (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z …
    ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  rw [hy] at key
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LT.lt (f y) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSu …
    ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  replace key := mul_lt_mul_of_pos_left key hxz
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LT.lt (HMul.hMul (HSub.hSub z x) (f y)) (HMul.hMul (HSub.hSub z x) (HAdd …
    ⊢ LT.lt (HAdd.hAdd (HDiv.hDiv (f y) (HSub.hSub y x)) (HDiv.hDiv (f y) (HSub.hS …
  -/
  field_simp [mul_comm (z - x) _] at key ⊢
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt 0 (HSub.hSub y x)
    hyz : LT.lt 0 (HSub.hSub z y)
    hxz : LT.lt 0 (HSub.hSub z x)
    hxz' : Ne x z
    a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
    b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
    hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
    key : LT.lt (HMul.hMul (f y) (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub …
    ⊢ LT.lt (HDiv.hDiv (HAdd.hAdd (HMul.hMul (f y) (HSub.hSub z y)) (HMul.hMul (f  …
  -/
  rw [div_lt_div_iff_of_pos_right]
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt 0 (HSub.hSub y x)
      hyz : LT.lt 0 (HSub.hSub z y)
      hxz : LT.lt 0 (HSub.hSub z x)
      hxz' : Ne x z
      a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
      b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
      hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
      key : LT.lt (HMul.hMul (f y) (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub …
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul (f y) (HSub.hSub z y)) (HMul.hMul (f y) (HSub.hS …
    -/
  · linarith
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt 0 (HSub.hSub y x)
      hyz : LT.lt 0 (HSub.hSub z y)
      hxz : LT.lt 0 (HSub.hSub z x)
      hxz' : Ne x z
      a : 𝕜 := HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)
      b : 𝕜 := HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x)
      hy : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z)) y
      key : LT.lt (HMul.hMul (f y) (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub …
      ⊢ LT.lt 0 (HMul.hMul (HSub.hSub y x) (HSub.hSub z y))
    -/
  · positivity
    /-
      🎉 no goals
    -/


/-- If `f : 𝕜 → 𝕜` is strictly concave, then for any three points `x < y < z` the slope of the
secant line of `f` on `[x, y]` is strictly greater than the slope of the secant line of `f` on
`[y, z]`. -/
theorem StrictConcaveOn.slope_anti_adjacent (hf : StrictConcaveOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s)
    (hz : z ∈ s) (hxy : x < y) (hyz : y < z) : (f z - f y) / (z - y) < (f y - f x) / (y - x) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (HSub.h …
  -/
  have := neg_lt_neg (StrictConvexOn.slope_strict_mono_adjacent hf.neg hx hz hxy hyz)
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    this : LT.lt (Neg.neg (HDiv.hDiv (HSub.hSub (Neg.neg f z) (Neg.neg f y)) (HSub …
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (HSub.h …
  -/
  simp only [Pi.neg_apply, ← neg_div, neg_sub', neg_neg] at this
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    this : LT.lt (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (H …
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y)) (HDiv.hDiv (HSub.h …
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- If for any three points `x < y < z`, the slope of the secant line of `f : 𝕜 → 𝕜` on `[x, y]` is
less than the slope of the secant line of `f` on `[y, z]`, then `f` is convex. -/
theorem convexOn_of_slope_mono_adjacent (hs : Convex 𝕜 s)
    (hf :
      ∀ {x y z : 𝕜},
        x ∈ s → z ∈ s → x < y → y < z → (f y - f x) / (y - x) ≤ (f z - f y) / (z - y)) :
    ConvexOn 𝕜 s f :=
  LinearOrder.convexOn_of_lt hs fun x hx z hz hxz a b ha hb hab => by
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hs : Convex 𝕜 s
      hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
      x : 𝕜
      hx : Membership.mem s x
      z : 𝕜
      hz : Membership.mem s z
      hxz : LT.lt x z
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z))) (HAdd.hAdd (HSMul. …
    -/
    let y := a * x + b * z
    have hxy : x < y := by
      rw [← one_mul x, ← hab, add_mul]
      exact add_lt_add_left ((mul_lt_mul_left hb).2 hxz) _
    have hyz : y < z := by
      rw [← one_mul z, ← hab, add_mul]
      exact add_lt_add_right ((mul_lt_mul_left ha).2 hxz) _
    have : (f y - f x) * (z - y) ≤ (f z - f y) * (y - x) :=
      (div_le_div_iff₀ (sub_pos.2 hxy) (sub_pos.2 hyz)).1 (hf hx hz hxy hyz)
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hs : Convex 𝕜 s
      hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
      x : 𝕜
      hx : Membership.mem s x
      z : 𝕜
      hz : Membership.mem s z
      hxz : LT.lt x z
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      y : 𝕜 := HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)
      hxy : LT.lt x y
      hyz : LT.lt y z
      this : LE.le (HMul.hMul (HSub.hSub (f y) (f x)) (HSub.hSub z y)) (HMul.hMul (H …
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z))) (HAdd.hAdd (HSMul. …
    -/
    have hxz : 0 < z - x := sub_pos.2 (hxy.trans hyz)
    have ha : (z - y) / (z - x) = a := by
      rw [eq_comm, ← sub_eq_iff_eq_add'] at hab
      dsimp [y]
      simp_rw [div_eq_iff hxz.ne', ← hab]
      ring
    have hb : (y - x) / (z - x) = b := by
      rw [eq_comm, ← sub_eq_iff_eq_add] at hab
      dsimp [y]
      simp_rw [div_eq_iff hxz.ne', ← hab]
      ring
    rwa [sub_mul, sub_mul, sub_le_iff_le_add', ← add_sub_assoc, le_sub_iff_add_le, ← mul_add,
      sub_add_sub_cancel, ← le_div_iff₀ hxz, add_div, mul_div_assoc, mul_div_assoc, mul_comm (f x),
      mul_comm (f z), ha, hb] at this


/-- If for any three points `x < y < z`, the slope of the secant line of `f : 𝕜 → 𝕜` on `[x, y]` is
greater than the slope of the secant line of `f` on `[y, z]`, then `f` is concave. -/
theorem concaveOn_of_slope_anti_adjacent (hs : Convex 𝕜 s)
    (hf :
      ∀ {x y z : 𝕜},
        x ∈ s → z ∈ s → x < y → y < z → (f z - f y) / (z - y) ≤ (f y - f x) / (y - x)) :
    ConcaveOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    ⊢ ConcaveOn 𝕜 s f
  -/
  rw [← neg_convexOn_iff]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    ⊢ ConvexOn 𝕜 s (Neg.neg f)
  -/
  refine convexOn_of_slope_mono_adjacent hs fun hx hz hxy hyz => ?_
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    x✝ y✝ z✝ : 𝕜
    hx : Membership.mem s x✝
    hz : Membership.mem s z✝
    hxy : LT.lt x✝ y✝
    hyz : LT.lt y✝ z✝
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (Neg.neg f y✝) (Neg.neg f x✝)) (HSub.hSub y✝ x✝) …
  -/
  rw [← neg_le_neg_iff]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    x✝ y✝ z✝ : 𝕜
    hx : Membership.mem s x✝
    hz : Membership.mem s z✝
    hxy : LT.lt x✝ y✝
    hyz : LT.lt y✝ z✝
    ⊢ LE.le (Neg.neg (HDiv.hDiv (HSub.hSub (Neg.neg f z✝) (Neg.neg f y✝)) (HSub.hS …
  -/
  simp_rw [← neg_div, neg_sub, Pi.neg_apply, neg_sub_neg]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    x✝ y✝ z✝ : 𝕜
    hx : Membership.mem s x✝
    hz : Membership.mem s z✝
    hxy : LT.lt x✝ y✝
    hyz : LT.lt y✝ z✝
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z✝) (f y✝)) (HSub.hSub z✝ y✝)) (HDiv.hDiv (HS …
  -/
  exact hf hx hz hxy hyz
  /-
    🎉 no goals
  -/


/-- If for any three points `x < y < z`, the slope of the secant line of `f : 𝕜 → 𝕜` on `[x, y]` is
strictly less than the slope of the secant line of `f` on `[y, z]`, then `f` is strictly convex. -/
theorem strictConvexOn_of_slope_strict_mono_adjacent (hs : Convex 𝕜 s)
    (hf :
      ∀ {x y z : 𝕜},
        x ∈ s → z ∈ s → x < y → y < z → (f y - f x) / (y - x) < (f z - f y) / (z - y)) :
    StrictConvexOn 𝕜 s f :=
  LinearOrder.strictConvexOn_of_lt hs fun x hx z hz hxz a b ha hb hab => by
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hs : Convex 𝕜 s
      hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
      x : 𝕜
      hx : Membership.mem s x
      z : 𝕜
      hz : Membership.mem s z
      hxz : LT.lt x z
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z))) (HAdd.hAdd (HSMul. …
    -/
    let y := a * x + b * z
    have hxy : x < y := by
      rw [← one_mul x, ← hab, add_mul]
      exact add_lt_add_left ((mul_lt_mul_left hb).2 hxz) _
    have hyz : y < z := by
      rw [← one_mul z, ← hab, add_mul]
      exact add_lt_add_right ((mul_lt_mul_left ha).2 hxz) _
    have : (f y - f x) * (z - y) < (f z - f y) * (y - x) :=
      (div_lt_div_iff₀ (sub_pos.2 hxy) (sub_pos.2 hyz)).1 (hf hx hz hxy hyz)
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hs : Convex 𝕜 s
      hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
      x : 𝕜
      hx : Membership.mem s x
      z : 𝕜
      hz : Membership.mem s z
      hxz : LT.lt x z
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      y : 𝕜 := HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)
      hxy : LT.lt x y
      hyz : LT.lt y z
      this : LT.lt (HMul.hMul (HSub.hSub (f y) (f x)) (HSub.hSub z y)) (HMul.hMul (H …
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z))) (HAdd.hAdd (HSMul. …
    -/
    have hxz : 0 < z - x := sub_pos.2 (hxy.trans hyz)
    have ha : (z - y) / (z - x) = a := by
      rw [eq_comm, ← sub_eq_iff_eq_add'] at hab
      dsimp [y]
      simp_rw [div_eq_iff hxz.ne', ← hab]
      ring
    have hb : (y - x) / (z - x) = b := by
      rw [eq_comm, ← sub_eq_iff_eq_add] at hab
      dsimp [y]
      simp_rw [div_eq_iff hxz.ne', ← hab]
      ring
    rwa [sub_mul, sub_mul, sub_lt_iff_lt_add', ← add_sub_assoc, lt_sub_iff_add_lt, ← mul_add,
      sub_add_sub_cancel, ← lt_div_iff₀ hxz, add_div, mul_div_assoc, mul_div_assoc, mul_comm (f x),
      mul_comm (f z), ha, hb] at this


/-- If for any three points `x < y < z`, the slope of the secant line of `f : 𝕜 → 𝕜` on `[x, y]` is
strictly greater than the slope of the secant line of `f` on `[y, z]`, then `f` is strictly concave.
-/
theorem strictConcaveOn_of_slope_strict_anti_adjacent (hs : Convex 𝕜 s)
    (hf :
      ∀ {x y z : 𝕜},
        x ∈ s → z ∈ s → x < y → y < z → (f z - f y) / (z - y) < (f y - f x) / (y - x)) :
    StrictConcaveOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    ⊢ StrictConcaveOn 𝕜 s f
  -/
  rw [← neg_strictConvexOn_iff]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    ⊢ StrictConvexOn 𝕜 s (Neg.neg f)
  -/
  refine strictConvexOn_of_slope_strict_mono_adjacent hs fun hx hz hxy hyz => ?_
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    x✝ y✝ z✝ : 𝕜
    hx : Membership.mem s x✝
    hz : Membership.mem s z✝
    hxy : LT.lt x✝ y✝
    hyz : LT.lt y✝ z✝
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Neg.neg f y✝) (Neg.neg f x✝)) (HSub.hSub y✝ x✝) …
  -/
  rw [← neg_lt_neg_iff]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    x✝ y✝ z✝ : 𝕜
    hx : Membership.mem s x✝
    hz : Membership.mem s z✝
    hxy : LT.lt x✝ y✝
    hyz : LT.lt y✝ z✝
    ⊢ LT.lt (Neg.neg (HDiv.hDiv (HSub.hSub (Neg.neg f z✝) (Neg.neg f y✝)) (HSub.hS …
  -/
  simp_rw [← neg_div, neg_sub, Pi.neg_apply, neg_sub_neg]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hs : Convex 𝕜 s
    hf : ∀ {x y z : 𝕜}, Membership.mem s x → Membership.mem s z → LT.lt x y → LT.l …
    x✝ y✝ z✝ : 𝕜
    hx : Membership.mem s x✝
    hz : Membership.mem s z✝
    hxy : LT.lt x✝ y✝
    hyz : LT.lt y✝ z✝
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z✝) (f y✝)) (HSub.hSub z✝ y✝)) (HDiv.hDiv (HS …
  -/
  exact hf hx hz hxy hyz
  /-
    🎉 no goals
  -/


/-- A function `f : 𝕜 → 𝕜` is convex iff for any three points `x < y < z` the slope of the secant
line of `f` on `[x, y]` is less than the slope of the secant line of `f` on `[y, z]`. -/
theorem convexOn_iff_slope_mono_adjacent :
    ConvexOn 𝕜 s f ↔
      Convex 𝕜 s ∧ ∀ ⦃x y z : 𝕜⦄,
          x ∈ s → z ∈ s → x < y → y < z → (f y - f x) / (y - x) ≤ (f z - f y) / (z - y) :=
  ⟨fun h => ⟨h.1, fun _ _ _ => h.slope_mono_adjacent⟩, fun h =>
    convexOn_of_slope_mono_adjacent h.1 (@fun _ _ _ hx hy => h.2 hx hy)⟩


/-- A function `f : 𝕜 → 𝕜` is concave iff for any three points `x < y < z` the slope of the secant
line of `f` on `[x, y]` is greater than the slope of the secant line of `f` on `[y, z]`. -/
theorem concaveOn_iff_slope_anti_adjacent :
    ConcaveOn 𝕜 s f ↔
      Convex 𝕜 s ∧
        ∀ ⦃x y z : 𝕜⦄,
          x ∈ s → z ∈ s → x < y → y < z → (f z - f y) / (z - y) ≤ (f y - f x) / (y - x) :=
  ⟨fun h => ⟨h.1, fun _ _ _ => h.slope_anti_adjacent⟩, fun h =>
    concaveOn_of_slope_anti_adjacent h.1 (@fun _ _ _ hx hy => h.2 hx hy)⟩


/-- A function `f : 𝕜 → 𝕜` is strictly convex iff for any three points `x < y < z` the slope of
the secant line of `f` on `[x, y]` is strictly less than the slope of the secant line of `f` on
`[y, z]`. -/
theorem strictConvexOn_iff_slope_strict_mono_adjacent :
    StrictConvexOn 𝕜 s f ↔
      Convex 𝕜 s ∧
        ∀ ⦃x y z : 𝕜⦄,
          x ∈ s → z ∈ s → x < y → y < z → (f y - f x) / (y - x) < (f z - f y) / (z - y) :=
  ⟨fun h => ⟨h.1, fun _ _ _ => h.slope_strict_mono_adjacent⟩, fun h =>
    strictConvexOn_of_slope_strict_mono_adjacent h.1 (@fun _ _ _ hx hy => h.2 hx hy)⟩


/-- A function `f : 𝕜 → 𝕜` is strictly concave iff for any three points `x < y < z` the slope of
the secant line of `f` on `[x, y]` is strictly greater than the slope of the secant line of `f` on
`[y, z]`. -/
theorem strictConcaveOn_iff_slope_strict_anti_adjacent :
    StrictConcaveOn 𝕜 s f ↔
      Convex 𝕜 s ∧
        ∀ ⦃x y z : 𝕜⦄,
          x ∈ s → z ∈ s → x < y → y < z → (f z - f y) / (z - y) < (f y - f x) / (y - x) :=
  ⟨fun h => ⟨h.1, fun _ _ _ => h.slope_anti_adjacent⟩, fun h =>
    strictConcaveOn_of_slope_strict_anti_adjacent h.1 (@fun _ _ _ hx hy => h.2 hx hy)⟩


theorem ConvexOn.secant_mono_aux1 (hf : ConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x < y) (hyz : y < z) : (z - x) * f y ≤ (z - y) * f x + (y - x) * f z := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LE.le (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  have hxy' : 0 < y - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    ⊢ LE.le (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  have hyz' : 0 < z - y := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    ⊢ LE.le (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  have hxz' : 0 < z - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  rw [← le_div_iff₀' hxz']
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (f y) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HSub.hSub z y) (f x)) (HMul.hM …
  -/
  have ha : 0 ≤ (z - y) / (z - x) := by positivity
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ha : LE.le 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
    ⊢ LE.le (f y) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HSub.hSub z y) (f x)) (HMul.hM …
  -/
  have hb : 0 ≤ (y - x) / (z - x) := by positivity
  calc
    f y = f ((z - y) / (z - x) * x + (y - x) / (z - x) * z) := ?_
    _ ≤ (z - y) / (z - x) * f x + (y - x) / (z - x) * f z := hf.2 hx hz ha hb ?_
    _ = ((z - y) * f x + (y - x) * f z) / (z - x) := ?_
    /-
      case calc_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LE.le 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LE.le 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (f y) (f (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x) …
    -/
  · congr 1
    /-
      case calc_1.e_a
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LE.le 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LE.le 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq y (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)) x) (H …
    -/
    field_simp
    /-
      case calc_1.e_a
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LE.le 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LE.le 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (HMul.hMul y (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y) x) (H …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LE.le 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LE.le 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
    -/
  · field_simp
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LE.le 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LE.le 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)) (f x))  …
    -/
  · field_simp
    /-
      🎉 no goals
    -/


theorem ConvexOn.secant_mono_aux2 (hf : ConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x < y) (hyz : y < z) : (f y - f x) / (y - x) ≤ (f z - f x) / (z - x) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxy' : 0 < y - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz' : 0 < z - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  rw [div_le_div_iff₀ hxy' hxz']
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (HMul.hMul (HSub.hSub (f y) (f x)) (HSub.hSub z x)) (HMul.hMul (HSub.h …
  -/
  linarith only [hf.secant_mono_aux1 hx hz hxy hyz]
  /-
    🎉 no goals
  -/


theorem ConvexOn.secant_mono_aux3 (hf : ConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x < y) (hyz : y < z) : (f z - f x) / (z - x) ≤ (f z - f y) / (z - y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z) (f x)) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
  -/
  have hyz' : 0 < z - y := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hyz' : LT.lt 0 (HSub.hSub z y)
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z) (f x)) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz' : 0 < z - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f z) (f x)) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
  -/
  rw [div_le_div_iff₀ hxz' hyz']
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LE.le (HMul.hMul (HSub.hSub (f z) (f x)) (HSub.hSub z y)) (HMul.hMul (HSub.h …
  -/
  linarith only [hf.secant_mono_aux1 hx hz hxy hyz]
  /-
    🎉 no goals
  -/


/-- If `f : 𝕜 → 𝕜` is convex, then for any point `a` the slope of the secant line of `f` through `a`
and `b ≠ a` is monotone with respect to `b`. -/
theorem ConvexOn.secant_mono (hf : ConvexOn 𝕜 s f) {a x y : 𝕜} (ha : a ∈ s) (hx : x ∈ s)
    (hy : y ∈ s) (hxa : x ≠ a) (hya : y ≠ a) (hxy : x ≤ y) :
    (f x - f a) / (x - a) ≤ (f y - f a) / (y - a) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy : LE.le x y
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
  -/
  rcases eq_or_lt_of_le hxy with (rfl | hxy)
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      a x : 𝕜
      ha : Membership.mem s a
      hx : Membership.mem s x
      hxa : Ne x a
      hy : Membership.mem s x
      hya : Ne x a
      hxy : LE.le x x
      ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy✝ : LE.le x y
    hxy : LT.lt x y
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
  -/
  cases' lt_or_gt_of_ne hxa with hxa hxa
    /-
      case inr.inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      a x y : 𝕜
      ha : Membership.mem s a
      hx : Membership.mem s x
      hy : Membership.mem s y
      hxa✝ : Ne x a
      hya : Ne y a
      hxy✝ : LE.le x y
      hxy : LT.lt x y
      hxa : LT.lt x a
      ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
    -/
  · cases' lt_or_gt_of_ne hya with hya hya
      /-
        case inr.inl.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        a x y : 𝕜
        ha : Membership.mem s a
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxa✝ : Ne x a
        hya✝ : Ne y a
        hxy✝ : LE.le x y
        hxy : LT.lt x y
        hxa : LT.lt x a
        hya : LT.lt y a
        ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    · convert hf.secant_mono_aux3 hx ha hxy hya using 1 <;> rw [← neg_div_neg_eq] <;> field_simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
      /-
        case inr.inl.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        a x y : 𝕜
        ha : Membership.mem s a
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxa✝ : Ne x a
        hya✝ : Ne y a
        hxy✝ : LE.le x y
        hxy : LT.lt x y
        hxa : LT.lt x a
        hya : GT.gt y a
        ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
      -/
    · convert hf.slope_mono_adjacent hx hy hxa hya using 1
      /-
        case h.e'_3
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        a x y : 𝕜
        ha : Membership.mem s a
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxa✝ : Ne x a
        hya✝ : Ne y a
        hxy✝ : LE.le x y
        hxy : LT.lt x y
        hxa : LT.lt x a
        hya : GT.gt y a
        ⊢ Eq (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.hSub …
      -/
      rw [← neg_div_neg_eq]; field_simp
                             /-
                               🎉 no goals
                             -/
    /-
      case inr.inr
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      a x y : 𝕜
      ha : Membership.mem s a
      hx : Membership.mem s x
      hy : Membership.mem s y
      hxa✝ : Ne x a
      hya : Ne y a
      hxy✝ : LE.le x y
      hxy : LT.lt x y
      hxa : GT.gt x a
      ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
    -/
  · exact hf.secant_mono_aux2 ha hy hxa hxy
    /-
      🎉 no goals
    -/


theorem StrictConvexOn.secant_strict_mono_aux1 (hf : StrictConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s)
    (hz : z ∈ s) (hxy : x < y) (hyz : y < z) : (z - x) * f y < (z - y) * f x + (y - x) * f z := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  have hxy' : 0 < y - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    ⊢ LT.lt (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  have hyz' : 0 < z - y := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    ⊢ LT.lt (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  have hxz' : 0 < z - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LT.lt (HMul.hMul (HSub.hSub z x) (f y)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y …
  -/
  rw [← lt_div_iff₀' hxz']
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LT.lt (f y) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HSub.hSub z y) (f x)) (HMul.hM …
  -/
  have ha : 0 < (z - y) / (z - x) := by positivity
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ha : LT.lt 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
    ⊢ LT.lt (f y) (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HSub.hSub z y) (f x)) (HMul.hM …
  -/
  have hb : 0 < (y - x) / (z - x) := by positivity
  calc
    f y = f ((z - y) / (z - x) * x + (y - x) / (z - x) * z) := ?_
    _ < (z - y) / (z - x) * f x + (y - x) / (z - x) * f z := hf.2 hx hz (by linarith) ha hb ?_
    _ = ((z - y) * f x + (y - x) * f z) / (z - x) := ?_
    /-
      case calc_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LT.lt 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LT.lt 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (f y) (f (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x) …
    -/
  · congr 1
    /-
      case calc_1.e_a
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LT.lt 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LT.lt 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq y (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)) x) (H …
    -/
    field_simp
    /-
      case calc_1.e_a
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LT.lt 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LT.lt 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (HMul.hMul y (HSub.hSub z x)) (HAdd.hAdd (HMul.hMul (HSub.hSub z y) x) (H …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LT.lt 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LT.lt 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
    -/
  · field_simp
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      x y z : 𝕜
      hx : Membership.mem s x
      hz : Membership.mem s z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxy' : LT.lt 0 (HSub.hSub y x)
      hyz' : LT.lt 0 (HSub.hSub z y)
      hxz' : LT.lt 0 (HSub.hSub z x)
      ha : LT.lt 0 (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x))
      hb : LT.lt 0 (HDiv.hDiv (HSub.hSub y x) (HSub.hSub z x))
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub z y) (HSub.hSub z x)) (f x))  …
    -/
  · field_simp
    /-
      🎉 no goals
    -/


theorem StrictConvexOn.secant_strict_mono_aux2 (hf : StrictConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s)
    (hz : z ∈ s) (hxy : x < y) (hyz : y < z) : (f y - f x) / (y - x) < (f z - f x) / (z - x) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxy' : 0 < y - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz' : 0 < z - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
  -/
  rw [div_lt_div_iff₀ hxy' hxz']
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hxy' : LT.lt 0 (HSub.hSub y x)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LT.lt (HMul.hMul (HSub.hSub (f y) (f x)) (HSub.hSub z x)) (HMul.hMul (HSub.h …
  -/
  linarith only [hf.secant_strict_mono_aux1 hx hz hxy hyz]
  /-
    🎉 no goals
  -/


theorem StrictConvexOn.secant_strict_mono_aux3 (hf : StrictConvexOn 𝕜 s f) {x y z : 𝕜} (hx : x ∈ s)
    (hz : z ∈ s) (hxy : x < y) (hyz : y < z) : (f z - f x) / (z - x) < (f z - f y) / (z - y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z) (f x)) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
  -/
  have hyz' : 0 < z - y := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hyz' : LT.lt 0 (HSub.hSub z y)
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z) (f x)) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
  -/
  have hxz' : 0 < z - x := by linarith
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f z) (f x)) (HSub.hSub z x)) (HDiv.hDiv (HSub.h …
  -/
  rw [div_lt_div_iff₀ hxz' hyz']
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    x y z : 𝕜
    hx : Membership.mem s x
    hz : Membership.mem s z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hyz' : LT.lt 0 (HSub.hSub z y)
    hxz' : LT.lt 0 (HSub.hSub z x)
    ⊢ LT.lt (HMul.hMul (HSub.hSub (f z) (f x)) (HSub.hSub z y)) (HMul.hMul (HSub.h …
  -/
  linarith only [hf.secant_strict_mono_aux1 hx hz hxy hyz]
  /-
    🎉 no goals
  -/


/-- If `f : 𝕜 → 𝕜` is strictly convex, then for any point `a` the slope of the secant line of `f`
through `a` and `b` is strictly monotone with respect to `b`. -/
theorem StrictConvexOn.secant_strict_mono (hf : StrictConvexOn 𝕜 s f) {a x y : 𝕜} (ha : a ∈ s)
    (hx : x ∈ s) (hy : y ∈ s) (hxa : x ≠ a) (hya : y ≠ a) (hxy : x < y) :
    (f x - f a) / (x - a) < (f y - f a) / (y - a) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConvexOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy : LT.lt x y
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
  -/
  cases' lt_or_gt_of_ne hxa with hxa hxa
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      a x y : 𝕜
      ha : Membership.mem s a
      hx : Membership.mem s x
      hy : Membership.mem s y
      hxa✝ : Ne x a
      hya : Ne y a
      hxy : LT.lt x y
      hxa : LT.lt x a
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
    -/
  · cases' lt_or_gt_of_ne hya with hya hya
      /-
        case inl.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : StrictConvexOn 𝕜 s f
        a x y : 𝕜
        ha : Membership.mem s a
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxa✝ : Ne x a
        hya✝ : Ne y a
        hxy : LT.lt x y
        hxa : LT.lt x a
        hya : LT.lt y a
        ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
      -/
    · convert hf.secant_strict_mono_aux3 hx ha hxy hya using 1 <;> rw [← neg_div_neg_eq] <;>
        /-
          case h.e'_3
          𝕜 : Type u_1
          inst✝ : LinearOrderedField 𝕜
          s : Set 𝕜
          f : 𝕜 → 𝕜
          hf : StrictConvexOn 𝕜 s f
          a x y : 𝕜
          ha : Membership.mem s a
          hx : Membership.mem s x
          hy : Membership.mem s y
          hxa✝ : Ne x a
          hya✝ : Ne y a
          hxy : LT.lt x y
          hxa : LT.lt x a
          hya : LT.lt y a
          ⊢ Eq (HDiv.hDiv (Neg.neg (HSub.hSub (f x) (f a))) (Neg.neg (HSub.hSub x a))) ( …
        -/
        /-
          🎉 no goals
        -/
        field_simp
        /-
          🎉 no goals
        -/
      /-
        case inl.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : StrictConvexOn 𝕜 s f
        a x y : 𝕜
        ha : Membership.mem s a
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxa✝ : Ne x a
        hya✝ : Ne y a
        hxy : LT.lt x y
        hxa : LT.lt x a
        hya : GT.gt y a
        ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
      -/
    · convert hf.slope_strict_mono_adjacent hx hy hxa hya using 1
      /-
        case h.e'_3
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : StrictConvexOn 𝕜 s f
        a x y : 𝕜
        ha : Membership.mem s a
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxa✝ : Ne x a
        hya✝ : Ne y a
        hxy : LT.lt x y
        hxa : LT.lt x a
        hya : GT.gt y a
        ⊢ Eq (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.hSub …
      -/
      rw [← neg_div_neg_eq]; field_simp
                             /-
                               🎉 no goals
                             -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : StrictConvexOn 𝕜 s f
      a x y : 𝕜
      ha : Membership.mem s a
      hx : Membership.mem s x
      hy : Membership.mem s y
      hxa✝ : Ne x a
      hya : Ne y a
      hxy : LT.lt x y
      hxa : GT.gt x a
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a)) (HDiv.hDiv (HSub.h …
    -/
  · exact hf.secant_strict_mono_aux2 ha hy hxa hxy
    /-
      🎉 no goals
    -/


/-- If `f : 𝕜 → 𝕜` is strictly concave, then for any point `a` the slope of the secant line of `f`
through `a` and `b` is strictly antitone with respect to `b`. -/
theorem StrictConcaveOn.secant_strict_mono (hf : StrictConcaveOn 𝕜 s f) {a x y : 𝕜} (ha : a ∈ s)
    (hx : x ∈ s) (hy : y ∈ s) (hxa : x ≠ a) (hya : y ≠ a) (hxy : x < y) :
    (f y - f a) / (y - a) < (f x - f a) / (x - a) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy : LT.lt x y
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f a)) (HSub.hSub y a)) (HDiv.hDiv (HSub.h …
  -/
  have key := hf.neg.secant_strict_mono ha hx hy hxa hya hxy
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy : LT.lt x y
    key : LT.lt (HDiv.hDiv (HSub.hSub (Neg.neg f x) (Neg.neg f a)) (HSub.hSub x a) …
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f a)) (HSub.hSub y a)) (HDiv.hDiv (HSub.h …
  -/
  simp only [Pi.neg_apply] at key
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy : LT.lt x y
    key : LT.lt (HDiv.hDiv (HSub.hSub (Neg.neg (f x)) (Neg.neg (f a))) (HSub.hSub  …
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f a)) (HSub.hSub y a)) (HDiv.hDiv (HSub.h …
  -/
  rw [← neg_lt_neg_iff]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : StrictConcaveOn 𝕜 s f
    a x y : 𝕜
    ha : Membership.mem s a
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxa : Ne x a
    hya : Ne y a
    hxy : LT.lt x y
    key : LT.lt (HDiv.hDiv (HSub.hSub (Neg.neg (f x)) (Neg.neg (f a))) (HSub.hSub  …
    ⊢ LT.lt (Neg.neg (HDiv.hDiv (HSub.hSub (f x) (f a)) (HSub.hSub x a))) (Neg.neg …
  -/
                                         /-
                                           🎉 no goals
                                         -/
  convert key using 1 <;> field_simp <;> ring
                                         /-
                                           🎉 no goals
                                         -/


/-- If `f` is convex on a set `s` in a linearly ordered field, and `f x < f y` for two points
`x < y` in `s`, then `f` is strictly monotone on `s ∩ [y, ∞)`. -/
theorem ConvexOn.strict_mono_of_lt (hf : ConvexOn 𝕜 s f) {x y : 𝕜} (hx : x ∈ s) (hxy : x < y)
    (hxy' : f x < f y) : StrictMonoOn f (s ∩ Set.Ici y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y : 𝕜
    hx : Membership.mem s x
    hxy : LT.lt x y
    hxy' : LT.lt (f x) (f y)
    ⊢ StrictMonoOn f (Inter.inter s (Set.Ici y))
  -/
  intro u hu v hv huv
  have step1 : ∀ {z : 𝕜}, z ∈ s ∩ Set.Ioi y → f y < f z := by
    intros z hz
    refine hf.lt_right_of_left_lt hx hz.1 ?_ hxy'
    rw [openSegment_eq_Ioo (hxy.trans hz.2)]
    exact ⟨hxy, hz.2⟩
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    hf : ConvexOn 𝕜 s f
    x y : 𝕜
    hx : Membership.mem s x
    hxy : LT.lt x y
    hxy' : LT.lt (f x) (f y)
    u : 𝕜
    hu : Membership.mem (Inter.inter s (Set.Ici y)) u
    v : 𝕜
    hv : Membership.mem (Inter.inter s (Set.Ici y)) v
    huv : LT.lt u v
    step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
    ⊢ LT.lt (f u) (f v)
  -/
  rcases eq_or_lt_of_le hu.2 with (rfl | hu2)
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y : 𝕜
      hx : Membership.mem s x
      hxy : LT.lt x y
      hxy' : LT.lt (f x) (f y)
      v : 𝕜
      hv : Membership.mem (Inter.inter s (Set.Ici y)) v
      step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
      hu : Membership.mem (Inter.inter s (Set.Ici y)) y
      huv : LT.lt y v
      ⊢ LT.lt (f y) (f v)
    -/
  · exact step1 ⟨hv.1, huv⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      s : Set 𝕜
      f : 𝕜 → 𝕜
      hf : ConvexOn 𝕜 s f
      x y : 𝕜
      hx : Membership.mem s x
      hxy : LT.lt x y
      hxy' : LT.lt (f x) (f y)
      u : 𝕜
      hu : Membership.mem (Inter.inter s (Set.Ici y)) u
      v : 𝕜
      hv : Membership.mem (Inter.inter s (Set.Ici y)) v
      huv : LT.lt u v
      step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
      hu2 : LT.lt y u
      ⊢ LT.lt (f u) (f v)
    -/
  · refine hf.lt_right_of_left_lt ?_ hv.1 ?_ (step1 ⟨hu.1, hu2⟩)
      /-
        case inr.refine_1
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        x y : 𝕜
        hx : Membership.mem s x
        hxy : LT.lt x y
        hxy' : LT.lt (f x) (f y)
        u : 𝕜
        hu : Membership.mem (Inter.inter s (Set.Ici y)) u
        v : 𝕜
        hv : Membership.mem (Inter.inter s (Set.Ici y)) v
        huv : LT.lt u v
        step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
        hu2 : LT.lt y u
        ⊢ Membership.mem s y
      -/
    · apply hf.1.segment_subset hx hu.1
      /-
        case inr.refine_1.a
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        x y : 𝕜
        hx : Membership.mem s x
        hxy : LT.lt x y
        hxy' : LT.lt (f x) (f y)
        u : 𝕜
        hu : Membership.mem (Inter.inter s (Set.Ici y)) u
        v : 𝕜
        hv : Membership.mem (Inter.inter s (Set.Ici y)) v
        huv : LT.lt u v
        step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
        hu2 : LT.lt y u
        ⊢ Membership.mem (segment 𝕜 x u) y
      -/
      rw [segment_eq_Icc (hxy.le.trans hu.2)]
      /-
        case inr.refine_1.a
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        x y : 𝕜
        hx : Membership.mem s x
        hxy : LT.lt x y
        hxy' : LT.lt (f x) (f y)
        u : 𝕜
        hu : Membership.mem (Inter.inter s (Set.Ici y)) u
        v : 𝕜
        hv : Membership.mem (Inter.inter s (Set.Ici y)) v
        huv : LT.lt u v
        step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
        hu2 : LT.lt y u
        ⊢ Membership.mem (Set.Icc x u) y
      -/
      exact ⟨hxy.le, hu.2⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        x y : 𝕜
        hx : Membership.mem s x
        hxy : LT.lt x y
        hxy' : LT.lt (f x) (f y)
        u : 𝕜
        hu : Membership.mem (Inter.inter s (Set.Ici y)) u
        v : 𝕜
        hv : Membership.mem (Inter.inter s (Set.Ici y)) v
        huv : LT.lt u v
        step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
        hu2 : LT.lt y u
        ⊢ Membership.mem (openSegment 𝕜 y v) u
      -/
    · rw [openSegment_eq_Ioo (hu2.trans huv)]
      /-
        case inr.refine_2
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        s : Set 𝕜
        f : 𝕜 → 𝕜
        hf : ConvexOn 𝕜 s f
        x y : 𝕜
        hx : Membership.mem s x
        hxy : LT.lt x y
        hxy' : LT.lt (f x) (f y)
        u : 𝕜
        hu : Membership.mem (Inter.inter s (Set.Ici y)) u
        v : 𝕜
        hv : Membership.mem (Inter.inter s (Set.Ici y)) v
        huv : LT.lt u v
        step1 : ∀ {z : 𝕜}, Membership.mem (Inter.inter s (Set.Ioi y)) z → LT.lt (f y)  …
        hu2 : LT.lt y u
        ⊢ Membership.mem (Set.Ioo y v) u
      -/
      exact ⟨hu2, huv⟩
      /-
        🎉 no goals
      -/

