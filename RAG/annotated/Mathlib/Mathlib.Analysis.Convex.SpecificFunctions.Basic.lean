/-- `Real.exp` is strictly convex on the whole real line. -/
theorem strictConvexOn_exp : StrictConvexOn ℝ univ exp := by
  /-
    ⊢ StrictConvexOn Real Set.univ Real.exp
  -/
  apply strictConvexOn_of_slope_strict_mono_adjacent convex_univ
  /-
    ⊢ ∀ {x y z : Real}, Membership.mem Set.univ x → Membership.mem Set.univ z → LT …
  -/
  rintro x y z - - hxy hyz
  /-
    x y z : Real
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.exp y) (Real.exp x)) (HSub.hSub y x)) (HDi …
  -/
  trans exp y
    /-
      x y z : Real
      hxy : LT.lt x y
      hyz : LT.lt y z
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.exp y) (Real.exp x)) (HSub.hSub y x)) (Rea …
    -/
  · have h1 : 0 < y - x := by linarith
    /-
      x y z : Real
      hxy : LT.lt x y
      hyz : LT.lt y z
      h1 : LT.lt 0 (HSub.hSub y x)
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.exp y) (Real.exp x)) (HSub.hSub y x)) (Rea …
    -/
    have h2 : x - y < 0 := by linarith
    /-
      x y z : Real
      hxy : LT.lt x y
      hyz : LT.lt y z
      h1 : LT.lt 0 (HSub.hSub y x)
      h2 : LT.lt (HSub.hSub x y) 0
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.exp y) (Real.exp x)) (HSub.hSub y x)) (Rea …
    -/
    rw [div_lt_iff₀ h1]
    calc
      exp y - exp x = exp y - exp y * exp (x - y) := by rw [← exp_add]; ring_nf
      _ = exp y * (1 - exp (x - y)) := by ring
      _ < exp y * -(x - y) := by gcongr; linarith [add_one_lt_exp h2.ne]
      _ = exp y * (y - x) := by ring
    /-
      x y z : Real
      hxy : LT.lt x y
      hyz : LT.lt y z
      ⊢ LT.lt (Real.exp y) (HDiv.hDiv (HSub.hSub (Real.exp z) (Real.exp y)) (HSub.hS …
    -/
  · have h1 : 0 < z - y := by linarith
    /-
      x y z : Real
      hxy : LT.lt x y
      hyz : LT.lt y z
      h1 : LT.lt 0 (HSub.hSub z y)
      ⊢ LT.lt (Real.exp y) (HDiv.hDiv (HSub.hSub (Real.exp z) (Real.exp y)) (HSub.hS …
    -/
    rw [lt_div_iff₀ h1]
    calc
      exp y * (z - y) < exp y * (exp (z - y) - 1) := by
        gcongr _ * ?_
        linarith [add_one_lt_exp h1.ne']
      _ = exp (z - y) * exp y - exp y := by ring
      _ ≤ exp z - exp y := by rw [← exp_add]; ring_nf; rfl


/-- `Real.exp` is convex on the whole real line. -/
theorem convexOn_exp : ConvexOn ℝ univ exp :=
  strictConvexOn_exp.convexOn


/-- `Real.log` is strictly concave on `(0, +∞)`. -/
theorem strictConcaveOn_log_Ioi : StrictConcaveOn ℝ (Ioi 0) log := by
  /-
    ⊢ StrictConcaveOn Real (Set.Ioi 0) Real.log
  -/
  apply strictConcaveOn_of_slope_strict_anti_adjacent (convex_Ioi (0 : ℝ))
  /-
    ⊢ ∀ {x y z : Real}, Membership.mem (Set.Ioi 0) x → Membership.mem (Set.Ioi 0)  …
  -/
  intro x y z (hx : 0 < x) (hz : 0 < z) hxy hyz
  /-
    x y z : Real
    hx : LT.lt 0 x
    hz : LT.lt 0 z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.log z) (Real.log y)) (HSub.hSub z y)) (HDi …
  -/
  have hy : 0 < y := hx.trans hxy
  /-
    x y z : Real
    hx : LT.lt 0 x
    hz : LT.lt 0 z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hy : LT.lt 0 y
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.log z) (Real.log y)) (HSub.hSub z y)) (HDi …
  -/
  trans y⁻¹
    /-
      x y z : Real
      hx : LT.lt 0 x
      hz : LT.lt 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.log z) (Real.log y)) (HSub.hSub z y)) (Inv …
    -/
  · have h : 0 < z - y := by linarith
    /-
      x y z : Real
      hx : LT.lt 0 x
      hz : LT.lt 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      h : LT.lt 0 (HSub.hSub z y)
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (Real.log z) (Real.log y)) (HSub.hSub z y)) (Inv …
    -/
    rw [div_lt_iff₀ h]
    /-
      x y z : Real
      hx : LT.lt 0 x
      hz : LT.lt 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      h : LT.lt 0 (HSub.hSub z y)
      ⊢ LT.lt (HSub.hSub (Real.log z) (Real.log y)) (HMul.hMul (Inv.inv y) (HSub.hSu …
    -/
    have hyz' : 0 < z / y := by positivity
    have hyz'' : z / y ≠ 1 := by
      contrapose! h
      rw [div_eq_one_iff_eq hy.ne'] at h
      simp [h]
    calc
      log z - log y = log (z / y) := by rw [← log_div hz.ne' hy.ne']
      _ < z / y - 1 := log_lt_sub_one_of_pos hyz' hyz''
      _ = y⁻¹ * (z - y) := by field_simp
    /-
      x y z : Real
      hx : LT.lt 0 x
      hz : LT.lt 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      ⊢ LT.lt (Inv.inv y) (HDiv.hDiv (HSub.hSub (Real.log y) (Real.log x)) (HSub.hSu …
    -/
  · have h : 0 < y - x := by linarith
    /-
      x y z : Real
      hx : LT.lt 0 x
      hz : LT.lt 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      h : LT.lt 0 (HSub.hSub y x)
      ⊢ LT.lt (Inv.inv y) (HDiv.hDiv (HSub.hSub (Real.log y) (Real.log x)) (HSub.hSu …
    -/
    rw [lt_div_iff₀ h]
    /-
      x y z : Real
      hx : LT.lt 0 x
      hz : LT.lt 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      h : LT.lt 0 (HSub.hSub y x)
      ⊢ LT.lt (HMul.hMul (Inv.inv y) (HSub.hSub y x)) (HSub.hSub (Real.log y) (Real. …
    -/
    have hxy' : 0 < x / y := by positivity
    have hxy'' : x / y ≠ 1 := by
      contrapose! h
      rw [div_eq_one_iff_eq hy.ne'] at h
      simp [h]
    calc
      y⁻¹ * (y - x) = 1 - x / y := by field_simp
      _ < -log (x / y) := by linarith [log_lt_sub_one_of_pos hxy' hxy'']
      _ = -(log x - log y) := by rw [log_div hx.ne' hy.ne']
      _ = log y - log x := by ring


/-- **Bernoulli's inequality** for real exponents, strict version: for `1 < p` and `-1 ≤ s`, with
`s ≠ 0`, we have `1 + p * s < (1 + s) ^ p`. -/
theorem one_add_mul_self_lt_rpow_one_add {s : ℝ} (hs : -1 ≤ s) (hs' : s ≠ 0) {p : ℝ} (hp : 1 < p) :
    1 + p * s < (1 + s) ^ p := by
  /-
    s : Real
    hs : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  have hp' : 0 < p := zero_lt_one.trans hp
  /-
    s : Real
    hs : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  rcases eq_or_lt_of_le hs with rfl | hs
    /-
      case inl
      p : Real
      hp : LT.lt 1 p
      hp' : LT.lt 0 p
      hs : LE.le (-1) (-1)
      hs' : Ne (-1) 0
      ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p (-1))) (HPow.hPow (HAdd.hAdd 1 (-1)) p)
    -/
  · rwa [add_neg_cancel, zero_rpow hp'.ne', mul_neg_one, add_neg_lt_iff_lt_add, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    hs : LT.lt (-1) s
    ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  have hs1 : 0 < 1 + s := neg_lt_iff_pos_add'.mp hs
  /-
    case inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  rcases le_or_lt (1 + p * s) 0 with hs2 | hs2
    /-
      case inr.inl
      s : Real
      hs✝ : LE.le (-1) s
      hs' : Ne s 0
      p : Real
      hp : LT.lt 1 p
      hp' : LT.lt 0 p
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LE.le (HAdd.hAdd 1 (HMul.hMul p s)) 0
      ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
    -/
  · exact hs2.trans_lt (rpow_pos_of_pos hs1 _)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  have hs3 : 1 + s ≠ 1 := hs' ∘ add_right_eq_self.mp
  have hs4 : 1 + p * s ≠ 1 := by
    contrapose! hs'; rwa [add_right_eq_self, mul_eq_zero, eq_false_intro hp'.ne', false_or] at hs'
  /-
    case inr.inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    hs3 : Ne (HAdd.hAdd 1 s) 1
    hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
    ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  rw [rpow_def_of_pos hs1, ← exp_log hs2]
  /-
    case inr.inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    hs3 : Ne (HAdd.hAdd 1 s) 1
    hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
    ⊢ LT.lt (Real.exp (Real.log (HAdd.hAdd 1 (HMul.hMul p s)))) (Real.exp (HMul.hM …
  -/
  apply exp_strictMono
  /-
    case inr.inr.a
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp : LT.lt 1 p
    hp' : LT.lt 0 p
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    hs3 : Ne (HAdd.hAdd 1 s) 1
    hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
    ⊢ LT.lt (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) (HMul.hMul (Real.log (HAdd.hA …
  -/
  cases' lt_or_gt_of_ne hs' with hs' hs'
    /-
      case inr.inr.a.inl
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp : LT.lt 1 p
      hp' : LT.lt 0 p
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : LT.lt s 0
      ⊢ LT.lt (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) (HMul.hMul (Real.log (HAdd.hA …
    -/
  · rw [← div_lt_iff₀ hp', ← div_lt_div_right_of_neg hs']
    /-
      case inr.inr.a.inl
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp : LT.lt 1 p
      hp' : LT.lt 0 p
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : LT.lt s 0
      ⊢ LT.lt (HDiv.hDiv (Real.log (HAdd.hAdd 1 s)) s) (HDiv.hDiv (HDiv.hDiv (Real.l …
    -/
    convert strictConcaveOn_log_Ioi.secant_strict_mono (zero_lt_one' ℝ) hs2 hs1 hs4 hs3 _ using 1
      /-
        case h.e'_3
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp : LT.lt 1 p
        hp' : LT.lt 0 p
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : LT.lt s 0
        ⊢ Eq (HDiv.hDiv (Real.log (HAdd.hAdd 1 s)) s) (HDiv.hDiv (HSub.hSub (Real.log  …
      -/
    · rw [add_sub_cancel_left, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp : LT.lt 1 p
        hp' : LT.lt 0 p
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : LT.lt s 0
        ⊢ Eq (HDiv.hDiv (HDiv.hDiv (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) p) s) (HDi …
      -/
    · rw [add_sub_cancel_left, div_div, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.a.inl
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp : LT.lt 1 p
        hp' : LT.lt 0 p
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : LT.lt s 0
        ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HAdd.hAdd 1 s)
      -/
    · apply add_lt_add_left (mul_lt_of_one_lt_left hs' hp)
      /-
        🎉 no goals
      -/
    /-
      case inr.inr.a.inr
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp : LT.lt 1 p
      hp' : LT.lt 0 p
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : GT.gt s 0
      ⊢ LT.lt (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) (HMul.hMul (Real.log (HAdd.hA …
    -/
  · rw [← div_lt_iff₀ hp', ← div_lt_div_iff_of_pos_right hs']
    /-
      case inr.inr.a.inr
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp : LT.lt 1 p
      hp' : LT.lt 0 p
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : GT.gt s 0
      ⊢ LT.lt (HDiv.hDiv (HDiv.hDiv (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) p) s) ( …
    -/
    convert strictConcaveOn_log_Ioi.secant_strict_mono (zero_lt_one' ℝ) hs1 hs2 hs3 hs4 _ using 1
      /-
        case h.e'_3
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp : LT.lt 1 p
        hp' : LT.lt 0 p
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : GT.gt s 0
        ⊢ Eq (HDiv.hDiv (HDiv.hDiv (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) p) s) (HDi …
      -/
    · rw [add_sub_cancel_left, div_div, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp : LT.lt 1 p
        hp' : LT.lt 0 p
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : GT.gt s 0
        ⊢ Eq (HDiv.hDiv (Real.log (HAdd.hAdd 1 s)) s) (HDiv.hDiv (HSub.hSub (Real.log  …
      -/
    · rw [add_sub_cancel_left, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.a.inr
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp : LT.lt 1 p
        hp' : LT.lt 0 p
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : GT.gt s 0
        ⊢ LT.lt (HAdd.hAdd 1 s) (HAdd.hAdd 1 (HMul.hMul p s))
      -/
    · apply add_lt_add_left (lt_mul_of_one_lt_left hs' hp)
      /-
        🎉 no goals
      -/


/-- **Bernoulli's inequality** for real exponents, non-strict version: for `1 ≤ p` and `-1 ≤ s`
we have `1 + p * s ≤ (1 + s) ^ p`. -/
theorem one_add_mul_self_le_rpow_one_add {s : ℝ} (hs : -1 ≤ s) {p : ℝ} (hp : 1 ≤ p) :
    1 + p * s ≤ (1 + s) ^ p := by
  /-
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  rcases eq_or_lt_of_le hp with (rfl | hp)
    /-
      case inl
      s : Real
      hs : LE.le (-1) s
      hp : LE.le 1 1
      ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul 1 s)) (HPow.hPow (HAdd.hAdd 1 s) 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  by_cases hs' : s = 0
    /-
      case pos
      s : Real
      hs : LE.le (-1) s
      p : Real
      hp✝ : LE.le 1 p
      hp : LT.lt 1 p
      hs' : Eq s 0
      ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
    -/
  · simp [hs']
    /-
      🎉 no goals
    -/
  /-
    case neg
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    hs' : Not (Eq s 0)
    ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul p s)) (HPow.hPow (HAdd.hAdd 1 s) p)
  -/
  exact (one_add_mul_self_lt_rpow_one_add hs hs' hp).le
  /-
    🎉 no goals
  -/


/-- **Bernoulli's inequality** for real exponents, strict version: for `0 < p < 1` and `-1 ≤ s`,
with `s ≠ 0`, we have `(1 + s) ^ p < 1 + p * s`. -/
theorem rpow_one_add_lt_one_add_mul_self {s : ℝ} (hs : -1 ≤ s) (hs' : s ≠ 0) {p : ℝ} (hp1 : 0 < p)
    (hp2 : p < 1) : (1 + s) ^ p < 1 + p * s := by
  /-
    s : Real
    hs : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    ⊢ LT.lt (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  rcases eq_or_lt_of_le hs with rfl | hs
    /-
      case inl
      p : Real
      hp1 : LT.lt 0 p
      hp2 : LT.lt p 1
      hs : LE.le (-1) (-1)
      hs' : Ne (-1) 0
      ⊢ LT.lt (HPow.hPow (HAdd.hAdd 1 (-1)) p) (HAdd.hAdd 1 (HMul.hMul p (-1)))
    -/
  · rwa [add_neg_cancel, zero_rpow hp1.ne', mul_neg_one, lt_add_neg_iff_add_lt, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    hs : LT.lt (-1) s
    ⊢ LT.lt (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  have hs1 : 0 < 1 + s := neg_lt_iff_pos_add'.mp hs
  have hs2 : 0 < 1 + p * s := by
    rw [← neg_lt_iff_pos_add']
    rcases lt_or_gt_of_ne hs' with h | h
    · exact hs.trans (lt_mul_of_lt_one_left h hp2)
    · exact neg_one_lt_zero.trans (mul_pos hp1 h)
  /-
    case inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    ⊢ LT.lt (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  have hs3 : 1 + s ≠ 1 := hs' ∘ add_right_eq_self.mp
  have hs4 : 1 + p * s ≠ 1 := by
    contrapose! hs'; rwa [add_right_eq_self, mul_eq_zero, eq_false_intro hp1.ne', false_or] at hs'
  /-
    case inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    hs3 : Ne (HAdd.hAdd 1 s) 1
    hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
    ⊢ LT.lt (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  rw [rpow_def_of_pos hs1, ← exp_log hs2]
  /-
    case inr
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    hs3 : Ne (HAdd.hAdd 1 s) 1
    hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
    ⊢ LT.lt (Real.exp (HMul.hMul (Real.log (HAdd.hAdd 1 s)) p)) (Real.exp (Real.lo …
  -/
  apply exp_strictMono
  /-
    case inr.a
    s : Real
    hs✝ : LE.le (-1) s
    hs' : Ne s 0
    p : Real
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    hs : LT.lt (-1) s
    hs1 : LT.lt 0 (HAdd.hAdd 1 s)
    hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
    hs3 : Ne (HAdd.hAdd 1 s) 1
    hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
    ⊢ LT.lt (HMul.hMul (Real.log (HAdd.hAdd 1 s)) p) (Real.log (HAdd.hAdd 1 (HMul. …
  -/
  cases' lt_or_gt_of_ne hs' with hs' hs'
    /-
      case inr.a.inl
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp1 : LT.lt 0 p
      hp2 : LT.lt p 1
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : LT.lt s 0
      ⊢ LT.lt (HMul.hMul (Real.log (HAdd.hAdd 1 s)) p) (Real.log (HAdd.hAdd 1 (HMul. …
    -/
  · rw [← lt_div_iff₀ hp1, ← div_lt_div_right_of_neg hs']
    /-
      case inr.a.inl
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp1 : LT.lt 0 p
      hp2 : LT.lt p 1
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : LT.lt s 0
      ⊢ LT.lt (HDiv.hDiv (HDiv.hDiv (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) p) s) ( …
    -/
    convert strictConcaveOn_log_Ioi.secant_strict_mono (zero_lt_one' ℝ) hs1 hs2 hs3 hs4 _ using 1
      /-
        case h.e'_3
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp1 : LT.lt 0 p
        hp2 : LT.lt p 1
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : LT.lt s 0
        ⊢ Eq (HDiv.hDiv (HDiv.hDiv (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) p) s) (HDi …
      -/
    · rw [add_sub_cancel_left, div_div, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp1 : LT.lt 0 p
        hp2 : LT.lt p 1
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : LT.lt s 0
        ⊢ Eq (HDiv.hDiv (Real.log (HAdd.hAdd 1 s)) s) (HDiv.hDiv (HSub.hSub (Real.log  …
      -/
    · rw [add_sub_cancel_left, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.a.inl
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp1 : LT.lt 0 p
        hp2 : LT.lt p 1
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : LT.lt s 0
        ⊢ LT.lt (HAdd.hAdd 1 s) (HAdd.hAdd 1 (HMul.hMul p s))
      -/
    · apply add_lt_add_left (lt_mul_of_lt_one_left hs' hp2)
      /-
        🎉 no goals
      -/
    /-
      case inr.a.inr
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp1 : LT.lt 0 p
      hp2 : LT.lt p 1
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : GT.gt s 0
      ⊢ LT.lt (HMul.hMul (Real.log (HAdd.hAdd 1 s)) p) (Real.log (HAdd.hAdd 1 (HMul. …
    -/
  · rw [← lt_div_iff₀ hp1, ← div_lt_div_iff_of_pos_right hs']
    /-
      case inr.a.inr
      s : Real
      hs✝ : LE.le (-1) s
      hs'✝ : Ne s 0
      p : Real
      hp1 : LT.lt 0 p
      hp2 : LT.lt p 1
      hs : LT.lt (-1) s
      hs1 : LT.lt 0 (HAdd.hAdd 1 s)
      hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
      hs3 : Ne (HAdd.hAdd 1 s) 1
      hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
      hs' : GT.gt s 0
      ⊢ LT.lt (HDiv.hDiv (Real.log (HAdd.hAdd 1 s)) s) (HDiv.hDiv (HDiv.hDiv (Real.l …
    -/
    convert strictConcaveOn_log_Ioi.secant_strict_mono (zero_lt_one' ℝ) hs2 hs1 hs4 hs3 _ using 1
      /-
        case h.e'_3
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp1 : LT.lt 0 p
        hp2 : LT.lt p 1
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : GT.gt s 0
        ⊢ Eq (HDiv.hDiv (Real.log (HAdd.hAdd 1 s)) s) (HDiv.hDiv (HSub.hSub (Real.log  …
      -/
    · rw [add_sub_cancel_left, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp1 : LT.lt 0 p
        hp2 : LT.lt p 1
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : GT.gt s 0
        ⊢ Eq (HDiv.hDiv (HDiv.hDiv (Real.log (HAdd.hAdd 1 (HMul.hMul p s))) p) s) (HDi …
      -/
    · rw [add_sub_cancel_left, div_div, log_one, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case inr.a.inr
        s : Real
        hs✝ : LE.le (-1) s
        hs'✝ : Ne s 0
        p : Real
        hp1 : LT.lt 0 p
        hp2 : LT.lt p 1
        hs : LT.lt (-1) s
        hs1 : LT.lt 0 (HAdd.hAdd 1 s)
        hs2 : LT.lt 0 (HAdd.hAdd 1 (HMul.hMul p s))
        hs3 : Ne (HAdd.hAdd 1 s) 1
        hs4 : Ne (HAdd.hAdd 1 (HMul.hMul p s)) 1
        hs' : GT.gt s 0
        ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p s)) (HAdd.hAdd 1 s)
      -/
    · apply add_lt_add_left (mul_lt_of_lt_one_left hs' hp2)
      /-
        🎉 no goals
      -/


/-- **Bernoulli's inequality** for real exponents, non-strict version: for `0 ≤ p ≤ 1` and `-1 ≤ s`
we have `(1 + s) ^ p ≤ 1 + p * s`. -/
theorem rpow_one_add_le_one_add_mul_self {s : ℝ} (hs : -1 ≤ s) {p : ℝ} (hp1 : 0 ≤ p) (hp2 : p ≤ 1) :
    (1 + s) ^ p ≤ 1 + p * s := by
  /-
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp1 : LE.le 0 p
    hp2 : LE.le p 1
    ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  rcases eq_or_lt_of_le hp1 with (rfl | hp1)
    /-
      case inl
      s : Real
      hs : LE.le (-1) s
      hp1 : LE.le 0 0
      hp2 : LE.le 0 1
      ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) 0) (HAdd.hAdd 1 (HMul.hMul 0 s))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp1✝ : LE.le 0 p
    hp2 : LE.le p 1
    hp1 : LT.lt 0 p
    ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  rcases eq_or_lt_of_le hp2 with (rfl | hp2)
    /-
      case inr.inl
      s : Real
      hs : LE.le (-1) s
      hp1✝ : LE.le 0 1
      hp2 : LE.le 1 1
      hp1 : LT.lt 0 1
      ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) 1) (HAdd.hAdd 1 (HMul.hMul 1 s))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp1✝ : LE.le 0 p
    hp2✝ : LE.le p 1
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  by_cases hs' : s = 0
    /-
      case pos
      s : Real
      hs : LE.le (-1) s
      p : Real
      hp1✝ : LE.le 0 p
      hp2✝ : LE.le p 1
      hp1 : LT.lt 0 p
      hp2 : LT.lt p 1
      hs' : Eq s 0
      ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
    -/
  · simp [hs']
    /-
      🎉 no goals
    -/
  /-
    case neg
    s : Real
    hs : LE.le (-1) s
    p : Real
    hp1✝ : LE.le 0 p
    hp2✝ : LE.le p 1
    hp1 : LT.lt 0 p
    hp2 : LT.lt p 1
    hs' : Not (Eq s 0)
    ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 s) p) (HAdd.hAdd 1 (HMul.hMul p s))
  -/
  exact (rpow_one_add_lt_one_add_mul_self hs hs' hp1 hp2).le
  /-
    🎉 no goals
  -/


/-- For `p : ℝ` with `1 < p`, `fun x ↦ x ^ p` is strictly convex on $[0, +∞)$. -/
theorem strictConvexOn_rpow {p : ℝ} (hp : 1 < p) : StrictConvexOn ℝ (Ici 0) fun x : ℝ ↦ x ^ p := by
  /-
    p : Real
    hp : LT.lt 1 p
    ⊢ StrictConvexOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  apply strictConvexOn_of_slope_strict_mono_adjacent (convex_Ici (0 : ℝ))
  /-
    p : Real
    hp : LT.lt 1 p
    ⊢ ∀ {x y z : Real}, Membership.mem (Set.Ici 0) x → Membership.mem (Set.Ici 0)  …
  -/
  intro x y z (hx : 0 ≤ x) (hz : 0 ≤ z) hxy hyz
  /-
    p : Real
    hp : LT.lt 1 p
    x y z : Real
    hx : LE.le 0 x
    hz : LE.le 0 z
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HPow.hPow y p) (HPow.hPow x p)) (HSub.hSub y x) …
  -/
  have hy : 0 < y := hx.trans_lt hxy
  /-
    p : Real
    hp : LT.lt 1 p
    x y z : Real
    hx : LE.le 0 x
    hz : LE.le 0 z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hy : LT.lt 0 y
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HPow.hPow y p) (HPow.hPow x p)) (HSub.hSub y x) …
  -/
  have hy' : 0 < y ^ p := rpow_pos_of_pos hy _
  /-
    p : Real
    hp : LT.lt 1 p
    x y z : Real
    hx : LE.le 0 x
    hz : LE.le 0 z
    hxy : LT.lt x y
    hyz : LT.lt y z
    hy : LT.lt 0 y
    hy' : LT.lt 0 (HPow.hPow y p)
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HPow.hPow y p) (HPow.hPow x p)) (HSub.hSub y x) …
  -/
  trans p * y ^ (p - 1)
    /-
      p : Real
      hp : LT.lt 1 p
      x y z : Real
      hx : LE.le 0 x
      hz : LE.le 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      hy' : LT.lt 0 (HPow.hPow y p)
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HPow.hPow y p) (HPow.hPow x p)) (HSub.hSub y x) …
    -/
  · have q : 0 < y - x := by rwa [sub_pos]
    rw [div_lt_iff₀ q, ← div_lt_div_iff_of_pos_right hy', _root_.sub_div, div_self hy'.ne',
      ← div_rpow hx hy.le, sub_lt_comm, ← add_sub_cancel_right (x / y) 1, add_comm, add_sub_assoc,
      ← div_mul_eq_mul_div, mul_div_assoc, ← rpow_sub hy, sub_sub_cancel_left, rpow_neg_one,
      mul_assoc, ← div_eq_inv_mul, sub_eq_add_neg, ← mul_neg, ← neg_div, neg_sub, _root_.sub_div,
      div_self hy.ne']
    /-
      p : Real
      hp : LT.lt 1 p
      x y z : Real
      hx : LE.le 0 x
      hz : LE.le 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      hy' : LT.lt 0 (HPow.hPow y p)
      q : LT.lt 0 (HSub.hSub y x)
      ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p (HSub.hSub (HDiv.hDiv x y) 1))) (HPow.hPow ( …
    -/
    apply one_add_mul_self_lt_rpow_one_add _ _ hp
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub y x)
        ⊢ LE.le (-1) (HSub.hSub (HDiv.hDiv x y) 1)
      -/
    · rw [le_sub_iff_add_le, neg_add_cancel, div_nonneg_iff]
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub y x)
        ⊢ Or (And (LE.le 0 x) (LE.le 0 y)) (And (LE.le x 0) (LE.le y 0))
      -/
      exact Or.inl ⟨hx, hy.le⟩
      /-
        🎉 no goals
      -/
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub y x)
        ⊢ Ne (HSub.hSub (HDiv.hDiv x y) 1) 0
      -/
    · rw [sub_ne_zero]
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub y x)
        ⊢ Ne (HDiv.hDiv x y) 1
      -/
      exact ((div_lt_one hy).mpr hxy).ne
      /-
        🎉 no goals
      -/
    /-
      p : Real
      hp : LT.lt 1 p
      x y z : Real
      hx : LE.le 0 x
      hz : LE.le 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      hy' : LT.lt 0 (HPow.hPow y p)
      ⊢ LT.lt (HMul.hMul p (HPow.hPow y (HSub.hSub p 1))) (HDiv.hDiv (HSub.hSub (HPo …
    -/
  · have q : 0 < z - y := by rwa [sub_pos]
    rw [lt_div_iff₀ q, ← div_lt_div_iff_of_pos_right hy', _root_.sub_div, div_self hy'.ne',
      ← div_rpow hz hy.le, lt_sub_iff_add_lt', ← add_sub_cancel_right (z / y) 1, add_comm _ 1,
      add_sub_assoc, ← div_mul_eq_mul_div, mul_div_assoc, ← rpow_sub hy, sub_sub_cancel_left,
      rpow_neg_one, mul_assoc, ← div_eq_inv_mul, _root_.sub_div, div_self hy.ne']
    /-
      p : Real
      hp : LT.lt 1 p
      x y z : Real
      hx : LE.le 0 x
      hz : LE.le 0 z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hy : LT.lt 0 y
      hy' : LT.lt 0 (HPow.hPow y p)
      q : LT.lt 0 (HSub.hSub z y)
      ⊢ LT.lt (HAdd.hAdd 1 (HMul.hMul p (HSub.hSub (HDiv.hDiv z y) 1))) (HPow.hPow ( …
    -/
    apply one_add_mul_self_lt_rpow_one_add _ _ hp
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub z y)
        ⊢ LE.le (-1) (HSub.hSub (HDiv.hDiv z y) 1)
      -/
    · rw [le_sub_iff_add_le, neg_add_cancel, div_nonneg_iff]
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub z y)
        ⊢ Or (And (LE.le 0 z) (LE.le 0 y)) (And (LE.le z 0) (LE.le y 0))
      -/
      exact Or.inl ⟨hz, hy.le⟩
      /-
        🎉 no goals
      -/
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub z y)
        ⊢ Ne (HSub.hSub (HDiv.hDiv z y) 1) 0
      -/
    · rw [sub_ne_zero]
      /-
        p : Real
        hp : LT.lt 1 p
        x y z : Real
        hx : LE.le 0 x
        hz : LE.le 0 z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hy : LT.lt 0 y
        hy' : LT.lt 0 (HPow.hPow y p)
        q : LT.lt 0 (HSub.hSub z y)
        ⊢ Ne (HDiv.hDiv z y) 1
      -/
      exact ((one_lt_div hy).mpr hyz).ne'
      /-
        🎉 no goals
      -/


theorem convexOn_rpow {p : ℝ} (hp : 1 ≤ p) : ConvexOn ℝ (Ici 0) fun x : ℝ ↦ x ^ p := by
  /-
    p : Real
    hp : LE.le 1 p
    ⊢ ConvexOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  rcases eq_or_lt_of_le hp with (rfl | hp)
    /-
      case inl
      hp : LE.le 1 1
      ⊢ ConvexOn Real (Set.Ici 0) fun x => HPow.hPow x 1
    -/
  · simpa using convexOn_id (convex_Ici _)
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    ⊢ ConvexOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  exact (strictConvexOn_rpow hp).convexOn
  /-
    🎉 no goals
  -/


theorem strictConcaveOn_log_Iio : StrictConcaveOn ℝ (Iio 0) log := by
  /-
    ⊢ StrictConcaveOn Real (Set.Iio 0) Real.log
  -/
  refine ⟨convex_Iio _, ?_⟩
  /-
    ⊢ ∀ ⦃x : Real⦄, Membership.mem (Set.Iio 0) x → ∀ ⦃y : Real⦄, Membership.mem (S …
  -/
  intro x (hx : x < 0) y (hy : y < 0) hxy a b ha hb hab
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (Real.log x)) (HSMul.hSMul b (Real.log y)))  …
  -/
  have hx' : 0 < -x := by linarith
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hx' : LT.lt 0 (Neg.neg x)
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (Real.log x)) (HSMul.hSMul b (Real.log y)))  …
  -/
  have hy' : 0 < -y := by linarith
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    hy : LT.lt y 0
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hx' : LT.lt 0 (Neg.neg x)
    hy' : LT.lt 0 (Neg.neg y)
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (Real.log x)) (HSMul.hSMul b (Real.log y)))  …
  -/
  have hxy' : -x ≠ -y := by contrapose! hxy; linarith
  calc
    a • log x + b • log y = a • log (-x) + b • log (-y) := by simp_rw [log_neg_eq_log]
    _ < log (a • -x + b • -y) := strictConcaveOn_log_Ioi.2 hx' hy' hxy' ha hb hab
    _ = log (-(a • x + b • y)) := by congr 1; simp only [Algebra.id.smul_eq_mul]; ring
    _ = _ := by rw [log_neg_eq_log]


lemma exp_mul_le_cosh_add_mul_sinh {t : ℝ} (ht : |t| ≤ 1) (x : ℝ) :
    exp (t * x) ≤ cosh x + t * sinh x := by
  /-
    t : Real
    ht : LE.le (abs t) 1
    x : Real
    ⊢ LE.le (Real.exp (HMul.hMul t x)) (HAdd.hAdd (Real.cosh x) (HMul.hMul t (Real …
  -/
  rw [abs_le] at ht
  calc
    _ = exp ((1 + t) / 2 * x + (1 - t) / 2 * (-x)) := by ring_nf
    _ ≤ (1 + t) / 2 * exp x + (1 - t) / 2 * exp (-x) :=
        convexOn_exp.2 (Set.mem_univ _) (Set.mem_univ _) (by linarith) (by linarith) <| by ring
    _ = _ := by rw [cosh_eq, sinh_eq]; ring


