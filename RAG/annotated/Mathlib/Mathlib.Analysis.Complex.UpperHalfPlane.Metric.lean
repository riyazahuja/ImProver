instance : Dist ℍ :=
  ⟨fun z w => 2 * arsinh (dist (z : ℂ) w / (2 * √(z.im * w.im)))⟩


theorem dist_eq (z w : ℍ) : dist z w = 2 * arsinh (dist (z : ℂ) w / (2 * √(z.im * w.im))) :=
  rfl


theorem sinh_half_dist (z w : ℍ) :
    sinh (dist z w / 2) = dist (z : ℂ) w / (2 * √(z.im * w.im)) := by
  /-
    z w : UpperHalfPlane
    ⊢ Eq (Real.sinh (HDiv.hDiv (Dist.dist z w) 2)) (HDiv.hDiv (Dist.dist ↑z ↑w) (H …
  -/
  rw [dist_eq, mul_div_cancel_left₀ (arsinh _) two_ne_zero, sinh_arsinh]
  /-
    🎉 no goals
  -/


theorem cosh_half_dist (z w : ℍ) :
    cosh (dist z w / 2) = dist (z : ℂ) (conj (w : ℂ)) / (2 * √(z.im * w.im)) := by
  /-
    z w : UpperHalfPlane
    ⊢ Eq (Real.cosh (HDiv.hDiv (Dist.dist z w) 2)) (HDiv.hDiv (Dist.dist (↑z) ((st …
  -/
  rw [← sq_eq_sq₀, cosh_sq', sinh_half_dist, div_pow, div_pow, one_add_div, mul_pow, sq_sqrt]
    /-
      z w : UpperHalfPlane
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul (HPow.hPow 2 2) (HMul.hMul z.im w.im)) ( …
    -/
  · congr 1
    simp only [Complex.dist_eq, Complex.sq_abs, Complex.normSq_sub, Complex.normSq_conj,
      Complex.conj_conj, Complex.mul_re, Complex.conj_re, Complex.conj_im, coe_im]
    /-
      case e_a
      z w : UpperHalfPlane
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow 2 2) (HMul.hMul z.im w.im)) (HSub.hSub ( …
    -/
    ring
    /-
      🎉 no goals
    -/
  /-
    z w : UpperHalfPlane
    ⊢ LE.le 0 (HMul.hMul z.im w.im)
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


theorem tanh_half_dist (z w : ℍ) :
    tanh (dist z w / 2) = dist (z : ℂ) w / dist (z : ℂ) (conj ↑w) := by
  /-
    z w : UpperHalfPlane
    ⊢ Eq (Real.tanh (HDiv.hDiv (Dist.dist z w) 2)) (HDiv.hDiv (Dist.dist ↑z ↑w) (D …
  -/
  rw [tanh_eq_sinh_div_cosh, sinh_half_dist, cosh_half_dist, div_div_div_comm, div_self, div_one]
  /-
    z w : UpperHalfPlane
    ⊢ Ne (HMul.hMul 2 (HMul.hMul z.im w.im).sqrt) 0
  -/
  positivity
  /-
    🎉 no goals
  -/


theorem exp_half_dist (z w : ℍ) :
    exp (dist z w / 2) = (dist (z : ℂ) w + dist (z : ℂ) (conj ↑w)) / (2 * √(z.im * w.im)) := by
  /-
    z w : UpperHalfPlane
    ⊢ Eq (Real.exp (HDiv.hDiv (Dist.dist z w) 2)) (HDiv.hDiv (HAdd.hAdd (Dist.dist …
  -/
  rw [← sinh_add_cosh, sinh_half_dist, cosh_half_dist, add_div]
  /-
    🎉 no goals
  -/


theorem cosh_dist (z w : ℍ) : cosh (dist z w) = 1 + dist (z : ℂ) w ^ 2 / (2 * z.im * w.im) := by
  rw [dist_eq, cosh_two_mul, cosh_sq', add_assoc, ← two_mul, sinh_arsinh, div_pow, mul_pow,
                                                                                      /-
                                                                                        case hc
                                                                                        z w : UpperHalfPlane
                                                                                        ⊢ Ne 2 0
                                                                                      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    sq_sqrt, sq (2 : ℝ), mul_assoc, ← mul_div_assoc, mul_assoc, mul_div_mul_left] <;> positivity
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem sinh_half_dist_add_dist (a b c : ℍ) : sinh ((dist a b + dist b c) / 2) =
    (dist (a : ℂ) b * dist (c : ℂ) (conj ↑b) + dist (b : ℂ) c * dist (a : ℂ) (conj ↑b)) /
      (2 * √(a.im * c.im) * dist (b : ℂ) (conj ↑b)) := by
  /-
    a b c : UpperHalfPlane
    ⊢ Eq (Real.sinh (HDiv.hDiv (HAdd.hAdd (Dist.dist a b) (Dist.dist b c)) 2)) (HD …
  -/
  simp only [add_div _ _ (2 : ℝ), sinh_add, sinh_half_dist, cosh_half_dist, div_mul_div_comm]
  rw [← add_div, Complex.dist_self_conj, coe_im, abs_of_pos b.im_pos, mul_comm (dist (b : ℂ) _),
    dist_comm (b : ℂ), Complex.dist_conj_comm, mul_mul_mul_comm, mul_mul_mul_comm _ _ _ b.im]
  /-
    a b c : UpperHalfPlane
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul (Dist.dist ↑a ↑b) (Dist.dist (↑c) ((star …
  -/
  congr 2
  rw [sqrt_mul, sqrt_mul, sqrt_mul, mul_comm (√a.im), mul_mul_mul_comm, mul_self_sqrt,
                    /-
                      case e_a.e_a
                      a b c : UpperHalfPlane
                      ⊢ LE.le 0 b.im
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
      mul_comm] <;> exact (im_pos _).le
                    /-
                      🎉 no goals
                    -/


protected theorem dist_comm (z w : ℍ) : dist z w = dist w z := by
  /-
    z w : UpperHalfPlane
    ⊢ Eq (Dist.dist z w) (Dist.dist w z)
  -/
  simp only [dist_eq, dist_comm (z : ℂ), mul_comm]
  /-
    🎉 no goals
  -/


theorem dist_le_iff_le_sinh :
    dist z w ≤ r ↔ dist (z : ℂ) w / (2 * √(z.im * w.im)) ≤ sinh (r / 2) := by
  /-
    z w : UpperHalfPlane
    r : Real
    ⊢ Iff (LE.le (Dist.dist z w) r) (LE.le (HDiv.hDiv (Dist.dist ↑z ↑w) (HMul.hMul …
  -/
  rw [← div_le_div_iff_of_pos_right (zero_lt_two' ℝ), ← sinh_le_sinh, sinh_half_dist]
  /-
    🎉 no goals
  -/


theorem dist_eq_iff_eq_sinh :
    dist z w = r ↔ dist (z : ℂ) w / (2 * √(z.im * w.im)) = sinh (r / 2) := by
  /-
    z w : UpperHalfPlane
    r : Real
    ⊢ Iff (Eq (Dist.dist z w) r) (Eq (HDiv.hDiv (Dist.dist ↑z ↑w) (HMul.hMul 2 (HM …
  -/
  rw [← div_left_inj' (two_ne_zero' ℝ), ← sinh_inj, sinh_half_dist]
  /-
    🎉 no goals
  -/


theorem dist_eq_iff_eq_sq_sinh (hr : 0 ≤ r) :
    dist z w = r ↔ dist (z : ℂ) w ^ 2 / (4 * z.im * w.im) = sinh (r / 2) ^ 2 := by
  /-
    z w : UpperHalfPlane
    r : Real
    hr : LE.le 0 r
    ⊢ Iff (Eq (Dist.dist z w) r) (Eq (HDiv.hDiv (HPow.hPow (Dist.dist ↑z ↑w) 2) (H …
  -/
  rw [dist_eq_iff_eq_sinh, ← sq_eq_sq₀, div_pow, mul_pow, sq_sqrt, mul_assoc]
    /-
      z w : UpperHalfPlane
      r : Real
      hr : LE.le 0 r
      ⊢ Iff (Eq (HDiv.hDiv (HPow.hPow (Dist.dist ↑z ↑w) 2) (HMul.hMul (HPow.hPow 2 2 …
    -/
  · norm_num
    /-
      🎉 no goals
    -/
  /-
    z w : UpperHalfPlane
    r : Real
    hr : LE.le 0 r
    ⊢ LE.le 0 (HMul.hMul z.im w.im)
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


protected theorem dist_triangle (a b c : ℍ) : dist a c ≤ dist a b + dist b c := by
  rw [dist_le_iff_le_sinh, sinh_half_dist_add_dist, div_mul_eq_div_div _ _ (dist _ _), le_div_iff₀,
    div_mul_eq_mul_div]
    /-
      a b c : UpperHalfPlane
      ⊢ LE.le (HDiv.hDiv (HMul.hMul (Dist.dist ↑a ↑c) (Dist.dist (↑b) ((starRingEnd  …
    -/
  · gcongr
    /-
      case hab
      a b c : UpperHalfPlane
      ⊢ LE.le (HMul.hMul (Dist.dist ↑a ↑c) (Dist.dist (↑b) ((starRingEnd Complex) ↑b …
    -/
    exact EuclideanGeometry.mul_dist_le_mul_dist_add_mul_dist (a : ℂ) b c (conj (b : ℂ))
    /-
      🎉 no goals
    -/
    /-
      a b c : UpperHalfPlane
      ⊢ LT.lt 0 (Dist.dist (↑b) ((starRingEnd Complex) ↑b))
    -/
  · rw [dist_comm, dist_pos, Ne, Complex.conj_eq_iff_im]
    /-
      a b c : UpperHalfPlane
      ⊢ Not (Eq (↑b).im 0)
    -/
    exact b.im_ne_zero
    /-
      🎉 no goals
    -/


theorem dist_le_dist_coe_div_sqrt (z w : ℍ) : dist z w ≤ dist (z : ℂ) w / √(z.im * w.im) := by
  /-
    z w : UpperHalfPlane
    ⊢ LE.le (Dist.dist z w) (HDiv.hDiv (Dist.dist ↑z ↑w) (HMul.hMul z.im w.im).sqrt)
  -/
  rw [dist_le_iff_le_sinh, ← div_mul_eq_div_div_swap, self_le_sinh_iff]
  /-
    z w : UpperHalfPlane
    ⊢ LE.le 0 (HDiv.hDiv (Dist.dist ↑z ↑w) (HMul.hMul 2 (HMul.hMul z.im w.im).sqrt))
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- An auxiliary `MetricSpace` instance on the upper half-plane. This instance has bad projection
to `TopologicalSpace`. We replace it later. -/
def metricSpaceAux : MetricSpace ℍ where
  dist := dist
                    /-
                      z✝ w : UpperHalfPlane
                      r : Real
                      z : UpperHalfPlane
                      ⊢ Eq (Dist.dist z z) 0
                    -/
  dist_self z := by rw [dist_eq, dist_self, zero_div, arsinh_zero, mul_zero]
                    /-
                      🎉 no goals
                    -/
  dist_comm := UpperHalfPlane.dist_comm
  dist_triangle := UpperHalfPlane.dist_triangle
  eq_of_dist_eq_zero {z w} h := by
    /-
      z✝ w✝ : UpperHalfPlane
      r : Real
      z w : UpperHalfPlane
      h : Eq (Dist.dist z w) 0
      ⊢ Eq z w
    -/
    simpa [dist_eq, Real.sqrt_eq_zero', (mul_pos z.im_pos w.im_pos).not_le, Set.ext_iff] using h
    /-
      🎉 no goals
    -/


theorem cosh_dist' (z w : ℍ) :
    Real.cosh (dist z w) = ((z.re - w.re) ^ 2 + z.im ^ 2 + w.im ^ 2) / (2 * z.im * w.im) := by
  /-
    z w : UpperHalfPlane
    ⊢ Eq (Real.cosh (Dist.dist z w)) (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd (HPow.hPow ( …
  -/
  field_simp [cosh_dist, Complex.dist_eq, Complex.sq_abs, normSq_apply]
  /-
    z w : UpperHalfPlane
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul 2 z.im) w.im) (HAdd.hAdd (HMul.hMul (HSu …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Euclidean center of the circle with center `z` and radius `r` in the hyperbolic metric. -/
def center (z : ℍ) (r : ℝ) : ℍ :=
                                  /-
                                    z✝ w : UpperHalfPlane
                                    r✝ : Real
                                    z : UpperHalfPlane
                                    r : Real
                                    ⊢ LT.lt 0 { re := z.re, im := HMul.hMul z.im (Real.cosh r) }.im
                                  -/
  ⟨⟨z.re, z.im * Real.cosh r⟩, by positivity⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem center_re (z r) : (center z r).re = z.re :=
  rfl


@[simp]
theorem center_im (z r) : (center z r).im = z.im * Real.cosh r :=
  rfl


@[simp]
theorem center_zero (z : ℍ) : center z 0 = z :=
                 /-
                   z : UpperHalfPlane
                   ⊢ Eq (z.center 0).im z.im
                 -/
  ext' rfl <| by rw [center_im, Real.cosh_zero, mul_one]
                 /-
                   🎉 no goals
                 -/


theorem dist_coe_center_sq (z w : ℍ) (r : ℝ) : dist (z : ℂ) (w.center r) ^ 2 =
    2 * z.im * w.im * (Real.cosh (dist z w) - Real.cosh r) + (w.im * Real.sinh r) ^ 2 := by
  /-
    z w : UpperHalfPlane
    r : Real
    ⊢ Eq (HPow.hPow (Dist.dist ↑z ↑(w.center r)) 2) (HAdd.hAdd (HMul.hMul (HMul.hM …
  -/
  have H : 2 * z.im * w.im ≠ 0 := by positivity
  simp only [Complex.dist_eq, Complex.sq_abs, normSq_apply, coe_re, coe_im, center_re, center_im,
    cosh_dist', mul_div_cancel₀ _ H, sub_sq z.im, mul_pow, Real.cosh_sq, sub_re, sub_im, mul_sub, ←
    sq]
  /-
    z w : UpperHalfPlane
    r : Real
    H : Ne (HMul.hMul (HMul.hMul 2 z.im) w.im) 0
    ⊢ Eq (HAdd.hAdd (HPow.hPow (HSub.hSub z.re w.re) 2) (HAdd.hAdd (HSub.hSub (HPo …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem dist_coe_center (z w : ℍ) (r : ℝ) : dist (z : ℂ) (w.center r) =
    √(2 * z.im * w.im * (Real.cosh (dist z w) - Real.cosh r) + (w.im * Real.sinh r) ^ 2) := by
  /-
    z w : UpperHalfPlane
    r : Real
    ⊢ Eq (Dist.dist ↑z ↑(w.center r)) (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul  …
  -/
  rw [← sqrt_sq dist_nonneg, dist_coe_center_sq]
  /-
    🎉 no goals
  -/


theorem cmp_dist_eq_cmp_dist_coe_center (z w : ℍ) (r : ℝ) :
    cmp (dist z w) r = cmp (dist (z : ℂ) (w.center r)) (w.im * Real.sinh r) := by
  /-
    z w : UpperHalfPlane
    r : Real
    ⊢ Eq (cmp (Dist.dist z w) r) (cmp (Dist.dist ↑z ↑(w.center r)) (HMul.hMul w.im …
  -/
  letI := metricSpaceAux
  /-
    z w : UpperHalfPlane
    r : Real
    this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
    ⊢ Eq (cmp (Dist.dist z w) r) (cmp (Dist.dist ↑z ↑(w.center r)) (HMul.hMul w.im …
  -/
  cases' lt_or_le r 0 with hr₀ hr₀
    /-
      case inl
      z w : UpperHalfPlane
      r : Real
      this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
      hr₀ : LT.lt r 0
      ⊢ Eq (cmp (Dist.dist z w) r) (cmp (Dist.dist ↑z ↑(w.center r)) (HMul.hMul w.im …
    -/
  · trans Ordering.gt
    exacts [(hr₀.trans_le dist_nonneg).cmp_eq_gt,
      ((mul_neg_of_pos_of_neg w.im_pos (sinh_neg_iff.2 hr₀)).trans_le dist_nonneg).cmp_eq_gt.symm]
  /-
    case inr
    z w : UpperHalfPlane
    r : Real
    this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
    hr₀ : LE.le 0 r
    ⊢ Eq (cmp (Dist.dist z w) r) (cmp (Dist.dist ↑z ↑(w.center r)) (HMul.hMul w.im …
  -/
  have hr₀' : 0 ≤ w.im * Real.sinh r := by positivity
  /-
    case inr
    z w : UpperHalfPlane
    r : Real
    this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
    hr₀ : LE.le 0 r
    hr₀' : LE.le 0 (HMul.hMul w.im (Real.sinh r))
    ⊢ Eq (cmp (Dist.dist z w) r) (cmp (Dist.dist ↑z ↑(w.center r)) (HMul.hMul w.im …
  -/
  have hzw₀ : 0 < 2 * z.im * w.im := by positivity
  #adaptation_note
  /--
  After the bug fix in https://github.com/leanprover/lean4/pull/6024,
  we need to give Lean the hint `(y := w.im * Real.sinh r)`.
  -/
  simp only [← cosh_strictMonoOn.cmp_map_eq dist_nonneg hr₀,
    ← (pow_left_strictMonoOn₀ two_ne_zero).cmp_map_eq dist_nonneg (y := w.im * Real.sinh r) hr₀',
    dist_coe_center_sq]
  /-
    case inr
    z w : UpperHalfPlane
    r : Real
    this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
    hr₀ : LE.le 0 r
    hr₀' : LE.le 0 (HMul.hMul w.im (Real.sinh r))
    hzw₀ : LT.lt 0 (HMul.hMul (HMul.hMul 2 z.im) w.im)
    ⊢ Eq (cmp (Real.cosh (Dist.dist z w)) (Real.cosh r)) (cmp (HAdd.hAdd (HMul.hMu …
  -/
  rw [← cmp_mul_pos_left hzw₀, ← cmp_sub_zero, ← mul_sub, ← cmp_add_right, zero_add]
  /-
    🎉 no goals
  -/


theorem dist_eq_iff_dist_coe_center_eq :
    dist z w = r ↔ dist (z : ℂ) (w.center r) = w.im * Real.sinh r :=
  eq_iff_eq_of_cmp_eq_cmp (cmp_dist_eq_cmp_dist_coe_center z w r)


@[simp]
theorem dist_self_center (z : ℍ) (r : ℝ) :
    dist (z : ℂ) (z.center r) = z.im * (Real.cosh r - 1) := by
  /-
    z : UpperHalfPlane
    r : Real
    ⊢ Eq (Dist.dist ↑z ↑(z.center r)) (HMul.hMul z.im (HSub.hSub (Real.cosh r) 1))
  -/
  rw [dist_of_re_eq (z.center_re r).symm, dist_comm, Real.dist_eq, mul_sub, mul_one]
  /-
    z : UpperHalfPlane
    r : Real
    ⊢ Eq (abs (HSub.hSub (↑(z.center r)).im (↑z).im)) (HSub.hSub (HMul.hMul z.im ( …
  -/
  exact abs_of_nonneg (sub_nonneg.2 <| le_mul_of_one_le_right z.im_pos.le (one_le_cosh _))
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_center_dist (z w : ℍ) :
    dist (z : ℂ) (w.center (dist z w)) = w.im * Real.sinh (dist z w) :=
  dist_eq_iff_dist_coe_center_eq.1 rfl


theorem dist_lt_iff_dist_coe_center_lt :
    dist z w < r ↔ dist (z : ℂ) (w.center r) < w.im * Real.sinh r :=
  lt_iff_lt_of_cmp_eq_cmp (cmp_dist_eq_cmp_dist_coe_center z w r)


theorem lt_dist_iff_lt_dist_coe_center :
    r < dist z w ↔ w.im * Real.sinh r < dist (z : ℂ) (w.center r) :=
  lt_iff_lt_of_cmp_eq_cmp (cmp_eq_cmp_symm.1 <| cmp_dist_eq_cmp_dist_coe_center z w r)


theorem dist_le_iff_dist_coe_center_le :
    dist z w ≤ r ↔ dist (z : ℂ) (w.center r) ≤ w.im * Real.sinh r :=
  le_iff_le_of_cmp_eq_cmp (cmp_dist_eq_cmp_dist_coe_center z w r)


theorem le_dist_iff_le_dist_coe_center :
    r < dist z w ↔ w.im * Real.sinh r < dist (z : ℂ) (w.center r) :=
  lt_iff_lt_of_cmp_eq_cmp (cmp_eq_cmp_symm.1 <| cmp_dist_eq_cmp_dist_coe_center z w r)


/-- For two points on the same vertical line, the distance is equal to the distance between the
logarithms of their imaginary parts. -/
nonrec theorem dist_of_re_eq (h : z.re = w.re) : dist z w = dist (log z.im) (log w.im) := by
  /-
    z w : UpperHalfPlane
    h : Eq z.re w.re
    ⊢ Eq (Dist.dist z w) (Dist.dist (Real.log z.im) (Real.log w.im))
  -/
  have h₀ : 0 < z.im / w.im := by positivity
  rw [dist_eq_iff_dist_coe_center_eq, Real.dist_eq, ← abs_sinh, ← log_div z.im_ne_zero w.im_ne_zero,
    sinh_log h₀, dist_of_re_eq, coe_im, coe_im, center_im, cosh_abs, cosh_log h₀, inv_div] <;>
  [skip; exact h]
  /-
    z w : UpperHalfPlane
    h : Eq z.re w.re
    h₀ : LT.lt 0 (HDiv.hDiv z.im w.im)
    ⊢ Eq (Dist.dist z.im (HMul.hMul w.im (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv z.im w.i …
  -/
  nth_rw 4 [← abs_of_pos w.im_pos]
  /-
    z w : UpperHalfPlane
    h : Eq z.re w.re
    h₀ : LT.lt 0 (HDiv.hDiv z.im w.im)
    ⊢ Eq (Dist.dist z.im (HMul.hMul w.im (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv z.im w.i …
  -/
  simp only [← _root_.abs_mul, coe_im, Real.dist_eq]
  /-
    z w : UpperHalfPlane
    h : Eq z.re w.re
    h₀ : LT.lt 0 (HDiv.hDiv z.im w.im)
    ⊢ Eq (abs (HSub.hSub z.im (HMul.hMul w.im (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv z.i …
  -/
  congr 1
  /-
    case e_a
    z w : UpperHalfPlane
    h : Eq z.re w.re
    h₀ : LT.lt 0 (HDiv.hDiv z.im w.im)
    ⊢ Eq (HSub.hSub z.im (HMul.hMul w.im (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv z.im w.i …
  -/
  field_simp
  /-
    case e_a
    z w : UpperHalfPlane
    h : Eq z.re w.re
    h₀ : LT.lt 0 (HDiv.hDiv z.im w.im)
    ⊢ Eq (HSub.hSub (HMul.hMul z.im (HMul.hMul (HMul.hMul w.im z.im) 2)) (HMul.hMu …
  -/
  ring
  /-
    🎉 no goals
  -/

/-- Hyperbolic distance between two points is greater than or equal to the distance between the
logarithms of their imaginary parts. -/
theorem dist_log_im_le (z w : ℍ) : dist (log z.im) (log w.im) ≤ dist z w :=
  calc
    dist (log z.im) (log w.im) = dist (mk ⟨0, z.im⟩ z.im_pos) (mk ⟨0, w.im⟩ w.im_pos) :=
      Eq.symm <| dist_of_re_eq rfl
    _ ≤ dist z w := by
      /-
        z w : UpperHalfPlane
        ⊢ LE.le (Dist.dist (UpperHalfPlane.mk { re := 0, im := z.im } ⋯) (UpperHalfPla …
      -/
      simp_rw [dist_eq]
      /-
        z w : UpperHalfPlane
        ⊢ LE.le (HMul.hMul 2 (Real.arsinh (HDiv.hDiv (Dist.dist ↑(UpperHalfPlane.mk {  …
      -/
      dsimp only [coe_mk, mk_im]
      /-
        z w : UpperHalfPlane
        ⊢ LE.le (HMul.hMul 2 (Real.arsinh (HDiv.hDiv (Dist.dist { re := 0, im := z.im  …
      -/
      gcongr
      /-
        case h.a.hab
        z w : UpperHalfPlane
        ⊢ LE.le (Dist.dist { re := 0, im := z.im } { re := 0, im := w.im }) (Dist.dist …
      -/
      simpa [sqrt_sq_eq_abs] using Complex.abs_im_le_abs (z - w)
      /-
        🎉 no goals
      -/


theorem im_le_im_mul_exp_dist (z w : ℍ) : z.im ≤ w.im * Real.exp (dist z w) := by
  /-
    z w : UpperHalfPlane
    ⊢ LE.le z.im (HMul.hMul w.im (Real.exp (Dist.dist z w)))
  -/
  rw [← div_le_iff₀' w.im_pos, ← exp_log z.im_pos, ← exp_log w.im_pos, ← Real.exp_sub, exp_le_exp]
  /-
    z w : UpperHalfPlane
    ⊢ LE.le (HSub.hSub (Real.log z.im) (Real.log w.im)) (Dist.dist z w)
  -/
  exact (le_abs_self _).trans (dist_log_im_le z w)
  /-
    🎉 no goals
  -/


theorem im_div_exp_dist_le (z w : ℍ) : z.im / Real.exp (dist z w) ≤ w.im :=
  (div_le_iff₀ (exp_pos _)).2 (im_le_im_mul_exp_dist z w)


/-- An upper estimate on the complex distance between two points in terms of the hyperbolic distance
and the imaginary part of one of the points. -/
theorem dist_coe_le (z w : ℍ) : dist (z : ℂ) w ≤ w.im * (Real.exp (dist z w) - 1) :=
  calc
    dist (z : ℂ) w ≤ dist (z : ℂ) (w.center (dist z w)) + dist (w : ℂ) (w.center (dist z w)) :=
      dist_triangle_right _ _ _
    _ = w.im * (Real.exp (dist z w) - 1) := by
      /-
        z w : UpperHalfPlane
        ⊢ Eq (HAdd.hAdd (Dist.dist ↑z ↑(w.center (Dist.dist z w))) (Dist.dist ↑w ↑(w.c …
      -/
      rw [dist_center_dist, dist_self_center, ← mul_add, ← add_sub_assoc, Real.sinh_add_cosh]
      /-
        🎉 no goals
      -/


/-- An upper estimate on the complex distance between two points in terms of the hyperbolic distance
and the imaginary part of one of the points. -/
theorem le_dist_coe (z w : ℍ) : w.im * (1 - Real.exp (-dist z w)) ≤ dist (z : ℂ) w :=
  calc
    w.im * (1 - Real.exp (-dist z w)) =
        dist (z : ℂ) (w.center (dist z w)) - dist (w : ℂ) (w.center (dist z w)) := by
      /-
        z w : UpperHalfPlane
        ⊢ Eq (HMul.hMul w.im (HSub.hSub 1 (Real.exp (Neg.neg (Dist.dist z w))))) (HSub …
      -/
      rw [dist_center_dist, dist_self_center, ← Real.cosh_sub_sinh]; ring
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    _ ≤ dist (z : ℂ) w := sub_le_iff_le_add.2 <| dist_triangle _ _ _


/-- The hyperbolic metric on the upper half plane. We ensure that the projection to
`TopologicalSpace` is definitionally equal to the subtype topology. -/
instance : MetricSpace ℍ :=
  metricSpaceAux.replaceTopology <| by
    /-
      z w : UpperHalfPlane
      r : Real
      ⊢ Eq UpperHalfPlane.instTopologicalSpace UniformSpace.toTopologicalSpace
    -/
    refine le_antisymm (continuous_id_iff_le.1 ?_) ?_
      /-
        case refine_1
        z w : UpperHalfPlane
        r : Real
        ⊢ Continuous id
      -/
    · refine (@continuous_iff_continuous_dist ℍ ℍ metricSpaceAux.toPseudoMetricSpace _ _).2 ?_
      /-
        case refine_1
        z w : UpperHalfPlane
        r : Real
        ⊢ Continuous fun x => Dist.dist (id x.1) (id x.2)
      -/
      have : ∀ x : ℍ × ℍ, 2 * √(x.1.im * x.2.im) ≠ 0 := fun x => by positivity
      -- `continuity` fails to apply `Continuous.div`
      apply_rules [Continuous.div, Continuous.mul, continuous_const, Continuous.arsinh,
        Continuous.dist, continuous_coe.comp, continuous_fst, continuous_snd,
        Real.continuous_sqrt.comp, continuous_im.comp]
      /-
        case refine_2
        z w : UpperHalfPlane
        r : Real
        ⊢ LE.le UniformSpace.toTopologicalSpace UpperHalfPlane.instTopologicalSpace
      -/
    · letI : MetricSpace ℍ := metricSpaceAux
      /-
        case refine_2
        z w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        ⊢ LE.le UniformSpace.toTopologicalSpace UpperHalfPlane.instTopologicalSpace
      -/
      refine le_of_nhds_le_nhds fun z => ?_
      /-
        case refine_2
        z✝ w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        ⊢ LE.le (nhds z) (nhds z)
      -/
      rw [nhds_induced]
      /-
        case refine_2
        z✝ w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        ⊢ LE.le (nhds z) (Filter.comap Subtype.val (nhds ↑z))
      -/
      refine (nhds_basis_ball.le_basis_iff (nhds_basis_ball.comap _)).2 fun R hR => ?_
      /-
        case refine_2
        z✝ w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        R : Real
        hR : LT.lt 0 R
        ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball z i) (Set.pre …
      -/
      have h₁ : 1 < R / im z + 1 := lt_add_of_pos_left _ (div_pos hR z.im_pos)
      /-
        case refine_2
        z✝ w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        R : Real
        hR : LT.lt 0 R
        h₁ : LT.lt 1 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball z i) (Set.pre …
      -/
      have h₀ : 0 < R / im z + 1 := one_pos.trans h₁
      /-
        case refine_2
        z✝ w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        R : Real
        hR : LT.lt 0 R
        h₁ : LT.lt 1 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        h₀ : LT.lt 0 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball z i) (Set.pre …
      -/
      refine ⟨log (R / im z + 1), Real.log_pos h₁, ?_⟩
      /-
        case refine_2
        z✝ w : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        R : Real
        hR : LT.lt 0 R
        h₁ : LT.lt 1 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        h₀ : LT.lt 0 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        ⊢ HasSubset.Subset (Metric.ball z (Real.log (HAdd.hAdd (HDiv.hDiv R z.im) 1))) …
      -/
      refine fun w hw => (dist_coe_le w z).trans_lt ?_
      /-
        case refine_2
        z✝ w✝ : UpperHalfPlane
        r : Real
        this : MetricSpace UpperHalfPlane := UpperHalfPlane.metricSpaceAux
        z : UpperHalfPlane
        R : Real
        hR : LT.lt 0 R
        h₁ : LT.lt 1 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        h₀ : LT.lt 0 (HAdd.hAdd (HDiv.hDiv R z.im) 1)
        w : UpperHalfPlane
        hw : Membership.mem (Metric.ball z (Real.log (HAdd.hAdd (HDiv.hDiv R z.im) 1)) …
        ⊢ LT.lt (HMul.hMul z.im (HSub.hSub (Real.exp (Dist.dist w z)) 1)) R
      -/
      rwa [← lt_div_iff₀' z.im_pos, sub_lt_iff_lt_add, ← Real.lt_log_iff_exp_lt h₀]
      /-
        🎉 no goals
      -/


theorem im_pos_of_dist_center_le {z : ℍ} {r : ℝ} {w : ℂ}
    (h : dist w (center z r) ≤ z.im * Real.sinh r) : 0 < w.im :=
  calc
    0 < z.im * (Real.cosh r - Real.sinh r) := mul_pos z.im_pos (sub_pos.2 <| sinh_lt_cosh _)
    _ = (z.center r).im - z.im * Real.sinh r := mul_sub _ _ _
                                                                         /-
                                                                           z : UpperHalfPlane
                                                                           r : Real
                                                                           w : Complex
                                                                           h : LE.le (Dist.dist w ↑(z.center r)) (HMul.hMul z.im (Real.sinh r))
                                                                           ⊢ LE.le (Dist.dist (↑(z.center r)) w) (HMul.hMul z.im (Real.sinh r))
                                                                         -/
    _ ≤ (z.center r).im - dist (z.center r : ℂ) w := sub_le_sub_left (by rwa [dist_comm]) _
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    _ ≤ w.im := sub_le_comm.1 <| (le_abs_self _).trans (abs_im_le_abs <| z.center r - w)


theorem image_coe_closedBall (z : ℍ) (r : ℝ) :
    ((↑) : ℍ → ℂ) '' closedBall (α := ℍ) z r = closedBall ↑(z.center r) (z.im * Real.sinh r) := by
  /-
    z : UpperHalfPlane
    r : Real
    ⊢ Eq (Set.image UpperHalfPlane.coe (Metric.closedBall z r)) (Metric.closedBall …
  -/
  ext w; constructor
    /-
      case h.mp
      z : UpperHalfPlane
      r : Real
      w : Complex
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.closedBall z r)) w → Me …
    -/
  · rintro ⟨w, hw, rfl⟩
    /-
      case h.mp.intro.intro
      z : UpperHalfPlane
      r : Real
      w : UpperHalfPlane
      hw : Membership.mem (Metric.closedBall z r) w
      ⊢ Membership.mem (Metric.closedBall (↑(z.center r)) (HMul.hMul z.im (Real.sinh …
    -/
    exact dist_le_iff_dist_coe_center_le.1 hw
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      z : UpperHalfPlane
      r : Real
      w : Complex
      ⊢ Membership.mem (Metric.closedBall (↑(z.center r)) (HMul.hMul z.im (Real.sinh …
    -/
  · intro hw
    /-
      case h.mpr
      z : UpperHalfPlane
      r : Real
      w : Complex
      hw : Membership.mem (Metric.closedBall (↑(z.center r)) (HMul.hMul z.im (Real.s …
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.closedBall z r)) w
    -/
    lift w to ℍ using im_pos_of_dist_center_le hw
    /-
      case h.mpr.intro
      z : UpperHalfPlane
      r : Real
      w : UpperHalfPlane
      hw : Membership.mem (Metric.closedBall (↑(z.center r)) (HMul.hMul z.im (Real.s …
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.closedBall z r)) ↑w
    -/
    exact mem_image_of_mem _ (dist_le_iff_dist_coe_center_le.2 hw)
    /-
      🎉 no goals
    -/


theorem image_coe_ball (z : ℍ) (r : ℝ) :
    ((↑) : ℍ → ℂ) '' ball (α := ℍ) z r = ball ↑(z.center r) (z.im * Real.sinh r) := by
  /-
    z : UpperHalfPlane
    r : Real
    ⊢ Eq (Set.image UpperHalfPlane.coe (Metric.ball z r)) (Metric.ball (↑(z.center …
  -/
  ext w; constructor
    /-
      case h.mp
      z : UpperHalfPlane
      r : Real
      w : Complex
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.ball z r)) w → Membersh …
    -/
  · rintro ⟨w, hw, rfl⟩
    /-
      case h.mp.intro.intro
      z : UpperHalfPlane
      r : Real
      w : UpperHalfPlane
      hw : Membership.mem (Metric.ball z r) w
      ⊢ Membership.mem (Metric.ball (↑(z.center r)) (HMul.hMul z.im (Real.sinh r))) ↑w
    -/
    exact dist_lt_iff_dist_coe_center_lt.1 hw
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      z : UpperHalfPlane
      r : Real
      w : Complex
      ⊢ Membership.mem (Metric.ball (↑(z.center r)) (HMul.hMul z.im (Real.sinh r)))  …
    -/
  · intro hw
    /-
      case h.mpr
      z : UpperHalfPlane
      r : Real
      w : Complex
      hw : Membership.mem (Metric.ball (↑(z.center r)) (HMul.hMul z.im (Real.sinh r) …
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.ball z r)) w
    -/
    lift w to ℍ using im_pos_of_dist_center_le (ball_subset_closedBall hw)
    /-
      case h.mpr.intro
      z : UpperHalfPlane
      r : Real
      w : UpperHalfPlane
      hw : Membership.mem (Metric.ball (↑(z.center r)) (HMul.hMul z.im (Real.sinh r) …
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.ball z r)) ↑w
    -/
    exact mem_image_of_mem _ (dist_lt_iff_dist_coe_center_lt.2 hw)
    /-
      🎉 no goals
    -/


theorem image_coe_sphere (z : ℍ) (r : ℝ) :
    ((↑) : ℍ → ℂ) '' sphere (α := ℍ) z r = sphere ↑(z.center r) (z.im * Real.sinh r) := by
  /-
    z : UpperHalfPlane
    r : Real
    ⊢ Eq (Set.image UpperHalfPlane.coe (Metric.sphere z r)) (Metric.sphere (↑(z.ce …
  -/
  ext w; constructor
    /-
      case h.mp
      z : UpperHalfPlane
      r : Real
      w : Complex
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.sphere z r)) w → Member …
    -/
  · rintro ⟨w, hw, rfl⟩
    /-
      case h.mp.intro.intro
      z : UpperHalfPlane
      r : Real
      w : UpperHalfPlane
      hw : Membership.mem (Metric.sphere z r) w
      ⊢ Membership.mem (Metric.sphere (↑(z.center r)) (HMul.hMul z.im (Real.sinh r)) …
    -/
    exact dist_eq_iff_dist_coe_center_eq.1 hw
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      z : UpperHalfPlane
      r : Real
      w : Complex
      ⊢ Membership.mem (Metric.sphere (↑(z.center r)) (HMul.hMul z.im (Real.sinh r)) …
    -/
  · intro hw
    /-
      case h.mpr
      z : UpperHalfPlane
      r : Real
      w : Complex
      hw : Membership.mem (Metric.sphere (↑(z.center r)) (HMul.hMul z.im (Real.sinh  …
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.sphere z r)) w
    -/
    lift w to ℍ using im_pos_of_dist_center_le (sphere_subset_closedBall hw)
    /-
      case h.mpr.intro
      z : UpperHalfPlane
      r : Real
      w : UpperHalfPlane
      hw : Membership.mem (Metric.sphere (↑(z.center r)) (HMul.hMul z.im (Real.sinh  …
      ⊢ Membership.mem (Set.image UpperHalfPlane.coe (Metric.sphere z r)) ↑w
    -/
    exact mem_image_of_mem _ (dist_eq_iff_dist_coe_center_eq.2 hw)
    /-
      🎉 no goals
    -/


instance : ProperSpace ℍ := by
  /-
    z w : UpperHalfPlane
    r : Real
    ⊢ ProperSpace UpperHalfPlane
  -/
  refine ⟨fun z r => ?_⟩
  /-
    z✝ w : UpperHalfPlane
    r✝ : Real
    z : UpperHalfPlane
    r : Real
    ⊢ IsCompact (Metric.closedBall z r)
  -/
  rw [IsInducing.subtypeVal.isCompact_iff (f := ((↑) : ℍ → ℂ)), image_coe_closedBall]
  /-
    z✝ w : UpperHalfPlane
    r✝ : Real
    z : UpperHalfPlane
    r : Real
    ⊢ IsCompact (Metric.closedBall (↑(z.center r)) (HMul.hMul z.im (Real.sinh r)))
  -/
  apply isCompact_closedBall
  /-
    🎉 no goals
  -/


theorem isometry_vertical_line (a : ℝ) : Isometry fun y => mk ⟨a, exp y⟩ (exp_pos y) := by
  /-
    a : Real
    ⊢ Isometry fun y => UpperHalfPlane.mk { re := a, im := Real.exp y } ⋯
  -/
  refine Isometry.of_dist_eq fun y₁ y₂ => ?_
  /-
    a y₁ y₂ : Real
    ⊢ Eq (Dist.dist (UpperHalfPlane.mk { re := a, im := Real.exp y₁ } ⋯) (UpperHal …
  -/
  rw [dist_of_re_eq]
  /-
    a y₁ y₂ : Real
    ⊢ Eq (Dist.dist (Real.log (UpperHalfPlane.mk { re := a, im := Real.exp y₁ } ⋯) …
  -/
  exacts [congr_arg₂ _ (log_exp _) (log_exp _), rfl]
  /-
    🎉 no goals
  -/


theorem isometry_real_vadd (a : ℝ) : Isometry (a +ᵥ · : ℍ → ℍ) :=
                                      /-
                                        a : Real
                                        y₁ y₂ : UpperHalfPlane
                                        ⊢ Eq (Dist.dist (HVAdd.hVAdd a y₁) (HVAdd.hVAdd a y₂)) (Dist.dist y₁ y₂)
                                      -/
  Isometry.of_dist_eq fun y₁ y₂ => by simp only [dist_eq, coe_vadd, vadd_im, dist_add_left]
                                      /-
                                        🎉 no goals
                                      -/


theorem isometry_pos_mul (a : { x : ℝ // 0 < x }) : Isometry (a • · : ℍ → ℍ) := by
  /-
    a : Subtype fun x => LT.lt 0 x
    ⊢ Isometry fun x => HSMul.hSMul a x
  -/
  refine Isometry.of_dist_eq fun y₁ y₂ => ?_
  /-
    a : Subtype fun x => LT.lt 0 x
    y₁ y₂ : UpperHalfPlane
    ⊢ Eq (Dist.dist (HSMul.hSMul a y₁) (HSMul.hSMul a y₂)) (Dist.dist y₁ y₂)
  -/
  simp only [dist_eq, coe_pos_real_smul, pos_real_im]; congr 2
  rw [dist_smul₀, mul_mul_mul_comm, Real.sqrt_mul (mul_self_nonneg _), Real.sqrt_mul_self_eq_abs,
    Real.norm_eq_abs, mul_left_comm]
  /-
    case e_a.e_x
    a : Subtype fun x => LT.lt 0 x
    y₁ y₂ : UpperHalfPlane
    ⊢ Eq (HDiv.hDiv (HMul.hMul (abs ↑a) (Dist.dist ↑y₁ ↑y₂)) (HMul.hMul (abs ↑a) ( …
  -/
  exact mul_div_mul_left _ _ (mt _root_.abs_eq_zero.1 a.2.ne')
  /-
    🎉 no goals
  -/


/-- `SL(2, ℝ)` acts on the upper half plane as an isometry. -/
instance : IsometricSMul SL(2, ℝ) ℍ :=
  ⟨fun g => by
    have h₀ : Isometry (fun z => ModularGroup.S • z : ℍ → ℍ) :=
      Isometry.of_dist_eq fun y₁ y₂ => by
        have h₁ : 0 ≤ im y₁ * im y₂ := mul_nonneg y₁.property.le y₂.property.le
        have h₂ : Complex.abs (y₁ * y₂) ≠ 0 := by simp [y₁.ne_zero, y₂.ne_zero]
        simp only [dist_eq, modular_S_smul, inv_neg, neg_div, div_mul_div_comm, coe_mk, mk_im,
          div_one, Complex.inv_im, Complex.neg_im, coe_im, neg_neg, Complex.normSq_neg,
          mul_eq_mul_left_iff, Real.arsinh_inj, one_ne_zero,
          dist_neg_neg, mul_neg, neg_mul, dist_inv_inv₀ y₁.ne_zero y₂.ne_zero, ←
          AbsoluteValue.map_mul, ← Complex.normSq_mul, Real.sqrt_div h₁, ← Complex.abs_apply,
          mul_div (2 : ℝ), div_div_div_comm, div_self h₂, Complex.norm_eq_abs]
    /-
      z w : UpperHalfPlane
      r : Real
      g : Matrix.SpecialLinearGroup (Fin 2) Real
      h₀ : Isometry fun z => HSMul.hSMul ModularGroup.S z
      ⊢ Isometry fun x => HSMul.hSMul g x
    -/
    by_cases hc : g 1 0 = 0
      /-
        case pos
        z w : UpperHalfPlane
        r : Real
        g : Matrix.SpecialLinearGroup (Fin 2) Real
        h₀ : Isometry fun z => HSMul.hSMul ModularGroup.S z
        hc : Eq (↑g 1 0) 0
        ⊢ Isometry fun x => HSMul.hSMul g x
      -/
    · obtain ⟨u, v, h⟩ := exists_SL2_smul_eq_of_apply_zero_one_eq_zero g hc
      /-
        case pos.intro.intro
        z w : UpperHalfPlane
        r : Real
        g : Matrix.SpecialLinearGroup (Fin 2) Real
        h₀ : Isometry fun z => HSMul.hSMul ModularGroup.S z
        hc : Eq (↑g 1 0) 0
        u : Subtype fun x => LT.lt 0 x
        v : Real
        h : Eq (fun x => HSMul.hSMul g x) (Function.comp (fun x => HVAdd.hVAdd v x) fu …
        ⊢ Isometry fun x => HSMul.hSMul g x
      -/
      rw [h]
      /-
        case pos.intro.intro
        z w : UpperHalfPlane
        r : Real
        g : Matrix.SpecialLinearGroup (Fin 2) Real
        h₀ : Isometry fun z => HSMul.hSMul ModularGroup.S z
        hc : Eq (↑g 1 0) 0
        u : Subtype fun x => LT.lt 0 x
        v : Real
        h : Eq (fun x => HSMul.hSMul g x) (Function.comp (fun x => HVAdd.hVAdd v x) fu …
        ⊢ Isometry (Function.comp (fun x => HVAdd.hVAdd v x) fun x => HSMul.hSMul u x)
      -/
      exact (isometry_real_vadd v).comp (isometry_pos_mul u)
      /-
        🎉 no goals
      -/
      /-
        case neg
        z w : UpperHalfPlane
        r : Real
        g : Matrix.SpecialLinearGroup (Fin 2) Real
        h₀ : Isometry fun z => HSMul.hSMul ModularGroup.S z
        hc : Not (Eq (↑g 1 0) 0)
        ⊢ Isometry fun x => HSMul.hSMul g x
      -/
    · obtain ⟨u, v, w, h⟩ := exists_SL2_smul_eq_of_apply_zero_one_ne_zero g hc
      /-
        case neg.intro.intro.intro
        z w✝ : UpperHalfPlane
        r : Real
        g : Matrix.SpecialLinearGroup (Fin 2) Real
        h₀ : Isometry fun z => HSMul.hSMul ModularGroup.S z
        hc : Not (Eq (↑g 1 0) 0)
        u : Subtype fun x => LT.lt 0 x
        v w : Real
        h : Eq (fun x => HSMul.hSMul g x) (Function.comp (fun x => HVAdd.hVAdd w x) (F …
        ⊢ Isometry fun x => HSMul.hSMul g x
      -/
      rw [h]
      exact
        (isometry_real_vadd w).comp (h₀.comp <| (isometry_real_vadd v).comp <| isometry_pos_mul u)⟩


