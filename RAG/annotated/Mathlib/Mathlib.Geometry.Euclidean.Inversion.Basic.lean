/-- Inversion in a sphere in an affine space. This map sends each point `x` to the point `y` such
that `y -ᵥ c = (R / dist x c) ^ 2 • (x -ᵥ c)`, where `c` and `R` are the center and the radius the
sphere. -/
def inversion (c : P) (R : ℝ) (x : P) : P :=
  (R / dist x c) ^ 2 • (x -ᵥ c) +ᵥ c


theorem inversion_def :
    inversion = fun (c : P) (R : ℝ) (x : P) => (R / dist x c) ^ 2 • (x -ᵥ c) +ᵥ c :=
  rfl


theorem inversion_eq_lineMap (c : P) (R : ℝ) (x : P) :
    inversion c R x = lineMap c x ((R / dist x c) ^ 2) :=
  rfl


theorem inversion_vsub_center (c : P) (R : ℝ) (x : P) :
    inversion c R x -ᵥ c = (R / dist x c) ^ 2 • (x -ᵥ c) :=
  vadd_vsub _ _


@[simp]
                                                                   /-
                                                                     V : Type u_1
                                                                     P : Type u_2
                                                                     inst✝³ : NormedAddCommGroup V
                                                                     inst✝² : InnerProductSpace Real V
                                                                     inst✝¹ : MetricSpace P
                                                                     inst✝ : NormedAddTorsor V P
                                                                     c : P
                                                                     R : Real
                                                                     ⊢ Eq (EuclideanGeometry.inversion c R c) c
                                                                   -/
theorem inversion_self (c : P) (R : ℝ) : inversion c R c = c := by simp [inversion]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
                                                                    /-
                                                                      V : Type u_1
                                                                      P : Type u_2
                                                                      inst✝³ : NormedAddCommGroup V
                                                                      inst✝² : InnerProductSpace Real V
                                                                      inst✝¹ : MetricSpace P
                                                                      inst✝ : NormedAddTorsor V P
                                                                      c x : P
                                                                      ⊢ Eq (EuclideanGeometry.inversion c 0 x) c
                                                                    -/
theorem inversion_zero_radius (c x : P) : inversion c 0 x = c := by simp [inversion]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem inversion_mul (c : P) (a R : ℝ) (x : P) :
    inversion c (a * R) x = homothety c (a ^ 2) (inversion c R x) := by
  simp only [inversion_eq_lineMap, ← homothety_eq_lineMap, ← homothety_mul_apply, mul_div_assoc,
    mul_pow]


@[simp]
theorem inversion_dist_center (c x : P) : inversion c (dist x c) x = x := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x : P
    ⊢ Eq (EuclideanGeometry.inversion c (Dist.dist x c) x) x
  -/
  rcases eq_or_ne x c with (rfl | hne)
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      x : P
      ⊢ Eq (EuclideanGeometry.inversion x (Dist.dist x x) x) x
    -/
  · apply inversion_self
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      c x : P
      hne : Ne x c
      ⊢ Eq (EuclideanGeometry.inversion c (Dist.dist x c) x) x
    -/
  · rw [inversion, div_self, one_pow, one_smul, vsub_vadd]
    /-
      case inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      c x : P
      hne : Ne x c
      ⊢ Ne (Dist.dist x c) 0
    -/
    rwa [dist_ne_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem inversion_dist_center' (c x : P) : inversion c (dist c x) x = x := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x : P
    ⊢ Eq (EuclideanGeometry.inversion c (Dist.dist c x) x) x
  -/
  rw [dist_comm, inversion_dist_center]
  /-
    🎉 no goals
  -/


theorem inversion_of_mem_sphere (h : x ∈ Metric.sphere c R) : inversion c R x = x :=
  h.out ▸ inversion_dist_center c x


/-- Distance from the image of a point under inversion to the center. This formula accidentally
works for `x = c`. -/
theorem dist_inversion_center (c x : P) (R : ℝ) : dist (inversion c R x) c = R ^ 2 / dist x c := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x : P
    R : Real
    ⊢ Eq (Dist.dist (EuclideanGeometry.inversion c R x) c) (HDiv.hDiv (HPow.hPow R …
  -/
  rcases eq_or_ne x c with (rfl | hx)
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      x : P
      R : Real
      ⊢ Eq (Dist.dist (EuclideanGeometry.inversion x R x) x) (HDiv.hDiv (HPow.hPow R …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x : P
    R : Real
    hx : Ne x c
    ⊢ Eq (Dist.dist (EuclideanGeometry.inversion c R x) c) (HDiv.hDiv (HPow.hPow R …
  -/
  have : dist x c ≠ 0 := dist_ne_zero.2 hx
  -- was `field_simp [inversion, norm_smul, abs_div, ← dist_eq_norm_vsub, sq, mul_assoc]`,
  -- but really slow. Replaced by `simp only ...` to speed up.
  -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): reinstate `field_simp` once it is faster.
  simp (disch := field_simp_discharge) only [inversion, sq, mul_div_assoc', div_mul_eq_mul_div,
    div_div, dist_vadd_left, norm_smul, norm_div, norm_mul, Real.norm_eq_abs, abs_mul_abs_self,
    abs_dist, ← dist_eq_norm_vsub, mul_assoc, eq_div_iff, div_eq_iff]


/-- Distance from the center of an inversion to the image of a point under the inversion. This
formula accidentally works for `x = c`. -/
theorem dist_center_inversion (c x : P) (R : ℝ) : dist c (inversion c R x) = R ^ 2 / dist c x := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x : P
    R : Real
    ⊢ Eq (Dist.dist c (EuclideanGeometry.inversion c R x)) (HDiv.hDiv (HPow.hPow R …
  -/
  rw [dist_comm c, dist_comm c, dist_inversion_center]
  /-
    🎉 no goals
  -/


@[simp]
theorem inversion_inversion (c : P) {R : ℝ} (hR : R ≠ 0) (x : P) :
    inversion c R (inversion c R x) = x := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c : P
    R : Real
    hR : Ne R 0
    x : P
    ⊢ Eq (EuclideanGeometry.inversion c R (EuclideanGeometry.inversion c R x)) x
  -/
  rcases eq_or_ne x c with (rfl | hne)
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      R : Real
      hR : Ne R 0
      x : P
      ⊢ Eq (EuclideanGeometry.inversion x R (EuclideanGeometry.inversion x R x)) x
    -/
  · rw [inversion_self, inversion_self]
    /-
      🎉 no goals
    -/
  · rw [inversion, dist_inversion_center, inversion_vsub_center, smul_smul, ← mul_pow,
      div_mul_div_comm, div_mul_cancel₀ _ (dist_ne_zero.2 hne), ← sq, div_self, one_pow, one_smul,
      vsub_vadd]
    /-
      case inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      c : P
      R : Real
      hR : Ne R 0
      x : P
      hne : Ne x c
      ⊢ Ne (HPow.hPow R 2) 0
    -/
    exact pow_ne_zero _ hR
    /-
      🎉 no goals
    -/


theorem inversion_involutive (c : P) {R : ℝ} (hR : R ≠ 0) : Involutive (inversion c R) :=
  inversion_inversion c hR


theorem inversion_surjective (c : P) {R : ℝ} (hR : R ≠ 0) : Surjective (inversion c R) :=
  (inversion_involutive c hR).surjective


theorem inversion_injective (c : P) {R : ℝ} (hR : R ≠ 0) : Injective (inversion c R) :=
  (inversion_involutive c hR).injective


theorem inversion_bijective (c : P) {R : ℝ} (hR : R ≠ 0) : Bijective (inversion c R) :=
  (inversion_involutive c hR).bijective


theorem inversion_eq_center (hR : R ≠ 0) : inversion c R x = c ↔ x = c :=
  (inversion_injective c hR).eq_iff' <| inversion_self _ _


@[simp]
theorem inversion_eq_center' : inversion c R x = c ↔ x = c ∨ R = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x : P
    R : Real
    ⊢ Iff (Eq (EuclideanGeometry.inversion c R x) c) (Or (Eq x c) (Eq R 0))
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hR : R = 0 <;> simp [inversion_eq_center, hR]
                          /-
                            🎉 no goals
                          -/


theorem center_eq_inversion (hR : R ≠ 0) : c = inversion c R x ↔ x = c :=
  eq_comm.trans (inversion_eq_center hR)


@[simp]
theorem center_eq_inversion' : c = inversion c R x ↔ x = c ∨ R = 0 :=
  eq_comm.trans inversion_eq_center'


/-- Distance between the images of two points under an inversion. -/
theorem dist_inversion_inversion (hx : x ≠ c) (hy : y ≠ c) (R : ℝ) :
    dist (inversion c R x) (inversion c R y) = R ^ 2 / (dist x c * dist y c) * dist x y := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    hx : Ne x c
    hy : Ne y c
    R : Real
    ⊢ Eq (Dist.dist (EuclideanGeometry.inversion c R x) (EuclideanGeometry.inversi …
  -/
  dsimp only [inversion]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    hx : Ne x c
    hy : Ne y c
    R : Real
    ⊢ Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul (HPow.hPow (HDiv.hDiv R (Dist.dist x …
  -/
  simp_rw [dist_vadd_cancel_right, dist_eq_norm_vsub V _ c]
  simpa only [dist_vsub_cancel_right] using
    dist_div_norm_sq_smul (vsub_ne_zero.2 hx) (vsub_ne_zero.2 hy) R


theorem dist_inversion_mul_dist_center_eq (hx : x ≠ c) (hy : y ≠ c) :
    dist (inversion c R x) y * dist x c = dist x (inversion c R y) * dist y c := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hx : Ne x c
    hy : Ne y c
    ⊢ Eq (HMul.hMul (Dist.dist (EuclideanGeometry.inversion c R x) y) (Dist.dist x …
  -/
  rcases eq_or_ne R 0 with rfl | hR; · simp [dist_comm, mul_comm]
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hx : Ne x c
    hy : Ne y c
    hR : Ne R 0
    ⊢ Eq (HMul.hMul (Dist.dist (EuclideanGeometry.inversion c R x) y) (Dist.dist x …
  -/
  have hy' : inversion c R y ≠ c := by simp [*]
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hx : Ne x c
    hy : Ne y c
    hR : Ne R 0
    hy' : Ne (EuclideanGeometry.inversion c R y) c
    ⊢ Eq (HMul.hMul (Dist.dist (EuclideanGeometry.inversion c R x) y) (Dist.dist x …
  -/
  conv in dist _ y => rw [← inversion_inversion c hR y]
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hx : Ne x c
    hy : Ne y c
    hR : Ne R 0
    hy' : Ne (EuclideanGeometry.inversion c R y) c
    ⊢ Eq (HMul.hMul (Dist.dist (EuclideanGeometry.inversion c R x) (EuclideanGeome …
  -/
  rw [dist_inversion_inversion hx hy', dist_inversion_center]
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hx : Ne x c
    hy : Ne y c
    hR : Ne R 0
    hy' : Ne (EuclideanGeometry.inversion c R y) c
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.hPow R 2) (HMul.hMul (Dist.dist x  …
  -/
  have : dist x c ≠ 0 := dist_ne_zero.2 hx
  -- used to be `field_simp`, but was really slow; replaced by `simp only ...` to speed up
  -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): reinstate `field_simp` once it is faster.
  simp (disch := field_simp_discharge) only [mul_div_assoc', div_div_eq_mul_div, div_mul_eq_mul_div,
    div_eq_iff]
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hx : Ne x c
    hy : Ne y c
    hR : Ne R 0
    hy' : Ne (EuclideanGeometry.inversion c R y) c
    this : Ne (Dist.dist x c) 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow R 2) (Dist.dist y c)) (Dist.d …
  -/
  ring
  /-
    🎉 no goals
  -/


include V in
/-- **Ptolemy's inequality**: in a quadrangle `ABCD`, `|AC| * |BD| ≤ |AB| * |CD| + |BC| * |AD|`. If
`ABCD` is a convex cyclic polygon, then this inequality becomes an equality, see
`EuclideanGeometry.mul_dist_add_mul_dist_eq_mul_dist_of_cospherical`. -/
theorem mul_dist_le_mul_dist_add_mul_dist (a b c d : P) :
    dist a c * dist b d ≤ dist a b * dist c d + dist b c * dist a d := by
  -- If one of the points `b`, `c`, `d` is equal to `a`, then the inequality is trivial.
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    ⊢ LE.le (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
  -/
  rcases eq_or_ne b a with (rfl | hb)
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      b c d : P
      ⊢ LE.le (HMul.hMul (Dist.dist b c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
    -/
  · rw [dist_self, zero_mul, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    hb : Ne b a
    ⊢ LE.le (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
  -/
  rcases eq_or_ne c a with (rfl | hc)
    /-
      case inr.inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      b c d : P
      hb : Ne b c
      ⊢ LE.le (HMul.hMul (Dist.dist c c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
    -/
  · rw [dist_self, zero_mul]
    /-
      case inr.inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      b c d : P
      hb : Ne b c
      ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul (Dist.dist c b) (Dist.dist c d)) (HMul.hMul (D …
    -/
    apply_rules [add_nonneg, mul_nonneg, dist_nonneg]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    hb : Ne b a
    hc : Ne c a
    ⊢ LE.le (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
  -/
  rcases eq_or_ne d a with (rfl | hd)
    /-
      case inr.inr.inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      b c d : P
      hb : Ne b d
      hc : Ne c d
      ⊢ LE.le (HMul.hMul (Dist.dist d c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
    -/
  · rw [dist_self, mul_zero, add_zero, dist_comm d, dist_comm d, mul_comm]
    /-
      🎉 no goals
    -/
  /- Otherwise, we apply the triangle inequality to `EuclideanGeometry.inversion a 1 b`,
    `EuclideanGeometry.inversion a 1 c`, and `EuclideanGeometry.inversion a 1 d`. -/
  /-
    case inr.inr.inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    hb : Ne b a
    hc : Ne c a
    hd : Ne d a
    ⊢ LE.le (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
  -/
  have H := dist_triangle (inversion a 1 b) (inversion a 1 c) (inversion a 1 d)
  rw [dist_inversion_inversion hb hd, dist_inversion_inversion hb hc,
    dist_inversion_inversion hc hd, one_pow] at H
  /-
    case inr.inr.inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    hb : Ne b a
    hc : Ne c a
    hd : Ne d a
    H : LE.le (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (Dist.dist b a) (Dist.dist d a))) …
    ⊢ LE.le (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
  -/
  rw [← dist_pos] at hb hc hd
  /-
    case inr.inr.inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    hb : LT.lt 0 (Dist.dist b a)
    hc : LT.lt 0 (Dist.dist c a)
    hd : LT.lt 0 (Dist.dist d a)
    H : LE.le (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (Dist.dist b a) (Dist.dist d a))) …
    ⊢ LE.le (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HAdd.hAdd (HMul.hMul (Dis …
  -/
  rw [← div_le_div_iff_of_pos_right (mul_pos hb (mul_pos hc hd))]
  /-
    case inr.inr.inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d : P
    hb : LT.lt 0 (Dist.dist b a)
    hc : LT.lt 0 (Dist.dist c a)
    hd : LT.lt 0 (Dist.dist d a)
    H : LE.le (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (Dist.dist b a) (Dist.dist d a))) …
    ⊢ LE.le (HDiv.hDiv (HMul.hMul (Dist.dist a c) (Dist.dist b d)) (HMul.hMul (Dis …
  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  convert H using 1 <;> (field_simp [hb.ne', hc.ne', hd.ne', dist_comm a]; ring)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


protected theorem Filter.Tendsto.inversion {α : Type*} {x c : P} {R : ℝ} {l : Filter α}
    {fc fx : α → P} {fR : α → ℝ} (hc : Tendsto fc l (𝓝 c)) (hR : Tendsto fR l (𝓝 R))
    (hx : Tendsto fx l (𝓝 x)) (hne : x ≠ c) :
    Tendsto (fun a ↦ inversion (fc a) (fR a) (fx a)) l (𝓝 (inversion c R x)) :=
  (((hR.div (hx.dist hc) <| dist_ne_zero.2 hne).pow 2).smul (hx.vsub hc)).vadd hc


protected nonrec theorem ContinuousWithinAt.inversion (hc : ContinuousWithinAt c s a₀)
    (hR : ContinuousWithinAt R s a₀) (hx : ContinuousWithinAt x s a₀) (hne : x a₀ ≠ c a₀) :
    ContinuousWithinAt (fun a ↦ inversion (c a) (R a) (x a)) s a₀ :=
  hc.inversion hR hx hne


protected nonrec theorem ContinuousAt.inversion (hc : ContinuousAt c a₀) (hR : ContinuousAt R a₀)
    (hx : ContinuousAt x a₀) (hne : x a₀ ≠ c a₀) :
    ContinuousAt (fun a ↦ inversion (c a) (R a) (x a)) a₀ :=
  hc.inversion hR hx hne


protected theorem ContinuousOn.inversion (hc : ContinuousOn c s) (hR : ContinuousOn R s)
    (hx : ContinuousOn x s) (hne : ∀ a ∈ s, x a ≠ c a) :
    ContinuousOn (fun a ↦ inversion (c a) (R a) (x a)) s := fun a ha ↦
  (hc a ha).inversion (hR a ha) (hx a ha) (hne a ha)


protected theorem Continuous.inversion (hc : Continuous c) (hR : Continuous R) (hx : Continuous x)
    (hne : ∀ a, x a ≠ c a) : Continuous (fun a ↦ inversion (c a) (R a) (x a)) :=
  continuous_iff_continuousAt.2 fun _ ↦
    hc.continuousAt.inversion hR.continuousAt hx.continuousAt (hne _)

