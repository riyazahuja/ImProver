/-- The integral of the Gaussian function over the vertical edges of a rectangle
with vertices at `(±T, 0)` and `(±T, c)`. -/
def verticalIntegral (b : ℂ) (c T : ℝ) : ℂ :=
  ∫ y : ℝ in (0 : ℝ)..c, I * (cexp (-b * (T + y * I) ^ 2) - cexp (-b * (T - y * I) ^ 2))


/-- Explicit formula for the norm of the Gaussian function along the vertical
edges. -/
theorem norm_cexp_neg_mul_sq_add_mul_I (b : ℂ) (c T : ℝ) :
    ‖cexp (-b * (T + c * I) ^ 2)‖ = exp (-(b.re * T ^ 2 - 2 * b.im * c * T - b.re * c ^ 2)) := by
  /-
    b : Complex
    c T : Real
    ⊢ Eq (Norm.norm (Complex.exp (HMul.hMul (Neg.neg b) (HPow.hPow (HAdd.hAdd (↑T) …
  -/
  rw [Complex.norm_eq_abs, Complex.abs_exp, neg_mul, neg_re, ← re_add_im b]
  /-
    b : Complex
    c T : Real
    ⊢ Eq (Real.exp (Neg.neg (HMul.hMul (HAdd.hAdd (↑b.re) (HMul.hMul (↑b.im) Compl …
  -/
  simp only [sq, re_add_im, mul_re, mul_im, add_re, add_im, ofReal_re, ofReal_im, I_re, I_im]
  /-
    b : Complex
    c T : Real
    ⊢ Eq (Real.exp (Neg.neg (HSub.hSub (HMul.hMul b.re (HSub.hSub (HMul.hMul (HAdd …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem norm_cexp_neg_mul_sq_add_mul_I' (hb : b.re ≠ 0) (c T : ℝ) :
    ‖cexp (-b * (T + c * I) ^ 2)‖ =
      exp (-(b.re * (T - b.im * c / b.re) ^ 2 - c ^ 2 * (b.im ^ 2 / b.re + b.re))) := by
  have :
    b.re * T ^ 2 - 2 * b.im * c * T - b.re * c ^ 2 =
      b.re * (T - b.im * c / b.re) ^ 2 - c ^ 2 * (b.im ^ 2 / b.re + b.re) := by
    field_simp; ring
  /-
    b : Complex
    hb : Ne b.re 0
    c T : Real
    this : Eq (HSub.hSub (HSub.hSub (HMul.hMul b.re (HPow.hPow T 2)) (HMul.hMul (H …
    ⊢ Eq (Norm.norm (Complex.exp (HMul.hMul (Neg.neg b) (HPow.hPow (HAdd.hAdd (↑T) …
  -/
  rw [norm_cexp_neg_mul_sq_add_mul_I, this]
  /-
    🎉 no goals
  -/


theorem verticalIntegral_norm_le (hb : 0 < b.re) (c : ℝ) {T : ℝ} (hT : 0 ≤ T) :
    ‖verticalIntegral b c T‖ ≤
      (2 : ℝ) * |c| * exp (-(b.re * T ^ 2 - (2 : ℝ) * |b.im| * |c| * T - b.re * c ^ 2)) := by
  -- first get uniform bound for integrand
  have vert_norm_bound :
    ∀ {T : ℝ},
      0 ≤ T →
        ∀ {c y : ℝ},
          |y| ≤ |c| →
            ‖cexp (-b * (T + y * I) ^ 2)‖ ≤
              exp (-(b.re * T ^ 2 - (2 : ℝ) * |b.im| * |c| * T - b.re * c ^ 2)) := by
    intro T hT c y hy
    rw [norm_cexp_neg_mul_sq_add_mul_I b]
    gcongr exp (- (_ - ?_ * _ - _ * ?_))
    · (conv_lhs => rw [mul_assoc]); (conv_rhs => rw [mul_assoc])
      gcongr _ * ?_
      refine (le_abs_self _).trans ?_
      rw [abs_mul]
      gcongr
    · rwa [sq_le_sq]
  -- now main proof
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c T : Real
    hT : LE.le 0 T
    vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
    ⊢ LE.le (Norm.norm (GaussianFourier.verticalIntegral b c T)) (HMul.hMul (HMul. …
  -/
  apply (intervalIntegral.norm_integral_le_of_norm_le_const _).trans
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      ⊢ LE.le (HMul.hMul ?m.31142 (abs (HSub.hSub c 0))) (HMul.hMul (HMul.hMul 2 (ab …
    -/
  · rw [sub_zero]
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      ⊢ LE.le (HMul.hMul ?m.31142 (abs c)) (HMul.hMul (HMul.hMul 2 (abs c)) (Real.ex …
    -/
    conv_lhs => simp only [mul_comm _ |c|]
    conv_rhs =>
      conv =>
        congr
        rw [mul_comm]
      rw [mul_assoc]
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      ⊢ ∀ (x : Real), Membership.mem (Set.uIoc 0 c) x → LE.le (Norm.norm (HMul.hMul  …
    -/
  · intro y hy
    have absy : |y| ≤ |c| := by
      rcases le_or_lt 0 c with (h | h)
      · rw [uIoc_of_le h] at hy
        rw [abs_of_nonneg h, abs_of_pos hy.1]
        exact hy.2
      · rw [uIoc_of_ge h.le] at hy
        rw [abs_of_neg h, abs_of_nonpos hy.2, neg_le_neg_iff]
        exact hy.1.le
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      y : Real
      hy : Membership.mem (Set.uIoc 0 c) y
      absy : LE.le (abs y) (abs c)
      ⊢ LE.le (Norm.norm (HMul.hMul Complex.I (HSub.hSub (Complex.exp (HMul.hMul (Ne …
    -/
    rw [norm_mul, Complex.norm_eq_abs, abs_I, one_mul, two_mul]
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      y : Real
      hy : Membership.mem (Set.uIoc 0 c) y
      absy : LE.le (abs y) (abs c)
      ⊢ LE.le (Norm.norm (HSub.hSub (Complex.exp (HMul.hMul (Neg.neg b) (HPow.hPow ( …
    -/
    refine (norm_sub_le _ _).trans (add_le_add (vert_norm_bound hT absy) ?_)
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      y : Real
      hy : Membership.mem (Set.uIoc 0 c) y
      absy : LE.le (abs y) (abs c)
      ⊢ LE.le (Norm.norm (Complex.exp (HMul.hMul (Neg.neg b) (HPow.hPow (HSub.hSub ( …
    -/
    rw [← abs_neg y] at absy
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c T : Real
      hT : LE.le 0 T
      vert_norm_bound : ∀ {T : Real}, LE.le 0 T → ∀ {c y : Real}, LE.le (abs y) (abs …
      y : Real
      hy : Membership.mem (Set.uIoc 0 c) y
      absy : LE.le (abs (Neg.neg y)) (abs c)
      ⊢ LE.le (Norm.norm (Complex.exp (HMul.hMul (Neg.neg b) (HPow.hPow (HSub.hSub ( …
    -/
    simpa only [neg_mul, ofReal_neg] using vert_norm_bound hT absy
    /-
      🎉 no goals
    -/


theorem tendsto_verticalIntegral (hb : 0 < b.re) (c : ℝ) :
    Tendsto (verticalIntegral b c) atTop (𝓝 0) := by
  -- complete proof using squeeze theorem:
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (GaussianFourier.verticalIntegral b c) Filter.atTop (nhds 0)
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  refine
    tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds ?_
      (Eventually.of_forall fun _ => norm_nonneg _)
      ((eventually_ge_atTop (0 : ℝ)).mp
        (Eventually.of_forall fun T hT => verticalIntegral_norm_le hb c hT))
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (fun T => HMul.hMul (HMul.hMul 2 (abs c)) (Real.exp (Neg.neg  …
  -/
  rw [(by ring : 0 = 2 * |c| * 0)]
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (fun T => HMul.hMul (HMul.hMul 2 (abs c)) (Real.exp (Neg.neg  …
  -/
  refine (tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp ?_)).const_mul _
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (fun T => HSub.hSub (HSub.hSub (HMul.hMul b.re (HPow.hPow T 2 …
  -/
  apply tendsto_atTop_add_const_right
  /-
    case hf
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (fun x => HSub.hSub (HMul.hMul b.re (HPow.hPow x 2)) (HMul.hM …
  -/
  simp_rw [sq, ← mul_assoc, ← sub_mul]
  /-
    case hf
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HSub.hSub (HMul.hMul b.re x) (HMul.hMul  …
  -/
  refine Tendsto.atTop_mul_atTop (tendsto_atTop_add_const_right _ _ ?_) tendsto_id
  /-
    case hf
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (HMul.hMul b.re) Filter.atTop Filter.atTop
  -/
  exact (tendsto_const_mul_atTop_of_pos hb).mpr tendsto_id
  /-
    🎉 no goals
  -/


theorem integrable_cexp_neg_mul_sq_add_real_mul_I (hb : 0 < b.re) (c : ℝ) :
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c : Real
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable fun x : ℝ => cexp (-b * (x + c * I) ^ 2) := by
    /-
      🎉 no goals
    -/
  refine
    ⟨(Complex.continuous_exp.comp
          (continuous_const.mul
            ((continuous_ofReal.add continuous_const).pow 2))).aestronglyMeasurable,
      ?_⟩
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => Complex.exp (HMul.hMul (Neg.neg b) …
  -/
  rw [← hasFiniteIntegral_norm_iff]
  simp_rw [norm_cexp_neg_mul_sq_add_mul_I' hb.ne', neg_sub _ (c ^ 2 * _),
    sub_eq_add_neg _ (b.re * _), Real.exp_add]
  suffices Integrable fun x : ℝ => exp (-(b.re * x ^ 2)) by
    exact (Integrable.comp_sub_right this (b.im * c / b.re)).hasFiniteIntegral.const_mul _
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ MeasureTheory.Integrable (fun x => Real.exp (Neg.neg (HMul.hMul b.re (HPow.h …
  -/
  simp_rw [← neg_mul]
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ MeasureTheory.Integrable (fun x => Real.exp (HMul.hMul (Neg.neg b.re) (HPow. …
  -/
  apply integrable_exp_neg_mul_sq hb
  /-
    🎉 no goals
  -/


theorem integral_cexp_neg_mul_sq_add_real_mul_I (hb : 0 < b.re) (c : ℝ) :
    ∫ x : ℝ, cexp (-b * (x + c * I) ^ 2) = (π / b) ^ (1 / 2 : ℂ) := by
  refine
    tendsto_nhds_unique
      (intervalIntegral_tendsto_integral (integrable_cexp_neg_mul_sq_add_real_mul_I hb c)
        tendsto_neg_atTop_atBot tendsto_id)
      ?_
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Complex.exp (HMul.hMul ( …
  -/
  set I₁ := fun T => ∫ x : ℝ in -T..T, cexp (-b * (x + c * I) ^ 2) with HI₁
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    I₁ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    HI₁ : Eq I₁ fun T => intervalIntegral (fun x => Complex.exp (HMul.hMul (Neg.ne …
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Complex.exp (HMul.hMul ( …
  -/
  let I₂ := fun T : ℝ => ∫ x : ℝ in -T..T, cexp (-b * (x : ℂ) ^ 2)
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    I₁ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    HI₁ : Eq I₁ fun T => intervalIntegral (fun x => Complex.exp (HMul.hMul (Neg.ne …
    I₂ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Complex.exp (HMul.hMul ( …
  -/
  let I₄ := fun T : ℝ => ∫ y : ℝ in (0 : ℝ)..c, cexp (-b * (T + y * I) ^ 2)
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    I₁ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    HI₁ : Eq I₁ fun T => intervalIntegral (fun x => Complex.exp (HMul.hMul (Neg.ne …
    I₂ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    I₄ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Complex.exp (HMul.hMul ( …
  -/
  let I₅ := fun T : ℝ => ∫ y : ℝ in (0 : ℝ)..c, cexp (-b * (-T + y * I) ^ 2)
  have C : ∀ T : ℝ, I₂ T - I₁ T + I * I₄ T - I * I₅ T = 0 := by
    intro T
    have :=
      integral_boundary_rect_eq_zero_of_differentiableOn (fun z => cexp (-b * z ^ 2)) (-T)
        (T + c * I)
        (by
          refine Differentiable.differentiableOn (Differentiable.const_mul ?_ _).cexp
          exact differentiable_pow 2)
    simpa only [neg_im, ofReal_im, neg_zero, ofReal_zero, zero_mul, add_zero, neg_re,
      ofReal_re, add_re, mul_re, I_re, mul_zero, I_im, tsub_zero, add_im, mul_im,
      mul_one, zero_add, Algebra.id.smul_eq_mul, ofReal_neg] using this
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    I₁ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    HI₁ : Eq I₁ fun T => intervalIntegral (fun x => Complex.exp (HMul.hMul (Neg.ne …
    I₂ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    I₄ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    I₅ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    C : ∀ (T : Real), Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (I₂ T) (I₁ T)) (HMul.hMu …
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Complex.exp (HMul.hMul ( …
  -/
  simp_rw [id, ← HI₁]
  have : I₁ = fun T : ℝ => I₂ T + verticalIntegral b c T := by
    ext1 T
    specialize C T
    rw [sub_eq_zero] at C
    unfold verticalIntegral
    rw [integral_const_mul, intervalIntegral.integral_sub]
    · simp_rw [(fun a b => by rw [sq]; ring_nf : ∀ a b : ℂ, (a - b * I) ^ 2 = (-a + b * I) ^ 2)]
      change I₁ T = I₂ T + I * (I₄ T - I₅ T)
      rw [mul_sub, ← C]
      abel
    all_goals apply Continuous.intervalIntegrable; continuity
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    I₁ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    HI₁ : Eq I₁ fun T => intervalIntegral (fun x => Complex.exp (HMul.hMul (Neg.ne …
    I₂ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    I₄ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    I₅ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    C : ∀ (T : Real), Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (I₂ T) (I₁ T)) (HMul.hMu …
    this : Eq I₁ fun T => HAdd.hAdd (I₂ T) (GaussianFourier.verticalIntegral b c T)
    ⊢ Filter.Tendsto I₁ Filter.atTop (nhds (HPow.hPow (HDiv.hDiv (↑Real.pi) b) (1  …
  -/
  rw [this, ← add_zero ((π / b : ℂ) ^ (1 / 2 : ℂ)), ← integral_gaussian_complex hb]
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Real
    I₁ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    HI₁ : Eq I₁ fun T => intervalIntegral (fun x => Complex.exp (HMul.hMul (Neg.ne …
    I₂ : Real → Complex := fun T => intervalIntegral (fun x => Complex.exp (HMul.h …
    I₄ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    I₅ : Real → Complex := fun T => intervalIntegral (fun y => Complex.exp (HMul.h …
    C : ∀ (T : Real), Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (I₂ T) (I₁ T)) (HMul.hMu …
    this : Eq I₁ fun T => HAdd.hAdd (I₂ T) (GaussianFourier.verticalIntegral b c T)
    ⊢ Filter.Tendsto (fun T => HAdd.hAdd (I₂ T) (GaussianFourier.verticalIntegral  …
  -/
  refine Tendsto.add ?_ (tendsto_verticalIntegral hb c)
  exact
    intervalIntegral_tendsto_integral (integrable_cexp_neg_mul_sq hb) tendsto_neg_atTop_atBot
      tendsto_id


theorem _root_.integral_cexp_quadratic (hb : b.re < 0) (c d : ℂ) :
    ∫ x : ℝ, cexp (b * x ^ 2 + c * x + d) = (π / -b) ^ (1 / 2 : ℂ) * cexp (d - c^2 / (4 * b)) := by
  /-
    b : Complex
    hb : LT.lt b.re 0
    c d : Complex
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
  -/
  have hb' : b ≠ 0 := by contrapose! hb; rw [hb, zero_re]
  have h (x : ℝ) : cexp (b * x ^ 2 + c * x + d) =
      cexp (- -b * (x + c / (2 * b)) ^ 2) * cexp (d - c ^ 2 / (4 * b)) := by
    simp_rw [← Complex.exp_add]
    congr 1
    field_simp
    ring_nf
  /-
    b : Complex
    hb : LT.lt b.re 0
    c d : Complex
    hb' : Ne b 0
    h : ∀ (x : Real), Eq (Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HPow.hPo …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
  -/
  simp_rw [h, integral_mul_right]
  /-
    b : Complex
    hb : LT.lt b.re 0
    c d : Complex
    hb' : Ne b 0
    h : ∀ (x : Real), Eq (Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HPow.hPo …
    ⊢ Eq (HMul.hMul (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun  …
  -/
  rw [← re_add_im (c / (2 * b))]
  /-
    b : Complex
    hb : LT.lt b.re 0
    c d : Complex
    hb' : Ne b 0
    h : ∀ (x : Real), Eq (Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HPow.hPo …
    ⊢ Eq (HMul.hMul (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun  …
  -/
  simp_rw [← add_assoc, ← ofReal_add]
  rw [integral_add_right_eq_self fun a : ℝ ↦ cexp (- -b * (↑a + ↑(c / (2 * b)).im * I) ^ 2),
    integral_cexp_neg_mul_sq_add_real_mul_I ((neg_re b).symm ▸ (neg_pos.mpr hb))]


lemma _root_.integrable_cexp_quadratic' (hb : b.re < 0) (c d : ℂ) :
    /-
      b : Complex
      hb : LT.lt b.re 0
      c d : Complex
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable (fun (x : ℝ) ↦ cexp (b * x ^ 2 + c * x + d)) := by
    /-
      🎉 no goals
    -/
  /-
    b : Complex
    hb : LT.lt b.re 0
    c d : Complex
    ⊢ MeasureTheory.Integrable (fun x => Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  have hb' : b ≠ 0 := by contrapose! hb; rw [hb, zero_re]
  /-
    b : Complex
    hb : LT.lt b.re 0
    c d : Complex
    hb' : Ne b 0
    ⊢ MeasureTheory.Integrable (fun x => Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  by_contra H
  simpa [hb', pi_ne_zero, Complex.exp_ne_zero, integral_undef H]
    using integral_cexp_quadratic hb c d


lemma _root_.integrable_cexp_quadratic (hb : 0 < b.re) (c d : ℂ) :
    /-
      b : Complex
      hb : LT.lt 0 b.re
      c d : Complex
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable (fun (x : ℝ) ↦ cexp (-b * x ^ 2 + c * x + d)) := by
    /-
      🎉 no goals
    -/
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c d : Complex
    ⊢ MeasureTheory.Integrable (fun x => Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  have : (-b).re < 0 := by simpa using hb
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c d : Complex
    this : LT.lt (Neg.neg b).re 0
    ⊢ MeasureTheory.Integrable (fun x => Complex.exp (HAdd.hAdd (HAdd.hAdd (HMul.h …
  -/
  exact integrable_cexp_quadratic' this c d
  /-
    🎉 no goals
  -/


theorem _root_.fourierIntegral_gaussian (hb : 0 < b.re) (t : ℂ) :
    ∫ x : ℝ, cexp (I * t * x) * cexp (-b * x ^ 2) =
    (π / b) ^ (1 / 2 : ℂ) * cexp (-t ^ 2 / (4 * b)) := by
  /-
    b : Complex
    hb : LT.lt 0 b.re
    t : Complex
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => HMul.h …
  -/
  conv => enter [1, 2, x]; rw [← Complex.exp_add, add_comm, ← add_zero (-b * x ^ 2 + I * t * x)]
  rw [integral_cexp_quadratic (show (-b).re < 0 by rwa [neg_re, neg_lt_zero]), neg_neg, zero_sub,
    mul_neg, div_neg, neg_neg, mul_pow, I_sq, neg_one_mul, mul_comm]


@[deprecated (since := "2024-02-21")]
alias _root_.fourier_transform_gaussian := fourierIntegral_gaussian


theorem _root_.fourierIntegral_gaussian_pi' (hb : 0 < b.re) (c : ℂ) :
    (𝓕 fun x : ℝ => cexp (-π * b * x ^ 2 + 2 * π * c * x)) = fun t : ℝ =>
    1 / b ^ (1 / 2 : ℂ) * cexp (-π / b * (t + I * c) ^ 2) := by
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Complex
    ⊢ Eq (Real.fourierIntegral fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hM …
  -/
  haveI : b ≠ 0 := by contrapose! hb; rw [hb, zero_re]
  have h : (-↑π * b).re < 0 := by
    simpa only [neg_mul, neg_re, re_ofReal_mul, neg_lt_zero] using mul_pos pi_pos hb
  /-
    b : Complex
    hb : LT.lt 0 b.re
    c : Complex
    this : Ne b 0
    h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
    ⊢ Eq (Real.fourierIntegral fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hM …
  -/
  ext1 t
  /-
    case h
    b : Complex
    hb : LT.lt 0 b.re
    c : Complex
    this : Ne b 0
    h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
    t : Real
    ⊢ Eq (Real.fourierIntegral (fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.h …
  -/
  simp_rw [fourierIntegral_real_eq_integral_exp_smul, smul_eq_mul, ← Complex.exp_add, ← add_assoc]
  have (x : ℝ) : ↑(-2 * π * x * t) * I + -π * b * x ^ 2 + 2 * π * c * x =
    -π * b * x ^ 2 + (-2 * π * I * t + 2 * π * c) * x + 0 := by push_cast; ring
  /-
    case h
    b : Complex
    hb : LT.lt 0 b.re
    c : Complex
    this✝ : Ne b 0
    h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
    t : Real
    this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  simp_rw [this, integral_cexp_quadratic h, neg_mul, neg_neg]
  /-
    case h
    b : Complex
    hb : LT.lt 0 b.re
    c : Complex
    this✝ : Ne b 0
    h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
    t : Real
    this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
    ⊢ Eq (HMul.hMul (HPow.hPow (HDiv.hDiv (↑Real.pi) (HMul.hMul (↑Real.pi) b)) (1  …
  -/
  congr 2
    /-
      case h.e_a
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Eq (HPow.hPow (HDiv.hDiv (↑Real.pi) (HMul.hMul (↑Real.pi) b)) (1 / 2)) (HDiv …
    -/
  · rw [← div_div, div_self <| ofReal_ne_zero.mpr pi_ne_zero, one_div, inv_cpow, ← one_div]
    /-
      case h.e_a.hx
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Ne b.arg Real.pi
    -/
    rw [Ne, arg_eq_pi_iff, not_and_or, not_lt]
    /-
      case h.e_a.hx
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Or (LE.le 0 b.re) (Not (Eq b.im 0))
    -/
    exact Or.inl hb.le
    /-
      🎉 no goals
    -/
    /-
      case h.e_a.e_z
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Eq (HSub.hSub 0 (HDiv.hDiv (HPow.hPow (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.h …
    -/
  · field_simp [ofReal_ne_zero.mpr pi_ne_zero]
    /-
      case h.e_a.e_z
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMu …
    -/
    ring_nf
    /-
      case h.e_a.e_z
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMu …
    -/
    simp only [I_sq]
    /-
      case h.e_a.e_z
      b : Complex
      hb : LT.lt 0 b.re
      c : Complex
      this✝ : Ne b 0
      h : LT.lt (HMul.hMul (Neg.neg ↑Real.pi) b).re 0
      t : Real
      this : ∀ (x : Real), Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hM …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMu …
    -/
    ring
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-21")]
alias _root_.fourier_transform_gaussian_pi' := _root_.fourierIntegral_gaussian_pi'


theorem _root_.fourierIntegral_gaussian_pi (hb : 0 < b.re) :
    (𝓕 fun (x : ℝ) ↦ cexp (-π * b * x ^ 2)) =
    fun t : ℝ ↦ 1 / b ^ (1 / 2 : ℂ) * cexp (-π / b * t ^ 2) := by
  /-
    b : Complex
    hb : LT.lt 0 b.re
    ⊢ Eq (Real.fourierIntegral fun x => Complex.exp (HMul.hMul (HMul.hMul (Neg.neg …
  -/
  simpa only [mul_zero, zero_mul, add_zero] using fourierIntegral_gaussian_pi' hb 0
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-21")]
alias root_.fourier_transform_gaussian_pi := _root_.fourierIntegral_gaussian_pi


theorem integrable_cexp_neg_sum_mul_add {ι : Type*} [Fintype ι] {b : ι → ℂ}
    (hb : ∀ i, 0 < (b i).re) (c : ι → ℂ) :
    /-
      b✝ : Complex
      V : Type u_1
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      ι : Type u_2
      inst✝ : Fintype ι
      b : ι → Complex
      hb : ∀ (i : ι), LT.lt 0 (b i).re
      c : ι → Complex
      ⊢ MeasureTheory.Measure (ι → Real)
    -/
    Integrable (fun (v : ι → ℝ) ↦ cexp (- ∑ i, b i * (v i : ℂ) ^ 2 + ∑ i, c i * v i)) := by
    /-
      🎉 no goals
    -/
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (Neg.neg (Finset.u …
  -/
  simp_rw [← Finset.sum_neg_distrib, ← Finset.sum_add_distrib, Complex.exp_sum, ← neg_mul]
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    ⊢ MeasureTheory.Integrable (fun v => Finset.univ.prod fun x => Complex.exp (HA …
  -/
  apply Integrable.fintype_prod (f := fun i (v : ℝ) ↦ cexp (-b i * v^2 + c i * v)) (fun i ↦ ?_)
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    i : ι
    ⊢ MeasureTheory.Integrable ((fun i v => Complex.exp (HAdd.hAdd (HMul.hMul (Neg …
  -/
  convert integrable_cexp_quadratic (hb i) (c i) 0 using 3 with x
  /-
    case h.e'_6.h.h.e'_1
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    i : ι
    x : Real
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (b i)) (HPow.hPow (↑x) 2)) (HMul.hMul (c i …
  -/
  simp only [add_zero]
  /-
    🎉 no goals
  -/


theorem integrable_cexp_neg_mul_sum_add {ι : Type*} [Fintype ι] (hb : 0 < b.re) (c : ι → ℂ) :
    /-
      b : Complex
      V : Type u_1
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : ι → Complex
      ⊢ MeasureTheory.Measure (ι → Real)
    -/
    Integrable (fun (v : ι → ℝ) ↦ cexp (- b * ∑ i, (v i : ℂ) ^ 2 + ∑ i, c i * v i)) := by
    /-
      🎉 no goals
    -/
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : ι → Complex
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (HMul.hMul (Neg.ne …
  -/
  simp_rw [neg_mul, Finset.mul_sum]
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : ι → Complex
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (Neg.neg (Finset.u …
  -/
  exact integrable_cexp_neg_sum_mul_add (fun _ ↦ hb) c
  /-
    🎉 no goals
  -/


theorem integrable_cexp_neg_mul_sq_norm_add_of_euclideanSpace
    {ι : Type*} [Fintype ι] (hb : 0 < b.re) (c : ℂ) (w : EuclideanSpace ℝ ι) :
    /-
      b : Complex
      V : Type u_1
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      ⊢ MeasureTheory.Measure (EuclideanSpace Real ι)
    -/
    Integrable (fun (v : EuclideanSpace ℝ ι) ↦ cexp (- b * ‖v‖^2 + c * ⟪w, v⟫)) := by
    /-
      🎉 no goals
    -/
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (HMul.hMul (Neg.ne …
  -/
  have := EuclideanSpace.volume_preserving_measurableEquiv ι
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (HMul.hMul (Neg.ne …
  -/
  rw [← MeasurePreserving.integrable_comp_emb this.symm (MeasurableEquiv.measurableEmbedding _)]
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
    ⊢ MeasureTheory.Integrable (Function.comp (fun v => Complex.exp (HAdd.hAdd (HM …
  -/
  simp only [neg_mul, Function.comp_def]
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
    ⊢ MeasureTheory.Integrable (fun x => Complex.exp (HAdd.hAdd (Neg.neg (HMul.hMu …
  -/
  convert integrable_cexp_neg_mul_sum_add hb (fun i ↦ c * w i) using 3 with v
  simp only [EuclideanSpace.measurableEquiv, MeasurableEquiv.symm_mk, MeasurableEquiv.coe_mk,
    EuclideanSpace.norm_eq, WithLp.equiv_symm_pi_apply, Real.norm_eq_abs, sq_abs, PiLp.inner_apply,
    RCLike.inner_apply, conj_trivial, ofReal_sum, ofReal_mul, Finset.mul_sum, neg_mul,
    Finset.sum_neg_distrib, mul_assoc, add_left_inj, neg_inj]
  /-
    case h.e'_6.h.h.e'_1
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
    v : ι → Real
    ⊢ Eq (HMul.hMul b (HPow.hPow (↑(Finset.univ.sum fun x => HPow.hPow (v x) 2).sq …
  -/
  norm_cast
  /-
    case h.e'_6.h.h.e'_1
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
    v : ι → Real
    ⊢ Eq (HMul.hMul b ↑(HPow.hPow (Finset.univ.sum fun x => HPow.hPow (v x) 2).sqr …
  -/
  rw [sq_sqrt]
    /-
      case h.e'_6.h.h.e'_1
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
      v : ι → Real
      ⊢ Eq (HMul.hMul b ↑(Finset.univ.sum fun x => HPow.hPow (v x) 2)) (Finset.univ. …
    -/
  · simp [Finset.mul_sum]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6.h.h.e'_1
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι)) M …
      v : ι → Real
      ⊢ LE.le 0 (Finset.univ.sum fun x => HPow.hPow (v x) 2)
    -/
  · exact Finset.sum_nonneg (fun i _hi ↦ by positivity)
    /-
      🎉 no goals
    -/


/-- In a real inner product space, the complex exponential of minus the square of the norm plus
a scalar product is integrable. Useful when discussing the Fourier transform of a Gaussian. -/
theorem integrable_cexp_neg_mul_sq_norm_add (hb : 0 < b.re) (c : ℂ) (w : V) :
    /-
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      c : Complex
      w : V
      ⊢ MeasureTheory.Measure V
    -/
    Integrable (fun (v : V) ↦ cexp (-b * ‖v‖^2 + c * ⟪w, v⟫)) := by
    /-
      🎉 no goals
    -/
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    c : Complex
    w : V
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (HMul.hMul (Neg.ne …
  -/
  let e := (stdOrthonormalBasis ℝ V).repr.symm
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    c : Complex
    w : V
    e : LinearIsometryEquiv (RingHom.id Real) (EuclideanSpace Real (Fin (Module.fi …
    ⊢ MeasureTheory.Integrable (fun v => Complex.exp (HAdd.hAdd (HMul.hMul (Neg.ne …
  -/
  rw [← e.measurePreserving.integrable_comp_emb e.toHomeomorph.measurableEmbedding]
  convert integrable_cexp_neg_mul_sq_norm_add_of_euclideanSpace
    hb c (e.symm w) with v
  simp only [neg_mul, Function.comp_apply, LinearIsometryEquiv.norm_map,
    LinearIsometryEquiv.symm_symm, conj_trivial, ofReal_sum,
    ofReal_mul, LinearIsometryEquiv.inner_map_eq_flip]


theorem integral_cexp_neg_sum_mul_add {ι : Type*} [Fintype ι] {b : ι → ℂ}
    (hb : ∀ i, 0 < (b i).re) (c : ι → ℂ) :
    ∫ v : ι → ℝ, cexp (- ∑ i, b i * (v i : ℂ) ^ 2 + ∑ i, c i * v i)
      = ∏ i, (π / b i) ^ (1 / 2 : ℂ) * cexp (c i ^ 2 / (4 * b i)) := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  simp_rw [← Finset.sum_neg_distrib, ← Finset.sum_add_distrib, Complex.exp_sum, ← neg_mul]
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Finset …
  -/
  rw [integral_fintype_prod_eq_prod (f := fun i (v : ℝ) ↦ cexp (-b i * v ^ 2 + c i * v))]
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    ⊢ Eq (Finset.univ.prod fun i => MeasureTheory.integral MeasureTheory.MeasureSp …
  -/
  congr with i
  /-
    case e_f.h
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    i : ι
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
  -/
  have : (-b i).re < 0 := by simpa using hb i
  /-
    case e_f.h
    ι : Type u_2
    inst✝ : Fintype ι
    b : ι → Complex
    hb : ∀ (i : ι), LT.lt 0 (b i).re
    c : ι → Complex
    i : ι
    this : LT.lt (Neg.neg (b i)).re 0
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  convert integral_cexp_quadratic this (c i) 0 using 1 <;> simp [div_neg]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem integral_cexp_neg_mul_sum_add {ι : Type*} [Fintype ι] (hb : 0 < b.re) (c : ι → ℂ) :
    ∫ v : ι → ℝ, cexp (- b * ∑ i, (v i : ℂ) ^ 2 + ∑ i, c i * v i)
      = (π / b) ^ (Fintype.card ι / 2 : ℂ) * cexp ((∑ i, c i ^ 2) / (4 * b)) := by
  simp_rw [neg_mul, Finset.mul_sum, integral_cexp_neg_sum_mul_add (fun _ ↦ hb) c, one_div,
    Finset.prod_mul_distrib, Finset.prod_const, ← cpow_nat_mul, ← Complex.exp_sum, Fintype.card,
    Finset.sum_div, div_eq_mul_inv]


theorem integral_cexp_neg_mul_sq_norm_add_of_euclideanSpace
    {ι : Type*} [Fintype ι] (hb : 0 < b.re) (c : ℂ) (w : EuclideanSpace ℝ ι) :
    ∫ v : EuclideanSpace ℝ ι, cexp (- b * ‖v‖^2 + c * ⟪w, v⟫) =
      (π / b) ^ (Fintype.card ι / 2 : ℂ) * cexp (c ^ 2 * ‖w‖^2 / (4 * b)) := by
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  have := (EuclideanSpace.volume_preserving_measurableEquiv ι).symm
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  rw [← this.integral_comp (MeasurableEquiv.measurableEmbedding _)]
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
  -/
  simp only [neg_mul, Function.comp_def]
  /-
    b : Complex
    ι : Type u_2
    inst✝ : Fintype ι
    hb : LT.lt 0 b.re
    c : Complex
    w : EuclideanSpace Real ι
    this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
  -/
  convert integral_cexp_neg_mul_sum_add hb (fun i ↦ c * w i) using 5 with _x y
  · simp only [EuclideanSpace.measurableEquiv, MeasurableEquiv.symm_mk, MeasurableEquiv.coe_mk,
      EuclideanSpace.norm_eq, WithLp.equiv_symm_pi_apply, Real.norm_eq_abs, sq_abs, neg_mul,
      neg_inj, mul_eq_mul_left_iff]
    /-
      case h.e'_2.h.e'_7.h.h.e'_1.h.e'_5
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      y : ι → Real
      ⊢ Or (Eq (HPow.hPow (↑(Finset.univ.sum fun x => HPow.hPow (y x) 2).sqrt) 2) (F …
    -/
    norm_cast
    /-
      case h.e'_2.h.e'_7.h.h.e'_1.h.e'_5
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      y : ι → Real
      ⊢ Or (Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (y x) 2).sqrt 2) (Fins …
    -/
    left
    /-
      case h.e'_2.h.e'_7.h.h.e'_1.h.e'_5.h
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      y : ι → Real
      ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (y x) 2).sqrt 2) (Finset.u …
    -/
    rw [sq_sqrt]
    /-
      case h.e'_2.h.e'_7.h.h.e'_1.h.e'_5.h
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      y : ι → Real
      ⊢ LE.le 0 (Finset.univ.sum fun x => HPow.hPow (y x) 2)
    -/
    exact Finset.sum_nonneg (fun i _hi ↦ by positivity)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_7.h.h.e'_1.h.e'_6
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      y : ι → Real
      ⊢ Eq (HMul.hMul c ↑(Inner.inner w ((EuclideanSpace.measurableEquiv ι).symm y)) …
    -/
  · simp [PiLp.inner_apply, EuclideanSpace.measurableEquiv, Finset.mul_sum, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6.h.e'_1.h.e'_5
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      ⊢ Eq (HMul.hMul (HPow.hPow c 2) (HPow.hPow (↑(Norm.norm w)) 2)) (Finset.univ.s …
    -/
  · simp only [EuclideanSpace.norm_eq, Real.norm_eq_abs, sq_abs, mul_pow, ← Finset.mul_sum]
    /-
      case h.e'_3.h.e'_6.h.e'_1.h.e'_5
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      ⊢ Eq (HMul.hMul (HPow.hPow c 2) (HPow.hPow (↑(Finset.univ.sum fun x => HPow.hP …
    -/
    congr
    /-
      case h.e'_3.h.e'_6.h.e'_1.h.e'_5.e_a
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      ⊢ Eq (HPow.hPow (↑(Finset.univ.sum fun x => HPow.hPow (w x) 2).sqrt) 2) (Finse …
    -/
    norm_cast
    /-
      case h.e'_3.h.e'_6.h.e'_1.h.e'_5.e_a
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (w x) 2).sqrt 2) (Finset.u …
    -/
    rw [sq_sqrt]
    /-
      case h.e'_3.h.e'_6.h.e'_1.h.e'_5.e_a
      b : Complex
      ι : Type u_2
      inst✝ : Fintype ι
      hb : LT.lt 0 b.re
      c : Complex
      w : EuclideanSpace Real ι
      this : MeasureTheory.MeasurePreserving (⇑(EuclideanSpace.measurableEquiv ι).sy …
      ⊢ LE.le 0 (Finset.univ.sum fun x => HPow.hPow (w x) 2)
    -/
    exact Finset.sum_nonneg (fun i _hi ↦ by positivity)
    /-
      🎉 no goals
    -/


theorem integral_cexp_neg_mul_sq_norm_add
    (hb : 0 < b.re) (c : ℂ) (w : V) :
    ∫ v : V, cexp (- b * ‖v‖^2 + c * ⟪w, v⟫) =
      (π / b) ^ (Module.finrank ℝ V / 2 : ℂ) * cexp (c ^ 2 * ‖w‖^2 / (4 * b)) := by
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    c : Complex
    w : V
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  let e := (stdOrthonormalBasis ℝ V).repr.symm
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    c : Complex
    w : V
    e : LinearIsometryEquiv (RingHom.id Real) (EuclideanSpace Real (Fin (Module.fi …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  rw [← e.measurePreserving.integral_comp e.toHomeomorph.measurableEmbedding]
  convert integral_cexp_neg_mul_sq_norm_add_of_euclideanSpace
                        /-
                          case h.e'_2.h.e'_7.h.h.e'_1.h.e'_5.h.e'_6.h.e'_5.h.e'_1
                          b : Complex
                          V : Type u_1
                          inst✝⁴ : NormedAddCommGroup V
                          inst✝³ : InnerProductSpace Real V
                          inst✝² : FiniteDimensional Real V
                          inst✝¹ : MeasurableSpace V
                          inst✝ : BorelSpace V
                          hb : LT.lt 0 b.re
                          c : Complex
                          w : V
                          e : LinearIsometryEquiv (RingHom.id Real) (EuclideanSpace Real (Fin (Module.fi …
                          x✝ : EuclideanSpace Real (Fin (Module.finrank Real V))
                          ⊢ Eq (Norm.norm (e x✝)) (Norm.norm x✝)
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
    hb c (e.symm w) <;> simp [LinearIsometryEquiv.inner_map_eq_flip]
                        /-
                          🎉 no goals
                        -/


theorem integral_cexp_neg_mul_sq_norm (hb : 0 < b.re) :
    ∫ v : V, cexp (- b * ‖v‖^2) = (π / b) ^ (Module.finrank ℝ V / 2 : ℂ) := by
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  simpa using integral_cexp_neg_mul_sq_norm_add hb 0 (0 : V)
  /-
    🎉 no goals
  -/


theorem integral_rexp_neg_mul_sq_norm {b : ℝ} (hb : 0 < b) :
    ∫ v : V, rexp (- b * ‖v‖^2) = (π / b) ^ (Module.finrank ℝ V / 2 : ℝ) := by
  /-
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    b : Real
    hb : LT.lt 0 b
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Real.e …
  -/
  rw [← ofReal_inj]
  /-
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    b : Real
    hb : LT.lt 0 b
    ⊢ Eq ↑(MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Real. …
  -/
  convert integral_cexp_neg_mul_sq_norm (show 0 < (b : ℂ).re from hb) (V := V)
    /-
      case h.e'_2
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      b : Real
      hb : LT.lt 0 b
      ⊢ Eq (↑(MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Real …
    -/
  · change ofRealLI (∫ (v : V), rexp (-b * ‖v‖ ^ 2)) = ∫ (v : V), cexp (-↑b * ↑‖v‖ ^ 2)
    /-
      case h.e'_2
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      b : Real
      hb : LT.lt 0 b
      ⊢ Eq (Complex.ofRealLI (MeasureTheory.integral MeasureTheory.MeasureSpace.volu …
    -/
    rw [← ofRealLI.integral_comp_comm]
    /-
      case h.e'_2
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      b : Real
      hb : LT.lt 0 b
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Comple …
    -/
    simp [ofRealLI]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      b : Real
      hb : LT.lt 0 b
      ⊢ Eq (↑(HPow.hPow (HDiv.hDiv Real.pi b) (HDiv.hDiv (↑(Module.finrank Real V))  …
    -/
  · rw [← ofReal_div, ofReal_cpow (by positivity)]
    /-
      case h.e'_3
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      b : Real
      hb : LT.lt 0 b
      ⊢ Eq (HPow.hPow ↑(HDiv.hDiv Real.pi b) ↑(HDiv.hDiv (↑(Module.finrank Real V))  …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem _root_.fourierIntegral_gaussian_innerProductSpace' (hb : 0 < b.re) (x w : V) :
    𝓕 (fun v ↦ cexp (- b * ‖v‖^2 + 2 * π * Complex.I * ⟪x, v⟫)) w =
      (π / b) ^ (Module.finrank ℝ V / 2 : ℂ) * cexp (-π ^ 2 * ‖x - w‖ ^ 2 / b) := by
  simp only [neg_mul, fourierIntegral_eq', ofReal_neg, ofReal_mul, ofReal_ofNat,
    smul_eq_mul, ← Complex.exp_add, real_inner_comm w]
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    x w : V
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => Comple …
  -/
  convert integral_cexp_neg_mul_sq_norm_add hb (2 * π * Complex.I) (x - w) using 3 with v
    /-
      case h.e'_2.h.e'_7.h
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      x w v : V
      ⊢ Eq (Complex.exp (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real …
    -/
  · congr 1
    /-
      case h.e'_2.h.e'_7.h.e_z
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      x w v : V
      ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) ↑(Inner. …
    -/
    simp [inner_sub_left]
    /-
      case h.e'_2.h.e'_7.h.e_z
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      x w v : V
      ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) ↑(Inner. …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6.h.e'_1
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      x w : V
      ⊢ Eq (HDiv.hDiv (Neg.neg (HMul.hMul (HPow.hPow (↑Real.pi) 2) (HPow.hPow (↑(Nor …
    -/
  · have : b ≠ 0 := by contrapose! hb; rw [hb, zero_re]
    /-
      case h.e'_3.h.e'_6.h.e'_1
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      x w : V
      this : Ne b 0
      ⊢ Eq (HDiv.hDiv (Neg.neg (HMul.hMul (HPow.hPow (↑Real.pi) 2) (HPow.hPow (↑(Nor …
    -/
    field_simp [mul_pow]
    /-
      case h.e'_3.h.e'_6.h.e'_1
      b : Complex
      V : Type u_1
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      hb : LT.lt 0 b.re
      x w : V
      this : Ne b 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) 2) (HPow.hPow (↑(Norm.norm (H …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem _root_.fourierIntegral_gaussian_innerProductSpace (hb : 0 < b.re) (w : V) :
    𝓕 (fun v ↦ cexp (- b * ‖v‖^2)) w =
      (π / b) ^ (Module.finrank ℝ V / 2 : ℂ) * cexp (-π ^ 2 * ‖w‖^2 / b) := by
  /-
    b : Complex
    V : Type u_1
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    hb : LT.lt 0 b.re
    w : V
    ⊢ Eq (Real.fourierIntegral (fun v => Complex.exp (HMul.hMul (Neg.neg b) (HPow. …
  -/
  simpa using fourierIntegral_gaussian_innerProductSpace' hb 0 w
  /-
    🎉 no goals
  -/


