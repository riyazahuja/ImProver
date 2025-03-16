/-- The vertical strip in the complex plane containing all `z ∈ ℂ` such that `z.re ∈ Ioo a b`. -/
def verticalStrip (a : ℝ) (b : ℝ) : Set ℂ := re ⁻¹' Ioo a b


/-- The vertical strip in the complex plane containing all `z ∈ ℂ` such that `z.re ∈ Icc a b`. -/
def verticalClosedStrip (a : ℝ) (b : ℝ) : Set ℂ := re ⁻¹' Icc a b


/-- The supremum of the norm of `f` on imaginary lines. (Fixed real part)
This is also known as the function `M` -/
noncomputable def sSupNormIm {E : Type*} [NormedAddCommGroup E]
    (f : ℂ → E) (x : ℝ) : ℝ :=
  sSup ((norm ∘ f) '' (re ⁻¹' {x}))


/--
The inverse of the interpolation of `sSupNormIm` on the two boundaries.
In other words, this is the inverse of the right side of the target inequality:
`|f(z)| ≤ |M(0) ^ (1-z)| * |M(1) ^ z|`.

Shifting this by a positive epsilon allows us to prove the case when either of the boundaries
is zero.-/
noncomputable def invInterpStrip (ε : ℝ) : ℂ :=
  (ε + sSupNormIm f 0) ^ (z - 1) * (ε + sSupNormIm f 1) ^ (-z)


/-- A function useful for the proofs steps. We will aim to show that it is bounded by 1. -/
noncomputable def F [NormedSpace ℂ E] (ε : ℝ) := fun z ↦ invInterpStrip f z ε • f z


/-- `sSup` of `norm` is nonneg applied to the image of `f` on the vertical line `re z = x` -/
lemma sSupNormIm_nonneg (x : ℝ) : 0 ≤ sSupNormIm f x := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    x : Real
    ⊢ LE.le 0 (Complex.HadamardThreeLines.sSupNormIm f x)
  -/
  apply Real.sSup_nonneg
  /-
    case hs
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    x : Real
    ⊢ ∀ (x_1 : Real), Membership.mem (Set.image (Function.comp Norm.norm f) (Set.p …
  -/
  rintro y ⟨z1, _, hz2⟩
  /-
    case hs.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    x y : Real
    z1 : Complex
    left✝ : Membership.mem (Set.preimage Complex.re (Singleton.singleton x)) z1
    hz2 : Eq (Function.comp Norm.norm f z1) y
    ⊢ LE.le 0 y
  -/
  simp only [← hz2, comp, norm_nonneg]
  /-
    🎉 no goals
  -/


/-- `sSup` of `norm` translated by `ε > 0` is positive applied to the image of `f` on the
vertical line `re z = x` -/
lemma sSupNormIm_eps_pos {ε : ℝ} (hε : ε > 0) (x : ℝ) : 0 < ε + sSupNormIm f x := by
   /-
     E : Type u_1
     inst✝ : NormedAddCommGroup E
     f : Complex → E
     ε : Real
     hε : GT.gt ε 0
     x : Real
     ⊢ LT.lt 0 (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f x))
   -/
   linarith [sSupNormIm_nonneg f x]
   /-
     🎉 no goals
   -/


/-- Useful rewrite for the absolute value of `invInterpStrip`-/
lemma abs_invInterpStrip {ε : ℝ} (hε : ε > 0) :
    abs (invInterpStrip f z ε) =
    (ε + sSupNormIm f 0) ^ (z.re - 1) * (ε + sSupNormIm f 1) ^ (-z.re) := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    ε : Real
    hε : GT.gt ε 0
    ⊢ Eq (Complex.abs (Complex.HadamardThreeLines.invInterpStrip f z ε)) (HMul.hMu …
  -/
  simp only [invInterpStrip, map_mul]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    ε : Real
    hε : GT.gt ε 0
    ⊢ Eq (HMul.hMul (Complex.abs (HPow.hPow (HAdd.hAdd ↑ε ↑(Complex.HadamardThreeL …
  -/
  repeat rw [← ofReal_add]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    ε : Real
    hε : GT.gt ε 0
    ⊢ Eq (HMul.hMul (Complex.abs (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeL …
  -/
  repeat rw [abs_cpow_eq_rpow_re_of_pos (sSupNormIm_eps_pos f hε _) _]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    ε : Real
    hε : GT.gt ε 0
    ⊢ Eq (HMul.hMul (HPow.hPow (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm …
  -/
  simp only [sub_re, one_re, neg_re]
  /-
    🎉 no goals
  -/


/-- The function `invInterpStrip` is `diffContOnCl`. -/
lemma diffContOnCl_invInterpStrip {ε : ℝ} (hε : ε > 0) :
    DiffContOnCl ℂ (fun z ↦ invInterpStrip f z ε) (verticalStrip 0 1) := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    ⊢ DiffContOnCl Complex (fun z => Complex.HadamardThreeLines.invInterpStrip f z …
  -/
  apply Differentiable.diffContOnCl
  /-
    case h
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    ⊢ Differentiable Complex fun z => Complex.HadamardThreeLines.invInterpStrip f  …
  -/
  apply Differentiable.mul
    /-
      case h.ha
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Differentiable Complex fun y => HPow.hPow (HAdd.hAdd ↑ε ↑(Complex.HadamardTh …
    -/
  · apply Differentiable.const_cpow (Differentiable.sub_const (differentiable_id') 1) _
    /-
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Or (Ne (HAdd.hAdd ↑ε ↑(Complex.HadamardThreeLines.sSupNormIm f 0)) 0) (∀ (x  …
    -/
    left
    /-
      case h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Ne (HAdd.hAdd ↑ε ↑(Complex.HadamardThreeLines.sSupNormIm f 0)) 0
    -/
    rw [← ofReal_add, ofReal_ne_zero]
    /-
      case h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Ne (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0)) 0
    -/
    simp only [ne_eq, ne_of_gt (sSupNormIm_eps_pos f hε 0), not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case h.hb
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Differentiable Complex fun y => HPow.hPow (HAdd.hAdd ↑ε ↑(Complex.HadamardTh …
    -/
  · apply Differentiable.const_cpow (Differentiable.neg differentiable_id')
    /-
      case h.hb
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Or (Ne (HAdd.hAdd ↑ε ↑(Complex.HadamardThreeLines.sSupNormIm f 1)) 0) (∀ (x  …
    -/
    apply Or.inl
    /-
      case h.hb.h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Ne (HAdd.hAdd ↑ε ↑(Complex.HadamardThreeLines.sSupNormIm f 1)) 0
    -/
    rw [← ofReal_add, ofReal_ne_zero]
    /-
      case h.hb.h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Ne (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 1)) 0
    -/
    exact (ne_of_gt (sSupNormIm_eps_pos f hε 1))
    /-
      🎉 no goals
    -/


/-- If `f` is bounded on the unit vertical strip, then `f` is bounded by `sSupNormIm` there. -/
lemma norm_le_sSupNormIm (f : ℂ → E) (z : ℂ) (hD : z ∈ verticalClosedStrip 0 1)
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) :
    ‖f z‖ ≤ sSupNormIm f (z.re) := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    hD : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    ⊢ LE.le (Norm.norm (f z)) (Complex.HadamardThreeLines.sSupNormIm f z.re)
  -/
  refine le_csSup ?_ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hD : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ⊢ BddAbove (Set.image (Function.comp Norm.norm f) (Set.preimage Complex.re (Si …
    -/
  · apply BddAbove.mono (image_subset (norm ∘ f) _) hB
    /-
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hD : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ⊢ HasSubset.Subset (Set.preimage Complex.re (Singleton.singleton z.re)) (Compl …
    -/
    exact preimage_mono (singleton_subset_iff.mpr hD)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hD : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ⊢ Membership.mem (Set.image (Function.comp Norm.norm f) (Set.preimage Complex. …
    -/
  · apply mem_image_of_mem (norm ∘ f)
    /-
      case refine_2
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hD : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ⊢ Membership.mem (Set.preimage Complex.re (Singleton.singleton z.re)) z
    -/
    simp only [mem_preimage, mem_singleton]
    /-
      🎉 no goals
    -/


/-- Alternative version of `norm_le_sSupNormIm` with a strict inequality and a positive `ε`. -/
lemma norm_lt_sSupNormIm_eps (f : ℂ → E) (ε : ℝ) (hε : ε > 0) (z : ℂ)
    (hD : z ∈ verticalClosedStrip 0 1) (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) :
    ‖f z‖ < ε + sSupNormIm f (z.re) :=
  lt_add_of_pos_of_le hε (norm_le_sSupNormIm f z hD hB)


/-- When the function `f` is bounded above on a vertical strip, then so is `F`. -/
lemma F_BddAbove (f : ℂ → E) (ε : ℝ) (hε : ε > 0)
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) :
    BddAbove ((norm ∘ (F f ε)) '' (verticalClosedStrip 0 1)) := by
 -- Rewriting goal
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    ⊢ BddAbove (Set.image (Function.comp Norm.norm (Complex.HadamardThreeLines.F f …
  -/
  simp only [F, image_congr, comp_apply, map_mul, invInterpStrip]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    ⊢ BddAbove (Set.image (fun a => Norm.norm (HSMul.hSMul (HMul.hMul (HPow.hPow ( …
  -/
  rw [bddAbove_def] at *
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    hB : Exists fun x => ∀ (y : Real), Membership.mem (Set.image (Function.comp No …
    ⊢ Exists fun x => ∀ (y : Real), Membership.mem (Set.image (fun a => Norm.norm  …
  -/
  rcases hB with ⟨B, hB⟩
  -- Using bound
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    B : Real
    hB : ∀ (y : Real), Membership.mem (Set.image (Function.comp Norm.norm f) (Comp …
    ⊢ Exists fun x => ∀ (y : Real), Membership.mem (Set.image (fun a => Norm.norm  …
  -/
  use ((max 1 ((ε + sSupNormIm f 0) ^ (-(1 : ℝ)))) * max 1 ((ε + sSupNormIm f 1) ^ (-(1 : ℝ)))) * B
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    B : Real
    hB : ∀ (y : Real), Membership.mem (Set.image (Function.comp Norm.norm f) (Comp …
    ⊢ ∀ (y : Real), Membership.mem (Set.image (fun a => Norm.norm (HSMul.hSMul (HM …
  -/
  simp only [mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    B : Real
    hB : ∀ (y : Real), Membership.mem (Set.image (Function.comp Norm.norm f) (Comp …
    ⊢ ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClosedSt …
  -/
  intros z hset
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    B : Real
    hB : ∀ (y : Real), Membership.mem (Set.image (Function.comp Norm.norm f) (Comp …
    z : Complex
    hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    ⊢ LE.le (Norm.norm (HSMul.hSMul (HMul.hMul (HPow.hPow (HAdd.hAdd ↑ε ↑(Complex. …
  -/
  specialize hB (‖f z‖) (by simpa [image_congr, mem_image, comp_apply] using ⟨z, hset, rfl⟩)
  -- Proof that the bound is correct
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    B : Real
    z : Complex
    hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    hB : LE.le (Norm.norm (f z)) B
    ⊢ LE.le (Norm.norm (HSMul.hSMul (HMul.hMul (HPow.hPow (HAdd.hAdd ↑ε ↑(Complex. …
  -/
  simp only [norm_smul, norm_mul, ← ofReal_add]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    B : Real
    z : Complex
    hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    hB : LE.le (Norm.norm (f z)) B
    ⊢ LE.le (HMul.hMul (HMul.hMul (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.Ha …
  -/
  gcongr
    -- Bounding individual terms
    /-
      case h.h₁.h₁
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      B : Real
      z : Complex
      hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hB : LE.le (Norm.norm (f z)) B
      ⊢ LE.le (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeLines.sSupN …
    -/
  · by_cases hM0_one : 1 ≤ ε + sSupNormIm f 0
    -- `1 ≤ (sSupNormIm f 0)`
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        ε : Real
        hε : GT.gt ε 0
        B : Real
        z : Complex
        hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hB : LE.le (Norm.norm (f z)) B
        hM0_one : LE.le 1 (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0))
        ⊢ LE.le (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeLines.sSupN …
      -/
    · apply le_trans _ (le_max_left _ _)
      simp only [norm_eq_abs, abs_cpow_eq_rpow_re_of_pos (sSupNormIm_eps_pos f hε 0), sub_re,
        one_re, Real.rpow_le_one_of_one_le_of_nonpos hM0_one (sub_nonpos.mpr hset.2)]
    -- `0 < (sSupNormIm f 0) < 1`
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        ε : Real
        hε : GT.gt ε 0
        B : Real
        z : Complex
        hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hB : LE.le (Norm.norm (f z)) B
        hM0_one : Not (LE.le 1 (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0 …
        ⊢ LE.le (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeLines.sSupN …
      -/
    · rw [not_le] at hM0_one; apply le_trans _ (le_max_right _ _)
      simp only [norm_eq_abs, abs_cpow_eq_rpow_re_of_pos (sSupNormIm_eps_pos f hε 0), sub_re,
        one_re]
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        ε : Real
        hε : GT.gt ε 0
        B : Real
        z : Complex
        hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hB : LE.le (Norm.norm (f z)) B
        hM0_one : LT.lt (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0)) 1
        ⊢ LE.le (HPow.hPow (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0)) ( …
      -/
      apply Real.rpow_le_rpow_of_exponent_ge (sSupNormIm_eps_pos f hε 0) (le_of_lt hM0_one) _
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        ε : Real
        hε : GT.gt ε 0
        B : Real
        z : Complex
        hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hB : LE.le (Norm.norm (f z)) B
        hM0_one : LT.lt (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0)) 1
        ⊢ LE.le (-1) (HSub.hSub z.re 1)
      -/
      simp only [neg_le_sub_iff_le_add, le_add_iff_nonneg_left, hset.1]
      /-
        🎉 no goals
      -/
    /-
      case h.h₁.h₂
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      B : Real
      z : Complex
      hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hB : LE.le (Norm.norm (f z)) B
      ⊢ LE.le (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeLines.sSupN …
    -/
  · by_cases hM1_one : 1 ≤ ε + sSupNormIm f 1
    -- `1 ≤ sSupNormIm f 1`
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        ε : Real
        hε : GT.gt ε 0
        B : Real
        z : Complex
        hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hB : LE.le (Norm.norm (f z)) B
        hM1_one : LE.le 1 (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 1))
        ⊢ LE.le (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeLines.sSupN …
      -/
    · apply le_trans _ (le_max_left _ _)
      simp only [norm_eq_abs, abs_cpow_eq_rpow_re_of_pos (sSupNormIm_eps_pos f hε 1), sub_re,
        one_re, neg_re, Real.rpow_le_one_of_one_le_of_nonpos
        hM1_one (Right.neg_nonpos_iff.mpr hset.1)]
    -- `0 < sSupNormIm f 1 < 1`
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        ε : Real
        hε : GT.gt ε 0
        B : Real
        z : Complex
        hset : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hB : LE.le (Norm.norm (f z)) B
        hM1_one : Not (LE.le 1 (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 1 …
        ⊢ LE.le (Norm.norm (HPow.hPow (↑(HAdd.hAdd ε (Complex.HadamardThreeLines.sSupN …
      -/
    · rw [not_le] at hM1_one; apply le_trans _ (le_max_right _ _)
      simp only [norm_eq_abs, abs_cpow_eq_rpow_re_of_pos (sSupNormIm_eps_pos f hε 1), sub_re,
        one_re, neg_re, Real.rpow_le_rpow_of_exponent_ge (sSupNormIm_eps_pos f hε 1)
        (le_of_lt hM1_one) (neg_le_neg_iff.mpr hset.2)]


/-- Proof that `F` is bounded by one one the edges. -/
lemma F_edge_le_one (f : ℂ → E) (ε : ℝ) (hε : ε > 0) (z : ℂ)
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) (hz : z ∈ re ⁻¹' {0, 1}) :
    ‖F f ε z‖ ≤ 1 := by
  simp only [F, norm_smul, norm_eq_abs, map_mul, abs_cpow_eq_rpow_re_of_pos,
    abs_invInterpStrip f z hε, sSupNormIm_eps_pos f hε 1,
    sub_re, one_re, neg_re]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : GT.gt ε 0
    z : Complex
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Set.preimage Complex.re (Insert.insert 0 (Singleton.singl …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (HAdd.hAdd ε (Complex.HadamardThreeLi …
  -/
  rcases hz with hz0 | hz1
  -- `z.re = 0`
  · simp only [hz0, zero_sub, Real.rpow_neg_one, neg_zero, Real.rpow_zero, mul_one,
      inv_mul_le_iff₀ (sSupNormIm_eps_pos f hε 0)]
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz0 : Eq z.re 0
      ⊢ LE.le (Norm.norm (f z)) (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm  …
    -/
    rw [← hz0]
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz0 : Eq z.re 0
      ⊢ LE.le (Norm.norm (f z)) (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm  …
    -/
    apply le_of_lt (norm_lt_sSupNormIm_eps f ε hε _ _ hB)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz0 : Eq z.re 0
      ⊢ Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    -/
    simp only [verticalClosedStrip, mem_preimage, zero_le_one, left_mem_Icc, hz0]
    /-
      🎉 no goals
    -/
  -- `z.re = 1`
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz1 : Membership.mem (Singleton.singleton 1) z.re
      ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (HAdd.hAdd ε (Complex.HadamardThreeLi …
    -/
  · rw [mem_singleton_iff] at hz1
    simp only [hz1, one_mul, Real.rpow_zero, sub_self, Real.rpow_neg_one,
      inv_mul_le_iff₀ (sSupNormIm_eps_pos f hε 1), mul_one]
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz1 : Eq z.re 1
      ⊢ LE.le (Norm.norm (f z)) (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm  …
    -/
    rw [← hz1]
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz1 : Eq z.re 1
      ⊢ LE.le (Norm.norm (f z)) (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm  …
    -/
    apply le_of_lt (norm_lt_sSupNormIm_eps f ε hε _ _ hB)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz1 : Eq z.re 1
      ⊢ Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    -/
    simp only [verticalClosedStrip, mem_preimage, zero_le_one, hz1, right_mem_Icc]
    /-
      🎉 no goals
    -/


theorem norm_mul_invInterpStrip_le_one_of_mem_verticalClosedStrip (f : ℂ → E) (ε : ℝ) (hε : 0 < ε)
    (z : ℂ) (hd : DiffContOnCl ℂ f (verticalStrip 0 1))
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) (hz : z ∈ verticalClosedStrip 0 1) :
    ‖F f ε z‖ ≤ 1 := by
  apply PhragmenLindelof.vertical_strip
    (DiffContOnCl.smul (diffContOnCl_invInterpStrip f hε) hd) _
    (fun x hx ↦ F_edge_le_one f ε hε x hB (Or.inl hx))
    (fun x hx ↦ F_edge_le_one f ε hε x hB (Or.inr hx)) hz.1 hz.2
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    ⊢ Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub 1 0))) (Exists fu …
  -/
  use 0
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    ⊢ And (LT.lt 0 (HDiv.hDiv Real.pi (HSub.hSub 1 0))) (Exists fun B => Asymptoti …
  -/
  rw [sub_zero, div_one]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    ⊢ And (LT.lt 0 Real.pi) (Exists fun B => Asymptotics.IsBigO (Min.min (Filter.c …
  -/
  refine ⟨ Real.pi_pos, ?_⟩
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    ⊢ Exists fun B => Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp _ro …
  -/
  obtain ⟨BF, hBF⟩ := F_BddAbove f ε hε hB
  simp only [comp_apply, mem_upperBounds, mem_image, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂] at hBF
  /-
    case h.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    ⊢ Exists fun B => Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp _ro …
  -/
  use BF
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    ⊢ Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp _root_.abs Complex. …
  -/
  rw [Asymptotics.isBigO_iff]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    ⊢ Exists fun c => Filter.Eventually (fun x => LE.le (Norm.norm (HSMul.hSMul (C …
  -/
  use 1
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSMul.hSMul (Complex.HadamardT …
  -/
  rw [eventually_inf_principal]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.preimage Complex.re (Set.Ioo …
  -/
  apply Eventually.of_forall
  /-
    case h.hp
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    ⊢ ∀ (x : Complex), Membership.mem (Set.preimage Complex.re (Set.Ioo 0 1)) x →  …
  -/
  intro x hx
  /-
    case h.hp
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    ε : Real
    hε : LT.lt 0 ε
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    BF : Real
    hBF : ∀ (a : Complex), Membership.mem (Complex.HadamardThreeLines.verticalClos …
    x : Complex
    hx : Membership.mem (Set.preimage Complex.re (Set.Ioo 0 1)) x
    ⊢ LE.le (Norm.norm (HSMul.hSMul (Complex.HadamardThreeLines.invInterpStrip f x …
  -/
  norm_num
  exact (hBF x ((preimage_mono Ioo_subset_Icc_self) hx)).trans
    ((le_of_lt (lt_add_one BF)).trans (Real.add_one_le_exp BF))


/--
The interpolation of `sSupNormIm` on the two boundaries.
In other words, this is the right side of the target inequality:
`|f(z)| ≤ |M(0) ^ (1-z)| * |M(1) ^ z|`.

Note that if `(sSupNormIm f 0) = 0 ∨ (sSupNormIm f 1) = 0` then the power is not continuous
since `0 ^ 0 = 1`. Hence the use of `ite`. -/
noncomputable def interpStrip (z : ℂ) : ℂ :=
  if (sSupNormIm f 0) = 0 ∨ (sSupNormIm f 1) = 0
    then 0
    else (sSupNormIm f 0) ^ (1-z) * (sSupNormIm f 1) ^ z


/-- Rewrite for `InterpStrip` when `0 < sSupNormIm f 0` and `0 < sSupNormIm f 1`. -/
lemma interpStrip_eq_of_pos (z : ℂ) (h0 : 0 < sSupNormIm f 0) (h1 : 0 < sSupNormIm f 1) :
    interpStrip f z = (sSupNormIm f 0) ^ (1 - z) * (sSupNormIm f 1) ^ z := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    h0 : LT.lt 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
    h1 : LT.lt 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
    ⊢ Eq (Complex.HadamardThreeLines.interpStrip f z) (HMul.hMul (HPow.hPow (↑(Com …
  -/
  simp only [ne_of_gt h0, ne_of_gt h1, interpStrip, if_false, or_false]
  /-
    🎉 no goals
  -/


/-- Rewrite for `InterpStrip` when `0 = sSupNormIm f 0` or `0 = sSupNormIm f 1`. -/
lemma interpStrip_eq_of_zero (z : ℂ) (h : sSupNormIm f 0 = 0 ∨ sSupNormIm f 1 = 0) :
    interpStrip f z = 0 :=
  if_pos h


/-- Rewrite for `InterpStrip` on the open vertical strip. -/
lemma interpStrip_eq_of_mem_verticalStrip (z : ℂ) (hz : z ∈ verticalStrip 0 1) :
    interpStrip f z = (sSupNormIm f 0) ^ (1 - z) * (sSupNormIm f 1) ^ z := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    z : Complex
    hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
    ⊢ Eq (Complex.HadamardThreeLines.interpStrip f z) (HMul.hMul (HPow.hPow (↑(Com …
  -/
  by_cases h : sSupNormIm f 0 = 0 ∨ sSupNormIm f 1 = 0
    /-
      case pos
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      h : Or (Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Eq (Complex.Hadamar …
      ⊢ Eq (Complex.HadamardThreeLines.interpStrip f z) (HMul.hMul (HPow.hPow (↑(Com …
    -/
  · rw [interpStrip_eq_of_zero _ z h]
    /-
      case pos
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      h : Or (Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Eq (Complex.Hadamar …
      ⊢ Eq 0 (HMul.hMul (HPow.hPow (↑(Complex.HadamardThreeLines.sSupNormIm f 0)) (H …
    -/
    rcases h with h0 | h1
      /-
        case pos.inl
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        z : Complex
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        h0 : Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0
        ⊢ Eq 0 (HMul.hMul (HPow.hPow (↑(Complex.HadamardThreeLines.sSupNormIm f 0)) (H …
      -/
    · simp only [h0, ofReal_zero, zero_eq_mul, cpow_eq_zero_iff, ne_eq, true_and, ofReal_eq_zero]
      /-
        case pos.inl
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        z : Complex
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        h0 : Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0
        ⊢ Or (Not (Eq (HSub.hSub 1 z) 0)) (And (Eq (Complex.HadamardThreeLines.sSupNor …
      -/
      left
      /-
        case pos.inl.h
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        z : Complex
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        h0 : Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0
        ⊢ Not (Eq (HSub.hSub 1 z) 0)
      -/
      rw [sub_eq_zero, eq_comm]
      simp only [ne_eq, Complex.ext_iff, one_re, ne_of_lt hz.2, or_iff_left, false_and,
        not_false_eq_true]
      /-
        case pos.inr
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        z : Complex
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        h1 : Eq (Complex.HadamardThreeLines.sSupNormIm f 1) 0
        ⊢ Eq 0 (HMul.hMul (HPow.hPow (↑(Complex.HadamardThreeLines.sSupNormIm f 0)) (H …
      -/
    · simp only [h1, ofReal_zero, zero_eq_mul, cpow_eq_zero_iff, ofReal_eq_zero, ne_eq, true_and]
      /-
        case pos.inr
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        z : Complex
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        h1 : Eq (Complex.HadamardThreeLines.sSupNormIm f 1) 0
        ⊢ Or (And (Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Not (Eq (HSub.hS …
      -/
      right
      /-
        case pos.inr.h
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        z : Complex
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        h1 : Eq (Complex.HadamardThreeLines.sSupNormIm f 1) 0
        ⊢ Not (Eq z 0)
      -/
      rw [eq_comm]
      simp only [ne_eq, Complex.ext_iff, zero_re, ne_of_lt hz.1, or_iff_left, false_and,
        not_false_eq_true]
    /-
      case neg
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      h : Not (Or (Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Eq (Complex.Ha …
      ⊢ Eq (Complex.HadamardThreeLines.interpStrip f z) (HMul.hMul (HPow.hPow (↑(Com …
    -/
  · push_neg at h
    replace h : (0 < sSupNormIm f 0) ∧ (0 < sSupNormIm f 1) :=
      ⟨(lt_of_le_of_ne (sSupNormIm_nonneg f 0) (ne_comm.mp h.1)),
        (lt_of_le_of_ne (sSupNormIm_nonneg f 1) (ne_comm.mp h.2))⟩
    /-
      case neg
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      z : Complex
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      h : And (LT.lt 0 (Complex.HadamardThreeLines.sSupNormIm f 0)) (LT.lt 0 (Comple …
      ⊢ Eq (Complex.HadamardThreeLines.interpStrip f z) (HMul.hMul (HPow.hPow (↑(Com …
    -/
    exact interpStrip_eq_of_pos f z h.1 h.2
    /-
      🎉 no goals
    -/


lemma diffContOnCl_interpStrip :
    DiffContOnCl ℂ (interpStrip f) (verticalStrip 0 1) := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Complex → E
    ⊢ DiffContOnCl Complex (Complex.HadamardThreeLines.interpStrip f) (Complex.Had …
  -/
  by_cases h : sSupNormIm f 0 = 0 ∨ sSupNormIm f 1 = 0
  -- Case everywhere 0
    /-
      case pos
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h : Or (Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Eq (Complex.Hadamar …
      ⊢ DiffContOnCl Complex (Complex.HadamardThreeLines.interpStrip f) (Complex.Had …
    -/
  · eta_expand; simp_rw [interpStrip_eq_of_zero f _ h]; exact diffContOnCl_const
                                                        /-
                                                          🎉 no goals
                                                        -/
  -- Case nowhere 0
    /-
      case neg
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h : Not (Or (Eq (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Eq (Complex.Ha …
      ⊢ DiffContOnCl Complex (Complex.HadamardThreeLines.interpStrip f) (Complex.Had …
    -/
  · push_neg at h
    /-
      case neg
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h : And (Ne (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (Ne (Complex.Hadama …
      ⊢ DiffContOnCl Complex (Complex.HadamardThreeLines.interpStrip f) (Complex.Had …
    -/
    rcases h with ⟨h0, h1⟩
    /-
      case neg.intro
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h0 : Ne (Complex.HadamardThreeLines.sSupNormIm f 0) 0
      h1 : Ne (Complex.HadamardThreeLines.sSupNormIm f 1) 0
      ⊢ DiffContOnCl Complex (Complex.HadamardThreeLines.interpStrip f) (Complex.Had …
    -/
    rw [ne_comm] at h0 h1
    /-
      case neg.intro
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
      h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
      ⊢ DiffContOnCl Complex (Complex.HadamardThreeLines.interpStrip f) (Complex.Had …
    -/
    apply Differentiable.diffContOnCl
    /-
      case neg.intro.h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
      h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
      ⊢ Differentiable Complex (Complex.HadamardThreeLines.interpStrip f)
    -/
    intro z
    /-
      case neg.intro.h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
      h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
      z : Complex
      ⊢ DifferentiableAt Complex (Complex.HadamardThreeLines.interpStrip f) z
    -/
    eta_expand
    simp_rw [interpStrip_eq_of_pos f _ (lt_of_le_of_ne (sSupNormIm_nonneg f 0) h0)
      (lt_of_le_of_ne (sSupNormIm_nonneg f 1) h1)]
    /-
      case neg.intro.h
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Complex → E
      h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
      h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
      z : Complex
      ⊢ DifferentiableAt Complex (fun z => HMul.hMul (HPow.hPow (↑(Complex.HadamardT …
    -/
    refine DifferentiableAt.mul ?_ ?_
      /-
        case neg.intro.h.refine_1
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
        h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
        z : Complex
        ⊢ DifferentiableAt Complex (fun z => HPow.hPow (↑(Complex.HadamardThreeLines.s …
      -/
    · apply DifferentiableAt.const_cpow (DifferentiableAt.const_sub (differentiableAt_id') 1) _
      /-
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
        h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
        z : Complex
        ⊢ Or (Ne (↑(Complex.HadamardThreeLines.sSupNormIm f 0)) 0) (Ne (HSub.hSub 1 z) …
      -/
      left; simp only [Ne, ofReal_eq_zero]; rwa [eq_comm]
                                            /-
                                              🎉 no goals
                                            -/
      /-
        case neg.intro.h.refine_2
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        f : Complex → E
        h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
        h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
        z : Complex
        ⊢ DifferentiableAt Complex (HPow.hPow ↑(Complex.HadamardThreeLines.sSupNormIm  …
      -/
    · refine DifferentiableAt.const_cpow ?_ ?_
        /-
          case neg.intro.h.refine_2.refine_1
          E : Type u_1
          inst✝ : NormedAddCommGroup E
          f : Complex → E
          h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
          h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
          z : Complex
          ⊢ DifferentiableAt Complex (fun x => x) z
        -/
      · apply differentiableAt_id'
        /-
          🎉 no goals
        -/
        /-
          case neg.intro.h.refine_2.refine_2
          E : Type u_1
          inst✝ : NormedAddCommGroup E
          f : Complex → E
          h0 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 0)
          h1 : Ne 0 (Complex.HadamardThreeLines.sSupNormIm f 1)
          z : Complex
          ⊢ Or (Ne (↑(Complex.HadamardThreeLines.sSupNormIm f 1)) 0) (Ne z 0)
        -/
      · left; simp only [Ne, ofReal_eq_zero]; rwa [eq_comm]
                                              /-
                                                🎉 no goals
                                              -/


lemma norm_le_interpStrip_of_mem_verticalClosedStrip_eps (ε : ℝ) (hε : ε > 0) (z : ℂ)
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1)))
    (hd : DiffContOnCl ℂ f (verticalStrip 0 1)) (hz : z ∈ verticalClosedStrip 0 1) :
    ‖f z‖ ≤  ‖((ε + sSupNormIm f 0) ^ (1-z) * (ε + sSupNormIm f 1) ^ z : ℂ)‖ := by
  simp only [F, abs_invInterpStrip _ _ hε, norm_smul, norm_mul, norm_eq_abs,
    ← ofReal_add, abs_cpow_eq_rpow_re_of_pos (sSupNormIm_eps_pos f hε _) _, sub_re, one_re]
  rw [← mul_inv_le_iff₀', ← one_mul (((ε + sSupNormIm f 1) ^ z.re)), ← mul_inv_le_iff₀,
    ← Real.rpow_neg_one, ← Real.rpow_neg_one]
  · simp only [← Real.rpow_mul (le_of_lt (sSupNormIm_eps_pos f hε _)),
    mul_neg, mul_one, neg_sub, mul_assoc]
    simpa [F, abs_invInterpStrip _ _ hε, norm_smul, mul_comm] using
      norm_mul_invInterpStrip_le_one_of_mem_verticalClosedStrip f ε hε z hd hB hz
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      inst✝ : NormedSpace Complex E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      ⊢ LT.lt 0 (HPow.hPow (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 1)) …
    -/
  · simp only [Real.rpow_pos_of_pos (sSupNormIm_eps_pos f hε _) z.re]
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      inst✝ : NormedSpace Complex E
      ε : Real
      hε : GT.gt ε 0
      z : Complex
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      ⊢ LT.lt 0 (HPow.hPow (HAdd.hAdd ε (Complex.HadamardThreeLines.sSupNormIm f 0)) …
    -/
  · simp only [Real.rpow_pos_of_pos (sSupNormIm_eps_pos f hε _) (1-z.re)]
    /-
      🎉 no goals
    -/


lemma eventuallyle (z : ℂ) (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1)))
    (hd : DiffContOnCl ℂ f (verticalStrip 0 1)) (hz : z ∈ verticalStrip 0 1) :
    (fun _ : ℝ ↦ ‖f z‖) ≤ᶠ[𝓝[>] 0]
    (fun ε ↦ ‖((ε + sSupNormIm f 0) ^ (1 - z) * (ε + sSupNormIm f 1) ^ z : ℂ)‖) := by
  filter_upwards [self_mem_nhdsWithin] with ε (hε : 0 < ε) using
    norm_le_interpStrip_of_mem_verticalClosedStrip_eps f ε hε z hB hd
      (mem_of_mem_of_subset hz (preimage_mono Ioo_subset_Icc_self))


lemma norm_le_interpStrip_of_mem_verticalStrip_zero (z : ℂ)
    (hd : DiffContOnCl ℂ f (verticalStrip 0 1))
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) (hz : z ∈ verticalStrip 0 1) :
    ‖f z‖ ≤ ‖interpStrip f z‖ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    f : Complex → E
    inst✝ : NormedSpace Complex E
    z : Complex
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
    ⊢ LE.le (Norm.norm (f z)) (Norm.norm (Complex.HadamardThreeLines.interpStrip f …
  -/
  apply tendsto_le_of_eventuallyLE _ _ (eventuallyle f z hB hd hz)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      inst✝ : NormedSpace Complex E
      z : Complex
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      ⊢ Filter.Tendsto (fun x => Norm.norm (f z)) (nhdsWithin 0 (Set.Ioi 0)) (nhds ( …
    -/
  · apply tendsto_inf_left
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      inst✝ : NormedSpace Complex E
      z : Complex
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      ⊢ Filter.Tendsto (fun x => Norm.norm (f z)) (nhds 0) (nhds (Norm.norm (f z)))
    -/
    simp only [tendsto_const_nhds_iff]
    /-
      🎉 no goals
    -/
  -- Proof that we can let epsilon tend to zero.
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      inst✝ : NormedSpace Complex E
      z : Complex
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      ⊢ Filter.Tendsto (fun ε => Norm.norm (HMul.hMul (HPow.hPow (HAdd.hAdd ↑ε ↑(Com …
    -/
  · rw [interpStrip_eq_of_mem_verticalStrip _ _ hz]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      f : Complex → E
      inst✝ : NormedSpace Complex E
      z : Complex
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
      ⊢ Filter.Tendsto (fun ε => Norm.norm (HMul.hMul (HPow.hPow (HAdd.hAdd ↑ε ↑(Com …
    -/
    convert ContinuousWithinAt.tendsto _ using 2
      /-
        case h.e'_5.h.e'_3
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        f : Complex → E
        inst✝ : NormedSpace Complex E
        z : Complex
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        ⊢ Eq (Norm.norm (HMul.hMul (HPow.hPow (↑(Complex.HadamardThreeLines.sSupNormIm …
      -/
    · simp only [ofReal_zero, zero_add]
      /-
        🎉 no goals
      -/
      /-
        case convert_8
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        f : Complex → E
        inst✝ : NormedSpace Complex E
        z : Complex
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        ⊢ ContinuousWithinAt (fun ε => Norm.norm (HMul.hMul (HPow.hPow (HAdd.hAdd ↑ε ↑ …
      -/
    · simp_rw [← ofReal_add, norm_eq_abs]
      have : ∀ x ∈ Ioi 0, (x + sSupNormIm f 0) ^ (1 - z.re) * (x + (sSupNormIm f 1)) ^ z.re
          = abs (↑(x + sSupNormIm f 0) ^ (1 - z) * ↑(x + sSupNormIm f 1) ^ z) := by
              intro x hx
              simp only [map_mul]
              repeat rw [abs_cpow_eq_rpow_re_of_nonneg (le_of_lt (sSupNormIm_eps_pos f hx _)) _]
              · simp only [sub_re, one_re]
              · simpa using (ne_comm.mpr (ne_of_lt hz.1))
              · simpa [sub_eq_zero] using (ne_comm.mpr (ne_of_lt hz.2))
      /-
        case convert_8
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        f : Complex → E
        inst✝ : NormedSpace Complex E
        z : Complex
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
        ⊢ ContinuousWithinAt (fun ε => Complex.abs (HMul.hMul (HPow.hPow (↑(HAdd.hAdd  …
      -/
      apply tendsto_nhdsWithin_congr this _
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        f : Complex → E
        inst✝ : NormedSpace Complex E
        z : Complex
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
        this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
        ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd x (Complex.Hadamard …
      -/
      simp only [zero_add]
      rw [map_mul, abs_cpow_eq_rpow_re_of_nonneg (sSupNormIm_nonneg _ _) _,
        abs_cpow_eq_rpow_re_of_nonneg (sSupNormIm_nonneg _ _) _]
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          f : Complex → E
          inst✝ : NormedSpace Complex E
          z : Complex
          hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
          hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
          hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
          this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
          ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd x (Complex.Hadamard …
        -/
      · apply Tendsto.mul
          /-
            case hf
            E : Type u_1
            inst✝¹ : NormedAddCommGroup E
            f : Complex → E
            inst✝ : NormedSpace Complex E
            z : Complex
            hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
            hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
            hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
            this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
            ⊢ Filter.Tendsto (fun x => HPow.hPow (HAdd.hAdd x (Complex.HadamardThreeLines. …
          -/
        · apply Tendsto.rpow_const
            /-
              case hf.hf
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              f : Complex → E
              inst✝ : NormedSpace Complex E
              z : Complex
              hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
              hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
              hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
              this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
              ⊢ Filter.Tendsto (fun a => HAdd.hAdd a (Complex.HadamardThreeLines.sSupNormIm  …
            -/
          · nth_rw 2 [← zero_add (sSupNormIm f 0)]
            exact Tendsto.add_const (sSupNormIm f 0) (tendsto_nhdsWithin_of_tendsto_nhds
              (Continuous.tendsto continuous_id' _))
            /-
              case hf.h
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              f : Complex → E
              inst✝ : NormedSpace Complex E
              z : Complex
              hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
              hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
              hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
              this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
              ⊢ Or (Ne (Complex.HadamardThreeLines.sSupNormIm f 0) 0) (LE.le 0 (HSub.hSub 1  …
            -/
          · right; simp only [sub_nonneg, le_of_lt hz.2]
                   /-
                     🎉 no goals
                   -/
          /-
            case hg
            E : Type u_1
            inst✝¹ : NormedAddCommGroup E
            f : Complex → E
            inst✝ : NormedSpace Complex E
            z : Complex
            hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
            hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
            hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
            this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
            ⊢ Filter.Tendsto (fun x => HPow.hPow (HAdd.hAdd x (Complex.HadamardThreeLines. …
          -/
        · apply Tendsto.rpow_const
            /-
              case hg.hf
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              f : Complex → E
              inst✝ : NormedSpace Complex E
              z : Complex
              hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
              hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
              hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
              this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
              ⊢ Filter.Tendsto (fun a => HAdd.hAdd a (Complex.HadamardThreeLines.sSupNormIm  …
            -/
          · nth_rw 2 [← zero_add (sSupNormIm f 1)]
            exact Tendsto.add_const (sSupNormIm f 1) (tendsto_nhdsWithin_of_tendsto_nhds
              (Continuous.tendsto continuous_id' _))
            /-
              case hg.h
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              f : Complex → E
              inst✝ : NormedSpace Complex E
              z : Complex
              hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
              hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
              hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
              this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
              ⊢ Or (Ne (Complex.HadamardThreeLines.sSupNormIm f 1) 0) (LE.le 0 z.re)
            -/
          · right; simp only [sub_nonneg, le_of_lt hz.1]
                   /-
                     🎉 no goals
                   -/
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          f : Complex → E
          inst✝ : NormedSpace Complex E
          z : Complex
          hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
          hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
          hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
          this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
          ⊢ Ne z.re 0
        -/
      · simpa using (ne_comm.mpr (ne_of_lt hz.1))
        /-
          🎉 no goals
        -/
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          f : Complex → E
          inst✝ : NormedSpace Complex E
          z : Complex
          hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
          hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
          hz : Membership.mem (Complex.HadamardThreeLines.verticalStrip 0 1) z
          this : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (HMul.hMul (HPow.hPow ( …
          ⊢ Ne (HSub.hSub 1 z).re 0
        -/
      · simpa [sub_eq_zero] using (ne_comm.mpr (ne_of_lt hz.2))
        /-
          🎉 no goals
        -/


/--
**Hadamard three-line theorem** on `re ⁻¹' [0,1]`: If `f` is a bounded function, continuous on the
closed strip `re ⁻¹' [0,1]` and differentiable on open strip `re ⁻¹' (0,1)`, then for
`M(x) := sup ((norm ∘ f) '' (re ⁻¹' {x}))` we have that for all `z` in the closed strip
`re ⁻¹' [0,1]` the inequality `‖f(z)‖ ≤ M(0) ^ (1 - z.re) * M(1) ^ z.re` holds. -/
lemma norm_le_interpStrip_of_mem_verticalClosedStrip (f : ℂ → E) {z : ℂ}
    (hz : z ∈ verticalClosedStrip 0 1) (hd : DiffContOnCl ℂ f (verticalStrip 0 1))
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1))) :
    ‖f z‖ ≤ ‖interpStrip f z‖ := by
  apply le_on_closure (fun w hw ↦ norm_le_interpStrip_of_mem_verticalStrip_zero f w hd hB hw)
    (Continuous.comp_continuousOn' continuous_norm hd.2)
    (Continuous.comp_continuousOn' continuous_norm (diffContOnCl_interpStrip f).2)
  /-
    case hx
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    z : Complex
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    ⊢ Membership.mem (closure (Complex.HadamardThreeLines.verticalStrip 0 1)) z
  -/
  rwa [verticalClosedStrip, ← closure_Ioo zero_ne_one, ← closure_preimage_re] at hz
  /-
    🎉 no goals
  -/


/-- **Hadamard three-line theorem** on `re ⁻¹' [0,1]` (Variant in simpler terms): Let `f` be a
bounded function, continuous on the closed strip `re ⁻¹' [0,1]` and differentiable on open strip
`re ⁻¹' (0,1)`. If, for all `z.re = 0`, `‖f z‖ ≤ a` for some `a ∈ ℝ` and, similarly, for all
`z.re = 1`, `‖f z‖ ≤ b` for some `b ∈ ℝ` then for all `z` in the closed strip
`re ⁻¹' [0,1]` the inequality `‖f(z)‖ ≤ a ^ (1 - z.re) * b ^ z.re` holds. -/
lemma norm_le_interp_of_mem_verticalClosedStrip' (f : ℂ → E) {z : ℂ} {a b : ℝ}
    (hz : z ∈ verticalClosedStrip 0 1) (hd : DiffContOnCl ℂ f (verticalStrip 0 1))
    (hB : BddAbove ((norm ∘ f) '' (verticalClosedStrip 0 1)))
    (ha : ∀ z ∈ re ⁻¹' {0}, ‖f z‖ ≤ a) (hb : ∀ z ∈ re ⁻¹' {1}, ‖f z‖ ≤ b) :
    ‖f z‖ ≤ a ^ (1 - z.re) * b ^ z.re := by
  have : ‖interpStrip f z‖ ≤ (sSupNormIm f 0) ^ (1 - z.re) * (sSupNormIm f 1) ^ z.re := by
    by_cases h : sSupNormIm f 0 = 0 ∨ sSupNormIm f 1 = 0
    · rw [interpStrip_eq_of_zero f z h, norm_zero, mul_nonneg_iff]
      left
      exact ⟨Real.rpow_nonneg (sSupNormIm_nonneg f _) _,
        Real.rpow_nonneg (sSupNormIm_nonneg f _) _ ⟩
    · push_neg at h
      rcases h with ⟨h0, h1⟩
      rw [ne_comm] at h0 h1
      simp_rw [interpStrip_eq_of_pos f _ (lt_of_le_of_ne (sSupNormIm_nonneg f 0) h0)
        (lt_of_le_of_ne (sSupNormIm_nonneg f 1) h1)]
      simp only [norm_eq_abs, map_mul]
      rw [abs_cpow_eq_rpow_re_of_pos ((Ne.le_iff_lt h0).mp (sSupNormIm_nonneg f _)) _]
      rw [abs_cpow_eq_rpow_re_of_pos ((Ne.le_iff_lt h1).mp (sSupNormIm_nonneg f _)) _]
      simp only [sub_re, one_re, le_refl]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    z : Complex
    a b : Real
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
    hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
    this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
    ⊢ LE.le (Norm.norm (f z)) (HMul.hMul (HPow.hPow a (HSub.hSub 1 z.re)) (HPow.hP …
  -/
  apply (norm_le_interpStrip_of_mem_verticalClosedStrip f hz hd hB).trans (this.trans _)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    z : Complex
    a b : Real
    hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
    hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
    hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
    ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
    hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
    this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Complex.HadamardThreeLines.sSupNormIm f 0) (HSu …
  -/
  apply mul_le_mul_of_nonneg _ _ (Real.rpow_nonneg (sSupNormIm_nonneg f _) _)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      z : Complex
      a b : Real
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
      ⊢ LE.le 0 (HPow.hPow b z.re)
    -/
  · apply (Real.rpow_nonneg _ _)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      z : Complex
      a b : Real
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
      ⊢ LE.le 0 b
    -/
    specialize hb 1
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      z : Complex
      a b : Real
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
      hb : Membership.mem (Set.preimage Complex.re (Singleton.singleton 1)) 1 → LE.l …
      ⊢ LE.le 0 b
    -/
    simp only [mem_preimage, one_re, mem_singleton_iff, forall_true_left] at hb
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      z : Complex
      a b : Real
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
      hb : LE.le (Norm.norm (f 1)) b
      ⊢ LE.le 0 b
    -/
    exact (norm_nonneg _).trans hb
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      z : Complex
      a b : Real
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
      ⊢ LE.le (HPow.hPow (Complex.HadamardThreeLines.sSupNormIm f 0) (HSub.hSub 1 z. …
    -/
  · apply Real.rpow_le_rpow (sSupNormIm_nonneg f _) _ (sub_nonneg.mpr hz.2)
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        z : Complex
        a b : Real
        hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
        ⊢ LE.le (Complex.HadamardThreeLines.sSupNormIm f 0) a
      -/
    · rw [sSupNormIm]
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        z : Complex
        a b : Real
        hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
        ⊢ LE.le (SupSet.sSup (Set.image (Function.comp Norm.norm f) (Set.preimage Comp …
      -/
      apply csSup_le _
      · simpa [comp_apply, mem_image, forall_exists_index,
          and_imp, forall_apply_eq_imp_iff₂] using ha
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          z : Complex
          a b : Real
          hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
          hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
          hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
          ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
          hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
          this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
          ⊢ (Set.image (Function.comp Norm.norm f) (Set.preimage Complex.re (Singleton.s …
        -/
      · use ‖(f 0)‖, 0
        simp only [mem_preimage, zero_re, mem_singleton_iff, comp_apply,
          and_self]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      z : Complex
      a b : Real
      hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
      hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
      hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
      ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
      this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
      ⊢ LE.le (HPow.hPow (Complex.HadamardThreeLines.sSupNormIm f 1) z.re) (HPow.hPo …
    -/
  · apply Real.rpow_le_rpow (sSupNormIm_nonneg f _) _ hz.1
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        z : Complex
        a b : Real
        hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
        ⊢ LE.le (Complex.HadamardThreeLines.sSupNormIm f 1) b
      -/
    · rw [sSupNormIm]
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Complex → E
        z : Complex
        a b : Real
        hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
        hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
        hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
        ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
        this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
        ⊢ LE.le (SupSet.sSup (Set.image (Function.comp Norm.norm f) (Set.preimage Comp …
      -/
      apply csSup_le _
      · simpa [comp_apply, mem_image, forall_exists_index,
          and_imp, forall_apply_eq_imp_iff₂] using hb
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Complex → E
          z : Complex
          a b : Real
          hz : Membership.mem (Complex.HadamardThreeLines.verticalClosedStrip 0 1) z
          hd : DiffContOnCl Complex f (Complex.HadamardThreeLines.verticalStrip 0 1)
          hB : BddAbove (Set.image (Function.comp Norm.norm f) (Complex.HadamardThreeLin …
          ha : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
          hb : ∀ (z : Complex), Membership.mem (Set.preimage Complex.re (Singleton.singl …
          this : LE.le (Norm.norm (Complex.HadamardThreeLines.interpStrip f z)) (HMul.hM …
          ⊢ (Set.image (Function.comp Norm.norm f) (Set.preimage Complex.re (Singleton.s …
        -/
      · use ‖(f 1)‖, 1
        simp only [mem_preimage, one_re, mem_singleton_iff, comp_apply,
          and_self]


