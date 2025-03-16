/-- We say that `l : Filter ℂ` is an *exponential comparison filter* if the real part tends to
infinity along `l` and the imaginary part grows subexponentially compared to the real part. These
properties guarantee that `(fun z ↦ z ^ a₁ * exp (b₁ * z)) =o[l] (fun z ↦ z ^ a₂ * exp (b₂ * z))`
for any complex `a₁`, `a₂` and real `b₁ < b₂`.

In particular, the second property is automatically satisfied if the imaginary part is bounded along
`l`. -/
structure IsExpCmpFilter (l : Filter ℂ) : Prop where
  tendsto_re : Tendsto re l atTop
  isBigO_im_pow_re : ∀ n : ℕ, (fun z : ℂ => z.im ^ n) =O[l] fun z => Real.exp z.re


theorem of_isBigO_im_re_rpow (hre : Tendsto re l atTop) (r : ℝ) (hr : im =O[l] fun z => z.re ^ r) :
    IsExpCmpFilter l :=
  ⟨hre, fun n =>
    IsLittleO.isBigO <|
      calc
        (fun z : ℂ => z.im ^ n) =O[l] fun z => (z.re ^ r) ^ n := hr.pow n
        _ =ᶠ[l] fun z => z.re ^ (r * n) :=
          ((hre.eventually_ge_atTop 0).mono fun z hz => by
            /-
              l : Filter Complex
              hre : Filter.Tendsto Complex.re l Filter.atTop
              r : Real
              hr : Asymptotics.IsBigO l Complex.im fun z => HPow.hPow z.re r
              n : Nat
              z : Complex
              hz : LE.le 0 z.re
              ⊢ Eq ((fun z => HPow.hPow (HPow.hPow z.re r) n) z) ((fun z => HPow.hPow z.re ( …
            -/
            simp only [Real.rpow_mul hz r n, Real.rpow_natCast])
            /-
              🎉 no goals
            -/
        _ =o[l] fun z => Real.exp z.re := (isLittleO_rpow_exp_atTop _).comp_tendsto hre ⟩


theorem of_isBigO_im_re_pow (hre : Tendsto re l atTop) (n : ℕ) (hr : im =O[l] fun z => z.re ^ n) :
    IsExpCmpFilter l :=
  of_isBigO_im_re_rpow hre n <| mod_cast hr


theorem of_boundedUnder_abs_im (hre : Tendsto re l atTop)
    (him : IsBoundedUnder (· ≤ ·) l fun z => |z.im|) : IsExpCmpFilter l :=
  of_isBigO_im_re_pow hre 0 <| by
    /-
      l : Filter Complex
      hre : Filter.Tendsto Complex.re l Filter.atTop
      him : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun z => _root_.abs z …
      ⊢ Asymptotics.IsBigO l Complex.im fun z => HPow.hPow z.re 0
    -/
    simpa only [pow_zero] using him.isBigO_const (f := im) one_ne_zero
    /-
      🎉 no goals
    -/


theorem of_boundedUnder_im (hre : Tendsto re l atTop) (him_le : IsBoundedUnder (· ≤ ·) l im)
    (him_ge : IsBoundedUnder (· ≥ ·) l im) : IsExpCmpFilter l :=
  of_boundedUnder_abs_im hre <| isBoundedUnder_le_abs.2 ⟨him_le, him_ge⟩


theorem eventually_ne (hl : IsExpCmpFilter l) : ∀ᶠ w : ℂ in l, w ≠ 0 :=
  hl.tendsto_re.eventually_ne_atTop' _


theorem tendsto_abs_re (hl : IsExpCmpFilter l) : Tendsto (fun z : ℂ => |z.re|) l atTop :=
  tendsto_abs_atTop_atTop.comp hl.tendsto_re


theorem tendsto_abs (hl : IsExpCmpFilter l) : Tendsto abs l atTop :=
  tendsto_atTop_mono abs_re_le_abs hl.tendsto_abs_re


theorem isLittleO_log_re_re (hl : IsExpCmpFilter l) : (fun z => Real.log z.re) =o[l] re :=
  Real.isLittleO_log_id_atTop.comp_tendsto hl.tendsto_re


theorem isLittleO_im_pow_exp_re (hl : IsExpCmpFilter l) (n : ℕ) :
    (fun z : ℂ => z.im ^ n) =o[l] fun z => Real.exp z.re :=
  flip IsLittleO.of_pow two_ne_zero <|
    calc
                                                                    /-
                                                                      l : Filter Complex
                                                                      hl : Complex.IsExpCmpFilter l
                                                                      n : Nat
                                                                      ⊢ Eq (fun z => HPow.hPow (HPow.hPow z.im n) 2) fun z => HPow.hPow z.im (HMul.h …
                                                                    -/
      (fun z : ℂ ↦ (z.im ^ n) ^ 2) = (fun z ↦ z.im ^ (2 * n)) := by simp only [pow_mul']
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
      _ =O[l] fun z ↦ Real.exp z.re := hl.isBigO_im_pow_re _
                                                /-
                                                  l : Filter Complex
                                                  hl : Complex.IsExpCmpFilter l
                                                  n : Nat
                                                  ⊢ Eq (fun z => Real.exp z.re) fun z => HPow.hPow (Real.exp z.re) 1
                                                -/
      _ =     fun z ↦ (Real.exp z.re) ^ 1 := by simp only [pow_one]
                                                /-
                                                  🎉 no goals
                                                -/
      _ =o[l] fun z ↦ (Real.exp z.re) ^ 2 :=
        (isLittleO_pow_pow_atTop_of_lt one_lt_two).comp_tendsto <|
          Real.tendsto_exp_atTop.comp hl.tendsto_re


theorem abs_im_pow_eventuallyLE_exp_re (hl : IsExpCmpFilter l) (n : ℕ) :
    (fun z : ℂ => |z.im| ^ n) ≤ᶠ[l] fun z => Real.exp z.re := by
  /-
    l : Filter Complex
    hl : Complex.IsExpCmpFilter l
    n : Nat
    ⊢ l.EventuallyLE (fun z => HPow.hPow (_root_.abs z.im) n) fun z => Real.exp z.re
  -/
  simpa using (hl.isLittleO_im_pow_exp_re n).bound zero_lt_one
  /-
    🎉 no goals
  -/


/-- If `l : Filter ℂ` is an "exponential comparison filter", then $\log |z| =o(ℜ z)$ along `l`.
This is the main lemma in the proof of `Complex.IsExpCmpFilter.isLittleO_cpow_exp` below.
-/
theorem isLittleO_log_abs_re (hl : IsExpCmpFilter l) : (fun z => Real.log (abs z)) =o[l] re :=
  calc
    (fun z => Real.log (abs z)) =O[l] fun z => Real.log (√2) + Real.log (max z.re |z.im|) :=
      IsBigO.of_bound 1 <|
        (hl.tendsto_re.eventually_ge_atTop 1).mono fun z hz => by
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            ⊢ LE.le (Norm.norm (Real.log (Complex.abs z))) (HMul.hMul 1 (Norm.norm (HAdd.h …
          -/
          have h2 : 0 < √2 := by simp
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            h2 : LT.lt 0 (Real.sqrt 2)
            ⊢ LE.le (Norm.norm (Real.log (Complex.abs z))) (HMul.hMul 1 (Norm.norm (HAdd.h …
          -/
          have hz' : 1 ≤ abs z := hz.trans (re_le_abs z)
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            h2 : LT.lt 0 (Real.sqrt 2)
            hz' : LE.le 1 (Complex.abs z)
            ⊢ LE.le (Norm.norm (Real.log (Complex.abs z))) (HMul.hMul 1 (Norm.norm (HAdd.h …
          -/
          have hm₀ : 0 < max z.re |z.im| := lt_max_iff.2 (Or.inl <| one_pos.trans_le hz)
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            h2 : LT.lt 0 (Real.sqrt 2)
            hz' : LE.le 1 (Complex.abs z)
            hm₀ : LT.lt 0 (Max.max z.re (_root_.abs z.im))
            ⊢ LE.le (Norm.norm (Real.log (Complex.abs z))) (HMul.hMul 1 (Norm.norm (HAdd.h …
          -/
          rw [one_mul, Real.norm_eq_abs, _root_.abs_of_nonneg (Real.log_nonneg hz')]
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            h2 : LT.lt 0 (Real.sqrt 2)
            hz' : LE.le 1 (Complex.abs z)
            hm₀ : LT.lt 0 (Max.max z.re (_root_.abs z.im))
            ⊢ LE.le (Real.log (Complex.abs z)) (Norm.norm (HAdd.hAdd (Real.log (Real.sqrt  …
          -/
          refine le_trans ?_ (le_abs_self _)
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            h2 : LT.lt 0 (Real.sqrt 2)
            hz' : LE.le 1 (Complex.abs z)
            hm₀ : LT.lt 0 (Max.max z.re (_root_.abs z.im))
            ⊢ LE.le (Real.log (Complex.abs z)) (HAdd.hAdd (Real.log (Real.sqrt 2)) (Real.l …
          -/
          rw [← Real.log_mul, Real.log_le_log_iff, ← _root_.abs_of_nonneg (le_trans zero_le_one hz)]
          /-
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            z : Complex
            hz : LE.le 1 z.re
            h2 : LT.lt 0 (Real.sqrt 2)
            hz' : LE.le 1 (Complex.abs z)
            hm₀ : LT.lt 0 (Max.max z.re (_root_.abs z.im))
            ⊢ LE.le (Complex.abs z) (HMul.hMul (Real.sqrt 2) (Max.max (_root_.abs z.re) (_ …
          -/
          exacts [abs_le_sqrt_two_mul_max z, one_pos.trans_le hz', mul_pos h2 hm₀, h2.ne', hm₀.ne']
          /-
            🎉 no goals
          -/
    _ =o[l] re :=
      IsLittleO.add (isLittleO_const_left.2 <| Or.inr <| hl.tendsto_abs_re) <|
        isLittleO_iff_nat_mul_le.2 fun n => by
          filter_upwards [isLittleO_iff_nat_mul_le'.1 hl.isLittleO_log_re_re n,
            hl.abs_im_pow_eventuallyLE_exp_re n,
            hl.tendsto_re.eventually_gt_atTop 1] with z hre him h₁
          /-
            case h
            l : Filter Complex
            hl : Complex.IsExpCmpFilter l
            n : Nat
            z : Complex
            hre : LE.le (HMul.hMul (↑n) (Norm.norm (Real.log z.re))) (Norm.norm z.re)
            him : LE.le (HPow.hPow (_root_.abs z.im) n) (Real.exp z.re)
            h₁ : LT.lt 1 z.re
            ⊢ LE.le (HMul.hMul (↑n) (Norm.norm (Real.log (Max.max z.re (_root_.abs z.im))) …
          -/
          rcases le_total |z.im| z.re with hle | hle
            /-
              case h.inl
              l : Filter Complex
              hl : Complex.IsExpCmpFilter l
              n : Nat
              z : Complex
              hre : LE.le (HMul.hMul (↑n) (Norm.norm (Real.log z.re))) (Norm.norm z.re)
              him : LE.le (HPow.hPow (_root_.abs z.im) n) (Real.exp z.re)
              h₁ : LT.lt 1 z.re
              hle : LE.le (_root_.abs z.im) z.re
              ⊢ LE.le (HMul.hMul (↑n) (Norm.norm (Real.log (Max.max z.re (_root_.abs z.im))) …
            -/
          · rwa [max_eq_left hle]
            /-
              🎉 no goals
            -/
            /-
              case h.inr
              l : Filter Complex
              hl : Complex.IsExpCmpFilter l
              n : Nat
              z : Complex
              hre : LE.le (HMul.hMul (↑n) (Norm.norm (Real.log z.re))) (Norm.norm z.re)
              him : LE.le (HPow.hPow (_root_.abs z.im) n) (Real.exp z.re)
              h₁ : LT.lt 1 z.re
              hle : LE.le z.re (_root_.abs z.im)
              ⊢ LE.le (HMul.hMul (↑n) (Norm.norm (Real.log (Max.max z.re (_root_.abs z.im))) …
            -/
          · have H : 1 < |z.im| := h₁.trans_le hle
            /-
              case h.inr
              l : Filter Complex
              hl : Complex.IsExpCmpFilter l
              n : Nat
              z : Complex
              hre : LE.le (HMul.hMul (↑n) (Norm.norm (Real.log z.re))) (Norm.norm z.re)
              him : LE.le (HPow.hPow (_root_.abs z.im) n) (Real.exp z.re)
              h₁ : LT.lt 1 z.re
              hle : LE.le z.re (_root_.abs z.im)
              H : LT.lt 1 (_root_.abs z.im)
              ⊢ LE.le (HMul.hMul (↑n) (Norm.norm (Real.log (Max.max z.re (_root_.abs z.im))) …
            -/
            norm_cast at *
            rwa [max_eq_right hle, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (Real.log_pos H),
              ← Real.log_pow, Real.log_le_iff_le_exp (pow_pos (one_pos.trans H) _),
              abs_of_pos (one_pos.trans h₁)]


lemma isTheta_cpow_exp_re_mul_log (hl : IsExpCmpFilter l) (a : ℂ) :
    (· ^ a) =Θ[l] fun z ↦ Real.exp (re a * Real.log (abs z)) :=
  calc
    (fun z => z ^ a) =Θ[l] (fun z : ℂ => (abs z ^ re a)) :=
      isTheta_cpow_const_rpow fun _ _ => hl.eventually_ne
    _ =ᶠ[l] fun z => Real.exp (re a * Real.log (abs z)) :=
                                            /-
                                              l : Filter Complex
                                              hl : Complex.IsExpCmpFilter l
                                              a z : Complex
                                              hz : Ne z 0
                                              ⊢ Eq ((fun z => HPow.hPow (Complex.abs z) a.re) z) ((fun z => Real.exp (HMul.h …
                                            -/
      (hl.eventually_ne.mono fun z hz => by simp only [Real.rpow_def_of_pos, abs.pos hz, mul_comm])
                                            /-
                                              🎉 no goals
                                            -/


/-- If `l : Filter ℂ` is an "exponential comparison filter", then for any complex `a` and any
positive real `b`, we have `(fun z ↦ z ^ a) =o[l] (fun z ↦ exp (b * z))`. -/
theorem isLittleO_cpow_exp (hl : IsExpCmpFilter l) (a : ℂ) {b : ℝ} (hb : 0 < b) :
    (fun z => z ^ a) =o[l] fun z => exp (b * z) :=
  calc
    (fun z => z ^ a) =Θ[l] fun z => Real.exp (re a * Real.log (abs z)) :=
      hl.isTheta_cpow_exp_re_mul_log a
    _ =o[l] fun z => exp (b * z) :=
      IsLittleO.of_norm_right <| by
        /-
          l : Filter Complex
          hl : Complex.IsExpCmpFilter l
          a : Complex
          b : Real
          hb : LT.lt 0 b
          ⊢ Asymptotics.IsLittleO l (fun z => Real.exp (HMul.hMul a.re (Real.log (Comple …
        -/
        simp only [norm_eq_abs, abs_exp, re_ofReal_mul, Real.isLittleO_exp_comp_exp_comp]
        refine (IsEquivalent.refl.sub_isLittleO ?_).symm.tendsto_atTop
          (hl.tendsto_re.const_mul_atTop hb)
        /-
          l : Filter Complex
          hl : Complex.IsExpCmpFilter l
          a : Complex
          b : Real
          hb : LT.lt 0 b
          ⊢ Asymptotics.IsLittleO l (fun x => HMul.hMul a.re (Real.log (Complex.abs x))) …
        -/
        exact (hl.isLittleO_log_abs_re.const_mul_left _).const_mul_right hb.ne'
        /-
          🎉 no goals
        -/


/-- If `l : Filter ℂ` is an "exponential comparison filter", then for any complex `a₁`, `a₂` and any
real `b₁ < b₂`, we have `(fun z ↦ z ^ a₁ * exp (b₁ * z)) =o[l] (fun z ↦ z ^ a₂ * exp (b₂ * z))`. -/
theorem isLittleO_cpow_mul_exp {b₁ b₂ : ℝ} (hl : IsExpCmpFilter l) (hb : b₁ < b₂) (a₁ a₂ : ℂ) :
    (fun z => z ^ a₁ * exp (b₁ * z)) =o[l] fun z => z ^ a₂ * exp (b₂ * z) :=
  calc
    (fun z => z ^ a₁ * exp (b₁ * z)) =ᶠ[l] fun z => z ^ a₂ * exp (b₁ * z) * z ^ (a₁ - a₂) :=
      hl.eventually_ne.mono fun z hz => by
        /-
          l : Filter Complex
          b₁ b₂ : Real
          hl : Complex.IsExpCmpFilter l
          hb : LT.lt b₁ b₂
          a₁ a₂ z : Complex
          hz : Ne z 0
          ⊢ Eq ((fun z => HMul.hMul (HPow.hPow z a₁) (Complex.exp (HMul.hMul (↑b₁) z)))  …
        -/
        simp only
        /-
          l : Filter Complex
          b₁ b₂ : Real
          hl : Complex.IsExpCmpFilter l
          hb : LT.lt b₁ b₂
          a₁ a₂ z : Complex
          hz : Ne z 0
          ⊢ Eq (HMul.hMul (HPow.hPow z a₁) (Complex.exp (HMul.hMul (↑b₁) z))) (HMul.hMul …
        -/
        rw [mul_right_comm, ← cpow_add _ _ hz, add_sub_cancel]
        /-
          🎉 no goals
        -/
    _ =o[l] fun z => z ^ a₂ * exp (b₁ * z) * exp (↑(b₂ - b₁) * z) :=
      ((isBigO_refl (fun z => z ^ a₂ * exp (b₁ * z)) l).mul_isLittleO <|
        hl.isLittleO_cpow_exp _ (sub_pos.2 hb))
    _ =ᶠ[l] fun z => z ^ a₂ * exp (b₂ * z) := by
      /-
        l : Filter Complex
        b₁ b₂ : Real
        hl : Complex.IsExpCmpFilter l
        hb : LT.lt b₁ b₂
        a₁ a₂ : Complex
        ⊢ l.EventuallyEq (fun z => HMul.hMul (HMul.hMul (HPow.hPow z a₂) (Complex.exp  …
      -/
      simp only [ofReal_sub, sub_mul, mul_assoc, ← exp_add, add_sub_cancel]
      /-
        l : Filter Complex
        b₁ b₂ : Real
        hl : Complex.IsExpCmpFilter l
        hb : LT.lt b₁ b₂
        a₁ a₂ : Complex
        ⊢ l.EventuallyEq (fun z => HMul.hMul (HPow.hPow z a₂) (Complex.exp (HMul.hMul  …
      -/
      norm_cast
      /-
        🎉 no goals
      -/


/-- If `l : Filter ℂ` is an "exponential comparison filter", then for any complex `a` and any
negative real `b`, we have `(fun z ↦ exp (b * z)) =o[l] (fun z ↦ z ^ a)`. -/
theorem isLittleO_exp_cpow (hl : IsExpCmpFilter l) (a : ℂ) {b : ℝ} (hb : b < 0) :
                                                      /-
                                                        l : Filter Complex
                                                        hl : Complex.IsExpCmpFilter l
                                                        a : Complex
                                                        b : Real
                                                        hb : LT.lt b 0
                                                        ⊢ Asymptotics.IsLittleO l (fun z => Complex.exp (HMul.hMul (↑b) z)) fun z => H …
                                                      -/
    (fun z => exp (b * z)) =o[l] fun z => z ^ a := by simpa using hl.isLittleO_cpow_mul_exp hb 0 a
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If `l : Filter ℂ` is an "exponential comparison filter", then for any complex `a₁`, `a₂` and any
natural `b₁ < b₂`, we have
`(fun z ↦ z ^ a₁ * exp (b₁ * z)) =o[l] (fun z ↦ z ^ a₂ * exp (b₂ * z))`. -/
theorem isLittleO_pow_mul_exp {b₁ b₂ : ℝ} (hl : IsExpCmpFilter l) (hb : b₁ < b₂) (m n : ℕ) :
    (fun z => z ^ m * exp (b₁ * z)) =o[l] fun z => z ^ n * exp (b₂ * z) := by
  /-
    l : Filter Complex
    b₁ b₂ : Real
    hl : Complex.IsExpCmpFilter l
    hb : LT.lt b₁ b₂
    m n : Nat
    ⊢ Asymptotics.IsLittleO l (fun z => HMul.hMul (HPow.hPow z m) (Complex.exp (HM …
  -/
  simpa only [cpow_natCast] using hl.isLittleO_cpow_mul_exp hb m n
  /-
    🎉 no goals
  -/


/-- If `l : Filter ℂ` is an "exponential comparison filter", then for any complex `a₁`, `a₂` and any
integer `b₁ < b₂`, we have
`(fun z ↦ z ^ a₁ * exp (b₁ * z)) =o[l] (fun z ↦ z ^ a₂ * exp (b₂ * z))`. -/
theorem isLittleO_zpow_mul_exp {b₁ b₂ : ℝ} (hl : IsExpCmpFilter l) (hb : b₁ < b₂) (m n : ℤ) :
    (fun z => z ^ m * exp (b₁ * z)) =o[l] fun z => z ^ n * exp (b₂ * z) := by
  /-
    l : Filter Complex
    b₁ b₂ : Real
    hl : Complex.IsExpCmpFilter l
    hb : LT.lt b₁ b₂
    m n : Int
    ⊢ Asymptotics.IsLittleO l (fun z => HMul.hMul (HPow.hPow z m) (Complex.exp (HM …
  -/
  simpa only [cpow_intCast] using hl.isLittleO_cpow_mul_exp hb m n
  /-
    🎉 no goals
  -/


