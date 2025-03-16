/-- Inverse of the `exp` function. Returns values such that `(log x).im > - π` and `(log x).im ≤ π`.
  `log 0 = 0`-/
@[pp_nodot]
noncomputable def log (x : ℂ) : ℂ :=
  x.abs.log + arg x * I


                                                    /-
                                                      x : Complex
                                                      ⊢ Eq (Complex.log x).re (Real.log (Complex.abs x))
                                                    -/
theorem log_re (x : ℂ) : x.log.re = x.abs.log := by simp [log]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                /-
                                                  x : Complex
                                                  ⊢ Eq (Complex.log x).im x.arg
                                                -/
theorem log_im (x : ℂ) : x.log.im = x.arg := by simp [log]
                                                /-
                                                  🎉 no goals
                                                -/


                                                         /-
                                                           x : Complex
                                                           ⊢ LT.lt (Neg.neg Real.pi) (Complex.log x).im
                                                         -/
theorem neg_pi_lt_log_im (x : ℂ) : -π < (log x).im := by simp only [log_im, neg_pi_lt_arg]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                    /-
                                                      x : Complex
                                                      ⊢ LE.le (Complex.log x).im Real.pi
                                                    -/
theorem log_im_le_pi (x : ℂ) : (log x).im ≤ π := by simp only [log_im, arg_le_pi]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem exp_log {x : ℂ} (hx : x ≠ 0) : exp (log x) = x := by
  rw [log, exp_add_mul_I, ← ofReal_sin, sin_arg, ← ofReal_cos, cos_arg hx, ← ofReal_exp,
    Real.exp_log (abs.pos hx), mul_add, ofReal_div, ofReal_div,
    mul_div_cancel₀ _ (ofReal_ne_zero.2 <| abs.ne_zero hx), ← mul_assoc,
    mul_div_cancel₀ _ (ofReal_ne_zero.2 <| abs.ne_zero hx), re_add_im]


@[simp]
theorem range_exp : Set.range exp = {0}ᶜ :=
  Set.ext fun x =>
    ⟨by
      /-
        x : Complex
        ⊢ Membership.mem (Set.range Complex.exp) x → Membership.mem (HasCompl.compl (S …
      -/
      rintro ⟨x, rfl⟩
      /-
        case intro
        x : Complex
        ⊢ Membership.mem (HasCompl.compl (Singleton.singleton 0)) (Complex.exp x)
      -/
      exact exp_ne_zero x, fun hx => ⟨log x, exp_log hx⟩⟩
      /-
        🎉 no goals
      -/


theorem log_exp {x : ℂ} (hx₁ : -π < x.im) (hx₂ : x.im ≤ π) : log (exp x) = x := by
  rw [log, abs_exp, Real.log_exp, exp_eq_exp_re_mul_sin_add_cos, ← ofReal_exp,
    arg_mul_cos_add_sin_mul_I (Real.exp_pos _) ⟨hx₁, hx₂⟩, re_add_im]


theorem exp_inj_of_neg_pi_lt_of_le_pi {x y : ℂ} (hx₁ : -π < x.im) (hx₂ : x.im ≤ π) (hy₁ : -π < y.im)
    (hy₂ : y.im ≤ π) (hxy : exp x = exp y) : x = y := by
  /-
    x y : Complex
    hx₁ : LT.lt (Neg.neg Real.pi) x.im
    hx₂ : LE.le x.im Real.pi
    hy₁ : LT.lt (Neg.neg Real.pi) y.im
    hy₂ : LE.le y.im Real.pi
    hxy : Eq (Complex.exp x) (Complex.exp y)
    ⊢ Eq x y
  -/
  rw [← log_exp hx₁ hx₂, ← log_exp hy₁ hy₂, hxy]
  /-
    🎉 no goals
  -/


theorem ofReal_log {x : ℝ} (hx : 0 ≤ x) : (x.log : ℂ) = log x :=
                  /-
                    x : Real
                    hx : LE.le 0 x
                    ⊢ Eq (↑(Real.log x)).re (Complex.log ↑x).re
                  -/
  Complex.ext (by rw [log_re, ofReal_re, abs_of_nonneg hx])
                  /-
                    🎉 no goals
                  -/
        /-
          x : Real
          hx : LE.le 0 x
          ⊢ Eq (↑(Real.log x)).im (Complex.log ↑x).im
        -/
    (by rw [ofReal_im, log_im, arg_ofReal_of_nonneg hx])
        /-
          🎉 no goals
        -/


@[simp, norm_cast]
lemma natCast_log {n : ℕ} : Real.log n = log n := ofReal_natCast n ▸ ofReal_log n.cast_nonneg


@[simp]
lemma ofNat_log {n : ℕ} [n.AtLeastTwo] :
    Real.log (no_index (OfNat.ofNat n)) = log (OfNat.ofNat n) :=
  natCast_log


                                                                    /-
                                                                      x : Real
                                                                      ⊢ Eq (Complex.log ↑x).re (Real.log x)
                                                                    -/
theorem log_ofReal_re (x : ℝ) : (log (x : ℂ)).re = Real.log x := by simp [log_re]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem log_ofReal_mul {r : ℝ} (hr : 0 < r) {x : ℂ} (hx : x ≠ 0) :
    log (r * x) = Real.log r + log x := by
  /-
    r : Real
    hr : LT.lt 0 r
    x : Complex
    hx : Ne x 0
    ⊢ Eq (Complex.log (HMul.hMul (↑r) x)) (HAdd.hAdd (↑(Real.log r)) (Complex.log  …
  -/
  replace hx := Complex.abs.ne_zero_iff.mpr hx
  simp_rw [log, map_mul, abs_ofReal, arg_real_mul _ hr, abs_of_pos hr, Real.log_mul hr.ne' hx,
    ofReal_add, add_assoc]


theorem log_mul_ofReal (r : ℝ) (hr : 0 < r) (x : ℂ) (hx : x ≠ 0) :
                                           /-
                                             r : Real
                                             hr : LT.lt 0 r
                                             x : Complex
                                             hx : Ne x 0
                                             ⊢ Eq (Complex.log (HMul.hMul x ↑r)) (HAdd.hAdd (↑(Real.log r)) (Complex.log x))
                                           -/
    log (x * r) = Real.log r + log x := by rw [mul_comm, log_ofReal_mul hr hx]
                                           /-
                                             🎉 no goals
                                           -/


lemma log_mul_eq_add_log_iff {x y : ℂ} (hx₀ : x ≠ 0) (hy₀ : y ≠ 0) :
    log (x * y) = log x + log y ↔ arg x + arg y ∈ Set.Ioc (-π) π := by
  /-
    x y : Complex
    hx₀ : Ne x 0
    hy₀ : Ne y 0
    ⊢ Iff (Eq (Complex.log (HMul.hMul x y)) (HAdd.hAdd (Complex.log x) (Complex.lo …
  -/
  refine Complex.ext_iff.trans <| Iff.trans ?_ <| arg_mul_eq_add_arg_iff hx₀ hy₀
  simp_rw [add_re, add_im, log_re, log_im, AbsoluteValue.map_mul,
    Real.log_mul (abs.ne_zero hx₀) (abs.ne_zero hy₀), true_and]


alias ⟨_, log_mul⟩ := log_mul_eq_add_log_iff


@[simp]
                                   /-
                                     ⊢ Eq (Complex.log 0) 0
                                   -/
theorem log_zero : log 0 = 0 := by simp [log]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
                                  /-
                                    ⊢ Eq (Complex.log 1) 0
                                  -/
theorem log_one : log 1 = 0 := by simp [log]
                                  /-
                                    🎉 no goals
                                  -/


                                             /-
                                               ⊢ Eq (Complex.log (-1)) (HMul.hMul (↑Real.pi) Complex.I)
                                             -/
theorem log_neg_one : log (-1) = π * I := by simp [log]
                                             /-
                                               🎉 no goals
                                             -/


                                        /-
                                          ⊢ Eq (Complex.log Complex.I) (HMul.hMul (HDiv.hDiv (↑Real.pi) 2) Complex.I)
                                        -/
theorem log_I : log I = π / 2 * I := by simp [log]
                                        /-
                                          🎉 no goals
                                        -/


                                                  /-
                                                    ⊢ Eq (Complex.log (Neg.neg Complex.I)) (HMul.hMul (Neg.neg (HDiv.hDiv (↑Real.p …
                                                  -/
theorem log_neg_I : log (-I) = -(π / 2) * I := by simp [log]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem log_conj_eq_ite (x : ℂ) : log (conj x) = if x.arg = π then log x else conj (log x) := by
  /-
    x : Complex
    ⊢ Eq (Complex.log ((starRingEnd Complex) x)) (ite (Eq x.arg Real.pi) (Complex. …
  -/
  simp_rw [log, abs_conj, arg_conj, map_add, map_mul, conj_ofReal]
  /-
    x : Complex
    ⊢ Eq (HAdd.hAdd (↑(Real.log (Complex.abs x))) (HMul.hMul (↑(ite (Eq x.arg Real …
  -/
  split_ifs with hx
    /-
      case pos
      x : Complex
      hx : Eq x.arg Real.pi
      ⊢ Eq (HAdd.hAdd (↑(Real.log (Complex.abs x))) (HMul.hMul (↑Real.pi) Complex.I) …
    -/
  · rw [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Complex
    hx : Not (Eq x.arg Real.pi)
    ⊢ Eq (HAdd.hAdd (↑(Real.log (Complex.abs x))) (HMul.hMul (↑(Neg.neg x.arg)) Co …
  -/
  simp_rw [ofReal_neg, conj_I, mul_neg, neg_mul]
  /-
    🎉 no goals
  -/


theorem log_conj (x : ℂ) (h : x.arg ≠ π) : log (conj x) = conj (log x) := by
  /-
    x : Complex
    h : Ne x.arg Real.pi
    ⊢ Eq (Complex.log ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex.l …
  -/
  rw [log_conj_eq_ite, if_neg h]
  /-
    🎉 no goals
  -/


theorem log_inv_eq_ite (x : ℂ) : log x⁻¹ = if x.arg = π then -conj (log x) else -log x := by
  /-
    x : Complex
    ⊢ Eq (Complex.log (Inv.inv x)) (ite (Eq x.arg Real.pi) (Neg.neg ((starRingEnd  …
  -/
  by_cases hx : x = 0
    /-
      case pos
      x : Complex
      hx : Eq x 0
      ⊢ Eq (Complex.log (Inv.inv x)) (ite (Eq x.arg Real.pi) (Neg.neg ((starRingEnd  …
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Complex
    hx : Not (Eq x 0)
    ⊢ Eq (Complex.log (Inv.inv x)) (ite (Eq x.arg Real.pi) (Neg.neg ((starRingEnd  …
  -/
  rw [inv_def, log_mul_ofReal, Real.log_inv, ofReal_neg, ← sub_eq_neg_add, log_conj_eq_ite]
  · simp_rw [log, map_add, map_mul, conj_ofReal, conj_I, normSq_eq_abs, Real.log_pow,
      Nat.cast_two, ofReal_mul, neg_add, mul_neg, neg_neg]
    /-
      case neg
      x : Complex
      hx : Not (Eq x 0)
      ⊢ Eq (HSub.hSub (ite (Eq x.arg Real.pi) (HAdd.hAdd (↑(Real.log (Complex.abs x) …
    -/
    norm_num; rw [two_mul] -- Porting note: added to simplify `↑2`
    /-
      case neg
      x : Complex
      hx : Not (Eq x 0)
      ⊢ Eq (HSub.hSub (ite (Eq x.arg Real.pi) (HAdd.hAdd (↑(Real.log (Complex.abs x) …
    -/
    split_ifs
      /-
        case pos
        x : Complex
        hx : Not (Eq x 0)
        h✝ : Eq x.arg Real.pi
        ⊢ Eq (HSub.hSub (HAdd.hAdd (↑(Real.log (Complex.abs x))) (HMul.hMul (↑x.arg) C …
      -/
    · rw [add_sub_right_comm, sub_add_cancel_left]
      /-
        🎉 no goals
      -/
      /-
        case neg
        x : Complex
        hx : Not (Eq x 0)
        h✝ : Not (Eq x.arg Real.pi)
        ⊢ Eq (HSub.hSub (HAdd.hAdd (↑(Real.log (Complex.abs x))) (Neg.neg (HMul.hMul ( …
      -/
    · rw [add_sub_right_comm, sub_add_cancel_left]
      /-
        🎉 no goals
      -/
    /-
      case neg.hr
      x : Complex
      hx : Not (Eq x 0)
      ⊢ LT.lt 0 (Inv.inv (Complex.normSq x))
    -/
  · rwa [inv_pos, Complex.normSq_pos]
    /-
      🎉 no goals
    -/
    /-
      case neg.hx
      x : Complex
      hx : Not (Eq x 0)
      ⊢ Ne ((starRingEnd Complex) x) 0
    -/
  · rwa [map_ne_zero]
    /-
      🎉 no goals
    -/


                                                                  /-
                                                                    x : Complex
                                                                    hx : Ne x.arg Real.pi
                                                                    ⊢ Eq (Complex.log (Inv.inv x)) (Neg.neg (Complex.log x))
                                                                  -/
theorem log_inv (x : ℂ) (hx : x.arg ≠ π) : log x⁻¹ = -log x := by rw [log_inv_eq_ite, if_neg hx]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                     /-
                                                       ⊢ Ne (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0
                                                     -/
theorem two_pi_I_ne_zero : (2 * π * I : ℂ) ≠ 0 := by norm_num [Real.pi_ne_zero, I_ne_zero]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem exp_eq_one_iff {x : ℂ} : exp x = 1 ↔ ∃ n : ℤ, x = n * (2 * π * I) := by
  /-
    x : Complex
    ⊢ Iff (Eq (Complex.exp x) 1) (Exists fun n => Eq x (HMul.hMul (↑n) (HMul.hMul  …
  -/
  constructor
    /-
      case mp
      x : Complex
      ⊢ Eq (Complex.exp x) 1 → Exists fun n => Eq x (HMul.hMul (↑n) (HMul.hMul (HMul …
    -/
  · intro h
    /-
      case mp
      x : Complex
      h : Eq (Complex.exp x) 1
      ⊢ Exists fun n => Eq x (HMul.hMul (↑n) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Compl …
    -/
    rcases existsUnique_add_zsmul_mem_Ioc Real.two_pi_pos x.im (-π) with ⟨n, hn, -⟩
    /-
      case mp.intro.intro
      x : Complex
      h : Eq (Complex.exp x) 1
      n : Int
      hn : Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi) (H …
      ⊢ Exists fun n => Eq x (HMul.hMul (↑n) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Compl …
    -/
    use -n
    /-
      case h
      x : Complex
      h : Eq (Complex.exp x) 1
      n : Int
      hn : Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi) (H …
      ⊢ Eq x (HMul.hMul (↑(Neg.neg n)) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I))
    -/
    rw [Int.cast_neg, neg_mul, eq_neg_iff_add_eq_zero]
    /-
      case h
      x : Complex
      h : Eq (Complex.exp x) 1
      n : Int
      hn : Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi) (H …
      ⊢ Eq (HAdd.hAdd x (HMul.hMul (↑n) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) …
    -/
    have : (x + n * (2 * π * I)).im ∈ Set.Ioc (-π) π := by simpa [two_mul, mul_add] using hn
    /-
      case h
      x : Complex
      h : Eq (Complex.exp x) 1
      n : Int
      hn : Membership.mem (Set.Ioc (Neg.neg Real.pi) (HAdd.hAdd (Neg.neg Real.pi) (H …
      this : Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd x (HMul.h …
      ⊢ Eq (HAdd.hAdd x (HMul.hMul (↑n) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) …
    -/
    rw [← log_exp this.1 this.2, exp_periodic.int_mul n, h, log_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : Complex
      ⊢ (Exists fun n => Eq x (HMul.hMul (↑n) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comp …
    -/
  · rintro ⟨n, rfl⟩
    /-
      case mpr.intro
      n : Int
      ⊢ Eq (Complex.exp (HMul.hMul (↑n) (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) …
    -/
    exact (exp_periodic.int_mul n).eq.trans exp_zero
    /-
      🎉 no goals
    -/


theorem exp_eq_exp_iff_exp_sub_eq_one {x y : ℂ} : exp x = exp y ↔ exp (x - y) = 1 := by
  /-
    x y : Complex
    ⊢ Iff (Eq (Complex.exp x) (Complex.exp y)) (Eq (Complex.exp (HSub.hSub x y)) 1)
  -/
  rw [exp_sub, div_eq_one_iff_eq (exp_ne_zero _)]
  /-
    🎉 no goals
  -/


theorem exp_eq_exp_iff_exists_int {x y : ℂ} : exp x = exp y ↔ ∃ n : ℤ, x = y + n * (2 * π * I) := by
  /-
    x y : Complex
    ⊢ Iff (Eq (Complex.exp x) (Complex.exp y)) (Exists fun n => Eq x (HAdd.hAdd y  …
  -/
  simp only [exp_eq_exp_iff_exp_sub_eq_one, exp_eq_one_iff, sub_eq_iff_eq_add']
  /-
    🎉 no goals
  -/


theorem log_exp_exists (z : ℂ) :
    ∃ n : ℤ, log (exp z) = z + n * (2 * π * I) := by
  /-
    z : Complex
    ⊢ Exists fun n => Eq (Complex.log (Complex.exp z)) (HAdd.hAdd z (HMul.hMul (↑n …
  -/
  rw [← exp_eq_exp_iff_exists_int, exp_log]
  /-
    z : Complex
    ⊢ Ne (Complex.exp z) 0
  -/
  exact exp_ne_zero z
  /-
    🎉 no goals
  -/


@[simp]
theorem countable_preimage_exp {s : Set ℂ} : (exp ⁻¹' s).Countable ↔ s.Countable := by
  /-
    s : Set Complex
    ⊢ Iff (Set.preimage Complex.exp s).Countable s.Countable
  -/
  refine ⟨fun hs => ?_, fun hs => ?_⟩
    /-
      case refine_1
      s : Set Complex
      hs : (Set.preimage Complex.exp s).Countable
      ⊢ s.Countable
    -/
  · refine ((hs.image exp).insert 0).mono ?_
    rw [Set.image_preimage_eq_inter_range, range_exp, ← Set.diff_eq, ← Set.union_singleton,
        Set.diff_union_self]
    /-
      case refine_1
      s : Set Complex
      hs : (Set.preimage Complex.exp s).Countable
      ⊢ HasSubset.Subset s (Union.union s (Singleton.singleton 0))
    -/
    exact Set.subset_union_left
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      s : Set Complex
      hs : s.Countable
      ⊢ (Set.preimage Complex.exp s).Countable
    -/
  · rw [← Set.biUnion_preimage_singleton]
    /-
      case refine_2
      s : Set Complex
      hs : s.Countable
      ⊢ (Set.iUnion fun y => Set.iUnion fun h => Set.preimage Complex.exp (Singleton …
    -/
    refine hs.biUnion fun z hz => ?_
    /-
      case refine_2
      s : Set Complex
      hs : s.Countable
      z : Complex
      hz : Membership.mem s z
      ⊢ (Set.preimage Complex.exp (Singleton.singleton z)).Countable
    -/
    rcases em (∃ w, exp w = z) with (⟨w, rfl⟩ | hne)
      /-
        case refine_2.inl.intro
        s : Set Complex
        hs : s.Countable
        w : Complex
        hz : Membership.mem s (Complex.exp w)
        ⊢ (Set.preimage Complex.exp (Singleton.singleton (Complex.exp w))).Countable
      -/
    · simp only [Set.preimage, Set.mem_singleton_iff, exp_eq_exp_iff_exists_int, Set.setOf_exists]
      /-
        case refine_2.inl.intro
        s : Set Complex
        hs : s.Countable
        w : Complex
        hz : Membership.mem s (Complex.exp w)
        ⊢ (Set.iUnion fun i => setOf fun x => Eq x (HAdd.hAdd w (HMul.hMul (↑i) (HMul. …
      -/
      exact Set.countable_iUnion fun m => Set.countable_singleton _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        s : Set Complex
        hs : s.Countable
        z : Complex
        hz : Membership.mem s z
        hne : Not (Exists fun w => Eq (Complex.exp w) z)
        ⊢ (Set.preimage Complex.exp (Singleton.singleton z)).Countable
      -/
    · push_neg at hne
      /-
        case refine_2.inr
        s : Set Complex
        hs : s.Countable
        z : Complex
        hz : Membership.mem s z
        hne : ∀ (w : Complex), Ne (Complex.exp w) z
        ⊢ (Set.preimage Complex.exp (Singleton.singleton z)).Countable
      -/
      simp [Set.preimage, hne]
      /-
        🎉 no goals
      -/


alias ⟨_, _root_.Set.Countable.preimage_cexp⟩ := countable_preimage_exp


theorem tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero {z : ℂ} (hre : z.re < 0)
    (him : z.im = 0) : Tendsto log (𝓝[{ z : ℂ | z.im < 0 }] z) (𝓝 <| Real.log (abs z) - π * I) := by
  convert
    (continuous_ofReal.continuousAt.comp_continuousWithinAt
            (continuous_abs.continuousWithinAt.log _)).tendsto.add
      (((continuous_ofReal.tendsto _).comp <|
            tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him).mul
        tendsto_const_nhds) using 1
    /-
      case h.e'_5
      z : Complex
      hre : LT.lt z.re 0
      him : Eq z.im 0
      ⊢ Eq (nhds (HSub.hSub (↑(Real.log (Complex.abs z))) (HMul.hMul (↑Real.pi) Comp …
    -/
  · simp [sub_eq_add_neg]
    /-
      🎉 no goals
    -/
    /-
      case convert_1
      z : Complex
      hre : LT.lt z.re 0
      him : Eq z.im 0
      ⊢ Ne (Complex.abs z) 0
    -/
  · lift z to ℝ using him
    /-
      case convert_1.intro
      z : Real
      hre : LT.lt (↑z).re 0
      ⊢ Ne (Complex.abs ↑z) 0
    -/
    simpa using hre.ne
    /-
      🎉 no goals
    -/


theorem continuousWithinAt_log_of_re_neg_of_im_zero {z : ℂ} (hre : z.re < 0) (him : z.im = 0) :
    ContinuousWithinAt log { z : ℂ | 0 ≤ z.im } z := by
  convert
    (continuous_ofReal.continuousAt.comp_continuousWithinAt
            (continuous_abs.continuousWithinAt.log _)).tendsto.add
      ((continuous_ofReal.continuousAt.comp_continuousWithinAt <|
            continuousWithinAt_arg_of_re_neg_of_im_zero hre him).mul
        tendsto_const_nhds) using 1
  /-
    case convert_1
    z : Complex
    hre : LT.lt z.re 0
    him : Eq z.im 0
    ⊢ Ne (Complex.abs z) 0
  -/
  lift z to ℝ using him
  /-
    case convert_1.intro
    z : Real
    hre : LT.lt (↑z).re 0
    ⊢ Ne (Complex.abs ↑z) 0
  -/
  simpa using hre.ne
  /-
    🎉 no goals
  -/


theorem tendsto_log_nhdsWithin_im_nonneg_of_re_neg_of_im_zero {z : ℂ} (hre : z.re < 0)
    (him : z.im = 0) : Tendsto log (𝓝[{ z : ℂ | 0 ≤ z.im }] z) (𝓝 <| Real.log (abs z) + π * I) := by
  simpa only [log, arg_eq_pi_iff.2 ⟨hre, him⟩] using
    (continuousWithinAt_log_of_re_neg_of_im_zero hre him).tendsto


@[simp]
theorem map_exp_comap_re_atBot : map exp (comap re atBot) = 𝓝[≠] 0 := by
  /-
    ⊢ Eq (Filter.map Complex.exp (Filter.comap Complex.re Filter.atBot)) (nhdsWith …
  -/
  rw [← comap_exp_nhds_zero, map_comap, range_exp, nhdsWithin]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_exp_comap_re_atTop : map exp (comap re atTop) = cobounded ℂ := by
  /-
    ⊢ Eq (Filter.map Complex.exp (Filter.comap Complex.re Filter.atTop)) (Bornolog …
  -/
  rw [← comap_exp_cobounded, map_comap, range_exp, inf_eq_left, le_principal_iff]
  /-
    ⊢ Membership.mem (Bornology.cobounded Complex) (HasCompl.compl (Singleton.sing …
  -/
  exact eventually_ne_cobounded _
  /-
    🎉 no goals
  -/


theorem continuousAt_clog {x : ℂ} (h : x ∈ slitPlane) : ContinuousAt log x := by
  /-
    x : Complex
    h : Membership.mem Complex.slitPlane x
    ⊢ ContinuousAt Complex.log x
  -/
  refine ContinuousAt.add ?_ ?_
    /-
      case refine_1
      x : Complex
      h : Membership.mem Complex.slitPlane x
      ⊢ ContinuousAt (fun x => ↑(Real.log (Complex.abs x))) x
    -/
  · refine continuous_ofReal.continuousAt.comp ?_
    /-
      case refine_1
      x : Complex
      h : Membership.mem Complex.slitPlane x
      ⊢ ContinuousAt (fun x => Real.log (Complex.abs x)) x
    -/
    refine (Real.continuousAt_log ?_).comp Complex.continuous_abs.continuousAt
    /-
      case refine_1
      x : Complex
      h : Membership.mem Complex.slitPlane x
      ⊢ Ne (Complex.abs x) 0
    -/
    exact Complex.abs.ne_zero_iff.mpr <| slitPlane_ne_zero h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x : Complex
      h : Membership.mem Complex.slitPlane x
      ⊢ ContinuousAt (fun x => HMul.hMul (↑x.arg) Complex.I) x
    -/
  · have h_cont_mul : Continuous fun x : ℂ => x * I := continuous_id'.mul continuous_const
    /-
      case refine_2
      x : Complex
      h : Membership.mem Complex.slitPlane x
      h_cont_mul : Continuous fun x => HMul.hMul x Complex.I
      ⊢ ContinuousAt (fun x => HMul.hMul (↑x.arg) Complex.I) x
    -/
    refine h_cont_mul.continuousAt.comp (continuous_ofReal.continuousAt.comp ?_)
    /-
      case refine_2
      x : Complex
      h : Membership.mem Complex.slitPlane x
      h_cont_mul : Continuous fun x => HMul.hMul x Complex.I
      ⊢ ContinuousAt Complex.arg x
    -/
    exact continuousAt_arg h
    /-
      🎉 no goals
    -/


theorem _root_.Filter.Tendsto.clog {l : Filter α} {f : α → ℂ} {x : ℂ} (h : Tendsto f l (𝓝 x))
    (hx : x ∈ slitPlane) : Tendsto (fun t => log (f t)) l (𝓝 <| log x) :=
  (continuousAt_clog hx).tendsto.comp h


nonrec
theorem _root_.ContinuousAt.clog {f : α → ℂ} {x : α} (h₁ : ContinuousAt f x)
    (h₂ : f x ∈ slitPlane) : ContinuousAt (fun t => log (f t)) x :=
  h₁.clog h₂


nonrec
theorem _root_.ContinuousWithinAt.clog {f : α → ℂ} {s : Set α} {x : α}
    (h₁ : ContinuousWithinAt f s x) (h₂ : f x ∈ slitPlane) :
    ContinuousWithinAt (fun t => log (f t)) s x :=
  h₁.clog h₂


nonrec
theorem _root_.ContinuousOn.clog {f : α → ℂ} {s : Set α} (h₁ : ContinuousOn f s)
    (h₂ : ∀ x ∈ s, f x ∈ slitPlane) : ContinuousOn (fun t => log (f t)) s := fun x hx =>
  (h₁ x hx).clog (h₂ x hx)


nonrec
theorem _root_.Continuous.clog {f : α → ℂ} (h₁ : Continuous f)
    (h₂ : ∀ x, f x ∈ slitPlane) : Continuous fun t => log (f t) :=
  continuous_iff_continuousAt.2 fun x => h₁.continuousAt.clog (h₂ x)


lemma Real.HasSum_rexp_HasProd (f : ι → α → ℝ) (hfn : ∀ x n, 0 < f n x)
    (hf : ∀ x : α, HasSum (fun n => log (f n x)) (∑' i, log (f i x))) (a : α) :
       HasProd (fun b ↦ f b a) (∏' n : ι, (f n a)) := by
  have : HasProd (fun b ↦ f b a) ((rexp ∘ fun a ↦ ∑' (n : ι), log (f n a)) a) := by
    apply ((hf a).rexp).congr
    intro _
    congr
    exact funext fun x ↦ exp_log (hfn a x)
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), HasSum (fun n => Real.log (f n x)) (tsum fun i => Real.log (f  …
    a : α
    this : HasProd (fun b => f b a) (Function.comp Real.exp (fun a => tsum fun n = …
    ⊢ HasProd (fun b => f b a) (tprod fun n => f n a)
  -/
  rwa [HasProd.tprod_eq this]
  /-
    🎉 no goals
  -/



/--The exponential of a infinite sum of real logs (which converges absolutely) is an infinite
product.-/
lemma Real.rexp_tsum_eq_tprod (f : ι → α → ℝ) (hfn : ∀ x n, 0 < f n x)
    (hf : ∀ x : α, Summable fun n => log ((f n x))) :
    (rexp ∘ (fun a : α => (∑' n : ι, log (f n a)))) = (fun a : α => ∏' n : ι, (f n a)) := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    ⊢ Eq (Function.comp Real.exp fun a => tsum fun n => Real.log (f n a)) fun a => …
  -/
  ext a
  /-
    case h
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    ⊢ Eq (Function.comp Real.exp (fun a => tsum fun n => Real.log (f n a)) a) (tpr …
  -/
  apply (HasProd.tprod_eq ?_).symm
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    ⊢ HasProd (fun b => f b a) (Function.comp Real.exp (fun a => tsum fun n => Rea …
  -/
  apply ((hf a).hasSum.rexp).congr
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    ⊢ ∀ (x : Finset ι), Eq (x.prod fun b => Function.comp Real.exp (fun n => Real. …
  -/
  intro _
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    x✝ : Finset ι
    ⊢ Eq (x✝.prod fun b => Function.comp Real.exp (fun n => Real.log (f n a)) b) ( …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    x✝ : Finset ι
    ⊢ Eq (fun b => Function.comp Real.exp (fun n => Real.log (f n a)) b) fun b =>  …
  -/
  exact funext fun x ↦ exp_log (hfn a x)
  /-
    🎉 no goals
  -/


lemma Real.summable_cexp_multipliable (f : ι → α → ℝ) (hfn : ∀ x n, 0 < f n x)
    (hf : ∀ x : α, Summable fun n => log (f n x)) (a : α) : Multipliable fun b ↦ f b a := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    ⊢ Multipliable fun b => f b a
  -/
  have := (Real.HasSum_rexp_HasProd f hfn fun a => (hf a).hasSum) a
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Real
    hfn : ∀ (x : α) (n : ι), LT.lt 0 (f n x)
    hf : ∀ (x : α), Summable fun n => Real.log (f n x)
    a : α
    this : HasProd (fun b => f b a) (tprod fun n => f n a)
    ⊢ Multipliable fun b => f b a
  -/
  use (∏' n : ι, (f n a))
  /-
    🎉 no goals
  -/


lemma Complex.HasSum_cexp_HasProd (f : ι → α → ℂ) (hfn : ∀ x n, f n x ≠ 0)
    (hf : ∀ x : α, HasSum (fun n => log (f n x)) (∑' i, log (f i x))) (a : α) :
    HasProd (fun b ↦ f b a) (∏' n : ι, (f n a)) := by
  have : HasProd (fun b ↦ f b a) ((cexp ∘ fun a ↦ ∑' (n : ι), log (f n a)) a) := by
    apply ((hf a).cexp).congr
    intro _
    congr
    exact funext fun x ↦ exp_log (hfn a x)
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), HasSum (fun n => Complex.log (f n x)) (tsum fun i => Complex.l …
    a : α
    this : HasProd (fun b => f b a) (Function.comp Complex.exp (fun a => tsum fun  …
    ⊢ HasProd (fun b => f b a) (tprod fun n => f n a)
  -/
  rwa [HasProd.tprod_eq this]
  /-
    🎉 no goals
  -/


lemma Complex.summable_cexp_multipliable (f : ι → α → ℂ) (hfn : ∀ x n, f n x ≠ 0)
    (hf : ∀ x : α, Summable fun n => log (f n x)) (a : α) :
    Multipliable fun b ↦ f b a := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    ⊢ Multipliable fun b => f b a
  -/
  have := (Complex.HasSum_cexp_HasProd f hfn fun a => (hf a).hasSum) a
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    this : HasProd (fun b => f b a) (tprod fun n => f n a)
    ⊢ Multipliable fun b => f b a
  -/
  use (∏' n : ι, (f n a))
  /-
    🎉 no goals
  -/


/--The exponential of a infinite sum of comples logs (which converges absolutely) is an infinite
product.-/
lemma Complex.cexp_tsum_eq_tprod (f : ι → α → ℂ) (hfn : ∀ x n, f n x ≠ 0)
    (hf : ∀ x : α, Summable fun n => log (f n x)) :
    (cexp ∘ (fun a : α => (∑' n : ι, log (f n a)))) = fun a : α => ∏' n : ι, f n a := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    ⊢ Eq (Function.comp Complex.exp fun a => tsum fun n => Complex.log (f n a)) fu …
  -/
  ext a
  /-
    case h
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    ⊢ Eq (Function.comp Complex.exp (fun a => tsum fun n => Complex.log (f n a)) a …
  -/
  apply (HasProd.tprod_eq ?_).symm
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    ⊢ HasProd (fun b => f b a) (Function.comp Complex.exp (fun a => tsum fun n =>  …
  -/
  apply ((hf a).hasSum.cexp).congr
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    ⊢ ∀ (x : Finset ι), Eq (x.prod fun b => Function.comp Complex.exp (fun n => Co …
  -/
  intro _
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    x✝ : Finset ι
    ⊢ Eq (x✝.prod fun b => Function.comp Complex.exp (fun n => Complex.log (f n a) …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    hfn : ∀ (x : α) (n : ι), Ne (f n x) 0
    hf : ∀ (x : α), Summable fun n => Complex.log (f n x)
    a : α
    x✝ : Finset ι
    ⊢ Eq (fun b => Function.comp Complex.exp (fun n => Complex.log (f n a)) b) fun …
  -/
  exact funext fun x ↦ exp_log (hfn a x)
  /-
    🎉 no goals
  -/


