lemma norm_eq_max_natAbs (x : Fin 2 → ℤ) : ‖x‖ = max (x 0).natAbs (x 1).natAbs := by
  /-
    x : Fin 2 → Int
    ⊢ Eq (Norm.norm x) ↑(Max.max (x 0).natAbs (x 1).natAbs)
  -/
  rw [← coe_nnnorm, ← NNReal.coe_natCast, NNReal.coe_inj, Nat.cast_max]
  /-
    x : Fin 2 → Int
    ⊢ Eq (NNNorm.nnnorm x) (Max.max ↑(x 0).natAbs ↑(x 1).natAbs)
  -/
  refine eq_of_forall_ge_iff fun c ↦ ?_
  /-
    x : Fin 2 → Int
    c : NNReal
    ⊢ Iff (LE.le (NNNorm.nnnorm x) c) (LE.le (Max.max ↑(x 0).natAbs ↑(x 1).natAbs) …
  -/
  simp only [pi_nnnorm_le_iff, Fin.forall_fin_two, max_le_iff, NNReal.natCast_natAbs]
  /-
    🎉 no goals
  -/


/-- Auxiliary function used for bounding Eisenstein series, defined as
  `z.im ^ 2 / (z.re ^ 2 + z.im ^ 2)`. -/
def r1 : ℝ := z.im ^ 2 / (z.re ^ 2 + z.im ^ 2)


lemma r1_eq : r1 z = 1 / ((z.re / z.im) ^ 2 + 1) := by
  /-
    z : UpperHalfPlane
    ⊢ Eq (EisensteinSeries.r1 z) (HDiv.hDiv 1 (HAdd.hAdd (HPow.hPow (HDiv.hDiv z.r …
  -/
  rw [div_pow, div_add_one (by positivity), one_div_div, r1]
  /-
    🎉 no goals
  -/


lemma r1_pos : 0 < r1 z := by
  /-
    z : UpperHalfPlane
    ⊢ LT.lt 0 (EisensteinSeries.r1 z)
  -/
  dsimp only [r1]
  /-
    z : UpperHalfPlane
    ⊢ LT.lt 0 (HDiv.hDiv (HPow.hPow z.im 2) (HAdd.hAdd (HPow.hPow z.re 2) (HPow.hP …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- For `c, d ∈ ℝ` with `1 ≤ d ^ 2`, we have `r1 z ≤ |c * z + d| ^ 2`. -/
lemma r1_aux_bound (c : ℝ) {d : ℝ} (hd : 1 ≤ d ^ 2) :
    r1 z ≤ (c * z.re + d) ^ 2 + (c * z.im) ^ 2 := by
  have H1 : (c * z.re + d) ^ 2 + (c * z.im) ^ 2 =
    c ^ 2 * (z.re ^ 2 + z.im ^ 2) + d * 2 * c * z.re + d ^ 2 := by ring
  have H2 : (c ^ 2 * (z.re ^ 2 + z.im ^ 2) + d * 2 * c * z.re + d ^ 2) * (z.re ^ 2 + z.im ^ 2)
    - z.im ^ 2 = (c * (z.re ^ 2 + z.im ^ 2) + d * z.re) ^ 2 + (d ^ 2 - 1) * z.im ^ 2 := by ring
  /-
    z : UpperHalfPlane
    c d : Real
    hd : LE.le 1 (HPow.hPow d 2)
    H1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul c z.re) d) 2) (HPow.hPow ( …
    H2 : Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow c 2) …
    ⊢ LE.le (EisensteinSeries.r1 z) (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul c  …
  -/
  rw [r1, H1, div_le_iff₀ (by positivity), ← sub_nonneg, H2]
  /-
    z : UpperHalfPlane
    c d : Real
    hd : LE.le 1 (HPow.hPow d 2)
    H1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul c z.re) d) 2) (HPow.hPow ( …
    H2 : Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow c 2) …
    ⊢ LE.le 0 (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul c (HAdd.hAdd (HPow.hPow  …
  -/
  exact add_nonneg (sq_nonneg _) (mul_nonneg (sub_nonneg.mpr hd) (sq_nonneg _))
  /-
    🎉 no goals
  -/


/-- This function is used to give an upper bound on the summands in Eisenstein series; it is
defined by `z ↦ min z.im √(z.im ^ 2 / (z.re ^ 2 + z.im ^ 2))`. -/
def r : ℝ := min z.im √(r1 z)


lemma r_pos : 0 < r z := by
  /-
    z : UpperHalfPlane
    ⊢ LT.lt 0 (EisensteinSeries.r z)
  -/
  simp only [r, lt_min_iff, im_pos, Real.sqrt_pos, r1_pos, and_self]
  /-
    🎉 no goals
  -/


lemma r_lower_bound_on_verticalStrip {A B : ℝ} (h : 0 < B) (hz : z ∈ verticalStrip A B) :
    r ⟨⟨A, B⟩, h⟩ ≤ r z := by
  /-
    z : UpperHalfPlane
    A B : Real
    h : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (EisensteinSeries.r ⟨{ re := A, im := B }, h⟩) (EisensteinSeries.r z)
  -/
  apply min_le_min hz.2
  /-
    z : UpperHalfPlane
    A B : Real
    h : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (EisensteinSeries.r1 ⟨{ re := A, im := B }, h⟩).sqrt (EisensteinSeries …
  -/
  rw [Real.sqrt_le_sqrt_iff (by apply (r1_pos z).le)]
  /-
    z : UpperHalfPlane
    A B : Real
    h : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (EisensteinSeries.r1 ⟨{ re := A, im := B }, h⟩) (EisensteinSeries.r1 z)
  -/
  simp only [r1_eq, div_pow, one_div]
  /-
    z : UpperHalfPlane
    A B : Real
    h : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (Inv.inv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (UpperHalfPlane.re ⟨{ re :=  …
  -/
  rw [inv_le_inv₀ (by positivity) (by positivity), add_le_add_iff_right, ← even_two.pow_abs]
  /-
    z : UpperHalfPlane
    A B : Real
    h : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (abs z.re) 2) (HPow.hPow z.im 2)) (HDiv.hDiv (HP …
  -/
  gcongr
  /-
    case hac.hab
    z : UpperHalfPlane
    A B : Real
    h : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (abs z.re) (UpperHalfPlane.re ⟨{ re := A, im := B }, h⟩)
  -/
  exacts [hz.1, hz.2]
  /-
    🎉 no goals
  -/


lemma auxbound1 {c : ℝ} (d : ℝ) (hc : 1 ≤ c ^ 2) : r z ≤ Complex.abs (c * z + d) := by
  /-
    z : UpperHalfPlane
    c d : Real
    hc : LE.le 1 (HPow.hPow c 2)
    ⊢ LE.le (EisensteinSeries.r z) (Complex.abs (HAdd.hAdd (HMul.hMul ↑c ↑z) ↑d))
  -/
  rcases z with ⟨z, hz⟩
  have H1 : z.im ≤ √((c * z.re + d) ^ 2 + (c * z).im ^ 2) := by
    rw [Real.le_sqrt' hz, im_ofReal_mul, mul_pow]
    exact (le_mul_of_one_le_left (sq_nonneg _) hc).trans <| le_add_of_nonneg_left (sq_nonneg _)
  simpa only [r, abs_apply, normSq_apply, add_re, re_ofReal_mul, coe_re, ← pow_two, add_im, mul_im,
    coe_im, ofReal_im, zero_mul, add_zero, min_le_iff] using Or.inl H1


lemma auxbound2 (c : ℝ) {d : ℝ} (hd : 1 ≤ d ^ 2) : r z ≤ Complex.abs (c * z + d) := by
  have H1 : √(r1 z) ≤ √((c * z.re + d) ^ 2 + (c * z.im) ^ 2) :=
    (Real.sqrt_le_sqrt_iff (by positivity)).mpr (r1_aux_bound _ _ hd)
  simpa only [r, abs_apply, normSq_apply, add_re, re_ofReal_mul, coe_re, ofReal_re, ← pow_two,
    add_im, im_ofReal_mul, coe_im, ofReal_im, add_zero, min_le_iff] using Or.inr H1


lemma div_max_sq_ge_one (x : Fin 2 → ℤ) (hx : x ≠ 0) :
    1 ≤ (x 0 / ‖x‖) ^ 2 ∨ 1 ≤ (x 1 / ‖x‖) ^ 2 := by
  /-
    x : Fin 2 → Int
    hx : Ne x 0
    ⊢ Or (LE.le 1 (HPow.hPow (HDiv.hDiv (↑(x 0)) (Norm.norm x)) 2)) (LE.le 1 (HPow …
  -/
  refine (max_choice (x 0).natAbs (x 1).natAbs).imp (fun H0 ↦ ?_) (fun H1 ↦ ?_)
  · have : x 0 ≠ 0 := by
      rwa [← norm_ne_zero_iff, norm_eq_max_natAbs, H0, Nat.cast_ne_zero, Int.natAbs_ne_zero] at hx
    simp only [norm_eq_max_natAbs, H0, Int.cast_natAbs, Int.cast_abs, div_pow, _root_.sq_abs, ne_eq,
      OfNat.ofNat_ne_zero, not_false_eq_true, pow_eq_zero_iff, Int.cast_eq_zero, this, div_self,
      le_refl]
  · have : x 1 ≠ 0 := by
      rwa [← norm_ne_zero_iff, norm_eq_max_natAbs, H1, Nat.cast_ne_zero, Int.natAbs_ne_zero] at hx
    simp only [norm_eq_max_natAbs, H1, Int.cast_natAbs, Int.cast_abs, div_pow, _root_.sq_abs, ne_eq,
      OfNat.ofNat_ne_zero, not_false_eq_true, pow_eq_zero_iff, Int.cast_eq_zero, this, div_self,
      le_refl]


lemma r_mul_max_le {x : Fin 2 → ℤ} (hx : x ≠ 0) : r z * ‖x‖ ≤ Complex.abs (x 0 * z + x 1) := by
  /-
    z : UpperHalfPlane
    x : Fin 2 → Int
    hx : Ne x 0
    ⊢ LE.le (HMul.hMul (EisensteinSeries.r z) (Norm.norm x)) (Complex.abs (HAdd.hA …
  -/
  have hn0 : ‖x‖ ≠ 0 := by rwa [norm_ne_zero_iff]
  have h11 : x 0 * (z : ℂ) + x 1 = (x 0 / ‖x‖ * z + x 1 / ‖x‖) * ‖x‖ := by
    rw [div_mul_eq_mul_div, ← add_div, div_mul_cancel₀ _ (mod_cast hn0)]
  /-
    z : UpperHalfPlane
    x : Fin 2 → Int
    hx : Ne x 0
    hn0 : Ne (Norm.norm x) 0
    h11 : Eq (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1)) (HMul.hMul (HAdd.hAdd (HMul. …
    ⊢ LE.le (HMul.hMul (EisensteinSeries.r z) (Norm.norm x)) (Complex.abs (HAdd.hA …
  -/
  rw [norm_eq_max_natAbs, h11, map_mul, Complex.abs_ofReal, abs_norm, norm_eq_max_natAbs]
  /-
    z : UpperHalfPlane
    x : Fin 2 → Int
    hx : Ne x 0
    hn0 : Ne (Norm.norm x) 0
    h11 : Eq (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1)) (HMul.hMul (HAdd.hAdd (HMul. …
    ⊢ LE.le (HMul.hMul (EisensteinSeries.r z) ↑(Max.max (x 0).natAbs (x 1).natAbs) …
  -/
  gcongr
    /-
      case h
      z : UpperHalfPlane
      x : Fin 2 → Int
      hx : Ne x 0
      hn0 : Ne (Norm.norm x) 0
      h11 : Eq (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1)) (HMul.hMul (HAdd.hAdd (HMul. …
      ⊢ LE.le (EisensteinSeries.r z) (Complex.abs (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑ …
    -/
  · rcases div_max_sq_ge_one x hx with H1 | H2
      /-
        case h.inl
        z : UpperHalfPlane
        x : Fin 2 → Int
        hx : Ne x 0
        hn0 : Ne (Norm.norm x) 0
        h11 : Eq (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1)) (HMul.hMul (HAdd.hAdd (HMul. …
        H1 : LE.le 1 (HPow.hPow (HDiv.hDiv (↑(x 0)) (Norm.norm x)) 2)
        ⊢ LE.le (EisensteinSeries.r z) (Complex.abs (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑ …
      -/
    · simpa only [norm_eq_max_natAbs, ofReal_div, ofReal_intCast] using auxbound1 z (x 1 / ‖x‖) H1
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        z : UpperHalfPlane
        x : Fin 2 → Int
        hx : Ne x 0
        hn0 : Ne (Norm.norm x) 0
        h11 : Eq (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1)) (HMul.hMul (HAdd.hAdd (HMul. …
        H2 : LE.le 1 (HPow.hPow (HDiv.hDiv (↑(x 1)) (Norm.norm x)) 2)
        ⊢ LE.le (EisensteinSeries.r z) (Complex.abs (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑ …
      -/
    · simpa only [norm_eq_max_natAbs, ofReal_div, ofReal_intCast] using auxbound2 z (x 0 / ‖x‖) H2
      /-
        🎉 no goals
      -/


/-- Upper bound for the summand `|c * z + d| ^ (-k)`, as a product of a function of `z` and a
function of `c, d`. -/
lemma summand_bound {k : ℝ} (hk : 0 ≤ k) (x : Fin 2 → ℤ) :
    Complex.abs (x 0 * z + x 1) ^ (-k) ≤ (r z) ^ (-k) * ‖x‖ ^ (-k) := by
  /-
    z : UpperHalfPlane
    k : Real
    hk : LE.le 0 k
    x : Fin 2 → Int
    ⊢ LE.le (HPow.hPow (Complex.abs (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1))) (Neg …
  -/
  by_cases hx : x = 0
    /-
      case pos
      z : UpperHalfPlane
      k : Real
      hk : LE.le 0 k
      x : Fin 2 → Int
      hx : Eq x 0
      ⊢ LE.le (HPow.hPow (Complex.abs (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1))) (Neg …
    -/
  · simp only [hx, Pi.zero_apply, Int.cast_zero, zero_mul, add_zero, ← norm_eq_abs, norm_zero]
    /-
      case pos
      z : UpperHalfPlane
      k : Real
      hk : LE.le 0 k
      x : Fin 2 → Int
      hx : Eq x 0
      ⊢ LE.le (HPow.hPow 0 (Neg.neg k)) (HMul.hMul (HPow.hPow (EisensteinSeries.r z) …
    -/
    by_cases h : -k = 0
      /-
        case pos
        z : UpperHalfPlane
        k : Real
        hk : LE.le 0 k
        x : Fin 2 → Int
        hx : Eq x 0
        h : Eq (Neg.neg k) 0
        ⊢ LE.le (HPow.hPow 0 (Neg.neg k)) (HMul.hMul (HPow.hPow (EisensteinSeries.r z) …
      -/
    · rw [h, Real.rpow_zero, Real.rpow_zero, one_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        z : UpperHalfPlane
        k : Real
        hk : LE.le 0 k
        x : Fin 2 → Int
        hx : Eq x 0
        h : Not (Eq (Neg.neg k) 0)
        ⊢ LE.le (HPow.hPow 0 (Neg.neg k)) (HMul.hMul (HPow.hPow (EisensteinSeries.r z) …
      -/
    · rw [Real.zero_rpow h, mul_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      z : UpperHalfPlane
      k : Real
      hk : LE.le 0 k
      x : Fin 2 → Int
      hx : Not (Eq x 0)
      ⊢ LE.le (HPow.hPow (Complex.abs (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1))) (Neg …
    -/
  · rw [← Real.mul_rpow (r_pos _).le (norm_nonneg _)]
    exact Real.rpow_le_rpow_of_nonpos (mul_pos (r_pos _) (norm_pos_iff.mpr hx)) (r_mul_max_le z hx)
      (neg_nonpos.mpr hk)


variable {z} in
lemma summand_bound_of_mem_verticalStrip {k : ℝ} (hk : 0 ≤ k) (x : Fin 2 → ℤ)
    {A B : ℝ} (hB : 0 < B) (hz : z ∈ verticalStrip A B) :
    Complex.abs (x 0 * z + x 1) ^ (-k) ≤ r ⟨⟨A, B⟩, hB⟩ ^ (-k) * ‖x‖ ^ (-k) := by
  /-
    z : UpperHalfPlane
    k : Real
    hk : LE.le 0 k
    x : Fin 2 → Int
    A B : Real
    hB : LT.lt 0 B
    hz : Membership.mem (UpperHalfPlane.verticalStrip A B) z
    ⊢ LE.le (HPow.hPow (Complex.abs (HAdd.hAdd (HMul.hMul ↑(x 0) ↑z) ↑(x 1))) (Neg …
  -/
  refine (summand_bound z hk x).trans (mul_le_mul_of_nonneg_right ?_ (by positivity))
  exact Real.rpow_le_rpow_of_nonpos (r_pos _) (r_lower_bound_on_verticalStrip z hB hz)
    (neg_nonpos.mpr hk)


/-- The function `ℤ ^ 2 → ℝ` given by `x ↦ ‖x‖ ^ (-k)` is summable if `2 < k`. We prove this by
splitting into boxes using `Finset.box`. -/
lemma summable_one_div_norm_rpow {k : ℝ} (hk : 2 < k) :
    Summable fun (x : Fin 2 → ℤ) ↦ ‖x‖ ^ (-k) := by
  /-
    k : Real
    hk : LT.lt 2 k
    ⊢ Summable fun x => HPow.hPow (Norm.norm x) (Neg.neg k)
  -/
  rw [← (finTwoArrowEquiv _).symm.summable_iff, summable_partition _ Int.existsUnique_mem_box]
    /-
      k : Real
      hk : LT.lt 2 k
      ⊢ And (∀ (j : Nat), Summable fun i => Function.comp (fun x => HPow.hPow (Norm. …
    -/
  · simp only [finTwoArrowEquiv_symm_apply, Function.comp_def]
    /-
      k : Real
      hk : LT.lt 2 k
      ⊢ And (∀ (j : Nat), Summable fun i => HPow.hPow (Norm.norm (Matrix.vecCons (↑i …
    -/
    refine ⟨fun n ↦ (hasSum_fintype (β := box (α := ℤ × ℤ) n) _).summable, ?_⟩
    suffices Summable fun n : ℕ ↦ ∑' (_ : box (α := ℤ × ℤ) n), (n : ℝ) ^ (-k) by
      refine this.congr fun n ↦ tsum_congr fun p ↦ ?_
      simp only [← Int.mem_box.mp p.2, Nat.cast_max, norm_eq_max_natAbs, Matrix.cons_val_zero,
        Matrix.cons_val_one, Matrix.head_cons]
    /-
      k : Real
      hk : LT.lt 2 k
      ⊢ Summable fun n => tsum fun x => HPow.hPow (↑n) (Neg.neg k)
    -/
    simp only [tsum_fintype, univ_eq_attach, sum_const, card_attach, nsmul_eq_mul]
    apply ((Real.summable_nat_rpow.mpr (by linarith : 1 - k < -1)).mul_left
      8).of_norm_bounded_eventually_nat
    /-
      k : Real
      hk : LT.lt 2 k
      ⊢ Filter.Eventually (fun i => LE.le (Norm.norm (HMul.hMul (↑(Finset.box i).car …
    -/
    filter_upwards [Filter.eventually_gt_atTop 0] with n hn
    rw [Int.card_box hn.ne', Real.norm_of_nonneg (by positivity), sub_eq_add_neg,
      Real.rpow_add (Nat.cast_pos.mpr hn), Real.rpow_one, Nat.cast_mul, Nat.cast_ofNat, mul_assoc]
    /-
      k : Real
      hk : LT.lt 2 k
      ⊢ LE.le 0 (Function.comp (fun x => HPow.hPow (Norm.norm x) (Neg.neg k)) ⇑(finT …
    -/
  · exact fun n ↦ Real.rpow_nonneg (norm_nonneg _) _
    /-
      🎉 no goals
    -/


/-- The sum defining the Eisenstein series (of weight `k` and level `Γ(N)` with congruence
condition `a : Fin 2 → ZMod N`) converges locally uniformly on `ℍ`. -/
theorem eisensteinSeries_tendstoLocallyUniformly {k : ℤ} (hk : 3 ≤ k) {N : ℕ} (a : Fin 2 → ZMod N) :
    TendstoLocallyUniformly (fun (s : Finset (gammaSet N a)) ↦ (∑ x ∈ s, eisSummand k x ·))
      (eisensteinSeries a k ·) Filter.atTop := by
  /-
    k : Int
    hk : LE.le 3 k
    N : Nat
    a : Fin 2 → ZMod N
    ⊢ TendstoLocallyUniformly (fun s x => s.sum fun x_1 => EisensteinSeries.eisSum …
  -/
  have hk' : (2 : ℝ) < k := by norm_cast
  have p_sum : Summable fun x : gammaSet N a ↦ ‖x.val‖ ^ (-k) :=
    mod_cast (summable_one_div_norm_rpow hk').subtype (gammaSet N a)
  /-
    k : Int
    hk : LE.le 3 k
    N : Nat
    a : Fin 2 → ZMod N
    hk' : LT.lt 2 ↑k
    p_sum : Summable fun x => HPow.hPow (Norm.norm ↑x) (Neg.neg k)
    ⊢ TendstoLocallyUniformly (fun s x => s.sum fun x_1 => EisensteinSeries.eisSum …
  -/
  simp only [tendstoLocallyUniformly_iff_forall_isCompact, eisensteinSeries]
  /-
    k : Int
    hk : LE.le 3 k
    N : Nat
    a : Fin 2 → ZMod N
    hk' : LT.lt 2 ↑k
    p_sum : Summable fun x => HPow.hPow (Norm.norm ↑x) (Neg.neg k)
    ⊢ ∀ (K : Set UpperHalfPlane), IsCompact K → TendstoUniformlyOn (fun s x => s.s …
  -/
  intro K hK
  /-
    k : Int
    hk : LE.le 3 k
    N : Nat
    a : Fin 2 → ZMod N
    hk' : LT.lt 2 ↑k
    p_sum : Summable fun x => HPow.hPow (Norm.norm ↑x) (Neg.neg k)
    K : Set UpperHalfPlane
    hK : IsCompact K
    ⊢ TendstoUniformlyOn (fun s x => s.sum fun x_1 => EisensteinSeries.eisSummand  …
  -/
  obtain ⟨A, B, hB, HABK⟩ := subset_verticalStrip_of_isCompact hK
  refine (tendstoUniformlyOn_tsum (hu := p_sum.mul_left <| r ⟨⟨A, B⟩, hB⟩ ^ (-k : ℝ))
    (fun p z hz ↦ ?_)).mono HABK
  simpa only [eisSummand, one_div, ← zpow_neg, norm_eq_abs, abs_zpow, ← Real.rpow_intCast,
    Int.cast_neg] using summand_bound_of_mem_verticalStrip (by positivity) p hB hz


/-- Variant of `eisensteinSeries_tendstoLocallyUniformly` formulated with maps `ℂ → ℂ`, which is
nice to have for holomorphicity later. -/
lemma eisensteinSeries_tendstoLocallyUniformlyOn {k : ℤ} {N : ℕ} (hk : 3 ≤ k)
    (a : Fin 2 → ZMod N) : TendstoLocallyUniformlyOn (fun (s : Finset (gammaSet N a )) ↦
      ↑ₕ(fun (z : ℍ) ↦ ∑ x in s, eisSummand k x z )) (↑ₕ(eisensteinSeries_SIF a k).toFun)
          Filter.atTop {z : ℂ | 0 < z.im} := by
  /-
    k : Int
    N : Nat
    hk : LE.le 3 k
    a : Fin 2 → ZMod N
    ⊢ TendstoLocallyUniformlyOn (fun s => Function.comp (fun z => s.sum fun x => E …
  -/
  rw [← Subtype.coe_image_univ {z : ℂ | 0 < z.im}]
  /-
    k : Int
    N : Nat
    hk : LE.le 3 k
    a : Fin 2 → ZMod N
    ⊢ TendstoLocallyUniformlyOn (fun s => Function.comp (fun z => s.sum fun x => E …
  -/
  apply TendstoLocallyUniformlyOn.comp (s := ⊤) _ _ _ (PartialHomeomorph.continuousOn_symm _)
    /-
      k : Int
      N : Nat
      hk : LE.le 3 k
      a : Fin 2 → ZMod N
      ⊢ TendstoLocallyUniformlyOn (fun n z => n.sum fun x => EisensteinSeries.eisSum …
    -/
  · simp only [SlashInvariantForm.toFun_eq_coe, Set.top_eq_univ, tendstoLocallyUniformlyOn_univ]
    /-
      k : Int
      N : Nat
      hk : LE.le 3 k
      a : Fin 2 → ZMod N
      ⊢ TendstoLocallyUniformly (fun n z => n.sum fun x => EisensteinSeries.eisSumma …
    -/
    apply eisensteinSeries_tendstoLocallyUniformly hk
    /-
      🎉 no goals
    -/
  · simp only [IsOpenEmbedding.toPartialHomeomorph_target, Set.top_eq_univ, mapsTo_range_iff,
    Set.mem_univ, forall_const]


