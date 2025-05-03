local notation "abs" z => Real.sqrt (normSq z)


private theorem mul_self_abs (z : ℂ) : ((abs z) * abs z) = normSq z :=
  Real.mul_self_sqrt (normSq_nonneg _)


private theorem abs_nonneg' (z : ℂ) : 0 ≤ abs z :=
  Real.sqrt_nonneg _


                                                      /-
                                                        z : Complex
                                                        ⊢ Eq (Complex.normSq ((starRingEnd Complex) z)).sqrt (Complex.normSq z).sqrt
                                                      -/
theorem abs_conj (z : ℂ) : (abs conj z) = abs z := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


private theorem abs_re_le_abs (z : ℂ) : |z.re| ≤ abs z := by
  /-
    z : Complex
    ⊢ LE.le («abs» z.re) (Complex.normSq z).sqrt
  -/
  rw [mul_self_le_mul_self_iff (abs_nonneg z.re) (abs_nonneg' _), abs_mul_abs_self, mul_self_abs]
  /-
    z : Complex
    ⊢ LE.le (HMul.hMul z.re z.re) (Complex.normSq z)
  -/
  apply re_sq_le_normSq
  /-
    🎉 no goals
  -/


private theorem re_le_abs (z : ℂ) : z.re ≤ abs z :=
  (abs_le.1 (abs_re_le_abs _)).2


private theorem abs_mul (z w : ℂ) : (abs z * w) = (abs z) * abs w := by
  /-
    z w : Complex
    ⊢ Eq (Complex.normSq (HMul.hMul z w)).sqrt (HMul.hMul (Complex.normSq z).sqrt  …
  -/
  rw [normSq_mul, Real.sqrt_mul (normSq_nonneg _)]
  /-
    🎉 no goals
  -/


private theorem abs_add (z w : ℂ) : (abs z + w) ≤ (abs z) + abs w :=
  (mul_self_le_mul_self_iff (abs_nonneg' (z + w))
      (add_nonneg (abs_nonneg' z) (abs_nonneg' w))).2 <| by
    rw [mul_self_abs, add_mul_self_eq, mul_self_abs, mul_self_abs, add_right_comm, normSq_add,
      add_le_add_iff_left, mul_assoc, mul_le_mul_left (zero_lt_two' ℝ), ←
      Real.sqrt_mul <| normSq_nonneg z, ← normSq_conj w, ← map_mul]
    /-
      z w : Complex
      ⊢ LE.le (HMul.hMul z ((starRingEnd Complex) w)).re (Complex.normSq (HMul.hMul  …
    -/
    exact re_le_abs (z * conj w)
    /-
      🎉 no goals
    -/


/-- The complex absolute value function, defined as the square root of the norm squared. -/
noncomputable def _root_.Complex.abs : AbsoluteValue ℂ ℝ where
  toFun x := abs x
  map_mul' := abs_mul
  nonneg' := abs_nonneg'
  eq_zero' _ := (Real.sqrt_eq_zero <| normSq_nonneg _).trans normSq_eq_zero
  add_le' := abs_add


theorem abs_def : (Complex.abs : ℂ → ℝ) = fun z => (normSq z).sqrt :=
  rfl


theorem abs_apply {z : ℂ} : Complex.abs z = (normSq z).sqrt :=
  rfl


@[simp, norm_cast]
theorem abs_ofReal (r : ℝ) : Complex.abs r = |r| := by
  /-
    r : Real
    ⊢ Eq (Complex.abs ↑r) (_root_.abs r)
  -/
  simp [Complex.abs, normSq_ofReal, Real.sqrt_mul_self_eq_abs]
  /-
    🎉 no goals
  -/


nonrec theorem abs_of_nonneg {r : ℝ} (h : 0 ≤ r) : Complex.abs r = r :=
  (Complex.abs_ofReal _).trans (abs_of_nonneg h)

-- Porting note: removed `norm_cast` attribute because the RHS can't start with `↑`

@[simp]
theorem abs_natCast (n : ℕ) : Complex.abs n = n := Complex.abs_of_nonneg (Nat.cast_nonneg n)


@[simp]
theorem abs_ofNat (n : ℕ) [n.AtLeastTwo] :
    Complex.abs ofNat(n) = ofNat(n) :=
  abs_natCast n


theorem mul_self_abs (z : ℂ) : Complex.abs z * Complex.abs z = normSq z :=
  Real.mul_self_sqrt (normSq_nonneg _)


theorem sq_abs (z : ℂ) : Complex.abs z ^ 2 = normSq z :=
  Real.sq_sqrt (normSq_nonneg _)


@[simp]
theorem sq_abs_sub_sq_re (z : ℂ) : Complex.abs z ^ 2 - z.re ^ 2 = z.im ^ 2 := by
  /-
    z : Complex
    ⊢ Eq (HSub.hSub (HPow.hPow (Complex.abs z) 2) (HPow.hPow z.re 2)) (HPow.hPow z …
  -/
  rw [sq_abs, normSq_apply, ← sq, ← sq, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem sq_abs_sub_sq_im (z : ℂ) : Complex.abs z ^ 2 - z.im ^ 2 = z.re ^ 2 := by
  /-
    z : Complex
    ⊢ Eq (HSub.hSub (HPow.hPow (Complex.abs z) 2) (HPow.hPow z.im 2)) (HPow.hPow z …
  -/
  rw [← sq_abs_sub_sq_re, sub_sub_cancel]
  /-
    🎉 no goals
  -/


lemma abs_add_mul_I (x y : ℝ) : abs (x + y * I) = (x ^ 2 + y ^ 2).sqrt := by
  /-
    x y : Real
    ⊢ Eq (Complex.abs (HAdd.hAdd (↑x) (HMul.hMul (↑y) Complex.I))) (HAdd.hAdd (HPo …
  -/
  rw [← normSq_add_mul_I]; rfl
                           /-
                             🎉 no goals
                           -/


lemma abs_eq_sqrt_sq_add_sq (z : ℂ) : abs z = (z.re ^ 2 + z.im ^ 2).sqrt := by
  /-
    z : Complex
    ⊢ Eq (Complex.abs z) (HAdd.hAdd (HPow.hPow z.re 2) (HPow.hPow z.im 2)).sqrt
  -/
  rw [abs_apply, normSq_apply, sq, sq]
  /-
    🎉 no goals
  -/


@[simp]
                                        /-
                                          ⊢ Eq (Complex.abs Complex.I) 1
                                        -/
theorem abs_I : Complex.abs I = 1 := by simp [Complex.abs]
                                        /-
                                          🎉 no goals
                                        -/


theorem abs_two : Complex.abs 2 = 2 := abs_ofNat 2


@[simp]
theorem range_abs : range Complex.abs = Ici 0 :=
  Subset.antisymm
        /-
          ⊢ HasSubset.Subset (Set.range ⇑Complex.abs) (Set.Ici 0)
        -/
    (by simp only [range_subset_iff, Ici, mem_setOf_eq, apply_nonneg, forall_const])
        /-
          🎉 no goals
        -/
    (fun x hx => ⟨x, Complex.abs_of_nonneg hx⟩)


@[simp]
theorem abs_conj (z : ℂ) : Complex.abs (conj z) = Complex.abs z :=
  AbsTheory.abs_conj z


theorem abs_prod {ι : Type*} (s : Finset ι) (f : ι → ℂ) :
    Complex.abs (s.prod f) = s.prod fun I => Complex.abs (f I) :=
  map_prod Complex.abs _ _


theorem abs_pow (z : ℂ) (n : ℕ) : Complex.abs (z ^ n) = Complex.abs z ^ n :=
  map_pow Complex.abs z n


theorem abs_zpow (z : ℂ) (n : ℤ) : Complex.abs (z ^ n) = Complex.abs z ^ n :=
  map_zpow₀ Complex.abs z n


@[bound]
theorem abs_re_le_abs (z : ℂ) : |z.re| ≤ Complex.abs z :=
  Real.abs_le_sqrt <| by
    /-
      z : Complex
      ⊢ LE.le (HPow.hPow z.re 2) (Complex.normSq z)
    -/
    rw [normSq_apply, ← sq]
    /-
      z : Complex
      ⊢ LE.le (HPow.hPow z.re 2) (HAdd.hAdd (HPow.hPow z.re 2) (HMul.hMul z.im z.im))
    -/
    exact le_add_of_nonneg_right (mul_self_nonneg _)
    /-
      🎉 no goals
    -/


@[bound]
theorem abs_im_le_abs (z : ℂ) : |z.im| ≤ Complex.abs z :=
  Real.abs_le_sqrt <| by
    /-
      z : Complex
      ⊢ LE.le (HPow.hPow z.im 2) (Complex.normSq z)
    -/
    rw [normSq_apply, ← sq, ← sq]
    /-
      z : Complex
      ⊢ LE.le (HPow.hPow z.im 2) (HAdd.hAdd (HPow.hPow z.re 2) (HPow.hPow z.im 2))
    -/
    exact le_add_of_nonneg_left (sq_nonneg _)
    /-
      🎉 no goals
    -/


theorem re_le_abs (z : ℂ) : z.re ≤ Complex.abs z :=
  (abs_le.1 (abs_re_le_abs _)).2


theorem im_le_abs (z : ℂ) : z.im ≤ Complex.abs z :=
  (abs_le.1 (abs_im_le_abs _)).2


@[simp]
theorem abs_re_lt_abs {z : ℂ} : |z.re| < Complex.abs z ↔ z.im ≠ 0 := by
  rw [Complex.abs, AbsoluteValue.coe_mk, MulHom.coe_mk, Real.lt_sqrt (abs_nonneg _), normSq_apply,
    _root_.sq_abs, ← sq, lt_add_iff_pos_right, mul_self_pos]


@[simp]
theorem abs_im_lt_abs {z : ℂ} : |z.im| < Complex.abs z ↔ z.re ≠ 0 := by
  /-
    z : Complex
    ⊢ Iff (LT.lt (_root_.abs z.im) (Complex.abs z)) (Ne z.re 0)
  -/
  simpa using @abs_re_lt_abs (z * I)
  /-
    🎉 no goals
  -/


@[simp]
lemma abs_re_eq_abs {z : ℂ} : |z.re| = abs z ↔ z.im = 0 :=
  not_iff_not.1 <| (abs_re_le_abs z).lt_iff_ne.symm.trans abs_re_lt_abs


@[simp]
lemma abs_im_eq_abs {z : ℂ} : |z.im| = abs z ↔ z.re = 0 :=
  not_iff_not.1 <| (abs_im_le_abs z).lt_iff_ne.symm.trans abs_im_lt_abs


@[simp]
theorem abs_abs (z : ℂ) : |Complex.abs z| = Complex.abs z :=
  _root_.abs_of_nonneg (AbsoluteValue.nonneg _ z)

-- Porting note: probably should be golfed

theorem abs_le_abs_re_add_abs_im (z : ℂ) : Complex.abs z ≤ |z.re| + |z.im| := by
  /-
    z : Complex
    ⊢ LE.le (Complex.abs z) (HAdd.hAdd (_root_.abs z.re) (_root_.abs z.im))
  -/
  simpa [re_add_im] using Complex.abs.add_le z.re (z.im * I)
  /-
    🎉 no goals
  -/


theorem abs_le_sqrt_two_mul_max (z : ℂ) : Complex.abs z ≤ Real.sqrt 2 * max |z.re| |z.im| := by
  /-
    z : Complex
    ⊢ LE.le (Complex.abs z) (HMul.hMul (Real.sqrt 2) (Max.max (_root_.abs z.re) (_ …
  -/
  cases' z with x y
  /-
    case mk
    x y : Real
    ⊢ LE.le (Complex.abs { re := x, im := y }) (HMul.hMul (Real.sqrt 2) (Max.max ( …
  -/
  simp only [abs_apply, normSq_mk, ← sq]
  /-
    case mk
    x y : Real
    ⊢ LE.le (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)).sqrt (HMul.hMul (Real.sqrt …
  -/
  by_cases hle : |x| ≤ |y|
  · calc
      Real.sqrt (x ^ 2 + y ^ 2) ≤ Real.sqrt (y ^ 2 + y ^ 2) :=
        Real.sqrt_le_sqrt (add_le_add_right (sq_le_sq.2 hle) _)
      _ = Real.sqrt 2 * max |x| |y| := by
        rw [max_eq_right hle, ← two_mul, Real.sqrt_mul two_pos.le, Real.sqrt_sq_eq_abs]
    /-
      case neg
      x y : Real
      hle : Not (LE.le (_root_.abs x) (_root_.abs y))
      ⊢ LE.le (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)).sqrt (HMul.hMul (Real.sqrt …
    -/
  · have hle' := le_of_not_le hle
    /-
      case neg
      x y : Real
      hle : Not (LE.le (_root_.abs x) (_root_.abs y))
      hle' : LE.le (_root_.abs y) (_root_.abs x)
      ⊢ LE.le (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)).sqrt (HMul.hMul (Real.sqrt …
    -/
    rw [add_comm]
    calc
      Real.sqrt (y ^ 2 + x ^ 2) ≤ Real.sqrt (x ^ 2 + x ^ 2) :=
        Real.sqrt_le_sqrt (add_le_add_right (sq_le_sq.2 hle') _)
      _ = Real.sqrt 2 * max |x| |y| := by
        rw [max_eq_left hle', ← two_mul, Real.sqrt_mul two_pos.le, Real.sqrt_sq_eq_abs]


theorem abs_re_div_abs_le_one (z : ℂ) : |z.re / Complex.abs z| ≤ 1 :=
                        /-
                          z : Complex
                          hz : Eq z 0
                          ⊢ LE.le (_root_.abs (HDiv.hDiv z.re (Complex.abs z))) 1
                        -/
  if hz : z = 0 then by simp [hz, zero_le_one]
                        /-
                          🎉 no goals
                        -/
  else by simp_rw [_root_.abs_div, abs_abs,
    div_le_iff₀ (AbsoluteValue.pos Complex.abs hz), one_mul, abs_re_le_abs]


theorem abs_im_div_abs_le_one (z : ℂ) : |z.im / Complex.abs z| ≤ 1 :=
                        /-
                          z : Complex
                          hz : Eq z 0
                          ⊢ LE.le (_root_.abs (HDiv.hDiv z.im (Complex.abs z))) 1
                        -/
  if hz : z = 0 then by simp [hz, zero_le_one]
                        /-
                          🎉 no goals
                        -/
  else by simp_rw [_root_.abs_div, abs_abs,
    div_le_iff₀ (AbsoluteValue.pos Complex.abs hz), one_mul, abs_im_le_abs]


                                                                  /-
                                                                    n : Int
                                                                    ⊢ Eq (Complex.abs ↑n) (_root_.abs ↑n)
                                                                  -/
@[simp, norm_cast] lemma abs_intCast (n : ℤ) : abs n = |↑n| := by rw [← ofReal_intCast, abs_ofReal]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[deprecated "No deprecation message was provided." (since := "2024-02-14")]
lemma int_cast_abs (n : ℤ) : |↑n| = Complex.abs n := (abs_intCast _).symm


theorem normSq_eq_abs (x : ℂ) : normSq x = (Complex.abs x) ^ 2 := by
  /-
    x : Complex
    ⊢ Eq (Complex.normSq x) (HPow.hPow (Complex.abs x) 2)
  -/
  simp [abs, sq, abs_def, Real.mul_self_sqrt (normSq_nonneg _)]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_normSq : range normSq = Ici 0 :=
  Subset.antisymm (range_subset_iff.2 normSq_nonneg) fun x hx =>
                     /-
                       x : Real
                       hx : Membership.mem (Set.Ici 0) x
                       ⊢ Eq (Complex.normSq ↑x.sqrt) x
                     -/
    ⟨Real.sqrt x, by rw [normSq_ofReal, Real.mul_self_sqrt hx]⟩
                     /-
                       🎉 no goals
                     -/


local notation "abs'" => _root_.abs


theorem isCauSeq_re (f : CauSeq ℂ Complex.abs) : IsCauSeq abs' fun n => (f n).re := fun _ ε0 =>
  (f.cauchy ε0).imp fun i H j ij =>
                       /-
                         f : CauSeq Complex ⇑Complex.abs
                         x✝ : Real
                         ε0 : GT.gt x✝ 0
                         i : Nat
                         H : ∀ (j : Nat), GE.ge j i → LT.lt (Complex.abs (HSub.hSub (↑f j) (↑f i))) x✝
                         j : Nat
                         ij : GE.ge j i
                         ⊢ LE.le (_root_.abs (HSub.hSub ((fun n => (↑f n).re) j) ((fun n => (↑f n).re)  …
                       -/
    lt_of_le_of_lt (by simpa using abs_re_le_abs (f j - f i)) (H _ ij)
                       /-
                         🎉 no goals
                       -/


theorem isCauSeq_im (f : CauSeq ℂ Complex.abs) : IsCauSeq abs' fun n => (f n).im := fun ε ε0 =>
  (f.cauchy ε0).imp fun i H j ij ↦ by
    /-
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      H : ∀ (j : Nat), GE.ge j i → LT.lt (Complex.abs (HSub.hSub (↑f j) (↑f i))) ε
      j : Nat
      ij : GE.ge j i
      ⊢ LT.lt (_root_.abs (HSub.hSub ((fun n => (↑f n).im) j) ((fun n => (↑f n).im)  …
    -/
    simpa only [← ofReal_sub, abs_ofReal, sub_re] using (abs_im_le_abs _).trans_lt <| H _ ij
    /-
      🎉 no goals
    -/


/-- The real part of a complex Cauchy sequence, as a real Cauchy sequence. -/
noncomputable def cauSeqRe (f : CauSeq ℂ Complex.abs) : CauSeq ℝ abs' :=
  ⟨_, isCauSeq_re f⟩


/-- The imaginary part of a complex Cauchy sequence, as a real Cauchy sequence. -/
noncomputable def cauSeqIm (f : CauSeq ℂ Complex.abs) : CauSeq ℝ abs' :=
  ⟨_, isCauSeq_im f⟩


theorem isCauSeq_abs {f : ℕ → ℂ} (hf : IsCauSeq Complex.abs f) :
    IsCauSeq abs' (Complex.abs ∘ f) := fun ε ε0 =>
  let ⟨i, hi⟩ := hf ε ε0
  ⟨i, fun j hj => lt_of_le_of_lt
    (Complex.abs.abs_abv_sub_le_abv_sub _ _) (hi j hj)⟩


/-- The limit of a Cauchy sequence of complex numbers. -/
noncomputable def limAux (f : CauSeq ℂ Complex.abs) : ℂ :=
  ⟨CauSeq.lim (cauSeqRe f), CauSeq.lim (cauSeqIm f)⟩


theorem equiv_limAux (f : CauSeq ℂ Complex.abs) :
    f ≈ CauSeq.const Complex.abs (limAux f) := fun ε ε0 =>
  (exists_forall_ge_and
  (CauSeq.equiv_lim ⟨_, isCauSeq_re f⟩ _ (half_pos ε0))
        (CauSeq.equiv_lim ⟨_, isCauSeq_im f⟩ _ (half_pos ε0))).imp
    fun _ H j ij => by
    /-
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑ …
      j : Nat
      ij : GE.ge j x✝
      ⊢ LT.lt (Complex.abs (↑(HSub.hSub f (CauSeq.const (⇑Complex.abs) (Complex.limA …
    -/
    cases' H _ ij with H₁ H₂
    /-
      case intro
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑ …
      j : Nat
      ij : GE.ge j x✝
      H₁ : LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑f n).re, ⋯⟩ (CauSeq.const _roo …
      H₂ : LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑f n).im, ⋯⟩ (CauSeq.const _roo …
      ⊢ LT.lt (Complex.abs (↑(HSub.hSub f (CauSeq.const (⇑Complex.abs) (Complex.limA …
    -/
    apply lt_of_le_of_lt (abs_le_abs_re_add_abs_im _)
    /-
      case intro
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑ …
      j : Nat
      ij : GE.ge j x✝
      H₁ : LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑f n).re, ⋯⟩ (CauSeq.const _roo …
      H₂ : LT.lt (_root_.abs (↑(HSub.hSub ⟨fun n => (↑f n).im, ⋯⟩ (CauSeq.const _roo …
      ⊢ LT.lt (HAdd.hAdd (_root_.abs (↑(HSub.hSub f (CauSeq.const (⇑Complex.abs) (Co …
    -/
    dsimp [limAux] at *
    /-
      case intro
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (_root_.abs (HSub.hSub (↑f j).re (Cau …
      j : Nat
      ij : GE.ge j x✝
      H₁ : LT.lt (_root_.abs (HSub.hSub (↑f j).re (CauSeq.lim ⟨fun n => (↑f n).re, ⋯ …
      H₂ : LT.lt (_root_.abs (HSub.hSub (↑f j).im (CauSeq.lim ⟨fun n => (↑f n).im, ⋯ …
      ⊢ LT.lt (HAdd.hAdd (_root_.abs (HSub.hSub (↑f j).re (Complex.cauSeqRe f).lim)) …
    -/
    have := add_lt_add H₁ H₂
    /-
      case intro
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      x✝ : Nat
      H : ∀ (j : Nat), GE.ge j x✝ → And (LT.lt (_root_.abs (HSub.hSub (↑f j).re (Cau …
      j : Nat
      ij : GE.ge j x✝
      H₁ : LT.lt (_root_.abs (HSub.hSub (↑f j).re (CauSeq.lim ⟨fun n => (↑f n).re, ⋯ …
      H₂ : LT.lt (_root_.abs (HSub.hSub (↑f j).im (CauSeq.lim ⟨fun n => (↑f n).im, ⋯ …
      this : LT.lt (HAdd.hAdd (_root_.abs (HSub.hSub (↑f j).re (CauSeq.lim ⟨fun n => …
      ⊢ LT.lt (HAdd.hAdd (_root_.abs (HSub.hSub (↑f j).re (Complex.cauSeqRe f).lim)) …
    -/
    rwa [add_halves] at this
    /-
      🎉 no goals
    -/


instance instIsComplete : CauSeq.IsComplete ℂ Complex.abs :=
  ⟨fun f => ⟨limAux f, equiv_limAux f⟩⟩


theorem lim_eq_lim_im_add_lim_re (f : CauSeq ℂ Complex.abs) :
    lim f = ↑(lim (cauSeqRe f)) + ↑(lim (cauSeqIm f)) * I :=
  lim_eq_of_equiv_const <|
    calc
      f ≈ _ := equiv_limAux f
      _ = CauSeq.const Complex.abs (↑(lim (cauSeqRe f)) + ↑(lim (cauSeqIm f)) * I) :=
        CauSeq.ext fun _ =>
                          /-
                            f : CauSeq Complex ⇑Complex.abs
                            x✝ : Nat
                            ⊢ Eq (↑(CauSeq.const (⇑Complex.abs) (Complex.limAux f)) x✝).re (↑(CauSeq.const …
                          -/
                          /-
                            🎉 no goals
                          -/
          Complex.ext (by simp [limAux, cauSeqRe, ofReal]) (by simp [limAux, cauSeqIm, ofReal])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem lim_re (f : CauSeq ℂ Complex.abs) : lim (cauSeqRe f) = (lim f).re := by
  /-
    f : CauSeq Complex ⇑Complex.abs
    ⊢ Eq (Complex.cauSeqRe f).lim f.lim.re
  -/
  rw [lim_eq_lim_im_add_lim_re]; simp [ofReal]
                                 /-
                                   🎉 no goals
                                 -/


theorem lim_im (f : CauSeq ℂ Complex.abs) : lim (cauSeqIm f) = (lim f).im := by
  /-
    f : CauSeq Complex ⇑Complex.abs
    ⊢ Eq (Complex.cauSeqIm f).lim f.lim.im
  -/
  rw [lim_eq_lim_im_add_lim_re]; simp [ofReal]
                                 /-
                                   🎉 no goals
                                 -/


theorem isCauSeq_conj (f : CauSeq ℂ Complex.abs) :
    IsCauSeq Complex.abs fun n => conj (f n) := fun ε ε0 =>
  let ⟨i, hi⟩ := f.2 ε ε0
  ⟨i, fun j hj => by
    /-
      f : CauSeq Complex ⇑Complex.abs
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      hi : ∀ (j : Nat), GE.ge j i → LT.lt (Complex.abs (HSub.hSub (↑f j) (↑f i))) ε
      j : Nat
      hj : GE.ge j i
      ⊢ LT.lt (Complex.abs (HSub.hSub ((fun n => (starRingEnd Complex) (↑f n)) j) (( …
    -/
    rw [← RingHom.map_sub, abs_conj]; exact hi j hj⟩
                                      /-
                                        🎉 no goals
                                      -/


/-- The complex conjugate of a complex Cauchy sequence, as a complex Cauchy sequence. -/
noncomputable def cauSeqConj (f : CauSeq ℂ Complex.abs) : CauSeq ℂ Complex.abs :=
  ⟨_, isCauSeq_conj f⟩


theorem lim_conj (f : CauSeq ℂ Complex.abs) : lim (cauSeqConj f) = conj (lim f) :=
                  /-
                    f : CauSeq Complex ⇑Complex.abs
                    ⊢ Eq (Complex.cauSeqConj f).lim.re ((starRingEnd Complex) f.lim).re
                  -/
  Complex.ext (by simp [cauSeqConj, (lim_re _).symm, cauSeqRe])
                  /-
                    🎉 no goals
                  -/
        /-
          f : CauSeq Complex ⇑Complex.abs
          ⊢ Eq (Complex.cauSeqConj f).lim.im ((starRingEnd Complex) f.lim).im
        -/
    (by simp [cauSeqConj, (lim_im _).symm, cauSeqIm, (lim_neg _).symm]; rfl)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The absolute value of a complex Cauchy sequence, as a real Cauchy sequence. -/
noncomputable def cauSeqAbs (f : CauSeq ℂ Complex.abs) : CauSeq ℝ abs' :=
  ⟨_, isCauSeq_abs f.2⟩


theorem lim_abs (f : CauSeq ℂ Complex.abs) : lim (cauSeqAbs f) = Complex.abs (lim f) :=
  lim_eq_of_equiv_const fun ε ε0 =>
    let ⟨i, hi⟩ := equiv_lim f ε ε0
    ⟨i, fun j hj => lt_of_le_of_lt (Complex.abs.abs_abv_sub_le_abv_sub _ _) (hi j hj)⟩


lemma ne_zero_of_one_lt_re {s : ℂ} (hs : 1 < s.re) : s ≠ 0 :=
  fun h ↦ ((zero_re ▸ h ▸ hs).trans zero_lt_one).false


lemma re_neg_ne_zero_of_one_lt_re {s : ℂ} (hs : 1 < s.re) : (-s).re ≠ 0 :=
                                                 /-
                                                   s : Complex
                                                   hs : LT.lt 1 s.re
                                                   ⊢ LT.lt (Neg.neg s.re) 0
                                                 -/
  ne_iff_lt_or_gt.mpr <| Or.inl <| neg_re s ▸ by linarith
                                                 /-
                                                   🎉 no goals
                                                 -/


