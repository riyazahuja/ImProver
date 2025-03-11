theorem isCauSeq_abs_exp (z : ℂ) :
    IsCauSeq _root_.abs fun n => ∑ m ∈ range n, abs (z ^ m / m.factorial) :=
  let ⟨n, hn⟩ := exists_nat_gt (abs z)
  have hn0 : (0 : ℝ) < n := lt_of_le_of_lt (abs.nonneg _) hn
  IsCauSeq.series_ratio_test n (abs z / n) (div_nonneg (abs.nonneg _) (le_of_lt hn0))
        /-
          z : Complex
          n : Nat
          hn : LT.lt (Complex.abs z) ↑n
          hn0 : LT.lt 0 ↑n
          ⊢ LT.lt (HDiv.hDiv (Complex.abs z) ↑n) 1
        -/
    (by rwa [div_lt_iff₀ hn0, one_mul]) fun m hm => by
        /-
          🎉 no goals
        -/
      rw [abs_abs, abs_abs, Nat.factorial_succ, pow_succ', mul_comm m.succ, Nat.cast_mul, ← div_div,
        mul_div_assoc, mul_div_right_comm, map_mul, map_div₀, abs_natCast]
      /-
        z : Complex
        n : Nat
        hn : LT.lt (Complex.abs z) ↑n
        hn0 : LT.lt 0 ↑n
        m : Nat
        hm : LE.le n m
        ⊢ LE.le (HMul.hMul (HDiv.hDiv (Complex.abs z) ↑m.succ) (Complex.abs (HDiv.hDiv …
      -/
      gcongr
      /-
        case h.h.h
        z : Complex
        n : Nat
        hn : LT.lt (Complex.abs z) ↑n
        hn0 : LT.lt 0 ↑n
        m : Nat
        hm : LE.le n m
        ⊢ LE.le n m.succ
      -/
      exact le_trans hm (Nat.le_succ _)
      /-
        🎉 no goals
      -/


theorem isCauSeq_exp (z : ℂ) : IsCauSeq abs fun n => ∑ m ∈ range n, z ^ m / m.factorial :=
  (isCauSeq_abs_exp z).of_abv


/-- The Cauchy sequence consisting of partial sums of the Taylor series of
the complex exponential function -/
@[pp_nodot]
def exp' (z : ℂ) : CauSeq ℂ Complex.abs :=
  ⟨fun n => ∑ m ∈ range n, z ^ m / m.factorial, isCauSeq_exp z⟩


/-- The complex exponential function, defined via its Taylor series -/
-- Porting note: removed `irreducible` attribute, so I can prove things
@[pp_nodot]
def exp (z : ℂ) : ℂ :=
  CauSeq.lim (exp' z)


/-- The complex sine function, defined via `exp` -/
@[pp_nodot]
def sin (z : ℂ) : ℂ :=
  (exp (-z * I) - exp (z * I)) * I / 2


/-- The complex cosine function, defined via `exp` -/
@[pp_nodot]
def cos (z : ℂ) : ℂ :=
  (exp (z * I) + exp (-z * I)) / 2


/-- The complex tangent function, defined as `sin z / cos z` -/
@[pp_nodot]
def tan (z : ℂ) : ℂ :=
  sin z / cos z


/-- The complex cotangent function, defined as `cos z / sin z` -/
def cot (z : ℂ) : ℂ :=
  cos z / sin z


/-- The complex hyperbolic sine function, defined via `exp` -/
@[pp_nodot]
def sinh (z : ℂ) : ℂ :=
  (exp z - exp (-z)) / 2


/-- The complex hyperbolic cosine function, defined via `exp` -/
@[pp_nodot]
def cosh (z : ℂ) : ℂ :=
  (exp z + exp (-z)) / 2


/-- The complex hyperbolic tangent function, defined as `sinh z / cosh z` -/
@[pp_nodot]
def tanh (z : ℂ) : ℂ :=
  sinh z / cosh z


/-- scoped notation for the complex exponential function -/
scoped notation "cexp" => Complex.exp


/-- The real exponential function, defined as the real part of the complex exponential -/
@[pp_nodot]
nonrec def exp (x : ℝ) : ℝ :=
  (exp x).re


/-- The real sine function, defined as the real part of the complex sine -/
@[pp_nodot]
nonrec def sin (x : ℝ) : ℝ :=
  (sin x).re


/-- The real cosine function, defined as the real part of the complex cosine -/
@[pp_nodot]
nonrec def cos (x : ℝ) : ℝ :=
  (cos x).re


/-- The real tangent function, defined as the real part of the complex tangent -/
@[pp_nodot]
nonrec def tan (x : ℝ) : ℝ :=
  (tan x).re


/-- The real cotangent function, defined as the real part of the complex cotangent -/
nonrec def cot (x : ℝ) : ℝ :=
  (cot x).re


/-- The real hypebolic sine function, defined as the real part of the complex hyperbolic sine -/
@[pp_nodot]
nonrec def sinh (x : ℝ) : ℝ :=
  (sinh x).re


/-- The real hypebolic cosine function, defined as the real part of the complex hyperbolic cosine -/
@[pp_nodot]
nonrec def cosh (x : ℝ) : ℝ :=
  (cosh x).re


/-- The real hypebolic tangent function, defined as the real part of
the complex hyperbolic tangent -/
@[pp_nodot]
nonrec def tanh (x : ℝ) : ℝ :=
  (tanh x).re


/-- scoped notation for the real exponential function -/
scoped notation "rexp" => Real.exp


@[simp]
theorem exp_zero : exp 0 = 1 := by
  /-
    ⊢ Eq (Complex.exp 0) 1
  -/
  rw [exp]
  /-
    ⊢ Eq (Complex.exp' 0).lim 1
  -/
  refine lim_eq_of_equiv_const fun ε ε0 => ⟨1, fun j hj => ?_⟩
  /-
    ε : Real
    ε0 : GT.gt ε 0
    j : Nat
    hj : GE.ge j 1
    ⊢ LT.lt (Complex.abs (↑(HSub.hSub (Complex.exp' 0) (CauSeq.const (⇑Complex.abs …
  -/
  convert (config := .unfoldSameFun) ε0 -- Porting note: ε0 : ε > 0 but goal is _ < ε
  /-
    case h.e'_3
    ε : Real
    ε0 : GT.gt ε 0
    j : Nat
    hj : GE.ge j 1
    ⊢ Eq (Complex.abs (↑(HSub.hSub (Complex.exp' 0) (CauSeq.const (⇑Complex.abs) 1 …
  -/
  cases' j with j j
    /-
      case h.e'_3.zero
      ε : Real
      ε0 : GT.gt ε 0
      hj : GE.ge 0 1
      ⊢ Eq (Complex.abs (↑(HSub.hSub (Complex.exp' 0) (CauSeq.const (⇑Complex.abs) 1 …
    -/
  · exact absurd hj (not_le_of_gt zero_lt_one)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.succ
      ε : Real
      ε0 : GT.gt ε 0
      j : Nat
      hj : GE.ge (HAdd.hAdd j 1) 1
      ⊢ Eq (Complex.abs (↑(HSub.hSub (Complex.exp' 0) (CauSeq.const (⇑Complex.abs) 1 …
    -/
  · dsimp [exp']
    /-
      case h.e'_3.succ
      ε : Real
      ε0 : GT.gt ε 0
      j : Nat
      hj : GE.ge (HAdd.hAdd j 1) 1
      ⊢ Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd.hAdd j 1)).sum fun m => HDiv …
    -/
    induction' j with j ih
      /-
        case h.e'_3.succ.zero
        ε : Real
        ε0 : GT.gt ε 0
        hj : GE.ge (HAdd.hAdd 0 1) 1
        ⊢ Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd.hAdd 0 1)).sum fun m => HDiv …
      -/
    · dsimp [exp']; simp [show Nat.succ 0 = 1 from rfl]
                    /-
                      🎉 no goals
                    -/
      /-
        case h.e'_3.succ.succ
        ε : Real
        ε0 : GT.gt ε 0
        j : Nat
        ih : GE.ge (HAdd.hAdd j 1) 1 → Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd …
        hj : GE.ge (HAdd.hAdd (HAdd.hAdd j 1) 1) 1
        ⊢ Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd.hAdd (HAdd.hAdd j 1) 1)).sum …
      -/
    · rw [← ih (by simp [Nat.succ_le_succ])]
      /-
        case h.e'_3.succ.succ
        ε : Real
        ε0 : GT.gt ε 0
        j : Nat
        ih : GE.ge (HAdd.hAdd j 1) 1 → Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd …
        hj : GE.ge (HAdd.hAdd (HAdd.hAdd j 1) 1) 1
        ⊢ Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd.hAdd (HAdd.hAdd j 1) 1)).sum …
      -/
      simp only [sum_range_succ, pow_succ]
      /-
        case h.e'_3.succ.succ
        ε : Real
        ε0 : GT.gt ε 0
        j : Nat
        ih : GE.ge (HAdd.hAdd j 1) 1 → Eq (Complex.abs (HSub.hSub ((Finset.range (HAdd …
        hj : GE.ge (HAdd.hAdd (HAdd.hAdd j 1) 1) 1
        ⊢ Eq (Complex.abs (HSub.hSub (HAdd.hAdd (HAdd.hAdd ((Finset.range j).sum fun m …
      -/
      simp
      /-
        🎉 no goals
      -/


theorem exp_add : exp (x + y) = exp x * exp y := by
  have hj : ∀ j : ℕ, (∑ m ∈ range j, (x + y) ^ m / m.factorial) =
        ∑ i ∈ range j, ∑ k ∈ range (i + 1), x ^ k / k.factorial *
          (y ^ (i - k) / (i - k).factorial) := by
    intro j
    refine Finset.sum_congr rfl fun m _ => ?_
    rw [add_pow, div_eq_mul_inv, sum_mul]
    refine Finset.sum_congr rfl fun I hi => ?_
    have h₁ : (m.choose I : ℂ) ≠ 0 :=
      Nat.cast_ne_zero.2 (pos_iff_ne_zero.1 (Nat.choose_pos (Nat.le_of_lt_succ (mem_range.1 hi))))
    have h₂ := Nat.choose_mul_factorial_mul_factorial (Nat.le_of_lt_succ <| Finset.mem_range.1 hi)
    rw [← h₂, Nat.cast_mul, Nat.cast_mul, mul_inv, mul_inv]
    simp only [mul_left_comm (m.choose I : ℂ), mul_assoc, mul_left_comm (m.choose I : ℂ)⁻¹,
      mul_comm (m.choose I : ℂ)]
    rw [inv_mul_cancel₀ h₁]
    simp [div_eq_mul_inv, mul_comm, mul_assoc, mul_left_comm]
  /-
    x y : Complex
    hj : ∀ (j : Nat), Eq ((Finset.range j).sum fun m => HDiv.hDiv (HPow.hPow (HAdd …
    ⊢ Eq (Complex.exp (HAdd.hAdd x y)) (HMul.hMul (Complex.exp x) (Complex.exp y))
  -/
  simp_rw [exp, exp', lim_mul_lim]
  /-
    x y : Complex
    hj : ∀ (j : Nat), Eq ((Finset.range j).sum fun m => HDiv.hDiv (HPow.hPow (HAdd …
    ⊢ Eq (CauSeq.lim ⟨fun n => (Finset.range n).sum fun m => HDiv.hDiv (HPow.hPow  …
  -/
  apply (lim_eq_lim_of_equiv _).symm
  /-
    x y : Complex
    hj : ∀ (j : Nat), Eq ((Finset.range j).sum fun m => HDiv.hDiv (HPow.hPow (HAdd …
    ⊢ HasEquiv.Equiv (HMul.hMul ⟨fun n => (Finset.range n).sum fun m => HDiv.hDiv  …
  -/
  simp only [hj]
  /-
    x y : Complex
    hj : ∀ (j : Nat), Eq ((Finset.range j).sum fun m => HDiv.hDiv (HPow.hPow (HAdd …
    ⊢ HasEquiv.Equiv (HMul.hMul ⟨fun n => (Finset.range n).sum fun m => HDiv.hDiv  …
  -/
  exact cauchy_product (isCauSeq_abs_exp x) (isCauSeq_exp y)
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): new definition

/-- the exponential function as a monoid hom from `Multiplicative ℂ` to `ℂ` -/
@[simps]
noncomputable def expMonoidHom : MonoidHom (Multiplicative ℂ) ℂ :=
  { toFun := fun z => exp z.toAdd,
                   /-
                     x y : Complex
                     ⊢ Eq ((fun z => Complex.exp (Multiplicative.toAdd z)) 1) 1
                   -/
    map_one' := by simp,
                   /-
                     🎉 no goals
                   -/
                   /-
                     x y : Complex
                     ⊢ ∀ (x y : Multiplicative Complex), Eq ({ toFun := fun z => Complex.exp (Multi …
                   -/
    map_mul' := by simp [exp_add] }
                   /-
                     🎉 no goals
                   -/


theorem exp_list_sum (l : List ℂ) : exp l.sum = (l.map exp).prod :=
  map_list_prod (M := Multiplicative ℂ) expMonoidHom l


theorem exp_multiset_sum (s : Multiset ℂ) : exp s.sum = (s.map exp).prod :=
  @MonoidHom.map_multiset_prod (Multiplicative ℂ) ℂ _ _ expMonoidHom s


theorem exp_sum {α : Type*} (s : Finset α) (f : α → ℂ) :
    exp (∑ x ∈ s, f x) = ∏ x ∈ s, exp (f x) :=
  map_prod (β := Multiplicative ℂ) expMonoidHom f s


lemma exp_nsmul (x : ℂ) (n : ℕ) : exp (n • x) = exp x ^ n :=
  @MonoidHom.map_pow (Multiplicative ℂ) ℂ _ _  expMonoidHom _ _


theorem exp_nat_mul (x : ℂ) : ∀ n : ℕ, exp (n * x) = exp x ^ n
            /-
              x : Complex
              ⊢ Eq (Complex.exp (HMul.hMul (↑0) x)) (HPow.hPow (Complex.exp x) 0)
            -/
  | 0 => by rw [Nat.cast_zero, zero_mul, exp_zero, pow_zero]
            /-
              🎉 no goals
            -/
                     /-
                       x : Complex
                       n : Nat
                       ⊢ Eq (Complex.exp (HMul.hMul (↑n.succ) x)) (HPow.hPow (Complex.exp x) n.succ)
                     -/
  | Nat.succ n => by rw [pow_succ, Nat.cast_add_one, add_mul, exp_add, ← exp_nat_mul _ n, one_mul]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem exp_ne_zero : exp x ≠ 0 := fun h =>
                             /-
                               x : Complex
                               h : Eq (Complex.exp x) 0
                               ⊢ Eq 0 1
                             -/
  zero_ne_one (α := ℂ) <| by rw [← exp_zero, ← add_neg_cancel x, exp_add, h]; simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem exp_neg : exp (-x) = (exp x)⁻¹ := by
  /-
    x : Complex
    ⊢ Eq (Complex.exp (Neg.neg x)) (Inv.inv (Complex.exp x))
  -/
  rw [← mul_right_inj' (exp_ne_zero x), ← exp_add]; simp [mul_inv_cancel₀ (exp_ne_zero x)]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem exp_sub : exp (x - y) = exp x / exp y := by
  /-
    x y : Complex
    ⊢ Eq (Complex.exp (HSub.hSub x y)) (HDiv.hDiv (Complex.exp x) (Complex.exp y))
  -/
  simp [sub_eq_add_neg, exp_add, exp_neg, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem exp_int_mul (z : ℂ) (n : ℤ) : Complex.exp (n * z) = Complex.exp z ^ n := by
  /-
    z : Complex
    n : Int
    ⊢ Eq (Complex.exp (HMul.hMul (↑n) z)) (HPow.hPow (Complex.exp z) n)
  -/
  cases n
    /-
      case ofNat
      z : Complex
      a✝ : Nat
      ⊢ Eq (Complex.exp (HMul.hMul (↑(Int.ofNat a✝)) z)) (HPow.hPow (Complex.exp z)  …
    -/
  · simp [exp_nat_mul]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      z : Complex
      a✝ : Nat
      ⊢ Eq (Complex.exp (HMul.hMul (↑(Int.negSucc a✝)) z)) (HPow.hPow (Complex.exp z …
    -/
  · simp [exp_add, add_mul, pow_add, exp_neg, exp_nat_mul]
    /-
      🎉 no goals
    -/


@[simp]
theorem exp_conj : exp (conj x) = conj (exp x) := by
  /-
    x : Complex
    ⊢ Eq (Complex.exp ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex.e …
  -/
  dsimp [exp]
  /-
    x : Complex
    ⊢ Eq (Complex.exp' ((starRingEnd Complex) x)).lim ((starRingEnd Complex) (Comp …
  -/
  rw [← lim_conj]
  /-
    x : Complex
    ⊢ Eq (Complex.exp' ((starRingEnd Complex) x)).lim (Complex.cauSeqConj (Complex …
  -/
  refine congr_arg CauSeq.lim (CauSeq.ext fun _ => ?_)
  /-
    x : Complex
    x✝ : Nat
    ⊢ Eq (↑(Complex.exp' ((starRingEnd Complex) x)) x✝) (↑(Complex.cauSeqConj (Com …
  -/
  dsimp [exp', Function.comp_def, cauSeqConj]
  /-
    x : Complex
    x✝ : Nat
    ⊢ Eq ((Finset.range x✝).sum fun m => HDiv.hDiv (HPow.hPow ((starRingEnd Comple …
  -/
  rw [map_sum (starRingEnd _)]
  /-
    x : Complex
    x✝ : Nat
    ⊢ Eq ((Finset.range x✝).sum fun m => HDiv.hDiv (HPow.hPow ((starRingEnd Comple …
  -/
  refine sum_congr rfl fun n _ => ?_
  /-
    x : Complex
    x✝¹ n : Nat
    x✝ : Membership.mem (Finset.range x✝¹) n
    ⊢ Eq (HDiv.hDiv (HPow.hPow ((starRingEnd Complex) x) n) ↑n.factorial) ((starRi …
  -/
  rw [map_div₀, map_pow, ← ofReal_natCast, conj_ofReal]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofReal_exp_ofReal_re (x : ℝ) : ((exp x).re : ℂ) = exp x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.exp ↑x)) (Complex.exp ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← exp_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_exp (x : ℝ) : (Real.exp x : ℂ) = exp x :=
  ofReal_exp_ofReal_re _


@[simp]
                                                     /-
                                                       x : Real
                                                       ⊢ Eq (Complex.exp ↑x).im 0
                                                     -/
theorem exp_ofReal_im (x : ℝ) : (exp x).im = 0 := by rw [← ofReal_exp_ofReal_re, ofReal_im]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem exp_ofReal_re (x : ℝ) : (exp x).re = Real.exp x :=
  rfl


theorem two_sinh : 2 * sinh x = exp x - exp (-x) :=
  mul_div_cancel₀ _ two_ne_zero


theorem two_cosh : 2 * cosh x = exp x + exp (-x) :=
  mul_div_cancel₀ _ two_ne_zero


@[simp]
                                     /-
                                       ⊢ Eq (Complex.sinh 0) 0
                                     -/
theorem sinh_zero : sinh 0 = 0 := by simp [sinh]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                             /-
                                               x : Complex
                                               ⊢ Eq (Complex.sinh (Neg.neg x)) (Neg.neg (Complex.sinh x))
                                             -/
theorem sinh_neg : sinh (-x) = -sinh x := by simp [sinh, exp_neg, (neg_div _ _).symm, add_mul]
                                             /-
                                               🎉 no goals
                                             -/


private theorem sinh_add_aux {a b c d : ℂ} :
                                                                      /-
                                                                        a b c d : Complex
                                                                        ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub a b) (HAdd.hAdd c d)) (HMul.hMul (HAdd.h …
                                                                      -/
    (a - b) * (c + d) + (a + b) * (c - d) = 2 * (a * c - b * d) := by ring
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem sinh_add : sinh (x + y) = sinh x * cosh y + cosh x * sinh y := by
  rw [← mul_right_inj' (two_ne_zero' ℂ), two_sinh, exp_add, neg_add, exp_add, eq_comm, mul_add, ←
    mul_assoc, two_sinh, mul_left_comm, two_sinh, ← mul_right_inj' (two_ne_zero' ℂ), mul_add,
    mul_left_comm, two_cosh, ← mul_assoc, two_cosh]
  /-
    x y : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (Complex.exp x) (Complex.exp (Neg.neg x) …
  -/
  exact sinh_add_aux
  /-
    🎉 no goals
  -/


@[simp]
                                     /-
                                       ⊢ Eq (Complex.cosh 0) 1
                                     -/
theorem cosh_zero : cosh 0 = 1 := by simp [cosh]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                            /-
                                              x : Complex
                                              ⊢ Eq (Complex.cosh (Neg.neg x)) (Complex.cosh x)
                                            -/
theorem cosh_neg : cosh (-x) = cosh x := by simp [add_comm, cosh, exp_neg]
                                            /-
                                              🎉 no goals
                                            -/


private theorem cosh_add_aux {a b c d : ℂ} :
                                                                      /-
                                                                        a b c d : Complex
                                                                        ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd a b) (HAdd.hAdd c d)) (HMul.hMul (HSub.h …
                                                                      -/
    (a + b) * (c + d) + (a - b) * (c - d) = 2 * (a * c + b * d) := by ring
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem cosh_add : cosh (x + y) = cosh x * cosh y + sinh x * sinh y := by
  rw [← mul_right_inj' (two_ne_zero' ℂ), two_cosh, exp_add, neg_add, exp_add, eq_comm, mul_add, ←
    mul_assoc, two_cosh, ← mul_assoc, two_sinh, ← mul_right_inj' (two_ne_zero' ℂ), mul_add,
    mul_left_comm, two_cosh, mul_left_comm, two_sinh]
  /-
    x y : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (Complex.exp x) (Complex.exp (Neg.neg x) …
  -/
  exact cosh_add_aux
  /-
    🎉 no goals
  -/


theorem sinh_sub : sinh (x - y) = sinh x * cosh y - cosh x * sinh y := by
  /-
    x y : Complex
    ⊢ Eq (Complex.sinh (HSub.hSub x y)) (HSub.hSub (HMul.hMul (Complex.sinh x) (Co …
  -/
  simp [sub_eq_add_neg, sinh_add, sinh_neg, cosh_neg]
  /-
    🎉 no goals
  -/


theorem cosh_sub : cosh (x - y) = cosh x * cosh y - sinh x * sinh y := by
  /-
    x y : Complex
    ⊢ Eq (Complex.cosh (HSub.hSub x y)) (HSub.hSub (HMul.hMul (Complex.cosh x) (Co …
  -/
  simp [sub_eq_add_neg, cosh_add, sinh_neg, cosh_neg]
  /-
    🎉 no goals
  -/


theorem sinh_conj : sinh (conj x) = conj (sinh x) := by
  /-
    x : Complex
    ⊢ Eq (Complex.sinh ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex. …
  -/
  rw [sinh, ← RingHom.map_neg, exp_conj, exp_conj, ← RingHom.map_sub, sinh, map_div₀]
  -- Porting note: not nice
  /-
    x : Complex
    ⊢ Eq (HDiv.hDiv ((starRingEnd Complex) (HSub.hSub (Complex.exp x) (Complex.exp …
  -/
  simp [← one_add_one_eq_two]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofReal_sinh_ofReal_re (x : ℝ) : ((sinh x).re : ℂ) = sinh x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.sinh ↑x)) (Complex.sinh ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← sinh_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_sinh (x : ℝ) : (Real.sinh x : ℂ) = sinh x :=
  ofReal_sinh_ofReal_re _


@[simp]
                                                       /-
                                                         x : Real
                                                         ⊢ Eq (Complex.sinh ↑x).im 0
                                                       -/
theorem sinh_ofReal_im (x : ℝ) : (sinh x).im = 0 := by rw [← ofReal_sinh_ofReal_re, ofReal_im]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem sinh_ofReal_re (x : ℝ) : (sinh x).re = Real.sinh x :=
  rfl


theorem cosh_conj : cosh (conj x) = conj (cosh x) := by
  /-
    x : Complex
    ⊢ Eq (Complex.cosh ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex. …
  -/
  rw [cosh, ← RingHom.map_neg, exp_conj, exp_conj, ← RingHom.map_add, cosh, map_div₀]
  -- Porting note: not nice
  /-
    x : Complex
    ⊢ Eq (HDiv.hDiv ((starRingEnd Complex) (HAdd.hAdd (Complex.exp x) (Complex.exp …
  -/
  simp [← one_add_one_eq_two]
  /-
    🎉 no goals
  -/


theorem ofReal_cosh_ofReal_re (x : ℝ) : ((cosh x).re : ℂ) = cosh x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.cosh ↑x)) (Complex.cosh ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← cosh_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_cosh (x : ℝ) : (Real.cosh x : ℂ) = cosh x :=
  ofReal_cosh_ofReal_re _


@[simp]
                                                       /-
                                                         x : Real
                                                         ⊢ Eq (Complex.cosh ↑x).im 0
                                                       -/
theorem cosh_ofReal_im (x : ℝ) : (cosh x).im = 0 := by rw [← ofReal_cosh_ofReal_re, ofReal_im]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem cosh_ofReal_re (x : ℝ) : (cosh x).re = Real.cosh x :=
  rfl


theorem tanh_eq_sinh_div_cosh : tanh x = sinh x / cosh x :=
  rfl


@[simp]
                                     /-
                                       ⊢ Eq (Complex.tanh 0) 0
                                     -/
theorem tanh_zero : tanh 0 = 0 := by simp [tanh]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                             /-
                                               x : Complex
                                               ⊢ Eq (Complex.tanh (Neg.neg x)) (Neg.neg (Complex.tanh x))
                                             -/
theorem tanh_neg : tanh (-x) = -tanh x := by simp [tanh, neg_div]
                                             /-
                                               🎉 no goals
                                             -/


theorem tanh_conj : tanh (conj x) = conj (tanh x) := by
  /-
    x : Complex
    ⊢ Eq (Complex.tanh ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex. …
  -/
  rw [tanh, sinh_conj, cosh_conj, ← map_div₀, tanh]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofReal_tanh_ofReal_re (x : ℝ) : ((tanh x).re : ℂ) = tanh x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.tanh ↑x)) (Complex.tanh ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← tanh_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_tanh (x : ℝ) : (Real.tanh x : ℂ) = tanh x :=
  ofReal_tanh_ofReal_re _


@[simp]
                                                       /-
                                                         x : Real
                                                         ⊢ Eq (Complex.tanh ↑x).im 0
                                                       -/
theorem tanh_ofReal_im (x : ℝ) : (tanh x).im = 0 := by rw [← ofReal_tanh_ofReal_re, ofReal_im]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem tanh_ofReal_re (x : ℝ) : (tanh x).re = Real.tanh x :=
  rfl


@[simp]
theorem cosh_add_sinh : cosh x + sinh x = exp x := by
  /-
    x : Complex
    ⊢ Eq (HAdd.hAdd (Complex.cosh x) (Complex.sinh x)) (Complex.exp x)
  -/
  rw [← mul_right_inj' (two_ne_zero' ℂ), mul_add, two_cosh, two_sinh, add_add_sub_cancel, two_mul]
  /-
    🎉 no goals
  -/


@[simp]
                                                      /-
                                                        x : Complex
                                                        ⊢ Eq (HAdd.hAdd (Complex.sinh x) (Complex.cosh x)) (Complex.exp x)
                                                      -/
theorem sinh_add_cosh : sinh x + cosh x = exp x := by rw [add_comm, cosh_add_sinh]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem exp_sub_cosh : exp x - cosh x = sinh x :=
  sub_eq_iff_eq_add.2 (sinh_add_cosh x).symm


@[simp]
theorem exp_sub_sinh : exp x - sinh x = cosh x :=
  sub_eq_iff_eq_add.2 (cosh_add_sinh x).symm


@[simp]
theorem cosh_sub_sinh : cosh x - sinh x = exp (-x) := by
  /-
    x : Complex
    ⊢ Eq (HSub.hSub (Complex.cosh x) (Complex.sinh x)) (Complex.exp (Neg.neg x))
  -/
  rw [← mul_right_inj' (two_ne_zero' ℂ), mul_sub, two_cosh, two_sinh, add_sub_sub_cancel, two_mul]
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            x : Complex
                                                            ⊢ Eq (HSub.hSub (Complex.sinh x) (Complex.cosh x)) (Neg.neg (Complex.exp (Neg. …
                                                          -/
theorem sinh_sub_cosh : sinh x - cosh x = -exp (-x) := by rw [← neg_sub, cosh_sub_sinh]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem cosh_sq_sub_sinh_sq : cosh x ^ 2 - sinh x ^ 2 = 1 := by
  /-
    x : Complex
    ⊢ Eq (HSub.hSub (HPow.hPow (Complex.cosh x) 2) (HPow.hPow (Complex.sinh x) 2)) 1
  -/
  rw [sq_sub_sq, cosh_add_sinh, cosh_sub_sinh, ← exp_add, add_neg_cancel, exp_zero]
  /-
    🎉 no goals
  -/


theorem cosh_sq : cosh x ^ 2 = sinh x ^ 2 + 1 := by
  /-
    x : Complex
    ⊢ Eq (HPow.hPow (Complex.cosh x) 2) (HAdd.hAdd (HPow.hPow (Complex.sinh x) 2) 1)
  -/
  rw [← cosh_sq_sub_sinh_sq x]
  /-
    x : Complex
    ⊢ Eq (HPow.hPow (Complex.cosh x) 2) (HAdd.hAdd (HPow.hPow (Complex.sinh x) 2)  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem sinh_sq : sinh x ^ 2 = cosh x ^ 2 - 1 := by
  /-
    x : Complex
    ⊢ Eq (HPow.hPow (Complex.sinh x) 2) (HSub.hSub (HPow.hPow (Complex.cosh x) 2) 1)
  -/
  rw [← cosh_sq_sub_sinh_sq x]
  /-
    x : Complex
    ⊢ Eq (HPow.hPow (Complex.sinh x) 2) (HSub.hSub (HPow.hPow (Complex.cosh x) 2)  …
  -/
  ring
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      x : Complex
                                                                      ⊢ Eq (Complex.cosh (HMul.hMul 2 x)) (HAdd.hAdd (HPow.hPow (Complex.cosh x) 2)  …
                                                                    -/
theorem cosh_two_mul : cosh (2 * x) = cosh x ^ 2 + sinh x ^ 2 := by rw [two_mul, cosh_add, sq, sq]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem sinh_two_mul : sinh (2 * x) = 2 * sinh x * cosh x := by
  /-
    x : Complex
    ⊢ Eq (Complex.sinh (HMul.hMul 2 x)) (HMul.hMul (HMul.hMul 2 (Complex.sinh x))  …
  -/
  rw [two_mul, sinh_add]
  /-
    x : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sinh x) (Complex.cosh x)) (HMul.hMul (Comp …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem cosh_three_mul : cosh (3 * x) = 4 * cosh x ^ 3 - 3 * cosh x := by
  /-
    x : Complex
    ⊢ Eq (Complex.cosh (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 4 (HPow.hPow (Comple …
  -/
  have h1 : x + 2 * x = 3 * x := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (Complex.cosh (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 4 (HPow.hPow (Comple …
  -/
  rw [← h1, cosh_add x (2 * x)]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.cosh x) (Complex.cosh (HMul.hMul 2 x))) (H …
  -/
  simp only [cosh_two_mul, sinh_two_mul]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.cosh x) (HAdd.hAdd (HPow.hPow (Complex.cos …
  -/
  have h2 : sinh x * (2 * sinh x * cosh x) = 2 * cosh x * sinh x ^ 2 := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul (Complex.sinh x) (HMul.hMul (HMul.hMul 2 (Complex.sinh x))  …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.cosh x) (HAdd.hAdd (HPow.hPow (Complex.cos …
  -/
  rw [h2, sinh_sq]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul (Complex.sinh x) (HMul.hMul (HMul.hMul 2 (Complex.sinh x))  …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.cosh x) (HAdd.hAdd (HPow.hPow (Complex.cos …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem sinh_three_mul : sinh (3 * x) = 4 * sinh x ^ 3 + 3 * sinh x := by
  /-
    x : Complex
    ⊢ Eq (Complex.sinh (HMul.hMul 3 x)) (HAdd.hAdd (HMul.hMul 4 (HPow.hPow (Comple …
  -/
  have h1 : x + 2 * x = 3 * x := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (Complex.sinh (HMul.hMul 3 x)) (HAdd.hAdd (HMul.hMul 4 (HPow.hPow (Comple …
  -/
  rw [← h1, sinh_add x (2 * x)]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sinh x) (Complex.cosh (HMul.hMul 2 x))) (H …
  -/
  simp only [cosh_two_mul, sinh_two_mul]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sinh x) (HAdd.hAdd (HPow.hPow (Complex.cos …
  -/
  have h2 : cosh x * (2 * sinh x * cosh x) = 2 * sinh x * cosh x ^ 2 := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul (Complex.cosh x) (HMul.hMul (HMul.hMul 2 (Complex.sinh x))  …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sinh x) (HAdd.hAdd (HPow.hPow (Complex.cos …
  -/
  rw [h2, cosh_sq]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul (Complex.cosh x) (HMul.hMul (HMul.hMul 2 (Complex.sinh x))  …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sinh x) (HAdd.hAdd (HAdd.hAdd (HPow.hPow ( …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
                                   /-
                                     ⊢ Eq (Complex.sin 0) 0
                                   -/
theorem sin_zero : sin 0 = 0 := by simp [sin]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem sin_neg : sin (-x) = -sin x := by
  /-
    x : Complex
    ⊢ Eq (Complex.sin (Neg.neg x)) (Neg.neg (Complex.sin x))
  -/
  simp [sin, sub_eq_add_neg, exp_neg, (neg_div _ _).symm, add_mul]
  /-
    🎉 no goals
  -/


theorem two_sin : 2 * sin x = (exp (-x * I) - exp (x * I)) * I :=
  mul_div_cancel₀ _ two_ne_zero


theorem two_cos : 2 * cos x = exp (x * I) + exp (-x * I) :=
  mul_div_cancel₀ _ two_ne_zero


theorem sinh_mul_I : sinh (x * I) = sin x * I := by
  rw [← mul_right_inj' (two_ne_zero' ℂ), two_sinh, ← mul_assoc, two_sin, mul_assoc, I_mul_I,
    mul_neg_one, neg_sub, neg_mul_eq_neg_mul]


theorem cosh_mul_I : cosh (x * I) = cos x := by
  /-
    x : Complex
    ⊢ Eq (Complex.cosh (HMul.hMul x Complex.I)) (Complex.cos x)
  -/
  rw [← mul_right_inj' (two_ne_zero' ℂ), two_cosh, two_cos, neg_mul_eq_neg_mul]
  /-
    🎉 no goals
  -/


theorem tanh_mul_I : tanh (x * I) = tan x * I := by
  /-
    x : Complex
    ⊢ Eq (Complex.tanh (HMul.hMul x Complex.I)) (HMul.hMul (Complex.tan x) Complex …
  -/
  rw [tanh_eq_sinh_div_cosh, cosh_mul_I, sinh_mul_I, mul_div_right_comm, tan]
  /-
    🎉 no goals
  -/


                                               /-
                                                 x : Complex
                                                 ⊢ Eq (Complex.cos (HMul.hMul x Complex.I)) (Complex.cosh x)
                                               -/
theorem cos_mul_I : cos (x * I) = cosh x := by rw [← cosh_mul_I]; ring_nf; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem sin_mul_I : sin (x * I) = sinh x * I := by
  have h : I * sin (x * I) = -sinh x := by
    rw [mul_comm, ← sinh_mul_I]
    ring_nf
    simp
  /-
    x : Complex
    h : Eq (HMul.hMul Complex.I (Complex.sin (HMul.hMul x Complex.I))) (Neg.neg (C …
    ⊢ Eq (Complex.sin (HMul.hMul x Complex.I)) (HMul.hMul (Complex.sinh x) Complex …
  -/
  rw [← neg_neg (sinh x), ← h]
  /-
    x : Complex
    h : Eq (HMul.hMul Complex.I (Complex.sin (HMul.hMul x Complex.I))) (Neg.neg (C …
    ⊢ Eq (Complex.sin (HMul.hMul x Complex.I)) (HMul.hMul (Neg.neg (HMul.hMul Comp …
  -/
                        /-
                          🎉 no goals
                        -/
  apply Complex.ext <;> simp
                        /-
                          🎉 no goals
                        -/


theorem tan_mul_I : tan (x * I) = tanh x * I := by
  /-
    x : Complex
    ⊢ Eq (Complex.tan (HMul.hMul x Complex.I)) (HMul.hMul (Complex.tanh x) Complex …
  -/
  rw [tan, sin_mul_I, cos_mul_I, mul_div_right_comm, tanh_eq_sinh_div_cosh]
  /-
    🎉 no goals
  -/


theorem sin_add : sin (x + y) = sin x * cos y + cos x * sin y := by
  rw [← mul_left_inj' I_ne_zero, ← sinh_mul_I, add_mul, add_mul, mul_right_comm, ← sinh_mul_I,
    mul_assoc, ← sinh_mul_I, ← cosh_mul_I, ← cosh_mul_I, sinh_add]


@[simp]
                                   /-
                                     ⊢ Eq (Complex.cos 0) 1
                                   -/
theorem cos_zero : cos 0 = 1 := by simp [cos]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
                                         /-
                                           x : Complex
                                           ⊢ Eq (Complex.cos (Neg.neg x)) (Complex.cos x)
                                         -/
theorem cos_neg : cos (-x) = cos x := by simp [cos, sub_eq_add_neg, exp_neg, add_comm]
                                         /-
                                           🎉 no goals
                                         -/


theorem cos_add : cos (x + y) = cos x * cos y - sin x * sin y := by
  rw [← cosh_mul_I, add_mul, cosh_add, cosh_mul_I, cosh_mul_I, sinh_mul_I, sinh_mul_I,
    mul_mul_mul_comm, I_mul_I, mul_neg_one, sub_eq_add_neg]


theorem sin_sub : sin (x - y) = sin x * cos y - cos x * sin y := by
  /-
    x y : Complex
    ⊢ Eq (Complex.sin (HSub.hSub x y)) (HSub.hSub (HMul.hMul (Complex.sin x) (Comp …
  -/
  simp [sub_eq_add_neg, sin_add, sin_neg, cos_neg]
  /-
    🎉 no goals
  -/


theorem cos_sub : cos (x - y) = cos x * cos y + sin x * sin y := by
  /-
    x y : Complex
    ⊢ Eq (Complex.cos (HSub.hSub x y)) (HAdd.hAdd (HMul.hMul (Complex.cos x) (Comp …
  -/
  simp [sub_eq_add_neg, cos_add, sin_neg, cos_neg]
  /-
    🎉 no goals
  -/


theorem sin_add_mul_I (x y : ℂ) : sin (x + y * I) = sin x * cosh y + cos x * sinh y * I := by
  /-
    x y : Complex
    ⊢ Eq (Complex.sin (HAdd.hAdd x (HMul.hMul y Complex.I))) (HAdd.hAdd (HMul.hMul …
  -/
  rw [sin_add, cos_mul_I, sin_mul_I, mul_assoc]
  /-
    🎉 no goals
  -/


theorem sin_eq (z : ℂ) : sin z = sin z.re * cosh z.im + cos z.re * sinh z.im * I := by
  /-
    z : Complex
    ⊢ Eq (Complex.sin z) (HAdd.hAdd (HMul.hMul (Complex.sin ↑z.re) (Complex.cosh ↑ …
  -/
  convert sin_add_mul_I z.re z.im; exact (re_add_im z).symm
                                   /-
                                     🎉 no goals
                                   -/


theorem cos_add_mul_I (x y : ℂ) : cos (x + y * I) = cos x * cosh y - sin x * sinh y * I := by
  /-
    x y : Complex
    ⊢ Eq (Complex.cos (HAdd.hAdd x (HMul.hMul y Complex.I))) (HSub.hSub (HMul.hMul …
  -/
  rw [cos_add, cos_mul_I, sin_mul_I, mul_assoc]
  /-
    🎉 no goals
  -/


theorem cos_eq (z : ℂ) : cos z = cos z.re * cosh z.im - sin z.re * sinh z.im * I := by
  /-
    z : Complex
    ⊢ Eq (Complex.cos z) (HSub.hSub (HMul.hMul (Complex.cos ↑z.re) (Complex.cosh ↑ …
  -/
  convert cos_add_mul_I z.re z.im; exact (re_add_im z).symm
                                   /-
                                     🎉 no goals
                                   -/


theorem sin_sub_sin : sin x - sin y = 2 * sin ((x - y) / 2) * cos ((x + y) / 2) := by
  /-
    x y : Complex
    ⊢ Eq (HSub.hSub (Complex.sin x) (Complex.sin y)) (HMul.hMul (HMul.hMul 2 (Comp …
  -/
  have s1 := sin_add ((x + y) / 2) ((x - y) / 2)
  /-
    x y : Complex
    s1 : Eq (Complex.sin (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    ⊢ Eq (HSub.hSub (Complex.sin x) (Complex.sin y)) (HMul.hMul (HMul.hMul 2 (Comp …
  -/
  have s2 := sin_sub ((x + y) / 2) ((x - y) / 2)
  /-
    x y : Complex
    s1 : Eq (Complex.sin (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    s2 : Eq (Complex.sin (HSub.hSub (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    ⊢ Eq (HSub.hSub (Complex.sin x) (Complex.sin y)) (HMul.hMul (HMul.hMul 2 (Comp …
  -/
  rw [div_add_div_same, add_sub, add_right_comm, add_sub_cancel_right, add_self_div_two] at s1
  /-
    x y : Complex
    s1 : Eq (Complex.sin x) (HAdd.hAdd (HMul.hMul (Complex.sin (HDiv.hDiv (HAdd.hA …
    s2 : Eq (Complex.sin (HSub.hSub (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    ⊢ Eq (HSub.hSub (Complex.sin x) (Complex.sin y)) (HMul.hMul (HMul.hMul 2 (Comp …
  -/
  rw [div_sub_div_same, ← sub_add, add_sub_cancel_left, add_self_div_two] at s2
  /-
    x y : Complex
    s1 : Eq (Complex.sin x) (HAdd.hAdd (HMul.hMul (Complex.sin (HDiv.hDiv (HAdd.hA …
    s2 : Eq (Complex.sin y) (HSub.hSub (HMul.hMul (Complex.sin (HDiv.hDiv (HAdd.hA …
    ⊢ Eq (HSub.hSub (Complex.sin x) (Complex.sin y)) (HMul.hMul (HMul.hMul 2 (Comp …
  -/
  rw [s1, s2]
  /-
    x y : Complex
    s1 : Eq (Complex.sin x) (HAdd.hAdd (HMul.hMul (Complex.sin (HDiv.hDiv (HAdd.hA …
    s2 : Eq (Complex.sin y) (HSub.hSub (HMul.hMul (Complex.sin (HDiv.hDiv (HAdd.hA …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Complex.sin (HDiv.hDiv (HAdd.hAdd x y)  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem cos_sub_cos : cos x - cos y = -2 * sin ((x + y) / 2) * sin ((x - y) / 2) := by
  /-
    x y : Complex
    ⊢ Eq (HSub.hSub (Complex.cos x) (Complex.cos y)) (HMul.hMul (HMul.hMul (-2) (C …
  -/
  have s1 := cos_add ((x + y) / 2) ((x - y) / 2)
  /-
    x y : Complex
    s1 : Eq (Complex.cos (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    ⊢ Eq (HSub.hSub (Complex.cos x) (Complex.cos y)) (HMul.hMul (HMul.hMul (-2) (C …
  -/
  have s2 := cos_sub ((x + y) / 2) ((x - y) / 2)
  /-
    x y : Complex
    s1 : Eq (Complex.cos (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    s2 : Eq (Complex.cos (HSub.hSub (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    ⊢ Eq (HSub.hSub (Complex.cos x) (Complex.cos y)) (HMul.hMul (HMul.hMul (-2) (C …
  -/
  rw [div_add_div_same, add_sub, add_right_comm, add_sub_cancel_right, add_self_div_two] at s1
  /-
    x y : Complex
    s1 : Eq (Complex.cos x) (HSub.hSub (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hA …
    s2 : Eq (Complex.cos (HSub.hSub (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hDiv (HSub …
    ⊢ Eq (HSub.hSub (Complex.cos x) (Complex.cos y)) (HMul.hMul (HMul.hMul (-2) (C …
  -/
  rw [div_sub_div_same, ← sub_add, add_sub_cancel_left, add_self_div_two] at s2
  /-
    x y : Complex
    s1 : Eq (Complex.cos x) (HSub.hSub (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hA …
    s2 : Eq (Complex.cos y) (HAdd.hAdd (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hA …
    ⊢ Eq (HSub.hSub (Complex.cos x) (Complex.cos y)) (HMul.hMul (HMul.hMul (-2) (C …
  -/
  rw [s1, s2]
  /-
    x y : Complex
    s1 : Eq (Complex.cos x) (HSub.hSub (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hA …
    s2 : Eq (Complex.cos y) (HAdd.hAdd (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hA …
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hAdd x y)  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem sin_add_sin : sin x + sin y = 2 * sin ((x + y) / 2) * cos ((x - y) / 2) := by
  /-
    x y : Complex
    ⊢ Eq (HAdd.hAdd (Complex.sin x) (Complex.sin y)) (HMul.hMul (HMul.hMul 2 (Comp …
  -/
  simpa using sin_sub_sin x (-y)
  /-
    🎉 no goals
  -/


theorem cos_add_cos : cos x + cos y = 2 * cos ((x + y) / 2) * cos ((x - y) / 2) := by
  calc
    cos x + cos y = cos ((x + y) / 2 + (x - y) / 2) + cos ((x + y) / 2 - (x - y) / 2) := ?_
    _ =
        cos ((x + y) / 2) * cos ((x - y) / 2) - sin ((x + y) / 2) * sin ((x - y) / 2) +
          (cos ((x + y) / 2) * cos ((x - y) / 2) + sin ((x + y) / 2) * sin ((x - y) / 2)) :=
      ?_
    _ = 2 * cos ((x + y) / 2) * cos ((x - y) / 2) := ?_

    /-
      case calc_1
      x y : Complex
      ⊢ Eq (HAdd.hAdd (Complex.cos x) (Complex.cos y)) (HAdd.hAdd (Complex.cos (HAdd …
    -/
              /-
                🎉 no goals
              -/
  · congr <;> field_simp
              /-
                🎉 no goals
              -/
    /-
      case calc_2
      x y : Complex
      ⊢ Eq (HAdd.hAdd (Complex.cos (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd x y) 2) (HDiv.hD …
    -/
  · rw [cos_add, cos_sub]
    /-
      🎉 no goals
    -/
  /-
    case calc_3
    x y : Complex
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (Complex.cos (HDiv.hDiv (HAdd.hAdd x y)  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem sin_conj : sin (conj x) = conj (sin x) := by
  rw [← mul_left_inj' I_ne_zero, ← sinh_mul_I, ← conj_neg_I, ← RingHom.map_mul, ← RingHom.map_mul,
    sinh_conj, mul_neg, sinh_neg, sinh_mul_I, mul_neg]


@[simp]
theorem ofReal_sin_ofReal_re (x : ℝ) : ((sin x).re : ℂ) = sin x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.sin ↑x)) (Complex.sin ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← sin_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_sin (x : ℝ) : (Real.sin x : ℂ) = sin x :=
  ofReal_sin_ofReal_re _


@[simp]
                                                     /-
                                                       x : Real
                                                       ⊢ Eq (Complex.sin ↑x).im 0
                                                     -/
theorem sin_ofReal_im (x : ℝ) : (sin x).im = 0 := by rw [← ofReal_sin_ofReal_re, ofReal_im]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem sin_ofReal_re (x : ℝ) : (sin x).re = Real.sin x :=
  rfl


theorem cos_conj : cos (conj x) = conj (cos x) := by
  /-
    x : Complex
    ⊢ Eq (Complex.cos ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex.c …
  -/
  rw [← cosh_mul_I, ← conj_neg_I, ← RingHom.map_mul, ← cosh_mul_I, cosh_conj, mul_neg, cosh_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofReal_cos_ofReal_re (x : ℝ) : ((cos x).re : ℂ) = cos x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.cos ↑x)) (Complex.cos ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← cos_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_cos (x : ℝ) : (Real.cos x : ℂ) = cos x :=
  ofReal_cos_ofReal_re _


@[simp]
                                                     /-
                                                       x : Real
                                                       ⊢ Eq (Complex.cos ↑x).im 0
                                                     -/
theorem cos_ofReal_im (x : ℝ) : (cos x).im = 0 := by rw [← ofReal_cos_ofReal_re, ofReal_im]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem cos_ofReal_re (x : ℝ) : (cos x).re = Real.cos x :=
  rfl


@[simp]
                                   /-
                                     ⊢ Eq (Complex.tan 0) 0
                                   -/
theorem tan_zero : tan 0 = 0 := by simp [tan]
                                   /-
                                     🎉 no goals
                                   -/


theorem tan_eq_sin_div_cos : tan x = sin x / cos x :=
  rfl


theorem cot_eq_cos_div_sin : cot x = cos x / sin x :=
  rfl


theorem tan_mul_cos {x : ℂ} (hx : cos x ≠ 0) : tan x * cos x = sin x := by
  /-
    x : Complex
    hx : Ne (Complex.cos x) 0
    ⊢ Eq (HMul.hMul (Complex.tan x) (Complex.cos x)) (Complex.sin x)
  -/
  rw [tan_eq_sin_div_cos, div_mul_cancel₀ _ hx]
  /-
    🎉 no goals
  -/


@[simp]
                                          /-
                                            x : Complex
                                            ⊢ Eq (Complex.tan (Neg.neg x)) (Neg.neg (Complex.tan x))
                                          -/
theorem tan_neg : tan (-x) = -tan x := by simp [tan, neg_div]
                                          /-
                                            🎉 no goals
                                          -/


                                                     /-
                                                       x : Complex
                                                       ⊢ Eq (Complex.tan ((starRingEnd Complex) x)) ((starRingEnd Complex) (Complex.t …
                                                     -/
theorem tan_conj : tan (conj x) = conj (tan x) := by rw [tan, sin_conj, cos_conj, ← map_div₀, tan]
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                     /-
                                                       x : Complex
                                                       ⊢ Eq ((starRingEnd Complex) x).cot ((starRingEnd Complex) x.cot)
                                                     -/
theorem cot_conj : cot (conj x) = conj (cot x) := by rw [cot, sin_conj, cos_conj, ← map_div₀, cot]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem ofReal_tan_ofReal_re (x : ℝ) : ((tan x).re : ℂ) = tan x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (Complex.tan ↑x)) (Complex.tan ↑x)
                         -/
  conj_eq_iff_re.1 <| by rw [← tan_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem ofReal_cot_ofReal_re (x : ℝ) : ((cot x).re : ℂ) = cot x :=
                         /-
                           x : Real
                           ⊢ Eq ((starRingEnd Complex) (↑x).cot) (↑x).cot
                         -/
  conj_eq_iff_re.1 <| by rw [← cot_conj, conj_ofReal]
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast]
theorem ofReal_tan (x : ℝ) : (Real.tan x : ℂ) = tan x :=
  ofReal_tan_ofReal_re _


@[simp, norm_cast]
theorem ofReal_cot (x : ℝ) : (Real.cot x : ℂ) = cot x :=
  ofReal_cot_ofReal_re _


@[simp]
                                                     /-
                                                       x : Real
                                                       ⊢ Eq (Complex.tan ↑x).im 0
                                                     -/
theorem tan_ofReal_im (x : ℝ) : (tan x).im = 0 := by rw [← ofReal_tan_ofReal_re, ofReal_im]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem tan_ofReal_re (x : ℝ) : (tan x).re = Real.tan x :=
  rfl


theorem cos_add_sin_I : cos x + sin x * I = exp (x * I) := by
  /-
    x : Complex
    ⊢ Eq (HAdd.hAdd (Complex.cos x) (HMul.hMul (Complex.sin x) Complex.I)) (Comple …
  -/
  rw [← cosh_add_sinh, sinh_mul_I, cosh_mul_I]
  /-
    🎉 no goals
  -/


theorem cos_sub_sin_I : cos x - sin x * I = exp (-x * I) := by
  /-
    x : Complex
    ⊢ Eq (HSub.hSub (Complex.cos x) (HMul.hMul (Complex.sin x) Complex.I)) (Comple …
  -/
  rw [neg_mul, ← cosh_sub_sinh, sinh_mul_I, cosh_mul_I]
  /-
    🎉 no goals
  -/


@[simp]
theorem sin_sq_add_cos_sq : sin x ^ 2 + cos x ^ 2 = 1 :=
               /-
                 x : Complex
                 ⊢ Eq (HAdd.hAdd (HPow.hPow (Complex.sin x) 2) (HPow.hPow (Complex.cos x) 2)) ( …
               -/
  Eq.trans (by rw [cosh_mul_I, sinh_mul_I, mul_pow, I_sq, mul_neg_one, sub_neg_eq_add, add_comm])
               /-
                 🎉 no goals
               -/
    (cosh_sq_sub_sinh_sq (x * I))


@[simp]
                                                            /-
                                                              x : Complex
                                                              ⊢ Eq (HAdd.hAdd (HPow.hPow (Complex.cos x) 2) (HPow.hPow (Complex.sin x) 2)) 1
                                                            -/
theorem cos_sq_add_sin_sq : cos x ^ 2 + sin x ^ 2 = 1 := by rw [add_comm, sin_sq_add_cos_sq]
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                                 /-
                                                                   x : Complex
                                                                   ⊢ Eq (Complex.cos (HMul.hMul 2 x)) (HSub.hSub (HPow.hPow (Complex.cos x) 2) (H …
                                                                 -/
theorem cos_two_mul' : cos (2 * x) = cos x ^ 2 - sin x ^ 2 := by rw [two_mul, cos_add, ← sq, ← sq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem cos_two_mul : cos (2 * x) = 2 * cos x ^ 2 - 1 := by
  rw [cos_two_mul', eq_sub_iff_add_eq.2 (sin_sq_add_cos_sq x), ← sub_add, sub_add_eq_add_sub,
    two_mul]


theorem sin_two_mul : sin (2 * x) = 2 * sin x * cos x := by
  /-
    x : Complex
    ⊢ Eq (Complex.sin (HMul.hMul 2 x)) (HMul.hMul (HMul.hMul 2 (Complex.sin x)) (C …
  -/
  rw [two_mul, sin_add, two_mul, add_mul, mul_comm]
  /-
    🎉 no goals
  -/


theorem cos_sq : cos x ^ 2 = 1 / 2 + cos (2 * x) / 2 := by
  /-
    x : Complex
    ⊢ Eq (HPow.hPow (Complex.cos x) 2) (HAdd.hAdd (1 / 2) (HDiv.hDiv (Complex.cos  …
  -/
  simp [cos_two_mul, div_add_div_same, mul_div_cancel_left₀, two_ne_zero, -one_div]
  /-
    🎉 no goals
  -/


                                                  /-
                                                    x : Complex
                                                    ⊢ Eq (HPow.hPow (Complex.cos x) 2) (HSub.hSub 1 (HPow.hPow (Complex.sin x) 2))
                                                  -/
theorem cos_sq' : cos x ^ 2 = 1 - sin x ^ 2 := by rw [← sin_sq_add_cos_sq x, add_sub_cancel_left]
                                                  /-
                                                    🎉 no goals
                                                  -/


                                                 /-
                                                   x : Complex
                                                   ⊢ Eq (HPow.hPow (Complex.sin x) 2) (HSub.hSub 1 (HPow.hPow (Complex.cos x) 2))
                                                 -/
theorem sin_sq : sin x ^ 2 = 1 - cos x ^ 2 := by rw [← sin_sq_add_cos_sq x, add_sub_cancel_right]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem inv_one_add_tan_sq {x : ℂ} (hx : cos x ≠ 0) : (1 + tan x ^ 2)⁻¹ = cos x ^ 2 := by
  /-
    x : Complex
    hx : Ne (Complex.cos x) 0
    ⊢ Eq (Inv.inv (HAdd.hAdd 1 (HPow.hPow (Complex.tan x) 2))) (HPow.hPow (Complex …
  -/
  rw [tan_eq_sin_div_cos, div_pow]
  /-
    x : Complex
    hx : Ne (Complex.cos x) 0
    ⊢ Eq (Inv.inv (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow (Complex.sin x) 2) (HPow.hPow …
  -/
  field_simp
  /-
    🎉 no goals
  -/


theorem tan_sq_div_one_add_tan_sq {x : ℂ} (hx : cos x ≠ 0) :
    tan x ^ 2 / (1 + tan x ^ 2) = sin x ^ 2 := by
  /-
    x : Complex
    hx : Ne (Complex.cos x) 0
    ⊢ Eq (HDiv.hDiv (HPow.hPow (Complex.tan x) 2) (HAdd.hAdd 1 (HPow.hPow (Complex …
  -/
  simp only [← tan_mul_cos hx, mul_pow, ← inv_one_add_tan_sq hx, div_eq_mul_inv, one_mul]
  /-
    🎉 no goals
  -/


theorem cos_three_mul : cos (3 * x) = 4 * cos x ^ 3 - 3 * cos x := by
  /-
    x : Complex
    ⊢ Eq (Complex.cos (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 4 (HPow.hPow (Complex …
  -/
  have h1 : x + 2 * x = 3 * x := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (Complex.cos (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 4 (HPow.hPow (Complex …
  -/
  rw [← h1, cos_add x (2 * x)]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HSub.hSub (HMul.hMul (Complex.cos x) (Complex.cos (HMul.hMul 2 x))) (HMu …
  -/
  simp only [cos_two_mul, sin_two_mul, mul_add, mul_sub, mul_one, sq]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (Complex.cos x) (HMul.hMul 2 (HMul.hMul  …
  -/
  have h2 : 4 * cos x ^ 3 = 2 * cos x * cos x * cos x + 2 * cos x * cos x ^ 2 := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul 4 (HPow.hPow (Complex.cos x) 3)) (HAdd.hAdd (HMul.hMul (HMu …
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (Complex.cos x) (HMul.hMul 2 (HMul.hMul  …
  -/
  rw [h2, cos_sq']
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul 4 (HPow.hPow (Complex.cos x) 3)) (HAdd.hAdd (HMul.hMul (HMu …
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (Complex.cos x) (HMul.hMul 2 (HMul.hMul  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem sin_three_mul : sin (3 * x) = 3 * sin x - 4 * sin x ^ 3 := by
  /-
    x : Complex
    ⊢ Eq (Complex.sin (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 3 (Complex.sin x)) (H …
  -/
  have h1 : x + 2 * x = 3 * x := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (Complex.sin (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 3 (Complex.sin x)) (H …
  -/
  rw [← h1, sin_add x (2 * x)]
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sin x) (Complex.cos (HMul.hMul 2 x))) (HMu …
  -/
  simp only [cos_two_mul, sin_two_mul, cos_sq']
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sin x) (HSub.hSub (HMul.hMul 2 (HSub.hSub  …
  -/
  have h2 : cos x * (2 * sin x * cos x) = 2 * sin x * cos x ^ 2 := by ring
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul (Complex.cos x) (HMul.hMul (HMul.hMul 2 (Complex.sin x)) (C …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sin x) (HSub.hSub (HMul.hMul 2 (HSub.hSub  …
  -/
  rw [h2, cos_sq']
  /-
    x : Complex
    h1 : Eq (HAdd.hAdd x (HMul.hMul 2 x)) (HMul.hMul 3 x)
    h2 : Eq (HMul.hMul (Complex.cos x) (HMul.hMul (HMul.hMul 2 (Complex.sin x)) (C …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Complex.sin x) (HSub.hSub (HMul.hMul 2 (HSub.hSub  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem exp_mul_I : exp (x * I) = cos x + sin x * I :=
  (cos_add_sin_I _).symm


                                                                            /-
                                                                              x y : Complex
                                                                              ⊢ Eq (Complex.exp (HAdd.hAdd x (HMul.hMul y Complex.I))) (HMul.hMul (Complex.e …
                                                                            -/
theorem exp_add_mul_I : exp (x + y * I) = exp x * (cos y + sin y * I) := by rw [exp_add, exp_mul_I]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem exp_eq_exp_re_mul_sin_add_cos : exp x = exp x.re * (cos x.im + sin x.im * I) := by
  /-
    x : Complex
    ⊢ Eq (Complex.exp x) (HMul.hMul (Complex.exp ↑x.re) (HAdd.hAdd (Complex.cos ↑x …
  -/
  rw [← exp_add_mul_I, re_add_im]
  /-
    🎉 no goals
  -/


theorem exp_re : (exp x).re = Real.exp x.re * Real.cos x.im := by
  /-
    x : Complex
    ⊢ Eq (Complex.exp x).re (HMul.hMul (Real.exp x.re) (Real.cos x.im))
  -/
  rw [exp_eq_exp_re_mul_sin_add_cos]
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (Complex.exp ↑x.re) (HAdd.hAdd (Complex.cos ↑x.im) (HMul.hMul  …
  -/
  simp [exp_ofReal_re, cos_ofReal_re]
  /-
    🎉 no goals
  -/


theorem exp_im : (exp x).im = Real.exp x.re * Real.sin x.im := by
  /-
    x : Complex
    ⊢ Eq (Complex.exp x).im (HMul.hMul (Real.exp x.re) (Real.sin x.im))
  -/
  rw [exp_eq_exp_re_mul_sin_add_cos]
  /-
    x : Complex
    ⊢ Eq (HMul.hMul (Complex.exp ↑x.re) (HAdd.hAdd (Complex.cos ↑x.im) (HMul.hMul  …
  -/
  simp [exp_ofReal_re, sin_ofReal_re]
  /-
    🎉 no goals
  -/


@[simp]
theorem exp_ofReal_mul_I_re (x : ℝ) : (exp (x * I)).re = Real.cos x := by
  /-
    x : Real
    ⊢ Eq (Complex.exp (HMul.hMul (↑x) Complex.I)).re (Real.cos x)
  -/
  simp [exp_mul_I, cos_ofReal_re]
  /-
    🎉 no goals
  -/


@[simp]
theorem exp_ofReal_mul_I_im (x : ℝ) : (exp (x * I)).im = Real.sin x := by
  /-
    x : Real
    ⊢ Eq (Complex.exp (HMul.hMul (↑x) Complex.I)).im (Real.sin x)
  -/
  simp [exp_mul_I, sin_ofReal_re]
  /-
    🎉 no goals
  -/


/-- **De Moivre's formula** -/
theorem cos_add_sin_mul_I_pow (n : ℕ) (z : ℂ) :
    (cos z + sin z * I) ^ n = cos (↑n * z) + sin (↑n * z) * I := by
  /-
    n : Nat
    z : Complex
    ⊢ Eq (HPow.hPow (HAdd.hAdd (Complex.cos z) (HMul.hMul (Complex.sin z) Complex. …
  -/
  rw [← exp_mul_I, ← exp_mul_I]
  /-
    n : Nat
    z : Complex
    ⊢ Eq (HPow.hPow (Complex.exp (HMul.hMul z Complex.I)) n) (Complex.exp (HMul.hM …
  -/
  induction' n with n ih
    /-
      case zero
      z : Complex
      ⊢ Eq (HPow.hPow (Complex.exp (HMul.hMul z Complex.I)) 0) (Complex.exp (HMul.hM …
    -/
  · rw [pow_zero, Nat.cast_zero, zero_mul, zero_mul, exp_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      z : Complex
      n : Nat
      ih : Eq (HPow.hPow (Complex.exp (HMul.hMul z Complex.I)) n) (Complex.exp (HMul …
      ⊢ Eq (HPow.hPow (Complex.exp (HMul.hMul z Complex.I)) (HAdd.hAdd n 1)) (Comple …
    -/
  · rw [pow_succ, ih, Nat.cast_succ, add_mul, add_mul, one_mul, exp_add]
    /-
      🎉 no goals
    -/


@[simp]
                                   /-
                                     ⊢ Eq (Real.exp 0) 1
                                   -/
theorem exp_zero : exp 0 = 1 := by simp [Real.exp]
                                   /-
                                     🎉 no goals
                                   -/


                                                           /-
                                                             x y : Real
                                                             ⊢ Eq (Real.exp (HAdd.hAdd x y)) (HMul.hMul (Real.exp x) (Real.exp y))
                                                           -/
nonrec theorem exp_add : exp (x + y) = exp x * exp y := by simp [exp_add, exp]
                                                           /-
                                                             🎉 no goals
                                                           -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): new definition

/-- the exponential function as a monoid hom from `Multiplicative ℝ` to `ℝ` -/
@[simps]
noncomputable def expMonoidHom : MonoidHom (Multiplicative ℝ) ℝ :=
  { toFun := fun x => exp x.toAdd,
                   /-
                     x y : Real
                     ⊢ Eq ((fun x => Real.exp (Multiplicative.toAdd x)) 1) 1
                   -/
    map_one' := by simp,
                   /-
                     🎉 no goals
                   -/
                   /-
                     x y : Real
                     ⊢ ∀ (x y : Multiplicative Real), Eq ({ toFun := fun x => Real.exp (Multiplicat …
                   -/
    map_mul' := by simp [exp_add] }
                   /-
                     🎉 no goals
                   -/


theorem exp_list_sum (l : List ℝ) : exp l.sum = (l.map exp).prod :=
  map_list_prod (M := Multiplicative ℝ) expMonoidHom l


theorem exp_multiset_sum (s : Multiset ℝ) : exp s.sum = (s.map exp).prod :=
  @MonoidHom.map_multiset_prod (Multiplicative ℝ) ℝ _ _ expMonoidHom s


theorem exp_sum {α : Type*} (s : Finset α) (f : α → ℝ) :
    exp (∑ x ∈ s, f x) = ∏ x ∈ s, exp (f x) :=
  map_prod (β := Multiplicative ℝ) expMonoidHom f s


lemma exp_nsmul (x : ℝ) (n : ℕ) : exp (n • x) = exp x ^ n :=
  @MonoidHom.map_pow (Multiplicative ℝ) ℝ _ _  expMonoidHom _ _


nonrec theorem exp_nat_mul (x : ℝ) (n : ℕ) : exp (n * x) = exp x ^ n :=
                       /-
                         x : Real
                         n : Nat
                         ⊢ Eq ↑(Real.exp (HMul.hMul (↑n) x)) ↑(HPow.hPow (Real.exp x) n)
                       -/
  ofReal_injective (by simp [exp_nat_mul])
                       /-
                         🎉 no goals
                       -/


@[simp]
nonrec theorem exp_ne_zero : exp x ≠ 0 := fun h =>
                      /-
                        x : Real
                        h : Eq (Real.exp x) 0
                        ⊢ Eq (Complex.exp ↑x) 0
                      -/
  exp_ne_zero x <| by rw [exp, ← ofReal_inj] at h; simp_all
                                                   /-
                                                     🎉 no goals
                                                   -/


nonrec theorem exp_neg : exp (-x) = (exp x)⁻¹ :=
                         /-
                           x : Real
                           ⊢ Eq ↑(Real.exp (Neg.neg x)) ↑(Inv.inv (Real.exp x))
                         -/
  ofReal_injective <| by simp [exp_neg]
                         /-
                           🎉 no goals
                         -/


theorem exp_sub : exp (x - y) = exp x / exp y := by
  /-
    x y : Real
    ⊢ Eq (Real.exp (HSub.hSub x y)) (HDiv.hDiv (Real.exp x) (Real.exp y))
  -/
  simp [sub_eq_add_neg, exp_add, exp_neg, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp]
                                   /-
                                     ⊢ Eq (Real.sin 0) 0
                                   -/
theorem sin_zero : sin 0 = 0 := by simp [sin]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
                                          /-
                                            x : Real
                                            ⊢ Eq (Real.sin (Neg.neg x)) (Neg.neg (Real.sin x))
                                          -/
theorem sin_neg : sin (-x) = -sin x := by simp [sin, exp_neg, (neg_div _ _).symm, add_mul]
                                          /-
                                            🎉 no goals
                                          -/


nonrec theorem sin_add : sin (x + y) = sin x * cos y + cos x * sin y :=
                         /-
                           x y : Real
                           ⊢ Eq ↑(Real.sin (HAdd.hAdd x y)) ↑(HAdd.hAdd (HMul.hMul (Real.sin x) (Real.cos …
                         -/
  ofReal_injective <| by simp [sin_add]
                         /-
                           🎉 no goals
                         -/


@[simp]
                                   /-
                                     ⊢ Eq (Real.cos 0) 1
                                   -/
theorem cos_zero : cos 0 = 1 := by simp [cos]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
                                         /-
                                           x : Real
                                           ⊢ Eq (Real.cos (Neg.neg x)) (Real.cos x)
                                         -/
theorem cos_neg : cos (-x) = cos x := by simp [cos, exp_neg]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem cos_abs : cos |x| = cos x := by
  /-
    x : Real
    ⊢ Eq (Real.cos (abs x)) (Real.cos x)
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total x 0 <;> simp only [*, _root_.abs_of_nonneg, abs_of_nonpos, cos_neg]
                         /-
                           🎉 no goals
                         -/


nonrec theorem cos_add : cos (x + y) = cos x * cos y - sin x * sin y :=
                         /-
                           x y : Real
                           ⊢ Eq ↑(Real.cos (HAdd.hAdd x y)) ↑(HSub.hSub (HMul.hMul (Real.cos x) (Real.cos …
                         -/
  ofReal_injective <| by simp [cos_add]
                         /-
                           🎉 no goals
                         -/


theorem sin_sub : sin (x - y) = sin x * cos y - cos x * sin y := by
  /-
    x y : Real
    ⊢ Eq (Real.sin (HSub.hSub x y)) (HSub.hSub (HMul.hMul (Real.sin x) (Real.cos y …
  -/
  simp [sub_eq_add_neg, sin_add, sin_neg, cos_neg]
  /-
    🎉 no goals
  -/


theorem cos_sub : cos (x - y) = cos x * cos y + sin x * sin y := by
  /-
    x y : Real
    ⊢ Eq (Real.cos (HSub.hSub x y)) (HAdd.hAdd (HMul.hMul (Real.cos x) (Real.cos y …
  -/
  simp [sub_eq_add_neg, cos_add, sin_neg, cos_neg]
  /-
    🎉 no goals
  -/


nonrec theorem sin_sub_sin : sin x - sin y = 2 * sin ((x - y) / 2) * cos ((x + y) / 2) :=
                         /-
                           x y : Real
                           ⊢ Eq ↑(HSub.hSub (Real.sin x) (Real.sin y)) ↑(HMul.hMul (HMul.hMul 2 (Real.sin …
                         -/
  ofReal_injective <| by simp [sin_sub_sin]
                         /-
                           🎉 no goals
                         -/


nonrec theorem cos_sub_cos : cos x - cos y = -2 * sin ((x + y) / 2) * sin ((x - y) / 2) :=
                         /-
                           x y : Real
                           ⊢ Eq ↑(HSub.hSub (Real.cos x) (Real.cos y)) ↑(HMul.hMul (HMul.hMul (-2) (Real. …
                         -/
  ofReal_injective <| by simp [cos_sub_cos]
                         /-
                           🎉 no goals
                         -/


nonrec theorem cos_add_cos : cos x + cos y = 2 * cos ((x + y) / 2) * cos ((x - y) / 2) :=
                         /-
                           x y : Real
                           ⊢ Eq ↑(HAdd.hAdd (Real.cos x) (Real.cos y)) ↑(HMul.hMul (HMul.hMul 2 (Real.cos …
                         -/
  ofReal_injective <| by simp [cos_add_cos]
                         /-
                           🎉 no goals
                         -/


theorem two_mul_sin_mul_sin (x y : ℝ) : 2 * sin x * sin y = cos (x - y) - cos (x + y) := by
  /-
    x y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.sin x)) (Real.sin y)) (HSub.hSub (Real.cos  …
  -/
  simp [cos_add, cos_sub]
  /-
    x y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.sin x)) (Real.sin y)) (HAdd.hAdd (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem two_mul_cos_mul_cos (x y : ℝ) : 2 * cos x * cos y = cos (x - y) + cos (x + y) := by
  /-
    x y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.cos x)) (Real.cos y)) (HAdd.hAdd (Real.cos  …
  -/
  simp [cos_add, cos_sub]
  /-
    x y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.cos x)) (Real.cos y)) (HAdd.hAdd (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem two_mul_sin_mul_cos (x y : ℝ) : 2 * sin x * cos y = sin (x - y) + sin (x + y) := by
  /-
    x y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.sin x)) (Real.cos y)) (HAdd.hAdd (Real.sin  …
  -/
  simp [sin_add, sin_sub]
  /-
    x y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.sin x)) (Real.cos y)) (HAdd.hAdd (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


nonrec theorem tan_eq_sin_div_cos : tan x = sin x / cos x :=
  ofReal_injective <| by simp only [ofReal_tan, tan_eq_sin_div_cos, ofReal_div, ofReal_sin,
    ofReal_cos]


nonrec theorem cot_eq_cos_div_sin : cot x = cos x / sin x :=
                         /-
                           x : Real
                           ⊢ Eq ↑x.cot ↑(HDiv.hDiv (Real.cos x) (Real.sin x))
                         -/
  ofReal_injective <| by simp [cot_eq_cos_div_sin]
                         /-
                           🎉 no goals
                         -/


theorem tan_mul_cos {x : ℝ} (hx : cos x ≠ 0) : tan x * cos x = sin x := by
  /-
    x : Real
    hx : Ne (Real.cos x) 0
    ⊢ Eq (HMul.hMul (Real.tan x) (Real.cos x)) (Real.sin x)
  -/
  rw [tan_eq_sin_div_cos, div_mul_cancel₀ _ hx]
  /-
    🎉 no goals
  -/


@[simp]
                                   /-
                                     ⊢ Eq (Real.tan 0) 0
                                   -/
theorem tan_zero : tan 0 = 0 := by simp [tan]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
                                          /-
                                            x : Real
                                            ⊢ Eq (Real.tan (Neg.neg x)) (Neg.neg (Real.tan x))
                                          -/
theorem tan_neg : tan (-x) = -tan x := by simp [tan, neg_div]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
nonrec theorem sin_sq_add_cos_sq : sin x ^ 2 + cos x ^ 2 = 1 :=
                       /-
                         x : Real
                         ⊢ Eq ↑(HAdd.hAdd (HPow.hPow (Real.sin x) 2) (HPow.hPow (Real.cos x) 2)) ↑1
                       -/
  ofReal_injective (by simp [sin_sq_add_cos_sq])
                       /-
                         🎉 no goals
                       -/


@[simp]
                                                            /-
                                                              x : Real
                                                              ⊢ Eq (HAdd.hAdd (HPow.hPow (Real.cos x) 2) (HPow.hPow (Real.sin x) 2)) 1
                                                            -/
theorem cos_sq_add_sin_sq : cos x ^ 2 + sin x ^ 2 = 1 := by rw [add_comm, sin_sq_add_cos_sq]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem sin_sq_le_one : sin x ^ 2 ≤ 1 := by
  /-
    x : Real
    ⊢ LE.le (HPow.hPow (Real.sin x) 2) 1
  -/
  rw [← sin_sq_add_cos_sq x]; exact le_add_of_nonneg_right (sq_nonneg _)
                              /-
                                🎉 no goals
                              -/


theorem cos_sq_le_one : cos x ^ 2 ≤ 1 := by
  /-
    x : Real
    ⊢ LE.le (HPow.hPow (Real.cos x) 2) 1
  -/
  rw [← sin_sq_add_cos_sq x]; exact le_add_of_nonneg_left (sq_nonneg _)
                              /-
                                🎉 no goals
                              -/


theorem abs_sin_le_one : |sin x| ≤ 1 :=
                                         /-
                                           x : Real
                                           ⊢ LE.le (HMul.hMul (Real.sin x) (Real.sin x)) 1
                                         -/
  abs_le_one_iff_mul_self_le_one.2 <| by simp only [← sq, sin_sq_le_one]
                                         /-
                                           🎉 no goals
                                         -/


theorem abs_cos_le_one : |cos x| ≤ 1 :=
                                         /-
                                           x : Real
                                           ⊢ LE.le (HMul.hMul (Real.cos x) (Real.cos x)) 1
                                         -/
  abs_le_one_iff_mul_self_le_one.2 <| by simp only [← sq, cos_sq_le_one]
                                         /-
                                           🎉 no goals
                                         -/


theorem sin_le_one : sin x ≤ 1 :=
  (abs_le.1 (abs_sin_le_one _)).2


theorem cos_le_one : cos x ≤ 1 :=
  (abs_le.1 (abs_cos_le_one _)).2


theorem neg_one_le_sin : -1 ≤ sin x :=
  (abs_le.1 (abs_sin_le_one _)).1


theorem neg_one_le_cos : -1 ≤ cos x :=
  (abs_le.1 (abs_cos_le_one _)).1


nonrec theorem cos_two_mul : cos (2 * x) = 2 * cos x ^ 2 - 1 :=
                         /-
                           x : Real
                           ⊢ Eq ↑(Real.cos (HMul.hMul 2 x)) ↑(HSub.hSub (HMul.hMul 2 (HPow.hPow (Real.cos …
                         -/
  ofReal_injective <| by simp [cos_two_mul]
                         /-
                           🎉 no goals
                         -/


nonrec theorem cos_two_mul' : cos (2 * x) = cos x ^ 2 - sin x ^ 2 :=
                         /-
                           x : Real
                           ⊢ Eq ↑(Real.cos (HMul.hMul 2 x)) ↑(HSub.hSub (HPow.hPow (Real.cos x) 2) (HPow. …
                         -/
  ofReal_injective <| by simp [cos_two_mul']
                         /-
                           🎉 no goals
                         -/


nonrec theorem sin_two_mul : sin (2 * x) = 2 * sin x * cos x :=
                         /-
                           x : Real
                           ⊢ Eq ↑(Real.sin (HMul.hMul 2 x)) ↑(HMul.hMul (HMul.hMul 2 (Real.sin x)) (Real. …
                         -/
  ofReal_injective <| by simp [sin_two_mul]
                         /-
                           🎉 no goals
                         -/


nonrec theorem cos_sq : cos x ^ 2 = 1 / 2 + cos (2 * x) / 2 :=
                         /-
                           x : Real
                           ⊢ Eq ↑(HPow.hPow (Real.cos x) 2) ↑(HAdd.hAdd (1 / 2) (HDiv.hDiv (Real.cos (HMu …
                         -/
  ofReal_injective <| by simp [cos_sq]
                         /-
                           🎉 no goals
                         -/


                                                  /-
                                                    x : Real
                                                    ⊢ Eq (HPow.hPow (Real.cos x) 2) (HSub.hSub 1 (HPow.hPow (Real.sin x) 2))
                                                  -/
theorem cos_sq' : cos x ^ 2 = 1 - sin x ^ 2 := by rw [← sin_sq_add_cos_sq x, add_sub_cancel_left]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem sin_sq : sin x ^ 2 = 1 - cos x ^ 2 :=
  eq_sub_iff_add_eq.2 <| sin_sq_add_cos_sq _


lemma sin_sq_eq_half_sub : sin x ^ 2 = 1 / 2 - cos (2 * x) / 2 := by
  /-
    x : Real
    ⊢ Eq (HPow.hPow (Real.sin x) 2) (HSub.hSub (1 / 2) (HDiv.hDiv (Real.cos (HMul. …
  -/
  rw [sin_sq, cos_sq, ← sub_sub, sub_half]
  /-
    🎉 no goals
  -/


theorem abs_sin_eq_sqrt_one_sub_cos_sq (x : ℝ) : |sin x| = √(1 - cos x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (abs (Real.sin x)) (HSub.hSub 1 (HPow.hPow (Real.cos x) 2)).sqrt
  -/
  rw [← sin_sq, sqrt_sq_eq_abs]
  /-
    🎉 no goals
  -/


theorem abs_cos_eq_sqrt_one_sub_sin_sq (x : ℝ) : |cos x| = √(1 - sin x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (abs (Real.cos x)) (HSub.hSub 1 (HPow.hPow (Real.sin x) 2)).sqrt
  -/
  rw [← cos_sq', sqrt_sq_eq_abs]
  /-
    🎉 no goals
  -/


theorem inv_one_add_tan_sq {x : ℝ} (hx : cos x ≠ 0) : (1 + tan x ^ 2)⁻¹ = cos x ^ 2 :=
  have : Complex.cos x ≠ 0 := mt (congr_arg re) hx
                     /-
                       x : Real
                       hx : Ne (Real.cos x) 0
                       this : Ne (Complex.cos ↑x) 0
                       ⊢ Eq ↑(Inv.inv (HAdd.hAdd 1 (HPow.hPow (Real.tan x) 2))) ↑(HPow.hPow (Real.cos …
                     -/
  ofReal_inj.1 <| by simpa using Complex.inv_one_add_tan_sq this
                     /-
                       🎉 no goals
                     -/


theorem tan_sq_div_one_add_tan_sq {x : ℝ} (hx : cos x ≠ 0) :
    tan x ^ 2 / (1 + tan x ^ 2) = sin x ^ 2 := by
  /-
    x : Real
    hx : Ne (Real.cos x) 0
    ⊢ Eq (HDiv.hDiv (HPow.hPow (Real.tan x) 2) (HAdd.hAdd 1 (HPow.hPow (Real.tan x …
  -/
  simp only [← tan_mul_cos hx, mul_pow, ← inv_one_add_tan_sq hx, div_eq_mul_inv, one_mul]
  /-
    🎉 no goals
  -/


theorem inv_sqrt_one_add_tan_sq {x : ℝ} (hx : 0 < cos x) : (√(1 + tan x ^ 2))⁻¹ = cos x := by
  /-
    x : Real
    hx : LT.lt 0 (Real.cos x)
    ⊢ Eq (Inv.inv (HAdd.hAdd 1 (HPow.hPow (Real.tan x) 2)).sqrt) (Real.cos x)
  -/
  rw [← sqrt_sq hx.le, ← sqrt_inv, inv_one_add_tan_sq hx.ne']
  /-
    🎉 no goals
  -/


theorem tan_div_sqrt_one_add_tan_sq {x : ℝ} (hx : 0 < cos x) :
    tan x / √(1 + tan x ^ 2) = sin x := by
  /-
    x : Real
    hx : LT.lt 0 (Real.cos x)
    ⊢ Eq (HDiv.hDiv (Real.tan x) (HAdd.hAdd 1 (HPow.hPow (Real.tan x) 2)).sqrt) (R …
  -/
  rw [← tan_mul_cos hx.ne', ← inv_sqrt_one_add_tan_sq hx, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


nonrec theorem cos_three_mul : cos (3 * x) = 4 * cos x ^ 3 - 3 * cos x := by
  /-
    x : Real
    ⊢ Eq (Real.cos (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 4 (HPow.hPow (Real.cos x …
  -/
  rw [← ofReal_inj]; simp [cos_three_mul]
                     /-
                       🎉 no goals
                     -/


nonrec theorem sin_three_mul : sin (3 * x) = 3 * sin x - 4 * sin x ^ 3 := by
  /-
    x : Real
    ⊢ Eq (Real.sin (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 3 (Real.sin x)) (HMul.hM …
  -/
  rw [← ofReal_inj]; simp [sin_three_mul]
                     /-
                       🎉 no goals
                     -/


/-- The definition of `sinh` in terms of `exp`. -/
nonrec theorem sinh_eq (x : ℝ) : sinh x = (exp x - exp (-x)) / 2 :=
                         /-
                           x : Real
                           ⊢ Eq ↑(Real.sinh x) ↑(HDiv.hDiv (HSub.hSub (Real.exp x) (Real.exp (Neg.neg x)) …
                         -/
  ofReal_injective <| by simp [Complex.sinh]
                         /-
                           🎉 no goals
                         -/


@[simp]
                                     /-
                                       ⊢ Eq (Real.sinh 0) 0
                                     -/
theorem sinh_zero : sinh 0 = 0 := by simp [sinh]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                             /-
                                               x : Real
                                               ⊢ Eq (Real.sinh (Neg.neg x)) (Neg.neg (Real.sinh x))
                                             -/
theorem sinh_neg : sinh (-x) = -sinh x := by simp [sinh, exp_neg, (neg_div _ _).symm, add_mul]
                                             /-
                                               🎉 no goals
                                             -/


nonrec theorem sinh_add : sinh (x + y) = sinh x * cosh y + cosh x * sinh y := by
  /-
    x y : Real
    ⊢ Eq (Real.sinh (HAdd.hAdd x y)) (HAdd.hAdd (HMul.hMul (Real.sinh x) (Real.cos …
  -/
  rw [← ofReal_inj]; simp [sinh_add]
                     /-
                       🎉 no goals
                     -/


/-- The definition of `cosh` in terms of `exp`. -/
theorem cosh_eq (x : ℝ) : cosh x = (exp x + exp (-x)) / 2 :=
  eq_div_of_mul_eq two_ne_zero <| by
    rw [cosh, exp, exp, Complex.ofReal_neg, Complex.cosh, mul_two, ← Complex.add_re, ← mul_two,
      div_mul_cancel₀ _ (two_ne_zero' ℂ), Complex.add_re]


@[simp]
                                     /-
                                       ⊢ Eq (Real.cosh 0) 1
                                     -/
theorem cosh_zero : cosh 0 = 1 := by simp [cosh]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem cosh_neg : cosh (-x) = cosh x :=
                     /-
                       x : Real
                       ⊢ Eq ↑(Real.cosh (Neg.neg x)) ↑(Real.cosh x)
                     -/
  ofReal_inj.1 <| by simp
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem cosh_abs : cosh |x| = cosh x := by
  /-
    x : Real
    ⊢ Eq (Real.cosh (abs x)) (Real.cosh x)
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total x 0 <;> simp [*, _root_.abs_of_nonneg, abs_of_nonpos]
                         /-
                           🎉 no goals
                         -/


nonrec theorem cosh_add : cosh (x + y) = cosh x * cosh y + sinh x * sinh y := by
  /-
    x y : Real
    ⊢ Eq (Real.cosh (HAdd.hAdd x y)) (HAdd.hAdd (HMul.hMul (Real.cosh x) (Real.cos …
  -/
  rw [← ofReal_inj]; simp [cosh_add]
                     /-
                       🎉 no goals
                     -/


theorem sinh_sub : sinh (x - y) = sinh x * cosh y - cosh x * sinh y := by
  /-
    x y : Real
    ⊢ Eq (Real.sinh (HSub.hSub x y)) (HSub.hSub (HMul.hMul (Real.sinh x) (Real.cos …
  -/
  simp [sub_eq_add_neg, sinh_add, sinh_neg, cosh_neg]
  /-
    🎉 no goals
  -/


theorem cosh_sub : cosh (x - y) = cosh x * cosh y - sinh x * sinh y := by
  /-
    x y : Real
    ⊢ Eq (Real.cosh (HSub.hSub x y)) (HSub.hSub (HMul.hMul (Real.cosh x) (Real.cos …
  -/
  simp [sub_eq_add_neg, cosh_add, sinh_neg, cosh_neg]
  /-
    🎉 no goals
  -/


nonrec theorem tanh_eq_sinh_div_cosh : tanh x = sinh x / cosh x :=
                     /-
                       x : Real
                       ⊢ Eq ↑(Real.tanh x) ↑(HDiv.hDiv (Real.sinh x) (Real.cosh x))
                     -/
  ofReal_inj.1 <| by simp [tanh_eq_sinh_div_cosh]
                     /-
                       🎉 no goals
                     -/


@[simp]
                                     /-
                                       ⊢ Eq (Real.tanh 0) 0
                                     -/
theorem tanh_zero : tanh 0 = 0 := by simp [tanh]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                             /-
                                               x : Real
                                               ⊢ Eq (Real.tanh (Neg.neg x)) (Neg.neg (Real.tanh x))
                                             -/
theorem tanh_neg : tanh (-x) = -tanh x := by simp [tanh, neg_div]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                                      /-
                                                        x : Real
                                                        ⊢ Eq (HAdd.hAdd (Real.cosh x) (Real.sinh x)) (Real.exp x)
                                                      -/
theorem cosh_add_sinh : cosh x + sinh x = exp x := by rw [← ofReal_inj]; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
                                                      /-
                                                        x : Real
                                                        ⊢ Eq (HAdd.hAdd (Real.sinh x) (Real.cosh x)) (Real.exp x)
                                                      -/
theorem sinh_add_cosh : sinh x + cosh x = exp x := by rw [add_comm, cosh_add_sinh]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem cosh_sub_sinh : cosh x - sinh x = exp (-x) := by
  /-
    x : Real
    ⊢ Eq (HSub.hSub (Real.cosh x) (Real.sinh x)) (Real.exp (Neg.neg x))
  -/
  rw [← ofReal_inj]
  /-
    x : Real
    ⊢ Eq ↑(HSub.hSub (Real.cosh x) (Real.sinh x)) ↑(Real.exp (Neg.neg x))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            x : Real
                                                            ⊢ Eq (HSub.hSub (Real.sinh x) (Real.cosh x)) (Neg.neg (Real.exp (Neg.neg x)))
                                                          -/
theorem sinh_sub_cosh : sinh x - cosh x = -exp (-x) := by rw [← neg_sub, cosh_sub_sinh]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                                        /-
                                                                          x : Real
                                                                          ⊢ Eq (HSub.hSub (HPow.hPow (Real.cosh x) 2) (HPow.hPow (Real.sinh x) 2)) 1
                                                                        -/
theorem cosh_sq_sub_sinh_sq (x : ℝ) : cosh x ^ 2 - sinh x ^ 2 = 1 := by rw [← ofReal_inj]; simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


                                                           /-
                                                             x : Real
                                                             ⊢ Eq (HPow.hPow (Real.cosh x) 2) (HAdd.hAdd (HPow.hPow (Real.sinh x) 2) 1)
                                                           -/
nonrec theorem cosh_sq : cosh x ^ 2 = sinh x ^ 2 + 1 := by rw [← ofReal_inj]; simp [cosh_sq]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem cosh_sq' : cosh x ^ 2 = 1 + sinh x ^ 2 :=
  (cosh_sq x).trans (add_comm _ _)


                                                           /-
                                                             x : Real
                                                             ⊢ Eq (HPow.hPow (Real.sinh x) 2) (HSub.hSub (HPow.hPow (Real.cosh x) 2) 1)
                                                           -/
nonrec theorem sinh_sq : sinh x ^ 2 = cosh x ^ 2 - 1 := by rw [← ofReal_inj]; simp [sinh_sq]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


nonrec theorem cosh_two_mul : cosh (2 * x) = cosh x ^ 2 + sinh x ^ 2 := by
  /-
    x : Real
    ⊢ Eq (Real.cosh (HMul.hMul 2 x)) (HAdd.hAdd (HPow.hPow (Real.cosh x) 2) (HPow. …
  -/
  rw [← ofReal_inj]; simp [cosh_two_mul]
                     /-
                       🎉 no goals
                     -/


nonrec theorem sinh_two_mul : sinh (2 * x) = 2 * sinh x * cosh x := by
  /-
    x : Real
    ⊢ Eq (Real.sinh (HMul.hMul 2 x)) (HMul.hMul (HMul.hMul 2 (Real.sinh x)) (Real. …
  -/
  rw [← ofReal_inj]; simp [sinh_two_mul]
                     /-
                       🎉 no goals
                     -/


nonrec theorem cosh_three_mul : cosh (3 * x) = 4 * cosh x ^ 3 - 3 * cosh x := by
  /-
    x : Real
    ⊢ Eq (Real.cosh (HMul.hMul 3 x)) (HSub.hSub (HMul.hMul 4 (HPow.hPow (Real.cosh …
  -/
  rw [← ofReal_inj]; simp [cosh_three_mul]
                     /-
                       🎉 no goals
                     -/


nonrec theorem sinh_three_mul : sinh (3 * x) = 4 * sinh x ^ 3 + 3 * sinh x := by
  /-
    x : Real
    ⊢ Eq (Real.sinh (HMul.hMul 3 x)) (HAdd.hAdd (HMul.hMul 4 (HPow.hPow (Real.sinh …
  -/
  rw [← ofReal_inj]; simp [sinh_three_mul]
                     /-
                       🎉 no goals
                     -/


theorem sum_le_exp_of_nonneg {x : ℝ} (hx : 0 ≤ x) (n : ℕ) : ∑ i ∈ range n, x ^ i / i ! ≤ exp x :=
  calc
    ∑ i ∈ range n, x ^ i / i ! ≤ lim (⟨_, isCauSeq_re (exp' x)⟩ : CauSeq ℝ abs) := by
      /-
        x : Real
        hx : LE.le 0 x
        n : Nat
        ⊢ LE.le ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow x i) ↑i.factorial) …
      -/
      refine le_lim (CauSeq.le_of_exists ⟨n, fun j hj => ?_⟩)
      /-
        x : Real
        hx : LE.le 0 x
        n j : Nat
        hj : GE.ge j n
        ⊢ LE.le (↑(CauSeq.const abs ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPo …
      -/
      simp only [exp', const_apply, re_sum]
      /-
        x : Real
        hx : LE.le 0 x
        n j : Nat
        hj : GE.ge j n
        ⊢ LE.le ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow x i) ↑i.factorial) …
      -/
      norm_cast
      /-
        x : Real
        hx : LE.le 0 x
        n j : Nat
        hj : GE.ge j n
        ⊢ LE.le ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow x i) ↑i.factorial) …
      -/
      refine sum_le_sum_of_subset_of_nonneg (range_mono hj) fun _ _ _ ↦ ?_
      /-
        x : Real
        hx : LE.le 0 x
        n j : Nat
        hj : GE.ge j n
        x✝² : Nat
        x✝¹ : Membership.mem (Finset.range j) x✝²
        x✝ : Not (Membership.mem (Finset.range n) x✝²)
        ⊢ LE.le 0 (HDiv.hDiv (HPow.hPow x x✝²) ↑x✝².factorial)
      -/
      positivity
      /-
        🎉 no goals
      -/
                    /-
                      x : Real
                      hx : LE.le 0 x
                      n : Nat
                      ⊢ Eq (CauSeq.lim ⟨fun n => (↑(Complex.exp' ↑x) n).re, ⋯⟩) (Real.exp x)
                    -/
    _ = exp x := by rw [exp, Complex.exp, ← cauSeqRe, lim_re]
                    /-
                      🎉 no goals
                    -/


lemma pow_div_factorial_le_exp (hx : 0 ≤ x) (n : ℕ) : x ^ n / n ! ≤ exp x :=
  calc
    x ^ n / n ! ≤ ∑ k ∈ range (n + 1), x ^ k / k ! :=
                                                               /-
                                                                 x : Real
                                                                 hx : LE.le 0 x
                                                                 n k : Nat
                                                                 x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
                                                                 ⊢ LE.le 0 ((fun k => HDiv.hDiv (HPow.hPow x k) ↑k.factorial) k)
                                                               -/
        single_le_sum (f := fun k ↦ x ^ k / k !) (fun k _ ↦ by positivity) (self_mem_range_succ n)
                                                               /-
                                                                 🎉 no goals
                                                               -/
    _ ≤ exp x := sum_le_exp_of_nonneg hx _


theorem quadratic_le_exp_of_nonneg {x : ℝ} (hx : 0 ≤ x) : 1 + x + x ^ 2 / 2 ≤ exp x :=
  calc
    1 + x + x ^ 2 / 2 = ∑ i ∈ range 3, x ^ i / i ! := by
        simp only [sum_range_succ, range_one, sum_singleton, _root_.pow_zero, factorial, cast_one,
          ne_eq, one_ne_zero, not_false_eq_true, div_self, pow_one, mul_one, div_one, Nat.mul_one,
          cast_succ, add_right_inj]
        /-
          x : Real
          hx : LE.le 0 x
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 x) (HDiv.hDiv (HPow.hPow x 2) 2)) (HAdd.hAdd (HAd …
        -/
        ring_nf
        /-
          🎉 no goals
        -/
    _ ≤ exp x := sum_le_exp_of_nonneg hx 3


private theorem add_one_lt_exp_of_pos {x : ℝ} (hx : 0 < x) : x + 1 < exp x :=
      /-
        x : Real
        hx : LT.lt 0 x
        ⊢ LT.lt (HAdd.hAdd x 1) (HAdd.hAdd (HAdd.hAdd 1 x) (HDiv.hDiv (HPow.hPow x 2)  …
      -/
  (by nlinarith : x + 1 < 1 + x + x ^ 2 / 2).trans_le (quadratic_le_exp_of_nonneg hx.le)
      /-
        🎉 no goals
      -/


private theorem add_one_le_exp_of_nonneg {x : ℝ} (hx : 0 ≤ x) : x + 1 ≤ exp x := by
  /-
    x : Real
    hx : LE.le 0 x
    ⊢ LE.le (HAdd.hAdd x 1) (Real.exp x)
  -/
  rcases eq_or_lt_of_le hx with (rfl | h)
    /-
      case inl
      hx : LE.le 0 0
      ⊢ LE.le (HAdd.hAdd 0 1) (Real.exp 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : Real
    hx : LE.le 0 x
    h : LT.lt 0 x
    ⊢ LE.le (HAdd.hAdd x 1) (Real.exp x)
  -/
  exact (add_one_lt_exp_of_pos h).le
  /-
    🎉 no goals
  -/


                                                          /-
                                                            x : Real
                                                            hx : LE.le 0 x
                                                            ⊢ LE.le 1 (Real.exp x)
                                                          -/
theorem one_le_exp {x : ℝ} (hx : 0 ≤ x) : 1 ≤ exp x := by linarith [add_one_le_exp_of_nonneg hx]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[bound]
theorem exp_pos (x : ℝ) : 0 < exp x :=
  (le_total 0 x).elim (lt_of_lt_of_le zero_lt_one ∘ one_le_exp) fun h => by
    /-
      x : Real
      h : LE.le x 0
      ⊢ LT.lt 0 (Real.exp x)
    -/
    rw [← neg_neg x, Real.exp_neg]
    /-
      x : Real
      h : LE.le x 0
      ⊢ LT.lt 0 (Inv.inv (Real.exp (Neg.neg x)))
    -/
    exact inv_pos.2 (lt_of_lt_of_le zero_lt_one (one_le_exp (neg_nonneg.2 h)))
    /-
      🎉 no goals
    -/


@[bound]
lemma exp_nonneg (x : ℝ) : 0 ≤ exp x := x.exp_pos.le


@[simp]
theorem abs_exp (x : ℝ) : |exp x| = exp x :=
  abs_of_pos (exp_pos _)


lemma exp_abs_le (x : ℝ) : exp |x| ≤ exp x + exp (-x) := by
  /-
    x : Real
    ⊢ LE.le (Real.exp (abs x)) (HAdd.hAdd (Real.exp x) (Real.exp (Neg.neg x)))
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total x 0 <;> simp [abs_of_nonpos, _root_.abs_of_nonneg, exp_nonneg, *]
                         /-
                           🎉 no goals
                         -/


@[mono]
theorem exp_strictMono : StrictMono exp := fun x y h => by
  /-
    x y : Real
    h : LT.lt x y
    ⊢ LT.lt (Real.exp x) (Real.exp y)
  -/
  rw [← sub_add_cancel y x, Real.exp_add]
  exact (lt_mul_iff_one_lt_left (exp_pos _)).2
      (lt_of_lt_of_le (by linarith) (add_one_le_exp_of_nonneg (by linarith)))


@[gcongr]
theorem exp_lt_exp_of_lt {x y : ℝ} (h : x < y) : exp x < exp y := exp_strictMono h


@[mono]
theorem exp_monotone : Monotone exp :=
  exp_strictMono.monotone


@[gcongr, bound]
theorem exp_le_exp_of_le {x y : ℝ} (h : x ≤ y) : exp x ≤ exp y := exp_monotone h


@[simp]
theorem exp_lt_exp {x y : ℝ} : exp x < exp y ↔ x < y :=
  exp_strictMono.lt_iff_lt


@[simp]
theorem exp_le_exp {x y : ℝ} : exp x ≤ exp y ↔ x ≤ y :=
  exp_strictMono.le_iff_le


theorem exp_injective : Function.Injective exp :=
  exp_strictMono.injective


@[simp]
theorem exp_eq_exp {x y : ℝ} : exp x = exp y ↔ x = y :=
  exp_injective.eq_iff


@[simp]
theorem exp_eq_one_iff : exp x = 1 ↔ x = 0 :=
  exp_injective.eq_iff' exp_zero


@[simp]
                                                         /-
                                                           x : Real
                                                           ⊢ Iff (LT.lt 1 (Real.exp x)) (LT.lt 0 x)
                                                         -/
theorem one_lt_exp_iff {x : ℝ} : 1 < exp x ↔ 0 < x := by rw [← exp_zero, exp_lt_exp]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[bound] private alias ⟨_, Bound.one_lt_exp_of_pos⟩ := one_lt_exp_iff


@[simp]
                                                         /-
                                                           x : Real
                                                           ⊢ Iff (LT.lt (Real.exp x) 1) (LT.lt x 0)
                                                         -/
theorem exp_lt_one_iff {x : ℝ} : exp x < 1 ↔ x < 0 := by rw [← exp_zero, exp_lt_exp]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem exp_le_one_iff {x : ℝ} : exp x ≤ 1 ↔ x ≤ 0 :=
  exp_zero ▸ exp_le_exp


@[simp]
theorem one_le_exp_iff {x : ℝ} : 1 ≤ exp x ↔ 0 ≤ x :=
  exp_zero ▸ exp_le_exp


/-- `Real.cosh` is always positive -/
theorem cosh_pos (x : ℝ) : 0 < Real.cosh x :=
  (cosh_eq x).symm ▸ half_pos (add_pos (exp_pos x) (exp_pos (-x)))


theorem sinh_lt_cosh : sinh x < cosh x :=
  lt_of_pow_lt_pow_left₀ 2 (cosh_pos _).le <| (cosh_sq x).symm ▸ lt_add_one _


theorem sum_div_factorial_le {α : Type*} [LinearOrderedField α] (n j : ℕ) (hn : 0 < n) :
    (∑ m ∈ range j with n ≤ m, (1 / m.factorial : α)) ≤ n.succ / (n.factorial * n) :=
  calc
    (∑ m ∈ range j with n ≤ m, (1 / m.factorial : α)) =
        ∑ m ∈ range (j - n), (1 / ((m + n).factorial : α)) := by
        /-
          α : Type u_1
          inst✝ : LinearOrderedField α
          n j : Nat
          hn : LT.lt 0 n
          ⊢ Eq ((Finset.filter (fun m => LE.le n m) (Finset.range j)).sum fun m => HDiv. …
        -/
        refine sum_nbij' (· - n) (· + n) ?_ ?_ ?_ ?_ ?_ <;>
          /-
            case refine_1
            α : Type u_1
            inst✝ : LinearOrderedField α
            n j : Nat
            hn : LT.lt 0 n
            ⊢ ∀ (a : Nat), Membership.mem (Finset.filter (fun m => LE.le n m) (Finset.rang …
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
          /-
            🎉 no goals
          -/
          simp +contextual [lt_tsub_iff_right, tsub_add_cancel_of_le]
          /-
            🎉 no goals
          -/
    _ ≤ ∑ m ∈ range (j - n), ((n.factorial : α) * (n.succ : α) ^ m)⁻¹ := by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        ⊢ LE.le ((Finset.range (HSub.hSub j n)).sum fun m => HDiv.hDiv 1 ↑(HAdd.hAdd m …
      -/
      simp_rw [one_div]
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        ⊢ LE.le ((Finset.range (HSub.hSub j n)).sum fun x => Inv.inv ↑(HAdd.hAdd x n). …
      -/
      gcongr
      /-
        case h.hba
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        i✝ : Nat
        a✝ : Membership.mem (Finset.range (HSub.hSub j n)) i✝
        ⊢ LE.le (HMul.hMul (↑n.factorial) (HPow.hPow (↑n.succ) i✝)) ↑(HAdd.hAdd i✝ n). …
      -/
      rw [← Nat.cast_pow, ← Nat.cast_mul, Nat.cast_le, add_comm]
      /-
        case h.hba
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        i✝ : Nat
        a✝ : Membership.mem (Finset.range (HSub.hSub j n)) i✝
        ⊢ LE.le (HMul.hMul n.factorial (HPow.hPow n.succ i✝)) (HAdd.hAdd n i✝).factorial
      -/
      exact Nat.factorial_mul_pow_le_factorial
      /-
        🎉 no goals
      -/
    _ = (n.factorial : α)⁻¹ * ∑ m ∈ range (j - n), (n.succ : α)⁻¹ ^ m := by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        ⊢ Eq ((Finset.range (HSub.hSub j n)).sum fun m => Inv.inv (HMul.hMul (↑n.facto …
      -/
      simp [mul_inv, ← mul_sum, ← sum_mul, mul_comm, inv_pow]
      /-
        🎉 no goals
      -/
    _ = ((n.succ : α) - n.succ * (n.succ : α)⁻¹ ^ (j - n)) / (n.factorial * n) := by
      have h₁ : (n.succ : α) ≠ 1 :=
        @Nat.cast_one α _ ▸ mt Nat.cast_inj.1 (mt Nat.succ.inj (pos_iff_ne_zero.1 hn))
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        h₁ : Ne (↑n.succ) 1
        ⊢ Eq (HMul.hMul (Inv.inv ↑n.factorial) ((Finset.range (HSub.hSub j n)).sum fun …
      -/
      have h₂ : (n.succ : α) ≠ 0 := by positivity
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        h₁ : Ne (↑n.succ) 1
        h₂ : Ne (↑n.succ) 0
        ⊢ Eq (HMul.hMul (Inv.inv ↑n.factorial) ((Finset.range (HSub.hSub j n)).sum fun …
      -/
      have h₃ : (n.factorial * n : α) ≠ 0 := by positivity
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        n j : Nat
        hn : LT.lt 0 n
        h₁ : Ne (↑n.succ) 1
        h₂ : Ne (↑n.succ) 0
        h₃ : Ne (HMul.hMul ↑n.factorial ↑n) 0
        ⊢ Eq (HMul.hMul (Inv.inv ↑n.factorial) ((Finset.range (HSub.hSub j n)).sum fun …
      -/
      have h₄ : (n.succ - 1 : α) = n := by simp
      rw [geom_sum_inv h₁ h₂, eq_div_iff_mul_eq h₃, mul_comm _ (n.factorial * n : α),
          ← mul_assoc (n.factorial⁻¹ : α), ← mul_inv_rev, h₄, ← mul_assoc (n.factorial * n : α),
          mul_comm (n : α) n.factorial, mul_inv_cancel₀ h₃, one_mul, mul_comm]
                                             /-
                                               α : Type u_1
                                               inst✝ : LinearOrderedField α
                                               n j : Nat
                                               hn : LT.lt 0 n
                                               ⊢ LE.le (HDiv.hDiv (HSub.hSub (↑n.succ) (HMul.hMul (↑n.succ) (HPow.hPow (Inv.i …
                                             -/
    _ ≤ n.succ / (n.factorial * n : α) := by gcongr; apply sub_le_self; positivity
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem exp_bound {x : ℂ} (hx : abs x ≤ 1) {n : ℕ} (hn : 0 < n) :
    abs (exp x - ∑ m ∈ range n, x ^ m / m.factorial) ≤
      abs x ^ n * ((n.succ : ℝ) * (n.factorial * n : ℝ)⁻¹) := by
  rw [← lim_const (abv := Complex.abs) (∑ m ∈ range n, _), exp, sub_eq_add_neg,
    ← lim_neg, lim_add, ← lim_abs]
  /-
    x : Complex
    hx : LE.le (Complex.abs x) 1
    n : Nat
    hn : LT.lt 0 n
    ⊢ LE.le (Complex.cauSeqAbs (HAdd.hAdd (Complex.exp' x) (Neg.neg (CauSeq.const  …
  -/
  refine lim_le (CauSeq.le_of_exists ⟨n, fun j hj => ?_⟩)
  /-
    x : Complex
    hx : LE.le (Complex.abs x) 1
    n : Nat
    hn : LT.lt 0 n
    j : Nat
    hj : GE.ge j n
    ⊢ LE.le (↑(Complex.cauSeqAbs (HAdd.hAdd (Complex.exp' x) (Neg.neg (CauSeq.cons …
  -/
  simp_rw [← sub_eq_add_neg]
  show
    abs ((∑ m ∈ range j, x ^ m / m.factorial) - ∑ m ∈ range n, x ^ m / m.factorial) ≤
      abs x ^ n * ((n.succ : ℝ) * (n.factorial * n : ℝ)⁻¹)
  /-
    x : Complex
    hx : LE.le (Complex.abs x) 1
    n : Nat
    hn : LT.lt 0 n
    j : Nat
    hj : GE.ge j n
    ⊢ LE.le (Complex.abs (HSub.hSub ((Finset.range j).sum fun m => HDiv.hDiv (HPow …
  -/
  rw [sum_range_sub_sum_range hj]
  calc
    abs (∑ m ∈ range j with n ≤ m, (x ^ m / m.factorial : ℂ))
      = abs (∑ m ∈ range j with n ≤ m, (x ^ n * (x ^ (m - n) / m.factorial) : ℂ)) := by
      refine congr_arg abs (sum_congr rfl fun m hm => ?_)
      rw [mem_filter, mem_range] at hm
      rw [← mul_div_assoc, ← pow_add, add_tsub_cancel_of_le hm.2]
    _ ≤ ∑ m ∈ range j with n ≤ m, abs (x ^ n * (x ^ (m - n) / m.factorial)) :=
      IsAbsoluteValue.abv_sum Complex.abs ..
    _ ≤ ∑ m ∈ range j with n ≤ m, abs x ^ n * (1 / m.factorial) := by
      simp_rw [map_mul, map_pow, map_div₀, abs_natCast]
      gcongr
      rw [abv_pow abs]
      exact pow_le_one₀ (abs.nonneg _) hx
    _ = abs x ^ n * ∑ m ∈ range j with n ≤ m, (1 / m.factorial : ℝ) := by
      simp [abs_mul, abv_pow abs, abs_div, ← mul_sum]
    _ ≤ abs x ^ n * (n.succ * (n.factorial * n : ℝ)⁻¹) := by
      gcongr
      exact sum_div_factorial_le _ _ hn


theorem exp_bound' {x : ℂ} {n : ℕ} (hx : abs x / n.succ ≤ 1 / 2) :
    abs (exp x - ∑ m ∈ range n, x ^ m / m.factorial) ≤ abs x ^ n / n.factorial * 2 := by
  rw [← lim_const (abv := Complex.abs) (∑ m ∈ range n, _),
    exp, sub_eq_add_neg, ← lim_neg, lim_add, ← lim_abs]
  /-
    x : Complex
    n : Nat
    hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
    ⊢ LE.le (Complex.cauSeqAbs (HAdd.hAdd (Complex.exp' x) (Neg.neg (CauSeq.const  …
  -/
  refine lim_le (CauSeq.le_of_exists ⟨n, fun j hj => ?_⟩)
  /-
    x : Complex
    n : Nat
    hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
    j : Nat
    hj : GE.ge j n
    ⊢ LE.le (↑(Complex.cauSeqAbs (HAdd.hAdd (Complex.exp' x) (Neg.neg (CauSeq.cons …
  -/
  simp_rw [← sub_eq_add_neg]
  show abs ((∑ m ∈ range j, x ^ m / m.factorial) - ∑ m ∈ range n, x ^ m / m.factorial) ≤
    abs x ^ n / n.factorial * 2
  /-
    x : Complex
    n : Nat
    hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
    j : Nat
    hj : GE.ge j n
    ⊢ LE.le (Complex.abs (HSub.hSub ((Finset.range j).sum fun m => HDiv.hDiv (HPow …
  -/
  let k := j - n
  /-
    x : Complex
    n : Nat
    hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
    j : Nat
    hj : GE.ge j n
    k : Nat := HSub.hSub j n
    ⊢ LE.le (Complex.abs (HSub.hSub ((Finset.range j).sum fun m => HDiv.hDiv (HPow …
  -/
  have hj : j = n + k := (add_tsub_cancel_of_le hj).symm
  /-
    x : Complex
    n : Nat
    hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
    j : Nat
    hj✝ : GE.ge j n
    k : Nat := HSub.hSub j n
    hj : Eq j (HAdd.hAdd n k)
    ⊢ LE.le (Complex.abs (HSub.hSub ((Finset.range j).sum fun m => HDiv.hDiv (HPow …
  -/
  rw [hj, sum_range_add_sub_sum_range]
  calc
    abs (∑ i ∈ range k, x ^ (n + i) / ((n + i).factorial : ℂ)) ≤
        ∑ i ∈ range k, abs (x ^ (n + i) / ((n + i).factorial : ℂ)) :=
      IsAbsoluteValue.abv_sum _ _ _
    _ ≤ ∑ i ∈ range k, abs x ^ (n + i) / (n + i).factorial := by
      simp [Complex.abs_natCast, map_div₀, abv_pow abs]
    _ ≤ ∑ i ∈ range k, abs x ^ (n + i) / ((n.factorial : ℝ) * (n.succ : ℝ) ^ i) := ?_
    _ = ∑ i ∈ range k, abs x ^ n / n.factorial * (abs x ^ i / (n.succ : ℝ) ^ i) := ?_
    _ ≤ abs x ^ n / ↑n.factorial * 2 := ?_
    /-
      case calc_1
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      ⊢ LE.le ((Finset.range k).sum fun i => HDiv.hDiv (HPow.hPow (Complex.abs x) (H …
    -/
  · gcongr
    /-
      case calc_1.h.h
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      i✝ : Nat
      a✝ : Membership.mem (Finset.range k) i✝
      ⊢ LE.le (HMul.hMul (↑n.factorial) (HPow.hPow (↑n.succ) i✝)) ↑(HAdd.hAdd n i✝). …
    -/
    exact mod_cast Nat.factorial_mul_pow_le_factorial
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      ⊢ Eq ((Finset.range k).sum fun i => HDiv.hDiv (HPow.hPow (Complex.abs x) (HAdd …
    -/
  · refine Finset.sum_congr rfl fun _ _ => ?_
    /-
      case calc_2
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      x✝¹ : Nat
      x✝ : Membership.mem (Finset.range k) x✝¹
      ⊢ Eq (HDiv.hDiv (HPow.hPow (Complex.abs x) (HAdd.hAdd n x✝¹)) (HMul.hMul (↑n.f …
    -/
    simp only [pow_add, div_eq_inv_mul, mul_inv, mul_left_comm, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      ⊢ LE.le ((Finset.range k).sum fun i => HMul.hMul (HDiv.hDiv (HPow.hPow (Comple …
    -/
  · rw [← mul_sum]
    /-
      case calc_3
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      ⊢ LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow (Complex.abs x) n) ↑n.factorial) ((Fi …
    -/
    gcongr
    /-
      case calc_3.h
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      ⊢ LE.le ((Finset.range k).sum fun i => HDiv.hDiv (HPow.hPow (Complex.abs x) i) …
    -/
    simp_rw [← div_pow]
    /-
      case calc_3.h
      x : Complex
      n : Nat
      hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
      j : Nat
      hj✝ : GE.ge j n
      k : Nat := HSub.hSub j n
      hj : Eq j (HAdd.hAdd n k)
      ⊢ LE.le ((Finset.range k).sum fun x_1 => HPow.hPow (HDiv.hDiv (Complex.abs x)  …
    -/
    rw [geom_sum_eq, div_le_iff_of_neg]
      /-
        case calc_3.h
        x : Complex
        n : Nat
        hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
        j : Nat
        hj✝ : GE.ge j n
        k : Nat := HSub.hSub j n
        hj : Eq j (HAdd.hAdd n k)
        ⊢ LE.le (HMul.hMul 2 (HSub.hSub (HDiv.hDiv (Complex.abs x) ↑n.succ) 1)) (HSub. …
      -/
    · trans (-1 : ℝ)
        /-
          x : Complex
          n : Nat
          hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
          j : Nat
          hj✝ : GE.ge j n
          k : Nat := HSub.hSub j n
          hj : Eq j (HAdd.hAdd n k)
          ⊢ LE.le (HMul.hMul 2 (HSub.hSub (HDiv.hDiv (Complex.abs x) ↑n.succ) 1)) (-1)
        -/
      · linarith
        /-
          🎉 no goals
        -/
        /-
          x : Complex
          n : Nat
          hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
          j : Nat
          hj✝ : GE.ge j n
          k : Nat := HSub.hSub j n
          hj : Eq j (HAdd.hAdd n k)
          ⊢ LE.le (-1) (HSub.hSub (HPow.hPow (HDiv.hDiv (Complex.abs x) ↑n.succ) k) 1)
        -/
      · simp only [neg_le_sub_iff_le_add, div_pow, Nat.cast_succ, le_add_iff_nonneg_left]
        /-
          x : Complex
          n : Nat
          hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
          j : Nat
          hj✝ : GE.ge j n
          k : Nat := HSub.hSub j n
          hj : Eq j (HAdd.hAdd n k)
          ⊢ LE.le 0 (HDiv.hDiv (HPow.hPow (Complex.abs x) k) (HPow.hPow (HAdd.hAdd (↑n)  …
        -/
        positivity
        /-
          🎉 no goals
        -/
      /-
        case calc_3.h
        x : Complex
        n : Nat
        hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
        j : Nat
        hj✝ : GE.ge j n
        k : Nat := HSub.hSub j n
        hj : Eq j (HAdd.hAdd n k)
        ⊢ LT.lt (HSub.hSub (HDiv.hDiv (Complex.abs x) ↑n.succ) 1) 0
      -/
    · linarith
      /-
        🎉 no goals
      -/
      /-
        case calc_3.h.h
        x : Complex
        n : Nat
        hx : LE.le (HDiv.hDiv (Complex.abs x) ↑n.succ) (1 / 2)
        j : Nat
        hj✝ : GE.ge j n
        k : Nat := HSub.hSub j n
        hj : Eq j (HAdd.hAdd n k)
        ⊢ Ne (HDiv.hDiv (Complex.abs x) ↑n.succ) 1
      -/
    · linarith
      /-
        🎉 no goals
      -/


theorem abs_exp_sub_one_le {x : ℂ} (hx : abs x ≤ 1) : abs (exp x - 1) ≤ 2 * abs x :=
  calc
                                                                             /-
                                                                               x : Complex
                                                                               hx : LE.le (Complex.abs x) 1
                                                                               ⊢ Eq (Complex.abs (HSub.hSub (Complex.exp x) 1)) (Complex.abs (HSub.hSub (Comp …
                                                                             -/
    abs (exp x - 1) = abs (exp x - ∑ m ∈ range 1, x ^ m / m.factorial) := by simp [sum_range_succ]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    _ ≤ abs x ^ 1 * ((Nat.succ 1 : ℝ) * ((Nat.factorial 1) * (1 : ℕ) : ℝ)⁻¹) :=
                        /-
                          x : Complex
                          hx : LE.le (Complex.abs x) 1
                          ⊢ LT.lt 0 1
                        -/
      (exp_bound hx (by decide))
                        /-
                          🎉 no goals
                        -/
                        /-
                          x : Complex
                          hx : LE.le (Complex.abs x) 1
                          ⊢ Eq (HMul.hMul (HPow.hPow (Complex.abs x) 1) (HMul.hMul (↑(Nat.succ 1)) (Inv. …
                        -/
    _ = 2 * abs x := by simp [two_mul, mul_two, mul_add, mul_comm, add_mul, Nat.factorial]
                        /-
                          🎉 no goals
                        -/


theorem abs_exp_sub_one_sub_id_le {x : ℂ} (hx : abs x ≤ 1) : abs (exp x - 1 - x) ≤ abs x ^ 2 :=
  calc
    abs (exp x - 1 - x) = abs (exp x - ∑ m ∈ range 2, x ^ m / m.factorial) := by
      /-
        x : Complex
        hx : LE.le (Complex.abs x) 1
        ⊢ Eq (Complex.abs (HSub.hSub (HSub.hSub (Complex.exp x) 1) x)) (Complex.abs (H …
      -/
      simp [sub_eq_add_neg, sum_range_succ_comm, add_assoc, Nat.factorial]
      /-
        🎉 no goals
      -/
    _ ≤ abs x ^ 2 * ((Nat.succ 2 : ℝ) * (Nat.factorial 2 * (2 : ℕ) : ℝ)⁻¹) :=
                        /-
                          x : Complex
                          hx : LE.le (Complex.abs x) 1
                          ⊢ LT.lt 0 2
                        -/
      (exp_bound hx (by decide))
                        /-
                          🎉 no goals
                        -/
                            /-
                              x : Complex
                              hx : LE.le (Complex.abs x) 1
                              ⊢ LE.le (HMul.hMul (HPow.hPow (Complex.abs x) 2) (HMul.hMul (↑(Nat.succ 2)) (I …
                            -/
    _ ≤ abs x ^ 2 * 1 := by gcongr; norm_num [Nat.factorial]
                                    /-
                                      🎉 no goals
                                    -/
                        /-
                          x : Complex
                          hx : LE.le (Complex.abs x) 1
                          ⊢ Eq (HMul.hMul (HPow.hPow (Complex.abs x) 2) 1) (HPow.hPow (Complex.abs x) 2)
                        -/
    _ = abs x ^ 2 := by rw [mul_one]
                        /-
                          🎉 no goals
                        -/


nonrec theorem exp_bound {x : ℝ} (hx : |x| ≤ 1) {n : ℕ} (hn : 0 < n) :
    |exp x - ∑ m ∈ range n, x ^ m / m.factorial| ≤ |x| ^ n * (n.succ / (n.factorial * n)) := by
  /-
    x : Real
    hx : LE.le (abs x) 1
    n : Nat
    hn : LT.lt 0 n
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hDiv  …
  -/
  have hxc : Complex.abs x ≤ 1 := mod_cast hx
  /-
    x : Real
    hx : LE.le (abs x) 1
    n : Nat
    hn : LT.lt 0 n
    hxc : LE.le (Complex.abs ↑x) 1
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hDiv  …
  -/
  convert exp_bound hxc hn using 2 <;>
  -- Porting note: was `norm_cast`
  simp only [← abs_ofReal, ← ofReal_sub, ← ofReal_exp, ← ofReal_sum, ← ofReal_pow,
    ← ofReal_div, ← ofReal_natCast]


theorem exp_bound' {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 1) {n : ℕ} (hn : 0 < n) :
    Real.exp x ≤ (∑ m ∈ Finset.range n, x ^ m / m.factorial) +
      x ^ n * (n + 1) / (n.factorial * n) := by
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  have h3 : |x| = x := by simpa
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    h3 : Eq (abs x) x
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  have h4 : |x| ≤ 1 := by rwa [h3]
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    h3 : Eq (abs x) x
    h4 : LE.le (abs x) 1
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  have h' := Real.exp_bound h4 hn
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    h3 : Eq (abs x) x
    h4 : LE.le (abs x) 1
    h' : LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hD …
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  rw [h3] at h'
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    h3 : Eq (abs x) x
    h4 : LE.le (abs x) 1
    h' : LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hD …
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  have h'' := (abs_sub_le_iff.1 h').1
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    h3 : Eq (abs x) x
    h4 : LE.le (abs x) 1
    h' : LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hD …
    h'' : LE.le (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hDiv ( …
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  have t := sub_le_iff_le_add'.1 h''
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LE.le x 1
    n : Nat
    hn : LT.lt 0 n
    h3 : Eq (abs x) x
    h4 : LE.le (abs x) 1
    h' : LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hD …
    h'' : LE.le (HSub.hSub (Real.exp x) ((Finset.range n).sum fun m => HDiv.hDiv ( …
    t : LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HP …
    ⊢ LE.le (Real.exp x) (HAdd.hAdd ((Finset.range n).sum fun m => HDiv.hDiv (HPow …
  -/
  simpa [mul_div_assoc] using t
  /-
    🎉 no goals
  -/


theorem abs_exp_sub_one_le {x : ℝ} (hx : |x| ≤ 1) : |exp x - 1| ≤ 2 * |x| := by
  /-
    x : Real
    hx : LE.le (abs x) 1
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) 1)) (HMul.hMul 2 (abs x))
  -/
  have : |x| ≤ 1 := mod_cast hx
  -- Porting note: was
  --exact_mod_cast Complex.abs_exp_sub_one_le (x := x) this
  /-
    x : Real
    hx this : LE.le (abs x) 1
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) 1)) (HMul.hMul 2 (abs x))
  -/
  have := Complex.abs_exp_sub_one_le (x := x) (by simpa using this)
  /-
    x : Real
    hx this✝ : LE.le (abs x) 1
    this : LE.le (Complex.abs (HSub.hSub (Complex.exp ↑x) 1)) (HMul.hMul 2 (Comple …
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) 1)) (HMul.hMul 2 (abs x))
  -/
  rw [← ofReal_exp, ← ofReal_one, ← ofReal_sub, abs_ofReal, abs_ofReal] at this
  /-
    x : Real
    hx this✝ : LE.le (abs x) 1
    this : LE.le (abs (HSub.hSub (Real.exp x) 1)) (HMul.hMul 2 (abs x))
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) 1)) (HMul.hMul 2 (abs x))
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem abs_exp_sub_one_sub_id_le {x : ℝ} (hx : |x| ≤ 1) : |exp x - 1 - x| ≤ x ^ 2 := by
  /-
    x : Real
    hx : LE.le (abs x) 1
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (Real.exp x) 1) x)) (HPow.hPow x 2)
  -/
  rw [← _root_.sq_abs]
  -- Porting note: was
  -- exact_mod_cast Complex.abs_exp_sub_one_sub_id_le this
  /-
    x : Real
    hx : LE.le (abs x) 1
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (Real.exp x) 1) x)) (HPow.hPow (abs x) 2)
  -/
  have : Complex.abs x ≤ 1 := mod_cast hx
  /-
    x : Real
    hx : LE.le (abs x) 1
    this : LE.le (Complex.abs ↑x) 1
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (Real.exp x) 1) x)) (HPow.hPow (abs x) 2)
  -/
  have := Complex.abs_exp_sub_one_sub_id_le this
  /-
    x : Real
    hx : LE.le (abs x) 1
    this✝ : LE.le (Complex.abs ↑x) 1
    this : LE.le (Complex.abs (HSub.hSub (HSub.hSub (Complex.exp ↑x) 1) ↑x)) (HPow …
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (Real.exp x) 1) x)) (HPow.hPow (abs x) 2)
  -/
  rw [← ofReal_one, ← ofReal_exp, ← ofReal_sub, ← ofReal_sub, abs_ofReal, abs_ofReal] at this
  /-
    x : Real
    hx : LE.le (abs x) 1
    this✝ : LE.le (Complex.abs ↑x) 1
    this : LE.le (abs (HSub.hSub (HSub.hSub (Real.exp x) 1) x)) (HPow.hPow (abs x) …
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (Real.exp x) 1) x)) (HPow.hPow (abs x) 2)
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- A finite initial segment of the exponential series, followed by an arbitrary tail.
For fixed `n` this is just a linear map wrt `r`, and each map is a simple linear function
of the previous (see `expNear_succ`), with `expNear n x r ⟶ exp x` as `n ⟶ ∞`,
for any `r`. -/
noncomputable def expNear (n : ℕ) (x r : ℝ) : ℝ :=
  (∑ m ∈ range n, x ^ m / m.factorial) + x ^ n / n.factorial * r


@[simp]
                                                     /-
                                                       x r : Real
                                                       ⊢ Eq (Real.expNear 0 x r) r
                                                     -/
theorem expNear_zero (x r) : expNear 0 x r = r := by simp [expNear]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem expNear_succ (n x r) : expNear (n + 1) x r = expNear n x (1 + x / (n + 1) * r) := by
  simp [expNear, range_succ, mul_add, add_left_comm, add_assoc, pow_succ, div_eq_mul_inv,
      mul_inv, Nat.factorial]
  /-
    n : Nat
    x r : Real
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow x n) x) (HMul.hMul (Inv.inv ↑ …
  -/
  ac_rfl
  /-
    🎉 no goals
  -/


theorem expNear_sub (n x r₁ r₂) : expNear n x r₁ -
    expNear n x r₂ = x ^ n / n.factorial * (r₁ - r₂) := by
  /-
    n : Nat
    x r₁ r₂ : Real
    ⊢ Eq (HSub.hSub (Real.expNear n x r₁) (Real.expNear n x r₂)) (HMul.hMul (HDiv. …
  -/
  simp [expNear, mul_sub]
  /-
    🎉 no goals
  -/


theorem exp_approx_end (n m : ℕ) (x : ℝ) (e₁ : n + 1 = m) (h : |x| ≤ 1) :
    |exp x - expNear m x 0| ≤ |x| ^ m / m.factorial * ((m + 1) / m) := by
  /-
    n m : Nat
    x : Real
    e₁ : Eq (HAdd.hAdd n 1) m
    h : LE.le (abs x) 1
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear m x 0))) (HMul.hMul (HDiv.h …
  -/
  simp only [expNear, mul_zero, add_zero]
  /-
    n m : Nat
    x : Real
    e₁ : Eq (HAdd.hAdd n 1) m
    h : LE.le (abs x) 1
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) ((Finset.range m).sum fun m => HDiv.hDiv  …
  -/
  convert exp_bound (n := m) h ?_ using 1
    /-
      case h.e'_4
      n m : Nat
      x : Real
      e₁ : Eq (HAdd.hAdd n 1) m
      h : LE.le (abs x) 1
      ⊢ Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (abs x) m) ↑m.factorial) (HDiv.hDiv (HAd …
    -/
  · field_simp [mul_comm]
    /-
      🎉 no goals
    -/
    /-
      n m : Nat
      x : Real
      e₁ : Eq (HAdd.hAdd n 1) m
      h : LE.le (abs x) 1
      ⊢ LT.lt 0 m
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem exp_approx_succ {n} {x a₁ b₁ : ℝ} (m : ℕ) (e₁ : n + 1 = m) (a₂ b₂ : ℝ)
    (e : |1 + x / m * a₂ - a₁| ≤ b₁ - |x| / m * b₂)
    (h : |exp x - expNear m x a₂| ≤ |x| ^ m / m.factorial * b₂) :
    |exp x - expNear n x a₁| ≤ |x| ^ n / n.factorial * b₁ := by
  /-
    n : Nat
    x a₁ b₁ : Real
    m : Nat
    e₁ : Eq (HAdd.hAdd n 1) m
    a₂ b₂ : Real
    e : LE.le (abs (HSub.hSub (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv x ↑m) a₂)) a₁)) ( …
    h : LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear m x a₂))) (HMul.hMul (HDi …
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear n x a₁))) (HMul.hMul (HDiv. …
  -/
  refine (abs_sub_le _ _ _).trans ((add_le_add_right h _).trans ?_)
  /-
    n : Nat
    x a₁ b₁ : Real
    m : Nat
    e₁ : Eq (HAdd.hAdd n 1) m
    a₂ b₂ : Real
    e : LE.le (abs (HSub.hSub (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv x ↑m) a₂)) a₁)) ( …
    h : LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear m x a₂))) (HMul.hMul (HDi …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HPow.hPow (abs x) m) ↑m.factorial) b …
  -/
  subst e₁; rw [expNear_succ, expNear_sub, abs_mul]
  convert mul_le_mul_of_nonneg_left (a := |x| ^ n / ↑(Nat.factorial n))
      (le_sub_iff_add_le'.1 e) ?_ using 1
    /-
      case h.e'_3
      n : Nat
      x a₁ b₁ a₂ b₂ : Real
      e : LE.le (abs (HSub.hSub (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv x ↑(HAdd.hAdd n 1 …
      h : LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear (HAdd.hAdd n 1) x a₂))) ( …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HPow.hPow (abs x) (HAdd.hAdd n 1)) ↑(HA …
    -/
  · simp [mul_add, pow_succ', div_eq_mul_inv, abs_mul, abs_inv, ← pow_abs, mul_inv, Nat.factorial]
    /-
      case h.e'_3
      n : Nat
      x a₁ b₁ a₂ b₂ : Real
      e : LE.le (abs (HSub.hSub (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv x ↑(HAdd.hAdd n 1 …
      h : LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear (HAdd.hAdd n 1) x a₂))) ( …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (abs x) (HPow.hPow (abs x) n)) (HMul.hMu …
    -/
    ac_rfl
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      x a₁ b₁ a₂ b₂ : Real
      e : LE.le (abs (HSub.hSub (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv x ↑(HAdd.hAdd n 1 …
      h : LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear (HAdd.hAdd n 1) x a₂))) ( …
      ⊢ LE.le 0 (HDiv.hDiv (HPow.hPow (abs x) n) ↑n.factorial)
    -/
  · simp [div_nonneg, abs_nonneg]
    /-
      🎉 no goals
    -/


theorem exp_approx_end' {n} {x a b : ℝ} (m : ℕ) (e₁ : n + 1 = m) (rm : ℝ) (er : ↑m = rm)
    (h : |x| ≤ 1) (e : |1 - a| ≤ b - |x| / rm * ((rm + 1) / rm)) :
    |exp x - expNear n x a| ≤ |x| ^ n / n.factorial * b := by
  /-
    n : Nat
    x a b : Real
    m : Nat
    e₁ : Eq (HAdd.hAdd n 1) m
    rm : Real
    er : Eq (↑m) rm
    h : LE.le (abs x) 1
    e : LE.le (abs (HSub.hSub 1 a)) (HSub.hSub b (HMul.hMul (HDiv.hDiv (abs x) rm) …
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear n x a))) (HMul.hMul (HDiv.h …
  -/
  subst er
  /-
    n : Nat
    x a b : Real
    m : Nat
    e₁ : Eq (HAdd.hAdd n 1) m
    h : LE.le (abs x) 1
    e : LE.le (abs (HSub.hSub 1 a)) (HSub.hSub b (HMul.hMul (HDiv.hDiv (abs x) ↑m) …
    ⊢ LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear n x a))) (HMul.hMul (HDiv.h …
  -/
  exact exp_approx_succ _ e₁ _ _ (by simpa using e) (exp_approx_end _ _ _ e₁ h)
  /-
    🎉 no goals
  -/


theorem exp_1_approx_succ_eq {n} {a₁ b₁ : ℝ} {m : ℕ} (en : n + 1 = m) {rm : ℝ} (er : ↑m = rm)
    (h : |exp 1 - expNear m 1 ((a₁ - 1) * rm)| ≤ |1| ^ m / m.factorial * (b₁ * rm)) :
    |exp 1 - expNear n 1 a₁| ≤ |1| ^ n / n.factorial * b₁ := by
  /-
    n : Nat
    a₁ b₁ : Real
    m : Nat
    en : Eq (HAdd.hAdd n 1) m
    rm : Real
    er : Eq (↑m) rm
    h : LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear m 1 (HMul.hMul (HSub.hSub …
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear n 1 a₁))) (HMul.hMul (HDiv. …
  -/
  subst er
  /-
    n : Nat
    a₁ b₁ : Real
    m : Nat
    en : Eq (HAdd.hAdd n 1) m
    h : LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear m 1 (HMul.hMul (HSub.hSub …
    ⊢ LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear n 1 a₁))) (HMul.hMul (HDiv. …
  -/
  refine exp_approx_succ _ en _ _ ?_ h
  /-
    n : Nat
    a₁ b₁ : Real
    m : Nat
    en : Eq (HAdd.hAdd n 1) m
    h : LE.le (abs (HSub.hSub (Real.exp 1) (Real.expNear m 1 (HMul.hMul (HSub.hSub …
    ⊢ LE.le (abs (HSub.hSub (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv 1 ↑m) (HMul.hMul (H …
  -/
  field_simp [show (m : ℝ) ≠ 0 by norm_cast; omega]
  /-
    🎉 no goals
  -/


theorem exp_approx_start (x a b : ℝ) (h : |exp x - expNear 0 x a| ≤ |x| ^ 0 / Nat.factorial 0 * b) :
                          /-
                            x a b : Real
                            h : LE.le (abs (HSub.hSub (Real.exp x) (Real.expNear 0 x a))) (HMul.hMul (HDiv …
                            ⊢ LE.le (abs (HSub.hSub (Real.exp x) a)) b
                          -/
    |exp x - a| ≤ b := by simpa using h
                          /-
                            🎉 no goals
                          -/


theorem cos_bound {x : ℝ} (hx : |x| ≤ 1) : |cos x - (1 - x ^ 2 / 2)| ≤ |x| ^ 4 * (5 / 96) :=
  calc
    |cos x - (1 - x ^ 2 / 2)| = Complex.abs (Complex.cos x - (1 - (x : ℂ) ^ 2 / 2)) := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ Eq (abs (HSub.hSub (Real.cos x) (HSub.hSub 1 (HDiv.hDiv (HPow.hPow x 2) 2))) …
      -/
      rw [← abs_ofReal]; simp
                         /-
                           🎉 no goals
                         -/
    _ = Complex.abs ((Complex.exp (x * I) + Complex.exp (-x * I) - (2 - (x : ℂ) ^ 2)) / 2) := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ Eq (Complex.abs (HSub.hSub (Complex.cos ↑x) (HSub.hSub 1 (HDiv.hDiv (HPow.hP …
      -/
      simp [Complex.cos, sub_div, add_div, neg_div, div_self (two_ne_zero' ℂ)]
      /-
        🎉 no goals
      -/
    _ = abs
          (((Complex.exp (x * I) - ∑ m ∈ range 4, (x * I) ^ m / m.factorial) +
              (Complex.exp (-x * I) - ∑ m ∈ range 4, (-x * I) ^ m / m.factorial)) / 2) :=
      (congr_arg Complex.abs
        (congr_arg (fun x : ℂ => x / 2)
          (by
            simp only [sum_range_succ, neg_mul, pow_succ, pow_zero, mul_one, range_zero, sum_empty,
              Nat.factorial, Nat.cast_one, ne_eq, one_ne_zero, not_false_eq_true, div_self,
              zero_add, div_one, Nat.mul_one, Nat.cast_succ, Nat.cast_mul, Nat.cast_ofNat, mul_neg,
              neg_neg]
            /-
              x : Real
              hx : LE.le (abs x) 1
              ⊢ Eq (HSub.hSub (HAdd.hAdd (Complex.exp (HMul.hMul (↑x) Complex.I)) (Complex.e …
            -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
            apply Complex.ext <;> simp [div_eq_mul_inv, normSq] <;> ring_nf
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
            )))
    _ ≤ abs ((Complex.exp (x * I) - ∑ m ∈ range 4, (x * I) ^ m / m.factorial) / 2) +
          abs ((Complex.exp (-x * I) - ∑ m ∈ range 4, (-x * I) ^ m / m.factorial) / 2) := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ LE.le (Complex.abs (HDiv.hDiv (HAdd.hAdd (HSub.hSub (Complex.exp (HMul.hMul  …
      -/
      rw [add_div]; exact Complex.abs.add_le _ _
                    /-
                      🎉 no goals
                    -/
    _ = abs (Complex.exp (x * I) - ∑ m ∈ range 4, (x * I) ^ m / m.factorial) / 2 +
          abs (Complex.exp (-x * I) - ∑ m ∈ range 4, (-x * I) ^ m / m.factorial) / 2 := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ Eq (HAdd.hAdd (Complex.abs (HDiv.hDiv (HSub.hSub (Complex.exp (HMul.hMul (↑x …
      -/
      simp [map_div₀]
      /-
        🎉 no goals
      -/
    _ ≤ Complex.abs (x * I) ^ 4 * (Nat.succ 4 * ((Nat.factorial 4) * (4 : ℕ) : ℝ)⁻¹) / 2 +
          Complex.abs (-x * I) ^ 4 * (Nat.succ 4 * ((Nat.factorial 4) * (4 : ℕ) : ℝ)⁻¹) / 2 := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (Complex.abs (HSub.hSub (Complex.exp (HMul.hMul  …
      -/
      gcongr
        /-
          case h₁.hab
          x : Real
          hx : LE.le (abs x) 1
          ⊢ LE.le (Complex.abs (HSub.hSub (Complex.exp (HMul.hMul (↑x) Complex.I)) ((Fin …
        -/
      · exact Complex.exp_bound (by simpa) (by decide)
        /-
          🎉 no goals
        -/
        /-
          case h₂.hab
          x : Real
          hx : LE.le (abs x) 1
          ⊢ LE.le (Complex.abs (HSub.hSub (Complex.exp (HMul.hMul (Neg.neg ↑x) Complex.I …
        -/
      · exact Complex.exp_bound (by simpa) (by decide)
        /-
          🎉 no goals
        -/
                                 /-
                                   x : Real
                                   hx : LE.le (abs x) 1
                                   ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (HMul.hMul (HPow.hPow (Complex.abs (HMul.hMul (↑ …
                                 -/
    _ ≤ |x| ^ 4 * (5 / 96) := by norm_num [Nat.factorial]
                                 /-
                                   🎉 no goals
                                 -/


theorem sin_bound {x : ℝ} (hx : |x| ≤ 1) : |sin x - (x - x ^ 3 / 6)| ≤ |x| ^ 4 * (5 / 96) :=
  calc
    |sin x - (x - x ^ 3 / 6)| = Complex.abs (Complex.sin x - (x - x ^ 3 / 6 : ℝ)) := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ Eq (abs (HSub.hSub (Real.sin x) (HSub.hSub x (HDiv.hDiv (HPow.hPow x 3) 6))) …
      -/
      rw [← abs_ofReal]; simp
                         /-
                           🎉 no goals
                         -/
    _ = Complex.abs (((Complex.exp (-x * I) - Complex.exp (x * I)) * I -
          (2 * x - x ^ 3 / 3 : ℝ)) / 2) := by
      simp [Complex.sin, sub_div, add_div, neg_div, mul_div_cancel_left₀ _ (two_ne_zero' ℂ),
        div_div, show (3 : ℂ) * 2 = 6 by norm_num]
    _ = Complex.abs (((Complex.exp (-x * I) - ∑ m ∈ range 4, (-x * I) ^ m / m.factorial) -
                (Complex.exp (x * I) - ∑ m ∈ range 4, (x * I) ^ m / m.factorial)) * I / 2) :=
      (congr_arg Complex.abs
        (congr_arg (fun x : ℂ => x / 2)
          (by
            simp only [sum_range_succ, neg_mul, pow_succ, pow_zero, mul_one, ofReal_sub, ofReal_mul,
              ofReal_ofNat, ofReal_div, range_zero, sum_empty, Nat.factorial, Nat.cast_one, ne_eq,
              one_ne_zero, not_false_eq_true, div_self, zero_add, div_one, mul_neg, neg_neg,
              Nat.mul_one, Nat.cast_succ, Nat.cast_mul, Nat.cast_ofNat]
            /-
              x : Real
              hx : LE.le (abs x) 1
              ⊢ Eq (HSub.hSub (HMul.hMul (HSub.hSub (Complex.exp (Neg.neg (HMul.hMul (↑x) Co …
            -/
                                  /-
                                    🎉 no goals
                                  -/
            apply Complex.ext <;> simp [div_eq_mul_inv, normSq]; ring)))
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    _ ≤ abs ((Complex.exp (-x * I) - ∑ m ∈ range 4, (-x * I) ^ m / m.factorial) * I / 2) +
          abs (-((Complex.exp (x * I) - ∑ m ∈ range 4, (x * I) ^ m / m.factorial) * I) / 2) := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ LE.le (Complex.abs (HDiv.hDiv (HMul.hMul (HSub.hSub (HSub.hSub (Complex.exp  …
      -/
      rw [sub_mul, sub_eq_add_neg, add_div]; exact Complex.abs.add_le _ _
                                             /-
                                               🎉 no goals
                                             -/
    _ = abs (Complex.exp (x * I) - ∑ m ∈ range 4, (x * I) ^ m / m.factorial) / 2 +
          abs (Complex.exp (-x * I) - ∑ m ∈ range 4, (-x * I) ^ m / m.factorial) / 2 := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ Eq (HAdd.hAdd (Complex.abs (HDiv.hDiv (HMul.hMul (HSub.hSub (Complex.exp (HM …
      -/
      simp [add_comm, map_div₀]
      /-
        🎉 no goals
      -/
    _ ≤ Complex.abs (x * I) ^ 4 * (Nat.succ 4 * (Nat.factorial 4 * (4 : ℕ) : ℝ)⁻¹) / 2 +
          Complex.abs (-x * I) ^ 4 * (Nat.succ 4 * (Nat.factorial 4 * (4 : ℕ) : ℝ)⁻¹) / 2 := by
      /-
        x : Real
        hx : LE.le (abs x) 1
        ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (Complex.abs (HSub.hSub (Complex.exp (HMul.hMul  …
      -/
      gcongr
        /-
          case h₁.hab
          x : Real
          hx : LE.le (abs x) 1
          ⊢ LE.le (Complex.abs (HSub.hSub (Complex.exp (HMul.hMul (↑x) Complex.I)) ((Fin …
        -/
      · exact Complex.exp_bound (by simpa) (by decide)
        /-
          🎉 no goals
        -/
        /-
          case h₂.hab
          x : Real
          hx : LE.le (abs x) 1
          ⊢ LE.le (Complex.abs (HSub.hSub (Complex.exp (HMul.hMul (Neg.neg ↑x) Complex.I …
        -/
      · exact Complex.exp_bound (by simpa) (by decide)
        /-
          🎉 no goals
        -/
                                 /-
                                   x : Real
                                   hx : LE.le (abs x) 1
                                   ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (HMul.hMul (HPow.hPow (Complex.abs (HMul.hMul (↑ …
                                 -/
    _ ≤ |x| ^ 4 * (5 / 96) := by norm_num [Nat.factorial]
                                 /-
                                   🎉 no goals
                                 -/


theorem cos_pos_of_le_one {x : ℝ} (hx : |x| ≤ 1) : 0 < cos x :=
  calc 0 < 1 - x ^ 2 / 2 - |x| ^ 4 * (5 / 96) :=
      sub_pos.2 <|
        lt_sub_iff_add_lt.2
          (calc
            |x| ^ 4 * (5 / 96) + x ^ 2 / 2 ≤ 1 * (5 / 96) + 1 / 2 := by
                  /-
                    x : Real
                    hx : LE.le (abs x) 1
                    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (abs x) 4) (5 / 96)) (HDiv.hDiv (HPow …
                  -/
                  gcongr
                    /-
                      case h₁.h
                      x : Real
                      hx : LE.le (abs x) 1
                      ⊢ LE.le (HPow.hPow (abs x) 4) 1
                    -/
                  · exact pow_le_one₀ (abs_nonneg _) hx
                    /-
                      🎉 no goals
                    -/
                    /-
                      case h₂.hab
                      x : Real
                      hx : LE.le (abs x) 1
                      ⊢ LE.le (HPow.hPow x 2) 1
                    -/
                  · rw [sq, ← abs_mul_self, abs_mul]
                    /-
                      case h₂.hab
                      x : Real
                      hx : LE.le (abs x) 1
                      ⊢ LE.le (HMul.hMul (abs x) (abs x)) 1
                    -/
                    exact mul_le_one₀ hx (abs_nonneg _) hx
                    /-
                      🎉 no goals
                    -/
                        /-
                          x : Real
                          hx : LE.le (abs x) 1
                          ⊢ LT.lt (HAdd.hAdd (HMul.hMul 1 (5 / 96)) (1 / 2)) 1
                        -/
            _ < 1 := by norm_num)
                        /-
                          🎉 no goals
                        -/
    _ ≤ cos x := sub_le_comm.1 (abs_sub_le_iff.1 (cos_bound hx)).2


theorem sin_pos_of_pos_of_le_one {x : ℝ} (hx0 : 0 < x) (hx : x ≤ 1) : 0 < sin x :=
  calc 0 < x - x ^ 3 / 6 - |x| ^ 4 * (5 / 96) :=
      sub_pos.2 <| lt_sub_iff_add_lt.2
          (calc
            |x| ^ 4 * (5 / 96) + x ^ 3 / 6 ≤ x * (5 / 96) + x / 6 := by
                /-
                  x : Real
                  hx0 : LT.lt 0 x
                  hx : LE.le x 1
                  ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (abs x) 4) (5 / 96)) (HDiv.hDiv (HPow …
                -/
                gcongr
                · calc
                    |x| ^ 4 ≤ |x| ^ 1 :=
                      pow_le_pow_of_le_one (abs_nonneg _)
                        (by rwa [_root_.abs_of_nonneg (le_of_lt hx0)]) (by decide)
                    _ = x := by simp [_root_.abs_of_nonneg (le_of_lt hx0)]
                · calc
                    x ^ 3 ≤ x ^ 1 := pow_le_pow_of_le_one (le_of_lt hx0) hx (by decide)
                    _ = x := pow_one _
                        /-
                          x : Real
                          hx0 : LT.lt 0 x
                          hx : LE.le x 1
                          ⊢ LT.lt (HAdd.hAdd (HMul.hMul x (5 / 96)) (HDiv.hDiv x 6)) x
                        -/
            _ < x := by linarith)
                        /-
                          🎉 no goals
                        -/
    _ ≤ sin x :=
                                                     /-
                                                       x : Real
                                                       hx0 : LT.lt 0 x
                                                       hx : LE.le x 1
                                                       ⊢ LE.le (abs x) 1
                                                     -/
      sub_le_comm.1 (abs_sub_le_iff.1 (sin_bound (by rwa [_root_.abs_of_nonneg (le_of_lt hx0)]))).2
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem sin_pos_of_pos_of_le_two {x : ℝ} (hx0 : 0 < x) (hx : x ≤ 2) : 0 < sin x :=
                                       /-
                                         x : Real
                                         hx0 : LT.lt 0 x
                                         hx : LE.le x 2
                                         ⊢ LT.lt 0 2
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  have : x / 2 ≤ 1 := (div_le_iff₀ (by norm_num)).mpr (by simpa)
                                                          /-
                                                            🎉 no goals
                                                          -/
  calc
    0 < 2 * sin (x / 2) * cos (x / 2) :=
                           /-
                             x : Real
                             hx0 : LT.lt 0 x
                             hx : LE.le x 2
                             this : LE.le (HDiv.hDiv x 2) 1
                             ⊢ LT.lt 0 2
                           -/
      mul_pos (mul_pos (by norm_num) (sin_pos_of_pos_of_le_one (half_pos hx0) this))
                           /-
                             🎉 no goals
                           -/
                               /-
                                 x : Real
                                 hx0 : LT.lt 0 x
                                 hx : LE.le x 2
                                 this : LE.le (HDiv.hDiv x 2) 1
                                 ⊢ LE.le (abs (HDiv.hDiv x 2)) 1
                               -/
        (cos_pos_of_le_one (by rwa [_root_.abs_of_nonneg (le_of_lt (half_pos hx0))]))
                               /-
                                 🎉 no goals
                               -/
                    /-
                      x : Real
                      hx0 : LT.lt 0 x
                      hx : LE.le x 2
                      this : LE.le (HDiv.hDiv x 2) 1
                      ⊢ Eq (HMul.hMul (HMul.hMul 2 (Real.sin (HDiv.hDiv x 2))) (Real.cos (HDiv.hDiv  …
                    -/
    _ = sin x := by rw [← sin_two_mul, two_mul, add_halves]
                    /-
                      🎉 no goals
                    -/


theorem cos_one_le : cos 1 ≤ 2 / 3 :=
  calc
    cos 1 ≤ |(1 : ℝ)| ^ 4 * (5 / 96) + (1 - 1 ^ 2 / 2) :=
                                                           /-
                                                             ⊢ LE.le (abs 1) 1
                                                           -/
      sub_le_iff_le_add.1 (abs_sub_le_iff.1 (cos_bound (by simp))).1
                                                           /-
                                                             🎉 no goals
                                                           -/
                    /-
                      ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (abs 1) 4) (5 / 96)) (HSub.hSub 1 (HD …
                    -/
    _ ≤ 2 / 3 := by norm_num
                    /-
                      🎉 no goals
                    -/


theorem cos_one_pos : 0 < cos 1 :=
  cos_pos_of_le_one (le_of_eq abs_one)


theorem cos_two_neg : cos 2 < 0 :=
  calc cos 2 = cos (2 * 1) := congr_arg cos (mul_one _).symm
    _ = _ := Real.cos_two_mul 1
    _ ≤ 2 * (2 / 3) ^ 2 - 1 := by
      /-
        ⊢ LE.le (HSub.hSub (HMul.hMul 2 (HPow.hPow (Real.cos 1) 2)) 1) (HSub.hSub (HMu …
      -/
      gcongr
        /-
          case h.h.ha
          ⊢ LE.le 0 (Real.cos 1)
        -/
      · exact cos_one_pos.le
        /-
          🎉 no goals
        -/
        /-
          case h.h.hab
          ⊢ LE.le (Real.cos 1) (2 / 3)
        -/
      · apply cos_one_le
        /-
          🎉 no goals
        -/
                /-
                  ⊢ LT.lt (HSub.hSub (HMul.hMul 2 (HPow.hPow (2 / 3) 2)) 1) 0
                -/
    _ < 0 := by norm_num
                /-
                  🎉 no goals
                -/


theorem exp_bound_div_one_sub_of_interval' {x : ℝ} (h1 : 0 < x) (h2 : x < 1) :
    Real.exp x < 1 / (1 - x) := by
  have H : 0 < 1 - (1 + x + x ^ 2) * (1 - x) := calc
    0 < x ^ 3 := by positivity
    _ = 1 - (1 + x + x ^ 2) * (1 - x) := by ring
  calc
    exp x ≤ _ := exp_bound' h1.le h2.le zero_lt_three
    _ ≤ 1 + x + x ^ 2 := by
      -- Porting note: was `norm_num [Finset.sum] <;> nlinarith`
      -- This proof should be restored after the norm_num plugin for big operators is ported.
      -- (It may also need the positivity extensions in https://github.com/leanprover-community/mathlib4/pull/3907.)
      erw [Finset.sum_range_succ]
      repeat rw [Finset.sum_range_succ]
      norm_num [Nat.factorial]
      nlinarith
    _ < 1 / (1 - x) := by rw [lt_div_iff₀] <;> nlinarith


theorem exp_bound_div_one_sub_of_interval {x : ℝ} (h1 : 0 ≤ x) (h2 : x < 1) :
    Real.exp x ≤ 1 / (1 - x) := by
  /-
    x : Real
    h1 : LE.le 0 x
    h2 : LT.lt x 1
    ⊢ LE.le (Real.exp x) (HDiv.hDiv 1 (HSub.hSub 1 x))
  -/
  rcases eq_or_lt_of_le h1 with (rfl | h1)
    /-
      case inl
      h1 : LE.le 0 0
      h2 : LT.lt 0 1
      ⊢ LE.le (Real.exp 0) (HDiv.hDiv 1 (HSub.hSub 1 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      h1✝ : LE.le 0 x
      h2 : LT.lt x 1
      h1 : LT.lt 0 x
      ⊢ LE.le (Real.exp x) (HDiv.hDiv 1 (HSub.hSub 1 x))
    -/
  · exact (exp_bound_div_one_sub_of_interval' h1 h2).le
    /-
      🎉 no goals
    -/


theorem add_one_lt_exp {x : ℝ} (hx : x ≠ 0) : x + 1 < Real.exp x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ LT.lt (HAdd.hAdd x 1) (Real.exp x)
  -/
  obtain hx | hx := hx.symm.lt_or_lt
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt 0 x
      ⊢ LT.lt (HAdd.hAdd x 1) (Real.exp x)
    -/
  · exact add_one_lt_exp_of_pos hx
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : Real
    hx✝ : Ne x 0
    hx : LT.lt x 0
    ⊢ LT.lt (HAdd.hAdd x 1) (Real.exp x)
  -/
  obtain h' | h' := le_or_lt 1 (-x)
    /-
      case inr.inl
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt x 0
      h' : LE.le 1 (Neg.neg x)
      ⊢ LT.lt (HAdd.hAdd x 1) (Real.exp x)
    -/
  · linarith [x.exp_pos]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    x : Real
    hx✝ : Ne x 0
    hx : LT.lt x 0
    h' : LT.lt (Neg.neg x) 1
    ⊢ LT.lt (HAdd.hAdd x 1) (Real.exp x)
  -/
  have hx' : 0 < x + 1 := by linarith
  simpa [add_comm, exp_neg, inv_lt_inv₀ (exp_pos _) hx']
    using exp_bound_div_one_sub_of_interval' (neg_pos.2 hx) h'


theorem add_one_le_exp (x : ℝ) : x + 1 ≤ Real.exp x := by
  /-
    x : Real
    ⊢ LE.le (HAdd.hAdd x 1) (Real.exp x)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      ⊢ LE.le (HAdd.hAdd 0 1) (Real.exp 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx : Ne x 0
      ⊢ LE.le (HAdd.hAdd x 1) (Real.exp x)
    -/
  · exact (add_one_lt_exp hx).le
    /-
      🎉 no goals
    -/


lemma one_sub_lt_exp_neg {x : ℝ} (hx : x ≠ 0) : 1 - x < exp (-x) :=
  (sub_eq_neg_add _ _).trans_lt <| add_one_lt_exp <| neg_ne_zero.2 hx


lemma one_sub_le_exp_neg (x : ℝ) : 1 - x ≤ exp (-x) :=
  (sub_eq_neg_add _ _).trans_le <| add_one_le_exp _


theorem one_sub_div_pow_le_exp_neg {n : ℕ} {t : ℝ} (ht' : t ≤ n) : (1 - t / n) ^ n ≤ exp (-t) := by
  /-
    n : Nat
    t : Real
    ht' : LE.le t ↑n
    ⊢ LE.le (HPow.hPow (HSub.hSub 1 (HDiv.hDiv t ↑n)) n) (Real.exp (Neg.neg t))
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      t : Real
      ht' : LE.le t ↑0
      ⊢ LE.le (HPow.hPow (HSub.hSub 1 (HDiv.hDiv t ↑0)) 0) (Real.exp (Neg.neg t))
    -/
  · simp
    /-
      case inl
      t : Real
      ht' : LE.le t ↑0
      ⊢ LE.le t 0
    -/
    rwa [Nat.cast_zero] at ht'
    /-
      🎉 no goals
    -/
  calc
    (1 - t / n) ^ n ≤ rexp (-(t / n)) ^ n := by
      gcongr
      · exact sub_nonneg.2 <| div_le_one_of_le₀ ht' n.cast_nonneg
      · exact one_sub_le_exp_neg _
    _ = rexp (-t) := by rw [← Real.exp_nat_mul, mul_neg, mul_comm, div_mul_cancel₀]; positivity


/-- Extension for the `positivity` tactic: `Real.exp` is always positive. -/
@[positivity Real.exp _]
def evalExp : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(Real.exp $a) =>
    assertInstancesCommute
    pure (.positive q(Real.exp_pos $a))
  | _, _, _ => throwError "not Real.exp"


/-- Extension for the `positivity` tactic: `Real.cosh` is always positive. -/
@[positivity Real.cosh _]
def evalCosh : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(Real.cosh $a) =>
    assertInstancesCommute
    return .positive q(Real.cosh_pos $a)
  | _, _, _ => throwError "not Real.cosh"


@[simp]
theorem abs_cos_add_sin_mul_I (x : ℝ) : abs (cos x + sin x * I) = 1 := by
  /-
    x : Real
    ⊢ Eq (Complex.abs (HAdd.hAdd (Complex.cos ↑x) (HMul.hMul (Complex.sin ↑x) Comp …
  -/
  have := Real.sin_sq_add_cos_sq x
  /-
    x : Real
    this : Eq (HAdd.hAdd (HPow.hPow (Real.sin x) 2) (HPow.hPow (Real.cos x) 2)) 1
    ⊢ Eq (Complex.abs (HAdd.hAdd (Complex.cos ↑x) (HMul.hMul (Complex.sin ↑x) Comp …
  -/
  simp_all [add_comm, abs, normSq, sq, sin_ofReal_re, cos_ofReal_re, mul_re]
  /-
    🎉 no goals
  -/


@[simp]
theorem abs_exp_ofReal (x : ℝ) : abs (exp x) = Real.exp x := by
  /-
    x : Real
    ⊢ Eq (Complex.abs (Complex.exp ↑x)) (Real.exp x)
  -/
  rw [← ofReal_exp]
  /-
    x : Real
    ⊢ Eq (Complex.abs ↑(Real.exp x)) (Real.exp x)
  -/
  exact abs_of_nonneg (le_of_lt (Real.exp_pos _))
  /-
    🎉 no goals
  -/


@[simp]
theorem abs_exp_ofReal_mul_I (x : ℝ) : abs (exp (x * I)) = 1 := by
  /-
    x : Real
    ⊢ Eq (Complex.abs (Complex.exp (HMul.hMul (↑x) Complex.I))) 1
  -/
  rw [exp_mul_I, abs_cos_add_sin_mul_I]
  /-
    🎉 no goals
  -/


theorem abs_exp (z : ℂ) : abs (exp z) = Real.exp z.re := by
  /-
    z : Complex
    ⊢ Eq (Complex.abs (Complex.exp z)) (Real.exp z.re)
  -/
  rw [exp_eq_exp_re_mul_sin_add_cos, map_mul, abs_exp_ofReal, abs_cos_add_sin_mul_I, mul_one]
  /-
    🎉 no goals
  -/


theorem abs_exp_eq_iff_re_eq {x y : ℂ} : abs (exp x) = abs (exp y) ↔ x.re = y.re := by
  /-
    x y : Complex
    ⊢ Iff (Eq (Complex.abs (Complex.exp x)) (Complex.abs (Complex.exp y))) (Eq x.r …
  -/
  rw [abs_exp, abs_exp, Real.exp_eq_exp]
  /-
    🎉 no goals
  -/


