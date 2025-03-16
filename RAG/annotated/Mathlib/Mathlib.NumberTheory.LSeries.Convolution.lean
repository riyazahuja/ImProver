/-- We turn any function `ℕ → R` into an `ArithmeticFunction R` by setting its value at `0`
to be zero. -/
def toArithmeticFunction {R : Type*} [Zero R] (f : ℕ → R) : ArithmeticFunction R where
  toFun n := if n = 0 then 0 else f n
  map_zero' := rfl


lemma toArithmeticFunction_congr {R : Type*} [Zero R] {f f' : ℕ → R}
    (h : ∀ {n}, n ≠ 0 → f n = f' n) :
    toArithmeticFunction f = toArithmeticFunction f' := by
  /-
    R : Type u_1
    inst✝ : Zero R
    f f' : Nat → R
    h : ∀ {n : Nat}, Ne n 0 → Eq (f n) (f' n)
    ⊢ Eq (toArithmeticFunction f) (toArithmeticFunction f')
  -/
  ext ⟨- | _⟩
    /-
      case h.zero
      R : Type u_1
      inst✝ : Zero R
      f f' : Nat → R
      h : ∀ {n : Nat}, Ne n 0 → Eq (f n) (f' n)
      ⊢ Eq ((toArithmeticFunction f) 0) ((toArithmeticFunction f') 0)
    -/
  · simp only [zero_eq, ArithmeticFunction.map_zero]
    /-
      🎉 no goals
    -/
  · simp only [toArithmeticFunction, ArithmeticFunction.coe_mk, succ_ne_zero, ↓reduceIte,
      ne_eq, not_false_eq_true, h]


/-- If we consider an arithmetic function just as a function and turn it back into an
arithmetic function, it is the same as before. -/
@[simp]
lemma ArithmeticFunction.toArithmeticFunction_eq_self {R : Type*} [Zero R]
    (f : ArithmeticFunction R) :
    toArithmeticFunction f = f := by
  /-
    R : Type u_1
    inst✝ : Zero R
    f : ArithmeticFunction R
    ⊢ Eq (toArithmeticFunction ⇑f) f
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : Zero R
    f : ArithmeticFunction R
    n : Nat
    ⊢ Eq ((toArithmeticFunction ⇑f) n) (f n)
  -/
  simp (config := {contextual := true}) [toArithmeticFunction, ArithmeticFunction.map_zero]
  /-
    🎉 no goals
  -/


/-- Dirichlet convolution of two sequences.

We define this in terms of the already existing definition for arithmetic functions. -/
noncomputable def LSeries.convolution {R : Type*} [Semiring R] (f g : ℕ → R) : ℕ → R :=
  ⇑(toArithmeticFunction f * toArithmeticFunction g)


@[inherit_doc]
scoped[LSeries.notation] infixl:70 " ⍟ " => LSeries.convolution


lemma LSeries.convolution_congr {R : Type*} [Semiring R] {f f' g g' : ℕ → R}
    (hf : ∀ {n}, n ≠ 0 → f n = f' n) (hg : ∀ {n}, n ≠ 0 → g n = g' n) :
    f ⍟ g = f' ⍟ g' := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f f' g g' : Nat → R
    hf : ∀ {n : Nat}, Ne n 0 → Eq (f n) (f' n)
    hg : ∀ {n : Nat}, Ne n 0 → Eq (g n) (g' n)
    ⊢ Eq (LSeries.convolution f g) (LSeries.convolution f' g')
  -/
  simp only [convolution, toArithmeticFunction_congr hf, toArithmeticFunction_congr hg]
  /-
    🎉 no goals
  -/


/-- The product of two arithmetic functions defines the same function as the Dirichlet convolution
of the functions defined by them. -/
lemma ArithmeticFunction.coe_mul {R : Type*} [Semiring R] (f g : ArithmeticFunction R) :
    f ⍟ g = ⇑(f * g) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : ArithmeticFunction R
    ⊢ Eq (LSeries.convolution ⇑f ⇑g) ⇑(HMul.hMul f g)
  -/
  simp only [convolution, ArithmeticFunction.toArithmeticFunction_eq_self]
  /-
    🎉 no goals
  -/


lemma convolution_def {R : Type*} [Semiring R] (f g : ℕ → R) :
    f ⍟ g = fun n ↦ ∑ p ∈ n.divisorsAntidiagonal, f p.1 * g p.2 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : Nat → R
    ⊢ Eq (LSeries.convolution f g) fun n => n.divisorsAntidiagonal.sum fun p => HM …
  -/
  ext n
  simp only [convolution, toArithmeticFunction, ArithmeticFunction.mul_apply,
    ArithmeticFunction.coe_mk, mul_ite, mul_zero, ite_mul, zero_mul]
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f g : Nat → R
    n : Nat
    ⊢ Eq (n.divisorsAntidiagonal.sum fun x => ite (Eq x.2 0) 0 (ite (Eq x.1 0) 0 ( …
  -/
  refine Finset.sum_congr rfl fun p hp ↦ ?_
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f g : Nat → R
    n : Nat
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    ⊢ Eq (ite (Eq p.2 0) 0 (ite (Eq p.1 0) 0 (HMul.hMul (f p.1) (g p.2)))) (HMul.h …
  -/
  obtain ⟨h₁, h₂⟩ := ne_zero_of_mem_divisorsAntidiagonal hp
  /-
    case h.intro
    R : Type u_1
    inst✝ : Semiring R
    f g : Nat → R
    n : Nat
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    h₁ : Ne p.1 0
    h₂ : Ne p.2 0
    ⊢ Eq (ite (Eq p.2 0) 0 (ite (Eq p.1 0) 0 (HMul.hMul (f p.1) (g p.2)))) (HMul.h …
  -/
  simp only [h₂, ↓reduceIte, h₁]
  /-
    🎉 no goals
  -/


@[simp]
lemma convolution_map_zero {R : Type*} [Semiring R] (f g : ℕ → R) : (f ⍟ g) 0 = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : Nat → R
    ⊢ Eq (LSeries.convolution f g 0) 0
  -/
  simp only [convolution_def, divisorsAntidiagonal_zero, Finset.sum_empty]
  /-
    🎉 no goals
  -/



/-- We give an expression of the `LSeries.term` of the convolution of two functions
in terms of a sum over `Nat.divisorsAntidiagonal`. -/
lemma term_convolution (f g : ℕ → ℂ) (s : ℂ) (n : ℕ) :
    term (f ⍟ g) s n = ∑ p ∈ n.divisorsAntidiagonal, term f s p.1 * term g s p.2 := by
  /-
    f g : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (LSeries.convolution f g) s n) (n.divisorsAntidiagonal.sum  …
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      f g : Nat → Complex
      s : Complex
      ⊢ Eq (LSeries.term (LSeries.convolution f g) s 0) ((Nat.divisorsAntidiagonal 0 …
    -/
  · simp only [term_zero, divisorsAntidiagonal_zero, Finset.sum_empty]
    /-
      🎉 no goals
    -/
  -- now `n ≠ 0`
  /-
    case inr
    f g : Nat → Complex
    s : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Eq (LSeries.term (LSeries.convolution f g) s n) (n.divisorsAntidiagonal.sum  …
  -/
  rw [term_of_ne_zero hn, convolution_def, Finset.sum_div]
  /-
    case inr
    f g : Nat → Complex
    s : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Eq (n.divisorsAntidiagonal.sum fun i => HDiv.hDiv (HMul.hMul (f i.1) (g i.2) …
  -/
  refine Finset.sum_congr rfl fun p hp ↦ ?_
  /-
    case inr
    f g : Nat → Complex
    s : Complex
    n : Nat
    hn : Ne n 0
    p : Prod Nat Nat
    hp : Membership.mem n.divisorsAntidiagonal p
    ⊢ Eq (HDiv.hDiv (HMul.hMul (f p.1) (g p.2)) (HPow.hPow (↑n) s)) (HMul.hMul (LS …
  -/
  have ⟨hp₁, hp₂⟩ := ne_zero_of_mem_divisorsAntidiagonal hp
  rw [term_of_ne_zero hp₁ f s, term_of_ne_zero hp₂ g s, mul_comm_div, div_div, ← mul_div_assoc,
    ← natCast_mul_natCast_cpow, ← cast_mul, mul_comm p.2, (mem_divisorsAntidiagonal.mp hp).1]


open Set in
/-- We give an expression of the `LSeries.term` of the convolution of two functions
in terms of an a priori infinite sum over all pairs `(k, m)` with `k * m = n`
(the set we sum over is infinite when `n = 0`). This is the version needed for the
proof that `L (f ⍟ g) = L f * L g`. -/
lemma term_convolution' (f g : ℕ → ℂ) (s : ℂ) :
    term (f ⍟ g) s = fun n ↦
      ∑' (b : (fun p : ℕ × ℕ ↦ p.1 * p.2) ⁻¹' {n}), term f s b.val.1 * term g s b.val.2 := by
  /-
    f g : Nat → Complex
    s : Complex
    ⊢ Eq (LSeries.term (LSeries.convolution f g) s) fun n => tsum fun b => HMul.hM …
  -/
  ext n
  /-
    case h
    f g : Nat → Complex
    s : Complex
    n : Nat
    ⊢ Eq (LSeries.term (LSeries.convolution f g) s n) (tsum fun b => HMul.hMul (LS …
  -/
  rcases eq_or_ne n 0 with rfl | hn
  · -- show that both sides vanish when `n = 0`; this is the hardest part of the proof!
    /-
      case h.inl
      f g : Nat → Complex
      s : Complex
      ⊢ Eq (LSeries.term (LSeries.convolution f g) s 0) (tsum fun b => HMul.hMul (LS …
    -/
    refine (term_zero ..).trans ?_
    -- the right hand sum is over the union below, but in each term, one factor is always zero
    have hS : (fun p ↦ p.1 * p.2) ⁻¹' {0} = {0} ×ˢ univ ∪ univ ×ˢ {0} := by
      ext
      simp only [mem_preimage, mem_singleton_iff, Nat.mul_eq_zero, mem_union, mem_prod, mem_univ,
        and_true, true_and]
    have : ∀ p : (fun p : ℕ × ℕ ↦ p.1 * p.2) ⁻¹' {0}, term f s p.val.1 * term g s p.val.2 = 0 := by
      rintro ⟨⟨p₁, p₂⟩, hp⟩
      rcases hS ▸ hp with ⟨rfl, -⟩ | ⟨-, rfl⟩ <;> simp only [term_zero, zero_mul, mul_zero]
    /-
      case h.inl
      f g : Nat → Complex
      s : Complex
      hS : Eq (Set.preimage (fun p => HMul.hMul p.1 p.2) (Singleton.singleton 0)) (U …
      this : ∀ (p : ↑(Set.preimage (fun p => HMul.hMul p.1 p.2) (Singleton.singleton …
      ⊢ Eq 0 (tsum fun b => HMul.hMul (LSeries.term f s (↑b).1) (LSeries.term g s (↑ …
    -/
    simp only [this, tsum_zero]
    /-
      🎉 no goals
    -/
  -- now `n ≠ 0`
  rw [show (fun p : ℕ × ℕ ↦ p.1 * p.2) ⁻¹' {n} = n.divisorsAntidiagonal by ext; simp [hn],
    Finset.tsum_subtype' n.divisorsAntidiagonal fun p ↦ term f s p.1 * term g s p.2,
    term_convolution f g s n]


open Set in
/-- The L-series of the convolution product `f ⍟ g` of two sequences `f` and `g`
equals the product of their L-series, assuming both L-series converge. -/
lemma LSeriesHasSum.convolution {f g : ℕ → ℂ} {s a b : ℂ} (hf : LSeriesHasSum f s a)
    (hg : LSeriesHasSum g s b) :
    LSeriesHasSum (f ⍟ g) s (a * b) := by
  /-
    f g : Nat → Complex
    s a b : Complex
    hf : LSeriesHasSum f s a
    hg : LSeriesHasSum g s b
    ⊢ LSeriesHasSum (LSeries.convolution f g) s (HMul.hMul a b)
  -/
  simp only [LSeriesHasSum, term_convolution']
  /-
    f g : Nat → Complex
    s a b : Complex
    hf : LSeriesHasSum f s a
    hg : LSeriesHasSum g s b
    ⊢ HasSum (fun n => tsum fun b => HMul.hMul (LSeries.term f s (↑b).1) (LSeries. …
  -/
  have hsum := summable_mul_of_summable_norm hf.summable.norm hg.summable.norm
  /-
    f g : Nat → Complex
    s a b : Complex
    hf : LSeriesHasSum f s a
    hg : LSeriesHasSum g s b
    hsum : Summable fun x => HMul.hMul (LSeries.term f s x.1) (LSeries.term g s x.2)
    ⊢ HasSum (fun n => tsum fun b => HMul.hMul (LSeries.term f s (↑b).1) (LSeries. …
  -/
  exact (HasSum.mul hf hg hsum).tsum_fiberwise (fun p ↦ p.1 * p.2)
  /-
    🎉 no goals
  -/


/-- The L-series of the convolution product `f ⍟ g` of two sequences `f` and `g`
equals the product of their L-series, assuming both L-series converge. -/
lemma LSeries_convolution' {f g : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s)
    (hg : LSeriesSummable g s) :
    LSeries (f ⍟ g) s = LSeries f s * LSeries g s :=
  (LSeriesHasSum.convolution hf.LSeriesHasSum hg.LSeriesHasSum).LSeries_eq


/-- The L-series of the convolution product `f ⍟ g` of two sequences `f` and `g`
equals the product of their L-series in their common half-plane of absolute convergence. -/
lemma LSeries_convolution {f g : ℕ → ℂ} {s : ℂ}
    (hf : abscissaOfAbsConv f < s.re) (hg : abscissaOfAbsConv g < s.re) :
    LSeries (f ⍟ g) s = LSeries f s * LSeries g s :=
  LSeries_convolution' (LSeriesSummable_of_abscissaOfAbsConv_lt_re hf)
    (LSeriesSummable_of_abscissaOfAbsConv_lt_re hg)


/-- The L-series of the convolution product `f ⍟ g` of two sequences `f` and `g`
is summable when both L-series are summable. -/
lemma LSeriesSummable.convolution {f g : ℕ → ℂ} {s : ℂ} (hf : LSeriesSummable f s)
    (hg : LSeriesSummable g s) :
    LSeriesSummable (f ⍟ g) s :=
  (LSeriesHasSum.convolution hf.LSeriesHasSum hg.LSeriesHasSum).LSeriesSummable


/-- The abscissa of absolute convergence of `f ⍟ g` is at most the maximum of those
of `f` and `g`. -/
lemma LSeries.abscissaOfAbsConv_convolution_le (f g : ℕ → ℂ) :
    abscissaOfAbsConv (f ⍟ g) ≤ max (abscissaOfAbsConv f) (abscissaOfAbsConv g) :=
  abscissaOfAbsConv_binop_le LSeriesSummable.convolution f g


/-- The L-series of the (convolution) product of two `ℂ`-valued arithmetic functions `f` and `g`
equals the product of their L-series, assuming both L-series converge. -/
lemma LSeriesHasSum_mul {f g : ArithmeticFunction ℂ} {s a b : ℂ} (hf : LSeriesHasSum ↗f s a)
    (hg : LSeriesHasSum ↗g s b) :
    LSeriesHasSum ↗(f * g) s (a * b) :=
  coe_mul f g ▸ hf.convolution hg


/-- The L-series of the (convolution) product of two `ℂ`-valued arithmetic functions `f` and `g`
equals the product of their L-series, assuming both L-series converge. -/
lemma LSeries_mul' {f g : ArithmeticFunction ℂ} {s : ℂ} (hf : LSeriesSummable ↗f s)
    (hg : LSeriesSummable ↗g s) :
    LSeries ↗(f * g) s = LSeries ↗f s * LSeries ↗g s :=
  coe_mul f g ▸ LSeries_convolution' hf hg


/-- The L-series of the (convolution) product of two `ℂ`-valued arithmetic functions `f` and `g`
equals the product of their L-series in their common half-plane of absolute convergence. -/
lemma LSeries_mul {f g : ArithmeticFunction ℂ} {s : ℂ}
    (hf : abscissaOfAbsConv ↗f < s.re) (hg : abscissaOfAbsConv ↗g < s.re) :
    LSeries ↗(f * g) s = LSeries ↗f s * LSeries ↗g s :=
  coe_mul f g ▸ LSeries_convolution hf hg


/-- The L-series of the (convolution) product of two `ℂ`-valued arithmetic functions `f` and `g`
is summable when both L-series are summable. -/
lemma LSeriesSummable_mul {f g : ArithmeticFunction ℂ} {s : ℂ} (hf : LSeriesSummable ↗f s)
    (hg : LSeriesSummable ↗g s) :
    LSeriesSummable ↗(f * g) s :=
  coe_mul f g ▸ hf.convolution hg


