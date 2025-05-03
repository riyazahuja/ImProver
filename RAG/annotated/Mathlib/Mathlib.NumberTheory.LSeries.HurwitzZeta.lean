/-- The Hurwitz zeta function, which is the meromorphic continuation of
`∑ (n : ℕ), 1 / (n + a) ^ s` if `0 ≤ a ≤ 1`. See `hasSum_hurwitzZeta_of_one_lt_re` for the relation
to the Dirichlet series in the convergence range. -/
noncomputable def hurwitzZeta (a : UnitAddCircle) (s : ℂ) :=
  hurwitzZetaEven a s + hurwitzZetaOdd a s


lemma hurwitzZetaEven_eq (a : UnitAddCircle) (s : ℂ) :
    hurwitzZetaEven a s = (hurwitzZeta a s + hurwitzZeta (-a) s) / 2 := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZetaEven a s) (HDiv.hDiv (HAdd.hAdd (HurwitzZeta.hurw …
  -/
  simp only [hurwitzZeta, hurwitzZetaEven_neg, hurwitzZetaOdd_neg]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZetaEven a s) (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd (Hurwi …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma hurwitzZetaOdd_eq (a : UnitAddCircle) (s : ℂ) :
    hurwitzZetaOdd a s = (hurwitzZeta a s - hurwitzZeta (-a) s) / 2 := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZetaOdd a s) (HDiv.hDiv (HSub.hSub (HurwitzZeta.hurwi …
  -/
  simp only [hurwitzZeta, hurwitzZetaEven_neg, hurwitzZetaOdd_neg]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.hurwitzZetaOdd a s) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (Hurwit …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- The Hurwitz zeta function is differentiable away from `s = 1`. -/
lemma differentiableAt_hurwitzZeta (a : UnitAddCircle) {s : ℂ} (hs : s ≠ 1) :
    DifferentiableAt ℂ (hurwitzZeta a) s :=
  (differentiableAt_hurwitzZetaEven a hs).add (differentiable_hurwitzZetaOdd a s)


/-- Formula for `hurwitzZeta s` as a Dirichlet series in the convergence range. We
restrict to `a ∈ Icc 0 1` to simplify the statement. -/
lemma hasSum_hurwitzZeta_of_one_lt_re {a : ℝ} (ha : a ∈ Icc 0 1) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ 1 / (n + a : ℂ) ^ s) (hurwitzZeta a s) := by
  convert (hasSum_nat_hurwitzZetaEven_of_mem_Icc ha hs).add
      (hasSum_nat_hurwitzZetaOdd_of_mem_Icc ha hs) using 1
  /-
    case h.e'_5
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (fun n => HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n ↑a) s)) fun b => HAdd.hAdd …
  -/
  ext1 n
  -- plain `ring_nf` works here, but the following is faster:
  /-
    case h.e'_5.h
    a : Real
    ha : Membership.mem (Set.Icc 0 1) a
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (HAdd.hAdd ↑n ↑a) s)) (HAdd.hAdd (HDiv.hDiv (HAdd …
  -/
  apply show ∀ (x y : ℂ), x = (x + y) / 2 + (x - y) / 2 by intros; ring
  /-
    🎉 no goals
  -/


/-- The residue of the Hurwitz zeta function at `s = 1` is `1`. -/
lemma hurwitzZeta_residue_one (a : UnitAddCircle) :
    Tendsto (fun s ↦ (s - 1) * hurwitzZeta a s) (𝓝[≠] 1) (𝓝 1) := by
  /-
    a : UnitAddCircle
    ⊢ Filter.Tendsto (fun s => HMul.hMul (HSub.hSub s 1) (HurwitzZeta.hurwitzZeta  …
  -/
  simp only [hurwitzZeta, mul_add, (by simp : 𝓝 (1 : ℂ) = 𝓝 (1 + (1 - 1) * hurwitzZetaOdd a 1))]
  /-
    a : UnitAddCircle
    ⊢ Filter.Tendsto (fun s => HAdd.hAdd (HMul.hMul (HSub.hSub s 1) (HurwitzZeta.h …
  -/
  refine (hurwitzZetaEven_residue_one a).add ((Tendsto.mul ?_ ?_).mono_left nhdsWithin_le_nhds)
  /-
    case refine_1
    a : UnitAddCircle
    ⊢ Filter.Tendsto (fun s => HSub.hSub s 1) (nhds 1) (nhds (HSub.hSub 1 1))
  -/
  exacts [tendsto_id.sub_const _, (differentiable_hurwitzZetaOdd a).continuous.tendsto _]
  /-
    🎉 no goals
  -/


lemma differentiableAt_hurwitzZeta_sub_one_div (a : UnitAddCircle) :
    DifferentiableAt ℂ (fun s ↦ hurwitzZeta a s - 1 / (s - 1) / Gammaℝ s) 1 := by
  /-
    a : UnitAddCircle
    ⊢ DifferentiableAt Complex (fun s => HSub.hSub (HurwitzZeta.hurwitzZeta a s) ( …
  -/
  simp only [hurwitzZeta, add_sub_right_comm]
  /-
    a : UnitAddCircle
    ⊢ DifferentiableAt Complex (fun s => HAdd.hAdd (HSub.hSub (HurwitzZeta.hurwitz …
  -/
  exact (differentiableAt_hurwitzZetaEven_sub_one_div a).add (differentiable_hurwitzZetaOdd a 1)
  /-
    🎉 no goals
  -/


/-- Expression for `hurwitzZeta a 1` as a limit. (Mathematically `hurwitzZeta a 1` is
undefined, but our construction assigns some value to it; this lemma is mostly of interest for
determining what that value is). -/
lemma tendsto_hurwitzZeta_sub_one_div_nhds_one (a : UnitAddCircle) :
    Tendsto (fun s ↦ hurwitzZeta a s - 1 / (s - 1) / Gammaℝ s) (𝓝 1) (𝓝 (hurwitzZeta a 1)) := by
  /-
    a : UnitAddCircle
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HurwitzZeta.hurwitzZeta a s) (HDiv.hDiv  …
  -/
  simp only [hurwitzZeta, add_sub_right_comm]
  /-
    a : UnitAddCircle
    ⊢ Filter.Tendsto (fun s => HAdd.hAdd (HSub.hSub (HurwitzZeta.hurwitzZetaEven a …
  -/
  refine (tendsto_hurwitzZetaEven_sub_one_div_nhds_one a).add ?_
  /-
    a : UnitAddCircle
    ⊢ Filter.Tendsto (HurwitzZeta.hurwitzZetaOdd a) (nhds 1) (nhds (HurwitzZeta.hu …
  -/
  exact (differentiable_hurwitzZetaOdd a 1).continuousAt.tendsto
  /-
    🎉 no goals
  -/


/-- The difference of two Hurwitz zeta functions is differentiable everywhere. -/
lemma differentiable_hurwitzZeta_sub_hurwitzZeta (a b : UnitAddCircle) :
    Differentiable ℂ (fun s ↦ hurwitzZeta a s - hurwitzZeta b s) := by
  /-
    a b : UnitAddCircle
    ⊢ Differentiable Complex fun s => HSub.hSub (HurwitzZeta.hurwitzZeta a s) (Hur …
  -/
  simp only [hurwitzZeta, add_sub_add_comm]
  /-
    a b : UnitAddCircle
    ⊢ Differentiable Complex fun s => HAdd.hAdd (HSub.hSub (HurwitzZeta.hurwitzZet …
  -/
  refine (differentiable_hurwitzZetaEven_sub_hurwitzZetaEven a b).add (Differentiable.sub ?_ ?_)
  /-
    case refine_1
    a b : UnitAddCircle
    ⊢ Differentiable Complex (HurwitzZeta.hurwitzZetaOdd a)
  -/
  all_goals apply differentiable_hurwitzZetaOdd
  /-
    🎉 no goals
  -/


/-- Meromorphic continuation of the series `∑' (n : ℕ), exp (2 * π * I * a * n) / n ^ s`.  See
`hasSum_expZeta_of_one_lt_re` for the relation to the Dirichlet series. -/
noncomputable def expZeta (a : UnitAddCircle) (s : ℂ) :=
  cosZeta a s + I * sinZeta a s


lemma cosZeta_eq (a : UnitAddCircle) (s : ℂ) :
    cosZeta a s = (expZeta a s + expZeta (-a) s) / 2 := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.cosZeta a s) (HDiv.hDiv (HAdd.hAdd (HurwitzZeta.expZeta a s) …
  -/
  rw [expZeta, expZeta, cosZeta_neg, sinZeta_neg]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.cosZeta a s) (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd (HurwitzZeta.c …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma sinZeta_eq (a : UnitAddCircle) (s : ℂ) :
    sinZeta a s = (expZeta a s - expZeta (-a) s) / (2 * I) := by
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.sinZeta a s) (HDiv.hDiv (HSub.hSub (HurwitzZeta.expZeta a s) …
  -/
  rw [expZeta, expZeta, cosZeta_neg, sinZeta_neg]
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HurwitzZeta.sinZeta a s) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HurwitzZeta.c …
  -/
  field_simp
  /-
    a : UnitAddCircle
    s : Complex
    ⊢ Eq (HMul.hMul (HurwitzZeta.sinZeta a s) (HMul.hMul 2 Complex.I)) (HAdd.hAdd  …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma hasSum_expZeta_of_one_lt_re (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    HasSum (fun n : ℕ ↦ cexp (2 * π * I * a * n) / n ^ s) (expZeta a s) := by
  /-
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ HasSum (fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HM …
  -/
  convert (hasSum_nat_cosZeta a hs).add ((hasSum_nat_sinZeta a hs).mul_left I) using 1
  /-
    case h.e'_5
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (fun n => HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HMul.h …
  -/
  ext1 n
  /-
    case h.e'_5.h
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Re …
  -/
  simp only [mul_right_comm _ I, ← cos_add_sin_I, push_cast]
  /-
    case h.e'_5.h
    a : Real
    s : Complex
    hs : LT.lt 1 s.re
    n : Nat
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Complex.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Re …
  -/
  rw [add_div, mul_div, mul_comm _ I]
  /-
    🎉 no goals
  -/


lemma differentiableAt_expZeta (a : UnitAddCircle) (s : ℂ) (hs : s ≠ 1 ∨ a ≠ 0) :
    DifferentiableAt ℂ (expZeta a) s := by
  /-
    a : UnitAddCircle
    s : Complex
    hs : Or (Ne s 1) (Ne a 0)
    ⊢ DifferentiableAt Complex (HurwitzZeta.expZeta a) s
  -/
  apply DifferentiableAt.add
    /-
      case hf
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 1) (Ne a 0)
      ⊢ DifferentiableAt Complex (HurwitzZeta.cosZeta a) s
    -/
  · exact differentiableAt_cosZeta a hs
    /-
      🎉 no goals
    -/
    /-
      case hg
      a : UnitAddCircle
      s : Complex
      hs : Or (Ne s 1) (Ne a 0)
      ⊢ DifferentiableAt Complex (fun y => HMul.hMul Complex.I (HurwitzZeta.sinZeta  …
    -/
  · apply (differentiableAt_const _).mul (differentiableAt_sinZeta a s)
    /-
      🎉 no goals
    -/


/-- If `a ≠ 0` then the exponential zeta function is analytic everywhere. -/
lemma differentiable_expZeta_of_ne_zero {a : UnitAddCircle} (ha : a ≠ 0) :
    Differentiable ℂ (expZeta a) :=
  (differentiableAt_expZeta a · (Or.inr ha))


/-- Reformulation of `hasSum_expZeta_of_one_lt_re` using `LSeriesHasSum`. -/
lemma LSeriesHasSum_exp (a : ℝ) {s : ℂ} (hs : 1 < re s) :
    LSeriesHasSum (cexp <| 2 * π * I * a * ·) s (expZeta a s) :=
  (hasSum_expZeta_of_one_lt_re a hs).congr_fun
    (LSeries.term_of_ne_zero' (ne_zero_of_one_lt_re hs) _)


lemma hurwitzZeta_one_sub (a : UnitAddCircle) {s : ℂ}
    (hs : ∀ (n : ℕ), s ≠ -n) (hs' : a ≠ 0 ∨ s ≠ 1) :
    hurwitzZeta a (1 - s) = (2 * π) ^ (-s) * Gamma s *
    (exp (-π * I * s / 2) * expZeta a s + exp (π * I * s / 2) * expZeta (-a) s) := by
  rw [hurwitzZeta, hurwitzZetaEven_one_sub a hs hs', hurwitzZetaOdd_one_sub a hs,
    expZeta, expZeta, Complex.cos, Complex.sin, sinZeta_neg, cosZeta_neg]
  rw [show ↑π * I * s / 2 = ↑π * s / 2 * I by ring,
    show -↑π * I * s / 2 = -(↑π * s / 2) * I by ring]
  -- these `generalize` commands are not strictly needed for the `ring_nf` call to succeed, but
  -- make it run faster:
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Ne a 0) (Ne s 1)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 (HPow.hPow (HMul …
  -/
  generalize (2 * π : ℂ) ^ (-s) = x
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Ne a 0) (Ne s 1)
    x : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 x) (Complex.Gamm …
  -/
  generalize (↑π * s / 2 * I).exp = y
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Ne a 0) (Ne s 1)
    x y : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 x) (Complex.Gamm …
  -/
  generalize (-(↑π * s / 2) * I).exp = z
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    hs' : Or (Ne a 0) (Ne s 1)
    x y z : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 x) (Complex.Gamm …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- Functional equation for the exponential zeta function. -/
lemma expZeta_one_sub (a : UnitAddCircle) {s : ℂ} (hs : ∀ (n : ℕ), s ≠ 1 - n) :
    expZeta a (1 - s) = (2 * π) ^ (-s) * Gamma s *
    (exp (π * I * s / 2) * hurwitzZeta a s + exp (-π * I * s / 2) * hurwitzZeta (-a) s) := by
  have hs' (n : ℕ) : s ≠ -↑n := by
    convert hs (n + 1) using 1
    push_cast
    ring
  rw [expZeta, cosZeta_one_sub a hs, sinZeta_one_sub a hs', hurwitzZeta, hurwitzZeta,
    hurwitzZetaEven_neg, hurwitzZetaOdd_neg, Complex.cos, Complex.sin]
  rw [show ↑π * I * s / 2 = ↑π * s / 2 * I by ring,
    show -↑π * I * s / 2 = -(↑π * s / 2) * I by ring]
  -- these `generalize` commands are not strictly needed for the `ring_nf` call to succeed, but
  -- make it run faster:
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    hs' : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 (HPow.hPow (HMul …
  -/
  generalize (2 * π : ℂ) ^ (-s) = x
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    hs' : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    x : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 x) (Complex.Gamm …
  -/
  generalize (↑π * s / 2 * I).exp = y
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    hs' : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    x y : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 x) (Complex.Gamm …
  -/
  generalize (-(↑π * s / 2) * I).exp = z
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    hs' : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    x y z : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 x) (Complex.Gamm …
  -/
  ring_nf
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    hs' : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    x y z : Complex
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul x (Comp …
  -/
  rw [I_sq]
  /-
    a : UnitAddCircle
    s : Complex
    hs : ∀ (n : Nat), Ne s (HSub.hSub 1 ↑n)
    hs' : ∀ (n : Nat), Ne s (Neg.neg ↑n)
    x y z : Complex
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul x (Comp …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


