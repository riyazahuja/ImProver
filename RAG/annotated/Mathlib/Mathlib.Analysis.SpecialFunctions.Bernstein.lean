/-- The Bernstein polynomials, as continuous functions on `[0,1]`.
-/
def bernstein (n ν : ℕ) : C(I, ℝ) :=
  (bernsteinPolynomial ℝ n ν).toContinuousMapOn I


@[simp]
theorem bernstein_apply (n ν : ℕ) (x : I) :
    bernstein n ν x = (n.choose ν : ℝ) * (x : ℝ) ^ ν * (1 - (x : ℝ)) ^ (n - ν) := by
  /-
    n ν : Nat
    x : ↑unitInterval
    ⊢ Eq ((bernstein n ν) x) (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow (↑x) …
  -/
  dsimp [bernstein, Polynomial.toContinuousMapOn, Polynomial.toContinuousMap, bernsteinPolynomial]
  /-
    n ν : Nat
    x : ↑unitInterval
    ⊢ Eq (Polynomial.eval (↑x) (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow Po …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem bernstein_nonneg {n ν : ℕ} {x : I} : 0 ≤ bernstein n ν x := by
  /-
    n ν : Nat
    x : ↑unitInterval
    ⊢ LE.le 0 ((bernstein n ν) x)
  -/
  simp only [bernstein_apply]
  /-
    n ν : Nat
    x : ↑unitInterval
    ⊢ LE.le 0 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow (↑x) ν)) (HPow.hPow …
  -/
  have h₁ : (0 : ℝ) ≤ x := by unit_interval
  /-
    n ν : Nat
    x : ↑unitInterval
    h₁ : LE.le 0 ↑x
    ⊢ LE.le 0 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow (↑x) ν)) (HPow.hPow …
  -/
  have h₂ : (0 : ℝ) ≤ 1 - x := by unit_interval
  /-
    n ν : Nat
    x : ↑unitInterval
    h₁ : LE.le 0 ↑x
    h₂ : LE.le 0 (HSub.hSub 1 ↑x)
    ⊢ LE.le 0 (HMul.hMul (HMul.hMul (↑(n.choose ν)) (HPow.hPow (↑x) ν)) (HPow.hPow …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- Extension of the `positivity` tactic for Bernstein polynomials: they are always non-negative. -/
@[positivity DFunLike.coe _ _]
def evalBernstein : PositivityExt where eval {_ _} _zα _pα e := do
  let .app (.app _coe (.app (.app _ n) ν)) x ← whnfR e | throwError "not bernstein polynomial"
  let p ← mkAppOptM ``bernstein_nonneg #[n, ν, x]
  pure (.nonnegative p)


/-- Send `k : Fin (n+1)` to the equally spaced points `k/n` in the unit interval.
-/
def z {n : ℕ} (k : Fin (n + 1)) : I :=
  ⟨(k : ℝ) / n, by
    /-
      n : Nat
      k : Fin (HAdd.hAdd n 1)
      ⊢ Membership.mem unitInterval (HDiv.hDiv ↑↑k ↑n)
    -/
    cases' n with n
      /-
        case zero
        k : Fin (HAdd.hAdd 0 1)
        ⊢ Membership.mem unitInterval (HDiv.hDiv ↑↑k ↑0)
      -/
    · norm_num
      /-
        🎉 no goals
      -/
      /-
        case succ
        n : Nat
        k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ Membership.mem unitInterval (HDiv.hDiv ↑↑k ↑(HAdd.hAdd n 1))
      -/
    · have h₁ : 0 < (n.succ : ℝ) := mod_cast Nat.succ_pos _
      /-
        case succ
        n : Nat
        k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        h₁ : LT.lt 0 ↑n.succ
        ⊢ Membership.mem unitInterval (HDiv.hDiv ↑↑k ↑(HAdd.hAdd n 1))
      -/
      have h₂ : ↑k ≤ n.succ := mod_cast Fin.le_last k
      /-
        case succ
        n : Nat
        k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        h₁ : LT.lt 0 ↑n.succ
        h₂ : LE.le (↑k) n.succ
        ⊢ Membership.mem unitInterval (HDiv.hDiv ↑↑k ↑(HAdd.hAdd n 1))
      -/
      rw [Set.mem_Icc, le_div_iff₀ h₁, div_le_iff₀ h₁]
      /-
        case succ
        n : Nat
        k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        h₁ : LT.lt 0 ↑n.succ
        h₂ : LE.le (↑k) n.succ
        ⊢ And (LE.le (HMul.hMul 0 ↑n.succ) ↑↑k) (LE.le (↑↑k) (HMul.hMul 1 ↑n.succ))
      -/
      norm_cast
      /-
        case succ
        n : Nat
        k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        h₁ : LT.lt 0 ↑n.succ
        h₂ : LE.le (↑k) n.succ
        ⊢ And (LE.le (HMul.hMul 0 n.succ) ↑k) (LE.le (↑k) (HMul.hMul 1 n.succ))
      -/
      simp [h₂]⟩
      /-
        🎉 no goals
      -/


local postfix:90 "/ₙ" => z


theorem probability (n : ℕ) (x : I) : (∑ k : Fin (n + 1), bernstein n k x) = 1 := by
  /-
    n : Nat
    x : ↑unitInterval
    ⊢ Eq (Finset.univ.sum fun k => (bernstein n ↑k) x) 1
  -/
  have := bernsteinPolynomial.sum ℝ n
  /-
    n : Nat
    x : ↑unitInterval
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => bernsteinPolynomial Rea …
    ⊢ Eq (Finset.univ.sum fun k => (bernstein n ↑k) x) 1
  -/
  apply_fun fun p => Polynomial.aeval (x : ℝ) p at this
  simp? [map_sum, Finset.sum_range] at this says
    simp only [Finset.sum_range, map_sum, Polynomial.coe_aeval_eq_eval, Polynomial.eval_one] at this
  /-
    n : Nat
    x : ↑unitInterval
    this : Eq (Finset.univ.sum fun x_1 => Polynomial.eval (↑x) (bernsteinPolynomia …
    ⊢ Eq (Finset.univ.sum fun k => (bernstein n ↑k) x) 1
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem variance {n : ℕ} (h : 0 < (n : ℝ)) (x : I) :
    (∑ k : Fin (n + 1), (x - k/ₙ : ℝ) ^ 2 * bernstein n k x) = (x : ℝ) * (1 - x) / n := by
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    ⊢ Eq (Finset.univ.sum fun k => HMul.hMul (HPow.hPow (HSub.hSub ↑x ↑(bernstein. …
  -/
  have h' : (n : ℝ) ≠ 0 := ne_of_gt h
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    ⊢ Eq (Finset.univ.sum fun k => HMul.hMul (HPow.hPow (HSub.hSub ↑x ↑(bernstein. …
  -/
  apply_fun fun x : ℝ => x * n using GroupWithZero.mul_left_injective h'
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    ⊢ Eq ((fun x => HMul.hMul x ↑n) (Finset.univ.sum fun k => HMul.hMul (HPow.hPow …
  -/
  apply_fun fun x : ℝ => x * n using GroupWithZero.mul_left_injective h'
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    ⊢ Eq ((fun x => HMul.hMul x ↑n) ((fun x => HMul.hMul x ↑n) (Finset.univ.sum fu …
  -/
  dsimp
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    ⊢ Eq (HMul.hMul (HMul.hMul (Finset.univ.sum fun k => HMul.hMul (HPow.hPow (HSu …
  -/
  conv_lhs => simp only [Finset.sum_mul, z]
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSu …
  -/
  conv_rhs => rw [div_mul_cancel₀ _ h']
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSu …
  -/
  have := bernsteinPolynomial.variance ℝ n
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    this : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun ν => HMul.hMul (HPow.hPow (H …
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSu …
  -/
  apply_fun fun p => Polynomial.aeval (x : ℝ) p at this
  simp? [map_sum, Finset.sum_range, ← Polynomial.natCast_mul] at this says
    simp only [nsmul_eq_mul, Finset.sum_range, map_sum, Polynomial.coe_aeval_eq_eval,
      Polynomial.eval_mul, Polynomial.eval_pow, Polynomial.eval_sub, Polynomial.eval_natCast,
      Polynomial.eval_X, Polynomial.eval_one] at this
  /-
    n : Nat
    h : LT.lt 0 ↑n
    x : ↑unitInterval
    h' : Ne (↑n) 0
    this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSu …
  -/
  convert this using 1
    /-
      case h.e'_2
      n : Nat
      h : LT.lt 0 ↑n
      x : ↑unitInterval
      h' : Ne (↑n) 0
      this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
      ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSu …
    -/
  · congr 1; funext k
    /-
      case h.e'_2.e_f.h
      n : Nat
      h : LT.lt 0 ↑n
      x : ↑unitInterval
      h' : Ne (↑n) 0
      this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
      k : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSub.hSub (↑x) (HDiv.hDiv ↑↑ …
    -/
    rw [mul_comm _ (n : ℝ), mul_comm _ (n : ℝ), ← mul_assoc, ← mul_assoc]
    /-
      case h.e'_2.e_f.h
      n : Nat
      h : LT.lt 0 ↑n
      x : ↑unitInterval
      h' : Ne (↑n) 0
      this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
      k : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑n ↑n) (HPow.hPow (HSub.hSub (↑x) (HDiv. …
    -/
    congr 1
    /-
      case h.e'_2.e_f.h.e_a
      n : Nat
      h : LT.lt 0 ↑n
      x : ↑unitInterval
      h' : Ne (↑n) 0
      this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
      k : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HMul.hMul (HMul.hMul ↑n ↑n) (HPow.hPow (HSub.hSub (↑x) (HDiv.hDiv ↑↑k ↑n …
    -/
    field_simp [h]
    /-
      case h.e'_2.e_f.h.e_a
      n : Nat
      h : LT.lt 0 ↑n
      x : ↑unitInterval
      h' : Ne (↑n) 0
      this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
      k : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HMul.hMul (HMul.hMul ↑n ↑n) (HPow.hPow (HSub.hSub (HMul.hMul ↑x ↑n) ↑↑k) …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      n : Nat
      h : LT.lt 0 ↑n
      x : ↑unitInterval
      h' : Ne (↑n) 0
      this : Eq (Finset.univ.sum fun x_1 => HMul.hMul (HPow.hPow (HSub.hSub (HMul.hM …
      ⊢ Eq (HMul.hMul (HMul.hMul (↑x) (HSub.hSub 1 ↑x)) ↑n) (HMul.hMul (HMul.hMul ↑n …
    -/
  · ring
    /-
      🎉 no goals
    -/


local postfix:1024 "/ₙ" => z


/-- The `n`-th approximation of a continuous function on `[0,1]` by Bernstein polynomials,
given by `∑ k, f (k/n) * bernstein n k x`.
-/
def bernsteinApproximation (n : ℕ) (f : C(I, ℝ)) : C(I, ℝ) :=
  ∑ k : Fin (n + 1), f k/ₙ • bernstein n k


@[simp]
theorem apply (n : ℕ) (f : C(I, ℝ)) (x : I) :
    bernsteinApproximation n f x = ∑ k : Fin (n + 1), f k/ₙ * bernstein n k x := by
  /-
    n : Nat
    f : ContinuousMap (↑unitInterval) Real
    x : ↑unitInterval
    ⊢ Eq ((bernsteinApproximation n f) x) (Finset.univ.sum fun k => HMul.hMul (f ( …
  -/
  simp [bernsteinApproximation]
  /-
    🎉 no goals
  -/


/-- The modulus of (uniform) continuity for `f`, chosen so `|f x - f y| < ε/2` when `|x - y| < δ`.
-/
def δ (f : C(I, ℝ)) (ε : ℝ) (h : 0 < ε) : ℝ :=
  f.modulus (ε / 2) (half_pos h)


theorem δ_pos {f : C(I, ℝ)} {ε : ℝ} {h : 0 < ε} : 0 < δ f ε h :=
  f.modulus_pos


/-- The set of points `k` so `k/n` is within `δ` of `x`.
-/
def S (f : C(I, ℝ)) (ε : ℝ) (h : 0 < ε) (n : ℕ) (x : I) : Finset (Fin (n + 1)) :=
  {k : Fin (n + 1) | dist k/ₙ x < δ f ε h}.toFinset


/-- If `k ∈ S`, then `f(k/n)` is close to `f x`.
-/
theorem lt_of_mem_S {f : C(I, ℝ)} {ε : ℝ} {h : 0 < ε} {n : ℕ} {x : I} {k : Fin (n + 1)}
    (m : k ∈ S f ε h n x) : |f k/ₙ - f x| < ε / 2 := by
  /-
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    n : Nat
    x : ↑unitInterval
    k : Fin (HAdd.hAdd n 1)
    m : Membership.mem (bernsteinApproximation.S f ε h n x) k
    ⊢ LT.lt (abs (HSub.hSub (f (bernstein.z k)) (f x))) (HDiv.hDiv ε 2)
  -/
  apply f.dist_lt_of_dist_lt_modulus (ε / 2) (half_pos h)
  -- Porting note: `simp` fails to apply `Set.mem_toFinset` on its own
  /-
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    n : Nat
    x : ↑unitInterval
    k : Fin (HAdd.hAdd n 1)
    m : Membership.mem (bernsteinApproximation.S f ε h n x) k
    ⊢ LT.lt (Dist.dist (bernstein.z k) x) (f.modulus (HDiv.hDiv ε 2) ⋯)
  -/
  simpa [S, (Set.mem_toFinset)] using m
  /-
    🎉 no goals
  -/


/-- If `k ∉ S`, then as `δ ≤ |x - k/n|`, we have the inequality `1 ≤ δ^-2 * (x - k/n)^2`.
This particular formulation will be helpful later.
-/
theorem le_of_mem_S_compl {f : C(I, ℝ)} {ε : ℝ} {h : 0 < ε} {n : ℕ} {x : I} {k : Fin (n + 1)}
    (m : k ∈ (S f ε h n x)ᶜ) : (1 : ℝ) ≤ δ f ε h ^ (-2 : ℤ) * ((x : ℝ) - k/ₙ) ^ 2 := by
  -- Porting note: added parentheses to help `simp`
  /-
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    n : Nat
    x : ↑unitInterval
    k : Fin (HAdd.hAdd n 1)
    m : Membership.mem (HasCompl.compl (bernsteinApproximation.S f ε h n x)) k
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (bernsteinApproximation.δ f ε h) (-2)) (HPow.h …
  -/
  simp only [Finset.mem_compl, not_lt, Set.mem_toFinset, Set.mem_setOf_eq, S] at m
  rw [zpow_neg, ← div_eq_inv_mul, zpow_two, ← pow_two, one_le_div (pow_pos δ_pos 2), sq_le_sq,
    abs_of_pos δ_pos]
  /-
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    n : Nat
    x : ↑unitInterval
    k : Fin (HAdd.hAdd n 1)
    m : LE.le (bernsteinApproximation.δ f ε h) (Dist.dist (bernstein.z k) x)
    ⊢ LE.le (bernsteinApproximation.δ f ε h) (abs (HSub.hSub ↑x ↑(bernstein.z k)))
  -/
  rwa [dist_comm] at m
  /-
    🎉 no goals
  -/


/-- The Bernstein approximations
```
∑ k : Fin (n+1), f (k/n : ℝ) * n.choose k * x^k * (1-x)^(n-k)
```
for a continuous function `f : C([0,1], ℝ)` converge uniformly to `f` as `n` tends to infinity.

This is the proof given in [Richard Beals' *Analysis, an introduction*][beals-analysis], §7D,
and reproduced on wikipedia.
-/
theorem bernsteinApproximation_uniform (f : C(I, ℝ)) :
    Tendsto (fun n : ℕ => bernsteinApproximation n f) atTop (𝓝 f) := by
  /-
    f : ContinuousMap (↑unitInterval) Real
    ⊢ Filter.Tendsto (fun n => bernsteinApproximation n f) Filter.atTop (nhds f)
  -/
  simp only [Metric.nhds_basis_ball.tendsto_right_iff, Metric.mem_ball, dist_eq_norm]
  /-
    f : ContinuousMap (↑unitInterval) Real
    ⊢ ∀ (i : Real), LT.lt 0 i → Filter.Eventually (fun x => LT.lt (Norm.norm (HSub …
  -/
  intro ε h
  /-
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (HSub.hSub (bernsteinApproximat …
  -/
  let δ := δ f ε h
  /-
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    δ : Real := bernsteinApproximation.δ f ε h
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (HSub.hSub (bernsteinApproximat …
  -/
  have nhds_zero := tendsto_const_div_atTop_nhds_zero_nat (2 * ‖f‖ * δ ^ (-2 : ℤ))
  filter_upwards [nhds_zero.eventually (gt_mem_nhds (half_pos h)), eventually_gt_atTop 0] with n nh
    npos'
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    δ : Real := bernsteinApproximation.δ f ε h
    nhds_zero : Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.n …
    n : Nat
    nh : LT.lt (HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.norm f)) (HPow.hPow δ (-2) …
    npos' : LT.lt 0 n
    ⊢ LT.lt (Norm.norm (HSub.hSub (bernsteinApproximation n f) f)) ε
  -/
  have npos : 0 < (n : ℝ) := by positivity
  -- As `[0,1]` is compact, it suffices to check the inequality pointwise.
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    δ : Real := bernsteinApproximation.δ f ε h
    nhds_zero : Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.n …
    n : Nat
    nh : LT.lt (HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.norm f)) (HPow.hPow δ (-2) …
    npos' : LT.lt 0 n
    npos : LT.lt 0 ↑n
    ⊢ LT.lt (Norm.norm (HSub.hSub (bernsteinApproximation n f) f)) ε
  -/
  rw [ContinuousMap.norm_lt_iff _ h]
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    δ : Real := bernsteinApproximation.δ f ε h
    nhds_zero : Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.n …
    n : Nat
    nh : LT.lt (HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.norm f)) (HPow.hPow δ (-2) …
    npos' : LT.lt 0 n
    npos : LT.lt 0 ↑n
    ⊢ ∀ (x : ↑unitInterval), LT.lt (Norm.norm ((HSub.hSub (bernsteinApproximation  …
  -/
  intro x
  -- The idea is to split up the sum over `k` into two sets,
  -- `S`, where `x - k/n < δ`, and its complement.
  /-
    case h
    f : ContinuousMap (↑unitInterval) Real
    ε : Real
    h : LT.lt 0 ε
    δ : Real := bernsteinApproximation.δ f ε h
    nhds_zero : Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.n …
    n : Nat
    nh : LT.lt (HDiv.hDiv (HMul.hMul (HMul.hMul 2 (Norm.norm f)) (HPow.hPow δ (-2) …
    npos' : LT.lt 0 n
    npos : LT.lt 0 ↑n
    x : ↑unitInterval
    ⊢ LT.lt (Norm.norm ((HSub.hSub (bernsteinApproximation n f) f) x)) ε
  -/
  let S := S f ε h n x
  calc
    |(bernsteinApproximation n f - f) x| = |bernsteinApproximation n f x - f x| := rfl
    _ = |bernsteinApproximation n f x - f x * 1| := by rw [mul_one]
    _ = |bernsteinApproximation n f x - f x * ∑ k : Fin (n + 1), bernstein n k x| := by
      rw [bernstein.probability]
    _ = |∑ k : Fin (n + 1), (f k/ₙ - f x) * bernstein n k x| := by
      simp [bernsteinApproximation, Finset.mul_sum, sub_mul]
    _ ≤ ∑ k : Fin (n + 1), |(f k/ₙ - f x) * bernstein n k x| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ k : Fin (n + 1), |f k/ₙ - f x| * bernstein n k x := by
      simp_rw [abs_mul, abs_eq_self.mpr bernstein_nonneg]
    _ = (∑ k ∈ S, |f k/ₙ - f x| * bernstein n k x) + ∑ k ∈ Sᶜ, |f k/ₙ - f x| * bernstein n k x :=
      (S.sum_add_sum_compl _).symm
    -- We'll now deal with the terms in `S` and the terms in `Sᶜ` in separate calc blocks.
    _ < ε / 2 + ε / 2 :=
      (add_lt_add_of_le_of_lt ?_ ?_)
    _ = ε := add_halves ε
  · -- We now work on the terms in `S`: uniform continuity and `bernstein.probability`
    -- quickly give us a bound.
    calc
      ∑ k ∈ S, |f k/ₙ - f x| * bernstein n k x ≤ ∑ k ∈ S, ε / 2 * bernstein n k x := by
        gcongr with _ m
        exact le_of_lt (lt_of_mem_S m)
      _ = ε / 2 * ∑ k ∈ S, bernstein n k x := by rw [Finset.mul_sum]
      -- In this step we increase the sum over `S` back to a sum over all of `Fin (n+1)`,
      -- so that we can use `bernstein.probability`.
      _ ≤ ε / 2 * ∑ k : Fin (n + 1), bernstein n k x := by gcongr; exact S.subset_univ
      _ = ε / 2 := by rw [bernstein.probability, mul_one]
  · -- We now turn to working on `Sᶜ`: we control the difference term just using `‖f‖`,
    -- and then insert a `δ^(-2) * (x - k/n)^2` factor
    -- (which is at least one because we are not in `S`).
    calc
      ∑ k ∈ Sᶜ, |f k/ₙ - f x| * bernstein n k x ≤ ∑ k ∈ Sᶜ, 2 * ‖f‖ * bernstein n k x := by
        gcongr
        apply f.dist_le_two_norm
      _ = 2 * ‖f‖ * ∑ k ∈ Sᶜ, bernstein n k x := by rw [Finset.mul_sum]
      _ ≤ 2 * ‖f‖ * ∑ k ∈ Sᶜ, δ ^ (-2 : ℤ) * ((x : ℝ) - k/ₙ) ^ 2 * bernstein n k x := by
        gcongr with _ m
        conv_lhs => rw [← one_mul (bernstein _ _ _)]
        gcongr
        exact le_of_mem_S_compl m
      -- Again enlarging the sum from `Sᶜ` to all of `Fin (n+1)`
      _ ≤ 2 * ‖f‖ * ∑ k : Fin (n + 1), δ ^ (-2 : ℤ) * ((x : ℝ) - k/ₙ) ^ 2 * bernstein n k x := by
        gcongr; exact Sᶜ.subset_univ
      _ = 2 * ‖f‖ * δ ^ (-2 : ℤ) * ∑ k : Fin (n + 1), ((x : ℝ) - k/ₙ) ^ 2 * bernstein n k x := by
        conv_rhs =>
          rw [mul_assoc, Finset.mul_sum]
          simp only [← mul_assoc]
      -- `bernstein.variance` and `x ∈ [0,1]` gives the uniform bound
      _ = 2 * ‖f‖ * δ ^ (-2 : ℤ) * x * (1 - x) / n := by rw [variance npos]; ring
      _ ≤ 2 * ‖f‖ * δ ^ (-2 : ℤ) * 1 * 1 / n := by gcongr <;> unit_interval
      _ < ε / 2 := by simp only [mul_one]; exact nh

