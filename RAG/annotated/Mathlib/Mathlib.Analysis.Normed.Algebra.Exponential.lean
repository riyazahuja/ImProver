/-- `expSeries 𝕂 𝔸` is the `FormalMultilinearSeries` whose `n`-th term is the map
`(xᵢ) : 𝔸ⁿ ↦ (1/n! : 𝕂) • ∏ xᵢ`. Its sum is the exponential map `NormedSpace.exp 𝕂 : 𝔸 → 𝔸`. -/
def expSeries : FormalMultilinearSeries 𝕂 𝔸 𝔸 := fun n =>
  (n !⁻¹ : 𝕂) • ContinuousMultilinearMap.mkPiAlgebraFin 𝕂 n 𝔸


/-- The exponential series as an `ofScalars` series. -/
theorem expSeries_eq_ofScalars : expSeries 𝕂 𝔸 = ofScalars 𝔸 fun n ↦ (n !⁻¹ : 𝕂) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    ⊢ Eq (NormedSpace.expSeries 𝕂 𝔸) (FormalMultilinearSeries.ofScalars 𝔸 fun n => …
  -/
  simp_rw [FormalMultilinearSeries.ext_iff, expSeries, ofScalars, implies_true]
  /-
    🎉 no goals
  -/


/-- `NormedSpace.exp 𝕂 : 𝔸 → 𝔸` is the exponential map determined by the action of `𝕂` on `𝔸`.
It is defined as the sum of the `FormalMultilinearSeries` `expSeries 𝕂 𝔸`.

Note that when `𝔸 = Matrix n n 𝕂`, this is the **Matrix Exponential**; see
[`MatrixExponential`](./Mathlib/Analysis/Normed/Algebra/MatrixExponential) for lemmas
specific to that case. -/
noncomputable def exp (x : 𝔸) : 𝔸 :=
  (expSeries 𝕂 𝔸).sum x


theorem expSeries_apply_eq (x : 𝔸) (n : ℕ) :
                                                             /-
                                                               𝕂 : Type u_1
                                                               𝔸 : Type u_2
                                                               inst✝⁴ : Field 𝕂
                                                               inst✝³ : Ring 𝔸
                                                               inst✝² : Algebra 𝕂 𝔸
                                                               inst✝¹ : TopologicalSpace 𝔸
                                                               inst✝ : TopologicalRing 𝔸
                                                               x : 𝔸
                                                               n : Nat
                                                               ⊢ Eq ((NormedSpace.expSeries 𝕂 𝔸 n) fun x_1 => x) (HSMul.hSMul (Inv.inv ↑n.fac …
                                                             -/
    (expSeries 𝕂 𝔸 n fun _ => x) = (n !⁻¹ : 𝕂) • x ^ n := by simp [expSeries]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem expSeries_apply_eq' (x : 𝔸) :
    (fun n => expSeries 𝕂 𝔸 n fun _ => x) = fun n => (n !⁻¹ : 𝕂) • x ^ n :=
  funext (expSeries_apply_eq x)


theorem expSeries_sum_eq (x : 𝔸) : (expSeries 𝕂 𝔸).sum x = ∑' n : ℕ, (n !⁻¹ : 𝕂) • x ^ n :=
  tsum_congr fun n => expSeries_apply_eq x n


theorem exp_eq_tsum : exp 𝕂 = fun x : 𝔸 => ∑' n : ℕ, (n !⁻¹ : 𝕂) • x ^ n :=
  funext expSeries_sum_eq


/-- The exponential sum as an `ofScalarsSum`. -/
theorem exp_eq_ofScalarsSum : exp 𝕂 = ofScalarsSum (E := 𝔸) fun n ↦ (n !⁻¹ : 𝕂) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂) (FormalMultilinearSeries.ofScalarsSum fun n => Inv.in …
  -/
  rw [exp_eq_tsum, ofScalarsSum_eq_tsum]
  /-
    🎉 no goals
  -/


theorem expSeries_apply_zero (n : ℕ) :
    (expSeries 𝕂 𝔸 n fun _ => (0 : 𝔸)) = Pi.single (f := fun _ => 𝔸) 0 1 n := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    n : Nat
    ⊢ Eq ((NormedSpace.expSeries 𝕂 𝔸 n) fun x => 0) (Pi.single 0 1 n)
  -/
  rw [expSeries_apply_eq]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    n : Nat
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow 0 n)) (Pi.single 0 1 n)
  -/
  cases' n with n
    /-
      case zero
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝⁴ : Field 𝕂
      inst✝³ : Ring 𝔸
      inst✝² : Algebra 𝕂 𝔸
      inst✝¹ : TopologicalSpace 𝔸
      inst✝ : TopologicalRing 𝔸
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑(Nat.factorial 0)) (HPow.hPow 0 0)) (Pi.single 0 1 …
    -/
  · rw [pow_zero, Nat.factorial_zero, Nat.cast_one, inv_one, one_smul, Pi.single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝⁴ : Field 𝕂
      inst✝³ : Ring 𝔸
      inst✝² : Algebra 𝕂 𝔸
      inst✝¹ : TopologicalSpace 𝔸
      inst✝ : TopologicalRing 𝔸
      n : Nat
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HAdd.hAdd n 1).factorial) (HPow.hPow 0 (HAdd.hAdd …
    -/
  · rw [zero_pow (Nat.succ_ne_zero _), smul_zero, Pi.single_eq_of_ne n.succ_ne_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem exp_zero : exp 𝕂 (0 : 𝔸) = 1 := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 0) 1
  -/
  simp_rw [exp_eq_tsum, ← expSeries_apply_eq, expSeries_apply_zero, tsum_pi_single]
  /-
    🎉 no goals
  -/


@[simp]
theorem exp_op [T2Space 𝔸] (x : 𝔸) : exp 𝕂 (MulOpposite.op x) = MulOpposite.op (exp 𝕂 x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : TopologicalSpace 𝔸
    inst✝¹ : TopologicalRing 𝔸
    inst✝ : T2Space 𝔸
    x : 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (MulOpposite.op x)) (MulOpposite.op (NormedSpace.exp 𝕂 …
  -/
  simp_rw [exp, expSeries_sum_eq, ← MulOpposite.op_pow, ← MulOpposite.op_smul, tsum_op]
  /-
    🎉 no goals
  -/


@[simp]
theorem exp_unop [T2Space 𝔸] (x : 𝔸ᵐᵒᵖ) :
    exp 𝕂 (MulOpposite.unop x) = MulOpposite.unop (exp 𝕂 x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : TopologicalSpace 𝔸
    inst✝¹ : TopologicalRing 𝔸
    inst✝ : T2Space 𝔸
    x : MulOpposite 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (MulOpposite.unop x)) (MulOpposite.unop (NormedSpace.e …
  -/
  simp_rw [exp, expSeries_sum_eq, ← MulOpposite.unop_pow, ← MulOpposite.unop_smul, tsum_unop]
  /-
    🎉 no goals
  -/


theorem star_exp [T2Space 𝔸] [StarRing 𝔸] [ContinuousStar 𝔸] (x : 𝔸) :
    star (exp 𝕂 x) = exp 𝕂 (star x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁷ : Field 𝕂
    inst✝⁶ : Ring 𝔸
    inst✝⁵ : Algebra 𝕂 𝔸
    inst✝⁴ : TopologicalSpace 𝔸
    inst✝³ : TopologicalRing 𝔸
    inst✝² : T2Space 𝔸
    inst✝¹ : StarRing 𝔸
    inst✝ : ContinuousStar 𝔸
    x : 𝔸
    ⊢ Eq (Star.star (NormedSpace.exp 𝕂 x)) (NormedSpace.exp 𝕂 (Star.star x))
  -/
  simp_rw [exp_eq_tsum, ← star_pow, ← star_inv_natCast_smul, ← tsum_star]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem _root_.IsSelfAdjoint.exp [T2Space 𝔸] [StarRing 𝔸] [ContinuousStar 𝔸] {x : 𝔸}
    (h : IsSelfAdjoint x) : IsSelfAdjoint (exp 𝕂 x) :=
  (star_exp x).trans <| h.symm ▸ rfl


theorem _root_.Commute.exp_right [T2Space 𝔸] {x y : 𝔸} (h : Commute x y) :
    Commute x (exp 𝕂 y) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : TopologicalSpace 𝔸
    inst✝¹ : TopologicalRing 𝔸
    inst✝ : T2Space 𝔸
    x y : 𝔸
    h : Commute x y
    ⊢ Commute x (NormedSpace.exp 𝕂 y)
  -/
  rw [exp_eq_tsum]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : TopologicalSpace 𝔸
    inst✝¹ : TopologicalRing 𝔸
    inst✝ : T2Space 𝔸
    x y : 𝔸
    h : Commute x y
    ⊢ Commute x ((fun x => tsum fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow. …
  -/
  exact Commute.tsum_right x fun n => (h.pow_right n).smul_right _
  /-
    🎉 no goals
  -/


theorem _root_.Commute.exp_left [T2Space 𝔸] {x y : 𝔸} (h : Commute x y) : Commute (exp 𝕂 x) y :=
  (h.symm.exp_right 𝕂).symm


theorem _root_.Commute.exp [T2Space 𝔸] {x y : 𝔸} (h : Commute x y) : Commute (exp 𝕂 x) (exp 𝕂 y) :=
  (h.exp_left _).exp_right _


theorem expSeries_apply_eq_div (x : 𝔸) (n : ℕ) : (expSeries 𝕂 𝔸 n fun _ => x) = x ^ n / n ! := by
  rw [div_eq_mul_inv, ← (Nat.cast_commute n ! (x ^ n)).inv_left₀.eq, ← smul_eq_mul,
    expSeries_apply_eq, inv_natCast_smul_eq 𝕂 𝔸]


theorem expSeries_apply_eq_div' (x : 𝔸) :
    (fun n => expSeries 𝕂 𝔸 n fun _ => x) = fun n => x ^ n / n ! :=
  funext (expSeries_apply_eq_div x)


theorem expSeries_sum_eq_div (x : 𝔸) : (expSeries 𝕂 𝔸).sum x = ∑' n : ℕ, x ^ n / n ! :=
  tsum_congr (expSeries_apply_eq_div x)


theorem exp_eq_tsum_div : exp 𝕂 = fun x : 𝔸 => ∑' n : ℕ, x ^ n / n ! :=
  funext expSeries_sum_eq_div


theorem norm_expSeries_summable_of_mem_ball (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    Summable fun n => ‖expSeries 𝕂 𝔸 n fun _ => x‖ :=
  (expSeries 𝕂 𝔸).summable_norm_apply hx


theorem norm_expSeries_summable_of_mem_ball' (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    Summable fun n => ‖(n !⁻¹ : 𝕂) • x ^ n‖ := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Summable fun n => Norm.norm (HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow x …
  -/
  change Summable (norm ∘ _)
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Summable (Function.comp Norm.norm fun n => HSMul.hSMul (Inv.inv ↑n.factorial …
  -/
  rw [← expSeries_apply_eq']
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Summable (Function.comp Norm.norm fun n => (NormedSpace.expSeries 𝕂 𝔸 n) fun …
  -/
  exact norm_expSeries_summable_of_mem_ball x hx
  /-
    🎉 no goals
  -/


theorem expSeries_summable_of_mem_ball (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    Summable fun n => expSeries 𝕂 𝔸 n fun _ => x :=
  (norm_expSeries_summable_of_mem_ball x hx).of_norm


theorem expSeries_summable_of_mem_ball' (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    Summable fun n => (n !⁻¹ : 𝕂) • x ^ n :=
  (norm_expSeries_summable_of_mem_ball' x hx).of_norm


theorem expSeries_hasSum_exp_of_mem_ball (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasSum (fun n => expSeries 𝕂 𝔸 n fun _ => x) (exp 𝕂 x) :=
  FormalMultilinearSeries.hasSum (expSeries 𝕂 𝔸) hx


theorem expSeries_hasSum_exp_of_mem_ball' (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasSum (fun n => (n !⁻¹ : 𝕂) • x ^ n) (exp 𝕂 x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow x n)) (Normed …
  -/
  rw [← expSeries_apply_eq']
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ HasSum (fun n => (NormedSpace.expSeries 𝕂 𝔸 n) fun x_1 => x) (NormedSpace.ex …
  -/
  exact expSeries_hasSum_exp_of_mem_ball x hx
  /-
    🎉 no goals
  -/


theorem hasFPowerSeriesOnBall_exp_of_radius_pos (h : 0 < (expSeries 𝕂 𝔸).radius) :
    HasFPowerSeriesOnBall (exp 𝕂) (expSeries 𝕂 𝔸) 0 (expSeries 𝕂 𝔸).radius :=
  (expSeries 𝕂 𝔸).hasFPowerSeriesOnBall h


theorem hasFPowerSeriesAt_exp_zero_of_radius_pos (h : 0 < (expSeries 𝕂 𝔸).radius) :
    HasFPowerSeriesAt (exp 𝕂) (expSeries 𝕂 𝔸) 0 :=
  (hasFPowerSeriesOnBall_exp_of_radius_pos h).hasFPowerSeriesAt


theorem continuousOn_exp : ContinuousOn (exp 𝕂 : 𝔸 → 𝔸) (EMetric.ball 0 (expSeries 𝕂 𝔸).radius) :=
  FormalMultilinearSeries.continuousOn


theorem analyticAt_exp_of_mem_ball (x : 𝔸) (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    AnalyticAt 𝕂 (exp 𝕂) x := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ AnalyticAt 𝕂 (NormedSpace.exp 𝕂) x
  -/
  by_cases h : (expSeries 𝕂 𝔸).radius = 0
    /-
      case pos
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝³ : NontriviallyNormedField 𝕂
      inst✝² : NormedRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      x : 𝔸
      hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
      h : Eq (NormedSpace.expSeries 𝕂 𝔸).radius 0
      ⊢ AnalyticAt 𝕂 (NormedSpace.exp 𝕂) x
    -/
  · rw [h] at hx; exact (ENNReal.not_lt_zero hx).elim
                  /-
                    🎉 no goals
                  -/
    /-
      case neg
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝³ : NontriviallyNormedField 𝕂
      inst✝² : NormedRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      x : 𝔸
      hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
      h : Not (Eq (NormedSpace.expSeries 𝕂 𝔸).radius 0)
      ⊢ AnalyticAt 𝕂 (NormedSpace.exp 𝕂) x
    -/
  · have h := pos_iff_ne_zero.mpr h
    /-
      case neg
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝³ : NontriviallyNormedField 𝕂
      inst✝² : NormedRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      x : 𝔸
      hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
      h✝ : Not (Eq (NormedSpace.expSeries 𝕂 𝔸).radius 0)
      h : LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
      ⊢ AnalyticAt 𝕂 (NormedSpace.exp 𝕂) x
    -/
    exact (hasFPowerSeriesOnBall_exp_of_radius_pos h).analyticAt_of_mem hx
    /-
      🎉 no goals
    -/


/-- In a Banach-algebra `𝔸` over a normed field `𝕂` of characteristic zero, if `x` and `y` are
in the disk of convergence and commute, then
`NormedSpace.exp 𝕂 (x + y) = (NormedSpace.exp 𝕂 x) * (NormedSpace.exp 𝕂 y)`. -/
theorem exp_add_of_commute_of_mem_ball [CharZero 𝕂] {x y : 𝔸} (hxy : Commute x y)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius)
    (hy : y ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) : exp 𝕂 (x + y) = exp 𝕂 x * exp 𝕂 y := by
  rw [exp_eq_tsum,
    tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm
      (norm_expSeries_summable_of_mem_ball' x hx) (norm_expSeries_summable_of_mem_ball' y hy)]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x y : 𝔸
    hxy : Commute x y
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hy : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) y
    ⊢ Eq ((fun x => tsum fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow x  …
  -/
  dsimp only
  conv_lhs =>
    congr
    ext
    rw [hxy.add_pow' _, Finset.smul_sum]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x y : 𝔸
    hxy : Commute x y
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hy : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) y
    ⊢ Eq (tsum fun x_1 => (Finset.HasAntidiagonal.antidiagonal x_1).sum fun x_2 => …
  -/
  refine tsum_congr fun n => Finset.sum_congr rfl fun kl hkl => ?_
  rw [← Nat.cast_smul_eq_nsmul 𝕂, smul_smul, smul_mul_smul_comm, ← Finset.mem_antidiagonal.mp hkl,
    Nat.cast_add_choose, Finset.mem_antidiagonal.mp hkl]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x y : 𝔸
    hxy : Commute x y
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hy : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) y
    n : Nat
    kl : Prod Nat Nat
    hkl : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv ↑n.factorial) (HDiv.hDiv (↑n.factorial)  …
  -/
  congr 1
  /-
    case e_a
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x y : 𝔸
    hxy : Commute x y
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hy : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) y
    n : Nat
    kl : Prod Nat Nat
    hkl : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ Eq (HMul.hMul (Inv.inv ↑n.factorial) (HDiv.hDiv (↑n.factorial) (HMul.hMul ↑k …
  -/
  have : (n ! : 𝕂) ≠ 0 := Nat.cast_ne_zero.mpr n.factorial_ne_zero
  /-
    case e_a
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕂
    inst✝³ : NormedRing 𝔸
    inst✝² : NormedAlgebra 𝕂 𝔸
    inst✝¹ : CompleteSpace 𝔸
    inst✝ : CharZero 𝕂
    x y : 𝔸
    hxy : Commute x y
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    hy : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) y
    n : Nat
    kl : Prod Nat Nat
    hkl : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    this : Ne (↑n.factorial) 0
    ⊢ Eq (HMul.hMul (Inv.inv ↑n.factorial) (HDiv.hDiv (↑n.factorial) (HMul.hMul ↑k …
  -/
  field_simp [this]
  /-
    🎉 no goals
  -/


/-- `NormedSpace.exp 𝕂 x` has explicit two-sided inverse `NormedSpace.exp 𝕂 (-x)`. -/
noncomputable def invertibleExpOfMemBall [CharZero 𝕂] {x : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) : Invertible (exp 𝕂 x) where
  invOf := exp 𝕂 (-x)
  invOf_mul_self := by
    have hnx : -x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius := by
      rw [EMetric.mem_ball, ← neg_zero, edist_neg_neg]
      exact hx
    rw [← exp_add_of_commute_of_mem_ball (Commute.neg_left <| Commute.refl x) hnx hx,
      neg_add_cancel, exp_zero]
  mul_invOf_self := by
    have hnx : -x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius := by
      rw [EMetric.mem_ball, ← neg_zero, edist_neg_neg]
      exact hx
    rw [← exp_add_of_commute_of_mem_ball (Commute.neg_right <| Commute.refl x) hx hnx,
      add_neg_cancel, exp_zero]


theorem isUnit_exp_of_mem_ball [CharZero 𝕂] {x : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) : IsUnit (exp 𝕂 x) :=
  @isUnit_of_invertible _ _ _ (invertibleExpOfMemBall hx)


theorem invOf_exp_of_mem_ball [CharZero 𝕂] {x : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) [Invertible (exp 𝕂 x)] :
    ⅟ (exp 𝕂 x) = exp 𝕂 (-x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕂
    inst✝⁴ : NormedRing 𝔸
    inst✝³ : NormedAlgebra 𝕂 𝔸
    inst✝² : CompleteSpace 𝔸
    inst✝¹ : CharZero 𝕂
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    inst✝ : Invertible (NormedSpace.exp 𝕂 x)
    ⊢ Eq (Invertible.invOf (NormedSpace.exp 𝕂 x)) (NormedSpace.exp 𝕂 (Neg.neg x))
  -/
  letI := invertibleExpOfMemBall hx; convert (rfl : ⅟ (exp 𝕂 x) = _)
                                     /-
                                       🎉 no goals
                                     -/


/-- Any continuous ring homomorphism commutes with `NormedSpace.exp`. -/
theorem map_exp_of_mem_ball {F} [FunLike F 𝔸 𝔹] [RingHomClass F 𝔸 𝔹] (f : F) (hf : Continuous f)
    (x : 𝔸) (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    f (exp 𝕂 x) = exp 𝕂 (f x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    𝔹 : Type u_3
    inst✝⁷ : NontriviallyNormedField 𝕂
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedRing 𝔹
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : NormedAlgebra 𝕂 𝔹
    inst✝² : CompleteSpace 𝔸
    F : Type u_4
    inst✝¹ : FunLike F 𝔸 𝔹
    inst✝ : RingHomClass F 𝔸 𝔹
    f : F
    hf : Continuous ⇑f
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Eq (f (NormedSpace.exp 𝕂 x)) (NormedSpace.exp 𝕂 (f x))
  -/
  rw [exp_eq_tsum, exp_eq_tsum]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    𝔹 : Type u_3
    inst✝⁷ : NontriviallyNormedField 𝕂
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedRing 𝔹
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : NormedAlgebra 𝕂 𝔹
    inst✝² : CompleteSpace 𝔸
    F : Type u_4
    inst✝¹ : FunLike F 𝔸 𝔹
    inst✝ : RingHomClass F 𝔸 𝔹
    f : F
    hf : Continuous ⇑f
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Eq (f ((fun x => tsum fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow …
  -/
  refine ((expSeries_summable_of_mem_ball' _ hx).hasSum.map f hf).tsum_eq.symm.trans ?_
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    𝔹 : Type u_3
    inst✝⁷ : NontriviallyNormedField 𝕂
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedRing 𝔹
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : NormedAlgebra 𝕂 𝔹
    inst✝² : CompleteSpace 𝔸
    F : Type u_4
    inst✝¹ : FunLike F 𝔸 𝔹
    inst✝ : RingHomClass F 𝔸 𝔹
    f : F
    hf : Continuous ⇑f
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Eq (tsum fun b => Function.comp (⇑f) (fun n => HSMul.hSMul (Inv.inv ↑n.facto …
  -/
  dsimp only [Function.comp_def]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    𝔹 : Type u_3
    inst✝⁷ : NontriviallyNormedField 𝕂
    inst✝⁶ : NormedRing 𝔸
    inst✝⁵ : NormedRing 𝔹
    inst✝⁴ : NormedAlgebra 𝕂 𝔸
    inst✝³ : NormedAlgebra 𝕂 𝔹
    inst✝² : CompleteSpace 𝔸
    F : Type u_4
    inst✝¹ : FunLike F 𝔸 𝔹
    inst✝ : RingHomClass F 𝔸 𝔹
    f : F
    hf : Continuous ⇑f
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Eq (tsum fun b => f (HSMul.hSMul (Inv.inv ↑b.factorial) (HPow.hPow x b))) (t …
  -/
  simp_rw [map_inv_natCast_smul f 𝕂 𝕂, map_pow]
  /-
    🎉 no goals
  -/


theorem algebraMap_exp_comm_of_mem_ball [CompleteSpace 𝕂] (x : 𝕂)
    (hx : x ∈ EMetric.ball (0 : 𝕂) (expSeries 𝕂 𝕂).radius) :
    algebraMap 𝕂 𝔸 (exp 𝕂 x) = exp 𝕂 (algebraMap 𝕂 𝔸 x) :=
  map_exp_of_mem_ball _ (continuous_algebraMap 𝕂 𝔸) _ hx


theorem norm_expSeries_div_summable_of_mem_ball (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    Summable fun n => ‖x ^ n / (n ! : 𝔸)‖ := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Summable fun n => Norm.norm (HDiv.hDiv (HPow.hPow x n) ↑n.factorial)
  -/
  change Summable (norm ∘ _)
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Summable (Function.comp Norm.norm fun n => HDiv.hDiv (HPow.hPow x n) ↑n.fact …
  -/
  rw [← expSeries_apply_eq_div' (𝕂 := 𝕂) x]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : NontriviallyNormedField 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ Summable (Function.comp Norm.norm fun n => (NormedSpace.expSeries 𝕂 𝔸 n) fun …
  -/
  exact norm_expSeries_summable_of_mem_ball x hx
  /-
    🎉 no goals
  -/


theorem expSeries_div_summable_of_mem_ball [CompleteSpace 𝔸] (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) : Summable fun n => x ^ n / n ! :=
  (norm_expSeries_div_summable_of_mem_ball 𝕂 x hx).of_norm


theorem expSeries_div_hasSum_exp_of_mem_ball [CompleteSpace 𝔸] (x : 𝔸)
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) :
    HasSum (fun n => x ^ n / n !) (exp 𝕂 x) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedDivisionRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow x n) ↑n.factorial) (NormedSpace.exp 𝕂 x)
  -/
  rw [← expSeries_apply_eq_div' (𝕂 := 𝕂) x]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕂
    inst✝² : NormedDivisionRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    x : 𝔸
    hx : Membership.mem (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸).radius) x
    ⊢ HasSum (fun n => (NormedSpace.expSeries 𝕂 𝔸 n) fun x_1 => x) (NormedSpace.ex …
  -/
  exact expSeries_hasSum_exp_of_mem_ball x hx
  /-
    🎉 no goals
  -/


theorem exp_neg_of_mem_ball [CharZero 𝕂] [CompleteSpace 𝔸] {x : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) : exp 𝕂 (-x) = (exp 𝕂 x)⁻¹ :=
  letI := invertibleExpOfMemBall hx
  invOf_eq_inv (exp 𝕂 x)


/-- In a commutative Banach-algebra `𝔸` over a normed field `𝕂` of characteristic zero,
`NormedSpace.exp 𝕂 (x+y) = (NormedSpace.exp 𝕂 x) * (NormedSpace.exp 𝕂 y)`
for all `x`, `y` in the disk of convergence. -/
theorem exp_add_of_mem_ball [CharZero 𝕂] {x y : 𝔸}
    (hx : x ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius)
    (hy : y ∈ EMetric.ball (0 : 𝔸) (expSeries 𝕂 𝔸).radius) : exp 𝕂 (x + y) = exp 𝕂 x * exp 𝕂 y :=
  exp_add_of_commute_of_mem_ball (Commute.all x y) hx hy


/-- In a normed algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ`, the series defining the exponential map
has an infinite radius of convergence. -/
theorem expSeries_radius_eq_top : (expSeries 𝕂 𝔸).radius = ∞ := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    ⊢ Eq (NormedSpace.expSeries 𝕂 𝔸).radius Top.top
  -/
  have {n : ℕ} : (Nat.factorial n : 𝕂) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero n)
  apply expSeries_eq_ofScalars 𝕂 𝔸 ▸
    ofScalars_radius_eq_top_of_tendsto 𝔸 _ (Eventually.of_forall fun n => ?_)
  · simp_rw [← norm_div, Nat.factorial_succ, Nat.cast_mul, mul_inv_rev, mul_div_right_comm,
      inv_div_inv, norm_mul, div_self this, norm_one, one_mul]
    /-
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      this : ∀ {n : Nat}, Ne (↑n.factorial) 0
      ⊢ Filter.Tendsto (fun n => Norm.norm (Inv.inv ↑(HAdd.hAdd n 1))) Filter.atTop  …
    -/
    apply norm_zero (E := 𝕂) ▸ Filter.Tendsto.norm
    /-
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      this : ∀ {n : Nat}, Ne (↑n.factorial) 0
      ⊢ Filter.Tendsto (fun x => Inv.inv ↑(HAdd.hAdd x 1)) Filter.atTop (nhds 0)
    -/
    apply (Filter.tendsto_add_atTop_iff_nat (f := fun n => (n : 𝕂)⁻¹) 1).mpr
    /-
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      this : ∀ {n : Nat}, Ne (↑n.factorial) 0
      ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (nhds 0)
    -/
    exact RCLike.tendsto_inverse_atTop_nhds_zero_nat 𝕂
    /-
      🎉 no goals
    -/
    /-
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      this : ∀ {n : Nat}, Ne (↑n.factorial) 0
      n : Nat
      ⊢ Ne (Inv.inv ↑n.factorial) 0
    -/
  · simp [this]
    /-
      🎉 no goals
    -/


theorem expSeries_radius_pos : 0 < (expSeries 𝕂 𝔸).radius := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    ⊢ LT.lt 0 (NormedSpace.expSeries 𝕂 𝔸).radius
  -/
  rw [expSeries_radius_eq_top]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    ⊢ LT.lt 0 Top.top
  -/
  exact WithTop.top_pos
  /-
    🎉 no goals
  -/


theorem norm_expSeries_summable (x : 𝔸) : Summable fun n => ‖expSeries 𝕂 𝔸 n fun _ => x‖ :=
  norm_expSeries_summable_of_mem_ball x ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


theorem norm_expSeries_summable' (x : 𝔸) : Summable fun n => ‖(n !⁻¹ : 𝕂) • x ^ n‖ :=
  norm_expSeries_summable_of_mem_ball' x ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


theorem expSeries_summable (x : 𝔸) : Summable fun n => expSeries 𝕂 𝔸 n fun _ => x :=
  (norm_expSeries_summable x).of_norm


theorem expSeries_summable' (x : 𝔸) : Summable fun n => (n !⁻¹ : 𝕂) • x ^ n :=
  (norm_expSeries_summable' x).of_norm


theorem expSeries_hasSum_exp (x : 𝔸) : HasSum (fun n => expSeries 𝕂 𝔸 n fun _ => x) (exp 𝕂 x) :=
  expSeries_hasSum_exp_of_mem_ball x ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


theorem exp_series_hasSum_exp' (x : 𝔸) : HasSum (fun n => (n !⁻¹ : 𝕂) • x ^ n) (exp 𝕂 x) :=
  expSeries_hasSum_exp_of_mem_ball' x ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


theorem exp_hasFPowerSeriesOnBall : HasFPowerSeriesOnBall (exp 𝕂) (expSeries 𝕂 𝔸) 0 ∞ :=
  expSeries_radius_eq_top 𝕂 𝔸 ▸ hasFPowerSeriesOnBall_exp_of_radius_pos (expSeries_radius_pos _ _)


theorem exp_hasFPowerSeriesAt_zero : HasFPowerSeriesAt (exp 𝕂) (expSeries 𝕂 𝔸) 0 :=
  exp_hasFPowerSeriesOnBall.hasFPowerSeriesAt


@[continuity]
theorem exp_continuous : Continuous (exp 𝕂 : 𝔸 → 𝔸) := by
  rw [continuous_iff_continuousOn_univ, ← Metric.eball_top_eq_univ (0 : 𝔸), ←
    expSeries_radius_eq_top 𝕂 𝔸]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : RCLike 𝕂
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ⊢ ContinuousOn (NormedSpace.exp 𝕂) (EMetric.ball 0 (NormedSpace.expSeries 𝕂 𝔸) …
  -/
  exact continuousOn_exp
  /-
    🎉 no goals
  -/


open Topology in
lemma _root_.Filter.Tendsto.exp {α : Type*} {l : Filter α} {f : α → 𝔸} {a : 𝔸}
    (hf : Tendsto f l (𝓝 a)) :
    Tendsto (fun x => exp 𝕂 (f x)) l (𝓝 (exp 𝕂 a)) :=
  (exp_continuous.tendsto _).comp hf


theorem exp_analytic (x : 𝔸) : AnalyticAt 𝕂 (exp 𝕂) x :=
  analyticAt_exp_of_mem_ball x ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


/-- In a Banach-algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ`, if `x` and `y` commute, then
`NormedSpace.exp 𝕂 (x+y) = (NormedSpace.exp 𝕂 x) * (NormedSpace.exp 𝕂 y)`. -/
theorem exp_add_of_commute {x y : 𝔸} (hxy : Commute x y) : exp 𝕂 (x + y) = exp 𝕂 x * exp 𝕂 y :=
  exp_add_of_commute_of_mem_ball hxy ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)
    ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


/-- `NormedSpace.exp 𝕂 x` has explicit two-sided inverse `NormedSpace.exp 𝕂 (-x)`. -/
noncomputable def invertibleExp (x : 𝔸) : Invertible (exp 𝕂 x) :=
  invertibleExpOfMemBall <| (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem isUnit_exp (x : 𝔸) : IsUnit (exp 𝕂 x) :=
  isUnit_exp_of_mem_ball <| (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem invOf_exp (x : 𝔸) [Invertible (exp 𝕂 x)] : ⅟ (exp 𝕂 x) = exp 𝕂 (-x) :=
  invOf_exp_of_mem_ball <| (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem _root_.Ring.inverse_exp (x : 𝔸) : Ring.inverse (exp 𝕂 x) = exp 𝕂 (-x) :=
  letI := invertibleExp 𝕂 x
  Ring.inverse_invertible _


theorem exp_mem_unitary_of_mem_skewAdjoint [StarRing 𝔸] [ContinuousStar 𝔸] {x : 𝔸}
    (h : x ∈ skewAdjoint 𝔸) : exp 𝕂 x ∈ unitary 𝔸 := by
  rw [unitary.mem_iff, star_exp, skewAdjoint.mem_iff.mp h, ←
    exp_add_of_commute (Commute.refl x).neg_left, ← exp_add_of_commute (Commute.refl x).neg_right,
    neg_add_cancel, add_neg_cancel, exp_zero, and_self_iff]


/-- In a Banach-algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ`, if a family of elements `f i` mutually
commute then `NormedSpace.exp 𝕂 (∑ i, f i) = ∏ i, NormedSpace.exp 𝕂 (f i)`. -/
theorem exp_sum_of_commute {ι} (s : Finset ι) (f : ι → 𝔸)
    (h : (s : Set ι).Pairwise (Commute on f)) :
    exp 𝕂 (∑ i ∈ s, f i) =
      s.noncommProd (fun i => exp 𝕂 (f i)) fun _ hi _ hj _ => (h.of_refl hi hj).exp 𝕂 := by
  classical
    induction' s using Finset.induction_on with a s ha ih
    · simp
    rw [Finset.noncommProd_insert_of_not_mem _ _ _ _ ha, Finset.sum_insert ha, exp_add_of_commute,
      ih (h.mono <| Finset.subset_insert _ _)]
    refine Commute.sum_right _ _ _ fun i hi => ?_
    exact h.of_refl (Finset.mem_insert_self _ _) (Finset.mem_insert_of_mem hi)


theorem exp_nsmul (n : ℕ) (x : 𝔸) : exp 𝕂 (n • x) = exp 𝕂 x ^ n := by
  induction n with
  | zero => rw [zero_smul, pow_zero, exp_zero]
  | succ n ih => rw [succ_nsmul, pow_succ, exp_add_of_commute ((Commute.refl x).smul_left n), ih]


/-- Any continuous ring homomorphism commutes with `NormedSpace.exp`. -/
theorem map_exp {F} [FunLike F 𝔸 𝔹] [RingHomClass F 𝔸 𝔹] (f : F) (hf : Continuous f) (x : 𝔸) :
    f (exp 𝕂 x) = exp 𝕂 (f x) :=
  map_exp_of_mem_ball f hf x <| (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem exp_smul {G} [Monoid G] [MulSemiringAction G 𝔸] [ContinuousConstSMul G 𝔸] (g : G) (x : 𝔸) :
    exp 𝕂 (g • x) = g • exp 𝕂 x :=
  (map_exp 𝕂 (MulSemiringAction.toRingHom G 𝔸 g) (continuous_const_smul g) x).symm


theorem exp_units_conj (y : 𝔸ˣ) (x : 𝔸) : exp 𝕂 (y * x * ↑y⁻¹ : 𝔸) = y * exp 𝕂 x * ↑y⁻¹ :=
  exp_smul _ (ConjAct.toConjAct y) x


theorem exp_units_conj' (y : 𝔸ˣ) (x : 𝔸) : exp 𝕂 (↑y⁻¹ * x * y) = ↑y⁻¹ * exp 𝕂 x * y :=
  exp_units_conj _ _ _


@[simp]
theorem _root_.Prod.fst_exp [CompleteSpace 𝔹] (x : 𝔸 × 𝔹) : (exp 𝕂 x).fst = exp 𝕂 x.fst :=
  map_exp _ (RingHom.fst 𝔸 𝔹) continuous_fst x


@[simp]
theorem _root_.Prod.snd_exp [CompleteSpace 𝔹] (x : 𝔸 × 𝔹) : (exp 𝕂 x).snd = exp 𝕂 x.snd :=
  map_exp _ (RingHom.snd 𝔸 𝔹) continuous_snd x


@[simp]
theorem _root_.Pi.coe_exp {ι : Type*} {𝔸 : ι → Type*} [Finite ι] [∀ i, NormedRing (𝔸 i)]
    [∀ i, NormedAlgebra 𝕂 (𝔸 i)] [∀ i, CompleteSpace (𝔸 i)] (x : ∀ i, 𝔸 i) (i : ι) :
    exp 𝕂 x i = exp 𝕂 (x i) :=
  let ⟨_⟩ := nonempty_fintype ι
  map_exp _ (Pi.evalRingHom 𝔸 i) (continuous_apply _) x


theorem _root_.Pi.exp_def {ι : Type*} {𝔸 : ι → Type*} [Finite ι] [∀ i, NormedRing (𝔸 i)]
    [∀ i, NormedAlgebra 𝕂 (𝔸 i)] [∀ i, CompleteSpace (𝔸 i)] (x : ∀ i, 𝔸 i) :
    exp 𝕂 x = fun i => exp 𝕂 (x i) :=
  funext <| Pi.coe_exp 𝕂 x


theorem _root_.Function.update_exp {ι : Type*} {𝔸 : ι → Type*} [Finite ι] [DecidableEq ι]
    [∀ i, NormedRing (𝔸 i)] [∀ i, NormedAlgebra 𝕂 (𝔸 i)] [∀ i, CompleteSpace (𝔸 i)] (x : ∀ i, 𝔸 i)
    (j : ι) (xj : 𝔸 j) :
    Function.update (exp 𝕂 x) j (exp 𝕂 xj) = exp 𝕂 (Function.update x j xj) := by
  /-
    𝕂 : Type u_1
    inst✝⁵ : RCLike 𝕂
    ι : Type u_4
    𝔸 : ι → Type u_5
    inst✝⁴ : Finite ι
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → NormedRing (𝔸 i)
    inst✝¹ : (i : ι) → NormedAlgebra 𝕂 (𝔸 i)
    inst✝ : ∀ (i : ι), CompleteSpace (𝔸 i)
    x : (i : ι) → 𝔸 i
    j : ι
    xj : 𝔸 j
    ⊢ Eq (Function.update (NormedSpace.exp 𝕂 x) j (NormedSpace.exp 𝕂 xj)) (NormedS …
  -/
  ext i
  /-
    case h
    𝕂 : Type u_1
    inst✝⁵ : RCLike 𝕂
    ι : Type u_4
    𝔸 : ι → Type u_5
    inst✝⁴ : Finite ι
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → NormedRing (𝔸 i)
    inst✝¹ : (i : ι) → NormedAlgebra 𝕂 (𝔸 i)
    inst✝ : ∀ (i : ι), CompleteSpace (𝔸 i)
    x : (i : ι) → 𝔸 i
    j : ι
    xj : 𝔸 j
    i : ι
    ⊢ Eq (Function.update (NormedSpace.exp 𝕂 x) j (NormedSpace.exp 𝕂 xj) i) (Norme …
  -/
  simp_rw [Pi.exp_def]
  /-
    case h
    𝕂 : Type u_1
    inst✝⁵ : RCLike 𝕂
    ι : Type u_4
    𝔸 : ι → Type u_5
    inst✝⁴ : Finite ι
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → NormedRing (𝔸 i)
    inst✝¹ : (i : ι) → NormedAlgebra 𝕂 (𝔸 i)
    inst✝ : ∀ (i : ι), CompleteSpace (𝔸 i)
    x : (i : ι) → 𝔸 i
    j : ι
    xj : 𝔸 j
    i : ι
    ⊢ Eq (Function.update (fun i => NormedSpace.exp 𝕂 (x i)) j (NormedSpace.exp 𝕂  …
  -/
  exact (Function.apply_update (fun i => exp 𝕂) x j xj i).symm
  /-
    🎉 no goals
  -/


theorem algebraMap_exp_comm (x : 𝕂) : algebraMap 𝕂 𝔸 (exp 𝕂 x) = exp 𝕂 (algebraMap 𝕂 𝔸 x) :=
  algebraMap_exp_comm_of_mem_ball x <| (expSeries_radius_eq_top 𝕂 𝕂).symm ▸ edist_lt_top _ _


theorem norm_expSeries_div_summable (x : 𝔸) : Summable fun n => ‖(x ^ n / n ! : 𝔸)‖ :=
  norm_expSeries_div_summable_of_mem_ball 𝕂 x
    ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


theorem expSeries_div_summable (x : 𝔸) : Summable fun n => x ^ n / n ! :=
  (norm_expSeries_div_summable 𝕂 x).of_norm


theorem expSeries_div_hasSum_exp (x : 𝔸) : HasSum (fun n => x ^ n / n !) (exp 𝕂 x) :=
  expSeries_div_hasSum_exp_of_mem_ball 𝕂 x ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


theorem exp_neg (x : 𝔸) : exp 𝕂 (-x) = (exp 𝕂 x)⁻¹ :=
  exp_neg_of_mem_ball <| (expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _


theorem exp_zsmul (z : ℤ) (x : 𝔸) : exp 𝕂 (z • x) = exp 𝕂 x ^ z := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : RCLike 𝕂
    inst✝² : NormedDivisionRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    z : Int
    x : 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul z x)) (HPow.hPow (NormedSpace.exp 𝕂 x) z)
  -/
  obtain ⟨n, rfl | rfl⟩ := z.eq_nat_or_neg
    /-
      case intro.inl
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝³ : RCLike 𝕂
      inst✝² : NormedDivisionRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      x : 𝔸
      n : Nat
      ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul (↑n) x)) (HPow.hPow (NormedSpace.exp 𝕂 x) …
    -/
  · rw [zpow_natCast, natCast_zsmul, exp_nsmul]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝³ : RCLike 𝕂
      inst✝² : NormedDivisionRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      x : 𝔸
      n : Nat
      ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul (Neg.neg ↑n) x)) (HPow.hPow (NormedSpace. …
    -/
  · rw [zpow_neg, zpow_natCast, neg_smul, exp_neg, natCast_zsmul, exp_nsmul]
    /-
      🎉 no goals
    -/


theorem exp_conj (y : 𝔸) (x : 𝔸) (hy : y ≠ 0) : exp 𝕂 (y * x * y⁻¹) = y * exp 𝕂 x * y⁻¹ :=
  exp_units_conj _ (Units.mk0 y hy) x


theorem exp_conj' (y : 𝔸) (x : 𝔸) (hy : y ≠ 0) : exp 𝕂 (y⁻¹ * x * y) = y⁻¹ * exp 𝕂 x * y :=
  exp_units_conj' _ (Units.mk0 y hy) x


/-- In a commutative Banach-algebra `𝔸` over `𝕂 = ℝ` or `𝕂 = ℂ`,
`NormedSpace.exp 𝕂 (x+y) = (NormedSpace.exp 𝕂 x) * (NormedSpace.exp 𝕂 y)`. -/
theorem exp_add {x y : 𝔸} : exp 𝕂 (x + y) = exp 𝕂 x * exp 𝕂 y :=
  exp_add_of_mem_ball ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)
    ((expSeries_radius_eq_top 𝕂 𝔸).symm ▸ edist_lt_top _ _)


/-- A version of `NormedSpace.exp_sum_of_commute` for a commutative Banach-algebra. -/
theorem exp_sum {ι} (s : Finset ι) (f : ι → 𝔸) : exp 𝕂 (∑ i ∈ s, f i) = ∏ i ∈ s, exp 𝕂 (f i) := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : RCLike 𝕂
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ι : Type u_3
    s : Finset ι
    f : ι → 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (s.sum fun i => f i)) (s.prod fun i => NormedSpace.exp …
  -/
  rw [exp_sum_of_commute, Finset.noncommProd_eq_prod]
  /-
    case h
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝³ : RCLike 𝕂
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ι : Type u_3
    s : Finset ι
    f : ι → 𝔸
    ⊢ (↑s).Pairwise (Function.onFun Commute f)
  -/
  exact fun i _hi j _hj _ => Commute.all _ _
  /-
    🎉 no goals
  -/


/-- If a normed ring `𝔸` is a normed algebra over two fields, then they define the same
`expSeries` on `𝔸`. -/
theorem expSeries_eq_expSeries (n : ℕ) (x : 𝔸) :
    (expSeries 𝕂 𝔸 n fun _ => x) = expSeries 𝕂' 𝔸 n fun _ => x := by
  /-
    𝕂 : Type u_1
    𝕂' : Type u_2
    𝔸 : Type u_3
    inst✝⁶ : Field 𝕂
    inst✝⁵ : Field 𝕂'
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : Algebra 𝕂' 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    n : Nat
    x : 𝔸
    ⊢ Eq ((NormedSpace.expSeries 𝕂 𝔸 n) fun x_1 => x) ((NormedSpace.expSeries 𝕂' 𝔸 …
  -/
  rw [expSeries_apply_eq, expSeries_apply_eq, inv_natCast_smul_eq 𝕂 𝕂']
  /-
    🎉 no goals
  -/


/-- If a normed ring `𝔸` is a normed algebra over two fields, then they define the same
exponential function on `𝔸`. -/
theorem exp_eq_exp : (exp 𝕂 : 𝔸 → 𝔸) = exp 𝕂' := by
  /-
    𝕂 : Type u_1
    𝕂' : Type u_2
    𝔸 : Type u_3
    inst✝⁶ : Field 𝕂
    inst✝⁵ : Field 𝕂'
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : Algebra 𝕂' 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂) (NormedSpace.exp 𝕂')
  -/
  ext x
  /-
    case h
    𝕂 : Type u_1
    𝕂' : Type u_2
    𝔸 : Type u_3
    inst✝⁶ : Field 𝕂
    inst✝⁵ : Field 𝕂'
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : Algebra 𝕂' 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    x : 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 x) (NormedSpace.exp 𝕂' x)
  -/
  rw [exp, exp]
  /-
    case h
    𝕂 : Type u_1
    𝕂' : Type u_2
    𝔸 : Type u_3
    inst✝⁶ : Field 𝕂
    inst✝⁵ : Field 𝕂'
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : Algebra 𝕂' 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    x : 𝔸
    ⊢ Eq ((NormedSpace.expSeries 𝕂 𝔸).sum x) ((NormedSpace.expSeries 𝕂' 𝔸).sum x)
  -/
  refine tsum_congr fun n => ?_
  /-
    case h
    𝕂 : Type u_1
    𝕂' : Type u_2
    𝔸 : Type u_3
    inst✝⁶ : Field 𝕂
    inst✝⁵ : Field 𝕂'
    inst✝⁴ : Ring 𝔸
    inst✝³ : Algebra 𝕂 𝔸
    inst✝² : Algebra 𝕂' 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    x : 𝔸
    n : Nat
    ⊢ Eq ((NormedSpace.expSeries 𝕂 𝔸 n) fun x_1 => x) ((NormedSpace.expSeries 𝕂' 𝔸 …
  -/
  rw [expSeries_eq_expSeries 𝕂 𝕂' 𝔸 n x]
  /-
    🎉 no goals
  -/


theorem exp_ℝ_ℂ_eq_exp_ℂ_ℂ : (exp ℝ : ℂ → ℂ) = exp ℂ :=
  exp_eq_exp ℝ ℂ ℂ


/-- A version of `Complex.ofReal_exp` for `NormedSpace.exp` instead of `Complex.exp` -/
@[simp, norm_cast]
theorem of_real_exp_ℝ_ℝ (r : ℝ) : ↑(exp ℝ r) = exp ℂ (r : ℂ) :=
  (map_exp ℝ (algebraMap ℝ ℂ) (continuous_algebraMap _ _) r).trans (congr_fun exp_ℝ_ℂ_eq_exp_ℂ_ℂ _)


