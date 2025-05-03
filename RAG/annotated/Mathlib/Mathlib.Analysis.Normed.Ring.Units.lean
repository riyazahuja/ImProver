/-- In a normed ring with summable geometric series, a perturbation of a unit `x` by an
element `t` of distance less than `‖x⁻¹‖⁻¹` from `x` is a unit.
Here we construct its `Units` structure. -/
@[simps! val]
def add (x : Rˣ) (t : R) (h : ‖t‖ < ‖(↑x⁻¹ : R)‖⁻¹) : Rˣ :=
  Units.copy -- to make `add_val` true definitionally, for convenience
    (x * Units.oneSub (-((x⁻¹).1 * t)) (by
      /-
        R : Type u_1
        inst✝¹ : NormedRing R
        inst✝ : HasSummableGeomSeries R
        x : Units R
        t : R
        h : LT.lt (Norm.norm t) (Inv.inv (Norm.norm ↑(Inv.inv x)))
        ⊢ LT.lt (Norm.norm (Neg.neg (HMul.hMul (↑(Inv.inv x)) t))) 1
      -/
      nontriviality R using zero_lt_one
      /-
        R : Type u_1
        inst✝¹ : NormedRing R
        inst✝ : HasSummableGeomSeries R
        x : Units R
        t : R
        h : LT.lt (Norm.norm t) (Inv.inv (Norm.norm ↑(Inv.inv x)))
        a✝ : Nontrivial R
        ⊢ LT.lt (Norm.norm (Neg.neg (HMul.hMul (↑(Inv.inv x)) t))) 1
      -/
      have hpos : 0 < ‖(↑x⁻¹ : R)‖ := Units.norm_pos x⁻¹
      calc
        ‖-(↑x⁻¹ * t)‖ = ‖↑x⁻¹ * t‖ := by rw [norm_neg]
        _ ≤ ‖(↑x⁻¹ : R)‖ * ‖t‖ := norm_mul_le (x⁻¹).1 _
        _ < ‖(↑x⁻¹ : R)‖ * ‖(↑x⁻¹ : R)‖⁻¹ := by nlinarith only [h, hpos]
        _ = 1 := mul_inv_cancel₀ (ne_of_gt hpos)))
                /-
                  R : Type u_1
                  inst✝¹ : NormedRing R
                  inst✝ : HasSummableGeomSeries R
                  x : Units R
                  t : R
                  h : LT.lt (Norm.norm t) (Inv.inv (Norm.norm ↑(Inv.inv x)))
                  ⊢ Eq (HAdd.hAdd (↑x) t) ↑(HMul.hMul x (Units.oneSub (Neg.neg (HMul.hMul (↑(Inv …
                -/
    (x + t) (by simp [mul_add]) _ rfl
                /-
                  🎉 no goals
                -/


/-- In a normed ring with summable geometric series, an element `y` of distance less
than `‖x⁻¹‖⁻¹` from `x` is a unit. Here we construct its `Units` structure. -/
@[simps! val]
def ofNearby (x : Rˣ) (y : R) (h : ‖y - x‖ < ‖(↑x⁻¹ : R)‖⁻¹) : Rˣ :=
                                   /-
                                     R : Type u_1
                                     inst✝¹ : NormedRing R
                                     inst✝ : HasSummableGeomSeries R
                                     x : Units R
                                     y : R
                                     h : LT.lt (Norm.norm (HSub.hSub y ↑x)) (Inv.inv (Norm.norm ↑(Inv.inv x)))
                                     ⊢ Eq y ↑(x.add (HSub.hSub y ↑x) h)
                                   -/
  (x.add (y - x : R) h).copy y (by simp) _ rfl
                                   /-
                                     🎉 no goals
                                   -/


/-- The group of units of a normed ring with summable geometric series is an open subset
of the ring. -/
protected theorem isOpen : IsOpen { x : R | IsUnit x } := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    ⊢ IsOpen (setOf fun x => IsUnit x)
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    a✝ : Nontrivial R
    ⊢ IsOpen (setOf fun x => IsUnit x)
  -/
  rw [Metric.isOpen_iff]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    a✝ : Nontrivial R
    ⊢ ∀ (x : R), Membership.mem (setOf fun x => IsUnit x) x → Exists fun ε => And  …
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    a✝ : Nontrivial R
    x : Units R
    ⊢ Exists fun ε => And (GT.gt ε 0) (HasSubset.Subset (Metric.ball (↑x) ε) (setO …
  -/
  refine ⟨‖(↑x⁻¹ : R)‖⁻¹, _root_.inv_pos.mpr (Units.norm_pos x⁻¹), fun y hy ↦ ?_⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    a✝ : Nontrivial R
    x : Units R
    y : R
    hy : Membership.mem (Metric.ball (↑x) (Inv.inv (Norm.norm ↑(Inv.inv x)))) y
    ⊢ Membership.mem (setOf fun x => IsUnit x) y
  -/
  rw [mem_ball_iff_norm] at hy
  /-
    case intro
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    a✝ : Nontrivial R
    x : Units R
    y : R
    hy : LT.lt (Norm.norm (HSub.hSub y ↑x)) (Inv.inv (Norm.norm ↑(Inv.inv x)))
    ⊢ Membership.mem (setOf fun x => IsUnit x) y
  -/
  exact (x.ofNearby y hy).isUnit
  /-
    🎉 no goals
  -/


protected theorem nhds (x : Rˣ) : { x : R | IsUnit x } ∈ 𝓝 (x : R) :=
  IsOpen.mem_nhds Units.isOpen x.isUnit


/-- The `nonunits` in a normed ring with summable geometric series are contained in the
complement of the ball of radius `1` centered at `1 : R`. -/
theorem subset_compl_ball : nonunits R ⊆ (Metric.ball (1 : R) 1)ᶜ := fun x hx h₁ ↦ hx <|
                                               /-
                                                 R : Type u_1
                                                 inst✝¹ : NormedRing R
                                                 inst✝ : HasSummableGeomSeries R
                                                 x : R
                                                 hx : Membership.mem (nonunits R) x
                                                 h₁ : Membership.mem (Metric.ball 1 1) x
                                                 ⊢ LT.lt (Norm.norm (HSub.hSub 1 x)) 1
                                               -/
  sub_sub_self 1 x ▸ (Units.oneSub (1 - x) (by rwa [mem_ball_iff_norm'] at h₁)).isUnit
                                               /-
                                                 🎉 no goals
                                               -/

-- The `nonunits` in a normed ring with summable geometric series are a closed set

protected theorem isClosed : IsClosed (nonunits R) :=
  Units.isOpen.isClosed_compl


theorem inverse_one_sub (t : R) (h : ‖t‖ < 1) : inverse (1 - t) = ↑(Units.oneSub t h)⁻¹ := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    h : LT.lt (Norm.norm t) 1
    ⊢ Eq (Ring.inverse (HSub.hSub 1 t)) ↑(Inv.inv (Units.oneSub t h))
  -/
  rw [← inverse_unit (Units.oneSub t h), Units.val_oneSub]
  /-
    🎉 no goals
  -/


/-- The formula `Ring.inverse (x + t) = Ring.inverse (1 + x⁻¹ * t) * x⁻¹` holds for `t` sufficiently
small. -/
theorem inverse_add (x : Rˣ) :
    ∀ᶠ t in 𝓝 0, inverse ((x : R) + t) = inverse (1 + ↑x⁻¹ * t) * ↑x⁻¹ := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    ⊢ Filter.Eventually (fun t => Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HMul.hMul  …
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    a✝ : Nontrivial R
    ⊢ Filter.Eventually (fun t => Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HMul.hMul  …
  -/
  rw [Metric.eventually_nhds_iff]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    a✝ : Nontrivial R
    ⊢ Exists fun ε => And (GT.gt ε 0) (∀ ⦃y : R⦄, LT.lt (Dist.dist y 0) ε → Eq (Ri …
  -/
  refine ⟨‖(↑x⁻¹ : R)‖⁻¹, by cancel_denoms, fun t ht ↦ ?_⟩
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    a✝ : Nontrivial R
    t : R
    ht : LT.lt (Dist.dist t 0) (Inv.inv (Norm.norm ↑(Inv.inv x)))
    ⊢ Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HMul.hMul (Ring.inverse (HAdd.hAdd 1 ( …
  -/
  rw [dist_zero_right] at ht
  rw [← x.val_add t ht, inverse_unit, Units.add, Units.copy_eq, mul_inv_rev, Units.val_mul,
    ← inverse_unit, Units.val_oneSub, sub_neg_eq_add]


theorem inverse_one_sub_nth_order' (n : ℕ) {t : R} (ht : ‖t‖ < 1) :
    inverse ((1 : R) - t) = (∑ i ∈ range n, t ^ i) + t ^ n * inverse (1 - t) :=
  have := _root_.summable_geometric_of_norm_lt_one ht
  calc inverse (1 - t) = ∑' i : ℕ, t ^ i := inverse_one_sub t ht
    _ = ∑ i ∈ range n, t ^ i + ∑' i : ℕ, t ^ (i + n) := (sum_add_tsum_nat_add _ this).symm
    _ = (∑ i ∈ range n, t ^ i) + t ^ n * inverse (1 - t) := by
      /-
        R : Type u_1
        inst✝¹ : NormedRing R
        inst✝ : HasSummableGeomSeries R
        n : Nat
        t : R
        ht : LT.lt (Norm.norm t) 1
        this : Summable fun n => HPow.hPow t n
        ⊢ Eq (HAdd.hAdd ((Finset.range n).sum fun i => HPow.hPow t i) (tsum fun i => H …
      -/
      simp only [inverse_one_sub t ht, add_comm _ n, pow_add, this.tsum_mul_left]; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem inverse_one_sub_nth_order (n : ℕ) :
    ∀ᶠ t in 𝓝 0, inverse ((1 : R) - t) = (∑ i ∈ range n, t ^ i) + t ^ n * inverse (1 - t) :=
  Metric.eventually_nhds_iff.2 ⟨1, one_pos, fun t ht ↦ inverse_one_sub_nth_order' n <| by
    /-
      R : Type u_1
      inst✝¹ : NormedRing R
      inst✝ : HasSummableGeomSeries R
      n : Nat
      t : R
      ht : LT.lt (Dist.dist t 0) 1
      ⊢ LT.lt (Norm.norm t) 1
    -/
    rwa [← dist_zero_right]⟩
    /-
      🎉 no goals
    -/



/-- The formula
`Ring.inverse (x + t) =
  (∑ i ∈ Finset.range n, (- x⁻¹ * t) ^ i) * x⁻¹ + (- x⁻¹ * t) ^ n * Ring.inverse (x + t)`
holds for `t` sufficiently small. -/
theorem inverse_add_nth_order (x : Rˣ) (n : ℕ) :
    ∀ᶠ t in 𝓝 0, inverse ((x : R) + t) =
      (∑ i ∈ range n, (-↑x⁻¹ * t) ^ i) * ↑x⁻¹ + (-↑x⁻¹ * t) ^ n * inverse (x + t) := by
  have hzero : Tendsto (-(↑x⁻¹ : R) * ·) (𝓝 0) (𝓝 0) :=
    (mulLeft_continuous _).tendsto' _ _ <| mul_zero _
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    hzero : Filter.Tendsto (fun x_1 => HMul.hMul (Neg.neg ↑(Inv.inv x)) x_1) (nhds …
    ⊢ Filter.Eventually (fun t => Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HAdd.hAdd  …
  -/
  filter_upwards [inverse_add x, hzero.eventually (inverse_one_sub_nth_order n)] with t ht ht'
  /-
    case h
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    hzero : Filter.Tendsto (fun x_1 => HMul.hMul (Neg.neg ↑(Inv.inv x)) x_1) (nhds …
    t : R
    ht : Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HMul.hMul (Ring.inverse (HAdd.hAdd  …
    ht' : Eq (Ring.inverse (HSub.hSub 1 (HMul.hMul (Neg.neg ↑(Inv.inv x)) t))) (HA …
    ⊢ Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HAdd.hAdd (HMul.hMul ((Finset.range n) …
  -/
  rw [neg_mul, sub_neg_eq_add] at ht'
  /-
    case h
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    hzero : Filter.Tendsto (fun x_1 => HMul.hMul (Neg.neg ↑(Inv.inv x)) x_1) (nhds …
    t : R
    ht : Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HMul.hMul (Ring.inverse (HAdd.hAdd  …
    ht' : Eq (Ring.inverse (HAdd.hAdd 1 (HMul.hMul (↑(Inv.inv x)) t))) (HAdd.hAdd  …
    ⊢ Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HAdd.hAdd (HMul.hMul ((Finset.range n) …
  -/
  conv_lhs => rw [ht, ht', add_mul, ← neg_mul, mul_assoc]
  /-
    case h
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    hzero : Filter.Tendsto (fun x_1 => HMul.hMul (Neg.neg ↑(Inv.inv x)) x_1) (nhds …
    t : R
    ht : Eq (Ring.inverse (HAdd.hAdd (↑x) t)) (HMul.hMul (Ring.inverse (HAdd.hAdd  …
    ht' : Eq (Ring.inverse (HAdd.hAdd 1 (HMul.hMul (↑(Inv.inv x)) t))) (HAdd.hAdd  …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow (HMul.hMul …
  -/
  rw [ht]
  /-
    🎉 no goals
  -/


theorem inverse_one_sub_norm : (fun t : R => inverse (1 - t)) =O[𝓝 0] (fun _t => 1 : R → ℝ) := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    ⊢ Asymptotics.IsBigO (nhds 0) (fun t => Ring.inverse (HSub.hSub 1 t)) fun _t = …
  -/
  simp only [IsBigO, IsBigOWith, Metric.eventually_nhds_iff]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    ⊢ Exists fun c => Exists fun ε => And (GT.gt ε 0) (∀ ⦃y : R⦄, LT.lt (Dist.dist …
  -/
  refine ⟨‖(1 : R)‖ + 1, (2 : ℝ)⁻¹, by norm_num, fun t ht ↦ ?_⟩
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    ht : LT.lt (Dist.dist t 0) (Inv.inv 2)
    ⊢ LE.le (Norm.norm (Ring.inverse (HSub.hSub 1 t))) (HMul.hMul (HAdd.hAdd (Norm …
  -/
  rw [dist_zero_right] at ht
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    ht : LT.lt (Norm.norm t) (Inv.inv 2)
    ⊢ LE.le (Norm.norm (Ring.inverse (HSub.hSub 1 t))) (HMul.hMul (HAdd.hAdd (Norm …
  -/
  have ht' : ‖t‖ < 1 := by linarith
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    ht : LT.lt (Norm.norm t) (Inv.inv 2)
    ht' : LT.lt (Norm.norm t) 1
    ⊢ LE.le (Norm.norm (Ring.inverse (HSub.hSub 1 t))) (HMul.hMul (HAdd.hAdd (Norm …
  -/
  simp only [inverse_one_sub t ht', norm_one, mul_one, Set.mem_setOf_eq]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    ht : LT.lt (Norm.norm t) (Inv.inv 2)
    ht' : LT.lt (Norm.norm t) 1
    ⊢ LE.le (Norm.norm ↑(Inv.inv (Units.oneSub t ht'))) (HAdd.hAdd (Norm.norm 1) 1)
  -/
  change ‖∑' n : ℕ, t ^ n‖ ≤ _
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    ht : LT.lt (Norm.norm t) (Inv.inv 2)
    ht' : LT.lt (Norm.norm t) 1
    ⊢ LE.le (Norm.norm (tsum fun n => HPow.hPow t n)) (HAdd.hAdd (Norm.norm 1) 1)
  -/
  have := tsum_geometric_le_of_norm_lt_one t ht'
  have : (1 - ‖t‖)⁻¹ ≤ 2 := by
    rw [← inv_inv (2 : ℝ)]
    refine inv_anti₀ (by norm_num) ?_
    linarith
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    t : R
    ht : LT.lt (Norm.norm t) (Inv.inv 2)
    ht' : LT.lt (Norm.norm t) 1
    this✝ : LE.le (Norm.norm (tsum fun n => HPow.hPow t n)) (HAdd.hAdd (HSub.hSub  …
    this : LE.le (Inv.inv (HSub.hSub 1 (Norm.norm t))) 2
    ⊢ LE.le (Norm.norm (tsum fun n => HPow.hPow t n)) (HAdd.hAdd (Norm.norm 1) 1)
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- The function `fun t ↦ inverse (x + t)` is O(1) as `t → 0`. -/
theorem inverse_add_norm (x : Rˣ) : (fun t : R => inverse (↑x + t)) =O[𝓝 0] fun _t => (1 : ℝ) := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    ⊢ Asymptotics.IsBigO (nhds 0) (fun t => Ring.inverse (HAdd.hAdd (↑x) t)) fun _ …
  -/
  refine EventuallyEq.trans_isBigO (inverse_add x) (one_mul (1 : ℝ) ▸ ?_)
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x_1 => HMul.hMul (Ring.inverse (HAdd.hAdd 1 …
  -/
  simp only [← sub_neg_eq_add, ← neg_mul]
  have hzero : Tendsto (-(↑x⁻¹ : R) * ·) (𝓝 0) (𝓝 0) :=
    (mulLeft_continuous _).tendsto' _ _ <| mul_zero _
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    hzero : Filter.Tendsto (fun x_1 => HMul.hMul (Neg.neg ↑(Inv.inv x)) x_1) (nhds …
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x_1 => HMul.hMul (Ring.inverse (HSub.hSub 1 …
  -/
  exact (inverse_one_sub_norm.comp_tendsto hzero).mul (isBigO_const_const _ one_ne_zero _)
  /-
    🎉 no goals
  -/


/-- The function
`fun t ↦ Ring.inverse (x + t) - (∑ i ∈ Finset.range n, (- x⁻¹ * t) ^ i) * x⁻¹`
is `O(t ^ n)` as `t → 0`. -/
theorem inverse_add_norm_diff_nth_order (x : Rˣ) (n : ℕ) :
    (fun t : R => inverse (↑x + t) - (∑ i ∈ range n, (-↑x⁻¹ * t) ^ i) * ↑x⁻¹) =O[𝓝 (0 : R)]
      fun t => ‖t‖ ^ n := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    ⊢ Asymptotics.IsBigO (nhds 0) (fun t => HSub.hSub (Ring.inverse (HAdd.hAdd (↑x …
  -/
  refine EventuallyEq.trans_isBigO (.sub (inverse_add_nth_order x n) (.refl _ _)) ?_
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x_1 => HSub.hSub (HAdd.hAdd (HMul.hMul ((Fi …
  -/
  simp only [add_sub_cancel_left]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x_1 => HMul.hMul (HPow.hPow (HMul.hMul (Neg …
  -/
  refine ((isBigO_refl _ _).norm_right.mul (inverse_add_norm x)).trans ?_
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x_1 => HMul.hMul (Norm.norm (HPow.hPow (HMu …
  -/
  simp only [mul_one, isBigO_norm_left]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    n : Nat
    ⊢ Asymptotics.IsBigO (nhds 0) (fun x_1 => HPow.hPow (HMul.hMul (Neg.neg ↑(Inv. …
  -/
  exact ((isBigO_refl _ _).norm_right.const_mul_left _).pow _
  /-
    🎉 no goals
  -/


/-- The function `fun t ↦ Ring.inverse (x + t) - x⁻¹` is `O(t)` as `t → 0`. -/
theorem inverse_add_norm_diff_first_order (x : Rˣ) :
    (fun t : R => inverse (↑x + t) - ↑x⁻¹) =O[𝓝 0] fun t => ‖t‖ := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    ⊢ Asymptotics.IsBigO (nhds 0) (fun t => HSub.hSub (Ring.inverse (HAdd.hAdd (↑x …
  -/
  simpa using inverse_add_norm_diff_nth_order x 1
  /-
    🎉 no goals
  -/


/-- The function `fun t ↦ Ring.inverse (x + t) - x⁻¹ + x⁻¹ * t * x⁻¹` is `O(t ^ 2)` as `t → 0`. -/
theorem inverse_add_norm_diff_second_order (x : Rˣ) :
    (fun t : R => inverse (↑x + t) - ↑x⁻¹ + ↑x⁻¹ * t * ↑x⁻¹) =O[𝓝 0] fun t => ‖t‖ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    ⊢ Asymptotics.IsBigO (nhds 0) (fun t => HAdd.hAdd (HSub.hSub (Ring.inverse (HA …
  -/
  convert inverse_add_norm_diff_nth_order x 2 using 2
  simp only [sum_range_succ, sum_range_zero, zero_add, pow_zero, pow_one, add_mul, one_mul,
    ← sub_sub, neg_mul, sub_neg_eq_add]


/-- The function `Ring.inverse` is continuous at each unit of `R`. -/
theorem inverse_continuousAt (x : Rˣ) : ContinuousAt inverse (x : R) := by
  have h_is_o : (fun t : R => inverse (↑x + t) - ↑x⁻¹) =o[𝓝 0] (fun _ => 1 : R → ℝ) :=
    (inverse_add_norm_diff_first_order x).trans_isLittleO (isLittleO_id_const one_ne_zero).norm_left
  have h_lim : Tendsto (fun y : R => y - x) (𝓝 x) (𝓝 0) := by
    refine tendsto_zero_iff_norm_tendsto_zero.mpr ?_
    exact tendsto_iff_norm_sub_tendsto_zero.mp tendsto_id
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    h_is_o : Asymptotics.IsLittleO (nhds 0) (fun t => HSub.hSub (Ring.inverse (HAd …
    h_lim : Filter.Tendsto (fun y => HSub.hSub y ↑x) (nhds ↑x) (nhds 0)
    ⊢ ContinuousAt Ring.inverse ↑x
  -/
  rw [ContinuousAt, tendsto_iff_norm_sub_tendsto_zero, inverse_unit]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : Units R
    h_is_o : Asymptotics.IsLittleO (nhds 0) (fun t => HSub.hSub (Ring.inverse (HAd …
    h_lim : Filter.Tendsto (fun y => HSub.hSub y ↑x) (nhds ↑x) (nhds 0)
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (Ring.inverse e) ↑(Inv.inv x)) …
  -/
  simpa [Function.comp_def] using h_is_o.norm_left.tendsto_div_nhds_zero.comp h_lim
  /-
    🎉 no goals
  -/


/-- In a normed ring with summable geometric series, the coercion from `Rˣ` (equipped with the
induced topology from the embedding in `R × R`) to `R` is an open embedding. -/
theorem isOpenEmbedding_val : IsOpenEmbedding (val : Rˣ → R) where
  toIsEmbedding := isEmbedding_val_mk'
    (fun _ ⟨u, hu⟩ ↦ hu ▸ (inverse_continuousAt u).continuousWithinAt) Ring.inverse_unit
  isOpen_range := Units.isOpen


@[deprecated (since := "2024-10-18")]
alias openEmbedding_val := isOpenEmbedding_val


/-- In a normed ring with summable geometric series, the coercion from `Rˣ` (equipped with the
induced topology from the embedding in `R × R`) to `R` is an open map. -/
theorem isOpenMap_val : IsOpenMap (val : Rˣ → R) :=
  isOpenEmbedding_val.isOpenMap


/-- An ideal which contains an element within `1` of `1 : R` is the unit ideal. -/
theorem eq_top_of_norm_lt_one (I : Ideal R) {x : R} (hxI : x ∈ I) (hx : ‖1 - x‖ < 1) : I = ⊤ :=
  let u := Units.oneSub (1 - x) hx
  I.eq_top_iff_one.mpr <| by
    /-
      R : Type u_1
      inst✝¹ : NormedRing R
      inst✝ : HasSummableGeomSeries R
      I : Ideal R
      x : R
      hxI : Membership.mem I x
      hx : LT.lt (Norm.norm (HSub.hSub 1 x)) 1
      u : Units R := Units.oneSub (HSub.hSub 1 x) hx
      ⊢ Membership.mem I 1
    -/
    simpa only [show u.inv * x = 1 by simp [u]] using I.mul_mem_left u.inv hxI
    /-
      🎉 no goals
    -/


/-- The `Ideal.closure` of a proper ideal in a normed ring with summable
geometric series is proper. -/
theorem closure_ne_top (I : Ideal R) (hI : I ≠ ⊤) : I.closure ≠ ⊤ := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    I : Ideal R
    hI : Ne I Top.top
    ⊢ Ne I.closure Top.top
  -/
  have h := closure_minimal (coe_subset_nonunits hI) nonunits.isClosed
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    I : Ideal R
    hI : Ne I Top.top
    h : HasSubset.Subset (closure ↑I) (nonunits R)
    ⊢ Ne I.closure Top.top
  -/
  simpa only [I.closure.eq_top_iff_one, Ne] using mt (@h 1) one_not_mem_nonunits
  /-
    🎉 no goals
  -/


/-- The `Ideal.closure` of a maximal ideal in a normed ring with summable
geometric series is the ideal itself. -/
theorem IsMaximal.closure_eq {I : Ideal R} (hI : I.IsMaximal) : I.closure = I :=
  (hI.eq_of_le (I.closure_ne_top hI.ne_top) subset_closure).symm


/-- Maximal ideals in normed rings with summable geometric series are closed. -/
instance IsMaximal.isClosed {I : Ideal R} [hI : I.IsMaximal] : IsClosed (I : Set R) :=
  isClosed_of_closure_subset <| Eq.subset <| congr_arg ((↑) : Ideal R → Set R) hI.closure_eq


