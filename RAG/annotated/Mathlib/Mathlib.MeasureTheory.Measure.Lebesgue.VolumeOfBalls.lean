theorem MeasureTheory.measure_unitBall_eq_integral_div_gamma {E : Type*} {p : ℝ}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E]
    [BorelSpace E] (μ : Measure E) [IsAddHaarMeasure μ] (hp : 0 < p) :
    μ (Metric.ball 0 1) =
      .ofReal ((∫ (x : E), Real.exp (- ‖x‖ ^ p) ∂μ) / Real.Gamma (finrank ℝ E / p + 1)) := by
  /-
    E : Type u_1
    p : Real
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hp : LT.lt 0 p
    ⊢ Eq (μ (Metric.ball 0 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureTheory.integral  …
  -/
  obtain hE | hE := subsingleton_or_nontrivial E
  · rw [(Metric.nonempty_ball.mpr zero_lt_one).eq_zero, ← setIntegral_univ,
      Set.univ_nonempty.eq_zero, integral_singleton, finrank_zero_of_subsingleton, Nat.cast_zero,
      zero_div, zero_add, Real.Gamma_one, div_one, norm_zero, Real.zero_rpow hp.ne', neg_zero,
      Real.exp_zero, smul_eq_mul, mul_one, ofReal_toReal (measure_ne_top μ {0})]
    /-
      case inr
      E : Type u_1
      p : Real
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hp : LT.lt 0 p
      hE : Nontrivial E
      ⊢ Eq (μ (Metric.ball 0 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureTheory.integral  …
    -/
  · have : (0 : ℝ) < finrank ℝ E := Nat.cast_pos.mpr finrank_pos
    have : ((∫ y in Set.Ioi (0 : ℝ), y ^ (finrank ℝ E - 1) • Real.exp (-y ^ p)) /
        Real.Gamma ((finrank ℝ E) / p + 1)) * (finrank ℝ E) = 1 := by
      simp_rw [← Real.rpow_natCast _ (finrank ℝ E - 1), smul_eq_mul, Nat.cast_sub finrank_pos,
        Nat.cast_one]
      rw [integral_rpow_mul_exp_neg_rpow hp (by linarith), sub_add_cancel,
        Real.Gamma_add_one (ne_of_gt (by positivity))]
      field_simp; ring
    rw [integral_fun_norm_addHaar μ (fun x => Real.exp (- x ^ p)), nsmul_eq_mul, smul_eq_mul,
      mul_div_assoc, mul_div_assoc, mul_comm, mul_assoc, this, mul_one, ofReal_toReal]
    /-
      case inr
      E : Type u_1
      p : Real
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hp : LT.lt 0 p
      hE : Nontrivial E
      this✝ : LT.lt 0 ↑(Module.finrank Real E)
      this : Eq (HMul.hMul (HDiv.hDiv (MeasureTheory.integral (MeasureTheory.Measure …
      ⊢ Ne (μ (Metric.ball 0 1)) Top.top
    -/
    exact ne_of_lt measure_ball_lt_top
    /-
      🎉 no goals
    -/


theorem MeasureTheory.measure_lt_one_eq_integral_div_gamma {p : ℝ} (hp : 0 < p) :
    μ {x : E | g x < 1} =
      .ofReal ((∫ (x : E), Real.exp (- (g x) ^ p) ∂μ) / Real.Gamma (finrank ℝ E / p + 1)) := by
  -- We copy `E` to a new type `F` on which we will put the norm defined by `g`
  /-
    E : Type u_1
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    inst✝⁵ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : BorelSpace E
    inst✝² : T2Space E
    inst✝¹ : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    p : Real
    hp : LT.lt 0 p
    ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureThe …
  -/
  letI F : Type _ := E
  letI : NormedAddCommGroup F :=
  { norm := g
    dist := fun x y => g (x - y)
    dist_self := by simp only [_root_.sub_self, h1, forall_const]
    dist_comm := fun _ _ => by dsimp [dist]; rw [← h2, neg_sub]
    dist_triangle := fun x y z => by convert h3 (x - y) (y - z) using 1; simp [F]
    edist := fun x y => .ofReal (g (x - y))
    edist_dist := fun _ _ => rfl
    eq_of_dist_eq_zero := by convert fun _ _ h => eq_of_sub_eq_zero (h4 h) }
  letI : NormedSpace ℝ F :=
  { norm_smul_le := fun _ _ ↦ h5 _ _ }
  -- We put the new topology on F
  /-
    E : Type u_1
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    inst✝⁵ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : BorelSpace E
    inst✝² : T2Space E
    inst✝¹ : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    p : Real
    hp : LT.lt 0 p
    F : Type u_1 := E
    this✝ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this : NormedSpace Real F := NormedSpace.mk ⋯
    ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureThe …
  -/
  letI : TopologicalSpace F := UniformSpace.toTopologicalSpace
  /-
    E : Type u_1
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    inst✝⁵ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : BorelSpace E
    inst✝² : T2Space E
    inst✝¹ : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    p : Real
    hp : LT.lt 0 p
    F : Type u_1 := E
    this✝¹ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝ : NormedSpace Real F := NormedSpace.mk ⋯
    this : TopologicalSpace F := UniformSpace.toTopologicalSpace
    ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureThe …
  -/
  letI : MeasurableSpace F := borel F
  /-
    E : Type u_1
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    inst✝⁵ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : BorelSpace E
    inst✝² : T2Space E
    inst✝¹ : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    p : Real
    hp : LT.lt 0 p
    F : Type u_1 := E
    this✝² : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝¹ : NormedSpace Real F := NormedSpace.mk ⋯
    this✝ : TopologicalSpace F := UniformSpace.toTopologicalSpace
    this : MeasurableSpace F := borel F
    ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureThe …
  -/
  have : BorelSpace F := { measurable_eq := rfl }
  -- The map between `E` and `F` as a continuous linear equivalence
  let φ := @LinearEquiv.toContinuousLinearEquiv ℝ _ E _ _ tE _ _ F _ _ _ _ _ _ _ _ _
    (LinearEquiv.refl ℝ E : E ≃ₗ[ℝ] F)
  -- The measure `ν` is the measure on `F` defined by `μ`
  -- Since we have two different topologies, it is necessary to specify the topology of E
  /-
    E : Type u_1
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    inst✝⁵ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : BorelSpace E
    inst✝² : T2Space E
    inst✝¹ : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    p : Real
    hp : LT.lt 0 p
    F : Type u_1 := E
    this✝³ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝² : NormedSpace Real F := NormedSpace.mk ⋯
    this✝¹ : TopologicalSpace F := UniformSpace.toTopologicalSpace
    this✝ : MeasurableSpace F := borel F
    this : BorelSpace F
    φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
    ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureThe …
  -/
  let ν : Measure F := @Measure.map E F mE _ φ μ
  have : IsAddHaarMeasure ν :=
    @ContinuousLinearEquiv.isAddHaarMeasure_map E F ℝ ℝ _ _ _ _ _ _ tE _ _ _ _ _ _ _ mE _ _ _ φ μ _
  /-
    E : Type u_1
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    inst✝⁵ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : BorelSpace E
    inst✝² : T2Space E
    inst✝¹ : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    p : Real
    hp : LT.lt 0 p
    F : Type u_1 := E
    this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
    this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
    ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
    this : ν.IsAddHaarMeasure
    ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ENNReal.ofReal (HDiv.hDiv (MeasureThe …
  -/
  convert (measure_unitBall_eq_integral_div_gamma ν hp) using 1
    /-
      case h.e'_2
      E : Type u_1
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module Real E
      inst✝⁵ : FiniteDimensional Real E
      mE : MeasurableSpace E
      tE : TopologicalSpace E
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : BorelSpace E
      inst✝² : T2Space E
      inst✝¹ : ContinuousSMul Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      g : E → Real
      h1 : Eq (g 0) 0
      h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
      h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
      h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
      h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
      p : Real
      hp : LT.lt 0 p
      F : Type u_1 := E
      this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
      this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
      this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
      ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
      this : ν.IsAddHaarMeasure
      ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (ν (Metric.ball 0 1))
    -/
  · rw [@Measure.map_apply E F mE _ μ φ _ _ measurableSet_ball]
      /-
        case h.e'_2
        E : Type u_1
        inst✝⁷ : AddCommGroup E
        inst✝⁶ : Module Real E
        inst✝⁵ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁴ : TopologicalAddGroup E
        inst✝³ : BorelSpace E
        inst✝² : T2Space E
        inst✝¹ : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        p : Real
        hp : LT.lt 0 p
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Eq (μ (setOf fun x => LT.lt (g x) 1)) (μ (Set.preimage (⇑φ) (Metric.ball 0 1 …
      -/
    · congr!
      /-
        case h.e'_2.h.e'_6.h.e'_1.h.a
        E : Type u_1
        inst✝⁷ : AddCommGroup E
        inst✝⁶ : Module Real E
        inst✝⁵ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁴ : TopologicalAddGroup E
        inst✝³ : BorelSpace E
        inst✝² : T2Space E
        inst✝¹ : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        p : Real
        hp : LT.lt 0 p
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        x✝ : E
        ⊢ Iff (LT.lt (g x✝) 1) (Metric.ball 0 1 x✝)
      -/
      simp_rw [Metric.ball, dist_zero_right]
      /-
        case h.e'_2.h.e'_6.h.e'_1.h.a
        E : Type u_1
        inst✝⁷ : AddCommGroup E
        inst✝⁶ : Module Real E
        inst✝⁵ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁴ : TopologicalAddGroup E
        inst✝³ : BorelSpace E
        inst✝² : T2Space E
        inst✝¹ : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        p : Real
        hp : LT.lt 0 p
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        x✝ : E
        ⊢ Iff (LT.lt (g x✝) 1) (setOf (fun y => LT.lt (Norm.norm y) 1) x✝)
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        E : Type u_1
        inst✝⁷ : AddCommGroup E
        inst✝⁶ : Module Real E
        inst✝⁵ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁴ : TopologicalAddGroup E
        inst✝³ : BorelSpace E
        inst✝² : T2Space E
        inst✝¹ : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        p : Real
        hp : LT.lt 0 p
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Measurable ⇑φ
      -/
    · refine @Continuous.measurable E F tE mE _ _ _ _ φ ?_
      /-
        E : Type u_1
        inst✝⁷ : AddCommGroup E
        inst✝⁶ : Module Real E
        inst✝⁵ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁴ : TopologicalAddGroup E
        inst✝³ : BorelSpace E
        inst✝² : T2Space E
        inst✝¹ : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        p : Real
        hp : LT.lt 0 p
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Continuous ⇑φ
      -/
      exact @ContinuousLinearEquiv.continuous ℝ ℝ _ _ _ _ _ _ E tE _ F _ _ _ _ φ
      /-
        🎉 no goals
      -/
  · -- The map between `E` and `F` as a measurable equivalence
    let ψ := @Homeomorph.toMeasurableEquiv E F tE mE _ _ _ _
      (@ContinuousLinearEquiv.toHomeomorph ℝ ℝ _ _ _ _ _ _ E tE _ F _ _ _ _ φ)
    -- The map `ψ` is measure preserving by construction
    have : @MeasurePreserving E F mE _ ψ μ ν :=
      @Measurable.measurePreserving E F mE _ ψ (@MeasurableEquiv.measurable E F mE _ ψ) _
    /-
      case h.e'_3
      E : Type u_1
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module Real E
      inst✝⁵ : FiniteDimensional Real E
      mE : MeasurableSpace E
      tE : TopologicalSpace E
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : BorelSpace E
      inst✝² : T2Space E
      inst✝¹ : ContinuousSMul Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      g : E → Real
      h1 : Eq (g 0) 0
      h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
      h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
      h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
      h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
      p : Real
      hp : LT.lt 0 p
      F : Type u_1 := E
      this✝⁵ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
      this✝⁴ : NormedSpace Real F := NormedSpace.mk ⋯
      this✝³ : TopologicalSpace F := UniformSpace.toTopologicalSpace
      this✝² : MeasurableSpace F := borel F
      this✝¹ : BorelSpace F
      φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
      ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
      this✝ : ν.IsAddHaarMeasure
      ψ : MeasurableEquiv E F := φ.toHomeomorph.toMeasurableEquiv
      this : MeasureTheory.MeasurePreserving (⇑ψ) μ ν
      ⊢ Eq (ENNReal.ofReal (HDiv.hDiv (MeasureTheory.integral μ fun x => Real.exp (N …
    -/
    rw [← this.integral_comp']
    /-
      case h.e'_3
      E : Type u_1
      inst✝⁷ : AddCommGroup E
      inst✝⁶ : Module Real E
      inst✝⁵ : FiniteDimensional Real E
      mE : MeasurableSpace E
      tE : TopologicalSpace E
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : BorelSpace E
      inst✝² : T2Space E
      inst✝¹ : ContinuousSMul Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      g : E → Real
      h1 : Eq (g 0) 0
      h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
      h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
      h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
      h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
      p : Real
      hp : LT.lt 0 p
      F : Type u_1 := E
      this✝⁵ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
      this✝⁴ : NormedSpace Real F := NormedSpace.mk ⋯
      this✝³ : TopologicalSpace F := UniformSpace.toTopologicalSpace
      this✝² : MeasurableSpace F := borel F
      this✝¹ : BorelSpace F
      φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
      ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
      this✝ : ν.IsAddHaarMeasure
      ψ : MeasurableEquiv E F := φ.toHomeomorph.toMeasurableEquiv
      this : MeasureTheory.MeasurePreserving (⇑ψ) μ ν
      ⊢ Eq (ENNReal.ofReal (HDiv.hDiv (MeasureTheory.integral μ fun x => Real.exp (N …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem MeasureTheory.measure_le_eq_lt [Nontrivial E] (r : ℝ) :
    μ {x : E | g x ≤ r} = μ {x : E | g x < r} := by
  -- We copy `E` to a new type `F` on which we will put the norm defined by `g`
  /-
    E : Type u_1
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    inst✝⁶ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : BorelSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    inst✝ : Nontrivial E
    r : Real
    ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (setOf fun x => LT.lt (g x) r))
  -/
  letI F : Type _ := E
  letI : NormedAddCommGroup F :=
  { norm := g
    dist := fun x y => g (x - y)
    dist_self := by simp only [_root_.sub_self, h1, forall_const]
    dist_comm := fun _ _ => by dsimp [dist]; rw [← h2, neg_sub]
    dist_triangle := fun x y z => by convert h3 (x - y) (y - z) using 1; simp [F]
    edist := fun x y => .ofReal (g (x - y))
    edist_dist := fun _ _ => rfl
    eq_of_dist_eq_zero := by convert fun _ _ h => eq_of_sub_eq_zero (h4 h) }
  letI : NormedSpace ℝ F :=
  { norm_smul_le := fun _ _ ↦ h5 _ _ }
  -- We put the new topology on F
  /-
    E : Type u_1
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    inst✝⁶ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : BorelSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    inst✝ : Nontrivial E
    r : Real
    F : Type u_1 := E
    this✝ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this : NormedSpace Real F := NormedSpace.mk ⋯
    ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (setOf fun x => LT.lt (g x) r))
  -/
  letI : TopologicalSpace F := UniformSpace.toTopologicalSpace
  /-
    E : Type u_1
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    inst✝⁶ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : BorelSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    inst✝ : Nontrivial E
    r : Real
    F : Type u_1 := E
    this✝¹ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝ : NormedSpace Real F := NormedSpace.mk ⋯
    this : TopologicalSpace F := UniformSpace.toTopologicalSpace
    ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (setOf fun x => LT.lt (g x) r))
  -/
  letI : MeasurableSpace F := borel F
  /-
    E : Type u_1
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    inst✝⁶ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : BorelSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    inst✝ : Nontrivial E
    r : Real
    F : Type u_1 := E
    this✝² : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝¹ : NormedSpace Real F := NormedSpace.mk ⋯
    this✝ : TopologicalSpace F := UniformSpace.toTopologicalSpace
    this : MeasurableSpace F := borel F
    ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (setOf fun x => LT.lt (g x) r))
  -/
  have : BorelSpace F := { measurable_eq := rfl }
  -- The map between `E` and `F` as a continuous linear equivalence
  let φ := @LinearEquiv.toContinuousLinearEquiv ℝ _ E _ _ tE _ _ F _ _ _ _ _ _ _ _ _
    (LinearEquiv.refl ℝ E : E ≃ₗ[ℝ] F)
  -- The measure `ν` is the measure on `F` defined by `μ`
  -- Since we have two different topologies, it is necessary to specify the topology of E
  /-
    E : Type u_1
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    inst✝⁶ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : BorelSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    inst✝ : Nontrivial E
    r : Real
    F : Type u_1 := E
    this✝³ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝² : NormedSpace Real F := NormedSpace.mk ⋯
    this✝¹ : TopologicalSpace F := UniformSpace.toTopologicalSpace
    this✝ : MeasurableSpace F := borel F
    this : BorelSpace F
    φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
    ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (setOf fun x => LT.lt (g x) r))
  -/
  let ν : Measure F := @Measure.map E F mE _ φ μ
  have : IsAddHaarMeasure ν :=
    @ContinuousLinearEquiv.isAddHaarMeasure_map E F ℝ ℝ _ _ _ _ _ _ tE _ _ _ _ _ _ _ mE _ _ _ φ μ _
  /-
    E : Type u_1
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    inst✝⁶ : FiniteDimensional Real E
    mE : MeasurableSpace E
    tE : TopologicalSpace E
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : BorelSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousSMul Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    g : E → Real
    h1 : Eq (g 0) 0
    h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
    h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
    h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
    inst✝ : Nontrivial E
    r : Real
    F : Type u_1 := E
    this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
    this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
    this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
    ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
    this : ν.IsAddHaarMeasure
    ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (setOf fun x => LT.lt (g x) r))
  -/
  convert addHaar_closedBall_eq_addHaar_ball ν 0 r using 1
    /-
      case h.e'_2
      E : Type u_1
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      inst✝⁶ : FiniteDimensional Real E
      mE : MeasurableSpace E
      tE : TopologicalSpace E
      inst✝⁵ : TopologicalAddGroup E
      inst✝⁴ : BorelSpace E
      inst✝³ : T2Space E
      inst✝² : ContinuousSMul Real E
      μ : MeasureTheory.Measure E
      inst✝¹ : μ.IsAddHaarMeasure
      g : E → Real
      h1 : Eq (g 0) 0
      h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
      h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
      h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
      h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
      inst✝ : Nontrivial E
      r : Real
      F : Type u_1 := E
      this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
      this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
      this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
      ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
      this : ν.IsAddHaarMeasure
      ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (ν (Metric.closedBall 0 r))
    -/
  · rw [@Measure.map_apply E F mE _ μ φ _ _ measurableSet_closedBall]
      /-
        case h.e'_2
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Eq (μ (setOf fun x => LE.le (g x) r)) (μ (Set.preimage (⇑φ) (Metric.closedBa …
      -/
    · congr!
      /-
        case h.e'_2.h.e'_6.h.e'_1.h.a
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        x✝ : E
        ⊢ Iff (LE.le (g x✝) r) (Metric.closedBall 0 r x✝)
      -/
      simp_rw [Metric.closedBall, dist_zero_right]
      /-
        case h.e'_2.h.e'_6.h.e'_1.h.a
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        x✝ : E
        ⊢ Iff (LE.le (g x✝) r) (setOf (fun y => LE.le (Norm.norm y) r) x✝)
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Measurable ⇑φ
      -/
    · refine @Continuous.measurable E F tE mE _ _ _ _ φ ?_
      /-
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Continuous ⇑φ
      -/
      exact @ContinuousLinearEquiv.continuous ℝ ℝ _ _ _ _ _ _ E tE _ F _ _ _ _ φ
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3
      E : Type u_1
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      inst✝⁶ : FiniteDimensional Real E
      mE : MeasurableSpace E
      tE : TopologicalSpace E
      inst✝⁵ : TopologicalAddGroup E
      inst✝⁴ : BorelSpace E
      inst✝³ : T2Space E
      inst✝² : ContinuousSMul Real E
      μ : MeasureTheory.Measure E
      inst✝¹ : μ.IsAddHaarMeasure
      g : E → Real
      h1 : Eq (g 0) 0
      h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
      h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
      h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
      h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
      inst✝ : Nontrivial E
      r : Real
      F : Type u_1 := E
      this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
      this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
      this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
      ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
      this : ν.IsAddHaarMeasure
      ⊢ Eq (μ (setOf fun x => LT.lt (g x) r)) (ν (Metric.ball 0 r))
    -/
  · rw [@Measure.map_apply E F mE _ μ φ _ _ measurableSet_ball]
      /-
        case h.e'_3
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Eq (μ (setOf fun x => LT.lt (g x) r)) (μ (Set.preimage (⇑φ) (Metric.ball 0 r …
      -/
    · congr!
      /-
        case h.e'_3.h.e'_6.h.e'_1.h.a
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        x✝ : E
        ⊢ Iff (LT.lt (g x✝) r) (Metric.ball 0 r x✝)
      -/
      simp_rw [Metric.ball, dist_zero_right]
      /-
        case h.e'_3.h.e'_6.h.e'_1.h.a
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        x✝ : E
        ⊢ Iff (LT.lt (g x✝) r) (setOf (fun y => LT.lt (Norm.norm y) r) x✝)
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Measurable ⇑φ
      -/
    · refine @Continuous.measurable E F tE mE _ _ _ _ φ ?_
      /-
        E : Type u_1
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module Real E
        inst✝⁶ : FiniteDimensional Real E
        mE : MeasurableSpace E
        tE : TopologicalSpace E
        inst✝⁵ : TopologicalAddGroup E
        inst✝⁴ : BorelSpace E
        inst✝³ : T2Space E
        inst✝² : ContinuousSMul Real E
        μ : MeasureTheory.Measure E
        inst✝¹ : μ.IsAddHaarMeasure
        g : E → Real
        h1 : Eq (g 0) 0
        h2 : ∀ (x : E), Eq (g (Neg.neg x)) (g x)
        h3 : ∀ (x y : E), LE.le (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
        h4 : ∀ {x : E}, Eq (g x) 0 → Eq x 0
        h5 : ∀ (r : Real) (x : E), LE.le (g (HSMul.hSMul r x)) (HMul.hMul (abs r) (g x))
        inst✝ : Nontrivial E
        r : Real
        F : Type u_1 := E
        this✝⁴ : NormedAddCommGroup F := NormedAddCommGroup.mk ⋯
        this✝³ : NormedSpace Real F := NormedSpace.mk ⋯
        this✝² : TopologicalSpace F := UniformSpace.toTopologicalSpace
        this✝¹ : MeasurableSpace F := borel F
        this✝ : BorelSpace F
        φ : ContinuousLinearEquiv (RingHom.id Real) E F := (LinearEquiv.refl Real E).t …
        ν : MeasureTheory.Measure F := MeasureTheory.Measure.map (⇑φ) μ
        this : ν.IsAddHaarMeasure
        ⊢ Continuous ⇑φ
      -/
      exact @ContinuousLinearEquiv.continuous ℝ ℝ _ _ _ _ _ _ E tE _ F _ _ _ _ φ
      /-
        🎉 no goals
      -/


theorem MeasureTheory.volume_sum_rpow_lt_one (hp : 1 ≤ p) :
    volume {x : ι → ℝ | ∑ i, |x i| ^ p < 1} =
      .ofReal ((2 * Gamma (1 / p + 1)) ^ card ι / Gamma (card ι / p + 1)) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have h₁ : 0 < p := by linarith
  have h₂ : ∀ x : ι → ℝ, 0 ≤ ∑ i, |x i| ^ p := by
    refine fun _ => Finset.sum_nonneg' ?_
    exact fun i => (fun _ => rpow_nonneg (abs_nonneg _) _) _
  -- We collect facts about `Lp` norms that will be used in `measure_lt_one_eq_integral_div_gamma`
  have eq_norm := fun x : ι → ℝ => (PiLp.norm_eq_sum (p := .ofReal p) (f := x)
    ((toReal_ofReal (le_of_lt h₁)).symm ▸ h₁))
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun i …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  simp_rw [toReal_ofReal (le_of_lt h₁), Real.norm_eq_abs] at eq_norm
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have : Fact (1 ≤ ENNReal.ofReal p) := fact_iff.mpr (ofReal_one ▸ (ofReal_le_ofReal hp))
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have nm_zero := norm_zero (E := PiLp (.ofReal p) (fun _ : ι => ℝ))
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have eq_zero := fun x : ι → ℝ => norm_eq_zero (E := PiLp (.ofReal p) (fun _ : ι => ℝ)) (a := x)
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Real), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have nm_neg := fun x : ι → ℝ => norm_neg (E := PiLp (.ofReal p) (fun _ : ι => ℝ)) x
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Real), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Real), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have nm_add := fun x y : ι → ℝ => norm_add_le (E := PiLp (.ofReal p) (fun _ : ι => ℝ)) x y
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Real), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Real), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    nm_add : ∀ (x y : ι → Real), LE.le (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Nor …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  simp_rw [eq_norm] at eq_zero nm_zero nm_neg nm_add
  have nm_smul := fun (r : ℝ) (x : ι → ℝ) =>
    norm_smul_le (β := PiLp (.ofReal p) (fun _ : ι => ℝ)) r x
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    eq_zero : ∀ (x : ι → Real), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPo …
    nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (abs (0 x)) p) (HD …
    nm_neg : ∀ (x : ι → Real), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow …
    nm_add : ∀ (x y : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => HPow …
    nm_smul : ∀ (r : Real) (x : ι → Real), LE.le (Norm.norm (HSMul.hSMul r x)) (HM …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  simp_rw [eq_norm, norm_eq_abs] at nm_smul
  -- We use `measure_lt_one_eq_integral_div_gamma` with `g` equals to the norm `L_p`
  convert (measure_lt_one_eq_integral_div_gamma (volume : Measure (ι → ℝ))
    (g := fun x => (∑ i, |x i| ^ p) ^ (1 / p)) nm_zero nm_neg nm_add (eq_zero _).mp
    (fun r x => nm_smul r x) (by linarith : 0 < p)) using 4
    /-
      case h.e'_2.h.e'_6.h.e'_2.h.a
      ι : Type u_1
      inst✝ : Fintype ι
      p : Real
      hp : LE.le 1 p
      h₁ : LT.lt 0 p
      h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
      eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
      this : Fact (LE.le 1 (ENNReal.ofReal p))
      eq_zero : ∀ (x : ι → Real), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPo …
      nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (abs (0 x)) p) (HD …
      nm_neg : ∀ (x : ι → Real), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow …
      nm_add : ∀ (x y : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => HPow …
      nm_smul : ∀ (r : Real) (x : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x …
      x✝ : ι → Real
      ⊢ Iff (LT.lt (Finset.univ.sum fun i => HPow.hPow (abs (x✝ i)) p) 1) (LT.lt (HP …
    -/
  · rw [rpow_lt_one_iff' _ (one_div_pos.mpr h₁)]
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      p : Real
      hp : LE.le 1 p
      h₁ : LT.lt 0 p
      h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
      eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
      this : Fact (LE.le 1 (ENNReal.ofReal p))
      eq_zero : ∀ (x : ι → Real), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPo …
      nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (abs (0 x)) p) (HD …
      nm_neg : ∀ (x : ι → Real), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow …
      nm_add : ∀ (x y : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => HPow …
      nm_smul : ∀ (r : Real) (x : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x …
      x✝ : ι → Real
      ⊢ LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x✝ i)) p)
    -/
    exact Finset.sum_nonneg' (fun _ => rpow_nonneg (abs_nonneg _) _)
    /-
      🎉 no goals
    -/
  · simp_rw [← rpow_mul (h₂ _), div_mul_cancel₀ _ (ne_of_gt h₁), Real.rpow_one,
      ← Finset.sum_neg_distrib, exp_sum]
    rw [integral_fintype_prod_eq_pow ι fun x : ℝ => exp (- |x| ^ p), integral_comp_abs
      (f := fun x => exp (- x ^ p)), integral_exp_neg_rpow h₁]
    /-
      case h.e'_3.h.e'_1.h.e'_6.h.e'_1
      ι : Type u_1
      inst✝ : Fintype ι
      p : Real
      hp : LE.le 1 p
      h₁ : LT.lt 0 p
      h₂ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
      eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
      this : Fact (LE.le 1 (ENNReal.ofReal p))
      eq_zero : ∀ (x : ι → Real), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPo …
      nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (abs (0 x)) p) (HD …
      nm_neg : ∀ (x : ι → Real), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow …
      nm_add : ∀ (x y : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => HPow …
      nm_smul : ∀ (r : Real) (x : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x …
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv (↑(Fintype.card ι)) p) 1) (HAdd.hAdd (HDiv.hDiv (↑( …
    -/
  · rw [finrank_fintype_fun_eq_card]
    /-
      🎉 no goals
    -/


theorem MeasureTheory.volume_sum_rpow_lt [Nonempty ι] {p : ℝ} (hp : 1 ≤ p) (r : ℝ) :
    volume {x : ι → ℝ | (∑ i, |x i| ^ p) ^ (1 / p) < r} = (.ofReal r) ^ card ι *
      .ofReal ((2 * Gamma (1 / p + 1)) ^ card ι / Gamma (card ι / p + 1)) := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
  -/
  have h₁ (x : ι → ℝ) : 0 ≤ ∑ i, |x i| ^ p := by positivity
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
  -/
  have h₂ : ∀ x : ι → ℝ, 0 ≤ (∑ i, |x i| ^ p) ^ (1 / p) := fun x => rpow_nonneg (h₁ x) _
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
    h₂ : ∀ (x : ι → Real), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hPow  …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
  -/
  obtain hr | hr := le_or_lt r 0
  · have : {x : ι → ℝ | (∑ i, |x i| ^ p) ^ (1 / p) < r} = ∅ := by
      ext x
      refine ⟨fun hx => ?_, fun hx => hx.elim⟩
      exact not_le.mpr (lt_of_lt_of_le (Set.mem_setOf.mp hx) hr) (h₂ x)
    /-
      case inl
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : Real
      hp : LE.le 1 p
      r : Real
      h₁ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
      h₂ : ∀ (x : ι → Real), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hPow  …
      hr : LE.le r 0
      this : Eq (setOf fun x => LT.lt (HPow.hPow (Finset.univ.sum fun i => HPow.hPow …
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
    -/
    rw [this, measure_empty, ← zero_eq_ofReal.mpr hr, zero_pow Fin.pos'.ne', zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : Real
      hp : LE.le 1 p
      r : Real
      h₁ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
      h₂ : ∀ (x : ι → Real), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hPow  …
      hr : LT.lt 0 r
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
    -/
  · rw [← volume_sum_rpow_lt_one _ hp, ← ofReal_pow (le_of_lt hr), ← finrank_pi ℝ]
    /-
      case inr
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : Real
      hp : LE.le 1 p
      r : Real
      h₁ : ∀ (x : ι → Real), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (abs (x i)) …
      h₂ : ∀ (x : ι → Real), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hPow  …
      hr : LT.lt 0 r
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
    -/
    convert addHaar_smul_of_nonneg volume (le_of_lt hr) {x : ι → ℝ | ∑ i, |x i| ^ p < 1} using 2
    simp_rw [← Set.preimage_smul_inv₀ (ne_of_gt hr), Set.preimage_setOf_eq, Pi.smul_apply,
      smul_eq_mul, abs_mul, mul_rpow (abs_nonneg _) (abs_nonneg _), abs_inv,
      inv_rpow (abs_nonneg _), ← Finset.mul_sum, abs_eq_self.mpr (le_of_lt hr),
      inv_mul_lt_iff₀ (rpow_pos_of_pos hr _), mul_one, ← rpow_lt_rpow_iff
      (rpow_nonneg (h₁ _) _) (le_of_lt hr) (by linarith : 0 < p), ← rpow_mul
      (h₁ _), div_mul_cancel₀ _ (ne_of_gt (by linarith) : p ≠ 0), Real.rpow_one]


theorem MeasureTheory.volume_sum_rpow_le [Nonempty ι] {p : ℝ} (hp : 1 ≤ p) (r : ℝ) :
    volume {x : ι → ℝ | (∑ i, |x i| ^ p) ^ (1 / p) ≤ r} = (.ofReal r) ^ card ι *
      .ofReal ((2 * Gamma (1 / p + 1)) ^ card ι / Gamma (card ι / p + 1)) := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have h₁ : 0 < p := by linarith
  -- We collect facts about `Lp` norms that will be used in `measure_le_one_eq_lt_one`
  have eq_norm := fun x : ι → ℝ => (PiLp.norm_eq_sum (p := .ofReal p) (f := x)
    ((toReal_ofReal (le_of_lt h₁)).symm ▸ h₁))
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun i …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  simp_rw [toReal_ofReal (le_of_lt h₁), Real.norm_eq_abs] at eq_norm
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have : Fact (1 ≤ ENNReal.ofReal p) := fact_iff.mpr (ofReal_one ▸ (ofReal_le_ofReal hp))
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have nm_zero := norm_zero (E := PiLp (.ofReal p) (fun _ : ι => ℝ))
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have eq_zero := fun x : ι → ℝ => norm_eq_zero (E := PiLp (.ofReal p) (fun _ : ι => ℝ)) (a := x)
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Real), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have nm_neg := fun x : ι → ℝ => norm_neg (E := PiLp (.ofReal p) (fun _ : ι => ℝ)) x
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Real), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Real), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have nm_add := fun x y : ι → ℝ => norm_add_le (E := PiLp (.ofReal p) (fun _ : ι => ℝ)) x y
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Real), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Real), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    nm_add : ∀ (x y : ι → Real), LE.le (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Nor …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  simp_rw [eq_norm] at eq_zero nm_zero nm_neg nm_add
  have nm_smul := fun (r : ℝ) (x : ι → ℝ) =>
    norm_smul_le (β := PiLp (.ofReal p) (fun _ : ι => ℝ)) r x
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Real), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fun x …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    eq_zero : ∀ (x : ι → Real), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPo …
    nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (abs (0 x)) p) (HD …
    nm_neg : ∀ (x : ι → Real), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow …
    nm_add : ∀ (x y : ι → Real), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => HPow …
    nm_smul : ∀ (r : Real) (x : ι → Real), LE.le (Norm.norm (HSMul.hSMul r x)) (HM …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  simp_rw [eq_norm, norm_eq_abs] at nm_smul
  rw [measure_le_eq_lt _ nm_zero (fun x ↦ nm_neg x) (fun x y ↦ nm_add x y) (eq_zero _).mp
    (fun r x => nm_smul r x), volume_sum_rpow_lt _ hp]


theorem Complex.volume_sum_rpow_lt_one {p : ℝ} (hp : 1 ≤ p) :
    volume {x : ι → ℂ | ∑ i, ‖x i‖ ^ p < 1} =
      .ofReal ((π * Real.Gamma (2 / p + 1)) ^ card ι / Real.Gamma (2 * card ι / p + 1)) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have h₁ : 0 < p := by linarith
  have h₂ : ∀ x : ι → ℂ, 0 ≤ ∑ i, ‖x i‖ ^ p := by
    refine fun _ => Finset.sum_nonneg' ?_
    exact fun i => (fun _ => rpow_nonneg (norm_nonneg _) _) _
  -- We collect facts about `Lp` norms that will be used in `measure_lt_one_eq_integral_div_gamma`
  have eq_norm := fun x : ι → ℂ => (PiLp.norm_eq_sum (p := .ofReal p) (f := x)
    ((toReal_ofReal (le_of_lt h₁)).symm ▸ h₁))
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  simp_rw [toReal_ofReal (le_of_lt h₁)] at eq_norm
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have : Fact (1 ≤ ENNReal.ofReal p) := fact_iff.mpr (ENNReal.ofReal_one ▸ (ofReal_le_ofReal hp))
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have nm_zero := norm_zero (E := PiLp (.ofReal p) (fun _ : ι => ℂ))
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have eq_zero := fun x : ι → ℂ => norm_eq_zero (E := PiLp (.ofReal p) (fun _ : ι => ℂ)) (a := x)
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have nm_neg := fun x : ι → ℂ => norm_neg (E := PiLp (.ofReal p) (fun _ : ι => ℂ)) x
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Complex), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  have nm_add := fun x y : ι → ℂ => norm_add_le (E := PiLp (.ofReal p) (fun _ : ι => ℂ)) x y
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Complex), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    nm_add : ∀ (x y : ι → Complex), LE.le (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd ( …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  simp_rw [eq_norm] at eq_zero nm_zero nm_neg nm_add
  have nm_smul := fun (r : ℝ) (x : ι → ℂ) =>
    norm_smul_le (β := PiLp (.ofReal p) (fun _ : ι => ℂ)) r x
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : Real
    hp : LE.le 1 p
    h₁ : LT.lt 0 p
    h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 =>  …
    nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (Norm.norm (0 x))  …
    nm_neg : ∀ (x : ι → Complex), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.h …
    nm_add : ∀ (x y : ι → Complex), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => H …
    nm_smul : ∀ (r : Real) (x : ι → Complex), LE.le (Norm.norm (HSMul.hSMul r x))  …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (Finset.univ.sum …
  -/
  simp_rw [eq_norm, norm_eq_abs] at nm_smul
  -- We use `measure_lt_one_eq_integral_div_gamma` with `g` equals to the norm `L_p`
  convert measure_lt_one_eq_integral_div_gamma (volume : Measure (ι → ℂ))
    (g := fun x => (∑ i, ‖x i‖ ^ p) ^ (1 / p)) nm_zero nm_neg nm_add (eq_zero _).mp
    (fun r x => nm_smul r x) (by linarith : 0 < p) using 4
    /-
      case h.e'_2.h.e'_6.h.e'_2.h.a
      ι : Type u_1
      inst✝ : Fintype ι
      p : Real
      hp : LE.le 1 p
      h₁ : LT.lt 0 p
      h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
      eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
      this : Fact (LE.le 1 (ENNReal.ofReal p))
      eq_zero : ∀ (x : ι → Complex), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 =>  …
      nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (Norm.norm (0 x))  …
      nm_neg : ∀ (x : ι → Complex), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.h …
      nm_add : ∀ (x y : ι → Complex), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => H …
      nm_smul : ∀ (r : Real) (x : ι → Complex), LE.le (HPow.hPow (Finset.univ.sum fu …
      x✝ : ι → Complex
      ⊢ Iff (LT.lt (Finset.univ.sum fun i => HPow.hPow (Norm.norm (x✝ i)) p) 1) (LT. …
    -/
  · rw [rpow_lt_one_iff' _ (one_div_pos.mpr h₁)]
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      p : Real
      hp : LE.le 1 p
      h₁ : LT.lt 0 p
      h₂ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
      eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
      this : Fact (LE.le 1 (ENNReal.ofReal p))
      eq_zero : ∀ (x : ι → Complex), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 =>  …
      nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (Norm.norm (0 x))  …
      nm_neg : ∀ (x : ι → Complex), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.h …
      nm_add : ∀ (x y : ι → Complex), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => H …
      nm_smul : ∀ (r : Real) (x : ι → Complex), LE.le (HPow.hPow (Finset.univ.sum fu …
      x✝ : ι → Complex
      ⊢ LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.norm (x✝ i)) p)
    -/
    exact Finset.sum_nonneg' (fun _ => rpow_nonneg (norm_nonneg _) _)
    /-
      🎉 no goals
    -/
  · simp_rw [← rpow_mul (h₂ _), div_mul_cancel₀ _ (ne_of_gt h₁), Real.rpow_one,
      ← Finset.sum_neg_distrib, Real.exp_sum]
    rw [integral_fintype_prod_eq_pow ι fun x : ℂ => Real.exp (- ‖x‖ ^ p),
      Complex.integral_exp_neg_rpow hp]
  · rw [finrank_pi_fintype, Complex.finrank_real_complex, Finset.sum_const, smul_eq_mul,
      Nat.cast_mul, Nat.cast_ofNat, Fintype.card, mul_comm]


theorem Complex.volume_sum_rpow_lt [Nonempty ι] {p : ℝ} (hp : 1 ≤ p) (r : ℝ) :
    volume {x : ι → ℂ | (∑ i, ‖x i‖ ^ p) ^ (1 / p) < r} = (.ofReal r) ^ (2 * card ι) *
      .ofReal ((π * Real.Gamma (2 / p + 1)) ^ card ι / Real.Gamma (2 * card ι / p + 1)) := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
  -/
  have h₁ (x : ι → ℂ) : 0 ≤ ∑ i, ‖x i‖ ^ p := by positivity
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
  -/
  have h₂ : ∀ x : ι → ℂ, 0 ≤ (∑ i, ‖x i‖ ^ p) ^ (1 / p) := fun x => rpow_nonneg (h₁ x) _
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
    h₂ : ∀ (x : ι → Complex), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hP …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
  -/
  obtain hr | hr := le_or_lt r 0
  · have : {x : ι → ℂ | (∑ i, ‖x i‖ ^ p) ^ (1 / p) < r} = ∅ := by
      ext x
      refine ⟨fun hx => ?_, fun hx => hx.elim⟩
      exact not_le.mpr (lt_of_lt_of_le (Set.mem_setOf.mp hx) hr) (h₂ x)
    /-
      case inl
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : Real
      hp : LE.le 1 p
      r : Real
      h₁ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
      h₂ : ∀ (x : ι → Complex), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hP …
      hr : LE.le r 0
      this : Eq (setOf fun x => LT.lt (HPow.hPow (Finset.univ.sum fun i => HPow.hPow …
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
    -/
    rw [this, measure_empty, ← zero_eq_ofReal.mpr hr, zero_pow Fin.pos'.ne', zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : Real
      hp : LE.le 1 p
      r : Real
      h₁ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
      h₂ : ∀ (x : ι → Complex), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hP …
      hr : LT.lt 0 r
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
    -/
  · rw [← Complex.volume_sum_rpow_lt_one _ hp, ← ENNReal.ofReal_pow (le_of_lt hr)]
    /-
      case inr
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : Nonempty ι
      p : Real
      hp : LE.le 1 p
      r : Real
      h₁ : ∀ (x : ι → Complex), LE.le 0 (Finset.univ.sum fun i => HPow.hPow (Norm.no …
      h₂ : ∀ (x : ι → Complex), LE.le 0 (HPow.hPow (Finset.univ.sum fun i => HPow.hP …
      hr : LT.lt 0 r
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LT.lt (HPow.hPow (Fins …
    -/
    convert addHaar_smul_of_nonneg volume (le_of_lt hr) {x : ι → ℂ |  ∑ i, ‖x i‖ ^ p < 1} using 2
    · simp_rw [← Set.preimage_smul_inv₀ (ne_of_gt hr), Set.preimage_setOf_eq, Pi.smul_apply,
        norm_smul, mul_rpow (norm_nonneg _) (norm_nonneg _), Real.norm_eq_abs, abs_inv, inv_rpow
        (abs_nonneg _), ← Finset.mul_sum, abs_eq_self.mpr (le_of_lt hr), inv_mul_lt_iff₀
        (rpow_pos_of_pos hr _), mul_one, ← rpow_lt_rpow_iff (rpow_nonneg (h₁ _) _)
        (le_of_lt hr) (by linarith : 0 < p), ← rpow_mul (h₁ _), div_mul_cancel₀ _
        (ne_of_gt (by linarith) : p ≠ 0), Real.rpow_one]
    · simp_rw [finrank_pi_fintype ℝ, Complex.finrank_real_complex, Finset.sum_const, smul_eq_mul,
        mul_comm, Fintype.card]


theorem Complex.volume_sum_rpow_le [Nonempty ι] {p : ℝ} (hp : 1 ≤ p) (r : ℝ) :
    volume {x : ι → ℂ | (∑ i, ‖x i‖ ^ p) ^ (1 / p) ≤ r} = (.ofReal r) ^ (2 * card ι) *
      .ofReal ((π * Real.Gamma (2 / p + 1)) ^ card ι / Real.Gamma (2 * card ι / p + 1)) := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have h₁ : 0 < p := by linarith
  -- We collect facts about `Lp` norms that will be used in `measure_lt_one_eq_integral_div_gamma`
  have eq_norm := fun x : ι → ℂ => (PiLp.norm_eq_sum (p := .ofReal p) (f := x)
    ((toReal_ofReal (le_of_lt h₁)).symm ▸ h₁))
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  simp_rw [toReal_ofReal (le_of_lt h₁)] at eq_norm
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have : Fact (1 ≤ ENNReal.ofReal p) := fact_iff.mpr (ENNReal.ofReal_one ▸ (ofReal_le_ofReal hp))
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have nm_zero := norm_zero (E := PiLp (.ofReal p) (fun _ : ι => ℂ))
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have eq_zero := fun x : ι → ℂ => norm_eq_zero (E := PiLp (.ofReal p) (fun _ : ι => ℂ)) (a := x)
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have nm_neg := fun x : ι → ℂ => norm_neg (E := PiLp (.ofReal p) (fun _ : ι => ℂ)) x
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Complex), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  have nm_add := fun x y : ι → ℂ => norm_add_le (E := PiLp (.ofReal p) (fun _ : ι => ℂ)) x y
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    nm_zero : Eq (Norm.norm 0) 0
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (Norm.norm x) 0) (Eq x 0)
    nm_neg : ∀ (x : ι → Complex), Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
    nm_add : ∀ (x y : ι → Complex), LE.le (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd ( …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  simp_rw [eq_norm] at eq_zero nm_zero nm_neg nm_add
  have nm_smul := fun (r : ℝ) (x : ι → ℂ) =>
    norm_smul_le (β := PiLp (.ofReal p) (fun _ : ι => ℂ)) r x
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    p : Real
    hp : LE.le 1 p
    r : Real
    h₁ : LT.lt 0 p
    eq_norm : ∀ (x : ι → Complex), Eq (Norm.norm x) (HPow.hPow (Finset.univ.sum fu …
    this : Fact (LE.le 1 (ENNReal.ofReal p))
    eq_zero : ∀ (x : ι → Complex), Iff (Eq (HPow.hPow (Finset.univ.sum fun x_1 =>  …
    nm_zero : Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (Norm.norm (0 x))  …
    nm_neg : ∀ (x : ι → Complex), Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.h …
    nm_add : ∀ (x y : ι → Complex), LE.le (HPow.hPow (Finset.univ.sum fun x_1 => H …
    nm_smul : ∀ (r : Real) (x : ι → Complex), LE.le (Norm.norm (HSMul.hSMul r x))  …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => LE.le (HPow.hPow (Fins …
  -/
  simp_rw [eq_norm, norm_eq_abs] at nm_smul
  rw [measure_le_eq_lt _ nm_zero (fun x ↦ nm_neg x) (fun x y ↦ nm_add x y) (eq_zero _).mp
    (fun r x => nm_smul r x), Complex.volume_sum_rpow_lt _ hp]


theorem volume_ball (x : EuclideanSpace ℝ ι) (r : ℝ) :
    volume (Metric.ball x r) = (.ofReal r) ^ card ι *
      .ofReal (Real.sqrt π ^ card ι / Gamma (card ι / 2 + 1)) := by
  /-
    ι : Type u_1
    inst✝¹ : Nonempty ι
    inst✝ : Fintype ι
    x : EuclideanSpace Real ι
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball x r)) (HMul.hMul (HPow.hP …
  -/
  obtain hr | hr := le_total r 0
  · rw [Metric.ball_eq_empty.mpr hr, measure_empty, ← zero_eq_ofReal.mpr hr, zero_pow card_ne_zero,
      zero_mul]
  · suffices volume (Metric.ball (0 : EuclideanSpace ℝ ι) 1) =
        .ofReal (Real.sqrt π ^ card ι / Gamma (card ι / 2 + 1)) by
      rw [Measure.addHaar_ball _ _ hr, this, ofReal_pow hr, finrank_euclideanSpace]
    rw [← ((volume_preserving_measurableEquiv _).symm).measure_preimage
      measurableSet_ball.nullMeasurableSet]
    /-
      case inr
      ι : Type u_1
      inst✝¹ : Nonempty ι
      inst✝ : Fintype ι
      x : EuclideanSpace Real ι
      r : Real
      hr : LE.le 0 r
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.preimage (⇑(EuclideanSpace.measur …
    -/
    convert (volume_sum_rpow_lt_one ι one_le_two) using 4
    · simp_rw [ball_zero_eq _ zero_le_one, one_pow, Real.rpow_two, sq_abs,
        Set.setOf_app_iff]
    · rw [Gamma_add_one (by norm_num), Gamma_one_half_eq, ← mul_assoc, mul_div_cancel₀ _
        two_ne_zero, one_mul]


theorem volume_closedBall (x : EuclideanSpace ℝ ι) (r : ℝ) :
    volume (Metric.closedBall x r) = (.ofReal r) ^ card ι *
      .ofReal (sqrt π ^ card ι / Gamma (card ι / 2 + 1)) := by
  /-
    ι : Type u_1
    inst✝¹ : Nonempty ι
    inst✝ : Fintype ι
    x : EuclideanSpace Real ι
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, EuclideanSpace.volume_ball]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-06")]
alias Euclidean_space.volume_ball := EuclideanSpace.volume_ball

@[deprecated (since := "2024-04-06")]
alias Euclidean_space.volume_closedBall := EuclideanSpace.volume_closedBall


theorem volume_ball (x : E) (r : ℝ) :
    volume (Metric.ball x r) = (.ofReal r) ^ finrank ℝ E *
      .ofReal (sqrt π ^ finrank ℝ E / Gamma (finrank ℝ E / 2 + 1)) := by
  rw [← ((stdOrthonormalBasis ℝ E).measurePreserving_repr_symm).measure_preimage
      measurableSet_ball.nullMeasurableSet]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.preimage (⇑(stdOrthonormalBasis R …
  -/
  have : Nonempty (Fin (finrank ℝ E)) := Fin.pos_iff_nonempty.mp finrank_pos
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    x : E
    r : Real
    this : Nonempty (Fin (Module.finrank Real E))
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.preimage (⇑(stdOrthonormalBasis R …
  -/
  have := EuclideanSpace.volume_ball (Fin (finrank ℝ E)) ((stdOrthonormalBasis ℝ E).repr x) r
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    x : E
    r : Real
    this✝ : Nonempty (Fin (Module.finrank Real E))
    this : Eq (MeasureTheory.MeasureSpace.volume (Metric.ball ((stdOrthonormalBasi …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.preimage (⇑(stdOrthonormalBasis R …
  -/
  simp_rw [Fintype.card_fin] at this
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    x : E
    r : Real
    this✝ : Nonempty (Fin (Module.finrank Real E))
    this : Eq (MeasureTheory.MeasureSpace.volume (Metric.ball ((stdOrthonormalBasi …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.preimage (⇑(stdOrthonormalBasis R …
  -/
  convert this
  /-
    case h.e'_2.h.e'_6
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    x : E
    r : Real
    this✝ : Nonempty (Fin (Module.finrank Real E))
    this : Eq (MeasureTheory.MeasureSpace.volume (Metric.ball ((stdOrthonormalBasi …
    ⊢ Eq (Set.preimage (⇑(stdOrthonormalBasis Real E).repr.symm) (Metric.ball x r) …
  -/
  simp only [LinearIsometryEquiv.preimage_ball, LinearIsometryEquiv.symm_symm, _root_.map_zero]
  /-
    🎉 no goals
  -/


theorem volume_closedBall (x : E) (r : ℝ) :
    volume (Metric.closedBall x r) = (.ofReal r) ^ finrank ℝ E *
      .ofReal (sqrt π ^ finrank ℝ E / Gamma (finrank ℝ E / 2 + 1)) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, InnerProductSpace.volume_ball _]
  /-
    🎉 no goals
  -/


lemma volume_ball_of_dim_even {k : ℕ} (hk : finrank ℝ E = 2 * k) (x : E) (r : ℝ) :
    volume (ball x r) = .ofReal r ^ finrank ℝ E * .ofReal (π ^ k / (k : ℕ)!) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    k : Nat
    hk : Eq (Module.finrank Real E) (HMul.hMul 2 k)
    x : E
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball x r)) (HMul.hMul (HPow.hP …
  -/
  rw [volume_ball, hk, pow_mul, pow_mul, sq_sqrt pi_nonneg]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    k : Nat
    hk : Eq (Module.finrank Real E) (HMul.hMul 2 k)
    x : E
    r : Real
    ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow (ENNReal.ofReal r) 2) k) (ENNReal.ofReal …
  -/
  congr
  /-
    case e_a.e_r.e_a
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    k : Nat
    hk : Eq (Module.finrank Real E) (HMul.hMul 2 k)
    x : E
    r : Real
    ⊢ Eq (Real.Gamma (HAdd.hAdd (HDiv.hDiv (↑(HMul.hMul 2 k)) 2) 1)) ↑k.factorial
  -/
  simp [Gamma_nat_eq_factorial]
  /-
    🎉 no goals
  -/


lemma volume_closedBall_of_dim_even {k : ℕ} (hk : finrank ℝ E = 2 * k) (x : E) (r : ℝ) :
    volume (closedBall x r) = .ofReal r ^ finrank ℝ E * .ofReal (π ^ k / (k : ℕ)!) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : Nontrivial E
    k : Nat
    hk : Eq (Module.finrank Real E) (HMul.hMul 2 k)
    x : E
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, volume_ball_of_dim_even hk x]
  /-
    🎉 no goals
  -/


lemma volume_ball_of_dim_odd {k : ℕ} (hk : finrank ℝ E = 2 * k + 1) (x : E) (r : ℝ) :
    volume (ball x r) =
      .ofReal r ^ finrank ℝ E * .ofReal (π ^ k * 2 ^ (k + 1) / (finrank ℝ E : ℕ)‼) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball x r)) (HMul.hMul (HPow.hP …
  -/
  have : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (hk ▸ (2 * k).succ_pos)
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    this : Nontrivial E
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball x r)) (HMul.hMul (HPow.hP …
  -/
  rw [volume_ball, hk, pow_succ (√π), pow_mul, sq_sqrt pi_nonneg, mul_div_assoc, mul_div_assoc]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    this : Nontrivial E
    ⊢ Eq (HMul.hMul (HPow.hPow (ENNReal.ofReal r) (HAdd.hAdd (HMul.hMul 2 k) 1)) ( …
  -/
  congr 3
  simp? [add_div, add_right_comm, -one_div, Gamma_nat_add_one_add_half] says
    simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, Nat.cast_one, add_div, ne_eq,
      OfNat.ofNat_ne_zero, not_false_eq_true, mul_div_cancel_left₀, add_right_comm,
      Gamma_nat_add_one_add_half]
  /-
    case e_a.e_r.e_a
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    this : Nontrivial E
    ⊢ Eq (HDiv.hDiv Real.pi.sqrt (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd (HMul.hMul 2 k …
  -/
  field_simp
  /-
    case e_a.e_r.e_a
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    this : Nontrivial E
    ⊢ Eq (HMul.hMul (HMul.hMul Real.pi.sqrt (HPow.hPow 2 (HAdd.hAdd k 1))) ↑(HAdd. …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma volume_closedBall_of_dim_odd {k : ℕ} (hk : finrank ℝ E = 2 * k + 1) (x : E) (r : ℝ) :
    volume (closedBall x r) =
      .ofReal r ^ finrank ℝ E * .ofReal (π ^ k * 2 ^ (k + 1) / (finrank ℝ E : ℕ)‼) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  have : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (hk ▸ (2 * k).succ_pos)
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    k : Nat
    hk : Eq (Module.finrank Real E) (HAdd.hAdd (HMul.hMul 2 k) 1)
    x : E
    r : Real
    this : Nontrivial E
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, volume_ball_of_dim_odd hk x r]
  /-
    🎉 no goals
  -/


@[simp]
lemma volume_ball_fin_two (x : EuclideanSpace ℝ (Fin 2)) (r : ℝ) :
    volume (ball x r) = .ofReal r ^ 2 * .ofReal π := by
  /-
    x : EuclideanSpace Real (Fin 2)
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball x r)) (HMul.hMul (HPow.hP …
  -/
  norm_num [InnerProductSpace.volume_ball_of_dim_even (k := 1) (by simp) x]
  /-
    🎉 no goals
  -/


@[simp]
lemma volume_closedBall_fin_two (x : EuclideanSpace ℝ (Fin 2)) (r : ℝ) :
    volume (closedBall x r) = .ofReal r ^ 2 * .ofReal π := by
  /-
    x : EuclideanSpace Real (Fin 2)
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, volume_ball_fin_two x r]
  /-
    🎉 no goals
  -/


@[simp]
lemma volume_ball_fin_three (x : EuclideanSpace ℝ (Fin 3)) (r : ℝ) :
    volume (ball x r) = .ofReal r ^ 3 * .ofReal (π * 4 / 3) := by
  /-
    x : EuclideanSpace Real (Fin 3)
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball x r)) (HMul.hMul (HPow.hP …
  -/
  norm_num [InnerProductSpace.volume_ball_of_dim_odd (k := 1) (by simp) x]
  /-
    🎉 no goals
  -/


@[simp]
lemma volume_closedBall_fin_three (x : EuclideanSpace ℝ (Fin 3)) (r : ℝ) :
    volume (closedBall x r) = .ofReal r ^ 3 * .ofReal (π * 4 / 3) := by
  /-
    x : EuclideanSpace Real (Fin 3)
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, volume_ball_fin_three x]
  /-
    🎉 no goals
  -/


@[simp]
theorem Complex.volume_ball (a : ℂ) (r : ℝ) :
    volume (Metric.ball a r) = .ofReal r ^ 2 * NNReal.pi := by
  simp [InnerProductSpace.volume_ball_of_dim_even (k := 1) (by simp) a,
    ← NNReal.coe_real_pi, ofReal_coe_nnreal]


@[simp]
theorem Complex.volume_closedBall (a : ℂ) (r : ℝ) :
    volume (Metric.closedBall a r) = .ofReal r ^ 2 * NNReal.pi := by
  /-
    a : Complex
    r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall a r)) (HMul.hMul (H …
  -/
  rw [addHaar_closedBall_eq_addHaar_ball, Complex.volume_ball]
  /-
    🎉 no goals
  -/


