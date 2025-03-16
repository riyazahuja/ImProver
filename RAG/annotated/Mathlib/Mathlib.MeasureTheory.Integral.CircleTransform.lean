/-- Given a function `f : ℂ → E`, `circleTransform R z w f` is the function mapping `θ` to
`(2 * ↑π * I)⁻¹ • deriv (circleMap z R) θ • ((circleMap z R θ) - w)⁻¹ • f (circleMap z R θ)`.

If `f` is differentiable and `w` is in the interior of the ball, then the integral from `0` to
`2 * π` of this gives the value `f(w)`. -/
def circleTransform (f : ℂ → E) (θ : ℝ) : E :=
  (2 * ↑π * I)⁻¹ • deriv (circleMap z R) θ • (circleMap z R θ - w)⁻¹ • f (circleMap z R θ)


/-- The derivative of `circleTransform` w.r.t `w`. -/
def circleTransformDeriv (f : ℂ → E) (θ : ℝ) : E :=
  (2 * ↑π * I)⁻¹ • deriv (circleMap z R) θ • ((circleMap z R θ - w) ^ 2)⁻¹ • f (circleMap z R θ)


theorem circleTransformDeriv_periodic (f : ℂ → E) :
    Periodic (circleTransformDeriv R z w f) (2 * π) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    ⊢ Function.Periodic (Complex.circleTransformDeriv R z w f) (HMul.hMul 2 Real.pi)
  -/
  have := periodic_circleMap
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    this : ∀ (c : Complex) (R : Real), Function.Periodic (circleMap c R) (HMul.hMu …
    ⊢ Function.Periodic (Complex.circleTransformDeriv R z w f) (HMul.hMul 2 Real.pi)
  -/
  simp_rw [Periodic] at *
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    this : ∀ (c : Complex) (R x : Real), Eq (circleMap c R (HAdd.hAdd x (HMul.hMul …
    ⊢ ∀ (x : Real), Eq (Complex.circleTransformDeriv R z w f (HAdd.hAdd x (HMul.hM …
  -/
  intro x
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    this : ∀ (c : Complex) (R x : Real), Eq (circleMap c R (HAdd.hAdd x (HMul.hMul …
    x : Real
    ⊢ Eq (Complex.circleTransformDeriv R z w f (HAdd.hAdd x (HMul.hMul 2 Real.pi)) …
  -/
  simp_rw [circleTransformDeriv, this]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    this : ∀ (c : Complex) (R x : Real), Eq (circleMap c R (HAdd.hAdd x (HMul.hMul …
    x : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (HSMu …
  -/
  congr 2
  /-
    case e_a.e_a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    this : ∀ (c : Complex) (R x : Real), Eq (circleMap c R (HAdd.hAdd x (HMul.hMul …
    x : Real
    ⊢ Eq (deriv (circleMap z R) (HAdd.hAdd x (HMul.hMul 2 Real.pi))) (deriv (circl …
  -/
  simp [this]
  /-
    🎉 no goals
  -/


theorem circleTransformDeriv_eq (f : ℂ → E) : circleTransformDeriv R z w f =
    fun θ => (circleMap z R θ - w)⁻¹ • circleTransform R z w f θ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    ⊢ Eq (Complex.circleTransformDeriv R z w f) fun θ => HSMul.hSMul (Inv.inv (HSu …
  -/
  ext
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    x✝ : Real
    ⊢ Eq (Complex.circleTransformDeriv R z w f x✝) (HSMul.hSMul (Inv.inv (HSub.hSu …
  -/
  simp_rw [circleTransformDeriv, circleTransform, ← mul_smul, ← mul_assoc]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    x✝ : Real
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real …
  -/
  ring_nf
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    x✝ : Real
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv ↑Real.p …
  -/
  rw [inv_pow]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    x✝ : Real
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv ↑Real.p …
  -/
  congr
  /-
    case h.e_a.e_a.e_a.e_a
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    x✝ : Real
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul (circleMap z R x✝) w …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem integral_circleTransform (f : ℂ → E) :
    (∫ θ : ℝ in (0)..2 * π, circleTransform R z w f θ) =
      (2 * ↑π * I)⁻¹ • ∮ z in C(z, R), (z - w)⁻¹ • f z := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    ⊢ Eq (intervalIntegral (fun θ => Complex.circleTransform R z w f θ) 0 (HMul.hM …
  -/
  simp_rw [circleTransform, circleIntegral, deriv_circleMap, circleMap]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    z w : Complex
    f : Complex → E
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem continuous_circleTransform {R : ℝ} (hR : 0 < R) {f : ℂ → E} {z w : ℂ}
    (hf : ContinuousOn f <| sphere z R) (hw : w ∈ ball z R) :
    Continuous (circleTransform R z w f) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    hR : LT.lt 0 R
    f : Complex → E
    z w : Complex
    hf : ContinuousOn f (Metric.sphere z R)
    hw : Membership.mem (Metric.ball z R) w
    ⊢ Continuous (Complex.circleTransform R z w f)
  -/
  apply_rules [Continuous.smul, continuous_const]
    /-
      case hg.hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R : Real
      hR : LT.lt 0 R
      f : Complex → E
      z w : Complex
      hf : ContinuousOn f (Metric.sphere z R)
      hw : Membership.mem (Metric.ball z R) w
      ⊢ Continuous (deriv (circleMap z R))
    -/
  · rw [funext <| deriv_circleMap _ _]
    /-
      case hg.hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R : Real
      hR : LT.lt 0 R
      f : Complex → E
      z w : Complex
      hf : ContinuousOn f (Metric.sphere z R)
      hw : Membership.mem (Metric.ball z R) w
      ⊢ Continuous fun x => HMul.hMul (circleMap 0 R x) Complex.I
    -/
    apply_rules [Continuous.mul, continuous_circleMap 0 R, continuous_const]
    /-
      🎉 no goals
    -/
    /-
      case hg.hg.hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R : Real
      hR : LT.lt 0 R
      f : Complex → E
      z w : Complex
      hf : ContinuousOn f (Metric.sphere z R)
      hw : Membership.mem (Metric.ball z R) w
      ⊢ Continuous fun x => Inv.inv (HSub.hSub (circleMap z R x) w)
    -/
  · exact continuous_circleMap_inv hw
    /-
      🎉 no goals
    -/
    /-
      case hg.hg.hg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R : Real
      hR : LT.lt 0 R
      f : Complex → E
      z w : Complex
      hf : ContinuousOn f (Metric.sphere z R)
      hw : Membership.mem (Metric.ball z R) w
      ⊢ Continuous fun x => f (circleMap z R x)
    -/
  · apply ContinuousOn.comp_continuous hf (continuous_circleMap z R)
    /-
      case hg.hg.hg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R : Real
      hR : LT.lt 0 R
      f : Complex → E
      z w : Complex
      hf : ContinuousOn f (Metric.sphere z R)
      hw : Membership.mem (Metric.ball z R) w
      ⊢ ∀ (x : Real), Membership.mem (Metric.sphere z R) (circleMap z R x)
    -/
    exact fun _ => (circleMap_mem_sphere _ hR.le) _
    /-
      🎉 no goals
    -/


theorem continuous_circleTransformDeriv {R : ℝ} (hR : 0 < R) {f : ℂ → E} {z w : ℂ}
    (hf : ContinuousOn f (sphere z R)) (hw : w ∈ ball z R) :
    Continuous (circleTransformDeriv R z w f) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    hR : LT.lt 0 R
    f : Complex → E
    z w : Complex
    hf : ContinuousOn f (Metric.sphere z R)
    hw : Membership.mem (Metric.ball z R) w
    ⊢ Continuous (Complex.circleTransformDeriv R z w f)
  -/
  rw [circleTransformDeriv_eq]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R : Real
    hR : LT.lt 0 R
    f : Complex → E
    z w : Complex
    hf : ContinuousOn f (Metric.sphere z R)
    hw : Membership.mem (Metric.ball z R) w
    ⊢ Continuous fun θ => HSMul.hSMul (Inv.inv (HSub.hSub (circleMap z R θ) w)) (C …
  -/
  exact (continuous_circleMap_inv hw).smul (continuous_circleTransform hR hf hw)
  /-
    🎉 no goals
  -/


/-- A useful bound for circle integrals (with complex codomain)-/
def circleTransformBoundingFunction (R : ℝ) (z : ℂ) (w : ℂ × ℝ) : ℂ :=
  circleTransformDeriv R z w.1 (fun _ => 1) w.2


theorem continuousOn_prod_circle_transform_function {R r : ℝ} (hr : r < R) {z : ℂ} :
    ContinuousOn (fun w : ℂ × ℝ => (circleMap z R w.snd - w.fst)⁻¹ ^ 2)
      (closedBall z r ×ˢ univ) := by
  /-
    R r : Real
    hr : LT.lt r R
    z : Complex
    ⊢ ContinuousOn (fun w => HPow.hPow (Inv.inv (HSub.hSub (circleMap z R w.2) w.1 …
  -/
  simp_rw [← one_div]
  /-
    R r : Real
    hr : LT.lt r R
    z : Complex
    ⊢ ContinuousOn (fun w => HPow.hPow (HDiv.hDiv 1 (HSub.hSub (circleMap z R w.2) …
  -/
  apply_rules [ContinuousOn.pow, ContinuousOn.div, continuousOn_const]
    /-
      case hf.hg
      R r : Real
      hr : LT.lt r R
      z : Complex
      ⊢ ContinuousOn (fun x => HSub.hSub (circleMap z R x.2) x.1) (SProd.sprod (Metr …
    -/
  · exact ((continuous_circleMap z R).comp_continuousOn continuousOn_snd).sub continuousOn_fst
    /-
      🎉 no goals
    -/
    /-
      case hf.h₀
      R r : Real
      hr : LT.lt r R
      z : Complex
      ⊢ ∀ (x : Prod Complex Real), Membership.mem (SProd.sprod (Metric.closedBall z  …
    -/
  · rintro ⟨a, b⟩ ⟨ha, -⟩
    /-
      case hf.h₀.mk.intro
      R r : Real
      hr : LT.lt r R
      z a : Complex
      b : Real
      ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
      ⊢ Ne (HSub.hSub (circleMap z R { fst := a, snd := b }.2) { fst := a, snd := b  …
    -/
    have ha2 : a ∈ ball z R := closedBall_subset_ball hr ha
    /-
      case hf.h₀.mk.intro
      R r : Real
      hr : LT.lt r R
      z a : Complex
      b : Real
      ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
      ha2 : Membership.mem (Metric.ball z R) a
      ⊢ Ne (HSub.hSub (circleMap z R { fst := a, snd := b }.2) { fst := a, snd := b  …
    -/
    exact sub_ne_zero.2 (circleMap_ne_mem_ball ha2 b)
    /-
      🎉 no goals
    -/


theorem continuousOn_abs_circleTransformBoundingFunction {R r : ℝ} (hr : r < R) (z : ℂ) :
    ContinuousOn (abs ∘ circleTransformBoundingFunction R z) (closedBall z r ×ˢ univ) := by
  have : ContinuousOn (circleTransformBoundingFunction R z) (closedBall z r ×ˢ univ) := by
    apply_rules [ContinuousOn.smul, continuousOn_const]
    · simp only [deriv_circleMap]
      apply_rules [ContinuousOn.mul, (continuous_circleMap 0 R).comp_continuousOn continuousOn_snd,
        continuousOn_const]
    · simpa only [inv_pow] using continuousOn_prod_circle_transform_function hr
  /-
    R r : Real
    hr : LT.lt r R
    z : Complex
    this : ContinuousOn (Complex.circleTransformBoundingFunction R z) (SProd.sprod …
    ⊢ ContinuousOn (Function.comp (⇑Complex.abs) (Complex.circleTransformBoundingF …
  -/
  exact this.norm
  /-
    🎉 no goals
  -/


theorem abs_circleTransformBoundingFunction_le {R r : ℝ} (hr : r < R) (hr' : 0 ≤ r) (z : ℂ) :
    ∃ x : closedBall z r ×ˢ [[0, 2 * π]], ∀ y : closedBall z r ×ˢ [[0, 2 * π]],
    abs (circleTransformBoundingFunction R z y) ≤ abs (circleTransformBoundingFunction R z x) := by
  /-
    R r : Real
    hr : LT.lt r R
    hr' : LE.le 0 r
    z : Complex
    ⊢ Exists fun x => ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HM …
  -/
  have cts := continuousOn_abs_circleTransformBoundingFunction hr z
  have comp : IsCompact (closedBall z r ×ˢ [[0, 2 * π]]) := by
    apply_rules [IsCompact.prod, ProperSpace.isCompact_closedBall z r, isCompact_uIcc]
  have none : (closedBall z r ×ˢ [[0, 2 * π]]).Nonempty :=
    (nonempty_closedBall.2 hr').prod nonempty_uIcc
  /-
    R r : Real
    hr : LT.lt r R
    hr' : LE.le 0 r
    z : Complex
    cts : ContinuousOn (Function.comp (⇑Complex.abs) (Complex.circleTransformBound …
    comp : IsCompact (SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 …
    none : (SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Real.pi)) …
    ⊢ Exists fun x => ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HM …
  -/
  have := IsCompact.exists_isMaxOn comp none (cts.mono <| prod_mono_right (subset_univ _))
  /-
    R r : Real
    hr : LT.lt r R
    hr' : LE.le 0 r
    z : Complex
    cts : ContinuousOn (Function.comp (⇑Complex.abs) (Complex.circleTransformBound …
    comp : IsCompact (SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 …
    none : (SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Real.pi)) …
    this : Exists fun x => And (Membership.mem (SProd.sprod (Metric.closedBall z r …
    ⊢ Exists fun x => ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HM …
  -/
  simpa [isMaxOn_iff] using this
  /-
    🎉 no goals
  -/


/-- The derivative of a `circleTransform` is locally bounded. -/
theorem circleTransformDeriv_bound {R : ℝ} (hR : 0 < R) {z x : ℂ} {f : ℂ → ℂ} (hx : x ∈ ball z R)
    (hf : ContinuousOn f (sphere z R)) : ∃ B ε : ℝ, 0 < ε ∧
      ball x ε ⊆ ball z R ∧ ∀ (t : ℝ), ∀ y ∈ ball x ε, ‖circleTransformDeriv R z y f t‖ ≤ B := by
  /-
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    ⊢ Exists fun B => Exists fun ε => And (LT.lt 0 ε) (And (HasSubset.Subset (Metr …
  -/
  obtain ⟨r, hr, hrx⟩ := exists_lt_mem_ball_of_mem_ball hx
  /-
    case intro.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ⊢ Exists fun B => Exists fun ε => And (LT.lt 0 ε) (And (HasSubset.Subset (Metr …
  -/
  obtain ⟨ε', hε', H⟩ := exists_ball_subset_ball hrx
  obtain ⟨⟨⟨a, b⟩, ⟨ha, hb⟩⟩, hab⟩ :=
    abs_circleTransformBoundingFunction_le hr (pos_of_mem_ball hrx).le z
  /-
    case intro.intro.intro.intro.intro.mk.mk.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ε' : Real
    hε' : GT.gt ε' 0
    H : HasSubset.Subset (Metric.ball x ε') (Metric.ball z r)
    a : Complex
    b : Real
    ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
    hb : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) { fst := a, snd := b }.2
    hab : ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Re …
    ⊢ Exists fun B => Exists fun ε => And (LT.lt 0 ε) (And (HasSubset.Subset (Metr …
  -/
  let V : ℝ → ℂ → ℂ := fun θ w => circleTransformDeriv R z w (fun _ => 1) θ
  obtain ⟨X, -, HX2⟩ := (isCompact_sphere z R).exists_isMaxOn
    (NormedSpace.sphere_nonempty.2 hR.le) hf.norm
  /-
    case intro.intro.intro.intro.intro.mk.mk.intro.intro.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ε' : Real
    hε' : GT.gt ε' 0
    H : HasSubset.Subset (Metric.ball x ε') (Metric.ball z r)
    a : Complex
    b : Real
    ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
    hb : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) { fst := a, snd := b }.2
    hab : ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Re …
    V : Real → Complex → Complex := fun θ w => Complex.circleTransformDeriv R z w  …
    X : Complex
    HX2 : IsMaxOn (fun x => Norm.norm (f x)) (Metric.sphere z R) X
    ⊢ Exists fun B => Exists fun ε => And (LT.lt 0 ε) (And (HasSubset.Subset (Metr …
  -/
  refine ⟨abs (V b a) * abs (f X), ε', hε', H.trans (ball_subset_ball hr.le), fun y v hv ↦ ?_⟩
  obtain ⟨y1, hy1, hfun⟩ :=
    Periodic.exists_mem_Ico₀ (circleTransformDeriv_periodic R z v f) Real.two_pi_pos y
  /-
    case intro.intro.intro.intro.intro.mk.mk.intro.intro.intro.intro.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ε' : Real
    hε' : GT.gt ε' 0
    H : HasSubset.Subset (Metric.ball x ε') (Metric.ball z r)
    a : Complex
    b : Real
    ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
    hb : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) { fst := a, snd := b }.2
    hab : ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Re …
    V : Real → Complex → Complex := fun θ w => Complex.circleTransformDeriv R z w  …
    X : Complex
    HX2 : IsMaxOn (fun x => Norm.norm (f x)) (Metric.sphere z R) X
    y : Real
    v : Complex
    hv : Membership.mem (Metric.ball x ε') v
    y1 : Real
    hy1 : Membership.mem (Set.Ico 0 (HMul.hMul 2 Real.pi)) y1
    hfun : Eq (Complex.circleTransformDeriv R z v f y) (Complex.circleTransformDer …
    ⊢ LE.le (Norm.norm (Complex.circleTransformDeriv R z v f y)) (HMul.hMul (Compl …
  -/
  have hy2 : y1 ∈ [[0, 2 * π]] := Icc_subset_uIcc <| Ico_subset_Icc_self hy1
  /-
    case intro.intro.intro.intro.intro.mk.mk.intro.intro.intro.intro.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ε' : Real
    hε' : GT.gt ε' 0
    H : HasSubset.Subset (Metric.ball x ε') (Metric.ball z r)
    a : Complex
    b : Real
    ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
    hb : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) { fst := a, snd := b }.2
    hab : ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Re …
    V : Real → Complex → Complex := fun θ w => Complex.circleTransformDeriv R z w  …
    X : Complex
    HX2 : IsMaxOn (fun x => Norm.norm (f x)) (Metric.sphere z R) X
    y : Real
    v : Complex
    hv : Membership.mem (Metric.ball x ε') v
    y1 : Real
    hy1 : Membership.mem (Set.Ico 0 (HMul.hMul 2 Real.pi)) y1
    hfun : Eq (Complex.circleTransformDeriv R z v f y) (Complex.circleTransformDer …
    hy2 : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) y1
    ⊢ LE.le (Norm.norm (Complex.circleTransformDeriv R z v f y)) (HMul.hMul (Compl …
  -/
  simp only [isMaxOn_iff, mem_sphere_iff_norm, norm_eq_abs] at HX2
  have := mul_le_mul (hab ⟨⟨v, y1⟩, ⟨ball_subset_closedBall (H hv), hy2⟩⟩)
    (HX2 (circleMap z R y1) (circleMap_mem_sphere z hR.le y1)) (Complex.abs.nonneg _)
    (Complex.abs.nonneg _)
  /-
    case intro.intro.intro.intro.intro.mk.mk.intro.intro.intro.intro.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ε' : Real
    hε' : GT.gt ε' 0
    H : HasSubset.Subset (Metric.ball x ε') (Metric.ball z r)
    a : Complex
    b : Real
    ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
    hb : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) { fst := a, snd := b }.2
    hab : ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Re …
    V : Real → Complex → Complex := fun θ w => Complex.circleTransformDeriv R z w  …
    X : Complex
    y : Real
    v : Complex
    hv : Membership.mem (Metric.ball x ε') v
    y1 : Real
    hy1 : Membership.mem (Set.Ico 0 (HMul.hMul 2 Real.pi)) y1
    hfun : Eq (Complex.circleTransformDeriv R z v f y) (Complex.circleTransformDer …
    hy2 : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) y1
    HX2 : ∀ (x : Complex), Eq (Complex.abs (HSub.hSub x z)) R → LE.le (Complex.abs …
    this : LE.le (HMul.hMul (Complex.abs (Complex.circleTransformBoundingFunction  …
    ⊢ LE.le (Norm.norm (Complex.circleTransformDeriv R z v f y)) (HMul.hMul (Compl …
  -/
  rw [hfun]
  /-
    case intro.intro.intro.intro.intro.mk.mk.intro.intro.intro.intro.intro
    R : Real
    hR : LT.lt 0 R
    z x : Complex
    f : Complex → Complex
    hx : Membership.mem (Metric.ball z R) x
    hf : ContinuousOn f (Metric.sphere z R)
    r : Real
    hr : LT.lt r R
    hrx : Membership.mem (Metric.ball z r) x
    ε' : Real
    hε' : GT.gt ε' 0
    H : HasSubset.Subset (Metric.ball x ε') (Metric.ball z r)
    a : Complex
    b : Real
    ha : Membership.mem (Metric.closedBall z r) { fst := a, snd := b }.1
    hb : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) { fst := a, snd := b }.2
    hab : ∀ (y : ↑(SProd.sprod (Metric.closedBall z r) (Set.uIcc 0 (HMul.hMul 2 Re …
    V : Real → Complex → Complex := fun θ w => Complex.circleTransformDeriv R z w  …
    X : Complex
    y : Real
    v : Complex
    hv : Membership.mem (Metric.ball x ε') v
    y1 : Real
    hy1 : Membership.mem (Set.Ico 0 (HMul.hMul 2 Real.pi)) y1
    hfun : Eq (Complex.circleTransformDeriv R z v f y) (Complex.circleTransformDer …
    hy2 : Membership.mem (Set.uIcc 0 (HMul.hMul 2 Real.pi)) y1
    HX2 : ∀ (x : Complex), Eq (Complex.abs (HSub.hSub x z)) R → LE.le (Complex.abs …
    this : LE.le (HMul.hMul (Complex.abs (Complex.circleTransformBoundingFunction  …
    ⊢ LE.le (Norm.norm (Complex.circleTransformDeriv R z v f y1)) (HMul.hMul (Comp …
  -/
  simpa [V, circleTransformBoundingFunction, circleTransformDeriv, mul_assoc] using this
  /-
    🎉 no goals
  -/


