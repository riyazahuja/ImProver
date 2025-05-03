theorem measurable_apply₂ [MeasurableSpace E] [OpensMeasurableSpace E]
    [SecondCountableTopologyEither (E →L[𝕜] F) E]
    [MeasurableSpace F] [BorelSpace F] : Measurable fun p : (E →L[𝕜] F) × E => p.1 p.2 :=
  isBoundedBilinearMap_apply.continuous.measurable


/-- The set `A f L r ε` is the set of points `x` around which the function `f` is well approximated
at scale `r` by the linear map `L`, up to an error `ε`. We tweak the definition to make sure that
this is an open set. -/
def A (f : E → F) (L : E →L[𝕜] F) (r ε : ℝ) : Set E :=
  { x | ∃ r' ∈ Ioc (r / 2) r, ∀ y ∈ ball x r', ∀ z ∈ ball x r', ‖f z - f y - L (z - y)‖ < ε * r }


/-- The set `B f K r s ε` is the set of points `x` around which there exists a continuous linear map
`L` belonging to `K` (a given set of continuous linear maps) that approximates well the
function `f` (up to an error `ε`), simultaneously at scales `r` and `s`. -/
def B (f : E → F) (K : Set (E →L[𝕜] F)) (r s ε : ℝ) : Set E :=
  ⋃ L ∈ K, A f L r ε ∩ A f L s ε


/-- The set `D f K` is a complicated set constructed using countable intersections and unions. Its
main use is that, when `K` is complete, it is exactly the set of points where `f` is differentiable,
with a derivative in `K`. -/
def D (f : E → F) (K : Set (E →L[𝕜] F)) : Set E :=
  ⋂ e : ℕ, ⋃ n : ℕ, ⋂ (p ≥ n) (q ≥ n), B f K ((1 / 2) ^ p) ((1 / 2) ^ q) ((1 / 2) ^ e)


theorem isOpen_A (L : E →L[𝕜] F) (r ε : ℝ) : IsOpen (A f L r ε) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    ⊢ IsOpen (FDerivMeasurableAux.A f L r ε)
  -/
  rw [Metric.isOpen_iff]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    ⊢ ∀ (x : E), Membership.mem (FDerivMeasurableAux.A f L r ε) x → Exists fun ε_1 …
  -/
  rintro x ⟨r', r'_mem, hr'⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    x : E
    r' : Real
    r'_mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    ⊢ Exists fun ε_1 => And (GT.gt ε_1 0) (HasSubset.Subset (Metric.ball x ε_1) (F …
  -/
  obtain ⟨s, s_gt, s_lt⟩ : ∃ s : ℝ, r / 2 < s ∧ s < r' := exists_between r'_mem.1
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    x : E
    r' : Real
    r'_mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    ⊢ Exists fun ε_1 => And (GT.gt ε_1 0) (HasSubset.Subset (Metric.ball x ε_1) (F …
  -/
  have : s ∈ Ioc (r / 2) r := ⟨s_gt, le_of_lt (s_lt.trans_le r'_mem.2)⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    x : E
    r' : Real
    r'_mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    ⊢ Exists fun ε_1 => And (GT.gt ε_1 0) (HasSubset.Subset (Metric.ball x ε_1) (F …
  -/
  refine ⟨r' - s, by linarith, fun x' hx' => ⟨s, this, ?_⟩⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    x : E
    r' : Real
    r'_mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    x' : E
    hx' : Membership.mem (Metric.ball x (HSub.hSub r' s)) x'
    ⊢ ∀ (y : E), Membership.mem (Metric.ball x' s) y → ∀ (z : E), Membership.mem ( …
  -/
  have B : ball x' s ⊆ ball x r' := ball_subset (le_of_lt hx')
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    x : E
    r' : Real
    r'_mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    x' : E
    hx' : Membership.mem (Metric.ball x (HSub.hSub r' s)) x'
    B : HasSubset.Subset (Metric.ball x' s) (Metric.ball x r')
    ⊢ ∀ (y : E), Membership.mem (Metric.ball x' s) y → ∀ (z : E), Membership.mem ( …
  -/
  intro y hy z hz
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε : Real
    x : E
    r' : Real
    r'_mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    x' : E
    hx' : Membership.mem (Metric.ball x (HSub.hSub r' s)) x'
    B : HasSubset.Subset (Metric.ball x' s) (Metric.ball x r')
    y : E
    hy : Membership.mem (Metric.ball x' s) y
    z : E
    hz : Membership.mem (Metric.ball x' s) z
    ⊢ LT.lt (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (L (HSub.hSub z y)))) (H …
  -/
  exact hr' y (B hy) z (B hz)
  /-
    🎉 no goals
  -/


theorem isOpen_B {K : Set (E →L[𝕜] F)} {r s ε : ℝ} : IsOpen (B f K r s ε) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    r s ε : Real
    ⊢ IsOpen (FDerivMeasurableAux.B f K r s ε)
  -/
  simp [B, isOpen_biUnion, IsOpen.inter, isOpen_A]
  /-
    🎉 no goals
  -/


theorem A_mono (L : E →L[𝕜] F) (r : ℝ) {ε δ : ℝ} (h : ε ≤ δ) : A f L r ε ⊆ A f L r δ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε δ : Real
    h : LE.le ε δ
    ⊢ HasSubset.Subset (FDerivMeasurableAux.A f L r ε) (FDerivMeasurableAux.A f L  …
  -/
  rintro x ⟨r', r'r, hr'⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε δ : Real
    h : LE.le ε δ
    x : E
    r' : Real
    r'r : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    ⊢ Membership.mem (FDerivMeasurableAux.A f L r δ) x
  -/
  refine ⟨r', r'r, fun y hy z hz => (hr' y hy z hz).trans_le (mul_le_mul_of_nonneg_right h ?_)⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    r ε δ : Real
    h : LE.le ε δ
    x : E
    r' : Real
    r'r : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    y : E
    hy : Membership.mem (Metric.ball x r') y
    z : E
    hz : Membership.mem (Metric.ball x r') z
    ⊢ LE.le 0 r
  -/
  linarith [mem_ball.1 hy, r'r.2, @dist_nonneg _ _ y x]
  /-
    🎉 no goals
  -/


theorem le_of_mem_A {r ε : ℝ} {L : E →L[𝕜] F} {x : E} (hx : x ∈ A f L r ε) {y z : E}
    (hy : y ∈ closedBall x (r / 2)) (hz : z ∈ closedBall x (r / 2)) :
    ‖f z - f y - L (z - y)‖ ≤ ε * r := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    r ε : Real
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hx : Membership.mem (FDerivMeasurableAux.A f L r ε) x
    y z : E
    hy : Membership.mem (Metric.closedBall x (HDiv.hDiv r 2)) y
    hz : Membership.mem (Metric.closedBall x (HDiv.hDiv r 2)) z
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (L (HSub.hSub z y)))) (H …
  -/
  rcases hx with ⟨r', r'mem, hr'⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    r ε : Real
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    x y z : E
    hy : Membership.mem (Metric.closedBall x (HDiv.hDiv r 2)) y
    hz : Membership.mem (Metric.closedBall x (HDiv.hDiv r 2)) z
    r' : Real
    r'mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (L (HSub.hSub z y)))) (H …
  -/
  apply le_of_lt
  /-
    case intro.intro.hab
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    r ε : Real
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    x y z : E
    hy : Membership.mem (Metric.closedBall x (HDiv.hDiv r 2)) y
    hz : Membership.mem (Metric.closedBall x (HDiv.hDiv r 2)) z
    r' : Real
    r'mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : E), Membership.mem (Metric.ball x r') y → ∀ (z : E), Membership.m …
    ⊢ LT.lt (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (L (HSub.hSub z y)))) (H …
  -/
  exact hr' _ ((mem_closedBall.1 hy).trans_lt r'mem.1) _ ((mem_closedBall.1 hz).trans_lt r'mem.1)
  /-
    🎉 no goals
  -/


theorem mem_A_of_differentiable {ε : ℝ} (hε : 0 < ε) {x : E} (hx : DifferentiableAt 𝕜 f x) :
    ∃ R > 0, ∀ r ∈ Ioo (0 : ℝ) R, x ∈ A f (fderiv 𝕜 f x) r ε := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ε : Real
    hε : LT.lt 0 ε
    x : E
    hx : DifferentiableAt 𝕜 f x
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (r : Real), Membership.mem (Set.Ioo 0 R)  …
  -/
  let δ := (ε / 2) / 2
  obtain ⟨R, R_pos, hR⟩ :
      ∃ R > 0, ∀ y ∈ ball x R, ‖f y - f x - fderiv 𝕜 f x (y - x)‖ ≤ δ * ‖y - x‖ :=
    eventually_nhds_iff_ball.1 <| hx.hasFDerivAt.isLittleO.bound <| by positivity
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ε : Real
    hε : LT.lt 0 ε
    x : E
    hx : DifferentiableAt 𝕜 f x
    δ : Real := HDiv.hDiv (HDiv.hDiv ε 2) 2
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (y : E), Membership.mem (Metric.ball x R) y → LE.le (Norm.norm (HSub.hS …
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (r : Real), Membership.mem (Set.Ioo 0 R)  …
  -/
  refine ⟨R, R_pos, fun r hr => ?_⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ε : Real
    hε : LT.lt 0 ε
    x : E
    hx : DifferentiableAt 𝕜 f x
    δ : Real := HDiv.hDiv (HDiv.hDiv ε 2) 2
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (y : E), Membership.mem (Metric.ball x R) y → LE.le (Norm.norm (HSub.hS …
    r : Real
    hr : Membership.mem (Set.Ioo 0 R) r
    ⊢ Membership.mem (FDerivMeasurableAux.A f (fderiv 𝕜 f x) r ε) x
  -/
  have : r ∈ Ioc (r / 2) r := right_mem_Ioc.2 <| half_lt_self hr.1
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ε : Real
    hε : LT.lt 0 ε
    x : E
    hx : DifferentiableAt 𝕜 f x
    δ : Real := HDiv.hDiv (HDiv.hDiv ε 2) 2
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (y : E), Membership.mem (Metric.ball x R) y → LE.le (Norm.norm (HSub.hS …
    r : Real
    hr : Membership.mem (Set.Ioo 0 R) r
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r
    ⊢ Membership.mem (FDerivMeasurableAux.A f (fderiv 𝕜 f x) r ε) x
  -/
  refine ⟨r, this, fun y hy z hz => ?_⟩
  calc
    ‖f z - f y - (fderiv 𝕜 f x) (z - y)‖ =
        ‖f z - f x - (fderiv 𝕜 f x) (z - x) - (f y - f x - (fderiv 𝕜 f x) (y - x))‖ := by
      simp only [map_sub]; abel_nf
    _ ≤ ‖f z - f x - (fderiv 𝕜 f x) (z - x)‖ + ‖f y - f x - (fderiv 𝕜 f x) (y - x)‖ :=
      norm_sub_le _ _
    _ ≤ δ * ‖z - x‖ + δ * ‖y - x‖ :=
      add_le_add (hR _ (ball_subset_ball hr.2.le hz)) (hR _ (ball_subset_ball hr.2.le hy))
    _ ≤ δ * r + δ * r := by rw [mem_ball_iff_norm] at hz hy; gcongr
    _ = (ε / 2) * r := by ring
    _ < ε * r := by gcongr; exacts [hr.1, half_lt_self hε]


theorem norm_sub_le_of_mem_A {c : 𝕜} (hc : 1 < ‖c‖) {r ε : ℝ} (hε : 0 < ε) (hr : 0 < r) {x : E}
    {L₁ L₂ : E →L[𝕜] F} (h₁ : x ∈ A f L₁ r ε) (h₂ : x ∈ A f L₂ r ε) : ‖L₁ - L₂‖ ≤ 4 * ‖c‖ * ε := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r ε : Real
    hε : LT.lt 0 ε
    hr : LT.lt 0 r
    x : E
    L₁ L₂ : ContinuousLinearMap (RingHom.id 𝕜) E F
    h₁ : Membership.mem (FDerivMeasurableAux.A f L₁ r ε) x
    h₂ : Membership.mem (FDerivMeasurableAux.A f L₂ r ε) x
    ⊢ LE.le (Norm.norm (HSub.hSub L₁ L₂)) (HMul.hMul (HMul.hMul 4 (Norm.norm c)) ε)
  -/
  refine opNorm_le_of_shell (half_pos hr) (by positivity) hc ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r ε : Real
    hε : LT.lt 0 ε
    hr : LT.lt 0 r
    x : E
    L₁ L₂ : ContinuousLinearMap (RingHom.id 𝕜) E F
    h₁ : Membership.mem (FDerivMeasurableAux.A f L₁ r ε) x
    h₂ : Membership.mem (FDerivMeasurableAux.A f L₂ r ε) x
    ⊢ ∀ (x : E), LE.le (HDiv.hDiv (HDiv.hDiv r 2) (Norm.norm c)) (Norm.norm x) → L …
  -/
  intro y ley ylt
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r ε : Real
    hε : LT.lt 0 ε
    hr : LT.lt 0 r
    x : E
    L₁ L₂ : ContinuousLinearMap (RingHom.id 𝕜) E F
    h₁ : Membership.mem (FDerivMeasurableAux.A f L₁ r ε) x
    h₂ : Membership.mem (FDerivMeasurableAux.A f L₂ r ε) x
    y : E
    ley : LE.le (HDiv.hDiv (HDiv.hDiv r 2) (Norm.norm c)) (Norm.norm y)
    ylt : LT.lt (Norm.norm y) (HDiv.hDiv r 2)
    ⊢ LE.le (Norm.norm ((HSub.hSub L₁ L₂) y)) (HMul.hMul (HMul.hMul (HMul.hMul 4 ( …
  -/
  rw [div_div, div_le_iff₀' (mul_pos (by norm_num : (0 : ℝ) < 2) (zero_lt_one.trans hc))] at ley
  calc
    ‖(L₁ - L₂) y‖ = ‖f (x + y) - f x - L₂ (x + y - x) - (f (x + y) - f x - L₁ (x + y - x))‖ := by
      simp
    _ ≤ ‖f (x + y) - f x - L₂ (x + y - x)‖ + ‖f (x + y) - f x - L₁ (x + y - x)‖ := norm_sub_le _ _
    _ ≤ ε * r + ε * r := by
      apply add_le_add
      · apply le_of_mem_A h₂
        · simp only [le_of_lt (half_pos hr), mem_closedBall, dist_self]
        · simp only [dist_eq_norm, add_sub_cancel_left, mem_closedBall, ylt.le]
      · apply le_of_mem_A h₁
        · simp only [le_of_lt (half_pos hr), mem_closedBall, dist_self]
        · simp only [dist_eq_norm, add_sub_cancel_left, mem_closedBall, ylt.le]
    _ = 2 * ε * r := by ring
    _ ≤ 2 * ε * (2 * ‖c‖ * ‖y‖) := by gcongr
    _ = 4 * ‖c‖ * ε * ‖y‖ := by ring


/-- Easy inclusion: a differentiability point with derivative in `K` belongs to `D f K`. -/
theorem differentiable_set_subset_D :
    { x | DifferentiableAt 𝕜 f x ∧ fderiv 𝕜 f x ∈ K } ⊆ D f K := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    ⊢ HasSubset.Subset (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.me …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
    ⊢ Membership.mem (FDerivMeasurableAux.D f K) x
  -/
  rw [D, mem_iInter]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
    ⊢ ∀ (i : Nat), Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iIn …
  -/
  intro e
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
    e : Nat
    ⊢ Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iInter fun h =>  …
  -/
  have : (0 : ℝ) < (1 / 2) ^ e := by positivity
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
    e : Nat
    this : LT.lt 0 (HPow.hPow (1 / 2) e)
    ⊢ Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iInter fun h =>  …
  -/
  rcases mem_A_of_differentiable this hx.1 with ⟨R, R_pos, hR⟩
  obtain ⟨n, hn⟩ : ∃ n : ℕ, (1 / 2) ^ n < R :=
    exists_pow_lt_of_lt_one R_pos (by norm_num : (1 : ℝ) / 2 < 1)
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
    e : Nat
    this : LT.lt 0 (HPow.hPow (1 / 2) e)
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (FDerivMeas …
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) R
    ⊢ Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iInter fun h =>  …
  -/
  simp only [mem_iUnion, mem_iInter, B, mem_inter_iff]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
    e : Nat
    this : LT.lt 0 (HPow.hPow (1 / 2) e)
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (FDerivMeas …
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) R
    ⊢ Exists fun i => ∀ (i_1 : Nat), GE.ge i_1 i → ∀ (i_3 : Nat), GE.ge i_3 i → Ex …
  -/
  refine ⟨n, fun p hp q hq => ⟨fderiv 𝕜 f x, hx.2, ⟨?_, ?_⟩⟩⟩ <;>
      /-
        case intro.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
        x : E
        hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
        e : Nat
        this : LT.lt 0 (HPow.hPow (1 / 2) e)
        R : Real
        R_pos : GT.gt R 0
        hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (FDerivMeas …
        n : Nat
        hn : LT.lt (HPow.hPow (1 / 2) n) R
        p : Nat
        hp : GE.ge p n
        q : Nat
        hq : GE.ge q n
        ⊢ Membership.mem (FDerivMeasurableAux.A f (fderiv 𝕜 f x) (HPow.hPow (1 / 2) p) …
      -/
      /-
        case intro.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
        x : E
        hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
        e : Nat
        this : LT.lt 0 (HPow.hPow (1 / 2) e)
        R : Real
        R_pos : GT.gt R 0
        hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (FDerivMeas …
        n : Nat
        hn : LT.lt (HPow.hPow (1 / 2) n) R
        p : Nat
        hp : GE.ge p n
        q : Nat
        hq : GE.ge q n
        ⊢ LE.le (HPow.hPow (1 / 2) p) (HPow.hPow (1 / 2) n)
      -/
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_2
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
        x : E
        hx : Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.m …
        e : Nat
        this : LT.lt 0 (HPow.hPow (1 / 2) e)
        R : Real
        R_pos : GT.gt R 0
        hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (FDerivMeas …
        n : Nat
        hn : LT.lt (HPow.hPow (1 / 2) n) R
        p : Nat
        hp : GE.ge p n
        q : Nat
        hq : GE.ge q n
        ⊢ LE.le (HPow.hPow (1 / 2) q) (HPow.hPow (1 / 2) n)
      -/
      exact pow_le_pow_of_le_one (by norm_num) (by norm_num) (by assumption)
      /-
        🎉 no goals
      -/


/-- Harder inclusion: at a point in `D f K`, the function `f` has a derivative, in `K`. -/
theorem D_subset_differentiable_set {K : Set (E →L[𝕜] F)} (hK : IsComplete K) :
    D f K ⊆ { x | DifferentiableAt 𝕜 f x ∧ fderiv 𝕜 f x ∈ K } := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    ⊢ HasSubset.Subset (FDerivMeasurableAux.D f K) (setOf fun x => And (Differenti …
  -/
  have P : ∀ {n : ℕ}, (0 : ℝ) < (1 / 2) ^ n := fun {n} => pow_pos (by norm_num) n
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    ⊢ HasSubset.Subset (FDerivMeasurableAux.D f K) (setOf fun x => And (Differenti …
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ HasSubset.Subset (FDerivMeasurableAux.D f K) (setOf fun x => And (Differenti …
  -/
  intro x hx
  have :
    ∀ e : ℕ, ∃ n : ℕ, ∀ p q, n ≤ p → n ≤ q →
      ∃ L ∈ K, x ∈ A f L ((1 / 2) ^ p) ((1 / 2) ^ e) ∩ A f L ((1 / 2) ^ q) ((1 / 2) ^ e) := by
    intro e
    have := mem_iInter.1 hx e
    rcases mem_iUnion.1 this with ⟨n, hn⟩
    refine ⟨n, fun p q hp hq => ?_⟩
    simp only [mem_iInter] at hn
    rcases mem_iUnion.1 (hn p hp q hq) with ⟨L, hL⟩
    exact ⟨L, exists_prop.mp <| mem_iUnion.1 hL⟩
  /- Recast the assumptions: for each `e`, there exist `n e` and linear maps `L e p q` in `K`
    such that, for `p, q ≥ n e`, then `f` is well approximated by `L e p q` at scale `2 ^ (-p)` and
    `2 ^ (-q)`, with an error `2 ^ (-e)`. -/
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    hx : Membership.mem (FDerivMeasurableAux.D f K) x
    this : ∀ (e : Nat), Exists fun n => ∀ (p q : Nat), LE.le n p → LE.le n q → Exi …
    ⊢ Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.mem  …
  -/
  choose! n L hn using this
  /- All the operators `L e p q` that show up are close to each other. To prove this, we argue
      that `L e p q` is close to `L e p r` (where `r` is large enough), as both approximate `f` at
      scale `2 ^(- p)`. And `L e p r` is close to `L e' p' r` as both approximate `f` at scale
      `2 ^ (- r)`. And `L e' p' r` is close to `L e' p' q'` as both approximate `f` at scale
      `2 ^ (- p')`. -/
  have M :
    ∀ e p q e' p' q',
      n e ≤ p →
        n e ≤ q →
          n e' ≤ p' → n e' ≤ q' → e ≤ e' → ‖L e p q - L e' p' q'‖ ≤ 12 * ‖c‖ * (1 / 2) ^ e := by
    intro e p q e' p' q' hp hq hp' hq' he'
    let r := max (n e) (n e')
    have I : ((1 : ℝ) / 2) ^ e' ≤ (1 / 2) ^ e :=
      pow_le_pow_of_le_one (by norm_num) (by norm_num) he'
    have J1 : ‖L e p q - L e p r‖ ≤ 4 * ‖c‖ * (1 / 2) ^ e := by
      have I1 : x ∈ A f (L e p q) ((1 / 2) ^ p) ((1 / 2) ^ e) := (hn e p q hp hq).2.1
      have I2 : x ∈ A f (L e p r) ((1 / 2) ^ p) ((1 / 2) ^ e) := (hn e p r hp (le_max_left _ _)).2.1
      exact norm_sub_le_of_mem_A hc P P I1 I2
    have J2 : ‖L e p r - L e' p' r‖ ≤ 4 * ‖c‖ * (1 / 2) ^ e := by
      have I1 : x ∈ A f (L e p r) ((1 / 2) ^ r) ((1 / 2) ^ e) := (hn e p r hp (le_max_left _ _)).2.2
      have I2 : x ∈ A f (L e' p' r) ((1 / 2) ^ r) ((1 / 2) ^ e') :=
        (hn e' p' r hp' (le_max_right _ _)).2.2
      exact norm_sub_le_of_mem_A hc P P I1 (A_mono _ _ I I2)
    have J3 : ‖L e' p' r - L e' p' q'‖ ≤ 4 * ‖c‖ * (1 / 2) ^ e := by
      have I1 : x ∈ A f (L e' p' r) ((1 / 2) ^ p') ((1 / 2) ^ e') :=
        (hn e' p' r hp' (le_max_right _ _)).2.1
      have I2 : x ∈ A f (L e' p' q') ((1 / 2) ^ p') ((1 / 2) ^ e') := (hn e' p' q' hp' hq').2.1
      exact norm_sub_le_of_mem_A hc P P (A_mono _ _ I I1) (A_mono _ _ I I2)
    calc
      ‖L e p q - L e' p' q'‖ =
          ‖L e p q - L e p r + (L e p r - L e' p' r) + (L e' p' r - L e' p' q')‖ := by
        congr 1; abel
      _ ≤ ‖L e p q - L e p r‖ + ‖L e p r - L e' p' r‖ + ‖L e' p' r - L e' p' q'‖ :=
        norm_add₃_le
      _ ≤ 4 * ‖c‖ * (1 / 2) ^ e + 4 * ‖c‖ * (1 / 2) ^ e + 4 * ‖c‖ * (1 / 2) ^ e := by gcongr
      _ = 12 * ‖c‖ * (1 / 2) ^ e := by ring
  /- For definiteness, use `L0 e = L e (n e) (n e)`, to have a single sequence. We claim that this
    is a Cauchy sequence. -/
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    hx : Membership.mem (FDerivMeasurableAux.D f K) x
    n : Nat → Nat
    L : Nat → Nat → Nat → ContinuousLinearMap (RingHom.id 𝕜) E F
    hn : ∀ (e p q : Nat), LE.le (n e) p → LE.le (n e) q → And (Membership.mem K (L …
    M : ∀ (e p q e' p' q' : Nat), LE.le (n e) p → LE.le (n e) q → LE.le (n e') p'  …
    ⊢ Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.mem  …
  -/
  let L0 : ℕ → E →L[𝕜] F := fun e => L e (n e) (n e)
  have : CauchySeq L0 := by
    rw [Metric.cauchySeq_iff']
    intro ε εpos
    obtain ⟨e, he⟩ : ∃ e : ℕ, (1 / 2) ^ e < ε / (12 * ‖c‖) :=
      exists_pow_lt_of_lt_one (by positivity) (by norm_num)
    refine ⟨e, fun e' he' => ?_⟩
    rw [dist_comm, dist_eq_norm]
    calc
      ‖L0 e - L0 e'‖ ≤ 12 * ‖c‖ * (1 / 2) ^ e := M _ _ _ _ _ _ le_rfl le_rfl le_rfl le_rfl he'
      _ < 12 * ‖c‖ * (ε / (12 * ‖c‖)) := by gcongr
      _ = ε := by field_simp
  -- As it is Cauchy, the sequence `L0` converges, to a limit `f'` in `K`.
  obtain ⟨f', f'K, hf'⟩ : ∃ f' ∈ K, Tendsto L0 atTop (𝓝 f') :=
    cauchySeq_tendsto_of_isComplete hK (fun e => (hn e (n e) (n e) le_rfl le_rfl).1) this
  have Lf' : ∀ e p, n e ≤ p → ‖L e (n e) p - f'‖ ≤ 12 * ‖c‖ * (1 / 2) ^ e := by
    intro e p hp
    apply le_of_tendsto (tendsto_const_nhds.sub hf').norm
    rw [eventually_atTop]
    exact ⟨e, fun e' he' => M _ _ _ _ _ _ le_rfl hp le_rfl le_rfl he'⟩
  -- Let us show that `f` has derivative `f'` at `x`.
  have : HasFDerivAt f f' x := by
    simp only [hasFDerivAt_iff_isLittleO_nhds_zero, isLittleO_iff]
    /- to get an approximation with a precision `ε`, we will replace `f` with `L e (n e) m` for
      some large enough `e` (yielding a small error by uniform approximation). As one can vary `m`,
      this makes it possible to cover all scales, and thus to obtain a good linear approximation in
      the whole ball of radius `(1/2)^(n e)`. -/
    intro ε εpos
    have pos : 0 < 4 + 12 * ‖c‖ := by positivity
    obtain ⟨e, he⟩ : ∃ e : ℕ, (1 / 2) ^ e < ε / (4 + 12 * ‖c‖) :=
      exists_pow_lt_of_lt_one (div_pos εpos pos) (by norm_num)
    rw [eventually_nhds_iff_ball]
    refine ⟨(1 / 2) ^ (n e + 1), P, fun y hy => ?_⟩
    -- We need to show that `f (x + y) - f x - f' y` is small. For this, we will work at scale
    -- `k` where `k` is chosen with `‖y‖ ∼ 2 ^ (-k)`.
    by_cases y_pos : y = 0
    · simp [y_pos]
    have yzero : 0 < ‖y‖ := norm_pos_iff.mpr y_pos
    have y_lt : ‖y‖ < (1 / 2) ^ (n e + 1) := by simpa using mem_ball_iff_norm.1 hy
    have yone : ‖y‖ ≤ 1 := le_trans y_lt.le (pow_le_one₀ (by norm_num) (by norm_num))
    -- define the scale `k`.
    obtain ⟨k, hk, h'k⟩ : ∃ k : ℕ, (1 / 2) ^ (k + 1) < ‖y‖ ∧ ‖y‖ ≤ (1 / 2) ^ k :=
      exists_nat_pow_near_of_lt_one yzero yone (by norm_num : (0 : ℝ) < 1 / 2)
        (by norm_num : (1 : ℝ) / 2 < 1)
    -- the scale is large enough (as `y` is small enough)
    have k_gt : n e < k := by
      have : ((1 : ℝ) / 2) ^ (k + 1) < (1 / 2) ^ (n e + 1) := lt_trans hk y_lt
      rw [pow_lt_pow_iff_right_of_lt_one₀ (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num)] at this
      omega
    set m := k - 1
    have m_ge : n e ≤ m := Nat.le_sub_one_of_lt k_gt
    have km : k = m + 1 := (Nat.succ_pred_eq_of_pos (lt_of_le_of_lt (zero_le _) k_gt)).symm
    rw [km] at hk h'k
    -- `f` is well approximated by `L e (n e) k` at the relevant scale
    -- (in fact, we use `m = k - 1` instead of `k` because of the precise definition of `A`).
    have J1 : ‖f (x + y) - f x - L e (n e) m (x + y - x)‖ ≤ (1 / 2) ^ e * (1 / 2) ^ m := by
      apply le_of_mem_A (hn e (n e) m le_rfl m_ge).2.2
      · simp only [mem_closedBall, dist_self]
        positivity
      · simpa only [dist_eq_norm, add_sub_cancel_left, mem_closedBall, pow_succ, mul_one_div] using
          h'k
    have J2 : ‖f (x + y) - f x - L e (n e) m y‖ ≤ 4 * (1 / 2) ^ e * ‖y‖ :=
      calc
        ‖f (x + y) - f x - L e (n e) m y‖ ≤ (1 / 2) ^ e * (1 / 2) ^ m := by
          simpa only [add_sub_cancel_left] using J1
        _ = 4 * (1 / 2) ^ e * (1 / 2) ^ (m + 2) := by field_simp; ring
        _ ≤ 4 * (1 / 2) ^ e * ‖y‖ := by gcongr
    -- use the previous estimates to see that `f (x + y) - f x - f' y` is small.
    calc
      ‖f (x + y) - f x - f' y‖ = ‖f (x + y) - f x - L e (n e) m y + (L e (n e) m - f') y‖ :=
        congr_arg _ (by simp)
      _ ≤ 4 * (1 / 2) ^ e * ‖y‖ + 12 * ‖c‖ * (1 / 2) ^ e * ‖y‖ :=
        norm_add_le_of_le J2 <| (le_opNorm _ _).trans <| by gcongr; exact Lf' _ _ m_ge
      _ = (4 + 12 * ‖c‖) * ‖y‖ * (1 / 2) ^ e := by ring
      _ ≤ (4 + 12 * ‖c‖) * ‖y‖ * (ε / (4 + 12 * ‖c‖)) := by gcongr
      _ = ε * ‖y‖ := by field_simp [ne_of_gt pos]; ring
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    hx : Membership.mem (FDerivMeasurableAux.D f K) x
    n : Nat → Nat
    L : Nat → Nat → Nat → ContinuousLinearMap (RingHom.id 𝕜) E F
    hn : ∀ (e p q : Nat), LE.le (n e) p → LE.le (n e) q → And (Membership.mem K (L …
    M : ∀ (e p q e' p' q' : Nat), LE.le (n e) p → LE.le (n e) q → LE.le (n e') p'  …
    L0 : Nat → ContinuousLinearMap (RingHom.id 𝕜) E F := fun e => L e (n e) (n e)
    this✝ : CauchySeq L0
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f'K : Membership.mem K f'
    hf' : Filter.Tendsto L0 Filter.atTop (nhds f')
    Lf' : ∀ (e p : Nat), LE.le (n e) p → LE.le (Norm.norm (HSub.hSub (L e (n e) p) …
    this : HasFDerivAt f f' x
    ⊢ Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.mem  …
  -/
  rw [← this.fderiv] at f'K
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    hx : Membership.mem (FDerivMeasurableAux.D f K) x
    n : Nat → Nat
    L : Nat → Nat → Nat → ContinuousLinearMap (RingHom.id 𝕜) E F
    hn : ∀ (e p q : Nat), LE.le (n e) p → LE.le (n e) q → And (Membership.mem K (L …
    M : ∀ (e p q e' p' q' : Nat), LE.le (n e) p → LE.le (n e) q → LE.le (n e') p'  …
    L0 : Nat → ContinuousLinearMap (RingHom.id 𝕜) E F := fun e => L e (n e) (n e)
    this✝ : CauchySeq L0
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f'K : Membership.mem K (fderiv 𝕜 f x)
    hf' : Filter.Tendsto L0 Filter.atTop (nhds f')
    Lf' : ∀ (e p : Nat), LE.le (n e) p → LE.le (Norm.norm (HSub.hSub (L e (n e) p) …
    this : HasFDerivAt f f' x
    ⊢ Membership.mem (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.mem  …
  -/
  exact ⟨this.differentiableAt, f'K⟩
  /-
    🎉 no goals
  -/


theorem differentiable_set_eq_D (hK : IsComplete K) :
    { x | DifferentiableAt 𝕜 f x ∧ fderiv 𝕜 f x ∈ K } = D f K :=
  Subset.antisymm (differentiable_set_subset_D _) (D_subset_differentiable_set hK)


/-- The set of differentiability points of a function, with derivative in a given complete set,
is Borel-measurable. -/
theorem measurableSet_of_differentiableAt_of_isComplete {K : Set (E →L[𝕜] F)} (hK : IsComplete K) :
    MeasurableSet { x | DifferentiableAt 𝕜 f x ∧ fderiv 𝕜 f x ∈ K } := by
  -- Porting note: was
  -- simp [differentiable_set_eq_D K hK, D, isOpen_B.measurableSet, MeasurableSet.iInter,
  --   MeasurableSet.iUnion]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    f : E → F
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    ⊢ MeasurableSet (setOf fun x => And (DifferentiableAt 𝕜 f x) (Membership.mem K …
  -/
  simp only [D, differentiable_set_eq_D K hK]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    f : E → F
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    ⊢ MeasurableSet (Set.iInter fun e => Set.iUnion fun n => Set.iInter fun p => S …
  -/
  repeat apply_rules [MeasurableSet.iUnion, MeasurableSet.iInter] <;> intro
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    f : E → F
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    b✝⁵ b✝⁴ b✝³ : Nat
    b✝² : GE.ge b✝³ b✝⁴
    b✝¹ : Nat
    b✝ : GE.ge b✝¹ b✝⁴
    ⊢ MeasurableSet (FDerivMeasurableAux.B f K (HPow.hPow (1 / 2) b✝³) (HPow.hPow  …
  -/
  exact isOpen_B.measurableSet
  /-
    🎉 no goals
  -/


/-- The set of differentiability points of a function taking values in a complete space is
Borel-measurable. -/
theorem measurableSet_of_differentiableAt : MeasurableSet { x | DifferentiableAt 𝕜 f x } := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    f : E → F
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    ⊢ MeasurableSet (setOf fun x => DifferentiableAt 𝕜 f x)
  -/
  have : IsComplete (univ : Set (E →L[𝕜] F)) := complete_univ
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    f : E → F
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    this : IsComplete Set.univ
    ⊢ MeasurableSet (setOf fun x => DifferentiableAt 𝕜 f x)
  -/
  convert measurableSet_of_differentiableAt_of_isComplete 𝕜 f this
  /-
    case h.e'_3.h.e'_2.h.a
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    f : E → F
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    this : IsComplete Set.univ
    x✝ : E
    ⊢ Iff (DifferentiableAt 𝕜 f x✝) (And (DifferentiableAt 𝕜 f x✝) (Membership.mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem measurable_fderiv : Measurable (fderiv 𝕜 f) := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    f : E → F
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    ⊢ Measurable (fderiv 𝕜 f)
  -/
  refine measurable_of_isClosed fun s hs => ?_
  have :
    fderiv 𝕜 f ⁻¹' s =
      { x | DifferentiableAt 𝕜 f x ∧ fderiv 𝕜 f x ∈ s } ∪
        { x | ¬DifferentiableAt 𝕜 f x } ∩ { _x | (0 : E →L[𝕜] F) ∈ s } :=
    Set.ext fun x => mem_preimage.trans fderiv_mem_iff
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    f : E → F
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    s : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hs : IsClosed s
    this : Eq (Set.preimage (fderiv 𝕜 f) s) (Union.union (setOf fun x => And (Diff …
    ⊢ MeasurableSet (Set.preimage (fderiv 𝕜 f) s)
  -/
  rw [this]
  exact
    (measurableSet_of_differentiableAt_of_isComplete _ _ hs.isComplete).union
      ((measurableSet_of_differentiableAt _ _).compl.inter (MeasurableSet.const _))


@[measurability, fun_prop]
theorem measurable_fderiv_apply_const [MeasurableSpace F] [BorelSpace F] (y : E) :
    Measurable fun x => fderiv 𝕜 f x y :=
  (ContinuousLinearMap.measurable_apply y).comp (measurable_fderiv 𝕜 f)


@[measurability, fun_prop]
theorem measurable_deriv [MeasurableSpace 𝕜] [OpensMeasurableSpace 𝕜] [MeasurableSpace F]
    [BorelSpace F] (f : 𝕜 → F) : Measurable (deriv f) := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    inst✝⁴ : CompleteSpace F
    inst✝³ : MeasurableSpace 𝕜
    inst✝² : OpensMeasurableSpace 𝕜
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    f : 𝕜 → F
    ⊢ Measurable (deriv f)
  -/
  simpa only [fderiv_deriv] using measurable_fderiv_apply_const 𝕜 f 1
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_deriv [MeasurableSpace 𝕜] [OpensMeasurableSpace 𝕜]
    [h : SecondCountableTopologyEither 𝕜 F] (f : 𝕜 → F) : StronglyMeasurable (deriv f) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace F
    inst✝¹ : MeasurableSpace 𝕜
    inst✝ : OpensMeasurableSpace 𝕜
    h : SecondCountableTopologyEither 𝕜 F
    f : 𝕜 → F
    ⊢ MeasureTheory.StronglyMeasurable (deriv f)
  -/
  borelize F
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace F
    inst✝¹ : MeasurableSpace 𝕜
    inst✝ : OpensMeasurableSpace 𝕜
    h : SecondCountableTopologyEither 𝕜 F
    f : 𝕜 → F
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    ⊢ MeasureTheory.StronglyMeasurable (deriv f)
  -/
  rcases h.out with h𝕜|hF
  · exact stronglyMeasurable_iff_measurable_separable.2
      ⟨measurable_deriv f, isSeparable_range_deriv _⟩
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      inst✝² : CompleteSpace F
      inst✝¹ : MeasurableSpace 𝕜
      inst✝ : OpensMeasurableSpace 𝕜
      h : SecondCountableTopologyEither 𝕜 F
      f : 𝕜 → F
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      hF : SecondCountableTopology F
      ⊢ MeasureTheory.StronglyMeasurable (deriv f)
    -/
  · exact (measurable_deriv f).stronglyMeasurable
    /-
      🎉 no goals
    -/


theorem aemeasurable_deriv [MeasurableSpace 𝕜] [OpensMeasurableSpace 𝕜] [MeasurableSpace F]
    [BorelSpace F] (f : 𝕜 → F) (μ : Measure 𝕜) : AEMeasurable (deriv f) μ :=
  (measurable_deriv f).aemeasurable


theorem aestronglyMeasurable_deriv [MeasurableSpace 𝕜] [OpensMeasurableSpace 𝕜]
    [SecondCountableTopologyEither 𝕜 F] (f : 𝕜 → F) (μ : Measure 𝕜) :
    AEStronglyMeasurable (deriv f) μ :=
  (stronglyMeasurable_deriv f).aestronglyMeasurable


/-- The set `A f L r ε` is the set of points `x` around which the function `f` is well approximated
at scale `r` by the linear map `h ↦ h • L`, up to an error `ε`. We tweak the definition to
make sure that this is open on the right. -/
def A (f : ℝ → F) (L : F) (r ε : ℝ) : Set ℝ :=
  { x | ∃ r' ∈ Ioc (r / 2) r, ∀ᵉ (y ∈ Icc x (x + r')) (z ∈ Icc x (x + r')),
    ‖f z - f y - (z - y) • L‖ ≤ ε * r }


/-- The set `B f K r s ε` is the set of points `x` around which there exists a vector
`L` belonging to `K` (a given set of vectors) such that `h • L` approximates well `f (x + h)`
(up to an error `ε`), simultaneously at scales `r` and `s`. -/
def B (f : ℝ → F) (K : Set F) (r s ε : ℝ) : Set ℝ :=
  ⋃ L ∈ K, A f L r ε ∩ A f L s ε


/-- The set `D f K` is a complicated set constructed using countable intersections and unions. Its
main use is that, when `K` is complete, it is exactly the set of points where `f` is differentiable,
with a derivative in `K`. -/
def D (f : ℝ → F) (K : Set F) : Set ℝ :=
  ⋂ e : ℕ, ⋃ n : ℕ, ⋂ (p ≥ n) (q ≥ n), B f K ((1 / 2) ^ p) ((1 / 2) ^ q) ((1 / 2) ^ e)


theorem A_mem_nhdsGT {L : F} {r ε x : ℝ} (hx : x ∈ A f L r ε) : A f L r ε ∈ 𝓝[>] x := by
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x : Real
    hx : Membership.mem (RightDerivMeasurableAux.A f L r ε) x
    ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (RightDerivMeasurableAux.A f L r ε)
  -/
  rcases hx with ⟨r', rr', hr'⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x r' : Real
    rr' : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (RightDerivMeasurableAux.A f L r ε)
  -/
  obtain ⟨s, s_gt, s_lt⟩ : ∃ s : ℝ, r / 2 < s ∧ s < r' := exists_between rr'.1
  /-
    case intro.intro.intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x r' : Real
    rr' : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (RightDerivMeasurableAux.A f L r ε)
  -/
  have : s ∈ Ioc (r / 2) r := ⟨s_gt, le_of_lt (s_lt.trans_le rr'.2)⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x r' : Real
    rr' : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (RightDerivMeasurableAux.A f L r ε)
  -/
  filter_upwards [Ioo_mem_nhdsGT <| show x < x + r' - s by linarith] with x' hx'
  /-
    case h
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x r' : Real
    rr' : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    x' : Real
    hx' : Membership.mem (Set.Ioo x (HSub.hSub (HAdd.hAdd x r') s)) x'
    ⊢ Membership.mem (RightDerivMeasurableAux.A f L r ε) x'
  -/
  use s, this
  have A : Icc x' (x' + s) ⊆ Icc x (x + r') := by
    apply Icc_subset_Icc hx'.1.le
    linarith [hx'.2]
  /-
    case right
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x r' : Real
    rr' : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    x' : Real
    hx' : Membership.mem (Set.Ioo x (HSub.hSub (HAdd.hAdd x r') s)) x'
    A : HasSubset.Subset (Set.Icc x' (HAdd.hAdd x' s)) (Set.Icc x (HAdd.hAdd x r'))
    ⊢ ∀ (y : Real), Membership.mem (Set.Icc x' (HAdd.hAdd x' s)) y → ∀ (z : Real), …
  -/
  intro y hy z hz
  /-
    case right
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε x r' : Real
    rr' : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    s : Real
    s_gt : LT.lt (HDiv.hDiv r 2) s
    s_lt : LT.lt s r'
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) s
    x' : Real
    hx' : Membership.mem (Set.Ioo x (HSub.hSub (HAdd.hAdd x r') s)) x'
    A : HasSubset.Subset (Set.Icc x' (HAdd.hAdd x' s)) (Set.Icc x (HAdd.hAdd x r'))
    y : Real
    hy : Membership.mem (Set.Icc x' (HAdd.hAdd x' s)) y
    z : Real
    hz : Membership.mem (Set.Icc x' (HAdd.hAdd x' s)) z
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (HSMul.hSMul (HSub.hSub  …
  -/
  exact hr' y (A hy) z (A hz)
  /-
    🎉 no goals
  -/


theorem B_mem_nhdsGT {K : Set F} {r s ε x : ℝ} (hx : x ∈ B f K r s ε) :
    B f K r s ε ∈ 𝓝[>] x := by
  obtain ⟨L, LK, hL₁, hL₂⟩ : ∃ L : F, L ∈ K ∧ x ∈ A f L r ε ∧ x ∈ A f L s ε := by
    simpa only [B, mem_iUnion, mem_inter_iff, exists_prop] using hx
  /-
    case intro.intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    r s ε x : Real
    hx : Membership.mem (RightDerivMeasurableAux.B f K r s ε) x
    L : F
    LK : Membership.mem K L
    hL₁ : Membership.mem (RightDerivMeasurableAux.A f L r ε) x
    hL₂ : Membership.mem (RightDerivMeasurableAux.A f L s ε) x
    ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (RightDerivMeasurableAux.B f K r s …
  -/
  filter_upwards [A_mem_nhdsGT hL₁, A_mem_nhdsGT hL₂] with y hy₁ hy₂
  /-
    case h
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    r s ε x : Real
    hx : Membership.mem (RightDerivMeasurableAux.B f K r s ε) x
    L : F
    LK : Membership.mem K L
    hL₁ : Membership.mem (RightDerivMeasurableAux.A f L r ε) x
    hL₂ : Membership.mem (RightDerivMeasurableAux.A f L s ε) x
    y : Real
    hy₁ : Membership.mem (RightDerivMeasurableAux.A f L r ε) y
    hy₂ : Membership.mem (RightDerivMeasurableAux.A f L s ε) y
    ⊢ Membership.mem (RightDerivMeasurableAux.B f K r s ε) y
  -/
  simp only [B, mem_iUnion, mem_inter_iff, exists_prop]
  /-
    case h
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    r s ε x : Real
    hx : Membership.mem (RightDerivMeasurableAux.B f K r s ε) x
    L : F
    LK : Membership.mem K L
    hL₁ : Membership.mem (RightDerivMeasurableAux.A f L r ε) x
    hL₂ : Membership.mem (RightDerivMeasurableAux.A f L s ε) x
    y : Real
    hy₁ : Membership.mem (RightDerivMeasurableAux.A f L r ε) y
    hy₂ : Membership.mem (RightDerivMeasurableAux.A f L s ε) y
    ⊢ Exists fun i => And (Membership.mem K i) (And (Membership.mem (RightDerivMea …
  -/
  exact ⟨L, LK, hy₁, hy₂⟩
  /-
    🎉 no goals
  -/


theorem measurableSet_B {K : Set F} {r s ε : ℝ} : MeasurableSet (B f K r s ε) :=
  .of_mem_nhdsGT fun _ hx => B_mem_nhdsGT hx


theorem A_mono (L : F) (r : ℝ) {ε δ : ℝ} (h : ε ≤ δ) : A f L r ε ⊆ A f L r δ := by
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε δ : Real
    h : LE.le ε δ
    ⊢ HasSubset.Subset (RightDerivMeasurableAux.A f L r ε) (RightDerivMeasurableAu …
  -/
  rintro x ⟨r', r'r, hr'⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε δ : Real
    h : LE.le ε δ
    x r' : Real
    r'r : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    ⊢ Membership.mem (RightDerivMeasurableAux.A f L r δ) x
  -/
  refine ⟨r', r'r, fun y hy z hz => (hr' y hy z hz).trans (mul_le_mul_of_nonneg_right h ?_)⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    L : F
    r ε δ : Real
    h : LE.le ε δ
    x r' : Real
    r'r : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    y : Real
    hy : Membership.mem (Set.Icc x (HAdd.hAdd x r')) y
    z : Real
    hz : Membership.mem (Set.Icc x (HAdd.hAdd x r')) z
    ⊢ LE.le 0 r
  -/
  linarith [hy.1, hy.2, r'r.2]
  /-
    🎉 no goals
  -/


theorem le_of_mem_A {r ε : ℝ} {L : F} {x : ℝ} (hx : x ∈ A f L r ε) {y z : ℝ}
    (hy : y ∈ Icc x (x + r / 2)) (hz : z ∈ Icc x (x + r / 2)) :
  ‖f z - f y - (z - y) • L‖ ≤ ε * r := by
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    r ε : Real
    L : F
    x : Real
    hx : Membership.mem (RightDerivMeasurableAux.A f L r ε) x
    y z : Real
    hy : Membership.mem (Set.Icc x (HAdd.hAdd x (HDiv.hDiv r 2))) y
    hz : Membership.mem (Set.Icc x (HAdd.hAdd x (HDiv.hDiv r 2))) z
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (HSMul.hSMul (HSub.hSub  …
  -/
  rcases hx with ⟨r', r'mem, hr'⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    r ε : Real
    L : F
    x y z : Real
    hy : Membership.mem (Set.Icc x (HAdd.hAdd x (HDiv.hDiv r 2))) y
    hz : Membership.mem (Set.Icc x (HAdd.hAdd x (HDiv.hDiv r 2))) z
    r' : Real
    r'mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (HSMul.hSMul (HSub.hSub  …
  -/
  have A : x + r / 2 ≤ x + r' := by linarith [r'mem.1]
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    r ε : Real
    L : F
    x y z : Real
    hy : Membership.mem (Set.Icc x (HAdd.hAdd x (HDiv.hDiv r 2))) y
    hz : Membership.mem (Set.Icc x (HAdd.hAdd x (HDiv.hDiv r 2))) z
    r' : Real
    r'mem : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r'
    hr' : ∀ (y : Real), Membership.mem (Set.Icc x (HAdd.hAdd x r')) y → ∀ (z : Rea …
    A : LE.le (HAdd.hAdd x (HDiv.hDiv r 2)) (HAdd.hAdd x r')
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f z) (f y)) (HSMul.hSMul (HSub.hSub  …
  -/
  exact hr' _ ((Icc_subset_Icc le_rfl A) hy) _ ((Icc_subset_Icc le_rfl A) hz)
  /-
    🎉 no goals
  -/


theorem mem_A_of_differentiable {ε : ℝ} (hε : 0 < ε) {x : ℝ}
    (hx : DifferentiableWithinAt ℝ f (Ici x) x) :
    ∃ R > 0, ∀ r ∈ Ioo (0 : ℝ) R, x ∈ A f (derivWithin f (Ici x) x) r ε := by
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    ε : Real
    hε : LT.lt 0 ε
    x : Real
    hx : DifferentiableWithinAt Real f (Set.Ici x) x
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (r : Real), Membership.mem (Set.Ioo 0 R)  …
  -/
  have := hx.hasDerivWithinAt
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    ε : Real
    hε : LT.lt 0 ε
    x : Real
    hx : DifferentiableWithinAt Real f (Set.Ici x) x
    this : HasDerivWithinAt f (derivWithin f (Set.Ici x) x) (Set.Ici x) x
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (r : Real), Membership.mem (Set.Ioo 0 R)  …
  -/
  simp_rw [hasDerivWithinAt_iff_isLittleO, isLittleO_iff] at this
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    ε : Real
    hε : LT.lt 0 ε
    x : Real
    hx : DifferentiableWithinAt Real f (Set.Ici x) x
    this : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.nor …
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (r : Real), Membership.mem (Set.Ioo 0 R)  …
  -/
  rcases mem_nhdsGE_iff_exists_Ico_subset.1 (this (half_pos hε)) with ⟨m, xm, hm⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    ε : Real
    hε : LT.lt 0 ε
    x : Real
    hx : DifferentiableWithinAt Real f (Set.Ici x) x
    this : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.nor …
    m : Real
    xm : Membership.mem (Set.Ioi x) m
    hm : HasSubset.Subset (Set.Ico x m) (setOf fun x_1 => (fun x_2 => LE.le (Norm. …
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (r : Real), Membership.mem (Set.Ioo 0 R)  …
  -/
  refine ⟨m - x, by linarith [show x < m from xm], fun r hr => ?_⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    ε : Real
    hε : LT.lt 0 ε
    x : Real
    hx : DifferentiableWithinAt Real f (Set.Ici x) x
    this : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.nor …
    m : Real
    xm : Membership.mem (Set.Ioi x) m
    hm : HasSubset.Subset (Set.Ico x m) (setOf fun x_1 => (fun x_2 => LE.le (Norm. …
    r : Real
    hr : Membership.mem (Set.Ioo 0 (HSub.hSub m x)) r
    ⊢ Membership.mem (RightDerivMeasurableAux.A f (derivWithin f (Set.Ici x) x) r  …
  -/
  have : r ∈ Ioc (r / 2) r := ⟨half_lt_self hr.1, le_rfl⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    ε : Real
    hε : LT.lt 0 ε
    x : Real
    hx : DifferentiableWithinAt Real f (Set.Ici x) x
    this✝ : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.no …
    m : Real
    xm : Membership.mem (Set.Ioi x) m
    hm : HasSubset.Subset (Set.Ico x m) (setOf fun x_1 => (fun x_2 => LE.le (Norm. …
    r : Real
    hr : Membership.mem (Set.Ioo 0 (HSub.hSub m x)) r
    this : Membership.mem (Set.Ioc (HDiv.hDiv r 2) r) r
    ⊢ Membership.mem (RightDerivMeasurableAux.A f (derivWithin f (Set.Ici x) x) r  …
  -/
  refine ⟨r, this, fun y hy z hz => ?_⟩
  calc
    ‖f z - f y - (z - y) • derivWithin f (Ici x) x‖ =
        ‖f z - f x - (z - x) • derivWithin f (Ici x) x -
            (f y - f x - (y - x) • derivWithin f (Ici x) x)‖ := by
      congr 1; simp only [sub_smul]; abel
    _ ≤
        ‖f z - f x - (z - x) • derivWithin f (Ici x) x‖ +
          ‖f y - f x - (y - x) • derivWithin f (Ici x) x‖ :=
      (norm_sub_le _ _)
    _ ≤ ε / 2 * ‖z - x‖ + ε / 2 * ‖y - x‖ :=
      (add_le_add (hm ⟨hz.1, hz.2.trans_lt (by linarith [hr.2])⟩)
        (hm ⟨hy.1, hy.2.trans_lt (by linarith [hr.2])⟩))
    _ ≤ ε / 2 * r + ε / 2 * r := by
      gcongr
      · rw [Real.norm_of_nonneg] <;> linarith [hz.1, hz.2]
      · rw [Real.norm_of_nonneg] <;> linarith [hy.1, hy.2]
    _ = ε * r := by ring


theorem norm_sub_le_of_mem_A {r x : ℝ} (hr : 0 < r) (ε : ℝ) {L₁ L₂ : F} (h₁ : x ∈ A f L₁ r ε)
    (h₂ : x ∈ A f L₂ r ε) : ‖L₁ - L₂‖ ≤ 4 * ε := by
  suffices H : ‖(r / 2) • (L₁ - L₂)‖ ≤ r / 2 * (4 * ε) by
    rwa [norm_smul, Real.norm_of_nonneg (half_pos hr).le, mul_le_mul_left (half_pos hr)] at H
  calc
    ‖(r / 2) • (L₁ - L₂)‖ =
        ‖f (x + r / 2) - f x - (x + r / 2 - x) • L₂ -
            (f (x + r / 2) - f x - (x + r / 2 - x) • L₁)‖ := by
      simp [smul_sub]
    _ ≤ ‖f (x + r / 2) - f x - (x + r / 2 - x) • L₂‖ +
          ‖f (x + r / 2) - f x - (x + r / 2 - x) • L₁‖ :=
      norm_sub_le _ _
    _ ≤ ε * r + ε * r := by
      apply add_le_add
      · apply le_of_mem_A h₂ <;> simp [(half_pos hr).le]
      · apply le_of_mem_A h₁ <;> simp [(half_pos hr).le]
    _ = r / 2 * (4 * ε) := by ring


/-- Easy inclusion: a differentiability point with derivative in `K` belongs to `D f K`. -/
theorem differentiable_set_subset_D :
    { x | DifferentiableWithinAt ℝ f (Ici x) x ∧ derivWithin f (Ici x) x ∈ K } ⊆ D f K := by
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    ⊢ HasSubset.Subset (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ici …
  -/
  intro x hx
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    x : Real
    hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
    ⊢ Membership.mem (RightDerivMeasurableAux.D f K) x
  -/
  rw [D, mem_iInter]
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    x : Real
    hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
    ⊢ ∀ (i : Nat), Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iIn …
  -/
  intro e
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    x : Real
    hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
    e : Nat
    ⊢ Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iInter fun h =>  …
  -/
  have : (0 : ℝ) < (1 / 2) ^ e := pow_pos (by norm_num) _
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    x : Real
    hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
    e : Nat
    this : LT.lt 0 (HPow.hPow (1 / 2) e)
    ⊢ Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iInter fun h =>  …
  -/
  rcases mem_A_of_differentiable this hx.1 with ⟨R, R_pos, hR⟩
  obtain ⟨n, hn⟩ : ∃ n : ℕ, (1 / 2) ^ n < R :=
    exists_pow_lt_of_lt_one R_pos (by norm_num : (1 : ℝ) / 2 < 1)
  /-
    case intro.intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    x : Real
    hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
    e : Nat
    this : LT.lt 0 (HPow.hPow (1 / 2) e)
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (RightDeriv …
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) R
    ⊢ Membership.mem (Set.iUnion fun n => Set.iInter fun p => Set.iInter fun h =>  …
  -/
  simp only [mem_iUnion, mem_iInter, B, mem_inter_iff]
  /-
    case intro.intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    x : Real
    hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
    e : Nat
    this : LT.lt 0 (HPow.hPow (1 / 2) e)
    R : Real
    R_pos : GT.gt R 0
    hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (RightDeriv …
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) R
    ⊢ Exists fun i => ∀ (i_1 : Nat), GE.ge i_1 i → ∀ (i_3 : Nat), GE.ge i_3 i → Ex …
  -/
  refine ⟨n, fun p hp q hq => ⟨derivWithin f (Ici x) x, hx.2, ⟨?_, ?_⟩⟩⟩ <;>
      /-
        case intro.intro.intro.refine_1
        F : Type u_1
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : Real → F
        K : Set F
        x : Real
        hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
        e : Nat
        this : LT.lt 0 (HPow.hPow (1 / 2) e)
        R : Real
        R_pos : GT.gt R 0
        hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (RightDeriv …
        n : Nat
        hn : LT.lt (HPow.hPow (1 / 2) n) R
        p : Nat
        hp : GE.ge p n
        q : Nat
        hq : GE.ge q n
        ⊢ Membership.mem (RightDerivMeasurableAux.A f (derivWithin f (Set.Ici x) x) (H …
      -/
      /-
        case intro.intro.intro.refine_1
        F : Type u_1
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : Real → F
        K : Set F
        x : Real
        hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
        e : Nat
        this : LT.lt 0 (HPow.hPow (1 / 2) e)
        R : Real
        R_pos : GT.gt R 0
        hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (RightDeriv …
        n : Nat
        hn : LT.lt (HPow.hPow (1 / 2) n) R
        p : Nat
        hp : GE.ge p n
        q : Nat
        hq : GE.ge q n
        ⊢ LE.le (HPow.hPow (1 / 2) p) (HPow.hPow (1 / 2) n)
      -/
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_2
        F : Type u_1
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : Real → F
        K : Set F
        x : Real
        hx : Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ic …
        e : Nat
        this : LT.lt 0 (HPow.hPow (1 / 2) e)
        R : Real
        R_pos : GT.gt R 0
        hR : ∀ (r : Real), Membership.mem (Set.Ioo 0 R) r → Membership.mem (RightDeriv …
        n : Nat
        hn : LT.lt (HPow.hPow (1 / 2) n) R
        p : Nat
        hp : GE.ge p n
        q : Nat
        hq : GE.ge q n
        ⊢ LE.le (HPow.hPow (1 / 2) q) (HPow.hPow (1 / 2) n)
      -/
      exact pow_le_pow_of_le_one (by norm_num) (by norm_num) (by assumption)
      /-
        🎉 no goals
      -/


/-- Harder inclusion: at a point in `D f K`, the function `f` has a derivative, in `K`. -/
theorem D_subset_differentiable_set {K : Set F} (hK : IsComplete K) :
    D f K ⊆ { x | DifferentiableWithinAt ℝ f (Ici x) x ∧ derivWithin f (Ici x) x ∈ K } := by
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    ⊢ HasSubset.Subset (RightDerivMeasurableAux.D f K) (setOf fun x => And (Differ …
  -/
  have P : ∀ {n : ℕ}, (0 : ℝ) < (1 / 2) ^ n := fun {n} => pow_pos (by norm_num) n
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    ⊢ HasSubset.Subset (RightDerivMeasurableAux.D f K) (setOf fun x => And (Differ …
  -/
  intro x hx
  have :
    ∀ e : ℕ, ∃ n : ℕ, ∀ p q, n ≤ p → n ≤ q →
      ∃ L ∈ K, x ∈ A f L ((1 / 2) ^ p) ((1 / 2) ^ e) ∩ A f L ((1 / 2) ^ q) ((1 / 2) ^ e) := by
    intro e
    have := mem_iInter.1 hx e
    rcases mem_iUnion.1 this with ⟨n, hn⟩
    refine ⟨n, fun p q hp hq => ?_⟩
    simp only [mem_iInter] at hn
    rcases mem_iUnion.1 (hn p hp q hq) with ⟨L, hL⟩
    exact ⟨L, exists_prop.mp <| mem_iUnion.1 hL⟩
  /- Recast the assumptions: for each `e`, there exist `n e` and linear maps `L e p q` in `K`
    such that, for `p, q ≥ n e`, then `f` is well approximated by `L e p q` at scale `2 ^ (-p)` and
    `2 ^ (-q)`, with an error `2 ^ (-e)`. -/
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    x : Real
    hx : Membership.mem (RightDerivMeasurableAux.D f K) x
    this : ∀ (e : Nat), Exists fun n => ∀ (p q : Nat), LE.le n p → LE.le n q → Exi …
    ⊢ Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ici x …
  -/
  choose! n L hn using this
  /- All the operators `L e p q` that show up are close to each other. To prove this, we argue
      that `L e p q` is close to `L e p r` (where `r` is large enough), as both approximate `f` at
      scale `2 ^(- p)`. And `L e p r` is close to `L e' p' r` as both approximate `f` at scale
      `2 ^ (- r)`. And `L e' p' r` is close to `L e' p' q'` as both approximate `f` at scale
      `2 ^ (- p')`. -/
  have M :
    ∀ e p q e' p' q',
      n e ≤ p →
        n e ≤ q → n e' ≤ p' → n e' ≤ q' → e ≤ e' → ‖L e p q - L e' p' q'‖ ≤ 12 * (1 / 2) ^ e := by
    intro e p q e' p' q' hp hq hp' hq' he'
    let r := max (n e) (n e')
    have I : ((1 : ℝ) / 2) ^ e' ≤ (1 / 2) ^ e :=
      pow_le_pow_of_le_one (by norm_num) (by norm_num) he'
    have J1 : ‖L e p q - L e p r‖ ≤ 4 * (1 / 2) ^ e := by
      have I1 : x ∈ A f (L e p q) ((1 / 2) ^ p) ((1 / 2) ^ e) := (hn e p q hp hq).2.1
      have I2 : x ∈ A f (L e p r) ((1 / 2) ^ p) ((1 / 2) ^ e) := (hn e p r hp (le_max_left _ _)).2.1
      exact norm_sub_le_of_mem_A P _ I1 I2
    have J2 : ‖L e p r - L e' p' r‖ ≤ 4 * (1 / 2) ^ e := by
      have I1 : x ∈ A f (L e p r) ((1 / 2) ^ r) ((1 / 2) ^ e) := (hn e p r hp (le_max_left _ _)).2.2
      have I2 : x ∈ A f (L e' p' r) ((1 / 2) ^ r) ((1 / 2) ^ e') :=
        (hn e' p' r hp' (le_max_right _ _)).2.2
      exact norm_sub_le_of_mem_A P _ I1 (A_mono _ _ I I2)
    have J3 : ‖L e' p' r - L e' p' q'‖ ≤ 4 * (1 / 2) ^ e := by
      have I1 : x ∈ A f (L e' p' r) ((1 / 2) ^ p') ((1 / 2) ^ e') :=
        (hn e' p' r hp' (le_max_right _ _)).2.1
      have I2 : x ∈ A f (L e' p' q') ((1 / 2) ^ p') ((1 / 2) ^ e') := (hn e' p' q' hp' hq').2.1
      exact norm_sub_le_of_mem_A P _ (A_mono _ _ I I1) (A_mono _ _ I I2)
    calc
      ‖L e p q - L e' p' q'‖ =
          ‖L e p q - L e p r + (L e p r - L e' p' r) + (L e' p' r - L e' p' q')‖ := by
        congr 1; abel
      _ ≤ ‖L e p q - L e p r‖ + ‖L e p r - L e' p' r‖ + ‖L e' p' r - L e' p' q'‖ :=
        (le_trans (norm_add_le _ _) (add_le_add_right (norm_add_le _ _) _))
      _ ≤ 4 * (1 / 2) ^ e + 4 * (1 / 2) ^ e + 4 * (1 / 2) ^ e := by gcongr
      _ = 12 * (1 / 2) ^ e := by ring

  /- For definiteness, use `L0 e = L e (n e) (n e)`, to have a single sequence. We claim that this
    is a Cauchy sequence. -/
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    x : Real
    hx : Membership.mem (RightDerivMeasurableAux.D f K) x
    n : Nat → Nat
    L : Nat → Nat → Nat → F
    hn : ∀ (e p q : Nat), LE.le (n e) p → LE.le (n e) q → And (Membership.mem K (L …
    M : ∀ (e p q e' p' q' : Nat), LE.le (n e) p → LE.le (n e) q → LE.le (n e') p'  …
    ⊢ Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ici x …
  -/
  let L0 : ℕ → F := fun e => L e (n e) (n e)
  have : CauchySeq L0 := by
    rw [Metric.cauchySeq_iff']
    intro ε εpos
    obtain ⟨e, he⟩ : ∃ e : ℕ, (1 / 2) ^ e < ε / 12 :=
      exists_pow_lt_of_lt_one (div_pos εpos (by norm_num)) (by norm_num)
    refine ⟨e, fun e' he' => ?_⟩
    rw [dist_comm, dist_eq_norm]
    calc
      ‖L0 e - L0 e'‖ ≤ 12 * (1 / 2) ^ e := M _ _ _ _ _ _ le_rfl le_rfl le_rfl le_rfl he'
      _ < 12 * (ε / 12) := mul_lt_mul' le_rfl he (le_of_lt P) (by norm_num)
      _ = ε := by field_simp [(by norm_num : (12 : ℝ) ≠ 0)]

  -- As it is Cauchy, the sequence `L0` converges, to a limit `f'` in `K`.
  obtain ⟨f', f'K, hf'⟩ : ∃ f' ∈ K, Tendsto L0 atTop (𝓝 f') :=
    cauchySeq_tendsto_of_isComplete hK (fun e => (hn e (n e) (n e) le_rfl le_rfl).1) this
  have Lf' : ∀ e p, n e ≤ p → ‖L e (n e) p - f'‖ ≤ 12 * (1 / 2) ^ e := by
    intro e p hp
    apply le_of_tendsto (tendsto_const_nhds.sub hf').norm
    rw [eventually_atTop]
    exact ⟨e, fun e' he' => M _ _ _ _ _ _ le_rfl hp le_rfl le_rfl he'⟩
  -- Let us show that `f` has right derivative `f'` at `x`.
  have : HasDerivWithinAt f f' (Ici x) x := by
    simp only [hasDerivWithinAt_iff_isLittleO, isLittleO_iff]
    /- to get an approximation with a precision `ε`, we will replace `f` with `L e (n e) m` for
      some large enough `e` (yielding a small error by uniform approximation). As one can vary `m`,
      this makes it possible to cover all scales, and thus to obtain a good linear approximation in
      the whole interval of length `(1/2)^(n e)`. -/
    intro ε εpos
    obtain ⟨e, he⟩ : ∃ e : ℕ, (1 / 2) ^ e < ε / 16 :=
      exists_pow_lt_of_lt_one (div_pos εpos (by norm_num)) (by norm_num)
    filter_upwards [Icc_mem_nhdsGE <| show x < x + (1 / 2) ^ (n e + 1) by simp] with y hy
    -- We need to show that `f y - f x - f' (y - x)` is small. For this, we will work at scale
    -- `k` where `k` is chosen with `‖y - x‖ ∼ 2 ^ (-k)`.
    rcases eq_or_lt_of_le hy.1 with (rfl | xy)
    · simp only [sub_self, zero_smul, norm_zero, mul_zero, le_rfl]
    have yzero : 0 < y - x := sub_pos.2 xy
    have y_le : y - x ≤ (1 / 2) ^ (n e + 1) := by linarith [hy.2]
    have yone : y - x ≤ 1 := le_trans y_le (pow_le_one₀ (by norm_num) (by norm_num))
    -- define the scale `k`.
    obtain ⟨k, hk, h'k⟩ : ∃ k : ℕ, (1 / 2) ^ (k + 1) < y - x ∧ y - x ≤ (1 / 2) ^ k :=
      exists_nat_pow_near_of_lt_one yzero yone (by norm_num : (0 : ℝ) < 1 / 2)
        (by norm_num : (1 : ℝ) / 2 < 1)
    -- the scale is large enough (as `y - x` is small enough)
    have k_gt : n e < k := by
      have : ((1 : ℝ) / 2) ^ (k + 1) < (1 / 2) ^ (n e + 1) := lt_of_lt_of_le hk y_le
      rw [pow_lt_pow_iff_right_of_lt_one₀ (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num)] at this
      omega
    set m := k - 1
    have m_ge : n e ≤ m := Nat.le_sub_one_of_lt k_gt
    have km : k = m + 1 := (Nat.succ_pred_eq_of_pos (lt_of_le_of_lt (zero_le _) k_gt)).symm
    rw [km] at hk h'k
    -- `f` is well approximated by `L e (n e) k` at the relevant scale
    -- (in fact, we use `m = k - 1` instead of `k` because of the precise definition of `A`).
    have J : ‖f y - f x - (y - x) • L e (n e) m‖ ≤ 4 * (1 / 2) ^ e * ‖y - x‖ :=
      calc
        ‖f y - f x - (y - x) • L e (n e) m‖ ≤ (1 / 2) ^ e * (1 / 2) ^ m := by
          apply le_of_mem_A (hn e (n e) m le_rfl m_ge).2.2
          · simp only [one_div, inv_pow, left_mem_Icc, le_add_iff_nonneg_right]
            positivity
          · simp only [pow_add, tsub_le_iff_left] at h'k
            simpa only [hy.1, mem_Icc, true_and, one_div, pow_one] using h'k
        _ = 4 * (1 / 2) ^ e * (1 / 2) ^ (m + 2) := by field_simp; ring
        _ ≤ 4 * (1 / 2) ^ e * (y - x) := by gcongr
        _ = 4 * (1 / 2) ^ e * ‖y - x‖ := by rw [Real.norm_of_nonneg yzero.le]
    calc
      ‖f y - f x - (y - x) • f'‖ =
          ‖f y - f x - (y - x) • L e (n e) m + (y - x) • (L e (n e) m - f')‖ := by
        simp only [smul_sub, sub_add_sub_cancel]
      _ ≤ 4 * (1 / 2) ^ e * ‖y - x‖ + ‖y - x‖ * (12 * (1 / 2) ^ e) :=
        norm_add_le_of_le J <| by rw [norm_smul]; gcongr; exact Lf' _ _ m_ge
      _ = 16 * ‖y - x‖ * (1 / 2) ^ e := by ring
      _ ≤ 16 * ‖y - x‖ * (ε / 16) := by gcongr
      _ = ε * ‖y - x‖ := by ring

  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    x : Real
    hx : Membership.mem (RightDerivMeasurableAux.D f K) x
    n : Nat → Nat
    L : Nat → Nat → Nat → F
    hn : ∀ (e p q : Nat), LE.le (n e) p → LE.le (n e) q → And (Membership.mem K (L …
    M : ∀ (e p q e' p' q' : Nat), LE.le (n e) p → LE.le (n e) q → LE.le (n e') p'  …
    L0 : Nat → F := fun e => L e (n e) (n e)
    this✝ : CauchySeq L0
    f' : F
    f'K : Membership.mem K f'
    hf' : Filter.Tendsto L0 Filter.atTop (nhds f')
    Lf' : ∀ (e p : Nat), LE.le (n e) p → LE.le (Norm.norm (HSub.hSub (L e (n e) p) …
    this : HasDerivWithinAt f f' (Set.Ici x) x
    ⊢ Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ici x …
  -/
  rw [← this.derivWithin (uniqueDiffOn_Ici x x Set.left_mem_Ici)] at f'K
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    P : ∀ {n : Nat}, LT.lt 0 (HPow.hPow (1 / 2) n)
    x : Real
    hx : Membership.mem (RightDerivMeasurableAux.D f K) x
    n : Nat → Nat
    L : Nat → Nat → Nat → F
    hn : ∀ (e p q : Nat), LE.le (n e) p → LE.le (n e) q → And (Membership.mem K (L …
    M : ∀ (e p q e' p' q' : Nat), LE.le (n e) p → LE.le (n e) q → LE.le (n e') p'  …
    L0 : Nat → F := fun e => L e (n e) (n e)
    this✝ : CauchySeq L0
    f' : F
    f'K : Membership.mem K (derivWithin f (Set.Ici x) x)
    hf' : Filter.Tendsto L0 Filter.atTop (nhds f')
    Lf' : ∀ (e p : Nat), LE.le (n e) p → LE.le (Norm.norm (HSub.hSub (L e (n e) p) …
    this : HasDerivWithinAt f f' (Set.Ici x) x
    ⊢ Membership.mem (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ici x …
  -/
  exact ⟨this.differentiableWithinAt, f'K⟩
  /-
    🎉 no goals
  -/


theorem differentiable_set_eq_D (hK : IsComplete K) :
    { x | DifferentiableWithinAt ℝ f (Ici x) x ∧ derivWithin f (Ici x) x ∈ K } = D f K :=
  Subset.antisymm (differentiable_set_subset_D _) (D_subset_differentiable_set hK)


/-- The set of right differentiability points of a function, with derivative in a given complete
set, is Borel-measurable. -/
theorem measurableSet_of_differentiableWithinAt_Ici_of_isComplete {K : Set F} (hK : IsComplete K) :
    MeasurableSet { x | DifferentiableWithinAt ℝ f (Ici x) x ∧ derivWithin f (Ici x) x ∈ K } := by
  -- simp [differentiable_set_eq_d K hK, D, measurableSet_b, MeasurableSet.iInter,
  --   MeasurableSet.iUnion]
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    ⊢ MeasurableSet (setOf fun x => And (DifferentiableWithinAt Real f (Set.Ici x) …
  -/
  simp only [differentiable_set_eq_D K hK, D]
  /-
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    ⊢ MeasurableSet (Set.iInter fun e => Set.iUnion fun n => Set.iInter fun p => S …
  -/
  repeat apply_rules [MeasurableSet.iUnion, MeasurableSet.iInter] <;> intro
  /-
    case h
    F : Type u_1
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    K : Set F
    hK : IsComplete K
    b✝⁵ b✝⁴ b✝³ : Nat
    b✝² : GE.ge b✝³ b✝⁴
    b✝¹ : Nat
    b✝ : GE.ge b✝¹ b✝⁴
    ⊢ MeasurableSet (RightDerivMeasurableAux.B f K (HPow.hPow (1 / 2) b✝³) (HPow.h …
  -/
  exact measurableSet_B
  /-
    🎉 no goals
  -/


/-- The set of right differentiability points of a function taking values in a complete space is
Borel-measurable. -/
theorem measurableSet_of_differentiableWithinAt_Ici :
    MeasurableSet { x | DifferentiableWithinAt ℝ f (Ici x) x } := by
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    ⊢ MeasurableSet (setOf fun x => DifferentiableWithinAt Real f (Set.Ici x) x)
  -/
  have : IsComplete (univ : Set F) := complete_univ
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    this : IsComplete Set.univ
    ⊢ MeasurableSet (setOf fun x => DifferentiableWithinAt Real f (Set.Ici x) x)
  -/
  convert measurableSet_of_differentiableWithinAt_Ici_of_isComplete f this
  /-
    case h.e'_3.h.e'_2.h.a
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    this : IsComplete Set.univ
    x✝ : Real
    ⊢ Iff (DifferentiableWithinAt Real f (Set.Ici x✝) x✝) (And (DifferentiableWith …
  -/
  simp
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem measurable_derivWithin_Ici [MeasurableSpace F] [BorelSpace F] :
    Measurable fun x => derivWithin f (Ici x) x := by
  /-
    F : Type u_1
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f : Real → F
    inst✝² : CompleteSpace F
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    ⊢ Measurable fun x => derivWithin f (Set.Ici x) x
  -/
  refine measurable_of_isClosed fun s hs => ?_
  have :
    (fun x => derivWithin f (Ici x) x) ⁻¹' s =
      { x | DifferentiableWithinAt ℝ f (Ici x) x ∧ derivWithin f (Ici x) x ∈ s } ∪
        { x | ¬DifferentiableWithinAt ℝ f (Ici x) x } ∩ { _x | (0 : F) ∈ s } :=
    Set.ext fun x => mem_preimage.trans derivWithin_mem_iff
  /-
    F : Type u_1
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f : Real → F
    inst✝² : CompleteSpace F
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    s : Set F
    hs : IsClosed s
    this : Eq (Set.preimage (fun x => derivWithin f (Set.Ici x) x) s) (Union.union …
    ⊢ MeasurableSet (Set.preimage (fun x => derivWithin f (Set.Ici x) x) s)
  -/
  rw [this]
  exact
    (measurableSet_of_differentiableWithinAt_Ici_of_isComplete _ hs.isComplete).union
      ((measurableSet_of_differentiableWithinAt_Ici _).compl.inter (MeasurableSet.const _))


theorem stronglyMeasurable_derivWithin_Ici :
    StronglyMeasurable (fun x ↦ derivWithin f (Ici x) x) := by
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    ⊢ MeasureTheory.StronglyMeasurable fun x => derivWithin f (Set.Ici x) x
  -/
  borelize F
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    ⊢ MeasureTheory.StronglyMeasurable fun x => derivWithin f (Set.Ici x) x
  -/
  apply stronglyMeasurable_iff_measurable_separable.2 ⟨measurable_derivWithin_Ici f, ?_⟩
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    ⊢ TopologicalSpace.IsSeparable (Set.range fun x => derivWithin f (Set.Ici x) x)
  -/
  obtain ⟨t, t_count, ht⟩ : ∃ t : Set ℝ, t.Countable ∧ Dense t := exists_countable_dense ℝ
  suffices H : range (fun x ↦ derivWithin f (Ici x) x) ⊆ closure (Submodule.span ℝ (f '' t)) from
    IsSeparable.mono (t_count.image f).isSeparable.span.closure H
  /-
    case intro.intro
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    t : Set Real
    t_count : t.Countable
    ht : Dense t
    ⊢ HasSubset.Subset (Set.range fun x => derivWithin f (Set.Ici x) x) (closure ↑ …
  -/
  rintro - ⟨x, rfl⟩
  suffices H' : range (fun y ↦ derivWithin f (Ici x) y) ⊆ closure (Submodule.span ℝ (f '' t)) from
    H' (mem_range_self _)
  /-
    case intro.intro.intro
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    t : Set Real
    t_count : t.Countable
    ht : Dense t
    x : Real
    ⊢ HasSubset.Subset (Set.range fun y => derivWithin f (Set.Ici x) y) (closure ↑ …
  -/
  apply range_derivWithin_subset_closure_span_image
  calc Ici x
    = closure (Ioi x ∩ closure t) := by simp [dense_iff_closure_eq.1 ht]
  _ ⊆ closure (closure (Ioi x ∩ t)) := by
      apply closure_mono
      simpa [inter_comm] using (isOpen_Ioi (a := x)).closure_inter (s := t)
  _ ⊆ closure (Ici x ∩ t) := by
      rw [closure_closure]
      exact closure_mono (inter_subset_inter_left _ Ioi_subset_Ici_self)


theorem aemeasurable_derivWithin_Ici [MeasurableSpace F] [BorelSpace F] (μ : Measure ℝ) :
    AEMeasurable (fun x => derivWithin f (Ici x) x) μ :=
  (measurable_derivWithin_Ici f).aemeasurable


theorem aestronglyMeasurable_derivWithin_Ici (μ : Measure ℝ) :
    AEStronglyMeasurable (fun x => derivWithin f (Ici x) x) μ :=
  (stronglyMeasurable_derivWithin_Ici f).aestronglyMeasurable


/-- The set of right differentiability points of a function taking values in a complete space is
Borel-measurable. -/
theorem measurableSet_of_differentiableWithinAt_Ioi :
    MeasurableSet { x | DifferentiableWithinAt ℝ f (Ioi x) x } := by
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    ⊢ MeasurableSet (setOf fun x => DifferentiableWithinAt Real f (Set.Ioi x) x)
  -/
  simpa [differentiableWithinAt_Ioi_iff_Ici] using measurableSet_of_differentiableWithinAt_Ici f
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem measurable_derivWithin_Ioi [MeasurableSpace F] [BorelSpace F] :
    Measurable fun x => derivWithin f (Ioi x) x := by
  /-
    F : Type u_1
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f : Real → F
    inst✝² : CompleteSpace F
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    ⊢ Measurable fun x => derivWithin f (Set.Ioi x) x
  -/
  simpa [derivWithin_Ioi_eq_Ici] using measurable_derivWithin_Ici f
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_derivWithin_Ioi :
    StronglyMeasurable (fun x ↦ derivWithin f (Ioi x) x) := by
  /-
    F : Type u_1
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    f : Real → F
    inst✝ : CompleteSpace F
    ⊢ MeasureTheory.StronglyMeasurable fun x => derivWithin f (Set.Ioi x) x
  -/
  simpa [derivWithin_Ioi_eq_Ici] using stronglyMeasurable_derivWithin_Ici f
  /-
    🎉 no goals
  -/


theorem aemeasurable_derivWithin_Ioi [MeasurableSpace F] [BorelSpace F] (μ : Measure ℝ) :
    AEMeasurable (fun x => derivWithin f (Ioi x) x) μ :=
  (measurable_derivWithin_Ioi f).aemeasurable


theorem aestronglyMeasurable_derivWithin_Ioi (μ : Measure ℝ) :
    AEStronglyMeasurable (fun x => derivWithin f (Ioi x) x) μ :=
  (stronglyMeasurable_derivWithin_Ioi f).aestronglyMeasurable


lemma isOpen_A_with_param {r s : ℝ} (hf : Continuous f.uncurry) (L : E →L[𝕜] F) :
    IsOpen {p : α × E | p.2 ∈ A (f p.1) L r s} := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ IsOpen (setOf fun p => Membership.mem (FDerivMeasurableAux.A (f p.1) L r s)  …
  -/
  have : ProperSpace E := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this : ProperSpace E
    ⊢ IsOpen (setOf fun p => Membership.mem (FDerivMeasurableAux.A (f p.1) L r s)  …
  -/
  simp only [A, half_lt_self_iff, not_lt, mem_Ioc, mem_ball, map_sub, mem_setOf_eq]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this : ProperSpace E
    ⊢ IsOpen (setOf fun p => Exists fun r' => And (And (LT.lt (HDiv.hDiv r 2) r')  …
  -/
  apply isOpen_iff_mem_nhds.2
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this : ProperSpace E
    ⊢ ∀ (x : Prod α E), Membership.mem (setOf fun p => Exists fun r' => And (And ( …
  -/
  rintro ⟨a, x⟩ ⟨r', ⟨Irr', Ir'r⟩, hr⟩
  /-
    case mk.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ⊢ Membership.mem (nhds { fst := a, snd := x }) (setOf fun p => Exists fun r' = …
  -/
  have ha : Continuous (f a) := hf.uncurry_left a
  /-
    case mk.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    ⊢ Membership.mem (nhds { fst := a, snd := x }) (setOf fun p => Exists fun r' = …
  -/
  rcases exists_between Irr' with ⟨t, hrt, htr'⟩
  /-
    case mk.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    ⊢ Membership.mem (nhds { fst := a, snd := x }) (setOf fun p => Exists fun r' = …
  -/
  rcases exists_between hrt with ⟨t', hrt', ht't⟩
  obtain ⟨b, b_lt, hb⟩ : ∃ b, b < s * r ∧ ∀ y ∈ closedBall x t, ∀ z ∈ closedBall x t,
      ‖f a z - f a y - (L z - L y)‖ ≤ b := by
    have B : Continuous (fun (p : E × E) ↦ ‖f a p.2 - f a p.1 - (L p.2 - L p.1)‖) := by fun_prop
    have C : (closedBall x t ×ˢ closedBall x t).Nonempty := by simp; linarith
    rcases ((isCompact_closedBall x t).prod (isCompact_closedBall x t)).exists_isMaxOn
      C B.continuousOn with ⟨p, pt, hp⟩
    simp only [mem_prod, mem_closedBall] at pt
    refine ⟨‖f a p.2 - f a p.1 - (L p.2 - L p.1)‖,
      hr p.1 (pt.1.trans_lt htr') p.2 (pt.2.trans_lt htr'), fun y hy z hz ↦ ?_⟩
    have D : (y, z) ∈ closedBall x t ×ˢ closedBall x t := mem_prod.2 ⟨hy, hz⟩
    exact hp D
  obtain ⟨ε, εpos, hε⟩ : ∃ ε, 0 < ε ∧ b + 2 * ε < s * r :=
    ⟨(s * r - b) / 3, by linarith, by linarith⟩
  obtain ⟨u, u_open, au, hu⟩ : ∃ u, IsOpen u ∧ a ∈ u ∧ ∀ (p : α × E),
      p.1 ∈ u → p.2 ∈ closedBall x t → dist (f.uncurry p) (f.uncurry (a, p.2)) < ε := by
    have C : Continuous (fun (p : α × E) ↦ f a p.2) := by fun_prop
    have D : ({a} ×ˢ closedBall x t).EqOn f.uncurry (fun p ↦ f a p.2) := by
      rintro ⟨b, y⟩ ⟨hb, -⟩
      simp only [mem_singleton_iff] at hb
      simp [hb]
    obtain ⟨v, v_open, sub_v, hv⟩ : ∃ v, IsOpen v ∧ {a} ×ˢ closedBall x t ⊆ v ∧
        ∀ p ∈ v, dist (Function.uncurry f p) (f a p.2) < ε :=
      Uniform.exists_is_open_mem_uniformity_of_forall_mem_eq (s := {a} ×ˢ closedBall x t)
        (fun p _ ↦ hf.continuousAt) (fun p _ ↦ C.continuousAt) D (dist_mem_uniformity εpos)
    obtain ⟨w, w', w_open, -, sub_w, sub_w', hww'⟩ : ∃ (w : Set α) (w' : Set E),
        IsOpen w ∧ IsOpen w' ∧ {a} ⊆ w ∧ closedBall x t ⊆ w' ∧ w ×ˢ w' ⊆ v :=
      generalized_tube_lemma isCompact_singleton (isCompact_closedBall x t) v_open sub_v
    refine ⟨w, w_open, sub_w rfl, ?_⟩
    rintro ⟨b, y⟩ h hby
    exact hv _ (hww' ⟨h, sub_w' hby⟩)
  have : u ×ˢ ball x (t - t') ∈ 𝓝 (a, x) :=
    prod_mem_nhds (u_open.mem_nhds au) (ball_mem_nhds _ (sub_pos.2 ht't))
  /-
    case mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this✝ : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    t' : Real
    hrt' : LT.lt (HDiv.hDiv r 2) t'
    ht't : LT.lt t' t
    b : Real
    b_lt : LT.lt b (HMul.hMul s r)
    hb : ∀ (y : E), Membership.mem (Metric.closedBall x t) y → ∀ (z : E), Membersh …
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt (HAdd.hAdd b (HMul.hMul 2 ε)) (HMul.hMul s r)
    u : Set α
    u_open : IsOpen u
    au : Membership.mem u a
    hu : ∀ (p : Prod α E), Membership.mem u p.1 → Membership.mem (Metric.closedBal …
    this : Membership.mem (nhds { fst := a, snd := x }) (SProd.sprod u (Metric.bal …
    ⊢ Membership.mem (nhds { fst := a, snd := x }) (setOf fun p => Exists fun r' = …
  -/
  filter_upwards [this]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this✝ : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    t' : Real
    hrt' : LT.lt (HDiv.hDiv r 2) t'
    ht't : LT.lt t' t
    b : Real
    b_lt : LT.lt b (HMul.hMul s r)
    hb : ∀ (y : E), Membership.mem (Metric.closedBall x t) y → ∀ (z : E), Membersh …
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt (HAdd.hAdd b (HMul.hMul 2 ε)) (HMul.hMul s r)
    u : Set α
    u_open : IsOpen u
    au : Membership.mem u a
    hu : ∀ (p : Prod α E), Membership.mem u p.1 → Membership.mem (Metric.closedBal …
    this : Membership.mem (nhds { fst := a, snd := x }) (SProd.sprod u (Metric.bal …
    ⊢ ∀ (a : Prod α E), Membership.mem (SProd.sprod u (Metric.ball x (HSub.hSub t  …
  -/
  rintro ⟨a', x'⟩ ha'x'
  /-
    case h.mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this✝ : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    t' : Real
    hrt' : LT.lt (HDiv.hDiv r 2) t'
    ht't : LT.lt t' t
    b : Real
    b_lt : LT.lt b (HMul.hMul s r)
    hb : ∀ (y : E), Membership.mem (Metric.closedBall x t) y → ∀ (z : E), Membersh …
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt (HAdd.hAdd b (HMul.hMul 2 ε)) (HMul.hMul s r)
    u : Set α
    u_open : IsOpen u
    au : Membership.mem u a
    hu : ∀ (p : Prod α E), Membership.mem u p.1 → Membership.mem (Metric.closedBal …
    this : Membership.mem (nhds { fst := a, snd := x }) (SProd.sprod u (Metric.bal …
    a' : α
    x' : E
    ha'x' : Membership.mem (SProd.sprod u (Metric.ball x (HSub.hSub t t'))) { fst  …
    ⊢ Exists fun r' => And (And (LT.lt (HDiv.hDiv r 2) r') (LE.le r' r)) (∀ (y : E …
  -/
  simp only [mem_prod, mem_ball] at ha'x'
  /-
    case h.mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this✝ : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    t' : Real
    hrt' : LT.lt (HDiv.hDiv r 2) t'
    ht't : LT.lt t' t
    b : Real
    b_lt : LT.lt b (HMul.hMul s r)
    hb : ∀ (y : E), Membership.mem (Metric.closedBall x t) y → ∀ (z : E), Membersh …
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt (HAdd.hAdd b (HMul.hMul 2 ε)) (HMul.hMul s r)
    u : Set α
    u_open : IsOpen u
    au : Membership.mem u a
    hu : ∀ (p : Prod α E), Membership.mem u p.1 → Membership.mem (Metric.closedBal …
    this : Membership.mem (nhds { fst := a, snd := x }) (SProd.sprod u (Metric.bal …
    a' : α
    x' : E
    ha'x' : And (Membership.mem u a') (LT.lt (Dist.dist x' x) (HSub.hSub t t'))
    ⊢ Exists fun r' => And (And (LT.lt (HDiv.hDiv r 2) r') (LE.le r' r)) (∀ (y : E …
  -/
  refine ⟨t', ⟨hrt', ht't.le.trans (htr'.le.trans Ir'r)⟩, fun y hy z hz ↦ ?_⟩
  /-
    case h.mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this✝ : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    t' : Real
    hrt' : LT.lt (HDiv.hDiv r 2) t'
    ht't : LT.lt t' t
    b : Real
    b_lt : LT.lt b (HMul.hMul s r)
    hb : ∀ (y : E), Membership.mem (Metric.closedBall x t) y → ∀ (z : E), Membersh …
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt (HAdd.hAdd b (HMul.hMul 2 ε)) (HMul.hMul s r)
    u : Set α
    u_open : IsOpen u
    au : Membership.mem u a
    hu : ∀ (p : Prod α E), Membership.mem u p.1 → Membership.mem (Metric.closedBal …
    this : Membership.mem (nhds { fst := a, snd := x }) (SProd.sprod u (Metric.bal …
    a' : α
    x' : E
    ha'x' : And (Membership.mem u a') (LT.lt (Dist.dist x' x) (HSub.hSub t t'))
    y : E
    hy : LT.lt (Dist.dist y { fst := a', snd := x' }.2) t'
    z : E
    hz : LT.lt (Dist.dist z { fst := a', snd := x' }.2) t'
    ⊢ LT.lt (Norm.norm (HSub.hSub (HSub.hSub (f { fst := a', snd := x' }.1 z) (f { …
  -/
  have dyx : dist y x ≤ t := by linarith [dist_triangle y x' x]
  /-
    case h.mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s : Real
    hf : Continuous (Function.uncurry f)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    this✝ : ProperSpace E
    a : α
    x : E
    r' : Real
    hr : ∀ (y : E), LT.lt (Dist.dist y { fst := a, snd := x }.2) r' → ∀ (z : E), L …
    Irr' : LT.lt (HDiv.hDiv r 2) r'
    Ir'r : LE.le r' r
    ha : Continuous (f a)
    t : Real
    hrt : LT.lt (HDiv.hDiv r 2) t
    htr' : LT.lt t r'
    t' : Real
    hrt' : LT.lt (HDiv.hDiv r 2) t'
    ht't : LT.lt t' t
    b : Real
    b_lt : LT.lt b (HMul.hMul s r)
    hb : ∀ (y : E), Membership.mem (Metric.closedBall x t) y → ∀ (z : E), Membersh …
    ε : Real
    εpos : LT.lt 0 ε
    hε : LT.lt (HAdd.hAdd b (HMul.hMul 2 ε)) (HMul.hMul s r)
    u : Set α
    u_open : IsOpen u
    au : Membership.mem u a
    hu : ∀ (p : Prod α E), Membership.mem u p.1 → Membership.mem (Metric.closedBal …
    this : Membership.mem (nhds { fst := a, snd := x }) (SProd.sprod u (Metric.bal …
    a' : α
    x' : E
    ha'x' : And (Membership.mem u a') (LT.lt (Dist.dist x' x) (HSub.hSub t t'))
    y : E
    hy : LT.lt (Dist.dist y { fst := a', snd := x' }.2) t'
    z : E
    hz : LT.lt (Dist.dist z { fst := a', snd := x' }.2) t'
    dyx : LE.le (Dist.dist y x) t
    ⊢ LT.lt (Norm.norm (HSub.hSub (HSub.hSub (f { fst := a', snd := x' }.1 z) (f { …
  -/
  have dzx : dist z x ≤ t := by linarith [dist_triangle z x' x]
  calc
  ‖f a' z - f a' y - (L z - L y)‖ =
    ‖(f a' z - f a z) + (f a y - f a' y) + (f a z - f a y - (L z - L y))‖ := by congr; abel
  _ ≤ ‖f a' z - f a z‖ + ‖f a y - f a' y‖ + ‖f a z - f a y - (L z - L y)‖ := norm_add₃_le
  _ ≤ ε + ε + b := by
      gcongr
      · rw [← dist_eq_norm]
        change dist (f.uncurry (a', z)) (f.uncurry (a, z)) ≤ ε
        apply (hu _ _ _).le
        · exact ha'x'.1
        · simp [dzx]
      · rw [← dist_eq_norm']
        change dist (f.uncurry (a', y)) (f.uncurry (a, y)) ≤ ε
        apply (hu _ _ _).le
        · exact ha'x'.1
        · simp [dyx]
      · simp [hb, dyx, dzx]
  _ < s * r := by linarith


lemma isOpen_B_with_param {r s t : ℝ} (hf : Continuous f.uncurry) (K : Set (E →L[𝕜] F)) :
    IsOpen {p : α × E | p.2 ∈ B (f p.1) K r s t} := by
  suffices H : IsOpen (⋃ L ∈ K,
      {p : α × E | p.2 ∈ A (f p.1) L r t ∧ p.2 ∈ A (f p.1) L s t}) by
    convert H; ext p; simp [B]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s t : Real
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    ⊢ IsOpen (Set.iUnion fun L => Set.iUnion fun h => setOf fun p => And (Membersh …
  -/
  refine isOpen_biUnion (fun L _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : LocallyCompactSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝ : TopologicalSpace α
    f : α → E → F
    r s t : Real
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    x✝ : Membership.mem K L
    ⊢ IsOpen (setOf fun p => And (Membership.mem (FDerivMeasurableAux.A (f p.1) L  …
  -/
  exact (isOpen_A_with_param hf L).inter (isOpen_A_with_param hf L)
  /-
    🎉 no goals
  -/


theorem measurableSet_of_differentiableAt_of_isComplete_with_param
    (hf : Continuous f.uncurry) {K : Set (E →L[𝕜] F)} (hK : IsComplete K) :
    MeasurableSet {p : α × E | DifferentiableAt 𝕜 (f p.1) p.2 ∧ fderiv 𝕜 (f p.1) p.2 ∈ K} := by
  have : {p : α × E | DifferentiableAt 𝕜 (f p.1) p.2 ∧ fderiv 𝕜 (f p.1) p.2 ∈ K}
          = {p : α × E | p.2 ∈ D (f p.1) K} := by simp [← differentiable_set_eq_D K hK]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    ⊢ MeasurableSet (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membersh …
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    ⊢ MeasurableSet (setOf fun p => Membership.mem (FDerivMeasurableAux.D (f p.1)  …
  -/
  simp only [D, mem_iInter, mem_iUnion]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    ⊢ MeasurableSet (setOf fun p => ∀ (i : Nat), Exists fun i_1 => ∀ (i_2 : Nat),  …
  -/
  simp only [setOf_forall, setOf_exists]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    ⊢ MeasurableSet (Set.iInter fun i => Set.iUnion fun i_1 => Set.iInter fun i_2  …
  -/
  refine MeasurableSet.iInter (fun _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    x✝ : Nat
    ⊢ MeasurableSet (Set.iUnion fun i => Set.iInter fun i_1 => Set.iInter fun x => …
  -/
  refine MeasurableSet.iUnion (fun _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    x✝¹ x✝ : Nat
    ⊢ MeasurableSet (Set.iInter fun i => Set.iInter fun x => Set.iInter fun i_1 => …
  -/
  refine MeasurableSet.iInter (fun _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    x✝² x✝¹ x✝ : Nat
    ⊢ MeasurableSet (Set.iInter fun x => Set.iInter fun i => Set.iInter fun i_1 => …
  -/
  refine MeasurableSet.iInter (fun _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    x✝³ x✝² x✝¹ : Nat
    x✝ : GE.ge x✝¹ x✝²
    ⊢ MeasurableSet (Set.iInter fun i => Set.iInter fun i_1 => setOf fun x => Memb …
  -/
  refine MeasurableSet.iInter (fun _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    x✝⁴ x✝³ x✝² : Nat
    x✝¹ : GE.ge x✝² x✝³
    x✝ : Nat
    ⊢ MeasurableSet (Set.iInter fun i => setOf fun x => Membership.mem (FDerivMeas …
  -/
  refine MeasurableSet.iInter (fun _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.mem …
    x✝⁵ x✝⁴ x✝³ : Nat
    x✝² : GE.ge x✝³ x✝⁴
    x✝¹ : Nat
    x✝ : GE.ge x✝¹ x✝⁴
    ⊢ MeasurableSet (setOf fun x => Membership.mem (FDerivMeasurableAux.B (f x.1)  …
  -/
  have : ProperSpace E := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁴ : TopologicalSpace α
    f : α → E → F
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    hf : Continuous (Function.uncurry f)
    K : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hK : IsComplete K
    this✝ : Eq (setOf fun p => And (DifferentiableAt 𝕜 (f p.1) p.2) (Membership.me …
    x✝⁵ x✝⁴ x✝³ : Nat
    x✝² : GE.ge x✝³ x✝⁴
    x✝¹ : Nat
    x✝ : GE.ge x✝¹ x✝⁴
    this : ProperSpace E
    ⊢ MeasurableSet (setOf fun x => Membership.mem (FDerivMeasurableAux.B (f x.1)  …
  -/
  exact (isOpen_B_with_param hf K).measurableSet
  /-
    🎉 no goals
  -/


/-- The set of differentiability points of a continuous function depending on a parameter taking
values in a complete space is Borel-measurable. -/
theorem measurableSet_of_differentiableAt_with_param (hf : Continuous f.uncurry) :
    MeasurableSet {p : α × E | DifferentiableAt 𝕜 (f p.1) p.2} := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁵ : TopologicalSpace α
    f : α → E → F
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    hf : Continuous (Function.uncurry f)
    ⊢ MeasurableSet (setOf fun p => DifferentiableAt 𝕜 (f p.1) p.2)
  -/
  have : IsComplete (univ : Set (E →L[𝕜] F)) := complete_univ
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁵ : TopologicalSpace α
    f : α → E → F
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    hf : Continuous (Function.uncurry f)
    this : IsComplete Set.univ
    ⊢ MeasurableSet (setOf fun p => DifferentiableAt 𝕜 (f p.1) p.2)
  -/
  convert measurableSet_of_differentiableAt_of_isComplete_with_param hf this
  /-
    case h.e'_3.h.e'_2.h.a
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁵ : TopologicalSpace α
    f : α → E → F
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    hf : Continuous (Function.uncurry f)
    this : IsComplete Set.univ
    x✝ : Prod α E
    ⊢ Iff (DifferentiableAt 𝕜 (f x✝.1) x✝.2) (And (DifferentiableAt 𝕜 (f x✝.1) x✝. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem measurable_fderiv_with_param (hf : Continuous f.uncurry) :
    Measurable (fun (p : α × E) ↦ fderiv 𝕜 (f p.1) p.2) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁵ : TopologicalSpace α
    f : α → E → F
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    hf : Continuous (Function.uncurry f)
    ⊢ Measurable fun p => fderiv 𝕜 (f p.1) p.2
  -/
  refine measurable_of_isClosed (fun s hs ↦ ?_)
  have :
    (fun (p : α × E) ↦ fderiv 𝕜 (f p.1) p.2) ⁻¹' s =
      {p | DifferentiableAt 𝕜 (f p.1) p.2 ∧ fderiv 𝕜 (f p.1) p.2 ∈ s } ∪
        { p | ¬DifferentiableAt 𝕜 (f p.1) p.2} ∩ { _p | (0 : E →L[𝕜] F) ∈ s} :=
    Set.ext (fun x ↦ mem_preimage.trans fderiv_mem_iff)
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    inst✝⁸ : LocallyCompactSpace E
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁵ : TopologicalSpace α
    f : α → E → F
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : CompleteSpace F
    hf : Continuous (Function.uncurry f)
    s : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    hs : IsClosed s
    this : Eq (Set.preimage (fun p => fderiv 𝕜 (f p.1) p.2) s) (Union.union (setOf …
    ⊢ MeasurableSet (Set.preimage (fun p => fderiv 𝕜 (f p.1) p.2) s)
  -/
  rw [this]
  exact
    (measurableSet_of_differentiableAt_of_isComplete_with_param hf hs.isComplete).union
      ((measurableSet_of_differentiableAt_with_param _ hf).compl.inter (MeasurableSet.const _))


theorem measurable_fderiv_apply_const_with_param [MeasurableSpace F] [BorelSpace F]
    (hf : Continuous f.uncurry) (y : E) :
    Measurable (fun (p : α × E) ↦ fderiv 𝕜 (f p.1) p.2 y) :=
  (ContinuousLinearMap.measurable_apply y).comp (measurable_fderiv_with_param 𝕜 hf)


theorem measurable_deriv_with_param [LocallyCompactSpace 𝕜] [MeasurableSpace 𝕜]
    [OpensMeasurableSpace 𝕜] [MeasurableSpace F]
    [BorelSpace F] {f : α → 𝕜 → F} (hf : Continuous f.uncurry) :
    Measurable (fun (p : α × 𝕜) ↦ deriv (f p.1) p.2) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : OpensMeasurableSpace α
    inst✝⁵ : CompleteSpace F
    inst✝⁴ : LocallyCompactSpace 𝕜
    inst✝³ : MeasurableSpace 𝕜
    inst✝² : OpensMeasurableSpace 𝕜
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    f : α → 𝕜 → F
    hf : Continuous (Function.uncurry f)
    ⊢ Measurable fun p => deriv (f p.1) p.2
  -/
  simpa only [fderiv_deriv] using measurable_fderiv_apply_const_with_param 𝕜 hf 1
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_deriv_with_param [LocallyCompactSpace 𝕜] [MeasurableSpace 𝕜]
    [OpensMeasurableSpace 𝕜] [h : SecondCountableTopologyEither α F]
    {f : α → 𝕜 → F} (hf : Continuous f.uncurry) :
    StronglyMeasurable (fun (p : α × 𝕜) ↦ deriv (f p.1) p.2) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : CompleteSpace F
    inst✝² : LocallyCompactSpace 𝕜
    inst✝¹ : MeasurableSpace 𝕜
    inst✝ : OpensMeasurableSpace 𝕜
    h : SecondCountableTopologyEither α F
    f : α → 𝕜 → F
    hf : Continuous (Function.uncurry f)
    ⊢ MeasureTheory.StronglyMeasurable fun p => deriv (f p.1) p.2
  -/
  borelize F
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    α : Type u_4
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : CompleteSpace F
    inst✝² : LocallyCompactSpace 𝕜
    inst✝¹ : MeasurableSpace 𝕜
    inst✝ : OpensMeasurableSpace 𝕜
    h : SecondCountableTopologyEither α F
    f : α → 𝕜 → F
    hf : Continuous (Function.uncurry f)
    this✝¹ : MeasurableSpace F := borel F
    this✝ : BorelSpace F
    ⊢ MeasureTheory.StronglyMeasurable fun p => deriv (f p.1) p.2
  -/
  rcases h.out with hα|hF
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁹ : NontriviallyNormedField 𝕜
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      α : Type u_4
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : OpensMeasurableSpace α
      inst✝³ : CompleteSpace F
      inst✝² : LocallyCompactSpace 𝕜
      inst✝¹ : MeasurableSpace 𝕜
      inst✝ : OpensMeasurableSpace 𝕜
      h : SecondCountableTopologyEither α F
      f : α → 𝕜 → F
      hf : Continuous (Function.uncurry f)
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      hα : SecondCountableTopology α
      ⊢ MeasureTheory.StronglyMeasurable fun p => deriv (f p.1) p.2
    -/
  · have : ProperSpace 𝕜 := .of_locallyCompactSpace 𝕜
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁹ : NontriviallyNormedField 𝕜
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      α : Type u_4
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : OpensMeasurableSpace α
      inst✝³ : CompleteSpace F
      inst✝² : LocallyCompactSpace 𝕜
      inst✝¹ : MeasurableSpace 𝕜
      inst✝ : OpensMeasurableSpace 𝕜
      h : SecondCountableTopologyEither α F
      f : α → 𝕜 → F
      hf : Continuous (Function.uncurry f)
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      hα : SecondCountableTopology α
      this : ProperSpace 𝕜
      ⊢ MeasureTheory.StronglyMeasurable fun p => deriv (f p.1) p.2
    -/
    apply stronglyMeasurable_iff_measurable_separable.2 ⟨measurable_deriv_with_param hf, ?_⟩
    have : range (fun (p : α × 𝕜) ↦ deriv (f p.1) p.2)
        ⊆ closure (Submodule.span 𝕜 (range f.uncurry)) := by
      rintro - ⟨p, rfl⟩
      have A : deriv (f p.1) p.2 ∈ closure (Submodule.span 𝕜 (range (f p.1))) := by
        rw [← image_univ]
        apply range_deriv_subset_closure_span_image _ dense_univ (mem_range_self _)
      have B : range (f p.1) ⊆ range (f.uncurry) := by
        rintro - ⟨x, rfl⟩
        exact mem_range_self (p.1, x)
      exact closure_mono (Submodule.span_mono B) A
    /-
      𝕜 : Type u_1
      inst✝⁹ : NontriviallyNormedField 𝕜
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      α : Type u_4
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : OpensMeasurableSpace α
      inst✝³ : CompleteSpace F
      inst✝² : LocallyCompactSpace 𝕜
      inst✝¹ : MeasurableSpace 𝕜
      inst✝ : OpensMeasurableSpace 𝕜
      h : SecondCountableTopologyEither α F
      f : α → 𝕜 → F
      hf : Continuous (Function.uncurry f)
      this✝² : MeasurableSpace F := borel F
      this✝¹ : BorelSpace F
      hα : SecondCountableTopology α
      this✝ : ProperSpace 𝕜
      this : HasSubset.Subset (Set.range fun p => deriv (f p.1) p.2) (closure ↑(Subm …
      ⊢ TopologicalSpace.IsSeparable (Set.range fun p => deriv (f p.1) p.2)
    -/
    exact (isSeparable_range hf).span.closure.mono this
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁹ : NontriviallyNormedField 𝕜
      F : Type u_3
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      α : Type u_4
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : OpensMeasurableSpace α
      inst✝³ : CompleteSpace F
      inst✝² : LocallyCompactSpace 𝕜
      inst✝¹ : MeasurableSpace 𝕜
      inst✝ : OpensMeasurableSpace 𝕜
      h : SecondCountableTopologyEither α F
      f : α → 𝕜 → F
      hf : Continuous (Function.uncurry f)
      this✝¹ : MeasurableSpace F := borel F
      this✝ : BorelSpace F
      hF : SecondCountableTopology F
      ⊢ MeasureTheory.StronglyMeasurable fun p => deriv (f p.1) p.2
    -/
  · exact (measurable_deriv_with_param hf).stronglyMeasurable
    /-
      🎉 no goals
    -/


theorem aemeasurable_deriv_with_param [LocallyCompactSpace 𝕜] [MeasurableSpace 𝕜]
    [OpensMeasurableSpace 𝕜] [MeasurableSpace F]
    [BorelSpace F] {f : α → 𝕜 → F} (hf : Continuous f.uncurry) (μ : Measure (α × 𝕜)) :
    AEMeasurable (fun (p : α × 𝕜) ↦ deriv (f p.1) p.2) μ :=
  (measurable_deriv_with_param hf).aemeasurable


theorem aestronglyMeasurable_deriv_with_param [LocallyCompactSpace 𝕜] [MeasurableSpace 𝕜]
    [OpensMeasurableSpace 𝕜] [SecondCountableTopologyEither α F]
    {f : α → 𝕜 → F} (hf : Continuous f.uncurry) (μ : Measure (α × 𝕜)) :
    AEStronglyMeasurable (fun (p : α × 𝕜) ↦ deriv (f p.1) p.2) μ :=
  (stronglyMeasurable_deriv_with_param hf).aestronglyMeasurable


