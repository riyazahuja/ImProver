/-- If a sequence of functions real or complex functions are eventually differentiable on a
neighborhood of `x`, they are Cauchy _at_ `x`, and their derivatives
are a uniform Cauchy sequence in a neighborhood of `x`, then the functions form a uniform Cauchy
sequence in a neighborhood of `x`. -/
theorem uniformCauchySeqOnFilter_of_fderiv (hf' : UniformCauchySeqOnFilter f' l (𝓝 x))
    (hf : ∀ᶠ n : ι × E in l ×ˢ 𝓝 x, HasFDerivAt (f n.1) (f' n.1 n.2) n.2)
    (hfg : Cauchy (map (fun n => f n x) l)) : UniformCauchySeqOnFilter f l (𝓝 x) := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : UniformCauchySeqOnFilter f' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    ⊢ UniformCauchySeqOnFilter f l (nhds x)
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : UniformCauchySeqOnFilter f' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ UniformCauchySeqOnFilter f l (nhds x)
  -/
  letI : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 _
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : UniformCauchySeqOnFilter f' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ UniformCauchySeqOnFilter f l (nhds x)
  -/
  rw [SeminormedAddGroup.uniformCauchySeqOnFilter_iff_tendstoUniformlyOnFilter_zero] at hf' ⊢
  suffices
    TendstoUniformlyOnFilter (fun (n : ι × ι) (z : E) => f n.1 z - f n.2 z - (f n.1 x - f n.2 x)) 0
        (l ×ˢ l) (𝓝 x) ∧
      TendstoUniformlyOnFilter (fun (n : ι × ι) (_ : E) => f n.1 x - f n.2 x) 0 (l ×ˢ l) (𝓝 x) by
    have := this.1.add this.2
    rw [add_zero] at this
    exact this.congr (by simp)
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    hf' : TendstoUniformlyOnFilter (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0  …
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ And (TendstoUniformlyOnFilter (fun n z => HSub.hSub (HSub.hSub (f n.1 z) (f  …
  -/
  constructor
  · -- This inequality follows from the mean value theorem. To apply it, we will need to shrink our
    -- neighborhood to small enough ball
    /-
      case left
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOnFilter (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0  …
      this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ⊢ TendstoUniformlyOnFilter (fun n z => HSub.hSub (HSub.hSub (f n.1 z) (f n.2 z …
    -/
    rw [Metric.tendstoUniformlyOnFilter_iff] at hf' ⊢
    /-
      case left
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (0 n. …
    -/
    intro ε hε
    /-
      case left
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSub.hSub ( …
    -/
    have := (tendsto_swap4_prod.eventually (hf.prod_mk hf)).diag_of_prod_right
    /-
      case left
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
      ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSub.hSub ( …
    -/
    obtain ⟨a, b, c, d, e⟩ := eventually_prod_iff.1 ((hf' ε hε).and this)
    /-
      case left.intro.intro.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
      a : Prod ι ι → Prop
      b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
      c : E → Prop
      d : Filter.Eventually (fun y => c y) (nhds x)
      e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
      ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSub.hSub ( …
    -/
    obtain ⟨R, hR, hR'⟩ := Metric.nhds_basis_ball.eventually_iff.mp d
    /-
      case left.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
      a : Prod ι ι → Prop
      b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
      c : E → Prop
      d : Filter.Eventually (fun y => c y) (nhds x)
      e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
      R : Real
      hR : LT.lt 0 R
      hR' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x R) x_1 → c x_1
      ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSub.hSub ( …
    -/
    let r := min 1 R
    /-
      case left.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
      a : Prod ι ι → Prop
      b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
      c : E → Prop
      d : Filter.Eventually (fun y => c y) (nhds x)
      e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
      R : Real
      hR : LT.lt 0 R
      hR' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x R) x_1 → c x_1
      r : Real := Min.min 1 R
      ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSub.hSub ( …
    -/
    have hr : 0 < r := by simp [r, hR]
    have hr' : ∀ ⦃y : E⦄, y ∈ Metric.ball x r → c y := fun y hy =>
      hR' (lt_of_lt_of_le (Metric.mem_ball.mp hy) (min_le_right _ _))
    have hxy : ∀ y : E, y ∈ Metric.ball x r → ‖y - x‖ < 1 := by
      intro y hy
      rw [Metric.mem_ball, dist_eq_norm] at hy
      exact lt_of_lt_of_le hy (min_le_left _ _)
    have hxyε : ∀ y : E, y ∈ Metric.ball x r → ε * ‖y - x‖ < ε := by
      intro y hy
      exact (mul_lt_iff_lt_one_right hε.lt).mpr (hxy y hy)
    -- With a small ball in hand, apply the mean value theorem
    refine
      eventually_prod_iff.mpr
        ⟨_, b, fun e : E => Metric.ball x r e,
          eventually_mem_set.mpr (Metric.nhds_basis_ball.mem_of_mem hr), fun {n} hn {y} hy => ?_⟩
    /-
      case left.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
      a : Prod ι ι → Prop
      b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
      c : E → Prop
      d : Filter.Eventually (fun y => c y) (nhds x)
      e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
      R : Real
      hR : LT.lt 0 R
      hR' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x R) x_1 → c x_1
      r : Real := Min.min 1 R
      hr : LT.lt 0 r
      hr' : ∀ ⦃y : E⦄, Membership.mem (Metric.ball x r) y → c y
      hxy : ∀ (y : E), Membership.mem (Metric.ball x r) y → LT.lt (Norm.norm (HSub.h …
      hxyε : ∀ (y : E), Membership.mem (Metric.ball x r) y → LT.lt (HMul.hMul ε (Nor …
      n : Prod ι ι
      hn : a n
      y : E
      hy : (fun e => Metric.ball x r e) y
      ⊢ LT.lt (Dist.dist (0 { fst := n, snd := y }.2) (HSub.hSub (HSub.hSub (f { fst …
    -/
    simp only [Pi.zero_apply, dist_zero_left] at e ⊢
    /-
      case left.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
      a : Prod ι ι → Prop
      b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
      c : E → Prop
      d : Filter.Eventually (fun y => c y) (nhds x)
      R : Real
      hR : LT.lt 0 R
      hR' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x R) x_1 → c x_1
      r : Real := Min.min 1 R
      hr : LT.lt 0 r
      hr' : ∀ ⦃y : E⦄, Membership.mem (Metric.ball x r) y → c y
      hxy : ∀ (y : E), Membership.mem (Metric.ball x r) y → LT.lt (Norm.norm (HSub.h …
      hxyε : ∀ (y : E), Membership.mem (Metric.ball x r) y → LT.lt (HMul.hMul ε (Nor …
      n : Prod ι ι
      hn : a n
      y : E
      hy : (fun e => Metric.ball x r e) y
      e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Norm.norm (HSub.hSub  …
      ⊢ LT.lt (Norm.norm (HSub.hSub (HSub.hSub (f n.1 y) (f n.2 y)) (HSub.hSub (f n. …
    -/
    refine lt_of_le_of_lt ?_ (hxyε y hy)
    exact
      Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
        (fun y hy => ((e hn (hr' hy)).2.1.sub (e hn (hr' hy)).2.2).hasFDerivWithinAt)
        (fun y hy => (e hn (hr' hy)).1.le) (convex_ball x r) (Metric.mem_ball_self hr) hy
  · -- This is just `hfg` run through `eventually_prod_iff`
    /-
      case right
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOnFilter (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0  …
      this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ⊢ TendstoUniformlyOnFilter (fun n x_1 => HSub.hSub (f n.1 x) (f n.2 x)) 0 (SPr …
    -/
    refine Metric.tendstoUniformlyOnFilter_iff.mpr fun ε hε => ?_
    /-
      case right
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOnFilter (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0  …
      this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f n.1.1 x)  …
    -/
    obtain ⟨t, ht, ht'⟩ := (Metric.cauchy_iff.mp hfg).2 ε hε
    exact
      eventually_prod_iff.mpr
        ⟨fun n : ι × ι => f n.1 x ∈ t ∧ f n.2 x ∈ t,
          eventually_prod_iff.mpr ⟨_, ht, _, ht, fun {n} hn {n'} hn' => ⟨hn, hn'⟩⟩,
          fun _ => True,
          by simp,
          fun {n} hn {y} _ => by simpa [norm_sub_rev, dist_eq_norm] using ht' _ hn.1 _ hn.2⟩


/-- A variant of the second fundamental theorem of calculus (FTC-2): If a sequence of functions
between real or complex normed spaces are differentiable on a ball centered at `x`, they
form a Cauchy sequence _at_ `x`, and their derivatives are Cauchy uniformly on the ball, then the
functions form a uniform Cauchy sequence on the ball.

NOTE: The fact that we work on a ball is typically all that is necessary to work with power series
and Dirichlet series (our primary use case). However, this can be generalized by replacing the ball
with any connected, bounded, open set and replacing uniform convergence with local uniform
convergence. See `cauchy_map_of_uniformCauchySeqOn_fderiv`.
-/
theorem uniformCauchySeqOn_ball_of_fderiv {r : ℝ} (hf' : UniformCauchySeqOn f' l (Metric.ball x r))
    (hf : ∀ n : ι, ∀ y : E, y ∈ Metric.ball x r → HasFDerivAt (f n) (f' n y) y)
    (hfg : Cauchy (map (fun n => f n x) l)) : UniformCauchySeqOn f l (Metric.ball x r) := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  letI : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 _
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  have : NeBot l := (cauchy_map_iff.1 hfg).1
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    this : l.NeBot
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  rcases le_or_lt r 0 with (hr | hr)
  · simp only [Metric.ball_eq_empty.2 hr, UniformCauchySeqOn, Set.mem_empty_iff_false,
      IsEmpty.forall_iff, eventually_const, imp_true_iff]
  /-
    case inr
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    this : l.NeBot
    hr : LT.lt 0 r
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  rw [SeminormedAddGroup.uniformCauchySeqOn_iff_tendstoUniformlyOn_zero] at hf' ⊢
  suffices
    TendstoUniformlyOn (fun (n : ι × ι) (z : E) => f n.1 z - f n.2 z - (f n.1 x - f n.2 x)) 0
        (l ×ˢ l) (Metric.ball x r) ∧
      TendstoUniformlyOn (fun (n : ι × ι) (_ : E) => f n.1 x - f n.2 x) 0
        (l ×ˢ l) (Metric.ball x r) by
    have := this.1.add this.2
    rw [add_zero] at this
    refine this.congr ?_
    filter_upwards with n z _ using (by simp)
  /-
    case inr
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    r : Real
    hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
    this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    this : l.NeBot
    hr : LT.lt 0 r
    ⊢ And (TendstoUniformlyOn (fun n z => HSub.hSub (HSub.hSub (f n.1 z) (f n.2 z) …
  -/
  constructor
  · -- This inequality follows from the mean value theorem
    /-
      case inr.left
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ⊢ TendstoUniformlyOn (fun n z => HSub.hSub (HSub.hSub (f n.1 z) (f n.2 z)) (HS …
    -/
    rw [Metric.tendstoUniformlyOn_iff] at hf' ⊢
    /-
      case inr.left
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membe …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membershi …
    -/
    intro ε hε
    obtain ⟨q, hqpos, hq⟩ : ∃ q : ℝ, 0 < q ∧ q * r < ε := by
      simp_rw [mul_comm]
      exact exists_pos_mul_lt hε.lt r
    /-
      case inr.left.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membe …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      q : Real
      hqpos : LT.lt 0 q
      hq : LT.lt (HMul.hMul q r) ε
      ⊢ Filter.Eventually (fun n => ∀ (x_1 : E), Membership.mem (Metric.ball x r) x_ …
    -/
    apply (hf' q hqpos.gt).mono
    /-
      case inr.left.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membe …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      q : Real
      hqpos : LT.lt 0 q
      hq : LT.lt (HMul.hMul q r) ε
      ⊢ ∀ (x_1 : Prod ι ι), (∀ (x_2 : E), Membership.mem (Metric.ball x r) x_2 → LT. …
    -/
    intro n hn y hy
    /-
      case inr.left.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membe …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      q : Real
      hqpos : LT.lt 0 q
      hq : LT.lt (HMul.hMul q r) ε
      n : Prod ι ι
      hn : ∀ (x_1 : E), Membership.mem (Metric.ball x r) x_1 → LT.lt (Dist.dist (0 x …
      y : E
      hy : Membership.mem (Metric.ball x r) y
      ⊢ LT.lt (Dist.dist (0 y) (HSub.hSub (HSub.hSub (f n.1 y) (f n.2 y)) (HSub.hSub …
    -/
    simp_rw [dist_eq_norm, Pi.zero_apply, zero_sub, norm_neg] at hn ⊢
    have mvt :=
      Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
        (fun z hz => ((hf n.1 z hz).sub (hf n.2 z hz)).hasFDerivWithinAt) (fun z hz => (hn z hz).le)
        (convex_ball x r) (Metric.mem_ball_self hr) hy
    /-
      case inr.left.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membe …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      q : Real
      hqpos : LT.lt 0 q
      hq : LT.lt (HMul.hMul q r) ε
      n : Prod ι ι
      y : E
      hy : Membership.mem (Metric.ball x r) y
      hn : ∀ (x_1 : E), Membership.mem (Metric.ball x r) x_1 → LT.lt (Norm.norm (HSu …
      mvt : LE.le (Norm.norm (HSub.hSub (HSub.hSub (f n.1 y) (f n.2 y)) (HSub.hSub ( …
      ⊢ LT.lt (Norm.norm (HSub.hSub (HSub.hSub (f n.1 y) (f n.2 y)) (HSub.hSub (f n. …
    -/
    refine lt_of_le_of_lt mvt ?_
    have : q * ‖y - x‖ < q * r :=
      mul_lt_mul' rfl.le (by simpa only [dist_eq_norm] using Metric.mem_ball.mp hy) (norm_nonneg _)
        hqpos
    /-
      case inr.left.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x_1 : E), Membe …
      this✝¹ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this✝ : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      q : Real
      hqpos : LT.lt 0 q
      hq : LT.lt (HMul.hMul q r) ε
      n : Prod ι ι
      y : E
      hy : Membership.mem (Metric.ball x r) y
      hn : ∀ (x_1 : E), Membership.mem (Metric.ball x r) x_1 → LT.lt (Norm.norm (HSu …
      mvt : LE.le (Norm.norm (HSub.hSub (HSub.hSub (f n.1 y) (f n.2 y)) (HSub.hSub ( …
      this : LT.lt (HMul.hMul q (Norm.norm (HSub.hSub y x))) (HMul.hMul q r)
      ⊢ LT.lt (HMul.hMul q (Norm.norm (HSub.hSub y x))) ε
    -/
    exact this.trans hq
    /-
      🎉 no goals
    -/
  · -- This is just `hfg` run through `eventually_prod_iff`
    /-
      case inr.right
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ⊢ TendstoUniformlyOn (fun n x_1 => HSub.hSub (f n.1 x) (f n.2 x)) 0 (SProd.spr …
    -/
    refine Metric.tendstoUniformlyOn_iff.mpr fun ε hε => ?_
    /-
      case inr.right
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun n => ∀ (x_1 : E), Membership.mem (Metric.ball x r) x_ …
    -/
    obtain ⟨t, ht, ht'⟩ := (Metric.cauchy_iff.mp hfg).2 ε hε
    /-
      case inr.right.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      t : Set G
      ht : Membership.mem (Filter.map (fun n => f n x) l) t
      ht' : ∀ (x : G), Membership.mem t x → ∀ (y : G), Membership.mem t y → LT.lt (D …
      ⊢ Filter.Eventually (fun n => ∀ (x_1 : E), Membership.mem (Metric.ball x r) x_ …
    -/
    rw [eventually_prod_iff]
    /-
      case inr.right.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      t : Set G
      ht : Membership.mem (Filter.map (fun n => f n x) l) t
      ht' : ∀ (x : G), Membership.mem t x → ∀ (y : G), Membership.mem t y → LT.lt (D …
      ⊢ Exists fun pa => And (Filter.Eventually (fun x => pa x) l) (Exists fun pb => …
    -/
    refine ⟨fun n => f n x ∈ t, ht, fun n => f n x ∈ t, ht, ?_⟩
    /-
      case inr.right.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      t : Set G
      ht : Membership.mem (Filter.map (fun n => f n x) l) t
      ht' : ∀ (x : G), Membership.mem t x → ∀ (y : G), Membership.mem t y → LT.lt (D …
      ⊢ ∀ {x_1 : ι}, (fun n => Membership.mem t (f n x)) x_1 → ∀ {y : ι}, (fun n =>  …
    -/
    intro n hn n' hn' z _
    /-
      case inr.right.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      t : Set G
      ht : Membership.mem (Filter.map (fun n => f n x) l) t
      ht' : ∀ (x : G), Membership.mem t x → ∀ (y : G), Membership.mem t y → LT.lt (D …
      n : ι
      hn : Membership.mem t (f n x)
      n' : ι
      hn' : Membership.mem t (f n' x)
      z : E
      a✝ : Membership.mem (Metric.ball x r) z
      ⊢ LT.lt (Dist.dist (0 z) (HSub.hSub (f { fst := n, snd := n' }.1 x) (f { fst : …
    -/
    rw [dist_eq_norm, Pi.zero_apply, zero_sub, norm_neg, ← dist_eq_norm]
    /-
      case inr.right.intro.intro
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : IsRCLikeNormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : ι → E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      r : Real
      hf : ∀ (n : ι) (y : E), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
      hfg : Cauchy (Filter.map (fun n => f n x) l)
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      hf' : TendstoUniformlyOn (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 (SProd …
      this✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      this : l.NeBot
      hr : LT.lt 0 r
      ε : Real
      hε : GT.gt ε 0
      t : Set G
      ht : Membership.mem (Filter.map (fun n => f n x) l) t
      ht' : ∀ (x : G), Membership.mem t x → ∀ (y : G), Membership.mem t y → LT.lt (D …
      n : ι
      hn : Membership.mem t (f n x)
      n' : ι
      hn' : Membership.mem t (f n' x)
      z : E
      a✝ : Membership.mem (Metric.ball x r) z
      ⊢ LT.lt (Dist.dist (f { fst := n, snd := n' }.1 x) (f { fst := n, snd := n' }. …
    -/
    exact ht' _ hn _ hn'
    /-
      🎉 no goals
    -/


/-- If a sequence of functions between real or complex normed spaces are differentiable on a
preconnected open set, they form a Cauchy sequence _at_ `x`, and their derivatives are Cauchy
uniformly on the set, then the functions form a Cauchy sequence at any point in the set. -/
theorem cauchy_map_of_uniformCauchySeqOn_fderiv {s : Set E} (hs : IsOpen s) (h's : IsPreconnected s)
    (hf' : UniformCauchySeqOn f' l s) (hf : ∀ n : ι, ∀ y : E, y ∈ s → HasFDerivAt (f n) (f' n y) y)
    {x₀ x : E} (hx₀ : x₀ ∈ s) (hx : x ∈ s) (hfg : Cauchy (map (fun n => f n x₀) l)) :
    Cauchy (map (fun n => f n x) l) := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    ⊢ Cauchy (Filter.map (fun n => f n x) l)
  -/
  have : NeBot l := (cauchy_map_iff.1 hfg).1
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    ⊢ Cauchy (Filter.map (fun n => f n x) l)
  -/
  let t := { y | y ∈ s ∧ Cauchy (map (fun n => f n y) l) }
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    t : Set E := setOf fun y => And (Membership.mem s y) (Cauchy (Filter.map (fun  …
    ⊢ Cauchy (Filter.map (fun n => f n x) l)
  -/
  suffices H : s ⊆ t from (H hx).2
  have A : ∀ x ε, x ∈ t → Metric.ball x ε ⊆ s → Metric.ball x ε ⊆ t := fun x ε xt hx y hy =>
    ⟨hx hy,
      (uniformCauchySeqOn_ball_of_fderiv (hf'.mono hx) (fun n y hy => hf n y (hx hy))
            xt.2).cauchy_map
        hy⟩
  have open_t : IsOpen t := by
    rw [Metric.isOpen_iff]
    intro x hx
    rcases Metric.isOpen_iff.1 hs x hx.1 with ⟨ε, εpos, hε⟩
    exact ⟨ε, εpos, A x ε hx hε⟩
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    t : Set E := setOf fun y => And (Membership.mem s y) (Cauchy (Filter.map (fun  …
    A : ∀ (x : E) (ε : Real), Membership.mem t x → HasSubset.Subset (Metric.ball x …
    open_t : IsOpen t
    ⊢ HasSubset.Subset s t
  -/
  have st_nonempty : (s ∩ t).Nonempty := ⟨x₀, hx₀, ⟨hx₀, hfg⟩⟩
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    t : Set E := setOf fun y => And (Membership.mem s y) (Cauchy (Filter.map (fun  …
    A : ∀ (x : E) (ε : Real), Membership.mem t x → HasSubset.Subset (Metric.ball x …
    open_t : IsOpen t
    st_nonempty : (Inter.inter s t).Nonempty
    ⊢ HasSubset.Subset s t
  -/
  suffices H : closure t ∩ s ⊆ t from h's.subset_of_closure_inter_subset open_t st_nonempty H
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    t : Set E := setOf fun y => And (Membership.mem s y) (Cauchy (Filter.map (fun  …
    A : ∀ (x : E) (ε : Real), Membership.mem t x → HasSubset.Subset (Metric.ball x …
    open_t : IsOpen t
    st_nonempty : (Inter.inter s t).Nonempty
    ⊢ HasSubset.Subset (Inter.inter (closure t) s) t
  -/
  rintro x ⟨xt, xs⟩
  /-
    case intro
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x✝ : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x✝
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    t : Set E := setOf fun y => And (Membership.mem s y) (Cauchy (Filter.map (fun  …
    A : ∀ (x : E) (ε : Real), Membership.mem t x → HasSubset.Subset (Metric.ball x …
    open_t : IsOpen t
    st_nonempty : (Inter.inter s t).Nonempty
    x : E
    xt : Membership.mem (closure t) x
    xs : Membership.mem s x
    ⊢ Membership.mem t x
  -/
  obtain ⟨ε, εpos, hε⟩ : ∃ (ε : ℝ), ε > 0 ∧ Metric.ball x ε ⊆ s := Metric.isOpen_iff.1 hs x xs
  obtain ⟨y, yt, hxy⟩ : ∃ (y : E), y ∈ t ∧ dist x y < ε / 2 :=
    Metric.mem_closure_iff.1 xt _ (half_pos εpos)
  have B : Metric.ball y (ε / 2) ⊆ Metric.ball x ε := by
    apply Metric.ball_subset_ball'; rw [dist_comm]; linarith
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    s : Set E
    hs : IsOpen s
    h's : IsPreconnected s
    hf' : UniformCauchySeqOn f' l s
    hf : ∀ (n : ι) (y : E), Membership.mem s y → HasFDerivAt (f n) (f' n y) y
    x₀ x✝ : E
    hx₀ : Membership.mem s x₀
    hx : Membership.mem s x✝
    hfg : Cauchy (Filter.map (fun n => f n x₀) l)
    this : l.NeBot
    t : Set E := setOf fun y => And (Membership.mem s y) (Cauchy (Filter.map (fun  …
    A : ∀ (x : E) (ε : Real), Membership.mem t x → HasSubset.Subset (Metric.ball x …
    open_t : IsOpen t
    st_nonempty : (Inter.inter s t).Nonempty
    x : E
    xt : Membership.mem (closure t) x
    xs : Membership.mem s x
    ε : Real
    εpos : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) s
    y : E
    yt : Membership.mem t y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv ε 2)
    B : HasSubset.Subset (Metric.ball y (HDiv.hDiv ε 2)) (Metric.ball x ε)
    ⊢ Membership.mem t x
  -/
  exact A y (ε / 2) yt (B.trans hε) (Metric.mem_ball.2 hxy)
  /-
    🎉 no goals
  -/


/-- If `f_n → g` pointwise and the derivatives `(f_n)' → h` _uniformly_ converge, then
in fact for a fixed `y`, the difference quotients `‖z - y‖⁻¹ • (f_n z - f_n y)` converge
_uniformly_ to `‖z - y‖⁻¹ • (g z - g y)` -/
theorem difference_quotients_converge_uniformly
    {E : Type*} [NormedAddCommGroup E] {𝕜 : Type*} [RCLike 𝕜]
    [NormedSpace 𝕜 E] {G : Type*} [NormedAddCommGroup G] [NormedSpace 𝕜 G] {f : ι → E → G}
    {g : E → G} {f' : ι → E → E →L[𝕜] G} {g' : E → E →L[𝕜] G} {x : E}
    (hf' : TendstoUniformlyOnFilter f' g' l (𝓝 x))
    (hf : ∀ᶠ n : ι × E in l ×ˢ 𝓝 x, HasFDerivAt (f n.1) (f' n.1 n.2) n.2)
    (hfg : ∀ᶠ y : E in 𝓝 x, Tendsto (fun n => f n y) l (𝓝 (g y))) :
    TendstoUniformlyOnFilter (fun n : ι => fun y : E => (‖y - x‖⁻¹ : 𝕜) • (f n y - f n x))
      (fun y : E => (‖y - x‖⁻¹ : 𝕜) • (g y - g x)) l (𝓝 x) := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    ⊢ TendstoUniformlyOnFilter (fun n y => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub. …
  -/
  let A : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 _
  refine
    UniformCauchySeqOnFilter.tendstoUniformlyOnFilter_of_tendsto ?_
      ((hfg.and (eventually_const.mpr hfg.self_of_nhds)).mono fun y hy =>
        (hy.1.sub hy.2).const_smul _)
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ UniformCauchySeqOnFilter (fun n y => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub. …
  -/
  rw [SeminormedAddGroup.uniformCauchySeqOnFilter_iff_tendstoUniformlyOnFilter_zero]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ TendstoUniformlyOnFilter (fun n z => HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm. …
  -/
  rw [Metric.tendstoUniformlyOnFilter_iff]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (0 n. …
  -/
  have hfg' := hf'.uniformCauchySeqOnFilter
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    hfg' : UniformCauchySeqOnFilter f' l (nhds x)
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (0 n. …
  -/
  rw [SeminormedAddGroup.uniformCauchySeqOnFilter_iff_tendstoUniformlyOnFilter_zero] at hfg'
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    hfg' : TendstoUniformlyOnFilter (fun n z => HSub.hSub (f' n.1 z) (f' n.2 z)) 0 …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (0 n. …
  -/
  rw [Metric.tendstoUniformlyOnFilter_iff] at hfg'
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    hfg' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (0 n. …
  -/
  intro ε hε
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    hfg' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSMul.hSMul …
  -/
  obtain ⟨q, hqpos, hqε⟩ := exists_pos_rat_lt hε
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    hfg' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSMul.hSMul …
  -/
  specialize hfg' (q : ℝ) (by simp [hqpos])
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSMul.hSMul …
  -/
  have := (tendsto_swap4_prod.eventually (hf.prod_mk hf)).diag_of_prod_right
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSMul.hSMul …
  -/
  obtain ⟨a, b, c, d, e⟩ := eventually_prod_iff.1 (hfg'.and this)
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSMul.hSMul …
  -/
  obtain ⟨r, hr, hr'⟩ := Metric.nhds_basis_ball.eventually_iff.mp d
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (HSMul.hSMul …
  -/
  rw [eventually_prod_iff]
  refine
    ⟨_, b, fun e : E => Metric.ball x r e,
      eventually_mem_set.mpr (Metric.nhds_basis_ball.mem_of_mem hr), fun {n} hn {y} hy => ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    ⊢ LT.lt (Dist.dist (0 { fst := n, snd := y }.2) (HSub.hSub (HSMul.hSMul (Inv.i …
  -/
  simp only [Pi.zero_apply, dist_zero_left]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    ⊢ LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub y x …
  -/
  rw [← smul_sub, norm_smul, norm_inv, RCLike.norm_coe_norm]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    ⊢ LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.hSub …
  -/
  refine lt_of_le_of_lt ?_ hqε
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    ⊢ LE.le (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.hSub …
  -/
  by_cases hyz' : x = y; · simp [hyz', hqpos.le]
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    hyz' : Not (Eq x y)
    ⊢ LE.le (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.hSub …
  -/
  have hyz : 0 < ‖y - x‖ := by rw [norm_pos_iff]; intro hy'; exact hyz' (eq_of_sub_eq_zero hy').symm
  /-
    case neg
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    hyz' : Not (Eq x y)
    hyz : LT.lt 0 (Norm.norm (HSub.hSub y x))
    ⊢ LE.le (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.hSub …
  -/
  rw [inv_mul_le_iff₀ hyz, mul_comm, sub_sub_sub_comm]
  /-
    case neg
    ι : Type u_1
    l : Filter ι
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    𝕜 : Type u_6
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    G : Type u_7
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    A : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ε : Real
    hε : GT.gt ε 0
    q : Rat
    hqpos : LT.lt 0 q
    hqε : LT.lt (↑q) ε
    hfg' : Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1 …
    this : Filter.Eventually (fun x => And (HasFDerivAt (f { fst := { fst := { fst …
    a : Prod ι ι → Prop
    b : Filter.Eventually (fun x => a x) (SProd.sprod l l)
    c : E → Prop
    d : Filter.Eventually (fun y => c y) (nhds x)
    e : ∀ {x : Prod ι ι}, a x → ∀ {y : E}, c y → And (LT.lt (Dist.dist (0 { fst := …
    r : Real
    hr : LT.lt 0 r
    hr' : ∀ ⦃x_1 : E⦄, Membership.mem (Metric.ball x r) x_1 → c x_1
    n : Prod ι ι
    hn : a n
    y : E
    hy : (fun e => Metric.ball x r e) y
    hyz' : Not (Eq x y)
    hyz : LT.lt 0 (Norm.norm (HSub.hSub y x))
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f n.1 y) (f n.2 y)) (HSub.hSub (f n. …
  -/
  simp only [Pi.zero_apply, dist_zero_left] at e
  refine
    Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun y hy => ((e hn (hr' hy)).2.1.sub (e hn (hr' hy)).2.2).hasFDerivWithinAt)
      (fun y hy => (e hn (hr' hy)).1.le) (convex_ball x r) (Metric.mem_ball_self hr) hy


/-- `(d/dx) lim_{n → ∞} f n x = lim_{n → ∞} f' n x` when the `f' n` converge
_uniformly_ to their limit at `x`.

In words the assumptions mean the following:
  * `hf'`: The `f'` converge "uniformly at" `x` to `g'`. This does not mean that the `f' n` even
    converge away from `x`!
  * `hf`: For all `(y, n)` with `y` sufficiently close to `x` and `n` sufficiently large, `f' n` is
    the derivative of `f n`
  * `hfg`: The `f n` converge pointwise to `g` on a neighborhood of `x` -/
theorem hasFDerivAt_of_tendstoUniformlyOnFilter [NeBot l]
    (hf' : TendstoUniformlyOnFilter f' g' l (𝓝 x))
    (hf : ∀ᶠ n : ι × E in l ×ˢ 𝓝 x, HasFDerivAt (f n.1) (f' n.1 n.2) n.2)
    (hfg : ∀ᶠ y in 𝓝 x, Tendsto (fun n => f n y) l (𝓝 (g y))) : HasFDerivAt g (g' x) x := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    ⊢ HasFDerivAt g (g' x) x
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  -- The proof strategy follows several steps:
  --   1. The quantifiers in the definition of the derivative are
  --      `∀ ε > 0, ∃δ > 0, ∀y ∈ B_δ(x)`. We will introduce a quantifier in the middle:
  --      `∀ ε > 0, ∃N, ∀n ≥ N, ∃δ > 0, ∀y ∈ B_δ(x)` which will allow us to introduce the `f(') n`
  --   2. The order of the quantifiers `hfg` are opposite to what we need. We will be able to swap
  --      the quantifiers using the uniform convergence assumption
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ HasFDerivAt g (g' x) x
  -/
  rw [hasFDerivAt_iff_tendsto]
  -- Introduce extra quantifier via curried filters
  suffices
    Tendsto (fun y : ι × E => ‖y.2 - x‖⁻¹ * ‖g y.2 - g x - (g' x) (y.2 - x)‖)
      (l.curry (𝓝 x)) (𝓝 0) by
    rw [Metric.tendsto_nhds] at this ⊢
    intro ε hε
    specialize this ε hε
    rw [eventually_curry_iff] at this
    simp only at this
    exact (eventually_const.mp this).mono (by simp only [imp_self, forall_const])
  -- With the new quantifier in hand, we can perform the famous `ε/3` proof. Specifically,
  -- we will break up the limit (the difference functions minus the derivative go to 0) into 3:
  --   * The difference functions of the `f n` converge *uniformly* to the difference functions
  --     of the `g n`
  --   * The `f' n` are the derivatives of the `f n`
  --   * The `f' n` converge to `g'` at `x`
  conv =>
    congr
    ext
    rw [← abs_norm, ← abs_inv, ← @RCLike.norm_ofReal 𝕜 _ _, RCLike.ofReal_inv, ← norm_smul]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Filter.Tendsto (fun x_1 => Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub …
  -/
  rw [← tendsto_zero_iff_norm_tendsto_zero]
  have :
    (fun a : ι × E => (‖a.2 - x‖⁻¹ : 𝕜) • (g a.2 - g x - (g' x) (a.2 - x))) =
      ((fun a : ι × E => (‖a.2 - x‖⁻¹ : 𝕜) • (g a.2 - g x - (f a.1 a.2 - f a.1 x))) +
          fun a : ι × E =>
          (‖a.2 - x‖⁻¹ : 𝕜) • (f a.1 a.2 - f a.1 x - ((f' a.1 x) a.2 - (f' a.1 x) x))) +
        fun a : ι × E => (‖a.2 - x‖⁻¹ : 𝕜) • (f' a.1 x - g' x) (a.2 - x) := by
    ext; simp only [Pi.add_apply]; rw [← smul_add, ← smul_add]; congr
    simp only [map_sub, sub_add_sub_cancel, ContinuousLinearMap.coe_sub', Pi.sub_apply]
    -- Porting note: added
    abel
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSub …
    ⊢ Filter.Tendsto (fun x_1 => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub x_1.2 …
  -/
  simp_rw [this]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSub …
    ⊢ Filter.Tendsto (HAdd.hAdd (HAdd.hAdd (fun a => HSMul.hSMul (Inv.inv ↑(Norm.n …
  -/
  have : 𝓝 (0 : G) = 𝓝 (0 + 0 + 0) := by simp only [add_zero]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
    this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
    ⊢ Filter.Tendsto (HAdd.hAdd (HAdd.hAdd (fun a => HSMul.hSMul (Inv.inv ↑(Norm.n …
  -/
  rw [this]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
    this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
    ⊢ Filter.Tendsto (HAdd.hAdd (HAdd.hAdd (fun a => HSMul.hSMul (Inv.inv ↑(Norm.n …
  -/
  refine Tendsto.add (Tendsto.add ?_ ?_) ?_
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ⊢ Filter.Tendsto (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x)) …
    -/
  · have := difference_quotients_converge_uniformly hf' hf hfg
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : TendstoUniformlyOnFilter (fun n y => HSMul.hSMul (Inv.inv ↑(Norm.norm ( …
      ⊢ Filter.Tendsto (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x)) …
    -/
    rw [Metric.tendstoUniformlyOnFilter_iff] at this
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ⊢ Filter.Tendsto (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x)) …
    -/
    rw [Metric.tendsto_nhds]
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dist (HS …
    -/
    intro ε hε
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun x_1 => LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm. …
    -/
    apply ((this ε hε).filter_mono curry_le_prod).mono
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ε : Real
      hε : GT.gt ε 0
      ⊢ ∀ (x_1 : Prod ι E), LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub …
    -/
    intro n hn
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ε : Real
      hε : GT.gt ε 0
      n : Prod ι E
      hn : LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub n.2 x))) (H …
      ⊢ LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub n.2 x))) (HSub …
    -/
    rw [dist_eq_norm] at hn ⊢
    /-
      case refine_1
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ε : Real
      hε : GT.gt ε 0
      n : Prod ι E
      hn : LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub  …
      ⊢ LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub n.2 …
    -/
    convert hn using 2
    /-
      case h.e'_3.h.e'_3
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist  …
      ε : Real
      hε : GT.gt ε 0
      n : Prod ι E
      hn : LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub  …
      ⊢ Eq (HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub n.2 x))) (HSub.hS …
    -/
    module
    /-
      🎉 no goals
    -/
  · -- (Almost) the definition of the derivatives
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ⊢ Filter.Tendsto (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x)) …
    -/
    rw [Metric.tendsto_nhds]
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dist (HS …
    -/
    intro ε hε
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun x_1 => LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm. …
    -/
    rw [eventually_curry_iff]
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      ⊢ Filter.Eventually (fun x_1 => Filter.Eventually (fun y => LT.lt (Dist.dist ( …
    -/
    refine hf.curry.mono fun n hn => ?_
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      ⊢ Filter.Eventually (fun y => LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.no …
    -/
    have := hn.self_of_nhds
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      this : HasFDerivAt (f { fst := n, snd := x }.1) (f' { fst := n, snd := x }.1 { …
      ⊢ Filter.Eventually (fun y => LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.no …
    -/
    rw [hasFDerivAt_iff_tendsto, Metric.tendsto_nhds] at this
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dis …
      ⊢ Filter.Eventually (fun y => LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.no …
    -/
    refine (this ε hε).mono fun y hy => ?_
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dis …
      y : E
      hy : LT.lt (Dist.dist (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y { fst := n,  …
      ⊢ LT.lt (Dist.dist (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub { fst := n, sn …
    -/
    rw [dist_eq_norm] at hy ⊢
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dis …
      y : E
      hy : LT.lt (Norm.norm (HSub.hSub (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y { …
      ⊢ LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub { f …
    -/
    simp only [sub_zero, map_sub, norm_mul, norm_inv, norm_norm] at hy ⊢
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dis …
      y : E
      hy : LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.h …
      ⊢ LT.lt (Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub y x))) (HSub.h …
    -/
    rw [norm_smul, norm_inv, RCLike.norm_coe_norm]
    /-
      case refine_2
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HS …
      this✝ : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ε : Real
      hε : GT.gt ε 0
      n : ι
      hn : Filter.Eventually (fun y => HasFDerivAt (f { fst := n, snd := y }.1) (f'  …
      this : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x_1 => LT.lt (Dist.dis …
      y : E
      hy : LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.h …
      ⊢ LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm.norm (HSub.hSub …
    -/
    exact hy
    /-
      🎉 no goals
    -/
  · -- hfg' after specializing to `x` and applying the definition of the operator norm
    /-
      case refine_3
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      ⊢ Filter.Tendsto (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x)) …
    -/
    refine Tendsto.mono_left ?_ curry_le_prod
    have h1 : Tendsto (fun n : ι × E => g' n.2 - f' n.1 n.2) (l ×ˢ 𝓝 x) (𝓝 0) := by
      rw [Metric.tendstoUniformlyOnFilter_iff] at hf'
      exact Metric.tendsto_nhds.mpr fun ε hε => by simpa using hf' ε hε
    have h2 : Tendsto (fun n : ι => g' x - f' n x) l (𝓝 0) := by
      rw [Metric.tendsto_nhds] at h1 ⊢
      exact fun ε hε => (h1 ε hε).curry.mono fun n hn => hn.self_of_nhds
    refine squeeze_zero_norm ?_
      (tendsto_zero_iff_norm_tendsto_zero.mp (tendsto_fst.comp (h2.prod_map tendsto_id)))
    /-
      case refine_3
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      ⊢ ∀ (n : Prod ι E), LE.le (Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.h …
    -/
    intro n
    /-
      case refine_3
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      n : Prod ι E
      ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub n.2 x))) ((HSu …
    -/
    simp_rw [norm_smul, norm_inv, RCLike.norm_coe_norm]
    /-
      case refine_3
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      n : Prod ι E
      ⊢ LE.le (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub n.2 x))) (Norm.norm ((HSub.h …
    -/
    by_cases hx : x = n.2; · simp [hx]
                             /-
                               🎉 no goals
                             -/
    have hnx : 0 < ‖n.2 - x‖ := by
      rw [norm_pos_iff]; intro hx'; exact hx (eq_of_sub_eq_zero hx').symm
    /-
      case neg
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      n : Prod ι E
      hx : Not (Eq x n.2)
      hnx : LT.lt 0 (Norm.norm (HSub.hSub n.2 x))
      ⊢ LE.le (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub n.2 x))) (Norm.norm ((HSub.h …
    -/
    rw [inv_mul_le_iff₀ hnx, mul_comm]
    /-
      case neg
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      n : Prod ι E
      hx : Not (Eq x n.2)
      hnx : LT.lt 0 (Norm.norm (HSub.hSub n.2 x))
      ⊢ LE.le (Norm.norm ((HSub.hSub (f' n.1 x) (g' x)) (HSub.hSub n.2 x))) (HMul.hM …
    -/
    simp only [Function.comp_apply, Prod.map_apply']
    /-
      case neg
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      n : Prod ι E
      hx : Not (Eq x n.2)
      hnx : LT.lt 0 (Norm.norm (HSub.hSub n.2 x))
      ⊢ LE.le (Norm.norm ((HSub.hSub (f' n.1 x) (g' x)) (HSub.hSub n.2 x))) (HMul.hM …
    -/
    rw [norm_sub_rev]
    /-
      case neg
      ι : Type u_1
      l : Filter ι
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : IsRCLikeNormedField 𝕜
      inst✝³ : NormedSpace 𝕜 E
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      f : ι → E → G
      g : E → G
      f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
      g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
      x : E
      inst✝ : l.NeBot
      hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
      hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
      hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
      this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝ : Eq (fun a => HSMul.hSMul (Inv.inv ↑(Norm.norm (HSub.hSub a.2 x))) (HSu …
      this : Eq (nhds 0) (nhds (HAdd.hAdd (HAdd.hAdd 0 0) 0))
      h1 : Filter.Tendsto (fun n => HSub.hSub (g' n.2) (f' n.1 n.2)) (SProd.sprod l  …
      h2 : Filter.Tendsto (fun n => HSub.hSub (g' x) (f' n x)) l (nhds 0)
      n : Prod ι E
      hx : Not (Eq x n.2)
      hnx : LT.lt 0 (Norm.norm (HSub.hSub n.2 x))
      ⊢ LE.le (Norm.norm ((HSub.hSub (f' n.1 x) (g' x)) (HSub.hSub n.2 x))) (HMul.hM …
    -/
    exact (f' n.1 x - g' x).le_opNorm (n.2 - x)
    /-
      🎉 no goals
    -/


theorem hasFDerivAt_of_tendstoLocallyUniformlyOn [NeBot l] {s : Set E} (hs : IsOpen s)
    (hf' : TendstoLocallyUniformlyOn f' g' l s) (hf : ∀ n, ∀ x ∈ s, HasFDerivAt (f n) (f' n x) x)
    (hfg : ∀ x ∈ s, Tendsto (fun n => f n x) l (𝓝 (g x))) (hx : x ∈ s) :
    HasFDerivAt g (g' x) x := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    s : Set E
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : ∀ (n : ι) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    ⊢ HasFDerivAt g (g' x) x
  -/
  have h1 : s ∈ 𝓝 x := hs.mem_nhds hx
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    s : Set E
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : ∀ (n : ι) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    h1 : Membership.mem (nhds x) s
    ⊢ HasFDerivAt g (g' x) x
  -/
  have h3 : Set.univ ×ˢ s ∈ l ×ˢ 𝓝 x := by simp only [h1, prod_mem_prod_iff, univ_mem, and_self_iff]
  have h4 : ∀ᶠ n : ι × E in l ×ˢ 𝓝 x, HasFDerivAt (f n.1) (f' n.1 n.2) n.2 :=
    eventually_of_mem h3 fun ⟨n, z⟩ ⟨_, hz⟩ => hf n z hz
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    s : Set E
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : ∀ (n : ι) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    h1 : Membership.mem (nhds x) s
    h3 : Membership.mem (SProd.sprod l (nhds x)) (SProd.sprod Set.univ s)
    h4 : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    ⊢ HasFDerivAt g (g' x) x
  -/
  refine hasFDerivAt_of_tendstoUniformlyOnFilter ?_ h4 (eventually_of_mem h1 hfg)
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    s : Set E
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : ∀ (n : ι) (x : E), Membership.mem s x → HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    h1 : Membership.mem (nhds x) s
    h3 : Membership.mem (SProd.sprod l (nhds x)) (SProd.sprod Set.univ s)
    h4 : Filter.Eventually (fun n => HasFDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd. …
    ⊢ TendstoUniformlyOnFilter f' g' l (nhds x)
  -/
  simpa [IsOpen.nhdsWithin_eq hs hx] using tendstoLocallyUniformlyOn_iff_filter.mp hf' x hx
  /-
    🎉 no goals
  -/


/-- A slight variant of `hasFDerivAt_of_tendstoLocallyUniformlyOn` with the assumption stated
in terms of `DifferentiableOn` rather than `HasFDerivAt`. This makes a few proofs nicer in
complex analysis where holomorphicity is assumed but the derivative is not known a priori. -/
theorem hasFDerivAt_of_tendsto_locally_uniformly_on' [NeBot l] {s : Set E} (hs : IsOpen s)
    (hf' : TendstoLocallyUniformlyOn (fderiv 𝕜 ∘ f) g' l s) (hf : ∀ n, DifferentiableOn 𝕜 (f n) s)
    (hfg : ∀ x ∈ s, Tendsto (fun n => f n x) l (𝓝 (g x))) (hx : x ∈ s) :
    HasFDerivAt g (g' x) x := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    s : Set E
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn (Function.comp (fderiv 𝕜) f) g' l s
    hf : ∀ (n : ι), DifferentiableOn 𝕜 (f n) s
    hfg : ∀ (x : E), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    ⊢ HasFDerivAt g (g' x) x
  -/
  refine hasFDerivAt_of_tendstoLocallyUniformlyOn hs hf' (fun n z hz => ?_) hfg hx
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    x : E
    inst✝ : l.NeBot
    s : Set E
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn (Function.comp (fderiv 𝕜) f) g' l s
    hf : ∀ (n : ι), DifferentiableOn 𝕜 (f n) s
    hfg : ∀ (x : E), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    n : ι
    z : E
    hz : Membership.mem s z
    ⊢ HasFDerivAt (f n) (Function.comp (fderiv 𝕜) f n z) z
  -/
  exact ((hf n z hz).differentiableAt (hs.mem_nhds hz)).hasFDerivAt
  /-
    🎉 no goals
  -/


/-- `(d/dx) lim_{n → ∞} f n x = lim_{n → ∞} f' n x` when the `f' n` converge
_uniformly_ to their limit on an open set containing `x`. -/
theorem hasFDerivAt_of_tendstoUniformlyOn [NeBot l] {s : Set E} (hs : IsOpen s)
    (hf' : TendstoUniformlyOn f' g' l s)
    (hf : ∀ n : ι, ∀ x : E, x ∈ s → HasFDerivAt (f n) (f' n x) x)
    (hfg : ∀ x : E, x ∈ s → Tendsto (fun n => f n x) l (𝓝 (g x))) (hx : x ∈ s) :
    HasFDerivAt g (g' x) x :=
  hasFDerivAt_of_tendstoLocallyUniformlyOn hs hf'.tendstoLocallyUniformlyOn hf hfg hx


/-- `(d/dx) lim_{n → ∞} f n x = lim_{n → ∞} f' n x` when the `f' n` converge
_uniformly_ to their limit. -/
theorem hasFDerivAt_of_tendstoUniformly [NeBot l] (hf' : TendstoUniformly f' g' l)
    (hf : ∀ n : ι, ∀ x : E, HasFDerivAt (f n) (f' n x) x)
    (hfg : ∀ x : E, Tendsto (fun n => f n x) l (𝓝 (g x))) (x : E) : HasFDerivAt g (g' x) x := by
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    inst✝ : l.NeBot
    hf' : TendstoUniformly f' g' l
    hf : ∀ (n : ι) (x : E), HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : E
    ⊢ HasFDerivAt g (g' x) x
  -/
  have hf : ∀ n : ι, ∀ x : E, x ∈ Set.univ → HasFDerivAt (f n) (f' n x) x := by simp [hf]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    inst✝ : l.NeBot
    hf' : TendstoUniformly f' g' l
    hf✝ : ∀ (n : ι) (x : E), HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : E
    hf : ∀ (n : ι) (x : E), Membership.mem Set.univ x → HasFDerivAt (f n) (f' n x) x
    ⊢ HasFDerivAt g (g' x) x
  -/
  have hfg : ∀ x : E, x ∈ Set.univ → Tendsto (fun n => f n x) l (𝓝 (g x)) := by simp [hfg]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    inst✝ : l.NeBot
    hf' : TendstoUniformly f' g' l
    hf✝ : ∀ (n : ι) (x : E), HasFDerivAt (f n) (f' n x) x
    hfg✝ : ∀ (x : E), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : E
    hf : ∀ (n : ι) (x : E), Membership.mem Set.univ x → HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Membership.mem Set.univ x → Filter.Tendsto (fun n => f n x) l …
    ⊢ HasFDerivAt g (g' x) x
  -/
  have hf' : TendstoUniformlyOn f' g' l Set.univ := by rwa [tendstoUniformlyOn_univ]
  /-
    ι : Type u_1
    l : Filter ι
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f : ι → E → G
    g : E → G
    f' : ι → E → ContinuousLinearMap (RingHom.id 𝕜) E G
    g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    inst✝ : l.NeBot
    hf'✝ : TendstoUniformly f' g' l
    hf✝ : ∀ (n : ι) (x : E), HasFDerivAt (f n) (f' n x) x
    hfg✝ : ∀ (x : E), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : E
    hf : ∀ (n : ι) (x : E), Membership.mem Set.univ x → HasFDerivAt (f n) (f' n x) x
    hfg : ∀ (x : E), Membership.mem Set.univ x → Filter.Tendsto (fun n => f n x) l …
    hf' : TendstoUniformlyOn f' g' l Set.univ
    ⊢ HasFDerivAt g (g' x) x
  -/
  exact hasFDerivAt_of_tendstoUniformlyOn isOpen_univ hf' hf hfg (Set.mem_univ x)
  /-
    🎉 no goals
  -/


/-- If our derivatives converge uniformly, then the Fréchet derivatives converge uniformly -/
theorem UniformCauchySeqOnFilter.one_smulRight {l' : Filter 𝕜}
    (hf' : UniformCauchySeqOnFilter f' l l') :
    UniformCauchySeqOnFilter (fun n => fun z => (1 : 𝕜 →L[𝕜] 𝕜).smulRight (f' n z)) l l' := by
  -- The tricky part of this proof is that operator norms are written in terms of `≤` whereas
  -- metrics are written in terms of `<`. So we need to shrink `ε` utilizing the archimedean
  -- property of `ℝ`
  rw [SeminormedAddGroup.uniformCauchySeqOnFilter_iff_tendstoUniformlyOnFilter_zero,
    Metric.tendstoUniformlyOnFilter_iff] at hf' ⊢
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (0 n. …
  -/
  intro ε hε
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (ContinuousL …
  -/
  obtain ⟨q, hq, hq'⟩ := exists_between hε.lt
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    ⊢ Filter.Eventually (fun n => LT.lt (Dist.dist (0 n.2) (HSub.hSub (ContinuousL …
  -/
  apply (hf' q hq).mono
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    ⊢ ∀ (x : Prod (Prod ι ι) 𝕜), LT.lt (Dist.dist (0 x.2) (HSub.hSub (f' x.1.1 x.2 …
  -/
  intro n hn
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    n : Prod (Prod ι ι) 𝕜
    hn : LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) q
    ⊢ LT.lt (Dist.dist (0 n.2) (HSub.hSub (ContinuousLinearMap.smulRight 1 (f' n.1 …
  -/
  refine lt_of_le_of_lt ?_ hq'
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    n : Prod (Prod ι ι) 𝕜
    hn : LT.lt (Dist.dist (0 n.2) (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) q
    ⊢ LE.le (Dist.dist (0 n.2) (HSub.hSub (ContinuousLinearMap.smulRight 1 (f' n.1 …
  -/
  simp only [dist_eq_norm, Pi.zero_apply, zero_sub, norm_neg] at hn ⊢
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    n : Prod (Prod ι ι) 𝕜
    hn : LT.lt (Norm.norm (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) q
    ⊢ LE.le (Norm.norm (HSub.hSub (ContinuousLinearMap.smulRight 1 (f' n.1.1 n.2)) …
  -/
  refine ContinuousLinearMap.opNorm_le_bound _ hq.le ?_
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    n : Prod (Prod ι ι) 𝕜
    hn : LT.lt (Norm.norm (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) q
    ⊢ ∀ (x : 𝕜), LE.le (Norm.norm ((HSub.hSub (ContinuousLinearMap.smulRight 1 (f' …
  -/
  intro z
  simp only [ContinuousLinearMap.coe_sub', Pi.sub_apply, ContinuousLinearMap.smulRight_apply,
    ContinuousLinearMap.one_apply]
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    n : Prod (Prod ι ι) 𝕜
    hn : LT.lt (Norm.norm (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) q
    z : 𝕜
    ⊢ LE.le (Norm.norm (HSub.hSub (HSMul.hSMul z (f' n.1.1 n.2)) (HSMul.hSMul z (f …
  -/
  rw [← smul_sub, norm_smul, mul_comm]
  /-
    case intro.intro
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f' : ι → 𝕜 → G
    l' : Filter 𝕜
    hf' : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist ( …
    ε : Real
    hε : GT.gt ε 0
    q : Real
    hq : LT.lt 0 q
    hq' : LT.lt q ε
    n : Prod (Prod ι ι) 𝕜
    hn : LT.lt (Norm.norm (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) q
    z : 𝕜
    ⊢ LE.le (HMul.hMul (Norm.norm (HSub.hSub (f' n.1.1 n.2) (f' n.1.2 n.2))) (Norm …
  -/
  gcongr
  /-
    🎉 no goals
  -/


theorem uniformCauchySeqOnFilter_of_deriv (hf' : UniformCauchySeqOnFilter f' l (𝓝 x))
    (hf : ∀ᶠ n : ι × 𝕜 in l ×ˢ 𝓝 x, HasDerivAt (f n.1) (f' n.1 n.2) n.2)
    (hfg : Cauchy (map (fun n => f n x) l)) : UniformCauchySeqOnFilter f l (𝓝 x) := by
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f f' : ι → 𝕜 → G
    x : 𝕜
    inst✝ : IsRCLikeNormedField 𝕜
    hf' : UniformCauchySeqOnFilter f' l (nhds x)
    hf : Filter.Eventually (fun n => HasDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd.s …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    ⊢ UniformCauchySeqOnFilter f l (nhds x)
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at hf
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f f' : ι → 𝕜 → G
    x : 𝕜
    inst✝ : IsRCLikeNormedField 𝕜
    hf' : UniformCauchySeqOnFilter f' l (nhds x)
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (ContinuousLinearMap.smul …
    ⊢ UniformCauchySeqOnFilter f l (nhds x)
  -/
  exact uniformCauchySeqOnFilter_of_fderiv hf'.one_smulRight hf hfg
  /-
    🎉 no goals
  -/


theorem uniformCauchySeqOn_ball_of_deriv {r : ℝ} (hf' : UniformCauchySeqOn f' l (Metric.ball x r))
    (hf : ∀ n : ι, ∀ y : 𝕜, y ∈ Metric.ball x r → HasDerivAt (f n) (f' n y) y)
    (hfg : Cauchy (map (fun n => f n x) l)) : UniformCauchySeqOn f l (Metric.ball x r) := by
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f f' : ι → 𝕜 → G
    x : 𝕜
    inst✝ : IsRCLikeNormedField 𝕜
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hf : ∀ (n : ι) (y : 𝕜), Membership.mem (Metric.ball x r) y → HasDerivAt (f n)  …
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at hf
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f f' : ι → 𝕜 → G
    x : 𝕜
    inst✝ : IsRCLikeNormedField 𝕜
    r : Real
    hf' : UniformCauchySeqOn f' l (Metric.ball x r)
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    hf : ∀ (n : ι) (y : 𝕜), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  rw [uniformCauchySeqOn_iff_uniformCauchySeqOnFilter] at hf'
  have hf' :
    UniformCauchySeqOn (fun n => fun z => (1 : 𝕜 →L[𝕜] 𝕜).smulRight (f' n z)) l
      (Metric.ball x r) := by
    rw [uniformCauchySeqOn_iff_uniformCauchySeqOnFilter]
    exact hf'.one_smulRight
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    f f' : ι → 𝕜 → G
    x : 𝕜
    inst✝ : IsRCLikeNormedField 𝕜
    r : Real
    hf'✝ : UniformCauchySeqOnFilter f' l (Filter.principal (Metric.ball x r))
    hfg : Cauchy (Filter.map (fun n => f n x) l)
    hf : ∀ (n : ι) (y : 𝕜), Membership.mem (Metric.ball x r) y → HasFDerivAt (f n) …
    hf' : UniformCauchySeqOn (fun n z => ContinuousLinearMap.smulRight 1 (f' n z)) …
    ⊢ UniformCauchySeqOn f l (Metric.ball x r)
  -/
  exact uniformCauchySeqOn_ball_of_fderiv hf' hf hfg
  /-
    🎉 no goals
  -/


theorem hasDerivAt_of_tendstoUniformlyOnFilter [NeBot l]
    (hf' : TendstoUniformlyOnFilter f' g' l (𝓝 x))
    (hf : ∀ᶠ n : ι × 𝕜 in l ×ˢ 𝓝 x, HasDerivAt (f n.1) (f' n.1 n.2) n.2)
    (hfg : ∀ᶠ y in 𝓝 x, Tendsto (fun n => f n y) l (𝓝 (g y))) : HasDerivAt g (g' x) x := by
  -- The first part of the proof rewrites `hf` and the goal to be functions so that Lean
  -- can recognize them when we apply `hasFDerivAt_of_tendstoUniformlyOnFilter`
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd.s …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    ⊢ HasDerivAt g (g' x) x
  -/
  let F' n z := (1 : 𝕜 →L[𝕜] 𝕜).smulRight (f' n z)
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd.s …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    F' : ι → 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) 𝕜 G := fun n z => ContinuousLi …
    ⊢ HasDerivAt g (g' x) x
  -/
  let G' z := (1 : 𝕜 →L[𝕜] 𝕜).smulRight (g' z)
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf' : TendstoUniformlyOnFilter f' g' l (nhds x)
    hf : Filter.Eventually (fun n => HasDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd.s …
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    F' : ι → 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) 𝕜 G := fun n z => ContinuousLi …
    G' : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) 𝕜 G := fun z => ContinuousLinearMa …
    ⊢ HasDerivAt g (g' x) x
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at hf ⊢
  -- Now we need to rewrite hf' in terms of `ContinuousLinearMap`s. The tricky part is that
  -- operator norms are written in terms of `≤` whereas metrics are written in terms of `<`. So we
  -- need to shrink `ε` utilizing the archimedean property of `ℝ`
  have hf' : TendstoUniformlyOnFilter F' G' l (𝓝 x) := by
    rw [Metric.tendstoUniformlyOnFilter_iff] at hf' ⊢
    intro ε hε
    obtain ⟨q, hq, hq'⟩ := exists_between hε.lt
    apply (hf' q hq).mono
    intro n hn
    refine lt_of_le_of_lt ?_ hq'
    simp only [dist_eq_norm] at hn ⊢
    refine ContinuousLinearMap.opNorm_le_bound _ hq.le ?_
    intro z
    simp only [F', G', ContinuousLinearMap.coe_sub', Pi.sub_apply,
      ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply]
    rw [← smul_sub, norm_smul, mul_comm]
    gcongr
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf'✝ : TendstoUniformlyOnFilter f' g' l (nhds x)
    hfg : Filter.Eventually (fun y => Filter.Tendsto (fun n => f n y) l (nhds (g y …
    F' : ι → 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) 𝕜 G := fun n z => ContinuousLi …
    G' : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) 𝕜 G := fun z => ContinuousLinearMa …
    hf : Filter.Eventually (fun n => HasFDerivAt (f n.1) (ContinuousLinearMap.smul …
    hf' : TendstoUniformlyOnFilter F' G' l (nhds x)
    ⊢ HasFDerivAt g (ContinuousLinearMap.smulRight 1 (g' x)) x
  -/
  exact hasFDerivAt_of_tendstoUniformlyOnFilter hf' hf hfg
  /-
    🎉 no goals
  -/


theorem hasDerivAt_of_tendstoLocallyUniformlyOn [NeBot l] {s : Set 𝕜} (hs : IsOpen s)
    (hf' : TendstoLocallyUniformlyOn f' g' l s)
    (hf : ∀ᶠ n in l, ∀ x ∈ s, HasDerivAt (f n) (f' n x) x)
    (hfg : ∀ x ∈ s, Tendsto (fun n => f n x) l (𝓝 (g x))) (hx : x ∈ s) : HasDerivAt g (g' x) x := by
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    s : Set 𝕜
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem s x → HasDerivAt (f …
    hfg : ∀ (x : 𝕜), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    ⊢ HasDerivAt g (g' x) x
  -/
  have h1 : s ∈ 𝓝 x := hs.mem_nhds hx
  have h2 : ∀ᶠ n : ι × 𝕜 in l ×ˢ 𝓝 x, HasDerivAt (f n.1) (f' n.1 n.2) n.2 :=
    eventually_prod_iff.2 ⟨_, hf, fun x => x ∈ s, h1, fun {n} => id⟩
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    s : Set 𝕜
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem s x → HasDerivAt (f …
    hfg : ∀ (x : 𝕜), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    h1 : Membership.mem (nhds x) s
    h2 : Filter.Eventually (fun n => HasDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd.s …
    ⊢ HasDerivAt g (g' x) x
  -/
  refine hasDerivAt_of_tendstoUniformlyOnFilter ?_ h2 (eventually_of_mem h1 hfg)
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    s : Set 𝕜
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn f' g' l s
    hf : Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem s x → HasDerivAt (f …
    hfg : ∀ (x : 𝕜), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    h1 : Membership.mem (nhds x) s
    h2 : Filter.Eventually (fun n => HasDerivAt (f n.1) (f' n.1 n.2) n.2) (SProd.s …
    ⊢ TendstoUniformlyOnFilter f' g' l (nhds x)
  -/
  simpa [IsOpen.nhdsWithin_eq hs hx] using tendstoLocallyUniformlyOn_iff_filter.mp hf' x hx
  /-
    🎉 no goals
  -/


/-- A slight variant of `hasDerivAt_of_tendstoLocallyUniformlyOn` with the assumption stated in
terms of `DifferentiableOn` rather than `HasDerivAt`. This makes a few proofs nicer in complex
analysis where holomorphicity is assumed but the derivative is not known a priori. -/
theorem hasDerivAt_of_tendsto_locally_uniformly_on' [NeBot l] {s : Set 𝕜} (hs : IsOpen s)
    (hf' : TendstoLocallyUniformlyOn (deriv ∘ f) g' l s)
    (hf : ∀ᶠ n in l, DifferentiableOn 𝕜 (f n) s)
    (hfg : ∀ x ∈ s, Tendsto (fun n => f n x) l (𝓝 (g x))) (hx : x ∈ s) : HasDerivAt g (g' x) x := by
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    s : Set 𝕜
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn (Function.comp deriv f) g' l s
    hf : Filter.Eventually (fun n => DifferentiableOn 𝕜 (f n) s) l
    hfg : ∀ (x : 𝕜), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    ⊢ HasDerivAt g (g' x) x
  -/
  refine hasDerivAt_of_tendstoLocallyUniformlyOn hs hf' ?_ hfg hx
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g g' : 𝕜 → G
    x : 𝕜
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    s : Set 𝕜
    hs : IsOpen s
    hf' : TendstoLocallyUniformlyOn (Function.comp deriv f) g' l s
    hf : Filter.Eventually (fun n => DifferentiableOn 𝕜 (f n) s) l
    hfg : ∀ (x : 𝕜), Membership.mem s x → Filter.Tendsto (fun n => f n x) l (nhds  …
    hx : Membership.mem s x
    ⊢ Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem s x → HasDerivAt (f n) …
  -/
  filter_upwards [hf] with n h z hz using ((h z hz).differentiableAt (hs.mem_nhds hz)).hasDerivAt
  /-
    🎉 no goals
  -/


theorem hasDerivAt_of_tendstoUniformlyOn [NeBot l] {s : Set 𝕜} (hs : IsOpen s)
    (hf' : TendstoUniformlyOn f' g' l s)
    (hf : ∀ᶠ n in l, ∀ x : 𝕜, x ∈ s → HasDerivAt (f n) (f' n x) x)
    (hfg : ∀ x : 𝕜, x ∈ s → Tendsto (fun n => f n x) l (𝓝 (g x))) (hx : x ∈ s) :
    HasDerivAt g (g' x) x :=
  hasDerivAt_of_tendstoLocallyUniformlyOn hs hf'.tendstoLocallyUniformlyOn hf hfg hx


theorem hasDerivAt_of_tendstoUniformly [NeBot l] (hf' : TendstoUniformly f' g' l)
    (hf : ∀ᶠ n in l, ∀ x : 𝕜, HasDerivAt (f n) (f' n x) x)
    (hfg : ∀ x : 𝕜, Tendsto (fun n => f n x) l (𝓝 (g x))) (x : 𝕜) : HasDerivAt g (g' x) x := by
  have hf : ∀ᶠ n in l, ∀ x : 𝕜, x ∈ Set.univ → HasDerivAt (f n) (f' n x) x := by
    filter_upwards [hf] with n h x _ using h x
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf' : TendstoUniformly f' g' l
    hf✝ : Filter.Eventually (fun n => ∀ (x : 𝕜), HasDerivAt (f n) (f' n x) x) l
    hfg : ∀ (x : 𝕜), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : 𝕜
    hf : Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem Set.univ x → HasDer …
    ⊢ HasDerivAt g (g' x) x
  -/
  have hfg : ∀ x : 𝕜, x ∈ Set.univ → Tendsto (fun n => f n x) l (𝓝 (g x)) := by simp [hfg]
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf' : TendstoUniformly f' g' l
    hf✝ : Filter.Eventually (fun n => ∀ (x : 𝕜), HasDerivAt (f n) (f' n x) x) l
    hfg✝ : ∀ (x : 𝕜), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : 𝕜
    hf : Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem Set.univ x → HasDer …
    hfg : ∀ (x : 𝕜), Membership.mem Set.univ x → Filter.Tendsto (fun n => f n x) l …
    ⊢ HasDerivAt g (g' x) x
  -/
  have hf' : TendstoUniformlyOn f' g' l Set.univ := by rwa [tendstoUniformlyOn_univ]
  /-
    ι : Type u_1
    l : Filter ι
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_3
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    f : ι → 𝕜 → G
    g : 𝕜 → G
    f' : ι → 𝕜 → G
    g' : 𝕜 → G
    inst✝¹ : IsRCLikeNormedField 𝕜
    inst✝ : l.NeBot
    hf'✝ : TendstoUniformly f' g' l
    hf✝ : Filter.Eventually (fun n => ∀ (x : 𝕜), HasDerivAt (f n) (f' n x) x) l
    hfg✝ : ∀ (x : 𝕜), Filter.Tendsto (fun n => f n x) l (nhds (g x))
    x : 𝕜
    hf : Filter.Eventually (fun n => ∀ (x : 𝕜), Membership.mem Set.univ x → HasDer …
    hfg : ∀ (x : 𝕜), Membership.mem Set.univ x → Filter.Tendsto (fun n => f n x) l …
    hf' : TendstoUniformlyOn f' g' l Set.univ
    ⊢ HasDerivAt g (g' x) x
  -/
  exact hasDerivAt_of_tendstoUniformlyOn isOpen_univ hf' hf hfg (Set.mem_univ x)
  /-
    🎉 no goals
  -/


