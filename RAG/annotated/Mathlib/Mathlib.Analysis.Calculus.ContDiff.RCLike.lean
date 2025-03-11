/-- If a function has a Taylor series at order at least 1, then at points in the interior of the
    domain of definition, the term of order 1 of this series is a strict derivative of `f`. -/
theorem HasFTaylorSeriesUpToOn.hasStrictFDerivAt {n : WithTop ℕ∞}
    {s : Set E'} {f : E' → F'} {x : E'}
    {p : E' → FormalMultilinearSeries 𝕂 E' F'} (hf : HasFTaylorSeriesUpToOn n f p s) (hn : 1 ≤ n)
    (hs : s ∈ 𝓝 x) : HasStrictFDerivAt f ((continuousMultilinearCurryFin1 𝕂 E' F') (p x 1)) x :=
  hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt (hf.eventually_hasFDerivAt hn hs) <|
    (continuousMultilinearCurryFin1 𝕂 E' F').continuousAt.comp <| (hf.cont 1 hn).continuousAt hs


/-- If a function is `C^n` with `1 ≤ n` around a point, and its derivative at that point is given to
us as `f'`, then `f'` is also a strict derivative. -/
theorem ContDiffAt.hasStrictFDerivAt' {f : E' → F'} {f' : E' →L[𝕂] F'} {x : E'}
    (hf : ContDiffAt 𝕂 n f x) (hf' : HasFDerivAt f f' x) (hn : 1 ≤ n) :
    HasStrictFDerivAt f f' x := by
  /-
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    f' : ContinuousLinearMap (RingHom.id 𝕂) E' F'
    x : E'
    hf : ContDiffAt 𝕂 n f x
    hf' : HasFDerivAt f f' x
    hn : LE.le 1 n
    ⊢ HasStrictFDerivAt f f' x
  -/
  rcases hf.of_le hn 1 le_rfl with ⟨u, H, p, hp⟩
  /-
    case intro.intro.intro
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    f' : ContinuousLinearMap (RingHom.id 𝕂) E' F'
    x : E'
    hf : ContDiffAt 𝕂 n f x
    hf' : HasFDerivAt f f' x
    hn : LE.le 1 n
    u : Set E'
    H : Membership.mem (nhdsWithin x (Insert.insert x Set.univ)) u
    p : E' → FormalMultilinearSeries 𝕂 E' F'
    hp : HasFTaylorSeriesUpToOn (↑1) f p u
    ⊢ HasStrictFDerivAt f f' x
  -/
  simp only [nhdsWithin_univ, mem_univ, insert_eq_of_mem] at H
  /-
    case intro.intro.intro
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    f' : ContinuousLinearMap (RingHom.id 𝕂) E' F'
    x : E'
    hf : ContDiffAt 𝕂 n f x
    hf' : HasFDerivAt f f' x
    hn : LE.le 1 n
    u : Set E'
    p : E' → FormalMultilinearSeries 𝕂 E' F'
    hp : HasFTaylorSeriesUpToOn (↑1) f p u
    H : Membership.mem (nhds x) u
    ⊢ HasStrictFDerivAt f f' x
  -/
  have := hp.hasStrictFDerivAt le_rfl H
  /-
    case intro.intro.intro
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    f' : ContinuousLinearMap (RingHom.id 𝕂) E' F'
    x : E'
    hf : ContDiffAt 𝕂 n f x
    hf' : HasFDerivAt f f' x
    hn : LE.le 1 n
    u : Set E'
    p : E' → FormalMultilinearSeries 𝕂 E' F'
    hp : HasFTaylorSeriesUpToOn (↑1) f p u
    H : Membership.mem (nhds x) u
    this : HasStrictFDerivAt f ((continuousMultilinearCurryFin1 𝕂 E' F') (p x 1)) x
    ⊢ HasStrictFDerivAt f f' x
  -/
  rwa [hf'.unique this.hasFDerivAt]
  /-
    🎉 no goals
  -/


/-- If a function is `C^n` with `1 ≤ n` around a point, and its derivative at that point is given to
us as `f'`, then `f'` is also a strict derivative. -/
theorem ContDiffAt.hasStrictDerivAt' {f : 𝕂 → F'} {f' : F'} {x : 𝕂} (hf : ContDiffAt 𝕂 n f x)
    (hf' : HasDerivAt f f' x) (hn : 1 ≤ n) : HasStrictDerivAt f f' x :=
  hf.hasStrictFDerivAt' hf' hn


/-- If a function is `C^n` with `1 ≤ n` around a point, then the derivative of `f` at this point
is also a strict derivative. -/
theorem ContDiffAt.hasStrictFDerivAt {f : E' → F'} {x : E'} (hf : ContDiffAt 𝕂 n f x) (hn : 1 ≤ n) :
    HasStrictFDerivAt f (fderiv 𝕂 f x) x :=
  hf.hasStrictFDerivAt' (hf.differentiableAt hn).hasFDerivAt hn


/-- If a function is `C^n` with `1 ≤ n` around a point, then the derivative of `f` at this point
is also a strict derivative. -/
theorem ContDiffAt.hasStrictDerivAt {f : 𝕂 → F'} {x : 𝕂} (hf : ContDiffAt 𝕂 n f x) (hn : 1 ≤ n) :
    HasStrictDerivAt f (deriv f x) x :=
  (hf.hasStrictFDerivAt hn).hasStrictDerivAt


/-- If a function is `C^n` with `1 ≤ n`, then the derivative of `f` is also a strict derivative. -/
theorem ContDiff.hasStrictFDerivAt {f : E' → F'} {x : E'} (hf : ContDiff 𝕂 n f) (hn : 1 ≤ n) :
    HasStrictFDerivAt f (fderiv 𝕂 f x) x :=
  hf.contDiffAt.hasStrictFDerivAt hn


/-- If a function is `C^n` with `1 ≤ n`, then the derivative of `f` is also a strict derivative. -/
theorem ContDiff.hasStrictDerivAt {f : 𝕂 → F'} {x : 𝕂} (hf : ContDiff 𝕂 n f) (hn : 1 ≤ n) :
    HasStrictDerivAt f (deriv f x) x :=
  hf.contDiffAt.hasStrictDerivAt hn


/-- If `f` has a formal Taylor series `p` up to order `1` on `{x} ∪ s`, where `s` is a convex set,
and `‖p x 1‖₊ < K`, then `f` is `K`-Lipschitz in a neighborhood of `x` within `s`. -/
theorem HasFTaylorSeriesUpToOn.exists_lipschitzOnWith_of_nnnorm_lt {E F : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F}
    {p : E → FormalMultilinearSeries ℝ E F} {s : Set E} {x : E}
    (hf : HasFTaylorSeriesUpToOn 1 f p (insert x s)) (hs : Convex ℝ s) (K : ℝ≥0)
    (hK : ‖p x 1‖₊ < K) : ∃ t ∈ 𝓝[s] x, LipschitzOnWith K f t := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    p : E → FormalMultilinearSeries Real E F
    s : Set E
    x : E
    hf : HasFTaylorSeriesUpToOn 1 f p (Insert.insert x s)
    hs : Convex Real s
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm (p x 1)) K
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (LipschitzOnWith K f …
  -/
  set f' := fun y => continuousMultilinearCurryFin1 ℝ E F (p y 1)
  have hder : ∀ y ∈ s, HasFDerivWithinAt f (f' y) s y := fun y hy =>
    (hf.hasFDerivWithinAt le_rfl (subset_insert x s hy)).mono (subset_insert x s)
  have hcont : ContinuousWithinAt f' s x :=
    (continuousMultilinearCurryFin1 ℝ E F).continuousAt.comp_continuousWithinAt
      ((hf.cont _ le_rfl _ (mem_insert _ _)).mono (subset_insert x s))
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    p : E → FormalMultilinearSeries Real E F
    s : Set E
    x : E
    hf : HasFTaylorSeriesUpToOn 1 f p (Insert.insert x s)
    hs : Convex Real s
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm (p x 1)) K
    f' : E → ContinuousLinearMap (RingHom.id Real) E F := fun y => (continuousMult …
    hder : ∀ (y : E), Membership.mem s y → HasFDerivWithinAt f (f' y) s y
    hcont : ContinuousWithinAt f' s x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (LipschitzOnWith K f …
  -/
  replace hK : ‖f' x‖₊ < K := by simpa only [f', LinearIsometryEquiv.nnnorm_map]
  exact
    hs.exists_nhdsWithin_lipschitzOnWith_of_hasFDerivWithinAt_of_nnnorm_lt
      (eventually_nhdsWithin_iff.2 <| Eventually.of_forall hder) hcont K hK


/-- If `f` has a formal Taylor series `p` up to order `1` on `{x} ∪ s`, where `s` is a convex set,
then `f` is Lipschitz in a neighborhood of `x` within `s`. -/
theorem HasFTaylorSeriesUpToOn.exists_lipschitzOnWith {E F : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F}
    {p : E → FormalMultilinearSeries ℝ E F} {s : Set E} {x : E}
    (hf : HasFTaylorSeriesUpToOn 1 f p (insert x s)) (hs : Convex ℝ s) :
    ∃ K, ∃ t ∈ 𝓝[s] x, LipschitzOnWith K f t :=
  (exists_gt _).imp <| hf.exists_lipschitzOnWith_of_nnnorm_lt hs


/-- If `f` is `C^1` within a convex set `s` at `x`, then it is Lipschitz on a neighborhood of `x`
within `s`. -/
theorem ContDiffWithinAt.exists_lipschitzOnWith {E F : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] {f : E → F} {s : Set E} {x : E}
    (hf : ContDiffWithinAt ℝ 1 f s x) (hs : Convex ℝ s) :
    ∃ K : ℝ≥0, ∃ t ∈ 𝓝[s] x, LipschitzOnWith K f t := by
  /-
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  rcases hf 1 le_rfl with ⟨t, hst, p, hp⟩
  /-
    case intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    t : Set E
    hst : Membership.mem (nhdsWithin x (Insert.insert x s)) t
    p : E → FormalMultilinearSeries Real E F
    hp : HasFTaylorSeriesUpToOn (↑1) f p t
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  rcases Metric.mem_nhdsWithin_iff.mp hst with ⟨ε, ε0, hε⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    t : Set E
    hst : Membership.mem (nhdsWithin x (Insert.insert x s)) t
    p : E → FormalMultilinearSeries Real E F
    hp : HasFTaylorSeriesUpToOn (↑1) f p t
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Inter.inter (Metric.ball x ε) (Insert.insert x s)) t
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  replace hp : HasFTaylorSeriesUpToOn 1 f p (Metric.ball x ε ∩ insert x s) := hp.mono hε
  /-
    case intro.intro.intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    t : Set E
    hst : Membership.mem (nhdsWithin x (Insert.insert x s)) t
    p : E → FormalMultilinearSeries Real E F
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Inter.inter (Metric.ball x ε) (Insert.insert x s)) t
    hp : HasFTaylorSeriesUpToOn 1 f p (Inter.inter (Metric.ball x ε) (Insert.inser …
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  clear hst hε t
  /-
    case intro.intro.intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    p : E → FormalMultilinearSeries Real E F
    ε : Real
    ε0 : GT.gt ε 0
    hp : HasFTaylorSeriesUpToOn 1 f p (Inter.inter (Metric.ball x ε) (Insert.inser …
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  rw [← insert_eq_of_mem (Metric.mem_ball_self ε0), ← insert_inter_distrib] at hp
  /-
    case intro.intro.intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    p : E → FormalMultilinearSeries Real E F
    ε : Real
    ε0 : GT.gt ε 0
    hp : HasFTaylorSeriesUpToOn 1 f p (Insert.insert x (Inter.inter (Metric.ball x …
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  rcases hp.exists_lipschitzOnWith ((convex_ball _ _).inter hs) with ⟨K, t, hst, hft⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    p : E → FormalMultilinearSeries Real E F
    ε : Real
    ε0 : GT.gt ε 0
    hp : HasFTaylorSeriesUpToOn 1 f p (Insert.insert x (Inter.inter (Metric.ball x …
    K : NNReal
    t : Set E
    hst : Membership.mem (nhdsWithin x (Inter.inter (Metric.ball x ε) s)) t
    hft : LipschitzOnWith K f t
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  rw [inter_comm, ← nhdsWithin_restrict' _ (Metric.ball_mem_nhds _ ε0)] at hst
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_4
    F : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    s : Set E
    x : E
    hf : ContDiffWithinAt Real 1 f s x
    hs : Convex Real s
    p : E → FormalMultilinearSeries Real E F
    ε : Real
    ε0 : GT.gt ε 0
    hp : HasFTaylorSeriesUpToOn 1 f p (Insert.insert x (Inter.inter (Metric.ball x …
    K : NNReal
    t : Set E
    hst : Membership.mem (nhdsWithin x s) t
    hft : LipschitzOnWith K f t
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  exact ⟨K, t, hst, hft⟩
  /-
    🎉 no goals
  -/


/-- If `f` is `C^1` at `x` and `K > ‖fderiv 𝕂 f x‖`, then `f` is `K`-Lipschitz in a neighborhood of
`x`. -/
theorem ContDiffAt.exists_lipschitzOnWith_of_nnnorm_lt {f : E' → F'} {x : E'}
    (hf : ContDiffAt 𝕂 1 f x) (K : ℝ≥0) (hK : ‖fderiv 𝕂 f x‖₊ < K) :
    ∃ t ∈ 𝓝 x, LipschitzOnWith K f t :=
  (hf.hasStrictFDerivAt le_rfl).exists_lipschitzOnWith_of_nnnorm_lt K hK


/-- If `f` is `C^1` at `x`, then `f` is Lipschitz in a neighborhood of `x`. -/
theorem ContDiffAt.exists_lipschitzOnWith {f : E' → F'} {x : E'} (hf : ContDiffAt 𝕂 1 f x) :
    ∃ K, ∃ t ∈ 𝓝 x, LipschitzOnWith K f t :=
  (hf.hasStrictFDerivAt le_rfl).exists_lipschitzOnWith


/-- If `f` is `C^1`, it is locally Lipschitz. -/
lemma ContDiff.locallyLipschitz {f : E' → F'} (hf : ContDiff 𝕂 1 f) : LocallyLipschitz f := by
  /-
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf : ContDiff 𝕂 1 f
    ⊢ LocallyLipschitz f
  -/
  intro x
  /-
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf : ContDiff 𝕂 1 f
    x : E'
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhds x) t) (LipschitzOn …
  -/
  rcases hf.contDiffAt.exists_lipschitzOnWith with ⟨K, t, ht, hf⟩
  /-
    case intro.intro.intro
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf✝ : ContDiff 𝕂 1 f
    x : E'
    K : NNReal
    t : Set E'
    ht : Membership.mem (nhds ?m.58984) t
    hf : LipschitzOnWith K f t
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhds x) t) (LipschitzOn …
  -/
  use K, t
  /-
    🎉 no goals
  -/


/-- A `C^1` function with compact support is Lipschitz. -/
theorem ContDiff.lipschitzWith_of_hasCompactSupport {f : E' → F'}
    (hf : HasCompactSupport f) (h'f : ContDiff 𝕂 n f) (hn : 1 ≤ n) :
    ∃ C, LipschitzWith C f := by
  /-
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf : HasCompactSupport f
    h'f : ContDiff 𝕂 n f
    hn : LE.le 1 n
    ⊢ Exists fun C => LipschitzWith C f
  -/
  obtain ⟨C, hC⟩ := (hf.fderiv 𝕂).exists_bound_of_continuous (h'f.continuous_fderiv hn)
  /-
    case intro
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf : HasCompactSupport f
    h'f : ContDiff 𝕂 n f
    hn : LE.le 1 n
    C : Real
    hC : ∀ (x : E'), LE.le (Norm.norm (fderiv 𝕂 f x)) C
    ⊢ Exists fun C => LipschitzWith C f
  -/
  refine ⟨⟨max C 0, le_max_right _ _⟩, ?_⟩
  /-
    case intro
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf : HasCompactSupport f
    h'f : ContDiff 𝕂 n f
    hn : LE.le 1 n
    C : Real
    hC : ∀ (x : E'), LE.le (Norm.norm (fderiv 𝕂 f x)) C
    ⊢ LipschitzWith ⟨Max.max C 0, ⋯⟩ f
  -/
  apply lipschitzWith_of_nnnorm_fderiv_le (h'f.differentiable hn) (fun x ↦ ?_)
  /-
    n : WithTop ENat
    𝕂 : Type u_1
    inst✝⁴ : RCLike 𝕂
    E' : Type u_2
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕂 E'
    F' : Type u_3
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕂 F'
    f : E' → F'
    hf : HasCompactSupport f
    h'f : ContDiff 𝕂 n f
    hn : LE.le 1 n
    C : Real
    hC : ∀ (x : E'), LE.le (Norm.norm (fderiv 𝕂 f x)) C
    x : E'
    ⊢ LE.le (NNNorm.nnnorm (fderiv 𝕂 f x)) ⟨Max.max C 0, ⋯⟩
  -/
  simp [← NNReal.coe_le_coe, hC x]
  /-
    🎉 no goals
  -/


