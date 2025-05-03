theorem TietzeExtension.of_tvs (𝕜 : Type v) [NontriviallyNormedField 𝕜] {E : Type w}
    [AddCommGroup E] [Module 𝕜 E] [TopologicalSpace E] [TopologicalAddGroup E] [ContinuousSMul 𝕜 E]
    [T2Space E] [FiniteDimensional 𝕜 E] [CompleteSpace 𝕜] [TietzeExtension.{u, v} 𝕜] :
    TietzeExtension.{u, w} E :=
  Basis.ofVectorSpace 𝕜 E |>.equivFun.toContinuousLinearEquiv.toHomeomorph |> .of_homeo


instance Complex.instTietzeExtension : TietzeExtension ℂ :=
  TietzeExtension.of_tvs ℝ


instance (priority := 900) RCLike.instTietzeExtension {𝕜 : Type*} [RCLike 𝕜] :
    TietzeExtension 𝕜 := TietzeExtension.of_tvs ℝ


instance RCLike.instTietzeExtensionTVS {𝕜 : Type v} [RCLike 𝕜] {E : Type w}
    [AddCommGroup E] [Module 𝕜 E] [TopologicalSpace E] [TopologicalAddGroup E]
    [ContinuousSMul 𝕜 E] [T2Space E] [FiniteDimensional 𝕜 E] :
    TietzeExtension.{u, w} E :=
  TietzeExtension.of_tvs 𝕜


instance Set.instTietzeExtensionUnitBall {𝕜 : Type v} [RCLike 𝕜] {E : Type w}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E] :
    TietzeExtension.{u, w} (Metric.ball (0 : E) 1) :=
  have : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 E
  .of_homeo Homeomorph.unitBall.symm


instance Set.instTietzeExtensionUnitClosedBall {𝕜 : Type v} [RCLike 𝕜] {E : Type w}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E] :
    TietzeExtension.{u, w} (Metric.closedBall (0 : E) 1) := by
  /-
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ TietzeExtension ↑(Metric.closedBall 0 1)
  -/
  have : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 E
  /-
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    this : NormedSpace Real E
    ⊢ TietzeExtension ↑(Metric.closedBall 0 1)
  -/
  have : IsScalarTower ℝ 𝕜 E := Real.isScalarTower
  -- I didn't find this retract in Mathlib.
  /-
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    this✝ : NormedSpace Real E
    this : IsScalarTower Real 𝕜 E
    ⊢ TietzeExtension ↑(Metric.closedBall 0 1)
  -/
  let g : E → E := fun x ↦ ‖x‖⁻¹ • x
  classical
  suffices this : Continuous (piecewise (Metric.closedBall 0 1) id g) by
    refine .of_retract ⟨Subtype.val, by fun_prop⟩ ⟨_, this.codRestrict fun x ↦ ?_⟩ ?_
    · by_cases hx : x ∈ Metric.closedBall 0 1
      · simpa [piecewise_eq_of_mem (hi := hx)] using hx
      · simp only [g, piecewise_eq_of_not_mem (hi := hx), RCLike.real_smul_eq_coe_smul (K := 𝕜)]
        by_cases hx' : x = 0 <;> simp [hx']
    · ext x
      simp [piecewise_eq_of_mem (hi := x.property)]
  refine continuous_piecewise (fun x hx ↦ ?_) continuousOn_id ?_
  · replace hx : ‖x‖ = 1 := by simpa [frontier_closedBall (0 : E) one_ne_zero] using hx
    simp [g, hx]
  · refine continuousOn_id.norm.inv₀ ?_ |>.smul continuousOn_id
    simp only [closure_compl, interior_closedBall (0 : E) one_ne_zero, mem_compl_iff,
      Metric.mem_ball, dist_zero_right, not_lt, id_eq, ne_eq, norm_eq_zero]
    exact fun x hx ↦ norm_pos_iff.mp <| one_pos.trans_le hx


theorem Metric.instTietzeExtensionBall {𝕜 : Type v} [RCLike 𝕜] {E : Type w}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E] {r : ℝ} (hr : 0 < r) :
    TietzeExtension.{u, w} (Metric.ball (0 : E) r) :=
  have : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 E
  .of_homeo <| show (Metric.ball (0 : E) r) ≃ₜ (Metric.ball (0 : E) 1) from
    PartialHomeomorph.unitBallBall (0 : E) r hr |>.toHomeomorphSourceTarget.symm


theorem Metric.instTietzeExtensionClosedBall (𝕜 : Type v) [RCLike 𝕜] {E : Type w}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E] (y : E) {r : ℝ} (hr : 0 < r) :
    TietzeExtension.{u, w} (Metric.closedBall y r) :=
  .of_homeo <| by
    /-
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type w
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : FiniteDimensional 𝕜 E
      y : E
      r : Real
      hr : LT.lt 0 r
      ⊢ Homeomorph ↑(Metric.closedBall y r) ?m.56665
    -/
    show (Metric.closedBall y r) ≃ₜ (Metric.closedBall (0 : E) 1)
    /-
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type w
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : FiniteDimensional 𝕜 E
      y : E
      r : Real
      hr : LT.lt 0 r
      ⊢ Homeomorph ↑(Metric.closedBall y r) ↑(Metric.closedBall 0 1)
    -/
    symm
    /-
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type w
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : FiniteDimensional 𝕜 E
      y : E
      r : Real
      hr : LT.lt 0 r
      ⊢ Homeomorph ↑(Metric.closedBall 0 1) ↑(Metric.closedBall y r)
    -/
    apply (DilationEquiv.smulTorsor y (k := (r : 𝕜)) <| by exact_mod_cast hr.ne').toHomeomorph.sets
    /-
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type w
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : FiniteDimensional 𝕜 E
      y : E
      r : Real
      hr : LT.lt 0 r
      ⊢ Eq (Metric.closedBall 0 1) (Set.preimage (⇑(DilationEquiv.smulTorsor y ⋯).to …
    -/
    ext x
    simp only [mem_closedBall, dist_zero_right, DilationEquiv.coe_toHomeomorph, Set.mem_preimage,
      DilationEquiv.smulTorsor_apply, vadd_eq_add, dist_add_self_left, norm_smul,
      RCLike.norm_ofReal, abs_of_nonneg hr.le]
    /-
      case h
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type w
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : FiniteDimensional 𝕜 E
      y : E
      r : Real
      hr : LT.lt 0 r
      x : E
      ⊢ Iff (LE.le (Norm.norm x) 1) (LE.le (HMul.hMul r (Norm.norm x)) r)
    -/
    exact (mul_le_iff_le_one_right hr).symm
    /-
      🎉 no goals
    -/


include 𝕜 hs in
/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version with a closed
embedding and bundled composition. If `e : C(X, Y)` is a closed embedding of a topological space
into a normal topological space and `f : X →ᵇ ℝ` is a bounded continuous function, then there exists
a bounded continuous function `g : Y →ᵇ ℝ` of the same norm such that `g ∘ e = f`. -/
theorem exists_norm_eq_restrict_eq (f : s →ᵇ E) :
    ∃ g : X →ᵇ E, ‖g‖ = ‖f‖ ∧ g.restrict s = f := by
  /-
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  by_cases hf : ‖f‖ = 0; · exact ⟨0, by aesop⟩
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  have := Metric.instTietzeExtensionClosedBall.{u, v} 𝕜 (0 : E) (by aesop : 0 < ‖f‖)
  /-
    case neg
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  have hf' x : f x ∈ Metric.closedBall 0 ‖f‖ := by simpa using f.norm_coe_le_norm x
  /-
    case neg
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  obtain ⟨g, hg_mem, hg⟩ := (f : C(s, E)).exists_forall_mem_restrict_eq hs hf'
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg_mem : ∀ (x : X), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (g x)
    hg : Eq (ContinuousMap.restrict s g) ↑f
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  simp only [Metric.mem_closedBall, dist_zero_right] at hg_mem
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  let g' : X →ᵇ E := .ofNormedAddCommGroup g (map_continuous g) ‖f‖ hg_mem
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    g' : BoundedContinuousFunction X E := BoundedContinuousFunction.ofNormedAddCom …
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.restrict s) f)
  -/
  refine ⟨g', ?_, by ext x; congrm($(hg) x)⟩
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    g' : BoundedContinuousFunction X E := BoundedContinuousFunction.ofNormedAddCom …
    ⊢ Eq (Norm.norm g') (Norm.norm f)
  -/
  apply le_antisymm ((g'.norm_le <| by positivity).mpr hg_mem)
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    g' : BoundedContinuousFunction X E := BoundedContinuousFunction.ofNormedAddCom …
    ⊢ LE.le (Norm.norm f) (Norm.norm g')
  -/
  refine (f.norm_le <| by positivity).mpr fun x ↦ ?_
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    g' : BoundedContinuousFunction X E := BoundedContinuousFunction.ofNormedAddCom …
    x : ↑s
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm g')
  -/
  have hx : f x = g' x := by simpa using congr($(hg) x).symm
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    g' : BoundedContinuousFunction X E := BoundedContinuousFunction.ofNormedAddCom …
    x : ↑s
    hx : Eq (f x) (g' ↑x)
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm g')
  -/
  rw [hx]
  /-
    case neg.intro.intro
    X : Type u
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormalSpace X
    s : Set X
    hs : IsClosed s
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type w
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    f : BoundedContinuousFunction (↑s) E
    hf : Not (Eq (Norm.norm f) 0)
    this : TietzeExtension ↑(Metric.closedBall 0 (Norm.norm f))
    hf' : ∀ (x : ↑s), Membership.mem (Metric.closedBall 0 (Norm.norm f)) (f x)
    g : ContinuousMap X E
    hg : Eq (ContinuousMap.restrict s g) ↑f
    hg_mem : ∀ (x : X), LE.le (Norm.norm (g x)) (Norm.norm f)
    g' : BoundedContinuousFunction X E := BoundedContinuousFunction.ofNormedAddCom …
    x : ↑s
    hx : Eq (f x) (g' ↑x)
    ⊢ LE.le (Norm.norm (g' ↑x)) (Norm.norm g')
  -/
  exact g'.norm_le (norm_nonneg g') |>.mp le_rfl x
  /-
    🎉 no goals
  -/


