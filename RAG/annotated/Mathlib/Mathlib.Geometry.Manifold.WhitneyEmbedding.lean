local notation "∞" => (⊤ : ℕ∞)


/-- Smooth embedding of `M` into `(E × ℝ) ^ ι`. -/
def embeddingPiTangent : C^∞⟮I, M; 𝓘(ℝ, ι → E × ℝ), ι → E × ℝ⟯ where
  val x i := (f i x • extChartAt I (f.c i) x, f i x)
  property :=
    contMDiff_pi_space.2 fun i =>
      ((f i).contMDiff_smul contMDiffOn_extChartAt).prod_mk_space (f i).contMDiff


@[local simp]
theorem embeddingPiTangent_coe :
    ⇑f.embeddingPiTangent = fun x i => (f i x • extChartAt I (f.c i) x, f i x) :=
  rfl


theorem embeddingPiTangent_injOn : InjOn f.embeddingPiTangent s := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    ⊢ Set.InjOn (⇑f.embeddingPiTangent) s
  -/
  intro x hx y _ h
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    y : M
    a✝ : Membership.mem s y
    h : Eq (f.embeddingPiTangent x) (f.embeddingPiTangent y)
    ⊢ Eq x y
  -/
  simp only [embeddingPiTangent_coe, funext_iff] at h
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    y : M
    a✝ : Membership.mem s y
    h : ∀ (x_1 : ι), Eq { fst := HSMul.hSMul (↑(f.toFun x_1) x) (↑(extChartAt I (f …
    ⊢ Eq x y
  -/
  obtain ⟨h₁, h₂⟩ := Prod.mk.inj_iff.1 (h (f.ind x hx))
  /-
    case intro
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    y : M
    a✝ : Membership.mem s y
    h : ∀ (x_1 : ι), Eq { fst := HSMul.hSMul (↑(f.toFun x_1) x) (↑(extChartAt I (f …
    h₁ : Eq (HSMul.hSMul (↑(f.toFun (f.ind x hx)) x) (↑(extChartAt I (f.c (f.ind x …
    h₂ : Eq (↑(f.toFun (f.ind x hx)) x) (↑(f.toFun (f.ind x hx)) y)
    ⊢ Eq x y
  -/
  rw [f.apply_ind x hx] at h₂
  /-
    case intro
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    y : M
    a✝ : Membership.mem s y
    h : ∀ (x_1 : ι), Eq { fst := HSMul.hSMul (↑(f.toFun x_1) x) (↑(extChartAt I (f …
    h₁ : Eq (HSMul.hSMul (↑(f.toFun (f.ind x hx)) x) (↑(extChartAt I (f.c (f.ind x …
    h₂ : Eq 1 (↑(f.toFun (f.ind x hx)) y)
    ⊢ Eq x y
  -/
  rw [← h₂, f.apply_ind x hx, one_smul, one_smul] at h₁
  /-
    case intro
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    y : M
    a✝ : Membership.mem s y
    h : ∀ (x_1 : ι), Eq { fst := HSMul.hSMul (↑(f.toFun x_1) x) (↑(extChartAt I (f …
    h₁ : Eq (↑(extChartAt I (f.c (f.ind x hx))) x) (↑(extChartAt I (f.c (f.ind x h …
    h₂ : Eq 1 (↑(f.toFun (f.ind x hx)) y)
    ⊢ Eq x y
  -/
  have := f.mem_extChartAt_source_of_eq_one h₂.symm
  /-
    case intro
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    y : M
    a✝ : Membership.mem s y
    h : ∀ (x_1 : ι), Eq { fst := HSMul.hSMul (↑(f.toFun x_1) x) (↑(extChartAt I (f …
    h₁ : Eq (↑(extChartAt I (f.c (f.ind x hx))) x) (↑(extChartAt I (f.c (f.ind x h …
    h₂ : Eq 1 (↑(f.toFun (f.ind x hx)) y)
    this : Membership.mem (extChartAt I (f.c (f.ind x hx))).source y
    ⊢ Eq x y
  -/
  exact (extChartAt I (f.c _)).injOn (f.mem_extChartAt_ind_source x hx) this h₁
  /-
    🎉 no goals
  -/


theorem embeddingPiTangent_injective (f : SmoothBumpCovering ι I M) :
    Injective f.embeddingPiTangent :=
  injective_iff_injOn_univ.2 f.embeddingPiTangent_injOn


theorem comp_embeddingPiTangent_mfderiv (x : M) (hx : x ∈ s) :
    ((ContinuousLinearMap.fst ℝ E ℝ).comp
            (@ContinuousLinearMap.proj ℝ _ ι (fun _ => E × ℝ) _ _ (fun _ => inferInstance)
              (f.ind x hx))).comp
        (mfderiv I 𝓘(ℝ, ι → E × ℝ) f.embeddingPiTangent x) =
      mfderiv I I (chartAt H (f.c (f.ind x hx))) x := by
  set L :=
    (ContinuousLinearMap.fst ℝ E ℝ).comp
      (@ContinuousLinearMap.proj ℝ _ ι (fun _ => E × ℝ) _ _ (fun _ => inferInstance) (f.ind x hx))
  have := L.hasMFDerivAt.comp x
    (f.embeddingPiTangent.contMDiff.mdifferentiableAt le_top).hasMFDerivAt
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    L : ContinuousLinearMap (RingHom.id Real) (ι → Prod E Real) E := (ContinuousLi …
    this : HasMFDerivAt I (modelWithCornersSelf Real E) (Function.comp ⇑L ⇑f.embed …
    ⊢ Eq (L.comp (mfderiv I (modelWithCornersSelf Real (ι → Prod E Real)) (⇑f.embe …
  -/
  convert hasMFDerivAt_unique this _
  /-
    case convert_2
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    L : ContinuousLinearMap (RingHom.id Real) (ι → Prod E Real) E := (ContinuousLi …
    this : HasMFDerivAt I (modelWithCornersSelf Real E) (Function.comp ⇑L ⇑f.embed …
    ⊢ HasMFDerivAt I (modelWithCornersSelf Real E) (Function.comp ⇑L ⇑f.embeddingP …
  -/
  refine (hasMFDerivAt_extChartAt (f.mem_chartAt_ind_source x hx)).congr_of_eventuallyEq ?_
  /-
    case convert_2
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    L : ContinuousLinearMap (RingHom.id Real) (ι → Prod E Real) E := (ContinuousLi …
    this : HasMFDerivAt I (modelWithCornersSelf Real E) (Function.comp ⇑L ⇑f.embed …
    ⊢ (nhds x).EventuallyEq (Function.comp ⇑L ⇑f.embeddingPiTangent) ↑(extChartAt  …
  -/
  refine (f.eventuallyEq_one x hx).mono fun y hy => ?_
  simp only [L, embeddingPiTangent_coe, ContinuousLinearMap.coe_comp', (· ∘ ·),
    ContinuousLinearMap.coe_fst', ContinuousLinearMap.proj_apply]
  /-
    case convert_2
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    L : ContinuousLinearMap (RingHom.id Real) (ι → Prod E Real) E := (ContinuousLi …
    this : HasMFDerivAt I (modelWithCornersSelf Real E) (Function.comp ⇑L ⇑f.embed …
    y : M
    hy : Eq (↑(f.toFun (f.ind x hx)) y) (1 y)
    ⊢ Eq (HSMul.hSMul (↑(f.toFun (f.ind x hx)) y) (↑(extChartAt I (f.c (f.ind x hx …
  -/
  rw [hy, Pi.one_apply, one_smul]
  /-
    🎉 no goals
  -/


theorem embeddingPiTangent_ker_mfderiv (x : M) (hx : x ∈ s) :
    LinearMap.ker (mfderiv I 𝓘(ℝ, ι → E × ℝ) f.embeddingPiTangent x) = ⊥ := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    ⊢ Eq (LinearMap.ker (mfderiv I (modelWithCornersSelf Real (ι → Prod E Real)) ( …
  -/
  apply bot_unique
  rw [← (mdifferentiable_chart (f.c (f.ind x hx))).ker_mfderiv_eq_bot
      (f.mem_chartAt_ind_source x hx),
    ← comp_embeddingPiTangent_mfderiv]
  /-
    case h
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : Fintype ι
    s : Set M
    f : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    ⊢ LE.le (LinearMap.ker (mfderiv I (modelWithCornersSelf Real (ι → Prod E Real) …
  -/
  exact LinearMap.ker_le_ker_comp _ _
  /-
    🎉 no goals
  -/


theorem embeddingPiTangent_injective_mfderiv (x : M) (hx : x ∈ s) :
    Injective (mfderiv I 𝓘(ℝ, ι → E × ℝ) f.embeddingPiTangent x) :=
  LinearMap.ker_eq_bot.1 (f.embeddingPiTangent_ker_mfderiv x hx)


/-- Baby version of the **Whitney weak embedding theorem**: if `M` admits a finite covering by
supports of bump functions, then for some `n` it can be immersed into the `n`-dimensional
Euclidean space. -/
theorem exists_immersion_euclidean {ι : Type*} [Finite ι] (f : SmoothBumpCovering ι I M) :
    ∃ (n : ℕ) (e : M → EuclideanSpace ℝ (Fin n)),
      ContMDiff I (𝓡 n) ⊤ e ∧ Injective e ∧ ∀ x : M, Injective (mfderiv I (𝓡 n) e x) := by
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    ι : Type u_1
    inst✝ : Finite ι
    f : SmoothBumpCovering ι I M
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    ι : Type u_1
    inst✝ : Finite ι
    f : SmoothBumpCovering ι I M
    val✝ : Fintype ι
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  set F := EuclideanSpace ℝ (Fin <| finrank ℝ (ι → E × ℝ))
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    ι : Type u_1
    inst✝ : Finite ι
    f : SmoothBumpCovering ι I M
    val✝ : Fintype ι
    F : Type := EuclideanSpace Real (Fin (Module.finrank Real (ι → Prod E Real)))
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  letI : IsNoetherian ℝ (E × ℝ) := IsNoetherian.iff_fg.2 inferInstance
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    ι : Type u_1
    inst✝ : Finite ι
    f : SmoothBumpCovering ι I M
    val✝ : Fintype ι
    F : Type := EuclideanSpace Real (Fin (Module.finrank Real (ι → Prod E Real)))
    this : IsNoetherian Real (Prod E Real) := IsNoetherian.iff_fg.mpr inferInstance
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  letI : FiniteDimensional ℝ (ι → E × ℝ) := IsNoetherian.iff_fg.1 inferInstance
  set eEF : (ι → E × ℝ) ≃L[ℝ] F :=
    ContinuousLinearEquiv.ofFinrankEq finrank_euclideanSpace_fin.symm
  refine ⟨_, eEF ∘ f.embeddingPiTangent,
    eEF.toDiffeomorph.contMDiff.comp f.embeddingPiTangent.contMDiff,
    eEF.injective.comp f.embeddingPiTangent_injective, fun x => ?_⟩
  rw [mfderiv_comp _ eEF.differentiableAt.mdifferentiableAt
      (f.embeddingPiTangent.contMDiff.mdifferentiableAt le_top),
    eEF.mfderiv_eq]
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    ι : Type u_1
    inst✝ : Finite ι
    f : SmoothBumpCovering ι I M
    val✝ : Fintype ι
    F : Type := EuclideanSpace Real (Fin (Module.finrank Real (ι → Prod E Real)))
    this✝ : IsNoetherian Real (Prod E Real) := IsNoetherian.iff_fg.mpr inferInstance
    this : FiniteDimensional Real (ι → Prod E Real) := IsNoetherian.iff_fg.mp infe …
    eEF : ContinuousLinearEquiv (RingHom.id Real) (ι → Prod E Real) F := Continuou …
    x : M
    ⊢ Function.Injective ⇑((↑eEF).comp (mfderiv I (modelWithCornersSelf Real (ι →  …
  -/
  exact eEF.injective.comp (f.embeddingPiTangent_injective_mfderiv _ trivial)
  /-
    🎉 no goals
  -/


/-- Baby version of the Whitney weak embedding theorem: if `M` admits a finite covering by
supports of bump functions, then for some `n` it can be embedded into the `n`-dimensional
Euclidean space. -/
theorem exists_embedding_euclidean_of_compact [T2Space M] [CompactSpace M] :
    ∃ (n : ℕ) (e : M → EuclideanSpace ℝ (Fin n)),
      ContMDiff I (𝓡 n) ⊤ e ∧ IsClosedEmbedding e ∧ ∀ x : M, Injective (mfderiv I (𝓡 n) e x) := by
  rcases SmoothBumpCovering.exists_isSubordinate I isClosed_univ fun (x : M) _ => univ_mem with
    ⟨ι, f, -⟩
  /-
    case intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : CompactSpace M
    ι : Type uM
    f : SmoothBumpCovering ι I M
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  haveI := f.fintype
  /-
    case intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : CompactSpace M
    ι : Type uM
    f : SmoothBumpCovering ι I M
    this : Fintype ι
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  rcases f.exists_immersion_euclidean with ⟨n, e, hsmooth, hinj, hinj_mfderiv⟩
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : FiniteDimensional Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : CompactSpace M
    ι : Type uM
    f : SmoothBumpCovering ι I M
    this : Fintype ι
    n : Nat
    e : M → EuclideanSpace Real (Fin n)
    hsmooth : ContMDiff I (modelWithCornersSelf Real (EuclideanSpace Real (Fin n)) …
    hinj : Function.Injective e
    hinj_mfderiv : ∀ (x : M), Function.Injective ⇑(mfderiv I (modelWithCornersSelf …
    ⊢ Exists fun n => Exists fun e => And (ContMDiff I (modelWithCornersSelf Real  …
  -/
  exact ⟨n, e, hsmooth, hsmooth.continuous.isClosedEmbedding hinj, hinj_mfderiv⟩
  /-
    🎉 no goals
  -/

