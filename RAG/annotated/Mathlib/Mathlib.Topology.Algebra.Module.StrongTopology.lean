/-- Given `E` and `F` two topological vector spaces and `𝔖 : Set (Set E)`, then
`UniformConvergenceCLM σ F 𝔖` is a type synonym of `E →SL[σ] F` equipped with the "topology of
uniform convergence on the elements of `𝔖`".

If the continuous linear image of any element of `𝔖` is bounded, this makes `E →SL[σ] F` a
topological vector space. -/
@[nolint unusedArguments]
def UniformConvergenceCLM [TopologicalSpace F] (_ : Set (Set E)) := E →SL[σ] F


instance instFunLike [TopologicalSpace F] (𝔖 : Set (Set E)) :
    FunLike (UniformConvergenceCLM σ F 𝔖) E F :=
  ContinuousLinearMap.funLike


instance instContinuousSemilinearMapClass [TopologicalSpace F] (𝔖 : Set (Set E)) :
    ContinuousSemilinearMapClass (UniformConvergenceCLM σ F 𝔖) σ E F :=
  ContinuousLinearMap.continuousSemilinearMapClass


instance instTopologicalSpace [TopologicalSpace F] [TopologicalAddGroup F] (𝔖 : Set (Set E)) :
    TopologicalSpace (UniformConvergenceCLM σ F 𝔖) :=
  (@UniformOnFun.topologicalSpace E F (TopologicalAddGroup.toUniformSpace F) 𝔖).induced
    (DFunLike.coe : (UniformConvergenceCLM σ F 𝔖) → (E →ᵤ[𝔖] F))


theorem topologicalSpace_eq [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    instTopologicalSpace σ F 𝔖 = TopologicalSpace.induced (UniformOnFun.ofFun 𝔖 ∘ DFunLike.coe)
      (UniformOnFun.topologicalSpace E F 𝔖) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    ⊢ Eq (UniformConvergenceCLM.instTopologicalSpace σ F 𝔖) (TopologicalSpace.indu …
  -/
  rw [instTopologicalSpace]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    ⊢ Eq (TopologicalSpace.induced DFunLike.coe (UniformOnFun.topologicalSpace E F …
  -/
  congr
  /-
    case e_t.h.e_3.h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    ⊢ Eq (TopologicalAddGroup.toUniformSpace F) inst✝¹
  -/
  exact UniformAddGroup.toUniformSpace_eq
  /-
    🎉 no goals
  -/


/-- The uniform structure associated with `ContinuousLinearMap.strongTopology`. We make sure
that this has nice definitional properties. -/
instance instUniformSpace [UniformSpace F] [UniformAddGroup F]
    (𝔖 : Set (Set E)) : UniformSpace (UniformConvergenceCLM σ F 𝔖) :=
  UniformSpace.replaceTopology
    ((UniformOnFun.uniformSpace E F 𝔖).comap (UniformOnFun.ofFun 𝔖 ∘ DFunLike.coe))
        /-
          𝕜₁ : Type u_1
          𝕜₂ : Type u_2
          inst✝⁸ : NormedField 𝕜₁
          inst✝⁷ : NormedField 𝕜₂
          σ : RingHom 𝕜₁ 𝕜₂
          E : Type u_3
          F : Type u_4
          inst✝⁶ : AddCommGroup E
          inst✝⁵ : Module 𝕜₁ E
          inst✝⁴ : TopologicalSpace E
          inst✝³ : AddCommGroup F
          inst✝² : Module 𝕜₂ F
          inst✝¹ : UniformSpace F
          inst✝ : UniformAddGroup F
          𝔖 : Set (Set E)
          ⊢ Eq (UniformConvergenceCLM.instTopologicalSpace σ F 𝔖) UniformSpace.toTopolog …
        -/
    (by rw [UniformConvergenceCLM.instTopologicalSpace, UniformAddGroup.toUniformSpace_eq]; rfl)
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem uniformSpace_eq [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    instUniformSpace σ F 𝔖 =
      UniformSpace.comap (UniformOnFun.ofFun 𝔖 ∘ DFunLike.coe)
        (UniformOnFun.uniformSpace E F 𝔖) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    ⊢ Eq (UniformConvergenceCLM.instUniformSpace σ F 𝔖) (UniformSpace.comap (Funct …
  -/
  rw [instUniformSpace, UniformSpace.replaceTopology_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem uniformity_toTopologicalSpace_eq [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    (UniformConvergenceCLM.instUniformSpace σ F 𝔖).toTopologicalSpace =
      UniformConvergenceCLM.instTopologicalSpace σ F 𝔖 :=
  rfl


theorem isUniformInducing_coeFn [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    IsUniformInducing (α := UniformConvergenceCLM σ F 𝔖) (UniformOnFun.ofFun 𝔖 ∘ DFunLike.coe) :=
  ⟨rfl⟩


theorem isUniformEmbedding_coeFn [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    IsUniformEmbedding (α := UniformConvergenceCLM σ F 𝔖) (UniformOnFun.ofFun 𝔖 ∘ DFunLike.coe) :=
  ⟨isUniformInducing_coeFn .., DFunLike.coe_injective⟩


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_coeFn := isUniformEmbedding_coeFn


theorem isEmbedding_coeFn [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    IsEmbedding (X := UniformConvergenceCLM σ F 𝔖) (Y := E →ᵤ[𝔖] F)
      (UniformOnFun.ofFun 𝔖 ∘ DFunLike.coe) :=
  IsUniformEmbedding.isEmbedding (isUniformEmbedding_coeFn _ _ _)


@[deprecated (since := "2024-10-26")]
alias embedding_coeFn := isEmbedding_coeFn


instance instAddCommGroup [TopologicalSpace F] [TopologicalAddGroup F] (𝔖 : Set (Set E)) :
    AddCommGroup (UniformConvergenceCLM σ F 𝔖) := ContinuousLinearMap.addCommGroup


@[simp]
theorem coe_zero [TopologicalSpace F] [TopologicalAddGroup F] (𝔖 : Set (Set E)) :
    ⇑(0 : UniformConvergenceCLM σ F 𝔖) = 0 :=
  rfl


instance instUniformAddGroup [UniformSpace F] [UniformAddGroup F] (𝔖 : Set (Set E)) :
    UniformAddGroup (UniformConvergenceCLM σ F 𝔖) := by
  let φ : (UniformConvergenceCLM σ F 𝔖) →+ E →ᵤ[𝔖] F :=
    ⟨⟨(DFunLike.coe : (UniformConvergenceCLM σ F 𝔖) → E →ᵤ[𝔖] F), rfl⟩, fun _ _ => rfl⟩
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    φ : AddMonoidHom (UniformConvergenceCLM σ F 𝔖) (UniformOnFun E F 𝔖) := { toFun …
    ⊢ UniformAddGroup (UniformConvergenceCLM σ F 𝔖)
  -/
  exact (isUniformEmbedding_coeFn _ _ _).uniformAddGroup φ
  /-
    🎉 no goals
  -/


instance instTopologicalAddGroup [TopologicalSpace F] [TopologicalAddGroup F]
    (𝔖 : Set (Set E)) : TopologicalAddGroup (UniformConvergenceCLM σ F 𝔖) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    ⊢ TopologicalAddGroup (UniformConvergenceCLM σ F 𝔖)
  -/
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ TopologicalAddGroup (UniformConvergenceCLM σ F 𝔖)
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ TopologicalAddGroup (UniformConvergenceCLM σ F 𝔖)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem continuousEvalConst [TopologicalSpace F] [TopologicalAddGroup F]
    (𝔖 : Set (Set E)) (h𝔖 : ⋃₀ 𝔖 = Set.univ) :
    ContinuousEvalConst (UniformConvergenceCLM σ F 𝔖) E F where
  continuous_eval_const x := by
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      inst✝⁸ : NormedField 𝕜₁
      inst✝⁷ : NormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      E : Type u_3
      F : Type u_4
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜₁ E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜₂ F
      inst✝¹ : TopologicalSpace F
      inst✝ : TopologicalAddGroup F
      𝔖 : Set (Set E)
      h𝔖 : Eq 𝔖.sUnion Set.univ
      x : E
      ⊢ Continuous fun f => f x
    -/
    letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      inst✝⁸ : NormedField 𝕜₁
      inst✝⁷ : NormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      E : Type u_3
      F : Type u_4
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜₁ E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜₂ F
      inst✝¹ : TopologicalSpace F
      inst✝ : TopologicalAddGroup F
      𝔖 : Set (Set E)
      h𝔖 : Eq 𝔖.sUnion Set.univ
      x : E
      this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
      ⊢ Continuous fun f => f x
    -/
    haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
    exact (UniformOnFun.uniformContinuous_eval h𝔖 x).continuous.comp
      (isEmbedding_coeFn σ F 𝔖).continuous


theorem t2Space [TopologicalSpace F] [TopologicalAddGroup F] [T2Space F]
    (𝔖 : Set (Set E)) (h𝔖 : ⋃₀ 𝔖 = univ) : T2Space (UniformConvergenceCLM σ F 𝔖) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁹ : NormedField 𝕜₁
    inst✝⁸ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜₂ F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : T2Space F
    𝔖 : Set (Set E)
    h𝔖 : Eq 𝔖.sUnion Set.univ
    ⊢ T2Space (UniformConvergenceCLM σ F 𝔖)
  -/
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁹ : NormedField 𝕜₁
    inst✝⁸ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜₂ F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : T2Space F
    𝔖 : Set (Set E)
    h𝔖 : Eq 𝔖.sUnion Set.univ
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ T2Space (UniformConvergenceCLM σ F 𝔖)
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁹ : NormedField 𝕜₁
    inst✝⁸ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜₂ F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : T2Space F
    𝔖 : Set (Set E)
    h𝔖 : Eq 𝔖.sUnion Set.univ
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ T2Space (UniformConvergenceCLM σ F 𝔖)
  -/
  haveI : T2Space (E →ᵤ[𝔖] F) := UniformOnFun.t2Space_of_covering h𝔖
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁹ : NormedField 𝕜₁
    inst✝⁸ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜₂ F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : T2Space F
    𝔖 : Set (Set E)
    h𝔖 : Eq 𝔖.sUnion Set.univ
    this✝¹ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this✝ : UniformAddGroup F
    this : T2Space (UniformOnFun E F 𝔖)
    ⊢ T2Space (UniformConvergenceCLM σ F 𝔖)
  -/
  exact (isEmbedding_coeFn σ F 𝔖).t2Space
  /-
    🎉 no goals
  -/


instance instDistribMulAction (M : Type*) [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜₂ M F]
    [TopologicalSpace F] [TopologicalAddGroup F] [ContinuousConstSMul M F] (𝔖 : Set (Set E)) :
    DistribMulAction M (UniformConvergenceCLM σ F 𝔖) := ContinuousLinearMap.distribMulAction


instance instModule (R : Type*) [Semiring R] [Module R F] [SMulCommClass 𝕜₂ R F]
    [TopologicalSpace F] [ContinuousConstSMul R F] [TopologicalAddGroup F] (𝔖 : Set (Set E)) :
    Module R (UniformConvergenceCLM σ F 𝔖) := ContinuousLinearMap.module


theorem continuousSMul [RingHomSurjective σ] [RingHomIsometric σ]
    [TopologicalSpace F] [TopologicalAddGroup F] [ContinuousSMul 𝕜₂ F] (𝔖 : Set (Set E))
    (h𝔖₃ : ∀ S ∈ 𝔖, IsVonNBounded 𝕜₁ S) :
    ContinuousSMul 𝕜₂ (UniformConvergenceCLM σ F 𝔖) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹¹ : NormedField 𝕜₁
    inst✝¹⁰ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜₁ E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    inst✝⁴ : RingHomSurjective σ
    inst✝³ : RingHomIsometric σ
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul 𝕜₂ F
    𝔖 : Set (Set E)
    h𝔖₃ : ∀ (S : Set E), Membership.mem 𝔖 S → Bornology.IsVonNBounded 𝕜₁ S
    ⊢ ContinuousSMul 𝕜₂ (UniformConvergenceCLM σ F 𝔖)
  -/
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹¹ : NormedField 𝕜₁
    inst✝¹⁰ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜₁ E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    inst✝⁴ : RingHomSurjective σ
    inst✝³ : RingHomIsometric σ
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul 𝕜₂ F
    𝔖 : Set (Set E)
    h𝔖₃ : ∀ (S : Set E), Membership.mem 𝔖 S → Bornology.IsVonNBounded 𝕜₁ S
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ ContinuousSMul 𝕜₂ (UniformConvergenceCLM σ F 𝔖)
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  let φ : (UniformConvergenceCLM σ F 𝔖) →ₗ[𝕜₂] E → F :=
    ⟨⟨DFunLike.coe, fun _ _ => rfl⟩, fun _ _ => rfl⟩
  exact UniformOnFun.continuousSMul_induced_of_image_bounded 𝕜₂ E F (UniformConvergenceCLM σ F 𝔖) φ
    ⟨rfl⟩ fun u s hs => (h𝔖₃ s hs).image u


theorem hasBasis_nhds_zero_of_basis [TopologicalSpace F] [TopologicalAddGroup F]
    {ι : Type*} (𝔖 : Set (Set E)) (h𝔖₁ : 𝔖.Nonempty) (h𝔖₂ : DirectedOn (· ⊆ ·) 𝔖) {p : ι → Prop}
    {b : ι → Set F} (h : (𝓝 0 : Filter F).HasBasis p b) :
    (𝓝 (0 : UniformConvergenceCLM σ F 𝔖)).HasBasis
      (fun Si : Set E × ι => Si.1 ∈ 𝔖 ∧ p Si.2)
      fun Si => { f : E →SL[σ] F | ∀ x ∈ Si.1, f x ∈ b Si.2 } := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    𝔖 : Set (Set E)
    h𝔖₁ : 𝔖.Nonempty
    h𝔖₂ : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) 𝔖
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    ⊢ (nhds 0).HasBasis (fun Si => And (Membership.mem 𝔖 Si.1) (p Si.2)) fun Si => …
  -/
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    𝔖 : Set (Set E)
    h𝔖₁ : 𝔖.Nonempty
    h𝔖₂ : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) 𝔖
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ (nhds 0).HasBasis (fun Si => And (Membership.mem 𝔖 Si.1) (p Si.2)) fun Si => …
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    𝔖 : Set (Set E)
    h𝔖₁ : 𝔖.Nonempty
    h𝔖₂ : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) 𝔖
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ (nhds 0).HasBasis (fun Si => And (Membership.mem 𝔖 Si.1) (p Si.2)) fun Si => …
  -/
  rw [(isEmbedding_coeFn σ F 𝔖).isInducing.nhds_eq_comap]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    𝔖 : Set (Set E)
    h𝔖₁ : 𝔖.Nonempty
    h𝔖₂ : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) 𝔖
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ (Filter.comap (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) DFunLike.coe) (nhds ( …
  -/
  exact (UniformOnFun.hasBasis_nhds_zero_of_basis 𝔖 h𝔖₁ h𝔖₂ h).comap DFunLike.coe
  /-
    🎉 no goals
  -/


theorem hasBasis_nhds_zero [TopologicalSpace F] [TopologicalAddGroup F]
    (𝔖 : Set (Set E)) (h𝔖₁ : 𝔖.Nonempty) (h𝔖₂ : DirectedOn (· ⊆ ·) 𝔖) :
    (𝓝 (0 : UniformConvergenceCLM σ F 𝔖)).HasBasis
      (fun SV : Set E × Set F => SV.1 ∈ 𝔖 ∧ SV.2 ∈ (𝓝 0 : Filter F)) fun SV =>
      { f : UniformConvergenceCLM σ F 𝔖 | ∀ x ∈ SV.1, f x ∈ SV.2 } :=
  hasBasis_nhds_zero_of_basis σ F 𝔖 h𝔖₁ h𝔖₂ (𝓝 0).basis_sets


theorem nhds_zero_eq_of_basis [TopologicalSpace F] [TopologicalAddGroup F] (𝔖 : Set (Set E))
    {ι : Type*} {p : ι → Prop} {b : ι → Set F} (h : (𝓝 0 : Filter F).HasBasis p b) :
    𝓝 (0 : UniformConvergenceCLM σ F 𝔖) =
      ⨅ (s : Set E) (_ : s ∈ 𝔖) (i : ι) (_ : p i),
        𝓟 {f : UniformConvergenceCLM σ F 𝔖 | MapsTo f s (b i)} := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    ⊢ Eq (nhds 0) (iInf fun s => iInf fun x => iInf fun i => iInf fun x => Filter. …
  -/
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ Eq (nhds 0) (iInf fun s => iInf fun x => iInf fun i => iInf fun x => Filter. …
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  rw [(isEmbedding_coeFn σ F 𝔖).isInducing.nhds_eq_comap,
    UniformOnFun.nhds_eq_of_basis _ _ h.uniformity_of_nhds_zero]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ Eq (Filter.comap (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) DFunLike.coe) (iIn …
  -/
  simp [MapsTo]
  /-
    🎉 no goals
  -/


theorem nhds_zero_eq [TopologicalSpace F] [TopologicalAddGroup F] (𝔖 : Set (Set E)) :
    𝓝 (0 : UniformConvergenceCLM σ F 𝔖) =
      ⨅ s ∈ 𝔖, ⨅ t ∈ 𝓝 (0 : F),
        𝓟 {f : UniformConvergenceCLM σ F 𝔖 | MapsTo f s t} :=
  nhds_zero_eq_of_basis _ _ _ (𝓝 0).basis_sets


variable {F} in
theorem eventually_nhds_zero_mapsTo [TopologicalSpace F] [TopologicalAddGroup F]
    {𝔖 : Set (Set E)} {s : Set E} (hs : s ∈ 𝔖) {U : Set F} (hu : U ∈ 𝓝 0) :
    ∀ᶠ f : UniformConvergenceCLM σ F 𝔖 in 𝓝 0, MapsTo f s U := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hu : Membership.mem (nhds 0) U
    ⊢ Filter.Eventually (fun f => Set.MapsTo (⇑f) s U) (nhds 0)
  -/
  rw [nhds_zero_eq]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    𝔖 : Set (Set E)
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hu : Membership.mem (nhds 0) U
    ⊢ Filter.Eventually (fun f => Set.MapsTo (⇑f) s U) (iInf fun s => iInf fun h = …
  -/
  apply_rules [mem_iInf_of_mem, mem_principal_self]
  /-
    🎉 no goals
  -/


variable {σ F} in
theorem isVonNBounded_image2_apply {R : Type*} [SeminormedRing R]
    [TopologicalSpace F] [TopologicalAddGroup F]
    [Module R F] [ContinuousConstSMul R F] [SMulCommClass 𝕜₂ R F]
    {𝔖 : Set (Set E)} {S : Set (UniformConvergenceCLM σ F 𝔖)} (hS : IsVonNBounded R S)
    {s : Set E} (hs : s ∈ 𝔖) : IsVonNBounded R (Set.image2 (fun f x ↦ f x) S s) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    hS : Bornology.IsVonNBounded R S
    s : Set E
    hs : Membership.mem 𝔖 s
    ⊢ Bornology.IsVonNBounded R (Set.image2 (fun f x => f x) S s)
  -/
  intro U hU
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    hS : Bornology.IsVonNBounded R S
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    ⊢ Absorbs R U (Set.image2 (fun f x => f x) S s)
  -/
  filter_upwards [hS (eventually_nhds_zero_mapsTo σ hs hU)] with c hc
  /-
    case h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    hS : Bornology.IsVonNBounded R S
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset S (HSMul.hSMul c (setOf fun x => Set.MapsTo (⇑x) s U))
    ⊢ HasSubset.Subset (Set.image2 (fun f x => f x) S s) (HSMul.hSMul c U)
  -/
  rw [image2_subset_iff]
  /-
    case h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    hS : Bornology.IsVonNBounded R S
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset S (HSMul.hSMul c (setOf fun x => Set.MapsTo (⇑x) s U))
    ⊢ ∀ (x : UniformConvergenceCLM σ F 𝔖), Membership.mem S x → ∀ (y : E), Members …
  -/
  intro f hf x hx
  /-
    case h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    hS : Bornology.IsVonNBounded R S
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset S (HSMul.hSMul c (setOf fun x => Set.MapsTo (⇑x) s U))
    f : UniformConvergenceCLM σ F 𝔖
    hf : Membership.mem S f
    x : E
    hx : Membership.mem s x
    ⊢ Membership.mem (HSMul.hSMul c U) (f x)
  -/
  rcases hc hf with ⟨g, hg, rfl⟩
  /-
    case h.intro.intro
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    hS : Bornology.IsVonNBounded R S
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset S (HSMul.hSMul c (setOf fun x => Set.MapsTo (⇑x) s U))
    x : E
    hx : Membership.mem s x
    g : UniformConvergenceCLM σ F 𝔖
    hg : Membership.mem (setOf fun x => Set.MapsTo (⇑x) s U) g
    hf : Membership.mem S ((fun x => HSMul.hSMul c x) g)
    ⊢ Membership.mem (HSMul.hSMul c U) (((fun x => HSMul.hSMul c x) g) x)
  -/
  exact smul_mem_smul_set (hg hx)
  /-
    🎉 no goals
  -/


variable {σ F} in
/-- A set `S` of continuous linear maps with topology of uniform convergence on sets `s ∈ 𝔖`
is von Neumann bounded iff for any `s ∈ 𝔖`,
the set `{f x | (f ∈ S) (x ∈ s)}` is von Neumann bounded. -/
theorem isVonNBounded_iff {R : Type*} [NormedDivisionRing R]
    [TopologicalSpace F] [TopologicalAddGroup F]
    [Module R F] [ContinuousConstSMul R F] [SMulCommClass 𝕜₂ R F]
    {𝔖 : Set (Set E)} {S : Set (UniformConvergenceCLM σ F 𝔖)} :
    IsVonNBounded R S ↔ ∀ s ∈ 𝔖, IsVonNBounded R (Set.image2 (fun f x ↦ f x) S s) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    ⊢ Iff (Bornology.IsVonNBounded R S) (∀ (s : Set E), Membership.mem 𝔖 s → Borno …
  -/
  refine ⟨fun hS s hs ↦ isVonNBounded_image2_apply hS hs, fun h ↦ ?_⟩
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    ⊢ Bornology.IsVonNBounded R S
  -/
  simp_rw [isVonNBounded_iff_absorbing_le, nhds_zero_eq, le_iInf_iff, le_principal_iff]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    ⊢ ∀ (i : Set E), Membership.mem 𝔖 i → ∀ (i_2 : Set F), Membership.mem (nhds 0) …
  -/
  intro s hs U hU
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    ⊢ Membership.mem (Filter.absorbing R S) (setOf fun f => Set.MapsTo (⇑f) s U)
  -/
  rw [Filter.mem_absorbing, Absorbs]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    ⊢ Filter.Eventually (fun a => HasSubset.Subset S (HSMul.hSMul a (setOf fun f = …
  -/
  filter_upwards [h s hs hU, eventually_ne_cobounded 0] with c hc hc₀ f hf
  /-
    case h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset (Set.image2 (fun f x => f x) S s) (HSMul.hSMul c U)
    hc₀ : Ne c 0
    f : UniformConvergenceCLM σ F 𝔖
    hf : Membership.mem S f
    ⊢ Membership.mem (HSMul.hSMul c (setOf fun f => Set.MapsTo (⇑f) s U)) f
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ hc₀]
  /-
    case h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset (Set.image2 (fun f x => f x) S s) (HSMul.hSMul c U)
    hc₀ : Ne c 0
    f : UniformConvergenceCLM σ F 𝔖
    hf : Membership.mem S f
    ⊢ Membership.mem (setOf fun f => Set.MapsTo (⇑f) s U) (HSMul.hSMul (Inv.inv c) …
  -/
  intro x hx
  /-
    case h
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    R : Type u_5
    inst✝⁵ : NormedDivisionRing R
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup F
    inst✝² : Module R F
    inst✝¹ : ContinuousConstSMul R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    S : Set (UniformConvergenceCLM σ F 𝔖)
    h : ∀ (s : Set E), Membership.mem 𝔖 s → Bornology.IsVonNBounded R (Set.image2  …
    s : Set E
    hs : Membership.mem 𝔖 s
    U : Set F
    hU : Membership.mem (nhds 0) U
    c : R
    hc : HasSubset.Subset (Set.image2 (fun f x => f x) S s) (HSMul.hSMul c U)
    hc₀ : Ne c 0
    f : UniformConvergenceCLM σ F 𝔖
    hf : Membership.mem S f
    x : E
    hx : Membership.mem s x
    ⊢ Membership.mem U ((HSMul.hSMul (Inv.inv c) f) x)
  -/
  simpa only [mem_smul_set_iff_inv_smul_mem₀ hc₀] using hc (mem_image2_of_mem hf hx)
  /-
    🎉 no goals
  -/


instance instUniformContinuousConstSMul (M : Type*)
    [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜₂ M F]
    [UniformSpace F] [UniformAddGroup F] [UniformContinuousConstSMul M F] (𝔖 : Set (Set E)) :
    UniformContinuousConstSMul M (UniformConvergenceCLM σ F 𝔖) :=
                                                                          /-
                                                                            𝕜₁ : Type u_1
                                                                            𝕜₂ : Type u_2
                                                                            inst✝¹² : NormedField 𝕜₁
                                                                            inst✝¹¹ : NormedField 𝕜₂
                                                                            σ : RingHom 𝕜₁ 𝕜₂
                                                                            E : Type u_3
                                                                            F : Type u_4
                                                                            inst✝¹⁰ : AddCommGroup E
                                                                            inst✝⁹ : Module 𝕜₁ E
                                                                            inst✝⁸ : TopologicalSpace E
                                                                            inst✝⁷ : AddCommGroup F
                                                                            inst✝⁶ : Module 𝕜₂ F
                                                                            M : Type u_5
                                                                            inst✝⁵ : Monoid M
                                                                            inst✝⁴ : DistribMulAction M F
                                                                            inst✝³ : SMulCommClass 𝕜₂ M F
                                                                            inst✝² : UniformSpace F
                                                                            inst✝¹ : UniformAddGroup F
                                                                            inst✝ : UniformContinuousConstSMul M F
                                                                            𝔖 : Set (Set E)
                                                                            x✝¹ : M
                                                                            x✝ : UniformConvergenceCLM σ F 𝔖
                                                                            ⊢ Eq (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) DFunLike.coe (HSMul.hSMul x✝¹ x✝ …
                                                                          -/
  (isUniformInducing_coeFn σ F 𝔖).uniformContinuousConstSMul fun _ _ ↦ by rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


instance instContinuousConstSMul (M : Type*)
    [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜₂ M F]
    [TopologicalSpace F] [TopologicalAddGroup F] [ContinuousConstSMul M F] (𝔖 : Set (Set E)) :
    ContinuousConstSMul M (UniformConvergenceCLM σ F 𝔖) :=
  let _ := TopologicalAddGroup.toUniformSpace F
  have _ : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  have _ := uniformContinuousConstSMul_of_continuousConstSMul M F
  inferInstance


theorem tendsto_iff_tendstoUniformlyOn {ι : Type*} {p : Filter ι} [UniformSpace F]
    [UniformAddGroup F] (𝔖 : Set (Set E)) {a : ι → UniformConvergenceCLM σ F 𝔖}
    {a₀ : UniformConvergenceCLM σ F 𝔖} :
    Filter.Tendsto a p (𝓝 a₀) ↔ ∀ s ∈ 𝔖, TendstoUniformlyOn (a · ·) a₀ p s := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    ι : Type u_5
    p : Filter ι
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    a : ι → UniformConvergenceCLM σ F 𝔖
    a₀ : UniformConvergenceCLM σ F 𝔖
    ⊢ Iff (Filter.Tendsto a p (nhds a₀)) (∀ (s : Set E), Membership.mem 𝔖 s → Tend …
  -/
  rw [(isEmbedding_coeFn σ F 𝔖).tendsto_nhds_iff, UniformOnFun.tendsto_iff_tendstoUniformlyOn]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    ι : Type u_5
    p : Filter ι
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    𝔖 : Set (Set E)
    a : ι → UniformConvergenceCLM σ F 𝔖
    a₀ : UniformConvergenceCLM σ F 𝔖
    ⊢ Iff (∀ (s : Set E), Membership.mem 𝔖 s → TendstoUniformlyOn (Function.comp ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable {F} in
theorem isUniformInducing_postcomp
    {G : Type*} [AddCommGroup G] [UniformSpace G] [UniformAddGroup G]
    {𝕜₃ : Type*} [NormedField 𝕜₃] [Module 𝕜₃ G]
    {τ : 𝕜₂ →+* 𝕜₃} {ρ : 𝕜₁ →+* 𝕜₃} [RingHomCompTriple σ τ ρ] [UniformSpace F] [UniformAddGroup F]
    (g : F →SL[τ] G) (hg : IsUniformInducing g) (𝔖 : Set (Set E)) :
    IsUniformInducing (α := UniformConvergenceCLM σ F 𝔖) (β := UniformConvergenceCLM ρ G 𝔖)
      g.comp := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁴ : NormedField 𝕜₁
    inst✝¹³ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜₁ E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module 𝕜₂ F
    G : Type u_5
    inst✝⁷ : AddCommGroup G
    inst✝⁶ : UniformSpace G
    inst✝⁵ : UniformAddGroup G
    𝕜₃ : Type u_6
    inst✝⁴ : NormedField 𝕜₃
    inst✝³ : Module 𝕜₃ G
    τ : RingHom 𝕜₂ 𝕜₃
    ρ : RingHom 𝕜₁ 𝕜₃
    inst✝² : RingHomCompTriple σ τ ρ
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    g : ContinuousLinearMap τ F G
    hg : IsUniformInducing ⇑g
    𝔖 : Set (Set E)
    ⊢ IsUniformInducing g.comp
  -/
  rw [← (isUniformInducing_coeFn _ _ _).of_comp_iff]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁴ : NormedField 𝕜₁
    inst✝¹³ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜₁ E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module 𝕜₂ F
    G : Type u_5
    inst✝⁷ : AddCommGroup G
    inst✝⁶ : UniformSpace G
    inst✝⁵ : UniformAddGroup G
    𝕜₃ : Type u_6
    inst✝⁴ : NormedField 𝕜₃
    inst✝³ : Module 𝕜₃ G
    τ : RingHom 𝕜₂ 𝕜₃
    ρ : RingHom 𝕜₁ 𝕜₃
    inst✝² : RingHomCompTriple σ τ ρ
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    g : ContinuousLinearMap τ F G
    hg : IsUniformInducing ⇑g
    𝔖 : Set (Set E)
    ⊢ IsUniformInducing (Function.comp (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) DF …
  -/
  exact (UniformOnFun.postcomp_isUniformInducing hg).comp (isUniformInducing_coeFn _ _ _)
  /-
    🎉 no goals
  -/


theorem completeSpace [UniformSpace F] [UniformAddGroup F] [ContinuousSMul 𝕜₂ F] [CompleteSpace F]
    {𝔖 : Set (Set E)} (h𝔖 : RestrictGenTopology 𝔖) (h𝔖U : ⋃₀ 𝔖 = univ) :
    CompleteSpace (UniformConvergenceCLM σ F 𝔖) := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹⁰ : NormedField 𝕜₁
    inst✝⁹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜₁ E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : UniformSpace F
    inst✝² : UniformAddGroup F
    inst✝¹ : ContinuousSMul 𝕜₂ F
    inst✝ : CompleteSpace F
    𝔖 : Set (Set E)
    h𝔖 : Topology.RestrictGenTopology 𝔖
    h𝔖U : Eq 𝔖.sUnion Set.univ
    ⊢ CompleteSpace (UniformConvergenceCLM σ F 𝔖)
  -/
  wlog hF : T2Space F generalizing F
  · rw [(isUniformInducing_postcomp σ (SeparationQuotient.mkCLM 𝕜₂ F)
      SeparationQuotient.isUniformInducing_mk _).completeSpace_congr]
    /-
      case inr
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      inst✝¹⁰ : NormedField 𝕜₁
      inst✝⁹ : NormedField 𝕜₂
      σ : RingHom 𝕜₁ 𝕜₂
      E : Type u_3
      F : Type u_4
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module 𝕜₁ E
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup F
      inst✝⁴ : Module 𝕜₂ F
      inst✝³ : UniformSpace F
      inst✝² : UniformAddGroup F
      inst✝¹ : ContinuousSMul 𝕜₂ F
      inst✝ : CompleteSpace F
      𝔖 : Set (Set E)
      h𝔖 : Topology.RestrictGenTopology 𝔖
      h𝔖U : Eq 𝔖.sUnion Set.univ
      this : ∀ (F : Type u_4) [inst : AddCommGroup F] [inst_1 : Module 𝕜₂ F] [inst_2 …
      hF : Not (T2Space F)
      ⊢ CompleteSpace (UniformConvergenceCLM σ (SeparationQuotient F) 𝔖)
    -/
    exacts [this _ inferInstance, SeparationQuotient.postcomp_mkCLM_surjective F σ E]
    /-
      🎉 no goals
    -/
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F✝ : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F✝
    inst✝⁶ : Module 𝕜₂ F✝
    𝔖 : Set (Set E)
    h𝔖 : Topology.RestrictGenTopology 𝔖
    h𝔖U : Eq 𝔖.sUnion Set.univ
    F : Type u_4
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : UniformSpace F
    inst✝² : UniformAddGroup F
    inst✝¹ : ContinuousSMul 𝕜₂ F
    inst✝ : CompleteSpace F
    hF : T2Space F
    ⊢ CompleteSpace (UniformConvergenceCLM σ F 𝔖)
  -/
  rw [completeSpace_iff_isComplete_range (isUniformInducing_coeFn _ _ _)]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F✝ : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F✝
    inst✝⁶ : Module 𝕜₂ F✝
    𝔖 : Set (Set E)
    h𝔖 : Topology.RestrictGenTopology 𝔖
    h𝔖U : Eq 𝔖.sUnion Set.univ
    F : Type u_4
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : UniformSpace F
    inst✝² : UniformAddGroup F
    inst✝¹ : ContinuousSMul 𝕜₂ F
    inst✝ : CompleteSpace F
    hF : T2Space F
    ⊢ IsComplete (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) DFunLike.coe))
  -/
  apply IsClosed.isComplete
  have H₁ : IsClosed {f : E →ᵤ[𝔖] F | Continuous ((UniformOnFun.toFun 𝔖) f)} :=
    UniformOnFun.isClosed_setOf_continuous h𝔖
  convert H₁.inter <| (LinearMap.isClosed_range_coe E F σ).preimage
    (UniformOnFun.uniformContinuous_toFun h𝔖U).continuous
  /-
    case h.e'_3
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝¹² : NormedField 𝕜₁
    inst✝¹¹ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F✝ : Type u_4
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜₁ E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F✝
    inst✝⁶ : Module 𝕜₂ F✝
    𝔖 : Set (Set E)
    h𝔖 : Topology.RestrictGenTopology 𝔖
    h𝔖U : Eq 𝔖.sUnion Set.univ
    F : Type u_4
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : UniformSpace F
    inst✝² : UniformAddGroup F
    inst✝¹ : ContinuousSMul 𝕜₂ F
    inst✝ : CompleteSpace F
    hF : T2Space F
    H₁ : IsClosed (setOf fun f => Continuous ((UniformOnFun.toFun 𝔖) f))
    ⊢ Eq (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) DFunLike.coe)) (Inter …
  -/
  exact ContinuousLinearMap.range_coeFn_eq
  /-
    🎉 no goals
  -/


theorem uniformSpace_mono [UniformSpace F] [UniformAddGroup F] (h : 𝔖₂ ⊆ 𝔖₁) :
    instUniformSpace σ F 𝔖₁ ≤ instUniformSpace σ F 𝔖₂ := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    𝔖₁ 𝔖₂ : Set (Set E)
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    h : HasSubset.Subset 𝔖₂ 𝔖₁
    ⊢ LE.le (UniformConvergenceCLM.instUniformSpace σ F 𝔖₁) (UniformConvergenceCLM …
  -/
  simp_rw [uniformSpace_eq]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    𝔖₁ 𝔖₂ : Set (Set E)
    inst✝¹ : UniformSpace F
    inst✝ : UniformAddGroup F
    h : HasSubset.Subset 𝔖₂ 𝔖₁
    ⊢ LE.le (UniformSpace.comap (Function.comp (⇑(UniformOnFun.ofFun 𝔖₁)) DFunLike …
  -/
  exact UniformSpace.comap_mono (UniformOnFun.mono (le_refl _) h)
  /-
    🎉 no goals
  -/


theorem topologicalSpace_mono [TopologicalSpace F] [TopologicalAddGroup F] (h : 𝔖₂ ⊆ 𝔖₁) :
    instTopologicalSpace σ F 𝔖₁ ≤ instTopologicalSpace σ F 𝔖₂ := by
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    𝔖₁ 𝔖₂ : Set (Set E)
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    h : HasSubset.Subset 𝔖₂ 𝔖₁
    ⊢ LE.le (UniformConvergenceCLM.instTopologicalSpace σ F 𝔖₁) (UniformConvergenc …
  -/
  letI := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    𝔖₁ 𝔖₂ : Set (Set E)
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    h : HasSubset.Subset 𝔖₂ 𝔖₁
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ LE.le (UniformConvergenceCLM.instTopologicalSpace σ F 𝔖₁) (UniformConvergenc …
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    𝔖₁ 𝔖₂ : Set (Set E)
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    h : HasSubset.Subset 𝔖₂ 𝔖₁
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ LE.le (UniformConvergenceCLM.instTopologicalSpace σ F 𝔖₁) (UniformConvergenc …
  -/
  simp_rw [← uniformity_toTopologicalSpace_eq]
  /-
    𝕜₁ : Type u_1
    𝕜₂ : Type u_2
    inst✝⁸ : NormedField 𝕜₁
    inst✝⁷ : NormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    E : Type u_3
    F : Type u_4
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    𝔖₁ 𝔖₂ : Set (Set E)
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    h : HasSubset.Subset 𝔖₂ 𝔖₁
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ LE.le UniformSpace.toTopologicalSpace UniformSpace.toTopologicalSpace
  -/
  exact UniformSpace.toTopologicalSpace_mono (uniformSpace_mono σ F h)
  /-
    🎉 no goals
  -/


/-- The topology of bounded convergence on `E →L[𝕜] F`. This coincides with the topology induced by
the operator norm when `E` and `F` are normed spaces. -/
instance topologicalSpace [TopologicalSpace F] [TopologicalAddGroup F] :
    TopologicalSpace (E →SL[σ] F) :=
  UniformConvergenceCLM.instTopologicalSpace σ F { S | IsVonNBounded 𝕜₁ S }


instance topologicalAddGroup [TopologicalSpace F] [TopologicalAddGroup F] :
    TopologicalAddGroup (E →SL[σ] F) :=
  UniformConvergenceCLM.instTopologicalAddGroup σ F _


instance continuousSMul [RingHomSurjective σ] [RingHomIsometric σ] [TopologicalSpace F]
    [TopologicalAddGroup F] [ContinuousSMul 𝕜₂ F] : ContinuousSMul 𝕜₂ (E →SL[σ] F) :=
  UniformConvergenceCLM.continuousSMul σ F { S | IsVonNBounded 𝕜₁ S } fun _ hs => hs


instance uniformSpace [UniformSpace F] [UniformAddGroup F] : UniformSpace (E →SL[σ] F) :=
  UniformConvergenceCLM.instUniformSpace σ F { S | IsVonNBounded 𝕜₁ S }


instance uniformAddGroup [UniformSpace F] [UniformAddGroup F] : UniformAddGroup (E →SL[σ] F) :=
  UniformConvergenceCLM.instUniformAddGroup σ F _


instance instContinuousEvalConst [TopologicalSpace F] [TopologicalAddGroup F]
    [ContinuousSMul 𝕜₁ E] : ContinuousEvalConst (E →SL[σ] F) E F :=
  UniformConvergenceCLM.continuousEvalConst σ F _ Bornology.isVonNBounded_covers


instance instT2Space [TopologicalSpace F] [TopologicalAddGroup F] [ContinuousSMul 𝕜₁ E]
    [T2Space F] : T2Space (E →SL[σ] F) :=
  UniformConvergenceCLM.t2Space σ F _ Bornology.isVonNBounded_covers


protected theorem hasBasis_nhds_zero_of_basis [TopologicalSpace F] [TopologicalAddGroup F]
    {ι : Type*} {p : ι → Prop} {b : ι → Set F} (h : (𝓝 0 : Filter F).HasBasis p b) :
    (𝓝 (0 : E →SL[σ] F)).HasBasis (fun Si : Set E × ι => IsVonNBounded 𝕜₁ Si.1 ∧ p Si.2)
      fun Si => { f : E →SL[σ] F | ∀ x ∈ Si.1, f x ∈ b Si.2 } :=
  UniformConvergenceCLM.hasBasis_nhds_zero_of_basis σ F { S | IsVonNBounded 𝕜₁ S }
    ⟨∅, isVonNBounded_empty 𝕜₁ E⟩
    (directedOn_of_sup_mem fun _ _ => IsVonNBounded.union) h


protected theorem hasBasis_nhds_zero [TopologicalSpace F] [TopologicalAddGroup F] :
    (𝓝 (0 : E →SL[σ] F)).HasBasis
      (fun SV : Set E × Set F => IsVonNBounded 𝕜₁ SV.1 ∧ SV.2 ∈ (𝓝 0 : Filter F))
      fun SV => { f : E →SL[σ] F | ∀ x ∈ SV.1, f x ∈ SV.2 } :=
  ContinuousLinearMap.hasBasis_nhds_zero_of_basis (𝓝 0).basis_sets


theorem isUniformEmbedding_toUniformOnFun [UniformSpace F] [UniformAddGroup F] :
    IsUniformEmbedding
      fun f : E →SL[σ] F ↦ UniformOnFun.ofFun {s | Bornology.IsVonNBounded 𝕜₁ s} f :=
  UniformConvergenceCLM.isUniformEmbedding_coeFn ..


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_toUniformOnFun := isUniformEmbedding_toUniformOnFun


instance uniformContinuousConstSMul
    {M : Type*} [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜₂ M F]
    [UniformSpace F] [UniformAddGroup F] [UniformContinuousConstSMul M F] :
    UniformContinuousConstSMul M (E →SL[σ] F) :=
  UniformConvergenceCLM.instUniformContinuousConstSMul σ F _ _


instance continuousConstSMul {M : Type*} [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜₂ M F]
    [TopologicalSpace F] [TopologicalAddGroup F] [ContinuousConstSMul M F] :
    ContinuousConstSMul M (E →SL[σ] F) :=
  UniformConvergenceCLM.instContinuousConstSMul σ F _ _


protected theorem nhds_zero_eq_of_basis [TopologicalSpace F] [TopologicalAddGroup F]
    {ι : Type*} {p : ι → Prop} {b : ι → Set F} (h : (𝓝 0 : Filter F).HasBasis p b) :
    𝓝 (0 : E →SL[σ] F) =
      ⨅ (s : Set E) (_ : IsVonNBounded 𝕜₁ s) (i : ι) (_ : p i),
        𝓟 {f : E →SL[σ] F | MapsTo f s (b i)} :=
  UniformConvergenceCLM.nhds_zero_eq_of_basis _ _ _ h


protected theorem nhds_zero_eq [TopologicalSpace F] [TopologicalAddGroup F] :
    𝓝 (0 : E →SL[σ] F) =
      ⨅ (s : Set E) (_ : IsVonNBounded 𝕜₁ s) (U : Set F) (_ : U ∈ 𝓝 0),
        𝓟 {f : E →SL[σ] F | MapsTo f s U} :=
  UniformConvergenceCLM.nhds_zero_eq ..


/-- If `s` is a von Neumann bounded set and `U` is a neighbourhood of zero,
then sufficiently small continuous linear maps map `s` to `U`. -/
theorem eventually_nhds_zero_mapsTo [TopologicalSpace F] [TopologicalAddGroup F]
    {s : Set E} (hs : IsVonNBounded 𝕜₁ s) {U : Set F} (hu : U ∈ 𝓝 0) :
    ∀ᶠ f : E →SL[σ] F in 𝓝 0, MapsTo f s U :=
  UniformConvergenceCLM.eventually_nhds_zero_mapsTo _ hs hu


/-- If `S` is a von Neumann bounded set of continuous linear maps `f : E →SL[σ] F`
and `s` is a von Neumann bounded set in the domain,
then the set `{f x | (f ∈ S) (x ∈ s)}` is von Neumann bounded.

See also `isVonNBounded_iff` for an `Iff` version with stronger typeclass assumptions. -/
theorem isVonNBounded_image2_apply {R : Type*} [SeminormedRing R]
    [TopologicalSpace F] [TopologicalAddGroup F]
    [Module R F] [ContinuousConstSMul R F] [SMulCommClass 𝕜₂ R F]
    {S : Set (E →SL[σ] F)} (hS : IsVonNBounded R S) {s : Set E} (hs : IsVonNBounded 𝕜₁ s) :
    IsVonNBounded R (Set.image2 (fun f x ↦ f x) S s) :=
  UniformConvergenceCLM.isVonNBounded_image2_apply hS hs


/-- A set `S` of continuous linear maps is von Neumann bounded
iff for any von Neumann bounded set `s`,
the set `{f x | (f ∈ S) (x ∈ s)}` is von Neumann bounded.

For the forward implication with weaker typeclass assumptions, see `isVonNBounded_image2_apply`. -/
theorem isVonNBounded_iff {R : Type*} [NormedDivisionRing R]
    [TopologicalSpace F] [TopologicalAddGroup F]
    [Module R F] [ContinuousConstSMul R F] [SMulCommClass 𝕜₂ R F]
    {S : Set (E →SL[σ] F)} :
    IsVonNBounded R S ↔
      ∀ s, IsVonNBounded 𝕜₁ s → IsVonNBounded R (Set.image2 (fun f x ↦ f x) S s) :=
  UniformConvergenceCLM.isVonNBounded_iff


theorem completeSpace [UniformSpace F] [UniformAddGroup F] [ContinuousSMul 𝕜₂ F] [CompleteSpace F]
    [ContinuousSMul 𝕜₁ E] (h : RestrictGenTopology {s : Set E | IsVonNBounded 𝕜₁ s}) :
    CompleteSpace (E →SL[σ] F) :=
  UniformConvergenceCLM.completeSpace _ _ h isVonNBounded_covers


instance instCompleteSpace [TopologicalAddGroup E] [ContinuousSMul 𝕜₁ E] [SequentialSpace E]
    [UniformSpace F] [UniformAddGroup F] [ContinuousSMul 𝕜₂ F] [CompleteSpace F] :
    CompleteSpace (E →SL[σ] F) :=
  completeSpace <| .of_seq fun _ _ h ↦ (h.isVonNBounded_range 𝕜₁).insert _


/-- Pre-composition by a *fixed* continuous linear map as a continuous linear map.
Note that in non-normed space it is not always true that composition is continuous
in both variables, so we have to fix one of them. -/
@[simps]
def precomp [TopologicalAddGroup G] [ContinuousConstSMul 𝕜₃ G] [RingHomSurjective σ]
    [RingHomIsometric σ] (L : E →SL[σ] F) : (F →SL[τ] G) →L[𝕜₃] E →SL[ρ] G where
  toFun f := f.comp L
  map_add' f g := add_comp f g L
  map_smul' a f := smul_comp a f L
  cont := by
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup G
      inst✝² : ContinuousConstSMul 𝕜₃ G
      inst✝¹ : RingHomSurjective σ
      inst✝ : RingHomIsometric σ
      L : ContinuousLinearMap σ E F
      ⊢ Continuous { toFun := fun f => f.comp L, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    letI : UniformSpace G := TopologicalAddGroup.toUniformSpace G
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup G
      inst✝² : ContinuousConstSMul 𝕜₃ G
      inst✝¹ : RingHomSurjective σ
      inst✝ : RingHomIsometric σ
      L : ContinuousLinearMap σ E F
      this : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      ⊢ Continuous { toFun := fun f => f.comp L, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    haveI : UniformAddGroup G := comm_topologicalAddGroup_is_uniform
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup G
      inst✝² : ContinuousConstSMul 𝕜₃ G
      inst✝¹ : RingHomSurjective σ
      inst✝ : RingHomIsometric σ
      L : ContinuousLinearMap σ E F
      this✝ : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      this : UniformAddGroup G
      ⊢ Continuous { toFun := fun f => f.comp L, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    rw [(UniformConvergenceCLM.isEmbedding_coeFn _ _ _).continuous_iff]
    -- Porting note: without this, the following doesn't work
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup G
      inst✝² : ContinuousConstSMul 𝕜₃ G
      inst✝¹ : RingHomSurjective σ
      inst✝ : RingHomIsometric σ
      L : ContinuousLinearMap σ E F
      this✝ : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      this : UniformAddGroup G
      ⊢ Continuous (Function.comp (Function.comp (⇑(UniformOnFun.ofFun (setOf fun S  …
    -/
    change Continuous ((fun f ↦ UniformOnFun.ofFun _ (f ∘ L)) ∘ DFunLike.coe)
    exact (UniformOnFun.precomp_uniformContinuous fun S hS => hS.image L).continuous.comp
        (UniformConvergenceCLM.isEmbedding_coeFn _ _ _).continuous


/-- Post-composition by a *fixed* continuous linear map as a continuous linear map.
Note that in non-normed space it is not always true that composition is continuous
in both variables, so we have to fix one of them. -/
@[simps]
def postcomp [TopologicalAddGroup F] [TopologicalAddGroup G] [ContinuousConstSMul 𝕜₃ G]
    [ContinuousConstSMul 𝕜₂ F] (L : F →SL[τ] G) : (E →SL[σ] F) →SL[τ] E →SL[ρ] G where
  toFun f := L.comp f
  map_add' := comp_add L
  map_smul' := comp_smulₛₗ L
  cont := by
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup F
      inst✝² : TopologicalAddGroup G
      inst✝¹ : ContinuousConstSMul 𝕜₃ G
      inst✝ : ContinuousConstSMul 𝕜₂ F
      L : ContinuousLinearMap τ F G
      ⊢ Continuous { toFun := fun f => L.comp f, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    letI : UniformSpace G := TopologicalAddGroup.toUniformSpace G
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup F
      inst✝² : TopologicalAddGroup G
      inst✝¹ : ContinuousConstSMul 𝕜₃ G
      inst✝ : ContinuousConstSMul 𝕜₂ F
      L : ContinuousLinearMap τ F G
      this : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      ⊢ Continuous { toFun := fun f => L.comp f, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    haveI : UniformAddGroup G := comm_topologicalAddGroup_is_uniform
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup F
      inst✝² : TopologicalAddGroup G
      inst✝¹ : ContinuousConstSMul 𝕜₃ G
      inst✝ : ContinuousConstSMul 𝕜₂ F
      L : ContinuousLinearMap τ F G
      this✝ : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      this : UniformAddGroup G
      ⊢ Continuous { toFun := fun f => L.comp f, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup F
      inst✝² : TopologicalAddGroup G
      inst✝¹ : ContinuousConstSMul 𝕜₃ G
      inst✝ : ContinuousConstSMul 𝕜₂ F
      L : ContinuousLinearMap τ F G
      this✝¹ : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      this✝ : UniformAddGroup G
      this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
      ⊢ Continuous { toFun := fun f => L.comp f, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
    /-
      𝕜₁ : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      inst✝¹⁶ : NormedField 𝕜₁
      inst✝¹⁵ : NormedField 𝕜₂
      inst✝¹⁴ : NormedField 𝕜₃
      σ : RingHom 𝕜₁ 𝕜₂
      τ : RingHom 𝕜₂ 𝕜₃
      ρ : RingHom 𝕜₁ 𝕜₃
      inst✝¹³ : RingHomCompTriple σ τ ρ
      E : Type u_4
      F : Type u_5
      G : Type u_6
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜₁ E
      inst✝¹⁰ : AddCommGroup F
      inst✝⁹ : Module 𝕜₂ F
      inst✝⁸ : AddCommGroup G
      inst✝⁷ : Module 𝕜₃ G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup F
      inst✝² : TopologicalAddGroup G
      inst✝¹ : ContinuousConstSMul 𝕜₃ G
      inst✝ : ContinuousConstSMul 𝕜₂ F
      L : ContinuousLinearMap τ F G
      this✝² : UniformSpace G := TopologicalAddGroup.toUniformSpace G
      this✝¹ : UniformAddGroup G
      this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
      this : UniformAddGroup F
      ⊢ Continuous { toFun := fun f => L.comp f, map_add' := ⋯, map_smul' := ⋯ }.toFun
    -/
    rw [(UniformConvergenceCLM.isEmbedding_coeFn _ _ _).continuous_iff]
    exact
      (UniformOnFun.postcomp_uniformContinuous L.uniformContinuous).continuous.comp
        (UniformConvergenceCLM.isEmbedding_coeFn _ _ _).continuous


/-- Send a continuous bilinear map to an abstract bilinear map (forgetting continuity). -/
def toLinearMap₂ (L : E →L[𝕜] F →L[𝕜] G) : E →ₗ[𝕜] F →ₗ[𝕜] G := (coeLM 𝕜).comp L.toLinearMap


@[simp] lemma toLinearMap₂_apply (L : E →L[𝕜] F →L[𝕜] G) (v : E) (w : F) :
    L.toLinearMap₂ v w = L v w := rfl


theorem isUniformEmbedding_restrictScalars :
    IsUniformEmbedding (restrictScalars 𝕜' : (E →L[𝕜] F) → (E →L[𝕜'] F)) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹³ : AddCommGroup E
    inst✝¹² : TopologicalSpace E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : ContinuousSMul 𝕜 E
    F : Type u_3
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    inst✝⁶ : Module 𝕜 F
    𝕜' : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜' 𝕜
    inst✝³ : Module 𝕜' E
    inst✝² : IsScalarTower 𝕜' 𝕜 E
    inst✝¹ : Module 𝕜' F
    inst✝ : IsScalarTower 𝕜' 𝕜 F
    ⊢ IsUniformEmbedding (ContinuousLinearMap.restrictScalars 𝕜')
  -/
  rw [← isUniformEmbedding_toUniformOnFun.of_comp_iff]
  /-
    𝕜 : Type u_1
    inst✝¹⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹³ : AddCommGroup E
    inst✝¹² : TopologicalSpace E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : ContinuousSMul 𝕜 E
    F : Type u_3
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    inst✝⁶ : Module 𝕜 F
    𝕜' : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜' 𝕜
    inst✝³ : Module 𝕜' E
    inst✝² : IsScalarTower 𝕜' 𝕜 E
    inst✝¹ : Module 𝕜' F
    inst✝ : IsScalarTower 𝕜' 𝕜 F
    ⊢ IsUniformEmbedding (Function.comp (fun f => (UniformOnFun.ofFun (setOf fun s …
  -/
  convert isUniformEmbedding_toUniformOnFun using 4 with s
  /-
    case h.e'_4.h.e'_4.h.e'_2.h.a
    𝕜 : Type u_1
    inst✝¹⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹³ : AddCommGroup E
    inst✝¹² : TopologicalSpace E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : ContinuousSMul 𝕜 E
    F : Type u_3
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    inst✝⁶ : Module 𝕜 F
    𝕜' : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜' 𝕜
    inst✝³ : Module 𝕜' E
    inst✝² : IsScalarTower 𝕜' 𝕜 E
    inst✝¹ : Module 𝕜' F
    inst✝ : IsScalarTower 𝕜' 𝕜 F
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜' s) (Bornology.IsVonNBounded 𝕜 s)
  -/
  exact ⟨fun h ↦ h.extend_scalars _, fun h ↦ h.restrict_scalars _⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_restrictScalars := isUniformEmbedding_restrictScalars


theorem uniformContinuous_restrictScalars :
    UniformContinuous (restrictScalars 𝕜' : (E →L[𝕜] F) → (E →L[𝕜'] F)) :=
  (isUniformEmbedding_restrictScalars 𝕜').uniformContinuous


theorem isEmbedding_restrictScalars :
    IsEmbedding (restrictScalars 𝕜' : (E →L[𝕜] F) → (E →L[𝕜'] F)) :=
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  (isUniformEmbedding_restrictScalars _).isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_restrictScalars := isEmbedding_restrictScalars


@[continuity, fun_prop]
theorem continuous_restrictScalars :
    Continuous (restrictScalars 𝕜' : (E →L[𝕜] F) → (E →L[𝕜'] F)) :=
   (isEmbedding_restrictScalars _).continuous


/-- `ContinuousLinearMap.restrictScalars` as a `ContinuousLinearMap`. -/
def restrictScalarsL : (E →L[𝕜] F) →L[𝕜''] E →L[𝕜'] F :=
  .mk <| restrictScalarsₗ 𝕜 E F 𝕜' 𝕜''


@[simp]
theorem coe_restrictScalarsL : (restrictScalarsL 𝕜 E F 𝕜' 𝕜'' : (E →L[𝕜] F) →ₗ[𝕜''] E →L[𝕜'] F) =
    restrictScalarsₗ 𝕜 E F 𝕜' 𝕜'' :=
  rfl


@[simp]
theorem coe_restrict_scalarsL' : ⇑(restrictScalarsL 𝕜 E F 𝕜' 𝕜'') = restrictScalars 𝕜' :=
  rfl


/-- A pair of continuous (semi)linear equivalences generates a (semi)linear equivalence between the
spaces of continuous (semi)linear maps. -/
@[simps]
def arrowCongrSL (e₁₂ : E ≃SL[σ₁₂] F) (e₄₃ : H ≃SL[σ₄₃] G) :
    (E →SL[σ₁₄] H) ≃SL[σ₄₃] F →SL[σ₂₃] G :=
{ e₁₂.arrowCongrEquiv e₄₃ with
    -- given explicitly to help `simps`
    toFun := fun L => (e₄₃ : H →SL[σ₄₃] G).comp (L.comp (e₁₂.symm : F →SL[σ₂₁] E))
    -- given explicitly to help `simps`
    invFun := fun L => (e₄₃.symm : G →SL[σ₃₄] H).comp (L.comp (e₁₂ : E →SL[σ₁₂] F))
                              /-
                                𝕜 : Type u_1
                                𝕜₂ : Type u_2
                                𝕜₃ : Type u_3
                                𝕜₄ : Type u_4
                                E : Type u_5
                                F : Type u_6
                                G : Type u_7
                                H : Type u_8
                                inst✝³¹ : AddCommGroup E
                                inst✝³⁰ : AddCommGroup F
                                inst✝²⁹ : AddCommGroup G
                                inst✝²⁸ : AddCommGroup H
                                inst✝²⁷ : NormedField 𝕜
                                inst✝²⁶ : NormedField 𝕜₂
                                inst✝²⁵ : NormedField 𝕜₃
                                inst✝²⁴ : NormedField 𝕜₄
                                inst✝²³ : Module 𝕜 E
                                inst✝²² : Module 𝕜₂ F
                                inst✝²¹ : Module 𝕜₃ G
                                inst✝²⁰ : Module 𝕜₄ H
                                inst✝¹⁹ : TopologicalSpace E
                                inst✝¹⁸ : TopologicalSpace F
                                inst✝¹⁷ : TopologicalSpace G
                                inst✝¹⁶ : TopologicalSpace H
                                inst✝¹⁵ : TopologicalAddGroup G
                                inst✝¹⁴ : TopologicalAddGroup H
                                inst✝¹³ : ContinuousConstSMul 𝕜₃ G
                                inst✝¹² : ContinuousConstSMul 𝕜₄ H
                                σ₁₂ : RingHom 𝕜 𝕜₂
                                σ₂₁ : RingHom 𝕜₂ 𝕜
                                σ₂₃ : RingHom 𝕜₂ 𝕜₃
                                σ₁₃ : RingHom 𝕜 𝕜₃
                                σ₃₄ : RingHom 𝕜₃ 𝕜₄
                                σ₄₃ : RingHom 𝕜₄ 𝕜₃
                                σ₂₄ : RingHom 𝕜₂ 𝕜₄
                                σ₁₄ : RingHom 𝕜 𝕜₄
                                inst✝¹¹ : RingHomInvPair σ₁₂ σ₂₁
                                inst✝¹⁰ : RingHomInvPair σ₂₁ σ₁₂
                                inst✝⁹ : RingHomInvPair σ₃₄ σ₄₃
                                inst✝⁸ : RingHomInvPair σ₄₃ σ₃₄
                                inst✝⁷ : RingHomCompTriple σ₂₁ σ₁₄ σ₂₄
                                inst✝⁶ : RingHomCompTriple σ₂₄ σ₄₃ σ₂₃
                                inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                inst✝⁴ : RingHomCompTriple σ₁₃ σ₃₄ σ₁₄
                                inst✝³ : RingHomCompTriple σ₂₃ σ₃₄ σ₂₄
                                inst✝² : RingHomCompTriple σ₁₂ σ₂₄ σ₁₄
                                inst✝¹ : RingHomIsometric σ₁₂
                                inst✝ : RingHomIsometric σ₂₁
                                e₁₂ : ContinuousLinearEquiv σ₁₂ E F
                                e₄₃ : ContinuousLinearEquiv σ₄₃ H G
                                f g : ContinuousLinearMap σ₁₄ E H
                                ⊢ Eq ((fun L => (↑e₄₃).comp (L.comp ↑e₁₂.symm)) (HAdd.hAdd f g)) (HAdd.hAdd (( …
                              -/
    map_add' := fun f g => by simp only [add_comp, comp_add]
                              /-
                                🎉 no goals
                              -/
                               /-
                                 𝕜 : Type u_1
                                 𝕜₂ : Type u_2
                                 𝕜₃ : Type u_3
                                 𝕜₄ : Type u_4
                                 E : Type u_5
                                 F : Type u_6
                                 G : Type u_7
                                 H : Type u_8
                                 inst✝³¹ : AddCommGroup E
                                 inst✝³⁰ : AddCommGroup F
                                 inst✝²⁹ : AddCommGroup G
                                 inst✝²⁸ : AddCommGroup H
                                 inst✝²⁷ : NormedField 𝕜
                                 inst✝²⁶ : NormedField 𝕜₂
                                 inst✝²⁵ : NormedField 𝕜₃
                                 inst✝²⁴ : NormedField 𝕜₄
                                 inst✝²³ : Module 𝕜 E
                                 inst✝²² : Module 𝕜₂ F
                                 inst✝²¹ : Module 𝕜₃ G
                                 inst✝²⁰ : Module 𝕜₄ H
                                 inst✝¹⁹ : TopologicalSpace E
                                 inst✝¹⁸ : TopologicalSpace F
                                 inst✝¹⁷ : TopologicalSpace G
                                 inst✝¹⁶ : TopologicalSpace H
                                 inst✝¹⁵ : TopologicalAddGroup G
                                 inst✝¹⁴ : TopologicalAddGroup H
                                 inst✝¹³ : ContinuousConstSMul 𝕜₃ G
                                 inst✝¹² : ContinuousConstSMul 𝕜₄ H
                                 σ₁₂ : RingHom 𝕜 𝕜₂
                                 σ₂₁ : RingHom 𝕜₂ 𝕜
                                 σ₂₃ : RingHom 𝕜₂ 𝕜₃
                                 σ₁₃ : RingHom 𝕜 𝕜₃
                                 σ₃₄ : RingHom 𝕜₃ 𝕜₄
                                 σ₄₃ : RingHom 𝕜₄ 𝕜₃
                                 σ₂₄ : RingHom 𝕜₂ 𝕜₄
                                 σ₁₄ : RingHom 𝕜 𝕜₄
                                 inst✝¹¹ : RingHomInvPair σ₁₂ σ₂₁
                                 inst✝¹⁰ : RingHomInvPair σ₂₁ σ₁₂
                                 inst✝⁹ : RingHomInvPair σ₃₄ σ₄₃
                                 inst✝⁸ : RingHomInvPair σ₄₃ σ₃₄
                                 inst✝⁷ : RingHomCompTriple σ₂₁ σ₁₄ σ₂₄
                                 inst✝⁶ : RingHomCompTriple σ₂₄ σ₄₃ σ₂₃
                                 inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                 inst✝⁴ : RingHomCompTriple σ₁₃ σ₃₄ σ₁₄
                                 inst✝³ : RingHomCompTriple σ₂₃ σ₃₄ σ₂₄
                                 inst✝² : RingHomCompTriple σ₁₂ σ₂₄ σ₁₄
                                 inst✝¹ : RingHomIsometric σ₁₂
                                 inst✝ : RingHomIsometric σ₂₁
                                 e₁₂ : ContinuousLinearEquiv σ₁₂ E F
                                 e₄₃ : ContinuousLinearEquiv σ₄₃ H G
                                 t : 𝕜₄
                                 f : ContinuousLinearMap σ₁₄ E H
                                 ⊢ Eq ({ toFun := fun L => (↑e₄₃).comp (L.comp ↑e₁₂.symm), map_add' := ⋯ }.toFu …
                               -/
    map_smul' := fun t f => by simp only [smul_comp, comp_smulₛₗ]
                               /-
                                 🎉 no goals
                               -/
    continuous_toFun := ((postcomp F e₄₃.toContinuousLinearMap).comp
      (precomp H e₁₂.symm.toContinuousLinearMap)).continuous
    continuous_invFun := ((precomp H e₁₂.toContinuousLinearMap).comp
      (postcomp F e₄₃.symm.toContinuousLinearMap)).continuous }

-- Porting note: the following two lemmas were autogenerated by `simps` in Lean3, but this is
-- no longer the case. The first one can already be proven by `simp`, but the second can't.


theorem arrowCongrSL_toLinearEquiv_apply (e₁₂ : E ≃SL[σ₁₂] F) (e₄₃ : H ≃SL[σ₄₃] G)
    (L : E →SL[σ₁₄] H) : (e₁₂.arrowCongrSL e₄₃).toLinearEquiv L =
      (e₄₃ : H →SL[σ₄₃] G).comp (L.comp (e₁₂.symm : F →SL[σ₂₁] E)) :=
  rfl


@[simp]
theorem arrowCongrSL_toLinearEquiv_symm_apply (e₁₂ : E ≃SL[σ₁₂] F) (e₄₃ : H ≃SL[σ₄₃] G)
    (L : F →SL[σ₂₃] G) : (e₁₂.arrowCongrSL e₄₃).toLinearEquiv.symm L =
      (e₄₃.symm : G →SL[σ₃₄] H).comp (L.comp (e₁₂ : E →SL[σ₁₂] F)) :=
  rfl


/-- A pair of continuous linear equivalences generates a continuous linear equivalence between
the spaces of continuous linear maps. -/
def arrowCongr (e₁ : E ≃L[𝕜] F) (e₂ : H ≃L[𝕜] G) : (E →L[𝕜] H) ≃L[𝕜] F →L[𝕜] G :=
  e₁.arrowCongrSL e₂


@[simp] lemma arrowCongr_apply (e₁ : E ≃L[𝕜] F) (e₂ : H ≃L[𝕜] G) (f : E →L[𝕜] H) (x : F) :
    e₁.arrowCongr e₂ f x = e₂ (f (e₁.symm x)) := rfl


@[simp] lemma arrowCongr_symm (e₁ : E ≃L[𝕜] F) (e₂ : H ≃L[𝕜] G) :
    (e₁.arrowCongr e₂).symm = e₁.symm.arrowCongr e₂.symm := rfl


