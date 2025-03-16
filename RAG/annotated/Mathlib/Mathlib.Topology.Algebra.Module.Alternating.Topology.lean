instance instTopologicalSpace : TopologicalSpace (E [⋀^ι]→L[𝕜] F) :=
  .induced toContinuousMultilinearMap inferInstance


lemma isClosed_range_toContinuousMultilinearMap [ContinuousSMul 𝕜 E] [T2Space F] :
    IsClosed (Set.range (toContinuousMultilinearMap : (E [⋀^ι]→L[𝕜] F) →
      ContinuousMultilinearMap 𝕜 (fun _ : ι ↦ E) F)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜 F
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : T2Space F
    ⊢ IsClosed (Set.range ContinuousAlternatingMap.toContinuousMultilinearMap)
  -/
  simp only [range_toContinuousMultilinearMap, setOf_forall]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜 F
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : T2Space F
    ⊢ IsClosed (Set.iInter fun i => Set.iInter fun i_1 => Set.iInter fun i_2 => Se …
  -/
  repeat refine isClosed_iInter fun _ ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜 F
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : T2Space F
    x✝⁴ : ι → E
    x✝³ x✝² : ι
    x✝¹ : Eq (x✝⁴ x✝³) (x✝⁴ x✝²)
    x✝ : Ne x✝³ x✝²
    ⊢ IsClosed (setOf fun x => Eq (x x✝⁴) 0)
  -/
  exact isClosed_singleton.preimage (continuous_eval_const _)
  /-
    🎉 no goals
  -/


instance instUniformSpace : UniformSpace (E [⋀^ι]→L[𝕜] F) :=
  .comap toContinuousMultilinearMap inferInstance


lemma isUniformEmbedding_toContinuousMultilinearMap :
    IsUniformEmbedding (toContinuousMultilinearMap : (E [⋀^ι]→L[𝕜] F) → _) where
  injective := toContinuousMultilinearMap_injective
  comap_uniformity := rfl


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_toContinuousMultilinearMap := isUniformEmbedding_toContinuousMultilinearMap


lemma uniformContinuous_toContinuousMultilinearMap :
    UniformContinuous (toContinuousMultilinearMap : (E [⋀^ι]→L[𝕜] F) → _) :=
  isUniformEmbedding_toContinuousMultilinearMap.uniformContinuous


theorem uniformContinuous_coe_fun [ContinuousSMul 𝕜 E] :
    UniformContinuous (DFunLike.coe : (E [⋀^ι]→L[𝕜] F) → (ι → E) → F) :=
  ContinuousMultilinearMap.uniformContinuous_coe_fun.comp
    uniformContinuous_toContinuousMultilinearMap


theorem uniformContinuous_eval_const [ContinuousSMul 𝕜 E] (x : ι → E) :
    UniformContinuous fun f : E [⋀^ι]→L[𝕜] F ↦ f x :=
  uniformContinuous_pi.1 uniformContinuous_coe_fun x


instance instUniformAddGroup : UniformAddGroup (E [⋀^ι]→L[𝕜] F) :=
  isUniformEmbedding_toContinuousMultilinearMap.uniformAddGroup
    (toContinuousMultilinearMapLinear (R := ℕ))


instance instUniformContinuousConstSMul {M : Type*}
    [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜 M F] [ContinuousConstSMul M F] :
    UniformContinuousConstSMul M (E [⋀^ι]→L[𝕜] F) :=
  isUniformEmbedding_toContinuousMultilinearMap.uniformContinuousConstSMul fun _ _ ↦ rfl


theorem isUniformInducing_postcomp {G : Type*} [AddCommGroup G] [UniformSpace G] [UniformAddGroup G]
    [Module 𝕜 G] (g : F →L[𝕜] G) (hg : IsUniformInducing g) :
    IsUniformInducing (g.compContinuousAlternatingMap : (E [⋀^ι]→L[𝕜] F) → (E [⋀^ι]→L[𝕜] G)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜 E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : UniformSpace F
    inst✝⁴ : UniformAddGroup F
    G : Type u_5
    inst✝³ : AddCommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformAddGroup G
    inst✝ : Module 𝕜 G
    g : ContinuousLinearMap (RingHom.id 𝕜) F G
    hg : IsUniformInducing ⇑g
    ⊢ IsUniformInducing g.compContinuousAlternatingMap
  -/
  rw [← isUniformEmbedding_toContinuousMultilinearMap.1.of_comp_iff]
  exact (ContinuousMultilinearMap.isUniformInducing_postcomp g hg).comp
    isUniformEmbedding_toContinuousMultilinearMap.1


open UniformOnFun in
theorem completeSpace (h : RestrictGenTopology {s : Set (ι → E) | IsVonNBounded 𝕜 s}) :
    CompleteSpace (E [⋀^ι]→L[𝕜] F) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝¹⁰ : NormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : UniformSpace F
    inst✝³ : UniformAddGroup F
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : ContinuousConstSMul 𝕜 F
    inst✝ : CompleteSpace F
    h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
    ⊢ CompleteSpace (ContinuousAlternatingMap 𝕜 E F ι)
  -/
  wlog hF : T2Space F generalizing F
  · rw [(isUniformInducing_postcomp (SeparationQuotient.mkCLM _ _)
      SeparationQuotient.isUniformInducing_mk).completeSpace_congr]
      /-
        case inr
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        ι : Type u_4
        inst✝¹⁰ : NormedField 𝕜
        inst✝⁹ : AddCommGroup E
        inst✝⁸ : Module 𝕜 E
        inst✝⁷ : TopologicalSpace E
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module 𝕜 F
        inst✝⁴ : UniformSpace F
        inst✝³ : UniformAddGroup F
        inst✝² : ContinuousSMul 𝕜 E
        inst✝¹ : ContinuousConstSMul 𝕜 F
        inst✝ : CompleteSpace F
        h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
        this : ∀ {F : Type u_3} [inst : AddCommGroup F] [inst_1 : Module 𝕜 F] [inst_2  …
        hF : Not (T2Space F)
        ⊢ CompleteSpace (ContinuousAlternatingMap 𝕜 E (SeparationQuotient F) ι)
      -/
    · exact this inferInstance
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        ι : Type u_4
        inst✝¹⁰ : NormedField 𝕜
        inst✝⁹ : AddCommGroup E
        inst✝⁸ : Module 𝕜 E
        inst✝⁷ : TopologicalSpace E
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module 𝕜 F
        inst✝⁴ : UniformSpace F
        inst✝³ : UniformAddGroup F
        inst✝² : ContinuousSMul 𝕜 E
        inst✝¹ : ContinuousConstSMul 𝕜 F
        inst✝ : CompleteSpace F
        h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
        this : ∀ {F : Type u_3} [inst : AddCommGroup F] [inst_1 : Module 𝕜 F] [inst_2  …
        hF : Not (T2Space F)
        ⊢ Function.Surjective (SeparationQuotient.mkCLM 𝕜 F).compContinuousAlternating …
      -/
    · intro f
      /-
        case inr
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        ι : Type u_4
        inst✝¹⁰ : NormedField 𝕜
        inst✝⁹ : AddCommGroup E
        inst✝⁸ : Module 𝕜 E
        inst✝⁷ : TopologicalSpace E
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module 𝕜 F
        inst✝⁴ : UniformSpace F
        inst✝³ : UniformAddGroup F
        inst✝² : ContinuousSMul 𝕜 E
        inst✝¹ : ContinuousConstSMul 𝕜 F
        inst✝ : CompleteSpace F
        h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
        this : ∀ {F : Type u_3} [inst : AddCommGroup F] [inst_1 : Module 𝕜 F] [inst_2  …
        hF : Not (T2Space F)
        f : ContinuousAlternatingMap 𝕜 E (SeparationQuotient F) ι
        ⊢ Exists fun a => Eq ((SeparationQuotient.mkCLM 𝕜 F).compContinuousAlternating …
      -/
      use (SeparationQuotient.outCLM _ _).compContinuousAlternatingMap f
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        ι : Type u_4
        inst✝¹⁰ : NormedField 𝕜
        inst✝⁹ : AddCommGroup E
        inst✝⁸ : Module 𝕜 E
        inst✝⁷ : TopologicalSpace E
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module 𝕜 F
        inst✝⁴ : UniformSpace F
        inst✝³ : UniformAddGroup F
        inst✝² : ContinuousSMul 𝕜 E
        inst✝¹ : ContinuousConstSMul 𝕜 F
        inst✝ : CompleteSpace F
        h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
        this : ∀ {F : Type u_3} [inst : AddCommGroup F] [inst_1 : Module 𝕜 F] [inst_2  …
        hF : Not (T2Space F)
        f : ContinuousAlternatingMap 𝕜 E (SeparationQuotient F) ι
        ⊢ Eq ((SeparationQuotient.mkCLM 𝕜 F).compContinuousAlternatingMap ((Separation …
      -/
      ext
      /-
        case h.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        ι : Type u_4
        inst✝¹⁰ : NormedField 𝕜
        inst✝⁹ : AddCommGroup E
        inst✝⁸ : Module 𝕜 E
        inst✝⁷ : TopologicalSpace E
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module 𝕜 F
        inst✝⁴ : UniformSpace F
        inst✝³ : UniformAddGroup F
        inst✝² : ContinuousSMul 𝕜 E
        inst✝¹ : ContinuousConstSMul 𝕜 F
        inst✝ : CompleteSpace F
        h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
        this : ∀ {F : Type u_3} [inst : AddCommGroup F] [inst_1 : Module 𝕜 F] [inst_2  …
        hF : Not (T2Space F)
        f : ContinuousAlternatingMap 𝕜 E (SeparationQuotient F) ι
        x✝ : ι → E
        ⊢ Eq (((SeparationQuotient.mkCLM 𝕜 F).compContinuousAlternatingMap ((Separatio …
      -/
      simp
      /-
        🎉 no goals
      -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    F✝ : Type u_3
    ι : Type u_4
    inst✝¹⁴ : NormedField 𝕜
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : AddCommGroup F✝
    inst✝⁹ : Module 𝕜 F✝
    inst✝⁸ : UniformSpace F✝
    inst✝⁷ : UniformAddGroup F✝
    inst✝⁶ : ContinuousSMul 𝕜 E
    h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
    F : Type u_3
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜 F
    inst✝³ : UniformSpace F
    inst✝² : UniformAddGroup F
    inst✝¹ : ContinuousConstSMul 𝕜 F
    inst✝ : CompleteSpace F
    hF : T2Space F
    ⊢ CompleteSpace (ContinuousAlternatingMap 𝕜 E F ι)
  -/
  have := ContinuousMultilinearMap.completeSpace (F := F) h
  rw [completeSpace_iff_isComplete_range
    isUniformEmbedding_toContinuousMultilinearMap.isUniformInducing]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F✝ : Type u_3
    ι : Type u_4
    inst✝¹⁴ : NormedField 𝕜
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : AddCommGroup F✝
    inst✝⁹ : Module 𝕜 F✝
    inst✝⁸ : UniformSpace F✝
    inst✝⁷ : UniformAddGroup F✝
    inst✝⁶ : ContinuousSMul 𝕜 E
    h : Topology.RestrictGenTopology (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
    F : Type u_3
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜 F
    inst✝³ : UniformSpace F
    inst✝² : UniformAddGroup F
    inst✝¹ : ContinuousConstSMul 𝕜 F
    inst✝ : CompleteSpace F
    hF : T2Space F
    this : CompleteSpace (ContinuousMultilinearMap 𝕜 (fun i => E) F)
    ⊢ IsComplete (Set.range ContinuousAlternatingMap.toContinuousMultilinearMap)
  -/
  apply isClosed_range_toContinuousMultilinearMap.isComplete
  /-
    🎉 no goals
  -/


instance instCompleteSpace [TopologicalAddGroup E] [SequentialSpace (ι → E)] :
    CompleteSpace (E [⋀^ι]→L[𝕜] F) :=
  completeSpace <| .of_seq fun _u x hux ↦ (hux.isVonNBounded_range 𝕜).insert x


theorem isUniformEmbedding_restrictScalars :
    IsUniformEmbedding (restrictScalars 𝕜' : E [⋀^ι]→L[𝕜] F → E [⋀^ι]→L[𝕜'] F) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝¹⁴ : NormedField 𝕜
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : AddCommGroup F
    inst✝⁹ : Module 𝕜 F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    𝕜' : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜'
    inst✝⁵ : NormedAlgebra 𝕜' 𝕜
    inst✝⁴ : Module 𝕜' E
    inst✝³ : IsScalarTower 𝕜' 𝕜 E
    inst✝² : Module 𝕜' F
    inst✝¹ : IsScalarTower 𝕜' 𝕜 F
    inst✝ : ContinuousSMul 𝕜 E
    ⊢ IsUniformEmbedding (ContinuousAlternatingMap.restrictScalars 𝕜')
  -/
  rw [← isUniformEmbedding_toContinuousMultilinearMap.of_comp_iff]
  exact (ContinuousMultilinearMap.isUniformEmbedding_restrictScalars 𝕜').comp
    isUniformEmbedding_toContinuousMultilinearMap


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_restrictScalars := isUniformEmbedding_restrictScalars


theorem uniformContinuous_restrictScalars :
    UniformContinuous (restrictScalars 𝕜' : E [⋀^ι]→L[𝕜] F → E [⋀^ι]→L[𝕜'] F) :=
  (isUniformEmbedding_restrictScalars 𝕜').uniformContinuous


lemma isEmbedding_toContinuousMultilinearMap :
    IsEmbedding (toContinuousMultilinearMap : (E [⋀^ι]→L[𝕜] F → _)) :=
  letI := TopologicalAddGroup.toUniformSpace F
  haveI := comm_topologicalAddGroup_is_uniform (G := F)
  isUniformEmbedding_toContinuousMultilinearMap.isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_toContinuousMultilinearMap := isEmbedding_toContinuousMultilinearMap


instance instTopologicalAddGroup : TopologicalAddGroup (E [⋀^ι]→L[𝕜] F) :=
  isEmbedding_toContinuousMultilinearMap.topologicalAddGroup
    (toContinuousMultilinearMapLinear (R := ℕ))


@[continuity, fun_prop]
lemma continuous_toContinuousMultilinearMap :
    Continuous (toContinuousMultilinearMap : (E [⋀^ι]→L[𝕜] F → _)) :=
  isEmbedding_toContinuousMultilinearMap.continuous


instance instContinuousConstSMul
    {M : Type*} [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜 M F] [ContinuousConstSMul M F] :
    ContinuousConstSMul M (E [⋀^ι]→L[𝕜] F) :=
  isEmbedding_toContinuousMultilinearMap.continuousConstSMul id rfl


instance instContinuousSMul [ContinuousSMul 𝕜 F] : ContinuousSMul 𝕜 (E [⋀^ι]→L[𝕜] F) :=
  isEmbedding_toContinuousMultilinearMap.continuousSMul continuous_id rfl


theorem hasBasis_nhds_zero_of_basis {ι' : Type*} {p : ι' → Prop} {b : ι' → Set F}
    (h : (𝓝 (0 : F)).HasBasis p b) :
    (𝓝 (0 : E [⋀^ι]→L[𝕜] F)).HasBasis
      (fun Si : Set (ι → E) × ι' => IsVonNBounded 𝕜 Si.1 ∧ p Si.2)
      fun Si => { f | MapsTo f Si.1 (b Si.2) } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι' : Type u_5
    p : ι' → Prop
    b : ι' → Set F
    h : (nhds 0).HasBasis p b
    ⊢ (nhds 0).HasBasis (fun Si => And (Bornology.IsVonNBounded 𝕜 Si.1) (p Si.2))  …
  -/
  rw [nhds_induced]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι' : Type u_5
    p : ι' → Prop
    b : ι' → Set F
    h : (nhds 0).HasBasis p b
    ⊢ (Filter.comap ContinuousAlternatingMap.toContinuousMultilinearMap (nhds (Con …
  -/
  exact (ContinuousMultilinearMap.hasBasis_nhds_zero_of_basis h).comap _
  /-
    🎉 no goals
  -/


theorem hasBasis_nhds_zero :
    (𝓝 (0 : E [⋀^ι]→L[𝕜] F)).HasBasis
      (fun SV : Set (ι → E) × Set F => IsVonNBounded 𝕜 SV.1 ∧ SV.2 ∈ 𝓝 0)
      fun SV => { f | MapsTo f SV.1 SV.2 } :=
  hasBasis_nhds_zero_of_basis (Filter.basis_sets _)


lemma isClosedEmbedding_toContinuousMultilinearMap [T2Space F] :
    IsClosedEmbedding (toContinuousMultilinearMap :
      (E [⋀^ι]→L[𝕜] F) → ContinuousMultilinearMap 𝕜 (fun _ : ι ↦ E) F) :=
  ⟨isEmbedding_toContinuousMultilinearMap, isClosed_range_toContinuousMultilinearMap⟩


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_toContinuousMultilinearMap := isClosedEmbedding_toContinuousMultilinearMap


instance instContinuousEvalConst : ContinuousEvalConst (E [⋀^ι]→L[𝕜] F) (ι → E) F :=
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    ι : Type u_4
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul 𝕜 E
    ⊢ ∀ (g : ContinuousAlternatingMap 𝕜 E F ι), Eq ⇑g.toContinuousMultilinearMap ⇑g
  -/
  .of_continuous_forget continuous_toContinuousMultilinearMap
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
protected alias continuous_eval_const := continuous_eval_const


@[deprecated (since := "2024-10-05")]
protected alias continuous_coe_fun := continuous_coeFun


instance instT2Space [T2Space F] : T2Space (E [⋀^ι]→L[𝕜] F) :=
  .of_injective_continuous DFunLike.coe_injective continuous_coeFun


instance instT3Space [T2Space F] : T3Space (E [⋀^ι]→L[𝕜] F) :=
  inferInstance


theorem isEmbedding_restrictScalars :
    IsEmbedding (restrictScalars 𝕜' : E [⋀^ι]→L[𝕜] F → E [⋀^ι]→L[𝕜'] F) :=
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  (isUniformEmbedding_restrictScalars _).isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_restrictScalars := isEmbedding_restrictScalars


@[continuity, fun_prop]
theorem continuous_restrictScalars :
    Continuous (restrictScalars 𝕜' : E [⋀^ι]→L[𝕜] F → E [⋀^ι]→L[𝕜'] F) :=
  isEmbedding_restrictScalars.continuous


variable (𝕜') in
/-- `ContinuousMultilinearMap.restrictScalars` as a `ContinuousLinearMap`. -/
@[simps (config := .asFn) apply]
def restrictScalarsCLM [ContinuousConstSMul 𝕜' F] :
    E [⋀^ι]→L[𝕜] F →L[𝕜'] E [⋀^ι]→L[𝕜'] F where
  toFun := restrictScalars 𝕜'
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- The application of a multilinear map as a `ContinuousLinearMap`. -/
def apply [ContinuousConstSMul 𝕜 F] (m : ι → E) : E [⋀^ι]→L[𝕜] F →L[𝕜] F where
  toFun c := c m
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  cont := continuous_eval_const m


@[simp]
lemma apply_apply [ContinuousConstSMul 𝕜 F] {m : ι → E} {c : E [⋀^ι]→L[𝕜] F} :
    apply 𝕜 E F m c = c m := rfl


theorem hasSum_eval {α : Type*} {p : α → E [⋀^ι]→L[𝕜] F}
    {q : E [⋀^ι]→L[𝕜] F} (h : HasSum p q) (m : ι → E) :
    HasSum (fun a => p a m) (q m) :=
  h.map (applyAddHom m) (continuous_eval_const m)


theorem tsum_eval [T2Space F] {α : Type*} {p : α → E [⋀^ι]→L[𝕜] F} (hp : Summable p)
    (m : ι → E) : (∑' a, p a) m = ∑' a, p a m :=
  (hasSum_eval hp.hasSum m).tsum_eq.symm


