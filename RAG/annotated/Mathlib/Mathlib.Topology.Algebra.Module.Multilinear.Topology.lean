/-- An auxiliary definition used to define topology on `ContinuousMultilinearMap 𝕜 E F`. -/
def toUniformOnFun [TopologicalSpace F] (f : ContinuousMultilinearMap 𝕜 E F) :
    (Π i, E i) →ᵤ[{s | IsVonNBounded 𝕜 s}] F :=
  UniformOnFun.ofFun _ f


open UniformOnFun in
lemma range_toUniformOnFun [DecidableEq ι] [TopologicalSpace F] :
    range toUniformOnFun =
      {f : (Π i, E i) →ᵤ[{s | IsVonNBounded 𝕜 s}] F |
        Continuous (toFun _ f) ∧
        (∀ (m : Π i, E i) i x y,
          toFun _ f (update m i (x + y)) = toFun _ f (update m i x) + toFun _ f (update m i y)) ∧
        (∀ (m : Π i, E i) i (c : 𝕜) x,
          toFun _ f (update m i (c • x)) = c • toFun _ f (update m i x))} := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : (i : ι) → TopologicalSpace (E i)
    inst✝⁵ : (i : ι) → AddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module 𝕜 (E i)
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : DecidableEq ι
    inst✝ : TopologicalSpace F
    ⊢ Eq (Set.range ContinuousMultilinearMap.toUniformOnFun) (setOf fun f => And ( …
  -/
  ext f
  /-
    case h
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : (i : ι) → TopologicalSpace (E i)
    inst✝⁵ : (i : ι) → AddCommGroup (E i)
    inst✝⁴ : (i : ι) → Module 𝕜 (E i)
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : DecidableEq ι
    inst✝ : TopologicalSpace F
    f : UniformOnFun ((i : ι) → E i) F (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
    ⊢ Iff (Membership.mem (Set.range ContinuousMultilinearMap.toUniformOnFun) f) ( …
  -/
  constructor
    /-
      case h.mp
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι) → AddCommGroup (E i)
      inst✝⁴ : (i : ι) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : DecidableEq ι
      inst✝ : TopologicalSpace F
      f : UniformOnFun ((i : ι) → E i) F (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
      ⊢ Membership.mem (Set.range ContinuousMultilinearMap.toUniformOnFun) f → Membe …
    -/
  · rintro ⟨f, rfl⟩
    /-
      case h.mp.intro
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι) → AddCommGroup (E i)
      inst✝⁴ : (i : ι) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : DecidableEq ι
      inst✝ : TopologicalSpace F
      f : ContinuousMultilinearMap 𝕜 E F
      ⊢ Membership.mem (setOf fun f => And (Continuous ((UniformOnFun.toFun (setOf f …
    -/
    exact ⟨f.cont, f.map_update_add, f.map_update_smul⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι) → AddCommGroup (E i)
      inst✝⁴ : (i : ι) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : DecidableEq ι
      inst✝ : TopologicalSpace F
      f : UniformOnFun ((i : ι) → E i) F (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
      ⊢ Membership.mem (setOf fun f => And (Continuous ((UniformOnFun.toFun (setOf f …
    -/
  · rintro ⟨hcont, hadd, hsmul⟩
    /-
      case h.mpr.intro.intro
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι) → AddCommGroup (E i)
      inst✝⁴ : (i : ι) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : DecidableEq ι
      inst✝ : TopologicalSpace F
      f : UniformOnFun ((i : ι) → E i) F (setOf fun s => Bornology.IsVonNBounded 𝕜 s)
      hcont : Continuous ((UniformOnFun.toFun (setOf fun s => Bornology.IsVonNBounde …
      hadd : ∀ (m : (i : ι) → E i) (i : ι) (x y : E i), Eq ((UniformOnFun.toFun (set …
      hsmul : ∀ (m : (i : ι) → E i) (i : ι) (c : 𝕜) (x : E i), Eq ((UniformOnFun.toF …
      ⊢ Membership.mem (Set.range ContinuousMultilinearMap.toUniformOnFun) f
    -/
    exact ⟨⟨⟨f, by intro; convert hadd, by intro; convert hsmul⟩, hcont⟩, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma toUniformOnFun_toFun [TopologicalSpace F] (f : ContinuousMultilinearMap 𝕜 E F) :
    UniformOnFun.toFun _ f.toUniformOnFun = f :=
  rfl


instance instTopologicalSpace [TopologicalSpace F] [TopologicalAddGroup F] :
    TopologicalSpace (ContinuousMultilinearMap 𝕜 E F) :=
  .induced toUniformOnFun <|
    @UniformOnFun.topologicalSpace _ _ (TopologicalAddGroup.toUniformSpace F) _


instance instUniformSpace [UniformSpace F] [UniformAddGroup F] :
    UniformSpace (ContinuousMultilinearMap 𝕜 E F) :=
  .replaceTopology (.comap toUniformOnFun <| UniformOnFun.uniformSpace _ _ _) <| by
    /-
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι) → AddCommGroup (E i)
      inst✝⁴ : (i : ι) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : UniformSpace F
      inst✝ : UniformAddGroup F
      ⊢ Eq ContinuousMultilinearMap.instTopologicalSpace UniformSpace.toTopologicalS …
    -/
    rw [instTopologicalSpace, UniformAddGroup.toUniformSpace_eq]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma isUniformInducing_toUniformOnFun :
    IsUniformInducing (toUniformOnFun :
      ContinuousMultilinearMap 𝕜 E F → ((Π i, E i) →ᵤ[{s | IsVonNBounded 𝕜 s}] F)) := ⟨rfl⟩


lemma isUniformEmbedding_toUniformOnFun :
    IsUniformEmbedding (toUniformOnFun : ContinuousMultilinearMap 𝕜 E F → _) :=
  ⟨isUniformInducing_toUniformOnFun, DFunLike.coe_injective⟩


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_toUniformOnFun := isUniformEmbedding_toUniformOnFun


lemma isEmbedding_toUniformOnFun :
    IsEmbedding (toUniformOnFun : ContinuousMultilinearMap 𝕜 E F →
      ((Π i, E i) →ᵤ[{s | IsVonNBounded 𝕜 s}] F)) :=
  isUniformEmbedding_toUniformOnFun.isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_toUniformOnFun := isEmbedding_toUniformOnFun


theorem uniformContinuous_coe_fun [∀ i, ContinuousSMul 𝕜 (E i)] :
    UniformContinuous (DFunLike.coe : ContinuousMultilinearMap 𝕜 E F → (Π i, E i) → F) :=
  (UniformOnFun.uniformContinuous_toFun isVonNBounded_covers).comp
    isUniformEmbedding_toUniformOnFun.uniformContinuous


theorem uniformContinuous_eval_const [∀ i, ContinuousSMul 𝕜 (E i)] (x : Π i, E i) :
    UniformContinuous fun f : ContinuousMultilinearMap 𝕜 E F ↦ f x :=
  uniformContinuous_pi.1 uniformContinuous_coe_fun x


instance instUniformAddGroup : UniformAddGroup (ContinuousMultilinearMap 𝕜 E F) :=
  let φ : ContinuousMultilinearMap 𝕜 E F →+ (Π i, E i) →ᵤ[{s | IsVonNBounded 𝕜 s}] F :=
    { toFun := toUniformOnFun, map_add' := fun _ _ ↦ rfl, map_zero' := rfl }
  isUniformEmbedding_toUniformOnFun.uniformAddGroup φ


instance instUniformContinuousConstSMul {M : Type*}
    [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜 M F] [ContinuousConstSMul M F] :
    UniformContinuousConstSMul M (ContinuousMultilinearMap 𝕜 E F) :=
  haveI := uniformContinuousConstSMul_of_continuousConstSMul M F
  isUniformEmbedding_toUniformOnFun.uniformContinuousConstSMul fun _ _ ↦ rfl


theorem isUniformInducing_postcomp
    {G : Type*} [AddCommGroup G] [UniformSpace G] [UniformAddGroup G] [Module 𝕜 G]
    (g : F →L[𝕜] G) (hg : IsUniformInducing g) :
    IsUniformInducing (g.compContinuousMultilinearMap :
      ContinuousMultilinearMap 𝕜 E F → ContinuousMultilinearMap 𝕜 E G) := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : (i : ι) → TopologicalSpace (E i)
    inst✝⁹ : (i : ι) → AddCommGroup (E i)
    inst✝⁸ : (i : ι) → Module 𝕜 (E i)
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
    ⊢ IsUniformInducing g.compContinuousMultilinearMap
  -/
  rw [← isUniformInducing_toUniformOnFun.of_comp_iff]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : (i : ι) → TopologicalSpace (E i)
    inst✝⁹ : (i : ι) → AddCommGroup (E i)
    inst✝⁸ : (i : ι) → Module 𝕜 (E i)
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
    ⊢ IsUniformInducing (Function.comp ContinuousMultilinearMap.toUniformOnFun g.c …
  -/
  exact (UniformOnFun.postcomp_isUniformInducing hg).comp isUniformInducing_toUniformOnFun
  /-
    🎉 no goals
  -/


open UniformOnFun in
theorem completeSpace (h : RestrictGenTopology {s : Set (Π i, E i) | IsVonNBounded 𝕜 s}) :
    CompleteSpace (ContinuousMultilinearMap 𝕜 E F) := by
  classical
  wlog hF : T2Space F generalizing F
  · rw [(isUniformInducing_postcomp (SeparationQuotient.mkCLM _ _)
      SeparationQuotient.isUniformInducing_mk).completeSpace_congr]
    · exact this inferInstance
    · intro f
      use (SeparationQuotient.outCLM _ _).compContinuousMultilinearMap f
      simp [DFunLike.ext_iff]
  have H : ∀ {m : Π i, E i},
      Continuous fun f : (Π i, E i) →ᵤ[{s | IsVonNBounded 𝕜 s}] F ↦ toFun _ f m :=
    (uniformContinuous_eval (isVonNBounded_covers) _).continuous
  rw [completeSpace_iff_isComplete_range isUniformInducing_toUniformOnFun, range_toUniformOnFun]
  simp only [setOf_and, setOf_forall]
  apply_rules [IsClosed.isComplete, IsClosed.inter]
  · exact UniformOnFun.isClosed_setOf_continuous h
  · exact isClosed_iInter fun m ↦ isClosed_iInter fun i ↦
      isClosed_iInter fun x ↦ isClosed_iInter fun y ↦ isClosed_eq H (H.add H)
  · exact isClosed_iInter fun m ↦ isClosed_iInter fun i ↦
      isClosed_iInter fun c ↦ isClosed_iInter fun x ↦ isClosed_eq H (H.const_smul _)


instance instCompleteSpace [∀ i, TopologicalAddGroup (E i)] [SequentialSpace (Π i, E i)] :
    CompleteSpace (ContinuousMultilinearMap 𝕜 E F) :=
  completeSpace <| .of_seq fun _u x hux ↦ (hux.isVonNBounded_range 𝕜).insert x


theorem isUniformEmbedding_restrictScalars :
    IsUniformEmbedding
      (restrictScalars 𝕜' : ContinuousMultilinearMap 𝕜 E F → ContinuousMultilinearMap 𝕜' E F) := by
  letI : NontriviallyNormedField 𝕜 :=
    ⟨let ⟨x, hx⟩ := @NontriviallyNormedField.non_trivial 𝕜' _; ⟨algebraMap 𝕜' 𝕜 x, by simpa⟩⟩
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹⁴ : NormedField 𝕜
    inst✝¹³ : (i : ι) → TopologicalSpace (E i)
    inst✝¹² : (i : ι) → AddCommGroup (E i)
    inst✝¹¹ : (i : ι) → Module 𝕜 (E i)
    inst✝¹⁰ : AddCommGroup F
    inst✝⁹ : Module 𝕜 F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    𝕜' : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜'
    inst✝⁵ : NormedAlgebra 𝕜' 𝕜
    inst✝⁴ : (i : ι) → Module 𝕜' (E i)
    inst✝³ : ∀ (i : ι), IsScalarTower 𝕜' 𝕜 (E i)
    inst✝² : Module 𝕜' F
    inst✝¹ : IsScalarTower 𝕜' 𝕜 F
    inst✝ : ∀ (i : ι), ContinuousSMul 𝕜 (E i)
    this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
    ⊢ IsUniformEmbedding (ContinuousMultilinearMap.restrictScalars 𝕜')
  -/
  rw [← isUniformEmbedding_toUniformOnFun.of_comp_iff]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹⁴ : NormedField 𝕜
    inst✝¹³ : (i : ι) → TopologicalSpace (E i)
    inst✝¹² : (i : ι) → AddCommGroup (E i)
    inst✝¹¹ : (i : ι) → Module 𝕜 (E i)
    inst✝¹⁰ : AddCommGroup F
    inst✝⁹ : Module 𝕜 F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    𝕜' : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜'
    inst✝⁵ : NormedAlgebra 𝕜' 𝕜
    inst✝⁴ : (i : ι) → Module 𝕜' (E i)
    inst✝³ : ∀ (i : ι), IsScalarTower 𝕜' 𝕜 (E i)
    inst✝² : Module 𝕜' F
    inst✝¹ : IsScalarTower 𝕜' 𝕜 F
    inst✝ : ∀ (i : ι), ContinuousSMul 𝕜 (E i)
    this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
    ⊢ IsUniformEmbedding (Function.comp ContinuousMultilinearMap.toUniformOnFun (C …
  -/
  convert isUniformEmbedding_toUniformOnFun using 4 with s
  /-
    case h.e'_4.h.e'_4.h.e'_2.h.a
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹⁴ : NormedField 𝕜
    inst✝¹³ : (i : ι) → TopologicalSpace (E i)
    inst✝¹² : (i : ι) → AddCommGroup (E i)
    inst✝¹¹ : (i : ι) → Module 𝕜 (E i)
    inst✝¹⁰ : AddCommGroup F
    inst✝⁹ : Module 𝕜 F
    inst✝⁸ : UniformSpace F
    inst✝⁷ : UniformAddGroup F
    𝕜' : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜'
    inst✝⁵ : NormedAlgebra 𝕜' 𝕜
    inst✝⁴ : (i : ι) → Module 𝕜' (E i)
    inst✝³ : ∀ (i : ι), IsScalarTower 𝕜' 𝕜 (E i)
    inst✝² : Module 𝕜' F
    inst✝¹ : IsScalarTower 𝕜' 𝕜 F
    inst✝ : ∀ (i : ι), ContinuousSMul 𝕜 (E i)
    this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
    s : Set ((i : ι) → E i)
    ⊢ Iff (Bornology.IsVonNBounded 𝕜' s) (Bornology.IsVonNBounded 𝕜 s)
  -/
  exact ⟨fun h ↦ h.extend_scalars _, fun h ↦ h.restrict_scalars _⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_restrictScalars := isUniformEmbedding_restrictScalars


theorem uniformContinuous_restrictScalars :
    UniformContinuous
      (restrictScalars 𝕜' : ContinuousMultilinearMap 𝕜 E F → ContinuousMultilinearMap 𝕜' E F) :=
  (isUniformEmbedding_restrictScalars 𝕜').uniformContinuous


instance instTopologicalAddGroup : TopologicalAddGroup (ContinuousMultilinearMap 𝕜 E F) :=
  letI := TopologicalAddGroup.toUniformSpace F
  haveI := comm_topologicalAddGroup_is_uniform (G := F)
  inferInstance


instance instContinuousConstSMul
    {M : Type*} [Monoid M] [DistribMulAction M F] [SMulCommClass 𝕜 M F] [ContinuousConstSMul M F] :
    ContinuousConstSMul M (ContinuousMultilinearMap 𝕜 E F) := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : (i : ι) → TopologicalSpace (E i)
    inst✝⁹ : (i : ι) → AddCommGroup (E i)
    inst✝⁸ : (i : ι) → Module 𝕜 (E i)
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : TopologicalAddGroup F
    M : Type u_5
    inst✝³ : Monoid M
    inst✝² : DistribMulAction M F
    inst✝¹ : SMulCommClass 𝕜 M F
    inst✝ : ContinuousConstSMul M F
    ⊢ ContinuousConstSMul M (ContinuousMultilinearMap 𝕜 E F)
  -/
  letI := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : (i : ι) → TopologicalSpace (E i)
    inst✝⁹ : (i : ι) → AddCommGroup (E i)
    inst✝⁸ : (i : ι) → Module 𝕜 (E i)
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : TopologicalAddGroup F
    M : Type u_5
    inst✝³ : Monoid M
    inst✝² : DistribMulAction M F
    inst✝¹ : SMulCommClass 𝕜 M F
    inst✝ : ContinuousConstSMul M F
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ ContinuousConstSMul M (ContinuousMultilinearMap 𝕜 E F)
  -/
  haveI := comm_topologicalAddGroup_is_uniform (G := F)
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : (i : ι) → TopologicalSpace (E i)
    inst✝⁹ : (i : ι) → AddCommGroup (E i)
    inst✝⁸ : (i : ι) → Module 𝕜 (E i)
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : TopologicalAddGroup F
    M : Type u_5
    inst✝³ : Monoid M
    inst✝² : DistribMulAction M F
    inst✝¹ : SMulCommClass 𝕜 M F
    inst✝ : ContinuousConstSMul M F
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ ContinuousConstSMul M (ContinuousMultilinearMap 𝕜 E F)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance instContinuousSMul [ContinuousSMul 𝕜 F] :
    ContinuousSMul 𝕜 (ContinuousMultilinearMap 𝕜 E F) :=
  letI := TopologicalAddGroup.toUniformSpace F
  haveI := comm_topologicalAddGroup_is_uniform (G := F)
  let φ : ContinuousMultilinearMap 𝕜 E F →ₗ[𝕜] (Π i, E i) → F :=
    { toFun := (↑), map_add' := fun _ _ ↦ rfl, map_smul' := fun _ _ ↦ rfl }
  UniformOnFun.continuousSMul_induced_of_image_bounded _ _ _ _ φ
    isEmbedding_toUniformOnFun.isInducing fun _ _ hu ↦ hu.image_multilinear _


theorem hasBasis_nhds_zero_of_basis {ι : Type*} {p : ι → Prop} {b : ι → Set F}
    (h : (𝓝 (0 : F)).HasBasis p b) :
    (𝓝 (0 : ContinuousMultilinearMap 𝕜 E F)).HasBasis
      (fun Si : Set (Π i, E i) × ι => IsVonNBounded 𝕜 Si.1 ∧ p Si.2)
      fun Si => { f | MapsTo f Si.1 (b Si.2) } := by
  /-
    𝕜 : Type u_1
    ι✝ : Type u_2
    E : ι✝ → Type u_3
    F : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : (i : ι✝) → TopologicalSpace (E i)
    inst✝⁵ : (i : ι✝) → AddCommGroup (E i)
    inst✝⁴ : (i : ι✝) → Module 𝕜 (E i)
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    ⊢ (nhds 0).HasBasis (fun Si => And (Bornology.IsVonNBounded 𝕜 Si.1) (p Si.2))  …
  -/
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  /-
    𝕜 : Type u_1
    ι✝ : Type u_2
    E : ι✝ → Type u_3
    F : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : (i : ι✝) → TopologicalSpace (E i)
    inst✝⁵ : (i : ι✝) → AddCommGroup (E i)
    inst✝⁴ : (i : ι✝) → Module 𝕜 (E i)
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    ⊢ (nhds 0).HasBasis (fun Si => And (Bornology.IsVonNBounded 𝕜 Si.1) (p Si.2))  …
  -/
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  /-
    𝕜 : Type u_1
    ι✝ : Type u_2
    E : ι✝ → Type u_3
    F : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : (i : ι✝) → TopologicalSpace (E i)
    inst✝⁵ : (i : ι✝) → AddCommGroup (E i)
    inst✝⁴ : (i : ι✝) → Module 𝕜 (E i)
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ (nhds 0).HasBasis (fun Si => And (Bornology.IsVonNBounded 𝕜 Si.1) (p Si.2))  …
  -/
  rw [nhds_induced]
  /-
    𝕜 : Type u_1
    ι✝ : Type u_2
    E : ι✝ → Type u_3
    F : Type u_4
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : (i : ι✝) → TopologicalSpace (E i)
    inst✝⁵ : (i : ι✝) → AddCommGroup (E i)
    inst✝⁴ : (i : ι✝) → Module 𝕜 (E i)
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    inst✝ : TopologicalAddGroup F
    ι : Type u_5
    p : ι → Prop
    b : ι → Set F
    h : (nhds 0).HasBasis p b
    this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
    this : UniformAddGroup F
    ⊢ (Filter.comap ContinuousMultilinearMap.toUniformOnFun (nhds (ContinuousMulti …
  -/
  refine (UniformOnFun.hasBasis_nhds_zero_of_basis _ ?_ ?_ h).comap DFunLike.coe
    /-
      case refine_1
      𝕜 : Type u_1
      ι✝ : Type u_2
      E : ι✝ → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι✝) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι✝) → AddCommGroup (E i)
      inst✝⁴ : (i : ι✝) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : TopologicalSpace F
      inst✝ : TopologicalAddGroup F
      ι : Type u_5
      p : ι → Prop
      b : ι → Set F
      h : (nhds 0).HasBasis p b
      this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
      this : UniformAddGroup F
      ⊢ (setOf fun s => Bornology.IsVonNBounded 𝕜 s).Nonempty
    -/
  · exact ⟨∅, isVonNBounded_empty _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      ι✝ : Type u_2
      E : ι✝ → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : (i : ι✝) → TopologicalSpace (E i)
      inst✝⁵ : (i : ι✝) → AddCommGroup (E i)
      inst✝⁴ : (i : ι✝) → Module 𝕜 (E i)
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜 F
      inst✝¹ : TopologicalSpace F
      inst✝ : TopologicalAddGroup F
      ι : Type u_5
      p : ι → Prop
      b : ι → Set F
      h : (nhds 0).HasBasis p b
      this✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
      this : UniformAddGroup F
      ⊢ DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) (setOf fun s => Bornology.I …
    -/
  · exact directedOn_of_sup_mem fun _ _ => Bornology.IsVonNBounded.union
    /-
      🎉 no goals
    -/


theorem hasBasis_nhds_zero :
    (𝓝 (0 : ContinuousMultilinearMap 𝕜 E F)).HasBasis
      (fun SV : Set (Π i, E i) × Set F => IsVonNBounded 𝕜 SV.1 ∧ SV.2 ∈ 𝓝 0) fun SV =>
      { f | MapsTo f SV.1 SV.2 } :=
  hasBasis_nhds_zero_of_basis (Filter.basis_sets _)


instance : ContinuousEvalConst (ContinuousMultilinearMap 𝕜 E F) (Π i, E i) F where
  continuous_eval_const x :=
    let _ := TopologicalAddGroup.toUniformSpace F
    have _ := comm_topologicalAddGroup_is_uniform (G := F)
    (uniformContinuous_eval_const x).continuous


@[deprecated (since := "2024-10-05")] protected alias continuous_eval_const := continuous_eval_const

@[deprecated (since := "2024-04-10")] alias continuous_eval_left := continuous_eval_const

@[deprecated (since := "2024-10-05")] protected alias continuous_coe_fun := continuous_coeFun


instance instT2Space [T2Space F] : T2Space (ContinuousMultilinearMap 𝕜 E F) :=
  .of_injective_continuous DFunLike.coe_injective continuous_coeFun


instance instT3Space [T2Space F] : T3Space (ContinuousMultilinearMap 𝕜 E F) :=
  inferInstance


theorem isEmbedding_restrictScalars :
    IsEmbedding
      (restrictScalars 𝕜' : ContinuousMultilinearMap 𝕜 E F → ContinuousMultilinearMap 𝕜' E F) :=
  letI : UniformSpace F := TopologicalAddGroup.toUniformSpace F
  haveI : UniformAddGroup F := comm_topologicalAddGroup_is_uniform
  (isUniformEmbedding_restrictScalars _).isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding_restrictScalars := isEmbedding_restrictScalars


@[continuity, fun_prop]
theorem continuous_restrictScalars :
    Continuous
      (restrictScalars 𝕜' : ContinuousMultilinearMap 𝕜 E F → ContinuousMultilinearMap 𝕜' E F) :=
   isEmbedding_restrictScalars.continuous


variable (𝕜') in
/-- `ContinuousMultilinearMap.restrictScalars` as a `ContinuousLinearMap`. -/
@[simps (config := .asFn) apply]
def restrictScalarsLinear [ContinuousConstSMul 𝕜' F] :
    ContinuousMultilinearMap 𝕜 E F →L[𝕜'] ContinuousMultilinearMap 𝕜' E F where
  toFun := restrictScalars 𝕜'
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- The application of a multilinear map as a `ContinuousLinearMap`. -/
def apply [ContinuousConstSMul 𝕜 F] (m : Π i, E i) : ContinuousMultilinearMap 𝕜 E F →L[𝕜] F where
  toFun c := c m
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  cont := continuous_eval_const m


@[simp]
lemma apply_apply [ContinuousConstSMul 𝕜 F] {m : Π i, E i} {c : ContinuousMultilinearMap 𝕜 E F} :
    apply 𝕜 E F m c = c m := rfl


theorem hasSum_eval {α : Type*} {p : α → ContinuousMultilinearMap 𝕜 E F}
    {q : ContinuousMultilinearMap 𝕜 E F} (h : HasSum p q) (m : Π i, E i) :
    HasSum (fun a => p a m) (q m) :=
  h.map (applyAddHom m) (continuous_eval_const m)


theorem tsum_eval [T2Space F] {α : Type*} {p : α → ContinuousMultilinearMap 𝕜 E F} (hp : Summable p)
    (m : Π i, E i) : (∑' a, p a) m = ∑' a, p a m :=
  (hasSum_eval hp.hasSum m).tsum_eq.symm


