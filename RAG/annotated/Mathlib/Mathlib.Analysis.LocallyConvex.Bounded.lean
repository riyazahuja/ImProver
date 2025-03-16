/-- A set `s` is von Neumann bounded if every neighborhood of 0 absorbs `s`. -/
def IsVonNBounded (s : Set E) : Prop :=
  ∀ ⦃V⦄, V ∈ 𝓝 (0 : E) → Absorbs 𝕜 V s


@[simp]
theorem isVonNBounded_empty : IsVonNBounded 𝕜 (∅ : Set E) := fun _ _ => Absorbs.empty


theorem isVonNBounded_iff (s : Set E) : IsVonNBounded 𝕜 s ↔ ∀ V ∈ 𝓝 (0 : E), Absorbs 𝕜 V s :=
  Iff.rfl


theorem _root_.Filter.HasBasis.isVonNBounded_iff {q : ι → Prop} {s : ι → Set E} {A : Set E}
    (h : (𝓝 (0 : E)).HasBasis q s) : IsVonNBounded 𝕜 A ↔ ∀ i, q i → Absorbs 𝕜 (s i) A := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    q : ι → Prop
    s : ι → Set E
    A : Set E
    h : (nhds 0).HasBasis q s
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 A) (∀ (i : ι), q i → Absorbs 𝕜 (s i) A)
  -/
  refine ⟨fun hA i hi => hA (h.mem_of_mem hi), fun hA V hV => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    q : ι → Prop
    s : ι → Set E
    A : Set E
    h : (nhds 0).HasBasis q s
    hA : ∀ (i : ι), q i → Absorbs 𝕜 (s i) A
    V : Set E
    hV : Membership.mem (nhds 0) V
    ⊢ Absorbs 𝕜 V A
  -/
  rcases h.mem_iff.mp hV with ⟨i, hi, hV⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    q : ι → Prop
    s : ι → Set E
    A : Set E
    h : (nhds 0).HasBasis q s
    hA : ∀ (i : ι), q i → Absorbs 𝕜 (s i) A
    V : Set E
    hV✝ : Membership.mem (nhds 0) V
    i : ι
    hi : q i
    hV : HasSubset.Subset (s i) V
    ⊢ Absorbs 𝕜 V A
  -/
  exact (hA i hi).mono_left hV
  /-
    🎉 no goals
  -/


/-- Subsets of bounded sets are bounded. -/
theorem IsVonNBounded.subset {s₁ s₂ : Set E} (h : s₁ ⊆ s₂) (hs₂ : IsVonNBounded 𝕜 s₂) :
    IsVonNBounded 𝕜 s₁ := fun _ hV => (hs₂ hV).mono_right h


@[simp]
theorem isVonNBounded_union {s t : Set E} :
    IsVonNBounded 𝕜 (s ∪ t) ↔ IsVonNBounded 𝕜 s ∧ IsVonNBounded 𝕜 t := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    s t : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Union.union s t)) (And (Bornology.IsVonNBoun …
  -/
  simp only [IsVonNBounded, absorbs_union, forall_and]
  /-
    🎉 no goals
  -/


/-- The union of two bounded sets is bounded. -/
theorem IsVonNBounded.union {s₁ s₂ : Set E} (hs₁ : IsVonNBounded 𝕜 s₁) (hs₂ : IsVonNBounded 𝕜 s₂) :
    IsVonNBounded 𝕜 (s₁ ∪ s₂) := isVonNBounded_union.2 ⟨hs₁, hs₂⟩


@[nontriviality]
theorem IsVonNBounded.of_boundedSpace [BoundedSpace 𝕜] {s : Set E} : IsVonNBounded 𝕜 s := fun _ _ ↦
  .of_boundedSpace


@[nontriviality]
theorem IsVonNBounded.of_subsingleton [Subsingleton E] {s : Set E} : IsVonNBounded 𝕜 s :=
  fun U hU ↦ .of_forall fun c ↦ calc
    s ⊆ univ := subset_univ s
    _ = c • U := .symm <| Subsingleton.eq_univ_of_nonempty <| (Filter.nonempty_of_mem hU).image _


@[simp]
theorem isVonNBounded_iUnion {ι : Sort*} [Finite ι] {s : ι → Set E} :
    IsVonNBounded 𝕜 (⋃ i, s i) ↔ ∀ i, IsVonNBounded 𝕜 (s i) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : SMul 𝕜 E
    inst✝² : Zero E
    inst✝¹ : TopologicalSpace E
    ι : Sort u_6
    inst✝ : Finite ι
    s : ι → Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Set.iUnion fun i => s i)) (∀ (i : ι), Bornol …
  -/
  simp only [IsVonNBounded, absorbs_iUnion, @forall_swap ι]
  /-
    🎉 no goals
  -/


theorem isVonNBounded_biUnion {ι : Type*} {I : Set ι} (hI : I.Finite) {s : ι → Set E} :
    IsVonNBounded 𝕜 (⋃ i ∈ I, s i) ↔ ∀ i ∈ I, IsVonNBounded 𝕜 (s i) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    ι : Type u_6
    I : Set ι
    hI : I.Finite
    s : ι → Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Set.iUnion fun i => Set.iUnion fun h => s i) …
  -/
  have _ := hI.to_subtype
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    ι : Type u_6
    I : Set ι
    hI : I.Finite
    s : ι → Set E
    x✝ : Finite ↑I
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Set.iUnion fun i => Set.iUnion fun h => s i) …
  -/
  rw [biUnion_eq_iUnion, isVonNBounded_iUnion, Subtype.forall]
  /-
    🎉 no goals
  -/


theorem isVonNBounded_sUnion {S : Set (Set E)} (hS : S.Finite) :
    IsVonNBounded 𝕜 (⋃₀ S) ↔ ∀ s ∈ S, IsVonNBounded 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    S : Set (Set E)
    hS : S.Finite
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 S.sUnion) (∀ (s : Set E), Membership.mem S s  …
  -/
  rw [sUnion_eq_biUnion, isVonNBounded_biUnion hS]
  /-
    🎉 no goals
  -/


protected theorem IsVonNBounded.add (hs : IsVonNBounded 𝕜 s) (ht : IsVonNBounded 𝕜 t) :
    IsVonNBounded 𝕜 (s + t) := fun U hU ↦ by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : AddZeroClass E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : DistribSMul 𝕜 E
    s t : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    ht : Bornology.IsVonNBounded 𝕜 t
    U : Set E
    hU : Membership.mem (nhds 0) U
    ⊢ Absorbs 𝕜 U (HAdd.hAdd s t)
  -/
  rcases exists_open_nhds_zero_add_subset hU with ⟨V, hVo, hV, hVU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : AddZeroClass E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : DistribSMul 𝕜 E
    s t : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    ht : Bornology.IsVonNBounded 𝕜 t
    U : Set E
    hU : Membership.mem (nhds 0) U
    V : Set E
    hVo : IsOpen V
    hV : Membership.mem V 0
    hVU : HasSubset.Subset (HAdd.hAdd V V) U
    ⊢ Absorbs 𝕜 U (HAdd.hAdd s t)
  -/
  exact ((hs <| hVo.mem_nhds hV).add (ht <| hVo.mem_nhds hV)).mono_left hVU
  /-
    🎉 no goals
  -/


protected theorem IsVonNBounded.neg (hs : IsVonNBounded 𝕜 s) : IsVonNBounded 𝕜 (-s) := fun U hU ↦ by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : AddGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : DistribMulAction 𝕜 E
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    U : Set E
    hU : Membership.mem (nhds 0) U
    ⊢ Absorbs 𝕜 U (Neg.neg s)
  -/
  rw [← neg_neg U]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : AddGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : DistribMulAction 𝕜 E
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    U : Set E
    hU : Membership.mem (nhds 0) U
    ⊢ Absorbs 𝕜 (Neg.neg (Neg.neg U)) (Neg.neg s)
  -/
  exact (hs <| neg_mem_nhds_zero _ hU).neg_neg
  /-
    🎉 no goals
  -/


@[simp]
theorem isVonNBounded_neg : IsVonNBounded 𝕜 (-s) ↔ IsVonNBounded 𝕜 s :=
  ⟨fun h ↦ neg_neg s ▸ h.neg, fun h ↦ h.neg⟩


alias ⟨IsVonNBounded.of_neg, _⟩ := isVonNBounded_neg


protected theorem IsVonNBounded.sub (hs : IsVonNBounded 𝕜 s) (ht : IsVonNBounded 𝕜 t) :
    IsVonNBounded 𝕜 (s - t) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : AddGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : DistribMulAction 𝕜 E
    s t : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    ht : Bornology.IsVonNBounded 𝕜 t
    ⊢ Bornology.IsVonNBounded 𝕜 (HSub.hSub s t)
  -/
  rw [sub_eq_add_neg]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : SeminormedRing 𝕜
    inst✝³ : AddGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : DistribMulAction 𝕜 E
    s t : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    ht : Bornology.IsVonNBounded 𝕜 t
    ⊢ Bornology.IsVonNBounded 𝕜 (HAdd.hAdd s (Neg.neg t))
  -/
  exact hs.add ht.neg
  /-
    🎉 no goals
  -/


/-- If a topology `t'` is coarser than `t`, then any set `s` that is bounded with respect to
`t` is bounded with respect to `t'`. -/
theorem IsVonNBounded.of_topologicalSpace_le {t t' : TopologicalSpace E} (h : t ≤ t') {s : Set E}
    (hs : @IsVonNBounded 𝕜 E _ _ _ t s) : @IsVonNBounded 𝕜 E _ _ _ t' s := fun _ hV =>
  hs <| (le_iff_nhds t t').mp h 0 hV


lemma isVonNBounded_iff_tendsto_smallSets_nhds {𝕜 E : Type*} [NormedDivisionRing 𝕜]
    [AddCommGroup E] [Module 𝕜 E] [TopologicalSpace E] {S : Set E} :
    IsVonNBounded 𝕜 S ↔ Tendsto (· • S : 𝕜 → Set E) (𝓝 0) (𝓝 0).smallSets := by
  /-
    𝕜 : Type u_6
    E : Type u_7
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : TopologicalSpace E
    S : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 S) (Filter.Tendsto (fun x => HSMul.hSMul x S) …
  -/
  rw [tendsto_smallSets_iff]
  /-
    𝕜 : Type u_6
    E : Type u_7
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : TopologicalSpace E
    S : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 S) (∀ (t : Set E), Membership.mem (nhds 0) t  …
  -/
  refine forall₂_congr fun V hV ↦ ?_
  /-
    𝕜 : Type u_6
    E : Type u_7
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : TopologicalSpace E
    S V : Set E
    hV : Membership.mem (nhds 0) V
    ⊢ Iff (Absorbs 𝕜 V S) (Filter.Eventually (fun x => HasSubset.Subset (HSMul.hSM …
  -/
  simp only [absorbs_iff_eventually_nhds_zero (mem_of_mem_nhds hV), mapsTo', image_smul]
  /-
    🎉 no goals
  -/


alias ⟨IsVonNBounded.tendsto_smallSets_nhds, _⟩ := isVonNBounded_iff_tendsto_smallSets_nhds


lemma isVonNBounded_iff_absorbing_le {𝕜 E : Type*} [NormedDivisionRing 𝕜]
    [AddCommGroup E] [Module 𝕜 E] [TopologicalSpace E] {S : Set E} :
    IsVonNBounded 𝕜 S ↔ Filter.absorbing 𝕜 S ≤ 𝓝 0 :=
  .rfl


lemma isVonNBounded_pi_iff {𝕜 ι : Type*} {E : ι → Type*} [NormedDivisionRing 𝕜]
    [∀ i, AddCommGroup (E i)] [∀ i, Module 𝕜 (E i)] [∀ i, TopologicalSpace (E i)]
    {S : Set (∀ i, E i)} : IsVonNBounded 𝕜 S ↔ ∀ i, IsVonNBounded 𝕜 (eval i '' S) := by
  simp_rw [isVonNBounded_iff_tendsto_smallSets_nhds, nhds_pi, Filter.pi, smallSets_iInf,
    smallSets_comap_eq_comap_image, tendsto_iInf, tendsto_comap_iff, Function.comp_def,
    ← image_smul, image_image, eval, Pi.smul_apply, Pi.zero_apply]


/-- A continuous linear image of a bounded set is bounded. -/
theorem IsVonNBounded.image {σ : 𝕜₁ →+* 𝕜₂} [RingHomSurjective σ] [RingHomIsometric σ] {s : Set E}
    (hs : IsVonNBounded 𝕜₁ s) (f : E →SL[σ] F) : IsVonNBounded 𝕜₂ (f '' s) := by
  /-
    E : Type u_3
    F : Type u_4
    𝕜₁ : Type u_6
    𝕜₂ : Type u_7
    inst✝⁹ : NormedDivisionRing 𝕜₁
    inst✝⁸ : NormedDivisionRing 𝕜₂
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalSpace F
    σ : RingHom 𝕜₁ 𝕜₂
    inst✝¹ : RingHomSurjective σ
    inst✝ : RingHomIsometric σ
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜₁ s
    f : ContinuousLinearMap σ E F
    ⊢ Bornology.IsVonNBounded 𝕜₂ (Set.image (⇑f) s)
  -/
  have σ_iso : Isometry σ := AddMonoidHomClass.isometry_of_norm σ fun x => RingHomIsometric.is_iso
  have : map σ (𝓝 0) = 𝓝 0 := by
    rw [σ_iso.isEmbedding.map_nhds_eq, σ.surjective.range_eq, nhdsWithin_univ, map_zero]
  /-
    E : Type u_3
    F : Type u_4
    𝕜₁ : Type u_6
    𝕜₂ : Type u_7
    inst✝⁹ : NormedDivisionRing 𝕜₁
    inst✝⁸ : NormedDivisionRing 𝕜₂
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalSpace F
    σ : RingHom 𝕜₁ 𝕜₂
    inst✝¹ : RingHomSurjective σ
    inst✝ : RingHomIsometric σ
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜₁ s
    f : ContinuousLinearMap σ E F
    σ_iso : Isometry ⇑σ
    this : Eq (Filter.map (⇑σ) (nhds 0)) (nhds 0)
    ⊢ Bornology.IsVonNBounded 𝕜₂ (Set.image (⇑f) s)
  -/
  have hf₀ : Tendsto f (𝓝 0) (𝓝 0) := f.continuous.tendsto' 0 0 (map_zero f)
  /-
    E : Type u_3
    F : Type u_4
    𝕜₁ : Type u_6
    𝕜₂ : Type u_7
    inst✝⁹ : NormedDivisionRing 𝕜₁
    inst✝⁸ : NormedDivisionRing 𝕜₂
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalSpace F
    σ : RingHom 𝕜₁ 𝕜₂
    inst✝¹ : RingHomSurjective σ
    inst✝ : RingHomIsometric σ
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜₁ s
    f : ContinuousLinearMap σ E F
    σ_iso : Isometry ⇑σ
    this : Eq (Filter.map (⇑σ) (nhds 0)) (nhds 0)
    hf₀ : Filter.Tendsto (⇑f) (nhds 0) (nhds 0)
    ⊢ Bornology.IsVonNBounded 𝕜₂ (Set.image (⇑f) s)
  -/
  simp only [isVonNBounded_iff_tendsto_smallSets_nhds, ← this, tendsto_map'_iff] at hs ⊢
  /-
    E : Type u_3
    F : Type u_4
    𝕜₁ : Type u_6
    𝕜₂ : Type u_7
    inst✝⁹ : NormedDivisionRing 𝕜₁
    inst✝⁸ : NormedDivisionRing 𝕜₂
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜₁ E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module 𝕜₂ F
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalSpace F
    σ : RingHom 𝕜₁ 𝕜₂
    inst✝¹ : RingHomSurjective σ
    inst✝ : RingHomIsometric σ
    s : Set E
    f : ContinuousLinearMap σ E F
    σ_iso : Isometry ⇑σ
    this : Eq (Filter.map (⇑σ) (nhds 0)) (nhds 0)
    hf₀ : Filter.Tendsto (⇑f) (nhds 0) (nhds 0)
    hs : Filter.Tendsto (fun x => HSMul.hSMul x s) (nhds 0) (nhds 0).smallSets
    ⊢ Filter.Tendsto (Function.comp (fun x => HSMul.hSMul x (Set.image (⇑f) s)) ⇑σ …
  -/
  simpa only [comp_def, image_smul_setₛₗ _ _ σ f] using hf₀.image_smallSets.comp hs
  /-
    🎉 no goals
  -/


theorem IsVonNBounded.smul_tendsto_zero [NormedField 𝕜]
    [AddCommGroup E] [Module 𝕜 E] [TopologicalSpace E]
    {S : Set E} {ε : ι → 𝕜} {x : ι → E} {l : Filter ι}
    (hS : IsVonNBounded 𝕜 S) (hxS : ∀ᶠ n in l, x n ∈ S) (hε : Tendsto ε l (𝓝 0)) :
    Tendsto (ε • x) l (𝓝 0) :=
  (hS.tendsto_smallSets_nhds.comp hε).of_smallSets <| hxS.mono fun _ ↦ smul_mem_smul_set


theorem isVonNBounded_of_smul_tendsto_zero {ε : ι → 𝕜} {l : Filter ι} [l.NeBot]
    (hε : ∀ᶠ n in l, ε n ≠ 0) {S : Set E}
    (H : ∀ x : ι → E, (∀ n, x n ∈ S) → Tendsto (ε • x) l (𝓝 0)) : IsVonNBounded 𝕜 S := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    ε : ι → 𝕜
    l : Filter ι
    inst✝ : l.NeBot
    hε : Filter.Eventually (fun n => Ne (ε n) 0) l
    S : Set E
    H : ∀ (x : ι → E), (∀ (n : ι), Membership.mem S (x n)) → Filter.Tendsto (HSMul …
    ⊢ Bornology.IsVonNBounded 𝕜 S
  -/
  rw [(nhds_basis_balanced 𝕜 E).isVonNBounded_iff]
  /-
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    ε : ι → 𝕜
    l : Filter ι
    inst✝ : l.NeBot
    hε : Filter.Eventually (fun n => Ne (ε n) 0) l
    S : Set E
    H : ∀ (x : ι → E), (∀ (n : ι), Membership.mem S (x n)) → Filter.Tendsto (HSMul …
    ⊢ ∀ (i : Set E), And (Membership.mem (nhds 0) i) (Balanced 𝕜 i) → Absorbs 𝕜 (i …
  -/
  by_contra! H'
  /-
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    ε : ι → 𝕜
    l : Filter ι
    inst✝ : l.NeBot
    hε : Filter.Eventually (fun n => Ne (ε n) 0) l
    S : Set E
    H : ∀ (x : ι → E), (∀ (n : ι), Membership.mem S (x n)) → Filter.Tendsto (HSMul …
    H' : Exists fun i => And (And (Membership.mem (nhds 0) i) (Balanced 𝕜 i)) (Not …
    ⊢ False
  -/
  rcases H' with ⟨V, ⟨hV, hVb⟩, hVS⟩
  have : ∀ᶠ n in l, ∃ x : S, ε n • (x : E) ∉ V := by
    filter_upwards [hε] with n hn
    rw [absorbs_iff_norm] at hVS
    push_neg at hVS
    rcases hVS ‖(ε n)⁻¹‖ with ⟨a, haε, haS⟩
    rcases Set.not_subset.mp haS with ⟨x, hxS, hx⟩
    refine ⟨⟨x, hxS⟩, fun hnx => ?_⟩
    rw [← Set.mem_inv_smul_set_iff₀ hn] at hnx
    exact hx (hVb.smul_mono haε hnx)
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    ε : ι → 𝕜
    l : Filter ι
    inst✝ : l.NeBot
    hε : Filter.Eventually (fun n => Ne (ε n) 0) l
    S : Set E
    H : ∀ (x : ι → E), (∀ (n : ι), Membership.mem S (x n)) → Filter.Tendsto (HSMul …
    V : Set E
    hVS : Not (Absorbs 𝕜 (id V) S)
    hV : Membership.mem (nhds 0) V
    hVb : Balanced 𝕜 V
    this : Filter.Eventually (fun n => Exists fun x => Not (Membership.mem V (HSMu …
    ⊢ False
  -/
  rcases this.choice with ⟨x, hx⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    ε : ι → 𝕜
    l : Filter ι
    inst✝ : l.NeBot
    hε : Filter.Eventually (fun n => Ne (ε n) 0) l
    S : Set E
    H : ∀ (x : ι → E), (∀ (n : ι), Membership.mem S (x n)) → Filter.Tendsto (HSMul …
    V : Set E
    hVS : Not (Absorbs 𝕜 (id V) S)
    hV : Membership.mem (nhds 0) V
    hVb : Balanced 𝕜 V
    this : Filter.Eventually (fun n => Exists fun x => Not (Membership.mem V (HSMu …
    x : ι → ↑S
    hx : Filter.Eventually (fun x_1 => Not (Membership.mem V (HSMul.hSMul (ε x_1)  …
    ⊢ False
  -/
  refine Filter.frequently_false l (Filter.Eventually.frequently ?_)
  filter_upwards [hx,
    (H (_ ∘ x) fun n => (x n).2).eventually (eventually_mem_set.mpr hV)] using fun n => id


/-- Given any sequence `ε` of scalars which tends to `𝓝[≠] 0`, we have that a set `S` is bounded
  if and only if for any sequence `x : ℕ → S`, `ε • x` tends to 0. This actually works for any
  indexing type `ι`, but in the special case `ι = ℕ` we get the important fact that convergent
  sequences fully characterize bounded sets. -/
theorem isVonNBounded_iff_smul_tendsto_zero {ε : ι → 𝕜} {l : Filter ι} [l.NeBot]
    (hε : Tendsto ε l (𝓝[≠] 0)) {S : Set E} :
    IsVonNBounded 𝕜 S ↔ ∀ x : ι → E, (∀ n, x n ∈ S) → Tendsto (ε • x) l (𝓝 0) :=
  ⟨fun hS _ hxS => hS.smul_tendsto_zero (Eventually.of_forall hxS) (le_trans hε nhdsWithin_le_nhds),
                                           /-
                                             𝕜 : Type u_1
                                             E : Type u_3
                                             ι : Type u_5
                                             inst✝⁵ : NontriviallyNormedField 𝕜
                                             inst✝⁴ : AddCommGroup E
                                             inst✝³ : Module 𝕜 E
                                             inst✝² : TopologicalSpace E
                                             inst✝¹ : ContinuousSMul 𝕜 E
                                             ε : ι → 𝕜
                                             l : Filter ι
                                             inst✝ : l.NeBot
                                             hε : Filter.Tendsto ε l (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0)))
                                             S : Set E
                                             ⊢ Filter.Eventually (fun n => Ne (ε n) 0) l
                                           -/
    isVonNBounded_of_smul_tendsto_zero (by exact hε self_mem_nhdsWithin)⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- If a set is von Neumann bounded with respect to a smaller field,
then it is also von Neumann bounded with respect to a larger field.
See also `Bornology.IsVonNBounded.restrict_scalars` below. -/
theorem IsVonNBounded.extend_scalars [NontriviallyNormedField 𝕜]
    {E : Type*} [AddCommGroup E] [Module 𝕜 E]
    (𝕝 : Type*) [NontriviallyNormedField 𝕝] [NormedAlgebra 𝕜 𝕝]
    [Module 𝕝 E] [TopologicalSpace E] [ContinuousSMul 𝕝 E] [IsScalarTower 𝕜 𝕝 E]
    {s : Set E} (h : IsVonNBounded 𝕜 s) : IsVonNBounded 𝕝 s := by
  obtain ⟨ε, hε, hε₀⟩ : ∃ ε : ℕ → 𝕜, Tendsto ε atTop (𝓝 0) ∧ ∀ᶠ n in atTop, ε n ≠ 0 := by
    simpa only [tendsto_nhdsWithin_iff] using exists_seq_tendsto (𝓝[≠] (0 : 𝕜))
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_6
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    𝕝 : Type u_7
    inst✝⁵ : NontriviallyNormedField 𝕝
    inst✝⁴ : NormedAlgebra 𝕜 𝕝
    inst✝³ : Module 𝕝 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕝 E
    inst✝ : IsScalarTower 𝕜 𝕝 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ε : Nat → 𝕜
    hε : Filter.Tendsto ε Filter.atTop (nhds 0)
    hε₀ : Filter.Eventually (fun n => Ne (ε n) 0) Filter.atTop
    ⊢ Bornology.IsVonNBounded 𝕝 s
  -/
  refine isVonNBounded_of_smul_tendsto_zero (ε := (ε · • 1)) (by simpa) fun x hx ↦ ?_
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_6
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    𝕝 : Type u_7
    inst✝⁵ : NontriviallyNormedField 𝕝
    inst✝⁴ : NormedAlgebra 𝕜 𝕝
    inst✝³ : Module 𝕝 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕝 E
    inst✝ : IsScalarTower 𝕜 𝕝 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ε : Nat → 𝕜
    hε : Filter.Tendsto ε Filter.atTop (nhds 0)
    hε₀ : Filter.Eventually (fun n => Ne (ε n) 0) Filter.atTop
    x : Nat → E
    hx : ∀ (n : Nat), Membership.mem s (x n)
    ⊢ Filter.Tendsto (HSMul.hSMul (fun x => HSMul.hSMul (ε x) 1) x) Filter.atTop ( …
  -/
  have := h.smul_tendsto_zero (.of_forall hx) hε
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_6
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    𝕝 : Type u_7
    inst✝⁵ : NontriviallyNormedField 𝕝
    inst✝⁴ : NormedAlgebra 𝕜 𝕝
    inst✝³ : Module 𝕝 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕝 E
    inst✝ : IsScalarTower 𝕜 𝕝 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ε : Nat → 𝕜
    hε : Filter.Tendsto ε Filter.atTop (nhds 0)
    hε₀ : Filter.Eventually (fun n => Ne (ε n) 0) Filter.atTop
    x : Nat → E
    hx : ∀ (n : Nat), Membership.mem s (x n)
    this : Filter.Tendsto (HSMul.hSMul ε x) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (HSMul.hSMul (fun x => HSMul.hSMul (ε x) 1) x) Filter.atTop ( …
  -/
  simpa only [Pi.smul_def', smul_one_smul]
  /-
    🎉 no goals
  -/


/-- Singletons are bounded. -/
theorem isVonNBounded_singleton (x : E) : IsVonNBounded 𝕜 ({x} : Set E) := fun _ hV =>
  (absorbent_nhds_zero hV).absorbs


@[simp]
theorem isVonNBounded_insert (x : E) {s : Set E} :
    IsVonNBounded 𝕜 (insert x s) ↔ IsVonNBounded 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    x : E
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Insert.insert x s)) (Bornology.IsVonNBounded …
  -/
  simp only [← singleton_union, isVonNBounded_union, isVonNBounded_singleton, true_and]
  /-
    🎉 no goals
  -/


protected alias ⟨_, IsVonNBounded.insert⟩ := isVonNBounded_insert


protected theorem IsVonNBounded.vadd (hs : IsVonNBounded 𝕜 s) (x : E) :
    IsVonNBounded 𝕜 (x +ᵥ s) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : ContinuousAdd E
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    x : E
    ⊢ Bornology.IsVonNBounded 𝕜 (HVAdd.hVAdd x s)
  -/
  rw [← singleton_vadd]
  -- TODO: dot notation timeouts in the next line
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : ContinuousAdd E
    s : Set E
    hs : Bornology.IsVonNBounded 𝕜 s
    x : E
    ⊢ Bornology.IsVonNBounded 𝕜 (HVAdd.hVAdd (Singleton.singleton x) s)
  -/
  exact IsVonNBounded.add (isVonNBounded_singleton x) hs
  /-
    🎉 no goals
  -/


@[simp]
theorem isVonNBounded_vadd (x : E) : IsVonNBounded 𝕜 (x +ᵥ s) ↔ IsVonNBounded 𝕜 s :=
              /-
                𝕜 : Type u_1
                E : Type u_3
                inst✝⁵ : NormedField 𝕜
                inst✝⁴ : AddCommGroup E
                inst✝³ : Module 𝕜 E
                inst✝² : TopologicalSpace E
                inst✝¹ : ContinuousSMul 𝕜 E
                inst✝ : ContinuousAdd E
                s : Set E
                x : E
                h : Bornology.IsVonNBounded 𝕜 (HVAdd.hVAdd x s)
                ⊢ Bornology.IsVonNBounded 𝕜 s
              -/
  ⟨fun h ↦ by simpa using h.vadd (-x), fun h ↦ h.vadd x⟩
              /-
                🎉 no goals
              -/


theorem IsVonNBounded.of_add_right (hst : IsVonNBounded 𝕜 (s + t)) (hs : s.Nonempty) :
    IsVonNBounded 𝕜 t :=
  let ⟨x, hx⟩ := hs
  (isVonNBounded_vadd x).mp <| hst.subset <| image_subset_image2_right hx


theorem IsVonNBounded.of_add_left (hst : IsVonNBounded 𝕜 (s + t)) (ht : t.Nonempty) :
    IsVonNBounded 𝕜 s :=
  ((add_comm s t).subst hst).of_add_right ht


theorem isVonNBounded_add_of_nonempty (hs : s.Nonempty) (ht : t.Nonempty) :
    IsVonNBounded 𝕜 (s + t) ↔ IsVonNBounded 𝕜 s ∧ IsVonNBounded 𝕜 t :=
  ⟨fun h ↦ ⟨h.of_add_left ht, h.of_add_right hs⟩, and_imp.2 IsVonNBounded.add⟩


theorem isVonNBounded_add :
    IsVonNBounded 𝕜 (s + t) ↔ s = ∅ ∨ t = ∅ ∨ IsVonNBounded 𝕜 s ∧ IsVonNBounded 𝕜 t := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : ContinuousAdd E
    s t : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (HAdd.hAdd s t)) (Or (Eq s EmptyCollection.em …
  -/
  rcases s.eq_empty_or_nonempty with rfl | hs; · simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : ContinuousAdd E
    s t : Set E
    hs : s.Nonempty
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (HAdd.hAdd s t)) (Or (Eq s EmptyCollection.em …
  -/
  rcases t.eq_empty_or_nonempty with rfl | ht; · simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case inr.inr
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : ContinuousAdd E
    s t : Set E
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (HAdd.hAdd s t)) (Or (Eq s EmptyCollection.em …
  -/
  simp [hs.ne_empty, ht.ne_empty, isVonNBounded_add_of_nonempty hs ht]
  /-
    🎉 no goals
  -/


@[simp]
theorem isVonNBounded_add_self : IsVonNBounded 𝕜 (s + s) ↔ IsVonNBounded 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : ContinuousAdd E
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (HAdd.hAdd s s)) (Bornology.IsVonNBounded 𝕜 s)
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  rcases s.eq_empty_or_nonempty with rfl | hs <;> simp [isVonNBounded_add_of_nonempty, *]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem IsVonNBounded.of_sub_left (hst : IsVonNBounded 𝕜 (s - t)) (ht : t.Nonempty) :
    IsVonNBounded 𝕜 s :=
  ((sub_eq_add_neg s t).subst hst).of_add_left ht.neg


theorem IsVonNBounded.of_sub_right (hst : IsVonNBounded 𝕜 (s - t)) (hs : s.Nonempty) :
    IsVonNBounded 𝕜 t :=
  (((sub_eq_add_neg s t).subst hst).of_add_right hs).of_neg


theorem isVonNBounded_sub_of_nonempty (hs : s.Nonempty) (ht : t.Nonempty) :
    IsVonNBounded 𝕜 (s - t) ↔ IsVonNBounded 𝕜 s ∧ IsVonNBounded 𝕜 t := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : TopologicalAddGroup E
    s t : Set E
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (HSub.hSub s t)) (And (Bornology.IsVonNBounde …
  -/
  simp [sub_eq_add_neg, isVonNBounded_add_of_nonempty, hs, ht]
  /-
    🎉 no goals
  -/


theorem isVonNBounded_sub :
    IsVonNBounded 𝕜 (s - t) ↔ s = ∅ ∨ t = ∅ ∨ IsVonNBounded 𝕜 s ∧ IsVonNBounded 𝕜 t := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : TopologicalAddGroup E
    s t : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (HSub.hSub s t)) (Or (Eq s EmptyCollection.em …
  -/
  simp [sub_eq_add_neg, isVonNBounded_add]
  /-
    🎉 no goals
  -/


/-- The union of all bounded set is the whole space. -/
theorem isVonNBounded_covers : ⋃₀ setOf (IsVonNBounded 𝕜) = (Set.univ : Set E) :=
  Set.eq_univ_iff_forall.mpr fun x =>
    Set.mem_sUnion.mpr ⟨{x}, isVonNBounded_singleton _, Set.mem_singleton _⟩


/-- The von Neumann bornology defined by the von Neumann bounded sets.

Note that this is not registered as an instance, in order to avoid diamonds with the
metric bornology. -/
abbrev vonNBornology : Bornology E :=
  Bornology.ofBounded (setOf (IsVonNBounded 𝕜)) (isVonNBounded_empty 𝕜 E)
    (fun _ hs _ ht => hs.subset ht) (fun _ hs _ => hs.union) isVonNBounded_singleton


@[simp]
theorem isBounded_iff_isVonNBounded {s : Set E} :
    @IsBounded _ (vonNBornology 𝕜 E) s ↔ IsVonNBounded 𝕜 s :=
  isBounded_ofBounded_iff _


theorem TotallyBounded.isVonNBounded {s : Set E} (hs : TotallyBounded s) :
    Bornology.IsVonNBounded 𝕜 s := by
  if h : ∃ x : 𝕜, 1 < ‖x‖ then
    letI : NontriviallyNormedField 𝕜 := ⟨h⟩
    rw [totallyBounded_iff_subset_finite_iUnion_nhds_zero] at hs
    intro U hU
    have h : Filter.Tendsto (fun x : E × E => x.fst + x.snd) (𝓝 0) (𝓝 0) :=
      continuous_add.tendsto' _ _ (zero_add _)
    have h' := (nhds_basis_balanced 𝕜 E).prod (nhds_basis_balanced 𝕜 E)
    simp_rw [← nhds_prod_eq, id] at h'
    rcases h.basis_left h' U hU with ⟨x, hx, h''⟩
    rcases hs x.snd hx.2.1 with ⟨t, ht, hs⟩
    refine Absorbs.mono_right ?_ hs
    rw [ht.absorbs_biUnion]
    have hx_fstsnd : x.fst + x.snd ⊆ U := add_subset_iff.mpr fun z1 hz1 z2 hz2 ↦
      h'' <| mk_mem_prod hz1 hz2
    refine fun y _ => Absorbs.mono_left ?_ hx_fstsnd
    -- TODO: with dot notation, Lean timeouts on the next line. Why?
    exact Absorbent.vadd_absorbs (absorbent_nhds_zero hx.1.1) hx.2.2.absorbs_self
  else
    haveI : BoundedSpace 𝕜 := ⟨Metric.isBounded_iff.2 ⟨1, by simp_all [dist_eq_norm]⟩⟩
    exact Bornology.IsVonNBounded.of_boundedSpace


variable (𝕜) in
theorem Filter.Tendsto.isVonNBounded_range [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    [TopologicalSpace E] [TopologicalAddGroup E] [ContinuousSMul 𝕜 E]
    {f : ℕ → E} {x : E} (hf : Tendsto f atTop (𝓝 x)) : Bornology.IsVonNBounded 𝕜 (range f) :=
  letI := TopologicalAddGroup.toUniformSpace E
  haveI := comm_topologicalAddGroup_is_uniform (G := E)
  hf.cauchySeq.totallyBounded_range.isVonNBounded 𝕜


variable (𝕜) in
protected theorem Bornology.IsVonNBounded.restrict_scalars_of_nontrivial
    [NormedField 𝕜] [NormedRing 𝕜'] [NormedAlgebra 𝕜 𝕜'] [Nontrivial 𝕜']
    [Zero E] [TopologicalSpace E]
    [SMul 𝕜 E] [MulAction 𝕜' E] [IsScalarTower 𝕜 𝕜' E] {s : Set E}
    (h : IsVonNBounded 𝕜' s) : IsVonNBounded 𝕜 s := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : NormedRing 𝕜'
    inst✝⁶ : NormedAlgebra 𝕜 𝕜'
    inst✝⁵ : Nontrivial 𝕜'
    inst✝⁴ : Zero E
    inst✝³ : TopologicalSpace E
    inst✝² : SMul 𝕜 E
    inst✝¹ : MulAction 𝕜' E
    inst✝ : IsScalarTower 𝕜 𝕜' E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜' s
    ⊢ Bornology.IsVonNBounded 𝕜 s
  -/
  intro V hV
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : NormedRing 𝕜'
    inst✝⁶ : NormedAlgebra 𝕜 𝕜'
    inst✝⁵ : Nontrivial 𝕜'
    inst✝⁴ : Zero E
    inst✝³ : TopologicalSpace E
    inst✝² : SMul 𝕜 E
    inst✝¹ : MulAction 𝕜' E
    inst✝ : IsScalarTower 𝕜 𝕜' E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜' s
    V : Set E
    hV : Membership.mem (nhds 0) V
    ⊢ Absorbs 𝕜 V s
  -/
  refine (h hV).restrict_scalars <| AntilipschitzWith.tendsto_cobounded (K := ‖(1 : 𝕜')‖₊⁻¹) ?_
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : NormedRing 𝕜'
    inst✝⁶ : NormedAlgebra 𝕜 𝕜'
    inst✝⁵ : Nontrivial 𝕜'
    inst✝⁴ : Zero E
    inst✝³ : TopologicalSpace E
    inst✝² : SMul 𝕜 E
    inst✝¹ : MulAction 𝕜' E
    inst✝ : IsScalarTower 𝕜 𝕜' E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜' s
    V : Set E
    hV : Membership.mem (nhds 0) V
    ⊢ AntilipschitzWith (Inv.inv (NNNorm.nnnorm 1)) fun x => HSMul.hSMul x 1
  -/
  refine AntilipschitzWith.of_le_mul_nndist fun x y ↦ ?_
  rw [nndist_eq_nnnorm, nndist_eq_nnnorm, ← sub_smul, nnnorm_smul, ← div_eq_inv_mul,
    mul_div_cancel_right₀ _ (nnnorm_ne_zero_iff.2 one_ne_zero)]


variable (𝕜) in
protected theorem Bornology.IsVonNBounded.restrict_scalars
    [NormedField 𝕜] [NormedRing 𝕜'] [NormedAlgebra 𝕜 𝕜']
    [Zero E] [TopologicalSpace E]
    [SMul 𝕜 E] [MulActionWithZero 𝕜' E] [IsScalarTower 𝕜 𝕜' E] {s : Set E}
    (h : IsVonNBounded 𝕜' s) : IsVonNBounded 𝕜 s :=
  match subsingleton_or_nontrivial 𝕜' with
  | .inl _ =>
    have : Subsingleton E := MulActionWithZero.subsingleton 𝕜' E
    IsVonNBounded.of_subsingleton
  | .inr _ =>
    h.restrict_scalars_of_nontrivial _


theorem isVonNBounded_of_isBounded {s : Set E} (h : Bornology.IsBounded s) :
    Bornology.IsVonNBounded 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsBounded s
    ⊢ Bornology.IsVonNBounded 𝕜 s
  -/
  rcases h.subset_ball 0 with ⟨r, hr⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsBounded s
    r : Real
    hr : HasSubset.Subset s (Metric.ball 0 r)
    ⊢ Bornology.IsVonNBounded 𝕜 s
  -/
  rw [Metric.nhds_basis_ball.isVonNBounded_iff]
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsBounded s
    r : Real
    hr : HasSubset.Subset s (Metric.ball 0 r)
    ⊢ ∀ (i : Real), LT.lt 0 i → Absorbs 𝕜 (Metric.ball 0 i) s
  -/
  rw [← ball_normSeminorm 𝕜 E] at hr ⊢
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsBounded s
    r : Real
    hr : HasSubset.Subset s ((normSeminorm 𝕜 E).ball 0 r)
    ⊢ ∀ (i : Real), LT.lt 0 i → Absorbs 𝕜 ((normSeminorm 𝕜 E).ball 0 i) s
  -/
  exact fun ε hε ↦ ((normSeminorm 𝕜 E).ball_zero_absorbs_ball_zero hε).mono_right hr
  /-
    🎉 no goals
  -/


theorem isVonNBounded_ball (r : ℝ) : Bornology.IsVonNBounded 𝕜 (Metric.ball (0 : E) r) :=
  isVonNBounded_of_isBounded _ Metric.isBounded_ball


theorem isVonNBounded_closedBall (r : ℝ) :
    Bornology.IsVonNBounded 𝕜 (Metric.closedBall (0 : E) r) :=
  isVonNBounded_of_isBounded _ Metric.isBounded_closedBall


theorem isVonNBounded_iff {s : Set E} : Bornology.IsVonNBounded 𝕜 s ↔ Bornology.IsBounded s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 s) (Bornology.IsBounded s)
  -/
  refine ⟨fun h ↦ ?_, isVonNBounded_of_isBounded _⟩
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ⊢ Bornology.IsBounded s
  -/
  rcases (h (Metric.ball_mem_nhds 0 zero_lt_one)).exists_pos with ⟨ρ, hρ, hρball⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ρ : Real
    hρ : GT.gt ρ 0
    hρball : ∀ (c : 𝕜), LE.le ρ (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c  …
    ⊢ Bornology.IsBounded s
  -/
  rcases NormedField.exists_lt_norm 𝕜 ρ with ⟨a, ha⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ρ : Real
    hρ : GT.gt ρ 0
    hρball : ∀ (c : 𝕜), LE.le ρ (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c  …
    a : 𝕜
    ha : LT.lt ρ (Norm.norm a)
    ⊢ Bornology.IsBounded s
  -/
  specialize hρball a ha.le
  rw [← ball_normSeminorm 𝕜 E, Seminorm.smul_ball_zero (norm_pos_iff.1 <| hρ.trans ha),
    ball_normSeminorm] at hρball
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    h : Bornology.IsVonNBounded 𝕜 s
    ρ : Real
    hρ : GT.gt ρ 0
    a : 𝕜
    ha : LT.lt ρ (Norm.norm a)
    hρball : HasSubset.Subset s (Metric.ball 0 (HMul.hMul (Norm.norm a) 1))
    ⊢ Bornology.IsBounded s
  -/
  exact Metric.isBounded_ball.subset hρball
  /-
    🎉 no goals
  -/


theorem isVonNBounded_iff' {s : Set E} :
    Bornology.IsVonNBounded 𝕜 s ↔ ∃ r : ℝ, ∀ x ∈ s, ‖x‖ ≤ r := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 s) (Exists fun r => ∀ (x : E), Membership.mem …
  -/
  rw [NormedSpace.isVonNBounded_iff, isBounded_iff_forall_norm_le]
  /-
    🎉 no goals
  -/


theorem image_isVonNBounded_iff {α : Type*} {f : α → E} {s : Set α} :
    Bornology.IsVonNBounded 𝕜 (f '' s) ↔ ∃ r : ℝ, ∀ x ∈ s, ‖f x‖ ≤ r := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    α : Type u_6
    f : α → E
    s : Set α
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Set.image f s)) (Exists fun r => ∀ (x : α),  …
  -/
  simp_rw [isVonNBounded_iff', Set.forall_mem_image]
  /-
    🎉 no goals
  -/


/-- In a normed space, the von Neumann bornology (`Bornology.vonNBornology`) is equal to the
metric bornology. -/
theorem vonNBornology_eq : Bornology.vonNBornology 𝕜 E = PseudoMetricSpace.toBornology := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ Eq (Bornology.vonNBornology 𝕜 E) PseudoMetricSpace.toBornology
  -/
  rw [Bornology.ext_iff_isBounded]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ ∀ (s : Set E), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
  -/
  intro s
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
  -/
  rw [Bornology.isBounded_iff_isVonNBounded]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 s) (Bornology.IsBounded s)
  -/
  exact isVonNBounded_iff _
  /-
    🎉 no goals
  -/


theorem isBounded_iff_subset_smul_ball {s : Set E} :
    Bornology.IsBounded s ↔ ∃ a : 𝕜, s ⊆ a • Metric.ball (0 : E) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsBounded s) (Exists fun a => HasSubset.Subset s (HSMul.hSMul …
  -/
  rw [← isVonNBounded_iff 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 s) (Exists fun a => HasSubset.Subset s (HSMul …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      ⊢ Bornology.IsVonNBounded 𝕜 s → Exists fun a => HasSubset.Subset s (HSMul.hSMu …
    -/
  · intro h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      h : Bornology.IsVonNBounded 𝕜 s
      ⊢ Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.ball 0 1))
    -/
    rcases (h (Metric.ball_mem_nhds 0 zero_lt_one)).exists_pos with ⟨ρ, _, hρball⟩
    /-
      case mp.intro.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      h : Bornology.IsVonNBounded 𝕜 s
      ρ : Real
      left✝ : GT.gt ρ 0
      hρball : ∀ (c : 𝕜), LE.le ρ (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c  …
      ⊢ Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.ball 0 1))
    -/
    rcases NormedField.exists_lt_norm 𝕜 ρ with ⟨a, ha⟩
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      h : Bornology.IsVonNBounded 𝕜 s
      ρ : Real
      left✝ : GT.gt ρ 0
      hρball : ∀ (c : 𝕜), LE.le ρ (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c  …
      a : 𝕜
      ha : LT.lt ρ (Norm.norm a)
      ⊢ Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.ball 0 1))
    -/
    exact ⟨a, hρball a ha.le⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      ⊢ (Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.ball 0 1))) → Bor …
    -/
  · rintro ⟨a, ha⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      a : 𝕜
      ha : HasSubset.Subset s (HSMul.hSMul a (Metric.ball 0 1))
      ⊢ Bornology.IsVonNBounded 𝕜 s
    -/
    exact ((isVonNBounded_ball 𝕜 E 1).image (a • (1 : E →L[𝕜] E))).subset ha
    /-
      🎉 no goals
    -/


theorem isBounded_iff_subset_smul_closedBall {s : Set E} :
    Bornology.IsBounded s ↔ ∃ a : 𝕜, s ⊆ a • Metric.closedBall (0 : E) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (Bornology.IsBounded s) (Exists fun a => HasSubset.Subset s (HSMul.hSMul …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      ⊢ Bornology.IsBounded s → Exists fun a => HasSubset.Subset s (HSMul.hSMul a (M …
    -/
  · rw [isBounded_iff_subset_smul_ball 𝕜]
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      ⊢ (Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.ball 0 1))) → Exi …
    -/
    exact Exists.imp fun a ha => ha.trans <| Set.smul_set_mono <| Metric.ball_subset_closedBall
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      ⊢ (Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.closedBall 0 1))) …
    -/
  · rw [← isVonNBounded_iff 𝕜]
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      ⊢ (Exists fun a => HasSubset.Subset s (HSMul.hSMul a (Metric.closedBall 0 1))) …
    -/
    rintro ⟨a, ha⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      a : 𝕜
      ha : HasSubset.Subset s (HSMul.hSMul a (Metric.closedBall 0 1))
      ⊢ Bornology.IsVonNBounded 𝕜 s
    -/
    exact ((isVonNBounded_closedBall 𝕜 E 1).image (a • (1 : E →L[𝕜] E))).subset ha
    /-
      🎉 no goals
    -/


