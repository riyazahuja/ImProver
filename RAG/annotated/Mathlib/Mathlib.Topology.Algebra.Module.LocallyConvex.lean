/-- A `LocallyConvexSpace` is a topological semimodule over an ordered semiring in which convex
neighborhoods of a point form a neighborhood basis at that point. -/
class LocallyConvexSpace (𝕜 E : Type*) [OrderedSemiring 𝕜] [AddCommMonoid E] [Module 𝕜 E]
    [TopologicalSpace E] : Prop where
  convex_basis : ∀ x : E, (𝓝 x).HasBasis (fun s : Set E => s ∈ 𝓝 x ∧ Convex 𝕜 s) id


theorem locallyConvexSpace_iff :
    LocallyConvexSpace 𝕜 E ↔ ∀ x : E, (𝓝 x).HasBasis (fun s : Set E => s ∈ 𝓝 x ∧ Convex 𝕜 s) id :=
  ⟨@LocallyConvexSpace.convex_basis _ _ _ _ _ _, LocallyConvexSpace.mk⟩


theorem LocallyConvexSpace.ofBases {ι : Type*} (b : E → ι → Set E) (p : E → ι → Prop)
    (hbasis : ∀ x : E, (𝓝 x).HasBasis (p x) (b x)) (hconvex : ∀ x i, p x i → Convex 𝕜 (b x i)) :
    LocallyConvexSpace 𝕜 E :=
  ⟨fun x =>
    (hbasis x).to_hasBasis
      (fun i hi => ⟨b x i, ⟨⟨(hbasis x).mem_of_mem hi, hconvex x i hi⟩, le_refl (b x i)⟩⟩)
      fun s hs =>
      ⟨(hbasis x).index s hs.1, ⟨(hbasis x).property_index hs.1, (hbasis x).set_index_subset hs.1⟩⟩⟩


theorem LocallyConvexSpace.convex_basis_zero [LocallyConvexSpace 𝕜 E] :
    (𝓝 0 : Filter E).HasBasis (fun s => s ∈ (𝓝 0 : Filter E) ∧ Convex 𝕜 s) id :=
  LocallyConvexSpace.convex_basis 0


theorem locallyConvexSpace_iff_exists_convex_subset :
    LocallyConvexSpace 𝕜 E ↔ ∀ x : E, ∀ U ∈ 𝓝 x, ∃ S ∈ 𝓝 x, Convex 𝕜 S ∧ S ⊆ U :=
  (locallyConvexSpace_iff 𝕜 E).trans (forall_congr' fun _ => hasBasis_self)


theorem LocallyConvexSpace.ofBasisZero {ι : Type*} (b : ι → Set E) (p : ι → Prop)
    (hbasis : (𝓝 0).HasBasis p b) (hconvex : ∀ i, p i → Convex 𝕜 (b i)) :
    LocallyConvexSpace 𝕜 E := by
  refine LocallyConvexSpace.ofBases 𝕜 E (fun (x : E) (i : ι) => (x + ·) '' b i) (fun _ => p)
    (fun x => ?_) fun x i hi => (hconvex i hi).translate x
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    ι : Type u_3
    b : ι → Set E
    p : ι → Prop
    hbasis : (nhds 0).HasBasis p b
    hconvex : ∀ (i : ι), p i → Convex 𝕜 (b i)
    x : E
    ⊢ (nhds x).HasBasis ((fun x => p) x) ((fun x i => Set.image (fun x_1 => HAdd.h …
  -/
  rw [← map_add_left_nhds_zero]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    ι : Type u_3
    b : ι → Set E
    p : ι → Prop
    hbasis : (nhds 0).HasBasis p b
    hconvex : ∀ (i : ι), p i → Convex 𝕜 (b i)
    x : E
    ⊢ (Filter.map (fun x_1 => HAdd.hAdd x x_1) (nhds 0)).HasBasis ((fun x => p) x) …
  -/
  exact hbasis.map _
  /-
    🎉 no goals
  -/


theorem locallyConvexSpace_iff_zero : LocallyConvexSpace 𝕜 E ↔
    (𝓝 0 : Filter E).HasBasis (fun s : Set E => s ∈ (𝓝 0 : Filter E) ∧ Convex 𝕜 s) id :=
  ⟨fun h => @LocallyConvexSpace.convex_basis _ _ _ _ _ _ h 0, fun h =>
    LocallyConvexSpace.ofBasisZero 𝕜 E _ _ h fun _ => And.right⟩


theorem locallyConvexSpace_iff_exists_convex_subset_zero :
    LocallyConvexSpace 𝕜 E ↔ ∀ U ∈ (𝓝 0 : Filter E), ∃ S ∈ (𝓝 0 : Filter E), Convex 𝕜 S ∧ S ⊆ U :=
  (locallyConvexSpace_iff_zero 𝕜 E).trans hasBasis_self

-- see Note [lower instance priority]

instance (priority := 100) LocallyConvexSpace.toLocPathConnectedSpace [Module ℝ E]
    [ContinuousSMul ℝ E] [LocallyConvexSpace ℝ E] : LocPathConnectedSpace E :=
  .of_bases (fun x ↦ convex_basis (𝕜 := ℝ) x)
    fun _ _ hs ↦ hs.2.isPathConnected <| nonempty_of_mem <| mem_of_mem_nhds hs.1


/-- Convex subsets of locally convex spaces are locally path-connected. -/
theorem Convex.locPathConnectedSpace [Module ℝ E] [ContinuousSMul ℝ E] [LocallyConvexSpace ℝ E]
    {S : Set E} (hS : Convex ℝ S) : LocPathConnectedSpace S := by
  /-
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    S : Set E
    hS : Convex Real S
    ⊢ LocPathConnectedSpace ↑S
  -/
  refine ⟨fun x ↦ ⟨fun s ↦ ⟨fun hs ↦ ?_, fun ⟨t, ht⟩ ↦ mem_of_superset ht.1.1 ht.2⟩⟩⟩
  /-
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    S : Set E
    hS : Convex Real S
    x : ↑S
    s : Set ↑S
    hs : Membership.mem (nhds x) s
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  let ⟨t, ht⟩ := (mem_nhds_subtype S x s).mp hs
  /-
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    S : Set E
    hS : Convex Real S
    x : ↑S
    s : Set ↑S
    hs : Membership.mem (nhds x) s
    t : Set E
    ht : And (Membership.mem (nhds ↑x) t) (HasSubset.Subset (Set.preimage Subtype. …
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  let ⟨t', ht'⟩ := (LocallyConvexSpace.convex_basis (𝕜 := ℝ) x.1).mem_iff.mp ht.1
  /-
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    S : Set E
    hS : Convex Real S
    x : ↑S
    s : Set ↑S
    hs : Membership.mem (nhds x) s
    t : Set E
    ht : And (Membership.mem (nhds ↑x) t) (HasSubset.Subset (Set.preimage Subtype. …
    t' : Set E
    ht' : And (And (Membership.mem (nhds ↑x) t') (Convex Real t')) (HasSubset.Subs …
    ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
  -/
  refine ⟨(↑) ⁻¹' t', ⟨?_, ?_⟩, (preimage_mono ht'.2).trans ht.2⟩
    /-
      case refine_1
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : Module Real E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      S : Set E
      hS : Convex Real S
      x : ↑S
      s : Set ↑S
      hs : Membership.mem (nhds x) s
      t : Set E
      ht : And (Membership.mem (nhds ↑x) t) (HasSubset.Subset (Set.preimage Subtype. …
      t' : Set E
      ht' : And (And (Membership.mem (nhds ↑x) t') (Convex Real t')) (HasSubset.Subs …
      ⊢ Membership.mem (nhds x) (Set.preimage Subtype.val t')
    -/
  · exact continuousAt_subtype_val.preimage_mem_nhds ht'.1.1
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : Module Real E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      S : Set E
      hS : Convex Real S
      x : ↑S
      s : Set ↑S
      hs : Membership.mem (nhds x) s
      t : Set E
      ht : And (Membership.mem (nhds ↑x) t) (HasSubset.Subset (Set.preimage Subtype. …
      t' : Set E
      ht' : And (And (Membership.mem (nhds ↑x) t') (Convex Real t')) (HasSubset.Subs …
      ⊢ IsPathConnected (Set.preimage Subtype.val t')
    -/
  · refine Subtype.preimage_coe_self_inter _ _ ▸ IsPathConnected.preimage_coe ?_ inter_subset_left
    /-
      case refine_2
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : Module Real E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      S : Set E
      hS : Convex Real S
      x : ↑S
      s : Set ↑S
      hs : Membership.mem (nhds x) s
      t : Set E
      ht : And (Membership.mem (nhds ↑x) t) (HasSubset.Subset (Set.preimage Subtype. …
      t' : Set E
      ht' : And (And (Membership.mem (nhds ↑x) t') (Convex Real t')) (HasSubset.Subs …
      ⊢ IsPathConnected (Inter.inter S t')
    -/
    exact (hS.inter ht'.1.2).isPathConnected ⟨x, x.2, mem_of_mem_nhds ht'.1.1⟩
    /-
      🎉 no goals
    -/


theorem LocallyConvexSpace.convex_open_basis_zero [LocallyConvexSpace 𝕜 E] :
    (𝓝 0 : Filter E).HasBasis (fun s => (0 : E) ∈ s ∧ IsOpen s ∧ Convex 𝕜 s) id :=
  (LocallyConvexSpace.convex_basis_zero 𝕜 E).to_hasBasis
    (fun s hs =>
      ⟨interior s, ⟨mem_interior_iff_mem_nhds.mpr hs.1, isOpen_interior, hs.2.interior⟩,
        interior_subset⟩)
    fun s hs => ⟨s, ⟨hs.2.1.mem_nhds hs.1, hs.2.2⟩, subset_rfl⟩


/-- In a locally convex space, if `s`, `t` are disjoint convex sets, `s` is compact and `t` is
closed, then we can find open disjoint convex sets containing them. -/
theorem Disjoint.exists_open_convexes [LocallyConvexSpace 𝕜 E] {s t : Set E} (disj : Disjoint s t)
    (hs₁ : Convex 𝕜 s) (hs₂ : IsCompact s) (ht₁ : Convex 𝕜 t) (ht₂ : IsClosed t) :
    ∃ u v, IsOpen u ∧ IsOpen v ∧ Convex 𝕜 u ∧ Convex 𝕜 v ∧ s ⊆ u ∧ t ⊆ v ∧ Disjoint u v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Convex  …
  -/
  letI : UniformSpace E := TopologicalAddGroup.toUniformSpace E
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Convex  …
  -/
  haveI : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this : UniformAddGroup E
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Convex  …
  -/
  have := (LocallyConvexSpace.convex_open_basis_zero 𝕜 E).comap fun x : E × E => x.2 - x.1
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝ : UniformAddGroup E
    this : (Filter.comap (fun x => HSub.hSub x.2 x.1) (nhds 0)).HasBasis (fun s => …
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Convex  …
  -/
  rw [← uniformity_eq_comap_nhds_zero] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝ : UniformAddGroup E
    this : (uniformity E).HasBasis (fun s => And (Membership.mem s 0) (And (IsOpen …
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Convex  …
  -/
  rcases disj.exists_uniform_thickening_of_basis this hs₂ ht₂ with ⟨V, ⟨hV0, hVopen, hVconvex⟩, hV⟩
  refine ⟨s + V, t + V, hVopen.add_left, hVopen.add_left, hs₁.add hVconvex, ht₁.add hVconvex,
    subset_add_left _ hV0, subset_add_left _ hV0, ?_⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝ : UniformAddGroup E
    this : (uniformity E).HasBasis (fun s => And (Membership.mem s 0) (And (IsOpen …
    V : Set E
    hV : Disjoint (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x (Se …
    hV0 : Membership.mem V 0
    hVopen : IsOpen V
    hVconvex : Convex 𝕜 V
    ⊢ Disjoint (HAdd.hAdd s V) (HAdd.hAdd t V)
  -/
  simp_rw [← iUnion_add_left_image, image_add_left]
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝ : UniformAddGroup E
    this : (uniformity E).HasBasis (fun s => And (Membership.mem s 0) (And (IsOpen …
    V : Set E
    hV : Disjoint (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x (Se …
    hV0 : Membership.mem V 0
    hVopen : IsOpen V
    hVconvex : Convex 𝕜 V
    ⊢ Disjoint (Set.iUnion fun a => Set.iUnion fun x => Set.preimage (fun x => HAd …
  -/
  simp_rw [UniformSpace.ball, ← preimage_comp, sub_eq_neg_add] at hV
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousConstSMul 𝕜 E
    inst✝ : LocallyConvexSpace 𝕜 E
    s t : Set E
    disj : Disjoint s t
    hs₁ : Convex 𝕜 s
    hs₂ : IsCompact s
    ht₁ : Convex 𝕜 t
    ht₂ : IsClosed t
    this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝ : UniformAddGroup E
    this : (uniformity E).HasBasis (fun s => And (Membership.mem s 0) (And (IsOpen …
    V : Set E
    hV0 : Membership.mem V 0
    hVopen : IsOpen V
    hVconvex : Convex 𝕜 V
    hV : Disjoint (Set.iUnion fun x => Set.iUnion fun x_1 => Set.preimage (Functio …
    ⊢ Disjoint (Set.iUnion fun a => Set.iUnion fun x => Set.preimage (fun x => HAd …
  -/
  exact hV
  /-
    🎉 no goals
  -/


theorem locallyConvexSpace_sInf {ts : Set (TopologicalSpace E)}
    (h : ∀ t ∈ ts, @LocallyConvexSpace 𝕜 E _ _ _ t) : @LocallyConvexSpace 𝕜 E _ _ _ (sInf ts) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    ts : Set (TopologicalSpace E)
    h : ∀ (t : TopologicalSpace E), Membership.mem ts t → LocallyConvexSpace 𝕜 E
    ⊢ LocallyConvexSpace 𝕜 E
  -/
  letI : TopologicalSpace E := sInf ts
  refine
    LocallyConvexSpace.ofBases 𝕜 E (fun _ => fun If : Set ts × (ts → Set E) => ⋂ i ∈ If.1, If.2 i)
      (fun x => fun If : Set ts × (ts → Set E) =>
        If.1.Finite ∧ ∀ i ∈ If.1, If.2 i ∈ @nhds _ (↑i) x ∧ Convex 𝕜 (If.2 i))
      (fun x => ?_) fun x If hif => convex_iInter fun i => convex_iInter fun hi => (hif.2 i hi).2
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    ts : Set (TopologicalSpace E)
    h : ∀ (t : TopologicalSpace E), Membership.mem ts t → LocallyConvexSpace 𝕜 E
    this : TopologicalSpace E := InfSet.sInf ts
    x : E
    ⊢ (nhds x).HasBasis ((fun x If => And If.1.Finite (∀ (i : ↑ts), Membership.mem …
  -/
  rw [nhds_sInf, ← iInf_subtype'']
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    ts : Set (TopologicalSpace E)
    h : ∀ (t : TopologicalSpace E), Membership.mem ts t → LocallyConvexSpace 𝕜 E
    this : TopologicalSpace E := InfSet.sInf ts
    x : E
    ⊢ (iInf fun i => nhds x).HasBasis ((fun x If => And If.1.Finite (∀ (i : ↑ts),  …
  -/
  exact hasBasis_iInf' fun i : ts => (@locallyConvexSpace_iff 𝕜 E _ _ _ ↑i).mp (h (↑i) i.2) x
  /-
    🎉 no goals
  -/


theorem locallyConvexSpace_iInf {ts' : ι → TopologicalSpace E}
    (h' : ∀ i, @LocallyConvexSpace 𝕜 E _ _ _ (ts' i)) :
    @LocallyConvexSpace 𝕜 E _ _ _ (⨅ i, ts' i) := by
  /-
    ι : Sort u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    ts' : ι → TopologicalSpace E
    h' : ∀ (i : ι), LocallyConvexSpace 𝕜 E
    ⊢ LocallyConvexSpace 𝕜 E
  -/
  refine locallyConvexSpace_sInf ?_
  /-
    ι : Sort u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    ts' : ι → TopologicalSpace E
    h' : ∀ (i : ι), LocallyConvexSpace 𝕜 E
    ⊢ ∀ (t : TopologicalSpace E), Membership.mem (Set.range fun i => ts' i) t → Lo …
  -/
  rwa [forall_mem_range]
  /-
    🎉 no goals
  -/


theorem locallyConvexSpace_inf {t₁ t₂ : TopologicalSpace E} (h₁ : @LocallyConvexSpace 𝕜 E _ _ _ t₁)
    (h₂ : @LocallyConvexSpace 𝕜 E _ _ _ t₂) : @LocallyConvexSpace 𝕜 E _ _ _ (t₁ ⊓ t₂) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    t₁ t₂ : TopologicalSpace E
    h₁ : LocallyConvexSpace 𝕜 E
    h₂ : LocallyConvexSpace 𝕜 E
    ⊢ LocallyConvexSpace 𝕜 E
  -/
  rw [inf_eq_iInf]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    t₁ t₂ : TopologicalSpace E
    h₁ : LocallyConvexSpace 𝕜 E
    h₂ : LocallyConvexSpace 𝕜 E
    ⊢ LocallyConvexSpace 𝕜 E
  -/
  refine locallyConvexSpace_iInf fun b => ?_
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    t₁ t₂ : TopologicalSpace E
    h₁ : LocallyConvexSpace 𝕜 E
    h₂ : LocallyConvexSpace 𝕜 E
    b : Bool
    ⊢ LocallyConvexSpace 𝕜 E
  -/
              /-
                🎉 no goals
              -/
  cases b <;> assumption
              /-
                🎉 no goals
              -/


theorem locallyConvexSpace_induced {t : TopologicalSpace F} [LocallyConvexSpace 𝕜 F]
    (f : E →ₗ[𝕜] F) : @LocallyConvexSpace 𝕜 E _ _ _ (t.induced f) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    t : TopologicalSpace F
    inst✝ : LocallyConvexSpace 𝕜 F
    f : LinearMap (RingHom.id 𝕜) E F
    ⊢ LocallyConvexSpace 𝕜 E
  -/
  letI : TopologicalSpace E := t.induced f
  refine LocallyConvexSpace.ofBases 𝕜 E (fun _ => preimage f)
    (fun x => fun s : Set F => s ∈ 𝓝 (f x) ∧ Convex 𝕜 s) (fun x => ?_) fun x s ⟨_, hs⟩ =>
    hs.linear_preimage f
  /-
    𝕜 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    t : TopologicalSpace F
    inst✝ : LocallyConvexSpace 𝕜 F
    f : LinearMap (RingHom.id 𝕜) E F
    this : TopologicalSpace E := TopologicalSpace.induced (⇑f) t
    x : E
    ⊢ (nhds x).HasBasis ((fun x s => And (Membership.mem (nhds (f x)) s) (Convex 𝕜 …
  -/
  rw [nhds_induced]
  /-
    𝕜 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    t : TopologicalSpace F
    inst✝ : LocallyConvexSpace 𝕜 F
    f : LinearMap (RingHom.id 𝕜) E F
    this : TopologicalSpace E := TopologicalSpace.induced (⇑f) t
    x : E
    ⊢ (Filter.comap (⇑f) (nhds (f x))).HasBasis ((fun x s => And (Membership.mem ( …
  -/
  exact (LocallyConvexSpace.convex_basis <| f x).comap f
  /-
    🎉 no goals
  -/


instance Pi.locallyConvexSpace {ι : Type*} {X : ι → Type*} [∀ i, AddCommMonoid (X i)]
    [∀ i, TopologicalSpace (X i)] [∀ i, Module 𝕜 (X i)] [∀ i, LocallyConvexSpace 𝕜 (X i)] :
    LocallyConvexSpace 𝕜 (∀ i, X i) :=
  locallyConvexSpace_iInf fun i => locallyConvexSpace_induced (LinearMap.proj i)


instance Prod.locallyConvexSpace [TopologicalSpace E] [TopologicalSpace F] [LocallyConvexSpace 𝕜 E]
    [LocallyConvexSpace 𝕜 F] : LocallyConvexSpace 𝕜 (E × F) :=
-- Porting note: had to specify `t₁` and `t₂`
  locallyConvexSpace_inf (t₁ := induced Prod.fst _) (t₂ := induced Prod.snd _)
    (locallyConvexSpace_induced (LinearMap.fst _ _ _))
    (locallyConvexSpace_induced (LinearMap.snd _ _ _))


