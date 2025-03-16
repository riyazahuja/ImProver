lemma tendsto_cofinite_cocompact_iff :
    Tendsto f cofinite (cocompact _) ↔ ∀ K, IsCompact K → Set.Finite (f ⁻¹' K) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (Filter.Tendsto f Filter.cofinite (Filter.cocompact Y)) (∀ (K : Set Y),  …
  -/
  rw [hasBasis_cocompact.tendsto_right_iff]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (∀ (i : Set Y), IsCompact i → Filter.Eventually (fun x => Membership.mem …
  -/
  refine forall₂_congr (fun K _ ↦ ?_)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace Y
    f : X → Y
    K : Set Y
    x✝ : IsCompact K
    ⊢ Iff (Filter.Eventually (fun x => Membership.mem (HasCompl.compl K) (f x)) Fi …
  -/
  simp only [mem_compl_iff, eventually_cofinite, not_not, preimage]
  /-
    🎉 no goals
  -/


lemma Continuous.discrete_of_tendsto_cofinite_cocompact [T1Space X] [WeaklyLocallyCompactSpace Y]
    (hf' : Continuous f) (hf : Tendsto f cofinite (cocompact _)) :
    DiscreteTopology X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace Y
    f : X → Y
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : WeaklyLocallyCompactSpace Y
    hf' : Continuous f
    hf : Filter.Tendsto f Filter.cofinite (Filter.cocompact Y)
    ⊢ DiscreteTopology X
  -/
  refine singletons_open_iff_discrete.mp (fun x ↦ ?_)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace Y
    f : X → Y
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : WeaklyLocallyCompactSpace Y
    hf' : Continuous f
    hf : Filter.Tendsto f Filter.cofinite (Filter.cocompact Y)
    x : X
    ⊢ IsOpen (Singleton.singleton x)
  -/
  obtain ⟨K : Set Y, hK : IsCompact K, hK' : K ∈ 𝓝 (f x)⟩ := exists_compact_mem_nhds (f x)
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace Y
    f : X → Y
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : WeaklyLocallyCompactSpace Y
    hf' : Continuous f
    hf : Filter.Tendsto f Filter.cofinite (Filter.cocompact Y)
    x : X
    K : Set Y
    hK : IsCompact K
    hK' : Membership.mem (nhds (f x)) K
    ⊢ IsOpen (Singleton.singleton x)
  -/
  obtain ⟨U : Set Y, hU₁ : U ⊆ K, hU₂ : IsOpen U, hU₃ : f x ∈ U⟩ := mem_nhds_iff.mp hK'
  have hU₄ : Set.Finite (f⁻¹' U) :=
    Finite.subset (tendsto_cofinite_cocompact_iff.mp hf K hK) (preimage_mono hU₁)
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace Y
    f : X → Y
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : WeaklyLocallyCompactSpace Y
    hf' : Continuous f
    hf : Filter.Tendsto f Filter.cofinite (Filter.cocompact Y)
    x : X
    K : Set Y
    hK : IsCompact K
    hK' : Membership.mem (nhds (f x)) K
    U : Set Y
    hU₁ : HasSubset.Subset U K
    hU₂ : IsOpen U
    hU₃ : Membership.mem U (f x)
    hU₄ : (Set.preimage f U).Finite
    ⊢ IsOpen (Singleton.singleton x)
  -/
  exact isOpen_singleton_of_finite_mem_nhds _ ((hU₂.preimage hf').mem_nhds hU₃) hU₄
  /-
    🎉 no goals
  -/


lemma tendsto_cofinite_cocompact_of_discrete [DiscreteTopology X]
    (hf : Tendsto f (cocompact _) (cocompact _)) :
    Tendsto f cofinite (cocompact _) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace Y
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    hf : Filter.Tendsto f (Filter.cocompact X) (Filter.cocompact Y)
    ⊢ Filter.Tendsto f Filter.cofinite (Filter.cocompact Y)
  -/
  convert hf
  /-
    case h.e'_4
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace Y
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    hf : Filter.Tendsto f (Filter.cocompact X) (Filter.cocompact Y)
    ⊢ Eq Filter.cofinite (Filter.cocompact X)
  -/
  rw [cocompact_eq_cofinite X]
  /-
    🎉 no goals
  -/


lemma IsClosed.tendsto_coe_cofinite_of_discreteTopology
    {s : Set X} (hs : IsClosed s) (_hs' : DiscreteTopology s) :
    Tendsto ((↑) : s → X) cofinite (cocompact _) :=
  tendsto_cofinite_cocompact_of_discrete hs.isClosedEmbedding_subtypeVal.tendsto_cocompact


lemma IsClosed.tendsto_coe_cofinite_iff [T1Space X] [WeaklyLocallyCompactSpace X]
    {s : Set X} (hs : IsClosed s) :
    Tendsto ((↑) : s → X) cofinite (cocompact _) ↔ DiscreteTopology s :=
  ⟨continuous_subtype_val.discrete_of_tendsto_cofinite_cocompact,
   fun _ ↦ hs.tendsto_coe_cofinite_of_discreteTopology inferInstance⟩


/-- Criterion for a subset `S ⊆ X` to be closed and discrete in terms of the punctured
neighbourhood filter at an arbitrary point of `X`. (Compare `discreteTopology_subtype_iff`.) -/
theorem isClosed_and_discrete_iff {S : Set X} :
    IsClosed S ∧ DiscreteTopology S ↔ ∀ x, Disjoint (𝓝[≠] x) (𝓟 S) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    ⊢ Iff (And (IsClosed S) (DiscreteTopology ↑S)) (∀ (x : X), Disjoint (nhdsWithi …
  -/
  rw [discreteTopology_subtype_iff, isClosed_iff_clusterPt, ← forall_and]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    ⊢ Iff (∀ (x : X), And (ClusterPt x (Filter.principal S) → Membership.mem S x)  …
  -/
  congrm (∀ x, ?_)
  /-
    case a
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    x : X
    ⊢ Iff (And (ClusterPt x (Filter.principal S) → Membership.mem S x) (Membership …
  -/
  rw [← not_imp_not, clusterPt_iff_not_disjoint, not_not, ← disjoint_iff]
  /-
    case a
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    x : X
    ⊢ Iff (And (Not (Membership.mem S x) → Disjoint (nhds x) (Filter.principal S)) …
  -/
  constructor <;> intro H
    /-
      case a.mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      x : X
      H : And (Not (Membership.mem S x) → Disjoint (nhds x) (Filter.principal S)) (M …
      ⊢ Disjoint (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.pri …
    -/
  · by_cases hx : x ∈ S
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      x : X
      H : And (Not (Membership.mem S x) → Disjoint (nhds x) (Filter.principal S)) (M …
      hx : Membership.mem S x
      ⊢ Disjoint (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.pri …
    -/
    exacts [H.2 hx, (H.1 hx).mono_left nhdsWithin_le_nhds]
    /-
      🎉 no goals
    -/
    /-
      case a.mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      x : X
      H : Disjoint (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.p …
      ⊢ And (Not (Membership.mem S x) → Disjoint (nhds x) (Filter.principal S)) (Mem …
    -/
  · refine ⟨fun hx ↦ ?_, fun _ ↦ H⟩
    /-
      case a.mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      x : X
      H : Disjoint (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.p …
      hx : Not (Membership.mem S x)
      ⊢ Disjoint (nhds x) (Filter.principal S)
    -/
    simpa [disjoint_iff, nhdsWithin, inf_assoc, hx] using H
    /-
      🎉 no goals
    -/


/-- The filter of sets with no accumulation points inside a set `S : Set X`, implemented
as the supremum over all punctured neighborhoods within `S`. -/
def Filter.codiscreteWithin (S : Set X) : Filter X := ⨆ x ∈ S, 𝓝[S \ {x}] x


lemma mem_codiscreteWithin {S T : Set X} :
    S ∈ codiscreteWithin T ↔ ∀ x ∈ T, Disjoint (𝓝[≠] x) (𝓟 (T \ S)) := by
  simp only [codiscreteWithin, mem_iSup, mem_nhdsWithin, disjoint_principal_right, subset_def,
    mem_diff, mem_inter_iff, mem_compl_iff]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S T : Set X
    ⊢ Iff (∀ (i : X), Membership.mem T i → Exists fun u => And (IsOpen u) (And (Me …
  -/
  congr! 7 with x - u y
  /-
    case a.h.h'.h.e'_2.h.h.e'_2.h.e'_2.h.a
    X : Type u_1
    inst✝ : TopologicalSpace X
    S T : Set X
    x : X
    u : Set X
    y : X
    ⊢ Iff (And (Membership.mem u y) (And (Membership.mem T y) (Not (Membership.mem …
  -/
  tauto
  /-
    🎉 no goals
  -/


lemma mem_codiscreteWithin_accPt {S T : Set X} :
    S ∈ codiscreteWithin T ↔ ∀ x ∈ T, ¬AccPt x (𝓟 (T \ S)) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S T : Set X
    ⊢ Iff (Membership.mem (Filter.codiscreteWithin T) S) (∀ (x : X), Membership.me …
  -/
  simp only [mem_codiscreteWithin, disjoint_iff, AccPt, not_neBot]
  /-
    🎉 no goals
  -/


/-- In any topological space, the open sets with discrete complement form a filter,
defined as the supremum of all punctured neighborhoods.

See `Filter.mem_codiscrete'` for the equivalence. -/
def Filter.codiscrete (X : Type*) [TopologicalSpace X] : Filter X := codiscreteWithin Set.univ


lemma mem_codiscrete {S : Set X} :
    S ∈ codiscrete X ↔ ∀ x, Disjoint (𝓝[≠] x) (𝓟 Sᶜ) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    ⊢ Iff (Membership.mem (Filter.codiscrete X) S) (∀ (x : X), Disjoint (nhdsWithi …
  -/
  simp [codiscrete, mem_codiscreteWithin, compl_eq_univ_diff]
  /-
    🎉 no goals
  -/


lemma mem_codiscrete_accPt {S : Set X} :
    S ∈ codiscrete X ↔ ∀ x, ¬AccPt x (𝓟 Sᶜ) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    ⊢ Iff (Membership.mem (Filter.codiscrete X) S) (∀ (x : X), Not (AccPt x (Filte …
  -/
  simp only [mem_codiscrete, disjoint_iff, AccPt, not_neBot]
  /-
    🎉 no goals
  -/


lemma mem_codiscrete' {S : Set X} :
    S ∈ codiscrete X ↔ IsOpen S ∧ DiscreteTopology ↑Sᶜ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    ⊢ Iff (Membership.mem (Filter.codiscrete X) S) (And (IsOpen S) (DiscreteTopolo …
  -/
  rw [mem_codiscrete, ← isClosed_compl_iff, isClosed_and_discrete_iff]
  /-
    🎉 no goals
  -/


lemma mem_codiscrete_subtype_iff_mem_codiscreteWithin {S : Set X} {U : Set S} :
    U ∈ codiscrete S ↔ (↑) '' U ∈ codiscreteWithin S := by
  simp [mem_codiscrete, disjoint_principal_right, compl_compl, Subtype.forall,
    mem_codiscreteWithin]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    U : Set ↑S
    ⊢ Iff (∀ (a : X) (b : Membership.mem S a), Membership.mem (nhdsWithin ⟨a, b⟩ ( …
  -/
  congr! with x hx
  /-
    case a.h.h.a
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set X
    U : Set ↑S
    x : X
    hx : Membership.mem S x
    ⊢ Iff (Membership.mem (nhdsWithin ⟨x, hx⟩ (HasCompl.compl (Singleton.singleton …
  -/
  constructor
    /-
      case a.h.h.a.mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      ⊢ Membership.mem (nhdsWithin ⟨x, hx⟩ (HasCompl.compl (Singleton.singleton ⟨x,  …
    -/
  · rw [nhdsWithin_subtype, mem_comap]
    /-
      case a.h.h.a.mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      ⊢ (Exists fun t => And (Membership.mem (nhdsWithin (↑⟨x, hx⟩) (Set.image Subty …
    -/
    rintro ⟨t, ht1, ht2⟩
    /-
      case a.h.h.a.mp.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      t : Set X
      ht1 : Membership.mem (nhdsWithin (↑⟨x, hx⟩) (Set.image Subtype.val (HasCompl.c …
      ht2 : HasSubset.Subset (Set.preimage Subtype.val t) U
      ⊢ Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (HasC …
    -/
    rw [mem_nhdsWithin] at ht1 ⊢
    /-
      case a.h.h.a.mp.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      t : Set X
      ht1 : Exists fun u => And (IsOpen u) (And (Membership.mem u ↑⟨x, hx⟩) (HasSubs …
      ht2 : HasSubset.Subset (Set.preimage Subtype.val t) U
      ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (HasSubset.Subset ( …
    -/
    obtain ⟨u, hu1, hu2, hu3⟩ := ht1
    /-
      case a.h.h.a.mp.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      t : Set X
      ht2 : HasSubset.Subset (Set.preimage Subtype.val t) U
      u : Set X
      hu1 : IsOpen u
      hu2 : Membership.mem u ↑⟨x, hx⟩
      hu3 : HasSubset.Subset (Inter.inter u (Set.image Subtype.val (HasCompl.compl ( …
      ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (HasSubset.Subset ( …
    -/
    refine ⟨u, hu1, hu2, fun v hv ↦ ?_⟩
    /-
      case a.h.h.a.mp.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      t : Set X
      ht2 : HasSubset.Subset (Set.preimage Subtype.val t) U
      u : Set X
      hu1 : IsOpen u
      hu2 : Membership.mem u ↑⟨x, hx⟩
      hu3 : HasSubset.Subset (Inter.inter u (Set.image Subtype.val (HasCompl.compl ( …
      v : X
      hv : Membership.mem (Inter.inter u (HasCompl.compl (Singleton.singleton x))) v
      ⊢ Membership.mem (HasCompl.compl (SDiff.sdiff S (Set.image Subtype.val U))) v
    -/
    simpa using fun hv2 ↦ ⟨hv2, ht2 <| hu3 <| by simpa [hv2]⟩
    /-
      🎉 no goals
    -/
    /-
      case a.h.h.a.mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      S : Set X
      U : Set ↑S
      x : X
      hx : Membership.mem S x
      ⊢ Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (HasC …
    -/
  · suffices Tendsto (↑) (𝓝[≠] (⟨x, hx⟩ : S)) (𝓝[≠] x) by convert tendsto_def.mp this _; ext; simp
    exact tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _
      continuous_subtype_val.continuousWithinAt <| eventually_mem_nhdsWithin.mono (by simp)


