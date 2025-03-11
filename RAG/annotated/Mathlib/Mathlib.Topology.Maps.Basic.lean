protected lemma IsInducing.induced (f : X → Y) : @IsInducing X Y (induced f ‹_›) _ f :=
  @IsInducing.mk _ _ (TopologicalSpace.induced f ‹_›) _ _ rfl


@[deprecated (since := "2024-10-28")] alias inducing_induced := IsInducing.induced


protected lemma IsInducing.id : IsInducing (@id X) := ⟨induced_id.symm⟩


@[deprecated (since := "2024-10-28")] alias inducing_id := IsInducing.id


protected lemma IsInducing.comp (hg : IsInducing g) (hf : IsInducing f) :
    IsInducing (g ∘ f) :=
      /-
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        f : X → Y
        g : Y → Z
        inst✝² : TopologicalSpace Y
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Z
        hg : Topology.IsInducing g
        hf : Topology.IsInducing f
        ⊢ Eq inst✝¹ (TopologicalSpace.induced (Function.comp g f) inst✝)
      -/
  ⟨by rw [hf.eq_induced, hg.eq_induced, induced_compose]⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-28")] alias Inducing.comp := IsInducing.comp


lemma IsInducing.of_comp_iff (hg : IsInducing g) : IsInducing (g ∘ f) ↔ IsInducing f := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Z
    hg : Topology.IsInducing g
    ⊢ Iff (Topology.IsInducing (Function.comp g f)) (Topology.IsInducing f)
  -/
  refine ⟨fun h ↦ ?_, hg.comp⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Z
    hg : Topology.IsInducing g
    h : Topology.IsInducing (Function.comp g f)
    ⊢ Topology.IsInducing f
  -/
  rw [isInducing_iff, hg.eq_induced, induced_compose, h.eq_induced]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias Inducing.of_comp_iff := IsInducing.of_comp_iff


lemma IsInducing.of_comp (hf : Continuous f) (hg : Continuous g) (hgf : IsInducing (g ∘ f)) :
    IsInducing f :=
                   /-
                     X : Type u_1
                     Y : Type u_2
                     Z : Type u_3
                     f : X → Y
                     g : Y → Z
                     inst✝² : TopologicalSpace Y
                     inst✝¹ : TopologicalSpace X
                     inst✝ : TopologicalSpace Z
                     hf : Continuous f
                     hg : Continuous g
                     hgf : Topology.IsInducing (Function.comp g f)
                     ⊢ LE.le inst✝¹ (TopologicalSpace.induced f inst✝²)
                   -/
  ⟨le_antisymm (by rwa [← continuous_iff_le_induced])
                   /-
                     🎉 no goals
                   -/
      (by
        /-
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          f : X → Y
          g : Y → Z
          inst✝² : TopologicalSpace Y
          inst✝¹ : TopologicalSpace X
          inst✝ : TopologicalSpace Z
          hf : Continuous f
          hg : Continuous g
          hgf : Topology.IsInducing (Function.comp g f)
          ⊢ LE.le (TopologicalSpace.induced f inst✝²) inst✝¹
        -/
        rw [hgf.eq_induced, ← induced_compose]
        /-
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          f : X → Y
          g : Y → Z
          inst✝² : TopologicalSpace Y
          inst✝¹ : TopologicalSpace X
          inst✝ : TopologicalSpace Z
          hf : Continuous f
          hg : Continuous g
          hgf : Topology.IsInducing (Function.comp g f)
          ⊢ LE.le (TopologicalSpace.induced f inst✝²) (TopologicalSpace.induced f (Topol …
        -/
        exact induced_mono hg.le_induced)⟩
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-10-28")] alias inducing_of_inducing_compose := IsInducing.of_comp


lemma isInducing_iff_nhds : IsInducing f ↔ ∀ x, 𝓝 x = comap f (𝓝 (f x)) :=
  (isInducing_iff _).trans (induced_iff_nhds_eq f)


@[deprecated (since := "2024-10-28")] alias inducing_iff_nhds := isInducing_iff_nhds


lemma nhds_eq_comap (hf : IsInducing f) : ∀ x : X, 𝓝 x = comap f (𝓝 <| f x) :=
  isInducing_iff_nhds.1 hf


lemma basis_nhds {p : ι → Prop} {s : ι → Set Y} (hf : IsInducing f) {x : X}
    (h_basis : (𝓝 (f x)).HasBasis p s) : (𝓝 x).HasBasis p (preimage f ∘ s) :=
  hf.nhds_eq_comap x ▸ h_basis.comap f


lemma nhdsSet_eq_comap (hf : IsInducing f) (s : Set X) :
    𝓝ˢ s = comap f (𝓝ˢ (f '' s)) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace X
    hf : Topology.IsInducing f
    s : Set X
    ⊢ Eq (nhdsSet s) (Filter.comap f (nhdsSet (Set.image f s)))
  -/
  simp only [nhdsSet, sSup_image, comap_iSup, hf.nhds_eq_comap, iSup_image]
  /-
    🎉 no goals
  -/


lemma map_nhds_eq (hf : IsInducing f) (x : X) : (𝓝 x).map f = 𝓝[range f] f x :=
  hf.eq_induced ▸ map_nhds_induced_eq x


lemma map_nhds_of_mem (hf : IsInducing f) (x : X) (h : range f ∈ 𝓝 (f x)) :
    (𝓝 x).map f = 𝓝 (f x) := hf.eq_induced ▸ map_nhds_induced_of_mem h


lemma mapClusterPt_iff (hf : IsInducing f) {x : X} {l : Filter X} :
    MapClusterPt (f x) l f ↔ ClusterPt x l := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace X
    hf : Topology.IsInducing f
    x : X
    l : Filter X
    ⊢ Iff (MapClusterPt (f x) l f) (ClusterPt x l)
  -/
  delta MapClusterPt ClusterPt
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace X
    hf : Topology.IsInducing f
    x : X
    l : Filter X
    ⊢ Iff (Min.min (nhds (f x)) (Filter.map f l)).NeBot (Min.min (nhds x) l).NeBot
  -/
  rw [← Filter.push_pull', ← hf.nhds_eq_comap, map_neBot_iff]
  /-
    🎉 no goals
  -/


lemma image_mem_nhdsWithin (hf : IsInducing f) {x : X} {s : Set X} (hs : s ∈ 𝓝 x) :
    f '' s ∈ 𝓝[range f] f x :=
  hf.map_nhds_eq x ▸ image_mem_map hs


lemma tendsto_nhds_iff {f : ι → Y} {l : Filter ι} {y : Y} (hg : IsInducing g) :
    Tendsto f l (𝓝 y) ↔ Tendsto (g ∘ f) l (𝓝 (g y)) := by
  /-
    Y : Type u_2
    Z : Type u_3
    ι : Type u_4
    g : Y → Z
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ι → Y
    l : Filter ι
    y : Y
    hg : Topology.IsInducing g
    ⊢ Iff (Filter.Tendsto f l (nhds y)) (Filter.Tendsto (Function.comp g f) l (nhd …
  -/
  rw [hg.nhds_eq_comap, tendsto_comap_iff]
  /-
    🎉 no goals
  -/


lemma continuousAt_iff (hg : IsInducing g) {x : X} :
    ContinuousAt f x ↔ ContinuousAt (g ∘ f) x :=
  hg.tendsto_nhds_iff


lemma continuous_iff (hg : IsInducing g) :
    Continuous f ↔ Continuous (g ∘ f) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Z
    hg : Topology.IsInducing g
    ⊢ Iff (Continuous f) (Continuous (Function.comp g f))
  -/
  simp_rw [continuous_iff_continuousAt, hg.continuousAt_iff]
  /-
    🎉 no goals
  -/


lemma continuousAt_iff' (hf : IsInducing f) {x : X} (h : range f ∈ 𝓝 (f x)) :
    ContinuousAt (g ∘ f) x ↔ ContinuousAt g (f x) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Z
    hf : Topology.IsInducing f
    x : X
    h : Membership.mem (nhds (f x)) (Set.range f)
    ⊢ Iff (ContinuousAt (Function.comp g f) x) (ContinuousAt g (f x))
  -/
  simp_rw [ContinuousAt, Filter.Tendsto, ← hf.map_nhds_of_mem _ h, Filter.map_map, comp]
  /-
    🎉 no goals
  -/


protected lemma continuous (hf : IsInducing f) : Continuous f :=
  hf.continuous_iff.mp continuous_id


lemma closure_eq_preimage_closure_image (hf : IsInducing f) (s : Set X) :
    closure s = f ⁻¹' closure (f '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace X
    hf : Topology.IsInducing f
    s : Set X
    ⊢ Eq (closure s) (Set.preimage f (closure (Set.image f s)))
  -/
  ext x
  /-
    case h
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace X
    hf : Topology.IsInducing f
    s : Set X
    x : X
    ⊢ Iff (Membership.mem (closure s) x) (Membership.mem (Set.preimage f (closure  …
  -/
  rw [Set.mem_preimage, ← closure_induced, hf.eq_induced]
  /-
    🎉 no goals
  -/


theorem isClosed_iff (hf : IsInducing f) {s : Set X} :
                                                     /-
                                                       X : Type u_1
                                                       Y : Type u_2
                                                       f : X → Y
                                                       inst✝¹ : TopologicalSpace Y
                                                       inst✝ : TopologicalSpace X
                                                       hf : Topology.IsInducing f
                                                       s : Set X
                                                       ⊢ Iff (IsClosed s) (Exists fun t => And (IsClosed t) (Eq (Set.preimage f t) s))
                                                     -/
    IsClosed s ↔ ∃ t, IsClosed t ∧ f ⁻¹' t = s := by rw [hf.eq_induced, isClosed_induced_iff]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem isClosed_iff' (hf : IsInducing f) {s : Set X} :
                                                           /-
                                                             X : Type u_1
                                                             Y : Type u_2
                                                             f : X → Y
                                                             inst✝¹ : TopologicalSpace Y
                                                             inst✝ : TopologicalSpace X
                                                             hf : Topology.IsInducing f
                                                             s : Set X
                                                             ⊢ Iff (IsClosed s) (∀ (x : X), Membership.mem (closure (Set.image f s)) (f x)  …
                                                           -/
    IsClosed s ↔ ∀ x, f x ∈ closure (f '' s) → x ∈ s := by rw [hf.eq_induced, isClosed_induced_iff']
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem isClosed_preimage (h : IsInducing f) (s : Set Y) (hs : IsClosed s) :
    IsClosed (f ⁻¹' s) :=
  (isClosed_iff h).mpr ⟨s, hs, rfl⟩


theorem isOpen_iff (hf : IsInducing f) {s : Set X} :
                                                 /-
                                                   X : Type u_1
                                                   Y : Type u_2
                                                   f : X → Y
                                                   inst✝¹ : TopologicalSpace Y
                                                   inst✝ : TopologicalSpace X
                                                   hf : Topology.IsInducing f
                                                   s : Set X
                                                   ⊢ Iff (IsOpen s) (Exists fun t => And (IsOpen t) (Eq (Set.preimage f t) s))
                                                 -/
    IsOpen s ↔ ∃ t, IsOpen t ∧ f ⁻¹' t = s := by rw [hf.eq_induced, isOpen_induced_iff]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem setOf_isOpen (hf : IsInducing f) :
    {s : Set X | IsOpen s} = preimage f '' {t | IsOpen t} :=
  Set.ext fun _ ↦ hf.isOpen_iff


theorem dense_iff (hf : IsInducing f) {s : Set X} :
    Dense s ↔ ∀ x, f x ∈ closure (f '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace X
    hf : Topology.IsInducing f
    s : Set X
    ⊢ Iff (Dense s) (∀ (x : X), Membership.mem (closure (Set.image f s)) (f x))
  -/
  simp only [Dense, hf.closure_eq_preimage_closure_image, mem_preimage]
  /-
    🎉 no goals
  -/


theorem of_subsingleton [Subsingleton X] (f : X → Y) : IsInducing f :=
  ⟨Subsingleton.elim _ _⟩


lemma induced [t : TopologicalSpace Y] (hf : Injective f) :
    @IsEmbedding X Y (t.induced f) t f :=
  @IsEmbedding.mk X Y (t.induced f) t _ (.induced f) hf


alias _root_.Function.Injective.isEmbedding_induced := IsEmbedding.induced


@[deprecated (since := "2024-10-26")]
alias Function.Injective.embedding_induced := _root_.Function.Injective.isEmbedding_induced


lemma isInducing (hf : IsEmbedding f) : IsInducing f := hf.toIsInducing


@[deprecated (since := "2024-10-28")] alias inducing := isInducing


lemma mk' (f : X → Y) (inj : Injective f) (induced : ∀ x, comap f (𝓝 (f x)) = 𝓝 x) :
    IsEmbedding f :=
  ⟨isInducing_iff_nhds.2 fun x => (induced x).symm, inj⟩


@[deprecated (since := "2024-10-26")]
alias Embedding.mk' := mk'


protected lemma id : IsEmbedding (@id X) := ⟨.id, fun _ _ h => h⟩


@[deprecated (since := "2024-10-26")]
alias embedding_id := IsEmbedding.id


protected lemma comp (hg : IsEmbedding g) (hf : IsEmbedding f) : IsEmbedding (g ∘ f) :=
  { hg.isInducing.comp hf.isInducing with injective := fun _ _ h => hf.injective <| hg.injective h }


@[deprecated (since := "2024-10-26")]
alias Embedding.comp := IsEmbedding.comp


lemma of_comp_iff (hg : IsEmbedding g) : IsEmbedding (g ∘ f) ↔ IsEmbedding f := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hg : Topology.IsEmbedding g
    ⊢ Iff (Topology.IsEmbedding (Function.comp g f)) (Topology.IsEmbedding f)
  -/
  simp_rw [isEmbedding_iff, hg.isInducing.of_comp_iff, hg.injective.of_comp_iff f]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias Embedding.of_comp_iff := of_comp_iff


protected lemma of_comp (hf : Continuous f) (hg : Continuous g) (hgf : IsEmbedding (g ∘ f)) :
    IsEmbedding f where
  toIsInducing := hgf.isInducing.of_comp hf hg
  injective := hgf.injective.of_comp


@[deprecated (since := "2024-10-26")]
alias embedding_of_embedding_compose := IsEmbedding.of_comp


lemma of_leftInverse {f : X → Y} {g : Y → X} (h : LeftInverse f g) (hf : Continuous f)
    (hg : Continuous g) : IsEmbedding g := .of_comp hg hf <| h.comp_eq_id.symm ▸ .id


alias _root_.Function.LeftInverse.isEmbedding := of_leftInverse


@[deprecated (since := "2024-10-26")]
alias _root_.Function.LeftInverse.embedding := Function.LeftInverse.isEmbedding


lemma map_nhds_eq (hf : IsEmbedding f) (x : X) :     (𝓝 x).map f = 𝓝[range f] f x :=
  hf.1.map_nhds_eq x


@[deprecated (since := "2024-10-26")]
alias Embedding.map_nhds_eq := map_nhds_eq


lemma map_nhds_of_mem (hf : IsEmbedding f) (x : X) (h : range f ∈ 𝓝 (f x)) :
    (𝓝 x).map f = 𝓝 (f x) :=
  hf.1.map_nhds_of_mem x h


@[deprecated (since := "2024-10-26")]
alias Embedding.map_nhds_of_mem := map_nhds_of_mem


lemma tendsto_nhds_iff {f : ι → Y} {l : Filter ι} {y : Y} (hg : IsEmbedding g) :
    Tendsto f l (𝓝 y) ↔ Tendsto (g ∘ f) l (𝓝 (g y)) := hg.isInducing.tendsto_nhds_iff


lemma continuous_iff (hg : IsEmbedding g) : Continuous f ↔ Continuous (g ∘ f) :=
  hg.isInducing.continuous_iff


@[deprecated (since := "2024-10-26")]
alias Embedding.continuous_iff := continuous_iff


lemma continuous (hf : IsEmbedding f) : Continuous f := hf.isInducing.continuous


lemma closure_eq_preimage_closure_image (hf : IsEmbedding f) (s : Set X) :
    closure s = f ⁻¹' closure (f '' s) :=
  hf.1.closure_eq_preimage_closure_image s


@[deprecated (since := "2024-10-26")]
alias Embedding.closure_eq_preimage_closure_image := closure_eq_preimage_closure_image


/-- The topology induced under an inclusion `f : X → Y` from a discrete topological space `Y`
is the discrete topology on `X`.

See also `DiscreteTopology.of_continuous_injective`. -/
lemma discreteTopology [DiscreteTopology Y] (hf : IsEmbedding f) : DiscreteTopology X :=
  .of_continuous_injective hf.continuous hf.injective


@[deprecated (since := "2024-10-26")]
alias Embedding.discreteTopology := discreteTopology


lemma of_subsingleton [Subsingleton X] (f : X → Y) : IsEmbedding f :=
  ⟨.of_subsingleton f, f.injective_of_subsingleton⟩


@[deprecated (since := "2024-10-26")]
alias Embedding.of_subsingleton := of_subsingleton


lemma isQuotientMap_iff : IsQuotientMap f ↔ Surjective f ∧ ∀ s, IsOpen s ↔ IsOpen (f ⁻¹' s) :=
  (isQuotientMap_iff' _).trans <| and_congr Iff.rfl TopologicalSpace.ext_iff


@[deprecated (since := "2024-10-22")]
alias quotientMap_iff := isQuotientMap_iff


theorem isQuotientMap_iff_isClosed :
    IsQuotientMap f ↔ Surjective f ∧ ∀ s : Set Y, IsClosed s ↔ IsClosed (f ⁻¹' s) :=
  isQuotientMap_iff.trans <| Iff.rfl.and <| compl_surjective.forall.trans <| by
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      ⊢ Iff (∀ (x : Set Y), Iff (IsOpen (HasCompl.compl x)) (IsOpen (Set.preimage f  …
    -/
    simp only [isOpen_compl_iff, preimage_compl]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-22")]
alias quotientMap_iff_closed := isQuotientMap_iff_isClosed

@[deprecated (since := "2024-11-19")]
alias isQuotientMap_iff_closed := isQuotientMap_iff_isClosed


protected theorem id : IsQuotientMap (@id X) :=
  ⟨fun x => ⟨x, rfl⟩, coinduced_id.symm⟩


protected theorem comp (hg : IsQuotientMap g) (hf : IsQuotientMap f) : IsQuotientMap (g ∘ f) :=
                                        /-
                                          X : Type u_1
                                          Y : Type u_2
                                          Z : Type u_3
                                          f : X → Y
                                          g : Y → Z
                                          inst✝² : TopologicalSpace X
                                          inst✝¹ : TopologicalSpace Y
                                          inst✝ : TopologicalSpace Z
                                          hg : Topology.IsQuotientMap g
                                          hf : Topology.IsQuotientMap f
                                          ⊢ Eq inst✝ (TopologicalSpace.coinduced (Function.comp g f) inst✝²)
                                        -/
  ⟨hg.surjective.comp hf.surjective, by rw [hg.eq_coinduced, hf.eq_coinduced, coinduced_compose]⟩
                                        /-
                                          🎉 no goals
                                        -/


protected theorem of_comp (hf : Continuous f) (hg : Continuous g)
    (hgf : IsQuotientMap (g ∘ f)) : IsQuotientMap g :=
  ⟨hgf.1.of_comp,
    le_antisymm
          /-
            X : Type u_1
            Y : Type u_2
            Z : Type u_3
            f : X → Y
            g : Y → Z
            inst✝² : TopologicalSpace X
            inst✝¹ : TopologicalSpace Y
            inst✝ : TopologicalSpace Z
            hf : Continuous f
            hg : Continuous g
            hgf : Topology.IsQuotientMap (Function.comp g f)
            ⊢ LE.le inst✝ (TopologicalSpace.coinduced g inst✝¹)
          -/
      (by rw [hgf.eq_coinduced, ← coinduced_compose]; exact coinduced_mono hf.coinduced_le)
                                                      /-
                                                        🎉 no goals
                                                      -/
      hg.coinduced_le⟩


@[deprecated (since := "2024-10-22")]
alias of_quotientMap_compose := IsQuotientMap.of_comp


theorem of_inverse {g : Y → X} (hf : Continuous f) (hg : Continuous g) (h : LeftInverse g f) :
    IsQuotientMap g := .of_comp hf hg <| h.comp_eq_id.symm ▸ IsQuotientMap.id


protected theorem continuous_iff (hf : IsQuotientMap f) : Continuous g ↔ Continuous (g ∘ f) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hf : Topology.IsQuotientMap f
    ⊢ Iff (Continuous g) (Continuous (Function.comp g f))
  -/
  rw [continuous_iff_coinduced_le, continuous_iff_coinduced_le, hf.eq_coinduced, coinduced_compose]
  /-
    🎉 no goals
  -/


protected theorem continuous (hf : IsQuotientMap f) : Continuous f :=
  hf.continuous_iff.mp continuous_id


protected lemma isOpen_preimage (hf : IsQuotientMap f) {s : Set Y} : IsOpen (f ⁻¹' s) ↔ IsOpen s :=
  ((isQuotientMap_iff.1 hf).2 s).symm


protected theorem isClosed_preimage (hf : IsQuotientMap f) {s : Set Y} :
    IsClosed (f ⁻¹' s) ↔ IsClosed s :=
  ((isQuotientMap_iff_isClosed.1 hf).2 s).symm


                                                           /-
                                                             X : Type u_1
                                                             inst✝ : TopologicalSpace X
                                                             s : Set X
                                                             hs : IsOpen s
                                                             ⊢ IsOpen (Set.image id s)
                                                           -/
protected theorem id : IsOpenMap (@id X) := fun s hs => by rwa [image_id]
                                                           /-
                                                             🎉 no goals
                                                           -/


protected theorem comp (hg : IsOpenMap g) (hf : IsOpenMap f) :
                                        /-
                                          X : Type u_1
                                          Y : Type u_2
                                          Z : Type u_3
                                          f : X → Y
                                          g : Y → Z
                                          inst✝² : TopologicalSpace X
                                          inst✝¹ : TopologicalSpace Y
                                          inst✝ : TopologicalSpace Z
                                          hg : IsOpenMap g
                                          hf : IsOpenMap f
                                          s : Set X
                                          hs : IsOpen s
                                          ⊢ IsOpen (Set.image (Function.comp g f) s)
                                        -/
    IsOpenMap (g ∘ f) := fun s hs => by rw [image_comp]; exact hg _ (hf _ hs)
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem isOpen_range (hf : IsOpenMap f) : IsOpen (range f) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : IsOpenMap f
    ⊢ IsOpen (Set.range f)
  -/
  rw [← image_univ]
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : IsOpenMap f
    ⊢ IsOpen (Set.image f Set.univ)
  -/
  exact hf _ isOpen_univ
  /-
    🎉 no goals
  -/


theorem image_mem_nhds (hf : IsOpenMap f) {x : X} {s : Set X} (hx : s ∈ 𝓝 x) : f '' s ∈ 𝓝 (f x) :=
  let ⟨t, hts, ht, hxt⟩ := mem_nhds_iff.1 hx
  mem_of_superset (IsOpen.mem_nhds (hf t ht) (mem_image_of_mem _ hxt)) (image_subset _ hts)


theorem range_mem_nhds (hf : IsOpenMap f) (x : X) : range f ∈ 𝓝 (f x) :=
  hf.isOpen_range.mem_nhds <| mem_range_self _


theorem mapsTo_interior (hf : IsOpenMap f) {s : Set X} {t : Set Y} (h : MapsTo f s t) :
    MapsTo f (interior s) (interior t) :=
  mapsTo'.2 <|
    interior_maximal (h.mono interior_subset Subset.rfl).image_subset (hf _ isOpen_interior)


theorem image_interior_subset (hf : IsOpenMap f) (s : Set X) :
    f '' interior s ⊆ interior (f '' s) :=
  (hf.mapsTo_interior (mapsTo_image f s)).image_subset


theorem nhds_le (hf : IsOpenMap f) (x : X) : 𝓝 (f x) ≤ (𝓝 x).map f :=
  le_map fun _ => hf.image_mem_nhds


theorem of_nhds_le (hf : ∀ x, 𝓝 (f x) ≤ map f (𝓝 x)) : IsOpenMap f := fun _s hs =>
  isOpen_iff_mem_nhds.2 fun _y ⟨_x, hxs, hxy⟩ => hxy ▸ hf _ (image_mem_map <| hs.mem_nhds hxs)


theorem of_sections
    (h : ∀ x, ∃ g : Y → X, ContinuousAt g (f x) ∧ g (f x) = x ∧ RightInverse g f) : IsOpenMap f :=
  of_nhds_le fun x =>
    let ⟨g, hgc, hgx, hgf⟩ := h x
    calc
                                              /-
                                                X : Type u_1
                                                Y : Type u_2
                                                f : X → Y
                                                inst✝¹ : TopologicalSpace X
                                                inst✝ : TopologicalSpace Y
                                                h : ∀ (x : X), Exists fun g => And (ContinuousAt g (f x)) (And (Eq (g (f x)) x …
                                                x : X
                                                g : Y → X
                                                hgc : ContinuousAt g (f x)
                                                hgx : Eq (g (f x)) x
                                                hgf : Function.RightInverse g f
                                                ⊢ Eq (nhds (f x)) (Filter.map f (Filter.map g (nhds (f x))))
                                              -/
      𝓝 (f x) = map f (map g (𝓝 (f x))) := by rw [map_map, hgf.comp_eq_id, map_id]
                                              /-
                                                🎉 no goals
                                              -/
      _ ≤ map f (𝓝 (g (f x))) := map_mono hgc
                            /-
                              X : Type u_1
                              Y : Type u_2
                              f : X → Y
                              inst✝¹ : TopologicalSpace X
                              inst✝ : TopologicalSpace Y
                              h : ∀ (x : X), Exists fun g => And (ContinuousAt g (f x)) (And (Eq (g (f x)) x …
                              x : X
                              g : Y → X
                              hgc : ContinuousAt g (f x)
                              hgx : Eq (g (f x)) x
                              hgf : Function.RightInverse g f
                              ⊢ Eq (Filter.map f (nhds (g (f x)))) (Filter.map f (nhds x))
                            -/
      _ = map f (𝓝 x) := by rw [hgx]
                            /-
                              🎉 no goals
                            -/


theorem of_inverse {f' : Y → X} (h : Continuous f') (l_inv : LeftInverse f f')
    (r_inv : RightInverse f f') : IsOpenMap f :=
  of_sections fun _ => ⟨f', h.continuousAt, r_inv _, l_inv⟩


/-- A continuous surjective open map is a quotient map. -/
theorem isQuotientMap (open_map : IsOpenMap f) (cont : Continuous f) (surj : Surjective f) :
    IsQuotientMap f :=
  isQuotientMap_iff.2
    ⟨surj, fun s => ⟨fun h => h.preimage cont, fun h => surj.image_preimage s ▸ open_map _ h⟩⟩


@[deprecated (since := "2024-10-22")]
alias to_quotientMap := isQuotientMap


theorem interior_preimage_subset_preimage_interior (hf : IsOpenMap f) {s : Set Y} :
    interior (f ⁻¹' s) ⊆ f ⁻¹' interior s :=
  hf.mapsTo_interior (mapsTo_preimage _ _)


theorem preimage_interior_eq_interior_preimage (hf₁ : IsOpenMap f) (hf₂ : Continuous f)
    (s : Set Y) : f ⁻¹' interior s = interior (f ⁻¹' s) :=
  Subset.antisymm (preimage_interior_subset_interior_preimage hf₂)
    (interior_preimage_subset_preimage_interior hf₁)


theorem preimage_closure_subset_closure_preimage (hf : IsOpenMap f) {s : Set Y} :
    f ⁻¹' closure s ⊆ closure (f ⁻¹' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : IsOpenMap f
    s : Set Y
    ⊢ HasSubset.Subset (Set.preimage f (closure s)) (closure (Set.preimage f s))
  -/
  rw [← compl_subset_compl]
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : IsOpenMap f
    s : Set Y
    ⊢ HasSubset.Subset (HasCompl.compl (closure (Set.preimage f s))) (HasCompl.com …
  -/
  simp only [← interior_compl, ← preimage_compl, hf.interior_preimage_subset_preimage_interior]
  /-
    🎉 no goals
  -/


theorem preimage_closure_eq_closure_preimage (hf : IsOpenMap f) (hfc : Continuous f) (s : Set Y) :
    f ⁻¹' closure s = closure (f ⁻¹' s) :=
  hf.preimage_closure_subset_closure_preimage.antisymm (hfc.closure_preimage_subset s)


theorem preimage_frontier_subset_frontier_preimage (hf : IsOpenMap f) {s : Set Y} :
    f ⁻¹' frontier s ⊆ frontier (f ⁻¹' s) := by
  simpa only [frontier_eq_closure_inter_closure, preimage_inter] using
    inter_subset_inter hf.preimage_closure_subset_closure_preimage
      hf.preimage_closure_subset_closure_preimage


theorem preimage_frontier_eq_frontier_preimage (hf : IsOpenMap f) (hfc : Continuous f) (s : Set Y) :
    f ⁻¹' frontier s = frontier (f ⁻¹' s) := by
  simp only [frontier_eq_closure_inter_closure, preimage_inter, preimage_compl,
    hf.preimage_closure_eq_closure_preimage hfc]


theorem of_isEmpty [h : IsEmpty X] (f : X → Y) : IsOpenMap f := of_nhds_le h.elim


theorem isOpenMap_iff_nhds_le : IsOpenMap f ↔ ∀ x : X, 𝓝 (f x) ≤ (𝓝 x).map f :=
  ⟨fun hf => hf.nhds_le, IsOpenMap.of_nhds_le⟩


theorem isOpenMap_iff_interior : IsOpenMap f ↔ ∀ s, f '' interior s ⊆ interior (f '' s) :=
  ⟨IsOpenMap.image_interior_subset, fun hs u hu =>
    subset_interior_iff_isOpen.mp <|
      calc
                                       /-
                                         X : Type u_1
                                         Y : Type u_2
                                         f : X → Y
                                         inst✝¹ : TopologicalSpace X
                                         inst✝ : TopologicalSpace Y
                                         hs : ∀ (s : Set X), HasSubset.Subset (Set.image f (interior s)) (interior (Set …
                                         u : Set X
                                         hu : IsOpen u
                                         ⊢ Eq (Set.image f u) (Set.image f (interior u))
                                       -/
        f '' u = f '' interior u := by rw [hu.interior_eq]
                                       /-
                                         🎉 no goals
                                       -/
        _ ⊆ interior (f '' u) := hs u⟩


/-- An inducing map with an open range is an open map. -/
protected lemma Topology.IsInducing.isOpenMap (hi : IsInducing f) (ho : IsOpen (range f)) :
    IsOpenMap f :=
  IsOpenMap.of_nhds_le fun _ => (hi.map_nhds_of_mem _ <| IsOpen.mem_nhds ho <| mem_range_self _).ge


@[deprecated (since := "2024-10-28")] alias Inducing.isOpenMap := IsInducing.isOpenMap


/-- Preimage of a dense set under an open map is dense. -/
protected theorem Dense.preimage {s : Set Y} (hs : Dense s) (hf : IsOpenMap f) :
    Dense (f ⁻¹' s) := fun x ↦
  hf.preimage_closure_subset_closure_preimage <| hs (f x)


                                                             /-
                                                               X : Type u_1
                                                               inst✝ : TopologicalSpace X
                                                               s : Set X
                                                               hs : IsClosed s
                                                               ⊢ IsClosed (Set.image id s)
                                                             -/
protected theorem id : IsClosedMap (@id X) := fun s hs => by rwa [image_id]
                                                             /-
                                                               🎉 no goals
                                                             -/


protected theorem comp (hg : IsClosedMap g) (hf : IsClosedMap f) : IsClosedMap (g ∘ f) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hg : IsClosedMap g
    hf : IsClosedMap f
    ⊢ IsClosedMap (Function.comp g f)
  -/
  intro s hs
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hg : IsClosedMap g
    hf : IsClosedMap f
    s : Set X
    hs : IsClosed s
    ⊢ IsClosed (Set.image (Function.comp g f) s)
  -/
  rw [image_comp]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hg : IsClosedMap g
    hf : IsClosedMap f
    s : Set X
    hs : IsClosed s
    ⊢ IsClosed (Set.image g (Set.image f s))
  -/
  exact hg _ (hf _ hs)
  /-
    🎉 no goals
  -/


protected theorem of_comp_surjective (hf : Surjective f) (hf' : Continuous f)
    (hfg : IsClosedMap (g ∘ f)) : IsClosedMap g := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hf : Function.Surjective f
    hf' : Continuous f
    hfg : IsClosedMap (Function.comp g f)
    ⊢ IsClosedMap g
  -/
  intro K hK
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hf : Function.Surjective f
    hf' : Continuous f
    hfg : IsClosedMap (Function.comp g f)
    K : Set Y
    hK : IsClosed K
    ⊢ IsClosed (Set.image g K)
  -/
  rw [← image_preimage_eq K hf, ← image_comp]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hf : Function.Surjective f
    hf' : Continuous f
    hfg : IsClosedMap (Function.comp g f)
    K : Set Y
    hK : IsClosed K
    ⊢ IsClosed (Set.image (Function.comp g f) (Set.preimage f K))
  -/
  exact hfg _ (hK.preimage hf')
  /-
    🎉 no goals
  -/


theorem closure_image_subset (hf : IsClosedMap f) (s : Set X) :
    closure (f '' s) ⊆ f '' closure s :=
  closure_minimal (image_subset _ subset_closure) (hf _ isClosed_closure)


theorem of_inverse {f' : Y → X} (h : Continuous f') (l_inv : LeftInverse f f')
    (r_inv : RightInverse f f') : IsClosedMap f := fun s hs => by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f' : Y → X
    h : Continuous f'
    l_inv : Function.LeftInverse f f'
    r_inv : Function.RightInverse f f'
    s : Set X
    hs : IsClosed s
    ⊢ IsClosed (Set.image f s)
  -/
  rw [image_eq_preimage_of_inverse r_inv l_inv]
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f' : Y → X
    h : Continuous f'
    l_inv : Function.LeftInverse f f'
    r_inv : Function.RightInverse f f'
    s : Set X
    hs : IsClosed s
    ⊢ IsClosed (Set.preimage f' s)
  -/
  exact hs.preimage h
  /-
    🎉 no goals
  -/


theorem of_nonempty (h : ∀ s, IsClosed s → s.Nonempty → IsClosed (f '' s)) :
    IsClosedMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : ∀ (s : Set X), IsClosed s → s.Nonempty → IsClosed (Set.image f s)
    ⊢ IsClosedMap f
  -/
  intro s hs; rcases eq_empty_or_nonempty s with h2s | h2s
    /-
      case inl
      X : Type u_1
      Y : Type u_2
      f : X → Y
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      h : ∀ (s : Set X), IsClosed s → s.Nonempty → IsClosed (Set.image f s)
      s : Set X
      hs : IsClosed s
      h2s : Eq s EmptyCollection.emptyCollection
      ⊢ IsClosed (Set.image f s)
    -/
  · simp_rw [h2s, image_empty, isClosed_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      Y : Type u_2
      f : X → Y
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      h : ∀ (s : Set X), IsClosed s → s.Nonempty → IsClosed (Set.image f s)
      s : Set X
      hs : IsClosed s
      h2s : s.Nonempty
      ⊢ IsClosed (Set.image f s)
    -/
  · exact h s hs h2s
    /-
      🎉 no goals
    -/


theorem isClosed_range (hf : IsClosedMap f) : IsClosed (range f) :=
  @image_univ _ _ f ▸ hf _ isClosed_univ


@[deprecated (since := "2024-03-17")] alias closed_range := isClosed_range


theorem isQuotientMap (hcl : IsClosedMap f) (hcont : Continuous f)
    (hsurj : Surjective f) : IsQuotientMap f :=
  isQuotientMap_iff_isClosed.2 ⟨hsurj, fun s =>
    ⟨fun hs => hs.preimage hcont, fun hs => hsurj.image_preimage s ▸ hcl _ hs⟩⟩


lemma Topology.IsInducing.isClosedMap (hf : IsInducing f) (h : IsClosed (range f)) :
    IsClosedMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsInducing f
    h : IsClosed (Set.range f)
    ⊢ IsClosedMap f
  -/
  intro s hs
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsInducing f
    h : IsClosed (Set.range f)
    s : Set X
    hs : IsClosed s
    ⊢ IsClosed (Set.image f s)
  -/
  rcases hf.isClosed_iff.1 hs with ⟨t, ht, rfl⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsInducing f
    h : IsClosed (Set.range f)
    t : Set Y
    ht : IsClosed t
    hs : IsClosed (Set.preimage f t)
    ⊢ IsClosed (Set.image f (Set.preimage f t))
  -/
  rw [image_preimage_eq_inter_range]
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsInducing f
    h : IsClosed (Set.range f)
    t : Set Y
    ht : IsClosed t
    hs : IsClosed (Set.preimage f t)
    ⊢ IsClosed (Inter.inter t (Set.range f))
  -/
  exact ht.inter h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias Inducing.isClosedMap := IsInducing.isClosedMap


theorem isClosedMap_iff_closure_image :
    IsClosedMap f ↔ ∀ s, closure (f '' s) ⊆ f '' closure s :=
  ⟨IsClosedMap.closure_image_subset, fun hs c hc =>
    isClosed_of_closure_subset <|
      calc
        closure (f '' c) ⊆ f '' closure c := hs c
                         /-
                           X : Type u_1
                           Y : Type u_2
                           f : X → Y
                           inst✝¹ : TopologicalSpace X
                           inst✝ : TopologicalSpace Y
                           hs : ∀ (s : Set X), HasSubset.Subset (closure (Set.image f s)) (Set.image f (c …
                           c : Set X
                           hc : IsClosed c
                           ⊢ Eq (Set.image f (closure c)) (Set.image f c)
                         -/
        _ = f '' c := by rw [hc.closure_eq]⟩
                         /-
                           🎉 no goals
                         -/


/-- A map `f : X → Y` is closed if and only if for all sets `s`, any cluster point of `f '' s` is
the image by `f` of some cluster point of `s`.
If you require this for all filters instead of just principal filters, and also that `f` is
continuous, you get the notion of **proper map**. See `isProperMap_iff_clusterPt`. -/
theorem isClosedMap_iff_clusterPt :
    IsClosedMap f ↔ ∀ s y, MapClusterPt y (𝓟 s) f → ∃ x, f x = y ∧ ClusterPt x (𝓟 s) := by
  simp [MapClusterPt, isClosedMap_iff_closure_image, subset_def, mem_closure_iff_clusterPt,
    and_comm]


theorem IsClosedMap.closure_image_eq_of_continuous
    (f_closed : IsClosedMap f) (f_cont : Continuous f) (s : Set X) :
    closure (f '' s) = f '' closure s :=
  subset_antisymm (f_closed.closure_image_subset s) (image_closure_subset_closure_image f_cont)


theorem IsClosedMap.lift'_closure_map_eq
    (f_closed : IsClosedMap f) (f_cont : Continuous f) (F : Filter X) :
    (map f F).lift' closure = map f (F.lift' closure) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f_closed : IsClosedMap f
    f_cont : Continuous f
    F : Filter X
    ⊢ Eq ((Filter.map f F).lift' closure) (Filter.map f (F.lift' closure))
  -/
  rw [map_lift'_eq2 (monotone_closure Y), map_lift'_eq (monotone_closure X)]
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f_closed : IsClosedMap f
    f_cont : Continuous f
    F : Filter X
    ⊢ Eq (F.lift' (Function.comp closure (Set.image f))) (F.lift' (Function.comp ( …
  -/
  congr
  /-
    case e_h
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f_closed : IsClosedMap f
    f_cont : Continuous f
    F : Filter X
    ⊢ Eq (Function.comp closure (Set.image f)) (Function.comp (Set.image f) closure)
  -/
  ext s : 1
  /-
    case e_h.h
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f_closed : IsClosedMap f
    f_cont : Continuous f
    F : Filter X
    s : Set X
    ⊢ Eq (Function.comp closure (Set.image f) s) (Function.comp (Set.image f) clos …
  -/
  exact f_closed.closure_image_eq_of_continuous f_cont s
  /-
    🎉 no goals
  -/


theorem IsClosedMap.mapClusterPt_iff_lift'_closure
    {F : Filter X} (f_closed : IsClosedMap f) (f_cont : Continuous f) {y : Y} :
    MapClusterPt y F f ↔ ((F.lift' closure) ⊓ 𝓟 (f ⁻¹' {y})).NeBot := by
  rw [MapClusterPt, clusterPt_iff_lift'_closure', f_closed.lift'_closure_map_eq f_cont,
      ← comap_principal, ← map_neBot_iff f, Filter.push_pull, principal_singleton]


lemma IsOpenEmbedding.isEmbedding (hf : IsOpenEmbedding f) : IsEmbedding f := hf.toIsEmbedding

lemma IsOpenEmbedding.isInducing (hf : IsOpenEmbedding f) : IsInducing f :=
  hf.isEmbedding.isInducing


@[deprecated (since := "2024-10-28")] alias IsOpenEmbedding.inducing := IsOpenEmbedding.isInducing


lemma IsOpenEmbedding.isOpenMap (hf : IsOpenEmbedding f) : IsOpenMap f :=
  hf.isEmbedding.isInducing.isOpenMap hf.isOpen_range


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.isOpenMap := IsOpenEmbedding.isOpenMap


theorem IsOpenEmbedding.map_nhds_eq (hf : IsOpenEmbedding f) (x : X) :
    map f (𝓝 x) = 𝓝 (f x) :=
  hf.isEmbedding.map_nhds_of_mem _ <| hf.isOpen_range.mem_nhds <| mem_range_self _


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.map_nhds_eq := IsOpenEmbedding.map_nhds_eq


lemma IsOpenEmbedding.isOpen_iff_image_isOpen (hf : IsOpenEmbedding f) {s : Set X} :
    IsOpen s ↔ IsOpen (f '' s) where
  mp := hf.isOpenMap s
              /-
                X : Type u_1
                Y : Type u_2
                f : X → Y
                inst✝¹ : TopologicalSpace X
                inst✝ : TopologicalSpace Y
                hf : Topology.IsOpenEmbedding f
                s : Set X
                h : IsOpen (Set.image f s)
                ⊢ IsOpen s
              -/
  mpr h := by convert ← h.preimage hf.isEmbedding.continuous; apply preimage_image_eq _ hf.injective
                                                              /-
                                                                🎉 no goals
                                                              -/


@[deprecated (since := "2024-10-30")]
alias IsOpenEmbedding.open_iff_image_open := IsOpenEmbedding.isOpen_iff_image_isOpen


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.open_iff_image_open := IsOpenEmbedding.isOpen_iff_image_isOpen


theorem IsOpenEmbedding.tendsto_nhds_iff [TopologicalSpace Z] {f : ι → Y} {l : Filter ι} {y : Y}
    (hg : IsOpenEmbedding g) : Tendsto f l (𝓝 y) ↔ Tendsto (g ∘ f) l (𝓝 (g y)) :=
  hg.isEmbedding.tendsto_nhds_iff


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.tendsto_nhds_iff := IsOpenEmbedding.tendsto_nhds_iff


theorem IsOpenEmbedding.tendsto_nhds_iff' (hf : IsOpenEmbedding f) {l : Filter Z} {x : X} :
    Tendsto (g ∘ f) (𝓝 x) l ↔ Tendsto g (𝓝 (f x)) l := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsOpenEmbedding f
    l : Filter Z
    x : X
    ⊢ Iff (Filter.Tendsto (Function.comp g f) (nhds x) l) (Filter.Tendsto g (nhds  …
  -/
  rw [Tendsto, ← map_map, hf.map_nhds_eq]; rfl
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.tendsto_nhds_iff' := IsOpenEmbedding.tendsto_nhds_iff'


theorem IsOpenEmbedding.continuousAt_iff [TopologicalSpace Z] (hf : IsOpenEmbedding f) {x : X} :
    ContinuousAt (g ∘ f) x ↔ ContinuousAt g (f x) :=
  hf.tendsto_nhds_iff'


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.continuousAt_iff := IsOpenEmbedding.continuousAt_iff


theorem IsOpenEmbedding.continuous (hf : IsOpenEmbedding f) : Continuous f :=
  hf.isEmbedding.continuous


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.continuous := IsOpenEmbedding.continuous


lemma IsOpenEmbedding.isOpen_iff_preimage_isOpen (hf : IsOpenEmbedding f) {s : Set Y}
    (hs : s ⊆ range f) : IsOpen s ↔ IsOpen (f ⁻¹' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsOpenEmbedding f
    s : Set Y
    hs : HasSubset.Subset s (Set.range f)
    ⊢ Iff (IsOpen s) (IsOpen (Set.preimage f s))
  -/
  rw [hf.isOpen_iff_image_isOpen, image_preimage_eq_inter_range, inter_eq_self_of_subset_left hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-30")]
alias IsOpenEmbedding.open_iff_preimage_open := IsOpenEmbedding.isOpen_iff_preimage_isOpen


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.open_iff_preimage_open := IsOpenEmbedding.isOpen_iff_preimage_isOpen


lemma IsOpenEmbedding.of_isEmbedding_isOpenMap (h₁ : IsEmbedding f) (h₂ : IsOpenMap f) :
    IsOpenEmbedding f :=
  ⟨h₁, h₂.isOpen_range⟩


@[deprecated (since := "2024-10-26")]
alias isOpenEmbedding_of_embedding_open := IsOpenEmbedding.of_isEmbedding_isOpenMap


@[deprecated (since := "2024-10-18")]
alias openEmbedding_of_embedding_open := IsOpenEmbedding.of_isEmbedding_isOpenMap


/-- A surjective embedding is an `IsOpenEmbedding`. -/
lemma IsEmbedding.isOpenEmbedding_of_surjective (hf : IsEmbedding f) (hsurj : f.Surjective) :
    IsOpenEmbedding f :=
  ⟨hf, hsurj.range_eq ▸ isOpen_univ⟩


@[deprecated (since := "2024-10-26")]
alias _root_.Embedding.toIsOpenEmbedding_of_surjective := IsEmbedding.isOpenEmbedding_of_surjective


alias IsOpenEmbedding.of_isEmbedding := IsEmbedding.isOpenEmbedding_of_surjective


@[deprecated (since := "2024-10-18")]
alias _root_.Embedding.toOpenEmbedding_of_surjective := IsEmbedding.isOpenEmbedding_of_surjective


lemma isOpenEmbedding_iff_isEmbedding_isOpenMap : IsOpenEmbedding f ↔ IsEmbedding f ∧ IsOpenMap f :=
  ⟨fun h => ⟨h.1, h.isOpenMap⟩, fun h => .of_isEmbedding_isOpenMap h.1 h.2⟩


@[deprecated (since := "2024-10-26")]
alias isOpenEmbedding_iff_embedding_open := isOpenEmbedding_iff_isEmbedding_isOpenMap


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_embedding_open := isOpenEmbedding_iff_isEmbedding_isOpenMap


theorem IsOpenEmbedding.of_continuous_injective_isOpenMap
    (h₁ : Continuous f) (h₂ : Injective f) (h₃ : IsOpenMap f) : IsOpenEmbedding f := by
  simp only [isOpenEmbedding_iff_isEmbedding_isOpenMap, isEmbedding_iff, isInducing_iff_nhds, *,
    and_true]
  exact fun x =>
    le_antisymm (h₁.tendsto _).le_comap (@comap_map _ _ (𝓝 x) _ h₂ ▸ comap_mono (h₃.nhds_le _))


lemma isOpenEmbedding_iff_continuous_injective_isOpenMap :
    IsOpenEmbedding f ↔ Continuous f ∧ Injective f ∧ IsOpenMap f :=
  ⟨fun h => ⟨h.continuous, h.injective, h.isOpenMap⟩, fun h =>
    .of_continuous_injective_isOpenMap h.1 h.2.1 h.2.2⟩


@[deprecated (since := "2024-10-30")]
alias isOpenEmbedding_iff_continuous_injective_open :=
  isOpenEmbedding_iff_continuous_injective_isOpenMap


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_continuous_injective_open :=
  isOpenEmbedding_iff_continuous_injective_isOpenMap


protected lemma id : IsOpenEmbedding (@id X) := ⟨.id, IsOpenMap.id.isOpen_range⟩


@[deprecated (since := "2024-10-18")]
alias _root_.openEmbedding_id := IsOpenEmbedding.id


protected lemma comp (hg : IsOpenEmbedding g)
    (hf : IsOpenEmbedding f) : IsOpenEmbedding (g ∘ f) :=
  ⟨hg.1.comp hf.1, (hg.isOpenMap.comp hf.isOpenMap).isOpen_range⟩


theorem isOpenMap_iff (hg : IsOpenEmbedding g) :
    IsOpenMap f ↔ IsOpenMap (g ∘ f) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : X → Y
    g : Y → Z
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    hg : Topology.IsOpenEmbedding g
    ⊢ Iff (IsOpenMap f) (IsOpenMap (Function.comp g f))
  -/
  simp_rw [isOpenMap_iff_nhds_le, ← map_map, comp, ← hg.map_nhds_eq, map_le_map_iff hg.injective]
  /-
    🎉 no goals
  -/


theorem of_comp_iff (f : X → Y) (hg : IsOpenEmbedding g) :
    IsOpenEmbedding (g ∘ f) ↔ IsOpenEmbedding f := by
  simp only [isOpenEmbedding_iff_continuous_injective_isOpenMap, ← hg.isOpenMap_iff, ←
    hg.1.continuous_iff, hg.injective.of_comp_iff]


lemma of_comp (f : X → Y) (hg : IsOpenEmbedding g) (h : IsOpenEmbedding (g ∘ f)) :
    IsOpenEmbedding f := (IsOpenEmbedding.of_comp_iff f hg).1 h


theorem of_isEmpty [IsEmpty X] (f : X → Y) : IsOpenEmbedding f :=
  of_isEmbedding_isOpenMap (.of_subsingleton f) (.of_isEmpty f)


theorem image_mem_nhds {f : X → Y} (hf : IsOpenEmbedding f) {s : Set X} {x : X} :
    f '' s ∈ 𝓝 (f x) ↔ s ∈ 𝓝 x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsOpenEmbedding f
    s : Set X
    x : X
    ⊢ Iff (Membership.mem (nhds (f x)) (Set.image f s)) (Membership.mem (nhds x) s)
  -/
  rw [← hf.map_nhds_eq, mem_map, preimage_image_eq _ hf.injective]
  /-
    🎉 no goals
  -/


lemma isEmbedding (hf : IsClosedEmbedding f) : IsEmbedding f := hf.toIsEmbedding

lemma isInducing (hf : IsClosedEmbedding f) : IsInducing f := hf.isEmbedding.isInducing

lemma continuous (hf : IsClosedEmbedding f) : Continuous f := hf.isEmbedding.continuous


lemma tendsto_nhds_iff {g : ι → X} {l : Filter ι} {x : X} (hf : IsClosedEmbedding f) :
    Tendsto g l (𝓝 x) ↔ Tendsto (f ∘ g) l (𝓝 (f x)) := hf.isEmbedding.tendsto_nhds_iff


lemma isClosedMap (hf : IsClosedEmbedding f) : IsClosedMap f :=
  hf.isEmbedding.isInducing.isClosedMap hf.isClosed_range


lemma isClosed_iff_image_isClosed (hf : IsClosedEmbedding f) {s : Set X} :
    IsClosed s ↔ IsClosed (f '' s) :=
  ⟨hf.isClosedMap s, fun h => by
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      hf : Topology.IsClosedEmbedding f
      s : Set X
      h : IsClosed (Set.image f s)
      ⊢ IsClosed s
    -/
    rw [← preimage_image_eq s hf.injective]
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      hf : Topology.IsClosedEmbedding f
      s : Set X
      h : IsClosed (Set.image f s)
      ⊢ IsClosed (Set.preimage f (Set.image f s))
    -/
    exact h.preimage hf.continuous⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-30")] alias closed_iff_image_closed := isClosed_iff_image_isClosed


lemma isClosed_iff_preimage_isClosed (hf : IsClosedEmbedding f) {s : Set Y}
    (hs : s ⊆ range f) : IsClosed s ↔ IsClosed (f ⁻¹' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    hf : Topology.IsClosedEmbedding f
    s : Set Y
    hs : HasSubset.Subset s (Set.range f)
    ⊢ Iff (IsClosed s) (IsClosed (Set.preimage f s))
  -/
  rw [hf.isClosed_iff_image_isClosed, image_preimage_eq_of_subset hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-30")]
alias closed_iff_preimage_closed := isClosed_iff_preimage_isClosed


lemma of_isEmbedding_isClosedMap (h₁ : IsEmbedding f) (h₂ : IsClosedMap f) :
    IsClosedEmbedding f :=
  ⟨h₁, image_univ (f := f) ▸ h₂ univ isClosed_univ⟩


@[deprecated (since := "2024-10-26")]
alias _root_.IsClosedEmbedding.of_embedding_closed := of_isEmbedding_isClosedMap


@[deprecated (since := "2024-10-20")]
alias _root_.closedEmbedding_of_embedding_closed := of_isEmbedding_isClosedMap


lemma of_continuous_injective_isClosedMap (h₁ : Continuous f) (h₂ : Injective f)
    (h₃ : IsClosedMap f) : IsClosedEmbedding f := by
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h₁ : Continuous f
    h₂ : Function.Injective f
    h₃ : IsClosedMap f
    ⊢ Topology.IsClosedEmbedding f
  -/
  refine .of_isEmbedding_isClosedMap ⟨⟨?_⟩, h₂⟩ h₃
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h₁ : Continuous f
    h₂ : Function.Injective f
    h₃ : IsClosedMap f
    ⊢ Eq inst✝¹ (TopologicalSpace.induced f inst✝)
  -/
  refine h₁.le_induced.antisymm fun s hs => ?_
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h₁ : Continuous f
    h₂ : Function.Injective f
    h₃ : IsClosedMap f
    s : Set X
    hs : IsOpen s
    ⊢ IsOpen s
  -/
  refine ⟨(f '' sᶜ)ᶜ, (h₃ _ hs.isClosed_compl).isOpen_compl, ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    f : X → Y
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h₁ : Continuous f
    h₂ : Function.Injective f
    h₃ : IsClosedMap f
    s : Set X
    hs : IsOpen s
    ⊢ Eq (Set.preimage f (HasCompl.compl (Set.image f (HasCompl.compl s)))) s
  -/
  rw [preimage_compl, preimage_image_eq _ h₂, compl_compl]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias _root_.closedEmbedding_of_continuous_injective_closed :=
  IsClosedEmbedding.of_continuous_injective_isClosedMap


protected theorem id : IsClosedEmbedding (@id X) := ⟨.id, IsClosedMap.id.isClosed_range⟩


@[deprecated (since := "2024-10-20")]
alias _root_.closedEmbedding_id := IsClosedEmbedding.id


theorem comp (hg : IsClosedEmbedding g) (hf : IsClosedEmbedding f) :
    IsClosedEmbedding (g ∘ f) :=
  ⟨hg.isEmbedding.comp hf.isEmbedding, (hg.isClosedMap.comp hf.isClosedMap).isClosed_range⟩


lemma of_comp_iff (hg : IsClosedEmbedding g) : IsClosedEmbedding (g ∘ f) ↔ IsClosedEmbedding f := by
  simp_rw [isClosedEmbedding_iff, hg.isEmbedding.of_comp_iff, Set.range_comp,
    ← hg.isClosed_iff_image_isClosed]


theorem closure_image_eq (hf : IsClosedEmbedding f) (s : Set X) :
    closure (f '' s) = f '' closure s :=
  hf.isClosedMap.closure_image_eq_of_continuous hf.continuous s


