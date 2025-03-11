/-- A T₀ space, also known as a Kolmogorov space, is a topological space such that for every pair
`x ≠ y`, there is an open set containing one but not the other. We formulate the definition in terms
of the `Inseparable` relation. -/
class T0Space (X : Type u) [TopologicalSpace X] : Prop where
  /-- Two inseparable points in a T₀ space are equal. -/
  t0 : ∀ ⦃x y : X⦄, Inseparable x y → x = y


theorem t0Space_iff_inseparable (X : Type u) [TopologicalSpace X] :
    T0Space X ↔ ∀ x y : X, Inseparable x y → x = y :=
  ⟨fun ⟨h⟩ => h, fun h => ⟨h⟩⟩


theorem t0Space_iff_not_inseparable (X : Type u) [TopologicalSpace X] :
    T0Space X ↔ Pairwise fun x y : X => ¬Inseparable x y := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ Iff (T0Space X) (Pairwise fun x y => Not (Inseparable x y))
  -/
  simp only [t0Space_iff_inseparable, Ne, not_imp_not, Pairwise]
  /-
    🎉 no goals
  -/


theorem Inseparable.eq [T0Space X] {x y : X} (h : Inseparable x y) : x = y :=
  T0Space.t0 h


/-- A topology inducing map from a T₀ space is injective. -/
protected theorem Topology.IsInducing.injective [TopologicalSpace Y] [T0Space X] {f : X → Y}
    (hf : IsInducing f) : Injective f := fun _ _ h =>
  (hf.inseparable_iff.1 <| .of_eq h).eq


@[deprecated (since := "2024-10-28")] alias Inducing.injective := IsInducing.injective


/-- A topology inducing map from a T₀ space is a topological embedding. -/
protected theorem Topology.IsInducing.isEmbedding [TopologicalSpace Y] [T0Space X] {f : X → Y}
    (hf : IsInducing f) : IsEmbedding f :=
  ⟨hf, hf.injective⟩


@[deprecated (since := "2024-10-28")] alias Inducing.isEmbedding := IsInducing.isEmbedding


@[deprecated (since := "2024-10-26")]
alias Inducing.embedding := Topology.IsInducing.isEmbedding


lemma isEmbedding_iff_isInducing [TopologicalSpace Y] [T0Space X] {f : X → Y} :
    IsEmbedding f ↔ IsInducing f :=
  ⟨IsEmbedding.isInducing, IsInducing.isEmbedding⟩


@[deprecated (since := "2024-10-28")] alias isEmbedding_iff_inducing := isEmbedding_iff_isInducing


@[deprecated (since := "2024-10-26")]
alias embedding_iff_inducing := isEmbedding_iff_isInducing


theorem t0Space_iff_nhds_injective (X : Type u) [TopologicalSpace X] :
    T0Space X ↔ Injective (𝓝 : X → Filter X) :=
  t0Space_iff_inseparable X


theorem nhds_injective [T0Space X] : Injective (𝓝 : X → Filter X) :=
  (t0Space_iff_nhds_injective X).1 ‹_›


theorem inseparable_iff_eq [T0Space X] {x y : X} : Inseparable x y ↔ x = y :=
  nhds_injective.eq_iff


@[simp]
theorem nhds_eq_nhds_iff [T0Space X] {a b : X} : 𝓝 a = 𝓝 b ↔ a = b :=
  nhds_injective.eq_iff


@[simp]
theorem inseparable_eq_eq [T0Space X] : Inseparable = @Eq X :=
  funext₂ fun _ _ => propext inseparable_iff_eq


theorem TopologicalSpace.IsTopologicalBasis.inseparable_iff {b : Set (Set X)}
    (hb : IsTopologicalBasis b) {x y : X} : Inseparable x y ↔ ∀ s ∈ b, (x ∈ s ↔ y ∈ s) :=
  ⟨fun h _ hs ↦ inseparable_iff_forall_isOpen.1 h _ (hb.isOpen hs),
    fun h ↦ hb.nhds_hasBasis.eq_of_same_basis <| by
      /-
        X : Type u_1
        inst✝ : TopologicalSpace X
        b : Set (Set X)
        hb : TopologicalSpace.IsTopologicalBasis b
        x y : X
        h : ∀ (s : Set X), Membership.mem b s → Iff (Membership.mem s x) (Membership.m …
        ⊢ (nhds y).HasBasis (fun t => And (Membership.mem b t) (Membership.mem t x)) f …
      -/
      convert hb.nhds_hasBasis using 2
      /-
        case h.e'_4.h.a
        X : Type u_1
        inst✝ : TopologicalSpace X
        b : Set (Set X)
        hb : TopologicalSpace.IsTopologicalBasis b
        x y : X
        h : ∀ (s : Set X), Membership.mem b s → Iff (Membership.mem s x) (Membership.m …
        x✝ : Set X
        ⊢ Iff (And (Membership.mem b x✝) (Membership.mem x✝ x)) (And (Membership.mem b …
      -/
      exact and_congr_right (h _)⟩
      /-
        🎉 no goals
      -/


theorem TopologicalSpace.IsTopologicalBasis.eq_iff [T0Space X] {b : Set (Set X)}
    (hb : IsTopologicalBasis b) {x y : X} : x = y ↔ ∀ s ∈ b, (x ∈ s ↔ y ∈ s) :=
  inseparable_iff_eq.symm.trans hb.inseparable_iff


theorem t0Space_iff_exists_isOpen_xor'_mem (X : Type u) [TopologicalSpace X] :
    T0Space X ↔ Pairwise fun x y => ∃ U : Set X, IsOpen U ∧ Xor' (x ∈ U) (y ∈ U) := by
  simp only [t0Space_iff_not_inseparable, xor_iff_not_iff, not_forall, exists_prop,
    inseparable_iff_forall_isOpen, Pairwise]


theorem exists_isOpen_xor'_mem [T0Space X] {x y : X} (h : x ≠ y) :
    ∃ U : Set X, IsOpen U ∧ Xor' (x ∈ U) (y ∈ U) :=
  (t0Space_iff_exists_isOpen_xor'_mem X).1 ‹_› h


/-- Specialization forms a partial order on a t0 topological space. -/
def specializationOrder (X) [TopologicalSpace X] [T0Space X] : PartialOrder X :=
  { specializationPreorder X, PartialOrder.lift (OrderDual.toDual ∘ 𝓝) nhds_injective with }


instance SeparationQuotient.instT0Space : T0Space (SeparationQuotient X) :=
  ⟨fun x y => Quotient.inductionOn₂' x y fun _ _ h =>
    SeparationQuotient.mk_eq_mk.2 <| SeparationQuotient.isInducing_mk.inseparable_iff.1 h⟩


theorem minimal_nonempty_closed_subsingleton [T0Space X] {s : Set X} (hs : IsClosed s)
    (hmin : ∀ t, t ⊆ s → t.Nonempty → IsClosed t → t = s) : s.Subsingleton := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsClosed s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
    ⊢ s.Subsingleton
  -/
  refine fun x hx y hy => of_not_not fun hxy => ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsClosed s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    ⊢ False
  -/
  rcases exists_isOpen_xor'_mem hxy with ⟨U, hUo, hU⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsClosed s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    ⊢ False
  -/
  wlog h : x ∈ U ∧ y ∉ U
    /-
      case intro.intro.inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Set X
      hs : IsClosed s
      hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
      x : X
      hx : Membership.mem s x
      y : X
      hy : Membership.mem s y
      hxy : Not (Eq x y)
      U : Set X
      hUo : IsOpen U
      hU : Xor' (Membership.mem U x) (Membership.mem U y)
      this : ∀ {X : Type u_1} [inst : TopologicalSpace X] [inst_1 : T0Space X] {s :  …
      h : Not (And (Membership.mem U x) (Not (Membership.mem U y)))
      ⊢ False
    -/
  · refine this hs hmin y hy x hx (Ne.symm hxy) U hUo hU.symm (hU.resolve_left h)
    /-
      🎉 no goals
    -/
  /-
    X✝ : Type u_1
    inst✝² : TopologicalSpace X✝
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsClosed s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    h : And (Membership.mem U x) (Not (Membership.mem U y))
    ⊢ False
  -/
  cases' h with hxU hyU
  /-
    case intro
    X✝ : Type u_1
    inst✝² : TopologicalSpace X✝
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsClosed s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    hxU : Membership.mem U x
    hyU : Not (Membership.mem U y)
    ⊢ False
  -/
  have : s \ U = s := hmin (s \ U) diff_subset ⟨y, hy, hyU⟩ (hs.sdiff hUo)
  /-
    case intro
    X✝ : Type u_1
    inst✝² : TopologicalSpace X✝
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsClosed s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsClosed t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    hxU : Membership.mem U x
    hyU : Not (Membership.mem U y)
    this : Eq (SDiff.sdiff s U) s
    ⊢ False
  -/
  exact (this.symm.subset hx).2 hxU
  /-
    🎉 no goals
  -/


theorem minimal_nonempty_closed_eq_singleton [T0Space X] {s : Set X} (hs : IsClosed s)
    (hne : s.Nonempty) (hmin : ∀ t, t ⊆ s → t.Nonempty → IsClosed t → t = s) : ∃ x, s = {x} :=
  exists_eq_singleton_iff_nonempty_subsingleton.2
    ⟨hne, minimal_nonempty_closed_subsingleton hs hmin⟩


/-- Given a closed set `S` in a compact T₀ space, there is some `x ∈ S` such that `{x}` is
closed. -/
theorem IsClosed.exists_closed_singleton [T0Space X] [CompactSpace X] {S : Set X}
    (hS : IsClosed S) (hne : S.Nonempty) : ∃ x : X, x ∈ S ∧ IsClosed ({x} : Set X) := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T0Space X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    ⊢ Exists fun x => And (Membership.mem S x) (IsClosed (Singleton.singleton x))
  -/
  obtain ⟨V, Vsub, Vne, Vcls, hV⟩ := hS.exists_minimal_nonempty_closed_subset hne
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T0Space X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    V : Set X
    Vsub : HasSubset.Subset V S
    Vne : V.Nonempty
    Vcls : IsClosed V
    hV : ∀ (V' : Set X), HasSubset.Subset V' V → V'.Nonempty → IsClosed V' → Eq V' V
    ⊢ Exists fun x => And (Membership.mem S x) (IsClosed (Singleton.singleton x))
  -/
  rcases minimal_nonempty_closed_eq_singleton Vcls Vne hV with ⟨x, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T0Space X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    x : X
    Vsub : HasSubset.Subset (Singleton.singleton x) S
    Vne : (Singleton.singleton x).Nonempty
    Vcls : IsClosed (Singleton.singleton x)
    hV : ∀ (V' : Set X), HasSubset.Subset V' (Singleton.singleton x) → V'.Nonempty …
    ⊢ Exists fun x => And (Membership.mem S x) (IsClosed (Singleton.singleton x))
  -/
  exact ⟨x, Vsub (mem_singleton x), Vcls⟩
  /-
    🎉 no goals
  -/


theorem minimal_nonempty_open_subsingleton [T0Space X] {s : Set X} (hs : IsOpen s)
    (hmin : ∀ t, t ⊆ s → t.Nonempty → IsOpen t → t = s) : s.Subsingleton := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsOpen s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
    ⊢ s.Subsingleton
  -/
  refine fun x hx y hy => of_not_not fun hxy => ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsOpen s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    ⊢ False
  -/
  rcases exists_isOpen_xor'_mem hxy with ⟨U, hUo, hU⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsOpen s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    ⊢ False
  -/
  wlog h : x ∈ U ∧ y ∉ U
    /-
      case intro.intro.inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Set X
      hs : IsOpen s
      hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
      x : X
      hx : Membership.mem s x
      y : X
      hy : Membership.mem s y
      hxy : Not (Eq x y)
      U : Set X
      hUo : IsOpen U
      hU : Xor' (Membership.mem U x) (Membership.mem U y)
      this : ∀ {X : Type u_1} [inst : TopologicalSpace X] [inst_1 : T0Space X] {s :  …
      h : Not (And (Membership.mem U x) (Not (Membership.mem U y)))
      ⊢ False
    -/
  · exact this hs hmin y hy x hx (Ne.symm hxy) U hUo hU.symm (hU.resolve_left h)
    /-
      🎉 no goals
    -/
  /-
    X✝ : Type u_1
    inst✝² : TopologicalSpace X✝
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsOpen s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    h : And (Membership.mem U x) (Not (Membership.mem U y))
    ⊢ False
  -/
  cases' h with hxU hyU
  /-
    case intro
    X✝ : Type u_1
    inst✝² : TopologicalSpace X✝
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsOpen s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    hxU : Membership.mem U x
    hyU : Not (Membership.mem U y)
    ⊢ False
  -/
  have : s ∩ U = s := hmin (s ∩ U) inter_subset_left ⟨x, hx, hxU⟩ (hs.inter hUo)
  /-
    case intro
    X✝ : Type u_1
    inst✝² : TopologicalSpace X✝
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hs : IsOpen s
    hmin : ∀ (t : Set X), HasSubset.Subset t s → t.Nonempty → IsOpen t → Eq t s
    x : X
    hx : Membership.mem s x
    y : X
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    U : Set X
    hUo : IsOpen U
    hU : Xor' (Membership.mem U x) (Membership.mem U y)
    hxU : Membership.mem U x
    hyU : Not (Membership.mem U y)
    this : Eq (Inter.inter s U) s
    ⊢ False
  -/
  exact hyU (this.symm.subset hy).2
  /-
    🎉 no goals
  -/


theorem minimal_nonempty_open_eq_singleton [T0Space X] {s : Set X} (hs : IsOpen s)
    (hne : s.Nonempty) (hmin : ∀ t, t ⊆ s → t.Nonempty → IsOpen t → t = s) : ∃ x, s = {x} :=
  exists_eq_singleton_iff_nonempty_subsingleton.2 ⟨hne, minimal_nonempty_open_subsingleton hs hmin⟩


/-- Given an open finite set `S` in a T₀ space, there is some `x ∈ S` such that `{x}` is open. -/
theorem exists_isOpen_singleton_of_isOpen_finite [T0Space X] {s : Set X} (hfin : s.Finite)
    (hne : s.Nonempty) (ho : IsOpen s) : ∃ x ∈ s, IsOpen ({x} : Set X) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Set X
    hfin : s.Finite
    hne : s.Nonempty
    ho : IsOpen s
    ⊢ Exists fun x => And (Membership.mem s x) (IsOpen (Singleton.singleton x))
  -/
  lift s to Finset X using hfin
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Finset X
    hne : (↑s).Nonempty
    ho : IsOpen ↑s
    ⊢ Exists fun x => And (Membership.mem (↑s) x) (IsOpen (Singleton.singleton x))
  -/
  induction' s using Finset.strongInductionOn with s ihs
  /-
    case intro.a
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    s : Finset X
    ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
    hne : (↑s).Nonempty
    ho : IsOpen ↑s
    ⊢ Exists fun x => And (Membership.mem (↑s) x) (IsOpen (Singleton.singleton x))
  -/
  rcases em (∃ t, t ⊂ s ∧ t.Nonempty ∧ IsOpen (t : Set X)) with (⟨t, hts, htne, hto⟩ | ht)
    /-
      case intro.a.inl.intro.intro.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      t : Finset X
      hts : HasSSubset.SSubset t s
      htne : t.Nonempty
      hto : IsOpen ↑t
      ⊢ Exists fun x => And (Membership.mem (↑s) x) (IsOpen (Singleton.singleton x))
    -/
  · rcases ihs t hts htne hto with ⟨x, hxt, hxo⟩
    /-
      case intro.a.inl.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      t : Finset X
      hts : HasSSubset.SSubset t s
      htne : t.Nonempty
      hto : IsOpen ↑t
      x : X
      hxt : Membership.mem (↑t) x
      hxo : IsOpen (Singleton.singleton x)
      ⊢ Exists fun x => And (Membership.mem (↑s) x) (IsOpen (Singleton.singleton x))
    -/
    exact ⟨x, hts.1 hxt, hxo⟩
    /-
      🎉 no goals
    -/
  · -- Porting note: was `rcases minimal_nonempty_open_eq_singleton ho hne _ with ⟨x, hx⟩`
    --               https://github.com/leanprover/std4/issues/116
    /-
      case intro.a.inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      ht : Not (Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen …
      ⊢ Exists fun x => And (Membership.mem (↑s) x) (IsOpen (Singleton.singleton x))
    -/
    rsuffices ⟨x, hx⟩ : ∃ x, s.toSet = {x}
      /-
        case intro.a.inr.intro
        X : Type u_1
        inst✝¹ : TopologicalSpace X
        inst✝ : T0Space X
        s : Finset X
        ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
        hne : (↑s).Nonempty
        ho : IsOpen ↑s
        ht : Not (Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen …
        x : X
        hx : Eq (↑s) (Singleton.singleton x)
        ⊢ Exists fun x => And (Membership.mem (↑s) x) (IsOpen (Singleton.singleton x))
      -/
    · exact ⟨x, hx.symm ▸ rfl, hx ▸ ho⟩
      /-
        🎉 no goals
      -/
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      ht : Not (Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen …
      ⊢ Exists fun x => Eq (↑s) (Singleton.singleton x)
    -/
    refine minimal_nonempty_open_eq_singleton ho hne ?_
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      ht : Not (Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen …
      ⊢ ∀ (t : Set X), HasSubset.Subset t ↑s → t.Nonempty → IsOpen t → Eq t ↑s
    -/
    refine fun t hts htne hto => of_not_not fun hts' => ht ?_
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      ht : Not (Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen …
      t : Set X
      hts : HasSubset.Subset t ↑s
      htne : t.Nonempty
      hto : IsOpen t
      hts' : Not (Eq t ↑s)
      ⊢ Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen ↑t))
    -/
    lift t to Finset X using s.finite_toSet.subset hts
    /-
      case intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T0Space X
      s : Finset X
      ihs : ∀ (t : Finset X), HasSSubset.SSubset t s → (↑t).Nonempty → IsOpen ↑t → E …
      hne : (↑s).Nonempty
      ho : IsOpen ↑s
      ht : Not (Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen …
      t : Finset X
      hts : HasSubset.Subset ↑t ↑s
      htne : (↑t).Nonempty
      hto : IsOpen ↑t
      hts' : Not (Eq ↑t ↑s)
      ⊢ Exists fun t => And (HasSSubset.SSubset t s) (And t.Nonempty (IsOpen ↑t))
    -/
    exact ⟨t, ssubset_iff_subset_ne.2 ⟨hts, mt Finset.coe_inj.2 hts'⟩, htne, hto⟩
    /-
      🎉 no goals
    -/


theorem exists_open_singleton_of_finite [T0Space X] [Finite X] [Nonempty X] :
    ∃ x : X, IsOpen ({x} : Set X) :=
  let ⟨x, _, h⟩ := exists_isOpen_singleton_of_isOpen_finite (Set.toFinite _)
    univ_nonempty isOpen_univ
  ⟨x, h⟩


theorem t0Space_of_injective_of_continuous [TopologicalSpace Y] {f : X → Y}
    (hf : Function.Injective f) (hf' : Continuous f) [T0Space Y] : T0Space X :=
  ⟨fun _ _ h => hf <| (h.map hf').eq⟩


protected theorem Topology.IsEmbedding.t0Space [TopologicalSpace Y] [T0Space Y] {f : X → Y}
    (hf : IsEmbedding f) : T0Space X :=
  t0Space_of_injective_of_continuous hf.injective hf.continuous


@[deprecated (since := "2024-10-26")]
alias Embedding.t0Space := IsEmbedding.t0Space


instance Subtype.t0Space [T0Space X] {p : X → Prop} : T0Space (Subtype p) :=
  IsEmbedding.subtypeVal.t0Space


theorem t0Space_iff_or_not_mem_closure (X : Type u) [TopologicalSpace X] :
    T0Space X ↔ Pairwise fun a b : X => a ∉ closure ({b} : Set X) ∨ b ∉ closure ({a} : Set X) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ Iff (T0Space X) (Pairwise fun a b => Or (Not (Membership.mem (closure (Singl …
  -/
  simp only [t0Space_iff_not_inseparable, inseparable_iff_mem_closure, not_and_or]
  /-
    🎉 no goals
  -/


instance Prod.instT0Space [TopologicalSpace Y] [T0Space X] [T0Space Y] : T0Space (X × Y) :=
  ⟨fun _ _ h => Prod.ext (h.map continuous_fst).eq (h.map continuous_snd).eq⟩


instance Pi.instT0Space {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)]
    [∀ i, T0Space (X i)] :
    T0Space (∀ i, X i) :=
  ⟨fun _ _ h => funext fun i => (h.map (continuous_apply i)).eq⟩


instance ULift.instT0Space [T0Space X] : T0Space (ULift X) := IsEmbedding.uliftDown.t0Space


theorem T0Space.of_cover (h : ∀ x y, Inseparable x y → ∃ s : Set X, x ∈ s ∧ y ∈ s ∧ T0Space s) :
    T0Space X := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (x y : X), Inseparable x y → Exists fun s => And (Membership.mem s x) (A …
    ⊢ T0Space X
  -/
  refine ⟨fun x y hxy => ?_⟩
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (x y : X), Inseparable x y → Exists fun s => And (Membership.mem s x) (A …
    x y : X
    hxy : Inseparable x y
    ⊢ Eq x y
  -/
  rcases h x y hxy with ⟨s, hxs, hys, hs⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (x y : X), Inseparable x y → Exists fun s => And (Membership.mem s x) (A …
    x y : X
    hxy : Inseparable x y
    s : Set X
    hxs : Membership.mem s x
    hys : Membership.mem s y
    hs : T0Space ↑s
    ⊢ Eq x y
  -/
  lift x to s using hxs; lift y to s using hys
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (x y : X), Inseparable x y → Exists fun s => And (Membership.mem s x) (A …
    s : Set X
    hs : T0Space ↑s
    x y : Subtype fun x => Membership.mem s x
    hxy : Inseparable ↑x ↑y
    ⊢ Eq ↑x ↑y
  -/
  rw [← subtype_inseparable_iff] at hxy
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (x y : X), Inseparable x y → Exists fun s => And (Membership.mem s x) (A …
    s : Set X
    hs : T0Space ↑s
    x y : Subtype fun x => Membership.mem s x
    hxy : Inseparable x y
    ⊢ Eq ↑x ↑y
  -/
  exact congr_arg Subtype.val hxy.eq
  /-
    🎉 no goals
  -/


theorem T0Space.of_open_cover (h : ∀ x, ∃ s : Set X, x ∈ s ∧ IsOpen s ∧ T0Space s) : T0Space X :=
  T0Space.of_cover fun x _ hxy =>
    let ⟨s, hxs, hso, hs⟩ := h x
    ⟨s, hxs, (hxy.mem_open_iff hso).1 hxs, hs⟩


/-- A topological space is called an R₀ space, if `Specializes` relation is symmetric.

In other words, given two points `x y : X`,
if every neighborhood of `y` contains `x`, then every neighborhood of `x` contains `y`. -/
@[mk_iff]
class R0Space (X : Type u) [TopologicalSpace X] : Prop where
  /-- In an R₀ space, the `Specializes` relation is symmetric. -/
  specializes_symmetric : Symmetric (Specializes : X → X → Prop)


/-- In an R₀ space, the `Specializes` relation is symmetric, dot notation version. -/
theorem Specializes.symm (h : x ⤳ y) : y ⤳ x := specializes_symmetric h


/-- In an R₀ space, the `Specializes` relation is symmetric, `Iff` version. -/
theorem specializes_comm : x ⤳ y ↔ y ⤳ x := ⟨Specializes.symm, Specializes.symm⟩


/-- In an R₀ space, `Specializes` is equivalent to `Inseparable`. -/
theorem specializes_iff_inseparable : x ⤳ y ↔ Inseparable x y :=
  ⟨fun h ↦ h.antisymm h.symm, Inseparable.specializes⟩


/-- In an R₀ space, `Specializes` implies `Inseparable`. -/
alias ⟨Specializes.inseparable, _⟩ := specializes_iff_inseparable


theorem Topology.IsInducing.r0Space [TopologicalSpace Y] {f : Y → X} (hf : IsInducing f) :
    R0Space Y where
  specializes_symmetric a b := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : R0Space X
      inst✝ : TopologicalSpace Y
      f : Y → X
      hf : Topology.IsInducing f
      a b : Y
      ⊢ Specializes a b → Specializes b a
    -/
    simpa only [← hf.specializes_iff] using Specializes.symm
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")] alias Inducing.r0Space := IsInducing.r0Space


instance {p : X → Prop} : R0Space {x // p x} := IsInducing.subtypeVal.r0Space


instance [TopologicalSpace Y] [R0Space Y] : R0Space (X × Y) where
  specializes_symmetric _ _ h := h.fst.symm.prod h.snd.symm


instance {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)] [∀ i, R0Space (X i)] :
    R0Space (∀ i, X i) where
  specializes_symmetric _ _ h := specializes_pi.2 fun i ↦ (specializes_pi.1 h i).symm


/-- In an R₀ space, the closure of a singleton is a compact set. -/
theorem isCompact_closure_singleton : IsCompact (closure {x}) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R0Space X
    x : X
    ⊢ IsCompact (closure (Singleton.singleton x))
  -/
  refine isCompact_of_finite_subcover fun U hUo hxU ↦ ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R0Space X
    x : X
    ι✝ : Type u_1
    U : ι✝ → Set X
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hxU : HasSubset.Subset (closure (Singleton.singleton x)) (Set.iUnion fun i =>  …
    ⊢ Exists fun t => HasSubset.Subset (closure (Singleton.singleton x)) (Set.iUni …
  -/
  obtain ⟨i, hi⟩ : ∃ i, x ∈ U i := mem_iUnion.1 <| hxU <| subset_closure rfl
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R0Space X
    x : X
    ι✝ : Type u_1
    U : ι✝ → Set X
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hxU : HasSubset.Subset (closure (Singleton.singleton x)) (Set.iUnion fun i =>  …
    i : ι✝
    hi : Membership.mem (U i) x
    ⊢ Exists fun t => HasSubset.Subset (closure (Singleton.singleton x)) (Set.iUni …
  -/
  refine ⟨{i}, fun y hy ↦ ?_⟩
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R0Space X
    x : X
    ι✝ : Type u_1
    U : ι✝ → Set X
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hxU : HasSubset.Subset (closure (Singleton.singleton x)) (Set.iUnion fun i =>  …
    i : ι✝
    hi : Membership.mem (U i) x
    y : X
    hy : Membership.mem (closure (Singleton.singleton x)) y
    ⊢ Membership.mem (Set.iUnion fun i_1 => Set.iUnion fun h => U i_1) y
  -/
  rw [← specializes_iff_mem_closure, specializes_comm] at hy
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R0Space X
    x : X
    ι✝ : Type u_1
    U : ι✝ → Set X
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hxU : HasSubset.Subset (closure (Singleton.singleton x)) (Set.iUnion fun i =>  …
    i : ι✝
    hi : Membership.mem (U i) x
    y : X
    hy : Specializes y x
    ⊢ Membership.mem (Set.iUnion fun i_1 => Set.iUnion fun h => U i_1) y
  -/
  simpa using hy.mem_open (hUo i) hi
  /-
    🎉 no goals
  -/


theorem Filter.coclosedCompact_le_cofinite : coclosedCompact X ≤ cofinite :=
  le_cofinite_iff_compl_singleton_mem.2 fun _ ↦
    compl_mem_coclosedCompact.2 isCompact_closure_singleton


/-- In an R₀ space, relatively compact sets form a bornology.
Its cobounded filter is `Filter.coclosedCompact`.
See also `Bornology.inCompact` the bornology of sets contained in a compact set. -/
def Bornology.relativelyCompact : Bornology X where
  cobounded' := Filter.coclosedCompact X
  le_cofinite' := Filter.coclosedCompact_le_cofinite


theorem Bornology.relativelyCompact.isBounded_iff {s : Set X} :
    @Bornology.IsBounded _ (Bornology.relativelyCompact X) s ↔ IsCompact (closure s) :=
  compl_mem_coclosedCompact


/-- In an R₀ space, the closure of a finite set is a compact set. -/
theorem Set.Finite.isCompact_closure {s : Set X} (hs : s.Finite) : IsCompact (closure s) :=
  let _ : Bornology X := .relativelyCompact X
  Bornology.relativelyCompact.isBounded_iff.1 hs.isBounded


/-- A T₁ space, also known as a Fréchet space, is a topological space
  where every singleton set is closed. Equivalently, for every pair
  `x ≠ y`, there is an open set containing `x` and not `y`. -/
class T1Space (X : Type u) [TopologicalSpace X] : Prop where
  /-- A singleton in a T₁ space is a closed set. -/
  t1 : ∀ x, IsClosed ({x} : Set X)


theorem isClosed_singleton [T1Space X] {x : X} : IsClosed ({x} : Set X) :=
  T1Space.t1 x


theorem isOpen_compl_singleton [T1Space X] {x : X} : IsOpen ({x}ᶜ : Set X) :=
  isClosed_singleton.isOpen_compl


theorem isOpen_ne [T1Space X] {x : X} : IsOpen { y | y ≠ x } :=
  isOpen_compl_singleton


@[to_additive]
theorem Continuous.isOpen_mulSupport [T1Space X] [One X] [TopologicalSpace Y] {f : Y → X}
    (hf : Continuous f) : IsOpen (mulSupport f) :=
  isOpen_ne.preimage hf


theorem Ne.nhdsWithin_compl_singleton [T1Space X] {x y : X} (h : x ≠ y) : 𝓝[{y}ᶜ] x = 𝓝 x :=
  isOpen_ne.nhdsWithin_eq h


theorem Ne.nhdsWithin_diff_singleton [T1Space X] {x y : X} (h : x ≠ y) (s : Set X) :
    𝓝[s \ {y}] x = 𝓝[s] x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    h : Ne x y
    s : Set X
    ⊢ Eq (nhdsWithin x (SDiff.sdiff s (Singleton.singleton y))) (nhdsWithin x s)
  -/
  rw [diff_eq, inter_comm, nhdsWithin_inter_of_mem]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    h : Ne x y
    s : Set X
    ⊢ Membership.mem (nhdsWithin x s) (HasCompl.compl (Singleton.singleton y))
  -/
  exact mem_nhdsWithin_of_mem_nhds (isOpen_ne.mem_nhds h)
  /-
    🎉 no goals
  -/


lemma nhdsWithin_compl_singleton_le [T1Space X] (x y : X) : 𝓝[{x}ᶜ] x ≤ 𝓝[{y}ᶜ] x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    ⊢ LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (nhdsWithin x  …
  -/
  rcases eq_or_ne x y with rfl|hy
    /-
      case inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      x : X
      ⊢ LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (nhdsWithin x  …
    -/
  · exact Eq.le rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      x y : X
      hy : Ne x y
      ⊢ LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (nhdsWithin x  …
    -/
  · rw [Ne.nhdsWithin_compl_singleton hy]
    /-
      case inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      x y : X
      hy : Ne x y
      ⊢ LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (nhds x)
    -/
    exact nhdsWithin_le_nhds
    /-
      🎉 no goals
    -/


theorem isOpen_setOf_eventually_nhdsWithin [T1Space X] {p : X → Prop} :
    IsOpen { x | ∀ᶠ y in 𝓝[≠] x, p y } := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    p : X → Prop
    ⊢ IsOpen (setOf fun x => Filter.Eventually (fun y => p y) (nhdsWithin x (HasCo …
  -/
  refine isOpen_iff_mem_nhds.mpr fun a ha => ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    p : X → Prop
    a : X
    ha : Membership.mem (setOf fun x => Filter.Eventually (fun y => p y) (nhdsWith …
    ⊢ Membership.mem (nhds a) (setOf fun x => Filter.Eventually (fun y => p y) (nh …
  -/
  filter_upwards [eventually_nhds_nhdsWithin.mpr ha] with b hb
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    p : X → Prop
    a : X
    ha : Membership.mem (setOf fun x => Filter.Eventually (fun y => p y) (nhdsWith …
    b : X
    hb : Filter.Eventually (fun x => p x) (nhdsWithin b (HasCompl.compl (Singleton …
    ⊢ Filter.Eventually (fun y => p y) (nhdsWithin b (HasCompl.compl (Singleton.si …
  -/
  rcases eq_or_ne a b with rfl | h
    /-
      case h.inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      p : X → Prop
      a : X
      ha : Membership.mem (setOf fun x => Filter.Eventually (fun y => p y) (nhdsWith …
      hb : Filter.Eventually (fun x => p x) (nhdsWithin a (HasCompl.compl (Singleton …
      ⊢ Filter.Eventually (fun y => p y) (nhdsWithin a (HasCompl.compl (Singleton.si …
    -/
  · exact hb
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      p : X → Prop
      a : X
      ha : Membership.mem (setOf fun x => Filter.Eventually (fun y => p y) (nhdsWith …
      b : X
      hb : Filter.Eventually (fun x => p x) (nhdsWithin b (HasCompl.compl (Singleton …
      h : Ne a b
      ⊢ Filter.Eventually (fun y => p y) (nhdsWithin b (HasCompl.compl (Singleton.si …
    -/
  · rw [h.symm.nhdsWithin_compl_singleton] at hb
    /-
      case h.inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      p : X → Prop
      a : X
      ha : Membership.mem (setOf fun x => Filter.Eventually (fun y => p y) (nhdsWith …
      b : X
      hb : Filter.Eventually (fun x => p x) (nhds b)
      h : Ne a b
      ⊢ Filter.Eventually (fun y => p y) (nhdsWithin b (HasCompl.compl (Singleton.si …
    -/
    exact hb.filter_mono nhdsWithin_le_nhds
    /-
      🎉 no goals
    -/


protected theorem Set.Finite.isClosed [T1Space X] {s : Set X} (hs : Set.Finite s) : IsClosed s := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    hs : s.Finite
    ⊢ IsClosed s
  -/
  rw [← biUnion_of_singleton s]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    hs : s.Finite
    ⊢ IsClosed (Set.iUnion fun x => Set.iUnion fun h => Singleton.singleton x)
  -/
  exact hs.isClosed_biUnion fun i _ => isClosed_singleton
  /-
    🎉 no goals
  -/


theorem TopologicalSpace.IsTopologicalBasis.exists_mem_of_ne [T1Space X] {b : Set (Set X)}
    (hb : IsTopologicalBasis b) {x y : X} (h : x ≠ y) : ∃ a ∈ b, x ∈ a ∧ y ∉ a := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis b
    x y : X
    h : Ne x y
    ⊢ Exists fun a => And (Membership.mem b a) (And (Membership.mem a x) (Not (Mem …
  -/
  rcases hb.isOpen_iff.1 isOpen_ne x h with ⟨a, ab, xa, ha⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis b
    x y : X
    h : Ne x y
    a : Set X
    ab : Membership.mem b a
    xa : Membership.mem a x
    ha : HasSubset.Subset a (setOf fun y_1 => Ne y_1 y)
    ⊢ Exists fun a => And (Membership.mem b a) (And (Membership.mem a x) (Not (Mem …
  -/
  exact ⟨a, ab, xa, fun h => ha h rfl⟩
  /-
    🎉 no goals
  -/


protected theorem Finset.isClosed [T1Space X] (s : Finset X) : IsClosed (s : Set X) :=
  s.finite_toSet.isClosed


theorem t1Space_TFAE (X : Type u) [TopologicalSpace X] :
    List.TFAE [T1Space X,
      ∀ x, IsClosed ({ x } : Set X),
      ∀ x, IsOpen ({ x }ᶜ : Set X),
      Continuous (@CofiniteTopology.of X),
      ∀ ⦃x y : X⦄, x ≠ y → {y}ᶜ ∈ 𝓝 x,
      ∀ ⦃x y : X⦄, x ≠ y → ∃ s ∈ 𝓝 x, y ∉ s,
      ∀ ⦃x y : X⦄, x ≠ y → ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ y ∉ U,
      ∀ ⦃x y : X⦄, x ≠ y → Disjoint (𝓝 x) (pure y),
      ∀ ⦃x y : X⦄, x ≠ y → Disjoint (pure x) (𝓝 y),
      ∀ ⦃x y : X⦄, x ⤳ y → x = y] := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ (List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.singleton  …
  -/
  tfae_have 1 ↔ 2 := ⟨fun h => h.1, fun h => ⟨h⟩⟩
  tfae_have 2 ↔ 3 := by
    simp only [isOpen_compl_iff]
  tfae_have 5 ↔ 3 := by
    refine forall_swap.trans ?_
    simp only [isOpen_iff_mem_nhds, mem_compl_iff, mem_singleton_iff]
  tfae_have 5 ↔ 6 := by
    simp only [← subset_compl_singleton_iff, exists_mem_subset_iff]
  tfae_have 5 ↔ 7 := by
    simp only [(nhds_basis_opens _).mem_iff, subset_compl_singleton_iff, exists_prop, and_assoc,
      and_left_comm]
  tfae_have 5 ↔ 8 := by
    simp only [← principal_singleton, disjoint_principal_right]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    tfae_1_iff_2 : Iff (T1Space X) (∀ (x : X), IsClosed (Singleton.singleton x))
    tfae_2_iff_3 : Iff (∀ (x : X), IsClosed (Singleton.singleton x)) (∀ (x : X), I …
    tfae_5_iff_3 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_5_iff_6 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_5_iff_7 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_5_iff_8 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    ⊢ (List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.singleton  …
  -/
  tfae_have 8 ↔ 9 := forall_swap.trans (by simp only [disjoint_comm, ne_comm])
  tfae_have 1 → 4 := by
    simp only [continuous_def, CofiniteTopology.isOpen_iff']
    rintro H s (rfl | hs)
    exacts [isOpen_empty, compl_compl s ▸ (@Set.Finite.isClosed _ _ H _ hs).isOpen_compl]
  tfae_have 4 → 2 :=
    fun h x => (CofiniteTopology.isClosed_iff.2 <| Or.inr (finite_singleton _)).preimage h
  tfae_have 2 ↔ 10 := by
    simp only [← closure_subset_iff_isClosed, specializes_iff_mem_closure, subset_def,
      mem_singleton_iff, eq_comm]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    tfae_1_iff_2 : Iff (T1Space X) (∀ (x : X), IsClosed (Singleton.singleton x))
    tfae_2_iff_3 : Iff (∀ (x : X), IsClosed (Singleton.singleton x)) (∀ (x : X), I …
    tfae_5_iff_3 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_5_iff_6 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_5_iff_7 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_5_iff_8 : Iff (∀ ⦃x y : X⦄, Ne x y → Membership.mem (nhds x) (HasCompl.co …
    tfae_8_iff_9 : Iff (∀ ⦃x y : X⦄, Ne x y → Disjoint (nhds x) (Pure.pure y)) (∀  …
    tfae_1_to_4 : T1Space X → Continuous ⇑CofiniteTopology.of
    tfae_4_to_2 : Continuous ⇑CofiniteTopology.of → ∀ (x : X), IsClosed (Singleton …
    tfae_2_iff_10 : Iff (∀ (x : X), IsClosed (Singleton.singleton x)) (∀ ⦃x y : X⦄ …
    ⊢ (List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.singleton  …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem t1Space_iff_continuous_cofinite_of : T1Space X ↔ Continuous (@CofiniteTopology.of X) :=
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq ((List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.single …
  -/
  /-
    🎉 no goals
  -/
  (t1Space_TFAE X).out 0 3
  /-
    🎉 no goals
  -/


theorem CofiniteTopology.continuous_of [T1Space X] : Continuous (@CofiniteTopology.of X) :=
  t1Space_iff_continuous_cofinite_of.mp ‹_›


theorem t1Space_iff_exists_open :
    T1Space X ↔ Pairwise fun x y => ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ y ∉ U :=
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq ((List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.single …
  -/
  /-
    🎉 no goals
  -/
  (t1Space_TFAE X).out 0 6
  /-
    🎉 no goals
  -/


theorem t1Space_iff_disjoint_pure_nhds : T1Space X ↔ ∀ ⦃x y : X⦄, x ≠ y → Disjoint (pure x) (𝓝 y) :=
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq ((List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.single …
  -/
  /-
    🎉 no goals
  -/
  (t1Space_TFAE X).out 0 8
  /-
    🎉 no goals
  -/


theorem t1Space_iff_disjoint_nhds_pure : T1Space X ↔ ∀ ⦃x y : X⦄, x ≠ y → Disjoint (𝓝 x) (pure y) :=
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq ((List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.single …
  -/
  /-
    🎉 no goals
  -/
  (t1Space_TFAE X).out 0 7
  /-
    🎉 no goals
  -/


theorem t1Space_iff_specializes_imp_eq : T1Space X ↔ ∀ ⦃x y : X⦄, x ⤳ y → x = y :=
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq ((List.cons (T1Space X) (List.cons (∀ (x : X), IsClosed (Singleton.single …
  -/
  /-
    🎉 no goals
  -/
  (t1Space_TFAE X).out 0 9
  /-
    🎉 no goals
  -/


theorem disjoint_pure_nhds [T1Space X] {x y : X} (h : x ≠ y) : Disjoint (pure x) (𝓝 y) :=
  t1Space_iff_disjoint_pure_nhds.mp ‹_› h


theorem disjoint_nhds_pure [T1Space X] {x y : X} (h : x ≠ y) : Disjoint (𝓝 x) (pure y) :=
  t1Space_iff_disjoint_nhds_pure.mp ‹_› h


theorem Specializes.eq [T1Space X] {x y : X} (h : x ⤳ y) : x = y :=
  t1Space_iff_specializes_imp_eq.1 ‹_› h


theorem specializes_iff_eq [T1Space X] {x y : X} : x ⤳ y ↔ x = y :=
  ⟨Specializes.eq, fun h => h ▸ specializes_rfl⟩


@[simp] theorem specializes_eq_eq [T1Space X] : (· ⤳ ·) = @Eq X :=
  funext₂ fun _ _ => propext specializes_iff_eq


@[simp]
theorem pure_le_nhds_iff [T1Space X] {a b : X} : pure a ≤ 𝓝 b ↔ a = b :=
  specializes_iff_pure.symm.trans specializes_iff_eq


@[simp]
theorem nhds_le_nhds_iff [T1Space X] {a b : X} : 𝓝 a ≤ 𝓝 b ↔ a = b :=
  specializes_iff_eq


instance (priority := 100) [T1Space X] : R0Space X where
                                  /-
                                    X : Type u_1
                                    Y : Type u_2
                                    inst✝¹ : TopologicalSpace X
                                    inst✝ : T1Space X
                                    x✝¹ x✝ : X
                                    ⊢ Specializes x✝¹ x✝ → Specializes x✝ x✝¹
                                  -/
  specializes_symmetric _ _ := by rw [specializes_iff_eq, specializes_iff_eq]; exact Eq.symm
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


instance : T1Space (CofiniteTopology X) :=
  t1Space_iff_continuous_cofinite_of.mpr continuous_id


theorem t1Space_antitone {X} : Antitone (@T1Space X) := fun a _ h _ =>
  @T1Space.mk _ a fun x => (T1Space.t1 x).mono h


theorem continuousWithinAt_update_of_ne [T1Space X] [DecidableEq X] [TopologicalSpace Y] {f : X → Y}
    {s : Set X} {x x' : X} {y : Y} (hne : x' ≠ x) :
    ContinuousWithinAt (Function.update f x y) s x' ↔ ContinuousWithinAt f s x' :=
  EventuallyEq.congr_continuousWithinAt
    (mem_nhdsWithin_of_mem_nhds <| mem_of_superset (isOpen_ne.mem_nhds hne) fun _y' hy' =>
      Function.update_of_ne hy' _ _)
    (Function.update_of_ne hne ..)


theorem continuousAt_update_of_ne [T1Space X] [DecidableEq X] [TopologicalSpace Y]
    {f : X → Y} {x x' : X} {y : Y} (hne : x' ≠ x) :
    ContinuousAt (Function.update f x y) x' ↔ ContinuousAt f x' := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : T1Space X
    inst✝¹ : DecidableEq X
    inst✝ : TopologicalSpace Y
    f : X → Y
    x x' : X
    y : Y
    hne : Ne x' x
    ⊢ Iff (ContinuousAt (Function.update f x y) x') (ContinuousAt f x')
  -/
  simp only [← continuousWithinAt_univ, continuousWithinAt_update_of_ne hne]
  /-
    🎉 no goals
  -/


theorem continuousOn_update_iff [T1Space X] [DecidableEq X] [TopologicalSpace Y] {f : X → Y}
    {s : Set X} {x : X} {y : Y} :
    ContinuousOn (Function.update f x y) s ↔
      ContinuousOn f (s \ {x}) ∧ (x ∈ s → Tendsto f (𝓝[s \ {x}] x) (𝓝 y)) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : T1Space X
    inst✝¹ : DecidableEq X
    inst✝ : TopologicalSpace Y
    f : X → Y
    s : Set X
    x : X
    y : Y
    ⊢ Iff (ContinuousOn (Function.update f x y) s) (And (ContinuousOn f (SDiff.sdi …
  -/
  rw [ContinuousOn, ← and_forall_ne x, and_comm]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : T1Space X
    inst✝¹ : DecidableEq X
    inst✝ : TopologicalSpace Y
    f : X → Y
    s : Set X
    x : X
    y : Y
    ⊢ Iff (And (∀ (b : X), Ne b x → Membership.mem s b → ContinuousWithinAt (Funct …
  -/
  refine and_congr ⟨fun H z hz => ?_, fun H z hzx hzs => ?_⟩ (forall_congr' fun _ => ?_)
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      H : ∀ (b : X), Ne b x → Membership.mem s b → ContinuousWithinAt (Function.upda …
      z : X
      hz : Membership.mem (SDiff.sdiff s (Singleton.singleton x)) z
      ⊢ ContinuousWithinAt f (SDiff.sdiff s (Singleton.singleton x)) z
    -/
  · specialize H z hz.2 hz.1
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      z : X
      hz : Membership.mem (SDiff.sdiff s (Singleton.singleton x)) z
      H : ContinuousWithinAt (Function.update f x y) s z
      ⊢ ContinuousWithinAt f (SDiff.sdiff s (Singleton.singleton x)) z
    -/
    rw [continuousWithinAt_update_of_ne hz.2] at H
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      z : X
      hz : Membership.mem (SDiff.sdiff s (Singleton.singleton x)) z
      H : ContinuousWithinAt f s z
      ⊢ ContinuousWithinAt f (SDiff.sdiff s (Singleton.singleton x)) z
    -/
    exact H.mono diff_subset
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      H : ContinuousOn f (SDiff.sdiff s (Singleton.singleton x))
      z : X
      hzx : Ne z x
      hzs : Membership.mem s z
      ⊢ ContinuousWithinAt (Function.update f x y) s z
    -/
  · rw [continuousWithinAt_update_of_ne hzx]
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      H : ContinuousOn f (SDiff.sdiff s (Singleton.singleton x))
      z : X
      hzx : Ne z x
      hzs : Membership.mem s z
      ⊢ ContinuousWithinAt f s z
    -/
    refine (H z ⟨hzs, hzx⟩).mono_of_mem_nhdsWithin (inter_mem_nhdsWithin _ ?_)
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      H : ContinuousOn f (SDiff.sdiff s (Singleton.singleton x))
      z : X
      hzx : Ne z x
      hzs : Membership.mem s z
      ⊢ Membership.mem (nhds z) fun a => Membership.mem (Singleton.singleton x) a →  …
    -/
    exact isOpen_ne.mem_nhds hzx
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : T1Space X
      inst✝¹ : DecidableEq X
      inst✝ : TopologicalSpace Y
      f : X → Y
      s : Set X
      x : X
      y : Y
      x✝ : Membership.mem s x
      ⊢ Iff (ContinuousWithinAt (Function.update f x y) s x) (Filter.Tendsto f (nhds …
    -/
  · exact continuousWithinAt_update_same
    /-
      🎉 no goals
    -/


theorem t1Space_of_injective_of_continuous [TopologicalSpace Y] {f : X → Y}
    (hf : Function.Injective f) (hf' : Continuous f) [T1Space Y] : T1Space X :=
  t1Space_iff_specializes_imp_eq.2 fun _ _ h => hf (h.map hf').eq


protected theorem Topology.IsEmbedding.t1Space [TopologicalSpace Y] [T1Space Y] {f : X → Y}
    (hf : IsEmbedding f) : T1Space X :=
  t1Space_of_injective_of_continuous hf.injective hf.continuous


@[deprecated (since := "2024-10-26")]
alias Embedding.t1Space := IsEmbedding.t1Space


instance Subtype.t1Space {X : Type u} [TopologicalSpace X] [T1Space X] {p : X → Prop} :
    T1Space (Subtype p) :=
  IsEmbedding.subtypeVal.t1Space


instance [TopologicalSpace Y] [T1Space X] [T1Space Y] : T1Space (X × Y) :=
  ⟨fun ⟨a, b⟩ => @singleton_prod_singleton _ _ a b ▸ isClosed_singleton.prod isClosed_singleton⟩


instance {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)] [∀ i, T1Space (X i)] :
    T1Space (∀ i, X i) :=
  ⟨fun f => univ_pi_singleton f ▸ isClosed_set_pi fun _ _ => isClosed_singleton⟩


instance ULift.instT1Space [T1Space X] : T1Space (ULift X) :=
  IsEmbedding.uliftDown.t1Space

-- see Note [lower instance priority]

instance (priority := 100) TotallyDisconnectedSpace.t1Space [h : TotallyDisconnectedSpace X] :
    T1Space X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    h : TotallyDisconnectedSpace X
    ⊢ T1Space X
  -/
  rw [((t1Space_TFAE X).out 0 1 :)]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    h : TotallyDisconnectedSpace X
    ⊢ ∀ (x : X), IsClosed (Singleton.singleton x)
  -/
  intro x
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    h : TotallyDisconnectedSpace X
    x : X
    ⊢ IsClosed (Singleton.singleton x)
  -/
  rw [← totallyDisconnectedSpace_iff_connectedComponent_singleton.mp h x]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    h : TotallyDisconnectedSpace X
    x : X
    ⊢ IsClosed (connectedComponent x)
  -/
  exact isClosed_connectedComponent
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) T1Space.t0Space [T1Space X] : T0Space X :=
  ⟨fun _ _ h => h.specializes.eq⟩


@[simp]
theorem compl_singleton_mem_nhds_iff [T1Space X] {x y : X} : {x}ᶜ ∈ 𝓝 y ↔ y ≠ x :=
  isOpen_compl_singleton.mem_nhds_iff


theorem compl_singleton_mem_nhds [T1Space X] {x y : X} (h : y ≠ x) : {x}ᶜ ∈ 𝓝 y :=
  compl_singleton_mem_nhds_iff.mpr h


@[simp]
theorem closure_singleton [T1Space X] {x : X} : closure ({x} : Set X) = {x} :=
  isClosed_singleton.closure_eq

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: the proof was `hs.induction_on (by simp) fun x => by simp`

theorem Set.Subsingleton.closure [T1Space X] {s : Set X} (hs : s.Subsingleton) :
    (closure s).Subsingleton := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    hs : s.Subsingleton
    ⊢ (_root_.closure s).Subsingleton
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  rcases hs.eq_empty_or_singleton with (rfl | ⟨x, rfl⟩) <;> simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem subsingleton_closure [T1Space X] {s : Set X} : (closure s).Subsingleton ↔ s.Subsingleton :=
  ⟨fun h => h.anti subset_closure, fun h => h.closure⟩


theorem isClosedMap_const {X Y} [TopologicalSpace X] [TopologicalSpace Y] [T1Space Y] {y : Y} :
    IsClosedMap (Function.const X y) :=
                                            /-
                                              X : Type u_3
                                              Y : Type u_4
                                              inst✝² : TopologicalSpace X
                                              inst✝¹ : TopologicalSpace Y
                                              inst✝ : T1Space Y
                                              y : Y
                                              s : Set X
                                              x✝ : IsClosed s
                                              h2s : s.Nonempty
                                              ⊢ IsClosed (Set.image (Function.const X y) s)
                                            -/
  IsClosedMap.of_nonempty fun s _ h2s => by simp_rw [const, h2s.image_const, isClosed_singleton]
                                            /-
                                              🎉 no goals
                                            -/


theorem nhdsWithin_insert_of_ne [T1Space X] {x y : X} {s : Set X} (hxy : x ≠ y) :
    𝓝[insert y s] x = 𝓝[s] x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s : Set X
    hxy : Ne x y
    ⊢ Eq (nhdsWithin x (Insert.insert y s)) (nhdsWithin x s)
  -/
  refine le_antisymm (Filter.le_def.2 fun t ht => ?_) (nhdsWithin_mono x <| subset_insert y s)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s : Set X
    hxy : Ne x y
    t : Set X
    ht : Membership.mem (nhdsWithin x s) t
    ⊢ Membership.mem (nhdsWithin x (Insert.insert y s)) t
  -/
  obtain ⟨o, ho, hxo, host⟩ := mem_nhdsWithin.mp ht
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s : Set X
    hxy : Ne x y
    t : Set X
    ht : Membership.mem (nhdsWithin x s) t
    o : Set X
    ho : IsOpen o
    hxo : Membership.mem o x
    host : HasSubset.Subset (Inter.inter o s) t
    ⊢ Membership.mem (nhdsWithin x (Insert.insert y s)) t
  -/
  refine mem_nhdsWithin.mpr ⟨o \ {y}, ho.sdiff isClosed_singleton, ⟨hxo, hxy⟩, ?_⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s : Set X
    hxy : Ne x y
    t : Set X
    ht : Membership.mem (nhdsWithin x s) t
    o : Set X
    ho : IsOpen o
    hxo : Membership.mem o x
    host : HasSubset.Subset (Inter.inter o s) t
    ⊢ HasSubset.Subset (Inter.inter (SDiff.sdiff o (Singleton.singleton y)) (Inser …
  -/
  rw [inter_insert_of_not_mem <| not_mem_diff_of_mem (mem_singleton y)]
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s : Set X
    hxy : Ne x y
    t : Set X
    ht : Membership.mem (nhdsWithin x s) t
    o : Set X
    ho : IsOpen o
    hxo : Membership.mem o x
    host : HasSubset.Subset (Inter.inter o s) t
    ⊢ HasSubset.Subset (Inter.inter (SDiff.sdiff o (Singleton.singleton y)) s) t
  -/
  exact (inter_subset_inter diff_subset Subset.rfl).trans host
  /-
    🎉 no goals
  -/


/-- If `t` is a subset of `s`, except for one point,
then `insert x s` is a neighborhood of `x` within `t`. -/
theorem insert_mem_nhdsWithin_of_subset_insert [T1Space X] {x y : X} {s t : Set X}
    (hu : t ⊆ insert y s) : insert x s ∈ 𝓝[t] x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s t : Set X
    hu : HasSubset.Subset t (Insert.insert y s)
    ⊢ Membership.mem (nhdsWithin x t) (Insert.insert x s)
  -/
  rcases eq_or_ne x y with (rfl | h)
    /-
      case inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      x : X
      s t : Set X
      hu : HasSubset.Subset t (Insert.insert x s)
      ⊢ Membership.mem (nhdsWithin x t) (Insert.insert x s)
    -/
  · exact mem_of_superset self_mem_nhdsWithin hu
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s t : Set X
    hu : HasSubset.Subset t (Insert.insert y s)
    h : Ne x y
    ⊢ Membership.mem (nhdsWithin x t) (Insert.insert x s)
  -/
  refine nhdsWithin_mono x hu ?_
  /-
    case inr
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s t : Set X
    hu : HasSubset.Subset t (Insert.insert y s)
    h : Ne x y
    ⊢ Membership.mem (nhdsWithin x (Insert.insert y s)) (Insert.insert x s)
  -/
  rw [nhdsWithin_insert_of_ne h]
  /-
    case inr
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x y : X
    s t : Set X
    hu : HasSubset.Subset t (Insert.insert y s)
    h : Ne x y
    ⊢ Membership.mem (nhdsWithin x s) (Insert.insert x s)
  -/
  exact mem_of_superset self_mem_nhdsWithin (subset_insert x s)
  /-
    🎉 no goals
  -/


lemma eventuallyEq_insert [T1Space X] {s t : Set X} {x y : X} (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    (insert x s : Set X) =ᶠ[𝓝 x] (insert x t : Set X) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    x y : X
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ (nhds x).EventuallyEq (Insert.insert x s) (Insert.insert x t)
  -/
  simp_rw [eventuallyEq_set] at h ⊢
  simp_rw [← union_singleton, ← nhdsWithin_univ, ← compl_union_self {x},
    nhdsWithin_union, eventually_sup, nhdsWithin_singleton,
    eventually_pure, union_singleton, mem_insert_iff, true_or, and_true]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    x y : X
    h : Filter.Eventually (fun x => Iff (Membership.mem s x) (Membership.mem t x)) …
    ⊢ Filter.Eventually (fun x_1 => Iff (Or (Eq x_1 x) (Membership.mem s x_1)) (Or …
  -/
  filter_upwards [nhdsWithin_compl_singleton_le x y h] with y using or_congr (Iff.rfl)
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_nhds [T1Space X] (x : X) : (𝓝 x).ker = {x} := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    ⊢ Eq (nhds x).ker (Singleton.singleton x)
  -/
  simp [ker_nhds_eq_specializes]
  /-
    🎉 no goals
  -/


theorem biInter_basis_nhds [T1Space X] {ι : Sort*} {p : ι → Prop} {s : ι → Set X} {x : X}
    (h : (𝓝 x).HasBasis p s) : ⋂ (i) (_ : p i), s i = {x} := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    ι : Sort u_3
    p : ι → Prop
    s : ι → Set X
    x : X
    h : (nhds x).HasBasis p s
    ⊢ Eq (Set.iInter fun i => Set.iInter fun x => s i) (Singleton.singleton x)
  -/
  rw [← h.ker, ker_nhds]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_singleton_mem_nhdsSet_iff [T1Space X] {x : X} {s : Set X} : {x}ᶜ ∈ 𝓝ˢ s ↔ x ∉ s := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    s : Set X
    ⊢ Iff (Membership.mem (nhdsSet s) (HasCompl.compl (Singleton.singleton x))) (N …
  -/
  rw [isOpen_compl_singleton.mem_nhdsSet, subset_compl_singleton_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem nhdsSet_le_iff [T1Space X] {s t : Set X} : 𝓝ˢ s ≤ 𝓝ˢ t ↔ s ⊆ t := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    ⊢ Iff (LE.le (nhdsSet s) (nhdsSet t)) (HasSubset.Subset s t)
  -/
  refine ⟨?_, fun h => monotone_nhdsSet h⟩
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    ⊢ LE.le (nhdsSet s) (nhdsSet t) → HasSubset.Subset s t
  -/
  simp_rw [Filter.le_def]; intro h x hx
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    h : ∀ (x : Set X), Membership.mem (nhdsSet t) x → Membership.mem (nhdsSet s) x
    x : X
    hx : Membership.mem s x
    ⊢ Membership.mem t x
  -/
  specialize h {x}ᶜ
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    x : X
    hx : Membership.mem s x
    h : Membership.mem (nhdsSet t) (HasCompl.compl (Singleton.singleton x)) → Memb …
    ⊢ Membership.mem t x
  -/
  simp_rw [compl_singleton_mem_nhdsSet_iff] at h
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    x : X
    hx : Membership.mem s x
    h : Not (Membership.mem t x) → Not (Membership.mem s x)
    ⊢ Membership.mem t x
  -/
  by_contra hxt
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    x : X
    hx : Membership.mem s x
    h : Not (Membership.mem t x) → Not (Membership.mem s x)
    hxt : Not (Membership.mem t x)
    ⊢ False
  -/
  exact h hxt hx
  /-
    🎉 no goals
  -/


@[simp]
theorem nhdsSet_inj_iff [T1Space X] {s t : Set X} : 𝓝ˢ s = 𝓝ˢ t ↔ s = t := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    ⊢ Iff (Eq (nhdsSet s) (nhdsSet t)) (Eq s t)
  -/
  simp_rw [le_antisymm_iff]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s t : Set X
    ⊢ Iff (And (LE.le (nhdsSet s) (nhdsSet t)) (LE.le (nhdsSet t) (nhdsSet s))) (A …
  -/
  exact and_congr nhdsSet_le_iff nhdsSet_le_iff
  /-
    🎉 no goals
  -/


theorem injective_nhdsSet [T1Space X] : Function.Injective (𝓝ˢ : Set X → Filter X) := fun _ _ hst =>
  nhdsSet_inj_iff.mp hst


theorem strictMono_nhdsSet [T1Space X] : StrictMono (𝓝ˢ : Set X → Filter X) :=
  monotone_nhdsSet.strictMono_of_injective injective_nhdsSet


@[simp]
theorem nhds_le_nhdsSet_iff [T1Space X] {s : Set X} {x : X} : 𝓝 x ≤ 𝓝ˢ s ↔ x ∈ s := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    x : X
    ⊢ Iff (LE.le (nhds x) (nhdsSet s)) (Membership.mem s x)
  -/
  rw [← nhdsSet_singleton, nhdsSet_le_iff, singleton_subset_iff]
  /-
    🎉 no goals
  -/


/-- Removing a non-isolated point from a dense set, one still obtains a dense set. -/
theorem Dense.diff_singleton [T1Space X] {s : Set X} (hs : Dense s) (x : X) [NeBot (𝓝[≠] x)] :
    Dense (s \ {x}) :=
  hs.inter_of_isOpen_right (dense_compl_singleton x) isOpen_compl_singleton


/-- Removing a finset from a dense set in a space without isolated points, one still
obtains a dense set. -/
theorem Dense.diff_finset [T1Space X] [∀ x : X, NeBot (𝓝[≠] x)] {s : Set X} (hs : Dense s)
    (t : Finset X) : Dense (s \ t) := by
  classical
  induction t using Finset.induction_on with
  | empty => simpa using hs
  | insert _ ih =>
    rw [Finset.coe_insert, ← union_singleton, ← diff_diff]
    exact ih.diff_singleton _


/-- Removing a finite set from a dense set in a space without isolated points, one still
obtains a dense set. -/
theorem Dense.diff_finite [T1Space X] [∀ x : X, NeBot (𝓝[≠] x)] {s : Set X} (hs : Dense s)
    {t : Set X} (ht : t.Finite) : Dense (s \ t) := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : ∀ (x : X), (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    s : Set X
    hs : Dense s
    t : Set X
    ht : t.Finite
    ⊢ Dense (SDiff.sdiff s t)
  -/
  convert hs.diff_finset ht.toFinset
  /-
    case h.e'_3.h.e'_4
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : ∀ (x : X), (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    s : Set X
    hs : Dense s
    t : Set X
    ht : t.Finite
    ⊢ Eq t ↑ht.toFinset
  -/
  exact (Finite.coe_toFinset _).symm
  /-
    🎉 no goals
  -/


/-- If a function to a `T1Space` tends to some limit `y` at some point `x`, then necessarily
`y = f x`. -/
theorem eq_of_tendsto_nhds [TopologicalSpace Y] [T1Space Y] {f : X → Y} {x : X} {y : Y}
    (h : Tendsto f (𝓝 x) (𝓝 y)) : f x = y :=
  by_contra fun hfa : f x ≠ y =>
    have fact₁ : {f x}ᶜ ∈ 𝓝 y := compl_singleton_mem_nhds hfa.symm
    have fact₂ : Tendsto f (pure x) (𝓝 y) := h.comp (tendsto_id'.2 <| pure_le_nhds x)
    fact₂ fact₁ (Eq.refl <| f x)


theorem Filter.Tendsto.eventually_ne {X} [TopologicalSpace Y] [T1Space Y] {g : X → Y}
    {l : Filter X} {b₁ b₂ : Y} (hg : Tendsto g l (𝓝 b₁)) (hb : b₁ ≠ b₂) : ∀ᶠ z in l, g z ≠ b₂ :=
  hg.eventually (isOpen_compl_singleton.eventually_mem hb)


theorem ContinuousAt.eventually_ne [TopologicalSpace Y] [T1Space Y] {g : X → Y} {x : X} {y : Y}
    (hg1 : ContinuousAt g x) (hg2 : g x ≠ y) : ∀ᶠ z in 𝓝 x, g z ≠ y :=
  hg1.tendsto.eventually_ne hg2


theorem eventually_ne_nhds [T1Space X] {a b : X} (h : a ≠ b) : ∀ᶠ x in 𝓝 a, x ≠ b :=
  IsOpen.eventually_mem isOpen_ne h


theorem eventually_ne_nhdsWithin [T1Space X] {a b : X} {s : Set X} (h : a ≠ b) :
    ∀ᶠ x in 𝓝[s] a, x ≠ b :=
  Filter.Eventually.filter_mono nhdsWithin_le_nhds <| eventually_ne_nhds h


theorem continuousWithinAt_insert [TopologicalSpace Y] [T1Space X]
    {x y : X} {s : Set X} {f : X → Y} :
    ContinuousWithinAt f (insert y s) x ↔ ContinuousWithinAt f s x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : T1Space X
    x y : X
    s : Set X
    f : X → Y
    ⊢ Iff (ContinuousWithinAt f (Insert.insert y s) x) (ContinuousWithinAt f s x)
  -/
  rcases eq_or_ne x y with (rfl | h)
    /-
      case inl
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : T1Space X
      x : X
      s : Set X
      f : X → Y
      ⊢ Iff (ContinuousWithinAt f (Insert.insert x s) x) (ContinuousWithinAt f s x)
    -/
  · exact continuousWithinAt_insert_self
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : T1Space X
    x y : X
    s : Set X
    f : X → Y
    h : Ne x y
    ⊢ Iff (ContinuousWithinAt f (Insert.insert y s) x) (ContinuousWithinAt f s x)
  -/
  simp_rw [ContinuousWithinAt, nhdsWithin_insert_of_ne h]
  /-
    🎉 no goals
  -/


alias ⟨ContinuousWithinAt.of_insert, ContinuousWithinAt.insert'⟩ := continuousWithinAt_insert


/-- See also `continuousWithinAt_diff_self` for the case `y = x` but not requiring `T1Space`. -/
theorem continuousWithinAt_diff_singleton [TopologicalSpace Y] [T1Space X]
    {x y : X} {s : Set X} {f : X → Y} :
    ContinuousWithinAt f (s \ {y}) x ↔ ContinuousWithinAt f s x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : T1Space X
    x y : X
    s : Set X
    f : X → Y
    ⊢ Iff (ContinuousWithinAt f (SDiff.sdiff s (Singleton.singleton y)) x) (Contin …
  -/
  rw [← continuousWithinAt_insert, insert_diff_singleton, continuousWithinAt_insert]
  /-
    🎉 no goals
  -/


/-- If two sets coincide locally around `x`, except maybe at `y`, then it is equivalent to be
continuous at `x` within one set or the other. -/
theorem continuousWithinAt_congr_set' [TopologicalSpace Y] [T1Space X]
    {x : X} {s t : Set X} {f : X → Y} (y : X) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    ContinuousWithinAt f s x ↔ ContinuousWithinAt f t x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : T1Space X
    x : X
    s t : Set X
    f : X → Y
    y : X
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt f t x)
  -/
  rw [← continuousWithinAt_insert_self (s := s), ← continuousWithinAt_insert_self (s := t)]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : T1Space X
    x : X
    s t : Set X
    f : X → Y
    y : X
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Iff (ContinuousWithinAt f (Insert.insert x s) x) (ContinuousWithinAt f (Inse …
  -/
  exact continuousWithinAt_congr_set (eventuallyEq_insert h)
  /-
    🎉 no goals
  -/


/-- To prove a function to a `T1Space` is continuous at some point `x`, it suffices to prove that
`f` admits *some* limit at `x`. -/
theorem continuousAt_of_tendsto_nhds [TopologicalSpace Y] [T1Space Y] {f : X → Y} {x : X} {y : Y}
    (h : Tendsto f (𝓝 x) (𝓝 y)) : ContinuousAt f x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : T1Space Y
    f : X → Y
    x : X
    y : Y
    h : Filter.Tendsto f (nhds x) (nhds y)
    ⊢ ContinuousAt f x
  -/
  rwa [ContinuousAt, eq_of_tendsto_nhds h]
  /-
    🎉 no goals
  -/


@[simp]
theorem tendsto_const_nhds_iff [T1Space X] {l : Filter Y} [NeBot l] {c d : X} :
                                               /-
                                                 X : Type u_1
                                                 Y : Type u_2
                                                 inst✝² : TopologicalSpace X
                                                 inst✝¹ : T1Space X
                                                 l : Filter Y
                                                 inst✝ : l.NeBot
                                                 c d : X
                                                 ⊢ Iff (Filter.Tendsto (fun x => c) l (nhds d)) (Eq c d)
                                               -/
    Tendsto (fun _ => c) l (𝓝 d) ↔ c = d := by simp_rw [Tendsto, Filter.map_const, pure_le_nhds_iff]
                                               /-
                                                 🎉 no goals
                                               -/


/-- A point with a finite neighborhood has to be isolated. -/
theorem isOpen_singleton_of_finite_mem_nhds [T1Space X] (x : X)
    {s : Set X} (hs : s ∈ 𝓝 x) (hsf : s.Finite) : IsOpen ({x} : Set X) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    ⊢ IsOpen (Singleton.singleton x)
  -/
  have A : {x} ⊆ s := by simp only [singleton_subset_iff, mem_of_mem_nhds hs]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    A : HasSubset.Subset (Singleton.singleton x) s
    ⊢ IsOpen (Singleton.singleton x)
  -/
  have B : IsClosed (s \ {x}) := (hsf.subset diff_subset).isClosed
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    A : HasSubset.Subset (Singleton.singleton x) s
    B : IsClosed (SDiff.sdiff s (Singleton.singleton x))
    ⊢ IsOpen (Singleton.singleton x)
  -/
  have C : (s \ {x})ᶜ ∈ 𝓝 x := B.isOpen_compl.mem_nhds fun h => h.2 rfl
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    A : HasSubset.Subset (Singleton.singleton x) s
    B : IsClosed (SDiff.sdiff s (Singleton.singleton x))
    C : Membership.mem (nhds x) (HasCompl.compl (SDiff.sdiff s (Singleton.singleto …
    ⊢ IsOpen (Singleton.singleton x)
  -/
  have D : {x} ∈ 𝓝 x := by simpa only [← diff_eq, diff_diff_cancel_left A] using inter_mem hs C
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    A : HasSubset.Subset (Singleton.singleton x) s
    B : IsClosed (SDiff.sdiff s (Singleton.singleton x))
    C : Membership.mem (nhds x) (HasCompl.compl (SDiff.sdiff s (Singleton.singleto …
    D : Membership.mem (nhds x) (Singleton.singleton x)
    ⊢ IsOpen (Singleton.singleton x)
  -/
  rwa [← mem_interior_iff_mem_nhds, ← singleton_subset_iff, subset_interior_iff_isOpen] at D
  /-
    🎉 no goals
  -/


/-- If the punctured neighborhoods of a point form a nontrivial filter, then any neighborhood is
infinite. -/
theorem infinite_of_mem_nhds {X} [TopologicalSpace X] [T1Space X] (x : X) [hx : NeBot (𝓝[≠] x)]
    {s : Set X} (hs : s ∈ 𝓝 x) : Set.Infinite s := by
  /-
    X : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    hx : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    s : Set X
    hs : Membership.mem (nhds x) s
    ⊢ s.Infinite
  -/
  refine fun hsf => hx.1 ?_
  /-
    X : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    hx : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    ⊢ Eq (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) Bot.bot
  -/
  rw [← isOpen_singleton_iff_punctured_nhds]
  /-
    X : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    x : X
    hx : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    s : Set X
    hs : Membership.mem (nhds x) s
    hsf : s.Finite
    ⊢ IsOpen (Singleton.singleton x)
  -/
  exact isOpen_singleton_of_finite_mem_nhds x hs hsf
  /-
    🎉 no goals
  -/


instance Finite.instDiscreteTopology [T1Space X] [Finite X] : DiscreteTopology X :=
  discreteTopology_iff_forall_isClosed.mpr (· |>.toFinite.isClosed)


theorem Set.Finite.continuousOn [T1Space X] [TopologicalSpace Y] {s : Set X} (hs : s.Finite)
    (f : X → Y) : ContinuousOn f s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : TopologicalSpace Y
    s : Set X
    hs : s.Finite
    f : X → Y
    ⊢ ContinuousOn f s
  -/
  rw [continuousOn_iff_continuous_restrict]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : TopologicalSpace Y
    s : Set X
    hs : s.Finite
    f : X → Y
    ⊢ Continuous (s.restrict f)
  -/
  have : Finite s := hs
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T1Space X
    inst✝ : TopologicalSpace Y
    s : Set X
    hs : s.Finite
    f : X → Y
    this : Finite ↑s
    ⊢ Continuous (s.restrict f)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


theorem PreconnectedSpace.trivial_of_discrete [PreconnectedSpace X] [DiscreteTopology X] :
    Subsingleton X := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : PreconnectedSpace X
    inst✝ : DiscreteTopology X
    ⊢ Subsingleton X
  -/
  rw [← not_nontrivial_iff_subsingleton]
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : PreconnectedSpace X
    inst✝ : DiscreteTopology X
    ⊢ Not (Nontrivial X)
  -/
  rintro ⟨x, y, hxy⟩
  /-
    case mk.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : PreconnectedSpace X
    inst✝ : DiscreteTopology X
    x y : X
    hxy : Ne x y
    ⊢ False
  -/
  rw [Ne, ← mem_singleton_iff, (isClopen_discrete _).eq_univ <| singleton_nonempty y] at hxy
  /-
    case mk.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : PreconnectedSpace X
    inst✝ : DiscreteTopology X
    x y : X
    hxy : Not (Membership.mem Set.univ x)
    ⊢ False
  -/
  exact hxy (mem_univ x)
  /-
    🎉 no goals
  -/


theorem IsPreconnected.infinite_of_nontrivial [T1Space X] {s : Set X} (h : IsPreconnected s)
    (hs : s.Nontrivial) : s.Infinite := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    h : IsPreconnected s
    hs : s.Nontrivial
    ⊢ s.Infinite
  -/
  refine mt (fun hf => (subsingleton_coe s).mp ?_) (not_subsingleton_iff.mpr hs)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    h : IsPreconnected s
    hs : s.Nontrivial
    hf : s.Finite
    ⊢ Subsingleton ↑s
  -/
  haveI := @Finite.instDiscreteTopology s _ _ hf.to_subtype
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    s : Set X
    h : IsPreconnected s
    hs : s.Nontrivial
    hf : s.Finite
    this : DiscreteTopology ↑s
    ⊢ Subsingleton ↑s
  -/
  exact @PreconnectedSpace.trivial_of_discrete _ _ (Subtype.preconnectedSpace h) _
  /-
    🎉 no goals
  -/


theorem ConnectedSpace.infinite [ConnectedSpace X] [Nontrivial X] [T1Space X] : Infinite X :=
  infinite_univ_iff.mp <| isPreconnected_univ.infinite_of_nontrivial nontrivial_univ


/-- A non-trivial connected T1 space has no isolated points. -/
instance (priority := 100) ConnectedSpace.neBot_nhdsWithin_compl_of_nontrivial_of_t1space
    [ConnectedSpace X] [Nontrivial X] [T1Space X] (x : X) :
    NeBot (𝓝[≠] x) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : ConnectedSpace X
    inst✝¹ : Nontrivial X
    inst✝ : T1Space X
    x : X
    ⊢ (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
  -/
  by_contra contra
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : ConnectedSpace X
    inst✝¹ : Nontrivial X
    inst✝ : T1Space X
    x : X
    contra : Not (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    ⊢ False
  -/
  rw [not_neBot, ← isOpen_singleton_iff_punctured_nhds] at contra
  replace contra := nonempty_inter isOpen_compl_singleton
    contra (compl_union_self _) (Set.nonempty_compl_of_nontrivial _) (singleton_nonempty _)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : ConnectedSpace X
    inst✝¹ : Nontrivial X
    inst✝ : T1Space X
    x : X
    contra : (Inter.inter (HasCompl.compl (Singleton.singleton x)) (Singleton.sing …
    ⊢ False
  -/
  simp [compl_inter_self {x}] at contra
  /-
    🎉 no goals
  -/


theorem SeparationQuotient.t1Space_iff : T1Space (SeparationQuotient X) ↔ R0Space X := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (T1Space (SeparationQuotient X)) (R0Space X)
  -/
  rw [r0Space_iff, ((t1Space_TFAE (SeparationQuotient X)).out 0 9 :)]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (∀ ⦃x y : SeparationQuotient X⦄, Specializes x y → Eq x y) (Symmetric Sp …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ (∀ ⦃x y : SeparationQuotient X⦄, Specializes x y → Eq x y) → Symmetric Speci …
    -/
  · intro h x y xspecy
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : ∀ ⦃x y : SeparationQuotient X⦄, Specializes x y → Eq x y
      x y : X
      xspecy : Specializes x y
      ⊢ Specializes y x
    -/
    rw [← IsInducing.specializes_iff isInducing_mk, h xspecy] at *
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ Symmetric Specializes → ∀ ⦃x y : SeparationQuotient X⦄, Specializes x y → Eq …
    -/
  · rintro h ⟨x⟩ ⟨y⟩ sxspecsy
    /-
      case mpr.mk.mk
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : Symmetric Specializes
      x✝ : SeparationQuotient X
      x : X
      y✝ : SeparationQuotient X
      y : X
      sxspecsy : Specializes (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(insepa …
      ⊢ Eq (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(inseparableSetoid X)) y)
    -/
    have xspecy : x ⤳ y := isInducing_mk.specializes_iff.mp sxspecsy
    /-
      case mpr.mk.mk
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : Symmetric Specializes
      x✝ : SeparationQuotient X
      x : X
      y✝ : SeparationQuotient X
      y : X
      sxspecsy : Specializes (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(insepa …
      xspecy : Specializes x y
      ⊢ Eq (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(inseparableSetoid X)) y)
    -/
    have yspecx : y ⤳ x := h xspecy
    /-
      case mpr.mk.mk
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : Symmetric Specializes
      x✝ : SeparationQuotient X
      x : X
      y✝ : SeparationQuotient X
      y : X
      sxspecsy : Specializes (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(insepa …
      xspecy : Specializes x y
      yspecx : Specializes y x
      ⊢ Eq (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(inseparableSetoid X)) y)
    -/
    erw [mk_eq_mk, inseparable_iff_specializes_and]
    /-
      case mpr.mk.mk
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : Symmetric Specializes
      x✝ : SeparationQuotient X
      x : X
      y✝ : SeparationQuotient X
      y : X
      sxspecsy : Specializes (Quot.mk (⇑(inseparableSetoid X)) x) (Quot.mk (⇑(insepa …
      xspecy : Specializes x y
      yspecx : Specializes y x
      ⊢ And (Specializes x y) (Specializes y x)
    -/
    exact ⟨xspecy, yspecx⟩
    /-
      🎉 no goals
    -/


lemma Set.Subsingleton.isClosed [T1Space X] {A : Set X} (h : A.Subsingleton) : IsClosed A := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    h : A.Subsingleton
    ⊢ IsClosed A
  -/
  rcases h.eq_empty_or_singleton with rfl | ⟨x, rfl⟩
    /-
      case inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      h : EmptyCollection.emptyCollection.Subsingleton
      ⊢ IsClosed EmptyCollection.emptyCollection
    -/
  · exact isClosed_empty
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      x : X
      h : (Singleton.singleton x).Subsingleton
      ⊢ IsClosed (Singleton.singleton x)
    -/
  · exact isClosed_singleton
    /-
      🎉 no goals
    -/


lemma isClosed_inter_singleton [T1Space X] {A : Set X} {a : X} : IsClosed (A ∩ {a}) :=
  Subsingleton.inter_singleton.isClosed


lemma isClosed_singleton_inter [T1Space X] {A : Set X} {a : X} : IsClosed ({a} ∩ A) :=
  Subsingleton.singleton_inter.isClosed


theorem singleton_mem_nhdsWithin_of_mem_discrete {s : Set X} [DiscreteTopology s] {x : X}
    (hx : x ∈ s) : {x} ∈ 𝓝[s] x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology ↑s
    x : X
    hx : Membership.mem s x
    ⊢ Membership.mem (nhdsWithin x s) (Singleton.singleton x)
  -/
  have : ({⟨x, hx⟩} : Set s) ∈ 𝓝 (⟨x, hx⟩ : s) := by simp [nhds_discrete]
  simpa only [nhdsWithin_eq_map_subtype_coe hx, image_singleton] using
    @image_mem_map _ _ _ ((↑) : s → X) _ this


/-- The neighbourhoods filter of `x` within `s`, under the discrete topology, is equal to
the pure `x` filter (which is the principal filter at the singleton `{x}`.) -/
theorem nhdsWithin_of_mem_discrete {s : Set X} [DiscreteTopology s] {x : X} (hx : x ∈ s) :
    𝓝[s] x = pure x :=
  le_antisymm (le_pure_iff.2 <| singleton_mem_nhdsWithin_of_mem_discrete hx) (pure_le_nhdsWithin hx)


theorem Filter.HasBasis.exists_inter_eq_singleton_of_mem_discrete {ι : Type*} {p : ι → Prop}
    {t : ι → Set X} {s : Set X} [DiscreteTopology s] {x : X} (hb : (𝓝 x).HasBasis p t)
    (hx : x ∈ s) : ∃ i, p i ∧ t i ∩ s = {x} := by
  rcases (nhdsWithin_hasBasis hb s).mem_iff.1 (singleton_mem_nhdsWithin_of_mem_discrete hx) with
    ⟨i, hi, hix⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_3
    p : ι → Prop
    t : ι → Set X
    s : Set X
    inst✝ : DiscreteTopology ↑s
    x : X
    hb : (nhds x).HasBasis p t
    hx : Membership.mem s x
    i : ι
    hi : p i
    hix : HasSubset.Subset (Inter.inter (t i) s) (Singleton.singleton x)
    ⊢ Exists fun i => And (p i) (Eq (Inter.inter (t i) s) (Singleton.singleton x))
  -/
  exact ⟨i, hi, hix.antisymm <| singleton_subset_iff.2 ⟨mem_of_mem_nhds <| hb.mem_of_mem hi, hx⟩⟩
  /-
    🎉 no goals
  -/


/-- A point `x` in a discrete subset `s` of a topological space admits a neighbourhood
that only meets `s` at `x`. -/
theorem nhds_inter_eq_singleton_of_mem_discrete {s : Set X} [DiscreteTopology s] {x : X}
    (hx : x ∈ s) : ∃ U ∈ 𝓝 x, U ∩ s = {x} := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology ↑s
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun U => And (Membership.mem (nhds x) U) (Eq (Inter.inter U s) (Singl …
  -/
  simpa using (𝓝 x).basis_sets.exists_inter_eq_singleton_of_mem_discrete hx
  /-
    🎉 no goals
  -/


/-- Let `x` be a point in a discrete subset `s` of a topological space, then there exists an open
set that only meets `s` at `x`. -/
theorem isOpen_inter_eq_singleton_of_mem_discrete {s : Set X} [DiscreteTopology s] {x : X}
    (hx : x ∈ s) : ∃ U : Set X, IsOpen U ∧ U ∩ s = {x} := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology ↑s
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun U => And (IsOpen U) (Eq (Inter.inter U s) (Singleton.singleton x))
  -/
  obtain ⟨U, hU_nhds, hU_inter⟩ := nhds_inter_eq_singleton_of_mem_discrete hx
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology ↑s
    x : X
    hx : Membership.mem s x
    U : Set X
    hU_nhds : Membership.mem (nhds x) U
    hU_inter : Eq (Inter.inter U s) (Singleton.singleton x)
    ⊢ Exists fun U => And (IsOpen U) (Eq (Inter.inter U s) (Singleton.singleton x))
  -/
  obtain ⟨t, ht_sub, ht_open, ht_x⟩ := mem_nhds_iff.mp hU_nhds
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology ↑s
    x : X
    hx : Membership.mem s x
    U : Set X
    hU_nhds : Membership.mem (nhds x) U
    hU_inter : Eq (Inter.inter U s) (Singleton.singleton x)
    t : Set X
    ht_sub : HasSubset.Subset t U
    ht_open : IsOpen t
    ht_x : Membership.mem t x
    ⊢ Exists fun U => And (IsOpen U) (Eq (Inter.inter U s) (Singleton.singleton x))
  -/
  refine ⟨t, ht_open, Set.Subset.antisymm ?_ ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s : Set X
      inst✝ : DiscreteTopology ↑s
      x : X
      hx : Membership.mem s x
      U : Set X
      hU_nhds : Membership.mem (nhds x) U
      hU_inter : Eq (Inter.inter U s) (Singleton.singleton x)
      t : Set X
      ht_sub : HasSubset.Subset t U
      ht_open : IsOpen t
      ht_x : Membership.mem t x
      ⊢ HasSubset.Subset (Inter.inter t s) (Singleton.singleton x)
    -/
  · exact hU_inter ▸ Set.inter_subset_inter_left s ht_sub
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s : Set X
      inst✝ : DiscreteTopology ↑s
      x : X
      hx : Membership.mem s x
      U : Set X
      hU_nhds : Membership.mem (nhds x) U
      hU_inter : Eq (Inter.inter U s) (Singleton.singleton x)
      t : Set X
      ht_sub : HasSubset.Subset t U
      ht_open : IsOpen t
      ht_x : Membership.mem t x
      ⊢ HasSubset.Subset (Singleton.singleton x) (Inter.inter t s)
    -/
  · rw [Set.subset_inter_iff, Set.singleton_subset_iff, Set.singleton_subset_iff]
    /-
      case intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s : Set X
      inst✝ : DiscreteTopology ↑s
      x : X
      hx : Membership.mem s x
      U : Set X
      hU_nhds : Membership.mem (nhds x) U
      hU_inter : Eq (Inter.inter U s) (Singleton.singleton x)
      t : Set X
      ht_sub : HasSubset.Subset t U
      ht_open : IsOpen t
      ht_x : Membership.mem t x
      ⊢ And (Membership.mem t x) (Membership.mem s x)
    -/
    exact ⟨ht_x, hx⟩
    /-
      🎉 no goals
    -/


/-- For point `x` in a discrete subset `s` of a topological space, there is a set `U`
such that
1. `U` is a punctured neighborhood of `x` (ie. `U ∪ {x}` is a neighbourhood of `x`),
2. `U` is disjoint from `s`.
-/
theorem disjoint_nhdsWithin_of_mem_discrete {s : Set X} [DiscreteTopology s] {x : X} (hx : x ∈ s) :
    ∃ U ∈ 𝓝[≠] x, Disjoint U s :=
  let ⟨V, h, h'⟩ := nhds_inter_eq_singleton_of_mem_discrete hx
  ⟨{x}ᶜ ∩ V, inter_mem_nhdsWithin _ h,
                                        /-
                                          X : Type u_1
                                          inst✝¹ : TopologicalSpace X
                                          s : Set X
                                          inst✝ : DiscreteTopology ↑s
                                          x : X
                                          hx : Membership.mem s x
                                          V : Set X
                                          h : Membership.mem (nhds x) V
                                          h' : Eq (Inter.inter V s) (Singleton.singleton x)
                                          ⊢ Eq (Inter.inter (Inter.inter (HasCompl.compl (Singleton.singleton x)) V) s)  …
                                        -/
    disjoint_iff_inter_eq_empty.mpr (by rw [inter_assoc, h', compl_inter_self])⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem isClosedEmbedding_update {ι : Type*} {β : ι → Type*}
    [DecidableEq ι] [(i : ι) → TopologicalSpace (β i)]
    (x : (i : ι) → β i) (i : ι) [(i : ι) → T1Space (β i)] :
    IsClosedEmbedding (update x i) := by
  refine .of_continuous_injective_isClosedMap (continuous_const.update i continuous_id)
    (update_injective x i) fun s hs ↦ ?_
  /-
    ι : Type u_3
    β : ι → Type u_4
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → TopologicalSpace (β i)
    x : (i : ι) → β i
    i : ι
    inst✝ : ∀ (i : ι), T1Space (β i)
    s : Set (β i)
    hs : IsClosed s
    ⊢ IsClosed (Set.image (Function.update x i) s)
  -/
  rw [update_image]
  /-
    ι : Type u_3
    β : ι → Type u_4
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → TopologicalSpace (β i)
    x : (i : ι) → β i
    i : ι
    inst✝ : ∀ (i : ι), T1Space (β i)
    s : Set (β i)
    hs : IsClosed s
    ⊢ IsClosed (Set.univ.pi (Function.update (fun j => Singleton.singleton (x j))  …
  -/
  apply isClosed_set_pi
  /-
    case hs
    ι : Type u_3
    β : ι → Type u_4
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → TopologicalSpace (β i)
    x : (i : ι) → β i
    i : ι
    inst✝ : ∀ (i : ι), T1Space (β i)
    s : Set (β i)
    hs : IsClosed s
    ⊢ ∀ (a : ι), Membership.mem Set.univ a → IsClosed (Function.update (fun j => S …
  -/
  simp [forall_update_iff, hs, isClosed_singleton]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_update := isClosedEmbedding_update


/-- A topological space is called a *preregular* (a.k.a. R₁) space,
if any two topologically distinguishable points have disjoint neighbourhoods. -/
@[mk_iff r1Space_iff_specializes_or_disjoint_nhds]
class R1Space (X : Type*) [TopologicalSpace X] : Prop where
  specializes_or_disjoint_nhds (x y : X) : Specializes x y ∨ Disjoint (𝓝 x) (𝓝 y)


instance (priority := 100) : R0Space X where
  specializes_symmetric _ _ h := (specializes_or_disjoint_nhds _ _).resolve_right <| fun hd ↦
    h.not_disjoint hd.symm


theorem disjoint_nhds_nhds_iff_not_specializes : Disjoint (𝓝 x) (𝓝 y) ↔ ¬x ⤳ y :=
  ⟨fun hd hspec ↦ hspec.not_disjoint hd, (specializes_or_disjoint_nhds _ _).resolve_left⟩


theorem specializes_iff_not_disjoint : x ⤳ y ↔ ¬Disjoint (𝓝 x) (𝓝 y) :=
  disjoint_nhds_nhds_iff_not_specializes.not_left.symm


theorem disjoint_nhds_nhds_iff_not_inseparable : Disjoint (𝓝 x) (𝓝 y) ↔ ¬Inseparable x y := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    x y : X
    ⊢ Iff (Disjoint (nhds x) (nhds y)) (Not (Inseparable x y))
  -/
  rw [disjoint_nhds_nhds_iff_not_specializes, specializes_iff_inseparable]
  /-
    🎉 no goals
  -/


theorem r1Space_iff_inseparable_or_disjoint_nhds {X : Type*} [TopologicalSpace X] :
    R1Space X ↔ ∀ x y : X, Inseparable x y ∨ Disjoint (𝓝 x) (𝓝 y) :=
  ⟨fun _h x y ↦ (specializes_or_disjoint_nhds x y).imp_left Specializes.inseparable, fun h ↦
    ⟨fun x y ↦ (h x y).imp_left Inseparable.specializes⟩⟩


theorem Inseparable.of_nhds_neBot {x y : X} (h : NeBot (𝓝 x ⊓ 𝓝 y)) :
    Inseparable x y :=
  (r1Space_iff_inseparable_or_disjoint_nhds.mp ‹_› _ _).resolve_right fun h' => h.ne h'.eq_bot


/-- Limits are unique up to separability.

A weaker version of `tendsto_nhds_unique` for `R1Space`. -/
theorem tendsto_nhds_unique_inseparable {f : Y → X} {l : Filter Y} {a b : X} [NeBot l]
    (ha : Tendsto f l (𝓝 a)) (hb : Tendsto f l (𝓝 b)) : Inseparable a b :=
  .of_nhds_neBot <| neBot_of_le <| le_inf ha hb


theorem isClosed_setOf_specializes : IsClosed { p : X × X | p.1 ⤳ p.2 } := by
  simp only [← isOpen_compl_iff, compl_setOf, ← disjoint_nhds_nhds_iff_not_specializes,
    isOpen_setOf_disjoint_nhds_nhds]


theorem isClosed_setOf_inseparable : IsClosed { p : X × X | Inseparable p.1 p.2 } := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    ⊢ IsClosed (setOf fun p => Inseparable p.1 p.2)
  -/
  simp only [← specializes_iff_inseparable, isClosed_setOf_specializes]
  /-
    🎉 no goals
  -/


/-- In an R₁ space, a point belongs to the closure of a compact set `K`
if and only if it is topologically inseparable from some point of `K`. -/
theorem IsCompact.mem_closure_iff_exists_inseparable {K : Set X} (hK : IsCompact K) :
    y ∈ closure K ↔ ∃ x ∈ K, Inseparable x y := by
  refine ⟨fun hy ↦ ?_, fun ⟨x, hxK, hxy⟩ ↦
    (hxy.mem_closed_iff isClosed_closure).1 <| subset_closure hxK⟩
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    y : X
    K : Set X
    hK : IsCompact K
    hy : Membership.mem (closure K) y
    ⊢ Exists fun x => And (Membership.mem K x) (Inseparable x y)
  -/
  contrapose! hy
  have : Disjoint (𝓝 y) (𝓝ˢ K) := hK.disjoint_nhdsSet_right.2 fun x hx ↦
    (disjoint_nhds_nhds_iff_not_inseparable.2 (hy x hx)).symm
  simpa only [disjoint_iff, not_mem_closure_iff_nhdsWithin_eq_bot]
    using this.mono_right principal_le_nhdsSet


theorem IsCompact.closure_eq_biUnion_inseparable {K : Set X} (hK : IsCompact K) :
    closure K = ⋃ x ∈ K, {y | Inseparable x y} := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K : Set X
    hK : IsCompact K
    ⊢ Eq (closure K) (Set.iUnion fun x => Set.iUnion fun h => setOf fun y => Insep …
  -/
  ext; simp [hK.mem_closure_iff_exists_inseparable]
       /-
         🎉 no goals
       -/


/-- In an R₁ space, the closure of a compact set is the union of the closures of its points. -/
theorem IsCompact.closure_eq_biUnion_closure_singleton {K : Set X} (hK : IsCompact K) :
    closure K = ⋃ x ∈ K, closure {x} := by
  simp only [hK.closure_eq_biUnion_inseparable, ← specializes_iff_inseparable,
    specializes_iff_mem_closure, setOf_mem_eq]


/-- In an R₁ space, if a compact set `K` is contained in an open set `U`,
then its closure is also contained in `U`. -/
theorem IsCompact.closure_subset_of_isOpen {K : Set X} (hK : IsCompact K)
    {U : Set X} (hU : IsOpen U) (hKU : K ⊆ U) : closure K ⊆ U := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K : Set X
    hK : IsCompact K
    U : Set X
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    ⊢ HasSubset.Subset (closure K) U
  -/
  rw [hK.closure_eq_biUnion_inseparable, iUnion₂_subset_iff]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K : Set X
    hK : IsCompact K
    U : Set X
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    ⊢ ∀ (i : X), Membership.mem K i → HasSubset.Subset (setOf fun y => Inseparable …
  -/
  exact fun x hx y hxy ↦ (hxy.mem_open_iff hU).1 (hKU hx)
  /-
    🎉 no goals
  -/


/-- The closure of a compact set in an R₁ space is a compact set. -/
protected theorem IsCompact.closure {K : Set X} (hK : IsCompact K) : IsCompact (closure K) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K : Set X
    hK : IsCompact K
    ⊢ IsCompact (closure K)
  -/
  refine isCompact_of_finite_subcover fun U hUo hKU ↦ ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K : Set X
    hK : IsCompact K
    ι✝ : Type u_1
    U : ι✝ → Set X
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hKU : HasSubset.Subset (closure K) (Set.iUnion fun i => U i)
    ⊢ Exists fun t => HasSubset.Subset (closure K) (Set.iUnion fun i => Set.iUnion …
  -/
  rcases hK.elim_finite_subcover U hUo (subset_closure.trans hKU) with ⟨t, ht⟩
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K : Set X
    hK : IsCompact K
    ι✝ : Type u_1
    U : ι✝ → Set X
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hKU : HasSubset.Subset (closure K) (Set.iUnion fun i => U i)
    t : Finset ι✝
    ht : HasSubset.Subset K (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ Exists fun t => HasSubset.Subset (closure K) (Set.iUnion fun i => Set.iUnion …
  -/
  exact ⟨t, hK.closure_subset_of_isOpen (isOpen_biUnion fun _ _ ↦ hUo _) ht⟩
  /-
    🎉 no goals
  -/


theorem IsCompact.closure_of_subset {s K : Set X} (hK : IsCompact K) (h : s ⊆ K) :
    IsCompact (closure s) :=
  hK.closure.of_isClosed_subset isClosed_closure (closure_mono h)


@[simp]
theorem exists_isCompact_superset_iff {s : Set X} :
    (∃ K, IsCompact K ∧ s ⊆ K) ↔ IsCompact (closure s) :=
  ⟨fun ⟨_K, hK, hsK⟩ => hK.closure_of_subset hsK, fun h => ⟨closure s, h, subset_closure⟩⟩


/-- If `K` and `L` are disjoint compact sets in an R₁ topological space
and `L` is also closed, then `K` and `L` have disjoint neighborhoods. -/
theorem SeparatedNhds.of_isCompact_isCompact_isClosed {K L : Set X} (hK : IsCompact K)
    (hL : IsCompact L) (h'L : IsClosed L) (hd : Disjoint K L) : SeparatedNhds K L := by
  simp_rw [separatedNhds_iff_disjoint, hK.disjoint_nhdsSet_left, hL.disjoint_nhdsSet_right,
    disjoint_nhds_nhds_iff_not_inseparable]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K L : Set X
    hK : IsCompact K
    hL : IsCompact L
    h'L : IsClosed L
    hd : Disjoint K L
    ⊢ ∀ (x : X), Membership.mem K x → ∀ (x_1 : X), Membership.mem L x_1 → Not (Ins …
  -/
  intro x hx y hy h
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K L : Set X
    hK : IsCompact K
    hL : IsCompact L
    h'L : IsClosed L
    hd : Disjoint K L
    x : X
    hx : Membership.mem K x
    y : X
    hy : Membership.mem L y
    h : Inseparable x y
    ⊢ False
  -/
  exact absurd ((h.mem_closed_iff h'L).2 hy) <| disjoint_left.1 hd hx
  /-
    🎉 no goals
  -/


/-- If a compact set is covered by two open sets, then we can cover it by two compact subsets. -/
theorem IsCompact.binary_compact_cover {K U V : Set X}
    (hK : IsCompact K) (hU : IsOpen U) (hV : IsOpen V) (h2K : K ⊆ U ∪ V) :
    ∃ K₁ K₂ : Set X, IsCompact K₁ ∧ IsCompact K₂ ∧ K₁ ⊆ U ∧ K₂ ⊆ V ∧ K = K₁ ∪ K₂ := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K U V : Set X
    hK : IsCompact K
    hU : IsOpen U
    hV : IsOpen V
    h2K : HasSubset.Subset K (Union.union U V)
    ⊢ Exists fun K₁ => Exists fun K₂ => And (IsCompact K₁) (And (IsCompact K₂) (An …
  -/
  have hK' : IsCompact (closure K) := hK.closure
  have : SeparatedNhds (closure K \ U) (closure K \ V) := by
    apply SeparatedNhds.of_isCompact_isCompact_isClosed (hK'.diff hU) (hK'.diff hV)
      (isClosed_closure.sdiff hV)
    rw [disjoint_iff_inter_eq_empty, diff_inter_diff, diff_eq_empty]
    exact hK.closure_subset_of_isOpen (hU.union hV) h2K
  have : SeparatedNhds (K \ U) (K \ V) :=
    this.mono (diff_subset_diff_left (subset_closure)) (diff_subset_diff_left (subset_closure))
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    K U V : Set X
    hK : IsCompact K
    hU : IsOpen U
    hV : IsOpen V
    h2K : HasSubset.Subset K (Union.union U V)
    hK' : IsCompact (closure K)
    this✝ : SeparatedNhds (SDiff.sdiff (closure K) U) (SDiff.sdiff (closure K) V)
    this : SeparatedNhds (SDiff.sdiff K U) (SDiff.sdiff K V)
    ⊢ Exists fun K₁ => Exists fun K₂ => And (IsCompact K₁) (And (IsCompact K₂) (An …
  -/
  rcases this with ⟨O₁, O₂, h1O₁, h1O₂, h2O₁, h2O₂, hO⟩
  exact ⟨K \ O₁, K \ O₂, hK.diff h1O₁, hK.diff h1O₂, diff_subset_comm.mp h2O₁,
    diff_subset_comm.mp h2O₂, by rw [← diff_inter, hO.inter_eq, diff_empty]⟩


/-- For every finite open cover `Uᵢ` of a compact set, there exists a compact cover `Kᵢ ⊆ Uᵢ`. -/
theorem IsCompact.finite_compact_cover {s : Set X} (hs : IsCompact s) {ι : Type*}
    (t : Finset ι) (U : ι → Set X) (hU : ∀ i ∈ t, IsOpen (U i)) (hsC : s ⊆ ⋃ i ∈ t, U i) :
    ∃ K : ι → Set X, (∀ i, IsCompact (K i)) ∧ (∀ i, K i ⊆ U i) ∧ s = ⋃ i ∈ t, K i := by
  classical
  induction' t using Finset.induction with x t hx ih generalizing U s
  · refine ⟨fun _ => ∅, fun _ => isCompact_empty, fun i => empty_subset _, ?_⟩
    simpa only [subset_empty_iff, Finset.not_mem_empty, iUnion_false, iUnion_empty] using hsC
  simp only [Finset.set_biUnion_insert] at hsC
  simp only [Finset.forall_mem_insert] at hU
  have hU' : ∀ i ∈ t, IsOpen (U i) := fun i hi => hU.2 i hi
  rcases hs.binary_compact_cover hU.1 (isOpen_biUnion hU') hsC with
    ⟨K₁, K₂, h1K₁, h1K₂, h2K₁, h2K₂, hK⟩
  rcases ih h1K₂ U hU' h2K₂ with ⟨K, h1K, h2K, h3K⟩
  refine ⟨update K x K₁, ?_, ?_, ?_⟩
  · intro i
    rcases eq_or_ne i x with rfl | hi
    · simp only [update_self, h1K₁]
    · simp only [update_of_ne hi, h1K]
  · intro i
    rcases eq_or_ne i x with rfl | hi
    · simp only [update_self, h2K₁]
    · simp only [update_of_ne hi, h2K]
  · simp only [Finset.set_biUnion_insert_update _ hx, hK, h3K]


theorem R1Space.of_continuous_specializes_imp [TopologicalSpace Y] {f : Y → X} (hc : Continuous f)
    (hspec : ∀ x y, f x ⤳ f y → x ⤳ y) : R1Space Y where
  specializes_or_disjoint_nhds x y := (specializes_or_disjoint_nhds (f x) (f y)).imp (hspec x y) <|
    ((hc.tendsto _).disjoint · (hc.tendsto _))


theorem Topology.IsInducing.r1Space [TopologicalSpace Y] {f : Y → X} (hf : IsInducing f) :
    R1Space Y := .of_continuous_specializes_imp hf.continuous fun _ _ ↦ hf.specializes_iff.1


@[deprecated (since := "2024-10-28")] alias Inducing.r1Space := IsInducing.r1Space


protected theorem R1Space.induced (f : Y → X) : @R1Space Y (.induced f ‹_›) :=
  @IsInducing.r1Space _ _ _ _ (.induced f _) f (.induced f)


instance (p : X → Prop) : R1Space (Subtype p) := .induced _


protected theorem R1Space.sInf {X : Type*} {T : Set (TopologicalSpace X)}
    (hT : ∀ t ∈ T, @R1Space X t) : @R1Space X (sInf T) := by
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
    ⊢ R1Space X
  -/
  let _ := sInf T
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
    x✝ : TopologicalSpace X := InfSet.sInf T
    ⊢ R1Space X
  -/
  refine ⟨fun x y ↦ ?_⟩
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
    x✝ : TopologicalSpace X := InfSet.sInf T
    x y : X
    ⊢ Or (Specializes x y) (Disjoint (nhds x) (nhds y))
  -/
  simp only [Specializes, nhds_sInf]
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
    x✝ : TopologicalSpace X := InfSet.sInf T
    x y : X
    ⊢ Or (LE.le (iInf fun t => iInf fun h => nhds x) (iInf fun t => iInf fun h =>  …
  -/
  rcases em (∃ t ∈ T, Disjoint (@nhds X t x) (@nhds X t y)) with ⟨t, htT, htd⟩ | hTd
    /-
      case inl.intro.intro
      X : Type u_3
      T : Set (TopologicalSpace X)
      hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
      x✝ : TopologicalSpace X := InfSet.sInf T
      x y : X
      t : TopologicalSpace X
      htT : Membership.mem T t
      htd : Disjoint (nhds x) (nhds y)
      ⊢ Or (LE.le (iInf fun t => iInf fun h => nhds x) (iInf fun t => iInf fun h =>  …
    -/
  · exact .inr <| htd.mono (iInf₂_le t htT) (iInf₂_le t htT)
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_3
      T : Set (TopologicalSpace X)
      hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
      x✝ : TopologicalSpace X := InfSet.sInf T
      x y : X
      hTd : Not (Exists fun t => And (Membership.mem T t) (Disjoint (nhds x) (nhds y …
      ⊢ Or (LE.le (iInf fun t => iInf fun h => nhds x) (iInf fun t => iInf fun h =>  …
    -/
  · push_neg at hTd
    /-
      case inr
      X : Type u_3
      T : Set (TopologicalSpace X)
      hT : ∀ (t : TopologicalSpace X), Membership.mem T t → R1Space X
      x✝ : TopologicalSpace X := InfSet.sInf T
      x y : X
      hTd : ∀ (t : TopologicalSpace X), Membership.mem T t → Not (Disjoint (nhds x)  …
      ⊢ Or (LE.le (iInf fun t => iInf fun h => nhds x) (iInf fun t => iInf fun h =>  …
    -/
    exact .inl <| iInf₂_mono fun t ht ↦ ((hT t ht).1 x y).resolve_right (hTd t ht)
    /-
      🎉 no goals
    -/


protected theorem R1Space.iInf {ι X : Type*} {t : ι → TopologicalSpace X}
    (ht : ∀ i, @R1Space X (t i)) : @R1Space X (iInf t) :=
  .sInf <| forall_mem_range.2 ht


protected theorem R1Space.inf {X : Type*} {t₁ t₂ : TopologicalSpace X}
    (h₁ : @R1Space X t₁) (h₂ : @R1Space X t₂) : @R1Space X (t₁ ⊓ t₂) := by
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : R1Space X
    h₂ : R1Space X
    ⊢ R1Space X
  -/
  rw [inf_eq_iInf]
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : R1Space X
    h₂ : R1Space X
    ⊢ R1Space X
  -/
  apply R1Space.iInf
  /-
    case ht
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : R1Space X
    h₂ : R1Space X
    ⊢ ∀ (i : Bool), R1Space X
  -/
  simp [*]
  /-
    🎉 no goals
  -/


instance [TopologicalSpace Y] [R1Space Y] : R1Space (X × Y) :=
  .inf (.induced _) (.induced _)


instance {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)] [∀ i, R1Space (X i)] :
    R1Space (∀ i, X i) :=
  .iInf fun _ ↦ .induced _


theorem exists_mem_nhds_isCompact_mapsTo_of_isCompact_mem_nhds
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y] [R1Space Y] {f : X → Y} {x : X}
    {K : Set X} {s : Set Y} (hf : Continuous f) (hs : s ∈ 𝓝 (f x)) (hKc : IsCompact K)
    (hKx : K ∈ 𝓝 x) : ∃ L ∈ 𝓝 x, IsCompact L ∧ MapsTo f L s := by
  /-
    X : Type u_3
    Y : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : R1Space Y
    f : X → Y
    x : X
    K : Set X
    s : Set Y
    hf : Continuous f
    hs : Membership.mem (nhds (f x)) s
    hKc : IsCompact K
    hKx : Membership.mem (nhds x) K
    ⊢ Exists fun L => And (Membership.mem (nhds x) L) (And (IsCompact L) (Set.Maps …
  -/
  have hc : IsCompact (f '' K \ interior s) := (hKc.image hf).diff isOpen_interior
  obtain ⟨U, V, Uo, Vo, hxU, hV, hd⟩ : SeparatedNhds {f x} (f '' K \ interior s) := by
    simp_rw [separatedNhds_iff_disjoint, nhdsSet_singleton, hc.disjoint_nhdsSet_right,
      disjoint_nhds_nhds_iff_not_inseparable]
    rintro y ⟨-, hys⟩ hxy
    refine hys <| (hxy.mem_open_iff isOpen_interior).1 ?_
    rwa [mem_interior_iff_mem_nhds]
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : R1Space Y
    f : X → Y
    x : X
    K : Set X
    s : Set Y
    hf : Continuous f
    hs : Membership.mem (nhds (f x)) s
    hKc : IsCompact K
    hKx : Membership.mem (nhds x) K
    hc : IsCompact (SDiff.sdiff (Set.image f K) (interior s))
    U V : Set Y
    Uo : IsOpen U
    Vo : IsOpen V
    hxU : HasSubset.Subset (Singleton.singleton (f x)) U
    hV : HasSubset.Subset (SDiff.sdiff (Set.image f K) (interior s)) V
    hd : Disjoint U V
    ⊢ Exists fun L => And (Membership.mem (nhds x) L) (And (IsCompact L) (Set.Maps …
  -/
  refine ⟨K \ f ⁻¹' V, diff_mem hKx ?_, hKc.diff <| Vo.preimage hf, fun y hy ↦ ?_⟩
  · filter_upwards [hf.continuousAt <| Uo.mem_nhds (hxU rfl)] with x hx
      using Set.disjoint_left.1 hd hx
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      X : Type u_3
      Y : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : R1Space Y
      f : X → Y
      x : X
      K : Set X
      s : Set Y
      hf : Continuous f
      hs : Membership.mem (nhds (f x)) s
      hKc : IsCompact K
      hKx : Membership.mem (nhds x) K
      hc : IsCompact (SDiff.sdiff (Set.image f K) (interior s))
      U V : Set Y
      Uo : IsOpen U
      Vo : IsOpen V
      hxU : HasSubset.Subset (Singleton.singleton (f x)) U
      hV : HasSubset.Subset (SDiff.sdiff (Set.image f K) (interior s)) V
      hd : Disjoint U V
      y : X
      hy : Membership.mem (SDiff.sdiff K (Set.preimage f V)) y
      ⊢ Membership.mem s (f y)
    -/
  · by_contra hys
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      X : Type u_3
      Y : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : R1Space Y
      f : X → Y
      x : X
      K : Set X
      s : Set Y
      hf : Continuous f
      hs : Membership.mem (nhds (f x)) s
      hKc : IsCompact K
      hKx : Membership.mem (nhds x) K
      hc : IsCompact (SDiff.sdiff (Set.image f K) (interior s))
      U V : Set Y
      Uo : IsOpen U
      Vo : IsOpen V
      hxU : HasSubset.Subset (Singleton.singleton (f x)) U
      hV : HasSubset.Subset (SDiff.sdiff (Set.image f K) (interior s)) V
      hd : Disjoint U V
      y : X
      hy : Membership.mem (SDiff.sdiff K (Set.preimage f V)) y
      hys : Not (Membership.mem s (f y))
      ⊢ False
    -/
    exact hy.2 (hV ⟨mem_image_of_mem _ hy.1, not_mem_subset interior_subset hys⟩)
    /-
      🎉 no goals
    -/


instance (priority := 900) {X Y : Type*} [TopologicalSpace X] [WeaklyLocallyCompactSpace X]
    [TopologicalSpace Y] [R1Space Y] : LocallyCompactPair X Y where
  exists_mem_nhds_isCompact_mapsTo hf hs :=
    let ⟨_K, hKc, hKx⟩ := exists_compact_mem_nhds _
    exists_mem_nhds_isCompact_mapsTo_of_isCompact_mem_nhds hf hs hKc hKx


/-- If a point in an R₁ space has a compact neighborhood,
then it has a basis of compact closed neighborhoods. -/
theorem IsCompact.isCompact_isClosed_basis_nhds {x : X} {L : Set X} (hLc : IsCompact L)
    (hxL : L ∈ 𝓝 x) : (𝓝 x).HasBasis (fun K ↦ K ∈ 𝓝 x ∧ IsCompact K ∧ IsClosed K) (·) :=
  hasBasis_self.2 fun _U hU ↦
    let ⟨K, hKx, hKc, hKU⟩ := exists_mem_nhds_isCompact_mapsTo_of_isCompact_mem_nhds
      continuous_id (interior_mem_nhds.2 hU) hLc hxL
    ⟨closure K, mem_of_superset hKx subset_closure, ⟨hKc.closure, isClosed_closure⟩,
      (hKc.closure_subset_of_isOpen isOpen_interior hKU).trans interior_subset⟩


/-- In an R₁ space, the filters `coclosedCompact` and `cocompact` are equal. -/
@[simp]
theorem Filter.coclosedCompact_eq_cocompact : coclosedCompact X = cocompact X := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    ⊢ Eq (Filter.coclosedCompact X) (Filter.cocompact X)
  -/
  refine le_antisymm ?_ cocompact_le_coclosedCompact
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    ⊢ LE.le (Filter.coclosedCompact X) (Filter.cocompact X)
  -/
  rw [hasBasis_coclosedCompact.le_basis_iff hasBasis_cocompact]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : R1Space X
    ⊢ ∀ (i' : Set X), IsCompact i' → Exists fun i => And (And (IsClosed i) (IsComp …
  -/
  exact fun K hK ↦ ⟨closure K, ⟨isClosed_closure, hK.closure⟩, compl_subset_compl.2 subset_closure⟩
  /-
    🎉 no goals
  -/


/-- In an R₁ space, the bornologies `relativelyCompact` and `inCompact` are equal. -/
@[simp]
theorem Bornology.relativelyCompact_eq_inCompact :
    Bornology.relativelyCompact X = Bornology.inCompact X :=
  Bornology.ext _ _ Filter.coclosedCompact_eq_cocompact


/-- In a (weakly) locally compact R₁ space, compact closed neighborhoods of a point `x`
form a basis of neighborhoods of `x`. -/
theorem isCompact_isClosed_basis_nhds (x : X) :
    (𝓝 x).HasBasis (fun K => K ∈ 𝓝 x ∧ IsCompact K ∧ IsClosed K) (·) :=
  let ⟨_L, hLc, hLx⟩ := exists_compact_mem_nhds x
  hLc.isCompact_isClosed_basis_nhds hLx


/-- In a (weakly) locally compact R₁ space, each point admits a compact closed neighborhood. -/
theorem exists_mem_nhds_isCompact_isClosed (x : X) : ∃ K ∈ 𝓝 x, IsCompact K ∧ IsClosed K :=
  (isCompact_isClosed_basis_nhds x).ex_mem

-- see Note [lower instance priority]

/-- A weakly locally compact R₁ space is locally compact. -/
instance (priority := 80) WeaklyLocallyCompactSpace.locallyCompactSpace : LocallyCompactSpace X :=
  .of_hasBasis isCompact_isClosed_basis_nhds fun _ _ ⟨_, h, _⟩ ↦ h


/-- In a weakly locally compact R₁ space,
every compact set has an open neighborhood with compact closure. -/
theorem exists_isOpen_superset_and_isCompact_closure {K : Set X} (hK : IsCompact K) :
    ∃ V, IsOpen V ∧ K ⊆ V ∧ IsCompact (closure V) := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : R1Space X
    inst✝ : WeaklyLocallyCompactSpace X
    K : Set X
    hK : IsCompact K
    ⊢ Exists fun V => And (IsOpen V) (And (HasSubset.Subset K V) (IsCompact (closu …
  -/
  rcases exists_compact_superset hK with ⟨K', hK', hKK'⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : R1Space X
    inst✝ : WeaklyLocallyCompactSpace X
    K : Set X
    hK : IsCompact K
    K' : Set X
    hK' : IsCompact K'
    hKK' : HasSubset.Subset K (interior K')
    ⊢ Exists fun V => And (IsOpen V) (And (HasSubset.Subset K V) (IsCompact (closu …
  -/
  exact ⟨interior K', isOpen_interior, hKK', hK'.closure_of_subset interior_subset⟩
  /-
    🎉 no goals
  -/


/-- In a weakly locally compact R₁ space,
every point has an open neighborhood with compact closure. -/
theorem exists_isOpen_mem_isCompact_closure (x : X) :
    ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ IsCompact (closure U) := by
  simpa only [singleton_subset_iff]
    using exists_isOpen_superset_and_isCompact_closure isCompact_singleton


