/-- A simplicial complex in a `𝕜`-module is a collection of simplices which glue nicely together.
Note that the textbook meaning of "glue nicely" is given in
`Geometry.SimplicialComplex.disjoint_or_exists_inter_eq_convexHull`. It is mostly useless, as
`Geometry.SimplicialComplex.convexHull_inter_convexHull` is enough for all purposes. -/
@[ext]
structure SimplicialComplex where
  /-- the faces of this simplicial complex: currently, given by their spanning vertices -/
  faces : Set (Finset E)
  /-- the empty set is not a face: hence, all faces are non-empty -/
  not_empty_mem : ∅ ∉ faces
  /-- the vertices in each face are affine independent: this is an implementation detail -/
  indep : ∀ {s}, s ∈ faces → AffineIndependent 𝕜 ((↑) : s → E)
  /-- faces are downward closed: a non-empty subset of its spanning vertices spans another face -/
  down_closed : ∀ {s t}, s ∈ faces → t ⊆ s → t ≠ ∅ → t ∈ faces
  inter_subset_convexHull : ∀ {s t}, s ∈ faces → t ∈ faces →
    convexHull 𝕜 ↑s ∩ convexHull 𝕜 ↑t ⊆ convexHull 𝕜 (s ∩ t : Set E)


/-- A `Finset` belongs to a `SimplicialComplex` if it's a face of it. -/
instance : Membership (Finset E) (SimplicialComplex 𝕜 E) :=
  ⟨fun K s => s ∈ K.faces⟩


/-- The underlying space of a simplicial complex is the union of its faces. -/
def space (K : SimplicialComplex 𝕜 E) : Set E :=
  ⋃ s ∈ K.faces, convexHull 𝕜 (s : Set E)

-- Porting note: Expanded `∃ s ∈ K.faces` to get the type to match more closely with Lean 3

theorem mem_space_iff : x ∈ K.space ↔ ∃ s ∈ K.faces, x ∈ convexHull 𝕜 (s : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    x : E
    ⊢ Iff (Membership.mem K.space x) (Exists fun s => And (Membership.mem K.faces  …
  -/
  simp [space]
  /-
    🎉 no goals
  -/

-- Porting note: Original proof was `:= subset_biUnion_of_mem hs`

theorem convexHull_subset_space (hs : s ∈ K.faces) : convexHull 𝕜 ↑s ⊆ K.space := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    s : Finset E
    hs : Membership.mem K.faces s
    ⊢ HasSubset.Subset ((convexHull 𝕜) ↑s) K.space
  -/
  convert subset_biUnion_of_mem hs
  /-
    case h.e'_3
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    s : Finset E
    hs : Membership.mem K.faces s
    ⊢ Eq ((convexHull 𝕜) ↑s) ((convexHull 𝕜) ↑s)
  -/
  rfl
  /-
    🎉 no goals
  -/


protected theorem subset_space (hs : s ∈ K.faces) : (s : Set E) ⊆ K.space :=
  (subset_convexHull 𝕜 _).trans <| convexHull_subset_space hs


theorem convexHull_inter_convexHull (hs : s ∈ K.faces) (ht : t ∈ K.faces) :
    convexHull 𝕜 ↑s ∩ convexHull 𝕜 ↑t = convexHull 𝕜 (s ∩ t : Set E) :=
  (K.inter_subset_convexHull hs ht).antisymm <|
    subset_inter (convexHull_mono Set.inter_subset_left) <|
      convexHull_mono Set.inter_subset_right


/-- The conclusion is the usual meaning of "glue nicely" in textbooks. It turns out to be quite
unusable, as it's about faces as sets in space rather than simplices. Further, additional structure
on `𝕜` means the only choice of `u` is `s ∩ t` (but it's hard to prove). -/
theorem disjoint_or_exists_inter_eq_convexHull (hs : s ∈ K.faces) (ht : t ∈ K.faces) :
    Disjoint (convexHull 𝕜 (s : Set E)) (convexHull 𝕜 ↑t) ∨
      ∃ u ∈ K.faces, convexHull 𝕜 (s : Set E) ∩ convexHull 𝕜 ↑t = convexHull 𝕜 ↑u := by
  classical
  by_contra! h
  refine h.2 (s ∩ t) (K.down_closed hs inter_subset_left fun hst => h.1 <|
    disjoint_iff_inf_le.mpr <| (K.inter_subset_convexHull hs ht).trans ?_) ?_
  · rw [← coe_inter, hst, coe_empty, convexHull_empty]
    rfl
  · rw [coe_inter, convexHull_inter_convexHull hs ht]


/-- Construct a simplicial complex by removing the empty face for you. -/
@[simps]
def ofErase (faces : Set (Finset E)) (indep : ∀ s ∈ faces, AffineIndependent 𝕜 ((↑) : s → E))
    (down_closed : ∀ s ∈ faces, ∀ t ⊆ s, t ∈ faces)
    (inter_subset_convexHull : ∀ᵉ (s ∈ faces) (t ∈ faces),
      convexHull 𝕜 ↑s ∩ convexHull 𝕜 ↑t ⊆ convexHull 𝕜 (s ∩ t : Set E)) :
    SimplicialComplex 𝕜 E where
  faces := faces \ {∅}
  not_empty_mem h := h.2 (mem_singleton _)
  indep hs := indep _ hs.1
  down_closed hs hts ht := ⟨down_closed _ hs.1 _ hts, ht⟩
  inter_subset_convexHull hs ht := inter_subset_convexHull _ hs.1 _ ht.1


/-- Construct a simplicial complex as a subset of a given simplicial complex. -/
@[simps]
def ofSubcomplex (K : SimplicialComplex 𝕜 E) (faces : Set (Finset E)) (subset : faces ⊆ K.faces)
    (down_closed : ∀ {s t}, s ∈ faces → t ⊆ s → t ∈ faces) : SimplicialComplex 𝕜 E :=
  { faces
    not_empty_mem := fun h => K.not_empty_mem (subset h)
    indep := fun hs => K.indep (subset hs)
    down_closed := fun hs hts _ => down_closed hs hts
    inter_subset_convexHull := fun hs ht => K.inter_subset_convexHull (subset hs) (subset ht) }


/-- The vertices of a simplicial complex are its zero dimensional faces. -/
def vertices (K : SimplicialComplex 𝕜 E) : Set E :=
  { x | {x} ∈ K.faces }


theorem mem_vertices : x ∈ K.vertices ↔ {x} ∈ K.faces := Iff.rfl


theorem vertices_eq : K.vertices = ⋃ k ∈ K.faces, (k : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    ⊢ Eq K.vertices (Set.iUnion fun k => Set.iUnion fun h => ↑k)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    x : E
    ⊢ Iff (Membership.mem K.vertices x) (Membership.mem (Set.iUnion fun k => Set.i …
  -/
  refine ⟨fun h => mem_biUnion h <| mem_coe.2 <| mem_singleton_self x, fun h => ?_⟩
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    x : E
    h : Membership.mem (Set.iUnion fun k => Set.iUnion fun h => ↑k) x
    ⊢ Membership.mem K.vertices x
  -/
  obtain ⟨s, hs, hx⟩ := mem_iUnion₂.1 h
  /-
    case h.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    x : E
    h : Membership.mem (Set.iUnion fun k => Set.iUnion fun h => ↑k) x
    s : Finset E
    hs : Membership.mem K.faces s
    hx : Membership.mem (↑s) x
    ⊢ Membership.mem K.vertices x
  -/
  exact K.down_closed hs (Finset.singleton_subset_iff.2 <| mem_coe.1 hx) (singleton_ne_empty _)
  /-
    🎉 no goals
  -/


theorem vertices_subset_space : K.vertices ⊆ K.space :=
  vertices_eq.subset.trans <| iUnion₂_mono fun x _ => subset_convexHull 𝕜 (x : Set E)


theorem vertex_mem_convexHull_iff (hx : x ∈ K.vertices) (hs : s ∈ K.faces) :
    x ∈ convexHull 𝕜 (s : Set E) ↔ x ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    s : Finset E
    x : E
    hx : Membership.mem K.vertices x
    hs : Membership.mem K.faces s
    ⊢ Iff (Membership.mem ((convexHull 𝕜) ↑s) x) (Membership.mem s x)
  -/
  refine ⟨fun h => ?_, fun h => subset_convexHull 𝕜 _ h⟩
  classical
  have h := K.inter_subset_convexHull hx hs ⟨by simp, h⟩
  by_contra H
  rwa [← coe_inter, Finset.disjoint_iff_inter_eq_empty.1 (Finset.disjoint_singleton_right.2 H).symm,
    coe_empty, convexHull_empty] at h


/-- A face is a subset of another one iff its vertices are. -/
theorem face_subset_face_iff (hs : s ∈ K.faces) (ht : t ∈ K.faces) :
    convexHull 𝕜 (s : Set E) ⊆ convexHull 𝕜 ↑t ↔ s ⊆ t :=
  ⟨fun h _ hxs =>
    (vertex_mem_convexHull_iff
          (K.down_closed hs (Finset.singleton_subset_iff.2 hxs) <| singleton_ne_empty _) ht).1
      (h (subset_convexHull 𝕜 (E := E) s hxs)),
    convexHull_mono⟩


/-- A facet of a simplicial complex is a maximal face. -/
def facets (K : SimplicialComplex 𝕜 E) : Set (Finset E) :=
  { s ∈ K.faces | ∀ ⦃t⦄, t ∈ K.faces → s ⊆ t → s = t }


theorem mem_facets : s ∈ K.facets ↔ s ∈ K.faces ∧ ∀ t ∈ K.faces, s ⊆ t → s = t :=
  mem_sep_iff


theorem facets_subset : K.facets ⊆ K.faces := fun _ hs => hs.1


theorem not_facet_iff_subface (hs : s ∈ K.faces) : s ∉ K.facets ↔ ∃ t, t ∈ K.faces ∧ s ⊂ t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    K : Geometry.SimplicialComplex 𝕜 E
    s : Finset E
    hs : Membership.mem K.faces s
    ⊢ Iff (Not (Membership.mem K.facets s)) (Exists fun t => And (Membership.mem K …
  -/
  refine ⟨fun hs' : ¬(_ ∧ _) => ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs : Membership.mem K.faces s
      hs' : Not (And (Membership.mem K.faces s) (∀ ⦃t : Finset E⦄, Membership.mem K. …
      ⊢ Exists fun t => And (Membership.mem K.faces t) (HasSSubset.SSubset s t)
    -/
  · push_neg at hs'
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs : Membership.mem K.faces s
      hs' : Membership.mem K.faces s → Exists fun ⦃t⦄ => And (Membership.mem K.faces …
      ⊢ Exists fun t => And (Membership.mem K.faces t) (HasSSubset.SSubset s t)
    -/
    obtain ⟨t, ht⟩ := hs' hs
    /-
      case refine_1.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs : Membership.mem K.faces s
      hs' : Membership.mem K.faces s → Exists fun ⦃t⦄ => And (Membership.mem K.faces …
      t : Finset E
      ht : And (Membership.mem K.faces t) (And (HasSubset.Subset s t) (Ne s t))
      ⊢ Exists fun t => And (Membership.mem K.faces t) (HasSSubset.SSubset s t)
    -/
    exact ⟨t, ht.1, ⟨ht.2.1, fun hts => ht.2.2 (Subset.antisymm ht.2.1 hts)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs : Membership.mem K.faces s
      ⊢ (Exists fun t => And (Membership.mem K.faces t) (HasSSubset.SSubset s t)) →  …
    -/
  · rintro ⟨t, ht⟩ ⟨hs, hs'⟩
    /-
      case refine_2.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs✝ : Membership.mem K.faces s
      t : Finset E
      ht : And (Membership.mem K.faces t) (HasSSubset.SSubset s t)
      hs : Membership.mem K.faces s
      hs' : ∀ ⦃t : Finset E⦄, Membership.mem K.faces t → HasSubset.Subset s t → Eq s t
      ⊢ False
    -/
    have := hs' ht.1 ht.2.1
    /-
      case refine_2.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs✝ : Membership.mem K.faces s
      t : Finset E
      ht : And (Membership.mem K.faces t) (HasSSubset.SSubset s t)
      hs : Membership.mem K.faces s
      hs' : ∀ ⦃t : Finset E⦄, Membership.mem K.faces t → HasSubset.Subset s t → Eq s t
      this : Eq s t
      ⊢ False
    -/
    rw [this] at ht
    /-
      case refine_2.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      K : Geometry.SimplicialComplex 𝕜 E
      s : Finset E
      hs✝ : Membership.mem K.faces s
      t : Finset E
      ht : And (Membership.mem K.faces t) (HasSSubset.SSubset t t)
      hs : Membership.mem K.faces s
      hs' : ∀ ⦃t : Finset E⦄, Membership.mem K.faces t → HasSubset.Subset s t → Eq s t
      this : Eq s t
      ⊢ False
    -/
    exact ht.2.2 (Subset.refl t)
    /-
      🎉 no goals
    -/


/-- The complex consisting of only the faces present in both of its arguments. -/
instance : Min (SimplicialComplex 𝕜 E) :=
  ⟨fun K L =>
    { faces := K.faces ∩ L.faces
      not_empty_mem := fun h => K.not_empty_mem (Set.inter_subset_left h)
      indep := fun hs => K.indep hs.1
      down_closed := fun hs hst ht => ⟨K.down_closed hs.1 hst ht, L.down_closed hs.2 hst ht⟩
      inter_subset_convexHull := fun hs ht => K.inter_subset_convexHull hs.1 ht.1 }⟩


instance : SemilatticeInf (SimplicialComplex 𝕜 E) :=
  { PartialOrder.lift faces (fun _ _ => SimplicialComplex.ext) with
    inf := (· ⊓ ·)
    inf_le_left := fun _ _ _ hs => hs.1
    inf_le_right := fun _ _ _ hs => hs.2
    le_inf := fun _ _ _ hKL hKM _ hs => ⟨hKL hs, hKM hs⟩ }


instance hasBot : Bot (SimplicialComplex 𝕜 E) :=
  ⟨{  faces := ∅
      not_empty_mem := Set.not_mem_empty ∅
      indep := fun hs => (Set.not_mem_empty _ hs).elim
      down_closed := fun hs => (Set.not_mem_empty _ hs).elim
      inter_subset_convexHull := fun hs => (Set.not_mem_empty _ hs).elim }⟩


instance : OrderBot (SimplicialComplex 𝕜 E) :=
  { SimplicialComplex.hasBot 𝕜 E with bot_le := fun _ => Set.empty_subset _ }


instance : Inhabited (SimplicialComplex 𝕜 E) :=
  ⟨⊥⟩


theorem faces_bot : (⊥ : SimplicialComplex 𝕜 E).faces = ∅ := rfl


theorem space_bot : (⊥ : SimplicialComplex 𝕜 E).space = ∅ :=
  Set.biUnion_empty _


theorem facets_bot : (⊥ : SimplicialComplex 𝕜 E).facets = ∅ :=
  eq_empty_of_subset_empty facets_subset


