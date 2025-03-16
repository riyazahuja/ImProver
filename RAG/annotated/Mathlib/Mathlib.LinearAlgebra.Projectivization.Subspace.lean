/-- A subspace of a projective space is a structure consisting of a set of points such that:
If two nonzero vectors determine points which are in the set, and the sum of the two vectors is
nonzero, then the point determined by the sum is also in the set. -/
@[ext]
structure Subspace where
  /-- The set of points. -/
  carrier : Set (ℙ K V)
  /-- The addition rule. -/
  mem_add' (v w : V) (hv : v ≠ 0) (hw : w ≠ 0) (hvw : v + w ≠ 0) :
    mk K v hv ∈ carrier → mk K w hw ∈ carrier → mk K (v + w) hvw ∈ carrier


instance : SetLike (Subspace K V) (ℙ K V) where
  coe := carrier
  coe_injective' A B := by
    /-
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      A B : Projectivization.Subspace K V
      ⊢ Eq A.carrier B.carrier → Eq A B
    -/
    cases A
    /-
      case mk
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : Projectivization.Subspace K V
      carrier✝ : Set (Projectivization K V)
      mem_add'✝ : ∀ (v w : V) (hv : Ne v 0) (hw : Ne w 0) (hvw : Ne (HAdd.hAdd v w)  …
      ⊢ Eq { carrier := carrier✝, mem_add' := mem_add'✝ }.carrier B.carrier → Eq { c …
    -/
    cases B
    /-
      case mk.mk
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      carrier✝¹ : Set (Projectivization K V)
      mem_add'✝¹ : ∀ (v w : V) (hv : Ne v 0) (hw : Ne w 0) (hvw : Ne (HAdd.hAdd v w) …
      carrier✝ : Set (Projectivization K V)
      mem_add'✝ : ∀ (v w : V) (hv : Ne v 0) (hw : Ne w 0) (hvw : Ne (HAdd.hAdd v w)  …
      ⊢ Eq { carrier := carrier✝¹, mem_add' := mem_add'✝¹ }.carrier { carrier := car …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_carrier_iff (A : Subspace K V) (x : ℙ K V) : x ∈ A.carrier ↔ x ∈ A :=
  Iff.refl _


theorem mem_add (T : Subspace K V) (v w : V) (hv : v ≠ 0) (hw : w ≠ 0) (hvw : v + w ≠ 0) :
    Projectivization.mk K v hv ∈ T →
      Projectivization.mk K w hw ∈ T → Projectivization.mk K (v + w) hvw ∈ T :=
  T.mem_add' v w hv hw hvw


/-- The span of a set of points in a projective space is defined inductively to be the set of points
which contains the original set, and contains all points determined by the (nonzero) sum of two
nonzero vectors, each of which determine points in the span. -/
inductive spanCarrier (S : Set (ℙ K V)) : Set (ℙ K V)
  | of (x : ℙ K V) (hx : x ∈ S) : spanCarrier S x
  | mem_add (v w : V) (hv : v ≠ 0) (hw : w ≠ 0) (hvw : v + w ≠ 0) :
      spanCarrier S (Projectivization.mk K v hv) →
      spanCarrier S (Projectivization.mk K w hw) → spanCarrier S (Projectivization.mk K (v + w) hvw)


/-- The span of a set of points in projective space is a subspace. -/
def span (S : Set (ℙ K V)) : Subspace K V where
  carrier := spanCarrier S
  mem_add' v w hv hw hvw := spanCarrier.mem_add v w hv hw hvw


/-- The span of a set of points contains the set of points. -/
theorem subset_span (S : Set (ℙ K V)) : S ⊆ span S := fun _x hx => spanCarrier.of _ hx


/-- The span of a set of points is a Galois insertion between sets of points of a projective space
and subspaces of the projective space. -/
def gi : GaloisInsertion (span : Set (ℙ K V) → Subspace K V) SetLike.coe where
  choice S _hS := span S
  gc A B :=
    ⟨fun h => le_trans (subset_span _) h, by
      /-
        K : Type u_1
        V : Type u_2
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        A : Set (Projectivization K V)
        B : Projectivization.Subspace K V
        ⊢ LE.le A ↑B → LE.le (Projectivization.Subspace.span A) B
      -/
      intro h x hx
      /-
        K : Type u_1
        V : Type u_2
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        A : Set (Projectivization K V)
        B : Projectivization.Subspace K V
        h : LE.le A ↑B
        x : Projectivization K V
        hx : Membership.mem (Projectivization.Subspace.span A) x
        ⊢ Membership.mem B x
      -/
      induction' hx with y hy
        /-
          case of
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          A : Set (Projectivization K V)
          B : Projectivization.Subspace K V
          h : LE.le A ↑B
          x y : Projectivization K V
          hy : Membership.mem A y
          ⊢ Membership.mem B y
        -/
      · apply h
        /-
          case of.a
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          A : Set (Projectivization K V)
          B : Projectivization.Subspace K V
          h : LE.le A ↑B
          x y : Projectivization K V
          hy : Membership.mem A y
          ⊢ Membership.mem A y
        -/
        assumption
        /-
          🎉 no goals
        -/
        /-
          case mem_add
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          A : Set (Projectivization K V)
          B : Projectivization.Subspace K V
          h : LE.le A ↑B
          x : Projectivization K V
          v✝ w✝ : V
          hv✝ : Ne v✝ 0
          hw✝ : Ne w✝ 0
          hvw✝ : Ne (HAdd.hAdd v✝ w✝) 0
          a✝¹ : Projectivization.Subspace.spanCarrier A (Projectivization.mk K v✝ hv✝)
          a✝ : Projectivization.Subspace.spanCarrier A (Projectivization.mk K w✝ hw✝)
          a_ih✝¹ : Membership.mem B (Projectivization.mk K v✝ hv✝)
          a_ih✝ : Membership.mem B (Projectivization.mk K w✝ hw✝)
          ⊢ Membership.mem B (Projectivization.mk K (HAdd.hAdd v✝ w✝) hvw✝)
        -/
      · apply B.mem_add
        /-
          case mem_add.a
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          A : Set (Projectivization K V)
          B : Projectivization.Subspace K V
          h : LE.le A ↑B
          x : Projectivization K V
          v✝ w✝ : V
          hv✝ : Ne v✝ 0
          hw✝ : Ne w✝ 0
          hvw✝ : Ne (HAdd.hAdd v✝ w✝) 0
          a✝¹ : Projectivization.Subspace.spanCarrier A (Projectivization.mk K v✝ hv✝)
          a✝ : Projectivization.Subspace.spanCarrier A (Projectivization.mk K w✝ hw✝)
          a_ih✝¹ : Membership.mem B (Projectivization.mk K v✝ hv✝)
          a_ih✝ : Membership.mem B (Projectivization.mk K w✝ hw✝)
          ⊢ Membership.mem B (Projectivization.mk K v✝ ?mem_add.hv)
        -/
        assumption'⟩
        /-
          🎉 no goals
        -/
  le_l_u _ := subset_span _
  choice_eq _ _ := rfl


/-- The span of a subspace is the subspace. -/
@[simp]
theorem span_coe (W : Subspace K V) : span ↑W = W :=
  GaloisInsertion.l_u_eq gi W


/-- The infimum of two subspaces exists. -/
instance instInf : Min (Subspace K V) :=
  ⟨fun A B =>
    ⟨A ⊓ B, fun _v _w hv hw _hvw h1 h2 =>
      ⟨A.mem_add _ _ hv hw _ h1.1 h2.1, B.mem_add _ _ hv hw _ h1.2 h2.2⟩⟩⟩

-- Porting note: delete the name of this instance since it causes problem since hasInf is already
-- defined above

/-- Infimums of arbitrary collections of subspaces exist. -/
instance instInfSet : InfSet (Subspace K V) :=
  ⟨fun A =>
    ⟨sInf (SetLike.coe '' A), fun v w hv hw hvw h1 h2 t => by
      /-
        K : Type u_1
        V : Type u_2
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        A : Set (Projectivization.Subspace K V)
        v w : V
        hv : Ne v 0
        hw : Ne w 0
        hvw : Ne (HAdd.hAdd v w) 0
        h1 : Membership.mem (InfSet.sInf (Set.image SetLike.coe A)) (Projectivization. …
        h2 : Membership.mem (InfSet.sInf (Set.image SetLike.coe A)) (Projectivization. …
        t : Set (Projectivization K V)
        ⊢ Membership.mem (Set.image SetLike.coe A) t → Membership.mem t (Projectivizat …
      -/
      rintro ⟨s, hs, rfl⟩
      /-
        case intro.intro
        K : Type u_1
        V : Type u_2
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        A : Set (Projectivization.Subspace K V)
        v w : V
        hv : Ne v 0
        hw : Ne w 0
        hvw : Ne (HAdd.hAdd v w) 0
        h1 : Membership.mem (InfSet.sInf (Set.image SetLike.coe A)) (Projectivization. …
        h2 : Membership.mem (InfSet.sInf (Set.image SetLike.coe A)) (Projectivization. …
        s : Projectivization.Subspace K V
        hs : Membership.mem A s
        ⊢ Membership.mem (↑s) (Projectivization.mk K (HAdd.hAdd v w) hvw)
      -/
      exact s.mem_add v w hv hw _ (h1 s ⟨s, hs, rfl⟩) (h2 s ⟨s, hs, rfl⟩)⟩⟩
      /-
        🎉 no goals
      -/


/-- The subspaces of a projective space form a complete lattice. -/
instance : CompleteLattice (Subspace K V) :=
  { __ := completeLatticeOfInf (Subspace K V)
      (by
        /-
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          ⊢ ∀ (s : Set (Projectivization.Subspace K V)), IsGLB s (InfSet.sInf s)
        -/
        refine fun s => ⟨fun a ha x hx => hx _ ⟨a, ha, rfl⟩, fun a ha x hx E => ?_⟩
        /-
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          s : Set (Projectivization.Subspace K V)
          a : Projectivization.Subspace K V
          ha : Membership.mem (lowerBounds s) a
          x : Projectivization K V
          hx : Membership.mem a x
          E : Set (Projectivization K V)
          ⊢ Membership.mem (Set.image SetLike.coe s) E → Membership.mem E x
        -/
        rintro ⟨E, hE, rfl⟩
        /-
          case intro.intro
          K : Type u_1
          V : Type u_2
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          s : Set (Projectivization.Subspace K V)
          a : Projectivization.Subspace K V
          ha : Membership.mem (lowerBounds s) a
          x : Projectivization K V
          hx : Membership.mem a x
          E : Projectivization.Subspace K V
          hE : Membership.mem s E
          ⊢ Membership.mem (↑E) x
        -/
        exact ha hE hx)
        /-
          🎉 no goals
        -/
    inf_le_left := fun A B _ hx => (@inf_le_left _ _ A B) hx
    inf_le_right := fun A B _ hx => (@inf_le_right _ _ A B) hx
    le_inf := fun _ _ _ h1 h2 _ hx => (le_inf h1 h2) hx }


instance subspaceInhabited : Inhabited (Subspace K V) where default := ⊤


/-- The span of the empty set is the bottom of the lattice of subspaces. -/
@[simp]
theorem span_empty : span (∅ : Set (ℙ K V)) = ⊥ := gi.gc.l_bot


/-- The span of the entire projective space is the top of the lattice of subspaces. -/
@[simp]
theorem span_univ : span (Set.univ : Set (ℙ K V)) = ⊤ := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Eq (Projectivization.Subspace.span Set.univ) Top.top
  -/
  rw [eq_top_iff, SetLike.le_def]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ ∀ ⦃x : Projectivization K V⦄, Membership.mem Top.top x → Membership.mem (Pro …
  -/
  intro x _hx
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : Projectivization K V
    _hx : Membership.mem Top.top x
    ⊢ Membership.mem (Projectivization.Subspace.span Set.univ) x
  -/
  exact subset_span _ (Set.mem_univ x)
  /-
    🎉 no goals
  -/


/-- The span of a set of points is contained in a subspace if and only if the set of points is
contained in the subspace. -/
theorem span_le_subspace_iff {S : Set (ℙ K V)} {W : Subspace K V} : span S ≤ W ↔ S ⊆ W :=
  gi.gc S W


/-- If a set of points is a subset of another set of points, then its span will be contained in the
span of that set. -/
@[mono]
theorem monotone_span : Monotone (span : Set (ℙ K V) → Subspace K V) :=
  gi.gc.monotone_l


@[gcongr]
lemma span_le_span {s t : Set (ℙ K V)} (hst : s ⊆ t) : span s ≤ span t := monotone_span hst


theorem subset_span_trans {S T U : Set (ℙ K V)} (hST : S ⊆ span T) (hTU : T ⊆ span U) :
    S ⊆ span U :=
  gi.gc.le_u_l_trans hST hTU


/-- The supremum of two subspaces is equal to the span of their union. -/
theorem span_union (S T : Set (ℙ K V)) : span (S ∪ T) = span S ⊔ span T :=
  (@gi K V _ _ _).gc.l_sup


/-- The supremum of a collection of subspaces is equal to the span of the union of the
collection. -/
theorem span_iUnion {ι} (s : ι → Set (ℙ K V)) : span (⋃ i, s i) = ⨆ i, span (s i) :=
  (@gi K V _ _ _).gc.l_iSup


/-- The supremum of a subspace and the span of a set of points is equal to the span of the union of
the subspace and the set of points. -/
theorem sup_span {S : Set (ℙ K V)} {W : Subspace K V} : W ⊔ span S = span (W ∪ S) := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    W : Projectivization.Subspace K V
    ⊢ Eq (Max.max W (Projectivization.Subspace.span S)) (Projectivization.Subspace …
  -/
  rw [span_union, span_coe]
  /-
    🎉 no goals
  -/


theorem span_sup {S : Set (ℙ K V)} {W : Subspace K V} : span S ⊔ W = span (S ∪ W) := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    W : Projectivization.Subspace K V
    ⊢ Eq (Max.max (Projectivization.Subspace.span S) W) (Projectivization.Subspace …
  -/
  rw [span_union, span_coe]
  /-
    🎉 no goals
  -/


/-- A point in a projective space is contained in the span of a set of points if and only if the
point is contained in all subspaces of the projective space which contain the set of points. -/
theorem mem_span {S : Set (ℙ K V)} (u : ℙ K V) :
    u ∈ span S ↔ ∀ W : Subspace K V, S ⊆ W → u ∈ W := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    u : Projectivization K V
    ⊢ Iff (Membership.mem (Projectivization.Subspace.span S) u) (∀ (W : Projectivi …
  -/
  simp_rw [← span_le_subspace_iff]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    u : Projectivization K V
    ⊢ Iff (Membership.mem (Projectivization.Subspace.span S) u) (∀ (W : Projectivi …
  -/
  exact ⟨fun hu W hW => hW hu, fun W => W (span S) (le_refl _)⟩
  /-
    🎉 no goals
  -/


/-- The span of a set of points in a projective space is equal to the infimum of the collection of
subspaces which contain the set. -/
theorem span_eq_sInf {S : Set (ℙ K V)} : span S = sInf { W : Subspace K V| S ⊆ W } := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    ⊢ Eq (Projectivization.Subspace.span S) (InfSet.sInf (setOf fun W => HasSubset …
  -/
  ext x
  /-
    case carrier.h
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    x : Projectivization K V
    ⊢ Iff (Membership.mem (Projectivization.Subspace.span S).carrier x) (Membershi …
  -/
  simp_rw [mem_carrier_iff, mem_span x]
  /-
    case carrier.h
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S : Set (Projectivization K V)
    x : Projectivization K V
    ⊢ Iff (∀ (W : Projectivization.Subspace K V), HasSubset.Subset S ↑W → Membersh …
  -/
  refine ⟨fun hx => ?_, fun hx W hW => ?_⟩
    /-
      case carrier.h.refine_1
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      S : Set (Projectivization K V)
      x : Projectivization K V
      hx : ∀ (W : Projectivization.Subspace K V), HasSubset.Subset S ↑W → Membership …
      ⊢ Membership.mem (InfSet.sInf (setOf fun W => HasSubset.Subset S ↑W)) x
    -/
  · rintro W ⟨T, hT, rfl⟩
    /-
      case carrier.h.refine_1.intro.intro
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      S : Set (Projectivization K V)
      x : Projectivization K V
      hx : ∀ (W : Projectivization.Subspace K V), HasSubset.Subset S ↑W → Membership …
      T : Projectivization.Subspace K V
      hT : Membership.mem (setOf fun W => HasSubset.Subset S ↑W) T
      ⊢ Membership.mem (↑T) x
    -/
    exact hx T hT
    /-
      🎉 no goals
    -/
    /-
      case carrier.h.refine_2
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      S : Set (Projectivization K V)
      x : Projectivization K V
      hx : Membership.mem (InfSet.sInf (setOf fun W => HasSubset.Subset S ↑W)) x
      W : Projectivization.Subspace K V
      hW : HasSubset.Subset S ↑W
      ⊢ Membership.mem W x
    -/
  · exact (@sInf_le _ _ { W : Subspace K V | S ⊆ ↑W } W hW) hx
    /-
      🎉 no goals
    -/


/-- If a set of points in projective space is contained in a subspace, and that subspace is
contained in the span of the set of points, then the span of the set of points is equal to
the subspace. -/
theorem span_eq_of_le {S : Set (ℙ K V)} {W : Subspace K V} (hS : S ⊆ W) (hW : W ≤ span S) :
    span S = W :=
  le_antisymm (span_le_subspace_iff.mpr hS) hW


/-- The spans of two sets of points in a projective space are equal if and only if each set of
points is contained in the span of the other set. -/
theorem span_eq_span_iff {S T : Set (ℙ K V)} : span S = span T ↔ S ⊆ span T ∧ T ⊆ span S :=
  ⟨fun h => ⟨h ▸ subset_span S, h.symm ▸ subset_span T⟩, fun h =>
    le_antisymm (span_le_subspace_iff.2 h.1) (span_le_subspace_iff.2 h.2)⟩


