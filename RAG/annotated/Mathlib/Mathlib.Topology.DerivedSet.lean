theorem AccPt.map {β : Type*} [TopologicalSpace β] {F : Filter X} {x : X}
    (h : AccPt x F) {f : X → β} (hf1 : ContinuousAt f x) (hf2 : Function.Injective f) :
    AccPt (f x) (map f F) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    F : Filter X
    x : X
    h : AccPt x F
    f : X → β
    hf1 : ContinuousAt f x
    hf2 : Function.Injective f
    ⊢ AccPt (f x) (Filter.map f F)
  -/
  apply map_neBot (m := f) (hf := h) |>.mono
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    F : Filter X
    x : X
    h : AccPt x F
    f : X → β
    hf1 : ContinuousAt f x
    hf2 : Function.Injective f
    ⊢ LE.le (Filter.map f (Min.min (nhdsWithin x (HasCompl.compl (Singleton.single …
  -/
  rw [Filter.map_inf hf2]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    F : Filter X
    x : X
    h : AccPt x F
    f : X → β
    hf1 : ContinuousAt f x
    hf2 : Function.Injective f
    ⊢ LE.le (Min.min (Filter.map f (nhdsWithin x (HasCompl.compl (Singleton.single …
  -/
  gcongr
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    F : Filter X
    x : X
    h : AccPt x F
    f : X → β
    hf1 : ContinuousAt f x
    hf2 : Function.Injective f
    ⊢ LE.le (Filter.map f (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))) …
  -/
  apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ hf1.continuousWithinAt
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    F : Filter X
    x : X
    h : AccPt x F
    f : X → β
    hf1 : ContinuousAt f x
    hf2 : Function.Injective f
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
  -/
  simpa [hf2.eq_iff] using eventually_mem_nhdsWithin
  /-
    🎉 no goals
  -/


/--
The derived set of a set is the set of all accumulation points of it.
-/
def derivedSet (A : Set X) : Set X := {x | AccPt x (𝓟 A)}


@[simp]
lemma mem_derivedSet {A : Set X} {x : X} : x ∈ derivedSet A ↔ AccPt x (𝓟 A) := Iff.rfl


lemma derivedSet_union (A B : Set X) : derivedSet (A ∪ B) = derivedSet A ∪ derivedSet B := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    A B : Set X
    ⊢ Eq (derivedSet (Union.union A B)) (Union.union (derivedSet A) (derivedSet B))
  -/
  ext x
  /-
    case h
    X : Type u_1
    inst✝ : TopologicalSpace X
    A B : Set X
    x : X
    ⊢ Iff (Membership.mem (derivedSet (Union.union A B)) x) (Membership.mem (Union …
  -/
  simp [derivedSet, ← sup_principal, accPt_sup]
  /-
    🎉 no goals
  -/


lemma derivedSet_mono (A B : Set X) (h : A ⊆ B) : derivedSet A ⊆ derivedSet B :=
  fun _ hx ↦ hx.mono <| le_principal_iff.mpr <| mem_principal.mpr h


theorem Continuous.image_derivedSet {β : Type*} [TopologicalSpace β] {A : Set X} {f : X → β}
    (hf1 : Continuous f) (hf2 : Function.Injective f) :
    f '' derivedSet A ⊆ derivedSet (f '' A) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    A : Set X
    f : X → β
    hf1 : Continuous f
    hf2 : Function.Injective f
    ⊢ HasSubset.Subset (Set.image f (derivedSet A)) (derivedSet (Set.image f A))
  -/
  intro x hx
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    A : Set X
    f : X → β
    hf1 : Continuous f
    hf2 : Function.Injective f
    x : β
    hx : Membership.mem (Set.image f (derivedSet A)) x
    ⊢ Membership.mem (derivedSet (Set.image f A)) x
  -/
  simp only [Set.mem_image, mem_derivedSet] at hx
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    A : Set X
    f : X → β
    hf1 : Continuous f
    hf2 : Function.Injective f
    x : β
    hx : Exists fun x_1 => And (AccPt x_1 (Filter.principal A)) (Eq (f x_1) x)
    ⊢ Membership.mem (derivedSet (Set.image f A)) x
  -/
  obtain ⟨y, hy1, rfl⟩ := hx
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    A : Set X
    f : X → β
    hf1 : Continuous f
    hf2 : Function.Injective f
    y : X
    hy1 : AccPt y (Filter.principal A)
    ⊢ Membership.mem (derivedSet (Set.image f A)) (f y)
  -/
  convert hy1.map hf1.continuousAt hf2
  /-
    case a
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    β : Type u_2
    inst✝ : TopologicalSpace β
    A : Set X
    f : X → β
    hf1 : Continuous f
    hf2 : Function.Injective f
    y : X
    hy1 : AccPt y (Filter.principal A)
    ⊢ Iff (Membership.mem (derivedSet (Set.image f A)) (f y)) (AccPt (f y) (Filter …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma derivedSet_subset_closure (A : Set X) : derivedSet A ⊆ closure A :=
  fun _ hx ↦ mem_closure_iff_clusterPt.mpr hx.clusterPt


lemma isClosed_iff_derivedSet_subset (A : Set X) : IsClosed A ↔ derivedSet A ⊆ A where
  mp h := derivedSet_subset_closure A |>.trans h.closure_subset
  mpr h := by
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      A : Set X
      h : HasSubset.Subset (derivedSet A) A
      ⊢ IsClosed A
    -/
    rw [isClosed_iff_clusterPt]
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      A : Set X
      h : HasSubset.Subset (derivedSet A) A
      ⊢ ∀ (a : X), ClusterPt a (Filter.principal A) → Membership.mem A a
    -/
    intro a ha
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      A : Set X
      h : HasSubset.Subset (derivedSet A) A
      a : X
      ha : ClusterPt a (Filter.principal A)
      ⊢ Membership.mem A a
    -/
    by_contra! nh
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      A : Set X
      h : HasSubset.Subset (derivedSet A) A
      a : X
      ha : ClusterPt a (Filter.principal A)
      nh : Not (Membership.mem A a)
      ⊢ False
    -/
    have : A = A \ {a} := by simp [nh]
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      A : Set X
      h : HasSubset.Subset (derivedSet A) A
      a : X
      ha : ClusterPt a (Filter.principal A)
      nh : Not (Membership.mem A a)
      this : Eq A (SDiff.sdiff A (Singleton.singleton a))
      ⊢ False
    -/
    rw [this, ← acc_principal_iff_cluster] at ha
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      A : Set X
      h : HasSubset.Subset (derivedSet A) A
      a : X
      ha : AccPt a (Filter.principal A)
      nh : Not (Membership.mem A a)
      this : Eq A (SDiff.sdiff A (Singleton.singleton a))
      ⊢ False
    -/
    exact nh (h ha)
    /-
      🎉 no goals
    -/


/-- In a `T1Space`, the `derivedSet` of the closure of a set is equal to the derived set of the
set itself.

Note: this doesn't hold in a space with the indiscrete topology. For example, if `X` is a type with
two elements, `x` and `y`, and `A := {x}`, then `closure A = Set.univ` and `derivedSet A = {y}`,
but `derivedSet Set.univ = Set.univ`. -/
lemma derivedSet_closure [T1Space X] (A : Set X) : derivedSet (closure A) = derivedSet A := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    ⊢ Eq (derivedSet (closure A)) (derivedSet A)
  -/
  refine le_antisymm (fun x hx => ?_) (derivedSet_mono _ _ subset_closure)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    x : X
    hx : Membership.mem (derivedSet (closure A)) x
    ⊢ Membership.mem (derivedSet A) x
  -/
  rw [mem_derivedSet, AccPt, (nhdsWithin_basis_open x {x}ᶜ).inf_principal_neBot_iff] at hx ⊢
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    x : X
    hx : ∀ ⦃i : Set X⦄, And (Membership.mem i x) (IsOpen i) → (Inter.inter (Inter. …
    ⊢ ∀ ⦃i : Set X⦄, And (Membership.mem i x) (IsOpen i) → (Inter.inter (Inter.int …
  -/
  peel hx with u hu _
  /-
    case h.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    x : X
    hx : ∀ ⦃i : Set X⦄, And (Membership.mem i x) (IsOpen i) → (Inter.inter (Inter. …
    u : Set X
    hu : And (Membership.mem u x) (IsOpen u)
    this : (Inter.inter (Inter.inter u (HasCompl.compl (Singleton.singleton x))) ( …
    ⊢ (Inter.inter (Inter.inter u (HasCompl.compl (Singleton.singleton x))) A).Non …
  -/
  obtain ⟨-, hu_open⟩ := hu
  exact mem_closure_iff.mp this.some_mem.2 (u ∩ {x}ᶜ) (hu_open.inter isOpen_compl_singleton)
    this.some_mem.1


@[simp]
lemma isClosed_derivedSet [T1Space X] (A : Set X) : IsClosed (derivedSet A) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    ⊢ IsClosed (derivedSet A)
  -/
  rw [← derivedSet_closure, isClosed_iff_derivedSet_subset]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    ⊢ HasSubset.Subset (derivedSet (derivedSet (closure A))) (derivedSet (closure  …
  -/
  apply derivedSet_mono
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    A : Set X
    ⊢ HasSubset.Subset (derivedSet (closure A)) (closure A)
  -/
  simp [← isClosed_iff_derivedSet_subset]
  /-
    🎉 no goals
  -/


lemma preperfect_iff_subset_derivedSet {U : Set X} : Preperfect U ↔ U ⊆ derivedSet U :=
  Iff.rfl


lemma perfect_iff_eq_derivedSet {U : Set X} : Perfect U ↔ U = derivedSet U := by
  rw [perfect_def, isClosed_iff_derivedSet_subset, preperfect_iff_subset_derivedSet,
    ← subset_antisymm_iff, eq_comm]


lemma IsPreconnected.inter_derivedSet_nonempty [T1Space X] {U : Set X} (hs : IsPreconnected U)
    (a b : Set X) (h : U ⊆ a ∪ b) (ha : (U ∩ derivedSet a).Nonempty)
    (hb : (U ∩ derivedSet b).Nonempty) : (U ∩ (derivedSet a ∩ derivedSet b)).Nonempty := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    U : Set X
    hs : IsPreconnected U
    a b : Set X
    h : HasSubset.Subset U (Union.union a b)
    ha : (Inter.inter U (derivedSet a)).Nonempty
    hb : (Inter.inter U (derivedSet b)).Nonempty
    ⊢ (Inter.inter U (Inter.inter (derivedSet a) (derivedSet b))).Nonempty
  -/
  by_cases hu : U.Nontrivial
    /-
      case pos
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      U : Set X
      hs : IsPreconnected U
      a b : Set X
      h : HasSubset.Subset U (Union.union a b)
      ha : (Inter.inter U (derivedSet a)).Nonempty
      hb : (Inter.inter U (derivedSet b)).Nonempty
      hu : U.Nontrivial
      ⊢ (Inter.inter U (Inter.inter (derivedSet a) (derivedSet b))).Nonempty
    -/
  · apply isPreconnected_closed_iff.mp hs
      /-
        case pos.a
        X : Type u_1
        inst✝¹ : TopologicalSpace X
        inst✝ : T1Space X
        U : Set X
        hs : IsPreconnected U
        a b : Set X
        h : HasSubset.Subset U (Union.union a b)
        ha : (Inter.inter U (derivedSet a)).Nonempty
        hb : (Inter.inter U (derivedSet b)).Nonempty
        hu : U.Nontrivial
        ⊢ IsClosed (derivedSet a)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case pos.a
        X : Type u_1
        inst✝¹ : TopologicalSpace X
        inst✝ : T1Space X
        U : Set X
        hs : IsPreconnected U
        a b : Set X
        h : HasSubset.Subset U (Union.union a b)
        ha : (Inter.inter U (derivedSet a)).Nonempty
        hb : (Inter.inter U (derivedSet b)).Nonempty
        hu : U.Nontrivial
        ⊢ IsClosed (derivedSet b)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case pos.a
        X : Type u_1
        inst✝¹ : TopologicalSpace X
        inst✝ : T1Space X
        U : Set X
        hs : IsPreconnected U
        a b : Set X
        h : HasSubset.Subset U (Union.union a b)
        ha : (Inter.inter U (derivedSet a)).Nonempty
        hb : (Inter.inter U (derivedSet b)).Nonempty
        hu : U.Nontrivial
        ⊢ HasSubset.Subset U (Union.union (derivedSet a) (derivedSet b))
      -/
    · trans derivedSet U
        /-
          X : Type u_1
          inst✝¹ : TopologicalSpace X
          inst✝ : T1Space X
          U : Set X
          hs : IsPreconnected U
          a b : Set X
          h : HasSubset.Subset U (Union.union a b)
          ha : (Inter.inter U (derivedSet a)).Nonempty
          hb : (Inter.inter U (derivedSet b)).Nonempty
          hu : U.Nontrivial
          ⊢ HasSubset.Subset U (derivedSet U)
        -/
      · apply hs.preperfect_of_nontrivial hu
        /-
          🎉 no goals
        -/
        /-
          X : Type u_1
          inst✝¹ : TopologicalSpace X
          inst✝ : T1Space X
          U : Set X
          hs : IsPreconnected U
          a b : Set X
          h : HasSubset.Subset U (Union.union a b)
          ha : (Inter.inter U (derivedSet a)).Nonempty
          hb : (Inter.inter U (derivedSet b)).Nonempty
          hu : U.Nontrivial
          ⊢ HasSubset.Subset (derivedSet U) (Union.union (derivedSet a) (derivedSet b))
        -/
      · rw [← derivedSet_union]
        /-
          X : Type u_1
          inst✝¹ : TopologicalSpace X
          inst✝ : T1Space X
          U : Set X
          hs : IsPreconnected U
          a b : Set X
          h : HasSubset.Subset U (Union.union a b)
          ha : (Inter.inter U (derivedSet a)).Nonempty
          hb : (Inter.inter U (derivedSet b)).Nonempty
          hu : U.Nontrivial
          ⊢ HasSubset.Subset (derivedSet U) (derivedSet (Union.union a b))
        -/
        exact derivedSet_mono _ _ h
        /-
          🎉 no goals
        -/
      /-
        case pos.a
        X : Type u_1
        inst✝¹ : TopologicalSpace X
        inst✝ : T1Space X
        U : Set X
        hs : IsPreconnected U
        a b : Set X
        h : HasSubset.Subset U (Union.union a b)
        ha : (Inter.inter U (derivedSet a)).Nonempty
        hb : (Inter.inter U (derivedSet b)).Nonempty
        hu : U.Nontrivial
        ⊢ (Inter.inter U (derivedSet a)).Nonempty
      -/
    · exact ha
      /-
        🎉 no goals
      -/
      /-
        case pos.a
        X : Type u_1
        inst✝¹ : TopologicalSpace X
        inst✝ : T1Space X
        U : Set X
        hs : IsPreconnected U
        a b : Set X
        h : HasSubset.Subset U (Union.union a b)
        ha : (Inter.inter U (derivedSet a)).Nonempty
        hb : (Inter.inter U (derivedSet b)).Nonempty
        hu : U.Nontrivial
        ⊢ (Inter.inter U (derivedSet b)).Nonempty
      -/
    · exact hb
      /-
        🎉 no goals
      -/
    /-
      case neg
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      U : Set X
      hs : IsPreconnected U
      a b : Set X
      h : HasSubset.Subset U (Union.union a b)
      ha : (Inter.inter U (derivedSet a)).Nonempty
      hb : (Inter.inter U (derivedSet b)).Nonempty
      hu : Not U.Nontrivial
      ⊢ (Inter.inter U (Inter.inter (derivedSet a) (derivedSet b))).Nonempty
    -/
  · obtain ⟨x, hx⟩ := ha.left.exists_eq_singleton_or_nontrivial.resolve_right hu
    /-
      case neg.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T1Space X
      U : Set X
      hs : IsPreconnected U
      a b : Set X
      h : HasSubset.Subset U (Union.union a b)
      ha : (Inter.inter U (derivedSet a)).Nonempty
      hb : (Inter.inter U (derivedSet b)).Nonempty
      hu : Not U.Nontrivial
      x : X
      hx : Eq U (Singleton.singleton x)
      ⊢ (Inter.inter U (Inter.inter (derivedSet a) (derivedSet b))).Nonempty
    -/
    simp_all
    /-
      🎉 no goals
    -/

