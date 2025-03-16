/-- A preconnected set is one where there is no non-trivial open partition. -/
def IsPreconnected (s : Set α) : Prop :=
  ∀ u v : Set α, IsOpen u → IsOpen v → s ⊆ u ∪ v → (s ∩ u).Nonempty → (s ∩ v).Nonempty →
    (s ∩ (u ∩ v)).Nonempty


/-- A connected set is one that is nonempty and where there is no non-trivial open partition. -/
def IsConnected (s : Set α) : Prop :=
  s.Nonempty ∧ IsPreconnected s


theorem IsConnected.nonempty {s : Set α} (h : IsConnected s) : s.Nonempty :=
  h.1


theorem IsConnected.isPreconnected {s : Set α} (h : IsConnected s) : IsPreconnected s :=
  h.2


theorem IsPreirreducible.isPreconnected {s : Set α} (H : IsPreirreducible s) : IsPreconnected s :=
  fun _ _ hu hv _ => H _ _ hu hv


theorem IsIrreducible.isConnected {s : Set α} (H : IsIrreducible s) : IsConnected s :=
  ⟨H.nonempty, H.isPreirreducible.isPreconnected⟩


theorem isPreconnected_empty : IsPreconnected (∅ : Set α) :=
  isPreirreducible_empty.isPreconnected


theorem isConnected_singleton {x} : IsConnected ({x} : Set α) :=
  isIrreducible_singleton.isConnected


theorem isPreconnected_singleton {x} : IsPreconnected ({x} : Set α) :=
  isConnected_singleton.isPreconnected


theorem Set.Subsingleton.isPreconnected {s : Set α} (hs : s.Subsingleton) : IsPreconnected s :=
  hs.induction_on isPreconnected_empty fun _ => isPreconnected_singleton


/-- If any point of a set is joined to a fixed point by a preconnected subset,
then the original set is preconnected as well. -/
theorem isPreconnected_of_forall {s : Set α} (x : α)
    (H : ∀ y ∈ s, ∃ t, t ⊆ s ∧ x ∈ t ∧ y ∈ t ∧ IsPreconnected t) : IsPreconnected s := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    x : α
    H : ∀ (y : α), Membership.mem s y → Exists fun t => And (HasSubset.Subset t s) …
    ⊢ IsPreconnected s
  -/
  rintro u v hu hv hs ⟨z, zs, zu⟩ ⟨y, ys, yv⟩
  have xs : x ∈ s := by
    rcases H y ys with ⟨t, ts, xt, -, -⟩
    exact ts xt
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: use `wlog xu : x ∈ u := hs xs using u v y z, v u z y`
  cases hs xs with
  | inl xu =>
    rcases H y ys with ⟨t, ts, xt, yt, ht⟩
    have := ht u v hu hv (ts.trans hs) ⟨x, xt, xu⟩ ⟨y, yt, yv⟩
    exact this.imp fun z hz => ⟨ts hz.1, hz.2⟩
  | inr xv =>
    rcases H z zs with ⟨t, ts, xt, zt, ht⟩
    have := ht v u hv hu (ts.trans <| by rwa [union_comm]) ⟨x, xt, xv⟩ ⟨z, zt, zu⟩
    exact this.imp fun _ h => ⟨ts h.1, h.2.2, h.2.1⟩


/-- If any two points of a set are contained in a preconnected subset,
then the original set is preconnected as well. -/
theorem isPreconnected_of_forall_pair {s : Set α}
    (H : ∀ x ∈ s, ∀ y ∈ s, ∃ t, t ⊆ s ∧ x ∈ t ∧ y ∈ t ∧ IsPreconnected t) :
    IsPreconnected s := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    H : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Exists fun …
    ⊢ IsPreconnected s
  -/
  rcases eq_empty_or_nonempty s with (rfl | ⟨x, hx⟩)
  /-
    case inl
    α : Type u
    inst✝ : TopologicalSpace α
    H : ∀ (x : α), Membership.mem EmptyCollection.emptyCollection x → ∀ (y : α), M …
    ⊢ IsPreconnected EmptyCollection.emptyCollection
  -/
  exacts [isPreconnected_empty, isPreconnected_of_forall x fun y => H x hx y]
  /-
    🎉 no goals
  -/


/-- A union of a family of preconnected sets with a common point is preconnected as well. -/
theorem isPreconnected_sUnion (x : α) (c : Set (Set α)) (H1 : ∀ s ∈ c, x ∈ s)
    (H2 : ∀ s ∈ c, IsPreconnected s) : IsPreconnected (⋃₀ c) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    c : Set (Set α)
    H1 : ∀ (s : Set α), Membership.mem c s → Membership.mem s x
    H2 : ∀ (s : Set α), Membership.mem c s → IsPreconnected s
    ⊢ IsPreconnected c.sUnion
  -/
  apply isPreconnected_of_forall x
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    c : Set (Set α)
    H1 : ∀ (s : Set α), Membership.mem c s → Membership.mem s x
    H2 : ∀ (s : Set α), Membership.mem c s → IsPreconnected s
    ⊢ ∀ (y : α), Membership.mem c.sUnion y → Exists fun t => And (HasSubset.Subset …
  -/
  rintro y ⟨s, sc, ys⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    c : Set (Set α)
    H1 : ∀ (s : Set α), Membership.mem c s → Membership.mem s x
    H2 : ∀ (s : Set α), Membership.mem c s → IsPreconnected s
    y : α
    s : Set α
    sc : Membership.mem c s
    ys : Membership.mem s y
    ⊢ Exists fun t => And (HasSubset.Subset t c.sUnion) (And (Membership.mem t x)  …
  -/
  exact ⟨s, subset_sUnion_of_mem sc, H1 s sc, ys, H2 s sc⟩
  /-
    🎉 no goals
  -/


theorem isPreconnected_iUnion {ι : Sort*} {s : ι → Set α} (h₁ : (⋂ i, s i).Nonempty)
    (h₂ : ∀ i, IsPreconnected (s i)) : IsPreconnected (⋃ i, s i) :=
  Exists.elim h₁ fun f hf => isPreconnected_sUnion f _ hf (forall_mem_range.2 h₂)


theorem IsPreconnected.union (x : α) {s t : Set α} (H1 : x ∈ s) (H2 : x ∈ t) (H3 : IsPreconnected s)
    (H4 : IsPreconnected t) : IsPreconnected (s ∪ t) :=
                                                       /-
                                                         α : Type u
                                                         inst✝ : TopologicalSpace α
                                                         x : α
                                                         s t : Set α
                                                         H1 : Membership.mem s x
                                                         H2 : Membership.mem t x
                                                         H3 : IsPreconnected s
                                                         H4 : IsPreconnected t
                                                         ⊢ ∀ (s_1 : Set α), Membership.mem (Insert.insert s (Singleton.singleton t)) s_ …
                                                       -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  sUnion_pair s t ▸ isPreconnected_sUnion x {s, t} (by rintro r (rfl | rfl | h) <;> assumption)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
        /-
          α : Type u
          inst✝ : TopologicalSpace α
          x : α
          s t : Set α
          H1 : Membership.mem s x
          H2 : Membership.mem t x
          H3 : IsPreconnected s
          H4 : IsPreconnected t
          ⊢ ∀ (s_1 : Set α), Membership.mem (Insert.insert s (Singleton.singleton t)) s_ …
        -/
                                     /-
                                       🎉 no goals
                                     -/
    (by rintro r (rfl | rfl | h) <;> assumption)
                                     /-
                                       🎉 no goals
                                     -/


theorem IsPreconnected.union' {s t : Set α} (H : (s ∩ t).Nonempty) (hs : IsPreconnected s)
    (ht : IsPreconnected t) : IsPreconnected (s ∪ t) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s t : Set α
    H : (Inter.inter s t).Nonempty
    hs : IsPreconnected s
    ht : IsPreconnected t
    ⊢ IsPreconnected (Union.union s t)
  -/
  rcases H with ⟨x, hxs, hxt⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    s t : Set α
    hs : IsPreconnected s
    ht : IsPreconnected t
    x : α
    hxs : Membership.mem s x
    hxt : Membership.mem t x
    ⊢ IsPreconnected (Union.union s t)
  -/
  exact hs.union x hxs hxt ht
  /-
    🎉 no goals
  -/


theorem IsConnected.union {s t : Set α} (H : (s ∩ t).Nonempty) (Hs : IsConnected s)
    (Ht : IsConnected t) : IsConnected (s ∪ t) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s t : Set α
    H : (Inter.inter s t).Nonempty
    Hs : IsConnected s
    Ht : IsConnected t
    ⊢ IsConnected (Union.union s t)
  -/
  rcases H with ⟨x, hx⟩
  /-
    case intro
    α : Type u
    inst✝ : TopologicalSpace α
    s t : Set α
    Hs : IsConnected s
    Ht : IsConnected t
    x : α
    hx : Membership.mem (Inter.inter s t) x
    ⊢ IsConnected (Union.union s t)
  -/
  refine ⟨⟨x, mem_union_left t (mem_of_mem_inter_left hx)⟩, ?_⟩
  exact Hs.isPreconnected.union x (mem_of_mem_inter_left hx) (mem_of_mem_inter_right hx)
    Ht.isPreconnected


/-- The directed sUnion of a set S of preconnected subsets is preconnected. -/
theorem IsPreconnected.sUnion_directed {S : Set (Set α)} (K : DirectedOn (· ⊆ ·) S)
    (H : ∀ s ∈ S, IsPreconnected s) : IsPreconnected (⋃₀ S) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    K : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) S
    H : ∀ (s : Set α), Membership.mem S s → IsPreconnected s
    ⊢ IsPreconnected S.sUnion
  -/
  rintro u v hu hv Huv ⟨a, ⟨s, hsS, has⟩, hau⟩ ⟨b, ⟨t, htS, hbt⟩, hbv⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    K : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) S
    H : ∀ (s : Set α), Membership.mem S s → IsPreconnected s
    u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    Huv : HasSubset.Subset S.sUnion (Union.union u v)
    a : α
    hau : Membership.mem u a
    s : Set α
    hsS : Membership.mem S s
    has : Membership.mem s a
    b : α
    hbv : Membership.mem v b
    t : Set α
    htS : Membership.mem S t
    hbt : Membership.mem t b
    ⊢ (Inter.inter S.sUnion (Inter.inter u v)).Nonempty
  -/
  obtain ⟨r, hrS, hsr, htr⟩ : ∃ r ∈ S, s ⊆ r ∧ t ⊆ r := K s hsS t htS
  have Hnuv : (r ∩ (u ∩ v)).Nonempty :=
    H _ hrS u v hu hv ((subset_sUnion_of_mem hrS).trans Huv) ⟨a, hsr has, hau⟩ ⟨b, htr hbt, hbv⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    K : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) S
    H : ∀ (s : Set α), Membership.mem S s → IsPreconnected s
    u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    Huv : HasSubset.Subset S.sUnion (Union.union u v)
    a : α
    hau : Membership.mem u a
    s : Set α
    hsS : Membership.mem S s
    has : Membership.mem s a
    b : α
    hbv : Membership.mem v b
    t : Set α
    htS : Membership.mem S t
    hbt : Membership.mem t b
    r : Set α
    hrS : Membership.mem S r
    hsr : HasSubset.Subset s r
    htr : HasSubset.Subset t r
    Hnuv : (Inter.inter r (Inter.inter u v)).Nonempty
    ⊢ (Inter.inter S.sUnion (Inter.inter u v)).Nonempty
  -/
  have Kruv : r ∩ (u ∩ v) ⊆ ⋃₀ S ∩ (u ∩ v) := inter_subset_inter_left _ (subset_sUnion_of_mem hrS)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    K : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) S
    H : ∀ (s : Set α), Membership.mem S s → IsPreconnected s
    u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    Huv : HasSubset.Subset S.sUnion (Union.union u v)
    a : α
    hau : Membership.mem u a
    s : Set α
    hsS : Membership.mem S s
    has : Membership.mem s a
    b : α
    hbv : Membership.mem v b
    t : Set α
    htS : Membership.mem S t
    hbt : Membership.mem t b
    r : Set α
    hrS : Membership.mem S r
    hsr : HasSubset.Subset s r
    htr : HasSubset.Subset t r
    Hnuv : (Inter.inter r (Inter.inter u v)).Nonempty
    Kruv : HasSubset.Subset (Inter.inter r (Inter.inter u v)) (Inter.inter S.sUnio …
    ⊢ (Inter.inter S.sUnion (Inter.inter u v)).Nonempty
  -/
  exact Hnuv.mono Kruv
  /-
    🎉 no goals
  -/


/-- The biUnion of a family of preconnected sets is preconnected if the graph determined by
whether two sets intersect is preconnected. -/
theorem IsPreconnected.biUnion_of_reflTransGen {ι : Type*} {t : Set ι} {s : ι → Set α}
    (H : ∀ i ∈ t, IsPreconnected (s i))
    (K : ∀ i, i ∈ t → ∀ j, j ∈ t → ReflTransGen (fun i j => (s i ∩ s j).Nonempty ∧ i ∈ t) i j) :
    IsPreconnected (⋃ n ∈ t, s n) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    t : Set ι
    s : ι → Set α
    H : ∀ (i : ι), Membership.mem t i → IsPreconnected (s i)
    K : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    ⊢ IsPreconnected (Set.iUnion fun n => Set.iUnion fun h => s n)
  -/
  let R := fun i j : ι => (s i ∩ s j).Nonempty ∧ i ∈ t
  have P : ∀ i, i ∈ t → ∀ j, j ∈ t → ReflTransGen R i j →
      ∃ p, p ⊆ t ∧ i ∈ p ∧ j ∈ p ∧ IsPreconnected (⋃ j ∈ p, s j) := fun i hi j hj h => by
    induction h with
    | refl =>
      refine ⟨{i}, singleton_subset_iff.mpr hi, mem_singleton i, mem_singleton i, ?_⟩
      rw [biUnion_singleton]
      exact H i hi
    | @tail j k _ hjk ih =>
      obtain ⟨p, hpt, hip, hjp, hp⟩ := ih hjk.2
      refine ⟨insert k p, insert_subset_iff.mpr ⟨hj, hpt⟩, mem_insert_of_mem k hip,
        mem_insert k p, ?_⟩
      rw [biUnion_insert]
      refine (H k hj).union' (hjk.1.mono ?_) hp
      rw [inter_comm]
      exact inter_subset_inter_right _ (subset_biUnion_of_mem hjp)
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    t : Set ι
    s : ι → Set α
    H : ∀ (i : ι), Membership.mem t i → IsPreconnected (s i)
    K : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    R : ι → ι → Prop := fun i j => And (Inter.inter (s i) (s j)).Nonempty (Members …
    P : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    ⊢ IsPreconnected (Set.iUnion fun n => Set.iUnion fun h => s n)
  -/
  refine isPreconnected_of_forall_pair ?_
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    t : Set ι
    s : ι → Set α
    H : ∀ (i : ι), Membership.mem t i → IsPreconnected (s i)
    K : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    R : ι → ι → Prop := fun i j => And (Inter.inter (s i) (s j)).Nonempty (Members …
    P : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    ⊢ ∀ (x : α), Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) x →  …
  -/
  intro x hx y hy
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    t : Set ι
    s : ι → Set α
    H : ∀ (i : ι), Membership.mem t i → IsPreconnected (s i)
    K : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    R : ι → ι → Prop := fun i j => And (Inter.inter (s i) (s j)).Nonempty (Members …
    P : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    x : α
    hx : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) x
    y : α
    hy : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) y
    ⊢ Exists fun t_1 => And (HasSubset.Subset t_1 (Set.iUnion fun n => Set.iUnion  …
  -/
  obtain ⟨i : ι, hi : i ∈ t, hxi : x ∈ s i⟩ := mem_iUnion₂.1 hx
  /-
    case intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    t : Set ι
    s : ι → Set α
    H : ∀ (i : ι), Membership.mem t i → IsPreconnected (s i)
    K : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    R : ι → ι → Prop := fun i j => And (Inter.inter (s i) (s j)).Nonempty (Members …
    P : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    x : α
    hx : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) x
    y : α
    hy : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) y
    i : ι
    hi : Membership.mem t i
    hxi : Membership.mem (s i) x
    ⊢ Exists fun t_1 => And (HasSubset.Subset t_1 (Set.iUnion fun n => Set.iUnion  …
  -/
  obtain ⟨j : ι, hj : j ∈ t, hyj : y ∈ s j⟩ := mem_iUnion₂.1 hy
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    t : Set ι
    s : ι → Set α
    H : ∀ (i : ι), Membership.mem t i → IsPreconnected (s i)
    K : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    R : ι → ι → Prop := fun i j => And (Inter.inter (s i) (s j)).Nonempty (Members …
    P : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Relation.R …
    x : α
    hx : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) x
    y : α
    hy : Membership.mem (Set.iUnion fun n => Set.iUnion fun h => s n) y
    i : ι
    hi : Membership.mem t i
    hxi : Membership.mem (s i) x
    j : ι
    hj : Membership.mem t j
    hyj : Membership.mem (s j) y
    ⊢ Exists fun t_1 => And (HasSubset.Subset t_1 (Set.iUnion fun n => Set.iUnion  …
  -/
  obtain ⟨p, hpt, hip, hjp, hp⟩ := P i hi j hj (K i hi j hj)
  exact ⟨⋃ j ∈ p, s j, biUnion_subset_biUnion_left hpt, mem_biUnion hip hxi,
    mem_biUnion hjp hyj, hp⟩


/-- The biUnion of a family of preconnected sets is preconnected if the graph determined by
whether two sets intersect is preconnected. -/
theorem IsConnected.biUnion_of_reflTransGen {ι : Type*} {t : Set ι} {s : ι → Set α}
    (ht : t.Nonempty) (H : ∀ i ∈ t, IsConnected (s i))
    (K : ∀ i, i ∈ t → ∀ j, j ∈ t → ReflTransGen (fun i j : ι => (s i ∩ s j).Nonempty ∧ i ∈ t) i j) :
    IsConnected (⋃ n ∈ t, s n) :=
  ⟨nonempty_biUnion.2 <| ⟨ht.some, ht.some_mem, (H _ ht.some_mem).nonempty⟩,
    IsPreconnected.biUnion_of_reflTransGen (fun i hi => (H i hi).isPreconnected) K⟩


/-- Preconnectedness of the iUnion of a family of preconnected sets
indexed by the vertices of a preconnected graph,
where two vertices are joined when the corresponding sets intersect. -/
theorem IsPreconnected.iUnion_of_reflTransGen {ι : Type*} {s : ι → Set α}
    (H : ∀ i, IsPreconnected (s i))
    (K : ∀ i j, ReflTransGen (fun i j : ι => (s i ∩ s j).Nonempty) i j) :
    IsPreconnected (⋃ n, s n) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    s : ι → Set α
    H : ∀ (i : ι), IsPreconnected (s i)
    K : ∀ (i j : ι), Relation.ReflTransGen (fun i j => (Inter.inter (s i) (s j)).N …
    ⊢ IsPreconnected (Set.iUnion fun n => s n)
  -/
  rw [← biUnion_univ]
  exact IsPreconnected.biUnion_of_reflTransGen (fun i _ => H i) fun i _ j _ => by
    simpa [mem_univ] using K i j


theorem IsConnected.iUnion_of_reflTransGen {ι : Type*} [Nonempty ι] {s : ι → Set α}
    (H : ∀ i, IsConnected (s i))
    (K : ∀ i j, ReflTransGen (fun i j : ι => (s i ∩ s j).Nonempty) i j) : IsConnected (⋃ n, s n) :=
  ⟨nonempty_iUnion.2 <| Nonempty.elim ‹_› fun i : ι => ⟨i, (H _).nonempty⟩,
    IsPreconnected.iUnion_of_reflTransGen (fun i => (H i).isPreconnected) K⟩


/-- The iUnion of connected sets indexed by a type with an archimedean successor (like `ℕ` or `ℤ`)
  such that any two neighboring sets meet is preconnected. -/
theorem IsPreconnected.iUnion_of_chain {s : β → Set α} (H : ∀ n, IsPreconnected (s n))
    (K : ∀ n, (s n ∩ s (succ n)).Nonempty) : IsPreconnected (⋃ n, s n) :=
  IsPreconnected.iUnion_of_reflTransGen H fun _ _ =>
    reflTransGen_of_succ _ (fun i _ => K i) fun i _ => by
      /-
        α : Type u
        β : Type v
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder β
        inst✝¹ : SuccOrder β
        inst✝ : IsSuccArchimedean β
        s : β → Set α
        H : ∀ (n : β), IsPreconnected (s n)
        K : ∀ (n : β), (Inter.inter (s n) (s (Order.succ n))).Nonempty
        x✝² x✝¹ i : β
        x✝ : Membership.mem (Set.Ico x✝¹ x✝²) i
        ⊢ (Inter.inter (s (Order.succ i)) (s i)).Nonempty
      -/
      rw [inter_comm]
      /-
        α : Type u
        β : Type v
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder β
        inst✝¹ : SuccOrder β
        inst✝ : IsSuccArchimedean β
        s : β → Set α
        H : ∀ (n : β), IsPreconnected (s n)
        K : ∀ (n : β), (Inter.inter (s n) (s (Order.succ n))).Nonempty
        x✝² x✝¹ i : β
        x✝ : Membership.mem (Set.Ico x✝¹ x✝²) i
        ⊢ (Inter.inter (s i) (s (Order.succ i))).Nonempty
      -/
      exact K i
      /-
        🎉 no goals
      -/


/-- The iUnion of connected sets indexed by a type with an archimedean successor (like `ℕ` or `ℤ`)
  such that any two neighboring sets meet is connected. -/
theorem IsConnected.iUnion_of_chain [Nonempty β] {s : β → Set α} (H : ∀ n, IsConnected (s n))
    (K : ∀ n, (s n ∩ s (succ n)).Nonempty) : IsConnected (⋃ n, s n) :=
  IsConnected.iUnion_of_reflTransGen H fun _ _ =>
    reflTransGen_of_succ _ (fun i _ => K i) fun i _ => by
      /-
        α : Type u
        β : Type v
        inst✝⁴ : TopologicalSpace α
        inst✝³ : LinearOrder β
        inst✝² : SuccOrder β
        inst✝¹ : IsSuccArchimedean β
        inst✝ : Nonempty β
        s : β → Set α
        H : ∀ (n : β), IsConnected (s n)
        K : ∀ (n : β), (Inter.inter (s n) (s (Order.succ n))).Nonempty
        x✝² x✝¹ i : β
        x✝ : Membership.mem (Set.Ico x✝¹ x✝²) i
        ⊢ (Inter.inter (s (Order.succ i)) (s i)).Nonempty
      -/
      rw [inter_comm]
      /-
        α : Type u
        β : Type v
        inst✝⁴ : TopologicalSpace α
        inst✝³ : LinearOrder β
        inst✝² : SuccOrder β
        inst✝¹ : IsSuccArchimedean β
        inst✝ : Nonempty β
        s : β → Set α
        H : ∀ (n : β), IsConnected (s n)
        K : ∀ (n : β), (Inter.inter (s n) (s (Order.succ n))).Nonempty
        x✝² x✝¹ i : β
        x✝ : Membership.mem (Set.Ico x✝¹ x✝²) i
        ⊢ (Inter.inter (s i) (s (Order.succ i))).Nonempty
      -/
      exact K i
      /-
        🎉 no goals
      -/


/-- The iUnion of preconnected sets indexed by a subset of a type with an archimedean successor
  (like `ℕ` or `ℤ`) such that any two neighboring sets meet is preconnected. -/
theorem IsPreconnected.biUnion_of_chain {s : β → Set α} {t : Set β} (ht : OrdConnected t)
    (H : ∀ n ∈ t, IsPreconnected (s n))
    (K : ∀ n : β, n ∈ t → succ n ∈ t → (s n ∩ s (succ n)).Nonempty) :
    IsPreconnected (⋃ n ∈ t, s n) := by
  have h1 : ∀ {i j k : β}, i ∈ t → j ∈ t → k ∈ Ico i j → k ∈ t := fun hi hj hk =>
    ht.out hi hj (Ico_subset_Icc_self hk)
  have h2 : ∀ {i j k : β}, i ∈ t → j ∈ t → k ∈ Ico i j → succ k ∈ t := fun hi hj hk =>
    ht.out hi hj ⟨hk.1.trans <| le_succ _, succ_le_of_lt hk.2⟩
  have h3 : ∀ {i j k : β}, i ∈ t → j ∈ t → k ∈ Ico i j → (s k ∩ s (succ k)).Nonempty :=
    fun hi hj hk => K _ (h1 hi hj hk) (h2 hi hj hk)
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder β
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    s : β → Set α
    t : Set β
    ht : t.OrdConnected
    H : ∀ (n : β), Membership.mem t n → IsPreconnected (s n)
    K : ∀ (n : β), Membership.mem t n → Membership.mem t (Order.succ n) → (Inter.i …
    h1 : ∀ {i j k : β}, Membership.mem t i → Membership.mem t j → Membership.mem ( …
    h2 : ∀ {i j k : β}, Membership.mem t i → Membership.mem t j → Membership.mem ( …
    h3 : ∀ {i j k : β}, Membership.mem t i → Membership.mem t j → Membership.mem ( …
    ⊢ IsPreconnected (Set.iUnion fun n => Set.iUnion fun h => s n)
  -/
  refine IsPreconnected.biUnion_of_reflTransGen H fun i hi j hj => ?_
  exact reflTransGen_of_succ _ (fun k hk => ⟨h3 hi hj hk, h1 hi hj hk⟩) fun k hk =>
      ⟨by rw [inter_comm]; exact h3 hj hi hk, h2 hj hi hk⟩


/-- The iUnion of connected sets indexed by a subset of a type with an archimedean successor
  (like `ℕ` or `ℤ`) such that any two neighboring sets meet is preconnected. -/
theorem IsConnected.biUnion_of_chain {s : β → Set α} {t : Set β} (hnt : t.Nonempty)
    (ht : OrdConnected t) (H : ∀ n ∈ t, IsConnected (s n))
    (K : ∀ n : β, n ∈ t → succ n ∈ t → (s n ∩ s (succ n)).Nonempty) : IsConnected (⋃ n ∈ t, s n) :=
  ⟨nonempty_biUnion.2 <| ⟨hnt.some, hnt.some_mem, (H _ hnt.some_mem).nonempty⟩,
    IsPreconnected.biUnion_of_chain ht (fun i hi => (H i hi).isPreconnected) K⟩


/-- Theorem of bark and tree: if a set is within a preconnected set and its closure, then it is
preconnected as well. See also `IsConnected.subset_closure`. -/
protected theorem IsPreconnected.subset_closure {s : Set α} {t : Set α} (H : IsPreconnected s)
    (Kst : s ⊆ t) (Ktcs : t ⊆ closure s) : IsPreconnected t :=
  fun u v hu hv htuv ⟨_y, hyt, hyu⟩ ⟨_z, hzt, hzv⟩ =>
  let ⟨p, hpu, hps⟩ := mem_closure_iff.1 (Ktcs hyt) u hu hyu
  let ⟨q, hqv, hqs⟩ := mem_closure_iff.1 (Ktcs hzt) v hv hzv
  let ⟨r, hrs, hruv⟩ := H u v hu hv (Subset.trans Kst htuv) ⟨p, hps, hpu⟩ ⟨q, hqs, hqv⟩
  ⟨r, Kst hrs, hruv⟩


/-- Theorem of bark and tree: if a set is within a connected set and its closure, then it is
connected as well. See also `IsPreconnected.subset_closure`. -/
protected theorem IsConnected.subset_closure {s : Set α} {t : Set α} (H : IsConnected s)
    (Kst : s ⊆ t) (Ktcs : t ⊆ closure s) : IsConnected t :=
  ⟨Nonempty.mono Kst H.left, IsPreconnected.subset_closure H.right Kst Ktcs⟩


/-- The closure of a preconnected set is preconnected as well. -/
protected theorem IsPreconnected.closure {s : Set α} (H : IsPreconnected s) :
    IsPreconnected (closure s) :=
  IsPreconnected.subset_closure H subset_closure Subset.rfl


/-- The closure of a connected set is connected as well. -/
protected theorem IsConnected.closure {s : Set α} (H : IsConnected s) : IsConnected (closure s) :=
  IsConnected.subset_closure H subset_closure <| Subset.rfl


/-- The image of a preconnected set is preconnected as well. -/
protected theorem IsPreconnected.image [TopologicalSpace β] {s : Set α} (H : IsPreconnected s)
    (f : α → β) (hf : ContinuousOn f s) : IsPreconnected (f '' s) := by
  -- Unfold/destruct definitions in hypotheses
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    H : IsPreconnected s
    f : α → β
    hf : ContinuousOn f s
    ⊢ IsPreconnected (Set.image f s)
  -/
  rintro u v hu hv huv ⟨_, ⟨x, xs, rfl⟩, xu⟩ ⟨_, ⟨y, ys, rfl⟩, yv⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    H : IsPreconnected s
    f : α → β
    hf : ContinuousOn f s
    u v : Set β
    hu : IsOpen u
    hv : IsOpen v
    huv : HasSubset.Subset (Set.image f s) (Union.union u v)
    x : α
    xs : Membership.mem s x
    xu : Membership.mem u (f x)
    y : α
    ys : Membership.mem s y
    yv : Membership.mem v (f y)
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rcases continuousOn_iff'.1 hf u hu with ⟨u', hu', u'_eq⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    H : IsPreconnected s
    f : α → β
    hf : ContinuousOn f s
    u v : Set β
    hu : IsOpen u
    hv : IsOpen v
    huv : HasSubset.Subset (Set.image f s) (Union.union u v)
    x : α
    xs : Membership.mem s x
    xu : Membership.mem u (f x)
    y : α
    ys : Membership.mem s y
    yv : Membership.mem v (f y)
    u' : Set α
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  rcases continuousOn_iff'.1 hf v hv with ⟨v', hv', v'_eq⟩
  -- Reformulate `huv : f '' s ⊆ u ∪ v` in terms of `u'` and `v'`
  replace huv : s ⊆ u' ∪ v' := by
    rw [image_subset_iff, preimage_union] at huv
    replace huv := subset_inter huv Subset.rfl
    rw [union_inter_distrib_right, u'_eq, v'_eq, ← union_inter_distrib_right] at huv
    exact (subset_inter_iff.1 huv).1
  -- Now `s ⊆ u' ∪ v'`, so we can apply `‹IsPreconnected s›`
  obtain ⟨z, hz⟩ : (s ∩ (u' ∩ v')).Nonempty := by
    refine H u' v' hu' hv' huv ⟨x, ?_⟩ ⟨y, ?_⟩ <;> rw [inter_comm]
    exacts [u'_eq ▸ ⟨xu, xs⟩, v'_eq ▸ ⟨yv, ys⟩]
  rw [← inter_self s, inter_assoc, inter_left_comm s u', ← inter_assoc, inter_comm s, inter_comm s,
    ← u'_eq, ← v'_eq] at hz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    H : IsPreconnected s
    f : α → β
    hf : ContinuousOn f s
    u v : Set β
    hu : IsOpen u
    hv : IsOpen v
    x : α
    xs : Membership.mem s x
    xu : Membership.mem u (f x)
    y : α
    ys : Membership.mem s y
    yv : Membership.mem v (f y)
    u' : Set α
    hu' : IsOpen u'
    u'_eq : Eq (Inter.inter (Set.preimage f u) s) (Inter.inter u' s)
    v' : Set α
    hv' : IsOpen v'
    v'_eq : Eq (Inter.inter (Set.preimage f v) s) (Inter.inter v' s)
    huv : HasSubset.Subset s (Union.union u' v')
    z : α
    hz : Membership.mem (Inter.inter (Inter.inter (Set.preimage f u) s) (Inter.int …
    ⊢ (Inter.inter (Set.image f s) (Inter.inter u v)).Nonempty
  -/
  exact ⟨f z, ⟨z, hz.1.2, rfl⟩, hz.1.1, hz.2.1⟩
  /-
    🎉 no goals
  -/


/-- The image of a connected set is connected as well. -/
protected theorem IsConnected.image [TopologicalSpace β] {s : Set α} (H : IsConnected s) (f : α → β)
    (hf : ContinuousOn f s) : IsConnected (f '' s) :=
  ⟨image_nonempty.mpr H.nonempty, H.isPreconnected.image f hf⟩


theorem isPreconnected_closed_iff {s : Set α} :
    IsPreconnected s ↔ ∀ t t', IsClosed t → IsClosed t' →
      s ⊆ t ∪ t' → (s ∩ t).Nonempty → (s ∩ t').Nonempty → (s ∩ (t ∩ t')).Nonempty :=
  ⟨by
      /-
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        ⊢ IsPreconnected s → ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Su …
      -/
      rintro h t t' ht ht' htt' ⟨x, xs, xt⟩ ⟨y, ys, yt'⟩
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        ⊢ (Inter.inter s (Inter.inter t t')).Nonempty
      -/
      rw [← not_disjoint_iff_nonempty_inter, ← subset_compl_iff_disjoint_right, compl_inter]
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        ⊢ Not (HasSubset.Subset s (Union.union (HasCompl.compl t) (HasCompl.compl t')))
      -/
      intro h'
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl t) (HasCompl.compl t'))
        ⊢ False
      -/
      have xt' : x ∉ t' := (h' xs).resolve_left (absurd xt)
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl t) (HasCompl.compl t'))
        xt' : Not (Membership.mem t' x)
        ⊢ False
      -/
      have yt : y ∉ t := (h' ys).resolve_right (absurd yt')
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl t) (HasCompl.compl t'))
        xt' : Not (Membership.mem t' x)
        yt : Not (Membership.mem t y)
        ⊢ False
      -/
      have := h _ _ ht.isOpen_compl ht'.isOpen_compl h' ⟨y, ys, yt⟩ ⟨x, xs, xt'⟩
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl t) (HasCompl.compl t'))
        xt' : Not (Membership.mem t' x)
        yt : Not (Membership.mem t y)
        this : (Inter.inter s (Inter.inter (HasCompl.compl t) (HasCompl.compl t'))).No …
        ⊢ False
      -/
      rw [← compl_union] at this
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : IsPreconnected s
        t t' : Set α
        ht : IsClosed t
        ht' : IsClosed t'
        htt' : HasSubset.Subset s (Union.union t t')
        x : α
        xs : Membership.mem s x
        xt : Membership.mem t x
        y : α
        ys : Membership.mem s y
        yt' : Membership.mem t' y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl t) (HasCompl.compl t'))
        xt' : Not (Membership.mem t' x)
        yt : Not (Membership.mem t y)
        this : (Inter.inter s (HasCompl.compl (Union.union t t'))).Nonempty
        ⊢ False
      -/
      exact this.ne_empty htt'.disjoint_compl_right.inter_eq,
      /-
        🎉 no goals
      -/
    by
      /-
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        ⊢ (∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.unio …
      -/
      rintro h u v hu hv huv ⟨x, xs, xu⟩ ⟨y, ys, yv⟩
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
      rw [← not_disjoint_iff_nonempty_inter, ← subset_compl_iff_disjoint_right, compl_inter]
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        ⊢ Not (HasSubset.Subset s (Union.union (HasCompl.compl u) (HasCompl.compl v)))
      -/
      intro h'
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl u) (HasCompl.compl v))
        ⊢ False
      -/
      have xv : x ∉ v := (h' xs).elim (absurd xu) id
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl u) (HasCompl.compl v))
        xv : Not (Membership.mem v x)
        ⊢ False
      -/
      have yu : y ∉ u := (h' ys).elim id (absurd yv)
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl u) (HasCompl.compl v))
        xv : Not (Membership.mem v x)
        yu : Not (Membership.mem u y)
        ⊢ False
      -/
      have := h _ _ hu.isClosed_compl hv.isClosed_compl h' ⟨y, ys, yu⟩ ⟨x, xs, xv⟩
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl u) (HasCompl.compl v))
        xv : Not (Membership.mem v x)
        yu : Not (Membership.mem u y)
        this : (Inter.inter s (Inter.inter (HasCompl.compl u) (HasCompl.compl v))).Non …
        ⊢ False
      -/
      rw [← compl_union] at this
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s : Set α
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
        u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : HasSubset.Subset s (Union.union u v)
        x : α
        xs : Membership.mem s x
        xu : Membership.mem u x
        y : α
        ys : Membership.mem s y
        yv : Membership.mem v y
        h' : HasSubset.Subset s (Union.union (HasCompl.compl u) (HasCompl.compl v))
        xv : Not (Membership.mem v x)
        yu : Not (Membership.mem u y)
        this : (Inter.inter s (HasCompl.compl (Union.union u v))).Nonempty
        ⊢ False
      -/
      exact this.ne_empty huv.disjoint_compl_right.inter_eq⟩
      /-
        🎉 no goals
      -/


theorem Topology.IsInducing.isPreconnected_image [TopologicalSpace β] {s : Set α} {f : α → β}
    (hf : IsInducing f) : IsPreconnected (f '' s) ↔ IsPreconnected s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    f : α → β
    hf : Topology.IsInducing f
    ⊢ Iff (IsPreconnected (Set.image f s)) (IsPreconnected s)
  -/
  refine ⟨fun h => ?_, fun h => h.image _ hf.continuous.continuousOn⟩
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    f : α → β
    hf : Topology.IsInducing f
    h : IsPreconnected (Set.image f s)
    ⊢ IsPreconnected s
  -/
  rintro u v hu' hv' huv ⟨x, hxs, hxu⟩ ⟨y, hys, hyv⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    f : α → β
    hf : Topology.IsInducing f
    h : IsPreconnected (Set.image f s)
    u v : Set α
    hu' : IsOpen u
    hv' : IsOpen v
    huv : HasSubset.Subset s (Union.union u v)
    x : α
    hxs : Membership.mem s x
    hxu : Membership.mem u x
    y : α
    hys : Membership.mem s y
    hyv : Membership.mem v y
    ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
  -/
  rcases hf.isOpen_iff.1 hu' with ⟨u, hu, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    f : α → β
    hf : Topology.IsInducing f
    h : IsPreconnected (Set.image f s)
    v : Set α
    hv' : IsOpen v
    x : α
    hxs : Membership.mem s x
    y : α
    hys : Membership.mem s y
    hyv : Membership.mem v y
    u : Set β
    hu : IsOpen u
    hu' : IsOpen (Set.preimage f u)
    huv : HasSubset.Subset s (Union.union (Set.preimage f u) v)
    hxu : Membership.mem (Set.preimage f u) x
    ⊢ (Inter.inter s (Inter.inter (Set.preimage f u) v)).Nonempty
  -/
  rcases hf.isOpen_iff.1 hv' with ⟨v, hv, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    f : α → β
    hf : Topology.IsInducing f
    h : IsPreconnected (Set.image f s)
    x : α
    hxs : Membership.mem s x
    y : α
    hys : Membership.mem s y
    u : Set β
    hu : IsOpen u
    hu' : IsOpen (Set.preimage f u)
    hxu : Membership.mem (Set.preimage f u) x
    v : Set β
    hv : IsOpen v
    hv' : IsOpen (Set.preimage f v)
    hyv : Membership.mem (Set.preimage f v) y
    huv : HasSubset.Subset s (Union.union (Set.preimage f u) (Set.preimage f v))
    ⊢ (Inter.inter s (Inter.inter (Set.preimage f u) (Set.preimage f v))).Nonempty
  -/
  replace huv : f '' s ⊆ u ∪ v := by rwa [image_subset_iff]
  rcases h u v hu hv huv ⟨f x, mem_image_of_mem _ hxs, hxu⟩ ⟨f y, mem_image_of_mem _ hys, hyv⟩ with
    ⟨_, ⟨z, hzs, rfl⟩, hzuv⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    f : α → β
    hf : Topology.IsInducing f
    h : IsPreconnected (Set.image f s)
    x : α
    hxs : Membership.mem s x
    y : α
    hys : Membership.mem s y
    u : Set β
    hu : IsOpen u
    hu' : IsOpen (Set.preimage f u)
    hxu : Membership.mem (Set.preimage f u) x
    v : Set β
    hv : IsOpen v
    hv' : IsOpen (Set.preimage f v)
    hyv : Membership.mem (Set.preimage f v) y
    huv : HasSubset.Subset (Set.image f s) (Union.union u v)
    z : α
    hzs : Membership.mem s z
    hzuv : Membership.mem (Inter.inter u v) (f z)
    ⊢ (Inter.inter s (Inter.inter (Set.preimage f u) (Set.preimage f v))).Nonempty
  -/
  exact ⟨z, hzs, hzuv⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.isPreconnected_image := IsInducing.isPreconnected_image

/- TODO: The following lemmas about connection of preimages hold more generally for strict maps
(the quotient and subspace topologies of the image agree) whose fibers are preconnected. -/


theorem IsPreconnected.preimage_of_isOpenMap [TopologicalSpace β] {f : α → β} {s : Set β}
    (hs : IsPreconnected s) (hinj : Function.Injective f) (hf : IsOpenMap f) (hsf : s ⊆ range f) :
    IsPreconnected (f ⁻¹' s) := fun u v hu hv hsuv hsu hsv => by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set β
    hs : IsPreconnected s
    hinj : Function.Injective f
    hf : IsOpenMap f
    hsf : HasSubset.Subset s (Set.range f)
    u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    hsuv : HasSubset.Subset (Set.preimage f s) (Union.union u v)
    hsu : (Inter.inter (Set.preimage f s) u).Nonempty
    hsv : (Inter.inter (Set.preimage f s) v).Nonempty
    ⊢ (Inter.inter (Set.preimage f s) (Inter.inter u v)).Nonempty
  -/
  replace hsf : f '' (f ⁻¹' s) = s := image_preimage_eq_of_subset hsf
  obtain ⟨_, has, ⟨a, hau, rfl⟩, hav⟩ : (s ∩ (f '' u ∩ f '' v)).Nonempty := by
    refine hs (f '' u) (f '' v) (hf u hu) (hf v hv) ?_ ?_ ?_
    · simpa only [hsf, image_union] using image_subset f hsuv
    · simpa only [image_preimage_inter] using hsu.image f
    · simpa only [image_preimage_inter] using hsv.image f
    /-
      case intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      hs : IsPreconnected s
      hinj : Function.Injective f
      hf : IsOpenMap f
      u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hsuv : HasSubset.Subset (Set.preimage f s) (Union.union u v)
      hsu : (Inter.inter (Set.preimage f s) u).Nonempty
      hsv : (Inter.inter (Set.preimage f s) v).Nonempty
      hsf : Eq (Set.image f (Set.preimage f s)) s
      a : α
      hau : Membership.mem u a
      has : Membership.mem s (f a)
      hav : Membership.mem (Set.image f v) (f a)
      ⊢ (Inter.inter (Set.preimage f s) (Inter.inter u v)).Nonempty
    -/
  · exact ⟨a, has, hau, hinj.mem_set_image.1 hav⟩
    /-
      🎉 no goals
    -/


theorem IsPreconnected.preimage_of_isClosedMap [TopologicalSpace β] {s : Set β}
    (hs : IsPreconnected s) {f : α → β} (hinj : Function.Injective f) (hf : IsClosedMap f)
    (hsf : s ⊆ range f) : IsPreconnected (f ⁻¹' s) :=
  isPreconnected_closed_iff.2 fun u v hu hv hsuv hsu hsv => by
    /-
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set β
      hs : IsPreconnected s
      f : α → β
      hinj : Function.Injective f
      hf : IsClosedMap f
      hsf : HasSubset.Subset s (Set.range f)
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hsuv : HasSubset.Subset (Set.preimage f s) (Union.union u v)
      hsu : (Inter.inter (Set.preimage f s) u).Nonempty
      hsv : (Inter.inter (Set.preimage f s) v).Nonempty
      ⊢ (Inter.inter (Set.preimage f s) (Inter.inter u v)).Nonempty
    -/
    replace hsf : f '' (f ⁻¹' s) = s := image_preimage_eq_of_subset hsf
    obtain ⟨_, has, ⟨a, hau, rfl⟩, hav⟩ : (s ∩ (f '' u ∩ f '' v)).Nonempty := by
      refine isPreconnected_closed_iff.1 hs (f '' u) (f '' v) (hf u hu) (hf v hv) ?_ ?_ ?_
      · simpa only [hsf, image_union] using image_subset f hsuv
      · simpa only [image_preimage_inter] using hsu.image f
      · simpa only [image_preimage_inter] using hsv.image f
      /-
        case intro.intro.intro.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        s : Set β
        hs : IsPreconnected s
        f : α → β
        hinj : Function.Injective f
        hf : IsClosedMap f
        u v : Set α
        hu : IsClosed u
        hv : IsClosed v
        hsuv : HasSubset.Subset (Set.preimage f s) (Union.union u v)
        hsu : (Inter.inter (Set.preimage f s) u).Nonempty
        hsv : (Inter.inter (Set.preimage f s) v).Nonempty
        hsf : Eq (Set.image f (Set.preimage f s)) s
        a : α
        hau : Membership.mem u a
        has : Membership.mem s (f a)
        hav : Membership.mem (Set.image f v) (f a)
        ⊢ (Inter.inter (Set.preimage f s) (Inter.inter u v)).Nonempty
      -/
    · exact ⟨a, has, hau, hinj.mem_set_image.1 hav⟩
      /-
        🎉 no goals
      -/


theorem IsConnected.preimage_of_isOpenMap [TopologicalSpace β] {s : Set β} (hs : IsConnected s)
    {f : α → β} (hinj : Function.Injective f) (hf : IsOpenMap f) (hsf : s ⊆ range f) :
    IsConnected (f ⁻¹' s) :=
  ⟨hs.nonempty.preimage' hsf, hs.isPreconnected.preimage_of_isOpenMap hinj hf hsf⟩


theorem IsConnected.preimage_of_isClosedMap [TopologicalSpace β] {s : Set β} (hs : IsConnected s)
    {f : α → β} (hinj : Function.Injective f) (hf : IsClosedMap f) (hsf : s ⊆ range f) :
    IsConnected (f ⁻¹' s) :=
  ⟨hs.nonempty.preimage' hsf, hs.isPreconnected.preimage_of_isClosedMap hinj hf hsf⟩


theorem IsPreconnected.subset_or_subset (hu : IsOpen u) (hv : IsOpen v) (huv : Disjoint u v)
    (hsuv : s ⊆ u ∪ v) (hs : IsPreconnected s) : s ⊆ u ∨ s ⊆ v := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    huv : Disjoint u v
    hsuv : HasSubset.Subset s (Union.union u v)
    hs : IsPreconnected s
    ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
  -/
  specialize hs u v hu hv hsuv
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    huv : Disjoint u v
    hsuv : HasSubset.Subset s (Union.union u v)
    hs : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s  …
    ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
  -/
  obtain hsu | hsu := (s ∩ u).eq_empty_or_nonempty
    /-
      case inl
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      huv : Disjoint u v
      hsuv : HasSubset.Subset s (Union.union u v)
      hs : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s  …
      hsu : Eq (Inter.inter s u) EmptyCollection.emptyCollection
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
  · exact Or.inr ((Set.disjoint_iff_inter_eq_empty.2 hsu).subset_right_of_subset_union hsuv)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      huv : Disjoint u v
      hsuv : HasSubset.Subset s (Union.union u v)
      hs : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s  …
      hsu : (Inter.inter s u).Nonempty
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
  · replace hs := mt (hs hsu)
    simp_rw [Set.not_nonempty_iff_eq_empty, ← Set.disjoint_iff_inter_eq_empty,
      disjoint_iff_inter_eq_empty.1 huv] at hs
    /-
      case inr
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      huv : Disjoint u v
      hsuv : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hs : Disjoint s EmptyCollection.emptyCollection → Disjoint s v
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
    exact Or.inl ((hs s.disjoint_empty).subset_left_of_subset_union hsuv)
    /-
      🎉 no goals
    -/


theorem IsPreconnected.subset_left_of_subset_union (hu : IsOpen u) (hv : IsOpen v)
    (huv : Disjoint u v) (hsuv : s ⊆ u ∪ v) (hsu : (s ∩ u).Nonempty) (hs : IsPreconnected s) :
    s ⊆ u :=
  Disjoint.subset_left_of_subset_union hsuv
    (by
      /-
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : Disjoint u v
        hsuv : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hs : IsPreconnected s
        ⊢ Disjoint s v
      -/
      by_contra hsv
      /-
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : Disjoint u v
        hsuv : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hs : IsPreconnected s
        hsv : Not (Disjoint s v)
        ⊢ False
      -/
      rw [not_disjoint_iff_nonempty_inter] at hsv
      /-
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : Disjoint u v
        hsuv : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hs : IsPreconnected s
        hsv : (Inter.inter s v).Nonempty
        ⊢ False
      -/
      obtain ⟨x, _, hx⟩ := hs u v hu hv hsuv hsu hsv
      /-
        case intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        huv : Disjoint u v
        hsuv : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hs : IsPreconnected s
        hsv : (Inter.inter s v).Nonempty
        x : α
        left✝ : Membership.mem s x
        hx : Membership.mem (Inter.inter u v) x
        ⊢ False
      -/
      exact Set.disjoint_iff.1 huv hx)
      /-
        🎉 no goals
      -/


theorem IsPreconnected.subset_right_of_subset_union (hu : IsOpen u) (hv : IsOpen v)
    (huv : Disjoint u v) (hsuv : s ⊆ u ∪ v) (hsv : (s ∩ v).Nonempty) (hs : IsPreconnected s) :
    s ⊆ v :=
  hs.subset_left_of_subset_union hv hu huv.symm (union_comm u v ▸ hsuv) hsv


/-- If a preconnected set `s` intersects an open set `u`, and limit points of `u` inside `s` are
contained in `u`, then the whole set `s` is contained in `u`. -/
theorem IsPreconnected.subset_of_closure_inter_subset (hs : IsPreconnected s) (hu : IsOpen u)
    (h'u : (s ∩ u).Nonempty) (h : closure u ∩ s ⊆ u) : s ⊆ u := by
  have A : s ⊆ u ∪ (closure u)ᶜ := by
    intro x hx
    by_cases xu : x ∈ u
    · exact Or.inl xu
    · right
      intro h'x
      exact xu (h (mem_inter h'x hx))
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s u : Set α
    hs : IsPreconnected s
    hu : IsOpen u
    h'u : (Inter.inter s u).Nonempty
    h : HasSubset.Subset (Inter.inter (closure u) s) u
    A : HasSubset.Subset s (Union.union u (HasCompl.compl (closure u)))
    ⊢ HasSubset.Subset s u
  -/
  apply hs.subset_left_of_subset_union hu isClosed_closure.isOpen_compl _ A h'u
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s u : Set α
    hs : IsPreconnected s
    hu : IsOpen u
    h'u : (Inter.inter s u).Nonempty
    h : HasSubset.Subset (Inter.inter (closure u) s) u
    A : HasSubset.Subset s (Union.union u (HasCompl.compl (closure u)))
    ⊢ Disjoint u (HasCompl.compl (closure u))
  -/
  exact disjoint_compl_right.mono_right (compl_subset_compl.2 subset_closure)
  /-
    🎉 no goals
  -/


theorem IsPreconnected.prod [TopologicalSpace β] {s : Set α} {t : Set β} (hs : IsPreconnected s)
    (ht : IsPreconnected t) : IsPreconnected (s ×ˢ t) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    t : Set β
    hs : IsPreconnected s
    ht : IsPreconnected t
    ⊢ IsPreconnected (SProd.sprod s t)
  -/
  apply isPreconnected_of_forall_pair
  /-
    case H
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    t : Set β
    hs : IsPreconnected s
    ht : IsPreconnected t
    ⊢ ∀ (x : Prod α β), Membership.mem (SProd.sprod s t) x → ∀ (y : Prod α β), Mem …
  -/
  rintro ⟨a₁, b₁⟩ ⟨ha₁, hb₁⟩ ⟨a₂, b₂⟩ ⟨ha₂, hb₂⟩
  /-
    case H.mk.intro.mk.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    t : Set β
    hs : IsPreconnected s
    ht : IsPreconnected t
    a₁ : α
    b₁ : β
    ha₁ : Membership.mem s { fst := a₁, snd := b₁ }.1
    hb₁ : Membership.mem t { fst := a₁, snd := b₁ }.2
    a₂ : α
    b₂ : β
    ha₂ : Membership.mem s { fst := a₂, snd := b₂ }.1
    hb₂ : Membership.mem t { fst := a₂, snd := b₂ }.2
    ⊢ Exists fun t_1 => And (HasSubset.Subset t_1 (SProd.sprod s t)) (And (Members …
  -/
  refine ⟨Prod.mk a₁ '' t ∪ flip Prod.mk b₂ '' s, ?_, .inl ⟨b₁, hb₁, rfl⟩, .inr ⟨a₂, ha₂, rfl⟩, ?_⟩
    /-
      case H.mk.intro.mk.intro.refine_1
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set α
      t : Set β
      hs : IsPreconnected s
      ht : IsPreconnected t
      a₁ : α
      b₁ : β
      ha₁ : Membership.mem s { fst := a₁, snd := b₁ }.1
      hb₁ : Membership.mem t { fst := a₁, snd := b₁ }.2
      a₂ : α
      b₂ : β
      ha₂ : Membership.mem s { fst := a₂, snd := b₂ }.1
      hb₂ : Membership.mem t { fst := a₂, snd := b₂ }.2
      ⊢ HasSubset.Subset (Union.union (Set.image (Prod.mk a₁) t) (Set.image (flip Pr …
    -/
  · rintro _ (⟨y, hy, rfl⟩ | ⟨x, hx, rfl⟩)
    /-
      case H.mk.intro.mk.intro.refine_1.inl.intro.intro
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set α
      t : Set β
      hs : IsPreconnected s
      ht : IsPreconnected t
      a₁ : α
      b₁ : β
      ha₁ : Membership.mem s { fst := a₁, snd := b₁ }.1
      hb₁ : Membership.mem t { fst := a₁, snd := b₁ }.2
      a₂ : α
      b₂ : β
      ha₂ : Membership.mem s { fst := a₂, snd := b₂ }.1
      hb₂ : Membership.mem t { fst := a₂, snd := b₂ }.2
      y : β
      hy : Membership.mem t y
      ⊢ Membership.mem (SProd.sprod s t) { fst := a₁, snd := y }
    -/
    exacts [⟨ha₁, hy⟩, ⟨hx, hb₂⟩]
    /-
      🎉 no goals
    -/
  · exact (ht.image _ (Continuous.Prod.mk _).continuousOn).union (a₁, b₂) ⟨b₂, hb₂, rfl⟩
      ⟨a₁, ha₁, rfl⟩ (hs.image _ (continuous_id.prod_mk continuous_const).continuousOn)


theorem IsConnected.prod [TopologicalSpace β] {s : Set α} {t : Set β} (hs : IsConnected s)
    (ht : IsConnected t) : IsConnected (s ×ˢ t) :=
  ⟨hs.1.prod ht.1, hs.2.prod ht.2⟩


theorem isPreconnected_univ_pi [∀ i, TopologicalSpace (π i)] {s : ∀ i, Set (π i)}
    (hs : ∀ i, IsPreconnected (s i)) : IsPreconnected (pi univ s) := by
  /-
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : (i : ι) → Set (π i)
    hs : ∀ (i : ι), IsPreconnected (s i)
    ⊢ IsPreconnected (Set.univ.pi s)
  -/
  rintro u v uo vo hsuv ⟨f, hfs, hfu⟩ ⟨g, hgs, hgv⟩
  classical
  rcases exists_finset_piecewise_mem_of_mem_nhds (uo.mem_nhds hfu) g with ⟨I, hI⟩
  induction' I using Finset.induction_on with i I _ ihI
  · refine ⟨g, hgs, ⟨?_, hgv⟩⟩
    simpa using hI
  · rw [Finset.piecewise_insert] at hI
    have := I.piecewise_mem_set_pi hfs hgs
    refine (hsuv this).elim ihI fun h => ?_
    set S := update (I.piecewise f g) i '' s i
    have hsub : S ⊆ pi univ s := by
      refine image_subset_iff.2 fun z hz => ?_
      rwa [update_preimage_univ_pi]
      exact fun j _ => this j trivial
    have hconn : IsPreconnected S :=
      (hs i).image _ (continuous_const.update i continuous_id).continuousOn
    have hSu : (S ∩ u).Nonempty := ⟨_, mem_image_of_mem _ (hfs _ trivial), hI⟩
    have hSv : (S ∩ v).Nonempty := ⟨_, ⟨_, this _ trivial, update_eq_self _ _⟩, h⟩
    refine (hconn u v uo vo (hsub.trans hsuv) hSu hSv).mono ?_
    exact inter_subset_inter_left _ hsub


@[simp]
theorem isConnected_univ_pi [∀ i, TopologicalSpace (π i)] {s : ∀ i, Set (π i)} :
    IsConnected (pi univ s) ↔ ∀ i, IsConnected (s i) := by
  /-
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : (i : ι) → Set (π i)
    ⊢ Iff (IsConnected (Set.univ.pi s)) (∀ (i : ι), IsConnected (s i))
  -/
  simp only [IsConnected, ← univ_pi_nonempty_iff, forall_and, and_congr_right_iff]
  /-
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : (i : ι) → Set (π i)
    ⊢ (Set.univ.pi s).Nonempty → Iff (IsPreconnected (Set.univ.pi s)) (∀ (x : ι),  …
  -/
  refine fun hne => ⟨fun hc i => ?_, isPreconnected_univ_pi⟩
  /-
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : (i : ι) → Set (π i)
    hne : (Set.univ.pi s).Nonempty
    hc : IsPreconnected (Set.univ.pi s)
    i : ι
    ⊢ IsPreconnected (s i)
  -/
  rw [← eval_image_univ_pi hne]
  /-
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : (i : ι) → Set (π i)
    hne : (Set.univ.pi s).Nonempty
    hc : IsPreconnected (Set.univ.pi s)
    i : ι
    ⊢ IsPreconnected (Set.image (fun f => f i) (Set.univ.pi s))
  -/
  exact hc.image _ (continuous_apply _).continuousOn
  /-
    🎉 no goals
  -/


/-- The connected component of a point is the maximal connected set
that contains this point. -/
def connectedComponent (x : α) : Set α :=
  ⋃₀ { s : Set α | IsPreconnected s ∧ x ∈ s }


open Classical in
/-- Given a set `F` in a topological space `α` and a point `x : α`, the connected
component of `x` in `F` is the connected component of `x` in the subtype `F` seen as
a set in `α`. This definition does not make sense if `x` is not in `F` so we return the
empty set in this case. -/
def connectedComponentIn (F : Set α) (x : α) : Set α :=
  if h : x ∈ F then (↑) '' connectedComponent (⟨x, h⟩ : F) else ∅


theorem connectedComponentIn_eq_image {F : Set α} {x : α} (h : x ∈ F) :
    connectedComponentIn F x = (↑) '' connectedComponent (⟨x, h⟩ : F) :=
  dif_pos h


theorem connectedComponentIn_eq_empty {F : Set α} {x : α} (h : x ∉ F) :
    connectedComponentIn F x = ∅ :=
  dif_neg h


theorem mem_connectedComponent {x : α} : x ∈ connectedComponent x :=
  mem_sUnion_of_mem (mem_singleton x) ⟨isPreconnected_singleton, mem_singleton x⟩


theorem mem_connectedComponentIn {x : α} {F : Set α} (hx : x ∈ F) :
    x ∈ connectedComponentIn F x := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    F : Set α
    hx : Membership.mem F x
    ⊢ Membership.mem (connectedComponentIn F x) x
  -/
  simp [connectedComponentIn_eq_image hx, mem_connectedComponent, hx]
  /-
    🎉 no goals
  -/


theorem connectedComponent_nonempty {x : α} : (connectedComponent x).Nonempty :=
  ⟨x, mem_connectedComponent⟩


theorem connectedComponentIn_nonempty_iff {x : α} {F : Set α} :
    (connectedComponentIn F x).Nonempty ↔ x ∈ F := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    F : Set α
    ⊢ Iff (connectedComponentIn F x).Nonempty (Membership.mem F x)
  -/
  rw [connectedComponentIn]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    F : Set α
    ⊢ Iff (dite (Membership.mem F x) (fun h => Set.image Subtype.val (connectedCom …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [connectedComponent_nonempty, *]
                /-
                  🎉 no goals
                -/


theorem connectedComponentIn_subset (F : Set α) (x : α) : connectedComponentIn F x ⊆ F := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    F : Set α
    x : α
    ⊢ HasSubset.Subset (connectedComponentIn F x) F
  -/
  rw [connectedComponentIn]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    F : Set α
    x : α
    ⊢ HasSubset.Subset (dite (Membership.mem F x) (fun h => Set.image Subtype.val  …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


theorem isPreconnected_connectedComponent {x : α} : IsPreconnected (connectedComponent x) :=
  isPreconnected_sUnion x _ (fun _ => And.right) fun _ => And.left


theorem isPreconnected_connectedComponentIn {x : α} {F : Set α} :
    IsPreconnected (connectedComponentIn F x) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    F : Set α
    ⊢ IsPreconnected (connectedComponentIn F x)
  -/
  rw [connectedComponentIn]; split_ifs
    /-
      case pos
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      F : Set α
      h✝ : Membership.mem F x
      ⊢ IsPreconnected (Set.image Subtype.val (connectedComponent ⟨x, h✝⟩))
    -/
  · exact IsInducing.subtypeVal.isPreconnected_image.mpr isPreconnected_connectedComponent
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      F : Set α
      h✝ : Not (Membership.mem F x)
      ⊢ IsPreconnected EmptyCollection.emptyCollection
    -/
  · exact isPreconnected_empty
    /-
      🎉 no goals
    -/


theorem isConnected_connectedComponent {x : α} : IsConnected (connectedComponent x) :=
  ⟨⟨x, mem_connectedComponent⟩, isPreconnected_connectedComponent⟩


theorem isConnected_connectedComponentIn_iff {x : α} {F : Set α} :
    IsConnected (connectedComponentIn F x) ↔ x ∈ F := by
  simp_rw [← connectedComponentIn_nonempty_iff, IsConnected, isPreconnected_connectedComponentIn,
    and_true]


theorem IsPreconnected.subset_connectedComponent {x : α} {s : Set α} (H1 : IsPreconnected s)
    (H2 : x ∈ s) : s ⊆ connectedComponent x := fun _z hz => mem_sUnion_of_mem hz ⟨H1, H2⟩


theorem IsPreconnected.subset_connectedComponentIn {x : α} {F : Set α} (hs : IsPreconnected s)
    (hxs : x ∈ s) (hsF : s ⊆ F) : s ⊆ connectedComponentIn F x := by
  have : IsPreconnected (((↑) : F → α) ⁻¹' s) := by
    refine IsInducing.subtypeVal.isPreconnected_image.mp ?_
    rwa [Subtype.image_preimage_coe, inter_eq_right.mpr hsF]
  have h2xs : (⟨x, hsF hxs⟩ : F) ∈ (↑) ⁻¹' s := by
    rw [mem_preimage]
    exact hxs
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    x : α
    F : Set α
    hs : IsPreconnected s
    hxs : Membership.mem s x
    hsF : HasSubset.Subset s F
    this : IsPreconnected (Set.preimage Subtype.val s)
    h2xs : Membership.mem (Set.preimage Subtype.val s) ⟨x, ⋯⟩
    ⊢ HasSubset.Subset s (connectedComponentIn F x)
  -/
  have := this.subset_connectedComponent h2xs
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    x : α
    F : Set α
    hs : IsPreconnected s
    hxs : Membership.mem s x
    hsF : HasSubset.Subset s F
    this✝ : IsPreconnected (Set.preimage Subtype.val s)
    h2xs : Membership.mem (Set.preimage Subtype.val s) ⟨x, ⋯⟩
    this : HasSubset.Subset (Set.preimage Subtype.val s) (connectedComponent ⟨x, ⋯⟩)
    ⊢ HasSubset.Subset s (connectedComponentIn F x)
  -/
  rw [connectedComponentIn_eq_image (hsF hxs)]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    x : α
    F : Set α
    hs : IsPreconnected s
    hxs : Membership.mem s x
    hsF : HasSubset.Subset s F
    this✝ : IsPreconnected (Set.preimage Subtype.val s)
    h2xs : Membership.mem (Set.preimage Subtype.val s) ⟨x, ⋯⟩
    this : HasSubset.Subset (Set.preimage Subtype.val s) (connectedComponent ⟨x, ⋯⟩)
    ⊢ HasSubset.Subset s (Set.image Subtype.val (connectedComponent ⟨x, ⋯⟩))
  -/
  refine Subset.trans ?_ (image_subset _ this)
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    x : α
    F : Set α
    hs : IsPreconnected s
    hxs : Membership.mem s x
    hsF : HasSubset.Subset s F
    this✝ : IsPreconnected (Set.preimage Subtype.val s)
    h2xs : Membership.mem (Set.preimage Subtype.val s) ⟨x, ⋯⟩
    this : HasSubset.Subset (Set.preimage Subtype.val s) (connectedComponent ⟨x, ⋯⟩)
    ⊢ HasSubset.Subset s (Set.image Subtype.val (Set.preimage Subtype.val s))
  -/
  rw [Subtype.image_preimage_coe, inter_eq_right.mpr hsF]
  /-
    🎉 no goals
  -/


theorem IsConnected.subset_connectedComponent {x : α} {s : Set α} (H1 : IsConnected s)
    (H2 : x ∈ s) : s ⊆ connectedComponent x :=
  H1.2.subset_connectedComponent H2


theorem IsPreconnected.connectedComponentIn {x : α} {F : Set α} (h : IsPreconnected F)
    (hx : x ∈ F) : connectedComponentIn F x = F :=
  (connectedComponentIn_subset F x).antisymm (h.subset_connectedComponentIn hx subset_rfl)


theorem connectedComponent_eq {x y : α} (h : y ∈ connectedComponent x) :
    connectedComponent x = connectedComponent y :=
  eq_of_subset_of_subset (isConnected_connectedComponent.subset_connectedComponent h)
    (isConnected_connectedComponent.subset_connectedComponent
      (Set.mem_of_mem_of_subset mem_connectedComponent
        (isConnected_connectedComponent.subset_connectedComponent h)))


theorem connectedComponent_eq_iff_mem {x y : α} :
    connectedComponent x = connectedComponent y ↔ x ∈ connectedComponent y :=
  ⟨fun h => h ▸ mem_connectedComponent, fun h => (connectedComponent_eq h).symm⟩


theorem connectedComponentIn_eq {x y : α} {F : Set α} (h : y ∈ connectedComponentIn F x) :
    connectedComponentIn F x = connectedComponentIn F y := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x y : α
    F : Set α
    h : Membership.mem (connectedComponentIn F x) y
    ⊢ Eq (connectedComponentIn F x) (connectedComponentIn F y)
  -/
  have hx : x ∈ F := connectedComponentIn_nonempty_iff.mp ⟨y, h⟩
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x y : α
    F : Set α
    h : Membership.mem (connectedComponentIn F x) y
    hx : Membership.mem F x
    ⊢ Eq (connectedComponentIn F x) (connectedComponentIn F y)
  -/
  simp_rw [connectedComponentIn_eq_image hx] at h ⊢
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x y : α
    F : Set α
    hx : Membership.mem F x
    h : Membership.mem (Set.image Subtype.val (connectedComponent ⟨x, hx⟩)) y
    ⊢ Eq (Set.image Subtype.val (connectedComponent ⟨x, hx⟩)) (connectedComponentI …
  -/
  obtain ⟨⟨y, hy⟩, h2y, rfl⟩ := h
  /-
    case intro.mk.intro
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    F : Set α
    hx : Membership.mem F x
    y : α
    hy : Membership.mem F y
    h2y : Membership.mem (connectedComponent ⟨x, hx⟩) ⟨y, hy⟩
    ⊢ Eq (Set.image Subtype.val (connectedComponent ⟨x, hx⟩)) (connectedComponentI …
  -/
  simp_rw [connectedComponentIn_eq_image hy, connectedComponent_eq h2y]
  /-
    🎉 no goals
  -/


theorem connectedComponentIn_univ (x : α) : connectedComponentIn univ x = connectedComponent x :=
  subset_antisymm
    (isPreconnected_connectedComponentIn.subset_connectedComponent <|
      mem_connectedComponentIn trivial)
    (isPreconnected_connectedComponent.subset_connectedComponentIn mem_connectedComponent <|
      subset_univ _)


theorem connectedComponent_disjoint {x y : α} (h : connectedComponent x ≠ connectedComponent y) :
    Disjoint (connectedComponent x) (connectedComponent y) :=
  Set.disjoint_left.2 fun _ h1 h2 =>
    h ((connectedComponent_eq h1).trans (connectedComponent_eq h2).symm)


theorem isClosed_connectedComponent {x : α} : IsClosed (connectedComponent x) :=
  closure_subset_iff_isClosed.1 <|
    isConnected_connectedComponent.closure.subset_connectedComponent <|
      subset_closure mem_connectedComponent


theorem Continuous.image_connectedComponent_subset [TopologicalSpace β] {f : α → β}
    (h : Continuous f) (a : α) : f '' connectedComponent a ⊆ connectedComponent (f a) :=
  (isConnected_connectedComponent.image f h.continuousOn).subset_connectedComponent
    ((mem_image f (connectedComponent a) (f a)).2 ⟨a, mem_connectedComponent, rfl⟩)


theorem Continuous.image_connectedComponentIn_subset [TopologicalSpace β] {f : α → β} {s : Set α}
    {a : α} (hf : Continuous f) (hx : a ∈ s) :
    f '' connectedComponentIn s a ⊆ connectedComponentIn (f '' s) (f a) :=
  (isPreconnected_connectedComponentIn.image _ hf.continuousOn).subset_connectedComponentIn
    (mem_image_of_mem _ <| mem_connectedComponentIn hx)
    (image_subset _ <| connectedComponentIn_subset _ _)


theorem Continuous.mapsTo_connectedComponent [TopologicalSpace β] {f : α → β} (h : Continuous f)
    (a : α) : MapsTo f (connectedComponent a) (connectedComponent (f a)) :=
  mapsTo'.2 <| h.image_connectedComponent_subset a


theorem Continuous.mapsTo_connectedComponentIn [TopologicalSpace β] {f : α → β} {s : Set α}
    (h : Continuous f) {a : α} (hx : a ∈ s) :
    MapsTo f (connectedComponentIn s a) (connectedComponentIn (f '' s) (f a)) :=
  mapsTo'.2 <| image_connectedComponentIn_subset h hx


theorem irreducibleComponent_subset_connectedComponent {x : α} :
    irreducibleComponent x ⊆ connectedComponent x :=
  isIrreducible_irreducibleComponent.isConnected.subset_connectedComponent mem_irreducibleComponent


@[mono]
theorem connectedComponentIn_mono (x : α) {F G : Set α} (h : F ⊆ G) :
    connectedComponentIn F x ⊆ connectedComponentIn G x := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    F G : Set α
    h : HasSubset.Subset F G
    ⊢ HasSubset.Subset (connectedComponentIn F x) (connectedComponentIn G x)
  -/
  by_cases hx : x ∈ F
  · rw [connectedComponentIn_eq_image hx, connectedComponentIn_eq_image (h hx), ←
      show ((↑) : G → α) ∘ inclusion h = (↑) from rfl, image_comp]
    /-
      case pos
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      F G : Set α
      h : HasSubset.Subset F G
      hx : Membership.mem F x
      ⊢ HasSubset.Subset (Set.image Subtype.val (Set.image (Set.inclusion h) (connec …
    -/
    exact image_subset _ ((continuous_inclusion h).image_connectedComponent_subset ⟨x, hx⟩)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      F G : Set α
      h : HasSubset.Subset F G
      hx : Not (Membership.mem F x)
      ⊢ HasSubset.Subset (connectedComponentIn F x) (connectedComponentIn G x)
    -/
  · rw [connectedComponentIn_eq_empty hx]
    /-
      case neg
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      F G : Set α
      h : HasSubset.Subset F G
      hx : Not (Membership.mem F x)
      ⊢ HasSubset.Subset EmptyCollection.emptyCollection (connectedComponentIn G x)
    -/
    exact Set.empty_subset _
    /-
      🎉 no goals
    -/


/-- A preconnected space is one where there is no non-trivial open partition. -/
class PreconnectedSpace (α : Type u) [TopologicalSpace α] : Prop where
  /-- The universal set `Set.univ` in a preconnected space is a preconnected set. -/
  isPreconnected_univ : IsPreconnected (univ : Set α)


/-- A connected space is a nonempty one where there is no non-trivial open partition. -/
class ConnectedSpace (α : Type u) [TopologicalSpace α] extends PreconnectedSpace α : Prop where
  /-- A connected space is nonempty. -/
  toNonempty : Nonempty α


theorem isConnected_univ [ConnectedSpace α] : IsConnected (univ : Set α) :=
  ⟨univ_nonempty, isPreconnected_univ⟩


lemma preconnectedSpace_iff_univ : PreconnectedSpace α ↔ IsPreconnected (univ : Set α) :=
  ⟨fun h ↦ h.1, fun h ↦ ⟨h⟩⟩


lemma connectedSpace_iff_univ : ConnectedSpace α ↔ IsConnected (univ : Set α) :=
  ⟨fun h ↦ ⟨univ_nonempty, h.1.1⟩,
   fun h ↦ ConnectedSpace.mk (toPreconnectedSpace := ⟨h.2⟩) ⟨h.1.some⟩⟩


theorem isPreconnected_range [TopologicalSpace β] [PreconnectedSpace α] {f : α → β}
    (h : Continuous f) : IsPreconnected (range f) :=
  @image_univ _ _ f ▸ isPreconnected_univ.image _ h.continuousOn


theorem isConnected_range [TopologicalSpace β] [ConnectedSpace α] {f : α → β} (h : Continuous f) :
    IsConnected (range f) :=
  ⟨range_nonempty f, isPreconnected_range h⟩


theorem Function.Surjective.connectedSpace [ConnectedSpace α] [TopologicalSpace β]
    {f : α → β} (hf : Surjective f) (hf' : Continuous f) : ConnectedSpace β := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : ConnectedSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Function.Surjective f
    hf' : Continuous f
    ⊢ ConnectedSpace β
  -/
  rw [connectedSpace_iff_univ, ← hf.range_eq]
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : ConnectedSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Function.Surjective f
    hf' : Continuous f
    ⊢ IsConnected (Set.range f)
  -/
  exact isConnected_range hf'
  /-
    🎉 no goals
  -/


instance Quotient.instConnectedSpace {s : Setoid α} [ConnectedSpace α] :
    ConnectedSpace (Quotient s) :=
  Quotient.mk'_surjective.connectedSpace continuous_coinduced_rng


theorem DenseRange.preconnectedSpace [TopologicalSpace β] [PreconnectedSpace α] {f : α → β}
    (hf : DenseRange f) (hc : Continuous f) : PreconnectedSpace β :=
  ⟨hf.closure_eq ▸ (isPreconnected_range hc).closure⟩


theorem connectedSpace_iff_connectedComponent :
    ConnectedSpace α ↔ ∃ x : α, connectedComponent x = univ := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (ConnectedSpace α) (Exists fun x => Eq (connectedComponent x) Set.univ)
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ ConnectedSpace α → Exists fun x => Eq (connectedComponent x) Set.univ
    -/
  · rintro ⟨⟨x⟩⟩
    exact
      ⟨x, eq_univ_of_univ_subset <| isPreconnected_univ.subset_connectedComponent (mem_univ x)⟩
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ (Exists fun x => Eq (connectedComponent x) Set.univ) → ConnectedSpace α
    -/
  · rintro ⟨x, h⟩
    haveI : PreconnectedSpace α :=
      ⟨by rw [← h]; exact isPreconnected_connectedComponent⟩
    /-
      case mpr.intro
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      h : Eq (connectedComponent x) Set.univ
      this : PreconnectedSpace α
      ⊢ ConnectedSpace α
    -/
    exact ⟨⟨x⟩⟩
    /-
      🎉 no goals
    -/


theorem preconnectedSpace_iff_connectedComponent :
    PreconnectedSpace α ↔ ∀ x : α, connectedComponent x = univ := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (PreconnectedSpace α) (∀ (x : α), Eq (connectedComponent x) Set.univ)
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ PreconnectedSpace α → ∀ (x : α), Eq (connectedComponent x) Set.univ
    -/
  · intro h x
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      h : PreconnectedSpace α
      x : α
      ⊢ Eq (connectedComponent x) Set.univ
    -/
    exact eq_univ_of_univ_subset <| isPreconnected_univ.subset_connectedComponent (mem_univ x)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ (∀ (x : α), Eq (connectedComponent x) Set.univ) → PreconnectedSpace α
    -/
  · intro h
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α), Eq (connectedComponent x) Set.univ
      ⊢ PreconnectedSpace α
    -/
    cases' isEmpty_or_nonempty α with hα hα
      /-
        case mpr.inl
        α : Type u
        inst✝ : TopologicalSpace α
        h : ∀ (x : α), Eq (connectedComponent x) Set.univ
        hα : IsEmpty α
        ⊢ PreconnectedSpace α
      -/
    · exact ⟨by rw [univ_eq_empty_iff.mpr hα]; exact isPreconnected_empty⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u
        inst✝ : TopologicalSpace α
        h : ∀ (x : α), Eq (connectedComponent x) Set.univ
        hα : Nonempty α
        ⊢ PreconnectedSpace α
      -/
    · exact ⟨by rw [← h (Classical.choice hα)]; exact isPreconnected_connectedComponent⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem PreconnectedSpace.connectedComponent_eq_univ {X : Type*} [TopologicalSpace X]
    [h : PreconnectedSpace X] (x : X) : connectedComponent x = univ :=
  preconnectedSpace_iff_connectedComponent.mp h x


instance [TopologicalSpace β] [PreconnectedSpace α] [PreconnectedSpace β] :
    PreconnectedSpace (α × β) :=
  ⟨by
    /-
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝³ : TopologicalSpace α
      s t u v : Set α
      inst✝² : TopologicalSpace β
      inst✝¹ : PreconnectedSpace α
      inst✝ : PreconnectedSpace β
      ⊢ IsPreconnected Set.univ
    -/
    rw [← univ_prod_univ]
    /-
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝³ : TopologicalSpace α
      s t u v : Set α
      inst✝² : TopologicalSpace β
      inst✝¹ : PreconnectedSpace α
      inst✝ : PreconnectedSpace β
      ⊢ IsPreconnected (SProd.sprod Set.univ Set.univ)
    -/
    exact isPreconnected_univ.prod isPreconnected_univ⟩
    /-
      🎉 no goals
    -/


instance [TopologicalSpace β] [ConnectedSpace α] [ConnectedSpace β] : ConnectedSpace (α × β) :=
  ⟨inferInstance⟩


instance [∀ i, TopologicalSpace (π i)] [∀ i, PreconnectedSpace (π i)] :
    PreconnectedSpace (∀ i, π i) :=
      /-
        α : Type u
        β : Type v
        ι : Type u_1
        π : ι → Type u_2
        inst✝² : TopologicalSpace α
        s t u v : Set α
        inst✝¹ : (i : ι) → TopologicalSpace (π i)
        inst✝ : ∀ (i : ι), PreconnectedSpace (π i)
        ⊢ IsPreconnected Set.univ
      -/
  ⟨by rw [← pi_univ univ]; exact isPreconnected_univ_pi fun i => isPreconnected_univ⟩
                           /-
                             🎉 no goals
                           -/


instance [∀ i, TopologicalSpace (π i)] [∀ i, ConnectedSpace (π i)] : ConnectedSpace (∀ i, π i) :=
  ⟨inferInstance⟩

-- see Note [lower instance priority]

instance (priority := 100) PreirreducibleSpace.preconnectedSpace (α : Type u) [TopologicalSpace α]
    [PreirreducibleSpace α] : PreconnectedSpace α :=
  ⟨isPreirreducible_univ.isPreconnected⟩

-- see Note [lower instance priority]

instance (priority := 100) IrreducibleSpace.connectedSpace (α : Type u) [TopologicalSpace α]
    [IrreducibleSpace α] : ConnectedSpace α where toNonempty := IrreducibleSpace.toNonempty


theorem Subtype.preconnectedSpace {s : Set α} (h : IsPreconnected s) : PreconnectedSpace s where
  isPreconnected_univ := by
    /-
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : IsPreconnected s
      ⊢ IsPreconnected Set.univ
    -/
    rwa [← IsInducing.subtypeVal.isPreconnected_image, image_univ, Subtype.range_val]
    /-
      🎉 no goals
    -/


theorem Subtype.connectedSpace {s : Set α} (h : IsConnected s) : ConnectedSpace s where
  toPreconnectedSpace := Subtype.preconnectedSpace h.isPreconnected
  toNonempty := h.nonempty.to_subtype


theorem isPreconnected_iff_preconnectedSpace {s : Set α} : IsPreconnected s ↔ PreconnectedSpace s :=
  ⟨Subtype.preconnectedSpace, fun h => by
    /-
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : PreconnectedSpace ↑s
      ⊢ IsPreconnected s
    -/
    simpa using isPreconnected_univ.image ((↑) : s → α) continuous_subtype_val.continuousOn⟩
    /-
      🎉 no goals
    -/


theorem isConnected_iff_connectedSpace {s : Set α} : IsConnected s ↔ ConnectedSpace s :=
  ⟨Subtype.connectedSpace, fun h =>
    ⟨nonempty_subtype.mp h.2, isPreconnected_iff_preconnectedSpace.mpr h.1⟩⟩


