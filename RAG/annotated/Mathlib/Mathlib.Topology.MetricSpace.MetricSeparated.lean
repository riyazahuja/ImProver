/-- Two sets in an (extended) metric space are called *metric separated* if the (extended) distance
between `x ∈ s` and `y ∈ t` is bounded from below by a positive constant. -/
def IsMetricSeparated {X : Type*} [EMetricSpace X] (s t : Set X) :=
  ∃ r, r ≠ 0 ∧ ∀ x ∈ s, ∀ y ∈ t, r ≤ edist x y


@[symm]
theorem symm (h : IsMetricSeparated s t) : IsMetricSeparated t s :=
  let ⟨r, r0, hr⟩ := h
  ⟨r, r0, fun y hy x hx => edist_comm x y ▸ hr x hx y hy⟩


theorem comm : IsMetricSeparated s t ↔ IsMetricSeparated t s :=
  ⟨symm, symm⟩


@[simp]
theorem empty_left (s : Set X) : IsMetricSeparated ∅ s :=
  ⟨1, one_ne_zero, fun _x => False.elim⟩


@[simp]
theorem empty_right (s : Set X) : IsMetricSeparated s ∅ :=
  (empty_left s).symm


protected theorem disjoint (h : IsMetricSeparated s t) : Disjoint s t :=
  let ⟨r, r0, hr⟩ := h
                                                  /-
                                                    X : Type u_1
                                                    inst✝ : EMetricSpace X
                                                    s t : Set X
                                                    h : IsMetricSeparated s t
                                                    r : ENNReal
                                                    r0 : Ne r 0
                                                    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
                                                    x : X
                                                    hx1 : Membership.mem s x
                                                    hx2 : Membership.mem t x
                                                    ⊢ Eq r 0
                                                  -/
  Set.disjoint_left.mpr fun x hx1 hx2 => r0 <| by simpa using hr x hx1 x hx2
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem subset_compl_right (h : IsMetricSeparated s t) : s ⊆ tᶜ := fun _ hs ht =>
  h.disjoint.le_bot ⟨hs, ht⟩


@[mono]
theorem mono {s' t'} (hs : s ⊆ s') (ht : t ⊆ t') :
    IsMetricSeparated s' t' → IsMetricSeparated s t := fun ⟨r, r0, hr⟩ =>
  ⟨r, r0, fun x hx y hy => hr x (hs hx) y (ht hy)⟩


theorem mono_left {s'} (h' : IsMetricSeparated s' t) (hs : s ⊆ s') : IsMetricSeparated s t :=
  h'.mono hs Subset.rfl


theorem mono_right {t'} (h' : IsMetricSeparated s t') (ht : t ⊆ t') : IsMetricSeparated s t :=
  h'.mono Subset.rfl ht


theorem union_left {s'} (h : IsMetricSeparated s t) (h' : IsMetricSeparated s' t) :
    IsMetricSeparated (s ∪ s') t := by
  /-
    X : Type u_1
    inst✝ : EMetricSpace X
    s t s' : Set X
    h : IsMetricSeparated s t
    h' : IsMetricSeparated s' t
    ⊢ IsMetricSeparated (Union.union s s') t
  -/
  rcases h, h' with ⟨⟨r, r0, hr⟩, ⟨r', r0', hr'⟩⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝ : EMetricSpace X
    s t s' : Set X
    r : ENNReal
    r0 : Ne r 0
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    r' : ENNReal
    r0' : Ne r' 0
    hr' : ∀ (x : X), Membership.mem s' x → ∀ (y : X), Membership.mem t y → LE.le r …
    ⊢ IsMetricSeparated (Union.union s s') t
  -/
  refine ⟨min r r', ?_, fun x hx y hy => hx.elim ?_ ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      inst✝ : EMetricSpace X
      s t s' : Set X
      r : ENNReal
      r0 : Ne r 0
      hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
      r' : ENNReal
      r0' : Ne r' 0
      hr' : ∀ (x : X), Membership.mem s' x → ∀ (y : X), Membership.mem t y → LE.le r …
      ⊢ Ne (Min.min r r') 0
    -/
  · rw [← pos_iff_ne_zero] at r0 r0' ⊢
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_1
      inst✝ : EMetricSpace X
      s t s' : Set X
      r : ENNReal
      r0 : LT.lt 0 r
      hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
      r' : ENNReal
      r0' : LT.lt 0 r'
      hr' : ∀ (x : X), Membership.mem s' x → ∀ (y : X), Membership.mem t y → LE.le r …
      ⊢ LT.lt 0 (Min.min r r')
    -/
    exact lt_min r0 r0'
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      X : Type u_1
      inst✝ : EMetricSpace X
      s t s' : Set X
      r : ENNReal
      r0 : Ne r 0
      hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
      r' : ENNReal
      r0' : Ne r' 0
      hr' : ∀ (x : X), Membership.mem s' x → ∀ (y : X), Membership.mem t y → LE.le r …
      x : X
      hx : Membership.mem (Union.union s s') x
      y : X
      hy : Membership.mem t y
      ⊢ Membership.mem s x → LE.le (Min.min r r') (EDist.edist x y)
    -/
  · exact fun hx => (min_le_left _ _).trans (hr _ hx _ hy)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_3
      X : Type u_1
      inst✝ : EMetricSpace X
      s t s' : Set X
      r : ENNReal
      r0 : Ne r 0
      hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
      r' : ENNReal
      r0' : Ne r' 0
      hr' : ∀ (x : X), Membership.mem s' x → ∀ (y : X), Membership.mem t y → LE.le r …
      x : X
      hx : Membership.mem (Union.union s s') x
      y : X
      hy : Membership.mem t y
      ⊢ Membership.mem s' x → LE.le (Min.min r r') (EDist.edist x y)
    -/
  · exact fun hx => (min_le_right _ _).trans (hr' _ hx _ hy)
    /-
      🎉 no goals
    -/


@[simp]
theorem union_left_iff {s'} :
    IsMetricSeparated (s ∪ s') t ↔ IsMetricSeparated s t ∧ IsMetricSeparated s' t :=
  ⟨fun h => ⟨h.mono_left subset_union_left, h.mono_left subset_union_right⟩, fun h =>
    h.1.union_left h.2⟩


theorem union_right {t'} (h : IsMetricSeparated s t) (h' : IsMetricSeparated s t') :
    IsMetricSeparated s (t ∪ t') :=
  (h.symm.union_left h'.symm).symm


@[simp]
theorem union_right_iff {t'} :
    IsMetricSeparated s (t ∪ t') ↔ IsMetricSeparated s t ∧ IsMetricSeparated s t' :=
  comm.trans <| union_left_iff.trans <| and_congr comm comm


theorem finite_iUnion_left_iff {ι : Type*} {I : Set ι} (hI : I.Finite) {s : ι → Set X}
    {t : Set X} : IsMetricSeparated (⋃ i ∈ I, s i) t ↔ ∀ i ∈ I, IsMetricSeparated (s i) t := by
  /-
    X : Type u_1
    inst✝ : EMetricSpace X
    ι : Type u_2
    I : Set ι
    hI : I.Finite
    s : ι → Set X
    t : Set X
    ⊢ Iff (IsMetricSeparated (Set.iUnion fun i => Set.iUnion fun h => s i) t) (∀ ( …
  -/
  refine Finite.induction_on hI (by simp) @fun i I _ _ hI => ?_
  /-
    X : Type u_1
    inst✝ : EMetricSpace X
    ι : Type u_2
    I✝ : Set ι
    hI✝ : I✝.Finite
    s : ι → Set X
    t : Set X
    i : ι
    I : Set ι
    x✝¹ : Not (Membership.mem I i)
    x✝ : I.Finite
    hI : Iff (IsMetricSeparated (Set.iUnion fun i => Set.iUnion fun h => s i) t) ( …
    ⊢ Iff (IsMetricSeparated (Set.iUnion fun i_1 => Set.iUnion fun h => s i_1) t)  …
  -/
  rw [biUnion_insert, forall_mem_insert, union_left_iff, hI]
  /-
    🎉 no goals
  -/


alias ⟨_, finite_iUnion_left⟩ := finite_iUnion_left_iff


theorem finite_iUnion_right_iff {ι : Type*} {I : Set ι} (hI : I.Finite) {s : Set X}
    {t : ι → Set X} : IsMetricSeparated s (⋃ i ∈ I, t i) ↔ ∀ i ∈ I, IsMetricSeparated s (t i) := by
  /-
    X : Type u_1
    inst✝ : EMetricSpace X
    ι : Type u_2
    I : Set ι
    hI : I.Finite
    s : Set X
    t : ι → Set X
    ⊢ Iff (IsMetricSeparated s (Set.iUnion fun i => Set.iUnion fun h => t i)) (∀ ( …
  -/
  simpa only [@comm _ _ s] using finite_iUnion_left_iff hI
  /-
    🎉 no goals
  -/


@[simp]
theorem finset_iUnion_left_iff {ι : Type*} {I : Finset ι} {s : ι → Set X} {t : Set X} :
    IsMetricSeparated (⋃ i ∈ I, s i) t ↔ ∀ i ∈ I, IsMetricSeparated (s i) t :=
  finite_iUnion_left_iff I.finite_toSet


alias ⟨_, finset_iUnion_left⟩ := finset_iUnion_left_iff


@[simp]
theorem finset_iUnion_right_iff {ι : Type*} {I : Finset ι} {s : Set X} {t : ι → Set X} :
    IsMetricSeparated s (⋃ i ∈ I, t i) ↔ ∀ i ∈ I, IsMetricSeparated s (t i) :=
  finite_iUnion_right_iff I.finite_toSet


alias ⟨_, finset_iUnion_right⟩ := finset_iUnion_right_iff


