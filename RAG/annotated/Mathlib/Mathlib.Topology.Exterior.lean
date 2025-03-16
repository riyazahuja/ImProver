                                                                              /-
                                                                                X : Type u_2
                                                                                inst✝ : TopologicalSpace X
                                                                                x : X
                                                                                ⊢ Eq (exterior (Singleton.singleton x)) (nhds x).ker
                                                                              -/
lemma exterior_singleton_eq_ker_nhds (x : X) : exterior {x} = (𝓝 x).ker := by simp [exterior]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem mem_exterior_singleton : x ∈ exterior {y} ↔ x ⤳ y := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    x y : X
    ⊢ Iff (Membership.mem (exterior (Singleton.singleton y)) x) (Specializes x y)
  -/
  rw [exterior_singleton_eq_ker_nhds, ker_nhds_eq_specializes, mem_setOf]
  /-
    🎉 no goals
  -/


lemma exterior_def (s : Set X) : exterior s = ⋂₀ {t : Set X | IsOpen t ∧ s ⊆ t} :=
  (hasBasis_nhdsSet _).ker.trans sInter_eq_biInter.symm


                                                                          /-
                                                                            X : Type u_2
                                                                            inst✝ : TopologicalSpace X
                                                                            s : Set X
                                                                            x : X
                                                                            ⊢ Iff (Membership.mem (exterior s) x) (∀ (U : Set X), IsOpen U → HasSubset.Sub …
                                                                          -/
lemma mem_exterior : x ∈ exterior s ↔ ∀ U, IsOpen U → s ⊆ U → x ∈ U := by simp [exterior_def]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma subset_exterior_iff : s ⊆ exterior t ↔ ∀ U, IsOpen U → t ⊆ U → s ⊆ U := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ Iff (HasSubset.Subset s (exterior t)) (∀ (U : Set X), IsOpen U → HasSubset.S …
  -/
  simp [exterior_def]
  /-
    🎉 no goals
  -/


lemma subset_exterior : s ⊆ exterior s := subset_exterior_iff.2 fun _ _ ↦ id


lemma exterior_minimal (h₁ : s ⊆ t) (h₂ : IsOpen t) : exterior s ⊆ t := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s t : Set X
    h₁ : HasSubset.Subset s t
    h₂ : IsOpen t
    ⊢ HasSubset.Subset (exterior s) t
  -/
  rw [exterior_def]; exact sInter_subset_of_mem ⟨h₂, h₁⟩
                     /-
                       🎉 no goals
                     -/


lemma IsOpen.exterior_eq (h : IsOpen s) : exterior s = s :=
  (exterior_minimal Subset.rfl h).antisymm subset_exterior


lemma IsOpen.exterior_subset (ht : IsOpen t) : exterior s ⊆ t ↔ s ⊆ t :=
  ⟨subset_exterior.trans, fun h ↦ exterior_minimal h ht⟩


@[deprecated (since := "2024-09-18")] alias IsOpen.exterior_subset_iff := IsOpen.exterior_subset


@[simp]
theorem exterior_iUnion (s : ι → Set X) : exterior (⋃ i, s i) = ⋃ i, exterior (s i) := by
  /-
    ι : Sort u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    s : ι → Set X
    ⊢ Eq (exterior (Set.iUnion fun i => s i)) (Set.iUnion fun i => exterior (s i))
  -/
  simp only [exterior, nhdsSet_iUnion, ker_iSup]
  /-
    🎉 no goals
  -/


@[simp]
theorem exterior_union (s t : Set X) : exterior (s ∪ t) = exterior s ∪ exterior t := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ Eq (exterior (Union.union s t)) (Union.union (exterior s) (exterior t))
  -/
  simp only [exterior, nhdsSet_union, ker_sup]
  /-
    🎉 no goals
  -/


@[simp]
theorem exterior_sUnion (S : Set (Set X)) : exterior (⋃₀ S) = ⋃ s ∈ S, exterior s := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    ⊢ Eq (exterior S.sUnion) (Set.iUnion fun s => Set.iUnion fun h => exterior s)
  -/
  simp only [sUnion_eq_biUnion, exterior_iUnion]
  /-
    🎉 no goals
  -/


theorem mem_exterior_iff_specializes : x ∈ exterior s ↔ ∃ y ∈ s, x ⤳ y := calc
                                                     /-
                                                       X : Type u_2
                                                       inst✝ : TopologicalSpace X
                                                       s : Set X
                                                       x : X
                                                       ⊢ Iff (Membership.mem (exterior s) x) (Membership.mem (exterior (Set.iUnion fu …
                                                     -/
  x ∈ exterior s ↔ x ∈ exterior (⋃ y ∈ s, {y}) := by simp
                                                     /-
                                                       🎉 no goals
                                                     -/
  _ ↔ ∃ y ∈ s, x ⤳ y := by
    /-
      X : Type u_2
      inst✝ : TopologicalSpace X
      s : Set X
      x : X
      ⊢ Iff (Membership.mem (exterior (Set.iUnion fun y => Set.iUnion fun h => Singl …
    -/
    simp only [exterior_iUnion, mem_exterior_singleton, mem_iUnion₂, exists_prop]
    /-
      🎉 no goals
    -/


@[mono] lemma exterior_mono : Monotone (exterior : Set X → Set X) :=
  fun _s _t h ↦ ker_mono <| nhdsSet_mono h


/-- This name was used to be used for the `Iff` version,
see `exterior_subset_exterior_iff_nhdsSet`.
-/
@[gcongr] lemma exterior_subset_exterior (h : s ⊆ t) : exterior s ⊆ exterior t := exterior_mono h


@[simp] lemma exterior_subset_exterior_iff_nhdsSet : exterior s ⊆ exterior t ↔ 𝓝ˢ s ≤ 𝓝ˢ t := by
  simp (config := {contextual := true}) only [subset_exterior_iff, (hasBasis_nhdsSet _).ge_iff,
    and_imp, IsOpen.mem_nhdsSet, IsOpen.exterior_subset]


theorem exterior_eq_exterior_iff_nhdsSet : exterior s = exterior t ↔ 𝓝ˢ s = 𝓝ˢ t := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ Iff (Eq (exterior s) (exterior t)) (Eq (nhdsSet s) (nhdsSet t))
  -/
  simp [le_antisymm_iff]
  /-
    🎉 no goals
  -/


lemma specializes_iff_exterior_subset : x ⤳ y ↔ exterior {x} ⊆ exterior {y} := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    x y : X
    ⊢ Iff (Specializes x y) (HasSubset.Subset (exterior (Singleton.singleton x)) ( …
  -/
  simp [Specializes]
  /-
    🎉 no goals
  -/


theorem exterior_iInter_subset {s : ι → Set X} : exterior (⋂ i, s i) ⊆ ⋂ i, exterior (s i) :=
  exterior_mono.map_iInf_le


theorem exterior_inter_subset {s t : Set X} : exterior (s ∩ t) ⊆ exterior s ∩ exterior t :=
  exterior_mono.map_inf_le _ _


theorem exterior_sInter_subset {s : Set (Set X)} : exterior (⋂₀ s) ⊆ ⋂ x ∈ s, exterior x :=
  exterior_mono.map_sInf_le


@[simp] lemma exterior_empty : exterior (∅ : Set X) = ∅ := isOpen_empty.exterior_eq

@[simp] lemma exterior_univ : exterior (univ : Set X) = univ := isOpen_univ.exterior_eq


@[simp] lemma exterior_eq_empty : exterior s = ∅ ↔ s = ∅ :=
                                   /-
                                     X : Type u_2
                                     inst✝ : TopologicalSpace X
                                     s : Set X
                                     ⊢ Eq s EmptyCollection.emptyCollection → Eq (exterior s) EmptyCollection.empty …
                                   -/
  ⟨eq_bot_mono subset_exterior, by rintro rfl; exact exterior_empty⟩
                                               /-
                                                 🎉 no goals
                                               -/


@[simp] lemma nhdsSet_exterior (s : Set X) : 𝓝ˢ (exterior s) = 𝓝ˢ s := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (nhdsSet (exterior s)) (nhdsSet s)
  -/
  refine le_antisymm ((hasBasis_nhdsSet _).ge_iff.2 ?_) (nhdsSet_mono subset_exterior)
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ ∀ (i' : Set X), And (IsOpen i') (HasSubset.Subset s i') → Membership.mem (nh …
  -/
  exact fun U ⟨hUo, hsU⟩ ↦ hUo.mem_nhdsSet.2 <| hUo.exterior_subset.2 hsU
  /-
    🎉 no goals
  -/


@[simp] lemma exterior_exterior (s : Set X) : exterior (exterior s) = exterior s := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (exterior (exterior s)) (exterior s)
  -/
  simp only [exterior_eq_exterior_iff_nhdsSet, nhdsSet_exterior]
  /-
    🎉 no goals
  -/

