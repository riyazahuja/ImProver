/-- A constructor for topologies by specifying the closed sets,
and showing that they satisfy the appropriate conditions. -/
def TopologicalSpace.ofClosed {X : Type u} (T : Set (Set X)) (empty_mem : ∅ ∈ T)
    (sInter_mem : ∀ A, A ⊆ T → ⋂₀ A ∈ T)
    (union_mem : ∀ A, A ∈ T → ∀ B, B ∈ T → A ∪ B ∈ T) : TopologicalSpace X where
  IsOpen X := Xᶜ ∈ T
                    /-
                      X : Type u
                      T : Set (Set X)
                      empty_mem : Membership.mem T EmptyCollection.emptyCollection
                      sInter_mem : ∀ (A : Set (Set X)), HasSubset.Subset A T → Membership.mem T A.sI …
                      union_mem : ∀ (A : Set X), Membership.mem T A → ∀ (B : Set X), Membership.mem  …
                      ⊢ (fun X_1 => Membership.mem T (HasCompl.compl X_1)) Set.univ
                    -/
  isOpen_univ := by simp [empty_mem]
                    /-
                      🎉 no goals
                    -/
                               /-
                                 X : Type u
                                 T : Set (Set X)
                                 empty_mem : Membership.mem T EmptyCollection.emptyCollection
                                 sInter_mem : ∀ (A : Set (Set X)), HasSubset.Subset A T → Membership.mem T A.sI …
                                 union_mem : ∀ (A : Set X), Membership.mem T A → ∀ (B : Set X), Membership.mem  …
                                 s t : Set X
                                 hs : (fun X_1 => Membership.mem T (HasCompl.compl X_1)) s
                                 ht : (fun X_1 => Membership.mem T (HasCompl.compl X_1)) t
                                 ⊢ (fun X_1 => Membership.mem T (HasCompl.compl X_1)) (Inter.inter s t)
                               -/
  isOpen_inter s t hs ht := by simpa only [compl_inter] using union_mem sᶜ hs tᶜ ht
                               /-
                                 🎉 no goals
                               -/
  isOpen_sUnion s hs := by
    /-
      X : Type u
      T : Set (Set X)
      empty_mem : Membership.mem T EmptyCollection.emptyCollection
      sInter_mem : ∀ (A : Set (Set X)), HasSubset.Subset A T → Membership.mem T A.sI …
      union_mem : ∀ (A : Set X), Membership.mem T A → ∀ (B : Set X), Membership.mem  …
      s : Set (Set X)
      hs : ∀ (t : Set X), Membership.mem s t → (fun X_1 => Membership.mem T (HasComp …
      ⊢ (fun X_1 => Membership.mem T (HasCompl.compl X_1)) s.sUnion
    -/
    simp only [Set.compl_sUnion]
    /-
      X : Type u
      T : Set (Set X)
      empty_mem : Membership.mem T EmptyCollection.emptyCollection
      sInter_mem : ∀ (A : Set (Set X)), HasSubset.Subset A T → Membership.mem T A.sI …
      union_mem : ∀ (A : Set X), Membership.mem T A → ∀ (B : Set X), Membership.mem  …
      s : Set (Set X)
      hs : ∀ (t : Set X), Membership.mem s t → (fun X_1 => Membership.mem T (HasComp …
      ⊢ Membership.mem T (Set.image HasCompl.compl s).sInter
    -/
    exact sInter_mem (compl '' s) fun z ⟨y, hy, hz⟩ => hz ▸ hs y hy
    /-
      🎉 no goals
    -/


lemma isOpen_mk {p h₁ h₂ h₃} : IsOpen[⟨p, h₁, h₂, h₃⟩] s ↔ p s := Iff.rfl


@[ext (iff := false)]
protected theorem TopologicalSpace.ext :
    ∀ {f g : TopologicalSpace X}, IsOpen[f] = IsOpen[g] → f = g
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩, rfl => rfl


protected theorem TopologicalSpace.ext_iff {t t' : TopologicalSpace X} :
    t = t' ↔ ∀ s, IsOpen[t] s ↔ IsOpen[t'] s :=
                                       /-
                                         X : Type u
                                         t t' : TopologicalSpace X
                                         h : ∀ (s : Set X), Iff (IsOpen s) (IsOpen s)
                                         ⊢ Eq t t'
                                       -/
  ⟨fun h _ => h ▸ Iff.rfl, fun h => by ext; exact h _⟩
                                            /-
                                              🎉 no goals
                                            -/


theorem isOpen_fold {t : TopologicalSpace X} : t.IsOpen s = IsOpen[t] s :=
  rfl


theorem isOpen_iUnion {f : ι → Set X} (h : ∀ i, IsOpen (f i)) : IsOpen (⋃ i, f i) :=
  isOpen_sUnion (forall_mem_range.2 h)


theorem isOpen_biUnion {s : Set α} {f : α → Set X} (h : ∀ i ∈ s, IsOpen (f i)) :
    IsOpen (⋃ i ∈ s, f i) :=
  isOpen_iUnion fun i => isOpen_iUnion fun hi => h i hi


theorem IsOpen.union (h₁ : IsOpen s₁) (h₂ : IsOpen s₂) : IsOpen (s₁ ∪ s₂) := by
  /-
    X : Type u
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    h₁ : IsOpen s₁
    h₂ : IsOpen s₂
    ⊢ IsOpen (Union.union s₁ s₂)
  -/
  rw [union_eq_iUnion]; exact isOpen_iUnion (Bool.forall_bool.2 ⟨h₂, h₁⟩)
                        /-
                          🎉 no goals
                        -/


lemma isOpen_iff_of_cover {f : α → Set X} (ho : ∀ i, IsOpen (f i)) (hU : (⋃ i, f i) = univ) :
    IsOpen s ↔ ∀ i, IsOpen (f i ∩ s) := by
  /-
    X : Type u
    α : Type u_1
    s : Set X
    inst✝ : TopologicalSpace X
    f : α → Set X
    ho : ∀ (i : α), IsOpen (f i)
    hU : Eq (Set.iUnion fun i => f i) Set.univ
    ⊢ Iff (IsOpen s) (∀ (i : α), IsOpen (Inter.inter (f i) s))
  -/
  refine ⟨fun h i ↦ (ho i).inter h, fun h ↦ ?_⟩
  /-
    X : Type u
    α : Type u_1
    s : Set X
    inst✝ : TopologicalSpace X
    f : α → Set X
    ho : ∀ (i : α), IsOpen (f i)
    hU : Eq (Set.iUnion fun i => f i) Set.univ
    h : ∀ (i : α), IsOpen (Inter.inter (f i) s)
    ⊢ IsOpen s
  -/
  rw [← s.inter_univ, inter_comm, ← hU, iUnion_inter]
  /-
    X : Type u
    α : Type u_1
    s : Set X
    inst✝ : TopologicalSpace X
    f : α → Set X
    ho : ∀ (i : α), IsOpen (f i)
    hU : Eq (Set.iUnion fun i => f i) Set.univ
    h : ∀ (i : α), IsOpen (Inter.inter (f i) s)
    ⊢ IsOpen (Set.iUnion fun i => Inter.inter (f i) s)
  -/
  exact isOpen_iUnion fun i ↦ h i
  /-
    🎉 no goals
  -/


@[simp] theorem isOpen_empty : IsOpen (∅ : Set X) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ IsOpen EmptyCollection.emptyCollection
  -/
  rw [← sUnion_empty]; exact isOpen_sUnion fun a => False.elim
                       /-
                         🎉 no goals
                       -/


theorem Set.Finite.isOpen_sInter {s : Set (Set X)} (hs : s.Finite) :
    (∀ t ∈ s, IsOpen t) → IsOpen (⋂₀ s) :=
                                      /-
                                        X : Type u
                                        inst✝ : TopologicalSpace X
                                        s : Set (Set X)
                                        hs : s.Finite
                                        x✝ : ∀ (t : Set X), Membership.mem EmptyCollection.emptyCollection t → IsOpen t
                                        ⊢ IsOpen EmptyCollection.emptyCollection.sInter
                                      -/
  Finite.induction_on hs (fun _ => by rw [sInter_empty]; exact isOpen_univ) fun _ _ ih h => by
                                                         /-
                                                           🎉 no goals
                                                         -/
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set (Set X)
      hs : s.Finite
      a✝ : Set X
      s✝ : Set (Set X)
      x✝¹ : Not (Membership.mem s✝ a✝)
      x✝ : s✝.Finite
      ih : (∀ (t : Set X), Membership.mem s✝ t → IsOpen t) → IsOpen s✝.sInter
      h : ∀ (t : Set X), Membership.mem (Insert.insert a✝ s✝) t → IsOpen t
      ⊢ IsOpen (Insert.insert a✝ s✝).sInter
    -/
    simp only [sInter_insert, forall_mem_insert] at h ⊢
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set (Set X)
      hs : s.Finite
      a✝ : Set X
      s✝ : Set (Set X)
      x✝¹ : Not (Membership.mem s✝ a✝)
      x✝ : s✝.Finite
      ih : (∀ (t : Set X), Membership.mem s✝ t → IsOpen t) → IsOpen s✝.sInter
      h : And (IsOpen a✝) (∀ (x : Set X), Membership.mem s✝ x → IsOpen x)
      ⊢ IsOpen (Inter.inter a✝ s✝.sInter)
    -/
    exact h.1.inter (ih h.2)
    /-
      🎉 no goals
    -/


theorem Set.Finite.isOpen_biInter {s : Set α} {f : α → Set X} (hs : s.Finite)
    (h : ∀ i ∈ s, IsOpen (f i)) :
    IsOpen (⋂ i ∈ s, f i) :=
  sInter_image f s ▸ (hs.image _).isOpen_sInter (forall_mem_image.2 h)


theorem isOpen_iInter_of_finite [Finite ι] {s : ι → Set X} (h : ∀ i, IsOpen (s i)) :
    IsOpen (⋂ i, s i) :=
  (finite_range _).isOpen_sInter (forall_mem_range.2 h)


theorem isOpen_biInter_finset {s : Finset α} {f : α → Set X} (h : ∀ i ∈ s, IsOpen (f i)) :
    IsOpen (⋂ i ∈ s, f i) :=
  s.finite_toSet.isOpen_biInter h


@[simp] -- Porting note: added `simp`
                                                              /-
                                                                X : Type u
                                                                inst✝ : TopologicalSpace X
                                                                p : Prop
                                                                ⊢ IsOpen (setOf fun _x => p)
                                                              -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
theorem isOpen_const {p : Prop} : IsOpen { _x : X | p } := by by_cases p <;> simp [*]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem IsOpen.and : IsOpen { x | p₁ x } → IsOpen { x | p₂ x } → IsOpen { x | p₁ x ∧ p₂ x } :=
  IsOpen.inter


@[simp] theorem isOpen_compl_iff : IsOpen sᶜ ↔ IsClosed s :=
  ⟨fun h => ⟨h⟩, fun h => h.isOpen_compl⟩


theorem TopologicalSpace.ext_iff_isClosed {X} {t₁ t₂ : TopologicalSpace X} :
    t₁ = t₂ ↔ ∀ s, IsClosed[t₁] s ↔ IsClosed[t₂] s := by
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    ⊢ Iff (Eq t₁ t₂) (∀ (s : Set X), Iff (IsClosed s) (IsClosed s))
  -/
  rw [TopologicalSpace.ext_iff, compl_surjective.forall]
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    ⊢ Iff (∀ (x : Set X), Iff (IsOpen (HasCompl.compl x)) (IsOpen (HasCompl.compl  …
  -/
  simp only [@isOpen_compl_iff _ _ t₁, @isOpen_compl_iff _ _ t₂]
  /-
    🎉 no goals
  -/


alias ⟨_, TopologicalSpace.ext_isClosed⟩ := TopologicalSpace.ext_iff_isClosed


theorem isClosed_const {p : Prop} : IsClosed { _x : X | p } := ⟨isOpen_const (p := ¬p)⟩


@[simp] theorem isClosed_empty : IsClosed (∅ : Set X) := isClosed_const


@[simp] theorem isClosed_univ : IsClosed (univ : Set X) := isClosed_const


lemma IsOpen.isLocallyClosed (hs : IsOpen s) : IsLocallyClosed s :=
  ⟨_, _, hs, isClosed_univ, (inter_univ _).symm⟩


lemma IsClosed.isLocallyClosed (hs : IsClosed s) : IsLocallyClosed s :=
  ⟨_, _, isOpen_univ, hs, (univ_inter _).symm⟩


theorem IsClosed.union : IsClosed s₁ → IsClosed s₂ → IsClosed (s₁ ∪ s₂) := by
  /-
    X : Type u
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    ⊢ IsClosed s₁ → IsClosed s₂ → IsClosed (Union.union s₁ s₂)
  -/
  simpa only [← isOpen_compl_iff, compl_union] using IsOpen.inter
  /-
    🎉 no goals
  -/


theorem isClosed_sInter {s : Set (Set X)} : (∀ t ∈ s, IsClosed t) → IsClosed (⋂₀ s) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set (Set X)
    ⊢ (∀ (t : Set X), Membership.mem s t → IsClosed t) → IsClosed s.sInter
  -/
  simpa only [← isOpen_compl_iff, compl_sInter, sUnion_image] using isOpen_biUnion
  /-
    🎉 no goals
  -/


theorem isClosed_iInter {f : ι → Set X} (h : ∀ i, IsClosed (f i)) : IsClosed (⋂ i, f i) :=
  isClosed_sInter <| forall_mem_range.2 h


theorem isClosed_biInter {s : Set α} {f : α → Set X} (h : ∀ i ∈ s, IsClosed (f i)) :
    IsClosed (⋂ i ∈ s, f i) :=
  isClosed_iInter fun i => isClosed_iInter <| h i


@[simp]
theorem isClosed_compl_iff {s : Set X} : IsClosed sᶜ ↔ IsOpen s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsClosed (HasCompl.compl s)) (IsOpen s)
  -/
  rw [← isOpen_compl_iff, compl_compl]
  /-
    🎉 no goals
  -/


alias ⟨_, IsOpen.isClosed_compl⟩ := isClosed_compl_iff


theorem IsOpen.sdiff (h₁ : IsOpen s) (h₂ : IsClosed t) : IsOpen (s \ t) :=
  IsOpen.inter h₁ h₂.isOpen_compl


theorem IsClosed.inter (h₁ : IsClosed s₁) (h₂ : IsClosed s₂) : IsClosed (s₁ ∩ s₂) := by
  /-
    X : Type u
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    h₁ : IsClosed s₁
    h₂ : IsClosed s₂
    ⊢ IsClosed (Inter.inter s₁ s₂)
  -/
  rw [← isOpen_compl_iff] at *
  /-
    X : Type u
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    h₁ : IsOpen (HasCompl.compl s₁)
    h₂ : IsOpen (HasCompl.compl s₂)
    ⊢ IsOpen (HasCompl.compl (Inter.inter s₁ s₂))
  -/
  rw [compl_inter]
  /-
    X : Type u
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    h₁ : IsOpen (HasCompl.compl s₁)
    h₂ : IsOpen (HasCompl.compl s₂)
    ⊢ IsOpen (Union.union (HasCompl.compl s₁) (HasCompl.compl s₂))
  -/
  exact IsOpen.union h₁ h₂
  /-
    🎉 no goals
  -/


theorem IsClosed.sdiff (h₁ : IsClosed s) (h₂ : IsOpen t) : IsClosed (s \ t) :=
  IsClosed.inter h₁ (isClosed_compl_iff.mpr h₂)


theorem Set.Finite.isClosed_biUnion {s : Set α} {f : α → Set X} (hs : s.Finite)
    (h : ∀ i ∈ s, IsClosed (f i)) :
    IsClosed (⋃ i ∈ s, f i) := by
  /-
    X : Type u
    α : Type u_1
    inst✝ : TopologicalSpace X
    s : Set α
    f : α → Set X
    hs : s.Finite
    h : ∀ (i : α), Membership.mem s i → IsClosed (f i)
    ⊢ IsClosed (Set.iUnion fun i => Set.iUnion fun h => f i)
  -/
  simp only [← isOpen_compl_iff, compl_iUnion] at *
  /-
    X : Type u
    α : Type u_1
    inst✝ : TopologicalSpace X
    s : Set α
    f : α → Set X
    hs : s.Finite
    h : ∀ (i : α), Membership.mem s i → IsOpen (HasCompl.compl (f i))
    ⊢ IsOpen (Set.iInter fun i => Set.iInter fun i_1 => HasCompl.compl (f i))
  -/
  exact hs.isOpen_biInter h
  /-
    🎉 no goals
  -/


lemma isClosed_biUnion_finset {s : Finset α} {f : α → Set X} (h : ∀ i ∈ s, IsClosed (f i)) :
    IsClosed (⋃ i ∈ s, f i) :=
  s.finite_toSet.isClosed_biUnion h


theorem isClosed_iUnion_of_finite [Finite ι] {s : ι → Set X} (h : ∀ i, IsClosed (s i)) :
    IsClosed (⋃ i, s i) := by
  /-
    X : Type u
    ι : Sort w
    inst✝¹ : TopologicalSpace X
    inst✝ : Finite ι
    s : ι → Set X
    h : ∀ (i : ι), IsClosed (s i)
    ⊢ IsClosed (Set.iUnion fun i => s i)
  -/
  simp only [← isOpen_compl_iff, compl_iUnion] at *
  /-
    X : Type u
    ι : Sort w
    inst✝¹ : TopologicalSpace X
    inst✝ : Finite ι
    s : ι → Set X
    h : ∀ (i : ι), IsOpen (HasCompl.compl (s i))
    ⊢ IsOpen (Set.iInter fun i => HasCompl.compl (s i))
  -/
  exact isOpen_iInter_of_finite h
  /-
    🎉 no goals
  -/


theorem isClosed_imp {p q : X → Prop} (hp : IsOpen { x | p x }) (hq : IsClosed { x | q x }) :
    IsClosed { x | p x → q x } := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    p q : X → Prop
    hp : IsOpen (setOf fun x => p x)
    hq : IsClosed (setOf fun x => q x)
    ⊢ IsClosed (setOf fun x => p x → q x)
  -/
  simpa only [imp_iff_not_or] using hp.isClosed_compl.union hq
  /-
    🎉 no goals
  -/


theorem IsClosed.not : IsClosed { a | p a } → IsOpen { a | ¬p a } :=
  isOpen_compl_iff.mpr


theorem mem_interior : x ∈ interior s ↔ ∃ t ⊆ s, IsOpen t ∧ x ∈ t := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Membership.mem (interior s) x) (Exists fun t => And (HasSubset.Subset t …
  -/
  simp only [interior, mem_sUnion, mem_setOf_eq, and_assoc, and_left_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isOpen_interior : IsOpen (interior s) :=
  isOpen_sUnion fun _ => And.left


theorem interior_subset : interior s ⊆ s :=
  sUnion_subset fun _ => And.right


theorem interior_maximal (h₁ : t ⊆ s) (h₂ : IsOpen t) : t ⊆ interior s :=
  subset_sUnion_of_mem ⟨h₂, h₁⟩


theorem IsOpen.interior_eq (h : IsOpen s) : interior s = s :=
  interior_subset.antisymm (interior_maximal (Subset.refl s) h)


theorem interior_eq_iff_isOpen : interior s = s ↔ IsOpen s :=
  ⟨fun h => h ▸ isOpen_interior, IsOpen.interior_eq⟩


theorem subset_interior_iff_isOpen : s ⊆ interior s ↔ IsOpen s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (HasSubset.Subset s (interior s)) (IsOpen s)
  -/
  simp only [interior_eq_iff_isOpen.symm, Subset.antisymm_iff, interior_subset, true_and]
  /-
    🎉 no goals
  -/


theorem IsOpen.subset_interior_iff (h₁ : IsOpen s) : s ⊆ interior t ↔ s ⊆ t :=
  ⟨fun h => Subset.trans h interior_subset, fun h₂ => interior_maximal h₂ h₁⟩


theorem subset_interior_iff : t ⊆ interior s ↔ ∃ U, IsOpen U ∧ t ⊆ U ∧ U ⊆ s :=
  ⟨fun h => ⟨interior s, isOpen_interior, h, interior_subset⟩, fun ⟨_U, hU, htU, hUs⟩ =>
    htU.trans (interior_maximal hUs hU)⟩


lemma interior_subset_iff : interior s ⊆ t ↔ ∀ U, IsOpen U → U ⊆ s → U ⊆ t := by
  /-
    X : Type u
    s t : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (HasSubset.Subset (interior s) t) (∀ (U : Set X), IsOpen U → HasSubset.S …
  -/
  simp [interior]
  /-
    🎉 no goals
  -/


@[mono, gcongr]
theorem interior_mono (h : s ⊆ t) : interior s ⊆ interior t :=
  interior_maximal (Subset.trans interior_subset h) isOpen_interior


@[simp]
theorem interior_empty : interior (∅ : Set X) = ∅ :=
  isOpen_empty.interior_eq


@[simp]
theorem interior_univ : interior (univ : Set X) = univ :=
  isOpen_univ.interior_eq


@[simp]
theorem interior_eq_univ : interior s = univ ↔ s = univ :=
  ⟨fun h => univ_subset_iff.mp <| h.symm.trans_le interior_subset, fun h => h.symm ▸ interior_univ⟩


@[simp]
theorem interior_interior : interior (interior s) = interior s :=
  isOpen_interior.interior_eq


@[simp]
theorem interior_inter : interior (s ∩ t) = interior s ∩ interior t :=
  (Monotone.map_inf_le (fun _ _ ↦ interior_mono) s t).antisymm <|
    interior_maximal (inter_subset_inter interior_subset interior_subset) <|
      isOpen_interior.inter isOpen_interior


theorem Set.Finite.interior_biInter {ι : Type*} {s : Set ι} (hs : s.Finite) (f : ι → Set X) :
    interior (⋂ i ∈ s, f i) = ⋂ i ∈ s, interior (f i) :=
                      /-
                        X : Type u
                        inst✝ : TopologicalSpace X
                        ι : Type u_3
                        s : Set ι
                        hs : s.Finite
                        f : ι → Set X
                        ⊢ Eq (interior (Set.iInter fun i => Set.iInter fun h => f i)) (Set.iInter fun  …
                      -/
                      /-
                        🎉 no goals
                      -/
  hs.induction_on (by simp) <| by intros; simp [*]
                                          /-
                                            🎉 no goals
                                          -/


theorem Set.Finite.interior_sInter {S : Set (Set X)} (hS : S.Finite) :
    interior (⋂₀ S) = ⋂ s ∈ S, interior s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    hS : S.Finite
    ⊢ Eq (interior S.sInter) (Set.iInter fun s => Set.iInter fun h => interior s)
  -/
  rw [sInter_eq_biInter, hS.interior_biInter]
  /-
    🎉 no goals
  -/


@[simp]
theorem Finset.interior_iInter {ι : Type*} (s : Finset ι) (f : ι → Set X) :
    interior (⋂ i ∈ s, f i) = ⋂ i ∈ s, interior (f i) :=
  s.finite_toSet.interior_biInter f


@[simp]
theorem interior_iInter_of_finite [Finite ι] (f : ι → Set X) :
    interior (⋂ i, f i) = ⋂ i, interior (f i) := by
  /-
    X : Type u
    ι : Sort w
    inst✝¹ : TopologicalSpace X
    inst✝ : Finite ι
    f : ι → Set X
    ⊢ Eq (interior (Set.iInter fun i => f i)) (Set.iInter fun i => interior (f i))
  -/
  rw [← sInter_range, (finite_range f).interior_sInter, biInter_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_iInter₂_lt_nat {n : ℕ} (f : ℕ → Set X) :
    interior (⋂ m < n, f m) = ⋂ m < n, interior (f m) :=
  (finite_lt_nat n).interior_biInter f


@[simp]
theorem interior_iInter₂_le_nat {n : ℕ} (f : ℕ → Set X) :
    interior (⋂ m ≤ n, f m) = ⋂ m ≤ n, interior (f m) :=
  (finite_le_nat n).interior_biInter f


theorem interior_union_isClosed_of_interior_empty (h₁ : IsClosed s)
    (h₂ : interior t = ∅) : interior (s ∪ t) = interior s :=
  have : interior (s ∪ t) ⊆ s := fun x ⟨u, ⟨(hu₁ : IsOpen u), (hu₂ : u ⊆ s ∪ t)⟩, (hx₁ : x ∈ u)⟩ =>
    by_contradiction fun hx₂ : x ∉ s =>
      have : u \ s ⊆ t := fun _ ⟨h₁, h₂⟩ => Or.resolve_left (hu₂ h₁) h₂
                                      /-
                                        X : Type u
                                        s t : Set X
                                        inst✝ : TopologicalSpace X
                                        h₁ : IsClosed s
                                        h₂ : Eq (interior t) EmptyCollection.emptyCollection
                                        x : X
                                        x✝ : Membership.mem (interior (Union.union s t)) x
                                        u : Set X
                                        hu₁ : IsOpen u
                                        hu₂ : HasSubset.Subset u (Union.union s t)
                                        hx₁ : Membership.mem u x
                                        hx₂ : Not (Membership.mem s x)
                                        this : HasSubset.Subset (SDiff.sdiff u s) t
                                        ⊢ HasSubset.Subset (SDiff.sdiff u s) (interior t)
                                      -/
      have : u \ s ⊆ interior t := by rwa [(IsOpen.sdiff hu₁ h₁).subset_interior_iff]
                                      /-
                                        🎉 no goals
                                      -/
                             /-
                               X : Type u
                               s t : Set X
                               inst✝ : TopologicalSpace X
                               h₁ : IsClosed s
                               h₂ : Eq (interior t) EmptyCollection.emptyCollection
                               x : X
                               x✝ : Membership.mem (interior (Union.union s t)) x
                               u : Set X
                               hu₁ : IsOpen u
                               hu₂ : HasSubset.Subset u (Union.union s t)
                               hx₁ : Membership.mem u x
                               hx₂ : Not (Membership.mem s x)
                               this✝ : HasSubset.Subset (SDiff.sdiff u s) t
                               this : HasSubset.Subset (SDiff.sdiff u s) (interior t)
                               ⊢ HasSubset.Subset (SDiff.sdiff u s) EmptyCollection.emptyCollection
                             -/
      have : u \ s ⊆ ∅ := by rwa [h₂] at this
                             /-
                               🎉 no goals
                             -/
      this ⟨hx₁, hx₂⟩
  Subset.antisymm (interior_maximal this isOpen_interior) (interior_mono subset_union_left)


theorem isOpen_iff_forall_mem_open : IsOpen s ↔ ∀ x ∈ s, ∃ t, t ⊆ s ∧ IsOpen t ∧ x ∈ t := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (IsOpen s) (∀ (x : X), Membership.mem s x → Exists fun t => And (HasSubs …
  -/
  rw [← subset_interior_iff_isOpen]
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (HasSubset.Subset s (interior s)) (∀ (x : X), Membership.mem s x → Exist …
  -/
  simp only [subset_def, mem_interior]
  /-
    🎉 no goals
  -/


theorem interior_iInter_subset (s : ι → Set X) : interior (⋂ i, s i) ⊆ ⋂ i, interior (s i) :=
  subset_iInter fun _ => interior_mono <| iInter_subset _ _


theorem interior_iInter₂_subset (p : ι → Sort*) (s : ∀ i, p i → Set X) :
    interior (⋂ (i) (j), s i j) ⊆ ⋂ (i) (j), interior (s i j) :=
  (interior_iInter_subset _).trans <| iInter_mono fun _ => interior_iInter_subset _


theorem interior_sInter_subset (S : Set (Set X)) : interior (⋂₀ S) ⊆ ⋂ s ∈ S, interior s :=
  calc
                                                  /-
                                                    X : Type u
                                                    inst✝ : TopologicalSpace X
                                                    S : Set (Set X)
                                                    ⊢ Eq (interior S.sInter) (interior (Set.iInter fun s => Set.iInter fun h => s))
                                                  -/
    interior (⋂₀ S) = interior (⋂ s ∈ S, s) := by rw [sInter_eq_biInter]
                                                  /-
                                                    🎉 no goals
                                                  -/
    _ ⊆ ⋂ s ∈ S, interior s := interior_iInter₂_subset _ _


theorem Filter.HasBasis.lift'_interior {l : Filter X} {p : ι → Prop} {s : ι → Set X}
    (h : l.HasBasis p s) : (l.lift' interior).HasBasis p fun i => interior (s i) :=
  h.lift' fun _ _ ↦ interior_mono


theorem Filter.lift'_interior_le (l : Filter X) : l.lift' interior ≤ l := fun _s hs ↦
  mem_of_superset (mem_lift' hs) interior_subset


theorem Filter.HasBasis.lift'_interior_eq_self {l : Filter X} {p : ι → Prop} {s : ι → Set X}
    (h : l.HasBasis p s) (ho : ∀ i, p i → IsOpen (s i)) : l.lift' interior = l :=
  le_antisymm l.lift'_interior_le <| h.lift'_interior.ge_iff.2 fun i hi ↦ by
    /-
      X : Type u
      ι : Sort w
      inst✝ : TopologicalSpace X
      l : Filter X
      p : ι → Prop
      s : ι → Set X
      h : l.HasBasis p s
      ho : ∀ (i : ι), p i → IsOpen (s i)
      i : ι
      hi : p i
      ⊢ Membership.mem l (interior (s i))
    -/
    simpa only [(ho i hi).interior_eq] using h.mem_of_mem hi
    /-
      🎉 no goals
    -/


@[simp]
theorem isClosed_closure : IsClosed (closure s) :=
  isClosed_sInter fun _ => And.left


theorem subset_closure : s ⊆ closure s :=
  subset_sInter fun _ => And.right


theorem not_mem_of_not_mem_closure {P : X} (hP : P ∉ closure s) : P ∉ s := fun h =>
  hP (subset_closure h)


theorem closure_minimal (h₁ : s ⊆ t) (h₂ : IsClosed t) : closure s ⊆ t :=
  sInter_subset_of_mem ⟨h₂, h₁⟩


theorem Disjoint.closure_left (hd : Disjoint s t) (ht : IsOpen t) :
    Disjoint (closure s) t :=
  disjoint_compl_left.mono_left <| closure_minimal hd.subset_compl_right ht.isClosed_compl


theorem Disjoint.closure_right (hd : Disjoint s t) (hs : IsOpen s) :
    Disjoint s (closure t) :=
  (hd.symm.closure_left hs).symm


theorem IsClosed.closure_eq (h : IsClosed s) : closure s = s :=
  Subset.antisymm (closure_minimal (Subset.refl s) h) subset_closure


theorem IsClosed.closure_subset (hs : IsClosed s) : closure s ⊆ s :=
  closure_minimal (Subset.refl _) hs


theorem IsClosed.closure_subset_iff (h₁ : IsClosed t) : closure s ⊆ t ↔ s ⊆ t :=
  ⟨Subset.trans subset_closure, fun h => closure_minimal h h₁⟩


theorem IsClosed.mem_iff_closure_subset (hs : IsClosed s) :
    x ∈ s ↔ closure ({x} : Set X) ⊆ s :=
  (hs.closure_subset_iff.trans Set.singleton_subset_iff).symm


@[mono, gcongr]
theorem closure_mono (h : s ⊆ t) : closure s ⊆ closure t :=
  closure_minimal (Subset.trans h subset_closure) isClosed_closure


theorem monotone_closure (X : Type*) [TopologicalSpace X] : Monotone (@closure X _) := fun _ _ =>
  closure_mono


theorem diff_subset_closure_iff : s \ t ⊆ closure t ↔ s ⊆ closure t := by
  /-
    X : Type u
    s t : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (HasSubset.Subset (SDiff.sdiff s t) (closure t)) (HasSubset.Subset s (cl …
  -/
  rw [diff_subset_iff, union_eq_self_of_subset_left subset_closure]
  /-
    🎉 no goals
  -/


theorem closure_inter_subset_inter_closure (s t : Set X) :
    closure (s ∩ t) ⊆ closure s ∩ closure t :=
  (monotone_closure X).map_inf_le s t


theorem isClosed_of_closure_subset (h : closure s ⊆ s) : IsClosed s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    h : HasSubset.Subset (closure s) s
    ⊢ IsClosed s
  -/
  rw [subset_closure.antisymm h]; exact isClosed_closure
                                  /-
                                    🎉 no goals
                                  -/


theorem closure_eq_iff_isClosed : closure s = s ↔ IsClosed s :=
  ⟨fun h => h ▸ isClosed_closure, IsClosed.closure_eq⟩


theorem closure_subset_iff_isClosed : closure s ⊆ s ↔ IsClosed s :=
  ⟨isClosed_of_closure_subset, IsClosed.closure_subset⟩


@[simp]
theorem closure_empty : closure (∅ : Set X) = ∅ :=
  isClosed_empty.closure_eq


@[simp]
theorem closure_empty_iff (s : Set X) : closure s = ∅ ↔ s = ∅ :=
  ⟨subset_eq_empty subset_closure, fun h => h.symm ▸ closure_empty⟩


@[simp]
theorem closure_nonempty_iff : (closure s).Nonempty ↔ s.Nonempty := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (closure s).Nonempty s.Nonempty
  -/
  simp only [nonempty_iff_ne_empty, Ne, closure_empty_iff]
  /-
    🎉 no goals
  -/


alias ⟨Set.Nonempty.of_closure, Set.Nonempty.closure⟩ := closure_nonempty_iff


@[simp]
theorem closure_univ : closure (univ : Set X) = univ :=
  isClosed_univ.closure_eq


@[simp]
theorem closure_closure : closure (closure s) = closure s :=
  isClosed_closure.closure_eq


theorem closure_eq_compl_interior_compl : closure s = (interior sᶜ)ᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (closure s) (HasCompl.compl (interior (HasCompl.compl s)))
  -/
  rw [interior, closure, compl_sUnion, compl_image_set_of]
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (setOf fun t => And (IsClosed t) (HasSubset.Subset s t)).sInter (setOf fu …
  -/
  simp only [compl_subset_compl, isOpen_compl_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_union : closure (s ∪ t) = closure s ∪ closure t := by
  /-
    X : Type u
    s t : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (closure (Union.union s t)) (Union.union (closure s) (closure t))
  -/
  simp [closure_eq_compl_interior_compl, compl_inter]
  /-
    🎉 no goals
  -/


theorem Set.Finite.closure_biUnion {ι : Type*} {s : Set ι} (hs : s.Finite) (f : ι → Set X) :
    closure (⋃ i ∈ s, f i) = ⋃ i ∈ s, closure (f i) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type u_3
    s : Set ι
    hs : s.Finite
    f : ι → Set X
    ⊢ Eq (closure (Set.iUnion fun i => Set.iUnion fun h => f i)) (Set.iUnion fun i …
  -/
  simp [closure_eq_compl_interior_compl, hs.interior_biInter]
  /-
    🎉 no goals
  -/


theorem Set.Finite.closure_sUnion {S : Set (Set X)} (hS : S.Finite) :
    closure (⋃₀ S) = ⋃ s ∈ S, closure s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    hS : S.Finite
    ⊢ Eq (closure S.sUnion) (Set.iUnion fun s => Set.iUnion fun h => closure s)
  -/
  rw [sUnion_eq_biUnion, hS.closure_biUnion]
  /-
    🎉 no goals
  -/


@[simp]
theorem Finset.closure_biUnion {ι : Type*} (s : Finset ι) (f : ι → Set X) :
    closure (⋃ i ∈ s, f i) = ⋃ i ∈ s, closure (f i) :=
  s.finite_toSet.closure_biUnion f


@[simp]
theorem closure_iUnion_of_finite [Finite ι] (f : ι → Set X) :
    closure (⋃ i, f i) = ⋃ i, closure (f i) := by
  /-
    X : Type u
    ι : Sort w
    inst✝¹ : TopologicalSpace X
    inst✝ : Finite ι
    f : ι → Set X
    ⊢ Eq (closure (Set.iUnion fun i => f i)) (Set.iUnion fun i => closure (f i))
  -/
  rw [← sUnion_range, (finite_range _).closure_sUnion, biUnion_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_iUnion₂_lt_nat {n : ℕ} (f : ℕ → Set X) :
    closure (⋃ m < n, f m) = ⋃ m < n, closure (f m) :=
  (finite_lt_nat n).closure_biUnion f


@[simp]
theorem closure_iUnion₂_le_nat {n : ℕ} (f : ℕ → Set X) :
    closure (⋃ m ≤ n, f m) = ⋃ m ≤ n, closure (f m) :=
  (finite_le_nat n).closure_biUnion f


theorem interior_subset_closure : interior s ⊆ closure s :=
  Subset.trans interior_subset subset_closure


@[simp]
theorem interior_compl : interior sᶜ = (closure s)ᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (interior (HasCompl.compl s)) (HasCompl.compl (closure s))
  -/
  simp [closure_eq_compl_interior_compl]
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_compl : closure sᶜ = (interior s)ᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (closure (HasCompl.compl s)) (HasCompl.compl (interior s))
  -/
  simp [closure_eq_compl_interior_compl]
  /-
    🎉 no goals
  -/


theorem mem_closure_iff :
    x ∈ closure s ↔ ∀ o, IsOpen o → x ∈ o → (o ∩ s).Nonempty :=
  ⟨fun h o oo ao =>
    by_contradiction fun os =>
      have : s ⊆ oᶜ := fun x xs xo => os ⟨x, xo, xs⟩
      closure_minimal this (isClosed_compl_iff.2 oo) h ao,
    fun H _ ⟨h₁, h₂⟩ =>
    by_contradiction fun nc =>
      let ⟨_, hc, hs⟩ := H _ h₁.isOpen_compl nc
      hc (h₂ hs)⟩


theorem closure_inter_open_nonempty_iff (h : IsOpen t) :
    (closure s ∩ t).Nonempty ↔ (s ∩ t).Nonempty :=
  ⟨fun ⟨_x, hxcs, hxt⟩ => inter_comm t s ▸ mem_closure_iff.1 hxcs t h hxt, fun h =>
    h.mono <| inf_le_inf_right t subset_closure⟩


theorem Filter.le_lift'_closure (l : Filter X) : l ≤ l.lift' closure :=
  le_lift'.2 fun _ h => mem_of_superset h subset_closure


theorem Filter.HasBasis.lift'_closure {l : Filter X} {p : ι → Prop} {s : ι → Set X}
    (h : l.HasBasis p s) : (l.lift' closure).HasBasis p fun i => closure (s i) :=
  h.lift' (monotone_closure X)


theorem Filter.HasBasis.lift'_closure_eq_self {l : Filter X} {p : ι → Prop} {s : ι → Set X}
    (h : l.HasBasis p s) (hc : ∀ i, p i → IsClosed (s i)) : l.lift' closure = l :=
  le_antisymm (h.ge_iff.2 fun i hi => (hc i hi).closure_eq ▸ mem_lift' (h.mem_of_mem hi))
    l.le_lift'_closure


@[simp]
theorem Filter.lift'_closure_eq_bot {l : Filter X} : l.lift' closure = ⊥ ↔ l = ⊥ :=
  ⟨fun h => bot_unique <| h ▸ l.le_lift'_closure, fun h =>
                /-
                  X : Type u
                  inst✝ : TopologicalSpace X
                  l : Filter X
                  h : Eq l Bot.bot
                  ⊢ Eq (Bot.bot.lift' closure) Bot.bot
                -/
    h.symm ▸ by rw [lift'_bot (monotone_closure _), closure_empty, principal_empty]⟩
                /-
                  🎉 no goals
                -/


theorem dense_iff_closure_eq : Dense s ↔ closure s = univ :=
  eq_univ_iff_forall.symm


alias ⟨Dense.closure_eq, _⟩ := dense_iff_closure_eq


theorem interior_eq_empty_iff_dense_compl : interior s = ∅ ↔ Dense sᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Eq (interior s) EmptyCollection.emptyCollection) (Dense (HasCompl.compl …
  -/
  rw [dense_iff_closure_eq, closure_compl, compl_univ_iff]
  /-
    🎉 no goals
  -/


theorem Dense.interior_compl (h : Dense s) : interior sᶜ = ∅ :=
                                            /-
                                              X : Type u
                                              s : Set X
                                              inst✝ : TopologicalSpace X
                                              h : Dense s
                                              ⊢ Dense (HasCompl.compl (HasCompl.compl s))
                                            -/
  interior_eq_empty_iff_dense_compl.2 <| by rwa [compl_compl]
                                            /-
                                              🎉 no goals
                                            -/


/-- The closure of a set `s` is dense if and only if `s` is dense. -/
@[simp]
theorem dense_closure : Dense (closure s) ↔ Dense s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Dense (closure s)) (Dense s)
  -/
  rw [Dense, Dense, closure_closure]
  /-
    🎉 no goals
  -/


protected alias ⟨_, Dense.closure⟩ := dense_closure

alias ⟨Dense.of_closure, _⟩ := dense_closure


@[simp]
theorem dense_univ : Dense (univ : Set X) := fun _ => subset_closure trivial


/-- A set is dense if and only if it has a nonempty intersection with each nonempty open set. -/
theorem dense_iff_inter_open :
    Dense s ↔ ∀ U, IsOpen U → U.Nonempty → (U ∩ s).Nonempty := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Dense s) (∀ (U : Set X), IsOpen U → U.Nonempty → (Inter.inter U s).None …
  -/
  constructor <;> intro h
    /-
      case mp
      X : Type u
      s : Set X
      inst✝ : TopologicalSpace X
      h : Dense s
      ⊢ ∀ (U : Set X), IsOpen U → U.Nonempty → (Inter.inter U s).Nonempty
    -/
  · rintro U U_op ⟨x, x_in⟩
    /-
      case mp.intro
      X : Type u
      s : Set X
      inst✝ : TopologicalSpace X
      h : Dense s
      U : Set X
      U_op : IsOpen U
      x : X
      x_in : Membership.mem U x
      ⊢ (Inter.inter U s).Nonempty
    -/
    exact mem_closure_iff.1 (h _) U U_op x_in
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      s : Set X
      inst✝ : TopologicalSpace X
      h : ∀ (U : Set X), IsOpen U → U.Nonempty → (Inter.inter U s).Nonempty
      ⊢ Dense s
    -/
  · intro x
    /-
      case mpr
      X : Type u
      s : Set X
      inst✝ : TopologicalSpace X
      h : ∀ (U : Set X), IsOpen U → U.Nonempty → (Inter.inter U s).Nonempty
      x : X
      ⊢ Membership.mem (closure s) x
    -/
    rw [mem_closure_iff]
    /-
      case mpr
      X : Type u
      s : Set X
      inst✝ : TopologicalSpace X
      h : ∀ (U : Set X), IsOpen U → U.Nonempty → (Inter.inter U s).Nonempty
      x : X
      ⊢ ∀ (o : Set X), IsOpen o → Membership.mem o x → (Inter.inter o s).Nonempty
    -/
    intro U U_op x_in
    /-
      case mpr
      X : Type u
      s : Set X
      inst✝ : TopologicalSpace X
      h : ∀ (U : Set X), IsOpen U → U.Nonempty → (Inter.inter U s).Nonempty
      x : X
      U : Set X
      U_op : IsOpen U
      x_in : Membership.mem U x
      ⊢ (Inter.inter U s).Nonempty
    -/
    exact h U U_op ⟨_, x_in⟩
    /-
      🎉 no goals
    -/


alias ⟨Dense.inter_open_nonempty, _⟩ := dense_iff_inter_open


theorem Dense.exists_mem_open (hs : Dense s) {U : Set X} (ho : IsOpen U)
    (hne : U.Nonempty) : ∃ x ∈ s, x ∈ U :=
  let ⟨x, hx⟩ := hs.inter_open_nonempty U ho hne
  ⟨x, hx.2, hx.1⟩


theorem Dense.nonempty_iff (hs : Dense s) : s.Nonempty ↔ Nonempty X :=
  ⟨fun ⟨x, _⟩ => ⟨x⟩, fun ⟨x⟩ =>
    let ⟨y, hy⟩ := hs.inter_open_nonempty _ isOpen_univ ⟨x, trivial⟩
    ⟨y, hy.2⟩⟩


theorem Dense.nonempty [h : Nonempty X] (hs : Dense s) : s.Nonempty :=
  hs.nonempty_iff.2 h


@[mono]
theorem Dense.mono (h : s₁ ⊆ s₂) (hd : Dense s₁) : Dense s₂ := fun x =>
  closure_mono h (hd x)


/-- Complement to a singleton is dense if and only if the singleton is not an open set. -/
theorem dense_compl_singleton_iff_not_open :
    Dense ({x}ᶜ : Set X) ↔ ¬IsOpen ({x} : Set X) := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Dense (HasCompl.compl (Singleton.singleton x))) (Not (IsOpen (Singleton …
  -/
  constructor
    /-
      case mp
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      ⊢ Dense (HasCompl.compl (Singleton.singleton x)) → Not (IsOpen (Singleton.sing …
    -/
  · intro hd ho
    /-
      case mp
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      hd : Dense (HasCompl.compl (Singleton.singleton x))
      ho : IsOpen (Singleton.singleton x)
      ⊢ False
    -/
    exact (hd.inter_open_nonempty _ ho (singleton_nonempty _)).ne_empty (inter_compl_self _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      ⊢ Not (IsOpen (Singleton.singleton x)) → Dense (HasCompl.compl (Singleton.sing …
    -/
  · refine fun ho => dense_iff_inter_open.2 fun U hU hne => inter_compl_nonempty_iff.2 fun hUx => ?_
    /-
      case mpr
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      ho : Not (IsOpen (Singleton.singleton x))
      U : Set X
      hU : IsOpen U
      hne : U.Nonempty
      hUx : HasSubset.Subset U (Singleton.singleton x)
      ⊢ False
    -/
    obtain rfl : U = {x} := eq_singleton_iff_nonempty_unique_mem.2 ⟨hne, hUx⟩
    /-
      case mpr
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      ho : Not (IsOpen (Singleton.singleton x))
      hU : IsOpen (Singleton.singleton x)
      hne : (Singleton.singleton x).Nonempty
      hUx : HasSubset.Subset (Singleton.singleton x) (Singleton.singleton x)
      ⊢ False
    -/
    exact ho hU
    /-
      🎉 no goals
    -/


/-- If a closed property holds for a dense subset, it holds for the whole space. -/
@[elab_as_elim]
lemma Dense.induction (hs : Dense s) {P : X → Prop}
    (mem : ∀ x ∈ s, P x) (isClosed : IsClosed { x | P x }) (x : X) : P x :=
  hs.closure_eq.symm.subset.trans (isClosed.closure_subset_iff.mpr mem) trivial


theorem IsOpen.subset_interior_closure {s : Set X} (s_open : IsOpen s) :
    s ⊆ interior (closure s) := s_open.subset_interior_iff.mpr subset_closure


theorem IsClosed.closure_interior_subset {s : Set X} (s_closed : IsClosed s) :
    closure (interior s) ⊆ s := s_closed.closure_subset_iff.mpr interior_subset


@[simp]
theorem closure_diff_interior (s : Set X) : closure s \ interior s = frontier s :=
  rfl


/-- Interior and frontier are disjoint. -/
lemma disjoint_interior_frontier : Disjoint (interior s) (frontier s) := by
  rw [disjoint_iff_inter_eq_empty, ← closure_diff_interior, diff_eq,
    ← inter_assoc, inter_comm, ← inter_assoc, compl_inter_self, empty_inter]


@[simp]
theorem closure_diff_frontier (s : Set X) : closure s \ frontier s = interior s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (SDiff.sdiff (closure s) (frontier s)) (interior s)
  -/
  rw [frontier, diff_diff_right_self, inter_eq_self_of_subset_right interior_subset_closure]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_diff_frontier (s : Set X) : s \ frontier s = interior s := by
  rw [frontier, diff_diff_right, diff_eq_empty.2 subset_closure,
    inter_eq_self_of_subset_right interior_subset, empty_union]


theorem frontier_eq_closure_inter_closure : frontier s = closure s ∩ closure sᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (frontier s) (Inter.inter (closure s) (closure (HasCompl.compl s)))
  -/
  rw [closure_compl, frontier, diff_eq]
  /-
    🎉 no goals
  -/


theorem frontier_subset_closure : frontier s ⊆ closure s :=
  diff_subset


theorem frontier_subset_iff_isClosed : frontier s ⊆ s ↔ IsClosed s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (HasSubset.Subset (frontier s) s) (IsClosed s)
  -/
  rw [frontier, diff_subset_iff, union_eq_right.mpr interior_subset, closure_subset_iff_isClosed]
  /-
    🎉 no goals
  -/


alias ⟨_, IsClosed.frontier_subset⟩ := frontier_subset_iff_isClosed


theorem frontier_closure_subset : frontier (closure s) ⊆ frontier s :=
  diff_subset_diff closure_closure.subset <| interior_mono subset_closure


theorem frontier_interior_subset : frontier (interior s) ⊆ frontier s :=
  diff_subset_diff (closure_mono interior_subset) interior_interior.symm.subset


/-- The complement of a set has the same frontier as the original set. -/
@[simp]
theorem frontier_compl (s : Set X) : frontier sᶜ = frontier s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (frontier (HasCompl.compl s)) (frontier s)
  -/
  simp only [frontier_eq_closure_inter_closure, compl_compl, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            X : Type u
                                                            inst✝ : TopologicalSpace X
                                                            ⊢ Eq (frontier Set.univ) EmptyCollection.emptyCollection
                                                          -/
theorem frontier_univ : frontier (univ : Set X) = ∅ := by simp [frontier]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                        /-
                                                          X : Type u
                                                          inst✝ : TopologicalSpace X
                                                          ⊢ Eq (frontier EmptyCollection.emptyCollection) EmptyCollection.emptyCollection
                                                        -/
theorem frontier_empty : frontier (∅ : Set X) = ∅ := by simp [frontier]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem frontier_inter_subset (s t : Set X) :
    frontier (s ∩ t) ⊆ frontier s ∩ closure t ∪ closure s ∩ frontier t := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ HasSubset.Subset (frontier (Inter.inter s t)) (Union.union (Inter.inter (fro …
  -/
  simp only [frontier_eq_closure_inter_closure, compl_inter, closure_union]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ HasSubset.Subset (Inter.inter (closure (Inter.inter s t)) (Union.union (clos …
  -/
  refine (inter_subset_inter_left _ (closure_inter_subset_inter_closure s t)).trans_eq ?_
  simp only [inter_union_distrib_left, union_inter_distrib_right, inter_assoc,
    inter_comm (closure t)]


theorem frontier_union_subset (s t : Set X) :
    frontier (s ∪ t) ⊆ frontier s ∩ closure tᶜ ∪ closure sᶜ ∩ frontier t := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ HasSubset.Subset (frontier (Union.union s t)) (Union.union (Inter.inter (fro …
  -/
  simpa only [frontier_compl, ← compl_union] using frontier_inter_subset sᶜ tᶜ
  /-
    🎉 no goals
  -/


theorem IsClosed.frontier_eq (hs : IsClosed s) : frontier s = s \ interior s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    hs : IsClosed s
    ⊢ Eq (frontier s) (SDiff.sdiff s (interior s))
  -/
  rw [frontier, hs.closure_eq]
  /-
    🎉 no goals
  -/


theorem IsOpen.frontier_eq (hs : IsOpen s) : frontier s = closure s \ s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    hs : IsOpen s
    ⊢ Eq (frontier s) (SDiff.sdiff (closure s) s)
  -/
  rw [frontier, hs.interior_eq]
  /-
    🎉 no goals
  -/


theorem IsOpen.inter_frontier_eq (hs : IsOpen s) : s ∩ frontier s = ∅ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    hs : IsOpen s
    ⊢ Eq (Inter.inter s (frontier s)) EmptyCollection.emptyCollection
  -/
  rw [hs.frontier_eq, inter_diff_self]
  /-
    🎉 no goals
  -/


theorem disjoint_frontier_iff_isOpen : Disjoint (frontier s) s ↔ IsOpen s := by
  rw [← isClosed_compl_iff, ← frontier_subset_iff_isClosed,
    frontier_compl, subset_compl_iff_disjoint_right]


/-- The frontier of a set is closed. -/
theorem isClosed_frontier : IsClosed (frontier s) := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ IsClosed (frontier s)
  -/
  rw [frontier_eq_closure_inter_closure]; exact IsClosed.inter isClosed_closure isClosed_closure
                                          /-
                                            🎉 no goals
                                          -/


/-- The frontier of a closed set has no interior point. -/
theorem interior_frontier (h : IsClosed s) : interior (frontier s) = ∅ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    h : IsClosed s
    ⊢ Eq (interior (frontier s)) EmptyCollection.emptyCollection
  -/
  have A : frontier s = s \ interior s := h.frontier_eq
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    h : IsClosed s
    A : Eq (frontier s) (SDiff.sdiff s (interior s))
    ⊢ Eq (interior (frontier s)) EmptyCollection.emptyCollection
  -/
  have B : interior (frontier s) ⊆ interior s := by rw [A]; exact interior_mono diff_subset
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    h : IsClosed s
    A : Eq (frontier s) (SDiff.sdiff s (interior s))
    B : HasSubset.Subset (interior (frontier s)) (interior s)
    ⊢ Eq (interior (frontier s)) EmptyCollection.emptyCollection
  -/
  have C : interior (frontier s) ⊆ frontier s := interior_subset
  have : interior (frontier s) ⊆ interior s ∩ (s \ interior s) :=
    subset_inter B (by simpa [A] using C)
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    h : IsClosed s
    A : Eq (frontier s) (SDiff.sdiff s (interior s))
    B : HasSubset.Subset (interior (frontier s)) (interior s)
    C : HasSubset.Subset (interior (frontier s)) (frontier s)
    this : HasSubset.Subset (interior (frontier s)) (Inter.inter (interior s) (SDi …
    ⊢ Eq (interior (frontier s)) EmptyCollection.emptyCollection
  -/
  rwa [inter_diff_self, subset_empty_iff] at this
  /-
    🎉 no goals
  -/


theorem closure_eq_interior_union_frontier (s : Set X) : closure s = interior s ∪ frontier s :=
  (union_diff_cancel interior_subset_closure).symm


theorem closure_eq_self_union_frontier (s : Set X) : closure s = s ∪ frontier s :=
  (union_diff_cancel' interior_subset subset_closure).symm


theorem Disjoint.frontier_left (ht : IsOpen t) (hd : Disjoint s t) : Disjoint (frontier s) t :=
  subset_compl_iff_disjoint_right.1 <|
    frontier_subset_closure.trans <| closure_minimal (disjoint_left.1 hd) <| isClosed_compl_iff.2 ht


theorem Disjoint.frontier_right (hs : IsOpen s) (hd : Disjoint s t) : Disjoint s (frontier t) :=
  (hd.symm.frontier_left hs).symm


theorem frontier_eq_inter_compl_interior :
    frontier s = (interior s)ᶜ ∩ (interior sᶜ)ᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (frontier s) (Inter.inter (HasCompl.compl (interior s)) (HasCompl.compl ( …
  -/
  rw [← frontier_compl, ← closure_compl, ← diff_eq, closure_diff_interior]
  /-
    🎉 no goals
  -/


theorem compl_frontier_eq_union_interior :
    (frontier s)ᶜ = interior s ∪ interior sᶜ := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (HasCompl.compl (frontier s)) (Union.union (interior s) (interior (HasCom …
  -/
  rw [frontier_eq_inter_compl_interior]
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Eq (HasCompl.compl (Inter.inter (HasCompl.compl (interior s)) (HasCompl.comp …
  -/
  simp only [compl_inter, compl_compl]
  /-
    🎉 no goals
  -/


theorem nhds_def' (x : X) : 𝓝 x = ⨅ (s : Set X) (_ : IsOpen s) (_ : x ∈ s), 𝓟 s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Eq (nhds x) (iInf fun s => iInf fun x_1 => iInf fun x => Filter.principal s)
  -/
  simp only [nhds_def, mem_setOf_eq, @and_comm (x ∈ _), iInf_and]
  /-
    🎉 no goals
  -/


/-- The open sets containing `x` are a basis for the neighborhood filter. See `nhds_basis_opens'`
for a variant using open neighborhoods instead. -/
theorem nhds_basis_opens (x : X) :
    (𝓝 x).HasBasis (fun s : Set X => x ∈ s ∧ IsOpen s) fun s => s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ (nhds x).HasBasis (fun s => And (Membership.mem s x) (IsOpen s)) fun s => s
  -/
  rw [nhds_def]
  exact hasBasis_biInf_principal
    (fun s ⟨has, hs⟩ t ⟨hat, ht⟩ =>
      ⟨s ∩ t, ⟨⟨has, hat⟩, IsOpen.inter hs ht⟩, ⟨inter_subset_left, inter_subset_right⟩⟩)
    ⟨univ, ⟨mem_univ x, isOpen_univ⟩⟩


theorem nhds_basis_closeds (x : X) : (𝓝 x).HasBasis (fun s : Set X => x ∉ s ∧ IsClosed s) compl :=
  ⟨fun t => (nhds_basis_opens x).mem_iff.trans <|
                                        /-
                                          X : Type u
                                          inst✝ : TopologicalSpace X
                                          x : X
                                          t : Set X
                                          ⊢ Iff (Exists fun x_1 => And (And (Membership.mem (HasCompl.compl x_1) x) (IsO …
                                        -/
    compl_surjective.exists.trans <| by simp only [isOpen_compl_iff, mem_compl_iff]⟩
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem lift'_nhds_interior (x : X) : (𝓝 x).lift' interior = 𝓝 x :=
  (nhds_basis_opens x).lift'_interior_eq_self fun _ ↦ And.right


theorem Filter.HasBasis.nhds_interior {x : X} {p : ι → Prop} {s : ι → Set X}
    (h : (𝓝 x).HasBasis p s) : (𝓝 x).HasBasis p (interior <| s ·) :=
  lift'_nhds_interior x ▸ h.lift'_interior


/-- A filter lies below the neighborhood filter at `x` iff it contains every open set around `x`. -/
                                                                                /-
                                                                                  X : Type u
                                                                                  x : X
                                                                                  inst✝ : TopologicalSpace X
                                                                                  f : Filter X
                                                                                  ⊢ Iff (LE.le f (nhds x)) (∀ (s : Set X), Membership.mem s x → IsOpen s → Membe …
                                                                                -/
theorem le_nhds_iff {f} : f ≤ 𝓝 x ↔ ∀ s : Set X, x ∈ s → IsOpen s → s ∈ f := by simp [nhds_def]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- To show a filter is above the neighborhood filter at `x`, it suffices to show that it is above
the principal filter of some open set `s` containing `x`. -/
theorem nhds_le_of_le {f} (h : x ∈ s) (o : IsOpen s) (sf : 𝓟 s ≤ f) : 𝓝 x ≤ f := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    f : Filter X
    h : Membership.mem s x
    o : IsOpen s
    sf : LE.le (Filter.principal s) f
    ⊢ LE.le (nhds x) f
  -/
  rw [nhds_def]; exact iInf₂_le_of_le s ⟨h, o⟩ sf
                 /-
                   🎉 no goals
                 -/


theorem mem_nhds_iff : s ∈ 𝓝 x ↔ ∃ t ⊆ s, IsOpen t ∧ x ∈ t :=
  (nhds_basis_opens x).mem_iff.trans <| exists_congr fun _ =>
    ⟨fun h => ⟨h.2, h.1.2, h.1.1⟩, fun h => ⟨⟨h.2.2, h.2.1⟩, h.1⟩⟩


/-- A predicate is true in a neighborhood of `x` iff it is true for all the points in an open set
containing `x`. -/
theorem eventually_nhds_iff {p : X → Prop} :
    (∀ᶠ y in 𝓝 x, p y) ↔ ∃ t : Set X, (∀ y ∈ t, p y) ∧ IsOpen t ∧ x ∈ t :=
                           /-
                             X : Type u
                             x : X
                             inst✝ : TopologicalSpace X
                             p : X → Prop
                             ⊢ Iff (Exists fun t => And (HasSubset.Subset t (setOf fun x => (fun y => p y)  …
                           -/
  mem_nhds_iff.trans <| by simp only [subset_def, exists_prop, mem_setOf_eq]
                           /-
                             🎉 no goals
                           -/


theorem frequently_nhds_iff {p : X → Prop} :
    (∃ᶠ y in 𝓝 x, p y) ↔ ∀ U : Set X, x ∈ U → IsOpen U → ∃ y ∈ U, p y :=
                                                  /-
                                                    X : Type u
                                                    x : X
                                                    inst✝ : TopologicalSpace X
                                                    p : X → Prop
                                                    ⊢ Iff (∀ (i : Set X), And (Membership.mem i x) (IsOpen i) → Exists fun x => An …
                                                  -/
  (nhds_basis_opens x).frequently_iff.trans <| by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem mem_interior_iff_mem_nhds : x ∈ interior s ↔ s ∈ 𝓝 x :=
  mem_interior.trans mem_nhds_iff.symm


theorem map_nhds {f : X → α} :
    map f (𝓝 x) = ⨅ s ∈ { s : Set X | x ∈ s ∧ IsOpen s }, 𝓟 (f '' s) :=
  ((nhds_basis_opens x).map f).eq_biInf


theorem mem_of_mem_nhds : s ∈ 𝓝 x → x ∈ s := fun H =>
  let ⟨_t, ht, _, hs⟩ := mem_nhds_iff.1 H; ht hs


/-- If a predicate is true in a neighborhood of `x`, then it is true for `x`. -/
theorem Filter.Eventually.self_of_nhds {p : X → Prop} (h : ∀ᶠ y in 𝓝 x, p y) : p x :=
  mem_of_mem_nhds h


theorem IsOpen.mem_nhds (hs : IsOpen s) (hx : x ∈ s) : s ∈ 𝓝 x :=
  mem_nhds_iff.2 ⟨s, Subset.refl _, hs, hx⟩


protected theorem IsOpen.mem_nhds_iff (hs : IsOpen s) : s ∈ 𝓝 x ↔ x ∈ s :=
  ⟨mem_of_mem_nhds, fun hx => mem_nhds_iff.2 ⟨s, Subset.rfl, hs, hx⟩⟩


theorem IsClosed.compl_mem_nhds (hs : IsClosed s) (hx : x ∉ s) : sᶜ ∈ 𝓝 x :=
  hs.isOpen_compl.mem_nhds (mem_compl hx)


theorem IsOpen.eventually_mem (hs : IsOpen s) (hx : x ∈ s) :
    ∀ᶠ x in 𝓝 x, x ∈ s :=
  IsOpen.mem_nhds hs hx


/-- The open neighborhoods of `x` are a basis for the neighborhood filter. See `nhds_basis_opens`
for a variant using open sets around `x` instead. -/
theorem nhds_basis_opens' (x : X) :
    (𝓝 x).HasBasis (fun s : Set X => s ∈ 𝓝 x ∧ IsOpen s) fun x => x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ (nhds x).HasBasis (fun s => And (Membership.mem (nhds x) s) (IsOpen s)) fun  …
  -/
  convert nhds_basis_opens x using 2
  /-
    case h.e'_4.h.a
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    x✝ : Set X
    ⊢ Iff (And (Membership.mem (nhds x) x✝) (IsOpen x✝)) (And (Membership.mem x✝ x …
  -/
  exact and_congr_left_iff.2 IsOpen.mem_nhds_iff
  /-
    🎉 no goals
  -/


/-- If `U` is a neighborhood of each point of a set `s` then it is a neighborhood of `s`:
it contains an open set containing `s`. -/
theorem exists_open_set_nhds {U : Set X} (h : ∀ x ∈ s, U ∈ 𝓝 x) :
    ∃ V : Set X, s ⊆ V ∧ IsOpen V ∧ V ⊆ U :=
  ⟨interior U, fun x hx => mem_interior_iff_mem_nhds.2 <| h x hx, isOpen_interior, interior_subset⟩


/-- If `U` is a neighborhood of each point of a set `s` then it is a neighborhood of s:
it contains an open set containing `s`. -/
theorem exists_open_set_nhds' {U : Set X} (h : U ∈ ⨆ x ∈ s, 𝓝 x) :
    ∃ V : Set X, s ⊆ V ∧ IsOpen V ∧ V ⊆ U :=
                           /-
                             X : Type u
                             s : Set X
                             inst✝ : TopologicalSpace X
                             U : Set X
                             h : Membership.mem (iSup fun x => iSup fun h => nhds x) U
                             ⊢ ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) U
                           -/
  exists_open_set_nhds (by simpa using h)
                           /-
                             🎉 no goals
                           -/


/-- If a predicate is true in a neighbourhood of `x`, then for `y` sufficiently close
to `x` this predicate is true in a neighbourhood of `y`. -/
theorem Filter.Eventually.eventually_nhds {p : X → Prop} (h : ∀ᶠ y in 𝓝 x, p y) :
    ∀ᶠ y in 𝓝 x, ∀ᶠ x in 𝓝 y, p x :=
  let ⟨t, htp, hto, ha⟩ := eventually_nhds_iff.1 h
  eventually_nhds_iff.2 ⟨t, fun _x hx => eventually_nhds_iff.2 ⟨t, htp, hto, hx⟩, hto, ha⟩


@[simp]
theorem eventually_eventually_nhds {p : X → Prop} :
    (∀ᶠ y in 𝓝 x, ∀ᶠ x in 𝓝 y, p x) ↔ ∀ᶠ x in 𝓝 x, p x :=
  ⟨fun h => h.self_of_nhds, fun h => h.eventually_nhds⟩


@[simp]
theorem frequently_frequently_nhds {p : X → Prop} :
    (∃ᶠ x' in 𝓝 x, ∃ᶠ x'' in 𝓝 x', p x'') ↔ ∃ᶠ x in 𝓝 x, p x := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    p : X → Prop
    ⊢ Iff (Filter.Frequently (fun x' => Filter.Frequently (fun x'' => p x'') (nhds …
  -/
  rw [← not_iff_not]
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    p : X → Prop
    ⊢ Iff (Not (Filter.Frequently (fun x' => Filter.Frequently (fun x'' => p x'')  …
  -/
  simp only [not_frequently, eventually_eventually_nhds]
  /-
    🎉 no goals
  -/


@[simp]
theorem eventually_mem_nhds_iff : (∀ᶠ x' in 𝓝 x, s ∈ 𝓝 x') ↔ s ∈ 𝓝 x :=
  eventually_eventually_nhds


@[deprecated (since := "2024-10-04")] alias eventually_mem_nhds := eventually_mem_nhds_iff


@[simp]
theorem nhds_bind_nhds : (𝓝 x).bind 𝓝 = 𝓝 x :=
  Filter.ext fun _ => eventually_eventually_nhds


@[simp]
theorem eventually_eventuallyEq_nhds {f g : X → α} :
    (∀ᶠ y in 𝓝 x, f =ᶠ[𝓝 y] g) ↔ f =ᶠ[𝓝 x] g :=
  eventually_eventually_nhds


theorem Filter.EventuallyEq.eq_of_nhds {f g : X → α} (h : f =ᶠ[𝓝 x] g) : f x = g x :=
  h.self_of_nhds


@[simp]
theorem eventually_eventuallyLE_nhds [LE α] {f g : X → α} :
    (∀ᶠ y in 𝓝 x, f ≤ᶠ[𝓝 y] g) ↔ f ≤ᶠ[𝓝 x] g :=
  eventually_eventually_nhds


/-- If two functions are equal in a neighbourhood of `x`, then for `y` sufficiently close
to `x` these functions are equal in a neighbourhood of `y`. -/
theorem Filter.EventuallyEq.eventuallyEq_nhds {f g : X → α} (h : f =ᶠ[𝓝 x] g) :
    ∀ᶠ y in 𝓝 x, f =ᶠ[𝓝 y] g :=
  h.eventually_nhds


/-- If `f x ≤ g x` in a neighbourhood of `x`, then for `y` sufficiently close to `x` we have
`f x ≤ g x` in a neighbourhood of `y`. -/
theorem Filter.EventuallyLE.eventuallyLE_nhds [LE α] {f g : X → α} (h : f ≤ᶠ[𝓝 x] g) :
    ∀ᶠ y in 𝓝 x, f ≤ᶠ[𝓝 y] g :=
  h.eventually_nhds


theorem all_mem_nhds (x : X) (P : Set X → Prop) (hP : ∀ s t, s ⊆ t → P s → P t) :
    (∀ s ∈ 𝓝 x, P s) ↔ ∀ s, IsOpen s → x ∈ s → P s :=
                                                   /-
                                                     X : Type u
                                                     inst✝ : TopologicalSpace X
                                                     x : X
                                                     P : Set X → Prop
                                                     hP : ∀ (s t : Set X), HasSubset.Subset s t → P s → P t
                                                     ⊢ Iff (∀ (i : Set X), And (Membership.mem i x) (IsOpen i) → P i) (∀ (s : Set X …
                                                   -/
  ((nhds_basis_opens x).forall_iff hP).trans <| by simp only [@and_comm (x ∈ _), and_imp]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem all_mem_nhds_filter (x : X) (f : Set X → Set α) (hf : ∀ s t, s ⊆ t → f s ⊆ f t)
    (l : Filter α) : (∀ s ∈ 𝓝 x, f s ∈ l) ↔ ∀ s, IsOpen s → x ∈ s → f s ∈ l :=
  all_mem_nhds _ _ fun s t ssubt h => mem_of_superset h (hf s t ssubt)


theorem tendsto_nhds {f : α → X} {l : Filter α} :
    Tendsto f l (𝓝 x) ↔ ∀ s, IsOpen s → x ∈ s → f ⁻¹' s ∈ l :=
  all_mem_nhds_filter _ _ (fun _ _ h => preimage_mono h) _


theorem tendsto_atTop_nhds [Nonempty α] [SemilatticeSup α] {f : α → X} :
    Tendsto f atTop (𝓝 x) ↔ ∀ U : Set X, x ∈ U → IsOpen U → ∃ N, ∀ n, N ≤ n → f n ∈ U :=
  (atTop_basis.tendsto_iff (nhds_basis_opens x)).trans <| by
    /-
      X : Type u
      α : Type u_1
      x : X
      inst✝² : TopologicalSpace X
      inst✝¹ : Nonempty α
      inst✝ : SemilatticeSup α
      f : α → X
      ⊢ Iff (∀ (ib : Set X), And (Membership.mem ib x) (IsOpen ib) → Exists fun ia = …
    -/
    simp only [and_imp, exists_prop, true_and, mem_Ici]
    /-
      🎉 no goals
    -/


theorem tendsto_const_nhds {f : Filter α} : Tendsto (fun _ : α => x) f (𝓝 x) :=
  tendsto_nhds.mpr fun _ _ ha => univ_mem' fun _ => ha


theorem tendsto_atTop_of_eventually_const {ι : Type*} [Preorder ι]
    {u : ι → X} {i₀ : ι} (h : ∀ i ≥ i₀, u i = x) : Tendsto u atTop (𝓝 x) :=
  Tendsto.congr' (EventuallyEq.symm ((eventually_ge_atTop i₀).mono h)) tendsto_const_nhds


theorem tendsto_atBot_of_eventually_const {ι : Type*} [Preorder ι]
    {u : ι → X} {i₀ : ι} (h : ∀ i ≤ i₀, u i = x) : Tendsto u atBot (𝓝 x) :=
  tendsto_atTop_of_eventually_const (ι := ιᵒᵈ) h


theorem pure_le_nhds : pure ≤ (𝓝 : X → Filter X) := fun _ _ hs => mem_pure.2 <| mem_of_mem_nhds hs


theorem tendsto_pure_nhds (f : α → X) (a : α) : Tendsto f (pure a) (𝓝 (f a)) :=
  (tendsto_pure_pure f a).mono_right (pure_le_nhds _)


theorem OrderTop.tendsto_atTop_nhds [PartialOrder α] [OrderTop α] (f : α → X) :
    Tendsto f atTop (𝓝 (f ⊤)) :=
  (tendsto_atTop_pure f).mono_right (pure_le_nhds _)


@[simp]
instance nhds_neBot : NeBot (𝓝 x) :=
  neBot_of_le (pure_le_nhds x)


theorem tendsto_nhds_of_eventually_eq {l : Filter α} {f : α → X} (h : ∀ᶠ x' in l, f x' = x) :
    Tendsto f l (𝓝 x) :=
  tendsto_const_nhds.congr' (.symm h)


theorem Filter.EventuallyEq.tendsto {l : Filter α} {f : α → X} (hf : f =ᶠ[l] fun _ ↦ x) :
    Tendsto f l (𝓝 x) :=
  tendsto_nhds_of_eventually_eq hf


theorem ClusterPt.neBot {F : Filter X} (h : ClusterPt x F) : NeBot (𝓝 x ⊓ F) :=
  h


theorem Filter.HasBasis.clusterPt_iff {ιX ιF} {pX : ιX → Prop} {sX : ιX → Set X} {pF : ιF → Prop}
    {sF : ιF → Set X} {F : Filter X} (hX : (𝓝 x).HasBasis pX sX) (hF : F.HasBasis pF sF) :
    ClusterPt x F ↔ ∀ ⦃i⦄, pX i → ∀ ⦃j⦄, pF j → (sX i ∩ sF j).Nonempty :=
  hX.inf_basis_neBot_iff hF


theorem Filter.HasBasis.clusterPt_iff_frequently {ι} {p : ι → Prop} {s : ι → Set X} {F : Filter X}
    (hx : (𝓝 x).HasBasis p s) : ClusterPt x F ↔ ∀ i, p i → ∃ᶠ x in F, x ∈ s i := by
  simp only [hx.clusterPt_iff F.basis_sets, Filter.frequently_iff, inter_comm (s _),
    Set.Nonempty, id, mem_inter_iff]


theorem clusterPt_iff {F : Filter X} :
    ClusterPt x F ↔ ∀ ⦃U : Set X⦄, U ∈ 𝓝 x → ∀ ⦃V⦄, V ∈ F → (U ∩ V).Nonempty :=
  inf_neBot_iff


theorem clusterPt_iff_not_disjoint {F : Filter X} :
    ClusterPt x F ↔ ¬Disjoint (𝓝 x) F := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    F : Filter X
    ⊢ Iff (ClusterPt x F) (Not (Disjoint (nhds x) F))
  -/
  rw [disjoint_iff, ClusterPt, neBot_iff]
  /-
    🎉 no goals
  -/


/-- `x` is a cluster point of a set `s` if every neighbourhood of `x` meets `s` on a nonempty
set. See also `mem_closure_iff_clusterPt`. -/
theorem clusterPt_principal_iff :
    ClusterPt x (𝓟 s) ↔ ∀ U ∈ 𝓝 x, (U ∩ s).Nonempty :=
  inf_principal_neBot_iff


theorem clusterPt_principal_iff_frequently :
    ClusterPt x (𝓟 s) ↔ ∃ᶠ y in 𝓝 x, y ∈ s := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (ClusterPt x (Filter.principal s)) (Filter.Frequently (fun y => Membersh …
  -/
  simp only [clusterPt_principal_iff, frequently_iff, Set.Nonempty, exists_prop, mem_inter_iff]
  /-
    🎉 no goals
  -/


theorem ClusterPt.of_le_nhds {f : Filter X} (H : f ≤ 𝓝 x) [NeBot f] : ClusterPt x f := by
  /-
    X : Type u
    x : X
    inst✝¹ : TopologicalSpace X
    f : Filter X
    H : LE.le f (nhds x)
    inst✝ : f.NeBot
    ⊢ ClusterPt x f
  -/
  rwa [ClusterPt, inf_eq_right.mpr H]
  /-
    🎉 no goals
  -/


theorem ClusterPt.of_le_nhds' {f : Filter X} (H : f ≤ 𝓝 x) (_hf : NeBot f) :
    ClusterPt x f :=
  ClusterPt.of_le_nhds H


theorem ClusterPt.of_nhds_le {f : Filter X} (H : 𝓝 x ≤ f) : ClusterPt x f := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    f : Filter X
    H : LE.le (nhds x) f
    ⊢ ClusterPt x f
  -/
  simp only [ClusterPt, inf_eq_left.mpr H, nhds_neBot]
  /-
    🎉 no goals
  -/


theorem ClusterPt.mono {f g : Filter X} (H : ClusterPt x f) (h : f ≤ g) : ClusterPt x g :=
  NeBot.mono H <| inf_le_inf_left _ h


theorem ClusterPt.of_inf_left {f g : Filter X} (H : ClusterPt x <| f ⊓ g) : ClusterPt x f :=
  H.mono inf_le_left


theorem ClusterPt.of_inf_right {f g : Filter X} (H : ClusterPt x <| f ⊓ g) :
    ClusterPt x g :=
  H.mono inf_le_right


theorem mapClusterPt_def : MapClusterPt x F u ↔ ClusterPt x (map u F) := Iff.rfl

alias ⟨MapClusterPt.clusterPt, _⟩ := mapClusterPt_def


theorem MapClusterPt.mono {G : Filter α} (h : MapClusterPt x F u) (hle : F ≤ G) :
    MapClusterPt x G u :=
  h.clusterPt.mono (map_mono hle)


theorem MapClusterPt.tendsto_comp' [TopologicalSpace Y] {f : X → Y} {y : Y}
    (hf : Tendsto f (𝓝 x ⊓ map u F) (𝓝 y)) (hu : MapClusterPt x F u) : MapClusterPt y F (f ∘ u) :=
  (tendsto_inf.2 ⟨hf, tendsto_map.mono_left inf_le_right⟩).neBot (hx := hu)


theorem MapClusterPt.tendsto_comp [TopologicalSpace Y] {f : X → Y} {y : Y}
    (hf : Tendsto f (𝓝 x) (𝓝 y)) (hu : MapClusterPt x F u) : MapClusterPt y F (f ∘ u) :=
  hu.tendsto_comp' (hf.mono_left inf_le_left)


theorem MapClusterPt.continuousAt_comp [TopologicalSpace Y] {f : X → Y} (hf : ContinuousAt f x)
    (hu : MapClusterPt x F u) : MapClusterPt (f x) F (f ∘ u) :=
  hu.tendsto_comp hf


theorem Filter.HasBasis.mapClusterPt_iff_frequently {ι : Sort*} {p : ι → Prop} {s : ι → Set X}
    (hx : (𝓝 x).HasBasis p s) : MapClusterPt x F u ↔ ∀ i, p i → ∃ᶠ a in F, u a ∈ s i := by
  /-
    X : Type u
    α : Type u_1
    inst✝ : TopologicalSpace X
    F : Filter α
    u : α → X
    x : X
    ι : Sort u_3
    p : ι → Prop
    s : ι → Set X
    hx : (nhds x).HasBasis p s
    ⊢ Iff (MapClusterPt x F u) (∀ (i : ι), p i → Filter.Frequently (fun a => Membe …
  -/
  simp_rw [MapClusterPt, hx.clusterPt_iff_frequently, frequently_map]
  /-
    🎉 no goals
  -/


theorem mapClusterPt_iff : MapClusterPt x F u ↔ ∀ s ∈ 𝓝 x, ∃ᶠ a in F, u a ∈ s :=
  (𝓝 x).basis_sets.mapClusterPt_iff_frequently


theorem mapClusterPt_comp {φ : α → β} {u : β → X} :
    MapClusterPt x F (u ∘ φ) ↔ MapClusterPt x (map φ F) u := Iff.rfl


theorem Filter.Tendsto.mapClusterPt [NeBot F] (h : Tendsto u F (𝓝 x)) : MapClusterPt x F u :=
  .of_le_nhds h


theorem MapClusterPt.of_comp {φ : β → α} {p : Filter β} (h : Tendsto φ p F)
    (H : MapClusterPt x p (u ∘ φ)) : MapClusterPt x F u :=
  H.clusterPt.mono <| map_mono h


@[deprecated MapClusterPt.of_comp (since := "2024-09-07")]
theorem mapClusterPt_of_comp {φ : β → α} {p : Filter β} [NeBot p]
    (h : Tendsto φ p F) (H : Tendsto (u ∘ φ) p (𝓝 x)) : MapClusterPt x F u :=
  .of_comp h H.mapClusterPt


theorem accPt_sup (x : X) (F G : Filter X) :
    AccPt x (F ⊔ G) ↔ AccPt x F ∨ AccPt x G := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    F G : Filter X
    ⊢ Iff (AccPt x (Max.max F G)) (Or (AccPt x F) (AccPt x G))
  -/
  simp only [AccPt, inf_sup_left, sup_neBot]
  /-
    🎉 no goals
  -/


theorem acc_iff_cluster (x : X) (F : Filter X) : AccPt x F ↔ ClusterPt x (𝓟 {x}ᶜ ⊓ F) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    F : Filter X
    ⊢ Iff (AccPt x F) (ClusterPt x (Min.min (Filter.principal (HasCompl.compl (Sin …
  -/
  rw [AccPt, nhdsWithin, ClusterPt, inf_assoc]
  /-
    🎉 no goals
  -/


/-- `x` is an accumulation point of a set `C` iff it is a cluster point of `C ∖ {x}`. -/
theorem acc_principal_iff_cluster (x : X) (C : Set X) :
    AccPt x (𝓟 C) ↔ ClusterPt x (𝓟 (C \ {x})) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    C : Set X
    ⊢ Iff (AccPt x (Filter.principal C)) (ClusterPt x (Filter.principal (SDiff.sdi …
  -/
  rw [acc_iff_cluster, inf_principal, inter_comm, diff_eq]
  /-
    🎉 no goals
  -/


/-- `x` is an accumulation point of a set `C` iff every neighborhood
of `x` contains a point of `C` other than `x`. -/
theorem accPt_iff_nhds (x : X) (C : Set X) : AccPt x (𝓟 C) ↔ ∀ U ∈ 𝓝 x, ∃ y ∈ U ∩ C, y ≠ x := by
  simp [acc_principal_iff_cluster, clusterPt_principal_iff, Set.Nonempty, exists_prop, and_assoc,
    @and_comm (¬_ = x)]


/-- `x` is an accumulation point of a set `C` iff
there are points near `x` in `C` and different from `x`. -/
theorem accPt_iff_frequently (x : X) (C : Set X) : AccPt x (𝓟 C) ↔ ∃ᶠ y in 𝓝 x, y ≠ x ∧ y ∈ C := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    C : Set X
    ⊢ Iff (AccPt x (Filter.principal C)) (Filter.Frequently (fun y => And (Ne y x) …
  -/
  simp [acc_principal_iff_cluster, clusterPt_principal_iff_frequently, and_comm]
  /-
    🎉 no goals
  -/


/-- If `x` is an accumulation point of `F` and `F ≤ G`, then
`x` is an accumulation point of `G`. -/
theorem AccPt.mono {F G : Filter X} (h : AccPt x F) (hFG : F ≤ G) : AccPt x G :=
  NeBot.mono h (inf_le_inf_left _ hFG)


theorem AccPt.clusterPt (x : X) (F : Filter X) (h : AccPt x F) : ClusterPt x F :=
  ((acc_iff_cluster x F).mp h).mono inf_le_right


theorem clusterPt_principal {x : X} {C : Set X} :
    ClusterPt x (𝓟 C) ↔ x ∈ C ∨ AccPt x (𝓟 C) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    C : Set X
    ⊢ Iff (ClusterPt x (Filter.principal C)) (Or (Membership.mem C x) (AccPt x (Fi …
  -/
  constructor
    /-
      case mp
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      C : Set X
      ⊢ ClusterPt x (Filter.principal C) → Or (Membership.mem C x) (AccPt x (Filter. …
    -/
  · intro h
    /-
      case mp
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      C : Set X
      h : ClusterPt x (Filter.principal C)
      ⊢ Or (Membership.mem C x) (AccPt x (Filter.principal C))
    -/
    by_contra! hc
    /-
      case mp
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      C : Set X
      h : ClusterPt x (Filter.principal C)
      hc : And (Not (Membership.mem C x)) (Not (AccPt x (Filter.principal C)))
      ⊢ False
    -/
    rw [acc_principal_iff_cluster] at hc
    /-
      case mp
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      C : Set X
      h : ClusterPt x (Filter.principal C)
      hc : And (Not (Membership.mem C x)) (Not (ClusterPt x (Filter.principal (SDiff …
      ⊢ False
    -/
    simp_all only [not_false_eq_true, diff_singleton_eq_self, not_true_eq_false, hc.1]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      C : Set X
      ⊢ Or (Membership.mem C x) (AccPt x (Filter.principal C)) → ClusterPt x (Filter …
    -/
  · rintro (h | h)
      /-
        case mpr.inl
        X : Type u
        inst✝ : TopologicalSpace X
        x : X
        C : Set X
        h : Membership.mem C x
        ⊢ ClusterPt x (Filter.principal C)
      -/
    · exact clusterPt_principal_iff.mpr fun _ mem ↦ ⟨x, ⟨mem_of_mem_nhds mem, h⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        X : Type u
        inst✝ : TopologicalSpace X
        x : X
        C : Set X
        h : AccPt x (Filter.principal C)
        ⊢ ClusterPt x (Filter.principal C)
      -/
    · exact h.clusterPt
      /-
        🎉 no goals
      -/


theorem interior_eq_nhds' : interior s = { x | s ∈ 𝓝 x } :=
                      /-
                        X : Type u
                        s : Set X
                        inst✝ : TopologicalSpace X
                        x : X
                        ⊢ Iff (Membership.mem (interior s) x) (Membership.mem (setOf fun x => Membersh …
                      -/
  Set.ext fun x => by simp only [mem_interior, mem_nhds_iff, mem_setOf_eq]
                      /-
                        🎉 no goals
                      -/


theorem interior_eq_nhds : interior s = { x | 𝓝 x ≤ 𝓟 s } :=
                                /-
                                  X : Type u
                                  s : Set X
                                  inst✝ : TopologicalSpace X
                                  ⊢ Eq (setOf fun x => Membership.mem (nhds x) s) (setOf fun x => LE.le (nhds x) …
                                -/
  interior_eq_nhds'.trans <| by simp only [le_principal_iff]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem interior_mem_nhds : interior s ∈ 𝓝 x ↔ s ∈ 𝓝 x :=
  ⟨fun h => mem_of_superset h interior_subset, fun h =>
    IsOpen.mem_nhds isOpen_interior (mem_interior_iff_mem_nhds.2 h)⟩


theorem interior_setOf_eq {p : X → Prop} : interior { x | p x } = { x | ∀ᶠ y in 𝓝 x, p y } :=
  interior_eq_nhds'


theorem isOpen_setOf_eventually_nhds {p : X → Prop} : IsOpen { x | ∀ᶠ y in 𝓝 x, p y } := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    p : X → Prop
    ⊢ IsOpen (setOf fun x => Filter.Eventually (fun y => p y) (nhds x))
  -/
  simp only [← interior_setOf_eq, isOpen_interior]
  /-
    🎉 no goals
  -/


theorem subset_interior_iff_nhds {V : Set X} : s ⊆ interior V ↔ ∀ x ∈ s, V ∈ 𝓝 x := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    V : Set X
    ⊢ Iff (HasSubset.Subset s (interior V)) (∀ (x : X), Membership.mem s x → Membe …
  -/
  simp_rw [subset_def, mem_interior_iff_mem_nhds]
  /-
    🎉 no goals
  -/


theorem isOpen_iff_nhds : IsOpen s ↔ ∀ x ∈ s, 𝓝 x ≤ 𝓟 s :=
  calc
    IsOpen s ↔ s ⊆ interior s := subset_interior_iff_isOpen.symm
                                 /-
                                   X : Type u
                                   s : Set X
                                   inst✝ : TopologicalSpace X
                                   ⊢ Iff (HasSubset.Subset s (interior s)) (∀ (x : X), Membership.mem s x → LE.le …
                                 -/
    _ ↔ ∀ x ∈ s, 𝓝 x ≤ 𝓟 s := by simp_rw [interior_eq_nhds, subset_def, mem_setOf]
                                 /-
                                   🎉 no goals
                                 -/


theorem TopologicalSpace.ext_iff_nhds {X} {t t' : TopologicalSpace X} :
    t = t' ↔ ∀ x, @nhds _ t x = @nhds _ t' x :=
                                                   /-
                                                     X : Type u_3
                                                     t t' : TopologicalSpace X
                                                     H : ∀ (x : X), Eq (nhds x) (nhds x)
                                                     ⊢ Eq t t'
                                                   -/
  ⟨fun H _ ↦ congrFun (congrArg _ H) _, fun H ↦ by ext; simp_rw [@isOpen_iff_nhds _ _ _, H]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


alias ⟨_, TopologicalSpace.ext_nhds⟩ := TopologicalSpace.ext_iff_nhds


theorem isOpen_iff_mem_nhds : IsOpen s ↔ ∀ x ∈ s, s ∈ 𝓝 x :=
  isOpen_iff_nhds.trans <| forall_congr' fun _ => imp_congr_right fun _ => le_principal_iff


/-- A set `s` is open iff for every point `x` in `s` and every `y` close to `x`, `y` is in `s`. -/
theorem isOpen_iff_eventually : IsOpen s ↔ ∀ x, x ∈ s → ∀ᶠ y in 𝓝 x, y ∈ s :=
  isOpen_iff_mem_nhds


theorem isOpen_singleton_iff_nhds_eq_pure (x : X) : IsOpen ({x} : Set X) ↔ 𝓝 x = pure x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Iff (IsOpen (Singleton.singleton x)) (Eq (nhds x) (Pure.pure x))
  -/
  constructor
    /-
      case mp
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      ⊢ IsOpen (Singleton.singleton x) → Eq (nhds x) (Pure.pure x)
    -/
  · intro h
    /-
      case mp
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      h : IsOpen (Singleton.singleton x)
      ⊢ Eq (nhds x) (Pure.pure x)
    -/
    apply le_antisymm _ (pure_le_nhds x)
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      h : IsOpen (Singleton.singleton x)
      ⊢ LE.le (nhds x) (Pure.pure x)
    -/
    rw [le_pure_iff]
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      h : IsOpen (Singleton.singleton x)
      ⊢ Membership.mem (nhds x) (Singleton.singleton x)
    -/
    exact h.mem_nhds (mem_singleton x)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      ⊢ Eq (nhds x) (Pure.pure x) → IsOpen (Singleton.singleton x)
    -/
  · intro h
    /-
      case mpr
      X : Type u
      inst✝ : TopologicalSpace X
      x : X
      h : Eq (nhds x) (Pure.pure x)
      ⊢ IsOpen (Singleton.singleton x)
    -/
    simp [isOpen_iff_nhds, h]
    /-
      🎉 no goals
    -/


theorem isOpen_singleton_iff_punctured_nhds (x : X) : IsOpen ({x} : Set X) ↔ 𝓝[≠] x = ⊥ := by
  rw [isOpen_singleton_iff_nhds_eq_pure, nhdsWithin, ← mem_iff_inf_principal_compl,
      le_antisymm_iff]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Iff (And (LE.le (nhds x) (Pure.pure x)) (LE.le (Pure.pure x) (nhds x))) (Mem …
  -/
  simp [pure_le_nhds x]
  /-
    🎉 no goals
  -/


theorem mem_closure_iff_frequently : x ∈ closure s ↔ ∃ᶠ x in 𝓝 x, x ∈ s := by
  rw [Filter.Frequently, Filter.Eventually, ← mem_interior_iff_mem_nhds,
    closure_eq_compl_interior_compl, mem_compl_iff, compl_def]


alias ⟨_, Filter.Frequently.mem_closure⟩ := mem_closure_iff_frequently


/-- A set `s` is closed iff for every point `x`, if there is a point `y` close to `x` that belongs
to `s` then `x` is in `s`. -/
theorem isClosed_iff_frequently : IsClosed s ↔ ∀ x, (∃ᶠ y in 𝓝 x, y ∈ s) → x ∈ s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (IsClosed s) (∀ (x : X), Filter.Frequently (fun y => Membership.mem s y) …
  -/
  rw [← closure_subset_iff_isClosed]
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (HasSubset.Subset (closure s) s) (∀ (x : X), Filter.Frequently (fun y => …
  -/
  refine forall_congr' fun x => ?_
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Iff (Membership.mem (closure s) x → Membership.mem s x) (Filter.Frequently ( …
  -/
  rw [mem_closure_iff_frequently]
  /-
    🎉 no goals
  -/


/-- The set of cluster points of a filter is closed. In particular, the set of limit points
of a sequence is closed. -/
theorem isClosed_setOf_clusterPt {f : Filter X} : IsClosed { x | ClusterPt x f } := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ IsClosed (setOf fun x => ClusterPt x f)
  -/
  simp only [ClusterPt, inf_neBot_iff_frequently_left, setOf_forall, imp_iff_not_or]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ IsClosed (Set.iInter fun i => setOf fun x => Or (Not (Filter.Eventually (fun …
  -/
  refine isClosed_iInter fun p => IsClosed.union ?_ ?_ <;> apply isClosed_compl_iff.2
  /-
    case refine_1
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    p : X → Prop
    ⊢ IsOpen fun x => (nhds x).sets (setOf fun x => (fun x => p x) x)
  -/
  exacts [isOpen_setOf_eventually_nhds, isOpen_const]
  /-
    🎉 no goals
  -/


theorem mem_closure_iff_clusterPt : x ∈ closure s ↔ ClusterPt x (𝓟 s) :=
  mem_closure_iff_frequently.trans clusterPt_principal_iff_frequently.symm


theorem mem_closure_iff_nhds_ne_bot : x ∈ closure s ↔ 𝓝 x ⊓ 𝓟 s ≠ ⊥ :=
  mem_closure_iff_clusterPt.trans neBot_iff


theorem mem_closure_iff_nhdsWithin_neBot : x ∈ closure s ↔ NeBot (𝓝[s] x) :=
  mem_closure_iff_clusterPt


lemma nhdsWithin_neBot : (𝓝[s] x).NeBot ↔ ∀ ⦃t⦄, t ∈ 𝓝 x → (t ∩ s).Nonempty := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (nhdsWithin x s).NeBot (∀ ⦃t : Set X⦄, Membership.mem (nhds x) t → (Inte …
  -/
  rw [nhdsWithin, inf_neBot_iff]
  exact forall₂_congr fun U _ ↦
    ⟨fun h ↦ h (mem_principal_self _), fun h u hsu ↦ h.mono <| inter_subset_inter_right _ hsu⟩


@[gcongr]
theorem nhdsWithin_mono (x : X) {s t : Set X} (h : s ⊆ t) : 𝓝[s] x ≤ 𝓝[t] x :=
  inf_le_inf_left _ (principal_mono.mpr h)


lemma not_mem_closure_iff_nhdsWithin_eq_bot : x ∉ closure s ↔ 𝓝[s] x = ⊥ := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Not (Membership.mem (closure s) x)) (Eq (nhdsWithin x s) Bot.bot)
  -/
  rw [mem_closure_iff_nhdsWithin_neBot, not_neBot]
  /-
    🎉 no goals
  -/


/-- If `x` is not an isolated point of a topological space, then `{x}ᶜ` is dense in the whole
space. -/
theorem dense_compl_singleton (x : X) [NeBot (𝓝[≠] x)] : Dense ({x}ᶜ : Set X) := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    ⊢ Dense (HasCompl.compl (Singleton.singleton x))
  -/
  intro y
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    y : X
    ⊢ Membership.mem (closure (HasCompl.compl (Singleton.singleton x))) y
  -/
  rcases eq_or_ne y x with (rfl | hne)
    /-
      case inl
      X : Type u
      inst✝¹ : TopologicalSpace X
      y : X
      inst✝ : (nhdsWithin y (HasCompl.compl (Singleton.singleton y))).NeBot
      ⊢ Membership.mem (closure (HasCompl.compl (Singleton.singleton y))) y
    -/
  · rwa [mem_closure_iff_nhdsWithin_neBot]
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
      y : X
      hne : Ne y x
      ⊢ Membership.mem (closure (HasCompl.compl (Singleton.singleton x))) y
    -/
  · exact subset_closure hne
    /-
      🎉 no goals
    -/


/-- If `x` is not an isolated point of a topological space, then the closure of `{x}ᶜ` is the whole
space. -/
theorem closure_compl_singleton (x : X) [NeBot (𝓝[≠] x)] : closure {x}ᶜ = (univ : Set X) :=
  (dense_compl_singleton x).closure_eq


/-- If `x` is not an isolated point of a topological space, then the interior of `{x}` is empty. -/
@[simp]
theorem interior_singleton (x : X) [NeBot (𝓝[≠] x)] : interior {x} = (∅ : Set X) :=
  interior_eq_empty_iff_dense_compl.2 (dense_compl_singleton x)


theorem not_isOpen_singleton (x : X) [NeBot (𝓝[≠] x)] : ¬IsOpen ({x} : Set X) :=
  dense_compl_singleton_iff_not_open.1 (dense_compl_singleton x)


theorem closure_eq_cluster_pts : closure s = { a | ClusterPt a (𝓟 s) } :=
  Set.ext fun _ => mem_closure_iff_clusterPt


theorem mem_closure_iff_nhds : x ∈ closure s ↔ ∀ t ∈ 𝓝 x, (t ∩ s).Nonempty :=
  mem_closure_iff_clusterPt.trans clusterPt_principal_iff


theorem mem_closure_iff_nhds' : x ∈ closure s ↔ ∀ t ∈ 𝓝 x, ∃ y : s, ↑y ∈ t := by
  /-
    X : Type u
    x : X
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (Membership.mem (closure s) x) (∀ (t : Set X), Membership.mem (nhds x) t …
  -/
  simp only [mem_closure_iff_nhds, Set.inter_nonempty_iff_exists_right, SetCoe.exists, exists_prop]
  /-
    🎉 no goals
  -/


theorem mem_closure_iff_comap_neBot :
    x ∈ closure s ↔ NeBot (comap ((↑) : s → X) (𝓝 x)) := by
  simp_rw [mem_closure_iff_nhds, comap_neBot_iff, Set.inter_nonempty_iff_exists_right,
    SetCoe.exists, exists_prop]


theorem mem_closure_iff_nhds_basis' {p : ι → Prop} {s : ι → Set X} (h : (𝓝 x).HasBasis p s) :
    x ∈ closure t ↔ ∀ i, p i → (s i ∩ t).Nonempty :=
  mem_closure_iff_clusterPt.trans <|
                                                         /-
                                                           X : Type u
                                                           ι : Sort w
                                                           x : X
                                                           t : Set X
                                                           inst✝ : TopologicalSpace X
                                                           p : ι → Prop
                                                           s : ι → Set X
                                                           h : (nhds x).HasBasis p s
                                                           ⊢ Iff (∀ ⦃i : ι⦄, p i → ∀ ⦃j : Unit⦄, True → (Inter.inter (s i) t).Nonempty) ( …
                                                         -/
    (h.clusterPt_iff (hasBasis_principal _)).trans <| by simp only [exists_prop, forall_const]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem mem_closure_iff_nhds_basis {p : ι → Prop} {s : ι → Set X} (h : (𝓝 x).HasBasis p s) :
    x ∈ closure t ↔ ∀ i, p i → ∃ y ∈ t, y ∈ s i :=
  (mem_closure_iff_nhds_basis' h).trans <| by
    /-
      X : Type u
      ι : Sort w
      x : X
      t : Set X
      inst✝ : TopologicalSpace X
      p : ι → Prop
      s : ι → Set X
      h : (nhds x).HasBasis p s
      ⊢ Iff (∀ (i : ι), p i → (Inter.inter (s i) t).Nonempty) (∀ (i : ι), p i → Exis …
    -/
    simp only [Set.Nonempty, mem_inter_iff, exists_prop, and_comm]
    /-
      🎉 no goals
    -/


theorem clusterPt_iff_forall_mem_closure {F : Filter X} :
    ClusterPt x F ↔ ∀ s ∈ F, x ∈ closure s := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    F : Filter X
    ⊢ Iff (ClusterPt x F) (∀ (s : Set X), Membership.mem F s → Membership.mem (clo …
  -/
  simp_rw [ClusterPt, inf_neBot_iff, mem_closure_iff_nhds]
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    F : Filter X
    ⊢ Iff (∀ ⦃s : Set X⦄, Membership.mem (nhds x) s → ∀ ⦃s' : Set X⦄, Membership.m …
  -/
  rw [forall₂_swap]
  /-
    🎉 no goals
  -/


theorem clusterPt_iff_lift'_closure {F : Filter X} :
    ClusterPt x F ↔ pure x ≤ (F.lift' closure) := by
  simp_rw [clusterPt_iff_forall_mem_closure,
    (hasBasis_pure _).le_basis_iff F.basis_sets.lift'_closure, id, singleton_subset_iff, true_and,
    exists_const]


theorem clusterPt_iff_lift'_closure' {F : Filter X} :
    ClusterPt x F ↔ (F.lift' closure ⊓ pure x).NeBot := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    F : Filter X
    ⊢ Iff (ClusterPt x F) (Min.min (F.lift' closure) (Pure.pure x)).NeBot
  -/
  rw [clusterPt_iff_lift'_closure, inf_comm]
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    F : Filter X
    ⊢ Iff (LE.le (Pure.pure x) (F.lift' closure)) (Min.min (Pure.pure x) (F.lift'  …
  -/
  constructor
    /-
      case mp
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      F : Filter X
      ⊢ LE.le (Pure.pure x) (F.lift' closure) → (Min.min (Pure.pure x) (F.lift' clos …
    -/
  · intro h
    /-
      case mp
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      F : Filter X
      h : LE.le (Pure.pure x) (F.lift' closure)
      ⊢ (Min.min (Pure.pure x) (F.lift' closure)).NeBot
    -/
    simp [h, pure_neBot]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      F : Filter X
      ⊢ (Min.min (Pure.pure x) (F.lift' closure)).NeBot → LE.le (Pure.pure x) (F.lif …
    -/
  · intro h U hU
    /-
      case mpr
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      F : Filter X
      h : (Min.min (Pure.pure x) (F.lift' closure)).NeBot
      U : Set X
      hU : Membership.mem (F.lift' closure) U
      ⊢ Membership.mem (Pure.pure x) U
    -/
    simp_rw [← forall_mem_nonempty_iff_neBot, mem_inf_iff] at h
    /-
      case mpr
      X : Type u
      x : X
      inst✝ : TopologicalSpace X
      F : Filter X
      U : Set X
      hU : Membership.mem (F.lift' closure) U
      h : ∀ (s : Set X), (Exists fun t₁ => And (Membership.mem (Pure.pure x) t₁) (Ex …
      ⊢ Membership.mem (Pure.pure x) U
    -/
    simpa using h ({x} ∩ U) ⟨{x}, by simp, U, hU, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem clusterPt_lift'_closure_iff {F : Filter X} :
    ClusterPt x (F.lift' closure) ↔ ClusterPt x F := by
  /-
    X : Type u
    x : X
    inst✝ : TopologicalSpace X
    F : Filter X
    ⊢ Iff (ClusterPt x (F.lift' closure)) (ClusterPt x F)
  -/
  simp [clusterPt_iff_lift'_closure, lift'_lift'_assoc (monotone_closure X) (monotone_closure X)]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_clusterPt : IsClosed s ↔ ∀ a, ClusterPt a (𝓟 s) → a ∈ s :=
  calc
    IsClosed s ↔ closure s ⊆ s := closure_subset_iff_isClosed.symm
                                             /-
                                               X : Type u
                                               s : Set X
                                               inst✝ : TopologicalSpace X
                                               ⊢ Iff (HasSubset.Subset (closure s) s) (∀ (a : X), ClusterPt a (Filter.princip …
                                             -/
    _ ↔ ∀ a, ClusterPt a (𝓟 s) → a ∈ s := by simp only [subset_def, mem_closure_iff_clusterPt]
                                             /-
                                               🎉 no goals
                                             -/


theorem isClosed_iff_nhds :
    IsClosed s ↔ ∀ x, (∀ U ∈ 𝓝 x, (U ∩ s).Nonempty) → x ∈ s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (IsClosed s) (∀ (x : X), (∀ (U : Set X), Membership.mem (nhds x) U → (In …
  -/
  simp_rw [isClosed_iff_clusterPt, ClusterPt, inf_principal_neBot_iff]
  /-
    🎉 no goals
  -/


lemma isClosed_iff_forall_filter :
    IsClosed s ↔ ∀ x, ∀ F : Filter X, F.NeBot → F ≤ 𝓟 s → F ≤ 𝓝 x → x ∈ s := by
  /-
    X : Type u
    s : Set X
    inst✝ : TopologicalSpace X
    ⊢ Iff (IsClosed s) (∀ (x : X) (F : Filter X), F.NeBot → LE.le F (Filter.princi …
  -/
  simp_rw [isClosed_iff_clusterPt]
  exact ⟨fun hs x F F_ne FS Fx ↦ hs _ <| NeBot.mono F_ne (le_inf Fx FS),
         fun hs x hx ↦ hs x (𝓝 x ⊓ 𝓟 s) hx inf_le_right inf_le_left⟩


theorem IsClosed.interior_union_left (_ : IsClosed s) :
    interior (s ∪ t) ⊆ s ∪ interior t := fun a ⟨u, ⟨⟨hu₁, hu₂⟩, ha⟩⟩ =>
  (Classical.em (a ∈ s)).imp_right fun h =>
    mem_interior.mpr
      ⟨u ∩ sᶜ, fun _x hx => (hu₂ hx.1).resolve_left hx.2, IsOpen.inter hu₁ IsClosed.isOpen_compl,
        ⟨ha, h⟩⟩


theorem IsClosed.interior_union_right (h : IsClosed t) :
    interior (s ∪ t) ⊆ interior s ∪ t := by
  /-
    X : Type u
    s t : Set X
    inst✝ : TopologicalSpace X
    h : IsClosed t
    ⊢ HasSubset.Subset (interior (Union.union s t)) (Union.union (interior s) t)
  -/
  simpa only [union_comm _ t] using h.interior_union_left
  /-
    🎉 no goals
  -/


theorem IsOpen.inter_closure (h : IsOpen s) : s ∩ closure t ⊆ closure (s ∩ t) :=
  compl_subset_compl.mp <| by
    /-
      X : Type u
      s t : Set X
      inst✝ : TopologicalSpace X
      h : IsOpen s
      ⊢ HasSubset.Subset (HasCompl.compl (closure (Inter.inter s t))) (HasCompl.comp …
    -/
    simpa only [← interior_compl, compl_inter] using IsClosed.interior_union_left h.isClosed_compl
    /-
      🎉 no goals
    -/


theorem IsOpen.closure_inter (h : IsOpen t) : closure s ∩ t ⊆ closure (s ∩ t) := by
  /-
    X : Type u
    s t : Set X
    inst✝ : TopologicalSpace X
    h : IsOpen t
    ⊢ HasSubset.Subset (Inter.inter (closure s) t) (closure (Inter.inter s t))
  -/
  simpa only [inter_comm t] using h.inter_closure
  /-
    🎉 no goals
  -/


theorem Dense.open_subset_closure_inter (hs : Dense s) (ht : IsOpen t) :
    t ⊆ closure (t ∩ s) :=
  calc
                            /-
                              X : Type u
                              s t : Set X
                              inst✝ : TopologicalSpace X
                              hs : Dense s
                              ht : IsOpen t
                              ⊢ Eq t (Inter.inter t (closure s))
                            -/
    t = t ∩ closure s := by rw [hs.closure_eq, inter_univ]
                            /-
                              🎉 no goals
                            -/
    _ ⊆ closure (t ∩ s) := ht.inter_closure


theorem mem_closure_of_mem_closure_union (h : x ∈ closure (s₁ ∪ s₂))
    (h₁ : s₁ᶜ ∈ 𝓝 x) : x ∈ closure s₂ := by
  /-
    X : Type u
    x : X
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    h : Membership.mem (closure (Union.union s₁ s₂)) x
    h₁ : Membership.mem (nhds x) (HasCompl.compl s₁)
    ⊢ Membership.mem (closure s₂) x
  -/
  rw [mem_closure_iff_nhds_ne_bot] at *
  /-
    X : Type u
    x : X
    s₁ s₂ : Set X
    inst✝ : TopologicalSpace X
    h : Ne (Min.min (nhds x) (Filter.principal (Union.union s₁ s₂))) Bot.bot
    h₁ : Membership.mem (nhds x) (HasCompl.compl s₁)
    ⊢ Ne (Min.min (nhds x) (Filter.principal s₂)) Bot.bot
  -/
  rwa [← sup_principal, inf_sup_left, inf_principal_eq_bot.mpr h₁, bot_sup_eq] at h
  /-
    🎉 no goals
  -/


/-- The intersection of an open dense set with a dense set is a dense set. -/
theorem Dense.inter_of_isOpen_left (hs : Dense s) (ht : Dense t) (hso : IsOpen s) :
    Dense (s ∩ t) := fun x =>
                                                           /-
                                                             X : Type u
                                                             s t : Set X
                                                             inst✝ : TopologicalSpace X
                                                             hs : Dense s
                                                             ht : Dense t
                                                             hso : IsOpen s
                                                             x : X
                                                             ⊢ Membership.mem (closure (Inter.inter s (closure t))) x
                                                           -/
  closure_minimal hso.inter_closure isClosed_closure <| by simp [hs.closure_eq, ht.closure_eq]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- The intersection of a dense set with an open dense set is a dense set. -/
theorem Dense.inter_of_isOpen_right (hs : Dense s) (ht : Dense t) (hto : IsOpen t) :
    Dense (s ∩ t) :=
  inter_comm t s ▸ ht.inter_of_isOpen_left hs hto


theorem Dense.inter_nhds_nonempty (hs : Dense s) (ht : t ∈ 𝓝 x) :
    (s ∩ t).Nonempty :=
  let ⟨U, hsub, ho, hx⟩ := mem_nhds_iff.1 ht
  (hs.inter_open_nonempty U ho ⟨x, hx⟩).mono fun _y hy => ⟨hy.2, hsub hy.1⟩


theorem closure_diff : closure s \ closure t ⊆ closure (s \ t) :=
  calc
                                                           /-
                                                             X : Type u
                                                             s t : Set X
                                                             inst✝ : TopologicalSpace X
                                                             ⊢ Eq (SDiff.sdiff (closure s) (closure t)) (Inter.inter (HasCompl.compl (closu …
                                                           -/
    closure s \ closure t = (closure t)ᶜ ∩ closure s := by simp only [diff_eq, inter_comm]
                                                           /-
                                                             🎉 no goals
                                                           -/
    _ ⊆ closure ((closure t)ᶜ ∩ s) := (isOpen_compl_iff.mpr <| isClosed_closure).inter_closure
                                      /-
                                        X : Type u
                                        s t : Set X
                                        inst✝ : TopologicalSpace X
                                        ⊢ Eq (closure (Inter.inter (HasCompl.compl (closure t)) s)) (closure (SDiff.sd …
                                      -/
    _ = closure (s \ closure t) := by simp only [diff_eq, inter_comm]
                                      /-
                                        🎉 no goals
                                      -/
    _ ⊆ closure (s \ t) := closure_mono <| diff_subset_diff (Subset.refl s) subset_closure


theorem Filter.Frequently.mem_of_closed (h : ∃ᶠ x in 𝓝 x, x ∈ s)
    (hs : IsClosed s) : x ∈ s :=
  hs.closure_subset h.mem_closure


theorem IsClosed.mem_of_frequently_of_tendsto {f : α → X} {b : Filter α}
    (hs : IsClosed s) (h : ∃ᶠ x in b, f x ∈ s) (hf : Tendsto f b (𝓝 x)) : x ∈ s :=
  (hf.frequently <| show ∃ᶠ x in b, (fun y => y ∈ s) (f x) from h).mem_of_closed hs


theorem IsClosed.mem_of_tendsto {f : α → X} {b : Filter α} [NeBot b]
    (hs : IsClosed s) (hf : Tendsto f b (𝓝 x)) (h : ∀ᶠ x in b, f x ∈ s) : x ∈ s :=
  hs.mem_of_frequently_of_tendsto h.frequently hf


theorem mem_closure_of_frequently_of_tendsto {f : α → X} {b : Filter α}
    (h : ∃ᶠ x in b, f x ∈ s) (hf : Tendsto f b (𝓝 x)) : x ∈ closure s :=
  (hf.frequently h).mem_closure


theorem mem_closure_of_tendsto {f : α → X} {b : Filter α} [NeBot b]
    (hf : Tendsto f b (𝓝 x)) (h : ∀ᶠ x in b, f x ∈ s) : x ∈ closure s :=
  mem_closure_of_frequently_of_tendsto h.frequently hf


/-- Suppose that `f` sends the complement to `s` to a single point `x`, and `l` is some filter.
Then `f` tends to `x` along `l` restricted to `s` if and only if it tends to `x` along `l`. -/
theorem tendsto_inf_principal_nhds_iff_of_forall_eq {f : α → X} {l : Filter α} {s : Set α}
    (h : ∀ a ∉ s, f a = x) : Tendsto f (l ⊓ 𝓟 s) (𝓝 x) ↔ Tendsto f l (𝓝 x) := by
  /-
    X : Type u
    α : Type u_1
    x : X
    inst✝ : TopologicalSpace X
    f : α → X
    l : Filter α
    s : Set α
    h : ∀ (a : α), Not (Membership.mem s a) → Eq (f a) x
    ⊢ Iff (Filter.Tendsto f (Min.min l (Filter.principal s)) (nhds x)) (Filter.Ten …
  -/
  rw [tendsto_iff_comap, tendsto_iff_comap]
  replace h : 𝓟 sᶜ ≤ comap f (𝓝 x) := by
    rintro U ⟨t, ht, htU⟩ x hx
    have : f x ∈ t := (h x hx).symm ▸ mem_of_mem_nhds ht
    exact htU this
  /-
    X : Type u
    α : Type u_1
    x : X
    inst✝ : TopologicalSpace X
    f : α → X
    l : Filter α
    s : Set α
    h : LE.le (Filter.principal (HasCompl.compl s)) (Filter.comap f (nhds x))
    ⊢ Iff (LE.le (Min.min l (Filter.principal s)) (Filter.comap f (nhds x))) (LE.l …
  -/
  refine ⟨fun h' => ?_, le_trans inf_le_left⟩
  /-
    X : Type u
    α : Type u_1
    x : X
    inst✝ : TopologicalSpace X
    f : α → X
    l : Filter α
    s : Set α
    h : LE.le (Filter.principal (HasCompl.compl s)) (Filter.comap f (nhds x))
    h' : LE.le (Min.min l (Filter.principal s)) (Filter.comap f (nhds x))
    ⊢ LE.le l (Filter.comap f (nhds x))
  -/
  have := sup_le h' h
  rw [sup_inf_right, sup_principal, union_compl_self, principal_univ, inf_top_eq, sup_le_iff]
    at this
  /-
    X : Type u
    α : Type u_1
    x : X
    inst✝ : TopologicalSpace X
    f : α → X
    l : Filter α
    s : Set α
    h : LE.le (Filter.principal (HasCompl.compl s)) (Filter.comap f (nhds x))
    h' : LE.le (Min.min l (Filter.principal s)) (Filter.comap f (nhds x))
    this : And (LE.le l (Filter.comap f (nhds x))) (LE.le (Filter.principal (HasCo …
    ⊢ LE.le l (Filter.comap f (nhds x))
  -/
  exact this.1
  /-
    🎉 no goals
  -/


/-- If a filter `f` is majorated by some `𝓝 x`, then it is majorated by `𝓝 (Filter.lim f)`. We
formulate this lemma with a `[Nonempty X]` argument of `lim` derived from `h` to make it useful for
types without a `[Nonempty X]` instance. Because of the built-in proof irrelevance, Lean will unify
this instance with any other instance. -/
theorem le_nhds_lim {f : Filter X} (h : ∃ x, f ≤ 𝓝 x) : f ≤ 𝓝 (@lim _ _ (nonempty_of_exists h) f) :=
  Classical.epsilon_spec h


/-- If `g` tends to some `𝓝 x` along `f`, then it tends to `𝓝 (Filter.limUnder f g)`. We formulate
this lemma with a `[Nonempty X]` argument of `lim` derived from `h` to make it useful for types
without a `[Nonempty X]` instance. Because of the built-in proof irrelevance, Lean will unify this
instance with any other instance. -/
theorem tendsto_nhds_limUnder {f : Filter α} {g : α → X} (h : ∃ x, Tendsto g f (𝓝 x)) :
    Tendsto g f (𝓝 (@limUnder _ _ _ (nonempty_of_exists h) f g)) :=
  le_nhds_lim h


theorem continuous_def {_ : TopologicalSpace X} {_ : TopologicalSpace Y} {f : X → Y} :
    Continuous f ↔ ∀ s, IsOpen s → IsOpen (f ⁻¹' s) :=
  ⟨fun hf => hf.1, fun h => ⟨h⟩⟩


theorem IsOpen.preimage (hf : Continuous f) {t : Set Y} (h : IsOpen t) :
    IsOpen (f ⁻¹' t) :=
  hf.isOpen_preimage t h


lemma Equiv.continuous_symm_iff (e : X ≃ Y) : Continuous e.symm ↔ IsOpenMap e := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    e : Equiv X Y
    ⊢ Iff (Continuous ⇑e.symm) (IsOpenMap ⇑e)
  -/
  simp_rw [continuous_def, ← Set.image_equiv_eq_preimage_symm, IsOpenMap]
  /-
    🎉 no goals
  -/


lemma Equiv.isOpenMap_symm_iff (e : X ≃ Y) : IsOpenMap e.symm ↔ Continuous e := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    e : Equiv X Y
    ⊢ Iff (IsOpenMap ⇑e.symm) (Continuous ⇑e)
  -/
  simp_rw [← Equiv.continuous_symm_iff, Equiv.symm_symm]
  /-
    🎉 no goals
  -/


theorem continuous_congr {g : X → Y} (h : ∀ x, f x = g x) :
    Continuous f ↔ Continuous g :=
  .of_eq <| congrArg _ <| funext h


theorem Continuous.congr {g : X → Y} (h : Continuous f) (h' : ∀ x, f x = g x) : Continuous g :=
  continuous_congr h' |>.mp h


theorem ContinuousAt.tendsto (h : ContinuousAt f x) :
    Tendsto f (𝓝 x) (𝓝 (f x)) :=
  h


theorem continuousAt_def : ContinuousAt f x ↔ ∀ A ∈ 𝓝 (f x), f ⁻¹' A ∈ 𝓝 x :=
  Iff.rfl


theorem continuousAt_congr {g : X → Y} (h : f =ᶠ[𝓝 x] g) :
    ContinuousAt f x ↔ ContinuousAt g x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    x : X
    g : X → Y
    h : (nhds x).EventuallyEq f g
    ⊢ Iff (ContinuousAt f x) (ContinuousAt g x)
  -/
  simp only [ContinuousAt, tendsto_congr' h, h.eq_of_nhds]
  /-
    🎉 no goals
  -/


theorem ContinuousAt.congr {g : X → Y} (hf : ContinuousAt f x) (h : f =ᶠ[𝓝 x] g) :
    ContinuousAt g x :=
  (continuousAt_congr h).1 hf


theorem ContinuousAt.preimage_mem_nhds {t : Set Y} (h : ContinuousAt f x)
    (ht : t ∈ 𝓝 (f x)) : f ⁻¹' t ∈ 𝓝 x :=
  h ht


/-- If `f x ∈ s ∈ 𝓝 (f x)` for continuous `f`, then `f y ∈ s` near `x`.

This is essentially `Filter.Tendsto.eventually_mem`, but infers in more cases when applied. -/
theorem ContinuousAt.eventually_mem {f : X → Y} {x : X} (hf : ContinuousAt f x) {s : Set Y}
    (hs : s ∈ 𝓝 (f x)) : ∀ᶠ y in 𝓝 x, f y ∈ s :=
  hf hs


/-- If a function `f` tends to somewhere other than `𝓝 (f x)` at `x`,
then `f` is not continuous at `x`
-/
lemma not_continuousAt_of_tendsto {f : X → Y} {l₁ : Filter X} {l₂ : Filter Y} {x : X}
    (hf : Tendsto f l₁ l₂) [l₁.NeBot] (hl₁ : l₁ ≤ 𝓝 x) (hl₂ : Disjoint (𝓝 (f x)) l₂) :
    ¬ ContinuousAt f x := fun cont ↦
  (cont.mono_left hl₁).not_tendsto hl₂ hf


theorem ClusterPt.map {lx : Filter X} {ly : Filter Y} (H : ClusterPt x lx)
    (hfc : ContinuousAt f x) (hf : Tendsto f lx ly) : ClusterPt (f x) ly :=
  (NeBot.map H f).mono <| hfc.tendsto.inf hf


/-- See also `interior_preimage_subset_preimage_interior`. -/
theorem preimage_interior_subset_interior_preimage {t : Set Y} (hf : Continuous f) :
    f ⁻¹' interior t ⊆ interior (f ⁻¹' t) :=
  interior_maximal (preimage_mono interior_subset) (isOpen_interior.preimage hf)


@[continuity]
theorem continuous_id : Continuous (id : X → X) :=
  continuous_def.2 fun _ => id

-- This is needed due to reducibility issues with the `continuity` tactic.

@[continuity, fun_prop]
theorem continuous_id' : Continuous (fun (x : X) => x) := continuous_id


theorem Continuous.comp {g : Y → Z} (hg : Continuous g) (hf : Continuous f) :
    Continuous (g ∘ f) :=
  continuous_def.2 fun _ h => (h.preimage hg).preimage hf

-- This is needed due to reducibility issues with the `continuity` tactic.

@[continuity, fun_prop]
theorem Continuous.comp' {g : Y → Z} (hg : Continuous g) (hf : Continuous f) :
    Continuous (fun x => g (f x)) := hg.comp hf


theorem Continuous.iterate {f : X → X} (h : Continuous f) (n : ℕ) : Continuous f^[n] :=
  Nat.recOn n continuous_id fun _ ihn => ihn.comp h


nonrec theorem ContinuousAt.comp {g : Y → Z} (hg : ContinuousAt g (f x))
    (hf : ContinuousAt f x) : ContinuousAt (g ∘ f) x :=
  hg.comp hf


@[fun_prop]
theorem ContinuousAt.comp' {g : Y → Z} {x : X} (hg : ContinuousAt g (f x))
    (hf : ContinuousAt f x) : ContinuousAt (fun x => g (f x)) x := ContinuousAt.comp hg hf


/-- See note [comp_of_eq lemmas] -/
theorem ContinuousAt.comp_of_eq {g : Y → Z} (hg : ContinuousAt g y)
                                                                          /-
                                                                            X : Type u_1
                                                                            Y : Type u_2
                                                                            Z : Type u_3
                                                                            inst✝² : TopologicalSpace X
                                                                            inst✝¹ : TopologicalSpace Y
                                                                            inst✝ : TopologicalSpace Z
                                                                            f : X → Y
                                                                            x : X
                                                                            y : Y
                                                                            g : Y → Z
                                                                            hg : ContinuousAt g y
                                                                            hf : ContinuousAt f x
                                                                            hy : Eq (f x) y
                                                                            ⊢ ContinuousAt (Function.comp g f) x
                                                                          -/
    (hf : ContinuousAt f x) (hy : f x = y) : ContinuousAt (g ∘ f) x := by subst hy; exact hg.comp hf
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem Continuous.tendsto (hf : Continuous f) (x) : Tendsto f (𝓝 x) (𝓝 (f x)) :=
  ((nhds_basis_opens x).tendsto_iff <| nhds_basis_opens <| f x).2 fun t ⟨hxt, ht⟩ =>
    ⟨f ⁻¹' t, ⟨hxt, ht.preimage hf⟩, Subset.rfl⟩


/-- A version of `Continuous.tendsto` that allows one to specify a simpler form of the limit.
E.g., one can write `continuous_exp.tendsto' 0 1 exp_zero`. -/
theorem Continuous.tendsto' (hf : Continuous f) (x : X) (y : Y) (h : f x = y) :
    Tendsto f (𝓝 x) (𝓝 y) :=
  h ▸ hf.tendsto x


@[fun_prop]
theorem Continuous.continuousAt (h : Continuous f) : ContinuousAt f x :=
  h.tendsto x


theorem continuous_iff_continuousAt : Continuous f ↔ ∀ x, ContinuousAt f x :=
  ⟨Continuous.tendsto, fun hf => continuous_def.2 fun _U hU => isOpen_iff_mem_nhds.2 fun x hx =>
    hf x <| hU.mem_nhds hx⟩


@[fun_prop]
theorem continuousAt_const : ContinuousAt (fun _ : X => y) x :=
  tendsto_const_nhds


@[continuity, fun_prop]
theorem continuous_const : Continuous fun _ : X => y :=
  continuous_iff_continuousAt.mpr fun _ => continuousAt_const


theorem Filter.EventuallyEq.continuousAt (h : f =ᶠ[𝓝 x] fun _ => y) :
    ContinuousAt f x :=
  (continuousAt_congr h).2 tendsto_const_nhds


theorem continuous_of_const (h : ∀ x y, f x = f y) : Continuous f :=
  continuous_iff_continuousAt.mpr fun x =>
    Filter.EventuallyEq.continuousAt <| Eventually.of_forall fun y => h y x


theorem continuousAt_id : ContinuousAt id x :=
  continuous_id.continuousAt


@[fun_prop]
theorem continuousAt_id' (y) : ContinuousAt (fun x : X => x) y := continuousAt_id


theorem ContinuousAt.iterate {f : X → X} (hf : ContinuousAt f x) (hx : f x = x) (n : ℕ) :
    ContinuousAt f^[n] x :=
  Nat.recOn n continuousAt_id fun _n ihn ↦ ihn.comp_of_eq hf hx


theorem continuous_iff_isClosed : Continuous f ↔ ∀ s, IsClosed s → IsClosed (f ⁻¹' s) :=
  continuous_def.trans <| compl_surjective.forall.trans <| by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      ⊢ Iff (∀ (x : Set Y), IsOpen (HasCompl.compl x) → IsOpen (Set.preimage f (HasC …
    -/
    simp only [isOpen_compl_iff, preimage_compl]
    /-
      🎉 no goals
    -/


theorem IsClosed.preimage (hf : Continuous f) {t : Set Y} (h : IsClosed t) :
    IsClosed (f ⁻¹' t) :=
  continuous_iff_isClosed.mp hf t h


theorem mem_closure_image (hf : ContinuousAt f x)
    (hx : x ∈ closure s) : f x ∈ closure (f '' s) :=
  mem_closure_of_frequently_of_tendsto
    ((mem_closure_iff_frequently.1 hx).mono fun _ => mem_image_of_mem _) hf


theorem Continuous.closure_preimage_subset (hf : Continuous f) (t : Set Y) :
    closure (f ⁻¹' t) ⊆ f ⁻¹' closure t := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    t : Set Y
    ⊢ HasSubset.Subset (closure (Set.preimage f t)) (Set.preimage f (closure t))
  -/
  rw [← (isClosed_closure.preimage hf).closure_eq]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    t : Set Y
    ⊢ HasSubset.Subset (closure (Set.preimage f t)) (closure (Set.preimage f (clos …
  -/
  exact closure_mono (preimage_mono subset_closure)
  /-
    🎉 no goals
  -/


theorem Continuous.frontier_preimage_subset (hf : Continuous f) (t : Set Y) :
    frontier (f ⁻¹' t) ⊆ f ⁻¹' frontier t :=
  diff_subset_diff (hf.closure_preimage_subset t) (preimage_interior_subset_interior_preimage hf)


/-- If a continuous map `f` maps `s` to `t`, then it maps `closure s` to `closure t`. -/
protected theorem Set.MapsTo.closure {t : Set Y} (h : MapsTo f s t)
    (hc : Continuous f) : MapsTo f (closure s) (closure t) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    s : Set X
    t : Set Y
    h : Set.MapsTo f s t
    hc : Continuous f
    ⊢ Set.MapsTo f (closure s) (closure t)
  -/
  simp only [MapsTo, mem_closure_iff_clusterPt]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    s : Set X
    t : Set Y
    h : Set.MapsTo f s t
    hc : Continuous f
    ⊢ ∀ ⦃x : X⦄, ClusterPt x (Filter.principal s) → ClusterPt (f x) (Filter.princi …
  -/
  exact fun x hx => hx.map hc.continuousAt (tendsto_principal_principal.2 h)
  /-
    🎉 no goals
  -/


/-- See also `IsClosedMap.closure_image_eq_of_continuous`. -/
theorem image_closure_subset_closure_image (h : Continuous f) :
    f '' closure s ⊆ closure (f '' s) :=
  ((mapsTo_image f s).closure h).image_subset


theorem closure_image_closure (h : Continuous f) :
    closure (f '' closure s) = closure (f '' s) :=
  Subset.antisymm
    (closure_minimal (image_closure_subset_closure_image h) isClosed_closure)
    (closure_mono <| image_subset _ subset_closure)


theorem closure_subset_preimage_closure_image (h : Continuous f) :
    closure s ⊆ f ⁻¹' closure (f '' s) :=
  (mapsTo_image _ _).closure h


theorem map_mem_closure {t : Set Y} (hf : Continuous f)
    (hx : x ∈ closure s) (ht : MapsTo f s t) : f x ∈ closure t :=
  ht.closure hf hx


/-- If a continuous map `f` maps `s` to a closed set `t`, then it maps `closure s` to `t`. -/
theorem Set.MapsTo.closure_left {t : Set Y} (h : MapsTo f s t)
    (hc : Continuous f) (ht : IsClosed t) : MapsTo f (closure s) t :=
  ht.closure_eq ▸ h.closure hc


theorem Filter.Tendsto.lift'_closure (hf : Continuous f) {l l'} (h : Tendsto f l l') :
    Tendsto f (l.lift' closure) (l'.lift' closure) :=
  tendsto_lift'.2 fun s hs ↦ by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : Continuous f
      l : Filter X
      l' : Filter Y
      h : Filter.Tendsto f l l'
      s : Set Y
      hs : Membership.mem l' s
      ⊢ Filter.Eventually (fun a => Membership.mem (closure s) (f a)) (l.lift' closu …
    -/
    filter_upwards [mem_lift' (h hs)] using (mapsTo_preimage _ _).closure hf
    /-
      🎉 no goals
    -/


theorem tendsto_lift'_closure_nhds (hf : Continuous f) (x : X) :
    Tendsto f ((𝓝 x).lift' closure) ((𝓝 (f x)).lift' closure) :=
  (hf.tendsto x).lift'_closure hf


/-- A surjective map has dense range. -/
theorem Function.Surjective.denseRange (hf : Function.Surjective f) : DenseRange f := fun x => by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    α : Type u_4
    f : α → X
    hf : Function.Surjective f
    x : X
    ⊢ Membership.mem (closure (Set.range f)) x
  -/
  simp [hf.range_eq]
  /-
    🎉 no goals
  -/


theorem denseRange_id : DenseRange (id : X → X) :=
  Function.Surjective.denseRange Function.surjective_id


theorem denseRange_iff_closure_range : DenseRange f ↔ closure (range f) = univ :=
  dense_iff_closure_eq


theorem DenseRange.closure_range (h : DenseRange f) : closure (range f) = univ :=
  h.closure_eq


@[simp]
lemma denseRange_subtype_val {p : X → Prop} : DenseRange (@Subtype.val _ p) ↔ Dense {x | p x} := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    p : X → Prop
    ⊢ Iff (DenseRange Subtype.val) (Dense (setOf fun x => p x))
  -/
  simp [DenseRange]
  /-
    🎉 no goals
  -/


theorem Dense.denseRange_val (h : Dense s) : DenseRange ((↑) : s → X) :=
  denseRange_subtype_val.2 h


theorem Continuous.range_subset_closure_image_dense {f : X → Y} (hf : Continuous f)
    (hs : Dense s) : range f ⊆ closure (f '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Continuous f
    hs : Dense s
    ⊢ HasSubset.Subset (Set.range f) (closure (Set.image f s))
  -/
  rw [← image_univ, ← hs.closure_eq]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Continuous f
    hs : Dense s
    ⊢ HasSubset.Subset (Set.image f (closure s)) (closure (Set.image f s))
  -/
  exact image_closure_subset_closure_image hf
  /-
    🎉 no goals
  -/


/-- The image of a dense set under a continuous map with dense range is a dense set. -/
theorem DenseRange.dense_image {f : X → Y} (hf' : DenseRange f) (hf : Continuous f)
    (hs : Dense s) : Dense (f '' s) :=
  (hf'.mono <| hf.range_subset_closure_image_dense hs).of_closure


/-- If `f` has dense range and `s` is an open set in the codomain of `f`, then the image of the
preimage of `s` under `f` is dense in `s`. -/
theorem DenseRange.subset_closure_image_preimage_of_isOpen (hf : DenseRange f) (hs : IsOpen s) :
    s ⊆ closure (f '' (f ⁻¹' s)) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    α : Type u_4
    f : α → X
    s : Set X
    hf : DenseRange f
    hs : IsOpen s
    ⊢ HasSubset.Subset s (closure (Set.image f (Set.preimage f s)))
  -/
  rw [image_preimage_eq_inter_range]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    α : Type u_4
    f : α → X
    s : Set X
    hf : DenseRange f
    hs : IsOpen s
    ⊢ HasSubset.Subset s (closure (Inter.inter s (Set.range f)))
  -/
  exact hf.open_subset_closure_inter hs
  /-
    🎉 no goals
  -/


/-- If a continuous map with dense range maps a dense set to a subset of `t`, then `t` is a dense
set. -/
theorem DenseRange.dense_of_mapsTo {f : X → Y} (hf' : DenseRange f) (hf : Continuous f)
    (hs : Dense s) {t : Set Y} (ht : MapsTo f s t) : Dense t :=
  (hf'.dense_image hf hs).mono ht.image_subset


/-- Composition of a continuous map with dense range and a function with dense range has dense
range. -/
theorem DenseRange.comp {g : Y → Z} {f : α → Y} (hg : DenseRange g) (hf : DenseRange f)
    (cg : Continuous g) : DenseRange (g ∘ f) := by
  /-
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    α : Type u_4
    g : Y → Z
    f : α → Y
    hg : DenseRange g
    hf : DenseRange f
    cg : Continuous g
    ⊢ DenseRange (Function.comp g f)
  -/
  rw [DenseRange, range_comp]
  /-
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    α : Type u_4
    g : Y → Z
    f : α → Y
    hg : DenseRange g
    hf : DenseRange f
    cg : Continuous g
    ⊢ Dense (Set.image g (Set.range f))
  -/
  exact hg.dense_image cg hf
  /-
    🎉 no goals
  -/


nonrec theorem DenseRange.nonempty_iff (hf : DenseRange f) : Nonempty α ↔ Nonempty X :=
  range_nonempty_iff_nonempty.symm.trans hf.nonempty_iff


theorem DenseRange.nonempty [h : Nonempty X] (hf : DenseRange f) : Nonempty α :=
  hf.nonempty_iff.mpr h


/-- Given a function `f : X → Y` with dense range and `y : Y`, returns some `x : X`. -/
def DenseRange.some (hf : DenseRange f) (x : X) : α :=
  Classical.choice <| hf.nonempty_iff.mpr ⟨x⟩


nonrec theorem DenseRange.exists_mem_open (hf : DenseRange f) (ho : IsOpen s) (hs : s.Nonempty) :
    ∃ a, f a ∈ s :=
  exists_range_iff.1 <| hf.exists_mem_open ho hs


theorem DenseRange.mem_nhds (h : DenseRange f) (hs : s ∈ 𝓝 x) :
    ∃ a, f a ∈ s :=
  let ⟨a, ha⟩ := h.exists_mem_open isOpen_interior ⟨x, mem_interior_iff_mem_nhds.2 hs⟩
  ⟨a, interior_subset ha⟩


