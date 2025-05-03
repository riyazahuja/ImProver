/-- A set `s` is said to be inaccessible by directed joins on `D` if, when the least upper bound of
a directed set `d` in `D` lies in `s` then `d` has non-empty intersection with `s`. -/
def DirSupInaccOn (D : Set (Set α)) (s : Set α) : Prop :=
  ∀ ⦃d⦄, d ∈ D → d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a⦄, IsLUB d a → a ∈ s → (d ∩ s).Nonempty


/-- A set `s` is said to be inaccessible by directed joins if, when the least upper bound of a
directed set `d` lies in `s` then `d` has non-empty intersection with `s`. -/
def DirSupInacc (s : Set α) : Prop :=
  ∀ ⦃d⦄, d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a⦄, IsLUB d a → a ∈ s → (d ∩ s).Nonempty


@[simp] lemma dirSupInaccOn_univ : DirSupInaccOn univ s ↔ DirSupInacc s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (DirSupInaccOn Set.univ s) (DirSupInacc s)
  -/
  simp [DirSupInaccOn, DirSupInacc]
  /-
    🎉 no goals
  -/


@[simp] lemma DirSupInacc.dirSupInaccOn {D : Set (Set α)} :
    DirSupInacc s → DirSupInaccOn D s := fun h _ _ d₂ d₃ _ hda => h d₂ d₃ hda


lemma DirSupInaccOn.mono {D₁ D₂ : Set (Set α)} (hD : D₁ ⊆ D₂) (hf : DirSupInaccOn D₂ s) :
    DirSupInaccOn D₁ s := fun ⦃_⦄ a ↦ hf (hD a)


/--
A set `s` is said to be closed under directed joins if, whenever a directed set `d` has a least
upper bound `a` and is a subset of `s` then `a` also lies in `s`.
-/
def DirSupClosed (s : Set α) : Prop :=
  ∀ ⦃d⦄, d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a⦄, IsLUB d a → d ⊆ s → a ∈ s


@[simp] lemma dirSupInacc_compl : DirSupInacc sᶜ ↔ DirSupClosed s := by
  simp [DirSupInacc, DirSupClosed, ← not_disjoint_iff_nonempty_inter, not_imp_not,
    disjoint_compl_right_iff]


@[simp] lemma dirSupClosed_compl : DirSupClosed sᶜ ↔ DirSupInacc s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (DirSupClosed (HasCompl.compl s)) (DirSupInacc s)
  -/
  rw [← dirSupInacc_compl, compl_compl]
  /-
    🎉 no goals
  -/


alias ⟨DirSupInacc.of_compl, DirSupClosed.compl⟩ := dirSupInacc_compl

alias ⟨DirSupClosed.of_compl, DirSupInacc.compl⟩ := dirSupClosed_compl


lemma DirSupClosed.inter (hs : DirSupClosed s) (ht : DirSupClosed t) : DirSupClosed (s ∩ t) :=
  fun _d hd hd' _a ha hds ↦ ⟨hs hd hd' ha <| hds.trans inter_subset_left,
    ht hd hd' ha <| hds.trans inter_subset_right⟩


lemma DirSupInacc.union (hs : DirSupInacc s) (ht : DirSupInacc t) : DirSupInacc (s ∪ t) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s t : Set α
    hs : DirSupInacc s
    ht : DirSupInacc t
    ⊢ DirSupInacc (Union.union s t)
  -/
  rw [← dirSupClosed_compl, compl_union]; exact hs.compl.inter ht.compl
                                          /-
                                            🎉 no goals
                                          -/


lemma IsUpperSet.dirSupClosed (hs : IsUpperSet s) : DirSupClosed s :=
  fun _d ⟨_b, hb⟩ _ _a ha hds ↦ hs (ha.1 hb) <| hds hb


lemma IsLowerSet.dirSupInacc (hs : IsLowerSet s) : DirSupInacc s := hs.compl.dirSupClosed.of_compl


lemma dirSupClosed_Iic (a : α) : DirSupClosed (Iic a) := fun _d _ _ _a ha ↦ (isLUB_le_iff ha).2


lemma dirSupInacc_iff_forall_sSup :
    DirSupInacc s ↔ ∀ ⦃d⦄, d.Nonempty → DirectedOn (· ≤ ·) d → sSup d ∈ s → (d ∩ s).Nonempty := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Iff (DirSupInacc s) (∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (fun x1 x2 => LE …
  -/
  simp [DirSupInacc, isLUB_iff_sSup_eq]
  /-
    🎉 no goals
  -/


lemma dirSupClosed_iff_forall_sSup :
    DirSupClosed s ↔ ∀ ⦃d⦄, d.Nonempty → DirectedOn (· ≤ ·) d → d ⊆ s → sSup d ∈ s := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Iff (DirSupClosed s) (∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (fun x1 x2 => L …
  -/
  simp [DirSupClosed, isLUB_iff_sSup_eq]
  /-
    🎉 no goals
  -/


/-- The Scott-Hausdorff topology.

A set `u` is open in the Scott-Hausdorff topology iff when the least upper bound of a directed set
`d` lies in `u` then there is a tail of `d` which is a subset of `u`. -/
def scottHausdorff (α : Type*) (D : Set (Set α)) [Preorder α] : TopologicalSpace α where
  IsOpen u := ∀ ⦃d : Set α⦄, d ∈ D → d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a : α⦄, IsLUB d a →
    a ∈ u → ∃ b ∈ d, Ici b ∩ d ⊆ u
  isOpen_univ := fun d _ ⟨b, hb⟩ _ _ _ _ ↦ ⟨b, hb, (Ici b ∩ d).subset_univ⟩
  isOpen_inter s t hs ht d hd₀ hd₁ hd₂ a hd₃ ha := by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      D : Set (Set α)
      inst✝ : Preorder α
      s t : Set α
      hs : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      ht : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hd₃ : IsLUB d a
      ha : Membership.mem (Inter.inter s t) a
      ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
    -/
    obtain ⟨b₁, hb₁d, hb₁ds⟩ := hs hd₀ hd₁ hd₂ hd₃ ha.1
    /-
      case intro.intro
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      D : Set (Set α)
      inst✝ : Preorder α
      s t : Set α
      hs : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      ht : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hd₃ : IsLUB d a
      ha : Membership.mem (Inter.inter s t) a
      b₁ : α
      hb₁d : Membership.mem d b₁
      hb₁ds : HasSubset.Subset (Inter.inter (Set.Ici b₁) d) s
      ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
    -/
    obtain ⟨b₂, hb₂d, hb₂dt⟩ := ht hd₀ hd₁ hd₂ hd₃ ha.2
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      D : Set (Set α)
      inst✝ : Preorder α
      s t : Set α
      hs : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      ht : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hd₃ : IsLUB d a
      ha : Membership.mem (Inter.inter s t) a
      b₁ : α
      hb₁d : Membership.mem d b₁
      hb₁ds : HasSubset.Subset (Inter.inter (Set.Ici b₁) d) s
      b₂ : α
      hb₂d : Membership.mem d b₂
      hb₂dt : HasSubset.Subset (Inter.inter (Set.Ici b₂) d) t
      ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
    -/
    obtain ⟨c, hcd, hc⟩ := hd₂ b₁ hb₁d b₂ hb₂d
    /-
      case intro.intro.intro.intro.intro.intro
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      D : Set (Set α)
      inst✝ : Preorder α
      s t : Set α
      hs : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      ht : (fun u => ∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn (fu …
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hd₃ : IsLUB d a
      ha : Membership.mem (Inter.inter s t) a
      b₁ : α
      hb₁d : Membership.mem d b₁
      hb₁ds : HasSubset.Subset (Inter.inter (Set.Ici b₁) d) s
      b₂ : α
      hb₂d : Membership.mem d b₂
      hb₂dt : HasSubset.Subset (Inter.inter (Set.Ici b₂) d) t
      c : α
      hcd : Membership.mem d c
      hc : And ((fun x1 x2 => LE.le x1 x2) b₁ c) ((fun x1 x2 => LE.le x1 x2) b₂ c)
      ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
    -/
    exact ⟨c, hcd, fun e ⟨hce, hed⟩ ↦ ⟨hb₁ds ⟨hc.1.trans hce, hed⟩, hb₂dt ⟨hc.2.trans hce, hed⟩⟩⟩
    /-
      🎉 no goals
    -/
  isOpen_sUnion := fun s h d hd₀ hd₁ hd₂ a hd₃ ⟨s₀, hs₀s, has₀⟩ ↦ by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      D : Set (Set α)
      inst✝ : Preorder α
      s : Set (Set α)
      h : ∀ (t : Set α), Membership.mem s t → (fun u => ∀ ⦃d : Set α⦄, Membership.me …
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hd₃ : IsLUB d a
      x✝ : Membership.mem s.sUnion a
      s₀ : Set α
      hs₀s : Membership.mem s s₀
      has₀ : Membership.mem s₀ a
      ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
    -/
    obtain ⟨b, hbd, hbds₀⟩ := h s₀ hs₀s hd₀ hd₁ hd₂ hd₃ has₀
    /-
      case intro.intro
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      D : Set (Set α)
      inst✝ : Preorder α
      s : Set (Set α)
      h : ∀ (t : Set α), Membership.mem s t → (fun u => ∀ ⦃d : Set α⦄, Membership.me …
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hd₃ : IsLUB d a
      x✝ : Membership.mem s.sUnion a
      s₀ : Set α
      hs₀s : Membership.mem s s₀
      has₀ : Membership.mem s₀ a
      b : α
      hbd : Membership.mem d b
      hbds₀ : HasSubset.Subset (Inter.inter (Set.Ici b) d) s₀
      ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
    -/
    exact ⟨b, hbd, Set.subset_sUnion_of_subset s s₀ hbds₀ hs₀s⟩
    /-
      🎉 no goals
    -/


/-- Predicate for an ordered topological space to be equipped with its Scott-Hausdorff topology.

A set `u` is open in the Scott-Hausdorff topology iff when the least upper bound of a directed set
`d` lies in `u` then there is a tail of `d` which is a subset of `u`. -/
class IsScottHausdorff : Prop where
  topology_eq_scottHausdorff : ‹TopologicalSpace α› = scottHausdorff α D


instance : @IsScottHausdorff α D _ (scottHausdorff α D) :=
  @IsScottHausdorff.mk _ _ _ (scottHausdorff α D) rfl


lemma topology_eq [IsScottHausdorff α D] : ‹_› = scottHausdorff α D := topology_eq_scottHausdorff


lemma isOpen_iff [IsScottHausdorff α D] :
    IsOpen s ↔ ∀ ⦃d : Set α⦄, d ∈ D → d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a : α⦄, IsLUB d a →
      a ∈ s → ∃ b ∈ d, Ici b ∩ d ⊆ s := by
  /-
    α : Type u_1
    D : Set (Set α)
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScottHausdorff α D
    ⊢ Iff (IsOpen s) (∀ ⦃d : Set α⦄, Membership.mem D d → d.Nonempty → DirectedOn  …
  -/
  simp [topology_eq_scottHausdorff (α := α) (D := D), IsOpen, scottHausdorff]
  /-
    🎉 no goals
  -/


lemma dirSupInaccOn_of_isOpen [IsScottHausdorff α D] (h : IsOpen s) : DirSupInaccOn D s :=
  fun d hd₀ hd₁ hd₂ a hda hd₃ ↦ by
    /-
      α : Type u_1
      D : Set (Set α)
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      s : Set α
      inst✝ : Topology.IsScottHausdorff α D
      h : IsOpen s
      d : Set α
      hd₀ : Membership.mem D d
      hd₁ : d.Nonempty
      hd₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hda : IsLUB d a
      hd₃ : Membership.mem s a
      ⊢ (Inter.inter d s).Nonempty
    -/
    obtain ⟨b, hbd, hb⟩ := isOpen_iff.mp h hd₀ hd₁ hd₂ hda hd₃; exact ⟨b, hbd, hb ⟨le_rfl, hbd⟩⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma dirSupClosed_of_isClosed [IsScottHausdorff α univ] (h : IsClosed s) : DirSupClosed s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScottHausdorff α Set.univ
    h : IsClosed s
    ⊢ DirSupClosed s
  -/
  apply DirSupInacc.of_compl
  /-
    case a
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScottHausdorff α Set.univ
    h : IsClosed s
    ⊢ DirSupInacc (HasCompl.compl s)
  -/
  rw [← dirSupInaccOn_univ]
  /-
    case a
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScottHausdorff α Set.univ
    h : IsClosed s
    ⊢ DirSupInaccOn Set.univ (HasCompl.compl s)
  -/
  exact (dirSupInaccOn_of_isOpen h.isOpen_compl)
  /-
    🎉 no goals
  -/


lemma isOpen_of_isLowerSet (h : IsLowerSet s) : IsOpen s :=
  (isOpen_iff (D := univ)).2 fun _d _ ⟨b, hb⟩ _ _ hda ha ↦
    ⟨b, hb, fun _ hc ↦ h (mem_upperBounds.1 hda.1 _ hc.2) ha⟩


lemma isClosed_of_isUpperSet (h : IsUpperSet s) : IsClosed s :=
  isOpen_compl_iff.1 <| isOpen_of_isLowerSet h.compl


/-- The Scott topology.

It is defined as the join of the topology of upper sets and the Scott-Hausdorff topology. -/
def scott (α : Type*) (D : Set (Set α)) [Preorder α] : TopologicalSpace α :=
  upperSet α ⊔ scottHausdorff α D


lemma upperSet_le_scott [Preorder α] : upperSet α ≤ scott α univ := le_sup_left


lemma scottHausdorff_le_scott [Preorder α] : scottHausdorff α univ ≤ scott α univ:= le_sup_right


/-- Predicate for an ordered topological space to be equipped with its Scott topology.

The Scott topology is defined as the join of the topology of upper sets and the Scott Hausdorff
topology. -/
class IsScott : Prop where
  topology_eq_scott : ‹TopologicalSpace α› = scott α D


lemma topology_eq [IsScott α D] : ‹_› = scott α D := topology_eq_scott


lemma isOpen_iff_isUpperSet_and_scottHausdorff_open [IsScott α D] :
                                                                 /-
                                                                   α : Type u_1
                                                                   D : Set (Set α)
                                                                   inst✝² : Preorder α
                                                                   inst✝¹ : TopologicalSpace α
                                                                   s : Set α
                                                                   inst✝ : Topology.IsScott α D
                                                                   ⊢ Iff (IsOpen s) (And (IsUpperSet s) (IsOpen s))
                                                                 -/
    IsOpen s ↔ IsUpperSet s ∧ IsOpen[scottHausdorff α D] s := by rw [topology_eq α D]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


lemma isOpen_iff_isUpperSet_and_dirSupInaccOn [IsScott α D] :
    IsOpen s ↔ IsUpperSet s ∧ DirSupInaccOn D s := by
  /-
    α : Type u_1
    D : Set (Set α)
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScott α D
    ⊢ Iff (IsOpen s) (And (IsUpperSet s) (DirSupInaccOn D s))
  -/
  rw [isOpen_iff_isUpperSet_and_scottHausdorff_open (D := D)]
  refine and_congr_right fun h ↦
    ⟨@IsScottHausdorff.dirSupInaccOn_of_isOpen _ _ _ (scottHausdorff α D) _ _,
      fun h' d d₀ d₁ d₂ _ d₃ ha ↦ ?_⟩
  /-
    α : Type u_1
    D : Set (Set α)
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScott α D
    h : IsUpperSet s
    h' : DirSupInaccOn D s
    d : Set α
    d₀ : Membership.mem D d
    d₁ : d.Nonempty
    d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    x✝ : α
    d₃ : IsLUB d x✝
    ha : Membership.mem s x✝
    ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
  -/
  obtain ⟨b, hbd, hbu⟩ := h' d₀ d₁ d₂ d₃ ha
  /-
    case intro.intro
    α : Type u_1
    D : Set (Set α)
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScott α D
    h : IsUpperSet s
    h' : DirSupInaccOn D s
    d : Set α
    d₀ : Membership.mem D d
    d₁ : d.Nonempty
    d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    x✝ : α
    d₃ : IsLUB d x✝
    ha : Membership.mem s x✝
    b : α
    hbd : Membership.mem d b
    hbu : Membership.mem s b
    ⊢ Exists fun b => And (Membership.mem d b) (HasSubset.Subset (Inter.inter (Set …
  -/
  exact ⟨b, hbd, Subset.trans inter_subset_left (h.Ici_subset hbu)⟩
  /-
    🎉 no goals
  -/


lemma isClosed_iff_isLowerSet_and_dirSupClosed [IsScott α univ] :
    IsClosed s ↔ IsLowerSet s ∧ DirSupClosed s := by
  rw [← isOpen_compl_iff, isOpen_iff_isUpperSet_and_dirSupInaccOn (D := univ), isUpperSet_compl,
    dirSupInaccOn_univ, dirSupInacc_compl]


lemma isUpperSet_of_isOpen [IsScott α D] : IsOpen s → IsUpperSet s := fun h ↦
  (isOpen_iff_isUpperSet_and_scottHausdorff_open (D := D).mp h).left


lemma isLowerSet_of_isClosed [IsScott α univ] : IsClosed s → IsLowerSet s := fun h ↦
  (isClosed_iff_isLowerSet_and_dirSupClosed.mp h).left


lemma dirSupClosed_of_isClosed [IsScott α univ] : IsClosed s → DirSupClosed s := fun h ↦
  (isClosed_iff_isLowerSet_and_dirSupClosed.mp h).right


lemma lowerClosure_subset_closure [IsScott α univ] : ↑(lowerClosure s) ⊆ closure s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : Topology.IsScott α Set.univ
    ⊢ HasSubset.Subset (↑(lowerClosure s)) (closure s)
  -/
  convert closure.mono (@upperSet_le_scott α _)
    /-
      case h.e'_3
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      s : Set α
      inst✝ : Topology.IsScott α Set.univ
      ⊢ Eq (↑(lowerClosure s)) (closure s)
    -/
  · rw [@IsUpperSet.closure_eq_lowerClosure α _ (upperSet α) ?_ s]
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      s : Set α
      inst✝ : Topology.IsScott α Set.univ
      ⊢ Topology.IsUpperSet α
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_2
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      s : Set α
      inst✝ : Topology.IsScott α Set.univ
      ⊢ Eq inst✝¹ (Topology.scott α Set.univ)
    -/
  · exact topology_eq α univ
    /-
      🎉 no goals
    -/


lemma isClosed_Iic [IsScott α univ] : IsClosed (Iic a) :=
  isClosed_iff_isLowerSet_and_dirSupClosed.2 ⟨isLowerSet_Iic _, dirSupClosed_Iic _⟩


/--
The closure of a singleton `{a}` in the Scott topology is the right-closed left-infinite interval
`(-∞,a]`.
-/
@[simp] lemma closure_singleton [IsScott α univ] : closure {a} = Iic a := le_antisymm
                       /-
                         α : Type u_1
                         inst✝² : Preorder α
                         inst✝¹ : TopologicalSpace α
                         a : α
                         inst✝ : Topology.IsScott α Set.univ
                         ⊢ HasSubset.Subset (Singleton.singleton a) (Set.Iic a)
                       -/
  (closure_minimal (by rw [singleton_subset_iff, mem_Iic]) isClosed_Iic) <| by
                       /-
                         🎉 no goals
                       -/
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      a : α
      inst✝ : Topology.IsScott α Set.univ
      ⊢ LE.le (Set.Iic a) (closure (Singleton.singleton a))
    -/
    rw [← LowerSet.coe_Iic, ← lowerClosure_singleton]
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      a : α
      inst✝ : Topology.IsScott α Set.univ
      ⊢ LE.le (↑(lowerClosure (Singleton.singleton a))) (closure (Singleton.singleto …
    -/
    apply lowerClosure_subset_closure
    /-
      🎉 no goals
    -/


lemma monotone_of_continuous [IsScott α D] (hf : Continuous f) : Monotone f := fun _ b hab ↦ by
  /-
    α : Type u_1
    β : Type u_2
    D : Set (Set α)
    inst✝⁵ : Preorder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : Preorder β
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsScott β Set.univ
    f : α → β
    inst✝ : Topology.IsScott α D
    hf : Continuous f
    x✝ b : α
    hab : LE.le x✝ b
    ⊢ LE.le (f x✝) (f b)
  -/
  by_contra h
  simpa only [mem_compl_iff, mem_preimage, mem_Iic, le_refl, not_true]
    using isUpperSet_of_isOpen (D := D) ((isOpen_compl_iff.2 isClosed_Iic).preimage hf) hab h


@[simp] lemma scottContinuous_iff_continuous {D : Set (Set α)} [Topology.IsScott α D]
    (hD : ∀ a b : α, a ≤ b → {a, b} ∈ D) : ScottContinuousOn D f ↔ Continuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : Preorder β
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsScott β Set.univ
    f : α → β
    D : Set (Set α)
    inst✝ : Topology.IsScott α D
    hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
    ⊢ Iff (ScottContinuousOn D f) (Continuous f)
  -/
  refine ⟨fun h ↦ continuous_def.2 fun u hu ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      h : ScottContinuousOn D f
      u : Set β
      hu : IsOpen u
      ⊢ IsOpen (Set.preimage f u)
    -/
  · rw [isOpen_iff_isUpperSet_and_dirSupInaccOn (D := D)]
    exact ⟨(isUpperSet_of_isOpen (D := univ) hu).preimage (h.monotone D hD),
      fun t h₀ hd₁ hd₂ a hd₃ ha ↦ image_inter_nonempty_iff.mp <|
        (isOpen_iff_isUpperSet_and_dirSupInaccOn (D := univ).mp hu).2 trivial (Nonempty.image f hd₁)
        (directedOn_image.mpr (hd₂.mono @(h.monotone D hD))) (h h₀ hd₁ hd₂ hd₃) ha⟩
  · refine fun hf t h₀ d₁ d₂ a d₃ ↦
      ⟨(monotone_of_continuous (D := D) hf).mem_upperBounds_image d₃.1,
      fun b hb ↦ ?_⟩
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : Continuous f
      t : Set α
      h₀ : Membership.mem D t
      d₁ : t.Nonempty
      d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) t
      a : α
      d₃ : IsLUB t a
      b : β
      hb : Membership.mem (upperBounds (Set.image f t)) b
      ⊢ LE.le (f a) b
    -/
    by_contra h
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : Continuous f
      t : Set α
      h₀ : Membership.mem D t
      d₁ : t.Nonempty
      d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) t
      a : α
      d₃ : IsLUB t a
      b : β
      hb : Membership.mem (upperBounds (Set.image f t)) b
      h : Not (LE.le (f a) b)
      ⊢ False
    -/
    let u := (Iic b)ᶜ
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : Continuous f
      t : Set α
      h₀ : Membership.mem D t
      d₁ : t.Nonempty
      d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) t
      a : α
      d₃ : IsLUB t a
      b : β
      hb : Membership.mem (upperBounds (Set.image f t)) b
      h : Not (LE.le (f a) b)
      u : Set β := HasCompl.compl (Set.Iic b)
      ⊢ False
    -/
    have hu : IsOpen (f ⁻¹' u) := (isOpen_compl_iff.2 Topology.IsScott.isClosed_Iic).preimage hf
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : Continuous f
      t : Set α
      h₀ : Membership.mem D t
      d₁ : t.Nonempty
      d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) t
      a : α
      d₃ : IsLUB t a
      b : β
      hb : Membership.mem (upperBounds (Set.image f t)) b
      h : Not (LE.le (f a) b)
      u : Set β := HasCompl.compl (Set.Iic b)
      hu : IsOpen (Set.preimage f u)
      ⊢ False
    -/
    rw [isOpen_iff_isUpperSet_and_dirSupInaccOn (D := D)] at hu
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : Continuous f
      t : Set α
      h₀ : Membership.mem D t
      d₁ : t.Nonempty
      d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) t
      a : α
      d₃ : IsLUB t a
      b : β
      hb : Membership.mem (upperBounds (Set.image f t)) b
      h : Not (LE.le (f a) b)
      u : Set β := HasCompl.compl (Set.Iic b)
      hu : And (IsUpperSet (Set.preimage f u)) (DirSupInaccOn D (Set.preimage f u))
      ⊢ False
    -/
    obtain ⟨c, hcd, hfcb⟩ := hu.2 h₀ d₁ d₂ d₃ h
    simp only [upperBounds, mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂,
      mem_setOf] at hb
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder β
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsScott β Set.univ
      f : α → β
      D : Set (Set α)
      inst✝ : Topology.IsScott α D
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : Continuous f
      t : Set α
      h₀ : Membership.mem D t
      d₁ : t.Nonempty
      d₂ : DirectedOn (fun x1 x2 => LE.le x1 x2) t
      a : α
      d₃ : IsLUB t a
      b : β
      h : Not (LE.le (f a) b)
      u : Set β := HasCompl.compl (Set.Iic b)
      hu : And (IsUpperSet (Set.preimage f u)) (DirSupInaccOn D (Set.preimage f u))
      c : α
      hcd : Membership.mem t c
      hfcb : Membership.mem (Set.preimage f u) c
      hb : ∀ (a : α), Membership.mem t a → LE.le (f a) b
      ⊢ False
    -/
    exact hfcb <| hb _ hcd
    /-
      🎉 no goals
    -/


/--
The Scott topology on a partial order is T₀.
-/
-- see Note [lower instance priority]
instance (priority := 90) : T0Space α :=
  (t0Space_iff_inseparable α).2 fun x y h ↦ Iic_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsScott α Set.univ
      x y : α
      h : Inseparable x y
      ⊢ Eq (Set.Iic x) (Set.Iic y)
    -/
    simpa only [inseparable_iff_closure_eq, IsScott.closure_singleton] using h
    /-
      🎉 no goals
    -/


lemma isOpen_iff_Iic_compl_or_univ [TopologicalSpace α] [Topology.IsScott α univ] (U : Set α) :
    IsOpen U ↔ U = univ ∨ ∃ a, (Iic a)ᶜ = U := by
  /-
    α : Type u_1
    inst✝² : CompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsScott α Set.univ
    U : Set α
    ⊢ Iff (IsOpen U) (Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set. …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝² : CompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsScott α Set.univ
      U : Set α
      ⊢ IsOpen U → Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set.Iic a …
    -/
  · intro hU
    /-
      case mp
      α : Type u_1
      inst✝² : CompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsScott α Set.univ
      U : Set α
      hU : IsOpen U
      ⊢ Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set.Iic a)) U)
    -/
    rcases eq_empty_or_nonempty Uᶜ with eUc | neUc
      /-
        case mp.inl
        α : Type u_1
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : Topology.IsScott α Set.univ
        U : Set α
        hU : IsOpen U
        eUc : Eq (HasCompl.compl U) EmptyCollection.emptyCollection
        ⊢ Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set.Iic a)) U)
      -/
    · exact Or.inl (compl_empty_iff.mp eUc)
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        α : Type u_1
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : Topology.IsScott α Set.univ
        U : Set α
        hU : IsOpen U
        neUc : (HasCompl.compl U).Nonempty
        ⊢ Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set.Iic a)) U)
      -/
    · apply Or.inr
      /-
        case mp.inr.h
        α : Type u_1
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : Topology.IsScott α Set.univ
        U : Set α
        hU : IsOpen U
        neUc : (HasCompl.compl U).Nonempty
        ⊢ Exists fun a => Eq (HasCompl.compl (Set.Iic a)) U
      -/
      use sSup Uᶜ
      /-
        case h
        α : Type u_1
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : Topology.IsScott α Set.univ
        U : Set α
        hU : IsOpen U
        neUc : (HasCompl.compl U).Nonempty
        ⊢ Eq (HasCompl.compl (Set.Iic (SupSet.sSup (HasCompl.compl U)))) U
      -/
      rw [compl_eq_comm, le_antisymm_iff]
      exact ⟨fun _ ha ↦ le_sSup ha, (isLowerSet_of_isClosed hU.isClosed_compl).Iic_subset
        (dirSupClosed_iff_forall_sSup.mp (dirSupClosed_of_isClosed hU.isClosed_compl)
        neUc (isChain_of_trichotomous Uᶜ).directedOn le_rfl)⟩
    /-
      case mpr
      α : Type u_1
      inst✝² : CompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsScott α Set.univ
      U : Set α
      ⊢ Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set.Iic a)) U) → IsO …
    -/
  · rintro (rfl | ⟨a, rfl⟩)
      /-
        case mpr.inl
        α : Type u_1
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : Topology.IsScott α Set.univ
        ⊢ IsOpen Set.univ
      -/
    · exact isOpen_univ
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        α : Type u_1
        inst✝² : CompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : Topology.IsScott α Set.univ
        a : α
        ⊢ IsOpen (HasCompl.compl (Set.Iic a))
      -/
    · exact isClosed_Iic.isOpen_compl
      /-
        🎉 no goals
      -/

-- N.B. A number of conditions equivalent to `scott α = upper α` are given in Gierz _et al_,
-- Chapter III, Exercise 3.23.

lemma scott_eq_upper_of_completeLinearOrder : scott α univ = upper α := by
  /-
    α : Type u_1
    inst✝ : CompleteLinearOrder α
    ⊢ Eq (Topology.scott α Set.univ) (Topology.upper α)
  -/
  letI := upper α
  /-
    α : Type u_1
    inst✝ : CompleteLinearOrder α
    this : TopologicalSpace α := Topology.upper α
    ⊢ Eq (Topology.scott α Set.univ) (Topology.upper α)
  -/
  ext U
  rw [@Topology.IsUpper.isTopologicalSpace_basis _ _ (upper α)
    ({ topology_eq_upperTopology := rfl }) U]
  /-
    case a.h.a
    α : Type u_1
    inst✝ : CompleteLinearOrder α
    this : TopologicalSpace α := Topology.upper α
    U : Set α
    ⊢ Iff (IsOpen U) (Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set. …
  -/
  letI := scott α univ
  /-
    case a.h.a
    α : Type u_1
    inst✝ : CompleteLinearOrder α
    this✝ : TopologicalSpace α := Topology.upper α
    U : Set α
    this : TopologicalSpace α := Topology.scott α Set.univ
    ⊢ Iff (IsOpen U) (Or (Eq U Set.univ) (Exists fun a => Eq (HasCompl.compl (Set. …
  -/
  rw [@isOpen_iff_Iic_compl_or_univ _ _ (scott α univ) ({ topology_eq_scott := rfl }) U]
  /-
    🎉 no goals
  -/

/- The upper topology on a complete linear order is the Scott topology -/

instance [TopologicalSpace α] [IsUpper α] : IsScott α univ where
  topology_eq_scott := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : CompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsUpper α
      ⊢ Eq inst✝¹ (Topology.scott α Set.univ)
    -/
    rw [scott_eq_upper_of_completeLinearOrder]
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : CompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsUpper α
      ⊢ Eq inst✝¹ (Topology.upper α)
    -/
    exact IsUpper.topology_eq α
    /-
      🎉 no goals
    -/


lemma isOpen_iff_scottContinuous_mem [Preorder α] {s : Set α} [TopologicalSpace α]
    [IsScott α univ] : IsOpen s ↔ ScottContinuous fun x ↦ x ∈ s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    s : Set α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsScott α Set.univ
    ⊢ Iff (IsOpen s) (ScottContinuous fun x => Membership.mem s x)
  -/
  rw [← scottContinuousOn_univ, scottContinuous_iff_continuous (fun _ _ _ ↦ by trivial)]
  /-
    α : Type u_1
    inst✝² : Preorder α
    s : Set α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsScott α Set.univ
    ⊢ Iff (IsOpen s) (Continuous fun x => Membership.mem s x)
  -/
  exact isOpen_iff_continuous_mem
  /-
    🎉 no goals
  -/


/--
Type synonym for a preorder equipped with the Scott topology
-/
def WithScott (α : Type*) := α


/-- `toScott` is the identity function to the `WithScott` of a type. -/
@[match_pattern] def toScott : α ≃ WithScott α := Equiv.refl _


/-- `ofScott` is the identity function from the `WithScott` of a type. -/
@[match_pattern] def ofScott : WithScott α ≃ α := Equiv.refl _


@[simp] lemma toScott_symm_eq : (@toScott α).symm = ofScott := rfl

@[simp] lemma ofScott_symm_eq : (@ofScott α).symm = toScott := rfl

@[simp] lemma toScott_ofScott (a : WithScott α) : toScott (ofScott a) = a := rfl

@[simp] lemma ofScott_toScott (a : α) : ofScott (toScott a) = a := rfl


lemma toScott_inj {a b : α} : toScott a = toScott b ↔ a = b := Iff.rfl


lemma ofScott_inj {a b : WithScott α} : ofScott a = ofScott b ↔ a = b := Iff.rfl

/-- A recursor for `WithScott`. Use as `induction x`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {β : WithScott α → Sort _}
    (h : ∀ a, β (toScott a)) : ∀ a, β a := fun a ↦ h (ofScott a)


instance [Nonempty α] : Nonempty (WithScott α) := ‹Nonempty α›

instance [Inhabited α] : Inhabited (WithScott α) := ‹Inhabited α›


instance : Preorder (WithScott α) := ‹Preorder α›

instance : TopologicalSpace (WithScott α) := scott α univ

instance : IsScott (WithScott α) univ := ⟨rfl⟩


lemma isOpen_iff_isUpperSet_and_scottHausdorff_open' {u : Set α} :
    IsOpen (WithScott.ofScott ⁻¹' u) ↔ IsUpperSet u ∧ (scottHausdorff α univ).IsOpen u := Iff.rfl


lemma scottHausdorff_le_lower : scottHausdorff α univ ≤ lower α :=
  fun s h => IsScottHausdorff.isOpen_of_isLowerSet (t := scottHausdorff α univ)
      <| (@IsLower.isLowerSet_of_isOpen (Topology.WithLower α) _ _ _ s h)


/-- If `α` is equipped with the Scott topology, then it is homeomorphic to `WithScott α`.
-/
def IsScott.withScottHomeomorph [IsScott α univ] : WithScott α ≃ₜ α :=
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   inst✝² : Preorder α
                                                   inst✝¹ : TopologicalSpace α
                                                   inst✝ : Topology.IsScott α Set.univ
                                                   ⊢ Eq Topology.WithScott.instTopologicalSpace (TopologicalSpace.induced (⇑Topol …
                                                 -/
  WithScott.ofScott.toHomeomorphOfIsInducing ⟨by erw [IsScott.topology_eq α univ, induced_id]; rfl⟩
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


lemma IsScott.scottHausdorff_le [IsScott α univ] :
    scottHausdorff α univ ≤ ‹TopologicalSpace α› := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsScott α Set.univ
    ⊢ LE.le (Topology.scottHausdorff α Set.univ) inst✝¹
  -/
  rw [IsScott.topology_eq α univ, scott]; exact le_sup_right
                                          /-
                                            🎉 no goals
                                          -/


lemma IsLower.scottHausdorff_le [IsLower α] : scottHausdorff α univ ≤ ‹TopologicalSpace α› :=
  fun _ h ↦
    IsScottHausdorff.isOpen_of_isLowerSet (t := scottHausdorff α univ)
      <| IsLower.isLowerSet_of_isOpen h


