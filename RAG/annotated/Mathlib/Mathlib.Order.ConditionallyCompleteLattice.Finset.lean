theorem Finset.Nonempty.csSup_eq_max' {s : Finset α} (h : s.Nonempty) : sSup ↑s = s.max' h :=
  eq_of_forall_ge_iff fun _ => (csSup_le_iff s.bddAbove h.to_set).trans (s.max'_le_iff h).symm


theorem Finset.Nonempty.csInf_eq_min' {s : Finset α} (h : s.Nonempty) : sInf ↑s = s.min' h :=
  @Finset.Nonempty.csSup_eq_max' αᵒᵈ _ s h


theorem Finset.Nonempty.csSup_mem {s : Finset α} (h : s.Nonempty) : sSup (s : Set α) ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Finset α
    h : s.Nonempty
    ⊢ Membership.mem s (SupSet.sSup ↑s)
  -/
  rw [h.csSup_eq_max']
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Finset α
    h : s.Nonempty
    ⊢ Membership.mem s (s.max' h)
  -/
  exact s.max'_mem _
  /-
    🎉 no goals
  -/


theorem Finset.Nonempty.csInf_mem {s : Finset α} (h : s.Nonempty) : sInf (s : Set α) ∈ s :=
  @Finset.Nonempty.csSup_mem αᵒᵈ _ _ h


theorem Set.Nonempty.csSup_mem (h : s.Nonempty) (hs : s.Finite) : sSup s ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    h : s.Nonempty
    hs : s.Finite
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  lift s to Finset α using hs
  /-
    case intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Finset α
    h : (↑s).Nonempty
    ⊢ Membership.mem (↑s) (SupSet.sSup ↑s)
  -/
  exact Finset.Nonempty.csSup_mem h
  /-
    🎉 no goals
  -/


theorem Set.Nonempty.csInf_mem (h : s.Nonempty) (hs : s.Finite) : sInf s ∈ s :=
  @Set.Nonempty.csSup_mem αᵒᵈ _ _ h hs


theorem Set.Finite.csSup_lt_iff (hs : s.Finite) (h : s.Nonempty) : sSup s < a ↔ ∀ x ∈ s, x < a :=
  ⟨fun h _ hx => (le_csSup hs.bddAbove hx).trans_lt h, fun H => H _ <| h.csSup_mem hs⟩


theorem Set.Finite.lt_csInf_iff (hs : s.Finite) (h : s.Nonempty) : a < sInf s ↔ ∀ x ∈ s, a < x :=
  @Set.Finite.csSup_lt_iff αᵒᵈ _ _ _ hs h


theorem Finset.ciSup_eq_max'_image {s : Finset ι} (h : ∃ x ∈ s, sSup ∅ ≤ f x)
    (h' : (s.image f).Nonempty := by classical exact image_nonempty.mpr (h.imp fun _ ↦ And.left)) :
    ⨆ i ∈ s, f i = (s.image f).max' h' := by
  classical
  rw [iSup, ← h'.csSup_eq_max', coe_image]
  refine csSup_eq_csSup_of_forall_exists_le ?_ ?_
  · simp only [ciSup_eq_ite, dite_eq_ite, Set.mem_range, Set.mem_image, mem_coe,
      exists_exists_and_eq_and, forall_exists_index, forall_apply_eq_imp_iff]
    intro i
    split_ifs
    · exact ⟨_, by assumption, le_rfl⟩
    · obtain ⟨a, ha, ha'⟩ := h
      exact ⟨a, ha, ha'⟩
  · simp only [Set.mem_image, mem_coe, ciSup_eq_ite, dite_eq_ite, Set.mem_range,
      exists_exists_eq_and, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    intro i hi
    refine ⟨i, ?_⟩
    simp [hi]


theorem Finset.ciInf_eq_min'_image {s : Finset ι} (h : ∃ x ∈ s, f x ≤ sInf ∅)
    (h' : (s.image f).Nonempty := by classical exact image_nonempty.mpr (h.imp fun _ ↦ And.left)) :
    ⨅ i ∈ s, f i = (s.image f).min' h' := by
  classical
  rw [← OrderDual.toDual_inj, toDual_min', toDual_iInf]
  simp only [Function.comp_apply, toDual_iInf]
  rw [ciSup_eq_max'_image _ h]
  simp only [image_image]
  congr


theorem Finset.ciSup_mem_image {s : Finset ι} (h : ∃ x ∈ s, sSup ∅ ≤ f x) :
    ⨆ i ∈ s, f i ∈ s.image f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
    ⊢ Membership.mem (Finset.image f s) (iSup fun i => iSup fun h => f i)
  -/
  rw [ciSup_eq_max'_image _ h]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
    ⊢ Membership.mem (Finset.image f s) ((Finset.image f s).max' ⋯)
  -/
  exact max'_mem (image f s) _
  /-
    🎉 no goals
  -/


theorem Finset.ciInf_mem_image {s : Finset ι} (h : ∃ x ∈ s, f x ≤ sInf ∅) :
    ⨅ i ∈ s, f i ∈ s.image f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
    ⊢ Membership.mem (Finset.image f s) (iInf fun i => iInf fun h => f i)
  -/
  rw [ciInf_eq_min'_image _ h]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
    ⊢ Membership.mem (Finset.image f s) ((Finset.image f s).min' ⋯)
  -/
  exact min'_mem (image f s) _
  /-
    🎉 no goals
  -/


theorem Set.Finite.ciSup_mem_image {s : Set ι} (hs : s.Finite) (h : ∃ x ∈ s, sSup ∅ ≤ f x) :
    ⨆ i ∈ s, f i ∈ f '' s := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Set ι
    hs : s.Finite
    h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
    ⊢ Membership.mem (Set.image f s) (iSup fun i => iSup fun h => f i)
  -/
  lift s to Finset ι using hs
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem (↑s) x) (LE.le (SupSet.sSup EmptyColle …
    ⊢ Membership.mem (Set.image f ↑s) (iSup fun i => iSup fun h => f i)
  -/
  simp only [Finset.mem_coe] at h
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
    ⊢ Membership.mem (Set.image f ↑s) (iSup fun i => iSup fun h => f i)
  -/
  simpa using Finset.ciSup_mem_image f h
  /-
    🎉 no goals
  -/


theorem Set.Finite.ciInf_mem_image {s : Set ι} (hs : s.Finite) (h : ∃ x ∈ s, f x ≤ sInf ∅) :
    ⨅ i ∈ s, f i ∈ f '' s := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Set ι
    hs : s.Finite
    h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
    ⊢ Membership.mem (Set.image f s) (iInf fun i => iInf fun h => f i)
  -/
  lift s to Finset ι using hs
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem (↑s) x) (LE.le (f x) (InfSet.sInf Empt …
    ⊢ Membership.mem (Set.image f ↑s) (iInf fun i => iInf fun h => f i)
  -/
  simp only [Finset.mem_coe] at h
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    f : ι → α
    s : Finset ι
    h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
    ⊢ Membership.mem (Set.image f ↑s) (iInf fun i => iInf fun h => f i)
  -/
  simpa using Finset.ciInf_mem_image f h
  /-
    🎉 no goals
  -/


theorem Set.Finite.ciSup_lt_iff {s : Set ι} {f : ι → α} (hs : s.Finite)
    (h : ∃ x ∈ s, sSup ∅ ≤ f x) :
    ⨆ i ∈ s, f i < a ↔ ∀ x ∈ s, f x < a := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    a : α
    s : Set ι
    f : ι → α
    hs : s.Finite
    h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
    ⊢ Iff (LT.lt (iSup fun i => iSup fun h => f i) a) (∀ (x : ι), Membership.mem s …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      ⊢ LT.lt (iSup fun i => iSup fun h => f i) a → ∀ (x : ι), Membership.mem s x →  …
    -/
  · intro h x hx
    /-
      case mp
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h✝ : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollect …
      h : LT.lt (iSup fun i => iSup fun h => f i) a
      x : ι
      hx : Membership.mem s x
      ⊢ LT.lt (f x) a
    -/
    refine h.trans_le' (le_csSup ?_ ?_)
    · classical
      refine (((hs.image f).union (finite_singleton (sSup ∅))).subset ?_).bddAbove
      intro
      simp only [ciSup_eq_ite, dite_eq_ite, mem_range, union_singleton, mem_insert_iff, mem_image,
        forall_exists_index]
      intro x hx
      split_ifs at hx
      · exact Or.inr ⟨_, by assumption, hx⟩
      · simp_all
      /-
        case mp.refine_2
        ι : Type u_1
        α : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        s : Set ι
        f : ι → α
        hs : s.Finite
        h✝ : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollect …
        h : LT.lt (iSup fun i => iSup fun h => f i) a
        x : ι
        hx : Membership.mem s x
        ⊢ Membership.mem (Set.range fun i => iSup fun h => f i) (f x)
      -/
    · simp only [mem_range]
      /-
        case mp.refine_2
        ι : Type u_1
        α : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        s : Set ι
        f : ι → α
        hs : s.Finite
        h✝ : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollect …
        h : LT.lt (iSup fun i => iSup fun h => f i) a
        x : ι
        hx : Membership.mem s x
        ⊢ Exists fun y => Eq (iSup fun h => f y) (f x)
      -/
      refine ⟨x, ?_⟩
      /-
        case mp.refine_2
        ι : Type u_1
        α : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        s : Set ι
        f : ι → α
        hs : s.Finite
        h✝ : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollect …
        h : LT.lt (iSup fun i => iSup fun h => f i) a
        x : ι
        hx : Membership.mem s x
        ⊢ Eq (iSup fun h => f x) (f x)
      -/
      simp [hx]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      ⊢ (∀ (x : ι), Membership.mem s x → LT.lt (f x) a) → LT.lt (iSup fun i => iSup  …
    -/
  · intro H
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      H : ∀ (x : ι), Membership.mem s x → LT.lt (f x) a
      ⊢ LT.lt (iSup fun i => iSup fun h => f i) a
    -/
    have := hs.ciSup_mem_image _ h
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      H : ∀ (x : ι), Membership.mem s x → LT.lt (f x) a
      this : Membership.mem (Set.image f s) (iSup fun i => iSup fun h => f i)
      ⊢ LT.lt (iSup fun i => iSup fun h => f i) a
    -/
    simp only [mem_image] at this
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      H : ∀ (x : ι), Membership.mem s x → LT.lt (f x) a
      this : Exists fun x => And (Membership.mem s x) (Eq (f x) (iSup fun i => iSup  …
      ⊢ LT.lt (iSup fun i => iSup fun h => f i) a
    -/
    obtain ⟨_, hmem, hx⟩ := this
    /-
      case mpr.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      H : ∀ (x : ι), Membership.mem s x → LT.lt (f x) a
      w✝ : ι
      hmem : Membership.mem s w✝
      hx : Eq (f w✝) (iSup fun i => iSup fun h => f i)
      ⊢ LT.lt (iSup fun i => iSup fun h => f i) a
    -/
    rw [← hx]
    /-
      case mpr.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollecti …
      H : ∀ (x : ι), Membership.mem s x → LT.lt (f x) a
      w✝ : ι
      hmem : Membership.mem s w✝
      hx : Eq (f w✝) (iSup fun i => iSup fun h => f i)
      ⊢ LT.lt (f w✝) a
    -/
    exact H _ hmem
    /-
      🎉 no goals
    -/


theorem Set.Finite.lt_ciInf_iff {s : Set ι} {f : ι → α} (hs : s.Finite)
    (h : ∃ x ∈ s, f x ≤ sInf ∅) :
    a < ⨅ i ∈ s, f i ↔ ∀ x ∈ s, a < f x := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    a : α
    s : Set ι
    f : ι → α
    hs : s.Finite
    h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
    ⊢ Iff (LT.lt a (iInf fun i => iInf fun h => f i)) (∀ (x : ι), Membership.mem s …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      ⊢ LT.lt a (iInf fun i => iInf fun h => f i) → ∀ (x : ι), Membership.mem s x →  …
    -/
  · intro h x hx
    /-
      case mp
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h✝ : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyC …
      h : LT.lt a (iInf fun i => iInf fun h => f i)
      x : ι
      hx : Membership.mem s x
      ⊢ LT.lt a (f x)
    -/
    refine h.trans_le (csInf_le ?_ ?_)
    · classical
      refine (((hs.image f).union (finite_singleton (sInf ∅))).subset ?_).bddBelow
      intro
      simp only [ciInf_eq_ite, dite_eq_ite, mem_range, union_singleton, mem_insert_iff, mem_image,
        forall_exists_index]
      intro x hx
      split_ifs at hx
      · exact Or.inr ⟨_, by assumption, hx⟩
      · simp_all
      /-
        case mp.refine_2
        ι : Type u_1
        α : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        s : Set ι
        f : ι → α
        hs : s.Finite
        h✝ : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyC …
        h : LT.lt a (iInf fun i => iInf fun h => f i)
        x : ι
        hx : Membership.mem s x
        ⊢ Membership.mem (Set.range fun i => iInf fun h => f i) (f x)
      -/
    · simp only [mem_range]
      /-
        case mp.refine_2
        ι : Type u_1
        α : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        s : Set ι
        f : ι → α
        hs : s.Finite
        h✝ : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyC …
        h : LT.lt a (iInf fun i => iInf fun h => f i)
        x : ι
        hx : Membership.mem s x
        ⊢ Exists fun y => Eq (iInf fun h => f y) (f x)
      -/
      refine ⟨x, ?_⟩
      /-
        case mp.refine_2
        ι : Type u_1
        α : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder α
        a : α
        s : Set ι
        f : ι → α
        hs : s.Finite
        h✝ : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyC …
        h : LT.lt a (iInf fun i => iInf fun h => f i)
        x : ι
        hx : Membership.mem s x
        ⊢ Eq (iInf fun h => f x) (f x)
      -/
      simp [hx]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      ⊢ (∀ (x : ι), Membership.mem s x → LT.lt a (f x)) → LT.lt a (iInf fun i => iIn …
    -/
  · intro H
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      H : ∀ (x : ι), Membership.mem s x → LT.lt a (f x)
      ⊢ LT.lt a (iInf fun i => iInf fun h => f i)
    -/
    have := hs.ciInf_mem_image _ h
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      H : ∀ (x : ι), Membership.mem s x → LT.lt a (f x)
      this : Membership.mem (Set.image f s) (iInf fun i => iInf fun h => f i)
      ⊢ LT.lt a (iInf fun i => iInf fun h => f i)
    -/
    simp only [mem_image] at this
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      H : ∀ (x : ι), Membership.mem s x → LT.lt a (f x)
      this : Exists fun x => And (Membership.mem s x) (Eq (f x) (iInf fun i => iInf  …
      ⊢ LT.lt a (iInf fun i => iInf fun h => f i)
    -/
    obtain ⟨_, hmem, hx⟩ := this
    /-
      case mpr.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      H : ∀ (x : ι), Membership.mem s x → LT.lt a (f x)
      w✝ : ι
      hmem : Membership.mem s w✝
      hx : Eq (f w✝) (iInf fun i => iInf fun h => f i)
      ⊢ LT.lt a (iInf fun i => iInf fun h => f i)
    -/
    rw [← hx]
    /-
      case mpr.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      a : α
      s : Set ι
      f : ι → α
      hs : s.Finite
      h : Exists fun x => And (Membership.mem s x) (LE.le (f x) (InfSet.sInf EmptyCo …
      H : ∀ (x : ι), Membership.mem s x → LT.lt a (f x)
      w✝ : ι
      hmem : Membership.mem s w✝
      hx : Eq (f w✝) (iInf fun i => iInf fun h => f i)
      ⊢ LT.lt a (f w✝)
    -/
    exact H _ hmem
    /-
      🎉 no goals
    -/


lemma List.iSup_mem_map_of_exists_sSup_empty_le {l : List ι} (f : ι → α)
    (h : ∃ x ∈ l, sSup ∅ ≤ f x) :
    ⨆ x ∈ l, f x ∈ l.map f := by
  classical
  simpa using l.toFinset.ciSup_mem_image f (by simpa using h)


lemma List.iInf_mem_map_of_exists_le_sInf_empty {l : List ι} (f : ι → α)
    (h : ∃ x ∈ l, f x ≤ sInf ∅) :
    ⨅ x ∈ l, f x ∈ l.map f := by
  classical
  simpa using l.toFinset.ciInf_mem_image f (by simpa using h)


lemma Multiset.iSup_mem_map_of_exists_sSup_empty_le {s : Multiset ι} (f : ι → α)
    (h : ∃ x ∈ s, sSup ∅ ≤ f x) :
    ⨆ x ∈ s, f x ∈ s.map f := by
  classical
  simpa using s.toFinset.ciSup_mem_image f (by simpa using h)


lemma Multiset.iInf_mem_map_of_exists_le_sInf_empty {s : Multiset ι} (f : ι → α)
    (h : ∃ x ∈ s, f x ≤ sInf ∅) :
    ⨅ x ∈ s, f x ∈ s.map f := by
  classical
  simpa using s.toFinset.ciInf_mem_image f (by simpa using h)


theorem exists_eq_ciSup_of_finite [Nonempty ι] [Finite ι] {f : ι → α} : ∃ i, f i = ⨆ i, f i :=
  Nonempty.csSup_mem (range_nonempty f) (finite_range f)


theorem exists_eq_ciInf_of_finite [Nonempty ι] [Finite ι] {f : ι → α} : ∃ i, f i = ⨅ i, f i :=
  Nonempty.csInf_mem (range_nonempty f) (finite_range f)


theorem sup'_eq_csSup_image (s : Finset ι) (H : s.Nonempty) (f : ι → α) :
    s.sup' H f = sSup (f '' s) :=
  eq_of_forall_ge_iff fun a => by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLattice α
      s : Finset ι
      H : s.Nonempty
      f : ι → α
      a : α
      ⊢ Iff (LE.le (s.sup' H f) a) (LE.le (SupSet.sSup (Set.image f ↑s)) a)
    -/
    simp [csSup_le_iff (s.finite_toSet.image f).bddAbove (H.to_set.image f)]
    /-
      🎉 no goals
    -/


theorem inf'_eq_csInf_image (s : Finset ι) (H : s.Nonempty) (f : ι → α) :
    s.inf' H f = sInf (f '' s) :=
  sup'_eq_csSup_image (α := αᵒᵈ) _ H _


theorem sup'_id_eq_csSup (s : Finset α) (hs) : s.sup' hs id = sSup s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLattice α
    s : Finset α
    hs : s.Nonempty
    ⊢ Eq (s.sup' hs id) (SupSet.sSup ↑s)
  -/
  rw [sup'_eq_csSup_image s hs, Set.image_id]
  /-
    🎉 no goals
  -/


theorem inf'_id_eq_csInf (s : Finset α) (hs) : s.inf' hs id = sInf s :=
  sup'_id_eq_csSup (α := αᵒᵈ) _ hs


lemma sup'_univ_eq_ciSup (f : ι → α) : univ.sup' univ_nonempty f = ⨆ i, f i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLattice α
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    f : ι → α
    ⊢ Eq (Finset.univ.sup' ⋯ f) (iSup fun i => f i)
  -/
  simp [sup'_eq_csSup_image, iSup]
  /-
    🎉 no goals
  -/


lemma inf'_univ_eq_ciInf (f : ι → α) : univ.inf' univ_nonempty f = ⨅ i, f i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : ConditionallyCompleteLattice α
    inst✝¹ : Fintype ι
    inst✝ : Nonempty ι
    f : ι → α
    ⊢ Eq (Finset.univ.inf' ⋯ f) (iInf fun i => f i)
  -/
  simp [inf'_eq_csInf_image, iInf]
  /-
    🎉 no goals
  -/


lemma sup_univ_eq_ciSup [Fintype ι] (f : ι → α) : univ.sup f = ⨆ i, f i :=
  le_antisymm
    (Finset.sup_le fun _ _ => le_ciSup (finite_range _).bddAbove _)
    (ciSup_le' fun _ => Finset.le_sup (mem_univ _))


theorem Finset.Nonempty.ciSup_eq_max'_image {s : Finset ι} (h : s.Nonempty)
    (h' : (s.image f).Nonempty := h.image f) :
    ⨆ i ∈ s, f i = (s.image f).max' h' :=
                                     /-
                                       ι : Type u_1
                                       α : Type u_2
                                       inst✝ : ConditionallyCompleteLinearOrderBot α
                                       f : ι → α
                                       s : Finset ι
                                       h : s.Nonempty
                                       h' : optParam (Finset.image f s).Nonempty ⋯
                                       ⊢ ∀ (a : ι), Membership.mem s a → And (Membership.mem s a) (LE.le (SupSet.sSup …
                                     -/
  s.ciSup_eq_max'_image _ (h.imp (by simp)) _
                                     /-
                                       🎉 no goals
                                     -/


theorem Finset.Nonempty.ciSup_mem_image {s : Finset ι} (h : s.Nonempty) :
    ⨆ i ∈ s, f i ∈ s.image f :=
                                 /-
                                   ι : Type u_1
                                   α : Type u_2
                                   inst✝ : ConditionallyCompleteLinearOrderBot α
                                   f : ι → α
                                   s : Finset ι
                                   h : s.Nonempty
                                   ⊢ ∀ (a : ι), Membership.mem s a → And (Membership.mem s a) (LE.le (SupSet.sSup …
                                 -/
  s.ciSup_mem_image _ (h.imp (by simp))
                                 /-
                                   🎉 no goals
                                 -/


theorem Set.Nonempty.ciSup_mem_image {s : Set ι} (h : s.Nonempty) (hs : s.Finite) :
    ⨆ i ∈ s, f i ∈ f '' s :=
                                  /-
                                    ι : Type u_1
                                    α : Type u_2
                                    inst✝ : ConditionallyCompleteLinearOrderBot α
                                    f : ι → α
                                    s : Set ι
                                    h : s.Nonempty
                                    hs : s.Finite
                                    ⊢ ∀ (a : ι), Membership.mem s a → And (Membership.mem s a) (LE.le (SupSet.sSup …
                                  -/
  hs.ciSup_mem_image _ (h.imp (by simp))
                                  /-
                                    🎉 no goals
                                  -/


theorem Set.Nonempty.ciSup_lt_iff {s : Set ι} {a : α} {f : ι → α} (h : s.Nonempty) (hs : s.Finite) :
    ⨆ i ∈ s, f i < a ↔ ∀ x ∈ s, f x < a :=
                             /-
                               ι : Type u_1
                               α : Type u_2
                               inst✝ : ConditionallyCompleteLinearOrderBot α
                               s : Set ι
                               a : α
                               f : ι → α
                               h : s.Nonempty
                               hs : s.Finite
                               ⊢ ∀ (a : ι), Membership.mem s a → And (Membership.mem s a) (LE.le (SupSet.sSup …
                             -/
  hs.ciSup_lt_iff (h.imp (by simp))
                             /-
                               🎉 no goals
                             -/


lemma List.iSup_mem_map_of_ne_nil {l : List ι} (f : ι → α) (h : l ≠ []) :
    ⨆ x ∈ l, f x ∈ l.map f :=
                                               /-
                                                 ι : Type u_1
                                                 α : Type u_2
                                                 inst✝ : ConditionallyCompleteLinearOrderBot α
                                                 l : List ι
                                                 f : ι → α
                                                 h : Ne l List.nil
                                                 ⊢ Exists fun x => And (Membership.mem l x) (LE.le (SupSet.sSup EmptyCollection …
                                               -/
  l.iSup_mem_map_of_exists_sSup_empty_le _ (by simpa using exists_mem_of_ne_nil _ h)
                                               /-
                                                 🎉 no goals
                                               -/


lemma Multiset.iSup_mem_map_of_ne_zero {s : Multiset ι} (f : ι → α) (h : s ≠ 0) :
    ⨆ x ∈ s, f x ∈ s.map f :=
                                               /-
                                                 ι : Type u_1
                                                 α : Type u_2
                                                 inst✝ : ConditionallyCompleteLinearOrderBot α
                                                 s : Multiset ι
                                                 f : ι → α
                                                 h : Ne s 0
                                                 ⊢ Exists fun x => And (Membership.mem s x) (LE.le (SupSet.sSup EmptyCollection …
                                               -/
  s.iSup_mem_map_of_exists_sSup_empty_le _ (by simpa using exists_mem_of_ne_zero h)
                                               /-
                                                 🎉 no goals
                                               -/


