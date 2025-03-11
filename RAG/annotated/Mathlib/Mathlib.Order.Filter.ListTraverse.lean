theorem sequence_mono : ∀ as bs : List (Filter α), Forall₂ (· ≤ ·) as bs → sequence as ≤ sequence bs
  | [], [], Forall₂.nil => le_rfl
  | _::as, _::bs, Forall₂.cons h hs => seq_mono (map_mono h) (sequence_mono as bs hs)


theorem mem_traverse :
    ∀ (fs : List β) (us : List γ),
      Forall₂ (fun b c => s c ∈ f b) fs us → traverse s us ∈ traverse f fs
  | [], [], Forall₂.nil => mem_pure.2 <| mem_singleton _
  | _::fs, _::us, Forall₂.cons h hs => seq_mem_seq (image_mem_map h) (mem_traverse fs us hs)

-- TODO: add a `Filter.HasBasis` statement

theorem mem_traverse_iff (fs : List β) (t : Set (List α)) :
    t ∈ traverse f fs ↔
      ∃ us : List (Set α), Forall₂ (fun b (s : Set α) => s ∈ f b) fs us ∧ sequence us ⊆ t := by
  /-
    α β : Type u
    f : β → Filter α
    fs : List β
    t : Set (List α)
    ⊢ Iff (Membership.mem (Traversable.traverse f fs) t) (Exists fun us => And (Li …
  -/
  constructor
  · induction fs generalizing t with
    | nil =>
      simp only [sequence, mem_pure, imp_self, forall₂_nil_left_iff, exists_eq_left, Set.pure_def,
        singleton_subset_iff, traverse_nil]
    | cons b fs ih =>
      intro ht
      rcases mem_seq_iff.1 ht with ⟨u, hu, v, hv, ht⟩
      rcases mem_map_iff_exists_image.1 hu with ⟨w, hw, hwu⟩
      rcases ih v hv with ⟨us, hus, hu⟩
      exact ⟨w::us, Forall₂.cons hw hus, (Set.seq_mono hwu hu).trans ht⟩
    /-
      case mpr
      α β : Type u
      f : β → Filter α
      fs : List β
      t : Set (List α)
      ⊢ (Exists fun us => And (List.Forall₂ (fun b s => Membership.mem (f b) s) fs u …
    -/
  · rintro ⟨us, hus, hs⟩
    /-
      case mpr.intro.intro
      α β : Type u
      f : β → Filter α
      fs : List β
      t : Set (List α)
      us : List (Set α)
      hus : List.Forall₂ (fun b s => Membership.mem (f b) s) fs us
      hs : HasSubset.Subset (sequence us) t
      ⊢ Membership.mem (Traversable.traverse f fs) t
    -/
    exact mem_of_superset (mem_traverse _ _ hus) hs
    /-
      🎉 no goals
    -/


