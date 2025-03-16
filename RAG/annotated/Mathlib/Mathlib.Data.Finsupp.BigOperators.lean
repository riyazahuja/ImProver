theorem List.support_sum_subset [AddMonoid M] (l : List (ι →₀ M)) :
    l.sum.support ⊆ l.foldr (Finsupp.support · ⊔ ·) ∅ := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddMonoid M
    l : List (Finsupp ι M)
    ⊢ HasSubset.Subset l.sum.support (List.foldr (fun x1 x2 => Max.max x1.support  …
  -/
  induction' l with hd tl IH
    /-
      case nil
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      ⊢ HasSubset.Subset List.nil.sum.support (List.foldr (fun x1 x2 => Max.max x1.s …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : HasSubset.Subset tl.sum.support (List.foldr (fun x1 x2 => Max.max x1.supp …
      ⊢ HasSubset.Subset (List.cons hd tl).sum.support (List.foldr (fun x1 x2 => Max …
    -/
  · simp only [List.sum_cons, Finset.union_comm]
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : HasSubset.Subset tl.sum.support (List.foldr (fun x1 x2 => Max.max x1.supp …
      ⊢ HasSubset.Subset (HAdd.hAdd hd tl.sum).support (List.foldr (fun x1 x2 => Max …
    -/
    refine Finsupp.support_add.trans (Finset.union_subset_union ?_ IH)
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : HasSubset.Subset tl.sum.support (List.foldr (fun x1 x2 => Max.max x1.supp …
      ⊢ HasSubset.Subset hd.support hd.support
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Multiset.support_sum_subset [AddCommMonoid M] (s : Multiset (ι →₀ M)) :
    s.sum.support ⊆ (s.map Finsupp.support).sup := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    s : Multiset (Finsupp ι M)
    ⊢ HasSubset.Subset s.sum.support (Multiset.map Finsupp.support s).sup
  -/
  induction s using Quot.inductionOn
  simpa only [Multiset.quot_mk_to_coe'', Multiset.sum_coe, Multiset.map_coe, Multiset.sup_coe,
    List.foldr_map] using List.support_sum_subset _


theorem Finset.support_sum_subset [AddCommMonoid M] (s : Finset (ι →₀ M)) :
    (s.sum id).support ⊆ Finset.sup s Finsupp.support := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    s : Finset (Finsupp ι M)
    ⊢ HasSubset.Subset (s.sum id).support (s.sup Finsupp.support)
  -/
  classical convert Multiset.support_sum_subset s.1; simp
  /-
    🎉 no goals
  -/


theorem List.mem_foldr_sup_support_iff [Zero M] {l : List (ι →₀ M)} {x : ι} :
    x ∈ l.foldr (Finsupp.support · ⊔ ·) ∅ ↔ ∃ f ∈ l, x ∈ f.support := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : Zero M
    l : List (Finsupp ι M)
    x : ι
    ⊢ Iff (Membership.mem (List.foldr (fun x1 x2 => Max.max x1.support x2) EmptyCo …
  -/
  simp only [Finset.sup_eq_union, List.foldr_map, Finsupp.mem_support_iff, exists_prop]
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : Zero M
    l : List (Finsupp ι M)
    x : ι
    ⊢ Iff (Membership.mem (List.foldr (fun x1 x2 => Union.union x1.support x2) Emp …
  -/
  induction' l with hd tl IH
    /-
      case nil
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : Zero M
      x : ι
      ⊢ Iff (Membership.mem (List.foldr (fun x1 x2 => Union.union x1.support x2) Emp …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp only [foldr, Function.comp_apply, Finset.mem_union, Finsupp.mem_support_iff, ne_eq, IH,
      find?, mem_cons, exists_eq_or_imp]


theorem Multiset.mem_sup_map_support_iff [Zero M] {s : Multiset (ι →₀ M)} {x : ι} :
    x ∈ (s.map Finsupp.support).sup ↔ ∃ f ∈ s, x ∈ f.support :=
  Quot.inductionOn s fun _ ↦ by
    simpa only [Multiset.quot_mk_to_coe'', Multiset.map_coe, Multiset.sup_coe, List.foldr_map]
    using List.mem_foldr_sup_support_iff


theorem Finset.mem_sup_support_iff [Zero M] {s : Finset (ι →₀ M)} {x : ι} :
    x ∈ s.sup Finsupp.support ↔ ∃ f ∈ s, x ∈ f.support :=
  Multiset.mem_sup_map_support_iff


theorem List.support_sum_eq [AddMonoid M] (l : List (ι →₀ M))
    (hl : l.Pairwise (_root_.Disjoint on Finsupp.support)) :
    l.sum.support = l.foldr (Finsupp.support · ⊔ ·) ∅ := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddMonoid M
    l : List (Finsupp ι M)
    hl : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) l
    ⊢ Eq l.sum.support (List.foldr (fun x1 x2 => Max.max x1.support x2) EmptyColle …
  -/
  induction' l with hd tl IH
    /-
      case nil
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hl : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) List.nil
      ⊢ Eq List.nil.sum.support (List.foldr (fun x1 x2 => Max.max x1.support x2) Emp …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) (List.cons …
      ⊢ Eq (List.cons hd tl).sum.support (List.foldr (fun x1 x2 => Max.max x1.suppor …
    -/
  · simp only [List.pairwise_cons] at hl
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : And (∀ (a' : Finsupp ι M), Membership.mem tl a' → Function.onFun _root_.D …
      ⊢ Eq (List.cons hd tl).sum.support (List.foldr (fun x1 x2 => Max.max x1.suppor …
    -/
    simp only [List.sum_cons, List.foldr_cons, Function.comp_apply]
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : And (∀ (a' : Finsupp ι M), Membership.mem tl a' → Function.onFun _root_.D …
      ⊢ Eq (HAdd.hAdd hd tl.sum).support (Max.max hd.support (List.foldr (fun x1 x2  …
    -/
    rw [Finsupp.support_add_eq, IH hl.right, Finset.sup_eq_union]
    suffices _root_.Disjoint hd.support (tl.foldr (fun x y ↦ (Finsupp.support x ⊔ y)) ∅) by
      exact Finset.disjoint_of_subset_right (List.support_sum_subset _) this
    rw [← List.foldr_map, ← Finset.bot_eq_empty, List.foldr_sup_eq_sup_toFinset,
      Finset.disjoint_sup_right]
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : And (∀ (a' : Finsupp ι M), Membership.mem tl a' → Function.onFun _root_.D …
      ⊢ ∀ ⦃i : Finset ι⦄, Membership.mem (List.map Finsupp.support tl).toFinset i →  …
    -/
    intro f hf
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : And (∀ (a' : Finsupp ι M), Membership.mem tl a' → Function.onFun _root_.D …
      f : Finset ι
      hf : Membership.mem (List.map Finsupp.support tl).toFinset f
      ⊢ _root_.Disjoint hd.support (id f)
    -/
    simp only [List.mem_toFinset, List.mem_map] at hf
    /-
      case cons
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : And (∀ (a' : Finsupp ι M), Membership.mem tl a' → Function.onFun _root_.D …
      f : Finset ι
      hf : Exists fun a => And (Membership.mem tl a) (Eq a.support f)
      ⊢ _root_.Disjoint hd.support (id f)
    -/
    obtain ⟨f, hf, rfl⟩ := hf
    /-
      case cons.intro.intro
      ι : Type u_1
      M : Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : AddMonoid M
      hd : Finsupp ι M
      tl : List (Finsupp ι M)
      IH : List.Pairwise (Function.onFun _root_.Disjoint Finsupp.support) tl → Eq tl …
      hl : And (∀ (a' : Finsupp ι M), Membership.mem tl a' → Function.onFun _root_.D …
      f : Finsupp ι M
      hf : Membership.mem tl f
      ⊢ _root_.Disjoint hd.support (id f.support)
    -/
    exact hl.left _ hf
    /-
      🎉 no goals
    -/


theorem Multiset.support_sum_eq [AddCommMonoid M] (s : Multiset (ι →₀ M))
    (hs : s.Pairwise (_root_.Disjoint on Finsupp.support)) :
    s.sum.support = (s.map Finsupp.support).sup := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    s : Multiset (Finsupp ι M)
    hs : Multiset.Pairwise (Function.onFun Disjoint Finsupp.support) s
    ⊢ Eq s.sum.support (Multiset.map Finsupp.support s).sup
  -/
  induction' s using Quot.inductionOn with a
  /-
    case h
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    a : List (Finsupp ι M)
    hs : Multiset.Pairwise (Function.onFun Disjoint Finsupp.support) (Quot.mk (⇑(L …
    ⊢ Eq (Multiset.sum (Quot.mk (⇑(List.isSetoid (Finsupp ι M))) a)).support (Mult …
  -/
  obtain ⟨l, hl, hd⟩ := hs
  suffices a.Pairwise (_root_.Disjoint on Finsupp.support) by
    convert List.support_sum_eq a this
    dsimp only [Function.comp_def]
    simp only [quot_mk_to_coe'', map_coe, sup_coe, Finset.le_eq_subset,
      Finset.sup_eq_union, Finset.bot_eq_empty, List.foldr_map]
  /-
    case h.intro.intro
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    a l : List (Finsupp ι M)
    hl : Eq (Quot.mk (⇑(List.isSetoid (Finsupp ι M))) a) ↑l
    hd : List.Pairwise (Function.onFun Disjoint Finsupp.support) l
    ⊢ List.Pairwise (Function.onFun Disjoint Finsupp.support) a
  -/
  simp only [Multiset.quot_mk_to_coe'', Multiset.map_coe, Multiset.coe_eq_coe] at hl
  /-
    case h.intro.intro
    ι : Type u_1
    M : Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid M
    a l : List (Finsupp ι M)
    hd : List.Pairwise (Function.onFun Disjoint Finsupp.support) l
    hl : a.Perm l
    ⊢ List.Pairwise (Function.onFun Disjoint Finsupp.support) a
  -/
  exact hl.symm.pairwise hd fun h ↦ _root_.Disjoint.symm h
  /-
    🎉 no goals
  -/


theorem Finset.support_sum_eq [AddCommMonoid M] (s : Finset (ι →₀ M))
    (hs : (s : Set (ι →₀ M)).PairwiseDisjoint Finsupp.support) :
    (s.sum id).support = Finset.sup s Finsupp.support := by
  classical
  suffices s.1.Pairwise (_root_.Disjoint on Finsupp.support) by
    convert Multiset.support_sum_eq s.1 this
    exact (Finset.sum_val _).symm
  obtain ⟨l, hl, hn⟩ : ∃ l : List (ι →₀ M), l.toFinset = s ∧ l.Nodup := by
    refine ⟨s.toList, ?_, Finset.nodup_toList _⟩
    simp
  subst hl
  rwa [List.toFinset_val, List.dedup_eq_self.mpr hn, Multiset.pairwise_coe_iff_pairwise, ←
    List.pairwiseDisjoint_iff_coe_toFinset_pairwise_disjoint hn]
  intro x y hxy
  exact symmetric_disjoint hxy

