/-- Given `M : Matroid α`, the `IndepMatroid α` whose independent sets are
  the subsets of `M.E` that are disjoint from some base of `M` -/
@[simps] def dualIndepMatroid (M : Matroid α) : IndepMatroid α where
  E := M.E
  Indep I := I ⊆ M.E ∧ ∃ B, M.Base B ∧ Disjoint I B
  indep_empty := ⟨empty_subset M.E, M.exists_base.imp (fun _ hB ↦ ⟨hB, empty_disjoint _⟩)⟩
  indep_subset := by
    /-
      α : Type u_1
      M✝ : Matroid α
      I B X : Set α
      M : Matroid α
      ⊢ ∀ ⦃I J : Set α⦄, (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And …
    -/
    rintro I J ⟨hJE, B, hB, hJB⟩ hIJ
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X : Set α
      M : Matroid α
      I J : Set α
      hJE : HasSubset.Subset J M.E
      B : Set α
      hB : M.Base B
      hJB : Disjoint J B
      hIJ : HasSubset.Subset I J
      ⊢ And (HasSubset.Subset I M.E) (Exists fun B => And (M.Base B) (Disjoint I B))
    -/
    exact ⟨hIJ.trans hJE, ⟨B, hB, disjoint_of_subset_left hIJ hJB⟩⟩
    /-
      🎉 no goals
    -/
  indep_aug := by
    /-
      α : Type u_1
      M✝ : Matroid α
      I B X : Set α
      M : Matroid α
      ⊢ ∀ ⦃I B : Set α⦄, (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And …
    -/
    rintro I X ⟨hIE, B, hB, hIB⟩ hI_not_max hX_max
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    have hXE := hX_max.1.1
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    have hB' := (base_compl_iff_maximal_disjoint_base hXE).mpr hX_max

    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      hB' : M.Base (SDiff.sdiff M.E X)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    set B' := M.E \ X with hX
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    have hI := (not_iff_not.mpr (base_compl_iff_maximal_disjoint_base)).mpr hI_not_max
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    obtain ⟨B'', hB'', hB''₁, hB''₂⟩ := (hB'.indep.diff I).exists_base_subset_union_base hB
    rw [← compl_subset_compl, ← hIB.sdiff_eq_right, ← union_diff_distrib, diff_eq, compl_inter,
      compl_compl, union_subset_iff, compl_subset_compl] at hB''₂

    have hssu := (subset_inter (hB''₂.2) hIE).ssubset_of_ne
      (by { rintro rfl; apply hI; convert hB''; simp [hB''.subset_ground] })

    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      B'' : Set α
      hB'' : M.Base B''
      hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
      hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
      hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    obtain ⟨e, ⟨(heB'' : e ∉ _), heE⟩, heI⟩ := exists_of_ssubset hssu
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      B'' : Set α
      hB'' : M.Base B''
      hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
      hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
      hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
      e : α
      heI : Not (Membership.mem I e)
      heB'' : Not (Membership.mem B'' e)
      heE : Membership.mem M.E e
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff X I) x) ((fun I => And (Has …
    -/
    use e
    /-
      case h
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      B'' : Set α
      hB'' : M.Base B''
      hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
      hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
      hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
      e : α
      heI : Not (Membership.mem I e)
      heB'' : Not (Membership.mem B'' e)
      heE : Membership.mem M.E e
      ⊢ And (Membership.mem (SDiff.sdiff X I) e) ((fun I => And (HasSubset.Subset I  …
    -/
    simp_rw [mem_diff, insert_subset_iff, and_iff_left heI, and_iff_right heE, and_iff_right hIE]
    /-
      case h
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      B'' : Set α
      hB'' : M.Base B''
      hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
      hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
      hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
      e : α
      heI : Not (Membership.mem I e)
      heB'' : Not (Membership.mem B'' e)
      heE : Membership.mem M.E e
      ⊢ And (Membership.mem X e) (Exists fun B => And (M.Base B) (Disjoint (Insert.i …
    -/
    refine ⟨by_contra (fun heX ↦ heB'' (hB''₁ ⟨?_, heI⟩)), ⟨B'', hB'', ?_⟩⟩
      /-
        case h.refine_1
        α : Type u_1
        M✝ : Matroid α
        I✝ B✝ X✝ : Set α
        M : Matroid α
        I X : Set α
        hIE : HasSubset.Subset I M.E
        B : Set α
        hB : M.Base B
        hIB : Disjoint I B
        hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
        hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
        hXE : HasSubset.Subset X M.E
        B' : Set α := SDiff.sdiff M.E X
        hB' : M.Base B'
        hX : Eq B' (SDiff.sdiff M.E X)
        hI : Not (M.Base (SDiff.sdiff M.E I))
        B'' : Set α
        hB'' : M.Base B''
        hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
        hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
        hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
        e : α
        heI : Not (Membership.mem I e)
        heB'' : Not (Membership.mem B'' e)
        heE : Membership.mem M.E e
        heX : Not (Membership.mem X e)
        ⊢ Membership.mem B' e
      -/
    · rw [hX]; exact ⟨heE, heX⟩
               /-
                 🎉 no goals
               -/
    /-
      case h.refine_2
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      B'' : Set α
      hB'' : M.Base B''
      hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
      hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
      hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
      e : α
      heI : Not (Membership.mem I e)
      heB'' : Not (Membership.mem B'' e)
      heE : Membership.mem M.E e
      ⊢ Disjoint (Insert.insert e I) B''
    -/
    rw [← union_singleton, disjoint_union_left, disjoint_singleton_left, and_iff_left heB'']
    /-
      case h.refine_2
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      I X : Set α
      hIE : HasSubset.Subset I M.E
      B : Set α
      hB : M.Base B
      hIB : Disjoint I B
      hI_not_max : Not (Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B …
      hX_max : Maximal (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
      hXE : HasSubset.Subset X M.E
      B' : Set α := SDiff.sdiff M.E X
      hB' : M.Base B'
      hX : Eq B' (SDiff.sdiff M.E X)
      hI : Not (M.Base (SDiff.sdiff M.E I))
      B'' : Set α
      hB'' : M.Base B''
      hB''₁ : HasSubset.Subset (SDiff.sdiff B' I) B''
      hB''₂ : And (HasSubset.Subset B'' (Union.union B' B)) (HasSubset.Subset I (Has …
      hssu : HasSSubset.SSubset I (Inter.inter (HasCompl.compl B'') M.E)
      e : α
      heI : Not (Membership.mem I e)
      heB'' : Not (Membership.mem B'' e)
      heE : Membership.mem M.E e
      ⊢ Disjoint I B''
    -/
    exact disjoint_of_subset_left hB''₂.2 disjoint_compl_left
    /-
      🎉 no goals
    -/
  indep_maximal := by
    /-
      α : Type u_1
      M✝ : Matroid α
      I B X : Set α
      M : Matroid α
      ⊢ ∀ (X : Set α), HasSubset.Subset X M.E → Matroid.ExistsMaximalSubsetProperty  …
    -/
    rintro X - I' ⟨hI'E, B, hB, hI'B⟩ hI'X
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      ⊢ Exists fun J => And (HasSubset.Subset I' J) (Maximal (fun K => And ((fun I = …
    -/
    obtain ⟨I, hI⟩ := M.exists_basis (M.E \ X)
    /-
      case intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I✝ B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      I : Set α
      hI : M.Basis I (SDiff.sdiff M.E X)
      ⊢ Exists fun J => And (HasSubset.Subset I' J) (Maximal (fun K => And ((fun I = …
    -/
    obtain ⟨B', hB', hIB', hB'IB⟩ := hI.indep.exists_base_subset_union_base hB

    obtain rfl : I = B' \ X := hI.eq_of_subset_indep (hB'.indep.diff _)
      (subset_diff.2 ⟨hIB', (subset_diff.1 hI.subset).2⟩)
      (diff_subset_diff_left hB'.subset_ground)
    /-
      case intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      B' : Set α
      hB' : M.Base B'
      hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
      hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
      ⊢ Exists fun J => And (HasSubset.Subset I' J) (Maximal (fun K => And ((fun I = …
    -/
    simp_rw [maximal_subset_iff']
    /-
      case intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      B' : Set α
      hB' : M.Base B'
      hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
      hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
      ⊢ Exists fun J => And (HasSubset.Subset I' J) (And (And (And (HasSubset.Subset …
    -/
    refine ⟨(X \ B') ∩ M.E, ?_, ⟨⟨inter_subset_right, ?_⟩, ?_⟩, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_1
        α : Type u_1
        M✝ : Matroid α
        I B✝ X✝ : Set α
        M : Matroid α
        X I' : Set α
        hI'E : HasSubset.Subset I' M.E
        B : Set α
        hB : M.Base B
        hI'B : Disjoint I' B
        hI'X : HasSubset.Subset I' X
        B' : Set α
        hB' : M.Base B'
        hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
        hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
        hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
        ⊢ HasSubset.Subset I' (Inter.inter (SDiff.sdiff X B') M.E)
      -/
    · rw [subset_inter_iff, and_iff_left hI'E, subset_diff, and_iff_right hI'X]
      exact Disjoint.mono_right hB'IB <| disjoint_union_right.2
        ⟨disjoint_sdiff_right.mono_left hI'X  , hI'B⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_2
        α : Type u_1
        M✝ : Matroid α
        I B✝ X✝ : Set α
        M : Matroid α
        X I' : Set α
        hI'E : HasSubset.Subset I' M.E
        B : Set α
        hB : M.Base B
        hI'B : Disjoint I' B
        hI'X : HasSubset.Subset I' X
        B' : Set α
        hB' : M.Base B'
        hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
        hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
        hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
        ⊢ Exists fun B => And (M.Base B) (Disjoint (Inter.inter (SDiff.sdiff X B') M.E …
      -/
    · exact ⟨B', hB', (disjoint_sdiff_left (t := X)).mono_left inter_subset_left⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_3
        α : Type u_1
        M✝ : Matroid α
        I B✝ X✝ : Set α
        M : Matroid α
        X I' : Set α
        hI'E : HasSubset.Subset I' M.E
        B : Set α
        hB : M.Base B
        hI'B : Disjoint I' B
        hI'X : HasSubset.Subset I' X
        B' : Set α
        hB' : M.Base B'
        hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
        hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
        hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
        ⊢ HasSubset.Subset (Inter.inter (SDiff.sdiff X B') M.E) X
      -/
    · exact inter_subset_left.trans diff_subset
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      B' : Set α
      hB' : M.Base B'
      hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
      hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
      ⊢ ∀ ⦃t : Set α⦄, And (And (HasSubset.Subset t M.E) (Exists fun B => And (M.Bas …
    -/
    simp only [subset_inter_iff, subset_diff, and_imp, forall_exists_index]
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      B' : Set α
      hB' : M.Base B'
      hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
      hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
      ⊢ ∀ ⦃t : Set α⦄, HasSubset.Subset t M.E → ∀ (x : Set α), M.Base x → Disjoint t …
    -/
    refine fun J hJE B'' hB'' hdj hJX hXJ ↦ ⟨⟨hJX, ?_⟩, hJE⟩

    have hI' : (B'' ∩ X) ∪ (B' \ X) ⊆ B' := by
      rw [union_subset_iff, and_iff_left diff_subset, ← union_diff_cancel hJX,
        inter_union_distrib_left, hdj.symm.inter_eq, empty_union, diff_eq, ← inter_assoc,
        ← diff_eq, diff_subset_comm, diff_eq, inter_assoc, ← diff_eq, inter_comm]
      exact subset_trans (inter_subset_inter_right _ hB''.subset_ground) hXJ

    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      B' : Set α
      hB' : M.Base B'
      hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
      hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
      J : Set α
      hJE : HasSubset.Subset J M.E
      B'' : Set α
      hB'' : M.Base B''
      hdj : Disjoint J B''
      hJX : HasSubset.Subset J X
      hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B') M.E) J
      hI' : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B' X)) B'
      ⊢ Disjoint J B'
    -/
    obtain ⟨B₁,hB₁,hI'B₁,hB₁I⟩ := (hB'.indep.subset hI').exists_base_subset_union_base hB''
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      B' : Set α
      hB' : M.Base B'
      hI : M.Basis (SDiff.sdiff B' X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B' X) B'
      hB'IB : HasSubset.Subset B' (Union.union (SDiff.sdiff B' X) B)
      J : Set α
      hJE : HasSubset.Subset J M.E
      B'' : Set α
      hB'' : M.Base B''
      hdj : Disjoint J B''
      hJX : HasSubset.Subset J X
      hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B') M.E) J
      hI' : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B' X)) B'
      B₁ : Set α
      hB₁ : M.Base B₁
      hI'B₁ : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B' X)) B₁
      hB₁I : HasSubset.Subset B₁ (Union.union (Union.union (Inter.inter B'' X) (SDif …
      ⊢ Disjoint J B'
    -/
    rw [union_comm, ← union_assoc, union_eq_self_of_subset_right inter_subset_left] at hB₁I

    obtain rfl : B₁ = B' := by
      refine hB₁.eq_of_subset_indep hB'.indep (fun e he ↦ ?_)
      refine (hB₁I he).elim (fun heB'' ↦ ?_) (fun h ↦ h.1)
      refine (em (e ∈ X)).elim (fun heX ↦ hI' (Or.inl ⟨heB'', heX⟩)) (fun heX ↦ hIB' ?_)
      refine hI.mem_of_insert_indep ⟨hB₁.subset_ground he, heX⟩ ?_
      exact hB₁.indep.subset (insert_subset he (subset_union_right.trans hI'B₁))
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      J : Set α
      hJE : HasSubset.Subset J M.E
      B'' : Set α
      hB'' : M.Base B''
      hdj : Disjoint J B''
      hJX : HasSubset.Subset J X
      B₁ : Set α
      hB₁ hB' : M.Base B₁
      hI : M.Basis (SDiff.sdiff B₁ X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B₁ X) B₁
      hB'IB : HasSubset.Subset B₁ (Union.union (SDiff.sdiff B₁ X) B)
      hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B₁) M.E) J
      hI' hI'B₁ : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B₁  …
      hB₁I : HasSubset.Subset B₁ (Union.union B'' (SDiff.sdiff B₁ X))
      ⊢ Disjoint J B₁
    -/
    by_contra hdj'
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      J : Set α
      hJE : HasSubset.Subset J M.E
      B'' : Set α
      hB'' : M.Base B''
      hdj : Disjoint J B''
      hJX : HasSubset.Subset J X
      B₁ : Set α
      hB₁ hB' : M.Base B₁
      hI : M.Basis (SDiff.sdiff B₁ X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B₁ X) B₁
      hB'IB : HasSubset.Subset B₁ (Union.union (SDiff.sdiff B₁ X) B)
      hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B₁) M.E) J
      hI' hI'B₁ : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B₁  …
      hB₁I : HasSubset.Subset B₁ (Union.union B'' (SDiff.sdiff B₁ X))
      hdj' : Not (Disjoint J B₁)
      ⊢ False
    -/
    obtain ⟨e, heJ, heB'⟩ := not_disjoint_iff.mp hdj'
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4.intro.intro.intro.intr …
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      J : Set α
      hJE : HasSubset.Subset J M.E
      B'' : Set α
      hB'' : M.Base B''
      hdj : Disjoint J B''
      hJX : HasSubset.Subset J X
      B₁ : Set α
      hB₁ hB' : M.Base B₁
      hI : M.Basis (SDiff.sdiff B₁ X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B₁ X) B₁
      hB'IB : HasSubset.Subset B₁ (Union.union (SDiff.sdiff B₁ X) B)
      hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B₁) M.E) J
      hI' hI'B₁ : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B₁  …
      hB₁I : HasSubset.Subset B₁ (Union.union B'' (SDiff.sdiff B₁ X))
      hdj' : Not (Disjoint J B₁)
      e : α
      heJ : Membership.mem J e
      heB' : Membership.mem B₁ e
      ⊢ False
    -/
    obtain (heB'' | ⟨-,heX⟩ ) := hB₁I heB'
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_4.intro.intro.intro.intr …
        α : Type u_1
        M✝ : Matroid α
        I B✝ X✝ : Set α
        M : Matroid α
        X I' : Set α
        hI'E : HasSubset.Subset I' M.E
        B : Set α
        hB : M.Base B
        hI'B : Disjoint I' B
        hI'X : HasSubset.Subset I' X
        J : Set α
        hJE : HasSubset.Subset J M.E
        B'' : Set α
        hB'' : M.Base B''
        hdj : Disjoint J B''
        hJX : HasSubset.Subset J X
        B₁ : Set α
        hB₁ hB' : M.Base B₁
        hI : M.Basis (SDiff.sdiff B₁ X) (SDiff.sdiff M.E X)
        hIB' : HasSubset.Subset (SDiff.sdiff B₁ X) B₁
        hB'IB : HasSubset.Subset B₁ (Union.union (SDiff.sdiff B₁ X) B)
        hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B₁) M.E) J
        hI' hI'B₁ : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B₁  …
        hB₁I : HasSubset.Subset B₁ (Union.union B'' (SDiff.sdiff B₁ X))
        hdj' : Not (Disjoint J B₁)
        e : α
        heJ : Membership.mem J e
        heB' : Membership.mem B₁ e
        heB'' : Membership.mem B'' e
        ⊢ False
      -/
    · exact hdj.ne_of_mem heJ heB'' rfl
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_4.intro.intro.intro.intr …
      α : Type u_1
      M✝ : Matroid α
      I B✝ X✝ : Set α
      M : Matroid α
      X I' : Set α
      hI'E : HasSubset.Subset I' M.E
      B : Set α
      hB : M.Base B
      hI'B : Disjoint I' B
      hI'X : HasSubset.Subset I' X
      J : Set α
      hJE : HasSubset.Subset J M.E
      B'' : Set α
      hB'' : M.Base B''
      hdj : Disjoint J B''
      hJX : HasSubset.Subset J X
      B₁ : Set α
      hB₁ hB' : M.Base B₁
      hI : M.Basis (SDiff.sdiff B₁ X) (SDiff.sdiff M.E X)
      hIB' : HasSubset.Subset (SDiff.sdiff B₁ X) B₁
      hB'IB : HasSubset.Subset B₁ (Union.union (SDiff.sdiff B₁ X) B)
      hXJ : HasSubset.Subset (Inter.inter (SDiff.sdiff X B₁) M.E) J
      hI' hI'B₁ : HasSubset.Subset (Union.union (Inter.inter B'' X) (SDiff.sdiff B₁  …
      hB₁I : HasSubset.Subset B₁ (Union.union B'' (SDiff.sdiff B₁ X))
      hdj' : Not (Disjoint J B₁)
      e : α
      heJ : Membership.mem J e
      heB' : Membership.mem B₁ e
      heX : Not (Membership.mem X e)
      ⊢ False
    -/
    exact heX (hJX heJ)
    /-
      🎉 no goals
    -/
                      /-
                        α : Type u_1
                        M✝ : Matroid α
                        I B X : Set α
                        M : Matroid α
                        ⊢ ∀ (I : Set α), (fun I => And (HasSubset.Subset I M.E) (Exists fun B => And ( …
                      -/
  subset_ground := by tauto
                      /-
                        🎉 no goals
                      -/


/-- The dual of a matroid; the bases are the complements (w.r.t `M.E`) of the bases of `M`. -/
def dual (M : Matroid α) : Matroid α := M.dualIndepMatroid.matroid


/-- The `✶` symbol, which denotes matroid duality.
  (This is distinct from the usual `*` symbol for multiplication, due to precedence issues. )-/
postfix:max "✶" => Matroid.dual


theorem dual_indep_iff_exists' : (M✶.Indep I) ↔ I ⊆ M.E ∧ (∃ B, M.Base B ∧ Disjoint I B) := Iff.rfl


@[simp] theorem dual_ground : M✶.E = M.E := rfl


@[simp] theorem dual_indep_iff_exists (hI : I ⊆ M.E := by aesop_mat) :
    M✶.Indep I ↔ (∃ B, M.Base B ∧ Disjoint I B) := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    ⊢ Iff (M.dual.Indep I) (Exists fun B => And (M.Base B) (Disjoint I B))
  -/
  rw [dual_indep_iff_exists', and_iff_right hI]
  /-
    🎉 no goals
  -/


theorem dual_dep_iff_forall : (M✶.Dep I) ↔ (∀ B, M.Base B → (I ∩ B).Nonempty) ∧ I ⊆ M.E := by
  simp_rw [dep_iff, dual_indep_iff_exists', dual_ground, and_congr_left_iff, not_and,
    not_exists, not_and, not_disjoint_iff_nonempty_inter, Classical.imp_iff_right_iff,
    iff_true_intro Or.inl]


instance dual_finite [M.Finite] : M✶.Finite :=
  ⟨M.ground_finite⟩


instance dual_nonempty [M.Nonempty] : M✶.Nonempty :=
  ⟨M.ground_nonempty⟩


@[simp] theorem dual_base_iff (hB : B ⊆ M.E := by aesop_mat) : M✶.Base B ↔ M.Base (M.E \ B) := by
  rw [base_compl_iff_maximal_disjoint_base, base_iff_maximal_indep, maximal_subset_iff,
    maximal_subset_iff]
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : autoParam (HasSubset.Subset B M.E) _auto✝
    ⊢ Iff (And (M.dual.Indep B) (∀ ⦃t : Set α⦄, M.dual.Indep t → HasSubset.Subset  …
  -/
  simp [dual_indep_iff_exists', hB]
  /-
    🎉 no goals
  -/


theorem dual_base_iff' : M✶.Base B ↔ M.Base (M.E \ B) ∧ B ⊆ M.E :=
                                  /-
                                    α : Type u_1
                                    M : Matroid α
                                    B : Set α
                                    h : HasSubset.Subset B M.E
                                    ⊢ Iff (M.dual.Base B) (And (M.Base (SDiff.sdiff M.E B)) (HasSubset.Subset B M. …
                                  -/
  (em (B ⊆ M.E)).elim (fun h ↦ by rw [dual_base_iff, and_iff_left h])
                                  /-
                                    🎉 no goals
                                  -/
    (fun h ↦ iff_of_false (h ∘ (fun h' ↦ h'.subset_ground)) (h ∘ And.right))


theorem setOf_dual_base_eq : {B | M✶.Base B} = (fun X ↦ M.E \ X) '' {B | M.Base B} := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq (setOf fun B => M.dual.Base B) (Set.image (fun X => SDiff.sdiff M.E X) (s …
  -/
  ext B
  /-
    case h
    α : Type u_1
    M : Matroid α
    B : Set α
    ⊢ Iff (Membership.mem (setOf fun B => M.dual.Base B) B) (Membership.mem (Set.i …
  -/
  simp only [mem_setOf_eq, mem_image, dual_base_iff']
  refine ⟨fun h ↦ ⟨_, h.1, diff_diff_cancel_left h.2⟩,
    fun ⟨B', hB', h⟩ ↦ ⟨?_,h.symm.trans_subset diff_subset⟩⟩
  /-
    case h
    α : Type u_1
    M : Matroid α
    B : Set α
    x✝ : Exists fun x => And (M.Base x) (Eq (SDiff.sdiff M.E x) B)
    B' : Set α
    hB' : M.Base B'
    h : Eq (SDiff.sdiff M.E B') B
    ⊢ M.Base (SDiff.sdiff M.E B)
  -/
  rwa [← h, diff_diff_cancel_left hB'.subset_ground]
  /-
    🎉 no goals
  -/


@[simp] theorem dual_dual (M : Matroid α) : M✶✶ = M :=
  ext_base rfl (fun B (h : B ⊆ M.E) ↦
       /-
         α : Type u_1
         M : Matroid α
         B : Set α
         h : HasSubset.Subset B M.E
         ⊢ Iff (M.dual.dual.Base B) (M.Base B)
       -/
    by rw [dual_base_iff, dual_base_iff, dual_ground, diff_diff_cancel_left h])
       /-
         🎉 no goals
       -/


theorem dual_involutive : Function.Involutive (dual : Matroid α → Matroid α) := dual_dual


theorem dual_injective : Function.Injective (dual : Matroid α → Matroid α) :=
  dual_involutive.injective


@[simp] theorem dual_inj {M₁ M₂ : Matroid α} : M₁✶ = M₂✶ ↔ M₁ = M₂ :=
  dual_injective.eq_iff


theorem eq_dual_comm {M₁ M₂ : Matroid α} : M₁ = M₂✶ ↔ M₂ = M₁✶ := by
  /-
    α : Type u_1
    M₁ M₂ : Matroid α
    ⊢ Iff (Eq M₁ M₂.dual) (Eq M₂ M₁.dual)
  -/
  rw [← dual_inj, dual_dual, eq_comm]
  /-
    🎉 no goals
  -/


theorem eq_dual_iff_dual_eq {M₁ M₂ : Matroid α} : M₁ = M₂✶ ↔ M₁✶ = M₂ :=
  dual_involutive.eq_iff.symm


theorem Base.compl_base_of_dual (h : M✶.Base B) : M.Base (M.E \ B) :=
  (dual_base_iff'.1 h).1


theorem Base.compl_base_dual (h : M.Base B) : M✶.Base (M.E \ B) := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    h : M.Base B
    ⊢ M.dual.Base (SDiff.sdiff M.E B)
  -/
  rwa [dual_base_iff, diff_diff_cancel_left h.subset_ground]
  /-
    🎉 no goals
  -/


theorem Base.compl_inter_basis_of_inter_basis (hB : M.Base B) (hBX : M.Basis (B ∩ X) X) :
    M✶.Basis ((M.E \ B) ∩ (M.E \ X)) (M.E \ X) := by
  /-
    α : Type u_1
    M : Matroid α
    B X : Set α
    hB : M.Base B
    hBX : M.Basis (Inter.inter B X) X
    ⊢ M.dual.Basis (Inter.inter (SDiff.sdiff M.E B) (SDiff.sdiff M.E X)) (SDiff.sd …
  -/
  refine Indep.basis_of_forall_insert ?_ inter_subset_right (fun e he ↦ ?_)
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      B X : Set α
      hB : M.Base B
      hBX : M.Basis (Inter.inter B X) X
      ⊢ M.dual.Indep (Inter.inter (SDiff.sdiff M.E B) (SDiff.sdiff M.E X))
    -/
  · rw [dual_indep_iff_exists]
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      B X : Set α
      hB : M.Base B
      hBX : M.Basis (Inter.inter B X) X
      ⊢ Exists fun B_1 => And (M.Base B_1) (Disjoint (Inter.inter (SDiff.sdiff M.E B …
    -/
    exact ⟨B, hB, disjoint_of_subset_left inter_subset_left disjoint_sdiff_left⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    B X : Set α
    hB : M.Base B
    hBX : M.Basis (Inter.inter B X) X
    e : α
    he : Membership.mem (SDiff.sdiff (SDiff.sdiff M.E X) (Inter.inter (SDiff.sdiff …
    ⊢ M.dual.Dep (Insert.insert e (Inter.inter (SDiff.sdiff M.E B) (SDiff.sdiff M. …
  -/
  simp only [diff_inter_self_eq_diff, mem_diff, not_and, not_not, imp_iff_right he.1.1] at he
  simp_rw [dual_dep_iff_forall, insert_subset_iff, and_iff_right he.1.1,
    and_iff_left (inter_subset_left.trans diff_subset)]
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    B X : Set α
    hB : M.Base B
    hBX : M.Basis (Inter.inter B X) X
    e : α
    he : And (And (Membership.mem M.E e) (Not (Membership.mem X e))) (Membership.m …
    ⊢ ∀ (B_1 : Set α), M.Base B_1 → (Inter.inter (Insert.insert e (Inter.inter (SD …
  -/
  refine fun B' hB' ↦ by_contra (fun hem ↦ ?_)
  rw [nonempty_iff_ne_empty, not_ne_iff, ← union_singleton, diff_inter_diff,
   union_inter_distrib_right, union_empty_iff, singleton_inter_eq_empty, diff_eq,
   inter_right_comm, inter_eq_self_of_subset_right hB'.subset_ground, ← diff_eq,
   diff_eq_empty] at hem
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    B X : Set α
    hB : M.Base B
    hBX : M.Basis (Inter.inter B X) X
    e : α
    he : And (And (Membership.mem M.E e) (Not (Membership.mem X e))) (Membership.m …
    B' : Set α
    hB' : M.Base B'
    hem : And (HasSubset.Subset B' (Union.union B X)) (Not (Membership.mem B' e))
    ⊢ False
  -/
  obtain ⟨f, hfb, hBf⟩ := hB.exchange hB' ⟨he.2, hem.2⟩

  have hi : M.Indep (insert f (B ∩ X)) := by
    refine hBf.indep.subset (insert_subset_insert ?_)
    simp_rw [subset_diff, and_iff_right inter_subset_left, disjoint_singleton_right,
      mem_inter_iff, iff_false_intro he.1.2, and_false, not_false_iff]
  /-
    case refine_2.intro.intro
    α : Type u_1
    M : Matroid α
    B X : Set α
    hB : M.Base B
    hBX : M.Basis (Inter.inter B X) X
    e : α
    he : And (And (Membership.mem M.E e) (Not (Membership.mem X e))) (Membership.m …
    B' : Set α
    hB' : M.Base B'
    hem : And (HasSubset.Subset B' (Union.union B X)) (Not (Membership.mem B' e))
    f : α
    hfb : Membership.mem (SDiff.sdiff B' B) f
    hBf : M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    hi : M.Indep (Insert.insert f (Inter.inter B X))
    ⊢ False
  -/
  exact hfb.2 (hBX.mem_of_insert_indep (Or.elim (hem.1 hfb.1) (False.elim ∘ hfb.2) id) hi).1
  /-
    🎉 no goals
  -/


theorem Base.inter_basis_iff_compl_inter_basis_dual (hB : M.Base B) (hX : X ⊆ M.E := by aesop_mat) :
    M.Basis (B ∩ X) X ↔ M✶.Basis ((M.E \ B) ∩ (M.E \ X)) (M.E \ X) := by
  /-
    α : Type u_1
    M : Matroid α
    B X : Set α
    hB : M.Base B
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (M.Basis (Inter.inter B X) X) (M.dual.Basis (Inter.inter (SDiff.sdiff M. …
  -/
  refine ⟨hB.compl_inter_basis_of_inter_basis, fun h ↦ ?_⟩
  simpa [inter_eq_self_of_subset_right hX, inter_eq_self_of_subset_right hB.subset_ground] using
    hB.compl_base_dual.compl_inter_basis_of_inter_basis h


theorem base_iff_dual_base_compl (hB : B ⊆ M.E := by aesop_mat) :
    M.Base B ↔ M✶.Base (M.E \ B) := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : autoParam (HasSubset.Subset B M.E) _auto✝
    ⊢ Iff (M.Base B) (M.dual.Base (SDiff.sdiff M.E B))
  -/
  rw [dual_base_iff, diff_diff_cancel_left hB]
  /-
    🎉 no goals
  -/


theorem ground_not_base (M : Matroid α) [h : RkPos M✶] : ¬M.Base M.E := by
  /-
    α : Type u_1
    M : Matroid α
    h : M.dual.RkPos
    ⊢ Not (M.Base M.E)
  -/
  rwa [rkPos_iff_empty_not_base, dual_base_iff, diff_empty] at h
  /-
    🎉 no goals
  -/


theorem Base.ssubset_ground [h : RkPos M✶] (hB : M.Base B) : B ⊂ M.E :=
                                     /-
                                       α : Type u_1
                                       M : Matroid α
                                       B : Set α
                                       h : M.dual.RkPos
                                       hB : M.Base B
                                       ⊢ Ne B M.E
                                     -/
  hB.subset_ground.ssubset_of_ne (by rintro rfl; exact M.ground_not_base hB)
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem Indep.ssubset_ground [h : RkPos M✶] (hI : M.Indep I) : I ⊂ M.E := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    h : M.dual.RkPos
    hI : M.Indep I
    ⊢ HasSSubset.SSubset I M.E
  -/
  obtain ⟨B, hB⟩ := hI.exists_base_superset; exact hB.2.trans_ssubset hB.1.ssubset_ground
                                             /-
                                               🎉 no goals
                                             -/


/-- A coindependent set of `M` is an independent set of the dual of `M✶`. we give it a separate
  definition to enable dot notation. Which spelling is better depends on context. -/
abbrev Coindep (M : Matroid α) (I : Set α) : Prop := M✶.Indep I


theorem coindep_def : M.Coindep X ↔ M✶.Indep X := Iff.rfl


theorem Coindep.indep (hX : M.Coindep X) : M✶.Indep X :=
  hX


@[simp] theorem dual_coindep_iff : M✶.Coindep X ↔ M.Indep X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    ⊢ Iff (M.dual.Coindep X) (M.Indep X)
  -/
  rw [Coindep, dual_dual]
  /-
    🎉 no goals
  -/


theorem Indep.coindep (hI : M.Indep I) : M✶.Coindep I :=
  dual_coindep_iff.2 hI


theorem coindep_iff_exists' : M.Coindep X ↔ (∃ B, M.Base B ∧ B ⊆ M.E \ X) ∧ X ⊆ M.E := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    ⊢ Iff (M.Coindep X) (And (Exists fun B => And (M.Base B) (HasSubset.Subset B ( …
  -/
  simp_rw [Coindep, dual_indep_iff_exists', and_comm (a := _ ⊆ _), and_congr_left_iff, subset_diff]
  exact fun _ ↦ ⟨fun ⟨B, hB, hXB⟩ ↦ ⟨B, hB, hB.subset_ground, hXB.symm⟩,
    fun ⟨B, hB, _, hBX⟩ ↦ ⟨B, hB, hBX.symm⟩⟩


theorem coindep_iff_exists (hX : X ⊆ M.E := by aesop_mat) :
    M.Coindep X ↔ ∃ B, M.Base B ∧ B ⊆ M.E \ X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (M.Coindep X) (Exists fun B => And (M.Base B) (HasSubset.Subset B (SDiff …
  -/
  rw [coindep_iff_exists', and_iff_left hX]
  /-
    🎉 no goals
  -/


theorem coindep_iff_subset_compl_base : M.Coindep X ↔ ∃ B, M.Base B ∧ X ⊆ M.E \ B := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    ⊢ Iff (M.Coindep X) (Exists fun B => And (M.Base B) (HasSubset.Subset X (SDiff …
  -/
  simp_rw [coindep_iff_exists', subset_diff]
  exact ⟨fun ⟨⟨B, hB, _, hBX⟩, hX⟩ ↦ ⟨B, hB, hX, hBX.symm⟩,
    fun ⟨B, hB, hXE, hXB⟩ ↦ ⟨⟨B, hB, hB.subset_ground,  hXB.symm⟩, hXE⟩⟩


@[aesop unsafe 10% (rule_sets := [Matroid])]
theorem Coindep.subset_ground (hX : M.Coindep X) : X ⊆ M.E :=
  hX.indep.subset_ground


theorem Coindep.exists_base_subset_compl (h : M.Coindep X) : ∃ B, M.Base B ∧ B ⊆ M.E \ X :=
  (coindep_iff_exists h.subset_ground).1 h


theorem Coindep.exists_subset_compl_base (h : M.Coindep X) : ∃ B, M.Base B ∧ X ⊆ M.E \ B :=
  coindep_iff_subset_compl_base.1 h


