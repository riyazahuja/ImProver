open Classical in
/-- `SupSet` structure on a nonempty subset `s` of a preorder with `SupSet`. This definition is
non-canonical (it uses `default s`); it should be used only as here, as an auxiliary instance in the
construction of the `ConditionallyCompleteLinearOrder` structure. -/
noncomputable def subsetSupSet [Inhabited s] : SupSet s where
  sSup t :=
    if ht : t.Nonempty ∧ BddAbove t ∧ sSup ((↑) '' t : Set α) ∈ s
    then ⟨sSup ((↑) '' t : Set α), ht.2.2⟩
    else default


open Classical in
@[simp]
theorem subset_sSup_def [Inhabited s] :
    @sSup s _ = fun t =>
      if ht : t.Nonempty ∧ BddAbove t ∧ sSup ((↑) '' t : Set α) ∈ s
      then ⟨sSup ((↑) '' t : Set α), ht.2.2⟩
      else default :=
  rfl


theorem subset_sSup_of_within [Inhabited s] {t : Set s}
    (h' : t.Nonempty) (h'' : BddAbove t) (h : sSup ((↑) '' t : Set α) ∈ s) :
                                                      /-
                                                        α : Type u_2
                                                        s : Set α
                                                        inst✝² : Preorder α
                                                        inst✝¹ : SupSet α
                                                        inst✝ : Inhabited ↑s
                                                        t : Set ↑s
                                                        h' : t.Nonempty
                                                        h'' : BddAbove t
                                                        h : Membership.mem s (SupSet.sSup (Set.image Subtype.val t))
                                                        ⊢ Eq (SupSet.sSup (Set.image Subtype.val t)) ↑(SupSet.sSup t)
                                                      -/
    sSup ((↑) '' t : Set α) = (@sSup s _ t : α) := by simp [dif_pos, h, h', h'']
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem subset_sSup_emptyset [Inhabited s] :
    sSup (∅ : Set s) = default := by
  /-
    α : Type u_2
    s : Set α
    inst✝² : Preorder α
    inst✝¹ : SupSet α
    inst✝ : Inhabited ↑s
    ⊢ Eq (SupSet.sSup EmptyCollection.emptyCollection) Inhabited.default
  -/
  simp [sSup]
  /-
    🎉 no goals
  -/


theorem subset_sSup_of_not_bddAbove [Inhabited s] {t : Set s} (ht : ¬BddAbove t) :
    sSup t = default := by
  /-
    α : Type u_2
    s : Set α
    inst✝² : Preorder α
    inst✝¹ : SupSet α
    inst✝ : Inhabited ↑s
    t : Set ↑s
    ht : Not (BddAbove t)
    ⊢ Eq (SupSet.sSup t) Inhabited.default
  -/
  simp [sSup, ht]
  /-
    🎉 no goals
  -/


open Classical in
/-- `InfSet` structure on a nonempty subset `s` of a preorder with `InfSet`. This definition is
non-canonical (it uses `default s`); it should be used only as here, as an auxiliary instance in the
construction of the `ConditionallyCompleteLinearOrder` structure. -/
noncomputable def subsetInfSet [Inhabited s] : InfSet s where
  sInf t :=
    if ht : t.Nonempty ∧ BddBelow t ∧ sInf ((↑) '' t : Set α) ∈ s
    then ⟨sInf ((↑) '' t : Set α), ht.2.2⟩
    else default


open Classical in
@[simp]
theorem subset_sInf_def [Inhabited s] :
    @sInf s _ = fun t =>
      if ht : t.Nonempty ∧ BddBelow t ∧ sInf ((↑) '' t : Set α) ∈ s
      then ⟨sInf ((↑) '' t : Set α), ht.2.2⟩ else
      default :=
  rfl


theorem subset_sInf_of_within [Inhabited s] {t : Set s}
    (h' : t.Nonempty) (h'' : BddBelow t) (h : sInf ((↑) '' t : Set α) ∈ s) :
                                                      /-
                                                        α : Type u_2
                                                        s : Set α
                                                        inst✝² : Preorder α
                                                        inst✝¹ : InfSet α
                                                        inst✝ : Inhabited ↑s
                                                        t : Set ↑s
                                                        h' : t.Nonempty
                                                        h'' : BddBelow t
                                                        h : Membership.mem s (InfSet.sInf (Set.image Subtype.val t))
                                                        ⊢ Eq (InfSet.sInf (Set.image Subtype.val t)) ↑(InfSet.sInf t)
                                                      -/
    sInf ((↑) '' t : Set α) = (@sInf s _ t : α) := by simp [dif_pos, h, h', h'']
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem subset_sInf_emptyset [Inhabited s] :
    sInf (∅ : Set s) = default := by
  /-
    α : Type u_2
    s : Set α
    inst✝² : Preorder α
    inst✝¹ : InfSet α
    inst✝ : Inhabited ↑s
    ⊢ Eq (InfSet.sInf EmptyCollection.emptyCollection) Inhabited.default
  -/
  simp [sInf]
  /-
    🎉 no goals
  -/


theorem subset_sInf_of_not_bddBelow [Inhabited s] {t : Set s} (ht : ¬BddBelow t) :
    sInf t = default := by
  /-
    α : Type u_2
    s : Set α
    inst✝² : Preorder α
    inst✝¹ : InfSet α
    inst✝ : Inhabited ↑s
    t : Set ↑s
    ht : Not (BddBelow t)
    ⊢ Eq (InfSet.sInf t) Inhabited.default
  -/
  simp [sInf, ht]
  /-
    🎉 no goals
  -/


/-- For a nonempty subset of a conditionally complete linear order to be a conditionally complete
linear order, it suffices that it contain the `sSup` of all its nonempty bounded-above subsets, and
the `sInf` of all its nonempty bounded-below subsets.
See note [reducible non-instances]. -/
noncomputable abbrev subsetConditionallyCompleteLinearOrder [Inhabited s]
    (h_Sup : ∀ {t : Set s} (_ : t.Nonempty) (_h_bdd : BddAbove t), sSup ((↑) '' t : Set α) ∈ s)
    (h_Inf : ∀ {t : Set s} (_ : t.Nonempty) (_h_bdd : BddBelow t), sInf ((↑) '' t : Set α) ∈ s) :
    ConditionallyCompleteLinearOrder s :=
  { subsetSupSet s, subsetInfSet s, DistribLattice.toLattice, (inferInstance : LinearOrder s) with
    le_csSup := by
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        ⊢ ∀ (s_1 : Set ↑s) (a : ↑s), BddAbove s_1 → Membership.mem s_1 a → LE.le a (Su …
      -/
      rintro t c h_bdd hct
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        c : ↑s
        h_bdd : BddAbove t
        hct : Membership.mem t c
        ⊢ LE.le c (SupSet.sSup t)
      -/
      rw [← Subtype.coe_le_coe, ← subset_sSup_of_within s ⟨c, hct⟩ h_bdd (h_Sup ⟨c, hct⟩ h_bdd)]
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        c : ↑s
        h_bdd : BddAbove t
        hct : Membership.mem t c
        ⊢ LE.le (↑c) (SupSet.sSup (Set.image Subtype.val t))
      -/
      exact (Subtype.mono_coe _).le_csSup_image hct h_bdd
      /-
        🎉 no goals
      -/
    csSup_le := by
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        ⊢ ∀ (s_1 : Set ↑s) (a : ↑s), s_1.Nonempty → Membership.mem (upperBounds s_1) a …
      -/
      rintro t B ht hB
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        B : ↑s
        ht : t.Nonempty
        hB : Membership.mem (upperBounds t) B
        ⊢ LE.le (SupSet.sSup t) B
      -/
      rw [← Subtype.coe_le_coe, ← subset_sSup_of_within s ht ⟨B, hB⟩ (h_Sup ht ⟨B, hB⟩)]
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        B : ↑s
        ht : t.Nonempty
        hB : Membership.mem (upperBounds t) B
        ⊢ LE.le (SupSet.sSup (Set.image Subtype.val t)) ↑B
      -/
      exact (Subtype.mono_coe s).csSup_image_le ht hB
      /-
        🎉 no goals
      -/
    le_csInf := by
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        ⊢ ∀ (s_1 : Set ↑s) (a : ↑s), s_1.Nonempty → Membership.mem (lowerBounds s_1) a …
      -/
      intro t B ht hB
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        B : ↑s
        ht : t.Nonempty
        hB : Membership.mem (lowerBounds t) B
        ⊢ LE.le B (InfSet.sInf t)
      -/
      rw [← Subtype.coe_le_coe, ← subset_sInf_of_within s ht ⟨B, hB⟩ (h_Inf ht ⟨B, hB⟩)]
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        ⊢ ∀ (s_1 : Set ↑s) (a : ↑s), BddBelow s_1 → Membership.mem s_1 a → LE.le (InfS …
      -/
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        B : ↑s
        ht : t.Nonempty
        hB : Membership.mem (lowerBounds t) B
        ⊢ LE.le (↑B) (InfSet.sInf (Set.image Subtype.val t))
      -/
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        c : ↑s
        h_bdd : BddBelow t
        hct : Membership.mem t c
        ⊢ LE.le (InfSet.sInf t) c
      -/
      exact (Subtype.mono_coe s).le_csInf_image ht hB
      /-
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLinearOrder α
        inst✝ : Inhabited ↑s
        h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
        h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
        t : Set ↑s
        c : ↑s
        h_bdd : BddBelow t
        hct : Membership.mem t c
        ⊢ LE.le (InfSet.sInf (Set.image Subtype.val t)) ↑c
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    csInf_le := by
      rintro t c h_bdd hct
      rw [← Subtype.coe_le_coe, ← subset_sInf_of_within s ⟨c, hct⟩ h_bdd (h_Inf ⟨c, hct⟩ h_bdd)]
      exact (Subtype.mono_coe s).csInf_image_le hct h_bdd
                                           /-
                                             ι : Sort u_1
                                             α : Type u_2
                                             s : Set α
                                             inst✝¹ : ConditionallyCompleteLinearOrder α
                                             inst✝ : Inhabited ↑s
                                             h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
                                             h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
                                             t : Set ↑s
                                             ht : Not (BddAbove t)
                                             ⊢ Eq (SupSet.sSup t) (SupSet.sSup EmptyCollection.emptyCollection)
                                           -/
    csSup_of_not_bddAbove := fun t ht ↦ by simp [ht]
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             ι : Sort u_1
                                             α : Type u_2
                                             s : Set α
                                             inst✝¹ : ConditionallyCompleteLinearOrder α
                                             inst✝ : Inhabited ↑s
                                             h_Sup : ∀ {t : Set ↑s}, t.Nonempty → BddAbove t → Membership.mem s (SupSet.sSu …
                                             h_Inf : ∀ {t : Set ↑s}, t.Nonempty → BddBelow t → Membership.mem s (InfSet.sIn …
                                             t : Set ↑s
                                             ht : Not (BddBelow t)
                                             ⊢ Eq (InfSet.sInf t) (InfSet.sInf EmptyCollection.emptyCollection)
                                           -/
    csInf_of_not_bddBelow := fun t ht ↦ by simp [ht] }
                                           /-
                                             🎉 no goals
                                           -/


/-- The `sSup` function on a nonempty `OrdConnected` set `s` in a conditionally complete linear
order takes values within `s`, for all nonempty bounded-above subsets of `s`. -/
theorem sSup_within_of_ordConnected {s : Set α} [hs : OrdConnected s] ⦃t : Set s⦄ (ht : t.Nonempty)
    (h_bdd : BddAbove t) : sSup ((↑) '' t : Set α) ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hs : s.OrdConnected
    t : Set ↑s
    ht : t.Nonempty
    h_bdd : BddAbove t
    ⊢ Membership.mem s (SupSet.sSup (Set.image Subtype.val t))
  -/
  obtain ⟨c, hct⟩ : ∃ c, c ∈ t := ht
  /-
    case intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hs : s.OrdConnected
    t : Set ↑s
    h_bdd : BddAbove t
    c : ↑s
    hct : Membership.mem t c
    ⊢ Membership.mem s (SupSet.sSup (Set.image Subtype.val t))
  -/
  obtain ⟨B, hB⟩ : ∃ B, B ∈ upperBounds t := h_bdd
  /-
    case intro.intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hs : s.OrdConnected
    t : Set ↑s
    c : ↑s
    hct : Membership.mem t c
    B : ↑s
    hB : Membership.mem (upperBounds t) B
    ⊢ Membership.mem s (SupSet.sSup (Set.image Subtype.val t))
  -/
  refine hs.out c.2 B.2 ⟨?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      s : Set α
      hs : s.OrdConnected
      t : Set ↑s
      c : ↑s
      hct : Membership.mem t c
      B : ↑s
      hB : Membership.mem (upperBounds t) B
      ⊢ LE.le (↑c) (SupSet.sSup (Set.image Subtype.val t))
    -/
  · exact (Subtype.mono_coe s).le_csSup_image hct ⟨B, hB⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      s : Set α
      hs : s.OrdConnected
      t : Set ↑s
      c : ↑s
      hct : Membership.mem t c
      B : ↑s
      hB : Membership.mem (upperBounds t) B
      ⊢ LE.le (SupSet.sSup (Set.image Subtype.val t)) ↑B
    -/
  · exact (Subtype.mono_coe s).csSup_image_le ⟨c, hct⟩ hB
    /-
      🎉 no goals
    -/


/-- The `sInf` function on a nonempty `OrdConnected` set `s` in a conditionally complete linear
order takes values within `s`, for all nonempty bounded-below subsets of `s`. -/
theorem sInf_within_of_ordConnected {s : Set α} [hs : OrdConnected s] ⦃t : Set s⦄ (ht : t.Nonempty)
    (h_bdd : BddBelow t) : sInf ((↑) '' t : Set α) ∈ s := by
  /-
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hs : s.OrdConnected
    t : Set ↑s
    ht : t.Nonempty
    h_bdd : BddBelow t
    ⊢ Membership.mem s (InfSet.sInf (Set.image Subtype.val t))
  -/
  obtain ⟨c, hct⟩ : ∃ c, c ∈ t := ht
  /-
    case intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hs : s.OrdConnected
    t : Set ↑s
    h_bdd : BddBelow t
    c : ↑s
    hct : Membership.mem t c
    ⊢ Membership.mem s (InfSet.sInf (Set.image Subtype.val t))
  -/
  obtain ⟨B, hB⟩ : ∃ B, B ∈ lowerBounds t := h_bdd
  /-
    case intro.intro
    α : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder α
    s : Set α
    hs : s.OrdConnected
    t : Set ↑s
    c : ↑s
    hct : Membership.mem t c
    B : ↑s
    hB : Membership.mem (lowerBounds t) B
    ⊢ Membership.mem s (InfSet.sInf (Set.image Subtype.val t))
  -/
  refine hs.out B.2 c.2 ⟨?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      s : Set α
      hs : s.OrdConnected
      t : Set ↑s
      c : ↑s
      hct : Membership.mem t c
      B : ↑s
      hB : Membership.mem (lowerBounds t) B
      ⊢ LE.le (↑B) (InfSet.sInf (Set.image Subtype.val t))
    -/
  · exact (Subtype.mono_coe s).le_csInf_image ⟨c, hct⟩ hB
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder α
      s : Set α
      hs : s.OrdConnected
      t : Set ↑s
      c : ↑s
      hct : Membership.mem t c
      B : ↑s
      hB : Membership.mem (lowerBounds t) B
      ⊢ LE.le (InfSet.sInf (Set.image Subtype.val t)) ↑c
    -/
  · exact (Subtype.mono_coe s).csInf_image_le hct ⟨B, hB⟩
    /-
      🎉 no goals
    -/


/-- A nonempty `OrdConnected` set in a conditionally complete linear order is naturally a
conditionally complete linear order. -/
noncomputable instance ordConnectedSubsetConditionallyCompleteLinearOrder [Inhabited s]
    [OrdConnected s] : ConditionallyCompleteLinearOrder s :=
  subsetConditionallyCompleteLinearOrder s
    (fun h => sSup_within_of_ordConnected h)
    (fun h => sInf_within_of_ordConnected h)


open Classical in
/-- Complete lattice structure on `Set.Icc` -/
noncomputable instance Set.Icc.completeLattice [ConditionallyCompleteLattice α]
    {a b : α} [Fact (a ≤ b)] : CompleteLattice (Set.Icc a b) where
  __ := (inferInstance : BoundedOrder ↑(Icc a b))
  sSup S := if hS : S = ∅ then ⟨a, le_rfl, Fact.out⟩ else ⟨sSup ((↑) '' S), by
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      hS : Not (Eq S EmptyCollection.emptyCollection)
      ⊢ Membership.mem (Set.Icc a b) (SupSet.sSup (Set.image Subtype.val S))
    -/
    rw [← Set.not_nonempty_iff_eq_empty, not_not] at hS
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      hS : S.Nonempty
      ⊢ Membership.mem (Set.Icc a b) (SupSet.sSup (Set.image Subtype.val S))
    -/
    refine ⟨?_, csSup_le (hS.image Subtype.val) (fun _ ⟨c, _, hc⟩ ↦ hc ▸ c.2.2)⟩
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      hS : S.Nonempty
      ⊢ LE.le a (SupSet.sSup (Set.image Subtype.val S))
    -/
    obtain ⟨c, hc⟩ := hS
    /-
      case intro
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      c : ↑(Set.Icc a b)
      hc : Membership.mem S c
      ⊢ LE.le a (SupSet.sSup (Set.image Subtype.val S))
    -/
    exact c.2.1.trans (le_csSup ⟨b, fun _ ⟨d, _, hd⟩ ↦ hd ▸ d.2.2⟩ ⟨c, hc, rfl⟩)⟩
    /-
      🎉 no goals
    -/
  le_sSup S c hc := by
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      c : ↑(Set.Icc a b)
      hc : Membership.mem S c
      ⊢ LE.le c (SupSet.sSup S)
    -/
    by_cases hS : S = ∅ <;> simp only [hS, dite_true, dite_false]
      /-
        case pos
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLattice α
        a b : α
        inst✝ : Fact (LE.le a b)
        S : Set ↑(Set.Icc a b)
        c : ↑(Set.Icc a b)
        hc : Membership.mem S c
        hS : Eq S EmptyCollection.emptyCollection
        ⊢ LE.le c ⟨a, ⋯⟩
      -/
    · simp [hS] at hc
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLattice α
        a b : α
        inst✝ : Fact (LE.le a b)
        S : Set ↑(Set.Icc a b)
        c : ↑(Set.Icc a b)
        hc : Membership.mem S c
        hS : Not (Eq S EmptyCollection.emptyCollection)
        ⊢ LE.le c ⟨SupSet.sSup (Set.image Subtype.val S), ⋯⟩
      -/
    · exact le_csSup ⟨b, fun _ ⟨d, _, hd⟩ ↦ hd ▸ d.2.2⟩ ⟨c, hc, rfl⟩
      /-
        🎉 no goals
      -/
  sSup_le S c hc := by
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      c : ↑(Set.Icc a b)
      hc : ∀ (b_1 : ↑(Set.Icc a b)), Membership.mem S b_1 → LE.le b_1 c
      ⊢ LE.le (SupSet.sSup S) c
    -/
    by_cases hS : S = ∅ <;> simp only [hS, dite_true, dite_false]
      /-
        case pos
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLattice α
        a b : α
        inst✝ : Fact (LE.le a b)
        S : Set ↑(Set.Icc a b)
        c : ↑(Set.Icc a b)
        hc : ∀ (b_1 : ↑(Set.Icc a b)), Membership.mem S b_1 → LE.le b_1 c
        hS : Eq S EmptyCollection.emptyCollection
        ⊢ LE.le ⟨a, ⋯⟩ c
      -/
    · exact c.2.1
      /-
        🎉 no goals
      -/
    · exact csSup_le ((Set.nonempty_iff_ne_empty.mpr hS).image Subtype.val)
        (fun _ ⟨d, h, hd⟩ ↦ hd ▸ hc d h)
  sInf S := if hS : S = ∅ then ⟨b, Fact.out, le_rfl⟩ else ⟨sInf ((↑) '' S), by
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      hS : Not (Eq S EmptyCollection.emptyCollection)
      ⊢ Membership.mem (Set.Icc a b) (InfSet.sInf (Set.image Subtype.val S))
    -/
    rw [← Set.not_nonempty_iff_eq_empty, not_not] at hS
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      hS : S.Nonempty
      ⊢ Membership.mem (Set.Icc a b) (InfSet.sInf (Set.image Subtype.val S))
    -/
    refine ⟨le_csInf (hS.image Subtype.val) (fun _ ⟨c, _, hc⟩ ↦ hc ▸ c.2.1), ?_⟩
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      hS : S.Nonempty
      ⊢ LE.le (InfSet.sInf (Set.image Subtype.val S)) b
    -/
    obtain ⟨c, hc⟩ := hS
    /-
      case intro
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      c : ↑(Set.Icc a b)
      hc : Membership.mem S c
      ⊢ LE.le (InfSet.sInf (Set.image Subtype.val S)) b
    -/
    exact le_trans (csInf_le ⟨a, fun _ ⟨d, _, hd⟩ ↦ hd ▸ d.2.1⟩ ⟨c, hc, rfl⟩) c.2.2⟩
    /-
      🎉 no goals
    -/
  sInf_le S c hc := by
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      c : ↑(Set.Icc a b)
      hc : Membership.mem S c
      ⊢ LE.le (InfSet.sInf S) c
    -/
    by_cases hS : S = ∅ <;> simp only [hS, dite_true, dite_false]
      /-
        case pos
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLattice α
        a b : α
        inst✝ : Fact (LE.le a b)
        S : Set ↑(Set.Icc a b)
        c : ↑(Set.Icc a b)
        hc : Membership.mem S c
        hS : Eq S EmptyCollection.emptyCollection
        ⊢ LE.le ⟨b, ⋯⟩ c
      -/
    · simp [hS] at hc
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLattice α
        a b : α
        inst✝ : Fact (LE.le a b)
        S : Set ↑(Set.Icc a b)
        c : ↑(Set.Icc a b)
        hc : Membership.mem S c
        hS : Not (Eq S EmptyCollection.emptyCollection)
        ⊢ LE.le ⟨InfSet.sInf (Set.image Subtype.val S), ⋯⟩ c
      -/
    · exact csInf_le ⟨a, fun _ ⟨d, _, hd⟩ ↦ hd ▸ d.2.1⟩ ⟨c, hc, rfl⟩
      /-
        🎉 no goals
      -/
  le_sInf S c hc := by
    /-
      ι : Sort u_1
      α : Type u_2
      s : Set α
      inst✝¹ : ConditionallyCompleteLattice α
      a b : α
      inst✝ : Fact (LE.le a b)
      S : Set ↑(Set.Icc a b)
      c : ↑(Set.Icc a b)
      hc : ∀ (b_1 : ↑(Set.Icc a b)), Membership.mem S b_1 → LE.le c b_1
      ⊢ LE.le c (InfSet.sInf S)
    -/
    by_cases hS : S = ∅ <;> simp only [hS, dite_true, dite_false]
      /-
        case pos
        ι : Sort u_1
        α : Type u_2
        s : Set α
        inst✝¹ : ConditionallyCompleteLattice α
        a b : α
        inst✝ : Fact (LE.le a b)
        S : Set ↑(Set.Icc a b)
        c : ↑(Set.Icc a b)
        hc : ∀ (b_1 : ↑(Set.Icc a b)), Membership.mem S b_1 → LE.le c b_1
        hS : Eq S EmptyCollection.emptyCollection
        ⊢ LE.le c ⟨b, ⋯⟩
      -/
    · exact c.2.2
      /-
        🎉 no goals
      -/
    · exact le_csInf ((Set.nonempty_iff_ne_empty.mpr hS).image Subtype.val)
        (fun _ ⟨d, h, hd⟩ ↦ hd ▸ hc d h)


/-- Complete linear order structure on `Set.Icc` -/
noncomputable instance [ConditionallyCompleteLinearOrder α] {a b : α} [Fact (a ≤ b)] :
    CompleteLinearOrder (Set.Icc a b) :=
  { Set.Icc.completeLattice, Subtype.instLinearOrder _, LinearOrder.toBiheytingAlgebra with }


lemma Set.Icc.coe_sSup [ConditionallyCompleteLattice α] {a b : α} (h : a ≤ b)
    {S : Set (Set.Icc a b)} (hS : S.Nonempty) : have : Fact (a ≤ b) := ⟨h⟩
    ↑(sSup S) = sSup ((↑) '' S : Set α) :=
  congrArg Subtype.val (dif_neg hS.ne_empty)


lemma Set.Icc.coe_sInf [ConditionallyCompleteLattice α] {a b : α} (h : a ≤ b)
    {S : Set (Set.Icc a b)} (hS : S.Nonempty) : have : Fact (a ≤ b) := ⟨h⟩
    ↑(sInf S) = sInf ((↑) '' S : Set α) :=
  congrArg Subtype.val (dif_neg hS.ne_empty)


lemma Set.Icc.coe_iSup [ConditionallyCompleteLattice α] {a b : α} (h : a ≤ b)
    [Nonempty ι] {S : ι → Set.Icc a b} : have : Fact (a ≤ b) := ⟨h⟩
    ↑(iSup S) = (⨆ i, S i : α) :=
  (Set.Icc.coe_sSup h (range_nonempty S)).trans (congrArg sSup (range_comp Subtype.val S).symm)


lemma Set.Icc.coe_iInf [ConditionallyCompleteLattice α] {a b : α} (h : a ≤ b)
    [Nonempty ι] {S : ι → Set.Icc a b} : have : Fact (a ≤ b) := ⟨h⟩
    ↑(iInf S) = (⨅ i, S i : α) :=
  (Set.Icc.coe_sInf h (range_nonempty S)).trans (congrArg sInf (range_comp Subtype.val S).symm)


instance instCompleteLattice : CompleteLattice (Iic a) where
                                 /-
                                   ι : Sort u_1
                                   α : Type u_2
                                   s : Set α
                                   inst✝ : CompleteLattice α
                                   a : α
                                   S : Set ↑(Set.Iic a)
                                   ⊢ Membership.mem (Set.Iic a) (SupSet.sSup (Set.image Subtype.val S))
                                 -/
  sSup S := ⟨sSup ((↑) '' S), by simpa using fun b hb _ ↦ hb⟩
                                 /-
                                   🎉 no goals
                                 -/
                                     /-
                                       ι : Sort u_1
                                       α : Type u_2
                                       s : Set α
                                       inst✝ : CompleteLattice α
                                       a : α
                                       S : Set ↑(Set.Iic a)
                                       ⊢ Membership.mem (Set.Iic a) (Min.min a (InfSet.sInf (Set.image Subtype.val S)))
                                     -/
  sInf S := ⟨a ⊓ sInf ((↑) '' S), by simp⟩
                                     /-
                                       🎉 no goals
                                     -/
  le_sSup _ _ hb := le_sSup <| mem_image_of_mem Subtype.val hb
  sSup_le _ _ hb := sSup_le <| fun _ ⟨c, hc, hc'⟩ ↦ hc' ▸ hb c hc
  sInf_le _ _ hb := inf_le_of_right_le <| sInf_le <| mem_image_of_mem Subtype.val hb
  le_sInf _ b hb := le_inf_iff.mpr ⟨b.property, le_sInf fun _ ⟨d, hd, hd'⟩  ↦ hd' ▸ hb d hd⟩
               /-
                 ι : Sort u_1
                 α : Type u_2
                 s : Set α
                 inst✝ : CompleteLattice α
                 a : α
                 ⊢ ∀ (x : ↑(Set.Iic a)), LE.le x Top.top
               -/
  le_top := by simp
               /-
                 🎉 no goals
               -/
               /-
                 ι : Sort u_1
                 α : Type u_2
                 s : Set α
                 inst✝ : CompleteLattice α
                 a : α
                 ⊢ ∀ (x : ↑(Set.Iic a)), LE.le Bot.bot x
               -/
  bot_le := by simp
               /-
                 🎉 no goals
               -/


@[simp] theorem coe_sSup : (↑(sSup S) : α) = sSup ((↑) '' S) := rfl


@[simp] theorem coe_iSup : (↑(⨆ i, f i) : α) = ⨆ i, (f i : α) := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    f : ι → ↑(Set.Iic a)
    ⊢ Eq (↑(iSup fun i => f i)) (iSup fun i => ↑(f i))
  -/
  rw [iSup, coe_sSup]; congr; ext; simp
                                   /-
                                     🎉 no goals
                                   -/


                                                                                       /-
                                                                                         ι : Sort u_1
                                                                                         α : Type u_2
                                                                                         inst✝ : CompleteLattice α
                                                                                         a : α
                                                                                         f : ι → ↑(Set.Iic a)
                                                                                         p : ι → Prop
                                                                                         ⊢ Eq (↑(iSup fun i => iSup fun x => f i)) (iSup fun i => iSup fun x => ↑(f i))
                                                                                       -/
theorem coe_biSup : (↑(⨆ i, ⨆ (_ : p i), f i) : α) = ⨆ i, ⨆ (_ : p i), (f i : α) := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp] theorem coe_sInf : (↑(sInf S) : α) = a ⊓ sInf ((↑) '' S) := rfl


@[simp] theorem coe_iInf : (↑(⨅ i, f i) : α) = a ⊓ ⨅ i, (f i : α) := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    f : ι → ↑(Set.Iic a)
    ⊢ Eq (↑(iInf fun i => f i)) (Min.min a (iInf fun i => ↑(f i)))
  -/
  rw [iInf, coe_sInf]; congr; ext; simp
                                   /-
                                     🎉 no goals
                                   -/


theorem coe_biInf : (↑(⨅ i, ⨅ (_ : p i), f i) : α) = a ⊓ ⨅ i, ⨅ (_ : p i), (f i : α) := by
  /-
    ι : Sort u_1
    α : Type u_2
    inst✝ : CompleteLattice α
    a : α
    f : ι → ↑(Set.Iic a)
    p : ι → Prop
    ⊢ Eq (↑(iInf fun i => iInf fun x => f i)) (Min.min a (iInf fun i => iInf fun x …
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      α : Type u_2
      inst✝ : CompleteLattice α
      a : α
      f : ι → ↑(Set.Iic a)
      p : ι → Prop
      h✝ : IsEmpty ι
      ⊢ Eq (↑(iInf fun i => iInf fun x => f i)) (Min.min a (iInf fun i => iInf fun x …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      α : Type u_2
      inst✝ : CompleteLattice α
      a : α
      f : ι → ↑(Set.Iic a)
      p : ι → Prop
      h✝ : Nonempty ι
      ⊢ Eq (↑(iInf fun i => iInf fun x => f i)) (Min.min a (iInf fun i => iInf fun x …
    -/
  · simp_rw [coe_iInf, ← inf_iInf, ← inf_assoc, inf_idem]
    /-
      🎉 no goals
    -/



