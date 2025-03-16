/-- Hitting time: given a stochastic process `u` and a set `s`, `hitting u s n m` is the first time
`u` is in `s` after time `n` and before time `m` (if `u` does not hit `s` after time `n` and
before `m` then the hitting time is simply `m`).

The hitting time is a stopping time if the process is adapted and discrete. -/
noncomputable def hitting [Preorder ι] [InfSet ι] (u : ι → Ω → β) (s : Set β) (n m : ι) : Ω → ι :=
  fun x => if ∃ j ∈ Set.Icc n m, u j x ∈ s then sInf (Set.Icc n m ∩ {i : ι | u i x ∈ s}) else m


theorem hitting_def [Preorder ι] [InfSet ι] (u : ι → Ω → β) (s : Set β) (n m : ι) :
    hitting u s n m =
    fun x => if ∃ j ∈ Set.Icc n m, u j x ∈ s then sInf (Set.Icc n m ∩ {i : ι | u i x ∈ s}) else m :=
  rfl


/-- This lemma is strictly weaker than `hitting_of_le`. -/
theorem hitting_of_lt {m : ι} (h : m < n) : hitting u s n m ω = m := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h : LT.lt m n
    ⊢ Eq (MeasureTheory.hitting u s n m ω) m
  -/
  simp_rw [hitting]
  have h_not : ¬∃ (j : ι) (_ : j ∈ Set.Icc n m), u j ω ∈ s := by
    push_neg
    intro j
    rw [Set.Icc_eq_empty_of_lt h]
    simp only [Set.mem_empty_iff_false, IsEmpty.forall_iff]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h : LT.lt m n
    h_not : Not (Exists fun j => Exists fun x => Membership.mem s (u j ω))
    ⊢ Eq (ite (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
  -/
  simp only [exists_prop] at h_not
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h : LT.lt m n
    h_not : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership. …
    ⊢ Eq (ite (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
  -/
  simp only [h_not, if_false]
  /-
    🎉 no goals
  -/


theorem hitting_le {m : ι} (ω : Ω) : hitting u s n m ω ≤ m := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    ω : Ω
    ⊢ LE.le (MeasureTheory.hitting u s n m ω) m
  -/
  simp only [hitting]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    ω : Ω
    ⊢ LE.le (ite (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership …
  -/
  split_ifs with h
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n m : ι
      ω : Ω
      h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
      ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem …
    -/
  · obtain ⟨j, hj₁, hj₂⟩ := h
    /-
      case pos.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n m : ι
      ω : Ω
      j : ι
      hj₁ : Membership.mem (Set.Icc n m) j
      hj₂ : Membership.mem s (u j ω)
      ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem …
    -/
    change j ∈ {i | u i ω ∈ s} at hj₂
    /-
      case pos.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n m : ι
      ω : Ω
      j : ι
      hj₁ : Membership.mem (Set.Icc n m) j
      hj₂ : Membership.mem (setOf fun i => Membership.mem s (u i ω)) j
      ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem …
    -/
    exact (csInf_le (BddBelow.inter_of_left bddBelow_Icc) (Set.mem_inter hj₁ hj₂)).trans hj₁.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n m : ι
      ω : Ω
      h : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem  …
      ⊢ LE.le m m
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


theorem not_mem_of_lt_hitting {m k : ι} (hk₁ : k < hitting u s n m ω) (hk₂ : n ≤ k) :
    u k ω ∉ s := by
  classical
  intro h
  have hexists : ∃ j ∈ Set.Icc n m, u j ω ∈ s := ⟨k, ⟨hk₂, le_trans hk₁.le <| hitting_le _⟩, h⟩
  refine not_le.2 hk₁ ?_
  simp_rw [hitting, if_pos hexists]
  exact csInf_le bddBelow_Icc.inter_of_left ⟨⟨hk₂, le_trans hk₁.le <| hitting_le _⟩, h⟩


theorem hitting_eq_end_iff {m : ι} : hitting u s n m ω = m ↔
    (∃ j ∈ Set.Icc n m, u j ω ∈ s) → sInf (Set.Icc n m ∩ {i : ι | u i ω ∈ s}) = m := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    ⊢ Iff (Eq (MeasureTheory.hitting u s n m ω) m) ((Exists fun j => And (Membersh …
  -/
  rw [hitting, ite_eq_right_iff]
  /-
    🎉 no goals
  -/


theorem hitting_of_le {m : ι} (hmn : m ≤ n) : hitting u s n m ω = m := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    hmn : LE.le m n
    ⊢ Eq (MeasureTheory.hitting u s n m ω) m
  -/
  obtain rfl | h := le_iff_eq_or_lt.1 hmn
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      ω : Ω
      m : ι
      hmn : LE.le m m
      ⊢ Eq (MeasureTheory.hitting u s m m ω) m
    -/
  · rw [hitting, ite_eq_right_iff, forall_exists_index]
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      ω : Ω
      m : ι
      hmn : LE.le m m
      ⊢ ∀ (x : ι), And (Membership.mem (Set.Icc m m) x) (Membership.mem s (u x ω)) → …
    -/
    conv => intro; rw [Set.mem_Icc, Set.Icc_self, and_imp, and_imp]
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      ω : Ω
      m : ι
      hmn : LE.le m m
      ⊢ ∀ (x : ι), LE.le m x → LE.le x m → Membership.mem s (u x ω) → Eq (InfSet.sIn …
    -/
    intro i hi₁ hi₂ hi
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      ω : Ω
      m : ι
      hmn : LE.le m m
      i : ι
      hi₁ : LE.le m i
      hi₂ : LE.le i m
      hi : Membership.mem s (u i ω)
      ⊢ Eq (InfSet.sInf (Inter.inter (Singleton.singleton m) (setOf fun i => Members …
    -/
    rw [Set.inter_eq_left.2, csInf_singleton]
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      ω : Ω
      m : ι
      hmn : LE.le m m
      i : ι
      hi₁ : LE.le m i
      hi₂ : LE.le i m
      hi : Membership.mem s (u i ω)
      ⊢ HasSubset.Subset (Singleton.singleton m) (setOf fun i => Membership.mem s (u …
    -/
    exact Set.singleton_subset_iff.2 (le_antisymm hi₂ hi₁ ▸ hi)
    /-
      🎉 no goals
    -/
    /-
      case inr
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      m : ι
      hmn : LE.le m n
      h : LT.lt m n
      ⊢ Eq (MeasureTheory.hitting u s n m ω) m
    -/
  · exact hitting_of_lt h
    /-
      🎉 no goals
    -/


theorem le_hitting {m : ι} (hnm : n ≤ m) (ω : Ω) : n ≤ hitting u s n m ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    hnm : LE.le n m
    ω : Ω
    ⊢ LE.le n (MeasureTheory.hitting u s n m ω)
  -/
  simp only [hitting]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    hnm : LE.le n m
    ω : Ω
    ⊢ LE.le n (ite (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membersh …
  -/
  split_ifs with h
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n m : ι
      hnm : LE.le n m
      ω : Ω
      h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
      ⊢ LE.le n (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i => Membership.m …
    -/
  · refine le_csInf ?_ fun b hb => ?_
      /-
        case pos.refine_1
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n m : ι
        hnm : LE.le n m
        ω : Ω
        h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
        ⊢ (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem s (u i ω))).Nonempty
      -/
    · obtain ⟨k, hk_Icc, hk_s⟩ := h
      /-
        case pos.refine_1.intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n m : ι
        hnm : LE.le n m
        ω : Ω
        k : ι
        hk_Icc : Membership.mem (Set.Icc n m) k
        hk_s : Membership.mem s (u k ω)
        ⊢ (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem s (u i ω))).Nonempty
      -/
      exact ⟨k, hk_Icc, hk_s⟩
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n m : ι
        hnm : LE.le n m
        ω : Ω
        h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
        b : ι
        hb : Membership.mem (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem  …
        ⊢ LE.le n b
      -/
    · rw [Set.mem_inter_iff] at hb
      /-
        case pos.refine_2
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n m : ι
        hnm : LE.le n m
        ω : Ω
        h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
        b : ι
        hb : And (Membership.mem (Set.Icc n m) b) (Membership.mem (setOf fun i => Memb …
        ⊢ LE.le n b
      -/
      exact hb.1.1
      /-
        🎉 no goals
      -/
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n m : ι
      hnm : LE.le n m
      ω : Ω
      h : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem  …
      ⊢ LE.le n m
    -/
  · exact hnm
    /-
      🎉 no goals
    -/


theorem le_hitting_of_exists {m : ι} (h_exists : ∃ j ∈ Set.Icc n m, u j ω ∈ s) :
    n ≤ hitting u s n m ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    ⊢ LE.le n (MeasureTheory.hitting u s n m ω)
  -/
  refine le_hitting ?_ ω
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    ⊢ LE.le n m
  -/
  by_contra h
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    h : Not (LE.le n m)
    ⊢ False
  -/
  rw [Set.Icc_eq_empty_of_lt (not_le.mp h)] at h_exists
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m : ι
    h_exists : Exists fun j => And (Membership.mem EmptyCollection.emptyCollection …
    h : Not (LE.le n m)
    ⊢ False
  -/
  simp at h_exists
  /-
    🎉 no goals
  -/


theorem hitting_mem_Icc {m : ι} (hnm : n ≤ m) (ω : Ω) : hitting u s n m ω ∈ Set.Icc n m :=
  ⟨le_hitting hnm ω, hitting_le ω⟩


theorem hitting_mem_set [WellFoundedLT ι] {m : ι} (h_exists : ∃ j ∈ Set.Icc n m, u j ω ∈ s) :
    u (hitting u s n m ω) ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    ⊢ Membership.mem s (u (MeasureTheory.hitting u s n m ω) ω)
  -/
  simp_rw [hitting, if_pos h_exists]
  have h_nonempty : (Set.Icc n m ∩ {i : ι | u i ω ∈ s}).Nonempty := by
    obtain ⟨k, hk₁, hk₂⟩ := h_exists
    exact ⟨k, Set.mem_inter hk₁ hk₂⟩
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    h_nonempty : (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem s (u i  …
    ⊢ Membership.mem s (u (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i =>  …
  -/
  have h_mem := csInf_mem h_nonempty
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    h_nonempty : (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem s (u i  …
    h_mem : Membership.mem (Inter.inter (Set.Icc n m) (setOf fun i => Membership.m …
    ⊢ Membership.mem s (u (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i =>  …
  -/
  rw [Set.mem_inter_iff] at h_mem
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    h_nonempty : (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem s (u i  …
    h_mem : And (Membership.mem (Set.Icc n m) (InfSet.sInf (Inter.inter (Set.Icc n …
    ⊢ Membership.mem s (u (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i =>  …
  -/
  exact h_mem.2
  /-
    🎉 no goals
  -/


theorem hitting_mem_set_of_hitting_lt [WellFoundedLT ι] {m : ι} (hl : hitting u s n m ω < m) :
    u (hitting u s n m ω) ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m : ι
    hl : LT.lt (MeasureTheory.hitting u s n m ω) m
    ⊢ Membership.mem s (u (MeasureTheory.hitting u s n m ω) ω)
  -/
  by_cases h : ∃ j ∈ Set.Icc n m, u j ω ∈ s
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      hl : LT.lt (MeasureTheory.hitting u s n m ω) m
      h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
      ⊢ Membership.mem s (u (MeasureTheory.hitting u s n m ω) ω)
    -/
  · exact hitting_mem_set h
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      hl : LT.lt (MeasureTheory.hitting u s n m ω) m
      h : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem  …
      ⊢ Membership.mem s (u (MeasureTheory.hitting u s n m ω) ω)
    -/
  · simp_rw [hitting, if_neg h] at hl
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      h : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem  …
      hl : LT.lt m m
      ⊢ Membership.mem s (u (MeasureTheory.hitting u s n m ω) ω)
    -/
    exact False.elim (hl.ne rfl)
    /-
      🎉 no goals
    -/


theorem hitting_le_of_mem {m : ι} (hin : n ≤ i) (him : i ≤ m) (his : u i ω ∈ s) :
    hitting u s n m ω ≤ i := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n i : ι
    ω : Ω
    m : ι
    hin : LE.le n i
    him : LE.le i m
    his : Membership.mem s (u i ω)
    ⊢ LE.le (MeasureTheory.hitting u s n m ω) i
  -/
  have h_exists : ∃ k ∈ Set.Icc n m, u k ω ∈ s := ⟨i, ⟨hin, him⟩, his⟩
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n i : ι
    ω : Ω
    m : ι
    hin : LE.le n i
    him : LE.le i m
    his : Membership.mem s (u i ω)
    h_exists : Exists fun k => And (Membership.mem (Set.Icc n m) k) (Membership.me …
    ⊢ LE.le (MeasureTheory.hitting u s n m ω) i
  -/
  simp_rw [hitting, if_pos h_exists]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n i : ι
    ω : Ω
    m : ι
    hin : LE.le n i
    him : LE.le i m
    his : Membership.mem s (u i ω)
    h_exists : Exists fun k => And (Membership.mem (Set.Icc n m) k) (Membership.me …
    ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i => Membership.mem …
  -/
  exact csInf_le (BddBelow.inter_of_left bddBelow_Icc) (Set.mem_inter ⟨hin, him⟩ his)
  /-
    🎉 no goals
  -/


theorem hitting_le_iff_of_exists [WellFoundedLT ι] {m : ι}
    (h_exists : ∃ j ∈ Set.Icc n m, u j ω ∈ s) :
    hitting u s n m ω ≤ i ↔ ∃ j ∈ Set.Icc n i, u j ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n i : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m : ι
    h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
    ⊢ Iff (LE.le (MeasureTheory.hitting u s n m ω) i) (Exists fun j => And (Member …
  -/
  constructor <;> intro h'
    /-
      case mp
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n i : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
      h' : LE.le (MeasureTheory.hitting u s n m ω) i
      ⊢ Exists fun j => And (Membership.mem (Set.Icc n i) j) (Membership.mem s (u j  …
    -/
  · exact ⟨hitting u s n m ω, ⟨le_hitting_of_exists h_exists, h'⟩, hitting_mem_set h_exists⟩
    /-
      🎉 no goals
    -/
  · have h'' : ∃ k ∈ Set.Icc n (min m i), u k ω ∈ s := by
      obtain ⟨k₁, hk₁_mem, hk₁_s⟩ := h_exists
      obtain ⟨k₂, hk₂_mem, hk₂_s⟩ := h'
      refine ⟨min k₁ k₂, ⟨le_min hk₁_mem.1 hk₂_mem.1, min_le_min hk₁_mem.2 hk₂_mem.2⟩, ?_⟩
      exact min_rec' (fun j => u j ω ∈ s) hk₁_s hk₂_s
    /-
      case mpr
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n i : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
      h' : Exists fun j => And (Membership.mem (Set.Icc n i) j) (Membership.mem s (u …
      h'' : Exists fun k => And (Membership.mem (Set.Icc n (Min.min m i)) k) (Member …
      ⊢ LE.le (MeasureTheory.hitting u s n m ω) i
    -/
    obtain ⟨k, hk₁, hk₂⟩ := h''
    /-
      case mpr.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n i : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
      h' : Exists fun j => And (Membership.mem (Set.Icc n i) j) (Membership.mem s (u …
      k : ι
      hk₁ : Membership.mem (Set.Icc n (Min.min m i)) k
      hk₂ : Membership.mem s (u k ω)
      ⊢ LE.le (MeasureTheory.hitting u s n m ω) i
    -/
    refine le_trans ?_ (hk₁.2.trans (min_le_right _ _))
    /-
      case mpr.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n i : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m : ι
      h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
      h' : Exists fun j => And (Membership.mem (Set.Icc n i) j) (Membership.mem s (u …
      k : ι
      hk₁ : Membership.mem (Set.Icc n (Min.min m i)) k
      hk₂ : Membership.mem s (u k ω)
      ⊢ LE.le (MeasureTheory.hitting u s n m ω) k
    -/
    exact hitting_le_of_mem hk₁.1 (hk₁.2.trans (min_le_left _ _)) hk₂
    /-
      🎉 no goals
    -/


theorem hitting_le_iff_of_lt [WellFoundedLT ι] {m : ι} (i : ι) (hi : i < m) :
    hitting u s n m ω ≤ i ↔ ∃ j ∈ Set.Icc n i, u j ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m i : ι
    hi : LT.lt i m
    ⊢ Iff (LE.le (MeasureTheory.hitting u s n m ω) i) (Exists fun j => And (Member …
  -/
  by_cases h_exists : ∃ j ∈ Set.Icc n m, u j ω ∈ s
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LT.lt i m
      h_exists : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.me …
      ⊢ Iff (LE.le (MeasureTheory.hitting u s n m ω) i) (Exists fun j => And (Member …
    -/
  · rw [hitting_le_iff_of_exists h_exists]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LT.lt i m
      h_exists : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membersh …
      ⊢ Iff (LE.le (MeasureTheory.hitting u s n m ω) i) (Exists fun j => And (Member …
    -/
  · simp_rw [hitting, if_neg h_exists]
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LT.lt i m
      h_exists : Not (Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membersh …
      ⊢ Iff (LE.le m i) (Exists fun j => And (Membership.mem (Set.Icc n i) j) (Membe …
    -/
    push_neg at h_exists
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LT.lt i m
      h_exists : ∀ (j : ι), Membership.mem (Set.Icc n m) j → Not (Membership.mem s ( …
      ⊢ Iff (LE.le m i) (Exists fun j => And (Membership.mem (Set.Icc n i) j) (Membe …
    -/
    simp only [not_le.mpr hi, Set.mem_Icc, false_iff, not_exists, not_and, and_imp]
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LT.lt i m
      h_exists : ∀ (j : ι), Membership.mem (Set.Icc n m) j → Not (Membership.mem s ( …
      ⊢ ∀ (x : ι), LE.le n x → LE.le x i → Not (Membership.mem s (u x ω))
    -/
    exact fun k hkn hki => h_exists k ⟨hkn, hki.trans hi.le⟩
    /-
      🎉 no goals
    -/


theorem hitting_lt_iff [WellFoundedLT ι] {m : ι} (i : ι) (hi : i ≤ m) :
    hitting u s n m ω < i ↔ ∃ j ∈ Set.Ico n i, u j ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    inst✝ : WellFoundedLT ι
    m i : ι
    hi : LE.le i m
    ⊢ Iff (LT.lt (MeasureTheory.hitting u s n m ω) i) (Exists fun j => And (Member …
  -/
  constructor <;> intro h'
  · have h : ∃ j ∈ Set.Icc n m, u j ω ∈ s := by
      by_contra h
      simp_rw [hitting, if_neg h, ← not_le] at h'
      exact h' hi
    /-
      case mp
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LE.le i m
      h' : LT.lt (MeasureTheory.hitting u s n m ω) i
      h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
      ⊢ Exists fun j => And (Membership.mem (Set.Ico n i) j) (Membership.mem s (u j  …
    -/
    exact ⟨hitting u s n m ω, ⟨le_hitting_of_exists h, h'⟩, hitting_mem_set h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LE.le i m
      h' : Exists fun j => And (Membership.mem (Set.Ico n i) j) (Membership.mem s (u …
      ⊢ LT.lt (MeasureTheory.hitting u s n m ω) i
    -/
  · obtain ⟨k, hk₁, hk₂⟩ := h'
    /-
      case mpr.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LE.le i m
      k : ι
      hk₁ : Membership.mem (Set.Ico n i) k
      hk₂ : Membership.mem s (u k ω)
      ⊢ LT.lt (MeasureTheory.hitting u s n m ω) i
    -/
    refine lt_of_le_of_lt ?_ hk₁.2
    /-
      case mpr.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      inst✝ : WellFoundedLT ι
      m i : ι
      hi : LE.le i m
      k : ι
      hk₁ : Membership.mem (Set.Ico n i) k
      hk₂ : Membership.mem s (u k ω)
      ⊢ LE.le (MeasureTheory.hitting u s n m ω) k
    -/
    exact hitting_le_of_mem hk₁.1 (hk₁.2.le.trans hi) hk₂
    /-
      🎉 no goals
    -/


theorem hitting_eq_hitting_of_exists {m₁ m₂ : ι} (h : m₁ ≤ m₂)
    (h' : ∃ j ∈ Set.Icc n m₁, u j ω ∈ s) : hitting u s n m₁ ω = hitting u s n m₂ ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m₁ m₂ : ι
    h : LE.le m₁ m₂
    h' : Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem s ( …
    ⊢ Eq (MeasureTheory.hitting u s n m₁ ω) (MeasureTheory.hitting u s n m₂ ω)
  -/
  simp only [hitting, if_pos h']
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m₁ m₂ : ι
    h : LE.le m₁ m₂
    h' : Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem s ( …
    ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc n m₁) (setOf fun i => Membership.mem s …
  -/
  obtain ⟨j, hj₁, hj₂⟩ := h'
  /-
    case intro.intro
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m₁ m₂ : ι
    h : LE.le m₁ m₂
    j : ι
    hj₁ : Membership.mem (Set.Icc n m₁) j
    hj₂ : Membership.mem s (u j ω)
    ⊢ Eq (InfSet.sInf (Inter.inter (Set.Icc n m₁) (setOf fun i => Membership.mem s …
  -/
  rw [if_pos]
  · refine le_antisymm ?_ (csInf_le_csInf bddBelow_Icc.inter_of_left ⟨j, hj₁, hj₂⟩
      (Set.inter_subset_inter_left _ (Set.Icc_subset_Icc_right h)))
    /-
      case intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      m₁ m₂ : ι
      h : LE.le m₁ m₂
      j : ι
      hj₁ : Membership.mem (Set.Icc n m₁) j
      hj₂ : Membership.mem s (u j ω)
      ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m₁) (setOf fun i => Membership.me …
    -/
    refine le_csInf ⟨j, Set.Icc_subset_Icc_right h hj₁, hj₂⟩ fun i hi => ?_
    /-
      case intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      m₁ m₂ : ι
      h : LE.le m₁ m₂
      j : ι
      hj₁ : Membership.mem (Set.Icc n m₁) j
      hj₂ : Membership.mem s (u j ω)
      i : ι
      hi : Membership.mem (Inter.inter (Set.Icc n m₂) (setOf fun i => Membership.mem …
      ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m₁) (setOf fun i => Membership.me …
    -/
    by_cases hi' : i ≤ m₁
      /-
        case pos
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        h : LE.le m₁ m₂
        j : ι
        hj₁ : Membership.mem (Set.Icc n m₁) j
        hj₂ : Membership.mem s (u j ω)
        i : ι
        hi : Membership.mem (Inter.inter (Set.Icc n m₂) (setOf fun i => Membership.mem …
        hi' : LE.le i m₁
        ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m₁) (setOf fun i => Membership.me …
      -/
    · exact csInf_le bddBelow_Icc.inter_of_left ⟨⟨hi.1.1, hi'⟩, hi.2⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        h : LE.le m₁ m₂
        j : ι
        hj₁ : Membership.mem (Set.Icc n m₁) j
        hj₂ : Membership.mem s (u j ω)
        i : ι
        hi : Membership.mem (Inter.inter (Set.Icc n m₂) (setOf fun i => Membership.mem …
        hi' : Not (LE.le i m₁)
        ⊢ LE.le (InfSet.sInf (Inter.inter (Set.Icc n m₁) (setOf fun i => Membership.me …
      -/
    · change j ∈ {i | u i ω ∈ s} at hj₂
      exact ((csInf_le bddBelow_Icc.inter_of_left ⟨hj₁, hj₂⟩).trans (hj₁.2.trans le_rfl)).trans
        (le_of_lt (not_le.1 hi'))
  /-
    case intro.intro.hc
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m₁ m₂ : ι
    h : LE.le m₁ m₂
    j : ι
    hj₁ : Membership.mem (Set.Icc n m₁) j
    hj₂ : Membership.mem s (u j ω)
    ⊢ Exists fun j => And (Membership.mem (Set.Icc n m₂) j) (Membership.mem s (u j …
  -/
  exact ⟨j, ⟨hj₁.1, hj₁.2.trans h⟩, hj₂⟩
  /-
    🎉 no goals
  -/


theorem hitting_mono {m₁ m₂ : ι} (hm : m₁ ≤ m₂) : hitting u s n m₁ ω ≤ hitting u s n m₂ ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : ConditionallyCompleteLinearOrder ι
    u : ι → Ω → β
    s : Set β
    n : ι
    ω : Ω
    m₁ m₂ : ι
    hm : LE.le m₁ m₂
    ⊢ LE.le (MeasureTheory.hitting u s n m₁ ω) (MeasureTheory.hitting u s n m₂ ω)
  -/
  by_cases h : ∃ j ∈ Set.Icc n m₁, u j ω ∈ s
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      m₁ m₂ : ι
      hm : LE.le m₁ m₂
      h : Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem s (u …
      ⊢ LE.le (MeasureTheory.hitting u s n m₁ ω) (MeasureTheory.hitting u s n m₂ ω)
    -/
  · exact (hitting_eq_hitting_of_exists hm h).le
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      m₁ m₂ : ι
      hm : LE.le m₁ m₂
      h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
      ⊢ LE.le (MeasureTheory.hitting u s n m₁ ω) (MeasureTheory.hitting u s n m₂ ω)
    -/
  · simp_rw [hitting, if_neg h]
    /-
      case neg
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : ConditionallyCompleteLinearOrder ι
      u : ι → Ω → β
      s : Set β
      n : ι
      ω : Ω
      m₁ m₂ : ι
      hm : LE.le m₁ m₂
      h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
      ⊢ LE.le m₁ (ite (Exists fun j => And (Membership.mem (Set.Icc n m₂) j) (Member …
    -/
    split_ifs with h'
      /-
        case pos
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        hm : LE.le m₁ m₂
        h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
        h' : Exists fun j => And (Membership.mem (Set.Icc n m₂) j) (Membership.mem s ( …
        ⊢ LE.le m₁ (InfSet.sInf (Inter.inter (Set.Icc n m₂) (setOf fun i => Membership …
      -/
    · obtain ⟨j, hj₁, hj₂⟩ := h'
      /-
        case pos.intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        hm : LE.le m₁ m₂
        h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
        j : ι
        hj₁ : Membership.mem (Set.Icc n m₂) j
        hj₂ : Membership.mem s (u j ω)
        ⊢ LE.le m₁ (InfSet.sInf (Inter.inter (Set.Icc n m₂) (setOf fun i => Membership …
      -/
      refine le_csInf ⟨j, hj₁, hj₂⟩ ?_
      /-
        case pos.intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        hm : LE.le m₁ m₂
        h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
        j : ι
        hj₁ : Membership.mem (Set.Icc n m₂) j
        hj₂ : Membership.mem s (u j ω)
        ⊢ ∀ (b : ι), Membership.mem (Inter.inter (Set.Icc n m₂) (setOf fun i => Member …
      -/
      by_contra hneg; push_neg at hneg
      /-
        case pos.intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        hm : LE.le m₁ m₂
        h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
        j : ι
        hj₁ : Membership.mem (Set.Icc n m₂) j
        hj₂ : Membership.mem s (u j ω)
        hneg : Exists fun b => And (Membership.mem (Inter.inter (Set.Icc n m₂) (setOf  …
        ⊢ False
      -/
      obtain ⟨i, hi₁, hi₂⟩ := hneg
      /-
        case pos.intro.intro.intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        hm : LE.le m₁ m₂
        h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
        j : ι
        hj₁ : Membership.mem (Set.Icc n m₂) j
        hj₂ : Membership.mem s (u j ω)
        i : ι
        hi₁ : Membership.mem (Inter.inter (Set.Icc n m₂) (setOf fun i => Membership.me …
        hi₂ : LT.lt i m₁
        ⊢ False
      -/
      exact h ⟨i, ⟨hi₁.1.1, hi₂.le⟩, hi₁.2⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : ConditionallyCompleteLinearOrder ι
        u : ι → Ω → β
        s : Set β
        n : ι
        ω : Ω
        m₁ m₂ : ι
        hm : LE.le m₁ m₂
        h : Not (Exists fun j => And (Membership.mem (Set.Icc n m₁) j) (Membership.mem …
        h' : Not (Exists fun j => And (Membership.mem (Set.Icc n m₂) j) (Membership.me …
        ⊢ LE.le m₁ m₂
      -/
    · exact hm
      /-
        🎉 no goals
      -/


/-- A discrete hitting time is a stopping time. -/
theorem hitting_isStoppingTime [ConditionallyCompleteLinearOrder ι] [WellFoundedLT ι]
    [Countable ι] [TopologicalSpace β] [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β]
    {f : Filtration ι m} {u : ι → Ω → β} {s : Set β} {n n' : ι} (hu : Adapted f u)
    (hs : MeasurableSet s) : IsStoppingTime f (hitting u s n n') := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : ConditionallyCompleteLinearOrder ι
    inst✝⁵ : WellFoundedLT ι
    inst✝⁴ : Countable ι
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    f : MeasureTheory.Filtration ι m
    u : ι → Ω → β
    s : Set β
    n n' : ι
    hu : MeasureTheory.Adapted f u
    hs : MeasurableSet s
    ⊢ MeasureTheory.IsStoppingTime f (MeasureTheory.hitting u s n n')
  -/
  intro i
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : ConditionallyCompleteLinearOrder ι
    inst✝⁵ : WellFoundedLT ι
    inst✝⁴ : Countable ι
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    f : MeasureTheory.Filtration ι m
    u : ι → Ω → β
    s : Set β
    n n' : ι
    hu : MeasureTheory.Adapted f u
    hs : MeasurableSet s
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le (MeasureTheory.hitting u s n n' ω) i)
  -/
  rcases le_or_lt n' i with hi | hi
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : ConditionallyCompleteLinearOrder ι
      inst✝⁵ : WellFoundedLT ι
      inst✝⁴ : Countable ι
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      f : MeasureTheory.Filtration ι m
      u : ι → Ω → β
      s : Set β
      n n' : ι
      hu : MeasureTheory.Adapted f u
      hs : MeasurableSet s
      i : ι
      hi : LE.le n' i
      ⊢ MeasurableSet (setOf fun ω => LE.le (MeasureTheory.hitting u s n n' ω) i)
    -/
  · have h_le : ∀ ω, hitting u s n n' ω ≤ i := fun x => (hitting_le x).trans hi
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : ConditionallyCompleteLinearOrder ι
      inst✝⁵ : WellFoundedLT ι
      inst✝⁴ : Countable ι
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      f : MeasureTheory.Filtration ι m
      u : ι → Ω → β
      s : Set β
      n n' : ι
      hu : MeasureTheory.Adapted f u
      hs : MeasurableSet s
      i : ι
      hi : LE.le n' i
      h_le : ∀ (ω : Ω), LE.le (MeasureTheory.hitting u s n n' ω) i
      ⊢ MeasurableSet (setOf fun ω => LE.le (MeasureTheory.hitting u s n n' ω) i)
    -/
    simp [h_le]
    /-
      🎉 no goals
    -/
  · have h_set_eq_Union : {ω | hitting u s n n' ω ≤ i} = ⋃ j ∈ Set.Icc n i, u j ⁻¹' s := by
      ext x
      rw [Set.mem_setOf_eq, hitting_le_iff_of_lt _ hi]
      simp only [Set.mem_Icc, exists_prop, Set.mem_iUnion, Set.mem_preimage]
    /-
      case inr
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : ConditionallyCompleteLinearOrder ι
      inst✝⁵ : WellFoundedLT ι
      inst✝⁴ : Countable ι
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      f : MeasureTheory.Filtration ι m
      u : ι → Ω → β
      s : Set β
      n n' : ι
      hu : MeasureTheory.Adapted f u
      hs : MeasurableSet s
      i : ι
      hi : LT.lt i n'
      h_set_eq_Union : Eq (setOf fun ω => LE.le (MeasureTheory.hitting u s n n' ω) i …
      ⊢ MeasurableSet (setOf fun ω => LE.le (MeasureTheory.hitting u s n n' ω) i)
    -/
    rw [h_set_eq_Union]
    exact MeasurableSet.iUnion fun j =>
      MeasurableSet.iUnion fun hj => f.mono hj.2 _ ((hu j).measurable hs)


theorem stoppedValue_hitting_mem [ConditionallyCompleteLinearOrder ι] [WellFoundedLT ι]
    {u : ι → Ω → β} {s : Set β} {n m : ι} {ω : Ω} (h : ∃ j ∈ Set.Icc n m, u j ω ∈ s) :
    stoppedValue u (hitting u s n m) ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    inst✝ : WellFoundedLT ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    ω : Ω
    h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
    ⊢ Membership.mem s (MeasureTheory.stoppedValue u (MeasureTheory.hitting u s n  …
  -/
  simp only [stoppedValue, hitting, if_pos h]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    inst✝ : WellFoundedLT ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    ω : Ω
    h : Exists fun j => And (Membership.mem (Set.Icc n m) j) (Membership.mem s (u  …
    ⊢ Membership.mem s (u (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i =>  …
  -/
  obtain ⟨j, hj₁, hj₂⟩ := h
  have : sInf (Set.Icc n m ∩ {i | u i ω ∈ s}) ∈ Set.Icc n m ∩ {i | u i ω ∈ s} :=
    csInf_mem (Set.nonempty_of_mem ⟨hj₁, hj₂⟩)
  /-
    case intro.intro
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrder ι
    inst✝ : WellFoundedLT ι
    u : ι → Ω → β
    s : Set β
    n m : ι
    ω : Ω
    j : ι
    hj₁ : Membership.mem (Set.Icc n m) j
    hj₂ : Membership.mem s (u j ω)
    this : Membership.mem (Inter.inter (Set.Icc n m) (setOf fun i => Membership.me …
    ⊢ Membership.mem s (u (InfSet.sInf (Inter.inter (Set.Icc n m) (setOf fun i =>  …
  -/
  exact this.2
  /-
    🎉 no goals
  -/


/-- The hitting time of a discrete process with the starting time indexed by a stopping time
is a stopping time. -/
theorem isStoppingTime_hitting_isStoppingTime [ConditionallyCompleteLinearOrder ι]
    [WellFoundedLT ι] [Countable ι] [TopologicalSpace ι] [OrderTopology ι]
    [FirstCountableTopology ι] [TopologicalSpace β] [PseudoMetrizableSpace β] [MeasurableSpace β]
    [BorelSpace β] {f : Filtration ι m} {u : ι → Ω → β} {τ : Ω → ι} (hτ : IsStoppingTime f τ)
    {N : ι} (hτbdd : ∀ x, τ x ≤ N) {s : Set β} (hs : MeasurableSet s) (hf : Adapted f u) :
    IsStoppingTime f fun x => hitting u s (τ x) N x := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : ConditionallyCompleteLinearOrder ι
    inst✝⁸ : WellFoundedLT ι
    inst✝⁷ : Countable ι
    inst✝⁶ : TopologicalSpace ι
    inst✝⁵ : OrderTopology ι
    inst✝⁴ : FirstCountableTopology ι
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    f : MeasureTheory.Filtration ι m
    u : ι → Ω → β
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    N : ι
    hτbdd : ∀ (x : Ω), LE.le (τ x) N
    s : Set β
    hs : MeasurableSet s
    hf : MeasureTheory.Adapted f u
    ⊢ MeasureTheory.IsStoppingTime f fun x => MeasureTheory.hitting u s (τ x) N x
  -/
  intro n
  have h₁ : {x | hitting u s (τ x) N x ≤ n} =
    (⋃ i ≤ n, {x | τ x = i} ∩ {x | hitting u s i N x ≤ n}) ∪
      ⋃ i > n, {x | τ x = i} ∩ {x | hitting u s i N x ≤ n} := by
    ext x
    simp [← exists_or, ← or_and_right, le_or_lt]
  have h₂ : ⋃ i > n, {x | τ x = i} ∩ {x | hitting u s i N x ≤ n} = ∅ := by
    ext x
    simp only [gt_iff_lt, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_setOf_eq, exists_prop,
      Set.mem_empty_iff_false, iff_false, not_exists, not_and, not_le]
    rintro m hm rfl
    exact lt_of_lt_of_le hm (le_hitting (hτbdd _) _)
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : ConditionallyCompleteLinearOrder ι
    inst✝⁸ : WellFoundedLT ι
    inst✝⁷ : Countable ι
    inst✝⁶ : TopologicalSpace ι
    inst✝⁵ : OrderTopology ι
    inst✝⁴ : FirstCountableTopology ι
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    f : MeasureTheory.Filtration ι m
    u : ι → Ω → β
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    N : ι
    hτbdd : ∀ (x : Ω), LE.le (τ x) N
    s : Set β
    hs : MeasurableSet s
    hf : MeasureTheory.Adapted f u
    n : ι
    h₁ : Eq (setOf fun x => LE.le (MeasureTheory.hitting u s (τ x) N x) n) (Union. …
    h₂ : Eq (Set.iUnion fun i => Set.iUnion fun h => Inter.inter (setOf fun x => E …
    ⊢ MeasurableSet (setOf fun ω => LE.le ((fun x => MeasureTheory.hitting u s (τ  …
  -/
  rw [h₁, h₂, Set.union_empty]
  exact MeasurableSet.iUnion fun i => MeasurableSet.iUnion fun hi =>
    (f.mono hi _ (hτ.measurableSet_eq i)).inter (hitting_isStoppingTime hf hs n)


theorem hitting_eq_sInf (ω : Ω) : hitting u s ⊥ ⊤ ω = sInf {i : ι | u i ω ∈ s} := by
  simp only [hitting, Set.mem_Icc, bot_le, le_top, and_self_iff, exists_true_left, Set.Icc_bot,
    Set.Iic_top, Set.univ_inter, ite_eq_left_iff, not_exists]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : CompleteLattice ι
    u : ι → Ω → β
    s : Set β
    ω : Ω
    ⊢ (∀ (x : ι), Not (And (Membership.mem Set.univ x) (Membership.mem s (u x ω))) …
  -/
  intro h_nmem_s
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : CompleteLattice ι
    u : ι → Ω → β
    s : Set β
    ω : Ω
    h_nmem_s : ∀ (x : ι), Not (And (Membership.mem Set.univ x) (Membership.mem s ( …
    ⊢ Eq Top.top (InfSet.sInf (setOf fun i => Membership.mem s (u i ω)))
  -/
  symm
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : CompleteLattice ι
    u : ι → Ω → β
    s : Set β
    ω : Ω
    h_nmem_s : ∀ (x : ι), Not (And (Membership.mem Set.univ x) (Membership.mem s ( …
    ⊢ Eq (InfSet.sInf (setOf fun i => Membership.mem s (u i ω))) Top.top
  -/
  rw [sInf_eq_top]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : CompleteLattice ι
    u : ι → Ω → β
    s : Set β
    ω : Ω
    h_nmem_s : ∀ (x : ι), Not (And (Membership.mem Set.univ x) (Membership.mem s ( …
    ⊢ ∀ (a : ι), Membership.mem (setOf fun i => Membership.mem s (u i ω)) a → Eq a …
  -/
  simp only [Set.mem_univ, true_and] at h_nmem_s
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : CompleteLattice ι
    u : ι → Ω → β
    s : Set β
    ω : Ω
    h_nmem_s : ∀ (x : ι), Not (Membership.mem s (u x ω))
    ⊢ ∀ (a : ι), Membership.mem (setOf fun i => Membership.mem s (u i ω)) a → Eq a …
  -/
  exact fun i hi_mem_s => absurd hi_mem_s (h_nmem_s i)
  /-
    🎉 no goals
  -/


theorem hitting_bot_le_iff {i n : ι} {ω : Ω} (hx : ∃ j, j ≤ n ∧ u j ω ∈ s) :
    hitting u s ⊥ n ω ≤ i ↔ ∃ j ≤ i, u j ω ∈ s := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : ConditionallyCompleteLinearOrderBot ι
    inst✝ : WellFoundedLT ι
    u : ι → Ω → β
    s : Set β
    i n : ι
    ω : Ω
    hx : Exists fun j => And (LE.le j n) (Membership.mem s (u j ω))
    ⊢ Iff (LE.le (MeasureTheory.hitting u s Bot.bot n ω) i) (Exists fun j => And ( …
  -/
  cases' lt_or_le i n with hi hi
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrderBot ι
      inst✝ : WellFoundedLT ι
      u : ι → Ω → β
      s : Set β
      i n : ι
      ω : Ω
      hx : Exists fun j => And (LE.le j n) (Membership.mem s (u j ω))
      hi : LT.lt i n
      ⊢ Iff (LE.le (MeasureTheory.hitting u s Bot.bot n ω) i) (Exists fun j => And ( …
    -/
  · rw [hitting_le_iff_of_lt _ hi]
    /-
      case inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrderBot ι
      inst✝ : WellFoundedLT ι
      u : ι → Ω → β
      s : Set β
      i n : ι
      ω : Ω
      hx : Exists fun j => And (LE.le j n) (Membership.mem s (u j ω))
      hi : LT.lt i n
      ⊢ Iff (Exists fun j => And (Membership.mem (Set.Icc Bot.bot i) j) (Membership. …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrderBot ι
      inst✝ : WellFoundedLT ι
      u : ι → Ω → β
      s : Set β
      i n : ι
      ω : Ω
      hx : Exists fun j => And (LE.le j n) (Membership.mem s (u j ω))
      hi : LE.le n i
      ⊢ Iff (LE.le (MeasureTheory.hitting u s Bot.bot n ω) i) (Exists fun j => And ( …
    -/
  · simp only [(hitting_le ω).trans hi, true_iff]
    /-
      case inr
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrderBot ι
      inst✝ : WellFoundedLT ι
      u : ι → Ω → β
      s : Set β
      i n : ι
      ω : Ω
      hx : Exists fun j => And (LE.le j n) (Membership.mem s (u j ω))
      hi : LE.le n i
      ⊢ Exists fun j => And (LE.le j i) (Membership.mem s (u j ω))
    -/
    obtain ⟨j, hj₁, hj₂⟩ := hx
    /-
      case inr.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝¹ : ConditionallyCompleteLinearOrderBot ι
      inst✝ : WellFoundedLT ι
      u : ι → Ω → β
      s : Set β
      i n : ι
      ω : Ω
      hi : LE.le n i
      j : ι
      hj₁ : LE.le j n
      hj₂ : Membership.mem s (u j ω)
      ⊢ Exists fun j => And (LE.le j i) (Membership.mem s (u j ω))
    -/
    exact ⟨j, hj₁.trans hi, hj₂⟩
    /-
      🎉 no goals
    -/


