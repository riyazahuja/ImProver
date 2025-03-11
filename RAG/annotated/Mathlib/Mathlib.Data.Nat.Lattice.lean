noncomputable instance : InfSet ℕ :=
  ⟨fun s ↦ if h : ∃ n, n ∈ s then @Nat.find (fun n ↦ n ∈ s) _ h else 0⟩


noncomputable instance : SupSet ℕ :=
  ⟨fun s ↦ if h : ∃ n, ∀ a ∈ s, a ≤ n then @Nat.find (fun n ↦ ∀ a ∈ s, a ≤ n) _ h else 0⟩


theorem sInf_def {s : Set ℕ} (h : s.Nonempty) : sInf s = @Nat.find (fun n ↦ n ∈ s) _ h :=
  dif_pos _


theorem sSup_def {s : Set ℕ} (h : ∃ n, ∀ a ∈ s, a ≤ n) :
    sSup s = @Nat.find (fun n ↦ ∀ a ∈ s, a ≤ n) _ h :=
  dif_pos _


theorem _root_.Set.Infinite.Nat.sSup_eq_zero {s : Set ℕ} (h : s.Infinite) : sSup s = 0 :=
  dif_neg fun ⟨n, hn⟩ ↦
    let ⟨k, hks, hk⟩ := h.exists_gt n
    (hn k hks).not_lt hk


@[simp]
theorem sInf_eq_zero {s : Set ℕ} : sInf s = 0 ↔ 0 ∈ s ∨ s = ∅ := by
  cases eq_empty_or_nonempty s with
  | inl h => subst h
             simp only [or_true, eq_self_iff_true, iInf, InfSet.sInf,
                        mem_empty_iff_false, exists_false, dif_neg, not_false_iff]
  | inr h => simp only [h.ne_empty, or_false, Nat.sInf_def, h, Nat.find_eq_zero]


@[simp]
theorem sInf_empty : sInf ∅ = 0 := by
  /-
    ⊢ Eq (InfSet.sInf EmptyCollection.emptyCollection) 0
  -/
  rw [sInf_eq_zero]
  /-
    ⊢ Or (Membership.mem EmptyCollection.emptyCollection 0) (Eq EmptyCollection.em …
  -/
  right
  /-
    case h
    ⊢ Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_of_empty {ι : Sort*} [IsEmpty ι] (f : ι → ℕ) : iInf f = 0 := by
  /-
    ι : Sort u_1
    inst✝ : IsEmpty ι
    f : ι → Nat
    ⊢ Eq (iInf f) 0
  -/
  rw [iInf_of_isEmpty, sInf_empty]
  /-
    🎉 no goals
  -/


/-- This combines `Nat.iInf_of_empty` with `ciInf_const`. -/
@[simp]
lemma iInf_const_zero {ι : Sort*} : ⨅ _ : ι, 0 = 0 :=
                                           /-
                                             ι : Sort u_1
                                             h : IsEmpty ι
                                             ⊢ Eq (iInf fun x => 0) 0
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  (isEmpty_or_nonempty ι).elim (fun h ↦ by simp) fun h ↦ sInf_eq_zero.2 <| by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem sInf_mem {s : Set ℕ} (h : s.Nonempty) : sInf s ∈ s := by
  /-
    s : Set Nat
    h : s.Nonempty
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  rw [Nat.sInf_def h]
  /-
    s : Set Nat
    h : s.Nonempty
    ⊢ Membership.mem s (Nat.find h)
  -/
  exact Nat.find_spec h
  /-
    🎉 no goals
  -/


theorem not_mem_of_lt_sInf {s : Set ℕ} {m : ℕ} (hm : m < sInf s) : m ∉ s := by
  cases eq_empty_or_nonempty s with
  | inl h => subst h; apply not_mem_empty
  | inr h => rw [Nat.sInf_def h] at hm; exact Nat.find_min h hm


protected theorem sInf_le {s : Set ℕ} {m : ℕ} (hm : m ∈ s) : sInf s ≤ m := by
  /-
    s : Set Nat
    m : Nat
    hm : Membership.mem s m
    ⊢ LE.le (InfSet.sInf s) m
  -/
  rw [Nat.sInf_def ⟨m, hm⟩]
  /-
    s : Set Nat
    m : Nat
    hm : Membership.mem s m
    ⊢ LE.le (Nat.find ⋯) m
  -/
  exact Nat.find_min' ⟨m, hm⟩ hm
  /-
    🎉 no goals
  -/


theorem nonempty_of_pos_sInf {s : Set ℕ} (h : 0 < sInf s) : s.Nonempty := by
  /-
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    ⊢ s.Nonempty
  -/
  by_contra contra
  /-
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    contra : Not s.Nonempty
    ⊢ False
  -/
  rw [Set.not_nonempty_iff_eq_empty] at contra
  /-
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    contra : Eq s EmptyCollection.emptyCollection
    ⊢ False
  -/
  have h' : sInf s ≠ 0 := ne_of_gt h
  /-
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    contra : Eq s EmptyCollection.emptyCollection
    h' : Ne (InfSet.sInf s) 0
    ⊢ False
  -/
  apply h'
  /-
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    contra : Eq s EmptyCollection.emptyCollection
    h' : Ne (InfSet.sInf s) 0
    ⊢ Eq (InfSet.sInf s) 0
  -/
  rw [Nat.sInf_eq_zero]
  /-
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    contra : Eq s EmptyCollection.emptyCollection
    h' : Ne (InfSet.sInf s) 0
    ⊢ Or (Membership.mem s 0) (Eq s EmptyCollection.emptyCollection)
  -/
  right
  /-
    case h
    s : Set Nat
    h : LT.lt 0 (InfSet.sInf s)
    contra : Eq s EmptyCollection.emptyCollection
    h' : Ne (InfSet.sInf s) 0
    ⊢ Eq s EmptyCollection.emptyCollection
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem nonempty_of_sInf_eq_succ {s : Set ℕ} {k : ℕ} (h : sInf s = k + 1) : s.Nonempty :=
  nonempty_of_pos_sInf (h.symm ▸ succ_pos k : sInf s > 0)


theorem eq_Ici_of_nonempty_of_upward_closed {s : Set ℕ} (hs : s.Nonempty)
    (hs' : ∀ k₁ k₂ : ℕ, k₁ ≤ k₂ → k₁ ∈ s → k₂ ∈ s) : s = Ici (sInf s) :=
  ext fun n ↦ ⟨fun H ↦ Nat.sInf_le H, fun H ↦ hs' (sInf s) n H (sInf_mem hs)⟩


theorem sInf_upward_closed_eq_succ_iff {s : Set ℕ} (hs : ∀ k₁ k₂ : ℕ, k₁ ≤ k₂ → k₁ ∈ s → k₂ ∈ s)
    (k : ℕ) : sInf s = k + 1 ↔ k + 1 ∈ s ∧ k ∉ s := by
  /-
    s : Set Nat
    hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
    k : Nat
    ⊢ Iff (Eq (InfSet.sInf s) (HAdd.hAdd k 1)) (And (Membership.mem s (HAdd.hAdd k …
  -/
  constructor
    /-
      case mp
      s : Set Nat
      hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
      k : Nat
      ⊢ Eq (InfSet.sInf s) (HAdd.hAdd k 1) → And (Membership.mem s (HAdd.hAdd k 1))  …
    -/
  · intro H
    /-
      case mp
      s : Set Nat
      hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
      k : Nat
      H : Eq (InfSet.sInf s) (HAdd.hAdd k 1)
      ⊢ And (Membership.mem s (HAdd.hAdd k 1)) (Not (Membership.mem s k))
    -/
    rw [eq_Ici_of_nonempty_of_upward_closed (nonempty_of_sInf_eq_succ _) hs, H, mem_Ici, mem_Ici]
      /-
        case mp
        s : Set Nat
        hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
        k : Nat
        H : Eq (InfSet.sInf s) (HAdd.hAdd k 1)
        ⊢ And (LE.le (HAdd.hAdd k 1) (HAdd.hAdd k 1)) (Not (LE.le (HAdd.hAdd k 1) k))
      -/
    · exact ⟨le_rfl, k.not_succ_le_self⟩
      /-
        🎉 no goals
      -/
      /-
        s : Set Nat
        hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
        k : Nat
        H : Eq (InfSet.sInf s) (HAdd.hAdd k 1)
        ⊢ Nat
      -/
    · exact k
      /-
        🎉 no goals
      -/
      /-
        s : Set Nat
        hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
        k : Nat
        H : Eq (InfSet.sInf s) (HAdd.hAdd k 1)
        ⊢ Eq (InfSet.sInf s) (HAdd.hAdd k 1)
      -/
    · assumption
      /-
        🎉 no goals
      -/
    /-
      case mpr
      s : Set Nat
      hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
      k : Nat
      ⊢ And (Membership.mem s (HAdd.hAdd k 1)) (Not (Membership.mem s k)) → Eq (InfS …
    -/
  · rintro ⟨H, H'⟩
    /-
      case mpr.intro
      s : Set Nat
      hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
      k : Nat
      H : Membership.mem s (HAdd.hAdd k 1)
      H' : Not (Membership.mem s k)
      ⊢ Eq (InfSet.sInf s) (HAdd.hAdd k 1)
    -/
    rw [sInf_def (⟨_, H⟩ : s.Nonempty), find_eq_iff]
    /-
      case mpr.intro
      s : Set Nat
      hs : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
      k : Nat
      H : Membership.mem s (HAdd.hAdd k 1)
      H' : Not (Membership.mem s k)
      ⊢ And (Membership.mem s (HAdd.hAdd k 1)) (∀ (n : Nat), LT.lt n (HAdd.hAdd k 1) …
    -/
    exact ⟨H, fun n hnk hns ↦ H' <| hs n k (Nat.lt_succ_iff.mp hnk) hns⟩
    /-
      🎉 no goals
    -/


/-- This instance is necessary, otherwise the lattice operations would be derived via
`ConditionallyCompleteLinearOrderBot` and marked as noncomputable. -/
instance : Lattice ℕ :=
  LinearOrder.toLattice


noncomputable instance : ConditionallyCompleteLinearOrderBot ℕ :=
  { (inferInstance : OrderBot ℕ), (LinearOrder.toLattice : Lattice ℕ),
    (inferInstance : LinearOrder ℕ) with
    -- sup := sSup -- Porting note: removed, unnecessary?
    -- inf := sInf -- Porting note: removed, unnecessary?
                                   /-
                                     s : Set Nat
                                     a : Nat
                                     hb : BddAbove s
                                     ha : Membership.mem s a
                                     ⊢ LE.le a (SupSet.sSup s)
                                   -/
    le_csSup := fun s a hb ha ↦ by rw [sSup_def hb]; revert a ha; exact @Nat.find_spec _ _ hb
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                  /-
                                    s : Set Nat
                                    a : Nat
                                    x✝ : s.Nonempty
                                    ha : Membership.mem (upperBounds s) a
                                    ⊢ LE.le (SupSet.sSup s) a
                                  -/
    csSup_le := fun s a _ ha ↦ by rw [sSup_def ⟨a, ha⟩]; exact Nat.find_min' _ ha
                                                         /-
                                                           🎉 no goals
                                                         -/
    le_csInf := fun s a hs hb ↦ by
      /-
        s : Set Nat
        a : Nat
        hs : s.Nonempty
        hb : Membership.mem (lowerBounds s) a
        ⊢ LE.le a (InfSet.sInf s)
      -/
                                  /-
                                    s : Set Nat
                                    a : Nat
                                    x✝ : BddBelow s
                                    ha : Membership.mem s a
                                    ⊢ LE.le (InfSet.sInf s) a
                                  -/
      rw [sInf_def hs]; exact hb (@Nat.find_spec (fun n ↦ n ∈ s) _ _)
                                                         /-
                                                           🎉 no goals
                                                         -/
                        /-
                          🎉 no goals
                        -/
    csInf_le := fun s a _ ha ↦ by rw [sInf_def ⟨a, ha⟩]; exact Nat.find_min' _ ha
    csSup_empty := by
      simp only [sSup_def, Set.mem_empty_iff_false, forall_const, forall_prop_of_false,
        not_false_iff, exists_const]
      /-
        ⊢ Eq (Nat.find ⋯) Bot.bot
      -/
      apply bot_unique (Nat.find_min' _ _)
      /-
        ⊢ True
      -/
      /-
        ⊢ ∀ (s : Set Nat), Not (BddAbove s) → Eq (SupSet.sSup s) (SupSet.sSup EmptyCol …
      -/
      trivial
      /-
        🎉 no goals
      -/
    csSup_of_not_bddAbove := by
      /-
        s : Set Nat
        hs : Not (BddAbove s)
        ⊢ Eq (dite (Exists fun n => ∀ (a : Nat), Membership.mem s a → LE.le a n) (fun  …
      -/
      intro s hs
        /-
          s : Set Nat
          hs : Not (BddAbove s)
          ⊢ Eq 0 (Nat.find ⋯)
        -/
      simp only [mem_univ, forall_true_left, sSup,
        /-
          🎉 no goals
        -/
        /-
          case hnc
          s : Set Nat
          hs : Not (BddAbove s)
          ⊢ Not (Exists fun n => ∀ (a : Nat), Membership.mem s a → LE.le a n)
        -/
        mem_empty_iff_false, IsEmpty.forall_iff, forall_const, exists_const, dite_true]
        /-
          🎉 no goals
        -/
                                           /-
                                             s : Set Nat
                                             hs : Not (BddBelow s)
                                             ⊢ Eq (InfSet.sInf s) (InfSet.sInf EmptyCollection.emptyCollection)
                                           -/
      rw [dif_neg]
                                           /-
                                             🎉 no goals
                                           -/
      · exact le_antisymm (zero_le _) (find_le trivial)
      · exact hs
    csInf_of_not_bddBelow := fun s hs ↦ by simp at hs }


theorem sSup_mem {s : Set ℕ} (h₁ : s.Nonempty) (h₂ : BddAbove s) : sSup s ∈ s :=
  let ⟨k, hk⟩ := h₂
  h₁.csSup_mem ((finite_le_nat k).subset hk)


theorem sInf_add {n : ℕ} {p : ℕ → Prop} (hn : n ≤ sInf { m | p m }) :
    sInf { m | p (m + n) } + n = sInf { m | p m } := by
  /-
    n : Nat
    p : Nat → Prop
    hn : LE.le n (InfSet.sInf (setOf fun m => p m))
    ⊢ Eq (HAdd.hAdd (InfSet.sInf (setOf fun m => p (HAdd.hAdd m n))) n) (InfSet.sI …
  -/
  obtain h | ⟨m, hm⟩ := { m | p (m + n) }.eq_empty_or_nonempty
    /-
      case inl
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      h : Eq (setOf fun m => p (HAdd.hAdd m n)) EmptyCollection.emptyCollection
      ⊢ Eq (HAdd.hAdd (InfSet.sInf (setOf fun m => p (HAdd.hAdd m n))) n) (InfSet.sI …
    -/
  · rw [h, Nat.sInf_empty, zero_add]
    /-
      case inl
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      h : Eq (setOf fun m => p (HAdd.hAdd m n)) EmptyCollection.emptyCollection
      ⊢ Eq n (InfSet.sInf (setOf fun m => p m))
    -/
    obtain hnp | hnp := hn.eq_or_lt
      /-
        case inl.inl
        n : Nat
        p : Nat → Prop
        hn : LE.le n (InfSet.sInf (setOf fun m => p m))
        h : Eq (setOf fun m => p (HAdd.hAdd m n)) EmptyCollection.emptyCollection
        hnp : Eq n (InfSet.sInf (setOf fun m => p m))
        ⊢ Eq n (InfSet.sInf (setOf fun m => p m))
      -/
    · exact hnp
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      h : Eq (setOf fun m => p (HAdd.hAdd m n)) EmptyCollection.emptyCollection
      hnp : LT.lt n (InfSet.sInf (setOf fun m => p m))
      ⊢ Eq n (InfSet.sInf (setOf fun m => p m))
    -/
    suffices hp : p (sInf { m | p m } - n + n) from (h.subset hp).elim
    /-
      case inl.inr
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      h : Eq (setOf fun m => p (HAdd.hAdd m n)) EmptyCollection.emptyCollection
      hnp : LT.lt n (InfSet.sInf (setOf fun m => p m))
      ⊢ p (HAdd.hAdd (HSub.hSub (InfSet.sInf (setOf fun m => p m)) n) n)
    -/
    rw [Nat.sub_add_cancel hn]
    /-
      case inl.inr
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      h : Eq (setOf fun m => p (HAdd.hAdd m n)) EmptyCollection.emptyCollection
      hnp : LT.lt n (InfSet.sInf (setOf fun m => p m))
      ⊢ p (InfSet.sInf (setOf fun m => p m))
    -/
    exact csInf_mem (nonempty_of_pos_sInf <| n.zero_le.trans_lt hnp)
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      m : Nat
      hm : Membership.mem (setOf fun m => p (HAdd.hAdd m n)) m
      ⊢ Eq (HAdd.hAdd (InfSet.sInf (setOf fun m => p (HAdd.hAdd m n))) n) (InfSet.sI …
    -/
  · have hp : ∃ n, n ∈ { m | p m } := ⟨_, hm⟩
    /-
      case inr.intro
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      m : Nat
      hm : Membership.mem (setOf fun m => p (HAdd.hAdd m n)) m
      hp : Exists fun n => Membership.mem (setOf fun m => p m) n
      ⊢ Eq (HAdd.hAdd (InfSet.sInf (setOf fun m => p (HAdd.hAdd m n))) n) (InfSet.sI …
    -/
    rw [Nat.sInf_def ⟨m, hm⟩, Nat.sInf_def hp]
    /-
      case inr.intro
      n : Nat
      p : Nat → Prop
      hn : LE.le n (InfSet.sInf (setOf fun m => p m))
      m : Nat
      hm : Membership.mem (setOf fun m => p (HAdd.hAdd m n)) m
      hp : Exists fun n => Membership.mem (setOf fun m => p m) n
      ⊢ Eq (HAdd.hAdd (Nat.find ⋯) n) (Nat.find hp)
    -/
    rw [Nat.sInf_def hp] at hn
    /-
      case inr.intro
      n : Nat
      p : Nat → Prop
      m : Nat
      hm : Membership.mem (setOf fun m => p (HAdd.hAdd m n)) m
      hp : Exists fun n => Membership.mem (setOf fun m => p m) n
      hn : LE.le n (Nat.find hp)
      ⊢ Eq (HAdd.hAdd (Nat.find ⋯) n) (Nat.find hp)
    -/
    exact find_add hn
    /-
      🎉 no goals
    -/


theorem sInf_add' {n : ℕ} {p : ℕ → Prop} (h : 0 < sInf { m | p m }) :
    sInf { m | p m } + n = sInf { m | p (m - n) } := by
  suffices h₁ : n ≤ sInf {m | p (m - n)} by
    convert sInf_add h₁
    simp_rw [Nat.add_sub_cancel_right]
  /-
    n : Nat
    p : Nat → Prop
    h : LT.lt 0 (InfSet.sInf (setOf fun m => p m))
    ⊢ LE.le n (InfSet.sInf (setOf fun m => p (HSub.hSub m n)))
  -/
  obtain ⟨m, hm⟩ := nonempty_of_pos_sInf h
  refine
    le_csInf ⟨m + n, ?_⟩ fun b hb ↦
      le_of_not_lt fun hbn ↦
        ne_of_mem_of_not_mem ?_ (not_mem_of_lt_sInf h) (Nat.sub_eq_zero_of_le hbn.le)
    /-
      case intro.refine_1
      n : Nat
      p : Nat → Prop
      h : LT.lt 0 (InfSet.sInf (setOf fun m => p m))
      m : Nat
      hm : Membership.mem (setOf fun m => p m) m
      ⊢ Membership.mem (setOf fun m => p (HSub.hSub m n)) (HAdd.hAdd m n)
    -/
  · dsimp
    /-
      case intro.refine_1
      n : Nat
      p : Nat → Prop
      h : LT.lt 0 (InfSet.sInf (setOf fun m => p m))
      m : Nat
      hm : Membership.mem (setOf fun m => p m) m
      ⊢ p (HSub.hSub (HAdd.hAdd m n) n)
    -/
    rwa [Nat.add_sub_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      n : Nat
      p : Nat → Prop
      h : LT.lt 0 (InfSet.sInf (setOf fun m => p m))
      m : Nat
      hm : Membership.mem (setOf fun m => p m) m
      b : Nat
      hb : Membership.mem (setOf fun m => p (HSub.hSub m n)) b
      hbn : LT.lt b n
      ⊢ Membership.mem (setOf fun m => p m) (HSub.hSub b n)
    -/
  · exact hb
    /-
      🎉 no goals
    -/


theorem iSup_lt_succ (u : ℕ → α) (n : ℕ) : ⨆ k < n + 1, u k = (⨆ k < n, u k) ⊔ u n := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    n : Nat
    ⊢ Eq (iSup fun k => iSup fun h => u k) (Max.max (iSup fun k => iSup fun h => u …
  -/
  simp [Nat.lt_succ_iff_lt_or_eq, iSup_or, iSup_sup_eq]
  /-
    🎉 no goals
  -/


theorem iSup_lt_succ' (u : ℕ → α) (n : ℕ) : ⨆ k < n + 1, u k = u 0 ⊔ ⨆ k < n, u (k + 1) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    n : Nat
    ⊢ Eq (iSup fun k => iSup fun h => u k) (Max.max (u 0) (iSup fun k => iSup fun  …
  -/
  rw [← sup_iSup_nat_succ]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    n : Nat
    ⊢ Eq (Max.max (iSup fun h => u 0) (iSup fun i => iSup fun h => u (HAdd.hAdd i  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iInf_lt_succ (u : ℕ → α) (n : ℕ) : ⨅ k < n + 1, u k = (⨅ k < n, u k) ⊓ u n :=
  @iSup_lt_succ αᵒᵈ _ _ _


theorem iInf_lt_succ' (u : ℕ → α) (n : ℕ) : ⨅ k < n + 1, u k = u 0 ⊓ ⨅ k < n, u (k + 1) :=
  @iSup_lt_succ' αᵒᵈ _ _ _


theorem iSup_le_succ (u : ℕ → α) (n : ℕ) : ⨆ k ≤ n + 1, u k = (⨆ k ≤ n, u k) ⊔ u (n + 1) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    n : Nat
    ⊢ Eq (iSup fun k => iSup fun h => u k) (Max.max (iSup fun k => iSup fun h => u …
  -/
  simp_rw [← Nat.lt_succ_iff, iSup_lt_succ]
  /-
    🎉 no goals
  -/


theorem iSup_le_succ' (u : ℕ → α) (n : ℕ) : ⨆ k ≤ n + 1, u k = u 0 ⊔ ⨆ k ≤ n, u (k + 1) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    n : Nat
    ⊢ Eq (iSup fun k => iSup fun h => u k) (Max.max (u 0) (iSup fun k => iSup fun  …
  -/
  simp_rw [← Nat.lt_succ_iff, iSup_lt_succ']
  /-
    🎉 no goals
  -/


theorem iInf_le_succ (u : ℕ → α) (n : ℕ) : ⨅ k ≤ n + 1, u k = (⨅ k ≤ n, u k) ⊓ u (n + 1) :=
  @iSup_le_succ αᵒᵈ _ _ _


theorem iInf_le_succ' (u : ℕ → α) (n : ℕ) : ⨅ k ≤ n + 1, u k = u 0 ⊓ ⨅ k ≤ n, u (k + 1) :=
  @iSup_le_succ' αᵒᵈ _ _ _


theorem biUnion_lt_succ (u : ℕ → Set α) (n : ℕ) : ⋃ k < n + 1, u k = (⋃ k < n, u k) ∪ u n :=
  Nat.iSup_lt_succ u n


theorem biUnion_lt_succ' (u : ℕ → Set α) (n : ℕ) : ⋃ k < n + 1, u k = u 0 ∪ ⋃ k < n, u (k + 1) :=
  Nat.iSup_lt_succ' u n


theorem biInter_lt_succ (u : ℕ → Set α) (n : ℕ) : ⋂ k < n + 1, u k = (⋂ k < n, u k) ∩ u n :=
  Nat.iInf_lt_succ u n


theorem biInter_lt_succ' (u : ℕ → Set α) (n : ℕ) : ⋂ k < n + 1, u k = u 0 ∩ ⋂ k < n, u (k + 1) :=
  Nat.iInf_lt_succ' u n


theorem biUnion_le_succ (u : ℕ → Set α) (n : ℕ) : ⋃ k ≤ n + 1, u k = (⋃ k ≤ n, u k) ∪ u (n + 1) :=
  Nat.iSup_le_succ u n


theorem biUnion_le_succ' (u : ℕ → Set α) (n : ℕ) : ⋃ k ≤ n + 1, u k = u 0 ∪ ⋃ k ≤ n, u (k + 1) :=
  Nat.iSup_le_succ' u n


theorem biInter_le_succ (u : ℕ → Set α) (n : ℕ) : ⋂ k ≤ n + 1, u k = (⋂ k ≤ n, u k) ∩ u (n + 1) :=
  Nat.iInf_le_succ u n


theorem biInter_le_succ' (u : ℕ → Set α) (n : ℕ) : ⋂ k ≤ n + 1, u k = u 0 ∩ ⋂ k ≤ n, u (k + 1) :=
  Nat.iInf_le_succ' u n


