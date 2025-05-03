@[to_additive]
lemma mulSupport_sup [SemilatticeSup M] (f g : α → M) :
    mulSupport (fun x ↦ f x ⊔ g x) ⊆ mulSupport f ∪ mulSupport g :=
  mulSupport_binop_subset (· ⊔ ·) (sup_idem _) f g


@[to_additive]
lemma mulSupport_inf [SemilatticeInf M] (f g : α → M) :
    mulSupport (fun x ↦ f x ⊓ g x) ⊆ mulSupport f ∪ mulSupport g :=
  mulSupport_binop_subset (· ⊓ ·) (inf_idem _) f g


@[to_additive]
lemma mulSupport_max [LinearOrder M] (f g : α → M) :
    mulSupport (fun x ↦ max (f x) (g x)) ⊆ mulSupport f ∪ mulSupport g := mulSupport_sup f g


@[to_additive]
lemma mulSupport_min [LinearOrder M] (f g : α → M) :
    mulSupport (fun x ↦ min (f x) (g x)) ⊆ mulSupport f ∪ mulSupport g := mulSupport_inf f g


@[to_additive]
lemma mulSupport_iSup [ConditionallyCompleteLattice M] [Nonempty ι] (f : ι → α → M) :
    mulSupport (fun x ↦ ⨆ i, f i x) ⊆ ⋃ i, mulSupport (f i) := by
  /-
    ι : Sort u_1
    α : Type u_2
    M : Type u_3
    inst✝² : One M
    inst✝¹ : ConditionallyCompleteLattice M
    inst✝ : Nonempty ι
    f : ι → α → M
    ⊢ HasSubset.Subset (Function.mulSupport fun x => iSup fun i => f i x) (Set.iUn …
  -/
  simp only [mulSupport_subset_iff', mem_iUnion, not_exists, nmem_mulSupport]
  /-
    ι : Sort u_1
    α : Type u_2
    M : Type u_3
    inst✝² : One M
    inst✝¹ : ConditionallyCompleteLattice M
    inst✝ : Nonempty ι
    f : ι → α → M
    ⊢ ∀ (x : α), (∀ (x_1 : ι), Eq (f x_1 x) 1) → Eq (iSup fun i => f i x) 1
  -/
  intro x hx
  /-
    ι : Sort u_1
    α : Type u_2
    M : Type u_3
    inst✝² : One M
    inst✝¹ : ConditionallyCompleteLattice M
    inst✝ : Nonempty ι
    f : ι → α → M
    x : α
    hx : ∀ (x_1 : ι), Eq (f x_1 x) 1
    ⊢ Eq (iSup fun i => f i x) 1
  -/
  simp only [hx, ciSup_const]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mulSupport_iInf [ConditionallyCompleteLattice M] [Nonempty ι] (f : ι → α → M) :
    mulSupport (fun x ↦ ⨅ i, f i x) ⊆ ⋃ i, mulSupport (f i) := mulSupport_iSup (M := Mᵒᵈ) f


@[to_additive]
lemma mulIndicator_apply_le' (hfg : a ∈ s → f a ≤ y) (hg : a ∉ s → 1 ≤ y) :
    mulIndicator s f a ≤ y := by
  /-
    α : Type u_2
    M : Type u_3
    inst✝¹ : LE M
    inst✝ : One M
    s : Set α
    f : α → M
    a : α
    y : M
    hfg : Membership.mem s a → LE.le (f a) y
    hg : Not (Membership.mem s a) → LE.le 1 y
    ⊢ LE.le (s.mulIndicator f a) y
  -/
  by_cases ha : a ∈ s
    /-
      case pos
      α : Type u_2
      M : Type u_3
      inst✝¹ : LE M
      inst✝ : One M
      s : Set α
      f : α → M
      a : α
      y : M
      hfg : Membership.mem s a → LE.le (f a) y
      hg : Not (Membership.mem s a) → LE.le 1 y
      ha : Membership.mem s a
      ⊢ LE.le (s.mulIndicator f a) y
    -/
  · simpa [ha] using hfg ha
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      M : Type u_3
      inst✝¹ : LE M
      inst✝ : One M
      s : Set α
      f : α → M
      a : α
      y : M
      hfg : Membership.mem s a → LE.le (f a) y
      hg : Not (Membership.mem s a) → LE.le 1 y
      ha : Not (Membership.mem s a)
      ⊢ LE.le (s.mulIndicator f a) y
    -/
  · simpa [ha] using hg ha
    /-
      🎉 no goals
    -/


@[to_additive]
lemma mulIndicator_le' (hfg : ∀ a ∈ s, f a ≤ g a) (hg : ∀ a, a ∉ s → 1 ≤ g a) :
    mulIndicator s f ≤ g := fun _ ↦ mulIndicator_apply_le' (hfg _) (hg _)


@[to_additive]
lemma le_mulIndicator_apply (hfg : a ∈ s → y ≤ g a) (hf : a ∉ s → y ≤ 1) :
    y ≤ mulIndicator s g a := mulIndicator_apply_le' (M := Mᵒᵈ) hfg hf


@[to_additive]
lemma le_mulIndicator (hfg : ∀ a ∈ s, f a ≤ g a) (hf : ∀ a ∉ s, f a ≤ 1) :
    f ≤ mulIndicator s g := fun _ ↦ le_mulIndicator_apply (hfg _) (hf _)


@[to_additive indicator_apply_nonneg]
lemma one_le_mulIndicator_apply (h : a ∈ s → 1 ≤ f a) : 1 ≤ mulIndicator s f a :=
  le_mulIndicator_apply h fun _ ↦ le_rfl


@[to_additive indicator_nonneg]
lemma one_le_mulIndicator (h : ∀ a ∈ s, 1 ≤ f a) (a : α) : 1 ≤ mulIndicator s f a :=
  one_le_mulIndicator_apply (h a)


@[to_additive]
lemma mulIndicator_apply_le_one (h : a ∈ s → f a ≤ 1) : mulIndicator s f a ≤ 1 :=
  mulIndicator_apply_le' h fun _ ↦ le_rfl


@[to_additive]
lemma mulIndicator_le_one (h : ∀ a ∈ s, f a ≤ 1) (a : α) : mulIndicator s f a ≤ 1 :=
  mulIndicator_apply_le_one (h a)


@[to_additive]
lemma mulIndicator_le_mulIndicator' (h : a ∈ s → f a ≤ g a) :
    mulIndicator s f a ≤ mulIndicator s g a :=
  mulIndicator_rel_mulIndicator le_rfl h


@[to_additive (attr := mono, gcongr)]
lemma mulIndicator_le_mulIndicator (h : f a ≤ g a) : mulIndicator s f a ≤ mulIndicator s g a :=
  mulIndicator_rel_mulIndicator le_rfl fun _ ↦ h


@[to_additive (attr := gcongr)]
lemma mulIndicator_mono (h : f ≤ g) : s.mulIndicator f ≤ s.mulIndicator g :=
  fun _ ↦ mulIndicator_le_mulIndicator (h _)


@[to_additive]
lemma mulIndicator_le_mulIndicator_apply_of_subset (h : s ⊆ t) (hf : 1 ≤ f a) :
    mulIndicator s f a ≤ mulIndicator t f a :=
  mulIndicator_apply_le'
    (fun ha ↦ le_mulIndicator_apply (fun _ ↦ le_rfl) fun hat ↦ (hat <| h ha).elim) fun _ ↦
    one_le_mulIndicator_apply fun _ ↦ hf


@[to_additive]
lemma mulIndicator_le_mulIndicator_of_subset (h : s ⊆ t) (hf : 1 ≤ f) :
    mulIndicator s f ≤ mulIndicator t f :=
  fun _ ↦ mulIndicator_le_mulIndicator_apply_of_subset h (hf _)


@[to_additive]
lemma mulIndicator_le_self' (hf : ∀ x ∉ s, 1 ≤ f x) : mulIndicator s f ≤ f :=
  mulIndicator_le' (fun _ _ ↦ le_rfl) hf


lemma indicator_le_indicator_nonneg (s : Set α) (f : α → M) :
    s.indicator f ≤ {a | 0 ≤ f a}.indicator f := by
  /-
    α : Type u_2
    M : Type u_3
    inst✝¹ : Zero M
    inst✝ : LinearOrder M
    s : Set α
    f : α → M
    ⊢ LE.le (s.indicator f) ((setOf fun a => LE.le 0 (f a)).indicator f)
  -/
  intro a
  classical
  simp_rw [indicator_apply]
  split_ifs
  exacts [le_rfl, (not_le.1 ‹_›).le, ‹_›, le_rfl]


lemma indicator_nonpos_le_indicator (s : Set α) (f : α → M) :
    {a | f a ≤ 0}.indicator f ≤ s.indicator f :=
  indicator_le_indicator_nonneg (M := Mᵒᵈ) _ _


@[to_additive]
lemma mulIndicator_iUnion_apply (h1 : (⊥ : M) = 1) (s : ι → Set α) (f : α → M) (x : α) :
    mulIndicator (⋃ i, s i) f x = ⨆ i, mulIndicator (s i) f x := by
  /-
    ι : Sort u_1
    α : Type u_2
    M : Type u_3
    inst✝¹ : CompleteLattice M
    inst✝ : One M
    h1 : Eq Bot.bot 1
    s : ι → Set α
    f : α → M
    x : α
    ⊢ Eq ((Set.iUnion fun i => s i).mulIndicator f x) (iSup fun i => (s i).mulIndi …
  -/
  by_cases hx : x ∈ ⋃ i, s i
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Membership.mem (Set.iUnion fun i => s i) x
      ⊢ Eq ((Set.iUnion fun i => s i).mulIndicator f x) (iSup fun i => (s i).mulIndi …
    -/
  · rw [mulIndicator_of_mem hx]
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Membership.mem (Set.iUnion fun i => s i) x
      ⊢ Eq (f x) (iSup fun i => (s i).mulIndicator f x)
    -/
    rw [mem_iUnion] at hx
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Exists fun i => Membership.mem (s i) x
      ⊢ Eq (f x) (iSup fun i => (s i).mulIndicator f x)
    -/
    refine le_antisymm ?_ (iSup_le fun i ↦ mulIndicator_le_self' (fun x _ ↦ h1 ▸ bot_le) x)
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Exists fun i => Membership.mem (s i) x
      ⊢ LE.le (f x) (iSup fun i => (s i).mulIndicator f x)
    -/
    rcases hx with ⟨i, hi⟩
    /-
      case pos.intro
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      i : ι
      hi : Membership.mem (s i) x
      ⊢ LE.le (f x) (iSup fun i => (s i).mulIndicator f x)
    -/
    exact le_iSup_of_le i (ge_of_eq <| mulIndicator_of_mem hi _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Not (Membership.mem (Set.iUnion fun i => s i) x)
      ⊢ Eq ((Set.iUnion fun i => s i).mulIndicator f x) (iSup fun i => (s i).mulIndi …
    -/
  · rw [mulIndicator_of_not_mem hx]
    /-
      case neg
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Not (Membership.mem (Set.iUnion fun i => s i) x)
      ⊢ Eq 1 (iSup fun i => (s i).mulIndicator f x)
    -/
    simp only [mem_iUnion, not_exists] at hx
    /-
      case neg
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝¹ : CompleteLattice M
      inst✝ : One M
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : ∀ (x_1 : ι), Not (Membership.mem (s x_1) x)
      ⊢ Eq 1 (iSup fun i => (s i).mulIndicator f x)
    -/
    simp [hx, ← h1]
    /-
      🎉 no goals
    -/


@[to_additive]
lemma mulIndicator_iInter_apply (h1 : (⊥ : M) = 1) (s : ι → Set α) (f : α → M) (x : α) :
    mulIndicator (⋂ i, s i) f x = ⨅ i, mulIndicator (s i) f x := by
  /-
    ι : Sort u_1
    α : Type u_2
    M : Type u_3
    inst✝² : CompleteLattice M
    inst✝¹ : One M
    inst✝ : Nonempty ι
    h1 : Eq Bot.bot 1
    s : ι → Set α
    f : α → M
    x : α
    ⊢ Eq ((Set.iInter fun i => s i).mulIndicator f x) (iInf fun i => (s i).mulIndi …
  -/
  by_cases hx : x ∈ ⋂ i, s i
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Membership.mem (Set.iInter fun i => s i) x
      ⊢ Eq ((Set.iInter fun i => s i).mulIndicator f x) (iInf fun i => (s i).mulIndi …
    -/
  · rw [mulIndicator_of_mem hx]
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Membership.mem (Set.iInter fun i => s i) x
      ⊢ Eq (f x) (iInf fun i => (s i).mulIndicator f x)
    -/
    rw [mem_iInter] at hx
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : ∀ (i : ι), Membership.mem (s i) x
      ⊢ Eq (f x) (iInf fun i => (s i).mulIndicator f x)
    -/
    refine le_antisymm ?_ (by simp only [mulIndicator_of_mem (hx _), ciInf_const, le_refl])
    /-
      case pos
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : ∀ (i : ι), Membership.mem (s i) x
      ⊢ LE.le (f x) (iInf fun i => (s i).mulIndicator f x)
    -/
    exact le_iInf (fun j ↦ by simp only [mulIndicator_of_mem (hx j), le_refl])
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Not (Membership.mem (Set.iInter fun i => s i) x)
      ⊢ Eq ((Set.iInter fun i => s i).mulIndicator f x) (iInf fun i => (s i).mulIndi …
    -/
  · rw [mulIndicator_of_not_mem hx]
    /-
      case neg
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Not (Membership.mem (Set.iInter fun i => s i) x)
      ⊢ Eq 1 (iInf fun i => (s i).mulIndicator f x)
    -/
    simp only [mem_iInter, not_exists, not_forall] at hx
    /-
      case neg
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      hx : Exists fun x_1 => Not (Membership.mem (s x_1) x)
      ⊢ Eq 1 (iInf fun i => (s i).mulIndicator f x)
    -/
    rcases hx with ⟨j, hj⟩
    /-
      case neg.intro
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      j : ι
      hj : Not (Membership.mem (s j) x)
      ⊢ Eq 1 (iInf fun i => (s i).mulIndicator f x)
    -/
    refine le_antisymm (by simp only [← h1, le_iInf_iff, bot_le, forall_const]) ?_
    /-
      case neg.intro
      ι : Sort u_1
      α : Type u_2
      M : Type u_3
      inst✝² : CompleteLattice M
      inst✝¹ : One M
      inst✝ : Nonempty ι
      h1 : Eq Bot.bot 1
      s : ι → Set α
      f : α → M
      x : α
      j : ι
      hj : Not (Membership.mem (s j) x)
      ⊢ LE.le (iInf fun i => (s i).mulIndicator f x) 1
    -/
    simpa [mulIndicator_of_not_mem hj] using (iInf_le (fun i ↦ (s i).mulIndicator f) j) x
    /-
      🎉 no goals
    -/


@[to_additive]
lemma iSup_mulIndicator {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {f : ι → α → M}
    {s : ι → Set α} (h1 : (⊥ : M) = 1) (hf : Monotone f) (hs : Monotone s) :
    ⨆ i, (s i).mulIndicator (f i) = (⋃ i, s i).mulIndicator (⨆ i, f i) := by
  /-
    α : Type u_2
    M : Type u_3
    inst✝³ : CompleteLattice M
    inst✝² : One M
    ι : Type u_4
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f : ι → α → M
    s : ι → Set α
    h1 : Eq Bot.bot 1
    hf : Monotone f
    hs : Monotone s
    ⊢ Eq (iSup fun i => (s i).mulIndicator (f i)) ((Set.iUnion fun i => s i).mulIn …
  -/
  simp only [le_antisymm_iff, iSup_le_iff]
  refine ⟨fun i ↦ (mulIndicator_mono (le_iSup _ _)).trans (mulIndicator_le_mulIndicator_of_subset
    (subset_iUnion _ _) (fun _ ↦ by simp [← h1])), fun a ↦ ?_⟩
  /-
    α : Type u_2
    M : Type u_3
    inst✝³ : CompleteLattice M
    inst✝² : One M
    ι : Type u_4
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f : ι → α → M
    s : ι → Set α
    h1 : Eq Bot.bot 1
    hf : Monotone f
    hs : Monotone s
    a : α
    ⊢ LE.le ((Set.iUnion fun i => s i).mulIndicator (iSup fun i => f i) a) (iSup ( …
  -/
  by_cases ha : a ∈ ⋃ i, s i
    /-
      case pos
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Membership.mem (Set.iUnion fun i => s i) a
      ⊢ LE.le ((Set.iUnion fun i => s i).mulIndicator (iSup fun i => f i) a) (iSup ( …
    -/
  · obtain ⟨i, hi⟩ : ∃ i, a ∈ s i := by simpa using ha
    /-
      case pos.intro
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Membership.mem (Set.iUnion fun i => s i) a
      i : ι
      hi : Membership.mem (s i) a
      ⊢ LE.le ((Set.iUnion fun i => s i).mulIndicator (iSup fun i => f i) a) (iSup ( …
    -/
    rw [mulIndicator_of_mem ha, iSup_apply, iSup_apply]
    /-
      case pos.intro
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Membership.mem (Set.iUnion fun i => s i) a
      i : ι
      hi : Membership.mem (s i) a
      ⊢ LE.le (iSup fun i => f i a) (iSup fun i => (s i).mulIndicator (f i) a)
    -/
    refine iSup_le fun j ↦ ?_
    /-
      case pos.intro
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Membership.mem (Set.iUnion fun i => s i) a
      i : ι
      hi : Membership.mem (s i) a
      j : ι
      ⊢ LE.le (f j a) (iSup fun i => (s i).mulIndicator (f i) a)
    -/
    obtain ⟨k, hik, hjk⟩ := exists_ge_ge i j
    /-
      case pos.intro.intro.intro
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Membership.mem (Set.iUnion fun i => s i) a
      i : ι
      hi : Membership.mem (s i) a
      j k : ι
      hik : LE.le i k
      hjk : LE.le j k
      ⊢ LE.le (f j a) (iSup fun i => (s i).mulIndicator (f i) a)
    -/
    refine le_iSup_of_le k <| (hf hjk _).trans_eq ?_
    /-
      case pos.intro.intro.intro
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Membership.mem (Set.iUnion fun i => s i) a
      i : ι
      hi : Membership.mem (s i) a
      j k : ι
      hik : LE.le i k
      hjk : LE.le j k
      ⊢ Eq (f k a) ((s k).mulIndicator (f k) a)
    -/
    rw [mulIndicator_of_mem (hs hik hi)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Not (Membership.mem (Set.iUnion fun i => s i) a)
      ⊢ LE.le ((Set.iUnion fun i => s i).mulIndicator (iSup fun i => f i) a) (iSup ( …
    -/
  · rw [mulIndicator_of_not_mem ha, ← h1]
    /-
      case neg
      α : Type u_2
      M : Type u_3
      inst✝³ : CompleteLattice M
      inst✝² : One M
      ι : Type u_4
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → α → M
      s : ι → Set α
      h1 : Eq Bot.bot 1
      hf : Monotone f
      hs : Monotone s
      a : α
      ha : Not (Membership.mem (Set.iUnion fun i => s i) a)
      ⊢ LE.le Bot.bot (iSup (fun i => (s i).mulIndicator (f i)) a)
    -/
    exact bot_le
    /-
      🎉 no goals
    -/


@[to_additive]
lemma mulIndicator_le_self (s : Set α) (f : α → M) : mulIndicator s f ≤ f :=
  mulIndicator_le_self' fun _ _ ↦ one_le _


@[to_additive]
lemma mulIndicator_apply_le {a : α} {s : Set α} {f g : α → M} (hfg : a ∈ s → f a ≤ g a) :
    mulIndicator s f a ≤ g a :=
  mulIndicator_apply_le' hfg fun _ ↦ one_le _


@[to_additive]
lemma mulIndicator_le {s : Set α} {f g : α → M} (hfg : ∀ a ∈ s, f a ≤ g a) :
    mulIndicator s f ≤ g :=
  mulIndicator_le' hfg fun _ _ ↦ one_le _


@[to_additive]
lemma mabs_mulIndicator_symmDiff (s t : Set α) (f : α → M) (x : α) :
    |mulIndicator (s ∆ t) f x|ₘ = |mulIndicator s f x / mulIndicator t f x|ₘ :=
  apply_mulIndicator_symmDiff mabs_inv s t f x


