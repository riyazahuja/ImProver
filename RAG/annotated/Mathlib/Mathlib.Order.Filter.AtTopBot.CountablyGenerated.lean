instance (priority := 200) atTop.isCountablyGenerated [Preorder α] [Countable α] :
    (atTop : Filter <| α).IsCountablyGenerated :=
  isCountablyGenerated_seq _


instance (priority := 200) atBot.isCountablyGenerated [Preorder α] [Countable α] :
    (atBot : Filter <| α).IsCountablyGenerated :=
  isCountablyGenerated_seq _


instance instIsCountablyGeneratedAtTopProd [Preorder α] [IsCountablyGenerated (atTop : Filter α)]
    [Preorder β] [IsCountablyGenerated (atTop : Filter β)] :
    IsCountablyGenerated (atTop : Filter (α × β)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Filter.atTop.IsCountablyGenerated
    inst✝¹ : Preorder β
    inst✝ : Filter.atTop.IsCountablyGenerated
    ⊢ Filter.atTop.IsCountablyGenerated
  -/
  rw [← prod_atTop_atTop_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Filter.atTop.IsCountablyGenerated
    inst✝¹ : Preorder β
    inst✝ : Filter.atTop.IsCountablyGenerated
    ⊢ (SProd.sprod Filter.atTop Filter.atTop).IsCountablyGenerated
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance instIsCountablyGeneratedAtBotProd [Preorder α] [IsCountablyGenerated (atBot : Filter α)]
    [Preorder β] [IsCountablyGenerated (atBot : Filter β)] :
    IsCountablyGenerated (atBot : Filter (α × β)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Filter.atBot.IsCountablyGenerated
    inst✝¹ : Preorder β
    inst✝ : Filter.atBot.IsCountablyGenerated
    ⊢ Filter.atBot.IsCountablyGenerated
  -/
  rw [← prod_atBot_atBot_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Filter.atBot.IsCountablyGenerated
    inst✝¹ : Preorder β
    inst✝ : Filter.atBot.IsCountablyGenerated
    ⊢ (SProd.sprod Filter.atBot Filter.atBot).IsCountablyGenerated
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance _root_.OrderDual.instIsCountablyGeneratedAtTop [Preorder α]
    [IsCountablyGenerated (atBot : Filter α)] : IsCountablyGenerated (atTop : Filter αᵒᵈ) := ‹_›


instance _root_.OrderDual.instIsCountablyGeneratedAtBot [Preorder α]
    [IsCountablyGenerated (atTop : Filter α)] : IsCountablyGenerated (atBot : Filter αᵒᵈ) := ‹_›


lemma atTop_countable_basis [Preorder α] [IsDirected α (· ≤ ·)] [Nonempty α] [Countable α] :
    HasCountableBasis (atTop : Filter α) (fun _ => True) Ici :=
  { atTop_basis with countable := to_countable _ }


lemma atBot_countable_basis [Preorder α] [IsDirected α (· ≥ ·)] [Nonempty α] [Countable α] :
    HasCountableBasis (atBot : Filter α) (fun _ => True) Iic :=
  { atBot_basis with countable := to_countable _ }


/-- If `f` is a nontrivial countably generated filter, then there exists a sequence that converges
to `f`. -/
theorem exists_seq_tendsto (f : Filter α) [IsCountablyGenerated f] [NeBot f] :
    ∃ x : ℕ → α, Tendsto x atTop f := by
  /-
    α : Type u_1
    f : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : f.NeBot
    ⊢ Exists fun x => Filter.Tendsto x Filter.atTop f
  -/
  obtain ⟨B, h⟩ := f.exists_antitone_basis
  /-
    case intro
    α : Type u_1
    f : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : f.NeBot
    B : Nat → Set α
    h : f.HasAntitoneBasis B
    ⊢ Exists fun x => Filter.Tendsto x Filter.atTop f
  -/
  choose x hx using fun n => Filter.nonempty_of_mem (h.mem n)
  /-
    case intro
    α : Type u_1
    f : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : f.NeBot
    B : Nat → Set α
    h : f.HasAntitoneBasis B
    x : Nat → α
    hx : ∀ (n : Nat), Membership.mem (B n) (x n)
    ⊢ Exists fun x => Filter.Tendsto x Filter.atTop f
  -/
  exact ⟨x, h.tendsto hx⟩
  /-
    🎉 no goals
  -/


theorem exists_seq_monotone_tendsto_atTop_atTop (α : Type*) [Preorder α] [Nonempty α]
    [IsDirected α (· ≤ ·)] [(atTop : Filter α).IsCountablyGenerated] :
    ∃ xs : ℕ → α, Monotone xs ∧ Tendsto xs atTop atTop := by
  /-
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    ⊢ Exists fun xs => And (Monotone xs) (Filter.Tendsto xs Filter.atTop Filter.at …
  -/
  obtain ⟨ys, h⟩ := exists_seq_tendsto (atTop : Filter α)
  /-
    case intro
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    ys : Nat → α
    h : Filter.Tendsto ys Filter.atTop Filter.atTop
    ⊢ Exists fun xs => And (Monotone xs) (Filter.Tendsto xs Filter.atTop Filter.at …
  -/
  choose c hleft hright using exists_ge_ge (α := α)
  /-
    case intro
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    ys : Nat → α
    h : Filter.Tendsto ys Filter.atTop Filter.atTop
    c : α → α → α
    hleft : ∀ (a b : α), LE.le a (c a b)
    hright : ∀ (a b : α), LE.le b (c a b)
    ⊢ Exists fun xs => And (Monotone xs) (Filter.Tendsto xs Filter.atTop Filter.at …
  -/
  set xs : ℕ → α := fun n => (List.range n).foldl (fun x n ↦ c x (ys n)) (ys 0)
  /-
    case intro
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    ys : Nat → α
    h : Filter.Tendsto ys Filter.atTop Filter.atTop
    c : α → α → α
    hleft : ∀ (a b : α), LE.le a (c a b)
    hright : ∀ (a b : α), LE.le b (c a b)
    xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
    ⊢ Exists fun xs => And (Monotone xs) (Filter.Tendsto xs Filter.atTop Filter.at …
  -/
  have hsucc (n : ℕ) : xs (n + 1) = c (xs n) (ys n) := by simp [xs, List.range_succ]
  /-
    case intro
    α : Type u_3
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    ys : Nat → α
    h : Filter.Tendsto ys Filter.atTop Filter.atTop
    c : α → α → α
    hleft : ∀ (a b : α), LE.le a (c a b)
    hright : ∀ (a b : α), LE.le b (c a b)
    xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
    hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
    ⊢ Exists fun xs => And (Monotone xs) (Filter.Tendsto xs Filter.atTop Filter.at …
  -/
  refine ⟨xs, ?_, ?_⟩
    /-
      case intro.refine_1
      α : Type u_3
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
      inst✝ : Filter.atTop.IsCountablyGenerated
      ys : Nat → α
      h : Filter.Tendsto ys Filter.atTop Filter.atTop
      c : α → α → α
      hleft : ∀ (a b : α), LE.le a (c a b)
      hright : ∀ (a b : α), LE.le b (c a b)
      xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
      hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
      ⊢ Monotone xs
    -/
  · refine monotone_nat_of_le_succ fun n ↦ ?_
    /-
      case intro.refine_1
      α : Type u_3
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
      inst✝ : Filter.atTop.IsCountablyGenerated
      ys : Nat → α
      h : Filter.Tendsto ys Filter.atTop Filter.atTop
      c : α → α → α
      hleft : ∀ (a b : α), LE.le a (c a b)
      hright : ∀ (a b : α), LE.le b (c a b)
      xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
      hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
      n : Nat
      ⊢ LE.le (xs n) (xs (HAdd.hAdd n 1))
    -/
    rw [hsucc]
    /-
      case intro.refine_1
      α : Type u_3
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
      inst✝ : Filter.atTop.IsCountablyGenerated
      ys : Nat → α
      h : Filter.Tendsto ys Filter.atTop Filter.atTop
      c : α → α → α
      hleft : ∀ (a b : α), LE.le a (c a b)
      hright : ∀ (a b : α), LE.le b (c a b)
      xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
      hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
      n : Nat
      ⊢ LE.le (xs n) (c (xs n) (ys n))
    -/
    apply hleft
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_3
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
      inst✝ : Filter.atTop.IsCountablyGenerated
      ys : Nat → α
      h : Filter.Tendsto ys Filter.atTop Filter.atTop
      c : α → α → α
      hleft : ∀ (a b : α), LE.le a (c a b)
      hright : ∀ (a b : α), LE.le b (c a b)
      xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
      hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
      ⊢ Filter.Tendsto xs Filter.atTop Filter.atTop
    -/
  · refine (tendsto_add_atTop_iff_nat 1).1 <| tendsto_atTop_mono (fun n ↦ ?_) h
    /-
      case intro.refine_2
      α : Type u_3
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
      inst✝ : Filter.atTop.IsCountablyGenerated
      ys : Nat → α
      h : Filter.Tendsto ys Filter.atTop Filter.atTop
      c : α → α → α
      hleft : ∀ (a b : α), LE.le a (c a b)
      hright : ∀ (a b : α), LE.le b (c a b)
      xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
      hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
      n : Nat
      ⊢ LE.le (ys n) (xs (HAdd.hAdd n 1))
    -/
    rw [hsucc]
    /-
      case intro.refine_2
      α : Type u_3
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
      inst✝ : Filter.atTop.IsCountablyGenerated
      ys : Nat → α
      h : Filter.Tendsto ys Filter.atTop Filter.atTop
      c : α → α → α
      hleft : ∀ (a b : α), LE.le a (c a b)
      hright : ∀ (a b : α), LE.le b (c a b)
      xs : Nat → α := fun n => List.foldl (fun x n => c x (ys n)) (ys 0) (List.range …
      hsucc : ∀ (n : Nat), Eq (xs (HAdd.hAdd n 1)) (c (xs n) (ys n))
      n : Nat
      ⊢ LE.le (ys n) (c (xs n) (ys n))
    -/
    apply hright
    /-
      🎉 no goals
    -/


theorem exists_seq_antitone_tendsto_atTop_atBot (α : Type*) [Preorder α] [Nonempty α]
    [IsDirected α (· ≥ ·)] [(atBot : Filter α).IsCountablyGenerated] :
    ∃ xs : ℕ → α, Antitone xs ∧ Tendsto xs atTop atBot :=
  exists_seq_monotone_tendsto_atTop_atTop αᵒᵈ


/-- An abstract version of continuity of sequentially continuous functions on metric spaces:
if a filter `k` is countably generated then `Tendsto f k l` iff for every sequence `u`
converging to `k`, `f ∘ u` tends to `l`. -/
theorem tendsto_iff_seq_tendsto {f : α → β} {k : Filter α} {l : Filter β} [k.IsCountablyGenerated] :
    Tendsto f k l ↔ ∀ x : ℕ → α, Tendsto x atTop k → Tendsto (f ∘ x) atTop l := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    ⊢ Iff (Filter.Tendsto f k l) (∀ (x : Nat → α), Filter.Tendsto x Filter.atTop k …
  -/
  refine ⟨fun h x hx => h.comp hx, fun H s hs => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    H : ∀ (x : Nat → α), Filter.Tendsto x Filter.atTop k → Filter.Tendsto (Functio …
    s : Set β
    hs : Membership.mem l s
    ⊢ Membership.mem (Filter.map f k) s
  -/
  contrapose! H
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    s : Set β
    hs : Membership.mem l s
    H : Not (Membership.mem (Filter.map f k) s)
    ⊢ Exists fun x => And (Filter.Tendsto x Filter.atTop k) (Not (Filter.Tendsto ( …
  -/
  have : NeBot (k ⊓ 𝓟 (f ⁻¹' sᶜ)) := by simpa [neBot_iff, inf_principal_eq_bot]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    s : Set β
    hs : Membership.mem l s
    H : Not (Membership.mem (Filter.map f k) s)
    this : (Min.min k (Filter.principal (Set.preimage f (HasCompl.compl s)))).NeBot
    ⊢ Exists fun x => And (Filter.Tendsto x Filter.atTop k) (Not (Filter.Tendsto ( …
  -/
  rcases (k ⊓ 𝓟 (f ⁻¹' sᶜ)).exists_seq_tendsto with ⟨x, hx⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    s : Set β
    hs : Membership.mem l s
    H : Not (Membership.mem (Filter.map f k) s)
    this : (Min.min k (Filter.principal (Set.preimage f (HasCompl.compl s)))).NeBot
    x : Nat → α
    hx : Filter.Tendsto x Filter.atTop (Min.min k (Filter.principal (Set.preimage  …
    ⊢ Exists fun x => And (Filter.Tendsto x Filter.atTop k) (Not (Filter.Tendsto ( …
  -/
  rw [tendsto_inf, tendsto_principal] at hx
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    s : Set β
    hs : Membership.mem l s
    H : Not (Membership.mem (Filter.map f k) s)
    this : (Min.min k (Filter.principal (Set.preimage f (HasCompl.compl s)))).NeBot
    x : Nat → α
    hx : And (Filter.Tendsto x Filter.atTop k) (Filter.Eventually (fun a => Member …
    ⊢ Exists fun x => And (Filter.Tendsto x Filter.atTop k) (Not (Filter.Tendsto ( …
  -/
  refine ⟨x, hx.1, fun h => ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    s : Set β
    hs : Membership.mem l s
    H : Not (Membership.mem (Filter.map f k) s)
    this : (Min.min k (Filter.principal (Set.preimage f (HasCompl.compl s)))).NeBot
    x : Nat → α
    hx : And (Filter.Tendsto x Filter.atTop k) (Filter.Eventually (fun a => Member …
    h : Filter.Tendsto (Function.comp f x) Filter.atTop l
    ⊢ False
  -/
  rcases (hx.2.and (h hs)).exists with ⟨N, hnmem, hmem⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    k : Filter α
    l : Filter β
    inst✝ : k.IsCountablyGenerated
    s : Set β
    hs : Membership.mem l s
    H : Not (Membership.mem (Filter.map f k) s)
    this : (Min.min k (Filter.principal (Set.preimage f (HasCompl.compl s)))).NeBot
    x : Nat → α
    hx : And (Filter.Tendsto x Filter.atTop k) (Filter.Eventually (fun a => Member …
    h : Filter.Tendsto (Function.comp f x) Filter.atTop l
    N : Nat
    hnmem : Membership.mem (Set.preimage f (HasCompl.compl s)) (x N)
    hmem : Membership.mem s (Function.comp f x N)
    ⊢ False
  -/
  exact hnmem hmem
  /-
    🎉 no goals
  -/


theorem tendsto_of_seq_tendsto {f : α → β} {k : Filter α} {l : Filter β} [k.IsCountablyGenerated] :
    (∀ x : ℕ → α, Tendsto x atTop k → Tendsto (f ∘ x) atTop l) → Tendsto f k l :=
  tendsto_iff_seq_tendsto.2


theorem eventually_iff_seq_eventually {ι : Type*} {l : Filter ι} {p : ι → Prop}
    [l.IsCountablyGenerated] :
    (∀ᶠ n in l, p n) ↔ ∀ x : ℕ → ι, Tendsto x atTop l → ∀ᶠ n : ℕ in atTop, p (x n) := by
  /-
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    ⊢ Iff (Filter.Eventually (fun n => p n) l) (∀ (x : Nat → ι), Filter.Tendsto x  …
  -/
  simpa using tendsto_iff_seq_tendsto (f := id) (l := 𝓟 {x | p x})
  /-
    🎉 no goals
  -/


theorem frequently_iff_seq_frequently {ι : Type*} {l : Filter ι} {p : ι → Prop}
    [l.IsCountablyGenerated] :
    (∃ᶠ n in l, p n) ↔ ∃ x : ℕ → ι, Tendsto x atTop l ∧ ∃ᶠ n : ℕ in atTop, p (x n) := by
  /-
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    ⊢ Iff (Filter.Frequently (fun n => p n) l) (Exists fun x => And (Filter.Tendst …
  -/
  simp only [Filter.Frequently, eventually_iff_seq_eventually (l := l)]
  /-
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    ⊢ Iff (Not (∀ (x : Nat → ι), Filter.Tendsto x Filter.atTop l → Filter.Eventual …
  -/
  push_neg; rfl
            /-
              🎉 no goals
            -/


theorem exists_seq_forall_of_frequently {ι : Type*} {l : Filter ι} {p : ι → Prop}
    [l.IsCountablyGenerated] (h : ∃ᶠ n in l, p n) :
    ∃ ns : ℕ → ι, Tendsto ns atTop l ∧ ∀ n, p (ns n) := by
  /-
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    h : Filter.Frequently (fun n => p n) l
    ⊢ Exists fun ns => And (Filter.Tendsto ns Filter.atTop l) (∀ (n : Nat), p (ns  …
  -/
  rw [frequently_iff_seq_frequently] at h
  /-
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    h : Exists fun x => And (Filter.Tendsto x Filter.atTop l) (Filter.Frequently ( …
    ⊢ Exists fun ns => And (Filter.Tendsto ns Filter.atTop l) (∀ (n : Nat), p (ns  …
  -/
  obtain ⟨x, hx_tendsto, hx_freq⟩ := h
  /-
    case intro.intro
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    x : Nat → ι
    hx_tendsto : Filter.Tendsto x Filter.atTop l
    hx_freq : Filter.Frequently (fun n => p (x n)) Filter.atTop
    ⊢ Exists fun ns => And (Filter.Tendsto ns Filter.atTop l) (∀ (n : Nat), p (ns  …
  -/
  obtain ⟨n_to_n, h_tendsto, h_freq⟩ := subseq_forall_of_frequently hx_tendsto hx_freq
  /-
    case intro.intro.intro.intro
    ι : Type u_3
    l : Filter ι
    p : ι → Prop
    inst✝ : l.IsCountablyGenerated
    x : Nat → ι
    hx_tendsto : Filter.Tendsto x Filter.atTop l
    hx_freq : Filter.Frequently (fun n => p (x n)) Filter.atTop
    n_to_n : Nat → Nat
    h_tendsto : Filter.Tendsto (fun n => x (n_to_n n)) Filter.atTop l
    h_freq : ∀ (n : Nat), p (x (n_to_n n))
    ⊢ Exists fun ns => And (Filter.Tendsto ns Filter.atTop l) (∀ (n : Nat), p (ns  …
  -/
  exact ⟨x ∘ n_to_n, h_tendsto, h_freq⟩
  /-
    🎉 no goals
  -/


lemma frequently_iff_seq_forall {ι : Type*} {l : Filter ι} {p : ι → Prop}
    [l.IsCountablyGenerated] :
    (∃ᶠ n in l, p n) ↔ ∃ ns : ℕ → ι, Tendsto ns atTop l ∧ ∀ n, p (ns n) :=
  ⟨exists_seq_forall_of_frequently, fun ⟨_ns, hnsl, hpns⟩ ↦
    hnsl.frequently <| Frequently.of_forall hpns⟩


/-- A sequence converges if every subsequence has a convergent subsequence. -/
theorem tendsto_of_subseq_tendsto {ι : Type*} {x : ι → α} {f : Filter α} {l : Filter ι}
    [l.IsCountablyGenerated]
    (hxy : ∀ ns : ℕ → ι, Tendsto ns atTop l →
      ∃ ms : ℕ → ℕ, Tendsto (fun n => x (ns <| ms n)) atTop f) :
    Tendsto x l f := by
  /-
    α : Type u_1
    ι : Type u_3
    x : ι → α
    f : Filter α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    hxy : ∀ (ns : Nat → ι), Filter.Tendsto ns Filter.atTop l → Exists fun ms => Fi …
    ⊢ Filter.Tendsto x l f
  -/
  contrapose! hxy
  obtain ⟨s, hs, hfreq⟩ : ∃ s ∈ f, ∃ᶠ n in l, x n ∉ s := by
    rwa [not_tendsto_iff_exists_frequently_nmem] at hxy
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_3
    x : ι → α
    f : Filter α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    hxy : Not (Filter.Tendsto x l f)
    s : Set α
    hs : Membership.mem f s
    hfreq : Filter.Frequently (fun n => Not (Membership.mem s (x n))) l
    ⊢ Exists fun ns => And (Filter.Tendsto ns Filter.atTop l) (∀ (ms : Nat → Nat), …
  -/
  obtain ⟨y, hy_tendsto, hy_freq⟩ := exists_seq_forall_of_frequently hfreq
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_3
    x : ι → α
    f : Filter α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    hxy : Not (Filter.Tendsto x l f)
    s : Set α
    hs : Membership.mem f s
    hfreq : Filter.Frequently (fun n => Not (Membership.mem s (x n))) l
    y : Nat → ι
    hy_tendsto : Filter.Tendsto y Filter.atTop l
    hy_freq : ∀ (n : Nat), Not (Membership.mem s (x (y n)))
    ⊢ Exists fun ns => And (Filter.Tendsto ns Filter.atTop l) (∀ (ms : Nat → Nat), …
  -/
  refine ⟨y, hy_tendsto, fun ms hms_tendsto ↦ ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_3
    x : ι → α
    f : Filter α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    hxy : Not (Filter.Tendsto x l f)
    s : Set α
    hs : Membership.mem f s
    hfreq : Filter.Frequently (fun n => Not (Membership.mem s (x n))) l
    y : Nat → ι
    hy_tendsto : Filter.Tendsto y Filter.atTop l
    hy_freq : ∀ (n : Nat), Not (Membership.mem s (x (y n)))
    ms : Nat → Nat
    hms_tendsto : Filter.Tendsto (fun n => x (y (ms n))) Filter.atTop f
    ⊢ False
  -/
  rcases (hms_tendsto.eventually_mem hs).exists with ⟨n, hn⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    ι : Type u_3
    x : ι → α
    f : Filter α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    hxy : Not (Filter.Tendsto x l f)
    s : Set α
    hs : Membership.mem f s
    hfreq : Filter.Frequently (fun n => Not (Membership.mem s (x n))) l
    y : Nat → ι
    hy_tendsto : Filter.Tendsto y Filter.atTop l
    hy_freq : ∀ (n : Nat), Not (Membership.mem s (x (y n)))
    ms : Nat → Nat
    hms_tendsto : Filter.Tendsto (fun n => x (y (ms n))) Filter.atTop f
    n : Nat
    hn : Membership.mem s (x (y (ms n)))
    ⊢ False
  -/
  exact absurd hn <| hy_freq _
  /-
    🎉 no goals
  -/


theorem subseq_tendsto_of_neBot {f : Filter α} [IsCountablyGenerated f] {u : ℕ → α}
    (hx : NeBot (f ⊓ map u atTop)) : ∃ θ : ℕ → ℕ, StrictMono θ ∧ Tendsto (u ∘ θ) atTop f := by
  /-
    α : Type u_1
    f : Filter α
    inst✝ : f.IsCountablyGenerated
    u : Nat → α
    hx : (Min.min f (Filter.map u Filter.atTop)).NeBot
    ⊢ Exists fun θ => And (StrictMono θ) (Filter.Tendsto (Function.comp u θ) Filte …
  -/
  rw [← Filter.push_pull', map_neBot_iff] at hx
  /-
    α : Type u_1
    f : Filter α
    inst✝ : f.IsCountablyGenerated
    u : Nat → α
    hx : (Min.min (Filter.comap u f) Filter.atTop).NeBot
    ⊢ Exists fun θ => And (StrictMono θ) (Filter.Tendsto (Function.comp u θ) Filte …
  -/
  rcases exists_seq_tendsto (comap u f ⊓ atTop) with ⟨φ, hφ⟩
  /-
    case intro
    α : Type u_1
    f : Filter α
    inst✝ : f.IsCountablyGenerated
    u : Nat → α
    hx : (Min.min (Filter.comap u f) Filter.atTop).NeBot
    φ : Nat → Nat
    hφ : Filter.Tendsto φ Filter.atTop (Min.min (Filter.comap u f) Filter.atTop)
    ⊢ Exists fun θ => And (StrictMono θ) (Filter.Tendsto (Function.comp u θ) Filte …
  -/
  rw [tendsto_inf, tendsto_comap_iff] at hφ
  obtain ⟨ψ, hψ, hψφ⟩ : ∃ ψ : ℕ → ℕ, StrictMono ψ ∧ StrictMono (φ ∘ ψ) :=
    strictMono_subseq_of_tendsto_atTop hφ.2
  /-
    case intro.intro.intro
    α : Type u_1
    f : Filter α
    inst✝ : f.IsCountablyGenerated
    u : Nat → α
    hx : (Min.min (Filter.comap u f) Filter.atTop).NeBot
    φ : Nat → Nat
    hφ : And (Filter.Tendsto (Function.comp u φ) Filter.atTop f) (Filter.Tendsto φ …
    ψ : Nat → Nat
    hψ : StrictMono ψ
    hψφ : StrictMono (Function.comp φ ψ)
    ⊢ Exists fun θ => And (StrictMono θ) (Filter.Tendsto (Function.comp u θ) Filte …
  -/
  exact ⟨φ ∘ ψ, hψφ, hφ.1.comp hψ.tendsto_atTop⟩
  /-
    🎉 no goals
  -/


