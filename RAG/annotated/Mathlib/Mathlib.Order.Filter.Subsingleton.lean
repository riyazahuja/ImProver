/-- We say that a filter is a *subsingleton* if there exists a subsingleton set
that belongs to the filter. -/
protected def Subsingleton (l : Filter α) : Prop := ∃ s ∈ l, Set.Subsingleton s


theorem HasBasis.subsingleton_iff {ι : Sort*} {p : ι → Prop} {s : ι → Set α} (h : l.HasBasis p s) :
    l.Subsingleton ↔ ∃ i, p i ∧ (s i).Subsingleton :=
  h.exists_iff fun _ _ hsub h ↦ h.anti hsub


theorem Subsingleton.anti {l'} (hl : l.Subsingleton) (hl' : l' ≤ l) : l'.Subsingleton :=
  let ⟨s, hsl, hs⟩ := hl; ⟨s, hl' hsl, hs⟩


@[nontriviality]
theorem Subsingleton.of_subsingleton [Subsingleton α] : l.Subsingleton :=
  ⟨univ, univ_mem, subsingleton_univ⟩


theorem Subsingleton.map (hl : l.Subsingleton) (f : α → β) : (map f l).Subsingleton :=
  let ⟨s, hsl, hs⟩ := hl; ⟨f '' s, image_mem_map hsl, hs.image f⟩


theorem Subsingleton.prod (hl : l.Subsingleton) {l' : Filter β} (hl' : l'.Subsingleton) :
    (l ×ˢ l').Subsingleton :=
  let ⟨s, hsl, hs⟩ := hl; let ⟨t, htl', ht⟩ := hl'; ⟨s ×ˢ t, prod_mem_prod hsl htl', hs.prod ht⟩


@[simp]
theorem subsingleton_pure {a : α} : Filter.Subsingleton (pure a) :=
  ⟨{a}, rfl, subsingleton_singleton⟩


@[simp]
theorem subsingleton_bot : Filter.Subsingleton (⊥ : Filter α) :=
  ⟨∅, trivial, subsingleton_empty⟩


/-- A nontrivial subsingleton filter is equal to `pure a` for some `a`. -/
theorem Subsingleton.exists_eq_pure [l.NeBot] (hl : l.Subsingleton) : ∃ a, l = pure a := by
  /-
    α : Type u_1
    l : Filter α
    inst✝ : l.NeBot
    hl : l.Subsingleton
    ⊢ Exists fun a => Eq l (Pure.pure a)
  -/
  rcases hl with ⟨s, hsl, hs⟩
  /-
    case intro.intro
    α : Type u_1
    l : Filter α
    inst✝ : l.NeBot
    s : Set α
    hsl : Membership.mem l s
    hs : s.Subsingleton
    ⊢ Exists fun a => Eq l (Pure.pure a)
  -/
  rcases exists_eq_singleton_iff_nonempty_subsingleton.2 ⟨nonempty_of_mem hsl, hs⟩ with ⟨a, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_1
    l : Filter α
    inst✝ : l.NeBot
    a : α
    hsl : Membership.mem l (Singleton.singleton a)
    hs : (Singleton.singleton a).Subsingleton
    ⊢ Exists fun a => Eq l (Pure.pure a)
  -/
  refine ⟨a, (NeBot.le_pure_iff ‹_›).1 ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    l : Filter α
    inst✝ : l.NeBot
    a : α
    hsl : Membership.mem l (Singleton.singleton a)
    hs : (Singleton.singleton a).Subsingleton
    ⊢ LE.le l (Pure.pure a)
  -/
  rwa [le_pure_iff]
  /-
    🎉 no goals
  -/


/-- A filter is a subsingleton iff it is equal to `⊥` or to `pure a` for some `a`. -/
theorem subsingleton_iff_bot_or_pure : l.Subsingleton ↔ l = ⊥ ∨ ∃ a, l = pure a := by
  /-
    α : Type u_1
    l : Filter α
    ⊢ Iff l.Subsingleton (Or (Eq l Bot.bot) (Exists fun a => Eq l (Pure.pure a)))
  -/
  refine ⟨fun hl ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      l : Filter α
      hl : l.Subsingleton
      ⊢ Or (Eq l Bot.bot) (Exists fun a => Eq l (Pure.pure a))
    -/
  · exact (eq_or_neBot l).imp_right (@Subsingleton.exists_eq_pure _ _ · hl)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      l : Filter α
      ⊢ Or (Eq l Bot.bot) (Exists fun a => Eq l (Pure.pure a)) → l.Subsingleton
    -/
                                /-
                                  🎉 no goals
                                -/
  · rintro (rfl | ⟨a, rfl⟩) <;> simp
                                /-
                                  🎉 no goals
                                -/


/-- In a nonempty type, a filter is a subsingleton iff
it is less than or equal to a pure filter. -/
theorem subsingleton_iff_exists_le_pure [Nonempty α] : l.Subsingleton ↔ ∃ a, l ≤ pure a := by
  /-
    α : Type u_1
    l : Filter α
    inst✝ : Nonempty α
    ⊢ Iff l.Subsingleton (Exists fun a => LE.le l (Pure.pure a))
  -/
  rcases eq_or_neBot l with rfl | hbot
    /-
      case inl
      α : Type u_1
      inst✝ : Nonempty α
      ⊢ Iff Bot.bot.Subsingleton (Exists fun a => LE.le Bot.bot (Pure.pure a))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      l : Filter α
      inst✝ : Nonempty α
      hbot : l.NeBot
      ⊢ Iff l.Subsingleton (Exists fun a => LE.le l (Pure.pure a))
    -/
  · simp [subsingleton_iff_bot_or_pure, ← hbot.le_pure_iff, hbot.ne]
    /-
      🎉 no goals
    -/


theorem subsingleton_iff_exists_singleton_mem [Nonempty α] : l.Subsingleton ↔ ∃ a, {a} ∈ l := by
  /-
    α : Type u_1
    l : Filter α
    inst✝ : Nonempty α
    ⊢ Iff l.Subsingleton (Exists fun a => Membership.mem l (Singleton.singleton a))
  -/
  simp only [subsingleton_iff_exists_le_pure, le_pure_iff]
  /-
    🎉 no goals
  -/


/-- A subsingleton filter on a nonempty type is less than or equal to `pure a` for some `a`. -/
alias ⟨Subsingleton.exists_le_pure, _⟩ := subsingleton_iff_exists_le_pure


lemma Subsingleton.isCountablyGenerated (hl : l.Subsingleton) : IsCountablyGenerated l := by
  /-
    α : Type u_1
    l : Filter α
    hl : l.Subsingleton
    ⊢ l.IsCountablyGenerated
  -/
  rcases subsingleton_iff_bot_or_pure.1 hl with rfl|⟨x, rfl⟩
    /-
      case inl
      α : Type u_1
      hl : Bot.bot.Subsingleton
      ⊢ Bot.bot.IsCountablyGenerated
    -/
  · exact isCountablyGenerated_bot
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      x : α
      hl : (Pure.pure x).Subsingleton
      ⊢ (Pure.pure x).IsCountablyGenerated
    -/
  · exact isCountablyGenerated_pure x
    /-
      🎉 no goals
    -/


