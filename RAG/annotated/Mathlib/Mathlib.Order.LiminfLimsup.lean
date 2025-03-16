/-- `f` is eventually bounded if and only if, there exists an admissible set on which it is
bounded. -/
theorem isBounded_iff : f.IsBounded r ↔ ∃ s ∈ f.sets, ∃ b, s ⊆ { x | r x b } :=
  Iff.intro (fun ⟨b, hb⟩ => ⟨{ a | r a b }, hb, b, Subset.refl _⟩) fun ⟨_, hs, b, hb⟩ =>
    ⟨b, mem_of_superset hs hb⟩


/-- A bounded function `u` is in particular eventually bounded. -/
theorem isBoundedUnder_of {f : Filter β} {u : β → α} : (∃ b, ∀ x, r (u x) b) → f.IsBoundedUnder r u
  | ⟨b, hb⟩ => ⟨b, show ∀ᶠ x in f, r (u x) b from Eventually.of_forall hb⟩


                                                         /-
                                                           α : Type u_1
                                                           r : α → α → Prop
                                                           ⊢ Iff (Filter.IsBounded r Bot.bot) (Nonempty α)
                                                         -/
theorem isBounded_bot : IsBounded r ⊥ ↔ Nonempty α := by simp [IsBounded, exists_true_iff_nonempty]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                              /-
                                                                α : Type u_1
                                                                r : α → α → Prop
                                                                ⊢ Iff (Filter.IsBounded r Top.top) (Exists fun t => ∀ (x : α), r x t)
                                                              -/
theorem isBounded_top : IsBounded r ⊤ ↔ ∃ t, ∀ x, r x t := by simp [IsBounded, eq_univ_iff_forall]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem isBounded_principal (s : Set α) : IsBounded r (𝓟 s) ↔ ∃ t, ∀ x ∈ s, r x t := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    ⊢ Iff (Filter.IsBounded r (Filter.principal s)) (Exists fun t => ∀ (x : α), Me …
  -/
  simp [IsBounded, subset_def]
  /-
    🎉 no goals
  -/


theorem isBounded_sup [IsTrans α r] [IsDirected α r] :
    IsBounded r f → IsBounded r g → IsBounded r (f ⊔ g)
  | ⟨b₁, h₁⟩, ⟨b₂, h₂⟩ =>
    let ⟨b, rb₁b, rb₂b⟩ := directed_of r b₁ b₂
    ⟨b, eventually_sup.mpr
      ⟨h₁.mono fun _ h => _root_.trans h rb₁b, h₂.mono fun _ h => _root_.trans h rb₂b⟩⟩


theorem IsBounded.mono (h : f ≤ g) : IsBounded r g → IsBounded r f
  | ⟨b, hb⟩ => ⟨b, h hb⟩


theorem IsBoundedUnder.mono {f g : Filter β} {u : β → α} (h : f ≤ g) :
    g.IsBoundedUnder r u → f.IsBoundedUnder r u := fun hg => IsBounded.mono (map_mono h) hg


theorem IsBoundedUnder.mono_le [Preorder β] {l : Filter α} {u v : α → β}
    (hu : IsBoundedUnder (· ≤ ·) l u) (hv : v ≤ᶠ[l] u) : IsBoundedUnder (· ≤ ·) l v := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    l : Filter α
    u v : α → β
    hu : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l u
    hv : l.EventuallyLE v u
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l v
  -/
  apply hu.imp
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    l : Filter α
    u v : α → β
    hu : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l u
    hv : l.EventuallyLE v u
    ⊢ ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (Filt …
  -/
  exact fun b hb => (eventually_map.1 hb).mp <| hv.mono fun x => le_trans
  /-
    🎉 no goals
  -/


theorem IsBoundedUnder.mono_ge [Preorder β] {l : Filter α} {u v : α → β}
    (hu : IsBoundedUnder (· ≥ ·) l u) (hv : u ≤ᶠ[l] v) : IsBoundedUnder (· ≥ ·) l v :=
  IsBoundedUnder.mono_le (β := βᵒᵈ) hu hv


theorem isBoundedUnder_const [IsRefl α r] {l : Filter β} {a : α} : IsBoundedUnder r l fun _ => a :=
  ⟨a, eventually_map.2 <| Eventually.of_forall fun _ => refl _⟩


theorem IsBounded.isBoundedUnder {q : β → β → Prop} {u : α → β}
    (hu : ∀ a₀ a₁, r a₀ a₁ → q (u a₀) (u a₁)) : f.IsBounded r → f.IsBoundedUnder q u
  | ⟨b, h⟩ => ⟨u b, show ∀ᶠ x in f, q (u x) (u b) from h.mono fun x => hu x b⟩


theorem IsBoundedUnder.comp {l : Filter γ} {q : β → β → Prop} {u : γ → α} {v : α → β}
    (hv : ∀ a₀ a₁, r a₀ a₁ → q (v a₀) (v a₁)) : l.IsBoundedUnder r u → l.IsBoundedUnder q (v ∘ u)
  | ⟨a, h⟩ => ⟨v a, show ∀ᶠ x in map u l, q (v x) (v a) from h.mono fun x => hv x a⟩


lemma IsBoundedUnder.eventually_le (h : IsBoundedUnder (· ≤ ·) f u) :
    ∃ a, ∀ᶠ x in f, u x ≤ a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder α
    f : Filter β
    u : β → α
    h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    ⊢ Exists fun a => Filter.Eventually (fun x => LE.le (u x) a) f
  -/
  obtain ⟨a, ha⟩ := h
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder α
    f : Filter β
    u : β → α
    a : α
    ha : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (Filter.map u …
    ⊢ Exists fun a => Filter.Eventually (fun x => LE.le (u x) a) f
  -/
  use a
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder α
    f : Filter β
    u : β → α
    a : α
    ha : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (Filter.map u …
    ⊢ Filter.Eventually (fun x => LE.le (u x) a) f
  -/
  exact eventually_map.1 ha
  /-
    🎉 no goals
  -/


lemma IsBoundedUnder.eventually_ge (h : IsBoundedUnder (· ≥ ·) f u) :
    ∃ a, ∀ᶠ x in f, a ≤ u x :=
  IsBoundedUnder.eventually_le (α := αᵒᵈ) h


lemma isBoundedUnder_of_eventually_le {a : α} (h : ∀ᶠ x in f, u x ≤ a) :
    IsBoundedUnder (· ≤ ·) f u := ⟨a, h⟩


lemma isBoundedUnder_of_eventually_ge {a : α} (h : ∀ᶠ x in f, a ≤ u x) :
    IsBoundedUnder (· ≥ ·) f u := ⟨a, h⟩


lemma isBoundedUnder_iff_eventually_bddAbove :
    f.IsBoundedUnder (· ≤ ·) u ↔ ∃ s, BddAbove (u '' s) ∧ ∀ᶠ x in f, x ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder α
    f : Filter β
    u : β → α
    ⊢ Iff (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) (Exists fun s =>  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : Preorder α
      f : Filter β
      u : β → α
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u → Exists fun s => And ( …
    -/
  · rintro ⟨b, hb⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      inst✝ : Preorder α
      f : Filter β
      u : β → α
      b : α
      hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map u …
      ⊢ Exists fun s => And (BddAbove (Set.image u s)) (Filter.Eventually (fun x =>  …
    -/
    exact ⟨{a | u a ≤ b}, ⟨b, by rintro _ ⟨a, ha, rfl⟩; exact ha⟩, hb⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝ : Preorder α
      f : Filter β
      u : β → α
      ⊢ (Exists fun s => And (BddAbove (Set.image u s)) (Filter.Eventually (fun x => …
    -/
  · rintro ⟨s, ⟨b, hb⟩, hs⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝ : Preorder α
      f : Filter β
      u : β → α
      s : Set β
      hs : Filter.Eventually (fun x => Membership.mem s x) f
      b : α
      hb : Membership.mem (upperBounds (Set.image u s)) b
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    -/
    exact ⟨b, hs.mono <| by simpa [upperBounds] using hb⟩
    /-
      🎉 no goals
    -/


lemma isBoundedUnder_iff_eventually_bddBelow :
    f.IsBoundedUnder (· ≥ ·) u ↔ ∃ s, BddBelow (u '' s) ∧ ∀ᶠ x in f, x ∈ s :=
  isBoundedUnder_iff_eventually_bddAbove (α := αᵒᵈ)


lemma _root_.BddAbove.isBoundedUnder (hs : s ∈ f) (hu : BddAbove (u '' s)) :
    f.IsBoundedUnder (· ≤ ·) u := isBoundedUnder_iff_eventually_bddAbove.2 ⟨_, hu, hs⟩


/-- A bounded above function `u` is in particular eventually bounded above. -/
lemma _root_.BddAbove.isBoundedUnder_of_range (hu : BddAbove (Set.range u)) :
                                                                                     /-
                                                                                       α : Type u_1
                                                                                       β : Type u_2
                                                                                       inst✝ : Preorder α
                                                                                       f : Filter β
                                                                                       u : β → α
                                                                                       hu : BddAbove (Set.range u)
                                                                                       ⊢ BddAbove (Set.image u Set.univ)
                                                                                     -/
    f.IsBoundedUnder (· ≤ ·) u := BddAbove.isBoundedUnder (s := univ) f.univ_mem (by simpa)
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma _root_.BddBelow.isBoundedUnder (hs : s ∈ f) (hu : BddBelow (u '' s)) :
    f.IsBoundedUnder (· ≥ ·) u := isBoundedUnder_iff_eventually_bddBelow.2 ⟨_, hu, hs⟩


/-- A bounded below function `u` is in particular eventually bounded below. -/
lemma _root_.BddBelow.isBoundedUnder_of_range (hu : BddBelow (Set.range u)) :
                                                                                     /-
                                                                                       α : Type u_1
                                                                                       β : Type u_2
                                                                                       inst✝ : Preorder α
                                                                                       f : Filter β
                                                                                       u : β → α
                                                                                       hu : BddBelow (Set.range u)
                                                                                       ⊢ BddBelow (Set.image u Set.univ)
                                                                                     -/
    f.IsBoundedUnder (· ≥ ·) u := BddBelow.isBoundedUnder (s := univ) f.univ_mem (by simpa)
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma IsBoundedUnder.le_of_finite [Nonempty α] [IsDirected α (· ≤ ·)] [Finite β]
    {f : Filter β} {u : β → α} : IsBoundedUnder (· ≤ ·) f u :=
  (Set.toFinite _).bddAbove.isBoundedUnder_of_range


lemma IsBoundedUnder.ge_of_finite [Nonempty α] [IsDirected α (· ≥ ·)] [Finite β]
    {f : Filter β} {u : β → α} : IsBoundedUnder (· ≥ ·) f u :=
  (Set.toFinite _).bddBelow.isBoundedUnder_of_range


theorem _root_.Monotone.isBoundedUnder_le_comp [Preorder α] [Preorder β] {l : Filter γ} {u : γ → α}
    {v : α → β} (hv : Monotone v) (hl : l.IsBoundedUnder (· ≤ ·) u) :
    l.IsBoundedUnder (· ≤ ·) (v ∘ u) :=
  hl.comp hv


theorem _root_.Monotone.isBoundedUnder_ge_comp [Preorder α] [Preorder β] {l : Filter γ} {u : γ → α}
    {v : α → β} (hv : Monotone v) (hl : l.IsBoundedUnder (· ≥ ·) u) :
    l.IsBoundedUnder (· ≥ ·) (v ∘ u) :=
  hl.comp (swap hv)


theorem _root_.Antitone.isBoundedUnder_le_comp [Preorder α] [Preorder β] {l : Filter γ} {u : γ → α}
    {v : α → β} (hv : Antitone v) (hl : l.IsBoundedUnder (· ≥ ·) u) :
    l.IsBoundedUnder (· ≤ ·) (v ∘ u) :=
  hl.comp (swap hv)


theorem _root_.Antitone.isBoundedUnder_ge_comp [Preorder α] [Preorder β] {l : Filter γ} {u : γ → α}
    {v : α → β} (hv : Antitone v) (hl : l.IsBoundedUnder (· ≤ ·) u) :
    l.IsBoundedUnder (· ≥ ·) (v ∘ u) :=
  hl.comp hv


theorem not_isBoundedUnder_of_tendsto_atTop [Preorder β] [NoMaxOrder β] {f : α → β} {l : Filter α}
    [l.NeBot] (hf : Tendsto f l atTop) : ¬IsBoundedUnder (· ≤ ·) l f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Preorder β
    inst✝¹ : NoMaxOrder β
    f : α → β
    l : Filter α
    inst✝ : l.NeBot
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l f)
  -/
  rintro ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : Preorder β
    inst✝¹ : NoMaxOrder β
    f : α → β
    l : Filter α
    inst✝ : l.NeBot
    hf : Filter.Tendsto f l Filter.atTop
    b : β
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map f …
    ⊢ False
  -/
  rw [eventually_map] at hb
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : Preorder β
    inst✝¹ : NoMaxOrder β
    f : α → β
    l : Filter α
    inst✝ : l.NeBot
    hf : Filter.Tendsto f l Filter.atTop
    b : β
    hb : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (f a) b) l
    ⊢ False
  -/
  obtain ⟨b', h⟩ := exists_gt b
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : Preorder β
    inst✝¹ : NoMaxOrder β
    f : α → β
    l : Filter α
    inst✝ : l.NeBot
    hf : Filter.Tendsto f l Filter.atTop
    b : β
    hb : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (f a) b) l
    b' : β
    h : LT.lt b b'
    ⊢ False
  -/
  have hb' := (tendsto_atTop.mp hf) b'
  have : { x : α | f x ≤ b } ∩ { x : α | b' ≤ f x } = ∅ :=
    eq_empty_of_subset_empty fun x hx => (not_le_of_lt h) (le_trans hx.2 hx.1)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : Preorder β
    inst✝¹ : NoMaxOrder β
    f : α → β
    l : Filter α
    inst✝ : l.NeBot
    hf : Filter.Tendsto f l Filter.atTop
    b : β
    hb : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (f a) b) l
    b' : β
    h : LT.lt b b'
    hb' : Filter.Eventually (fun a => LE.le b' (f a)) l
    this : Eq (Inter.inter (setOf fun x => LE.le (f x) b) (setOf fun x => LE.le b' …
    ⊢ False
  -/
  exact (nonempty_of_mem (hb.and hb')).ne_empty this
  /-
    🎉 no goals
  -/


theorem not_isBoundedUnder_of_tendsto_atBot [Preorder β] [NoMinOrder β] {f : α → β} {l : Filter α}
    [l.NeBot] (hf : Tendsto f l atBot) : ¬IsBoundedUnder (· ≥ ·) l f :=
  not_isBoundedUnder_of_tendsto_atTop (β := βᵒᵈ) hf


theorem IsBoundedUnder.bddAbove_range_of_cofinite [Preorder β] [IsDirected β (· ≤ ·)] {f : α → β}
    (hf : IsBoundedUnder (· ≤ ·) cofinite f) : BddAbove (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : IsDirected β fun x1 x2 => LE.le x1 x2
    f : α → β
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.cofinite f
    ⊢ BddAbove (Set.range f)
  -/
  rcases hf with ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : IsDirected β fun x1 x2 => LE.le x1 x2
    f : α → β
    b : β
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map f …
    ⊢ BddAbove (Set.range f)
  -/
  haveI : Nonempty β := ⟨b⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : IsDirected β fun x1 x2 => LE.le x1 x2
    f : α → β
    b : β
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map f …
    this : Nonempty β
    ⊢ BddAbove (Set.range f)
  -/
  rw [← image_univ, ← union_compl_self { x | f x ≤ b }, image_union, bddAbove_union]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : IsDirected β fun x1 x2 => LE.le x1 x2
    f : α → β
    b : β
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (Filter.map f …
    this : Nonempty β
    ⊢ And (BddAbove (Set.image f (setOf fun x => LE.le (f x) b))) (BddAbove (Set.i …
  -/
  exact ⟨⟨b, forall_mem_image.2 fun x => id⟩, (hb.image f).bddAbove⟩
  /-
    🎉 no goals
  -/


theorem IsBoundedUnder.bddBelow_range_of_cofinite [Preorder β] [IsDirected β (· ≥ ·)] {f : α → β}
    (hf : IsBoundedUnder (· ≥ ·) cofinite f) : BddBelow (range f) :=
  IsBoundedUnder.bddAbove_range_of_cofinite (β := βᵒᵈ) hf


theorem IsBoundedUnder.bddAbove_range [Preorder β] [IsDirected β (· ≤ ·)] {f : ℕ → β}
    (hf : IsBoundedUnder (· ≤ ·) atTop f) : BddAbove (range f) := by
  /-
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : IsDirected β fun x1 x2 => LE.le x1 x2
    f : Nat → β
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop f
    ⊢ BddAbove (Set.range f)
  -/
  rw [← Nat.cofinite_eq_atTop] at hf
  /-
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : IsDirected β fun x1 x2 => LE.le x1 x2
    f : Nat → β
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.cofinite f
    ⊢ BddAbove (Set.range f)
  -/
  exact hf.bddAbove_range_of_cofinite
  /-
    🎉 no goals
  -/


theorem IsBoundedUnder.bddBelow_range [Preorder β] [IsDirected β (· ≥ ·)] {f : ℕ → β}
    (hf : IsBoundedUnder (· ≥ ·) atTop f) : BddBelow (range f) :=
  IsBoundedUnder.bddAbove_range (β := βᵒᵈ) hf


/-- To check that a filter is frequently bounded, it suffices to have a witness
which bounds `f` at some point for every admissible set.

This is only an implication, as the other direction is wrong for the trivial filter. -/
theorem IsCobounded.mk [IsTrans α r] (a : α) (h : ∀ s ∈ f, ∃ x ∈ s, r a x) : f.IsCobounded r :=
  ⟨a, fun _ s =>
    let ⟨_, h₁, h₂⟩ := h _ s
    _root_.trans h₂ h₁⟩


/-- A filter which is eventually bounded is in particular frequently bounded (in the opposite
direction). At least if the filter is not trivial. -/
theorem IsBounded.isCobounded_flip [IsTrans α r] [NeBot f] : f.IsBounded r → f.IsCobounded (flip r)
  | ⟨a, ha⟩ =>
    ⟨a, fun b hb =>
      let ⟨_, rxa, rbx⟩ := (ha.and hb).exists
      show r b a from _root_.trans rbx rxa⟩


theorem IsBounded.isCobounded_ge [Preorder α] [NeBot f] (h : f.IsBounded (· ≤ ·)) :
    f.IsCobounded (· ≥ ·) :=
  h.isCobounded_flip


theorem IsBounded.isCobounded_le [Preorder α] [NeBot f] (h : f.IsBounded (· ≥ ·)) :
    f.IsCobounded (· ≤ ·) :=
  h.isCobounded_flip


theorem IsBoundedUnder.isCoboundedUnder_flip {u : γ → α} {l : Filter γ} [IsTrans α r] [NeBot l]
    (h : l.IsBoundedUnder r u) : l.IsCoboundedUnder (flip r) u :=
  h.isCobounded_flip


theorem IsBoundedUnder.isCoboundedUnder_le {u : γ → α} {l : Filter γ} [Preorder α] [NeBot l]
    (h : l.IsBoundedUnder (· ≥ ·) u) : l.IsCoboundedUnder (· ≤ ·) u :=
  h.isCoboundedUnder_flip


theorem IsBoundedUnder.isCoboundedUnder_ge {u : γ → α} {l : Filter γ} [Preorder α] [NeBot l]
    (h : l.IsBoundedUnder (· ≤ ·) u) : l.IsCoboundedUnder (· ≥ ·) u :=
  h.isCoboundedUnder_flip


lemma isCoboundedUnder_le_of_eventually_le [Preorder α] (l : Filter ι) [NeBot l] {f : ι → α} {x : α}
    (hf : ∀ᶠ i in l, x ≤ f i) :
    IsCoboundedUnder (· ≤ ·) l f :=
  IsBoundedUnder.isCoboundedUnder_le ⟨x, hf⟩


lemma isCoboundedUnder_ge_of_eventually_le [Preorder α] (l : Filter ι) [NeBot l] {f : ι → α} {x : α}
    (hf : ∀ᶠ i in l, f i ≤ x) :
    IsCoboundedUnder (· ≥ ·) l f :=
  IsBoundedUnder.isCoboundedUnder_ge ⟨x, hf⟩


lemma isCoboundedUnder_le_of_le [Preorder α] (l : Filter ι) [NeBot l] {f : ι → α} {x : α}
    (hf : ∀ i, x ≤ f i) :
    IsCoboundedUnder (· ≤ ·) l f :=
  isCoboundedUnder_le_of_eventually_le l (Eventually.of_forall hf)


lemma isCoboundedUnder_ge_of_le [Preorder α] (l : Filter ι) [NeBot l] {f : ι → α} {x : α}
    (hf : ∀ i, f i ≤ x) :
    IsCoboundedUnder (· ≥ ·) l f :=
  isCoboundedUnder_ge_of_eventually_le l (Eventually.of_forall hf)


                                                                  /-
                                                                    α : Type u_1
                                                                    r : α → α → Prop
                                                                    ⊢ Iff (Filter.IsCobounded r Bot.bot) (Exists fun b => ∀ (x : α), r b x)
                                                                  -/
theorem isCobounded_bot : IsCobounded r ⊥ ↔ ∃ b, ∀ x, r b x := by simp [IsCobounded]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem isCobounded_top : IsCobounded r ⊤ ↔ Nonempty α := by
  simp +contextual [IsCobounded, eq_univ_iff_forall,
    exists_true_iff_nonempty]


theorem isCobounded_principal (s : Set α) :
                                                                   /-
                                                                     α : Type u_1
                                                                     r : α → α → Prop
                                                                     s : Set α
                                                                     ⊢ Iff (Filter.IsCobounded r (Filter.principal s)) (Exists fun b => ∀ (a : α),  …
                                                                   -/
    (𝓟 s).IsCobounded r ↔ ∃ b, ∀ a, (∀ x ∈ s, r x a) → r b a := by simp [IsCobounded, subset_def]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem IsCobounded.mono (h : f ≤ g) : f.IsCobounded r → g.IsCobounded r
  | ⟨b, hb⟩ => ⟨b, fun a ha => hb a (h ha)⟩


/-- For nontrivial filters in linear orders, coboundedness for `≤` implies frequent boundedness
from below. -/
lemma IsCobounded.frequently_ge [LinearOrder α] [NeBot f] (cobdd : IsCobounded (· ≤ ·) f) :
    ∃ l, ∃ᶠ x in f, l ≤ x := by
  /-
    α : Type u_1
    f : Filter α
    inst✝¹ : LinearOrder α
    inst✝ : f.NeBot
    cobdd : Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f
    ⊢ Exists fun l => Filter.Frequently (fun x => LE.le l x) f
  -/
  obtain ⟨t, ht⟩ := cobdd
  /-
    case intro
    α : Type u_1
    f : Filter α
    inst✝¹ : LinearOrder α
    inst✝ : f.NeBot
    t : α
    ht : ∀ (a : α), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) f  …
    ⊢ Exists fun l => Filter.Frequently (fun x => LE.le l x) f
  -/
  rcases isBot_or_exists_lt t with tbot | ⟨t', ht'⟩
    /-
      case intro.inl
      α : Type u_1
      f : Filter α
      inst✝¹ : LinearOrder α
      inst✝ : f.NeBot
      t : α
      ht : ∀ (a : α), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) f  …
      tbot : IsBot t
      ⊢ Exists fun l => Filter.Frequently (fun x => LE.le l x) f
    -/
  · exact ⟨t, .of_forall fun r ↦ tbot r⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.inr.intro
    α : Type u_1
    f : Filter α
    inst✝¹ : LinearOrder α
    inst✝ : f.NeBot
    t : α
    ht : ∀ (a : α), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) f  …
    t' : α
    ht' : LT.lt t' t
    ⊢ Exists fun l => Filter.Frequently (fun x => LE.le l x) f
  -/
  refine ⟨t', fun ev ↦ ?_⟩
  /-
    case intro.inr.intro
    α : Type u_1
    f : Filter α
    inst✝¹ : LinearOrder α
    inst✝ : f.NeBot
    t : α
    ht : ∀ (a : α), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) f  …
    t' : α
    ht' : LT.lt t' t
    ev : Filter.Eventually (fun x => Not ((fun x => LE.le t' x) x)) f
    ⊢ False
  -/
  specialize ht t' (by filter_upwards [ev] with _ h using (not_le.mp h).le)
  /-
    case intro.inr.intro
    α : Type u_1
    f : Filter α
    inst✝¹ : LinearOrder α
    inst✝ : f.NeBot
    t t' : α
    ht' : LT.lt t' t
    ev : Filter.Eventually (fun x => Not ((fun x => LE.le t' x) x)) f
    ht : LE.le t t'
    ⊢ False
  -/
  exact not_lt_of_le ht ht'
  /-
    🎉 no goals
  -/


/-- For nontrivial filters in linear orders, coboundedness for `≥` implies frequent boundedness
from above. -/
lemma IsCobounded.frequently_le [LinearOrder α] [NeBot f] (cobdd : IsCobounded (· ≥ ·) f) :
    ∃ u, ∃ᶠ x in f, x ≤ u :=
  cobdd.frequently_ge (α := αᵒᵈ)


/-- In linear orders, frequent boundedness from below implies coboundedness for `≤`. -/
lemma IsCobounded.of_frequently_ge [LinearOrder α] {l : α} (freq_ge : ∃ᶠ x in f, l ≤ x) :
    IsCobounded (· ≤ ·) f := by
  /-
    α : Type u_1
    f : Filter α
    inst✝ : LinearOrder α
    l : α
    freq_ge : Filter.Frequently (fun x => LE.le l x) f
    ⊢ Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f
  -/
  rcases isBot_or_exists_lt l with lbot | ⟨l', hl'⟩
    /-
      case inl
      α : Type u_1
      f : Filter α
      inst✝ : LinearOrder α
      l : α
      freq_ge : Filter.Frequently (fun x => LE.le l x) f
      lbot : IsBot l
      ⊢ Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f
    -/
  · exact ⟨l, fun x _ ↦ lbot x⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u_1
    f : Filter α
    inst✝ : LinearOrder α
    l : α
    freq_ge : Filter.Frequently (fun x => LE.le l x) f
    l' : α
    hl' : LT.lt l' l
    ⊢ Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f
  -/
  refine ⟨l', fun u hu ↦ ?_⟩
  /-
    case inr.intro
    α : Type u_1
    f : Filter α
    inst✝ : LinearOrder α
    l : α
    freq_ge : Filter.Frequently (fun x => LE.le l x) f
    l' : α
    hl' : LT.lt l' l
    u : α
    hu : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x u) f
    ⊢ (fun x1 x2 => LE.le x1 x2) l' u
  -/
  obtain ⟨w, l_le_w, w_le_u⟩ := (freq_ge.and_eventually hu).exists
  /-
    case inr.intro.intro.intro
    α : Type u_1
    f : Filter α
    inst✝ : LinearOrder α
    l : α
    freq_ge : Filter.Frequently (fun x => LE.le l x) f
    l' : α
    hl' : LT.lt l' l
    u : α
    hu : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x u) f
    w : α
    l_le_w : LE.le l w
    w_le_u : LE.le w u
    ⊢ LE.le l' u
  -/
  exact hl'.le.trans (l_le_w.trans w_le_u)
  /-
    🎉 no goals
  -/


/-- In linear orders, frequent boundedness from above implies coboundedness for `≥`. -/
lemma IsCobounded.of_frequently_le [LinearOrder α] {u : α} (freq_le : ∃ᶠ r in f, r ≤ u) :
    IsCobounded (· ≥ ·) f :=
  IsCobounded.of_frequently_ge (α := αᵒᵈ) freq_le


lemma IsCoboundedUnder.frequently_ge [LinearOrder α] {f : Filter ι} [NeBot f] {u : ι → α}
    (h : IsCoboundedUnder (· ≤ ·) f u) :
    ∃ a, ∃ᶠ x in f, a ≤ u x :=
  IsCobounded.frequently_ge h


lemma IsCoboundedUnder.frequently_le [LinearOrder α] {f : Filter ι} [NeBot f] {u : ι → α}
    (h : IsCoboundedUnder (· ≥ ·) f u) :
    ∃ a, ∃ᶠ x in f, u x ≤ a :=
  IsCobounded.frequently_le h


lemma IsCoboundedUnder.of_frequently_ge [LinearOrder α] {f : Filter ι} {u : ι → α}
    {a : α} (freq_ge : ∃ᶠ x in f, a ≤ u x) :
    IsCoboundedUnder (· ≤ ·) f u :=
  IsCobounded.of_frequently_ge freq_ge


lemma IsCoboundedUnder.of_frequently_le [LinearOrder α] {f : Filter ι} {u : ι → α}
    {a : α} (freq_le : ∃ᶠ x in f, u x ≤ a) :
    IsCoboundedUnder (· ≥ ·) f u :=
  IsCobounded.of_frequently_le freq_le


lemma isBoundedUnder_sum {κ : Type*} [AddCommMonoid R] {r : R → R → Prop}
    (hr : ∀ (v₁ v₂ : α → R), f.IsBoundedUnder r v₁ → f.IsBoundedUnder r v₂
      → f.IsBoundedUnder r (v₁ + v₂)) (hr₀ : r 0 0)
    {u : κ → α → R} (s : Finset κ) (h : ∀ k ∈ s, f.IsBoundedUnder r (u k)) :
    f.IsBoundedUnder r (∑ k ∈ s, u k) := by
  /-
    α : Type u_6
    f : Filter α
    R : Type u_7
    κ : Type u_8
    inst✝ : AddCommMonoid R
    r : R → R → Prop
    hr : ∀ (v₁ v₂ : α → R), Filter.IsBoundedUnder r f v₁ → Filter.IsBoundedUnder r …
    hr₀ : r 0 0
    u : κ → α → R
    s : Finset κ
    h : ∀ (k : κ), Membership.mem s k → Filter.IsBoundedUnder r f (u k)
    ⊢ Filter.IsBoundedUnder r f (s.sum fun k => u k)
  -/
  induction s using Finset.cons_induction
  case empty =>
    rw [Finset.sum_empty]
    exact ⟨0, by simp_all only [eventually_map, Pi.zero_apply, eventually_true]⟩
  case cons k₀ s k₀_notin_s ih =>
    simp only [Finset.forall_mem_cons] at *
    simpa only [Finset.sum_cons] using hr _ _ h.1 (ih h.2)


lemma isBoundedUnder_ge_add [Add R] [AddLeftMono R] [AddRightMono R]
    {u v : α → R} (u_bdd_ge : f.IsBoundedUnder (· ≥ ·) u) (v_bdd_ge : f.IsBoundedUnder (· ≥ ·) v) :
    f.IsBoundedUnder (· ≥ ·) (u + v) := by
  /-
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    u_bdd_ge : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u
    v_bdd_ge : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨U, hU⟩ := u_bdd_ge
  /-
    case intro
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    v_bdd_ge : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => (fun x1 x2 => GE.ge x1 x2) x U) (Filter.map u …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨V, hV⟩ := v_bdd_ge
  /-
    case intro.intro
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    U : R
    hU : Filter.Eventually (fun x => (fun x1 x2 => GE.ge x1 x2) x U) (Filter.map u …
    V : R
    hV : Filter.Eventually (fun x => (fun x1 x2 => GE.ge x1 x2) x V) (Filter.map v …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
  -/
  use U + V
  /-
    case h
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    U : R
    hU : Filter.Eventually (fun x => (fun x1 x2 => GE.ge x1 x2) x U) (Filter.map u …
    V : R
    hV : Filter.Eventually (fun x => (fun x1 x2 => GE.ge x1 x2) x V) (Filter.map v …
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => GE.ge x1 x2) x (HAdd.hAdd U V)) (F …
  -/
  simp only [eventually_map, Pi.add_apply] at hU hV ⊢
  /-
    case h
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    U V : R
    hU : Filter.Eventually (fun a => GE.ge (u a) U) f
    hV : Filter.Eventually (fun a => GE.ge (v a) V) f
    ⊢ Filter.Eventually (fun a => GE.ge (HAdd.hAdd (u a) (v a)) (HAdd.hAdd U V)) f
  -/
  filter_upwards [hU, hV] with a hu hv using add_le_add hu hv
  /-
    🎉 no goals
  -/


lemma isBoundedUnder_le_add [Add R] [AddLeftMono R] [AddRightMono R]
    {u v : α → R} (u_bdd_le : f.IsBoundedUnder (· ≤ ·) u) (v_bdd_le : f.IsBoundedUnder (· ≤ ·) v) :
    f.IsBoundedUnder (· ≤ ·) (u + v) := by
  /-
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    u_bdd_le : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    v_bdd_le : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨U, hU⟩ := u_bdd_le
  /-
    case intro
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    v_bdd_le : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x U) (Filter.map u …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨V, hV⟩ := v_bdd_le
  /-
    case intro.intro
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    U : R
    hU : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x U) (Filter.map u …
    V : R
    hV : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x V) (Filter.map v …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
  -/
  use U + V
  /-
    case h
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    U : R
    hU : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x U) (Filter.map u …
    V : R
    hV : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x V) (Filter.map v …
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (HAdd.hAdd U V)) (F …
  -/
  simp only [eventually_map, Pi.add_apply] at hU hV ⊢
  /-
    case h
    α : Type u_6
    f : Filter α
    R : Type u_7
    inst✝³ : Preorder R
    inst✝² : Add R
    inst✝¹ : AddLeftMono R
    inst✝ : AddRightMono R
    u v : α → R
    U V : R
    hU : Filter.Eventually (fun a => LE.le (u a) U) f
    hV : Filter.Eventually (fun a => LE.le (v a) V) f
    ⊢ Filter.Eventually (fun a => LE.le (HAdd.hAdd (u a) (v a)) (HAdd.hAdd U V)) f
  -/
  filter_upwards [hU, hV] with a hu hv using add_le_add hu hv
  /-
    🎉 no goals
  -/


lemma isBoundedUnder_le_sum {κ : Type*} [AddCommMonoid R] [AddLeftMono R] [AddRightMono R]
    {u : κ → α → R} (s : Finset κ) :
    (∀ k ∈ s, f.IsBoundedUnder (· ≤ ·) (u k)) → f.IsBoundedUnder (· ≤ ·) (∑ k ∈ s, u k) :=
  fun h ↦ isBoundedUnder_sum (fun _ _ ↦ isBoundedUnder_le_add) le_rfl s h


lemma isBoundedUnder_ge_sum {κ : Type*} [AddCommMonoid R] [AddLeftMono R] [AddRightMono R]
    {u : κ → α → R} (s : Finset κ) :
    (∀ k ∈ s, f.IsBoundedUnder (· ≥ ·) (u k)) →
      f.IsBoundedUnder (· ≥ ·) (∑ k ∈ s, u k) :=
  fun h ↦ isBoundedUnder_sum (fun _ _ ↦ isBoundedUnder_ge_add) le_rfl s h


lemma isCoboundedUnder_ge_add (hu : f.IsBoundedUnder (· ≤ ·) u)
    (hv : f.IsCoboundedUnder (· ≥ ·) v) :
    f.IsCoboundedUnder (· ≥ ·) (u + v) := by
  /-
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨U, hU⟩ := hu.eventually_le
  /-
    case intro
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨V, hV⟩ := hv.frequently_le
  /-
    case intro.intro
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    V : R
    hV : Filter.Frequently (fun x => LE.le (v x) V) f
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
  -/
  apply IsCoboundedUnder.of_frequently_le (a := U + V)
  /-
    case intro.intro
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    V : R
    hV : Filter.Frequently (fun x => LE.le (v x) V) f
    ⊢ Filter.Frequently (fun x => LE.le (HAdd.hAdd u v x) (HAdd.hAdd U V)) f
  -/
  exact (hV.and_eventually hU).mono fun x hx ↦ add_le_add hx.2 hx.1
  /-
    🎉 no goals
  -/


lemma isCoboundedUnder_le_add (hu : f.IsBoundedUnder (· ≥ ·) u)
    (hv : f.IsCoboundedUnder (· ≤ ·) v) :
    f.IsCoboundedUnder (· ≤ ·) (u + v) := by
  /-
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨U, hU⟩ := hu.eventually_ge
  /-
    case intro
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => LE.le U (u x)) f
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
  -/
  obtain ⟨V, hV⟩ := hv.frequently_ge
  /-
    case intro.intro
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => LE.le U (u x)) f
    V : R
    hV : Filter.Frequently (fun x => LE.le V (v x)) f
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
  -/
  apply IsCoboundedUnder.of_frequently_ge (a := U + V)
  /-
    case intro.intro
    α : Type u_6
    R : Type u_7
    inst✝⁴ : LinearOrder R
    inst✝³ : Add R
    f : Filter α
    inst✝² : f.NeBot
    inst✝¹ : CovariantClass R R (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    inst✝ : CovariantClass R R (fun a b => HAdd.hAdd b a) fun x1 x2 => LE.le x1 x2
    u v : α → R
    hu : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u
    hv : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : R
    hU : Filter.Eventually (fun x => LE.le U (u x)) f
    V : R
    hV : Filter.Frequently (fun x => LE.le V (v x)) f
    ⊢ Filter.Frequently (fun x => LE.le (HAdd.hAdd U V) (HAdd.hAdd u v x)) f
  -/
  exact (hV.and_eventually hU).mono fun x hx ↦ add_le_add hx.2 hx.1
  /-
    🎉 no goals
  -/


lemma isBoundedUnder_le_mul_of_nonneg [Mul α] [Zero α] [Preorder α] [PosMulMono α]
    [MulPosMono α] {f : Filter ι} {u v : ι → α} (h₁ : 0 ≤ᶠ[f] u)
    (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u)
    (h₃ : 0 ≤ᶠ[f] v)
    (h₄ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f v) :
    IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f (u * v) := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝⁴ : Mul α
    inst✝³ : Zero α
    inst✝² : Preorder α
    inst✝¹ : PosMulMono α
    inst✝ : MulPosMono α
    f : Filter ι
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
  -/
  obtain ⟨U, hU⟩ := h₂.eventually_le
  /-
    case intro
    α : Type u_1
    ι : Type u_4
    inst✝⁴ : Mul α
    inst✝³ : Zero α
    inst✝² : Preorder α
    inst✝¹ : PosMulMono α
    inst✝ : MulPosMono α
    f : Filter ι
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : α
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
  -/
  obtain ⟨V, hV⟩ := h₄.eventually_le
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_4
    inst✝⁴ : Mul α
    inst✝³ : Zero α
    inst✝² : Preorder α
    inst✝¹ : PosMulMono α
    inst✝ : MulPosMono α
    f : Filter ι
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : α
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    V : α
    hV : Filter.Eventually (fun x => LE.le (v x) V) f
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
  -/
  refine isBoundedUnder_of_eventually_le (a := U * V) ?_
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_4
    inst✝⁴ : Mul α
    inst✝³ : Zero α
    inst✝² : Preorder α
    inst✝¹ : PosMulMono α
    inst✝ : MulPosMono α
    f : Filter ι
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : α
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    V : α
    hV : Filter.Eventually (fun x => LE.le (v x) V) f
    ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul u v x) (HMul.hMul U V)) f
  -/
  filter_upwards [hU, hV, h₁, h₃] with x x_U x_V u_0 v_0
  /-
    case h
    α : Type u_1
    ι : Type u_4
    inst✝⁴ : Mul α
    inst✝³ : Zero α
    inst✝² : Preorder α
    inst✝¹ : PosMulMono α
    inst✝ : MulPosMono α
    f : Filter ι
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    U : α
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    V : α
    hV : Filter.Eventually (fun x => LE.le (v x) V) f
    x : ι
    x_U : LE.le (u x) U
    x_V : LE.le (v x) V
    u_0 : LE.le (0 x) (u x)
    v_0 : LE.le (0 x) (v x)
    ⊢ LE.le (HMul.hMul u v x) (HMul.hMul U V)
  -/
  exact mul_le_mul x_U x_V v_0 (u_0.trans x_U)
  /-
    🎉 no goals
  -/


lemma isCoboundedUnder_ge_mul_of_nonneg [Mul α] [Zero α] [LinearOrder α] [PosMulMono α]
    [MulPosMono α] {f : Filter ι} [f.NeBot] {u v : ι → α} (h₁ : 0 ≤ᶠ[f] u)
    (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u)
    (h₃ : 0 ≤ᶠ[f] v)
    (h₄ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v) :
    IsCoboundedUnder (fun x1 x2 ↦ x1 ≥ x2) f (u * v) := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝⁵ : Mul α
    inst✝⁴ : Zero α
    inst✝³ : LinearOrder α
    inst✝² : PosMulMono α
    inst✝¹ : MulPosMono α
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
  -/
  obtain ⟨U, hU⟩ := h₂.eventually_le
  /-
    case intro
    α : Type u_1
    ι : Type u_4
    inst✝⁵ : Mul α
    inst✝⁴ : Zero α
    inst✝³ : LinearOrder α
    inst✝² : PosMulMono α
    inst✝¹ : MulPosMono α
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    U : α
    hU : Filter.Eventually (fun x => LE.le (u x) U) f
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
  -/
  obtain ⟨V, hV⟩ := h₄.frequently_le
  exact IsCoboundedUnder.of_frequently_le (a := U * V)
    <| (hV.and_eventually (hU.and (h₁.and h₃))).mono fun x ⟨x_V, x_U, u_0, v_0⟩ ↦
    mul_le_mul x_U x_V v_0 (u_0.trans x_U)


theorem isBounded_le_atBot : (atBot : Filter α).IsBounded (· ≤ ·) :=
  ‹Nonempty α›.elim fun a => ⟨a, eventually_le_atBot _⟩


theorem isBounded_ge_atTop : (atTop : Filter α).IsBounded (· ≥ ·) :=
  ‹Nonempty α›.elim fun a => ⟨a, eventually_ge_atTop _⟩


theorem Tendsto.isBoundedUnder_le_atBot (h : Tendsto u f atBot) : f.IsBoundedUnder (· ≤ ·) u :=
  isBounded_le_atBot.mono h


theorem Tendsto.isBoundedUnder_ge_atTop (h : Tendsto u f atTop) : f.IsBoundedUnder (· ≥ ·) u :=
  isBounded_ge_atTop.mono h


theorem bddAbove_range_of_tendsto_atTop_atBot [IsDirected α (· ≤ ·)] {u : ℕ → α}
    (hx : Tendsto u atTop atBot) : BddAbove (Set.range u) :=
  hx.isBoundedUnder_le_atBot.bddAbove_range


theorem bddBelow_range_of_tendsto_atTop_atTop [IsDirected α (· ≥ ·)] {u : ℕ → α}
    (hx : Tendsto u atTop atTop) : BddBelow (Set.range u) :=
  hx.isBoundedUnder_ge_atTop.bddBelow_range


theorem isCobounded_le_of_bot [Preorder α] [OrderBot α] {f : Filter α} : f.IsCobounded (· ≤ ·) :=
  ⟨⊥, fun _ _ => bot_le⟩


theorem isCobounded_ge_of_top [Preorder α] [OrderTop α] {f : Filter α} : f.IsCobounded (· ≥ ·) :=
  ⟨⊤, fun _ _ => le_top⟩


theorem isBounded_le_of_top [Preorder α] [OrderTop α] {f : Filter α} : f.IsBounded (· ≤ ·) :=
  ⟨⊤, Eventually.of_forall fun _ => le_top⟩


theorem isBounded_ge_of_bot [Preorder α] [OrderBot α] {f : Filter α} : f.IsBounded (· ≥ ·) :=
  ⟨⊥, Eventually.of_forall fun _ => bot_le⟩


@[simp]
theorem _root_.OrderIso.isBoundedUnder_le_comp [Preorder α] [Preorder β] (e : α ≃o β) {l : Filter γ}
    {u : γ → α} : (IsBoundedUnder (· ≤ ·) l fun x => e (u x)) ↔ IsBoundedUnder (· ≤ ·) l u :=
  (Function.Surjective.exists e.surjective).trans <|
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               inst✝¹ : Preorder α
                               inst✝ : Preorder β
                               e : OrderIso α β
                               l : Filter γ
                               u : γ → α
                               a : α
                               ⊢ Iff (Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (e a)) (Filter …
                             -/
    exists_congr fun a => by simp only [eventually_map, e.le_iff_le]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem _root_.OrderIso.isBoundedUnder_ge_comp [Preorder α] [Preorder β] (e : α ≃o β) {l : Filter γ}
    {u : γ → α} : (IsBoundedUnder (· ≥ ·) l fun x => e (u x)) ↔ IsBoundedUnder (· ≥ ·) l u :=
  OrderIso.isBoundedUnder_le_comp e.dual


@[to_additive (attr := simp)]
theorem isBoundedUnder_le_inv [OrderedCommGroup α] {l : Filter β} {u : β → α} :
    (IsBoundedUnder (· ≤ ·) l fun x => (u x)⁻¹) ↔ IsBoundedUnder (· ≥ ·) l u :=
  (OrderIso.inv α).isBoundedUnder_ge_comp


@[to_additive (attr := simp)]
theorem isBoundedUnder_ge_inv [OrderedCommGroup α] {l : Filter β} {u : β → α} :
    (IsBoundedUnder (· ≥ ·) l fun x => (u x)⁻¹) ↔ IsBoundedUnder (· ≤ ·) l u :=
  (OrderIso.inv α).isBoundedUnder_le_comp


theorem IsBoundedUnder.sup [SemilatticeSup α] {f : Filter β} {u v : β → α} :
    f.IsBoundedUnder (· ≤ ·) u →
      f.IsBoundedUnder (· ≤ ·) v → f.IsBoundedUnder (· ≤ ·) fun a => u a ⊔ v a
  | ⟨bu, (hu : ∀ᶠ x in f, u x ≤ bu)⟩, ⟨bv, (hv : ∀ᶠ x in f, v x ≤ bv)⟩ =>
    ⟨bu ⊔ bv, show ∀ᶠ x in f, u x ⊔ v x ≤ bu ⊔ bv
         /-
           α : Type u_1
           β : Type u_2
           inst✝ : SemilatticeSup α
           f : Filter β
           u v : β → α
           bu : α
           hu : Filter.Eventually (fun x => LE.le (u x) bu) f
           bv : α
           hv : Filter.Eventually (fun x => LE.le (v x) bv) f
           ⊢ Filter.Eventually (fun x => LE.le (Max.max (u x) (v x)) (Max.max bu bv)) f
         -/
      by filter_upwards [hu, hv] with _ using sup_le_sup⟩
         /-
           🎉 no goals
         -/


@[simp]
theorem isBoundedUnder_le_sup [SemilatticeSup α] {f : Filter β} {u v : β → α} :
    (f.IsBoundedUnder (· ≤ ·) fun a => u a ⊔ v a) ↔
      f.IsBoundedUnder (· ≤ ·) u ∧ f.IsBoundedUnder (· ≤ ·) v :=
  ⟨fun h =>
    ⟨h.mono_le <| Eventually.of_forall fun _ => le_sup_left,
      h.mono_le <| Eventually.of_forall fun _ => le_sup_right⟩,
    fun h => h.1.sup h.2⟩


theorem IsBoundedUnder.inf [SemilatticeInf α] {f : Filter β} {u v : β → α} :
    f.IsBoundedUnder (· ≥ ·) u →
      f.IsBoundedUnder (· ≥ ·) v → f.IsBoundedUnder (· ≥ ·) fun a => u a ⊓ v a :=
  IsBoundedUnder.sup (α := αᵒᵈ)


@[simp]
theorem isBoundedUnder_ge_inf [SemilatticeInf α] {f : Filter β} {u v : β → α} :
    (f.IsBoundedUnder (· ≥ ·) fun a => u a ⊓ v a) ↔
      f.IsBoundedUnder (· ≥ ·) u ∧ f.IsBoundedUnder (· ≥ ·) v :=
  isBoundedUnder_le_sup (α := αᵒᵈ)


theorem isBoundedUnder_le_abs [LinearOrderedAddCommGroup α] {f : Filter β} {u : β → α} :
    (f.IsBoundedUnder (· ≤ ·) fun a => |u a|) ↔
      f.IsBoundedUnder (· ≤ ·) u ∧ f.IsBoundedUnder (· ≥ ·) u :=
  isBoundedUnder_le_sup.trans <| and_congr Iff.rfl isBoundedUnder_le_neg


/-- Filters are automatically bounded or cobounded in complete lattices. To use the same statements
in complete and conditionally complete lattices but let automation fill automatically the
boundedness proofs in complete lattices, we use the tactic `isBoundedDefault` in the statements,
in the form `(hf : f.IsBounded (≥) := by isBoundedDefault)`. -/

macro "isBoundedDefault" : tactic =>
  `(tactic| first
    | apply isCobounded_le_of_bot
    | apply isCobounded_ge_of_top
    | apply isBounded_le_of_top
    | apply isBounded_ge_of_bot
    | assumption)

-- Porting note: The above is a lean 4 reconstruction of (note that applyc is not available (yet?)):
-- unsafe def is_bounded_default : tactic Unit :=
--   tactic.applyc `` is_cobounded_le_of_bot <|>
--     tactic.applyc `` is_cobounded_ge_of_top <|>
--       tactic.applyc `` is_bounded_le_of_top <|> tactic.applyc `` is_bounded_ge_of_bot



/-- The `limsSup` of a filter `f` is the infimum of the `a` such that, eventually for `f`,
holds `x ≤ a`. -/
def limsSup (f : Filter α) : α :=
  sInf { a | ∀ᶠ n in f, n ≤ a }


/-- The `limsInf` of a filter `f` is the supremum of the `a` such that, eventually for `f`,
holds `x ≥ a`. -/
def limsInf (f : Filter α) : α :=
  sSup { a | ∀ᶠ n in f, a ≤ n }


/-- The `limsup` of a function `u` along a filter `f` is the infimum of the `a` such that,
eventually for `f`, holds `u x ≤ a`. -/
def limsup (u : β → α) (f : Filter β) : α :=
  limsSup (map u f)


/-- The `liminf` of a function `u` along a filter `f` is the supremum of the `a` such that,
eventually for `f`, holds `u x ≥ a`. -/
def liminf (u : β → α) (f : Filter β) : α :=
  limsInf (map u f)


/-- The `blimsup` of a function `u` along a filter `f`, bounded by a predicate `p`, is the infimum
of the `a` such that, eventually for `f`, `u x ≤ a` whenever `p x` holds. -/
def blimsup (u : β → α) (f : Filter β) (p : β → Prop) :=
  sInf { a | ∀ᶠ x in f, p x → u x ≤ a }


/-- The `bliminf` of a function `u` along a filter `f`, bounded by a predicate `p`, is the supremum
of the `a` such that, eventually for `f`, `a ≤ u x` whenever `p x` holds. -/
def bliminf (u : β → α) (f : Filter β) (p : β → Prop) :=
  sSup { a | ∀ᶠ x in f, p x → a ≤ u x }


theorem limsup_eq : limsup u f = sInf { a | ∀ᶠ n in f, u n ≤ a } :=
  rfl


theorem liminf_eq : liminf u f = sSup { a | ∀ᶠ n in f, a ≤ u n } :=
  rfl


theorem blimsup_eq : blimsup u f p = sInf { a | ∀ᶠ x in f, p x → u x ≤ a } :=
  rfl


theorem bliminf_eq : bliminf u f p = sSup { a | ∀ᶠ x in f, p x → a ≤ u x } :=
  rfl


lemma liminf_comp (u : β → α) (v : γ → β) (f : Filter γ) :
    liminf (u ∘ v) f = liminf u (map v f) := rfl


lemma limsup_comp (u : β → α) (v : γ → β) (f : Filter γ) :
    limsup (u ∘ v) f = limsup u (map v f) := rfl


@[simp]
theorem blimsup_true (f : Filter β) (u : β → α) : (blimsup u f fun _ => True) = limsup u f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLattice α
    f : Filter β
    u : β → α
    ⊢ Eq (Filter.blimsup u f fun x => True) (Filter.limsup u f)
  -/
  simp [blimsup_eq, limsup_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem bliminf_true (f : Filter β) (u : β → α) : (bliminf u f fun _ => True) = liminf u f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLattice α
    f : Filter β
    u : β → α
    ⊢ Eq (Filter.bliminf u f fun x => True) (Filter.liminf u f)
  -/
  simp [bliminf_eq, liminf_eq]
  /-
    🎉 no goals
  -/


lemma blimsup_eq_limsup {f : Filter β} {u : β → α} {p : β → Prop} :
    blimsup u f p = limsup u (f ⊓ 𝓟 {x | p x}) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLattice α
    f : Filter β
    u : β → α
    p : β → Prop
    ⊢ Eq (Filter.blimsup u f p) (Filter.limsup u (Min.min f (Filter.principal (set …
  -/
  simp only [blimsup_eq, limsup_eq, eventually_inf_principal, mem_setOf_eq]
  /-
    🎉 no goals
  -/


lemma bliminf_eq_liminf {f : Filter β} {u : β → α} {p : β → Prop} :
    bliminf u f p = liminf u (f ⊓ 𝓟 {x | p x}) :=
  blimsup_eq_limsup (α := αᵒᵈ)


theorem blimsup_eq_limsup_subtype {f : Filter β} {u : β → α} {p : β → Prop} :
    blimsup u f p = limsup (u ∘ ((↑) : { x | p x } → β)) (comap (↑) f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLattice α
    f : Filter β
    u : β → α
    p : β → Prop
    ⊢ Eq (Filter.blimsup u f p) (Filter.limsup (Function.comp u Subtype.val) (Filt …
  -/
  rw [blimsup_eq_limsup, limsup, limsup, ← map_map, map_comap_setCoe_val]
  /-
    🎉 no goals
  -/


theorem bliminf_eq_liminf_subtype {f : Filter β} {u : β → α} {p : β → Prop} :
    bliminf u f p = liminf (u ∘ ((↑) : { x | p x } → β)) (comap (↑) f) :=
  blimsup_eq_limsup_subtype (α := αᵒᵈ)


theorem limsSup_le_of_le {f : Filter α} {a}
    (hf : f.IsCobounded (· ≤ ·) := by isBoundedDefault)
    (h : ∀ᶠ n in f, n ≤ a) : limsSup f ≤ a :=
  csInf_le hf h


theorem le_limsInf_of_le {f : Filter α} {a}
    (hf : f.IsCobounded (· ≥ ·) := by isBoundedDefault)
    (h : ∀ᶠ n in f, a ≤ n) : a ≤ limsInf f :=
  le_csSup hf h


theorem limsup_le_of_le {f : Filter β} {u : β → α} {a}
    (hf : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h : ∀ᶠ n in f, u n ≤ a) : limsup u f ≤ a :=
  csInf_le hf h


theorem le_liminf_of_le {f : Filter β} {u : β → α} {a}
    (hf : f.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h : ∀ᶠ n in f, a ≤ u n) : a ≤ liminf u f :=
  le_csSup hf h


theorem le_limsSup_of_le {f : Filter α} {a}
    (hf : f.IsBounded (· ≤ ·) := by isBoundedDefault)
    (h : ∀ b, (∀ᶠ n in f, n ≤ b) → a ≤ b) : a ≤ limsSup f :=
  le_csInf hf h


theorem limsInf_le_of_le {f : Filter α} {a}
    (hf : f.IsBounded (· ≥ ·) := by isBoundedDefault)
    (h : ∀ b, (∀ᶠ n in f, b ≤ n) → b ≤ a) : limsInf f ≤ a :=
  csSup_le hf h


theorem le_limsup_of_le {f : Filter β} {u : β → α} {a}
    (hf : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h : ∀ b, (∀ᶠ n in f, u n ≤ b) → a ≤ b) : a ≤ limsup u f :=
  le_csInf hf h


theorem liminf_le_of_le {f : Filter β} {u : β → α} {a}
    (hf : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h : ∀ b, (∀ᶠ n in f, b ≤ u n) → b ≤ a) : liminf u f ≤ a :=
  csSup_le hf h


theorem limsInf_le_limsSup {f : Filter α} [NeBot f]
    (h₁ : f.IsBounded (· ≤ ·) := by isBoundedDefault)
    (h₂ : f.IsBounded (· ≥ ·) := by isBoundedDefault) :
    limsInf f ≤ limsSup f :=
  liminf_le_of_le h₂ fun a₀ ha₀ =>
    le_limsup_of_le h₁ fun a₁ ha₁ =>
      show a₀ ≤ a₁ from
        let ⟨_, hb₀, hb₁⟩ := (ha₀.and ha₁).exists
        le_trans hb₀ hb₁


theorem liminf_le_limsup {f : Filter β} [NeBot f] {u : β → α}
    (h : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h' : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) :
    liminf u f ≤ limsup u f :=
  limsInf_le_limsSup h h'


theorem limsSup_le_limsSup {f g : Filter α}
    (hf : f.IsCobounded (· ≤ ·) := by isBoundedDefault)
    (hg : g.IsBounded (· ≤ ·) := by isBoundedDefault)
    (h : ∀ a, (∀ᶠ n in g, n ≤ a) → ∀ᶠ n in f, n ≤ a) : limsSup f ≤ limsSup g :=
  csInf_le_csInf hf hg h


theorem limsInf_le_limsInf {f g : Filter α}
    (hf : f.IsBounded (· ≥ ·) := by isBoundedDefault)
    (hg : g.IsCobounded (· ≥ ·) := by isBoundedDefault)
    (h : ∀ a, (∀ᶠ n in f, a ≤ n) → ∀ᶠ n in g, a ≤ n) : limsInf f ≤ limsInf g :=
  csSup_le_csSup hg hf h


theorem limsup_le_limsup {α : Type*} [ConditionallyCompleteLattice β] {f : Filter α} {u v : α → β}
    (h : u ≤ᶠ[f] v)
    (hu : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (hv : f.IsBoundedUnder (· ≤ ·) v := by isBoundedDefault) :
    limsup u f ≤ limsup v f :=
  limsSup_le_limsSup hu hv fun _ => h.trans


theorem liminf_le_liminf {α : Type*} [ConditionallyCompleteLattice β] {f : Filter α} {u v : α → β}
    (h : ∀ᶠ a in f, u a ≤ v a)
    (hu : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault)
    (hv : f.IsCoboundedUnder (· ≥ ·) v := by isBoundedDefault) :
    liminf u f ≤ liminf v f :=
  limsup_le_limsup (β := βᵒᵈ) h hv hu


theorem limsSup_le_limsSup_of_le {f g : Filter α} (h : f ≤ g)
    (hf : f.IsCobounded (· ≤ ·) := by isBoundedDefault)
    (hg : g.IsBounded (· ≤ ·) := by isBoundedDefault) :
    limsSup f ≤ limsSup g :=
  limsSup_le_limsSup hf hg fun _ ha => h ha


theorem limsInf_le_limsInf_of_le {f g : Filter α} (h : g ≤ f)
    (hf : f.IsBounded (· ≥ ·) := by isBoundedDefault)
    (hg : g.IsCobounded (· ≥ ·) := by isBoundedDefault) :
    limsInf f ≤ limsInf g :=
  limsInf_le_limsInf hf hg fun _ ha => h ha


theorem limsup_le_limsup_of_le {α β} [ConditionallyCompleteLattice β] {f g : Filter α} (h : f ≤ g)
    {u : α → β}
    (hf : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (hg : g.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault) :
    limsup u f ≤ limsup u g :=
  limsSup_le_limsSup_of_le (map_mono h) hf hg


theorem liminf_le_liminf_of_le {α β} [ConditionallyCompleteLattice β] {f g : Filter α} (h : g ≤ f)
    {u : α → β}
    (hf : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault)
    (hg : g.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault) :
    liminf u f ≤ liminf u g :=
  limsInf_le_limsInf_of_le (map_mono h) hf hg


lemma limsSup_principal_eq_csSup (h : BddAbove s) (hs : s.Nonempty) : limsSup (𝓟 s) = sSup s := by
  /-
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    s : Set α
    h : BddAbove s
    hs : s.Nonempty
    ⊢ Eq (Filter.principal s).limsSup (SupSet.sSup s)
  -/
  simp only [limsSup, eventually_principal]; exact csInf_upperBounds_eq_csSup h hs
                                             /-
                                               🎉 no goals
                                             -/


lemma limsInf_principal_eq_csSup (h : BddBelow s) (hs : s.Nonempty) : limsInf (𝓟 s) = sInf s :=
  limsSup_principal_eq_csSup (α := αᵒᵈ) h hs


lemma limsup_top_eq_ciSup [Nonempty β] (hu : BddAbove (range u)) : limsup u ⊤ = ⨆ i, u i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : ConditionallyCompleteLattice α
    u : β → α
    inst✝ : Nonempty β
    hu : BddAbove (Set.range u)
    ⊢ Eq (Filter.limsup u Top.top) (iSup fun i => u i)
  -/
  rw [limsup, map_top, limsSup_principal_eq_csSup hu (range_nonempty _), sSup_range]
  /-
    🎉 no goals
  -/


lemma liminf_top_eq_ciInf [Nonempty β] (hu : BddBelow (range u)) : liminf u ⊤ = ⨅ i, u i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : ConditionallyCompleteLattice α
    u : β → α
    inst✝ : Nonempty β
    hu : BddBelow (Set.range u)
    ⊢ Eq (Filter.liminf u Top.top) (iInf fun i => u i)
  -/
  rw [liminf, map_top, limsInf_principal_eq_csSup hu (range_nonempty _), sInf_range]
  /-
    🎉 no goals
  -/


theorem limsup_congr {α : Type*} [ConditionallyCompleteLattice β] {f : Filter α} {u v : α → β}
    (h : ∀ᶠ a in f, u a = v a) : limsup u f = limsup v f := by
  /-
    β : Type u_2
    α : Type u_6
    inst✝ : ConditionallyCompleteLattice β
    f : Filter α
    u v : α → β
    h : Filter.Eventually (fun a => Eq (u a) (v a)) f
    ⊢ Eq (Filter.limsup u f) (Filter.limsup v f)
  -/
  rw [limsup_eq]
  /-
    β : Type u_2
    α : Type u_6
    inst✝ : ConditionallyCompleteLattice β
    f : Filter α
    u v : α → β
    h : Filter.Eventually (fun a => Eq (u a) (v a)) f
    ⊢ Eq (InfSet.sInf (setOf fun a => Filter.Eventually (fun n => LE.le (u n) a) f …
  -/
  congr with b
  /-
    case e_a.h
    β : Type u_2
    α : Type u_6
    inst✝ : ConditionallyCompleteLattice β
    f : Filter α
    u v : α → β
    h : Filter.Eventually (fun a => Eq (u a) (v a)) f
    b : β
    ⊢ Iff (Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le (u n)  …
  -/
  exact eventually_congr (h.mono fun x hx => by simp [hx])
  /-
    🎉 no goals
  -/


theorem blimsup_congr {f : Filter β} {u v : β → α} {p : β → Prop} (h : ∀ᶠ a in f, p a → u a = v a) :
    blimsup u f p = blimsup v f p := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLattice α
    f : Filter β
    u v : β → α
    p : β → Prop
    h : Filter.Eventually (fun a => p a → Eq (u a) (v a)) f
    ⊢ Eq (Filter.blimsup u f p) (Filter.blimsup v f p)
  -/
  simpa only [blimsup_eq_limsup] using limsup_congr <| eventually_inf_principal.2 h
  /-
    🎉 no goals
  -/


theorem bliminf_congr {f : Filter β} {u v : β → α} {p : β → Prop} (h : ∀ᶠ a in f, p a → u a = v a) :
    bliminf u f p = bliminf v f p :=
  blimsup_congr (α := αᵒᵈ) h


theorem liminf_congr {α : Type*} [ConditionallyCompleteLattice β] {f : Filter α} {u v : α → β}
    (h : ∀ᶠ a in f, u a = v a) : liminf u f = liminf v f :=
  limsup_congr (β := βᵒᵈ) h


@[simp]
theorem limsup_const {α : Type*} [ConditionallyCompleteLattice β] {f : Filter α} [NeBot f]
    (b : β) : limsup (fun _ => b) f = b := by
  /-
    β : Type u_2
    α : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    f : Filter α
    inst✝ : f.NeBot
    b : β
    ⊢ Eq (Filter.limsup (fun x => b) f) b
  -/
  simpa only [limsup_eq, eventually_const] using csInf_Ici
  /-
    🎉 no goals
  -/


@[simp]
theorem liminf_const {α : Type*} [ConditionallyCompleteLattice β] {f : Filter α} [NeBot f]
    (b : β) : liminf (fun _ => b) f = b :=
  limsup_const (β := βᵒᵈ) b


theorem HasBasis.liminf_eq_sSup_iUnion_iInter {ι ι' : Type*} {f : ι → α} {v : Filter ι}
    {p : ι' → Prop} {s : ι' → Set ι} (hv : v.HasBasis p s) :
    liminf f v = sSup (⋃ (j : Subtype p), ⋂ (i : s j), Iic (f i)) := by
  /-
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    ι : Type u_6
    ι' : Type u_7
    f : ι → α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    hv : v.HasBasis p s
    ⊢ Eq (Filter.liminf f v) (SupSet.sSup (Set.iUnion fun j => Set.iInter fun i => …
  -/
  simp_rw [liminf_eq, hv.eventually_iff]
  /-
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    ι : Type u_6
    ι' : Type u_7
    f : ι → α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    hv : v.HasBasis p s
    ⊢ Eq (SupSet.sSup (setOf fun a => Exists fun i => And (p i) (∀ ⦃x : ι⦄, Member …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    ι : Type u_6
    ι' : Type u_7
    f : ι → α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    hv : v.HasBasis p s
    ⊢ Eq (setOf fun a => Exists fun i => And (p i) (∀ ⦃x : ι⦄, Membership.mem (s i …
  -/
  ext x
  simp only [mem_setOf_eq, iInter_coe_set, mem_iUnion, mem_iInter, mem_Iic, Subtype.exists,
    exists_prop]


theorem HasBasis.liminf_eq_sSup_univ_of_empty {f : ι → α} {v : Filter ι}
    {p : ι' → Prop} {s : ι' → Set ι} (hv : v.HasBasis p s) (i : ι') (hi : p i) (h'i : s i = ∅) :
    liminf f v = sSup univ := by
  /-
    α : Type u_1
    ι : Type u_4
    ι' : Type u_5
    inst✝ : ConditionallyCompleteLattice α
    f : ι → α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    hv : v.HasBasis p s
    i : ι'
    hi : p i
    h'i : Eq (s i) EmptyCollection.emptyCollection
    ⊢ Eq (Filter.liminf f v) (SupSet.sSup Set.univ)
  -/
  simp [hv.eq_bot_iff.2 ⟨i, hi, h'i⟩, liminf_eq]
  /-
    🎉 no goals
  -/


theorem HasBasis.limsup_eq_sInf_iUnion_iInter {ι ι' : Type*} {f : ι → α} {v : Filter ι}
    {p : ι' → Prop} {s : ι' → Set ι} (hv : v.HasBasis p s) :
    limsup f v = sInf (⋃ (j : Subtype p), ⋂ (i : s j), Ici (f i)) :=
  HasBasis.liminf_eq_sSup_iUnion_iInter (α := αᵒᵈ) hv


theorem HasBasis.limsup_eq_sInf_univ_of_empty {f : ι → α} {v : Filter ι}
    {p : ι' → Prop} {s : ι' → Set ι} (hv : v.HasBasis p s) (i : ι') (hi : p i) (h'i : s i = ∅) :
    limsup f v = sInf univ :=
  HasBasis.liminf_eq_sSup_univ_of_empty (α := αᵒᵈ) hv i hi h'i


@[simp]
theorem liminf_nat_add (f : ℕ → α) (k : ℕ) :
    liminf (fun i => f (i + k)) atTop = liminf f atTop := by
  /-
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    f : Nat → α
    k : Nat
    ⊢ Eq (Filter.liminf (fun i => f (HAdd.hAdd i k)) Filter.atTop) (Filter.liminf  …
  -/
  change liminf (f ∘ (· + k)) atTop = liminf f atTop
  /-
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    f : Nat → α
    k : Nat
    ⊢ Eq (Filter.liminf (Function.comp f fun x => HAdd.hAdd x k) Filter.atTop) (Fi …
  -/
  rw [liminf, liminf, ← map_map, map_add_atTop_eq_nat]
  /-
    🎉 no goals
  -/


@[simp]
theorem limsup_nat_add (f : ℕ → α) (k : ℕ) : limsup (fun i => f (i + k)) atTop = limsup f atTop :=
  @liminf_nat_add αᵒᵈ _ f k


@[simp]
theorem limsSup_bot : limsSup (⊥ : Filter α) = ⊥ :=
                              /-
                                α : Type u_1
                                inst✝ : CompleteLattice α
                                ⊢ Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le n a) Bot.bo …
                              -/
  bot_unique <| sInf_le <| by simp
                              /-
                                🎉 no goals
                              -/


                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                inst✝ : CompleteLattice α
                                                                f : β → α
                                                                ⊢ Eq (Filter.limsup f Bot.bot) Bot.bot
                                                              -/
@[simp] theorem limsup_bot (f : β → α) : limsup f ⊥ = ⊥ := by simp [limsup]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem limsInf_bot : limsInf (⊥ : Filter α) = ⊤ :=
                              /-
                                α : Type u_1
                                inst✝ : CompleteLattice α
                                ⊢ Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le a n) Bot.bo …
                              -/
  top_unique <| le_sSup <| by simp
                              /-
                                🎉 no goals
                              -/


                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                inst✝ : CompleteLattice α
                                                                f : β → α
                                                                ⊢ Eq (Filter.liminf f Bot.bot) Top.top
                                                              -/
@[simp] theorem liminf_bot (f : β → α) : liminf f ⊥ = ⊤ := by simp [liminf]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem limsSup_top : limsSup (⊤ : Filter α) = ⊤ :=
                              /-
                                α : Type u_1
                                inst✝ : CompleteLattice α
                                ⊢ ∀ (b : α), Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le  …
                              -/
  top_unique <| le_sInf <| by simpa [eq_univ_iff_forall] using fun b hb => top_unique <| hb _
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem limsInf_top : limsInf (⊤ : Filter α) = ⊥ :=
                              /-
                                α : Type u_1
                                inst✝ : CompleteLattice α
                                ⊢ ∀ (b : α), Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le  …
                              -/
  bot_unique <| sSup_le <| by simpa [eq_univ_iff_forall] using fun b hb => bot_unique <| hb _
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem blimsup_false {f : Filter β} {u : β → α} : (blimsup u f fun _ => False) = ⊥ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    u : β → α
    ⊢ Eq (Filter.blimsup u f fun x => False) Bot.bot
  -/
  simp [blimsup_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem bliminf_false {f : Filter β} {u : β → α} : (bliminf u f fun _ => False) = ⊤ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    u : β → α
    ⊢ Eq (Filter.bliminf u f fun x => False) Top.top
  -/
  simp [bliminf_eq]
  /-
    🎉 no goals
  -/


/-- Same as limsup_const applied to `⊥` but without the `NeBot f` assumption -/
@[simp]
theorem limsup_const_bot {f : Filter β} : limsup (fun _ : β => (⊥ : α)) f = (⊥ : α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    ⊢ Eq (Filter.limsup (fun x => Bot.bot) f) Bot.bot
  -/
  rw [limsup_eq, eq_bot_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    ⊢ LE.le (InfSet.sInf (setOf fun a => Filter.Eventually (fun n => LE.le Bot.bot …
  -/
  exact sInf_le (Eventually.of_forall fun _ => le_rfl)
  /-
    🎉 no goals
  -/


/-- Same as limsup_const applied to `⊤` but without the `NeBot f` assumption -/
@[simp]
theorem liminf_const_top {f : Filter β} : liminf (fun _ : β => (⊤ : α)) f = (⊤ : α) :=
  limsup_const_bot (α := αᵒᵈ)


theorem HasBasis.limsSup_eq_iInf_sSup {ι} {p : ι → Prop} {s} {f : Filter α} (h : f.HasBasis p s) :
    limsSup f = ⨅ (i) (_ : p i), sSup (s i) :=
  le_antisymm (le_iInf₂ fun i hi => sInf_le <| h.eventually_iff.2 ⟨i, hi, fun _ => le_sSup⟩)
    (le_sInf fun _ ha =>
      let ⟨_, hi, ha⟩ := h.eventually_iff.1 ha
      iInf₂_le_of_le _ hi <| sSup_le ha)


theorem HasBasis.limsInf_eq_iSup_sInf {p : ι → Prop} {s : ι → Set α} {f : Filter α}
    (h : f.HasBasis p s) : limsInf f = ⨆ (i) (_ : p i), sInf (s i) :=
  HasBasis.limsSup_eq_iInf_sSup (α := αᵒᵈ) h


theorem limsSup_eq_iInf_sSup {f : Filter α} : limsSup f = ⨅ s ∈ f, sSup s :=
  f.basis_sets.limsSup_eq_iInf_sSup


theorem limsInf_eq_iSup_sInf {f : Filter α} : limsInf f = ⨆ s ∈ f, sInf s :=
  limsSup_eq_iInf_sSup (α := αᵒᵈ)


theorem limsup_le_iSup {f : Filter β} {u : β → α} : limsup u f ≤ ⨆ n, u n :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝ : CompleteLattice α
                        f : Filter β
                        u : β → α
                        ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u
                      -/
  limsup_le_of_le (by isBoundedDefault) (Eventually.of_forall (le_iSup u))
                      /-
                        🎉 no goals
                      -/


theorem iInf_le_liminf {f : Filter β} {u : β → α} : ⨅ n, u n ≤ liminf u f :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝ : CompleteLattice α
                        f : Filter β
                        u : β → α
                        ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f u
                      -/
  le_liminf_of_le (by isBoundedDefault) (Eventually.of_forall (iInf_le u))
                      /-
                        🎉 no goals
                      -/


/-- In a complete lattice, the limsup of a function is the infimum over sets `s` in the filter
of the supremum of the function over `s` -/
theorem limsup_eq_iInf_iSup {f : Filter β} {u : β → α} : limsup u f = ⨅ s ∈ f, ⨆ a ∈ s, u a :=
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          inst✝ : CompleteLattice α
                                                          f : Filter β
                                                          u : β → α
                                                          ⊢ Eq (iInf fun i => iInf fun x => SupSet.sSup (Set.image u (id i))) (iInf fun  …
                                                        -/
  (f.basis_sets.map u).limsSup_eq_iInf_sSup.trans <| by simp only [sSup_image, id]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem limsup_eq_iInf_iSup_of_nat {u : ℕ → α} : limsup u atTop = ⨅ n : ℕ, ⨆ i ≥ n, u i :=
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : CompleteLattice α
                                                         u : Nat → α
                                                         ⊢ Eq (iInf fun i => iInf fun x => SupSet.sSup (Set.image u (Set.Ici i))) (iInf …
                                                       -/
  (atTop_basis.map u).limsSup_eq_iInf_sSup.trans <| by simp only [sSup_image, iInf_const]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem limsup_eq_iInf_iSup_of_nat' {u : ℕ → α} : limsup u atTop = ⨅ n : ℕ, ⨆ i : ℕ, u (i + n) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    u : Nat → α
    ⊢ Eq (Filter.limsup u Filter.atTop) (iInf fun n => iSup fun i => u (HAdd.hAdd  …
  -/
  simp only [limsup_eq_iInf_iSup_of_nat, iSup_ge_eq_iSup_nat_add]
  /-
    🎉 no goals
  -/


theorem HasBasis.limsup_eq_iInf_iSup {p : ι → Prop} {s : ι → Set β} {f : Filter β} {u : β → α}
    (h : f.HasBasis p s) : limsup u f = ⨅ (i) (_ : p i), ⨆ a ∈ s i, u a :=
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               ι : Type u_4
                                               inst✝ : CompleteLattice α
                                               p : ι → Prop
                                               s : ι → Set β
                                               f : Filter β
                                               u : β → α
                                               h : f.HasBasis p s
                                               ⊢ Eq (iInf fun i => iInf fun x => SupSet.sSup (Set.image u (s i))) (iInf fun i …
                                             -/
  (h.map u).limsSup_eq_iInf_sSup.trans <| by simp only [sSup_image, id]
                                             /-
                                               🎉 no goals
                                             -/


lemma limsSup_principal_eq_sSup (s : Set α) : limsSup (𝓟 s) = sSup s := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Eq (Filter.principal s).limsSup (SupSet.sSup s)
  -/
  simpa only [limsSup, eventually_principal] using sInf_upperBounds_eq_csSup s
  /-
    🎉 no goals
  -/


lemma limsInf_principal_eq_sInf (s : Set α) : limsInf (𝓟 s) = sInf s := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Eq (Filter.principal s).limsInf (InfSet.sInf s)
  -/
  simpa only [limsInf, eventually_principal] using sSup_lowerBounds_eq_sInf s
  /-
    🎉 no goals
  -/


@[simp] lemma limsup_top_eq_iSup (u : β → α) : limsup u ⊤ = ⨆ i, u i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    u : β → α
    ⊢ Eq (Filter.limsup u Top.top) (iSup fun i => u i)
  -/
  rw [limsup, map_top, limsSup_principal_eq_sSup, sSup_range]
  /-
    🎉 no goals
  -/


@[simp] lemma liminf_top_eq_iInf (u : β → α) : liminf u ⊤ = ⨅ i, u i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    u : β → α
    ⊢ Eq (Filter.liminf u Top.top) (iInf fun i => u i)
  -/
  rw [liminf, map_top, limsInf_principal_eq_sInf, sInf_range]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-27")] alias limsSup_principal := limsSup_principal_eq_sSup

@[deprecated (since := "2024-08-27")] alias limsInf_principal := limsInf_principal_eq_sInf

@[deprecated (since := "2024-08-27")] alias limsup_top := limsup_top_eq_iSup

@[deprecated (since := "2024-08-27")] alias liminf_top := liminf_top_eq_iInf


theorem blimsup_congr' {f : Filter β} {p q : β → Prop} {u : β → α}
    (h : ∀ᶠ x in f, u x ≠ ⊥ → (p x ↔ q x)) : blimsup u f p = blimsup u f q := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    p q : β → Prop
    u : β → α
    h : Filter.Eventually (fun x => Ne (u x) Bot.bot → Iff (p x) (q x)) f
    ⊢ Eq (Filter.blimsup u f p) (Filter.blimsup u f q)
  -/
  simp only [blimsup_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    p q : β → Prop
    u : β → α
    h : Filter.Eventually (fun x => Ne (u x) Bot.bot → Iff (p x) (q x)) f
    ⊢ Eq (InfSet.sInf (setOf fun a => Filter.Eventually (fun x => p x → LE.le (u x …
  -/
  congr with a
  /-
    case e_a.h
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    p q : β → Prop
    u : β → α
    h : Filter.Eventually (fun x => Ne (u x) Bot.bot → Iff (p x) (q x)) f
    a : α
    ⊢ Iff (Membership.mem (setOf fun a => Filter.Eventually (fun x => p x → LE.le  …
  -/
  refine eventually_congr (h.mono fun b hb => ?_)
  /-
    case e_a.h
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    p q : β → Prop
    u : β → α
    h : Filter.Eventually (fun x => Ne (u x) Bot.bot → Iff (p x) (q x)) f
    a : α
    b : β
    hb : Ne (u b) Bot.bot → Iff (p b) (q b)
    ⊢ Iff (p b → LE.le (u b) a) (q b → LE.le (u b) a)
  -/
  rcases eq_or_ne (u b) ⊥ with hu | hu; · simp [hu]
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case e_a.h.inr
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    p q : β → Prop
    u : β → α
    h : Filter.Eventually (fun x => Ne (u x) Bot.bot → Iff (p x) (q x)) f
    a : α
    b : β
    hb : Ne (u b) Bot.bot → Iff (p b) (q b)
    hu : Ne (u b) Bot.bot
    ⊢ Iff (p b → LE.le (u b) a) (q b → LE.le (u b) a)
  -/
  rw [hb hu]
  /-
    🎉 no goals
  -/


theorem bliminf_congr' {f : Filter β} {p q : β → Prop} {u : β → α}
    (h : ∀ᶠ x in f, u x ≠ ⊤ → (p x ↔ q x)) : bliminf u f p = bliminf u f q :=
  blimsup_congr' (α := αᵒᵈ) h


lemma HasBasis.blimsup_eq_iInf_iSup {p : ι → Prop} {s : ι → Set β} {f : Filter β} {u : β → α}
    (hf : f.HasBasis p s) {q : β → Prop} :
    blimsup u f q = ⨅ (i) (_ : p i), ⨆ a ∈ s i, ⨆ (_ : q a), u a := by
  simp only [blimsup_eq_limsup, (hf.inf_principal _).limsup_eq_iInf_iSup, mem_inter_iff, iSup_and,
    mem_setOf_eq]


theorem blimsup_eq_iInf_biSup {f : Filter β} {p : β → Prop} {u : β → α} :
    blimsup u f p = ⨅ s ∈ f, ⨆ (b) (_ : p b ∧ b ∈ s), u b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteLattice α
    f : Filter β
    p : β → Prop
    u : β → α
    ⊢ Eq (Filter.blimsup u f p) (iInf fun s => iInf fun h => iSup fun b => iSup fu …
  -/
  simp only [f.basis_sets.blimsup_eq_iInf_iSup, iSup_and', id, and_comm]
  /-
    🎉 no goals
  -/


theorem blimsup_eq_iInf_biSup_of_nat {p : ℕ → Prop} {u : ℕ → α} :
    blimsup u atTop p = ⨅ i, ⨆ (j) (_ : p j ∧ i ≤ j), u j := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    p : Nat → Prop
    u : Nat → α
    ⊢ Eq (Filter.blimsup u Filter.atTop p) (iInf fun i => iSup fun j => iSup fun x …
  -/
  simp only [atTop_basis.blimsup_eq_iInf_iSup, @and_comm (p _), iSup_and, mem_Ici, iInf_true]
  /-
    🎉 no goals
  -/


/-- In a complete lattice, the liminf of a function is the infimum over sets `s` in the filter
of the supremum of the function over `s` -/
theorem liminf_eq_iSup_iInf {f : Filter β} {u : β → α} : liminf u f = ⨆ s ∈ f, ⨅ a ∈ s, u a :=
  limsup_eq_iInf_iSup (α := αᵒᵈ)


theorem liminf_eq_iSup_iInf_of_nat {u : ℕ → α} : liminf u atTop = ⨆ n : ℕ, ⨅ i ≥ n, u i :=
  @limsup_eq_iInf_iSup_of_nat αᵒᵈ _ u


theorem liminf_eq_iSup_iInf_of_nat' {u : ℕ → α} : liminf u atTop = ⨆ n : ℕ, ⨅ i : ℕ, u (i + n) :=
  @limsup_eq_iInf_iSup_of_nat' αᵒᵈ _ _


theorem HasBasis.liminf_eq_iSup_iInf {p : ι → Prop} {s : ι → Set β} {f : Filter β} {u : β → α}
    (h : f.HasBasis p s) : liminf u f = ⨆ (i) (_ : p i), ⨅ a ∈ s i, u a :=
  HasBasis.limsup_eq_iInf_iSup (α := αᵒᵈ) h


theorem bliminf_eq_iSup_biInf {f : Filter β} {p : β → Prop} {u : β → α} :
    bliminf u f p = ⨆ s ∈ f, ⨅ (b) (_ : p b ∧ b ∈ s), u b :=
  @blimsup_eq_iInf_biSup αᵒᵈ β _ f p u


theorem bliminf_eq_iSup_biInf_of_nat {p : ℕ → Prop} {u : ℕ → α} :
    bliminf u atTop p = ⨆ i, ⨅ (j) (_ : p j ∧ i ≤ j), u j :=
  @blimsup_eq_iInf_biSup_of_nat αᵒᵈ _ p u


theorem limsup_eq_sInf_sSup {ι R : Type*} (F : Filter ι) [CompleteLattice R] (a : ι → R) :
    limsup a F = sInf ((fun I => sSup (a '' I)) '' F.sets) := by
  /-
    ι : Type u_6
    R : Type u_7
    F : Filter ι
    inst✝ : CompleteLattice R
    a : ι → R
    ⊢ Eq (Filter.limsup a F) (InfSet.sInf (Set.image (fun I => SupSet.sSup (Set.im …
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      ⊢ LE.le (Filter.limsup a F) (InfSet.sInf (Set.image (fun I => SupSet.sSup (Set …
    -/
  · rw [limsup_eq]
    /-
      case a
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      ⊢ LE.le (InfSet.sInf (setOf fun a_1 => Filter.Eventually (fun n => LE.le (a n) …
    -/
    refine sInf_le_sInf fun x hx => ?_
    /-
      case a
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      x : R
      hx : Membership.mem (Set.image (fun I => SupSet.sSup (Set.image a I)) F.sets) x
      ⊢ Membership.mem (setOf fun a_1 => Filter.Eventually (fun n => LE.le (a n) a_1 …
    -/
    rcases (mem_image _ F.sets x).mp hx with ⟨I, ⟨I_mem_F, hI⟩⟩
    /-
      case a.intro.intro
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      x : R
      hx : Membership.mem (Set.image (fun I => SupSet.sSup (Set.image a I)) F.sets) x
      I : Set ι
      I_mem_F : Membership.mem F.sets I
      hI : Eq (SupSet.sSup (Set.image a I)) x
      ⊢ Membership.mem (setOf fun a_1 => Filter.Eventually (fun n => LE.le (a n) a_1 …
    -/
    filter_upwards [I_mem_F] with i hi
    /-
      case h
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      x : R
      hx : Membership.mem (Set.image (fun I => SupSet.sSup (Set.image a I)) F.sets) x
      I : Set ι
      I_mem_F : Membership.mem F.sets I
      hI : Eq (SupSet.sSup (Set.image a I)) x
      i : ι
      hi : Membership.mem I i
      ⊢ LE.le (a i) x
    -/
    exact hI ▸ le_sSup (mem_image_of_mem _ hi)
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      ⊢ LE.le (InfSet.sInf (Set.image (fun I => SupSet.sSup (Set.image a I)) F.sets) …
    -/
  · refine le_sInf fun b hb => sInf_le_of_le (mem_image_of_mem _ hb) <| sSup_le ?_
    /-
      case a
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      b : R
      hb : Membership.mem (setOf fun a_1 => Filter.Eventually (fun n => LE.le n a_1) …
      ⊢ ∀ (b_1 : R), Membership.mem (Set.image a (Set.preimage a (setOf fun x => (fu …
    -/
    rintro _ ⟨_, h, rfl⟩
    /-
      case a.intro.intro
      ι : Type u_6
      R : Type u_7
      F : Filter ι
      inst✝ : CompleteLattice R
      a : ι → R
      b : R
      hb : Membership.mem (setOf fun a_1 => Filter.Eventually (fun n => LE.le n a_1) …
      w✝ : ι
      h : Membership.mem (Set.preimage a (setOf fun x => (fun n => LE.le n b) x)) w✝
      ⊢ LE.le (a w✝) b
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem liminf_eq_sSup_sInf {ι R : Type*} (F : Filter ι) [CompleteLattice R] (a : ι → R) :
    liminf a F = sSup ((fun I => sInf (a '' I)) '' F.sets) :=
  @Filter.limsup_eq_sInf_sSup ι (OrderDual R) _ _ a


theorem liminf_le_of_frequently_le' {α β} [CompleteLattice β] {f : Filter α} {u : α → β} {x : β}
    (h : ∃ᶠ a in f, u a ≤ x) : liminf u f ≤ x := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : CompleteLattice β
    f : Filter α
    u : α → β
    x : β
    h : Filter.Frequently (fun a => LE.le (u a) x) f
    ⊢ LE.le (Filter.liminf u f) x
  -/
  rw [liminf_eq]
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : CompleteLattice β
    f : Filter α
    u : α → β
    x : β
    h : Filter.Frequently (fun a => LE.le (u a) x) f
    ⊢ LE.le (SupSet.sSup (setOf fun a => Filter.Eventually (fun n => LE.le a (u n) …
  -/
  refine sSup_le fun b hb => ?_
  have hbx : ∃ᶠ _ in f, b ≤ x := by
    revert h
    rw [← not_imp_not, not_frequently, not_frequently]
    exact fun h => hb.mp (h.mono fun a hbx hba hax => hbx (hba.trans hax))
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : CompleteLattice β
    f : Filter α
    u : α → β
    x : β
    h : Filter.Frequently (fun a => LE.le (u a) x) f
    b : β
    hb : Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le a (u n)) …
    hbx : Filter.Frequently (fun x_1 => LE.le b x) f
    ⊢ LE.le b x
  -/
  exact hbx.exists.choose_spec
  /-
    🎉 no goals
  -/


theorem le_limsup_of_frequently_le' {α β} [CompleteLattice β] {f : Filter α} {u : α → β} {x : β}
    (h : ∃ᶠ a in f, x ≤ u a) : x ≤ limsup u f :=
  liminf_le_of_frequently_le' (β := βᵒᵈ) h


/-- If `f : α → α` is a morphism of complete lattices, then the limsup of its iterates of any
`a : α` is a fixed point. -/
@[simp]
theorem _root_.CompleteLatticeHom.apply_limsup_iterate (f : CompleteLatticeHom α α) (a : α) :
    f (limsup (fun n => f^[n] a) atTop) = limsup (fun n => f^[n] a) atTop := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : CompleteLatticeHom α α
    a : α
    ⊢ Eq (f (Filter.limsup (fun n => Nat.iterate (⇑f) n a) Filter.atTop)) (Filter. …
  -/
  rw [limsup_eq_iInf_iSup_of_nat', map_iInf]
  simp_rw [_root_.map_iSup, ← Function.comp_apply (f := f), ← Function.iterate_succ' f,
    ← Nat.add_succ]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : CompleteLatticeHom α α
    a : α
    ⊢ Eq (iInf fun i => iSup fun i_1 => Nat.iterate (⇑f) (HAdd.hAdd i_1 i.succ) a) …
  -/
  conv_rhs => rw [iInf_split _ (0 < ·)]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : CompleteLatticeHom α α
    a : α
    ⊢ Eq (iInf fun i => iSup fun i_1 => Nat.iterate (⇑f) (HAdd.hAdd i_1 i.succ) a) …
  -/
  simp only [not_lt, Nat.le_zero, iInf_iInf_eq_left, add_zero, iInf_nat_gt_zero_eq, left_eq_inf]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : CompleteLatticeHom α α
    a : α
    ⊢ LE.le (iInf fun i => iSup fun i_1 => Nat.iterate (⇑f) (HAdd.hAdd i_1 i.succ) …
  -/
  refine (iInf_le (fun i => ⨆ j, f^[j + (i + 1)] a) 0).trans ?_
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : CompleteLatticeHom α α
    a : α
    ⊢ LE.le (iSup fun j => Nat.iterate (⇑f) (HAdd.hAdd j (HAdd.hAdd 0 1)) a) (iSup …
  -/
  simp only [zero_add, Function.comp_apply, iSup_le_iff]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : CompleteLatticeHom α α
    a : α
    ⊢ ∀ (i : Nat), LE.le (Nat.iterate (⇑f) (HAdd.hAdd i 1) a) (iSup fun i => Nat.i …
  -/
  exact fun i => le_iSup (fun i => f^[i] a) (i + 1)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-21")]
alias CompleteLatticeHom.apply_limsup_iterate := CompleteLatticeHom.apply_limsup_iterate


/-- If `f : α → α` is a morphism of complete lattices, then the liminf of its iterates of any
`a : α` is a fixed point. -/
theorem _root_.CompleteLatticeHom.apply_liminf_iterate (f : CompleteLatticeHom α α) (a : α) :
    f (liminf (fun n => f^[n] a) atTop) = liminf (fun n => f^[n] a) atTop :=
  (CompleteLatticeHom.dual f).apply_limsup_iterate _


@[deprecated (since := "2024-07-21")]
alias CompleteLatticeHom.apply_liminf_iterate := CompleteLatticeHom.apply_liminf_iterate


theorem blimsup_mono (h : ∀ x, p x → q x) : blimsup u f p ≤ blimsup u f q :=
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           inst✝ : CompleteLattice α
                                           f : Filter β
                                           p q : β → Prop
                                           u : β → α
                                           h : ∀ (x : β), p x → q x
                                           a : α
                                           ha : Membership.mem (setOf fun a => Filter.Eventually (fun x => q x → LE.le (u …
                                           ⊢ ∀ (x : β), (q x → LE.le (u x) a) → p x → LE.le (u x) a
                                         -/
  sInf_le_sInf fun a ha => ha.mono <| by tauto
                                         /-
                                           🎉 no goals
                                         -/


theorem bliminf_antitone (h : ∀ x, p x → q x) : bliminf u f q ≤ bliminf u f p :=
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           inst✝ : CompleteLattice α
                                           f : Filter β
                                           p q : β → Prop
                                           u : β → α
                                           h : ∀ (x : β), p x → q x
                                           a : α
                                           ha : Membership.mem (setOf fun a => Filter.Eventually (fun x => q x → LE.le a  …
                                           ⊢ ∀ (x : β), (q x → LE.le a (u x)) → p x → LE.le a (u x)
                                         -/
  sSup_le_sSup fun a ha => ha.mono <| by tauto
                                         /-
                                           🎉 no goals
                                         -/


theorem mono_blimsup' (h : ∀ᶠ x in f, p x → u x ≤ v x) : blimsup u f p ≤ blimsup v f p :=
  sInf_le_sInf fun _ ha => (ha.and h).mono fun _ hx hx' => (hx.2 hx').trans (hx.1 hx')


theorem mono_blimsup (h : ∀ x, p x → u x ≤ v x) : blimsup u f p ≤ blimsup v f p :=
  mono_blimsup' <| Eventually.of_forall h


theorem mono_bliminf' (h : ∀ᶠ x in f, p x → u x ≤ v x) : bliminf u f p ≤ bliminf v f p :=
  sSup_le_sSup fun _ ha => (ha.and h).mono fun _ hx hx' => (hx.1 hx').trans (hx.2 hx')


theorem mono_bliminf (h : ∀ x, p x → u x ≤ v x) : bliminf u f p ≤ bliminf v f p :=
  mono_bliminf' <| Eventually.of_forall h


theorem bliminf_antitone_filter (h : f ≤ g) : bliminf u g p ≤ bliminf u f p :=
  sSup_le_sSup fun _ ha => ha.filter_mono h


theorem blimsup_monotone_filter (h : f ≤ g) : blimsup u f p ≤ blimsup u g p :=
  sInf_le_sInf fun _ ha => ha.filter_mono h

-- @[simp] -- Porting note: simp_nf linter, lhs simplifies, added _aux versions below

theorem blimsup_and_le_inf : (blimsup u f fun x => p x ∧ q x) ≤ blimsup u f p ⊓ blimsup u f q :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝ : CompleteLattice α
                               f : Filter β
                               p q : β → Prop
                               u : β → α
                               ⊢ ∀ (x : β), And (p x) (q x) → p x
                             -/
                             /-
                               🎉 no goals
                             -/
  le_inf (blimsup_mono <| by tauto) (blimsup_mono <| by tauto)
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem bliminf_sup_le_inf_aux_left :
    (blimsup u f fun x => p x ∧ q x) ≤ blimsup u f p :=
  blimsup_and_le_inf.trans inf_le_left


@[simp]
theorem bliminf_sup_le_inf_aux_right :
    (blimsup u f fun x => p x ∧ q x) ≤ blimsup u f q :=
  blimsup_and_le_inf.trans inf_le_right

-- @[simp] -- Porting note: simp_nf linter, lhs simplifies, added _aux simp version below

theorem bliminf_sup_le_and : bliminf u f p ⊔ bliminf u f q ≤ bliminf u f fun x => p x ∧ q x :=
  blimsup_and_le_inf (α := αᵒᵈ)


@[simp]
theorem bliminf_sup_le_and_aux_left : bliminf u f p ≤ bliminf u f fun x => p x ∧ q x :=
  le_sup_left.trans bliminf_sup_le_and


@[simp]
theorem bliminf_sup_le_and_aux_right : bliminf u f q ≤ bliminf u f fun x => p x ∧ q x :=
  le_sup_right.trans bliminf_sup_le_and


/-- See also `Filter.blimsup_or_eq_sup`. -/
-- @[simp] -- Porting note: simp_nf linter, lhs simplifies, added _aux simp versions below
theorem blimsup_sup_le_or : blimsup u f p ⊔ blimsup u f q ≤ blimsup u f fun x => p x ∨ q x :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝ : CompleteLattice α
                               f : Filter β
                               p q : β → Prop
                               u : β → α
                               ⊢ ∀ (x : β), p x → Or (p x) (q x)
                             -/
                             /-
                               🎉 no goals
                             -/
  sup_le (blimsup_mono <| by tauto) (blimsup_mono <| by tauto)
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem bliminf_sup_le_or_aux_left : blimsup u f p ≤ blimsup u f fun x => p x ∨ q x :=
  le_sup_left.trans blimsup_sup_le_or


@[simp]
theorem bliminf_sup_le_or_aux_right : blimsup u f q ≤ blimsup u f fun x => p x ∨ q x :=
  le_sup_right.trans blimsup_sup_le_or


/-- See also `Filter.bliminf_or_eq_inf`. -/
--@[simp] -- Porting note: simp_nf linter, lhs simplifies, added _aux simp versions below
theorem bliminf_or_le_inf : (bliminf u f fun x => p x ∨ q x) ≤ bliminf u f p ⊓ bliminf u f q :=
  blimsup_sup_le_or (α := αᵒᵈ)


@[simp]
theorem bliminf_or_le_inf_aux_left : (bliminf u f fun x => p x ∨ q x) ≤ bliminf u f p :=
  bliminf_or_le_inf.trans inf_le_left


@[simp]
theorem bliminf_or_le_inf_aux_right : (bliminf u f fun x => p x ∨ q x) ≤ bliminf u f q :=
  bliminf_or_le_inf.trans inf_le_right


theorem _root_.OrderIso.apply_blimsup [CompleteLattice γ] (e : α ≃o γ) :
    e (blimsup u f p) = blimsup (e ∘ u) f p := by
  simp only [blimsup_eq, map_sInf, Function.comp_apply, e.image_eq_preimage,
    Set.preimage_setOf_eq, e.le_symm_apply]


@[deprecated (since := "2024-07-21")]
alias OrderIso.apply_blimsup := OrderIso.apply_blimsup


theorem _root_.OrderIso.apply_bliminf [CompleteLattice γ] (e : α ≃o γ) :
    e (bliminf u f p) = bliminf (e ∘ u) f p :=
  e.dual.apply_blimsup


@[deprecated (since := "2024-07-21")]
alias OrderIso.apply_bliminf := OrderIso.apply_bliminf


theorem _root_.sSupHom.apply_blimsup_le [CompleteLattice γ] (g : sSupHom α γ) :
    g (blimsup u f p) ≤ blimsup (g ∘ u) f p := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : CompleteLattice α
    f : Filter β
    p : β → Prop
    u : β → α
    inst✝ : CompleteLattice γ
    g : sSupHom α γ
    ⊢ LE.le (g (Filter.blimsup u f p)) (Filter.blimsup (Function.comp (⇑g) u) f p)
  -/
  simp only [blimsup_eq_iInf_biSup, Function.comp]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : CompleteLattice α
    f : Filter β
    p : β → Prop
    u : β → α
    inst✝ : CompleteLattice γ
    g : sSupHom α γ
    ⊢ LE.le (g (iInf fun s => iInf fun h => iSup fun b => iSup fun x => u b)) (iIn …
  -/
  refine ((OrderHomClass.mono g).map_iInf₂_le _).trans ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : CompleteLattice α
    f : Filter β
    p : β → Prop
    u : β → α
    inst✝ : CompleteLattice γ
    g : sSupHom α γ
    ⊢ LE.le (iInf fun i => iInf fun j => g (iSup fun b => iSup fun x => u b)) (iIn …
  -/
  simp only [_root_.map_iSup, le_refl]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-21")]
alias SupHom.apply_blimsup_le := sSupHom.apply_blimsup_le


theorem _root_.sInfHom.le_apply_bliminf [CompleteLattice γ] (g : sInfHom α γ) :
    bliminf (g ∘ u) f p ≤ g (bliminf u f p) :=
  (sInfHom.dual g).apply_blimsup_le


@[deprecated (since := "2024-07-21")]
alias InfHom.le_apply_bliminf := sInfHom.le_apply_bliminf


lemma limsup_sup_filter {g} : limsup u (f ⊔ g) = limsup u f ⊔ limsup u g := by
  refine le_antisymm ?_
    (sup_le (limsup_le_limsup_of_le le_sup_left) (limsup_le_limsup_of_le le_sup_right))
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    g : Filter β
    ⊢ LE.le (Filter.limsup u (Max.max f g)) (Max.max (Filter.limsup u f) (Filter.l …
  -/
  simp_rw [limsup_eq, sInf_sup_eq, sup_sInf_eq, mem_setOf_eq, le_iInf₂_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    g : Filter β
    ⊢ ∀ (i : α), Filter.Eventually (fun n => LE.le (u n) i) f → ∀ (i_1 : α), Filte …
  -/
  intro a ha b hb
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    g : Filter β
    a : α
    ha : Filter.Eventually (fun n => LE.le (u n) a) f
    b : α
    hb : Filter.Eventually (fun n => LE.le (u n) b) g
    ⊢ LE.le (InfSet.sInf (setOf fun a => Filter.Eventually (fun n => LE.le (u n) a …
  -/
  exact sInf_le ⟨ha.mono fun _ h ↦ h.trans le_sup_left, hb.mono fun _ h ↦ h.trans le_sup_right⟩
  /-
    🎉 no goals
  -/


lemma liminf_sup_filter {g} : liminf u (f ⊔ g) = liminf u f ⊓ liminf u g :=
  limsup_sup_filter (α := αᵒᵈ)


@[simp]
theorem blimsup_or_eq_sup : (blimsup u f fun x => p x ∨ q x) = blimsup u f p ⊔ blimsup u f q := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    p q : β → Prop
    u : β → α
    ⊢ Eq (Filter.blimsup u f fun x => Or (p x) (q x)) (Max.max (Filter.blimsup u f …
  -/
  simp only [blimsup_eq_limsup, ← limsup_sup_filter, ← inf_sup_left, sup_principal, setOf_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem bliminf_or_eq_inf : (bliminf u f fun x => p x ∨ q x) = bliminf u f p ⊓ bliminf u f q :=
  blimsup_or_eq_sup (α := αᵒᵈ)


@[simp]
lemma blimsup_sup_not : blimsup u f p ⊔ blimsup u f (¬p ·) = limsup u f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    p : β → Prop
    u : β → α
    ⊢ Eq (Max.max (Filter.blimsup u f p) (Filter.blimsup u f fun x => Not (p x)))  …
  -/
  simp_rw [← blimsup_or_eq_sup, or_not, blimsup_true]
  /-
    🎉 no goals
  -/


@[simp]
lemma bliminf_inf_not : bliminf u f p ⊓ bliminf u f (¬p ·) = liminf u f :=
  blimsup_sup_not (α := αᵒᵈ)


@[simp]
lemma blimsup_not_sup : blimsup u f (¬p ·) ⊔ blimsup u f p = limsup u f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    p : β → Prop
    u : β → α
    ⊢ Eq (Max.max (Filter.blimsup u f fun x => Not (p x)) (Filter.blimsup u f p))  …
  -/
  simpa only [not_not] using blimsup_sup_not (p := (¬p ·))
  /-
    🎉 no goals
  -/


@[simp]
lemma bliminf_not_inf : bliminf u f (¬p ·) ⊓ bliminf u f p = liminf u f :=
  blimsup_not_sup (α := αᵒᵈ)


lemma limsup_piecewise {s : Set β} [DecidablePred (· ∈ s)] {v} :
    limsup (s.piecewise u v) f = blimsup u f (· ∈ s) ⊔ blimsup v f (· ∉ s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    s : Set β
    inst✝ : DecidablePred fun x => Membership.mem s x
    v : β → α
    ⊢ Eq (Filter.limsup (s.piecewise u v) f) (Max.max (Filter.blimsup u f fun x => …
  -/
  rw [← blimsup_sup_not (p := (· ∈ s))]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    s : Set β
    inst✝ : DecidablePred fun x => Membership.mem s x
    v : β → α
    ⊢ Eq (Max.max (Filter.blimsup (s.piecewise u v) f fun x => Membership.mem s x) …
  -/
  refine congr_arg₂ _ (blimsup_congr ?_) (blimsup_congr ?_) <;>
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteDistribLattice α
      f : Filter β
      u : β → α
      s : Set β
      inst✝ : DecidablePred fun x => Membership.mem s x
      v : β → α
      ⊢ Filter.Eventually (fun a => Membership.mem s a → Eq (s.piecewise u v a) (u a …
    -/
    /-
      🎉 no goals
    -/
    filter_upwards with _ h using by simp [h]
    /-
      🎉 no goals
    -/


lemma liminf_piecewise {s : Set β} [DecidablePred (· ∈ s)] {v} :
    liminf (s.piecewise u v) f = bliminf u f (· ∈ s) ⊓ bliminf v f (· ∉ s) :=
  limsup_piecewise (α := αᵒᵈ)


theorem sup_limsup [NeBot f] (a : α) : a ⊔ limsup u f = limsup (fun x => a ⊔ u x) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    inst✝ : f.NeBot
    a : α
    ⊢ Eq (Max.max a (Filter.limsup u f)) (Filter.limsup (fun x => Max.max a (u x)) …
  -/
  simp only [limsup_eq_iInf_iSup, iSup_sup_eq, sup_iInf₂_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    inst✝ : f.NeBot
    a : α
    ⊢ Eq (iInf fun i => iInf fun j => Max.max a (iSup fun a => iSup fun h => u a)) …
  -/
  congr; ext s; congr; ext hs; congr
  /-
    case e_s.h.e_s.h.e_a
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    inst✝ : f.NeBot
    a : α
    s : Set β
    hs : Membership.mem f s
    ⊢ Eq a (iSup fun x => iSup fun x => a)
  -/
  exact (biSup_const (nonempty_of_mem hs)).symm
  /-
    🎉 no goals
  -/


theorem inf_liminf [NeBot f] (a : α) : a ⊓ liminf u f = liminf (fun x => a ⊓ u x) f :=
  sup_limsup (α := αᵒᵈ) a


theorem sup_liminf (a : α) : a ⊔ liminf u f = liminf (fun x => a ⊔ u x) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (Max.max a (Filter.liminf u f)) (Filter.liminf (fun x => Max.max a (u x)) …
  -/
  simp only [liminf_eq_iSup_iInf]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (Max.max a (iSup fun s => iSup fun h => iInf fun a => iInf fun h => u a)) …
  -/
  rw [sup_comm, biSup_sup (⟨univ, univ_mem⟩ : ∃ i : Set β, i ∈ f)]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteDistribLattice α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (iSup fun i => iSup fun h => Max.max (iInf fun a => iInf fun h => u a) a) …
  -/
  simp_rw [iInf₂_sup_eq, sup_comm (a := a)]
  /-
    🎉 no goals
  -/


theorem inf_limsup (a : α) : a ⊓ limsup u f = limsup (fun x => a ⊓ u x) f :=
  sup_liminf (α := αᵒᵈ) a


theorem limsup_compl : (limsup u f)ᶜ = liminf (compl ∘ u) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    ⊢ Eq (HasCompl.compl (Filter.limsup u f)) (Filter.liminf (Function.comp HasCom …
  -/
  simp only [limsup_eq_iInf_iSup, compl_iInf, compl_iSup, liminf_eq_iSup_iInf, Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem liminf_compl : (liminf u f)ᶜ = limsup (compl ∘ u) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    ⊢ Eq (HasCompl.compl (Filter.liminf u f)) (Filter.limsup (Function.comp HasCom …
  -/
  simp only [limsup_eq_iInf_iSup, compl_iInf, compl_iSup, liminf_eq_iSup_iInf, Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem limsup_sdiff (a : α) : limsup u f \ a = limsup (fun b => u b \ a) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (SDiff.sdiff (Filter.limsup u f) a) (Filter.limsup (fun b => SDiff.sdiff  …
  -/
  simp only [limsup_eq_iInf_iSup, sdiff_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (Min.min (iInf fun s => iInf fun h => iSup fun a => iSup fun h => u a) (H …
  -/
  rw [biInf_inf (⟨univ, univ_mem⟩ : ∃ i : Set β, i ∈ f)]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (iInf fun i => iInf fun h => Min.min (iSup fun a => iSup fun h => u a) (H …
  -/
  simp_rw [inf_comm, inf_iSup₂_eq, inf_comm]
  /-
    🎉 no goals
  -/


theorem liminf_sdiff [NeBot f] (a : α) : liminf u f \ a = liminf (fun b => u b \ a) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    inst✝ : f.NeBot
    a : α
    ⊢ Eq (SDiff.sdiff (Filter.liminf u f) a) (Filter.liminf (fun b => SDiff.sdiff  …
  -/
  simp only [sdiff_eq, inf_comm _ aᶜ, inf_liminf]
  /-
    🎉 no goals
  -/


theorem sdiff_limsup [NeBot f] (a : α) : a \ limsup u f = liminf (fun b => a \ u b) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    inst✝ : f.NeBot
    a : α
    ⊢ Eq (SDiff.sdiff a (Filter.limsup u f)) (Filter.liminf (fun b => SDiff.sdiff  …
  -/
  rw [← compl_inj_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    inst✝ : f.NeBot
    a : α
    ⊢ Eq (HasCompl.compl (SDiff.sdiff a (Filter.limsup u f))) (HasCompl.compl (Fil …
  -/
  simp only [sdiff_eq, liminf_compl, comp_def, compl_inf, compl_compl, sup_limsup]
  /-
    🎉 no goals
  -/


theorem sdiff_liminf (a : α) : a \ liminf u f = limsup (fun b => a \ u b) f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (SDiff.sdiff a (Filter.liminf u f)) (Filter.limsup (fun b => SDiff.sdiff  …
  -/
  rw [← compl_inj_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : CompleteBooleanAlgebra α
    f : Filter β
    u : β → α
    a : α
    ⊢ Eq (HasCompl.compl (SDiff.sdiff a (Filter.liminf u f))) (HasCompl.compl (Fil …
  -/
  simp only [sdiff_eq, limsup_compl, comp_def, compl_inf, compl_compl, sup_liminf]
  /-
    🎉 no goals
  -/


lemma mem_liminf_iff_eventually_mem : (a ∈ liminf s 𝓕) ↔ (∀ᶠ i in 𝓕, a ∈ s i) := by
  simpa only [liminf_eq_iSup_iInf, iSup_eq_iUnion, iInf_eq_iInter, mem_iUnion, mem_iInter]
    using ⟨fun ⟨S, hS, hS'⟩ ↦ mem_of_superset hS (by tauto), fun h ↦ ⟨{i | a ∈ s i}, h, by tauto⟩⟩


lemma mem_limsup_iff_frequently_mem : (a ∈ limsup s 𝓕) ↔ (∃ᶠ i in 𝓕, a ∈ s i) := by
  simp only [Filter.Frequently, iff_not_comm, ← mem_compl_iff, limsup_compl, comp_apply,
    mem_liminf_iff_eventually_mem]


theorem cofinite.blimsup_set_eq :
    blimsup s cofinite p = { x | { n | p n ∧ x ∈ s n }.Infinite } := by
  /-
    α : Type u_1
    ι : Type u_4
    p : ι → Prop
    s : ι → Set α
    ⊢ Eq (Filter.blimsup s Filter.cofinite p) (setOf fun x => (setOf fun n => And  …
  -/
  simp only [blimsup_eq, le_eq_subset, eventually_cofinite, not_forall, sInf_eq_sInter, exists_prop]
  /-
    α : Type u_1
    ι : Type u_4
    p : ι → Prop
    s : ι → Set α
    ⊢ Eq (setOf fun a => (setOf fun x => And (p x) (Not (HasSubset.Subset (s x) a) …
  -/
  ext x
  /-
    case h
    α : Type u_1
    ι : Type u_4
    p : ι → Prop
    s : ι → Set α
    x : α
    ⊢ Iff (Membership.mem (setOf fun a => (setOf fun x => And (p x) (Not (HasSubse …
  -/
  refine ⟨fun h => ?_, fun hx t h => ?_⟩ <;> contrapose! h
    /-
      case h.refine_1
      α : Type u_1
      ι : Type u_4
      p : ι → Prop
      s : ι → Set α
      x : α
      h : Not (Membership.mem (setOf fun x => (setOf fun n => And (p n) (Membership. …
      ⊢ Not (Membership.mem (setOf fun a => (setOf fun x => And (p x) (Not (HasSubse …
    -/
  · simp only [mem_sInter, mem_setOf_eq, not_forall, exists_prop]
    /-
      case h.refine_1
      α : Type u_1
      ι : Type u_4
      p : ι → Prop
      s : ι → Set α
      x : α
      h : Not (Membership.mem (setOf fun x => (setOf fun n => And (p n) (Membership. …
      ⊢ Exists fun x_1 => And (setOf fun x => And (p x) (Not (HasSubset.Subset (s x) …
    -/
    exact ⟨{x}ᶜ, by simpa using h, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_1
      ι : Type u_4
      p : ι → Prop
      s : ι → Set α
      x : α
      hx : Membership.mem (setOf fun x => (setOf fun n => And (p n) (Membership.mem  …
      t : Set α
      h : Not (Membership.mem t x)
      ⊢ Not (Membership.mem (setOf fun a => (setOf fun x => And (p x) (Not (HasSubse …
    -/
  · exact hx.mono fun i hi => ⟨hi.1, fun hit => h (hit hi.2)⟩
    /-
      🎉 no goals
    -/


theorem cofinite.bliminf_set_eq : bliminf s cofinite p = { x | { n | p n ∧ x ∉ s n }.Finite } := by
  /-
    α : Type u_1
    ι : Type u_4
    p : ι → Prop
    s : ι → Set α
    ⊢ Eq (Filter.bliminf s Filter.cofinite p) (setOf fun x => (setOf fun n => And  …
  -/
  rw [← compl_inj_iff]
  simp only [bliminf_eq_iSup_biInf, compl_iInf, compl_iSup, ← blimsup_eq_iInf_biSup,
    cofinite.blimsup_set_eq]
  /-
    α : Type u_1
    ι : Type u_4
    p : ι → Prop
    s : ι → Set α
    ⊢ Eq (setOf fun x => (setOf fun n => And (p n) (Membership.mem (HasCompl.compl …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- In other words, `limsup cofinite s` is the set of elements lying inside the family `s`
infinitely often. -/
theorem cofinite.limsup_set_eq : limsup s cofinite = { x | { n | x ∈ s n }.Infinite } := by
  /-
    α : Type u_1
    ι : Type u_4
    s : ι → Set α
    ⊢ Eq (Filter.limsup s Filter.cofinite) (setOf fun x => (setOf fun n => Members …
  -/
  simp only [← cofinite.blimsup_true s, cofinite.blimsup_set_eq, true_and]
  /-
    🎉 no goals
  -/


/-- In other words, `liminf cofinite s` is the set of elements lying outside the family `s`
finitely often. -/
theorem cofinite.liminf_set_eq : liminf s cofinite = { x | { n | x ∉ s n }.Finite } := by
  /-
    α : Type u_1
    ι : Type u_4
    s : ι → Set α
    ⊢ Eq (Filter.liminf s Filter.cofinite) (setOf fun x => (setOf fun n => Not (Me …
  -/
  simp only [← cofinite.bliminf_true s, cofinite.bliminf_set_eq, true_and]
  /-
    🎉 no goals
  -/


theorem exists_forall_mem_of_hasBasis_mem_blimsup {l : Filter β} {b : ι → Set β} {q : ι → Prop}
    (hl : l.HasBasis q b) {u : β → Set α} {p : β → Prop} {x : α} (hx : x ∈ blimsup u l p) :
    ∃ f : { i | q i } → β, ∀ i, x ∈ u (f i) ∧ p (f i) ∧ f i ∈ b i := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    l : Filter β
    b : ι → Set β
    q : ι → Prop
    hl : l.HasBasis q b
    u : β → Set α
    p : β → Prop
    x : α
    hx : Membership.mem (Filter.blimsup u l p) x
    ⊢ Exists fun f => ∀ (i : ↑(setOf fun i => q i)), And (Membership.mem (u (f i)) …
  -/
  rw [blimsup_eq_iInf_biSup] at hx
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    l : Filter β
    b : ι → Set β
    q : ι → Prop
    hl : l.HasBasis q b
    u : β → Set α
    p : β → Prop
    x : α
    hx : Membership.mem (iInf fun s => iInf fun h => iSup fun b => iSup fun x => u …
    ⊢ Exists fun f => ∀ (i : ↑(setOf fun i => q i)), And (Membership.mem (u (f i)) …
  -/
  simp only [iSup_eq_iUnion, iInf_eq_iInter, mem_iInter, mem_iUnion, exists_prop] at hx
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    l : Filter β
    b : ι → Set β
    q : ι → Prop
    hl : l.HasBasis q b
    u : β → Set α
    p : β → Prop
    x : α
    hx : ∀ (i : Set β), Membership.mem l i → Exists fun i_2 => And (And (p i_2) (M …
    ⊢ Exists fun f => ∀ (i : ↑(setOf fun i => q i)), And (Membership.mem (u (f i)) …
  -/
  choose g hg hg' using hx
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    l : Filter β
    b : ι → Set β
    q : ι → Prop
    hl : l.HasBasis q b
    u : β → Set α
    p : β → Prop
    x : α
    g : (i : Set β) → Membership.mem l i → β
    hg : ∀ (i : Set β) (i_1 : Membership.mem l i), And (p (g i i_1)) (Membership.m …
    hg' : ∀ (i : Set β) (i_1 : Membership.mem l i), Membership.mem (u (g i i_1)) x
    ⊢ Exists fun f => ∀ (i : ↑(setOf fun i => q i)), And (Membership.mem (u (f i)) …
  -/
  refine ⟨fun i : { i | q i } => g (b i) (hl.mem_of_mem i.2), fun i => ⟨?_, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      l : Filter β
      b : ι → Set β
      q : ι → Prop
      hl : l.HasBasis q b
      u : β → Set α
      p : β → Prop
      x : α
      g : (i : Set β) → Membership.mem l i → β
      hg : ∀ (i : Set β) (i_1 : Membership.mem l i), And (p (g i i_1)) (Membership.m …
      hg' : ∀ (i : Set β) (i_1 : Membership.mem l i), Membership.mem (u (g i i_1)) x
      i : ↑(setOf fun i => q i)
      ⊢ Membership.mem (u ((fun i => g (b ↑i) ⋯) i)) x
    -/
  · exact hg' (b i) (hl.mem_of_mem i.2)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      l : Filter β
      b : ι → Set β
      q : ι → Prop
      hl : l.HasBasis q b
      u : β → Set α
      p : β → Prop
      x : α
      g : (i : Set β) → Membership.mem l i → β
      hg : ∀ (i : Set β) (i_1 : Membership.mem l i), And (p (g i i_1)) (Membership.m …
      hg' : ∀ (i : Set β) (i_1 : Membership.mem l i), Membership.mem (u (g i i_1)) x
      i : ↑(setOf fun i => q i)
      ⊢ And (p ((fun i => g (b ↑i) ⋯) i)) (Membership.mem (b ↑i) ((fun i => g (b ↑i) …
    -/
  · exact hg (b i) (hl.mem_of_mem i.2)
    /-
      🎉 no goals
    -/


theorem exists_forall_mem_of_hasBasis_mem_blimsup' {l : Filter β} {b : ι → Set β}
    (hl : l.HasBasis (fun _ => True) b) {u : β → Set α} {p : β → Prop} {x : α}
    (hx : x ∈ blimsup u l p) : ∃ f : ι → β, ∀ i, x ∈ u (f i) ∧ p (f i) ∧ f i ∈ b i := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    l : Filter β
    b : ι → Set β
    hl : l.HasBasis (fun x => True) b
    u : β → Set α
    p : β → Prop
    x : α
    hx : Membership.mem (Filter.blimsup u l p) x
    ⊢ Exists fun f => ∀ (i : ι), And (Membership.mem (u (f i)) x) (And (p (f i)) ( …
  -/
  obtain ⟨f, hf⟩ := exists_forall_mem_of_hasBasis_mem_blimsup hl hx
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    l : Filter β
    b : ι → Set β
    hl : l.HasBasis (fun x => True) b
    u : β → Set α
    p : β → Prop
    x : α
    hx : Membership.mem (Filter.blimsup u l p) x
    f : ↑(setOf fun i => True) → β
    hf : ∀ (i : ↑(setOf fun i => True)), And (Membership.mem (u (f i)) x) (And (p  …
    ⊢ Exists fun f => ∀ (i : ι), And (Membership.mem (u (f i)) x) (And (p (f i)) ( …
  -/
  exact ⟨fun i => f ⟨i, trivial⟩, fun i => hf ⟨i, trivial⟩⟩
  /-
    🎉 no goals
  -/


theorem frequently_lt_of_lt_limsSup {f : Filter α} [ConditionallyCompleteLinearOrder α] {a : α}
    (hf : f.IsCobounded (· ≤ ·) := by isBoundedDefault)
    (h : a < limsSup f) : ∃ᶠ n in f, a < n := by
  /-
    α : Type u_1
    f : Filter α
    inst✝ : ConditionallyCompleteLinearOrder α
    a : α
    hf : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f) _auto✝
    h : LT.lt a f.limsSup
    ⊢ Filter.Frequently (fun n => LT.lt a n) f
  -/
  contrapose! h
  /-
    α : Type u_1
    f : Filter α
    inst✝ : ConditionallyCompleteLinearOrder α
    a : α
    hf : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f) _auto✝
    h : Not (Filter.Frequently (fun n => LT.lt a n) f)
    ⊢ LE.le f.limsSup a
  -/
  simp only [not_frequently, not_lt] at h
  /-
    α : Type u_1
    f : Filter α
    inst✝ : ConditionallyCompleteLinearOrder α
    a : α
    hf : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) f) _auto✝
    h : Filter.Eventually (fun x => LE.le x a) f
    ⊢ LE.le f.limsSup a
  -/
  exact limsSup_le_of_le hf h
  /-
    🎉 no goals
  -/


theorem frequently_lt_of_limsInf_lt {f : Filter α} [ConditionallyCompleteLinearOrder α] {a : α}
    (hf : f.IsCobounded (· ≥ ·) := by isBoundedDefault)
    (h : limsInf f < a) : ∃ᶠ n in f, n < a :=
  frequently_lt_of_lt_limsSup (α := OrderDual α) hf h


theorem eventually_lt_of_lt_liminf {f : Filter α} [ConditionallyCompleteLinearOrder β] {u : α → β}
    {b : β} (h : b < liminf u f)
    (hu : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) :
    ∀ᶠ a in f, b < u a := by
  obtain ⟨c, hc, hbc⟩ : ∃ (c : β) (_ : c ∈ { c : β | ∀ᶠ n : α in f, c ≤ u n }), b < c := by
    simp_rw [exists_prop]
    exact exists_lt_of_lt_csSup hu h
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : Filter α
    inst✝ : ConditionallyCompleteLinearOrder β
    u : α → β
    b : β
    h : LT.lt b (Filter.liminf u f)
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    c : β
    hc : Membership.mem (setOf fun c => Filter.Eventually (fun n => LE.le c (u n)) …
    hbc : LT.lt b c
    ⊢ Filter.Eventually (fun a => LT.lt b (u a)) f
  -/
  exact hc.mono fun x hx => lt_of_lt_of_le hbc hx
  /-
    🎉 no goals
  -/


theorem eventually_lt_of_limsup_lt {f : Filter α} [ConditionallyCompleteLinearOrder β] {u : α → β}
    {b : β} (h : limsup u f < b)
    (hu : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault) :
    ∀ᶠ a in f, u a < b :=
  eventually_lt_of_lt_liminf (β := βᵒᵈ) h hu


theorem le_limsup_of_frequently_le {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α}
    {u : α → β} {b : β} (hu_le : ∃ᶠ x in f, b ≤ u x)
    (hu : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault) :
    b ≤ limsup u f := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu_le : Filter.Frequently (fun x => LE.le b (u x)) f
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ LE.le b (Filter.limsup u f)
  -/
  revert hu_le
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ Filter.Frequently (fun x => LE.le b (u x)) f → LE.le b (Filter.limsup u f)
  -/
  rw [← not_imp_not, not_frequently]
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ Not (LE.le b (Filter.limsup u f)) → Filter.Eventually (fun x => Not (LE.le b …
  -/
  simp_rw [← lt_iff_not_ge]
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ LT.lt (Filter.limsup u f) b → Filter.Eventually (fun x => LT.lt (u x) b) f
  -/
  exact fun h => eventually_lt_of_limsup_lt h hu
  /-
    🎉 no goals
  -/


theorem liminf_le_of_frequently_le {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α}
    {u : α → β} {b : β} (hu_le : ∃ᶠ x in f, u x ≤ b)
    (hu : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) :
    liminf u f ≤ b :=
  le_limsup_of_frequently_le (β := βᵒᵈ) hu_le hu


theorem frequently_lt_of_lt_limsup {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α}
    {u : α → β} {b : β}
    (hu : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h : b < limsup u f) : ∃ᶠ x in f, b < u x := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h : LT.lt b (Filter.limsup u f)
    ⊢ Filter.Frequently (fun x => LT.lt b (u x)) f
  -/
  contrapose! h
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h : Not (Filter.Frequently (fun x => LT.lt b (u x)) f)
    ⊢ LE.le (Filter.limsup u f) b
  -/
  apply limsSup_le_of_le hu
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    b : β
    hu : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h : Not (Filter.Frequently (fun x => LT.lt b (u x)) f)
    ⊢ Filter.Eventually (fun n => LE.le n b) (Filter.map u f)
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem frequently_lt_of_liminf_lt {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α}
    {u : α → β} {b : β}
    (hu : f.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h : liminf u f < b) : ∃ᶠ x in f, u x < b :=
  frequently_lt_of_lt_limsup (β := βᵒᵈ) hu h


theorem limsup_le_iff {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α} {u : α → β} {x : β}
    (h₁ : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h₂ : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault) :
    limsup u f ≤ x ↔ ∀ y > x, ∀ᶠ a in f, u a < y := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    x : β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ Iff (LE.le (Filter.limsup u f) x) (∀ (y : β), GT.gt y x → Filter.Eventually  …
  -/
  refine ⟨fun h _ h' ↦ eventually_lt_of_limsup_lt (lt_of_le_of_lt h h') h₂, fun h ↦ ?_⟩
  --Two cases: Either `x` is a cluster point from above, or it is not.
  --In the first case, we use `forall_lt_iff_le'` and split an interval.
  --In the second case, the function `u` must eventually be smaller or equal to `x`.
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    x : β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
    ⊢ LE.le (Filter.limsup u f) x
  -/
  by_cases h' : ∀ y > x, ∃ z, x < z ∧ z < y
    /-
      case pos
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : ∀ (y : β), GT.gt y x → Exists fun z => And (LT.lt x z) (LT.lt z y)
      ⊢ LE.le (Filter.limsup u f) x
    -/
  · rw [← forall_lt_iff_le']
    /-
      case pos
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : ∀ (y : β), GT.gt y x → Exists fun z => And (LT.lt x z) (LT.lt z y)
      ⊢ ∀ ⦃c : β⦄, LT.lt x c → LT.lt (Filter.limsup u f) c
    -/
    intro y x_y
    /-
      case pos
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : ∀ (y : β), GT.gt y x → Exists fun z => And (LT.lt x z) (LT.lt z y)
      y : β
      x_y : LT.lt x y
      ⊢ LT.lt (Filter.limsup u f) y
    -/
    rcases h' y x_y with ⟨z, x_z, z_y⟩
    /-
      case pos.intro.intro
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : ∀ (y : β), GT.gt y x → Exists fun z => And (LT.lt x z) (LT.lt z y)
      y : β
      x_y : LT.lt x y
      z : β
      x_z : LT.lt x z
      z_y : LT.lt z y
      ⊢ LT.lt (Filter.limsup u f) y
    -/
    exact lt_of_le_of_lt (limsup_le_of_le h₁ ((h z x_z).mono (fun _ ↦ le_of_lt))) z_y
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : Not (∀ (y : β), GT.gt y x → Exists fun z => And (LT.lt x z) (LT.lt z y))
      ⊢ LE.le (Filter.limsup u f) x
    -/
  · apply limsup_le_of_le h₁
    /-
      case neg
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : Not (∀ (y : β), GT.gt y x → Exists fun z => And (LT.lt x z) (LT.lt z y))
      ⊢ Filter.Eventually (fun n => LE.le (u n) x) f
    -/
    set_option push_neg.use_distrib true in push_neg at h'
    /-
      case neg
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      h' : Exists fun y => And (GT.gt y x) (∀ (z : β), Or (LE.le z x) (LE.le y z))
      ⊢ Filter.Eventually (fun n => LE.le (u n) x) f
    -/
    rcases h' with ⟨z, x_z, hz⟩
    /-
      case neg.intro.intro
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), GT.gt y x → Filter.Eventually (fun a => LT.lt (u a) y) f
      z : β
      x_z : GT.gt z x
      hz : ∀ (z_1 : β), Or (LE.le z_1 x) (LE.le z z_1)
      ⊢ Filter.Eventually (fun n => LE.le (u n) x) f
    -/
    exact (h z x_z).mono  <| fun w hw ↦ (or_iff_left (not_le_of_lt hw)).1 (hz (u w))
    /-
      🎉 no goals
    -/


theorem le_limsup_iff {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α} {u : α → β} {x : β}
    (h₁ : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h₂ : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault) :
    x ≤ limsup u f ↔ ∀ y < x, ∃ᶠ a in f, y < u a := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    x : β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ Iff (LE.le x (Filter.limsup u f)) (∀ (y : β), LT.lt y x → Filter.Frequently  …
  -/
  refine ⟨fun h _ h' ↦ frequently_lt_of_lt_limsup h₁ (lt_of_lt_of_le h' h), fun h ↦ ?_⟩
  --Two cases: Either `x` is a cluster point from below, or it is not.
  --In the first case, we use `forall_lt_iff_le` and split an interval.
  --In the second case, the function `u` must frequently be larger or equal to `x`.
  /-
    α : Type u_6
    β : Type u_7
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u : α → β
    x : β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
    ⊢ LE.le x (Filter.limsup u f)
  -/
  by_cases h' : ∀ y < x, ∃ z, y < z ∧ z < x
    /-
      case pos
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : ∀ (y : β), LT.lt y x → Exists fun z => And (LT.lt y z) (LT.lt z x)
      ⊢ LE.le x (Filter.limsup u f)
    -/
  · rw [← forall_lt_iff_le]
    /-
      case pos
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : ∀ (y : β), LT.lt y x → Exists fun z => And (LT.lt y z) (LT.lt z x)
      ⊢ ∀ ⦃c : β⦄, LT.lt c x → LT.lt c (Filter.limsup u f)
    -/
    intro y y_x
    /-
      case pos
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : ∀ (y : β), LT.lt y x → Exists fun z => And (LT.lt y z) (LT.lt z x)
      y : β
      y_x : LT.lt y x
      ⊢ LT.lt y (Filter.limsup u f)
    -/
    rcases h' y y_x with ⟨z, y_z, z_x⟩
    /-
      case pos.intro.intro
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : ∀ (y : β), LT.lt y x → Exists fun z => And (LT.lt y z) (LT.lt z x)
      y : β
      y_x : LT.lt y x
      z : β
      y_z : LT.lt y z
      z_x : LT.lt z x
      ⊢ LT.lt y (Filter.limsup u f)
    -/
    exact lt_of_lt_of_le y_z (le_limsup_of_frequently_le ((h z z_x).mono (fun _ ↦ le_of_lt)) h₂)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : Not (∀ (y : β), LT.lt y x → Exists fun z => And (LT.lt y z) (LT.lt z x))
      ⊢ LE.le x (Filter.limsup u f)
    -/
  · apply le_limsup_of_frequently_le _ h₂
    /-
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : Not (∀ (y : β), LT.lt y x → Exists fun z => And (LT.lt y z) (LT.lt z x))
      ⊢ Filter.Frequently (fun x_1 => LE.le x (u x_1)) f
    -/
    set_option push_neg.use_distrib true in push_neg at h'
    /-
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      h' : Exists fun y => And (LT.lt y x) (∀ (z : β), Or (LE.le z y) (LE.le x z))
      ⊢ Filter.Frequently (fun x_1 => LE.le x (u x_1)) f
    -/
    rcases h' with ⟨z, z_x, hz⟩
    /-
      case intro.intro
      α : Type u_6
      β : Type u_7
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u : α → β
      x : β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h : ∀ (y : β), LT.lt y x → Filter.Frequently (fun a => LT.lt y (u a)) f
      z : β
      z_x : LT.lt z x
      hz : ∀ (z_1 : β), Or (LE.le z_1 z) (LE.le x z_1)
      ⊢ Filter.Frequently (fun x_1 => LE.le x (u x_1)) f
    -/
    exact (h z z_x).mono <| fun w hw ↦ (or_iff_right (not_le_of_lt hw)).1 (hz (u w))
    /-
      🎉 no goals
    -/


theorem le_liminf_iff {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α} {u : α → β} {x : β}
    (h₁ : f.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h₂ : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) :
    x ≤ liminf u f ↔ ∀ y < x, ∀ᶠ a in f, y < u a := limsup_le_iff (β := βᵒᵈ) h₁ h₂


theorem liminf_le_iff {α β} [ConditionallyCompleteLinearOrder β] {f : Filter α} {u : α → β} {x : β}
    (h₁ : f.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h₂ : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) :
    liminf u f ≤ x ↔ ∀ y > x, ∃ᶠ a in f, u a < y := le_limsup_iff (β := βᵒᵈ) h₁ h₂


set_option linter.unusedVariables false in
theorem lt_mem_sets_of_limsSup_lt (h : f.IsBounded (· ≤ ·)) (l : f.limsSup < b) :
    ∀ᶠ a in f, a < b :=
  let ⟨c, (h : ∀ᶠ a in f, a ≤ c), hcb⟩ := exists_lt_of_csInf_lt h l
  mem_of_superset h fun _a => hcb.trans_le'


theorem gt_mem_sets_of_limsInf_gt : f.IsBounded (· ≥ ·) → b < f.limsInf → ∀ᶠ a in f, b < a :=
  @lt_mem_sets_of_limsSup_lt αᵒᵈ _ _ _


open Classical in
/-- Given an indexed family of sets `s j` over `j : Subtype p` and a function `f`, then
`liminf_reparam j` is equal to `j` if `f` is bounded below on `s j`, and otherwise to some
index `k` such that `f` is bounded below on `s k` (if there exists one).
To ensure good measurability behavior, this index `k` is chosen as the minimal suitable index.
This function is used to write down a liminf in a measurable way,
in `Filter.HasBasis.liminf_eq_ciSup_ciInf` and `Filter.HasBasis.liminf_eq_ite`. -/
noncomputable def liminf_reparam
    (f : ι → α) (s : ι' → Set ι) (p : ι' → Prop) [Countable (Subtype p)] [Nonempty (Subtype p)]
    (j : Subtype p) : Subtype p :=
  let m : Set (Subtype p) := {j | BddBelow (range (fun (i : s j) ↦ f i))}
  let g : ℕ → Subtype p := (exists_surjective_nat _).choose
  have Z : ∃ n, g n ∈ m ∨ ∀ j, j ∉ m := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      f✝ : Filter α
      b : α
      f : ι → α
      s : ι' → Set ι
      p : ι' → Prop
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      j : Subtype p
      m : Set (Subtype p) := setOf fun j => BddBelow (Set.range fun i => f ↑i)
      g : Nat → Subtype p := ⋯.choose
      ⊢ Exists fun n => Or (Membership.mem m (g n)) (∀ (j : Subtype p), Not (Members …
    -/
    by_cases H : ∃ j, j ∈ m
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        ι' : Type u_5
        inst✝² : ConditionallyCompleteLinearOrder α
        f✝ : Filter α
        b : α
        f : ι → α
        s : ι' → Set ι
        p : ι' → Prop
        inst✝¹ : Countable (Subtype p)
        inst✝ : Nonempty (Subtype p)
        j : Subtype p
        m : Set (Subtype p) := setOf fun j => BddBelow (Set.range fun i => f ↑i)
        g : Nat → Subtype p := ⋯.choose
        H : Exists fun j => Membership.mem m j
        ⊢ Exists fun n => Or (Membership.mem m (g n)) (∀ (j : Subtype p), Not (Members …
      -/
    · rcases H with ⟨j, hj⟩
      /-
        case pos.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        ι' : Type u_5
        inst✝² : ConditionallyCompleteLinearOrder α
        f✝ : Filter α
        b : α
        f : ι → α
        s : ι' → Set ι
        p : ι' → Prop
        inst✝¹ : Countable (Subtype p)
        inst✝ : Nonempty (Subtype p)
        j✝ : Subtype p
        m : Set (Subtype p) := setOf fun j => BddBelow (Set.range fun i => f ↑i)
        g : Nat → Subtype p := ⋯.choose
        j : Subtype p
        hj : Membership.mem m j
        ⊢ Exists fun n => Or (Membership.mem m (g n)) (∀ (j : Subtype p), Not (Members …
      -/
      rcases (exists_surjective_nat (Subtype p)).choose_spec j with ⟨n, rfl⟩
      /-
        case pos.intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        ι' : Type u_5
        inst✝² : ConditionallyCompleteLinearOrder α
        f✝ : Filter α
        b : α
        f : ι → α
        s : ι' → Set ι
        p : ι' → Prop
        inst✝¹ : Countable (Subtype p)
        inst✝ : Nonempty (Subtype p)
        j : Subtype p
        m : Set (Subtype p) := setOf fun j => BddBelow (Set.range fun i => f ↑i)
        g : Nat → Subtype p := ⋯.choose
        n : Nat
        hj : Membership.mem m (⋯.choose n)
        ⊢ Exists fun n => Or (Membership.mem m (g n)) (∀ (j : Subtype p), Not (Members …
      -/
      exact ⟨n, Or.inl hj⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        ι' : Type u_5
        inst✝² : ConditionallyCompleteLinearOrder α
        f✝ : Filter α
        b : α
        f : ι → α
        s : ι' → Set ι
        p : ι' → Prop
        inst✝¹ : Countable (Subtype p)
        inst✝ : Nonempty (Subtype p)
        j : Subtype p
        m : Set (Subtype p) := setOf fun j => BddBelow (Set.range fun i => f ↑i)
        g : Nat → Subtype p := ⋯.choose
        H : Not (Exists fun j => Membership.mem m j)
        ⊢ Exists fun n => Or (Membership.mem m (g n)) (∀ (j : Subtype p), Not (Members …
      -/
    · push_neg at H
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        ι' : Type u_5
        inst✝² : ConditionallyCompleteLinearOrder α
        f✝ : Filter α
        b : α
        f : ι → α
        s : ι' → Set ι
        p : ι' → Prop
        inst✝¹ : Countable (Subtype p)
        inst✝ : Nonempty (Subtype p)
        j : Subtype p
        m : Set (Subtype p) := setOf fun j => BddBelow (Set.range fun i => f ↑i)
        g : Nat → Subtype p := ⋯.choose
        H : ∀ (j : Subtype p), Not (Membership.mem m j)
        ⊢ Exists fun n => Or (Membership.mem m (g n)) (∀ (j : Subtype p), Not (Members …
      -/
      exact ⟨0, Or.inr H⟩
      /-
        🎉 no goals
      -/
  if j ∈ m then j else g (Nat.find Z)


/-- Writing a liminf as a supremum of infimum, in a (possibly non-complete) conditionally complete
linear order. A reparametrization trick is needed to avoid taking the infimum of sets which are
not bounded below. -/
theorem HasBasis.liminf_eq_ciSup_ciInf {v : Filter ι}
    {p : ι' → Prop} {s : ι' → Set ι} [Countable (Subtype p)] [Nonempty (Subtype p)]
    (hv : v.HasBasis p s) {f : ι → α} (hs : ∀ (j : Subtype p), (s j).Nonempty)
    (H : ∃ (j : Subtype p), BddBelow (range (fun (i : s j) ↦ f i))) :
    liminf f v = ⨆ (j : Subtype p), ⨅ (i : s (liminf_reparam f s p j)), f i := by
  classical
  rcases H with ⟨j0, hj0⟩
  let m : Set (Subtype p) := {j | BddBelow (range (fun (i : s j) ↦ f i))}
  have : ∀ (j : Subtype p), Nonempty (s j) := fun j ↦ Nonempty.coe_sort (hs j)
  have A : ⋃ (j : Subtype p), ⋂ (i : s j), Iic (f i) =
         ⋃ (j : Subtype p), ⋂ (i : s (liminf_reparam f s p j)), Iic (f i) := by
    apply Subset.antisymm
    · apply iUnion_subset (fun j ↦ ?_)
      by_cases hj : j ∈ m
      · have : j = liminf_reparam f s p j := by simp only [m, liminf_reparam, hj, ite_true]
        conv_lhs => rw [this]
        apply subset_iUnion _ j
      · simp only [m, mem_setOf_eq, ← nonempty_iInter_Iic_iff, not_nonempty_iff_eq_empty] at hj
        simp only [hj, empty_subset]
    · apply iUnion_subset (fun j ↦ ?_)
      exact subset_iUnion (fun (k : Subtype p) ↦ (⋂ (i : s k), Iic (f i))) (liminf_reparam f s p j)
  have B : ∀ (j : Subtype p), ⋂ (i : s (liminf_reparam f s p j)), Iic (f i) =
                                Iic (⨅ (i : s (liminf_reparam f s p j)), f i) := by
    intro j
    apply (Iic_ciInf _).symm
    change liminf_reparam f s p j ∈ m
    by_cases Hj : j ∈ m
    · simpa only [m, liminf_reparam, if_pos Hj] using Hj
    · simp only [m, liminf_reparam, if_neg Hj]
      have Z : ∃ n, (exists_surjective_nat (Subtype p)).choose n ∈ m ∨ ∀ j, j ∉ m := by
        rcases (exists_surjective_nat (Subtype p)).choose_spec j0 with ⟨n, rfl⟩
        exact ⟨n, Or.inl hj0⟩
      rcases Nat.find_spec Z with hZ|hZ
      · exact hZ
      · exact (hZ j0 hj0).elim
  simp_rw [hv.liminf_eq_sSup_iUnion_iInter, A, B, sSup_iUnion_Iic]


open Classical in
/-- Writing a liminf as a supremum of infimum, in a (possibly non-complete) conditionally complete
linear order. A reparametrization trick is needed to avoid taking the infimum of sets which are
not bounded below. -/
theorem HasBasis.liminf_eq_ite {v : Filter ι} {p : ι' → Prop} {s : ι' → Set ι}
    [Countable (Subtype p)] [Nonempty (Subtype p)] (hv : v.HasBasis p s) (f : ι → α) :
    liminf f v = if ∃ (j : Subtype p), s j = ∅ then sSup univ else
      if ∀ (j : Subtype p), ¬BddBelow (range (fun (i : s j) ↦ f i)) then sSup ∅
      else ⨆ (j : Subtype p), ⨅ (i : s (liminf_reparam f s p j)), f i := by
  /-
    α : Type u_1
    ι : Type u_4
    ι' : Type u_5
    inst✝² : ConditionallyCompleteLinearOrder α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    inst✝¹ : Countable (Subtype p)
    inst✝ : Nonempty (Subtype p)
    hv : v.HasBasis p s
    f : ι → α
    ⊢ Eq (Filter.liminf f v) (ite (Exists fun j => Eq (s ↑j) EmptyCollection.empty …
  -/
  by_cases H : ∃ (j : Subtype p), s j = ∅
    /-
      case pos
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H : Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection
      ⊢ Eq (Filter.liminf f v) (ite (Exists fun j => Eq (s ↑j) EmptyCollection.empty …
    -/
  · rw [if_pos H]
    /-
      case pos
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H : Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection
      ⊢ Eq (Filter.liminf f v) (SupSet.sSup Set.univ)
    -/
    rcases H with ⟨j, hj⟩
    /-
      case pos.intro
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      j : Subtype p
      hj : Eq (s ↑j) EmptyCollection.emptyCollection
      ⊢ Eq (Filter.liminf f v) (SupSet.sSup Set.univ)
    -/
    simp [hv.liminf_eq_sSup_univ_of_empty j j.2 hj]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    ι : Type u_4
    ι' : Type u_5
    inst✝² : ConditionallyCompleteLinearOrder α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    inst✝¹ : Countable (Subtype p)
    inst✝ : Nonempty (Subtype p)
    hv : v.HasBasis p s
    f : ι → α
    H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
    ⊢ Eq (Filter.liminf f v) (ite (Exists fun j => Eq (s ↑j) EmptyCollection.empty …
  -/
  rw [if_neg H]
  /-
    case neg
    α : Type u_1
    ι : Type u_4
    ι' : Type u_5
    inst✝² : ConditionallyCompleteLinearOrder α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    inst✝¹ : Countable (Subtype p)
    inst✝ : Nonempty (Subtype p)
    hv : v.HasBasis p s
    f : ι → α
    H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
    ⊢ Eq (Filter.liminf f v) (ite (∀ (j : Subtype p), Not (BddBelow (Set.range fun …
  -/
  by_cases H' : ∀ (j : Subtype p), ¬BddBelow (range (fun (i : s j) ↦ f i))
  · have A : ∀ (j : Subtype p), ⋂ (i : s j), Iic (f i) = ∅ := by
      simp_rw [← not_nonempty_iff_eq_empty, nonempty_iInter_Iic_iff]
      exact H'
    /-
      case pos
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
      H' : ∀ (j : Subtype p), Not (BddBelow (Set.range fun i => f ↑i))
      A : ∀ (j : Subtype p), Eq (Set.iInter fun i => Set.Iic (f ↑i)) EmptyCollection …
      ⊢ Eq (Filter.liminf f v) (ite (∀ (j : Subtype p), Not (BddBelow (Set.range fun …
    -/
    simp_rw [if_pos H', hv.liminf_eq_sSup_iUnion_iInter, A, iUnion_empty]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    ι : Type u_4
    ι' : Type u_5
    inst✝² : ConditionallyCompleteLinearOrder α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    inst✝¹ : Countable (Subtype p)
    inst✝ : Nonempty (Subtype p)
    hv : v.HasBasis p s
    f : ι → α
    H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
    H' : Not (∀ (j : Subtype p), Not (BddBelow (Set.range fun i => f ↑i)))
    ⊢ Eq (Filter.liminf f v) (ite (∀ (j : Subtype p), Not (BddBelow (Set.range fun …
  -/
  rw [if_neg H']
  /-
    case neg
    α : Type u_1
    ι : Type u_4
    ι' : Type u_5
    inst✝² : ConditionallyCompleteLinearOrder α
    v : Filter ι
    p : ι' → Prop
    s : ι' → Set ι
    inst✝¹ : Countable (Subtype p)
    inst✝ : Nonempty (Subtype p)
    hv : v.HasBasis p s
    f : ι → α
    H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
    H' : Not (∀ (j : Subtype p), Not (BddBelow (Set.range fun i => f ↑i)))
    ⊢ Eq (Filter.liminf f v) (iSup fun j => iInf fun i => f ↑i)
  -/
  apply hv.liminf_eq_ciSup_ciInf
    /-
      case neg.hs
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
      H' : Not (∀ (j : Subtype p), Not (BddBelow (Set.range fun i => f ↑i)))
      ⊢ ∀ (j : Subtype p), (s ↑j).Nonempty
    -/
  · push_neg at H
    /-
      case neg.hs
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H' : Not (∀ (j : Subtype p), Not (BddBelow (Set.range fun i => f ↑i)))
      H : ∀ (j : Subtype p), (s ↑j).Nonempty
      ⊢ ∀ (j : Subtype p), (s ↑j).Nonempty
    -/
    simpa only [nonempty_iff_ne_empty] using H
    /-
      🎉 no goals
    -/
    /-
      case neg.H
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
      H' : Not (∀ (j : Subtype p), Not (BddBelow (Set.range fun i => f ↑i)))
      ⊢ Exists fun j => BddBelow (Set.range fun i => f ↑i)
    -/
  · push_neg at H'
    /-
      case neg.H
      α : Type u_1
      ι : Type u_4
      ι' : Type u_5
      inst✝² : ConditionallyCompleteLinearOrder α
      v : Filter ι
      p : ι' → Prop
      s : ι' → Set ι
      inst✝¹ : Countable (Subtype p)
      inst✝ : Nonempty (Subtype p)
      hv : v.HasBasis p s
      f : ι → α
      H : Not (Exists fun j => Eq (s ↑j) EmptyCollection.emptyCollection)
      H' : Exists fun j => BddBelow (Set.range fun i => f ↑i)
      ⊢ Exists fun j => BddBelow (Set.range fun i => f ↑i)
    -/
    exact H'
    /-
      🎉 no goals
    -/


/-- Given an indexed family of sets `s j` and a function `f`, then `limsup_reparam j` is equal
to `j` if `f` is bounded above on `s j`, and otherwise to some index `k` such that `f` is bounded
above on `s k` (if there exists one). To ensure good measurability behavior, this index `k` is
chosen as the minimal suitable index. This function is used to write down a limsup in a measurable
way, in `Filter.HasBasis.limsup_eq_ciInf_ciSup` and `Filter.HasBasis.limsup_eq_ite`. -/
noncomputable def limsup_reparam
    (f : ι → α) (s : ι' → Set ι) (p : ι' → Prop) [Countable (Subtype p)] [Nonempty (Subtype p)]
    (j : Subtype p) : Subtype p :=
  liminf_reparam (α := αᵒᵈ) f s p j


/-- Writing a limsup as an infimum of supremum, in a (possibly non-complete) conditionally complete
linear order. A reparametrization trick is needed to avoid taking the supremum of sets which are
not bounded above. -/
theorem HasBasis.limsup_eq_ciInf_ciSup {v : Filter ι}
    {p : ι' → Prop} {s : ι' → Set ι} [Countable (Subtype p)] [Nonempty (Subtype p)]
    (hv : v.HasBasis p s) {f : ι → α} (hs : ∀ (j : Subtype p), (s j).Nonempty)
    (H : ∃ (j : Subtype p), BddAbove (range (fun (i : s j) ↦ f i))) :
    limsup f v = ⨅ (j : Subtype p), ⨆ (i : s (limsup_reparam f s p j)), f i :=
  HasBasis.liminf_eq_ciSup_ciInf (α := αᵒᵈ) hv hs H


open Classical in
/-- Writing a limsup as an infimum of supremum, in a (possibly non-complete) conditionally complete
linear order. A reparametrization trick is needed to avoid taking the supremum of sets which are
not bounded below. -/
theorem HasBasis.limsup_eq_ite {v : Filter ι} {p : ι' → Prop} {s : ι' → Set ι}
    [Countable (Subtype p)] [Nonempty (Subtype p)] (hv : v.HasBasis p s) (f : ι → α) :
    limsup f v = if ∃ (j : Subtype p), s j = ∅ then sInf univ else
      if ∀ (j : Subtype p), ¬BddAbove (range (fun (i : s j) ↦ f i)) then sInf ∅
      else ⨅ (j : Subtype p), ⨆ (i : s (limsup_reparam f s p j)), f i :=
  HasBasis.liminf_eq_ite (α := αᵒᵈ) hv f


theorem Monotone.isBoundedUnder_le_comp_iff [Nonempty β] [LinearOrder β] [Preorder γ] [NoMaxOrder γ]
    {g : β → γ} {f : α → β} {l : Filter α} (hg : Monotone g) (hg' : Tendsto g atTop atTop) :
    IsBoundedUnder (· ≤ ·) l (g ∘ f) ↔ IsBoundedUnder (· ≤ ·) l f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : Nonempty β
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : NoMaxOrder γ
    g : β → γ
    f : α → β
    l : Filter α
    hg : Monotone g
    hg' : Filter.Tendsto g Filter.atTop Filter.atTop
    ⊢ Iff (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp g f)) …
  -/
  refine ⟨?_, fun h => h.isBoundedUnder (α := β) hg⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : Nonempty β
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : NoMaxOrder γ
    g : β → γ
    f : α → β
    l : Filter α
    hg : Monotone g
    hg' : Filter.Tendsto g Filter.atTop Filter.atTop
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp g f) → Fil …
  -/
  rintro ⟨c, hc⟩; rw [eventually_map] at hc
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : Nonempty β
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : NoMaxOrder γ
    g : β → γ
    f : α → β
    l : Filter α
    hg : Monotone g
    hg' : Filter.Tendsto g Filter.atTop Filter.atTop
    c : γ
    hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp g f …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l f
  -/
  obtain ⟨b, hb⟩ : ∃ b, ∀ a ≥ b, c < g a := eventually_atTop.1 (hg'.eventually_gt_atTop c)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : Nonempty β
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : NoMaxOrder γ
    g : β → γ
    f : α → β
    l : Filter α
    hg : Monotone g
    hg' : Filter.Tendsto g Filter.atTop Filter.atTop
    c : γ
    hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Function.comp g f …
    b : β
    hb : ∀ (a : β), GE.ge a b → LT.lt c (g a)
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l f
  -/
  exact ⟨b, hc.mono fun x hx => not_lt.1 fun h => (hb _ h.le).not_le hx⟩
  /-
    🎉 no goals
  -/


theorem Monotone.isBoundedUnder_ge_comp_iff [Nonempty β] [LinearOrder β] [Preorder γ] [NoMinOrder γ]
    {g : β → γ} {f : α → β} {l : Filter α} (hg : Monotone g) (hg' : Tendsto g atBot atBot) :
    IsBoundedUnder (· ≥ ·) l (g ∘ f) ↔ IsBoundedUnder (· ≥ ·) l f :=
  hg.dual.isBoundedUnder_le_comp_iff hg'


theorem Antitone.isBoundedUnder_le_comp_iff [Nonempty β] [LinearOrder β] [Preorder γ] [NoMaxOrder γ]
    {g : β → γ} {f : α → β} {l : Filter α} (hg : Antitone g) (hg' : Tendsto g atBot atTop) :
    IsBoundedUnder (· ≤ ·) l (g ∘ f) ↔ IsBoundedUnder (· ≥ ·) l f :=
  hg.dual_right.isBoundedUnder_ge_comp_iff hg'


theorem Antitone.isBoundedUnder_ge_comp_iff [Nonempty β] [LinearOrder β] [Preorder γ] [NoMinOrder γ]
    {g : β → γ} {f : α → β} {l : Filter α} (hg : Antitone g) (hg' : Tendsto g atTop atBot) :
    IsBoundedUnder (· ≥ ·) l (g ∘ f) ↔ IsBoundedUnder (· ≤ ·) l f :=
  hg.dual_right.isBoundedUnder_le_comp_iff hg'


theorem GaloisConnection.l_limsup_le [ConditionallyCompleteLattice β]
    [ConditionallyCompleteLattice γ] {f : Filter α} {v : α → β} {l : β → γ} {u : γ → β}
    (gc : GaloisConnection l u)
    (hlv : f.IsBoundedUnder (· ≤ ·) fun x => l (v x) := by isBoundedDefault)
    (hv_co : f.IsCoboundedUnder (· ≤ ·) v := by isBoundedDefault) :
    l (limsup v f) ≤ limsup (fun x => l (v x)) f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    v : α → β
    l : β → γ
    u : γ → β
    gc : GaloisConnection l u
    hlv : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => l …
    hv_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _au …
    ⊢ LE.le (l (Filter.limsup v f)) (Filter.limsup (fun x => l (v x)) f)
  -/
  refine le_limsSup_of_le hlv fun c hc => ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    v : α → β
    l : β → γ
    u : γ → β
    gc : GaloisConnection l u
    hlv : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => l …
    hv_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _au …
    c : γ
    hc : Filter.Eventually (fun n => LE.le n c) (Filter.map (fun x => l (v x)) f)
    ⊢ LE.le (l (Filter.limsup v f)) c
  -/
  rw [Filter.eventually_map] at hc
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    v : α → β
    l : β → γ
    u : γ → β
    gc : GaloisConnection l u
    hlv : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => l …
    hv_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _au …
    c : γ
    hc : Filter.Eventually (fun a => LE.le (l (v a)) c) f
    ⊢ LE.le (l (Filter.limsup v f)) c
  -/
  simp_rw [gc _ _] at hc ⊢
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    v : α → β
    l : β → γ
    u : γ → β
    gc : GaloisConnection l u
    hlv : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => l …
    hv_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _au …
    c : γ
    hc : Filter.Eventually (fun a => LE.le (v a) (u c)) f
    ⊢ LE.le (Filter.limsup v f) (u c)
  -/
  exact limsSup_le_of_le hv_co hc
  /-
    🎉 no goals
  -/


theorem OrderIso.limsup_apply {γ} [ConditionallyCompleteLattice β] [ConditionallyCompleteLattice γ]
    {f : Filter α} {u : α → β} (g : β ≃o γ)
    (hu : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (hu_co : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (hgu : f.IsBoundedUnder (· ≤ ·) fun x => g (u x) := by isBoundedDefault)
    (hgu_co : f.IsCoboundedUnder (· ≤ ·) fun x => g (u x) := by isBoundedDefault) :
    g (limsup u f) = limsup (fun x => g (u x)) f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    ⊢ Eq (g (Filter.limsup u f)) (Filter.limsup (fun x => g (u x)) f)
  -/
  refine le_antisymm ((OrderIso.to_galoisConnection g).l_limsup_le hgu hu_co) ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    ⊢ LE.le (Filter.limsup (fun x => g (u x)) f) (g (Filter.limsup u f))
  -/
  rw [← g.symm.symm_apply_apply <| limsup (fun x => g (u x)) f, g.symm_symm]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    ⊢ LE.le (g (g.symm (Filter.limsup (fun x => g (u x)) f))) (g (Filter.limsup u  …
  -/
  refine g.monotone ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    ⊢ LE.le (g.symm (Filter.limsup (fun x => g (u x)) f)) (Filter.limsup u f)
  -/
  have hf : u = fun i => g.symm (g (u i)) := funext fun i => (g.symm_apply_apply (u i)).symm
  -- Porting note: nth_rw 1 to nth_rw 2
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    hf : Eq u fun i => g.symm (g (u i))
    ⊢ LE.le (g.symm (Filter.limsup (fun x => g (u x)) f)) (Filter.limsup u f)
  -/
  nth_rw 2 [hf]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    hf : Eq u fun i => g.symm (g (u i))
    ⊢ LE.le (g.symm (Filter.limsup (fun x => g (u x)) f)) (Filter.limsup (fun i => …
  -/
  refine (OrderIso.to_galoisConnection g.symm).l_limsup_le ?_ hgu_co
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    hf : Eq u fun i => g.symm (g (u i))
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g.symm (g (u x))
  -/
  simp_rw [g.symm_apply_apply]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_6
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : ConditionallyCompleteLattice γ
    f : Filter α
    u : α → β
    g : OrderIso β γ
    hu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    hu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _au …
    hgu : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => g …
    hgu_co : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x …
    hf : Eq u fun i => g.symm (g (u i))
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => u x
  -/
  exact hu
  /-
    🎉 no goals
  -/


theorem OrderIso.liminf_apply {γ} [ConditionallyCompleteLattice β] [ConditionallyCompleteLattice γ]
    {f : Filter α} {u : α → β} (g : β ≃o γ)
    (hu : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault)
    (hu_co : f.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault)
    (hgu : f.IsBoundedUnder (· ≥ ·) fun x => g (u x) := by isBoundedDefault)
    (hgu_co : f.IsCoboundedUnder (· ≥ ·) fun x => g (u x) := by isBoundedDefault) :
    g (liminf u f) = liminf (fun x => g (u x)) f :=
  OrderIso.limsup_apply (β := βᵒᵈ) (γ := γᵒᵈ) g.dual hu hu_co hgu hgu_co


theorem isCoboundedUnder_le_max [LinearOrder β] {f : Filter α} {u v : α → β}
    (h : f.IsCoboundedUnder (· ≤ ·) u ∨ f.IsCoboundedUnder (· ≤ ·) v) :
    f.IsCoboundedUnder (· ≤ ·) (fun a ↦ max (u a) (v a)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrder β
    f : Filter α
    u v : α → β
    h : Or (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) (Filter.IsCobo …
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max (u a)  …
  -/
  rcases h with (h' | h') <;>
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      h' : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u
      ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max (u a)  …
    -/
    /-
      case inl.intro
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max (u a)  …
    -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      ⊢ ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (Filt …
    -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map ( …
      ⊢ LE.le b c
    -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map ( …
      ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map u f)
    -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Max.max (u a) (v  …
      ⊢ Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (u a) c) f
    -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Max.max (u a) (v  …
      x✝ : α
      ⊢ (fun x1 x2 => LE.le x1 x2) (Max.max (u x✝) (v x✝)) c → (fun x1 x2 => LE.le x …
    -/
    /-
      🎉 no goals
    -/
    apply hb c
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map ( …
      ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map v f)
    -/
    rw [eventually_map] at hc ⊢
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Max.max (u a) (v  …
      ⊢ Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (v a) c) f
    -/
    refine hc.mono (fun _ ↦ ?_)
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : Filter α
      u v : α → β
      b : β
      hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
      c : β
      hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (Max.max (u a) (v  …
      x✝ : α
      ⊢ (fun x1 x2 => LE.le x1 x2) (Max.max (u x✝) (v x✝)) c → (fun x1 x2 => LE.le x …
    -/
    simp +contextual only [implies_true, max_le_iff, and_imp]
    /-
      🎉 no goals
    -/


theorem limsup_max [ConditionallyCompleteLinearOrder β] {f : Filter α} {u v : α → β}
    (h₁ : f.IsCoboundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h₂ : f.IsCoboundedUnder (· ≤ ·) v := by isBoundedDefault)
    (h₃ : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h₄ : f.IsBoundedUnder (· ≤ ·) v := by isBoundedDefault) :
    limsup (fun a ↦ max (u a) (v a)) f = max (limsup u f) (limsup v f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u v : α → β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    ⊢ Eq (Filter.limsup (fun a => Max.max (u a) (v a)) f) (Max.max (Filter.limsup  …
  -/
  have bddmax := IsBoundedUnder.sup h₃ h₄
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u v : α → β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    bddmax : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max ( …
    ⊢ Eq (Filter.limsup (fun a => Max.max (u a) (v a)) f) (Max.max (Filter.limsup  …
  -/
  have cobddmax := isCoboundedUnder_le_max (v := v) (Or.inl h₁)
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    u v : α → β
    h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    bddmax : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max ( …
    cobddmax : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.m …
    ⊢ Eq (Filter.limsup (fun a => Max.max (u a) (v a)) f) (Max.max (Filter.limsup  …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u v : α → β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      bddmax : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max ( …
      cobddmax : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.m …
      ⊢ LE.le (Filter.limsup (fun a => Max.max (u a) (v a)) f) (Max.max (Filter.lims …
    -/
  · refine (limsup_le_iff cobddmax bddmax).2 (fun b hb ↦ ?_)
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u v : α → β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      bddmax : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max ( …
      cobddmax : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.m …
      b : β
      hb : GT.gt b (Max.max (Filter.limsup u f) (Filter.limsup v f))
      ⊢ Filter.Eventually (fun a => LT.lt (Max.max (u a) (v a)) b) f
    -/
    have hu := eventually_lt_of_limsup_lt (lt_of_le_of_lt (le_max_left _ _) hb) h₃
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u v : α → β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      bddmax : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max ( …
      cobddmax : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.m …
      b : β
      hb : GT.gt b (Max.max (Filter.limsup u f) (Filter.limsup v f))
      hu : Filter.Eventually (fun a => LT.lt (u a) b) f
      ⊢ Filter.Eventually (fun a => LT.lt (Max.max (u a) (v a)) b) f
    -/
    have hv := eventually_lt_of_limsup_lt (lt_of_le_of_lt (le_max_right _ _) hb) h₄
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      u v : α → β
      h₁ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
      bddmax : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.max ( …
      cobddmax : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => Max.m …
      b : β
      hb : GT.gt b (Max.max (Filter.limsup u f) (Filter.limsup v f))
      hu : Filter.Eventually (fun a => LT.lt (u a) b) f
      hv : Filter.Eventually (fun a => LT.lt (v a) b) f
      ⊢ Filter.Eventually (fun a => LT.lt (Max.max (u a) (v a)) b) f
    -/
    refine mem_of_superset (inter_mem hu hv) (fun _ ↦ by simp)
    /-
      🎉 no goals
    -/
  · exact max_le (c := limsup (fun a ↦ max (u a) (v a)) f)
      (limsup_le_limsup (Eventually.of_forall (fun a : α ↦ le_max_left (u a) (v a))) h₁ bddmax)
      (limsup_le_limsup (Eventually.of_forall (fun a : α ↦ le_max_right (u a) (v a))) h₂ bddmax)


theorem liminf_min [ConditionallyCompleteLinearOrder β] {f : Filter α} {u v : α → β}
    (h₁ : f.IsCoboundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h₂ : f.IsCoboundedUnder (· ≥ ·) v := by isBoundedDefault)
    (h₃ : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault)
    (h₄ : f.IsBoundedUnder (· ≥ ·) v := by isBoundedDefault) :
    liminf (fun a ↦ min (u a) (v a)) f = min (liminf u f) (liminf v f) :=
  limsup_max (β := βᵒᵈ) h₁ h₂ h₃ h₄


theorem isBoundedUnder_le_finset_sup' [LinearOrder β] [Nonempty β] {f : Filter α} {F : ι → α → β}
    {s : Finset ι} (hs : s.Nonempty) (h : ∀ i ∈ s, f.IsBoundedUnder (· ≤ ·) (F i)) :
    f.IsBoundedUnder (· ≤ ·) (fun a ↦ sup' s hs (fun i ↦ F i a)) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    h : ∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1 x2 => LE.le  …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs fun i  …
  -/
  choose! m hm using h
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    m : ι → β
    hm : ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun x => (fun x1 x2 => …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs fun i  …
  -/
  use sup' s hs m
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    m : ι → β
    hm : ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun x => (fun x1 x2 => …
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (s.sup' hs m)) (Fil …
  -/
  simp only [eventually_map] at hm ⊢
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    m : ι → β
    hm : ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun a => LE.le (F i a) …
    ⊢ Filter.Eventually (fun a => LE.le (s.sup' hs fun i => F i a) (s.sup' hs m)) f
  -/
  rw [← eventually_all_finset s] at hm
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    m : ι → β
    hm : Filter.Eventually (fun x => ∀ (i : ι), Membership.mem s i → LE.le (F i x) …
    ⊢ Filter.Eventually (fun a => LE.le (s.sup' hs fun i => F i a) (s.sup' hs m)) f
  -/
  refine hm.mono fun a h ↦ ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    m : ι → β
    hm : Filter.Eventually (fun x => ∀ (i : ι), Membership.mem s i → LE.le (F i x) …
    a : α
    h : ∀ (i : ι), Membership.mem s i → LE.le (F i a) (m i)
    ⊢ LE.le (s.sup' hs fun i => F i a) (s.sup' hs m)
  -/
  simp only [Finset.sup'_apply, sup'_le_iff]
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : Nonempty β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    m : ι → β
    hm : Filter.Eventually (fun x => ∀ (i : ι), Membership.mem s i → LE.le (F i x) …
    a : α
    h : ∀ (i : ι), Membership.mem s i → LE.le (F i a) (m i)
    ⊢ ∀ (b : ι), Membership.mem s b → LE.le (F b a) (s.sup' hs m)
  -/
  exact fun i i_s ↦ le_trans (h i i_s) (le_sup' m i_s)
  /-
    🎉 no goals
  -/


theorem isCoboundedUnder_le_finset_sup' [LinearOrder β] {f : Filter α} {F : ι → α → β}
    {s : Finset ι} (hs : s.Nonempty) (h : ∃ i ∈ s, f.IsCoboundedUnder (· ≤ ·) (F i)) :
    f.IsCoboundedUnder (· ≤ ·) (fun a ↦ sup' s hs (fun i ↦ F i a)) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    h : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1  …
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs fun  …
  -/
  rcases h with ⟨i, i_s, b, hb⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    i : ι
    i_s : Membership.mem s i
    b : β
    hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs fun  …
  -/
  use b
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    i : ι
    i_s : Membership.mem s i
    b : β
    hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
    ⊢ ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (Filt …
  -/
  refine fun c hc ↦ hb c ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    i : ι
    i_s : Membership.mem s i
    b : β
    hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
    c : β
    hc : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map ( …
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map (F i …
  -/
  rw [eventually_map] at hc ⊢
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    i : ι
    i_s : Membership.mem s i
    b : β
    hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
    c : β
    hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (s.sup' hs fun i = …
    ⊢ Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (F i a) c) f
  -/
  refine hc.mono fun a h ↦ ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    i : ι
    i_s : Membership.mem s i
    b : β
    hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
    c : β
    hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (s.sup' hs fun i = …
    a : α
    h : (fun x1 x2 => LE.le x1 x2) (s.sup' hs fun i => F i a) c
    ⊢ (fun x1 x2 => LE.le x1 x2) (F i a) c
  -/
  simp only [Finset.sup'_apply, sup'_le_iff] at h ⊢
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : LinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    i : ι
    i_s : Membership.mem s i
    b : β
    hb : ∀ (a : β), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (F …
    c : β
    hc : Filter.Eventually (fun a => (fun x1 x2 => LE.le x1 x2) (s.sup' hs fun i = …
    a : α
    h : ∀ (b : ι), Membership.mem s b → LE.le (F b a) c
    ⊢ LE.le (F i a) c
  -/
  exact h i i_s
  /-
    🎉 no goals
  -/


theorem isBoundedUnder_le_finset_sup [LinearOrder β] [OrderBot β] {f : Filter α} {F : ι → α → β}
    {s : Finset ι} (h : ∀ i ∈ s, f.IsBoundedUnder (· ≤ ·) (F i)) :
    f.IsBoundedUnder (· ≤ ·) (fun a ↦ sup s (fun i ↦ F i a)) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h : ∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1 x2 => LE.le  …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup fun i => F …
  -/
  choose! m hm using h
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    m : ι → β
    hm : ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun x => (fun x1 x2 => …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup fun i => F …
  -/
  use sup s m
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    m : ι → β
    hm : ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun x => (fun x1 x2 => …
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (s.sup m)) (Filter. …
  -/
  simp only [eventually_map] at hm ⊢
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    m : ι → β
    hm : ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun a => LE.le (F i a) …
    ⊢ Filter.Eventually (fun a => LE.le (s.sup fun i => F i a) (s.sup m)) f
  -/
  rw [← eventually_all_finset s] at hm
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : LinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    m : ι → β
    hm : Filter.Eventually (fun x => ∀ (i : ι), Membership.mem s i → LE.le (F i x) …
    ⊢ Filter.Eventually (fun a => LE.le (s.sup fun i => F i a) (s.sup m)) f
  -/
  exact hm.mono fun _ h ↦ sup_mono_fun h
  /-
    🎉 no goals
  -/


theorem isBoundedUnder_ge_finset_inf' [LinearOrder β] [Nonempty β] {f : Filter α} {F : ι → α → β}
    {s : Finset ι} (hs : s.Nonempty) (h : ∀ i ∈ s, f.IsBoundedUnder (· ≥ ·) (F i)) :
    f.IsBoundedUnder (· ≥ ·) (fun a ↦ inf' s hs (fun i ↦ F i a)) :=
  isBoundedUnder_le_finset_sup' (β := βᵒᵈ) hs h


theorem isCoboundedUnder_ge_finset_inf' [LinearOrder β] {f : Filter α} {F : ι → α → β}
    {s : Finset ι} (hs : s.Nonempty) (h : ∃ i ∈ s, f.IsCoboundedUnder (· ≥ ·) (F i)) :
    f.IsCoboundedUnder (· ≥ ·) (fun a ↦ inf' s hs (fun i ↦ F i a)) :=
  isCoboundedUnder_le_finset_sup' (β := βᵒᵈ) hs h


theorem isBoundedUnder_ge_finset_inf [LinearOrder β] [OrderTop β] {f : Filter α} {F : ι → α → β}
    {s : Finset ι} (h : ∀ i ∈ s, f.IsBoundedUnder (· ≥ ·) (F i)) :
    f.IsBoundedUnder (· ≥ ·) (fun a ↦ inf s (fun i ↦ F i a)) :=
  isBoundedUnder_le_finset_sup (β := βᵒᵈ) h


theorem limsup_finset_sup' [ConditionallyCompleteLinearOrder β] {f : Filter α}
    {F : ι → α → β} {s : Finset ι} (hs : s.Nonempty)
    (h₁ : ∀ i ∈ s, f.IsCoboundedUnder (· ≤ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault)
    (h₂ : ∀ i ∈ s, f.IsBoundedUnder (· ≤ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault) :
    limsup (fun a ↦ sup' s hs (fun i ↦ F i a)) f = sup' s hs (fun i ↦ limsup (F i) f) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    ⊢ Eq (Filter.limsup (fun a => s.sup' hs fun i => F i a) f) (s.sup' hs fun i => …
  -/
  have bddsup := isBoundedUnder_le_finset_sup' hs h₂
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝ : ConditionallyCompleteLinearOrder β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    hs : s.Nonempty
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
    ⊢ Eq (Filter.limsup (fun a => s.sup' hs fun i => F i a) f) (s.sup' hs fun i => …
  -/
  apply le_antisymm
  · have h₃ : ∃ i ∈ s, f.IsCoboundedUnder (· ≤ ·) (F i) := by
      rcases hs with ⟨i, i_s⟩
      use i, i_s
      exact h₁ i i_s
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
      ⊢ LE.le (Filter.limsup (fun a => s.sup' hs fun i => F i a) f) (s.sup' hs fun i …
    -/
    have cobddsup := isCoboundedUnder_le_finset_sup' hs h₃
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
      cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
      ⊢ LE.le (Filter.limsup (fun a => s.sup' hs fun i => F i a) f) (s.sup' hs fun i …
    -/
    refine (limsup_le_iff cobddsup bddsup).2 (fun b hb ↦ ?_)
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
      cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
      b : β
      hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
      ⊢ Filter.Eventually (fun a => LT.lt (s.sup' hs fun i => F i a) b) f
    -/
    rw [eventually_iff_exists_mem]
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
      cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
      b : β
      hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
      ⊢ Exists fun v => And (Membership.mem f v) (∀ (y : α), Membership.mem v y → LT …
    -/
    use ⋂ i ∈ s, {a | F i a < b}
    /-
      case h
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
      cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
      b : β
      hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
      ⊢ And (Membership.mem f (Set.iInter fun i => Set.iInter fun h => setOf fun a = …
    -/
    split_ands
      /-
        case h.refine_1
        α : Type u_1
        β : Type u_2
        ι : Type u_4
        inst✝ : ConditionallyCompleteLinearOrder β
        f : Filter α
        F : ι → α → β
        s : Finset ι
        hs : s.Nonempty
        h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
        h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
        bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
        h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
        cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
        b : β
        hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
        ⊢ Membership.mem f (Set.iInter fun i => Set.iInter fun h => setOf fun a => LT. …
      -/
    · rw [biInter_finset_mem]
      /-
        case h.refine_1
        α : Type u_1
        β : Type u_2
        ι : Type u_4
        inst✝ : ConditionallyCompleteLinearOrder β
        f : Filter α
        F : ι → α → β
        s : Finset ι
        hs : s.Nonempty
        h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
        h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
        bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
        h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
        cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
        b : β
        hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
        ⊢ ∀ (i : ι), Membership.mem s i → Membership.mem f (setOf fun a => LT.lt (F i  …
      -/
      suffices key : ∀ i ∈ s, ∀ᶠ a in f, F i a < b from fun i i_s ↦ eventually_iff.1 (key i i_s)
      /-
        case h.refine_1
        α : Type u_1
        β : Type u_2
        ι : Type u_4
        inst✝ : ConditionallyCompleteLinearOrder β
        f : Filter α
        F : ι → α → β
        s : Finset ι
        hs : s.Nonempty
        h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
        h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
        bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
        h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
        cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
        b : β
        hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
        ⊢ ∀ (i : ι), Membership.mem s i → Filter.Eventually (fun a => LT.lt (F i a) b) f
      -/
      intro i i_s
      /-
        case h.refine_1
        α : Type u_1
        β : Type u_2
        ι : Type u_4
        inst✝ : ConditionallyCompleteLinearOrder β
        f : Filter α
        F : ι → α → β
        s : Finset ι
        hs : s.Nonempty
        h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
        h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
        bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
        h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
        cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
        b : β
        hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
        i : ι
        i_s : Membership.mem s i
        ⊢ Filter.Eventually (fun a => LT.lt (F i a) b) f
      -/
      apply eventually_lt_of_limsup_lt _ (h₂ i i_s)
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_4
        inst✝ : ConditionallyCompleteLinearOrder β
        f : Filter α
        F : ι → α → β
        s : Finset ι
        hs : s.Nonempty
        h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
        h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
        bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
        h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
        cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
        b : β
        hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
        i : ι
        i_s : Membership.mem s i
        ⊢ LT.lt (Filter.limsup (F i) f) b
      -/
      exact lt_of_le_of_lt (Finset.le_sup' (f := fun i ↦ limsup (F i) f) i_s) hb
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        α : Type u_1
        β : Type u_2
        ι : Type u_4
        inst✝ : ConditionallyCompleteLinearOrder β
        f : Filter α
        F : ι → α → β
        s : Finset ι
        hs : s.Nonempty
        h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
        h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
        bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
        h₃ : Exists fun i => And (Membership.mem s i) (Filter.IsCoboundedUnder (fun x1 …
        cobddsup : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup …
        b : β
        hb : GT.gt b (s.sup' hs fun i => Filter.limsup (F i) f)
        ⊢ ∀ (y : α), Membership.mem (Set.iInter fun i => Set.iInter fun h => setOf fun …
      -/
    · simp only [mem_iInter, mem_setOf_eq, Finset.sup'_apply, sup'_lt_iff, imp_self, implies_true]
      /-
        🎉 no goals
      -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      ⊢ LE.le (s.sup' hs fun i => Filter.limsup (F i) f) (Filter.limsup (fun a => s. …
    -/
  · apply Finset.sup'_le hs (fun i ↦ limsup (F i) f)
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      ⊢ ∀ (b : ι), Membership.mem s b → LE.le (Filter.limsup (F b) f) (Filter.limsup …
    -/
    refine fun i i_s ↦ limsup_le_limsup (Eventually.of_forall (fun a ↦ ?_)) (h₁ i i_s) bddsup
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      i : ι
      i_s : Membership.mem s i
      a : α
      ⊢ LE.le (F i a) ((fun a => s.sup' hs fun i => F i a) a)
    -/
    simp only [Finset.sup'_apply, le_sup'_iff]
    /-
      case a
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝ : ConditionallyCompleteLinearOrder β
      f : Filter α
      F : ι → α → β
      s : Finset ι
      hs : s.Nonempty
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      bddsup : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f fun a => s.sup' hs …
      i : ι
      i_s : Membership.mem s i
      a : α
      ⊢ Exists fun b => And (Membership.mem s b) (LE.le (F i a) (F b a))
    -/
    use i, i_s
    /-
      🎉 no goals
    -/


theorem limsup_finset_sup [ConditionallyCompleteLinearOrder β] [OrderBot β] {f : Filter α}
    {F : ι → α → β} {s : Finset ι}
    (h₁ : ∀ i ∈ s, f.IsCoboundedUnder (· ≤ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault)
    (h₂ : ∀ i ∈ s, f.IsBoundedUnder (· ≤ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault) :
    limsup (fun a ↦ sup s (fun i ↦ F i a)) f = sup s (fun i ↦ limsup (F i) f) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    ⊢ Eq (Filter.limsup (fun a => s.sup fun i => F i a) f) (s.sup fun i => Filter. …
  -/
  rcases eq_or_neBot f with (rfl | _)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝¹ : ConditionallyCompleteLinearOrder β
      inst✝ : OrderBot β
      F : ι → α → β
      s : Finset ι
      h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
      h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
      ⊢ Eq (Filter.limsup (fun a => s.sup fun i => F i a) Bot.bot) (s.sup fun i => F …
    -/
  · simp [limsup_eq, csInf_univ]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    h✝ : f.NeBot
    ⊢ Eq (Filter.limsup (fun a => s.sup fun i => F i a) f) (s.sup fun i => Filter. …
  -/
  rcases Finset.eq_empty_or_nonempty s with (rfl | s_nemp)
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      ι : Type u_4
      inst✝¹ : ConditionallyCompleteLinearOrder β
      inst✝ : OrderBot β
      f : Filter α
      F : ι → α → β
      h✝ : f.NeBot
      h₁ : autoParam (∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i →  …
      h₂ : autoParam (∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i →  …
      ⊢ Eq (Filter.limsup (fun a => EmptyCollection.emptyCollection.sup fun i => F i …
    -/
  · simp only [Finset.sup_apply, sup_empty, limsup_const]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    h✝ : f.NeBot
    s_nemp : s.Nonempty
    ⊢ Eq (Filter.limsup (fun a => s.sup fun i => F i a) f) (s.sup fun i => Filter. …
  -/
  rw [← Finset.sup'_eq_sup s_nemp fun i ↦ limsup (F i) f, ← limsup_finset_sup' s_nemp h₁ h₂]
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    h✝ : f.NeBot
    s_nemp : s.Nonempty
    ⊢ Eq (Filter.limsup (fun a => s.sup fun i => F i a) f) (Filter.limsup (fun a = …
  -/
  congr
  /-
    case inr.inr.e_u
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    h✝ : f.NeBot
    s_nemp : s.Nonempty
    ⊢ Eq (fun a => s.sup fun i => F i a) fun a => s.sup' s_nemp fun i => F i a
  -/
  ext a
  /-
    case inr.inr.e_u.h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderBot β
    f : Filter α
    F : ι → α → β
    s : Finset ι
    h₁ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsCoboundedUnder (fun x …
    h₂ : autoParam (∀ (i : ι), Membership.mem s i → Filter.IsBoundedUnder (fun x1  …
    h✝ : f.NeBot
    s_nemp : s.Nonempty
    a : α
    ⊢ Eq (s.sup fun i => F i a) (s.sup' s_nemp fun i => F i a)
  -/
  exact Eq.symm (Finset.sup'_eq_sup s_nemp (fun i ↦ F i a))
  /-
    🎉 no goals
  -/


theorem liminf_finset_inf' [ConditionallyCompleteLinearOrder β] {f : Filter α}
    {F : ι → α → β} {s : Finset ι} (hs : s.Nonempty)
    (h₁ : ∀ i ∈ s, f.IsCoboundedUnder (· ≥ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault)
    (h₂ : ∀ i ∈ s, f.IsBoundedUnder (· ≥ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault) :
    liminf (fun a ↦ inf' s hs (fun i ↦ F i a)) f = inf' s hs (fun i ↦ liminf (F i) f) :=
  limsup_finset_sup' (β := βᵒᵈ) hs h₁ h₂


theorem liminf_finset_inf [ConditionallyCompleteLinearOrder β] [OrderTop β] {f : Filter α}
    {F : ι → α → β} {s : Finset ι}
    (h₁ : ∀ i ∈ s, f.IsCoboundedUnder (· ≥ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault)
    (h₂ : ∀ i ∈ s, f.IsBoundedUnder (· ≥ ·) (F i) := by exact fun _ _ ↦ by isBoundedDefault) :
    liminf (fun a ↦ inf s (fun i ↦ F i a)) f = inf s (fun i ↦ liminf (F i) f) :=
  limsup_finset_sup (β := βᵒᵈ) h₁ h₂


lemma Monotone.frequently_ge_map_of_frequently_ge {f : R → S} (f_incr : Monotone f)
    {l : R} (freq_ge : ∃ᶠ x in F, l ≤ x) :
    ∃ᶠ x' in F.map f, f l ≤ x' := by
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    l : R
    freq_ge : Filter.Frequently (fun x => LE.le l x) F
    ⊢ Filter.Frequently (fun x' => LE.le (f l) x') (Filter.map f F)
  -/
  refine fun ev ↦ freq_ge ?_
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    l : R
    freq_ge : Filter.Frequently (fun x => LE.le l x) F
    ev : Filter.Eventually (fun x => Not ((fun x' => LE.le (f l) x') x)) (Filter.m …
    ⊢ Filter.Eventually (fun x => Not ((fun x => LE.le l x) x)) F
  -/
  simp only [not_le, not_lt] at ev freq_ge ⊢
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    l : R
    freq_ge : Filter.Frequently (fun x => LE.le l x) F
    ev : Filter.Eventually (fun x => LT.lt x (f l)) (Filter.map f F)
    ⊢ Filter.Eventually (fun x => LT.lt x l) F
  -/
  filter_upwards [ev] with z hz
  /-
    case h
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    l : R
    freq_ge : Filter.Frequently (fun x => LE.le l x) F
    ev : Filter.Eventually (fun x => LT.lt x (f l)) (Filter.map f F)
    z : R
    hz : Membership.mem (Set.preimage f (setOf fun x => LT.lt x (f l))) z
    ⊢ LT.lt z l
  -/
  by_contra con
  /-
    case h
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    l : R
    freq_ge : Filter.Frequently (fun x => LE.le l x) F
    ev : Filter.Eventually (fun x => LT.lt x (f l)) (Filter.map f F)
    z : R
    hz : Membership.mem (Set.preimage f (setOf fun x => LT.lt x (f l))) z
    con : Not (LT.lt z l)
    ⊢ False
  -/
  exact lt_irrefl (f l) <| lt_of_le_of_lt (f_incr <| not_lt.mp con) hz
  /-
    🎉 no goals
  -/


lemma Monotone.frequently_le_map_of_frequently_le {f : R → S} (f_incr : Monotone f)
    {u : R} (freq_le : ∃ᶠ x in F, x ≤ u) :
    ∃ᶠ y in F.map f, y ≤ f u := by
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    u : R
    freq_le : Filter.Frequently (fun x => LE.le x u) F
    ⊢ Filter.Frequently (fun y => LE.le y (f u)) (Filter.map f F)
  -/
  refine fun ev ↦ freq_le ?_
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    u : R
    freq_le : Filter.Frequently (fun x => LE.le x u) F
    ev : Filter.Eventually (fun x => Not ((fun y => LE.le y (f u)) x)) (Filter.map …
    ⊢ Filter.Eventually (fun x => Not ((fun x => LE.le x u) x)) F
  -/
  simp only [not_le, not_lt] at ev freq_le ⊢
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    u : R
    freq_le : Filter.Frequently (fun x => LE.le x u) F
    ev : Filter.Eventually (fun x => LT.lt (f u) x) (Filter.map f F)
    ⊢ Filter.Eventually (fun x => LT.lt u x) F
  -/
  filter_upwards [ev] with z hz
  /-
    case h
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    u : R
    freq_le : Filter.Frequently (fun x => LE.le x u) F
    ev : Filter.Eventually (fun x => LT.lt (f u) x) (Filter.map f F)
    z : R
    hz : Membership.mem (Set.preimage f (setOf fun x => LT.lt (f u) x)) z
    ⊢ LT.lt u z
  -/
  by_contra con
  /-
    case h
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝¹ : LinearOrder R
    inst✝ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    u : R
    freq_le : Filter.Frequently (fun x => LE.le x u) F
    ev : Filter.Eventually (fun x => LT.lt (f u) x) (Filter.map f F)
    z : R
    hz : Membership.mem (Set.preimage f (setOf fun x => LT.lt (f u) x)) z
    con : Not (LT.lt u z)
    ⊢ False
  -/
  apply lt_irrefl (f u) <| lt_of_lt_of_le hz <| f_incr (not_lt.mp con)
  /-
    🎉 no goals
  -/


lemma Antitone.frequently_le_map_of_frequently_ge {f : R → S} (f_decr : Antitone f)
    {l : R} (frbdd : ∃ᶠ x in F, l ≤ x) :
    ∃ᶠ y in F.map f, y ≤ f l :=
  Monotone.frequently_ge_map_of_frequently_ge (S := Sᵒᵈ) f_decr frbdd


lemma Antitone.frequently_ge_map_of_frequently_le {f : R → S} (f_decr : Antitone f)
    {u : R} (frbdd : ∃ᶠ x in F, x ≤ u) :
    ∃ᶠ y in F.map f, f u ≤ y :=
  Monotone.frequently_le_map_of_frequently_le (S := Sᵒᵈ) f_decr frbdd


lemma Monotone.isCoboundedUnder_le_of_isCobounded {f : R → S} (f_incr : Monotone f)
    [NeBot F] (cobdd : IsCobounded (· ≤ ·) F) :
    F.IsCoboundedUnder (· ≤ ·) f := by
  /-
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝² : LinearOrder R
    inst✝¹ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    inst✝ : F.NeBot
    cobdd : Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) F f
  -/
  obtain ⟨l, hl⟩ := IsCobounded.frequently_ge cobdd
  /-
    case intro
    R : Type u_6
    S : Type u_7
    F : Filter R
    inst✝² : LinearOrder R
    inst✝¹ : LinearOrder S
    f : R → S
    f_incr : Monotone f
    inst✝ : F.NeBot
    cobdd : Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F
    l : R
    hl : Filter.Frequently (fun x => LE.le l x) F
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) F f
  -/
  exact IsCobounded.of_frequently_ge <| f_incr.frequently_ge_map_of_frequently_ge hl
  /-
    🎉 no goals
  -/


lemma Monotone.isCoboundedUnder_ge_of_isCobounded {f : R → S} (f_incr : Monotone f)
    [NeBot F] (cobdd : IsCobounded (· ≥ ·) F) :
    F.IsCoboundedUnder (· ≥ ·) f :=
  Monotone.isCoboundedUnder_le_of_isCobounded (R := Rᵒᵈ) (S := Sᵒᵈ) f_incr.dual cobdd


lemma Antitone.isCoboundedUnder_le_of_isCobounded {f : R → S} (f_decr : Antitone f)
    [NeBot F] (cobdd : IsCobounded (· ≥ ·) F) :
    F.IsCoboundedUnder (· ≤ ·) f :=
  Monotone.isCoboundedUnder_le_of_isCobounded (R := Rᵒᵈ) f_decr.dual cobdd


lemma Antitone.isCoboundedUnder_ge_of_isCobounded {f : R → S} (f_decr : Antitone f)
    [NeBot F] (cobdd : IsCobounded (· ≤ ·) F) :
    F.IsCoboundedUnder (· ≥ ·) f :=
  Monotone.isCoboundedUnder_le_of_isCobounded (S := Sᵒᵈ) f_decr cobdd


