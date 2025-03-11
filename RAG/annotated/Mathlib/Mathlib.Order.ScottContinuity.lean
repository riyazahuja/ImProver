/-- A function between preorders is said to be Scott continuous on a set `D` of directed sets if it
preserves `IsLUB` on elements of `D`.

The dual notion

```lean
∀ ⦃d : Set α⦄, d ∈ D →  d.Nonempty → DirectedOn (· ≥ ·) d → ∀ ⦃a⦄, IsGLB d a → IsGLB (f '' d) (f a)
```

does not appear to play a significant role in the literature, so is omitted here.
-/
def ScottContinuousOn (D : Set (Set α)) (f : α → β) : Prop :=
  ∀ ⦃d : Set α⦄, d ∈ D → d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a⦄, IsLUB d a → IsLUB (f '' d) (f a)


lemma ScottContinuousOn.mono (hD : D₁ ⊆ D₂) (hf : ScottContinuousOn D₂ f) :
    ScottContinuousOn D₁ f := fun _  hdD₁ hd₁ hd₂ _ hda => hf (hD hdD₁) hd₁ hd₂ hda


protected theorem ScottContinuousOn.monotone (D : Set (Set α)) (hD : ∀ a b : α, a ≤ b → {a, b} ∈ D)
    (h : ScottContinuousOn D f) : Monotone f := by
  refine fun a b hab =>
    (h (hD a b hab) (insert_nonempty _ _) (directedOn_pair le_refl hab) ?_).1
      (mem_image_of_mem _ <| mem_insert _ _)
  rw [IsLUB, upperBounds_insert, upperBounds_singleton,
    inter_eq_self_of_subset_right (Ici_subset_Ici.2 hab)]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    D : Set (Set α)
    hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
    h : ScottContinuousOn D f
    a b : α
    hab : LE.le a b
    ⊢ IsLeast (Set.Ici b) b
  -/
  exact isLeast_Ici
  /-
    🎉 no goals
  -/


                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : Preorder α
                                                                              D : Set (Set α)
                                                                              ⊢ ScottContinuousOn D _root_.id
                                                                            -/
@[simp] lemma ScottContinuousOn.id : ScottContinuousOn D (id : α → α) := by simp [ScottContinuousOn]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


lemma ScottContinuousOn.prodMk (hD : ∀ a b : α, a ≤ b → {a, b} ∈ D)
    (hf : ScottContinuousOn D f) (hg : ScottContinuousOn D g) :
    ScottContinuousOn D fun x => (f x, g x) := fun d hd₁ hd₂ hd₃ a hda => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    D : Set (Set α)
    f g : α → β
    hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
    hf : ScottContinuousOn D f
    hg : ScottContinuousOn D g
    d : Set α
    hd₁ : Membership.mem D d
    hd₂ : d.Nonempty
    hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    a : α
    hda : IsLUB d a
    ⊢ IsLUB (Set.image (fun x => { fst := f x, snd := g x }) d) ((fun x => { fst : …
  -/
  rw [IsLUB, IsLeast, upperBounds]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    D : Set (Set α)
    f g : α → β
    hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
    hf : ScottContinuousOn D f
    hg : ScottContinuousOn D g
    d : Set α
    hd₁ : Membership.mem D d
    hd₂ : d.Nonempty
    hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    a : α
    hda : IsLUB d a
    ⊢ And (Membership.mem (setOf fun x => ∀ ⦃a : Prod β β⦄, Membership.mem (Set.im …
  -/
  constructor
  · simp only [mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, mem_setOf_eq,
      Prod.mk_le_mk]
    /-
      case left
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      D : Set (Set α)
      f g : α → β
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : ScottContinuousOn D f
      hg : ScottContinuousOn D g
      d : Set α
      hd₁ : Membership.mem D d
      hd₂ : d.Nonempty
      hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hda : IsLUB d a
      ⊢ ∀ (a_1 : α), Membership.mem d a_1 → And (LE.le (f a_1) (f a)) (LE.le (g a_1) …
    -/
    intro b hb
    /-
      case left
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      D : Set (Set α)
      f g : α → β
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : ScottContinuousOn D f
      hg : ScottContinuousOn D g
      d : Set α
      hd₁ : Membership.mem D d
      hd₂ : d.Nonempty
      hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hda : IsLUB d a
      b : α
      hb : Membership.mem d b
      ⊢ And (LE.le (f b) (f a)) (LE.le (g b) (g a))
    -/
    exact ⟨hf.monotone D hD (hda.1 hb), hg.monotone D hD (hda.1 hb)⟩
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      D : Set (Set α)
      f g : α → β
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : ScottContinuousOn D f
      hg : ScottContinuousOn D g
      d : Set α
      hd₁ : Membership.mem D d
      hd₂ : d.Nonempty
      hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hda : IsLUB d a
      ⊢ Membership.mem (lowerBounds (setOf fun x => ∀ ⦃a : Prod β β⦄, Membership.mem …
    -/
  · intro ⟨p₁, p₂⟩ hp
    simp only [mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, mem_setOf_eq,
      Prod.mk_le_mk] at hp
    /-
      case right
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      D : Set (Set α)
      f g : α → β
      hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
      hf : ScottContinuousOn D f
      hg : ScottContinuousOn D g
      d : Set α
      hd₁ : Membership.mem D d
      hd₂ : d.Nonempty
      hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      a : α
      hda : IsLUB d a
      p₁ p₂ : β
      hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
      ⊢ LE.le ((fun x => { fst := f x, snd := g x }) a) { fst := p₁, snd := p₂ }
    -/
    constructor
      /-
        case right.left
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        ⊢ LE.le ((fun x => { fst := f x, snd := g x }) a).1 { fst := p₁, snd := p₂ }.1
      -/
    · rw [isLUB_le_iff (hf hd₁ hd₂ hd₃ hda), upperBounds]
      /-
        case right.left
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        ⊢ Membership.mem (setOf fun x => ∀ ⦃a : β⦄, Membership.mem (Set.image f d) a → …
      -/
      simp only [mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, mem_setOf_eq]
      /-
        case right.left
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        ⊢ ∀ (a : α), Membership.mem d a → LE.le (f a) p₁
      -/
      intro _ hb
      /-
        case right.left
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        a✝ : α
        hb : Membership.mem d a✝
        ⊢ LE.le (f a✝) p₁
      -/
      exact (hp _ hb).1
      /-
        🎉 no goals
      -/
      /-
        case right.right
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        ⊢ LE.le ((fun x => { fst := f x, snd := g x }) a).2 { fst := p₁, snd := p₂ }.2
      -/
    · rw [isLUB_le_iff (hg hd₁ hd₂ hd₃ hda), upperBounds]
      /-
        case right.right
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        ⊢ Membership.mem (setOf fun x => ∀ ⦃a : β⦄, Membership.mem (Set.image g d) a → …
      -/
      simp only [mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, mem_setOf_eq]
      /-
        case right.right
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        ⊢ ∀ (a : α), Membership.mem d a → LE.le (g a) p₂
      -/
      intro _ hb
      /-
        case right.right
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        D : Set (Set α)
        f g : α → β
        hD : ∀ (a b : α), LE.le a b → Membership.mem D (Insert.insert a (Singleton.sin …
        hf : ScottContinuousOn D f
        hg : ScottContinuousOn D g
        d : Set α
        hd₁ : Membership.mem D d
        hd₂ : d.Nonempty
        hd₃ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
        a : α
        hda : IsLUB d a
        p₁ p₂ : β
        hp : ∀ (a : α), Membership.mem d a → And (LE.le (f a) p₁) (LE.le (g a) p₂)
        a✝ : α
        hb : Membership.mem d a✝
        ⊢ LE.le (g a✝) p₂
      -/
      exact (hp _ hb).2
      /-
        🎉 no goals
      -/


/-- A function between preorders is said to be Scott continuous if it preserves `IsLUB` on directed
sets. It can be shown that a function is Scott continuous if and only if it is continuous wrt the
Scott topology.
-/
def ScottContinuous (f : α → β) : Prop :=
  ∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (· ≤ ·) d → ∀ ⦃a⦄, IsLUB d a → IsLUB (f '' d) (f a)


@[simp] lemma scottContinuousOn_univ : ScottContinuousOn univ f ↔ ScottContinuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (ScottContinuousOn Set.univ f) (ScottContinuous f)
  -/
  simp [ScottContinuousOn, ScottContinuous]
  /-
    🎉 no goals
  -/


lemma ScottContinuous.scottContinuousOn {D : Set (Set α)} :
    ScottContinuous f → ScottContinuousOn D f := fun h _ _ d₂ d₃ _ hda => h d₂ d₃ hda


protected theorem ScottContinuous.monotone (h : ScottContinuous f) : Monotone f :=
  h.scottContinuousOn.monotone univ (fun _ _ _ ↦ mem_univ _)


                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : Preorder α
                                                                        ⊢ ScottContinuous _root_.id
                                                                      -/
@[simp] lemma ScottContinuous.id : ScottContinuous (id : α → α) := by simp [ScottContinuous]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma ScottContinuousOn.sup₂ {D : Set (Set (β × β))} :
    ScottContinuousOn D fun (a, b) => (a ⊔ b : β) := by
  /-
    β : Type u_2
    inst✝ : SemilatticeSup β
    D : Set (Set (Prod β β))
    ⊢ ScottContinuousOn D fun x => ScottContinuousOn.sup₂.match_1 (fun x => β) x f …
  -/
  simp only
  /-
    β : Type u_2
    inst✝ : SemilatticeSup β
    D : Set (Set (Prod β β))
    ⊢ ScottContinuousOn D fun x => Max.max x.1 x.2
  -/
  intro d _ _ _ ⟨p₁, p₂⟩ hdp
  /-
    β : Type u_2
    inst✝ : SemilatticeSup β
    D : Set (Set (Prod β β))
    d : Set (Prod β β)
    a✝² : Membership.mem D d
    a✝¹ : d.Nonempty
    a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    p₁ p₂ : β
    hdp : IsLUB d { fst := p₁, snd := p₂ }
    ⊢ IsLUB (Set.image (fun x => Max.max x.1 x.2) d) ((fun x => Max.max x.1 x.2) { …
  -/
  rw [IsLUB, IsLeast, upperBounds] at hdp
  /-
    β : Type u_2
    inst✝ : SemilatticeSup β
    D : Set (Set (Prod β β))
    d : Set (Prod β β)
    a✝² : Membership.mem D d
    a✝¹ : d.Nonempty
    a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    p₁ p₂ : β
    hdp : And (Membership.mem (setOf fun x => ∀ ⦃a : Prod β β⦄, Membership.mem d a …
    ⊢ IsLUB (Set.image (fun x => Max.max x.1 x.2) d) ((fun x => Max.max x.1 x.2) { …
  -/
  simp only [Prod.forall, mem_setOf_eq, Prod.mk_le_mk] at hdp
  /-
    β : Type u_2
    inst✝ : SemilatticeSup β
    D : Set (Set (Prod β β))
    d : Set (Prod β β)
    a✝² : Membership.mem D d
    a✝¹ : d.Nonempty
    a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    p₁ p₂ : β
    hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
    ⊢ IsLUB (Set.image (fun x => Max.max x.1 x.2) d) ((fun x => Max.max x.1 x.2) { …
  -/
  rw [IsLUB, IsLeast, upperBounds]
  /-
    β : Type u_2
    inst✝ : SemilatticeSup β
    D : Set (Set (Prod β β))
    d : Set (Prod β β)
    a✝² : Membership.mem D d
    a✝¹ : d.Nonempty
    a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    p₁ p₂ : β
    hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
    ⊢ And (Membership.mem (setOf fun x => ∀ ⦃a : β⦄, Membership.mem (Set.image (fu …
  -/
  constructor
    /-
      case left
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      ⊢ Membership.mem (setOf fun x => ∀ ⦃a : β⦄, Membership.mem (Set.image (fun x = …
    -/
  · simp only [mem_image, Prod.exists, forall_exists_index, and_imp, mem_setOf_eq]
    /-
      case left
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      ⊢ ∀ ⦃a : β⦄ (x x_1 : β), Membership.mem d { fst := x, snd := x_1 } → Eq (Max.m …
    -/
    intro a b₁ b₂ hbd hba
    /-
      case left
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      a b₁ b₂ : β
      hbd : Membership.mem d { fst := b₁, snd := b₂ }
      hba : Eq (Max.max b₁ b₂) a
      ⊢ LE.le a (Max.max p₁ p₂)
    -/
    rw [← hba]
    /-
      case left
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      a b₁ b₂ : β
      hbd : Membership.mem d { fst := b₁, snd := b₂ }
      hba : Eq (Max.max b₁ b₂) a
      ⊢ LE.le (Max.max b₁ b₂) (Max.max p₁ p₂)
    -/
    exact sup_le_sup (hdp.1 _ _ hbd).1 (hdp.1 _ _ hbd).2
    /-
      🎉 no goals
    -/
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      ⊢ Membership.mem (lowerBounds (setOf fun x => ∀ ⦃a : β⦄, Membership.mem (Set.i …
    -/
  · simp only [mem_image, Prod.exists, forall_exists_index, and_imp]
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      ⊢ Membership.mem (lowerBounds (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Members …
    -/
    intro b hb
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      ⊢ LE.le (Max.max p₁ p₂) b
    -/
    simp only [sup_le_iff]
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      ⊢ And (LE.le p₁ b) (LE.le p₂ b)
    -/
    have e1 : (p₁, p₂) ∈ lowerBounds {x | ∀ (b₁ b₂ : β), (b₁, b₂) ∈ d → (b₁, b₂) ≤ x} := hdp.2
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      e1 : Membership.mem (lowerBounds (setOf fun x => ∀ (b₁ b₂ : β), Membership.mem …
      ⊢ And (LE.le p₁ b) (LE.le p₂ b)
    -/
    rw [lowerBounds] at e1
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      e1 : Membership.mem (setOf fun x => ∀ ⦃a : Prod β β⦄, Membership.mem (setOf fu …
      ⊢ And (LE.le p₁ b) (LE.le p₂ b)
    -/
    simp only [mem_setOf_eq, Prod.forall, Prod.mk_le_mk] at e1
    /-
      case right
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      e1 : ∀ (a b : β), (∀ (b₁ b₂ : β), Membership.mem d { fst := b₁, snd := b₂ } →  …
      ⊢ And (LE.le p₁ b) (LE.le p₂ b)
    -/
    apply e1
    /-
      case right.a
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      e1 : ∀ (a b : β), (∀ (b₁ b₂ : β), Membership.mem d { fst := b₁, snd := b₂ } →  …
      ⊢ ∀ (b₁ b₂ : β), Membership.mem d { fst := b₁, snd := b₂ } → And (LE.le b₁ b)  …
    -/
    intro b₁ b₂ hb'
    /-
      case right.a
      β : Type u_2
      inst✝ : SemilatticeSup β
      D : Set (Set (Prod β β))
      d : Set (Prod β β)
      a✝² : Membership.mem D d
      a✝¹ : d.Nonempty
      a✝ : DirectedOn (fun x1 x2 => LE.le x1 x2) d
      p₁ p₂ : β
      hdp : And (∀ (a b : β), Membership.mem d { fst := a, snd := b } → And (LE.le a …
      b : β
      hb : Membership.mem (setOf fun x => ∀ ⦃a : β⦄ (x_1 x_2 : β), Membership.mem d  …
      e1 : ∀ (a b : β), (∀ (b₁ b₂ : β), Membership.mem d { fst := b₁, snd := b₂ } →  …
      b₁ b₂ : β
      hb' : Membership.mem d { fst := b₁, snd := b₂ }
      ⊢ And (LE.le b₁ b) (LE.le b₂ b)
    -/
    exact sup_le_iff.mp (hb b₁ b₂ hb' rfl)
    /-
      🎉 no goals
    -/


