/-- The lexicographical order on a sigma type. It takes in a relation on the index type and a
relation for each summand. `a` is related to `b` iff their summands are related or they are in the
same summand and are related through the summand's relation. -/
inductive Lex (r : ι → ι → Prop) (s : ∀ i, α i → α i → Prop) : ∀ _ _ : Σ i, α i, Prop
  | left {i j : ι} (a : α i) (b : α j) : r i j → Lex r s ⟨i, a⟩ ⟨j, b⟩
  | right {i : ι} (a b : α i) : s i a b → Lex r s ⟨i, a⟩ ⟨i, b⟩


theorem lex_iff : Lex r s a b ↔ r a.1 b.1 ∨ ∃ h : a.1 = b.1, s b.1 (h.rec a.2) b.2 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    a b : Sigma fun i => α i
    ⊢ Iff (Sigma.Lex r s a b) (Or (r a.fst b.fst) (Exists fun h => s b.fst (Eq.rec …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      ⊢ Sigma.Lex r s a b → Or (r a.fst b.fst) (Exists fun h => s b.fst (Eq.rec a.sn …
    -/
  · rintro (⟨a, b, hij⟩ | ⟨a, b, hab⟩)
      /-
        case mp.left
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i✝ j✝ : ι
        a : α i✝
        b : α j✝
        hij : r i✝ j✝
        ⊢ Or (r ⟨i✝, a⟩.fst ⟨j✝, b⟩.fst) (Exists fun h => s ⟨j✝, b⟩.fst (Eq.rec ⟨i✝, a …
      -/
    · exact Or.inl hij
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i✝ : ι
        a b : α i✝
        hab : s i✝ a b
        ⊢ Or (r ⟨i✝, a⟩.fst ⟨i✝, b⟩.fst) (Exists fun h => s ⟨i✝, b⟩.fst (Eq.rec ⟨i✝, a …
      -/
    · exact Or.inr ⟨rfl, hab⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      ⊢ Or (r a.fst b.fst) (Exists fun h => s b.fst (Eq.rec a.snd h) b.snd) → Sigma. …
    -/
  · obtain ⟨i, a⟩ := a
    /-
      case mpr.mk
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      b : Sigma fun i => α i
      i : ι
      a : α i
      ⊢ Or (r ⟨i, a⟩.fst b.fst) (Exists fun h => s b.fst (Eq.rec ⟨i, a⟩.snd h) b.snd …
    -/
    obtain ⟨j, b⟩ := b
    /-
      case mpr.mk.mk
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ Or (r ⟨i, a⟩.fst ⟨j, b⟩.fst) (Exists fun h => s ⟨j, b⟩.fst (Eq.rec ⟨i, a⟩.sn …
    -/
    dsimp only
    /-
      case mpr.mk.mk
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ Or (r i j) (Exists fun h => s j (Eq.rec a h) b) → Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩
    -/
    rintro (h | ⟨rfl, h⟩)
      /-
        case mpr.mk.mk.inl
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i : ι
        a : α i
        j : ι
        b : α j
        h : r i j
        ⊢ Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩
      -/
    · exact Lex.left _ _ h
      /-
        🎉 no goals
      -/
      /-
        case mpr.mk.mk.inr.intro
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i : ι
        a b : α i
        h : s i (Eq.rec a ⋯) b
        ⊢ Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩
      -/
    · exact Lex.right _ _ h
      /-
        🎉 no goals
      -/


instance Lex.decidable (r : ι → ι → Prop) (s : ∀ i, α i → α i → Prop) [DecidableEq ι]
    [DecidableRel r] [∀ i, DecidableRel (s i)] : DecidableRel (Lex r s) := fun _ _ =>
  decidable_of_decidable_of_iff lex_iff.symm


theorem Lex.mono (hr : ∀ a b, r₁ a b → r₂ a b) (hs : ∀ i a b, s₁ i a b → s₂ i a b) {a b : Σ i, α i}
    (h : Lex r₁ s₁ a b) : Lex r₂ s₂ a b := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    r₁ r₂ : ι → ι → Prop
    s₁ s₂ : (i : ι) → α i → α i → Prop
    hr : ∀ (a b : ι), r₁ a b → r₂ a b
    hs : ∀ (i : ι) (a b : α i), s₁ i a b → s₂ i a b
    a b : Sigma fun i => α i
    h : Sigma.Lex r₁ s₁ a b
    ⊢ Sigma.Lex r₂ s₂ a b
  -/
  obtain ⟨a, b, hij⟩ | ⟨a, b, hab⟩ := h
    /-
      case left
      ι : Type u_1
      α : ι → Type u_2
      r₁ r₂ : ι → ι → Prop
      s₁ s₂ : (i : ι) → α i → α i → Prop
      hr : ∀ (a b : ι), r₁ a b → r₂ a b
      hs : ∀ (i : ι) (a b : α i), s₁ i a b → s₂ i a b
      i✝ j✝ : ι
      a : α i✝
      b : α j✝
      hij : r₁ i✝ j✝
      ⊢ Sigma.Lex r₂ s₂ ⟨i✝, a⟩ ⟨j✝, b⟩
    -/
  · exact Lex.left _ _ (hr _ _ hij)
    /-
      🎉 no goals
    -/
    /-
      case right
      ι : Type u_1
      α : ι → Type u_2
      r₁ r₂ : ι → ι → Prop
      s₁ s₂ : (i : ι) → α i → α i → Prop
      hr : ∀ (a b : ι), r₁ a b → r₂ a b
      hs : ∀ (i : ι) (a b : α i), s₁ i a b → s₂ i a b
      i✝ : ι
      a b : α i✝
      hab : s₁ i✝ a b
      ⊢ Sigma.Lex r₂ s₂ ⟨i✝, a⟩ ⟨i✝, b⟩
    -/
  · exact Lex.right _ _ (hs _ _ _ hab)
    /-
      🎉 no goals
    -/


theorem Lex.mono_left (hr : ∀ a b, r₁ a b → r₂ a b) {a b : Σ i, α i} (h : Lex r₁ s a b) :
    Lex r₂ s a b :=
  h.mono hr fun _ _ _ => id


theorem Lex.mono_right (hs : ∀ i a b, s₁ i a b → s₂ i a b) {a b : Σ i, α i} (h : Lex r s₁ a b) :
    Lex r s₂ a b :=
  h.mono (fun _ _ => id) hs


theorem lex_swap : Lex (Function.swap r) s a b ↔ Lex r (fun i => Function.swap (s i)) b a := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    a b : Sigma fun i => α i
    ⊢ Iff (Sigma.Lex (Function.swap r) s a b) (Sigma.Lex r (fun i => Function.swap …
  -/
  constructor <;>
      /-
        case mp
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        a b : Sigma fun i => α i
        ⊢ Sigma.Lex (Function.swap r) s a b → Sigma.Lex r (fun i => Function.swap (s i …
      -/
      /-
        case mp.left
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i✝ j✝ : ι
        a : α i✝
        b : α j✝
        h : Function.swap r i✝ j✝
        ⊢ Sigma.Lex r (fun i => Function.swap (s i)) ⟨j✝, b⟩ ⟨i✝, a⟩
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr.left
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i✝ j✝ : ι
        a : α i✝
        b : α j✝
        h : r i✝ j✝
        ⊢ Sigma.Lex (Function.swap r) s ⟨j✝, b⟩ ⟨i✝, a⟩
      -/
      exacts [Lex.left _ _ h, Lex.right _ _ h]
      /-
        🎉 no goals
      -/


instance [∀ i, IsRefl (α i) (s i)] : IsRefl _ (Lex r s) :=
  ⟨fun ⟨_, _⟩ => Lex.right _ _ <| refl _⟩


instance [IsIrrefl ι r] [∀ i, IsIrrefl (α i) (s i)] : IsIrrefl _ (Lex r s) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      inst✝¹ : IsIrrefl ι r
      inst✝ : ∀ (i : ι), IsIrrefl (α i) (s i)
      ⊢ ∀ (a : Sigma fun i => α i), Not (Sigma.Lex r s a a)
    -/
    rintro _ (⟨a, b, hi⟩ | ⟨a, b, ha⟩)
      /-
        case left
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b : Sigma fun i => α i
        inst✝¹ : IsIrrefl ι r
        inst✝ : ∀ (i : ι), IsIrrefl (α i) (s i)
        i✝ : ι
        a : α i✝
        hi : r i✝ i✝
        ⊢ False
      -/
    · exact irrefl _ hi
      /-
        🎉 no goals
      -/
      /-
        case right
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b : Sigma fun i => α i
        inst✝¹ : IsIrrefl ι r
        inst✝ : ∀ (i : ι), IsIrrefl (α i) (s i)
        i✝ : ι
        a : α i✝
        ha : s i✝ a a
        ⊢ False
      -/
    · exact irrefl _ ha
      /-
        🎉 no goals
      -/
      ⟩


instance [IsTrans ι r] [∀ i, IsTrans (α i) (s i)] : IsTrans _ (Lex r s) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      inst✝¹ : IsTrans ι r
      inst✝ : ∀ (i : ι), IsTrans (α i) (s i)
      ⊢ ∀ (a b c : Sigma fun i => α i), Sigma.Lex r s a b → Sigma.Lex r s b c → Sigm …
    -/
    rintro _ _ _ (⟨a, b, hij⟩ | ⟨a, b, hab⟩) (⟨_, c, hk⟩ | ⟨_, c, hc⟩)
      /-
        case left.left
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrans ι r
        inst✝ : ∀ (i : ι), IsTrans (α i) (s i)
        i✝ j✝¹ : ι
        a : α i✝
        b : α j✝¹
        hij : r i✝ j✝¹
        j✝ : ι
        c : α j✝
        hk : r j✝¹ j✝
        ⊢ Sigma.Lex r s ⟨i✝, a⟩ ⟨j✝, c⟩
      -/
    · exact Lex.left _ _ (_root_.trans hij hk)
      /-
        🎉 no goals
      -/
      /-
        case left.right
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrans ι r
        inst✝ : ∀ (i : ι), IsTrans (α i) (s i)
        i✝ j✝ : ι
        a : α i✝
        b : α j✝
        hij : r i✝ j✝
        c : α j✝
        hc : s j✝ b c
        ⊢ Sigma.Lex r s ⟨i✝, a⟩ ⟨j✝, c⟩
      -/
    · exact Lex.left _ _ hij
      /-
        🎉 no goals
      -/
      /-
        case right.left
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrans ι r
        inst✝ : ∀ (i : ι), IsTrans (α i) (s i)
        i✝ : ι
        a b : α i✝
        hab : s i✝ a b
        j✝ : ι
        c : α j✝
        hk : r i✝ j✝
        ⊢ Sigma.Lex r s ⟨i✝, a⟩ ⟨j✝, c⟩
      -/
    · exact Lex.left _ _ hk
      /-
        🎉 no goals
      -/
      /-
        case right.right
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrans ι r
        inst✝ : ∀ (i : ι), IsTrans (α i) (s i)
        i✝ : ι
        a b : α i✝
        hab : s i✝ a b
        c : α i✝
        hc : s i✝ b c
        ⊢ Sigma.Lex r s ⟨i✝, a⟩ ⟨i✝, c⟩
      -/
    · exact Lex.right _ _ (_root_.trans hab hc)⟩
      /-
        🎉 no goals
      -/


instance [IsSymm ι r] [∀ i, IsSymm (α i) (s i)] : IsSymm _ (Lex r s) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      inst✝¹ : IsSymm ι r
      inst✝ : ∀ (i : ι), IsSymm (α i) (s i)
      ⊢ ∀ (a b : Sigma fun i => α i), Sigma.Lex r s a b → Sigma.Lex r s b a
    -/
    rintro _ _ (⟨a, b, hij⟩ | ⟨a, b, hab⟩)
      /-
        case left
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsSymm ι r
        inst✝ : ∀ (i : ι), IsSymm (α i) (s i)
        i✝ j✝ : ι
        a : α i✝
        b : α j✝
        hij : r i✝ j✝
        ⊢ Sigma.Lex r s ⟨j✝, b⟩ ⟨i✝, a⟩
      -/
    · exact Lex.left _ _ (symm hij)
      /-
        🎉 no goals
      -/
      /-
        case right
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsSymm ι r
        inst✝ : ∀ (i : ι), IsSymm (α i) (s i)
        i✝ : ι
        a b : α i✝
        hab : s i✝ a b
        ⊢ Sigma.Lex r s ⟨i✝, b⟩ ⟨i✝, a⟩
      -/
    · exact Lex.right _ _ (symm hab)
      /-
        🎉 no goals
      -/
      ⟩


instance [IsAsymm ι r] [∀ i, IsAntisymm (α i) (s i)] : IsAntisymm _ (Lex r s) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      inst✝¹ : IsAsymm ι r
      inst✝ : ∀ (i : ι), IsAntisymm (α i) (s i)
      ⊢ ∀ (a b : Sigma fun i => α i), Sigma.Lex r s a b → Sigma.Lex r s b a → Eq a b
    -/
    rintro _ _ (⟨a, b, hij⟩ | ⟨a, b, hab⟩) (⟨_, _, hji⟩ | ⟨_, _, hba⟩)
      /-
        case left.left
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsAsymm ι r
        inst✝ : ∀ (i : ι), IsAntisymm (α i) (s i)
        i✝ j✝ : ι
        a : α i✝
        b : α j✝
        hij : r i✝ j✝
        hji : r j✝ i✝
        ⊢ Eq ⟨i✝, a⟩ ⟨j✝, b⟩
      -/
    · exact (asymm hij hji).elim
      /-
        🎉 no goals
      -/
      /-
        case left.right
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsAsymm ι r
        inst✝ : ∀ (i : ι), IsAntisymm (α i) (s i)
        i✝ : ι
        a b : α i✝
        hij : r i✝ i✝
        hba : s i✝ b a
        ⊢ Eq ⟨i✝, a⟩ ⟨i✝, b⟩
      -/
    · exact (irrefl _ hij).elim
      /-
        🎉 no goals
      -/
      /-
        case right.left
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsAsymm ι r
        inst✝ : ∀ (i : ι), IsAntisymm (α i) (s i)
        i✝ : ι
        a b : α i✝
        hab : s i✝ a b
        hji : r i✝ i✝
        ⊢ Eq ⟨i✝, a⟩ ⟨i✝, b⟩
      -/
    · exact (irrefl _ hji).elim
      /-
        🎉 no goals
      -/
      /-
        case right.right
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsAsymm ι r
        inst✝ : ∀ (i : ι), IsAntisymm (α i) (s i)
        i✝ : ι
        a b : α i✝
        hab : s i✝ a b
        hba : s i✝ b a
        ⊢ Eq ⟨i✝, a⟩ ⟨i✝, b⟩
      -/
    · exact congr_arg (Sigma.mk _ ·) <| antisymm hab hba⟩
      /-
        🎉 no goals
      -/


instance [IsTrichotomous ι r] [∀ i, IsTotal (α i) (s i)] : IsTotal _ (Lex r s) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      inst✝¹ : IsTrichotomous ι r
      inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
      ⊢ ∀ (a b : Sigma fun i => α i), Or (Sigma.Lex r s a b) (Sigma.Lex r s b a)
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩
    /-
      case mk.mk
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a✝ b✝ : Sigma fun i => α i
      inst✝¹ : IsTrichotomous ι r
      inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩) (Sigma.Lex r s ⟨j, b⟩ ⟨i, a⟩)
    -/
    obtain hij | rfl | hji := trichotomous_of r i j
      /-
        case mk.mk.inl
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrichotomous ι r
        inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
        i : ι
        a : α i
        j : ι
        b : α j
        hij : r i j
        ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩) (Sigma.Lex r s ⟨j, b⟩ ⟨i, a⟩)
      -/
    · exact Or.inl (Lex.left _ _ hij)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.inr.inl
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrichotomous ι r
        inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
        i : ι
        a b : α i
        ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩) (Sigma.Lex r s ⟨i, b⟩ ⟨i, a⟩)
      -/
    · obtain hab | hba := total_of (s i) a b
        /-
          case mk.mk.inr.inl.inl
          ι : Type u_1
          α : ι → Type u_2
          r r₁ r₂ : ι → ι → Prop
          s s₁ s₂ : (i : ι) → α i → α i → Prop
          a✝ b✝ : Sigma fun i => α i
          inst✝¹ : IsTrichotomous ι r
          inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
          i : ι
          a b : α i
          hab : s i a b
          ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩) (Sigma.Lex r s ⟨i, b⟩ ⟨i, a⟩)
        -/
      · exact Or.inl (Lex.right _ _ hab)
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.inr.inl.inr
          ι : Type u_1
          α : ι → Type u_2
          r r₁ r₂ : ι → ι → Prop
          s s₁ s₂ : (i : ι) → α i → α i → Prop
          a✝ b✝ : Sigma fun i => α i
          inst✝¹ : IsTrichotomous ι r
          inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
          i : ι
          a b : α i
          hba : s i b a
          ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩) (Sigma.Lex r s ⟨i, b⟩ ⟨i, a⟩)
        -/
      · exact Or.inr (Lex.right _ _ hba)
        /-
          🎉 no goals
        -/
      /-
        case mk.mk.inr.inr
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrichotomous ι r
        inst✝ : ∀ (i : ι), IsTotal (α i) (s i)
        i : ι
        a : α i
        j : ι
        b : α j
        hji : r j i
        ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩) (Sigma.Lex r s ⟨j, b⟩ ⟨i, a⟩)
      -/
    · exact Or.inr (Lex.left _ _ hji)⟩
      /-
        🎉 no goals
      -/


instance [IsTrichotomous ι r] [∀ i, IsTrichotomous (α i) (s i)] : IsTrichotomous _ (Lex r s) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a b : Sigma fun i => α i
      inst✝¹ : IsTrichotomous ι r
      inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
      ⊢ ∀ (a b : Sigma fun i => α i), Or (Sigma.Lex r s a b) (Or (Eq a b) (Sigma.Lex …
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩
    /-
      case mk.mk
      ι : Type u_1
      α : ι → Type u_2
      r r₁ r₂ : ι → ι → Prop
      s s₁ s₂ : (i : ι) → α i → α i → Prop
      a✝ b✝ : Sigma fun i => α i
      inst✝¹ : IsTrichotomous ι r
      inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩) (Or (Eq ⟨i, a⟩ ⟨j, b⟩) (Sigma.Lex r s ⟨j, b …
    -/
    obtain hij | rfl | hji := trichotomous_of r i j
      /-
        case mk.mk.inl
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrichotomous ι r
        inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
        i : ι
        a : α i
        j : ι
        b : α j
        hij : r i j
        ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩) (Or (Eq ⟨i, a⟩ ⟨j, b⟩) (Sigma.Lex r s ⟨j, b …
      -/
    · exact Or.inl (Lex.left _ _ hij)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.inr.inl
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrichotomous ι r
        inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
        i : ι
        a b : α i
        ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩) (Or (Eq ⟨i, a⟩ ⟨i, b⟩) (Sigma.Lex r s ⟨i, b …
      -/
    · obtain hab | rfl | hba := trichotomous_of (s i) a b
        /-
          case mk.mk.inr.inl.inl
          ι : Type u_1
          α : ι → Type u_2
          r r₁ r₂ : ι → ι → Prop
          s s₁ s₂ : (i : ι) → α i → α i → Prop
          a✝ b✝ : Sigma fun i => α i
          inst✝¹ : IsTrichotomous ι r
          inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
          i : ι
          a b : α i
          hab : s i a b
          ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩) (Or (Eq ⟨i, a⟩ ⟨i, b⟩) (Sigma.Lex r s ⟨i, b …
        -/
      · exact Or.inl (Lex.right _ _ hab)
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.inr.inl.inr.inl
          ι : Type u_1
          α : ι → Type u_2
          r r₁ r₂ : ι → ι → Prop
          s s₁ s₂ : (i : ι) → α i → α i → Prop
          a✝ b : Sigma fun i => α i
          inst✝¹ : IsTrichotomous ι r
          inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
          i : ι
          a : α i
          ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, a⟩) (Or (Eq ⟨i, a⟩ ⟨i, a⟩) (Sigma.Lex r s ⟨i, a …
        -/
      · exact Or.inr (Or.inl rfl)
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.inr.inl.inr.inr
          ι : Type u_1
          α : ι → Type u_2
          r r₁ r₂ : ι → ι → Prop
          s s₁ s₂ : (i : ι) → α i → α i → Prop
          a✝ b✝ : Sigma fun i => α i
          inst✝¹ : IsTrichotomous ι r
          inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
          i : ι
          a b : α i
          hba : s i b a
          ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨i, b⟩) (Or (Eq ⟨i, a⟩ ⟨i, b⟩) (Sigma.Lex r s ⟨i, b …
        -/
      · exact Or.inr (Or.inr <| Lex.right _ _ hba)
        /-
          🎉 no goals
        -/
      /-
        case mk.mk.inr.inr
        ι : Type u_1
        α : ι → Type u_2
        r r₁ r₂ : ι → ι → Prop
        s s₁ s₂ : (i : ι) → α i → α i → Prop
        a✝ b✝ : Sigma fun i => α i
        inst✝¹ : IsTrichotomous ι r
        inst✝ : ∀ (i : ι), IsTrichotomous (α i) (s i)
        i : ι
        a : α i
        j : ι
        b : α j
        hji : r j i
        ⊢ Or (Sigma.Lex r s ⟨i, a⟩ ⟨j, b⟩) (Or (Eq ⟨i, a⟩ ⟨j, b⟩) (Sigma.Lex r s ⟨j, b …
      -/
    · exact Or.inr (Or.inr <| Lex.left _ _ hji)⟩
      /-
        🎉 no goals
      -/


theorem lex_iff {a b : Σ' i, α i} :
    Lex r s a b ↔ r a.1 b.1 ∨ ∃ h : a.1 = b.1, s b.1 (h.rec a.2) b.2 := by
  /-
    ι : Sort u_1
    α : ι → Sort u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    a b : PSigma fun i => α i
    ⊢ Iff (PSigma.Lex r s a b) (Or (r a.fst b.fst) (Exists fun h => s b.fst (Eq.re …
  -/
  constructor
    /-
      case mp
      ι : Sort u_1
      α : ι → Sort u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      a b : PSigma fun i => α i
      ⊢ PSigma.Lex r s a b → Or (r a.fst b.fst) (Exists fun h => s b.fst (Eq.rec a.s …
    -/
  · rintro (⟨a, b, hij⟩ | ⟨i, hab⟩)
      /-
        case mp.left
        ι : Sort u_1
        α : ι → Sort u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        a₁✝ : ι
        a : α a₁✝
        a₂✝ : ι
        b : α a₂✝
        hij : r a₁✝ a₂✝
        ⊢ Or (r ⟨a₁✝, a⟩.fst ⟨a₂✝, b⟩.fst) (Exists fun h => s ⟨a₂✝, b⟩.fst (Eq.rec ⟨a₁ …
      -/
    · exact Or.inl hij
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        ι : Sort u_1
        α : ι → Sort u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i : ι
        b₁✝ b₂✝ : α i
        hab : s i b₁✝ b₂✝
        ⊢ Or (r ⟨i, b₁✝⟩.fst ⟨i, b₂✝⟩.fst) (Exists fun h => s ⟨i, b₂✝⟩.fst (Eq.rec ⟨i, …
      -/
    · exact Or.inr ⟨rfl, hab⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Sort u_1
      α : ι → Sort u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      a b : PSigma fun i => α i
      ⊢ Or (r a.fst b.fst) (Exists fun h => s b.fst (Eq.rec a.snd h) b.snd) → PSigma …
    -/
  · obtain ⟨i, a⟩ := a
    /-
      case mpr.mk
      ι : Sort u_1
      α : ι → Sort u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      b : PSigma fun i => α i
      i : ι
      a : α i
      ⊢ Or (r ⟨i, a⟩.fst b.fst) (Exists fun h => s b.fst (Eq.rec ⟨i, a⟩.snd h) b.snd …
    -/
    obtain ⟨j, b⟩ := b
    /-
      case mpr.mk.mk
      ι : Sort u_1
      α : ι → Sort u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ Or (r ⟨i, a⟩.fst ⟨j, b⟩.fst) (Exists fun h => s ⟨j, b⟩.fst (Eq.rec ⟨i, a⟩.sn …
    -/
    dsimp only
    /-
      case mpr.mk.mk
      ι : Sort u_1
      α : ι → Sort u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      i : ι
      a : α i
      j : ι
      b : α j
      ⊢ Or (r i j) (Exists fun h => s j (Eq.rec a h) b) → PSigma.Lex r s ⟨i, a⟩ ⟨j, b⟩
    -/
    rintro (h | ⟨rfl, h⟩)
      /-
        case mpr.mk.mk.inl
        ι : Sort u_1
        α : ι → Sort u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i : ι
        a : α i
        j : ι
        b : α j
        h : r i j
        ⊢ PSigma.Lex r s ⟨i, a⟩ ⟨j, b⟩
      -/
    · exact Lex.left _ _ h
      /-
        🎉 no goals
      -/
      /-
        case mpr.mk.mk.inr.intro
        ι : Sort u_1
        α : ι → Sort u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        i : ι
        a b : α i
        h : s i (Eq.rec a ⋯) b
        ⊢ PSigma.Lex r s ⟨i, a⟩ ⟨i, b⟩
      -/
    · exact Lex.right _ h
      /-
        🎉 no goals
      -/


theorem Lex.mono {r₁ r₂ : ι → ι → Prop} {s₁ s₂ : ∀ i, α i → α i → Prop}
    (hr : ∀ a b, r₁ a b → r₂ a b) (hs : ∀ i a b, s₁ i a b → s₂ i a b) {a b : Σ' i, α i}
    (h : Lex r₁ s₁ a b) : Lex r₂ s₂ a b := by
  /-
    ι : Sort u_1
    α : ι → Sort u_2
    r₁ r₂ : ι → ι → Prop
    s₁ s₂ : (i : ι) → α i → α i → Prop
    hr : ∀ (a b : ι), r₁ a b → r₂ a b
    hs : ∀ (i : ι) (a b : α i), s₁ i a b → s₂ i a b
    a b : PSigma fun i => α i
    h : PSigma.Lex r₁ s₁ a b
    ⊢ PSigma.Lex r₂ s₂ a b
  -/
  obtain ⟨a, b, hij⟩ | ⟨i, hab⟩ := h
    /-
      case left
      ι : Sort u_1
      α : ι → Sort u_2
      r₁ r₂ : ι → ι → Prop
      s₁ s₂ : (i : ι) → α i → α i → Prop
      hr : ∀ (a b : ι), r₁ a b → r₂ a b
      hs : ∀ (i : ι) (a b : α i), s₁ i a b → s₂ i a b
      a₁✝ : ι
      a : α a₁✝
      a₂✝ : ι
      b : α a₂✝
      hij : r₁ a₁✝ a₂✝
      ⊢ PSigma.Lex r₂ s₂ ⟨a₁✝, a⟩ ⟨a₂✝, b⟩
    -/
  · exact Lex.left _ _ (hr _ _ hij)
    /-
      🎉 no goals
    -/
    /-
      case right
      ι : Sort u_1
      α : ι → Sort u_2
      r₁ r₂ : ι → ι → Prop
      s₁ s₂ : (i : ι) → α i → α i → Prop
      hr : ∀ (a b : ι), r₁ a b → r₂ a b
      hs : ∀ (i : ι) (a b : α i), s₁ i a b → s₂ i a b
      i : ι
      b₁✝ b₂✝ : α i
      hab : s₁ i b₁✝ b₂✝
      ⊢ PSigma.Lex r₂ s₂ ⟨i, b₁✝⟩ ⟨i, b₂✝⟩
    -/
  · exact Lex.right _ (hs _ _ _ hab)
    /-
      🎉 no goals
    -/


theorem Lex.mono_left {r₁ r₂ : ι → ι → Prop} {s : ∀ i, α i → α i → Prop}
    (hr : ∀ a b, r₁ a b → r₂ a b) {a b : Σ' i, α i} (h : Lex r₁ s a b) : Lex r₂ s a b :=
  h.mono hr fun _ _ _ => id


theorem Lex.mono_right {r : ι → ι → Prop} {s₁ s₂ : ∀ i, α i → α i → Prop}
    (hs : ∀ i a b, s₁ i a b → s₂ i a b) {a b : Σ' i, α i} (h : Lex r s₁ a b) : Lex r s₂ a b :=
  h.mono (fun _ _ => id) hs


