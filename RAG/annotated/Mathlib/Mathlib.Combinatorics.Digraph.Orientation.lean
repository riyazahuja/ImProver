/--
Orientation-forgetting map from `Digraph` to `SimpleGraph` that gives an unoriented edge if
either orientation is present.
-/
def toSimpleGraphInclusive (G : Digraph V) : SimpleGraph V := SimpleGraph.fromRel G.Adj


/--
Orientation-forgetting map from `Digraph` to `SimpleGraph` that gives an unoriented edge if
both orientations are present.
-/
def toSimpleGraphStrict (G : Digraph V) : SimpleGraph V where
  Adj v w := v ≠ w ∧ G.Adj v w ∧ G.Adj w v
  symm _ _ h := And.intro h.1.symm h.2.symm
  loopless _ h := h.1 rfl


lemma toSimpleGraphStrict_subgraph_toSimpleGraphInclusive (G : Digraph V) :
    G.toSimpleGraphStrict ≤ G.toSimpleGraphInclusive :=
  fun _ _ h ↦ ⟨h.1, Or.inl h.2.1⟩


@[mono]
lemma toSimpleGraphInclusive_mono : Monotone (toSimpleGraphInclusive : _ → SimpleGraph V) := by
  /-
    V : Type u_1
    ⊢ Monotone Digraph.toSimpleGraphInclusive
  -/
  intro _ _ h₁ _ _ h₂
  /-
    V : Type u_1
    a✝ b✝ : Digraph V
    h₁ : LE.le a✝ b✝
    v✝ w✝ : V
    h₂ : a✝.toSimpleGraphInclusive.Adj v✝ w✝
    ⊢ b✝.toSimpleGraphInclusive.Adj v✝ w✝
  -/
  apply And.intro h₂.1
  /-
    V : Type u_1
    a✝ b✝ : Digraph V
    h₁ : LE.le a✝ b✝
    v✝ w✝ : V
    h₂ : a✝.toSimpleGraphInclusive.Adj v✝ w✝
    ⊢ Or (b✝.Adj v✝ w✝) (b✝.Adj w✝ v✝)
  -/
  cases h₂.2
    /-
      case inl
      V : Type u_1
      a✝ b✝ : Digraph V
      h₁ : LE.le a✝ b✝
      v✝ w✝ : V
      h₂ : a✝.toSimpleGraphInclusive.Adj v✝ w✝
      h✝ : a✝.Adj v✝ w✝
      ⊢ Or (b✝.Adj v✝ w✝) (b✝.Adj w✝ v✝)
    -/
  · exact Or.inl <| h₁ ‹_›
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      a✝ b✝ : Digraph V
      h₁ : LE.le a✝ b✝
      v✝ w✝ : V
      h₂ : a✝.toSimpleGraphInclusive.Adj v✝ w✝
      h✝ : a✝.Adj w✝ v✝
      ⊢ Or (b✝.Adj v✝ w✝) (b✝.Adj w✝ v✝)
    -/
  · exact Or.inr <| h₁ ‹_›
    /-
      🎉 no goals
    -/


@[mono]
lemma toSimpleGraphStrict_mono : Monotone (toSimpleGraphStrict : _ → SimpleGraph V) :=
  fun _ _ h₁ _ _ h₂ ↦ And.intro h₂.1 <| And.intro (h₁ h₂.2.1) (h₁ h₂.2.2)


@[simp]
lemma toSimpleGraphInclusive_top : (⊤ : Digraph V).toSimpleGraphInclusive = ⊤ := by
  /-
    V : Type u_1
    ⊢ Eq Top.top.toSimpleGraphInclusive Top.top
  -/
  ext; exact ⟨And.left, fun h ↦ ⟨h.ne, Or.inl trivial⟩⟩
       /-
         🎉 no goals
       -/


@[simp]
lemma toSimpleGraphStrict_top : (⊤ : Digraph V).toSimpleGraphStrict = ⊤ := by
  /-
    V : Type u_1
    ⊢ Eq Top.top.toSimpleGraphStrict Top.top
  -/
  ext; exact ⟨And.left, fun h ↦ ⟨h.ne, trivial, trivial⟩⟩
       /-
         🎉 no goals
       -/


@[simp]
lemma toSimpleGraphInclusive_bot : (⊥ : Digraph V).toSimpleGraphInclusive = ⊥ := by
  /-
    V : Type u_1
    ⊢ Eq Bot.bot.toSimpleGraphInclusive Bot.bot
  -/
  ext; exact ⟨fun ⟨_, h⟩ ↦ by tauto, False.elim⟩
       /-
         🎉 no goals
       -/


@[simp]
lemma toSimpleGraphStrict_bot : (⊥ : Digraph V).toSimpleGraphStrict = ⊥ := by
  /-
    V : Type u_1
    ⊢ Eq Bot.bot.toSimpleGraphStrict Bot.bot
  -/
  ext; exact ⟨fun ⟨_, h⟩ ↦ by tauto, False.elim⟩
       /-
         🎉 no goals
       -/


