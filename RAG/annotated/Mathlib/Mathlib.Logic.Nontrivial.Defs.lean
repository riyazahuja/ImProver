/-- Predicate typeclass for expressing that a type is not reduced to a single element. In rings,
this is equivalent to `0 ≠ 1`. In vector spaces, this is equivalent to positive dimension. -/
class Nontrivial (α : Type*) : Prop where
  /-- In a nontrivial type, there exists a pair of distinct terms. -/
  exists_pair_ne : ∃ x y : α, x ≠ y


theorem nontrivial_iff : Nontrivial α ↔ ∃ x y : α, x ≠ y :=
  ⟨fun h ↦ h.exists_pair_ne, fun h ↦ ⟨h⟩⟩


theorem exists_pair_ne (α : Type*) [Nontrivial α] : ∃ x y : α, x ≠ y :=
  Nontrivial.exists_pair_ne

-- See Note [decidable namespace]

protected theorem Decidable.exists_ne [Nontrivial α] [DecidableEq α] (x : α) : ∃ y, y ≠ x := by
  /-
    α : Type u_1
    inst✝¹ : Nontrivial α
    inst✝ : DecidableEq α
    x : α
    ⊢ Exists fun y => Ne y x
  -/
  rcases exists_pair_ne α with ⟨y, y', h⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Nontrivial α
    inst✝ : DecidableEq α
    x y y' : α
    h : Ne y y'
    ⊢ Exists fun y => Ne y x
  -/
  by_cases hx : x = y
    /-
      case pos
      α : Type u_1
      inst✝¹ : Nontrivial α
      inst✝ : DecidableEq α
      x y y' : α
      h : Ne y y'
      hx : Eq x y
      ⊢ Exists fun y => Ne y x
    -/
  · rw [← hx] at h
    /-
      case pos
      α : Type u_1
      inst✝¹ : Nontrivial α
      inst✝ : DecidableEq α
      x y y' : α
      h : Ne x y'
      hx : Eq x y
      ⊢ Exists fun y => Ne y x
    -/
    exact ⟨y', h.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : Nontrivial α
      inst✝ : DecidableEq α
      x y y' : α
      h : Ne y y'
      hx : Not (Eq x y)
      ⊢ Exists fun y => Ne y x
    -/
  · exact ⟨y, Ne.symm hx⟩
    /-
      🎉 no goals
    -/


open Classical in
theorem exists_ne [Nontrivial α] (x : α) : ∃ y, y ≠ x := Decidable.exists_ne x

-- `x` and `y` are explicit here, as they are often needed to guide typechecking of `h`.

theorem nontrivial_of_ne (x y : α) (h : x ≠ y) : Nontrivial α :=
  ⟨⟨x, y, h⟩⟩


theorem nontrivial_iff_exists_ne (x : α) : Nontrivial α ↔ ∃ y, y ≠ x :=
  ⟨fun h ↦ @exists_ne α h x, fun ⟨_, hy⟩ ↦ nontrivial_of_ne _ _ hy⟩


instance : Nontrivial Prop :=
  ⟨⟨True, False, true_ne_false⟩⟩


/-- See Note [lower instance priority]

Note that since this and `instNonemptyOfInhabited` are the most "obvious" way to find a nonempty
instance if no direct instance can be found, we give this a higher priority than the usual `100`.
-/
instance (priority := 500) Nontrivial.to_nonempty [Nontrivial α] : Nonempty α :=
  let ⟨x, _⟩ := _root_.exists_pair_ne α
  ⟨x⟩


theorem subsingleton_iff : Subsingleton α ↔ ∀ x y : α, x = y :=
  ⟨by
    /-
      α : Type u_1
      ⊢ Subsingleton α → ∀ (x y : α), Eq x y
    -/
    intro h
    /-
      α : Type u_1
      h : Subsingleton α
      ⊢ ∀ (x y : α), Eq x y
    -/
    exact Subsingleton.elim, fun h ↦ ⟨h⟩⟩
    /-
      🎉 no goals
    -/


theorem not_nontrivial_iff_subsingleton : ¬Nontrivial α ↔ Subsingleton α := by
  /-
    α : Type u_1
    ⊢ Iff (Not (Nontrivial α)) (Subsingleton α)
  -/
  simp only [nontrivial_iff, subsingleton_iff, not_exists, Classical.not_not]
  /-
    🎉 no goals
  -/


theorem not_nontrivial (α) [Subsingleton α] : ¬Nontrivial α :=
  fun ⟨⟨x, y, h⟩⟩ ↦ h <| Subsingleton.elim x y


theorem not_subsingleton (α) [Nontrivial α] : ¬Subsingleton α :=
  fun _ => not_nontrivial _ ‹_›


lemma not_subsingleton_iff_nontrivial : ¬ Subsingleton α ↔ Nontrivial α := by
  /-
    α : Type u_1
    ⊢ Iff (Not (Subsingleton α)) (Nontrivial α)
  -/
  rw [← not_nontrivial_iff_subsingleton, Classical.not_not]
  /-
    🎉 no goals
  -/


/-- A type is either a subsingleton or nontrivial. -/
theorem subsingleton_or_nontrivial (α : Type*) : Subsingleton α ∨ Nontrivial α := by
  /-
    α : Type u_3
    ⊢ Or (Subsingleton α) (Nontrivial α)
  -/
  rw [← not_nontrivial_iff_subsingleton, or_comm]
  /-
    α : Type u_3
    ⊢ Or (Nontrivial α) (Not (Nontrivial α))
  -/
  exact Classical.em _
  /-
    🎉 no goals
  -/


theorem false_of_nontrivial_of_subsingleton (α : Type*) [Nontrivial α] [Subsingleton α] : False :=
  not_nontrivial _ ‹_›


/-- Pullback a `Nontrivial` instance along a surjective function. -/
protected theorem Function.Surjective.nontrivial [Nontrivial β] {f : α → β}
    (hf : Function.Surjective f) : Nontrivial α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nontrivial β
    f : α → β
    hf : Function.Surjective f
    ⊢ Nontrivial α
  -/
  rcases exists_pair_ne β with ⟨x, y, h⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : Nontrivial β
    f : α → β
    hf : Function.Surjective f
    x y : β
    h : Ne x y
    ⊢ Nontrivial α
  -/
  rcases hf x with ⟨x', hx'⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : Nontrivial β
    f : α → β
    hf : Function.Surjective f
    x y : β
    h : Ne x y
    x' : α
    hx' : Eq (f x') x
    ⊢ Nontrivial α
  -/
  rcases hf y with ⟨y', hy'⟩
  have : x' ≠ y' := by
    refine fun H ↦ h ?_
    rw [← hx', ← hy', H]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : Nontrivial β
    f : α → β
    hf : Function.Surjective f
    x y : β
    h : Ne x y
    x' : α
    hx' : Eq (f x') x
    y' : α
    hy' : Eq (f y') y
    this : Ne x' y'
    ⊢ Nontrivial α
  -/
  exact ⟨⟨x', y', this⟩⟩
  /-
    🎉 no goals
  -/


instance : Nontrivial Bool :=
  ⟨⟨true, false, nofun⟩⟩


