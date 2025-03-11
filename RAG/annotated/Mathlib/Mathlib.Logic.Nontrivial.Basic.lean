theorem nontrivial_of_lt [Preorder α] (x y : α) (h : x < y) : Nontrivial α :=
  ⟨⟨x, y, ne_of_lt h⟩⟩


theorem exists_pair_lt (α : Type*) [Nontrivial α] [LinearOrder α] : ∃ x y : α, x < y := by
  /-
    α : Type u_3
    inst✝¹ : Nontrivial α
    inst✝ : LinearOrder α
    ⊢ Exists fun x => Exists fun y => LT.lt x y
  -/
  rcases exists_pair_ne α with ⟨x, y, hxy⟩
  /-
    case intro.intro
    α : Type u_3
    inst✝¹ : Nontrivial α
    inst✝ : LinearOrder α
    x y : α
    hxy : Ne x y
    ⊢ Exists fun x => Exists fun y => LT.lt x y
  -/
                               /-
                                 🎉 no goals
                               -/
  cases lt_or_gt_of_ne hxy <;> exact ⟨_, _, ‹_›⟩
                               /-
                                 🎉 no goals
                               -/


theorem nontrivial_iff_lt [LinearOrder α] : Nontrivial α ↔ ∃ x y : α, x < y :=
  ⟨fun h ↦ @exists_pair_lt α h _, fun ⟨x, y, h⟩ ↦ nontrivial_of_lt x y h⟩


theorem Subtype.nontrivial_iff_exists_ne (p : α → Prop) (x : Subtype p) :
    Nontrivial (Subtype p) ↔ ∃ (y : α) (_ : p y), y ≠ x := by
  /-
    α : Type u_1
    p : α → Prop
    x : Subtype p
    ⊢ Iff (Nontrivial (Subtype p)) (Exists fun y => Exists fun x_1 => Ne y x.val)
  -/
  simp only [_root_.nontrivial_iff_exists_ne x, Subtype.exists, Ne, Subtype.ext_iff]
  /-
    🎉 no goals
  -/


open Classical in
/-- An inhabited type is either nontrivial, or has a unique element. -/
noncomputable def nontrivialPSumUnique (α : Type*) [Inhabited α] :
    Nontrivial α ⊕' Unique α :=
  if h : Nontrivial α then PSum.inl h
  else
    PSum.inr
      { default := default,
        uniq := fun x : α ↦ by
          /-
            α✝ : Type u_1
            β : Type u_2
            α : Type u_3
            inst✝ : Inhabited α
            h : Not (Nontrivial α)
            x : α
            ⊢ Eq x Inhabited.default
          -/
          by_contra H
          /-
            α✝ : Type u_1
            β : Type u_2
            α : Type u_3
            inst✝ : Inhabited α
            h : Not (Nontrivial α)
            x : α
            H : Not (Eq x Inhabited.default)
            ⊢ False
          -/
          exact h ⟨_, _, H⟩ }
          /-
            🎉 no goals
          -/


instance Option.nontrivial [Nonempty α] : Nontrivial (Option α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty α
    ⊢ Nontrivial (Option α)
  -/
  inhabit α
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty α
    inhabited_h : Inhabited α
    ⊢ Nontrivial (Option α)
  -/
  exact ⟨none, some default, nofun⟩
  /-
    🎉 no goals
  -/


/-- Pushforward a `Nontrivial` instance along an injective function. -/
protected theorem Function.Injective.nontrivial [Nontrivial α] {f : α → β}
    (hf : Function.Injective f) : Nontrivial β :=
  let ⟨x, y, h⟩ := exists_pair_ne α
  ⟨⟨f x, f y, hf.ne h⟩⟩


/-- An injective function from a nontrivial type has an argument at
which it does not take a given value. -/
protected theorem Function.Injective.exists_ne [Nontrivial α] {f : α → β}
    (hf : Function.Injective f) (y : β) : ∃ x, f x ≠ y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nontrivial α
    f : α → β
    hf : Function.Injective f
    y : β
    ⊢ Exists fun x => Ne (f x) y
  -/
  rcases exists_pair_ne α with ⟨x₁, x₂, hx⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : Nontrivial α
    f : α → β
    hf : Function.Injective f
    y : β
    x₁ x₂ : α
    hx : Ne x₁ x₂
    ⊢ Exists fun x => Ne (f x) y
  -/
  by_cases h : f x₂ = y
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝ : Nontrivial α
      f : α → β
      hf : Function.Injective f
      y : β
      x₁ x₂ : α
      hx : Ne x₁ x₂
      h : Eq (f x₂) y
      ⊢ Exists fun x => Ne (f x) y
    -/
  · exact ⟨x₁, (hf.ne_iff' h).2 hx⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝ : Nontrivial α
      f : α → β
      hf : Function.Injective f
      y : β
      x₁ x₂ : α
      hx : Ne x₁ x₂
      h : Not (Eq (f x₂) y)
      ⊢ Exists fun x => Ne (f x) y
    -/
  · exact ⟨x₂, h⟩
    /-
      🎉 no goals
    -/


instance nontrivial_prod_right [Nonempty α] [Nontrivial β] : Nontrivial (α × β) :=
  Prod.snd_surjective.nontrivial


instance nontrivial_prod_left [Nontrivial α] [Nonempty β] : Nontrivial (α × β) :=
  Prod.fst_surjective.nontrivial


open Classical in
/-- A pi type is nontrivial if it's nonempty everywhere and nontrivial somewhere. -/
theorem nontrivial_at (i' : I) [inst : ∀ i, Nonempty (f i)] [Nontrivial (f i')] :
    Nontrivial (∀ i : I, f i) := by
  /-
    I : Type u_3
    f : I → Type u_4
    i' : I
    inst : ∀ (i : I), Nonempty (f i)
    inst✝ : Nontrivial (f i')
    ⊢ Nontrivial ((i : I) → f i)
  -/
  letI := Classical.decEq (∀ i : I, f i)
  /-
    I : Type u_3
    f : I → Type u_4
    i' : I
    inst : ∀ (i : I), Nonempty (f i)
    inst✝ : Nontrivial (f i')
    this : DecidableEq ((i : I) → f i) := Classical.decEq ((i : I) → f i)
    ⊢ Nontrivial ((i : I) → f i)
  -/
  exact (Function.update_injective (fun i ↦ Classical.choice (inst i)) i').nontrivial
  /-
    🎉 no goals
  -/


/-- As a convenience, provide an instance automatically if `(f default)` is nontrivial.

If a different index has the non-trivial type, then use `haveI := nontrivial_at that_index`.
-/
instance nontrivial [Inhabited I] [∀ i, Nonempty (f i)] [Nontrivial (f default)] :
    Nontrivial (∀ i : I, f i) :=
  nontrivial_at default


instance Function.nontrivial [h : Nonempty α] [Nontrivial β] : Nontrivial (α → β) :=
  h.elim fun a ↦ Pi.nontrivial_at a


@[nontriviality]
protected theorem Subsingleton.le [Preorder α] [Subsingleton α] (x y : α) : x ≤ y :=
  le_of_eq (Subsingleton.elim x y)

