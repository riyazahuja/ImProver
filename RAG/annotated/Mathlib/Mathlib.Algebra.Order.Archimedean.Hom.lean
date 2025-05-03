/-- There is at most one ordered ring homomorphism from a linear ordered field to an archimedean
linear ordered field. -/
instance OrderRingHom.subsingleton [LinearOrderedField α] [LinearOrderedField β] [Archimedean β] :
    Subsingleton (α →+*o β) :=
  ⟨fun f g => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean β
      f g : OrderRingHom α β
      ⊢ Eq f g
    -/
    ext x
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean β
      f g : OrderRingHom α β
      x : α
      ⊢ Eq (f x) (g x)
    -/
    by_contra! h' : f x ≠ g x
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean β
      f g : OrderRingHom α β
      x : α
      h' : Ne (f x) (g x)
      ⊢ False
    -/
    wlog h : f x < g x with h₂
      /-
        case h.inr
        α : Type u_1
        β : Type u_2
        inst✝² : LinearOrderedField α
        inst✝¹ : LinearOrderedField β
        inst✝ : Archimedean β
        f g : OrderRingHom α β
        x : α
        h' : Ne (f x) (g x)
        h₂ : ∀ {α : Type u_1} {β : Type u_2} [inst : LinearOrderedField α] [inst_1 : L …
        h : Not (LT.lt (f x) (g x))
        ⊢ False
      -/
    · exact h₂ g f x (Ne.symm h') (h'.lt_or_lt.resolve_left h)
      /-
        🎉 no goals
      -/
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean β
      f g : OrderRingHom α β
      x : α
      h' : Ne (f x) (g x)
      h : LT.lt (f x) (g x)
      ⊢ False
    -/
    obtain ⟨q, hf, hg⟩ := exists_rat_btwn h
    /-
      case intro.intro
      α✝ : Type u_1
      β✝ : Type u_2
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean β
      f g : OrderRingHom α β
      x : α
      h' : Ne (f x) (g x)
      h : LT.lt (f x) (g x)
      q : Rat
      hf : LT.lt (f x) ↑q
      hg : LT.lt (↑q) (g x)
      ⊢ False
    -/
    rw [← map_ratCast f] at hf
    /-
      case intro.intro
      α✝ : Type u_1
      β✝ : Type u_2
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean β
      f g : OrderRingHom α β
      x : α
      h' : Ne (f x) (g x)
      h : LT.lt (f x) (g x)
      q : Rat
      hf : LT.lt (f x) (f ↑q)
      hg : LT.lt (↑q) (g x)
      ⊢ False
    -/
    rw [← map_ratCast g] at hg
    exact
      (lt_asymm ((OrderHomClass.mono g).reflect_lt hg) <|
          (OrderHomClass.mono f).reflect_lt hf).elim⟩


/-- There is at most one ordered ring isomorphism between a linear ordered field and an archimedean
linear ordered field. -/
instance OrderRingIso.subsingleton_right [LinearOrderedField α] [LinearOrderedField β]
    [Archimedean β] : Subsingleton (α ≃+*o β) :=
  OrderRingIso.toOrderRingHom_injective.subsingleton


/-- There is at most one ordered ring isomorphism between an archimedean linear ordered field and a
linear ordered field. -/
instance OrderRingIso.subsingleton_left [LinearOrderedField α] [Archimedean α]
    [LinearOrderedField β] : Subsingleton (α ≃+*o β) :=
  OrderRingIso.symm_bijective.injective.subsingleton

