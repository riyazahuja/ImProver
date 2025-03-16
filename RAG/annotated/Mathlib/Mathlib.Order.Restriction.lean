/-- Restrict domain of a function `f` indexed by `α` to elements `≤ a`. -/
def restrictLe (a : α) := (Iic a).restrict (π := π)


@[simp]
lemma restrictLe_apply (a : α) (f : (a : α) → π a) (i : Iic a) : restrictLe a f i = f i := rfl


/-- If a function `f` indexed by `α` is restricted to elements `≤ π`, and `a ≤ b`,
this is the restriction to elements `≤ a`. -/
def restrictLe₂ {a b : α} (hab : a ≤ b) := Set.restrict₂ (π := π) (Iic_subset_Iic.2 hab)


@[simp]
lemma restrictLe₂_apply {a b : α} (hab : a ≤ b) (f : (i : Iic b) → π i) (i : Iic a) :
    restrictLe₂ hab f i = f ⟨i.1, Iic_subset_Iic.2 hab i.2⟩ := rfl


theorem restrictLe₂_comp_restrictLe {a b : α} (hab : a ≤ b) :
    (restrictLe₂ (π := π) hab) ∘ (restrictLe b) = restrictLe a := rfl


theorem restrictLe₂_comp_restrictLe₂ {a b c : α} (hab : a ≤ b) (hbc : b ≤ c) :
    (restrictLe₂ (π := π) hab) ∘ (restrictLe₂ hbc) = restrictLe₂ (hab.trans hbc) := rfl


/-- Restrict domain of a function `f` indexed by `α` to elements `≤ α`, seen as a finite set. -/
def frestrictLe (a : α) := (Iic a).restrict (π := π)


@[simp]
lemma frestrictLe_apply (a : α) (f : (a : α) → π a) (i : Iic a) : frestrictLe a f i = f i := rfl


/-- If a function `f` indexed by `α` is restricted to elements `≤ b`, and `a ≤ b`,
this is the restriction to elements `≤ b`. Intervals are seen as finite sets. -/
def frestrictLe₂ {a b : α} (hab : a ≤ b) := Finset.restrict₂ (π := π) (Iic_subset_Iic.2 hab)


@[simp]
lemma frestrictLe₂_apply {a b : α} (hab : a ≤ b) (f : (i : Iic b) → π i) (i : Iic a) :
    frestrictLe₂ hab f i = f ⟨i.1, Iic_subset_Iic.2 hab i.2⟩ := rfl


theorem frestrictLe₂_comp_frestrictLe {a b : α} (hab : a ≤ b) :
    (frestrictLe₂ (π := π) hab) ∘ (frestrictLe b) = frestrictLe a := rfl


theorem frestrictLe₂_comp_frestrictLe₂ {a b c : α} (hab : a ≤ b) (hbc : b ≤ c) :
    (frestrictLe₂ (π := π) hab) ∘ (frestrictLe₂ hbc) = frestrictLe₂ (hab.trans hbc) := rfl


