/--
The infimum of all coatoms.

This notion specializes, e.g. in the subgroup lattice of a group to the Frattini subgroup,
or in the lattices of ideals in a ring `R` to the Jacobson ideal.
-/
def Order.radical (α : Type*) [Preorder α] [OrderTop α] [InfSet α] : α :=
   ⨅ a ∈ {H | IsCoatom H}, a


lemma Order.radical_le_coatom {a : α} (h : IsCoatom a) : radical α ≤ a := biInf_le _ h


theorem OrderIso.map_radical (f : α ≃o β) : f (Order.radical α) = Order.radical β := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    β : Type u_2
    inst✝ : CompleteLattice β
    f : OrderIso α β
    ⊢ Eq (f (Order.radical α)) (Order.radical β)
  -/
  unfold Order.radical
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    β : Type u_2
    inst✝ : CompleteLattice β
    f : OrderIso α β
    ⊢ Eq (f (iInf fun a => iInf fun h => a)) (iInf fun a => iInf fun h => a)
  -/
  simp only [OrderIso.map_iInf]
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    β : Type u_2
    inst✝ : CompleteLattice β
    f : OrderIso α β
    ⊢ Eq (iInf fun i => iInf fun i_1 => f i) (iInf fun a => iInf fun h => a)
  -/
  fapply Equiv.iInf_congr
    /-
      case e
      α : Type u_1
      inst✝¹ : CompleteLattice α
      β : Type u_2
      inst✝ : CompleteLattice β
      f : OrderIso α β
      ⊢ Equiv α β
    -/
  · exact f.toEquiv
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      inst✝¹ : CompleteLattice α
      β : Type u_2
      inst✝ : CompleteLattice β
      f : OrderIso α β
      ⊢ ∀ (x : α), Eq (iInf fun h => f.toEquiv x) (iInf fun i => f x)
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem Order.radical_nongenerating [IsCoatomic α] {a : α} (h : a ⊔ radical α = ⊤) : a = ⊤ := by
  -- Since the lattice is coatomic, either `a` is already the top element,
  -- or there is a coatom above it.
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    inst✝ : IsCoatomic α
    a : α
    h : Eq (Max.max a (Order.radical α)) Top.top
    ⊢ Eq a Top.top
  -/
  obtain (rfl | w) := eq_top_or_exists_le_coatom a
  · -- In the first case, we're done, this was already the goal.
    /-
      case inl
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : IsCoatomic α
      h : Eq (Max.max Top.top (Order.radical α)) Top.top
      ⊢ Eq Top.top Top.top
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : IsCoatomic α
      a : α
      h : Eq (Max.max a (Order.radical α)) Top.top
      w : Exists fun a_1 => And (IsCoatom a_1) (LE.le a a_1)
      ⊢ Eq a Top.top
    -/
  · obtain ⟨m, c, le⟩ := w
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : IsCoatomic α
      a : α
      h : Eq (Max.max a (Order.radical α)) Top.top
      m : α
      c : IsCoatom m
      le : LE.le a m
      ⊢ Eq a Top.top
    -/
    have q : a ⊔ radical α ≤ m := sup_le le (radical_le_coatom c)
    -- Now note that `a ⊔ radical α ≤ m` since both `a ≤ m` and `radical α ≤ m`.
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : IsCoatomic α
      a : α
      h : Eq (Max.max a (Order.radical α)) Top.top
      m : α
      c : IsCoatom m
      le : LE.le a m
      q : LE.le (Max.max a (Order.radical α)) m
      ⊢ Eq a Top.top
    -/
    rw [h, top_le_iff] at q
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : IsCoatomic α
      a : α
      h : Eq (Max.max a (Order.radical α)) Top.top
      m : α
      c : IsCoatom m
      le : LE.le a m
      q : Eq m Top.top
      ⊢ Eq a Top.top
    -/
    simpa using c.1 q
    /-
      🎉 no goals
    -/

