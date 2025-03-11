/--
Complete partial orders are partial orders where every directed set has a least upper bound.
-/
class CompletePartialOrder (α : Type*) extends PartialOrder α, SupSet α where
  /-- For each directed set `d`, `sSup d` is the least upper bound of `d`. -/
  lubOfDirected : ∀ d, DirectedOn (· ≤ ·) d → IsLUB d (sSup d)


protected lemma DirectedOn.isLUB_sSup : DirectedOn (· ≤ ·) d → IsLUB d (sSup d) :=
CompletePartialOrder.lubOfDirected _


protected lemma DirectedOn.le_sSup (hd : DirectedOn (· ≤ ·) d) (ha : a ∈ d) : a ≤ sSup d :=
hd.isLUB_sSup.1 ha


protected lemma DirectedOn.sSup_le (hd : DirectedOn (· ≤ ·) d) (ha : ∀ b ∈ d, b ≤ a) : sSup d ≤ a :=
hd.isLUB_sSup.2 ha


protected lemma Directed.le_iSup (hf : Directed (· ≤ ·) f) (i : ι) : f i ≤ ⨆ j, f j :=
hf.directedOn_range.le_sSup <| Set.mem_range_self _


protected lemma Directed.iSup_le (hf : Directed (· ≤ ·) f) (ha : ∀ i, f i ≤ a) :  ⨆ i, f i ≤ a :=
hf.directedOn_range.sSup_le <| Set.forall_mem_range.2 ha

--TODO: We could mimic more `sSup`/`iSup` lemmas


/-- Scott-continuity takes on a simpler form in complete partial orders. -/
lemma CompletePartialOrder.scottContinuous {f : α → β} :
    ScottContinuous f ↔
    ∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (· ≤ ·) d → IsLUB (f '' d) (f (sSup d)) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompletePartialOrder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (ScottContinuous f) (∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (fun x1 x2 = …
  -/
  refine ⟨fun h d hd₁ hd₂ ↦ h hd₁ hd₂ hd₂.isLUB_sSup, fun h d hne hd a hda ↦ ?_⟩
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompletePartialOrder α
    inst✝ : Preorder β
    f : α → β
    h : ∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) d → IsLU …
    d : Set α
    hne : d.Nonempty
    hd : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    a : α
    hda : IsLUB d a
    ⊢ IsLUB (Set.image f d) (f a)
  -/
  rw [hda.unique hd.isLUB_sSup]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompletePartialOrder α
    inst✝ : Preorder β
    f : α → β
    h : ∀ ⦃d : Set α⦄, d.Nonempty → DirectedOn (fun x1 x2 => LE.le x1 x2) d → IsLU …
    d : Set α
    hne : d.Nonempty
    hd : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    a : α
    hda : IsLUB d a
    ⊢ IsLUB (Set.image f d) (f (SupSet.sSup d))
  -/
  exact h hne hd
  /-
    🎉 no goals
  -/


/-- A complete partial order is an ω-complete partial order. -/
instance CompletePartialOrder.toOmegaCompletePartialOrder : OmegaCompletePartialOrder α where
  ωSup c := ⨆ n, c n
  le_ωSup c := c.directed.le_iSup
  ωSup_le c _ := c.directed.iSup_le


/-- A complete lattice is a complete partial order. -/
instance CompleteLattice.toCompletePartialOrder [CompleteLattice α] : CompletePartialOrder α where
  sSup := sSup
  lubOfDirected _ _ := isLUB_sSup _

