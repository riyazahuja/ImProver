/-- The `Finset` of `l : List α` that, given `m : Multiset α`, have the property `⟦l⟧ = m`.
-/
def lists : Multiset α → Finset (List α) := fun s =>
  Quotient.liftOn s (fun l => l.permutations.toFinset) fun l l' (h : l ~ l') => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      l l' : List α
      h : l.Perm l'
      ⊢ Eq ((fun l => l.permutations.toFinset) l) ((fun l => l.permutations.toFinset …
    -/
    ext sl
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      l l' : List α
      h : l.Perm l'
      sl : List α
      ⊢ Iff (Membership.mem ((fun l => l.permutations.toFinset) l) sl) (Membership.m …
    -/
    simp only [mem_permutations, List.mem_toFinset]
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      l l' : List α
      h : l.Perm l'
      sl : List α
      ⊢ Iff (sl.Perm l) (sl.Perm l')
    -/
    exact ⟨fun hs => hs.trans h, fun hs => hs.trans h.symm⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem lists_coe (l : List α) : lists (l : Multiset α) = l.permutations.toFinset :=
  rfl


@[simp]
theorem mem_lists_iff (s : Multiset α) (l : List α) : l ∈ lists s ↔ s = ⟦l⟧ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    l : List α
    ⊢ Iff (Membership.mem s.lists l) (Eq s (Quotient.mk (List.isSetoid α) l))
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    l a✝ : List α
    ⊢ Iff (Membership.mem (Multiset.lists (Quotient.mk (List.isSetoid α) a✝)) l) ( …
  -/
  simpa using perm_comm
  /-
    🎉 no goals
  -/


instance fintypeNodupList [Fintype α] : Fintype { l : List α // l.Nodup } :=
  Fintype.subtype ((Finset.univ : Finset α).powerset.biUnion fun s => s.val.lists) fun l => by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      l : List α
      ⊢ Iff (Membership.mem (Finset.univ.powerset.biUnion fun s => s.val.lists) l) l …
    -/
    suffices (∃ a : Finset α, a.val = ↑l) ↔ l.Nodup by simpa
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      l : List α
      ⊢ Iff (Exists fun a => Eq a.val ↑l) l.Nodup
    -/
    constructor
      /-
        case mp
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        l : List α
        ⊢ (Exists fun a => Eq a.val ↑l) → l.Nodup
      -/
    · rintro ⟨s, hs⟩
      /-
        case mp.intro
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        l : List α
        s : Finset α
        hs : Eq s.val ↑l
        ⊢ l.Nodup
      -/
      simpa [← Multiset.coe_nodup, ← hs] using s.nodup
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        l : List α
        ⊢ l.Nodup → Exists fun a => Eq a.val ↑l
      -/
    · intro hl
      /-
        case mpr
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        l : List α
        hl : l.Nodup
        ⊢ Exists fun a => Eq a.val ↑l
      -/
      refine ⟨⟨↑l, hl⟩, ?_⟩
      /-
        case mpr
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        l : List α
        hl : l.Nodup
        ⊢ Eq { val := ↑l, nodup := hl }.val ↑l
      -/
      simp
      /-
        🎉 no goals
      -/

