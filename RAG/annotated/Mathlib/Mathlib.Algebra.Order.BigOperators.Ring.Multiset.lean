@[simp]
lemma CanonicallyOrderedCommSemiring.multiset_prod_pos {R : Type*}
    [CanonicallyOrderedCommSemiring R] [Nontrivial R] {m : Multiset R} :
    0 < m.prod ↔ ∀ x ∈ m, 0 < x := by
  /-
    R : Type u_1
    inst✝¹ : CanonicallyOrderedCommSemiring R
    inst✝ : Nontrivial R
    m : Multiset R
    ⊢ Iff (LT.lt 0 m.prod) (∀ (x : R), Membership.mem m x → LT.lt 0 x)
  -/
  rcases m with ⟨l⟩
  /-
    case mk
    R : Type u_1
    inst✝¹ : CanonicallyOrderedCommSemiring R
    inst✝ : Nontrivial R
    m : Multiset R
    l : List R
    ⊢ Iff (LT.lt 0 (Multiset.prod (Quot.mk (⇑(List.isSetoid R)) l))) (∀ (x : R), M …
  -/
  rw [Multiset.quot_mk_to_coe'', Multiset.prod_coe]
  /-
    case mk
    R : Type u_1
    inst✝¹ : CanonicallyOrderedCommSemiring R
    inst✝ : Nontrivial R
    m : Multiset R
    l : List R
    ⊢ Iff (LT.lt 0 l.prod) (∀ (x : R), Membership.mem (↑l) x → LT.lt 0 x)
  -/
  exact CanonicallyOrderedCommSemiring.list_prod_pos
  /-
    🎉 no goals
  -/

