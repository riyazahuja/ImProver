/-- A variant of `List.prod_pos` for `CanonicallyOrderedCommSemiring`. -/
@[simp] lemma CanonicallyOrderedCommSemiring.list_prod_pos
    {α : Type*} [CanonicallyOrderedCommSemiring α] [Nontrivial α] :
    ∀ {l : List α}, 0 < l.prod ↔ (∀ x ∈ l, (0 : α) < x)
             /-
               α : Type u_2
               inst✝¹ : CanonicallyOrderedCommSemiring α
               inst✝ : Nontrivial α
               ⊢ Iff (LT.lt 0 List.nil.prod) (∀ (x : α), Membership.mem List.nil x → LT.lt 0 x)
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | (x :: xs) => by simp_rw [List.prod_cons, List.forall_mem_cons,
      CanonicallyOrderedCommSemiring.mul_pos, list_prod_pos]

