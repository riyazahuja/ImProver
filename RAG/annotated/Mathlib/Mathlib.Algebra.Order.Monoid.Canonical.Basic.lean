                                                                 /-
                                                                   ι : Type u_1
                                                                   α : Type u_2
                                                                   inst✝ : CanonicallyLinearOrderedAddCommMonoid α
                                                                   s : Finset ι
                                                                   f : ι → α
                                                                   ⊢ Iff (Eq (s.sup f) 0) (∀ (i : ι), Membership.mem s i → Eq (f i) 0)
                                                                 -/
@[simp] lemma sup_eq_zero : s.sup f = 0 ↔ ∀ i ∈ s, f i = 0 := by simp [← bot_eq_zero']
                                                                 /-
                                                                   🎉 no goals
                                                                 -/

                                                                           /-
                                                                             ι : Type u_1
                                                                             α : Type u_2
                                                                             inst✝ : CanonicallyLinearOrderedAddCommMonoid α
                                                                             s : Finset ι
                                                                             f : ι → α
                                                                             hs : s.Nonempty
                                                                             ⊢ Iff (Eq (s.sup' hs f) 0) (∀ (i : ι), Membership.mem s i → Eq (f i) 0)
                                                                           -/
@[simp] lemma sup'_eq_zero (hs) : s.sup' hs f = 0 ↔ ∀ i ∈ s, f i = 0 := by simp [sup'_eq_sup]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


