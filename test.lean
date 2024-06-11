theorem A (p q : Prop) : p ∧ q → q ∧ p := by
  intro hpq
  exact ⟨hpq.right, hpq.left⟩
