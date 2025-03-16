/-- A small category with products is a thin category.

in Lean, a preorder category is one where the morphisms are in Prop, which is weaker than the usual
notion of a preorder/thin category which says that each homset is subsingleton; we show the latter
rather than providing a `Preorder C` instance.
-/
instance (priority := 100) : Quiver.IsThin C := fun X Y =>
  ⟨fun r s => by
    classical
      by_contra r_ne_s
      have z : (2 : Cardinal) ≤ #(X ⟶ Y) := by
        rw [Cardinal.two_le_iff]
        exact ⟨_, _, r_ne_s⟩
      let md := ΣZ W : C, Z ⟶ W
      let α := #md
      apply not_le_of_lt (Cardinal.cantor α)
      let yp : C := ∏ᶜ fun _ : md => Y
      apply _root_.trans _ _
      · exact #(X ⟶ yp)
      · apply le_trans (Cardinal.power_le_power_right z)
        rw [Cardinal.power_def]
        apply le_of_eq
        rw [Cardinal.eq]
        refine ⟨⟨Pi.lift, fun f k => f ≫ Pi.π _ k, ?_, ?_⟩⟩
        · intro f
          ext k
          simp [yp]
        · intro f
          ext ⟨j⟩
          simp [yp]
      · apply Cardinal.mk_le_of_injective _
        · intro f
          exact ⟨_, _, f⟩
        · rintro f g k
          cases k
          rfl⟩


