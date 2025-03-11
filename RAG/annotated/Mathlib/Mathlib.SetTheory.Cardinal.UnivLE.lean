theorem univLE_iff_cardinal_le : UnivLE.{u, v} ↔ univ.{u, v+1} ≤ univ.{v, u+1} := by
  /-
    ⊢ Iff UnivLE.{u, v} (LE.le Cardinal.univ.{u, v + 1} Cardinal.univ.{v, u + 1})
  -/
  rw [← not_iff_not, UnivLE]; simp_rw [small_iff_lift_mk_lt_univ]; push_neg
  -- strange: simp_rw [univ_umax.{v,u}] doesn't work
  /-
    ⊢ Iff (Exists fun α => LE.le Cardinal.univ.{v, max u (v + 1)} (Cardinal.lift.{ …
  -/
  refine ⟨fun ⟨α, le⟩ ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      x✝ : Exists fun α => LE.le Cardinal.univ.{v, max u (v + 1)} (Cardinal.lift.{v  …
      α : Type u
      le : LE.le Cardinal.univ.{v, max u (v + 1)} (Cardinal.lift.{v + 1, u} (Cardina …
      ⊢ LT.lt Cardinal.univ.{v, u + 1} Cardinal.univ.{u, v + 1}
    -/
  · rw [univ_umax.{v,u}, ← lift_le.{u+1}, lift_univ, lift_lift] at le
    /-
      case refine_1
      x✝ : Exists fun α => LE.le Cardinal.univ.{v, max u (v + 1)} (Cardinal.lift.{v  …
      α : Type u
      le : LE.le Cardinal.univ.{v, u + 1} (Cardinal.lift.{max (v + 1) (u + 1), u} (C …
      ⊢ LT.lt Cardinal.univ.{v, u + 1} Cardinal.univ.{u, v + 1}
    -/
    exact le.trans_lt (lift_lt_univ'.{u,v+1} #α)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      h : LT.lt Cardinal.univ.{v, u + 1} Cardinal.univ.{u, v + 1}
      ⊢ Exists fun α => LE.le Cardinal.univ.{v, max u (v + 1)} (Cardinal.lift.{v + 1 …
    -/
  · obtain ⟨⟨α⟩, h⟩ := lt_univ'.mp h; use α
    /-
      case h
      h✝ : LT.lt Cardinal.univ.{v, u + 1} Cardinal.univ.{u, v + 1}
      w✝ : Cardinal.{u}
      α : Type u
      h : Eq Cardinal.univ.{v, u + 1} (Cardinal.lift.{max (u + 1) (v + 1), u} (Quot. …
      ⊢ LE.le Cardinal.univ.{v, max u (v + 1)} (Cardinal.lift.{v + 1, u} (Cardinal.m …
    -/
    rw [univ_umax.{v,u}, ← lift_le.{u+1}, lift_univ, lift_lift]
    /-
      case h
      h✝ : LT.lt Cardinal.univ.{v, u + 1} Cardinal.univ.{u, v + 1}
      w✝ : Cardinal.{u}
      α : Type u
      h : Eq Cardinal.univ.{v, u + 1} (Cardinal.lift.{max (u + 1) (v + 1), u} (Quot. …
      ⊢ LE.le Cardinal.univ.{v, u + 1} (Cardinal.lift.{max (v + 1) (u + 1), u} (Card …
    -/
    exact h.le
    /-
      🎉 no goals
    -/


theorem univLE_iff_exists_embedding : UnivLE.{u, v} ↔ Nonempty (Ordinal.{u} ↪ Ordinal.{v}) := by
  /-
    ⊢ Iff UnivLE.{u, v} (Nonempty (Function.Embedding Ordinal.{u} Ordinal.{v}))
  -/
  rw [univLE_iff_cardinal_le]
  /-
    ⊢ Iff (LE.le Cardinal.univ.{u, v + 1} Cardinal.univ.{v, u + 1}) (Nonempty (Fun …
  -/
  exact lift_mk_le'
  /-
    🎉 no goals
  -/


theorem Ordinal.univLE_of_injective {f : Ordinal.{u} → Ordinal.{v}} (h : f.Injective) :
    UnivLE.{u, v} :=
  univLE_iff_exists_embedding.2 ⟨f, h⟩


/-- Together with transitivity, this shows UnivLE "IsTotalPreorder". -/
theorem univLE_total : UnivLE.{u, v} ∨ UnivLE.{v, u} := by
  /-
    ⊢ Or UnivLE.{u, v} UnivLE.{v, u}
  -/
  simp_rw [univLE_iff_cardinal_le]; apply le_total
                                    /-
                                      🎉 no goals
                                    -/

