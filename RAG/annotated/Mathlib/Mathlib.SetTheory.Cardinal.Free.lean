@[to_additive]
instance [Nonempty α] : Infinite (FreeMonoid α) := inferInstanceAs <| Infinite (List α)


@[to_additive]
instance [Nonempty α] : Infinite (FreeGroup α) := by
  classical
  exact Infinite.of_surjective FreeGroup.norm FreeGroup.norm_surjective


instance [Nonempty α] : Infinite (FreeAbelianGroup α) :=
  (FreeAbelianGroup.equivFinsupp α).toEquiv.infinite_iff.2 inferInstance


                                       /-
                                         α : Type u
                                         ⊢ Infinite (FreeRing α)
                                       -/
instance : Infinite (FreeRing α) := by unfold FreeRing; infer_instance
                                                        /-
                                                          🎉 no goals
                                                        -/


                                           /-
                                             α : Type u
                                             ⊢ Infinite (FreeCommRing α)
                                           -/
instance : Infinite (FreeCommRing α) := by unfold FreeCommRing; infer_instance
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem mk_abelianization_le (G : Type u) [Group G] :
    #(Abelianization G) ≤ #G := Cardinal.mk_le_of_surjective Quotient.mk_surjective


@[to_additive (attr := simp)]
theorem mk_freeMonoid [Nonempty α] : #(FreeMonoid α) = max #α ℵ₀ :=
    Cardinal.mk_list_eq_max_mk_aleph0 _


@[to_additive (attr := simp)]
theorem mk_freeGroup [Nonempty α] : #(FreeGroup α) = max #α ℵ₀ := by
  classical
  apply le_antisymm
  · apply (mk_le_of_injective (FreeGroup.toWord_injective (α := α))).trans_eq
    simp [Cardinal.mk_list_eq_max_mk_aleph0]
    obtain hα | hα := lt_or_le #α ℵ₀
    · simp only [hα.le, max_eq_right, max_eq_right_iff]
      exact (mul_lt_aleph0 hα (nat_lt_aleph0 2)).le
    · rw [max_eq_left hα, max_eq_left (hα.trans <| Cardinal.le_mul_right two_ne_zero),
        Cardinal.mul_eq_left hα _ (by simp)]
      exact (nat_lt_aleph0 2).le.trans hα
  · apply max_le
    · exact mk_le_of_injective FreeGroup.of_injective
    · simp


@[simp]
theorem mk_freeAbelianGroup [Nonempty α] : #(FreeAbelianGroup α) = max #α ℵ₀ := by
  /-
    α : Type u
    inst✝ : Nonempty α
    ⊢ Eq (Cardinal.mk (FreeAbelianGroup α)) (Max.max (Cardinal.mk α) Cardinal.alep …
  -/
  rw [Cardinal.mk_congr (FreeAbelianGroup.equivFinsupp α).toEquiv]
  /-
    α : Type u
    inst✝ : Nonempty α
    ⊢ Eq (Cardinal.mk (Finsupp α Int)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_freeRing : #(FreeRing α) = max #α ℵ₀ := by
  /-
    α : Type u
    ⊢ Eq (Cardinal.mk (FreeRing α)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases isEmpty_or_nonempty α <;> simp [FreeRing]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem mk_freeCommRing : #(FreeCommRing α) = max #α ℵ₀ := by
  /-
    α : Type u
    ⊢ Eq (Cardinal.mk (FreeCommRing α)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases isEmpty_or_nonempty α <;> simp [FreeCommRing]
                                  /-
                                    🎉 no goals
                                  -/


/-- A commutative ring can be constructed on any non-empty type.

See also `Infinite.nonempty_field`. -/
instance nonempty_commRing [Nonempty α] : Nonempty (CommRing α) := by
  /-
    α : Type u
    inst✝ : Nonempty α
    ⊢ Nonempty (CommRing α)
  -/
  obtain hR | hR := finite_or_infinite α
    /-
      case inl
      α : Type u
      inst✝ : Nonempty α
      hR : Finite α
      ⊢ Nonempty (CommRing α)
    -/
  · obtain ⟨x⟩ := nonempty_fintype α
    /-
      case inl.intro
      α : Type u
      inst✝ : Nonempty α
      hR : Finite α
      x : Fintype α
      ⊢ Nonempty (CommRing α)
    -/
    have : NeZero (Fintype.card α) := ⟨by inhabit α; simp⟩
    classical
    obtain ⟨e⟩ := Fintype.truncEquivFin α
    exact ⟨e.commRing⟩
    /-
      case inr
      α : Type u
      inst✝ : Nonempty α
      hR : Infinite α
      ⊢ Nonempty (CommRing α)
    -/
  · have ⟨e⟩ : Nonempty (α ≃ FreeCommRing α) := by simp [← Cardinal.eq]
    /-
      case inr
      α : Type u
      inst✝ : Nonempty α
      hR : Infinite α
      e : Equiv α (FreeCommRing α)
      ⊢ Nonempty (CommRing α)
    -/
    exact ⟨e.commRing⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem nonempty_commRing_iff : Nonempty (CommRing α) ↔ Nonempty α :=
  ⟨Nonempty.map (·.zero), fun _ => nonempty_commRing _⟩


@[simp]
theorem nonempty_ring_iff : Nonempty (Ring α) ↔ Nonempty α :=
  ⟨Nonempty.map (·.zero), fun _ => (nonempty_commRing _).map (·.toRing)⟩


@[simp]
theorem nonempty_commSemiring_iff : Nonempty (CommSemiring α) ↔ Nonempty α :=
  ⟨Nonempty.map (·.zero), fun _ => (nonempty_commRing _).map (·.toCommSemiring)⟩


@[simp]
theorem nonempty_semiring_iff : Nonempty (Semiring α) ↔ Nonempty α :=
  ⟨Nonempty.map (·.zero), fun _ => (nonempty_commRing _).map (·.toSemiring)⟩


