@[simp]
theorem mk_finsupp_lift_of_fintype (α : Type u) (β : Type v) [Fintype α] [Zero β] :
    #(α →₀ β) = lift.{u} #β ^ Fintype.card α := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Fintype α
    inst✝ : Zero β
    ⊢ Eq (Cardinal.mk (Finsupp α β)) (HPow.hPow (Cardinal.lift.{u, v} (Cardinal.mk …
  -/
  simpa using (@Finsupp.equivFunOnFinite α β _ _).cardinal_eq
  /-
    🎉 no goals
  -/


theorem mk_finsupp_of_fintype (α β : Type u) [Fintype α] [Zero β] :
                                          /-
                                            α β : Type u
                                            inst✝¹ : Fintype α
                                            inst✝ : Zero β
                                            ⊢ Eq (Cardinal.mk (Finsupp α β)) (HPow.hPow (Cardinal.mk β) (Fintype.card α))
                                          -/
    #(α →₀ β) = #β ^ Fintype.card α := by simp
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem mk_finsupp_lift_of_infinite (α : Type u) (β : Type v) [Infinite α] [Zero β] [Nontrivial β] :
    #(α →₀ β) = max (lift.{v} #α) (lift.{u} #β) := by
  /-
    α : Type u
    β : Type v
    inst✝² : Infinite α
    inst✝¹ : Zero β
    inst✝ : Nontrivial β
    ⊢ Eq (Cardinal.mk (Finsupp α β)) (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α …
  -/
  apply le_antisymm
  · calc
      #(α →₀ β) ≤ #(Finset (α × β)) := mk_le_of_injective (Finsupp.graph_injective α β)
      _ = #(α × β) := mk_finset_of_infinite _
      _ = max (lift.{v} #α) (lift.{u} #β) := by
        rw [mk_prod, mul_eq_max_of_aleph0_le_left] <;> simp

    /-
      case a
      α : Type u
      β : Type v
      inst✝² : Infinite α
      inst✝¹ : Zero β
      inst✝ : Nontrivial β
      ⊢ LE.le (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v}  …
    -/
  · apply max_le <;> rw [← lift_id #(α →₀ β), ← lift_umax]
      /-
        case a.h₁
        α : Type u
        β : Type v
        inst✝² : Infinite α
        inst✝¹ : Zero β
        inst✝ : Nontrivial β
        ⊢ LE.le (Cardinal.lift.{max u v, u} (Cardinal.mk α)) (Cardinal.lift.{max v u,  …
      -/
    · cases' exists_ne (0 : β) with b hb
      /-
        case a.h₁.intro
        α : Type u
        β : Type v
        inst✝² : Infinite α
        inst✝¹ : Zero β
        inst✝ : Nontrivial β
        b : β
        hb : Ne b 0
        ⊢ LE.le (Cardinal.lift.{max u v, u} (Cardinal.mk α)) (Cardinal.lift.{max v u,  …
      -/
      exact lift_mk_le.{v}.2 ⟨⟨_, Finsupp.single_left_injective hb⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case a.h₂
        α : Type u
        β : Type v
        inst✝² : Infinite α
        inst✝¹ : Zero β
        inst✝ : Nontrivial β
        ⊢ LE.le (Cardinal.lift.{max v u, v} (Cardinal.mk β)) (Cardinal.lift.{max v u,  …
      -/
    · inhabit α
      /-
        case a.h₂
        α : Type u
        β : Type v
        inst✝² : Infinite α
        inst✝¹ : Zero β
        inst✝ : Nontrivial β
        inhabited_h : Inhabited α
        ⊢ LE.le (Cardinal.lift.{max v u, v} (Cardinal.mk β)) (Cardinal.lift.{max v u,  …
      -/
      exact lift_mk_le.{u}.2 ⟨⟨_, Finsupp.single_injective default⟩⟩
      /-
        🎉 no goals
      -/


theorem mk_finsupp_of_infinite (α β : Type u) [Infinite α] [Zero β] [Nontrivial β] :
                                /-
                                  α β : Type u
                                  inst✝² : Infinite α
                                  inst✝¹ : Zero β
                                  inst✝ : Nontrivial β
                                  ⊢ Eq (Cardinal.mk (Finsupp α β)) (Max.max (Cardinal.mk α) (Cardinal.mk β))
                                -/
    #(α →₀ β) = max #α #β := by simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem mk_finsupp_lift_of_infinite' (α : Type u) (β : Type v) [Nonempty α] [Zero β] [Infinite β] :
    #(α →₀ β) = max (lift.{v} #α) (lift.{u} #β) := by
  /-
    α : Type u
    β : Type v
    inst✝² : Nonempty α
    inst✝¹ : Zero β
    inst✝ : Infinite β
    ⊢ Eq (Cardinal.mk (Finsupp α β)) (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α …
  -/
  cases fintypeOrInfinite α
    /-
      case inl
      α : Type u
      β : Type v
      inst✝² : Nonempty α
      inst✝¹ : Zero β
      inst✝ : Infinite β
      val✝ : Fintype α
      ⊢ Eq (Cardinal.mk (Finsupp α β)) (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α …
    -/
  · rw [mk_finsupp_lift_of_fintype]
    /-
      case inl
      α : Type u
      β : Type v
      inst✝² : Nonempty α
      inst✝¹ : Zero β
      inst✝ : Infinite β
      val✝ : Fintype α
      ⊢ Eq (HPow.hPow (Cardinal.lift.{u, v} (Cardinal.mk β)) (Fintype.card α)) (Max. …
    -/
    have : ℵ₀ ≤ (#β).lift := aleph0_le_lift.2 (aleph0_le_mk β)
    /-
      case inl
      α : Type u
      β : Type v
      inst✝² : Nonempty α
      inst✝¹ : Zero β
      inst✝ : Infinite β
      val✝ : Fintype α
      this : LE.le Cardinal.aleph0 (Cardinal.lift.{?u.16722, v} (Cardinal.mk β))
      ⊢ Eq (HPow.hPow (Cardinal.lift.{u, v} (Cardinal.mk β)) (Fintype.card α)) (Max. …
    -/
    rw [max_eq_right (le_trans _ this), power_nat_eq this]
    /-
      case inl
      α : Type u
      β : Type v
      inst✝² : Nonempty α
      inst✝¹ : Zero β
      inst✝ : Infinite β
      val✝ : Fintype α
      this : LE.le Cardinal.aleph0 (Cardinal.lift.{u, v} (Cardinal.mk β))
      ⊢ LE.le 1 (Fintype.card α)
    -/
    exacts [Fintype.card_pos, lift_le_aleph0.2 (lt_aleph0_of_finite _).le]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      β : Type v
      inst✝² : Nonempty α
      inst✝¹ : Zero β
      inst✝ : Infinite β
      val✝ : Infinite α
      ⊢ Eq (Cardinal.mk (Finsupp α β)) (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α …
    -/
  · apply mk_finsupp_lift_of_infinite
    /-
      🎉 no goals
    -/


theorem mk_finsupp_of_infinite' (α β : Type u) [Nonempty α] [Zero β] [Infinite β] :
                                /-
                                  α β : Type u
                                  inst✝² : Nonempty α
                                  inst✝¹ : Zero β
                                  inst✝ : Infinite β
                                  ⊢ Eq (Cardinal.mk (Finsupp α β)) (Max.max (Cardinal.mk α) (Cardinal.mk β))
                                -/
    #(α →₀ β) = max #α #β := by simp
                                /-
                                  🎉 no goals
                                -/


                                                                               /-
                                                                                 α : Type u
                                                                                 inst✝ : Nonempty α
                                                                                 ⊢ Eq (Cardinal.mk (Finsupp α Nat)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
                                                                               -/
theorem mk_finsupp_nat (α : Type u) [Nonempty α] : #(α →₀ ℕ) = max #α ℵ₀ := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem mk_multiset_of_isEmpty (α : Type u) [IsEmpty α] : #(Multiset α) = 1 :=
                                                   /-
                                                     α : Type u
                                                     inst✝ : IsEmpty α
                                                     ⊢ Eq (Cardinal.mk (Finsupp α Nat)) 1
                                                   -/
  Multiset.toFinsupp.toEquiv.cardinal_eq.trans (by simp)
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem mk_multiset_of_nonempty (α : Type u) [Nonempty α] : #(Multiset α) = max #α ℵ₀ := by
  classical
  exact Multiset.toFinsupp.toEquiv.cardinal_eq.trans (mk_finsupp_nat α)


                                                                                     /-
                                                                                       α : Type u
                                                                                       inst✝ : Infinite α
                                                                                       ⊢ Eq (Cardinal.mk (Multiset α)) (Cardinal.mk α)
                                                                                     -/
theorem mk_multiset_of_infinite (α : Type u) [Infinite α] : #(Multiset α) = #α := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem mk_multiset_of_countable (α : Type u) [Countable α] [Nonempty α] : #(Multiset α) = ℵ₀ := by
  classical
  exact Multiset.toFinsupp.toEquiv.cardinal_eq.trans (by simp)


