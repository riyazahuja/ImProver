/-- `leftDual` maps any set `J` of elements of type `α` to the set `{b : β | ∀ a ∈ J, R a b}` of
elements `b` of type `β` such that `R a b` for every element `a` of `J`. -/
def leftDual (J : Set α) : Set β := {b : β | ∀ ⦃a⦄, a ∈ J → R a b}


/-- `rightDual` maps any set `I` of elements of type `β` to the set `{a : α | ∀ b ∈ I, R a b}`
of elements `a` of type `α` such that `R a b` for every element `b` of `I`. -/
def rightDual (I : Set β) : Set α := {a : α | ∀ ⦃b⦄, b ∈ I → R a b}


/-- The pair of functions `toDual ∘ leftDual` and `rightDual ∘ ofDual` forms a Galois connection. -/
theorem gc_leftDual_rightDual : GaloisConnection (toDual ∘ R.leftDual) (R.rightDual ∘ ofDual) :=
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       R : Rel α β
                                       x✝³ : Set α
                                       x✝² : OrderDual (Set β)
                                       h : LE.le (Function.comp (⇑OrderDual.toDual) R.leftDual x✝³) x✝²
                                       x✝¹ : α
                                       ha : Membership.mem x✝³ x✝¹
                                       x✝ : β
                                       hb : Membership.mem (OrderDual.ofDual x✝²) x✝
                                       ⊢ Membership.mem x✝² x✝
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  fun _ _ ↦ ⟨fun h _ ha _ hb ↦ h (by simpa) ha, fun h _ hb _ ha ↦ h (by simpa) hb⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- `leftFixedPoints` is the set of elements `J : Set α` satisfying `rightDual (leftDual J) = J`. -/
def leftFixedPoints := {J : Set α | R.rightDual (R.leftDual J) = J}


/-- `rightFixedPoints` is the set of elements `I : Set β` satisfying `leftDual (rightDual I) = I`.
-/
def rightFixedPoints := {I : Set β | R.leftDual (R.rightDual I) = I}


/-- `leftDual` maps every element `J` to `rightFixedPoints`. -/
theorem leftDual_mem_rightFixedPoint (J : Set α) : R.leftDual J ∈ R.rightFixedPoints := by
  /-
    α : Type u_1
    β : Type u_2
    R : Rel α β
    J : Set α
    ⊢ Membership.mem R.rightFixedPoints (R.leftDual J)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : Type u_2
      R : Rel α β
      J : Set α
      ⊢ LE.le (R.leftDual (R.rightDual (R.leftDual J))) (R.leftDual J)
    -/
  · apply R.gc_leftDual_rightDual.monotone_l; exact R.gc_leftDual_rightDual.le_u_l J
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      R : Rel α β
      J : Set α
      ⊢ LE.le (R.leftDual J) (R.leftDual (R.rightDual (R.leftDual J)))
    -/
  · exact R.gc_leftDual_rightDual.l_u_le (R.leftDual J)
    /-
      🎉 no goals
    -/


/-- `rightDual` maps every element `I` to `leftFixedPoints`. -/
theorem rightDual_mem_leftFixedPoint (I : Set β) : R.rightDual I ∈ R.leftFixedPoints := by
  /-
    α : Type u_1
    β : Type u_2
    R : Rel α β
    I : Set β
    ⊢ Membership.mem R.leftFixedPoints (R.rightDual I)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      β : Type u_2
      R : Rel α β
      I : Set β
      ⊢ LE.le (R.rightDual (R.leftDual (R.rightDual I))) (R.rightDual I)
    -/
  · apply R.gc_leftDual_rightDual.monotone_u; exact R.gc_leftDual_rightDual.l_u_le I
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      R : Rel α β
      I : Set β
      ⊢ LE.le (R.rightDual I) (R.rightDual (R.leftDual (R.rightDual I)))
    -/
  · exact R.gc_leftDual_rightDual.le_u_l (R.rightDual I)
    /-
      🎉 no goals
    -/


/-- The maps `leftDual` and `rightDual` induce inverse bijections between the sets of fixed points.
-/
def equivFixedPoints : R.leftFixedPoints ≃ R.rightFixedPoints where
  toFun := fun ⟨J, _⟩ => ⟨R.leftDual J, R.leftDual_mem_rightFixedPoint J⟩
  invFun := fun ⟨I, _⟩ => ⟨R.rightDual I, R.rightDual_mem_leftFixedPoint I⟩
                   /-
                     α : Type u_1
                     β : Type u_2
                     R : Rel α β
                     J : ↑R.leftFixedPoints
                     ⊢ Eq ((fun x => Rel.equivFixedPoints.match_2 R (fun x => ↑R.leftFixedPoints) x …
                   -/
  left_inv J := by cases' J with J hJ; rw [Subtype.mk.injEq, hJ]
                                       /-
                                         🎉 no goals
                                       -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      R : Rel α β
                      I : ↑R.rightFixedPoints
                      ⊢ Eq ((fun x => Rel.equivFixedPoints.match_1 R (fun x => ↑R.rightFixedPoints)  …
                    -/
  right_inv I := by cases' I with I hI; rw [Subtype.mk.injEq, hI]
                                        /-
                                          🎉 no goals
                                        -/


theorem rightDual_leftDual_le_of_le {J J' : Set α} (h : J' ∈ R.leftFixedPoints) (h₁ : J ≤ J') :
    R.rightDual (R.leftDual J) ≤ J' := by
  /-
    α : Type u_1
    β : Type u_2
    R : Rel α β
    J J' : Set α
    h : Membership.mem R.leftFixedPoints J'
    h₁ : LE.le J J'
    ⊢ LE.le (R.rightDual (R.leftDual J)) J'
  -/
  rw [← h]
  /-
    α : Type u_1
    β : Type u_2
    R : Rel α β
    J J' : Set α
    h : Membership.mem R.leftFixedPoints J'
    h₁ : LE.le J J'
    ⊢ LE.le (R.rightDual (R.leftDual J)) (R.rightDual (R.leftDual J'))
  -/
  apply R.gc_leftDual_rightDual.monotone_u
  /-
    case a
    α : Type u_1
    β : Type u_2
    R : Rel α β
    J J' : Set α
    h : Membership.mem R.leftFixedPoints J'
    h₁ : LE.le J J'
    ⊢ LE.le (fun b => ∀ ⦃a : α⦄, Membership.mem J a → R a b) fun b => ∀ ⦃a : α⦄, M …
  -/
  apply R.gc_leftDual_rightDual.monotone_l
  /-
    case a.a
    α : Type u_1
    β : Type u_2
    R : Rel α β
    J J' : Set α
    h : Membership.mem R.leftFixedPoints J'
    h₁ : LE.le J J'
    ⊢ LE.le J J'
  -/
  exact h₁
  /-
    🎉 no goals
  -/


theorem leftDual_rightDual_le_of_le {I I' : Set β} (h : I' ∈ R.rightFixedPoints) (h₁ : I ≤ I') :
    R.leftDual (R.rightDual I) ≤ I' := by
  /-
    α : Type u_1
    β : Type u_2
    R : Rel α β
    I I' : Set β
    h : Membership.mem R.rightFixedPoints I'
    h₁ : LE.le I I'
    ⊢ LE.le (R.leftDual (R.rightDual I)) I'
  -/
  rw [← h]
  /-
    α : Type u_1
    β : Type u_2
    R : Rel α β
    I I' : Set β
    h : Membership.mem R.rightFixedPoints I'
    h₁ : LE.le I I'
    ⊢ LE.le (R.leftDual (R.rightDual I)) (R.leftDual (R.rightDual I'))
  -/
  apply R.gc_leftDual_rightDual.monotone_l
  /-
    case a
    α : Type u_1
    β : Type u_2
    R : Rel α β
    I I' : Set β
    h : Membership.mem R.rightFixedPoints I'
    h₁ : LE.le I I'
    ⊢ LE.le (R.rightDual I') (R.rightDual I)
  -/
  apply R.gc_leftDual_rightDual.monotone_u
  /-
    case a.a
    α : Type u_1
    β : Type u_2
    R : Rel α β
    I I' : Set β
    h : Membership.mem R.rightFixedPoints I'
    h₁ : LE.le I I'
    ⊢ LE.le I' I
  -/
  exact h₁
  /-
    🎉 no goals
  -/


