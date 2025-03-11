theorem Pi.exists_forall_pos_add_lt [ExistsAddOfLE α] [Finite ι] {x y : ι → α}
    (h : ∀ i, x i < y i) : ∃ ε, 0 < ε ∧ ∀ i, x i + ε < y i := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x y : ι → α
    h : ∀ (i : ι), LT.lt (x i) (y i)
    ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
  -/
  cases nonempty_fintype ι
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x y : ι → α
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
  -/
  cases isEmpty_or_nonempty ι
    /-
      case intro.inl
      α : Type u_1
      ι : Type u_2
      inst✝⁴ : LinearOrderedCancelAddCommMonoid α
      inst✝³ : Nontrivial α
      inst✝² : DenselyOrdered α
      inst✝¹ : ExistsAddOfLE α
      inst✝ : Finite ι
      x y : ι → α
      h : ∀ (i : ι), LT.lt (x i) (y i)
      val✝ : Fintype ι
      h✝ : IsEmpty ι
      ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
    -/
  · obtain ⟨a, ha⟩ := exists_ne (0 : α)
    /-
      case intro.inl.intro
      α : Type u_1
      ι : Type u_2
      inst✝⁴ : LinearOrderedCancelAddCommMonoid α
      inst✝³ : Nontrivial α
      inst✝² : DenselyOrdered α
      inst✝¹ : ExistsAddOfLE α
      inst✝ : Finite ι
      x y : ι → α
      h : ∀ (i : ι), LT.lt (x i) (y i)
      val✝ : Fintype ι
      h✝ : IsEmpty ι
      a : α
      ha : Ne a 0
      ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
    -/
    obtain ha | ha := ha.lt_or_lt <;> obtain ⟨b, hb, -⟩ := exists_pos_add_of_lt' ha <;>
      /-
        case intro.inl.intro.inl.intro.intro
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : LinearOrderedCancelAddCommMonoid α
        inst✝³ : Nontrivial α
        inst✝² : DenselyOrdered α
        inst✝¹ : ExistsAddOfLE α
        inst✝ : Finite ι
        x y : ι → α
        h : ∀ (i : ι), LT.lt (x i) (y i)
        val✝ : Fintype ι
        h✝ : IsEmpty ι
        a : α
        ha✝ : Ne a 0
        ha : LT.lt a 0
        b : α
        hb : LT.lt 0 b
        ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
      -/
      /-
        🎉 no goals
      -/
      exact ⟨b, hb, isEmptyElim⟩
      /-
        🎉 no goals
      -/
  /-
    case intro.inr
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x y : ι → α
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    h✝ : Nonempty ι
    ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
  -/
  choose ε hε hxε using fun i => exists_pos_add_of_lt' (h i)
  /-
    case intro.inr
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x y : ι → α
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    h✝ : Nonempty ι
    ε : ι → α
    hε : ∀ (i : ι), LT.lt 0 (ε i)
    hxε : ∀ (i : ι), Eq (HAdd.hAdd (x i) (ε i)) (y i)
    ⊢ Exists fun ε => And (LT.lt 0 ε) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i))
  -/
  obtain rfl : x + ε = y := funext hxε
  /-
    case intro.inr
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x : ι → α
    val✝ : Fintype ι
    h✝ : Nonempty ι
    ε : ι → α
    hε : ∀ (i : ι), LT.lt 0 (ε i)
    h : ∀ (i : ι), LT.lt (x i) (HAdd.hAdd x ε i)
    hxε : ∀ (i : ι), Eq (HAdd.hAdd (x i) (ε i)) (HAdd.hAdd x ε i)
    ⊢ Exists fun ε_1 => And (LT.lt 0 ε_1) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε_1)  …
  -/
  have hε : 0 < Finset.univ.inf' Finset.univ_nonempty ε := (Finset.lt_inf'_iff _).2 fun i _ => hε _
  /-
    case intro.inr
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x : ι → α
    val✝ : Fintype ι
    h✝ : Nonempty ι
    ε : ι → α
    hε✝ : ∀ (i : ι), LT.lt 0 (ε i)
    h : ∀ (i : ι), LT.lt (x i) (HAdd.hAdd x ε i)
    hxε : ∀ (i : ι), Eq (HAdd.hAdd (x i) (ε i)) (HAdd.hAdd x ε i)
    hε : LT.lt 0 (Finset.univ.inf' ⋯ ε)
    ⊢ Exists fun ε_1 => And (LT.lt 0 ε_1) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε_1)  …
  -/
  obtain ⟨δ, hδ, hδε⟩ := exists_between hε
  /-
    case intro.inr.intro.intro
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : LinearOrderedCancelAddCommMonoid α
    inst✝³ : Nontrivial α
    inst✝² : DenselyOrdered α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : Finite ι
    x : ι → α
    val✝ : Fintype ι
    h✝ : Nonempty ι
    ε : ι → α
    hε✝ : ∀ (i : ι), LT.lt 0 (ε i)
    h : ∀ (i : ι), LT.lt (x i) (HAdd.hAdd x ε i)
    hxε : ∀ (i : ι), Eq (HAdd.hAdd (x i) (ε i)) (HAdd.hAdd x ε i)
    hε : LT.lt 0 (Finset.univ.inf' ⋯ ε)
    δ : α
    hδ : LT.lt 0 δ
    hδε : LT.lt δ (Finset.univ.inf' ⋯ ε)
    ⊢ Exists fun ε_1 => And (LT.lt 0 ε_1) (∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε_1)  …
  -/
  exact ⟨δ, hδ, fun i ↦ add_lt_add_left (hδε.trans_le <| Finset.inf'_le _ <| Finset.mem_univ _) _⟩
  /-
    🎉 no goals
  -/

