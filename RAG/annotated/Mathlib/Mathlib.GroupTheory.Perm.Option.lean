@[simp]
theorem Equiv.optionCongr_one {α : Type*} : (1 : Perm α).optionCongr = 1 :=
  Equiv.optionCongr_refl


@[simp]
theorem Equiv.optionCongr_swap {α : Type*} [DecidableEq α] (x y : α) :
    optionCongr (swap x y) = swap (some x) (some y) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    ⊢ Eq (Equiv.optionCongr (Equiv.swap x y)) (Equiv.swap (Option.some x) (Option. …
  -/
  ext (_ | i)
    /-
      case H.none.a
      α : Type u_1
      inst✝ : DecidableEq α
      x y a✝ : α
      ⊢ Iff (Membership.mem ((Equiv.optionCongr (Equiv.swap x y)) Option.none) a✝) ( …
    -/
  · simp [swap_apply_of_ne_of_ne]
    /-
      🎉 no goals
    -/
    /-
      case H.some.a
      α : Type u_1
      inst✝ : DecidableEq α
      x y i a✝ : α
      ⊢ Iff (Membership.mem ((Equiv.optionCongr (Equiv.swap x y)) (Option.some i)) a …
    -/
  · by_cases hx : i = x
    · simp only [hx, optionCongr_apply, Option.map_some', swap_apply_left, Option.mem_def,
             Option.some.injEq]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      x y i a✝ : α
      hx : Not (Eq i x)
      ⊢ Iff (Membership.mem ((Equiv.optionCongr (Equiv.swap x y)) (Option.some i)) a …
    -/
                            /-
                              🎉 no goals
                            -/
    by_cases hy : i = y <;> simp [hx, hy, swap_apply_of_ne_of_ne]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem Equiv.optionCongr_sign {α : Type*} [DecidableEq α] [Fintype α] (e : Perm α) :
    Perm.sign e.optionCongr = Perm.sign e := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    e : Equiv.Perm α
    ⊢ Eq (Equiv.Perm.sign (Equiv.optionCongr e)) (Equiv.Perm.sign e)
  -/
  refine Perm.swap_induction_on e ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      e : Equiv.Perm α
      ⊢ Eq (Equiv.Perm.sign (Equiv.optionCongr 1)) (Equiv.Perm.sign 1)
    -/
  · simp [Perm.one_def]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      e : Equiv.Perm α
      ⊢ ∀ (f : Equiv.Perm α) (x y : α), Ne x y → Eq (Equiv.Perm.sign (Equiv.optionCo …
    -/
  · intro f x y hne h
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      e f : Equiv.Perm α
      x y : α
      hne : Ne x y
      h : Eq (Equiv.Perm.sign (Equiv.optionCongr f)) (Equiv.Perm.sign f)
      ⊢ Eq (Equiv.Perm.sign (Equiv.optionCongr (HMul.hMul (Equiv.swap x y) f))) (Equ …
    -/
    simp [h, hne, Perm.mul_def, ← Equiv.optionCongr_trans]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_equiv_removeNone {α : Type*} [DecidableEq α] (σ : Perm (Option α)) :
    (removeNone σ).optionCongr = swap none (σ none) * σ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    σ : Equiv.Perm (Option α)
    ⊢ Eq (Equiv.removeNone σ).optionCongr (HMul.hMul (Equiv.swap Option.none (σ Op …
  -/
  ext1 x
  have : Option.map (⇑(removeNone σ)) x = (swap none (σ none)) (σ x) := by
    cases' x with x
    · simp
    · cases h : σ (some _)
      · simp [removeNone_none _ h]
      · have hn : σ (some x) ≠ none := by simp [h]
        have hσn : σ (some x) ≠ σ none := σ.injective.ne (by simp)
        simp [removeNone_some _ ⟨_, h⟩, ← h, swap_apply_of_ne_of_ne hn hσn]
  /-
    case H
    α : Type u_1
    inst✝ : DecidableEq α
    σ : Equiv.Perm (Option α)
    x : Option α
    this : Eq (Option.map (⇑(Equiv.removeNone σ)) x) ((Equiv.swap Option.none (σ O …
    ⊢ Eq ((Equiv.removeNone σ).optionCongr x) ((HMul.hMul (Equiv.swap Option.none  …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


/-- Permutations of `Option α` are equivalent to fixing an
`Option α` and permuting the remaining with a `Perm α`.
The fixed `Option α` is swapped with `none`. -/
@[simps]
def Equiv.Perm.decomposeOption {α : Type*} [DecidableEq α] :
    Perm (Option α) ≃ Option α × Perm α where
  toFun σ := (σ none, removeNone σ)
  invFun i := swap none i.1 * i.2.optionCongr
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     σ : Equiv.Perm (Option α)
                     ⊢ Eq ((fun i => HMul.hMul (Equiv.swap Option.none i.1) (Equiv.optionCongr i.2) …
                   -/
  left_inv σ := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv := fun ⟨x, σ⟩ => by
    have : removeNone (swap none x * σ.optionCongr) = σ :=
      Equiv.optionCongr_injective (by simp [← mul_assoc])
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      x✝ : Prod (Option α) (Equiv.Perm α)
      x : Option α
      σ : Equiv.Perm α
      this : Eq (Equiv.removeNone (HMul.hMul (Equiv.swap Option.none x) (Equiv.optio …
      ⊢ Eq ((fun σ => { fst := σ Option.none, snd := Equiv.removeNone σ }) ((fun i = …
    -/
    simp [← Perm.eq_inv_iff_eq, this]
    /-
      🎉 no goals
    -/


theorem Equiv.Perm.decomposeOption_symm_of_none_apply {α : Type*} [DecidableEq α] (e : Perm α)
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   inst✝ : DecidableEq α
                                                                                   e : Equiv.Perm α
                                                                                   i : Option α
                                                                                   ⊢ Eq ((Equiv.Perm.decomposeOption.symm { fst := Option.none, snd := e }) i) (O …
                                                                                 -/
    (i : Option α) : Equiv.Perm.decomposeOption.symm (none, e) i = i.map e := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem Equiv.Perm.decomposeOption_symm_sign {α : Type*} [DecidableEq α] [Fintype α] (e : Perm α) :
                                                                              /-
                                                                                α : Type u_1
                                                                                inst✝¹ : DecidableEq α
                                                                                inst✝ : Fintype α
                                                                                e : Equiv.Perm α
                                                                                ⊢ Eq (Equiv.Perm.sign (Equiv.Perm.decomposeOption.symm { fst := Option.none, s …
                                                                              -/
    Perm.sign (Equiv.Perm.decomposeOption.symm (none, e)) = Perm.sign e := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- The set of all permutations of `Option α` can be constructed by augmenting the set of
permutations of `α` by each element of `Option α` in turn. -/
theorem Finset.univ_perm_option {α : Type*} [DecidableEq α] [Fintype α] :
    @Finset.univ (Perm <| Option α) _ =
      (Finset.univ : Finset <| Option α × Perm α).map Equiv.Perm.decomposeOption.symm.toEmbedding :=
  (Finset.univ_map_equiv_to_embedding _).symm

