/-- A special case of `Finset.sup_eq_iSup` that omits the useless `x ∈ univ` binder. -/
theorem sup_univ_eq_iSup [CompleteLattice β] (f : α → β) : Finset.univ.sup f = iSup f :=
  (sup_eq_iSup _ f).trans <| congr_arg _ <| funext fun _ => iSup_pos (mem_univ _)


/-- A special case of `Finset.inf_eq_iInf` that omits the useless `x ∈ univ` binder. -/
theorem inf_univ_eq_iInf [CompleteLattice β] (f : α → β) : Finset.univ.inf f = iInf f :=
  @sup_univ_eq_iSup _ βᵒᵈ _ _ (f : α → βᵒᵈ)


@[simp]
theorem fold_inf_univ [SemilatticeInf α] [OrderBot α] (a : α) :
    -- Porting note: added `haveI`
    haveI : Std.Commutative (α := α) (· ⊓ ·) := inferInstance
    (Finset.univ.fold (· ⊓ ·) a fun x => x) = ⊥ :=
  eq_bot_iff.2 <|
    ((Finset.fold_op_rel_iff_and <| @le_inf_iff α _).1 le_rfl).2 ⊥ <| Finset.mem_univ _


@[simp]
theorem fold_sup_univ [SemilatticeSup α] [OrderTop α] (a : α) :
    -- Porting note: added `haveI`
    haveI : Std.Commutative (α := α) (· ⊔ ·) := inferInstance
    (Finset.univ.fold (· ⊔ ·) a fun x => x) = ⊤ :=
  @fold_inf_univ αᵒᵈ _ _ _ _


lemma mem_inf [DecidableEq α] {s : Finset ι} {f : ι → Finset α} {a : α} :
                                         /-
                                           ι : Type u_1
                                           α : Type u_2
                                           inst✝¹ : Fintype α
                                           inst✝ : DecidableEq α
                                           s : Finset ι
                                           f : ι → Finset α
                                           a : α
                                           ⊢ Iff (Membership.mem (s.inf f) a) (∀ (i : ι), Membership.mem s i → Membership …
                                         -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    a ∈ s.inf f ↔ ∀ i ∈ s, a ∈ f i := by induction' s using Finset.cons_induction <;> simp [*]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem Finite.exists_max [Finite α] [Nonempty α] [LinearOrder β] (f : α → β) :
    ∃ x₀ : α, ∀ x, f x ≤ f x₀ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Finite α
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ Exists fun x₀ => ∀ (x : α), LE.le (f x) (f x₀)
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝² : Finite α
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder β
    f : α → β
    val✝ : Fintype α
    ⊢ Exists fun x₀ => ∀ (x : α), LE.le (f x) (f x₀)
  -/
  simpa using exists_max_image univ f univ_nonempty
  /-
    🎉 no goals
  -/


theorem Finite.exists_min [Finite α] [Nonempty α] [LinearOrder β] (f : α → β) :
    ∃ x₀ : α, ∀ x, f x₀ ≤ f x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Finite α
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ Exists fun x₀ => ∀ (x : α), LE.le (f x₀) (f x)
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝² : Finite α
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder β
    f : α → β
    val✝ : Fintype α
    ⊢ Exists fun x₀ => ∀ (x : α), LE.le (f x₀) (f x)
  -/
  simpa using exists_min_image univ f univ_nonempty
  /-
    🎉 no goals
  -/

