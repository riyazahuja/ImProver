/-- Propositions form a distributive lattice. -/
instance Prop.instDistribLattice : DistribLattice Prop where
  sup := Or
  le_sup_left := @Or.inl
  le_sup_right := @Or.inr
  sup_le := fun _ _ _ => Or.rec
  inf := And
  inf_le_left := @And.left
  inf_le_right := @And.right
  le_inf := fun _ _ _ Hab Hac Ha => And.intro (Hab Ha) (Hac Ha)
  le_sup_inf := fun _ _ _ => or_and_left.2


/-- Propositions form a bounded order. -/
instance Prop.instBoundedOrder : BoundedOrder Prop where
  top := True
  le_top _ _ := True.intro
  bot := False
  bot_le := @False.elim


@[simp]
theorem Prop.bot_eq_false : (⊥ : Prop) = False :=
  rfl


@[simp]
theorem Prop.top_eq_true : (⊤ : Prop) = True :=
  rfl


instance Prop.le_isTotal : IsTotal Prop (· ≤ ·) :=
                 /-
                   p q : Prop
                   ⊢ Or (LE.le p q) (LE.le q p)
                 -/
                                    /-
                                      🎉 no goals
                                    -/
  ⟨fun p q => by by_cases h : q <;> simp [h]⟩
                                    /-
                                      🎉 no goals
                                    -/


noncomputable instance Prop.linearOrder : LinearOrder Prop := by
  classical
  exact Lattice.toLinearOrder Prop


@[simp]
theorem sup_Prop_eq : (· ⊔ ·) = (· ∨ ·) :=
  rfl


@[simp]
theorem inf_Prop_eq : (· ⊓ ·) = (· ∧ ·) :=
  rfl


theorem disjoint_iff [∀ i, OrderBot (α' i)] {f g : ∀ i, α' i} :
    Disjoint f g ↔ ∀ i, Disjoint (f i) (g i) := by
  classical
  constructor
  · intro h i x hf hg
    exact (update_le_iff.mp <| h (update_le_iff.mpr ⟨hf, fun _ _ => bot_le⟩)
      (update_le_iff.mpr ⟨hg, fun _ _ => bot_le⟩)).1
  · intro h x hf hg i
    apply h i (hf i) (hg i)


theorem codisjoint_iff [∀ i, OrderTop (α' i)] {f g : ∀ i, α' i} :
    Codisjoint f g ↔ ∀ i, Codisjoint (f i) (g i) :=
  @disjoint_iff _ (fun i => (α' i)ᵒᵈ) _ _ _ _


theorem isCompl_iff [∀ i, BoundedOrder (α' i)] {f g : ∀ i, α' i} :
    IsCompl f g ↔ ∀ i, IsCompl (f i) (g i) := by
  /-
    ι : Type u_1
    α' : ι → Type u_2
    inst✝¹ : (i : ι) → PartialOrder (α' i)
    inst✝ : (i : ι) → BoundedOrder (α' i)
    f g : (i : ι) → α' i
    ⊢ Iff (IsCompl f g) (∀ (i : ι), IsCompl (f i) (g i))
  -/
  simp_rw [_root_.isCompl_iff, disjoint_iff, codisjoint_iff, forall_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem Prop.disjoint_iff {P Q : Prop} : Disjoint P Q ↔ ¬(P ∧ Q) :=
  disjoint_iff_inf_le


@[simp]
theorem Prop.codisjoint_iff {P Q : Prop} : Codisjoint P Q ↔ P ∨ Q :=
  codisjoint_iff_le_sup.trans <| forall_const True


@[simp]
theorem Prop.isCompl_iff {P Q : Prop} : IsCompl P Q ↔ ¬(P ↔ Q) := by
  /-
    P Q : Prop
    ⊢ Iff (IsCompl P Q) (Not (Iff P Q))
  -/
  rw [_root_.isCompl_iff, Prop.disjoint_iff, Prop.codisjoint_iff, not_iff]
  /-
    P Q : Prop
    ⊢ Iff (And (Not (And P Q)) (Or P Q)) (Iff (Not P) Q)
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  by_cases P <;> by_cases Q <;> simp [*]
                                /-
                                  🎉 no goals
                                -/

-- Porting note: Lean 3 would unfold these for us, but we need to do it manually now

instance Prop.decidablePredBot : DecidablePred (⊥ : α → Prop) := fun _ => instDecidableFalse


instance Prop.decidablePredTop : DecidablePred (⊤ : α → Prop) := fun _ => instDecidableTrue


instance Prop.decidableRelBot : DecidableRel (⊥ : α → α → Prop) := fun _ _ => instDecidableFalse


instance Prop.decidableRelTop : DecidableRel (⊤ : α → α → Prop) := fun _ _ => instDecidableTrue


