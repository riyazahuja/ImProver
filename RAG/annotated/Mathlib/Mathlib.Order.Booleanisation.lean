/-- Boolean algebra containing a given generalised Boolean algebra `α` as a sublattice.

This should be thought of as made of a copy of `α` (representing elements of `α`) living under
another copy of `α` (representing complements of elements of `α`). -/
def Booleanisation (α : Type*) := α ⊕ α


instance instDecidableEq [DecidableEq α] : DecidableEq (Booleanisation α) :=
  inferInstanceAs <| DecidableEq (α ⊕ α)


/-- The natural inclusion `a ↦ a` from a generalized Boolean algebra to its generated Boolean
algebra. -/
@[match_pattern] def lift : α → Booleanisation α := Sum.inl


/-- The inclusion `a ↦ aᶜ from a generalized Boolean algebra to its generated Boolean algebra. -/
@[match_pattern] def comp : α → Booleanisation α := Sum.inr


/-- The complement operator on `Booleanisation α` sends `a` to `aᶜ` and `aᶜ` to `a`, for `a : α`. -/
instance instCompl : HasCompl (Booleanisation α) where
  compl x := match x with
    | lift a => comp a
    | comp a => lift a


@[simp] lemma compl_lift (a : α) : (lift a)ᶜ = comp a := rfl

@[simp] lemma compl_comp (a : α) : (comp a)ᶜ = lift a := rfl


/-- The order on `Booleanisation α` is as follows: For `a b : α`,
* `a ≤ b` iff `a ≤ b` in `α`
* `a ≤ bᶜ` iff `a` and `b` are disjoint in `α`
* `aᶜ ≤ bᶜ` iff `b ≤ a` in `α`
* `¬ aᶜ ≤ b` -/
protected inductive LE : Booleanisation α → Booleanisation α → Prop
  | protected lift {a b} : a ≤ b → Booleanisation.LE (lift a) (lift b)
  | protected comp {a b} : a ≤ b → Booleanisation.LE (comp b) (comp a)
  | protected sep {a b} : Disjoint a b → Booleanisation.LE (lift a) (comp b)


/-- The order on `Booleanisation α` is as follows: For `a b : α`,
* `a < b` iff `a < b` in `α`
* `a < bᶜ` iff `a` and `b` are disjoint in `α`
* `aᶜ < bᶜ` iff `b < a` in `α`
* `¬ aᶜ < b` -/
protected inductive LT : Booleanisation α → Booleanisation α → Prop
  | protected lift {a b} : a < b → Booleanisation.LT (lift a) (lift b)
  | protected comp {a b} : a < b → Booleanisation.LT (comp b) (comp a)
  | protected sep {a b} : Disjoint a b → Booleanisation.LT (lift a) (comp b)


@[inherit_doc Booleanisation.LE]
instance instLE : LE (Booleanisation α) where
  le := Booleanisation.LE


@[inherit_doc Booleanisation.LT]
instance instLT : LT (Booleanisation α) where
  lt := Booleanisation.LT


/-- The supremum on `Booleanisation α` is as follows: For `a b : α`,
* `a ⊔ b` is `a ⊔ b`
* `a ⊔ bᶜ` is `(b \ a)ᶜ`
* `aᶜ ⊔ b` is `(a \ b)ᶜ`
* `aᶜ ⊔ bᶜ` is `(a ⊓ b)ᶜ` -/
instance instSup : Max (Booleanisation α) where
  max x y := match x, y with
    | lift a, lift b => lift (a ⊔ b)
    | lift a, comp b => comp (b \ a)
    | comp a, lift b => comp (a \ b)
    | comp a, comp b => comp (a ⊓ b)


/-- The infimum on `Booleanisation α` is as follows: For `a b : α`,
* `a ⊓ b` is `a ⊓ b`
* `a ⊓ bᶜ` is `a \ b`
* `aᶜ ⊓ b` is `b \ a`
* `aᶜ ⊓ bᶜ` is `(a ⊔ b)ᶜ` -/
instance instInf : Min (Booleanisation α) where
  min x y := match x, y with
    | lift a, lift b => lift (a ⊓ b)
    | lift a, comp b => lift (a \ b)
    | comp a, lift b => lift (b \ a)
    | comp a, comp b => comp (a ⊔ b)


/-- The bottom element of `Booleanisation α` is the bottom element of `α`. -/
instance instBot : Bot (Booleanisation α) where
  bot := lift ⊥


/-- The top element of `Booleanisation α` is the complement of the bottom element of `α`. -/
instance instTop : Top (Booleanisation α) where
  top := comp ⊥


/-- The difference operator on `Booleanisation α` is as follows: For `a b : α`,
* `a \ b` is `a \ b`
* `a \ bᶜ` is `a ⊓ b`
* `aᶜ \ b` is `(a ⊔ b)ᶜ`
* `aᶜ \ bᶜ` is `b \ a` -/
instance instSDiff : SDiff (Booleanisation α) where
  sdiff x y := match x, y with
    | lift a, lift b => lift (a \ b)
    | lift a, comp b => lift (a ⊓ b)
    | comp a, lift b => comp (a ⊔ b)
    | comp a, comp b => lift (b \ a)


                                                            /-
                                                              α : Type u_1
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              a b : α
                                                              ⊢ LE.le (Booleanisation.lift a) (Booleanisation.lift b) → LE.le a b
                                                            -/
@[simp] lemma lift_le_lift : lift a ≤ lift b ↔ a ≤ b := ⟨by rintro ⟨_⟩; assumption, LE.lift⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

                                                            /-
                                                              α : Type u_1
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              a b : α
                                                              ⊢ LE.le (Booleanisation.comp a) (Booleanisation.comp b) → LE.le b a
                                                            -/
@[simp] lemma comp_le_comp : comp a ≤ comp b ↔ b ≤ a := ⟨by rintro ⟨_⟩; assumption, LE.comp⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : GeneralizedBooleanAlgebra α
                                                                     a b : α
                                                                     ⊢ LE.le (Booleanisation.lift a) (Booleanisation.comp b) → Disjoint a b
                                                                   -/
@[simp] lemma lift_le_comp : lift a ≤ comp b ↔ Disjoint a b := ⟨by rintro ⟨_⟩; assumption, LE.sep⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

@[simp] lemma not_comp_le_lift : ¬ comp a ≤ lift b := fun h ↦ nomatch h


                                                            /-
                                                              α : Type u_1
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              a b : α
                                                              ⊢ LT.lt (Booleanisation.lift a) (Booleanisation.lift b) → LT.lt a b
                                                            -/
@[simp] lemma lift_lt_lift : lift a < lift b ↔ a < b := ⟨by rintro ⟨_⟩; assumption, LT.lift⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

                                                            /-
                                                              α : Type u_1
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              a b : α
                                                              ⊢ LT.lt (Booleanisation.comp a) (Booleanisation.comp b) → LT.lt b a
                                                            -/
@[simp] lemma comp_lt_comp : comp a < comp b ↔ b < a := ⟨by rintro ⟨_⟩; assumption, LT.comp⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : GeneralizedBooleanAlgebra α
                                                                     a b : α
                                                                     ⊢ LT.lt (Booleanisation.lift a) (Booleanisation.comp b) → Disjoint a b
                                                                   -/
@[simp] lemma lift_lt_comp : lift a < comp b ↔ Disjoint a b := ⟨by rintro ⟨_⟩; assumption, LT.sep⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

@[simp] lemma not_comp_lt_lift : ¬ comp a < lift b := fun h ↦ nomatch h


@[simp] lemma lift_sup_lift (a b : α) : lift a ⊔ lift b = lift (a ⊔ b) := rfl

@[simp] lemma lift_sup_comp (a b : α) : lift a ⊔ comp b = comp (b \ a) := rfl

@[simp] lemma comp_sup_lift (a b : α) : comp a ⊔ lift b = comp (a \ b) := rfl

@[simp] lemma comp_sup_comp (a b : α) : comp a ⊔ comp b = comp (a ⊓ b) := rfl


@[simp] lemma lift_inf_lift (a b : α) : lift a ⊓ lift b = lift (a ⊓ b) := rfl

@[simp] lemma lift_inf_comp (a b : α) : lift a ⊓ comp b = lift (a \ b) := rfl

@[simp] lemma comp_inf_lift (a b : α) : comp a ⊓ lift b = lift (b \ a) := rfl

@[simp] lemma comp_inf_comp (a b : α) : comp a ⊓ comp b = comp (a ⊔ b) := rfl


@[simp] lemma lift_bot : lift (⊥ : α) = ⊥ := rfl

@[simp] lemma comp_bot : comp (⊥ : α) = ⊤ := rfl


@[simp] lemma lift_sdiff_lift (a b : α) : lift a \ lift b = lift (a \ b) := rfl

@[simp] lemma lift_sdiff_comp (a b : α) : lift a \ comp b = lift (a ⊓ b) := rfl

@[simp] lemma comp_sdiff_lift (a b : α) : comp a \ lift b = comp (a ⊔ b) := rfl

@[simp] lemma comp_sdiff_comp (a b : α) : comp a \ comp b = lift (b \ a) := rfl


instance instPreorder : Preorder (Booleanisation α) where
  lt := (· < ·)
  lt_iff_le_not_le x y := match x, y with
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Iff (LT.lt (Booleanisation.lift a) (Booleanisation.lift b)) (And (LE.le (Boo …
                           -/
    | lift a, lift b => by simp [lt_iff_le_not_le]
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Iff (LT.lt (Booleanisation.lift a) (Booleanisation.comp b)) (And (LE.le (Boo …
                           -/
    | lift a, comp b => by simp
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Iff (LT.lt (Booleanisation.comp a) (Booleanisation.lift b)) (And (LE.le (Boo …
                           -/
    | comp a, lift b => by simp
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Iff (LT.lt (Booleanisation.comp a) (Booleanisation.comp b)) (And (LE.le (Boo …
                           -/
    | comp a, comp b => by simp [lt_iff_le_not_le]
                           /-
                             🎉 no goals
                           -/
  le_refl x := match x with
    | lift _ => LE.lift le_rfl
    | comp _ => LE.comp le_rfl
  le_trans x y z hxy hyz := match x, y, z, hxy, hyz with
    | lift _, lift _, lift _, LE.lift hab, LE.lift hbc => LE.lift <| hab.trans hbc
    | lift _, lift _, comp _, LE.lift hab, LE.sep hbc => LE.sep <| hbc.mono_left hab
    | lift _, comp _, comp _, LE.sep hab, LE.comp hcb => LE.sep <| hab.mono_right hcb
    | comp _, comp _, comp _, LE.comp hba, LE.comp hcb => LE.comp <| hcb.trans hba


instance instPartialOrder : PartialOrder (Booleanisation α) where
  le_antisymm x y hxy hyx := match x, y, hxy, hyx with
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : GeneralizedBooleanAlgebra α
                                                       a✝ b✝ : α
                                                       x y : Booleanisation α
                                                       hxy : LE.le x y
                                                       hyx : LE.le y x
                                                       a b : α
                                                       hab : LE.le a b
                                                       hba : LE.le b a
                                                       ⊢ Eq (Booleanisation.lift a) (Booleanisation.lift b)
                                                     -/
    | lift a, lift b, LE.lift hab, LE.lift hba => by rw [hab.antisymm hba]
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : GeneralizedBooleanAlgebra α
                                                       a✝ b✝ : α
                                                       x y : Booleanisation α
                                                       hxy : LE.le x y
                                                       hyx : LE.le y x
                                                       a b : α
                                                       hab : LE.le b a
                                                       hba : LE.le a b
                                                       ⊢ Eq (Booleanisation.comp a) (Booleanisation.comp b)
                                                     -/
    | comp a, comp b, LE.comp hab, LE.comp hba => by rw [hab.antisymm hba]
                                                     /-
                                                       🎉 no goals
                                                     -/

-- The linter significantly hinders readability here.

set_option linter.unusedVariables false in
instance instSemilatticeSup : SemilatticeSup (Booleanisation α) where
  sup x y := max x y
  le_sup_left x y := match x, y with
    | lift a, lift b => LE.lift le_sup_left
    | lift a, comp b => LE.sep disjoint_sdiff_self_right
    | comp a, lift b => LE.comp sdiff_le
    | comp a, comp b => LE.comp inf_le_left
  le_sup_right x y := match x, y with
    | lift a, lift b => LE.lift le_sup_right
    | lift a, comp b => LE.comp sdiff_le
    | comp a, lift b => LE.sep disjoint_sdiff_self_right
    | comp a, comp b => LE.comp inf_le_right
  sup_le x y z hxz hyz := match x, y, z, hxz, hyz with
    | lift a, lift b, lift c, LE.lift hac, LE.lift hbc => LE.lift <| sup_le hac hbc
    | lift a, lift b, comp c, LE.sep hac, LE.sep hbc => LE.sep <| hac.sup_left hbc
    | lift a, comp b, comp c, LE.sep hac, LE.comp hcb => LE.comp <| le_sdiff.2 ⟨hcb, hac.symm⟩
    | comp a, lift b, comp c, LE.comp hca, LE.sep hbc => LE.comp <| le_sdiff.2 ⟨hca, hbc.symm⟩
    | comp a, comp b, comp c, LE.comp hca, LE.comp hcb => LE.comp <| le_inf hca hcb

-- The linter significantly hinders readability here.

set_option linter.unusedVariables false in
instance instSemilatticeInf : SemilatticeInf (Booleanisation α) where
  inf x y := min x y
  inf_le_left x y := match x, y with
    | lift a, lift b => LE.lift inf_le_left
    | lift a, comp b => LE.lift sdiff_le
    | comp a, lift b => LE.sep disjoint_sdiff_self_left
    | comp a, comp b => LE.comp le_sup_left
  inf_le_right x y := match x, y with
    | lift a, lift b => LE.lift inf_le_right
    | lift a, comp b => LE.sep disjoint_sdiff_self_left
    | comp a, lift b => LE.lift sdiff_le
    | comp a, comp b => LE.comp le_sup_right
  le_inf x y z hxz hyz := match x, y, z, hxz, hyz with
    | lift a, lift b, lift c, LE.lift hab, LE.lift hac => LE.lift <| le_inf hab hac
    | lift a, lift b, comp c, LE.lift hab, LE.sep hac => LE.lift <| le_sdiff.2 ⟨hab, hac⟩
    | lift a, comp b, lift c, LE.sep hab, LE.lift hac => LE.lift <| le_sdiff.2 ⟨hac, hab⟩
    | lift a, comp b, comp c, LE.sep hab, LE.sep hac => LE.sep <| hab.sup_right hac
    | comp a, comp b, comp c, LE.comp hba, LE.comp hca => LE.comp <| sup_le hba hca


instance instDistribLattice : DistribLattice (Booleanisation α) where
  inf_le_left _ _ := inf_le_left
  inf_le_right _ _ := inf_le_right
  le_inf _ _ _ := le_inf
  le_sup_inf x y z := match x, y, z with
    | lift _, lift _, lift _ => LE.lift le_sup_inf
                                              /-
                                                α : Type u_1
                                                inst✝ : GeneralizedBooleanAlgebra α
                                                a✝ b✝ : α
                                                x y z : Booleanisation α
                                                a b c : α
                                                ⊢ LE.le (SDiff.sdiff (Max.max a b) (SDiff.sdiff c a)) (Max.max a (SDiff.sdiff  …
                                              -/
    | lift a, lift b, comp c => LE.lift <| by simp [sup_left_comm, sup_comm, sup_assoc]
                                              /-
                                                🎉 no goals
                                              -/
    | lift a, comp b, lift c => LE.lift <| by
      /-
        α : Type u_1
        inst✝ : GeneralizedBooleanAlgebra α
        a✝ b✝ : α
        x y z : Booleanisation α
        a b c : α
        ⊢ LE.le (SDiff.sdiff (Max.max a c) (SDiff.sdiff b a)) (Max.max a (SDiff.sdiff  …
      -/
      simp [sup_left_comm (a := b \ a), sup_comm (a := b \ a)]
      /-
        🎉 no goals
      -/
                                              /-
                                                α : Type u_1
                                                inst✝ : GeneralizedBooleanAlgebra α
                                                a✝ b✝ : α
                                                x y z : Booleanisation α
                                                a b c : α
                                                ⊢ LE.le (SDiff.sdiff (Max.max b c) a) (Max.max (SDiff.sdiff b a) (SDiff.sdiff  …
                                              -/
    | lift a, comp b, comp c => LE.comp <| by rw [sup_sdiff]
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                α : Type u_1
                                                inst✝ : GeneralizedBooleanAlgebra α
                                                a✝ b✝ : α
                                                x y z : Booleanisation α
                                                a b c : α
                                                ⊢ LE.le (SDiff.sdiff a (Min.min b c)) (Max.max (SDiff.sdiff a b) (SDiff.sdiff  …
                                              -/
    | comp a, lift b, lift c => LE.comp <| by rw [sdiff_inf]
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                α : Type u_1
                                                inst✝ : GeneralizedBooleanAlgebra α
                                                a✝ b✝ : α
                                                x y z : Booleanisation α
                                                a b c : α
                                                ⊢ LE.le (SDiff.sdiff a (SDiff.sdiff b c)) (Max.max (SDiff.sdiff a b) (Min.min  …
                                              -/
    | comp a, lift b, comp c => LE.comp <| by rw [sdiff_sdiff_right']
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                α : Type u_1
                                                inst✝ : GeneralizedBooleanAlgebra α
                                                a✝ b✝ : α
                                                x y z : Booleanisation α
                                                a b c : α
                                                ⊢ LE.le (SDiff.sdiff a (SDiff.sdiff c b)) (Max.max (Min.min a b) (SDiff.sdiff  …
                                              -/
    | comp a, comp b, lift c => LE.comp <| by rw [sdiff_sdiff_right', sup_comm]
                                              /-
                                                🎉 no goals
                                              -/
    | comp _, comp _, comp _ => LE.comp (inf_sup_left _ _ _).le

-- The linter significantly hinders readability here.

set_option linter.unusedVariables false in
instance instBoundedOrder : BoundedOrder (Booleanisation α) where
  le_top x := match x with
    | lift a => LE.sep disjoint_bot_right
    | comp a => LE.comp bot_le
  bot_le x := match x with
    | lift a => LE.lift bot_le
    | comp a => LE.sep disjoint_bot_left


instance instBooleanAlgebra : BooleanAlgebra (Booleanisation α) where
  le_top _ := le_top
  bot_le _ := bot_le
  inf_compl_le_bot x := match x with
                   /-
                     α : Type u_1
                     inst✝ : GeneralizedBooleanAlgebra α
                     a✝ b : α
                     x : Booleanisation α
                     a : α
                     ⊢ LE.le (Min.min (Booleanisation.lift a) (HasCompl.compl (Booleanisation.lift  …
                   -/
    | lift a => by simp
                   /-
                     🎉 no goals
                   -/
                   /-
                     α : Type u_1
                     inst✝ : GeneralizedBooleanAlgebra α
                     a✝ b : α
                     x : Booleanisation α
                     a : α
                     ⊢ LE.le (Min.min (Booleanisation.comp a) (HasCompl.compl (Booleanisation.comp  …
                   -/
    | comp a => by simp
                   /-
                     🎉 no goals
                   -/
  top_le_sup_compl x := match x with
                   /-
                     α : Type u_1
                     inst✝ : GeneralizedBooleanAlgebra α
                     a✝ b : α
                     x : Booleanisation α
                     a : α
                     ⊢ LE.le Top.top (Max.max (Booleanisation.lift a) (HasCompl.compl (Booleanisati …
                   -/
    | lift a => by simp
                   /-
                     🎉 no goals
                   -/
                   /-
                     α : Type u_1
                     inst✝ : GeneralizedBooleanAlgebra α
                     a✝ b : α
                     x : Booleanisation α
                     a : α
                     ⊢ LE.le Top.top (Max.max (Booleanisation.comp a) (HasCompl.compl (Booleanisati …
                   -/
    | comp a => by simp
                   /-
                     🎉 no goals
                   -/
  sdiff_eq x y := match x, y with
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Eq (SDiff.sdiff (Booleanisation.lift a) (Booleanisation.lift b)) (Min.min (B …
                           -/
    | lift a, lift b => by simp
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Eq (SDiff.sdiff (Booleanisation.lift a) (Booleanisation.comp b)) (Min.min (B …
                           -/
    | lift a, comp b => by simp
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Eq (SDiff.sdiff (Booleanisation.comp a) (Booleanisation.lift b)) (Min.min (B …
                           -/
    | comp a, lift b => by simp
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             inst✝ : GeneralizedBooleanAlgebra α
                             a✝ b✝ : α
                             x y : Booleanisation α
                             a b : α
                             ⊢ Eq (SDiff.sdiff (Booleanisation.comp a) (Booleanisation.comp b)) (Min.min (B …
                           -/
    | comp a, comp b => by simp
                           /-
                             🎉 no goals
                           -/


/-- The embedding from a generalised Boolean algebra to its generated Boolean algebra. -/
def liftLatticeHom : LatticeHom α (Booleanisation α) where
  toFun := lift
  map_sup' _ _ := rfl
  map_inf' _ _ := rfl


lemma liftLatticeHom_injective : Injective (liftLatticeHom (α := α)) := Sum.inl_injective


