/-- The antisymmetrization relation. -/
def AntisymmRel (a b : α) : Prop :=
  r a b ∧ r b a


theorem antisymmRel_swap : AntisymmRel (swap r) = AntisymmRel r :=
  funext fun _ => funext fun _ => propext and_comm


@[refl]
theorem antisymmRel_refl [IsRefl α r] (a : α) : AntisymmRel r a a :=
  ⟨refl _, refl _⟩


@[symm]
theorem AntisymmRel.symm {a b : α} : AntisymmRel r a b → AntisymmRel r b a :=
  And.symm


@[trans]
theorem AntisymmRel.trans [IsTrans α r] {a b c : α} (hab : AntisymmRel r a b)
    (hbc : AntisymmRel r b c) : AntisymmRel r a c :=
  ⟨_root_.trans hab.1 hbc.1, _root_.trans hbc.2 hab.2⟩


instance AntisymmRel.decidableRel [DecidableRel r] : DecidableRel (AntisymmRel r) := fun _ _ =>
  instDecidableAnd


@[simp]
theorem antisymmRel_iff_eq [IsRefl α r] [IsAntisymm α r] {a b : α} : AntisymmRel r a b ↔ a = b :=
  antisymm_iff


alias ⟨AntisymmRel.eq, _⟩ := antisymmRel_iff_eq


/-- The antisymmetrization relation as an equivalence relation. -/
@[simps]
def AntisymmRel.setoid : Setoid α :=
  ⟨AntisymmRel r, antisymmRel_refl _, AntisymmRel.symm, AntisymmRel.trans⟩


/-- The partial order derived from a preorder by making pairwise comparable elements equal. This is
the quotient by `fun a b => a ≤ b ∧ b ≤ a`. -/
def Antisymmetrization : Type _ :=
  Quotient <| AntisymmRel.setoid α r


/-- Turn an element into its antisymmetrization. -/
def toAntisymmetrization : α → Antisymmetrization α r :=
  Quotient.mk _


/-- Get a representative from the antisymmetrization. -/
noncomputable def ofAntisymmetrization : Antisymmetrization α r → α :=
  Quotient.out


instance [Inhabited α] : Inhabited (Antisymmetrization α r) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    inst✝¹ : IsPreorder α r
    inst✝ : Inhabited α
    ⊢ Inhabited (Antisymmetrization α r)
  -/
  unfold Antisymmetrization; infer_instance
                             /-
                               🎉 no goals
                             -/


@[elab_as_elim]
protected theorem Antisymmetrization.ind {p : Antisymmetrization α r → Prop} :
    (∀ a, p <| toAntisymmetrization r a) → ∀ q, p q :=
  Quot.ind


@[elab_as_elim]
protected theorem Antisymmetrization.induction_on {p : Antisymmetrization α r → Prop}
    (a : Antisymmetrization α r) (h : ∀ a, p <| toAntisymmetrization r a) : p a :=
  Quotient.inductionOn' a h


@[simp]
theorem toAntisymmetrization_ofAntisymmetrization (a : Antisymmetrization α r) :
    toAntisymmetrization r (ofAntisymmetrization r a) = a :=
  Quotient.out_eq' _


theorem AntisymmRel.image {a b : α} (h : AntisymmRel (· ≤ ·) a b) {f : α → β} (hf : Monotone f) :
    AntisymmRel (· ≤ ·) (f a) (f b) :=
  ⟨hf h.1, hf h.2⟩


instance instPartialOrderAntisymmetrization : PartialOrder (Antisymmetrization α (· ≤ ·)) where
  le :=
    Quotient.lift₂ (· ≤ ·) fun (_ _ _ _ : α) h₁ h₂ =>
      propext ⟨fun h => h₁.2.trans <| h.trans h₂.1, fun h => h₁.1.trans <| h.trans h₂.2⟩
  lt :=
    Quotient.lift₂ (· < ·) fun (_ _ _ _ : α) h₁ h₂ =>
      propext ⟨fun h => h₁.2.trans_lt <| h.trans_le h₂.1, fun h =>
                h₁.1.trans_lt <| h.trans_le h₂.2⟩
  le_refl a := Quotient.inductionOn' a le_refl
  le_trans a b c := Quotient.inductionOn₃' a b c fun _ _ _ => le_trans
  lt_iff_le_not_le a b := Quotient.inductionOn₂' a b fun _ _ => lt_iff_le_not_le
  le_antisymm a b := Quotient.inductionOn₂' a b fun _ _ hab hba => Quotient.sound' ⟨hab, hba⟩


theorem antisymmetrization_fibration :
    Relation.Fibration (· < ·) (· < ·) (@toAntisymmetrization α (· ≤ ·) _) := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Relation.Fibration (fun x1 x2 => LT.lt x1 x2) (fun x1 x2 => LT.lt x1 x2) (to …
  -/
  rintro a ⟨b⟩ h
  /-
    case mk
    α : Type u_1
    inst✝ : Preorder α
    a : α
    b✝ : Antisymmetrization α fun x1 x2 => LE.le x1 x2
    b : α
    h : LT.lt (Quot.mk (⇑(AntisymmRel.setoid α fun x1 x2 => LE.le x1 x2)) b) (toAn …
    ⊢ Exists fun a' => And ((fun x1 x2 => LT.lt x1 x2) a' a) (Eq (toAntisymmetriza …
  -/
  exact ⟨b, h, rfl⟩
  /-
    🎉 no goals
  -/


theorem acc_antisymmetrization_iff : Acc (· < ·)
    (@toAntisymmetrization α (· ≤ ·) _ a) ↔ Acc (· < ·) a :=
  acc_lift₂_iff


theorem wellFounded_antisymmetrization_iff :
    WellFounded (@LT.lt (Antisymmetrization α (· ≤ ·)) _) ↔ WellFounded (@LT.lt α _) :=
  wellFounded_lift₂_iff


theorem wellFoundedLT_antisymmetrization_iff :
    WellFoundedLT (Antisymmetrization α (· ≤ ·)) ↔ WellFoundedLT α := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (WellFoundedLT (Antisymmetrization α fun x1 x2 => LE.le x1 x2)) (WellFou …
  -/
  simp_rw [isWellFounded_iff, wellFounded_antisymmetrization_iff]
  /-
    🎉 no goals
  -/


theorem wellFoundedGT_antisymmetrization_iff :
    WellFoundedGT (Antisymmetrization α (· ≤ ·)) ↔ WellFoundedGT α := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (WellFoundedGT (Antisymmetrization α fun x1 x2 => LE.le x1 x2)) (WellFou …
  -/
  simp_rw [isWellFounded_iff]
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (WellFounded fun x1 x2 => GT.gt x1 x2) (WellFounded fun x1 x2 => GT.gt x …
  -/
  convert wellFounded_liftOn₂'_iff with ⟨_⟩ ⟨_⟩
  exact fun _ _ _ _ h₁ h₂ ↦ propext
    ⟨fun h ↦ (h₂.2.trans_lt h).trans_le h₁.1, fun h ↦ (h₂.1.trans_lt h).trans_le h₁.2⟩


instance [WellFoundedLT α] : WellFoundedLT (Antisymmetrization α (· ≤ ·)) :=
  wellFoundedLT_antisymmetrization_iff.mpr ‹_›


instance [WellFoundedGT α] : WellFoundedGT (Antisymmetrization α (· ≤ ·)) :=
  wellFoundedGT_antisymmetrization_iff.mpr ‹_›


instance [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)] [IsTotal α (· ≤ ·)] :
    LinearOrder (Antisymmetrization α (· ≤ ·)) :=
  { instPartialOrderAntisymmetrization with
    le_total := fun a b => Quotient.inductionOn₂' a b <| total_of (· ≤ ·),
    decidableLE := fun _ _ => show Decidable (Quotient.liftOn₂' _ _ _ _) from inferInstance,
    decidableLT := fun _ _ => show Decidable (Quotient.liftOn₂' _ _ _ _) from inferInstance }


@[simp]
theorem toAntisymmetrization_le_toAntisymmetrization_iff :
    @toAntisymmetrization α (· ≤ ·) _ a ≤ @toAntisymmetrization α (· ≤ ·) _ b ↔ a ≤ b :=
  Iff.rfl


@[simp]
theorem toAntisymmetrization_lt_toAntisymmetrization_iff :
    @toAntisymmetrization α (· ≤ ·) _ a < @toAntisymmetrization α (· ≤ ·) _ b ↔ a < b :=
  Iff.rfl


@[simp]
theorem ofAntisymmetrization_le_ofAntisymmetrization_iff {a b : Antisymmetrization α (· ≤ ·)} :
    ofAntisymmetrization (· ≤ ·) a ≤ ofAntisymmetrization (· ≤ ·) b ↔ a ≤ b :=
  (Quotient.outRelEmbedding _).map_rel_iff


@[simp]
theorem ofAntisymmetrization_lt_ofAntisymmetrization_iff {a b : Antisymmetrization α (· ≤ ·)} :
    ofAntisymmetrization (· ≤ ·) a < ofAntisymmetrization (· ≤ ·) b ↔ a < b :=
  (Quotient.outRelEmbedding _).map_rel_iff


@[mono]
theorem toAntisymmetrization_mono : Monotone (@toAntisymmetrization α (· ≤ ·) _) := fun _ _ => id


private theorem liftFun_antisymmRel (f : α →o β) :
    ((AntisymmRel.setoid α (· ≤ ·)).r ⇒ (AntisymmRel.setoid β (· ≤ ·)).r) f f := fun _ _ h =>
  ⟨f.mono h.1, f.mono h.2⟩


/-- Turns an order homomorphism from `α` to `β` into one from `Antisymmetrization α` to
`Antisymmetrization β`. `Antisymmetrization` is actually a functor. See `Preorder_to_PartialOrder`.
-/
protected def OrderHom.antisymmetrization (f : α →o β) :
    Antisymmetrization α (· ≤ ·) →o Antisymmetrization β (· ≤ ·) :=
  ⟨Quotient.map' f <| liftFun_antisymmRel f, fun a b => Quotient.inductionOn₂' a b <| f.mono⟩


@[simp]
theorem OrderHom.coe_antisymmetrization (f : α →o β) :
    ⇑f.antisymmetrization = Quotient.map' f (liftFun_antisymmRel f) :=
  rfl

/- Porting note: Removed @[simp] attribute. With this `simp` lemma the LHS of
`OrderHom.antisymmetrization_apply_mk` is not in normal-form -/

theorem OrderHom.antisymmetrization_apply (f : α →o β) (a : Antisymmetrization α (· ≤ ·)) :
    f.antisymmetrization a = Quotient.map' f (liftFun_antisymmRel f) a :=
  rfl


@[simp]
theorem OrderHom.antisymmetrization_apply_mk (f : α →o β) (a : α) :
    f.antisymmetrization (toAntisymmetrization _ a) = toAntisymmetrization _ (f a) :=
  @Quotient.map_mk _ _ (_root_.id _) (_root_.id _) f (liftFun_antisymmRel f) _


/-- `ofAntisymmetrization` as an order embedding. -/
@[simps]
noncomputable def OrderEmbedding.ofAntisymmetrization : Antisymmetrization α (· ≤ ·) ↪o α :=
  { Quotient.outRelEmbedding _ with toFun := _root_.ofAntisymmetrization _ }


/-- `Antisymmetrization` and `orderDual` commute. -/
def OrderIso.dualAntisymmetrization :
    (Antisymmetrization α (· ≤ ·))ᵒᵈ ≃o Antisymmetrization αᵒᵈ (· ≤ ·) where
  toFun := (Quotient.map' id) fun _ _ => And.symm
  invFun := (Quotient.map' id) fun _ _ => And.symm
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      inst✝¹ : Preorder α
                                                      inst✝ : Preorder β
                                                      a✝¹ b : α
                                                      a✝ : OrderDual (Antisymmetrization α fun x1 x2 => LE.le x1 x2)
                                                      a : α
                                                      ⊢ Eq (Quotient.map' id ⋯ (Quotient.map' id ⋯ (Quotient.mk'' a))) (Quotient.mk' …
                                                    -/
  left_inv a := Quotient.inductionOn' a fun a => by simp_rw [Quotient.map'_mk'', id]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                     /-
                                                       α : Type u_1
                                                       β : Type u_2
                                                       inst✝¹ : Preorder α
                                                       inst✝ : Preorder β
                                                       a✝¹ b : α
                                                       a✝ : Antisymmetrization (OrderDual α) fun x1 x2 => LE.le x1 x2
                                                       a : OrderDual α
                                                       ⊢ Eq (Quotient.map' id ⋯ (Quotient.map' id ⋯ (Quotient.mk'' a))) (Quotient.mk' …
                                                     -/
  right_inv a := Quotient.inductionOn' a fun a => by simp_rw [Quotient.map'_mk'', id]
                                                     /-
                                                       🎉 no goals
                                                     -/
  map_rel_iff' := @fun a b => Quotient.inductionOn₂' a b fun _ _ => Iff.rfl


@[simp]
theorem OrderIso.dualAntisymmetrization_apply (a : α) :
    OrderIso.dualAntisymmetrization _ (toDual <| toAntisymmetrization _ a) =
      toAntisymmetrization _ (toDual a) :=
  rfl


@[simp]
theorem OrderIso.dualAntisymmetrization_symm_apply (a : α) :
    (OrderIso.dualAntisymmetrization _).symm (toAntisymmetrization _ <| toDual a) =
      toDual (toAntisymmetrization _ a) :=
  rfl


/-- The antisymmetrization of a product preorder is order isomorphic
to the product of antisymmetrizations. -/
def prodEquiv : Antisymmetrization (α × β) (· ≤ ·) ≃o
    Antisymmetrization α (· ≤ ·) × Antisymmetrization β (· ≤ ·) where
  toFun := Quotient.lift (fun ab ↦ (⟦ab.1⟧, ⟦ab.2⟧)) fun ab₁ ab₂ h ↦
    Prod.mk.inj_iff.mpr ⟨Quotient.sound ⟨h.1.1, h.2.1⟩, Quotient.sound ⟨h.1.2, h.2.2⟩⟩
  invFun := Function.uncurry <| Quotient.lift₂ (fun a b ↦ ⟦(a, b)⟧)
    fun a₁ b₁ a₂ b₂ h₁ h₂ ↦ Quotient.sound ⟨⟨h₁.1, h₂.1⟩, h₁.2, h₂.2⟩
                 /-
                   α : Type u_1
                   β : Type u_2
                   inst✝¹ : Preorder α
                   inst✝ : Preorder β
                   ⊢ Function.LeftInverse (Function.uncurry (Quotient.lift₂ (fun a b => Quotient. …
                 -/
  left_inv := by rintro ⟨_⟩; rfl
                             /-
                               🎉 no goals
                             -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝¹ : Preorder α
                    inst✝ : Preorder β
                    ⊢ Function.RightInverse (Function.uncurry (Quotient.lift₂ (fun a b => Quotient …
                  -/
  right_inv := by rintro ⟨⟨_⟩, ⟨_⟩⟩; rfl
                                     /-
                                       🎉 no goals
                                     -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝¹ : Preorder α
                       inst✝ : Preorder β
                       ⊢ ∀ {a b : Antisymmetrization (Prod α β) fun x1 x2 => LE.le x1 x2}, Iff (LE.le …
                     -/
  map_rel_iff' := by rintro ⟨_⟩ ⟨_⟩; rfl
                                     /-
                                       🎉 no goals
                                     -/


@[simp] lemma prodEquiv_apply_mk {ab} : prodEquiv α β ⟦ab⟧ = (⟦ab.1⟧, ⟦ab.2⟧) := rfl

@[simp] lemma prodEquiv_symm_apply_mk {a b} : (prodEquiv α β).symm (⟦a⟧, ⟦b⟧) = ⟦(a, b)⟧ := rfl


instance Prod.wellFoundedLT [WellFoundedLT α] [WellFoundedLT β] : WellFoundedLT (α × β) :=
  wellFoundedLT_antisymmetrization_iff.mp <|
    (Antisymmetrization.prodEquiv α β).strictMono.wellFoundedLT


instance Prod.wellFoundedGT [WellFoundedGT α] [WellFoundedGT β] : WellFoundedGT (α × β) :=
  wellFoundedGT_antisymmetrization_iff.mp <|
    (Antisymmetrization.prodEquiv α β).strictMono.wellFoundedGT


