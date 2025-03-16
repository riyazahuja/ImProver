/-- `Finsupp.Lex r s` is the lexicographic relation on `α →₀ N`, where `α` is ordered by `r`,
and `N` is ordered by `s`.

The type synonym `Lex (α →₀ N)` has an order given by `Finsupp.Lex (· < ·) (· < ·)`.
-/
protected def Lex (r : α → α → Prop) (s : N → N → Prop) (x y : α →₀ N) : Prop :=
  Pi.Lex r s x y

-- Porting note: Added `_root_` to better align with Lean 3.

theorem _root_.Pi.lex_eq_finsupp_lex {r : α → α → Prop} {s : N → N → Prop} (a b : α →₀ N) :
    Pi.Lex r s a b = Finsupp.Lex r s a b :=
  rfl


theorem lex_def {r : α → α → Prop} {s : N → N → Prop} {a b : α →₀ N} :
    Finsupp.Lex r s a b ↔ ∃ j, (∀ d, r d j → a d = b d) ∧ s (a j) (b j) :=
  Iff.rfl


theorem lex_eq_invImage_dfinsupp_lex (r : α → α → Prop) (s : N → N → Prop) :
    Finsupp.Lex r s = InvImage (DFinsupp.Lex r fun _ ↦ s) toDFinsupp :=
  rfl


instance [LT α] [LT N] : LT (Lex (α →₀ N)) :=
  ⟨fun f g ↦ Finsupp.Lex (· < ·) (· < ·) (ofLex f) (ofLex g)⟩


theorem lex_lt_of_lt_of_preorder [Preorder N] (r) [IsStrictOrder α r] {x y : α →₀ N} (hlt : x < y) :
    ∃ i, (∀ j, r j i → x j ≤ y j ∧ y j ≤ x j) ∧ x i < y i :=
  DFinsupp.lex_lt_of_lt_of_preorder r (id hlt : x.toDFinsupp < y.toDFinsupp)


theorem lex_lt_of_lt [PartialOrder N] (r) [IsStrictOrder α r] {x y : α →₀ N} (hlt : x < y) :
    Pi.Lex r (· < ·) x y :=
  DFinsupp.lex_lt_of_lt r (id hlt : x.toDFinsupp < y.toDFinsupp)


instance Lex.isStrictOrder [LinearOrder α] [PartialOrder N] :
    IsStrictOrder (Lex (α →₀ N)) (· < ·) where
  irrefl _ := lt_irrefl (α := Lex (α → N)) _
  trans _ _ _ := lt_trans (α := Lex (α → N))


/-- The partial order on `Finsupp`s obtained by the lexicographic ordering.
See `Finsupp.Lex.linearOrder` for a proof that this partial order is in fact linear. -/
instance Lex.partialOrder [PartialOrder N] : PartialOrder (Lex (α →₀ N)) where
  lt := (· < ·)
  le x y := ⇑(ofLex x) = ⇑(ofLex y) ∨ x < y
  __ := PartialOrder.lift (fun x : Lex (α →₀ N) ↦ toLex (⇑(ofLex x)))
    (DFunLike.coe_injective (F := Finsupp α N))


/-- The linear order on `Finsupp`s obtained by the lexicographic ordering. -/
instance Lex.linearOrder [LinearOrder N] : LinearOrder (Lex (α →₀ N)) where
  lt := (· < ·)
  le := (· ≤ ·)
  __ := LinearOrder.lift' (toLex ∘ toDFinsupp ∘ ofLex) finsuppEquivDFinsupp.injective


theorem Lex.single_strictAnti : StrictAnti (fun (a : α) ↦ toLex (single a 1)) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    ⊢ StrictAnti fun a => toLex (Finsupp.single a 1)
  -/
  intro a b h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LT.lt a b
    ⊢ LT.lt ((fun a => toLex (Finsupp.single a 1)) b) ((fun a => toLex (Finsupp.si …
  -/
  simp only [LT.lt, Finsupp.lex_def]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LT.lt a b
    ⊢ Exists fun j => And (∀ (d : α), LT.lt d j → Eq ((ofLex (toLex (Finsupp.singl …
  -/
  simp only [ofLex_toLex, Nat.lt_eq]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LT.lt a b
    ⊢ Exists fun j => And (∀ (d : α), LT.lt d j → Eq ((Finsupp.single b 1) d) ((Fi …
  -/
  use a
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LT.lt a b
    ⊢ And (∀ (d : α), LT.lt d a → Eq ((Finsupp.single b 1) d) ((Finsupp.single a 1 …
  -/
  constructor
    /-
      case h.left
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : LT.lt a b
      ⊢ ∀ (d : α), LT.lt d a → Eq ((Finsupp.single b 1) d) ((Finsupp.single a 1) d)
    -/
  · intro d hd
    simp only [Finsupp.single_eq_of_ne (ne_of_lt hd).symm,
      Finsupp.single_eq_of_ne (ne_of_gt (lt_trans hd h))]
    /-
      case h.right
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : LT.lt a b
      ⊢ LT.lt ((Finsupp.single b 1) a) ((Finsupp.single a 1) a)
    -/
  · simp only [single_eq_same, single_eq_of_ne (ne_of_lt h).symm, zero_lt_one]
    /-
      🎉 no goals
    -/


theorem Lex.single_lt_iff {a b : α} : toLex (single b 1) < toLex (single a 1) ↔ a < b :=
  Lex.single_strictAnti.lt_iff_lt


theorem Lex.single_le_iff {a b : α} : toLex (single b 1) ≤ toLex (single a 1) ↔ a ≤ b :=
  Lex.single_strictAnti.le_iff_le


theorem Lex.single_antitone : Antitone (fun (a : α) ↦ toLex (single a 1)) :=
  Lex.single_strictAnti.antitone


theorem toLex_monotone : Monotone (@toLex (α →₀ N)) :=
  fun a b h ↦ DFinsupp.toLex_monotone (id h : ∀ i, ofLex (toDFinsupp a) i ≤ ofLex (toDFinsupp b) i)


theorem lt_of_forall_lt_of_lt (a b : Lex (α →₀ N)) (i : α) :
    (∀ j < i, ofLex a j = ofLex b j) → ofLex a i < ofLex b i → a < b :=
  fun h1 h2 ↦ ⟨i, h1, h2⟩


instance Lex.addLeftStrictMono : AddLeftStrictMono (Lex (α →₀ N)) :=
  ⟨fun _ _ _ ⟨a, lta, ha⟩ ↦ ⟨a, fun j ja ↦ congr_arg _ (lta j ja), add_lt_add_left ha _⟩⟩


instance Lex.addLeftMono : AddLeftMono (Lex (α →₀ N)) :=
  addLeftMono_of_addLeftStrictMono _


instance Lex.addRightStrictMono : AddRightStrictMono (Lex (α →₀ N)) :=
  ⟨fun f _ _ ⟨a, lta, ha⟩ ↦
    ⟨a, fun j ja ↦ congr_arg (· + ofLex f j) (lta j ja), add_lt_add_right ha _⟩⟩


instance Lex.addRightMono : AddRightMono (Lex (α →₀ N)) :=
  addRightMono_of_addRightStrictMono _


instance Lex.orderBot [CanonicallyOrderedAddCommMonoid N] : OrderBot (Lex (α →₀ N)) where
  bot := 0
  bot_le _ := Finsupp.toLex_monotone bot_le


noncomputable instance Lex.orderedAddCancelCommMonoid [OrderedCancelAddCommMonoid N] :
    OrderedCancelAddCommMonoid (Lex (α →₀ N)) where
  add_le_add_left _ _ h _ := add_le_add_left (α := Lex (α → N)) h _
  le_of_add_le_add_left _ _ _ := le_of_add_le_add_left (α := Lex (α → N))


noncomputable instance Lex.orderedAddCommGroup [OrderedAddCommGroup N] :
    OrderedAddCommGroup (Lex (α →₀ N)) where
  add_le_add_left _ _ := add_le_add_left


noncomputable instance Lex.linearOrderedCancelAddCommMonoid [LinearOrderedCancelAddCommMonoid N] :
    LinearOrderedCancelAddCommMonoid (Lex (α →₀ N)) where
  __ : LinearOrder (Lex (α →₀ N)) := inferInstance
  __ : OrderedCancelAddCommMonoid (Lex (α →₀ N)) := inferInstance


noncomputable instance Lex.linearOrderedAddCommGroup [LinearOrderedAddCommGroup N] :
    LinearOrderedAddCommGroup (Lex (α →₀ N)) where
  __ : LinearOrder (Lex (α →₀ N)) := inferInstance
  add_le_add_left _ _ := add_le_add_left


