/-- Bundled structure registering a well order on a type. Ordinals will be defined as a quotient
of this type. -/
structure WellOrder : Type (u + 1) where
  /-- The underlying type of the order. -/
  α : Type u
  /-- The underlying relation of the order. -/
  r : α → α → Prop
  /-- The proposition that `r` is a well-ordering for `α`. -/
  wo : IsWellOrder α r


instance inhabited : Inhabited WellOrder :=
  ⟨⟨PEmpty, _, inferInstanceAs (IsWellOrder PEmpty EmptyRelation)⟩⟩


@[deprecated "No deprecation message was provided." (since := "2024-10-24")]
theorem eta (o : WellOrder) : mk o.α o.r o.wo = o := rfl


/-- Equivalence relation on well orders on arbitrary types in universe `u`, given by order
isomorphism. -/
instance Ordinal.isEquivalent : Setoid WellOrder where
  r := fun ⟨_, r, _⟩ ⟨_, s, _⟩ => Nonempty (r ≃r s)
  iseqv :=
    ⟨fun _ => ⟨RelIso.refl _⟩, fun ⟨e⟩ => ⟨e.symm⟩, fun ⟨e₁⟩ ⟨e₂⟩ => ⟨e₁.trans e₂⟩⟩


/-- `Ordinal.{u}` is the type of well orders in `Type u`, up to order isomorphism. -/
@[pp_with_univ]
def Ordinal : Type (u + 1) :=
  Quotient Ordinal.isEquivalent


/-- A "canonical" type order-isomorphic to the ordinal `o`, living in the same universe. This is
defined through the axiom of choice.

Use this over `Iio o` only when it is paramount to have a `Type u` rather than a `Type (u + 1)`. -/
def Ordinal.toType (o : Ordinal.{u}) : Type u :=
  o.out.α


instance hasWellFounded_toType (o : Ordinal) : WellFoundedRelation o.toType :=
  ⟨o.out.r, o.out.wo.wf⟩


instance linearOrder_toType (o : Ordinal) : LinearOrder o.toType :=
  @IsWellOrder.linearOrder _ o.out.r o.out.wo


instance wellFoundedLT_toType_lt (o : Ordinal) : WellFoundedLT o.toType :=
  o.out.wo.toIsWellFounded


/-- The order type of a well order is an ordinal. -/
def type (r : α → α → Prop) [wo : IsWellOrder α r] : Ordinal :=
  ⟦⟨α, r, wo⟩⟧


/-- `typeLT α` is an abbreviation for the order type of the `<` relation of `α`. -/
scoped notation "typeLT " α:70 => @Ordinal.type α (· < ·) inferInstance


instance zero : Zero Ordinal :=
  ⟨type <| @EmptyRelation PEmpty⟩


instance inhabited : Inhabited Ordinal :=
  ⟨0⟩


instance one : One Ordinal :=
  ⟨type <| @EmptyRelation PUnit⟩


@[deprecated "Avoid using `Quotient.mk` to construct an `Ordinal` directly."
  (since := "2024-10-24")]
theorem type_def' (w : WellOrder) : ⟦w⟧ = type w.r := rfl



@[deprecated "Avoid using `Quotient.mk` to construct an `Ordinal` directly."
  (since := "2024-10-24")]
theorem type_def (r) [wo : IsWellOrder α r] : (⟦⟨α, r, wo⟩⟧ : Ordinal) = type r := rfl


@[simp]
theorem type_toType (o : Ordinal) : typeLT o.toType = o :=
  o.out_eq


@[deprecated type_toType (since := "2024-10-22")]
theorem type_lt (o : Ordinal) : typeLT o.toType = o :=
  o.out_eq


@[deprecated type_toType (since := "2024-08-26")]
theorem type_out (o : Ordinal) : Ordinal.type o.out.r = o :=
  type_toType o


theorem type_eq {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r] [IsWellOrder β s] :
    type r = type s ↔ Nonempty (r ≃r s) :=
  Quotient.eq'


theorem _root_.RelIso.ordinal_type_eq {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r]
    [IsWellOrder β s] (h : r ≃r s) : type r = type s :=
  type_eq.2 ⟨h⟩


theorem type_eq_zero_of_empty (r) [IsWellOrder α r] [IsEmpty α] : type r = 0 :=
  (RelIso.relIsoOfIsEmpty r _).ordinal_type_eq


@[simp]
theorem type_eq_zero_iff_isEmpty [IsWellOrder α r] : type r = 0 ↔ IsEmpty α :=
  ⟨fun h =>
    let ⟨s⟩ := type_eq.1 h
    s.toEquiv.isEmpty,
    @type_eq_zero_of_empty α r _⟩


                                                                                    /-
                                                                                      α : Type u
                                                                                      r : α → α → Prop
                                                                                      inst✝ : IsWellOrder α r
                                                                                      ⊢ Iff (Ne (Ordinal.type r) 0) (Nonempty α)
                                                                                    -/
theorem type_ne_zero_iff_nonempty [IsWellOrder α r] : type r ≠ 0 ↔ Nonempty α := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem type_ne_zero_of_nonempty (r) [IsWellOrder α r] [h : Nonempty α] : type r ≠ 0 :=
  type_ne_zero_iff_nonempty.2 h


theorem type_pEmpty : type (@EmptyRelation PEmpty) = 0 :=
  rfl


theorem type_empty : type (@EmptyRelation Empty) = 0 :=
  type_eq_zero_of_empty _


theorem type_eq_one_of_unique (r) [IsWellOrder α r] [Nonempty α] [Subsingleton α] : type r = 1 := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝² : IsWellOrder α r
    inst✝¹ : Nonempty α
    inst✝ : Subsingleton α
    ⊢ Eq (Ordinal.type r) 1
  -/
  cases nonempty_unique α
  /-
    case intro
    α : Type u
    r : α → α → Prop
    inst✝² : IsWellOrder α r
    inst✝¹ : Nonempty α
    inst✝ : Subsingleton α
    val✝ : Unique α
    ⊢ Eq (Ordinal.type r) 1
  -/
  exact (RelIso.ofUniqueOfIrrefl r _).ordinal_type_eq
  /-
    🎉 no goals
  -/


@[simp]
theorem type_eq_one_iff_unique [IsWellOrder α r] : type r = 1 ↔ Nonempty (Unique α) :=
  ⟨fun h ↦ let ⟨s⟩ := type_eq.1 h; ⟨s.toEquiv.unique⟩,
    fun ⟨_⟩ ↦ type_eq_one_of_unique r⟩


theorem type_pUnit : type (@EmptyRelation PUnit) = 1 :=
  rfl


theorem type_unit : type (@EmptyRelation Unit) = 1 :=
  rfl


@[simp]
theorem toType_empty_iff_eq_zero {o : Ordinal} : IsEmpty o.toType ↔ o = 0 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (IsEmpty o.toType) (Eq o 0)
  -/
  rw [← @type_eq_zero_iff_isEmpty o.toType (· < ·), type_toType]
  /-
    🎉 no goals
  -/


@[deprecated toType_empty_iff_eq_zero (since := "2024-08-26")]
alias out_empty_iff_eq_zero := toType_empty_iff_eq_zero


@[deprecated toType_empty_iff_eq_zero (since := "2024-08-26")]
theorem eq_zero_of_out_empty (o : Ordinal) [h : IsEmpty o.toType] : o = 0 :=
  toType_empty_iff_eq_zero.1 h


instance isEmpty_toType_zero : IsEmpty (toType 0) :=
  toType_empty_iff_eq_zero.2 rfl


@[simp]
theorem toType_nonempty_iff_ne_zero {o : Ordinal} : Nonempty o.toType ↔ o ≠ 0 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (Nonempty o.toType) (Ne o 0)
  -/
  rw [← @type_ne_zero_iff_nonempty o.toType (· < ·), type_toType]
  /-
    🎉 no goals
  -/


@[deprecated toType_nonempty_iff_ne_zero (since := "2024-08-26")]
alias out_nonempty_iff_ne_zero := toType_nonempty_iff_ne_zero


@[deprecated toType_nonempty_iff_ne_zero (since := "2024-08-26")]
theorem ne_zero_of_out_nonempty (o : Ordinal) [h : Nonempty o.toType] : o ≠ 0 :=
  toType_nonempty_iff_ne_zero.1 h


protected theorem one_ne_zero : (1 : Ordinal) ≠ 0 :=
  type_ne_zero_of_nonempty _


instance nontrivial : Nontrivial Ordinal.{u} :=
  ⟨⟨1, 0, Ordinal.one_ne_zero⟩⟩


/-- `Quotient.inductionOn` specialized to ordinals.

Not to be confused with well-founded recursion `Ordinal.induction`. -/
@[elab_as_elim]
theorem inductionOn {C : Ordinal → Prop} (o : Ordinal)
    (H : ∀ (α r) [IsWellOrder α r], C (type r)) : C o :=
  Quot.inductionOn o fun ⟨α, r, wo⟩ => @H α r wo


/-- `Quotient.inductionOn₂` specialized to ordinals.

Not to be confused with well-founded recursion `Ordinal.induction`. -/
@[elab_as_elim]
theorem inductionOn₂ {C : Ordinal → Ordinal → Prop} (o₁ o₂ : Ordinal)
    (H : ∀ (α r) [IsWellOrder α r] (β s) [IsWellOrder β s], C (type r) (type s)) : C o₁ o₂ :=
  Quotient.inductionOn₂ o₁ o₂ fun ⟨α, r, wo₁⟩ ⟨β, s, wo₂⟩ => @H α r wo₁ β s wo₂


/-- `Quotient.inductionOn₃` specialized to ordinals.

Not to be confused with well-founded recursion `Ordinal.induction`. -/
@[elab_as_elim]
theorem inductionOn₃ {C : Ordinal → Ordinal → Ordinal → Prop} (o₁ o₂ o₃ : Ordinal)
    (H : ∀ (α r) [IsWellOrder α r] (β s) [IsWellOrder β s] (γ t) [IsWellOrder γ t],
      C (type r) (type s) (type t)) : C o₁ o₂ o₃ :=
  Quotient.inductionOn₃ o₁ o₂ o₃ fun ⟨α, r, wo₁⟩ ⟨β, s, wo₂⟩ ⟨γ, t, wo₃⟩ =>
    @H α r wo₁ β s wo₂ γ t wo₃


open Classical in
/-- To prove a result on ordinals, it suffices to prove it for order types of well-orders. -/
@[elab_as_elim]
theorem inductionOnWellOrder {C : Ordinal → Prop} (o : Ordinal)
    (H : ∀ (α) [LinearOrder α] [WellFoundedLT α], C (typeLT α)) : C o :=
  inductionOn o fun α r wo ↦ @H α (linearOrderOfSTO r) wo.toIsWellFounded


open Classical in
/-- To define a function on ordinals, it suffices to define them on order types of well-orders.

Since `LinearOrder` is data-carrying, `liftOnWellOrder_type` is not a definitional equality, unlike
`Quotient.liftOn_mk` which is always def-eq. -/
def liftOnWellOrder {δ : Sort v} (o : Ordinal) (f : ∀ (α) [LinearOrder α] [WellFoundedLT α], δ)
    (c : ∀ (α) [LinearOrder α] [WellFoundedLT α] (β) [LinearOrder β] [WellFoundedLT β],
      typeLT α = typeLT β → f α = f β) : δ :=
  Quotient.liftOn o (fun w ↦ @f w.α (linearOrderOfSTO w.r) w.wo.toIsWellFounded)
    fun w₁ w₂ h ↦ @c
      w₁.α (linearOrderOfSTO w₁.r) w₁.wo.toIsWellFounded
      w₂.α (linearOrderOfSTO w₂.r) w₂.wo.toIsWellFounded
      (Quotient.sound h)


@[simp]
theorem liftOnWellOrder_type {δ : Sort v} (f : ∀ (α) [LinearOrder α] [WellFoundedLT α], δ)
    (c : ∀ (α) [LinearOrder α] [WellFoundedLT α] (β) [LinearOrder β] [WellFoundedLT β],
      typeLT α = typeLT β → f α = f β) {γ} [LinearOrder γ] [WellFoundedLT γ] :
    liftOnWellOrder (typeLT γ) f c = f γ := by
  /-
    δ : Sort v
    f : (α : Type u_1) → [inst : LinearOrder α] → [inst : WellFoundedLT α] → δ
    c : ∀ (α : Type u_1) [inst : LinearOrder α] [inst_1 : WellFoundedLT α] (β : Ty …
    γ : Type u_1
    inst✝¹ : LinearOrder γ
    inst✝ : WellFoundedLT γ
    ⊢ Eq ((Ordinal.type fun x1 x2 => LT.lt x1 x2).liftOnWellOrder f c) (f γ)
  -/
  change Quotient.liftOn' ⟦_⟧ _ _ = _
  /-
    δ : Sort v
    f : (α : Type u_1) → [inst : LinearOrder α] → [inst : WellFoundedLT α] → δ
    c : ∀ (α : Type u_1) [inst : LinearOrder α] [inst_1 : WellFoundedLT α] (β : Ty …
    γ : Type u_1
    inst✝¹ : LinearOrder γ
    inst✝ : WellFoundedLT γ
    ⊢ Eq ((Quotient.mk Ordinal.isEquivalent { α := γ, r := fun x1 x2 => LT.lt x1 x …
  -/
  rw [Quotient.liftOn'_mk]
  /-
    δ : Sort v
    f : (α : Type u_1) → [inst : LinearOrder α] → [inst : WellFoundedLT α] → δ
    c : ∀ (α : Type u_1) [inst : LinearOrder α] [inst_1 : WellFoundedLT α] (β : Ty …
    γ : Type u_1
    inst✝¹ : LinearOrder γ
    inst✝ : WellFoundedLT γ
    ⊢ Eq (f { α := γ, r := fun x1 x2 => LT.lt x1 x2, wo := ⋯ }.α) (f γ)
  -/
  congr
  /-
    case h.e_2.h
    δ : Sort v
    f : (α : Type u_1) → [inst : LinearOrder α] → [inst : WellFoundedLT α] → δ
    c : ∀ (α : Type u_1) [inst : LinearOrder α] [inst_1 : WellFoundedLT α] (β : Ty …
    γ : Type u_1
    inst✝¹ : LinearOrder γ
    inst✝ : WellFoundedLT γ
    ⊢ Eq (linearOrderOfSTO { α := γ, r := fun x1 x2 => LT.lt x1 x2, wo := ⋯ }.r) i …
  -/
  exact LinearOrder.ext_lt fun _ _ ↦ Iff.rfl
  /-
    🎉 no goals
  -/


/--
For `Ordinal`:

* less-equal is defined such that well orders `r` and `s` satisfy `type r ≤ type s` if there exists
  a function embedding `r` as an *initial* segment of `s`.
* less-than is defined such that well orders `r` and `s` satisfy `type r < type s` if there exists
  a function embedding `r` as a *principal* segment of `s`.

Note that most of the relevant results on initial and principal segments are proved in the
`Order.InitialSeg` file.
-/
instance partialOrder : PartialOrder Ordinal where
  le a b :=
    Quotient.liftOn₂ a b (fun ⟨_, r, _⟩ ⟨_, s, _⟩ => Nonempty (r ≼i s))
      fun _ _ _ _ ⟨f⟩ ⟨g⟩ => propext
        ⟨fun ⟨h⟩ => ⟨f.symm.toInitialSeg.trans <| h.trans g.toInitialSeg⟩, fun ⟨h⟩ =>
          ⟨f.toInitialSeg.trans <| h.trans g.symm.toInitialSeg⟩⟩
  lt a b :=
    Quotient.liftOn₂ a b (fun ⟨_, r, _⟩ ⟨_, s, _⟩ => Nonempty (r ≺i s))
      fun _ _ _ _ ⟨f⟩ ⟨g⟩ => propext
        ⟨fun ⟨h⟩ => ⟨PrincipalSeg.relIsoTrans f.symm <| h.transRelIso g⟩,
          fun ⟨h⟩ => ⟨PrincipalSeg.relIsoTrans f <| h.transRelIso g.symm⟩⟩
  le_refl := Quot.ind fun ⟨_, _, _⟩ => ⟨InitialSeg.refl _⟩
  le_trans a b c :=
    Quotient.inductionOn₃ a b c fun _ _ _ ⟨f⟩ ⟨g⟩ => ⟨f.trans g⟩
  lt_iff_le_not_le a b :=
    Quotient.inductionOn₂ a b fun _ _ =>
      ⟨fun ⟨f⟩ => ⟨⟨f⟩, fun ⟨g⟩ => (f.transInitial g).irrefl⟩, fun ⟨⟨f⟩, h⟩ =>
        f.principalSumRelIso.recOn (fun g => ⟨g⟩) fun g => (h ⟨g.symm.toInitialSeg⟩).elim⟩
  le_antisymm a b :=
    Quotient.inductionOn₂ a b fun _ _ ⟨h₁⟩ ⟨h₂⟩ =>
      Quot.sound ⟨InitialSeg.antisymm h₁ h₂⟩


instance : LinearOrder Ordinal :=
  {inferInstanceAs (PartialOrder Ordinal) with
    le_total := fun a b => Quotient.inductionOn₂ a b fun ⟨_, r, _⟩ ⟨_, s, _⟩ =>
      (InitialSeg.total r s).recOn (fun f => Or.inl ⟨f⟩) fun f => Or.inr ⟨f⟩
    decidableLE := Classical.decRel _ }


theorem _root_.InitialSeg.ordinal_type_le {α β} {r : α → α → Prop} {s : β → β → Prop}
    [IsWellOrder α r] [IsWellOrder β s] (h : r ≼i s) : type r ≤ type s :=
  ⟨h⟩


theorem _root_.RelEmbedding.ordinal_type_le {α β} {r : α → α → Prop} {s : β → β → Prop}
    [IsWellOrder α r] [IsWellOrder β s] (h : r ↪r s) : type r ≤ type s :=
  ⟨h.collapse⟩


theorem _root_.PrincipalSeg.ordinal_type_lt {α β} {r : α → α → Prop} {s : β → β → Prop}
    [IsWellOrder α r] [IsWellOrder β s] (h : r ≺i s) : type r < type s :=
  ⟨h⟩


@[simp]
protected theorem zero_le (o : Ordinal) : 0 ≤ o :=
  inductionOn o fun _ r _ => (InitialSeg.ofIsEmpty _ r).ordinal_type_le


instance : OrderBot Ordinal where
  bot := 0
  bot_le := Ordinal.zero_le


@[simp]
theorem bot_eq_zero : (⊥ : Ordinal) = 0 :=
  rfl


@[simp]
protected theorem le_zero {o : Ordinal} : o ≤ 0 ↔ o = 0 :=
  le_bot_iff


protected theorem pos_iff_ne_zero {o : Ordinal} : 0 < o ↔ o ≠ 0 :=
  bot_lt_iff_ne_bot


protected theorem not_lt_zero (o : Ordinal) : ¬o < 0 :=
  not_lt_bot


theorem eq_zero_or_pos : ∀ a : Ordinal, a = 0 ∨ 0 < a :=
  eq_bot_or_bot_lt


instance : ZeroLEOneClass Ordinal :=
  ⟨Ordinal.zero_le _⟩


instance instNeZeroOne : NeZero (1 : Ordinal) :=
  ⟨Ordinal.one_ne_zero⟩


theorem type_le_iff {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r]
    [IsWellOrder β s] : type r ≤ type s ↔ Nonempty (r ≼i s) :=
  Iff.rfl


theorem type_le_iff' {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r]
    [IsWellOrder β s] : type r ≤ type s ↔ Nonempty (r ↪r s) :=
  ⟨fun ⟨f⟩ => ⟨f⟩, fun ⟨f⟩ => ⟨f.collapse⟩⟩


theorem type_lt_iff {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r]
    [IsWellOrder β s] : type r < type s ↔ Nonempty (r ≺i s) :=
  Iff.rfl


/-- Given two ordinals `α ≤ β`, then `initialSegToType α β` is the initial segment embedding of
`α.toType` into `β.toType`. -/
def initialSegToType {α β : Ordinal} (h : α ≤ β) : α.toType ≤i β.toType := by
  /-
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r : α✝ → α✝ → Prop
    s : β✝ → β✝ → Prop
    t : γ → γ → Prop
    α β : Ordinal.{?u.49906}
    h : LE.le α β
    ⊢ InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
  -/
  apply Classical.choice (type_le_iff.mp _)
  /-
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r : α✝ → α✝ → Prop
    s : β✝ → β✝ → Prop
    t : γ → γ → Prop
    α β : Ordinal.{?u.49906}
    h : LE.le α β
    ⊢ LE.le (Ordinal.type fun x1 x2 => LT.lt x1 x2) (Ordinal.type fun x1 x2 => LT. …
  -/
  rwa [type_toType, type_toType]
  /-
    🎉 no goals
  -/


@[deprecated initialSegToType (since := "2024-08-26")]
noncomputable alias initialSegOut := initialSegToType


/-- Given two ordinals `α < β`, then `principalSegToType α β` is the principal segment embedding
of `α.toType` into `β.toType`. -/
def principalSegToType {α β : Ordinal} (h : α < β) : α.toType <i β.toType := by
  /-
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r : α✝ → α✝ → Prop
    s : β✝ → β✝ → Prop
    t : γ → γ → Prop
    α β : Ordinal.{?u.51132}
    h : LT.lt α β
    ⊢ PrincipalSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
  -/
  apply Classical.choice (type_lt_iff.mp _)
  /-
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r : α✝ → α✝ → Prop
    s : β✝ → β✝ → Prop
    t : γ → γ → Prop
    α β : Ordinal.{?u.51132}
    h : LT.lt α β
    ⊢ LT.lt (Ordinal.type fun x1 x2 => LT.lt x1 x2) (Ordinal.type fun x1 x2 => LT. …
  -/
  rwa [type_toType, type_toType]
  /-
    🎉 no goals
  -/


@[deprecated principalSegToType (since := "2024-08-26")]
noncomputable alias principalSegOut := principalSegToType


/-- The order type of an element inside a well order.

This is registered as a principal segment embedding into the ordinals, with top `type r`. -/
def typein (r : α → α → Prop) [IsWellOrder α r] : @PrincipalSeg α Ordinal.{u} r (· < ·) := by
  refine ⟨RelEmbedding.ofMonotone _ fun a b ha ↦
    ((PrincipalSeg.ofElement r a).codRestrict _ ?_ ?_).ordinal_type_lt, type r, fun a ↦ ⟨?_, ?_⟩⟩
    /-
      case refine_1
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a b : α
      ha : r a b
      ⊢ ∀ (a_1 : Subtype fun b => r b a), Membership.mem (setOf fun b_1 => r b_1 b)  …
    -/
  · rintro ⟨c, hc⟩
    /-
      case refine_1.mk
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a b : α
      ha : r a b
      c : α
      hc : r c a
      ⊢ Membership.mem (setOf fun b_1 => r b_1 b) ((PrincipalSeg.ofElement r a).toRe …
    -/
    exact trans hc ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a b : α
      ha : r a b
      ⊢ Membership.mem (setOf fun b_1 => r b_1 b) (PrincipalSeg.ofElement r a).top
    -/
  · exact ha
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a : Ordinal.{u}
      ⊢ Membership.mem (Set.range ⇑(RelEmbedding.ofMonotone (fun a => Ordinal.type ( …
    -/
  · rintro ⟨b, rfl⟩
    /-
      case refine_3.intro
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      b : α
      ⊢ LT.lt ((RelEmbedding.ofMonotone (fun a => Ordinal.type (Subrel r (setOf fun  …
    -/
    exact (PrincipalSeg.ofElement _ _).ordinal_type_lt
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a : Ordinal.{u}
      ⊢ LT.lt a (Ordinal.type r) → Membership.mem (Set.range ⇑(RelEmbedding.ofMonoto …
    -/
  · refine inductionOn a ?_
    /-
      case refine_4
      α : Type u
      β : Type v
      γ : Type w
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a : Ordinal.{u}
      ⊢ ∀ (α_1 : Type u) (r_1 : α_1 → α_1 → Prop) [inst : IsWellOrder α_1 r_1], LT.l …
    -/
    rintro β s wo ⟨g⟩
    /-
      case refine_4.intro
      α : Type u
      β✝ : Type v
      γ : Type w
      r✝ : α → α → Prop
      s✝ : β✝ → β✝ → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      a : Ordinal.{u}
      β : Type u
      s : β → β → Prop
      wo : IsWellOrder β s
      g : PrincipalSeg s r
      ⊢ Membership.mem (Set.range ⇑(RelEmbedding.ofMonotone (fun a => Ordinal.type ( …
    -/
    exact ⟨_, g.subrelIso.ordinal_type_eq⟩
    /-
      🎉 no goals
    -/


@[deprecated typein (since := "2024-10-09")]
alias typein.principalSeg := typein


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-09")]
theorem typein.principalSeg_coe (r : α → α → Prop) [IsWellOrder α r] :
    (typein.principalSeg r : α → Ordinal) = typein r :=
  rfl


@[simp]
theorem type_subrel (r : α → α → Prop) [IsWellOrder α r] (a : α) :
    type (Subrel r { b | r b a }) = typein r a :=
  rfl


@[simp]
theorem top_typein (r : α → α → Prop) [IsWellOrder α r] : (typein r).top = type r :=
  rfl


theorem typein_lt_type (r : α → α → Prop) [IsWellOrder α r] (a : α) : typein r a < type r :=
  (typein r).lt_top a


theorem typein_lt_self {o : Ordinal} (i : o.toType) : typein (α := o.toType) (· < ·) i < o := by
  /-
    o : Ordinal.{u_1}
    i : o.toType
    ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding i) o
  -/
  simp_rw [← type_toType o]
  /-
    o : Ordinal.{u_1}
    i : o.toType
    ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding i) (Ordinal. …
  -/
  apply typein_lt_type
  /-
    🎉 no goals
  -/


@[simp]
theorem typein_top {α β} {r : α → α → Prop} {s : β → β → Prop}
    [IsWellOrder α r] [IsWellOrder β s] (f : r ≺i s) : typein s f.top = type r :=
  f.subrelIso.ordinal_type_eq


@[simp]
theorem typein_lt_typein (r : α → α → Prop) [IsWellOrder α r] {a b : α} :
    typein r a < typein r b ↔ r a b :=
  (typein r).map_rel_iff


@[simp]
theorem typein_le_typein (r : α → α → Prop) [IsWellOrder α r] {a b : α} :
    typein r a ≤ typein r b ↔ ¬r b a := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    a b : α
    ⊢ Iff (LE.le ((Ordinal.typein r).toRelEmbedding a) ((Ordinal.typein r).toRelEm …
  -/
  rw [← not_lt, typein_lt_typein]
  /-
    🎉 no goals
  -/


theorem typein_injective (r : α → α → Prop) [IsWellOrder α r] : Injective (typein r) :=
  (typein r).injective


theorem typein_inj (r : α → α → Prop) [IsWellOrder α r] {a b} : typein r a = typein r b ↔ a = b :=
  (typein_injective r).eq_iff


theorem mem_range_typein_iff (r : α → α → Prop) [IsWellOrder α r] {o} :
    o ∈ Set.range (typein r) ↔ o < type r :=
  (typein r).mem_range_iff_rel


theorem typein_surj (r : α → α → Prop) [IsWellOrder α r] {o} (h : o < type r) :
    o ∈ Set.range (typein r) :=
  (typein r).mem_range_of_rel_top h


theorem typein_surjOn (r : α → α → Prop) [IsWellOrder α r] :
    Set.SurjOn (typein r) Set.univ (Set.Iio (type r)) :=
  (typein r).surjOn


/-- A well order `r` is order-isomorphic to the set of ordinals smaller than `type r`.
`enum r ⟨o, h⟩` is the `o`-th element of `α` ordered by `r`.

That is, `enum` maps an initial segment of the ordinals, those less than the order type of `r`, to
the elements of `α`. -/
-- The explicit typing is required in order for `simp` to work properly.
@[simps! symm_apply_coe]
def enum (r : α → α → Prop) [IsWellOrder α r] :
    @RelIso { o // o < type r } α (Subrel (· < ·) { o | o < type r }) r :=
  (typein r).subrelIso


@[simp]
theorem typein_enum (r : α → α → Prop) [IsWellOrder α r] {o} (h : o < type r) :
    typein r (enum r ⟨o, h⟩) = o :=
  (typein r).apply_subrelIso _


theorem enum_type {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r] [IsWellOrder β s]
    (f : s ≺i r) {h : type s < type r} : enum r ⟨type s, h⟩ = f.top :=
  (typein r).injective <| (typein_enum _ _).trans (typein_top _).symm


@[simp]
theorem enum_typein (r : α → α → Prop) [IsWellOrder α r] (a : α) :
    enum r ⟨typein r a, typein_lt_type r a⟩ = a :=
  enum_type (PrincipalSeg.ofElement r a)


theorem enum_lt_enum {r : α → α → Prop} [IsWellOrder α r] {o₁ o₂ : {o // o < type r}} :
    r (enum r o₁) (enum r o₂) ↔ o₁ < o₂ :=
  (enum _).map_rel_iff


theorem enum_le_enum (r : α → α → Prop) [IsWellOrder α r] {o₁ o₂ : {o // o < type r}} :
    ¬r (enum r o₁) (enum r o₂) ↔ o₂ ≤ o₁ := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    o₁ o₂ : Subtype fun o => LT.lt o (Ordinal.type r)
    ⊢ Iff (Not (r ((Ordinal.enum r) o₁) ((Ordinal.enum r) o₂))) (LE.le o₂ o₁)
  -/
  rw [enum_lt_enum (r := r), not_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem enum_le_enum' (a : Ordinal) {o₁ o₂ : {o // o < type (· < ·)}} :
    enum (· < ·) o₁ ≤ enum (α := a.toType) (· < ·) o₂ ↔ o₁ ≤ o₂ := by
  /-
    a : Ordinal.{u_1}
    o₁ o₂ : Subtype fun o => LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Iff (LE.le ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) o₁) ((Ordinal.enum fun x …
  -/
  rw [← enum_le_enum, not_lt]
  /-
    🎉 no goals
  -/


theorem enum_inj {r : α → α → Prop} [IsWellOrder α r] {o₁ o₂ : {o // o < type r}} :
    enum r o₁ = enum r o₂ ↔ o₁ = o₂ :=
  EmbeddingLike.apply_eq_iff_eq _


theorem enum_zero_le {r : α → α → Prop} [IsWellOrder α r] (h0 : 0 < type r) (a : α) :
    ¬r a (enum r ⟨0, h0⟩) := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    h0 : LT.lt 0 (Ordinal.type r)
    a : α
    ⊢ Not (r a ((Ordinal.enum r) ⟨0, h0⟩))
  -/
  rw [← enum_typein r a, enum_le_enum r]
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    h0 : LT.lt 0 (Ordinal.type r)
    a : α
    ⊢ LE.le ⟨0, h0⟩ ⟨(Ordinal.typein r).toRelEmbedding a, ⋯⟩
  -/
  apply Ordinal.zero_le
  /-
    🎉 no goals
  -/


theorem enum_zero_le' {o : Ordinal} (h0 : 0 < o) (a : o.toType) :
    enum (α := o.toType) (· < ·) ⟨0, type_toType _ ▸ h0⟩ ≤ a := by
  /-
    o : Ordinal.{u_1}
    h0 : LT.lt 0 o
    a : o.toType
    ⊢ LE.le ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨0, ⋯⟩) a
  -/
  rw [← not_lt]
  /-
    o : Ordinal.{u_1}
    h0 : LT.lt 0 o
    a : o.toType
    ⊢ Not (LT.lt a ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨0, ⋯⟩))
  -/
  apply enum_zero_le
  /-
    🎉 no goals
  -/


theorem relIso_enum' {α β : Type u} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r]
    [IsWellOrder β s] (f : r ≃r s) (o : Ordinal) :
    ∀ (hr : o < type r) (hs : o < type s), f (enum r ⟨o, hr⟩) = enum s ⟨o, hs⟩ := by
  /-
    α β : Type u
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    f : RelIso r s
    o : Ordinal.{u}
    ⊢ ∀ (hr : LT.lt o (Ordinal.type r)) (hs : LT.lt o (Ordinal.type s)), Eq (f ((O …
  -/
  refine inductionOn o ?_; rintro γ t wo ⟨g⟩ ⟨h⟩
  /-
    case intro.intro
    α β : Type u
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    f : RelIso r s
    o : Ordinal.{u}
    γ : Type u
    t : γ → γ → Prop
    wo : IsWellOrder γ t
    g : PrincipalSeg t r
    h : PrincipalSeg t s
    ⊢ Eq (f ((Ordinal.enum r) ⟨Ordinal.type t, ⋯⟩)) ((Ordinal.enum s) ⟨Ordinal.typ …
  -/
  rw [enum_type g, enum_type (g.transRelIso f)]; rfl
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem relIso_enum {α β : Type u} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r]
    [IsWellOrder β s] (f : r ≃r s) (o : Ordinal) (hr : o < type r) :
    f (enum r ⟨o, hr⟩) = enum s ⟨o, hr.trans_eq (Quotient.sound ⟨f⟩)⟩ :=
  relIso_enum' _ _ _ _


/-- The order isomorphism between ordinals less than `o` and `o.toType`. -/
@[simps! (config := .lemmasOnly)]
noncomputable def enumIsoToType (o : Ordinal) : Set.Iio o ≃o o.toType where
  toFun x := enum (α := o.toType) (· < ·) ⟨x.1, type_toType _ ▸ x.2⟩
  invFun x := ⟨typein (α := o.toType) (· < ·) x, typein_lt_self x⟩
  left_inv _ := Subtype.ext_val (typein_enum _ _)
  right_inv _ := enum_typein _ _
  map_rel_iff' := enum_le_enum' _


@[deprecated "No deprecation message was provided."  (since := "2024-08-26")]
alias enumIsoOut := enumIsoToType


instance small_Iio (o : Ordinal.{u}) : Small.{u} (Iio o) :=
  ⟨_, ⟨(enumIsoToType _).toEquiv⟩⟩


instance small_Iic (o : Ordinal.{u}) : Small.{u} (Iic o) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    o : Ordinal.{u}
    ⊢ Small.{u, u + 1} ↑(Set.Iic o)
  -/
  rw [← Iio_union_right]
  /-
    α : Type u
    β : Type v
    γ : Type w
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    o : Ordinal.{u}
    ⊢ Small.{u, u + 1} ↑(Union.union (Set.Iio o) (Singleton.singleton o))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance small_Ico (a b : Ordinal.{u}) : Small.{u} (Ico a b) := small_subset Ico_subset_Iio_self

instance small_Icc (a b : Ordinal.{u}) : Small.{u} (Icc a b) := small_subset Icc_subset_Iic_self

instance small_Ioo (a b : Ordinal.{u}) : Small.{u} (Ioo a b) := small_subset Ioo_subset_Iio_self

instance small_Ioc (a b : Ordinal.{u}) : Small.{u} (Ioc a b) := small_subset Ioc_subset_Iic_self


/-- `o.toType` is an `OrderBot` whenever `0 < o`. -/
def toTypeOrderBotOfPos {o : Ordinal} (ho : 0 < o) : OrderBot o.toType where
  bot_le := enum_zero_le' ho


@[deprecated toTypeOrderBotOfPos (since := "2024-08-26")]
noncomputable alias outOrderBotOfPos := toTypeOrderBotOfPos


theorem enum_zero_eq_bot {o : Ordinal} (ho : 0 < o) :
                                        /-
                                          α : Type u
                                          β : Type v
                                          γ : Type w
                                          r : α → α → Prop
                                          s : β → β → Prop
                                          t : γ → γ → Prop
                                          o : Ordinal.{?u.75110}
                                          ho : LT.lt 0 o
                                          ⊢ LT.lt 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
                                        -/
    enum (α := o.toType) (· < ·) ⟨0, by rwa [type_toType]⟩ =
                                        /-
                                          🎉 no goals
                                        -/
      have H := toTypeOrderBotOfPos ho
      (⊥ : o.toType) :=
  rfl


theorem lt_wf : @WellFounded Ordinal (· < ·) :=
  wellFounded_iff_wellFounded_subrel.mpr (·.induction_on fun ⟨_, _, wo⟩ ↦
    RelHomClass.wellFounded (enum _) wo.wf)


instance wellFoundedRelation : WellFoundedRelation Ordinal :=
  ⟨(· < ·), lt_wf⟩


instance wellFoundedLT : WellFoundedLT Ordinal :=
  ⟨lt_wf⟩


instance : ConditionallyCompleteLinearOrderBot Ordinal :=
  WellFoundedLT.conditionallyCompleteLinearOrderBot _


/-- Reformulation of well founded induction on ordinals as a lemma that works with the
`induction` tactic, as in `induction i using Ordinal.induction with | h i IH => ?_`. -/
theorem induction {p : Ordinal.{u} → Prop} (i : Ordinal.{u}) (h : ∀ j, (∀ k, k < j → p k) → p j) :
    p i :=
  lt_wf.induction i h


theorem typein_apply {α β} {r : α → α → Prop} {s : β → β → Prop} [IsWellOrder α r] [IsWellOrder β s]
    (f : r ≼i s) (a : α) : typein s (f a) = typein r a := by
  /-
    α β : Type u_1
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    a : α
    ⊢ Eq ((Ordinal.typein s).toRelEmbedding (f a)) ((Ordinal.typein r).toRelEmbedd …
  -/
  rw [← f.transPrincipal_apply _ a, (f.transPrincipal _).eq]
  /-
    🎉 no goals
  -/


/-- The cardinal of an ordinal is the cardinality of any type on which a relation with that order
type is defined. -/
def card : Ordinal → Cardinal :=
  Quotient.map WellOrder.α fun _ _ ⟨e⟩ => ⟨e.toEquiv⟩


@[simp]
theorem card_type (r : α → α → Prop) [IsWellOrder α r] : card (type r) = #α :=
  rfl


@[simp]
theorem card_typein {r : α → α → Prop} [IsWellOrder α r] (x : α) :
    #{ y // r y x } = (typein r x).card :=
  rfl


theorem card_le_card {o₁ o₂ : Ordinal} : o₁ ≤ o₂ → card o₁ ≤ card o₂ :=
  inductionOn o₁ fun _ _ _ => inductionOn o₂ fun _ _ _ ⟨⟨⟨f, _⟩, _⟩⟩ => ⟨f⟩


@[simp]
theorem card_zero : card 0 = 0 := mk_eq_zero _


@[simp]
theorem card_one : card 1 = 1 := mk_eq_one _


/-- The universe lift operation for ordinals, which embeds `Ordinal.{u}` as
  a proper initial segment of `Ordinal.{v}` for `v > u`. For the initial segment version,
  see `liftInitialSeg`. -/
@[pp_with_univ]
def lift (o : Ordinal.{v}) : Ordinal.{max v u} :=
  Quotient.liftOn o (fun w => type <| ULift.down.{u} ⁻¹'o w.r) fun ⟨_, r, _⟩ ⟨_, s, _⟩ ⟨f⟩ =>
    Quot.sound
      ⟨(RelIso.preimage Equiv.ulift r).trans <| f.trans (RelIso.preimage Equiv.ulift s).symm⟩


@[simp]
theorem type_uLift (r : α → α → Prop) [IsWellOrder α r] :
    type (ULift.down ⁻¹'o r) = lift.{v} (type r) :=
  rfl


theorem _root_.RelIso.ordinal_lift_type_eq {r : α → α → Prop} {s : β → β → Prop}
    [IsWellOrder α r] [IsWellOrder β s] (f : r ≃r s) : lift.{v} (type r) = lift.{u} (type s) :=
  ((RelIso.preimage Equiv.ulift r).trans <|
      f.trans (RelIso.preimage Equiv.ulift s).symm).ordinal_type_eq


@[simp]
theorem type_preimage {α β : Type u} (r : α → α → Prop) [IsWellOrder α r] (f : β ≃ α) :
    type (f ⁻¹'o r) = type r :=
  (RelIso.preimage f r).ordinal_type_eq


@[simp]
theorem type_lift_preimage (r : α → α → Prop) [IsWellOrder α r]
    (f : β ≃ α) : lift.{u} (type (f ⁻¹'o r)) = lift.{v} (type r) :=
  (RelIso.preimage f r).ordinal_lift_type_eq


@[deprecated type_lift_preimage_aux (since := "2024-10-22")]
theorem type_lift_preimage_aux (r : α → α → Prop) [IsWellOrder α r] (f : β ≃ α) :
    lift.{u} (@type _ (fun x y => r (f x) (f y))
      (inferInstanceAs (IsWellOrder β (f ⁻¹'o r)))) = lift.{v} (type r) :=
  type_lift_preimage r f


/-- `lift.{max u v, u}` equals `lift.{v, u}`.

Unfortunately, the simp lemma doesn't seem to work. -/
theorem lift_umax : lift.{max u v, u} = lift.{v, u} :=
  funext fun a =>
    inductionOn a fun _ r _ =>
      Quotient.sound ⟨(RelIso.preimage Equiv.ulift r).trans (RelIso.preimage Equiv.ulift r).symm⟩


/-- `lift.{max v u, u}` equals `lift.{v, u}`.

Unfortunately, the simp lemma doesn't seem to work. -/
@[deprecated lift_umax (since := "2024-10-24")]
theorem lift_umax' : lift.{max v u, u} = lift.{v, u} :=
  lift_umax


/-- An ordinal lifted to a lower or equal universe equals itself.

Unfortunately, the simp lemma doesn't work. -/
theorem lift_id' (a : Ordinal) : lift a = a :=
  inductionOn a fun _ r _ => Quotient.sound ⟨RelIso.preimage Equiv.ulift r⟩


/-- An ordinal lifted to the same universe equals itself. -/
@[simp]
theorem lift_id : ∀ a, lift.{u, u} a = a :=
  lift_id'.{u, u}


/-- An ordinal lifted to the zero universe equals itself. -/
@[simp]
theorem lift_uzero (a : Ordinal.{u}) : lift.{0} a = a :=
  lift_id' a


theorem lift_type_le {α : Type u} {β : Type v} {r s} [IsWellOrder α r] [IsWellOrder β s] :
    lift.{max v w} (type r) ≤ lift.{max u w} (type s) ↔ Nonempty (r ≼i s) := by
  /-
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    ⊢ Iff (LE.le (Ordinal.lift.{max v w, u} (Ordinal.type r)) (Ordinal.lift.{max u …
  -/
  constructor <;> refine fun ⟨f⟩ ↦ ⟨?_⟩
  · exact (RelIso.preimage Equiv.ulift r).symm.toInitialSeg.trans
      (f.trans (RelIso.preimage Equiv.ulift s).toInitialSeg)
  · exact (RelIso.preimage Equiv.ulift r).toInitialSeg.trans
      (f.trans (RelIso.preimage Equiv.ulift s).symm.toInitialSeg)


theorem lift_type_eq {α : Type u} {β : Type v} {r s} [IsWellOrder α r] [IsWellOrder β s] :
    lift.{max v w} (type r) = lift.{max u w} (type s) ↔ Nonempty (r ≃r s) := by
  /-
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    ⊢ Iff (Eq (Ordinal.lift.{max v w, u} (Ordinal.type r)) (Ordinal.lift.{max u w, …
  -/
  refine Quotient.eq'.trans ⟨?_, ?_⟩ <;> refine fun ⟨f⟩ ↦ ⟨?_⟩
    /-
      case refine_1
      α : Type u
      β : Type v
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      x✝ : Ordinal.isEquivalent { α := ULift.{max v w, u} { α := α, r := r, wo := in …
      f : RelIso (Order.Preimage ULift.down { α := α, r := r, wo := inst✝¹ }.r) (Ord …
      ⊢ RelIso r s
    -/
  · exact (RelIso.preimage Equiv.ulift r).symm.trans <| f.trans (RelIso.preimage Equiv.ulift s)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      x✝ : Nonempty (RelIso r s)
      f : RelIso r s
      ⊢ RelIso (Order.Preimage ULift.down { α := α, r := r, wo := inst✝¹ }.r) (Order …
    -/
  · exact (RelIso.preimage Equiv.ulift r).trans <| f.trans (RelIso.preimage Equiv.ulift s).symm
    /-
      🎉 no goals
    -/


theorem lift_type_lt {α : Type u} {β : Type v} {r s} [IsWellOrder α r] [IsWellOrder β s] :
    lift.{max v w} (type r) < lift.{max u w} (type s) ↔ Nonempty (r ≺i s) := by
  /-
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    ⊢ Iff (LT.lt (Ordinal.lift.{max v w, u} (Ordinal.type r)) (Ordinal.lift.{max u …
  -/
  constructor <;> refine fun ⟨f⟩ ↦ ⟨?_⟩
  · exact (f.relIsoTrans (RelIso.preimage Equiv.ulift r).symm).transInitial
      (RelIso.preimage Equiv.ulift s).toInitialSeg
  · exact (f.relIsoTrans (RelIso.preimage Equiv.ulift r)).transInitial
      (RelIso.preimage Equiv.ulift s).symm.toInitialSeg


@[simp]
theorem lift_le {a b : Ordinal} : lift.{u, v} a ≤ lift.{u, v} b ↔ a ≤ b :=
  inductionOn₂ a b fun α r _ β s _ => by
    /-
      a b : Ordinal.{v}
      α : Type v
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type v
      s : β → β → Prop
      x✝ : IsWellOrder β s
      ⊢ Iff (LE.le (Ordinal.lift.{u, v} (Ordinal.type r)) (Ordinal.lift.{u, v} (Ordi …
    -/
    rw [← lift_umax]
    /-
      a b : Ordinal.{v}
      α : Type v
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type v
      s : β → β → Prop
      x✝ : IsWellOrder β s
      ⊢ Iff (LE.le (Ordinal.lift.{max v u, v} (Ordinal.type r)) (Ordinal.lift.{max v …
    -/
    exact lift_type_le.{_,_,u}
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_inj {a b : Ordinal} : lift.{u, v} a = lift.{u, v} b ↔ a = b := by
  /-
    a b : Ordinal.{v}
    ⊢ Iff (Eq (Ordinal.lift.{u, v} a) (Ordinal.lift.{u, v} b)) (Eq a b)
  -/
  simp_rw [le_antisymm_iff, lift_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_lt {a b : Ordinal} : lift.{u, v} a < lift.{u, v} b ↔ a < b := by
  /-
    a b : Ordinal.{v}
    ⊢ Iff (LT.lt (Ordinal.lift.{u, v} a) (Ordinal.lift.{u, v} b)) (LT.lt a b)
  -/
  simp_rw [lt_iff_le_not_le, lift_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_typein_top {r : α → α → Prop} {s : β → β → Prop}
    [IsWellOrder α r] [IsWellOrder β s] (f : r ≺i s) : lift.{u} (typein s f.top) = lift (type r) :=
  f.subrelIso.ordinal_lift_type_eq


/-- Initial segment version of the lift operation on ordinals, embedding `Ordinal.{u}` in
`Ordinal.{v}` as an initial segment when `u ≤ v`. -/
def liftInitialSeg : Ordinal.{v} ≤i Ordinal.{max u v} := by
  refine ⟨RelEmbedding.ofMonotone lift.{u} (by simp),
    fun a b ↦ Ordinal.inductionOn₂ a b fun α r _ β s _ h ↦ ?_⟩
  rw [RelEmbedding.ofMonotone_coe, ← lift_id'.{max u v} (type s),
    ← lift_umax.{v, u}, lift_type_lt] at h
  /-
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r✝ : α✝ → α✝ → Prop
    s✝ : β✝ → β✝ → Prop
    t : γ → γ → Prop
    a : Ordinal.{v}
    b : Ordinal.{max u v}
    α : Type v
    r : α → α → Prop
    x✝¹ : IsWellOrder α r
    β : Type (max u v)
    s : β → β → Prop
    x✝ : IsWellOrder β s
    h : Nonempty (PrincipalSeg s r)
    ⊢ Membership.mem (Set.range ⇑(RelEmbedding.ofMonotone Ordinal.lift.{u, v} ⋯))  …
  -/
  obtain ⟨f⟩ := h
  /-
    case intro
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r✝ : α✝ → α✝ → Prop
    s✝ : β✝ → β✝ → Prop
    t : γ → γ → Prop
    a : Ordinal.{v}
    b : Ordinal.{max u v}
    α : Type v
    r : α → α → Prop
    x✝¹ : IsWellOrder α r
    β : Type (max u v)
    s : β → β → Prop
    x✝ : IsWellOrder β s
    f : PrincipalSeg s r
    ⊢ Membership.mem (Set.range ⇑(RelEmbedding.ofMonotone Ordinal.lift.{u, v} ⋯))  …
  -/
  use typein r f.top
  /-
    case h
    α✝ : Type u
    β✝ : Type v
    γ : Type w
    r✝ : α✝ → α✝ → Prop
    s✝ : β✝ → β✝ → Prop
    t : γ → γ → Prop
    a : Ordinal.{v}
    b : Ordinal.{max u v}
    α : Type v
    r : α → α → Prop
    x✝¹ : IsWellOrder α r
    β : Type (max u v)
    s : β → β → Prop
    x✝ : IsWellOrder β s
    f : PrincipalSeg s r
    ⊢ Eq ((RelEmbedding.ofMonotone Ordinal.lift.{u, v} ⋯) ((Ordinal.typein r).toRe …
  -/
  rw [RelEmbedding.ofMonotone_coe, ← lift_umax, lift_typein_top, lift_id']
  /-
    🎉 no goals
  -/


@[deprecated liftInitialSeg (since := "2024-09-21")]
alias lift.initialSeg := liftInitialSeg


@[simp]
theorem liftInitialSeg_coe : (liftInitialSeg.{v, u} : Ordinal → Ordinal) = lift.{v, u} :=
  rfl


set_option linter.deprecated false in
@[deprecated liftInitialSeg_coe (since := "2024-09-21")]
theorem lift.initialSeg_coe : (lift.initialSeg.{v, u} : Ordinal → Ordinal) = lift.{v, u} :=
  rfl


@[simp]
theorem lift_lift (a : Ordinal.{u}) : lift.{w} (lift.{v} a) = lift.{max v w} a :=
  (liftInitialSeg.trans liftInitialSeg).eq liftInitialSeg a


@[simp]
theorem lift_zero : lift 0 = 0 :=
  type_eq_zero_of_empty _


@[simp]
theorem lift_one : lift 1 = 1 :=
  type_eq_one_of_unique _


@[simp]
theorem lift_card (a) : Cardinal.lift.{u, v} (card a) = card (lift.{u} a) :=
  inductionOn a fun _ _ _ => rfl


theorem mem_range_lift_of_le {a : Ordinal.{u}} {b : Ordinal.{max u v}} (h : b ≤ lift.{v} a) :
    b ∈ Set.range lift.{v} :=
  liftInitialSeg.mem_range_of_le h


@[deprecated mem_range_lift_of_le (since := "2024-10-07")]
theorem lift_down {a : Ordinal.{u}} {b : Ordinal.{max u v}} (h : b ≤ lift.{v,u} a) :
    ∃ a', lift.{v,u} a' = b :=
  mem_range_lift_of_le h


theorem le_lift_iff {a : Ordinal.{u}} {b : Ordinal.{max u v}} :
    b ≤ lift.{v} a ↔ ∃ a' ≤ a, lift.{v} a' = b :=
  liftInitialSeg.le_apply_iff


theorem lt_lift_iff {a : Ordinal.{u}} {b : Ordinal.{max u v}} :
    b < lift.{v} a ↔ ∃ a' < a, lift.{v} a' = b :=
  liftInitialSeg.lt_apply_iff


/-- `ω` is the first infinite ordinal, defined as the order type of `ℕ`. -/
def omega0 : Ordinal.{u} :=
  lift (typeLT ℕ)


@[inherit_doc]
scoped notation "ω" => Ordinal.omega0


/-- Note that the presence of this lemma makes `simp [omega0]` form a loop. -/
@[simp]
theorem type_nat_lt : typeLT ℕ = ω :=
  (lift_id _).symm


@[simp]
theorem card_omega0 : card ω = ℵ₀ :=
  rfl


@[simp]
theorem lift_omega0 : lift ω = ω :=
  lift_lift _


/-- `o₁ + o₂` is the order on the disjoint union of `o₁` and `o₂` obtained by declaring that
every element of `o₁` is smaller than every element of `o₂`. -/
instance add : Add Ordinal.{u} :=
  ⟨fun o₁ o₂ => Quotient.liftOn₂ o₁ o₂ (fun ⟨_, r, _⟩ ⟨_, s, _⟩ => type (Sum.Lex r s))
    fun _ _ _ _ ⟨f⟩ ⟨g⟩ => (RelIso.sumLexCongr f g).ordinal_type_eq⟩


instance addMonoidWithOne : AddMonoidWithOne Ordinal.{u} where
  add := (· + ·)
  zero := 0
  one := 1
  zero_add o :=
    inductionOn o fun α _ _ =>
      Eq.symm <| Quotient.sound ⟨⟨(emptySum PEmpty α).symm, Sum.lex_inr_inr⟩⟩
  add_zero o :=
    inductionOn o fun α _ _ =>
      Eq.symm <| Quotient.sound ⟨⟨(sumEmpty α PEmpty).symm, Sum.lex_inl_inl⟩⟩
  add_assoc o₁ o₂ o₃ :=
    Quotient.inductionOn₃ o₁ o₂ o₃ fun ⟨α, r, _⟩ ⟨β, s, _⟩ ⟨γ, t, _⟩ =>
      Quot.sound
        ⟨⟨sumAssoc _ _ _, by
          /-
            α✝ : Type u
            β✝ : Type v
            γ✝ : Type w
            r✝ : α✝ → α✝ → Prop
            s✝ : β✝ → β✝ → Prop
            t✝ : γ✝ → γ✝ → Prop
            o₁ o₂ o₃ : Ordinal.{u}
            x✝² x✝¹ x✝ : WellOrder
            α : Type u
            r : α → α → Prop
            wo✝² : IsWellOrder α r
            β : Type u
            s : β → β → Prop
            wo✝¹ : IsWellOrder β s
            γ : Type u
            t : γ → γ → Prop
            wo✝ : IsWellOrder γ t
            ⊢ ∀ {a b : Sum (Sum α β) γ}, Iff (Sum.Lex r (Sum.Lex s t) ((Equiv.sumAssoc α β …
          -/
          intros a b
          /-
            α✝ : Type u
            β✝ : Type v
            γ✝ : Type w
            r✝ : α✝ → α✝ → Prop
            s✝ : β✝ → β✝ → Prop
            t✝ : γ✝ → γ✝ → Prop
            o₁ o₂ o₃ : Ordinal.{u}
            x✝² x✝¹ x✝ : WellOrder
            α : Type u
            r : α → α → Prop
            wo✝² : IsWellOrder α r
            β : Type u
            s : β → β → Prop
            wo✝¹ : IsWellOrder β s
            γ : Type u
            t : γ → γ → Prop
            wo✝ : IsWellOrder γ t
            a b : Sum (Sum α β) γ
            ⊢ Iff (Sum.Lex r (Sum.Lex s t) ((Equiv.sumAssoc α β γ) a) ((Equiv.sumAssoc α β …
          -/
          rcases a with (⟨a | a⟩ | a) <;> rcases b with (⟨b | b⟩ | b) <;>
            simp only [sumAssoc_apply_inl_inl, sumAssoc_apply_inl_inr, sumAssoc_apply_inr,
              Sum.lex_inl_inl, Sum.lex_inr_inr, Sum.Lex.sep, Sum.lex_inr_inl]⟩⟩
  nsmul := nsmulRec


@[simp]
theorem card_add (o₁ o₂ : Ordinal) : card (o₁ + o₂) = card o₁ + card o₂ :=
  inductionOn o₁ fun _ __ => inductionOn o₂ fun _ _ _ => rfl


@[simp]
theorem type_sum_lex {α β : Type u} (r : α → α → Prop) (s : β → β → Prop) [IsWellOrder α r]
    [IsWellOrder β s] : type (Sum.Lex r s) = type r + type s :=
  rfl


@[simp]
theorem card_nat (n : ℕ) : card.{u} n = n := by
  /-
    n : Nat
    ⊢ Eq (↑n).card ↑n
  -/
  induction n <;> [simp; simp only [card_add, card_one, Nat.cast_succ, *]]
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem card_ofNat (n : ℕ) [n.AtLeastTwo] :
    card.{u} (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  card_nat n


instance instAddLeftMono : AddLeftMono Ordinal.{u} where
  elim c a b := by
    refine inductionOn₃ a b c fun α r _ β s _ γ t _ ⟨f⟩ ↦
      (RelEmbedding.ofMonotone (Sum.recOn · Sum.inl (Sum.inr ∘ f)) ?_).ordinal_type_le
    /-
      α✝ : Type u
      β✝ : Type v
      γ✝ : Type w
      r✝ : α✝ → α✝ → Prop
      s✝ : β✝ → β✝ → Prop
      t✝ : γ✝ → γ✝ → Prop
      c a b : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝³ : IsWellOrder α r
      β : Type u
      s : β → β → Prop
      x✝² : IsWellOrder β s
      γ : Type u
      t : γ → γ → Prop
      x✝¹ : IsWellOrder γ t
      x✝ : LE.le (Ordinal.type r) (Ordinal.type s)
      f : InitialSeg r s
      ⊢ ∀ (a b : Sum γ α), Sum.Lex t r a b → Sum.Lex t s ((fun x => Sum.recOn x Sum. …
    -/
    simp [f.map_rel_iff]
    /-
      🎉 no goals
    -/


instance instAddRightMono : AddRightMono Ordinal.{u} where
  elim c a b := by
    refine inductionOn₃ a b c fun α r _ β s _ γ t _  ⟨f⟩ ↦
      (RelEmbedding.ofMonotone (Sum.recOn · (Sum.inl ∘ f) Sum.inr) ?_).ordinal_type_le
    /-
      α✝ : Type u
      β✝ : Type v
      γ✝ : Type w
      r✝ : α✝ → α✝ → Prop
      s✝ : β✝ → β✝ → Prop
      t✝ : γ✝ → γ✝ → Prop
      c a b : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝³ : IsWellOrder α r
      β : Type u
      s : β → β → Prop
      x✝² : IsWellOrder β s
      γ : Type u
      t : γ → γ → Prop
      x✝¹ : IsWellOrder γ t
      x✝ : LE.le (Ordinal.type r) (Ordinal.type s)
      f : InitialSeg r s
      ⊢ ∀ (a b : Sum α γ), Sum.Lex r t a b → Sum.Lex s t ((fun x => Sum.recOn x (Fun …
    -/
    simp [f.map_rel_iff]
    /-
      🎉 no goals
    -/


theorem le_add_right (a b : Ordinal) : a ≤ a + b := by
  /-
    a b : Ordinal.{u_1}
    ⊢ LE.le a (HAdd.hAdd a b)
  -/
  simpa only [add_zero] using add_le_add_left (Ordinal.zero_le b) a
  /-
    🎉 no goals
  -/


theorem le_add_left (a b : Ordinal) : a ≤ b + a := by
  /-
    a b : Ordinal.{u_1}
    ⊢ LE.le a (HAdd.hAdd b a)
  -/
  simpa only [zero_add] using add_le_add_right (Ordinal.zero_le b) a
  /-
    🎉 no goals
  -/


theorem max_zero_left : ∀ a : Ordinal, max 0 a = a :=
  max_bot_left


theorem max_zero_right : ∀ a : Ordinal, max a 0 = a :=
  max_bot_right


@[simp]
theorem max_eq_zero {a b : Ordinal} : max a b = 0 ↔ a = 0 ∧ b = 0 :=
  max_eq_bot


@[simp]
theorem sInf_empty : sInf (∅ : Set Ordinal) = 0 :=
  dif_neg Set.not_nonempty_empty


private theorem succ_le_iff' {a b : Ordinal} : a + 1 ≤ b ↔ a < b := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (LE.le (HAdd.hAdd a 1) b) (LT.lt a b)
  -/
  refine inductionOn₂ a b fun α r _ β s _ ↦ ⟨?_, ?_⟩ <;> rintro ⟨f⟩
    /-
      case refine_1.intro
      a b : Ordinal.{u_1}
      α : Type u_1
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type u_1
      s : β → β → Prop
      x✝ : IsWellOrder β s
      f : InitialSeg (Sum.Lex r EmptyRelation) s
      ⊢ LT.lt (Ordinal.type r) (Ordinal.type s)
    -/
  · refine ⟨((InitialSeg.leAdd _ _).trans f).toPrincipalSeg fun h ↦ ?_⟩
    /-
      case refine_1.intro
      a b : Ordinal.{u_1}
      α : Type u_1
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type u_1
      s : β → β → Prop
      x✝ : IsWellOrder β s
      f : InitialSeg (Sum.Lex r EmptyRelation) s
      h : Function.Surjective ⇑((InitialSeg.leAdd r EmptyRelation).trans f)
      ⊢ False
    -/
    simpa using h (f (Sum.inr PUnit.unit))
    /-
      🎉 no goals
    -/
    /-
      case refine_2.intro
      a b : Ordinal.{u_1}
      α : Type u_1
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type u_1
      s : β → β → Prop
      x✝ : IsWellOrder β s
      f : PrincipalSeg r s
      ⊢ LE.le (HAdd.hAdd (Ordinal.type r) 1) (Ordinal.type s)
    -/
  · apply (RelEmbedding.ofMonotone (Sum.recOn · f fun _ ↦ f.top) ?_).ordinal_type_le
    /-
      a b : Ordinal.{u_1}
      α : Type u_1
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type u_1
      s : β → β → Prop
      x✝ : IsWellOrder β s
      f : PrincipalSeg r s
      ⊢ ∀ (a b : Sum α PUnit.{u_1 + 1}), Sum.Lex r EmptyRelation a b → s ((fun x =>  …
    -/
    simpa [f.map_rel_iff] using f.lt_top
    /-
      🎉 no goals
    -/


instance : NoMaxOrder Ordinal :=
  ⟨fun _ => ⟨_, succ_le_iff'.1 le_rfl⟩⟩


instance : SuccOrder Ordinal.{u} :=
  SuccOrder.ofSuccLeIff (fun o => o + 1) succ_le_iff'


instance : SuccAddOrder Ordinal := ⟨fun _ => rfl⟩


@[simp]
theorem add_one_eq_succ (o : Ordinal) : o + 1 = succ o :=
  rfl


@[simp]
theorem succ_zero : succ (0 : Ordinal) = 1 :=
  zero_add 1

-- Porting note: Proof used to be rfl

@[simp]
                                                /-
                                                  ⊢ Eq (Order.succ 1) 2
                                                -/
theorem succ_one : succ (1 : Ordinal) = 2 := by congr; simp only [Nat.unaryCast, zero_add]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem add_succ (o₁ o₂ : Ordinal) : o₁ + succ o₂ = succ (o₁ + o₂) :=
  (add_assoc _ _ _).symm


@[deprecated Order.one_le_iff_pos (since := "2024-09-04")]
protected theorem one_le_iff_pos {o : Ordinal} : 1 ≤ o ↔ 0 < o :=
  Order.one_le_iff_pos


theorem one_le_iff_ne_zero {o : Ordinal} : 1 ≤ o ↔ o ≠ 0 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le 1 o) (Ne o 0)
  -/
  rw [Order.one_le_iff_pos, Ordinal.pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem succ_pos (o : Ordinal) : 0 < succ o :=
  bot_lt_succ o


theorem succ_ne_zero (o : Ordinal) : succ o ≠ 0 :=
  ne_of_gt <| succ_pos o


@[simp]
theorem lt_one_iff_zero {a : Ordinal} : a < 1 ↔ a = 0 := by
  /-
    a : Ordinal.{u_1}
    ⊢ Iff (LT.lt a 1) (Eq a 0)
  -/
  simpa using @lt_succ_bot_iff _ _ _ a _ _
  /-
    🎉 no goals
  -/


theorem le_one_iff {a : Ordinal} : a ≤ 1 ↔ a = 0 ∨ a = 1 := by
  /-
    a : Ordinal.{u_1}
    ⊢ Iff (LE.le a 1) (Or (Eq a 0) (Eq a 1))
  -/
  simpa using @le_succ_bot_iff _ _ _ a _
  /-
    🎉 no goals
  -/


@[simp]
theorem card_succ (o : Ordinal) : card (succ o) = card o + 1 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Order.succ o).card (HAdd.hAdd o.card 1)
  -/
  simp only [← add_one_eq_succ, card_add, card_one]
  /-
    🎉 no goals
  -/


theorem natCast_succ (n : ℕ) : ↑n.succ = succ (n : Ordinal) :=
  rfl


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_succ := natCast_succ


instance uniqueIioOne : Unique (Iio (1 : Ordinal)) where
                    /-
                      α : Type u
                      β : Type v
                      γ : Type w
                      r : α → α → Prop
                      s : β → β → Prop
                      t : γ → γ → Prop
                      ⊢ Membership.mem (Set.Iio 1) 0
                    -/
  default := ⟨0, by simp⟩
                    /-
                      🎉 no goals
                    -/
  uniq a := Subtype.ext <| lt_one_iff_zero.1 a.2


instance uniqueToTypeOne : Unique (toType 1) where
                                                 /-
                                                   α : Type u
                                                   β : Type v
                                                   γ : Type w
                                                   r : α → α → Prop
                                                   s : β → β → Prop
                                                   t : γ → γ → Prop
                                                   ⊢ LT.lt 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
                                                 -/
  default := enum (α := toType 1) (· < ·) ⟨0, by simp⟩
                                                 /-
                                                   🎉 no goals
                                                 -/
  uniq a := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      a : Ordinal.toType 1
      ⊢ Eq a Inhabited.default
    -/
    unfold default
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      a : Ordinal.toType 1
      ⊢ Eq a { default := (Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨0, ⋯⟩ }.1
    -/
    rw [← enum_typein (α := toType 1) (· < ·) a]
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      a : Ordinal.toType 1
      ⊢ Eq ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨(Ordinal.typein fun x1 x2 => LT …
    -/
    congr
    /-
      case h.e_6.h.e_val
      α : Type u
      β : Type v
      γ : Type w
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      a : Ordinal.toType 1
      ⊢ Eq ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) 0
    -/
    rw [← lt_one_iff_zero]
    /-
      case h.e_6.h.e_val
      α : Type u
      β : Type v
      γ : Type w
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      a : Ordinal.toType 1
      ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) 1
    -/
    apply typein_lt_self
    /-
      🎉 no goals
    -/


                                                               /-
                                                                 α : Type u
                                                                 β : Type v
                                                                 γ : Type w
                                                                 r : α → α → Prop
                                                                 s : β → β → Prop
                                                                 t : γ → γ → Prop
                                                                 x : Ordinal.toType 1
                                                                 ⊢ LT.lt 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
                                                               -/
theorem one_toType_eq (x : toType 1) : x = enum (· < ·) ⟨0, by simp⟩ :=
                                                               /-
                                                                 🎉 no goals
                                                               -/
  Unique.eq_default x


@[deprecated one_toType_eq (since := "2024-08-26")]
alias one_out_eq := one_toType_eq


@[simp]
theorem typein_one_toType (x : toType 1) : typein (α := toType 1) (· < ·) x = 0 := by
  /-
    x : Ordinal.toType 1
    ⊢ Eq ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding x) 0
  -/
  rw [one_toType_eq x, typein_enum]
  /-
    🎉 no goals
  -/


@[deprecated typein_one_toType (since := "2024-08-26")]
alias typein_one_out := typein_one_toType


theorem typein_le_typein' (o : Ordinal) {x y : o.toType} :
    typein (α := o.toType) (· < ·) x ≤ typein (α := o.toType) (· < ·) y ↔ x ≤ y := by
  /-
    o : Ordinal.{u_1}
    x y : o.toType
    ⊢ Iff (LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding x) ((Or …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem le_enum_succ {o : Ordinal} (a : (succ o).toType) :
    a ≤ enum (α := (succ o).toType) (· < ·) ⟨o, (type_toType _ ▸ lt_succ o)⟩ := by
  rw [← enum_typein (α := (succ o).toType) (· < ·) a, enum_le_enum', Subtype.mk_le_mk,
    ← lt_succ_iff]
  /-
    o : Ordinal.{u_1}
    a : (Order.succ o).toType
    ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (Order.su …
  -/
  apply typein_lt_self
  /-
    🎉 no goals
  -/


/-- `univ.{u v}` is the order type of the ordinals of `Type u` as a member
  of `Ordinal.{v}` (when `u < v`). It is an inaccessible cardinal. -/
@[pp_with_univ, nolint checkUnivs]
def univ : Ordinal.{max (u + 1) v} :=
  lift.{v, u + 1} (typeLT Ordinal)


theorem univ_id : univ.{u, u + 1} = typeLT Ordinal :=
  lift_id _


@[simp]
theorem lift_univ : lift.{w} univ.{u, v} = univ.{u, max v w} :=
  lift_lift _


theorem univ_umax : univ.{u, max (u + 1) v} = univ.{u, v} :=
  congr_fun lift_umax _


/-- Principal segment version of the lift operation on ordinals, embedding `Ordinal.{u}` in
`Ordinal.{v}` as a principal segment when `u < v`. -/
def liftPrincipalSeg : Ordinal.{u} <i Ordinal.{max (u + 1) v} :=
  ⟨↑liftInitialSeg.{max (u + 1) v, u}, univ.{u, v}, by
    /-
      α : Type u
      β : Type v
      γ : Type w
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      ⊢ ∀ (b : Ordinal.{max (u + 1) v}), Iff (Membership.mem (Set.range ⇑Ordinal.lif …
    -/
    refine fun b => inductionOn b ?_; intro β s _
    /-
      α : Type u
      β✝ : Type v
      γ : Type w
      r : α → α → Prop
      s✝ : β✝ → β✝ → Prop
      t : γ → γ → Prop
      b : Ordinal.{max (u + 1) v}
      β : Type (max (u + 1) v)
      s : β → β → Prop
      inst✝ : IsWellOrder β s
      ⊢ Iff (Membership.mem (Set.range ⇑Ordinal.liftInitialSeg.toRelEmbedding) (Ordi …
    -/
    rw [univ, ← lift_umax]; constructor <;> intro h
      /-
        case mp
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : Membership.mem (Set.range ⇑Ordinal.liftInitialSeg.toRelEmbedding) (Ordinal …
        ⊢ LT.lt (Ordinal.type s) (Ordinal.lift.{max (u + 1) v, u + 1} (Ordinal.type fu …
      -/
    · cases' h with a e
      /-
        case mp.intro
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        a : Ordinal.{u}
        e : Eq (Ordinal.liftInitialSeg.toRelEmbedding a) (Ordinal.type s)
        ⊢ LT.lt (Ordinal.type s) (Ordinal.lift.{max (u + 1) v, u + 1} (Ordinal.type fu …
      -/
      rw [← e]
      /-
        case mp.intro
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        a : Ordinal.{u}
        e : Eq (Ordinal.liftInitialSeg.toRelEmbedding a) (Ordinal.type s)
        ⊢ LT.lt (Ordinal.liftInitialSeg.toRelEmbedding a) (Ordinal.lift.{max (u + 1) v …
      -/
      refine inductionOn a ?_
      /-
        case mp.intro
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        a : Ordinal.{u}
        e : Eq (Ordinal.liftInitialSeg.toRelEmbedding a) (Ordinal.type s)
        ⊢ ∀ (α : Type u) (r : α → α → Prop) [inst : IsWellOrder α r], LT.lt (Ordinal.l …
      -/
      intro α r _
      /-
        case mp.intro
        α✝ : Type u
        β✝ : Type v
        γ : Type w
        r✝ : α✝ → α✝ → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝¹ : IsWellOrder β s
        a : Ordinal.{u}
        e : Eq (Ordinal.liftInitialSeg.toRelEmbedding a) (Ordinal.type s)
        α : Type u
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        ⊢ LT.lt (Ordinal.liftInitialSeg.toRelEmbedding (Ordinal.type r)) (Ordinal.lift …
      -/
      exact lift_type_lt.{u, u + 1, max (u + 1) v}.2 ⟨typein r⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.type s) (Ordinal.lift.{max (u + 1) v, u + 1} (Ordinal.type  …
        ⊢ Membership.mem (Set.range ⇑Ordinal.liftInitialSeg.toRelEmbedding) (Ordinal.t …
      -/
    · rw [← lift_id (type s)] at h ⊢
      /-
        case mpr
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
        ⊢ Membership.mem (Set.range ⇑Ordinal.liftInitialSeg.toRelEmbedding) (Ordinal.l …
      -/
      cases' lift_type_lt.{_,_,v}.1 h with f
      /-
        case mpr.intro
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
        f : PrincipalSeg s fun x1 x2 => LT.lt x1 x2
        ⊢ Membership.mem (Set.range ⇑Ordinal.liftInitialSeg.toRelEmbedding) (Ordinal.l …
      -/
      cases' f with f a hf
      /-
        case mpr.intro.mk
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
        f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
        a : Ordinal.{u}
        hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b a)
        ⊢ Membership.mem (Set.range ⇑Ordinal.liftInitialSeg.toRelEmbedding) (Ordinal.l …
      -/
      exists a
      /-
        case mpr.intro.mk
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
        f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
        a : Ordinal.{u}
        hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b a)
        ⊢ Eq (Ordinal.liftInitialSeg.toRelEmbedding a) (Ordinal.lift.{max (u + 1) v, m …
      -/
      revert hf
      -- Porting note: apply inductionOn does not work, refine does
      /-
        case mpr.intro.mk
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
        f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
        a : Ordinal.{u}
        ⊢ (∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b a)) → E …
      -/
      refine inductionOn a ?_
      /-
        case mpr.intro.mk
        α : Type u
        β✝ : Type v
        γ : Type w
        r : α → α → Prop
        s✝ : β✝ → β✝ → Prop
        t : γ → γ → Prop
        b : Ordinal.{max (u + 1) v}
        β : Type (max (u + 1) v)
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
        f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
        a : Ordinal.{u}
        ⊢ ∀ (α : Type u) (r : α → α → Prop) [inst : IsWellOrder α r], (∀ (b : Ordinal. …
      -/
      intro α r _ hf
      refine lift_type_eq.{u, max (u + 1) v, max (u + 1) v}.2
        ⟨(RelIso.ofSurjective (RelEmbedding.ofMonotone ?_ ?_) ?_).symm⟩
        /-
          case mpr.intro.mk.refine_1
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          ⊢ β → α
        -/
      · exact fun b => enum r ⟨f b, (hf _).1 ⟨_, rfl⟩⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.mk.refine_2
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          ⊢ ∀ (a b : β), s a b → r ((Ordinal.enum r) ⟨f a, ⋯⟩) ((Ordinal.enum r) ⟨f b, ⋯⟩)
        -/
      · refine fun a b h => (typein_lt_typein r).1 ?_
        /-
          case mpr.intro.mk.refine_2
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b✝ : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h✝ : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ord …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a✝ : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          a b : β
          h : s a b
          ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding ((Ordinal.enum r) ⟨f a, ⋯⟩)) ((Ordi …
        -/
        rw [typein_enum, typein_enum]
        /-
          case mpr.intro.mk.refine_2
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b✝ : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h✝ : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ord …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a✝ : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          a b : β
          h : s a b
          ⊢ LT.lt (f a) (f b)
        -/
        exact f.map_rel_iff.2 h
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.mk.refine_3
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          ⊢ Function.Surjective ⇑(RelEmbedding.ofMonotone (fun b => (Ordinal.enum r) ⟨f  …
        -/
      · intro a'
        /-
          case mpr.intro.mk.refine_3
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          a' : α
          ⊢ Exists fun a => Eq ((RelEmbedding.ofMonotone (fun b => (Ordinal.enum r) ⟨f b …
        -/
        cases' (hf _).2 (typein_lt_type _ a') with b e
        /-
          case mpr.intro.mk.refine_3.intro
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b✝ : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          a' : α
          b : β
          e : Eq (f b) ((Ordinal.typein r).toRelEmbedding a')
          ⊢ Exists fun a => Eq ((RelEmbedding.ofMonotone (fun b => (Ordinal.enum r) ⟨f b …
        -/
        exists b
        /-
          case mpr.intro.mk.refine_3.intro
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b✝ : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          a' : α
          b : β
          e : Eq (f b) ((Ordinal.typein r).toRelEmbedding a')
          ⊢ Eq ((RelEmbedding.ofMonotone (fun b => (Ordinal.enum r) ⟨f b, ⋯⟩) ⋯) b) a'
        -/
        simp only [RelEmbedding.ofMonotone_coe]
        /-
          case mpr.intro.mk.refine_3.intro
          α✝ : Type u
          β✝ : Type v
          γ : Type w
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t : γ → γ → Prop
          b✝ : Ordinal.{max (u + 1) v}
          β : Type (max (u + 1) v)
          s : β → β → Prop
          inst✝¹ : IsWellOrder β s
          h : LT.lt (Ordinal.lift.{max (u + 1) v, max (u + 1) v} (Ordinal.type s)) (Ordi …
          f : RelEmbedding s fun x1 x2 => LT.lt x1 x2
          a : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          inst✝ : IsWellOrder α r
          hf : ∀ (b : Ordinal.{u}), Iff (Membership.mem (Set.range ⇑f) b) (LT.lt b (Ordi …
          a' : α
          b : β
          e : Eq (f b) ((Ordinal.typein r).toRelEmbedding a')
          ⊢ Eq ((Ordinal.enum r) ⟨f b, ⋯⟩) a'
        -/
        simp [e]⟩
        /-
          🎉 no goals
        -/


@[deprecated liftPrincipalSeg (since := "2024-09-21")]
alias lift.principalSeg := liftPrincipalSeg


@[simp]
theorem liftPrincipalSeg_coe :
    (liftPrincipalSeg.{u, v} : Ordinal → Ordinal) = lift.{max (u + 1) v} :=
  rfl


set_option linter.deprecated false in
@[deprecated liftPrincipalSeg_coe (since := "2024-09-21")]
theorem lift.principalSeg_coe :
    (lift.principalSeg.{u, v} : Ordinal → Ordinal) = lift.{max (u + 1) v} :=
  rfl


@[simp]
theorem liftPrincipalSeg_top : (liftPrincipalSeg.{u, v}).top = univ.{u, v} :=
  rfl


set_option linter.deprecated false in
@[deprecated liftPrincipalSeg_top (since := "2024-09-21")]
theorem lift.principalSeg_top : (lift.principalSeg.{u, v}).top = univ.{u, v} :=
  rfl


theorem liftPrincipalSeg_top' : liftPrincipalSeg.{u, u + 1}.top = typeLT Ordinal := by
  /-
    ⊢ Eq Ordinal.liftPrincipalSeg.top (Ordinal.type fun x1 x2 => LT.lt x1 x2)
  -/
  simp only [liftPrincipalSeg_top, univ_id]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated liftPrincipalSeg_top (since := "2024-09-21")]
theorem lift.principalSeg_top' : lift.principalSeg.{u, u + 1}.top = typeLT Ordinal := by
  /-
    ⊢ Eq Ordinal.lift.principalSeg.top (Ordinal.type fun x1 x2 => LT.lt x1 x2)
  -/
  simp only [lift.principalSeg_top, univ_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_toType (o : Ordinal) : #o.toType = o.card :=
                                         /-
                                           o : Ordinal.{u_1}
                                           ⊢ Eq (Ordinal.type ?m.147221).card o.card
                                         -/
  (Ordinal.card_type _).symm.trans <| by rw [Ordinal.type_toType]
                                         /-
                                           🎉 no goals
                                         -/


@[deprecated mk_toType (since := "2024-08-26")]
alias mk_ordinal_out := mk_toType


/-- The ordinal corresponding to a cardinal `c` is the least ordinal
  whose cardinal is `c`. For the order-embedding version, see `ord.order_embedding`. -/
def ord (c : Cardinal) : Ordinal :=
  let F := fun α : Type u => ⨅ r : { r // IsWellOrder α r }, @type α r.1 r.2
  Quot.liftOn c F
    (by
      suffices ∀ {α β}, α ≈ β → F α ≤ F β from
        fun α β h => (this h).antisymm (this (Setoid.symm h))
      /-
        α : Type u
        β : Type v
        γ : Type w
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        c : Cardinal.{u}
        F : Type u → Ordinal.{u} := fun α => iInf fun r => Ordinal.type ↑r
        ⊢ ∀ {α β : Type u}, HasEquiv.Equiv α β → LE.le (F α) (F β)
      -/
      rintro α β ⟨f⟩
      /-
        case intro
        α✝ : Type u
        β✝ : Type v
        γ : Type w
        r : α✝ → α✝ → Prop
        s : β✝ → β✝ → Prop
        t : γ → γ → Prop
        c : Cardinal.{u}
        F : Type u → Ordinal.{u} := fun α => iInf fun r => Ordinal.type ↑r
        α β : Type u
        f : Equiv α β
        ⊢ LE.le (F α) (F β)
      -/
      refine le_ciInf_iff'.2 fun i => ?_
      /-
        case intro
        α✝ : Type u
        β✝ : Type v
        γ : Type w
        r : α✝ → α✝ → Prop
        s : β✝ → β✝ → Prop
        t : γ → γ → Prop
        c : Cardinal.{u}
        F : Type u → Ordinal.{u} := fun α => iInf fun r => Ordinal.type ↑r
        α β : Type u
        f : Equiv α β
        i : Subtype fun r => IsWellOrder β r
        ⊢ LE.le (F α) (Ordinal.type ↑i)
      -/
      haveI := @RelEmbedding.isWellOrder _ _ (f ⁻¹'o i.1) _ (↑(RelIso.preimage f i.1)) i.2
      exact
        (ciInf_le' _
              (Subtype.mk (f ⁻¹'o i.val)
                (@RelEmbedding.isWellOrder _ _ _ _ (↑(RelIso.preimage f i.1)) i.2))).trans_eq
          (Quot.sound ⟨RelIso.preimage f i.1⟩))


theorem ord_eq_Inf (α : Type u) : ord #α = ⨅ r : { r // IsWellOrder α r }, @type α r.1 r.2 :=
  rfl


theorem ord_eq (α) : ∃ (r : α → α → Prop) (wo : IsWellOrder α r), ord #α = @type α r wo :=
  let ⟨r, wo⟩ := ciInf_mem fun r : { r // IsWellOrder α r } => @type α r.1 r.2
  ⟨r.1, r.2, wo.symm⟩


theorem ord_le_type (r : α → α → Prop) [h : IsWellOrder α r] : ord #α ≤ type r :=
  ciInf_le' _ (Subtype.mk r h)


theorem ord_le {c o} : ord c ≤ o ↔ c ≤ o.card :=
  inductionOn c fun α =>
    Ordinal.inductionOn o fun β s _ => by
      /-
        c : Cardinal.{u_1}
        o : Ordinal.{u_1}
        α β : Type u_1
        s : β → β → Prop
        x✝ : IsWellOrder β s
        ⊢ Iff (LE.le (Cardinal.mk α).ord (Ordinal.type s)) (LE.le (Cardinal.mk α) (Ord …
      -/
      let ⟨r, _, e⟩ := ord_eq α
      /-
        c : Cardinal.{u_1}
        o : Ordinal.{u_1}
        α β : Type u_1
        s : β → β → Prop
        x✝ : IsWellOrder β s
        r : α → α → Prop
        w✝ : IsWellOrder α r
        e : Eq (Cardinal.mk α).ord (Ordinal.type r)
        ⊢ Iff (LE.le (Cardinal.mk α).ord (Ordinal.type s)) (LE.le (Cardinal.mk α) (Ord …
      -/
      simp only [card_type]; constructor <;> intro h
        /-
          case mp
          c : Cardinal.{u_1}
          o : Ordinal.{u_1}
          α β : Type u_1
          s : β → β → Prop
          x✝ : IsWellOrder β s
          r : α → α → Prop
          w✝ : IsWellOrder α r
          e : Eq (Cardinal.mk α).ord (Ordinal.type r)
          h : LE.le (Cardinal.mk α).ord (Ordinal.type s)
          ⊢ LE.le (Cardinal.mk α) (Cardinal.mk β)
        -/
      · rw [e] at h
        exact
          let ⟨f⟩ := h
          ⟨f.toEmbedding⟩
        /-
          case mpr
          c : Cardinal.{u_1}
          o : Ordinal.{u_1}
          α β : Type u_1
          s : β → β → Prop
          x✝ : IsWellOrder β s
          r : α → α → Prop
          w✝ : IsWellOrder α r
          e : Eq (Cardinal.mk α).ord (Ordinal.type r)
          h : LE.le (Cardinal.mk α) (Cardinal.mk β)
          ⊢ LE.le (Cardinal.mk α).ord (Ordinal.type s)
        -/
      · cases' h with f
        /-
          case mpr.intro
          c : Cardinal.{u_1}
          o : Ordinal.{u_1}
          α β : Type u_1
          s : β → β → Prop
          x✝ : IsWellOrder β s
          r : α → α → Prop
          w✝ : IsWellOrder α r
          e : Eq (Cardinal.mk α).ord (Ordinal.type r)
          f : Function.Embedding α β
          ⊢ LE.le (Cardinal.mk α).ord (Ordinal.type s)
        -/
        have g := RelEmbedding.preimage f s
        /-
          case mpr.intro
          c : Cardinal.{u_1}
          o : Ordinal.{u_1}
          α β : Type u_1
          s : β → β → Prop
          x✝ : IsWellOrder β s
          r : α → α → Prop
          w✝ : IsWellOrder α r
          e : Eq (Cardinal.mk α).ord (Ordinal.type r)
          f : Function.Embedding α β
          g : RelEmbedding (Order.Preimage (⇑f) s) s
          ⊢ LE.le (Cardinal.mk α).ord (Ordinal.type s)
        -/
        haveI := RelEmbedding.isWellOrder g
        /-
          case mpr.intro
          c : Cardinal.{u_1}
          o : Ordinal.{u_1}
          α β : Type u_1
          s : β → β → Prop
          x✝ : IsWellOrder β s
          r : α → α → Prop
          w✝ : IsWellOrder α r
          e : Eq (Cardinal.mk α).ord (Ordinal.type r)
          f : Function.Embedding α β
          g : RelEmbedding (Order.Preimage (⇑f) s) s
          this : IsWellOrder α (Order.Preimage (⇑f) s)
          ⊢ LE.le (Cardinal.mk α).ord (Ordinal.type s)
        -/
        exact le_trans (ord_le_type _) g.ordinal_type_le
        /-
          🎉 no goals
        -/


theorem gc_ord_card : GaloisConnection ord card := fun _ _ => ord_le


theorem lt_ord {c o} : o < ord c ↔ o.card < c :=
  gc_ord_card.lt_iff_lt


@[simp]
theorem card_ord (c) : (ord c).card = c :=
  c.inductionOn fun α ↦ let ⟨r, _, e⟩ := ord_eq α; e ▸ card_type r


theorem card_surjective : Function.Surjective card :=
  fun c ↦ ⟨_, card_ord c⟩


/-- Galois coinsertion between `Cardinal.ord` and `Ordinal.card`. -/
def gciOrdCard : GaloisCoinsertion ord card :=
  gc_ord_card.toGaloisCoinsertion fun c => c.card_ord.le


theorem ord_card_le (o : Ordinal) : o.card.ord ≤ o :=
  gc_ord_card.l_u_le _


theorem lt_ord_succ_card (o : Ordinal) : o < (succ o.card).ord :=
  lt_ord.2 <| lt_succ _


theorem card_le_iff {o : Ordinal} {c : Cardinal} : o.card ≤ c ↔ o < (succ c).ord := by
  /-
    o : Ordinal.{u_1}
    c : Cardinal.{u_1}
    ⊢ Iff (LE.le o.card c) (LT.lt o (Order.succ c).ord)
  -/
  rw [lt_ord, lt_succ_iff]
  /-
    🎉 no goals
  -/


/--
A variation on `Cardinal.lt_ord` using `≤`: If `o` is no greater than the
initial ordinal of cardinality `c`, then its cardinal is no greater than `c`.

The converse, however, is false (for instance, `o = ω+1` and `c = ℵ₀`).
-/
lemma card_le_of_le_ord {o : Ordinal} {c : Cardinal} (ho : o ≤ c.ord) :
    o.card ≤ c := by
  /-
    o : Ordinal.{u_1}
    c : Cardinal.{u_1}
    ho : LE.le o c.ord
    ⊢ LE.le o.card c
  -/
  rw [← card_ord c]; exact Ordinal.card_le_card ho
                     /-
                       🎉 no goals
                     -/


@[mono]
theorem ord_strictMono : StrictMono ord :=
  gciOrdCard.strictMono_l


@[mono]
theorem ord_mono : Monotone ord :=
  gc_ord_card.monotone_l


@[simp]
theorem ord_le_ord {c₁ c₂} : ord c₁ ≤ ord c₂ ↔ c₁ ≤ c₂ :=
  gciOrdCard.l_le_l_iff


@[simp]
theorem ord_lt_ord {c₁ c₂} : ord c₁ < ord c₂ ↔ c₁ < c₂ :=
  ord_strictMono.lt_iff_lt


@[simp]
theorem ord_zero : ord 0 = 0 :=
  gc_ord_card.l_bot


@[simp]
theorem ord_nat (n : ℕ) : ord n = n :=
  (ord_le.2 (card_nat n).ge).antisymm
    (by
      /-
        n : Nat
        ⊢ LE.le (↑n) (↑n).ord
      -/
      induction' n with n IH
        /-
          case zero
          ⊢ LE.le (↑0) (↑0).ord
        -/
      · apply Ordinal.zero_le
        /-
          🎉 no goals
        -/
        /-
          case succ
          n : Nat
          IH : LE.le (↑n) (↑n).ord
          ⊢ LE.le (↑(HAdd.hAdd n 1)) (↑(HAdd.hAdd n 1)).ord
        -/
      · exact succ_le_of_lt (IH.trans_lt <| ord_lt_ord.2 <| Nat.cast_lt.2 (Nat.lt_succ_self n)))
        /-
          🎉 no goals
        -/


@[simp]
                                  /-
                                    ⊢ Eq (Cardinal.ord 1) 1
                                  -/
theorem ord_one : ord 1 = 1 := by simpa using ord_nat 1
                                  /-
                                    🎉 no goals
                                  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ord_ofNat (n : ℕ) [n.AtLeastTwo] : ord (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  ord_nat n


@[simp]
theorem ord_aleph0 : ord.{u} ℵ₀ = ω :=
  le_antisymm (ord_le.2 le_rfl) <|
    le_of_forall_lt fun o h => by
      /-
        o : Ordinal.{u}
        h : LT.lt o Ordinal.omega0
        ⊢ LT.lt o Cardinal.aleph0.ord
      -/
      rcases Ordinal.lt_lift_iff.1 h with ⟨o, h', rfl⟩
      /-
        case intro.intro
        o : Ordinal.{0}
        h' : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        h : LT.lt (Ordinal.lift.{u, 0} o) Ordinal.omega0
        ⊢ LT.lt (Ordinal.lift.{u, 0} o) Cardinal.aleph0.ord
      -/
      rw [lt_ord, ← lift_card, lift_lt_aleph0, ← typein_enum (· < ·) h']
      /-
        case intro.intro
        o : Ordinal.{0}
        h' : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        h : LT.lt (Ordinal.lift.{u, 0} o) Ordinal.omega0
        ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding ((Ordinal.en …
      -/
      exact lt_aleph0_iff_fintype.2 ⟨Set.fintypeLTNat _⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem lift_ord (c) : Ordinal.lift.{u,v} (ord c) = ord (lift.{u,v} c) := by
  /-
    c : Cardinal.{v}
    ⊢ Eq (Ordinal.lift.{u, v} c.ord) (Cardinal.lift.{u, v} c).ord
  -/
  refine le_antisymm (le_of_forall_lt fun a ha => ?_) ?_
    /-
      case refine_1
      c : Cardinal.{v}
      a : Ordinal.{max v u}
      ha : LT.lt a (Ordinal.lift.{u, v} c.ord)
      ⊢ LT.lt a (Cardinal.lift.{u, v} c).ord
    -/
  · rcases Ordinal.lt_lift_iff.1 ha with ⟨a, _, rfl⟩
    /-
      case refine_1.intro.intro
      c : Cardinal.{v}
      a : Ordinal.{v}
      left✝ : LT.lt a c.ord
      ha : LT.lt (Ordinal.lift.{u, v} a) (Ordinal.lift.{u, v} c.ord)
      ⊢ LT.lt (Ordinal.lift.{u, v} a) (Cardinal.lift.{u, v} c).ord
    -/
    rwa [lt_ord, ← lift_card, lift_lt, ← lt_ord, ← Ordinal.lift_lt]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      c : Cardinal.{v}
      ⊢ LE.le (Cardinal.lift.{u, v} c).ord (Ordinal.lift.{u, v} c.ord)
    -/
  · rw [ord_le, ← lift_card, card_ord]
    /-
      🎉 no goals
    -/


                                                               /-
                                                                 c : Cardinal.{u_1}
                                                                 ⊢ Eq (Cardinal.mk c.ord.toType) c
                                                               -/
theorem mk_ord_toType (c : Cardinal) : #c.ord.toType = c := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[deprecated mk_ord_toType (since := "2024-08-26")]
alias mk_ord_out := mk_ord_toType


theorem card_typein_lt (r : α → α → Prop) [IsWellOrder α r] (x : α) (h : ord #α = type r) :
    card (typein r x) < #α := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    x : α
    h : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding x).card (Cardinal.mk α)
  -/
  rw [← lt_ord, h]
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    x : α
    h : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding x) (Ordinal.type r)
  -/
  apply typein_lt_type
  /-
    🎉 no goals
  -/


theorem card_typein_toType_lt (c : Cardinal) (x : c.ord.toType) :
    card (typein (α := c.ord.toType) (· < ·) x) < c := by
  /-
    c : Cardinal.{u_1}
    x : c.ord.toType
    ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding x).card c
  -/
  rw [← lt_ord]
  /-
    c : Cardinal.{u_1}
    x : c.ord.toType
    ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding x) c.ord
  -/
  apply typein_lt_self
  /-
    🎉 no goals
  -/


@[deprecated card_typein_toType_lt (since := "2024-08-26")]
alias card_typein_out_lt := card_typein_toType_lt


theorem mk_Iio_ord_toType {c : Cardinal} (i : c.ord.toType) : #(Iio i) < c :=
  card_typein_toType_lt c i


@[deprecated "No deprecation message was provided."  (since := "2024-08-26")]
alias mk_Iio_ord_out_α := mk_Iio_ord_toType


theorem ord_injective : Injective ord := by
  /-
    ⊢ Function.Injective Cardinal.ord
  -/
  intro c c' h
  /-
    c c' : Cardinal.{u_1}
    h : Eq c.ord c'.ord
    ⊢ Eq c c'
  -/
  rw [← card_ord c, ← card_ord c', h]
  /-
    🎉 no goals
  -/


@[simp]
theorem ord_inj {a b : Cardinal} : a.ord = b.ord ↔ a = b :=
  ord_injective.eq_iff


@[simp]
theorem ord_eq_zero {a : Cardinal} : a.ord = 0 ↔ a = 0 :=
  ord_injective.eq_iff' ord_zero


@[simp]
theorem ord_eq_one {a : Cardinal} : a.ord = 1 ↔ a = 1 :=
  ord_injective.eq_iff' ord_one


@[simp]
theorem omega0_le_ord {a : Cardinal} : ω ≤ a.ord ↔ ℵ₀ ≤ a := by
  /-
    a : Cardinal.{u_1}
    ⊢ Iff (LE.le Ordinal.omega0 a.ord) (LE.le Cardinal.aleph0 a)
  -/
  rw [← ord_aleph0, ord_le_ord]
  /-
    🎉 no goals
  -/


@[simp]
theorem ord_le_omega0 {a : Cardinal} : a.ord ≤ ω ↔ a ≤ ℵ₀ := by
  /-
    a : Cardinal.{u_1}
    ⊢ Iff (LE.le a.ord Ordinal.omega0) (LE.le a Cardinal.aleph0)
  -/
  rw [← ord_aleph0, ord_le_ord]
  /-
    🎉 no goals
  -/


@[simp]
theorem ord_lt_omega0 {a : Cardinal} : a.ord < ω ↔ a < ℵ₀ :=
  le_iff_le_iff_lt_iff_lt.1 omega0_le_ord


@[simp]
theorem omega0_lt_ord {a : Cardinal} : ω < a.ord ↔ ℵ₀ < a :=
  le_iff_le_iff_lt_iff_lt.1 ord_le_omega0


@[simp]
theorem ord_eq_omega0 {a : Cardinal} : a.ord = ω ↔ a = ℵ₀ :=
  ord_injective.eq_iff' ord_aleph0


/-- The ordinal corresponding to a cardinal `c` is the least ordinal
  whose cardinal is `c`. This is the order-embedding version. For the regular function, see `ord`.
-/
def ord.orderEmbedding : Cardinal ↪o Ordinal :=
  RelEmbedding.orderEmbeddingOfLTEmbedding
    (RelEmbedding.ofMonotone Cardinal.ord fun _ _ => Cardinal.ord_lt_ord.2)


@[simp]
theorem ord.orderEmbedding_coe : (ord.orderEmbedding : Cardinal → Ordinal) = ord :=
  rfl

-- intended to be used with explicit universe parameters

/-- The cardinal `univ` is the cardinality of ordinal `univ`, or
  equivalently the cardinal of `Ordinal.{u}`, or `Cardinal.{u}`,
  as an element of `Cardinal.{v}` (when `u < v`). -/
@[pp_with_univ, nolint checkUnivs]
def univ :=
  lift.{v, u + 1} #Ordinal


theorem univ_id : univ.{u, u + 1} = #Ordinal :=
  lift_id _


theorem lift_lt_univ (c : Cardinal) : lift.{u + 1, u} c < univ.{u, u + 1} := by
  simpa only [liftPrincipalSeg_coe, lift_ord, lift_succ, ord_le, succ_le_iff] using
    le_of_lt (liftPrincipalSeg.{u, u + 1}.lt_top (succ c).ord)


theorem lift_lt_univ' (c : Cardinal) : lift.{max (u + 1) v, u} c < univ.{u, v} := by
  /-
    c : Cardinal.{u}
    ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
  -/
  have := lift_lt.{_, max (u+1) v}.2 (lift_lt_univ c)
  /-
    c : Cardinal.{u}
    this : LT.lt (Cardinal.lift.{max (u + 1) v, u + 1} (Cardinal.lift.{u + 1, u} c …
    ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
  -/
  rw [lift_lift, lift_univ, univ_umax.{u,v}] at this
  /-
    c : Cardinal.{u}
    this : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
    ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
  -/
  exact this
  /-
    🎉 no goals
  -/


@[simp]
theorem ord_univ : ord univ.{u, v} = Ordinal.univ.{u, v} := by
  /-
    ⊢ Eq Cardinal.univ.{u, v}.ord Ordinal.univ.{u, v}
  -/
  refine le_antisymm (ord_card_le _) <| le_of_forall_lt fun o h => lt_ord.2 ?_
  /-
    o : Ordinal.{max (u + 1) v}
    h : LT.lt o Ordinal.univ.{u, v}
    ⊢ LT.lt o.card Cardinal.univ.{u, v}
  -/
  have := liftPrincipalSeg.mem_range_of_rel_top (by simpa only [liftPrincipalSeg_coe] using h)
  /-
    o : Ordinal.{max (u + 1) v}
    h : LT.lt o Ordinal.univ.{u, v}
    this : Membership.mem (Set.range ⇑Ordinal.liftPrincipalSeg.toRelEmbedding) o
    ⊢ LT.lt o.card Cardinal.univ.{u, v}
  -/
  rcases this with ⟨o, h'⟩
  /-
    case intro
    o✝ : Ordinal.{max (u + 1) v}
    h : LT.lt o✝ Ordinal.univ.{u, v}
    o : Ordinal.{u}
    h' : Eq (Ordinal.liftPrincipalSeg.toRelEmbedding o) o✝
    ⊢ LT.lt o✝.card Cardinal.univ.{u, v}
  -/
  rw [← h', liftPrincipalSeg_coe, ← lift_card]
  /-
    case intro
    o✝ : Ordinal.{max (u + 1) v}
    h : LT.lt o✝ Ordinal.univ.{u, v}
    o : Ordinal.{u}
    h' : Eq (Ordinal.liftPrincipalSeg.toRelEmbedding o) o✝
    ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} o.card) Cardinal.univ.{u, v}
  -/
  apply lift_lt_univ'
  /-
    🎉 no goals
  -/


theorem lt_univ {c} : c < univ.{u, u + 1} ↔ ∃ c', c = lift.{u + 1, u} c' :=
  ⟨fun h => by
    /-
      c : Cardinal.{u + 1}
      h : LT.lt c Cardinal.univ.{u, u + 1}
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{u + 1, u} c')
    -/
    have := ord_lt_ord.2 h
    /-
      c : Cardinal.{u + 1}
      h : LT.lt c Cardinal.univ.{u, u + 1}
      this : LT.lt c.ord Cardinal.univ.{u, u + 1}.ord
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{u + 1, u} c')
    -/
    rw [ord_univ] at this
    /-
      c : Cardinal.{u + 1}
      h : LT.lt c Cardinal.univ.{u, u + 1}
      this : LT.lt c.ord Ordinal.univ.{u, u + 1}
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{u + 1, u} c')
    -/
    cases' liftPrincipalSeg.mem_range_of_rel_top (by simpa only [liftPrincipalSeg_top]) with o e
    /-
      case intro
      c : Cardinal.{u + 1}
      h : LT.lt c Cardinal.univ.{u, u + 1}
      this : LT.lt c.ord Ordinal.univ.{u, u + 1}
      o : Ordinal.{u}
      e : Eq (Ordinal.liftPrincipalSeg.toRelEmbedding o) c.ord
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{u + 1, u} c')
    -/
    have := card_ord c
    /-
      case intro
      c : Cardinal.{u + 1}
      h : LT.lt c Cardinal.univ.{u, u + 1}
      this✝ : LT.lt c.ord Ordinal.univ.{u, u + 1}
      o : Ordinal.{u}
      e : Eq (Ordinal.liftPrincipalSeg.toRelEmbedding o) c.ord
      this : Eq c.ord.card c
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{u + 1, u} c')
    -/
    rw [← e, liftPrincipalSeg_coe, ← lift_card] at this
    /-
      case intro
      c : Cardinal.{u + 1}
      h : LT.lt c Cardinal.univ.{u, u + 1}
      this✝ : LT.lt c.ord Ordinal.univ.{u, u + 1}
      o : Ordinal.{u}
      e : Eq (Ordinal.liftPrincipalSeg.toRelEmbedding o) c.ord
      this : Eq (Cardinal.lift.{u + 1, u} o.card) c
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{u + 1, u} c')
    -/
    exact ⟨_, this.symm⟩, fun ⟨_, e⟩ => e.symm ▸ lift_lt_univ _⟩
    /-
      🎉 no goals
    -/


theorem lt_univ' {c} : c < univ.{u, v} ↔ ∃ c', c = lift.{max (u + 1) v, u} c' :=
  ⟨fun h => by
    /-
      c : Cardinal.{max (u + 1) v}
      h : LT.lt c Cardinal.univ.{u, v}
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{max (u + 1) v, u} c')
    -/
    let ⟨a, h', e⟩ := lt_lift_iff.1 h
    /-
      c : Cardinal.{max (u + 1) v}
      h : LT.lt c Cardinal.univ.{u, v}
      a : Cardinal.{u + 1}
      h' : LT.lt a (Cardinal.mk Ordinal.{u})
      e : Eq (Cardinal.lift.{v, u + 1} a) c
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{max (u + 1) v, u} c')
    -/
    rw [← univ_id] at h'
    /-
      c : Cardinal.{max (u + 1) v}
      h : LT.lt c Cardinal.univ.{u, v}
      a : Cardinal.{u + 1}
      h' : LT.lt a Cardinal.univ.{u, u + 1}
      e : Eq (Cardinal.lift.{v, u + 1} a) c
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{max (u + 1) v, u} c')
    -/
    rcases lt_univ.{u}.1 h' with ⟨c', rfl⟩
    /-
      case intro
      c : Cardinal.{max (u + 1) v}
      h : LT.lt c Cardinal.univ.{u, v}
      c' : Cardinal.{u}
      h' : LT.lt (Cardinal.lift.{u + 1, u} c') Cardinal.univ.{u, u + 1}
      e : Eq (Cardinal.lift.{v, u + 1} (Cardinal.lift.{u + 1, u} c')) c
      ⊢ Exists fun c' => Eq c (Cardinal.lift.{max (u + 1) v, u} c')
    -/
    exact ⟨c', by simp only [e.symm, lift_lift]⟩, fun ⟨_, e⟩ => e.symm ▸ lift_lt_univ' _⟩
    /-
      🎉 no goals
    -/


theorem small_iff_lift_mk_lt_univ {α : Type u} :
    Small.{v} α ↔ Cardinal.lift.{v+1,_} #α < univ.{v, max u (v + 1)} := by
  /-
    α : Type u
    ⊢ Iff (Small.{v, u} α) (LT.lt (Cardinal.lift.{v + 1, u} (Cardinal.mk α)) Cardi …
  -/
  rw [lt_univ']
  /-
    α : Type u
    ⊢ Iff (Small.{v, u} α) (Exists fun c' => Eq (Cardinal.lift.{v + 1, u} (Cardina …
  -/
  constructor
    /-
      case mp
      α : Type u
      ⊢ Small.{v, u} α → Exists fun c' => Eq (Cardinal.lift.{v + 1, u} (Cardinal.mk  …
    -/
  · rintro ⟨β, e⟩
    /-
      case mp.mk.intro
      α : Type u
      β : Type v
      e : Nonempty (Equiv α β)
      ⊢ Exists fun c' => Eq (Cardinal.lift.{v + 1, u} (Cardinal.mk α)) (Cardinal.lif …
    -/
    exact ⟨#β, lift_mk_eq.{u, _, v + 1}.2 e⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      ⊢ (Exists fun c' => Eq (Cardinal.lift.{v + 1, u} (Cardinal.mk α)) (Cardinal.li …
    -/
  · rintro ⟨c, hc⟩
    /-
      case mpr.intro
      α : Type u
      c : Cardinal.{v}
      hc : Eq (Cardinal.lift.{v + 1, u} (Cardinal.mk α)) (Cardinal.lift.{max u (v +  …
      ⊢ Small.{v, u} α
    -/
    exact ⟨⟨c.out, lift_mk_eq.{u, _, v + 1}.1 (hc.trans (congr rfl c.mk_out.symm))⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem card_univ : card univ.{u,v} = Cardinal.univ.{u,v} :=
  rfl


@[simp]
theorem nat_le_card {o} {n : ℕ} : (n : Cardinal) ≤ card o ↔ (n : Ordinal) ≤ o := by
  /-
    o : Ordinal.{u_1}
    n : Nat
    ⊢ Iff (LE.le (↑n) o.card) (LE.le (↑n) o)
  -/
  rw [← Cardinal.ord_le, Cardinal.ord_nat]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_le_card {o} : 1 ≤ card o ↔ 1 ≤ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le 1 o.card) (LE.le 1 o)
  -/
  simpa using nat_le_card (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_le_card {o} {n : ℕ} [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Cardinal)) ≤ card o ↔ (OfNat.ofNat n : Ordinal) ≤ o :=
  nat_le_card


@[simp]
theorem aleph0_le_card {o} : ℵ₀ ≤ card o ↔ ω ≤ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le Cardinal.aleph0 o.card) (LE.le Ordinal.omega0 o)
  -/
  rw [← ord_le, ord_aleph0]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_lt_aleph0 {o} : card o < ℵ₀ ↔ o < ω :=
  le_iff_le_iff_lt_iff_lt.1 aleph0_le_card


@[simp]
theorem nat_lt_card {o} {n : ℕ} : (n : Cardinal) < card o ↔ (n : Ordinal) < o := by
  /-
    o : Ordinal.{u_1}
    n : Nat
    ⊢ Iff (LT.lt (↑n) o.card) (LT.lt (↑n) o)
  -/
  rw [← succ_le_iff, ← succ_le_iff, ← nat_succ, nat_le_card]
  /-
    o : Ordinal.{u_1}
    n : Nat
    ⊢ Iff (LE.le (↑n.succ) o) (LE.le (Order.succ ↑n) o)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_lt_card {o} : 0 < card o ↔ 0 < o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LT.lt 0 o.card) (LT.lt 0 o)
  -/
  simpa using nat_lt_card (n := 0)
  /-
    🎉 no goals
  -/


@[simp]
theorem one_lt_card {o} : 1 < card o ↔ 1 < o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LT.lt 1 o.card) (LT.lt 1 o)
  -/
  simpa using nat_lt_card (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_lt_card {o} {n : ℕ} [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Cardinal)) < card o ↔ (OfNat.ofNat n : Ordinal) < o :=
  nat_lt_card


@[simp]
theorem card_lt_nat {o} {n : ℕ} : card o < n ↔ o < n :=
  lt_iff_lt_of_le_iff_le nat_le_card

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem card_lt_ofNat {o} {n : ℕ} [n.AtLeastTwo] :
    card o < (no_index (OfNat.ofNat n)) ↔ o < OfNat.ofNat n :=
  card_lt_nat


@[simp]
theorem card_le_nat {o} {n : ℕ} : card o ≤ n ↔ o ≤ n :=
  le_iff_le_iff_lt_iff_lt.2 nat_lt_card


@[simp]
theorem card_le_one {o} : card o ≤ 1 ↔ o ≤ 1 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le o.card 1) (LE.le o 1)
  -/
  simpa using card_le_nat (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem card_le_ofNat {o} {n : ℕ} [n.AtLeastTwo] :
    card o ≤ (no_index (OfNat.ofNat n)) ↔ o ≤ OfNat.ofNat n :=
  card_le_nat


@[simp]
theorem card_eq_nat {o} {n : ℕ} : card o = n ↔ o = n := by
  /-
    o : Ordinal.{u_1}
    n : Nat
    ⊢ Iff (Eq o.card ↑n) (Eq o ↑n)
  -/
  simp only [le_antisymm_iff, card_le_nat, nat_le_card]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_eq_zero {o} : card o = 0 ↔ o = 0 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (Eq o.card 0) (Eq o 0)
  -/
  simpa using card_eq_nat (n := 0)
  /-
    🎉 no goals
  -/


@[simp]
theorem card_eq_one {o} : card o = 1 ↔ o = 1 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (Eq o.card 1) (Eq o 1)
  -/
  simpa using card_eq_nat (n := 1)
  /-
    🎉 no goals
  -/


theorem mem_range_lift_of_card_le {a : Cardinal.{u}} {b : Ordinal.{max u v}}
    (h : card b ≤ Cardinal.lift.{v, u} a) : b ∈ Set.range lift.{v, u} := by
  /-
    a : Cardinal.{u}
    b : Ordinal.{max u v}
    h : LE.le b.card (Cardinal.lift.{v, u} a)
    ⊢ Membership.mem (Set.range Ordinal.lift.{v, u}) b
  -/
  rw [card_le_iff, ← lift_succ, ← lift_ord] at h
  /-
    a : Cardinal.{u}
    b : Ordinal.{max u v}
    h : LT.lt b (Ordinal.lift.{v, u} (Order.succ a).ord)
    ⊢ Membership.mem (Set.range Ordinal.lift.{v, u}) b
  -/
  exact mem_range_lift_of_le h.le
  /-
    🎉 no goals
  -/


@[deprecated mem_range_lift_of_card_le (since := "2024-10-07")]
theorem lift_down' {a : Cardinal.{u}} {b : Ordinal.{max u v}}
    (h : card.{max u v} b ≤ Cardinal.lift.{v, u} a) : ∃ a', lift.{v, u} a' = b :=
  mem_range_lift_of_card_le h

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem card_eq_ofNat {o} {n : ℕ} [n.AtLeastTwo] :
    card o = (no_index (OfNat.ofNat n)) ↔ o = OfNat.ofNat n :=
  card_eq_nat


@[simp]
theorem type_fintype (r : α → α → Prop) [IsWellOrder α r] [Fintype α] :
                                  /-
                                    α : Type u
                                    r : α → α → Prop
                                    inst✝¹ : IsWellOrder α r
                                    inst✝ : Fintype α
                                    ⊢ Eq (Ordinal.type r) ↑(Fintype.card α)
                                  -/
    type r = Fintype.card α := by rw [← card_eq_nat, card_type, mk_fintype]
                                  /-
                                    🎉 no goals
                                  -/


                                                    /-
                                                      n : Nat
                                                      ⊢ Eq (Ordinal.type fun x1 x2 => LT.lt x1 x2) ↑n
                                                    -/
theorem type_fin (n : ℕ) : typeLT (Fin n) = n := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem List.Sorted.lt_ord_of_lt [LinearOrder α] [WellFoundedLT α] {l m : List α}
    {o : Ordinal} (hl : l.Sorted (· > ·)) (hm : m.Sorted (· > ·)) (hmltl : m < l)
    (hlt : ∀ i ∈ l, Ordinal.typein (α := α) (· < ·) i < o) :
      ∀ i ∈ m, Ordinal.typein (α := α) (· < ·) i < o := by
  /-
    α : Type u
    inst✝¹ : LinearOrder α
    inst✝ : WellFoundedLT α
    l m : List α
    o : Ordinal.{u}
    hl : List.Sorted (fun x1 x2 => GT.gt x1 x2) l
    hm : List.Sorted (fun x1 x2 => GT.gt x1 x2) m
    hmltl : LT.lt m l
    hlt : ∀ (i : α), Membership.mem l i → LT.lt ((Ordinal.typein fun x1 x2 => LT.l …
    ⊢ ∀ (i : α), Membership.mem m i → LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 …
  -/
  replace hmltl : List.Lex (· < ·) m l := hmltl
  cases l with
  | nil => simp at hmltl
  | cons a as =>
    cases m with
    | nil => intro i hi; simp at hi
    | cons b bs =>
      intro i hi
      suffices h : i ≤ a by refine lt_of_le_of_lt ?_ (hlt a (mem_cons_self a as)); simpa
      cases hi with
      | head as => exact List.head_le_of_lt hmltl
      | tail b hi => exact le_of_lt (lt_of_lt_of_le (List.rel_of_sorted_cons hm _ hi)
          (List.head_le_of_lt hmltl))

