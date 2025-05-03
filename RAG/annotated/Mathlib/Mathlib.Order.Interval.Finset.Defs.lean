/-- This is a mixin class describing a locally finite order,
that is, is an order where bounded intervals are finite.
When you don't care too much about definitional equality, you can use `LocallyFiniteOrder.ofIcc` or
`LocallyFiniteOrder.ofFiniteIcc` to build a locally finite order from just `Finset.Icc`. -/
class LocallyFiniteOrder (α : Type*) [Preorder α] where
  /-- Left-closed right-closed interval -/
  finsetIcc : α → α → Finset α
  /-- Left-closed right-open interval -/
  finsetIco : α → α → Finset α
  /-- Left-open right-closed interval -/
  finsetIoc : α → α → Finset α
  /-- Left-open right-open interval -/
  finsetIoo : α → α → Finset α
  /-- `x ∈ finsetIcc a b ↔ a ≤ x ∧ x ≤ b` -/
  finset_mem_Icc : ∀ a b x : α, x ∈ finsetIcc a b ↔ a ≤ x ∧ x ≤ b
  /-- `x ∈ finsetIco a b ↔ a ≤ x ∧ x < b` -/
  finset_mem_Ico : ∀ a b x : α, x ∈ finsetIco a b ↔ a ≤ x ∧ x < b
  /-- `x ∈ finsetIoc a b ↔ a < x ∧ x ≤ b` -/
  finset_mem_Ioc : ∀ a b x : α, x ∈ finsetIoc a b ↔ a < x ∧ x ≤ b
  /-- `x ∈ finsetIoo a b ↔ a < x ∧ x < b` -/
  finset_mem_Ioo : ∀ a b x : α, x ∈ finsetIoo a b ↔ a < x ∧ x < b


/-- This mixin class describes an order where all intervals bounded below are finite. This is
slightly weaker than `LocallyFiniteOrder` + `OrderTop` as it allows empty types. -/
class LocallyFiniteOrderTop (α : Type*) [Preorder α] where
  /-- Left-open right-infinite interval -/
  finsetIoi : α → Finset α
  /-- Left-closed right-infinite interval -/
  finsetIci : α → Finset α
  /-- `x ∈ finsetIci a ↔ a ≤ x` -/
  finset_mem_Ici : ∀ a x : α, x ∈ finsetIci a ↔ a ≤ x
  /-- `x ∈ finsetIoi a ↔ a < x` -/
  finset_mem_Ioi : ∀ a x : α, x ∈ finsetIoi a ↔ a < x


/-- This mixin class describes an order where all intervals bounded above are finite. This is
slightly weaker than `LocallyFiniteOrder` + `OrderBot` as it allows empty types. -/
class LocallyFiniteOrderBot (α : Type*) [Preorder α] where
  /-- Left-infinite right-open interval -/
  finsetIio : α → Finset α
  /-- Left-infinite right-closed interval -/
  finsetIic : α → Finset α
  /-- `x ∈ finsetIic a ↔ x ≤ a` -/
  finset_mem_Iic : ∀ a x : α, x ∈ finsetIic a ↔ x ≤ a
  /-- `x ∈ finsetIio a ↔ x < a` -/
  finset_mem_Iio : ∀ a x : α, x ∈ finsetIio a ↔ x < a


/-- A constructor from a definition of `Finset.Icc` alone, the other ones being derived by removing
the ends. As opposed to `LocallyFiniteOrder.ofIcc`, this one requires `DecidableRel (· ≤ ·)` but
only `Preorder`. -/
def LocallyFiniteOrder.ofIcc' (α : Type*) [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (finsetIcc : α → α → Finset α) (mem_Icc : ∀ a b x, x ∈ finsetIcc a b ↔ a ≤ x ∧ x ≤ b) :
    LocallyFiniteOrder α where
  finsetIcc := finsetIcc
  finsetIco a b := {x ∈ finsetIcc a b | ¬b ≤ x}
  finsetIoc a b := {x ∈ finsetIcc a b | ¬x ≤ a}
  finsetIoo a b := {x ∈ finsetIcc a b | ¬x ≤ a ∧ ¬b ≤ x}
  finset_mem_Icc := mem_Icc
                             /-
                               α : Type u_1
                               inst✝¹ : Preorder α
                               inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                               finsetIcc : α → α → Finset α
                               mem_Icc : ∀ (a b x : α), Iff (Membership.mem (finsetIcc a b) x) (And (LE.le a  …
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => Finset.filter (fun x => Not (LE.le b x)) (f …
                             -/
  finset_mem_Ico a b x := by rw [Finset.mem_filter, mem_Icc, and_assoc, lt_iff_le_not_le]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               inst✝¹ : Preorder α
                               inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                               finsetIcc : α → α → Finset α
                               mem_Icc : ∀ (a b x : α), Iff (Membership.mem (finsetIcc a b) x) (And (LE.le a  …
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => Finset.filter (fun x => Not (LE.le x a)) (f …
                             -/
  finset_mem_Ioc a b x := by rw [Finset.mem_filter, mem_Icc, and_right_comm, lt_iff_le_not_le]
                             /-
                               🎉 no goals
                             -/
  finset_mem_Ioo a b x := by
    /-
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      finsetIcc : α → α → Finset α
      mem_Icc : ∀ (a b x : α), Iff (Membership.mem (finsetIcc a b) x) (And (LE.le a  …
      a b x : α
      ⊢ Iff (Membership.mem ((fun a b => Finset.filter (fun x => And (Not (LE.le x a …
    -/
    rw [Finset.mem_filter, mem_Icc, and_and_and_comm, lt_iff_le_not_le, lt_iff_le_not_le]
    /-
      🎉 no goals
    -/


/-- A constructor from a definition of `Finset.Icc` alone, the other ones being derived by removing
the ends. As opposed to `LocallyFiniteOrder.ofIcc'`, this one requires `PartialOrder` but only
`DecidableEq`. -/
def LocallyFiniteOrder.ofIcc (α : Type*) [PartialOrder α] [DecidableEq α]
    (finsetIcc : α → α → Finset α) (mem_Icc : ∀ a b x, x ∈ finsetIcc a b ↔ a ≤ x ∧ x ≤ b) :
    LocallyFiniteOrder α where
  finsetIcc := finsetIcc
  finsetIco a b := {x ∈ finsetIcc a b | x ≠ b}
  finsetIoc a b := {x ∈ finsetIcc a b | a ≠ x}
  finsetIoo a b := {x ∈ finsetIcc a b | a ≠ x ∧ x ≠ b}
  finset_mem_Icc := mem_Icc
                             /-
                               α : Type u_1
                               inst✝¹ : PartialOrder α
                               inst✝ : DecidableEq α
                               finsetIcc : α → α → Finset α
                               mem_Icc : ∀ (a b x : α), Iff (Membership.mem (finsetIcc a b) x) (And (LE.le a  …
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => Finset.filter (fun x => Ne x b) (finsetIcc  …
                             -/
  finset_mem_Ico a b x := by rw [Finset.mem_filter, mem_Icc, and_assoc, lt_iff_le_and_ne]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               inst✝¹ : PartialOrder α
                               inst✝ : DecidableEq α
                               finsetIcc : α → α → Finset α
                               mem_Icc : ∀ (a b x : α), Iff (Membership.mem (finsetIcc a b) x) (And (LE.le a  …
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => Finset.filter (fun x => Ne a x) (finsetIcc  …
                             -/
  finset_mem_Ioc a b x := by rw [Finset.mem_filter, mem_Icc, and_right_comm, lt_iff_le_and_ne]
                             /-
                               🎉 no goals
                             -/
  finset_mem_Ioo a b x := by
    /-
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : DecidableEq α
      finsetIcc : α → α → Finset α
      mem_Icc : ∀ (a b x : α), Iff (Membership.mem (finsetIcc a b) x) (And (LE.le a  …
      a b x : α
      ⊢ Iff (Membership.mem ((fun a b => Finset.filter (fun x => And (Ne a x) (Ne x  …
    -/
    rw [Finset.mem_filter, mem_Icc, and_and_and_comm, lt_iff_le_and_ne, lt_iff_le_and_ne]
    /-
      🎉 no goals
    -/


/-- A constructor from a definition of `Finset.Ici` alone, the other ones being derived by removing
the ends. As opposed to `LocallyFiniteOrderTop.ofIci`, this one requires `DecidableRel (· ≤ ·)` but
only `Preorder`. -/
def LocallyFiniteOrderTop.ofIci' (α : Type*) [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (finsetIci : α → Finset α) (mem_Ici : ∀ a x, x ∈ finsetIci a ↔ a ≤ x) :
    LocallyFiniteOrderTop α where
  finsetIci := finsetIci
  finsetIoi a := {x ∈ finsetIci a | ¬x ≤ a}
  finset_mem_Ici := mem_Ici
                           /-
                             α : Type u_1
                             inst✝¹ : Preorder α
                             inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                             finsetIci : α → Finset α
                             mem_Ici : ∀ (a x : α), Iff (Membership.mem (finsetIci a) x) (LE.le a x)
                             a x : α
                             ⊢ Iff (Membership.mem ((fun a => Finset.filter (fun x => Not (LE.le x a)) (fin …
                           -/
  finset_mem_Ioi a x := by rw [mem_filter, mem_Ici, lt_iff_le_not_le]
                           /-
                             🎉 no goals
                           -/


/-- A constructor from a definition of `Finset.Ici` alone, the other ones being derived by removing
the ends. As opposed to `LocallyFiniteOrderTop.ofIci'`, this one requires `PartialOrder` but
only `DecidableEq`. -/
def LocallyFiniteOrderTop.ofIci (α : Type*) [PartialOrder α] [DecidableEq α]
    (finsetIci : α → Finset α) (mem_Ici : ∀ a x, x ∈ finsetIci a ↔ a ≤ x) :
    LocallyFiniteOrderTop α where
  finsetIci := finsetIci
  finsetIoi a := {x ∈ finsetIci a | a ≠ x}
  finset_mem_Ici := mem_Ici
                           /-
                             α : Type u_1
                             inst✝¹ : PartialOrder α
                             inst✝ : DecidableEq α
                             finsetIci : α → Finset α
                             mem_Ici : ∀ (a x : α), Iff (Membership.mem (finsetIci a) x) (LE.le a x)
                             a x : α
                             ⊢ Iff (Membership.mem ((fun a => Finset.filter (fun x => Ne a x) (finsetIci a) …
                           -/
  finset_mem_Ioi a x := by rw [mem_filter, mem_Ici, lt_iff_le_and_ne]
                           /-
                             🎉 no goals
                           -/


/-- A constructor from a definition of `Finset.Iic` alone, the other ones being derived by removing
the ends. As opposed to `LocallyFiniteOrderBot.ofIic`, this one requires `DecidableRel (· ≤ ·)` but
only `Preorder`. -/
def LocallyFiniteOrderBot.ofIic' (α : Type*) [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (finsetIic : α → Finset α) (mem_Iic : ∀ a x, x ∈ finsetIic a ↔ x ≤ a) :
    LocallyFiniteOrderBot α where
  finsetIic := finsetIic
  finsetIio a := {x ∈ finsetIic a | ¬a ≤ x}
  finset_mem_Iic := mem_Iic
                           /-
                             α : Type u_1
                             inst✝¹ : Preorder α
                             inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                             finsetIic : α → Finset α
                             mem_Iic : ∀ (a x : α), Iff (Membership.mem (finsetIic a) x) (LE.le x a)
                             a x : α
                             ⊢ Iff (Membership.mem ((fun a => Finset.filter (fun x => Not (LE.le a x)) (fin …
                           -/
  finset_mem_Iio a x := by rw [mem_filter, mem_Iic, lt_iff_le_not_le]
                           /-
                             🎉 no goals
                           -/


/-- A constructor from a definition of `Finset.Iic` alone, the other ones being derived by removing
the ends. As opposed to `LocallyFiniteOrderBot.ofIic'`, this one requires `PartialOrder` but
only `DecidableEq`. -/
def LocallyFiniteOrderBot.ofIic (α : Type*) [PartialOrder α] [DecidableEq α]
    (finsetIic : α → Finset α) (mem_Iic : ∀ a x, x ∈ finsetIic a ↔ x ≤ a) :
    LocallyFiniteOrderBot α where
  finsetIic := finsetIic
  finsetIio a := {x ∈ finsetIic a | x ≠ a}
  finset_mem_Iic := mem_Iic
                           /-
                             α : Type u_1
                             inst✝¹ : PartialOrder α
                             inst✝ : DecidableEq α
                             finsetIic : α → Finset α
                             mem_Iic : ∀ (a x : α), Iff (Membership.mem (finsetIic a) x) (LE.le x a)
                             a x : α
                             ⊢ Iff (Membership.mem ((fun a => Finset.filter (fun x => Ne x a) (finsetIic a) …
                           -/
  finset_mem_Iio a x := by rw [mem_filter, mem_Iic, lt_iff_le_and_ne]
                           /-
                             🎉 no goals
                           -/


/-- An empty type is locally finite.

This is not an instance as it would not be defeq to more specific instances. -/
protected abbrev IsEmpty.toLocallyFiniteOrder [Preorder α] [IsEmpty α] : LocallyFiniteOrder α where
  finsetIcc := isEmptyElim
  finsetIco := isEmptyElim
  finsetIoc := isEmptyElim
  finsetIoo := isEmptyElim
  finset_mem_Icc := isEmptyElim
  finset_mem_Ico := isEmptyElim
  finset_mem_Ioc := isEmptyElim
  finset_mem_Ioo := isEmptyElim

-- See note [reducible non-instances]

/-- An empty type is locally finite.

This is not an instance as it would not be defeq to more specific instances. -/
protected abbrev IsEmpty.toLocallyFiniteOrderTop [Preorder α] [IsEmpty α] :
    LocallyFiniteOrderTop α where
  finsetIci := isEmptyElim
  finsetIoi := isEmptyElim
  finset_mem_Ici := isEmptyElim
  finset_mem_Ioi := isEmptyElim

-- See note [reducible non-instances]

/-- An empty type is locally finite.

This is not an instance as it would not be defeq to more specific instances. -/
protected abbrev IsEmpty.toLocallyFiniteOrderBot [Preorder α] [IsEmpty α] :
    LocallyFiniteOrderBot α where
  finsetIic := isEmptyElim
  finsetIio := isEmptyElim
  finset_mem_Iic := isEmptyElim
  finset_mem_Iio := isEmptyElim


/-- The finset $[a, b]$ of elements `x` such that `a ≤ x` and `x ≤ b`. Basically `Set.Icc a b` as a
finset. -/
def Icc (a b : α) : Finset α :=
  LocallyFiniteOrder.finsetIcc a b


/-- The finset $[a, b)$ of elements `x` such that `a ≤ x` and `x < b`. Basically `Set.Ico a b` as a
finset. -/
def Ico (a b : α) : Finset α :=
  LocallyFiniteOrder.finsetIco a b


/-- The finset $(a, b]$ of elements `x` such that `a < x` and `x ≤ b`. Basically `Set.Ioc a b` as a
finset. -/
def Ioc (a b : α) : Finset α :=
  LocallyFiniteOrder.finsetIoc a b


/-- The finset $(a, b)$ of elements `x` such that `a < x` and `x < b`. Basically `Set.Ioo a b` as a
finset. -/
def Ioo (a b : α) : Finset α :=
  LocallyFiniteOrder.finsetIoo a b


@[simp]
theorem mem_Icc : x ∈ Icc a b ↔ a ≤ x ∧ x ≤ b :=
  LocallyFiniteOrder.finset_mem_Icc a b x


@[simp]
theorem mem_Ico : x ∈ Ico a b ↔ a ≤ x ∧ x < b :=
  LocallyFiniteOrder.finset_mem_Ico a b x


@[simp]
theorem mem_Ioc : x ∈ Ioc a b ↔ a < x ∧ x ≤ b :=
  LocallyFiniteOrder.finset_mem_Ioc a b x


@[simp]
theorem mem_Ioo : x ∈ Ioo a b ↔ a < x ∧ x < b :=
  LocallyFiniteOrder.finset_mem_Ioo a b x


@[simp, norm_cast]
theorem coe_Icc (a b : α) : (Icc a b : Set α) = Set.Icc a b :=
  Set.ext fun _ => mem_Icc


@[simp, norm_cast]
theorem coe_Ico (a b : α) : (Ico a b : Set α) = Set.Ico a b :=
  Set.ext fun _ => mem_Ico


@[simp, norm_cast]
theorem coe_Ioc (a b : α) : (Ioc a b : Set α) = Set.Ioc a b :=
  Set.ext fun _ => mem_Ioc


@[simp, norm_cast]
theorem coe_Ioo (a b : α) : (Ioo a b : Set α) = Set.Ioo a b :=
  Set.ext fun _ => mem_Ioo


/-- The finset $[a, ∞)$ of elements `x` such that `a ≤ x`. Basically `Set.Ici a` as a finset. -/
def Ici (a : α) : Finset α :=
  LocallyFiniteOrderTop.finsetIci a


/-- The finset $(a, ∞)$ of elements `x` such that `a < x`. Basically `Set.Ioi a` as a finset. -/
def Ioi (a : α) : Finset α :=
  LocallyFiniteOrderTop.finsetIoi a


@[simp]
theorem mem_Ici : x ∈ Ici a ↔ a ≤ x :=
  LocallyFiniteOrderTop.finset_mem_Ici _ _


@[simp]
theorem mem_Ioi : x ∈ Ioi a ↔ a < x :=
  LocallyFiniteOrderTop.finset_mem_Ioi _ _


@[simp, norm_cast]
theorem coe_Ici (a : α) : (Ici a : Set α) = Set.Ici a :=
  Set.ext fun _ => mem_Ici


@[simp, norm_cast]
theorem coe_Ioi (a : α) : (Ioi a : Set α) = Set.Ioi a :=
  Set.ext fun _ => mem_Ioi


/-- The finset $(-∞, b]$ of elements `x` such that `x ≤ b`. Basically `Set.Iic b` as a finset. -/
def Iic (b : α) : Finset α :=
  LocallyFiniteOrderBot.finsetIic b


/-- The finset $(-∞, b)$ of elements `x` such that `x < b`. Basically `Set.Iio b` as a finset. -/
def Iio (b : α) : Finset α :=
  LocallyFiniteOrderBot.finsetIio b


@[simp]
theorem mem_Iic : x ∈ Iic a ↔ x ≤ a :=
  LocallyFiniteOrderBot.finset_mem_Iic _ _


@[simp]
theorem mem_Iio : x ∈ Iio a ↔ x < a :=
  LocallyFiniteOrderBot.finset_mem_Iio _ _


@[simp, norm_cast]
theorem coe_Iic (a : α) : (Iic a : Set α) = Set.Iic a :=
  Set.ext fun _ => mem_Iic


@[simp, norm_cast]
theorem coe_Iio (a : α) : (Iio a : Set α) = Set.Iio a :=
  Set.ext fun _ => mem_Iio


instance (priority := 100) _root_.LocallyFiniteOrder.toLocallyFiniteOrderTop :
    LocallyFiniteOrderTop α where
  finsetIci b := Icc b ⊤
  finsetIoi b := Ioc b ⊤
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             inst✝¹ : LocallyFiniteOrder α
                             inst✝ : OrderTop α
                             a✝ x✝ a x : α
                             ⊢ Iff (Membership.mem ((fun b => Finset.Icc b Top.top) a) x) (LE.le a x)
                           -/
  finset_mem_Ici a x := by rw [mem_Icc, and_iff_left le_top]
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             inst✝¹ : LocallyFiniteOrder α
                             inst✝ : OrderTop α
                             a✝ x✝ a x : α
                             ⊢ Iff (Membership.mem ((fun b => Finset.Ioc b Top.top) a) x) (LT.lt a x)
                           -/
  finset_mem_Ioi a x := by rw [mem_Ioc, and_iff_left le_top]
                           /-
                             🎉 no goals
                           -/


theorem Ici_eq_Icc (a : α) : Ici a = Icc a ⊤ :=
  rfl


theorem Ioi_eq_Ioc (a : α) : Ioi a = Ioc a ⊤ :=
  rfl


instance (priority := 100) LocallyFiniteOrder.toLocallyFiniteOrderBot :
    LocallyFiniteOrderBot α where
  finsetIic := Icc ⊥
  finsetIio := Ico ⊥
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             inst✝¹ : OrderBot α
                             inst✝ : LocallyFiniteOrder α
                             b x✝ a x : α
                             ⊢ Iff (Membership.mem (Finset.Icc Bot.bot a) x) (LE.le x a)
                           -/
  finset_mem_Iic a x := by rw [mem_Icc, and_iff_right bot_le]
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             inst✝¹ : OrderBot α
                             inst✝ : LocallyFiniteOrder α
                             b x✝ a x : α
                             ⊢ Iff (Membership.mem (Finset.Ico Bot.bot a) x) (LT.lt x a)
                           -/
  finset_mem_Iio a x := by rw [mem_Ico, and_iff_right bot_le]
                           /-
                             🎉 no goals
                           -/


theorem Iic_eq_Icc : Iic = Icc (⊥ : α) :=
  rfl


theorem Iio_eq_Ico : Iio = Ico (⊥ : α) :=
  rfl


/-- `Finset.uIcc a b` is the set of elements lying between `a` and `b`, with `a` and `b` included.
Note that we define it more generally in a lattice as `Finset.Icc (a ⊓ b) (a ⊔ b)`. In a
product type, `Finset.uIcc` corresponds to the bounding box of the two elements. -/
def uIcc (a b : α) : Finset α :=
  Icc (a ⊓ b) (a ⊔ b)


@[inherit_doc]
scoped[FinsetInterval] notation "[[" a ", " b "]]" => Finset.uIcc a b


@[simp]
theorem mem_uIcc : x ∈ uIcc a b ↔ a ⊓ b ≤ x ∧ x ≤ a ⊔ b :=
  mem_Icc


@[simp, norm_cast]
theorem coe_uIcc (a b : α) : (Finset.uIcc a b : Set α) = Set.uIcc a b :=
  coe_Icc _ _


/-- Elaborate set builder notation for `Finset`.

* `{x ≤ a | p x}` is elaborated as `Finset.filter (fun x ↦ p x) (Finset.Iic a)` if the expected type
  is `Finset ?α`.
* `{x ≥ a | p x}` is elaborated as `Finset.filter (fun x ↦ p x) (Finset.Ici a)` if the expected type
  is `Finset ?α`.
* `{x < a | p x}` is elaborated as `Finset.filter (fun x ↦ p x) (Finset.Iio a)` if the expected type
  is `Finset ?α`.
* `{x > a | p x}` is elaborated as `Finset.filter (fun x ↦ p x) (Finset.Ioi a)` if the expected type
  is `Finset ?α`.

See also
* `Data.Set.Defs` for the `Set` builder notation elaborator that this elaborator partly overrides.
* `Data.Finset.Basic` for the `Finset` builder notation elaborator partly overriding this one for
  syntax of the form `{x ∈ s | p x}`.
* `Data.Fintype.Basic` for the `Finset` builder notation elaborator handling syntax of the form
  `{x | p x}`, `{x : α | p x}`, `{x ∉ s | p x}`, `{x ≠ a | p x}`.

TODO: Write a delaborator
-/
@[term_elab setBuilder]
def elabFinsetBuilderIxx : TermElab
  | `({ $x:ident ≤ $a | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) (Finset.Iic $a))) expectedType?
  | `({ $x:ident ≥ $a | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) (Finset.Ici $a))) expectedType?
  | `({ $x:ident < $a | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) (Finset.Iio $a))) expectedType?
  | `({ $x:ident > $a | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) (Finset.Ioi $a))) expectedType?
  | _, _ => throwUnsupportedSyntax


instance fintypeIcc : Fintype (Icc a b) := Fintype.ofFinset (Finset.Icc a b) fun _ => Finset.mem_Icc


instance fintypeIco : Fintype (Ico a b) := Fintype.ofFinset (Finset.Ico a b) fun _ => Finset.mem_Ico


instance fintypeIoc : Fintype (Ioc a b) := Fintype.ofFinset (Finset.Ioc a b) fun _ => Finset.mem_Ioc


instance fintypeIoo : Fintype (Ioo a b) := Fintype.ofFinset (Finset.Ioo a b) fun _ => Finset.mem_Ioo


theorem finite_Icc : (Icc a b).Finite :=
  (Icc a b).toFinite


theorem finite_Ico : (Ico a b).Finite :=
  (Ico a b).toFinite


theorem finite_Ioc : (Ioc a b).Finite :=
  (Ioc a b).toFinite


theorem finite_Ioo : (Ioo a b).Finite :=
  (Ioo a b).toFinite


instance fintypeIci : Fintype (Ici a) := Fintype.ofFinset (Finset.Ici a) fun _ => Finset.mem_Ici


instance fintypeIoi : Fintype (Ioi a) := Fintype.ofFinset (Finset.Ioi a) fun _ => Finset.mem_Ioi


theorem finite_Ici : (Ici a).Finite :=
  (Ici a).toFinite


theorem finite_Ioi : (Ioi a).Finite :=
  (Ioi a).toFinite


instance fintypeIic : Fintype (Iic b) := Fintype.ofFinset (Finset.Iic b) fun _ => Finset.mem_Iic


instance fintypeIio : Fintype (Iio b) := Fintype.ofFinset (Finset.Iio b) fun _ => Finset.mem_Iio


theorem finite_Iic : (Iic b).Finite :=
  (Iic b).toFinite


theorem finite_Iio : (Iio b).Finite :=
  (Iio b).toFinite


instance fintypeUIcc : Fintype (uIcc a b) :=
  Fintype.ofFinset (Finset.uIcc a b) fun _ => Finset.mem_uIcc


@[simp]
theorem finite_interval : (uIcc a b).Finite := (uIcc _ _).toFinite


/-- A noncomputable constructor from the finiteness of all closed intervals. -/
noncomputable def LocallyFiniteOrder.ofFiniteIcc (h : ∀ a b : α, (Set.Icc a b).Finite) :
    LocallyFiniteOrder α :=
  @LocallyFiniteOrder.ofIcc' α _ (Classical.decRel _) (fun a b => (h a b).toFinset) fun a b x => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h : ∀ (a b : α), (Set.Icc a b).Finite
      a b x : α
      ⊢ Iff (Membership.mem ((fun a b => ⋯.toFinset) a b) x) (And (LE.le a x) (LE.le …
    -/
    rw [Set.Finite.mem_toFinset, Set.mem_Icc]
    /-
      🎉 no goals
    -/


/-- A fintype is a locally finite order.

This is not an instance as it would not be defeq to better instances such as
`Fin.locallyFiniteOrder`.
-/
abbrev Fintype.toLocallyFiniteOrder [Fintype α] [DecidableRel (α := α) (· < ·)]
    [DecidableRel (α := α) (· ≤ ·)] : LocallyFiniteOrder α where
  finsetIcc a b := (Set.Icc a b).toFinset
  finsetIco a b := (Set.Ico a b).toFinset
  finsetIoc a b := (Set.Ioc a b).toFinset
  finsetIoo a b := (Set.Ioo a b).toFinset
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝⁴ : Preorder α
                               inst✝³ : Preorder β
                               inst✝² : Fintype α
                               inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
                               inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Set.Icc a b).toFinset) a b) x) (And (LE.le …
                             -/
  finset_mem_Icc a b x := by simp only [Set.mem_toFinset, Set.mem_Icc]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝⁴ : Preorder α
                               inst✝³ : Preorder β
                               inst✝² : Fintype α
                               inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
                               inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Set.Ico a b).toFinset) a b) x) (And (LE.le …
                             -/
  finset_mem_Ico a b x := by simp only [Set.mem_toFinset, Set.mem_Ico]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝⁴ : Preorder α
                               inst✝³ : Preorder β
                               inst✝² : Fintype α
                               inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
                               inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Set.Ioc a b).toFinset) a b) x) (And (LT.lt …
                             -/
  finset_mem_Ioc a b x := by simp only [Set.mem_toFinset, Set.mem_Ioc]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝⁴ : Preorder α
                               inst✝³ : Preorder β
                               inst✝² : Fintype α
                               inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
                               inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Set.Ioo a b).toFinset) a b) x) (And (LT.lt …
                             -/
  finset_mem_Ioo a b x := by simp only [Set.mem_toFinset, Set.mem_Ioo]
                             /-
                               🎉 no goals
                             -/


instance : Subsingleton (LocallyFiniteOrder α) :=
  Subsingleton.intro fun h₀ h₁ => by
    cases' h₀ with h₀_finset_Icc h₀_finset_Ico h₀_finset_Ioc h₀_finset_Ioo
      h₀_finset_mem_Icc h₀_finset_mem_Ico h₀_finset_mem_Ioc h₀_finset_mem_Ioo
    cases' h₁ with h₁_finset_Icc h₁_finset_Ico h₁_finset_Ioc h₁_finset_Ioo
      h₁_finset_mem_Icc h₁_finset_mem_Ico h₁_finset_mem_Ioc h₁_finset_mem_Ioo
    have hIcc : h₀_finset_Icc = h₁_finset_Icc := by
      ext a b x
      rw [h₀_finset_mem_Icc, h₁_finset_mem_Icc]
    have hIco : h₀_finset_Ico = h₁_finset_Ico := by
      ext a b x
      rw [h₀_finset_mem_Ico, h₁_finset_mem_Ico]
    have hIoc : h₀_finset_Ioc = h₁_finset_Ioc := by
      ext a b x
      rw [h₀_finset_mem_Ioc, h₁_finset_mem_Ioc]
    have hIoo : h₀_finset_Ioo = h₁_finset_Ioo := by
      ext a b x
      rw [h₀_finset_mem_Ioo, h₁_finset_mem_Ioo]
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₀_finset_Icc h₀_finset_Ico h₀_finset_Ioc h₀_finset_Ioo : α → α → Finset α
      h₀_finset_mem_Icc : ∀ (a b x : α), Iff (Membership.mem (h₀_finset_Icc a b) x)  …
      h₀_finset_mem_Ico : ∀ (a b x : α), Iff (Membership.mem (h₀_finset_Ico a b) x)  …
      h₀_finset_mem_Ioc : ∀ (a b x : α), Iff (Membership.mem (h₀_finset_Ioc a b) x)  …
      h₀_finset_mem_Ioo : ∀ (a b x : α), Iff (Membership.mem (h₀_finset_Ioo a b) x)  …
      h₁_finset_Icc h₁_finset_Ico h₁_finset_Ioc h₁_finset_Ioo : α → α → Finset α
      h₁_finset_mem_Icc : ∀ (a b x : α), Iff (Membership.mem (h₁_finset_Icc a b) x)  …
      h₁_finset_mem_Ico : ∀ (a b x : α), Iff (Membership.mem (h₁_finset_Ico a b) x)  …
      h₁_finset_mem_Ioc : ∀ (a b x : α), Iff (Membership.mem (h₁_finset_Ioc a b) x)  …
      h₁_finset_mem_Ioo : ∀ (a b x : α), Iff (Membership.mem (h₁_finset_Ioo a b) x)  …
      hIcc : Eq h₀_finset_Icc h₁_finset_Icc
      hIco : Eq h₀_finset_Ico h₁_finset_Ico
      hIoc : Eq h₀_finset_Ioc h₁_finset_Ioc
      hIoo : Eq h₀_finset_Ioo h₁_finset_Ioo
      ⊢ Eq { finsetIcc := h₀_finset_Icc, finsetIco := h₀_finset_Ico, finsetIoc := h₀ …
    -/
    simp_rw [hIcc, hIco, hIoc, hIoo]
    /-
      🎉 no goals
    -/


instance : Subsingleton (LocallyFiniteOrderTop α) :=
  Subsingleton.intro fun h₀ h₁ => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₀ h₁ : LocallyFiniteOrderTop α
      ⊢ Eq h₀ h₁
    -/
    cases' h₀ with h₀_finset_Ioi h₀_finset_Ici h₀_finset_mem_Ici h₀_finset_mem_Ioi
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₁ : LocallyFiniteOrderTop α
      h₀_finset_Ioi h₀_finset_Ici : α → Finset α
      h₀_finset_mem_Ici : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Ici a) x) (LE. …
      h₀_finset_mem_Ioi : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Ioi a) x) (LT. …
      ⊢ Eq { finsetIoi := h₀_finset_Ioi, finsetIci := h₀_finset_Ici, finset_mem_Ici  …
    -/
    cases' h₁ with h₁_finset_Ioi h₁_finset_Ici h₁_finset_mem_Ici h₁_finset_mem_Ioi
    have hIci : h₀_finset_Ici = h₁_finset_Ici := by
      ext a b
      rw [h₀_finset_mem_Ici, h₁_finset_mem_Ici]
    have hIoi : h₀_finset_Ioi = h₁_finset_Ioi := by
      ext a b
      rw [h₀_finset_mem_Ioi, h₁_finset_mem_Ioi]
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₀_finset_Ioi h₀_finset_Ici : α → Finset α
      h₀_finset_mem_Ici : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Ici a) x) (LE. …
      h₀_finset_mem_Ioi : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Ioi a) x) (LT. …
      h₁_finset_Ioi h₁_finset_Ici : α → Finset α
      h₁_finset_mem_Ici : ∀ (a x : α), Iff (Membership.mem (h₁_finset_Ici a) x) (LE. …
      h₁_finset_mem_Ioi : ∀ (a x : α), Iff (Membership.mem (h₁_finset_Ioi a) x) (LT. …
      hIci : Eq h₀_finset_Ici h₁_finset_Ici
      hIoi : Eq h₀_finset_Ioi h₁_finset_Ioi
      ⊢ Eq { finsetIoi := h₀_finset_Ioi, finsetIci := h₀_finset_Ici, finset_mem_Ici  …
    -/
    simp_rw [hIci, hIoi]
    /-
      🎉 no goals
    -/


instance : Subsingleton (LocallyFiniteOrderBot α) :=
  Subsingleton.intro fun h₀ h₁ => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₀ h₁ : LocallyFiniteOrderBot α
      ⊢ Eq h₀ h₁
    -/
    cases' h₀ with h₀_finset_Iio h₀_finset_Iic h₀_finset_mem_Iic h₀_finset_mem_Iio
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₁ : LocallyFiniteOrderBot α
      h₀_finset_Iio h₀_finset_Iic : α → Finset α
      h₀_finset_mem_Iic : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Iic a) x) (LE. …
      h₀_finset_mem_Iio : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Iio a) x) (LT. …
      ⊢ Eq { finsetIio := h₀_finset_Iio, finsetIic := h₀_finset_Iic, finset_mem_Iic  …
    -/
    cases' h₁ with h₁_finset_Iio h₁_finset_Iic h₁_finset_mem_Iic h₁_finset_mem_Iio
    have hIic : h₀_finset_Iic = h₁_finset_Iic := by
      ext a b
      rw [h₀_finset_mem_Iic, h₁_finset_mem_Iic]
    have hIio : h₀_finset_Iio = h₁_finset_Iio := by
      ext a b
      rw [h₀_finset_mem_Iio, h₁_finset_mem_Iio]
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      h₀_finset_Iio h₀_finset_Iic : α → Finset α
      h₀_finset_mem_Iic : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Iic a) x) (LE. …
      h₀_finset_mem_Iio : ∀ (a x : α), Iff (Membership.mem (h₀_finset_Iio a) x) (LT. …
      h₁_finset_Iio h₁_finset_Iic : α → Finset α
      h₁_finset_mem_Iic : ∀ (a x : α), Iff (Membership.mem (h₁_finset_Iic a) x) (LE. …
      h₁_finset_mem_Iio : ∀ (a x : α), Iff (Membership.mem (h₁_finset_Iio a) x) (LT. …
      hIic : Eq h₀_finset_Iic h₁_finset_Iic
      hIio : Eq h₀_finset_Iio h₁_finset_Iio
      ⊢ Eq { finsetIio := h₀_finset_Iio, finsetIic := h₀_finset_Iic, finset_mem_Iic  …
    -/
    simp_rw [hIic, hIio]
    /-
      🎉 no goals
    -/

-- Should this be called `LocallyFiniteOrder.lift`?

/-- Given an order embedding `α ↪o β`, pulls back the `LocallyFiniteOrder` on `β` to `α`. -/
protected noncomputable def OrderEmbedding.locallyFiniteOrder [LocallyFiniteOrder β] (f : α ↪o β) :
    LocallyFiniteOrder α where
  finsetIcc a b := (Icc (f a) (f b)).preimage f f.toEmbedding.injective.injOn
  finsetIco a b := (Ico (f a) (f b)).preimage f f.toEmbedding.injective.injOn
  finsetIoc a b := (Ioc (f a) (f b)).preimage f f.toEmbedding.injective.injOn
  finsetIoo a b := (Ioo (f a) (f b)).preimage f f.toEmbedding.injective.injOn
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : Preorder α
                               inst✝¹ : Preorder β
                               inst✝ : LocallyFiniteOrder β
                               f : OrderEmbedding α β
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Finset.Icc (f a) (f b)).preimage ⇑f ⋯) a b …
                             -/
  finset_mem_Icc a b x := by rw [mem_preimage, mem_Icc, f.le_iff_le, f.le_iff_le]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : Preorder α
                               inst✝¹ : Preorder β
                               inst✝ : LocallyFiniteOrder β
                               f : OrderEmbedding α β
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Finset.Ico (f a) (f b)).preimage ⇑f ⋯) a b …
                             -/
  finset_mem_Ico a b x := by rw [mem_preimage, mem_Ico, f.le_iff_le, f.lt_iff_lt]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : Preorder α
                               inst✝¹ : Preorder β
                               inst✝ : LocallyFiniteOrder β
                               f : OrderEmbedding α β
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Finset.Ioc (f a) (f b)).preimage ⇑f ⋯) a b …
                             -/
  finset_mem_Ioc a b x := by rw [mem_preimage, mem_Ioc, f.lt_iff_lt, f.le_iff_le]
                             /-
                               🎉 no goals
                             -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : Preorder α
                               inst✝¹ : Preorder β
                               inst✝ : LocallyFiniteOrder β
                               f : OrderEmbedding α β
                               a b x : α
                               ⊢ Iff (Membership.mem ((fun a b => (Finset.Ioo (f a) (f b)).preimage ⇑f ⋯) a b …
                             -/
  finset_mem_Ioo a b x := by rw [mem_preimage, mem_Ioo, f.lt_iff_lt, f.lt_iff_lt]
                             /-
                               🎉 no goals
                             -/


/-- Note we define `Icc (toDual a) (toDual b)` as `Icc α _ _ b a` (which has type `Finset α` not
`Finset αᵒᵈ`!) instead of `(Icc b a).map toDual.toEmbedding` as this means the
following is defeq:
```
lemma this : (Icc (toDual (toDual a)) (toDual (toDual b)) : _) = (Icc a b : _) := rfl
```
-/
instance OrderDual.instLocallyFiniteOrder : LocallyFiniteOrder αᵒᵈ where
  finsetIcc a b := @Icc α _ _ (ofDual b) (ofDual a)
  finsetIco a b := @Ioc α _ _ (ofDual b) (ofDual a)
  finsetIoc a b := @Ico α _ _ (ofDual b) (ofDual a)
  finsetIoo a b := @Ioo α _ _ (ofDual b) (ofDual a)
  finset_mem_Icc _ _ _ := (mem_Icc (α := α)).trans and_comm
  finset_mem_Ico _ _ _ := (mem_Ioc (α := α)).trans and_comm
  finset_mem_Ioc _ _ _ := (mem_Ico (α := α)).trans and_comm
  finset_mem_Ioo _ _ _ := (mem_Ioo (α := α)).trans and_comm


lemma Finset.Icc_orderDual_def (a b : αᵒᵈ) :
    Icc a b = (Icc (ofDual b) (ofDual a)).map toDual.toEmbedding := map_refl.symm


lemma Finset.Ico_orderDual_def (a b : αᵒᵈ) :
    Ico a b = (Ioc (ofDual b) (ofDual a)).map toDual.toEmbedding := map_refl.symm


lemma Finset.Ioc_orderDual_def (a b : αᵒᵈ) :
    Ioc a b = (Ico (ofDual b) (ofDual a)).map toDual.toEmbedding := map_refl.symm


lemma Finset.Ioo_orderDual_def (a b : αᵒᵈ) :
    Ioo a b = (Ioo (ofDual b) (ofDual a)).map toDual.toEmbedding := map_refl.symm


lemma Finset.Icc_toDual : Icc (toDual a) (toDual b) = (Icc b a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Ico_toDual : Ico (toDual a) (toDual b) = (Ioc b a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Ioc_toDual : Ioc (toDual a) (toDual b) = (Ico b a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Ioo_toDual : Ioo (toDual a) (toDual b) = (Ioo b a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Icc_ofDual (a b : αᵒᵈ) :
    Icc (ofDual a) (ofDual b) = (Icc b a).map ofDual.toEmbedding := map_refl.symm


lemma Finset.Ico_ofDual (a b : αᵒᵈ) :
    Ico (ofDual a) (ofDual b) = (Ioc b a).map ofDual.toEmbedding := map_refl.symm


lemma Finset.Ioc_ofDual (a b : αᵒᵈ) :
    Ioc (ofDual a) (ofDual b) = (Ico b a).map ofDual.toEmbedding := map_refl.symm


lemma Finset.Ioo_ofDual (a b : αᵒᵈ) :
    Ioo (ofDual a) (ofDual b) = (Ioo b a).map ofDual.toEmbedding := map_refl.symm


/-- Note we define `Iic (toDual a)` as `Ici a` (which has type `Finset α` not `Finset αᵒᵈ`!)
instead of `(Ici a).map toDual.toEmbedding` as this means the following is defeq:
```
lemma this : (Iic (toDual (toDual a)) : _) = (Iic a : _) := rfl
```
-/
instance OrderDual.instLocallyFiniteOrderBot : LocallyFiniteOrderBot αᵒᵈ where
  finsetIic a := @Ici α _ _ (ofDual a)
  finsetIio a := @Ioi α _ _ (ofDual a)
  finset_mem_Iic _ _ := mem_Ici (α := α)
  finset_mem_Iio _ _ := mem_Ioi (α := α)


lemma Iic_orderDual_def (a : αᵒᵈ) : Iic a = (Ici (ofDual a)).map toDual.toEmbedding := map_refl.symm

lemma Iio_orderDual_def (a : αᵒᵈ) : Iio a = (Ioi (ofDual a)).map toDual.toEmbedding := map_refl.symm


lemma Finset.Iic_toDual (a : α) : Iic (toDual a) = (Ici a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Iio_toDual (a : α) : Iio (toDual a) = (Ioi a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Ici_ofDual (a : αᵒᵈ) : Ici (ofDual a) = (Iic a).map ofDual.toEmbedding :=
  map_refl.symm


lemma Finset.Ioi_ofDual (a : αᵒᵈ) : Ioi (ofDual a) = (Iio a).map ofDual.toEmbedding :=
  map_refl.symm


/-- Note we define `Ici (toDual a)` as `Iic a` (which has type `Finset α` not `Finset αᵒᵈ`!)
instead of `(Iic a).map toDual.toEmbedding` as this means the following is defeq:
```
lemma this : (Ici (toDual (toDual a)) : _) = (Ici a : _) := rfl
```
-/
instance OrderDual.instLocallyFiniteOrderTop : LocallyFiniteOrderTop αᵒᵈ where
  finsetIci a := @Iic α _ _ (ofDual a)
  finsetIoi a := @Iio α _ _ (ofDual a)
  finset_mem_Ici _ _ := mem_Iic (α := α)
  finset_mem_Ioi _ _ := mem_Iio (α := α)


lemma Ici_orderDual_def (a : αᵒᵈ) : Ici a = (Iic (ofDual a)).map toDual.toEmbedding := map_refl.symm

lemma Ioi_orderDual_def (a : αᵒᵈ) : Ioi a = (Iio (ofDual a)).map toDual.toEmbedding := map_refl.symm


lemma Finset.Ici_toDual (a : α) : Ici (toDual a) = (Iic a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Ioi_toDual (a : α) : Ioi (toDual a) = (Iio a).map toDual.toEmbedding :=
  map_refl.symm


lemma Finset.Iic_ofDual (a : αᵒᵈ) : Iic (ofDual a) = (Ici a).map ofDual.toEmbedding :=
  map_refl.symm


lemma Finset.Iio_ofDual (a : αᵒᵈ) : Iio (ofDual a) = (Ioi a).map ofDual.toEmbedding :=
  map_refl.symm


instance Prod.instLocallyFiniteOrder : LocallyFiniteOrder (α × β) :=
  LocallyFiniteOrder.ofIcc' (α × β) (fun x y ↦ Icc x.1 y.1 ×ˢ Icc x.2 y.2) fun a b x => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      inst✝³ : Preorder β
      inst✝² : LocallyFiniteOrder α
      inst✝¹ : LocallyFiniteOrder β
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      a b x : Prod α β
      ⊢ Iff (Membership.mem ((fun x y => SProd.sprod (Finset.Icc x.1 y.1) (Finset.Ic …
    -/
    rw [mem_product, mem_Icc, mem_Icc, and_and_and_comm, le_def, le_def]
    /-
      🎉 no goals
    -/


lemma Finset.Icc_prod_def (x y : α × β) : Icc x y = Icc x.1 y.1 ×ˢ Icc x.2 y.2 := rfl


lemma Finset.Icc_product_Icc (a₁ a₂ : α) (b₁ b₂ : β) :
    Icc a₁ a₂ ×ˢ Icc b₁ b₂ = Icc (a₁, b₁) (a₂, b₂) := rfl


lemma Finset.card_Icc_prod (x y : α × β) : #(Icc x y) = #(Icc x.1 y.1) * #(Icc x.2 y.2) :=
  card_product ..


instance Prod.instLocallyFiniteOrderTop : LocallyFiniteOrderTop (α × β) :=
  LocallyFiniteOrderTop.ofIci' (α × β) (fun x => Ici x.1 ×ˢ Ici x.2) fun a x => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      inst✝³ : Preorder β
      inst✝² : LocallyFiniteOrderTop α
      inst✝¹ : LocallyFiniteOrderTop β
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      a x : Prod α β
      ⊢ Iff (Membership.mem ((fun x => SProd.sprod (Finset.Ici x.1) (Finset.Ici x.2) …
    -/
    rw [mem_product, mem_Ici, mem_Ici, le_def]
    /-
      🎉 no goals
    -/


lemma Finset.Ici_prod_def (x : α × β) : Ici x = Ici x.1 ×ˢ Ici x.2 := rfl

lemma Finset.Ici_product_Ici (a : α) (b : β) : Ici a ×ˢ Ici b = Ici (a, b) := rfl

lemma Finset.card_Ici_prod (x : α × β) : #(Ici x) = #(Ici x.1) * #(Ici x.2) :=
  card_product _ _


instance Prod.instLocallyFiniteOrderBot : LocallyFiniteOrderBot (α × β) :=
  LocallyFiniteOrderBot.ofIic' (α × β) (fun x ↦ Iic x.1 ×ˢ Iic x.2) fun a x ↦ by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      inst✝³ : Preorder β
      inst✝² : LocallyFiniteOrderBot α
      inst✝¹ : LocallyFiniteOrderBot β
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      a x : Prod α β
      ⊢ Iff (Membership.mem ((fun x => SProd.sprod (Finset.Iic x.1) (Finset.Iic x.2) …
    -/
    rw [mem_product, mem_Iic, mem_Iic, le_def]
    /-
      🎉 no goals
    -/


lemma Finset.Iic_prod_def (x : α × β) : Iic x = Iic x.1 ×ˢ Iic x.2 := rfl

lemma Finset.Iic_product_Iic (a : α) (b : β) : Iic a ×ˢ Iic b = Iic (a, b) := rfl

lemma Finset.card_Iic_prod (x : α × β) : #(Iic x) = #(Iic x.1) * #(Iic x.2) := card_product ..


lemma Finset.uIcc_prod_def (x y : α × β) : uIcc x y = uIcc x.1 y.1 ×ˢ uIcc x.2 y.2 := rfl


lemma Finset.uIcc_product_uIcc (a₁ a₂ : α) (b₁ b₂ : β) :
    uIcc a₁ a₂ ×ˢ uIcc b₁ b₂ = uIcc (a₁, b₁) (a₂, b₂) := rfl


lemma Finset.card_uIcc_prod (x y : α × β) : #(uIcc x y) = #(uIcc x.1 y.1) * #(uIcc x.2 y.2) :=
  card_product ..


private lemma aux (x : α) (p : α → Prop) :
    (∃ a : α, p a ∧ WithTop.some a = WithTop.some x) ↔ p x := by
  /-
    α : Type u_1
    x : α
    p : α → Prop
    ⊢ Iff (Exists fun a => And (p a) (Eq ↑a ↑x)) (p x)
  -/
  simp
  /-
    🎉 no goals
  -/


instance locallyFiniteOrder : LocallyFiniteOrder (WithTop α) where
  finsetIcc a b :=
    match a, b with
    | ⊤, ⊤ => {⊤}
    | ⊤, (b : α) => ∅
    | (a : α), ⊤ => insertNone (Ici a)
    | (a : α), (b : α) => (Icc a b).map Embedding.some
  finsetIco a b :=
    match a, b with
    | ⊤, _ => ∅
    | (a : α), ⊤ => (Ici a).map Embedding.some
    | (a : α), (b : α) => (Ico a b).map Embedding.some
  finsetIoc a b :=
    match a, b with
    | ⊤, _ => ∅
    | (a : α), ⊤ => insertNone (Ioi a)
    | (a : α), (b : α) => (Ioc a b).map Embedding.some
  finsetIoo a b :=
    match a, b with
    | ⊤, _ => ∅
    | (a : α), ⊤ => (Ioi a).map Embedding.some
    | (a : α), (b : α) => (Ioo a b).map Embedding.some
  -- Porting note: the proofs below got much worse
  finset_mem_Icc a b x :=
    match a, b, x with
    | ⊤, ⊤, _ => mem_singleton.trans (le_antisymm_iff.trans and_comm)
    | ⊤, (b : α), _ =>
      iff_of_false (not_mem_empty _) fun h => (h.1.trans h.2).not_lt <| coe_lt_top _
                          /-
                            α : Type u_1
                            β : Type u_2
                            inst✝² : PartialOrder α
                            inst✝¹ : OrderTop α
                            inst✝ : LocallyFiniteOrder α
                            a✝ b x : WithTop α
                            a : α
                            ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_1 α (fun a …
                          -/
    | (a : α), ⊤, ⊤ => by simp [WithTop.some, WithTop.top, insertNone]
                          /-
                            🎉 no goals
                          -/
    | (a : α), ⊤, (x : α) => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝² : PartialOrder α
          inst✝¹ : OrderTop α
          inst✝ : LocallyFiniteOrder α
          a✝ b x✝ : WithTop α
          a x : α
          ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_1 α (fun a …
        -/
        simp only [le_eq_subset, coe_le_coe, le_top, and_true]
        /-
          α : Type u_1
          β : Type u_2
          inst✝² : PartialOrder α
          inst✝¹ : OrderTop α
          inst✝ : LocallyFiniteOrder α
          a✝ b x✝ : WithTop α
          a x : α
          ⊢ Iff (Membership.mem (Finset.insertNone (Finset.Ici a)) ↑x) (LE.le a x)
        -/
        rw [← some_eq_coe, some_mem_insertNone, mem_Ici]
        /-
          🎉 no goals
        -/
    | (a : α), (b : α), ⊤ => by
        simp only [Embedding.some, mem_map, mem_Icc, and_false, exists_const, some, le_top,
          top_le_iff, reduceCtorEq]
    | (a : α), (b : α), (x : α) => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝² : PartialOrder α
          inst✝¹ : OrderTop α
          inst✝ : LocallyFiniteOrder α
          a✝ b✝ x✝ : WithTop α
          a b x : α
          ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_1 α (fun a …
        -/
        simp only [le_eq_subset, Embedding.some, mem_map, mem_Icc, Embedding.coeFn_mk, coe_le_coe]
        -- This used to be in the above `simp` before https://github.com/leanprover/lean4/pull/2644
        /-
          α : Type u_1
          β : Type u_2
          inst✝² : PartialOrder α
          inst✝¹ : OrderTop α
          inst✝ : LocallyFiniteOrder α
          a✝ b✝ x✝ : WithTop α
          a b x : α
          ⊢ Iff (Exists fun a_1 => And (And (LE.le a a_1) (LE.le a_1 b)) (Eq ({ toFun := …
        -/
        erw [aux]
        /-
          🎉 no goals
        -/
  finset_mem_Ico a b x :=
    match a, b, x with
    | ⊤, _, _ => iff_of_false (not_mem_empty _) fun h => not_top_lt <| h.1.trans_lt h.2
                          /-
                            α : Type u_1
                            β : Type u_2
                            inst✝² : PartialOrder α
                            inst✝¹ : OrderTop α
                            inst✝ : LocallyFiniteOrder α
                            a✝ b x : WithTop α
                            a : α
                            ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                          -/
    | (a : α), ⊤, ⊤ => by simp [some, Embedding.some]
                          /-
                            🎉 no goals
                          -/
    | (a : α), ⊤, (x : α) => by
        simp only [Embedding.some, mem_map, mem_Ici, Embedding.coeFn_mk, coe_le_coe, aux,
          coe_lt_top, and_true]
        -- This used to be in the above `simp` before https://github.com/leanprover/lean4/pull/2644
        /-
          α : Type u_1
          β : Type u_2
          inst✝² : PartialOrder α
          inst✝¹ : OrderTop α
          inst✝ : LocallyFiniteOrder α
          a✝ b x✝ : WithTop α
          a x : α
          ⊢ Iff (Exists fun a_1 => And (LE.le a a_1) (Eq ({ toFun := Option.some, inj' : …
        -/
        erw [aux]
        /-
          🎉 no goals
        -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b✝ x : WithTop α
                                  a b : α
                                  ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                -/
    | (a : α), (b : α), ⊤ => by simp [some, Embedding.some]
                                /-
                                  🎉 no goals
                                -/
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝² : PartialOrder α
                                        inst✝¹ : OrderTop α
                                        inst✝ : LocallyFiniteOrder α
                                        a✝ b✝ x✝ : WithTop α
                                        a b x : α
                                        ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                      -/
    | (a : α), (b : α), (x : α) => by simp [some, Embedding.some, aux]
                                      -- This used to be in the above `simp` before
                                      -- https://github.com/leanprover/lean4/pull/2644
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝² : PartialOrder α
                                        inst✝¹ : OrderTop α
                                        inst✝ : LocallyFiniteOrder α
                                        a✝ b✝ x✝ : WithTop α
                                        a b x : α
                                        ⊢ Iff (Exists fun a_1 => And (And (LE.le a a_1) (LT.lt a_1 b)) (Eq ({ toFun := …
                                      -/
                                      erw [aux]
                                      /-
                                        🎉 no goals
                                      -/
  finset_mem_Ioc a b x :=
    match a, b, x with
    | ⊤, _, _ => iff_of_false (not_mem_empty _) fun h => not_top_lt <| h.1.trans_le h.2
                          /-
                            α : Type u_1
                            β : Type u_2
                            inst✝² : PartialOrder α
                            inst✝¹ : OrderTop α
                            inst✝ : LocallyFiniteOrder α
                            a✝ b x : WithTop α
                            a : α
                            ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                          -/
    | (a : α), ⊤, ⊤ => by simp [some, insertNone, top]
                          /-
                            🎉 no goals
                          -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b x✝ : WithTop α
                                  a x : α
                                  ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                -/
    | (a : α), ⊤, (x : α) => by simp [some, Embedding.some, insertNone, aux]
                                -- This used to be in the above `simp` before
                                -- https://github.com/leanprover/lean4/pull/2644
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b x✝ : WithTop α
                                  a x : α
                                  ⊢ Iff (Exists fun a_1 => And (LT.lt a a_1) (Eq ({ toFun := Option.some, inj' : …
                                -/
                                erw [aux]
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b✝ x : WithTop α
                                  a b : α
                                  ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                -/
    | (a : α), (b : α), ⊤ => by simp [some, Embedding.some, insertNone]
                                /-
                                  🎉 no goals
                                -/
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝² : PartialOrder α
                                        inst✝¹ : OrderTop α
                                        inst✝ : LocallyFiniteOrder α
                                        a✝ b✝ x✝ : WithTop α
                                        a b x : α
                                        ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                      -/
    | (a : α), (b : α), (x : α) => by simp [some, Embedding.some, insertNone, aux]
                                      -- This used to be in the above `simp` before
                                      -- https://github.com/leanprover/lean4/pull/2644
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝² : PartialOrder α
                                        inst✝¹ : OrderTop α
                                        inst✝ : LocallyFiniteOrder α
                                        a✝ b✝ x✝ : WithTop α
                                        a b x : α
                                        ⊢ Iff (Exists fun a_1 => And (And (LT.lt a a_1) (LE.le a_1 b)) (Eq ({ toFun := …
                                      -/
                                      erw [aux]
                                      /-
                                        🎉 no goals
                                      -/
  finset_mem_Ioo a b x :=
    match a, b, x with
    | ⊤, _, _ => iff_of_false (not_mem_empty _) fun h => not_top_lt <| h.1.trans h.2
                          /-
                            α : Type u_1
                            β : Type u_2
                            inst✝² : PartialOrder α
                            inst✝¹ : OrderTop α
                            inst✝ : LocallyFiniteOrder α
                            a✝ b x : WithTop α
                            a : α
                            ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                          -/
    | (a : α), ⊤, ⊤ => by simp [some, Embedding.some, insertNone]
                          /-
                            🎉 no goals
                          -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b x✝ : WithTop α
                                  a x : α
                                  ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                -/
    | (a : α), ⊤, (x : α) => by simp [some, Embedding.some, insertNone, aux, top]
                                -- This used to be in the above `simp` before
                                -- https://github.com/leanprover/lean4/pull/2644
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b x✝ : WithTop α
                                  a x : α
                                  ⊢ Iff (Exists fun a_1 => And (LT.lt a a_1) (Eq ({ toFun := Option.some, inj' : …
                                -/
                                erw [aux]
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝² : PartialOrder α
                                  inst✝¹ : OrderTop α
                                  inst✝ : LocallyFiniteOrder α
                                  a✝ b✝ x : WithTop α
                                  a b : α
                                  ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
                                -/
    | (a : α), (b : α), ⊤ => by simp [some, Embedding.some, insertNone]
                                /-
                                  🎉 no goals
                                -/
    | (a : α), (b : α), (x : α) => by
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : OrderTop α
        inst✝ : LocallyFiniteOrder α
        a✝ b✝ x✝ : WithTop α
        a b x : α
        ⊢ Iff (Membership.mem ((fun a b => WithTop.locallyFiniteOrder.match_2 α (fun a …
      -/
      simp [some, Embedding.some, insertNone, aux]
      -- This used to be in the above `simp` before
      -- https://github.com/leanprover/lean4/pull/2644
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : OrderTop α
        inst✝ : LocallyFiniteOrder α
        a✝ b✝ x✝ : WithTop α
        a b x : α
        ⊢ Iff (Exists fun a_1 => And (And (LT.lt a a_1) (LT.lt a_1 b)) (Eq ({ toFun := …
      -/
      erw [aux]
      /-
        🎉 no goals
      -/


theorem Icc_coe_top : Icc (a : WithTop α) ⊤ = insertNone (Ici a) :=
  rfl


theorem Icc_coe_coe : Icc (a : WithTop α) b = (Icc a b).map Embedding.some :=
  rfl


theorem Ico_coe_top : Ico (a : WithTop α) ⊤ = (Ici a).map Embedding.some :=
  rfl


theorem Ico_coe_coe : Ico (a : WithTop α) b = (Ico a b).map Embedding.some :=
  rfl


theorem Ioc_coe_top : Ioc (a : WithTop α) ⊤ = insertNone (Ioi a) :=
  rfl


theorem Ioc_coe_coe : Ioc (a : WithTop α) b = (Ioc a b).map Embedding.some :=
  rfl


theorem Ioo_coe_top : Ioo (a : WithTop α) ⊤ = (Ioi a).map Embedding.some :=
  rfl


theorem Ioo_coe_coe : Ioo (a : WithTop α) b = (Ioo a b).map Embedding.some :=
  rfl


instance instLocallyFiniteOrder : LocallyFiniteOrder (WithBot α) :=
  OrderDual.instLocallyFiniteOrder (α := WithTop αᵒᵈ)


theorem Icc_bot_coe : Icc (⊥ : WithBot α) b = insertNone (Iic b) :=
  rfl


theorem Icc_coe_coe : Icc (a : WithBot α) b = (Icc a b).map Embedding.some :=
  rfl


theorem Ico_bot_coe : Ico (⊥ : WithBot α) b = insertNone (Iio b) :=
  rfl


theorem Ico_coe_coe : Ico (a : WithBot α) b = (Ico a b).map Embedding.some :=
  rfl


theorem Ioc_bot_coe : Ioc (⊥ : WithBot α) b = (Iic b).map Embedding.some :=
  rfl


theorem Ioc_coe_coe : Ioc (a : WithBot α) b = (Ioc a b).map Embedding.some :=
  rfl


theorem Ioo_bot_coe : Ioo (⊥ : WithBot α) b = (Iio b).map Embedding.some :=
  rfl


theorem Ioo_coe_coe : Ioo (a : WithBot α) b = (Ioo a b).map Embedding.some :=
  rfl


/-- Transfer `LocallyFiniteOrder` across an `OrderIso`. -/
abbrev locallyFiniteOrder [LocallyFiniteOrder β] (f : α ≃o β) : LocallyFiniteOrder α where
  finsetIcc a b := (Icc (f a) (f b)).map f.symm.toEquiv.toEmbedding
  finsetIco a b := (Ico (f a) (f b)).map f.symm.toEquiv.toEmbedding
  finsetIoc a b := (Ioc (f a) (f b)).map f.symm.toEquiv.toEmbedding
  finsetIoo a b := (Ioo (f a) (f b)).map f.symm.toEquiv.toEmbedding
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrder β
                         f : OrderIso α β
                         ⊢ ∀ (a b x : α), Iff (Membership.mem ((fun a b => Finset.map f.symm.toEmbeddin …
                       -/
  finset_mem_Icc := by simp
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrder β
                         f : OrderIso α β
                         ⊢ ∀ (a b x : α), Iff (Membership.mem ((fun a b => Finset.map f.symm.toEmbeddin …
                       -/
  finset_mem_Ico := by simp
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrder β
                         f : OrderIso α β
                         ⊢ ∀ (a b x : α), Iff (Membership.mem ((fun a b => Finset.map f.symm.toEmbeddin …
                       -/
  finset_mem_Ioc := by simp
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrder β
                         f : OrderIso α β
                         ⊢ ∀ (a b x : α), Iff (Membership.mem ((fun a b => Finset.map f.symm.toEmbeddin …
                       -/
  finset_mem_Ioo := by simp
                       /-
                         🎉 no goals
                       -/

-- See note [reducible non-instances]

/-- Transfer `LocallyFiniteOrderTop` across an `OrderIso`. -/
abbrev locallyFiniteOrderTop [LocallyFiniteOrderTop β] (f : α ≃o β) : LocallyFiniteOrderTop α where
  finsetIci a := (Ici (f a)).map f.symm.toEquiv.toEmbedding
  finsetIoi a := (Ioi (f a)).map f.symm.toEquiv.toEmbedding
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrderTop β
                         f : OrderIso α β
                         ⊢ ∀ (a x : α), Iff (Membership.mem ((fun a => Finset.map f.symm.toEmbedding (F …
                       -/
  finset_mem_Ici := by simp
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrderTop β
                         f : OrderIso α β
                         ⊢ ∀ (a x : α), Iff (Membership.mem ((fun a => Finset.map f.symm.toEmbedding (F …
                       -/
  finset_mem_Ioi := by simp
                       /-
                         🎉 no goals
                       -/

-- See note [reducible non-instances]

/-- Transfer `LocallyFiniteOrderBot` across an `OrderIso`. -/
abbrev locallyFiniteOrderBot [LocallyFiniteOrderBot β] (f : α ≃o β) : LocallyFiniteOrderBot α where
  finsetIic a := (Iic (f a)).map f.symm.toEquiv.toEmbedding
  finsetIio a := (Iio (f a)).map f.symm.toEquiv.toEmbedding
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrderBot β
                         f : OrderIso α β
                         ⊢ ∀ (a x : α), Iff (Membership.mem ((fun a => Finset.map f.symm.toEmbedding (F …
                       -/
  finset_mem_Iic := by simp
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : Preorder α
                         inst✝¹ : Preorder β
                         inst✝ : LocallyFiniteOrderBot β
                         f : OrderIso α β
                         ⊢ ∀ (a x : α), Iff (Membership.mem ((fun a => Finset.map f.symm.toEmbedding (F …
                       -/
  finset_mem_Iio := by simp
                       /-
                         🎉 no goals
                       -/


instance Subtype.instLocallyFiniteOrder [LocallyFiniteOrder α] :
    LocallyFiniteOrder (Subtype p) where
  finsetIcc a b := (Icc (a : α) b).subtype p
  finsetIco a b := (Ico (a : α) b).subtype p
  finsetIoc a b := (Ioc (a : α) b).subtype p
  finsetIoo a b := (Ioo (a : α) b).subtype p
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : Preorder α
                               p : α → Prop
                               inst✝¹ : DecidablePred p
                               inst✝ : LocallyFiniteOrder α
                               a b x : Subtype p
                               ⊢ Iff (Membership.mem ((fun a b => Finset.subtype p (Finset.Icc ↑a ↑b)) a b) x …
                             -/
  finset_mem_Icc a b x := by simp_rw [Finset.mem_subtype, mem_Icc, Subtype.coe_le_coe]
                             /-
                               🎉 no goals
                             -/
  finset_mem_Ico a b x := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : LocallyFiniteOrder α
      a b x : Subtype p
      ⊢ Iff (Membership.mem ((fun a b => Finset.subtype p (Finset.Ico ↑a ↑b)) a b) x …
    -/
    simp_rw [Finset.mem_subtype, mem_Ico, Subtype.coe_le_coe, Subtype.coe_lt_coe]
    /-
      🎉 no goals
    -/
  finset_mem_Ioc a b x := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : LocallyFiniteOrder α
      a b x : Subtype p
      ⊢ Iff (Membership.mem ((fun a b => Finset.subtype p (Finset.Ioc ↑a ↑b)) a b) x …
    -/
    simp_rw [Finset.mem_subtype, mem_Ioc, Subtype.coe_le_coe, Subtype.coe_lt_coe]
    /-
      🎉 no goals
    -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : Preorder α
                               p : α → Prop
                               inst✝¹ : DecidablePred p
                               inst✝ : LocallyFiniteOrder α
                               a b x : Subtype p
                               ⊢ Iff (Membership.mem ((fun a b => Finset.subtype p (Finset.Ioo ↑a ↑b)) a b) x …
                             -/
  finset_mem_Ioo a b x := by simp_rw [Finset.mem_subtype, mem_Ioo, Subtype.coe_lt_coe]
                             /-
                               🎉 no goals
                             -/


instance Subtype.instLocallyFiniteOrderTop [LocallyFiniteOrderTop α] :
    LocallyFiniteOrderTop (Subtype p) where
  finsetIci a := (Ici (a : α)).subtype p
  finsetIoi a := (Ioi (a : α)).subtype p
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             p : α → Prop
                             inst✝¹ : DecidablePred p
                             inst✝ : LocallyFiniteOrderTop α
                             a x : Subtype p
                             ⊢ Iff (Membership.mem ((fun a => Finset.subtype p (Finset.Ici ↑a)) a) x) (LE.l …
                           -/
  finset_mem_Ici a x := by simp_rw [Finset.mem_subtype, mem_Ici, Subtype.coe_le_coe]
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             p : α → Prop
                             inst✝¹ : DecidablePred p
                             inst✝ : LocallyFiniteOrderTop α
                             a x : Subtype p
                             ⊢ Iff (Membership.mem ((fun a => Finset.subtype p (Finset.Ioi ↑a)) a) x) (LT.l …
                           -/
  finset_mem_Ioi a x := by simp_rw [Finset.mem_subtype, mem_Ioi, Subtype.coe_lt_coe]
                           /-
                             🎉 no goals
                           -/


instance Subtype.instLocallyFiniteOrderBot [LocallyFiniteOrderBot α] :
    LocallyFiniteOrderBot (Subtype p) where
  finsetIic a := (Iic (a : α)).subtype p
  finsetIio a := (Iio (a : α)).subtype p
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             p : α → Prop
                             inst✝¹ : DecidablePred p
                             inst✝ : LocallyFiniteOrderBot α
                             a x : Subtype p
                             ⊢ Iff (Membership.mem ((fun a => Finset.subtype p (Finset.Iic ↑a)) a) x) (LE.l …
                           -/
  finset_mem_Iic a x := by simp_rw [Finset.mem_subtype, mem_Iic, Subtype.coe_le_coe]
                           /-
                             🎉 no goals
                           -/
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝² : Preorder α
                             p : α → Prop
                             inst✝¹ : DecidablePred p
                             inst✝ : LocallyFiniteOrderBot α
                             a x : Subtype p
                             ⊢ Iff (Membership.mem ((fun a => Finset.subtype p (Finset.Iio ↑a)) a) x) (LT.l …
                           -/
  finset_mem_Iio a x := by simp_rw [Finset.mem_subtype, mem_Iio, Subtype.coe_lt_coe]
                           /-
                             🎉 no goals
                           -/


theorem subtype_Icc_eq : Icc a b = (Icc (a : α) b).subtype p :=
  rfl


theorem subtype_Ico_eq : Ico a b = (Ico (a : α) b).subtype p :=
  rfl


theorem subtype_Ioc_eq : Ioc a b = (Ioc (a : α) b).subtype p :=
  rfl


theorem subtype_Ioo_eq : Ioo a b = (Ioo (a : α) b).subtype p :=
  rfl


theorem map_subtype_embedding_Icc (hp : ∀ ⦃a b x⦄, a ≤ x → x ≤ b → p a → p b → p x):
    (Icc a b).map (Embedding.subtype p) = (Icc a b : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Icc a b)) (Finset.Icc  …
  -/
  rw [subtype_Icc_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Icc  …
  -/
  refine Finset.subtype_map_of_mem fun x hx => ?_
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : Membership.mem (Finset.Icc ↑a ↑b) x
    ⊢ p x
  -/
  rw [mem_Icc] at hx
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : And (LE.le (↑a) x) (LE.le x ↑b)
    ⊢ p x
  -/
  exact hp hx.1 hx.2 a.prop b.prop
  /-
    🎉 no goals
  -/


theorem map_subtype_embedding_Ico (hp : ∀ ⦃a b x⦄, a ≤ x → x ≤ b → p a → p b → p x):
    (Ico a b).map (Embedding.subtype p) = (Ico a b : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Ico a b)) (Finset.Ico  …
  -/
  rw [subtype_Ico_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Ico  …
  -/
  refine Finset.subtype_map_of_mem fun x hx => ?_
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : Membership.mem (Finset.Ico ↑a ↑b) x
    ⊢ p x
  -/
  rw [mem_Ico] at hx
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : And (LE.le (↑a) x) (LT.lt x ↑b)
    ⊢ p x
  -/
  exact hp hx.1 hx.2.le a.prop b.prop
  /-
    🎉 no goals
  -/


theorem map_subtype_embedding_Ioc (hp : ∀ ⦃a b x⦄, a ≤ x → x ≤ b → p a → p b → p x):
    (Ioc a b).map (Embedding.subtype p) = (Ioc a b : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Ioc a b)) (Finset.Ioc  …
  -/
  rw [subtype_Ioc_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Ioc  …
  -/
  refine Finset.subtype_map_of_mem fun x hx => ?_
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : Membership.mem (Finset.Ioc ↑a ↑b) x
    ⊢ p x
  -/
  rw [mem_Ioc] at hx
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : And (LT.lt (↑a) x) (LE.le x ↑b)
    ⊢ p x
  -/
  exact hp hx.1.le hx.2 a.prop b.prop
  /-
    🎉 no goals
  -/


theorem map_subtype_embedding_Ioo (hp : ∀ ⦃a b x⦄, a ≤ x → x ≤ b → p a → p b → p x):
    (Ioo a b).map (Embedding.subtype p) = (Ioo a b : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Ioo a b)) (Finset.Ioo  …
  -/
  rw [subtype_Ioo_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Ioo  …
  -/
  refine Finset.subtype_map_of_mem fun x hx => ?_
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : Membership.mem (Finset.Ioo ↑a ↑b) x
    ⊢ p x
  -/
  rw [mem_Ioo] at hx
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrder α
    a b : Subtype p
    hp : ∀ ⦃a b x : α⦄, LE.le a x → LE.le x b → p a → p b → p x
    x : α
    hx : And (LT.lt (↑a) x) (LT.lt x ↑b)
    ⊢ p x
  -/
  exact hp hx.1.le hx.2.le a.prop b.prop
  /-
    🎉 no goals
  -/


theorem subtype_Ici_eq : Ici a = (Ici (a : α)).subtype p :=
  rfl


theorem subtype_Ioi_eq : Ioi a = (Ioi (a : α)).subtype p :=
  rfl


theorem map_subtype_embedding_Ici (hp : ∀ ⦃a x⦄, a ≤ x → p a → p x) :
    (Ici a).map (Embedding.subtype p) = (Ici a : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderTop α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le a x → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Ici a)) (Finset.Ici ↑a)
  -/
  rw [subtype_Ici_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderTop α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le a x → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Ici  …
  -/
  exact Finset.subtype_map_of_mem fun x hx => hp (mem_Ici.1 hx) a.prop
  /-
    🎉 no goals
  -/


theorem map_subtype_embedding_Ioi (hp : ∀ ⦃a x⦄, a ≤ x → p a → p x) :
    (Ioi a).map (Embedding.subtype p) = (Ioi a : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderTop α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le a x → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Ioi a)) (Finset.Ioi ↑a)
  -/
  rw [subtype_Ioi_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderTop α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le a x → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Ioi  …
  -/
  exact Finset.subtype_map_of_mem fun x hx => hp (mem_Ioi.1 hx).le a.prop
  /-
    🎉 no goals
  -/


theorem subtype_Iic_eq : Iic a = (Iic (a : α)).subtype p :=
  rfl


theorem subtype_Iio_eq : Iio a = (Iio (a : α)).subtype p :=
  rfl



theorem map_subtype_embedding_Iic (hp : ∀ ⦃a x⦄, x ≤ a → p a → p x) :
    (Iic a).map (Embedding.subtype p) = (Iic a : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderBot α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le x a → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Iic a)) (Finset.Iic ↑a)
  -/
  rw [subtype_Iic_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderBot α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le x a → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Iic  …
  -/
  exact Finset.subtype_map_of_mem fun x hx => hp (mem_Iic.1 hx) a.prop
  /-
    🎉 no goals
  -/


theorem map_subtype_embedding_Iio (hp : ∀ ⦃a x⦄, x ≤ a → p a → p x) :
    (Iio a).map (Embedding.subtype p) = (Iio a : Finset α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderBot α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le x a → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.Iio a)) (Finset.Iio ↑a)
  -/
  rw [subtype_Iio_eq]
  /-
    α : Type u_1
    inst✝² : Preorder α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : LocallyFiniteOrderBot α
    a : Subtype p
    hp : ∀ ⦃a x : α⦄, LE.le x a → p a → p x
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p (Finset.Iio  …
  -/
  exact Finset.subtype_map_of_mem fun x hx => hp (mem_Iio.1 hx).le a.prop
  /-
    🎉 no goals
  -/


theorem BddBelow.finite_of_bddAbove [Preorder α] [LocallyFiniteOrder α]
    {s : Set α} (h₀ : BddBelow s) (h₁ : BddAbove s) :
    s.Finite :=
  let ⟨a, ha⟩ := h₀
  let ⟨b, hb⟩ := h₁
  (Set.finite_Icc a b).subset fun _x hx ↦ ⟨ha hx, hb hx⟩


theorem Set.finite_iff_bddAbove [SemilatticeSup α] [LocallyFiniteOrder α] [OrderBot α] :
    s.Finite ↔ BddAbove s :=
  ⟨fun h ↦ ⟨h.toFinset.sup id, fun _ hx ↦ Finset.le_sup (f := id) ((Finite.mem_toFinset h).mpr hx)⟩,
    fun ⟨m, hm⟩ ↦ (Set.finite_Icc ⊥ m).subset (fun _ hx ↦ ⟨bot_le, hm hx⟩)⟩


theorem Set.finite_iff_bddBelow [SemilatticeInf α] [LocallyFiniteOrder α] [OrderTop α] :
    s.Finite ↔ BddBelow s :=
  finite_iff_bddAbove (α := αᵒᵈ)


theorem Set.finite_iff_bddBelow_bddAbove [Nonempty α] [Lattice α] [LocallyFiniteOrder α] :
    s.Finite ↔ BddBelow s ∧ BddAbove s := by
  /-
    α : Type u_3
    s : Set α
    inst✝² : Nonempty α
    inst✝¹ : Lattice α
    inst✝ : LocallyFiniteOrder α
    ⊢ Iff s.Finite (And (BddBelow s) (BddAbove s))
  -/
  obtain (rfl | hs) := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_3
      inst✝² : Nonempty α
      inst✝¹ : Lattice α
      inst✝ : LocallyFiniteOrder α
      ⊢ Iff EmptyCollection.emptyCollection.Finite (And (BddBelow EmptyCollection.em …
    -/
  · simp only [Set.finite_empty, bddBelow_empty, bddAbove_empty, and_self]
    /-
      🎉 no goals
    -/
  exact ⟨fun h ↦ ⟨⟨h.toFinset.inf' ((Finite.toFinset_nonempty h).mpr hs) id,
    fun x hx ↦ Finset.inf'_le id ((Finite.mem_toFinset h).mpr hx)⟩,
    ⟨h.toFinset.sup' ((Finite.toFinset_nonempty h).mpr hs) id, fun x hx ↦ Finset.le_sup' id
    ((Finite.mem_toFinset h).mpr hx)⟩⟩,
    fun ⟨h₀, h₁⟩ ↦ BddBelow.finite_of_bddAbove h₀ h₁⟩


instance (priority := low) [Preorder α] [DecidableRel ((· : α) ≤ ·)] [LocallyFiniteOrder α] :
    LocallyFiniteOrderTop { x : α // x ≤ y } where
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       inst✝⁴ : Preorder α
                                       p : α → Prop
                                       inst✝³ : DecidablePred p
                                       y : α
                                       inst✝² : Preorder α
                                       inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                       inst✝ : LocallyFiniteOrder α
                                       a : Subtype fun x => LE.le x y
                                       ⊢ LE.le y y
                                     -/
  finsetIoi a := Finset.Ioc a ⟨y, by rfl⟩
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       inst✝⁴ : Preorder α
                                       p : α → Prop
                                       inst✝³ : DecidablePred p
                                       y : α
                                       inst✝² : Preorder α
                                       inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                       inst✝ : LocallyFiniteOrder α
                                       a : Subtype fun x => LE.le x y
                                       ⊢ LE.le y y
                                     -/
  finsetIci a := Finset.Icc a ⟨y, by rfl⟩
                                     /-
                                       🎉 no goals
                                     -/
  finset_mem_Ici a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le x y
      ⊢ Iff (Membership.mem ((fun a => Finset.Icc a ⟨y, ⋯⟩) a) b) (LE.le a b)
    -/
    simp only [Finset.mem_Icc, and_iff_left_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le x y
      ⊢ LE.le a b → LE.le b ⟨y, ⋯⟩
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/
  finset_mem_Ioi a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le x y
      ⊢ Iff (Membership.mem ((fun a => Finset.Ioc a ⟨y, ⋯⟩) a) b) (LT.lt a b)
    -/
    simp only [Finset.mem_Ioc, and_iff_left_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le x y
      ⊢ LT.lt a b → LE.le b ⟨y, ⋯⟩
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/


instance (priority := low) [Preorder α] [DecidableRel ((· : α) < ·)] [LocallyFiniteOrder α] :
    LocallyFiniteOrderTop { x : α // x < y } where
  finsetIoi a := (Finset.Ioo ↑a y).subtype _
  finsetIci a := (Finset.Ico ↑a y).subtype _
  finset_mem_Ici a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt x y
      ⊢ Iff (Membership.mem ((fun a => Finset.subtype (fun x => LT.lt x y) (Finset.I …
    -/
    simp only [Finset.mem_subtype, Finset.mem_Ico, Subtype.coe_le_coe, and_iff_left_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt x y
      ⊢ LE.le a b → LT.lt (↑b) y
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/
  finset_mem_Ioi a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt x y
      ⊢ Iff (Membership.mem ((fun a => Finset.subtype (fun x => LT.lt x y) (Finset.I …
    -/
    simp only [Finset.mem_subtype, Finset.mem_Ioo, Subtype.coe_lt_coe, and_iff_left_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt x y
      ⊢ LT.lt a b → LT.lt (↑b) y
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/


instance (priority := low) [Preorder α] [DecidableRel ((· : α) ≤ ·)] [LocallyFiniteOrder α] :
    LocallyFiniteOrderBot { x : α // y ≤ x } where
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝⁴ : Preorder α
                                     p : α → Prop
                                     inst✝³ : DecidablePred p
                                     y : α
                                     inst✝² : Preorder α
                                     inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                     inst✝ : LocallyFiniteOrder α
                                     a : Subtype fun x => LE.le y x
                                     ⊢ LE.le y y
                                   -/
  finsetIio a := Finset.Ico ⟨y, by rfl⟩ a
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝⁴ : Preorder α
                                     p : α → Prop
                                     inst✝³ : DecidablePred p
                                     y : α
                                     inst✝² : Preorder α
                                     inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
                                     inst✝ : LocallyFiniteOrder α
                                     a : Subtype fun x => LE.le y x
                                     ⊢ LE.le y y
                                   -/
  finsetIic a := Finset.Icc ⟨y, by rfl⟩ a
                                   /-
                                     🎉 no goals
                                   -/
  finset_mem_Iic a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le y x
      ⊢ Iff (Membership.mem ((fun a => Finset.Icc ⟨y, ⋯⟩ a) a) b) (LE.le b a)
    -/
    simp only [Finset.mem_Icc, and_iff_right_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le y x
      ⊢ LE.le b a → LE.le ⟨y, ⋯⟩ b
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/
  finset_mem_Iio a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le y x
      ⊢ Iff (Membership.mem ((fun a => Finset.Ico ⟨y, ⋯⟩ a) a) b) (LT.lt b a)
    -/
    simp only [Finset.mem_Ico, and_iff_right_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LE.le y x
      ⊢ LT.lt b a → LE.le ⟨y, ⋯⟩ b
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/


instance (priority := low) [Preorder α] [DecidableRel ((· : α) < ·)] [LocallyFiniteOrder α] :
    LocallyFiniteOrderBot { x : α // y < x } where
  finsetIio a := (Finset.Ioo y ↑a).subtype _
  finsetIic a := (Finset.Ioc y ↑a).subtype _
  finset_mem_Iic a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt y x
      ⊢ Iff (Membership.mem ((fun a => Finset.subtype (fun x => LT.lt y x) (Finset.I …
    -/
    simp only [Finset.mem_subtype, Finset.mem_Ioc, Subtype.coe_le_coe, and_iff_right_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt y x
      ⊢ LE.le b a → LT.lt y ↑b
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/
  finset_mem_Iio a b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt y x
      ⊢ Iff (Membership.mem ((fun a => Finset.subtype (fun x => LT.lt y x) (Finset.I …
    -/
    simp only [Finset.mem_subtype, Finset.mem_Ioo, Subtype.coe_lt_coe, and_iff_right_iff_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : Preorder α
      p : α → Prop
      inst✝³ : DecidablePred p
      y : α
      inst✝² : Preorder α
      inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
      inst✝ : LocallyFiniteOrder α
      a b : Subtype fun x => LT.lt y x
      ⊢ LT.lt b a → LT.lt y ↑b
    -/
    exact fun _ => b.property
    /-
      🎉 no goals
    -/


instance [Preorder α] [LocallyFiniteOrderBot α] : Finite { x : α // x ≤ y } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    p : α → Prop
    inst✝² : DecidablePred p
    y : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderBot α
    ⊢ Finite (Subtype fun x => LE.le x y)
  -/
  simpa only  [coe_Iic] using (Finset.Iic y).finite_toSet
  /-
    🎉 no goals
  -/


instance [Preorder α] [LocallyFiniteOrderBot α] : Finite { x : α // x < y } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    p : α → Prop
    inst✝² : DecidablePred p
    y : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderBot α
    ⊢ Finite (Subtype fun x => LT.lt x y)
  -/
  simpa only [coe_Iio] using (Finset.Iio y).finite_toSet
  /-
    🎉 no goals
  -/


instance [Preorder α] [LocallyFiniteOrderTop α] : Finite { x : α // y ≤ x } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    p : α → Prop
    inst✝² : DecidablePred p
    y : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderTop α
    ⊢ Finite (Subtype fun x => LE.le y x)
  -/
  simpa only [coe_Ici] using (Finset.Ici y).finite_toSet
  /-
    🎉 no goals
  -/


instance [Preorder α] [LocallyFiniteOrderTop α] : Finite { x : α // y < x } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    p : α → Prop
    inst✝² : DecidablePred p
    y : α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrderTop α
    ⊢ Finite (Subtype fun x => LT.lt y x)
  -/
  simpa only [coe_Ioi] using (Finset.Ioi y).finite_toSet
  /-
    🎉 no goals
  -/


@[simp] lemma toFinset_Icc (a b : α) [Fintype (Icc a b)] : (Icc a b).toFinset = Finset.Icc a b := by
  /-
    α : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : Fintype ↑(Set.Icc a b)
    ⊢ Eq (Set.Icc a b).toFinset (Finset.Icc a b)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp] lemma toFinset_Ico (a b : α) [Fintype (Ico a b)] : (Ico a b).toFinset = Finset.Ico a b := by
  /-
    α : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : Fintype ↑(Set.Ico a b)
    ⊢ Eq (Set.Ico a b).toFinset (Finset.Ico a b)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp] lemma toFinset_Ioc (a b : α) [Fintype (Ioc a b)] : (Ioc a b).toFinset = Finset.Ioc a b := by
  /-
    α : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : Fintype ↑(Set.Ioc a b)
    ⊢ Eq (Set.Ioc a b).toFinset (Finset.Ioc a b)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp] lemma toFinset_Ioo (a b : α) [Fintype (Ioo a b)] : (Ioo a b).toFinset = Finset.Ioo a b := by
  /-
    α : Type u_3
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    a b : α
    inst✝ : Fintype ↑(Set.Ioo a b)
    ⊢ Eq (Set.Ioo a b).toFinset (Finset.Ioo a b)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
                                                                                     /-
                                                                                       α : Type u_3
                                                                                       inst✝² : Preorder α
                                                                                       inst✝¹ : LocallyFiniteOrderTop α
                                                                                       a : α
                                                                                       inst✝ : Fintype ↑(Set.Ici a)
                                                                                       ⊢ Eq (Set.Ici a).toFinset (Finset.Ici a)
                                                                                     -/
lemma toFinset_Ici (a : α) [Fintype (Ici a)] : (Ici a).toFinset = Finset.Ici a := by ext; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
                                                                                     /-
                                                                                       α : Type u_3
                                                                                       inst✝² : Preorder α
                                                                                       inst✝¹ : LocallyFiniteOrderTop α
                                                                                       a : α
                                                                                       inst✝ : Fintype ↑(Set.Ioi a)
                                                                                       ⊢ Eq (Set.Ioi a).toFinset (Finset.Ioi a)
                                                                                     -/
lemma toFinset_Ioi (a : α) [Fintype (Ioi a)] : (Ioi a).toFinset = Finset.Ioi a := by ext; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
                                                                                     /-
                                                                                       α : Type u_3
                                                                                       inst✝² : Preorder α
                                                                                       inst✝¹ : LocallyFiniteOrderBot α
                                                                                       a : α
                                                                                       inst✝ : Fintype ↑(Set.Iic a)
                                                                                       ⊢ Eq (Set.Iic a).toFinset (Finset.Iic a)
                                                                                     -/
lemma toFinset_Iic (a : α) [Fintype (Iic a)] : (Iic a).toFinset = Finset.Iic a := by ext; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
                                                                                     /-
                                                                                       α : Type u_3
                                                                                       inst✝² : Preorder α
                                                                                       inst✝¹ : LocallyFiniteOrderBot α
                                                                                       a : α
                                                                                       inst✝ : Fintype ↑(Set.Iio a)
                                                                                       ⊢ Eq (Set.Iio a).toFinset (Finset.Iio a)
                                                                                     -/
lemma toFinset_Iio (a : α) [Fintype (Iio a)] : (Iio a).toFinset = Finset.Iio a := by ext; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- A `LocallyFiniteOrder` can be transferred across an order isomorphism. -/
-- See note [reducible non instances]
abbrev LocallyFiniteOrder.ofOrderIsoClass {F M N : Type*} [Preorder M] [Preorder N]
    [EquivLike F M N] [OrderIsoClass F M N] (f : F) [LocallyFiniteOrder N] :
    LocallyFiniteOrder M where
  finsetIcc x y := (finsetIcc (f x) (f y)).map ⟨EquivLike.inv f, (EquivLike.right_inv f).injective⟩
  finsetIco x y := (finsetIco (f x) (f y)).map ⟨EquivLike.inv f, (EquivLike.right_inv f).injective⟩
  finsetIoc x y := (finsetIoc (f x) (f y)).map ⟨EquivLike.inv f, (EquivLike.right_inv f).injective⟩
  finsetIoo x y := (finsetIoo (f x) (f y)).map ⟨EquivLike.inv f, (EquivLike.right_inv f).injective⟩
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝⁶ : Preorder α
                         p : α → Prop
                         inst✝⁵ : DecidablePred p
                         y : α
                         F : Type u_3
                         M : Type u_4
                         N : Type u_5
                         inst✝⁴ : Preorder M
                         inst✝³ : Preorder N
                         inst✝² : EquivLike F M N
                         inst✝¹ : OrderIsoClass F M N
                         f : F
                         inst✝ : LocallyFiniteOrder N
                         ⊢ ∀ (a b x : M), Iff (Membership.mem ((fun x y => Finset.map { toFun := EquivL …
                       -/
  finset_mem_Icc := by simp [finset_mem_Icc, EquivLike.inv_apply_eq_iff_eq_apply]
                       /-
                         🎉 no goals
                       -/
  finset_mem_Ico := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁶ : Preorder α
      p : α → Prop
      inst✝⁵ : DecidablePred p
      y : α
      F : Type u_3
      M : Type u_4
      N : Type u_5
      inst✝⁴ : Preorder M
      inst✝³ : Preorder N
      inst✝² : EquivLike F M N
      inst✝¹ : OrderIsoClass F M N
      f : F
      inst✝ : LocallyFiniteOrder N
      ⊢ ∀ (a b x : M), Iff (Membership.mem ((fun x y => Finset.map { toFun := EquivL …
    -/
    simp [finset_mem_Ico, EquivLike.inv_apply_eq_iff_eq_apply, map_lt_map_iff]
    /-
      🎉 no goals
    -/
  finset_mem_Ioc := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁶ : Preorder α
      p : α → Prop
      inst✝⁵ : DecidablePred p
      y : α
      F : Type u_3
      M : Type u_4
      N : Type u_5
      inst✝⁴ : Preorder M
      inst✝³ : Preorder N
      inst✝² : EquivLike F M N
      inst✝¹ : OrderIsoClass F M N
      f : F
      inst✝ : LocallyFiniteOrder N
      ⊢ ∀ (a b x : M), Iff (Membership.mem ((fun x y => Finset.map { toFun := EquivL …
    -/
    simp [finset_mem_Ioc, EquivLike.inv_apply_eq_iff_eq_apply, map_lt_map_iff]
    /-
      🎉 no goals
    -/
  finset_mem_Ioo := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁶ : Preorder α
      p : α → Prop
      inst✝⁵ : DecidablePred p
      y : α
      F : Type u_3
      M : Type u_4
      N : Type u_5
      inst✝⁴ : Preorder M
      inst✝³ : Preorder N
      inst✝² : EquivLike F M N
      inst✝¹ : OrderIsoClass F M N
      f : F
      inst✝ : LocallyFiniteOrder N
      ⊢ ∀ (a b x : M), Iff (Membership.mem ((fun x y => Finset.map { toFun := EquivL …
    -/
    simp [finset_mem_Ioo, EquivLike.inv_apply_eq_iff_eq_apply, map_lt_map_iff]
    /-
      🎉 no goals
    -/

