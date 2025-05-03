/-- A type synonym for ordinals with nimber addition and multiplication. -/
def Nimber : Type _ :=
  Ordinal deriving Zero, Inhabited, One, Nontrivial, WellFoundedRelation


instance Nimber.instLinearOrder : LinearOrder Nimber := Ordinal.instLinearOrder

instance Nimber.instSuccOrder : SuccOrder Nimber := Ordinal.instSuccOrder

instance Nimber.instOrderBot : OrderBot Nimber := Ordinal.instOrderBot

instance Nimber.instNoMaxOrder : NoMaxOrder Nimber := Ordinal.instNoMaxOrder

instance Nimber.instZeroLEOneClass : ZeroLEOneClass Nimber := Ordinal.instZeroLEOneClass

instance Nimber.instNeZeroOne : NeZero (1 : Nimber) := Ordinal.instNeZeroOne


/-- The identity function between `Ordinal` and `Nimber`. -/
@[match_pattern]
def Ordinal.toNimber : Ordinal ≃o Nimber :=
  OrderIso.refl _


/-- The identity function between `Nimber` and `Ordinal`. -/
@[match_pattern]
def Nimber.toOrdinal : Nimber ≃o Ordinal :=
  OrderIso.refl _


@[inherit_doc]
scoped[Nimber] prefix:75 "∗" => Ordinal.toNimber


@[simp]
theorem toOrdinal_symm_eq : Nimber.toOrdinal.symm = Ordinal.toNimber :=
  rfl


@[simp]
theorem toOrdinal_toNimber (a : Nimber) : ∗(Nimber.toOrdinal a) = a :=
  rfl


theorem lt_wf : @WellFounded Nimber (· < ·) :=
  Ordinal.lt_wf


instance : WellFoundedLT Nimber :=
  Ordinal.wellFoundedLT


instance : ConditionallyCompleteLinearOrderBot Nimber :=
  WellFoundedLT.conditionallyCompleteLinearOrderBot _


@[simp]
theorem bot_eq_zero : ⊥ = 0 :=
  rfl


@[simp]
theorem toOrdinal_zero : toOrdinal 0 = 0 :=
  rfl


@[simp]
theorem toOrdinal_one : toOrdinal 1 = 1 :=
  rfl


@[simp]
theorem toOrdinal_eq_zero {a} : toOrdinal a = 0 ↔ a = 0 :=
  Iff.rfl


@[simp]
theorem toOrdinal_eq_one {a} : toOrdinal a = 1 ↔ a = 1 :=
  Iff.rfl


@[simp]
theorem toOrdinal_max (a b : Nimber) : toOrdinal (max a b) = max (toOrdinal a) (toOrdinal b) :=
  rfl


@[simp]
theorem toOrdinal_min (a b : Nimber) : toOrdinal (min a b) = min (toOrdinal a) (toOrdinal b) :=
  rfl


theorem succ_def (a : Nimber) : succ a = ∗(toOrdinal a + 1) :=
  rfl


/-- A recursor for `Nimber`. Use as `induction x`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {β : Nimber → Sort*} (h : ∀ a, β (∗a)) : ∀ a, β a := fun a =>
  h (toOrdinal a)


/-- `Ordinal.induction` but for `Nimber`. -/
theorem induction {p : Nimber → Prop} : ∀ (i) (_ : ∀ j, (∀ k, k < j → p k) → p j), p i :=
  Ordinal.induction


protected theorem le_zero {a : Nimber} : a ≤ 0 ↔ a = 0 :=
  Ordinal.le_zero


protected theorem not_lt_zero (a : Nimber) : ¬ a < 0 :=
  Ordinal.not_lt_zero a


protected theorem pos_iff_ne_zero {a : Nimber} : 0 < a ↔ a ≠ 0 :=
  Ordinal.pos_iff_ne_zero


theorem lt_one_iff_zero {a : Nimber} : a < 1 ↔ a = 0 :=
  Ordinal.lt_one_iff_zero


theorem eq_nat_of_le_nat {a : Nimber} {b : ℕ} (h : a ≤ ∗b) : ∃ c : ℕ, a = ∗c :=
  Ordinal.lt_omega0.1 (h.trans_lt (nat_lt_omega0 b))


instance small_Iio (a : Nimber.{u}) : Small.{u} (Set.Iio a) := Ordinal.small_Iio a

instance small_Iic (a : Nimber.{u}) : Small.{u} (Set.Iic a) := Ordinal.small_Iic a

instance small_Ico (a b : Nimber.{u}) : Small.{u} (Set.Ico a b) := Ordinal.small_Ico a b

instance small_Icc (a b : Nimber.{u}) : Small.{u} (Set.Icc a b) := Ordinal.small_Icc a b

instance small_Ioo (a b : Nimber.{u}) : Small.{u} (Set.Ioo a b) := Ordinal.small_Ioo a b

instance small_Ioc (a b : Nimber.{u}) : Small.{u} (Set.Ioc a b) := Ordinal.small_Ioc a b


theorem not_bddAbove_compl_of_small (s : Set Nimber.{u}) [Small.{u} s] : ¬ BddAbove sᶜ :=
  Ordinal.not_bddAbove_compl_of_small s


theorem not_small_nimber : ¬ Small.{u} Nimber.{max u v} :=
  not_small_ordinal


@[simp]
theorem toNimber_symm_eq : toNimber.symm = Nimber.toOrdinal :=
  rfl


@[simp]
theorem toNimber_toOrdinal (a : Ordinal) : Nimber.toOrdinal (∗a) = a :=
  rfl


@[simp]
theorem toNimber_zero : ∗0 = 0 :=
  rfl


@[simp]
theorem toNimber_one : ∗1 = 1 :=
  rfl


@[simp]
theorem toNimber_eq_zero {a} : ∗a = 0 ↔ a = 0 :=
  Iff.rfl


@[simp]
theorem toNimber_eq_one {a} : ∗a = 1 ↔ a = 1 :=
  Iff.rfl


@[simp]
theorem toNimber_max (a b : Ordinal) : ∗(max a b) = max (∗a) (∗b) :=
  rfl


@[simp]
theorem toNimber_min (a b : Ordinal) : ∗(min a b) = min (∗a) (∗b) :=
  rfl


/-- Nimber addition is recursively defined so that `a + b` is the smallest nimber not equal to
`a' + b` or `a + b'` for `a' < a` and `b' < b`. -/
-- We write the binders like this so that the termination checker works.
protected def add (a b : Nimber.{u}) : Nimber.{u} :=
  sInf {x | (∃ a', ∃ (_ : a' < a), Nimber.add a' b = x) ∨
    ∃ b', ∃ (_ : b' < b), Nimber.add a b' = x}ᶜ
termination_by (a, b)


instance : Add Nimber :=
  ⟨Nimber.add⟩


theorem add_def (a b : Nimber) :
    a + b = sInf {x | (∃ a' < a, a' + b = x) ∨ ∃ b' < b, a + b' = x}ᶜ := by
  /-
    a b : Nimber
    ⊢ Eq (HAdd.hAdd a b) (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists f …
  -/
  change Nimber.add a b = _
  /-
    a b : Nimber
    ⊢ Eq (a.add b) (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a'  …
  -/
  rw [Nimber.add]
  /-
    a b : Nimber
    ⊢ Eq (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a' => Exists  …
  -/
  simp_rw [exists_prop]
  /-
    a b : Nimber
    ⊢ Eq (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a' => And (LT …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The set in the definition of `Nimber.add` is nonempty. -/
private theorem add_nonempty (a b : Nimber.{u}) :
    {x | (∃ a' < a, a' + b = x) ∨ ∃ b' < b, a + b' = x}ᶜ.Nonempty :=
  nonempty_of_not_bddAbove <| not_bddAbove_compl_of_small
    ((· + b) '' Set.Iio a ∪ (a + ·) '' Set.Iio b)


theorem exists_of_lt_add (h : c < a + b) : (∃ a' < a, a' + b = c) ∨ ∃ b' < b, a + b' = c := by
  /-
    a b c : Nimber
    h : LT.lt c (HAdd.hAdd a b)
    ⊢ Or (Exists fun a' => And (LT.lt a' a) (Eq (HAdd.hAdd a' b) c)) (Exists fun b …
  -/
  rw [add_def] at h
  /-
    a b c : Nimber
    h : LT.lt c (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a' =>  …
    ⊢ Or (Exists fun a' => And (LT.lt a' a) (Eq (HAdd.hAdd a' b) c)) (Exists fun b …
  -/
  have := not_mem_of_lt_csInf' h
  /-
    a b c : Nimber
    h : LT.lt c (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a' =>  …
    this : Not (Membership.mem (HasCompl.compl (setOf fun x => Or (Exists fun a' = …
    ⊢ Or (Exists fun a' => And (LT.lt a' a) (Eq (HAdd.hAdd a' b) c)) (Exists fun b …
  -/
  rwa [Set.mem_compl_iff, not_not] at this
  /-
    🎉 no goals
  -/


theorem add_le_of_forall_ne (h₁ : ∀ a' < a, a' + b ≠ c) (h₂ : ∀ b' < b, a + b' ≠ c) :
    a + b ≤ c := by
  /-
    a b c : Nimber
    h₁ : ∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) c
    h₂ : ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd a b') c
    ⊢ LE.le (HAdd.hAdd a b) c
  -/
  by_contra! h
  /-
    a b c : Nimber
    h₁ : ∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) c
    h₂ : ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd a b') c
    h : LT.lt c (HAdd.hAdd a b)
    ⊢ False
  -/
  have := exists_of_lt_add h
  /-
    a b c : Nimber
    h₁ : ∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) c
    h₂ : ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd a b') c
    h : LT.lt c (HAdd.hAdd a b)
    this : Or (Exists fun a' => And (LT.lt a' a) (Eq (HAdd.hAdd a' b) c)) (Exists  …
    ⊢ False
  -/
  tauto
  /-
    🎉 no goals
  -/


private theorem add_ne_of_lt (a b : Nimber) :
    (∀ a' < a, a' + b ≠ a + b) ∧ ∀ b' < b, a + b' ≠ a + b := by
  /-
    a b : Nimber
    ⊢ And (∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) (HAdd.hAdd a b)) (∀ ( …
  -/
  have H := csInf_mem (add_nonempty a b)
  /-
    a b : Nimber
    H : Membership.mem (HasCompl.compl (setOf fun x => Or (Exists fun a' => And (L …
    ⊢ And (∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) (HAdd.hAdd a b)) (∀ ( …
  -/
  rw [← add_def] at H
  /-
    a b : Nimber
    H : Membership.mem (HasCompl.compl (setOf fun x => Or (Exists fun a' => And (L …
    ⊢ And (∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) (HAdd.hAdd a b)) (∀ ( …
  -/
  simpa using H
  /-
    🎉 no goals
  -/


instance : IsLeftCancelAdd Nimber := by
  /-
    a b c : Nimber
    ⊢ IsLeftCancelAdd Nimber
  -/
  constructor
  /-
    case add_left_cancel
    a b c : Nimber
    ⊢ ∀ (a b c : Nimber), Eq (HAdd.hAdd a b) (HAdd.hAdd a c) → Eq b c
  -/
  intro a b c h
  /-
    case add_left_cancel
    a✝ b✝ c✝ : Nimber
    a b c : Nimber
    h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
    ⊢ Eq b c
  -/
  apply le_antisymm <;>
  /-
    case add_left_cancel.a
    a✝ b✝ c✝ : Nimber
    a b c : Nimber
    h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
    ⊢ LE.le b c
  -/
  apply le_of_not_lt
    /-
      case add_left_cancel.a.h
      a✝ b✝ c✝ : Nimber
      a b c : Nimber
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
      ⊢ Not (LT.lt c b)
    -/
  · exact fun hc => (add_ne_of_lt a b).2 c hc h.symm
    /-
      🎉 no goals
    -/
    /-
      case add_left_cancel.a.h
      a✝ b✝ c✝ : Nimber
      a b c : Nimber
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
      ⊢ Not (LT.lt b c)
    -/
  · exact fun hb => (add_ne_of_lt a c).2 b hb h
    /-
      🎉 no goals
    -/


instance : IsRightCancelAdd Nimber := by
  /-
    a b c : Nimber
    ⊢ IsRightCancelAdd Nimber
  -/
  constructor
  /-
    case add_right_cancel
    a b c : Nimber
    ⊢ ∀ (a b c : Nimber), Eq (HAdd.hAdd a b) (HAdd.hAdd c b) → Eq a c
  -/
  intro a b c h
  /-
    case add_right_cancel
    a✝ b✝ c✝ : Nimber
    a b c : Nimber
    h : Eq (HAdd.hAdd a b) (HAdd.hAdd c b)
    ⊢ Eq a c
  -/
  apply le_antisymm <;>
  /-
    case add_right_cancel.a
    a✝ b✝ c✝ : Nimber
    a b c : Nimber
    h : Eq (HAdd.hAdd a b) (HAdd.hAdd c b)
    ⊢ LE.le a c
  -/
  apply le_of_not_lt
    /-
      case add_right_cancel.a.h
      a✝ b✝ c✝ : Nimber
      a b c : Nimber
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd c b)
      ⊢ Not (LT.lt c a)
    -/
  · exact fun hc => (add_ne_of_lt a b).1 c hc h.symm
    /-
      🎉 no goals
    -/
    /-
      case add_right_cancel.a.h
      a✝ b✝ c✝ : Nimber
      a b c : Nimber
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd c b)
      ⊢ Not (LT.lt a c)
    -/
  · exact fun ha => (add_ne_of_lt c b).1 a ha h
    /-
      🎉 no goals
    -/


protected theorem add_comm (a b : Nimber) : a + b = b + a := by
  /-
    a b : Nimber
    ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
  -/
  rw [add_def, add_def]
  /-
    a b : Nimber
    ⊢ Eq (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a' => And (LT …
  -/
  simp_rw [or_comm]
  /-
    a b : Nimber
    ⊢ Eq (InfSet.sInf (HasCompl.compl (setOf fun x => Or (Exists fun a' => And (LT …
  -/
  congr! 7 <;>
     /-
       case h.e'_3.h.e'_3.h.e'_2.h.h.e'_1.h.e'_2.h.a
       a b x✝¹ x✝ : Nimber
       ⊢ Iff (And (LT.lt x✝ a) (Eq (HAdd.hAdd x✝ b) x✝¹)) (And (LT.lt x✝ a) (Eq (HAdd …
     -/
                                      /-
                                        🎉 no goals
                                      -/
    (rw [and_congr_right_iff]; intro; rw [Nimber.add_comm])
                                      /-
                                        🎉 no goals
                                      -/
termination_by (a, b)


theorem add_eq_zero {a b : Nimber} : a + b = 0 ↔ a = b := by
  /-
    a b : Nimber
    ⊢ Iff (Eq (HAdd.hAdd a b) 0) (Eq a b)
  -/
  constructor <;>
    /-
      case mp
      a b : Nimber
      ⊢ Eq (HAdd.hAdd a b) 0 → Eq a b
    -/
    intro hab
    /-
      case mp
      a b : Nimber
      hab : Eq (HAdd.hAdd a b) 0
      ⊢ Eq a b
    -/
  · obtain h | rfl | h := lt_trichotomy a b
      /-
        case mp.inl
        a b : Nimber
        hab : Eq (HAdd.hAdd a b) 0
        h : LT.lt a b
        ⊢ Eq a b
      -/
    · have ha : a + a = 0 := add_eq_zero.2 rfl
      /-
        case mp.inl
        a b : Nimber
        hab : Eq (HAdd.hAdd a b) 0
        h : LT.lt a b
        ha : Eq (HAdd.hAdd a a) 0
        ⊢ Eq a b
      -/
      rwa [← ha, add_right_inj, eq_comm] at hab
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inl
        a : Nimber
        hab : Eq (HAdd.hAdd a a) 0
        ⊢ Eq a a
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inr
        a b : Nimber
        hab : Eq (HAdd.hAdd a b) 0
        h : LT.lt b a
        ⊢ Eq a b
      -/
    · have hb : b + b = 0 := add_eq_zero.2 rfl
      /-
        case mp.inr.inr
        a b : Nimber
        hab : Eq (HAdd.hAdd a b) 0
        h : LT.lt b a
        hb : Eq (HAdd.hAdd b b) 0
        ⊢ Eq a b
      -/
      rwa [← hb, add_left_inj] at hab
      /-
        🎉 no goals
      -/
    /-
      case mpr
      a b : Nimber
      hab : Eq a b
      ⊢ Eq (HAdd.hAdd a b) 0
    -/
  · rw [← Nimber.le_zero]
    /-
      case mpr
      a b : Nimber
      hab : Eq a b
      ⊢ LE.le (HAdd.hAdd a b) 0
    -/
    apply add_le_of_forall_ne <;>
    /-
      case mpr.h₁
      a b : Nimber
      hab : Eq a b
      ⊢ ∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' b) 0
    -/
    simp_rw [ne_eq] <;>
    /-
      case mpr.h₁
      a b : Nimber
      hab : Eq a b
      ⊢ ∀ (a' : Nimber), LT.lt a' a → Not (Eq (HAdd.hAdd a' b) 0)
    -/
    intro x hx
      /-
        case mpr.h₁
        a b : Nimber
        hab : Eq a b
        x : Nimber
        hx : LT.lt x a
        ⊢ Not (Eq (HAdd.hAdd x b) 0)
      -/
    · rw [add_eq_zero, ← hab]
      /-
        case mpr.h₁
        a b : Nimber
        hab : Eq a b
        x : Nimber
        hx : LT.lt x a
        ⊢ Not (Eq x a)
      -/
      exact hx.ne
      /-
        🎉 no goals
      -/
      /-
        case mpr.h₂
        a b : Nimber
        hab : Eq a b
        x : Nimber
        hx : LT.lt x b
        ⊢ Not (Eq (HAdd.hAdd a x) 0)
      -/
    · rw [add_eq_zero, hab]
      /-
        case mpr.h₂
        a b : Nimber
        hab : Eq a b
        x : Nimber
        hx : LT.lt x b
        ⊢ Not (Eq b x)
      -/
      exact hx.ne'
      /-
        🎉 no goals
      -/
termination_by (a, b)


theorem add_ne_zero_iff : a + b ≠ 0 ↔ a ≠ b :=
  add_eq_zero.not


@[simp]
theorem add_self (a : Nimber) : a + a = 0 :=
  add_eq_zero.2 rfl


protected theorem add_assoc (a b c : Nimber) : a + b + c = a + (b + c) := by
  /-
    a b c : Nimber
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd b c))
  -/
  apply le_antisymm <;>
    /-
      case a
      a b c : Nimber
      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd b c))
    -/
    apply add_le_of_forall_ne <;>
    /-
      case a.h₁
      a b c : Nimber
      ⊢ ∀ (a' : Nimber), LT.lt a' (HAdd.hAdd a b) → Ne (HAdd.hAdd a' c) (HAdd.hAdd a …
    -/
    intro x hx <;>
    /-
      case a.h₁
      a b c x : Nimber
      hx : LT.lt x (HAdd.hAdd a b)
      ⊢ Ne (HAdd.hAdd x c) (HAdd.hAdd a (HAdd.hAdd b c))
    -/
    try obtain ⟨y, hy, rfl⟩ | ⟨y, hy, rfl⟩ := exists_of_lt_add hx
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne (HAdd.hAdd (HAdd.hAdd y b) c) (HAdd.hAdd a (HAdd.hAdd b c))
  -/
  on_goal 1 => rw [Nimber.add_assoc y, add_ne_add_left]
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne y a
  -/
  on_goal 2 => rw [Nimber.add_assoc _ y, add_ne_add_right, add_ne_add_left]
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne y a
  -/
  on_goal 3 => rw [Nimber.add_assoc _ _ x, add_ne_add_right, add_ne_add_right]
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne y a
  -/
  on_goal 4 => rw [← Nimber.add_assoc x, add_ne_add_left, add_ne_add_left]
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne y a
  -/
  on_goal 5 => rw [← Nimber.add_assoc _ y, add_ne_add_left, add_ne_add_right]
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne y a
  -/
  on_goal 6 => rw [← Nimber.add_assoc _ _ y, add_ne_add_right]
  /-
    case a.h₁.inl.intro.intro
    a b c y : Nimber
    hy : LT.lt y a
    hx : LT.lt (HAdd.hAdd y b) (HAdd.hAdd a b)
    ⊢ Ne y a
  -/
  all_goals apply ne_of_lt; assumption
  /-
    🎉 no goals
  -/
termination_by (a, b, c)


protected theorem add_zero (a : Nimber) : a + 0 = a := by
  /-
    a : Nimber
    ⊢ Eq (HAdd.hAdd a 0) a
  -/
  apply le_antisymm
    /-
      case a
      a : Nimber
      ⊢ LE.le (HAdd.hAdd a 0) a
    -/
  · apply add_le_of_forall_ne
      /-
        case a.h₁
        a : Nimber
        ⊢ ∀ (a' : Nimber), LT.lt a' a → Ne (HAdd.hAdd a' 0) a
      -/
    · intro a' ha
      /-
        case a.h₁
        a a' : Nimber
        ha : LT.lt a' a
        ⊢ Ne (HAdd.hAdd a' 0) a
      -/
      rw [Nimber.add_zero]
      /-
        case a.h₁
        a a' : Nimber
        ha : LT.lt a' a
        ⊢ Ne a' a
      -/
      exact ha.ne
      /-
        🎉 no goals
      -/
      /-
        case a.h₂
        a : Nimber
        ⊢ ∀ (b' : Nimber), LT.lt b' 0 → Ne (HAdd.hAdd a b') a
      -/
    · intro _ h
      /-
        case a.h₂
        a b'✝ : Nimber
        h : LT.lt b'✝ 0
        ⊢ Ne (HAdd.hAdd a b'✝) a
      -/
      exact (Nimber.not_lt_zero _ h).elim
      /-
        🎉 no goals
      -/
    /-
      case a
      a : Nimber
      ⊢ LE.le a (HAdd.hAdd a 0)
    -/
  · by_contra! h
    /-
      case a
      a : Nimber
      h : LT.lt (HAdd.hAdd a 0) a
      ⊢ False
    -/
    replace h := h -- needed to remind `termination_by`
    /-
      case a
      a : Nimber
      h : LT.lt (HAdd.hAdd a 0) a
      ⊢ False
    -/
    have := Nimber.add_zero (a + 0)
    /-
      case a
      a : Nimber
      h : LT.lt (HAdd.hAdd a 0) a
      this : Eq (HAdd.hAdd (HAdd.hAdd a 0) 0) (HAdd.hAdd a 0)
      ⊢ False
    -/
    rw [add_left_inj] at this
    /-
      case a
      a : Nimber
      h : LT.lt (HAdd.hAdd a 0) a
      this : Eq (HAdd.hAdd a 0) a
      ⊢ False
    -/
    exact this.not_lt h
    /-
      🎉 no goals
    -/
termination_by a


protected theorem zero_add (a : Nimber) : 0 + a = a := by
  /-
    a : Nimber
    ⊢ Eq (HAdd.hAdd 0 a) a
  -/
  rw [Nimber.add_comm, Nimber.add_zero]
  /-
    🎉 no goals
  -/


instance : Neg Nimber :=
  ⟨id⟩


@[simp]
protected theorem neg_eq (a : Nimber) : -a = a :=
  rfl


instance : AddCommGroupWithOne Nimber where
  add_assoc := Nimber.add_assoc
  add_zero := Nimber.add_zero
  zero_add := Nimber.zero_add
  nsmul := nsmulRec
  zsmul := zsmulRec
  neg_add_cancel := add_self
  add_comm := Nimber.add_comm


@[simp]
theorem add_cancel_right (a b : Nimber) : a + b + b = a := by
  /-
    a b : Nimber
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd a b) b) a
  -/
  rw [add_assoc, add_self, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_cancel_left (a b : Nimber) : a + (a + b) = b := by
  /-
    a b : Nimber
    ⊢ Eq (HAdd.hAdd a (HAdd.hAdd a b)) b
  -/
  rw [← add_assoc, add_self, zero_add]
  /-
    🎉 no goals
  -/


theorem add_trichotomy {a b c : Nimber} (h : a + b + c ≠ 0) :
    b + c < a ∨ c + a < b ∨ a + b < c := by
  /-
    a b c : Nimber
    h : Ne (HAdd.hAdd (HAdd.hAdd a b) c) 0
    ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
  -/
  rw [← Nimber.pos_iff_ne_zero] at h
  /-
    a b c : Nimber
    h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
    ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
  -/
  obtain ⟨x, hx, hx'⟩ | ⟨x, hx, hx'⟩ := exists_of_lt_add h <;>
  /-
    case inl.intro.intro
    a b c : Nimber
    h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
    x : Nimber
    hx : LT.lt x (HAdd.hAdd a b)
    hx' : Eq (HAdd.hAdd x c) 0
    ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
  -/
  rw [add_eq_zero] at hx'
    /-
      case inl.intro.intro
      a b c : Nimber
      h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
      x : Nimber
      hx : LT.lt x (HAdd.hAdd a b)
      hx' : Eq x c
      ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
    -/
  · obtain ⟨x, hx, hx'⟩ | ⟨x, hx, hx'⟩ := exists_of_lt_add (hx' ▸ hx)
      /-
        case inl.intro.intro.inl.intro.intro
        a b c : Nimber
        h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
        x✝ : Nimber
        hx✝ : LT.lt x✝ (HAdd.hAdd a b)
        hx'✝ : Eq x✝ c
        x : Nimber
        hx : LT.lt x a
        hx' : Eq (HAdd.hAdd x b) c
        ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
      -/
    · rw [← hx', add_comm, add_cancel_right]
      /-
        case inl.intro.intro.inl.intro.intro
        a b c : Nimber
        h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
        x✝ : Nimber
        hx✝ : LT.lt x✝ (HAdd.hAdd a b)
        hx'✝ : Eq x✝ c
        x : Nimber
        hx : LT.lt x a
        hx' : Eq (HAdd.hAdd x b) c
        ⊢ Or (LT.lt x a) (Or (LT.lt (HAdd.hAdd (HAdd.hAdd x b) a) b) (LT.lt (HAdd.hAdd …
      -/
      exact Or.inl hx
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.intro.inr.intro.intro
        a b c : Nimber
        h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
        x✝ : Nimber
        hx✝ : LT.lt x✝ (HAdd.hAdd a b)
        hx'✝ : Eq x✝ c
        x : Nimber
        hx : LT.lt x b
        hx' : Eq (HAdd.hAdd a x) c
        ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
      -/
    · rw [← hx', add_comm a, add_cancel_right]
      /-
        case inl.intro.intro.inr.intro.intro
        a b c : Nimber
        h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
        x✝ : Nimber
        hx✝ : LT.lt x✝ (HAdd.hAdd a b)
        hx'✝ : Eq x✝ c
        x : Nimber
        hx : LT.lt x b
        hx' : Eq (HAdd.hAdd a x) c
        ⊢ Or (LT.lt (HAdd.hAdd b (HAdd.hAdd x a)) a) (Or (LT.lt x b) (LT.lt (HAdd.hAdd …
      -/
      exact Or.inr <| Or.inl hx
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.intro
      a b c : Nimber
      h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
      x : Nimber
      hx : LT.lt x c
      hx' : Eq (HAdd.hAdd a b) x
      ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
    -/
  · rw [← hx'] at hx
    /-
      case inr.intro.intro
      a b c : Nimber
      h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd a b) c)
      x : Nimber
      hx : LT.lt (HAdd.hAdd a b) c
      hx' : Eq (HAdd.hAdd a b) x
      ⊢ Or (LT.lt (HAdd.hAdd b c) a) (Or (LT.lt (HAdd.hAdd c a) b) (LT.lt (HAdd.hAdd …
    -/
    exact Or.inr <| Or.inr hx
    /-
      🎉 no goals
    -/


theorem lt_add_cases {a b c : Nimber} (h : a < b + c) : a + c < b ∨ a + b < c := by
  /-
    a b c : Nimber
    h : LT.lt a (HAdd.hAdd b c)
    ⊢ Or (LT.lt (HAdd.hAdd a c) b) (LT.lt (HAdd.hAdd a b) c)
  -/
  obtain ha | hb | hc := add_trichotomy <| add_assoc a b c ▸ add_ne_zero_iff.2 h.ne
  /-
    case inl
    a b c : Nimber
    h : LT.lt a (HAdd.hAdd b c)
    ha : LT.lt (HAdd.hAdd b c) a
    ⊢ Or (LT.lt (HAdd.hAdd a c) b) (LT.lt (HAdd.hAdd a b) c)
  -/
  exacts [(h.asymm ha).elim, Or.inl <| add_comm c a ▸ hb, Or.inr hc]
  /-
    🎉 no goals
  -/


/-- Nimber addition of naturals corresponds to the bitwise XOR operation. -/
theorem add_nat (a b : ℕ) : ∗a + ∗b = ∗(a ^^^ b) := by
  /-
    a b : Nat
    ⊢ Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNimber …
  -/
  apply le_antisymm
    /-
      case a
      a b : Nat
      ⊢ LE.le (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNim …
    -/
  · apply add_le_of_forall_ne
    all_goals
      intro c hc
      obtain ⟨c, rfl⟩ := eq_nat_of_le_nat hc.le
      rw [OrderIso.lt_iff_lt] at hc
      replace hc := Nat.cast_lt.1 hc
      rw [add_nat]
      simpa using hc.ne
    /-
      case a
      a b : Nat
      ⊢ LE.le (Ordinal.toNimber ↑(HXor.hXor a b)) (HAdd.hAdd (Ordinal.toNimber ↑a) ( …
    -/
  · apply le_of_not_lt
    /-
      case a.h
      a b : Nat
      ⊢ Not (LT.lt (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal. …
    -/
    intro hc
    /-
      case a.h
      a b : Nat
      hc : LT.lt (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.to …
      ⊢ False
    -/
    obtain ⟨c, hc'⟩ := eq_nat_of_le_nat hc.le
    /-
      case a.h.intro
      a b : Nat
      hc : LT.lt (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.to …
      c : Nat
      hc' : Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNi …
      ⊢ False
    -/
    rw [hc', OrderIso.lt_iff_lt, Nat.cast_lt] at hc
    /-
      case a.h.intro
      a b c : Nat
      hc : LT.lt c (HXor.hXor a b)
      hc' : Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNi …
      ⊢ False
    -/
    obtain h | h := Nat.lt_xor_cases hc
      /-
        case a.h.intro.inl
        a b c : Nat
        hc : LT.lt c (HXor.hXor a b)
        hc' : Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNi …
        h : LT.lt (HXor.hXor c b) a
        ⊢ False
      -/
    · apply h.ne
      /-
        case a.h.intro.inl
        a b c : Nat
        hc : LT.lt c (HXor.hXor a b)
        hc' : Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNi …
        h : LT.lt (HXor.hXor c b) a
        ⊢ Eq (HXor.hXor c b) a
      -/
      simpa [Nat.xor_comm, Nat.xor_cancel_left, ← hc'] using add_nat (c ^^^ b) b
      /-
        🎉 no goals
      -/
      /-
        case a.h.intro.inr
        a b c : Nat
        hc : LT.lt c (HXor.hXor a b)
        hc' : Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNi …
        h : LT.lt (HXor.hXor c a) b
        ⊢ False
      -/
    · apply h.ne
      /-
        case a.h.intro.inr
        a b c : Nat
        hc : LT.lt c (HXor.hXor a b)
        hc' : Eq (HAdd.hAdd (Ordinal.toNimber ↑a) (Ordinal.toNimber ↑b)) (Ordinal.toNi …
        h : LT.lt (HXor.hXor c a) b
        ⊢ Eq (HXor.hXor c a) b
      -/
      simpa [Nat.xor_comm, Nat.xor_cancel_left, ← hc'] using add_nat a (c ^^^ a)
      /-
        🎉 no goals
      -/


