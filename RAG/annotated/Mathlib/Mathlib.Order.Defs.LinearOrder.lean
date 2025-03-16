/-- Default definition of `max`. -/
def maxDefault [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] (a b : α) :=
  if a ≤ b then b else a


/-- Default definition of `min`. -/
def minDefault [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] (a b : α) :=
  if a ≤ b then a else b


/-- This attempts to prove that a given instance of `compare` is equal to `compareOfLessAndEq` by
introducing the arguments and trying the following approaches in order:

1. seeing if `rfl` works
2. seeing if the `compare` at hand is nonetheless essentially `compareOfLessAndEq`, but, because of
implicit arguments, requires us to unfold the defs and split the `if`s in the definition of
`compareOfLessAndEq`
3. seeing if we can split by cases on the arguments, then see if the defs work themselves out
  (useful when `compare` is defined via a `match` statement, as it is for `Bool`) -/
macro "compareOfLessAndEq_rfl" : tactic =>
  `(tactic| (intros a b; first | rfl |
    (simp only [compare, compareOfLessAndEq]; split_ifs <;> rfl) |
    (induction a <;> induction b <;> simp +decide only)))


/-- A linear order is reflexive, transitive, antisymmetric and total relation `≤`.
We assume that every linear ordered type has decidable `(≤)`, `(<)`, and `(=)`. -/
class LinearOrder (α : Type*) extends PartialOrder α, Min α, Max α, Ord α where
  /-- A linear order is total. -/
  le_total (a b : α) : a ≤ b ∨ b ≤ a
  /-- In a linearly ordered type, we assume the order relations are all decidable. -/
  decidableLE : DecidableRel (· ≤ · : α → α → Prop)
  /-- In a linearly ordered type, we assume the order relations are all decidable. -/
  decidableEq : DecidableEq α := @decidableEqOfDecidableLE _ _ decidableLE
  /-- In a linearly ordered type, we assume the order relations are all decidable. -/
  decidableLT : DecidableRel (· < · : α → α → Prop) :=
    @decidableLTOfDecidableLE _ _ decidableLE
  min := fun a b => if a ≤ b then a else b
  max := fun a b => if a ≤ b then b else a
  /-- The minimum function is equivalent to the one you get from `minOfLe`. -/
  min_def : ∀ a b, min a b = if a ≤ b then a else b := by intros; rfl
  /-- The minimum function is equivalent to the one you get from `maxOfLe`. -/
  max_def : ∀ a b, max a b = if a ≤ b then b else a := by intros; rfl
  compare a b := compareOfLessAndEq a b
  /-- Comparison via `compare` is equal to the canonical comparison given decidable `<` and `=`. -/
  compare_eq_compareOfLessAndEq : ∀ a b, compare a b = compareOfLessAndEq a b := by
    compareOfLessAndEq_rfl


lemma le_total : ∀ a b : α, a ≤ b ∨ b ≤ a := LinearOrder.le_total


lemma le_of_not_ge : ¬a ≥ b → a ≤ b := (le_total b a).resolve_left

lemma le_of_not_le : ¬a ≤ b → b ≤ a := (le_total a b).resolve_left

lemma lt_of_not_ge (h : ¬a ≥ b) : a < b := lt_of_le_not_le (le_of_not_ge h) h


lemma lt_trichotomy (a b : α) : a < b ∨ a = b ∨ b < a :=
  Or.elim (le_total a b)
    (fun h : a ≤ b =>
      Or.elim (Decidable.lt_or_eq_of_le h) (fun h : a < b => Or.inl h) fun h : a = b =>
        Or.inr (Or.inl h))
    fun h : b ≤ a =>
    Or.elim (Decidable.lt_or_eq_of_le h) (fun h : b < a => Or.inr (Or.inr h)) fun h : b = a =>
      Or.inr (Or.inl h.symm)


lemma le_of_not_lt (h : ¬b < a) : a ≤ b :=
  match lt_trichotomy a b with
  | Or.inl hlt => le_of_lt hlt
  | Or.inr (Or.inl HEq) => HEq ▸ le_refl a
  | Or.inr (Or.inr hgt) => absurd hgt h


lemma le_of_not_gt : ¬a > b → a ≤ b := le_of_not_lt


lemma lt_or_le (a b : α) : a < b ∨ b ≤ a :=
  if hba : b ≤ a then Or.inr hba else Or.inl <| lt_of_not_ge hba


lemma le_or_lt (a b : α) : a ≤ b ∨ b < a := (lt_or_le b a).symm

lemma lt_or_ge : ∀ a b : α, a < b ∨ a ≥ b := lt_or_le

lemma le_or_gt : ∀ a b : α, a ≤ b ∨ a > b := le_or_lt


                                                       /-
                                                         α : Type u_1
                                                         inst✝ : LinearOrder α
                                                         a b : α
                                                         h : Ne a b
                                                         ⊢ Or (LT.lt a b) (GT.gt a b)
                                                       -/
lemma lt_or_gt_of_ne (h : a ≠ b) : a < b ∨ a > b := by simpa [h] using lt_trichotomy a b
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma ne_iff_lt_or_gt : a ≠ b ↔ a < b ∨ a > b := ⟨lt_or_gt_of_ne, (Or.elim · ne_of_lt ne_of_gt)⟩


lemma lt_iff_not_ge (x y : α) : x < y ↔ ¬x ≥ y := ⟨not_le_of_gt, lt_of_not_ge⟩


@[simp] lemma not_lt : ¬a < b ↔ b ≤ a := ⟨le_of_not_gt, not_lt_of_ge⟩

@[simp] lemma not_le : ¬a ≤ b ↔ b < a := (lt_iff_not_ge _ _).symm


instance (priority := 900) (a b : α) : Decidable (a < b) := LinearOrder.decidableLT a b

instance (priority := 900) (a b : α) : Decidable (a ≤ b) := LinearOrder.decidableLE a b

instance (priority := 900) (a b : α) : Decidable (a = b) := LinearOrder.decidableEq a b


lemma eq_or_lt_of_not_lt (h : ¬a < b) : a = b ∨ b < a :=
  if h₁ : a = b then Or.inl h₁ else Or.inr (lt_of_not_ge fun hge => h (lt_of_le_of_ne hge h₁))


/-- Perform a case-split on the ordering of `x` and `y` in a decidable linear order. -/
def ltByCases (x y : α) {P : Sort*} (h₁ : x < y → P) (h₂ : x = y → P) (h₃ : y < x → P) : P :=
  if h : x < y then h₁ h
  else if h' : y < x then h₃ h' else h₂ (le_antisymm (le_of_not_gt h') (le_of_not_gt h))


@[deprecated "No deprecation message was provided." (since := "2024-08-23")]
protected def ltGeByCases {a b : Nat} {C : Sort*} (h₁ : a < b → C) (h₂ : b ≤ a → C) : C :=
  Decidable.byCases h₁ fun h => h₂ (Or.elim (Nat.lt_or_ge a b) (fun a => absurd a h) fun a => a)


set_option linter.deprecated false in
@[deprecated ltByCases (since := "2024-08-23")]
protected def ltByCases {a b : Nat} {C : Sort*} (h₁ : a < b → C) (h₂ : a = b → C)
    (h₃ : b < a → C) : C :=
  Nat.ltGeByCases h₁ fun h₁ => Nat.ltGeByCases h₃ fun h => h₂ (Nat.le_antisymm h h₁)


theorem le_imp_le_of_lt_imp_lt {α β} [Preorder α] [LinearOrder β] {a b : α} {c d : β}
    (H : d < c → b < a) (h : a ≤ b) : c ≤ d :=
  le_of_not_lt fun h' => not_le_of_gt (H h') h


                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : LinearOrder α
                                                                   a b : α
                                                                   ⊢ Eq (Min.min a b) (ite (LE.le a b) a b)
                                                                 -/
lemma min_def (a b : α) : min a b = if a ≤ b then a else b := by rw [LinearOrder.min_def a]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/

                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : LinearOrder α
                                                                   a b : α
                                                                   ⊢ Eq (Max.max a b) (ite (LE.le a b) b a)
                                                                 -/
lemma max_def (a b : α) : max a b = if a ≤ b then b else a := by rw [LinearOrder.max_def a]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/

-- Porting note: no `min_tac` tactic in the following series of lemmas


lemma min_le_left (a b : α) : min a b ≤ a := by
  if h : a ≤ b
  then simp [min_def, if_pos h, le_refl]
  else simpa [min_def, if_neg h] using le_of_not_le h


lemma min_le_right (a b : α) : min a b ≤ b := by
  if h : a ≤ b
  then simpa [min_def, if_pos h] using h
  else simp [min_def, if_neg h, le_refl]


lemma le_min (h₁ : c ≤ a) (h₂ : c ≤ b) : c ≤ min a b := by
  if h : a ≤ b
  then simpa [min_def, if_pos h] using h₁
  else simpa [min_def, if_neg h] using h₂


lemma le_max_left (a b : α) : a ≤ max a b := by
  if h : a ≤ b
  then simpa [max_def, if_pos h] using h
  else simp [max_def, if_neg h, le_refl]


lemma le_max_right (a b : α) : b ≤ max a b := by
  if h : a ≤ b
  then simp [max_def, if_pos h, le_refl]
  else simpa [max_def, if_neg h] using le_of_not_le h


lemma max_le (h₁ : a ≤ c) (h₂ : b ≤ c) : max a b ≤ c := by
  if h : a ≤ b
  then simpa [max_def, if_pos h] using h₂
  else simpa [max_def, if_neg h] using h₁


lemma eq_min (h₁ : c ≤ a) (h₂ : c ≤ b) (h₃ : ∀ {d}, d ≤ a → d ≤ b → d ≤ c) : c = min a b :=
  le_antisymm (le_min h₁ h₂) (h₃ (min_le_left a b) (min_le_right a b))


lemma min_comm (a b : α) : min a b = min b a :=
  eq_min (min_le_right a b) (min_le_left a b) fun h₁ h₂ => le_min h₂ h₁


lemma min_assoc (a b c : α) : min (min a b) c = min a (min b c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Eq (Min.min (Min.min a b) c) (Min.min a (Min.min b c))
  -/
  apply eq_min
    /-
      case h₁
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      ⊢ LE.le (Min.min (Min.min a b) c) a
    -/
  · apply le_trans (min_le_left ..) (min_le_left ..)
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      ⊢ LE.le (Min.min (Min.min a b) c) (Min.min b c)
    -/
  · apply le_min
      /-
        case h₂.h₁
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : α
        ⊢ LE.le (Min.min (Min.min a b) c) b
      -/
    · apply le_trans (min_le_left ..) (min_le_right ..)
      /-
        🎉 no goals
      -/
      /-
        case h₂.h₂
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : α
        ⊢ LE.le (Min.min (Min.min a b) c) c
      -/
    · apply min_le_right
      /-
        🎉 no goals
      -/
    /-
      case h₃
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      ⊢ ∀ {d : α}, LE.le d a → LE.le d (Min.min b c) → LE.le d (Min.min (Min.min a b …
    -/
  · intro d h₁ h₂; apply le_min
      /-
        case h₃.h₁
        α : Type u_1
        inst✝ : LinearOrder α
        a b c d : α
        h₁ : LE.le d a
        h₂ : LE.le d (Min.min b c)
        ⊢ LE.le d (Min.min a b)
      -/
    · apply le_min h₁; apply le_trans h₂; apply min_le_left
                                          /-
                                            🎉 no goals
                                          -/
      /-
        case h₃.h₂
        α : Type u_1
        inst✝ : LinearOrder α
        a b c d : α
        h₁ : LE.le d a
        h₂ : LE.le d (Min.min b c)
        ⊢ LE.le d c
      -/
    · apply le_trans h₂; apply min_le_right
                         /-
                           🎉 no goals
                         -/


lemma min_left_comm (a b c : α) : min a (min b c) = min b (min a c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Eq (Min.min a (Min.min b c)) (Min.min b (Min.min a c))
  -/
  rw [← min_assoc, min_comm a, min_assoc]
  /-
    🎉 no goals
  -/


                                                   /-
                                                     α : Type u_1
                                                     inst✝ : LinearOrder α
                                                     a : α
                                                     ⊢ Eq (Min.min a a) a
                                                   -/
@[simp] lemma min_self (a : α) : min a a = a := by simp [min_def]
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma min_eq_left (h : a ≤ b) : min a b = a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    ⊢ Eq (Min.min a b) a
  -/
  apply Eq.symm; apply eq_min (le_refl _) h; intros; assumption
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma min_eq_right (h : b ≤ a) : min a b = b := min_comm b a ▸ min_eq_left h


lemma eq_max (h₁ : a ≤ c) (h₂ : b ≤ c) (h₃ : ∀ {d}, a ≤ d → b ≤ d → c ≤ d) :
    c = max a b :=
  le_antisymm (h₃ (le_max_left a b) (le_max_right a b)) (max_le h₁ h₂)


lemma max_comm (a b : α) : max a b = max b a :=
  eq_max (le_max_right a b) (le_max_left a b) fun h₁ h₂ => max_le h₂ h₁


lemma max_assoc (a b c : α) : max (max a b) c = max a (max b c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Eq (Max.max (Max.max a b) c) (Max.max a (Max.max b c))
  -/
  apply eq_max
    /-
      case h₁
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      ⊢ LE.le a (Max.max (Max.max a b) c)
    -/
  · apply le_trans (le_max_left a b) (le_max_left ..)
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      ⊢ LE.le (Max.max b c) (Max.max (Max.max a b) c)
    -/
  · apply max_le
      /-
        case h₂.h₁
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : α
        ⊢ LE.le b (Max.max (Max.max a b) c)
      -/
    · apply le_trans (le_max_right a b) (le_max_left ..)
      /-
        🎉 no goals
      -/
      /-
        case h₂.h₂
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : α
        ⊢ LE.le c (Max.max (Max.max a b) c)
      -/
    · apply le_max_right
      /-
        🎉 no goals
      -/
    /-
      case h₃
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      ⊢ ∀ {d : α}, LE.le a d → LE.le (Max.max b c) d → LE.le (Max.max (Max.max a b)  …
    -/
  · intro d h₁ h₂; apply max_le
      /-
        case h₃.h₁
        α : Type u_1
        inst✝ : LinearOrder α
        a b c d : α
        h₁ : LE.le a d
        h₂ : LE.le (Max.max b c) d
        ⊢ LE.le (Max.max a b) d
      -/
    · apply max_le h₁; apply le_trans (le_max_left _ _) h₂
                       /-
                         🎉 no goals
                       -/
      /-
        case h₃.h₂
        α : Type u_1
        inst✝ : LinearOrder α
        a b c d : α
        h₁ : LE.le a d
        h₂ : LE.le (Max.max b c) d
        ⊢ LE.le c d
      -/
    · apply le_trans (le_max_right _ _) h₂
      /-
        🎉 no goals
      -/


lemma max_left_comm (a b c : α) : max a (max b c) = max b (max a c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Eq (Max.max a (Max.max b c)) (Max.max b (Max.max a c))
  -/
  rw [← max_assoc, max_comm a, max_assoc]
  /-
    🎉 no goals
  -/


                                                   /-
                                                     α : Type u_1
                                                     inst✝ : LinearOrder α
                                                     a : α
                                                     ⊢ Eq (Max.max a a) a
                                                   -/
@[simp] lemma max_self (a : α) : max a a = a := by simp [max_def]
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma max_eq_left (h : b ≤ a) : max a b = a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le b a
    ⊢ Eq (Max.max a b) a
  -/
  apply Eq.symm; apply eq_max (le_refl _) h; intros; assumption
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma max_eq_right (h : a ≤ b) : max a b = b := max_comm b a ▸ max_eq_left h


lemma min_eq_left_of_lt (h : a < b) : min a b = a := min_eq_left (le_of_lt h)

lemma min_eq_right_of_lt (h : b < a) : min a b = b := min_eq_right (le_of_lt h)

lemma max_eq_left_of_lt (h : b < a) : max a b = a := max_eq_left (le_of_lt h)

lemma max_eq_right_of_lt (h : a < b) : max a b = b := max_eq_right (le_of_lt h)


lemma lt_min (h₁ : a < b) (h₂ : a < c) : a < min b c := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    h₁ : LT.lt a b
    h₂ : LT.lt a c
    ⊢ LT.lt a (Min.min b c)
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total b c <;> simp [min_eq_left, min_eq_right, *]
                         /-
                           🎉 no goals
                         -/


lemma max_lt (h₁ : a < c) (h₂ : b < c) : max a b < c := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    h₁ : LT.lt a c
    h₂ : LT.lt b c
    ⊢ LT.lt (Max.max a b) c
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total a b <;> simp [max_eq_left, max_eq_right, *]
                         /-
                           🎉 no goals
                         -/


lemma compare_lt_iff_lt : compare a b = .lt ↔ a < b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Eq (Ord.compare a b) Ordering.lt) (LT.lt a b)
  -/
  rw [LinearOrder.compare_eq_compareOfLessAndEq, compareOfLessAndEq]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Eq (ite (LT.lt a b) Ordering.lt (ite (Eq a b) Ordering.eq Ordering.gt)) …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp only [*, lt_irrefl, reduceCtorEq]
                /-
                  🎉 no goals
                -/


lemma compare_gt_iff_gt : compare a b = .gt ↔ a > b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Eq (Ord.compare a b) Ordering.gt) (GT.gt a b)
  -/
  rw [LinearOrder.compare_eq_compareOfLessAndEq, compareOfLessAndEq]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Eq (ite (LT.lt a b) Ordering.lt (ite (Eq a b) Ordering.eq Ordering.gt)) …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp only [*, lt_irrefl, not_lt_of_gt, reduceCtorEq]
  case _ h₁ h₂ =>
    have h : b < a := lt_trichotomy a b |>.resolve_left h₁ |>.resolve_left h₂
    rwa [true_iff]


lemma compare_eq_iff_eq : compare a b = .eq ↔ a = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Eq (Ord.compare a b) Ordering.eq) (Eq a b)
  -/
  rw [LinearOrder.compare_eq_compareOfLessAndEq, compareOfLessAndEq]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Eq (ite (LT.lt a b) Ordering.lt (ite (Eq a b) Ordering.eq Ordering.gt)) …
  -/
  split_ifs <;> try simp only [reduceCtorEq]
  /-
    case pos
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h✝ : LT.lt a b
    ⊢ Iff False (Eq a b)
  -/
  case _ h   => rw [false_iff]; exact ne_iff_lt_or_gt.2 <| .inl h
  /-
    case pos
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h✝¹ : Not (LT.lt a b)
    h✝ : Eq a b
    ⊢ Iff True (Eq a b)
  -/
  case _ _ h => rwa [true_iff]
  /-
    case neg
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h✝¹ : Not (LT.lt a b)
    h✝ : Not (Eq a b)
    ⊢ Iff False (Eq a b)
  -/
  case _ _ h => rwa [false_iff]
  /-
    🎉 no goals
  -/


lemma compare_le_iff_le : compare a b ≠ .gt ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Ne (Ord.compare a b) Ordering.gt) (LE.le a b)
  -/
  cases h : compare a b <;> simp
    /-
      case lt
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : Eq (Ord.compare a b) Ordering.lt
      ⊢ LE.le a b
    -/
  · exact le_of_lt <| compare_lt_iff_lt.1 h
    /-
      🎉 no goals
    -/
    /-
      case eq
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : Eq (Ord.compare a b) Ordering.eq
      ⊢ LE.le a b
    -/
  · exact le_of_eq <| compare_eq_iff_eq.1 h
    /-
      🎉 no goals
    -/
    /-
      case gt
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : Eq (Ord.compare a b) Ordering.gt
      ⊢ LT.lt b a
    -/
  · exact compare_gt_iff_gt.1 h
    /-
      🎉 no goals
    -/


lemma compare_ge_iff_ge : compare a b ≠ .lt ↔ a ≥ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Ne (Ord.compare a b) Ordering.lt) (GE.ge a b)
  -/
  cases h : compare a b <;> simp
    /-
      case lt
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : Eq (Ord.compare a b) Ordering.lt
      ⊢ LT.lt a b
    -/
  · exact compare_lt_iff_lt.1 h
    /-
      🎉 no goals
    -/
    /-
      case eq
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : Eq (Ord.compare a b) Ordering.eq
      ⊢ LE.le b a
    -/
  · exact le_of_eq <| (·.symm) <| compare_eq_iff_eq.1 h
    /-
      🎉 no goals
    -/
    /-
      case gt
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h : Eq (Ord.compare a b) Ordering.gt
      ⊢ LE.le b a
    -/
  · exact le_of_lt <| compare_gt_iff_gt.1 h
    /-
      🎉 no goals
    -/


lemma compare_iff (a b : α) {o : Ordering} : compare a b = o ↔ o.Compares a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    o : Ordering
    ⊢ Iff (Eq (Ord.compare a b) o) (o.Compares a b)
  -/
  cases o <;> simp only [Ordering.Compares]
    /-
      case lt
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      ⊢ Iff (Eq (Ord.compare a b) Ordering.lt) (LT.lt a b)
    -/
  · exact compare_lt_iff_lt
    /-
      🎉 no goals
    -/
    /-
      case eq
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      ⊢ Iff (Eq (Ord.compare a b) Ordering.eq) (Eq a b)
    -/
  · exact compare_eq_iff_eq
    /-
      🎉 no goals
    -/
    /-
      case gt
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      ⊢ Iff (Eq (Ord.compare a b) Ordering.gt) (GT.gt a b)
    -/
  · exact compare_gt_iff_gt
    /-
      🎉 no goals
    -/


theorem cmp_eq_compare (a b : α) : cmp a b = compare a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (cmp a b) (Ord.compare a b)
  -/
  refine ((compare_iff ..).2 ?_).symm
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ (cmp a b).Compares a b
  -/
  unfold cmp cmpUsing; split_ifs with h1 h2
    /-
      case pos
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h1 : (fun x1 x2 => LT.lt x1 x2) a b
      ⊢ Ordering.lt.Compares a b
    -/
  · exact h1
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h1 : Not ((fun x1 x2 => LT.lt x1 x2) a b)
      h2 : (fun x1 x2 => LT.lt x1 x2) b a
      ⊢ Ordering.gt.Compares a b
    -/
  · exact h2
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : LinearOrder α
      a b : α
      h1 : Not ((fun x1 x2 => LT.lt x1 x2) a b)
      h2 : Not ((fun x1 x2 => LT.lt x1 x2) b a)
      ⊢ Ordering.eq.Compares a b
    -/
  · exact le_antisymm (not_lt.1 h2) (not_lt.1 h1)
    /-
      🎉 no goals
    -/


theorem cmp_eq_compareOfLessAndEq (a b : α) : cmp a b = compareOfLessAndEq a b :=
  (cmp_eq_compare ..).trans (LinearOrder.compare_eq_compareOfLessAndEq ..)


instance : Batteries.LawfulCmp (compare (α := α)) where
  symm a b := by
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a✝ b✝ c a b : α
      ⊢ Eq (Ord.compare a b).swap (Ord.compare b a)
    -/
    cases h : compare a b <;>
    /-
      case lt
      α : Type u_1
      inst✝ : LinearOrder α
      a✝ b✝ c a b : α
      h : Eq (Ord.compare a b) Ordering.lt
      ⊢ Eq Ordering.lt.swap (Ord.compare b a)
    -/
    simp only [Ordering.swap] <;> symm
      /-
        case lt
        α : Type u_1
        inst✝ : LinearOrder α
        a✝ b✝ c a b : α
        h : Eq (Ord.compare a b) Ordering.lt
        ⊢ Eq (Ord.compare b a) Ordering.gt
      -/
    · exact compare_gt_iff_gt.2 <| compare_lt_iff_lt.1 h
      /-
        🎉 no goals
      -/
      /-
        case eq
        α : Type u_1
        inst✝ : LinearOrder α
        a✝ b✝ c a b : α
        h : Eq (Ord.compare a b) Ordering.eq
        ⊢ Eq (Ord.compare b a) Ordering.eq
      -/
    · exact compare_eq_iff_eq.2 <| compare_eq_iff_eq.1 h |>.symm
      /-
        🎉 no goals
      -/
      /-
        case gt
        α : Type u_1
        inst✝ : LinearOrder α
        a✝ b✝ c a b : α
        h : Eq (Ord.compare a b) Ordering.gt
        ⊢ Eq (Ord.compare b a) Ordering.lt
      -/
    · exact compare_lt_iff_lt.2 <| compare_gt_iff_gt.1 h
      /-
        🎉 no goals
      -/
  le_trans := fun h₁ h₂ ↦
    compare_le_iff_le.2 <| le_trans (compare_le_iff_le.1 h₁) (compare_le_iff_le.1 h₂)
                    /-
                      α : Type u_1
                      inst✝ : LinearOrder α
                      a b c : α
                      ⊢ ∀ {x y : α}, Iff (Eq (Ord.compare x y) Ordering.eq) (Eq (BEq.beq x y) Bool.t …
                    -/
  cmp_iff_beq := by simp [compare_eq_iff_eq]
                    /-
                      🎉 no goals
                    -/
                   /-
                     α : Type u_1
                     inst✝ : LinearOrder α
                     a b c : α
                     ⊢ ∀ {x y : α}, Iff (Eq (Ord.compare x y) Ordering.lt) (LT.lt x y)
                   -/
  cmp_iff_lt := by simp [compare_lt_iff_lt]
                   /-
                     🎉 no goals
                   -/
                   /-
                     α : Type u_1
                     inst✝ : LinearOrder α
                     a b c : α
                     ⊢ ∀ {x y : α}, Iff (Ne (Ord.compare x y) Ordering.gt) (LE.le x y)
                   -/
  cmp_iff_le := by simp [compare_le_iff_le]
                   /-
                     🎉 no goals
                   -/


