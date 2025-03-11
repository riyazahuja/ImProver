/-- If `α` has a subtraction and a bottom element, we can extend the subtraction to `WithTop α`, by
setting `x - ⊤ = ⊥` and `⊤ - x = ⊤`. -/
protected def sub : ∀ _ _ : WithTop α, WithTop α
  | _, ⊤ => (⊥ : α)
  | ⊤, (x : α) => ⊤
  | (x : α), (y : α) => (x - y : α)


instance : Sub (WithTop α) :=
  ⟨WithTop.sub⟩


@[simp, norm_cast]
theorem coe_sub {a b : α} : (↑(a - b) : WithTop α) = ↑a - ↑b :=
  rfl


@[simp]
theorem top_sub_coe {a : α} : (⊤ : WithTop α) - a = ⊤ :=
  rfl


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          inst✝¹ : Sub α
                                                          inst✝ : Bot α
                                                          a : WithTop α
                                                          ⊢ Eq (HSub.hSub a Top.top) ↑Bot.bot
                                                        -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
theorem sub_top {a : WithTop α} : a - ⊤ = (⊥ : α) := by cases a <;> rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp] theorem sub_eq_top_iff {a b : WithTop α} : a - b = ⊤ ↔ a = ⊤ ∧ b ≠ ⊤ := by
  /-
    α : Type u_1
    inst✝¹ : Sub α
    inst✝ : Bot α
    a b : WithTop α
    ⊢ Iff (Eq (HSub.hSub a b) Top.top) (And (Eq a Top.top) (Ne b Top.top))
  -/
  induction a <;> induction b <;>
    simp only [← coe_sub, coe_ne_top, sub_top, zero_ne_top, top_sub_coe, false_and, Ne,
      not_true_eq_false, not_false_eq_true, and_false, and_self]


                                                                         /-
                                                                           α : Type u_1
                                                                           inst✝¹ : Sub α
                                                                           inst✝ : Bot α
                                                                           a b : WithTop α
                                                                           ⊢ Iff (Ne (HSub.hSub a b) Top.top) (Or (Ne a Top.top) (Eq b Top.top))
                                                                         -/
lemma sub_ne_top_iff {a b : WithTop α} : a - b ≠ ⊤ ↔ a ≠ ⊤ ∨ b = ⊤ := by simp [or_iff_not_imp_left]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem map_sub [Sub β] [Bot β] {f : α → β} (h : ∀ x y, f (x - y) = f x - f y) (h₀ : f ⊥ = ⊥) :
    ∀ x y : WithTop α, (x - y).map f = x.map f - y.map f
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝³ : Sub α
                 inst✝² : Bot α
                 inst✝¹ : Sub β
                 inst✝ : Bot β
                 f : α → β
                 h : ∀ (x y : α), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                 h₀ : Eq (f Bot.bot) Bot.bot
                 x✝ : WithTop α
                 ⊢ Eq (WithTop.map f (HSub.hSub x✝ Top.top)) (HSub.hSub (WithTop.map f x✝) (Wit …
               -/
  | _, ⊤ => by simp only [sub_top, map_coe, h₀, map_top]
               /-
                 🎉 no goals
               -/
  | ⊤, (x : α) => rfl
                           /-
                             α : Type u_1
                             β : Type u_2
                             inst✝³ : Sub α
                             inst✝² : Bot α
                             inst✝¹ : Sub β
                             inst✝ : Bot β
                             f : α → β
                             h : ∀ (x y : α), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                             h₀ : Eq (f Bot.bot) Bot.bot
                             x y : α
                             ⊢ Eq (WithTop.map f (HSub.hSub ↑x ↑y)) (HSub.hSub (WithTop.map f ↑x) (WithTop. …
                           -/
  | (x : α), (y : α) => by simp only [← coe_sub, map_coe, h]
                           /-
                             🎉 no goals
                           -/


instance : OrderedSub (WithTop α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    ⊢ OrderedSub (WithTop α)
  -/
  constructor
  /-
    case tsub_le_iff_right
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    ⊢ ∀ (a b c : WithTop α), Iff (LE.le (HSub.hSub a b) c) (LE.le a (HAdd.hAdd c b))
  -/
  rintro x y z
  /-
    case tsub_le_iff_right
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y z : WithTop α
    ⊢ Iff (LE.le (HSub.hSub x y) z) (LE.le x (HAdd.hAdd z y))
  -/
  cases y
    /-
      case tsub_le_iff_right.top
      α : Type u_1
      β : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      x z : WithTop α
      ⊢ Iff (LE.le (HSub.hSub x Top.top) z) (LE.le x (HAdd.hAdd z Top.top))
    -/
                /-
                  🎉 no goals
                -/
  · cases z <;> simp
                /-
                  🎉 no goals
                -/
  /-
    case tsub_le_iff_right.coe
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x z : WithTop α
    a✝ : α
    ⊢ Iff (LE.le (HSub.hSub x ↑a✝) z) (LE.le x (HAdd.hAdd z ↑a✝))
  -/
  cases x
    /-
      case tsub_le_iff_right.coe.top
      α : Type u_1
      β : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      z : WithTop α
      a✝ : α
      ⊢ Iff (LE.le (HSub.hSub Top.top ↑a✝) z) (LE.le Top.top (HAdd.hAdd z ↑a✝))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case tsub_le_iff_right.coe.coe
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    z : WithTop α
    a✝¹ a✝ : α
    ⊢ Iff (LE.le (HSub.hSub ↑a✝ ↑a✝¹) z) (LE.le (↑a✝) (HAdd.hAdd z ↑a✝¹))
  -/
  cases z
    /-
      case tsub_le_iff_right.coe.coe.top
      α : Type u_1
      β : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      a✝¹ a✝ : α
      ⊢ Iff (LE.le (HSub.hSub ↑a✝ ↑a✝¹) Top.top) (LE.le (↑a✝) (HAdd.hAdd Top.top ↑a✝ …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case tsub_le_iff_right.coe.coe.coe
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a✝² a✝¹ a✝ : α
    ⊢ Iff (LE.le (HSub.hSub ↑a✝¹ ↑a✝²) ↑a✝) (LE.le (↑a✝¹) (HAdd.hAdd ↑a✝ ↑a✝²))
  -/
  norm_cast
  /-
    case tsub_le_iff_right.coe.coe.coe
    α : Type u_1
    β : Type u_2
    inst✝² : CanonicallyOrderedAddCommMonoid α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a✝² a✝¹ a✝ : α
    ⊢ Iff (LE.le (HSub.hSub a✝¹ a✝²) a✝) (LE.le a✝¹ (HAdd.hAdd a✝ a✝²))
  -/
  exact tsub_le_iff_right
  /-
    🎉 no goals
  -/


