/-- A linearly ordered commutative monoid with an additively absorbing `⊤` element.
  Instances should include number systems with an infinite element adjoined. -/
class LinearOrderedAddCommMonoidWithTop (α : Type*) extends LinearOrderedAddCommMonoid α,
    OrderTop α where
  /-- In a `LinearOrderedAddCommMonoidWithTop`, the `⊤` element is invariant under addition. -/
  protected top_add' : ∀ x : α, ⊤ + x = ⊤


/-- A linearly ordered commutative group with an additively absorbing `⊤` element.
  Instances should include number systems with an infinite element adjoined. -/
class LinearOrderedAddCommGroupWithTop (α : Type*) extends LinearOrderedAddCommMonoidWithTop α,
  SubNegMonoid α, Nontrivial α where
  protected neg_top : -(⊤ : α) = ⊤
  protected add_neg_cancel : ∀ a : α, a ≠ ⊤ → a + -a = 0


instance WithTop.linearOrderedAddCommMonoidWithTop [LinearOrderedAddCommMonoid α] :
    LinearOrderedAddCommMonoidWithTop (WithTop α) :=
  { WithTop.orderTop, WithTop.linearOrder, WithTop.orderedAddCommMonoid with
    top_add' := WithTop.top_add }


@[simp]
theorem top_add (a : α) : ⊤ + a = ⊤ :=
  LinearOrderedAddCommMonoidWithTop.top_add' a


@[simp]
theorem add_top (a : α) : a + ⊤ = ⊤ :=
  Trans.trans (add_comm _ _) (top_add _)


instance instNeg : Neg (WithTop α) where neg := Option.map fun a : α => -a


/-- If `α` has subtraction, we can extend the subtraction to `WithTop α`, by
setting `x - ⊤ = ⊤` and `⊤ - x = ⊤`. This definition is only registered as an instance on linearly
ordered additive commutative groups, to avoid conflicting with the instance `WithTop.instSub` on
types with a bottom element. -/
protected def sub : ∀ _ _ : WithTop α, WithTop α
  | _, ⊤ => ⊤
  | ⊤, (x : α) => ⊤
  | (x : α), (y : α) => (x - y : α)


instance instSub : Sub (WithTop α) where sub := WithTop.LinearOrderedAddCommGroup.sub


@[simp, norm_cast]
theorem coe_neg (a : α) : ((-a : α) : WithTop α) = -a :=
  rfl


@[simp]
theorem neg_top : -(⊤ : WithTop α) = ⊤ := rfl


@[simp, norm_cast]
theorem coe_sub {a b : α} : (↑(a - b) : WithTop α) = ↑a - ↑b := rfl


@[simp]
theorem top_sub {a : WithTop α} : (⊤ : WithTop α) - a = ⊤ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a : WithTop α
    ⊢ Eq (HSub.hSub Top.top a) Top.top
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
                                                  /-
                                                    α : Type u_1
                                                    inst✝ : LinearOrderedAddCommGroup α
                                                    a : WithTop α
                                                    ⊢ Eq (HSub.hSub a Top.top) Top.top
                                                  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
theorem sub_top {a : WithTop α} : a - ⊤ = ⊤ := by cases a <;> rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
lemma sub_eq_top_iff {a b : WithTop α} : a - b = ⊤ ↔ (a = ⊤ ∨ b = ⊤) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : WithTop α
    ⊢ Iff (Eq (HSub.hSub a b) Top.top) (Or (Eq a Top.top) (Eq b Top.top))
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
  cases a <;> cases b <;> simp [← coe_sub]
                          /-
                            🎉 no goals
                          -/


instance : LinearOrderedAddCommGroupWithTop (WithTop α) where
  __ := WithTop.linearOrderedAddCommMonoidWithTop
  __ := Option.nontrivial
  sub_eq_add_neg a b := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b : WithTop α
      ⊢ Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
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
    cases a <;> cases b <;> simp [← coe_sub, ← coe_neg, sub_eq_add_neg]
                            /-
                              🎉 no goals
                            -/
  neg_top := Option.map_none
  zsmul := zsmulRec
  add_neg_cancel := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      ⊢ ∀ (a : WithTop α), Ne a Top.top → Eq (HAdd.hAdd a (Neg.neg a)) 0
    -/
    rintro (a | a) ha
      /-
        case none
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        ha : Ne Option.none Top.top
        ⊢ Eq (HAdd.hAdd Option.none (Neg.neg Option.none)) 0
      -/
    · exact (ha rfl).elim
      /-
        🎉 no goals
      -/
      /-
        case some
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        a : α
        ha : Ne (Option.some a) Top.top
        ⊢ Eq (HAdd.hAdd (Option.some a) (Neg.neg (Option.some a))) 0
      -/
    · exact (WithTop.coe_add ..).symm.trans (WithTop.coe_eq_coe.2 (add_neg_cancel a))
      /-
        🎉 no goals
      -/


lemma add_neg_cancel_of_ne_top {α : Type*} [LinearOrderedAddCommGroupWithTop α]
    {a : α} (h : a ≠ ⊤) :
    a + -a = 0 :=
  LinearOrderedAddCommGroupWithTop.add_neg_cancel a h


@[simp]
lemma add_eq_top : a + b = ⊤ ↔ a = ⊤ ∨ b = ⊤ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    a b : α
    ⊢ Iff (Eq (HAdd.hAdd a b) Top.top) (Or (Eq a Top.top) (Eq b Top.top))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      ⊢ Eq (HAdd.hAdd a b) Top.top → Or (Eq a Top.top) (Eq b Top.top)
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h : Eq (HAdd.hAdd a b) Top.top
      ⊢ Or (Eq a Top.top) (Eq b Top.top)
    -/
    by_contra nh
    /-
      case mp
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h : Eq (HAdd.hAdd a b) Top.top
      nh : Not (Or (Eq a Top.top) (Eq b Top.top))
      ⊢ False
    -/
    rw [not_or] at nh
    /-
      case mp
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h : Eq (HAdd.hAdd a b) Top.top
      nh : And (Not (Eq a Top.top)) (Not (Eq b Top.top))
      ⊢ False
    -/
    replace h := congrArg (-a + ·) h
    /-
      case mp
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      nh : And (Not (Eq a Top.top)) (Not (Eq b Top.top))
      h : Eq ((fun x => HAdd.hAdd (Neg.neg a) x) (HAdd.hAdd a b)) ((fun x => HAdd.hA …
      ⊢ False
    -/
    dsimp only at h
    rw [add_top, ← add_assoc, add_comm (-a), add_neg_cancel_of_ne_top,
      zero_add] at h
      /-
        case mp
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        nh : And (Not (Eq a Top.top)) (Not (Eq b Top.top))
        h : Eq b Top.top
        ⊢ False
      -/
    · exact nh.2 h
      /-
        🎉 no goals
      -/
      /-
        case mp
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        nh : And (Not (Eq a Top.top)) (Not (Eq b Top.top))
        h : Eq (HAdd.hAdd (HAdd.hAdd a (Neg.neg a)) b) Top.top
        ⊢ Ne a Top.top
      -/
    · exact nh.1
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      ⊢ Or (Eq a Top.top) (Eq b Top.top) → Eq (HAdd.hAdd a b) Top.top
    -/
  · rintro (rfl | rfl)
      /-
        case mpr.inl
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        b : α
        ⊢ Eq (HAdd.hAdd Top.top b) Top.top
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a : α
        ⊢ Eq (HAdd.hAdd a Top.top) Top.top
      -/
    · simp
      /-
        🎉 no goals
      -/


@[simp]
lemma top_ne_zero :
    (⊤ : α) ≠ 0 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    ⊢ Ne Top.top 0
  -/
  intro nh
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    nh : Eq Top.top 0
    ⊢ False
  -/
  have ⟨a, b, h⟩ := Nontrivial.exists_pair_ne (α := α)
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    nh : Eq Top.top 0
    a b : α
    h : Ne a b
    ⊢ False
  -/
  have : a + 0 ≠ b + 0 := by simpa
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    nh : Eq Top.top 0
    a b : α
    h : Ne a b
    this : Ne (HAdd.hAdd a 0) (HAdd.hAdd b 0)
    ⊢ False
  -/
  rw [← nh] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    nh : Eq Top.top 0
    a b : α
    h : Ne a b
    this : Ne (HAdd.hAdd a Top.top) (HAdd.hAdd b Top.top)
    ⊢ False
  -/
  simp at this
  /-
    🎉 no goals
  -/


@[simp] lemma neg_eq_top {a : α} : -a = ⊤ ↔ a = ⊤ where
  mp h := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a : α
      h : Eq (Neg.neg a) Top.top
      ⊢ Eq a Top.top
    -/
    by_contra nh
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a : α
      h : Eq (Neg.neg a) Top.top
      nh : Not (Eq a Top.top)
      ⊢ False
    -/
    replace nh := add_neg_cancel_of_ne_top nh
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a : α
      h : Eq (Neg.neg a) Top.top
      nh : Eq (HAdd.hAdd a (Neg.neg a)) 0
      ⊢ False
    -/
    rw [h, add_top] at nh
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a : α
      h : Eq (Neg.neg a) Top.top
      nh : Eq Top.top 0
      ⊢ False
    -/
    exact top_ne_zero nh
    /-
      🎉 no goals
    -/
              /-
                α : Type u_1
                inst✝ : LinearOrderedAddCommGroupWithTop α
                a : α
                h : Eq a Top.top
                ⊢ Eq (Neg.neg a) Top.top
              -/
  mpr h := by simp [h]
              /-
                🎉 no goals
              -/


instance (priority := 100) toSubtractionMonoid : SubtractionMonoid α where
  neg_neg a := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b a : α
      ⊢ Eq (Neg.neg (Neg.neg a)) a
    -/
    by_cases h : a = ⊤
      /-
        case pos
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b a : α
        h : Eq a Top.top
        ⊢ Eq (Neg.neg (Neg.neg a)) a
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b a : α
        h : Not (Eq a Top.top)
        ⊢ Eq (Neg.neg (Neg.neg a)) a
      -/
    · have h2 : ¬ -a = ⊤ := fun nh ↦ h <| neg_eq_top.mp nh
      /-
        case neg
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b a : α
        h : Not (Eq a Top.top)
        h2 : Not (Eq (Neg.neg a) Top.top)
        ⊢ Eq (Neg.neg (Neg.neg a)) a
      -/
      replace h2 : a + (-a + - -a) = a + 0 := congrArg (a + ·) (add_neg_cancel_of_ne_top h2)
      /-
        case neg
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b a : α
        h : Not (Eq a Top.top)
        h2 : Eq (HAdd.hAdd a (HAdd.hAdd (Neg.neg a) (Neg.neg (Neg.neg a)))) (HAdd.hAdd …
        ⊢ Eq (Neg.neg (Neg.neg a)) a
      -/
      rw [← add_assoc, add_neg_cancel_of_ne_top h] at h2
      /-
        case neg
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b a : α
        h : Not (Eq a Top.top)
        h2 : Eq (HAdd.hAdd 0 (Neg.neg (Neg.neg a))) (HAdd.hAdd a 0)
        ⊢ Eq (Neg.neg (Neg.neg a)) a
      -/
      simp only [zero_add, add_zero] at h2
      /-
        case neg
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b a : α
        h : Not (Eq a Top.top)
        h2 : Eq (Neg.neg (Neg.neg a)) a
        ⊢ Eq (Neg.neg (Neg.neg a)) a
      -/
      exact h2
      /-
        🎉 no goals
      -/
  neg_add_rev a b := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      ⊢ Eq (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg b) (Neg.neg a))
    -/
    by_cases ha : a = ⊤
      /-
        case pos
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Eq a Top.top
        ⊢ Eq (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg b) (Neg.neg a))
      -/
    · simp [ha]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      ha : Not (Eq a Top.top)
      ⊢ Eq (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg b) (Neg.neg a))
    -/
    by_cases hb : b = ⊤
      /-
        case pos
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Eq b Top.top
        ⊢ Eq (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg b) (Neg.neg a))
      -/
    · simp [hb]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      ha : Not (Eq a Top.top)
      hb : Not (Eq b Top.top)
      ⊢ Eq (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg b) (Neg.neg a))
    -/
    apply (_ : Function.Injective (a + b + ·))
      /-
        case neg.a
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        ⊢ Eq ((fun x => HAdd.hAdd (HAdd.hAdd a b) x) (Neg.neg (HAdd.hAdd a b))) ((fun  …
      -/
    · dsimp
      rw [add_neg_cancel_of_ne_top, ← add_assoc, add_assoc a,
        add_neg_cancel_of_ne_top hb, add_zero,
        add_neg_cancel_of_ne_top ha]
      /-
        case neg.a
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        ⊢ Ne (HAdd.hAdd a b) Top.top
      -/
      simp [ha, hb]
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        ⊢ Function.Injective fun x => HAdd.hAdd (HAdd.hAdd a b) x
      -/
    · apply Function.LeftInverse.injective (g := (-(a + b) + ·))
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        ⊢ Function.LeftInverse (fun x => HAdd.hAdd (Neg.neg (HAdd.hAdd a b)) x) fun x  …
      -/
      intro x
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        x : α
        ⊢ Eq ((fun x => HAdd.hAdd (Neg.neg (HAdd.hAdd a b)) x) ((fun x => HAdd.hAdd (H …
      -/
      dsimp only
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        x : α
        ⊢ Eq (HAdd.hAdd (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (HAdd.hAdd a b) x)) x
      -/
      rw [← add_assoc, add_comm (-(a + b)), add_neg_cancel_of_ne_top, zero_add]
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        ha : Not (Eq a Top.top)
        hb : Not (Eq b Top.top)
        x : α
        ⊢ Ne (HAdd.hAdd a b) Top.top
      -/
      simp [ha, hb]
      /-
        🎉 no goals
      -/
  neg_eq_of_add a b h := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      h : Eq (HAdd.hAdd a b) 0
      ⊢ Eq (Neg.neg a) b
    -/
    have oh := congrArg (-a + ·) h
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      h : Eq (HAdd.hAdd a b) 0
      oh : Eq ((fun x => HAdd.hAdd (Neg.neg a) x) (HAdd.hAdd a b)) ((fun x => HAdd.h …
      ⊢ Eq (Neg.neg a) b
    -/
    dsimp only at oh
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      h : Eq (HAdd.hAdd a b) 0
      oh : Eq (HAdd.hAdd (Neg.neg a) (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg a) 0)
      ⊢ Eq (Neg.neg a) b
    -/
    rw [add_zero, ← add_assoc, add_comm (-a), add_neg_cancel_of_ne_top, zero_add] at oh
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a✝ b✝ a b : α
        h : Eq (HAdd.hAdd a b) 0
        oh : Eq b (Neg.neg a)
        ⊢ Eq (Neg.neg a) b
      -/
    · exact oh.symm
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      h : Eq (HAdd.hAdd a b) 0
      oh : Eq (HAdd.hAdd (HAdd.hAdd a (Neg.neg a)) b) (Neg.neg a)
      ⊢ Ne a Top.top
    -/
    intro v
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a✝ b✝ a b : α
      h : Eq (HAdd.hAdd a b) 0
      oh : Eq (HAdd.hAdd (HAdd.hAdd a (Neg.neg a)) b) (Neg.neg a)
      v : Eq a Top.top
      ⊢ False
    -/
    simp [v] at h
    /-
      🎉 no goals
    -/


lemma injective_add_left_of_ne_top (b : α) (h : b ≠ ⊤) : Function.Injective (fun x ↦ x + b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    b : α
    h : Ne b Top.top
    ⊢ Function.Injective fun x => HAdd.hAdd x b
  -/
  intro x y h2
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    b : α
    h : Ne b Top.top
    x y : α
    h2 : Eq ((fun x => HAdd.hAdd x b) x) ((fun x => HAdd.hAdd x b) y)
    ⊢ Eq x y
  -/
  replace h2 : x + (b + -b) = y + (b + -b) := by simp [← add_assoc, h2]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    b : α
    h : Ne b Top.top
    x y : α
    h2 : Eq (HAdd.hAdd x (HAdd.hAdd b (Neg.neg b))) (HAdd.hAdd y (HAdd.hAdd b (Neg …
    ⊢ Eq x y
  -/
  simpa only [LinearOrderedAddCommGroupWithTop.add_neg_cancel _ h, add_zero] using h2
  /-
    🎉 no goals
  -/


lemma injective_add_right_of_ne_top (b : α) (h : b ≠ ⊤) : Function.Injective (fun x ↦ b + x) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    b : α
    h : Ne b Top.top
    ⊢ Function.Injective fun x => HAdd.hAdd b x
  -/
  simpa [add_comm] using injective_add_left_of_ne_top b h
  /-
    🎉 no goals
  -/


lemma strictMono_add_left_of_ne_top (b : α) (h : b ≠ ⊤) : StrictMono (fun x ↦ x + b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    b : α
    h : Ne b Top.top
    ⊢ StrictMono fun x => HAdd.hAdd x b
  -/
  apply Monotone.strictMono_of_injective
    /-
      case h₁
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      b : α
      h : Ne b Top.top
      ⊢ Monotone fun x => HAdd.hAdd x b
    -/
  · apply Monotone.add_const monotone_id
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      b : α
      h : Ne b Top.top
      ⊢ Function.Injective fun x => HAdd.hAdd x b
    -/
  · apply injective_add_left_of_ne_top _ h
    /-
      🎉 no goals
    -/


lemma strictMono_add_right_of_ne_top (b : α) (h : b ≠ ⊤) : StrictMono (fun x ↦ b + x) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroupWithTop α
    b : α
    h : Ne b Top.top
    ⊢ StrictMono fun x => HAdd.hAdd b x
  -/
  simpa [add_comm] using strictMono_add_left_of_ne_top b h
  /-
    🎉 no goals
  -/


lemma sub_pos (a b : α) : 0 < a - b ↔ b < a ∨ b = ⊤ where
  mp h := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h : LT.lt 0 (HSub.hSub a b)
      ⊢ Or (LT.lt b a) (Eq b Top.top)
    -/
    refine or_iff_not_imp_right.mpr fun h2 ↦ ?_
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h : LT.lt 0 (HSub.hSub a b)
      h2 : Not (Eq b Top.top)
      ⊢ LT.lt b a
    -/
    replace h := strictMono_add_left_of_ne_top _ h2 h
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h2 : Not (Eq b Top.top)
      h : LT.lt ((fun x => HAdd.hAdd x b) 0) ((fun x => HAdd.hAdd x b) (HSub.hSub a  …
      ⊢ LT.lt b a
    -/
    simp only [zero_add] at h
    rw [sub_eq_add_neg, add_assoc, add_comm (-b),
      add_neg_cancel_of_ne_top h2, add_zero] at h
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h2 : Not (Eq b Top.top)
      h : LT.lt b a
      ⊢ LT.lt b a
    -/
    exact h
    /-
      🎉 no goals
    -/
  mpr h := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroupWithTop α
      a b : α
      h : Or (LT.lt b a) (Eq b Top.top)
      ⊢ LT.lt 0 (HSub.hSub a b)
    -/
    rcases h with h | h
      /-
        case inl
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        h : LT.lt b a
        ⊢ LT.lt 0 (HSub.hSub a b)
      -/
    · convert strictMono_add_left_of_ne_top (-b) (by simp [h.ne_top]) h using 1
        /-
          case h.e'_3
          α : Type u_1
          inst✝ : LinearOrderedAddCommGroupWithTop α
          a b : α
          h : LT.lt b a
          ⊢ Eq 0 ((fun x => HAdd.hAdd x (Neg.neg b)) b)
        -/
      · simp [add_neg_cancel_of_ne_top h.ne_top]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_4
          α : Type u_1
          inst✝ : LinearOrderedAddCommGroupWithTop α
          a b : α
          h : LT.lt b a
          ⊢ Eq (HSub.hSub a b) ((fun x => HAdd.hAdd x (Neg.neg b)) a)
        -/
      · simp [sub_eq_add_neg]
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        h : Eq b Top.top
        ⊢ LT.lt 0 (HSub.hSub a b)
      -/
    · rw [h]
      /-
        case inr
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        h : Eq b Top.top
        ⊢ LT.lt 0 (HSub.hSub a Top.top)
      -/
      simp only [sub_eq_add_neg, LinearOrderedAddCommGroupWithTop.neg_top, add_top]
      /-
        case inr
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        h : Eq b Top.top
        ⊢ LT.lt 0 Top.top
      -/
      apply lt_of_le_of_ne le_top
      /-
        case inr
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroupWithTop α
        a b : α
        h : Eq b Top.top
        ⊢ Ne 0 Top.top
      -/
      exact Ne.symm top_ne_zero
      /-
        🎉 no goals
      -/


