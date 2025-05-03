/-- Type of natural numbers with infinity (`⊤`) -/
def PartENat : Type :=
  Part ℕ


/-- The computable embedding `ℕ → PartENat`.

This coincides with the coercion `coe : ℕ → PartENat`, see `PartENat.some_eq_natCast`. -/
@[coe]
def some : ℕ → PartENat :=
  Part.some


instance : Zero PartENat :=
  ⟨some 0⟩


instance : Inhabited PartENat :=
  ⟨0⟩


instance : One PartENat :=
  ⟨some 1⟩


instance : Add PartENat :=
  ⟨fun x y => ⟨x.Dom ∧ y.Dom, fun h => get x h.1 + get y h.2⟩⟩


instance (n : ℕ) : Decidable (some n).Dom :=
  isTrue trivial


@[simp]
theorem dom_some (x : ℕ) : (some x).Dom :=
  trivial


instance addCommMonoid : AddCommMonoid PartENat where
  add := (· + ·)
  zero := 0
  add_comm _ _ := Part.ext' and_comm fun _ _ => add_comm _ _
  zero_add _ := Part.ext' (iff_of_eq (true_and _)) fun _ _ => zero_add _
  add_zero _ := Part.ext' (iff_of_eq (and_true _)) fun _ _ => add_zero _
  add_assoc _ _ _ := Part.ext' and_assoc fun _ _ => add_assoc _ _ _
  nsmul := nsmulRec


instance : AddCommMonoidWithOne PartENat :=
  { PartENat.addCommMonoid with
    one := 1
    natCast := some
    natCast_zero := rfl
    natCast_succ := fun _ => Part.ext' (iff_of_eq (true_and _)).symm fun _ _ => rfl }


theorem some_eq_natCast (n : ℕ) : some n = n :=
  rfl


instance : CharZero PartENat where
  cast_injective := Part.some_injective


/-- Alias of `Nat.cast_inj` specialized to `PartENat` --/
theorem natCast_inj {x y : ℕ} : (x : PartENat) = y ↔ x = y :=
  Nat.cast_inj


@[simp]
theorem dom_natCast (x : ℕ) : (x : PartENat).Dom :=
  trivial

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem dom_ofNat (x : ℕ) [x.AtLeastTwo] : (no_index (OfNat.ofNat x : PartENat)).Dom :=
  trivial


@[simp]
theorem dom_zero : (0 : PartENat).Dom :=
  trivial


@[simp]
theorem dom_one : (1 : PartENat).Dom :=
  trivial


instance : CanLift PartENat ℕ (↑) Dom :=
  ⟨fun n hn => ⟨n.get hn, Part.some_get _⟩⟩


instance : LE PartENat :=
  ⟨fun x y => ∃ h : y.Dom → x.Dom, ∀ hy : y.Dom, x.get (h hy) ≤ y.get hy⟩


instance : Top PartENat :=
  ⟨none⟩


instance : Bot PartENat :=
  ⟨0⟩


instance : Max PartENat :=
  ⟨fun x y => ⟨x.Dom ∧ y.Dom, fun h => x.get h.1 ⊔ y.get h.2⟩⟩


theorem le_def (x y : PartENat) :
    x ≤ y ↔ ∃ h : y.Dom → x.Dom, ∀ hy : y.Dom, x.get (h hy) ≤ y.get hy :=
  Iff.rfl


@[elab_as_elim]
protected theorem casesOn' {P : PartENat → Prop} :
    ∀ a : PartENat, P ⊤ → (∀ n : ℕ, P (some n)) → P a :=
  Part.induction_on


@[elab_as_elim]
protected theorem casesOn {P : PartENat → Prop} : ∀ a : PartENat, P ⊤ → (∀ n : ℕ, P n) → P a := by
  /-
    P : PartENat → Prop
    ⊢ ∀ (a : PartENat), P Top.top → (∀ (n : Nat), P ↑n) → P a
  -/
  exact PartENat.casesOn'
  /-
    🎉 no goals
  -/

-- not a simp lemma as we will provide a `LinearOrderedAddCommMonoidWithTop` instance later

theorem top_add (x : PartENat) : ⊤ + x = ⊤ :=
  Part.ext' (iff_of_eq (false_and _)) fun h => h.left.elim

-- not a simp lemma as we will provide a `LinearOrderedAddCommMonoidWithTop` instance later

                                                 /-
                                                   x : PartENat
                                                   ⊢ Eq (HAdd.hAdd x Top.top) Top.top
                                                 -/
theorem add_top (x : PartENat) : x + ⊤ = ⊤ := by rw [add_comm, top_add]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem natCast_get {x : PartENat} (h : x.Dom) : (x.get h : PartENat) = x := by
  /-
    x : PartENat
    h : x.Dom
    ⊢ Eq (↑(x.get h)) x
  -/
  exact Part.ext' (iff_of_true trivial h) fun _ _ => rfl
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem get_natCast' (x : ℕ) (h : (x : PartENat).Dom) : get (x : PartENat) h = x := by
  /-
    x : Nat
    h : (↑x).Dom
    ⊢ Eq ((↑x).get h) x
  -/
  rw [← natCast_inj, natCast_get]
  /-
    🎉 no goals
  -/


theorem get_natCast {x : ℕ} : get (x : PartENat) (dom_natCast x) = x :=
  get_natCast' _ _


theorem coe_add_get {x : ℕ} {y : PartENat} (h : ((x : PartENat) + y).Dom) :
    get ((x : PartENat) + y) h = x + get y h.2 := by
  /-
    x : Nat
    y : PartENat
    h : (HAdd.hAdd (↑x) y).Dom
    ⊢ Eq ((HAdd.hAdd (↑x) y).get h) (HAdd.hAdd x (y.get ⋯))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem get_add {x y : PartENat} (h : (x + y).Dom) : get (x + y) h = x.get h.1 + y.get h.2 :=
  rfl


@[simp]
theorem get_zero (h : (0 : PartENat).Dom) : (0 : PartENat).get h = 0 :=
  rfl


@[simp]
theorem get_one (h : (1 : PartENat).Dom) : (1 : PartENat).get h = 1 :=
  rfl

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem get_ofNat' (x : ℕ) [x.AtLeastTwo] (h : (no_index (OfNat.ofNat x : PartENat)).Dom) :
    Part.get (no_index (OfNat.ofNat x : PartENat)) h = (no_index (OfNat.ofNat x)) :=
  get_natCast' x h


nonrec theorem get_eq_iff_eq_some {a : PartENat} {ha : a.Dom} {b : ℕ} : a.get ha = b ↔ a = some b :=
  get_eq_iff_eq_some


theorem get_eq_iff_eq_coe {a : PartENat} {ha : a.Dom} {b : ℕ} : a.get ha = b ↔ a = b := by
  /-
    a : PartENat
    ha : a.Dom
    b : Nat
    ⊢ Iff (Eq (a.get ha) b) (Eq a ↑b)
  -/
  rw [get_eq_iff_eq_some]
  /-
    a : PartENat
    ha : a.Dom
    b : Nat
    ⊢ Iff (Eq a ↑b) (Eq a ↑b)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem dom_of_le_of_dom {x y : PartENat} : x ≤ y → y.Dom → x.Dom := fun ⟨h, _⟩ => h


theorem dom_of_le_some {x : PartENat} {y : ℕ} (h : x ≤ some y) : x.Dom :=
  dom_of_le_of_dom h trivial


theorem dom_of_le_natCast {x : PartENat} {y : ℕ} (h : x ≤ y) : x.Dom := by
  /-
    x : PartENat
    y : Nat
    h : LE.le x ↑y
    ⊢ x.Dom
  -/
  exact dom_of_le_some h
  /-
    🎉 no goals
  -/


instance decidableLe (x y : PartENat) [Decidable x.Dom] [Decidable y.Dom] : Decidable (x ≤ y) :=
  if hx : x.Dom then
    decidable_of_decidable_of_iff (le_def x y).symm
  else
    if hy : y.Dom then isFalse fun h => hx <| dom_of_le_of_dom h hy
    else isTrue ⟨fun h => (hy h).elim, fun h => (hy h).elim⟩

-- Porting note: Removed. Use `Nat.castAddMonoidHom` instead.


instance partialOrder : PartialOrder PartENat where
  le := (· ≤ ·)
  le_refl _ := ⟨id, fun _ => le_rfl⟩
  le_trans := fun _ _ _ ⟨hxy₁, hxy₂⟩ ⟨hyz₁, hyz₂⟩ =>
    ⟨hxy₁ ∘ hyz₁, fun _ => le_trans (hxy₂ _) (hyz₂ _)⟩
  lt_iff_le_not_le _ _ := Iff.rfl
  le_antisymm := fun _ _ ⟨hxy₁, hxy₂⟩ ⟨hyx₁, hyx₂⟩ =>
    Part.ext' ⟨hyx₁, hxy₁⟩ fun _ _ => le_antisymm (hxy₂ _) (hyx₂ _)


theorem lt_def (x y : PartENat) : x < y ↔ ∃ hx : x.Dom, ∀ hy : y.Dom, x.get hx < y.get hy := by
  /-
    x y : PartENat
    ⊢ Iff (LT.lt x y) (Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy))
  -/
  rw [lt_iff_le_not_le, le_def, le_def, not_exists]
  /-
    x y : PartENat
    ⊢ Iff (And (Exists fun h => ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)) (∀ (x_ …
  -/
  constructor
    /-
      case mp
      x y : PartENat
      ⊢ And (Exists fun h => ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)) (∀ (x_1 : x …
    -/
  · rintro ⟨⟨hyx, H⟩, h⟩
    /-
      case mp.intro.intro
      x y : PartENat
      h : ∀ (x_1 : x.Dom → y.Dom), Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
      hyx : y.Dom → x.Dom
      H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
      ⊢ Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
    -/
    by_cases hx : x.Dom
      /-
        case pos
        x y : PartENat
        h : ∀ (x_1 : x.Dom → y.Dom), Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : x.Dom
        ⊢ Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      -/
    · use hx
      /-
        case h
        x y : PartENat
        h : ∀ (x_1 : x.Dom → y.Dom), Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : x.Dom
        ⊢ ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      -/
      intro hy
      /-
        case h
        x y : PartENat
        h : ∀ (x_1 : x.Dom → y.Dom), Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : x.Dom
        hy : y.Dom
        ⊢ LT.lt (x.get hx) (y.get hy)
      -/
      specialize H hy
      /-
        case h
        x y : PartENat
        h : ∀ (x_1 : x.Dom → y.Dom), Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
        hyx : y.Dom → x.Dom
        hx : x.Dom
        hy : y.Dom
        H : LE.le (x.get ⋯) (y.get hy)
        ⊢ LT.lt (x.get hx) (y.get hy)
      -/
      specialize h fun _ => hy
      /-
        case h
        x y : PartENat
        hyx : y.Dom → x.Dom
        hx : x.Dom
        hy : y.Dom
        H : LE.le (x.get ⋯) (y.get hy)
        h : Not (∀ (hy_1 : x.Dom), LE.le (y.get hy) (x.get hy_1))
        ⊢ LT.lt (x.get hx) (y.get hy)
      -/
      rw [not_forall] at h
      /-
        case h
        x y : PartENat
        hyx : y.Dom → x.Dom
        hx : x.Dom
        hy : y.Dom
        H : LE.le (x.get ⋯) (y.get hy)
        h : Exists fun x_1 => Not (LE.le (y.get hy) (x.get x_1))
        ⊢ LT.lt (x.get hx) (y.get hy)
      -/
      cases' h with hx' h
      /-
        case h.intro
        x y : PartENat
        hyx : y.Dom → x.Dom
        hx : x.Dom
        hy : y.Dom
        H : LE.le (x.get ⋯) (y.get hy)
        hx' : x.Dom
        h : Not (LE.le (y.get hy) (x.get hx'))
        ⊢ LT.lt (x.get hx) (y.get hy)
      -/
      rw [not_le] at h
      /-
        case h.intro
        x y : PartENat
        hyx : y.Dom → x.Dom
        hx : x.Dom
        hy : y.Dom
        H : LE.le (x.get ⋯) (y.get hy)
        hx' : x.Dom
        h : LT.lt (x.get hx') (y.get hy)
        ⊢ LT.lt (x.get hx) (y.get hy)
      -/
      exact h
      /-
        🎉 no goals
      -/
      /-
        case neg
        x y : PartENat
        h : ∀ (x_1 : x.Dom → y.Dom), Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : Not x.Dom
        ⊢ Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      -/
    · specialize h fun hx' => (hx hx').elim
      /-
        case neg
        x y : PartENat
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : Not x.Dom
        h : Not (∀ (hy : x.Dom), LE.le (y.get ⋯) (x.get hy))
        ⊢ Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      -/
      rw [not_forall] at h
      /-
        case neg
        x y : PartENat
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : Not x.Dom
        h : Exists fun x_1 => Not (LE.le (y.get ⋯) (x.get x_1))
        ⊢ Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      -/
      cases' h with hx' h
      /-
        case neg.intro
        x y : PartENat
        hyx : y.Dom → x.Dom
        H : ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)
        hx : Not x.Dom
        hx' : x.Dom
        h : Not (LE.le (y.get ⋯) (x.get hx'))
        ⊢ Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      -/
      exact (hx hx').elim
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x y : PartENat
      ⊢ (Exists fun hx => ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)) → And (Exists …
    -/
  · rintro ⟨hx, H⟩
    /-
      case mpr.intro
      x y : PartENat
      hx : x.Dom
      H : ∀ (hy : y.Dom), LT.lt (x.get hx) (y.get hy)
      ⊢ And (Exists fun h => ∀ (hy : y.Dom), LE.le (x.get ⋯) (y.get hy)) (∀ (x_1 : x …
    -/
    exact ⟨⟨fun _ => hx, fun hy => (H hy).le⟩, fun hxy h => not_lt_of_le (h _) (H _)⟩
    /-
      🎉 no goals
    -/


noncomputable instance orderedAddCommMonoid : OrderedAddCommMonoid PartENat :=
  { PartENat.partialOrder, PartENat.addCommMonoid with
    add_le_add_left := fun a b ⟨h₁, h₂⟩ c =>
                             /-
                               a b : PartENat
                               x✝ : LE.le a b
                               c : PartENat
                               h₁ : b.Dom → a.Dom
                               h₂ : ∀ (hy : b.Dom), LE.le (a.get ⋯) (b.get hy)
                               ⊢ LE.le (HAdd.hAdd Top.top a) (HAdd.hAdd Top.top b)
                             -/
      PartENat.casesOn c (by simp [top_add]) fun c =>
                             /-
                               🎉 no goals
                             -/
        ⟨fun h => And.intro (dom_natCast _) (h₁ h.2), fun h => by
          /-
            a b : PartENat
            x✝ : LE.le a b
            c✝ : PartENat
            h₁ : b.Dom → a.Dom
            h₂ : ∀ (hy : b.Dom), LE.le (a.get ⋯) (b.get hy)
            c : Nat
            h : (HAdd.hAdd (↑c) b).Dom
            ⊢ LE.le ((HAdd.hAdd (↑c) a).get ⋯) ((HAdd.hAdd (↑c) b).get h)
          -/
          simpa only [coe_add_get] using add_le_add_left (h₂ _) c⟩ }
          /-
            🎉 no goals
          -/


instance semilatticeSup : SemilatticeSup PartENat :=
  { PartENat.partialOrder with
    sup := (· ⊔ ·)
    le_sup_left := fun _ _ => ⟨And.left, fun _ => le_sup_left⟩
    le_sup_right := fun _ _ => ⟨And.right, fun _ => le_sup_right⟩
    sup_le := fun _ _ _ ⟨hx₁, hx₂⟩ ⟨hy₁, hy₂⟩ =>
      ⟨fun hz => ⟨hx₁ hz, hy₁ hz⟩, fun _ => sup_le (hx₂ _) (hy₂ _)⟩ }


instance orderBot : OrderBot PartENat where
  bot := ⊥
  bot_le _ := ⟨fun _ => trivial, fun _ => Nat.zero_le _⟩


instance orderTop : OrderTop PartENat where
  top := ⊤
  le_top _ := ⟨fun h => False.elim h, fun hy => False.elim hy⟩


instance : ZeroLEOneClass PartENat where
  zero_le_one := bot_le


/-- Alias of `Nat.cast_le` specialized to `PartENat` --/
theorem coe_le_coe {x y : ℕ} : (x : PartENat) ≤ y ↔ x ≤ y := Nat.cast_le


/-- Alias of `Nat.cast_lt` specialized to `PartENat` --/
theorem coe_lt_coe {x y : ℕ} : (x : PartENat) < y ↔ x < y := Nat.cast_lt


@[simp]
theorem get_le_get {x y : PartENat} {hx : x.Dom} {hy : y.Dom} : x.get hx ≤ y.get hy ↔ x ≤ y := by
  conv =>
    lhs
    rw [← coe_le_coe, natCast_get, natCast_get]


theorem le_coe_iff (x : PartENat) (n : ℕ) : x ≤ n ↔ ∃ h : x.Dom, x.get h ≤ n := by
  /-
    x : PartENat
    n : Nat
    ⊢ Iff (LE.le x ↑n) (Exists fun h => LE.le (x.get h) n)
  -/
  show (∃ h : True → x.Dom, _) ↔ ∃ h : x.Dom, x.get h ≤ n
  /-
    x : PartENat
    n : Nat
    ⊢ Iff (Exists fun h => ∀ (hy : (↑n).Dom), LE.le (x.get ⋯) ((↑n).get hy)) (Exis …
  -/
  simp only [forall_prop_of_true, dom_natCast, get_natCast']
  /-
    🎉 no goals
  -/


theorem lt_coe_iff (x : PartENat) (n : ℕ) : x < n ↔ ∃ h : x.Dom, x.get h < n := by
  /-
    x : PartENat
    n : Nat
    ⊢ Iff (LT.lt x ↑n) (Exists fun h => LT.lt (x.get h) n)
  -/
  simp only [lt_def, forall_prop_of_true, get_natCast', dom_natCast]
  /-
    🎉 no goals
  -/


theorem coe_le_iff (n : ℕ) (x : PartENat) : (n : PartENat) ≤ x ↔ ∀ h : x.Dom, n ≤ x.get h := by
  /-
    n : Nat
    x : PartENat
    ⊢ Iff (LE.le (↑n) x) (∀ (h : x.Dom), LE.le n (x.get h))
  -/
  rw [← some_eq_natCast]
  /-
    n : Nat
    x : PartENat
    ⊢ Iff (LE.le (↑n) x) (∀ (h : x.Dom), LE.le n (x.get h))
  -/
  simp only [le_def, exists_prop_of_true, dom_some, forall_true_iff]
  /-
    n : Nat
    x : PartENat
    ⊢ Iff (∀ (hy : x.Dom), LE.le ((↑n).get ⋯) (x.get hy)) (∀ (h : x.Dom), LE.le n  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coe_lt_iff (n : ℕ) (x : PartENat) : (n : PartENat) < x ↔ ∀ h : x.Dom, n < x.get h := by
  /-
    n : Nat
    x : PartENat
    ⊢ Iff (LT.lt (↑n) x) (∀ (h : x.Dom), LT.lt n (x.get h))
  -/
  rw [← some_eq_natCast]
  /-
    n : Nat
    x : PartENat
    ⊢ Iff (LT.lt (↑n) x) (∀ (h : x.Dom), LT.lt n (x.get h))
  -/
  simp only [lt_def, exists_prop_of_true, dom_some, forall_true_iff]
  /-
    n : Nat
    x : PartENat
    ⊢ Iff (∀ (hy : x.Dom), LT.lt ((↑n).get ⋯) (x.get hy)) (∀ (h : x.Dom), LT.lt n  …
  -/
  rfl
  /-
    🎉 no goals
  -/


nonrec theorem eq_zero_iff {x : PartENat} : x = 0 ↔ x ≤ 0 :=
  eq_bot_iff


theorem ne_zero_iff {x : PartENat} : x ≠ 0 ↔ ⊥ < x :=
  bot_lt_iff_ne_bot.symm


theorem dom_of_lt {x y : PartENat} : x < y → x.Dom :=
  PartENat.casesOn x not_top_lt fun _ _ => dom_natCast _


theorem top_eq_none : (⊤ : PartENat) = Part.none :=
  rfl


@[simp]
theorem natCast_lt_top (x : ℕ) : (x : PartENat) < ⊤ :=
                                                    /-
                                                      x : Nat
                                                      h : Eq (↑x) Top.top
                                                      ⊢ Not (Eq (↑x).Dom Top.top.Dom)
                                                    -/
  Ne.lt_top fun h => absurd (congr_arg Dom h) <| by simp only [dom_natCast]; exact true_ne_false
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem zero_lt_top : (0 : PartENat) < ⊤ :=
  natCast_lt_top 0


@[simp]
theorem one_lt_top : (1 : PartENat) < ⊤ :=
  natCast_lt_top 1

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_lt_top (x : ℕ) [x.AtLeastTwo] : (no_index (OfNat.ofNat x : PartENat)) < ⊤ :=
  natCast_lt_top x


@[simp]
theorem natCast_ne_top (x : ℕ) : (x : PartENat) ≠ ⊤ :=
  ne_of_lt (natCast_lt_top x)


@[simp]
theorem zero_ne_top : (0 : PartENat) ≠ ⊤ :=
  natCast_ne_top 0


@[simp]
theorem one_ne_top : (1 : PartENat) ≠ ⊤ :=
  natCast_ne_top 1

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_ne_top (x : ℕ) [x.AtLeastTwo] : (no_index (OfNat.ofNat x : PartENat)) ≠ ⊤ :=
  natCast_ne_top x


theorem not_isMax_natCast (x : ℕ) : ¬IsMax (x : PartENat) :=
  not_isMax_of_lt (natCast_lt_top x)


theorem ne_top_iff {x : PartENat} : x ≠ ⊤ ↔ ∃ n : ℕ, x = n := by
  /-
    x : PartENat
    ⊢ Iff (Ne x Top.top) (Exists fun n => Eq x ↑n)
  -/
  simpa only [← some_eq_natCast] using Part.ne_none_iff
  /-
    🎉 no goals
  -/


theorem ne_top_iff_dom {x : PartENat} : x ≠ ⊤ ↔ x.Dom := by
  /-
    x : PartENat
    ⊢ Iff (Ne x Top.top) x.Dom
  -/
  classical exact not_iff_comm.1 Part.eq_none_iff'.symm
  /-
    🎉 no goals
  -/


theorem not_dom_iff_eq_top {x : PartENat} : ¬x.Dom ↔ x = ⊤ :=
  Iff.not_left ne_top_iff_dom.symm


theorem ne_top_of_lt {x y : PartENat} (h : x < y) : x ≠ ⊤ :=
  ne_of_lt <| lt_of_lt_of_le h le_top


theorem eq_top_iff_forall_lt (x : PartENat) : x = ⊤ ↔ ∀ n : ℕ, (n : PartENat) < x := by
  /-
    x : PartENat
    ⊢ Iff (Eq x Top.top) (∀ (n : Nat), LT.lt (↑n) x)
  -/
  constructor
    /-
      case mp
      x : PartENat
      ⊢ Eq x Top.top → ∀ (n : Nat), LT.lt (↑n) x
    -/
  · rintro rfl n
    /-
      case mp
      n : Nat
      ⊢ LT.lt (↑n) Top.top
    -/
    exact natCast_lt_top _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : PartENat
      ⊢ (∀ (n : Nat), LT.lt (↑n) x) → Eq x Top.top
    -/
  · contrapose!
    /-
      case mpr
      x : PartENat
      ⊢ Ne x Top.top → Exists fun n => Not (LT.lt (↑n) x)
    -/
    rw [ne_top_iff]
    /-
      case mpr
      x : PartENat
      ⊢ (Exists fun n => Eq x ↑n) → Exists fun n => Not (LT.lt (↑n) x)
    -/
    rintro ⟨n, rfl⟩
    /-
      case mpr.intro
      n : Nat
      ⊢ Exists fun n_1 => Not (LT.lt ↑n_1 ↑n)
    -/
    exact ⟨n, irrefl _⟩
    /-
      🎉 no goals
    -/


theorem eq_top_iff_forall_le (x : PartENat) : x = ⊤ ↔ ∀ n : ℕ, (n : PartENat) ≤ x :=
  (eq_top_iff_forall_lt x).trans
    ⟨fun h n => (h n).le, fun h n => lt_of_lt_of_le (coe_lt_coe.mpr n.lt_succ_self) (h (n + 1))⟩


theorem pos_iff_one_le {x : PartENat} : 0 < x ↔ 1 ≤ x :=
  PartENat.casesOn x
        /-
          x : PartENat
          ⊢ Iff (LT.lt 0 Top.top) (LE.le 1 Top.top)
        -/
    (by simp only [le_top, natCast_lt_top, ← @Nat.cast_zero PartENat])
        /-
          🎉 no goals
        -/
    fun n => by
      /-
        x : PartENat
        n : Nat
        ⊢ Iff (LT.lt 0 ↑n) (LE.le 1 ↑n)
      -/
      rw [← Nat.cast_zero, ← Nat.cast_one, PartENat.coe_lt_coe, PartENat.coe_le_coe]
      /-
        x : PartENat
        n : Nat
        ⊢ Iff (LT.lt 0 n) (LE.le 1 n)
      -/
      rfl
      /-
        🎉 no goals
      -/


instance isTotal : IsTotal PartENat (· ≤ ·) where
  total x y :=
    PartENat.casesOn (P := fun z => z ≤ y ∨ y ≤ z) x (Or.inr le_top)
      (PartENat.casesOn y (fun _ => Or.inl le_top) fun x y =>
        (le_total x y).elim (Or.inr ∘ coe_le_coe.2) (Or.inl ∘ coe_le_coe.2))


noncomputable instance linearOrder : LinearOrder PartENat :=
  { PartENat.partialOrder with
    le_total := IsTotal.total
    decidableLE := Classical.decRel _
    max := (· ⊔ ·)
    -- Porting note: was `max_def := @sup_eq_maxDefault _ _ (id _) _ }`
    max_def := fun a b => by
      /-
        a b : PartENat
        ⊢ Eq (Max.max a b) (ite (LE.le a b) b a)
      -/
      change (fun a b => a ⊔ b) a b = _
      /-
        a b : PartENat
        ⊢ Eq ((fun a b => Max.max a b) a b) (ite (LE.le a b) b a)
      -/
      rw [@sup_eq_maxDefault PartENat _ (id _) _]
      /-
        a b : PartENat
        ⊢ Eq (maxDefault a b) (ite (LE.le a b) b a)
      -/
      rfl }
      /-
        🎉 no goals
      -/


instance boundedOrder : BoundedOrder PartENat :=
  { PartENat.orderTop, PartENat.orderBot with }


noncomputable instance lattice : Lattice PartENat :=
  { PartENat.semilatticeSup with
    inf := min
    inf_le_left := min_le_left
    inf_le_right := min_le_right
    le_inf := fun _ _ _ => le_min }


noncomputable instance : CanonicallyOrderedAddCommMonoid PartENat :=
  { PartENat.semilatticeSup, PartENat.orderBot,
    PartENat.orderedAddCommMonoid with
    le_self_add := fun a b =>
      PartENat.casesOn b (le_top.trans_eq (add_top _).symm) fun _ =>
        PartENat.casesOn a (top_add _).ge fun _ =>
          (coe_le_coe.2 le_self_add).trans_eq (Nat.cast_add _ _)
    exists_add_of_le := fun {a b} =>
      PartENat.casesOn b (fun _ => ⟨⊤, (add_top _).symm⟩) fun b =>
        PartENat.casesOn a (fun h => ((natCast_lt_top _).not_le h).elim) fun a h =>
          ⟨(b - a : ℕ), by
            /-
              a✝ b✝ : PartENat
              b a : Nat
              h : LE.le ↑a ↑b
              ⊢ Eq (↑b) (HAdd.hAdd ↑a ↑(HSub.hSub b a))
            -/
            rw [← Nat.cast_add, natCast_inj, add_comm, tsub_add_cancel_of_le (coe_le_coe.1 h)]⟩ }
            /-
              🎉 no goals
            -/


theorem eq_natCast_sub_of_add_eq_natCast {x y : PartENat} {n : ℕ} (h : x + y = n) :
    x = ↑(n - y.get (dom_of_le_natCast ((le_add_left le_rfl).trans_eq h))) := by
  /-
    x y : PartENat
    n : Nat
    h : Eq (HAdd.hAdd x y) ↑n
    ⊢ Eq x ↑(HSub.hSub n (y.get ⋯))
  -/
  lift x to ℕ using dom_of_le_natCast ((le_add_right le_rfl).trans_eq h)
  /-
    case intro
    y : PartENat
    n x : Nat
    h : Eq (HAdd.hAdd (↑x) y) ↑n
    ⊢ Eq ↑x ↑(HSub.hSub n (y.get ⋯))
  -/
  lift y to ℕ using dom_of_le_natCast ((le_add_left le_rfl).trans_eq h)
  /-
    case intro.intro
    n x y : Nat
    h : Eq (HAdd.hAdd ↑x ↑y) ↑n
    ⊢ Eq ↑x ↑(HSub.hSub n ((↑y).get ⋯))
  -/
  rw [← Nat.cast_add, natCast_inj] at h
  /-
    case intro.intro
    n x y : Nat
    h✝ : Eq (HAdd.hAdd ↑x ↑y) ↑n
    h : Eq (HAdd.hAdd x y) n
    ⊢ Eq ↑x ↑(HSub.hSub n ((↑y).get ⋯))
  -/
  rw [get_natCast, natCast_inj, eq_tsub_of_add_eq h]
  /-
    🎉 no goals
  -/


protected theorem add_lt_add_right {x y z : PartENat} (h : x < y) (hz : z ≠ ⊤) : x + z < y + z := by
  /-
    x y z : PartENat
    h : LT.lt x y
    hz : Ne z Top.top
    ⊢ LT.lt (HAdd.hAdd x z) (HAdd.hAdd y z)
  -/
  rcases ne_top_iff.mp (ne_top_of_lt h) with ⟨m, rfl⟩
  /-
    case intro
    y z : PartENat
    hz : Ne z Top.top
    m : Nat
    h : LT.lt (↑m) y
    ⊢ LT.lt (HAdd.hAdd (↑m) z) (HAdd.hAdd y z)
  -/
  rcases ne_top_iff.mp hz with ⟨k, rfl⟩
  /-
    case intro.intro
    y : PartENat
    m : Nat
    h : LT.lt (↑m) y
    k : Nat
    hz : Ne (↑k) Top.top
    ⊢ LT.lt (HAdd.hAdd ↑m ↑k) (HAdd.hAdd y ↑k)
  -/
  induction' y using PartENat.casesOn with n
    /-
      case intro.intro.a
      m k : Nat
      hz : Ne (↑k) Top.top
      h : LT.lt (↑m) Top.top
      ⊢ LT.lt (HAdd.hAdd ↑m ↑k) (HAdd.hAdd Top.top ↑k)
    -/
  · rw [top_add]
    -- Porting note: was apply_mod_cast natCast_lt_top
    /-
      case intro.intro.a
      m k : Nat
      hz : Ne (↑k) Top.top
      h : LT.lt (↑m) Top.top
      ⊢ LT.lt (HAdd.hAdd ↑m ↑k) Top.top
    -/
    norm_cast; apply natCast_lt_top
               /-
                 🎉 no goals
               -/
  /-
    case intro.intro.a
    m k : Nat
    hz : Ne (↑k) Top.top
    n : Nat
    h : LT.lt ↑m ↑n
    ⊢ LT.lt (HAdd.hAdd ↑m ↑k) (HAdd.hAdd ↑n ↑k)
  -/
  norm_cast at h
  -- Porting note: was `apply_mod_cast add_lt_add_right h`
  /-
    case intro.intro.a
    m k : Nat
    hz : Ne (↑k) Top.top
    n : Nat
    h : LT.lt m n
    ⊢ LT.lt (HAdd.hAdd ↑m ↑k) (HAdd.hAdd ↑n ↑k)
  -/
  norm_cast; apply add_lt_add_right h
             /-
               🎉 no goals
             -/


protected theorem add_lt_add_iff_right {x y z : PartENat} (hz : z ≠ ⊤) : x + z < y + z ↔ x < y :=
  ⟨lt_of_add_lt_add_right, fun h => PartENat.add_lt_add_right h hz⟩


protected theorem add_lt_add_iff_left {x y z : PartENat} (hz : z ≠ ⊤) : z + x < z + y ↔ x < y := by
  /-
    x y z : PartENat
    hz : Ne z Top.top
    ⊢ Iff (LT.lt (HAdd.hAdd z x) (HAdd.hAdd z y)) (LT.lt x y)
  -/
  rw [add_comm z, add_comm z, PartENat.add_lt_add_iff_right hz]
  /-
    🎉 no goals
  -/


protected theorem lt_add_iff_pos_right {x y : PartENat} (hx : x ≠ ⊤) : x < x + y ↔ 0 < y := by
  /-
    x y : PartENat
    hx : Ne x Top.top
    ⊢ Iff (LT.lt x (HAdd.hAdd x y)) (LT.lt 0 y)
  -/
  conv_rhs => rw [← PartENat.add_lt_add_iff_left hx]
  /-
    x y : PartENat
    hx : Ne x Top.top
    ⊢ Iff (LT.lt x (HAdd.hAdd x y)) (LT.lt (HAdd.hAdd x 0) (HAdd.hAdd x y))
  -/
  rw [add_zero]
  /-
    🎉 no goals
  -/


theorem lt_add_one {x : PartENat} (hx : x ≠ ⊤) : x < x + 1 := by
  /-
    x : PartENat
    hx : Ne x Top.top
    ⊢ LT.lt x (HAdd.hAdd x 1)
  -/
  rw [PartENat.lt_add_iff_pos_right hx]
  /-
    x : PartENat
    hx : Ne x Top.top
    ⊢ LT.lt 0 1
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem le_of_lt_add_one {x y : PartENat} (h : x < y + 1) : x ≤ y := by
  /-
    x y : PartENat
    h : LT.lt x (HAdd.hAdd y 1)
    ⊢ LE.le x y
  -/
  induction' y using PartENat.casesOn with n
    /-
      case a
      x : PartENat
      h : LT.lt x (HAdd.hAdd Top.top 1)
      ⊢ LE.le x Top.top
    -/
  · apply le_top
    /-
      🎉 no goals
    -/
  /-
    case a
    x : PartENat
    n : Nat
    h : LT.lt x (HAdd.hAdd (↑n) 1)
    ⊢ LE.le x ↑n
  -/
  rcases ne_top_iff.mp (ne_top_of_lt h) with ⟨m, rfl⟩
  -- Porting note: was `apply_mod_cast Nat.le_of_lt_succ; apply_mod_cast h`
  /-
    case a.intro
    n m : Nat
    h : LT.lt (↑m) (HAdd.hAdd (↑n) 1)
    ⊢ LE.le ↑m ↑n
  -/
  norm_cast; apply Nat.le_of_lt_succ; norm_cast at h
                                      /-
                                        🎉 no goals
                                      -/


theorem add_one_le_of_lt {x y : PartENat} (h : x < y) : x + 1 ≤ y := by
  /-
    x y : PartENat
    h : LT.lt x y
    ⊢ LE.le (HAdd.hAdd x 1) y
  -/
  induction' y using PartENat.casesOn with n
    /-
      case a
      x : PartENat
      h : LT.lt x Top.top
      ⊢ LE.le (HAdd.hAdd x 1) Top.top
    -/
  · apply le_top
    /-
      🎉 no goals
    -/
  /-
    case a
    x : PartENat
    n : Nat
    h : LT.lt x ↑n
    ⊢ LE.le (HAdd.hAdd x 1) ↑n
  -/
  rcases ne_top_iff.mp (ne_top_of_lt h) with ⟨m, rfl⟩
  -- Porting note: was `apply_mod_cast Nat.succ_le_of_lt; apply_mod_cast h`
  /-
    case a.intro
    n m : Nat
    h : LT.lt ↑m ↑n
    ⊢ LE.le (HAdd.hAdd (↑m) 1) ↑n
  -/
  norm_cast; apply Nat.succ_le_of_lt; norm_cast at h
                                      /-
                                        🎉 no goals
                                      -/


theorem add_one_le_iff_lt {x y : PartENat} (hx : x ≠ ⊤) : x + 1 ≤ y ↔ x < y := by
  /-
    x y : PartENat
    hx : Ne x Top.top
    ⊢ Iff (LE.le (HAdd.hAdd x 1) y) (LT.lt x y)
  -/
  refine ⟨fun h => ?_, add_one_le_of_lt⟩
  /-
    x y : PartENat
    hx : Ne x Top.top
    h : LE.le (HAdd.hAdd x 1) y
    ⊢ LT.lt x y
  -/
  rcases ne_top_iff.mp hx with ⟨m, rfl⟩
  /-
    case intro
    y : PartENat
    m : Nat
    hx : Ne (↑m) Top.top
    h : LE.le (HAdd.hAdd (↑m) 1) y
    ⊢ LT.lt (↑m) y
  -/
  induction' y using PartENat.casesOn with n
    /-
      case intro.a
      m : Nat
      hx : Ne (↑m) Top.top
      h : LE.le (HAdd.hAdd (↑m) 1) Top.top
      ⊢ LT.lt (↑m) Top.top
    -/
  · apply natCast_lt_top
    /-
      🎉 no goals
    -/
  -- Porting note: was `apply_mod_cast Nat.lt_of_succ_le; apply_mod_cast h`
  /-
    case intro.a
    m : Nat
    hx : Ne (↑m) Top.top
    n : Nat
    h : LE.le (HAdd.hAdd (↑m) 1) ↑n
    ⊢ LT.lt ↑m ↑n
  -/
  norm_cast; apply Nat.lt_of_succ_le; norm_cast at h
                                      /-
                                        🎉 no goals
                                      -/


theorem coe_succ_le_iff {n : ℕ} {e : PartENat} : ↑n.succ ≤ e ↔ ↑n < e := by
  /-
    n : Nat
    e : PartENat
    ⊢ Iff (LE.le (↑n.succ) e) (LT.lt (↑n) e)
  -/
  rw [Nat.succ_eq_add_one n, Nat.cast_add, Nat.cast_one, add_one_le_iff_lt (natCast_ne_top n)]
  /-
    🎉 no goals
  -/


theorem lt_add_one_iff_lt {x y : PartENat} (hx : x ≠ ⊤) : x < y + 1 ↔ x ≤ y := by
  /-
    x y : PartENat
    hx : Ne x Top.top
    ⊢ Iff (LT.lt x (HAdd.hAdd y 1)) (LE.le x y)
  -/
  refine ⟨le_of_lt_add_one, fun h => ?_⟩
  /-
    x y : PartENat
    hx : Ne x Top.top
    h : LE.le x y
    ⊢ LT.lt x (HAdd.hAdd y 1)
  -/
  rcases ne_top_iff.mp hx with ⟨m, rfl⟩
  /-
    case intro
    y : PartENat
    m : Nat
    hx : Ne (↑m) Top.top
    h : LE.le (↑m) y
    ⊢ LT.lt (↑m) (HAdd.hAdd y 1)
  -/
  induction' y using PartENat.casesOn with n
    /-
      case intro.a
      m : Nat
      hx : Ne (↑m) Top.top
      h : LE.le (↑m) Top.top
      ⊢ LT.lt (↑m) (HAdd.hAdd Top.top 1)
    -/
  · rw [top_add]
    /-
      case intro.a
      m : Nat
      hx : Ne (↑m) Top.top
      h : LE.le (↑m) Top.top
      ⊢ LT.lt (↑m) Top.top
    -/
    apply natCast_lt_top
    /-
      🎉 no goals
    -/
  -- Porting note: was `apply_mod_cast Nat.lt_succ_of_le; apply_mod_cast h`
  /-
    case intro.a
    m : Nat
    hx : Ne (↑m) Top.top
    n : Nat
    h : LE.le ↑m ↑n
    ⊢ LT.lt (↑m) (HAdd.hAdd (↑n) 1)
  -/
  norm_cast; apply Nat.lt_succ_of_le; norm_cast at h
                                      /-
                                        🎉 no goals
                                      -/


lemma lt_coe_succ_iff_le {x : PartENat} {n : ℕ} (hx : x ≠ ⊤) : x < n.succ ↔ x ≤ n := by
  /-
    x : PartENat
    n : Nat
    hx : Ne x Top.top
    ⊢ Iff (LT.lt x ↑n.succ) (LE.le x ↑n)
  -/
  rw [Nat.succ_eq_add_one n, Nat.cast_add, Nat.cast_one, lt_add_one_iff_lt hx]
  /-
    🎉 no goals
  -/


theorem add_eq_top_iff {a b : PartENat} : a + b = ⊤ ↔ a = ⊤ ∨ b = ⊤ := by
  /-
    a b : PartENat
    ⊢ Iff (Eq (HAdd.hAdd a b) Top.top) (Or (Eq a Top.top) (Eq b Top.top))
  -/
  refine PartENat.casesOn a ?_ ?_
      /-
        case refine_1
        a b : PartENat
        ⊢ Iff (Eq (HAdd.hAdd Top.top b) Top.top) (Or (Eq Top.top Top.top) (Eq b Top.to …
      -/
  <;> refine PartENat.casesOn b ?_ ?_
      /-
        case refine_1.refine_1
        a b : PartENat
        ⊢ Iff (Eq (HAdd.hAdd Top.top Top.top) Top.top) (Or (Eq Top.top Top.top) (Eq To …
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
  <;> simp [top_add, add_top]
  /-
    case refine_2.refine_2
    a b : PartENat
    ⊢ ∀ (n n_1 : Nat), Not (Eq (HAdd.hAdd ↑n_1 ↑n) Top.top)
  -/
  simp only [← Nat.cast_add, PartENat.natCast_ne_top, forall_const, not_false_eq_true]
  /-
    🎉 no goals
  -/


protected theorem add_right_cancel_iff {a b c : PartENat} (hc : c ≠ ⊤) : a + c = b + c ↔ a = b := by
  /-
    a b c : PartENat
    hc : Ne c Top.top
    ⊢ Iff (Eq (HAdd.hAdd a c) (HAdd.hAdd b c)) (Eq a b)
  -/
  rcases ne_top_iff.1 hc with ⟨c, rfl⟩
  /-
    case intro
    a b : PartENat
    c : Nat
    hc : Ne (↑c) Top.top
    ⊢ Iff (Eq (HAdd.hAdd a ↑c) (HAdd.hAdd b ↑c)) (Eq a b)
  -/
  refine PartENat.casesOn a ?_ ?_
      /-
        case intro.refine_1
        a b : PartENat
        c : Nat
        hc : Ne (↑c) Top.top
        ⊢ Iff (Eq (HAdd.hAdd Top.top ↑c) (HAdd.hAdd b ↑c)) (Eq Top.top b)
      -/
  <;> refine PartENat.casesOn b ?_ ?_
      /-
        case intro.refine_1.refine_1
        a b : PartENat
        c : Nat
        hc : Ne (↑c) Top.top
        ⊢ Iff (Eq (HAdd.hAdd Top.top ↑c) (HAdd.hAdd Top.top ↑c)) (Eq Top.top Top.top)
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
  <;> simp [add_eq_top_iff, natCast_ne_top, @eq_comm _ (⊤ : PartENat), top_add]
  /-
    case intro.refine_2.refine_2
    a b : PartENat
    c : Nat
    hc : Ne (↑c) Top.top
    ⊢ ∀ (n n_1 : Nat), Iff (Eq (HAdd.hAdd ↑n_1 ↑c) (HAdd.hAdd ↑n ↑c)) (Eq n_1 n)
  -/
  simp only [← Nat.cast_add, add_left_cancel_iff, PartENat.natCast_inj, add_comm, forall_const]
  /-
    🎉 no goals
  -/


protected theorem add_left_cancel_iff {a b c : PartENat} (ha : a ≠ ⊤) : a + b = a + c ↔ b = c := by
  /-
    a b c : PartENat
    ha : Ne a Top.top
    ⊢ Iff (Eq (HAdd.hAdd a b) (HAdd.hAdd a c)) (Eq b c)
  -/
  rw [add_comm a, add_comm a, PartENat.add_right_cancel_iff ha]
  /-
    🎉 no goals
  -/


/-- Computably converts a `PartENat` to a `ℕ∞`. -/
def toWithTop (x : PartENat) [Decidable x.Dom] : ℕ∞ :=
  x.toOption


theorem toWithTop_top :
    have : Decidable (⊤ : PartENat).Dom := Part.noneDecidable
    toWithTop ⊤ = ⊤ :=
  rfl


@[simp]
theorem toWithTop_top' {h : Decidable (⊤ : PartENat).Dom} : toWithTop ⊤ = ⊤ := by
  /-
    h : Decidable Top.top.Dom
    ⊢ Eq Top.top.toWithTop Top.top
  -/
  convert toWithTop_top
  /-
    🎉 no goals
  -/


theorem toWithTop_zero :
    have : Decidable (0 : PartENat).Dom := someDecidable 0
    toWithTop 0 = 0 :=
  rfl


@[simp]
theorem toWithTop_zero' {h : Decidable (0 : PartENat).Dom} : toWithTop 0 = 0 := by
  /-
    h : Decidable (Part.Dom 0)
    ⊢ Eq (PartENat.toWithTop 0) 0
  -/
  convert toWithTop_zero
  /-
    🎉 no goals
  -/


theorem toWithTop_one :
    have : Decidable (1 : PartENat).Dom := someDecidable 1
    toWithTop 1 = 1 :=
  rfl


@[simp]
theorem toWithTop_one' {h : Decidable (1 : PartENat).Dom} : toWithTop 1 = 1 := by
  /-
    h : Decidable (Part.Dom 1)
    ⊢ Eq (PartENat.toWithTop 1) 1
  -/
  convert toWithTop_one
  /-
    🎉 no goals
  -/


theorem toWithTop_some (n : ℕ) : toWithTop (some n) = n :=
  rfl


theorem toWithTop_natCast (n : ℕ) {_ : Decidable (n : PartENat).Dom} : toWithTop n = n := by
  /-
    n : Nat
    x✝ : Decidable (↑n).Dom
    ⊢ Eq (↑n).toWithTop ↑n
  -/
  simp only [← toWithTop_some]
  /-
    n : Nat
    x✝ : Decidable (↑n).Dom
    ⊢ Eq (↑n).toWithTop (↑n).toWithTop
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem toWithTop_natCast' (n : ℕ) {_ : Decidable (n : PartENat).Dom} :
    toWithTop (n : PartENat) = n := by
  /-
    n : Nat
    x✝ : Decidable (↑n).Dom
    ⊢ Eq (↑n).toWithTop ↑n
  -/
  rw [toWithTop_natCast n]
  /-
    🎉 no goals
  -/


@[simp]
theorem toWithTop_ofNat (n : ℕ) [n.AtLeastTwo] {_ : Decidable (OfNat.ofNat n : PartENat).Dom} :
    toWithTop (no_index (OfNat.ofNat n : PartENat)) = OfNat.ofNat n := toWithTop_natCast' n

-- Porting note: statement changed. Mathlib 3 statement was
-- ```
-- @[simp] lemma to_with_top_le {x y : part_enat} :
--   Π [decidable x.dom] [decidable y.dom], by exactI to_with_top x ≤ to_with_top y ↔ x ≤ y :=
-- ```
-- This used to be really slow to typecheck when the definition of `ENat`
-- was still `deriving AddCommMonoidWithOne`. Now that I removed that it is fine.
-- (The problem was that the last `simp` got stuck at `CharZero ℕ∞ ≟ CharZero ℕ∞` where
-- one side used `instENatAddCommMonoidWithOne` and the other used
-- `NonAssocSemiring.toAddCommMonoidWithOne`. Now the former doesn't exist anymore.)

@[simp]
theorem toWithTop_le {x y : PartENat} [hx : Decidable x.Dom] [hy : Decidable y.Dom] :
    toWithTop x ≤ toWithTop y ↔ x ≤ y := by
  /-
    x y : PartENat
    hx : Decidable x.Dom
    hy : Decidable y.Dom
    ⊢ Iff (LE.le x.toWithTop y.toWithTop) (LE.le x y)
  -/
  induction y using PartENat.casesOn generalizing hy
    /-
      case a
      x : PartENat
      hx : Decidable x.Dom
      hy : Decidable Top.top.Dom
      ⊢ Iff (LE.le x.toWithTop Top.top.toWithTop) (LE.le x Top.top)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case a
    x : PartENat
    hx : Decidable x.Dom
    n✝ : Nat
    hy : Decidable (↑n✝).Dom
    ⊢ Iff (LE.le x.toWithTop (↑n✝).toWithTop) (LE.le x ↑n✝)
  -/
  induction x using PartENat.casesOn generalizing hx
    /-
      case a.a
      n✝ : Nat
      hy : Decidable (↑n✝).Dom
      hx : Decidable Top.top.Dom
      ⊢ Iff (LE.le Top.top.toWithTop (↑n✝).toWithTop) (LE.le Top.top ↑n✝)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a.a
      n✝¹ : Nat
      hy : Decidable (↑n✝¹).Dom
      n✝ : Nat
      hx : Decidable (↑n✝).Dom
      ⊢ Iff (LE.le (↑n✝).toWithTop (↑n✝¹).toWithTop) (LE.le ↑n✝ ↑n✝¹)
    -/
  · simp -- Porting note: this takes too long.
    /-
      🎉 no goals
    -/

/-
Porting note: As part of the investigation above, I noticed that Lean4 does not
find the following two instances which it could find in Lean3 automatically:
```
#synth Decidable (⊤ : PartENat).Dom
variable {n : ℕ}
#synth Decidable (n : PartENat).Dom
```
-/


@[simp]
theorem toWithTop_lt {x y : PartENat} [Decidable x.Dom] [Decidable y.Dom] :
    toWithTop x < toWithTop y ↔ x < y :=
  lt_iff_lt_of_le_iff_le toWithTop_le


/-- Coercion from `ℕ∞` to `PartENat`. -/
@[coe]
def ofENat : ℕ∞ → PartENat :=
  fun x => match x with
  | Option.none => none
  | Option.some n => some n

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance : Coe ℕ∞ PartENat := ⟨ofENat⟩

-- Porting note: new. This could probably be moved to tests or removed.

@[simp, norm_cast]
lemma ofENat_top : ofENat ⊤ = ⊤ := rfl


@[simp, norm_cast]
lemma ofENat_coe (n : ℕ) : ofENat n = n := rfl


@[simp, norm_cast]
theorem ofENat_zero : ofENat 0 = 0 := rfl


@[simp, norm_cast]
theorem ofENat_one : ofENat 1 = 1 := rfl


@[simp, norm_cast]
theorem ofENat_ofNat (n : Nat) [n.AtLeastTwo] : ofENat (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  rfl


@[simp, norm_cast]
theorem toWithTop_ofENat (n : ℕ∞) {_ : Decidable (n : PartENat).Dom} : toWithTop (↑n) = n := by
  cases n with
  | top => simp
  | coe n => simp


@[simp, norm_cast]
theorem ofENat_toWithTop (x : PartENat) {_ : Decidable (x : PartENat).Dom} : toWithTop x = x := by
  /-
    x : PartENat
    x✝ : Decidable x.Dom
    ⊢ Eq (↑x.toWithTop) x
  -/
                                         /-
                                           🎉 no goals
                                         -/
  induction x using PartENat.casesOn <;> simp
                                         /-
                                           🎉 no goals
                                         -/


@[simp, norm_cast]
theorem ofENat_le {x y : ℕ∞} : ofENat x ≤ ofENat y ↔ x ≤ y := by
  classical
  rw [← toWithTop_le, toWithTop_ofENat, toWithTop_ofENat]


@[simp, norm_cast]
theorem ofENat_lt {x y : ℕ∞} : ofENat x < ofENat y ↔ x < y := by
  classical
  rw [← toWithTop_lt, toWithTop_ofENat, toWithTop_ofENat]


@[simp]
theorem toWithTop_add {x y : PartENat} : toWithTop (x + y) = toWithTop x + toWithTop y := by
  /-
    x y : PartENat
    ⊢ Eq (HAdd.hAdd x y).toWithTop (HAdd.hAdd x.toWithTop y.toWithTop)
  -/
  refine PartENat.casesOn y ?_ ?_ <;> refine PartENat.casesOn x ?_ ?_
  -- Porting note: was `simp [← Nat.cast_add, ← ENat.coe_add]`
    /-
      case refine_1.refine_1
      x y : PartENat
      ⊢ Eq (HAdd.hAdd Top.top Top.top).toWithTop (HAdd.hAdd Top.top.toWithTop Top.to …
    -/
  · simp only [add_top, toWithTop_top', _root_.add_top]
    /-
      🎉 no goals
    -/
    /-
      case refine_1.refine_2
      x y : PartENat
      ⊢ ∀ (n : Nat), Eq (HAdd.hAdd (↑n) Top.top).toWithTop (HAdd.hAdd (↑n).toWithTop …
    -/
  · simp only [add_top, toWithTop_top', toWithTop_natCast', _root_.add_top, forall_const]
    /-
      🎉 no goals
    -/
    /-
      case refine_2.refine_1
      x y : PartENat
      ⊢ ∀ (n : Nat), Eq (HAdd.hAdd Top.top ↑n).toWithTop (HAdd.hAdd Top.top.toWithTo …
    -/
  · simp only [top_add, toWithTop_top', toWithTop_natCast', _root_.top_add, forall_const]
    /-
      🎉 no goals
    -/
    /-
      case refine_2.refine_2
      x y : PartENat
      ⊢ ∀ (n n_1 : Nat), Eq (HAdd.hAdd ↑n ↑n_1).toWithTop (HAdd.hAdd (↑n).toWithTop  …
    -/
  · simp_rw [toWithTop_natCast', ← Nat.cast_add, toWithTop_natCast', forall_const]
    /-
      🎉 no goals
    -/


/-- `Equiv` between `PartENat` and `ℕ∞` (for the order isomorphism see
`withTopOrderIso`). -/
@[simps]
noncomputable def withTopEquiv : PartENat ≃ ℕ∞ where
  toFun x := toWithTop x
  invFun x := ↑x
                   /-
                     x : PartENat
                     ⊢ Eq ((fun x => ↑x) ((fun x => x.toWithTop) x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      x : ENat
                      ⊢ Eq ((fun x => x.toWithTop) ((fun x => ↑x) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


theorem withTopEquiv_top : withTopEquiv ⊤ = ⊤ := by
  /-
    ⊢ Eq (PartENat.withTopEquiv Top.top) Top.top
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_natCast (n : Nat) : withTopEquiv n = n := by
  /-
    n : Nat
    ⊢ Eq (PartENat.withTopEquiv ↑n) ↑n
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_zero : withTopEquiv 0 = 0 := by
  /-
    ⊢ Eq (PartENat.withTopEquiv 0) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_one : withTopEquiv 1 = 1 := by
  /-
    ⊢ Eq (PartENat.withTopEquiv 1) 1
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_ofNat (n : Nat) [n.AtLeastTwo] :
    withTopEquiv (no_index (OfNat.ofNat n)) = OfNat.ofNat n := by
  /-
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (PartENat.withTopEquiv (OfNat.ofNat n)) (OfNat.ofNat n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_le {x y : PartENat} : withTopEquiv x ≤ withTopEquiv y ↔ x ≤ y := by
  /-
    x y : PartENat
    ⊢ Iff (LE.le (PartENat.withTopEquiv x) (PartENat.withTopEquiv y)) (LE.le x y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_lt {x y : PartENat} : withTopEquiv x < withTopEquiv y ↔ x < y := by
  /-
    x y : PartENat
    ⊢ Iff (LT.lt (PartENat.withTopEquiv x) (PartENat.withTopEquiv y)) (LT.lt x y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_top : withTopEquiv.symm ⊤ = ⊤ := by
  /-
    ⊢ Eq (PartENat.withTopEquiv.symm Top.top) Top.top
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_coe (n : Nat) : withTopEquiv.symm n = n := by
  /-
    n : Nat
    ⊢ Eq (PartENat.withTopEquiv.symm ↑n) ↑n
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_zero : withTopEquiv.symm 0 = 0 := by
  /-
    ⊢ Eq (PartENat.withTopEquiv.symm 0) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_one : withTopEquiv.symm 1 = 1 := by
  /-
    ⊢ Eq (PartENat.withTopEquiv.symm 1) 1
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_ofNat (n : Nat) [n.AtLeastTwo] :
    withTopEquiv.symm (no_index (OfNat.ofNat n)) = OfNat.ofNat n := by
  /-
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (PartENat.withTopEquiv.symm (OfNat.ofNat n)) (OfNat.ofNat n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_le {x y : ℕ∞} : withTopEquiv.symm x ≤ withTopEquiv.symm y ↔ x ≤ y := by
  /-
    x y : ENat
    ⊢ Iff (LE.le (PartENat.withTopEquiv.symm x) (PartENat.withTopEquiv.symm y)) (L …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem withTopEquiv_symm_lt {x y : ℕ∞} : withTopEquiv.symm x < withTopEquiv.symm y ↔ x < y := by
  /-
    x y : ENat
    ⊢ Iff (LT.lt (PartENat.withTopEquiv.symm x) (PartENat.withTopEquiv.symm y)) (L …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `toWithTop` induces an order isomorphism between `PartENat` and `ℕ∞`. -/
noncomputable def withTopOrderIso : PartENat ≃o ℕ∞ :=
  { withTopEquiv with map_rel_iff' := @fun _ _ => withTopEquiv_le }


/-- `toWithTop` induces an additive monoid isomorphism between `PartENat` and `ℕ∞`. -/
noncomputable def withTopAddEquiv : PartENat ≃+ ℕ∞ :=
  { withTopEquiv with
    map_add' := fun x y => by
      /-
        x y : PartENat
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (__src✝.toFun x) (__src✝.toFun  …
      -/
      simp only [withTopEquiv]
      /-
        x y : PartENat
        ⊢ Eq (HAdd.hAdd x y).toWithTop (HAdd.hAdd x.toWithTop y.toWithTop)
      -/
      exact toWithTop_add }
      /-
        🎉 no goals
      -/


theorem lt_wf : @WellFounded PartENat (· < ·) := by
  classical
    change WellFounded fun a b : PartENat => a < b
    simp_rw [← withTopEquiv_lt]
    exact InvImage.wf _ wellFounded_lt


instance : WellFoundedLT PartENat :=
  ⟨lt_wf⟩


instance wellFoundedRelation : WellFoundedRelation PartENat :=
  ⟨(· < ·), lt_wf⟩


/-- The smallest `PartENat` satisfying a (decidable) predicate `P : ℕ → Prop` -/
def find : PartENat :=
  ⟨∃ n, P n, Nat.find⟩


@[simp]
theorem find_get (h : (find P).Dom) : (find P).get h = Nat.find h :=
  rfl


theorem find_dom (h : ∃ n, P n) : (find P).Dom :=
  h


theorem lt_find (n : ℕ) (h : ∀ m ≤ n, ¬P m) : (n : PartENat) < find P := by
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    ⊢ LT.lt (↑n) (PartENat.find P)
  -/
  rw [coe_lt_iff]
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    ⊢ ∀ (h : (PartENat.find P).Dom), LT.lt n ((PartENat.find P).get h)
  -/
  intro h₁
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    h₁ : (PartENat.find P).Dom
    ⊢ LT.lt n ((PartENat.find P).get h₁)
  -/
  rw [find_get]
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    h₁ : (PartENat.find P).Dom
    ⊢ LT.lt n (Nat.find h₁)
  -/
  have h₂ := @Nat.find_spec P _ h₁
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    h₁ : (PartENat.find P).Dom
    h₂ : P (Nat.find h₁)
    ⊢ LT.lt n (Nat.find h₁)
  -/
  revert h₂
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    h₁ : (PartENat.find P).Dom
    ⊢ P (Nat.find h₁) → LT.lt n (Nat.find h₁)
  -/
  contrapose!
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : ∀ (m : Nat), LE.le m n → Not (P m)
    h₁ : (PartENat.find P).Dom
    ⊢ LE.le (Nat.find h₁) n → Not (P (Nat.find h₁))
  -/
  exact h _
  /-
    🎉 no goals
  -/


theorem lt_find_iff (n : ℕ) : (n : PartENat) < find P ↔ ∀ m ≤ n, ¬P m := by
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    ⊢ Iff (LT.lt (↑n) (PartENat.find P)) (∀ (m : Nat), LE.le m n → Not (P m))
  -/
  refine ⟨?_, lt_find P n⟩
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    ⊢ LT.lt (↑n) (PartENat.find P) → ∀ (m : Nat), LE.le m n → Not (P m)
  -/
  intro h m hm
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : LT.lt (↑n) (PartENat.find P)
    m : Nat
    hm : LE.le m n
    ⊢ Not (P m)
  -/
  by_cases H : (find P).Dom
    /-
      case pos
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      h : LT.lt (↑n) (PartENat.find P)
      m : Nat
      hm : LE.le m n
      H : (PartENat.find P).Dom
      ⊢ Not (P m)
    -/
  · apply Nat.find_min H
    /-
      case pos
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      h : LT.lt (↑n) (PartENat.find P)
      m : Nat
      hm : LE.le m n
      H : (PartENat.find P).Dom
      ⊢ LT.lt m (Nat.find H)
    -/
    rw [coe_lt_iff] at h
    /-
      case pos
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      h : ∀ (h : (PartENat.find P).Dom), LT.lt n ((PartENat.find P).get h)
      m : Nat
      hm : LE.le m n
      H : (PartENat.find P).Dom
      ⊢ LT.lt m (Nat.find H)
    -/
    specialize h H
    /-
      case pos
      P : Nat → Prop
      inst✝ : DecidablePred P
      n m : Nat
      hm : LE.le m n
      H : (PartENat.find P).Dom
      h : LT.lt n ((PartENat.find P).get H)
      ⊢ LT.lt m (Nat.find H)
    -/
    exact lt_of_le_of_lt hm h
    /-
      🎉 no goals
    -/
    /-
      case neg
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      h : LT.lt (↑n) (PartENat.find P)
      m : Nat
      hm : LE.le m n
      H : Not (PartENat.find P).Dom
      ⊢ Not (P m)
    -/
  · exact not_exists.mp H m
    /-
      🎉 no goals
    -/


theorem find_le (n : ℕ) (h : P n) : find P ≤ n := by
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : P n
    ⊢ LE.le (PartENat.find P) ↑n
  -/
  rw [le_coe_iff]
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : P n
    ⊢ Exists fun h => LE.le ((PartENat.find P).get h) n
  -/
  exact ⟨⟨_, h⟩, @Nat.find_min' P _ _ _ h⟩
  /-
    🎉 no goals
  -/


theorem find_eq_top_iff : find P = ⊤ ↔ ∀ n, ¬P n :=
  (eq_top_iff_forall_lt _).trans
    ⟨fun h n => (lt_find_iff P n).mp (h n) _ le_rfl, fun h n => lt_find P n fun _ _ => h _⟩


noncomputable instance : LinearOrderedAddCommMonoidWithTop PartENat :=
  { PartENat.linearOrder, PartENat.orderedAddCommMonoid, PartENat.orderTop with
    top_add' := top_add }


noncomputable instance : CompleteLinearOrder PartENat :=
  { lattice, withTopOrderIso.symm.toGaloisInsertion.liftCompleteLattice,
    linearOrder, LinearOrder.toBiheytingAlgebra with }


