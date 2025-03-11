                                                     /-
                                                       ⊢ Not (Eq Bool.true Bool.false)
                                                     -/
theorem true_eq_false_eq_False : ¬true = false := by decide
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                     /-
                                                       ⊢ Not (Eq Bool.false Bool.true)
                                                     -/
theorem false_eq_true_eq_False : ¬false = true := by decide
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                                             /-
                                                                               b : Bool
                                                                               ⊢ Eq (Not (Eq b Bool.true)) (Eq b Bool.false)
                                                                             -/
theorem eq_false_eq_not_eq_true (b : Bool) : (¬b = true) = (b = false) := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                                             /-
                                                                               b : Bool
                                                                               ⊢ Eq (Not (Eq b Bool.false)) (Eq b Bool.true)
                                                                             -/
theorem eq_true_eq_not_eq_false (b : Bool) : (¬b = false) = (b = true) := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem eq_false_of_not_eq_true {b : Bool} : ¬b = true → b = false :=
  Eq.mp (eq_false_eq_not_eq_true b)


theorem eq_true_of_not_eq_false {b : Bool} : ¬b = false → b = true :=
  Eq.mp (eq_true_eq_not_eq_false b)


theorem and_eq_true_eq_eq_true_and_eq_true (a b : Bool) :
                                                    /-
                                                      a b : Bool
                                                      ⊢ Eq (Eq (a.and b) Bool.true) (And (Eq a Bool.true) (Eq b Bool.true))
                                                    -/
    ((a && b) = true) = (a = true ∧ b = true) := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem or_eq_true_eq_eq_true_or_eq_true (a b : Bool) :
                                                    /-
                                                      a b : Bool
                                                      ⊢ Eq (Eq (a.or b) Bool.true) (Or (Eq a Bool.true) (Eq b Bool.true))
                                                    -/
    ((a || b) = true) = (a = true ∨ b = true) := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                                                /-
                                                                                  a : Bool
                                                                                  ⊢ Eq (Eq a.not Bool.true) (Eq a Bool.false)
                                                                                -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
theorem not_eq_true_eq_eq_false (a : Bool) : (not a = true) = (a = false) := by cases a <;> simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem and_eq_false_eq_eq_false_or_eq_false (a b : Bool) :
    ((a && b) = false) = (a = false ∨ b = false) := by
  /-
    a b : Bool
    ⊢ Eq (Eq (a.and b) Bool.false) (Or (Eq a Bool.false) (Eq b Bool.false))
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
  cases a <;> cases b <;> simp
                          /-
                            🎉 no goals
                          -/


theorem or_eq_false_eq_eq_false_and_eq_false (a b : Bool) :
    ((a || b) = false) = (a = false ∧ b = false) := by
  /-
    a b : Bool
    ⊢ Eq (Eq (a.or b) Bool.false) (And (Eq a Bool.false) (Eq b Bool.false))
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
  cases a <;> cases b <;> simp
                          /-
                            🎉 no goals
                          -/


                                                                                /-
                                                                                  a : Bool
                                                                                  ⊢ Eq (Eq a.not Bool.false) (Eq a Bool.true)
                                                                                -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
theorem not_eq_false_eq_eq_true (a : Bool) : (not a = false) = (a = true) := by cases a <;> simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


                                         /-
                                           ⊢ Eq (Eq Bool.false Bool.true) False
                                         -/
theorem coe_false : ↑false = False := by simp
                                         /-
                                           🎉 no goals
                                         -/


                                      /-
                                        ⊢ Eq (Eq Bool.true Bool.true) True
                                      -/
theorem coe_true : ↑true = True := by simp
                                      /-
                                        🎉 no goals
                                      -/


                                                      /-
                                                        ⊢ Eq (Eq Bool.false Bool.true) False
                                                      -/
theorem coe_sort_false : (false : Prop) = False := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                   /-
                                                     ⊢ Eq (Eq Bool.true Bool.true) True
                                                   -/
theorem coe_sort_true : (true : Prop) = True := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                                            /-
                                                                              p : Prop
                                                                              d : Decidable p
                                                                              ⊢ Iff (Eq (Decidable.decide p) Bool.true) p
                                                                            -/
theorem decide_iff (p : Prop) [d : Decidable p] : decide p = true ↔ p := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem decide_true {p : Prop} [Decidable p] : p → decide p :=
  (decide_iff p).2


theorem of_decide_true {p : Prop} [Decidable p] : decide p → p :=
  (decide_iff p).1


                                                         /-
                                                           b : Bool
                                                           ⊢ Iff (Not (Eq b Bool.true)) (Eq b Bool.false)
                                                         -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
theorem bool_iff_false {b : Bool} : ¬b ↔ b = false := by cases b <;> decide
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem bool_eq_false {b : Bool} : ¬b → b = false :=
  bool_iff_false.1


theorem decide_false_iff (p : Prop) {_ : Decidable p} : decide p = false ↔ ¬p :=
  bool_iff_false.symm.trans (not_congr (decide_iff _))


theorem decide_false {p : Prop} [Decidable p] : ¬p → decide p = false :=
  (decide_false_iff p).2


theorem of_decide_false {p : Prop} [Decidable p] : decide p = false → ¬p :=
  (decide_false_iff p).1


theorem decide_congr {p q : Prop} [Decidable p] [Decidable q] (h : p ↔ q) : decide p = decide q :=
  decide_eq_decide.mpr h


@[deprecated (since := "2024-06-07")] alias coe_or_iff := or_eq_true_iff


@[deprecated (since := "2024-06-07")] alias coe_and_iff := and_eq_true_iff


theorem coe_xor_iff (a b : Bool) : xor a b ↔ Xor' (a = true) (b = true) := by
  /-
    a b : Bool
    ⊢ Iff (Eq (a.xor b) Bool.true) (Xor' (Eq a Bool.true) (Eq b Bool.true))
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
  cases a <;> cases b <;> decide
                          /-
                            🎉 no goals
                          -/


@[deprecated (since := "2024-06-07")] alias decide_True := decide_true


@[deprecated (since := "2024-06-07")] alias decide_False := decide_false


@[deprecated (since := "2024-06-07")] alias coe_decide := decide_eq_true_iff


@[deprecated decide_eq_true_iff (since := "2024-06-07")]
alias of_decide_iff := decide_eq_true_iff


@[deprecated (since := "2024-06-07")] alias decide_not := decide_not


@[deprecated (since := "2024-06-07")] alias not_false' := false_ne_true


@[deprecated (since := "2024-06-07")] alias eq_iff_eq_true_iff := eq_iff_iff


                                                          /-
                                                            b : Bool
                                                            ⊢ Or (Eq b Bool.false) (Eq b Bool.true)
                                                          -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
theorem dichotomy (b : Bool) : b = false ∨ b = true := by cases b <;> simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem not_ne_id : not ≠ id := fun h ↦ false_ne_true <| congrFun h true


@[deprecated (since := "2024-06-07")] alias eq_true_of_ne_false := eq_true_of_ne_false


@[deprecated (since := "2024-06-07")] alias eq_false_of_ne_true := eq_false_of_ne_true


                                                   /-
                                                     a b : Bool
                                                     H : Eq a Bool.true
                                                     ⊢ Eq (a.or b) Bool.true
                                                   -/
theorem or_inl {a b : Bool} (H : a) : a || b := by simp [H]
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                   /-
                                                     a b : Bool
                                                     H : Eq b Bool.true
                                                     ⊢ Eq (a.or b) Bool.true
                                                   -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
theorem or_inr {a b : Bool} (H : b) : a || b := by cases a <;> simp [H]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                         /-
                                                           ⊢ ∀ {a b : Bool}, Eq (a.and b) Bool.true → Eq a Bool.true
                                                         -/
theorem and_elim_left : ∀ {a b : Bool}, a && b → a := by decide
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                         /-
                                                           ⊢ ∀ {a b : Bool}, Eq a Bool.true → Eq b Bool.true → Eq (a.and b) Bool.true
                                                         -/
theorem and_intro : ∀ {a b : Bool}, a → b → a && b := by decide
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                          /-
                                                            ⊢ ∀ {a b : Bool}, Eq (a.and b) Bool.true → Eq b Bool.true
                                                          -/
theorem and_elim_right : ∀ {a b : Bool}, a && b → b := by decide
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                        /-
                                                          ⊢ ∀ {a b : Bool}, Iff (Eq a b.not) (Ne a b)
                                                        -/
lemma eq_not_iff : ∀ {a b : Bool}, a = !b ↔ a ≠ b := by decide
                                                        /-
                                                          🎉 no goals
                                                        -/


                                                        /-
                                                          ⊢ ∀ {a b : Bool}, Iff (Eq (Decidable.decide (Eq a b)).not Bool.true) (Ne a b)
                                                        -/
lemma not_eq_iff : ∀ {a b : Bool}, !a = b ↔ a ≠ b := by decide
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem ne_not {a b : Bool} : a ≠ !b ↔ a = b :=
  not_eq_not


@[deprecated (since := "2024-06-07")] alias not_ne := not_not_eq


                                               /-
                                                 ⊢ ∀ (b : Bool), Ne b.not b
                                               -/
lemma not_ne_self : ∀ b : Bool, (!b) ≠ b := by decide
                                               /-
                                                 🎉 no goals
                                               -/


                                             /-
                                               ⊢ ∀ (b : Bool), Ne b b.not
                                             -/
lemma self_ne_not : ∀ b : Bool, b ≠ !b := by decide
                                             /-
                                               🎉 no goals
                                             -/


                                                 /-
                                                   ⊢ ∀ (a b : Bool), Or (Eq a b) (Eq a b.not)
                                                 -/
lemma eq_or_eq_not : ∀ a b, a = b ∨ a = !b := by decide
                                                 /-
                                                   🎉 no goals
                                                 -/

-- Porting note: naming issue again: these two `not` are different.

                                                  /-
                                                    ⊢ ∀ {b : Bool}, Iff (Eq b.not Bool.true) (Not (Eq b Bool.true))
                                                  -/
theorem not_iff_not : ∀ {b : Bool}, !b ↔ ¬b := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem eq_true_of_not_eq_false' {a : Bool} : !a = false → a = true := by
  /-
    a : Bool
    ⊢ Eq (Decidable.decide (Eq a Bool.false)).not Bool.true → Eq a Bool.true
  -/
              /-
                🎉 no goals
              -/
  cases a <;> decide
              /-
                🎉 no goals
              -/


theorem eq_false_of_not_eq_true' {a : Bool} : !a = true → a = false := by
  /-
    a : Bool
    ⊢ Eq (Decidable.decide (Eq a Bool.true)).not Bool.true → Eq a Bool.false
  -/
              /-
                🎉 no goals
              -/
  cases a <;> decide
              /-
                🎉 no goals
              -/


                                     /-
                                       ⊢ Eq bne Bool.xor
                                     -/
theorem bne_eq_xor : bne = xor := by funext a b; revert a b; decide
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                                  /-
                                                                    ⊢ ∀ {x y : Bool}, Iff (Eq (x.xor y) Bool.true) (Ne x y)
                                                                  -/
theorem xor_iff_ne : ∀ {x y : Bool}, xor x y = true ↔ x ≠ y := by decide
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance linearOrder : LinearOrder Bool where
                /-
                  ⊢ ∀ (a : Bool), LE.le a a
                -/
  le_refl := by decide
                /-
                  🎉 no goals
                -/
                 /-
                   ⊢ ∀ (a b c : Bool), LE.le a b → LE.le b c → LE.le a c
                 -/
  le_trans := by decide
                 /-
                   🎉 no goals
                 -/
                    /-
                      ⊢ ∀ (a b : Bool), LE.le a b → LE.le b a → Eq a b
                    -/
  le_antisymm := by decide
                    /-
                      🎉 no goals
                    -/
                 /-
                   ⊢ ∀ (a b : Bool), Or (LE.le a b) (LE.le b a)
                 -/
  le_total := by decide
                         /-
                           ⊢ ∀ (a b : Bool), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
                         -/
                 /-
                   🎉 no goals
                 -/
                         /-
                           🎉 no goals
                         -/
  decidableLE := inferInstance
  decidableEq := inferInstance
  decidableLT := inferInstance
  lt_iff_le_not_le := by decide
                /-
                  ⊢ ∀ (a b : Bool), Eq (Max.max a b) (ite (LE.le a b) b a)
                -/
                /-
                  ⊢ ∀ (a b : Bool), Eq (Min.min a b) (ite (LE.le a b) a b)
                -/
  max_def := by decide
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  min_def := by decide


                                                                    /-
                                                                      ⊢ ∀ {x y : Bool}, Iff (LT.lt x y) (And (Eq x Bool.false) (Eq y Bool.true))
                                                                    -/
theorem lt_iff : ∀ {x y : Bool}, x < y ↔ x = false ∧ y = true := by decide
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem false_lt_true : false < true :=
  lt_iff.2 ⟨rfl, rfl⟩


                                                         /-
                                                           ⊢ ∀ {x y : Bool}, Iff (LE.le x y) (Eq x Bool.true → Eq y Bool.true)
                                                         -/
theorem le_iff_imp : ∀ {x y : Bool}, x ≤ y ↔ x → y := by decide
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                       /-
                                                         ⊢ ∀ (x y : Bool), LE.le (x.and y) x
                                                       -/
theorem and_le_left : ∀ x y : Bool, (x && y) ≤ x := by decide
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                        /-
                                                          ⊢ ∀ (x y : Bool), LE.le (x.and y) y
                                                        -/
theorem and_le_right : ∀ x y : Bool, (x && y) ≤ y := by decide
                                                        /-
                                                          🎉 no goals
                                                        -/


                                                                      /-
                                                                        ⊢ ∀ {x y z : Bool}, LE.le x y → LE.le x z → LE.le x (y.and z)
                                                                      -/
theorem le_and : ∀ {x y z : Bool}, x ≤ y → x ≤ z → x ≤ (y && z) := by decide
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                                      /-
                                                        ⊢ ∀ (x y : Bool), LE.le x (x.or y)
                                                      -/
theorem left_le_or : ∀ x y : Bool, x ≤ (x || y) := by decide
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                       /-
                                                         ⊢ ∀ (x y : Bool), LE.le y (x.or y)
                                                       -/
theorem right_le_or : ∀ x y : Bool, y ≤ (x || y) := by decide
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                              /-
                                                                ⊢ ∀ {x y z : Bool}, LE.le x z → LE.le y z → LE.le (x.or y) z
                                                              -/
theorem or_le : ∀ {x y z}, x ≤ z → y ≤ z → (x || y) ≤ z := by decide
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- convert a `ℕ` to a `Bool`, `0 -> false`, everything else -> `true` -/
def ofNat (n : Nat) : Bool :=
  decide (n ≠ 0)


                                                                    /-
                                                                      b : Bool
                                                                      ⊢ Eq (BEq.beq b.toNat 0) b.not
                                                                    -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
@[simp] lemma toNat_beq_zero (b : Bool) : (b.toNat == 0) = !b := by cases b <;> rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                    /-
                                                                      b : Bool
                                                                      ⊢ Eq (bne b.toNat 0) b
                                                                    -/
@[simp] lemma toNat_bne_zero (b : Bool) : (b.toNat != 0) =  b := by simp [bne]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

                                                                   /-
                                                                     b : Bool
                                                                     ⊢ Eq (BEq.beq b.toNat 1) b
                                                                   -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
@[simp] lemma toNat_beq_one (b : Bool) : (b.toNat == 1) =  b := by cases b <;> rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

                                                                   /-
                                                                     b : Bool
                                                                     ⊢ Eq (bne b.toNat 1) b.not
                                                                   -/
@[simp] lemma toNat_bne_one (b : Bool) : (b.toNat != 1) = !b := by simp [bne]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem ofNat_le_ofNat {n m : Nat} (h : n ≤ m) : ofNat n ≤ ofNat m := by
  /-
    n m : Nat
    h : LE.le n m
    ⊢ LE.le (Bool.ofNat n) (Bool.ofNat m)
  -/
  simp only [ofNat, ne_eq, _root_.decide_not]
  cases Nat.decEq n 0 with
  | isTrue hn => rw [_root_.decide_eq_true hn]; exact Bool.false_le _
  | isFalse hn =>
    cases Nat.decEq m 0 with
    | isFalse hm => rw [_root_.decide_eq_false hm]; exact Bool.le_true _
    | isTrue hm => subst hm; have h := Nat.le_antisymm h (Nat.zero_le n); contradiction


theorem toNat_le_toNat {b₀ b₁ : Bool} (h : b₀ ≤ b₁) : toNat b₀ ≤ toNat b₁ := by
  /-
    b₀ b₁ : Bool
    h : LE.le b₀ b₁
    ⊢ LE.le b₀.toNat b₁.toNat
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
  cases b₀ <;> cases b₁ <;> simp_all +decide
                            /-
                              🎉 no goals
                            -/


theorem ofNat_toNat (b : Bool) : ofNat (toNat b) = b := by
  /-
    b : Bool
    ⊢ Eq (Bool.ofNat b.toNat) b
  -/
              /-
                🎉 no goals
              -/
  cases b <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem injective_iff {α : Sort*} {f : Bool → α} : Function.Injective f ↔ f false ≠ f true :=
  ⟨fun Hinj Heq ↦ false_ne_true (Hinj Heq), fun H x y hxy ↦ by
    /-
      α : Sort u_1
      f : Bool → α
      H : Ne (f Bool.false) (f Bool.true)
      x y : Bool
      hxy : Eq (f x) (f y)
      ⊢ Eq x y
    -/
    cases x <;> cases y
      /-
        case false.false
        α : Sort u_1
        f : Bool → α
        H : Ne (f Bool.false) (f Bool.true)
        hxy : Eq (f Bool.false) (f Bool.false)
        ⊢ Eq Bool.false Bool.false
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case false.true
        α : Sort u_1
        f : Bool → α
        H : Ne (f Bool.false) (f Bool.true)
        hxy : Eq (f Bool.false) (f Bool.true)
        ⊢ Eq Bool.false Bool.true
      -/
    · exact (H hxy).elim
      /-
        🎉 no goals
      -/
      /-
        case true.false
        α : Sort u_1
        f : Bool → α
        H : Ne (f Bool.false) (f Bool.true)
        hxy : Eq (f Bool.true) (f Bool.false)
        ⊢ Eq Bool.true Bool.false
      -/
    · exact (H hxy.symm).elim
      /-
        🎉 no goals
      -/
      /-
        case true.true
        α : Sort u_1
        f : Bool → α
        H : Ne (f Bool.false) (f Bool.true)
        hxy : Eq (f Bool.true) (f Bool.true)
        ⊢ Eq Bool.true Bool.true
      -/
    · rfl⟩
      /-
        🎉 no goals
      -/


/-- **Kaminski's Equation** -/
theorem apply_apply_apply (f : Bool → Bool) (x : Bool) : f (f (f x)) = f x := by
  /-
    f : Bool → Bool
    x : Bool
    ⊢ Eq (f (f (f x))) (f x)
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
                                                           /-
                                                             🎉 no goals
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
  cases x <;> cases h₁ : f true <;> cases h₂ : f false <;> simp only [h₁, h₂]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- `xor3 x y c` is `((x XOR y) XOR c)`. -/
protected def xor3 (x y c : Bool) :=
  xor (xor x y) c


/-- `carry x y c` is `x && y || x && c || y && c`. -/
protected def carry (x y c : Bool) :=
  x && y || x && c || y && c


