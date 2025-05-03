/-- The type `ℝ` of real numbers constructed as equivalence classes of Cauchy sequences of rational
numbers. -/
structure Real where ofCauchy ::
  /-- The underlying Cauchy completion -/
  cauchy : CauSeq.Completion.Cauchy (abs : ℚ → ℚ)


@[inherit_doc]
notation "ℝ" => Real

-- Porting note: unknown attribute
-- attribute [pp_using_anonymous_constructor] Real


@[simp]
theorem ofRat_rat {abv : ℚ → ℚ} [IsAbsoluteValue abv] (q : ℚ) :
    ofRat (q : ℚ) = (q : Cauchy abv) :=
  rfl


theorem ext_cauchy_iff : ∀ {x y : Real}, x = y ↔ x.cauchy = y.cauchy
                   /-
                     a b : CauSeq.Completion.Cauchy abs
                     ⊢ Iff (Eq { cauchy := a } { cauchy := b }) (Eq { cauchy := a }.cauchy { cauchy …
                   -/
  | ⟨a⟩, ⟨b⟩ => by rw [ofCauchy.injEq]
                   /-
                     🎉 no goals
                   -/


theorem ext_cauchy {x y : Real} : x.cauchy = y.cauchy → x = y :=
  ext_cauchy_iff.2


/-- The real numbers are isomorphic to the quotient of Cauchy sequences on the rationals. -/
def equivCauchy : ℝ ≃ CauSeq.Completion.Cauchy (abs : ℚ → ℚ) :=
  ⟨Real.cauchy, Real.ofCauchy, fun ⟨_⟩ => rfl, fun _ => rfl⟩

-- irreducible doesn't work for instances: https://github.com/leanprover-community/lean/issues/511

private irreducible_def zero : ℝ :=
  ⟨0⟩


private irreducible_def one : ℝ :=
  ⟨1⟩


private irreducible_def add : ℝ → ℝ → ℝ
  | ⟨a⟩, ⟨b⟩ => ⟨a + b⟩


private irreducible_def neg : ℝ → ℝ
  | ⟨a⟩ => ⟨-a⟩


private irreducible_def mul : ℝ → ℝ → ℝ
  | ⟨a⟩, ⟨b⟩ => ⟨a * b⟩


private noncomputable irreducible_def inv' : ℝ → ℝ
  | ⟨a⟩ => ⟨a⁻¹⟩


instance : Zero ℝ :=
  ⟨zero⟩


instance : One ℝ :=
  ⟨one⟩


instance : Add ℝ :=
  ⟨add⟩


instance : Neg ℝ :=
  ⟨neg⟩


instance : Mul ℝ :=
  ⟨mul⟩


instance : Sub ℝ :=
  ⟨fun a b => a + -b⟩


noncomputable instance : Inv ℝ :=
  ⟨inv'⟩


theorem ofCauchy_zero : (⟨0⟩ : ℝ) = 0 :=
  zero_def.symm


theorem ofCauchy_one : (⟨1⟩ : ℝ) = 1 :=
  one_def.symm


theorem ofCauchy_add (a b) : (⟨a + b⟩ : ℝ) = ⟨a⟩ + ⟨b⟩ :=
  (add_def _ _).symm


theorem ofCauchy_neg (a) : (⟨-a⟩ : ℝ) = -⟨a⟩ :=
  (neg_def _).symm


theorem ofCauchy_sub (a b) : (⟨a - b⟩ : ℝ) = ⟨a⟩ - ⟨b⟩ := by
  /-
    a b : CauSeq.Completion.Cauchy abs
    ⊢ Eq { cauchy := HSub.hSub a b } (HSub.hSub { cauchy := a } { cauchy := b })
  -/
  rw [sub_eq_add_neg, ofCauchy_add, ofCauchy_neg]
  /-
    a b : CauSeq.Completion.Cauchy abs
    ⊢ Eq (HAdd.hAdd { cauchy := a } (Neg.neg { cauchy := b })) (HSub.hSub { cauchy …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ofCauchy_mul (a b) : (⟨a * b⟩ : ℝ) = ⟨a⟩ * ⟨b⟩ :=
  (mul_def _ _).symm


theorem ofCauchy_inv {f} : (⟨f⁻¹⟩ : ℝ) = ⟨f⟩⁻¹ :=
                     /-
                       f : CauSeq.Completion.Cauchy abs
                       ⊢ Eq { cauchy := Inv.inv f } (Real.inv' { cauchy := f })
                     -/
  show _ = inv' _ by rw [inv']
                     /-
                       🎉 no goals
                     -/


theorem cauchy_zero : (0 : ℝ).cauchy = 0 :=
                          /-
                            ⊢ Eq Real.zero.cauchy 0
                          -/
  show zero.cauchy = 0 by rw [zero_def]
                          /-
                            🎉 no goals
                          -/


theorem cauchy_one : (1 : ℝ).cauchy = 1 :=
                         /-
                           ⊢ Eq Real.one.cauchy 1
                         -/
  show one.cauchy = 1 by rw [one_def]
                         /-
                           🎉 no goals
                         -/


theorem cauchy_add : ∀ a b, (a + b : ℝ).cauchy = a.cauchy + b.cauchy
                                             /-
                                               a b : CauSeq.Completion.Cauchy abs
                                               ⊢ Eq (Real.add { cauchy := a } { cauchy := b }).cauchy (HAdd.hAdd { cauchy :=  …
                                             -/
  | ⟨a⟩, ⟨b⟩ => show (add _ _).cauchy = _ by rw [add_def]
                                             /-
                                               🎉 no goals
                                             -/


theorem cauchy_neg : ∀ a, (-a : ℝ).cauchy = -a.cauchy
                                      /-
                                        a : CauSeq.Completion.Cauchy abs
                                        ⊢ Eq (Real.neg { cauchy := a }).cauchy (Neg.neg { cauchy := a }.cauchy)
                                      -/
  | ⟨a⟩ => show (neg _).cauchy = _ by rw [neg_def]
                                      /-
                                        🎉 no goals
                                      -/


theorem cauchy_mul : ∀ a b, (a * b : ℝ).cauchy = a.cauchy * b.cauchy
                                             /-
                                               a b : CauSeq.Completion.Cauchy abs
                                               ⊢ Eq (Real.mul { cauchy := a } { cauchy := b }).cauchy (HMul.hMul { cauchy :=  …
                                             -/
  | ⟨a⟩, ⟨b⟩ => show (mul _ _).cauchy = _ by rw [mul_def]
                                             /-
                                               🎉 no goals
                                             -/


theorem cauchy_sub : ∀ a b, (a - b : ℝ).cauchy = a.cauchy - b.cauchy
  | ⟨a⟩, ⟨b⟩ => by
    /-
      a b : CauSeq.Completion.Cauchy abs
      ⊢ Eq (HSub.hSub { cauchy := a } { cauchy := b }).cauchy (HSub.hSub { cauchy := …
    -/
    rw [sub_eq_add_neg, ← cauchy_neg, ← cauchy_add]
    /-
      a b : CauSeq.Completion.Cauchy abs
      ⊢ Eq (HSub.hSub { cauchy := a } { cauchy := b }).cauchy (HAdd.hAdd { cauchy := …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem cauchy_inv : ∀ f, (f⁻¹ : ℝ).cauchy = f.cauchy⁻¹
                                       /-
                                         f : CauSeq.Completion.Cauchy abs
                                         ⊢ Eq (Real.inv' { cauchy := f }).cauchy (Inv.inv { cauchy := f }.cauchy)
                                       -/
  | ⟨f⟩ => show (inv' _).cauchy = _ by rw [inv']
                                       /-
                                         🎉 no goals
                                       -/


instance instNatCast : NatCast ℝ where natCast n := ⟨n⟩

instance instIntCast : IntCast ℝ where intCast z := ⟨z⟩

instance instNNRatCast : NNRatCast ℝ where nnratCast q := ⟨q⟩

instance instRatCast : RatCast ℝ where ratCast q := ⟨q⟩


lemma ofCauchy_natCast (n : ℕ) : (⟨n⟩ : ℝ) = n := rfl

lemma ofCauchy_intCast (z : ℤ) : (⟨z⟩ : ℝ) = z := rfl

lemma ofCauchy_nnratCast (q : ℚ≥0) : (⟨q⟩ : ℝ) = q := rfl

lemma ofCauchy_ratCast (q : ℚ) : (⟨q⟩ : ℝ) = q := rfl


lemma cauchy_natCast (n : ℕ) : (n : ℝ).cauchy = n := rfl

lemma cauchy_intCast (z : ℤ) : (z : ℝ).cauchy = z := rfl

lemma cauchy_nnratCast (q : ℚ≥0) : (q : ℝ).cauchy = q := rfl

lemma cauchy_ratCast (q : ℚ) : (q : ℝ).cauchy = q := rfl


instance commRing : CommRing ℝ where
  natCast n := ⟨n⟩
  intCast z := ⟨z⟩
  zero := (0 : ℝ)
  one := (1 : ℝ)
  mul := (· * ·)
  add := (· + ·)
  neg := @Neg.neg ℝ _
  sub := @Sub.sub ℝ _
  npow := @npowRec ℝ ⟨1⟩ ⟨(· * ·)⟩
  nsmul := @nsmulRec ℝ ⟨0⟩ ⟨(· + ·)⟩
  zsmul := @zsmulRec ℝ ⟨0⟩ ⟨(· + ·)⟩ ⟨@Neg.neg ℝ _⟩ (@nsmulRec ℝ ⟨0⟩ ⟨(· + ·)⟩)
                   /-
                     x a : Real
                     ⊢ Eq (HAdd.hAdd a 0) a
                   -/
                   /-
                     x a : Real
                     ⊢ Eq (HAdd.hAdd 0 a) a
                   -/
  add_zero a := by apply ext_cauchy; simp [cauchy_add, cauchy_zero]
                        /-
                          x a b c : Real
                          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd b c))
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
  zero_add a := by apply ext_cauchy; simp [cauchy_add, cauchy_zero]
                     /-
                       x a b : Real
                       ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                     -/
  add_comm a b := by apply ext_cauchy; simp only [cauchy_add, add_comm]
                                       /-
                                         🎉 no goals
                                       -/
  add_assoc a b c := by apply ext_cauchy; simp only [cauchy_add, add_assoc]
                   /-
                     x a : Real
                     ⊢ Eq (HMul.hMul a 0) 0
                   -/
                   /-
                     x a : Real
                     ⊢ Eq (HMul.hMul 0 a) 0
                   -/
  mul_zero a := by apply ext_cauchy; simp [cauchy_mul, cauchy_zero]
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  zero_mul a := by apply ext_cauchy; simp [cauchy_mul, cauchy_zero]
                           /-
                             x a b c : Real
                             ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
                           -/
                  /-
                    x a : Real
                    ⊢ Eq (HMul.hMul a 1) a
                  -/
                                             /-
                                               🎉 no goals
                                             -/
                            /-
                              x a b c : Real
                              ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
                            -/
                  /-
                    x a : Real
                    ⊢ Eq (HMul.hMul 1 a) a
                  -/
                                              /-
                                                🎉 no goals
                                              -/
  mul_one a := by apply ext_cauchy; simp [cauchy_mul, cauchy_one]
                        /-
                          x a b c : Real
                          ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
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
  one_mul a := by apply ext_cauchy; simp [cauchy_mul, cauchy_one]
                     /-
                       x a b : Real
                       ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
                     -/
  mul_comm a b := by apply ext_cauchy; simp only [cauchy_mul, mul_comm]
                                       /-
                                         🎉 no goals
                                       -/
  mul_assoc a b c := by apply ext_cauchy; simp only [cauchy_mul, mul_assoc]
                         /-
                           x a : Real
                           ⊢ Eq (HAdd.hAdd (Neg.neg a) a) 0
                         -/
                     /-
                       x : Real
                       ⊢ Eq (NatCast.natCast 0) 0
                     -/
  left_distrib a b c := by apply ext_cauchy; simp only [cauchy_add, cauchy_mul, mul_add]
                                       /-
                                         🎉 no goals
                                       -/
                       /-
                         x : Real
                         n : Nat
                         ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
                       -/
                                           /-
                                             🎉 no goals
                                           -/
                                         /-
                                           🎉 no goals
                                         -/
  right_distrib a b c := by apply ext_cauchy; simp only [cauchy_add, cauchy_mul, add_mul]
  neg_add_cancel a := by apply ext_cauchy; simp [cauchy_add, cauchy_neg, cauchy_zero]
                          /-
                            x : Real
                            z : Nat
                            ⊢ Eq (IntCast.intCast (Int.negSucc z)) (Neg.neg ↑(HAdd.hAdd z 1))
                          -/
  natCast_zero := by apply ext_cauchy; simp [cauchy_zero]
                                            /-
                                              🎉 no goals
                                            -/
  natCast_succ n := by apply ext_cauchy; simp [cauchy_one, cauchy_add]
  intCast_negSucc z := by apply ext_cauchy; simp [cauchy_neg, cauchy_natCast]


/-- `Real.equivCauchy` as a ring equivalence. -/
@[simps]
def ringEquivCauchy : ℝ ≃+* CauSeq.Completion.Cauchy (abs : ℚ → ℚ) :=
  { equivCauchy with
    toFun := cauchy
    invFun := ofCauchy
    map_add' := cauchy_add
    map_mul' := cauchy_mul }


                                 /-
                                   x : Real
                                   ⊢ Ring Real
                                 -/
instance instRing : Ring ℝ := by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


                                /-
                                  x : Real
                                  ⊢ CommSemiring Real
                                -/
instance : CommSemiring ℝ := by infer_instance
                                /-
                                  🎉 no goals
                                -/


                                     /-
                                       x : Real
                                       ⊢ Semiring Real
                                     -/
instance semiring : Semiring ℝ := by infer_instance
                                     /-
                                       🎉 no goals
                                     -/


                                      /-
                                        x : Real
                                        ⊢ CommMonoidWithZero Real
                                      -/
instance : CommMonoidWithZero ℝ := by infer_instance
                                      /-
                                        🎉 no goals
                                      -/


                                  /-
                                    x : Real
                                    ⊢ MonoidWithZero Real
                                  -/
instance : MonoidWithZero ℝ := by infer_instance
                                  /-
                                    🎉 no goals
                                  -/


                                /-
                                  x : Real
                                  ⊢ AddCommGroup Real
                                -/
instance : AddCommGroup ℝ := by infer_instance
                                /-
                                  🎉 no goals
                                -/


                            /-
                              x : Real
                              ⊢ AddGroup Real
                            -/
instance : AddGroup ℝ := by infer_instance
                            /-
                              🎉 no goals
                            -/


                                 /-
                                   x : Real
                                   ⊢ AddCommMonoid Real
                                 -/
instance : AddCommMonoid ℝ := by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


                             /-
                               x : Real
                               ⊢ AddMonoid Real
                             -/
instance : AddMonoid ℝ := by infer_instance
                             /-
                               🎉 no goals
                             -/


                                          /-
                                            x : Real
                                            ⊢ AddLeftCancelSemigroup Real
                                          -/
instance : AddLeftCancelSemigroup ℝ := by infer_instance
                                          /-
                                            🎉 no goals
                                          -/


                                           /-
                                             x : Real
                                             ⊢ AddRightCancelSemigroup Real
                                           -/
instance : AddRightCancelSemigroup ℝ := by infer_instance
                                           /-
                                             🎉 no goals
                                           -/


                                    /-
                                      x : Real
                                      ⊢ AddCommSemigroup Real
                                    -/
instance : AddCommSemigroup ℝ := by infer_instance
                                    /-
                                      🎉 no goals
                                    -/


                                /-
                                  x : Real
                                  ⊢ AddSemigroup Real
                                -/
instance : AddSemigroup ℝ := by infer_instance
                                /-
                                  🎉 no goals
                                -/


                              /-
                                x : Real
                                ⊢ CommMonoid Real
                              -/
instance : CommMonoid ℝ := by infer_instance
                              /-
                                🎉 no goals
                              -/


                          /-
                            x : Real
                            ⊢ Monoid Real
                          -/
instance : Monoid ℝ := by infer_instance
                          /-
                            🎉 no goals
                          -/


                                 /-
                                   x : Real
                                   ⊢ CommSemigroup Real
                                 -/
instance : CommSemigroup ℝ := by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


                             /-
                               x : Real
                               ⊢ Semigroup Real
                             -/
instance : Semigroup ℝ := by infer_instance
                             /-
                               🎉 no goals
                             -/


instance : Inhabited ℝ :=
  ⟨0⟩


/-- Make a real number from a Cauchy sequence of rationals (by taking the equivalence class). -/
def mk (x : CauSeq ℚ abs) : ℝ :=
  ⟨CauSeq.Completion.mk x⟩


theorem mk_eq {f g : CauSeq ℚ abs} : mk f = mk g ↔ f ≈ g :=
  ext_cauchy_iff.trans CauSeq.Completion.mk_eq


private irreducible_def lt : ℝ → ℝ → Prop
  | ⟨x⟩, ⟨y⟩ =>
    (Quotient.liftOn₂ x y (· < ·)) fun _ _ _ _ hf hg =>
      propext <|
        ⟨fun h => lt_of_eq_of_lt (Setoid.symm hf) (lt_of_lt_of_eq h hg), fun h =>
          lt_of_eq_of_lt hf (lt_of_lt_of_eq h (Setoid.symm hg))⟩


instance : LT ℝ :=
  ⟨lt⟩


theorem lt_cauchy {f g} : (⟨⟦f⟧⟩ : ℝ) < ⟨⟦g⟧⟩ ↔ f < g :=
                     /-
                       f g : CauSeq Rat abs
                       ⊢ Iff (Real.lt { cauchy := Quotient.mk CauSeq.equiv f } { cauchy := Quotient.m …
                     -/
  show lt _ _ ↔ _ by rw [lt_def]; rfl
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem mk_lt {f g : CauSeq ℚ abs} : mk f < mk g ↔ f < g :=
  lt_cauchy


                                 /-
                                   ⊢ Eq (Real.mk 0) 0
                                 -/
theorem mk_zero : mk 0 = 0 := by rw [← ofCauchy_zero]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


                                /-
                                  ⊢ Eq (Real.mk 1) 1
                                -/
theorem mk_one : mk 1 = 1 := by rw [← ofCauchy_one]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


                                                                     /-
                                                                       f g : CauSeq Rat abs
                                                                       ⊢ Eq (Real.mk (HAdd.hAdd f g)) (HAdd.hAdd (Real.mk f) (Real.mk g))
                                                                     -/
theorem mk_add {f g : CauSeq ℚ abs} : mk (f + g) = mk f + mk g := by simp [mk, ← ofCauchy_add]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                     /-
                                                                       f g : CauSeq Rat abs
                                                                       ⊢ Eq (Real.mk (HMul.hMul f g)) (HMul.hMul (Real.mk f) (Real.mk g))
                                                                     -/
theorem mk_mul {f g : CauSeq ℚ abs} : mk (f * g) = mk f * mk g := by simp [mk, ← ofCauchy_mul]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                          /-
                                                            f : CauSeq Rat abs
                                                            ⊢ Eq (Real.mk (Neg.neg f)) (Neg.neg (Real.mk f))
                                                          -/
theorem mk_neg {f : CauSeq ℚ abs} : mk (-f) = -mk f := by simp [mk, ← ofCauchy_neg]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem mk_pos {f : CauSeq ℚ abs} : 0 < mk f ↔ Pos f := by
  /-
    f : CauSeq Rat abs
    ⊢ Iff (LT.lt 0 (Real.mk f)) f.Pos
  -/
  rw [← mk_zero, mk_lt]
  /-
    f : CauSeq Rat abs
    ⊢ Iff (LT.lt 0 f) f.Pos
  -/
  exact iff_of_eq (congr_arg Pos (sub_zero f))
  /-
    🎉 no goals
  -/


private irreducible_def le (x y : ℝ) : Prop :=
  x < y ∨ x = y


instance : LE ℝ :=
  ⟨le⟩


private theorem le_def' {x y : ℝ} : x ≤ y ↔ x < y ∨ x = y :=
  iff_of_eq <| le_def _ _


@[simp]
theorem mk_le {f g : CauSeq ℚ abs} : mk f ≤ mk g ↔ f ≤ g := by
  /-
    f g : CauSeq Rat abs
    ⊢ Iff (LE.le (Real.mk f) (Real.mk g)) (LE.le f g)
  -/
  simp only [le_def', mk_lt, mk_eq]; rfl
                                     /-
                                       🎉 no goals
                                     -/


@[elab_as_elim]
protected theorem ind_mk {C : Real → Prop} (x : Real) (h : ∀ y, C (mk y)) : C x := by
  /-
    C : Real → Prop
    x : Real
    h : ∀ (y : CauSeq Rat abs), C (Real.mk y)
    ⊢ C x
  -/
  cases' x with x
  /-
    case ofCauchy
    C : Real → Prop
    h : ∀ (y : CauSeq Rat abs), C (Real.mk y)
    x : CauSeq.Completion.Cauchy abs
    ⊢ C { cauchy := x }
  -/
  induction' x using Quot.induction_on with x
  /-
    case ofCauchy.h
    C : Real → Prop
    h : ∀ (y : CauSeq Rat abs), C (Real.mk y)
    x : CauSeq Rat abs
    ⊢ C { cauchy := Quot.mk (⇑CauSeq.equiv) x }
  -/
  exact h x
  /-
    🎉 no goals
  -/


theorem add_lt_add_iff_left {a b : ℝ} (c : ℝ) : c + a < c + b ↔ a < b := by
  /-
    a b c : Real
    ⊢ Iff (LT.lt (HAdd.hAdd c a) (HAdd.hAdd c b)) (LT.lt a b)
  -/
  induction a using Real.ind_mk
  /-
    case h
    b c : Real
    y✝ : CauSeq Rat abs
    ⊢ Iff (LT.lt (HAdd.hAdd c (Real.mk y✝)) (HAdd.hAdd c b)) (LT.lt (Real.mk y✝) b)
  -/
  induction b using Real.ind_mk
  /-
    case h.h
    c : Real
    y✝¹ y✝ : CauSeq Rat abs
    ⊢ Iff (LT.lt (HAdd.hAdd c (Real.mk y✝¹)) (HAdd.hAdd c (Real.mk y✝))) (LT.lt (R …
  -/
  induction c using Real.ind_mk
  /-
    case h.h.h
    y✝² y✝¹ y✝ : CauSeq Rat abs
    ⊢ Iff (LT.lt (HAdd.hAdd (Real.mk y✝) (Real.mk y✝²)) (HAdd.hAdd (Real.mk y✝) (R …
  -/
  simp only [mk_lt, ← mk_add]
  /-
    case h.h.h
    y✝² y✝¹ y✝ : CauSeq Rat abs
    ⊢ Iff (LT.lt (HAdd.hAdd y✝ y✝²) (HAdd.hAdd y✝ y✝¹)) (LT.lt y✝² y✝¹)
  -/
  show Pos _ ↔ Pos _; rw [add_sub_add_left_eq_sub]
                      /-
                        🎉 no goals
                      -/


instance partialOrder : PartialOrder ℝ where
  le := (· ≤ ·)
  lt := (· < ·)
  lt_iff_le_not_le a b := by
    /-
      x a b : Real
      ⊢ Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    induction' a using Real.ind_mk with a
    /-
      case h
      x b : Real
      a : CauSeq Rat abs
      ⊢ Iff (LT.lt (Real.mk a) b) (And (LE.le (Real.mk a) b) (Not (LE.le b (Real.mk  …
    -/
    induction' b using Real.ind_mk with b
    /-
      x a : Real
      ⊢ LE.le a a
    -/
    /-
      case h.h
      x : Real
      a b : CauSeq Rat abs
      ⊢ Iff (LT.lt (Real.mk a) (Real.mk b)) (And (LE.le (Real.mk a) (Real.mk b)) (No …
    -/
    /-
      case h
      x : Real
      a : CauSeq Rat abs
      ⊢ LE.le (Real.mk a) (Real.mk a)
    -/
    simpa using lt_iff_le_not_le
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      x a b c : Real
      ⊢ LE.le a b → LE.le b c → LE.le a c
    -/
  le_refl a := by
    /-
      case h
      x b c : Real
      a : CauSeq Rat abs
      ⊢ LE.le (Real.mk a) b → LE.le b c → LE.le (Real.mk a) c
    -/
    induction' a using Real.ind_mk with a
    /-
      case h.h
      x c : Real
      a b : CauSeq Rat abs
      ⊢ LE.le (Real.mk a) (Real.mk b) → LE.le (Real.mk b) c → LE.le (Real.mk a) c
    -/
    rw [mk_le]
    /-
      case h.h.h
      x : Real
      a b c : CauSeq Rat abs
      ⊢ LE.le (Real.mk a) (Real.mk b) → LE.le (Real.mk b) (Real.mk c) → LE.le (Real. …
    -/
  le_trans a b c := by
    /-
      🎉 no goals
    -/
    induction' a using Real.ind_mk with a
    induction' b using Real.ind_mk with b
    induction' c using Real.ind_mk with c
    simpa using le_trans
  le_antisymm a b := by
    /-
      x a b : Real
      ⊢ LE.le a b → LE.le b a → Eq a b
    -/
    induction' a using Real.ind_mk with a
    /-
      case h
      x b : Real
      a : CauSeq Rat abs
      ⊢ LE.le (Real.mk a) b → LE.le b (Real.mk a) → Eq (Real.mk a) b
    -/
    induction' b using Real.ind_mk with b
    /-
      case h.h
      x : Real
      a b : CauSeq Rat abs
      ⊢ LE.le (Real.mk a) (Real.mk b) → LE.le (Real.mk b) (Real.mk a) → Eq (Real.mk  …
    -/
    simpa [mk_eq] using @CauSeq.le_antisymm _ _ a b
    /-
      🎉 no goals
    -/


                            /-
                              x : Real
                              ⊢ Preorder Real
                            -/
instance : Preorder ℝ := by infer_instance
                            /-
                              🎉 no goals
                            -/


theorem ratCast_lt {x y : ℚ} : (x : ℝ) < (y : ℝ) ↔ x < y := by
  /-
    x y : Rat
    ⊢ Iff (LT.lt ↑x ↑y) (LT.lt x y)
  -/
  erw [mk_lt]
  /-
    x y : Rat
    ⊢ Iff (LT.lt (CauSeq.const abs ↑x) (CauSeq.const abs ↑y)) (LT.lt x y)
  -/
  exact const_lt
  /-
    🎉 no goals
  -/


protected theorem zero_lt_one : (0 : ℝ) < 1 := by
  /-
    ⊢ LT.lt 0 1
  -/
                                       /-
                                         🎉 no goals
                                       -/
  convert ratCast_lt.2 zero_lt_one <;> simp [← ofCauchy_ratCast, ofCauchy_one, ofCauchy_zero]
                                       /-
                                         🎉 no goals
                                       -/


protected theorem fact_zero_lt_one : Fact ((0 : ℝ) < 1) :=
  ⟨Real.zero_lt_one⟩


@[deprecated mul_pos (since := "2024-08-15")]
protected theorem mul_pos {a b : ℝ} : 0 < a → 0 < b → 0 < a * b := by
  /-
    a b : Real
    ⊢ LT.lt 0 a → LT.lt 0 b → LT.lt 0 (HMul.hMul a b)
  -/
  induction' a using Real.ind_mk with a
  /-
    case h
    b : Real
    a : CauSeq Rat abs
    ⊢ LT.lt 0 (Real.mk a) → LT.lt 0 b → LT.lt 0 (HMul.hMul (Real.mk a) b)
  -/
  induction' b using Real.ind_mk with b
  /-
    case h.h
    a b : CauSeq Rat abs
    ⊢ LT.lt 0 (Real.mk a) → LT.lt 0 (Real.mk b) → LT.lt 0 (HMul.hMul (Real.mk a) ( …
  -/
  simpa only [mk_lt, mk_pos, ← mk_mul] using CauSeq.mul_pos
  /-
    🎉 no goals
  -/


instance instStrictOrderedCommRing : StrictOrderedCommRing ℝ where
  __ := Real.commRing
  exists_pair_ne := ⟨0, 1, Real.zero_lt_one.ne⟩
  add_le_add_left := by
    /-
      x : Real
      ⊢ ∀ (a b : Real), LE.le a b → ∀ (c : Real), LE.le (HAdd.hAdd c a) (HAdd.hAdd c …
    -/
    simp only [le_iff_eq_or_lt]
    /-
      x : Real
      ⊢ ∀ (a b : Real), Or (Eq a b) (LT.lt a b) → ∀ (c : Real), Or (Eq (HAdd.hAdd c  …
    -/
    rintro a b ⟨rfl, h⟩
      /-
        case inl.refl
        x a : Real
        ⊢ ∀ (c : Real), Or (Eq (HAdd.hAdd c a) (HAdd.hAdd c a)) (LT.lt (HAdd.hAdd c a) …
      -/
    · simp only [lt_self_iff_false, or_false, forall_const]
      /-
        🎉 no goals
      -/
      /-
        case inr
        x a b : Real
        h✝ : LT.lt a b
        ⊢ ∀ (c : Real), Or (Eq (HAdd.hAdd c a) (HAdd.hAdd c b)) (LT.lt (HAdd.hAdd c a) …
      -/
    · exact fun c => Or.inr ((add_lt_add_iff_left c).2 ‹_›)
      /-
        🎉 no goals
      -/
  zero_le_one := le_of_lt Real.zero_lt_one
  mul_pos a b :=  by
    /-
      x a b : Real
      ⊢ LT.lt 0 a → LT.lt 0 b → LT.lt 0 (HMul.hMul a b)
    -/
    induction' a using Real.ind_mk with a
    /-
      case h
      x b : Real
      a : CauSeq Rat abs
      ⊢ LT.lt 0 (Real.mk a) → LT.lt 0 b → LT.lt 0 (HMul.hMul (Real.mk a) b)
    -/
    induction' b using Real.ind_mk with b
    /-
      case h.h
      x : Real
      a b : CauSeq Rat abs
      ⊢ LT.lt 0 (Real.mk a) → LT.lt 0 (Real.mk b) → LT.lt 0 (HMul.hMul (Real.mk a) ( …
    -/
    simpa only [mk_lt, mk_pos, ← mk_mul] using CauSeq.mul_pos
    /-
      🎉 no goals
    -/


instance strictOrderedRing : StrictOrderedRing ℝ :=
  inferInstance


instance strictOrderedCommSemiring : StrictOrderedCommSemiring ℝ :=
  inferInstance


instance strictOrderedSemiring : StrictOrderedSemiring ℝ :=
  inferInstance


instance orderedRing : OrderedRing ℝ :=
  inferInstance


instance orderedSemiring : OrderedSemiring ℝ :=
  inferInstance


instance orderedAddCommGroup : OrderedAddCommGroup ℝ :=
  inferInstance


instance orderedCancelAddCommMonoid : OrderedCancelAddCommMonoid ℝ :=
  inferInstance


instance orderedAddCommMonoid : OrderedAddCommMonoid ℝ :=
  inferInstance


instance nontrivial : Nontrivial ℝ :=
  inferInstance


private irreducible_def sup : ℝ → ℝ → ℝ
  | ⟨x⟩, ⟨y⟩ => ⟨Quotient.map₂ (· ⊔ ·) (fun _ _ hx _ _ hy => sup_equiv_sup hx hy) x y⟩


instance : Max ℝ :=
  ⟨sup⟩


theorem ofCauchy_sup (a b) : (⟨⟦a ⊔ b⟧⟩ : ℝ) = ⟨⟦a⟧⟩ ⊔ ⟨⟦b⟧⟩ :=
  show _ = sup _ _ by
    /-
      a b : CauSeq Rat abs
      ⊢ Eq { cauchy := Quotient.mk CauSeq.equiv (Max.max a b) } (Real.sup { cauchy : …
    -/
    rw [sup_def]
    /-
      a b : CauSeq Rat abs
      ⊢ Eq { cauchy := Quotient.mk CauSeq.equiv (Max.max a b) } (Real.definition.mat …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem mk_sup (a b) : (mk (a ⊔ b) : ℝ) = mk a ⊔ mk b :=
  ofCauchy_sup _ _


private irreducible_def inf : ℝ → ℝ → ℝ
  | ⟨x⟩, ⟨y⟩ => ⟨Quotient.map₂ (· ⊓ ·) (fun _ _ hx _ _ hy => inf_equiv_inf hx hy) x y⟩


instance : Min ℝ :=
  ⟨inf⟩


theorem ofCauchy_inf (a b) : (⟨⟦a ⊓ b⟧⟩ : ℝ) = ⟨⟦a⟧⟩ ⊓ ⟨⟦b⟧⟩ :=
  show _ = inf _ _ by
    /-
      a b : CauSeq Rat abs
      ⊢ Eq { cauchy := Quotient.mk CauSeq.equiv (Min.min a b) } (Real.inf { cauchy : …
    -/
    rw [inf_def]
    /-
      a b : CauSeq Rat abs
      ⊢ Eq { cauchy := Quotient.mk CauSeq.equiv (Min.min a b) } (Real.definition.mat …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem mk_inf (a b) : (mk (a ⊓ b) : ℝ) = mk a ⊓ mk b :=
  ofCauchy_inf _ _


instance : DistribLattice ℝ :=
  { Real.partialOrder with
    sup := (· ⊔ ·)
    le := (· ≤ ·)
    le_sup_left := by
      /-
        x : Real
        ⊢ ∀ (a b : Real), LE.le a ((fun x1 x2 => Max.max x1 x2) a b)
      -/
      intros a b
      /-
        x a b : Real
        ⊢ LE.le a ((fun x1 x2 => Max.max x1 x2) a b)
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b : Real
        a : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) ((fun x1 x2 => Max.max x1 x2) (Real.mk a) b)
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) ((fun x1 x2 => Max.max x1 x2) (Real.mk a) (Real.mk b))
      -/
      dsimp only; rw [← mk_sup, mk_le]
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le a (Max.max a b)
      -/
      exact CauSeq.le_sup_left
      /-
        🎉 no goals
      -/
    le_sup_right := by
      /-
        x : Real
        ⊢ ∀ (a b : Real), LE.le b ((fun x1 x2 => Max.max x1 x2) a b)
      -/
      intros a b
      /-
        x a b : Real
        ⊢ LE.le b ((fun x1 x2 => Max.max x1 x2) a b)
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b : Real
        a : CauSeq Rat abs
        ⊢ LE.le b ((fun x1 x2 => Max.max x1 x2) (Real.mk a) b)
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Real.mk b) ((fun x1 x2 => Max.max x1 x2) (Real.mk a) (Real.mk b))
      -/
      dsimp only; rw [← mk_sup, mk_le]
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le b (Max.max a b)
      -/
      exact CauSeq.le_sup_right
      /-
        🎉 no goals
      -/
    sup_le := by
      /-
        x : Real
        ⊢ ∀ (a b c : Real), LE.le a c → LE.le b c → LE.le ((fun x1 x2 => Max.max x1 x2 …
      -/
      intros a b c
      /-
        x a b c : Real
        ⊢ LE.le a c → LE.le b c → LE.le ((fun x1 x2 => Max.max x1 x2) a b) c
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b c : Real
        a : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) c → LE.le b c → LE.le ((fun x1 x2 => Max.max x1 x2) (Real. …
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x c : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) c → LE.le (Real.mk b) c → LE.le ((fun x1 x2 => Max.max x1  …
      -/
      induction' c using Real.ind_mk with c
      /-
        case h.h.h
        x : Real
        a b c : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) (Real.mk c) → LE.le (Real.mk b) (Real.mk c) → LE.le ((fun  …
      -/
      simp_rw [← mk_sup, mk_le]
      /-
        case h.h.h
        x : Real
        a b c : CauSeq Rat abs
        ⊢ LE.le a c → LE.le b c → LE.le (Max.max a b) c
      -/
      exact CauSeq.sup_le
      /-
        🎉 no goals
      -/
    inf := (· ⊓ ·)
    inf_le_left := by
      /-
        x : Real
        ⊢ ∀ (a b : Real), LE.le ((fun x1 x2 => Min.min x1 x2) a b) a
      -/
      intros a b
      /-
        x a b : Real
        ⊢ LE.le ((fun x1 x2 => Min.min x1 x2) a b) a
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b : Real
        a : CauSeq Rat abs
        ⊢ LE.le ((fun x1 x2 => Min.min x1 x2) (Real.mk a) b) (Real.mk a)
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le ((fun x1 x2 => Min.min x1 x2) (Real.mk a) (Real.mk b)) (Real.mk a)
      -/
      dsimp only; rw [← mk_inf, mk_le]
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Min.min a b) a
      -/
      exact CauSeq.inf_le_left
      /-
        🎉 no goals
      -/
    inf_le_right := by
      /-
        x : Real
        ⊢ ∀ (a b : Real), LE.le ((fun x1 x2 => Min.min x1 x2) a b) b
      -/
      intros a b
      /-
        x a b : Real
        ⊢ LE.le ((fun x1 x2 => Min.min x1 x2) a b) b
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b : Real
        a : CauSeq Rat abs
        ⊢ LE.le ((fun x1 x2 => Min.min x1 x2) (Real.mk a) b) b
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le ((fun x1 x2 => Min.min x1 x2) (Real.mk a) (Real.mk b)) (Real.mk b)
      -/
      dsimp only; rw [← mk_inf, mk_le]
      /-
        case h.h
        x : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Min.min a b) b
      -/
      exact CauSeq.inf_le_right
      /-
        🎉 no goals
      -/
    le_inf := by
      /-
        x : Real
        ⊢ ∀ (a b c : Real), LE.le a b → LE.le a c → LE.le a ((fun x1 x2 => Min.min x1  …
      -/
      intros a b c
      /-
        x a b c : Real
        ⊢ LE.le a b → LE.le a c → LE.le a ((fun x1 x2 => Min.min x1 x2) b c)
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b c : Real
        a : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) b → LE.le (Real.mk a) c → LE.le (Real.mk a) ((fun x1 x2 => …
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x c : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) (Real.mk b) → LE.le (Real.mk a) c → LE.le (Real.mk a) ((fu …
      -/
      induction' c using Real.ind_mk with c
      /-
        case h.h.h
        x : Real
        a b c : CauSeq Rat abs
        ⊢ LE.le (Real.mk a) (Real.mk b) → LE.le (Real.mk a) (Real.mk c) → LE.le (Real. …
      -/
      simp_rw [← mk_inf, mk_le]
      /-
        case h.h.h
        x : Real
        a b c : CauSeq Rat abs
        ⊢ LE.le a b → LE.le a c → LE.le a (Min.min b c)
      -/
      exact CauSeq.le_inf
      /-
        🎉 no goals
      -/
    le_sup_inf := by
      /-
        x : Real
        ⊢ ∀ (x y z : Real), LE.le (Min.min (Max.max x y) (Max.max x z)) (Max.max x (Mi …
      -/
      intros a b c
      /-
        x a b c : Real
        ⊢ LE.le (Min.min (Max.max a b) (Max.max a c)) (Max.max a (Min.min b c))
      -/
      induction' a using Real.ind_mk with a
      /-
        case h
        x b c : Real
        a : CauSeq Rat abs
        ⊢ LE.le (Min.min (Max.max (Real.mk a) b) (Max.max (Real.mk a) c)) (Max.max (Re …
      -/
      induction' b using Real.ind_mk with b
      /-
        case h.h
        x c : Real
        a b : CauSeq Rat abs
        ⊢ LE.le (Min.min (Max.max (Real.mk a) (Real.mk b)) (Max.max (Real.mk a) c)) (M …
      -/
      induction' c using Real.ind_mk with c
      /-
        case h.h.h
        x : Real
        a b c : CauSeq Rat abs
        ⊢ LE.le (Min.min (Max.max (Real.mk a) (Real.mk b)) (Max.max (Real.mk a) (Real. …
      -/
      apply Eq.le
      /-
        case h.h.h.hab
        x : Real
        a b c : CauSeq Rat abs
        ⊢ Eq (Min.min (Max.max (Real.mk a) (Real.mk b)) (Max.max (Real.mk a) (Real.mk  …
      -/
      simp only [← mk_sup, ← mk_inf]
      /-
        case h.h.h.hab
        x : Real
        a b c : CauSeq Rat abs
        ⊢ Eq (Real.mk (Min.min (Max.max a b) (Max.max a c))) (Real.mk (Max.max a (Min. …
      -/
      exact congr_arg mk (CauSeq.sup_inf_distrib_left _ _ _).symm }
      /-
        🎉 no goals
      -/

-- Extra instances to short-circuit type class resolution

instance lattice : Lattice ℝ :=
  inferInstance


instance : SemilatticeInf ℝ :=
  inferInstance


instance : SemilatticeSup ℝ :=
  inferInstance


instance leTotal_R : IsTotal ℝ (· ≤ ·) :=
  ⟨by
    /-
      x : Real
      ⊢ ∀ (a b : Real), Or (LE.le a b) (LE.le b a)
    -/
    intros a b
    /-
      x a b : Real
      ⊢ Or (LE.le a b) (LE.le b a)
    -/
    induction' a using Real.ind_mk with a
    /-
      case h
      x b : Real
      a : CauSeq Rat abs
      ⊢ Or (LE.le (Real.mk a) b) (LE.le b (Real.mk a))
    -/
    induction' b using Real.ind_mk with b
    /-
      case h.h
      x : Real
      a b : CauSeq Rat abs
      ⊢ Or (LE.le (Real.mk a) (Real.mk b)) (LE.le (Real.mk b) (Real.mk a))
    -/
    simpa using le_total a b⟩
    /-
      🎉 no goals
    -/


open scoped Classical in
noncomputable instance linearOrder : LinearOrder ℝ :=
  Lattice.toLinearOrder ℝ


noncomputable instance linearOrderedCommRing : LinearOrderedCommRing ℝ :=
  { Real.nontrivial, Real.strictOrderedRing, Real.commRing, Real.linearOrder with }

-- Extra instances to short-circuit type class resolution

                                                   /-
                                                     x : Real
                                                     ⊢ LinearOrderedRing Real
                                                   -/
noncomputable instance : LinearOrderedRing ℝ := by infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                       /-
                                                         x : Real
                                                         ⊢ LinearOrderedSemiring Real
                                                       -/
noncomputable instance : LinearOrderedSemiring ℝ := by infer_instance
                                                       /-
                                                         🎉 no goals
                                                       -/


instance : IsDomain ℝ :=
  { Real.nontrivial, Real.commRing, LinearOrderedRing.isDomain with }


noncomputable instance instDivInvMonoid : DivInvMonoid ℝ where


lemma ofCauchy_div (f g) : (⟨f / g⟩ : ℝ) = (⟨f⟩ : ℝ) / (⟨g⟩ : ℝ) := by
  /-
    f g : CauSeq.Completion.Cauchy abs
    ⊢ Eq { cauchy := HDiv.hDiv f g } (HDiv.hDiv { cauchy := f } { cauchy := g })
  -/
  simp_rw [div_eq_mul_inv, ofCauchy_mul, ofCauchy_inv]
  /-
    🎉 no goals
  -/


noncomputable instance instLinearOrderedField : LinearOrderedField ℝ where
  toLinearOrderedCommRing := linearOrderedCommRing
  mul_inv_cancel := by
    /-
      x : Real
      ⊢ ∀ (a : Real), Ne a 0 → Eq (HMul.hMul a (Inv.inv a)) 1
    -/
    rintro ⟨a⟩ h
    /-
      case ofCauchy
      x : Real
      a : CauSeq.Completion.Cauchy abs
      h : Ne { cauchy := a } 0
      ⊢ Eq (HMul.hMul { cauchy := a } (Inv.inv { cauchy := a })) 1
    -/
    rw [mul_comm]
    simp only [← ofCauchy_inv, ← ofCauchy_mul, ← ofCauchy_one, ← ofCauchy_zero,
      Ne, ofCauchy.injEq] at *
    /-
      case ofCauchy
      x : Real
      a : CauSeq.Completion.Cauchy abs
      h : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (Inv.inv a) a) 1
    -/
    exact CauSeq.Completion.inv_mul_cancel h
    /-
      🎉 no goals
    -/
                 /-
                   x : Real
                   ⊢ Eq (Inv.inv 0) 0
                 -/
  inv_zero := by simp [← ofCauchy_zero, ← ofCauchy_inv]
                 /-
                   🎉 no goals
                 -/
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl
  nnratCast_def q := by
    /-
      x : Real
      q : NNRat
      ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
    -/
    rw [← ofCauchy_nnratCast, NNRat.cast_def, ofCauchy_div, ofCauchy_natCast, ofCauchy_natCast]
    /-
      🎉 no goals
    -/
  ratCast_def q := by
    /-
      x : Real
      q : Rat
      ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
    -/
    rw [← ofCauchy_ratCast, Rat.cast_def, ofCauchy_div, ofCauchy_natCast, ofCauchy_intCast]
    /-
      🎉 no goals
    -/

-- Extra instances to short-circuit type class resolution

                                                           /-
                                                             x : Real
                                                             ⊢ LinearOrderedAddCommGroup Real
                                                           -/
noncomputable instance : LinearOrderedAddCommGroup ℝ := by infer_instance
                                                           /-
                                                             🎉 no goals
                                                           -/


                                             /-
                                               x : Real
                                               ⊢ Field Real
                                             -/
noncomputable instance field : Field ℝ := by infer_instance
                                             /-
                                               🎉 no goals
                                             -/


                                              /-
                                                x : Real
                                                ⊢ DivisionRing Real
                                              -/
noncomputable instance : DivisionRing ℝ := by infer_instance
                                              /-
                                                🎉 no goals
                                              -/


                                                                       /-
                                                                         x a b : Real
                                                                         ⊢ Decidable (LT.lt a b)
                                                                       -/
noncomputable instance decidableLT (a b : ℝ) : Decidable (a < b) := by infer_instance
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                       /-
                                                                         x a b : Real
                                                                         ⊢ Decidable (LE.le a b)
                                                                       -/
noncomputable instance decidableLE (a b : ℝ) : Decidable (a ≤ b) := by infer_instance
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                       /-
                                                                         x a b : Real
                                                                         ⊢ Decidable (Eq a b)
                                                                       -/
noncomputable instance decidableEq (a b : ℝ) : Decidable (a = b) := by infer_instance
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- Show an underlying cauchy sequence for real numbers.

The representative chosen is the one passed in the VM to `Quot.mk`, so two cauchy sequences
converging to the same number may be printed differently.
-/
unsafe instance : Repr ℝ where reprPrec r _ := "Real.ofCauchy " ++ repr r.cauchy


theorem le_mk_of_forall_le {f : CauSeq ℚ abs} : (∃ i, ∀ j ≥ i, x ≤ f j) → x ≤ mk f := by
  /-
    x : Real
    f : CauSeq Rat abs
    ⊢ (Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le x ↑(↑f j)) → LE.le x (Real.m …
  -/
  intro h
  /-
    x : Real
    f : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le x ↑(↑f j)
    ⊢ LE.le x (Real.mk f)
  -/
  induction' x using Real.ind_mk with x
  /-
    case h
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    ⊢ LE.le (Real.mk x) (Real.mk f)
  -/
  apply le_of_not_lt
  /-
    case h.h
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    ⊢ Not (LT.lt (Real.mk f) (Real.mk x))
  -/
  rw [mk_lt]
  /-
    case h.h
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    ⊢ Not (LT.lt f x)
  -/
  rintro ⟨K, K0, hK⟩
  /-
    case h.h.intro.intro
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    K : Rat
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (↑(HSub.hSub x f) j)
    ⊢ False
  -/
  obtain ⟨i, H⟩ := exists_forall_ge_and h (exists_forall_ge_and hK (f.cauchy₃ <| half_pos K0))
  /-
    case h.h.intro.intro.intro
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    K : Rat
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (↑(HSub.hSub x f) j)
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → And (LE.le (Real.mk x) ↑(↑f j)) (And (LE.le K (↑( …
    ⊢ False
  -/
  apply not_lt_of_le (H _ le_rfl).1
  /-
    case h.h.intro.intro.intro
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    K : Rat
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (↑(HSub.hSub x f) j)
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → And (LE.le (Real.mk x) ↑(↑f j)) (And (LE.le K (↑( …
    ⊢ LT.lt (↑(↑f i)) (Real.mk x)
  -/
  erw [mk_lt]
  /-
    case h.h.intro.intro.intro
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    K : Rat
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (↑(HSub.hSub x f) j)
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → And (LE.le (Real.mk x) ↑(↑f j)) (And (LE.le K (↑( …
    ⊢ LT.lt (CauSeq.const abs ↑(↑f i)) x
  -/
  refine ⟨_, half_pos K0, i, fun j ij => ?_⟩
  /-
    case h.h.intro.intro.intro
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    K : Rat
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (↑(HSub.hSub x f) j)
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → And (LE.le (Real.mk x) ↑(↑f j)) (And (LE.le K (↑( …
    j : Nat
    ij : GE.ge j i
    ⊢ LE.le (HDiv.hDiv K 2) (↑(HSub.hSub x (CauSeq.const abs ↑(↑f i))) j)
  -/
  have := add_le_add (H _ ij).2.1 (le_of_lt (abs_lt.1 <| (H _ le_rfl).2.2 _ ij).1)
  /-
    case h.h.intro.intro.intro
    x✝ : Real
    f x : CauSeq Rat abs
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (Real.mk x) ↑(↑f j)
    K : Rat
    K0 : GT.gt K 0
    hK : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le K (↑(HSub.hSub x f) j)
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → And (LE.le (Real.mk x) ↑(↑f j)) (And (LE.le K (↑( …
    j : Nat
    ij : GE.ge j i
    this : LE.le (HAdd.hAdd K (Neg.neg (HDiv.hDiv K 2))) (HAdd.hAdd (↑(HSub.hSub x …
    ⊢ LE.le (HDiv.hDiv K 2) (↑(HSub.hSub x (CauSeq.const abs ↑(↑f i))) j)
  -/
  rwa [← sub_eq_add_neg, sub_self_div_two, sub_apply, sub_add_sub_cancel] at this
  /-
    🎉 no goals
  -/


theorem mk_le_of_forall_le {f : CauSeq ℚ abs} {x : ℝ} (h : ∃ i, ∀ j ≥ i, (f j : ℝ) ≤ x) :
    mk f ≤ x := by
  /-
    f : CauSeq Rat abs
    x : Real
    h : Exists fun i => ∀ (j : Nat), GE.ge j i → LE.le (↑(↑f j)) x
    ⊢ LE.le (Real.mk f) x
  -/
  cases' h with i H
  /-
    case intro
    f : CauSeq Rat abs
    x : Real
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → LE.le (↑(↑f j)) x
    ⊢ LE.le (Real.mk f) x
  -/
  rw [← neg_le_neg_iff, ← mk_neg]
  /-
    case intro
    f : CauSeq Rat abs
    x : Real
    i : Nat
    H : ∀ (j : Nat), GE.ge j i → LE.le (↑(↑f j)) x
    ⊢ LE.le (Neg.neg x) (Real.mk (Neg.neg f))
  -/
  exact le_mk_of_forall_le ⟨i, fun j ij => by simp [H _ ij]⟩
  /-
    🎉 no goals
  -/


theorem mk_near_of_forall_near {f : CauSeq ℚ abs} {x : ℝ} {ε : ℝ}
    (H : ∃ i, ∀ j ≥ i, |(f j : ℝ) - x| ≤ ε) : |mk f - x| ≤ ε :=
  abs_sub_le_iff.2
    ⟨sub_le_iff_le_add'.2 <|
        mk_le_of_forall_le <|
          H.imp fun _ h j ij => sub_le_iff_le_add'.1 (abs_sub_le_iff.1 <| h j ij).1,
      sub_le_comm.1 <|
        le_mk_of_forall_le <| H.imp fun _ h j ij => sub_le_comm.1 (abs_sub_le_iff.1 <| h j ij).2⟩


lemma mul_add_one_le_add_one_pow {a : ℝ} (ha : 0 ≤ a) (b : ℕ) : a * b + 1 ≤ (a + 1) ^ b := by
  /-
    a : Real
    ha : LE.le 0 a
    b : Nat
    ⊢ LE.le (HAdd.hAdd (HMul.hMul a ↑b) 1) (HPow.hPow (HAdd.hAdd a 1) b)
  -/
  rcases ha.eq_or_lt with rfl|ha'
    /-
      case inl
      b : Nat
      ha : LE.le 0 0
      ⊢ LE.le (HAdd.hAdd (HMul.hMul 0 ↑b) 1) (HPow.hPow (HAdd.hAdd 0 1) b)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    a : Real
    ha : LE.le 0 a
    b : Nat
    ha' : LT.lt 0 a
    ⊢ LE.le (HAdd.hAdd (HMul.hMul a ↑b) 1) (HPow.hPow (HAdd.hAdd a 1) b)
  -/
  clear ha
  induction b generalizing a with
  | zero => simp
  | succ b hb =>
    calc
      a * ↑(b + 1) + 1 = (0 + 1) ^ b * a + (a * b + 1) := by
        simp [mul_add, add_assoc, add_left_comm]
      _ ≤ (a + 1) ^ b * a + (a + 1) ^ b := by
        gcongr
        · norm_num
        · exact hb ha'
      _ = (a + 1) ^ (b + 1) := by simp [pow_succ, mul_add]


/-- A function `f : R → ℝ≥0` is nonarchimedean if it satisfies the strong triangle inequality
  `f (r + s) ≤ max (f r) (f s)` for all `r s : R`. -/
def IsNonarchimedean {A : Type*} [Add A] (f : A → ℝ) : Prop :=
  ∀ r s, f (r + s) ≤ max (f r) (f s)


/-- A function `f : R → ℝ` is power-multiplicative if for all `r ∈ R` and all positive `n ∈ ℕ`,
`f (r ^ n) = (f r) ^ n`. -/
def IsPowMul {R : Type*} [Pow R ℕ] (f : R → ℝ) :=
  ∀ (a : R) {n : ℕ}, 1 ≤ n → f (a ^ n) = f a ^ n


/-- A ring homomorphism `f : α →+* β` is bounded with respect to the functions `nα : α → ℝ` and
  `nβ : β → ℝ` if there exists a positive constant `C` such that for all `x` in `α`,
  `nβ (f x) ≤ C * nα x`. -/
def RingHom.IsBoundedWrt {α : Type*} [Ring α] {β : Type*} [Ring β] (nα : α → ℝ) (nβ : β → ℝ)
    (f : α →+* β) : Prop :=
  ∃ C : ℝ, 0 < C ∧ ∀ x : α, nβ (f x) ≤ C * nα x

