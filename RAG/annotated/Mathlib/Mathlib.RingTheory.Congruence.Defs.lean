/-- A congruence relation on a type with an addition and multiplication is an equivalence relation
which preserves both. -/
structure RingCon (R : Type*) [Add R] [Mul R] extends Con R, AddCon R where


/-- The inductively defined smallest ring congruence relation containing a given binary
    relation. -/
inductive RingConGen.Rel [Add R] [Mul R] (r : R → R → Prop) : R → R → Prop
  | of : ∀ x y, r x y → RingConGen.Rel r x y
  | refl : ∀ x, RingConGen.Rel r x x
  | symm : ∀ {x y}, RingConGen.Rel r x y → RingConGen.Rel r y x
  | trans : ∀ {x y z}, RingConGen.Rel r x y → RingConGen.Rel r y z → RingConGen.Rel r x z
  | add : ∀ {w x y z}, RingConGen.Rel r w x → RingConGen.Rel r y z →
      RingConGen.Rel r (w + y) (x + z)
  | mul : ∀ {w x y z}, RingConGen.Rel r w x → RingConGen.Rel r y z →
      RingConGen.Rel r (w * y) (x * z)


/-- The inductively defined smallest ring congruence relation containing a given binary
    relation. -/
def ringConGen [Add R] [Mul R] (r : R → R → Prop) : RingCon R where
  r := RingConGen.Rel r
  iseqv := ⟨RingConGen.Rel.refl, @RingConGen.Rel.symm _ _ _ _, @RingConGen.Rel.trans _ _ _ _⟩
  add' := RingConGen.Rel.add
  mul' := RingConGen.Rel.mul


/-- A coercion from a congruence relation to its underlying binary relation. -/
instance : FunLike (RingCon R) R (R → Prop) where
  coe c := c.r
  coe_injective' x y h := by
    /-
      R : Type u_1
      inst✝¹ : Add R
      inst✝ : Mul R
      c x y : RingCon R
      h : Eq ((fun c => ⇑c.toSetoid) x) ((fun c => ⇑c.toSetoid) y)
      ⊢ Eq x y
    -/
    rcases x with ⟨⟨x, _⟩, _⟩
    /-
      case mk.mk
      R : Type u_1
      inst✝¹ : Add R
      inst✝ : Mul R
      c y : RingCon R
      x : Setoid R
      mul'✝ : ∀ {w x_1 y z : R}, x w x_1 → x y z → x (HMul.hMul w y) (HMul.hMul x_1 z)
      add'✝ : ∀ {w x_1 y z : R}, { toSetoid := x, mul' := mul'✝ }.toSetoid w x_1 → { …
      h : Eq ((fun c => ⇑c.toSetoid) { toSetoid := x, mul' := mul'✝, add' := add'✝ } …
      ⊢ Eq { toSetoid := x, mul' := mul'✝, add' := add'✝ } y
    -/
    rcases y with ⟨⟨y, _⟩, _⟩
    /-
      case mk.mk.mk.mk
      R : Type u_1
      inst✝¹ : Add R
      inst✝ : Mul R
      c : RingCon R
      x : Setoid R
      mul'✝¹ : ∀ {w x_1 y z : R}, x w x_1 → x y z → x (HMul.hMul w y) (HMul.hMul x_1 …
      add'✝¹ : ∀ {w x_1 y z : R}, { toSetoid := x, mul' := mul'✝¹ }.toSetoid w x_1 → …
      y : Setoid R
      mul'✝ : ∀ {w x y_1 z : R}, y w x → y y_1 z → y (HMul.hMul w y_1) (HMul.hMul x z)
      add'✝ : ∀ {w x y_1 z : R}, { toSetoid := y, mul' := mul'✝ }.toSetoid w x → { t …
      h : Eq ((fun c => ⇑c.toSetoid) { toSetoid := x, mul' := mul'✝¹, add' := add'✝¹ …
      ⊢ Eq { toSetoid := x, mul' := mul'✝¹, add' := add'✝¹ } { toSetoid := y, mul' : …
    -/
    congr!
    /-
      case mk.mk.mk.mk.h.e'_4.h.e'_3
      R : Type u_1
      inst✝¹ : Add R
      inst✝ : Mul R
      c : RingCon R
      x : Setoid R
      mul'✝¹ : ∀ {w x_1 y z : R}, x w x_1 → x y z → x (HMul.hMul w y) (HMul.hMul x_1 …
      add'✝¹ : ∀ {w x_1 y z : R}, { toSetoid := x, mul' := mul'✝¹ }.toSetoid w x_1 → …
      y : Setoid R
      mul'✝ : ∀ {w x y_1 z : R}, y w x → y y_1 z → y (HMul.hMul w y_1) (HMul.hMul x z)
      add'✝ : ∀ {w x y_1 z : R}, { toSetoid := y, mul' := mul'✝ }.toSetoid w x → { t …
      h : Eq ((fun c => ⇑c.toSetoid) { toSetoid := x, mul' := mul'✝¹, add' := add'✝¹ …
      ⊢ Eq x y
    -/
    rw [Setoid.ext_iff, (show ⇑x = ⇑y from h)]
    /-
      case mk.mk.mk.mk.h.e'_4.h.e'_3
      R : Type u_1
      inst✝¹ : Add R
      inst✝ : Mul R
      c : RingCon R
      x : Setoid R
      mul'✝¹ : ∀ {w x_1 y z : R}, x w x_1 → x y z → x (HMul.hMul w y) (HMul.hMul x_1 …
      add'✝¹ : ∀ {w x_1 y z : R}, { toSetoid := x, mul' := mul'✝¹ }.toSetoid w x_1 → …
      y : Setoid R
      mul'✝ : ∀ {w x y_1 z : R}, y w x → y y_1 z → y (HMul.hMul w y_1) (HMul.hMul x z)
      add'✝ : ∀ {w x y_1 z : R}, { toSetoid := y, mul' := mul'✝ }.toSetoid w x → { t …
      h : Eq ((fun c => ⇑c.toSetoid) { toSetoid := x, mul' := mul'✝¹, add' := add'✝¹ …
      ⊢ ∀ (a b : R), Iff (y a b) (y a b)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem rel_eq_coe : c.r = c :=
  rfl


@[simp]
theorem toCon_coe_eq_coe : (c.toCon : R → R → Prop) = c :=
  rfl


protected theorem refl (x) : c x x :=
  c.refl' x


protected theorem symm {x y} : c x y → c y x :=
  c.symm'


protected theorem trans {x y z} : c x y → c y z → c x z :=
  c.trans'


protected theorem add {w x y z} : c w x → c y z → c (w + y) (x + z) :=
  c.add'


protected theorem mul {w x y z} : c w x → c y z → c (w * y) (x * z) :=
  c.mul'


protected theorem sub {S : Type*} [AddGroup S] [Mul S] (t : RingCon S)
    {a b c d : S} (h : t a b) (h' : t c d) : t (a - c) (b - d) := t.toAddCon.sub h h'


protected theorem neg {S : Type*} [AddGroup S] [Mul S] (t : RingCon S)
    {a b} (h : t a b) : t (-a) (-b) := t.toAddCon.neg h


protected theorem nsmul {S : Type*} [AddGroup S] [Mul S] (t : RingCon S)
    (m : ℕ) {x y : S} (hx : t x y) : t (m • x) (m • y) := t.toAddCon.nsmul m hx


protected theorem zsmul {S : Type*} [AddGroup S] [Mul S] (t : RingCon S)
    (z : ℤ) {x y : S} (hx : t x y) : t (z • x) (z • y) := t.toAddCon.zsmul z hx


instance : Inhabited (RingCon R) :=
  ⟨ringConGen EmptyRelation⟩


@[simp]
theorem rel_mk {s : Con R} {h a b} : RingCon.mk s h a b ↔ s a b :=
  Iff.rfl


/-- The map sending a congruence relation to its underlying binary relation is injective. -/
theorem ext' {c d : RingCon R} (H : ⇑c = ⇑d) : c = d := DFunLike.coe_injective H


/-- Extensionality rule for congruence relations. -/
theorem ext {c d : RingCon R} (H : ∀ x y, c x y ↔ d x y) : c = d :=
             /-
               R : Type u_1
               inst✝¹ : Add R
               inst✝ : Mul R
               c d : RingCon R
               H : ∀ (x y : R), Iff (c x y) (d x y)
               ⊢ Eq ⇑c ⇑d
             -/
  ext' <| by ext; apply H
                  /-
                    🎉 no goals
                  -/


/--
Pulling back a `RingCon` across a ring homomorphism.
-/
def comap {R R' F : Type*} [Add R] [Add R']
    [FunLike F R R'] [AddHomClass F R R'] [Mul R] [Mul R'] [MulHomClass F R R']
    (J : RingCon R') (f : F) :
    RingCon R where
  __ := J.toCon.comap f (map_mul f)
  __ := J.toAddCon.comap f (map_add f)


/-- Defining the quotient by a congruence relation of a type with addition and multiplication. -/
protected def Quotient :=
  Quotient c.toSetoid


/-- The morphism into the quotient by a congruence relation -/
@[coe] def toQuotient (r : R) : c.Quotient :=
  @Quotient.mk'' _ c.toSetoid r


/-- Coercion from a type with addition and multiplication to its quotient by a congruence relation.

See Note [use has_coe_t]. -/
instance : CoeTC R c.Quotient :=
  ⟨toQuotient⟩

-- Lower the priority since it unifies with any quotient type.

/-- The quotient by a decidable congruence relation has decidable equality. -/
instance (priority := 500) [_d : ∀ a b, Decidable (c a b)] : DecidableEq c.Quotient :=
  inferInstanceAs (DecidableEq (Quotient c.toSetoid))


@[simp]
theorem quot_mk_eq_coe (x : R) : Quot.mk c x = (x : c.Quotient) :=
  rfl


/-- Two elements are related by a congruence relation `c` iff they are represented by the same
element of the quotient by `c`. -/
@[simp]
protected theorem eq {a b : R} : (a : c.Quotient) = (b : c.Quotient) ↔ c a b :=
  Quotient.eq''


instance : Add c.Quotient := inferInstanceAs (Add c.toAddCon.Quotient)


@[simp, norm_cast]
theorem coe_add (x y : R) : (↑(x + y) : c.Quotient) = ↑x + ↑y :=
  rfl


instance : Mul c.Quotient := inferInstanceAs (Mul c.toCon.Quotient)


@[simp, norm_cast]
theorem coe_mul (x y : R) : (↑(x * y) : c.Quotient) = ↑x * ↑y :=
  rfl


instance : Zero c.Quotient := inferInstanceAs (Zero c.toAddCon.Quotient)


@[simp, norm_cast]
theorem coe_zero : (↑(0 : R) : c.Quotient) = 0 :=
  rfl


instance : One c.Quotient := inferInstanceAs (One c.toCon.Quotient)


@[simp, norm_cast]
theorem coe_one : (↑(1 : R) : c.Quotient) = 1 :=
  rfl


instance : Neg c.Quotient := inferInstanceAs (Neg c.toAddCon.Quotient)


@[simp, norm_cast]
theorem coe_neg (x : R) : (↑(-x) : c.Quotient) = -x :=
  rfl


instance : Sub c.Quotient := inferInstanceAs (Sub c.toAddCon.Quotient)


@[simp, norm_cast]
theorem coe_sub (x y : R) : (↑(x - y) : c.Quotient) = x - y :=
  rfl


instance hasZSMul : SMul ℤ c.Quotient := inferInstanceAs (SMul ℤ c.toAddCon.Quotient)


@[simp, norm_cast]
theorem coe_zsmul (z : ℤ) (x : R) : (↑(z • x) : c.Quotient) = z • (x : c.Quotient) :=
  rfl


instance hasNSMul : SMul ℕ c.Quotient := inferInstanceAs (SMul ℕ c.toAddCon.Quotient)


@[simp, norm_cast]
theorem coe_nsmul (n : ℕ) (x : R) : (↑(n • x) : c.Quotient) = n • (x : c.Quotient) :=
  rfl


instance : Pow c.Quotient ℕ := inferInstanceAs (Pow c.toCon.Quotient ℕ)


@[simp, norm_cast]
theorem coe_pow (x : R) (n : ℕ) : (↑(x ^ n) : c.Quotient) = (x : c.Quotient) ^ n :=
  rfl


instance : NatCast c.Quotient :=
  ⟨fun n => ↑(n : R)⟩


@[simp, norm_cast]
theorem coe_natCast (n : ℕ) : (↑(n : R) : c.Quotient) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := coe_natCast


instance : IntCast c.Quotient :=
  ⟨fun z => ↑(z : R)⟩


@[simp, norm_cast]
theorem coe_intCast (n : ℕ) : (↑(n : R) : c.Quotient) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_int_cast := coe_intCast


instance [Inhabited R] [Add R] [Mul R] (c : RingCon R) : Inhabited c.Quotient :=
  ⟨↑(default : R)⟩


instance [NonUnitalNonAssocSemiring R] (c : RingCon R) : NonUnitalNonAssocSemiring c.Quotient :=
  Function.Surjective.nonUnitalNonAssocSemiring _ Quotient.mk''_surjective rfl
    (fun _ _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


instance [NonAssocSemiring R] (c : RingCon R) : NonAssocSemiring c.Quotient :=
  Function.Surjective.nonAssocSemiring _ Quotient.mk''_surjective rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) fun _ => rfl


instance [NonUnitalSemiring R] (c : RingCon R) : NonUnitalSemiring c.Quotient :=
  Function.Surjective.nonUnitalSemiring _ Quotient.mk''_surjective rfl (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


instance [Semiring R] (c : RingCon R) : Semiring c.Quotient :=
  Function.Surjective.semiring _ Quotient.mk''_surjective rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) fun _ => rfl


instance [CommSemiring R] (c : RingCon R) : CommSemiring c.Quotient :=
  Function.Surjective.commSemiring _ Quotient.mk''_surjective rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) fun _ => rfl


instance [NonUnitalNonAssocRing R] (c : RingCon R) : NonUnitalNonAssocRing c.Quotient :=
  Function.Surjective.nonUnitalNonAssocRing _ Quotient.mk''_surjective rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


instance [NonAssocRing R] (c : RingCon R) : NonAssocRing c.Quotient :=
  Function.Surjective.nonAssocRing _ Quotient.mk''_surjective rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ => rfl) fun _ => rfl


instance [NonUnitalRing R] (c : RingCon R) : NonUnitalRing c.Quotient :=
  Function.Surjective.nonUnitalRing _ Quotient.mk''_surjective rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


instance [Ring R] (c : RingCon R) : Ring c.Quotient :=
  Function.Surjective.ring _ Quotient.mk''_surjective rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) fun _ => rfl


instance [CommRing R] (c : RingCon R) : CommRing c.Quotient :=
  Function.Surjective.commRing _ Quotient.mk''_surjective rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) fun _ => rfl


/-- The natural homomorphism from a ring to its quotient by a congruence relation. -/
def mk' [NonAssocSemiring R] (c : RingCon R) : R →+* c.Quotient where
  toFun := toQuotient
  map_zero' := rfl
  map_one' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl


