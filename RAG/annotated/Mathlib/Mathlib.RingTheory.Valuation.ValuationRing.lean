/-- A magma is called a `PreValuationRing` provided that for any pair
of elements `a b : A`, either `a` divides `b` or vice versa. -/
class PreValuationRing (A : Type u) [Mul A] : Prop where
  cond' : ∀ a b : A, ∃ c : A, a * c = b ∨ b * c = a


lemma PreValuationRing.cond {A : Type u} [Mul A] [PreValuationRing A] (a b : A) :
    ∃ c : A, a * c = b ∨ b * c = a := @PreValuationRing.cond' A _ _ _ _


/-- An integral domain is called a `ValuationRing` provided that for any pair
of elements `a b : A`, either `a` divides `b` or vice versa. -/
class ValuationRing (A : Type u) [CommRing A] [IsDomain A] extends PreValuationRing A : Prop

-- Porting note: this lemma is needed since infer kinds are unsupported in Lean 4

lemma ValuationRing.cond {A : Type u} [CommRing A] [IsDomain A] [ValuationRing A] (a b : A) :
    ∃ c : A, a * c = b ∨ b * c = a := PreValuationRing.cond _ _


/-- The value group of the valuation ring `A`. Note: this is actually a group with zero. -/
def ValueGroup : Type v := Quotient (MulAction.orbitRel Aˣ K)


instance : Inhabited (ValueGroup A K) := ⟨Quotient.mk'' 0⟩


instance : LE (ValueGroup A K) :=
  LE.mk fun x y =>
    Quotient.liftOn₂' x y (fun a b => ∃ c : A, c • b = a)
      (by
        /-
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          ⊢ ∀ (a₁ a₂ b₁ b₂ : K), (MulAction.orbitRel (Units A) K) a₁ b₁ → (MulAction.orb …
        -/
        rintro _ _ a b ⟨c, rfl⟩ ⟨d, rfl⟩; ext
        /-
          case intro.intro.a
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          a b : K
          c d : Units A
          ⊢ Iff ((fun a b => Exists fun c => Eq (HSMul.hSMul c b) a) ((fun m => HSMul.hS …
        -/
        constructor
          /-
            case intro.intro.a.mp
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            ⊢ (fun a b => Exists fun c => Eq (HSMul.hSMul c b) a) ((fun m => HSMul.hSMul m …
          -/
        · rintro ⟨e, he⟩; use (c⁻¹ : Aˣ) * e * d
          /-
            case h
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            e : A
            he : Eq (HSMul.hSMul e ((fun m => HSMul.hSMul m b) d)) ((fun m => HSMul.hSMul  …
            ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (↑(Inv.inv c)) e) ↑d) b) a
          -/
          apply_fun fun t => c⁻¹ • t at he
          /-
            case h
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            e : A
            he : Eq (HSMul.hSMul (Inv.inv c) (HSMul.hSMul e ((fun m => HSMul.hSMul m b) d) …
            ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (↑(Inv.inv c)) e) ↑d) b) a
          -/
          simpa [mul_smul] using he
          /-
            🎉 no goals
          -/
          /-
            case intro.intro.a.mpr
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            ⊢ (fun a b => Exists fun c => Eq (HSMul.hSMul c b) a) a b → (fun a b => Exists …
          -/
        · rintro ⟨e, he⟩; dsimp
          /-
            case intro.intro.a.mpr.intro
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            e : A
            he : Eq (HSMul.hSMul e b) a
            ⊢ Exists fun c_1 => Eq (HSMul.hSMul c_1 (HSMul.hSMul d b)) (HSMul.hSMul c a)
          -/
          use c * e * (d⁻¹ : Aˣ)
          /-
            case h
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            e : A
            he : Eq (HSMul.hSMul e b) a
            ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (↑c) e) ↑(Inv.inv d)) (HSMul.hSMul d b …
          -/
          simp_rw [Units.smul_def, ← he, mul_smul]
          /-
            case h
            A : Type u
            inst✝² : CommRing A
            K : Type v
            inst✝¹ : Field K
            inst✝ : Algebra A K
            x y : ValuationRing.ValueGroup A K
            a b : K
            c d : Units A
            e : A
            he : Eq (HSMul.hSMul e b) a
            ⊢ Eq (HSMul.hSMul (↑c) (HSMul.hSMul e (HSMul.hSMul (↑(Inv.inv d)) (HSMul.hSMul …
          -/
          rw [← mul_smul _ _ b, Units.inv_mul, one_smul])
          /-
            🎉 no goals
          -/


instance : Zero (ValueGroup A K) := ⟨Quotient.mk'' 0⟩


instance : One (ValueGroup A K) := ⟨Quotient.mk'' 1⟩


instance : Mul (ValueGroup A K) :=
  Mul.mk fun x y =>
    Quotient.liftOn₂' x y (fun a b => Quotient.mk'' <| a * b)
      (by
        /-
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          ⊢ ∀ (a₁ a₂ b₁ b₂ : K), (MulAction.orbitRel (Units A) K) a₁ b₁ → (MulAction.orb …
        -/
        rintro _ _ a b ⟨c, rfl⟩ ⟨d, rfl⟩
        /-
          case intro.intro
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          a b : K
          c d : Units A
          ⊢ Eq ((fun a b => Quotient.mk'' (HMul.hMul a b)) ((fun m => HSMul.hSMul m a) c …
        -/
        apply Quotient.sound'
        /-
          case intro.intro.a
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          a b : K
          c d : Units A
          ⊢ (MulAction.orbitRel (Units A) K) (HMul.hMul ((fun m => HSMul.hSMul m a) c) ( …
        -/
        dsimp
        /-
          case intro.intro.a
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          a b : K
          c d : Units A
          ⊢ (MulAction.orbitRel (Units A) K) (HMul.hMul (HSMul.hSMul c a) (HSMul.hSMul d …
        -/
        use c * d
        /-
          case h
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          a b : K
          c d : Units A
          ⊢ Eq ((fun m => HSMul.hSMul m (HMul.hMul a b)) (HMul.hMul c d)) (HMul.hMul (HS …
        -/
        simp only [mul_smul, Algebra.smul_def, Units.smul_def, RingHom.map_mul, Units.val_mul]
        /-
          case h
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x y : ValuationRing.ValueGroup A K
          a b : K
          c d : Units A
          ⊢ Eq (HMul.hMul ((algebraMap A K) ↑c) (HMul.hMul ((algebraMap A K) ↑d) (HMul.h …
        -/
        ring)
        /-
          🎉 no goals
        -/


instance : Inv (ValueGroup A K) :=
  Inv.mk fun x =>
    Quotient.liftOn' x (fun a => Quotient.mk'' a⁻¹)
      (by
        /-
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x : ValuationRing.ValueGroup A K
          ⊢ ∀ (a b : K), (MulAction.orbitRel (Units A) K) a b → Eq ((fun a => Quotient.m …
        -/
        rintro _ a ⟨b, rfl⟩
        /-
          case intro
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x : ValuationRing.ValueGroup A K
          a : K
          b : Units A
          ⊢ Eq ((fun a => Quotient.mk'' (Inv.inv a)) ((fun m => HSMul.hSMul m a) b)) ((f …
        -/
        apply Quotient.sound'
        /-
          case intro.a
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x : ValuationRing.ValueGroup A K
          a : K
          b : Units A
          ⊢ (MulAction.orbitRel (Units A) K) (Inv.inv ((fun m => HSMul.hSMul m a) b)) (I …
        -/
        use b⁻¹
        /-
          case h
          A : Type u
          inst✝² : CommRing A
          K : Type v
          inst✝¹ : Field K
          inst✝ : Algebra A K
          x : ValuationRing.ValueGroup A K
          a : K
          b : Units A
          ⊢ Eq ((fun m => HSMul.hSMul m (Inv.inv a)) (Inv.inv b)) (Inv.inv ((fun m => HS …
        -/
        dsimp
        rw [Units.smul_def, Units.smul_def, Algebra.smul_def, Algebra.smul_def, mul_inv,
          map_units_inv])


protected theorem le_total (a b : ValueGroup A K) : a ≤ b ∨ b ≤ a := by
  /-
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    a b : ValuationRing.ValueGroup A K
    ⊢ Or (LE.le a b) (LE.le b a)
  -/
  rcases a with ⟨a⟩; rcases b with ⟨b⟩
  /-
    case mk.mk
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    a✝ b✝ : ValuationRing.ValueGroup A K
    a b : K
    ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) (Quot.mk (⇑(MulAct …
  -/
  obtain ⟨xa, ya, hya, rfl⟩ : ∃ a b : A, _ := IsFractionRing.div_surjective a
  /-
    case mk.mk.intro.intro.intro
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    a b✝ : ValuationRing.ValueGroup A K
    b : K
    xa ya : A
    hya : Membership.mem (nonZeroDivisors A) ya
    ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraM …
  -/
  obtain ⟨xb, yb, hyb, rfl⟩ : ∃ a b : A, _ := IsFractionRing.div_surjective b
  /-
    case mk.mk.intro.intro.intro.intro.intro.intro
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    a b : ValuationRing.ValueGroup A K
    xa ya : A
    hya : Membership.mem (nonZeroDivisors A) ya
    xb yb : A
    hyb : Membership.mem (nonZeroDivisors A) yb
    ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraM …
  -/
  have : (algebraMap A K) ya ≠ 0 := IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors hya
  /-
    case mk.mk.intro.intro.intro.intro.intro.intro
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    a b : ValuationRing.ValueGroup A K
    xa ya : A
    hya : Membership.mem (nonZeroDivisors A) ya
    xb yb : A
    hyb : Membership.mem (nonZeroDivisors A) yb
    this : Ne ((algebraMap A K) ya) 0
    ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraM …
  -/
  have : (algebraMap A K) yb ≠ 0 := IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors hyb
  /-
    case mk.mk.intro.intro.intro.intro.intro.intro
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    a b : ValuationRing.ValueGroup A K
    xa ya : A
    hya : Membership.mem (nonZeroDivisors A) ya
    xb yb : A
    hyb : Membership.mem (nonZeroDivisors A) yb
    this✝ : Ne ((algebraMap A K) ya) 0
    this : Ne ((algebraMap A K) yb) 0
    ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraM …
  -/
  obtain ⟨c, h | h⟩ := ValuationRing.cond (xa * yb) (xb * ya)
    /-
      case mk.mk.intro.intro.intro.intro.intro.intro.intro.inl
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
      ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraM …
    -/
  · right
    /-
      case mk.mk.intro.intro.intro.intro.intro.intro.intro.inl.h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
      ⊢ LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraMap A …
    -/
    use c
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
      ⊢ Eq (HSMul.hSMul c (HDiv.hDiv ((algebraMap A K) xa) ((algebraMap A K) ya))) ( …
    -/
    rw [Algebra.smul_def]
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
      ⊢ Eq (HMul.hMul ((algebraMap A K) c) (HDiv.hDiv ((algebraMap A K) xa) ((algebr …
    -/
    field_simp
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
      ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap A K) c) ((algebraMap A K) xa)) ((algeb …
    -/
    simp only [← RingHom.map_mul, ← h]; congr 1; ring
                                                 /-
                                                   🎉 no goals
                                                 -/
    /-
      case mk.mk.intro.intro.intro.intro.intro.intro.intro.inr
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
      ⊢ Or (LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraM …
    -/
  · left
    /-
      case mk.mk.intro.intro.intro.intro.intro.intro.intro.inr.h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
      ⊢ LE.le (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HDiv.hDiv ((algebraMap A …
    -/
    use c
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
      ⊢ Eq (HSMul.hSMul c (HDiv.hDiv ((algebraMap A K) xb) ((algebraMap A K) yb))) ( …
    -/
    rw [Algebra.smul_def]
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
      ⊢ Eq (HMul.hMul ((algebraMap A K) c) (HDiv.hDiv ((algebraMap A K) xb) ((algebr …
    -/
    field_simp
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : ValuationRing.ValueGroup A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      c : A
      h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
      ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap A K) c) ((algebraMap A K) xb)) ((algeb …
    -/
    simp only [← RingHom.map_mul, ← h]; congr 1; ring
                                                 /-
                                                   🎉 no goals
                                                 -/

-- Porting note: it is much faster to split the instance `LinearOrderedCommGroupWithZero`
-- into two parts

noncomputable instance linearOrder : LinearOrder (ValueGroup A K) where
                /-
                  A : Type u
                  inst✝⁵ : CommRing A
                  K : Type v
                  inst✝⁴ : Field K
                  inst✝³ : Algebra A K
                  inst✝² : IsDomain A
                  inst✝¹ : ValuationRing A
                  inst✝ : IsFractionRing A K
                  ⊢ ∀ (a : ValuationRing.ValueGroup A K), LE.le a a
                -/
  le_refl := by rintro ⟨⟩; use 1; rw [one_smul]
                                  /-
                                    🎉 no goals
                                  -/
                 /-
                   A : Type u
                   inst✝⁵ : CommRing A
                   K : Type v
                   inst✝⁴ : Field K
                   inst✝³ : Algebra A K
                   inst✝² : IsDomain A
                   inst✝¹ : ValuationRing A
                   inst✝ : IsFractionRing A K
                   ⊢ ∀ (a b c : ValuationRing.ValueGroup A K), LE.le a b → LE.le b c → LE.le a c
                 -/
  le_trans := by rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ ⟨e, rfl⟩ ⟨f, rfl⟩; use e * f; rw [mul_smul]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  le_antisymm := by
    /-
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      ⊢ ∀ (a b : ValuationRing.ValueGroup A K), LE.le a b → LE.le b a → Eq a b
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨e, rfl⟩ ⟨f, hf⟩
    /-
      case mk.mk.intro.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a✝ b✝ : ValuationRing.ValueGroup A K
      b : K
      e f : A
      hf : Eq (HSMul.hSMul f (HSMul.hSMul e b)) b
      ⊢ Eq (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HSMul.hSMul e b)) (Quot.mk  …
    -/
    by_cases hb : b = 0; · simp [hb]
                           /-
                             🎉 no goals
                           -/
    have : IsUnit e := by
      apply isUnit_of_dvd_one
      use f
      rw [mul_comm]
      rw [← mul_smul, Algebra.smul_def] at hf
      nth_rw 2 [← one_mul b] at hf
      rw [← (algebraMap A K).map_one] at hf
      exact IsFractionRing.injective _ _ (mul_right_cancel₀ hb hf).symm
    /-
      case neg
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a✝ b✝ : ValuationRing.ValueGroup A K
      b : K
      e f : A
      hf : Eq (HSMul.hSMul f (HSMul.hSMul e b)) b
      hb : Not (Eq b 0)
      this : IsUnit e
      ⊢ Eq (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) (HSMul.hSMul e b)) (Quot.mk  …
    -/
    apply Quotient.sound'
    /-
      case neg.a
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a✝ b✝ : ValuationRing.ValueGroup A K
      b : K
      e f : A
      hf : Eq (HSMul.hSMul f (HSMul.hSMul e b)) b
      hb : Not (Eq b 0)
      this : IsUnit e
      ⊢ (MulAction.orbitRel (Units A) K) (HSMul.hSMul e b) b
    -/
    exact ⟨this.unit, rfl⟩
    /-
      🎉 no goals
    -/
  le_total := ValuationRing.le_total _ _
                    /-
                      A : Type u
                      inst✝⁵ : CommRing A
                      K : Type v
                      inst✝⁴ : Field K
                      inst✝³ : Algebra A K
                      inst✝² : IsDomain A
                      inst✝¹ : ValuationRing A
                      inst✝ : IsFractionRing A K
                      ⊢ DecidableRel fun x1 x2 => LE.le x1 x2
                    -/
  decidableLE := by classical infer_instance
                    /-
                      🎉 no goals
                    -/


noncomputable instance linearOrderedCommGroupWithZero :
    LinearOrderedCommGroupWithZero (ValueGroup A K) :=
  { linearOrder .. with
                    /-
                      A : Type u
                      inst✝⁵ : CommRing A
                      K : Type v
                      inst✝⁴ : Field K
                      inst✝³ : Algebra A K
                      inst✝² : IsDomain A
                      inst✝¹ : ValuationRing A
                      inst✝ : IsFractionRing A K
                      ⊢ ∀ (a b c : ValuationRing.ValueGroup A K), Eq (HMul.hMul (HMul.hMul a b) c) ( …
                    -/
    mul_assoc := by rintro ⟨a⟩ ⟨b⟩ ⟨c⟩; apply Quotient.sound'; rw [mul_assoc]
                                                               /-
                                                                 🎉 no goals
                                                               -/
                  /-
                    A : Type u
                    inst✝⁵ : CommRing A
                    K : Type v
                    inst✝⁴ : Field K
                    inst✝³ : Algebra A K
                    inst✝² : IsDomain A
                    inst✝¹ : ValuationRing A
                    inst✝ : IsFractionRing A K
                    ⊢ ∀ (a : ValuationRing.ValueGroup A K), Eq (HMul.hMul 1 a) a
                  -/
    one_mul := by rintro ⟨a⟩; apply Quotient.sound'; rw [one_mul]
                                                     /-
                                                       🎉 no goals
                                                     -/
                  /-
                    A : Type u
                    inst✝⁵ : CommRing A
                    K : Type v
                    inst✝⁴ : Field K
                    inst✝³ : Algebra A K
                    inst✝² : IsDomain A
                    inst✝¹ : ValuationRing A
                    inst✝ : IsFractionRing A K
                    ⊢ ∀ (a : ValuationRing.ValueGroup A K), Eq (HMul.hMul a 1) a
                  -/
    mul_one := by rintro ⟨a⟩; apply Quotient.sound'; rw [mul_one]
                                                     /-
                                                       🎉 no goals
                                                     -/
                   /-
                     A : Type u
                     inst✝⁵ : CommRing A
                     K : Type v
                     inst✝⁴ : Field K
                     inst✝³ : Algebra A K
                     inst✝² : IsDomain A
                     inst✝¹ : ValuationRing A
                     inst✝ : IsFractionRing A K
                     ⊢ ∀ (a b : ValuationRing.ValueGroup A K), Eq (HMul.hMul a b) (HMul.hMul b a)
                   -/
    mul_comm := by rintro ⟨a⟩ ⟨b⟩; apply Quotient.sound'; rw [mul_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/
    mul_le_mul_left := by
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        ⊢ ∀ (a b : ValuationRing.ValueGroup A K), LE.le a b → ∀ (c : ValuationRing.Val …
      -/
      rintro ⟨a⟩ ⟨b⟩ ⟨c, rfl⟩ ⟨d⟩
      /-
        case mk.mk.intro.mk
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ b✝ : ValuationRing.ValueGroup A K
        b : K
        c : A
        c✝ : ValuationRing.ValueGroup A K
        d : K
        ⊢ LE.le (HMul.hMul (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) d) (Quot.mk (⇑ …
      -/
      use c; simp only [Algebra.smul_def]; ring
                                           /-
                                             🎉 no goals
                                           -/
                   /-
                     A : Type u
                     inst✝⁵ : CommRing A
                     K : Type v
                     inst✝⁴ : Field K
                     inst✝³ : Algebra A K
                     inst✝² : IsDomain A
                     inst✝¹ : ValuationRing A
                     inst✝ : IsFractionRing A K
                     ⊢ ∀ (a : ValuationRing.ValueGroup A K), Eq (HMul.hMul 0 a) 0
                   -/
    zero_mul := by rintro ⟨a⟩; apply Quotient.sound'; rw [zero_mul]
                                                      /-
                                                        🎉 no goals
                                                      -/
                   /-
                     A : Type u
                     inst✝⁵ : CommRing A
                     K : Type v
                     inst✝⁴ : Field K
                     inst✝³ : Algebra A K
                     inst✝² : IsDomain A
                     inst✝¹ : ValuationRing A
                     inst✝ : IsFractionRing A K
                     ⊢ ∀ (a : ValuationRing.ValueGroup A K), Eq (HMul.hMul a 0) 0
                   -/
    mul_zero := by rintro ⟨a⟩; apply Quotient.sound'; rw [mul_zero]
                                                      /-
                                                        🎉 no goals
                                                      -/
                          /-
                            A : Type u
                            inst✝⁵ : CommRing A
                            K : Type v
                            inst✝⁴ : Field K
                            inst✝³ : Algebra A K
                            inst✝² : IsDomain A
                            inst✝¹ : ValuationRing A
                            inst✝ : IsFractionRing A K
                            ⊢ Eq (HSMul.hSMul 0 1) 0
                          -/
    zero_le_one := ⟨0, by rw [zero_smul]⟩
                          /-
                            🎉 no goals
                          -/
    exists_pair_ne := by
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        ⊢ Exists fun x => Exists fun y => Ne x y
      -/
      use 0, 1
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        ⊢ Ne 0 1
      -/
      intro c; obtain ⟨d, hd⟩ := Quotient.exact' c
      /-
        case h.intro
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        c : Eq 0 1
        d : Units A
        hd : Eq ((fun m => HSMul.hSMul m 1) d) 0
        ⊢ False
      -/
      apply_fun fun t => d⁻¹ • t at hd
      /-
        case h.intro
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        c : Eq 0 1
        d : Units A
        hd : Eq (HSMul.hSMul (Inv.inv d) ((fun m => HSMul.hSMul m 1) d)) (HSMul.hSMul  …
        ⊢ False
      -/
      simp only [inv_smul_smul, smul_zero, one_ne_zero] at hd
      /-
        🎉 no goals
      -/
                   /-
                     A : Type u
                     inst✝⁵ : CommRing A
                     K : Type v
                     inst✝⁴ : Field K
                     inst✝³ : Algebra A K
                     inst✝² : IsDomain A
                     inst✝¹ : ValuationRing A
                     inst✝ : IsFractionRing A K
                     ⊢ Eq (Inv.inv 0) 0
                   -/
    inv_zero := by apply Quotient.sound'; rw [inv_zero]
                                          /-
                                            🎉 no goals
                                          -/
    mul_inv_cancel := by
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        ⊢ ∀ (a : ValuationRing.ValueGroup A K), Ne a 0 → Eq (HMul.hMul a (Inv.inv a)) 1
      -/
      rintro ⟨a⟩ ha
      /-
        case mk
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Ne (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0
        ⊢ Eq (HMul.hMul (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) (Inv.inv (Quot …
      -/
      apply Quotient.sound'
      /-
        case mk.a
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Ne (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0
        ⊢ (MulAction.orbitRel (Units A) K) (HMul.hMul a (Inv.inv a)) 1
      -/
      use 1
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Ne (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0
        ⊢ Eq ((fun m => HSMul.hSMul m 1) 1) (HMul.hMul a (Inv.inv a))
      -/
      simp only [one_smul, ne_eq]
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Ne (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0
        ⊢ Eq 1 (HMul.hMul a (Inv.inv a))
      -/
      apply (mul_inv_cancel₀ _).symm
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Ne (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0
        ⊢ Ne a 0
      -/
      contrapose ha
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Not (Ne a 0)
        ⊢ Not (Ne (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0)
      -/
      simp only [Classical.not_not] at ha ⊢
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Eq a 0
        ⊢ Eq (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) a) 0
      -/
      rw [ha]
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        a✝ : ValuationRing.ValueGroup A K
        a : K
        ha : Eq a 0
        ⊢ Eq (Quot.mk (⇑(MulAction.orbitRel (Units A) K)) 0) 0
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- Any valuation ring induces a valuation on its fraction field. -/
def valuation : Valuation K (ValueGroup A K) where
  toFun := Quotient.mk''
  map_zero' := rfl
  map_one' := rfl
  map_mul' _ _ := rfl
  map_add_le_max' := by
    /-
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      ⊢ ∀ (x y : K), LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' :=  …
    -/
    intro a b
    /-
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      a b : K
      ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
    -/
    obtain ⟨xa, ya, hya, rfl⟩ : ∃ a b : A, _ := IsFractionRing.div_surjective a
    /-
      case intro.intro.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      b : K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
    -/
    obtain ⟨xb, yb, hyb, rfl⟩ : ∃ a b : A, _ := IsFractionRing.div_surjective b
    /-
      case intro.intro.intro.intro.intro.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
    -/
    have : (algebraMap A K) ya ≠ 0 := IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors hya
    /-
      case intro.intro.intro.intro.intro.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this : Ne ((algebraMap A K) ya) 0
      ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
    -/
    have : (algebraMap A K) yb ≠ 0 := IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors hyb
    /-
      case intro.intro.intro.intro.intro.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      xa ya : A
      hya : Membership.mem (nonZeroDivisors A) ya
      xb yb : A
      hyb : Membership.mem (nonZeroDivisors A) yb
      this✝ : Ne ((algebraMap A K) ya) 0
      this : Ne ((algebraMap A K) yb) 0
      ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
    -/
    obtain ⟨c, h | h⟩ := ValuationRing.cond (xa * yb) (xb * ya)
      /-
        case intro.intro.intro.intro.intro.intro.intro.inl
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
      -/
    · dsimp
      /-
        case intro.intro.intro.intro.intro.intro.intro.inl
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ LE.le (Quotient.mk'' (HAdd.hAdd (HDiv.hDiv ((algebraMap A K) xa) ((algebraMa …
      -/
      apply le_trans _ (le_max_left _ _)
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ LE.le (Quotient.mk'' (HAdd.hAdd (HDiv.hDiv ((algebraMap A K) xa) ((algebraMa …
      -/
      use c + 1
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd c 1) (HDiv.hDiv ((algebraMap A K) xa) ((algebraMa …
      -/
      rw [Algebra.smul_def]
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ Eq (HMul.hMul ((algebraMap A K) (HAdd.hAdd c 1)) (HDiv.hDiv ((algebraMap A K …
      -/
      field_simp
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd ((algebraMap A K) c) 1) ((algebraMap A K …
      -/
      simp only [← RingHom.map_mul, ← RingHom.map_add, ← (algebraMap A K).map_one, ← h]
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xa yb) c) (HMul.hMul xb ya)
        ⊢ Eq ((algebraMap A K) (HMul.hMul (HMul.hMul (HAdd.hAdd c 1) xa) (HMul.hMul ya …
      -/
      congr 1; ring
               /-
                 🎉 no goals
               -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.inr
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
        ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
      -/
    · apply le_trans _ (le_max_right _ _)
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
        ⊢ LE.le ((↑{ toFun := Quotient.mk'', map_zero' := ⋯, map_one' := ⋯, map_mul' : …
      -/
      use c + 1
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd c 1) (HDiv.hDiv ((algebraMap A K) xb) ((algebraMa …
      -/
      rw [Algebra.smul_def]
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
        ⊢ Eq (HMul.hMul ((algebraMap A K) (HAdd.hAdd c 1)) (HDiv.hDiv ((algebraMap A K …
      -/
      field_simp
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
        ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd ((algebraMap A K) c) 1) ((algebraMap A K …
      -/
      simp only [← RingHom.map_mul, ← RingHom.map_add, ← (algebraMap A K).map_one, ← h]
      /-
        case h
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        xa ya : A
        hya : Membership.mem (nonZeroDivisors A) ya
        xb yb : A
        hyb : Membership.mem (nonZeroDivisors A) yb
        this✝ : Ne ((algebraMap A K) ya) 0
        this : Ne ((algebraMap A K) yb) 0
        c : A
        h : Eq (HMul.hMul (HMul.hMul xb ya) c) (HMul.hMul xa yb)
        ⊢ Eq ((algebraMap A K) (HMul.hMul (HMul.hMul (HAdd.hAdd c 1) xb) (HMul.hMul ya …
      -/
      congr 1; ring
               /-
                 🎉 no goals
               -/


theorem mem_integer_iff (x : K) : x ∈ (valuation A K).integer ↔ ∃ a : A, algebraMap A K a = x := by
  /-
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    x : K
    ⊢ Iff (Membership.mem (ValuationRing.valuation A K).integer x) (Exists fun a = …
  -/
  constructor
    /-
      case mp
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      x : K
      ⊢ Membership.mem (ValuationRing.valuation A K).integer x → Exists fun a => Eq  …
    -/
  · rintro ⟨c, rfl⟩
    /-
      case mp.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      c : A
      ⊢ Exists fun a => Eq ((algebraMap A K) a) (HSMul.hSMul c 1)
    -/
    use c
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      c : A
      ⊢ Eq ((algebraMap A K) c) (HSMul.hSMul c 1)
    -/
    rw [Algebra.smul_def, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      x : K
      ⊢ (Exists fun a => Eq ((algebraMap A K) a) x) → Membership.mem (ValuationRing. …
    -/
  · rintro ⟨c, rfl⟩
    /-
      case mpr.intro
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      c : A
      ⊢ Membership.mem (ValuationRing.valuation A K).integer ((algebraMap A K) c)
    -/
    use c
    /-
      case h
      A : Type u
      inst✝⁵ : CommRing A
      K : Type v
      inst✝⁴ : Field K
      inst✝³ : Algebra A K
      inst✝² : IsDomain A
      inst✝¹ : ValuationRing A
      inst✝ : IsFractionRing A K
      c : A
      ⊢ Eq (HSMul.hSMul c 1) ((algebraMap A K) c)
    -/
    rw [Algebra.smul_def, mul_one]
    /-
      🎉 no goals
    -/


/-- The valuation ring `A` is isomorphic to the ring of integers of its associated valuation. -/
noncomputable def equivInteger : A ≃+* (valuation A K).integer :=
  RingEquiv.ofBijective
    (show A →ₙ+* (valuation A K).integer from
      { toFun := fun a => ⟨algebraMap A K a, (mem_integer_iff _ _ _).mpr ⟨a, rfl⟩⟩
                                  /-
                                    A : Type u
                                    inst✝⁵ : CommRing A
                                    K : Type v
                                    inst✝⁴ : Field K
                                    inst✝³ : Algebra A K
                                    inst✝² : IsDomain A
                                    inst✝¹ : ValuationRing A
                                    inst✝ : IsFractionRing A K
                                    x✝¹ x✝ : A
                                    ⊢ Eq ((fun a => ⟨(algebraMap A K) a, ⋯⟩) (HMul.hMul x✝¹ x✝)) (HMul.hMul ((fun  …
                                  -/
        map_mul' := fun _ _ => by ext1; exact (algebraMap A K).map_mul _ _
                                        /-
                                          🎉 no goals
                                        -/
                        /-
                          A : Type u
                          inst✝⁵ : CommRing A
                          K : Type v
                          inst✝⁴ : Field K
                          inst✝³ : Algebra A K
                          inst✝² : IsDomain A
                          inst✝¹ : ValuationRing A
                          inst✝ : IsFractionRing A K
                          ⊢ Eq ({ toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map_mul' := ⋯ }.toFun 0) 0
                        -/
        map_zero' := by ext1; exact (algebraMap A K).map_zero
                              /-
                                🎉 no goals
                              -/
                                  /-
                                    A : Type u
                                    inst✝⁵ : CommRing A
                                    K : Type v
                                    inst✝⁴ : Field K
                                    inst✝³ : Algebra A K
                                    inst✝² : IsDomain A
                                    inst✝¹ : ValuationRing A
                                    inst✝ : IsFractionRing A K
                                    x✝¹ x✝ : A
                                    ⊢ Eq ({ toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map_mul' := ⋯ }.toFun (HAdd …
                                  -/
        map_add' := fun _ _ => by ext1; exact (algebraMap A K).map_add _ _ })
                                        /-
                                          🎉 no goals
                                        -/
    (by
      /-
        A : Type u
        inst✝⁵ : CommRing A
        K : Type v
        inst✝⁴ : Field K
        inst✝³ : Algebra A K
        inst✝² : IsDomain A
        inst✝¹ : ValuationRing A
        inst✝ : IsFractionRing A K
        ⊢ Function.Bijective ⇑(letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map …
      -/
      constructor
        /-
          case left
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          ⊢ Function.Injective ⇑(letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map …
        -/
      · intro x y h
        /-
          case left
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          x y : A
          h : Eq ((letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map_mul' := ⋯, ma …
          ⊢ Eq x y
        -/
        apply_fun (algebraMap (valuation A K).integer K) at h
        /-
          case left
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          x y : A
          h : Eq ((algebraMap (Subtype fun x => Membership.mem (ValuationRing.valuation  …
          ⊢ Eq x y
        -/
        exact IsFractionRing.injective _ _ h
        /-
          🎉 no goals
        -/
        /-
          case right
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          ⊢ Function.Surjective ⇑(letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, ma …
        -/
      · rintro ⟨-, ha⟩
        /-
          case right.mk
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          val✝ : K
          ha : Membership.mem (ValuationRing.valuation A K).integer val✝
          ⊢ Exists fun a => Eq ((letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map …
        -/
        rw [mem_integer_iff] at ha
        /-
          case right.mk
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          val✝ : K
          ha✝ : Membership.mem (ValuationRing.valuation A K).integer val✝
          ha : Exists fun a => Eq ((algebraMap A K) a) val✝
          ⊢ Exists fun a => Eq ((letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, map …
        -/
        obtain ⟨a, rfl⟩ := ha
        /-
          case right.mk.intro
          A : Type u
          inst✝⁵ : CommRing A
          K : Type v
          inst✝⁴ : Field K
          inst✝³ : Algebra A K
          inst✝² : IsDomain A
          inst✝¹ : ValuationRing A
          inst✝ : IsFractionRing A K
          a : A
          ha : Membership.mem (ValuationRing.valuation A K).integer ((algebraMap A K) a)
          ⊢ Exists fun a_1 => Eq ((letFun { toFun := fun a => ⟨(algebraMap A K) a, ⋯⟩, m …
        -/
        exact ⟨a, rfl⟩)
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_equivInteger_apply (a : A) : (equivInteger A K a : K) = algebraMap A K a := rfl


theorem range_algebraMap_eq : (valuation A K).integer = (algebraMap A K).range := by
  /-
    A : Type u
    inst✝⁵ : CommRing A
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra A K
    inst✝² : IsDomain A
    inst✝¹ : ValuationRing A
    inst✝ : IsFractionRing A K
    ⊢ Eq (ValuationRing.valuation A K).integer (algebraMap A K).range
  -/
  ext; exact mem_integer_iff _ _ _
       /-
         🎉 no goals
       -/


instance (priority := 100) isLocalRing : IsLocalRing A :=
  IsLocalRing.of_isUnit_or_isUnit_one_sub_self
    (by
      /-
        A : Type u
        inst✝² : CommRing A
        inst✝¹ : Nontrivial A
        inst✝ : PreValuationRing A
        ⊢ ∀ (a : A), Or (IsUnit a) (IsUnit (HSub.hSub 1 a))
      -/
      intro a
      /-
        A : Type u
        inst✝² : CommRing A
        inst✝¹ : Nontrivial A
        inst✝ : PreValuationRing A
        a : A
        ⊢ Or (IsUnit a) (IsUnit (HSub.hSub 1 a))
      -/
      obtain ⟨c, h | h⟩ := PreValuationRing.cond a (1 - a)
        /-
          case intro.inl
          A : Type u
          inst✝² : CommRing A
          inst✝¹ : Nontrivial A
          inst✝ : PreValuationRing A
          a c : A
          h : Eq (HMul.hMul a c) (HSub.hSub 1 a)
          ⊢ Or (IsUnit a) (IsUnit (HSub.hSub 1 a))
        -/
      · left
        /-
          case intro.inl.h
          A : Type u
          inst✝² : CommRing A
          inst✝¹ : Nontrivial A
          inst✝ : PreValuationRing A
          a c : A
          h : Eq (HMul.hMul a c) (HSub.hSub 1 a)
          ⊢ IsUnit a
        -/
        apply isUnit_of_mul_eq_one _ (c + 1)
        /-
          case intro.inl.h
          A : Type u
          inst✝² : CommRing A
          inst✝¹ : Nontrivial A
          inst✝ : PreValuationRing A
          a c : A
          h : Eq (HMul.hMul a c) (HSub.hSub 1 a)
          ⊢ Eq (HMul.hMul a (HAdd.hAdd c 1)) 1
        -/
        simp [mul_add, h]
        /-
          🎉 no goals
        -/
        /-
          case intro.inr
          A : Type u
          inst✝² : CommRing A
          inst✝¹ : Nontrivial A
          inst✝ : PreValuationRing A
          a c : A
          h : Eq (HMul.hMul (HSub.hSub 1 a) c) a
          ⊢ Or (IsUnit a) (IsUnit (HSub.hSub 1 a))
        -/
      · right
        /-
          case intro.inr.h
          A : Type u
          inst✝² : CommRing A
          inst✝¹ : Nontrivial A
          inst✝ : PreValuationRing A
          a c : A
          h : Eq (HMul.hMul (HSub.hSub 1 a) c) a
          ⊢ IsUnit (HSub.hSub 1 a)
        -/
        apply isUnit_of_mul_eq_one _ (c + 1)
        /-
          case intro.inr.h
          A : Type u
          inst✝² : CommRing A
          inst✝¹ : Nontrivial A
          inst✝ : PreValuationRing A
          a c : A
          h : Eq (HMul.hMul (HSub.hSub 1 a) c) a
          ⊢ Eq (HMul.hMul (HSub.hSub 1 a) (HAdd.hAdd c 1)) 1
        -/
        simp [mul_add, h])
        /-
          🎉 no goals
        -/


instance le_total_ideal : IsTotal (Ideal A) LE.le := by
  /-
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    ⊢ IsTotal (Ideal A) LE.le
  -/
  constructor; intro α β
  /-
    case total
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    ⊢ Or (LE.le α β) (LE.le β α)
  -/
  by_cases h : α ≤ β; · exact Or.inl h
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    h : Not (LE.le α β)
    ⊢ Or (LE.le α β) (LE.le β α)
  -/
  erw [not_forall] at h
  /-
    case neg
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    h : Exists fun x => Not (Membership.mem α x → Membership.mem β x)
    ⊢ Or (LE.le α β) (LE.le β α)
  -/
  push_neg at h
  /-
    case neg
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    h : Exists fun x => And (Membership.mem α x) (Not (Membership.mem β x))
    ⊢ Or (LE.le α β) (LE.le β α)
  -/
  obtain ⟨a, h₁, h₂⟩ := h
  /-
    case neg.intro.intro
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    a : A
    h₁ : Membership.mem α a
    h₂ : Not (Membership.mem β a)
    ⊢ Or (LE.le α β) (LE.le β α)
  -/
  right
  /-
    case neg.intro.intro.h
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    a : A
    h₁ : Membership.mem α a
    h₂ : Not (Membership.mem β a)
    ⊢ LE.le β α
  -/
  intro b hb
  /-
    case neg.intro.intro.h
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : Nontrivial A
    inst✝ : PreValuationRing A
    α β : Ideal A
    a : A
    h₁ : Membership.mem α a
    h₂ : Not (Membership.mem β a)
    b : A
    hb : Membership.mem β b
    ⊢ Membership.mem α b
  -/
  obtain ⟨c, h | h⟩ := PreValuationRing.cond a b
    /-
      case neg.intro.intro.h.intro.inl
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : Nontrivial A
      inst✝ : PreValuationRing A
      α β : Ideal A
      a : A
      h₁ : Membership.mem α a
      h₂ : Not (Membership.mem β a)
      b : A
      hb : Membership.mem β b
      c : A
      h : Eq (HMul.hMul a c) b
      ⊢ Membership.mem α b
    -/
  · rw [← h]
    /-
      case neg.intro.intro.h.intro.inl
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : Nontrivial A
      inst✝ : PreValuationRing A
      α β : Ideal A
      a : A
      h₁ : Membership.mem α a
      h₂ : Not (Membership.mem β a)
      b : A
      hb : Membership.mem β b
      c : A
      h : Eq (HMul.hMul a c) b
      ⊢ Membership.mem α (HMul.hMul a c)
    -/
    exact Ideal.mul_mem_right _ _ h₁
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.h.intro.inr
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : Nontrivial A
      inst✝ : PreValuationRing A
      α β : Ideal A
      a : A
      h₁ : Membership.mem α a
      h₂ : Not (Membership.mem β a)
      b : A
      hb : Membership.mem β b
      c : A
      h : Eq (HMul.hMul b c) a
      ⊢ Membership.mem α b
    -/
  · exfalso; apply h₂; rw [← h]
    /-
      case neg.intro.intro.h.intro.inr
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : Nontrivial A
      inst✝ : PreValuationRing A
      α β : Ideal A
      a : A
      h₁ : Membership.mem α a
      h₂ : Not (Membership.mem β a)
      b : A
      hb : Membership.mem β b
      c : A
      h : Eq (HMul.hMul b c) a
      ⊢ Membership.mem β (HMul.hMul b c)
    -/
    apply Ideal.mul_mem_right _ _ hb
    /-
      🎉 no goals
    -/


instance [DecidableRel ((· ≤ ·) : Ideal A → Ideal A → Prop)] : LinearOrder (Ideal A) :=
  have := decidableEqOfDecidableLE (α := Ideal A)
  have := decidableLTOfDecidableLE (α := Ideal A)
  Lattice.toLinearOrder (Ideal A)


theorem _root_.PreValuationRing.iff_dvd_total [Monoid R] :
    PreValuationRing R ↔ IsTotal R (· ∣ ·) := by
  classical
  refine ⟨fun H => ⟨fun a b => ?_⟩, fun H => ⟨fun a b => ?_⟩⟩
  · obtain ⟨c, rfl | rfl⟩ := PreValuationRing.cond a b <;> simp
  · obtain ⟨c, rfl⟩ | ⟨c, rfl⟩ := @IsTotal.total _ _ H a b <;> use c <;> simp


theorem _root_.PreValuationRing.iff_ideal_total [CommRing R] :
    PreValuationRing R ↔ IsTotal (Ideal R) (· ≤ ·) := by
  classical
  refine ⟨fun _ => ⟨le_total⟩, fun H => PreValuationRing.iff_dvd_total.mpr ⟨fun a b => ?_⟩⟩
  have := @IsTotal.total _ _ H (Ideal.span {a}) (Ideal.span {b})
  simp_rw [Ideal.span_singleton_le_span_singleton] at this
  exact this.symm


theorem dvd_total [Monoid R] [h : PreValuationRing R] (x y : R) : x ∣ y ∨ y ∣ x :=
  @IsTotal.total _ _ (PreValuationRing.iff_dvd_total.mp h) x y


theorem iff_dvd_total : ValuationRing R ↔ IsTotal R (· ∣ ·) :=
  Iff.trans (⟨fun inst ↦ inst.toPreValuationRing, fun _ ↦ .mk⟩)
    PreValuationRing.iff_dvd_total


theorem iff_ideal_total : ValuationRing R ↔ IsTotal (Ideal R) (· ≤ ·) :=
  Iff.trans (⟨fun inst ↦ inst.toPreValuationRing, fun _ ↦ .mk⟩)
    PreValuationRing.iff_ideal_total


theorem unique_irreducible [ValuationRing R] ⦃p q : R⦄ (hp : Irreducible p) (hq : Irreducible q) :
    Associated p q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : ValuationRing R
    p q : R
    hp : Irreducible p
    hq : Irreducible q
    ⊢ Associated p q
  -/
  have := dvd_total p q
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : ValuationRing R
    p q : R
    hp : Irreducible p
    hq : Irreducible q
    this : Or (Dvd.dvd p q) (Dvd.dvd q p)
    ⊢ Associated p q
  -/
  rw [Irreducible.dvd_comm hp hq, or_self_iff] at this
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : ValuationRing R
    p q : R
    hp : Irreducible p
    hq : Irreducible q
    this : Dvd.dvd q p
    ⊢ Associated p q
  -/
  exact associated_of_dvd_dvd (Irreducible.dvd_symm hq hp this) this
  /-
    🎉 no goals
  -/


theorem iff_isInteger_or_isInteger :
    ValuationRing R ↔ ∀ x : K, IsLocalization.IsInteger R x ∨ IsLocalization.IsInteger R x⁻¹ := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    ⊢ Iff (ValuationRing R) (∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocal …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      ⊢ ValuationRing R → ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalizati …
    -/
  · intro H x
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ValuationRing R
      x : K
      ⊢ Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R (Inv.inv x))
    -/
    obtain ⟨x : R, y, hy, rfl⟩ := IsFractionRing.div_surjective (A := R) x
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ValuationRing R
      x y : R
      hy : Membership.mem (nonZeroDivisors R) y
      ⊢ Or (IsLocalization.IsInteger R (HDiv.hDiv ((algebraMap R K) x) ((algebraMap  …
    -/
    have := (map_ne_zero_iff _ (IsFractionRing.injective R K)).mpr (nonZeroDivisors.ne_zero hy)
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ValuationRing R
      x y : R
      hy : Membership.mem (nonZeroDivisors R) y
      this : Ne ((algebraMap R K) y) 0
      ⊢ Or (IsLocalization.IsInteger R (HDiv.hDiv ((algebraMap R K) x) ((algebraMap  …
    -/
    obtain ⟨s, rfl | rfl⟩ := ValuationRing.cond x y
    · exact Or.inr
        ⟨s, eq_inv_of_mul_eq_one_left <| by rwa [mul_div, div_eq_one_iff_eq, map_mul, mul_comm]⟩
      /-
        case mp.intro.intro.intro.intro.inr
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        H : ValuationRing R
        y : R
        hy : Membership.mem (nonZeroDivisors R) y
        this : Ne ((algebraMap R K) y) 0
        s : R
        ⊢ Or (IsLocalization.IsInteger R (HDiv.hDiv ((algebraMap R K) (HMul.hMul y s)) …
      -/
    · exact Or.inl ⟨s, by rwa [eq_div_iff, map_mul, mul_comm]⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      ⊢ (∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R (I …
    -/
  · intro H
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      ⊢ ValuationRing R
    -/
    suffices PreValuationRing R from mk
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      ⊢ PreValuationRing R
    -/
    constructor
    /-
      case mpr.cond'
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      ⊢ ∀ (a b : R), Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    intro a b
    /-
      case mpr.cond'
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      a b : R
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    by_cases ha : a = 0; · subst ha; exact ⟨0, Or.inr <| mul_zero b⟩
                                     /-
                                       🎉 no goals
                                     -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      a b : R
      ha : Not (Eq a 0)
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    by_cases hb : b = 0; · subst hb; exact ⟨0, Or.inl <| mul_zero a⟩
                                     /-
                                       🎉 no goals
                                     -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      a b : R
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    replace ha := (map_ne_zero_iff _ (IsFractionRing.injective R K)).mpr ha
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      a b : R
      hb : Not (Eq b 0)
      ha : Ne ((algebraMap R K) a) 0
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    replace hb := (map_ne_zero_iff _ (IsFractionRing.injective R K)).mpr hb
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
      a b : R
      ha : Ne ((algebraMap R K) a) 0
      hb : Ne ((algebraMap R K) b) 0
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    obtain ⟨c, e⟩ | ⟨c, e⟩ := H (algebraMap R K a / algebraMap R K b)
      /-
        case neg.inl.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
        a b : R
        ha : Ne ((algebraMap R K) a) 0
        hb : Ne ((algebraMap R K) b) 0
        c : R
        e : Eq ((algebraMap R K) c) (HDiv.hDiv ((algebraMap R K) a) ((algebraMap R K)  …
        ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
      -/
    · rw [eq_div_iff hb, ← map_mul, (IsFractionRing.injective R K).eq_iff, mul_comm] at e
      /-
        case neg.inl.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
        a b : R
        ha : Ne ((algebraMap R K) a) 0
        hb : Ne ((algebraMap R K) b) 0
        c : R
        e : Eq (HMul.hMul b c) a
        ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
      -/
      exact ⟨c, Or.inr e⟩
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
        a b : R
        ha : Ne ((algebraMap R K) a) 0
        hb : Ne ((algebraMap R K) b) 0
        c : R
        e : Eq ((algebraMap R K) c) (Inv.inv (HDiv.hDiv ((algebraMap R K) a) ((algebra …
        ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
      -/
    · rw [inv_div, eq_div_iff ha, ← map_mul, (IsFractionRing.injective R K).eq_iff, mul_comm c] at e
      /-
        case neg.inr.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        inst✝³ : IsDomain R
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        H : ∀ (x : K), Or (IsLocalization.IsInteger R x) (IsLocalization.IsInteger R ( …
        a b : R
        ha : Ne ((algebraMap R K) a) 0
        hb : Ne ((algebraMap R K) b) 0
        c : R
        e : Eq (HMul.hMul a c) b
        ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
      -/
      exact ⟨c, Or.inl e⟩
      /-
        🎉 no goals
      -/


theorem isInteger_or_isInteger [h : ValuationRing R] (x : K) :
    IsLocalization.IsInteger R x ∨ IsLocalization.IsInteger R x⁻¹ :=
  (iff_isInteger_or_isInteger R K).mp h x


instance (priority := 100) [ValuationRing R] : IsBezout R := by
  classical
  rw [IsBezout.iff_span_pair_isPrincipal]
  intro x y
  rw [Ideal.span_insert]
  rcases le_total (Ideal.span {x} : Ideal R) (Ideal.span {y}) with h | h
  · rw [sup_eq_right.mpr h]; exact ⟨⟨_, rfl⟩⟩
  · rw [sup_eq_left.mpr h]; exact ⟨⟨_, rfl⟩⟩


instance (priority := 100) [IsLocalRing R] [IsBezout R] : ValuationRing R := by
  classical
  refine iff_dvd_total.mpr ⟨fun a b => ?_⟩
  obtain ⟨g, e : _ = Ideal.span _⟩ := IsBezout.span_pair_isPrincipal a b
  obtain ⟨a, rfl⟩ := Ideal.mem_span_singleton'.mp
      (show a ∈ Ideal.span {g} by rw [← e]; exact Ideal.subset_span (by simp))
  obtain ⟨b, rfl⟩ := Ideal.mem_span_singleton'.mp
      (show b ∈ Ideal.span {g} by rw [← e]; exact Ideal.subset_span (by simp))
  obtain ⟨x, y, e'⟩ := Ideal.mem_span_pair.mp
      (show g ∈ Ideal.span {a * g, b * g} by rw [e]; exact Ideal.subset_span (by simp))
  rcases eq_or_ne g 0 with h | h
  · simp [h]
  have : x * a + y * b = 1 := by
    apply mul_left_injective₀ h; convert e' using 1 <;> ring
  cases' IsLocalRing.isUnit_or_isUnit_of_add_one this with h' h' <;> [left; right]
  all_goals exact mul_dvd_mul_right (isUnit_iff_forall_dvd.mp (isUnit_of_mul_isUnit_right h') _) _


theorem iff_local_bezout_domain : ValuationRing R ↔ IsLocalRing R ∧ IsBezout R :=
  ⟨fun _ ↦ ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ ↦ inferInstance⟩


protected theorem TFAE (R : Type u) [CommRing R] [IsDomain R] :
    List.TFAE
      [ValuationRing R,
        ∀ x : FractionRing R, IsLocalization.IsInteger R x ∨ IsLocalization.IsInteger R x⁻¹,
        IsTotal R (· ∣ ·), IsTotal (Ideal R) (· ≤ ·), IsLocalRing R ∧ IsBezout R] := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ (List.cons (ValuationRing R) (List.cons (∀ (x : FractionRing R), Or (IsLocal …
  -/
  tfae_have 1 ↔ 2 := iff_isInteger_or_isInteger R _
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    tfae_1_iff_2 : Iff (ValuationRing R) (∀ (x : FractionRing R), Or (IsLocalizati …
    ⊢ (List.cons (ValuationRing R) (List.cons (∀ (x : FractionRing R), Or (IsLocal …
  -/
  tfae_have 1 ↔ 3 := iff_dvd_total
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    tfae_1_iff_2 : Iff (ValuationRing R) (∀ (x : FractionRing R), Or (IsLocalizati …
    tfae_1_iff_3 : Iff (ValuationRing R) (IsTotal R fun x1 x2 => Dvd.dvd x1 x2)
    ⊢ (List.cons (ValuationRing R) (List.cons (∀ (x : FractionRing R), Or (IsLocal …
  -/
  tfae_have 1 ↔ 4 := iff_ideal_total
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    tfae_1_iff_2 : Iff (ValuationRing R) (∀ (x : FractionRing R), Or (IsLocalizati …
    tfae_1_iff_3 : Iff (ValuationRing R) (IsTotal R fun x1 x2 => Dvd.dvd x1 x2)
    tfae_1_iff_4 : Iff (ValuationRing R) (IsTotal (Ideal R) fun x1 x2 => LE.le x1  …
    ⊢ (List.cons (ValuationRing R) (List.cons (∀ (x : FractionRing R), Or (IsLocal …
  -/
  tfae_have 1 ↔ 5 := iff_local_bezout_domain
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    tfae_1_iff_2 : Iff (ValuationRing R) (∀ (x : FractionRing R), Or (IsLocalizati …
    tfae_1_iff_3 : Iff (ValuationRing R) (IsTotal R fun x1 x2 => Dvd.dvd x1 x2)
    tfae_1_iff_4 : Iff (ValuationRing R) (IsTotal (Ideal R) fun x1 x2 => LE.le x1  …
    tfae_1_iff_5 : Iff (ValuationRing R) (And (IsLocalRing R) (IsBezout R))
    ⊢ (List.cons (ValuationRing R) (List.cons (∀ (x : FractionRing R), Or (IsLocal …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem _root_.Function.Surjective.preValuationRing {R S : Type*} [Mul R] [PreValuationRing R]
    [Mul S] (f : R →ₙ* S) (hf : Function.Surjective f) :
    PreValuationRing S :=
  ⟨fun a b => by
    /-
      R : Type u_1
      S : Type u_2
      inst✝² : Mul R
      inst✝¹ : PreValuationRing R
      inst✝ : Mul S
      f : MulHom R S
      hf : Function.Surjective ⇑f
      a b : S
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    obtain ⟨⟨a, rfl⟩, ⟨b, rfl⟩⟩ := hf a, hf b
    /-
      case intro.intro
      R : Type u_1
      S : Type u_2
      inst✝² : Mul R
      inst✝¹ : PreValuationRing R
      inst✝ : Mul S
      f : MulHom R S
      hf : Function.Surjective ⇑f
      a b : R
      ⊢ Exists fun c => Or (Eq (HMul.hMul (f a) c) (f b)) (Eq (HMul.hMul (f b) c) (f …
    -/
    obtain ⟨c, rfl | rfl⟩ := PreValuationRing.cond a b
    /-
      case intro.intro.intro.inl
      R : Type u_1
      S : Type u_2
      inst✝² : Mul R
      inst✝¹ : PreValuationRing R
      inst✝ : Mul S
      f : MulHom R S
      hf : Function.Surjective ⇑f
      a c : R
      ⊢ Exists fun c_1 => Or (Eq (HMul.hMul (f a) c_1) (f (HMul.hMul a c))) (Eq (HMu …
    -/
    exacts [⟨f c, Or.inl <| (map_mul _ _ _).symm⟩, ⟨f c, Or.inr <| (map_mul _ _ _).symm⟩]⟩
    /-
      🎉 no goals
    -/


theorem _root_.Function.Surjective.valuationRing {R S : Type*} [CommRing R] [IsDomain R]
    [ValuationRing R] [CommRing S] [IsDomain S] (f : R →+* S) (hf : Function.Surjective f) :
    ValuationRing S :=
  have : PreValuationRing S := Function.Surjective.preValuationRing (R := R) f hf
  .mk


/-- If `𝒪` satisfies `v.integers 𝒪` where `v` is a valuation on a field, then `𝒪`
is a valuation ring. -/
theorem of_integers (v : Valuation K Γ) (hh : v.Integers 𝒪) : ValuationRing 𝒪 := by
  /-
    𝒪 : Type u
    K : Type v
    Γ : Type w
    inst✝⁴ : CommRing 𝒪
    inst✝³ : IsDomain 𝒪
    inst✝² : Field K
    inst✝¹ : Algebra 𝒪 K
    inst✝ : LinearOrderedCommGroupWithZero Γ
    v : Valuation K Γ
    hh : v.Integers 𝒪
    ⊢ ValuationRing 𝒪
  -/
  suffices PreValuationRing 𝒪 from .mk
  /-
    𝒪 : Type u
    K : Type v
    Γ : Type w
    inst✝⁴ : CommRing 𝒪
    inst✝³ : IsDomain 𝒪
    inst✝² : Field K
    inst✝¹ : Algebra 𝒪 K
    inst✝ : LinearOrderedCommGroupWithZero Γ
    v : Valuation K Γ
    hh : v.Integers 𝒪
    ⊢ PreValuationRing 𝒪
  -/
  constructor
  /-
    case cond'
    𝒪 : Type u
    K : Type v
    Γ : Type w
    inst✝⁴ : CommRing 𝒪
    inst✝³ : IsDomain 𝒪
    inst✝² : Field K
    inst✝¹ : Algebra 𝒪 K
    inst✝ : LinearOrderedCommGroupWithZero Γ
    v : Valuation K Γ
    hh : v.Integers 𝒪
    ⊢ ∀ (a b : 𝒪), Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
  -/
  intro a b
  /-
    case cond'
    𝒪 : Type u
    K : Type v
    Γ : Type w
    inst✝⁴ : CommRing 𝒪
    inst✝³ : IsDomain 𝒪
    inst✝² : Field K
    inst✝¹ : Algebra 𝒪 K
    inst✝ : LinearOrderedCommGroupWithZero Γ
    v : Valuation K Γ
    hh : v.Integers 𝒪
    a b : 𝒪
    ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
  -/
  rcases le_total (v (algebraMap 𝒪 K a)) (v (algebraMap 𝒪 K b)) with h | h
    /-
      case cond'.inl
      𝒪 : Type u
      K : Type v
      Γ : Type w
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : LinearOrderedCommGroupWithZero Γ
      v : Valuation K Γ
      hh : v.Integers 𝒪
      a b : 𝒪
      h : LE.le (v ((algebraMap 𝒪 K) a)) (v ((algebraMap 𝒪 K) b))
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
  · obtain ⟨c, hc⟩ := Valuation.Integers.dvd_of_le hh h
    /-
      case cond'.inl.intro
      𝒪 : Type u
      K : Type v
      Γ : Type w
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : LinearOrderedCommGroupWithZero Γ
      v : Valuation K Γ
      hh : v.Integers 𝒪
      a b : 𝒪
      h : LE.le (v ((algebraMap 𝒪 K) a)) (v ((algebraMap 𝒪 K) b))
      c : 𝒪
      hc : Eq a (HMul.hMul b c)
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    use c; exact Or.inr hc.symm
           /-
             🎉 no goals
           -/
    /-
      case cond'.inr
      𝒪 : Type u
      K : Type v
      Γ : Type w
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : LinearOrderedCommGroupWithZero Γ
      v : Valuation K Γ
      hh : v.Integers 𝒪
      a b : 𝒪
      h : LE.le (v ((algebraMap 𝒪 K) b)) (v ((algebraMap 𝒪 K) a))
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
  · obtain ⟨c, hc⟩ := Valuation.Integers.dvd_of_le hh h
    /-
      case cond'.inr.intro
      𝒪 : Type u
      K : Type v
      Γ : Type w
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : LinearOrderedCommGroupWithZero Γ
      v : Valuation K Γ
      hh : v.Integers 𝒪
      a b : 𝒪
      h : LE.le (v ((algebraMap 𝒪 K) b)) (v ((algebraMap 𝒪 K) a))
      c : 𝒪
      hc : Eq b (HMul.hMul a c)
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
    use c; exact Or.inl hc.symm
           /-
             🎉 no goals
           -/


instance instValuationRingInteger (v : Valuation K Γ) : ValuationRing v.integer :=
  of_integers (v := v) (Valuation.integer.integers v)


theorem isFractionRing_iff [ValuationRing 𝒪] :
    IsFractionRing 𝒪 K ↔
      (∀ (x : K), ∃ a : 𝒪, x = algebraMap 𝒪 K a ∨ x⁻¹ = algebraMap 𝒪 K a) ∧
        Function.Injective (algebraMap 𝒪 K) := by
  /-
    𝒪 : Type u
    K : Type v
    inst✝⁴ : CommRing 𝒪
    inst✝³ : IsDomain 𝒪
    inst✝² : Field K
    inst✝¹ : Algebra 𝒪 K
    inst✝ : ValuationRing 𝒪
    ⊢ Iff (IsFractionRing 𝒪 K) (And (∀ (x : K), Exists fun a => Or (Eq x ((algebra …
  -/
  refine ⟨fun h ↦ ⟨fun x ↦ ?_, IsFractionRing.injective _ _⟩, fun h ↦ ?_⟩
    /-
      case refine_1
      𝒪 : Type u
      K : Type v
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : ValuationRing 𝒪
      h : IsFractionRing 𝒪 K
      x : K
      ⊢ Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.inv x) ((algebraMap  …
    -/
  · obtain (⟨a, e⟩ | ⟨a, e⟩) := isInteger_or_isInteger 𝒪 x
    /-
      case refine_1.inl.intro
      𝒪 : Type u
      K : Type v
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : ValuationRing 𝒪
      h : IsFractionRing 𝒪 K
      x : K
      a : 𝒪
      e : Eq ((algebraMap 𝒪 K) a) x
      ⊢ Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.inv x) ((algebraMap  …
    -/
    exacts [⟨a, .inl e.symm⟩, ⟨a, .inr e.symm⟩]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝒪 : Type u
      K : Type v
      inst✝⁴ : CommRing 𝒪
      inst✝³ : IsDomain 𝒪
      inst✝² : Field K
      inst✝¹ : Algebra 𝒪 K
      inst✝ : ValuationRing 𝒪
      h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
      ⊢ IsFractionRing 𝒪 K
    -/
  · constructor
      /-
        case refine_2.map_units'
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        ⊢ ∀ (y : Subtype fun x => Membership.mem (nonZeroDivisors 𝒪) x), IsUnit ((alge …
      -/
    · intro a
      /-
        case refine_2.map_units'
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        a : Subtype fun x => Membership.mem (nonZeroDivisors 𝒪) x
        ⊢ IsUnit ((algebraMap 𝒪 K) ↑a)
      -/
      simpa using h.2.ne_iff.mpr (nonZeroDivisors.ne_zero a.2)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.surj'
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        ⊢ ∀ (z : K), Exists fun x => Eq (HMul.hMul z ((algebraMap 𝒪 K) ↑x.2)) ((algebr …
      -/
    · intro x
      /-
        case refine_2.surj'
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        x : K
        ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap 𝒪 K) ↑x_1.2)) ((algebraMap 𝒪  …
      -/
      obtain ⟨a, ha⟩ := h.1 x
      /-
        case refine_2.surj'.intro
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        x : K
        a : 𝒪
        ha : Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.inv x) ((algebraMap 𝒪 K) a))
        ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap 𝒪 K) ↑x_1.2)) ((algebraMap 𝒪  …
      -/
      by_cases h0 : a = 0
        /-
          case pos
          𝒪 : Type u
          K : Type v
          inst✝⁴ : CommRing 𝒪
          inst✝³ : IsDomain 𝒪
          inst✝² : Field K
          inst✝¹ : Algebra 𝒪 K
          inst✝ : ValuationRing 𝒪
          h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
          x : K
          a : 𝒪
          ha : Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.inv x) ((algebraMap 𝒪 K) a))
          h0 : Eq a 0
          ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap 𝒪 K) ↑x_1.2)) ((algebraMap 𝒪  …
        -/
      · exact ⟨⟨0, 1⟩, by simpa [h0] using ha⟩
        /-
          🎉 no goals
        -/
        /-
          case neg
          𝒪 : Type u
          K : Type v
          inst✝⁴ : CommRing 𝒪
          inst✝³ : IsDomain 𝒪
          inst✝² : Field K
          inst✝¹ : Algebra 𝒪 K
          inst✝ : ValuationRing 𝒪
          h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
          x : K
          a : 𝒪
          ha : Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.inv x) ((algebraMap 𝒪 K) a))
          h0 : Not (Eq a 0)
          ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap 𝒪 K) ↑x_1.2)) ((algebraMap 𝒪  …
        -/
      · have : algebraMap 𝒪 K a ≠ 0 := by simpa using h.2.ne_iff.mpr h0
        /-
          case neg
          𝒪 : Type u
          K : Type v
          inst✝⁴ : CommRing 𝒪
          inst✝³ : IsDomain 𝒪
          inst✝² : Field K
          inst✝¹ : Algebra 𝒪 K
          inst✝ : ValuationRing 𝒪
          h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
          x : K
          a : 𝒪
          ha : Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.inv x) ((algebraMap 𝒪 K) a))
          h0 : Not (Eq a 0)
          this : Ne ((algebraMap 𝒪 K) a) 0
          ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap 𝒪 K) ↑x_1.2)) ((algebraMap 𝒪  …
        -/
        rw [inv_eq_iff_eq_inv, ← one_div, eq_div_iff this] at ha
        cases ha with
        | inl ha => exact ⟨⟨a, 1⟩, by simpa⟩
        | inr ha => exact ⟨⟨1, ⟨a, mem_nonZeroDivisors_of_ne_zero h0⟩⟩, by simpa using ha⟩
      /-
        case refine_2.exists_of_eq
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        ⊢ ∀ {x y : 𝒪}, Eq ((algebraMap 𝒪 K) x) ((algebraMap 𝒪 K) y) → Exists fun c =>  …
      -/
    · intro _ _ hab
      /-
        case refine_2.exists_of_eq
        𝒪 : Type u
        K : Type v
        inst✝⁴ : CommRing 𝒪
        inst✝³ : IsDomain 𝒪
        inst✝² : Field K
        inst✝¹ : Algebra 𝒪 K
        inst✝ : ValuationRing 𝒪
        h : And (∀ (x : K), Exists fun a => Or (Eq x ((algebraMap 𝒪 K) a)) (Eq (Inv.in …
        x✝ y✝ : 𝒪
        hab : Eq ((algebraMap 𝒪 K) x✝) ((algebraMap 𝒪 K) y✝)
        ⊢ Exists fun c => Eq (HMul.hMul (↑c) x✝) (HMul.hMul (↑c) y✝)
      -/
      exact ⟨1, by simp only [OneMemClass.coe_one, h.2 hab, one_mul]⟩
      /-
        🎉 no goals
      -/


instance instIsFractionRingInteger (v : Valuation K Γ) : IsFractionRing v.integer K :=
  ValuationRing.isFractionRing_iff.mpr
    ⟨Valuation.Integers.eq_algebraMap_or_inv_eq_algebraMap (Valuation.integer.integers v),
    Subtype.coe_injective⟩


/-- A field is a valuation ring. -/
instance (priority := 100) of_field : ValuationRing K := inferInstance


