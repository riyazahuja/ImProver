/-- The type of Ring functions, to be used in the definition of the language of rings.
It contains the operations (+,*,-,0,1) -/
inductive ringFunc : ℕ → Type
  | add : ringFunc 2
  | mul : ringFunc 2
  | neg : ringFunc 1
  | zero : ringFunc 0
  | one : ringFunc 0
  deriving DecidableEq


/-- The language of rings contains the operations (+,*,-,0,1) -/
def Language.ring : Language :=
  { Functions := ringFunc
    Relations := fun _ => Empty }
  deriving IsAlgebraic


/-- `RingFunc.add`, but with the defeq type `Language.ring.Functions 2` instead
of `RingFunc 2` -/
abbrev addFunc : Language.ring.Functions 2 := add


/-- `RingFunc.mul`, but with the defeq type `Language.ring.Functions 2` instead
of `RingFunc 2` -/
abbrev mulFunc : Language.ring.Functions 2 := mul


/-- `RingFunc.neg`, but with the defeq type `Language.ring.Functions 1` instead
of `RingFunc 1` -/
abbrev negFunc : Language.ring.Functions 1 := neg


/-- `RingFunc.zero`, but with the defeq type `Language.ring.Functions 0` instead
of `RingFunc 0` -/
abbrev zeroFunc : Language.ring.Functions 0 := zero


/-- `RingFunc.one`, but with the defeq type `Language.ring.Functions 0` instead
of `RingFunc 0` -/
abbrev oneFunc : Language.ring.Functions 0 := one


instance (α : Type*) : Zero (Language.ring.Term α) :=
{ zero := Constants.term zeroFunc }


theorem zero_def (α : Type*) : (0 : Language.ring.Term α) = Constants.term zeroFunc := rfl


instance (α : Type*) : One (Language.ring.Term α) :=
{ one := Constants.term oneFunc }


theorem one_def (α : Type*) : (1 : Language.ring.Term α) = Constants.term oneFunc := rfl


instance (α : Type*) : Add (Language.ring.Term α) :=
{ add := addFunc.apply₂ }


theorem add_def (α : Type*) (t₁ t₂ : Language.ring.Term α) :
    t₁ + t₂ = addFunc.apply₂ t₁ t₂ := rfl


instance (α : Type*) : Mul (Language.ring.Term α) :=
{ mul := mulFunc.apply₂ }


theorem mul_def (α : Type*) (t₁ t₂ : Language.ring.Term α) :
    t₁ * t₂ = mulFunc.apply₂ t₁ t₂ := rfl


instance (α : Type*) : Neg (Language.ring.Term α) :=
{ neg := negFunc.apply₁ }


theorem neg_def (α : Type*) (t : Language.ring.Term α) :
    -t = negFunc.apply₁ t := rfl


instance : Fintype Language.ring.Symbols :=
  ⟨⟨Multiset.ofList
      [Sum.inl ⟨2, .add⟩,
       Sum.inl ⟨2, .mul⟩,
       Sum.inl ⟨1, .neg⟩,
       Sum.inl ⟨0, .zero⟩,
       Sum.inl ⟨0, .one⟩], by
    /-
      α : Type u_1
      ⊢ (↑(List.cons (Sum.inl ⟨2, FirstOrder.ringFunc.add⟩) (List.cons (Sum.inl ⟨2,  …
    -/
    dsimp [Language.Symbols]; decide⟩, by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      ⊢ ∀ (x : FirstOrder.Language.ring.Symbols), Membership.mem { val := ↑(List.con …
    -/
    intro x
    /-
      α : Type u_1
      x : FirstOrder.Language.ring.Symbols
      ⊢ Membership.mem { val := ↑(List.cons (Sum.inl ⟨2, FirstOrder.ringFunc.add⟩) ( …
    -/
    dsimp [Language.Symbols]
    /-
      α : Type u_1
      x : FirstOrder.Language.ring.Symbols
      ⊢ Membership.mem { val := ↑(List.cons (Sum.inl ⟨2, FirstOrder.ringFunc.add⟩) ( …
    -/
    rcases x with ⟨_, f⟩ | ⟨_, f⟩
      /-
        case inl.mk
        α : Type u_1
        fst✝ : Nat
        f : FirstOrder.Language.ring.Functions fst✝
        ⊢ Membership.mem { val := ↑(List.cons (Sum.inl ⟨2, FirstOrder.ringFunc.add⟩) ( …
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
    · cases f <;> decide
                  /-
                    🎉 no goals
                  -/
      /-
        case inr.mk
        α : Type u_1
        fst✝ : Nat
        f : FirstOrder.Language.ring.Relations fst✝
        ⊢ Membership.mem { val := ↑(List.cons (Sum.inl ⟨2, FirstOrder.ringFunc.add⟩) ( …
      -/
    · cases f ⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem card_ring : card Language.ring = 5 := by
  /-
    ⊢ Eq FirstOrder.Language.ring.card 5
  -/
  have : Fintype.card Language.ring.Symbols = 5 := rfl
  /-
    this : Eq (Fintype.card FirstOrder.Language.ring.Symbols) 5
    ⊢ Eq FirstOrder.Language.ring.card 5
  -/
  simp [Language.card, this]
  /-
    🎉 no goals
  -/


/-- A Type `R` is a `CompatibleRing` if it is a structure for the language of rings and this
structure is the same as the structure already given on `R` by the classes `Add`, `Mul` etc.

It is recommended to use this type class as a hypothesis to any theorem whose statement
requires a type to have be both a `Ring` (or `Field` etc.) and a
`Language.ring.Structure`  -/
/- This class does not extend `Add` etc, because this way it can be used in
combination with a `Ring`, or `Field` instance without having multiple different
`Add` structures on the Type. -/
class CompatibleRing (R : Type*) [Add R] [Mul R] [Neg R] [One R] [Zero R]
    extends Language.ring.Structure R where
  /-- Addition in the `Language.ring.Structure` is the same as the addition given by the
    `Add` instance -/
  funMap_add : ∀ x, funMap addFunc x = x 0 + x 1
  /-- Multiplication in the `Language.ring.Structure` is the same as the multiplication given by the
    `Mul` instance -/
  funMap_mul : ∀ x, funMap mulFunc x = x 0 * x 1
  /-- Negation in the `Language.ring.Structure` is the same as the negation given by the
    `Neg` instance -/
  funMap_neg : ∀ x, funMap negFunc x = -x 0
  /-- The constant `0` in the `Language.ring.Structure` is the same as the constant given by the
    `Zero` instance -/
  funMap_zero : ∀ x, funMap (zeroFunc : Language.ring.Constants) x = 0
  /-- The constant `1` in the `Language.ring.Structure` is the same as the constant given by the
    `One` instance -/
  funMap_one : ∀ x, funMap (oneFunc : Language.ring.Constants) x = 1


@[simp]
theorem realize_add (x y : ring.Term α) (v : α → R) :
    Term.realize v (x + y) = Term.realize v x + Term.realize v y := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝⁵ : Add R
    inst✝⁴ : Mul R
    inst✝³ : Neg R
    inst✝² : One R
    inst✝¹ : Zero R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    x y : FirstOrder.Language.ring.Term α
    v : α → R
    ⊢ Eq (FirstOrder.Language.Term.realize v (HAdd.hAdd x y)) (HAdd.hAdd (FirstOrd …
  -/
  simp [add_def, funMap_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_mul (x y : ring.Term α) (v : α → R) :
    Term.realize v (x * y) = Term.realize v x * Term.realize v y := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝⁵ : Add R
    inst✝⁴ : Mul R
    inst✝³ : Neg R
    inst✝² : One R
    inst✝¹ : Zero R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    x y : FirstOrder.Language.ring.Term α
    v : α → R
    ⊢ Eq (FirstOrder.Language.Term.realize v (HMul.hMul x y)) (HMul.hMul (FirstOrd …
  -/
  simp [mul_def, funMap_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_neg (x : ring.Term α) (v : α → R) :
    Term.realize v (-x) = -Term.realize v x := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝⁵ : Add R
    inst✝⁴ : Mul R
    inst✝³ : Neg R
    inst✝² : One R
    inst✝¹ : Zero R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    x : FirstOrder.Language.ring.Term α
    v : α → R
    ⊢ Eq (FirstOrder.Language.Term.realize v (Neg.neg x)) (Neg.neg (FirstOrder.Lan …
  -/
  simp [neg_def, funMap_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_zero (v : α → R) : Term.realize v (0 : ring.Term α) = 0 := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝⁵ : Add R
    inst✝⁴ : Mul R
    inst✝³ : Neg R
    inst✝² : One R
    inst✝¹ : Zero R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    v : α → R
    ⊢ Eq (FirstOrder.Language.Term.realize v 0) 0
  -/
  simp [zero_def, funMap_zero, constantMap]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_one (v : α → R) : Term.realize v (1 : ring.Term α) = 1 := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝⁵ : Add R
    inst✝⁴ : Mul R
    inst✝³ : Neg R
    inst✝² : One R
    inst✝¹ : Zero R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    v : α → R
    ⊢ Eq (FirstOrder.Language.Term.realize v 1) 1
  -/
  simp [one_def, funMap_one, constantMap]
  /-
    🎉 no goals
  -/


/-- Given a Type `R` with instances for each of the `Ring` operations, make a
`Language.ring.Structure R` instance, along with a proof that the operations given
by the `Language.ring.Structure` are the same as those given by the `Add` or `Mul` etc.
instances.

This definition can be used when applying a theorem about the model theory of rings
to a literal ring `R`, by writing `let _ := compatibleRingOfRing R`. After this, if,
for example, `R` is a field, then Lean will be able to find the instance for
`Theory.field.Model R`, and it will be possible to apply theorems about the model theory
of fields.

This is a `def` and not an `instance`, because the path
`Ring` => `Language.ring.Structure` => `Ring` cannot be made to
commute by definition
-/
def compatibleRingOfRing (R : Type*) [Add R] [Mul R] [Neg R] [One R] [Zero R] :
    CompatibleRing R :=
  { funMap := fun {n} f =>
      match n, f with
      | _, .add => fun x => x 0 + x 1
      | _, .mul => fun x => x 0 * x 1
      | _, .neg => fun x => -x 0
      | _, .zero => fun _ => 0
      | _, .one => fun _ => 1
    funMap_add := fun _ => rfl,
    funMap_mul := fun _ => rfl,
    funMap_neg := fun _ => rfl,
    funMap_zero := fun _ => rfl,
    funMap_one := fun _ => rfl }


/-- An isomorphism in the language of rings is a ring isomorphism -/
def languageEquivEquivRingEquiv {R S : Type*}
    [NonAssocRing R] [NonAssocRing S]
    [CompatibleRing R] [CompatibleRing S] :
    (Language.ring.Equiv R S) ≃ (R ≃+* S) :=
  { toFun := fun f =>
    { f with
      map_add' := by
        /-
          α : Type u_1
          R : Type u_2
          S : Type u_3
          inst✝³ : NonAssocRing R
          inst✝² : NonAssocRing S
          inst✝¹ : FirstOrder.Ring.CompatibleRing R
          inst✝ : FirstOrder.Ring.CompatibleRing S
          f : FirstOrder.Language.ring.Equiv R S
          ⊢ ∀ (x y : R), Eq (f.toFun (HAdd.hAdd x y)) (HAdd.hAdd (f.toFun x) (f.toFun y))
        -/
        intro x y
        /-
          α : Type u_1
          R : Type u_2
          S : Type u_3
          inst✝³ : NonAssocRing R
          inst✝² : NonAssocRing S
          inst✝¹ : FirstOrder.Ring.CompatibleRing R
          inst✝ : FirstOrder.Ring.CompatibleRing S
          f : FirstOrder.Language.ring.Equiv R S
          x y : R
          ⊢ Eq (f.toFun (HAdd.hAdd x y)) (HAdd.hAdd (f.toFun x) (f.toFun y))
        -/
        /-
          α : Type u_1
          R : Type u_2
          S : Type u_3
          inst✝³ : NonAssocRing R
          inst✝² : NonAssocRing S
          inst✝¹ : FirstOrder.Ring.CompatibleRing R
          inst✝ : FirstOrder.Ring.CompatibleRing S
          f : FirstOrder.Language.ring.Equiv R S
          ⊢ ∀ (x y : R), Eq (f.toFun (HMul.hMul x y)) (HMul.hMul (f.toFun x) (f.toFun y))
        -/
        simpa using f.map_fun addFunc ![x, y]
        /-
          α : Type u_1
          R : Type u_2
          S : Type u_3
          inst✝³ : NonAssocRing R
          inst✝² : NonAssocRing S
          inst✝¹ : FirstOrder.Ring.CompatibleRing R
          inst✝ : FirstOrder.Ring.CompatibleRing S
          f : FirstOrder.Language.ring.Equiv R S
          x y : R
          ⊢ Eq (f.toFun (HMul.hMul x y)) (HMul.hMul (f.toFun x) (f.toFun y))
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      map_mul' := by
        intro x y
        simpa using f.map_fun mulFunc ![x, y] }
    invFun := fun f =>
    { f with
      map_fun' := fun {n} f => by
        /-
          α : Type u_1
          R : Type u_2
          S : Type u_3
          inst✝³ : NonAssocRing R
          inst✝² : NonAssocRing S
          inst✝¹ : FirstOrder.Ring.CompatibleRing R
          inst✝ : FirstOrder.Ring.CompatibleRing S
          f✝ : RingEquiv R S
          n : Nat
          f : FirstOrder.Language.ring.Functions n
          ⊢ ∀ (x : Fin n → R), Eq (f✝.toFun (FirstOrder.Language.Structure.funMap f x))  …
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
        cases f <;> simp
                    /-
                      🎉 no goals
                    -/
                                  /-
                                    α : Type u_1
                                    R : Type u_2
                                    S : Type u_3
                                    inst✝³ : NonAssocRing R
                                    inst✝² : NonAssocRing S
                                    inst✝¹ : FirstOrder.Ring.CompatibleRing R
                                    inst✝ : FirstOrder.Ring.CompatibleRing S
                                    f✝ : RingEquiv R S
                                    n : Nat
                                    f : FirstOrder.Language.ring.Relations n
                                    ⊢ ∀ (x : Fin n → R), Iff (FirstOrder.Language.Structure.RelMap f (Function.com …
                                  -/
      map_rel' := fun {n} f => by cases f },
                                  /-
                                    🎉 no goals
                                  -/
                            /-
                              α : Type u_1
                              R : Type u_2
                              S : Type u_3
                              inst✝³ : NonAssocRing R
                              inst✝² : NonAssocRing S
                              inst✝¹ : FirstOrder.Ring.CompatibleRing R
                              inst✝ : FirstOrder.Ring.CompatibleRing S
                              f : FirstOrder.Language.ring.Equiv R S
                              ⊢ Eq ((fun f => { toEquiv := f.toEquiv, map_fun' := ⋯, map_rel' := ⋯ }) ((fun  …
                            -/
    left_inv := fun f => by ext; rfl
                                 /-
                                   🎉 no goals
                                 -/
                             /-
                               α : Type u_1
                               R : Type u_2
                               S : Type u_3
                               inst✝³ : NonAssocRing R
                               inst✝² : NonAssocRing S
                               inst✝¹ : FirstOrder.Ring.CompatibleRing R
                               inst✝ : FirstOrder.Ring.CompatibleRing S
                               f : RingEquiv R S
                               ⊢ Eq ((fun f => { toEquiv := f.toEquiv, map_mul' := ⋯, map_add' := ⋯ }) ((fun  …
                             -/
    right_inv := fun f => by ext; rfl }
                                  /-
                                    🎉 no goals
                                  -/


/-- A def to put an `Add` instance on a type with a `Language.ring.Structure` instance.

To be used sparingly, usually only when defining a more useful definition like,
`[Language.ring.Structure K] -> [Theory.field.Model K] -> Field K` -/
abbrev addOfRingStructure : Add R :=
  { add := fun x y => funMap addFunc ![x, y] }


/-- A def to put an `Mul` instance on a type with a `Language.ring.Structure` instance.

To be used sparingly, usually only when defining a more useful definition like,
`[Language.ring.Structure K] -> [Theory.field.Model K] -> Field K` -/
abbrev mulOfRingStructure : Mul R :=
  { mul := fun x y => funMap mulFunc ![x, y] }


/-- A def to put an `Neg` instance on a type with a `Language.ring.Structure` instance.

To be used sparingly, usually only when defining a more useful definition like,
`[Language.ring.Structure K] -> [Theory.field.Model K] -> Field K` -/
abbrev negOfRingStructure : Neg R :=
  { neg := fun x => funMap negFunc ![x] }


/-- A def to put an `Zero` instance on a type with a `Language.ring.Structure` instance.

To be used sparingly, usually only when defining a more useful definition like,
`[Language.ring.Structure K] -> [Theory.field.Model K] -> Field K` -/
abbrev zeroOfRingStructure : Zero R :=
  { zero := funMap zeroFunc ![] }


/-- A def to put an `One` instance on a type with a `Language.ring.Structure` instance.

To be used sparingly, usually only when defining a more useful definition like,
`[Language.ring.Structure K] -> [Theory.field.Model K] -> Field K` -/
abbrev oneOfRingStructure : One R :=
  { one := funMap oneFunc ![] }


/--
Given a Type `R` with a `Language.ring.Structure R`, the instance given by
`addOfRingStructure` etc are compatible with the `Language.ring.Structure` instance on `R`.

This definition is only to be used when `addOfRingStructure`, `mulOfRingStructure` etc
are local instances.
-/
abbrev compatibleRingOfRingStructure : CompatibleRing R :=
  { funMap_add := by
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (x : Fin 2 → R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring. …
      -/
      simp only [Fin.forall_fin_succ_pi, Fin.cons_zero, Fin.forall_fin_zero_pi]
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (a a_1 : R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring.addF …
      -/
      intros; rfl
              /-
                🎉 no goals
              -/
    funMap_mul := by
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (x : Fin 2 → R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring. …
      -/
      simp only [Fin.forall_fin_succ_pi, Fin.cons_zero, Fin.forall_fin_zero_pi]
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (a a_1 : R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring.mulF …
      -/
      intros; rfl
              /-
                🎉 no goals
              -/
    funMap_neg := by
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (x : Fin 1 → R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring. …
      -/
      simp only [Fin.forall_fin_succ_pi, Fin.cons_zero, Fin.forall_fin_zero_pi]
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (a : R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring.negFunc  …
      -/
      intros; rfl
              /-
                🎉 no goals
              -/
    funMap_zero := by
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (x : Fin 0 → R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring. …
      -/
      simp only [Fin.forall_fin_succ_pi, Fin.cons_zero, Fin.forall_fin_zero_pi]
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring.zeroFunc finZeroEli …
      -/
      rfl
      /-
        🎉 no goals
      -/
    funMap_one := by
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ ∀ (x : Fin 0 → R), Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring. …
      -/
      simp only [Fin.forall_fin_succ_pi, Fin.cons_zero, Fin.forall_fin_zero_pi]
      /-
        α : Type u_1
        R : Type u_2
        inst✝ : FirstOrder.Language.ring.Structure R
        ⊢ Eq (FirstOrder.Language.Structure.funMap FirstOrder.Ring.oneFunc finZeroElim …
      -/
      rfl  }
      /-
        🎉 no goals
      -/


