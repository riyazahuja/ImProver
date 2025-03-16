local macro:max "local_hAdd[" type:term ", " inst:term "]" : term =>
  `(term| (letI := $inst; HAdd.hAdd : $type → $type → $type))

local macro:max "local_hMul[" type:term ", " inst:term "]" : term =>
  `(term| (letI := $inst; HMul.hMul : $type → $type → $type))


@[ext] theorem ext ⦃inst₁ inst₂ : Distrib R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  -- Split into `add` and `mul` functions and properties.
  /-
    R : Type u
    inst₁ inst₂ : Distrib R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq inst₁ inst₂
  -/
  rcases inst₁ with @⟨⟨⟩, ⟨⟩⟩
  /-
    case mk.mk.mk
    R : Type u
    inst₂ : Distrib R
    mul✝ add✝ : R → R → R
    left_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HM …
    right_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (H …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq (Distrib.mk left_distrib✝ right_distrib✝) inst₂
  -/
  rcases inst₂ with @⟨⟨⟩, ⟨⟩⟩
  -- Prove equality of parts using function extensionality.
  /-
    case mk.mk.mk.mk.mk.mk
    R : Type u
    mul✝¹ add✝¹ : R → R → R
    left_distrib✝¹ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (H …
    right_distrib✝¹ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd ( …
    mul✝ add✝ : R → R → R
    left_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HM …
    right_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (H …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq (Distrib.mk left_distrib✝¹ right_distrib✝¹) (Distrib.mk left_distrib✝ rig …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalNonAssocSemiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  -- Split into `AddMonoid` instance, `mul` function and properties.
  /-
    R : Type u
    inst₁ inst₂ : NonUnitalNonAssocSemiring R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq inst₁ inst₂
  -/
  rcases inst₁ with @⟨_, ⟨⟩⟩
  /-
    case mk.mk
    R : Type u
    inst₂ : NonUnitalNonAssocSemiring R
    toAddCommMonoid✝ : AddCommMonoid R
    mul✝ : R → R → R
    left_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HM …
    right_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (H …
    zero_mul✝ : ∀ (a : R), Eq (HMul.hMul 0 a) 0
    mul_zero✝ : ∀ (a : R), Eq (HMul.hMul a 0) 0
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq (NonUnitalNonAssocSemiring.mk left_distrib✝ right_distrib✝ zero_mul✝ mul_ …
  -/
  rcases inst₂ with @⟨_, ⟨⟩⟩
  -- Prove equality of parts using already-proved extensionality lemmas.
  /-
    case mk.mk.mk.mk
    R : Type u
    toAddCommMonoid✝¹ : AddCommMonoid R
    mul✝¹ : R → R → R
    left_distrib✝¹ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (H …
    right_distrib✝¹ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd ( …
    zero_mul✝¹ : ∀ (a : R), Eq (HMul.hMul 0 a) 0
    mul_zero✝¹ : ∀ (a : R), Eq (HMul.hMul a 0) 0
    toAddCommMonoid✝ : AddCommMonoid R
    mul✝ : R → R → R
    left_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HM …
    right_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (H …
    zero_mul✝ : ∀ (a : R), Eq (HMul.hMul 0 a) 0
    mul_zero✝ : ∀ (a : R), Eq (HMul.hMul a 0) 0
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq (NonUnitalNonAssocSemiring.mk left_distrib✝¹ right_distrib✝¹ zero_mul✝¹ m …
  -/
  congr; ext : 1; assumption
                  /-
                    🎉 no goals
                  -/


theorem toDistrib_injective : Function.Injective (@toDistrib R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalNonAssocSemiring.toDistrib R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : NonUnitalNonAssocSemiring R
    h : Eq NonUnitalNonAssocSemiring.toDistrib NonUnitalNonAssocSemiring.toDistrib
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : NonUnitalNonAssocSemiring R
      h : Eq NonUnitalNonAssocSemiring.toDistrib NonUnitalNonAssocSemiring.toDistrib
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : NonUnitalNonAssocSemiring R
      h : Eq NonUnitalNonAssocSemiring.toDistrib NonUnitalNonAssocSemiring.toDistrib
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


theorem toNonUnitalNonAssocSemiring_injective :
    Function.Injective (@toNonUnitalNonAssocSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalSemiring.toNonUnitalNonAssocSemiring R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalSemiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toNonUnitalNonAssocSemiring_injective <|
    NonUnitalNonAssocSemiring.ext h_add h_mul


@[ext] theorem AddMonoidWithOne.ext ⦃inst₁ inst₂ : AddMonoidWithOne R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_one : (letI := inst₁; One.one : R) = (letI := inst₂; One.one : R)) :
    inst₁ = inst₂ := by
  /-
    R : Type u
    inst₁ inst₂ : AddMonoidWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    ⊢ Eq inst₁ inst₂
  -/
  have h_monoid : inst₁.toAddMonoid = inst₂.toAddMonoid := by ext : 1; exact h_add
  /-
    R : Type u
    inst₁ inst₂ : AddMonoidWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    h_monoid : Eq AddMonoidWithOne.toAddMonoid AddMonoidWithOne.toAddMonoid
    ⊢ Eq inst₁ inst₂
  -/
  have h_zero' : inst₁.toZero = inst₂.toZero := congrArg (·.toZero) h_monoid
  have h_one' : inst₁.toOne = inst₂.toOne :=
    congrArg One.mk h_one
  have h_natCast : inst₁.toNatCast.natCast = inst₂.toNatCast.natCast := by
    funext n; induction n with
    | zero     => rewrite [inst₁.natCast_zero, inst₂.natCast_zero]
                  exact congrArg (@Zero.zero R) h_zero'
    | succ n h => rw [inst₁.natCast_succ, inst₂.natCast_succ, h_add]
                  exact congrArg₂ _ h h_one
  /-
    R : Type u
    inst₁ inst₂ : AddMonoidWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    h_monoid : Eq AddMonoidWithOne.toAddMonoid AddMonoidWithOne.toAddMonoid
    h_zero' : Eq AddMonoid.toZero AddMonoid.toZero
    h_one' : Eq AddMonoidWithOne.toOne AddMonoidWithOne.toOne
    h_natCast : Eq NatCast.natCast NatCast.natCast
    ⊢ Eq inst₁ inst₂
  -/
  rcases inst₁ with @⟨⟨⟩⟩; rcases inst₂ with @⟨⟨⟩⟩
  /-
    case mk.mk.mk.mk
    R : Type u
    toAddMonoid✝¹ : AddMonoid R
    toOne✝¹ : One R
    natCast✝¹ : Nat → R
    natCast_zero✝¹ : Eq (NatCast.natCast 0) 0
    natCast_succ✝¹ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd  …
    toAddMonoid✝ : AddMonoid R
    toOne✝ : One R
    natCast✝ : Nat → R
    natCast_zero✝ : Eq (NatCast.natCast 0) 0
    natCast_succ✝ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd ( …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    h_monoid : Eq AddMonoidWithOne.toAddMonoid AddMonoidWithOne.toAddMonoid
    h_zero' : Eq AddMonoid.toZero AddMonoid.toZero
    h_one' : Eq AddMonoidWithOne.toOne AddMonoidWithOne.toOne
    h_natCast : Eq NatCast.natCast NatCast.natCast
    ⊢ Eq (AddMonoidWithOne.mk natCast_zero✝¹ natCast_succ✝¹) (AddMonoidWithOne.mk  …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem AddCommMonoidWithOne.toAddMonoidWithOne_injective :
    Function.Injective (@AddCommMonoidWithOne.toAddMonoidWithOne R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@AddCommMonoidWithOne.toAddMonoidWithOne R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem AddCommMonoidWithOne.ext ⦃inst₁ inst₂ : AddCommMonoidWithOne R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_one : (letI := inst₁; One.one : R) = (letI := inst₂; One.one : R)) :
    inst₁ = inst₂ :=
  AddCommMonoidWithOne.toAddMonoidWithOne_injective <|
    AddMonoidWithOne.ext h_add h_one


@[ext] theorem ext ⦃inst₁ inst₂ : NonAssocSemiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  have h : inst₁.toNonUnitalNonAssocSemiring = inst₂.toNonUnitalNonAssocSemiring := by
    ext : 1 <;> assumption
  have h_zero : (inst₁.toMulZeroClass).toZero.zero = (inst₂.toMulZeroClass).toZero.zero :=
    congrArg (fun inst => (inst.toMulZeroClass).toZero.zero) h
  have h_one' : (inst₁.toMulZeroOneClass).toMulOneClass.toOne
                = (inst₂.toMulZeroOneClass).toMulOneClass.toOne :=
    congrArg (@MulOneClass.toOne R) <| by ext : 1; exact h_mul
  have h_one : (inst₁.toMulZeroOneClass).toMulOneClass.toOne.one
               = (inst₂.toMulZeroOneClass).toMulOneClass.toOne.one :=
    congrArg (@One.one R) h_one'
  have : inst₁.toAddCommMonoidWithOne = inst₂.toAddCommMonoidWithOne := by
    ext : 1 <;> assumption
  have : inst₁.toNatCast = inst₂.toNatCast :=
    congrArg (·.toNatCast) this
  -- Split into `NonUnitalNonAssocSemiring`, `One` and `natCast` instances.
  /-
    R : Type u
    inst₁ inst₂ : NonAssocSemiring R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h : Eq NonAssocSemiring.toNonUnitalNonAssocSemiring NonAssocSemiring.toNonUnit …
    h_zero : Eq Zero.zero Zero.zero
    h_one' : Eq MulOneClass.toOne MulOneClass.toOne
    h_one : Eq One.one One.one
    this✝ : Eq NonAssocSemiring.toAddCommMonoidWithOne NonAssocSemiring.toAddCommM …
    this : Eq NonAssocSemiring.toNatCast NonAssocSemiring.toNatCast
    ⊢ Eq inst₁ inst₂
  -/
  cases inst₁; cases inst₂
  /-
    case mk.mk
    R : Type u
    toNonUnitalNonAssocSemiring✝¹ : NonUnitalNonAssocSemiring R
    toOne✝¹ : One R
    one_mul✝¹ : ∀ (a : R), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : R), Eq (HMul.hMul a 1) a
    toNatCast✝¹ : NatCast R
    natCast_zero✝¹ : Eq (NatCast.natCast 0) 0
    natCast_succ✝¹ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd  …
    toNonUnitalNonAssocSemiring✝ : NonUnitalNonAssocSemiring R
    toOne✝ : One R
    one_mul✝ : ∀ (a : R), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : R), Eq (HMul.hMul a 1) a
    toNatCast✝ : NatCast R
    natCast_zero✝ : Eq (NatCast.natCast 0) 0
    natCast_succ✝ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd ( …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h : Eq NonAssocSemiring.toNonUnitalNonAssocSemiring NonAssocSemiring.toNonUnit …
    h_zero : Eq Zero.zero Zero.zero
    h_one' : Eq MulOneClass.toOne MulOneClass.toOne
    h_one : Eq One.one One.one
    this✝ : Eq NonAssocSemiring.toAddCommMonoidWithOne NonAssocSemiring.toAddCommM …
    this : Eq NonAssocSemiring.toNatCast NonAssocSemiring.toNatCast
    ⊢ Eq (NonAssocSemiring.mk one_mul✝¹ mul_one✝¹ natCast_zero✝¹ natCast_succ✝¹) ( …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem toNonUnitalNonAssocSemiring_injective :
    Function.Injective (@toNonUnitalNonAssocSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonAssocSemiring.toNonUnitalNonAssocSemiring R)
  -/
  intro _ _ _
  /-
    R : Type u
    a₁✝ a₂✝ : NonAssocSemiring R
    a✝ : Eq NonAssocSemiring.toNonUnitalNonAssocSemiring NonAssocSemiring.toNonUni …
    ⊢ Eq a₁✝ a₂✝
  -/
          /-
            🎉 no goals
          -/
  ext <;> congr
          /-
            🎉 no goals
          -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalNonAssocRing R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  -- Split into `AddCommGroup` instance, `mul` function and properties.
  /-
    R : Type u
    inst₁ inst₂ : NonUnitalNonAssocRing R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq inst₁ inst₂
  -/
  rcases inst₁ with @⟨_, ⟨⟩⟩; rcases inst₂ with @⟨_, ⟨⟩⟩
  /-
    case mk.mk.mk.mk
    R : Type u
    toAddCommGroup✝¹ : AddCommGroup R
    mul✝¹ : R → R → R
    left_distrib✝¹ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (H …
    right_distrib✝¹ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd ( …
    zero_mul✝¹ : ∀ (a : R), Eq (HMul.hMul 0 a) 0
    mul_zero✝¹ : ∀ (a : R), Eq (HMul.hMul a 0) 0
    toAddCommGroup✝ : AddCommGroup R
    mul✝ : R → R → R
    left_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HM …
    right_distrib✝ : ∀ (a b c : R), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (H …
    zero_mul✝ : ∀ (a : R), Eq (HMul.hMul 0 a) 0
    mul_zero✝ : ∀ (a : R), Eq (HMul.hMul a 0) 0
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq (NonUnitalNonAssocRing.mk left_distrib✝¹ right_distrib✝¹ zero_mul✝¹ mul_z …
  -/
  congr; (ext : 1; assumption)
                   /-
                     🎉 no goals
                   -/


theorem toNonUnitalNonAssocSemiring_injective :
    Function.Injective (@toNonUnitalNonAssocSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalNonAssocRing.toNonUnitalNonAssocSemiring R)
  -/
  intro _ _ h
  -- Use above extensionality lemma to prove injectivity by showing that `h_add` and `h_mul` hold.
  /-
    R : Type u
    a₁✝ a₂✝ : NonUnitalNonAssocRing R
    h : Eq NonUnitalNonAssocRing.toNonUnitalNonAssocSemiring NonUnitalNonAssocRing …
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : NonUnitalNonAssocRing R
      h : Eq NonUnitalNonAssocRing.toNonUnitalNonAssocSemiring NonUnitalNonAssocRing …
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : NonUnitalNonAssocRing R
      h : Eq NonUnitalNonAssocRing.toNonUnitalNonAssocSemiring NonUnitalNonAssocRing …
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalRing R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  have : inst₁.toNonUnitalNonAssocRing = inst₂.toNonUnitalNonAssocRing := by
    ext : 1 <;> assumption
  -- Split into fields and prove they are equal using the above.
  /-
    R : Type u
    inst₁ inst₂ : NonUnitalRing R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    this : Eq NonUnitalRing.toNonUnitalNonAssocRing NonUnitalRing.toNonUnitalNonAs …
    ⊢ Eq inst₁ inst₂
  -/
  cases inst₁; cases inst₂
  /-
    case mk.mk
    R : Type u
    toNonUnitalNonAssocRing✝¹ : NonUnitalNonAssocRing R
    mul_assoc✝¹ : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HM …
    toNonUnitalNonAssocRing✝ : NonUnitalNonAssocRing R
    mul_assoc✝ : ∀ (a b c : R), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMu …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    this : Eq NonUnitalRing.toNonUnitalNonAssocRing NonUnitalRing.toNonUnitalNonAs …
    ⊢ Eq (NonUnitalRing.mk mul_assoc✝¹) (NonUnitalRing.mk mul_assoc✝)
  -/
  congr
  /-
    🎉 no goals
  -/


theorem toNonUnitalSemiring_injective :
    Function.Injective (@toNonUnitalSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalRing.toNonUnitalSemiring R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : NonUnitalRing R
    h : Eq NonUnitalRing.toNonUnitalSemiring NonUnitalRing.toNonUnitalSemiring
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : NonUnitalRing R
      h : Eq NonUnitalRing.toNonUnitalSemiring NonUnitalRing.toNonUnitalSemiring
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : NonUnitalRing R
      h : Eq NonUnitalRing.toNonUnitalSemiring NonUnitalRing.toNonUnitalSemiring
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


theorem toNonUnitalNonAssocring_injective :
    Function.Injective (@toNonUnitalNonAssocRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalRing.toNonUnitalNonAssocRing R)
  -/
  intro _ _ _
  /-
    R : Type u
    a₁✝ a₂✝ : NonUnitalRing R
    a✝ : Eq NonUnitalRing.toNonUnitalNonAssocRing NonUnitalRing.toNonUnitalNonAsso …
    ⊢ Eq a₁✝ a₂✝
  -/
          /-
            🎉 no goals
          -/
  ext <;> congr
          /-
            🎉 no goals
          -/


@[ext] theorem AddGroupWithOne.ext ⦃inst₁ inst₂ : AddGroupWithOne R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_one : (letI := inst₁; One.one : R) = (letI := inst₂; One.one)) :
    inst₁ = inst₂ := by
  have : inst₁.toAddMonoidWithOne = inst₂.toAddMonoidWithOne :=
    AddMonoidWithOne.ext h_add h_one
  /-
    R : Type u
    inst₁ inst₂ : AddGroupWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this : Eq AddGroupWithOne.toAddMonoidWithOne AddGroupWithOne.toAddMonoidWithOne
    ⊢ Eq inst₁ inst₂
  -/
  have : inst₁.toNatCast = inst₂.toNatCast := congrArg (·.toNatCast) this
  /-
    R : Type u
    inst₁ inst₂ : AddGroupWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this✝ : Eq AddGroupWithOne.toAddMonoidWithOne AddGroupWithOne.toAddMonoidWithOne
    this : Eq AddMonoidWithOne.toNatCast AddMonoidWithOne.toNatCast
    ⊢ Eq inst₁ inst₂
  -/
  have h_group : inst₁.toAddGroup = inst₂.toAddGroup := by ext : 1; exact h_add
  -- Extract equality of necessary substructures from h_group
  /-
    R : Type u
    inst₁ inst₂ : AddGroupWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this✝ : Eq AddGroupWithOne.toAddMonoidWithOne AddGroupWithOne.toAddMonoidWithOne
    this : Eq AddMonoidWithOne.toNatCast AddMonoidWithOne.toNatCast
    h_group : Eq AddGroupWithOne.toAddGroup AddGroupWithOne.toAddGroup
    ⊢ Eq inst₁ inst₂
  -/
  injection h_group with h_group; injection h_group
  have : inst₁.toIntCast.intCast = inst₂.toIntCast.intCast := by
    funext n; cases n with
    | ofNat n   => rewrite [Int.ofNat_eq_coe, inst₁.intCast_ofNat, inst₂.intCast_ofNat]; congr
    | negSucc n => rewrite [inst₁.intCast_negSucc, inst₂.intCast_negSucc]; congr
  /-
    R : Type u
    inst₁ inst₂ : AddGroupWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this✝¹ : Eq AddGroupWithOne.toAddMonoidWithOne AddGroupWithOne.toAddMonoidWith …
    this✝ : Eq AddMonoidWithOne.toNatCast AddMonoidWithOne.toNatCast
    toAddMonoid_eq✝ : Eq AddMonoidWithOne.toAddMonoid AddMonoidWithOne.toAddMonoid
    toNeg_eq✝ : Eq AddGroupWithOne.toNeg AddGroupWithOne.toNeg
    toSub_eq✝ : Eq AddGroupWithOne.toSub AddGroupWithOne.toSub
    zsmul_eq✝ : Eq AddGroupWithOne.zsmul AddGroupWithOne.zsmul
    this : Eq IntCast.intCast IntCast.intCast
    ⊢ Eq inst₁ inst₂
  -/
  rcases inst₁ with @⟨⟨⟩⟩; rcases inst₂ with @⟨⟨⟩⟩
  /-
    case mk.mk.mk.mk
    R : Type u
    toAddMonoidWithOne✝¹ : AddMonoidWithOne R
    toNeg✝¹ : Neg R
    toSub✝¹ : Sub R
    sub_eq_add_neg✝¹ : ∀ (a b : R), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
    zsmul✝¹ : Int → R → R
    zsmul_zero'✝¹ : ∀ (a : R), Eq (zsmul✝¹ 0 a) 0
    zsmul_succ'✝¹ : ∀ (n : Nat) (a : R), Eq (zsmul✝¹ (↑n.succ) a) (HAdd.hAdd (zsmu …
    zsmul_neg'✝¹ : ∀ (n : Nat) (a : R), Eq (zsmul✝¹ (Int.negSucc n) a) (Neg.neg (z …
    neg_add_cancel✝¹ : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
    intCast✝¹ : Int → R
    intCast_ofNat✝¹ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝¹ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg …
    toAddMonoidWithOne✝ : AddMonoidWithOne R
    toNeg✝ : Neg R
    toSub✝ : Sub R
    sub_eq_add_neg✝ : ∀ (a b : R), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
    zsmul✝ : Int → R → R
    zsmul_zero'✝ : ∀ (a : R), Eq (zsmul✝ 0 a) 0
    zsmul_succ'✝ : ∀ (n : Nat) (a : R), Eq (zsmul✝ (↑n.succ) a) (HAdd.hAdd (zsmul✝ …
    zsmul_neg'✝ : ∀ (n : Nat) (a : R), Eq (zsmul✝ (Int.negSucc n) a) (Neg.neg (zsm …
    neg_add_cancel✝ : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
    intCast✝ : Int → R
    intCast_ofNat✝ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg  …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this✝¹ : Eq AddGroupWithOne.toAddMonoidWithOne AddGroupWithOne.toAddMonoidWith …
    this✝ : Eq AddMonoidWithOne.toNatCast AddMonoidWithOne.toNatCast
    toAddMonoid_eq✝ : Eq AddMonoidWithOne.toAddMonoid AddMonoidWithOne.toAddMonoid
    toNeg_eq✝ : Eq AddGroupWithOne.toNeg AddGroupWithOne.toNeg
    toSub_eq✝ : Eq AddGroupWithOne.toSub AddGroupWithOne.toSub
    zsmul_eq✝ : Eq AddGroupWithOne.zsmul AddGroupWithOne.zsmul
    this : Eq IntCast.intCast IntCast.intCast
    ⊢ Eq (AddGroupWithOne.mk sub_eq_add_neg✝¹ zsmul✝¹ zsmul_zero'✝¹ zsmul_succ'✝¹  …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext] theorem AddCommGroupWithOne.ext ⦃inst₁ inst₂ : AddCommGroupWithOne R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_one : (letI := inst₁; One.one : R) = (letI := inst₂; One.one)) :
    inst₁ = inst₂ := by
  have : inst₁.toAddCommGroup = inst₂.toAddCommGroup :=
    AddCommGroup.ext h_add
  have : inst₁.toAddGroupWithOne = inst₂.toAddGroupWithOne :=
    AddGroupWithOne.ext h_add h_one
  /-
    R : Type u
    inst₁ inst₂ : AddCommGroupWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this✝ : Eq AddCommGroupWithOne.toAddCommGroup AddCommGroupWithOne.toAddCommGroup
    this : Eq AddCommGroupWithOne.toAddGroupWithOne AddCommGroupWithOne.toAddGroup …
    ⊢ Eq inst₁ inst₂
  -/
  injection this with _ h_addMonoidWithOne; injection h_addMonoidWithOne
  /-
    R : Type u
    inst₁ inst₂ : AddCommGroupWithOne R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this : Eq AddCommGroupWithOne.toAddCommGroup AddCommGroupWithOne.toAddCommGroup
    toIntCast_eq✝ : Eq AddCommGroupWithOne.toIntCast AddCommGroupWithOne.toIntCast
    toNeg_eq✝ : Eq SubNegMonoid.toNeg SubNegMonoid.toNeg
    toSub_eq✝ : Eq SubNegMonoid.toSub SubNegMonoid.toSub
    zsmul_eq✝ : Eq SubNegMonoid.zsmul SubNegMonoid.zsmul
    toNatCast_eq✝ : Eq AddCommGroupWithOne.toNatCast AddCommGroupWithOne.toNatCast
    toAddMonoid_eq✝ : Eq SubNegMonoid.toAddMonoid SubNegMonoid.toAddMonoid
    toOne_eq✝ : Eq AddCommGroupWithOne.toOne AddCommGroupWithOne.toOne
    ⊢ Eq inst₁ inst₂
  -/
  cases inst₁; cases inst₂
  /-
    case mk.mk
    R : Type u
    toAddCommGroup✝¹ : AddCommGroup R
    toIntCast✝¹ : IntCast R
    toNatCast✝¹ : NatCast R
    toOne✝¹ : One R
    natCast_zero✝¹ : Eq (NatCast.natCast 0) 0
    natCast_succ✝¹ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd  …
    intCast_ofNat✝¹ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝¹ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg …
    toAddCommGroup✝ : AddCommGroup R
    toIntCast✝ : IntCast R
    toNatCast✝ : NatCast R
    toOne✝ : One R
    natCast_zero✝ : Eq (NatCast.natCast 0) 0
    natCast_succ✝ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd ( …
    intCast_ofNat✝ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg  …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_one : Eq One.one One.one
    this : Eq AddCommGroupWithOne.toAddCommGroup AddCommGroupWithOne.toAddCommGroup
    toIntCast_eq✝ : Eq AddCommGroupWithOne.toIntCast AddCommGroupWithOne.toIntCast
    toNeg_eq✝ : Eq SubNegMonoid.toNeg SubNegMonoid.toNeg
    toSub_eq✝ : Eq SubNegMonoid.toSub SubNegMonoid.toSub
    zsmul_eq✝ : Eq SubNegMonoid.zsmul SubNegMonoid.zsmul
    toNatCast_eq✝ : Eq AddCommGroupWithOne.toNatCast AddCommGroupWithOne.toNatCast
    toAddMonoid_eq✝ : Eq SubNegMonoid.toAddMonoid SubNegMonoid.toAddMonoid
    toOne_eq✝ : Eq AddCommGroupWithOne.toOne AddCommGroupWithOne.toOne
    ⊢ Eq (AddCommGroupWithOne.mk natCast_zero✝¹ natCast_succ✝¹ intCast_ofNat✝¹ int …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonAssocRing R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  have h₁ : inst₁.toNonUnitalNonAssocRing = inst₂.toNonUnitalNonAssocRing := by
    ext : 1 <;> assumption
  have h₂ : inst₁.toNonAssocSemiring = inst₂.toNonAssocSemiring := by
    ext : 1 <;> assumption
  -- Mathematically non-trivial fact: `intCast` is determined by the rest.
  have h₃ : inst₁.toAddCommGroupWithOne = inst₂.toAddCommGroupWithOne :=
    AddCommGroupWithOne.ext h_add (congrArg (·.toOne.one) h₂)
  /-
    R : Type u
    inst₁ inst₂ : NonAssocRing R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h₁ : Eq NonAssocRing.toNonUnitalNonAssocRing NonAssocRing.toNonUnitalNonAssocR …
    h₂ : Eq NonAssocRing.toNonAssocSemiring NonAssocRing.toNonAssocSemiring
    h₃ : Eq NonAssocRing.toAddCommGroupWithOne NonAssocRing.toAddCommGroupWithOne
    ⊢ Eq inst₁ inst₂
  -/
  cases inst₁; cases inst₂
  /-
    case mk.mk
    R : Type u
    toNonUnitalNonAssocRing✝¹ : NonUnitalNonAssocRing R
    toOne✝¹ : One R
    one_mul✝¹ : ∀ (a : R), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : R), Eq (HMul.hMul a 1) a
    toNatCast✝¹ : NatCast R
    natCast_zero✝¹ : Eq (NatCast.natCast 0) 0
    natCast_succ✝¹ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd  …
    toIntCast✝¹ : IntCast R
    intCast_ofNat✝¹ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝¹ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg …
    toNonUnitalNonAssocRing✝ : NonUnitalNonAssocRing R
    toOne✝ : One R
    one_mul✝ : ∀ (a : R), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : R), Eq (HMul.hMul a 1) a
    toNatCast✝ : NatCast R
    natCast_zero✝ : Eq (NatCast.natCast 0) 0
    natCast_succ✝ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd ( …
    toIntCast✝ : IntCast R
    intCast_ofNat✝ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg  …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h₁ : Eq NonAssocRing.toNonUnitalNonAssocRing NonAssocRing.toNonUnitalNonAssocR …
    h₂ : Eq NonAssocRing.toNonAssocSemiring NonAssocRing.toNonAssocSemiring
    h₃ : Eq NonAssocRing.toAddCommGroupWithOne NonAssocRing.toAddCommGroupWithOne
    ⊢ Eq (NonAssocRing.mk one_mul✝¹ mul_one✝¹ natCast_zero✝¹ natCast_succ✝¹ intCas …
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  congr <;> solve| injection h₁ | injection h₂ | injection h₃
            /-
              🎉 no goals
            -/


theorem toNonAssocSemiring_injective :
    Function.Injective (@toNonAssocSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonAssocRing.toNonAssocSemiring R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : NonAssocRing R
    h : Eq NonAssocRing.toNonAssocSemiring NonAssocRing.toNonAssocSemiring
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : NonAssocRing R
      h : Eq NonAssocRing.toNonAssocSemiring NonAssocRing.toNonAssocSemiring
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : NonAssocRing R
      h : Eq NonAssocRing.toNonAssocSemiring NonAssocRing.toNonAssocSemiring
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


theorem toNonUnitalNonAssocring_injective :
    Function.Injective (@toNonUnitalNonAssocRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonAssocRing.toNonUnitalNonAssocRing R)
  -/
  intro _ _ _
  /-
    R : Type u
    a₁✝ a₂✝ : NonAssocRing R
    a✝ : Eq NonAssocRing.toNonUnitalNonAssocRing NonAssocRing.toNonUnitalNonAssocR …
    ⊢ Eq a₁✝ a₂✝
  -/
          /-
            🎉 no goals
          -/
  ext <;> congr
          /-
            🎉 no goals
          -/


@[ext] theorem ext ⦃inst₁ inst₂ : Semiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  -- Show that enough substructures are equal.
  have h₁ : inst₁.toNonUnitalSemiring = inst₂.toNonUnitalSemiring := by
    ext : 1 <;> assumption
  have h₂ : inst₁.toNonAssocSemiring = inst₂.toNonAssocSemiring := by
    ext : 1 <;> assumption
  have h₃ : (inst₁.toMonoidWithZero).toMonoid = (inst₂.toMonoidWithZero).toMonoid := by
    ext : 1; exact h_mul
  -- Split into fields and prove they are equal using the above.
  /-
    R : Type u
    inst₁ inst₂ : Semiring R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h₁ : Eq Semiring.toNonUnitalSemiring Semiring.toNonUnitalSemiring
    h₂ : Eq Semiring.toNonAssocSemiring Semiring.toNonAssocSemiring
    h₃ : Eq MonoidWithZero.toMonoid MonoidWithZero.toMonoid
    ⊢ Eq inst₁ inst₂
  -/
  cases inst₁; cases inst₂
  /-
    case mk.mk
    R : Type u
    toNonUnitalSemiring✝¹ : NonUnitalSemiring R
    toOne✝¹ : One R
    one_mul✝¹ : ∀ (a : R), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : R), Eq (HMul.hMul a 1) a
    toNatCast✝¹ : NatCast R
    natCast_zero✝¹ : Eq (NatCast.natCast 0) 0
    natCast_succ✝¹ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd  …
    npow✝¹ : Nat → R → R
    npow_zero✝¹ : ∀ (x : R), Eq (npow✝¹ 0 x) 1
    npow_succ✝¹ : ∀ (n : Nat) (x : R), Eq (npow✝¹ (HAdd.hAdd n 1) x) (HMul.hMul (n …
    toNonUnitalSemiring✝ : NonUnitalSemiring R
    toOne✝ : One R
    one_mul✝ : ∀ (a : R), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : R), Eq (HMul.hMul a 1) a
    toNatCast✝ : NatCast R
    natCast_zero✝ : Eq (NatCast.natCast 0) 0
    natCast_succ✝ : ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd ( …
    npow✝ : Nat → R → R
    npow_zero✝ : ∀ (x : R), Eq (npow✝ 0 x) 1
    npow_succ✝ : ∀ (n : Nat) (x : R), Eq (npow✝ (HAdd.hAdd n 1) x) (HMul.hMul (npo …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h₁ : Eq Semiring.toNonUnitalSemiring Semiring.toNonUnitalSemiring
    h₂ : Eq Semiring.toNonAssocSemiring Semiring.toNonAssocSemiring
    h₃ : Eq MonoidWithZero.toMonoid MonoidWithZero.toMonoid
    ⊢ Eq (Semiring.mk one_mul✝¹ mul_one✝¹ natCast_zero✝¹ natCast_succ✝¹ npow✝¹ npo …
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  congr <;> solve| injection h₁ | injection h₂ | injection h₃
            /-
              🎉 no goals
            -/


theorem toNonUnitalSemiring_injective :
    Function.Injective (@toNonUnitalSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@Semiring.toNonUnitalSemiring R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : Semiring R
    h : Eq Semiring.toNonUnitalSemiring Semiring.toNonUnitalSemiring
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : Semiring R
      h : Eq Semiring.toNonUnitalSemiring Semiring.toNonUnitalSemiring
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : Semiring R
      h : Eq Semiring.toNonUnitalSemiring Semiring.toNonUnitalSemiring
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


theorem toNonAssocSemiring_injective :
    Function.Injective (@toNonAssocSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@Semiring.toNonAssocSemiring R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : Semiring R
    h : Eq Semiring.toNonAssocSemiring Semiring.toNonAssocSemiring
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : Semiring R
      h : Eq Semiring.toNonAssocSemiring Semiring.toNonAssocSemiring
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : Semiring R
      h : Eq Semiring.toNonAssocSemiring Semiring.toNonAssocSemiring
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


@[ext] theorem ext ⦃inst₁ inst₂ : Ring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ := by
  -- Show that enough substructures are equal.
  have h₁ : inst₁.toSemiring = inst₂.toSemiring := by
    ext : 1 <;> assumption
  have h₂ : inst₁.toNonAssocRing = inst₂.toNonAssocRing := by
    ext : 1 <;> assumption
  /- We prove that the `SubNegMonoid`s are equal because they are one
  field away from `Sub` and `Neg`, enabling use of `injection`. -/
  have h₃ : (inst₁.toAddCommGroup).toAddGroup.toSubNegMonoid
            = (inst₂.toAddCommGroup).toAddGroup.toSubNegMonoid :=
    congrArg (@AddGroup.toSubNegMonoid R) <| by ext : 1; exact h_add
  -- Split into fields and prove they are equal using the above.
  /-
    R : Type u
    inst₁ inst₂ : Ring R
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h₁ : Eq Ring.toSemiring Ring.toSemiring
    h₂ : Eq Ring.toNonAssocRing Ring.toNonAssocRing
    h₃ : Eq AddGroup.toSubNegMonoid AddGroup.toSubNegMonoid
    ⊢ Eq inst₁ inst₂
  -/
  cases inst₁; cases inst₂
  /-
    case mk.mk
    R : Type u
    toSemiring✝¹ : Semiring R
    toNeg✝¹ : Neg R
    toSub✝¹ : Sub R
    sub_eq_add_neg✝¹ : ∀ (a b : R), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
    zsmul✝¹ : Int → R → R
    zsmul_zero'✝¹ : ∀ (a : R), Eq (zsmul✝¹ 0 a) 0
    zsmul_succ'✝¹ : ∀ (n : Nat) (a : R), Eq (zsmul✝¹ (↑n.succ) a) (HAdd.hAdd (zsmu …
    zsmul_neg'✝¹ : ∀ (n : Nat) (a : R), Eq (zsmul✝¹ (Int.negSucc n) a) (Neg.neg (z …
    neg_add_cancel✝¹ : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
    toIntCast✝¹ : IntCast R
    intCast_ofNat✝¹ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝¹ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg …
    toSemiring✝ : Semiring R
    toNeg✝ : Neg R
    toSub✝ : Sub R
    sub_eq_add_neg✝ : ∀ (a b : R), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
    zsmul✝ : Int → R → R
    zsmul_zero'✝ : ∀ (a : R), Eq (zsmul✝ 0 a) 0
    zsmul_succ'✝ : ∀ (n : Nat) (a : R), Eq (zsmul✝ (↑n.succ) a) (HAdd.hAdd (zsmul✝ …
    zsmul_neg'✝ : ∀ (n : Nat) (a : R), Eq (zsmul✝ (Int.negSucc n) a) (Neg.neg (zsm …
    neg_add_cancel✝ : ∀ (a : R), Eq (HAdd.hAdd (Neg.neg a) a) 0
    toIntCast✝ : IntCast R
    intCast_ofNat✝ : ∀ (n : Nat), Eq (IntCast.intCast ↑n) ↑n
    intCast_negSucc✝ : ∀ (n : Nat), Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg  …
    h_add : Eq HAdd.hAdd HAdd.hAdd
    h_mul : Eq HMul.hMul HMul.hMul
    h₁ : Eq Ring.toSemiring Ring.toSemiring
    h₂ : Eq Ring.toNonAssocRing Ring.toNonAssocRing
    h₃ : Eq AddGroup.toSubNegMonoid AddGroup.toSubNegMonoid
    ⊢ Eq (Ring.mk sub_eq_add_neg✝¹ zsmul✝¹ zsmul_zero'✝¹ zsmul_succ'✝¹ zsmul_neg'✝ …
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
  congr <;> solve | injection h₂ | injection h₃
            /-
              🎉 no goals
            -/


theorem toNonUnitalRing_injective :
    Function.Injective (@toNonUnitalRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@Ring.toNonUnitalRing R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : Ring R
    h : Eq Ring.toNonUnitalRing Ring.toNonUnitalRing
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : Ring R
      h : Eq Ring.toNonUnitalRing Ring.toNonUnitalRing
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : Ring R
      h : Eq Ring.toNonUnitalRing Ring.toNonUnitalRing
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


theorem toNonAssocRing_injective :
    Function.Injective (@toNonAssocRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@Ring.toNonAssocRing R)
  -/
  intro _ _ _
  /-
    R : Type u
    a₁✝ a₂✝ : Ring R
    a✝ : Eq Ring.toNonAssocRing Ring.toNonAssocRing
    ⊢ Eq a₁✝ a₂✝
  -/
          /-
            🎉 no goals
          -/
  ext <;> congr
          /-
            🎉 no goals
          -/


theorem toSemiring_injective :
    Function.Injective (@toSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@Ring.toSemiring R)
  -/
  intro _ _ h
  /-
    R : Type u
    a₁✝ a₂✝ : Ring R
    h : Eq Ring.toSemiring Ring.toSemiring
    ⊢ Eq a₁✝ a₂✝
  -/
  ext x y
    /-
      case h_add.h.h
      R : Type u
      a₁✝ a₂✝ : Ring R
      h : Eq Ring.toSemiring Ring.toSemiring
      x y : R
      ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd x y)
    -/
  · exact congrArg (·.toAdd.add x y) h
    /-
      🎉 no goals
    -/
    /-
      case h_mul.h.h
      R : Type u
      a₁✝ a₂✝ : Ring R
      h : Eq Ring.toSemiring Ring.toSemiring
      x y : R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul x y)
    -/
  · exact congrArg (·.toMul.mul x y) h
    /-
      🎉 no goals
    -/


theorem toNonUnitalNonAssocSemiring_injective :
    Function.Injective (@toNonUnitalNonAssocSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalNonAssocCommSemiring.toNonUnitalNonAssocSemiri …
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalNonAssocCommSemiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toNonUnitalNonAssocSemiring_injective <|
    NonUnitalNonAssocSemiring.ext h_add h_mul


theorem toNonUnitalSemiring_injective :
    Function.Injective (@toNonUnitalSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalCommSemiring.toNonUnitalSemiring R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalCommSemiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toNonUnitalSemiring_injective <|
    NonUnitalSemiring.ext h_add h_mul


theorem toNonUnitalNonAssocRing_injective :
    Function.Injective (@toNonUnitalNonAssocRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalNonAssocCommRing.toNonUnitalNonAssocRing R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalNonAssocCommRing R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toNonUnitalNonAssocRing_injective <|
    NonUnitalNonAssocRing.ext h_add h_mul


theorem toNonUnitalRing_injective :
    Function.Injective (@toNonUnitalRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@NonUnitalCommRing.toNonUnitalRing R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : NonUnitalCommRing R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toNonUnitalRing_injective <|
    NonUnitalRing.ext h_add h_mul


theorem toSemiring_injective :
    Function.Injective (@toSemiring R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@CommSemiring.toSemiring R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : CommSemiring R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toSemiring_injective <|
    Semiring.ext h_add h_mul


theorem toRing_injective : Function.Injective (@toRing R) := by
  /-
    R : Type u
    ⊢ Function.Injective (@CommRing.toRing R)
  -/
  rintro ⟨⟩ ⟨⟩ _; congr
                  /-
                    🎉 no goals
                  -/


@[ext] theorem ext ⦃inst₁ inst₂ : CommRing R⦄
    (h_add : local_hAdd[R, inst₁] = local_hAdd[R, inst₂])
    (h_mul : local_hMul[R, inst₁] = local_hMul[R, inst₂]) :
    inst₁ = inst₂ :=
  toRing_injective <| Ring.ext h_add h_mul


