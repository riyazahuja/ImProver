/-- Helper function to synthesize a typed `CharZero α` expression given `Ring α`. -/
def inferCharZeroOfRing {α : Q(Type u)} (_i : Q(Ring $α) := by with_reducible assumption) :
    MetaM Q(CharZero $α) :=
  return ← synthInstanceQ (q(CharZero $α) : Q(Prop)) <|>
    throwError "not a characteristic zero ring"


/-- Helper function to synthesize a typed `CharZero α` expression given `Ring α`, if it exists. -/
def inferCharZeroOfRing? {α : Q(Type u)} (_i : Q(Ring $α) := by with_reducible assumption) :
    MetaM (Option Q(CharZero $α)) :=
  return (← trySynthInstanceQ (q(CharZero $α) : Q(Prop))).toOption


/-- Helper function to synthesize a typed `CharZero α` expression given `AddMonoidWithOne α`. -/
def inferCharZeroOfAddMonoidWithOne {α : Q(Type u)}
    (_i : Q(AddMonoidWithOne $α) := by with_reducible assumption) : MetaM Q(CharZero $α) :=
  return ← synthInstanceQ (q(CharZero $α) : Q(Prop)) <|>
    throwError "not a characteristic zero AddMonoidWithOne"


/-- Helper function to synthesize a typed `CharZero α` expression given `AddMonoidWithOne α`, if it
exists. -/
def inferCharZeroOfAddMonoidWithOne? {α : Q(Type u)}
    (_i : Q(AddMonoidWithOne $α) := by with_reducible assumption) :
      MetaM (Option Q(CharZero $α)) :=
  return (← trySynthInstanceQ (q(CharZero $α) : Q(Prop))).toOption


/-- Helper function to synthesize a typed `CharZero α` expression given `DivisionRing α`. -/
def inferCharZeroOfDivisionRing {α : Q(Type u)}
    (_i : Q(DivisionRing $α) := by with_reducible assumption) : MetaM Q(CharZero $α) :=
  return ← synthInstanceQ (q(CharZero $α) : Q(Prop)) <|>
    throwError "not a characteristic zero division ring"


/-- Helper function to synthesize a typed `CharZero α` expression given `DivisionRing α`, if it
exists. -/
def inferCharZeroOfDivisionRing? {α : Q(Type u)}
    (_i : Q(DivisionRing $α) := by with_reducible assumption) : MetaM (Option Q(CharZero $α)) :=
  return (← trySynthInstanceQ (q(CharZero $α) : Q(Prop))).toOption


theorem isRat_mkRat : {a na n : ℤ} → {b nb d : ℕ} → IsInt a na → IsNat b nb →
    IsRat (na / nb : ℚ) n d → IsRat (mkRat a b) n d
                                                 /-
                                                   n✝¹ num✝ : Int
                                                   n✝ denom✝ : Nat
                                                   inv✝ : Invertible ↑denom✝
                                                   h : Eq (HDiv.hDiv ↑n✝¹ ↑n✝) (HMul.hMul (↑num✝) (Invertible.invOf ↑denom✝))
                                                   ⊢ Mathlib.Meta.NormNum.IsRat (mkRat ↑n✝¹ ↑n✝) num✝ denom✝
                                                 -/
  | _, _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, ⟨_, h⟩ => by rw [Rat.mkRat_eq_div]; exact ⟨_, h⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `mkRat a b`,
such that `norm_num` successfully recognises both `a` and `b`, and returns `a / b`. -/
@[norm_num mkRat _ _]
def evalMkRat : NormNumExt where eval {u α} (e : Q(ℚ)) : MetaM (Result e) := do
  let .app (.app (.const ``mkRat _) (a : Q(ℤ))) (b : Q(ℕ)) ← whnfR e | failure
  haveI' : $e =Q mkRat $a $b := ⟨⟩
  let ra ← derive a
  let some ⟨_, na, pa⟩ := ra.toInt (q(Int.instRing) : Q(Ring Int)) | failure
  let ⟨nb, pb⟩ ← deriveNat q($b) q(AddCommMonoidWithOne.toAddMonoidWithOne)
  let rab ← derive q($na / $nb : Rat)
  let ⟨q, n, d, p⟩ ← rab.toRat' q(Rat.instDivisionRing)
  return .isRat' _ q n d q(isRat_mkRat $pa $pb $p)


theorem isNat_ratCast {R : Type*} [DivisionRing R] : {q : ℚ} → {n : ℕ} →
    IsNat q n → IsNat (q : R) n
                       /-
                         R : Type u_1
                         inst✝ : DivisionRing R
                         n✝ : Nat
                         ⊢ Eq ↑↑n✝ ↑n✝
                       -/
  | _, _, ⟨rfl⟩ => ⟨by simp⟩
                       /-
                         🎉 no goals
                       -/


theorem isInt_ratCast {R : Type*} [DivisionRing R] : {q : ℚ} → {n : ℤ} →
    IsInt q n → IsInt (q : R) n
                       /-
                         R : Type u_1
                         inst✝ : DivisionRing R
                         n✝ : Int
                         ⊢ Eq ↑↑n✝ ↑n✝
                       -/
  | _, _, ⟨rfl⟩ => ⟨by simp⟩
                       /-
                         🎉 no goals
                       -/


theorem isRat_ratCast {R : Type*} [DivisionRing R] [CharZero R] : {q : ℚ} → {n : ℤ} → {d : ℕ} →
    IsRat q n d → IsRat (q : R) n d
                                         /-
                                           R : Type u_1
                                           inst✝¹ : DivisionRing R
                                           inst✝ : CharZero R
                                           num✝ : Int
                                           denom✝ : Nat
                                           qi : Rat
                                           invOf_mul_self✝ : Eq (HMul.hMul qi ↑denom✝) 1
                                           mul_invOf_self✝ : Eq (HMul.hMul (↑denom✝) qi) 1
                                           ⊢ Eq (HMul.hMul ↑qi ↑denom✝) 1
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  | _, _, _, ⟨⟨qi,_,_⟩, rfl⟩ => ⟨⟨qi, by norm_cast, by norm_cast⟩, by simp only []; norm_cast⟩
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- The `norm_num` extension which identifies an expression `RatCast.ratCast q` where `norm_num`
recognizes `q`, returning the cast of `q`. -/
@[norm_num Rat.cast _, RatCast.ratCast _] def evalRatCast : NormNumExt where eval {u α} e := do
  let dα ← inferDivisionRing α
  let .app r (a : Q(ℚ)) ← whnfR e | failure
  guard <|← withNewMCtxDepth <| isDefEq r q(Rat.cast (K := $α))
  let r ← derive q($a)
  haveI' : $e =Q Rat.cast $a := ⟨⟩
  match r with
  | .isNat _ na pa =>
    assumeInstancesCommute
    return .isNat _ na q(isNat_ratCast $pa)
  | .isNegNat _ na pa =>
    assumeInstancesCommute
    return .isNegNat _ na q(isInt_ratCast $pa)
  | .isRat _ qa na da pa =>
    assumeInstancesCommute
    let i ← inferCharZeroOfDivisionRing dα
    return .isRat dα qa na da q(isRat_ratCast $pa)
  | _ => failure


theorem isRat_inv_pos {α} [DivisionRing α] [CharZero α] {a : α} {n d : ℕ} :
    IsRat a (.ofNat (Nat.succ n)) d → IsRat a⁻¹ (.ofNat d) (Nat.succ n) := by
  /-
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    a : α
    n d : Nat
    ⊢ Mathlib.Meta.NormNum.IsRat a (Int.ofNat n.succ) d → Mathlib.Meta.NormNum.IsR …
  -/
  rintro ⟨_, rfl⟩
  /-
    case mk
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    n d : Nat
    inv✝ : Invertible ↑d
    ⊢ Mathlib.Meta.NormNum.IsRat (Inv.inv (HMul.hMul (↑(Int.ofNat n.succ)) (Invert …
  -/
  have := invertibleOfNonzero (α := α) (Nat.cast_ne_zero.2 (Nat.succ_ne_zero n))
  /-
    case mk
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    n d : Nat
    inv✝ : Invertible ↑d
    this : Invertible ↑n.succ
    ⊢ Mathlib.Meta.NormNum.IsRat (Inv.inv (HMul.hMul (↑(Int.ofNat n.succ)) (Invert …
  -/
  exact ⟨this, by simp⟩
  /-
    🎉 no goals
  -/


theorem isRat_inv_one {α} [DivisionRing α] : {a : α} →
    IsNat a (nat_lit 1) → IsNat a⁻¹ (nat_lit 1)
                    /-
                      α : Type u_1
                      inst✝ : DivisionRing α
                      ⊢ Eq (Inv.inv ↑1) ↑1
                    -/
  | _, ⟨rfl⟩ => ⟨by simp⟩
                    /-
                      🎉 no goals
                    -/


theorem isRat_inv_zero {α} [DivisionRing α] : {a : α} →
    IsNat a (nat_lit 0) → IsNat a⁻¹ (nat_lit 0)
                    /-
                      α : Type u_1
                      inst✝ : DivisionRing α
                      ⊢ Eq (Inv.inv ↑0) ↑0
                    -/
  | _, ⟨rfl⟩ => ⟨by simp⟩
                    /-
                      🎉 no goals
                    -/


theorem isRat_inv_neg_one {α} [DivisionRing α] : {a : α} →
    IsInt a (.negOfNat (nat_lit 1)) → IsInt a⁻¹ (.negOfNat (nat_lit 1))
                    /-
                      α : Type u_1
                      inst✝ : DivisionRing α
                      ⊢ Eq (Inv.inv ↑(Int.negOfNat 1)) ↑(Int.negOfNat 1)
                    -/
  | _, ⟨rfl⟩ => ⟨by simp [inv_neg_one]⟩
                    /-
                      🎉 no goals
                    -/


theorem isRat_inv_neg {α} [DivisionRing α] [CharZero α] {a : α} {n d : ℕ} :
    IsRat a (.negOfNat (Nat.succ n)) d → IsRat a⁻¹ (.negOfNat d) (Nat.succ n) := by
  /-
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    a : α
    n d : Nat
    ⊢ Mathlib.Meta.NormNum.IsRat a (Int.negOfNat n.succ) d → Mathlib.Meta.NormNum. …
  -/
  rintro ⟨_, rfl⟩
  /-
    case mk
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    n d : Nat
    inv✝ : Invertible ↑d
    ⊢ Mathlib.Meta.NormNum.IsRat (Inv.inv (HMul.hMul (↑(Int.negOfNat n.succ)) (Inv …
  -/
  simp only [Int.negOfNat_eq]
  /-
    case mk
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    n d : Nat
    inv✝ : Invertible ↑d
    ⊢ Mathlib.Meta.NormNum.IsRat (Inv.inv (HMul.hMul (↑(Neg.neg (Int.ofNat n.succ) …
  -/
  have := invertibleOfNonzero (α := α) (Nat.cast_ne_zero.2 (Nat.succ_ne_zero n))
  /-
    case mk
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    n d : Nat
    inv✝ : Invertible ↑d
    this : Invertible ↑n.succ
    ⊢ Mathlib.Meta.NormNum.IsRat (Inv.inv (HMul.hMul (↑(Neg.neg (Int.ofNat n.succ) …
  -/
  generalize Nat.succ n = n at *
  /-
    case mk
    α : Type u_1
    inst✝¹ : DivisionRing α
    inst✝ : CharZero α
    n✝ d : Nat
    inv✝ : Invertible ↑d
    n : Nat
    this : Invertible ↑n
    ⊢ Mathlib.Meta.NormNum.IsRat (Inv.inv (HMul.hMul (↑(Neg.neg (Int.ofNat n))) (I …
  -/
  use this; simp only [Int.ofNat_eq_coe, Int.cast_neg,
    Int.cast_natCast, invOf_eq_inv, inv_neg, neg_mul, mul_inv_rev, inv_inv]


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `a⁻¹`,
such that `norm_num` successfully recognises `a`. -/
@[norm_num _⁻¹] def evalInv : NormNumExt where eval {u α} e := do
  let .app f (a : Q($α)) ← whnfR e | failure
  let ra ← derive a
  let dα ← inferDivisionRing α
  let i ← inferCharZeroOfDivisionRing? dα
  guard <|← withNewMCtxDepth <| isDefEq f q(Inv.inv (α := $α))
  haveI' : $e =Q $a⁻¹ := ⟨⟩
  assumeInstancesCommute
  let rec
  /-- Main part of `evalInv`. -/
  core : Option (Result e) := do
    let ⟨qa, na, da, pa⟩ ← ra.toRat' dα
    let qb := qa⁻¹
    if qa > 0 then
      if let some i := i then
        have lit : Q(ℕ) := na.appArg!
        haveI : $na =Q Int.ofNat $lit := ⟨⟩
        have lit2 : Q(ℕ) := mkRawNatLit (lit.natLit! - 1)
        haveI : $lit =Q ($lit2).succ := ⟨⟩
        return .isRat' dα qb q(.ofNat $da) lit q(isRat_inv_pos $pa)
      else
        guard (qa = 1)
        let .isNat inst n pa := ra | failure
        haveI' : $n =Q nat_lit 1 := ⟨⟩
        assumeInstancesCommute
        return .isNat inst n q(isRat_inv_one $pa)
    else if qa < 0 then
      if let some i := i then
        have lit : Q(ℕ) := na.appArg!
        haveI : $na =Q Int.negOfNat $lit := ⟨⟩
        have lit2 : Q(ℕ) := mkRawNatLit (lit.natLit! - 1)
        haveI : $lit =Q ($lit2).succ := ⟨⟩
        return .isRat' dα qb q(.negOfNat $da) lit q(isRat_inv_neg $pa)
      else
        guard (qa = -1)
        let .isNegNat inst n pa := ra | failure
        haveI' : $n =Q nat_lit 1 := ⟨⟩
        assumeInstancesCommute
        return .isNegNat inst n q(isRat_inv_neg_one $pa)
    else
      let .isNat inst n pa := ra | failure
      haveI' : $n =Q nat_lit 0 := ⟨⟩
      assumeInstancesCommute
      return .isNat inst n q(isRat_inv_zero $pa)
  core


