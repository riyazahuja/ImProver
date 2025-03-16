theorem IsInt.raw_refl (n : ℤ) : IsInt n n := ⟨rfl⟩


theorem isNat_zero (α) [AddMonoidWithOne α] : IsNat (Zero.zero : α) (nat_lit 0) :=
  ⟨Nat.cast_zero.symm⟩


/-- The `norm_num` extension which identifies the expression `Zero.zero`, returning `0`. -/
@[norm_num Zero.zero] def evalZero : NormNumExt where eval {u α} e := do
  let sα ← inferAddMonoidWithOne α
  match e with
  | ~q(Zero.zero) => return .isNat sα (mkRawNatLit 0) q(isNat_zero $α)


theorem isNat_one (α) [AddMonoidWithOne α] : IsNat (One.one : α) (nat_lit 1) := ⟨Nat.cast_one.symm⟩


/-- The `norm_num` extension which identifies the expression `One.one`, returning `1`. -/
@[norm_num One.one] def evalOne : NormNumExt where eval {u α} e := do
  let sα ← inferAddMonoidWithOne α
  match e with
  | ~q(One.one) => return .isNat sα (mkRawNatLit 1) q(isNat_one $α)


theorem isNat_ofNat (α : Type u) [AddMonoidWithOne α] {a : α} {n : ℕ}
    (h : n = a) : IsNat a n := ⟨h.symm⟩


/-- The `norm_num` extension which identifies an expression `OfNat.ofNat n`, returning `n`. -/
@[norm_num OfNat.ofNat _] def evalOfNat : NormNumExt where eval {u α} e := do
  let sα ← inferAddMonoidWithOne α
  match e with
  | ~q(@OfNat.ofNat _ $n $oα) =>
    let n : Q(ℕ) ← whnf n
    guard n.isRawNatLit
    let ⟨a, (pa : Q($n = $e))⟩ ← mkOfNat α sα n
    guard <|← isDefEq a e
    return .isNat sα n q(isNat_ofNat $α $pa)


theorem isNat_intOfNat : {n n' : ℕ} → IsNat n n' → IsNat (Int.ofNat n) n'
  | _, _, ⟨rfl⟩ => ⟨rfl⟩


/-- The `norm_num` extension which identifies the constructor application `Int.ofNat n` such that
`norm_num` successfully recognizes `n`, returning `n`. -/
@[norm_num Int.ofNat _] def evalIntOfNat : NormNumExt where eval {u α} e := do
  let .app (.const ``Int.ofNat _) (n : Q(ℕ)) ← whnfR e | failure
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q Int := ⟨⟩
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let sℤ : Q(AddMonoidWithOne ℤ) := q(instAddMonoidWithOne)
  let ⟨n', p⟩ ← deriveNat n sℕ
  haveI' x : $e =Q Int.ofNat $n := ⟨⟩
  return .isNat sℤ n' q(isNat_intOfNat $p)


theorem isNat_natAbs_pos : {n : ℤ} → {a : ℕ} → IsNat n a → IsNat n.natAbs a
  | _, _, ⟨rfl⟩ => ⟨rfl⟩


theorem isNat_natAbs_neg : {n : ℤ} → {a : ℕ} → IsInt n (.negOfNat a) → IsNat n.natAbs a
                       /-
                         a✝ : Nat
                         ⊢ Eq (↑(Int.negOfNat a✝)).natAbs ↑a✝
                       -/
  | _, _, ⟨rfl⟩ => ⟨by simp⟩
                       /-
                         🎉 no goals
                       -/


/-- The `norm_num` extension which identifies the expression `Int.natAbs n` such that
`norm_num` successfully recognizes `n`. -/
@[norm_num Int.natAbs (_ : ℤ)] def evalIntNatAbs : NormNumExt where eval {u α} e := do
  let .app (.const ``Int.natAbs _) (x : Q(ℤ)) ← whnfR e | failure
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℕ := ⟨⟩
  haveI' : $e =Q Int.natAbs $x := ⟨⟩
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  match ← derive (u := .zero) x with
  | .isNat    _ a p => assumeInstancesCommute; return .isNat sℕ a q(isNat_natAbs_pos $p)
  | .isNegNat _ a p => assumeInstancesCommute; return .isNat sℕ a q(isNat_natAbs_neg $p)
  | _ => failure


theorem isNat_natCast {R} [AddMonoidWithOne R] (n m : ℕ) :
                                      /-
                                        R : Type u_1
                                        inst✝ : AddMonoidWithOne R
                                        n m : Nat
                                        ⊢ Mathlib.Meta.NormNum.IsNat n m → Mathlib.Meta.NormNum.IsNat (↑n) m
                                      -/
    IsNat n m → IsNat (n : R) m := by rintro ⟨⟨⟩⟩; exact ⟨rfl⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-04-17")]
alias isNat_cast := isNat_natCast


/-- The `norm_num` extension which identifies an expression `Nat.cast n`, returning `n`. -/
@[norm_num Nat.cast _, NatCast.natCast _] def evalNatCast : NormNumExt where eval {u α} e := do
  let sα ← inferAddMonoidWithOne α
  let .app n (a : Q(ℕ)) ← whnfR e | failure
  guard <|← withNewMCtxDepth <| isDefEq n q(Nat.cast (R := $α))
  let ⟨na, pa⟩ ← deriveNat a q(instAddMonoidWithOneNat)
  haveI' : $e =Q $a := ⟨⟩
  return .isNat sα na q(isNat_natCast $a $na $pa)


theorem isNat_intCast {R} [Ring R] (n : ℤ) (m : ℕ) :
                                      /-
                                        R : Type u_1
                                        inst✝ : Ring R
                                        n : Int
                                        m : Nat
                                        ⊢ Mathlib.Meta.NormNum.IsNat n m → Mathlib.Meta.NormNum.IsNat (↑n) m
                                      -/
    IsNat n m → IsNat (n : R) m := by rintro ⟨⟨⟩⟩; exact ⟨by simp⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-04-17")]
alias isNat_int_cast := isNat_intCast


theorem isintCast {R} [Ring R] (n m : ℤ) :
                                      /-
                                        R : Type u_1
                                        inst✝ : Ring R
                                        n m : Int
                                        ⊢ Mathlib.Meta.NormNum.IsInt n m → Mathlib.Meta.NormNum.IsInt (↑n) m
                                      -/
    IsInt n m → IsInt (n : R) m := by rintro ⟨⟨⟩⟩; exact ⟨rfl⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-04-17")]
alias isInt_cast := isintCast


/-- The `norm_num` extension which identifies an expression `Int.cast n`, returning `n`. -/
@[norm_num Int.cast _, IntCast.intCast _] def evalIntCast : NormNumExt where eval {u α} e := do
  let rα ← inferRing α
  let .app i (a : Q(ℤ)) ← whnfR e | failure
  guard <|← withNewMCtxDepth <| isDefEq i q(Int.cast (R := $α))
  match ← derive (α := q(ℤ)) a with
  | .isNat _ na pa =>
    assumeInstancesCommute
    haveI' : $e =Q Int.cast $a := ⟨⟩
    return .isNat _ na q(isNat_intCast $a $na $pa)
  | .isNegNat _ na pa =>
    assumeInstancesCommute
    haveI' : $e =Q Int.cast $a := ⟨⟩
    return .isNegNat _ na q(isintCast $a (.negOfNat $na) $pa)
  | _ => failure


theorem isNat_add {α} [AddMonoidWithOne α] : ∀ {f : α → α → α} {a b : α} {a' b' c : ℕ},
    f = HAdd.hAdd → IsNat a a' → IsNat b b' → Nat.add a' b' = c → IsNat (f a b) c
  | _, _, _, _, _, _, rfl, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨(Nat.cast_add _ _).symm⟩

-- see note [norm_num lemma function equality]

theorem isInt_add {α} [Ring α] : ∀ {f : α → α → α} {a b : α} {a' b' c : ℤ},
    f = HAdd.hAdd → IsInt a a' → IsInt b b' → Int.add a' b' = c → IsInt (f a b) c
  | _, _, _, _, _, _, rfl, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨(Int.cast_add ..).symm⟩


/-- If `b` divides `a` and `a` is invertible, then `b` is invertible. -/
def invertibleOfMul {α} [Semiring α] (k : ℕ) (b : α) :
    ∀ (a : α) [Invertible a], a = k * b → Invertible b
  | _, ⟨c, hc1, hc2⟩, rfl => by
    /-
      α : Type ?u.16640
      inst✝ : Semiring α
      k : Nat
      b c : α
      hc1 : Eq (HMul.hMul c (HMul.hMul (↑k) b)) 1
      hc2 : Eq (HMul.hMul (HMul.hMul (↑k) b) c) 1
      ⊢ Invertible b
    -/
    rw [← mul_assoc] at hc1
    /-
      α : Type ?u.16640
      inst✝ : Semiring α
      k : Nat
      b c : α
      hc1 : Eq (HMul.hMul (HMul.hMul c ↑k) b) 1
      hc2 : Eq (HMul.hMul (HMul.hMul (↑k) b) c) 1
      ⊢ Invertible b
    -/
    rw [Nat.cast_commute k, mul_assoc, Nat.cast_commute k] at hc2
    /-
      α : Type ?u.16640
      inst✝ : Semiring α
      k : Nat
      b c : α
      hc1 : Eq (HMul.hMul (HMul.hMul c ↑k) b) 1
      hc2 : Eq (HMul.hMul b (HMul.hMul c ↑k)) 1
      ⊢ Invertible b
    -/
    exact ⟨_, hc1, hc2⟩
    /-
      🎉 no goals
    -/


/-- If `b` divides `a` and `a` is invertible, then `b` is invertible. -/
def invertibleOfMul' {α} [Semiring α] {a k b : ℕ} [Invertible (a : α)]
                                                                           /-
                                                                             α : Type ?u.18399
                                                                             inst✝¹ : Semiring α
                                                                             a k b : Nat
                                                                             inst✝ : Invertible ↑a
                                                                             h : Eq a (HMul.hMul k b)
                                                                             ⊢ Eq (↑a) (HMul.hMul ↑k ↑b)
                                                                           -/
    (h : a = k * b) : Invertible (b : α) := invertibleOfMul k (b:α) ↑a (by simp [h])
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

-- TODO: clean up and move it somewhere in mathlib? It's a bit much for this file
-- see note [norm_num lemma function equality]

theorem isRat_add {α} [Ring α] {f : α → α → α} {a b : α} {na nb nc : ℤ} {da db dc k : ℕ} :
    f = HAdd.hAdd → IsRat a na da → IsRat b nb db →
    Int.add (Int.mul na db) (Int.mul nb da) = Int.mul k nc →
    Nat.mul da db = Nat.mul k dc →
    IsRat (f a b) nc dc := by
  /-
    α : Type u_1
    inst✝ : Ring α
    f : α → α → α
    a b : α
    na nb nc : Int
    da db dc k : Nat
    ⊢ Eq f HAdd.hAdd → Mathlib.Meta.NormNum.IsRat a na da → Mathlib.Meta.NormNum.I …
  -/
  rintro rfl ⟨_, rfl⟩ ⟨_, rfl⟩ (h₁ : na * db + nb * da = k * nc) (h₂ : da * db = k * dc)
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    ⊢ Mathlib.Meta.NormNum.IsRat (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da …
  -/
  have : Invertible (↑(da * db) : α) := by simpa using invertibleMul (da:α) db
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this : Invertible ↑(HMul.hMul da db)
    ⊢ Mathlib.Meta.NormNum.IsRat (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da …
  -/
  have := invertibleOfMul' (α := α) h₂
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    ⊢ Mathlib.Meta.NormNum.IsRat (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da …
  -/
  use this
  /-
    case eq
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  have H := (Nat.cast_commute (α := α) da db).invOf_left.invOf_right.right_comm
  /-
    case eq
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  have h₁ := congr_arg (↑· * (⅟↑da * ⅟↑db : α)) h₁
  simp only [Int.cast_add, Int.cast_mul, Int.cast_natCast, ← mul_assoc,
    add_mul, mul_invOf_cancel_right] at h₁
  /-
    case eq
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul ↑na ↑db) (Invertible.invOf …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  have h₂ := congr_arg (↑nc * ↑· * (⅟↑da * ⅟↑db * ⅟↑dc : α)) h₂
  /-
    case eq
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂✝ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul ↑na ↑db) (Invertible.invOf …
    h₂ : Eq ((fun x => HMul.hMul (HMul.hMul ↑nc ↑x) (HMul.hMul (HMul.hMul (Inverti …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  simp only [H, mul_invOf_cancel_right', Nat.cast_mul, ← mul_assoc] at h₁ h₂
  /-
    case eq
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HAdd.hAdd (HMul.hMul na ↑db) (HMul.hMul nb ↑da)) (HMul.hMul (↑k) nc)
    h₂✝ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) ( …
    h₂ : Eq (HMul.hMul (↑nc) (Invertible.invOf ↑dc)) (HMul.hMul (HMul.hMul (HMul.h …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  rw [h₁, h₂, Nat.cast_commute]
  simp only [mul_invOf_cancel_right,
    (Nat.cast_commute (α := α) da dc).invOf_left.invOf_right.right_comm,
    (Nat.cast_commute (α := α) db dc).invOf_left.invOf_right.right_comm]


/-- Consider an `Option` as an object in the `MetaM` monad, by throwing an error on `none`. -/
def _root_.Mathlib.Meta.monadLiftOptionMetaM : MonadLift Option MetaM where
  monadLift
  | none => failure
  | some e => pure e


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `a + b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num _ + _] def evalAdd : NormNumExt where eval {u α} e := do
  let .app (.app (f : Q($α → $α → $α)) (a : Q($α))) (b : Q($α)) ← whnfR e | failure
  let ra ← derive a; let rb ← derive b
  match ra, rb with
  | .isBool .., _ | _, .isBool .. => failure
  | .isNat _ .., .isNat _ .. | .isNat _ .., .isNegNat _ .. | .isNat _ .., .isRat _ ..
  | .isNegNat _ .., .isNat _ .. | .isNegNat _ .., .isNegNat _ .. | .isNegNat _ .., .isRat _ ..
  | .isRat _ .., .isNat _ .. | .isRat _ .., .isNegNat _ .. | .isRat _ .., .isRat _ .. =>
    guard <|← withNewMCtxDepth <| isDefEq f q(HAdd.hAdd (α := $α))
  let rec
  /-- Main part of `evalAdd`. -/
  core : Option (Result e) := do
    let rec intArm (rα : Q(Ring $α)) := do
      haveI' : $e =Q $a + $b := ⟨⟩
      let ⟨za, na, pa⟩ ← ra.toInt _; let ⟨zb, nb, pb⟩ ← rb.toInt _
      haveI' : $f =Q HAdd.hAdd := ⟨⟩
      let zc := za + zb
      have c := mkRawIntLit zc
      haveI' : Int.add $na $nb =Q $c := ⟨⟩
      return .isInt rα c zc q(isInt_add (f := $f) (.refl $f) $pa $pb (.refl $c))
    let rec ratArm (dα : Q(DivisionRing $α)) : Option (Result _) := do
      haveI' : $e =Q $a + $b := ⟨⟩
      haveI' : $f =Q HAdd.hAdd := ⟨⟩
      let ⟨qa, na, da, pa⟩ ← ra.toRat' dα; let ⟨qb, nb, db, pb⟩ ← rb.toRat' dα
      let qc := qa + qb
      let dd := qa.den * qb.den
      let k := dd / qc.den
      have t1 : Q(ℤ) := mkRawIntLit (k * qc.num)
      have t2 : Q(ℕ) := mkRawNatLit dd
      have nc : Q(ℤ) := mkRawIntLit qc.num
      have dc : Q(ℕ) := mkRawNatLit qc.den
      have k : Q(ℕ) := mkRawNatLit k
      let r1 : Q(Int.add (Int.mul $na $db) (Int.mul $nb $da) = Int.mul $k $nc) :=
        (q(Eq.refl $t1) : Expr)
      let r2 : Q(Nat.mul $da $db = Nat.mul $k $dc) := (q(Eq.refl $t2) : Expr)
      return .isRat' dα qc nc dc q(isRat_add (f := $f) (.refl $f) $pa $pb $r1 $r2)
    match ra, rb with
    | .isBool .., _ | _, .isBool .. => failure
    | .isRat dα .., _ | _, .isRat dα .. => ratArm dα
    | .isNegNat rα .., _ | _, .isNegNat rα .. => intArm rα
    | .isNat _ na pa, .isNat sα nb pb =>
      haveI' : $e =Q $a + $b := ⟨⟩
      haveI' : $f =Q HAdd.hAdd := ⟨⟩
      assumeInstancesCommute
      have c : Q(ℕ) := mkRawNatLit (na.natLit! + nb.natLit!)
      haveI' : Nat.add $na $nb =Q $c := ⟨⟩
      return .isNat sα c q(isNat_add (f := $f) (.refl $f) $pa $pb (.refl $c))
  core

-- see note [norm_num lemma function equality]

theorem isInt_neg {α} [Ring α] : ∀ {f : α → α} {a : α} {a' b : ℤ},
    f = Neg.neg → IsInt a a' → Int.neg a' = b → IsInt (-a) b
  | _, _, _, _, rfl, ⟨rfl⟩, rfl => ⟨(Int.cast_neg ..).symm⟩

-- see note [norm_num lemma function equality]

theorem isRat_neg {α} [Ring α] : ∀ {f : α → α} {a : α} {n n' : ℤ} {d : ℕ},
    f = Neg.neg → IsRat a n d → Int.neg n = n' → IsRat (-a) n' d
                                                /-
                                                  α : Type u_1
                                                  inst✝ : Ring α
                                                  num✝ : Int
                                                  denom✝ : Nat
                                                  h : Invertible ↑denom✝
                                                  ⊢ Eq (Neg.neg (HMul.hMul (↑num✝) (Invertible.invOf ↑denom✝))) (HMul.hMul (↑num …
                                                -/
  | _, _, _, _, _, rfl, ⟨h, rfl⟩, rfl => ⟨h, by rw [← neg_mul, ← Int.cast_neg]; rfl⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `-a`,
such that `norm_num` successfully recognises `a`. -/
@[norm_num -_] def evalNeg : NormNumExt where eval {u α} e := do
  let .app (f : Q($α → $α)) (a : Q($α)) ← whnfR e | failure
  let ra ← derive a
  let rα ← inferRing α
  let ⟨(_f_eq : $f =Q Neg.neg)⟩ ← withNewMCtxDepth <| assertDefEqQ _ _
  haveI' _e_eq : $e =Q -$a := ⟨⟩
  let rec
  /-- Main part of `evalNeg`. -/
  core : Option (Result e) := do
    let intArm (rα : Q(Ring $α)) := do
      assumeInstancesCommute
      let ⟨za, na, pa⟩ ← ra.toInt rα
      let zb := -za
      have b := mkRawIntLit zb
      haveI' : Int.neg $na =Q $b := ⟨⟩
      return .isInt rα b zb q(isInt_neg (f := $f) (.refl $f) $pa (.refl $b))
    let ratArm (dα : Q(DivisionRing $α)) : Option (Result _) := do
      assumeInstancesCommute
      let ⟨qa, na, da, pa⟩ ← ra.toRat' dα
      let qb := -qa
      have nb := mkRawIntLit qb.num
      haveI' : Int.neg $na =Q $nb := ⟨⟩
      return .isRat' dα qb nb da q(isRat_neg (f := $f) (.refl $f) $pa (.refl $nb))
    match ra with
    | .isBool _ .. => failure
    | .isNat _ .. => intArm rα
    | .isNegNat rα .. => intArm rα
    | .isRat dα .. => ratArm dα
  core

-- see note [norm_num lemma function equality]

theorem isInt_sub {α} [Ring α] : ∀ {f : α → α → α} {a b : α} {a' b' c : ℤ},
    f = HSub.hSub → IsInt a a' → IsInt b b' → Int.sub a' b' = c → IsInt (f a b) c
  | _, _, _, _, _, _, rfl, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨(Int.cast_sub ..).symm⟩

-- see note [norm_num lemma function equality]

theorem isRat_sub {α} [Ring α] {f : α → α → α} {a b : α} {na nb nc : ℤ} {da db dc k : ℕ}
    (hf : f = HSub.hSub) (ra : IsRat a na da) (rb : IsRat b nb db)
    (h₁ : Int.sub (Int.mul na db) (Int.mul nb da) = Int.mul k nc)
    (h₂ : Nat.mul da db = Nat.mul k dc) :
    IsRat (f a b) nc dc := by
  /-
    α : Type u_1
    inst✝ : Ring α
    f : α → α → α
    a b : α
    na nb nc : Int
    da db dc k : Nat
    hf : Eq f HSub.hSub
    ra : Mathlib.Meta.NormNum.IsRat a na da
    rb : Mathlib.Meta.NormNum.IsRat b nb db
    h₁ : Eq ((na.mul ↑db).sub (nb.mul ↑da)) ((↑k).mul nc)
    h₂ : Eq (da.mul db) (k.mul dc)
    ⊢ Mathlib.Meta.NormNum.IsRat (f a b) nc dc
  -/
  rw [hf, sub_eq_add_neg]
  /-
    α : Type u_1
    inst✝ : Ring α
    f : α → α → α
    a b : α
    na nb nc : Int
    da db dc k : Nat
    hf : Eq f HSub.hSub
    ra : Mathlib.Meta.NormNum.IsRat a na da
    rb : Mathlib.Meta.NormNum.IsRat b nb db
    h₁ : Eq ((na.mul ↑db).sub (nb.mul ↑da)) ((↑k).mul nc)
    h₂ : Eq (da.mul db) (k.mul dc)
    ⊢ Mathlib.Meta.NormNum.IsRat (HAdd.hAdd a (Neg.neg b)) nc dc
  -/
  refine isRat_add rfl ra (isRat_neg (n' := -nb) rfl rb rfl) (k := k) (nc := nc) ?_ h₂
  /-
    α : Type u_1
    inst✝ : Ring α
    f : α → α → α
    a b : α
    na nb nc : Int
    da db dc k : Nat
    hf : Eq f HSub.hSub
    ra : Mathlib.Meta.NormNum.IsRat a na da
    rb : Mathlib.Meta.NormNum.IsRat b nb db
    h₁ : Eq ((na.mul ↑db).sub (nb.mul ↑da)) ((↑k).mul nc)
    h₂ : Eq (da.mul db) (k.mul dc)
    ⊢ Eq ((na.mul ↑db).add ((Neg.neg nb).mul ↑da)) ((↑k).mul nc)
  -/
  rw [show Int.mul (-nb) _ = _ from neg_mul ..]; exact h₁
                                                 /-
                                                   🎉 no goals
                                                 -/


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `a - b` in a ring,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num _ - _] def evalSub : NormNumExt where eval {u α} e := do
  let .app (.app (f : Q($α → $α → $α)) (a : Q($α))) (b : Q($α)) ← whnfR e | failure
  let rα ← inferRing α
  let ⟨(_f_eq : $f =Q HSub.hSub)⟩ ← withNewMCtxDepth <| assertDefEqQ _ _
  let ra ← derive a; let rb ← derive b
  haveI' _e_eq : $e =Q $a - $b := ⟨⟩
  let rec
  /-- Main part of `evalAdd`. -/
  core : Option (Result e) := do
    let intArm (rα : Q(Ring $α)) := do
      assumeInstancesCommute
      let ⟨za, na, pa⟩ ← ra.toInt rα; let ⟨zb, nb, pb⟩ ← rb.toInt rα
      let zc := za - zb
      have c := mkRawIntLit zc
      haveI' : Int.sub $na $nb =Q $c := ⟨⟩
      return Result.isInt rα c zc q(isInt_sub (f := $f) (.refl $f) $pa $pb (.refl $c))
    let ratArm (dα : Q(DivisionRing $α)) : Option (Result _) := do
      assumeInstancesCommute
      let ⟨qa, na, da, pa⟩ ← ra.toRat' dα; let ⟨qb, nb, db, pb⟩ ← rb.toRat' dα
      let qc := qa - qb
      let dd := qa.den * qb.den
      let k := dd / qc.den
      have t1 : Q(ℤ) := mkRawIntLit (k * qc.num)
      have t2 : Q(ℕ) := mkRawNatLit dd
      have nc : Q(ℤ) := mkRawIntLit qc.num
      have dc : Q(ℕ) := mkRawNatLit qc.den
      have k : Q(ℕ) := mkRawNatLit k
      let r1 : Q(Int.sub (Int.mul $na $db) (Int.mul $nb $da) = Int.mul $k $nc) :=
        (q(Eq.refl $t1) : Expr)
      let r2 : Q(Nat.mul $da $db = Nat.mul $k $dc) := (q(Eq.refl $t2) : Expr)
      return .isRat' dα qc nc dc q(isRat_sub (f := $f) (.refl $f) $pa $pb $r1 $r2)
    match ra, rb with
    | .isBool .., _ | _, .isBool .. => failure
    | .isRat dα .., _ | _, .isRat dα .. => ratArm dα
    | .isNegNat rα .., _ | _, .isNegNat rα ..
    | .isNat _ .., .isNat _ .. => intArm rα
  core

-- see note [norm_num lemma function equality]

theorem isNat_mul {α} [Semiring α] : ∀ {f : α → α → α} {a b : α} {a' b' c : ℕ},
    f = HMul.hMul → IsNat a a' → IsNat b b' → Nat.mul a' b' = c → IsNat (a * b) c
  | _, _, _, _, _, _, rfl, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨(Nat.cast_mul ..).symm⟩

-- see note [norm_num lemma function equality]

theorem isInt_mul {α} [Ring α] : ∀ {f : α → α → α} {a b : α} {a' b' c : ℤ},
    f = HMul.hMul → IsInt a a' → IsInt b b' → Int.mul a' b' = c → IsInt (a * b) c
  | _, _, _, _, _, _, rfl, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨(Int.cast_mul ..).symm⟩


theorem isRat_mul {α} [Ring α] {f : α → α → α} {a b : α} {na nb nc : ℤ} {da db dc k : ℕ} :
    f = HMul.hMul → IsRat a na da → IsRat b nb db →
    Int.mul na nb = Int.mul k nc →
    Nat.mul da db = Nat.mul k dc →
    IsRat (f a b) nc dc := by
  /-
    α : Type u_1
    inst✝ : Ring α
    f : α → α → α
    a b : α
    na nb nc : Int
    da db dc k : Nat
    ⊢ Eq f HMul.hMul → Mathlib.Meta.NormNum.IsRat a na da → Mathlib.Meta.NormNum.I …
  -/
  rintro rfl ⟨_, rfl⟩ ⟨_, rfl⟩ (h₁ : na * nb = k * nc) (h₂ : da * db = k * dc)
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    ⊢ Mathlib.Meta.NormNum.IsRat (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da …
  -/
  have : Invertible (↑(da * db) : α) := by simpa using invertibleMul (da:α) db
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this : Invertible ↑(HMul.hMul da db)
    ⊢ Mathlib.Meta.NormNum.IsRat (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da …
  -/
  have := invertibleOfMul' (α := α) h₂
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    ⊢ Mathlib.Meta.NormNum.IsRat (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da …
  -/
  refine ⟨this, ?_⟩
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    ⊢ Eq (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  have H := (Nat.cast_commute (α := α) da db).invOf_left.invOf_right.right_comm
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    ⊢ Eq (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  have h₁ := congr_arg (Int.cast (R := α)) h₁
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq ↑(HMul.hMul na nb) ↑(HMul.hMul (↑k) nc)
    ⊢ Eq (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  simp only [Int.cast_mul, Int.cast_natCast] at h₁
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HMul.hMul ↑na ↑nb) (HMul.hMul ↑k ↑nc)
    ⊢ Eq (HMul.hMul (HMul.hMul (↑na) (Invertible.invOf ↑da)) (HMul.hMul (↑nb) (Inv …
  -/
  simp only [← mul_assoc, (Nat.cast_commute (α := α) da nb).invOf_left.right_comm, h₁]
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HMul.hMul ↑na ↑nb) (HMul.hMul ↑k ↑nc)
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑k ↑nc) (Invertible.invOf ↑da)) (Inverti …
  -/
  have h₂ := congr_arg (↑nc * ↑· * (⅟↑da * ⅟↑db * ⅟↑dc : α)) h₂
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂✝ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HMul.hMul ↑na ↑nb) (HMul.hMul ↑k ↑nc)
    h₂ : Eq ((fun x => HMul.hMul (HMul.hMul ↑nc ↑x) (HMul.hMul (HMul.hMul (Inverti …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑k ↑nc) (Invertible.invOf ↑da)) (Inverti …
  -/
  simp only [Nat.cast_mul, ← mul_assoc] at h₂; rw [H] at h₂
  /-
    case mk.mk
    α : Type u_1
    inst✝ : Ring α
    na nb nc : Int
    da db dc k : Nat
    inv✝¹ : Invertible ↑da
    inv✝ : Invertible ↑db
    h₁✝ : Eq (HMul.hMul na nb) (HMul.hMul (↑k) nc)
    h₂✝ : Eq (HMul.hMul da db) (HMul.hMul k dc)
    this✝ : Invertible ↑(HMul.hMul da db)
    this : Invertible ↑dc
    H : ∀ (a : α), Eq (HMul.hMul (HMul.hMul a (Invertible.invOf ↑da)) (Invertible. …
    h₁ : Eq (HMul.hMul ↑na ↑nb) (HMul.hMul ↑k ↑nc)
    h₂ : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul ↑nc ↑da) ↑db) ( …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑k ↑nc) (Invertible.invOf ↑da)) (Inverti …
  -/
  simp only [mul_invOf_cancel_right'] at h₂; rw [h₂, Nat.cast_commute]
  simp only [mul_invOf_cancel_right,
    (Nat.cast_commute (α := α) da dc).invOf_left.invOf_right.right_comm,
    (Nat.cast_commute (α := α) db dc).invOf_left.invOf_right.right_comm]


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `a * b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num _ * _] def evalMul : NormNumExt where eval {u α} e := do
  let .app (.app (f : Q($α → $α → $α)) (a : Q($α))) (b : Q($α)) ← whnfR e | failure
  let sα ← inferSemiring α
  let ra ← derive a; let rb ← derive b
  guard <|← withNewMCtxDepth <| isDefEq f q(HMul.hMul (α := $α))
  haveI' : $f =Q HMul.hMul := ⟨⟩
  haveI' : $e =Q $a * $b := ⟨⟩
  let rec
  /-- Main part of `evalMul`. -/
  core : Option (Result e) := do
    let rec intArm (rα : Q(Ring $α)) := do
      assumeInstancesCommute
      let ⟨za, na, pa⟩ ← ra.toInt rα; let ⟨zb, nb, pb⟩ ← rb.toInt rα
      let zc := za * zb
      have c := mkRawIntLit zc
      haveI' : Int.mul $na $nb =Q $c := ⟨⟩
      return .isInt rα c zc q(isInt_mul (f := $f) (.refl $f) $pa $pb (.refl $c))
    let rec ratArm (dα : Q(DivisionRing $α)) : Option (Result _) := do
      assumeInstancesCommute
      let ⟨qa, na, da, pa⟩ ← ra.toRat' dα; let ⟨qb, nb, db, pb⟩ ← rb.toRat' dα
      let qc := qa * qb
      let dd := qa.den * qb.den
      let k := dd / qc.den
      have nc : Q(ℤ) := mkRawIntLit qc.num
      have dc : Q(ℕ) := mkRawNatLit qc.den
      have k : Q(ℕ) := mkRawNatLit k
      let r1 : Q(Int.mul $na $nb = Int.mul $k $nc) :=
        (q(Eq.refl (Int.mul $na $nb)) : Expr)
      have t2 : Q(ℕ) := mkRawNatLit dd
      let r2 : Q(Nat.mul $da $db = Nat.mul $k $dc) := (q(Eq.refl $t2) : Expr)
      return .isRat' dα qc nc dc q(isRat_mul (f := $f) (.refl $f) $pa $pb $r1 $r2)
    match ra, rb with
    | .isBool .., _ | _, .isBool .. => failure
    | .isRat dα .., _ | _, .isRat dα .. => ratArm dα
    | .isNegNat rα .., _ | _, .isNegNat rα .. => intArm rα
    | .isNat mα' na pa, .isNat mα nb pb =>
                         /-
                           «$α» : Type u
                           «$a» «$b» : «$α»
                           «$sα» : Semiring «$α»
                           «$f» : «$α» → «$α» → «$α» := HMul.hMul
                           «$e» : «$α» := HMul.hMul «$a» «$b»
                           «$mα'» : AddMonoidWithOne «$α»
                           «$na» : Nat
                           «$pa» : Mathlib.Meta.NormNum.IsNat «$a» «$na»
                           «$mα» : AddMonoidWithOne «$α»
                           «$nb» : Nat
                           «$pb» : Mathlib.Meta.NormNum.IsNat «$b» «$nb»
                           ⊢ AddMonoidWithOne «$α»
                         -/
      haveI' : $mα =Q by clear! $mα $mα'; apply AddCommMonoidWithOne.toAddMonoidWithOne := ⟨⟩
                                          /-
                                            🎉 no goals
                                          -/
      assumeInstancesCommute
      have c : Q(ℕ) := mkRawNatLit (na.natLit! * nb.natLit!)
      haveI' : Nat.mul $na $nb =Q $c := ⟨⟩
      return .isNat mα c q(isNat_mul (f := $f) (.refl $f) $pa $pb (.refl $c))
  core


theorem isRat_div {α : Type u} [DivisionRing α] : {a b : α} → {cn : ℤ} → {cd : ℕ} →
    IsRat (a * b⁻¹) cn cd → IsRat (a / b) cn cd
                        /-
                          α : Type u
                          inst✝ : DivisionRing α
                          x✝³ x✝² : α
                          x✝¹ : Int
                          x✝ : Nat
                          h : Mathlib.Meta.NormNum.IsRat (HMul.hMul x✝³ (Inv.inv x✝²)) x✝¹ x✝
                          ⊢ Mathlib.Meta.NormNum.IsRat (HDiv.hDiv x✝³ x✝²) x✝¹ x✝
                        -/
  | _, _, _, _, h => by simpa [div_eq_mul_inv] using h
                        /-
                          🎉 no goals
                        -/


/-- Helper function to synthesize a typed `DivisionRing α` expression. -/
def inferDivisionRing {u : Level} (α : Q(Type u)) : MetaM Q(DivisionRing $α) :=
  return ← synthInstanceQ (q(DivisionRing $α) : Q(Type u)) <|> throwError "not a division ring"


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `a / b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num _ / _] def evalDiv : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q($α))) (b : Q($α)) ← whnfR e | failure
  let dα ← inferDivisionRing α
  haveI' : $e =Q $a / $b := ⟨⟩
  guard <|← withNewMCtxDepth <| isDefEq f q(HDiv.hDiv (α := $α))
  let rab ← derive (q($a * $b⁻¹) : Q($α))
  let ⟨qa, na, da, pa⟩ ← rab.toRat' dα
  assumeInstancesCommute
  return .isRat' dα qa na da q(isRat_div $pa)


/-- The `norm_num` extension which identifies `True`. -/
@[norm_num True] def evalTrue : NormNumExt where eval {u α} e :=
  return (.isTrue q(True.intro) : Result q(True))


/-- The `norm_num` extension which identifies `False`. -/
@[norm_num False] def evalFalse : NormNumExt where eval {u α} e :=
  return (.isFalse q(not_false) : Result q(False))


/-- The `norm_num` extension which identifies expressions of the form `¬a`,
such that `norm_num` successfully recognises `a`. -/
@[norm_num ¬_] def evalNot : NormNumExt where eval {u α} e := do
  let .app (.const ``Not _) (a : Q(Prop)) ← whnfR e | failure
  guard <|← withNewMCtxDepth <| isDefEq α q(Prop)
  let ⟨b, p⟩ ← deriveBool q($a)
  match b with
  | true => return .isFalse q(not_not_intro $p)
  | false => return .isTrue q($p)


theorem isNat_eq_true [AddMonoidWithOne α] : {a b : α} → {c : ℕ} →
    IsNat a c → IsNat b c → a = b
  | _, _, _, ⟨rfl⟩, ⟨rfl⟩ => rfl


theorem ble_eq_false {x y : ℕ} : x.ble y = false ↔ y < x := by
  /-
    x y : Nat
    ⊢ Iff (Eq (x.ble y) Bool.false) (LT.lt y x)
  -/
  rw [← Nat.not_le, ← Bool.not_eq_true, Nat.ble_eq]
  /-
    🎉 no goals
  -/


theorem isInt_eq_true [Ring α] : {a b : α} → {z : ℤ} → IsInt a z → IsInt b z → a = b
  | _, _, _, ⟨rfl⟩, ⟨rfl⟩ => rfl


theorem isRat_eq_true [Ring α] : {a b : α} → {n : ℤ} → {d : ℕ} →
    IsRat a n d → IsRat b n d → a = b
                                         /-
                                           α : Type u
                                           inst✝ : Ring α
                                           num✝ : Int
                                           denom✝ : Nat
                                           inv✝¹ inv✝ : Invertible ↑denom✝
                                           ⊢ Eq (HMul.hMul (↑num✝) (Invertible.invOf ↑denom✝)) (HMul.hMul (↑num✝) (Invert …
                                         -/
  | _, _, _, _, ⟨_, rfl⟩, ⟨_, rfl⟩ => by congr; apply Subsingleton.elim
                                                /-
                                                  🎉 no goals
                                                -/


theorem eq_of_true {a b : Prop} (ha : a) (hb : b) : a = b := propext (iff_of_true ha hb)

theorem ne_of_false_of_true {a b : Prop} (ha : ¬a) (hb : b) : a ≠ b := mt (· ▸ hb) ha

theorem ne_of_true_of_false {a b : Prop} (ha : a) (hb : ¬b) : a ≠ b := mt (· ▸ ha) hb

theorem eq_of_false {a b : Prop} (ha : ¬a) (hb : ¬b) : a = b := propext (iff_of_false ha hb)


theorem isNat_natSucc : {a : ℕ} → {a' c : ℕ} →
    IsNat a a' → Nat.succ a' = c → IsNat (a.succ) c
                              /-
                                n✝ : Nat
                                ⊢ Eq (↑n✝).succ ↑n✝.succ
                              -/
  | _, _,_, ⟨rfl⟩, rfl => ⟨by simp⟩
                              /-
                                🎉 no goals
                              -/


/-- The `norm_num` extension which identifies expressions of the form `Nat.succ a`,
such that `norm_num` successfully recognises `a`. -/
@[norm_num Nat.succ _] def evalNatSucc : NormNumExt where eval {u α} e := do
  let .app f (a : Q(ℕ)) ← whnfR e | failure
  guard <|← withNewMCtxDepth <| isDefEq f q(Nat.succ)
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℕ := ⟨⟩
  haveI' : $e =Q Nat.succ $a := ⟨⟩
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let ⟨na, pa⟩ ← deriveNat a sℕ
  have nc : Q(ℕ) := mkRawNatLit (na.natLit!.succ)
  haveI' : $nc =Q ($na).succ := ⟨⟩
  return .isNat sℕ nc q(isNat_natSucc $pa (.refl $nc))


theorem isNat_natSub : {a b : ℕ} → {a' b' c : ℕ} →
    IsNat a a' → IsNat b b' → Nat.sub a' b' = c → IsNat (a - b) c
                                            /-
                                              n✝¹ n✝ : Nat
                                              ⊢ Eq (HSub.hSub ↑n✝¹ ↑n✝) ↑(n✝¹.sub n✝)
                                            -/
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨by simp⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- The `norm_num` extension which identifies expressions of the form `Nat.sub a b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℕ) - _] def evalNatSub :
    NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℕ))) (b : Q(ℕ)) ← whnfR e | failure
  -- We assert that the default instance for `HSub` is `Nat.sub` when the first parameter is `ℕ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(HSub.hSub (α := ℕ))
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℕ := ⟨⟩
  haveI' : $e =Q $a - $b := ⟨⟩
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let ⟨na, pa⟩ ← deriveNat a sℕ; let ⟨nb, pb⟩ ← deriveNat b sℕ
  have nc : Q(ℕ) := mkRawNatLit (na.natLit! - nb.natLit!)
  haveI' : Nat.sub $na $nb =Q $nc := ⟨⟩
  return .isNat sℕ nc q(isNat_natSub $pa $pb (.refl $nc))


theorem isNat_natMod : {a b : ℕ} → {a' b' c : ℕ} →
    IsNat a a' → IsNat b b' → Nat.mod a' b' = c → IsNat (a % b) c
                                            /-
                                              n✝¹ n✝ : Nat
                                              ⊢ Eq (HMod.hMod ↑n✝¹ ↑n✝) ↑(n✝¹.mod n✝)
                                            -/
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨by aesop⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- The `norm_num` extension which identifies expressions of the form `Nat.mod a b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℕ) % _] def evalNatMod :
    NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℕ))) (b : Q(ℕ)) ← whnfR e | failure
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℕ := ⟨⟩
  haveI' : $e =Q $a % $b := ⟨⟩
  -- We assert that the default instance for `HMod` is `Nat.mod` when the first parameter is `ℕ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(HMod.hMod (α := ℕ))
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let ⟨na, pa⟩ ← deriveNat a sℕ; let ⟨nb, pb⟩ ← deriveNat b sℕ
  have nc : Q(ℕ) := mkRawNatLit (na.natLit! % nb.natLit!)
  haveI' : Nat.mod $na $nb =Q $nc := ⟨⟩
  return .isNat sℕ nc q(isNat_natMod $pa $pb (.refl $nc))


theorem isNat_natDiv : {a b : ℕ} → {a' b' c : ℕ} →
    IsNat a a' → IsNat b b' → Nat.div a' b' = c → IsNat (a / b) c
                                            /-
                                              n✝¹ n✝ : Nat
                                              ⊢ Eq (HDiv.hDiv ↑n✝¹ ↑n✝) ↑(n✝¹.div n✝)
                                            -/
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨by aesop⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- The `norm_num` extension which identifies expressions of the form `Nat.div a b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℕ) / _]
def evalNatDiv : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℕ))) (b : Q(ℕ)) ← whnfR e | failure
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℕ := ⟨⟩
  haveI' : $e =Q $a / $b := ⟨⟩
  -- We assert that the default instance for `HDiv` is `Nat.div` when the first parameter is `ℕ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(HDiv.hDiv (α := ℕ))
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let ⟨na, pa⟩ ← deriveNat a sℕ; let ⟨nb, pb⟩ ← deriveNat b sℕ
  have nc : Q(ℕ) := mkRawNatLit (na.natLit! / nb.natLit!)
  haveI' : Nat.div $na $nb =Q $nc := ⟨⟩
  return .isNat sℕ nc q(isNat_natDiv $pa $pb (.refl $nc))


theorem isNat_dvd_true : {a b : ℕ} → {a' b' : ℕ} →
    IsNat a a' → IsNat b b' → Nat.mod b' a' = nat_lit 0 → a ∣ b
  | _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, e => Nat.dvd_of_mod_eq_zero e


theorem isNat_dvd_false : {a b : ℕ} → {a' b' c : ℕ} →
    IsNat a a' → IsNat b b' → Nat.mod b' a' = Nat.succ c → ¬a ∣ b
  | _, _, _, _, c, ⟨rfl⟩, ⟨rfl⟩, e => mt Nat.mod_eq_zero_of_dvd (e.symm ▸ Nat.succ_ne_zero c :)


/-- The `norm_num` extension which identifies expressions of the form `(a : ℕ) | b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℕ) ∣ _] def evalNatDvd : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℕ))) (b : Q(ℕ)) ← whnfR e | failure
  -- We assert that the default instance for `Dvd` is `Nat.dvd` when the first parameter is `ℕ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(Dvd.dvd (α := ℕ))
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let ⟨na, pa⟩ ← deriveNat a sℕ; let ⟨nb, pb⟩ ← deriveNat b sℕ
  match nb.natLit! % na.natLit! with
  | 0 =>
    have : Q(Nat.mod $nb $na = nat_lit 0) := (q(Eq.refl (nat_lit 0)) : Expr)
    return .isTrue q(isNat_dvd_true $pa $pb $this)
  | c+1 =>
    have nc : Q(ℕ) := mkRawNatLit c
    have : Q(Nat.mod $nb $na = Nat.succ $nc) := (q(Eq.refl (Nat.succ $nc)) : Expr)
    return .isFalse q(isNat_dvd_false $pa $pb $this)


