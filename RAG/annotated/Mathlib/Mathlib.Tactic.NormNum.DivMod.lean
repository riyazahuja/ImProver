lemma isInt_ediv_zero : ∀ {a b r : ℤ}, IsInt a r → IsNat b (nat_lit 0) → IsNat (a / b) (nat_lit 0)
                                 /-
                                   n✝ : Int
                                   ⊢ Eq (HDiv.hDiv ↑n✝ ↑0) ↑0
                                 -/
  | _, _, _, ⟨rfl⟩, ⟨rfl⟩ => ⟨by simp [Int.ediv_zero]⟩
                                 /-
                                   🎉 no goals
                                 -/


lemma isInt_ediv {a b q m a' : ℤ} {b' r : ℕ}
    (ha : IsInt a a') (hb : IsNat b b')
    (hm : q * b' = m) (h : r + m = a') (h₂ : Nat.blt r b' = true) :
    IsInt (a / b) q := ⟨by
  /-
    a b q m a' : Int
    b' r : Nat
    ha : Mathlib.Meta.NormNum.IsInt a a'
    hb : Mathlib.Meta.NormNum.IsNat b b'
    hm : Eq (HMul.hMul q ↑b') m
    h : Eq (HAdd.hAdd (↑r) m) a'
    h₂ : Eq (r.blt b') Bool.true
    ⊢ Eq (HDiv.hDiv a b) ↑q
  -/
  obtain ⟨⟨rfl⟩, ⟨rfl⟩⟩ := ha, hb
  /-
    case mk.mk
    a q m : Int
    b' r : Nat
    hm : Eq (HMul.hMul q ↑b') m
    h₂ : Eq (r.blt b') Bool.true
    h : Eq (HAdd.hAdd (↑r) m) a
    ⊢ Eq (HDiv.hDiv a ↑b') ↑q
  -/
  simp only [Nat.blt_eq] at h₂; simp only [← h, ← hm, Int.cast_id]
  /-
    case mk.mk
    a q m : Int
    b' r : Nat
    hm : Eq (HMul.hMul q ↑b') m
    h : Eq (HAdd.hAdd (↑r) m) a
    h₂ : LT.lt r b'
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (↑r) (HMul.hMul q ↑b')) ↑b') q
  -/
  rw [Int.add_mul_ediv_right _ _ (Int.ofNat_ne_zero.2 ((Nat.zero_le ..).trans_lt h₂).ne')]
  /-
    case mk.mk
    a q m : Int
    b' r : Nat
    hm : Eq (HMul.hMul q ↑b') m
    h : Eq (HAdd.hAdd (↑r) m) a
    h₂ : LT.lt r b'
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv ↑r ↑b') q) q
  -/
  rw [Int.ediv_eq_zero_of_lt, zero_add] <;> [simp; simpa using h₂]⟩
  /-
    🎉 no goals
  -/


lemma isInt_ediv_neg {a b q q' : ℤ} (h : IsInt (a / -b) q) (hq : -q = q') : IsInt (a / b) q' :=
      /-
        a b q q' : Int
        h : Mathlib.Meta.NormNum.IsInt (HDiv.hDiv a (Neg.neg b)) q
        hq : Eq (Neg.neg q) q'
        ⊢ Eq (HDiv.hDiv a b) ↑q'
      -/
  ⟨by rw [Int.cast_id, ← hq, ← @Int.cast_id q, ← h.out, ← Int.ediv_neg, Int.neg_neg]⟩
      /-
        🎉 no goals
      -/


lemma isNat_neg_of_isNegNat {a : ℤ} {b : ℕ} (h : IsInt a (.negOfNat b)) : IsNat (-a) b :=
      /-
        a : Int
        b : Nat
        h : Mathlib.Meta.NormNum.IsInt a (Int.negOfNat b)
        ⊢ Eq (Neg.neg a) ↑b
      -/
  ⟨by simp [h.out]⟩
      /-
        🎉 no goals
      -/


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `Int.ediv a b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℤ) / _, Int.ediv _ _]
partial def evalIntDiv : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℤ))) (b : Q(ℤ)) ← whnfR e | failure
  -- We assert that the default instance for `HDiv` is `Int.div` when the first parameter is `ℤ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(HDiv.hDiv (α := ℤ))
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℤ := ⟨⟩
  haveI' : $e =Q ($a / $b) := ⟨⟩
  let rℤ : Q(Ring ℤ) := q(Int.instRing)
  let ⟨za, na, pa⟩ ← (← derive a).toInt rℤ
  match ← derive (u := .zero) b with
  | .isNat inst nb pb =>
    assumeInstancesCommute
    if nb.natLit! == 0 then
      have _ : $nb =Q nat_lit 0 := ⟨⟩
      return .isNat q(instAddMonoidWithOne) q(nat_lit 0) q(isInt_ediv_zero $pa $pb)
    else
      let ⟨zq, q, p⟩ := core a na za pa b nb pb
      return .isInt rℤ q zq p
  | .isNegNat _ nb pb =>
    assumeInstancesCommute
    let ⟨zq, q, p⟩ := core a na za pa q(-$b) nb q(isNat_neg_of_isNegNat $pb)
    have q' := mkRawIntLit (-zq)
    have : Q(-$q = $q') := (q(Eq.refl $q') :)
    return .isInt rℤ q' (-zq) q(isInt_ediv_neg $p $this)
  | _ => failure
where
  /-- Given a result for evaluating `a b` in `ℤ` where `b > 0`, evaluate `a / b`. -/
  core (a na : Q(ℤ)) (za : ℤ) (pa : Q(IsInt $a $na))
      (b : Q(ℤ)) (nb : Q(ℕ)) (pb : Q(IsNat $b $nb)) :
      ℤ × (q : Q(ℤ)) × Q(IsInt ($a / $b) $q) :=
    let b := nb.natLit!
    let q := za / b
    have nq := mkRawIntLit q
    let r := za.natMod b
    have nr : Q(ℕ) := mkRawNatLit r
    let m := q * b
    have nm := mkRawIntLit m
    have pf₁ : Q($nq * $nb = $nm) := (q(Eq.refl $nm) :)
    have pf₂ : Q($nr + $nm = $na) := (q(Eq.refl $na) :)
    have pf₃ : Q(Nat.blt $nr $nb = true) := (q(Eq.refl true) :)
    ⟨q, nq, q(isInt_ediv $pa $pb $pf₁ $pf₂ $pf₃)⟩


lemma isInt_emod_zero : ∀ {a b r : ℤ}, IsInt a r → IsNat b (nat_lit 0) → IsInt (a % b) r
                            /-
                              x✝¹ x✝ : Int
                              e : Mathlib.Meta.NormNum.IsInt x✝¹ x✝
                              ⊢ Mathlib.Meta.NormNum.IsInt (HMod.hMod x✝¹ ↑0) x✝
                            -/
  | _, _, _, e, ⟨rfl⟩ => by simp [e]
                            /-
                              🎉 no goals
                            -/


lemma isInt_emod {a b q m a' : ℤ} {b' r : ℕ}
    (ha : IsInt a a') (hb : IsNat b b')
    (hm : q * b' = m) (h : r + m = a') (h₂ : Nat.blt r b' = true) :
    IsNat (a % b) r := ⟨by
  /-
    a b q m a' : Int
    b' r : Nat
    ha : Mathlib.Meta.NormNum.IsInt a a'
    hb : Mathlib.Meta.NormNum.IsNat b b'
    hm : Eq (HMul.hMul q ↑b') m
    h : Eq (HAdd.hAdd (↑r) m) a'
    h₂ : Eq (r.blt b') Bool.true
    ⊢ Eq (HMod.hMod a b) ↑r
  -/
  obtain ⟨⟨rfl⟩, ⟨rfl⟩⟩ := ha, hb
  /-
    case mk.mk
    a q m : Int
    b' r : Nat
    hm : Eq (HMul.hMul q ↑b') m
    h₂ : Eq (r.blt b') Bool.true
    h : Eq (HAdd.hAdd (↑r) m) a
    ⊢ Eq (HMod.hMod a ↑b') ↑r
  -/
  simp only [← h, ← hm, Int.add_mul_emod_self]
  /-
    case mk.mk
    a q m : Int
    b' r : Nat
    hm : Eq (HMul.hMul q ↑b') m
    h₂ : Eq (r.blt b') Bool.true
    h : Eq (HAdd.hAdd (↑r) m) a
    ⊢ Eq (HMod.hMod ↑r ↑b') ↑r
  -/
  rw [Int.emod_eq_of_lt] <;> [simp; simpa using h₂]⟩
  /-
    🎉 no goals
  -/


lemma isInt_emod_neg {a b : ℤ} {r : ℕ} (h : IsNat (a % -b) r) : IsNat (a % b) r :=
      /-
        a b : Int
        r : Nat
        h : Mathlib.Meta.NormNum.IsNat (HMod.hMod a (Neg.neg b)) r
        ⊢ Eq (HMod.hMod a b) ↑r
      -/
  ⟨by rw [← Int.emod_neg, h.out]⟩
      /-
        🎉 no goals
      -/


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `Int.emod a b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℤ) % _, Int.emod _ _]
partial def evalIntMod : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℤ))) (b : Q(ℤ)) ← whnfR e | failure
  -- We assert that the default instance for `HMod` is `Int.mod` when the first parameter is `ℤ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(HMod.hMod (α := ℤ))
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q ℤ := ⟨⟩
  haveI' : $e =Q ($a % $b) := ⟨⟩
  let rℤ : Q(Ring ℤ) := q(Int.instRing)
  let some ⟨za, na, pa⟩ := (← derive a).toInt rℤ | failure
  go a na za pa b (← derive (u := .zero) b)
where
  /-- Given a result for evaluating `a b` in `ℤ`, evaluate `a % b`. -/
  go (a na : Q(ℤ)) (za : ℤ) (pa : Q(IsInt $a $na))
      (b : Q(ℤ)) : Result b → Option (Result q($a % $b))
    | .isNat inst nb pb => do
      assumeInstancesCommute
      if nb.natLit! == 0 then
        have _ : $nb =Q nat_lit 0 := ⟨⟩
        return .isInt q(Int.instRing) na za q(isInt_emod_zero $pa $pb)
      else
        let ⟨r, p⟩ := core a na za pa b nb pb
        return .isNat q(instAddMonoidWithOne) r p
    | .isNegNat _ nb pb => do
      assumeInstancesCommute
      let ⟨r, p⟩ := core a na za pa q(-$b) nb q(isNat_neg_of_isNegNat $pb)
      return .isNat q(instAddMonoidWithOne) r q(isInt_emod_neg $p)
    | _ => none

  /-- Given a result for evaluating `a b` in `ℤ` where `b > 0`, evaluate `a % b`. -/
  core (a na : Q(ℤ)) (za : ℤ) (pa : Q(IsInt $a $na))
      (b : Q(ℤ)) (nb : Q(ℕ)) (pb : Q(IsNat $b $nb)) :
      (r : Q(ℕ)) × Q(IsNat ($a % $b) $r) :=
    let b := nb.natLit!
    let q := za / b
    have nq := mkRawIntLit q
    let r := za.natMod b
    have nr : Q(ℕ) := mkRawNatLit r
    let m := q * b
    have nm := mkRawIntLit m
    have pf₁ : Q($nq * $nb = $nm) := (q(Eq.refl $nm) :)
    have pf₂ : Q($nr + $nm = $na) := (q(Eq.refl $na) :)
    have pf₃ : Q(Nat.blt $nr $nb = true) := (q(Eq.refl true) :)
    ⟨nr, q(isInt_emod $pa $pb $pf₁ $pf₂ $pf₃)⟩


theorem isInt_dvd_true : {a b : ℤ} → {a' b' c : ℤ} →
    IsInt a a' → IsInt b b' → Int.mul a' c = b' → a ∣ b
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨_, rfl⟩


theorem isInt_dvd_false : {a b : ℤ} → {a' b' : ℤ} →
    IsInt a a' → IsInt b b' → Int.emod b' a' != 0 → ¬a ∣ b
                                                                  /-
                                                                    n✝¹ n✝ : Int
                                                                    e : Eq (bne (n✝.emod n✝¹) 0) Bool.true
                                                                    ⊢ Not (Eq (HMod.hMod ↑n✝ ↑n✝¹) 0)
                                                                  -/
  | _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, e => mt Int.emod_eq_zero_of_dvd (by simpa using e)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


attribute [local instance] monadLiftOptionMetaM in
/-- The `norm_num` extension which identifies expressions of the form `(a : ℤ) ∣ b`,
such that `norm_num` successfully recognises both `a` and `b`. -/
@[norm_num (_ : ℤ) ∣ _] def evalIntDvd : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℤ))) (b : Q(ℤ)) ← whnfR e | failure
  haveI' : u =QL 0 := ⟨⟩; haveI' : $α =Q Prop := ⟨⟩
  haveI' : $e =Q ($a ∣ $b) := ⟨⟩
  -- We assert that the default instance for `Dvd` is `Int.dvd` when the first parameter is `ℕ`.
  guard <|← withNewMCtxDepth <| isDefEq f q(Dvd.dvd (α := ℤ))
  let rℤ : Q(Ring ℤ) := q(Int.instRing)
  let ⟨za, na, pa⟩ ← (← derive a).toInt rℤ
  let ⟨zb, nb, pb⟩ ← (← derive b).toInt rℤ
  if zb % za == 0 then
    let zc := zb / za
    have c := mkRawIntLit zc
    haveI' : Int.mul $na $c =Q $nb := ⟨⟩
    return .isTrue q(isInt_dvd_true $pa $pb (.refl $nb))
  else
    have : Q(Int.emod $nb $na != 0) := (q(Eq.refl true) : Expr)
    return .isFalse q(isInt_dvd_false $pa $pb $this)


