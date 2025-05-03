           /-
             ⊢ Lean.Name
           -/
initialize registerTraceClass `CancelDenoms
           /-
             🎉 no goals
           -/


theorem mul_subst {α} [CommRing α] {n1 n2 k e1 e2 t1 t2 : α}
    (h1 : n1 * e1 = t1) (h2 : n2 * e2 = t2) (h3 : n1 * n2 = k) : k * (e1 * e2) = t1 * t2 := by
  rw [← h3, mul_comm n1, mul_assoc n2, ← mul_assoc n1, h1,
      ← mul_assoc n2, mul_comm n2, mul_assoc, h2]


theorem div_subst {α} [Field α] {n1 n2 k e1 e2 t1 : α}
    (h1 : n1 * e1 = t1) (h2 : n2 / e2 = 1) (h3 : n1 * n2 = k) : k * (e1 / e2) = t1 := by
  /-
    α : Type u_1
    inst✝ : Field α
    n1 n2 k e1 e2 t1 : α
    h1 : Eq (HMul.hMul n1 e1) t1
    h2 : Eq (HDiv.hDiv n2 e2) 1
    h3 : Eq (HMul.hMul n1 n2) k
    ⊢ Eq (HMul.hMul k (HDiv.hDiv e1 e2)) t1
  -/
  rw [← h3, mul_assoc, mul_div_left_comm, h2, ← mul_assoc, h1, mul_comm, one_mul]
  /-
    🎉 no goals
  -/


theorem cancel_factors_eq_div {α} [Field α] {n e e' : α}
    (h : n * e = e') (h2 : n ≠ 0) : e = e' / n :=
                            /-
                              α : Type u_1
                              inst✝ : Field α
                              n e e' : α
                              h : Eq (HMul.hMul n e) e'
                              h2 : Ne n 0
                              ⊢ Eq (HMul.hMul e n) e'
                            -/
  eq_div_of_mul_eq h2 <| by rwa [mul_comm] at h
                            /-
                              🎉 no goals
                            -/


theorem add_subst {α} [Ring α] {n e1 e2 t1 t2 : α} (h1 : n * e1 = t1) (h2 : n * e2 = t2) :
                                  /-
                                    α : Type u_1
                                    inst✝ : Ring α
                                    n e1 e2 t1 t2 : α
                                    h1 : Eq (HMul.hMul n e1) t1
                                    h2 : Eq (HMul.hMul n e2) t2
                                    ⊢ Eq (HMul.hMul n (HAdd.hAdd e1 e2)) (HAdd.hAdd t1 t2)
                                  -/
    n * (e1 + e2) = t1 + t2 := by simp [left_distrib, *]
                                  /-
                                    🎉 no goals
                                  -/


theorem sub_subst {α} [Ring α] {n e1 e2 t1 t2 : α} (h1 : n * e1 = t1) (h2 : n * e2 = t2) :
                                  /-
                                    α : Type u_1
                                    inst✝ : Ring α
                                    n e1 e2 t1 t2 : α
                                    h1 : Eq (HMul.hMul n e1) t1
                                    h2 : Eq (HMul.hMul n e2) t2
                                    ⊢ Eq (HMul.hMul n (HSub.hSub e1 e2)) (HSub.hSub t1 t2)
                                  -/
    n * (e1 - e2) = t1 - t2 := by simp [left_distrib, *, sub_eq_add_neg]
                                  /-
                                    🎉 no goals
                                  -/


                                                                                /-
                                                                                  α : Type u_1
                                                                                  inst✝ : Ring α
                                                                                  n e t : α
                                                                                  h1 : Eq (HMul.hMul n e) t
                                                                                  ⊢ Eq (HMul.hMul n (Neg.neg e)) (Neg.neg t)
                                                                                -/
theorem neg_subst {α} [Ring α] {n e t : α} (h1 : n * e = t) : n * -e = -t := by simp [*]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem pow_subst {α} [CommRing α] {n e1 t1 k l : α} {e2 : ℕ}
    (h1 : n * e1 = t1) (h2 : l * n ^ e2 = k) : k * (e1 ^ e2) = l * t1 ^ e2 := by
  /-
    α : Type u_1
    inst✝ : CommRing α
    n e1 t1 k l : α
    e2 : Nat
    h1 : Eq (HMul.hMul n e1) t1
    h2 : Eq (HMul.hMul l (HPow.hPow n e2)) k
    ⊢ Eq (HMul.hMul k (HPow.hPow e1 e2)) (HMul.hMul l (HPow.hPow t1 e2))
  -/
  rw [← h2, ← h1, mul_pow, mul_assoc]
  /-
    🎉 no goals
  -/


theorem inv_subst {α} [Field α] {n k e : α} (h2 : e ≠ 0) (h3 : n * e = k) :
                         /-
                           α : Type u_1
                           inst✝ : Field α
                           n k e : α
                           h2 : Ne e 0
                           h3 : Eq (HMul.hMul n e) k
                           ⊢ Eq (HMul.hMul k (Inv.inv e)) n
                         -/
    k * (e ⁻¹) = n := by rw [← div_eq_mul_inv, ← h3, mul_div_cancel_right₀ _ h2]
                         /-
                           🎉 no goals
                         -/


theorem cancel_factors_lt {α} [LinearOrderedField α] {a b ad bd a' b' gcd : α}
    (ha : ad * a = a') (hb : bd * b = b') (had : 0 < ad) (hbd : 0 < bd) (hgcd : 0 < gcd) :
    (a < b) = (1 / gcd * (bd * a') < 1 / gcd * (ad * b')) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b ad bd a' b' gcd : α
    ha : Eq (HMul.hMul ad a) a'
    hb : Eq (HMul.hMul bd b) b'
    had : LT.lt 0 ad
    hbd : LT.lt 0 bd
    hgcd : LT.lt 0 gcd
    ⊢ Eq (LT.lt a b) (LT.lt (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul bd a')) (HMul. …
  -/
  rw [mul_lt_mul_left, ← ha, ← hb, ← mul_assoc, ← mul_assoc, mul_comm bd, mul_lt_mul_left]
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : LT.lt 0 ad
      hbd : LT.lt 0 bd
      hgcd : LT.lt 0 gcd
      ⊢ LT.lt 0 (HMul.hMul ad bd)
    -/
  · exact mul_pos had hbd
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : LT.lt 0 ad
      hbd : LT.lt 0 bd
      hgcd : LT.lt 0 gcd
      ⊢ LT.lt 0 (HDiv.hDiv 1 gcd)
    -/
  · exact one_div_pos.2 hgcd
    /-
      🎉 no goals
    -/


theorem cancel_factors_le {α} [LinearOrderedField α] {a b ad bd a' b' gcd : α}
    (ha : ad * a = a') (hb : bd * b = b') (had : 0 < ad) (hbd : 0 < bd) (hgcd : 0 < gcd) :
    (a ≤ b) = (1 / gcd * (bd * a') ≤ 1 / gcd * (ad * b')) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b ad bd a' b' gcd : α
    ha : Eq (HMul.hMul ad a) a'
    hb : Eq (HMul.hMul bd b) b'
    had : LT.lt 0 ad
    hbd : LT.lt 0 bd
    hgcd : LT.lt 0 gcd
    ⊢ Eq (LE.le a b) (LE.le (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul bd a')) (HMul. …
  -/
  rw [mul_le_mul_left, ← ha, ← hb, ← mul_assoc, ← mul_assoc, mul_comm bd, mul_le_mul_left]
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : LT.lt 0 ad
      hbd : LT.lt 0 bd
      hgcd : LT.lt 0 gcd
      ⊢ LT.lt 0 (HMul.hMul ad bd)
    -/
  · exact mul_pos had hbd
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : LT.lt 0 ad
      hbd : LT.lt 0 bd
      hgcd : LT.lt 0 gcd
      ⊢ LT.lt 0 (HDiv.hDiv 1 gcd)
    -/
  · exact one_div_pos.2 hgcd
    /-
      🎉 no goals
    -/


theorem cancel_factors_eq {α} [Field α] {a b ad bd a' b' gcd : α} (ha : ad * a = a')
    (hb : bd * b = b') (had : ad ≠ 0) (hbd : bd ≠ 0) (hgcd : gcd ≠ 0) :
    (a = b) = (1 / gcd * (bd * a') = 1 / gcd * (ad * b')) := by
  /-
    α : Type u_1
    inst✝ : Field α
    a b ad bd a' b' gcd : α
    ha : Eq (HMul.hMul ad a) a'
    hb : Eq (HMul.hMul bd b) b'
    had : Ne ad 0
    hbd : Ne bd 0
    hgcd : Ne gcd 0
    ⊢ Eq (Eq a b) (Eq (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul bd a')) (HMul.hMul ( …
  -/
  rw [← ha, ← hb, ← mul_assoc bd, ← mul_assoc ad, mul_comm bd]
  /-
    α : Type u_1
    inst✝ : Field α
    a b ad bd a' b' gcd : α
    ha : Eq (HMul.hMul ad a) a'
    hb : Eq (HMul.hMul bd b) b'
    had : Ne ad 0
    hbd : Ne bd 0
    hgcd : Ne gcd 0
    ⊢ Eq (Eq a b) (Eq (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul (HMul.hMul ad bd) a) …
  -/
  ext; constructor
    /-
      case a.mp
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      ⊢ Eq a b → Eq (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul (HMul.hMul ad bd) a)) (H …
    -/
  · rintro rfl
    /-
      case a.mp
      α : Type u_1
      inst✝ : Field α
      a ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      hb : Eq (HMul.hMul bd a) b'
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul (HMul.hMul ad bd) a)) (HMul.hMul  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case a.mpr
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul (HMul.hMul ad bd) a)) (HMul.hMul  …
    -/
  · intro h
    /-
      case a.mpr
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      h : Eq (HMul.hMul (HDiv.hDiv 1 gcd) (HMul.hMul (HMul.hMul ad bd) a)) (HMul.hMu …
      ⊢ Eq a b
    -/
    simp only [← mul_assoc] at h
    /-
      case a.mpr
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      h : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HDiv.hDiv 1 gcd) ad) bd) a) (HMul.hMu …
      ⊢ Eq a b
    -/
    refine mul_left_cancel₀ (mul_ne_zero ?_ ?_) h
    /-
      case a.mpr.refine_1
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      h : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HDiv.hDiv 1 gcd) ad) bd) a) (HMul.hMu …
      ⊢ Ne (HMul.hMul (HDiv.hDiv 1 gcd) ad) 0
    -/
    on_goal 1 => apply mul_ne_zero
    /-
      case a.mpr.refine_1.ha
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      h : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HDiv.hDiv 1 gcd) ad) bd) a) (HMul.hMu …
      ⊢ Ne (HDiv.hDiv 1 gcd) 0
    -/
    on_goal 1 => apply div_ne_zero
      /-
        case a.mpr.refine_1.ha.ha
        α : Type u_1
        inst✝ : Field α
        a b ad bd a' b' gcd : α
        ha : Eq (HMul.hMul ad a) a'
        hb : Eq (HMul.hMul bd b) b'
        had : Ne ad 0
        hbd : Ne bd 0
        hgcd : Ne gcd 0
        h : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HDiv.hDiv 1 gcd) ad) bd) a) (HMul.hMu …
        ⊢ Ne 1 0
      -/
    · exact one_ne_zero
      /-
        🎉 no goals
      -/
    /-
      case a.mpr.refine_1.ha.hb
      α : Type u_1
      inst✝ : Field α
      a b ad bd a' b' gcd : α
      ha : Eq (HMul.hMul ad a) a'
      hb : Eq (HMul.hMul bd b) b'
      had : Ne ad 0
      hbd : Ne bd 0
      hgcd : Ne gcd 0
      h : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HDiv.hDiv 1 gcd) ad) bd) a) (HMul.hMu …
      ⊢ Ne gcd 0
    -/
    all_goals assumption
    /-
      🎉 no goals
    -/


theorem cancel_factors_ne {α} [Field α] {a b ad bd a' b' gcd : α} (ha : ad * a = a')
    (hb : bd * b = b') (had : ad ≠ 0) (hbd : bd ≠ 0) (hgcd : gcd ≠ 0) :
    (a ≠ b) = (1 / gcd * (bd * a') ≠ 1 / gcd * (ad * b')) := by
  classical
  rw [eq_iff_iff, not_iff_not, cancel_factors_eq ha hb had hbd hgcd]


/--
`findCancelFactor e` produces a natural number `n`, such that multiplying `e` by `n` will
be able to cancel all the numeric denominators in `e`. The returned `Tree` describes how to
distribute the value `n` over products inside `e`.
-/
partial def findCancelFactor (e : Expr) : ℕ × Tree ℕ :=
  match e.getAppFnArgs with
  | (``HAdd.hAdd, #[_, _, _, _, e1, e2]) | (``HSub.hSub, #[_, _, _, _, e1, e2]) =>
    let (v1, t1) := findCancelFactor e1
    let (v2, t2) := findCancelFactor e2
    let lcm := v1.lcm v2
    (lcm, .node lcm t1 t2)
  | (``HMul.hMul, #[_, _, _, _, e1, e2]) =>
    let (v1, t1) := findCancelFactor e1
    let (v2, t2) := findCancelFactor e2
    let pd := v1 * v2
    (pd, .node pd t1 t2)
  | (``HDiv.hDiv, #[_, _, _, _, e1, e2]) =>
    -- If e2 is a rational, then it's a natural number due to the simp lemmas in `deriveThms`.
    match e2.nat? with
    | some q =>
      let (v1, t1) := findCancelFactor e1
      let n := v1 * q
      (n, .node n t1 <| .node q .nil .nil)
    | none => (1, .node 1 .nil .nil)
  | (``Neg.neg, #[_, _, e]) => findCancelFactor e
  | (``HPow.hPow, #[_, ℕ, _, _, e1, e2]) =>
    match e2.nat? with
    | some k =>
      let (v1, t1) := findCancelFactor e1
      let n := v1 ^ k
      (n, .node n t1 <| .node k .nil .nil)
    | none => (1, .node 1 .nil .nil)
  | (``Inv.inv, #[_, _, e]) =>
    match e.nat? with
    | some q => (q, .node q .nil <| .node q .nil .nil)
    | none => (1, .node 1 .nil .nil)
  | _ => (1, .node 1 .nil .nil)


def synthesizeUsingNormNum (type : Q(Prop)) : MetaM Q($type) := do
  try
    synthesizeUsingTactic' type (← `(tactic| norm_num))
  catch e =>
    throwError "Could not prove {type} using norm_num. {e.toMessageData}"


/-- `CancelResult mα e v'` provides a value for `v * e` where the denominators have been cancelled.
-/
structure CancelResult {u : Level} {α : Q(Type u)} (mα : Q(Mul $α)) (e : Q($α)) (v : Q($α)) where
  /-- An expression with denominators cancelled. -/
  cancelled : Q($α)
  /-- The proof that `cancelled` is valid. -/
  pf : Q($v * $e = $cancelled)


/--
`mkProdPrf α sα v v' tr e` produces a proof of `v'*e = e'`, where numeric denominators have been
canceled in `e'`, distributing `v` proportionally according to the tree `tr` computed
by `findCancelFactor`.

The `v'` argument is a numeral expression corresponding to `v`, which we need in order to state
the return type accurately.
-/
partial def mkProdPrf {u : Level} (α : Q(Type u)) (sα : Q(Field $α)) (v : ℕ) (v' : Q($α))
    (t : Tree ℕ) (e : Q($α)) : MetaM (CancelResult q(inferInstance) e v') := do
  let amwo : Q(AddMonoidWithOne $α) := q(inferInstance)
  trace[CancelDenoms] "mkProdPrf {e} {v}"
  match t, e with
  | .node _ lhs rhs, ~q($e1 + $e2) => do
    let ⟨v1, hv1⟩ ← mkProdPrf α sα v v' lhs e1
    let ⟨v2, hv2⟩ ← mkProdPrf α sα v v' rhs e2
    return ⟨q($v1 + $v2), q(CancelDenoms.add_subst $hv1 $hv2)⟩
  | .node _ lhs rhs, ~q($e1 - $e2) => do
    let ⟨v1, hv1⟩ ← mkProdPrf α sα v v' lhs e1
    let ⟨v2, hv2⟩ ← mkProdPrf α sα v v' rhs e2
    return ⟨q($v1 - $v2), q(CancelDenoms.sub_subst $hv1 $hv2)⟩
  | .node _ lhs@(.node ln _ _) rhs, ~q($e1 * $e2) => do
    trace[CancelDenoms] "recursing into mul"
    have ln' := (← mkOfNat α amwo <| mkRawNatLit ln).1
    have vln' := (← mkOfNat α amwo <| mkRawNatLit (v/ln)).1
    let ⟨v1, hv1⟩ ← mkProdPrf α sα ln ln' lhs e1
    let ⟨v2, hv2⟩ ← mkProdPrf α sα (v / ln) vln' rhs e2
    let npf ← synthesizeUsingNormNum q($ln' * $vln' = $v')
    return ⟨q($v1 * $v2), q(CancelDenoms.mul_subst $hv1 $hv2 $npf)⟩
  | .node _ lhs (.node rn _ _), ~q($e1 / $e2) => do
    -- Invariant: e2 is equal to the natural number rn
    have rn' := (← mkOfNat α amwo <| mkRawNatLit rn).1
    have vrn' := (← mkOfNat α amwo <| mkRawNatLit <| v / rn).1
    let ⟨v1, hv1⟩ ← mkProdPrf α sα (v / rn) vrn' lhs e1
    let npf ← synthesizeUsingNormNum q($rn' / $e2 = 1)
    let npf2 ← synthesizeUsingNormNum q($vrn' * $rn' = $v')
    return ⟨q($v1), q(CancelDenoms.div_subst $hv1 $npf $npf2)⟩
  | t, ~q(-$e) => do
    let ⟨v, hv⟩ ← mkProdPrf α sα v v' t e
    return ⟨q(-$v), q(CancelDenoms.neg_subst $hv)⟩
  | .node _ lhs@(.node k1 _ _) (.node k2 .nil .nil), ~q($e1 ^ $e2) => do
    have k1' := (← mkOfNat α amwo <| mkRawNatLit k1).1
    let ⟨v1, hv1⟩ ← mkProdPrf α sα k1 k1' lhs e1
    have l : ℕ := v / (k1 ^ k2)
    have l' := (← mkOfNat α amwo <| mkRawNatLit l).1
    let npf ← synthesizeUsingNormNum q($l' * $k1' ^ $e2 = $v')
    return ⟨q($l' * $v1 ^ $e2), q(CancelDenoms.pow_subst $hv1 $npf)⟩
  | .node _ .nil (.node rn _ _), ~q($ei ⁻¹) => do
    have rn' := (← mkOfNat α amwo <| mkRawNatLit rn).1
    have vrn' := (← mkOfNat α amwo <| mkRawNatLit <| v / rn).1
    have _ : $rn' =Q $ei := ⟨⟩
    let npf ← synthesizeUsingNormNum q($rn' ≠ 0)
    let npf2 ← synthesizeUsingNormNum q($vrn' * $rn' = $v')
    return ⟨q($vrn'), q(CancelDenoms.inv_subst $npf $npf2)⟩
  | _, _ => do
    return ⟨q($v' * $e), q(rfl)⟩


/-- Theorems to get expression into a form that `findCancelFactor` and `mkProdPrf`
can more easily handle. These are important for dividing by rationals and negative integers. -/
def deriveThms : List Name :=
  [``div_div_eq_mul_div, ``div_neg]


/-- Helper lemma to chain together a `simp` proof and the result of `mkProdPrf`. -/
theorem derive_trans {α} [Mul α] {a b c d : α} (h : a = b) (h' : c * b = d) : c * a = d := h ▸ h'


/-- Helper lemma to chain together two `simp` proofs and the result of `mkProdPrf`. -/
theorem derive_trans₂ {α} [Mul α] {a b c d e : α} (h : a = b) (h' : b = c) (h'' : d * c = e) :
    d * a = e := h ▸ h' ▸ h''


/--
Given `e`, a term with rational division, produces a natural number `n` and a proof of `n*e = e'`,
where `e'` has no division. Assumes "well-behaved" division.
-/
def derive (e : Expr) : MetaM (ℕ × Expr) := do
  trace[CancelDenoms] "e = {e}"
  let eSimp ← simpOnlyNames (config := Simp.neutralConfig) deriveThms e
  trace[CancelDenoms] "e simplified = {eSimp.expr}"
  let eSimpNormNum ← Mathlib.Meta.NormNum.deriveSimp (← Simp.mkContext) false eSimp.expr
  trace[CancelDenoms] "e norm_num'd = {eSimpNormNum.expr}"
  let (n, t) := findCancelFactor eSimpNormNum.expr
  let ⟨u, tp, e⟩ ← inferTypeQ' eSimpNormNum.expr
  let stp : Q(Field $tp) ← synthInstanceQ q(Field $tp)
  try
    have n' := (← mkOfNat tp q(inferInstance) <| mkRawNatLit <| n).1
    let r ← mkProdPrf tp stp n n' t e
    trace[CancelDenoms] "pf : {← inferType r.pf}"
    let pf' ←
      match eSimp.proof?, eSimpNormNum.proof? with
      | some pfSimp, some pfSimp' => mkAppM ``derive_trans₂ #[pfSimp, pfSimp', r.pf]
      | some pfSimp, none | none, some pfSimp => mkAppM ``derive_trans #[pfSimp, r.pf]
      | none, none => pure r.pf
    return (n, pf')
  catch E => do
    throwError "CancelDenoms.derive failed to normalize {e}.\n{E.toMessageData}"


/--
`findCompLemma e` arranges `e` in the form `lhs R rhs`, where `R ∈ {<, ≤, =, ≠}`, and returns
`lhs`, `rhs`, the `cancel_factors` lemma corresponding to `R`, and a boolean indicating whether
`R` involves the order (i.e. `<` and `≤`) or not (i.e. `=` and `≠`).
In the case of `LT`, `LE`, `GE`, and `GT` an order on the type is needed, in the last case
it is not, the final component of the return value tracks this.
-/
def findCompLemma (e : Expr) : MetaM (Option (Expr × Expr × Name × Bool)) := do
  match (← whnfR e).getAppFnArgs with
  | (``LT.lt, #[_, _, a, b]) => return (a, b, ``cancel_factors_lt, true)
  | (``LE.le, #[_, _, a, b]) => return (a, b, ``cancel_factors_le, true)
  | (``Eq, #[_, a, b]) => return (a, b, ``cancel_factors_eq, false)
  -- `a ≠ b` reduces to `¬ a = b` under `whnf`
  | (``Not, #[p]) => match (← whnfR p).getAppFnArgs with
    | (``Eq, #[_, a, b]) => return (a, b, ``cancel_factors_ne, false)
    | _ => return none
  | (``GE.ge, #[_, _, a, b]) => return (b, a, ``cancel_factors_le, true)
  | (``GT.gt, #[_, _, a, b]) => return (b, a, ``cancel_factors_lt, true)
  | _ => return none


/--
`cancelDenominatorsInType h` assumes that `h` is of the form `lhs R rhs`,
where `R ∈ {<, ≤, =, ≠, ≥, >}`.
It produces an Expression `h'` of the form `lhs' R rhs'` and a proof that `h = h'`.
Numeric denominators have been canceled in `lhs'` and `rhs'`.
-/
def cancelDenominatorsInType (h : Expr) : MetaM (Expr × Expr) := do
  let some (lhs, rhs, lem, ord) ← findCompLemma h | throwError m!"cannot kill factors"
  let (al, lhs_p) ← derive lhs
  let ⟨u, α, _⟩ ← inferTypeQ' lhs
  let amwo ← synthInstanceQ q(AddMonoidWithOne $α)
  let (ar, rhs_p) ← derive rhs
  let gcd := al.gcd ar
  have al := (← mkOfNat α amwo <| mkRawNatLit al).1
  have ar := (← mkOfNat α amwo <| mkRawNatLit ar).1
  have gcd := (← mkOfNat α amwo <| mkRawNatLit gcd).1
  let (al_cond, ar_cond, gcd_cond) ← if ord then do
      let _ ← synthInstanceQ q(LinearOrderedField $α)
      let al_pos : Q(Prop) := q(0 < $al)
      let ar_pos : Q(Prop) := q(0 < $ar)
      let gcd_pos : Q(Prop) := q(0 < $gcd)
      pure (al_pos, ar_pos, gcd_pos)
    else do
      let _ ← synthInstanceQ q(Field $α)
      let al_ne : Q(Prop) := q($al ≠ 0)
      let ar_ne : Q(Prop) := q($ar ≠ 0)
      let gcd_ne : Q(Prop) := q($gcd ≠ 0)
      pure (al_ne, ar_ne, gcd_ne)
  let al_cond ← synthesizeUsingNormNum al_cond
  let ar_cond ← synthesizeUsingNormNum ar_cond
  let gcd_cond ← synthesizeUsingNormNum gcd_cond
  let pf ← mkAppM lem #[lhs_p, rhs_p, al_cond, ar_cond, gcd_cond]
  let pf_tp ← inferType pf
  return ((← findCompLemma pf_tp).elim default (Prod.fst ∘ Prod.snd), pf)


/--
`cancel_denoms` attempts to remove numerals from the denominators of fractions.
It works on propositions that are field-valued inequalities.

```lean
variable [LinearOrderedField α] (a b c : α)

example (h : a / 5 + b / 4 < c) : 4*a + 5*b < 20*c := by
  cancel_denoms at h
  exact h

example (h : a > 0) : a / 5 > 0 := by
  cancel_denoms
  exact h
```
-/
syntax (name := cancelDenoms) "cancel_denoms" (location)? : tactic


def cancelDenominatorsAt (fvar : FVarId) : TacticM Unit := do
  let t ← instantiateMVars (← fvar.getDecl).type
  let (new, eqPrf) ← CancelDenoms.cancelDenominatorsInType t
  liftMetaTactic' fun g => do
    let res ← g.replaceLocalDecl fvar new eqPrf
    return res.mvarId


def cancelDenominatorsTarget : TacticM Unit := do
  let (new, eqPrf) ← CancelDenoms.cancelDenominatorsInType (← getMainTarget)
  liftMetaTactic' fun g => g.replaceTargetEq new eqPrf


def cancelDenominators (loc : Location) : TacticM Unit := do
  withLocation loc cancelDenominatorsAt cancelDenominatorsTarget
    (fun _ ↦ throwError "Failed to cancel any denominators")


elab "cancel_denoms" loc?:(location)? : tactic => do
  cancelDenominators (expandOptLocation (Lean.mkOptionalNode loc?))
  Lean.Elab.Tactic.evalTactic (← `(tactic| try norm_num [← mul_assoc] $[$loc?]?))

