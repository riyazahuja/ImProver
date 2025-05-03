theorem isRat_ofScientific_of_true [DivisionRing α] :
    {m e : ℕ} → {n : ℤ} → {d : ℕ} →
    IsRat (mkRat m (10 ^ e) : α) n d → IsRat (OfScientific.ofScientific m true e : α) n d
  | _, _, _, _, ⟨_, eq⟩ => ⟨‹_›, by
    /-
      α : Type u_1
      inst✝ : DivisionRing α
      x✝¹ x✝ : Nat
      num✝ : Int
      denom✝ : Nat
      inv✝ : Invertible ↑denom✝
      eq : Eq (↑(mkRat (↑x✝¹) (HPow.hPow 10 x✝))) (HMul.hMul (↑num✝) (Invertible.inv …
      ⊢ Eq (OfScientific.ofScientific x✝¹ Bool.true x✝) (HMul.hMul (↑num✝) (Invertib …
    -/
    rwa [← Rat.cast_ofScientific, ← Rat.ofScientific_eq_ofScientific, Rat.ofScientific_true_def]⟩
    /-
      🎉 no goals
    -/

-- see note [norm_num lemma function equality]

theorem isNat_ofScientific_of_false [DivisionRing α] : {m e nm ne n : ℕ} →
    IsNat m nm → IsNat e ne → n = Nat.mul nm ((10 : ℕ) ^ ne) →
    IsNat (OfScientific.ofScientific m false e : α) n
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, h => ⟨by
    /-
      α : Type u_1
      inst✝ : DivisionRing α
      n✝¹ n✝ x✝ : Nat
      h : Eq x✝ (n✝¹.mul (HPow.hPow 10 n✝))
      ⊢ Eq (OfScientific.ofScientific (↑n✝¹) Bool.false ↑n✝) ↑x✝
    -/
    rw [← Rat.cast_ofScientific, ← Rat.ofScientific_eq_ofScientific]
    simp only [Nat.cast_id, Rat.ofScientific_false_def, Nat.cast_mul, Nat.cast_pow,
      Nat.cast_ofNat, h, Nat.mul_eq]
    /-
      α : Type u_1
      inst✝ : DivisionRing α
      n✝¹ n✝ x✝ : Nat
      h : Eq x✝ (n✝¹.mul (HPow.hPow 10 n✝))
      ⊢ Eq (↑(HMul.hMul (↑(OfNat.ofNat n✝¹)) (HPow.hPow 10 (OfNat.ofNat n✝)))) (HMul …
    -/
    norm_cast⟩
    /-
      🎉 no goals
    -/


/-- The `norm_num` extension which identifies expressions in scientific notation, normalizing them
to rat casts if the scientific notation is inherited from the one for rationals. -/
@[norm_num OfScientific.ofScientific _ _ _] def evalOfScientific :
    NormNumExt where eval {u α} e := do
  let .app (.app (.app f (m : Q(ℕ))) (b : Q(Bool))) (exp : Q(ℕ)) ← whnfR e | failure
  let dα ← inferDivisionRing α
  guard <|← withNewMCtxDepth <| isDefEq f q(OfScientific.ofScientific (α := $α))
  assumeInstancesCommute
  haveI' : $e =Q OfScientific.ofScientific $m $b $exp := ⟨⟩
  match b with
  | ~q(true) =>
    let rme ← derive (q(mkRat $m (10 ^ $exp)) : Q($α))
    let some ⟨q, n, d, p⟩ := rme.toRat' dα | failure
    return .isRat' dα q n d q(isRat_ofScientific_of_true $p)
  | ~q(false) =>
    let ⟨nm, pm⟩ ← deriveNat m q(AddCommMonoidWithOne.toAddMonoidWithOne)
    let ⟨ne, pe⟩ ← deriveNat exp q(AddCommMonoidWithOne.toAddMonoidWithOne)
    have pm : Q(IsNat $m $nm) := pm
    have pe : Q(IsNat $exp $ne) := pe
    let m' := nm.natLit!
    let exp' := ne.natLit!
    let n' := Nat.mul m' (Nat.pow (10 : ℕ) exp')
    have n : Q(ℕ) := mkRawNatLit n'
    haveI : $n =Q Nat.mul $nm ((10 : ℕ) ^ $ne) := ⟨⟩
    return .isNat _ n q(isNat_ofScientific_of_false $pm $pe (.refl $n))


