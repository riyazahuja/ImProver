/-- `ℙ k V` is equivalent to the quotient of the non-zero elements of `V` by `kˣ`. -/
def equivQuotientOrbitRel : ℙ k V ≃ Quotient (MulAction.orbitRel kˣ { v : V // v ≠ 0 }) :=
  Quotient.congr (Equiv.refl _) (fun x y ↦ (Units.orbitRel_nonZero_iff k V x y).symm)


/-- The non-zero elements of `V` are equivalent to the product of `ℙ k V` with the units of `k`. -/
noncomputable def nonZeroEquivProjectivizationProdUnits : { v : V // v ≠ 0 } ≃ ℙ k V × kˣ :=
  let e := MulAction.selfEquivOrbitsQuotientProd <| fun b ↦ by
    rw [(Units.nonZeroSubMul k V).stabilizer_of_subMul,
      Module.stabilizer_units_eq_bot_of_ne_zero k b.property]
  e.trans (Equiv.prodCongrLeft (fun _ ↦ (equivQuotientOrbitRel k V).symm))


instance isEmpty_of_subsingleton [Subsingleton V] : IsEmpty (ℙ k V) := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Subsingleton V
    ⊢ IsEmpty (Projectivization k V)
  -/
  have : IsEmpty { v : V // v ≠ 0 } := ⟨fun v ↦ v.2 (Subsingleton.elim v.1 0)⟩
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Subsingleton V
    this : IsEmpty (Subtype fun v => Ne v 0)
    ⊢ IsEmpty (Projectivization k V)
  -/
  simpa using (nonZeroEquivProjectivizationProdUnits k V).symm.isEmpty
  /-
    🎉 no goals
  -/


/-- If `V` is a finite `k`-module and `k` is finite, `ℙ k V` is finite. -/
instance finite_of_finite [Finite V] : Finite (ℙ k V) :=
  have : Finite (ℙ k V × kˣ) := Finite.of_equiv _ (nonZeroEquivProjectivizationProdUnits k V)
  Finite.prod_left kˣ


lemma finite_iff_of_finite [Finite k] : Finite (ℙ k V) ↔ Finite V := by
  classical
  refine ⟨fun h ↦ ?_, fun h ↦ inferInstance⟩
  let e := nonZeroEquivProjectivizationProdUnits k V
  have : Finite { v : V // v ≠ 0 } := Finite.of_equiv _ e.symm
  let eq : { v : V // v ≠ 0 } ⊕ Unit ≃ V :=
    ⟨(Sum.elim Subtype.val (fun _ ↦ 0)), fun v ↦ if h : v = 0 then Sum.inr () else Sum.inl ⟨v, h⟩,
      by intro x; aesop, by intro x; aesop⟩
  exact Finite.of_equiv _ eq


/-- Fraction free cardinality formula for the points of `ℙ k V` if `k` and `V` are finite
(for silly reasons the formula also holds when `k` and `V` are infinite).
See `Projectivization.card'` and `Projectivization.card''` for other spellings of the formula. -/
lemma card : Nat.card V - 1 = Nat.card (ℙ k V) * (Nat.card k - 1) := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
  -/
  nontriviality V
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    a✝ : Nontrivial V
    ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
  -/
  wlog h : Finite k
    /-
      case inr
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      this : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddCo …
      h : Not (Finite k)
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
  · simp only [not_finite_iff_infinite] at h
    /-
      case inr
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      this : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddCo …
      h : Infinite k
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
    have : Infinite V := Module.Free.infinite k V
    /-
      case inr
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      this✝ : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddC …
      h : Infinite k
      this : Infinite V
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁵ : DivisionRing k✝
    inst✝⁴ : AddCommGroup V✝
    inst✝³ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝² : DivisionRing k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    a✝ : Nontrivial V
    h : Finite k
    ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
  -/
  wlog h : Finite V
    /-
      case inr
      k✝ : Type u_1
      V✝ : Type u_2
      inst✝⁵ : DivisionRing k✝
      inst✝⁴ : AddCommGroup V✝
      inst✝³ : Module k✝ V✝
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      h✝ : Finite k
      this : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddCo …
      h : Not (Finite V)
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
  · simp only [not_finite_iff_infinite] at h
    /-
      case inr
      k✝ : Type u_1
      V✝ : Type u_2
      inst✝⁵ : DivisionRing k✝
      inst✝⁴ : AddCommGroup V✝
      inst✝³ : Module k✝ V✝
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      h✝ : Finite k
      this : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddCo …
      h : Infinite V
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
    have := not_iff_not.mpr (finite_iff_of_finite k V)
    /-
      case inr
      k✝ : Type u_1
      V✝ : Type u_2
      inst✝⁵ : DivisionRing k✝
      inst✝⁴ : AddCommGroup V✝
      inst✝³ : Module k✝ V✝
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      h✝ : Finite k
      this✝ : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddC …
      h : Infinite V
      this : Iff (Not (Finite (Projectivization k V))) (Not (Finite V))
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
    simp only [not_finite_iff_infinite] at this
    /-
      case inr
      k✝ : Type u_1
      V✝ : Type u_2
      inst✝⁵ : DivisionRing k✝
      inst✝⁴ : AddCommGroup V✝
      inst✝³ : Module k✝ V✝
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      h✝ : Finite k
      this✝ : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : AddC …
      h : Infinite V
      this : Iff (Infinite (Projectivization k V)) (Infinite V)
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
    have : Infinite (ℙ k V) := by rwa [this]
    /-
      case inr
      k✝ : Type u_1
      V✝ : Type u_2
      inst✝⁵ : DivisionRing k✝
      inst✝⁴ : AddCommGroup V✝
      inst✝³ : Module k✝ V✝
      k : Type u_1
      V : Type u_2
      inst✝² : DivisionRing k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      a✝ : Nontrivial V
      h✝ : Finite k
      this✝¹ : ∀ (k : Type u_1) (V : Type u_2) [inst : DivisionRing k] [inst_1 : Add …
      h : Infinite V
      this✝ : Iff (Infinite (Projectivization k V)) (Infinite V)
      this : Infinite (Projectivization k V)
      ⊢ Eq (HSub.hSub (Nat.card V) 1) (HMul.hMul (Nat.card (Projectivization k V)) ( …
    -/
    simp
    /-
      🎉 no goals
    -/
  classical
  haveI : Fintype V := Fintype.ofFinite V
  haveI : Fintype (ℙ k V) := Fintype.ofFinite (ℙ k V)
  haveI : Fintype k := Fintype.ofFinite k
  have hV : Fintype.card { v : V // v ≠ 0 } = Fintype.card V - 1 := by simp
  simp_rw [← Fintype.card_eq_nat_card, ← Fintype.card_units (α := k), ← hV]
  rw [Fintype.card_congr (nonZeroEquivProjectivizationProdUnits k V), Fintype.card_prod]


/-- Cardinality formula for the points of `ℙ k V` if `k` and `V` are finite with less
natural subtraction. -/
lemma card' [Finite V] : Nat.card V = Nat.card (ℙ k V) * (Nat.card k - 1) + 1 := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite V
    ⊢ Eq (Nat.card V) (HAdd.hAdd (HMul.hMul (Nat.card (Projectivization k V)) (HSu …
  -/
  rw [← card k V]
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite V
    ⊢ Eq (Nat.card V) (HAdd.hAdd (HSub.hSub (Nat.card V) 1) 1)
  -/
  have : Nat.card V > 0 := Nat.card_pos
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite V
    this : GT.gt (Nat.card V) 0
    ⊢ Eq (Nat.card V) (HAdd.hAdd (HSub.hSub (Nat.card V) 1) 1)
  -/
  omega
  /-
    🎉 no goals
  -/


/-- Cardinality formula for the points of `ℙ k V` if `k` and `V` are finite expressed
as a fraction. -/
lemma card'' [Finite k] : Nat.card (ℙ k V) = (Nat.card V - 1) / (Nat.card k - 1) := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    ⊢ Eq (Nat.card (Projectivization k V)) (HDiv.hDiv (HSub.hSub (Nat.card V) 1) ( …
  -/
  haveI : Fintype k := Fintype.ofFinite k
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    this : Fintype k
    ⊢ Eq (Nat.card (Projectivization k V)) (HDiv.hDiv (HSub.hSub (Nat.card V) 1) ( …
  -/
  rw [card k]
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    this : Fintype k
    ⊢ Eq (Nat.card (Projectivization k V)) (HDiv.hDiv (HMul.hMul (Nat.card (Projec …
  -/
  have : 1 < Nat.card k := Finite.one_lt_card
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    this✝ : Fintype k
    this : LT.lt 1 (Nat.card k)
    ⊢ Eq (Nat.card (Projectivization k V)) (HDiv.hDiv (HMul.hMul (Nat.card (Projec …
  -/
  have h : 0 ≠ (Nat.card k - 1) := by omega
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    this✝ : Fintype k
    this : LT.lt 1 (Nat.card k)
    h : Ne 0 (HSub.hSub (Nat.card k) 1)
    ⊢ Eq (Nat.card (Projectivization k V)) (HDiv.hDiv (HMul.hMul (Nat.card (Projec …
  -/
  exact Nat.eq_div_of_mul_eq_left (Ne.symm h) rfl
  /-
    🎉 no goals
  -/


lemma card_of_finrank [Finite k] {n : ℕ} (h : Module.finrank k V = n) :
    Nat.card (ℙ k V) = ∑ i ∈ Finset.range n, Nat.card k ^ i := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    ⊢ Eq (Nat.card (Projectivization k V)) ((Finset.range n).sum fun i => HPow.hPo …
  -/
  wlog hf : Finite V
    /-
      case inr
      k : Type u_1
      V : Type u_2
      inst✝³ : Field k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : Finite k
      n : Nat
      h : Eq (Module.finrank k V) n
      this : ∀ (k : Type u_1) (V : Type u_2) [inst : Field k] [inst_1 : AddCommGroup …
      hf : Not (Finite V)
      ⊢ Eq (Nat.card (Projectivization k V)) ((Finset.range n).sum fun i => HPow.hPo …
    -/
  · simp only [not_finite_iff_infinite] at hf
    have : Infinite (ℙ k V) := by
      rw [← not_finite_iff_infinite, not_iff_not.mpr (finite_iff_of_finite k V)]
      simpa
    have : n = 0 := by
      rw [← h]
      apply Module.finrank_of_not_finite
      contrapose! hf
      simpa using Module.finite_of_finite k
    /-
      case inr
      k : Type u_1
      V : Type u_2
      inst✝³ : Field k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : Finite k
      n : Nat
      h : Eq (Module.finrank k V) n
      this✝¹ : ∀ (k : Type u_1) (V : Type u_2) [inst : Field k] [inst_1 : AddCommGro …
      hf : Infinite V
      this✝ : Infinite (Projectivization k V)
      this : Eq n 0
      ⊢ Eq (Nat.card (Projectivization k V)) ((Finset.range n).sum fun i => HPow.hPo …
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    ⊢ Eq (Nat.card (Projectivization k V)) ((Finset.range n).sum fun i => HPow.hPo …
  -/
  have : 1 < Nat.card k := Finite.one_lt_card
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    ⊢ Eq (Nat.card (Projectivization k V)) ((Finset.range n).sum fun i => HPow.hPo …
  -/
  refine Nat.mul_right_cancel (m := Nat.card k - 1) (by omega) ?_
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    ⊢ Eq (HMul.hMul (Nat.card (Projectivization k V)) (HSub.hSub (Nat.card k) 1))  …
  -/
  let e : V ≃ₗ[k] (Fin n → k) := LinearEquiv.ofFinrankEq _ _ (by simpa)
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    e : LinearEquiv (RingHom.id k) V (Fin n → k) := LinearEquiv.ofFinrankEq V (Fin …
    ⊢ Eq (HMul.hMul (Nat.card (Projectivization k V)) (HSub.hSub (Nat.card k) 1))  …
  -/
  have hc : Nat.card V = Nat.card k ^ n := by simp [Nat.card_congr e.toEquiv, Nat.card_fun]
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    e : LinearEquiv (RingHom.id k) V (Fin n → k) := LinearEquiv.ofFinrankEq V (Fin …
    hc : Eq (Nat.card V) (HPow.hPow (Nat.card k) n)
    ⊢ Eq (HMul.hMul (Nat.card (Projectivization k V)) (HSub.hSub (Nat.card k) 1))  …
  -/
  zify
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    e : LinearEquiv (RingHom.id k) V (Fin n → k) := LinearEquiv.ofFinrankEq V (Fin …
    hc : Eq (Nat.card V) (HPow.hPow (Nat.card k) n)
    ⊢ Eq (HMul.hMul ↑(Nat.card (Projectivization k V)) ↑(HSub.hSub (Nat.card k) 1) …
  -/
  have hn : 1 ≤ Nat.card k := Nat.one_le_of_lt Finite.one_lt_card
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    e : LinearEquiv (RingHom.id k) V (Fin n → k) := LinearEquiv.ofFinrankEq V (Fin …
    hc : Eq (Nat.card V) (HPow.hPow (Nat.card k) n)
    hn : LE.le 1 (Nat.card k)
    ⊢ Eq (HMul.hMul ↑(Nat.card (Projectivization k V)) ↑(HSub.hSub (Nat.card k) 1) …
  -/
  conv_rhs => rw [Int.natCast_sub hn, Int.natCast_one, geom_sum_mul]
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    e : LinearEquiv (RingHom.id k) V (Fin n → k) := LinearEquiv.ofFinrankEq V (Fin …
    hc : Eq (Nat.card V) (HPow.hPow (Nat.card k) n)
    hn : LE.le 1 (Nat.card k)
    ⊢ Eq (HMul.hMul ↑(Nat.card (Projectivization k V)) ↑(HSub.hSub (Nat.card k) 1) …
  -/
  rw [← Int.natCast_mul, ← card k V, hc]
  /-
    k✝ : Type u_1
    V✝ : Type u_2
    inst✝⁶ : Field k✝
    inst✝⁵ : AddCommGroup V✝
    inst✝⁴ : Module k✝ V✝
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    n : Nat
    h : Eq (Module.finrank k V) n
    hf : Finite V
    this : LT.lt 1 (Nat.card k)
    e : LinearEquiv (RingHom.id k) V (Fin n → k) := LinearEquiv.ofFinrankEq V (Fin …
    hc : Eq (Nat.card V) (HPow.hPow (Nat.card k) n)
    hn : LE.le 1 (Nat.card k)
    ⊢ Eq (↑(HSub.hSub (HPow.hPow (Nat.card k) n) 1)) (HSub.hSub (HPow.hPow (↑(Nat. …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma card_of_finrank_two [Finite k] (h : Module.finrank k V = 2) :
    Nat.card (ℙ k V) = Nat.card k + 1 := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝³ : Field k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : Finite k
    h : Eq (Module.finrank k V) 2
    ⊢ Eq (Nat.card (Projectivization k V)) (HAdd.hAdd (Nat.card k) 1)
  -/
  simp [card_of_finrank k V h]
  /-
    🎉 no goals
  -/


