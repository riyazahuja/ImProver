local notation "‖" x "‖" => Fintype.card x


/-- A finite field has prime power cardinality. -/
theorem Fintype.isPrimePow_card_of_field {α} [Fintype α] [Field α] : IsPrimePow ‖α‖ := by
  -- TODO: `Algebra` version of `CharP.exists`, of type `∀ p, Algebra (ZMod p) α`
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    ⊢ IsPrimePow (Fintype.card α)
  -/
  cases' CharP.exists α with p _
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    p : Nat
    h✝ : CharP α p
    ⊢ IsPrimePow (Fintype.card α)
  -/
  haveI hp := Fact.mk (CharP.char_is_prime α p)
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    p : Nat
    h✝ : CharP α p
    hp : Fact (Nat.Prime p)
    ⊢ IsPrimePow (Fintype.card α)
  -/
  letI : Algebra (ZMod p) α := ZMod.algebra _ _
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    p : Nat
    h✝ : CharP α p
    hp : Fact (Nat.Prime p)
    this : Algebra (ZMod p) α := ZMod.algebra α p
    ⊢ IsPrimePow (Fintype.card α)
  -/
  let b := IsNoetherian.finsetBasis (ZMod p) α
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    p : Nat
    h✝ : CharP α p
    hp : Fact (Nat.Prime p)
    this : Algebra (ZMod p) α := ZMod.algebra α p
    b : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex (ZMo …
    ⊢ IsPrimePow (Fintype.card α)
  -/
  rw [Module.card_fintype b, ZMod.card, isPrimePow_pow_iff]
    /-
      case intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : Field α
      p : Nat
      h✝ : CharP α p
      hp : Fact (Nat.Prime p)
      this : Algebra (ZMod p) α := ZMod.algebra α p
      b : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex (ZMo …
      ⊢ IsPrimePow p
    -/
  · exact hp.1.isPrimePow
    /-
      🎉 no goals
    -/
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    p : Nat
    h✝ : CharP α p
    hp : Fact (Nat.Prime p)
    this : Algebra (ZMod p) α := ZMod.algebra α p
    b : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex (ZMo …
    ⊢ Ne (Fintype.card (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisI …
  -/
  rw [← Module.finrank_eq_card_basis b]
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Field α
    p : Nat
    h✝ : CharP α p
    hp : Fact (Nat.Prime p)
    this : Algebra (ZMod p) α := ZMod.algebra α p
    b : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex (ZMo …
    ⊢ Ne (Module.finrank (ZMod p) α) 0
  -/
  exact Module.finrank_pos.ne'
  /-
    🎉 no goals
  -/


/-- A `Fintype` can be given a field structure iff its cardinality is a prime power. -/
theorem Fintype.nonempty_field_iff {α} [Fintype α] : Nonempty (Field α) ↔ IsPrimePow ‖α‖ := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    ⊢ Iff (Nonempty (Field α)) (IsPrimePow (Fintype.card α))
  -/
  refine ⟨fun ⟨h⟩ => Fintype.isPrimePow_card_of_field, ?_⟩
  /-
    α : Type u_1
    inst✝ : Fintype α
    ⊢ IsPrimePow (Fintype.card α) → Nonempty (Field α)
  -/
  rintro ⟨p, n, hp, hn, hα⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : Fintype α
    p n : Nat
    hp : Prime p
    hn : LT.lt 0 n
    hα : Eq (HPow.hPow p n) (Fintype.card α)
    ⊢ Nonempty (Field α)
  -/
  haveI := Fact.mk hp.nat_prime
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : Fintype α
    p n : Nat
    hp : Prime p
    hn : LT.lt 0 n
    hα : Eq (HPow.hPow p n) (Fintype.card α)
    this : Fact (Nat.Prime p)
    ⊢ Nonempty (Field α)
  -/
  haveI : Fintype (GaloisField p n) := Fintype.ofFinite (GaloisField p n)
  exact ⟨(Fintype.equivOfCardEq
    (((Fintype.card_eq_nat_card).trans (GaloisField.card p n hn.ne')).trans hα)).symm.field⟩


theorem Fintype.not_isField_of_card_not_prime_pow {α} [Fintype α] [Ring α] :
    ¬IsPrimePow ‖α‖ → ¬IsField α :=
  mt fun h => Fintype.nonempty_field_iff.mp ⟨h.toField⟩


/-- Any infinite type can be endowed a field structure. -/
theorem Infinite.nonempty_field {α : Type u} [Infinite α] : Nonempty (Field α) := by
  suffices #α = #(FractionRing (MvPolynomial α <| ULift.{u} ℚ)) from
    (Cardinal.eq.1 this).map (·.field)
  /-
    α : Type u
    inst✝ : Infinite α
    ⊢ Eq (Cardinal.mk α) (Cardinal.mk (FractionRing (MvPolynomial α (ULift.{u, 0}  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- There is a field structure on type if and only if its cardinality is a prime power. -/
theorem Field.nonempty_iff {α : Type u} : Nonempty (Field α) ↔ IsPrimePow #α := by
  /-
    α : Type u
    ⊢ Iff (Nonempty (Field α)) (IsPrimePow (Cardinal.mk α))
  -/
  rw [Cardinal.isPrimePow_iff]
  /-
    α : Type u
    ⊢ Iff (Nonempty (Field α)) (Or (LE.le Cardinal.aleph0 (Cardinal.mk α)) (Exists …
  -/
  cases' fintypeOrInfinite α with h h
  · simpa only [Cardinal.mk_fintype, Nat.cast_inj, exists_eq_left',
      (Cardinal.nat_lt_aleph0 _).not_le, false_or] using Fintype.nonempty_field_iff
    /-
      case inr
      α : Type u
      h : Infinite α
      ⊢ Iff (Nonempty (Field α)) (Or (LE.le Cardinal.aleph0 (Cardinal.mk α)) (Exists …
    -/
  · simpa only [← Cardinal.infinite_iff, h, true_or, iff_true] using Infinite.nonempty_field
    /-
      🎉 no goals
    -/

