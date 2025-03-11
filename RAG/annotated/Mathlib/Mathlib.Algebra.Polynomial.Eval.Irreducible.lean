/-- A polynomial over an integral domain `R` is irreducible if it is monic and
irreducible after mapping into an integral domain `S`.

A special case of this lemma is that a polynomial over `ℤ` is irreducible if
it is monic and irreducible over `ℤ/pℤ` for some prime `p`.
-/
lemma Monic.irreducible_of_irreducible_map (f : R[X]) (h_mon : Monic f)
    (h_irr : Irreducible (f.map φ)) : Irreducible f := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    h_mon : f.Monic
    h_irr : Irreducible (Polynomial.map φ f)
    ⊢ Irreducible f
  -/
  refine ⟨h_irr.not_unit ∘ IsUnit.map (mapRingHom φ), fun a b h => ?_⟩
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    h_mon : f.Monic
    h_irr : Irreducible (Polynomial.map φ f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  dsimp [Monic] at h_mon
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    h_mon : Eq f.leadingCoeff 1
    h_irr : Irreducible (Polynomial.map φ f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  have q := (leadingCoeff_mul a b).symm
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    h_mon : Eq f.leadingCoeff 1
    h_irr : Irreducible (Polynomial.map φ f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    q : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) (HMul.hMul a b).leadingCoeff
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rw [← h, h_mon] at q
  refine (h_irr.isUnit_or_isUnit <|
    (congr_arg (Polynomial.map φ) h).trans (Polynomial.map_mul φ)).imp ?_ ?_ <;>
      /-
        case refine_1
        R : Type u
        S : Type v
        inst✝³ : CommRing R
        inst✝² : IsDomain R
        inst✝¹ : CommRing S
        inst✝ : IsDomain S
        φ : RingHom R S
        f : Polynomial R
        h_mon : Eq f.leadingCoeff 1
        h_irr : Irreducible (Polynomial.map φ f)
        a b : Polynomial R
        h : Eq f (HMul.hMul a b)
        q : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
        ⊢ IsUnit (Polynomial.map φ a) → IsUnit a
      -/
      apply isUnit_of_isUnit_leadingCoeff_of_isUnit_map <;>
    /-
      case refine_1.hf
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      h_mon : Eq f.leadingCoeff 1
      h_irr : Irreducible (Polynomial.map φ f)
      a b : Polynomial R
      h : Eq f (HMul.hMul a b)
      q : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      ⊢ IsUnit a.leadingCoeff
    -/
    apply isUnit_of_mul_eq_one
    /-
      case refine_1.hf.h
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      h_mon : Eq f.leadingCoeff 1
      h_irr : Irreducible (Polynomial.map φ f)
      a b : Polynomial R
      h : Eq f (HMul.hMul a b)
      q : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      ⊢ Eq (HMul.hMul a.leadingCoeff ?refine_1.hf.b) 1
    -/
  · exact q
    /-
      🎉 no goals
    -/
    /-
      case refine_2.hf.h
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      h_mon : Eq f.leadingCoeff 1
      h_irr : Irreducible (Polynomial.map φ f)
      a b : Polynomial R
      h : Eq f (HMul.hMul a b)
      q : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      ⊢ Eq (HMul.hMul b.leadingCoeff ?refine_2.hf.b) 1
    -/
  · rw [mul_comm]
    /-
      case refine_2.hf.h
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      h_mon : Eq f.leadingCoeff 1
      h_irr : Irreducible (Polynomial.map φ f)
      a b : Polynomial R
      h : Eq f (HMul.hMul a b)
      q : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      ⊢ Eq (HMul.hMul ?refine_2.hf.h.b b.leadingCoeff) 1
    -/
    exact q
    /-
      🎉 no goals
    -/


